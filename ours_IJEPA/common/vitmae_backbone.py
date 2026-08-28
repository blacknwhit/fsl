from __future__ import annotations

import math
from typing import Any, Dict, Iterable

import torch
import torch.nn as nn

from .models import vision_transformer as ijepa_vit


_MODEL_ALIASES: Dict[str, Dict[str, Any]] = {
    "ijepa_vit_huge_patch16": {
        "builder": "vit_huge",
        "patch_size": 16,
        "embed_dim": 1280,
        "depth": 32,
        "num_heads": 16,
        "base_image_size": 224,
    },
    "vit_huge_patch16": {"alias_for": "ijepa_vit_huge_patch16"},
    "ijepa_vith16": {"alias_for": "ijepa_vit_huge_patch16"},
    "vit_h16": {"alias_for": "ijepa_vit_huge_patch16"},
}


def _resolve_model_name(model_name: str) -> Dict[str, Any]:
    key = str(model_name).strip().lower()
    if key not in _MODEL_ALIASES:
        raise ValueError(f"Unsupported IJEPA backbone: {model_name!r}")
    spec = dict(_MODEL_ALIASES[key])
    while "alias_for" in spec:
        spec = dict(_MODEL_ALIASES[str(spec["alias_for"])])
    return spec


def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _iter_candidate_states(obj: object) -> Iterable[dict[str, torch.Tensor]]:
    if isinstance(obj, dict):
        if obj and all(isinstance(k, str) for k in obj.keys()) and any(torch.is_tensor(v) for v in obj.values()):
            yield obj
        for key in ("state_dict", "model", "module", "backbone", "encoder", "target_encoder"):
            child = obj.get(key)
            if isinstance(child, dict):
                yield from _iter_candidate_states(child)


def _normalize_backbone_state_dict(state: dict[str, torch.Tensor], target_keys: set[str]) -> dict[str, torch.Tensor]:
    prefixes = (
        "",
        "module.",
        "backbone.",
        "module.backbone.",
        "shared.",
        "module.shared.",
        "shared.backbone.",
        "module.shared.backbone.",
        "encoder.",
        "module.encoder.",
    )
    normalized: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            continue
        for prefix in prefixes:
            if prefix and not key.startswith(prefix):
                continue
            candidate = key[len(prefix) :] if prefix else key
            if candidate in target_keys:
                normalized[candidate] = value
                break
    return normalized


def load_backbone_checkpoint(backbone: nn.Module, checkpoint_path: str) -> None:
    obj = _torch_load_cpu(checkpoint_path)
    target_state = backbone.state_dict()
    target_keys = set(target_state.keys())
    for candidate in _iter_candidate_states(obj):
        normalized = _normalize_backbone_state_dict(candidate, target_keys)
        if normalized:
            compatible: dict[str, torch.Tensor] = {}
            for key, value in normalized.items():
                target = target_state.get(key)
                if target is None:
                    continue
                if tuple(value.shape) == tuple(target.shape):
                    compatible[key] = value
                    continue
                if key == "pos_embed" and value.ndim == 3 and target.ndim == 3:
                    try:
                        resized = _interpolate_pos_embed(value.to(dtype=target.dtype), int(target.shape[1]))
                        if tuple(resized.shape) == tuple(target.shape):
                            compatible[key] = resized
                    except Exception:
                        pass
            backbone.load_state_dict(compatible, strict=False)
            return
    raise ValueError(f"Could not find IJEPA encoder weights in checkpoint: {checkpoint_path}")


def build_vitmae_model(model_name: str, *, image_size: int, pretrained: bool) -> nn.Module:
    spec = _resolve_model_name(model_name)
    builder = getattr(ijepa_vit, str(spec["builder"]))
    base_image_size = int(spec["base_image_size"])
    _ = pretrained  # kept for API compatibility with existing call sites
    return builder(patch_size=int(spec["patch_size"]), img_size=[base_image_size])


def _interpolate_pos_embed(pos_embed: torch.Tensor, num_patches: int) -> torch.Tensor:
    if pos_embed.ndim != 3:
        raise ValueError(f"Unexpected pos_embed shape: {tuple(pos_embed.shape)}")
    if pos_embed.shape[1] == num_patches:
        return pos_embed

    old_tokens = int(pos_embed.shape[1])
    dim = int(pos_embed.shape[2])
    old_size = int(round(math.sqrt(old_tokens)))
    new_size = int(round(math.sqrt(num_patches)))
    if old_size * old_size != old_tokens or new_size * new_size != num_patches:
        raise ValueError(
            f"Cannot interpolate positional embeddings from {old_tokens} to {num_patches} patches"
        )
    pos = pos_embed.reshape(1, old_size, old_size, dim).permute(0, 3, 1, 2)
    pos = nn.functional.interpolate(pos, size=(new_size, new_size), mode="bicubic", align_corners=False)
    return pos.permute(0, 2, 3, 1).reshape(1, num_patches, dim)


def build_vitmae_embeddings(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    tokens = backbone.patch_embed(x)
    pos_embed = _interpolate_pos_embed(backbone.pos_embed, int(tokens.shape[1]))
    return tokens + pos_embed.to(device=tokens.device, dtype=tokens.dtype)


def extract_vitmae_patch_tokens(backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    hidden = build_vitmae_embeddings(backbone, x)
    for block in backbone.blocks:
        hidden = block(hidden)
    if getattr(backbone, "norm", None) is not None:
        hidden = backbone.norm(hidden)
    return hidden


class SharedViTMAEBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = "ijepa_vit_huge_patch16",
        image_size: int = 448,
        checkpoint_path: str | None = None,
        *,
        pretrained: bool | None = None,
    ) -> None:
        super().__init__()
        self.model_name = str(model_name)
        self.image_size = int(image_size)
        self.pretrained = bool(checkpoint_path is None) if pretrained is None else bool(pretrained)
        self.backbone = build_vitmae_model(
            self.model_name,
            image_size=self.image_size,
            pretrained=self.pretrained and checkpoint_path is None,
        )
        if checkpoint_path:
            load_backbone_checkpoint(self.backbone, checkpoint_path)

        self.embed_dim = int(self.backbone.embed_dim)
        patch = int(self.backbone.patch_embed.patch_size)
        self.patch_size = (patch, patch)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = bool(trainable)

    def _trainable(self) -> bool:
        return any(p.requires_grad for p in self.backbone.parameters())

    def _forward_backbone(self, x: torch.Tensor, *, trainable: bool) -> torch.Tensor:
        self.backbone.train(self.training and bool(trainable))
        with torch.set_grad_enabled(self.training and bool(trainable)):
            return extract_vitmae_patch_tokens(self.backbone, x)

    def forward_features(self, x: torch.Tensor, *, trainable_override: bool | None = None) -> dict[str, torch.Tensor]:
        trainable = bool(trainable_override) if trainable_override is not None else self._trainable()
        return {"x_norm_patchtokens": self._forward_backbone(x, trainable=trainable)}


SharedIJEPABackbone = SharedViTMAEBackbone

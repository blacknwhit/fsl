from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from transformers import ViTMAEConfig, ViTMAEModel
except ImportError:
    ViTMAEModel = None
    ViTMAEConfig = None

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from .lora import LoRAConfig, count_trainable_params, inject_lora_into_mae_mlp, mark_only_lora_as_trainable

try:
    from .vit_mae_utils import (
        DEFAULT_IMAGE_MEAN,
        DEFAULT_IMAGE_STD,
        DEFAULT_MAE_MODEL_NAME,
        get_vit_image_stats,
        load_mae_backbone_state,
        resolve_backbone_source,
    )
except ImportError:
    from vit_mae_utils import (
        DEFAULT_IMAGE_MEAN,
        DEFAULT_IMAGE_STD,
        DEFAULT_MAE_MODEL_NAME,
        get_vit_image_stats,
        load_mae_backbone_state,
        resolve_backbone_source,
    )


def _extract_hidden_states(block_output: object) -> torch.Tensor:
    if isinstance(block_output, (tuple, list)):
        return block_output[0]
    if torch.is_tensor(block_output):
        return block_output
    raise TypeError(f"Unsupported transformer block output type: {type(block_output)}")


def _run_encoder_block(block: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    try:
        outputs = block(hidden_states, head_mask=None, output_attentions=False)
    except TypeError:
        try:
            outputs = block(hidden_states, output_attentions=False)
        except TypeError:
            outputs = block(hidden_states)
    return _extract_hidden_states(outputs)


def _is_torch26_guard_error(exc: Exception) -> bool:
    msg = str(exc)
    return "upgrade torch to at least v2.6" in msg and "torch.load" in msg


def _load_local_mae_from_bin_dir(source: str) -> ViTMAEModel:
    if ViTMAEModel is None or ViTMAEConfig is None:
        raise ImportError("Cannot import transformers ViTMAEModel/ViTMAEConfig - install transformers first")

    src_dir = Path(source)
    config_path = src_dir / "config.json"
    weight_path = src_dir / "pytorch_model.bin"
    if not config_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(f"Local MAE directory is missing required files: {src_dir}")

    config = ViTMAEConfig.from_pretrained(str(src_dir), local_files_only=True)
    model = ViTMAEModel(config)
    state = load_mae_backbone_state(str(weight_path))
    model_keys = set(model.state_dict().keys())
    filtered_state = {}
    for key, value in state.items():
        mapped_key = key[len("vit.") :] if key.startswith("vit.") else key
        if mapped_key in model_keys:
            filtered_state[mapped_key] = value

    missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
    if unexpected_keys:
        print(f"[mae][warn] unexpected keys in local load: {len(unexpected_keys)}")
    if len(missing_keys) > 0:
        print(f"[mae][warn] missing keys in local load: {len(missing_keys)}")
    return model


def _load_mae_backbone(source: str) -> ViTMAEModel:
    if ViTMAEModel is None:
        raise ImportError("Cannot import transformers.ViTMAEModel - install transformers first")

    try:
        return ViTMAEModel.from_pretrained(source)
    except ValueError as exc:
        src_dir = Path(source)
        if src_dir.is_dir() and _is_torch26_guard_error(exc):
            print(f"[mae] fallback to local config+bin load for torch<2.6: {source}")
            return _load_local_mae_from_bin_dir(source)
        raise


class SharedMAEVisionBackbone(nn.Module):
    def __init__(
        self,
        model_name: str = DEFAULT_MAE_MODEL_NAME,
        image_size: int = 448,
        checkpoint_path: str | None = None,
        *,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        if ViTMAEModel is None:
            raise ImportError("Cannot import transformers.ViTMAEModel - install transformers first")

        source = resolve_backbone_source(model_name, checkpoint_path)
        self.backbone = _load_mae_backbone(source)
        self.use_lora = bool(use_lora)
        self.image_size = int(image_size)
        self.image_mean, self.image_std = get_vit_image_stats(source)
        self.lora_config = (
            LoRAConfig(rank=int(lora_rank), alpha=float(lora_alpha), dropout=float(lora_dropout)) if self.use_lora else None
        )

        if checkpoint_path:
            ckpt_path = Path(checkpoint_path)
            if ckpt_path.exists() and ckpt_path.is_file():
                state = load_mae_backbone_state(str(ckpt_path))
                self.backbone.load_state_dict(state, strict=False)

        self.embed_dim = int(self.backbone.config.hidden_size)
        ps = self.backbone.config.patch_size
        if isinstance(ps, (tuple, list)):
            ph, pw = int(ps[0]), int(ps[1])
        else:
            ph = pw = int(ps)
        self.patch_size = (ph, pw)

        if self.use_lora:
            replaced = inject_lora_into_mae_mlp(self.backbone, cfg=self.lora_config)  # type: ignore[arg-type]
            mark_only_lora_as_trainable(self.backbone)
            trainable = count_trainable_params(self.backbone)
            if trainable == 0:
                raise RuntimeError("LoRA enabled but no trainable params found after injection")
            expected = int(self.backbone.config.num_hidden_layers) * 2
            if replaced != expected:
                raise RuntimeError(f"Expected {expected} MAE MLP Linear layers to be replaced, got {replaced}")
            print(f"[lora] enabled: replaced_linear={replaced}, trainable_params={trainable}")

    def _encoder_layers(self):
        encoder = self.backbone.encoder
        layers = getattr(encoder, "layer", None)
        if layers is None:
            layers = getattr(encoder, "layers", None)
        if layers is None:
            raise AttributeError("Cannot find encoder layers on ViTMAEModel")
        return layers

    def _final_norm(self) -> nn.Module | None:
        norm = getattr(self.backbone, "layernorm", None)
        if norm is None:
            norm = getattr(self.backbone, "norm", None)
        return norm

    def _embed_no_mask(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.backbone.embeddings
        patch_embeddings = getattr(embeddings, "patch_embeddings", None)
        if patch_embeddings is None:
            patch_embeddings = getattr(embeddings, "patch_embed", None)
        if patch_embeddings is None:
            raise AttributeError("ViTMAE embeddings missing patch_embeddings")

        try:
            patch_tokens = patch_embeddings(x, interpolate_pos_encoding=True)
        except TypeError:
            patch_tokens = patch_embeddings(x)

        batch_size = patch_tokens.shape[0]
        cls_token = getattr(embeddings, "cls_token", None)
        if cls_token is None:
            raise AttributeError("ViTMAE embeddings missing cls_token")
        cls_tokens = cls_token.expand(batch_size, -1, -1)
        hidden_states = torch.cat((cls_tokens, patch_tokens), dim=1)

        pos_embeddings = None
        interpolate_pos_encoding = getattr(embeddings, "interpolate_pos_encoding", None)
        if callable(interpolate_pos_encoding):
            try:
                pos_embeddings = interpolate_pos_encoding(hidden_states, x.shape[-2], x.shape[-1])
            except TypeError:
                pos_embeddings = interpolate_pos_encoding(hidden_states)
        if pos_embeddings is None:
            pos_embeddings = getattr(embeddings, "position_embeddings", None)
        if pos_embeddings is None:
            pos_embeddings = getattr(embeddings, "pos_embed", None)
        if pos_embeddings is not None:
            hidden_states = hidden_states + pos_embeddings

        dropout = getattr(embeddings, "dropout", None)
        if dropout is None:
            dropout = getattr(embeddings, "pos_drop", None)
        if dropout is not None:
            hidden_states = dropout(hidden_states)
        return hidden_states

    def _forward_features_no_mask(self, x: torch.Tensor) -> dict:
        hidden_states = self._embed_no_mask(x)
        for block in self._encoder_layers():
            hidden_states = _run_encoder_block(block, hidden_states)
        norm = self._final_norm()
        if norm is not None:
            hidden_states = norm(hidden_states)
        return {"x_norm_patchtokens": hidden_states[:, 1:, :]}

    def _trainable(self) -> bool:
        return any(p.requires_grad for p in self.backbone.parameters())

    def forward_features(
        self,
        x: torch.Tensor,
        *,
        trainable_override: bool | None = None,
    ) -> dict:
        if self.use_lora:
            trainable_override = None
        trainable = bool(trainable_override) if trainable_override is not None else self._trainable()
        self.backbone.train(self.training and trainable)
        with torch.set_grad_enabled(self.training and trainable):
            return self._forward_features_no_mask(x)


SharedCLIPVisionBackbone = SharedMAEVisionBackbone
SharedDinoV3Backbone = SharedMAEVisionBackbone


class _DetBackboneAdapter(nn.Module):
    """
    Torchvision detector backbone adapter.
    Returns a single feature map tensor and exposes out_channels.
    """

    def __init__(
        self,
        shared: SharedMAEVisionBackbone,
        out_channels: int = 256,
        *,
        trainable_backbone: bool = True,
    ):
        super().__init__()
        self.shared = shared
        self.proj = nn.Conv2d(shared.embed_dim, out_channels, kernel_size=1)
        self.out_channels = int(out_channels)
        self.trainable_backbone = bool(trainable_backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        ph, pw = self.shared.patch_size

        pad_h = (ph - height % ph) % ph
        pad_w = (pw - width % pw) % pw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            height += pad_h
            width += pad_w

        tokens = self.shared.forward_features(
            x,
            trainable_override=self.trainable_backbone,
        )["x_norm_patchtokens"]
        bsz, n, c = tokens.shape
        h_patch = height // ph
        w_patch = width // pw
        if h_patch * w_patch != n:
            raise ValueError(f"Token mismatch: h_patch*w_patch={h_patch*w_patch} vs N={n}")

        feat = tokens.reshape(bsz, h_patch, w_patch, c).permute(0, 3, 1, 2)
        return self.proj(feat)


class DinoSegHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, feats: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        logits = self.decode(feats)
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


class DinoCountHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, feats: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        density = self.decode(feats)
        density = F.softplus(density)
        return F.interpolate(density, size=out_size, mode="bilinear", align_corners=False)


@dataclass(frozen=True)
class TaskOutputs:
    det_loss: torch.Tensor
    seg_loss: torch.Tensor
    cnt_loss: torch.Tensor


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        shared: SharedMAEVisionBackbone,
        det_num_classes: int,
        seg_num_classes: int,
        cnt_num_classes: int,
        image_size: int = 448,
        det_out_channels: int = 256,
        det_image_mean: tuple[float, float, float] | None = None,
        det_image_std: tuple[float, float, float] | None = None,
        det_train_backbone: bool = True,
        seg_train_backbone: bool = True,
        cnt_train_backbone: bool = True,
    ):
        super().__init__()
        self.shared = shared
        self.det_train_backbone = bool(det_train_backbone)
        self.seg_train_backbone = bool(seg_train_backbone)
        self.cnt_train_backbone = bool(cnt_train_backbone)
        det_image_mean = tuple(det_image_mean or shared.image_mean or DEFAULT_IMAGE_MEAN)
        det_image_std = tuple(det_image_std or shared.image_std or DEFAULT_IMAGE_STD)

        det_backbone = _DetBackboneAdapter(
            shared,
            out_channels=det_out_channels,
            trainable_backbone=self.det_train_backbone,
        )
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
        self.detector = FasterRCNN(
            det_backbone,
            num_classes=int(det_num_classes) + 1,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=image_size,
            max_size=image_size,
            image_mean=det_image_mean,
            image_std=det_image_std,
        )

        self.seg_head = DinoSegHead(shared.embed_dim, int(seg_num_classes))
        self.cnt_head = DinoCountHead(shared.embed_dim, int(cnt_num_classes))
        self.cnt_num_classes = int(cnt_num_classes)

    def forward_det(self, images, targets=None):
        return self.detector(images, targets)

    def forward_seg(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, h, w = x.shape
        ph, pw = self.shared.patch_size
        tokens = self.shared.forward_features(
            x,
            trainable_override=self.seg_train_backbone,
        )["x_norm_patchtokens"]
        _, n, c = tokens.shape
        h_patch = h // ph
        w_patch = w // pw
        feat = tokens.reshape(bsz, h_patch, w_patch, c).permute(0, 3, 1, 2)
        return self.seg_head(feat, (h, w))

    @staticmethod
    def _cnt_feat_with_scaled_backbone_grad(feat: torch.Tensor, mult: float) -> torch.Tensor:
        m = float(mult)
        if m == 1.0:
            return feat
        if not feat.requires_grad:
            return feat
        feat2 = feat.clone()
        feat2.register_hook(lambda g: g * m)
        return feat2

    def forward_cnt(self, x: torch.Tensor, *, cnt_backbone_grad_mult: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, orig_h, orig_w = x.shape
        ph, pw = self.shared.patch_size
        pad_h = (ph - orig_h % ph) % ph
        pad_w = (pw - orig_w % pw) % pw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        height = orig_h + pad_h
        width = orig_w + pad_w

        tokens = self.shared.forward_features(
            x,
            trainable_override=self.cnt_train_backbone,
        )["x_norm_patchtokens"]
        _, n, c = tokens.shape
        h_patch = height // ph
        w_patch = width // pw
        if h_patch * w_patch != n:
            raise ValueError(f"Token mismatch: h_patch*w_patch={h_patch*w_patch}, num_tokens={n}")
        feat = tokens.reshape(bsz, h_patch, w_patch, c).permute(0, 3, 1, 2)

        feat = self._cnt_feat_with_scaled_backbone_grad(feat, cnt_backbone_grad_mult)

        density = self.cnt_head(feat, (height, width))
        density = density[:, :, :orig_h, :orig_w]
        counts = density.flatten(2).sum(dim=2)
        return density, counts

    def forward(self, mode: str, *args, **kwargs):
        mode = str(mode).lower()
        if mode == "det":
            return self.forward_det(*args, **kwargs)
        if mode == "seg":
            return self.forward_seg(*args, **kwargs)
        if mode == "cnt":
            return self.forward_cnt(*args, **kwargs)
        if mode == "seg_cnt":
            return self.forward_seg_and_cnt(*args, **kwargs)
        raise ValueError(f"Unknown forward mode: {mode}")

    def forward_seg_and_cnt(
        self,
        seg_x: torch.Tensor,
        cnt_x: torch.Tensor,
        *,
        cnt_backbone_grad_mult: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.seg_train_backbone != self.cnt_train_backbone:
            raise ValueError("forward_seg_and_cnt requires seg_train_backbone == cnt_train_backbone")

        if seg_x.dim() != 4 or cnt_x.dim() != 4:
            raise ValueError("Expected seg_x/cnt_x to be [B,3,H,W] tensors")
        if seg_x.shape[1:] != cnt_x.shape[1:]:
            raise ValueError(f"seg_x and cnt_x must have same C/H/W, got {tuple(seg_x.shape)} vs {tuple(cnt_x.shape)}")

        seg_bsz, _, h, w = seg_x.shape
        cnt_bsz = int(cnt_x.shape[0])
        ph, pw = self.shared.patch_size
        if (h % ph) or (w % pw):
            raise ValueError(f"Fused seg+cnt requires patch-aligned H/W, got {(h, w)} for patch {self.shared.patch_size}")

        x = torch.cat([seg_x, cnt_x], dim=0)
        tokens = self.shared.forward_features(x, trainable_override=self.seg_train_backbone)["x_norm_patchtokens"]
        _, n, c = tokens.shape
        h_patch = h // ph
        w_patch = w // pw
        if h_patch * w_patch != n:
            raise ValueError(f"Token mismatch: h_patch*w_patch={h_patch*w_patch}, num_tokens={n}")

        feat = tokens.reshape(seg_bsz + cnt_bsz, h_patch, w_patch, c).permute(0, 3, 1, 2)
        seg_feat = feat[:seg_bsz]
        cnt_feat = feat[seg_bsz:]

        seg_logits = self.seg_head(seg_feat, (h, w))
        cnt_feat = self._cnt_feat_with_scaled_backbone_grad(cnt_feat, cnt_backbone_grad_mult)
        density = self.cnt_head(cnt_feat, (h, w))
        counts = density.flatten(2).sum(dim=2)
        return seg_logits, density, counts

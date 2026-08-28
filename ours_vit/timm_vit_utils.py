from __future__ import annotations

import os

import torch

try:
    import timm
except ImportError:
    timm = None


DEFAULT_TIMM_MODEL_NAME = "vit_large_patch16_224.orig_in21k"
SKIP_PRETRAINED_BACKBONE = "__skip_pretrained_backbone__"


def _env_true(name: str) -> bool:
    val = os.getenv(name, "")
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _looks_like_network_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    keys = (
        "maxretryerror",
        "newconnectionerror",
        "network is unreachable",
        "temporary failure in name resolution",
        "failed to establish a new connection",
        "connection refused",
        "timed out",
        "read timeout",
        "httpsconnectionpool",
        "huggingface.co",
        "hf-mirror",
    )
    return any(k in msg for k in keys)


def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _strip_prefix(state: dict, prefix: str) -> dict:
    if not any(isinstance(k, str) and k.startswith(prefix) for k in state.keys()):
        return state
    stripped = {
        k[len(prefix) :]: v
        for k, v in state.items()
        if isinstance(k, str) and k.startswith(prefix)
    }
    return stripped or state


def load_timm_backbone_state(path: str) -> dict:
    state = _torch_load_cpu(path)
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")

    for key in ("backbone", "state_dict", "model"):
        nested = state.get(key)
        if isinstance(nested, dict):
            state = nested

    state = _strip_prefix(state, "module.")
    state = _strip_prefix(state, "backbone.")
    return state


def build_timm_vit_backbone(
    *,
    model_name: str,
    image_size: int,
    checkpoint_path: str | None,
) -> torch.nn.Module:
    if timm is None:
        raise ImportError("Cannot import timm - install timm first")

    path = checkpoint_path
    use_pretrained = path is None
    if path in {"", SKIP_PRETRAINED_BACKBONE}:
        use_pretrained = False
        path = None

    # Respect explicit offline flags when no local checkpoint is provided.
    if use_pretrained and (_env_true("TIMM_FORCE_OFFLINE") or _env_true("HF_HUB_OFFLINE") or _env_true("TRANSFORMERS_OFFLINE")):
        use_pretrained = False
        print("[timm] offline mode enabled, skip pretrained download and initialize backbone randomly")

    try:
        backbone = timm.create_model(
            str(model_name or DEFAULT_TIMM_MODEL_NAME),
            pretrained=bool(use_pretrained),
            img_size=int(image_size),
            num_classes=0,
            global_pool="",
            dynamic_img_size=True,
        )
    except TypeError as exc:
        raise RuntimeError("Installed timm does not support dynamic_img_size for this ViT backbone") from exc
    except Exception as exc:
        if use_pretrained and _looks_like_network_error(exc):
            print(f"[timm][warn] pretrained download failed ({exc}); fallback to pretrained=False")
            backbone = timm.create_model(
                str(model_name or DEFAULT_TIMM_MODEL_NAME),
                pretrained=False,
                img_size=int(image_size),
                num_classes=0,
                global_pool="",
                dynamic_img_size=True,
            )
        else:
            raise

    if path:
        state = load_timm_backbone_state(str(path))
        backbone.load_state_dict(state, strict=False)
    return backbone


def get_patch_size(backbone: torch.nn.Module) -> tuple[int, int]:
    patch_embed = getattr(backbone, "patch_embed", None)
    patch_size = getattr(patch_embed, "patch_size", None)
    if patch_size is None:
        patch_size = getattr(backbone, "patch_size", None)
    if patch_size is None:
        raise AttributeError("Cannot determine patch size for timm ViT backbone")
    if isinstance(patch_size, (tuple, list)):
        return int(patch_size[0]), int(patch_size[1])
    size = int(patch_size)
    return size, size


def get_num_prefix_tokens(backbone: torch.nn.Module) -> int:
    num_prefix_tokens = getattr(backbone, "num_prefix_tokens", None)
    if num_prefix_tokens is not None:
        return int(num_prefix_tokens)

    count = 0
    if getattr(backbone, "cls_token", None) is not None:
        count += 1
    reg_token = getattr(backbone, "reg_token", None)
    if reg_token is not None:
        count += int(reg_token.shape[-2]) if reg_token.ndim >= 2 else 1
    return max(count, 0)


def _extract_token_tensor(outputs: object) -> torch.Tensor:
    if isinstance(outputs, dict):
        tokens = outputs.get("x_norm_patchtokens")
        if not torch.is_tensor(tokens):
            raise TypeError("Unsupported dict backbone output without x_norm_patchtokens tensor")
        return tokens
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]
    if not torch.is_tensor(outputs):
        raise TypeError(f"Unsupported backbone output type: {type(outputs)}")
    return outputs


def extract_patch_tokens(backbone: torch.nn.Module, outputs: object) -> torch.Tensor:
    tokens = _extract_token_tensor(outputs)
    n_prefix = get_num_prefix_tokens(backbone)
    return tokens[:, n_prefix:]


def forward_timm_vit_prelude(backbone: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    x = backbone.patch_embed(x)
    if hasattr(backbone, "_pos_embed"):
        x = backbone._pos_embed(x)
    else:
        raise AttributeError("timm ViT backbone is missing _pos_embed")

    patch_drop = getattr(backbone, "patch_drop", None)
    if patch_drop is not None:
        x = patch_drop(x)

    norm_pre = getattr(backbone, "norm_pre", None)
    if norm_pre is not None:
        x = norm_pre(x)
    return x


def finalize_timm_vit_tokens(backbone: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    norm = getattr(backbone, "norm", None)
    if norm is not None:
        x = norm(x)
    return x

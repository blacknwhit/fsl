from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch

try:
    from transformers import ViTImageProcessor
    from transformers.image_utils import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
except ImportError:
    ViTImageProcessor = None
    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


DEFAULT_MAE_MODEL_NAME = "facebook/vit-mae-large"
DEFAULT_IMAGE_MEAN = tuple(float(x) for x in IMAGENET_DEFAULT_MEAN)
DEFAULT_IMAGE_STD = tuple(float(x) for x in IMAGENET_DEFAULT_STD)


def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_backbone_source(model_name: str | None, checkpoint_path: str | None) -> str:
    source = str(model_name or DEFAULT_MAE_MODEL_NAME)
    if not checkpoint_path:
        return source

    ckpt_path = Path(checkpoint_path)
    if ckpt_path.exists() and ckpt_path.is_dir():
        return str(ckpt_path)
    if not ckpt_path.exists():
        return str(checkpoint_path)
    return source


def load_mae_backbone_state(path: str) -> dict:
    state = _torch_load_cpu(path)
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {path}")
    if isinstance(state.get("backbone"), dict):
        state = state["backbone"]
    if isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    if any(isinstance(k, str) and k.startswith("backbone.") for k in state.keys()):
        state = {
            k[len("backbone.") :]: v
            for k, v in state.items()
            if isinstance(k, str) and k.startswith("backbone.")
        }
    return state


def get_vit_image_stats(model_name: str | None = None) -> Tuple[tuple[float, float, float], tuple[float, float, float]]:
    source = str(model_name or DEFAULT_MAE_MODEL_NAME)
    if ViTImageProcessor is not None:
        try:
            processor = ViTImageProcessor.from_pretrained(source, local_files_only=True)
            mean = tuple(float(x) for x in processor.image_mean)
            std = tuple(float(x) for x in processor.image_std)
            return mean, std
        except Exception:
            pass
    return DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD

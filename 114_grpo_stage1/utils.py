from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

import torch


def parse_loss_weights(text: str) -> Tuple[float, float, float]:
    s = (text or "").strip()
    if not s:
        return 1.0, 1.0, 1.0
    for sep in (",", ":", "/"):
        if sep in s:
            parts = [p.strip() for p in s.split(sep)]
            break
    else:
        parts = [s]
    if len(parts) != 3:
        raise ValueError("--loss-weights must have 3 numbers, e.g. '1,1,1' or '1:1:1'")
    w = tuple(float(p) for p in parts)
    return float(w[0]), float(w[1]), float(w[2])


def infinite_loader(loader: Iterable) -> Iterator:
    while True:
        for batch in loader:
            yield batch


def choose_primary(lengths: Dict[str, int], override: str | None) -> str:
    if override:
        key = override.lower()
        if key not in lengths:
            raise ValueError(f"--primary-task must be one of {sorted(lengths.keys())}")
        return key
    return max(lengths.items(), key=lambda kv: kv[1])[0]


def _filter_det_head_state(det_state: dict) -> dict:
    drop_prefix = "backbone.shared.backbone."
    return {k: v for k, v in det_state.items() if not k.startswith(drop_prefix)}


def save_multitask_checkpoint(
    path: str,
    *,
    model,
    optimizer,
    epoch: int,
    best_by: str,
    metrics: Dict[str, float],
    loss_weights: Tuple[float, float, float],
    phi_state: Dict | None = None,
    config: Dict | None = None,
    state_feature_state: Dict | None = None,
):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "version": 1,
        "epoch": int(epoch),
        "best_by": str(best_by),
        "metrics": dict(metrics),
        "loss_weights": tuple(float(x) for x in loss_weights),
        "phi_state": phi_state,
        "config": dict(config) if config is not None else None,
        "state_feature_state": dict(state_feature_state) if state_feature_state is not None else None,
        # NOTE: save the full shared module so LoRA-MoE/task-embeddings are preserved.
        # Downstream single-task eval loaders will ignore extra keys with strict=False.
        "backbone": model.shared.state_dict(),
        "det_head": _filter_det_head_state(model.detector.state_dict()),
        "seg_head": model.seg_head.state_dict(),
        "cnt_head": model.cnt_head.state_dict(),
        "optimizer": optimizer.state_dict() if (optimizer is not None and hasattr(optimizer, "state_dict")) else None,
    }
    if isinstance(optimizer, dict):
        ckpt["optimizers"] = {k: (v.state_dict() if v is not None else None) for k, v in optimizer.items()}
    torch.save(ckpt, p)


def load_multitask_checkpoint(path: str, *, model, map_location: str | torch.device = "cpu") -> dict:
    p = Path(path)
    ckpt = torch.load(p, map_location=map_location)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Invalid checkpoint format: {p}")
    shared_state = ckpt.get("backbone")
    det_state = ckpt.get("det_head")
    seg_state = ckpt.get("seg_head")
    cnt_state = ckpt.get("cnt_head")
    if not all(isinstance(x, dict) for x in (shared_state, det_state, seg_state, cnt_state)):
        raise ValueError(f"Checkpoint missing multitask weights: {p}")

    model.shared.load_state_dict(shared_state, strict=False)
    model.detector.load_state_dict(det_state, strict=False)
    model.seg_head.load_state_dict(seg_state, strict=False)
    model.cnt_head.load_state_dict(cnt_state, strict=False)
    return ckpt

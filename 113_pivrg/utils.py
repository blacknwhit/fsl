from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, Sequence, Tuple

import torch


def parse_int_list(text: str | Sequence[int], *, expected_len: int | None = None) -> tuple[int, ...]:
    if isinstance(text, (list, tuple)):
        values = tuple(int(value) for value in text)
    else:
        raw = str(text or "").strip()
        if not raw:
            values = tuple()
        else:
            values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if expected_len is not None and len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} integers, got {len(values)} from {text!r}")
    return values


def parse_loss_weights(text: str | Sequence[float]) -> Tuple[float, float, float]:
    if isinstance(text, (list, tuple)):
        values = tuple(float(value) for value in text)
    else:
        raw = str(text or "").strip()
        if not raw:
            values = (15.0, 8.0, 1.0)
        else:
            for sep in (",", ":", "/"):
                if sep in raw:
                    parts = [part.strip() for part in raw.split(sep) if part.strip()]
                    break
            else:
                parts = [raw]
            values = tuple(float(part) for part in parts)
    if len(values) != 3:
        raise ValueError("Expected exactly 3 loss weights in det,seg,cnt order.")
    return float(values[0]), float(values[1]), float(values[2])


def infinite_loader(loader: Iterable) -> Iterator:
    while True:
        for batch in loader:
            yield batch


def choose_primary(lengths: Dict[str, int], override: str | None) -> str:
    if override:
        key = str(override).lower()
        if key not in lengths:
            raise ValueError(f"--primary-task must be one of {sorted(lengths.keys())}")
        return key
    return max(lengths.items(), key=lambda item: item[1])[0]


def _filter_det_head_state(det_state: dict) -> dict:
    drop_prefix = "backbone.shared."
    return {key: value for key, value in det_state.items() if not key.startswith(drop_prefix)}


def save_multitask_checkpoint(
    path: str,
    *,
    model,
    optimizer,
    epoch: int,
    metrics: Dict[str, float],
    loss_weights: Tuple[float, float, float],
    model_config: Dict | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "version": 1,
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "loss_weights": tuple(float(weight) for weight in loss_weights),
        "model_config": dict(model_config or {}),
        "backbone": model.shared.state_dict(),
        "det_head": _filter_det_head_state(model.detector.state_dict()),
        "seg_head": model.seg_head.state_dict(),
        "cnt_head": model.cnt_head.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None and hasattr(optimizer, "state_dict") else None,
    }
    torch.save(checkpoint, target)

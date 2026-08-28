from pathlib import Path
from typing import Optional

import torch


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    path: str,
    save_full_model: bool = False,
    meta: Optional[dict] = None,
):
    """
    推荐默认只存 backbone+head（不重复存整模）：
    - backbone/head: 适合全参训练/只训head，两者都能恢复
    - model: 可选（save_full_model=True 时才存），会与 backbone/head 重复，占空间
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "version": 2,
        "epoch": epoch,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "backbone": getattr(model, "backbone", None).state_dict() if hasattr(model, "backbone") else None,
        "head": getattr(model, "head", None).state_dict() if hasattr(model, "head") else None,
        "meta": meta or {},
    }
    if save_full_model:
        ckpt["model"] = model.state_dict()

    torch.save(ckpt, path)


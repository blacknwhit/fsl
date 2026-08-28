from pathlib import Path
from typing import Optional, Sequence

import torch


def save_checkpoint(model, optimizer, epoch: int, path: str, save_full_model: bool = False):
    """
    保存 checkpoint（推荐：不重复存整模，默认只存 backbone+head）：
    - backbone/head: 适合全参训练/只训head，两者都能恢复
    - model: 可选（save_full_model=True 时才存），会与 backbone/head 重复，占用更多空间
    - optimizer/epoch: 训练恢复用
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "version": 2,
        "epoch": epoch,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "backbone": getattr(model, "backbone", None).state_dict() if hasattr(model, "backbone") else None,
        "head": getattr(model, "head", None).state_dict() if hasattr(model, "head") else None,
    }
    if save_full_model:
        ckpt["model"] = model.state_dict()

    torch.save(ckpt, path)


@torch.no_grad()
def compute_iou(logits: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: Optional[int] = None):
    # logits: [B, C, H, W], target: [B, H, W]
    preds = logits.argmax(dim=1)
    ious = []
    for cls in range(num_classes):
        if ignore_index is not None:
            mask = target != ignore_index
            pred_cls = preds.eq(cls) & mask
            target_cls = target.eq(cls) & mask
        else:
            pred_cls = preds.eq(cls)
            target_cls = target.eq(cls)
        inter = (pred_cls & target_cls).sum().item()
        union = pred_cls.sum().item() + target_cls.sum().item() - inter
        if union == 0:
            continue
        ious.append(inter / union)
    if not ious:
        return 0.0
    return sum(ious) / len(ious)


@torch.no_grad()
def update_confusion_matrix(
    conf: torch.Tensor,
    logits_or_preds: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    ignore_indices: Optional[Sequence[int]] = None,
):
    """Accumulate a confusion matrix over a dataset.

    Args:
        conf: [K, K] int64 tensor on CPU (preferred) to be updated in-place.
        logits_or_preds: [B, C, H, W] logits or [B, H, W] preds.
        target: [B, H, W] int tensor with class ids.
        num_classes: K.
        ignore_indices: values in target to ignore (e.g., 255).
    """
    if logits_or_preds.dim() == 4:
        preds = logits_or_preds.argmax(dim=1)
    elif logits_or_preds.dim() == 3:
        preds = logits_or_preds
    else:
        raise ValueError(f"Expected logits [B,C,H,W] or preds [B,H,W], got {tuple(logits_or_preds.shape)}")

    preds = preds.reshape(-1).to(torch.int64)
    target = target.reshape(-1).to(torch.int64)

    valid = (target >= 0) & (target < num_classes)
    if ignore_indices:
        for v in ignore_indices:
            valid &= target.ne(int(v))

    if valid.any():
        target_v = target[valid]
        preds_v = preds[valid]
        idx = target_v * num_classes + preds_v
        hist = torch.bincount(idx, minlength=num_classes * num_classes)
        conf += hist.reshape(num_classes, num_classes).cpu()


@torch.no_grad()
def per_class_iou_from_confusion(conf: torch.Tensor):
    """Compute per-class IoU and mean IoU from a confusion matrix.

    conf: [K,K] where rows=GT, cols=Pred.
    Returns:
        iou: [K] float tensor with NaN for classes with union==0
        miou: float tensor (nanmean over classes)
    """
    conf_f = conf.to(dtype=torch.float64)
    tp = torch.diag(conf_f)
    gt = conf_f.sum(dim=1)
    pred = conf_f.sum(dim=0)
    union = gt + pred - tp
    iou = torch.where(union > 0, tp / union, torch.full_like(union, float("nan")))
    miou = torch.nanmean(iou)
    return iou, miou

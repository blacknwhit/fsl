from pathlib import Path
from typing import Dict, List, Tuple

import torch


def save_checkpoint(model, optimizer, epoch: int, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        path,
    )


@torch.no_grad()
def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Pairwise IoU between two sets of boxes.
    boxes1: [N,4], boxes2: [M,4] in xyxy.
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


@torch.no_grad()
def compute_precision_recall(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
) -> Tuple[int, int, int]:
    """
    Simple overall precision/recall at IoU threshold.

    Returns (tp, fp, fn) across the batch.
    """
    true_pos = 0
    false_pos = 0
    false_neg = 0

    for pred, tgt in zip(predictions, targets):
        pred_boxes = pred.get("boxes", torch.zeros((0, 4), device=pred["labels"].device))
        pred_labels = pred.get("labels", torch.zeros((0,), dtype=torch.int64, device=pred_boxes.device))
        pred_scores = pred.get("scores", torch.ones((pred_boxes.shape[0],), device=pred_boxes.device))

        keep = pred_scores >= score_threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        if pred_boxes.numel():
            order = pred_scores.argsort(descending=True)
            pred_boxes = pred_boxes[order]
            pred_labels = pred_labels[order]

        gt_boxes = tgt["boxes"]
        gt_labels = tgt["labels"]
        matched_gt = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool, device=gt_boxes.device)

        for pb, pl in zip(pred_boxes, pred_labels):
            candidates_mask = (gt_labels == pl) & (~matched_gt)
            if not candidates_mask.any():
                false_pos += 1
                continue

            candidates = gt_boxes[candidates_mask]
            ious = box_iou(pb.unsqueeze(0), candidates).squeeze(0)
            max_iou, max_pos = ious.max(dim=0)
            if max_iou >= iou_threshold:
                global_idx = candidates_mask.nonzero(as_tuple=False)[max_pos].item()
                matched_gt[global_idx] = True
                true_pos += 1
            else:
                false_pos += 1

        false_neg += (~matched_gt).sum().item()

    return true_pos, false_pos, false_neg


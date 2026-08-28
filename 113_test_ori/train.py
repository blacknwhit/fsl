from __future__ import annotations

import argparse
import math
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler

from .datasets import build_cnt_loaders, build_det_loaders, build_seg_loaders
from .models import MultiTaskModel, SharedDinoV3Backbone
from .utils import choose_primary, infinite_loader, parse_loss_weights, save_multitask_checkpoint
from .auto_weighted_loss import AutomaticWeightedLoss
from segmentation.utils import per_class_iou_from_confusion, update_confusion_matrix


def parse_args():
    p = argparse.ArgumentParser(description="Multi-task training (det/seg/count) with shared DINOv3 backbone")

    # Backbone
    p.add_argument("--model-name", type=str, default="dinov3_vitl16")
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--backbone-checkpoint", type=str, default=None)
    p.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="legacy: force unfreeze shared backbone for all tasks (overrides per-task freeze flags)",
    )
    
    # LoRA-MoE parameters (private + shared pools)
    p.add_argument("--use-lora-moe", action="store_true", help="Enable LoRA-MoE adapters")
    p.add_argument("--lora-rank", type=int, default=8, help="LoRA rank for experts (shared across pools)")
    p.add_argument("--num-experts-private", type=int, default=2, help="Private experts per task per block")
    p.add_argument("--num-experts-shared", type=int, default=6, help="Shared experts per block")
    p.add_argument("--moe-k-private", type=int, default=2, help="Top-k private experts per token")
    p.add_argument("--moe-k-shared", type=int, default=2, help="Top-k shared experts per token")

    # MI loss on shared pool
    p.add_argument("--moe-mi-loss-shared", type=float, default=0.005, help="MI loss weight for shared experts")
    mi = p.add_mutually_exclusive_group()
    mi.add_argument("--use-mi-shared", dest="use_mi_shared", action="store_true", help="Enable MI loss on shared pool")
    mi.add_argument("--no-mi-shared", dest="use_mi_shared", action="store_false", help="Disable MI loss on shared pool")
    p.set_defaults(use_mi_shared=True)
    
    # Use AutomaticWeightedLoss for main task losses
    p.add_argument("--use-auto-weighted-loss", action="store_true", help="Use uncertainty-based automatic loss weighting")
    # Learnable linear weights (beta1~3) for det/seg/cnt
    p.add_argument(
        "--dynamic-loss-weight",
        action="store_true",
        help="Learn beta1~3 for det/seg/cnt; total=beta1*det+beta2*seg+beta3*cnt",
    )

    # Memory optimization (does NOT change losses): gradient checkpointing.
    gc = p.add_mutually_exclusive_group()
    gc.add_argument(
        "--grad-checkpointing",
        dest="grad_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing in LoRA-MoE backbone blocks to reduce VRAM.",
    )
    gc.add_argument(
        "--no-grad-checkpointing",
        dest="grad_checkpointing",
        action="store_false",
        help="Disable gradient checkpointing.",
    )
    p.set_defaults(grad_checkpointing=True)

    # Per-task backbone finetune flags (match single-task defaults)
    det_ft = p.add_mutually_exclusive_group()
    det_ft.add_argument(
        "--det-unfreeze-backbone",
        dest="det_unfreeze_backbone",
        action="store_true",
        help="detection: train backbone (default: on in multitask)",
    )
    det_ft.add_argument(
        "--det-freeze-backbone",
        dest="det_unfreeze_backbone",
        action="store_false",
        help="detection: freeze backbone",
    )
    p.set_defaults(det_unfreeze_backbone=True)
    seg_ft = p.add_mutually_exclusive_group()
    seg_ft.add_argument("--seg-full-finetune", dest="seg_full_finetune", action="store_true")
    seg_ft.add_argument("--seg-freeze-backbone", dest="seg_full_finetune", action="store_false")
    p.set_defaults(seg_full_finetune=True)
    cnt_ft = p.add_mutually_exclusive_group()
    cnt_ft.add_argument("--cnt-full-finetune", dest="cnt_full_finetune", action="store_true")
    cnt_ft.add_argument("--cnt-freeze-backbone", dest="cnt_full_finetune", action="store_false")
    p.set_defaults(cnt_full_finetune=True)

    # Detection dataset
    p.add_argument("--det-data-root", type=str, required=True)
    p.add_argument("--det-train-ann", type=str, default=None)
    p.add_argument("--det-val-ann", type=str, default=None)
    p.add_argument("--det-train-img-dir", type=str, default=None)
    p.add_argument("--det-val-img-dir", type=str, default=None)
    p.add_argument("--det-num-classes", type=int, default=None, help="foreground class count (auto if None)")

    # Seg dataset
    p.add_argument("--seg-train-dir", type=str, required=True)
    p.add_argument("--seg-val-dir", type=str, required=True)
    p.add_argument("--seg-num-classes", type=int, default=11)

    # Count dataset
    p.add_argument("--cnt-data-root", type=str, required=True)
    p.add_argument("--cnt-train-dir", type=str, default=None)
    p.add_argument("--cnt-val-dir", type=str, default=None)
    p.add_argument("--cnt-num-classes", type=int, default=8)
    p.add_argument("--cnt-count-loss-weight", type=float, default=1, help="aux L1 weight inside counting loss")
    aspect = p.add_mutually_exclusive_group()
    aspect.add_argument("--cnt-keep-aspect", dest="cnt_keep_aspect", action="store_true")
    aspect.add_argument("--cnt-no-keep-aspect", dest="cnt_keep_aspect", action="store_false")
    p.set_defaults(cnt_keep_aspect=True)

    # Training
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument(
        "--val-every",
        type=int,
        default=1,
        help="Run validation every N epochs (default: 1 = validate every epoch).",
    )
    p.add_argument("--det-batch-size", type=int, default=2)
    p.add_argument("--seg-batch-size", type=int, default=8)
    p.add_argument("--cnt-batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--backbone-lr",
        type=float,
        default=None,
        help="shared backbone lr (default: lr * backbone_lr_mult)",
    )
    p.add_argument(
        "--backbone-lr-mult",
        "--cnt-backbone-lr-mult",
        dest="backbone_lr_mult",
        type=float,
        default=0.1,
        help="shared backbone lr multiplier relative to --lr (default: 0.1). `--cnt-backbone-lr-mult` is a deprecated alias.",
    )
    p.add_argument("--det-lr", type=float, default=None, help="override detection lr (default: --lr)")
    p.add_argument("--seg-lr", type=float, default=None, help="override segmentation lr (default: --lr)")
    p.add_argument("--cnt-lr", type=float, default=None, help="override counting lr (default: --lr)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument(
        "--backbone-weight-decay",
        type=float,
        default=None,
        help="shared backbone weight decay (default: --weight-decay)",
    )
    p.add_argument("--det-weight-decay", type=float, default=None, help="override detection weight decay (default: --weight-decay)")
    p.add_argument("--seg-weight-decay", type=float, default=None, help="override segmentation weight decay (default: --weight-decay)")
    p.add_argument("--cnt-weight-decay", type=float, default=None, help="override counting weight decay (default: --weight-decay)")
    p.add_argument("--loss-weights", type=str, default="1,1,1", help="det,seg,cnt e.g. 1,1,1")
    p.add_argument("--primary-task", type=str, default=None, help="override primary task: det|seg|cnt")
    p.add_argument("--best-by", type=str, default="total", choices=["total", "det", "seg", "cnt"])
    p.add_argument("--save-dir", type=str, default="runs/multitask")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--amp", action="store_true")
    p.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help="clip grad norm (0 disables). Applied to ALL trainable parameters, like counting single-task.",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max-train-steps", type=int, default=0, help="0 = no limit")
    p.add_argument("--max-val-steps", type=int, default=0, help="0 = no limit")

    p.add_argument(
        "--save-epochs",
        type=str,
        default="60,100",
        help='额外保存指定 epoch 的权重（逗号分隔），例如："60,100"；留空则不额外保存',
    )

    fuse = p.add_mutually_exclusive_group()
    fuse.add_argument("--fuse-seg-cnt-backbone", dest="fuse_seg_cnt_backbone", action="store_true")
    fuse.add_argument("--no-fuse-seg-cnt-backbone", dest="fuse_seg_cnt_backbone", action="store_false")
    p.set_defaults(fuse_seg_cnt_backbone=True)

    p.add_argument(
        "--cnt-backbone-grad-mult",
        type=float,
        default=1.0,
        help=(
            "Scale ONLY the counting-task gradient flowing into the shared backbone. "
            "1.0 = no scaling; 0.0 = counting does not update backbone; "
            "cnt head gradients are NOT scaled."
        ),
    )

    return p.parse_args()


def _parse_save_epochs(spec: str) -> set[int]:
    spec = (spec or "").strip()
    if not spec:
        return set()
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            e = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"--save-epochs 非法值: {part!r}（需要整数，用逗号分隔）") from exc
        if e <= 0:
            raise argparse.ArgumentTypeError(f"--save-epochs 非法值: {e}（epoch 必须为正整数）")
        out.add(e)
    return out


def _to_device_det(batch, device: torch.device):
    images, targets = batch
    images = [img.to(device, non_blocking=True) for img in images]
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return images, targets


def _to_device_seg(batch, device: torch.device):
    imgs, masks = batch
    return imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)


def _to_device_cnt(batch, device: torch.device):
    imgs, dens = batch
    imgs = imgs.to(device, non_blocking=True).float()
    dens = dens.to(device, non_blocking=True).float()
    return imgs, dens


@torch.no_grad()
def _eval_det_loss(model: MultiTaskModel, loader, device: torch.device, *, amp: bool, max_steps: int) -> float:
    model.detector.train()  # FasterRCNN only returns losses in train mode
    total = 0.0
    steps = 0
    autocast_device = device.type if device.type in {"cuda", "cpu"} else "cuda"
    for images, targets in loader:
        images, targets = _to_device_det((images, targets), device)
        with torch.amp.autocast(autocast_device, enabled=amp):
            loss_dict = model.forward_det(images, targets)
            loss = sum(loss_dict.values())
        total += float(loss.item())
        steps += 1
        if max_steps and steps >= max_steps:
            break
    return total / max(steps, 1)


@torch.no_grad()
def _eval_seg_loss(
    model: MultiTaskModel,
    loader,
    device: torch.device,
    *,
    amp: bool,
    max_steps: int,
    num_classes: int,
) -> tuple[float, float]:
    model.eval()
    total = 0.0
    samples = 0
    steps = 0
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    autocast_device = device.type if device.type in {"cuda", "cpu"} else "cuda"
    for imgs, masks in loader:
        imgs, masks = _to_device_seg((imgs, masks), device)
        with torch.amp.autocast(autocast_device, enabled=amp):
            logits = model.forward_seg(imgs)
            loss = F.cross_entropy(logits, masks)
        bsz = imgs.size(0)
        total += float(loss.item()) * bsz
        samples += bsz
        steps += 1
        update_confusion_matrix(
            conf=conf,
            logits_or_preds=logits.detach(),
            target=masks.detach(),
            num_classes=num_classes,
            ignore_indices=(255, 11),
        )
        if max_steps and steps >= max_steps:
            break
    _, miou = per_class_iou_from_confusion(conf)
    return total / max(samples, 1), float(miou.item())


@torch.no_grad()
def _eval_cnt_loss(
    model: MultiTaskModel,
    loader,
    device: torch.device,
    *,
    amp: bool,
    max_steps: int,
    count_loss_weight: float,
) -> tuple[float, float, float, float]:
    model.eval()
    total = 0.0
    total_density = 0.0
    total_count_mae = 0.0
    total_total_mae = 0.0
    steps = 0
    samples = 0
    autocast_device = device.type if device.type in {"cuda", "cpu"} else "cuda"
    for imgs, dens in loader:
        imgs, dens = _to_device_cnt((imgs, dens), device)
        gt_counts = dens.flatten(2).sum(dim=2)
        with torch.amp.autocast(autocast_device, enabled=amp):
            pred_dens, pred_counts = model.forward_cnt(imgs)
            dens_loss = F.mse_loss(pred_dens, dens, reduction="sum") / imgs.size(0)
            cnt_l1 = F.l1_loss(pred_counts, gt_counts)
            loss = dens_loss + float(count_loss_weight) * cnt_l1
            count_mae = (pred_counts - gt_counts).abs().mean()
            pred_total = pred_counts.sum(dim=1)
            gt_total = gt_counts.sum(dim=1)
            total_mae = (pred_total - gt_total).abs().mean()
        bsz = imgs.size(0)
        samples += bsz
        total += float(loss.item()) * bsz
        total_density += float(dens_loss.item()) * bsz
        total_count_mae += float(count_mae.item()) * bsz
        total_total_mae += float(total_mae.item()) * bsz
        steps += 1
        if max_steps and steps >= max_steps:
            break
    denom = max(samples, 1)
    return (
        total / denom,
        total_density / denom,
        total_count_mae / denom,
        total_total_mae / denom,
    )


def _box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(0.0, br - tl)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def _ap_from_pr(rec: np.ndarray, prec: np.ndarray) -> float:
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = prec[rec >= t].max() if np.any(rec >= t) else 0.0
        ap += p
    return ap / 101.0


@torch.no_grad()
def _eval_det_ap50_fast(
    model: MultiTaskModel,
    loader,
    device: torch.device,
    num_classes: int,
    score_thresh: float = 0.0,
) -> tuple[float, Dict[str, float]]:
    model.detector.eval()
    preds_by_cls = {c: [] for c in range(1, num_classes + 1)}
    gts_by_cls = {c: {} for c in range(1, num_classes + 1)}
    total_gt = 0
    total_pred = 0
    total_tp = 0

    img_counter = 0
    for images, targets in loader:
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model.detector(images)

        for out, tgt in zip(outputs, targets):
            img_id = int(tgt.get("image_id", torch.tensor([img_counter])).item())
            img_counter += 1

            gt_boxes = tgt["boxes"].detach().cpu().numpy()
            gt_labels = tgt["labels"].detach().cpu().numpy().astype(int)
            for box, cls in zip(gt_boxes, gt_labels):
                gts_by_cls[cls].setdefault(img_id, []).append(box)
                total_gt += 1

            pred_boxes = out["boxes"].detach().cpu().numpy()
            pred_labels = out["labels"].detach().cpu().numpy().astype(int)
            pred_scores = out["scores"].detach().cpu().numpy()

            keep = pred_scores >= score_thresh
            for box, cls, score in zip(pred_boxes[keep], pred_labels[keep], pred_scores[keep]):
                preds_by_cls[cls].append((img_id, float(score), box))
                total_pred += 1

    ap_list = []
    for cls in range(1, num_classes + 1):
        preds = preds_by_cls[cls]
        gts = gts_by_cls[cls]
        num_gt = sum(len(v) for v in gts.values())
        if num_gt == 0:
            continue

        preds.sort(key=lambda x: x[1], reverse=True)
        matched = {k: [False] * len(v) for k, v in gts.items()}

        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)

        for i, (img_id, score, pbox) in enumerate(preds):
            if img_id not in gts:
                fp[i] = 1.0
                continue
            gt_boxes = np.array(gts[img_id], dtype=np.float32)
            ious = _box_iou_np(np.array([pbox], dtype=np.float32), gt_boxes)[0]
            j = int(np.argmax(ious)) if ious.size > 0 else -1
            if j >= 0 and ious[j] >= 0.5 and not matched[img_id][j]:
                tp[i] = 1.0
                matched[img_id][j] = True
                total_tp += 1
            else:
                fp[i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / max(num_gt, 1)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        ap_list.append(_ap_from_pr(rec, prec))

    ap50 = float(np.mean(ap_list)) if ap_list else 0.0
    diag = {
        "det_ap50_num_gt": float(total_gt),
        "det_ap50_num_pred": float(total_pred),
        "det_ap50_num_tp": float(total_tp),
    }
    return ap50, diag


def main():
    args = parse_args()
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_epochs = _parse_save_epochs(getattr(args, "save_epochs", ""))

    w_det, w_seg, w_cnt = parse_loss_weights(args.loss_weights)
    backbone_lr = float(args.backbone_lr) if args.backbone_lr is not None else float(args.lr) * float(args.backbone_lr_mult)
    det_lr = float(args.det_lr) if args.det_lr is not None else float(args.lr)
    seg_lr = float(args.seg_lr) if args.seg_lr is not None else float(args.lr)
    cnt_lr = float(args.cnt_lr) if args.cnt_lr is not None else float(args.lr)
    backbone_wd = float(args.backbone_weight_decay) if args.backbone_weight_decay is not None else float(args.weight_decay)
    det_wd = float(args.det_weight_decay) if args.det_weight_decay is not None else float(args.weight_decay)
    seg_wd = float(args.seg_weight_decay) if args.seg_weight_decay is not None else float(args.weight_decay)
    cnt_wd = float(args.cnt_weight_decay) if args.cnt_weight_decay is not None else float(args.weight_decay)

    det_train_backbone = bool(args.det_unfreeze_backbone)
    seg_train_backbone = bool(args.seg_full_finetune)
    cnt_train_backbone = bool(args.cnt_full_finetune)
    if args.unfreeze_backbone:
        det_train_backbone = True
        seg_train_backbone = True
        cnt_train_backbone = True

    det_train_ds, det_val_ds, det_train_loader, det_val_loader = build_det_loaders(
        data_root=args.det_data_root,
        image_size=args.image_size,
        batch_size=args.det_batch_size,
        num_workers=args.num_workers,
        train_ann=args.det_train_ann,
        val_ann=args.det_val_ann,
        train_img_dir=args.det_train_img_dir,
        val_img_dir=args.det_val_img_dir,
    )
    seg_train_ds, seg_val_ds, seg_train_loader, seg_val_loader = build_seg_loaders(
        train_dir=args.seg_train_dir,
        val_dir=args.seg_val_dir,
        image_size=args.image_size,
        batch_size=args.seg_batch_size,
        num_workers=args.num_workers,
    )
    cnt_train_ds, cnt_val_ds, cnt_train_loader, cnt_val_loader = build_cnt_loaders(
        data_root=args.cnt_data_root,
        train_dir=args.cnt_train_dir,
        val_dir=args.cnt_val_dir,
        image_size=args.image_size,
        num_classes=args.cnt_num_classes,
        keep_aspect=bool(args.cnt_keep_aspect),
        batch_size=args.cnt_batch_size,
        num_workers=1,
    )

    det_num_classes = int(args.det_num_classes) if args.det_num_classes else int(det_train_ds.num_classes)

    # Create shared backbone with optional LoRA-MoE
    shared = SharedDinoV3Backbone(
        model_name=args.model_name,
        image_size=args.image_size,
        checkpoint_path=args.backbone_checkpoint,
        # LoRA-MoE parameters
        use_lora_moe=bool(args.use_lora_moe),
        task_num=3,  # det, seg, cnt
        lora_rank=int(args.lora_rank),
        num_experts_private=int(args.num_experts_private),
        num_experts_shared=int(args.num_experts_shared),
        moe_k_private=int(args.moe_k_private),
        moe_k_shared=int(args.moe_k_shared),
        use_mi_shared=bool(args.use_mi_shared),
        grad_checkpointing=bool(args.grad_checkpointing),
    )
    
    # When using LoRA-MoE, backbone is frozen (handled in SharedDinoV3Backbone._setup_lora_moe)
    # Otherwise, set backbone trainability based on per-task flags
    if not args.use_lora_moe:
        any_train_backbone = det_train_backbone or seg_train_backbone or cnt_train_backbone
        for p in shared.backbone.parameters():
            p.requires_grad = bool(any_train_backbone)

    model = MultiTaskModel(
        shared=shared,
        det_num_classes=det_num_classes,
        seg_num_classes=args.seg_num_classes,
        cnt_num_classes=args.cnt_num_classes,
        image_size=args.image_size,
        det_train_backbone=det_train_backbone,
        seg_train_backbone=seg_train_backbone,
        cnt_train_backbone=cnt_train_backbone,
    ).to(device)

    # Build parameter groups
    shared_backbone_params = list(model.shared.backbone.parameters())
    shared_backbone_ids = {id(p) for p in shared_backbone_params}
    
    # Collect trainable parameters
    param_groups = []
    
    if args.use_lora_moe:
        # LoRA-MoE mode: train LoRA params + task heads
        lora_moe_params = []
        for lora_moe in model.shared.lora_moes:
            lora_moe_params.extend([p for p in lora_moe.parameters() if p.requires_grad])
        
        if lora_moe_params:
            param_groups.append({"params": lora_moe_params, "lr": float(args.lr), "weight_decay": float(args.weight_decay)})
        
        # Detection head (excluding backbone params)
        det_head_params = [p for p in model.detector.parameters() if p.requires_grad and id(p) not in shared_backbone_ids]
        # Also exclude LoRA-MoE params from det_head_params
        lora_moe_ids = {id(p) for p in lora_moe_params}
        det_head_params = [p for p in det_head_params if id(p) not in lora_moe_ids]
        
        if det_head_params:
            param_groups.append({"params": det_head_params, "lr": det_lr, "weight_decay": det_wd})
        
        seg_params = [p for p in model.seg_head.parameters() if p.requires_grad]
        if seg_params:
            param_groups.append({"params": seg_params, "lr": seg_lr, "weight_decay": seg_wd})
        
        cnt_head_params = [p for p in model.cnt_head.parameters() if p.requires_grad]
        if cnt_head_params:
            param_groups.append({"params": cnt_head_params, "lr": cnt_lr, "weight_decay": cnt_wd})
        
        print(f"[LoRA-MoE] LoRA params: {sum(p.numel() for p in lora_moe_params)}")
        print(f"[LoRA-MoE] Det head params: {sum(p.numel() for p in det_head_params)}")
        print(f"[LoRA-MoE] Seg head params: {sum(p.numel() for p in seg_params)}")
        print(f"[LoRA-MoE] Cnt head params: {sum(p.numel() for p in cnt_head_params)}")
    else:
        # Standard mode: train backbone + task heads
        backbone_params = [p for p in shared_backbone_params if p.requires_grad]
        det_head_params = [p for p in model.detector.parameters() if p.requires_grad and id(p) not in shared_backbone_ids]
        seg_params = [p for p in model.seg_head.parameters() if p.requires_grad]
        cnt_head_params = [p for p in model.cnt_head.parameters() if p.requires_grad]

        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": backbone_wd})
        if det_head_params:
            param_groups.append({"params": det_head_params, "lr": det_lr, "weight_decay": det_wd})
        if seg_params:
            param_groups.append({"params": seg_params, "lr": seg_lr, "weight_decay": seg_wd})
        if cnt_head_params:
            param_groups.append({"params": cnt_head_params, "lr": cnt_lr, "weight_decay": cnt_wd})
    
    if not param_groups:
        raise RuntimeError("No trainable parameters.")

    if bool(args.dynamic_loss_weight) and bool(args.use_auto_weighted_loss):
        raise ValueError("--dynamic-loss-weight 与 --use-auto-weighted-loss 不能同时开启")

    # AutomaticWeightedLoss (optional): only enable when explicitly requested.
    # In fixed-weight mode, det/seg/cnt are combined via --loss-weights.
    awl = None
    if bool(args.use_lora_moe) and bool(args.use_auto_weighted_loss):
        awl = AutomaticWeightedLoss(num=3).to(device)  # 3 tasks: det, seg, cnt
        param_groups.append({"params": awl.parameters(), "lr": 0.01, "weight_decay": 0.0})
        print("[LoRA-MoE] AutomaticWeightedLoss enabled (3 learnable params)")

    # Learnable beta weights (optional)
    beta_params = None
    if bool(args.dynamic_loss_weight):
        beta_params = torch.nn.Parameter(torch.ones(3, device=device))
        param_groups.append({"params": [beta_params], "lr": float(args.lr), "weight_decay": 0.0})
        print("[train] Dynamic loss weighting enabled (beta1~3 via softplus)")

    optimizer = torch.optim.AdamW(param_groups)
    scaler = GradScaler(device.type, enabled=bool(args.amp))
    autocast_device = device.type if device.type in {"cuda", "cpu"} else "cuda"

    train_loaders = {"det": det_train_loader, "seg": seg_train_loader, "cnt": cnt_train_loader}
    val_loaders = {"det": det_val_loader, "seg": seg_val_loader, "cnt": cnt_val_loader}
    lengths = {"det": len(det_train_ds), "seg": len(seg_train_ds), "cnt": len(cnt_train_ds)}
    primary = choose_primary(lengths, args.primary_task)

    cyc = {k: infinite_loader(v) for k, v in train_loaders.items() if k != primary}

    best_metric = -math.inf
    best_path = save_dir / "best_combo.pt"
    best_state = None
    best_epoch = None

    def _current_loss_weights() -> tuple[float, float, float]:
        if beta_params is not None:
            b = F.softplus(beta_params.detach()).float().cpu().tolist()
            return float(b[0]), float(b[1]), float(b[2])
        return float(w_det), float(w_seg), float(w_cnt)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        det_loss_sum = 0.0
        seg_loss_sum = 0.0
        cnt_loss_sum = 0.0
        aux_loss_sum = 0.0  # For LoRA-MoE auxiliary losses

        for step, primary_batch in enumerate(train_loaders[primary], start=1):
            batches = {primary: primary_batch}
            for k in train_loaders.keys():
                if k != primary:
                    batches[k] = next(cyc[k])

            det_images, det_targets = _to_device_det(batches["det"], device)
            seg_imgs, seg_masks = _to_device_seg(batches["seg"], device)
            cnt_imgs, cnt_dens = _to_device_cnt(batches["cnt"], device)
            cnt_gt_counts = cnt_dens.flatten(2).sum(dim=2)

            optimizer.zero_grad(set_to_none=True)

            # NOTE: When using gradient checkpointing, some modules with side effects
            # (e.g., MoE aux-stat accumulation) may run again during backward-time
            # recomputation. Clear any leftover aux stats at step boundaries to avoid
            # carrying stale graph tensors into the next step.
            if bool(args.use_lora_moe) and bool(getattr(args, "grad_checkpointing", False)):
                with torch.no_grad():
                    _ = model.shared.get_aux_loss_and_clear()

            with torch.amp.autocast(autocast_device, enabled=bool(args.amp)):
                det_loss_dict = model.forward_det(det_images, det_targets)
                det_loss = sum(det_loss_dict.values())

                # LoRA-MoE mode: always use separate forwards (no fused seg+cnt)
                if args.use_lora_moe:
                    seg_logits = model.forward_seg(seg_imgs)
                    pred_dens, pred_counts = model.forward_cnt(
                        cnt_imgs,
                        cnt_backbone_grad_mult=float(args.cnt_backbone_grad_mult),
                    )
                else:
                    fuse_ok = (
                        bool(args.fuse_seg_cnt_backbone)
                        and bool(seg_train_backbone) == bool(cnt_train_backbone)
                        and (seg_imgs.shape[2:] == cnt_imgs.shape[2:])
                        and (seg_imgs.dtype == cnt_imgs.dtype)
                        and (seg_imgs.device == cnt_imgs.device)
                        and (seg_imgs.shape[2] % model.shared.patch_size[0] == 0)
                        and (seg_imgs.shape[3] % model.shared.patch_size[1] == 0)
                    )
                    if fuse_ok:
                        seg_logits, pred_dens, pred_counts = model.forward_seg_and_cnt(
                            seg_imgs,
                            cnt_imgs,
                            cnt_backbone_grad_mult=float(args.cnt_backbone_grad_mult),
                        )
                    else:
                        seg_logits = model.forward_seg(seg_imgs)
                        pred_dens, pred_counts = model.forward_cnt(
                            cnt_imgs,
                            cnt_backbone_grad_mult=float(args.cnt_backbone_grad_mult),
                        )

                seg_loss = F.cross_entropy(seg_logits, seg_masks)
                dens_loss = F.mse_loss(pred_dens, cnt_dens, reduction="sum") / cnt_imgs.size(0)
                cnt_l1 = F.l1_loss(pred_counts, cnt_gt_counts)
                cnt_loss = dens_loss + float(args.cnt_count_loss_weight) * cnt_l1
                
                # Compute total loss
                aux_loss = det_loss.new_tensor(0.0)
                if bool(args.use_lora_moe):
                    # MI loss is only computed for the shared expert pool.
                    _cv_loss, _switch_loss, _z_loss, mi_loss = model.shared.get_aux_loss_and_clear()
                    if bool(args.use_mi_shared):
                        aux_loss = float(args.moe_mi_loss_shared) * mi_loss

                if awl is not None:
                    # Use AutomaticWeightedLoss for det/seg/cnt weighting.
                    main_loss = awl([det_loss, seg_loss, cnt_loss])
                elif beta_params is not None:
                    # Learnable beta weights: det/seg/cnt = beta1~3.
                    beta = F.softplus(beta_params)
                    main_loss = beta[0] * det_loss + beta[1] * seg_loss + beta[2] * cnt_loss
                else:
                    # Fixed weights: det/seg/cnt = --loss-weights.
                    main_loss = float(w_det) * det_loss + float(w_seg) * seg_loss + float(w_cnt) * cnt_loss

                total = main_loss + aux_loss

            scaler.scale(total).backward()
            if float(args.grad_clip_norm) > 0:
                # Match counting single-task: unscale first, then clip, then step.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()

            if bool(args.use_lora_moe) and bool(getattr(args, "grad_checkpointing", False)):
                with torch.no_grad():
                    _ = model.shared.get_aux_loss_and_clear()

            det_loss_v = float(det_loss.detach().item())
            seg_loss_v = float(seg_loss.detach().item())
            cnt_loss_v = float(cnt_loss.detach().item())
            det_loss_sum += det_loss_v
            seg_loss_sum += seg_loss_v
            cnt_loss_sum += cnt_loss_v
            
            # Track auxiliary loss for LoRA-MoE mode
            if bool(args.use_lora_moe):
                aux_loss_sum += float(aux_loss.detach().item())

            total_loss += float(total.detach().item())
            steps += 1

            if args.log_interval and step % args.log_interval == 0:
                if bool(args.use_lora_moe):
                    if awl is not None:
                        params = awl.params.detach().cpu().numpy()
                        awl_text = f" | awl_params [{params[0]:.2f}, {params[1]:.2f}, {params[2]:.2f}]"
                    else:
                        awl_text = ""
                    if beta_params is not None:
                        b = F.softplus(beta_params.detach()).float().cpu().tolist()
                        beta_text = f" | beta [{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}]"
                    else:
                        beta_text = ""
                    print(
                        f"epoch {epoch}/{args.epochs} step {step} | "
                        f"loss {total_loss/max(steps,1):.4f} | "
                        f"det {det_loss_v:.4f} seg {seg_loss_v:.4f} cnt {cnt_loss_v:.4f} | "
                        f"aux {aux_loss_sum/max(steps,1):.4f}"
                        f"{awl_text}{beta_text}"
                    )
                else:
                    if beta_params is not None:
                        b = F.softplus(beta_params.detach()).float().cpu().tolist()
                        beta_text = f" | beta [{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}]"
                    else:
                        beta_text = ""
                    print(
                        f"epoch {epoch}/{args.epochs} step {step} | "
                        f"loss {total_loss/max(steps,1):.4f} | "
                        f"det {det_loss_v:.4f} seg {seg_loss_v:.4f} cnt {cnt_loss_v:.4f}"
                        f"{beta_text}"
                    )

            if args.max_train_steps and step >= args.max_train_steps:
                break

        avg_train = total_loss / max(steps, 1)
        avg_det = det_loss_sum / max(steps, 1)
        avg_seg = seg_loss_sum / max(steps, 1)
        avg_cnt = cnt_loss_sum / max(steps, 1)

        val_every = int(getattr(args, "val_every", 1) or 1)
        if val_every < 1:
            raise ValueError("--val-every must be >= 1")
        do_val = (val_every == 1) or (epoch % val_every == 0) or (epoch == args.epochs)

        if not do_val:
            print(f"epoch {epoch}/{args.epochs} | train {avg_train:.4f} | (skip val; --val-every={val_every})")
            continue

        val_det = _eval_det_loss(model, val_loaders["det"], device, amp=bool(args.amp), max_steps=args.max_val_steps)
        val_seg, val_seg_miou = _eval_seg_loss(
            model,
            val_loaders["seg"],
            device,
            amp=bool(args.amp),
            max_steps=args.max_val_steps,
            num_classes=int(args.seg_num_classes),
        )
        val_cnt, val_cnt_density, val_cnt_mae, val_cnt_total_mae = _eval_cnt_loss(
            model,
            val_loaders["cnt"],
            device,
            amp=bool(args.amp),
            max_steps=args.max_val_steps,
            count_loss_weight=float(args.cnt_count_loss_weight),
        )
        val_ap50, val_ap50_diag = _eval_det_ap50_fast(
            model,
            val_loaders["det"],
            device,
            num_classes=det_num_classes,
            score_thresh=0.0,
        )
        combo_metric = float(val_ap50) + float(val_seg_miou) + 1.0 / max(float(val_cnt_mae), 1e-8)

        metrics = {
            "train_loss": float(avg_train),
            "train_det_loss": float(avg_det),
            "train_seg_loss": float(avg_seg),
            "train_cnt_loss": float(avg_cnt),
            "val_det_loss": float(val_det),
            "val_seg_loss": float(val_seg),
            "val_seg_miou": float(val_seg_miou),
            "val_cnt_loss": float(val_cnt),
            "val_cnt_density_mse": float(val_cnt_density),
            "val_cnt_mae": float(val_cnt_mae),
            "val_cnt_total_mae": float(val_cnt_total_mae),
            "val_ap50": float(val_ap50),
            "det_ap50_num_gt": float(val_ap50_diag["det_ap50_num_gt"]),
            "det_ap50_num_pred": float(val_ap50_diag["det_ap50_num_pred"]),
            "det_ap50_num_tp": float(val_ap50_diag["det_ap50_num_tp"]),
            "selected_metric": float(combo_metric),
        }

        print(
            f"epoch {epoch}/{args.epochs} | train {avg_train:.4f} | "
            f"val det {val_det:.4f} seg {val_seg:.4f} miou {val_seg_miou:.4f} "
            f"cnt {val_cnt:.4f} dens {val_cnt_density:.6e} mae {val_cnt_mae:.4f} total_mae {val_cnt_total_mae:.4f} | "
            f"ap50 {val_ap50:.4f} (gt {int(val_ap50_diag['det_ap50_num_gt'])} "
            f"pred {int(val_ap50_diag['det_ap50_num_pred'])} tp {int(val_ap50_diag['det_ap50_num_tp'])}) | "
            f"combo {combo_metric:.6f}"
        )

        if combo_metric > best_metric:
            best_metric = float(combo_metric)
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            print(f"[ckpt] new best cached (combo {best_metric:.6f})")

    if best_state is not None:
        model.load_state_dict(best_state)
    save_multitask_checkpoint(
        str(best_path),
        model=model,
        optimizer=optimizer,
        epoch=best_epoch or args.epochs,
        best_by="combo",
        metrics={"best_metric": float(best_metric)},
        loss_weights=(w_det, w_seg, w_cnt),
    )
    print(f"[ckpt] saved best -> {best_path} (combo {best_metric:.6f})")


if __name__ == "__main__":
    main()

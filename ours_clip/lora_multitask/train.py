from __future__ import annotations

import argparse
import builtins
import math
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import GradScaler
from copy import deepcopy
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .datasets import build_cnt_loaders, build_det_loaders, build_seg_loaders
from .models import MultiTaskModel, SharedCLIPVisionBackbone
from .utils import choose_primary, infinite_loader, parse_loss_weights, save_multitask_checkpoint
from object_detection.dataset import collate_fn
from segmentation.utils import per_class_iou_from_confusion, update_confusion_matrix


def parse_args():
    p = argparse.ArgumentParser(description="Multi-task training (det/seg/count) with shared CLIP vision backbone")

    # Enable/disable tasks (all enabled by default to preserve existing behavior)
    p.add_argument("--enable-det", action="store_true", help="enable detection task")
    p.add_argument("--disable-det", dest="enable_det", action="store_false", help="disable detection task")
    p.set_defaults(enable_det=True)
    p.add_argument("--enable-seg", action="store_true", help="enable segmentation task")
    p.add_argument("--disable-seg", dest="enable_seg", action="store_false", help="disable segmentation task")
    p.set_defaults(enable_seg=True)
    p.add_argument("--enable-cnt", action="store_true", help="enable counting task")
    p.add_argument("--disable-cnt", dest="enable_cnt", action="store_false", help="disable counting task")
    p.set_defaults(enable_cnt=True)

    # Backbone
    p.add_argument("--model-name", type=str, default="openai/clip-vit-large-patch14")
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--backbone-checkpoint", type=str, default=None)
    p.add_argument(
        "--lora",
        action="store_true",
        help="Enable LoRA finetuning on ViT FFN; freezes backbone weights and trains only LoRA + task heads.",
    )
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-lr", type=float, default=None, help="LoRA learning rate (default: --lr)")
    p.add_argument("--lora-weight-decay", type=float, default=0.0)
    p.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="legacy: force unfreeze shared backbone for all tasks (overrides per-task freeze flags)",
    )

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
    p.add_argument("--det-data-root", type=str, required=False)
    p.add_argument("--det-train-ann", type=str, default=None)
    p.add_argument("--det-val-ann", type=str, default=None)
    p.add_argument("--det-train-img-dir", type=str, default=None)
    p.add_argument("--det-val-img-dir", type=str, default=None)
    p.add_argument("--det-num-classes", type=int, default=None, help="foreground class count (auto if None)")

    # Seg dataset
    p.add_argument("--seg-train-dir", type=str, required=False)
    p.add_argument("--seg-val-dir", type=str, required=False)
    p.add_argument("--seg-num-classes", type=int, default=11)

    # Count dataset
    p.add_argument("--cnt-data-root", type=str, required=False)
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
    # NEW: det 单任务时用 COCO AP50 选 best
    p.add_argument(
        "--det-use-coco-eval-for-best",
        action="store_true",
        help="single-task det: use COCO eval AP@0.50 for best selection (higher is better)",
    )
    p.add_argument("--save-dir", type=str, default="runs/multitask")
    p.add_argument("--val-interval", type=int, default=1, help="run validation every N epochs (0 disables)")
    p.add_argument(
        "--val-start-epoch",
        type=int,
        default=1,
        help="first epoch eligible for validation/best selection (default: 1)",
    )
    p.add_argument("--save-epochs", type=str, default="60,100", help="extra checkpoint epochs, e.g. '60,100'")
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
) -> float:
    model.detector.eval()
    preds_by_cls = {c: [] for c in range(1, num_classes + 1)}
    gts_by_cls = {c: {} for c in range(1, num_classes + 1)}

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

            pred_boxes = out["boxes"].detach().cpu().numpy()
            pred_labels = out["labels"].detach().cpu().numpy().astype(int)
            pred_scores = out["scores"].detach().cpu().numpy()

            keep = pred_scores >= score_thresh
            for box, cls, score in zip(pred_boxes[keep], pred_labels[keep], pred_scores[keep]):
                preds_by_cls[cls].append((img_id, float(score), box))

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
            else:
                fp[i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / max(num_gt, 1)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        ap_list.append(_ap_from_pr(rec, prec))

    return float(np.mean(ap_list)) if ap_list else 0.0


def _extract_metrics_path(output: str) -> Optional[str]:
    for line in output.splitlines():
        s = line.strip()
        if s.startswith("Saved metrics to "):
            return s[len("Saved metrics to ") :].strip()
    return None


def _find_ap50(metrics: Any) -> Optional[float]:
    if isinstance(metrics, dict):
        # direct keys
        for k, v in metrics.items():
            key = str(k).lower()
            if key in {"ap50", "ap@0.50", "ap@0.5", "ap50_bbox", "bbox/ap50"} and isinstance(v, (int, float)):
                return float(v)
        # nested dicts
        for v in metrics.values():
            out = _find_ap50(v)
            if out is not None:
                return out
    elif isinstance(metrics, list):
        for v in metrics:
            out = _find_ap50(v)
            if out is not None:
                return out
    return None


def _eval_det_ap50_via_coco(
    *,
    checkpoint: Path,
    args,
    repo_root: Path,
    stats_dir: Path,
) -> float:
    cmd = [sys.executable, "object_detection/eval.py", "--checkpoint", str(checkpoint), "--stats-dir", str(stats_dir), "--use-coco-eval"]
    if args.det_data_root:
        cmd += ["--data-root", args.det_data_root]
    if args.det_val_ann:
        cmd += ["--ann-file", args.det_val_ann]
    if args.det_val_img_dir:
        cmd += ["--img-dir", args.det_val_img_dir]
    if args.det_num_classes is not None:
        cmd += ["--num-classes", str(args.det_num_classes)]
    if args.image_size is not None:
        cmd += ["--image-size", str(args.image_size)]
    if args.model_name is not None:
        cmd += ["--model-name", args.model_name]
    if args.device is not None:
        cmd += ["--device", args.device]

    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    output_lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        output_lines.append(line)
    proc.wait()
    out = "".join(output_lines)

    metrics_path = _extract_metrics_path(out)
    if not metrics_path or not Path(metrics_path).is_file():
        raise RuntimeError("COCO eval did not produce metrics file; check object_detection/eval.py output.")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    ap50 = _find_ap50(metrics)
    if ap50 is None:
        raise RuntimeError(f"Could not find AP50 in metrics: {metrics_path}")
    return float(ap50)


def main():
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if use_ddp and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if use_ddp and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)

    is_main_process = (not use_ddp) or rank == 0
    if use_ddp and not is_main_process:
        builtins.print = lambda *args, **kwargs: None

    def _ddp_barrier() -> None:
        if not use_ddp or not dist.is_initialized():
            return
        if dist.get_backend() == "nccl" and device.type == "cuda":
            barrier_device = int(device.index) if device.index is not None else int(local_rank)
            dist.barrier(device_ids=[barrier_device])
        else:
            dist.barrier()

    save_dir = Path(args.save_dir)
    if is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
    _ddp_barrier()

    if args.enable_det and not args.det_data_root:
        raise SystemExit("--det-data-root is required when detection is enabled.")
    if args.enable_seg and (not args.seg_train_dir or not args.seg_val_dir):
        raise SystemExit("--seg-train-dir and --seg-val-dir are required when segmentation is enabled.")
    if args.enable_cnt and not args.cnt_data_root:
        raise SystemExit("--cnt-data-root is required when counting is enabled.")
    if args.det_use_coco_eval_for_best and (not args.det_val_ann or not args.det_val_img_dir):
        raise SystemExit("--det-val-ann and --det-val-img-dir are required for COCO AP50 best selection.")

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
    if args.lora:
        # LoRA mode ignores per-task freeze/unfreeze flags; only LoRA params are trainable anyway.
        det_train_backbone = True
        seg_train_backbone = True
        cnt_train_backbone = True
    elif args.unfreeze_backbone:
        det_train_backbone = True
        seg_train_backbone = True
        cnt_train_backbone = True

    det_train_ds = det_val_ds = det_train_loader = det_val_loader = None
    seg_train_ds = seg_val_ds = seg_train_loader = seg_val_loader = None
    cnt_train_ds = cnt_val_ds = cnt_train_loader = cnt_val_loader = None

    if args.enable_det:
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
    if args.enable_seg:
        seg_train_ds, seg_val_ds, seg_train_loader, seg_val_loader = build_seg_loaders(
            train_dir=args.seg_train_dir,
            val_dir=args.seg_val_dir,
            image_size=args.image_size,
            batch_size=args.seg_batch_size,
            num_workers=args.num_workers,
        )
    if args.enable_cnt:
        cnt_train_ds, cnt_val_ds, cnt_train_loader, cnt_val_loader = build_cnt_loaders(
            data_root=args.cnt_data_root,
            train_dir=args.cnt_train_dir,
            val_dir=args.cnt_val_dir,
            image_size=args.image_size,
            num_classes=args.cnt_num_classes,
            keep_aspect=bool(args.cnt_keep_aspect),
            batch_size=args.cnt_batch_size,
            num_workers=args.num_workers,
        )

    det_train_sampler = None
    seg_train_sampler = None
    cnt_train_sampler = None
    if use_ddp:
        if args.enable_det:
            det_train_sampler = DistributedSampler(det_train_ds, num_replicas=world_size, rank=rank, shuffle=True)
            det_train_loader = DataLoader(
                det_train_ds,
                batch_size=args.det_batch_size,
                sampler=det_train_sampler,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )
        if args.enable_seg:
            seg_train_sampler = DistributedSampler(seg_train_ds, num_replicas=world_size, rank=rank, shuffle=True)
            seg_train_loader = DataLoader(
                seg_train_ds,
                batch_size=args.seg_batch_size,
                sampler=seg_train_sampler,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
        if args.enable_cnt:
            cnt_kwargs = dict(
                batch_size=args.cnt_batch_size,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            if int(args.num_workers) > 0:
                cnt_kwargs["persistent_workers"] = True
                cnt_kwargs["prefetch_factor"] = 1

            cnt_train_sampler = DistributedSampler(
                cnt_train_ds,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                drop_last=True,
            )
            cnt_train_loader = DataLoader(
                cnt_train_ds,
                sampler=cnt_train_sampler,
                shuffle=False,
                drop_last=True,
                **cnt_kwargs,
            )

    if args.enable_det:
        det_num_classes = int(args.det_num_classes) if args.det_num_classes else int(det_train_ds.num_classes)
    else:
        det_num_classes = int(args.det_num_classes) if args.det_num_classes is not None else 1

    shared = SharedCLIPVisionBackbone(
        model_name=args.model_name,
        image_size=args.image_size,
        checkpoint_path=args.backbone_checkpoint,
        use_lora=bool(args.lora),
        lora_rank=int(args.lora_rank),
        lora_alpha=float(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
    )
    if not args.lora:
        any_train_backbone = det_train_backbone or seg_train_backbone or cnt_train_backbone
        for p in shared.backbone.parameters():
            p.requires_grad = bool(any_train_backbone)

    raw_model = MultiTaskModel(
        shared=shared,
        det_num_classes=det_num_classes,
        seg_num_classes=args.seg_num_classes,
        cnt_num_classes=args.cnt_num_classes,
        image_size=args.image_size,
        det_train_backbone=det_train_backbone,
        seg_train_backbone=seg_train_backbone,
        cnt_train_backbone=cnt_train_backbone,
    ).to(device)

    shared_params = list(raw_model.shared.parameters())
    shared_param_ids = {id(p) for p in shared_params}
    shared_backbone_params = list(raw_model.shared.backbone.parameters())
    shared_backbone_ids = {id(p) for p in shared_backbone_params}

    if not args.enable_det:
        for p in raw_model.detector.parameters():
            if id(p) not in shared_param_ids:
                p.requires_grad_(False)
    if not args.enable_seg:
        raw_model.seg_head.requires_grad_(False)
    if not args.enable_cnt:
        raw_model.cnt_head.requires_grad_(False)

    backbone_params = [p for p in shared_backbone_params if p.requires_grad]
    det_head_params = [p for p in raw_model.detector.parameters() if p.requires_grad and id(p) not in shared_param_ids]
    seg_params = [p for p in raw_model.seg_head.parameters() if p.requires_grad]
    cnt_head_params = [p for p in raw_model.cnt_head.parameters() if p.requires_grad]

    param_groups = []
    if backbone_params:
        if args.lora:
            lora_lr = float(args.lora_lr) if args.lora_lr is not None else float(args.lr)
            param_groups.append({"params": backbone_params, "lr": lora_lr, "weight_decay": float(args.lora_weight_decay)})
        else:
            param_groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": backbone_wd})
    if det_head_params:
        param_groups.append({"params": det_head_params, "lr": det_lr, "weight_decay": det_wd})
    if seg_params:
        param_groups.append({"params": seg_params, "lr": seg_lr, "weight_decay": seg_wd})
    if cnt_head_params:
        param_groups.append({"params": cnt_head_params, "lr": cnt_lr, "weight_decay": cnt_wd})
    if not param_groups:
        raise RuntimeError("No trainable parameters.")

    trainable_total = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    print(f"[params] trainable_total={int(trainable_total)}")

    optimizer = torch.optim.AdamW(param_groups)
    scaler = GradScaler(device.type, enabled=bool(args.amp))
    autocast_device = device.type if device.type in {"cuda", "cpu"} else "cuda"
    trainable_params = [p for p in raw_model.parameters() if p.requires_grad]

    if use_ddp:
        ddp_kwargs = {
            "broadcast_buffers": False,
            "find_unused_parameters": False,
        }
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [local_rank]
            ddp_kwargs["output_device"] = local_rank
        model = DDP(raw_model, **ddp_kwargs)
    else:
        model = raw_model

    train_loaders = {}
    val_loaders = {}
    lengths = {}
    if args.enable_det:
        train_loaders["det"] = det_train_loader
        val_loaders["det"] = det_val_loader
        lengths["det"] = len(det_train_ds)
    if args.enable_seg:
        train_loaders["seg"] = seg_train_loader
        val_loaders["seg"] = seg_val_loader
        lengths["seg"] = len(seg_train_ds)
    if args.enable_cnt:
        train_loaders["cnt"] = cnt_train_loader
        val_loaders["cnt"] = cnt_val_loader
        lengths["cnt"] = len(cnt_train_ds)
    if not train_loaders:
        raise RuntimeError("No enabled tasks. Use --enable-det/--enable-seg/--enable-cnt.")
    primary = choose_primary(lengths, args.primary_task)

    cyc = {k: infinite_loader(v) for k, v in train_loaders.items() if k != primary}

    def _parse_save_epochs(text: str) -> list[int]:
        s = (text or "").strip()
        if not s:
            return []
        parts = [p.strip() for p in s.replace(";", ",").replace(" ", ",").split(",") if p.strip()]
        epochs = []
        for p in parts:
            try:
                epochs.append(int(p))
            except ValueError:
                raise ValueError(f"Invalid --save-epochs entry: {p}")
        return sorted(set(e for e in epochs if e > 0))

    save_epochs = set(_parse_save_epochs(args.save_epochs))
    val_start_epoch = max(int(args.val_start_epoch), 1)

    # ===== 单任务 best 指标选择 =====
    enabled_tasks = [t for t in ("det", "seg", "cnt") if getattr(args, f"enable_{t}")]
    single_task = (len(enabled_tasks) == 1)
    single_det = single_task and enabled_tasks[0] == "det"
    single_seg = single_task and enabled_tasks[0] == "seg"
    single_cnt = single_task and enabled_tasks[0] == "cnt"

    if single_det:
        selected_metric_name = "det_ap50"
    elif single_seg:
        selected_metric_name = "seg_miou"
    elif single_cnt:
        selected_metric_name = "cnt_mae"
    else:
        selected_metric_name = "combo"

    best_metric = -math.inf
    best_metric_report = None
    best_path = save_dir / "best_combo.pt"
    best_state = None
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        if use_ddp:
            if det_train_sampler is not None:
                det_train_sampler.set_epoch(epoch)
            if seg_train_sampler is not None:
                seg_train_sampler.set_epoch(epoch)
            if cnt_train_sampler is not None:
                cnt_train_sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        steps = 0
        det_loss_sum = 0.0
        seg_loss_sum = 0.0
        cnt_loss_sum = 0.0

        for step, primary_batch in enumerate(train_loaders[primary], start=1):
            batches = {primary: primary_batch}
            for k in train_loaders.keys():
                if k != primary:
                    batches[k] = next(cyc[k])

            det_images = det_targets = None
            seg_imgs = seg_masks = None
            cnt_imgs = cnt_dens = None
            cnt_gt_counts = None
            if args.enable_det:
                det_images, det_targets = _to_device_det(batches["det"], device)
            if args.enable_seg:
                seg_imgs, seg_masks = _to_device_seg(batches["seg"], device)
            if args.enable_cnt:
                cnt_imgs, cnt_dens = _to_device_cnt(batches["cnt"], device)
                cnt_gt_counts = cnt_dens.flatten(2).sum(dim=2)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(autocast_device, enabled=bool(args.amp)):
                det_loss = torch.tensor(0.0, device=device)
                seg_loss = torch.tensor(0.0, device=device)
                cnt_loss = torch.tensor(0.0, device=device)

                if args.enable_det:
                    det_loss_dict = model("det", det_images, det_targets)
                    det_loss = sum(det_loss_dict.values())

                if args.enable_seg and args.enable_cnt:
                    fuse_ok = (
                        bool(args.fuse_seg_cnt_backbone)
                        and bool(seg_train_backbone) == bool(cnt_train_backbone)
                        and (seg_imgs.shape[2:] == cnt_imgs.shape[2:])
                        and (seg_imgs.dtype == cnt_imgs.dtype)
                        and (seg_imgs.device == cnt_imgs.device)
                        and (seg_imgs.shape[2] % raw_model.shared.patch_size[0] == 0)
                        and (seg_imgs.shape[3] % raw_model.shared.patch_size[1] == 0)
                    )
                    if fuse_ok:
                        seg_logits, pred_dens, pred_counts = model(
                            "seg_cnt",
                            seg_imgs,
                            cnt_imgs,
                            cnt_backbone_grad_mult=float(args.cnt_backbone_grad_mult),
                        )
                    else:
                        seg_logits = model("seg", seg_imgs)
                        pred_dens, pred_counts = model(
                            "cnt",
                            cnt_imgs,
                            cnt_backbone_grad_mult=float(args.cnt_backbone_grad_mult),
                        )
                    seg_loss = F.cross_entropy(seg_logits, seg_masks)
                    dens_loss = F.mse_loss(pred_dens, cnt_dens, reduction="sum") / cnt_imgs.size(0)
                    cnt_l1 = F.l1_loss(pred_counts, cnt_gt_counts)
                    cnt_loss = dens_loss + float(args.cnt_count_loss_weight) * cnt_l1
                elif args.enable_seg:
                    seg_logits = model("seg", seg_imgs)
                    seg_loss = F.cross_entropy(seg_logits, seg_masks)
                elif args.enable_cnt:
                    pred_dens, pred_counts = model(
                        "cnt",
                        cnt_imgs,
                        cnt_backbone_grad_mult=float(args.cnt_backbone_grad_mult),
                    )
                    dens_loss = F.mse_loss(pred_dens, cnt_dens, reduction="sum") / cnt_imgs.size(0)
                    cnt_l1 = F.l1_loss(pred_counts, cnt_gt_counts)
                    cnt_loss = dens_loss + float(args.cnt_count_loss_weight) * cnt_l1

                total = float(w_det) * det_loss + float(w_seg) * seg_loss + float(w_cnt) * cnt_loss

            local_non_finite = not bool(torch.isfinite(total).item())
            if use_ddp:
                flag = torch.tensor(1 if local_non_finite else 0, device=device, dtype=torch.int32)
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                any_non_finite = bool(flag.item() > 0)
            else:
                any_non_finite = local_non_finite

            if any_non_finite:
                if is_main_process:
                    det_v = float(det_loss.detach().item())
                    seg_v = float(seg_loss.detach().item())
                    cnt_v = float(cnt_loss.detach().item())
                    total_v = float(total.detach().item())
                    print(
                        f"[warn] non-finite loss at epoch {epoch} step {step}; skip optimizer step "
                        f"(total={total_v:.6g}, det={det_v:.6g}, seg={seg_v:.6g}, cnt={cnt_v:.6g})"
                    )
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(total).backward()
            if use_ddp:
                scaler.unscale_(optimizer)
                if float(args.grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(args.grad_clip_norm))
            elif float(args.grad_clip_norm) > 0:
                # Match counting single-task: unscale first, then clip, then step.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(args.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()

            det_loss_v = float(det_loss.detach().item())
            seg_loss_v = float(seg_loss.detach().item())
            cnt_loss_v = float(cnt_loss.detach().item())
            det_loss_sum += det_loss_v
            seg_loss_sum += seg_loss_v
            cnt_loss_sum += cnt_loss_v

            total_loss += float(total.detach().item())
            steps += 1

            if args.log_interval and step % args.log_interval == 0:
                print(
                    f"epoch {epoch}/{args.epochs} step {step} | "
                    f"loss {total_loss/max(steps,1):.4f} | "
                    f"det {det_loss_v:.4f} seg {seg_loss_v:.4f} cnt {cnt_loss_v:.4f}"
                )

            if args.max_train_steps and step >= args.max_train_steps:
                break

        avg_train = total_loss / max(steps, 1)
        avg_det = det_loss_sum / max(steps, 1)
        avg_seg = seg_loss_sum / max(steps, 1)
        avg_cnt = cnt_loss_sum / max(steps, 1)

        # Validation
        val_det = 0.0
        val_seg = 0.0
        val_seg_miou = 0.0
        val_cnt = 0.0
        val_cnt_density = 0.0
        val_cnt_mae = 0.0
        val_cnt_total_mae = 0.0
        val_det_ap50 = 0.0

        do_eval = (
            int(args.val_interval) > 0
            and epoch >= val_start_epoch
            and (epoch % int(args.val_interval) == 0)
        )
        if do_eval and is_main_process:
            if args.enable_det:
                val_det = _eval_det_loss(raw_model, val_loaders["det"], device, amp=bool(args.amp), max_steps=args.max_val_steps)
            if args.enable_seg:
                val_seg, val_seg_miou = _eval_seg_loss(
                    raw_model,
                    val_loaders["seg"],
                    device,
                    amp=bool(args.amp),
                    max_steps=args.max_val_steps,
                    num_classes=int(args.seg_num_classes),
                )
            if args.enable_cnt:
                val_cnt, val_cnt_density, val_cnt_mae, val_cnt_total_mae = _eval_cnt_loss(
                    raw_model,
                    val_loaders["cnt"],
                    device,
                    amp=bool(args.amp),
                    max_steps=args.max_val_steps,
                    count_loss_weight=float(args.cnt_count_loss_weight),
                )
            if args.enable_det:
                val_det_ap50 = _eval_det_ap50_fast(
                    raw_model,
                    val_loaders["det"],
                    device,
                    num_classes=int(det_num_classes),
                    score_thresh=0.0,
                )

        _ddp_barrier()

        val_total = float(w_det) * val_det + float(w_seg) * val_seg + float(w_cnt) * val_cnt
        combo_metric = float(val_det_ap50) + float(val_seg_miou) + 1.0 / max(float(val_cnt_mae), 1e-8)

        # 单任务只按该任务指标选 best：det->AP50 最大，seg->mIoU 最大，cnt->MAE 最小。
        if single_det:
            selected_metric = float(val_det_ap50)
            selected_metric_report = float(val_det_ap50)
        elif single_seg:
            selected_metric = float(val_seg_miou)
            selected_metric_report = float(val_seg_miou)
        elif single_cnt:
            selected_metric = -float(val_cnt_mae)
            selected_metric_report = float(val_cnt_mae)
        else:
            selected_metric = float(combo_metric)
            selected_metric_report = float(combo_metric)

        metrics: Dict[str, float] = {
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
            "val_total_loss": float(val_total),
            "val_det_ap50": float(val_det_ap50),
            "selected_metric": float(selected_metric_report),
        }

        if is_main_process:
            if single_cnt:
                selected_text = f"selected cnt_mae(min) {selected_metric_report:.6f}"
            else:
                selected_text = f"selected {selected_metric_name} {selected_metric_report:.6f}"
            print(
                f"epoch {epoch}/{args.epochs} | train {avg_train:.4f} | "
                f"val det {val_det:.4f} seg {val_seg:.4f} miou {val_seg_miou:.4f} "
                f"cnt {val_cnt:.4f} dens {val_cnt_density:.6e} mae {val_cnt_mae:.4f} total_mae {val_cnt_total_mae:.4f} | "
                f"ap50 {val_det_ap50:.4f} | {selected_text}"
            )
        if do_eval and is_main_process and (selected_metric > best_metric):
            best_metric = float(selected_metric)
            best_metric_report = float(selected_metric_report)
            best_state = deepcopy(raw_model.state_dict())
            best_epoch = epoch
            print(f"[ckpt] new best cached ({selected_metric_name} {selected_metric_report:.6f})")

    # Save the best validation checkpoint once at the end of training.
    if is_main_process and best_state is not None:
        pass

    # 训练结束后，仅保存一次“验证最优”
    if is_main_process and best_state is not None:
        best_metric_to_save = float(best_metric_report) if best_metric_report is not None else float(best_metric)
        raw_model.load_state_dict(best_state)
        save_multitask_checkpoint(
            str(best_path),
            model=raw_model,
            optimizer=optimizer,
            epoch=best_epoch or args.epochs,
            best_by=selected_metric_name,
            metrics={"best_metric": float(best_metric_to_save)},
            loss_weights=(w_det, w_seg, w_cnt),
            config={
                "use_lora": bool(args.lora),
                "lora_rank": int(args.lora_rank),
                "lora_alpha": float(args.lora_alpha),
                "lora_dropout": float(args.lora_dropout),
            },
        )
        print(f"[ckpt] saved best -> {best_path} ({selected_metric_name} {best_metric_to_save:.6f})")

    if use_ddp and dist.is_initialized():
        _ddp_barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import builtins
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from object_detection.dataset import collate_fn
from segmentation.utils import per_class_iou_from_confusion, update_confusion_matrix

from .datasets import build_cnt_loaders, build_det_loaders, build_seg_loaders
from .models import MultiTaskModel, SharedDinoV3Backbone
from .utils import choose_primary, infinite_loader, parse_loss_weights, save_multitask_checkpoint


def parse_args():
    p = argparse.ArgumentParser(description="Single-stage multitask training with 113_test backbone + shared-only LoRA-MoE + MI")
    p.add_argument("--model-name", type=str, default="dinov3_vitl16")
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--backbone-checkpoint", type=str, default=None)

    use_moe = p.add_mutually_exclusive_group()
    use_moe.add_argument("--use-lora-moe", dest="use_lora_moe", action="store_true")
    use_moe.add_argument("--no-use-lora-moe", dest="use_lora_moe", action="store_false")
    p.set_defaults(use_lora_moe=True)

    gc = p.add_mutually_exclusive_group()
    gc.add_argument("--grad-checkpointing", dest="grad_checkpointing", action="store_true")
    gc.add_argument("--no-grad-checkpointing", dest="grad_checkpointing", action="store_false")
    p.set_defaults(grad_checkpointing=True)

    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--num-experts-private", type=int, default=0)
    p.add_argument("--num-experts-shared", type=int, default=6)
    p.add_argument("--moe-k-private", type=int, default=0)
    p.add_argument("--moe-k-shared", type=int, default=2)
    p.add_argument("--det-out-channels", type=int, default=256)

    p.add_argument("--det-data-root", type=str, required=True)
    p.add_argument("--det-train-ann", type=str, default=None)
    p.add_argument("--det-val-ann", type=str, default=None)
    p.add_argument("--det-train-img-dir", type=str, default=None)
    p.add_argument("--det-val-img-dir", type=str, default=None)
    p.add_argument("--det-num-classes", type=int, default=None)
    p.add_argument("--seg-train-dir", type=str, required=True)
    p.add_argument("--seg-val-dir", type=str, required=True)
    p.add_argument("--seg-num-classes", type=int, default=11)
    p.add_argument("--cnt-data-root", type=str, required=True)
    p.add_argument("--cnt-train-dir", type=str, default=None)
    p.add_argument("--cnt-val-dir", type=str, default=None)
    p.add_argument("--cnt-num-classes", type=int, default=8)
    p.add_argument("--cnt-count-loss-weight", type=float, default=1.0)

    aspect = p.add_mutually_exclusive_group()
    aspect.add_argument("--cnt-keep-aspect", dest="cnt_keep_aspect", action="store_true")
    aspect.add_argument("--cnt-no-keep-aspect", dest="cnt_keep_aspect", action="store_false")
    p.set_defaults(cnt_keep_aspect=True)

    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--val-every", type=int, default=1)
    p.add_argument("--val-last-n-epochs", type=int, default=50)
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument("--det-batch-size", type=int, default=2)
    p.add_argument("--seg-batch-size", type=int, default=2)
    p.add_argument("--cnt-batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--det-lr", type=float, default=None)
    p.add_argument("--seg-lr", type=float, default=None)
    p.add_argument("--cnt-lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--det-weight-decay", type=float, default=None)
    p.add_argument("--seg-weight-decay", type=float, default=None)
    p.add_argument("--cnt-weight-decay", type=float, default=None)
    p.add_argument("--loss-weights", type=str, default="15:8:1")
    p.add_argument("--mi-loss-weight", type=float, default=0.005)
    p.add_argument("--primary-task", type=str, default=None)
    p.add_argument("--save-dir", type=str, default="runs/mod_squad_plaintrain")
    p.add_argument("--save-epochs", type=str, default="")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad-clip-norm", type=float, default=100.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-steps", type=int, default=0)
    p.add_argument("--max-val-steps", type=int, default=0)
    p.add_argument("--det-ap-score-thr", type=float, default=0.0)
    p.add_argument("--cnt-backbone-grad-mult", type=float, default=1.0)
    p.add_argument("--ddp-broadcast-buffers", action="store_true")

    # Legacy launcher compatibility.
    p.add_argument("--unfreeze-backbone", action="store_true")
    p.add_argument("--backbone-lr", type=float, default=None)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--backbone-weight-decay", type=float, default=None)
    p.add_argument("--det-unfreeze-backbone", action="store_true")
    p.add_argument("--seg-full-finetune", action="store_true")
    p.add_argument("--cnt-full-finetune", action="store_true")
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
        epoch = int(part)
        if epoch <= 0:
            raise ValueError(f"--save-epochs expects positive integers, got {epoch}")
        out.add(epoch)
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
    return imgs.to(device, non_blocking=True).float(), dens.to(device, non_blocking=True).float()


@torch.no_grad()
def _count_diag_stats(
    pred_dens: torch.Tensor,
    gt_dens: torch.Tensor,
    pred_counts: torch.Tensor,
    gt_counts: torch.Tensor,
) -> Dict[str, float]:
    pred_dens_f = pred_dens.detach().float()
    gt_dens_f = gt_dens.detach().float()
    pred_counts_f = pred_counts.detach().float()
    gt_counts_f = gt_counts.detach().float()
    pred_dens_mean = float(pred_dens_f.mean().item())
    gt_dens_mean = float(gt_dens_f.mean().item())
    pred_count_mean = float(pred_counts_f.mean().item())
    gt_count_mean = float(gt_counts_f.mean().item())
    pred_total_mean = float(pred_counts_f.sum(dim=1).mean().item())
    gt_total_mean = float(gt_counts_f.sum(dim=1).mean().item())
    count_mae = float((pred_counts_f - gt_counts_f).abs().mean().item())
    pixels = int(pred_dens_f.shape[-2] * pred_dens_f.shape[-1])
    eps = 1e-12
    return {
        "pred_dens_mean": pred_dens_mean,
        "gt_dens_mean": gt_dens_mean,
        "dens_ratio": pred_dens_mean / max(gt_dens_mean, eps),
        "pred_count_mean": pred_count_mean,
        "gt_count_mean": gt_count_mean,
        "count_ratio": pred_count_mean / max(gt_count_mean, eps),
        "count_mae": count_mae,
        "pred_total_mean": pred_total_mean,
        "gt_total_mean": gt_total_mean,
        "pixels": float(pixels),
    }


def _format_count_diag(stats: Dict[str, float]) -> str:
    return (
        f"dens(mean {stats['pred_dens_mean']:.6e}/{stats['gt_dens_mean']:.6e}, ratio {stats['dens_ratio']:.3e}) | "
        f"count(mean {stats['pred_count_mean']:.3f}/{stats['gt_count_mean']:.3f}, ratio {stats['count_ratio']:.3e}, "
        f"mae {stats['count_mae']:.3f}, total {stats['pred_total_mean']:.3f}/{stats['gt_total_mean']:.3f}) | "
        f"pixels {int(stats['pixels'])}"
    )


@torch.no_grad()
def _eval_det_loss(model: MultiTaskModel, loader, device: torch.device, *, amp: bool, max_steps: int) -> float:
    model.train()
    total = 0.0
    samples = 0
    steps = 0
    autocast_device = device.type if device.type in {"cuda", "cpu"} else "cuda"
    for images, targets in loader:
        images, targets = _to_device_det((images, targets), device)
        with torch.amp.autocast(autocast_device, enabled=amp):
            loss_dict = model("det", images, targets)
            loss = sum(loss_dict.values())
        bsz = len(images)
        total += float(loss.item()) * bsz
        samples += bsz
        steps += 1
        if max_steps and steps >= max_steps:
            break
    if dist.is_initialized():
        t = torch.tensor([total, float(samples)], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total = float(t[0].item())
        samples = int(t[1].item())
    return total / max(samples, 1)


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
            logits = model("seg", imgs)
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
    if dist.is_initialized():
        t = torch.tensor([total, float(samples)], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total = float(t[0].item())
        samples = int(t[1].item())
        conf_dev = conf.to(device=device)
        dist.all_reduce(conf_dev, op=dist.ReduceOp.SUM)
        conf = conf_dev.cpu()
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
            pred_dens, pred_counts = model("cnt", imgs)
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
    if dist.is_initialized():
        t = torch.tensor(
            [total, total_density, total_count_mae, total_total_mae, float(samples)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total = float(t[0].item())
        total_density = float(t[1].item())
        total_count_mae = float(t[2].item())
        total_total_mae = float(t[3].item())
        samples = int(t[4].item())
    denom = max(samples, 1)
    return total / denom, total_density / denom, total_count_mae / denom, total_total_mae / denom


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
    for thr in np.linspace(0, 1, 101):
        p = prec[rec >= thr].max() if np.any(rec >= thr) else 0.0
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
    model.eval()
    preds_by_cls = {c: [] for c in range(1, num_classes + 1)}
    gts_by_cls = {c: {} for c in range(1, num_classes + 1)}
    dist_rank = dist.get_rank() if dist.is_initialized() else 0
    img_counter = 0
    for images, targets in loader:
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model("det", images)
        for out, tgt in zip(outputs, targets):
            default_img_id = dist_rank * 10_000_000_000 + img_counter
            img_id = int(tgt.get("image_id", torch.tensor([default_img_id])).item())
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
    if dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, (preds_by_cls, gts_by_cls))
        merged_preds = {c: [] for c in range(1, num_classes + 1)}
        merged_gts = {c: {} for c in range(1, num_classes + 1)}
        for rank_preds, rank_gts in gathered:
            for cls in range(1, num_classes + 1):
                merged_preds[cls].extend(rank_preds.get(cls, []))
                for img_id, boxes in rank_gts.get(cls, {}).items():
                    merged_gts[cls].setdefault(img_id, []).extend(boxes)
        preds_by_cls = merged_preds
        gts_by_cls = merged_gts
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
        for i, (img_id, _score, pbox) in enumerate(preds):
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


def _all_trainable_params(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def _build_model(args, det_num_classes: int, device: torch.device) -> MultiTaskModel:
    num_experts_private = int(args.num_experts_private)
    moe_k_private = int(args.moe_k_private)
    if num_experts_private != 0 or moe_k_private != 0:
        print("[warn] shared-only experiment: forcing num_experts_private=0 and moe_k_private=0")
        num_experts_private = 0
        moe_k_private = 0

    shared = SharedDinoV3Backbone(
        model_name=args.model_name,
        image_size=args.image_size,
        checkpoint_path=args.backbone_checkpoint,
        use_lora_moe=bool(args.use_lora_moe),
        task_num=3,
        lora_rank=int(args.lora_rank),
        num_experts_private=int(num_experts_private),
        num_experts_shared=int(args.num_experts_shared),
        moe_k_private=int(moe_k_private),
        moe_k_shared=int(args.moe_k_shared),
        grad_checkpointing=bool(args.grad_checkpointing),
    )
    for p in shared.backbone.parameters():
        p.requires_grad = False
    model = MultiTaskModel(
        shared=shared,
        det_num_classes=int(det_num_classes),
        seg_num_classes=int(args.seg_num_classes),
        cnt_num_classes=int(args.cnt_num_classes),
        image_size=int(args.image_size),
        det_out_channels=int(args.det_out_channels),
        det_train_backbone=False,
        seg_train_backbone=False,
        cnt_train_backbone=False,
    )
    return model.to(device)


def _make_optimizer(args, model_for_state: MultiTaskModel) -> tuple[torch.optim.Optimizer, Dict[str, int]]:
    shared_params = [p for n, p in model_for_state.shared.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    shared_ids = {id(p) for p in shared_params}
    det_params = [p for p in model_for_state.detector.parameters() if p.requires_grad and id(p) not in shared_ids]
    seg_params = [p for p in model_for_state.seg_head.parameters() if p.requires_grad]
    cnt_params = [p for p in model_for_state.cnt_head.parameters() if p.requires_grad]
    param_groups = []
    if shared_params:
        param_groups.append({"params": shared_params, "lr": float(args.lr), "weight_decay": float(args.weight_decay)})
    if det_params:
        param_groups.append(
            {
                "params": det_params,
                "lr": float(args.det_lr) if args.det_lr is not None else float(args.lr),
                "weight_decay": float(args.det_weight_decay) if args.det_weight_decay is not None else float(args.weight_decay),
            }
        )
    if seg_params:
        param_groups.append(
            {
                "params": seg_params,
                "lr": float(args.seg_lr) if args.seg_lr is not None else float(args.lr),
                "weight_decay": float(args.seg_weight_decay) if args.seg_weight_decay is not None else float(args.weight_decay),
            }
        )
    if cnt_params:
        param_groups.append(
            {
                "params": cnt_params,
                "lr": float(args.cnt_lr) if args.cnt_lr is not None else float(args.lr),
                "weight_decay": float(args.cnt_weight_decay) if args.cnt_weight_decay is not None else float(args.weight_decay),
            }
        )
    if not param_groups:
        raise RuntimeError("No trainable parameters found.")
    optimizer = torch.optim.AdamW(param_groups)
    stats = {
        "shared": sum(p.numel() for p in shared_params),
        "det": sum(p.numel() for p in det_params),
        "seg": sum(p.numel() for p in seg_params),
        "cnt": sum(p.numel() for p in cnt_params),
    }
    return optimizer, stats


def _ddp_barrier(local_rank: int) -> None:
    if not dist.is_initialized():
        return
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        dist.barrier(device_ids=[int(local_rank)])
    else:
        dist.barrier()


def main():
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if use_ddp and bool(args.use_lora_moe) and bool(args.grad_checkpointing):
        print("[warn] DDP + LoRA-MoE + grad-checkpointing may trigger autograd inplace-version errors; disabling grad-checkpointing.")
        args.grad_checkpointing = False

    try:
        if use_ddp and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if use_ddp and not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        if use_ddp and torch.cuda.is_available():
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device(args.device)

        random.seed(int(args.seed))
        np.random.seed(int(args.seed))
        torch.manual_seed(int(args.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(args.seed))

        is_main_process = (not use_ddp) or rank == 0
        if use_ddp and not is_main_process:
            builtins.print = lambda *a, **k: None

        if int(args.val_every) < 1:
            raise ValueError("--val-every must be >= 1")
        if int(args.val_last_n_epochs) < 0:
            raise ValueError("--val-last-n-epochs must be >= 0")
        if float(args.mi_loss_weight) < 0:
            raise ValueError("--mi-loss-weight must be >= 0")
        if int(args.num_experts_shared) <= 0:
            raise ValueError("--num-experts-shared must be > 0")
        if int(args.moe_k_shared) <= 0:
            raise ValueError("--moe-k-shared must be > 0")

        if (
            bool(args.unfreeze_backbone)
            or bool(args.det_unfreeze_backbone)
            or bool(args.seg_full_finetune)
            or bool(args.cnt_full_finetune)
            or args.backbone_lr is not None
            or args.backbone_weight_decay is not None
        ):
            print("[warn] Shared DINO backbone is forced frozen in mod_squad_plaintrain; ignoring backbone finetune args.")

        save_dir = Path(args.save_dir)
        if is_main_process:
            save_dir.mkdir(parents=True, exist_ok=True)
        if use_ddp:
            _ddp_barrier(local_rank)

        save_epochs = _parse_save_epochs(args.save_epochs)
        loss_weights = parse_loss_weights(args.loss_weights)
        print(f"[train] fixed task weights det/seg/cnt = [{loss_weights[0]:.3f}, {loss_weights[1]:.3f}, {loss_weights[2]:.3f}]")
        print(f"[train] mi_loss_weight = {float(args.mi_loss_weight):.6f}")

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

        if use_ddp:
            det_train_sampler = DistributedSampler(det_train_ds, num_replicas=world_size, rank=rank, shuffle=True)
            det_val_sampler = DistributedSampler(det_val_ds, num_replicas=world_size, rank=rank, shuffle=False)
            det_train_loader = DataLoader(
                det_train_ds,
                batch_size=args.det_batch_size,
                sampler=det_train_sampler,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )
            det_val_loader = DataLoader(
                det_val_ds,
                batch_size=args.det_batch_size,
                sampler=det_val_sampler,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
            )
            seg_train_sampler = DistributedSampler(seg_train_ds, num_replicas=world_size, rank=rank, shuffle=True)
            seg_val_sampler = DistributedSampler(seg_val_ds, num_replicas=world_size, rank=rank, shuffle=False)
            seg_train_loader = DataLoader(
                seg_train_ds,
                batch_size=args.seg_batch_size,
                sampler=seg_train_sampler,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            seg_val_loader = DataLoader(
                seg_val_ds,
                batch_size=args.seg_batch_size,
                sampler=seg_val_sampler,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            cnt_kwargs = {"batch_size": args.cnt_batch_size, "num_workers": 1, "pin_memory": True}
            cnt_kwargs["persistent_workers"] = True
            cnt_kwargs["multiprocessing_context"] = "spawn"
            cnt_kwargs["prefetch_factor"] = 2
            cnt_train_sampler = DistributedSampler(cnt_train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
            cnt_val_sampler = DistributedSampler(cnt_val_ds, num_replicas=world_size, rank=rank, shuffle=False)
            cnt_train_loader = DataLoader(cnt_train_ds, sampler=cnt_train_sampler, shuffle=False, drop_last=True, **cnt_kwargs)
            cnt_val_loader = DataLoader(cnt_val_ds, sampler=cnt_val_sampler, shuffle=False, **cnt_kwargs)

        det_num_classes = int(args.det_num_classes) if args.det_num_classes is not None else int(det_train_ds.num_classes)
        model = _build_model(args, det_num_classes=det_num_classes, device=device)
        if use_ddp:
            ddp_device_ids = [local_rank] if device.type == "cuda" else None
            ddp_output_device = local_rank if device.type == "cuda" else None
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=ddp_device_ids,
                output_device=ddp_output_device,
                find_unused_parameters=True,
                broadcast_buffers=bool(args.ddp_broadcast_buffers),
            )
        model_for_state = model.module if use_ddp else model
        optimizer, param_stats = _make_optimizer(args, model_for_state)
        print(
            "[train] trainable params | "
            f"shared={param_stats['shared']} det={param_stats['det']} seg={param_stats['seg']} cnt={param_stats['cnt']}"
        )
        scaler = GradScaler(device.type, enabled=bool(args.amp))
        autocast_device = device.type if device.type in {"cuda", "cpu"} else "cuda"
        trainable_params = _all_trainable_params(model_for_state)

        train_loaders = {"det": det_train_loader, "seg": seg_train_loader, "cnt": cnt_train_loader}
        val_loaders = {"det": det_val_loader, "seg": seg_val_loader, "cnt": cnt_val_loader}
        primary = choose_primary({k: len(v) for k, v in train_loaders.items()}, args.primary_task)
        print(f"[train] primary task loader = {primary}")

        model_config = model_for_state.shared.export_model_config()
        model_config["mi_loss_weight"] = float(args.mi_loss_weight)
        total_epochs = int(args.epochs)
        val_last_n_epochs = int(args.val_last_n_epochs)
        val_start_epoch = max(1, total_epochs - val_last_n_epochs + 1) if val_last_n_epochs > 0 else 1
        if is_main_process and not bool(args.skip_validation):
            if val_last_n_epochs > 0:
                print(
                    f"[train] validation/model selection enabled from epoch {val_start_epoch} "
                    f"to {total_epochs} (last {val_last_n_epochs} epochs)"
                )
            else:
                print("[train] validation/model selection enabled from epoch 1")

        def _compute_losses(det_batch, seg_batch, cnt_batch, *, collect_cnt_stats: bool):
            det_images, det_targets = _to_device_det(det_batch, device)
            seg_imgs, seg_masks = _to_device_seg(seg_batch, device)
            cnt_imgs, cnt_dens = _to_device_cnt(cnt_batch, device)
            cnt_gt_counts = cnt_dens.flatten(2).sum(dim=2)
            with torch.amp.autocast(autocast_device, enabled=bool(args.amp)):
                det_losses, seg_logits, pred_dens, pred_counts = model(
                    "all",
                    det_images,
                    det_targets,
                    seg_imgs,
                    cnt_imgs,
                    cnt_backbone_grad_mult=float(args.cnt_backbone_grad_mult),
                )
                det_loss = sum(det_losses.values())
                seg_loss = F.cross_entropy(seg_logits, seg_masks)
                dens_loss = F.mse_loss(pred_dens, cnt_dens, reduction="sum") / cnt_imgs.size(0)
                cnt_l1 = F.l1_loss(pred_counts, cnt_gt_counts)
                cnt_loss = dens_loss + float(args.cnt_count_loss_weight) * cnt_l1
            cnt_stats = None
            if collect_cnt_stats:
                cnt_stats = _count_diag_stats(pred_dens, cnt_dens, pred_counts, cnt_gt_counts)
            return det_loss, seg_loss, cnt_loss, cnt_stats

        def _run_validation(epoch: int) -> tuple[float, Dict[str, float]]:
            model_for_state.clear_aux_stats()
            model_for_state.set_collect_aux_stats(False)
            try:
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
                val_ap50 = _eval_det_ap50_fast(
                    model,
                    val_loaders["det"],
                    device,
                    num_classes=det_num_classes,
                    score_thresh=float(args.det_ap_score_thr),
                )
            finally:
                model_for_state.clear_aux_stats()
                model_for_state.set_collect_aux_stats(True)
            combo_metric = float(val_ap50) + float(val_seg_miou) + 1.0 / max(float(val_cnt_mae), 1e-8)
            metrics = {
                "val_det_loss": float(val_det),
                "val_seg_loss": float(val_seg),
                "val_seg_miou": float(val_seg_miou),
                "val_cnt_loss": float(val_cnt),
                "val_cnt_density_mse": float(val_cnt_density),
                "val_cnt_mae": float(val_cnt_mae),
                "val_cnt_total_mae": float(val_cnt_total_mae),
                "val_ap50": float(val_ap50),
                "selected_metric": float(combo_metric),
                "epoch": float(epoch),
            }
            print(
                f"[val] epoch {epoch} | det {val_det:.4f} seg {val_seg:.4f} miou {val_seg_miou:.4f} "
                f"cnt {val_cnt:.4f} dens {val_cnt_density:.6e} mae {val_cnt_mae:.4f} total_mae {val_cnt_total_mae:.4f} "
                f"| ap50 {val_ap50:.4f} | combo {combo_metric:.6f}"
            )
            return combo_metric, metrics

        best_metric = float("-inf")
        best_path = save_dir / "best_combo.pt"
        last_path = save_dir / "last.pt"
        last_metrics: Dict[str, float] = {"selected_metric": float("nan")}

        for epoch in range(1, total_epochs + 1):
            if use_ddp:
                for loader in (det_train_loader, seg_train_loader, cnt_train_loader):
                    sampler = getattr(loader, "sampler", None)
                    if isinstance(sampler, DistributedSampler):
                        sampler.set_epoch(epoch)

            model.train()
            model_for_state.set_collect_aux_stats(True)
            total_loss_sum = 0.0
            det_loss_sum = 0.0
            seg_loss_sum = 0.0
            cnt_loss_sum = 0.0
            mi_loss_sum = 0.0
            steps = 0

            full_cyc = {k: infinite_loader(v) for k, v in train_loaders.items() if k != primary}
            for step, primary_batch in enumerate(train_loaders[primary], start=1):
                batches = {primary: primary_batch}
                for key in train_loaders.keys():
                    if key != primary:
                        batches[key] = next(full_cyc[key])

                optimizer.zero_grad(set_to_none=True)
                model_for_state.clear_aux_stats()
                det_loss, seg_loss, cnt_loss, cnt_stats = _compute_losses(
                    batches["det"],
                    batches["seg"],
                    batches["cnt"],
                    collect_cnt_stats=(step == 1),
                )
                mi_loss = model_for_state.get_mi_loss_and_clear()
                total = (
                    float(loss_weights[0]) * det_loss
                    + float(loss_weights[1]) * seg_loss
                    + float(loss_weights[2]) * cnt_loss
                    + float(args.mi_loss_weight) * mi_loss
                )

                if scaler.is_enabled():
                    scaler.scale(total).backward()
                    if float(args.grad_clip_norm) > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(args.grad_clip_norm))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total.backward()
                    if float(args.grad_clip_norm) > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(args.grad_clip_norm))
                    optimizer.step()

                total_loss_sum += float(total.detach().item())
                det_loss_sum += float(det_loss.detach().item())
                seg_loss_sum += float(seg_loss.detach().item())
                cnt_loss_sum += float(cnt_loss.detach().item())
                mi_loss_sum += float(mi_loss.detach().item())
                steps += 1

                if args.log_interval and step % int(args.log_interval) == 0:
                    print(
                        f"[train] epoch {epoch}/{args.epochs} step {step} | "
                        f"loss {total_loss_sum/max(steps, 1):.4f} | "
                        f"det {det_loss_sum/max(steps, 1):.4f} seg {seg_loss_sum/max(steps, 1):.4f} "
                        f"cnt {cnt_loss_sum/max(steps, 1):.4f} mi {mi_loss_sum/max(steps, 1):.4f}"
                    )
                if step == 1 and cnt_stats is not None:
                    print(f"[diag][cnt][train] epoch {epoch}/{args.epochs} step {step} | {_format_count_diag(cnt_stats)}")
                if args.max_train_steps and step >= int(args.max_train_steps):
                    break

            epoch_metrics: Dict[str, float] = {
                "train_total_loss": total_loss_sum / max(steps, 1),
                "train_det_loss": det_loss_sum / max(steps, 1),
                "train_seg_loss": seg_loss_sum / max(steps, 1),
                "train_cnt_loss": cnt_loss_sum / max(steps, 1),
                "train_mi_loss": mi_loss_sum / max(steps, 1),
                "selected_metric": float("nan"),
            }
            print(
                f"[train] epoch {epoch}/{args.epochs} | "
                f"loss {epoch_metrics['train_total_loss']:.4f} | "
                f"det {epoch_metrics['train_det_loss']:.4f} seg {epoch_metrics['train_seg_loss']:.4f} "
                f"cnt {epoch_metrics['train_cnt_loss']:.4f} mi {epoch_metrics['train_mi_loss']:.4f}"
            )

            do_val = (
                (not bool(args.skip_validation))
                and (epoch >= val_start_epoch)
                and (epoch % int(args.val_every) == 0)
            )
            if do_val:
                combo_metric, val_metrics = _run_validation(epoch)
                epoch_metrics.update(val_metrics)
                if combo_metric > best_metric and is_main_process:
                    best_metric = float(combo_metric)
                    save_multitask_checkpoint(
                        str(best_path),
                        model=model_for_state,
                        optimizer=optimizer,
                        epoch=epoch,
                        best_by="combo",
                        metrics=epoch_metrics,
                        loss_weights=loss_weights,
                        phi_state=None,
                        model_config=model_config,
                    )
                    print(f"[ckpt] saved new best -> {best_path} (combo {best_metric:.6f})")
                elif combo_metric > best_metric:
                    best_metric = float(combo_metric)
            last_metrics = dict(epoch_metrics)

            if is_main_process:
                save_multitask_checkpoint(
                    str(last_path),
                    model=model_for_state,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_by="last",
                    metrics=epoch_metrics,
                    loss_weights=loss_weights,
                    phi_state=None,
                    model_config=model_config,
                )
                print(f"[ckpt] saved last -> {last_path}")
                if epoch in save_epochs:
                    epoch_path = save_dir / f"epoch_{epoch}.pt"
                    save_multitask_checkpoint(
                        str(epoch_path),
                        model=model_for_state,
                        optimizer=optimizer,
                        epoch=epoch,
                        best_by="epoch",
                        metrics=epoch_metrics,
                        loss_weights=loss_weights,
                        phi_state=None,
                        model_config=model_config,
                    )
                    print(f"[ckpt] saved extra epoch checkpoint -> {epoch_path}")

            if is_main_process and not best_path.exists():
                save_multitask_checkpoint(
                    str(best_path),
                    model=model_for_state,
                    optimizer=optimizer,
                    epoch=int(args.epochs),
                    best_by="combo",
                    metrics=last_metrics,
                    loss_weights=loss_weights,
                    phi_state=None,
                    model_config=model_config,
                )
                print(f"[ckpt] validation skipped or no best update; wrote fallback best checkpoint -> {best_path}")

    finally:
        if use_ddp and dist.is_initialized():
            try:
                _ddp_barrier(local_rank)
            finally:
                dist.destroy_process_group()


if __name__ == "__main__":
    main()

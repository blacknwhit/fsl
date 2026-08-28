from __future__ import annotations

import argparse
import builtins
import importlib
import os
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    from scipy.optimize import least_squares as scipy_least_squares
except Exception:
    scipy_least_squares = None

try:
    from .models import MultiTaskModel, SharedDinoV3Backbone
    from .utils import choose_primary, infinite_loader, parse_loss_weights, save_multitask_checkpoint
except ImportError:
    from models import MultiTaskModel, SharedDinoV3Backbone
    from utils import choose_primary, infinite_loader, parse_loss_weights, save_multitask_checkpoint


def _load_113_test_datasets():
    module = importlib.import_module("113_test.datasets")
    return module.build_det_loaders, module.build_seg_loaders, module.build_cnt_loaders


def _load_segmentation_utils():
    module = importlib.import_module("segmentation.utils")
    return module.per_class_iou_from_confusion, module.update_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Train the 113_pivrg multitask baseline")
    parser.add_argument("--model-name", type=str, default="dinov3_vitl16")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--backbone-checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument("--det-batch-size", type=int, default=2)
    parser.add_argument("--seg-batch-size", type=int, default=2)
    parser.add_argument("--cnt-batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-lr", type=float, default=None)
    parser.add_argument("--lora-weight-decay", type=float, default=0.0)
    parser.add_argument("--det-lr", type=float, default=None)
    parser.add_argument("--seg-lr", type=float, default=None)
    parser.add_argument("--cnt-lr", type=float, default=None)
    parser.add_argument("--det-weight-decay", type=float, default=None)
    parser.add_argument("--seg-weight-decay", type=float, default=None)
    parser.add_argument("--cnt-weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip-norm", type=float, default=100.0)
    parser.add_argument("--loss-weights", type=str, default="15:8:1")
    parser.add_argument("--save-dir", type=str, default="runs/113_pivrg")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--primary-task", type=str, default=None, help="det|seg|cnt")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--det-ap-score-thr", type=float, default=0.0)
    parser.add_argument("--det-out-channels", type=int, default=256)
    parser.add_argument("--cnt-count-loss-weight", type=float, default=1.0)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-val-steps", type=int, default=0)

    parser.add_argument("--pivrg-base-det-ap50", type=float, default=0.0)
    parser.add_argument("--pivrg-base-seg-miou", type=float, default=0.0)
    parser.add_argument("--pivrg-base-cnt-mae", type=float, default=0.0)
    parser.add_argument("--pivrg-bound", type=float, default=2.0)
    parser.add_argument("--pivrg-mintemp", type=float, default=10.0)
    parser.add_argument("--pivrg-warmup-epochs", type=int, default=10)

    parser.add_argument("--det-data-root", type=str, required=True)
    parser.add_argument("--det-train-ann", type=str, default=None)
    parser.add_argument("--det-val-ann", type=str, default=None)
    parser.add_argument("--det-train-img-dir", type=str, default=None)
    parser.add_argument("--det-val-img-dir", type=str, default=None)
    parser.add_argument("--det-num-classes", type=int, default=None)

    parser.add_argument("--seg-train-dir", type=str, required=True)
    parser.add_argument("--seg-val-dir", type=str, required=True)
    parser.add_argument("--seg-num-classes", type=int, default=11)

    parser.add_argument("--cnt-data-root", type=str, required=True)
    parser.add_argument("--cnt-train-dir", type=str, default=None)
    parser.add_argument("--cnt-val-dir", type=str, default=None)
    parser.add_argument("--cnt-num-classes", type=int, default=8)
    aspect = parser.add_mutually_exclusive_group()
    aspect.add_argument("--cnt-keep-aspect", dest="cnt_keep_aspect", action="store_true")
    aspect.add_argument("--cnt-no-keep-aspect", dest="cnt_keep_aspect", action="store_false")
    parser.set_defaults(cnt_keep_aspect=True)

    return parser.parse_args()


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_distributed(args) -> tuple[torch.device, bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_ddp = world_size > 1

    if use_ddp and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)

    if use_ddp and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device(args.device)
    return device, use_ddp, world_size, rank, local_rank


def _to_device_det(batch, device: torch.device):
    images, targets = batch
    images = [img.to(device, non_blocking=True) for img in images]
    targets = [{k: v.to(device, non_blocking=True) for k, v in target.items()} for target in targets]
    return images, targets


def _to_device_seg(batch, device: torch.device):
    images, masks = batch
    return images.to(device, non_blocking=True), masks.to(device, non_blocking=True)


def _to_device_cnt(batch, device: torch.device):
    images, density = batch
    return images.to(device, non_blocking=True).float(), density.to(device, non_blocking=True).float()


def _build_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    collate_fn,
    use_ddp: bool,
    world_size: int,
    rank: int,
    count_loader: bool,
):
    sampler = None
    if use_ddp:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": sampler is None and shuffle,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if collate_fn is not None:
        loader_kwargs["collate_fn"] = collate_fn
    if sampler is not None:
        loader_kwargs["sampler"] = sampler
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        if count_loader:
            loader_kwargs["multiprocessing_context"] = "spawn"
            loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(dataset, **loader_kwargs)
    return loader, sampler


def _count_diag_stats(
    pred_density: torch.Tensor,
    gt_density: torch.Tensor,
    pred_counts: torch.Tensor,
    gt_counts: torch.Tensor,
) -> Dict[str, float]:
    pred_density = pred_density.detach().float()
    gt_density = gt_density.detach().float()
    pred_counts = pred_counts.detach().float()
    gt_counts = gt_counts.detach().float()
    return {
        "pred_dens_mean": float(pred_density.mean().item()),
        "gt_dens_mean": float(gt_density.mean().item()),
        "pred_count_mean": float(pred_counts.mean().item()),
        "gt_count_mean": float(gt_counts.mean().item()),
        "count_mae": float((pred_counts - gt_counts).abs().mean().item()),
        "pred_total_mean": float(pred_counts.sum(dim=1).mean().item()),
        "gt_total_mean": float(gt_counts.sum(dim=1).mean().item()),
    }


def _state_dict_cpu_clone(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if torch.is_tensor(value):
            out[key] = value.detach().cpu().clone()
        else:
            out[key] = value
    return out


def _format_count_diag(stats: Dict[str, float]) -> str:
    return (
        f"dens {stats['pred_dens_mean']:.6e}/{stats['gt_dens_mean']:.6e} | "
        f"count {stats['pred_count_mean']:.3f}/{stats['gt_count_mean']:.3f} | "
        f"mae {stats['count_mae']:.3f} | "
        f"total {stats['pred_total_mean']:.3f}/{stats['gt_total_mean']:.3f}"
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
    for threshold in np.linspace(0, 1, 101):
        precision = prec[rec >= threshold].max() if np.any(rec >= threshold) else 0.0
        ap += precision
    return ap / 101.0


@torch.no_grad()
def _eval_det_ap50_fast(model, loader, device: torch.device, num_classes: int, score_thresh: float) -> float:
    model.eval()
    preds_by_cls = {cls: [] for cls in range(1, num_classes + 1)}
    gts_by_cls = {cls: {} for cls in range(1, num_classes + 1)}
    dist_rank = dist.get_rank() if dist.is_initialized() else 0

    image_counter = 0
    for images, targets in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        outputs = model("det", images)

        for output, target in zip(outputs, targets):
            default_image_id = dist_rank * 10_000_000_000 + image_counter
            image_id = int(target.get("image_id", torch.tensor([default_image_id])).item())
            image_counter += 1

            gt_boxes = target["boxes"].detach().cpu().numpy()
            gt_labels = target["labels"].detach().cpu().numpy().astype(int)
            for box, cls in zip(gt_boxes, gt_labels):
                gts_by_cls[cls].setdefault(image_id, []).append(box)

            pred_boxes = output["boxes"].detach().cpu().numpy()
            pred_labels = output["labels"].detach().cpu().numpy().astype(int)
            pred_scores = output["scores"].detach().cpu().numpy()
            keep = pred_scores >= score_thresh
            for box, cls, score in zip(pred_boxes[keep], pred_labels[keep], pred_scores[keep]):
                preds_by_cls[cls].append((image_id, float(score), box))

    if dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, (preds_by_cls, gts_by_cls))

        merged_preds = {cls: [] for cls in range(1, num_classes + 1)}
        merged_gts = {cls: {} for cls in range(1, num_classes + 1)}
        for rank_preds, rank_gts in gathered:
            for cls in range(1, num_classes + 1):
                merged_preds[cls].extend(rank_preds.get(cls, []))
                for image_id, boxes in rank_gts.get(cls, {}).items():
                    merged_gts[cls].setdefault(image_id, []).extend(boxes)
        preds_by_cls = merged_preds
        gts_by_cls = merged_gts

    ap_list = []
    for cls in range(1, num_classes + 1):
        preds = preds_by_cls[cls]
        gts = gts_by_cls[cls]
        num_gt = sum(len(boxes) for boxes in gts.values())
        if num_gt == 0:
            continue

        preds.sort(key=lambda item: item[1], reverse=True)
        matched = {image_id: [False] * len(boxes) for image_id, boxes in gts.items()}
        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)

        for index, (image_id, _score, pred_box) in enumerate(preds):
            if image_id not in gts:
                fp[index] = 1.0
                continue
            gt_boxes = np.array(gts[image_id], dtype=np.float32)
            ious = _box_iou_np(np.array([pred_box], dtype=np.float32), gt_boxes)[0]
            best = int(np.argmax(ious)) if ious.size > 0 else -1
            if best >= 0 and ious[best] >= 0.5 and not matched[image_id][best]:
                tp[index] = 1.0
                matched[image_id][best] = True
            else:
                fp[index] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / max(num_gt, 1)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        ap_list.append(_ap_from_pr(rec, prec))
    return float(np.mean(ap_list)) if ap_list else 0.0


@torch.no_grad()
def _eval_det_loss(model, loader, device: torch.device) -> float:
    model.eval()
    model.detector.train()
    model.shared.eval()
    model.seg_head.eval()
    model.cnt_head.eval()
    total = 0.0
    samples = 0
    for images, targets in loader:
        images, targets = _to_device_det((images, targets), device)
        loss_dict = model("det", images, targets)
        loss = sum(loss_dict.values())
        batch_size = len(images)
        total += float(loss.item()) * batch_size
        samples += batch_size
    if dist.is_initialized():
        reduced = torch.tensor([total, float(samples)], device=device, dtype=torch.float64)
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        total = float(reduced[0].item())
        samples = int(reduced[1].item())
    return total / max(samples, 1)


@torch.no_grad()
def _eval_seg_loss(model, loader, device: torch.device, *, num_classes: int, max_steps: int) -> tuple[float, float]:
    per_class_iou_from_confusion, update_confusion_matrix = _load_segmentation_utils()

    model.eval()
    total = 0.0
    samples = 0
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for step, (images, masks) in enumerate(loader, start=1):
        images, masks = _to_device_seg((images, masks), device)
        logits = model("seg", images)
        loss = F.cross_entropy(logits, masks)
        batch_size = images.size(0)
        total += float(loss.item()) * batch_size
        samples += batch_size
        update_confusion_matrix(
            conf=conf,
            logits_or_preds=logits.detach(),
            target=masks.detach(),
            num_classes=num_classes,
            ignore_indices=(255, 11),
        )
        if max_steps and step >= max_steps:
            break
    if dist.is_initialized():
        reduced = torch.tensor([total, float(samples)], device=device, dtype=torch.float64)
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        total = float(reduced[0].item())
        samples = int(reduced[1].item())
        conf_device = conf.to(device=device)
        dist.all_reduce(conf_device, op=dist.ReduceOp.SUM)
        conf = conf_device.cpu()

    _, miou = per_class_iou_from_confusion(conf)
    return total / max(samples, 1), float(miou.item())


@torch.no_grad()
def _eval_cnt_loss(model, loader, device: torch.device, *, count_loss_weight: float, max_steps: int) -> tuple[float, float, float, float]:
    model.eval()
    total = 0.0
    total_density = 0.0
    total_count_mae = 0.0
    total_total_mae = 0.0
    samples = 0
    for step, (images, density) in enumerate(loader, start=1):
        images, density = _to_device_cnt((images, density), device)
        gt_counts = density.flatten(2).sum(dim=2)
        pred_density, pred_counts = model("cnt", images)
        density_loss = F.mse_loss(pred_density, density, reduction="sum") / images.size(0)
        count_l1 = F.l1_loss(pred_counts, gt_counts)
        loss = density_loss + float(count_loss_weight) * count_l1
        count_mae = (pred_counts - gt_counts).abs().mean()
        pred_total = pred_counts.sum(dim=1)
        gt_total = gt_counts.sum(dim=1)
        total_mae = (pred_total - gt_total).abs().mean()

        batch_size = images.size(0)
        samples += batch_size
        total += float(loss.item()) * batch_size
        total_density += float(density_loss.item()) * batch_size
        total_count_mae += float(count_mae.item()) * batch_size
        total_total_mae += float(total_mae.item()) * batch_size
        if max_steps and step >= max_steps:
            break

    if dist.is_initialized():
        reduced = torch.tensor(
            [total, total_density, total_count_mae, total_total_mae, float(samples)],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        total = float(reduced[0].item())
        total_density = float(reduced[1].item())
        total_count_mae = float(reduced[2].item())
        total_total_mae = float(reduced[3].item())
        samples = int(reduced[4].item())

    denom = max(samples, 1)
    return total / denom, total_density / denom, total_count_mae / denom, total_total_mae / denom


def _shared_grad_dims(parameters) -> list[int]:
    return [parameter.numel() for parameter in parameters]


def _flatten_shared_grads(parameters, grad_dims: list[int], device: torch.device) -> torch.Tensor:
    grads = torch.zeros(sum(grad_dims), device=device, dtype=torch.float32)
    offset = 0
    for parameter, dim in zip(parameters, grad_dims):
        if parameter.grad is not None:
            grads[offset : offset + dim].copy_(parameter.grad.detach().reshape(-1).float())
        offset += dim
    return grads


def _clear_parameter_grads(parameters) -> None:
    for parameter in parameters:
        parameter.grad = None


def _overwrite_shared_grads(parameters, grad_dims: list[int], new_grad: torch.Tensor) -> None:
    offset = 0
    for parameter, dim in zip(parameters, grad_dims):
        parameter.grad = new_grad[offset : offset + dim].view_as(parameter).clone()
        offset += dim


def _fixed_point_pivrg(A: np.ndarray, w: np.ndarray, x0: np.ndarray, max_iters: int = 256, tol: float = 1e-8) -> np.ndarray:
    x = np.clip(np.asarray(x0, dtype=np.float64), 1e-12, None)
    for _ in range(max_iters):
        Ax = np.maximum(A.dot(x), 1e-12)
        candidate = np.clip(w / np.maximum(Ax * Ax, 1e-12), 1e-12, None)
        updated = 0.5 * x + 0.5 * candidate
        if not np.all(np.isfinite(updated)):
            break
        if np.max(np.abs(updated - x)) < tol:
            x = updated
            break
        x = updated
    return np.clip(x, 1e-12, None)


def _solve_pivrg(G: torch.Tensor, scores: np.ndarray, *, bound: float, mintemp: float) -> tuple[torch.Tensor, np.ndarray, float]:
    num_tasks = int(G.shape[1])
    if num_tasks <= 0:
        raise ValueError("PIVRG requires at least one task gradient")
    if bound <= 1.0:
        raise ValueError("--pivrg-bound must be > 1.0")

    GG = G.t().mm(G).detach().cpu().numpy().astype(np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    score_span = float(scores.max() - scores.min()) if scores.size > 0 else 0.0
    temp = max(score_span / np.log(bound) if score_span > 0 else 0.0, float(mintemp))
    shifted = (scores - scores.max()) / temp
    pref = num_tasks * (np.exp(shifted) / np.exp(shifted).sum())
    x0 = np.full(num_tasks, 1.0 / num_tasks, dtype=np.float64)

    if scipy_least_squares is not None:
        def residual(x):
            x = np.clip(x, 1e-12, None)
            return np.dot(GG, x) - np.sqrt(pref / x)

        result = scipy_least_squares(residual, x0, bounds=(1e-12, np.inf))
        coeffs = np.clip(result.x, 1e-12, None)
    else:
        coeffs = _fixed_point_pivrg(GG, pref, x0)

    coeff_tensor = torch.from_numpy(coeffs.astype(np.float32)).to(G.device)
    return coeff_tensor, pref, float(temp)


def _sync_grads(parameters, world_size: int) -> None:
    if not dist.is_initialized():
        return
    for parameter in parameters:
        has_grad = torch.tensor(1 if parameter.grad is not None else 0, device=parameter.device, dtype=torch.int32)
        dist.all_reduce(has_grad, op=dist.ReduceOp.SUM)
        if int(has_grad.item()) == 0:
            parameter.grad = None
            continue
        grad = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
        dist.all_reduce(grad, op=dist.ReduceOp.SUM)
        grad /= float(world_size)
        parameter.grad = grad


def _reduce_train_sums(values: list[float], device: torch.device) -> list[float]:
    if not dist.is_initialized():
        return values
    reduced = torch.tensor(values, device=device, dtype=torch.float64)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    return [float(item) for item in reduced.cpu().tolist()]


def _validate_pivrg_args(args) -> None:
    if args.amp:
        raise SystemExit("113_pivrg v1 does not support --amp. Leave AMP=0 in the launch script.")
    baseline_values = (
        float(args.pivrg_base_det_ap50),
        float(args.pivrg_base_seg_miou),
        float(args.pivrg_base_cnt_mae),
    )
    if any(value <= 0 for value in baseline_values):
        raise SystemExit(
            "PIVRG baseline metrics must all be > 0. "
            "Set --pivrg-base-det-ap50, --pivrg-base-seg-miou, and --pivrg-base-cnt-mae."
        )


def _compute_scores(args, *, val_ap50: float, val_seg_miou: float, val_cnt_mae: float) -> np.ndarray:
    return np.asarray(
        [
            (float(args.pivrg_base_det_ap50) - float(val_ap50)) / float(args.pivrg_base_det_ap50),
            (float(args.pivrg_base_seg_miou) - float(val_seg_miou)) / float(args.pivrg_base_seg_miou),
            (float(val_cnt_mae) - float(args.pivrg_base_cnt_mae)) / float(args.pivrg_base_cnt_mae),
        ],
        dtype=np.float64,
    )


def main():
    args = parse_args()
    _validate_pivrg_args(args)

    device, use_ddp, world_size, rank, _local_rank = _init_distributed(args)
    is_main_process = (not use_ddp) or rank == 0
    if use_ddp and not is_main_process:
        builtins.print = lambda *unused_args, **unused_kwargs: None

    _set_random_seed(int(args.seed))
    save_dir = Path(args.save_dir)
    if is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

    build_det_loaders, build_seg_loaders, build_cnt_loaders = _load_113_test_datasets()

    det_train_ds, det_val_ds, det_train_loader_base, det_val_loader_base = build_det_loaders(
        data_root=args.det_data_root,
        image_size=args.image_size,
        batch_size=args.det_batch_size,
        num_workers=args.num_workers,
        train_ann=args.det_train_ann,
        val_ann=args.det_val_ann,
        train_img_dir=args.det_train_img_dir,
        val_img_dir=args.det_val_img_dir,
    )
    seg_train_ds, seg_val_ds, _, _ = build_seg_loaders(
        train_dir=args.seg_train_dir,
        val_dir=args.seg_val_dir,
        image_size=args.image_size,
        batch_size=args.seg_batch_size,
        num_workers=args.num_workers,
    )
    cnt_train_ds, cnt_val_ds, _, _ = build_cnt_loaders(
        data_root=args.cnt_data_root,
        train_dir=args.cnt_train_dir,
        val_dir=args.cnt_val_dir,
        image_size=args.image_size,
        num_classes=args.cnt_num_classes,
        keep_aspect=bool(args.cnt_keep_aspect),
        batch_size=args.cnt_batch_size,
        num_workers=args.num_workers,
    )

    det_train_loader, det_train_sampler = _build_loader(
        det_train_ds,
        batch_size=args.det_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=det_train_loader_base.collate_fn,
        use_ddp=use_ddp,
        world_size=world_size,
        rank=rank,
        count_loader=False,
    )
    det_val_loader, _ = _build_loader(
        det_val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=det_val_loader_base.collate_fn,
        use_ddp=use_ddp,
        world_size=world_size,
        rank=rank,
        count_loader=False,
    )
    seg_train_loader, seg_train_sampler = _build_loader(
        seg_train_ds,
        batch_size=args.seg_batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=None,
        use_ddp=use_ddp,
        world_size=world_size,
        rank=rank,
        count_loader=False,
    )
    seg_val_loader, _ = _build_loader(
        seg_val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=None,
        use_ddp=use_ddp,
        world_size=world_size,
        rank=rank,
        count_loader=False,
    )
    cnt_train_loader, cnt_train_sampler = _build_loader(
        cnt_train_ds,
        batch_size=args.cnt_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        collate_fn=None,
        use_ddp=use_ddp,
        world_size=world_size,
        rank=rank,
        count_loader=True,
    )
    cnt_val_loader, _ = _build_loader(
        cnt_val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=None,
        use_ddp=use_ddp,
        world_size=world_size,
        rank=rank,
        count_loader=True,
    )

    det_num_classes = int(args.det_num_classes) if args.det_num_classes else int(det_train_ds.num_classes)
    loss_weights = parse_loss_weights(args.loss_weights)

    shared = SharedDinoV3Backbone(
        model_name=args.model_name,
        image_size=args.image_size,
        checkpoint_path=args.backbone_checkpoint,
        use_lora=True,
        lora_rank=int(args.lora_rank),
        lora_alpha=float(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
    )
    model = MultiTaskModel(
        shared=shared,
        det_num_classes=det_num_classes,
        seg_num_classes=args.seg_num_classes,
        cnt_num_classes=args.cnt_num_classes,
        image_size=args.image_size,
        det_out_channels=args.det_out_channels,
    ).to(device)

    shared_backbone_params = list(model.shared.backbone.parameters())
    shared_backbone_ids = {id(parameter) for parameter in shared_backbone_params}
    shared_lora_params = [parameter for parameter in shared_backbone_params if parameter.requires_grad]
    det_head_params = [parameter for parameter in model.detector.parameters() if parameter.requires_grad and id(parameter) not in shared_backbone_ids]
    seg_head_params = [parameter for parameter in model.seg_head.parameters() if parameter.requires_grad]
    cnt_head_params = [parameter for parameter in model.cnt_head.parameters() if parameter.requires_grad]

    if not shared_lora_params:
        raise RuntimeError("No trainable shared LoRA parameters found in model.shared.backbone")

    lora_lr = float(args.lora_lr) if args.lora_lr is not None else float(args.lr)
    det_lr = float(args.det_lr) if args.det_lr is not None else float(args.lr)
    seg_lr = float(args.seg_lr) if args.seg_lr is not None else float(args.lr)
    cnt_lr = float(args.cnt_lr) if args.cnt_lr is not None else float(args.lr)
    det_wd = float(args.det_weight_decay) if args.det_weight_decay is not None else float(args.weight_decay)
    seg_wd = float(args.seg_weight_decay) if args.seg_weight_decay is not None else float(args.weight_decay)
    cnt_wd = float(args.cnt_weight_decay) if args.cnt_weight_decay is not None else float(args.weight_decay)

    optimizer = torch.optim.AdamW(
        [
            {"params": shared_lora_params, "lr": lora_lr, "weight_decay": float(args.lora_weight_decay)},
            {"params": det_head_params, "lr": det_lr, "weight_decay": det_wd},
            {"params": seg_head_params, "lr": seg_lr, "weight_decay": seg_wd},
            {"params": cnt_head_params, "lr": cnt_lr, "weight_decay": cnt_wd},
        ]
    )
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]

    if is_main_process:
        shared_ids = {id(parameter) for parameter in shared_lora_params}
        print(f"[train] trainable params: {sum(parameter.numel() for parameter in trainable_params)}")
        print(f"[train] shared LoRA params: {sum(parameter.numel() for parameter in shared_lora_params)}")
        print(
            f"[train] task head params: {sum(parameter.numel() for parameter in trainable_params if id(parameter) not in shared_ids)}"
        )

    train_loaders = {"det": det_train_loader, "seg": seg_train_loader, "cnt": cnt_train_loader}
    val_loaders = {"det": det_val_loader, "seg": seg_val_loader, "cnt": cnt_val_loader}
    train_lengths = {name: len(loader) for name, loader in train_loaders.items()}
    primary_task = choose_primary(train_lengths, args.primary_task)
    cyc_loaders = {name: infinite_loader(loader) for name, loader in train_loaders.items() if name != primary_task}

    task_weights = {"det": float(loss_weights[0]), "seg": float(loss_weights[1]), "cnt": float(loss_weights[2])}
    grad_dims = _shared_grad_dims(shared_lora_params)
    prev_scores = np.zeros(3, dtype=np.float64)

    best_metric = -float("inf")
    best_epoch = 0
    best_path = save_dir / "best_combo.pt"
    best_state = None
    best_metrics = None

    for epoch in range(1, int(args.epochs) + 1):
        if use_ddp:
            for sampler in (det_train_sampler, seg_train_sampler, cnt_train_sampler):
                if sampler is not None:
                    sampler.set_epoch(epoch)

        model.train()
        use_pivrg = epoch > int(args.pivrg_warmup_epochs)
        if is_main_process:
            print(
                f"[train] epoch {epoch}/{args.epochs} | mode {'pivrg' if use_pivrg else 'warmup'} | "
                f"scores det={prev_scores[0]:.6f} seg={prev_scores[1]:.6f} cnt={prev_scores[2]:.6f}"
            )

        total_loss = 0.0
        det_loss_sum = 0.0
        seg_loss_sum = 0.0
        cnt_loss_sum = 0.0
        steps = 0
        coeff_sum = np.zeros(3, dtype=np.float64)
        pref_sum = np.zeros(3, dtype=np.float64)
        temp_sum = 0.0
        pivrg_steps = 0

        for step, primary_batch in enumerate(train_loaders[primary_task], start=1):
            batches = {primary_task: primary_batch}
            for name in train_loaders.keys():
                if name != primary_task:
                    batches[name] = next(cyc_loaders[name])

            optimizer.zero_grad(set_to_none=True)
            weighted_shared_grads = []
            raw_shared_grads = []

            det_images, det_targets = _to_device_det(batches["det"], device)
            det_loss_dict = model("det", det_images, det_targets)
            det_loss = sum(det_loss_dict.values())
            (det_loss * task_weights["det"]).backward()
            det_shared_weighted = _flatten_shared_grads(shared_lora_params, grad_dims, device)
            weighted_shared_grads.append(det_shared_weighted)
            raw_shared_grads.append(
                det_shared_weighted / task_weights["det"] if task_weights["det"] != 0 else torch.zeros_like(det_shared_weighted)
            )
            _clear_parameter_grads(shared_lora_params)

            seg_images, seg_masks = _to_device_seg(batches["seg"], device)
            seg_logits = model("seg", seg_images)
            seg_loss = F.cross_entropy(seg_logits, seg_masks)
            (seg_loss * task_weights["seg"]).backward()
            seg_shared_weighted = _flatten_shared_grads(shared_lora_params, grad_dims, device)
            weighted_shared_grads.append(seg_shared_weighted)
            raw_shared_grads.append(
                seg_shared_weighted / task_weights["seg"] if task_weights["seg"] != 0 else torch.zeros_like(seg_shared_weighted)
            )
            _clear_parameter_grads(shared_lora_params)

            cnt_images, cnt_density = _to_device_cnt(batches["cnt"], device)
            cnt_gt_counts = cnt_density.flatten(2).sum(dim=2)
            pred_density, pred_counts = model("cnt", cnt_images)
            density_loss = F.mse_loss(pred_density, cnt_density, reduction="sum") / cnt_images.size(0)
            count_l1 = F.l1_loss(pred_counts, cnt_gt_counts)
            cnt_loss = density_loss + float(args.cnt_count_loss_weight) * count_l1
            (cnt_loss * task_weights["cnt"]).backward()
            cnt_shared_weighted = _flatten_shared_grads(shared_lora_params, grad_dims, device)
            weighted_shared_grads.append(cnt_shared_weighted)
            raw_shared_grads.append(
                cnt_shared_weighted / task_weights["cnt"] if task_weights["cnt"] != 0 else torch.zeros_like(cnt_shared_weighted)
            )
            _clear_parameter_grads(shared_lora_params)

            if use_pivrg:
                shared_grad_matrix = torch.stack(raw_shared_grads, dim=1)
                coeffs, pref, temp = _solve_pivrg(
                    shared_grad_matrix,
                    prev_scores,
                    bound=float(args.pivrg_bound),
                    mintemp=float(args.pivrg_mintemp),
                )
                combined_shared_grad = (shared_grad_matrix * coeffs.view(1, -1)).sum(dim=1) * shared_grad_matrix.shape[1]
                coeff_sum += coeffs.detach().cpu().numpy().astype(np.float64)
                pref_sum += pref.astype(np.float64)
                temp_sum += float(temp)
                pivrg_steps += 1
            else:
                combined_shared_grad = torch.stack(weighted_shared_grads, dim=1).sum(dim=1)

            _overwrite_shared_grads(shared_lora_params, grad_dims, combined_shared_grad)
            _sync_grads(trainable_params, world_size)
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(args.grad_clip_norm))
            optimizer.step()

            total = (
                float(task_weights["det"]) * float(det_loss.detach().item())
                + float(task_weights["seg"]) * float(seg_loss.detach().item())
                + float(task_weights["cnt"]) * float(cnt_loss.detach().item())
            )
            total_loss += total
            det_loss_sum += float(det_loss.detach().item())
            seg_loss_sum += float(seg_loss.detach().item())
            cnt_loss_sum += float(cnt_loss.detach().item())
            steps += 1

            if args.log_interval and step % int(args.log_interval) == 0:
                cnt_stats = _count_diag_stats(pred_density, cnt_density, pred_counts, cnt_gt_counts)
                print(
                    f"[train] epoch {epoch}/{args.epochs} step {step} | "
                    f"loss {total_loss / max(steps, 1):.4f} | "
                    f"det {det_loss_sum / max(steps, 1):.4f} seg {seg_loss_sum / max(steps, 1):.4f} "
                    f"cnt {cnt_loss_sum / max(steps, 1):.4f} | "
                    f"{_format_count_diag(cnt_stats)}"
                )

            if int(args.max_train_steps) > 0 and step >= int(args.max_train_steps):
                break

        train_sums = _reduce_train_sums([total_loss, det_loss_sum, seg_loss_sum, cnt_loss_sum, float(steps)], device)
        avg_train = train_sums[0] / max(train_sums[4], 1.0)
        avg_det = train_sums[1] / max(train_sums[4], 1.0)
        avg_seg = train_sums[2] / max(train_sums[4], 1.0)
        avg_cnt = train_sums[3] / max(train_sums[4], 1.0)

        print(
            f"[train] epoch {epoch}/{args.epochs} | train {avg_train:.4f} | "
            f"det {avg_det:.4f} seg {avg_seg:.4f} cnt {avg_cnt:.4f}"
        )
        if pivrg_steps > 0:
            coeff_avg = coeff_sum / float(pivrg_steps)
            pref_avg = pref_sum / float(pivrg_steps)
            temp_avg = temp_sum / float(pivrg_steps)
            print(
                f"[pivrg] epoch {epoch} | coeffs det={coeff_avg[0]:.6f} seg={coeff_avg[1]:.6f} cnt={coeff_avg[2]:.6f} | "
                f"pref det={pref_avg[0]:.6f} seg={pref_avg[1]:.6f} cnt={pref_avg[2]:.6f} | temp {temp_avg:.6f}"
            )

        should_validate = int(args.val_every) > 0 and epoch % int(args.val_every) == 0
        if not should_validate:
            continue

        val_det = _eval_det_loss(model, val_loaders["det"], device)
        val_seg, val_seg_miou = _eval_seg_loss(
            model,
            val_loaders["seg"],
            device,
            num_classes=int(args.seg_num_classes),
            max_steps=int(args.max_val_steps),
        )
        val_cnt, val_cnt_density, val_cnt_mae, val_cnt_total_mae = _eval_cnt_loss(
            model,
            val_loaders["cnt"],
            device,
            count_loss_weight=float(args.cnt_count_loss_weight),
            max_steps=int(args.max_val_steps),
        )
        val_ap50 = _eval_det_ap50_fast(
            model,
            val_loaders["det"],
            device,
            num_classes=det_num_classes,
            score_thresh=float(args.det_ap_score_thr),
        )
        combo_metric = float(val_ap50) + float(val_seg_miou) + 1.0 / max(float(val_cnt_mae), 1e-8)
        next_scores = _compute_scores(
            args,
            val_ap50=float(val_ap50),
            val_seg_miou=float(val_seg_miou),
            val_cnt_mae=float(val_cnt_mae),
        )
        print(
            f"[val] epoch {epoch} | "
            f"det {val_det:.4f} seg {val_seg:.4f} miou {val_seg_miou:.4f} "
            f"cnt {val_cnt:.4f} dens {val_cnt_density:.6e} mae {val_cnt_mae:.4f} total_mae {val_cnt_total_mae:.4f} | "
            f"ap50 {val_ap50:.4f} | combo {combo_metric:.6f} | "
            f"scores det={next_scores[0]:.6f} seg={next_scores[1]:.6f} cnt={next_scores[2]:.6f}"
        )
        prev_scores = next_scores

        if is_main_process and combo_metric > best_metric:
            best_metric = float(combo_metric)
            best_epoch = int(epoch)
            best_metrics = {
                "best_metric": float(best_metric),
                "best_epoch": int(best_epoch),
                "val_det_loss": float(val_det),
                "val_seg_loss": float(val_seg),
                "val_seg_miou": float(val_seg_miou),
                "val_cnt_loss": float(val_cnt),
                "val_cnt_density_mse": float(val_cnt_density),
                "val_cnt_mae": float(val_cnt_mae),
                "val_cnt_total_mae": float(val_cnt_total_mae),
                "val_ap50": float(val_ap50),
                "selected_metric": float(combo_metric),
                "score_det": float(next_scores[0]),
                "score_seg": float(next_scores[1]),
                "score_cnt": float(next_scores[2]),
            }
            best_state = _state_dict_cpu_clone(model.state_dict())
            print(f"[ckpt] cached best -> epoch {best_epoch} (combo {best_metric:.6f})")

    if is_main_process:
        if best_state is not None:
            model.load_state_dict(best_state)
            save_epoch = best_epoch
            metrics = best_metrics or {"best_metric": float(best_metric), "best_epoch": int(best_epoch)}
            print(f"[ckpt] saving cached best -> {best_path}")
        else:
            save_epoch = int(args.epochs)
            metrics = {
                "best_metric": float("nan"),
                "best_epoch": int(save_epoch),
                "score_det": float(prev_scores[0]),
                "score_seg": float(prev_scores[1]),
                "score_cnt": float(prev_scores[2]),
            }
            print(f"[ckpt] no validation best found; saving final model -> {best_path}")

        save_multitask_checkpoint(
            str(best_path),
            model=model,
            optimizer=optimizer,
            epoch=int(save_epoch),
            metrics=metrics,
            loss_weights=loss_weights,
            model_config=model.export_config(),
        )

    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

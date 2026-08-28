from __future__ import annotations

import argparse
import builtins
import importlib
import os
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.amp import GradScaler

    def _make_grad_scaler(device_type: str, enabled: bool) -> GradScaler:
        return GradScaler(device=device_type, enabled=enabled)

except Exception:
    from torch.cuda.amp import GradScaler  # type: ignore

    def _make_grad_scaler(device_type: str, enabled: bool) -> GradScaler:
        return GradScaler(enabled=enabled)

if __package__:
    from .models import MultiTaskModel, SharedDinoV3Backbone
    from .utils import TaskAffinityController, choose_primary, infinite_loader, parse_loss_weights, save_multitask_checkpoint
else:
    package_root = Path(__file__).resolve().parent.parent
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    models_module = importlib.import_module("113_selupdate.models")
    utils_module = importlib.import_module("113_selupdate.utils")
    MultiTaskModel = models_module.MultiTaskModel
    SharedDinoV3Backbone = models_module.SharedDinoV3Backbone
    TaskAffinityController = utils_module.TaskAffinityController
    choose_primary = utils_module.choose_primary
    infinite_loader = utils_module.infinite_loader
    parse_loss_weights = utils_module.parse_loss_weights
    save_multitask_checkpoint = utils_module.save_multitask_checkpoint


def _load_113_test_datasets():
    module = importlib.import_module("113_test.datasets")
    return module.build_det_loaders, module.build_seg_loaders, module.build_cnt_loaders


def _load_segmentation_utils():
    module = importlib.import_module("segmentation.utils")
    return module.per_class_iou_from_confusion, module.update_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Train the 113_selupdate multitask baseline")
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
    parser.add_argument("--grad-clip-norm", type=float, default=100.0)
    parser.add_argument("--loss-weights", type=str, default="15:8:1")
    parser.add_argument("--save-dir", type=str, default="runs/113_selupdate")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--primary-task", type=str, default=None, help="det|seg|cnt")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--det-ap-score-thr", type=float, default=0.0)
    parser.add_argument("--det-out-channels", type=int, default=256)
    parser.add_argument("--cnt-count-loss-weight", type=float, default=1.0)

    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--grad-checkpointing", dest="grad_checkpointing", action="store_true")
    checkpoint_group.add_argument("--no-grad-checkpointing", dest="grad_checkpointing", action="store_false")
    parser.set_defaults(grad_checkpointing=True)

    parser.add_argument("--lora", action="store_true", help="Enable LoRA finetuning on ViT FFN.")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-lr", type=float, default=None, help="LoRA learning rate (default: --lr)")
    parser.add_argument("--lora-weight-decay", type=float, default=0.0)

    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--backbone-lr-mult", type=float, default=0.1)
    parser.add_argument("--backbone-weight-decay", type=float, default=None)
    parser.add_argument("--det-lr", type=float, default=None)
    parser.add_argument("--seg-lr", type=float, default=None)
    parser.add_argument("--cnt-lr", type=float, default=None)
    parser.add_argument("--det-weight-decay", type=float, default=None)
    parser.add_argument("--seg-weight-decay", type=float, default=None)
    parser.add_argument("--cnt-weight-decay", type=float, default=None)
    parser.add_argument("--unfreeze-backbone", action="store_true")

    det_ft = parser.add_mutually_exclusive_group()
    det_ft.add_argument("--det-unfreeze-backbone", dest="det_unfreeze_backbone", action="store_true")
    det_ft.add_argument("--det-freeze-backbone", dest="det_unfreeze_backbone", action="store_false")
    parser.set_defaults(det_unfreeze_backbone=True)
    seg_ft = parser.add_mutually_exclusive_group()
    seg_ft.add_argument("--seg-full-finetune", dest="seg_full_finetune", action="store_true")
    seg_ft.add_argument("--seg-freeze-backbone", dest="seg_full_finetune", action="store_false")
    parser.set_defaults(seg_full_finetune=True)
    cnt_ft = parser.add_mutually_exclusive_group()
    cnt_ft.add_argument("--cnt-full-finetune", dest="cnt_full_finetune", action="store_true")
    cnt_ft.add_argument("--cnt-freeze-backbone", dest="cnt_full_finetune", action="store_false")
    parser.set_defaults(cnt_full_finetune=True)
    parser.add_argument("--cnt-backbone-grad-mult", type=float, default=1.0)

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

    parser.add_argument("--sel-prepos", type=int, default=100)
    parser.add_argument("--sel-affin-decay", type=float, default=1e-3)
    parser.add_argument("--sel-preference", type=str, default="None")
    parser.add_argument("--sel-convergence-iter", type=int, default=50)
    parser.add_argument("--sel-max-cluster-retries", type=int, default=10)
    parser.add_argument(
        "--sel-group-norm",
        action="store_true",
        help="Normalize each group loss by the number of tasks in the group before backward.",
    )
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


def _parse_sel_preference(value: str | None) -> float | None:
    raw = str(value or "").strip()
    if not raw or raw.lower() == "none":
        return None
    return float(raw)


def _build_initial_groups(tasks: list[str]) -> Dict[str, list[str]]:
    return {f"group{index + 1}": [task] for index, task in enumerate(tasks)}


def _broadcast_group_plan(
    controller: TaskAffinityController,
    train_groups: list[str],
    *,
    use_ddp: bool,
    rank: int,
) -> tuple[Dict[str, list[str]], list[str]]:
    payload = [controller.group_map, train_groups]
    if use_ddp and dist.is_initialized():
        if rank != 0:
            payload = [None, None]
        dist.broadcast_object_list(payload, src=0)
    group_map = {str(name): [str(member) for member in members] for name, members in (payload[0] or {}).items()}
    group_order = [str(item) for item in (payload[1] or [])]
    return group_map, group_order


def _compute_task_losses(
    model,
    batches: Dict[str, object],
    device: torch.device,
    *,
    amp: bool,
    autocast_device: str,
    cnt_count_loss_weight: float,
    enabled_tasks: list[str],
    group_tasks: list[str],
    cnt_backbone_grad_mult: float,
):
    det_batch = None
    seg_batch = None
    cnt_batch = None

    if "det" in enabled_tasks:
        det_batch = _to_device_det(batches["det"], device)
    if "seg" in enabled_tasks:
        seg_batch = _to_device_seg(batches["seg"], device)
    if "cnt" in enabled_tasks:
        cnt_batch = _to_device_cnt(batches["cnt"], device)

    with torch.amp.autocast(autocast_device, enabled=amp):
        losses, cnt_stats = model(
            "selective_train",
            det_batch=det_batch,
            seg_batch=seg_batch,
            cnt_batch=cnt_batch,
            group_tasks=group_tasks,
            cnt_count_loss_weight=float(cnt_count_loss_weight),
            cnt_backbone_grad_mult=float(cnt_backbone_grad_mult),
            collect_cnt_stats=True,
        )

    return losses, cnt_stats


def _reduce_loss_values(losses: Dict[str, torch.Tensor], *, use_ddp: bool) -> Dict[str, float]:
    reduced: Dict[str, float] = {}
    for name, loss in losses.items():
        value = loss.detach().float()
        if use_ddp and dist.is_initialized():
            reduced_value = value.clone()
            dist.all_reduce(reduced_value, op=dist.ReduceOp.SUM)
            reduced_value /= dist.get_world_size()
            reduced[name] = float(reduced_value.item())
        else:
            reduced[name] = float(value.item())
    return reduced


def _broadcast_module_state(module: torch.nn.Module, *, use_ddp: bool) -> None:
    if not use_ddp or not dist.is_initialized():
        return
    for parameter in module.parameters():
        dist.broadcast(parameter.data, src=0)
    for buffer in module.buffers():
        dist.broadcast(buffer.data, src=0)


def _allreduce_param_grads(params, *, use_ddp: bool, world_size: int) -> None:
    if not use_ddp or not dist.is_initialized():
        return
    scale = float(world_size)
    for parameter in params:
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(scale)


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
def _eval_det_loss(model, loader, device: torch.device, *, amp: bool) -> float:
    model.eval()
    model_for_state = model.module if hasattr(model, "module") else model
    model_for_state.detector.train()
    model_for_state.shared.eval()
    model_for_state.seg_head.eval()
    model_for_state.cnt_head.eval()
    total = 0.0
    samples = 0
    autocast_device = device.type if device.type in {"cuda", "cpu"} else "cuda"
    for images, targets in loader:
        images, targets = _to_device_det((images, targets), device)
        with torch.amp.autocast(autocast_device, enabled=amp):
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
def _eval_seg_loss(model, loader, device: torch.device, *, amp: bool, num_classes: int) -> tuple[float, float]:
    per_class_iou_from_confusion, update_confusion_matrix = _load_segmentation_utils()

    model.eval()
    total = 0.0
    samples = 0
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    autocast_device = device.type if device.type in {"cuda", "cpu"} else "cuda"
    for images, masks in loader:
        images, masks = _to_device_seg((images, masks), device)
        with torch.amp.autocast(autocast_device, enabled=amp):
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
def _eval_cnt_loss(model, loader, device: torch.device, *, amp: bool, count_loss_weight: float) -> tuple[float, float, float, float]:
    model.eval()
    total = 0.0
    total_density = 0.0
    total_count_mae = 0.0
    total_total_mae = 0.0
    samples = 0
    autocast_device = device.type if device.type in {"cuda", "cpu"} else "cuda"
    for images, density in loader:
        images, density = _to_device_cnt((images, density), device)
        gt_counts = density.flatten(2).sum(dim=2)
        with torch.amp.autocast(autocast_device, enabled=amp):
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


def main():
    args = parse_args()
    device, use_ddp, world_size, rank, local_rank = _init_distributed(args)
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
    loss_weight_map = {"det": float(loss_weights[0]), "seg": float(loss_weights[1]), "cnt": float(loss_weights[2])}

    det_train_backbone = bool(args.det_unfreeze_backbone)
    seg_train_backbone = bool(args.seg_full_finetune)
    cnt_train_backbone = bool(args.cnt_full_finetune)
    if args.lora or args.unfreeze_backbone:
        det_train_backbone = True
        seg_train_backbone = True
        cnt_train_backbone = True

    shared = SharedDinoV3Backbone(
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
        for parameter in shared.backbone.parameters():
            parameter.requires_grad = bool(any_train_backbone)

    model = MultiTaskModel(
        shared=shared,
        det_num_classes=det_num_classes,
        seg_num_classes=args.seg_num_classes,
        cnt_num_classes=args.cnt_num_classes,
        image_size=args.image_size,
        det_out_channels=args.det_out_channels,
        det_train_backbone=det_train_backbone,
        seg_train_backbone=seg_train_backbone,
        cnt_train_backbone=cnt_train_backbone,
    ).to(device)

    if use_ddp:
        ddp_device_ids = [local_rank] if device.type == "cuda" else None
        ddp_output_device = local_rank if device.type == "cuda" else None
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=ddp_device_ids,
            output_device=ddp_output_device,
            find_unused_parameters=True,
        )
    model_for_state = model.module if use_ddp else model

    backbone_lr = float(args.backbone_lr) if args.backbone_lr is not None else float(args.lr) * float(args.backbone_lr_mult)
    det_lr = float(args.det_lr) if args.det_lr is not None else float(args.lr)
    seg_lr = float(args.seg_lr) if args.seg_lr is not None else float(args.lr)
    cnt_lr = float(args.cnt_lr) if args.cnt_lr is not None else float(args.lr)
    backbone_weight_decay = (
        float(args.backbone_weight_decay)
        if args.backbone_weight_decay is not None
        else float(args.weight_decay)
    )
    det_weight_decay = float(args.det_weight_decay) if args.det_weight_decay is not None else float(args.weight_decay)
    seg_weight_decay = float(args.seg_weight_decay) if args.seg_weight_decay is not None else float(args.weight_decay)
    cnt_weight_decay = float(args.cnt_weight_decay) if args.cnt_weight_decay is not None else float(args.weight_decay)

    shared_backbone_params = list(model_for_state.shared.backbone.parameters())
    shared_backbone_ids = {id(parameter) for parameter in shared_backbone_params}
    backbone_params = [parameter for parameter in shared_backbone_params if parameter.requires_grad]
    det_head_params = [
        parameter
        for parameter in model_for_state.detector.parameters()
        if parameter.requires_grad and id(parameter) not in shared_backbone_ids
    ]
    seg_params = [parameter for parameter in model_for_state.seg_head.parameters() if parameter.requires_grad]
    cnt_head_params = [parameter for parameter in model_for_state.cnt_head.parameters() if parameter.requires_grad]

    optimizer_param_groups = []
    if backbone_params:
        if args.lora:
            lora_lr = float(args.lora_lr) if args.lora_lr is not None else float(args.lr)
            optimizer_param_groups.append(
                {"params": backbone_params, "lr": lora_lr, "weight_decay": float(args.lora_weight_decay)}
            )
        else:
            optimizer_param_groups.append(
                {"params": backbone_params, "lr": backbone_lr, "weight_decay": backbone_weight_decay}
            )
    if det_head_params:
        optimizer_param_groups.append({"params": det_head_params, "lr": det_lr, "weight_decay": det_weight_decay})
    if seg_params:
        optimizer_param_groups.append({"params": seg_params, "lr": seg_lr, "weight_decay": seg_weight_decay})
    if cnt_head_params:
        optimizer_param_groups.append({"params": cnt_head_params, "lr": cnt_lr, "weight_decay": cnt_weight_decay})
    if not optimizer_param_groups:
        raise RuntimeError("No trainable parameters.")

    trainable_params = [parameter for parameter in model_for_state.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(optimizer_param_groups)
    scaler = _make_grad_scaler(device.type, enabled=bool(args.amp))

    if is_main_process:
        adapter_params = sum(
            parameter.numel()
            for name, parameter in model_for_state.shared.named_parameters()
            if parameter.requires_grad and "lora_" in name
        )
        head_params = sum(
            parameter.numel()
            for name, parameter in model_for_state.named_parameters()
            if parameter.requires_grad and not name.startswith("shared.")
        )
        print(f"[train] trainable params: {sum(parameter.numel() for parameter in trainable_params)}")
        print(f"[train] lora params: {adapter_params}")
        print(f"[train] task head params: {head_params}")

    train_loaders = {"det": det_train_loader, "seg": seg_train_loader, "cnt": cnt_train_loader}
    val_loaders = {"det": det_val_loader, "seg": seg_val_loader, "cnt": cnt_val_loader}
    enabled_tasks = [task for task in ("det", "seg", "cnt") if task in train_loaders]
    train_lengths = {name: len(loader) for name, loader in train_loaders.items()}
    primary_task = choose_primary(train_lengths, args.primary_task)
    cyc_loaders = {name: infinite_loader(loader) for name, loader in train_loaders.items() if name != primary_task}

    controller = TaskAffinityController(
        enabled_tasks,
        initial_groups=_build_initial_groups(enabled_tasks),
        warmup_steps=int(args.sel_prepos),
        affin_decay=float(args.sel_affin_decay),
        preference=_parse_sel_preference(args.sel_preference),
        convergence_iter=int(args.sel_convergence_iter),
        max_cluster_retries=int(args.sel_max_cluster_retries),
    )

    total_epochs = int(args.epochs)
    validate_start_epoch = max(1, total_epochs - 49)
    if is_main_process:
        print(f"[train] validation starts at epoch {validate_start_epoch}/{total_epochs}")

    best_metric = -float("inf")
    best_epoch = 0
    best_path = save_dir / "best_combo.pt"
    best_state = None
    best_metrics = None
    global_step = 0
    last_group_signature = None

    for epoch in range(1, total_epochs + 1):
        if use_ddp:
            for sampler in (det_train_sampler, seg_train_sampler, cnt_train_sampler):
                if sampler is not None:
                    sampler.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        det_loss_sum = 0.0
        seg_loss_sum = 0.0
        cnt_loss_sum = 0.0
        steps = 0
        autocast_device = device.type if device.type in {"cuda", "cpu"} else "cuda"

        for step, primary_batch in enumerate(train_loaders[primary_task], start=1):
            global_step += 1
            batches = {primary_task: primary_batch}
            for name in train_loaders.keys():
                if name != primary_task:
                    batches[name] = next(cyc_loaders[name])

            if is_main_process:
                controller.maybe_recluster(global_step)
                signature = tuple((name, tuple(members)) for name, members in controller.group_map.items())
                if signature != last_group_signature:
                    print(f"[sel] step {global_step} groups: {controller.group_map}")
                    last_group_signature = signature
                current_group_names = controller.shuffled_group_names()
            else:
                current_group_names = []

            group_map, current_group_names = _broadcast_group_plan(
                controller,
                current_group_names,
                use_ddp=use_ddp,
                rank=rank,
            )
            controller.group_map = group_map
            controller.init_pre_loss()

            step_loss_sum = {task: 0.0 for task in enabled_tasks}
            step_cnt_stats = None

            for group_name in current_group_names:
                task_losses, cnt_stats = _compute_task_losses(
                    model,
                    batches,
                    device,
                    amp=bool(args.amp),
                    autocast_device=autocast_device,
                    cnt_count_loss_weight=float(args.cnt_count_loss_weight),
                    enabled_tasks=enabled_tasks,
                    group_tasks=controller.group_map[group_name],
                    cnt_backbone_grad_mult=float(args.cnt_backbone_grad_mult),
                )
                optimizer.zero_grad(set_to_none=True)
                group_loss = torch.zeros((), device=device)
                group_members = controller.group_map[group_name]
                for task_name in group_members:
                    group_loss = group_loss + float(loss_weight_map[task_name]) * task_losses[task_name]
                if bool(args.sel_group_norm):
                    group_loss = group_loss / max(len(group_members), 1)
                scaler.scale(group_loss).backward()
                scaler.unscale_(optimizer)
                if float(args.grad_clip_norm) > 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(args.grad_clip_norm))
                scaler.step(optimizer)
                scaler.update()

                reduced_losses = _reduce_loss_values(task_losses, use_ddp=use_ddp)
                if is_main_process:
                    controller.update(group_name, reduced_losses)
                for task_name, value in reduced_losses.items():
                    step_loss_sum[task_name] += float(value)
                if cnt_stats is not None:
                    step_cnt_stats = cnt_stats

            denom = max(len(current_group_names), 1)
            step_losses = {task: step_loss_sum[task] / denom for task in enabled_tasks}
            total = sum(float(loss_weight_map[task]) * step_losses[task] for task in enabled_tasks)
            total_loss += total
            det_loss_sum += float(step_losses.get("det", 0.0))
            seg_loss_sum += float(step_losses.get("seg", 0.0))
            cnt_loss_sum += float(step_losses.get("cnt", 0.0))
            steps += 1

            if args.log_interval and step % int(args.log_interval) == 0:
                message = (
                    f"[train] epoch {epoch}/{total_epochs} step {step} | "
                    f"loss {total_loss / max(steps, 1):.4f} | "
                    f"det {det_loss_sum / max(steps, 1):.4f} seg {seg_loss_sum / max(steps, 1):.4f} "
                    f"cnt {cnt_loss_sum / max(steps, 1):.4f} | groups {current_group_names}"
                )
                if step_cnt_stats is not None:
                    message += f" | {_format_count_diag(step_cnt_stats)}"
                print(message)

        print(f"[train] epoch {epoch}/{total_epochs} | train {total_loss / max(steps, 1):.4f}")

        should_validate = (
            int(args.val_every) > 0
            and epoch >= validate_start_epoch
            and epoch % int(args.val_every) == 0
        )
        if not should_validate:
            continue

        val_det = _eval_det_loss(model, val_loaders["det"], device, amp=bool(args.amp))
        val_seg, val_seg_miou = _eval_seg_loss(
            model,
            val_loaders["seg"],
            device,
            amp=bool(args.amp),
            num_classes=int(args.seg_num_classes),
        )
        val_cnt, val_cnt_density, val_cnt_mae, val_cnt_total_mae = _eval_cnt_loss(
            model,
            val_loaders["cnt"],
            device,
            amp=bool(args.amp),
            count_loss_weight=float(args.cnt_count_loss_weight),
        )
        val_ap50 = _eval_det_ap50_fast(
            model,
            val_loaders["det"],
            device,
            num_classes=det_num_classes,
            score_thresh=float(args.det_ap_score_thr),
        )
        combo_metric = float(val_ap50) + float(val_seg_miou) + 1.0 / max(float(val_cnt_mae), 1e-8)
        print(
            f"[val] epoch {epoch} | "
            f"det {val_det:.4f} seg {val_seg:.4f} miou {val_seg_miou:.4f} "
            f"cnt {val_cnt:.4f} dens {val_cnt_density:.6e} mae {val_cnt_mae:.4f} total_mae {val_cnt_total_mae:.4f} | "
            f"ap50 {val_ap50:.4f} | combo {combo_metric:.6f}"
        )

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
            }
            best_state = _state_dict_cpu_clone(model_for_state.state_dict())
            print(f"[ckpt] new best cached (epoch {best_epoch}, combo {best_metric:.6f})")

    if is_main_process:
        if best_state is not None:
            model_for_state.load_state_dict(best_state)
            save_epoch = best_epoch
            metrics = best_metrics or {"best_metric": float(best_metric), "best_epoch": int(best_epoch)}
            print(f"[ckpt] saving cached best -> {best_path}")
        else:
            save_epoch = total_epochs
            metrics = {
                "best_metric": float("nan"),
                "best_epoch": int(save_epoch),
                "selected_metric": float("nan"),
            }
            print(f"[ckpt] no validation best found; saving final model -> {best_path}")

        save_multitask_checkpoint(
            str(best_path),
            model=model_for_state,
            optimizer=optimizer,
            epoch=int(save_epoch),
            metrics=metrics,
            loss_weights=loss_weights,
            model_config=model_for_state.export_config(),
            sel_config={
                "sel_prepos": int(args.sel_prepos),
                "sel_affin_decay": float(args.sel_affin_decay),
                "sel_preference": _parse_sel_preference(args.sel_preference),
                "sel_convergence_iter": int(args.sel_convergence_iter),
                "sel_max_cluster_retries": int(args.sel_max_cluster_retries),
                "sel_group_norm": bool(args.sel_group_norm),
            },
            train_strategy="selective_task_group_updates",
        )

    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

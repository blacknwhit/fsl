from __future__ import annotations

import argparse
import builtins
import importlib
import math
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
    from torch.amp import GradScaler

    def _make_grad_scaler(device_type: str, enabled: bool) -> GradScaler:
        return GradScaler(device=device_type, enabled=enabled)

except Exception:
    from torch.cuda.amp import GradScaler  # type: ignore

    def _make_grad_scaler(device_type: str, enabled: bool) -> GradScaler:
        return GradScaler(enabled=enabled)

try:
    from .models import MultiTaskModel, SharedDinoV3Backbone
    from .utils import choose_primary, infinite_loader, save_multitask_checkpoint, parse_int_list
except ImportError:
    from models import MultiTaskModel, SharedDinoV3Backbone
    from utils import choose_primary, infinite_loader, save_multitask_checkpoint, parse_int_list


TASK_WEIGHTS = {"det": 15.0, "seg": 8.0, "cnt": 1.0}
TASK_LB_STEP_SCALE = 1.0 / len(TASK_WEIGHTS)


def _load_113_test_datasets():
    module = importlib.import_module("113_test.datasets")
    return module.build_det_loaders, module.build_seg_loaders, module.build_cnt_loaders


def _load_segmentation_utils():
    module = importlib.import_module("segmentation.utils")
    return module.per_class_iou_from_confusion, module.update_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Train the 113_mlore multitask comparison model")
    parser.add_argument("--model-name", type=str, default="dinov3_vitl16")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--backbone-checkpoint", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--val-every", type=int, default=1)
    parser.add_argument(
        "--validate-last-n-epochs",
        type=int,
        default=0,
        help="Only run validation during the last N epochs. 0 means validate across the whole training.",
    )
    parser.add_argument("--det-batch-size", type=int, default=2)
    parser.add_argument("--seg-batch-size", type=int, default=2)
    parser.add_argument("--cnt-batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=100.0)
    parser.add_argument("--save-dir", type=str, default="runs/113_mlore")
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

    parser.add_argument("--mlore-decoder-dim", type=int, default=256)
    parser.add_argument("--mlore-rank-list", type=str, default="8,16,24,32,40,48")
    parser.add_argument("--mlore-topk", type=int, default=4)
    parser.add_argument("--mlore-task-rank", type=int, default=32)
    parser.add_argument("--mlore-pre-softmax", action="store_true")
    parser.add_argument("--mlore-load-balancing-weight", type=float, default=3e-4)
    parser.add_argument("--mlore-select-layers", type=str, default="23")
    parser.add_argument("--mlore-num-stages", type=int, default=1)
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
    pred_dens: torch.Tensor,
    gt_dens: torch.Tensor,
    pred_counts: torch.Tensor,
    gt_counts: torch.Tensor,
) -> Dict[str, float]:
    pred_dens_f = pred_dens.detach().float()
    gt_dens_f = gt_dens.detach().float()
    pred_counts_f = pred_counts.detach().float()
    gt_counts_f = gt_counts.detach().float()

    pixels = int(pred_dens_f.shape[-2] * pred_dens_f.shape[-1])
    return {
        "pred_dens_mean": float(pred_dens_f.mean().item()),
        "gt_dens_mean": float(gt_dens_f.mean().item()),
        "pred_count_mean": float(pred_counts_f.mean().item()),
        "gt_count_mean": float(gt_counts_f.mean().item()),
        "count_mae": float((pred_counts_f - gt_counts_f).abs().mean().item()),
        "pred_total_mean": float(pred_counts_f.sum(dim=1).mean().item()),
        "gt_total_mean": float(gt_counts_f.sum(dim=1).mean().item()),
        "pixels": float(pixels),
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

    img_counter = 0
    for images, targets in loader:
        images = [img.to(device, non_blocking=True) for img in images]
        outputs = model("det", images)

        for out, target in zip(outputs, targets):
            default_img_id = dist_rank * 10_000_000_000 + img_counter
            image_id = int(target.get("image_id", torch.tensor([default_img_id])).item())
            img_counter += 1

            gt_boxes = target["boxes"].detach().cpu().numpy()
            gt_labels = target["labels"].detach().cpu().numpy().astype(int)
            for box, cls in zip(gt_boxes, gt_labels):
                gts_by_cls[cls].setdefault(image_id, []).append(box)

            pred_boxes = out["boxes"].detach().cpu().numpy()
            pred_labels = out["labels"].detach().cpu().numpy().astype(int)
            pred_scores = out["scores"].detach().cpu().numpy()
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
    model_for_state.shared.decoder.eval()
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
    seg_train_ds, seg_val_ds, seg_train_loader_base, seg_val_loader_base = build_seg_loaders(
        train_dir=args.seg_train_dir,
        val_dir=args.seg_val_dir,
        image_size=args.image_size,
        batch_size=args.seg_batch_size,
        num_workers=args.num_workers,
    )
    cnt_train_ds, cnt_val_ds, cnt_train_loader_base, cnt_val_loader_base = build_cnt_loaders(
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
        batch_size=args.det_batch_size,
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
        batch_size=args.seg_batch_size,
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
        batch_size=args.cnt_batch_size,
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
    decoder_grad_checkpointing = bool(args.grad_checkpointing)
    if use_ddp and decoder_grad_checkpointing:
        if is_main_process:
            print("[warn] disabling decoder grad checkpointing under DDP auto sync for stability")
        decoder_grad_checkpointing = False

    shared = SharedDinoV3Backbone(
        model_name=args.model_name,
        image_size=args.image_size,
        checkpoint_path=args.backbone_checkpoint,
        mlore_decoder_dim=args.mlore_decoder_dim,
        mlore_rank_list=parse_int_list(args.mlore_rank_list),
        mlore_topk=args.mlore_topk,
        mlore_task_rank=args.mlore_task_rank,
        mlore_pre_softmax=bool(args.mlore_pre_softmax),
        mlore_load_balancing_weight=args.mlore_load_balancing_weight,
        mlore_select_layers=parse_int_list(args.mlore_select_layers),
        mlore_num_stages=args.mlore_num_stages,
        grad_checkpointing=decoder_grad_checkpointing,
    )
    model = MultiTaskModel(
        shared=shared,
        det_num_classes=det_num_classes,
        seg_num_classes=args.seg_num_classes,
        cnt_num_classes=args.cnt_num_classes,
        image_size=args.image_size,
        det_out_channels=args.det_out_channels,
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

    trainable_params = [param for param in model_for_state.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    scaler = _make_grad_scaler(device.type, enabled=bool(args.amp))

    if is_main_process:
        total_params = sum(param.numel() for param in trainable_params)
        print(f"[train] trainable params: {total_params}")
        print("[train] mode: sequential_backward_per_task, grad_sync=ddp_auto, lb_mode=task_local_avg")

    train_loaders = {"det": det_train_loader, "seg": seg_train_loader, "cnt": cnt_train_loader}
    val_loaders = {"det": det_val_loader, "seg": seg_val_loader, "cnt": cnt_val_loader}
    train_lengths = {name: len(loader) for name, loader in train_loaders.items()}
    primary_task = choose_primary(train_lengths, args.primary_task)
    cyc_loaders = {name: infinite_loader(loader) for name, loader in train_loaders.items() if name != primary_task}

    best_metric = -float("inf")
    best_epoch = 0
    best_path = save_dir / "best_combo.pt"
    best_state = None
    best_metrics = None
    validate_last_n = max(int(args.validate_last_n_epochs), 0)

    for epoch in range(1, int(args.epochs) + 1):
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

        for step, primary_batch in enumerate(train_loaders[primary_task], start=1):
            batches = {primary_task: primary_batch}
            for name in train_loaders.keys():
                if name != primary_task:
                    batches[name] = next(cyc_loaders[name])

            optimizer.zero_grad(set_to_none=True)
            autocast_device = device.type if device.type in {"cuda", "cpu"} else "cuda"

            det_images, det_targets = _to_device_det(batches["det"], device)
            with torch.amp.autocast(autocast_device, enabled=bool(args.amp)):
                det_loss_dict, det_lb_loss = model("det", det_images, det_targets, return_lb=True)
                det_loss = sum(det_loss_dict.values())
            scaler.scale(det_loss * TASK_WEIGHTS["det"] + det_lb_loss * TASK_LB_STEP_SCALE).backward()

            seg_images, seg_masks = _to_device_seg(batches["seg"], device)
            with torch.amp.autocast(autocast_device, enabled=bool(args.amp)):
                seg_logits, seg_lb_loss = model("seg", seg_images, return_lb=True)
                seg_loss = F.cross_entropy(seg_logits, seg_masks)
            scaler.scale(seg_loss * TASK_WEIGHTS["seg"] + seg_lb_loss * TASK_LB_STEP_SCALE).backward()

            cnt_images, cnt_density = _to_device_cnt(batches["cnt"], device)
            cnt_gt_counts = cnt_density.flatten(2).sum(dim=2)
            with torch.amp.autocast(autocast_device, enabled=bool(args.amp)):
                pred_density, pred_counts, cnt_lb_loss = model("cnt", cnt_images, return_lb=True)
                density_loss = F.mse_loss(pred_density, cnt_density, reduction="sum") / cnt_images.size(0)
                count_l1 = F.l1_loss(pred_counts, cnt_gt_counts)
                cnt_loss = density_loss + float(args.cnt_count_loss_weight) * count_l1
            scaler.scale(cnt_loss * TASK_WEIGHTS["cnt"] + cnt_lb_loss * TASK_LB_STEP_SCALE).backward()

            scaler.unscale_(optimizer)
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float(args.grad_clip_norm))

            scaler.step(optimizer)
            scaler.update()

            total = (
                TASK_WEIGHTS["det"] * float(det_loss.detach().item())
                + TASK_WEIGHTS["seg"] * float(seg_loss.detach().item())
                + TASK_WEIGHTS["cnt"] * float(cnt_loss.detach().item())
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

        print(f"[train] epoch {epoch}/{args.epochs} | train {total_loss / max(steps, 1):.4f}")

        validation_window_start = 1
        if validate_last_n > 0:
            validation_window_start = max(int(args.epochs) - validate_last_n + 1, 1)
        should_validate = (
            int(args.val_every) > 0
            and epoch % int(args.val_every) == 0
            and epoch >= validation_window_start
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
            print(f"[ckpt] cached best -> epoch {best_epoch} (combo {best_metric:.6f})")

    if is_main_process:
        if best_state is not None:
            model_for_state.load_state_dict(best_state)
            save_multitask_checkpoint(
                str(best_path),
                model=model_for_state,
                optimizer=optimizer,
                epoch=int(best_epoch),
                metrics=best_metrics or {"best_metric": float(best_metric), "best_epoch": int(best_epoch)},
                loss_weights=(TASK_WEIGHTS["det"], TASK_WEIGHTS["seg"], TASK_WEIGHTS["cnt"]),
                model_config=model_for_state.export_config(),
            )
            print(f"[ckpt] saved cached best -> {best_path} (combo {best_metric:.6f})")
        else:
            save_multitask_checkpoint(
                str(best_path),
                model=model_for_state,
                optimizer=optimizer,
                epoch=int(args.epochs),
                metrics={"best_metric": float("nan"), "best_epoch": int(args.epochs)},
                loss_weights=(TASK_WEIGHTS["det"], TASK_WEIGHTS["seg"], TASK_WEIGHTS["cnt"]),
                model_config=model_for_state.export_config(),
            )
            print(f"[ckpt] validation did not run; saved final weights -> {best_path}")

    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

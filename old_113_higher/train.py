from __future__ import annotations

import argparse
import os
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torch.utils.data.distributed import DistributedSampler

try:
    import higher
except ImportError:
    _HIGHER_DIR = Path(__file__).resolve().parents[1] / "higher-main"
    if _HIGHER_DIR.exists():
        sys.path.insert(0, str(_HIGHER_DIR))
        import higher
    else:
        raise

from .datasets import build_cnt_loaders, build_det_loaders, build_seg_loaders
from .models import MultiTaskModel, SharedDinoV3Backbone
from .utils import choose_primary, infinite_loader, save_multitask_checkpoint
from .weight_net import JointWeightGenerator
from object_detection.dataset import collate_fn
from segmentation.utils import per_class_iou_from_confusion, update_confusion_matrix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="113_higher: higher-based multitask meta-weighting")

    p.add_argument("--model-name", type=str, default="dinov3_vitl16")
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--backbone-checkpoint", type=str, default=None)
    p.add_argument("--unfreeze-backbone", action="store_true")
    p.add_argument("--use-lora-moe", action="store_true")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--num-experts-private", type=int, default=3)
    p.add_argument("--num-experts-shared", type=int, default=6)
    p.add_argument("--moe-k-private", type=int, default=2)
    p.add_argument("--moe-k-shared", type=int, default=2)
    gc = p.add_mutually_exclusive_group()
    gc.add_argument("--grad-checkpointing", dest="grad_checkpointing", action="store_true")
    gc.add_argument("--no-grad-checkpointing", dest="grad_checkpointing", action="store_false")
    p.set_defaults(grad_checkpointing=True)

    p.add_argument("--det-data-root", type=str, required=True)
    p.add_argument("--det-train-ann", type=str, default=None)
    p.add_argument("--det-val-ann", type=str, default=None)
    p.add_argument("--det-train-img-dir", type=str, default=None)
    p.add_argument("--det-val-img-dir", type=str, default=None)
    p.add_argument("--det-num-classes", type=int, default=None)
    p.add_argument("--det-batch-size", type=int, default=2)
    p.add_argument("--det-ap-score-thr", type=float, default=0.0)

    p.add_argument("--seg-train-dir", type=str, required=True)
    p.add_argument("--seg-val-dir", type=str, required=True)
    p.add_argument("--seg-num-classes", type=int, default=11)
    p.add_argument("--seg-batch-size", type=int, default=2)

    p.add_argument("--cnt-data-root", type=str, required=True)
    p.add_argument("--cnt-train-dir", type=str, default=None)
    p.add_argument("--cnt-val-dir", type=str, default=None)
    p.add_argument("--cnt-num-classes", type=int, default=8)
    p.add_argument("--cnt-batch-size", type=int, default=2)
    p.add_argument("--cnt-count-loss-weight", type=float, default=1.0)
    p.add_argument("--cnt-backbone-grad-mult", type=float, default=1.0)
    aspect = p.add_mutually_exclusive_group()
    aspect.add_argument("--cnt-keep-aspect", dest="cnt_keep_aspect", action="store_true")
    aspect.add_argument("--cnt-no-keep-aspect", dest="cnt_keep_aspect", action="store_false")
    p.set_defaults(cnt_keep_aspect=True)

    p.add_argument("--stage1-epochs", type=int, default=100)
    p.add_argument("--stage2-epochs", type=int, default=50)
    p.add_argument("--meta-split", type=float, default=0.2)
    p.add_argument("--meta-seed", type=int, default=42)
    p.add_argument("--stage1-inner-lr", type=float, default=1e-4)
    p.add_argument("--stage1-phi-lr", type=float, default=1e-3)
    p.add_argument("--stage2-lr", type=float, default=1e-4)
    p.add_argument("--stage2-weight-decay", type=float, default=1e-4)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", type=str, required=True)
    p.add_argument("--grad-clip-norm", type=float, default=100.0)
    p.add_argument("--max-train-steps", type=int, default=0)
    p.add_argument("--max-val-steps", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument("--select-best-from-stage2", action="store_true")
    return p.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _setup_dist(args: argparse.Namespace) -> tuple[torch.device, bool, int, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_ddp = world_size > 1

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        if use_ddp:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    if use_ddp and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend)

    is_main = rank == 0
    return device, use_ddp, rank, local_rank, world_size, is_main


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


def _state_dict_cpu_clone(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
        else:
            out[k] = deepcopy(v)
    return out


def _normalize_grad(vec: torch.Tensor) -> torch.Tensor:
    vec = torch.nan_to_num(vec.detach().float(), nan=0.0, posinf=1e6, neginf=0.0)
    norm = vec.norm()
    if not torch.isfinite(norm):
        return torch.zeros_like(vec)
    return vec / (norm + 1e-12)


def _flatten_grads(grads: Sequence[torch.Tensor | None], params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    ref_device = params[0].device if params else torch.device("cpu")
    for grad, param in zip(grads, params):
        if grad is None:
            parts.append(torch.zeros(param.numel(), device=ref_device))
        else:
            parts.append(grad.reshape(-1).float())
    if not parts:
        return torch.zeros(0, device=ref_device)
    return torch.cat(parts, dim=0)


def _sync_mean_tensor(tensor: torch.Tensor, use_ddp: bool, world_size: int) -> torch.Tensor:
    if not use_ddp:
        return tensor
    out = tensor.detach().clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out /= float(world_size)
    return out


def _sync_mean_tensor_list(tensors: Sequence[torch.Tensor], use_ddp: bool, world_size: int) -> list[torch.Tensor]:
    return [_sync_mean_tensor(t, use_ddp, world_size) for t in tensors]


def _broadcast_module_state(module: torch.nn.Module, src: int = 0) -> None:
    if not dist.is_initialized():
        return
    for tensor in module.state_dict().values():
        if torch.is_tensor(tensor):
            dist.broadcast(tensor, src=src)


def _split_dataset(ds: Dataset, meta_split: float, seed: int) -> tuple[Subset, Subset]:
    n_total = len(ds)
    n_meta = max(1, int(round(n_total * meta_split)))
    n_inner = n_total - n_meta
    if n_inner <= 0:
        raise ValueError(f"meta split too large: total={n_total}, meta={n_meta}")
    gen = torch.Generator().manual_seed(int(seed))
    inner_ds, meta_ds = random_split(ds, [n_inner, n_meta], generator=gen)
    return inner_ds, meta_ds


def _make_loader(
    ds: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    use_ddp: bool,
    rank: int,
    world_size: int,
    shuffle: bool,
    drop_last: bool = False,
    collate_fn_override=None,
    persistent_workers: bool | None = None,
    multiprocessing_context: str | None = None,
    prefetch_factor: int | None = None,
):
    sampler = None
    if use_ddp:
        sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last,
        )
    kwargs = {
        "batch_size": int(batch_size),
        "sampler": sampler,
        "shuffle": bool(shuffle) and sampler is None,
        "num_workers": int(num_workers),
        "pin_memory": True,
        "drop_last": bool(drop_last),
    }
    if collate_fn_override is not None:
        kwargs["collate_fn"] = collate_fn_override
    if num_workers > 0 and persistent_workers is not None:
        kwargs["persistent_workers"] = bool(persistent_workers)
    if num_workers > 0 and multiprocessing_context is not None:
        kwargs["multiprocessing_context"] = multiprocessing_context
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    loader = DataLoader(ds, **kwargs)
    return loader, sampler


@torch.no_grad()
def _count_diag_stats(pred_dens: torch.Tensor, gt_dens: torch.Tensor, pred_counts: torch.Tensor, gt_counts: torch.Tensor) -> Dict[str, float]:
    pred_dens_f = pred_dens.detach().float()
    gt_dens_f = gt_dens.detach().float()
    pred_counts_f = pred_counts.detach().float()
    gt_counts_f = gt_counts.detach().float()
    pred_dens_mean = float(pred_dens_f.mean().item())
    gt_dens_mean = float(gt_dens_f.mean().item())
    pred_count_mean = float(pred_counts_f.mean().item())
    gt_count_mean = float(gt_counts_f.mean().item())
    pixels = int(pred_dens_f.shape[-2] * pred_dens_f.shape[-1])
    eps = 1e-12
    return {
        "pred_dens_mean": pred_dens_mean,
        "gt_dens_mean": gt_dens_mean,
        "dens_ratio": pred_dens_mean / max(gt_dens_mean, eps),
        "pred_count_mean": pred_count_mean,
        "gt_count_mean": gt_count_mean,
        "count_ratio": pred_count_mean / max(gt_count_mean, eps),
        "count_mae": float((pred_counts_f - gt_counts_f).abs().mean().item()),
        "pred_total_mean": float(pred_counts_f.sum(dim=1).mean().item()),
        "gt_total_mean": float(gt_counts_f.sum(dim=1).mean().item()),
        "pixels": float(pixels),
        "pred_count_from_dens_mean": pred_dens_mean * float(pixels),
    }


def _format_count_diag(stats: Dict[str, float]) -> str:
    return (
        f"dens(mean {stats['pred_dens_mean']:.6e}/{stats['gt_dens_mean']:.6e}, ratio {stats['dens_ratio']:.3e}) | "
        f"count(mean {stats['pred_count_mean']:.3f}/{stats['gt_count_mean']:.3f}, "
        f"ratio {stats['count_ratio']:.3e}, mae {stats['count_mae']:.3f}, "
        f"total {stats['pred_total_mean']:.3f}/{stats['gt_total_mean']:.3f}) | "
        f"pixels {int(stats['pixels'])} | pred_count_from_dens_mean {stats['pred_count_from_dens_mean']:.3f}"
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


def _named_param_map(module: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    return dict(module.named_parameters())


def _extract_grad_vector(
    loss: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
    *,
    retain_graph: bool = False,
) -> torch.Tensor:
    grads = torch.autograd.grad(loss, list(params), retain_graph=retain_graph, create_graph=False, allow_unused=True)
    return _normalize_grad(_flatten_grads(grads, params))


def _extract_task_grads(
    loss: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
) -> tuple[torch.Tensor | None, ...]:
    grads = torch.autograd.grad(loss, list(params), retain_graph=False, create_graph=False, allow_unused=True)
    out = []
    for grad in grads:
        out.append(None if grad is None else grad.detach())
    return tuple(out)


def _get_stage1_fast_named_params(
    model: torch.nn.Module,
    fmodel: torch.nn.Module,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    names = [name for name, _ in model.named_parameters()]
    fast_params = list(fmodel.fast_params)
    if len(names) != len(fast_params):
        raise RuntimeError(f"fast param size mismatch: model={len(names)} fast={len(fast_params)}")
    return {name: fast_params[idx] for idx, name in enumerate(names)}, names


def _copy_fast_params_to_model(
    model: torch.nn.Module,
    fmodel: torch.nn.Module,
    *,
    use_ddp: bool,
    world_size: int,
) -> None:
    named_fast, names = _get_stage1_fast_named_params(model, fmodel)
    named_model = _named_param_map(model)
    synced = _sync_mean_tensor_list([named_fast[name].detach() for name in names], use_ddp, world_size)
    with torch.no_grad():
        for name, tensor in zip(names, synced):
            if name not in named_model:
                raise KeyError(f"Missing model parameter during fast-weight copy: {name}")
            named_model[name].copy_(tensor)


def _copy_fast_param_tensors_to_model(
    model: torch.nn.Module,
    names: Sequence[str],
    tensors: Sequence[torch.Tensor],
    *,
    use_ddp: bool,
    world_size: int,
) -> None:
    named_model = _named_param_map(model)
    synced = _sync_mean_tensor_list([tensor.detach() for tensor in tensors], use_ddp, world_size)
    with torch.no_grad():
        for name, tensor in zip(names, synced):
            if name not in named_model:
                raise KeyError(f"Missing model parameter during fast-weight copy: {name}")
            named_model[name].copy_(tensor)


def _build_stage1_fast_model(
    model: torch.nn.Module,
    *,
    device: torch.device,
    weights: torch.Tensor,
    theta_params: Sequence[torch.nn.Parameter],
    theta_name_to_index: dict[str, int],
    full_param_names: Sequence[str],
    det_task_grads: Sequence[torch.Tensor | None],
    seg_task_grads: Sequence[torch.Tensor | None],
    cnt_task_grads: Sequence[torch.Tensor | None],
    stage1_lr: float,
):
    fmodel = higher.monkeypatch(
        model,
        device=device,
        copy_initial_weights=True,
        track_higher_grads=True,
    )
    fast_params = list(fmodel.fast_params)
    new_fast_params: list[torch.Tensor] = []
    for full_idx, name in enumerate(full_param_names):
        fast_param = fast_params[full_idx]
        theta_idx = theta_name_to_index.get(name)
        if theta_idx is None:
            new_fast_params.append(fast_param)
            continue
        upd = None
        g_det = det_task_grads[theta_idx]
        g_seg = seg_task_grads[theta_idx]
        g_cnt = cnt_task_grads[theta_idx]
        if g_det is not None:
            upd = weights[0] * g_det if upd is None else upd + weights[0] * g_det
        if g_seg is not None:
            upd = weights[1] * g_seg if upd is None else upd + weights[1] * g_seg
        if g_cnt is not None:
            upd = weights[2] * g_cnt if upd is None else upd + weights[2] * g_cnt
        if upd is None:
            new_fast_params.append(fast_param)
        else:
            new_fast_params.append(fast_param - float(stage1_lr) * upd)
    fmodel.update_params(new_fast_params)
    del fast_params, new_fast_params
    return fmodel


def _build_model(args: argparse.Namespace, det_num_classes: int, device: torch.device) -> MultiTaskModel:
    shared = SharedDinoV3Backbone(
        model_name=args.model_name,
        image_size=args.image_size,
        checkpoint_path=args.backbone_checkpoint,
        use_lora_moe=bool(args.use_lora_moe),
        backbone_trainable=bool(args.unfreeze_backbone),
        task_num=3,
        lora_rank=int(args.lora_rank),
        num_experts_private=int(args.num_experts_private),
        num_experts_shared=int(args.num_experts_shared),
        moe_k_private=int(args.moe_k_private),
        moe_k_shared=int(args.moe_k_shared),
        grad_checkpointing=bool(args.grad_checkpointing),
    )
    model = MultiTaskModel(
        shared=shared,
        det_num_classes=det_num_classes,
        seg_num_classes=int(args.seg_num_classes),
        cnt_num_classes=int(args.cnt_num_classes),
        image_size=int(args.image_size),
        det_train_backbone=bool(args.unfreeze_backbone),
        seg_train_backbone=bool(args.unfreeze_backbone),
        cnt_train_backbone=bool(args.unfreeze_backbone),
    )
    return model.to(device)


def _build_theta_param_groups(model: MultiTaskModel, args: argparse.Namespace) -> tuple[list[dict], list[torch.nn.Parameter]]:
    shared_backbone_params = list(model.shared.backbone.parameters())
    shared_backbone_ids = {id(p) for p in shared_backbone_params}
    backbone_lr = float(args.stage2_lr) * float(args.backbone_lr_mult)
    groups: list[dict] = []

    backbone_params = [p for p in shared_backbone_params if p.requires_grad]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": float(args.stage2_weight_decay)})

    lora_params: list[torch.nn.Parameter] = []
    if bool(args.use_lora_moe):
        for lora_moe in model.shared.lora_moes:
            lora_params.extend([p for p in lora_moe.parameters() if p.requires_grad])
        if lora_params:
            groups.append({"params": lora_params, "lr": float(args.stage2_lr), "weight_decay": float(args.stage2_weight_decay)})

    lora_ids = {id(p) for p in lora_params}
    det_params = [
        p
        for p in model.detector.parameters()
        if p.requires_grad and id(p) not in shared_backbone_ids and id(p) not in lora_ids
    ]
    seg_params = [p for p in model.seg_head.parameters() if p.requires_grad]
    cnt_params = [p for p in model.cnt_head.parameters() if p.requires_grad]
    if det_params:
        groups.append({"params": det_params, "lr": float(args.stage2_lr), "weight_decay": float(args.stage2_weight_decay)})
    if seg_params:
        groups.append({"params": seg_params, "lr": float(args.stage2_lr), "weight_decay": float(args.stage2_weight_decay)})
    if cnt_params:
        groups.append({"params": cnt_params, "lr": float(args.stage2_lr), "weight_decay": float(args.stage2_weight_decay)})

    theta_params = [p for g in groups for p in g["params"]]
    if not theta_params:
        raise RuntimeError("No trainable theta parameters found.")
    return groups, theta_params


def _compute_losses(
    model: torch.nn.Module,
    det_batch,
    seg_batch,
    cnt_batch,
    *,
    device: torch.device,
    cnt_count_loss_weight: float,
    cnt_backbone_grad_mult: float,
    collect_cnt_stats: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float] | None]:
    det_images, det_targets = _to_device_det(det_batch, device)
    seg_imgs, seg_masks = _to_device_seg(seg_batch, device)
    cnt_imgs, cnt_dens = _to_device_cnt(cnt_batch, device)
    cnt_gt_counts = cnt_dens.flatten(2).sum(dim=2)

    det_loss_dict = model("det", det_images, det_targets)
    det_loss = sum(det_loss_dict.values())

    seg_logits = model("seg", seg_imgs)
    seg_loss = F.cross_entropy(seg_logits, seg_masks)

    pred_dens, pred_counts = model("cnt", cnt_imgs, cnt_backbone_grad_mult=float(cnt_backbone_grad_mult))
    dens_loss = F.mse_loss(pred_dens, cnt_dens, reduction="sum") / cnt_imgs.size(0)
    cnt_l1 = F.l1_loss(pred_counts, cnt_gt_counts)
    cnt_loss = dens_loss + float(cnt_count_loss_weight) * cnt_l1

    cnt_stats = None
    if collect_cnt_stats:
        cnt_stats = _count_diag_stats(pred_dens, cnt_dens, pred_counts, cnt_gt_counts)
    return det_loss, seg_loss, cnt_loss, cnt_stats


def _compute_det_loss_only(model: torch.nn.Module, det_batch, *, device: torch.device) -> torch.Tensor:
    det_images, det_targets = _to_device_det(det_batch, device)
    det_loss_dict = model("det", det_images, det_targets)
    return sum(det_loss_dict.values())


def _compute_seg_loss_only(model: torch.nn.Module, seg_batch, *, device: torch.device) -> torch.Tensor:
    seg_imgs, seg_masks = _to_device_seg(seg_batch, device)
    seg_logits = model("seg", seg_imgs)
    return F.cross_entropy(seg_logits, seg_masks)


def _compute_cnt_loss_only(
    model: torch.nn.Module,
    cnt_batch,
    *,
    device: torch.device,
    cnt_count_loss_weight: float,
    cnt_backbone_grad_mult: float,
    collect_cnt_stats: bool = False,
) -> tuple[torch.Tensor, Dict[str, float] | None]:
    cnt_imgs, cnt_dens = _to_device_cnt(cnt_batch, device)
    cnt_gt_counts = cnt_dens.flatten(2).sum(dim=2)
    pred_dens, pred_counts = model("cnt", cnt_imgs, cnt_backbone_grad_mult=float(cnt_backbone_grad_mult))
    dens_loss = F.mse_loss(pred_dens, cnt_dens, reduction="sum") / cnt_imgs.size(0)
    cnt_l1 = F.l1_loss(pred_counts, cnt_gt_counts)
    cnt_loss = dens_loss + float(cnt_count_loss_weight) * cnt_l1
    cnt_stats = None
    if collect_cnt_stats:
        cnt_stats = _count_diag_stats(pred_dens, cnt_dens, pred_counts, cnt_gt_counts)
    return cnt_loss, cnt_stats


@torch.no_grad()
def _eval_det_loss(model: torch.nn.Module, loader, device: torch.device, *, max_steps: int) -> float:
    model.train()
    total = 0.0
    samples = 0
    steps = 0
    for batch in loader:
        images, targets = _to_device_det(batch, device)
        loss = sum(model("det", images, targets).values())
        bsz = len(images)
        total += float(loss.item()) * bsz
        samples += bsz
        steps += 1
        if max_steps and steps >= max_steps:
            break
    if dist.is_initialized():
        t = torch.tensor([total, float(samples)], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        total, samples = float(t[0].item()), int(t[1].item())
    return total / max(samples, 1)


@torch.no_grad()
def _eval_seg_loss(model: torch.nn.Module, loader, device: torch.device, *, num_classes: int, max_steps: int) -> tuple[float, float]:
    model.eval()
    total = 0.0
    samples = 0
    steps = 0
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for batch in loader:
        imgs, masks = _to_device_seg(batch, device)
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
        total, samples = float(t[0].item()), int(t[1].item())
        conf_dev = conf.to(device=device)
        dist.all_reduce(conf_dev, op=dist.ReduceOp.SUM)
        conf = conf_dev.cpu()
    _, miou = per_class_iou_from_confusion(conf)
    return total / max(samples, 1), float(miou.item())


@torch.no_grad()
def _eval_cnt_loss(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    cnt_count_loss_weight: float,
    max_steps: int,
) -> tuple[float, float, float, float]:
    model.eval()
    total = 0.0
    total_density = 0.0
    total_count_mae = 0.0
    total_total_mae = 0.0
    samples = 0
    steps = 0
    for batch in loader:
        imgs, dens = _to_device_cnt(batch, device)
        gt_counts = dens.flatten(2).sum(dim=2)
        pred_dens, pred_counts = model("cnt", imgs)
        dens_loss = F.mse_loss(pred_dens, dens, reduction="sum") / imgs.size(0)
        cnt_l1 = F.l1_loss(pred_counts, gt_counts)
        loss = dens_loss + float(cnt_count_loss_weight) * cnt_l1
        count_mae = (pred_counts - gt_counts).abs().mean()
        pred_total = pred_counts.sum(dim=1)
        gt_total = gt_counts.sum(dim=1)
        total_mae = (pred_total - gt_total).abs().mean()
        bsz = imgs.size(0)
        total += float(loss.item()) * bsz
        total_density += float(dens_loss.item()) * bsz
        total_count_mae += float(count_mae.item()) * bsz
        total_total_mae += float(total_mae.item()) * bsz
        samples += bsz
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


@torch.no_grad()
def _eval_det_ap50_fast(model: torch.nn.Module, loader, device: torch.device, *, num_classes: int, score_thresh: float) -> float:
    model.eval()
    preds_by_cls = {c: [] for c in range(1, num_classes + 1)}
    gts_by_cls = {c: {} for c in range(1, num_classes + 1)}
    dist_rank = dist.get_rank() if dist.is_initialized() else 0
    image_offset = 0
    for batch in loader:
        images, targets = _to_device_det(batch, device)
        outputs = model("det", images)
        for local_idx, (target, output) in enumerate(zip(targets, outputs)):
            default_img_id = dist_rank * 10_000_000_000 + image_offset + local_idx
            if "image_id" in target:
                image_id = int(target["image_id"].detach().view(-1)[0].item())
            else:
                image_id = int(default_img_id)
            gt_boxes = target["boxes"].detach().cpu().numpy()
            gt_labels = target["labels"].detach().cpu().numpy()
            for cls_id in range(1, num_classes + 1):
                gts_by_cls[cls_id][image_id] = gt_boxes[gt_labels == cls_id]

            boxes = output["boxes"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            labels = output["labels"].detach().cpu().numpy()
            keep = scores >= float(score_thresh)
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            for box, score, label in zip(boxes, scores, labels):
                if 1 <= int(label) <= num_classes:
                    preds_by_cls[int(label)].append((image_id, float(score), box))
        image_offset += len(images)

    if dist.is_initialized():
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, (preds_by_cls, gts_by_cls))
        merged_preds = {c: [] for c in range(1, num_classes + 1)}
        merged_gts = {c: {} for c in range(1, num_classes + 1)}
        for rank_preds, rank_gts in gathered:
            for cls_id in range(1, num_classes + 1):
                merged_preds[cls_id].extend(rank_preds.get(cls_id, []))
                for img_id, boxes in rank_gts.get(cls_id, {}).items():
                    merged_gts[cls_id].setdefault(img_id, [])
                    merged_gts[cls_id][img_id].extend(list(boxes))
        preds_by_cls = merged_preds
        gts_by_cls = merged_gts

    aps = []
    for cls_id in range(1, num_classes + 1):
        preds = sorted(preds_by_cls[cls_id], key=lambda x: x[1], reverse=True)
        gt_map = gts_by_cls[cls_id]
        npos = int(sum(len(v) if isinstance(v, list) else v.shape[0] for v in gt_map.values()))
        if npos == 0:
            continue
        matched = {img_id: np.zeros(len(boxes), dtype=bool) for img_id, boxes in gt_map.items()}
        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)
        for idx, (img_id, _score, box) in enumerate(preds):
            gt_boxes = gt_map.get(img_id)
            if gt_boxes is None:
                fp[idx] = 1.0
                continue
            gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
            if gt_boxes.size == 0:
                fp[idx] = 1.0
                continue
            ious = _box_iou_np(np.asarray(box, dtype=np.float32)[None, :], gt_boxes).reshape(-1)
            best = int(ious.argmax()) if ious.size > 0 else -1
            if best >= 0 and ious[best] >= 0.5 and not matched[img_id][best]:
                tp[idx] = 1.0
                matched[img_id][best] = True
            else:
                fp[idx] = 1.0
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / max(float(npos), 1.0)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        aps.append(_ap_from_pr(rec, prec))
    return float(np.mean(aps)) if aps else 0.0


def _grad_features_from_losses(
    *,
    det_loss: torch.Tensor,
    seg_loss: torch.Tensor,
    cnt_loss: torch.Tensor,
    named_params: dict[str, torch.nn.Parameter | torch.Tensor],
    det_names: Sequence[str],
    seg_names: Sequence[str],
    cnt_names: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    det_params = [named_params[name] for name in det_names]
    seg_params = [named_params[name] for name in seg_names]
    cnt_params = [named_params[name] for name in cnt_names]
    # Stage2 still needs to backprop through the combined weighted loss afterwards,
    # so these feature extractions must preserve the original loss graphs.
    det_vec = _extract_grad_vector(det_loss, det_params, retain_graph=True)
    seg_vec = _extract_grad_vector(seg_loss, seg_params, retain_graph=True)
    cnt_vec = _extract_grad_vector(cnt_loss, cnt_params, retain_graph=True)
    return det_vec, seg_vec, cnt_vec


def main() -> None:
    args = parse_args()
    _set_seed(int(args.seed))
    device, use_ddp, rank, local_rank, world_size, is_main = _setup_dist(args)
    if device.type == "cuda":
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            torch.backends.cuda.enable_math_sdp(True)

    save_dir = Path(args.save_dir)
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)

    det_train_ds, det_val_ds, _, _ = build_det_loaders(
        data_root=args.det_data_root,
        image_size=int(args.image_size),
        batch_size=int(args.det_batch_size),
        num_workers=int(args.num_workers),
        train_ann=args.det_train_ann,
        val_ann=args.det_val_ann,
        train_img_dir=args.det_train_img_dir,
        val_img_dir=args.det_val_img_dir,
    )
    seg_train_ds, seg_val_ds, _, _ = build_seg_loaders(
        train_dir=args.seg_train_dir,
        val_dir=args.seg_val_dir,
        image_size=int(args.image_size),
        batch_size=int(args.seg_batch_size),
        num_workers=int(args.num_workers),
    )
    cnt_train_ds, cnt_val_ds, _, _ = build_cnt_loaders(
        data_root=args.cnt_data_root,
        train_dir=args.cnt_train_dir,
        val_dir=args.cnt_val_dir,
        image_size=int(args.image_size),
        num_classes=int(args.cnt_num_classes),
        keep_aspect=bool(args.cnt_keep_aspect),
        batch_size=int(args.cnt_batch_size),
        num_workers=1,
    )

    det_inner_ds, det_meta_ds = _split_dataset(det_train_ds, float(args.meta_split), int(args.meta_seed) + 0)
    seg_inner_ds, seg_meta_ds = _split_dataset(seg_train_ds, float(args.meta_split), int(args.meta_seed) + 1)
    cnt_inner_ds, cnt_meta_ds = _split_dataset(cnt_train_ds, float(args.meta_split), int(args.meta_seed) + 2)

    det_train_loader, det_train_sampler = _make_loader(
        det_train_ds,
        batch_size=int(args.det_batch_size),
        num_workers=int(args.num_workers),
        use_ddp=use_ddp,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        collate_fn_override=collate_fn,
    )
    det_val_loader, det_val_sampler = _make_loader(
        det_val_ds,
        batch_size=int(args.det_batch_size),
        num_workers=int(args.num_workers),
        use_ddp=use_ddp,
        rank=rank,
        world_size=world_size,
        shuffle=False,
        collate_fn_override=collate_fn,
    )
    seg_train_loader, seg_train_sampler = _make_loader(
        seg_train_ds,
        batch_size=int(args.seg_batch_size),
        num_workers=int(args.num_workers),
        use_ddp=use_ddp,
        rank=rank,
        world_size=world_size,
        shuffle=True,
    )
    seg_val_loader, seg_val_sampler = _make_loader(
        seg_val_ds,
        batch_size=int(args.seg_batch_size),
        num_workers=int(args.num_workers),
        use_ddp=use_ddp,
        rank=rank,
        world_size=world_size,
        shuffle=False,
    )
    cnt_loader_kwargs = dict(
        num_workers=1,
        persistent_workers=True,
        multiprocessing_context="spawn",
        prefetch_factor=2,
    )
    cnt_train_loader, cnt_train_sampler = _make_loader(
        cnt_train_ds,
        batch_size=int(args.cnt_batch_size),
        use_ddp=use_ddp,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        drop_last=True,
        **cnt_loader_kwargs,
    )
    cnt_val_loader, cnt_val_sampler = _make_loader(
        cnt_val_ds,
        batch_size=int(args.cnt_batch_size),
        use_ddp=use_ddp,
        rank=rank,
        world_size=world_size,
        shuffle=False,
        drop_last=False,
        **cnt_loader_kwargs,
    )
    det_inner_loader, det_inner_sampler = _make_loader(
        det_inner_ds,
        batch_size=int(args.det_batch_size),
        num_workers=int(args.num_workers),
        use_ddp=use_ddp,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        collate_fn_override=collate_fn,
    )
    det_meta_loader, det_meta_sampler = _make_loader(
        det_meta_ds,
        batch_size=int(args.det_batch_size),
        num_workers=int(args.num_workers),
        use_ddp=use_ddp,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        collate_fn_override=collate_fn,
    )
    seg_inner_loader, seg_inner_sampler = _make_loader(
        seg_inner_ds,
        batch_size=int(args.seg_batch_size),
        num_workers=int(args.num_workers),
        use_ddp=use_ddp,
        rank=rank,
        world_size=world_size,
        shuffle=True,
    )
    seg_meta_loader, seg_meta_sampler = _make_loader(
        seg_meta_ds,
        batch_size=int(args.seg_batch_size),
        num_workers=int(args.num_workers),
        use_ddp=use_ddp,
        rank=rank,
        world_size=world_size,
        shuffle=True,
    )
    cnt_inner_loader, cnt_inner_sampler = _make_loader(
        cnt_inner_ds,
        batch_size=int(args.cnt_batch_size),
        use_ddp=use_ddp,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        drop_last=True,
        **cnt_loader_kwargs,
    )
    cnt_meta_loader, cnt_meta_sampler = _make_loader(
        cnt_meta_ds,
        batch_size=int(args.cnt_batch_size),
        use_ddp=use_ddp,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        drop_last=False,
        **cnt_loader_kwargs,
    )

    det_num_classes = int(args.det_num_classes) if args.det_num_classes is not None else int(det_train_ds.num_classes)
    model = _build_model(args, det_num_classes, device)
    theta_param_groups, theta_params = _build_theta_param_groups(model, args)

    named_model_params = _named_param_map(model)
    det_last_names = [
        "detector.roi_heads.box_predictor.cls_score.weight",
        "detector.roi_heads.box_predictor.cls_score.bias",
        "detector.roi_heads.box_predictor.bbox_pred.weight",
        "detector.roi_heads.box_predictor.bbox_pred.bias",
    ]
    seg_last_names = ["seg_head.decode.3.weight", "seg_head.decode.3.bias"]
    cnt_last_names = ["cnt_head.decode.3.weight", "cnt_head.decode.3.bias"]
    for name in det_last_names + seg_last_names + cnt_last_names:
        if name not in named_model_params:
            raise KeyError(f"Missing expected task-head last-layer parameter: {name}")

    det_last_dim = int(sum(named_model_params[name].numel() for name in det_last_names))
    seg_last_dim = int(sum(named_model_params[name].numel() for name in seg_last_names))
    cnt_last_conv = model.cnt_head.decode[3]
    if not isinstance(cnt_last_conv, torch.nn.Conv2d) or cnt_last_conv.bias is None or cnt_last_conv.kernel_size != (1, 1):
        raise RuntimeError("Counting head last layer must be 1x1 Conv2d with bias.")

    phi_core = JointWeightGenerator(
        det_in_dim=det_last_dim,
        seg_in_dim=seg_last_dim,
        cnt_in_channels=int(cnt_last_conv.in_channels),
        cnt_num_classes=int(cnt_last_conv.out_channels),
        base_loss_weights=(15.0, 8.0, 1.0),
    ).to(device)

    if use_ddp:
        _broadcast_module_state(model, src=0)
        _broadcast_module_state(phi_core, src=0)
        ddp_device_ids = [local_rank] if device.type == "cuda" else None
        ddp_output_device = local_rank if device.type == "cuda" else None
        phi_train = DistributedDataParallel(
            phi_core,
            device_ids=ddp_device_ids,
            output_device=ddp_output_device,
            find_unused_parameters=False,
        )
    else:
        phi_train = phi_core

    train_full_loaders = {"det": det_train_loader, "seg": seg_train_loader, "cnt": cnt_train_loader}
    train_inner_loaders = {"det": det_inner_loader, "seg": seg_inner_loader, "cnt": cnt_inner_loader}
    meta_loaders = {"det": det_meta_loader, "seg": seg_meta_loader, "cnt": cnt_meta_loader}
    val_loaders = {"det": det_val_loader, "seg": seg_val_loader, "cnt": cnt_val_loader}

    train_samplers = [
        det_train_sampler,
        seg_train_sampler,
        cnt_train_sampler,
        det_inner_sampler,
        seg_inner_sampler,
        cnt_inner_sampler,
        det_meta_sampler,
        seg_meta_sampler,
        cnt_meta_sampler,
        det_val_sampler,
        seg_val_sampler,
        cnt_val_sampler,
    ]
    primary_inner = choose_primary({k: len(v) for k, v in train_inner_loaders.items()}, None)
    primary_full = choose_primary({k: len(v) for k, v in train_full_loaders.items()}, None)
    inner_cyc = {k: infinite_loader(v) for k, v in train_inner_loaders.items() if k != primary_inner}
    meta_cyc = {k: infinite_loader(v) for k, v in meta_loaders.items()}
    full_cyc = {k: infinite_loader(v) for k, v in train_full_loaders.items() if k != primary_full}

    optimizer_phi = torch.optim.AdamW(phi_core.parameters(), lr=float(args.stage1_phi_lr), weight_decay=0.0)
    theta_named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    theta_name_to_index = {name: idx for idx, (name, _) in enumerate(theta_named_params)}
    full_param_names = [name for name, _ in model.named_parameters()]

    best_metric = float("-inf")
    best_state = None
    best_phi_state = None
    best_epoch = None

    def run_validation(eval_model: torch.nn.Module, epoch: int) -> tuple[float, Dict[str, float]]:
        val_det = _eval_det_loss(eval_model, val_loaders["det"], device, max_steps=int(args.max_val_steps))
        val_seg, val_seg_miou = _eval_seg_loss(
            eval_model,
            val_loaders["seg"],
            device,
            num_classes=int(args.seg_num_classes),
            max_steps=int(args.max_val_steps),
        )
        val_cnt, val_cnt_density, val_cnt_mae, val_cnt_total_mae = _eval_cnt_loss(
            eval_model,
            val_loaders["cnt"],
            device,
            cnt_count_loss_weight=float(args.cnt_count_loss_weight),
            max_steps=int(args.max_val_steps),
        )
        val_ap50 = _eval_det_ap50_fast(
            eval_model,
            val_loaders["det"],
            device,
            num_classes=det_num_classes,
            score_thresh=float(args.det_ap_score_thr),
        )
        combo = float(val_ap50) + float(val_seg_miou) + 1.0 / max(float(val_cnt_mae), 1e-8)
        metrics = {
            "val_det_loss": float(val_det),
            "val_seg_loss": float(val_seg),
            "val_seg_miou": float(val_seg_miou),
            "val_cnt_loss": float(val_cnt),
            "val_cnt_density_mse": float(val_cnt_density),
            "val_cnt_mae": float(val_cnt_mae),
            "val_cnt_total_mae": float(val_cnt_total_mae),
            "val_ap50": float(val_ap50),
            "selected_metric": float(combo),
        }
        if is_main:
            print(
                f"[stage2] epoch {epoch} | val det {val_det:.4f} seg {val_seg:.4f} miou {val_seg_miou:.4f} "
                f"cnt {val_cnt:.4f} dens {val_cnt_density:.6e} mae {val_cnt_mae:.4f} total_mae {val_cnt_total_mae:.4f} "
                f"| ap50 {val_ap50:.4f} | combo {combo:.6f}"
            )
        return combo, metrics

    for epoch in range(1, int(args.stage1_epochs) + 1):
        for sampler in train_samplers:
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
        model.train()
        phi_train.train()
        loss_sum = 0.0
        steps = 0

        for step, primary_batch in enumerate(train_inner_loaders[primary_inner], start=1):
            support_batches = {primary_inner: primary_batch}
            for task_name in train_inner_loaders:
                if task_name != primary_inner:
                    support_batches[task_name] = next(inner_cyc[task_name])
            meta_batches = {task_name: next(meta_cyc[task_name]) for task_name in meta_loaders}

            model.zero_grad(set_to_none=True)
            optimizer_phi.zero_grad(set_to_none=True)

            # Extract detached gradient features on the real model first, sequentially, to
            # avoid building a three-task graph outside the higher inner loop.
            feat_det = _compute_det_loss_only(model, support_batches["det"], device=device)
            support_det_value = float(feat_det.detach().item())
            det_vec = _extract_grad_vector(
                feat_det,
                [named_model_params[name] for name in det_last_names],
                retain_graph=True,
            )
            det_task_grads = _extract_task_grads(feat_det, theta_params)
            del feat_det
            model.zero_grad(set_to_none=True)

            feat_seg = _compute_seg_loss_only(model, support_batches["seg"], device=device)
            support_seg_value = float(feat_seg.detach().item())
            seg_vec = _extract_grad_vector(
                feat_seg,
                [named_model_params[name] for name in seg_last_names],
                retain_graph=True,
            )
            seg_task_grads = _extract_task_grads(feat_seg, theta_params)
            del feat_seg
            model.zero_grad(set_to_none=True)

            feat_cnt, sup_cnt_stats = _compute_cnt_loss_only(
                model,
                support_batches["cnt"],
                device=device,
                cnt_count_loss_weight=float(args.cnt_count_loss_weight),
                cnt_backbone_grad_mult=float(args.cnt_backbone_grad_mult),
                collect_cnt_stats=(step == 1 and is_main),
            )
            support_cnt_value = float(feat_cnt.detach().item())
            cnt_vec = _extract_grad_vector(
                feat_cnt,
                [named_model_params[name] for name in cnt_last_names],
                retain_graph=True,
            )
            cnt_task_grads = _extract_task_grads(feat_cnt, theta_params)
            del feat_cnt
            model.zero_grad(set_to_none=True)
            stage1_lr = float(args.stage1_inner_lr)
            meta_total_value = 0.0
            log_weights = None
            staged_fast_names: list[str] | None = None
            staged_fast_tensors: list[torch.Tensor] | None = None

            weights = phi_train(det_vec, seg_vec, cnt_vec)
            log_weights = [float(x) for x in weights.detach().cpu().tolist()]
            fmodel = _build_stage1_fast_model(
                model,
                device=device,
                weights=weights,
                theta_params=theta_params,
                theta_name_to_index=theta_name_to_index,
                full_param_names=full_param_names,
                det_task_grads=det_task_grads,
                seg_task_grads=seg_task_grads,
                cnt_task_grads=cnt_task_grads,
                stage1_lr=stage1_lr,
            )
            meta_det = _compute_det_loss_only(fmodel, meta_batches["det"], device=device)
            meta_total_value += float(meta_det.detach().item())
            meta_det.backward()
            del meta_det
            del fmodel, weights

            weights = phi_train(det_vec, seg_vec, cnt_vec)
            fmodel = _build_stage1_fast_model(
                model,
                device=device,
                weights=weights,
                theta_params=theta_params,
                theta_name_to_index=theta_name_to_index,
                full_param_names=full_param_names,
                det_task_grads=det_task_grads,
                seg_task_grads=seg_task_grads,
                cnt_task_grads=cnt_task_grads,
                stage1_lr=stage1_lr,
            )
            meta_seg = _compute_seg_loss_only(fmodel, meta_batches["seg"], device=device)
            meta_total_value += float(meta_seg.detach().item())
            meta_seg.backward()
            del meta_seg
            del fmodel, weights

            weights = phi_train(det_vec, seg_vec, cnt_vec)
            fmodel = _build_stage1_fast_model(
                model,
                device=device,
                weights=weights,
                theta_params=theta_params,
                theta_name_to_index=theta_name_to_index,
                full_param_names=full_param_names,
                det_task_grads=det_task_grads,
                seg_task_grads=seg_task_grads,
                cnt_task_grads=cnt_task_grads,
                stage1_lr=stage1_lr,
            )
            meta_cnt, _ = _compute_cnt_loss_only(
                fmodel,
                meta_batches["cnt"],
                device=device,
                cnt_count_loss_weight=float(args.cnt_count_loss_weight),
                cnt_backbone_grad_mult=float(args.cnt_backbone_grad_mult),
                collect_cnt_stats=False,
            )
            meta_total_value += float(meta_cnt.detach().item())
            meta_cnt.backward()
            del meta_cnt
            del fmodel, weights

            copy_weights = phi_train(det_vec, seg_vec, cnt_vec).detach()
            copy_fmodel = _build_stage1_fast_model(
                model,
                device=device,
                weights=copy_weights,
                theta_params=theta_params,
                theta_name_to_index=theta_name_to_index,
                full_param_names=full_param_names,
                det_task_grads=det_task_grads,
                seg_task_grads=seg_task_grads,
                cnt_task_grads=cnt_task_grads,
                stage1_lr=stage1_lr,
            )
            staged_fast_names = [name for name, _ in model.named_parameters()]
            staged_fast_tensors = [param.detach() for param in copy_fmodel.fast_params]
            del copy_fmodel, copy_weights

            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(phi_core.parameters(), max_norm=float(args.grad_clip_norm))
            optimizer_phi.step()

            if staged_fast_names is not None and staged_fast_tensors is not None:
                _copy_fast_param_tensors_to_model(
                    model,
                    staged_fast_names,
                    staged_fast_tensors,
                    use_ddp=use_ddp,
                    world_size=world_size,
                )
                if use_ddp:
                    _broadcast_module_state(model, src=0)

            del det_vec, seg_vec, cnt_vec, det_task_grads, seg_task_grads, cnt_task_grads
            if staged_fast_tensors is not None:
                del staged_fast_tensors

            model.zero_grad(set_to_none=True)
            loss_sum += float(meta_total_value)
            steps += 1

            if is_main and int(args.log_interval) > 0 and step % int(args.log_interval) == 0:
                w = log_weights if log_weights is not None else [0.0, 0.0, 0.0]
                print(
                    f"[stage1] epoch {epoch}/{int(args.stage1_epochs)} step {step} | "
                    f"support [{support_det_value:.4f}, {support_seg_value:.4f}, {support_cnt_value:.4f}] | "
                    f"meta {meta_total_value:.4f} | w [{w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}]"
                )
                if sup_cnt_stats is not None:
                    print(f"[diag][cnt][stage1] {_format_count_diag(sup_cnt_stats)}")

            if int(args.max_train_steps) > 0 and step >= int(args.max_train_steps):
                break

        if is_main:
            print(f"[stage1] epoch {epoch}/{int(args.stage1_epochs)} | meta_loss {loss_sum / max(steps, 1):.4f}")

    stage1_last = save_dir / "stage1_last.pt"
    if is_main:
        save_multitask_checkpoint(
            str(stage1_last),
            model=model,
            optimizer=None,
            epoch=int(args.stage1_epochs),
            best_by="stage1_last",
            metrics={"epoch": float(args.stage1_epochs)},
            loss_weights=(15.0, 8.0, 1.0),
            phi_state=_state_dict_cpu_clone(phi_core.state_dict()),
            config={
                "stage": "stage1_last",
                "use_lora_moe": bool(args.use_lora_moe),
                "unfreeze_backbone": bool(args.unfreeze_backbone),
            },
        )
        print(f"[ckpt] saved stage1_last -> {stage1_last}")

    for p in phi_core.parameters():
        p.requires_grad = False
    phi_core.eval()
    if use_ddp and dist.is_initialized():
        dist.barrier()

    if bool(args.select_best_from_stage2):
        best_metric = float("-inf")
        best_state = None
        best_phi_state = None
        best_epoch = None
        if is_main:
            print("[train] model selection starts from Stage2.")

    if use_ddp:
        ddp_device_ids = [local_rank] if device.type == "cuda" else None
        ddp_output_device = local_rank if device.type == "cuda" else None
        stage2_model = DistributedDataParallel(
            model,
            device_ids=ddp_device_ids,
            output_device=ddp_output_device,
            find_unused_parameters=False,
        )
    else:
        stage2_model = model

    optimizer_stage2 = torch.optim.AdamW(theta_param_groups)

    for epoch in range(1, int(args.stage2_epochs) + 1):
        if isinstance(det_train_sampler, DistributedSampler):
            det_train_sampler.set_epoch(int(args.stage1_epochs) + epoch)
        if isinstance(seg_train_sampler, DistributedSampler):
            seg_train_sampler.set_epoch(int(args.stage1_epochs) + epoch)
        if isinstance(cnt_train_sampler, DistributedSampler):
            cnt_train_sampler.set_epoch(int(args.stage1_epochs) + epoch)

        stage2_model.train()
        loss_sum = 0.0
        steps = 0

        for step, primary_batch in enumerate(train_full_loaders[primary_full], start=1):
            batches = {primary_full: primary_batch}
            for task_name in train_full_loaders:
                if task_name != primary_full:
                    batches[task_name] = next(full_cyc[task_name])

            optimizer_stage2.zero_grad(set_to_none=True)
            det_loss, seg_loss, cnt_loss, cnt_stats = _compute_losses(
                stage2_model,
                batches["det"],
                batches["seg"],
                batches["cnt"],
                device=device,
                cnt_count_loss_weight=float(args.cnt_count_loss_weight),
                cnt_backbone_grad_mult=float(args.cnt_backbone_grad_mult),
                collect_cnt_stats=(step == 1 and is_main),
            )
            named_params = _named_param_map(model)
            det_vec, seg_vec, cnt_vec = _grad_features_from_losses(
                det_loss=det_loss,
                seg_loss=seg_loss,
                cnt_loss=cnt_loss,
                named_params=named_params,
                det_names=det_last_names,
                seg_names=seg_last_names,
                cnt_names=cnt_last_names,
            )
            with torch.no_grad():
                weights = phi_core(det_vec, seg_vec, cnt_vec)
            total = weights[0] * det_loss + weights[1] * seg_loss + weights[2] * cnt_loss
            total.backward()
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(theta_params, max_norm=float(args.grad_clip_norm))
            optimizer_stage2.step()

            loss_sum += float(total.detach().item())
            steps += 1

            if is_main and int(args.log_interval) > 0 and step % int(args.log_interval) == 0:
                w = [float(x) for x in weights.detach().cpu().tolist()]
                print(
                    f"[stage2] epoch {epoch}/{int(args.stage2_epochs)} step {step} | "
                    f"loss {total.item():.4f} | det {det_loss.item():.4f} seg {seg_loss.item():.4f} cnt {cnt_loss.item():.4f} | "
                    f"w [{w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f}]"
                )
                if cnt_stats is not None:
                    print(f"[diag][cnt][stage2] {_format_count_diag(cnt_stats)}")

            if int(args.max_train_steps) > 0 and step >= int(args.max_train_steps):
                break

        if is_main:
            print(f"[stage2] epoch {epoch}/{int(args.stage2_epochs)} | train {loss_sum / max(steps, 1):.4f}")

        if not bool(args.skip_validation):
            combo_metric, metrics = run_validation(stage2_model, epoch)
            if combo_metric > best_metric:
                best_metric = float(combo_metric)
                best_epoch = epoch
                if is_main:
                    best_state = _state_dict_cpu_clone(model.state_dict())
                    best_phi_state = _state_dict_cpu_clone(phi_core.state_dict())
                    print(f"[ckpt] new best cached (stage2 epoch {epoch}, combo {best_metric:.6f})")

    stage2_last = save_dir / "stage2_last.pt"
    if is_main:
        save_multitask_checkpoint(
            str(stage2_last),
            model=model,
            optimizer=optimizer_stage2,
            epoch=int(args.stage2_epochs),
            best_by="stage2_last",
            metrics={"epoch": float(args.stage2_epochs)},
            loss_weights=(15.0, 8.0, 1.0),
            phi_state=_state_dict_cpu_clone(phi_core.state_dict()),
            config={
                "stage": "stage2_last",
                "use_lora_moe": bool(args.use_lora_moe),
                "unfreeze_backbone": bool(args.unfreeze_backbone),
            },
        )
        print(f"[ckpt] saved stage2_last -> {stage2_last}")

    best_path = save_dir / "best_combo.pt"
    if is_main and best_state is not None:
        model.load_state_dict(best_state, strict=True)
        save_multitask_checkpoint(
            str(best_path),
            model=model,
            optimizer=optimizer_stage2,
            epoch=int(best_epoch or args.stage2_epochs),
            best_by="combo",
            metrics={"best_metric": float(best_metric), "best_epoch": int(best_epoch or args.stage2_epochs)},
            loss_weights=(15.0, 8.0, 1.0),
            phi_state=best_phi_state,
            config={
                "use_lora_moe": bool(args.use_lora_moe),
                "unfreeze_backbone": bool(args.unfreeze_backbone),
                "lora_rank": int(args.lora_rank),
                "num_experts_private": int(args.num_experts_private),
                "num_experts_shared": int(args.num_experts_shared),
                "moe_k_private": int(args.moe_k_private),
                "moe_k_shared": int(args.moe_k_shared),
            },
        )
        print(f"[ckpt] saved best -> {best_path} (combo {best_metric:.6f})")

    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

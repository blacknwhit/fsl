from __future__ import annotations

import argparse
import builtins
import sys
import math
import os
import random
from copy import deepcopy
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import GradScaler
try:
    from torch.func import functional_call
except Exception:
    # Backward compatibility for older PyTorch versions
    from torch.nn.utils.stateless import functional_call
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from .datasets import build_cnt_loaders, build_det_loaders, build_seg_loaders
from .models import MultiTaskModel, SharedDinoV3Backbone
from .utils import choose_primary, infinite_loader, parse_loss_weights, save_multitask_checkpoint
from segmentation.utils import per_class_iou_from_confusion, update_confusion_matrix
from object_detection.dataset import collate_fn


class CountGradProjector(nn.Module):
    """
    Structure-aware projector for the counting head's final 1x1 conv gradients.

    The flat gradient vector is assumed to come from concatenating:
    1. conv.weight.grad with shape [num_classes, in_channels, 1, 1]
    2. conv.bias.grad with shape [num_classes]

        We restore it as per-class gradients [num_classes, in_channels + 1], apply a
        shared 1x1 Conv1d over classes, then flatten the class descriptors and map to
        the 64-d joint weight-net embedding.

        Default architecture (before joint concat):
            [C, D+1] -> [1, D+1, C] -> Conv1d(D+1, 64, k=1) -> LeakyReLU
            -> Flatten(64*C) -> Linear(64*C, 64)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        out_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.class_feat_dim = self.in_channels + 1
        self.expected_dim = self.num_classes * self.class_feat_dim

        self.class_proj = nn.Conv1d(self.class_feat_dim, hidden_dim, kernel_size=1, bias=True)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.out_proj = nn.Linear(hidden_dim * self.num_classes, out_dim)

    def forward(self, flat_grad: torch.Tensor) -> torch.Tensor:
        flat_grad = flat_grad.reshape(-1)
        if int(flat_grad.numel()) != self.expected_dim:
            raise ValueError(
                "CountGradProjector input dim mismatch: "
                f"got {int(flat_grad.numel())}, expected {self.expected_dim}"
            )

        weight_dim = self.num_classes * self.in_channels

        # Preserve the original Conv2d parameter layout:
        # [class0 weights..., class1 weights..., ...] followed by per-class bias.
        grad_w = flat_grad[:weight_dim].reshape(self.num_classes, self.in_channels)
        grad_b = flat_grad[weight_dim:].reshape(self.num_classes, 1)

        class_grads = torch.cat([grad_w, grad_b], dim=1)  # [C, D + 1]
        class_grads = class_grads.transpose(0, 1).unsqueeze(0)  # [1, D + 1, C]

        feat = self.class_proj(class_grads)  # [1, hidden_dim, C]
        feat = self.act(feat)
        feat = feat.flatten(start_dim=1)  # [1, hidden_dim * C]
        return self.out_proj(feat).squeeze(0)


def parse_args():
    p = argparse.ArgumentParser(description="Multi-task training (det/seg/count) with shared DINOv3 backbone")

    # Backbone
    p.add_argument("--model-name", type=str, default="dinov3_vitl16")
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--backbone-checkpoint", type=str, default=None)
    p.add_argument(
        "--unfreeze-backbone",
        action="store_true",
        help="train the shared backbone; default is to keep it frozen",
    )
    
    # LoRA-MoE parameters (private + shared pools)
    p.add_argument("--use-lora-moe", action="store_true", help="Enable LoRA-MoE adapters")
    p.add_argument("--lora-rank", type=int, default=8, help="LoRA rank for experts (shared across pools)")
    p.add_argument("--num-experts-private", type=int, default=2, help="Private experts per task per block")
    p.add_argument("--num-experts-shared", type=int, default=6, help="Shared experts per block")
    p.add_argument("--moe-k-private", type=int, default=2, help="Top-k private experts per token")
    p.add_argument("--moe-k-shared", type=int, default=2, help="Top-k shared experts per token")

    # Learnable loss weights
    p.add_argument(
        "--learn-loss-weights",
        action="store_true",
        help="learn task loss weights via softplus(beta); base bias comes from --loss-weight-bias",
    )
    p.add_argument(
        "--learn-loss-weights-mlp",
        action="store_true",
        help="learn task loss weights via MLP on prev-step head gradients; base bias comes from --loss-weight-bias",
    )
    p.add_argument(
        "--weight-net-arch",
        type=str,
        default="per_task_shared",
        choices=["per_task_shared", "joint"],
        help=(
            "Loss-weight generator architecture: "
            "'per_task_shared' = per-task projector + shared 64->16->1 generator; "
            "'joint' = concat projected features and predict 3 weights jointly."
        ),
    )
    p.add_argument(
        "--joint-weight-out-act",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "leakyrelu"],
        help=(
            "Output activation for joint weight-net head (only used when --weight-net-arch=joint). "
            "'sigmoid' constrains to [0,1]; 'leakyrelu' keeps signed output."
        ),
    )
    p.add_argument(
        "--joint-leakyrelu-slope",
        type=float,
        default=0.01,
        help="negative_slope for joint output LeakyReLU when --joint-weight-out-act=leakyrelu.",
    )
    p.add_argument(
        "--weight-net-dropout",
        type=float,
        default=0.0,
        help="Dropout probability used inside loss-weight generator networks (per_task_shared/joint).",
    )
    p.add_argument(
        "--grad-vec-normalize",
        type=str,
        default="l2",
        choices=["l2", "none"],
        help=(
            "Gradient preprocessing before weight-net projector. "
            "'l2' = per-task L2 normalization after nan_to_num; "
            "'none' = only nan_to_num, keep magnitude."
        ),
    )
    p.add_argument(
        "--weight-net-use-multilayer-grads",
        action="store_true",
        help=(
            "Use multi-layer gradient features for loss-weight MLP. "
            "Per eligible layer (Linear/Conv2d): flattened grads -> layer embed; "
            "per task: concat layer embeds -> task embed."
        ),
    )
    p.add_argument(
        "--weight-net-layer-embed-dim",
        type=int,
        default=16,
        help="Layer-level embedding dim used when --weight-net-use-multilayer-grads is enabled.",
    )
    p.add_argument(
        "--weight-net-task-embed-dim",
        type=int,
        default=64,
        help="Task-level embedding dim used when --weight-net-use-multilayer-grads is enabled.",
    )
    p.add_argument(
        "--weight-net-cnt-grad-hidden-dim",
        type=int,
        default=64,
        help="Hidden dim for the counting weight-net Conv1d projector when joint/non-multilayer mode is used.",
    )
    # Learnable linear weights (beta1~3) for det/seg/cnt (legacy)
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

    # Deprecated per-task backbone flags. Full MAML now uses the global --unfreeze-backbone switch.
    det_ft = p.add_mutually_exclusive_group()
    det_ft.add_argument(
        "--det-unfreeze-backbone",
        dest="det_unfreeze_backbone",
        action="store_true",
        help="deprecated: ignored; use --unfreeze-backbone",
    )
    det_ft.add_argument(
        "--det-freeze-backbone",
        dest="det_unfreeze_backbone",
        action="store_false",
        help="deprecated: ignored; use --unfreeze-backbone",
    )
    p.set_defaults(det_unfreeze_backbone=None)
    seg_ft = p.add_mutually_exclusive_group()
    seg_ft.add_argument("--seg-full-finetune", dest="seg_full_finetune", action="store_true")
    seg_ft.add_argument("--seg-freeze-backbone", dest="seg_full_finetune", action="store_false")
    p.set_defaults(seg_full_finetune=None)
    cnt_ft = p.add_mutually_exclusive_group()
    cnt_ft.add_argument("--cnt-full-finetune", dest="cnt_full_finetune", action="store_true")
    cnt_ft.add_argument("--cnt-freeze-backbone", dest="cnt_full_finetune", action="store_false")
    p.set_defaults(cnt_full_finetune=None)

    # Detection dataset
    p.add_argument("--det-data-root", type=str, required=True)
    p.add_argument("--det-train-ann", type=str, default=None)
    p.add_argument("--det-val-ann", type=str, default=None)
    p.add_argument("--det-train-img-dir", type=str, default=None)
    p.add_argument("--det-val-img-dir", type=str, default=None)
    p.add_argument("--det-num-classes", type=int, default=None, help="foreground class count (auto if None)")

    # Detection validation metric (fast AP50)
    p.add_argument(
        "--det-ap-score-thr",
        type=float,
        default=0.0,
        help=(
            "Score threshold used by the in-train fast AP50 validator. "
            "Default 0.0 keeps all predictions (torchvision will still cap to detections_per_img, typically 100). "
            "If you see pred counts like N_images*100, try 0.05 or 0.1 for a more interpretable diagnostic."
        ),
    )

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
    p.add_argument("--epochs", type=int, default=12, help="(ignored) legacy total epochs")
    p.add_argument(
        "--val-every",
        type=int,
        default=1,
        help="Run validation every N epochs (default: 1 = validate every epoch).",
    )
    p.add_argument(
        "--select-best-from-stage2",
        action="store_true",
        help="Disable Stage1 validation/best selection; start validation and model selection from Stage2.",
    )
    p.add_argument("--skip-validation", action="store_true", help="Skip validation in both Stage1 and Stage2.")
    p.add_argument("--stage1-epochs", type=int, default=100, help="meta-training epochs for Stage1")
    p.add_argument("--stage2-epochs", type=int, default=20, help="training epochs for Stage2")
    p.add_argument("--meta-split", type=float, default=0.2, help="train split ratio for meta validation")
    p.add_argument("--meta-seed", type=int, default=42, help="random seed for meta split (fixed to 42)")
    p.add_argument("--meta-alpha", type=float, default=1e-4, help="inner-loop step size")
    p.add_argument("--meta-beta", type=float, default=1e-4, help="outer-loop step size")
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
    p.add_argument(
        "--loss-weights",
        type=str,
        default="1,1,1",
        help="legacy fixed task weights (det,seg,cnt), e.g. 1,1,1; Full MAML uses --loss-weight-bias",
    )
    p.add_argument(
        "--loss-weight-bias",
        type=str,
        default="15:8:1",
        help="constant bias added to learned MLP task weights (det,seg,cnt), e.g. 15:8:1",
    )
    p.add_argument(
        "--loss-weight-prior-mul",
        action="store_true",
        help=(
            "Fuse learned loss weights with --loss-weight-bias by multiplication "
            "(w_final = w_net * prior). Default is additive fusion (w_final = w_net + prior)."
        ),
    )
    p.add_argument("--primary-task", type=str, default=None, help="override primary task: det|seg|cnt")
    p.add_argument("--best-by", type=str, default="total", choices=["total", "det", "seg", "cnt"])
    p.add_argument("--save-dir", type=str, default="runs/multitask")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--debug-cnt", action="store_true", help="Enable detailed counting-branch diagnostics")
    p.add_argument(
        "--debug-cnt-interval",
        type=int,
        default=0,
        help="Print counting diagnostics every N steps (0 disables interval-based debug prints)",
    )
    p.add_argument(
        "--debug-first-n-steps",
        type=int,
        default=0,
        help="Always print counting diagnostics for the first N steps of each epoch",
    )
    p.add_argument("--amp", action="store_true")
    p.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help="clip grad norm (0 disables). Applied to ALL trainable parameters, like counting single-task.",
    )
    p.add_argument(
        "--phi-grad-clip-norm",
        type=float,
        default=0.0,
        help=(
            "clip grad norm for loss-weight generator (phi) in Stage1 outer update only "
            "(0 disables)."
        ),
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42, help="global random seed")
    p.add_argument("--max-train-steps", type=int, default=0, help="0 = no limit")
    p.add_argument("--max-val-steps", type=int, default=0, help="0 = no limit")
    fomaml = p.add_mutually_exclusive_group()
    fomaml.add_argument(
        "--first-order",
        dest="first_order",
        action="store_true",
        help="Use first-order MAML (stop-grad through inner update).",
    )
    fomaml.add_argument(
        "--full-maml",
        dest="first_order",
        action="store_false",
        help="Use full MAML (second-order).",
    )
    p.set_defaults(first_order=True)

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


def _param_grad_l2_norm(params: list[torch.nn.Parameter]) -> tuple[float, int]:
    total_sq = 0.0
    grad_param_count = 0
    for p in params:
        g = p.grad
        if g is None:
            continue
        gg = g.detach().float()
        total_sq += float(gg.pow(2).sum().item())
        grad_param_count += 1
    return math.sqrt(total_sq), grad_param_count


def _param_delta_l2_norm(params: list[torch.nn.Parameter], refs: list[torch.Tensor]) -> float:
    total_sq = 0.0
    for p, r in zip(params, refs):
        d = p.detach().float() - r
        total_sq += float(d.pow(2).sum().item())
    return math.sqrt(total_sq)


def _grad_l2_norm_from_map(
    params: list[torch.nn.Parameter], grad_map: Dict[int, torch.Tensor]
) -> tuple[float, int]:
    total_sq = 0.0
    grad_param_count = 0
    for p in params:
        g = grad_map.get(id(p))
        if g is None:
            continue
        gg = g.detach().float()
        total_sq += float(gg.pow(2).sum().item())
        grad_param_count += 1
    return math.sqrt(total_sq), grad_param_count


def _grads_l2_norm(grads: tuple[torch.Tensor | None, ...]) -> float:
    total_sq = 0.0
    for g in grads:
        if g is None:
            continue
        total_sq += float(g.detach().float().pow(2).sum().item())
    return math.sqrt(total_sq)


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
    pred_count_from_dens_mean = pred_dens_mean * float(pixels)

    eps = 1e-12
    return {
        "pred_dens_mean": pred_dens_mean,
        "gt_dens_mean": gt_dens_mean,
        "dens_ratio": pred_dens_mean / max(gt_dens_mean, eps),
        "pred_dens_min": float(pred_dens_f.min().item()),
        "pred_dens_max": float(pred_dens_f.max().item()),
        "gt_dens_min": float(gt_dens_f.min().item()),
        "gt_dens_max": float(gt_dens_f.max().item()),
        "pred_dens_nz": float((pred_dens_f > 0).float().mean().item()),
        "gt_dens_nz": float((gt_dens_f > 0).float().mean().item()),
        "pred_count_mean": pred_count_mean,
        "gt_count_mean": gt_count_mean,
        "count_ratio": pred_count_mean / max(gt_count_mean, eps),
        "count_mae": count_mae,
        "pred_total_mean": pred_total_mean,
        "gt_total_mean": gt_total_mean,
        "pixels": float(pixels),
        "pred_count_from_dens_mean": pred_count_from_dens_mean,
    }


def _format_count_diag(stats: Dict[str, float]) -> str:
    return (
        "dens(mean "
        f"{stats['pred_dens_mean']:.6e}/{stats['gt_dens_mean']:.6e}, "
        f"ratio {stats['dens_ratio']:.3e}, "
        f"minmax {stats['pred_dens_min']:.3e}..{stats['pred_dens_max']:.3e}/"
        f"{stats['gt_dens_min']:.3e}..{stats['gt_dens_max']:.3e}, "
        f"nz {stats['pred_dens_nz']:.3f}/{stats['gt_dens_nz']:.3f}) | "
        "count(mean "
        f"{stats['pred_count_mean']:.3f}/{stats['gt_count_mean']:.3f}, "
        f"ratio {stats['count_ratio']:.3e}, mae {stats['count_mae']:.3f}, "
        f"total {stats['pred_total_mean']:.3f}/{stats['gt_total_mean']:.3f}) | "
        f"pixels {int(stats['pixels'])} | "
        f"pred_count_from_dens_mean {stats['pred_count_from_dens_mean']:.3f}"
    )


@torch.no_grad()
def _eval_det_loss(model: MultiTaskModel, loader, device: torch.device, *, amp: bool, max_steps: int) -> float:
    model.train()  # FasterRCNN only returns losses in train mode
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
    debug_cnt: bool,
) -> tuple[float, float, float, float]:
    model.eval()
    total = 0.0
    total_density = 0.0
    total_count_mae = 0.0
    total_total_mae = 0.0
    steps = 0
    samples = 0
    printed_diag = False
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
        if not printed_diag:
            diag = _count_diag_stats(pred_dens, dens, pred_counts, gt_counts)
            if debug_cnt:
                print(f"[diag][cnt][val] {_format_count_diag(diag)}")
            else:
                print(
                    "[diag][cnt][val] "
                    f"pred_count_mean {diag['pred_count_mean']:.6f} "
                    f"gt_count_mean {diag['gt_count_mean']:.6f} "
                    f"pred_dens_mean {diag['pred_dens_mean']:.6f} "
                    f"gt_dens_mean {diag['gt_dens_mean']:.6f}"
                )
            printed_diag = True
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
    """Fast AP50 for in-train diagnostics.

    This follows the ori implementation: no mutation of detector internals,
    and no extra label filtering/diagnostics that could change behavior.
    """

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
                cls_gts = rank_gts.get(cls, {})
                for img_id, boxes in cls_gts.items():
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

    # Global seeding for reproducibility.
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    is_main_process = (not use_ddp) or rank == 0

    if use_ddp and not is_main_process:
        builtins.print = lambda *args, **kwargs: None

    if device.type == "cuda":
        # Full MAML needs higher-order grads; disable fused SDPA kernels.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    save_dir = Path(args.save_dir)
    if is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
    if use_ddp:
        dist.barrier()

    save_epochs = _parse_save_epochs(getattr(args, "save_epochs", ""))

    if "--epochs" in sys.argv:
        print("[warn] --epochs is ignored; use --stage1-epochs/--stage2-epochs.")
    if bool(args.learn_loss_weights) or bool(args.dynamic_loss_weight):
        raise ValueError("Full MAML only supports --learn-loss-weights-mlp.")
    if not bool(args.learn_loss_weights_mlp):
        print("[warn] enabling --learn-loss-weights-mlp for Full MAML.")
        args.learn_loss_weights_mlp = True
    if bool(args.first_order):
        print("[warn] Using first-order MAML (stop-grad through inner update).")
    if float(args.joint_leakyrelu_slope) < 0.0:
        raise ValueError(
            f"--joint-leakyrelu-slope must be >= 0, got {args.joint_leakyrelu_slope}"
        )
    if not (0.0 <= float(args.weight_net_dropout) < 1.0):
        raise ValueError(f"--weight-net-dropout must satisfy 0 <= p < 1, got {args.weight_net_dropout}")
    if int(args.weight_net_layer_embed_dim) <= 0:
        raise ValueError(f"--weight-net-layer-embed-dim must be > 0, got {args.weight_net_layer_embed_dim}")
    if int(args.weight_net_task_embed_dim) <= 0:
        raise ValueError(f"--weight-net-task-embed-dim must be > 0, got {args.weight_net_task_embed_dim}")
    if int(args.weight_net_cnt_grad_hidden_dim) <= 0:
        raise ValueError(
            f"--weight-net-cnt-grad-hidden-dim must be > 0, got {args.weight_net_cnt_grad_hidden_dim}"
        )
    if float(args.grad_clip_norm) < 0.0:
        raise ValueError(f"--grad-clip-norm must be >= 0, got {args.grad_clip_norm}")
    if float(args.phi_grad_clip_norm) < 0.0:
        raise ValueError(f"--phi-grad-clip-norm must be >= 0, got {args.phi_grad_clip_norm}")
    if int(args.meta_seed) != 42:
        print("[warn] meta_seed is fixed to 42; overriding.")
        args.meta_seed = 42
    wb_det, wb_seg, wb_cnt = parse_loss_weights(args.loss_weight_bias)
    base_loss_weights = torch.tensor([float(wb_det), float(wb_seg), float(wb_cnt)], device=device)
    init_loss_weights = torch.tensor([float(wb_det), float(wb_seg), float(wb_cnt)], device=device)
    print(
        "[train] using --loss-weight-bias as base/init: "
        f"[{float(wb_det):.3f}, {float(wb_seg):.3f}, {float(wb_cnt):.3f}]"
    )
    backbone_lr = float(args.backbone_lr) if args.backbone_lr is not None else float(args.lr) * float(args.backbone_lr_mult)
    det_lr = float(args.det_lr) if args.det_lr is not None else float(args.lr)
    seg_lr = float(args.seg_lr) if args.seg_lr is not None else float(args.lr)
    cnt_lr = float(args.cnt_lr) if args.cnt_lr is not None else float(args.lr)
    backbone_wd = float(args.backbone_weight_decay) if args.backbone_weight_decay is not None else float(args.weight_decay)
    det_wd = float(args.det_weight_decay) if args.det_weight_decay is not None else float(args.weight_decay)
    seg_wd = float(args.seg_weight_decay) if args.seg_weight_decay is not None else float(args.weight_decay)
    cnt_wd = float(args.cnt_weight_decay) if args.cnt_weight_decay is not None else float(args.weight_decay)

    if any(flag is not None for flag in (args.det_unfreeze_backbone, args.seg_full_finetune, args.cnt_full_finetune)):
        print("[warn] per-task backbone freeze flags are ignored; use --unfreeze-backbone only.")

    det_train_backbone = bool(args.unfreeze_backbone)
    seg_train_backbone = bool(args.unfreeze_backbone)
    cnt_train_backbone = bool(args.unfreeze_backbone)
    if det_train_backbone:
        print("[info] Backbone is trainable.")
    else:
        print("[info] Backbone is frozen.")

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

    det_train_sampler = det_val_sampler = None
    seg_train_sampler = seg_val_sampler = None
    cnt_train_sampler = cnt_val_sampler = None

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

        cnt_kwargs = dict(
            batch_size=args.cnt_batch_size,
            num_workers=1,
            pin_memory=True,
        )
        cnt_kwargs["persistent_workers"] = True
        cnt_kwargs["multiprocessing_context"] = "spawn"
        cnt_kwargs["prefetch_factor"] = 2

        cnt_train_sampler = DistributedSampler(cnt_train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        cnt_val_sampler = DistributedSampler(cnt_val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        cnt_train_loader = DataLoader(cnt_train_ds, sampler=cnt_train_sampler, shuffle=False, drop_last=True, **cnt_kwargs)
        cnt_val_loader = DataLoader(cnt_val_ds, sampler=cnt_val_sampler, shuffle=False, **cnt_kwargs)

    meta_split = float(args.meta_split)
    if not (0.0 < meta_split < 1.0):
        raise ValueError("--meta-split must be in (0, 1)")

    def _split_train(ds, *, name: str, seed_offset: int):
        n_total = len(ds)
        n_meta = max(1, int(round(n_total * meta_split)))
        n_inner = n_total - n_meta
        if n_inner <= 0:
            raise ValueError(f"meta split too large for {name}: total={n_total}, meta={n_meta}")
        gen = torch.Generator().manual_seed(int(args.meta_seed) + int(seed_offset))
        return random_split(ds, [n_inner, n_meta], generator=gen)

    det_inner_ds, det_meta_ds = _split_train(det_train_ds, name="det", seed_offset=0)
    seg_inner_ds, seg_meta_ds = _split_train(seg_train_ds, name="seg", seed_offset=1)
    cnt_inner_ds, cnt_meta_ds = _split_train(cnt_train_ds, name="cnt", seed_offset=2)

    det_inner_sampler = DistributedSampler(det_inner_ds, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    det_meta_sampler = DistributedSampler(det_meta_ds, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    seg_inner_sampler = DistributedSampler(seg_inner_ds, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    seg_meta_sampler = DistributedSampler(seg_meta_ds, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None

    det_inner_loader = DataLoader(
        det_inner_ds,
        batch_size=args.det_batch_size,
        sampler=det_inner_sampler,
        shuffle=det_inner_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    det_meta_loader = DataLoader(
        det_meta_ds,
        batch_size=args.det_batch_size,
        sampler=det_meta_sampler,
        shuffle=det_meta_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    seg_inner_loader = DataLoader(
        seg_inner_ds,
        batch_size=args.seg_batch_size,
        sampler=seg_inner_sampler,
        shuffle=seg_inner_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    seg_meta_loader = DataLoader(
        seg_meta_ds,
        batch_size=args.seg_batch_size,
        sampler=seg_meta_sampler,
        shuffle=seg_meta_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    cnt_num_workers = 1
    cnt_kwargs = dict(
        batch_size=args.cnt_batch_size,
        num_workers=cnt_num_workers,
        pin_memory=True,
    )
    if cnt_num_workers > 0:
        cnt_kwargs["persistent_workers"] = True
        cnt_kwargs["multiprocessing_context"] = "spawn"
        cnt_kwargs["prefetch_factor"] = 2
    cnt_inner_sampler = DistributedSampler(cnt_inner_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True) if use_ddp else None
    cnt_meta_sampler = DistributedSampler(cnt_meta_ds, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    cnt_inner_loader = DataLoader(
        cnt_inner_ds,
        sampler=cnt_inner_sampler,
        shuffle=cnt_inner_sampler is None,
        drop_last=True,
        **cnt_kwargs,
    )
    cnt_meta_loader = DataLoader(
        cnt_meta_ds,
        sampler=cnt_meta_sampler,
        shuffle=cnt_meta_sampler is None,
        drop_last=False,
        **cnt_kwargs,
    )

    det_num_classes = int(args.det_num_classes) if args.det_num_classes else int(det_train_ds.num_classes)

    # Create shared backbone with optional LoRA-MoE
    shared = SharedDinoV3Backbone(
        model_name=args.model_name,
        image_size=args.image_size,
        checkpoint_path=args.backbone_checkpoint,
        # LoRA-MoE parameters
        use_lora_moe=bool(args.use_lora_moe),
        backbone_trainable=bool(args.unfreeze_backbone),
        task_num=3,  # det, seg, cnt
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
        seg_num_classes=args.seg_num_classes,
        cnt_num_classes=args.cnt_num_classes,
        image_size=args.image_size,
        det_train_backbone=det_train_backbone,
        seg_train_backbone=seg_train_backbone,
        cnt_train_backbone=cnt_train_backbone,
    ).to(device)

    # Full MAML + learn-loss-weights-mlp can be unstable with DDP-wrapped model
    # (inplace-version autograd conflict). Keep multi-process training, but avoid
    # wrapping model by DDP and manually all-reduce theta grads.
    use_model_ddp = use_ddp and (not bool(args.learn_loss_weights_mlp))

    if use_model_ddp:
        ddp_device_ids = [local_rank] if device.type == "cuda" else None
        ddp_output_device = local_rank if device.type == "cuda" else None
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=ddp_device_ids,
            output_device=ddp_output_device,
            find_unused_parameters=False,
        )
    elif use_ddp and is_main_process:
        print("[ddp] multi-process mode without DDP wrapper (manual theta grad all-reduce)")

    model_for_state = model.module if use_model_ddp else model
    manual_theta_grad_sync = use_ddp and (not use_model_ddp)

    # Build theta parameter groups: optional backbone + optional LoRA-MoE + task heads.
    shared_backbone_params = list(model_for_state.shared.backbone.parameters())
    shared_backbone_ids = {id(p) for p in shared_backbone_params}
    backbone_params = [p for p in shared_backbone_params if p.requires_grad]

    theta_param_groups = []
    if backbone_params:
        theta_param_groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": backbone_wd})

    if args.use_lora_moe:
        lora_moe_params = []
        for lora_moe in model_for_state.shared.lora_moes:
            lora_moe_params.extend([p for p in lora_moe.parameters() if p.requires_grad])
        if lora_moe_params:
            theta_param_groups.append({"params": lora_moe_params, "lr": float(args.lr), "weight_decay": float(args.weight_decay)})

        det_head_params = [p for p in model_for_state.detector.parameters() if p.requires_grad and id(p) not in shared_backbone_ids]
        lora_moe_ids = {id(p) for p in lora_moe_params}
        det_head_params = [p for p in det_head_params if id(p) not in lora_moe_ids]
        if det_head_params:
            theta_param_groups.append({"params": det_head_params, "lr": det_lr, "weight_decay": det_wd})

        seg_params = [p for p in model_for_state.seg_head.parameters() if p.requires_grad]
        if seg_params:
            theta_param_groups.append({"params": seg_params, "lr": seg_lr, "weight_decay": seg_wd})

        cnt_head_params = [p for p in model_for_state.cnt_head.parameters() if p.requires_grad]
        if cnt_head_params:
            theta_param_groups.append({"params": cnt_head_params, "lr": cnt_lr, "weight_decay": cnt_wd})

        print(f"[train] Backbone params: {sum(p.numel() for p in backbone_params)}")
        print(f"[LoRA-MoE] LoRA params: {sum(p.numel() for p in lora_moe_params)}")
        print(f"[LoRA-MoE] Det head params: {sum(p.numel() for p in det_head_params)}")
        print(f"[LoRA-MoE] Seg head params: {sum(p.numel() for p in seg_params)}")
        print(f"[LoRA-MoE] Cnt head params: {sum(p.numel() for p in cnt_head_params)}")
    else:
        det_head_params = [p for p in model_for_state.detector.parameters() if p.requires_grad and id(p) not in shared_backbone_ids]
        seg_params = [p for p in model_for_state.seg_head.parameters() if p.requires_grad]
        cnt_head_params = [p for p in model_for_state.cnt_head.parameters() if p.requires_grad]
        if det_head_params:
            theta_param_groups.append({"params": det_head_params, "lr": det_lr, "weight_decay": det_wd})
        if seg_params:
            theta_param_groups.append({"params": seg_params, "lr": seg_lr, "weight_decay": seg_wd})
        if cnt_head_params:
            theta_param_groups.append({"params": cnt_head_params, "lr": cnt_lr, "weight_decay": cnt_wd})
        print(f"[train] Backbone params: {sum(p.numel() for p in backbone_params)}")
        print(f"[train] Det head params: {sum(p.numel() for p in det_head_params)}")
        print(f"[train] Seg head params: {sum(p.numel() for p in seg_params)}")
        print(f"[train] Cnt head params: {sum(p.numel() for p in cnt_head_params)}")

    if not theta_param_groups:
        raise RuntimeError("No trainable parameters.")

    weight_net_use_multilayer_grads = bool(args.weight_net_use_multilayer_grads)
    layer_embed_dim = int(args.weight_net_layer_embed_dim)
    task_embed_dim = int(args.weight_net_task_embed_dim)
    cnt_grad_hidden_dim = int(args.weight_net_cnt_grad_hidden_dim)

    # Baseline (legacy) gradient-input tensors: last layers only.
    det_last_params = list(model_for_state.detector.roi_heads.box_predictor.cls_score.parameters()) + list(
        model_for_state.detector.roi_heads.box_predictor.bbox_pred.parameters()
    )
    seg_last_params = list(model_for_state.seg_head.decode[3].parameters())
    cnt_last_params = list(model_for_state.cnt_head.decode[3].parameters())
    det_last_dim_legacy = int(sum(p.numel() for p in det_last_params))
    seg_last_dim_legacy = int(sum(p.numel() for p in seg_last_params))
    cnt_last_dim_legacy = int(sum(p.numel() for p in cnt_last_params))
    if det_last_dim_legacy <= 0 or seg_last_dim_legacy <= 0 or cnt_last_dim_legacy <= 0:
        raise RuntimeError("Invalid last-layer grad dims for learn-loss-weights-mlp.")

    grad_layer_param_groups = None
    grad_layer_proj = None
    grad_task_fuser = None
    if weight_net_use_multilayer_grads:
        def _collect_layer_param_groups(task_module: nn.Module, task_name: str) -> list[list[torch.nn.Parameter]]:
            layer_param_groups: list[list[torch.nn.Parameter]] = []
            seen_param_ids: set[int] = set()
            for _mod_name, submodule in task_module.named_modules():
                if not isinstance(submodule, (nn.Linear, nn.Conv2d)):
                    continue
                layer_params: list[torch.nn.Parameter] = []
                for p in submodule.parameters(recurse=False):
                    if not p.requires_grad:
                        continue
                    pid = id(p)
                    if pid in seen_param_ids:
                        continue
                    seen_param_ids.add(pid)
                    layer_params.append(p)
                if layer_params:
                    layer_param_groups.append(layer_params)
            if not layer_param_groups:
                raise RuntimeError(
                    f"Multilayer-grad mode: task '{task_name}' has no trainable Linear/Conv2d layers."
                )
            return layer_param_groups

        det_layer_param_groups = _collect_layer_param_groups(model_for_state.detector, "det")
        seg_layer_param_groups = _collect_layer_param_groups(model_for_state.seg_head, "seg")
        cnt_layer_param_groups = _collect_layer_param_groups(model_for_state.cnt_head, "cnt")

        grad_layer_param_groups = {
            "det": det_layer_param_groups,
            "seg": seg_layer_param_groups,
            "cnt": cnt_layer_param_groups,
        }
        det_layer_dims = [int(sum(p.numel() for p in g)) for g in det_layer_param_groups]
        seg_layer_dims = [int(sum(p.numel() for p in g)) for g in seg_layer_param_groups]
        cnt_layer_dims = [int(sum(p.numel() for p in g)) for g in cnt_layer_param_groups]

        grad_layer_proj = nn.ModuleDict(
            {
                "det": nn.ModuleList([nn.Linear(in_dim, layer_embed_dim) for in_dim in det_layer_dims]),
                "seg": nn.ModuleList([nn.Linear(in_dim, layer_embed_dim) for in_dim in seg_layer_dims]),
                "cnt": nn.ModuleList([nn.Linear(in_dim, layer_embed_dim) for in_dim in cnt_layer_dims]),
            }
        )
        grad_task_fuser = nn.ModuleDict(
            {
                "det": nn.Sequential(
                    nn.Linear(len(det_layer_param_groups) * layer_embed_dim, task_embed_dim),
                    nn.LeakyReLU(negative_slope=0.01, inplace=False),
                    nn.Dropout(p=float(args.weight_net_dropout)),
                ),
                "seg": nn.Sequential(
                    nn.Linear(len(seg_layer_param_groups) * layer_embed_dim, task_embed_dim),
                    nn.LeakyReLU(negative_slope=0.01, inplace=False),
                    nn.Dropout(p=float(args.weight_net_dropout)),
                ),
                "cnt": nn.Sequential(
                    nn.Linear(len(cnt_layer_param_groups) * layer_embed_dim, task_embed_dim),
                    nn.LeakyReLU(negative_slope=0.01, inplace=False),
                    nn.Dropout(p=float(args.weight_net_dropout)),
                ),
            }
        )

        # In multilayer mode, each task gradient vector is the learned task embedding.
        det_last_dim = task_embed_dim
        seg_last_dim = task_embed_dim
        cnt_last_dim = task_embed_dim
    else:
        det_last_dim = det_last_dim_legacy
        seg_last_dim = seg_last_dim_legacy
        cnt_last_dim = cnt_last_dim_legacy

    grad_embed_dim = 64
    det_grad_proj = None
    seg_grad_proj = None
    cnt_grad_proj = None
    joint_det_grad_proj = None
    joint_seg_grad_proj = None
    joint_cnt_grad_proj = None
    weight_generator = None
    weight_joint_generator = None
    if args.weight_net_arch == "per_task_shared":
        # Task-specific projectors: gradient/task vector -> 64-d feature (no activation).
        det_grad_proj = nn.Linear(det_last_dim, grad_embed_dim).to(device)
        seg_grad_proj = nn.Linear(seg_last_dim, grad_embed_dim).to(device)
        cnt_grad_proj = nn.Linear(cnt_last_dim, grad_embed_dim).to(device)

        # Shared generator: 64 -> 16 -> 1, with LeakyReLU after each linear.
        weight_generator = nn.Sequential(
            nn.Linear(grad_embed_dim, 16),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Dropout(p=float(args.weight_net_dropout)),
            nn.Linear(16, 1),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
        ).to(device)
    else:
        # Joint architecture:
        # - multilayer off: per-task projection to 64-d, then concatenate.
        # - multilayer on : concatenate task vectors directly.
        # Output activation is applied in _mlp_weights_from_g (configurable).
        joint_in_dim = det_last_dim + seg_last_dim + cnt_last_dim
        if not weight_net_use_multilayer_grads:
            cnt_last_conv = model_for_state.cnt_head.decode[3]
            if not isinstance(cnt_last_conv, nn.Conv2d):
                raise TypeError("Expected counting head last layer to be nn.Conv2d.")
            if cnt_last_conv.kernel_size != (1, 1):
                raise ValueError(
                    f"Expected counting head last conv to be 1x1, got kernel_size={cnt_last_conv.kernel_size}"
                )
            if cnt_last_conv.bias is None:
                raise ValueError("Expected counting head last conv to have bias for grad reshaping.")
            joint_det_grad_proj = nn.Linear(det_last_dim, grad_embed_dim).to(device)
            joint_seg_grad_proj = nn.Linear(seg_last_dim, grad_embed_dim).to(device)
            joint_cnt_grad_proj = CountGradProjector(
                in_channels=int(cnt_last_conv.in_channels),
                num_classes=int(cnt_last_conv.out_channels),
                out_dim=grad_embed_dim,
                hidden_dim=cnt_grad_hidden_dim,
            ).to(device)
            joint_in_dim = grad_embed_dim * 3
        weight_joint_generator = nn.Sequential(
            nn.Linear(joint_in_dim, 16),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Dropout(p=float(args.weight_net_dropout)),
            nn.Linear(16, 3),
        ).to(device)

    phi_dict = {}
    if weight_net_use_multilayer_grads:
        phi_dict["grad_layer_proj"] = grad_layer_proj
        phi_dict["grad_task_fuser"] = grad_task_fuser
    if args.weight_net_arch == "per_task_shared":
        phi_dict["det_grad_proj"] = det_grad_proj
        phi_dict["seg_grad_proj"] = seg_grad_proj
        phi_dict["cnt_grad_proj"] = cnt_grad_proj
        phi_dict["weight_generator"] = weight_generator
    else:
        if not weight_net_use_multilayer_grads:
            phi_dict["joint_det_grad_proj"] = joint_det_grad_proj
            phi_dict["joint_seg_grad_proj"] = joint_seg_grad_proj
            phi_dict["joint_cnt_grad_proj"] = joint_cnt_grad_proj
        phi_dict["weight_joint_generator"] = weight_joint_generator
    phi_modules = nn.ModuleDict(phi_dict).to(device)
    phi_params = list(phi_modules.parameters())
    if use_ddp:
        for p in phi_params:
            dist.broadcast(p.data, src=0)
    print(
        "[train] Learnable loss weights MLP enabled (Full MAML), "
        f"multilayer_grads={'on' if weight_net_use_multilayer_grads else 'off'}, "
        f"arch={args.weight_net_arch}, grad_vec_normalize={args.grad_vec_normalize}, "
        f"prior_fuse={'mul' if bool(args.loss_weight_prior_mul) else 'add'}, "
        f"joint_task_proj64={'on' if (args.weight_net_arch == 'joint' and (not weight_net_use_multilayer_grads)) else 'off'}, "
        f"joint_out={args.joint_weight_out_act if args.weight_net_arch == 'joint' else 'leakyrelu'}, "
        f"joint_leaky_slope={float(args.joint_leakyrelu_slope):.3f}, "
        f"weight_net_dropout={float(args.weight_net_dropout):.3f}, "
        f"cnt_grad_hidden_dim={cnt_grad_hidden_dim}"
    )
    if weight_net_use_multilayer_grads:
        print(
            "[train] multilayer-grad encoder: "
            f"det_layers={len(grad_layer_param_groups['det'])}, "
            f"seg_layers={len(grad_layer_param_groups['seg'])}, "
            f"cnt_layers={len(grad_layer_param_groups['cnt'])}, "
            f"layer_embed_dim={layer_embed_dim}, task_embed_dim={task_embed_dim}"
        )

    theta_named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    theta_params = [p for _, p in theta_named_params]

    train_full_loaders = {"det": det_train_loader, "seg": seg_train_loader, "cnt": cnt_train_loader}
    train_inner_loaders = {"det": det_inner_loader, "seg": seg_inner_loader, "cnt": cnt_inner_loader}
    meta_loaders = {"det": det_meta_loader, "seg": seg_meta_loader, "cnt": cnt_meta_loader}
    val_loaders = {"det": det_val_loader, "seg": seg_val_loader, "cnt": cnt_val_loader}

    lengths_full = {"det": len(det_train_ds), "seg": len(seg_train_ds), "cnt": len(cnt_train_ds)}
    lengths_inner = {"det": len(det_inner_ds), "seg": len(seg_inner_ds), "cnt": len(cnt_inner_ds)}
    primary_full = choose_primary(lengths_full, args.primary_task)
    primary_inner = choose_primary(lengths_inner, args.primary_task)

    inner_cyc = {k: infinite_loader(v) for k, v in train_inner_loaders.items() if k != primary_inner}
    meta_cyc = {k: infinite_loader(v) for k, v in meta_loaders.items()}
    full_cyc = {k: infinite_loader(v) for k, v in train_full_loaders.items() if k != primary_full}

    autocast_device = device.type if device.type in {"cuda", "cpu"} else "cuda"
    scaler = GradScaler(device.type, enabled=bool(args.amp))
    debug_cnt_enabled = bool(getattr(args, "debug_cnt", False))
    debug_cnt_interval = max(int(getattr(args, "debug_cnt_interval", 0) or 0), 0)
    debug_first_n_steps = max(int(getattr(args, "debug_first_n_steps", 0) or 0), 0)

    def _should_debug_step(step: int) -> bool:
        if not debug_cnt_enabled:
            return False
        if debug_first_n_steps > 0 and step <= debug_first_n_steps:
            return True
        return debug_cnt_interval > 0 and (step % debug_cnt_interval == 0)

    def _ddp_mean_scalar(x: float) -> float:
        if not use_ddp:
            return float(x)
        t = torch.tensor(float(x), device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= float(world_size)
        return float(t.item())

    def _ddp_allreduce_param_grads(params: list[torch.nn.Parameter]) -> None:
        if not use_ddp:
            return
        for p in params:
            if p.grad is None:
                zero = torch.zeros_like(p, memory_format=torch.preserve_format)
                dist.all_reduce(zero, op=dist.ReduceOp.SUM)
                continue
            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
            p.grad.div_(float(world_size))

    def _mlp_weights_from_g(det_vec: torch.Tensor, seg_vec: torch.Tensor, cnt_vec: torch.Tensor) -> torch.Tensor:
        if args.weight_net_arch == "per_task_shared":
            det_feat = det_grad_proj(det_vec)
            seg_feat = seg_grad_proj(seg_vec)
            cnt_feat = cnt_grad_proj(cnt_vec)
            det_w = weight_generator(det_feat).squeeze(-1)
            seg_w = weight_generator(seg_feat).squeeze(-1)
            cnt_w = weight_generator(cnt_feat).squeeze(-1)
            w = torch.stack([det_w, seg_w, cnt_w], dim=0)
        else:
            if not weight_net_use_multilayer_grads:
                det_vec = joint_det_grad_proj(det_vec)
                seg_vec = joint_seg_grad_proj(seg_vec)
                cnt_vec = joint_cnt_grad_proj(cnt_vec)
            feat = torch.cat([det_vec, seg_vec, cnt_vec], dim=0)
            w = weight_joint_generator(feat)
            if args.joint_weight_out_act == "sigmoid":
                w = torch.sigmoid(w)
            else:
                w = F.leaky_relu(w, negative_slope=float(args.joint_leakyrelu_slope))
        return torch.nan_to_num(w, nan=0.0, posinf=1e6, neginf=-1e6)

    def _compose_task_weights(raw_w: torch.Tensor) -> torch.Tensor:
        raw_w = torch.nan_to_num(raw_w, nan=0.0, posinf=1e6, neginf=-1e6)
        if bool(args.loss_weight_prior_mul):
            fused = raw_w * base_loss_weights
        else:
            fused = raw_w + base_loss_weights
        return torch.nan_to_num(fused, nan=0.0, posinf=1e6, neginf=-1e6)

    def _normalize_grad(vec: torch.Tensor) -> torch.Tensor:
        vec = torch.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=0.0)
        if args.grad_vec_normalize == "none":
            return vec.detach()
        eps = 1e-12
        norm = vec.norm()
        if not torch.isfinite(norm):
            return torch.zeros_like(vec)
        return (vec / (norm + eps)).detach()

    def _flat_grads(
        params: list[torch.nn.Parameter],
        total_dim: int,
        grad_map: Dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        parts = []
        for p in params:
            g = grad_map.get(id(p)) if grad_map is not None else p.grad
            if g is None:
                parts.append(torch.zeros(p.numel(), device=device))
            else:
                parts.append(g.detach().float().reshape(-1))
        if not parts:
            return torch.zeros(total_dim, device=device)
        return torch.cat(parts, dim=0)

    def _task_grad_vec_from_layers(
        task_name: str,
        layer_param_groups: list[list[torch.nn.Parameter]],
        grad_map: Dict[int, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        layer_feats = []
        for idx, layer_params in enumerate(layer_param_groups):
            layer_dim = int(sum(p.numel() for p in layer_params))
            layer_grad_vec = _flat_grads(layer_params, layer_dim, grad_map)
            layer_in = layer_grad_vec
            layer_in = _normalize_grad(layer_in)
            layer_feat = grad_layer_proj[task_name][idx](layer_in)
            layer_feat = F.leaky_relu(layer_feat, negative_slope=0.01)
            layer_feats.append(layer_feat)
        if not layer_feats:
            return torch.zeros(task_embed_dim, device=device)
        task_in = torch.cat(layer_feats, dim=0)
        task_vec = grad_task_fuser[task_name](task_in)
        return task_vec

    def _grad_vecs_from_grads(grad_map: Dict[int, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if weight_net_use_multilayer_grads:
            det_vec = _task_grad_vec_from_layers("det", grad_layer_param_groups["det"], grad_map)
            seg_vec = _task_grad_vec_from_layers("seg", grad_layer_param_groups["seg"], grad_map)
            cnt_vec = _task_grad_vec_from_layers("cnt", grad_layer_param_groups["cnt"], grad_map)
        else:
            det_vec = _normalize_grad(_flat_grads(det_last_params, det_last_dim, grad_map))
            seg_vec = _normalize_grad(_flat_grads(seg_last_params, seg_last_dim, grad_map))
            cnt_vec = _normalize_grad(_flat_grads(cnt_last_params, cnt_last_dim, grad_map))
        return det_vec, seg_vec, cnt_vec

    def _grad_vecs_from_param_grads() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if weight_net_use_multilayer_grads:
            det_vec = _task_grad_vec_from_layers("det", grad_layer_param_groups["det"])
            seg_vec = _task_grad_vec_from_layers("seg", grad_layer_param_groups["seg"])
            cnt_vec = _task_grad_vec_from_layers("cnt", grad_layer_param_groups["cnt"])
        else:
            det_vec = _normalize_grad(_flat_grads(det_last_params, det_last_dim))
            seg_vec = _normalize_grad(_flat_grads(seg_last_params, seg_last_dim))
            cnt_vec = _normalize_grad(_flat_grads(cnt_last_params, cnt_last_dim))
        return det_vec, seg_vec, cnt_vec

    def _ddp_avg_grad_vecs(g_vecs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not use_ddp:
            return g_vecs
        out = []
        for v in g_vecs:
            vv = v.detach().clone()
            dist.all_reduce(vv, op=dist.ReduceOp.SUM)
            vv /= float(world_size)
            out.append(vv)
        return out[0], out[1], out[2]

    def _clip_grads(grads: tuple[torch.Tensor, ...], max_norm: float) -> tuple[torch.Tensor, ...]:
        if max_norm <= 0:
            return grads
        total_sq = 0.0
        for g in grads:
            if g is None:
                continue
            total_sq += float(g.detach().float().pow(2).sum().item())
        total_norm = math.sqrt(total_sq)
        if total_norm <= max_norm:
            return grads
        scale = float(max_norm) / (total_norm + 1e-6)
        out = []
        for g in grads:
            out.append(g * scale if g is not None else None)
        return tuple(out)

    def _build_param_dict(theta_updates: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        params = {name: p for name, p in model.named_parameters()}
        params.update(theta_updates)
        return params

    def _compute_losses(
        det_batch,
        seg_batch,
        cnt_batch,
        *,
        params: Dict[str, torch.Tensor] | None = None,
        amp: bool = False,
        collect_cnt_stats: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, float] | None]:
        det_images, det_targets = _to_device_det(det_batch, device)
        seg_imgs, seg_masks = _to_device_seg(seg_batch, device)
        cnt_imgs, cnt_dens = _to_device_cnt(cnt_batch, device)
        cnt_gt_counts = cnt_dens.flatten(2).sum(dim=2)

        with torch.amp.autocast(autocast_device, enabled=amp):
            if params is None:
                det_loss_dict = model("det", det_images, det_targets)
            else:
                det_loss_dict = functional_call(model, params, args=("det", det_images, det_targets), kwargs={})
            det_loss = sum(det_loss_dict.values())

            if params is None:
                seg_logits = model("seg", seg_imgs)
                pred_dens, pred_counts = model(
                    "cnt",
                    cnt_imgs,
                    cnt_backbone_grad_mult=float(args.cnt_backbone_grad_mult),
                )
            else:
                seg_logits = functional_call(model, params, args=("seg", seg_imgs), kwargs={})
                pred_dens, pred_counts = functional_call(
                    model,
                    params,
                    args=("cnt", cnt_imgs),
                    kwargs={"cnt_backbone_grad_mult": float(args.cnt_backbone_grad_mult)},
                )

            seg_loss = F.cross_entropy(seg_logits, seg_masks)
            dens_loss = F.mse_loss(pred_dens, cnt_dens, reduction="sum") / cnt_imgs.size(0)
            cnt_l1 = F.l1_loss(pred_counts, cnt_gt_counts)
            cnt_loss = dens_loss + float(args.cnt_count_loss_weight) * cnt_l1

        cnt_stats = None
        if collect_cnt_stats:
            cnt_stats = _count_diag_stats(pred_dens, cnt_dens, pred_counts, cnt_gt_counts)

        return det_loss, seg_loss, cnt_loss, cnt_stats

    def _current_loss_weights(g_vecs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> tuple[float, float, float]:
        with torch.no_grad():
            w = _compose_task_weights(_mlp_weights_from_g(*g_vecs)).detach().float().cpu().tolist()
        return float(w[0]), float(w[1]), float(w[2])

    def _run_validation(epoch: int, stage: str) -> tuple[float, Dict[str, float]]:
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
            debug_cnt=debug_cnt_enabled,
        )
        val_ap50 = _eval_det_ap50_fast(
            model,
            val_loaders["det"],
            device,
            num_classes=det_num_classes,
            score_thresh=float(getattr(args, "det_ap_score_thr", 0.0)),
        )

        combo_metric = float(val_ap50) + float(val_seg_miou) + 1.0 / max(float(val_cnt_mae), 1e-8)

        print(
            f"[{stage}] epoch {epoch} | "
            f"val det {val_det:.4f} seg {val_seg:.4f} miou {val_seg_miou:.4f} "
            f"cnt {val_cnt:.4f} dens {val_cnt_density:.6e} mae {val_cnt_mae:.4f} total_mae {val_cnt_total_mae:.4f} | "
            f"ap50 {val_ap50:.4f} | "
            f"combo {combo_metric:.6f}"
        )
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
        }
        return combo_metric, metrics

    def _state_dict_cpu_clone(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if torch.is_tensor(v):
                out[k] = v.detach().cpu().clone()
            else:
                out[k] = deepcopy(v)
        return out

    def _init_g0() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Initialize g0 from one inner-batch gradient snapshot (single-task style).
        In distributed mode, ONLY rank0 computes g0 once, then broadcasts to all ranks.

        Fallback to deterministic zeros on any runtime issue to keep training robust.
        """
        g0_det = torch.zeros(det_last_dim, device=device)
        g0_seg = torch.zeros(seg_last_dim, device=device)
        g0_cnt = torch.zeros(cnt_last_dim, device=device)

        should_compute = (not use_ddp) or (rank == 0)
        if should_compute:
            model.train()
            phi_modules.train()
            try:
                det_batch = next(iter(train_inner_loaders["det"]))
                seg_batch = next(iter(train_inner_loaders["seg"]))
                cnt_batch = next(iter(train_inner_loaders["cnt"]))

                det_loss, seg_loss, cnt_loss, _ = _compute_losses(det_batch, seg_batch, cnt_batch, amp=False)
                l_in = init_loss_weights[0] * det_loss + init_loss_weights[1] * seg_loss + init_loss_weights[2] * cnt_loss

                grads = torch.autograd.grad(l_in, theta_params, create_graph=False, allow_unused=True)
                grads = _clip_grads(grads, float(args.grad_clip_norm))
                grad_map = {id(p): g for p, g in zip(theta_params, grads) if g is not None}
                g0_det, g0_seg, g0_cnt = _grad_vecs_from_grads(grad_map)
                if is_main_process:
                    print("[init] g0 initialized on rank0 from first inner-batch gradients and broadcast to all ranks.")
            except Exception as exc:
                if is_main_process:
                    print(f"[warn] g0 gradient init on rank0 failed ({exc}); fallback to deterministic zeros.")

        if use_ddp:
            dist.broadcast(g0_det, src=0)
            dist.broadcast(g0_seg, src=0)
            dist.broadcast(g0_cnt, src=0)

        return g0_det, g0_seg, g0_cnt

    # -------------------------
    # Stage1: Full MAML
    # -------------------------
    g_t = _init_g0()
    stage1_epochs = int(args.stage1_epochs)
    val_every = int(getattr(args, "val_every", 1) or 1)
    if val_every != 1:
        print("[warn] --val-every is ignored; validation runs every epoch when enabled.")
    if val_every < 1:
        raise ValueError("--val-every must be >= 1")
    stage1_do_validation = (not bool(args.skip_validation)) and (not bool(args.select_best_from_stage2))

    best_metric = -math.inf
    best_state = None
    best_phi_state = None
    best_epoch = None
    best_stage = None
    best_loss_weights = None

    # Stage1 first-order AdamW-MAML:
    # - inner update (theta) by AdamW on inner batch loss
    # - outer update (phi) by AdamW on meta batch loss using updated theta
    optimizer_theta_s1 = torch.optim.AdamW(theta_param_groups)
    optimizer_phi_s1 = torch.optim.AdamW(phi_params, lr=float(args.meta_beta), weight_decay=0.0)

    for epoch in range(1, stage1_epochs + 1):
        if use_ddp:
            for sampler in (det_inner_loader.sampler, seg_inner_loader.sampler, cnt_inner_loader.sampler, det_meta_loader.sampler, seg_meta_loader.sampler, cnt_meta_loader.sampler):
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(epoch)
        model.train()
        phi_modules.train()
        total_loss = 0.0
        det_loss_sum = 0.0
        seg_loss_sum = 0.0
        cnt_loss_sum = 0.0
        steps = 0

        for step, primary_batch in enumerate(train_inner_loaders[primary_inner], start=1):
            batches_in = {primary_inner: primary_batch}
            for k in train_inner_loaders.keys():
                if k != primary_inner:
                    batches_in[k] = next(inner_cyc[k])
            batches_meta = {k: next(meta_cyc[k]) for k in meta_loaders.keys()}
            debug_this_step = _should_debug_step(step)

            w_t_inner = _compose_task_weights(_mlp_weights_from_g(*g_t)).detach()

            optimizer_theta_s1.zero_grad(set_to_none=True)
            det_loss, seg_loss, cnt_loss, cnt_stats = _compute_losses(
                batches_in["det"],
                batches_in["seg"],
                batches_in["cnt"],
                amp=False,
                collect_cnt_stats=debug_this_step,
            )
            l_in = w_t_inner[0] * det_loss + w_t_inner[1] * seg_loss + w_t_inner[2] * cnt_loss

            l_in.backward()
            if manual_theta_grad_sync:
                _ddp_allreduce_param_grads(theta_params)
            grad_norm_before_clip, _ = _param_grad_l2_norm(theta_params)
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(theta_params, max_norm=float(args.grad_clip_norm))
            grad_norm_after_clip, _ = _param_grad_l2_norm(theta_params)
            cnt_grad_norm, cnt_grad_param_count = _param_grad_l2_norm(cnt_head_params)
            cnt_step_delta_est = float(optimizer_theta_s1.param_groups[0]["lr"]) * cnt_grad_norm

            with torch.no_grad():
                g_t = _ddp_avg_grad_vecs(_grad_vecs_from_param_grads())

            optimizer_theta_s1.step()

            det_meta, seg_meta, cnt_meta, _ = _compute_losses(
                batches_meta["det"],
                batches_meta["seg"],
                batches_meta["cnt"],
                amp=False,
            )
            w_t_meta = _compose_task_weights(_mlp_weights_from_g(*g_t))
            l_meta = w_t_meta[0] * det_meta + w_t_meta[1] * seg_meta + w_t_meta[2] * cnt_meta

            phi_grads = torch.autograd.grad(l_meta, phi_params, allow_unused=True)
            optimizer_phi_s1.zero_grad(set_to_none=True)
            for p, g in zip(phi_params, phi_grads):
                if g is None:
                    continue
                p.grad = g
            _ddp_allreduce_param_grads(phi_params)
            if float(args.phi_grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(phi_params, max_norm=float(args.phi_grad_clip_norm))
            optimizer_phi_s1.step()

            det_loss_sum += float(det_loss.detach().item())
            seg_loss_sum += float(seg_loss.detach().item())
            cnt_loss_sum += float(cnt_loss.detach().item())
            total_loss += float(l_in.detach().item())
            steps += 1

            if args.log_interval and step % args.log_interval == 0:
                w0, w1, w2 = [float(x) for x in w_t_meta.detach().cpu().tolist()]
                print(
                    f"[stage1] epoch {epoch}/{stage1_epochs} step {step} | "
                    f"loss {total_loss/max(steps,1):.4f} | "
                    f"det {det_loss_sum/max(steps,1):.4f} seg {seg_loss_sum/max(steps,1):.4f} "
                    f"cnt {cnt_loss_sum/max(steps,1):.4f} | "
                    f"w [{w0:.3f}, {w1:.3f}, {w2:.3f}]"
                )
            if debug_this_step and cnt_stats is not None:
                print(
                    f"[diag][cnt][stage1] epoch {epoch}/{stage1_epochs} step {step} | "
                    f"{_format_count_diag(cnt_stats)} | "
                    f"grad(all) {grad_norm_before_clip:.6e}->{grad_norm_after_clip:.6e} | "
                    f"grad(cnt_head) {cnt_grad_norm:.6e} params {cnt_grad_param_count} | "
                    f"meta_alpha {float(args.meta_alpha):.3e} est_cnt_delta_l2 {cnt_step_delta_est:.6e}"
                )

            if args.max_train_steps and step >= args.max_train_steps:
                break

        avg_train = total_loss / max(steps, 1)
        print(f"[stage1] epoch {epoch}/{stage1_epochs} | train {avg_train:.4f}")
        w0, w1, w2 = _current_loss_weights(g_t)
        print(f"[stage1] epoch {epoch}/{stage1_epochs} | w [{w0:.3f}, {w1:.3f}, {w2:.3f}]")
        if stage1_do_validation:
            combo_metric, _metrics = _run_validation(epoch, "stage1")
            if combo_metric > best_metric:
                best_metric = float(combo_metric)
                best_epoch = epoch
                best_stage = "stage1"
                best_loss_weights = _current_loss_weights(g_t)
                if is_main_process:
                    best_state = _state_dict_cpu_clone(model_for_state.state_dict())
                    best_phi_state = _state_dict_cpu_clone(phi_modules.state_dict())
                    print(f"[ckpt] new best cached (stage1 epoch {epoch}, combo {best_metric:.6f})")

    stage1_phi_path = save_dir / "stage1_phi_last.pt"
    if is_main_process:
        torch.save(
            {
                "epoch": int(stage1_epochs),
                "phi_state": phi_modules.state_dict(),
                "loss_weights": _current_loss_weights(g_t),
            },
            stage1_phi_path,
        )
        print(f"[ckpt] saved stage1 phi -> {stage1_phi_path}")

    # Release Stage1 optimizer states before Stage2 to avoid duplicated Adam moments in VRAM.
    del optimizer_theta_s1
    del optimizer_phi_s1
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # -------------------------
    # Stage2: Freeze phi, train theta
    # -------------------------
    for p in phi_params:
        p.requires_grad = False
    phi_modules.eval()
    if bool(args.select_best_from_stage2):
        best_metric = float("-inf")
        best_state = None
        best_phi_state = None
        best_epoch = None
        best_stage = None
        best_loss_weights = None
        if is_main_process:
            print("[train] validation/model selection start from Stage2 (Stage1 validation disabled).")

    optimizer = torch.optim.AdamW(theta_param_groups)
    best_path = save_dir / "best_combo.pt"

    stage2_epochs = int(args.stage2_epochs)
    for epoch in range(1, stage2_epochs + 1):
        if use_ddp:
            for sampler in (det_train_loader.sampler, seg_train_loader.sampler, cnt_train_loader.sampler):
                if isinstance(sampler, DistributedSampler):
                    sampler.set_epoch(epoch + stage1_epochs)
        model.train()
        total_loss = 0.0
        det_loss_sum = 0.0
        seg_loss_sum = 0.0
        cnt_loss_sum = 0.0
        steps = 0
        cnt_epoch_start = [p.detach().float().clone() for p in cnt_head_params]

        for step, primary_batch in enumerate(train_full_loaders[primary_full], start=1):
            batches = {primary_full: primary_batch}
            for k in train_full_loaders.keys():
                if k != primary_full:
                    batches[k] = next(full_cyc[k])
            debug_this_step = _should_debug_step(step)

            optimizer.zero_grad(set_to_none=True)

            w_t = _compose_task_weights(_mlp_weights_from_g(*g_t))
            det_loss, seg_loss, cnt_loss, cnt_stats = _compute_losses(
                batches["det"],
                batches["seg"],
                batches["cnt"],
                amp=False,
                collect_cnt_stats=debug_this_step,
            )
            total = w_t[0] * det_loss + w_t[1] * seg_loss + w_t[2] * cnt_loss

            total.backward()
            if manual_theta_grad_sync:
                _ddp_allreduce_param_grads(theta_params)

            cnt_grad_norm, cnt_grad_param_count = _param_grad_l2_norm(cnt_head_params)
            if float(args.grad_clip_norm) > 0:
                torch.nn.utils.clip_grad_norm_(theta_params, max_norm=float(args.grad_clip_norm))

            with torch.no_grad():
                g_t = _ddp_avg_grad_vecs(_grad_vecs_from_param_grads())

            optimizer.step()

            det_loss_sum += float(det_loss.detach().item())
            seg_loss_sum += float(seg_loss.detach().item())
            cnt_loss_sum += float(cnt_loss.detach().item())
            total_loss += float(total.detach().item())
            steps += 1

            if args.log_interval and step % args.log_interval == 0:
                w0, w1, w2 = [float(x) for x in w_t.detach().cpu().tolist()]
                print(
                    f"[stage2] epoch {epoch}/{stage2_epochs} step {step} | "
                    f"loss {total_loss/max(steps,1):.4f} | "
                    f"det {det_loss_sum/max(steps,1):.4f} seg {seg_loss_sum/max(steps,1):.4f} "
                    f"cnt {cnt_loss_sum/max(steps,1):.4f} | "
                    f"w [{w0:.3f}, {w1:.3f}, {w2:.3f}] | "
                    f"cnt_gnorm {cnt_grad_norm:.6e} cnt_gparams {cnt_grad_param_count}"
                )
            if debug_this_step and cnt_stats is not None:
                cnt_head_last = model_for_state.cnt_head.decode[-1] if len(model_for_state.cnt_head.decode) > 0 else None
                if isinstance(cnt_head_last, nn.Conv2d) and cnt_head_last.bias is not None:
                    bias_mean = float(cnt_head_last.bias.detach().float().mean().item())
                    bias_min = float(cnt_head_last.bias.detach().float().min().item())
                    bias_max = float(cnt_head_last.bias.detach().float().max().item())
                else:
                    bias_mean = float("nan")
                    bias_min = float("nan")
                    bias_max = float("nan")
                print(
                    f"[diag][cnt][stage2] epoch {epoch}/{stage2_epochs} step {step} | "
                    f"{_format_count_diag(cnt_stats)} | "
                    f"grad(cnt_head) {cnt_grad_norm:.6e} params {cnt_grad_param_count} | "
                    f"head_last_bias mean/min/max {bias_mean:.6e}/{bias_min:.6e}/{bias_max:.6e}"
                )

            if args.max_train_steps and step >= args.max_train_steps:
                break

        avg_train = total_loss / max(steps, 1)
        cnt_param_delta = _param_delta_l2_norm(cnt_head_params, cnt_epoch_start)
        print(f"[stage2] epoch {epoch}/{stage2_epochs} | train {avg_train:.4f}")
        w0, w1, w2 = _current_loss_weights(g_t)
        print(f"[stage2] epoch {epoch}/{stage2_epochs} | w [{w0:.3f}, {w1:.3f}, {w2:.3f}]")
        print(f"[diag][cnt][stage2] epoch {epoch}/{stage2_epochs} | param_delta_l2 {cnt_param_delta:.6e}")

        if not bool(args.skip_validation):
            combo_metric, metrics = _run_validation(epoch, "stage2")
            if combo_metric > best_metric:
                best_metric = float(combo_metric)
                best_epoch = epoch
                best_stage = "stage2"
                best_loss_weights = _current_loss_weights(g_t)
                if is_main_process:
                    best_state = _state_dict_cpu_clone(model_for_state.state_dict())
                    best_phi_state = _state_dict_cpu_clone(phi_modules.state_dict())
                    print(f"[ckpt] new best cached (stage2 epoch {epoch}, combo {best_metric:.6f})")

    if is_main_process and best_state is not None:
        model_for_state.load_state_dict(best_state)
        save_multitask_checkpoint(
            str(best_path),
            model=model_for_state,
            optimizer=optimizer,
            epoch=int(best_epoch or stage2_epochs),
            best_by="combo",
            metrics={
                "best_metric": float(best_metric),
                "best_stage": str(best_stage or ""),
                "best_epoch": int(best_epoch or stage2_epochs),
            },
            loss_weights=best_loss_weights or _current_loss_weights(g_t),
            phi_state=best_phi_state,
            config={
                "use_lora_moe": bool(args.use_lora_moe),
                "unfreeze_backbone": bool(args.unfreeze_backbone),
                "det_train_backbone": bool(det_train_backbone),
                "seg_train_backbone": bool(seg_train_backbone),
                "cnt_train_backbone": bool(cnt_train_backbone),
                "backbone_lr": float(backbone_lr),
                "backbone_weight_decay": float(backbone_wd),
                "lora_rank": int(args.lora_rank),
                "num_experts_private": int(args.num_experts_private),
                "num_experts_shared": int(args.num_experts_shared),
                "moe_k_private": int(args.moe_k_private),
                "moe_k_shared": int(args.moe_k_shared),
            },
        )
    if is_main_process:
        print(f"[ckpt] saved best -> {best_path} (combo {best_metric:.6f})")

    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

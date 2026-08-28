from __future__ import annotations

import argparse
import builtins
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from object_detection.dataset import collate_fn

from .datasets import build_cnt_loaders, build_det_loaders, build_seg_loaders
from .models import MultiTaskModel, SharedDinoV3Backbone
from .policy import DirichletWeightGenerator
from .swanlab_logger import MetricLogger, create_metric_logger
from .state_features import SharedExpertStateTracker
from .trainer_stage1 import run_stage1
from .trainer_stage2 import run_stage2
from .trainer_warmup import run_warmup
from .utils import choose_primary, load_multitask_checkpoint, parse_loss_weights, save_multitask_checkpoint


def parse_args() -> argparse.Namespace:
    # 训练入口参数：只保留新方案真正会用到的项。
    p = argparse.ArgumentParser(description="GRPO multitask training with DINOv3 + LoRA-MoE")

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
    p.add_argument("--stage2-det-val-ann", type=str, default=None)
    p.add_argument("--stage2-det-val-img-dir", type=str, default=None)
    p.add_argument("--det-num-classes", type=int, default=None)
    p.add_argument("--det-ap-score-thr", type=float, default=0.0)

    p.add_argument("--seg-train-dir", type=str, required=True)
    p.add_argument("--seg-val-dir", type=str, required=True)
    p.add_argument("--stage2-seg-val-dir", type=str, default=None)
    p.add_argument("--seg-num-classes", type=int, default=11)

    p.add_argument("--cnt-data-root", type=str, required=True)
    p.add_argument("--cnt-train-dir", type=str, default=None)
    p.add_argument("--cnt-val-dir", type=str, default=None)
    p.add_argument("--stage2-cnt-val-dir", type=str, default=None)
    p.add_argument("--cnt-num-classes", type=int, default=8)
    p.add_argument("--cnt-count-loss-weight", type=float, default=1.0)
    aspect = p.add_mutually_exclusive_group()
    aspect.add_argument("--cnt-keep-aspect", dest="cnt_keep_aspect", action="store_true")
    aspect.add_argument("--cnt-no-keep-aspect", dest="cnt_keep_aspect", action="store_false")
    p.set_defaults(cnt_keep_aspect=True)

    p.add_argument("--stage1-epochs", type=int, default=100)
    p.add_argument("--stage2-epochs", type=int, default=50)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--warmup-loss-weights", type=str, default="15,8,1")
    p.add_argument("--warmup-save-path", type=str, default=None)
    p.add_argument("--warmup-load-path", type=str, default=None)
    p.add_argument("--meta-alpha", type=float, default=5e-4, help="virtual update step size")
    p.add_argument("--meta-beta", type=float, default=1e-3, help="generator learning rate")
    p.add_argument("--det-batch-size", type=int, default=2)
    p.add_argument("--seg-batch-size", type=int, default=2)
    p.add_argument("--cnt-batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr", type=float, default=None)
    p.add_argument("--backbone-lr-mult", type=float, default=0.1)
    p.add_argument("--det-lr", type=float, default=None)
    p.add_argument("--seg-lr", type=float, default=None)
    p.add_argument("--cnt-lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--backbone-weight-decay", type=float, default=None)
    p.add_argument("--det-weight-decay", type=float, default=None)
    p.add_argument("--seg-weight-decay", type=float, default=None)
    p.add_argument("--cnt-weight-decay", type=float, default=None)
    p.add_argument("--primary-task", type=str, default=None)
    p.add_argument("--save-dir", type=str, default="runs/grpo_multitask")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--debug-step-timing", action="store_true")
    p.add_argument("--debug-step-timing-interval", type=int, default=1)
    p.add_argument("--debug-reward-details", action="store_true")
    p.add_argument("--debug-reward-details-interval", type=int, default=1)
    p.add_argument("--grad-clip-norm", type=float, default=100.0)
    p.add_argument("--phi-grad-clip-norm", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=4)
    # 这里不提前探测 CUDA，避免在参数解析阶段触发驱动检查。
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-steps", type=int, default=0)
    p.add_argument("--max-val-steps", type=int, default=0)
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument("--select-best-from-stage2", action="store_true", default=True)
    p.add_argument("--cnt-backbone-grad-mult", type=float, default=1.0)

    p.add_argument("--generator-hidden-dim", type=int, default=192)
    p.add_argument("--policy-weight-prior", type=str, default="15,8,1")
    p.add_argument("--num-candidates", type=int, default=6)
    p.add_argument("--candidate-smoothing-gamma", type=float, default=0.05)
    p.add_argument("--grpo-clip-eps", type=float, default=0.2)
    p.add_argument("--policy-kl-beta", type=float, default=5e-2)
    p.add_argument("--state-matrix-proj-dim", type=int, default=128)
    p.add_argument("--state-task-hidden-dim", type=int, default=128, help="deprecated: task projector is now a direct linear map to state-task-dim")
    p.add_argument("--state-task-dim", type=int, default=128)
    p.add_argument("--use-swanlab", action="store_true")
    p.add_argument("--swanlab-project", type=str, default=None)
    p.add_argument("--swanlab-workspace", type=str, default=None)
    p.add_argument("--swanlab-experiment-name", type=str, default=None)
    p.add_argument("--swanlab-mode", type=str, default=None)
    p.add_argument("--swanlab-logdir", type=str, default=None)
    # Deprecated: warmup 直接复用主训练集，这些参数仅保留给旧脚本兼容。
    p.add_argument("--warmup-det-data-root", type=str, default=None)
    p.add_argument("--warmup-det-train-ann", type=str, default=None)
    p.add_argument("--warmup-det-train-img-dir", type=str, default=None)
    p.add_argument("--warmup-seg-train-dir", type=str, default=None)
    p.add_argument("--warmup-cnt-data-root", type=str, default=None)
    p.add_argument("--warmup-cnt-train-dir", type=str, default=None)

    return p.parse_args()


def _rebuild_ddp_loader(dataset, *, batch_size: int, sampler, num_workers: int, collate=None, drop_last: bool = False):
    # 按统一规则重建 DDP loader，避免三套重复代码。
    kwargs = {
        "batch_size": batch_size,
        "sampler": sampler,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": drop_last,
    }
    if collate is not None:
        kwargs["collate_fn"] = collate
    if num_workers > 0 and collate is None:
        kwargs["persistent_workers"] = True
    return DataLoader(dataset, **kwargs)


def _broadcast_module_state(module: torch.nn.Module) -> None:
    # 手动广播初始化状态，确保各 rank 起点一致。
    if not dist.is_initialized():
        return
    for tensor in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(tensor.data, src=0)


def main() -> None:
    # main 负责环境初始化、模块装配和两阶段调度。
    args = parse_args()
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

    is_main_process = (not use_ddp) or rank == 0
    if use_ddp and not is_main_process:
        builtins.print = lambda *a, **k: None

    # 固定随机种子，尽量保证可复现。
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    save_dir = Path(args.save_dir)
    if is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)
    if use_ddp:
        dist.barrier()
    metric_logger: MetricLogger = create_metric_logger(
        args=args,
        is_main_process=is_main_process,
        save_dir=save_dir,
    )
    warmup_save_path = Path(args.warmup_save_path) if args.warmup_save_path else (save_dir / "warmup_best.pt")
    if args.warmup_load_path is None:
        warmup_load_path = warmup_save_path
    else:
        warmup_load_path = Path(args.warmup_load_path) if str(args.warmup_load_path).strip() else None

    if float(args.grad_clip_norm) < 0:
        raise ValueError("--grad-clip-norm must be >= 0")
    if float(args.phi_grad_clip_norm) < 0:
        raise ValueError("--phi-grad-clip-norm must be >= 0")
    if float(args.policy_kl_beta) < 0:
        raise ValueError("--policy-kl-beta must be >= 0")
    # 新状态特征直接来自共享专家，因此必须启用 LoRA-MoE。
    if not bool(args.use_lora_moe):
        raise ValueError("GRPO training requires --use-lora-moe because state features use shared experts.")

    # 先构建三任务数据集和默认 loader。
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

    warmup_train_ds = None
    warmup_train_loaders = None
    warmup_primary_task = None
    # Legacy warmup-only dataset path is disabled; warmup reuses the main training loaders below.
    if False and int(args.warmup_epochs) > 0:
        # 预热单独使用旧项目训练集，避免影响当前 Stage1/Stage2 数据配置。
        warmup_det_root = args.warmup_det_data_root or args.det_data_root
        warmup_det_ann = args.warmup_det_train_ann or args.det_train_ann
        warmup_det_img_dir = args.warmup_det_train_img_dir
        warmup_seg_dir = args.warmup_seg_train_dir or args.seg_train_dir
        warmup_cnt_root = args.warmup_cnt_data_root or args.cnt_data_root
        warmup_cnt_dir = args.warmup_cnt_train_dir or args.cnt_train_dir

        warmup_det_train_ds, _, warmup_det_train_loader, _ = build_det_loaders(
            data_root=warmup_det_root,
            image_size=args.image_size,
            batch_size=args.det_batch_size,
            num_workers=args.num_workers,
            train_ann=warmup_det_ann,
            val_ann=args.stage2_det_val_ann or args.det_val_ann,
            train_img_dir=warmup_det_img_dir,
            val_img_dir=args.stage2_det_val_img_dir or args.det_val_img_dir,
        )
        warmup_seg_train_ds, _, warmup_seg_train_loader, _ = build_seg_loaders(
            train_dir=warmup_seg_dir,
            val_dir=args.stage2_seg_val_dir or args.seg_val_dir,
            image_size=args.image_size,
            batch_size=args.seg_batch_size,
            num_workers=args.num_workers,
        )
        warmup_cnt_train_ds, _, warmup_cnt_train_loader, _ = build_cnt_loaders(
            data_root=warmup_cnt_root,
            train_dir=warmup_cnt_dir,
            val_dir=args.stage2_cnt_val_dir or args.cnt_val_dir,
            image_size=args.image_size,
            num_classes=args.cnt_num_classes,
            keep_aspect=bool(args.cnt_keep_aspect),
            batch_size=args.cnt_batch_size,
            num_workers=1,
        )
        warmup_train_ds = {
            "det": warmup_det_train_ds,
            "seg": warmup_seg_train_ds,
            "cnt": warmup_cnt_train_ds,
        }
        warmup_train_loaders = {
            "det": warmup_det_train_loader,
            "seg": warmup_seg_train_loader,
            "cnt": warmup_cnt_train_loader,
        }
        warmup_primary_task = choose_primary(
            {
                "det": len(warmup_det_train_ds),
                "seg": len(warmup_seg_train_ds),
                "cnt": len(warmup_cnt_train_ds),
            },
            args.primary_task,
        )

    use_stage2_custom_val = any(
        [
            args.stage2_det_val_ann,
            args.stage2_det_val_img_dir,
            args.stage2_seg_val_dir,
            args.stage2_cnt_val_dir,
        ]
    )
    if use_stage2_custom_val:
        # Stage2 选模可单独指定原始 val，和 Stage1 reward 的验证集解耦。
        _, det_stage2_val_ds, _, det_stage2_val_loader = build_det_loaders(
            data_root=args.det_data_root,
            image_size=args.image_size,
            batch_size=args.det_batch_size,
            num_workers=args.num_workers,
            train_ann=args.det_train_ann,
            val_ann=args.stage2_det_val_ann or args.det_val_ann,
            train_img_dir=args.det_train_img_dir,
            val_img_dir=args.stage2_det_val_img_dir or args.det_val_img_dir,
        )
        _, seg_stage2_val_ds, _, seg_stage2_val_loader = build_seg_loaders(
            train_dir=args.seg_train_dir,
            val_dir=args.stage2_seg_val_dir or args.seg_val_dir,
            image_size=args.image_size,
            batch_size=args.seg_batch_size,
            num_workers=args.num_workers,
        )
        _, cnt_stage2_val_ds, _, cnt_stage2_val_loader = build_cnt_loaders(
            data_root=args.cnt_data_root,
            train_dir=args.cnt_train_dir,
            val_dir=args.stage2_cnt_val_dir or args.cnt_val_dir,
            image_size=args.image_size,
            num_classes=args.cnt_num_classes,
            keep_aspect=bool(args.cnt_keep_aspect),
            batch_size=args.cnt_batch_size,
            num_workers=1,
        )
    else:
        det_stage2_val_ds, det_stage2_val_loader = det_val_ds, det_val_loader
        seg_stage2_val_ds, seg_stage2_val_loader = seg_val_ds, seg_val_loader
        cnt_stage2_val_ds, cnt_stage2_val_loader = cnt_val_ds, cnt_val_loader

    if use_ddp:
        # DDP 下用 DistributedSampler 重建 loader。
        det_train_loader = _rebuild_ddp_loader(
            det_train_ds,
            batch_size=args.det_batch_size,
            sampler=DistributedSampler(det_train_ds, num_replicas=world_size, rank=rank, shuffle=True),
            num_workers=args.num_workers,
            collate=collate_fn,
        )
        det_val_loader = _rebuild_ddp_loader(
            det_val_ds,
            batch_size=args.det_batch_size,
            sampler=DistributedSampler(det_val_ds, num_replicas=world_size, rank=rank, shuffle=False),
            num_workers=args.num_workers,
            collate=collate_fn,
        )
        seg_train_loader = _rebuild_ddp_loader(
            seg_train_ds,
            batch_size=args.seg_batch_size,
            sampler=DistributedSampler(seg_train_ds, num_replicas=world_size, rank=rank, shuffle=True),
            num_workers=args.num_workers,
        )
        seg_val_loader = _rebuild_ddp_loader(
            seg_val_ds,
            batch_size=args.seg_batch_size,
            sampler=DistributedSampler(seg_val_ds, num_replicas=world_size, rank=rank, shuffle=False),
            num_workers=args.num_workers,
        )
        cnt_train_loader = _rebuild_ddp_loader(
            cnt_train_ds,
            batch_size=args.cnt_batch_size,
            sampler=DistributedSampler(cnt_train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True),
            num_workers=1,
            drop_last=True,
        )
        cnt_val_loader = _rebuild_ddp_loader(
            cnt_val_ds,
            batch_size=args.cnt_batch_size,
            sampler=DistributedSampler(cnt_val_ds, num_replicas=world_size, rank=rank, shuffle=False),
            num_workers=1,
        )
        if warmup_train_ds is not None and warmup_train_loaders is not None:
            warmup_train_loaders["det"] = _rebuild_ddp_loader(
                warmup_train_ds["det"],
                batch_size=args.det_batch_size,
                sampler=DistributedSampler(warmup_train_ds["det"], num_replicas=world_size, rank=rank, shuffle=True),
                num_workers=args.num_workers,
                collate=collate_fn,
            )
            warmup_train_loaders["seg"] = _rebuild_ddp_loader(
                warmup_train_ds["seg"],
                batch_size=args.seg_batch_size,
                sampler=DistributedSampler(warmup_train_ds["seg"], num_replicas=world_size, rank=rank, shuffle=True),
                num_workers=args.num_workers,
            )
            warmup_train_loaders["cnt"] = _rebuild_ddp_loader(
                warmup_train_ds["cnt"],
                batch_size=args.cnt_batch_size,
                sampler=DistributedSampler(warmup_train_ds["cnt"], num_replicas=world_size, rank=rank, shuffle=True, drop_last=True),
                num_workers=1,
                drop_last=True,
            )
        if use_stage2_custom_val:
            det_stage2_val_loader = _rebuild_ddp_loader(
                det_stage2_val_ds,
                batch_size=args.det_batch_size,
                sampler=DistributedSampler(det_stage2_val_ds, num_replicas=world_size, rank=rank, shuffle=False),
                num_workers=args.num_workers,
                collate=collate_fn,
            )
            seg_stage2_val_loader = _rebuild_ddp_loader(
                seg_stage2_val_ds,
                batch_size=args.seg_batch_size,
                sampler=DistributedSampler(seg_stage2_val_ds, num_replicas=world_size, rank=rank, shuffle=False),
                num_workers=args.num_workers,
            )
            cnt_stage2_val_loader = _rebuild_ddp_loader(
                cnt_stage2_val_ds,
                batch_size=args.cnt_batch_size,
                sampler=DistributedSampler(cnt_stage2_val_ds, num_replicas=world_size, rank=rank, shuffle=False),
                num_workers=1,
            )
        else:
            det_stage2_val_loader = det_val_loader
            seg_stage2_val_loader = seg_val_loader
            cnt_stage2_val_loader = cnt_val_loader

    # 沿用旧项目的学习率和权重衰减分组策略。
    det_num_classes = int(args.det_num_classes) if args.det_num_classes else int(det_train_ds.num_classes)
    backbone_lr = float(args.backbone_lr) if args.backbone_lr is not None else float(args.lr) * float(args.backbone_lr_mult)
    det_lr = float(args.det_lr) if args.det_lr is not None else float(args.lr)
    seg_lr = float(args.seg_lr) if args.seg_lr is not None else float(args.lr)
    cnt_lr = float(args.cnt_lr) if args.cnt_lr is not None else float(args.lr)
    backbone_wd = float(args.backbone_weight_decay) if args.backbone_weight_decay is not None else float(args.weight_decay)
    det_wd = float(args.det_weight_decay) if args.det_weight_decay is not None else float(args.weight_decay)
    seg_wd = float(args.seg_weight_decay) if args.seg_weight_decay is not None else float(args.weight_decay)
    cnt_wd = float(args.cnt_weight_decay) if args.cnt_weight_decay is not None else float(args.weight_decay)

    # 主模型仍是 DINOv3 + LoRA-MoE + 三任务头。
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
    raw_model = MultiTaskModel(
        shared=shared,
        det_num_classes=det_num_classes,
        seg_num_classes=args.seg_num_classes,
        cnt_num_classes=args.cnt_num_classes,
        image_size=args.image_size,
        det_train_backbone=bool(args.unfreeze_backbone),
        seg_train_backbone=bool(args.unfreeze_backbone),
        cnt_train_backbone=bool(args.unfreeze_backbone),
    ).to(device)
    if use_ddp:
        # Multi-GPU runs need synchronized BN statistics for seg/cnt heads.
        raw_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(raw_model)
        raw_model = raw_model.to(device)
    # 生成器单独建模，但 Stage1/Stage2 共用同一主模型实例。
    warmup_loaded = False
    if warmup_load_path is not None and warmup_load_path.is_file():
        if is_main_process:
            load_multitask_checkpoint(str(warmup_load_path), model=raw_model, map_location=device)
            print(f"[ckpt] loaded warmup -> {warmup_load_path}")
        warmup_loaded = True
    elif warmup_load_path is not None and is_main_process:
        print(f"[warmup] checkpoint not found, run warmup from scratch: {warmup_load_path}")
    if use_ddp:
        if warmup_loaded:
            dist.barrier()
        _broadcast_module_state(raw_model)
    model_for_state = raw_model
    if use_ddp:
        ddp_kwargs = {
            "broadcast_buffers": False,
            "find_unused_parameters": True,
        }
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [local_rank]
            ddp_kwargs["output_device"] = local_rank
        model = DDP(model_for_state, **ddp_kwargs)
    else:
        model = model_for_state
    state_tracker = SharedExpertStateTracker(
        model_for_state,
        matrix_proj_dim=int(args.state_matrix_proj_dim),
        task_state_dim=int(args.state_task_dim),
    ).to(device)
    base_loss_weights = parse_loss_weights(args.policy_weight_prior)
    policy = DirichletWeightGenerator(
        state_dim=int(state_tracker.feature_dim),
        hidden_dim=int(args.generator_hidden_dim),
        prior_weights=base_loss_weights,
    ).to(device)
    if use_ddp:
        _broadcast_module_state(state_tracker)
        _broadcast_module_state(policy)

    manual_theta_grad_sync = use_ddp

    # 下面按“backbone / lora / 三个任务头”拆参数组。
    shared_backbone_params = list(model_for_state.shared.backbone.parameters())
    shared_backbone_ids = {id(p) for p in shared_backbone_params}
    backbone_params = [p for p in shared_backbone_params if p.requires_grad]
    lora_moe_params = []
    if bool(args.use_lora_moe):
        for lora_moe in model_for_state.shared.lora_moes:
            lora_moe_params.extend([p for p in lora_moe.parameters() if p.requires_grad])
    lora_moe_ids = {id(p) for p in lora_moe_params}
    det_head_params = [
        p for p in model_for_state.detector.parameters() if p.requires_grad and id(p) not in shared_backbone_ids and id(p) not in lora_moe_ids
    ]
    seg_params = [p for p in model_for_state.seg_head.parameters() if p.requires_grad]
    cnt_head_params = [p for p in model_for_state.cnt_head.parameters() if p.requires_grad]

    theta_param_groups = []
    if backbone_params:
        theta_param_groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": backbone_wd})
    if lora_moe_params:
        theta_param_groups.append({"params": lora_moe_params, "lr": float(args.lr), "weight_decay": float(args.weight_decay)})
    if det_head_params:
        theta_param_groups.append({"params": det_head_params, "lr": det_lr, "weight_decay": det_wd})
    if seg_params:
        theta_param_groups.append({"params": seg_params, "lr": seg_lr, "weight_decay": seg_wd})
    if cnt_head_params:
        theta_param_groups.append({"params": cnt_head_params, "lr": cnt_lr, "weight_decay": cnt_wd})
    if not theta_param_groups:
        raise RuntimeError("No trainable parameters.")

    # 打印各部分参数量，便于和 113_test 对齐。
    print(f"[train] Backbone params: {sum(p.numel() for p in backbone_params)}")
    print(f"[train] LoRA-MoE params: {sum(p.numel() for p in lora_moe_params)}")
    print(f"[train] Det head params: {sum(p.numel() for p in det_head_params)}")
    print(f"[train] Seg head params: {sum(p.numel() for p in seg_params)}")
    print(f"[train] Cnt head params: {sum(p.numel() for p in cnt_head_params)}")
    metric_logger.update_config(
        {
            "world_size": int(world_size),
            "use_ddp": int(use_ddp),
            "backbone_param_count": int(sum(p.numel() for p in backbone_params)),
            "lora_moe_param_count": int(sum(p.numel() for p in lora_moe_params)),
            "det_head_param_count": int(sum(p.numel() for p in det_head_params)),
            "seg_head_param_count": int(sum(p.numel() for p in seg_params)),
            "cnt_head_param_count": int(sum(p.numel() for p in cnt_head_params)),
            "fixed_loss_weights": tuple(float(x) for x in base_loss_weights),
            "loss_weight_scope": "shared_experts_only",
            "shared_expert_dynamic_param_count": int(
                sum(
                    (lora_moe.lora_A_shared.numel() if lora_moe.lora_A_shared.requires_grad else 0)
                    + (lora_moe.lora_B_shared.numel() if lora_moe.lora_B_shared.requires_grad else 0)
                    for lora_moe in model_for_state.shared.lora_moes
                )
            ),
        }
    )

    # theta_params 是主模型所有可训练参数；policy 单独优化。
    theta_named_params = [(name, p) for name, p in model_for_state.named_parameters() if p.requires_grad]
    theta_param_names = [name for name, _ in theta_named_params]
    theta_params = [param for _, param in theta_named_params]
    shared_param_mask = tuple(
        name.startswith("shared.lora_moes.") and (name.endswith("lora_A_shared") or name.endswith("lora_B_shared"))
        for name in theta_param_names
    )

    # 主任务用于决定哪个 loader 驱动一个 epoch。
    train_loaders = {"det": det_train_loader, "seg": seg_train_loader, "cnt": cnt_train_loader}
    stage1_val_loaders = {"det": det_val_loader, "seg": seg_val_loader, "cnt": cnt_val_loader}
    stage2_val_loaders = {
        "det": det_stage2_val_loader,
        "seg": seg_stage2_val_loader,
        "cnt": cnt_stage2_val_loader,
    }
    primary_task = choose_primary(
        {"det": len(det_train_ds), "seg": len(seg_train_ds), "cnt": len(cnt_train_ds)},
        args.primary_task,
    )
    # Warmup 直接使用 train_10per/train；验证选模固定使用正式 val。
    warmup_train_loaders = train_loaders
    warmup_primary_task = primary_task
    if warmup_loaded and is_main_process:
        print(f"[warmup] skipped, reuse checkpoint: {warmup_load_path}")
    if (not warmup_loaded) and int(args.warmup_epochs) > 0 and warmup_train_loaders is not None and warmup_primary_task is not None:
        optimizer_theta_warmup = torch.optim.AdamW(theta_param_groups)
        warmup_artifacts = run_warmup(
            args=args,
            model=model_for_state,
            model_for_state=model_for_state,
            optimizer_theta=optimizer_theta_warmup,
            theta_params=theta_params,
            train_loaders=warmup_train_loaders,
            val_loaders=stage2_val_loaders,
            primary_task=warmup_primary_task,
            device=device,
            use_ddp=use_ddp,
            world_size=world_size,
            manual_theta_grad_sync=manual_theta_grad_sync,
            det_num_classes=det_num_classes,
            is_main_process=is_main_process,
            metric_logger=metric_logger,
        )
        if use_ddp:
            _broadcast_module_state(model_for_state)
        if is_main_process:
            save_multitask_checkpoint(
                str(warmup_save_path),
                model=model_for_state,
                optimizer=optimizer_theta_warmup,
                epoch=int(warmup_artifacts.best_epoch),
                best_by="warmup",
                metrics={
                    "warmup_epochs": float(int(args.warmup_epochs)),
                    "warmup_best_metric": float(warmup_artifacts.best_metric),
                    "warmup_best_epoch": float(int(warmup_artifacts.best_epoch)),
                },
                loss_weights=parse_loss_weights(args.warmup_loss_weights),
                config={
                    "warmup_epochs": int(args.warmup_epochs),
                    "warmup_loss_weights": str(args.warmup_loss_weights),
                    "warmup_selected_by": warmup_artifacts.selected_by,
                    "warmup_det_data_root": args.det_data_root,
                    "warmup_det_train_ann": args.det_train_ann,
                    "warmup_det_train_img_dir": args.det_train_img_dir,
                    "warmup_seg_train_dir": args.seg_train_dir,
                    "warmup_cnt_data_root": args.cnt_data_root,
                    "warmup_cnt_train_dir": args.cnt_train_dir,
                    "warmup_det_val_ann": args.stage2_det_val_ann or args.det_val_ann,
                    "warmup_seg_val_dir": args.stage2_seg_val_dir or args.seg_val_dir,
                    "warmup_cnt_val_dir": args.stage2_cnt_val_dir or args.cnt_val_dir,
                },
            )
            print(f"[ckpt] saved warmup -> {warmup_save_path}")
        del optimizer_theta_warmup
        if device.type == "cuda":
            torch.cuda.empty_cache()

    stage1_val_loaders = {"det": det_val_loader, "seg": seg_val_loader, "cnt": cnt_val_loader}
    stage2_val_loaders = {
        "det": det_stage2_val_loader,
        "seg": seg_stage2_val_loader,
        "cnt": cnt_stage2_val_loader,
    }
    primary_task = choose_primary(
        {"det": len(det_train_ds), "seg": len(seg_train_ds), "cnt": len(cnt_train_ds)},
        args.primary_task,
    )

    # 状态跟踪器负责维护共享专家梯度统计的 EMA。

    # Stage1：每个 step 同时更新生成器和主模型。
    optimizer_theta_s1 = torch.optim.AdamW(theta_param_groups)
    optimizer_policy = torch.optim.AdamW(
        list(policy.parameters()) + list(state_tracker.parameters()),
        lr=float(args.meta_beta),
        weight_decay=0.0,
    )

    stage1_artifacts = run_stage1(
        args=args,
        model=model,
        model_for_state=model_for_state,
        policy=policy,
        state_tracker=state_tracker,
        optimizer_theta=optimizer_theta_s1,
        optimizer_policy=optimizer_policy,
        theta_params=theta_params,
        theta_param_names=theta_param_names,
        base_loss_weights=base_loss_weights,
        shared_param_mask=shared_param_mask,
        train_loaders=train_loaders,
        val_loaders=stage1_val_loaders,
        primary_task=primary_task,
        device=device,
        use_ddp=use_ddp,
        world_size=world_size,
        manual_theta_grad_sync=manual_theta_grad_sync,
        metric_logger=metric_logger,
    )

    if is_main_process:
        # 单独保存 Stage1 末尾的生成器和 EMA 状态。
        torch.save(
            {
                "epoch": int(args.stage1_epochs),
                "generator_state": stage1_artifacts.generator_state,
                "state_feature_state": stage1_artifacts.state_feature_state,
                "loss_weights": stage1_artifacts.last_weights,
                "fixed_loss_weights": tuple(float(x) for x in base_loss_weights),
                "loss_weight_scope": "shared_experts_only",
            },
            save_dir / "stage1_generator_last.pt",
        )
        print(f"[ckpt] saved stage1 generator -> {save_dir / 'stage1_generator_last.pt'}")

    del optimizer_theta_s1
    del optimizer_policy
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # Stage2：从 Stage1 结束的主模型继续训练，并按旧逻辑选模。
    optimizer_theta_s2 = torch.optim.AdamW(theta_param_groups)
    run_stage2(
        args=args,
        model=model,
        model_for_state=model_for_state,
        policy=policy,
        state_tracker=state_tracker,
        optimizer_theta=optimizer_theta_s2,
        theta_params=theta_params,
        base_loss_weights=base_loss_weights,
        shared_param_mask=shared_param_mask,
        train_loaders=train_loaders,
        val_loaders=stage2_val_loaders,
        primary_task=primary_task,
        device=device,
        use_ddp=use_ddp,
        world_size=world_size,
        manual_theta_grad_sync=manual_theta_grad_sync,
        det_num_classes=det_num_classes,
        save_dir=save_dir,
        is_main_process=is_main_process,
        metric_logger=metric_logger,
    )

    metric_logger.finish()
    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

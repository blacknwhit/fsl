from __future__ import annotations

import argparse
import builtins
import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from object_detection.dataset import collate_fn

from .block_selection_logger import BlockSelectionRecorder
from .datasets import build_cnt_loaders, build_det_loaders, build_seg_loaders
from .models import MultiTaskModel, SharedDinoV3Backbone
from .swanlab_logger import MetricLogger, create_metric_logger
from .train_utils import build_shared_expert_matrix_index
from .trainer_main import run_stage1_plain, run_stage2_matrix_pair
from .utils import choose_primary, load_multitask_checkpoint, parse_loss_weights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stage multitask training with DINOv3 + optional plain LoRA + LoRA-MoE")

    p.add_argument("--model-name", type=str, default="dinov3_vitl16")
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--backbone-checkpoint", type=str, default=None)
    p.add_argument("--unfreeze-backbone", action="store_true")

    p.add_argument("--lora", action="store_true")
    p.add_argument("--use-lora-moe", action="store_true")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--lora-alpha", type=float, default=16.0)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-lr", type=float, default=None)
    p.add_argument("--lora-weight-decay", type=float, default=0.0)
    p.add_argument("--num-experts-private", type=int, default=3)
    p.add_argument("--num-experts-shared", type=int, default=6)
    p.add_argument("--moe-k-private", type=int, default=2)
    p.add_argument("--moe-k-shared", type=int, default=2)
    private_pool = p.add_mutually_exclusive_group()
    private_pool.add_argument("--moe-private-experts", dest="use_private_experts", action="store_true")
    private_pool.add_argument("--no-moe-private-experts", dest="use_private_experts", action="store_false")
    p.set_defaults(use_private_experts=True)
    shared_pool = p.add_mutually_exclusive_group()
    shared_pool.add_argument("--moe-shared-experts", dest="use_shared_experts", action="store_true")
    shared_pool.add_argument("--no-moe-shared-experts", dest="use_shared_experts", action="store_false")
    p.set_defaults(use_shared_experts=True)

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
    p.add_argument("--det-out-channels", type=int, default=256)
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

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--stage1-epochs", type=int, default=None)
    p.add_argument("--stage2-epochs", type=int, default=50)
    p.add_argument("--stage2-init-checkpoint", type=str, default=None)
    p.add_argument(
        "--stage1-resume-checkpoint",
        type=str,
        default=None,
        help="load a multitask/stage1 checkpoint and continue normal stage1 training",
    )
    p.add_argument(
        "--stage1-only",
        action="store_true",
        help="run stage1 only and exit after saving stage1_best.pt",
    )
    p.add_argument("--stage1-val-last-k-epochs", type=int, default=50)
    p.add_argument("--loss-weights", type=str, default="15,8,1")
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
    p.add_argument("--save-dir", type=str, default="runs/latest_ours")
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--debug-step-timing", action="store_true")
    p.add_argument("--debug-step-timing-interval", type=int, default=1)
    p.add_argument("--grad-clip-norm", type=float, default=100.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-steps", type=int, default=0)
    p.add_argument("--max-val-steps", type=int, default=0)
    p.add_argument("--skip-validation", action="store_true")
    p.add_argument("--cnt-backbone-grad-mult", type=float, default=1.0)
    p.add_argument("--use-swanlab", action="store_true")
    p.add_argument("--swanlab-project", type=str, default=None)
    p.add_argument("--swanlab-workspace", type=str, default=None)
    p.add_argument("--swanlab-experiment-name", type=str, default=None)
    p.add_argument("--swanlab-mode", type=str, default=None)
    p.add_argument("--swanlab-logdir", type=str, default=None)
    return p.parse_args()


def _rebuild_ddp_loader(dataset, *, batch_size: int, sampler, num_workers: int, collate=None, drop_last: bool = False):
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
    if not dist.is_initialized():
        return
    for tensor in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(tensor.data, src=0)


def _print_checkpoint_load_status(ckpt_path: str, load_info: dict) -> None:
    load_complete = bool(load_info.get("_load_complete", False))
    status = "complete" if load_complete else "partial"
    print(f"[ckpt] load {status}: {ckpt_path}")
    load_report = load_info.get("_load_report", {})
    if not isinstance(load_report, dict):
        return
    for module_name in ("shared", "detector", "seg_head", "cnt_head"):
        module_report = load_report.get(module_name, {})
        if not isinstance(module_report, dict):
            continue
        missing_count = int(module_report.get("missing_count", 0))
        unexpected_count = int(module_report.get("unexpected_count", 0))
        module_complete = bool(module_report.get("complete", False))
        print(
            f"[ckpt]   {module_name}: complete={int(module_complete)} "
            f"missing={missing_count} unexpected={unexpected_count}"
        )
        missing_keys = list(module_report.get("missing_keys", []))
        unexpected_keys = list(module_report.get("unexpected_keys", []))
        if missing_keys:
            print(f"[ckpt]     missing_keys(sample): {missing_keys[:5]}")
        if unexpected_keys:
            print(f"[ckpt]     unexpected_keys(sample): {unexpected_keys[:5]}")


def _is_plain_lora_param_name(name: str) -> bool:
    lowered = str(name).lower()
    return ".lora_a" in lowered or ".lora_b" in lowered


def _torch_load_cpu(path: str) -> dict:
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid checkpoint format: {path}")
    return obj


def _is_multitask_checkpoint(obj: Mapping[str, Any]) -> bool:
    return (
        isinstance(obj.get("backbone"), dict)
        and isinstance(obj.get("det_head"), dict)
        and isinstance(obj.get("seg_head"), dict)
        and isinstance(obj.get("cnt_head"), dict)
    )


def _state_dict_has_lora(state_dict: Mapping[str, Any]) -> bool:
    return any(
        isinstance(k, str) and (".lora_a" in k.lower() or ".lora_b" in k.lower())
        for k in state_dict.keys()
    )


def _infer_fg_num_classes_from_det_state(det_state: Mapping[str, Any]) -> int | None:
    w = det_state.get("roi_heads.box_predictor.cls_score.weight")
    if hasattr(w, "shape") and len(getattr(w, "shape", [])) >= 1:
        total = int(w.shape[0])
        if total >= 2:
            return total - 1
    return None


def _infer_num_classes_from_conv1x1_weight(state: Mapping[str, Any], weight_key: str) -> int | None:
    w = state.get(weight_key)
    if hasattr(w, "shape") and len(getattr(w, "shape", [])) >= 1:
        return int(w.shape[0])
    return None


def _infer_det_out_channels(det_state: Mapping[str, Any], default: int = 256) -> int:
    w = det_state.get("backbone.proj.weight")
    if hasattr(w, "shape") and len(getattr(w, "shape", [])) >= 1:
        return int(w.shape[0])
    return int(default)


def _config_bool(cfg: Mapping[str, Any] | None, key: str, default: bool) -> bool:
    if not isinstance(cfg, dict) or key not in cfg:
        return bool(default)
    value = cfg.get(key)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _config_int(cfg: Mapping[str, Any] | None, key: str, default: int) -> int:
    if not isinstance(cfg, dict):
        return int(default)
    try:
        value = int(cfg.get(key, default))
    except (TypeError, ValueError):
        return int(default)
    return value if value >= 1 else int(default)


def _config_float(cfg: Mapping[str, Any] | None, key: str, default: float) -> float:
    if not isinstance(cfg, dict):
        return float(default)
    try:
        return float(cfg.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _first_tensor(state: Mapping[str, Any], suffix: str):
    preferred = (
        f"lora_moes.0.{suffix}",
        f"wrapped_blocks.0.lora_moe.{suffix}",
    )
    for key in preferred:
        value = state.get(key)
        if value is not None:
            return value
    for key, value in state.items():
        if isinstance(key, str) and key.endswith(f".{suffix}"):
            return value
    return None


def _infer_checkpoint_architecture(path: str) -> dict[str, Any]:
    ckpt = _torch_load_cpu(path)
    if not _is_multitask_checkpoint(ckpt):
        raise ValueError(
            f"--stage1-resume-checkpoint must be a multitask checkpoint with backbone/det_head/seg_head/cnt_head: {path}"
        )

    shared_state = ckpt["backbone"]
    det_state = ckpt["det_head"]
    seg_state = ckpt["seg_head"]
    cnt_state = ckpt["cnt_head"]
    cfg = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
    lora_meta = ckpt.get("lora") if isinstance(ckpt.get("lora"), dict) else {}
    assert isinstance(shared_state, dict) and isinstance(det_state, dict)
    assert isinstance(seg_state, dict) and isinstance(cnt_state, dict)

    has_lora_moes = any(isinstance(k, str) and k.startswith("lora_moes.") for k in shared_state.keys())
    has_wrapped = any(isinstance(k, str) and k.startswith("wrapped_blocks.") for k in shared_state.keys())
    has_private = any(isinstance(k, str) and k.endswith(".lora_A_private") for k in shared_state.keys())
    has_shared = any(isinstance(k, str) and k.endswith(".lora_A_shared") for k in shared_state.keys())
    use_lora_moe = bool(has_lora_moes or has_wrapped or cfg.get("use_lora_moe", False))
    use_lora = bool(cfg.get("use_lora", False)) or bool(lora_meta) or _state_dict_has_lora(shared_state)

    lora_rank = _config_int(cfg, "lora_rank", int(lora_meta.get("rank", 8) if isinstance(lora_meta, dict) else 8))
    num_experts_private = _config_int(cfg, "num_experts_private", 3)
    num_experts_shared = _config_int(cfg, "num_experts_shared", 6)
    task_num = _config_int(cfg, "task_num", 3)

    A_private = _first_tensor(shared_state, "lora_A_private")
    if hasattr(A_private, "shape") and len(getattr(A_private, "shape", [])) == 4:
        task_num = int(A_private.shape[0])
        num_experts_private = int(A_private.shape[1])
        lora_rank = int(A_private.shape[3])

    A_shared = _first_tensor(shared_state, "lora_A_shared")
    if hasattr(A_shared, "shape") and len(getattr(A_shared, "shape", [])) == 3:
        num_experts_shared = int(A_shared.shape[0])
        lora_rank = int(A_shared.shape[2])

    gate_indices: list[int] = []
    for key in shared_state.keys():
        if not isinstance(key, str) or not key.endswith(".weight"):
            continue
        parts = key.split(".")
        for gate_name in ("f_gate_private", "f_gate_shared"):
            if gate_name in parts:
                pos = parts.index(gate_name) + 1
                if pos < len(parts):
                    try:
                        gate_indices.append(int(parts[pos]))
                    except ValueError:
                        pass
                break
    if gate_indices:
        task_num = max(gate_indices) + 1

    return {
        "path": str(path),
        "use_lora": bool(use_lora),
        "use_lora_moe": bool(use_lora_moe),
        "lora_rank": int(lora_rank),
        "lora_alpha": _config_float(cfg, "lora_alpha", float(lora_meta.get("alpha", 16.0) if isinstance(lora_meta, dict) else 16.0)),
        "lora_dropout": _config_float(cfg, "lora_dropout", float(lora_meta.get("dropout", 0.05) if isinstance(lora_meta, dict) else 0.05)),
        "num_experts_private": int(num_experts_private),
        "num_experts_shared": int(num_experts_shared),
        "moe_k_private": _config_int(cfg, "moe_k_private", 2),
        "moe_k_shared": _config_int(cfg, "moe_k_shared", 2),
        "use_private_experts": _config_bool(cfg, "use_private_experts", has_private),
        "use_shared_experts": _config_bool(cfg, "use_shared_experts", has_shared),
        "task_num": int(task_num),
        "det_num_classes": _infer_fg_num_classes_from_det_state(det_state),
        "seg_num_classes": _infer_num_classes_from_conv1x1_weight(seg_state, "decode.3.weight"),
        "cnt_num_classes": _infer_num_classes_from_conv1x1_weight(cnt_state, "decode.3.weight"),
        "det_out_channels": _infer_det_out_channels(det_state, default=256),
    }


def _apply_checkpoint_architecture(args: argparse.Namespace, arch: dict[str, Any]) -> None:
    args.lora = bool(arch["use_lora"])
    args.use_lora_moe = bool(arch["use_lora_moe"])
    args.lora_rank = int(arch["lora_rank"])
    args.lora_alpha = float(arch["lora_alpha"])
    args.lora_dropout = float(arch["lora_dropout"])
    args.num_experts_private = int(arch["num_experts_private"])
    args.num_experts_shared = int(arch["num_experts_shared"])
    args.moe_k_private = int(arch["moe_k_private"])
    args.moe_k_shared = int(arch["moe_k_shared"])
    args.use_private_experts = bool(arch["use_private_experts"])
    args.use_shared_experts = bool(arch["use_shared_experts"])
    if arch.get("det_num_classes") is not None:
        args.det_num_classes = int(arch["det_num_classes"])
    if arch.get("seg_num_classes") is not None:
        args.seg_num_classes = int(arch["seg_num_classes"])
    if arch.get("cnt_num_classes") is not None:
        args.cnt_num_classes = int(arch["cnt_num_classes"])
    args.det_out_channels = int(arch["det_out_channels"])


def main() -> None:
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

    if args.stage1_epochs is None:
        args.stage1_epochs = int(args.epochs) if args.epochs is not None else 100
    resume_stage1 = bool(args.stage1_resume_checkpoint)
    skip_stage1 = bool(args.stage2_init_checkpoint)
    if resume_stage1 and skip_stage1:
        raise ValueError("--stage1-resume-checkpoint cannot be used together with --stage2-init-checkpoint")
    if resume_stage1:
        args.stage1_val_last_k_epochs = int(args.stage1_epochs)
    arch_checkpoint_path = args.stage1_resume_checkpoint or args.stage2_init_checkpoint
    checkpoint_arch = None
    if arch_checkpoint_path:
        checkpoint_arch = _infer_checkpoint_architecture(arch_checkpoint_path)
        _apply_checkpoint_architecture(args, checkpoint_arch)
    if skip_stage1:
        args.stage1_epochs = 0
    args.stage1_epochs = int(args.stage1_epochs)
    args.stage2_epochs = int(args.stage2_epochs)
    args.stage1_val_last_k_epochs = int(args.stage1_val_last_k_epochs)

    if float(args.grad_clip_norm) < 0:
        raise ValueError("--grad-clip-norm must be >= 0")
    if skip_stage1:
        if int(args.stage1_epochs) < 0:
            raise ValueError("--stage1-epochs must be >= 0 when --stage2-init-checkpoint is set")
    elif int(args.stage1_epochs) < 1:
        raise ValueError("--stage1-epochs must be >= 1")
    if int(args.stage2_epochs) < 1:
        raise ValueError("--stage2-epochs must be >= 1")
    if int(args.stage1_val_last_k_epochs) < 1:
        raise ValueError("--stage1-val-last-k-epochs must be >= 1")
    if not bool(args.use_lora_moe) and not bool(args.stage1_only):
        raise ValueError("Two-stage training requires --use-lora-moe for shared expert filtering.")
    if bool(args.use_lora_moe) and not bool(args.use_private_experts) and not bool(args.use_shared_experts):
        raise ValueError("--use-lora-moe requires at least one of private/shared expert pools")
    if checkpoint_arch is not None and is_main_process:
        print(
            "[ckpt-arch] "
            f"path={checkpoint_arch['path']} use_lora={int(args.lora)} use_lora_moe={int(args.use_lora_moe)} "
            f"private={int(args.use_private_experts)} shared={int(args.use_shared_experts)} "
            f"rank={int(args.lora_rank)} experts_private={int(args.num_experts_private)} experts_shared={int(args.num_experts_shared)} "
            f"det_classes={args.det_num_classes} seg_classes={args.seg_num_classes} cnt_classes={args.cnt_num_classes} "
            f"det_out_channels={args.det_out_channels}"
        )

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

    use_stage2_custom_val = any(
        [
            args.stage2_det_val_ann,
            args.stage2_det_val_img_dir,
            args.stage2_seg_val_dir,
            args.stage2_cnt_val_dir,
        ]
    )
    if use_stage2_custom_val:
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

    det_num_classes = int(args.det_num_classes) if args.det_num_classes else int(det_train_ds.num_classes)
    backbone_lr = float(args.backbone_lr) if args.backbone_lr is not None else float(args.lr) * float(args.backbone_lr_mult)
    lora_lr = float(args.lora_lr) if args.lora_lr is not None else float(args.lr)
    det_lr = float(args.det_lr) if args.det_lr is not None else float(args.lr)
    seg_lr = float(args.seg_lr) if args.seg_lr is not None else float(args.lr)
    cnt_lr = float(args.cnt_lr) if args.cnt_lr is not None else float(args.lr)
    backbone_wd = float(args.backbone_weight_decay) if args.backbone_weight_decay is not None else float(args.weight_decay)
    lora_wd = float(args.lora_weight_decay)
    det_wd = float(args.det_weight_decay) if args.det_weight_decay is not None else float(args.weight_decay)
    seg_wd = float(args.seg_weight_decay) if args.seg_weight_decay is not None else float(args.weight_decay)
    cnt_wd = float(args.cnt_weight_decay) if args.cnt_weight_decay is not None else float(args.weight_decay)

    shared = SharedDinoV3Backbone(
        model_name=args.model_name,
        image_size=args.image_size,
        checkpoint_path=args.backbone_checkpoint,
        use_lora=bool(args.lora),
        use_lora_moe=bool(args.use_lora_moe),
        backbone_trainable=bool(args.unfreeze_backbone),
        task_num=int(checkpoint_arch["task_num"]) if checkpoint_arch is not None else 3,
        lora_rank=int(args.lora_rank),
        lora_alpha=float(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        num_experts_private=int(args.num_experts_private),
        num_experts_shared=int(args.num_experts_shared),
        moe_k_private=int(args.moe_k_private),
        moe_k_shared=int(args.moe_k_shared),
        use_private_experts=bool(args.use_private_experts),
        use_shared_experts=bool(args.use_shared_experts),
        grad_checkpointing=bool(args.grad_checkpointing),
    )
    raw_model = MultiTaskModel(
        shared=shared,
        det_num_classes=det_num_classes,
        seg_num_classes=args.seg_num_classes,
        cnt_num_classes=args.cnt_num_classes,
        image_size=args.image_size,
        det_out_channels=int(args.det_out_channels),
        det_train_backbone=bool(args.unfreeze_backbone),
        seg_train_backbone=bool(args.unfreeze_backbone),
        cnt_train_backbone=bool(args.unfreeze_backbone),
    ).to(device)
    if use_ddp:
        raw_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(raw_model)
        raw_model = raw_model.to(device)
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

    base_loss_weights = parse_loss_weights(args.loss_weights)

    shared_backbone_named_params = list(model_for_state.shared.backbone.named_parameters())
    shared_backbone_params = [param for _, param in shared_backbone_named_params]
    shared_backbone_ids = {id(param) for param in shared_backbone_params}
    plain_lora_params = [
        param for name, param in shared_backbone_named_params if param.requires_grad and _is_plain_lora_param_name(name)
    ]
    plain_lora_ids = {id(param) for param in plain_lora_params}
    backbone_params = [
        param for _, param in shared_backbone_named_params if param.requires_grad and id(param) not in plain_lora_ids
    ]
    lora_moe_params = []
    for lora_moe in model_for_state.shared.lora_moes:
        lora_moe_params.extend([p for p in lora_moe.parameters() if p.requires_grad])
    lora_moe_ids = {id(p) for p in lora_moe_params}
    det_head_params = [
        p
        for p in model_for_state.detector.parameters()
        if p.requires_grad and id(p) not in shared_backbone_ids and id(p) not in lora_moe_ids
    ]
    seg_params = [p for p in model_for_state.seg_head.parameters() if p.requires_grad]
    cnt_head_params = [p for p in model_for_state.cnt_head.parameters() if p.requires_grad]

    theta_param_groups = []
    if backbone_params:
        theta_param_groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": backbone_wd})
    if plain_lora_params:
        theta_param_groups.append({"params": plain_lora_params, "lr": lora_lr, "weight_decay": lora_wd})
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

    print(f"[train] Backbone params: {sum(p.numel() for p in backbone_params)}")
    print(f"[train] Plain LoRA params: {sum(p.numel() for p in plain_lora_params)}")
    print(f"[train] LoRA-MoE params: {sum(p.numel() for p in lora_moe_params)}")
    print(f"[train] Det head params: {sum(p.numel() for p in det_head_params)}")
    print(f"[train] Seg head params: {sum(p.numel() for p in seg_params)}")
    print(f"[train] Cnt head params: {sum(p.numel() for p in cnt_head_params)}")

    theta_named_params = [(name, p) for name, p in model_for_state.named_parameters() if p.requires_grad]
    theta_param_names = [name for name, _ in theta_named_params]
    theta_params = [param for _, param in theta_named_params]
    shared_matrix_index = build_shared_expert_matrix_index(theta_param_names, theta_params)
    has_shared_expert_matrices = bool(shared_matrix_index.matrix_units)
    if not has_shared_expert_matrices and bool(args.use_shared_experts):
        raise RuntimeError(
            "No shared expert matrices were detected from trainable parameter names. "
            "Check LoRA-MoE parameter naming and shared matrix index construction."
        )
    if not has_shared_expert_matrices and is_main_process:
        print("[train] Shared LoRA-MoE expert pool disabled; stage2 will use weighted task gradients for all params.")

    block_selection_recorder = BlockSelectionRecorder(
        save_dir=save_dir,
        enabled=has_shared_expert_matrices,
        is_main_process=is_main_process,
    )
    metric_logger.update_config(
        {
            "world_size": int(world_size),
            "use_ddp": int(use_ddp),
            "training_mode": "two_stage",
            "stage1_resume_checkpoint": str(args.stage1_resume_checkpoint) if args.stage1_resume_checkpoint else None,
            "stage1_only": int(bool(args.stage1_only)),
            "use_lora": int(bool(args.lora)),
            "use_lora_moe": int(bool(args.use_lora_moe)),
            "use_private_experts": int(bool(args.use_private_experts)),
            "use_shared_experts": int(bool(args.use_shared_experts)),
            "skip_stage1": int(skip_stage1),
            "stage1_epochs": int(args.stage1_epochs),
            "stage2_epochs": int(args.stage2_epochs),
            "stage2_init_checkpoint": str(args.stage2_init_checkpoint) if args.stage2_init_checkpoint else None,
            "loss_weights": tuple(float(x) for x in base_loss_weights),
            "stage1_val_last_k_epochs": int(args.stage1_val_last_k_epochs),
            "shared_update_rule": "local_top_pair_per_matrix_unit" if has_shared_expert_matrices else "none_weighted_all_params",
            "backbone_param_count": int(sum(p.numel() for p in backbone_params)),
            "plain_lora_param_count": int(sum(p.numel() for p in plain_lora_params)),
            "lora_moe_param_count": int(sum(p.numel() for p in lora_moe_params)),
            "det_head_param_count": int(sum(p.numel() for p in det_head_params)),
            "seg_head_param_count": int(sum(p.numel() for p in seg_params)),
            "cnt_head_param_count": int(sum(p.numel() for p in cnt_head_params)),
            "shared_expert_block_count": int(len(shared_matrix_index.block_ids)),
            "shared_expert_matrix_unit_count": int(len(shared_matrix_index.matrix_units)),
            "shared_expert_dynamic_param_count": int(
                sum(
                    (
                        lora_moe.lora_A_shared.numel()
                        if getattr(lora_moe, "lora_A_shared", None) is not None
                        and lora_moe.lora_A_shared.requires_grad
                        else 0
                    )
                    + (
                        lora_moe.lora_B_shared.numel()
                        if getattr(lora_moe, "lora_B_shared", None) is not None
                        and lora_moe.lora_B_shared.requires_grad
                        else 0
                    )
                    for lora_moe in model_for_state.shared.lora_moes
                )
            ),
            "block_selection_history_path": str(block_selection_recorder.history_path),
            "block_selection_epoch_summary_path": str(block_selection_recorder.epoch_summary_path),
        }
    )

    train_loaders = {"det": det_train_loader, "seg": seg_train_loader, "cnt": cnt_train_loader}
    stage1_val_loaders = {
        "det": det_val_loader,
        "seg": seg_val_loader,
        "cnt": cnt_val_loader,
    }
    stage2_val_loaders = {
        "det": det_stage2_val_loader,
        "seg": seg_stage2_val_loader,
        "cnt": cnt_stage2_val_loader,
    }
    primary_task = choose_primary(
        {"det": len(det_train_ds), "seg": len(seg_train_ds), "cnt": len(cnt_train_ds)},
        args.primary_task,
    )

    def build_theta_optimizer() -> torch.optim.Optimizer:
        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": backbone_wd})
        if plain_lora_params:
            param_groups.append({"params": plain_lora_params, "lr": lora_lr, "weight_decay": lora_wd})
        if lora_moe_params:
            param_groups.append({"params": lora_moe_params, "lr": float(args.lr), "weight_decay": float(args.weight_decay)})
        if det_head_params:
            param_groups.append({"params": det_head_params, "lr": det_lr, "weight_decay": det_wd})
        if seg_params:
            param_groups.append({"params": seg_params, "lr": seg_lr, "weight_decay": seg_wd})
        if cnt_head_params:
            param_groups.append({"params": cnt_head_params, "lr": cnt_lr, "weight_decay": cnt_wd})
        return torch.optim.AdamW(param_groups)

    manual_theta_grad_sync = use_ddp

    if use_ddp:
        _broadcast_module_state(model_for_state)

    if resume_stage1:
        if is_main_process:
            print(f"[stage1] resume from checkpoint: {args.stage1_resume_checkpoint}")
            load_info = load_multitask_checkpoint(
                args.stage1_resume_checkpoint,
                model=model_for_state,
                map_location="cpu",
            )
            _print_checkpoint_load_status(args.stage1_resume_checkpoint, load_info)
        if use_ddp:
            dist.barrier()
            _broadcast_module_state(model_for_state)
            dist.barrier()

    if skip_stage1:
        if is_main_process:
            print(f"[stage1] skipped; initialize stage2 from checkpoint: {args.stage2_init_checkpoint}")
            load_info = load_multitask_checkpoint(
                args.stage2_init_checkpoint,
                model=model_for_state,
                map_location="cpu",
            )
            _print_checkpoint_load_status(args.stage2_init_checkpoint, load_info)
        if use_ddp:
            dist.barrier()
            _broadcast_module_state(model_for_state)
            dist.barrier()
    else:
        optimizer_theta_stage1 = build_theta_optimizer()
        stage1_artifacts = run_stage1_plain(
            args=args,
            model=model,
            model_for_state=model_for_state,
            optimizer_theta=optimizer_theta_stage1,
            theta_params=theta_params,
            base_loss_weights=base_loss_weights,
            train_loaders=train_loaders,
            val_loaders=stage1_val_loaders,
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

        if use_ddp:
            dist.barrier()

        if is_main_process:
            load_info = load_multitask_checkpoint(
                stage1_artifacts.checkpoint_path,
                model=model_for_state,
                map_location="cpu",
            )
            _print_checkpoint_load_status(stage1_artifacts.checkpoint_path, load_info)
        if use_ddp:
            _broadcast_module_state(model_for_state)
            dist.barrier()

    if bool(args.stage1_only):
        if is_main_process:
            print("[stage1] stage1-only mode complete; skip stage2")
        metric_logger.finish()
        if use_ddp and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
        return

    optimizer_theta_stage2 = build_theta_optimizer()
    run_stage2_matrix_pair(
        args=args,
        model=model,
        model_for_state=model_for_state,
        optimizer_theta=optimizer_theta_stage2,
        theta_params=theta_params,
        base_loss_weights=base_loss_weights,
        shared_matrix_index=shared_matrix_index,
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
        block_selection_recorder=block_selection_recorder,
    )

    metric_logger.finish()
    if use_ddp and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

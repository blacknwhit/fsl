from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from .datasets import CountTransform, DetTransform, SegTransform, IMAGENET_MEAN, IMAGENET_STD
    from .train_utils import _safe_cosine_similarity, build_shared_expert_matrix_index, freeze_batchnorm_stats
    from .trainer_main import _compute_task_loss_and_grads
    from .utils import load_multitask_checkpoint
except ImportError:
    from datasets import CountTransform, DetTransform, SegTransform, IMAGENET_MEAN, IMAGENET_STD
    from train_utils import _safe_cosine_similarity, build_shared_expert_matrix_index, freeze_batchnorm_stats
    from trainer_main import _compute_task_loss_and_grads
    from utils import load_multitask_checkpoint

from counting.dataset import DSACADensityH5Dataset
from object_detection.dataset import CocoDetectionDataset, collate_fn
from segmentation.dataset import SegmentationDataset


PAIR_ORDER = ("det_seg", "det_cnt", "seg_cnt")
PAIR_COLORS = {
    "det_seg": "#1f77b4",
    "det_cnt": "#ff7f0e",
    "seg_cnt": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize stage2 shared-expert gradient conflict on LoRA A matrices. "
            "One run samples one train batch per task and saves one grouped-bar chart per target block."
        )
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Stage2 multitask checkpoint.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory. Default: <checkpoint_dir>/grad_conflict")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model-name", type=str, default="dinov3_vitl16")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument(
        "--backbone-init-image-size",
        type=int,
        default=64,
        help="Dummy init image size used only to infer embed_dim during backbone construction.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-ids", type=str, default="5,11,17,23", help="0-based block ids, comma-separated.")

    parser.add_argument(
        "--det-data-root",
        type=str,
        default="/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/train_10per",
    )
    parser.add_argument(
        "--det-train-ann",
        type=str,
        default="/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_train_10per.json",
    )
    parser.add_argument(
        "--det-train-img-dir",
        type=str,
        default="/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco",
    )
    parser.add_argument("--det-batch-size", type=int, default=2)

    parser.add_argument(
        "--seg-train-dir",
        type=str,
        default="/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/train_5500_10per",
    )
    parser.add_argument("--seg-batch-size", type=int, default=4)

    parser.add_argument(
        "--cnt-data-root",
        type=str,
        default="/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/train_10per",
    )
    parser.add_argument(
        "--cnt-train-dir",
        type=str,
        default="/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA/train_data_class8_10per",
    )
    parser.add_argument("--cnt-num-classes", type=int, default=None)
    parser.add_argument("--cnt-batch-size", type=int, default=2)
    aspect = parser.add_mutually_exclusive_group()
    aspect.add_argument("--cnt-keep-aspect", dest="cnt_keep_aspect", action="store_true")
    aspect.add_argument("--cnt-no-keep-aspect", dest="cnt_keep_aspect", action="store_false")
    parser.set_defaults(cnt_keep_aspect=True)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cnt-count-loss-weight", type=float, default=1.0)
    parser.add_argument("--cnt-backbone-grad-mult", type=float, default=1.0)
    return parser.parse_args()


def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _is_multitask_checkpoint(obj: object) -> bool:
    return (
        isinstance(obj, dict)
        and isinstance(obj.get("backbone"), dict)
        and isinstance(obj.get("det_head"), dict)
        and isinstance(obj.get("seg_head"), dict)
        and isinstance(obj.get("cnt_head"), dict)
    )


def _infer_fg_num_classes_from_det_state(det_state: Mapping) -> int:
    weight = det_state.get("roi_heads.box_predictor.cls_score.weight")
    if hasattr(weight, "shape") and len(getattr(weight, "shape", [])) >= 1:
        total = int(weight.shape[0])
        if total >= 2:
            return total - 1
    raise ValueError("Cannot infer detection class count from checkpoint det_head.")


def _infer_num_classes_from_conv1x1_weight(state: Mapping, weight_key: str) -> int:
    weight = state.get(weight_key)
    if hasattr(weight, "shape") and len(getattr(weight, "shape", [])) >= 1:
        return int(weight.shape[0])
    raise ValueError(f"Cannot infer num_classes from checkpoint key: {weight_key}")


def _infer_det_out_channels(det_state: Mapping, default: int = 256) -> int:
    weight = det_state.get("backbone.proj.weight")
    if hasattr(weight, "shape") and len(getattr(weight, "shape", [])) >= 1:
        return int(weight.shape[0])
    return int(default)


def _infer_lora_moe_config_from_shared_state(
    shared_state: Mapping,
    ckpt_config: Mapping | None = None,
) -> Dict[str, object]:
    def _config_int(key: str, default: int) -> int:
        if not isinstance(ckpt_config, dict):
            return int(default)
        value = ckpt_config.get(key)
        try:
            value = int(value)
        except (TypeError, ValueError):
            return int(default)
        return int(value) if int(value) >= 1 else int(default)

    default_moe_k_private = _config_int("moe_k_private", 2)
    default_moe_k_shared = _config_int("moe_k_shared", 2)

    has_lora_moes = any(isinstance(k, str) and k.startswith("lora_moes.") for k in shared_state.keys())
    has_wrapped = any(isinstance(k, str) and k.startswith("wrapped_blocks.") for k in shared_state.keys())
    use_lora_moe = bool(has_lora_moes or has_wrapped)
    if not use_lora_moe:
        return {
            "use_lora_moe": False,
            "task_num": 3,
            "lora_rank": 8,
            "num_experts_private": 2,
            "num_experts_shared": 6,
            "moe_k_private": default_moe_k_private,
            "moe_k_shared": default_moe_k_shared,
        }

    lora_rank = 8
    num_experts_private = 2
    num_experts_shared = 6
    task_num = 3

    a_private = shared_state.get("lora_moes.0.lora_A_private")
    if hasattr(a_private, "shape") and len(getattr(a_private, "shape", [])) == 4:
        task_num = int(a_private.shape[0])
        num_experts_private = int(a_private.shape[1])
        lora_rank = int(a_private.shape[3])

    a_shared = shared_state.get("lora_moes.0.lora_A_shared")
    if hasattr(a_shared, "shape") and len(getattr(a_shared, "shape", [])) == 3:
        num_experts_shared = int(a_shared.shape[0])
        lora_rank = int(a_shared.shape[2])

    gate_indices: List[int] = []
    for key in shared_state.keys():
        if not isinstance(key, str):
            continue
        if key.startswith("lora_moes.0.f_gate_private.") and key.endswith(".weight"):
            parts = key.split(".")
            if len(parts) >= 4:
                try:
                    gate_indices.append(int(parts[3]))
                except ValueError:
                    pass
    if gate_indices:
        task_num = max(gate_indices) + 1

    return {
        "use_lora_moe": True,
        "task_num": int(task_num),
        "lora_rank": int(lora_rank),
        "num_experts_private": int(num_experts_private),
        "num_experts_shared": int(num_experts_shared),
        "moe_k_private": default_moe_k_private,
        "moe_k_shared": default_moe_k_shared,
    }


def _import_multitask_models():
    try:
        from .models import MultiTaskModel, SharedDinoV3Backbone
    except ImportError:
        from models import MultiTaskModel, SharedDinoV3Backbone
    return MultiTaskModel, SharedDinoV3Backbone


def _parse_block_ids(text: str) -> List[int]:
    items = [part.strip() for part in (text or "").split(",") if part.strip()]
    if not items:
        raise ValueError("--block-ids must contain at least one integer.")
    block_ids = []
    for item in items:
        value = int(item)
        if value < 0:
            raise ValueError(f"Block id must be >= 0, got {value}.")
        if value not in block_ids:
            block_ids.append(value)
    return block_ids


def _set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        out = Path(args.checkpoint).resolve().parent / "grad_conflict"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_det_train_paths(data_root: str, train_ann: str | None, train_img_dir: str | None) -> tuple[Path, Path]:
    root = Path(data_root)
    if not (root / "annotations").exists() and (root / "coco" / "annotations").exists():
        root = root / "coco"
    ann_path = Path(train_ann) if train_ann else root / "annotations" / "instances_train.json"
    img_dir = Path(train_img_dir) if train_img_dir else root / "images" / "train"
    return ann_path, img_dir


def _resolve_cnt_train_dir(data_root: str, train_dir: str | None) -> Path:
    root = Path(data_root)
    if train_dir:
        return Path(train_dir)
    return root / "train_data_class8"


def _build_train_loaders(
    *,
    args: argparse.Namespace,
    cnt_num_classes: int,
) -> Dict[str, DataLoader]:
    num_workers = int(args.num_workers)

    det_ann, det_img_dir = _resolve_det_train_paths(args.det_data_root, args.det_train_ann, args.det_train_img_dir)
    det_ds = CocoDetectionDataset(str(det_ann), str(det_img_dir), transform=DetTransform(True))
    seg_ds = SegmentationDataset(
        args.seg_train_dir,
        transform=SegTransform(True, IMAGENET_MEAN, IMAGENET_STD),
        image_size=int(args.image_size),
    )
    cnt_train_dir = _resolve_cnt_train_dir(args.cnt_data_root, args.cnt_train_dir)
    cnt_ds = DSACADensityH5Dataset(
        str(cnt_train_dir),
        num_classes=int(cnt_num_classes),
        transform=CountTransform(True, IMAGENET_MEAN, IMAGENET_STD),
        image_size=int(args.image_size),
        keep_aspect=bool(args.cnt_keep_aspect),
    )

    det_gen = torch.Generator().manual_seed(int(args.seed) + 101)
    seg_gen = torch.Generator().manual_seed(int(args.seed) + 202)
    cnt_gen = torch.Generator().manual_seed(int(args.seed) + 303)

    det_loader = DataLoader(
        det_ds,
        batch_size=int(args.det_batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        generator=det_gen,
    )
    seg_loader = DataLoader(
        seg_ds,
        batch_size=int(args.seg_batch_size),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=seg_gen,
    )

    cnt_kwargs = {
        "batch_size": int(args.cnt_batch_size),
        "shuffle": True,
        "drop_last": True,
        "num_workers": num_workers,
        "pin_memory": True,
        "generator": cnt_gen,
    }
    if num_workers > 0:
        cnt_kwargs["persistent_workers"] = True
        cnt_kwargs["multiprocessing_context"] = "spawn"
        cnt_kwargs["prefetch_factor"] = 2
    cnt_loader = DataLoader(cnt_ds, **cnt_kwargs)
    return {"det": det_loader, "seg": seg_loader, "cnt": cnt_loader}


def _build_model_from_checkpoint(args: argparse.Namespace) -> tuple[torch.nn.Module, Dict[str, object]]:
    ckpt = _torch_load_cpu(args.checkpoint)
    if not _is_multitask_checkpoint(ckpt):
        raise SystemExit(f"Checkpoint is not a multitask checkpoint: {args.checkpoint}")

    shared_state = ckpt["backbone"]
    det_state = ckpt["det_head"]
    seg_state = ckpt["seg_head"]
    cnt_state = ckpt["cnt_head"]
    ckpt_config = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}

    if not all(isinstance(x, dict) for x in (shared_state, det_state, seg_state, cnt_state)):
        raise SystemExit("Checkpoint backbone/det_head/seg_head/cnt_head entries are malformed.")

    det_fg = _infer_fg_num_classes_from_det_state(det_state)
    seg_nc = _infer_num_classes_from_conv1x1_weight(seg_state, "decode.3.weight")
    cnt_nc = _infer_num_classes_from_conv1x1_weight(cnt_state, "decode.3.weight")
    det_out_channels = _infer_det_out_channels(det_state)
    moe_cfg = _infer_lora_moe_config_from_shared_state(shared_state, ckpt_config)
    if not bool(moe_cfg["use_lora_moe"]):
        raise SystemExit("This checkpoint does not contain LoRA-MoE shared experts.")

    MultiTaskModel, SharedDinoV3Backbone = _import_multitask_models()

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    unfreeze_backbone = bool(ckpt_config.get("unfreeze_backbone", False))
    shared = SharedDinoV3Backbone(
        model_name=str(args.model_name),
        image_size=int(args.backbone_init_image_size),
        checkpoint_path=None,
        use_lora_moe=True,
        backbone_trainable=unfreeze_backbone,
        task_num=int(moe_cfg["task_num"]),
        lora_rank=int(moe_cfg["lora_rank"]),
        num_experts_private=int(moe_cfg["num_experts_private"]),
        num_experts_shared=int(moe_cfg["num_experts_shared"]),
        moe_k_private=int(moe_cfg["moe_k_private"]),
        moe_k_shared=int(moe_cfg["moe_k_shared"]),
        grad_checkpointing=False,
    )
    model = MultiTaskModel(
        shared=shared,
        det_num_classes=int(det_fg),
        seg_num_classes=int(seg_nc),
        cnt_num_classes=int(cnt_nc),
        image_size=int(args.image_size),
        det_out_channels=int(det_out_channels),
        det_train_backbone=unfreeze_backbone,
        seg_train_backbone=unfreeze_backbone,
        cnt_train_backbone=unfreeze_backbone,
    ).to(device)
    load_info = load_multitask_checkpoint(args.checkpoint, model=model, map_location="cpu")
    if not bool(load_info.get("_load_complete", False)):
        raise SystemExit(f"Checkpoint load is incomplete for {args.checkpoint}: {load_info.get('_load_report')}")

    meta = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "model_name": str(args.model_name),
        "image_size": int(args.image_size),
        "backbone_init_image_size": int(args.backbone_init_image_size),
        "det_num_classes": int(det_fg),
        "seg_num_classes": int(seg_nc),
        "cnt_num_classes": int(cnt_nc),
        "det_out_channels": int(det_out_channels),
        "unfreeze_backbone": int(unfreeze_backbone),
        "moe_cfg": moe_cfg,
    }
    return model, meta


def _theta_params(model: torch.nn.Module):
    theta_named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    theta_param_names = [name for name, _ in theta_named_params]
    theta_params = [param for _, param in theta_named_params]
    if not theta_params:
        raise RuntimeError("No trainable parameters found in loaded model.")
    return theta_param_names, theta_params


def _select_shared_a_param_indices(
    theta_param_names: List[str],
    theta_params: List[torch.nn.Parameter],
    block_ids: Iterable[int],
) -> Dict[int, Dict[str, object]]:
    matrix_index = build_shared_expert_matrix_index(theta_param_names, theta_params)
    block_set = set(int(block_id) for block_id in block_ids)
    selected: Dict[int, Dict[str, object]] = {}
    for unit in matrix_index.matrix_units:
        if unit.matrix_key != "a" or unit.block_id not in block_set:
            continue
        item = selected.setdefault(
            int(unit.block_id),
            {"param_index": int(unit.param_index), "expert_indices": []},
        )
        if int(item["param_index"]) != int(unit.param_index):
            raise RuntimeError(f"Block {unit.block_id} maps to multiple lora_A_shared param indices.")
        item["expert_indices"].append(int(unit.expert_index))

    missing_blocks = [int(block_id) for block_id in block_ids if int(block_id) not in selected]
    if missing_blocks:
        raise RuntimeError(f"Missing target blocks in shared LoRA-A params: {missing_blocks}")

    for block_id, item in selected.items():
        expert_indices = sorted(set(int(idx) for idx in item["expert_indices"]))
        param_index = int(item["param_index"])
        num_experts = int(theta_params[param_index].shape[0])
        if expert_indices != list(range(num_experts)):
            raise RuntimeError(f"Block {block_id} experts are incomplete: got {expert_indices}, expected 0..{num_experts-1}")
        item["expert_indices"] = expert_indices
        item["num_experts"] = num_experts
    return dict(sorted(selected.items()))


def _grad_slice_vector(grad: torch.Tensor | None, expert_index: int) -> torch.Tensor | None:
    if grad is None:
        return None
    return grad[expert_index].detach().float().reshape(-1)


def _vector_norm(vec: torch.Tensor | None) -> float:
    if vec is None:
        return 0.0
    return float(vec.norm().item())


def _sample_train_batches(loaders: Dict[str, DataLoader], seed: int) -> Dict[str, object]:
    batches = {}
    task_seed_offsets = {"det": 1001, "seg": 2002, "cnt": 3003}
    for task_name in ("det", "seg", "cnt"):
        _set_global_seed(int(seed) + task_seed_offsets[task_name])
        batches[task_name] = next(iter(loaders[task_name]))
    return batches


def _task_batch_size(task_name: str, batch: object) -> int:
    if task_name == "det":
        images, _targets = batch
        return len(images)
    images, _target = batch
    return int(images.size(0))


def _compute_task_grads(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    theta_params: List[torch.nn.Parameter],
    batches: Dict[str, object],
    device: torch.device,
) -> Dict[str, tuple[torch.Tensor | None, ...]]:
    optimizer = torch.optim.SGD(theta_params, lr=0.0)
    task_grads: Dict[str, tuple[torch.Tensor | None, ...]] = {}
    for task_name in ("det", "seg", "cnt"):
        model.train()
        with freeze_batchnorm_stats(model):
            _loss, grads = _compute_task_loss_and_grads(
                args=args,
                model=model,
                optimizer_theta=optimizer,
                theta_params=theta_params,
                task_name=task_name,
                batch=batches[task_name],
                device=device,
                use_ddp=False,
                world_size=1,
            )
        task_grads[task_name] = grads
    return task_grads


def _summarize_block_cosines(
    *,
    task_grads: Dict[str, tuple[torch.Tensor | None, ...]],
    shared_a_params: Dict[int, Dict[str, object]],
) -> Dict[int, List[Dict[str, object]]]:
    block_results: Dict[int, List[Dict[str, object]]] = {}
    for block_id, info in shared_a_params.items():
        param_index = int(info["param_index"])
        rows: List[Dict[str, object]] = []
        for expert_index in info["expert_indices"]:
            det_vec = _grad_slice_vector(task_grads["det"][param_index], expert_index)
            seg_vec = _grad_slice_vector(task_grads["seg"][param_index], expert_index)
            cnt_vec = _grad_slice_vector(task_grads["cnt"][param_index], expert_index)
            rows.append(
                {
                    "expert_index": int(expert_index),
                    "det_seg": float(_safe_cosine_similarity(det_vec, seg_vec)),
                    "det_cnt": float(_safe_cosine_similarity(det_vec, cnt_vec)),
                    "seg_cnt": float(_safe_cosine_similarity(seg_vec, cnt_vec)),
                    "det_norm": float(_vector_norm(det_vec)),
                    "seg_norm": float(_vector_norm(seg_vec)),
                    "cnt_norm": float(_vector_norm(cnt_vec)),
                }
            )
        block_results[int(block_id)] = rows
    return block_results


def _plot_block(block_id: int, rows: List[Dict[str, object]], output_dir: Path, checkpoint_name: str, seed: int) -> Path:
    expert_labels = [f"Expert {int(row['expert_index']) + 1}" for row in rows]
    indices = np.arange(len(expert_labels), dtype=np.float32)
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    offsets = {
        "det_seg": -width,
        "det_cnt": 0.0,
        "seg_cnt": width,
    }
    for pair_name in PAIR_ORDER:
        values = [float(row[pair_name]) for row in rows]
        ax.bar(
            indices + offsets[pair_name],
            values,
            width=width,
            label=pair_name.replace("_", "-"),
            color=PAIR_COLORS[pair_name],
        )

    ax.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    ax.set_xticks(indices)
    ax.set_xticklabels(expert_labels)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel("Shared Experts")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"Stage2 Shared LoRA-A Gradient Cosines | block {block_id} | {checkpoint_name} | seed {seed}")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.35)

    path = output_dir / f"block_{block_id:02d}_shared_lora_a_cosine.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _save_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "block_id",
        "expert_index",
        "det_seg",
        "det_cnt",
        "seg_cnt",
        "det_norm",
        "seg_norm",
        "cnt_norm",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    _set_global_seed(int(args.seed))
    output_dir = _resolve_output_dir(args)
    block_ids = _parse_block_ids(args.block_ids)

    model, model_meta = _build_model_from_checkpoint(args)
    if args.cnt_num_classes is not None and int(args.cnt_num_classes) != int(model_meta["cnt_num_classes"]):
        raise RuntimeError(
            f"--cnt-num-classes={int(args.cnt_num_classes)} does not match checkpoint cnt classes "
            f"{int(model_meta['cnt_num_classes'])}."
        )
    cnt_num_classes = int(args.cnt_num_classes) if args.cnt_num_classes is not None else int(model_meta["cnt_num_classes"])
    loaders = _build_train_loaders(args=args, cnt_num_classes=cnt_num_classes)

    theta_param_names, theta_params = _theta_params(model)
    shared_a_params = _select_shared_a_param_indices(theta_param_names, theta_params, block_ids)
    for block_id, item in shared_a_params.items():
        num_experts = int(item["num_experts"])
        if num_experts != 6:
            raise RuntimeError(f"Expected 6 shared experts at block {block_id}, but found {num_experts}.")

    batches = _sample_train_batches(loaders, seed=int(args.seed))
    task_grads = _compute_task_grads(
        args=args,
        model=model,
        theta_params=theta_params,
        batches=batches,
        device=torch.device(args.device),
    )
    block_results = _summarize_block_cosines(task_grads=task_grads, shared_a_params=shared_a_params)

    checkpoint_name = Path(args.checkpoint).stem
    figure_paths: Dict[str, str] = {}
    csv_rows: List[Dict[str, object]] = []
    for block_id, rows in block_results.items():
        figure_path = _plot_block(
            block_id=block_id,
            rows=rows,
            output_dir=output_dir,
            checkpoint_name=checkpoint_name,
            seed=int(args.seed),
        )
        figure_paths[str(block_id)] = str(figure_path)
        for row in rows:
            csv_rows.append({"block_id": int(block_id), **row})

    summary = {
        "meta": {
            **model_meta,
            "seed": int(args.seed),
            "block_ids": [int(block_id) for block_id in block_ids],
            "output_dir": str(output_dir.resolve()),
            "num_workers": int(args.num_workers),
            "cnt_count_loss_weight": float(args.cnt_count_loss_weight),
            "cnt_backbone_grad_mult": float(args.cnt_backbone_grad_mult),
            "sampled_batch_sizes": {
                task_name: int(_task_batch_size(task_name, batches[task_name]))
                for task_name in ("det", "seg", "cnt")
            },
            "pair_order": list(PAIR_ORDER),
            "figure_paths": figure_paths,
        },
        "blocks": {str(block_id): rows for block_id, rows in block_results.items()},
    }
    json_path = output_dir / "cosine_summary.json"
    csv_path = output_dir / "cosine_summary.csv"
    _save_json(json_path, summary)
    _save_csv(csv_path, csv_rows)

    print(f"Saved figures to {output_dir}")
    print(f"Saved JSON summary to {json_path}")
    print(f"Saved CSV summary to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


_MODULE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _MODULE_ROOT.parents[1]
for _path in (_MODULE_ROOT, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


try:
    from .datasets import CountTransform, DetTransform, IMAGENET_MEAN, IMAGENET_STD, SegTransform
    from .eval_train_model import _build_and_load_model
    from .train_utils import freeze_batchnorm_stats, to_device_cnt, to_device_det, to_device_seg
except Exception:
    from datasets import CountTransform, DetTransform, IMAGENET_MEAN, IMAGENET_STD, SegTransform
    from eval_train_model import _build_and_load_model
    from train_utils import freeze_batchnorm_stats, to_device_cnt, to_device_det, to_device_seg

from counting.dataset import DSACADensityH5Dataset
from object_detection.dataset import CocoDetectionDataset, collate_fn
from segmentation.dataset import SegmentationDataset


DEFAULT_DET_DATA_ROOT = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/train_10per"
DEFAULT_DET_TRAIN_ANN = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_train.json"
DEFAULT_DET_TRAIN_IMG_DIR = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco"
DEFAULT_SEG_TRAIN_DIR = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/train_5500"
DEFAULT_CNT_DATA_ROOT = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/train_10per"
DEFAULT_CNT_TRAIN_DIR = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA/train_data_class8"

TASK_NAMES = ("det", "seg", "cnt")
PAIR_ORDER = (("det", "seg"), ("det", "cnt"), ("seg", "cnt"))
PAIR_STYLES = {
    "det_seg": {"color": "#C62828", "offset": -0.22, "label": "det-seg"},
    "det_cnt": {"color": "#1565C0", "offset": 0.00, "label": "det-cnt"},
    "seg_cnt": {"color": "#2E7D32", "offset": 0.22, "label": "seg-cnt"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize gradient cosine conflicts for shared LoRA-A experts on N random multitask training batches."
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--blocks", type=str, default="6,12,18,24")
    p.add_argument("--max-batches", type=int, default=200)
    p.add_argument("--min-mixed-experts", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--backbone-init-image-size", type=int, default=64)
    p.add_argument("--model-name", type=str, default="dinov3_vitl16")
    p.add_argument("--figure-dpi", type=int, default=220)
    p.add_argument("--cnt-count-loss-weight", type=float, default=1.0)

    p.add_argument("--det-data-root", type=str, default=DEFAULT_DET_DATA_ROOT)
    p.add_argument("--det-train-ann", type=str, default=DEFAULT_DET_TRAIN_ANN)
    p.add_argument("--det-train-img-dir", type=str, default=DEFAULT_DET_TRAIN_IMG_DIR)
    p.add_argument("--det-batch-size", type=int, default=2)
    p.add_argument("--det-num-workers", type=int, default=0)

    p.add_argument("--seg-train-dir", type=str, default=DEFAULT_SEG_TRAIN_DIR)
    p.add_argument("--seg-num-classes", type=int, default=None)
    p.add_argument("--seg-batch-size", type=int, default=2)
    p.add_argument("--seg-num-workers", type=int, default=0)

    p.add_argument("--cnt-data-root", type=str, default=DEFAULT_CNT_DATA_ROOT)
    p.add_argument("--cnt-train-dir", type=str, default=DEFAULT_CNT_TRAIN_DIR)
    p.add_argument("--cnt-num-classes", type=int, default=None)
    p.add_argument("--cnt-batch-size", type=int, default=2)
    p.add_argument("--cnt-num-workers", type=int, default=0)
    aspect = p.add_mutually_exclusive_group()
    aspect.add_argument("--cnt-keep-aspect", dest="cnt_keep_aspect", action="store_true")
    aspect.add_argument("--cnt-no-keep-aspect", dest="cnt_keep_aspect", action="store_false")
    p.set_defaults(cnt_keep_aspect=True)
    return p.parse_args()


def _seed_everything(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_blocks(text: str) -> List[int]:
    items = [part.strip() for part in str(text).split(",") if part.strip()]
    if not items:
        raise ValueError("--blocks cannot be empty")
    blocks = [int(item) for item in items]
    if any(block < 1 for block in blocks):
        raise ValueError("--blocks must use 1-based block ids, e.g. 6,12,18,24")
    deduped: List[int] = []
    for block in blocks:
        if block not in deduped:
            deduped.append(block)
    return deduped


def _default_output_dir(checkpoint: str, output_dir: str | None) -> Path:
    if output_dir:
        path = Path(output_dir)
    else:
        path = Path(checkpoint).resolve().parent / "gradient_conflict_vis"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_cnt_split_root(path_text: str) -> str:
    path = Path(path_text)
    if path.name in {"gt_density_map", "gt_density_map_compressed"}:
        return str(path.parent)
    return str(path)


def _build_loader_kwargs(
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool,
    generator: torch.Generator,
    collate=None,
) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": True,
        "generator": generator,
    }
    if collate is not None:
        kwargs["collate_fn"] = collate
    if int(num_workers) > 0 and collate is None:
        kwargs["persistent_workers"] = True
    return kwargs


def _build_det_loader(args: argparse.Namespace) -> DataLoader:
    ds = CocoDetectionDataset(
        str(args.det_train_ann),
        str(args.det_train_img_dir),
        transform=DetTransform(False),
    )
    generator = torch.Generator()
    generator.manual_seed(int(args.seed) + 11)
    return DataLoader(
        ds,
        **_build_loader_kwargs(
            args.det_batch_size,
            args.det_num_workers,
            shuffle=True,
            generator=generator,
            collate=collate_fn,
        ),
    )


def _build_seg_loader(args: argparse.Namespace) -> DataLoader:
    ds = SegmentationDataset(
        str(args.seg_train_dir),
        transform=SegTransform(False, IMAGENET_MEAN, IMAGENET_STD),
        image_size=int(args.image_size),
    )
    generator = torch.Generator()
    generator.manual_seed(int(args.seed) + 23)
    return DataLoader(
        ds,
        **_build_loader_kwargs(
            args.seg_batch_size,
            args.seg_num_workers,
            shuffle=True,
            generator=generator,
        ),
    )


def _build_cnt_loader(args: argparse.Namespace, *, num_classes: int) -> DataLoader:
    ds = DSACADensityH5Dataset(
        _normalize_cnt_split_root(str(args.cnt_train_dir)),
        num_classes=int(num_classes),
        transform=CountTransform(False, IMAGENET_MEAN, IMAGENET_STD),
        image_size=int(args.image_size),
        keep_aspect=bool(args.cnt_keep_aspect),
    )
    generator = torch.Generator()
    generator.manual_seed(int(args.seed) + 37)
    kwargs = _build_loader_kwargs(
        args.cnt_batch_size,
        args.cnt_num_workers,
        shuffle=True,
        generator=generator,
    )
    if int(args.cnt_num_workers) > 0:
        kwargs["multiprocessing_context"] = "spawn"
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


def _next_batch(loader: DataLoader):
    try:
        return next(iter(loader))
    except StopIteration as exc:
        raise RuntimeError("DataLoader returned no batch. Check dataset path and batch-size settings.") from exc


def _next_batch_from_iterator(
    loader: DataLoader,
    iterator,
):
    try:
        batch = next(iterator)
        return batch, iterator
    except StopIteration:
        iterator = iter(loader)
        try:
            batch = next(iterator)
            return batch, iterator
        except StopIteration as exc:
            raise RuntimeError("DataLoader returned no batch. Check dataset path and batch-size settings.") from exc


def _safe_cosine_with_meta(a: torch.Tensor | None, b: torch.Tensor | None) -> tuple[float, bool, float, float]:
    if a is None or b is None:
        return 0.0, False, 0.0, 0.0
    norm_a = float(a.norm().item())
    norm_b = float(b.norm().item())
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0, False, norm_a, norm_b
    cosine = float(torch.dot(a, b).item() / max(norm_a * norm_b, 1e-12))
    cosine = max(min(cosine, 1.0), -1.0)
    return cosine, True, norm_a, norm_b


def _extract_lora_a_grads(model, block_map: Dict[int, int]) -> Dict[int, torch.Tensor | None]:
    out: Dict[int, torch.Tensor | None] = {}
    for ext_block, internal_block in block_map.items():
        grad = model.shared.lora_moes[int(internal_block)].lora_A_shared.grad
        out[int(ext_block)] = None if grad is None else grad.detach().cpu().clone()
    return out


def _compute_task_loss_and_shared_grads(
    *,
    model,
    task_name: str,
    batch,
    block_map: Dict[int, int],
    device: torch.device,
    cnt_count_loss_weight: float,
) -> tuple[float, Dict[int, torch.Tensor | None], Dict[str, object]]:
    model.zero_grad(set_to_none=True)
    model.train()

    batch_meta: Dict[str, object]
    with freeze_batchnorm_stats(model):
        if task_name == "det":
            images, targets = to_device_det(batch, device)
            batch_meta = {
                "batch_size": int(len(images)),
                "image_ids": [int(tgt["image_id"].item()) for tgt in targets if "image_id" in tgt],
            }
            loss_dict = model("det", images, targets)
            loss = sum(loss_dict.values())
        elif task_name == "seg":
            images, masks = to_device_seg(batch, device)
            batch_meta = {
                "batch_size": int(images.shape[0]),
                "image_shape": [int(v) for v in images.shape],
                "mask_shape": [int(v) for v in masks.shape],
            }
            logits = model("seg", images)
            loss = F.cross_entropy(logits, masks)
        elif task_name == "cnt":
            images, density = to_device_cnt(batch, device)
            gt_counts = density.flatten(2).sum(dim=2)
            batch_meta = {
                "batch_size": int(images.shape[0]),
                "image_shape": [int(v) for v in images.shape],
                "density_shape": [int(v) for v in density.shape],
            }
            pred_density, pred_counts = model("cnt", images, cnt_backbone_grad_mult=1.0)
            density_loss = F.mse_loss(pred_density, density, reduction="sum") / images.size(0)
            count_loss = F.l1_loss(pred_counts, gt_counts)
            loss = density_loss + float(cnt_count_loss_weight) * count_loss
        else:
            raise ValueError(f"Unknown task: {task_name}")
        loss.backward()

    grads = _extract_lora_a_grads(model, block_map)
    loss_value = float(loss.detach().item())
    model.zero_grad(set_to_none=True)
    return loss_value, grads, batch_meta


def _build_cosine_rows(
    *,
    task_grads: Dict[str, Dict[int, torch.Tensor | None]],
    block_map: Dict[int, int],
    num_shared_experts: int,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for ext_block, internal_block in block_map.items():
        for expert_index in range(int(num_shared_experts)):
            task_vectors: Dict[str, torch.Tensor | None] = {}
            for task_name in TASK_NAMES:
                block_grad = task_grads[task_name][ext_block]
                task_vectors[task_name] = None if block_grad is None else block_grad[expert_index].float().reshape(-1)

            for task_a, task_b in PAIR_ORDER:
                pair_name = f"{task_a}_{task_b}"
                cosine, valid, norm_a, norm_b = _safe_cosine_with_meta(task_vectors[task_a], task_vectors[task_b])
                rows.append(
                    {
                        "block_external": int(ext_block),
                        "block_internal": int(internal_block),
                        "expert_index": int(expert_index),
                        "pair": pair_name,
                        "task_a": task_a,
                        "task_b": task_b,
                        "cosine": float(cosine),
                        "valid": int(valid),
                        "task_a_grad_norm": float(norm_a),
                        "task_b_grad_norm": float(norm_b),
                    }
                )
    return rows


def _write_rows_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "block_external",
        "block_internal",
        "expert_index",
        "pair",
        "task_a",
        "task_b",
        "cosine",
        "valid",
        "task_a_grad_norm",
        "task_b_grad_norm",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            formatted["cosine"] = f"{float(row['cosine']):.8f}"
            formatted["task_a_grad_norm"] = f"{float(row['task_a_grad_norm']):.8f}"
            formatted["task_b_grad_norm"] = f"{float(row['task_b_grad_norm']):.8f}"
            writer.writerow(formatted)


def _summarize_rows(rows: Sequence[Dict[str, object]], block_map: Dict[int, int]) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for ext_block, internal_block in block_map.items():
        block_rows = [row for row in rows if int(row["block_external"]) == int(ext_block)]
        pair_summary: Dict[str, object] = {}
        for task_a, task_b in PAIR_ORDER:
            pair_name = f"{task_a}_{task_b}"
            pair_rows = [row for row in block_rows if row["pair"] == pair_name]
            valid_rows = [row for row in pair_rows if int(row["valid"]) == 1]
            cosines = [float(row["cosine"]) for row in valid_rows]
            negative_count = sum(1 for value in cosines if value < 0.0)
            pair_summary[pair_name] = {
                "valid_points": int(len(valid_rows)),
                "invalid_points": int(len(pair_rows) - len(valid_rows)),
                "negative_points": int(negative_count),
                "negative_ratio": float(negative_count / max(len(valid_rows), 1)),
                "mean_cosine": float(sum(cosines) / len(cosines)) if cosines else 0.0,
                "min_cosine": float(min(cosines)) if cosines else 0.0,
                "max_cosine": float(max(cosines)) if cosines else 0.0,
            }
        summary[str(ext_block)] = {
            "block_internal": int(internal_block),
            "pairs": pair_summary,
        }
    return summary


def _build_block_filter_summary(
    rows: Sequence[Dict[str, object]],
    block_map: Dict[int, int],
    *,
    min_mixed_experts: int = 4,
) -> Dict[str, object]:
    summary: Dict[str, object] = {}
    for ext_block, internal_block in block_map.items():
        block_rows = [row for row in rows if int(row["block_external"]) == int(ext_block)]
        mixed_experts: List[int] = []
        expert_details: Dict[str, object] = {}
        for expert_index in sorted({int(row["expert_index"]) for row in block_rows}):
            expert_rows = [row for row in block_rows if int(row["expert_index"]) == int(expert_index)]
            valid_cosines = [float(row["cosine"]) for row in expert_rows if int(row["valid"]) == 1]
            positive_count = sum(1 for value in valid_cosines if value > 0.0)
            negative_count = sum(1 for value in valid_cosines if value < 0.0)
            is_mixed = positive_count > 0 and negative_count > 0
            if is_mixed:
                mixed_experts.append(int(expert_index))
            expert_details[str(expert_index)] = {
                "valid_pair_count": int(len(valid_cosines)),
                "positive_count": int(positive_count),
                "negative_count": int(negative_count),
                "is_mixed_sign": bool(is_mixed),
            }

        keep_block = len(mixed_experts) >= int(min_mixed_experts)
        summary[str(ext_block)] = {
            "block_internal": int(internal_block),
            "mixed_sign_experts": mixed_experts,
            "mixed_sign_expert_count": int(len(mixed_experts)),
            "keep": bool(keep_block),
            "expert_details": expert_details,
        }
    return summary


def _render_block_plot(
    *,
    rows: Sequence[Dict[str, object]],
    block_external: int,
    output_path: Path,
    dpi: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to render gradient conflict plots") from exc

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))
    expert_ids = sorted({int(row["expert_index"]) for row in rows})
    base_x = np.asarray(expert_ids, dtype=np.float32)
    bar_width = 0.22

    ax.axhline(0.0, color="#424242", linestyle="--", linewidth=1.0, alpha=0.9)
    for task_a, task_b in PAIR_ORDER:
        pair_name = f"{task_a}_{task_b}"
        style = PAIR_STYLES[pair_name]
        pair_rows = sorted(
            (row for row in rows if row["pair"] == pair_name),
            key=lambda row: int(row["expert_index"]),
        )
        xs = np.asarray([float(row["expert_index"]) + float(style["offset"]) for row in pair_rows], dtype=np.float32)
        ys = np.asarray([float(row["cosine"]) for row in pair_rows], dtype=np.float32)
        ax.bar(
            xs,
            ys,
            width=bar_width,
            color=style["color"],
            edgecolor="white",
            linewidth=0.6,
            alpha=0.82,
            label=str(style["label"]),
            zorder=3,
        )

    pair_text_parts: List[str] = []
    for task_a, task_b in PAIR_ORDER:
        pair_name = f"{task_a}_{task_b}"
        pair_rows = [row for row in rows if row["pair"] == pair_name]
        valid_rows = [row for row in pair_rows if int(row["valid"]) == 1]
        neg_count = sum(1 for row in valid_rows if float(row["cosine"]) < 0.0)
        pair_text_parts.append(f"{PAIR_STYLES[pair_name]['label']}: {neg_count}/{len(valid_rows)}<0")
    ax.text(
        0.02,
        0.02,
        "\n".join(pair_text_parts),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BDBDBD", "alpha": 0.9},
    )

    ax.set_title(f"Block {int(block_external)}")
    ax.set_xlabel("Shared expert index")
    ax.set_ylabel("Cosine similarity")
    ax.set_xticks(base_x)
    ax.set_xticklabels([f"E{idx}" for idx in expert_ids])
    ax.set_ylim(-1.05, 1.05)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.5)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.08),
            ncol=min(len(labels), 4),
            frameon=False,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    fig.savefig(output_path, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def _analyze_one_batch(
    *,
    batch_index: int,
    batches: Dict[str, object],
    model,
    block_map: Dict[int, int],
    device: torch.device,
    cnt_count_loss_weight: float,
    num_shared_experts: int,
    min_mixed_experts: int,
) -> Dict[str, object]:
    task_losses: Dict[str, float] = {}
    task_grads: Dict[str, Dict[int, torch.Tensor | None]] = {}
    task_batch_meta: Dict[str, Dict[str, object]] = {}
    for task_name in TASK_NAMES:
        loss_value, grad_map, batch_meta = _compute_task_loss_and_shared_grads(
            model=model,
            task_name=task_name,
            batch=batches[task_name],
            block_map=block_map,
            device=device,
            cnt_count_loss_weight=float(cnt_count_loss_weight),
        )
        task_losses[task_name] = float(loss_value)
        task_grads[task_name] = grad_map
        task_batch_meta[task_name] = batch_meta

    rows = _build_cosine_rows(
        task_grads=task_grads,
        block_map=block_map,
        num_shared_experts=num_shared_experts,
    )
    block_summary = _summarize_rows(rows, block_map)
    block_filter = _build_block_filter_summary(rows, block_map, min_mixed_experts=int(min_mixed_experts))

    return {
        "batch_index": int(batch_index),
        "task_losses": {task: float(value) for task, value in task_losses.items()},
        "task_batch_meta": task_batch_meta,
        "rows": rows,
        "block_summary": block_summary,
        "block_filter": block_filter,
    }


def main() -> int:
    args = parse_args()
    _seed_everything(args.seed)
    if int(args.max_batches) < 1:
        raise SystemExit("--max-batches must be >= 1")
    if int(args.min_mixed_experts) < 1:
        raise SystemExit("--min-mixed-experts must be >= 1")

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    output_dir = _default_output_dir(args.checkpoint, args.output_dir)
    requested_blocks = _parse_blocks(args.blocks)

    model, meta, _load_summary = _build_and_load_model(args)
    if not bool(meta.get("use_lora_moe", False)):
        raise SystemExit("Checkpoint does not contain LoRA-MoE weights; cannot analyze shared expert gradient conflicts")
    if int(meta["moe_cfg"]["task_num"]) != 3:
        raise SystemExit(f"Expected 3 tasks (det/seg/cnt), got {meta['moe_cfg']['task_num']}")

    total_blocks = len(model.shared.lora_moes)
    block_map: Dict[int, int] = {}
    for ext_block in requested_blocks:
        internal_block = int(ext_block) - 1
        if internal_block < 0 or internal_block >= total_blocks:
            raise SystemExit(
                f"Requested block {ext_block} maps to internal block {internal_block}, "
                f"but checkpoint only has {total_blocks} blocks"
            )
        block_map[int(ext_block)] = internal_block

    num_shared_experts = int(meta["moe_cfg"]["num_experts_shared"])
    cnt_num_classes = int(args.cnt_num_classes) if args.cnt_num_classes is not None else int(meta["cnt_num_classes"])

    det_loader = _build_det_loader(args)
    seg_loader = _build_seg_loader(args)
    cnt_loader = _build_cnt_loader(args, num_classes=cnt_num_classes)
    loader_map = {
        "det": det_loader,
        "seg": seg_loader,
        "cnt": cnt_loader,
    }
    loader_iters = {task_name: iter(loader) for task_name, loader in loader_map.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_blocks: Dict[str, object] = {}
    stop_reason = "max_batches_reached"
    batches_processed = 0

    for batch_index in range(1, int(args.max_batches) + 1):
        batches_processed = int(batch_index)
        batches: Dict[str, object] = {}
        for task_name in TASK_NAMES:
            batch, loader_iters[task_name] = _next_batch_from_iterator(loader_map[task_name], loader_iters[task_name])
            batches[task_name] = batch

        batch_analysis = _analyze_one_batch(
            batch_index=batch_index,
            batches=batches,
            model=model,
            block_map=block_map,
            device=device,
            cnt_count_loss_weight=float(args.cnt_count_loss_weight),
            num_shared_experts=num_shared_experts,
            min_mixed_experts=int(args.min_mixed_experts),
        )
        rows = batch_analysis["rows"]
        block_filter = batch_analysis["block_filter"]
        block_summary = batch_analysis["block_summary"]

        newly_selected: List[str] = []
        for ext_block, internal_block in block_map.items():
            block_key = str(ext_block)
            if block_key in selected_blocks:
                continue
            filter_info = dict(block_filter[block_key])
            if not bool(filter_info["keep"]):
                continue

            block_rows = [row for row in rows if int(row["block_external"]) == int(ext_block)]
            block_tag = f"block_{int(ext_block):02d}"
            csv_path = output_dir / f"{block_tag}_shared_loraA_gradient_cosines.csv"
            figure_path = output_dir / f"{block_tag}_shared_loraA_gradient_cosines.png"
            _write_rows_csv(csv_path, block_rows)
            _render_block_plot(
                rows=block_rows,
                block_external=int(ext_block),
                output_path=figure_path,
                dpi=int(args.figure_dpi),
            )
            selected_blocks[block_key] = {
                "batch_index": int(batch_index),
                "block_internal": int(internal_block),
                "task_losses": dict(batch_analysis["task_losses"]),
                "task_batch_meta": dict(batch_analysis["task_batch_meta"]),
                "block_summary": dict(block_summary[block_key]),
                "filter": filter_info,
                "outputs": {
                    "cosine_csv": str(csv_path.resolve()),
                    "figure_png": str(figure_path.resolve()),
                },
            }
            newly_selected.append(block_key)

        print(
            f"[grad_conflict] batch {batch_index}: "
            f"new={newly_selected} selected={sorted(selected_blocks.keys())}"
        )
        if len(selected_blocks) == len(block_map):
            stop_reason = "all_requested_blocks_found"
            break

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "output_dir": str(output_dir.resolve()),
        "seed": int(args.seed),
        "max_batches": int(args.max_batches),
        "batches_processed": int(batches_processed),
        "stop_reason": stop_reason,
        "visualized_parameter": "shared LoRA-A",
        "tasks": list(TASK_NAMES),
        "pair_order": [f"{task_a}_{task_b}" for task_a, task_b in PAIR_ORDER],
        "block_mapping": {str(ext): int(internal) for ext, internal in block_map.items()},
        "num_shared_experts": int(num_shared_experts),
        "min_mixed_experts": int(args.min_mixed_experts),
        "selection_rule": "for each requested block, stop at the first batch where at least min_mixed_experts experts have both positive and negative valid cosine values",
        "selected_blocks": selected_blocks,
        "missing_blocks": [str(ext_block) for ext_block in block_map if str(ext_block) not in selected_blocks],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[grad_conflict] checkpoint: {Path(args.checkpoint).resolve()}")
    print(f"[grad_conflict] output_dir: {output_dir.resolve()}")
    print(f"[grad_conflict] summary: {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

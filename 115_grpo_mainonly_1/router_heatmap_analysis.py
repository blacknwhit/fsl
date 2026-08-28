from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader


_MODULE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _MODULE_ROOT.parents[1]
for _path in (_MODULE_ROOT, _REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


try:
    from .datasets import CountTransform, DetTransform, IMAGENET_MEAN, IMAGENET_STD, SegTransform
    from .eval_train_model import _build_and_load_model
except Exception:
    from datasets import CountTransform, DetTransform, IMAGENET_MEAN, IMAGENET_STD, SegTransform
    from eval_train_model import _build_and_load_model

from counting.dataset import DSACADensityH5Dataset
from object_detection.dataset import CocoDetectionDataset, collate_fn
from segmentation.dataset import SegmentationDataset


DEFAULT_DET_DATA_ROOT = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/train_10per"
DEFAULT_DET_VAL_ANN = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_val.json"
DEFAULT_DET_VAL_IMG_DIR = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco"
DEFAULT_SEG_VAL_DIR = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/val"
DEFAULT_CNT_DATA_ROOT = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/train_10per"
DEFAULT_CNT_VAL_DIR = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA/val_data_class8"
TASK_NAMES = ("det", "seg", "cnt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize task routing counts for LoRA-MoE blocks.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--blocks", type=str, default="6,12,18,24")
    p.add_argument("--max-batches-per-task", type=int, default=16)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--backbone-init-image-size", type=int, default=64)
    p.add_argument("--model-name", type=str, default="dinov3_vitl16")

    p.add_argument("--det-data-root", type=str, default=DEFAULT_DET_DATA_ROOT)
    p.add_argument("--det-val-ann", type=str, default=DEFAULT_DET_VAL_ANN)
    p.add_argument("--det-val-img-dir", type=str, default=DEFAULT_DET_VAL_IMG_DIR)
    p.add_argument("--det-batch-size", type=int, default=2)
    p.add_argument("--det-num-workers", type=int, default=4)

    p.add_argument("--seg-val-dir", type=str, default=DEFAULT_SEG_VAL_DIR)
    p.add_argument("--seg-num-classes", type=int, default=None)
    p.add_argument("--seg-batch-size", type=int, default=2)
    p.add_argument("--seg-num-workers", type=int, default=4)

    p.add_argument("--cnt-data-root", type=str, default=DEFAULT_CNT_DATA_ROOT)
    p.add_argument("--cnt-val-dir", type=str, default=DEFAULT_CNT_VAL_DIR)
    p.add_argument("--cnt-num-classes", type=int, default=None)
    p.add_argument("--cnt-batch-size", type=int, default=2)
    p.add_argument("--cnt-num-workers", type=int, default=1)
    aspect = p.add_mutually_exclusive_group()
    aspect.add_argument("--cnt-keep-aspect", dest="cnt_keep_aspect", action="store_true")
    aspect.add_argument("--cnt-no-keep-aspect", dest="cnt_keep_aspect", action="store_false")
    p.set_defaults(cnt_keep_aspect=True)
    return p.parse_args()


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
        path = Path(checkpoint).resolve().parent / "router_vis"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_loader_kwargs(batch_size: int, num_workers: int, *, collate=None) -> Dict[str, object]:
    kwargs: Dict[str, object] = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": True,
    }
    if collate is not None:
        kwargs["collate_fn"] = collate
    if int(num_workers) > 0 and collate is None:
        kwargs["persistent_workers"] = True
    return kwargs


def _build_det_loader(args: argparse.Namespace) -> DataLoader:
    ds = CocoDetectionDataset(
        str(args.det_val_ann),
        str(args.det_val_img_dir),
        transform=DetTransform(False),
    )
    return DataLoader(
        ds,
        **_build_loader_kwargs(args.det_batch_size, args.det_num_workers, collate=collate_fn),
    )


def _build_seg_loader(args: argparse.Namespace) -> DataLoader:
    ds = SegmentationDataset(
        str(args.seg_val_dir),
        transform=SegTransform(False, IMAGENET_MEAN, IMAGENET_STD),
        image_size=int(args.image_size),
    )
    return DataLoader(
        ds,
        **_build_loader_kwargs(args.seg_batch_size, args.seg_num_workers),
    )


def _build_cnt_loader(args: argparse.Namespace, *, num_classes: int) -> DataLoader:
    ds = DSACADensityH5Dataset(
        str(args.cnt_val_dir),
        num_classes=int(num_classes),
        transform=CountTransform(False, IMAGENET_MEAN, IMAGENET_STD),
        image_size=int(args.image_size),
        keep_aspect=bool(args.cnt_keep_aspect),
    )
    kwargs = _build_loader_kwargs(args.cnt_batch_size, args.cnt_num_workers)
    if int(args.cnt_num_workers) > 0:
        kwargs["multiprocessing_context"] = "spawn"
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


@dataclass
class TaskRunStats:
    batches: int = 0
    images: int = 0
    patch_tokens: int = 0


class RoutingCollector:
    def __init__(
        self,
        *,
        task_names: Sequence[str],
        block_map: Dict[int, int],
        num_shared: int,
        num_private: int,
        n_prefix: int,
    ) -> None:
        self.task_names = list(task_names)
        self.block_map = dict(block_map)
        self.num_shared = int(num_shared)
        self.num_private = int(num_private)
        self.n_prefix = int(n_prefix)
        self.shared_counts = {
            ext: torch.zeros((len(self.task_names), self.num_shared), dtype=torch.long)
            for ext in self.block_map
        }
        self.private_counts = {
            ext: torch.zeros((len(self.task_names), self.num_private), dtype=torch.long)
            for ext in self.block_map
        }
        self._handles: list[object] = []

    def attach(self, model) -> None:
        for ext_block, internal_block in self.block_map.items():
            module = model.shared.lora_moes[int(internal_block)]
            handle = module.register_forward_hook(self._build_hook(ext_block))
            self._handles.append(handle)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _build_hook(self, ext_block: int):
        def _hook(module, inputs, _output) -> None:
            if len(inputs) < 2:
                raise RuntimeError("Unexpected LoRA-MoE forward signature while collecting routes")
            x = inputs[0]
            task_id = int(inputs[1])
            seq_len = int(x.shape[1])
            self._accumulate(
                ext_block=ext_block,
                task_id=task_id,
                seq_len=seq_len,
                expert_ids=module.shared_batch_expert,
                batch_index=module.shared_batch_index,
                out=self.shared_counts[ext_block],
                num_experts=self.num_shared,
            )
            self._accumulate(
                ext_block=ext_block,
                task_id=task_id,
                seq_len=seq_len,
                expert_ids=module.private_batch_expert,
                batch_index=module.private_batch_index,
                out=self.private_counts[ext_block],
                num_experts=self.num_private,
            )

        return _hook

    def _accumulate(
        self,
        *,
        ext_block: int,
        task_id: int,
        seq_len: int,
        expert_ids: torch.Tensor,
        batch_index: torch.Tensor,
        out: torch.Tensor,
        num_experts: int,
    ) -> None:
        if task_id < 0 or task_id >= len(self.task_names):
            raise ValueError(f"Unexpected task_id={task_id} while collecting routing for block {ext_block}")
        token_pos = torch.remainder(batch_index, seq_len)
        keep = token_pos >= self.n_prefix
        if not torch.any(keep):
            return
        selected = expert_ids[keep].detach().to(device="cpu", dtype=torch.long)
        bincount = torch.bincount(selected, minlength=int(num_experts))
        out[task_id] += bincount[: int(num_experts)]


def _write_matrix_csv(path: Path, *, row_labels: Sequence[str], col_labels: Sequence[str], matrix: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task", *col_labels])
        for row_label, row in zip(row_labels, matrix.tolist()):
            formatted = []
            for value in row:
                if isinstance(value, float):
                    formatted.append(f"{value:.8f}")
                else:
                    formatted.append(str(int(value)))
            writer.writerow([row_label, *formatted])


def _render_heatmap(
    *,
    block_external: int,
    block_internal: int,
    shared_probs: torch.Tensor,
    private_probs: torch.Tensor,
    row_labels: Sequence[str],
    shared_labels: Sequence[str],
    private_labels: Sequence[str],
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required to render router heatmaps") from exc

    shared = shared_probs.detach().cpu()
    private = private_probs.detach().cpu()

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(max(8.0, 1.1 * len(shared_labels) + 4.0), 4.8),
        gridspec_kw={"width_ratios": [max(len(shared_labels), 1), max(len(private_labels), 1)]},
    )
    cmap = "YlOrRd"

    for ax, data, title, col_labels in (
        (axes[0], shared, "Shared Experts", shared_labels),
        (axes[1], private, "Private Experts (task-local)", private_labels),
    ):
        im = ax.imshow(data.numpy(), cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_title(title)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_xlabel("Experts")
        ax.set_ylabel("Tasks")
        ax.set_xticks([x - 0.5 for x in range(1, len(col_labels))], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, len(row_labels))], minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row_idx in range(data.shape[0]):
            for col_idx in range(data.shape[1]):
                value = float(data[row_idx, col_idx].item())
                color = "white" if value >= 0.5 else "black"
                ax.text(col_idx, row_idx, f"{value:.3f}", ha="center", va="center", color=color, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Block {block_external:02d} routing count after pool-wise softmax (internal block {block_internal})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _summarize_task_forward(stats: TaskRunStats, batch_tensor: torch.Tensor, *, patch_size: Sequence[int]) -> None:
    ph, pw = int(patch_size[0]), int(patch_size[1])
    batch_size = int(batch_tensor.shape[0])
    height = int(batch_tensor.shape[-2])
    width = int(batch_tensor.shape[-1])
    stats.batches += 1
    stats.images += batch_size
    stats.patch_tokens += batch_size * (height // ph) * (width // pw)


def _run_detection(model, *, device: torch.device, loader: DataLoader, max_batches: int, task_stats: TaskRunStats) -> None:
    for batch_idx, (images, _targets) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        task_stats.batches += 1
        for image in images:
            image = image.to(device, non_blocking=True)
            image_list, _ = model.detector.transform([image], None)
            ph, pw = int(model.shared.patch_size[0]), int(model.shared.patch_size[1])
            height = int(image_list.tensors.shape[-2])
            width = int(image_list.tensors.shape[-1])
            task_stats.images += 1
            task_stats.patch_tokens += (height // ph) * (width // pw)
            model.shared.forward_features(image_list.tensors, task_id=model.TASK_ID_DET)


def _run_segmentation(model, *, device: torch.device, loader: DataLoader, max_batches: int, task_stats: TaskRunStats) -> None:
    for batch_idx, (images, _masks) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        _summarize_task_forward(task_stats, images, patch_size=model.shared.patch_size)
        model.shared.forward_features(images, task_id=model.TASK_ID_SEG)


def _run_counting(model, *, device: torch.device, loader: DataLoader, max_batches: int, task_stats: TaskRunStats) -> None:
    for batch_idx, (images, _density) in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        _summarize_task_forward(task_stats, images, patch_size=model.shared.patch_size)
        model.shared.forward_features(images, task_id=model.TASK_ID_CNT)


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    output_dir = _default_output_dir(args.checkpoint, args.output_dir)
    requested_blocks = _parse_blocks(args.blocks)

    model, meta, _load_summary = _build_and_load_model(args)
    model.eval()

    if not bool(meta.get("use_lora_moe", False)):
        raise SystemExit("Checkpoint does not contain LoRA-MoE routing weights; cannot visualize routers")
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

    num_shared = int(meta["moe_cfg"]["num_experts_shared"])
    num_private = int(meta["moe_cfg"]["num_experts_private"])
    n_prefix = 1 + int(getattr(model.shared.backbone, "n_storage_tokens", 0))

    shared_labels = [f"S{i}" for i in range(num_shared)]
    private_labels = [f"P{i}" for i in range(num_private)]
    task_stats = {task: TaskRunStats() for task in TASK_NAMES}

    det_loader = _build_det_loader(args)
    seg_loader = _build_seg_loader(args)
    cnt_num_classes = int(args.cnt_num_classes) if args.cnt_num_classes is not None else int(meta["cnt_num_classes"])
    cnt_loader = _build_cnt_loader(args, num_classes=cnt_num_classes)

    collector = RoutingCollector(
        task_names=TASK_NAMES,
        block_map=block_map,
        num_shared=num_shared,
        num_private=num_private,
        n_prefix=n_prefix,
    )
    collector.attach(model)

    max_batches = int(args.max_batches_per_task)
    with torch.no_grad():
        try:
            _run_detection(model, device=device, loader=det_loader, max_batches=max_batches, task_stats=task_stats["det"])
            _run_segmentation(model, device=device, loader=seg_loader, max_batches=max_batches, task_stats=task_stats["seg"])
            _run_counting(model, device=device, loader=cnt_loader, max_batches=max_batches, task_stats=task_stats["cnt"])
        finally:
            collector.remove()

    for task_name, stats in task_stats.items():
        if stats.images <= 0:
            raise SystemExit(f"No images were processed for task '{task_name}'. Check dataset paths or batch settings.")

    block_outputs: Dict[str, Dict[str, object]] = {}
    for ext_block, internal_block in block_map.items():
        shared_counts = collector.shared_counts[ext_block]
        private_counts = collector.private_counts[ext_block]
        shared_probs = torch.softmax(shared_counts.to(torch.float32), dim=1)
        private_probs = torch.softmax(private_counts.to(torch.float32), dim=1)

        block_tag = f"block_{ext_block:02d}"
        shared_counts_path = output_dir / f"{block_tag}_shared_counts.csv"
        shared_softmax_path = output_dir / f"{block_tag}_shared_softmax.csv"
        private_counts_path = output_dir / f"{block_tag}_private_counts.csv"
        private_softmax_path = output_dir / f"{block_tag}_private_softmax.csv"
        heatmap_path = output_dir / f"{block_tag}_router_heatmap.png"

        _write_matrix_csv(shared_counts_path, row_labels=TASK_NAMES, col_labels=shared_labels, matrix=shared_counts)
        _write_matrix_csv(shared_softmax_path, row_labels=TASK_NAMES, col_labels=shared_labels, matrix=shared_probs)
        _write_matrix_csv(private_counts_path, row_labels=TASK_NAMES, col_labels=private_labels, matrix=private_counts)
        _write_matrix_csv(private_softmax_path, row_labels=TASK_NAMES, col_labels=private_labels, matrix=private_probs)
        _render_heatmap(
            block_external=ext_block,
            block_internal=internal_block,
            shared_probs=shared_probs,
            private_probs=private_probs,
            row_labels=TASK_NAMES,
            shared_labels=shared_labels,
            private_labels=private_labels,
            output_path=heatmap_path,
        )

        block_outputs[str(ext_block)] = {
            "internal_block": int(internal_block),
            "shared_counts_csv": str(shared_counts_path.resolve()),
            "shared_softmax_csv": str(shared_softmax_path.resolve()),
            "private_counts_csv": str(private_counts_path.resolve()),
            "private_softmax_csv": str(private_softmax_path.resolve()),
            "heatmap_png": str(heatmap_path.resolve()),
        }

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "output_dir": str(output_dir.resolve()),
        "routing_statistic": "top-k routing count",
        "normalization": {
            "shared": "row-wise softmax over shared experts",
            "private": "row-wise softmax over private experts",
        },
        "task_names": list(TASK_NAMES),
        "expert_labels": {
            "shared": shared_labels,
            "private": private_labels,
        },
        "private_label_semantics": "P0..Pk denote each task's own private experts within that row, not a shared cross-task index space.",
        "block_mapping": {str(ext): int(internal) for ext, internal in block_map.items()},
        "max_batches_per_task": int(max_batches),
        "task_stats": {
            task: {
                "batches": int(stats.batches),
                "images": int(stats.images),
                "patch_tokens": int(stats.patch_tokens),
            }
            for task, stats in task_stats.items()
        },
        "outputs": block_outputs,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[router_vis] checkpoint: {Path(args.checkpoint).resolve()}")
    print(f"[router_vis] output_dir: {output_dir.resolve()}")
    for ext_block, paths in block_outputs.items():
        print(f"[router_vis] block {ext_block}: {paths['heatmap_png']}")
    print(f"[router_vis] summary: {summary_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

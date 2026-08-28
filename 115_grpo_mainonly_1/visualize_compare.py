from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
OURS_ROOT = Path(__file__).resolve().parents[1]
THIS_DIR = Path(__file__).resolve().parent

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_MODEL_NAME = "dinov3_vitl16"
DEFAULT_IMAGE_SIZE = 448
DEFAULT_SAMPLES_PER_TASK = 8
DEFAULT_TILE_SIZE = 320

DEFAULT_DET_DATA_ROOT = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco"
DEFAULT_DET_ANN_FILE = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_test.json"
DEFAULT_DET_IMG_DIR = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/images/test"
DEFAULT_SEG_DATA_DIR = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/test"
DEFAULT_CNT_DATA_ROOT = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA"
DEFAULT_CNT_TEST_DIR = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA/test_data_class8"

_OURS_EVAL_MODULE: Any | None = None
_LORA_EVAL_MODULE: Any | None = None


def _ensure_import_paths() -> None:
    for path in (REPO_ROOT, REPO_ROOT / "object_detection", OURS_ROOT):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _ours_eval_module():
    global _OURS_EVAL_MODULE
    if _OURS_EVAL_MODULE is None:
        _OURS_EVAL_MODULE = _load_module_from_path(
            "_visualize_compare_ours_eval",
            THIS_DIR / "eval.py",
        )
    return _OURS_EVAL_MODULE


def _lora_eval_module():
    global _LORA_EVAL_MODULE
    if _LORA_EVAL_MODULE is None:
        _LORA_EVAL_MODULE = _load_module_from_path(
            "_visualize_compare_lora_eval",
            OURS_ROOT / "lora_multitask" / "eval.py",
        )
    return _LORA_EVAL_MODULE


def _parse_tasks(text: str) -> list[str]:
    tasks = [item.strip().lower() for item in (text or "").split(",") if item.strip()]
    valid = {"det", "seg", "cnt"}
    invalid = [task for task in tasks if task not in valid]
    if invalid:
        raise SystemExit(f"Invalid task(s): {invalid}; valid values are det, seg, cnt")
    out: list[str] = []
    for task in tasks:
        if task not in out:
            out.append(task)
    if not out:
        raise SystemExit("No tasks selected")
    return out


def _slugify(text: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text.strip())
    safe = safe.strip("_")
    return safe or "sample"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return text[: limit - 3] + "..."


def _default_output_dir(ours_ckpt: Path, lora_ckpt: Path) -> Path:
    ours_tag = _slugify(ours_ckpt.stem)
    lora_tag = _slugify(lora_ckpt.stem)
    return ours_ckpt.resolve().parent / f"vis_compare_{ours_tag}__vs__{lora_tag}"


def _relative_to_output(path: Path, output_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(output_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def _jsonify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, torch.Tensor):
        return _jsonify(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _measure_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image_cpu = image.detach().cpu().float().clamp(0.0, 1.0)
    array = (image_cpu.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def _denormalize_to_pil(image: torch.Tensor, mean: tuple[float, float, float], std: tuple[float, float, float]) -> Image.Image:
    mean_t = torch.tensor(mean, dtype=image.dtype, device=image.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype, device=image.device).view(3, 1, 1)
    image_cpu = image.detach().cpu().float()
    image_cpu = image_cpu * std_t.cpu() + mean_t.cpu()
    return _tensor_to_pil(image_cpu)


def _fit_image(image: Image.Image, width: int, height: int, background: tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    canvas = Image.new("RGB", (width, height), background)
    if image.width == 0 or image.height == 0:
        return canvas
    scale = min(width / image.width, height / image.height)
    new_w = max(1, int(round(image.width * scale)))
    new_h = max(1, int(round(image.height * scale)))
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    offset = ((width - new_w) // 2, (height - new_h) // 2)
    canvas.paste(resized, offset)
    return canvas


def _palette_color(index: int) -> tuple[int, int, int]:
    colors = [
        (230, 25, 75),
        (60, 180, 75),
        (255, 225, 25),
        (0, 130, 200),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
        (240, 50, 230),
        (210, 245, 60),
        (250, 190, 212),
        (0, 128, 128),
        (220, 190, 255),
        (170, 110, 40),
        (255, 250, 200),
        (128, 0, 0),
        (170, 255, 195),
    ]
    if index < len(colors):
        return colors[index]
    return ((index * 67) % 256, (index * 113) % 256, (index * 197) % 256)


def _resolve_coco_image_path(img_dir: Path, file_name: str) -> Path:
    file_path = Path(file_name)
    if file_path.is_absolute():
        return file_path
    candidate1 = img_dir / file_path
    candidate2 = img_dir.parent / file_path
    candidate3 = img_dir.parent.parent / file_path
    if candidate1.exists():
        return candidate1
    if candidate2.exists():
        return candidate2
    if candidate3.exists():
        return candidate3
    return candidate1


def _filter_detection_output(output: dict[str, torch.Tensor], score_thr: float) -> dict[str, list[Any]]:
    boxes = output.get("boxes", torch.zeros((0, 4), dtype=torch.float32)).detach().cpu()
    labels = output.get("labels", torch.zeros((0,), dtype=torch.int64)).detach().cpu()
    scores = output.get("scores", torch.zeros((boxes.shape[0],), dtype=torch.float32)).detach().cpu()
    keep = scores >= float(score_thr)
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    return {
        "boxes": [[float(v) for v in row] for row in boxes.tolist()],
        "labels": [int(v) for v in labels.tolist()],
        "scores": [float(v) for v in scores.tolist()],
    }


def _label_names(labels: list[int], label_to_name: dict[int, str]) -> list[str]:
    return [label_to_name.get(int(label), str(int(label))) for label in labels]


def _draw_boxes(
    image: Image.Image,
    boxes: list[list[float]],
    labels: list[int],
    label_to_name: dict[int, str],
    scores: list[float] | None = None,
) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    line_width = max(2, int(round(min(canvas.size) / 180.0)))

    for idx, box in enumerate(boxes):
        color = _palette_color(int(labels[idx]))
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
        label = label_to_name.get(int(labels[idx]), str(int(labels[idx])))
        if scores is not None:
            label = f"{label} {scores[idx]:.2f}"
        text_w, text_h = _measure_text(draw, label, font)
        text_x = max(0, int(round(x1)))
        text_y = max(0, int(round(y1)) - text_h - 4)
        draw.rectangle((text_x, text_y, text_x + text_w + 4, text_y + text_h + 4), fill=color)
        draw.text((text_x + 2, text_y + 2), label, fill=(255, 255, 255), font=font)
    return canvas


def _overlay_segmentation_mask(
    image: Image.Image,
    mask: np.ndarray,
    num_classes: int,
    alpha: float = 0.45,
) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    overlay = base.copy()
    valid = np.zeros(mask.shape, dtype=bool)
    for class_id in range(1, int(num_classes)):
        class_mask = mask == class_id
        if not np.any(class_mask):
            continue
        overlay[class_mask] = np.asarray(_palette_color(class_id), dtype=np.float32)
        valid |= class_mask
    blended = base.copy()
    blended[valid] = (1.0 - alpha) * base[valid] + alpha * overlay[valid]
    return Image.fromarray(np.clip(blended, 0.0, 255.0).astype(np.uint8))


def _mask_to_png(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(mask.astype(np.uint8), mode="L")


def _heatmap_rgb(norm: np.ndarray) -> np.ndarray:
    norm = np.clip(norm.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * norm - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * norm - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * norm - 1.0), 0.0, 1.0)
    heat = np.stack([r, g, b], axis=-1)
    return (heat * 255.0).round().astype(np.uint8)


def _render_count_tile(image: Image.Image, density: np.ndarray, total_count: float, vmax: float, tile_size: int) -> Image.Image:
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    total_density = density.sum(axis=0).astype(np.float32)
    if vmax <= 0.0:
        norm = np.zeros_like(total_density, dtype=np.float32)
    else:
        norm = np.clip(total_density / float(vmax), 0.0, 1.0)
    heat = _heatmap_rgb(norm).astype(np.float32)
    alpha_map = 0.7 * norm[..., None]
    blended = base * (1.0 - alpha_map) + heat * alpha_map
    overlay = Image.fromarray(np.clip(blended, 0.0, 255.0).astype(np.uint8))
    tile = _fit_image(overlay, tile_size, tile_size)

    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    label = f"total={total_count:.1f}"
    text_w, text_h = _measure_text(draw, label, font)
    pad = 4
    box = (0, tile_size - text_h - pad * 2 - 2, text_w + pad * 2 + 2, tile_size)
    draw.rectangle(box, fill=(30, 30, 30))
    draw.text((pad + 1, tile_size - text_h - pad - 1), label, fill=(255, 255, 255), font=font)
    return tile


def _assemble_grid(task_title: str, row_labels: list[str], col_labels: list[str], tiles: list[list[Image.Image]]) -> Image.Image:
    if not tiles or not tiles[0]:
        raise ValueError(f"No tiles provided for {task_title}")
    cell_w = max(img.width for row in tiles for img in row)
    cell_h = max(img.height for row in tiles for img in row)
    rows = len(tiles)
    cols = len(tiles[0])
    margin = 16
    left_w = 72
    gap_x = 10
    gap_y = 10
    title_h = 34
    header_h = 34

    width = margin * 2 + left_w + cols * cell_w + max(cols - 1, 0) * gap_x
    height = margin * 2 + title_h + header_h + rows * cell_h + max(rows - 1, 0) * gap_y
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text((margin, margin), task_title, fill=(0, 0, 0), font=font)
    header_y = margin + title_h

    for col_idx, label in enumerate(col_labels):
        text = _truncate(label, 18)
        x = margin + left_w + col_idx * (cell_w + gap_x)
        draw.text((x, header_y), text, fill=(0, 0, 0), font=font)

    grid_y0 = header_y + header_h
    for row_idx, row_label in enumerate(row_labels):
        row_y = grid_y0 + row_idx * (cell_h + gap_y)
        label_w, label_h = _measure_text(draw, row_label, font)
        draw.text((margin + max(0, left_w - label_w - 10), row_y + max(0, (cell_h - label_h) // 2)), row_label, fill=(0, 0, 0), font=font)
        for col_idx, tile in enumerate(tiles[row_idx]):
            x = margin + left_w + col_idx * (cell_w + gap_x)
            if tile.size != (cell_w, cell_h):
                fitted = _fit_image(tile, cell_w, cell_h)
            else:
                fitted = tile
            canvas.paste(fitted, (x, row_y))
    return canvas


def _build_seg_transform():
    def _transform(img, mask):
        image = TF.to_tensor(img)
        image = TF.normalize(image, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        mask_tensor = torch.as_tensor(TF.pil_to_tensor(mask), dtype=torch.long).squeeze(0)
        return image, mask_tensor

    return _transform


class _CountNormalizeTransform:
    def __call__(self, img, density):
        if not torch.is_tensor(img):
            img = TF.to_tensor(img)
        img = TF.normalize(img, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        return img, density


def _collect_detection_samples(args) -> tuple[list[dict[str, Any]], dict[int, str]]:
    from object_detection.dataset import CocoDetectionDataset

    def _transform(img, target):
        return TF.to_tensor(img), target

    dataset = CocoDetectionDataset(
        ann_file=str(Path(args.det_ann_file)),
        img_dir=str(Path(args.det_img_dir)),
        transform=_transform,
    )

    label_to_name: dict[int, str] = {}
    for category in dataset.categories:
        cat_id = int(category["id"])
        label = int(dataset.cat_id_to_label[cat_id])
        label_to_name[label] = str(category.get("name", cat_id))

    samples: list[dict[str, Any]] = []
    for idx in range(min(DEFAULT_SAMPLES_PER_TASK, len(dataset))):
        image_tensor, target = dataset[idx]
        image_id = int(target["image_id"].item())
        info = dataset.image_id_to_info[image_id]
        image_path = _resolve_coco_image_path(Path(args.det_img_dir), str(info["file_name"]))
        sample_id = f"{idx:02d}_{_slugify(Path(str(info['file_name'])).stem)}"
        gt_labels = [int(v) for v in target["labels"].tolist()]
        samples.append(
            {
                "index": idx,
                "sample_id": sample_id,
                "display_name": Path(str(info["file_name"])).name,
                "image_id": image_id,
                "image_path": image_path,
                "image_tensor": image_tensor.detach().cpu(),
                "image_pil": Image.open(image_path).convert("RGB"),
                "gt_boxes": [[float(v) for v in row] for row in target["boxes"].tolist()],
                "gt_labels": gt_labels,
                "gt_label_names": _label_names(gt_labels, label_to_name),
            }
        )
    return samples, label_to_name


def _collect_segmentation_samples(args) -> list[dict[str, Any]]:
    from segmentation.dataset import SegmentationDataset

    dataset = SegmentationDataset(
        root=str(Path(args.seg_data_dir)),
        transform=_build_seg_transform(),
        image_size=int(args.image_size),
    )

    samples: list[dict[str, Any]] = []
    for idx in range(min(DEFAULT_SAMPLES_PER_TASK, len(dataset))):
        image_tensor, mask = dataset[idx]
        image_path = Path(dataset.image_paths[idx])
        mask_path = Path(dataset.mask_paths[idx])
        sample_id = f"{idx:02d}_{_slugify(image_path.stem)}"
        samples.append(
            {
                "index": idx,
                "sample_id": sample_id,
                "display_name": image_path.name,
                "image_path": image_path,
                "mask_path": mask_path,
                "image_tensor": image_tensor.detach().cpu(),
                "display_image": _denormalize_to_pil(image_tensor, IMAGENET_MEAN, IMAGENET_STD),
                "gt_mask": mask.detach().cpu().numpy().astype(np.uint8),
            }
        )
    return samples


def _collect_counting_samples(args, num_classes: int) -> list[dict[str, Any]]:
    from counting.dataset import DSACADensityH5Dataset

    dataset = DSACADensityH5Dataset(
        split_root=str(Path(args.cnt_test_dir)),
        num_classes=int(num_classes),
        transform=_CountNormalizeTransform(),
        image_size=int(args.image_size),
        keep_aspect=bool(args.cnt_keep_aspect),
    )

    samples: list[dict[str, Any]] = []
    for idx in range(min(DEFAULT_SAMPLES_PER_TASK, len(dataset))):
        image_tensor, density = dataset[idx]
        image_path, density_path = dataset.samples[idx]
        sample_id = f"{idx:02d}_{_slugify(Path(image_path).stem)}"
        gt_density = density.detach().cpu().numpy().astype(np.float32)
        gt_counts = density.detach().cpu().reshape(int(num_classes), -1).sum(dim=1).tolist()
        samples.append(
            {
                "index": idx,
                "sample_id": sample_id,
                "display_name": Path(image_path).name,
                "image_path": Path(image_path),
                "density_path": Path(density_path),
                "image_tensor": image_tensor.detach().cpu(),
                "display_image": _denormalize_to_pil(image_tensor, IMAGENET_MEAN, IMAGENET_STD),
                "gt_density": gt_density,
                "gt_counts": [float(v) for v in gt_counts],
            }
        )
    return samples


def _predict_detection(model, device: torch.device, samples: list[dict[str, Any]], score_thr: float) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    model.eval()
    with torch.no_grad():
        for sample in samples:
            image = sample["image_tensor"].to(device)
            prediction = model.forward_det([image])[0]
            outputs[sample["sample_id"]] = _filter_detection_output(prediction, score_thr)
    return outputs


def _predict_segmentation(model, device: torch.device, samples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    model.eval()
    with torch.no_grad():
        for sample in samples:
            image = sample["image_tensor"].unsqueeze(0).to(device)
            logits = model.forward_seg(image)
            pred = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
            outputs[sample["sample_id"]] = pred
    return outputs


def _predict_counting(model, device: torch.device, samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    model.eval()
    with torch.no_grad():
        for sample in samples:
            image = sample["image_tensor"].unsqueeze(0).to(device)
            pred_density, pred_counts = model.forward_cnt(image)
            density = pred_density.squeeze(0).detach().cpu().numpy().astype(np.float32)
            counts = pred_counts.squeeze(0).detach().cpu().tolist()
            outputs[sample["sample_id"]] = {
                "density": density,
                "counts": [float(v) for v in counts],
            }
    return outputs


def _release_model(model: Any, device: torch.device) -> None:
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _load_lora_model(args):
    module = _lora_eval_module()
    model, meta, device = module._build_full_multitask_model_from_ckpt(
        str(Path(args.lora_ckpt).resolve()),
        device=args.device,
        model_name=args.model_name,
        image_size=args.image_size,
    )
    return model, meta, device


def _load_ours_model(args):
    module = _ours_eval_module()
    model, meta, _shared_state, _det_state, _seg_state, _cnt_state, device = module._build_full_multitask_model_from_ckpt(
        str(Path(args.ours_ckpt).resolve()),
        device=args.device,
        model_name=args.model_name,
        image_size=args.image_size,
    )
    return model, meta, device


def _write_detection_outputs(
    output_dir: Path,
    samples: list[dict[str, Any]],
    label_to_name: dict[int, str],
    lora_outputs: dict[str, dict[str, Any]],
    ours_outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    task_dir = output_dir / "det"
    task_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("gt", "lora", "ours"):
        (task_dir / subdir).mkdir(exist_ok=True)

    tiles: list[list[Image.Image]] = [[], [], []]
    manifest_samples: list[dict[str, Any]] = []

    for sample in samples:
        sample_id = sample["sample_id"]
        gt_tile = _fit_image(
            _draw_boxes(sample["image_pil"], sample["gt_boxes"], sample["gt_labels"], label_to_name),
            DEFAULT_TILE_SIZE,
            DEFAULT_TILE_SIZE,
        )
        lora_out = lora_outputs[sample_id]
        lora_tile = _fit_image(
            _draw_boxes(sample["image_pil"], lora_out["boxes"], lora_out["labels"], label_to_name, lora_out["scores"]),
            DEFAULT_TILE_SIZE,
            DEFAULT_TILE_SIZE,
        )
        ours_out = ours_outputs[sample_id]
        ours_tile = _fit_image(
            _draw_boxes(sample["image_pil"], ours_out["boxes"], ours_out["labels"], label_to_name, ours_out["scores"]),
            DEFAULT_TILE_SIZE,
            DEFAULT_TILE_SIZE,
        )

        gt_path = task_dir / "gt" / f"{sample_id}.png"
        lora_path = task_dir / "lora" / f"{sample_id}.png"
        ours_path = task_dir / "ours" / f"{sample_id}.png"
        gt_tile.save(gt_path)
        lora_tile.save(lora_path)
        ours_tile.save(ours_path)

        tiles[0].append(gt_tile)
        tiles[1].append(lora_tile)
        tiles[2].append(ours_tile)

        manifest_samples.append(
            {
                "sample_id": sample_id,
                "display_name": sample["display_name"],
                "image_id": int(sample["image_id"]),
                "image_path": _relative_to_output(sample["image_path"], output_dir),
                "gt": {
                    "tile": _relative_to_output(gt_path, output_dir),
                    "boxes": sample["gt_boxes"],
                    "labels": sample["gt_labels"],
                    "label_names": sample["gt_label_names"],
                },
                "lora": {
                    "tile": _relative_to_output(lora_path, output_dir),
                    "boxes": lora_out["boxes"],
                    "labels": lora_out["labels"],
                    "label_names": _label_names(lora_out["labels"], label_to_name),
                    "scores": lora_out["scores"],
                },
                "ours": {
                    "tile": _relative_to_output(ours_path, output_dir),
                    "boxes": ours_out["boxes"],
                    "labels": ours_out["labels"],
                    "label_names": _label_names(ours_out["labels"], label_to_name),
                    "scores": ours_out["scores"],
                },
            }
        )

    compare = _assemble_grid(
        task_title="Detection Comparison",
        row_labels=["GT", "LoRA", "Ours"],
        col_labels=[sample["display_name"] for sample in samples],
        tiles=tiles,
    )
    compare_path = task_dir / "compare.png"
    compare.save(compare_path)
    return {
        "task": "detection",
        "compare_figure": _relative_to_output(compare_path, output_dir),
        "samples": manifest_samples,
    }


def _write_segmentation_outputs(
    output_dir: Path,
    samples: list[dict[str, Any]],
    num_classes: int,
    lora_outputs: dict[str, np.ndarray],
    ours_outputs: dict[str, np.ndarray],
) -> dict[str, Any]:
    task_dir = output_dir / "seg"
    task_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("gt", "lora", "ours", "raw_masks"):
        (task_dir / subdir).mkdir(exist_ok=True)
    for subdir in ("gt", "lora", "ours"):
        (task_dir / "raw_masks" / subdir).mkdir(parents=True, exist_ok=True)

    tiles: list[list[Image.Image]] = [[], [], []]
    manifest_samples: list[dict[str, Any]] = []

    for sample in samples:
        sample_id = sample["sample_id"]
        gt_mask = sample["gt_mask"]
        lora_mask = lora_outputs[sample_id]
        ours_mask = ours_outputs[sample_id]

        gt_tile = _fit_image(
            _overlay_segmentation_mask(sample["display_image"], gt_mask, num_classes),
            DEFAULT_TILE_SIZE,
            DEFAULT_TILE_SIZE,
        )
        lora_tile = _fit_image(
            _overlay_segmentation_mask(sample["display_image"], lora_mask, num_classes),
            DEFAULT_TILE_SIZE,
            DEFAULT_TILE_SIZE,
        )
        ours_tile = _fit_image(
            _overlay_segmentation_mask(sample["display_image"], ours_mask, num_classes),
            DEFAULT_TILE_SIZE,
            DEFAULT_TILE_SIZE,
        )

        gt_path = task_dir / "gt" / f"{sample_id}.png"
        lora_path = task_dir / "lora" / f"{sample_id}.png"
        ours_path = task_dir / "ours" / f"{sample_id}.png"
        gt_tile.save(gt_path)
        lora_tile.save(lora_path)
        ours_tile.save(ours_path)

        gt_mask_path = task_dir / "raw_masks" / "gt" / f"{sample_id}.png"
        lora_mask_path = task_dir / "raw_masks" / "lora" / f"{sample_id}.png"
        ours_mask_path = task_dir / "raw_masks" / "ours" / f"{sample_id}.png"
        _mask_to_png(gt_mask).save(gt_mask_path)
        _mask_to_png(lora_mask).save(lora_mask_path)
        _mask_to_png(ours_mask).save(ours_mask_path)

        tiles[0].append(gt_tile)
        tiles[1].append(lora_tile)
        tiles[2].append(ours_tile)

        manifest_samples.append(
            {
                "sample_id": sample_id,
                "display_name": sample["display_name"],
                "image_path": _relative_to_output(sample["image_path"], output_dir),
                "mask_path": _relative_to_output(sample["mask_path"], output_dir),
                "gt": {
                    "tile": _relative_to_output(gt_path, output_dir),
                    "mask": _relative_to_output(gt_mask_path, output_dir),
                },
                "lora": {
                    "tile": _relative_to_output(lora_path, output_dir),
                    "mask": _relative_to_output(lora_mask_path, output_dir),
                },
                "ours": {
                    "tile": _relative_to_output(ours_path, output_dir),
                    "mask": _relative_to_output(ours_mask_path, output_dir),
                },
            }
        )

    compare = _assemble_grid(
        task_title="Segmentation Comparison",
        row_labels=["GT", "LoRA", "Ours"],
        col_labels=[sample["display_name"] for sample in samples],
        tiles=tiles,
    )
    compare_path = task_dir / "compare.png"
    compare.save(compare_path)
    return {
        "task": "segmentation",
        "compare_figure": _relative_to_output(compare_path, output_dir),
        "samples": manifest_samples,
    }


def _write_counting_outputs(
    output_dir: Path,
    samples: list[dict[str, Any]],
    lora_outputs: dict[str, dict[str, Any]],
    ours_outputs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    task_dir = output_dir / "cnt"
    task_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("gt", "lora", "ours", "raw_density"):
        (task_dir / subdir).mkdir(exist_ok=True)
    for subdir in ("gt", "lora", "ours"):
        (task_dir / "raw_density" / subdir).mkdir(parents=True, exist_ok=True)

    tiles: list[list[Image.Image]] = [[], [], []]
    manifest_samples: list[dict[str, Any]] = []

    for sample in samples:
        sample_id = sample["sample_id"]
        gt_density = sample["gt_density"]
        lora_density = lora_outputs[sample_id]["density"]
        ours_density = ours_outputs[sample_id]["density"]
        gt_total_density = gt_density.sum(axis=0)
        lora_total_density = lora_density.sum(axis=0)
        ours_total_density = ours_density.sum(axis=0)
        heatmap_vmax = float(
            max(
                float(np.max(gt_total_density)) if gt_total_density.size else 0.0,
                float(np.max(lora_total_density)) if lora_total_density.size else 0.0,
                float(np.max(ours_total_density)) if ours_total_density.size else 0.0,
            )
        )

        gt_counts = sample["gt_counts"]
        lora_counts = lora_outputs[sample_id]["counts"]
        ours_counts = ours_outputs[sample_id]["counts"]

        gt_tile = _render_count_tile(sample["display_image"], gt_density, sum(gt_counts), heatmap_vmax, DEFAULT_TILE_SIZE)
        lora_tile = _render_count_tile(sample["display_image"], lora_density, sum(lora_counts), heatmap_vmax, DEFAULT_TILE_SIZE)
        ours_tile = _render_count_tile(sample["display_image"], ours_density, sum(ours_counts), heatmap_vmax, DEFAULT_TILE_SIZE)

        gt_path = task_dir / "gt" / f"{sample_id}.png"
        lora_path = task_dir / "lora" / f"{sample_id}.png"
        ours_path = task_dir / "ours" / f"{sample_id}.png"
        gt_tile.save(gt_path)
        lora_tile.save(lora_path)
        ours_tile.save(ours_path)

        gt_density_path = task_dir / "raw_density" / "gt" / f"{sample_id}.npy"
        lora_density_path = task_dir / "raw_density" / "lora" / f"{sample_id}.npy"
        ours_density_path = task_dir / "raw_density" / "ours" / f"{sample_id}.npy"
        np.save(gt_density_path, gt_density.astype(np.float32))
        np.save(lora_density_path, lora_density.astype(np.float32))
        np.save(ours_density_path, ours_density.astype(np.float32))

        tiles[0].append(gt_tile)
        tiles[1].append(lora_tile)
        tiles[2].append(ours_tile)

        manifest_samples.append(
            {
                "sample_id": sample_id,
                "display_name": sample["display_name"],
                "image_path": _relative_to_output(sample["image_path"], output_dir),
                "density_path": _relative_to_output(sample["density_path"], output_dir),
                "heatmap_vmax": heatmap_vmax,
                "gt": {
                    "tile": _relative_to_output(gt_path, output_dir),
                    "density": _relative_to_output(gt_density_path, output_dir),
                    "counts": gt_counts,
                    "total_count": float(sum(gt_counts)),
                },
                "lora": {
                    "tile": _relative_to_output(lora_path, output_dir),
                    "density": _relative_to_output(lora_density_path, output_dir),
                    "counts": lora_counts,
                    "total_count": float(sum(lora_counts)),
                },
                "ours": {
                    "tile": _relative_to_output(ours_path, output_dir),
                    "density": _relative_to_output(ours_density_path, output_dir),
                    "counts": ours_counts,
                    "total_count": float(sum(ours_counts)),
                },
            }
        )

    compare = _assemble_grid(
        task_title="Counting Comparison",
        row_labels=["GT", "LoRA", "Ours"],
        col_labels=[sample["display_name"] for sample in samples],
        tiles=tiles,
    )
    compare_path = task_dir / "compare.png"
    compare.save(compare_path)
    return {
        "task": "counting",
        "compare_figure": _relative_to_output(compare_path, output_dir),
        "samples": manifest_samples,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize LoRA baseline vs Ours on detection/segmentation/counting test sets.")
    parser.add_argument("--lora-ckpt", type=str, required=True, help="checkpoint trained by ours/lora_multitask")
    parser.add_argument("--ours-ckpt", type=str, required=True, help="checkpoint trained by ours/115_grpo_mainonly")
    parser.add_argument("--output-dir", type=str, default=None, help="output directory for figures and manifest")
    parser.add_argument("--tasks", type=str, default="det,seg,cnt", help="comma-separated: det,seg,cnt")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)

    parser.add_argument("--det-data-root", type=str, default=DEFAULT_DET_DATA_ROOT)
    parser.add_argument("--det-ann-file", type=str, default=DEFAULT_DET_ANN_FILE)
    parser.add_argument("--det-img-dir", type=str, default=DEFAULT_DET_IMG_DIR)
    parser.add_argument("--det-score-thr", type=float, default=0.0)

    parser.add_argument("--seg-data-dir", type=str, default=DEFAULT_SEG_DATA_DIR)

    parser.add_argument("--cnt-data-root", type=str, default=DEFAULT_CNT_DATA_ROOT)
    parser.add_argument("--cnt-test-dir", type=str, default=DEFAULT_CNT_TEST_DIR)
    aspect = parser.add_mutually_exclusive_group()
    aspect.add_argument("--cnt-keep-aspect", dest="cnt_keep_aspect", action="store_true")
    aspect.add_argument("--cnt-no-keep-aspect", dest="cnt_keep_aspect", action="store_false")
    parser.set_defaults(cnt_keep_aspect=True)
    return parser.parse_args()


def main() -> int:
    _ensure_import_paths()
    args = parse_args()
    tasks = _parse_tasks(args.tasks)

    lora_ckpt = Path(args.lora_ckpt).resolve()
    ours_ckpt = Path(args.ours_ckpt).resolve()
    if not lora_ckpt.is_file():
        raise SystemExit(f"LoRA checkpoint not found: {lora_ckpt}")
    if not ours_ckpt.is_file():
        raise SystemExit(f"Ours checkpoint not found: {ours_ckpt}")

    output_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir(ours_ckpt, lora_ckpt)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[visualize] output_dir: {output_dir}")
    print("[visualize] loading LoRA model...")
    lora_model, lora_meta, device = _load_lora_model(args)
    lora_device = device
    print(f"[visualize] LoRA model loaded on {lora_device}")

    det_samples: list[dict[str, Any]] = []
    seg_samples: list[dict[str, Any]] = []
    cnt_samples: list[dict[str, Any]] = []
    label_to_name: dict[int, str] = {}

    if "det" in tasks:
        det_samples, label_to_name = _collect_detection_samples(args)
        print(f"[visualize] selected {len(det_samples)} detection samples")
    if "seg" in tasks:
        seg_samples = _collect_segmentation_samples(args)
        print(f"[visualize] selected {len(seg_samples)} segmentation samples")
    if "cnt" in tasks:
        cnt_samples = _collect_counting_samples(args, int(lora_meta["cnt_num_classes"]))
        print(f"[visualize] selected {len(cnt_samples)} counting samples")

    lora_outputs: dict[str, Any] = {}
    if "det" in tasks:
        lora_outputs["det"] = _predict_detection(lora_model, lora_device, det_samples, float(args.det_score_thr))
    if "seg" in tasks:
        lora_outputs["seg"] = _predict_segmentation(lora_model, lora_device, seg_samples)
    if "cnt" in tasks:
        lora_outputs["cnt"] = _predict_counting(lora_model, lora_device, cnt_samples)
    _release_model(lora_model, lora_device)
    lora_model = None

    print("[visualize] loading Ours model...")
    ours_model, ours_meta, ours_device = _load_ours_model(args)
    print(f"[visualize] Ours model loaded on {ours_device}")

    for key in ("det_fg_classes", "seg_num_classes", "cnt_num_classes"):
        if int(lora_meta[key]) != int(ours_meta[key]):
            raise SystemExit(f"Checkpoint mismatch for {key}: LoRA={lora_meta[key]} vs Ours={ours_meta[key]}")

    ours_outputs: dict[str, Any] = {}
    if "det" in tasks:
        ours_outputs["det"] = _predict_detection(ours_model, ours_device, det_samples, float(args.det_score_thr))
    if "seg" in tasks:
        ours_outputs["seg"] = _predict_segmentation(ours_model, ours_device, seg_samples)
    if "cnt" in tasks:
        ours_outputs["cnt"] = _predict_counting(ours_model, ours_device, cnt_samples)
    _release_model(ours_model, ours_device)
    ours_model = None

    manifest: dict[str, Any] = {
        "output_dir": str(output_dir),
        "config": {
            "tasks": tasks,
            "samples_per_task": DEFAULT_SAMPLES_PER_TASK,
            "device": str(args.device),
            "image_size": int(args.image_size),
            "model_name": str(args.model_name),
            "det_score_thr": float(args.det_score_thr),
            "cnt_keep_aspect": bool(args.cnt_keep_aspect),
            "paths": {
                "det_data_root": args.det_data_root,
                "det_ann_file": args.det_ann_file,
                "det_img_dir": args.det_img_dir,
                "seg_data_dir": args.seg_data_dir,
                "cnt_data_root": args.cnt_data_root,
                "cnt_test_dir": args.cnt_test_dir,
            },
        },
        "models": {
            "lora": {
                "checkpoint": str(lora_ckpt),
                "meta": _jsonify(lora_meta),
            },
            "ours": {
                "checkpoint": str(ours_ckpt),
                "meta": _jsonify(ours_meta),
            },
        },
        "tasks": {},
    }

    if "det" in tasks:
        manifest["tasks"]["det"] = _write_detection_outputs(
            output_dir=output_dir,
            samples=det_samples,
            label_to_name=label_to_name,
            lora_outputs=lora_outputs["det"],
            ours_outputs=ours_outputs["det"],
        )
    if "seg" in tasks:
        manifest["tasks"]["seg"] = _write_segmentation_outputs(
            output_dir=output_dir,
            samples=seg_samples,
            num_classes=int(ours_meta["seg_num_classes"]),
            lora_outputs=lora_outputs["seg"],
            ours_outputs=ours_outputs["seg"],
        )
    if "cnt" in tasks:
        manifest["tasks"]["cnt"] = _write_counting_outputs(
            output_dir=output_dir,
            samples=cnt_samples,
            lora_outputs=lora_outputs["cnt"],
            ours_outputs=ours_outputs["cnt"],
        )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(_jsonify(manifest), fh, ensure_ascii=False, indent=2)

    print(f"[visualize] manifest: {manifest_path}")
    for task in tasks:
        compare_path = manifest["tasks"][task]["compare_figure"]
        print(f"[visualize] {task}: {output_dir / compare_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

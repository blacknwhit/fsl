from __future__ import annotations

import argparse
import gc
import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

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
DEFAULT_SELECTION_SCAN_LIMIT = 512
DEFAULT_TILE_SIZE = 320
COUNT_SAMPLE_TILE_SIZE = 180
COUNT_VIS_BACKGROUND_QUANTILE = 0.80
COUNT_VIS_SPATIAL_FLOOR = 0.10
COUNT_VIS_MIN_COUNT = 1.0
COUNT_VIS_GAMMA = 0.45
COUNT_VIS_CONTRAST = 1.35
DET_CLEAR_AP50_GAIN = 0.10
SEG_CLEAR_MIOU_GAIN = 0.05
CNT_CLEAR_MAE_GAIN = 0.75
CNT_CLEAR_REL_GAIN = 0.15

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
    for threshold in np.linspace(0.0, 1.0, 101):
        precision = prec[rec >= threshold].max() if np.any(rec >= threshold) else 0.0
        ap += precision
    return float(ap / 101.0)


def _single_image_ap50(prediction: dict[str, list[Any]], gt_boxes: list[list[float]], gt_labels: list[int], num_classes: int) -> float:
    pred_boxes = np.asarray(prediction.get("boxes", []), dtype=np.float32).reshape(-1, 4)
    pred_labels = np.asarray(prediction.get("labels", []), dtype=np.int64).reshape(-1)
    pred_scores = np.asarray(prediction.get("scores", []), dtype=np.float32).reshape(-1)
    gt_boxes_np = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
    gt_labels_np = np.asarray(gt_labels, dtype=np.int64).reshape(-1)

    ap_list: list[float] = []
    for class_id in range(1, int(num_classes) + 1):
        gt_keep = gt_labels_np == class_id
        num_gt = int(gt_keep.sum())
        if num_gt == 0:
            continue

        cls_gt_boxes = gt_boxes_np[gt_keep]
        pred_keep = pred_labels == class_id
        cls_pred_boxes = pred_boxes[pred_keep]
        cls_pred_scores = pred_scores[pred_keep]
        order = np.argsort(-cls_pred_scores)
        cls_pred_boxes = cls_pred_boxes[order]

        matched = np.zeros(num_gt, dtype=bool)
        tp = np.zeros(len(cls_pred_boxes), dtype=np.float32)
        fp = np.zeros(len(cls_pred_boxes), dtype=np.float32)
        for idx, pred_box in enumerate(cls_pred_boxes):
            ious = _box_iou_np(np.asarray([pred_box], dtype=np.float32), cls_gt_boxes)[0]
            best_idx = int(np.argmax(ious)) if ious.size > 0 else -1
            if best_idx >= 0 and ious[best_idx] >= 0.5 and not matched[best_idx]:
                tp[idx] = 1.0
                matched[best_idx] = True
            else:
                fp[idx] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / max(num_gt, 1)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        ap_list.append(_ap_from_pr(rec, prec))
    return float(np.mean(ap_list)) if ap_list else 0.0


def _single_image_miou(pred_mask: np.ndarray, gt_mask: np.ndarray, num_classes: int) -> float:
    pred = np.asarray(pred_mask)
    gt = np.asarray(gt_mask)
    valid = (gt != 255) & (gt != 11)
    ious: list[float] = []
    for class_id in range(int(num_classes)):
        pred_c = (pred == class_id) & valid
        gt_c = (gt == class_id) & valid
        union = pred_c | gt_c
        if not np.any(union):
            continue
        ious.append(float(np.logical_and(pred_c, gt_c).sum() / max(float(union.sum()), 1.0)))
    return float(np.mean(ious)) if ious else 0.0


def _single_image_count_mae(pred_counts: list[float], gt_counts: list[float]) -> float:
    pred = np.asarray(pred_counts, dtype=np.float32)
    gt = np.asarray(gt_counts, dtype=np.float32)
    return float(np.mean(np.abs(pred - gt))) if pred.size else 0.0


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


def _render_count_class_heatmap(
    image: Image.Image,
    class_density: np.ndarray,
    class_count: float,
    count_vmax: float,
    label: str,
    *,
    suppress_background: bool,
) -> Image.Image:
    saliency = _count_spatial_saliency(class_density, suppress_background=suppress_background)
    if class_count < COUNT_VIS_MIN_COUNT or count_vmax <= 0.0:
        norm = np.zeros_like(saliency, dtype=np.float32)
    else:
        count_scale = min(1.0, max(float(class_count), 0.0) / float(count_vmax))
        spatial_floor = max(0.02, COUNT_VIS_SPATIAL_FLOOR * (1.0 - 0.7 * count_scale))
        saliency = np.where(saliency >= spatial_floor, saliency, 0.0)
        norm = saliency * count_scale
        norm = np.clip(norm * COUNT_VIS_CONTRAST, 0.0, 1.0)
        norm = np.power(norm, COUNT_VIS_GAMMA, dtype=np.float32)
    heat = _heatmap_rgb(norm).astype(np.float32)
    black = np.zeros((*saliency.shape, 3), dtype=np.float32)
    heat_mask = norm > 0.0
    black[heat_mask] = heat[heat_mask]
    out = Image.fromarray(np.clip(black, 0.0, 255.0).astype(np.uint8))

    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    text = f"{label}={class_count:.2f}"
    text_w, text_h = _measure_text(draw, text, font)
    pad = 4
    draw.rectangle((0, 0, text_w + pad * 2 + 2, text_h + pad * 2 + 2), fill=(30, 30, 30))
    draw.text((pad + 1, pad + 1), text, fill=(255, 255, 255), font=font)
    return out


def _count_spatial_saliency(class_density: np.ndarray, *, suppress_background: bool) -> np.ndarray:
    density = np.maximum(np.asarray(class_density, dtype=np.float32), 0.0)
    if density.ndim != 2:
        raise ValueError(f"Expected a 2D class density map, got {density.shape}")
    if suppress_background:
        positive = density[density > 0.0]
        if positive.size:
            background = float(np.quantile(positive, COUNT_VIS_BACKGROUND_QUANTILE))
            density = np.maximum(density - background, 0.0)
    max_value = float(np.max(density)) if density.size else 0.0
    if max_value <= 0.0:
        return np.zeros_like(density, dtype=np.float32)
    return np.clip(density / max_value, 0.0, 1.0).astype(np.float32)


def _draw_metric_badge(tile: Image.Image, text: str) -> Image.Image:
    if not text:
        return tile
    canvas = tile.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    text_w, text_h = _measure_text(draw, text, font)
    pad = 4
    draw.rectangle((0, 0, text_w + pad * 2 + 2, text_h + pad * 2 + 2), fill=(30, 30, 30))
    draw.text((pad + 1, pad + 1), text, fill=(255, 255, 255), font=font)
    return canvas


def _prepare_task_dir(task_dir: Path) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in task_dir.glob("compare*.png"):
        if stale_path.is_file():
            stale_path.unlink()
    for subdir_name in ("original", "gt", "lora", "ours", "sample_compare"):
        subdir = task_dir / subdir_name
        if subdir.is_dir():
            shutil.rmtree(subdir)
        subdir.mkdir(parents=True, exist_ok=True)


def _sample_file_stem(sample: dict[str, Any]) -> str:
    return f"{int(sample['index']):04d}_{_slugify(Path(str(sample['display_name'])).stem)}"


def _save_original_sample_image(sample: dict[str, Any], path: Path) -> None:
    if "image_path" in sample and Path(sample["image_path"]).is_file():
        Image.open(sample["image_path"]).convert("RGB").save(path)
        return
    if "image_pil" in sample:
        sample["image_pil"].save(path)
        return
    sample["display_image"].save(path)


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


def _build_detection_dataset(args):
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
    return dataset, label_to_name


def _iter_detection_samples(args, dataset, label_to_name: dict[int, str]) -> Iterable[dict[str, Any]]:
    for idx in range(len(dataset)):
        image_tensor, target = dataset[idx]
        image_id = int(target["image_id"].item())
        info = dataset.image_id_to_info[image_id]
        image_path = _resolve_coco_image_path(Path(args.det_img_dir), str(info["file_name"]))
        sample_id = f"{idx:02d}_{_slugify(Path(str(info['file_name'])).stem)}"
        gt_labels = [int(v) for v in target["labels"].tolist()]
        yield {
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


def _collect_detection_samples(args) -> tuple[list[dict[str, Any]], dict[int, str]]:
    dataset, label_to_name = _build_detection_dataset(args)
    samples: list[dict[str, Any]] = []
    for sample in _iter_detection_samples(args, dataset, label_to_name):
        samples.append(sample)
    return samples, label_to_name


def _build_segmentation_dataset(args):
    from segmentation.dataset import SegmentationDataset

    return SegmentationDataset(
        root=str(Path(args.seg_data_dir)),
        transform=_build_seg_transform(),
        image_size=int(args.image_size),
    )


def _iter_segmentation_samples(dataset) -> Iterable[dict[str, Any]]:
    for idx in range(len(dataset)):
        image_tensor, mask = dataset[idx]
        image_path = Path(dataset.image_paths[idx])
        mask_path = Path(dataset.mask_paths[idx])
        sample_id = f"{idx:02d}_{_slugify(image_path.stem)}"
        yield {
            "index": idx,
            "sample_id": sample_id,
            "display_name": image_path.name,
            "image_path": image_path,
            "mask_path": mask_path,
            "image_tensor": image_tensor.detach().cpu(),
            "display_image": _denormalize_to_pil(image_tensor, IMAGENET_MEAN, IMAGENET_STD),
            "gt_mask": mask.detach().cpu().numpy().astype(np.uint8),
        }


def _collect_segmentation_samples(args) -> list[dict[str, Any]]:
    dataset = _build_segmentation_dataset(args)
    samples: list[dict[str, Any]] = []
    for sample in _iter_segmentation_samples(dataset):
        samples.append(sample)
    return samples


def _build_counting_dataset(args, num_classes: int):
    from counting.dataset import DSACADensityH5Dataset

    return DSACADensityH5Dataset(
        split_root=str(Path(args.cnt_test_dir)),
        num_classes=int(num_classes),
        transform=_CountNormalizeTransform(),
        image_size=int(args.image_size),
        keep_aspect=bool(args.cnt_keep_aspect),
    )


def _iter_counting_samples(dataset, num_classes: int) -> Iterable[dict[str, Any]]:
    for idx in range(len(dataset)):
        image_tensor, density = dataset[idx]
        image_path, density_path = dataset.samples[idx]
        sample_id = f"{idx:02d}_{_slugify(Path(image_path).stem)}"
        gt_density = density.detach().cpu().numpy().astype(np.float32)
        gt_counts = density.detach().cpu().reshape(int(num_classes), -1).sum(dim=1).tolist()
        yield {
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


def _collect_counting_samples(args, num_classes: int) -> list[dict[str, Any]]:
    dataset = _build_counting_dataset(args, num_classes)
    samples: list[dict[str, Any]] = []
    for sample in _iter_counting_samples(dataset, num_classes):
        samples.append(sample)
    return samples


@torch.no_grad()
def _predict_detection_one(model, device: torch.device, sample: dict[str, Any], score_thr: float) -> dict[str, Any]:
    image = sample["image_tensor"].to(device)
    prediction = model.forward_det([image])[0]
    return _filter_detection_output(prediction, score_thr)


@torch.no_grad()
def _predict_segmentation_one(model, device: torch.device, sample: dict[str, Any]) -> np.ndarray:
    image = sample["image_tensor"].unsqueeze(0).to(device)
    logits = model.forward_seg(image)
    return logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)


@torch.no_grad()
def _predict_counting_one(model, device: torch.device, sample: dict[str, Any]) -> dict[str, Any]:
    image = sample["image_tensor"].unsqueeze(0).to(device)
    pred_density, pred_counts = model.forward_cnt(image)
    density = pred_density.squeeze(0).detach().cpu().numpy().astype(np.float32)
    counts = pred_counts.squeeze(0).detach().cpu().tolist()
    return {
        "density": density,
        "counts": [float(v) for v in counts],
    }


def _predict_detection(model, device: torch.device, samples: list[dict[str, Any]], score_thr: float) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    model.eval()
    for sample in samples:
        outputs[sample["sample_id"]] = _predict_detection_one(model, device, sample, score_thr)
    return outputs


def _predict_segmentation(model, device: torch.device, samples: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    outputs: dict[str, np.ndarray] = {}
    model.eval()
    for sample in samples:
        outputs[sample["sample_id"]] = _predict_segmentation_one(model, device, sample)
    return outputs


def _predict_counting(model, device: torch.device, samples: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    model.eval()
    for sample in samples:
        outputs[sample["sample_id"]] = _predict_counting_one(model, device, sample)
    return outputs


def _finalize_ranked_candidates(
    candidates: list[tuple[dict[str, Any], Any, Any]],
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    candidates.sort(key=lambda item: float(item[0].get("metric_improvement", 0.0)), reverse=True)
    selected = candidates[: int(limit)]
    selected_samples = [item[0] for item in selected]
    selected_lora_outputs = {item[0]["sample_id"]: item[1] for item in selected}
    selected_ours_outputs = {item[0]["sample_id"]: item[2] for item in selected}
    return selected_samples, selected_lora_outputs, selected_ours_outputs


def _select_detection_samples(
    samples: Iterable[dict[str, Any]],
    lora_model,
    lora_device: torch.device,
    ours_model,
    ours_device: torch.device,
    num_classes: int,
    score_thr: float,
    limit: int = DEFAULT_SAMPLES_PER_TASK,
    scan_limit: int = DEFAULT_SELECTION_SCAN_LIMIT,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], int]:
    candidates: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    lora_model.eval()
    ours_model.eval()
    scanned = 0
    for sample in samples:
        if scan_limit > 0 and scanned >= int(scan_limit):
            break
        scanned += 1
        sample_id = sample["sample_id"]
        lora_out = _predict_detection_one(lora_model, lora_device, sample, score_thr)
        ours_out = _predict_detection_one(ours_model, ours_device, sample, score_thr)
        lora_ap50 = _single_image_ap50(lora_out, sample["gt_boxes"], sample["gt_labels"], num_classes)
        ours_ap50 = _single_image_ap50(ours_out, sample["gt_boxes"], sample["gt_labels"], num_classes)
        ap50_gain = ours_ap50 - lora_ap50
        if ap50_gain < DET_CLEAR_AP50_GAIN:
            continue
        scored_sample = dict(sample)
        scored_sample["metric_name"] = "AP50"
        scored_sample["metric_lora"] = lora_ap50
        scored_sample["metric_ours"] = ours_ap50
        scored_sample["metric_improvement"] = ap50_gain
        candidates.append((scored_sample, lora_out, ours_out))
    selected_samples, selected_lora_outputs, selected_ours_outputs = _finalize_ranked_candidates(candidates, limit)
    return selected_samples, selected_lora_outputs, selected_ours_outputs, scanned


def _select_segmentation_samples(
    samples: Iterable[dict[str, Any]],
    lora_model,
    lora_device: torch.device,
    ours_model,
    ours_device: torch.device,
    num_classes: int,
    limit: int = DEFAULT_SAMPLES_PER_TASK,
    scan_limit: int = DEFAULT_SELECTION_SCAN_LIMIT,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, np.ndarray], int]:
    candidates: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    lora_model.eval()
    ours_model.eval()
    scanned = 0
    for sample in samples:
        if scan_limit > 0 and scanned >= int(scan_limit):
            break
        scanned += 1
        sample_id = sample["sample_id"]
        lora_mask = _predict_segmentation_one(lora_model, lora_device, sample)
        ours_mask = _predict_segmentation_one(ours_model, ours_device, sample)
        lora_miou = _single_image_miou(lora_mask, sample["gt_mask"], num_classes)
        ours_miou = _single_image_miou(ours_mask, sample["gt_mask"], num_classes)
        miou_gain = ours_miou - lora_miou
        if miou_gain < SEG_CLEAR_MIOU_GAIN:
            continue
        scored_sample = dict(sample)
        scored_sample["metric_name"] = "mIoU"
        scored_sample["metric_lora"] = lora_miou
        scored_sample["metric_ours"] = ours_miou
        scored_sample["metric_improvement"] = miou_gain
        candidates.append((scored_sample, lora_mask, ours_mask))
    selected_samples, selected_lora_outputs, selected_ours_outputs = _finalize_ranked_candidates(candidates, limit)
    return selected_samples, selected_lora_outputs, selected_ours_outputs, scanned


def _select_counting_samples(
    samples: Iterable[dict[str, Any]],
    lora_model,
    lora_device: torch.device,
    ours_model,
    ours_device: torch.device,
    limit: int = DEFAULT_SAMPLES_PER_TASK,
    scan_limit: int = DEFAULT_SELECTION_SCAN_LIMIT,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, dict[str, Any]], int]:
    candidates: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    lora_model.eval()
    ours_model.eval()
    scanned = 0
    for sample in samples:
        if scan_limit > 0 and scanned >= int(scan_limit):
            break
        scanned += 1
        sample_id = sample["sample_id"]
        lora_out = _predict_counting_one(lora_model, lora_device, sample)
        ours_out = _predict_counting_one(ours_model, ours_device, sample)
        lora_mae = _single_image_count_mae(lora_out["counts"], sample["gt_counts"])
        ours_mae = _single_image_count_mae(ours_out["counts"], sample["gt_counts"])
        mae_gain = lora_mae - ours_mae
        rel_gain = mae_gain / max(float(lora_mae), 1e-8)
        gt_total = float(np.sum(sample["gt_counts"]))
        lora_total_mae = abs(float(np.sum(lora_out["counts"])) - gt_total)
        ours_total_mae = abs(float(np.sum(ours_out["counts"])) - gt_total)
        if mae_gain < CNT_CLEAR_MAE_GAIN or rel_gain < CNT_CLEAR_REL_GAIN or ours_total_mae > lora_total_mae:
            continue
        scored_sample = dict(sample)
        scored_sample["metric_name"] = "MAE"
        scored_sample["metric_lora"] = lora_mae
        scored_sample["metric_ours"] = ours_mae
        scored_sample["metric_improvement"] = mae_gain + 0.1 * (lora_total_mae - ours_total_mae)
        scored_sample["metric_relative_improvement"] = rel_gain
        scored_sample["metric_lora_total_mae"] = lora_total_mae
        scored_sample["metric_ours_total_mae"] = ours_total_mae
        candidates.append((scored_sample, lora_out, ours_out))
    selected_samples, selected_lora_outputs, selected_ours_outputs = _finalize_ranked_candidates(candidates, limit)
    return selected_samples, selected_lora_outputs, selected_ours_outputs, scanned


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
    task_dir: Path,
    samples: list[dict[str, Any]],
    label_to_name: dict[int, str],
    lora_outputs: dict[str, dict[str, Any]],
    ours_outputs: dict[str, dict[str, Any]],
    compare_name: str,
) -> Path:
    tiles: list[list[Image.Image]] = [[], [], []]

    for sample in samples:
        sample_id = sample["sample_id"]
        file_stem = _sample_file_stem(sample)
        original_path = task_dir / "original" / f"{file_stem}.png"
        _save_original_sample_image(sample, original_path)
        gt_vis = _draw_boxes(sample["image_pil"], sample["gt_boxes"], sample["gt_labels"], label_to_name)
        gt_vis.save(task_dir / "gt" / f"{file_stem}.png")
        gt_tile = _fit_image(
            gt_vis,
            DEFAULT_TILE_SIZE,
            DEFAULT_TILE_SIZE,
        )
        lora_out = lora_outputs[sample_id]
        lora_vis = _draw_boxes(sample["image_pil"], lora_out["boxes"], lora_out["labels"], label_to_name, lora_out["scores"])
        lora_vis.save(task_dir / "lora" / f"{file_stem}.png")
        lora_tile = _fit_image(
            lora_vis,
            DEFAULT_TILE_SIZE,
            DEFAULT_TILE_SIZE,
        )
        lora_tile = _draw_metric_badge(lora_tile, f"AP50={float(sample.get('metric_lora', 0.0)):.3f}")
        ours_out = ours_outputs[sample_id]
        ours_vis = _draw_boxes(sample["image_pil"], ours_out["boxes"], ours_out["labels"], label_to_name, ours_out["scores"])
        ours_vis.save(task_dir / "ours" / f"{file_stem}.png")
        ours_tile = _fit_image(
            ours_vis,
            DEFAULT_TILE_SIZE,
            DEFAULT_TILE_SIZE,
        )
        ours_tile = _draw_metric_badge(ours_tile, f"AP50={float(sample.get('metric_ours', 0.0)):.3f}")

        tiles[0].append(gt_tile)
        tiles[1].append(lora_tile)
        tiles[2].append(ours_tile)

    compare = _assemble_grid(
        task_title="Detection Comparison",
        row_labels=["GT", "LoRA", "Ours"],
        col_labels=[sample["display_name"] for sample in samples],
        tiles=tiles,
    )
    compare_path = task_dir / compare_name
    compare.save(compare_path)
    return compare_path


def _write_segmentation_outputs(
    task_dir: Path,
    samples: list[dict[str, Any]],
    num_classes: int,
    lora_outputs: dict[str, np.ndarray],
    ours_outputs: dict[str, np.ndarray],
    compare_name: str,
) -> Path:
    tiles: list[list[Image.Image]] = [[], [], []]

    for sample in samples:
        sample_id = sample["sample_id"]
        file_stem = _sample_file_stem(sample)
        sample["display_image"].save(task_dir / "original" / f"{file_stem}.png")
        gt_mask = sample["gt_mask"]
        lora_mask = lora_outputs[sample_id]
        ours_mask = ours_outputs[sample_id]

        gt_vis = _overlay_segmentation_mask(sample["display_image"], gt_mask, num_classes)
        gt_vis.save(task_dir / "gt" / f"{file_stem}.png")
        lora_vis = _overlay_segmentation_mask(sample["display_image"], lora_mask, num_classes)
        lora_vis.save(task_dir / "lora" / f"{file_stem}.png")
        ours_vis = _overlay_segmentation_mask(sample["display_image"], ours_mask, num_classes)
        ours_vis.save(task_dir / "ours" / f"{file_stem}.png")

        gt_tile = _fit_image(
            gt_vis,
            DEFAULT_TILE_SIZE,
            DEFAULT_TILE_SIZE,
        )
        lora_tile = _fit_image(
            lora_vis,
            DEFAULT_TILE_SIZE,
            DEFAULT_TILE_SIZE,
        )
        lora_tile = _draw_metric_badge(lora_tile, f"mIoU={float(sample.get('metric_lora', 0.0)):.3f}")
        ours_tile = _fit_image(
            ours_vis,
            DEFAULT_TILE_SIZE,
            DEFAULT_TILE_SIZE,
        )
        ours_tile = _draw_metric_badge(ours_tile, f"mIoU={float(sample.get('metric_ours', 0.0)):.3f}")

        tiles[0].append(gt_tile)
        tiles[1].append(lora_tile)
        tiles[2].append(ours_tile)

    compare = _assemble_grid(
        task_title="Segmentation Comparison",
        row_labels=["GT", "LoRA", "Ours"],
        col_labels=[sample["display_name"] for sample in samples],
        tiles=tiles,
    )
    compare_path = task_dir / compare_name
    compare.save(compare_path)
    return compare_path


def _write_counting_outputs(
    task_dir: Path,
    samples: list[dict[str, Any]],
    lora_outputs: dict[str, dict[str, Any]],
    ours_outputs: dict[str, dict[str, Any]],
    compare_name: str,
) -> Path:
    tiles: list[list[Image.Image]] = [[], [], []]
    num_classes = int(samples[0]["gt_density"].shape[0]) if samples else 0
    count_vmax = 0.0
    for sample in samples:
        sample_id = sample["sample_id"]
        for counts in (
            sample["gt_counts"],
            lora_outputs[sample_id]["counts"],
            ours_outputs[sample_id]["counts"],
        ):
            counts_np = np.asarray(counts, dtype=np.float32)
            if counts_np.size:
                count_vmax = max(count_vmax, float(np.max(counts_np)))

    for sample in samples:
        sample_id = sample["sample_id"]
        file_stem = _sample_file_stem(sample)
        sample["display_image"].save(task_dir / "original" / f"{file_stem}.png")
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
        sample_class_tiles: list[list[Image.Image]] = [[], [], []]
        class_labels: list[str] = []
        for class_idx in range(num_classes):
            class_name = f"class{class_idx:02d}"
            gt_class = _render_count_class_heatmap(
                sample["display_image"],
                gt_density[class_idx],
                float(gt_counts[class_idx]),
                count_vmax,
                class_name,
                suppress_background=False,
            )
            lora_class = _render_count_class_heatmap(
                sample["display_image"],
                lora_density[class_idx],
                float(lora_counts[class_idx]),
                count_vmax,
                class_name,
                suppress_background=True,
            )
            ours_class = _render_count_class_heatmap(
                sample["display_image"],
                ours_density[class_idx],
                float(ours_counts[class_idx]),
                count_vmax,
                class_name,
                suppress_background=True,
            )
            gt_class.save(task_dir / "gt" / f"{file_stem}_{class_name}.png")
            lora_class.save(task_dir / "lora" / f"{file_stem}_{class_name}.png")
            ours_class.save(task_dir / "ours" / f"{file_stem}_{class_name}.png")
            sample_class_tiles[0].append(_fit_image(gt_class, COUNT_SAMPLE_TILE_SIZE, COUNT_SAMPLE_TILE_SIZE, background=(0, 0, 0)))
            sample_class_tiles[1].append(_fit_image(lora_class, COUNT_SAMPLE_TILE_SIZE, COUNT_SAMPLE_TILE_SIZE, background=(0, 0, 0)))
            sample_class_tiles[2].append(_fit_image(ours_class, COUNT_SAMPLE_TILE_SIZE, COUNT_SAMPLE_TILE_SIZE, background=(0, 0, 0)))
            class_labels.append(class_name)

        sample_compare = _assemble_grid(
            task_title=f"Counting Classes - {sample['display_name']}",
            row_labels=["GT", "LoRA", "Ours"],
            col_labels=class_labels,
            tiles=sample_class_tiles,
        )
        sample_compare.save(task_dir / "sample_compare" / f"{file_stem}.png")

        gt_tile = _render_count_tile(sample["display_image"], gt_density, sum(gt_counts), heatmap_vmax, DEFAULT_TILE_SIZE)
        lora_tile = _render_count_tile(sample["display_image"], lora_density, sum(lora_counts), heatmap_vmax, DEFAULT_TILE_SIZE)
        lora_tile = _draw_metric_badge(lora_tile, f"MAE={float(sample.get('metric_lora', 0.0)):.2f}")
        ours_tile = _render_count_tile(sample["display_image"], ours_density, sum(ours_counts), heatmap_vmax, DEFAULT_TILE_SIZE)
        ours_tile = _draw_metric_badge(ours_tile, f"MAE={float(sample.get('metric_ours', 0.0)):.2f}")

        tiles[0].append(gt_tile)
        tiles[1].append(lora_tile)
        tiles[2].append(ours_tile)

    compare = _assemble_grid(
        task_title="Counting Comparison",
        row_labels=["GT", "LoRA", "Ours"],
        col_labels=[sample["display_name"] for sample in samples],
        tiles=tiles,
    )
    compare_path = task_dir / compare_name
    compare.save(compare_path)
    return compare_path


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize LoRA baseline vs Ours on detection/segmentation/counting test sets.")
    parser.add_argument("--lora-ckpt", type=str, required=True, help="checkpoint trained by ours/lora_multitask")
    parser.add_argument("--ours-ckpt", type=str, required=True, help="checkpoint trained by ours/115_grpo_mainonly")
    parser.add_argument("--output-dir", type=str, default=None, help="output directory for compare figures")
    parser.add_argument("--tasks", type=str, default="cnt", help="comma-separated: det,seg,cnt")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--selection-scan-limit", type=int, default=DEFAULT_SELECTION_SCAN_LIMIT, help="max samples to scan per task before ranking clear Ours wins; <=0 scans all")

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

    det_dataset = None
    seg_dataset = None
    cnt_dataset = None
    label_to_name: dict[int, str] = {}

    if "det" in tasks:
        det_dataset, label_to_name = _build_detection_dataset(args)
        print(f"[visualize] prepared detection dataset with {len(det_dataset)} samples")
    if "seg" in tasks:
        seg_dataset = _build_segmentation_dataset(args)
        print(f"[visualize] prepared segmentation dataset with {len(seg_dataset)} samples")
    if "cnt" in tasks:
        cnt_dataset = _build_counting_dataset(args, int(lora_meta["cnt_num_classes"]))
        print(f"[visualize] prepared counting dataset with {len(cnt_dataset)} samples")

    print("[visualize] loading Ours model while LoRA remains loaded...")
    ours_model, ours_meta, ours_device = _load_ours_model(args)
    print(f"[visualize] Ours model loaded on {ours_device}")

    for key in ("det_fg_classes", "seg_num_classes", "cnt_num_classes"):
        if int(lora_meta[key]) != int(ours_meta[key]):
            raise SystemExit(f"Checkpoint mismatch for {key}: LoRA={lora_meta[key]} vs Ours={ours_meta[key]}")

    selected_samples: dict[str, list[dict[str, Any]]] = {}
    lora_outputs: dict[str, Any] = {}
    ours_outputs: dict[str, Any] = {}
    if "det" in tasks:
        selected_samples["det"], lora_outputs["det"], ours_outputs["det"], scanned = _select_detection_samples(
            _iter_detection_samples(args, det_dataset, label_to_name),
            lora_model,
            lora_device,
            ours_model,
            ours_device,
            int(ours_meta["det_fg_classes"]),
            float(args.det_score_thr),
            scan_limit=int(args.selection_scan_limit),
        )
        print(f"[visualize] detection clear ours-better samples: {len(selected_samples['det'])}/{DEFAULT_SAMPLES_PER_TASK} after scanning {scanned}")
    if "seg" in tasks:
        selected_samples["seg"], lora_outputs["seg"], ours_outputs["seg"], scanned = _select_segmentation_samples(
            _iter_segmentation_samples(seg_dataset),
            lora_model,
            lora_device,
            ours_model,
            ours_device,
            int(ours_meta["seg_num_classes"]),
            scan_limit=int(args.selection_scan_limit),
        )
        print(f"[visualize] segmentation clear ours-better samples: {len(selected_samples['seg'])}/{DEFAULT_SAMPLES_PER_TASK} after scanning {scanned}")
    if "cnt" in tasks:
        selected_samples["cnt"], lora_outputs["cnt"], ours_outputs["cnt"], scanned = _select_counting_samples(
            _iter_counting_samples(cnt_dataset, int(ours_meta["cnt_num_classes"])),
            lora_model,
            lora_device,
            ours_model,
            ours_device,
            scan_limit=int(args.selection_scan_limit),
        )
        print(f"[visualize] counting clear ours-better samples: {len(selected_samples['cnt'])}/{DEFAULT_SAMPLES_PER_TASK} after scanning {scanned}")

    _release_model(lora_model, lora_device)
    lora_model = None
    _release_model(ours_model, ours_device)
    ours_model = None

    produced_paths: list[Path] = []
    if "det" in tasks:
        task_dir = output_dir / "det"
        _prepare_task_dir(task_dir)
        if selected_samples["det"]:
            compare_path = _write_detection_outputs(
                task_dir=task_dir,
                samples=selected_samples["det"],
                label_to_name=label_to_name,
                lora_outputs=lora_outputs["det"],
                ours_outputs=ours_outputs["det"],
                compare_name="compare.png",
            )
            produced_paths.append(compare_path)
    if "seg" in tasks:
        task_dir = output_dir / "seg"
        _prepare_task_dir(task_dir)
        if selected_samples["seg"]:
            compare_path = _write_segmentation_outputs(
                task_dir=task_dir,
                samples=selected_samples["seg"],
                num_classes=int(ours_meta["seg_num_classes"]),
                lora_outputs=lora_outputs["seg"],
                ours_outputs=ours_outputs["seg"],
                compare_name="compare.png",
            )
            produced_paths.append(compare_path)
    if "cnt" in tasks:
        task_dir = output_dir / "cnt"
        _prepare_task_dir(task_dir)
        if selected_samples["cnt"]:
            compare_path = _write_counting_outputs(
                task_dir=task_dir,
                samples=selected_samples["cnt"],
                lora_outputs=lora_outputs["cnt"],
                ours_outputs=ours_outputs["cnt"],
                compare_name="compare.png",
            )
            produced_paths.append(compare_path)

    for compare_path in produced_paths:
        print(f"[visualize] compare: {compare_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

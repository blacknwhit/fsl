#!/usr/bin/env python
"""Compute per-category AP50 for COCO-format detection predictions."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_ann_file(pred_path: Path, explicit_ann_file: str | None) -> Path:
    if explicit_ann_file:
        ann_path = Path(explicit_ann_file).expanduser()
        if ann_path.exists():
            return ann_path.resolve()
        raise FileNotFoundError(f"--ann-file does not exist: {ann_path}")

    pred_dir = pred_path.parent
    for metrics_path in sorted(pred_dir.glob("metrics*.json")):
        try:
            ann_value = load_json(metrics_path).get("ann_file")
        except Exception:
            continue
        if not ann_value:
            continue

        ann_path = Path(ann_value)
        if ann_path.exists():
            return ann_path.resolve()

        basename = ann_path.name
        for candidate in (
            pred_dir / basename,
            pred_dir / "annotations" / basename,
            pred_dir.parent / "annotations" / basename,
        ):
            if candidate.exists():
                return candidate.resolve()

    for candidate in (
        pred_dir / "instances_test.json",
        pred_dir / "annotations" / "instances_test.json",
        pred_dir.parent / "annotations" / "instances_test.json",
    ):
        if candidate.exists():
            return candidate.resolve()

    raise FileNotFoundError(
        "Could not find the COCO ground-truth annotation file. "
        "Pass it explicitly with --ann-file."
    )


def count_gt_by_category(coco_gt: COCO, cat_ids: list[int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for cat_id in cat_ids:
        ann_ids = coco_gt.getAnnIds(catIds=[cat_id])
        anns = coco_gt.loadAnns(ann_ids)
        counts[cat_id] = sum(1 for ann in anns if not ann.get("iscrowd", 0))
    return counts


def count_predictions_by_category(pred_file: Path, max_dets: int) -> dict[int, int]:
    predictions = load_json(pred_file)
    per_cat_img: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for det in predictions:
        per_cat_img[int(det["category_id"])][int(det["image_id"])] += 1
    return {
        cat_id: sum(min(n, max_dets) for n in per_img.values())
        for cat_id, per_img in per_cat_img.items()
    }


def compute_per_category_ap50(
    ann_file: Path, pred_file: Path, max_dets: int
) -> tuple[list[dict[str, Any]], float]:
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(ann_file))
        # Some converted COCO files omit optional top-level metadata. pycocotools
        # loadRes copies these keys even though bbox evaluation does not use them.
        coco_gt.dataset.setdefault("info", {})
        coco_gt.dataset.setdefault("licenses", [])
        coco_dt = coco_gt.loadRes(str(pred_file))
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.iouThrs = [0.5]
        coco_eval.params.maxDets = [1, 10, max_dets]
        coco_eval.evaluate()
        coco_eval.accumulate()

    cat_ids = list(coco_eval.params.catIds)
    categories = {cat["id"]: cat for cat in coco_gt.loadCats(cat_ids)}
    gt_counts = count_gt_by_category(coco_gt, cat_ids)
    pred_counts = count_predictions_by_category(pred_file, max_dets)

    # precision shape: [iou, recall, category, area, max_dets]
    precision = coco_eval.eval["precision"]
    rows: list[dict[str, Any]] = []
    valid_ap50: list[float] = []
    for cat_index, cat_id in enumerate(cat_ids):
        values = precision[0, :, cat_index, 0, 2]
        values = values[values > -1]
        ap50 = float(values.mean()) if values.size else float("nan")
        if not math.isnan(ap50):
            valid_ap50.append(ap50)

        rows.append(
            {
                "category_id": int(cat_id),
                "category_name": categories[cat_id].get("name", str(cat_id)),
                "ap50": float(ap50),
                "num_gt": int(gt_counts.get(cat_id, 0)),
                "num_predictions": int(pred_counts.get(cat_id, 0)),
            }
        )

    mean_ap50 = sum(valid_ap50) / len(valid_ap50) if valid_ap50 else float("nan")
    return rows, mean_ap50


def write_outputs(
    pred_file: Path,
    ann_file: Path,
    rows: list[dict[str, Any]],
    mean_ap50: float,
    max_dets: int,
) -> tuple[Path, Path]:
    out_base = pred_file.with_suffix("")
    csv_path = out_base.with_name(out_base.name + "_per_category_ap50.csv")
    json_path = out_base.with_name(out_base.name + "_per_category_ap50.json")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["category_id", "category_name", "ap50", "num_gt", "num_predictions"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "ap50": "" if math.isnan(row["ap50"]) else f"{row['ap50']:.6f}",
                }
            )

    payload = {
        "prediction_file": str(pred_file),
        "annotation_file": str(ann_file),
        "backend": "pycocotools",
        "iou_threshold": 0.5,
        "max_dets_per_image_per_category": max_dets,
        "mean_ap50": mean_ap50,
        "per_category": rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, allow_nan=True)
        f.write("\n")

    return csv_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-category AP50 from a COCO detection-result JSON. "
            "The prediction file alone is not enough; this script reads the "
            "ground-truth ann_file from same-directory metrics*.json, or from --ann-file."
        )
    )
    parser.add_argument("pred_file", help="Path to predictions_instances_test.json")
    parser.add_argument("--ann-file", default=None, help="Path to COCO ground-truth instances JSON")
    parser.add_argument("--max-dets", type=int, default=100, help="COCO maxDets value, default 100")
    args = parser.parse_args()

    pred_file = Path(args.pred_file).expanduser().resolve()
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file does not exist: {pred_file}")

    ann_file = resolve_ann_file(pred_file, args.ann_file)
    rows, mean_ap50 = compute_per_category_ap50(ann_file, pred_file, args.max_dets)
    csv_path, json_path = write_outputs(pred_file, ann_file, rows, mean_ap50, args.max_dets)

    print(f"annotation_file: {ann_file}")
    print("backend: pycocotools")
    print(f"mean_ap50: {mean_ap50:.6f}")
    print(f"csv: {csv_path}")
    print(f"json: {json_path}")


if __name__ == "__main__":
    main()

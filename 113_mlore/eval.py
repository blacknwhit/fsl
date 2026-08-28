from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch

try:
    from .models import MultiTaskModel, SharedDinoV3Backbone
except ImportError:
    from models import MultiTaskModel, SharedDinoV3Backbone


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


def _parse_tasks(text: str) -> List[str]:
    tasks = [item.strip().lower() for item in str(text or "").split(",") if item.strip()]
    valid = {"det", "seg", "cnt"}
    invalid = [task for task in tasks if task not in valid]
    if invalid:
        raise ValueError(f"Invalid task(s): {invalid}; valid tasks are {sorted(valid)}")
    out: List[str] = []
    for task in tasks:
        if task not in out:
            out.append(task)
    return out


def _default_stats_dir(checkpoint: str, stats_dir: Optional[str]) -> Path:
    if stats_dir:
        path = Path(stats_dir)
    else:
        path = Path(checkpoint).parent / "stats"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _infer_fg_num_classes_from_det_state(det_state: Dict) -> int:
    weight = det_state.get("roi_heads.box_predictor.cls_score.weight")
    if not hasattr(weight, "shape"):
        raise ValueError("Missing roi_heads.box_predictor.cls_score.weight in det_head state")
    total_classes = int(weight.shape[0])
    if total_classes < 2:
        raise ValueError("Detector cls_score weight is too small to infer foreground class count")
    return total_classes - 1


def _infer_num_classes_from_conv1x1_weight(state: Dict, weight_key: str) -> int:
    weight = state.get(weight_key)
    if not hasattr(weight, "shape"):
        raise ValueError(f"Missing {weight_key} in state dict")
    return int(weight.shape[0])


def _infer_model_config(ckpt: Dict) -> Dict:
    config = dict(ckpt.get("model_config") or {})
    if config:
        return config

    det_state = ckpt["det_head"]
    seg_state = ckpt["seg_head"]
    cnt_state = ckpt["cnt_head"]
    shared_state = ckpt["backbone"]

    config["model_name"] = "dinov3_vitl16"
    config["image_size"] = 448
    config["det_num_classes"] = _infer_fg_num_classes_from_det_state(det_state)
    config["seg_num_classes"] = _infer_num_classes_from_conv1x1_weight(seg_state, "decode.3.weight")
    config["cnt_num_classes"] = _infer_num_classes_from_conv1x1_weight(cnt_state, "decode.3.weight")
    config["det_out_channels"] = 256
    config["mlore_decoder_dim"] = int(seg_state["decode.0.weight"].shape[1])

    rank_list = []
    index = 0
    while True:
        key = f"decoder.stage1_blocks.0.lora_list.{index}.W.weight"
        if key not in shared_state:
            break
        rank_list.append(int(shared_state[key].shape[0]))
        index += 1
    config["mlore_rank_list"] = rank_list or [8, 16, 24, 32, 40, 48]
    config["mlore_topk"] = 4
    task_rank_weight = shared_state.get("decoder.stage1_blocks.0.conv2.det.W.weight")
    config["mlore_task_rank"] = int(task_rank_weight.shape[0]) if hasattr(task_rank_weight, "shape") else 32
    config["mlore_pre_softmax"] = False
    config["mlore_load_balancing_weight"] = 3e-4
    config["mlore_select_layers"] = [23]
    config["mlore_num_stages"] = 2 if "decoder.stage2_blocks.0.conv1.det.weight" in shared_state else 1
    return config


def _build_model_from_checkpoint(
    checkpoint_path: str,
    *,
    device: str | None,
    model_name: str | None,
    image_size: int | None,
) -> tuple[MultiTaskModel, Dict, torch.device]:
    ckpt = _torch_load_cpu(checkpoint_path)
    if not _is_multitask_checkpoint(ckpt):
        raise SystemExit(f"Unsupported checkpoint format: {checkpoint_path}")

    config = _infer_model_config(ckpt)
    if model_name is not None:
        config["model_name"] = model_name
    if image_size is not None:
        config["image_size"] = int(image_size)

    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        device_index = dev.index if dev.index is not None else 0
        torch.cuda.set_device(device_index)
        dev = torch.device(f"cuda:{device_index}")

    shared = SharedDinoV3Backbone(
        model_name=config["model_name"],
        image_size=int(config["image_size"]),
        checkpoint_path=None,
        mlore_decoder_dim=int(config["mlore_decoder_dim"]),
        mlore_rank_list=config["mlore_rank_list"],
        mlore_topk=int(config["mlore_topk"]),
        mlore_task_rank=int(config["mlore_task_rank"]),
        mlore_pre_softmax=bool(config["mlore_pre_softmax"]),
        mlore_load_balancing_weight=float(config["mlore_load_balancing_weight"]),
        mlore_select_layers=config["mlore_select_layers"],
        mlore_num_stages=int(config.get("mlore_num_stages", 1)),
        grad_checkpointing=False,
    )
    model = MultiTaskModel(
        shared=shared,
        det_num_classes=int(config["det_num_classes"]),
        seg_num_classes=int(config["seg_num_classes"]),
        cnt_num_classes=int(config["cnt_num_classes"]),
        image_size=int(config["image_size"]),
        det_out_channels=int(config.get("det_out_channels", 256)),
    ).to(dev)
    model.eval()

    shared_missing, shared_unexpected = model.shared.load_state_dict(ckpt["backbone"], strict=False)
    det_missing, det_unexpected = model.detector.load_state_dict(ckpt["det_head"], strict=False)
    seg_missing, seg_unexpected = model.seg_head.load_state_dict(ckpt["seg_head"], strict=False)
    cnt_missing, cnt_unexpected = model.cnt_head.load_state_dict(ckpt["cnt_head"], strict=False)

    if shared_missing or shared_unexpected:
        print(f"[eval][warn] shared load mismatch: missing={len(shared_missing)} unexpected={len(shared_unexpected)}")
    if det_unexpected:
        print(f"[eval][warn] detector unexpected keys: {len(det_unexpected)}")
    det_real_missing = [key for key in det_missing if not key.startswith("backbone.shared.")]
    if det_real_missing:
        print(f"[eval][warn] detector real missing keys: {len(det_real_missing)}")
    if seg_missing or seg_unexpected:
        print(f"[eval][warn] seg_head load mismatch: missing={len(seg_missing)} unexpected={len(seg_unexpected)}")
    if cnt_missing or cnt_unexpected:
        print(f"[eval][warn] cnt_head load mismatch: missing={len(cnt_missing)} unexpected={len(cnt_unexpected)}")

    return model, config, dev


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the 113_mlore multitask checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="det,seg,cnt")
    parser.add_argument("--stats-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--check-load-only", action="store_true")

    parser.add_argument("--det-data-root", type=str, default=None)
    parser.add_argument("--det-ann-file", type=str, default=None)
    parser.add_argument("--det-img-dir", type=str, default=None)
    parser.add_argument("--det-score-thr", type=float, default=0.0)
    parser.add_argument("--det-use-coco-eval", action="store_true")
    parser.add_argument("--det-batch-size", type=int, default=8)
    parser.add_argument("--det-num-workers", type=int, default=0)

    parser.add_argument("--seg-data-dir", type=str, default=None)
    parser.add_argument("--seg-num-classes", type=int, default=None)

    parser.add_argument("--cnt-data-root", type=str, default=None)
    parser.add_argument("--cnt-test-dir", type=str, default=None)
    parser.add_argument("--cnt-num-classes", type=int, default=None)
    aspect = parser.add_mutually_exclusive_group()
    aspect.add_argument("--cnt-keep-aspect", dest="cnt_keep_aspect", action="store_true")
    aspect.add_argument("--cnt-no-keep-aspect", dest="cnt_keep_aspect", action="store_false")
    parser.set_defaults(cnt_keep_aspect=None)
    parser.add_argument("--cnt-batch-size", type=int, default=4)
    parser.add_argument("--cnt-num-workers", type=int, default=0)
    return parser.parse_args()


def _eval_full_model_det(
    model,
    *,
    stats_dir: Path,
    ann_file: str,
    img_dir: str,
    device: torch.device,
    score_thr: float,
    use_coco_eval: bool,
    batch_size: int,
    num_workers: int,
) -> dict:
    from object_detection.dataset import CocoDetectionDataset, collate_fn
    from object_detection.utils import compute_precision_recall
    from torch.utils.data import DataLoader
    import torchvision.transforms.functional as TF

    try:
        from pycocotools.coco import COCO  # type: ignore
        from pycocotools.cocoeval import COCOeval  # type: ignore
    except Exception:
        COCO = None
        COCOeval = None

    dataset = CocoDetectionDataset(str(ann_file), str(img_dir), transform=lambda img, tgt: (TF.to_tensor(img), tgt))
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    use_coco = bool(use_coco_eval and COCO is not None and COCOeval is not None)
    coco_results: List[Dict] = []
    total_tp = total_fp = total_fn = 0

    model.eval()
    with torch.no_grad():
        for index, (images, targets) in enumerate(loader):
            images = [img.to(device, non_blocking=True) for img in images]
            outputs = model.forward_det(images)

            if use_coco:
                for out, target in zip(outputs, targets):
                    image_id = int(target["image_id"])
                    boxes = out["boxes"].detach().cpu()
                    labels = out["labels"].detach().cpu()
                    scores = out["scores"].detach().cpu()
                    keep = scores >= float(score_thr)
                    boxes = boxes[keep]
                    labels = labels[keep]
                    scores = scores[keep]
                    for box, label, score in zip(boxes, labels, scores):
                        x1, y1, x2, y2 = box.tolist()
                        coco_results.append(
                            {
                                "image_id": image_id,
                                "category_id": int(dataset.label_to_cat_id[int(label)]),
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "score": float(score),
                            }
                        )
            else:
                targets_dev = [
                    {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in target.items()}
                    for target in targets
                ]
                tp, fp, fn = compute_precision_recall(outputs, targets_dev, score_threshold=float(score_thr))
                total_tp += int(tp)
                total_fp += int(fp)
                total_fn += int(fn)

            if (index + 1) % 50 == 0:
                print(f"[eval][det] processed {index + 1}/{len(loader)}")

    split_name = Path(ann_file).stem
    if use_coco:
        coco_gt = COCO(str(ann_file))
        coco_gt.dataset.setdefault("info", {})
        coco_gt.dataset.setdefault("licenses", [])
        if coco_results:
            coco_dt = coco_gt.loadRes(coco_results)
            coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats = [float(x) for x in getattr(coco_eval, "stats", [])]
        else:
            stats = []
        metric_names = [
            "AP@[.50:.95]",
            "AP@0.50",
            "AP@0.75",
            "AP_small",
            "AP_medium",
            "AP_large",
            "AR@1",
            "AR@10",
            "AR@100",
            "AR_small",
            "AR_medium",
            "AR_large",
        ]
        metrics = {name: stats[i] if i < len(stats) else None for i, name in enumerate(metric_names)}
        metrics.update(
            {
                "task": "detection",
                "ann_file": str(ann_file),
                "img_dir": str(img_dir),
                "score_thr": float(score_thr),
                "num_predictions": int(len(coco_results)),
                "use_coco_eval": True,
            }
        )
        pred_path = stats_dir / f"predictions_det_{split_name}.json"
        with pred_path.open("w", encoding="utf-8") as handle:
            json.dump(coco_results, handle, ensure_ascii=False, indent=2)
        print(f"[eval][det] saved predictions: {pred_path}")
    else:
        precision = float(total_tp / max(total_tp + total_fp, 1))
        recall = float(total_tp / max(total_tp + total_fn, 1))
        metrics = {
            "task": "detection",
            "ann_file": str(ann_file),
            "img_dir": str(img_dir),
            "score_thr": float(score_thr),
            "precision@0.5": precision,
            "recall@0.5": recall,
            "tp": int(total_tp),
            "fp": int(total_fp),
            "fn": int(total_fn),
            "use_coco_eval": False,
        }

    metrics_path = stats_dir / f"metrics_det_{split_name}.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(f"[eval][det] saved metrics: {metrics_path}")
    return metrics


def _eval_full_model_seg(model, *, stats_dir: Path, data_dir: str, device: torch.device, num_classes: int, image_size: int) -> dict:
    from segmentation.dataset import SegmentationDataset
    seg_utils = importlib.import_module("segmentation.utils")
    per_class_iou_from_confusion = seg_utils.per_class_iou_from_confusion
    update_confusion_matrix = seg_utils.update_confusion_matrix
    import torchvision.transforms.functional as TF

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def _transform(img, mask):
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=mean, std=std)
        mask_tensor = torch.as_tensor(TF.pil_to_tensor(mask), dtype=torch.long).squeeze(0)
        return img, mask_tensor

    dataset = SegmentationDataset(data_dir, transform=_transform, image_size=int(image_size))
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    conf = torch.zeros((int(num_classes), int(num_classes)), dtype=torch.int64)
    model.eval()
    with torch.no_grad():
        for index, (images, masks) in enumerate(loader):
            if index % 10 == 0:
                print(f"[eval][seg] processed {index}/{len(loader)}")
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model.forward_seg(images)
            update_confusion_matrix(
                conf=conf,
                logits_or_preds=logits.detach(),
                target=masks.detach(),
                num_classes=int(num_classes),
                ignore_indices=(255, 11),
            )

    per_class_iou, miou = per_class_iou_from_confusion(conf)
    split_name = Path(data_dir).name or "seg"
    metrics = {
        "task": "segmentation",
        "data_dir": str(Path(data_dir)),
        "num_classes": int(num_classes),
        "image_size": int(image_size),
        "ignore_indices": [255, 11],
        "per_class_iou": [None if (value != value) else float(value.item()) for value in per_class_iou],
        "miou": None if (float(miou.item()) != float(miou.item())) else float(miou.item()),
    }
    metrics_path = stats_dir / f"metrics_seg_{split_name}.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(f"[eval][seg] saved metrics: {metrics_path}")
    return metrics


def _eval_full_model_cnt(
    model,
    *,
    stats_dir: Path,
    data_root: str,
    test_dir: str | None,
    device: torch.device,
    num_classes: int,
    image_size: int,
    keep_aspect: bool,
    batch_size: int,
    num_workers: int,
) -> dict:
    from counting.dataset import DSACADensityH5Dataset
    import torchvision.transforms.functional as TF

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    class _NormalizeTransform:
        def __call__(self, img, density):
            if not torch.is_tensor(img):
                img = TF.to_tensor(img)
            img = TF.normalize(img, mean=mean, std=std)
            return img, density

    root = Path(data_root)
    data_dir = Path(test_dir) if test_dir else (root / "test_data_class8")
    dataset = DSACADensityH5Dataset(
        str(data_dir),
        num_classes=int(num_classes),
        transform=_NormalizeTransform(),
        image_size=int(image_size),
        keep_aspect=bool(keep_aspect),
    )
    loader_kwargs = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "num_workers": int(num_workers),
        "pin_memory": True,
    }
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
    loader = torch.utils.data.DataLoader(dataset, **loader_kwargs)

    sum_abs = torch.zeros(int(num_classes), dtype=torch.float64)
    sum_sq = torch.zeros(int(num_classes), dtype=torch.float64)
    count = 0

    model.eval()
    with torch.no_grad():
        for index, (images, density) in enumerate(loader):
            if (index + 1) % 50 == 0:
                print(f"[eval][cnt] processed {index + 1}/{len(loader)}")
            images = images.to(device, non_blocking=True)
            density = density.to(device, non_blocking=True)
            _, pred_counts = model.forward_cnt(images)
            gt_counts = density.flatten(2).sum(dim=2)
            error = (pred_counts - gt_counts).to(torch.float64)
            sum_abs += error.abs().sum(dim=0).cpu()
            sum_sq += (error * error).sum(dim=0).cpu()
            count += int(images.shape[0])

    mae = sum_abs / max(1, count)
    rmse = (sum_sq / max(1, count)).sqrt()
    split_name = data_dir.name or "count"
    metrics = {
        "task": "counting",
        "data_root": str(root),
        "test_dir": str(data_dir),
        "num_classes": int(num_classes),
        "image_size": int(image_size),
        "keep_aspect": bool(keep_aspect),
        "per_class_mae": [float(value.item()) for value in mae],
        "per_class_rmse": [float(value.item()) for value in rmse],
        "mae": float(mae.mean().item()),
        "rmse": float(rmse.mean().item()),
        "n_images": int(count),
    }
    metrics_path = stats_dir / f"metrics_cnt_{split_name}.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(f"[eval][cnt] saved metrics: {metrics_path}")
    return metrics


def main():
    args = parse_args()
    model, config, device = _build_model_from_checkpoint(
        args.checkpoint,
        device=args.device,
        model_name=args.model_name,
        image_size=args.image_size,
    )
    if args.check_load_only:
        print(json.dumps({"status": "ok", "model_config": config}, ensure_ascii=False, indent=2))
        return

    stats_dir = _default_stats_dir(args.checkpoint, args.stats_dir)
    tasks = _parse_tasks(args.tasks)
    results = {"model_config": config, "tasks": tasks, "results": {}}

    if "det" in tasks:
        if not (args.det_ann_file and args.det_img_dir):
            raise SystemExit("det task selected but --det-ann-file/--det-img-dir is missing")
        results["results"]["det"] = _eval_full_model_det(
            model,
            stats_dir=stats_dir,
            ann_file=args.det_ann_file,
            img_dir=args.det_img_dir,
            device=device,
            score_thr=float(args.det_score_thr),
            use_coco_eval=bool(args.det_use_coco_eval),
            batch_size=int(args.det_batch_size),
            num_workers=int(args.det_num_workers),
        )

    if "seg" in tasks:
        if not args.seg_data_dir:
            raise SystemExit("seg task selected but --seg-data-dir is missing")
        seg_num_classes = int(args.seg_num_classes) if args.seg_num_classes is not None else int(config["seg_num_classes"])
        image_size = int(args.image_size) if args.image_size is not None else int(config["image_size"])
        results["results"]["seg"] = _eval_full_model_seg(
            model,
            stats_dir=stats_dir,
            data_dir=args.seg_data_dir,
            device=device,
            num_classes=seg_num_classes,
            image_size=image_size,
        )

    if "cnt" in tasks:
        if not args.cnt_data_root:
            raise SystemExit("cnt task selected but --cnt-data-root is missing")
        cnt_num_classes = int(args.cnt_num_classes) if args.cnt_num_classes is not None else int(config["cnt_num_classes"])
        image_size = int(args.image_size) if args.image_size is not None else int(config["image_size"])
        keep_aspect = True if args.cnt_keep_aspect is None else bool(args.cnt_keep_aspect)
        results["results"]["cnt"] = _eval_full_model_cnt(
            model,
            stats_dir=stats_dir,
            data_root=args.cnt_data_root,
            test_dir=args.cnt_test_dir,
            device=device,
            num_classes=cnt_num_classes,
            image_size=image_size,
            keep_aspect=keep_aspect,
            batch_size=int(args.cnt_batch_size),
            num_workers=int(args.cnt_num_workers),
        )

    summary_path = stats_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"[eval] saved summary: {summary_path}")


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    main()

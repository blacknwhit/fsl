from __future__ import annotations

import argparse
import importlib
import json
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
    path = Path(stats_dir) if stats_dir else Path(checkpoint).parent / "stats"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _infer_fg_num_classes_from_det_state(det_state: Dict) -> int:
    weight = det_state.get("roi_heads.box_predictor.cls_score.weight")
    if not hasattr(weight, "shape"):
        raise ValueError("Missing roi_heads.box_predictor.cls_score.weight in det_head state")
    return int(weight.shape[0]) - 1


def _infer_num_classes_from_conv1x1_weight(state: Dict, weight_key: str) -> int:
    weight = state.get(weight_key)
    if not hasattr(weight, "shape"):
        raise ValueError(f"Missing {weight_key} in state dict")
    return int(weight.shape[0])


def _infer_det_out_channels(det_state: Dict, default: int = 256) -> int:
    weight = det_state.get("backbone.proj.weight")
    return int(weight.shape[0]) if hasattr(weight, "shape") else int(default)


def _infer_backbone_config(shared_state: Dict) -> Dict:
    config = {
        "model_name": "dinov3_vitl16",
        "image_size": 448,
        "lora_rank": 8,
        "num_shared_experts": 9,
        "lora_alpha": 32.0,
        "adapter_dropout": 0.05,
        "routing_group_size": 512,
        "grad_checkpointing": False,
    }
    for key, value in shared_state.items():
        if key.endswith(".adapter.lora_A.weight") and hasattr(value, "shape"):
            config["lora_rank"] = int(value.shape[0])
            break
    expert_indices = []
    for key in shared_state.keys():
        parts = str(key).split(".")
        if len(parts) >= 6 and parts[-4:-2] == ["adapter", "lora_B"]:
            try:
                expert_indices.append(int(parts[-2]))
            except ValueError:
                pass
    if expert_indices:
        config["num_shared_experts"] = max(expert_indices) + 1
    router_weight = None
    for key, value in shared_state.items():
        if key.endswith(".adapter.routers.det.weight") and hasattr(value, "shape"):
            router_weight = value
            break
    if router_weight is not None and config["num_shared_experts"] > 0:
        dim = int(router_weight.shape[1])
        num_groups = int(router_weight.shape[0]) // int(config["num_shared_experts"])
        if num_groups > 0 and dim % num_groups == 0:
            config["routing_group_size"] = dim // num_groups
    return config


def _infer_model_config(ckpt: Dict) -> Dict:
    config = dict(ckpt.get("model_config") or {})
    if config:
        return config
    config = _infer_backbone_config(ckpt["backbone"])
    config["det_num_classes"] = _infer_fg_num_classes_from_det_state(ckpt["det_head"])
    config["seg_num_classes"] = _infer_num_classes_from_conv1x1_weight(ckpt["seg_head"], "decode.3.weight")
    config["cnt_num_classes"] = _infer_num_classes_from_conv1x1_weight(ckpt["cnt_head"], "decode.3.weight")
    config["det_out_channels"] = _infer_det_out_channels(ckpt["det_head"], default=256)
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
        lora_rank=int(config.get("lora_rank", 8)),
        num_shared_experts=int(config.get("num_shared_experts", 9)),
        lora_alpha=float(config.get("lora_alpha", 32.0)),
        adapter_dropout=float(config.get("adapter_dropout", 0.05)),
        routing_group_size=int(config.get("routing_group_size", 512)),
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
    parser = argparse.ArgumentParser(description="Evaluate the 113_mtlora_vision multitask checkpoint")
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
    parser.add_argument("--cnt-batch-size", type=int, default=8)
    parser.add_argument("--cnt-num-workers", type=int, default=0)
    keep_aspect = parser.add_mutually_exclusive_group()
    keep_aspect.add_argument("--cnt-keep-aspect", dest="cnt_keep_aspect", action="store_true")
    keep_aspect.add_argument("--cnt-no-keep-aspect", dest="cnt_keep_aspect", action="store_false")
    parser.set_defaults(cnt_keep_aspect=True)
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
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, num_workers=int(num_workers), pin_memory=True, collate_fn=collate_fn)

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
                    for box, label, score in zip(boxes[keep], labels[keep], scores[keep]):
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
                targets_dev = [{k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in target.items()} for target in targets]
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
        stats = []
        if coco_results:
            coco_dt = coco_gt.loadRes(coco_results)
            coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats = [float(x) for x in getattr(coco_eval, "stats", [])]
        metric_names = ["AP@[.50:.95]", "AP@0.50", "AP@0.75", "AP_small", "AP_medium", "AP_large", "AR@1", "AR@10", "AR@100", "AR_small", "AR_medium", "AR_large"]
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
        metrics = {
            "task": "detection",
            "ann_file": str(ann_file),
            "img_dir": str(img_dir),
            "score_thr": float(score_thr),
            "precision@0.5": float(total_tp / max(total_tp + total_fp, 1)),
            "recall@0.5": float(total_tp / max(total_tp + total_fn, 1)),
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
    import torchvision.transforms.functional as TF

    seg_utils = importlib.import_module("segmentation.utils")
    per_class_iou_from_confusion = seg_utils.per_class_iou_from_confusion
    update_confusion_matrix = seg_utils.update_confusion_matrix
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def _transform(img, mask):
        img = TF.normalize(TF.to_tensor(img), mean=mean, std=std)
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
            update_confusion_matrix(conf=conf, logits_or_preds=logits.detach(), target=masks.detach(), num_classes=int(num_classes), ignore_indices=(255, 11))
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


def _eval_full_model_cnt(model, *, stats_dir: Path, data_root: str, test_dir: str | None, device: torch.device, num_classes: int, image_size: int, keep_aspect: bool, batch_size: int, num_workers: int) -> dict:
    import inspect
    import torchvision.transforms.functional as TF
    from counting.dataset import DSACADensityH5Dataset
    from torch.utils.data import DataLoader

    target_dir = test_dir if test_dir is not None else str(Path(data_root) / "test_data_class8")
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    class _NormalizeTransform:
        def __init__(self, mean_, std_):
            self.mean = mean_
            self.std = std_

        def __call__(self, img, density):
            if not torch.is_tensor(img):
                img = TF.to_tensor(img)
            return TF.normalize(img, mean=self.mean, std=self.std), density

    init_params = inspect.signature(DSACADensityH5Dataset.__init__).parameters
    if "root" in init_params and "split_dir" in init_params:
        dataset_kwargs = {"root": str(data_root), "split_dir": str(target_dir), "image_size": int(image_size), "num_classes": int(num_classes), "keep_aspect": bool(keep_aspect)}
        if "transform" in init_params:
            dataset_kwargs["transform"] = _NormalizeTransform(mean, std)
        dataset = DSACADensityH5Dataset(**dataset_kwargs)
    else:
        dataset = DSACADensityH5Dataset(str(target_dir), num_classes=int(num_classes), transform=_NormalizeTransform(mean, std), image_size=int(image_size), keep_aspect=bool(keep_aspect))
    loader_kwargs = {"batch_size": int(batch_size), "shuffle": False, "num_workers": int(num_workers), "pin_memory": True}
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(dataset, **loader_kwargs)

    total_mae = 0.0
    total_rmse = 0.0
    sum_abs = torch.zeros(int(num_classes), dtype=torch.float64)
    sum_sq = torch.zeros(int(num_classes), dtype=torch.float64)
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for index, (images, density) in enumerate(loader):
            if index % 20 == 0:
                print(f"[eval][cnt] processed {index}/{len(loader)}")
            images = images.to(device, non_blocking=True).float()
            density = density.to(device, non_blocking=True).float()
            _, pred_counts = model.forward_cnt(images)
            gt_counts = density.flatten(2).sum(dim=2)
            err = (pred_counts - gt_counts).to(torch.float64)
            sum_abs += err.abs().sum(dim=0).cpu()
            sum_sq += (err * err).sum(dim=0).cpu()
            total_err = (pred_counts.sum(dim=1) - gt_counts.sum(dim=1)).to(torch.float64)
            total_mae += float(total_err.abs().sum().item())
            total_rmse += float((total_err ** 2).sum().item())
            num_samples += int(images.size(0))
    denom = max(num_samples, 1)
    per_class_mae = sum_abs / denom
    per_class_rmse = (sum_sq / denom).sqrt()
    split_name = Path(target_dir).name or "count"
    metrics = {
        "task": "counting",
        "data_root": str(Path(data_root)),
        "test_dir": str(Path(target_dir)),
        "image_size": int(image_size),
        "num_classes": int(num_classes),
        "keep_aspect": bool(keep_aspect),
        "per_class_mae": [float(x.item()) for x in per_class_mae],
        "per_class_rmse": [float(x.item()) for x in per_class_rmse],
        "mae": float(per_class_mae.mean().item()),
        "rmse": float(per_class_rmse.mean().item()),
        "mae_total": float(total_mae / denom),
        "rmse_total": float((total_rmse / denom) ** 0.5),
        "mae_per_class_mean": float(per_class_mae.mean().item()),
        "num_samples": int(num_samples),
        "n_images": int(num_samples),
    }
    metrics_path = stats_dir / f"metrics_cnt_{split_name}.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(f"[eval][cnt] saved metrics: {metrics_path}")
    return metrics


def main():
    args = parse_args()
    tasks = _parse_tasks(args.tasks)
    stats_dir = _default_stats_dir(args.checkpoint, args.stats_dir)
    model, config, device = _build_model_from_checkpoint(args.checkpoint, device=args.device, model_name=args.model_name, image_size=args.image_size)
    meta = {"checkpoint": str(Path(args.checkpoint).resolve()), "device": str(device), "model_config": config}
    meta_path = stats_dir / "checkpoint_meta.json"
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    print(f"[eval] saved checkpoint meta: {meta_path}")
    if args.check_load_only:
        print("[eval] check-load-only complete")
        return

    results = {"meta": meta, "results": {}}
    if "det" in tasks:
        det_data_root = args.det_data_root
        det_ann_file = args.det_ann_file or (str(Path(det_data_root) / "annotations" / "instances_test.json") if det_data_root else None)
        det_img_dir = args.det_img_dir or (str(Path(det_data_root) / "images" / "test") if det_data_root else None)
        if not det_ann_file or not det_img_dir:
            raise SystemExit("Detection evaluation requires --det-data-root or both --det-ann-file and --det-img-dir")
        results["results"]["det"] = _eval_full_model_det(model, stats_dir=stats_dir, ann_file=det_ann_file, img_dir=det_img_dir, device=device, score_thr=float(args.det_score_thr), use_coco_eval=bool(args.det_use_coco_eval), batch_size=int(args.det_batch_size), num_workers=int(args.det_num_workers))
    if "seg" in tasks:
        if not args.seg_data_dir:
            raise SystemExit("Segmentation evaluation requires --seg-data-dir")
        seg_num_classes = int(args.seg_num_classes) if args.seg_num_classes is not None else int(config["seg_num_classes"])
        results["results"]["seg"] = _eval_full_model_seg(model, stats_dir=stats_dir, data_dir=args.seg_data_dir, device=device, num_classes=seg_num_classes, image_size=int(config["image_size"]))
    if "cnt" in tasks:
        if not args.cnt_data_root:
            raise SystemExit("Counting evaluation requires --cnt-data-root")
        cnt_num_classes = int(args.cnt_num_classes) if args.cnt_num_classes is not None else int(config["cnt_num_classes"])
        results["results"]["cnt"] = _eval_full_model_cnt(model, stats_dir=stats_dir, data_root=args.cnt_data_root, test_dir=args.cnt_test_dir, device=device, num_classes=cnt_num_classes, image_size=int(config["image_size"]), keep_aspect=bool(args.cnt_keep_aspect), batch_size=int(args.cnt_batch_size), num_workers=int(args.cnt_num_workers))
    results_path = stats_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    print(f"[eval] saved combined results: {results_path}")


if __name__ == "__main__":
    main()

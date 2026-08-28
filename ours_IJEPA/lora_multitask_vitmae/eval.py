from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision.transforms.functional as TF


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


def _is_full_shared_state(shared_state: Mapping) -> bool:
    return any(isinstance(k, str) and k.startswith("backbone.") for k in shared_state.keys())


def _state_dict_has_lora(state_dict: Mapping) -> bool:
    return any(
        isinstance(k, str) and (".lora_a" in k.lower() or ".lora_b" in k.lower())
        for k in state_dict.keys()
    )


def _parse_tasks(text: str) -> List[str]:
    items = [s.strip().lower() for s in (text or "").split(",") if s.strip()]
    valid = {"det", "seg", "cnt"}
    bad = [item for item in items if item not in valid]
    if bad:
        raise SystemExit(f"Invalid task(s): {bad}; valid values are det, seg, cnt")
    out: List[str] = []
    for item in items:
        if item not in out:
            out.append(item)
    return out


def _default_stats_dir(checkpoint: str, stats_dir: Optional[str]) -> Path:
    if stats_dir:
        out = Path(stats_dir)
    else:
        out = Path(checkpoint).resolve().parent / "stats"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _extract_artifacts(output: str) -> Dict[str, str]:
    artifacts: Dict[str, str] = {}
    for line in output.splitlines():
        s = line.strip()
        if s.startswith("Saved metrics to "):
            artifacts["metrics"] = s[len("Saved metrics to ") :].strip()
        elif s.startswith("Saved predictions to "):
            artifacts["predictions"] = s[len("Saved predictions to ") :].strip()
    return artifacts


def _run_cmd(cmd: List[str], *, cwd: Path, env: Optional[dict] = None) -> Tuple[int, str]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    lines: List[str] = []
    for line in proc.stdout:
        sys.stdout.write(line)
        lines.append(line)
    proc.wait()
    return int(proc.returncode), "".join(lines)


def _maybe_write_compat_checkpoint(checkpoint_path: str, *, stats_dir: Path) -> str:
    ckpt = _torch_load_cpu(checkpoint_path)
    if not _is_multitask_checkpoint(ckpt):
        return checkpoint_path

    backbone_state = ckpt.get("backbone")
    if not isinstance(backbone_state, dict) or not _is_full_shared_state(backbone_state):
        return checkpoint_path

    stripped = {
        k[len("backbone.") :]: v
        for k, v in backbone_state.items()
        if isinstance(k, str) and k.startswith("backbone.")
    }
    if not stripped:
        return checkpoint_path

    compat = dict(ckpt)
    compat["backbone"] = stripped
    compat["compat_note"] = "auto-generated for legacy single-task eval"
    compat_path = stats_dir / "compat_multitask_ckpt.pt"
    torch.save(compat, compat_path)
    print(f"[multitask/eval] wrote compat checkpoint: {compat_path}")
    return str(compat_path)


def _infer_fg_num_classes_from_det_state(det_state: Dict[str, Any]) -> int:
    weight = det_state.get("roi_heads.box_predictor.cls_score.weight")
    if hasattr(weight, "shape") and len(weight.shape) >= 1 and int(weight.shape[0]) >= 2:
        return int(weight.shape[0]) - 1
    raise SystemExit("Could not infer detection class count from checkpoint")


def _infer_num_classes_from_head_state(state: Dict[str, Any], key: str) -> int:
    weight = state.get(key)
    if hasattr(weight, "shape") and len(weight.shape) >= 1:
        return int(weight.shape[0])
    raise SystemExit(f"Could not infer class count from checkpoint key: {key}")


def _infer_det_out_channels(det_state: Dict[str, Any], default: int = 256) -> int:
    weight = det_state.get("backbone.proj.weight")
    if hasattr(weight, "shape") and len(weight.shape) >= 1:
        return int(weight.shape[0])
    return int(default)


def _infer_lora_config(ckpt: Dict[str, Any], shared_state: Dict[str, Any]) -> Dict[str, Any]:
    cfg = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
    lora_meta = ckpt.get("lora") if isinstance(ckpt.get("lora"), dict) else {}
    use_lora = bool(cfg.get("use_lora", False)) or bool(lora_meta) or _state_dict_has_lora(shared_state)
    return {
        "use_lora": bool(use_lora),
        "lora_rank": int(cfg.get("lora_rank", lora_meta.get("rank", 8))),
        "lora_alpha": float(cfg.get("lora_alpha", lora_meta.get("alpha", 16.0))),
        "lora_dropout": float(cfg.get("lora_dropout", lora_meta.get("dropout", 0.05))),
    }


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _import_multitask_models():
    workspace_root = _workspace_root()
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))
    from ours_IJEPA.lora_multitask_vitmae.models import MultiTaskModel, SharedViTMAEBackboneWithLoRA

    return MultiTaskModel, SharedViTMAEBackboneWithLoRA


def _build_full_multitask_model_from_ckpt(
    checkpoint_path: str,
    *,
    device: str | None,
    model_name: str | None,
    image_size: int | None,
):
    ckpt = _torch_load_cpu(checkpoint_path)
    if not _is_multitask_checkpoint(ckpt):
        raise SystemExit(f"Checkpoint is not a multitask checkpoint: {checkpoint_path}")

    shared_state = ckpt["backbone"]
    det_state = ckpt["det_head"]
    seg_state = ckpt["seg_head"]
    cnt_state = ckpt["cnt_head"]
    if not isinstance(shared_state, dict):
        raise SystemExit("Checkpoint backbone entry is not a state_dict")

    det_fg = _infer_fg_num_classes_from_det_state(det_state)
    seg_nc = _infer_num_classes_from_head_state(seg_state, "decode.3.weight")
    cnt_nc = _infer_num_classes_from_head_state(cnt_state, "decode.3.weight")
    det_out_channels = _infer_det_out_channels(det_state)
    lora_cfg = _infer_lora_config(ckpt, shared_state)

    name = model_name or "ijepa_vit_huge_patch16"
    img_size = int(image_size) if image_size is not None else 448
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        dev_index = dev.index if dev.index is not None else 0
        torch.cuda.set_device(dev_index)
        dev = torch.device(f"cuda:{dev_index}")

    MultiTaskModel, SharedViTMAEBackboneWithLoRA = _import_multitask_models()
    shared = SharedViTMAEBackboneWithLoRA(
        model_name=name,
        image_size=img_size,
        checkpoint_path=None,
        use_lora=bool(lora_cfg["use_lora"]),
        lora_rank=int(lora_cfg["lora_rank"]),
        lora_alpha=float(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
    )
    model = MultiTaskModel(
        shared=shared,
        det_num_classes=int(det_fg),
        seg_num_classes=int(seg_nc),
        cnt_num_classes=int(cnt_nc),
        image_size=img_size,
        det_out_channels=int(det_out_channels),
        det_train_backbone=True,
        seg_train_backbone=True,
        cnt_train_backbone=True,
    ).to(dev)
    model.eval()

    if _is_full_shared_state(shared_state):
        missing_s, unexpected_s = model.shared.load_state_dict(shared_state, strict=False)
        total_s = len(model.shared.state_dict())
        matched_s = max(total_s - len(missing_s), 0)
        print(
            "[multitask/eval] shared(full): "
            f"total={total_s}, matched={matched_s} ({(100.0 * matched_s / max(total_s, 1)):.1f}%), "
            f"missing={len(missing_s)}, unexpected={len(unexpected_s)}"
        )
    else:
        missing_s, unexpected_s = model.shared.backbone.load_state_dict(shared_state, strict=False)
        total_s = len(model.shared.backbone.state_dict())
        matched_s = max(total_s - len(missing_s), 0)
        print(
            "[multitask/eval] shared(legacy backbone): "
            f"total={total_s}, matched={matched_s} ({(100.0 * matched_s / max(total_s, 1)):.1f}%), "
            f"missing={len(missing_s)}, unexpected={len(unexpected_s)}"
        )

    missing_d, unexpected_d = model.detector.load_state_dict(det_state, strict=False)
    expected_missing = [
        k for k in missing_d if isinstance(k, str) and k.startswith("backbone.shared.")
    ]
    real_missing = [
        k for k in missing_d if not (isinstance(k, str) and k.startswith("backbone.shared."))
    ]
    print(
        "[multitask/eval] detector: "
        f"missing={len(missing_d)} "
        f"(expected_shared_missing={len(expected_missing)}, real_missing={len(real_missing)}), "
        f"unexpected={len(unexpected_d)}"
    )
    missing_seg, unexpected_seg = model.seg_head.load_state_dict(seg_state, strict=False)
    missing_cnt, unexpected_cnt = model.cnt_head.load_state_dict(cnt_state, strict=False)
    print(f"[multitask/eval] seg_head: missing={len(missing_seg)}, unexpected={len(unexpected_seg)}")
    print(f"[multitask/eval] cnt_head: missing={len(missing_cnt)}, unexpected={len(unexpected_cnt)}")

    meta = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "model_name": str(name),
        "image_size": int(img_size),
        "device": str(dev),
        "det_fg_classes": int(det_fg),
        "det_out_channels": int(det_out_channels),
        "seg_num_classes": int(seg_nc),
        "cnt_num_classes": int(cnt_nc),
        "use_lora": bool(lora_cfg["use_lora"]),
        "lora_cfg": dict(lora_cfg),
    }
    return model, meta, dev


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
) -> Dict[str, Any]:
    from object_detection.dataset import CocoDetectionDataset, collate_fn
    from object_detection.utils import compute_precision_recall

    try:
        from pycocotools.coco import COCO  # type: ignore
        from pycocotools.cocoeval import COCOeval  # type: ignore
    except Exception:
        COCO = None
        COCOeval = None

    ds = CocoDetectionDataset(str(ann_file), str(img_dir), transform=lambda img, tgt: (TF.to_tensor(img), tgt))
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    use_coco = bool(use_coco_eval and COCO is not None and COCOeval is not None)
    coco_results: List[Dict[str, Any]] = []
    total_tp = total_fp = total_fn = 0
    model.eval()
    with torch.no_grad():
        for idx, (images, targets) in enumerate(loader):
            images = [img.to(device, non_blocking=True) for img in images]
            outputs = model.detector(images)
            if use_coco:
                for out, tgt in zip(outputs, targets):
                    image_id = int(tgt["image_id"])
                    boxes = out["boxes"].detach().cpu()
                    labels = out["labels"].detach().cpu()
                    scores = out["scores"].detach().cpu()
                    keep = scores >= float(score_thr)
                    for box, label, score in zip(boxes[keep], labels[keep], scores[keep]):
                        x1, y1, x2, y2 = box.tolist()
                        coco_results.append(
                            {
                                "image_id": image_id,
                                "category_id": int(ds.label_to_cat_id[int(label)]),
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "score": float(score),
                            }
                        )
            else:
                targets_dev = [
                    {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in tgt.items()}
                    for tgt in targets
                ]
                tp, fp, fn = compute_precision_recall(outputs, targets_dev, score_threshold=float(score_thr))
                total_tp += int(tp)
                total_fp += int(fp)
                total_fn += int(fn)

            if (idx + 1) % 50 == 0:
                print(f"[multitask/eval][det] processed {idx + 1}/{len(loader)}")

    split_name = Path(ann_file).stem
    if use_coco:
        coco_gt = COCO(str(ann_file))
        coco_gt.dataset.setdefault("info", {})
        coco_gt.dataset.setdefault("licenses", [])
        stats: List[float] = []
        if coco_results:
            coco_dt = coco_gt.loadRes(coco_results)
            coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            stats = [float(x) for x in getattr(coco_eval, "stats", [])]
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
        metrics: Dict[str, Any] = {
            name: stats[i] if i < len(stats) else None for i, name in enumerate(metric_names)
        }
        metrics.update(
            {
                "task": "detection",
                "ann_file": str(ann_file),
                "img_dir": str(img_dir),
                "score_thr": float(score_thr),
            }
        )
        pred_path = stats_dir / f"predictions_det_{split_name}.json"
        with pred_path.open("w", encoding="utf-8") as f:
            json.dump(coco_results, f)
        print(f"Saved predictions to {pred_path}")
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
        }

    metrics_path = stats_dir / f"metrics_det_{split_name}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {metrics_path}")
    return metrics


def _eval_full_model_seg(
    model,
    *,
    stats_dir: Path,
    data_dir: str,
    device: torch.device,
    num_classes: int,
    image_size: int,
) -> Dict[str, Any]:
    from segmentation.dataset import SegmentationDataset
    from segmentation.utils import per_class_iou_from_confusion, update_confusion_matrix

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def _transform(img, mask):
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=mean, std=std)
        mask = torch.as_tensor(TF.pil_to_tensor(mask), dtype=torch.long).squeeze(0)
        return img, mask

    ds = SegmentationDataset(data_dir, transform=_transform, image_size=int(image_size))
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    conf = torch.zeros((int(num_classes), int(num_classes)), dtype=torch.int64)
    model.eval()
    with torch.no_grad():
        for idx, (imgs, masks) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model.forward_seg(imgs)
            update_confusion_matrix(
                conf=conf,
                logits_or_preds=logits.detach(),
                target=masks.detach(),
                num_classes=int(num_classes),
                ignore_indices=(255, 11),
            )
            if idx % 10 == 0:
                print(f"[multitask/eval][seg] processed {idx}/{len(loader)}")

    per_class_iou, miou = per_class_iou_from_confusion(conf)
    split_name = Path(data_dir).name or "seg"
    metrics = {
        "task": "segmentation",
        "data_dir": str(Path(data_dir)),
        "num_classes": int(num_classes),
        "image_size": int(image_size),
        "per_class_iou": [None if torch.isnan(v) else float(v.item()) for v in per_class_iou],
        "miou": None if torch.isnan(miou) else float(miou.item()),
    }
    metrics_path = stats_dir / f"metrics_seg_{split_name}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {metrics_path}")
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
) -> Dict[str, Any]:
    from counting.dataset import DSACADensityH5Dataset

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    class _NormalizeTransform:
        def __call__(self, img, density):
            if not torch.is_tensor(img):
                img = TF.to_tensor(img)
            img = TF.normalize(img, mean=mean, std=std)
            return img, density

    root = Path(data_root)
    split_dir = Path(test_dir) if test_dir else (root / "test_data_class8")
    ds = DSACADensityH5Dataset(
        str(split_dir),
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
    loader = torch.utils.data.DataLoader(ds, **loader_kwargs)

    sum_abs = torch.zeros(int(num_classes), dtype=torch.float64)
    sum_sq = torch.zeros(int(num_classes), dtype=torch.float64)
    n_images = 0
    model.eval()
    with torch.no_grad():
        for idx, (imgs, dens) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            dens = dens.to(device, non_blocking=True)
            _, pred_counts = model.forward_cnt(imgs)
            gt_counts = dens.flatten(2).sum(dim=2)
            err = (pred_counts - gt_counts).to(torch.float64)
            sum_abs += err.abs().sum(dim=0).cpu()
            sum_sq += (err * err).sum(dim=0).cpu()
            n_images += int(imgs.shape[0])
            if (idx + 1) % 50 == 0:
                print(f"[multitask/eval][cnt] processed {idx + 1}/{len(loader)}")

    mae = sum_abs / max(n_images, 1)
    rmse = (sum_sq / max(n_images, 1)).sqrt()
    split_name = split_dir.name or "count"
    metrics = {
        "task": "counting",
        "data_root": str(root),
        "test_dir": str(split_dir),
        "num_classes": int(num_classes),
        "image_size": int(image_size),
        "keep_aspect": bool(keep_aspect),
        "per_class_mae": [float(x.item()) for x in mae],
        "per_class_rmse": [float(x.item()) for x in rmse],
        "mae": float(mae.mean().item()),
        "rmse": float(rmse.mean().item()),
        "n_images": int(n_images),
    }
    metrics_path = stats_dir / f"metrics_cnt_{split_name}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {metrics_path}")
    return metrics


def _run_full_model_eval(args, tasks: List[str]) -> int:
    if "det" in tasks and not (args.det_data_root and args.det_ann_file and args.det_img_dir):
        raise SystemExit("det task selected but --det-data-root/--det-ann-file/--det-img-dir is missing")
    if "seg" in tasks and not args.seg_data_dir:
        raise SystemExit("seg task selected but --seg-data-dir is missing")
    if "cnt" in tasks and not args.cnt_data_root:
        raise SystemExit("cnt task selected but --cnt-data-root is missing")

    model, meta, device = _build_full_multitask_model_from_ckpt(
        args.checkpoint,
        device=args.device,
        model_name=args.model_name,
        image_size=args.image_size,
    )
    stats_dir = _default_stats_dir(args.checkpoint, args.stats_dir)
    results: Dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "stats_dir": str(stats_dir),
        "tasks": list(tasks),
        "meta": meta,
        "results": {},
    }

    if "det" in tasks:
        results["results"]["det"] = _eval_full_model_det(
            model,
            stats_dir=stats_dir,
            ann_file=args.det_ann_file,
            img_dir=args.det_img_dir,
            device=device,
            score_thr=float(args.det_score_thr) if args.det_score_thr is not None else 0.0,
            use_coco_eval=bool(args.det_use_coco_eval),
            batch_size=int(args.det_batch_size),
            num_workers=int(args.det_num_workers),
        )

    if "seg" in tasks:
        results["results"]["seg"] = _eval_full_model_seg(
            model,
            stats_dir=stats_dir,
            data_dir=args.seg_data_dir,
            device=device,
            num_classes=int(args.seg_num_classes) if args.seg_num_classes is not None else int(meta["seg_num_classes"]),
            image_size=int(args.image_size) if args.image_size is not None else int(meta["image_size"]),
        )

    if "cnt" in tasks:
        results["results"]["cnt"] = _eval_full_model_cnt(
            model,
            stats_dir=stats_dir,
            data_root=args.cnt_data_root,
            test_dir=args.cnt_test_dir,
            device=device,
            num_classes=int(args.cnt_num_classes) if args.cnt_num_classes is not None else int(meta["cnt_num_classes"]),
            image_size=int(args.image_size) if args.image_size is not None else int(meta["image_size"]),
            keep_aspect=True if args.cnt_keep_aspect is None else bool(args.cnt_keep_aspect),
            batch_size=int(args.cnt_batch_size),
            num_workers=int(args.cnt_num_workers),
        )

    summary_path = stats_dir / "multitask_full_model_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[multitask/eval] wrote full-model summary: {summary_path}")
    return 0


def _build_det_cmd(args, *, python: str, stats_dir: Path) -> List[str]:
    cmd = [python, "object_detection/eval.py", "--checkpoint", args.checkpoint, "--stats-dir", str(stats_dir)]
    if args.det_data_root:
        cmd += ["--data-root", args.det_data_root]
    if args.det_ann_file:
        cmd += ["--ann-file", args.det_ann_file]
    if args.det_img_dir:
        cmd += ["--img-dir", args.det_img_dir]
    if args.det_num_classes is not None:
        cmd += ["--num-classes", str(args.det_num_classes)]
    if args.image_size is not None:
        cmd += ["--image-size", str(args.image_size)]
    if args.model_name is not None:
        cmd += ["--model-name", args.model_name]
    if args.device is not None:
        cmd += ["--device", args.device]
    if args.det_backbone_checkpoint:
        cmd += ["--backbone-checkpoint", args.det_backbone_checkpoint]
    if args.det_backbone_source:
        cmd += ["--backbone-source", args.det_backbone_source]
    if args.det_out_channels is not None:
        cmd += ["--out-channels", str(args.det_out_channels)]
    if args.det_score_thr is not None:
        cmd += ["--score-thr", str(args.det_score_thr)]
    if args.det_use_coco_eval:
        cmd += ["--use-coco-eval"]
    if args.det_batch_size is not None:
        cmd += ["--batch-size", str(args.det_batch_size)]
    if args.det_num_workers is not None:
        cmd += ["--num-workers", str(args.det_num_workers)]
    return cmd


def _build_seg_cmd(args, *, python: str, stats_dir: Path) -> List[str]:
    cmd = [python, "segmentation/eval.py", "--checkpoint", args.checkpoint, "--stats-dir", str(stats_dir)]
    if args.seg_data_dir:
        cmd += ["--data-dir", args.seg_data_dir]
    if args.seg_num_classes is not None:
        cmd += ["--num-classes", str(args.seg_num_classes)]
    if args.image_size is not None:
        cmd += ["--image-size", str(args.image_size)]
    if args.model_name is not None:
        cmd += ["--model-name", args.model_name]
    if args.device is not None:
        cmd += ["--device", args.device]
    if args.seg_backbone_checkpoint:
        cmd += ["--backbone-checkpoint", args.seg_backbone_checkpoint]
    if args.seg_save_preds:
        cmd += ["--save-preds", args.seg_save_preds]
    if args.seg_vis_dir:
        cmd += ["--vis-dir", args.seg_vis_dir]
    return cmd


def _build_cnt_cmd(args, *, python: str, stats_dir: Path) -> List[str]:
    cmd = [python, "counting/eval.py", "--checkpoint", args.checkpoint, "--stats-dir", str(stats_dir)]
    if args.cnt_data_root:
        cmd += ["--data-root", args.cnt_data_root]
    if args.cnt_test_dir:
        cmd += ["--test-dir", args.cnt_test_dir]
    if args.cnt_num_classes is not None:
        cmd += ["--num-classes", str(args.cnt_num_classes)]
    if args.image_size is not None:
        cmd += ["--image-size", str(args.image_size)]
    if args.model_name is not None:
        cmd += ["--model-name", args.model_name]
    if args.device is not None:
        cmd += ["--device", args.device]
    if args.cnt_backbone_checkpoint:
        cmd += ["--backbone-checkpoint", args.cnt_backbone_checkpoint]
    if args.cnt_backbone_source:
        cmd += ["--backbone-source", args.cnt_backbone_source]
    if args.cnt_keep_aspect is True:
        cmd += ["--keep-aspect"]
    elif args.cnt_keep_aspect is False:
        cmd += ["--no-keep-aspect"]
    if args.cnt_batch_size is not None:
        cmd += ["--batch-size", str(args.cnt_batch_size)]
    if args.cnt_num_workers is not None:
        cmd += ["--num-workers", str(args.cnt_num_workers)]
    return cmd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an IJEPA-backbone multitask checkpoint via full-model loading or legacy single-task wrappers"
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--task", type=str, default=None, choices=["det", "seg", "cnt"])
    parser.add_argument("--tasks", type=str, default="det,seg,cnt")
    parser.add_argument("--stats-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--det-data-root", type=str, default=None)
    parser.add_argument("--det-ann-file", type=str, default=None)
    parser.add_argument("--det-img-dir", type=str, default=None)
    parser.add_argument("--det-num-classes", type=int, default=None)
    parser.add_argument("--det-out-channels", type=int, default=None)
    parser.add_argument("--det-score-thr", type=float, default=None)
    parser.add_argument("--det-use-coco-eval", action="store_true")
    parser.add_argument("--det-backbone-checkpoint", type=str, default=None)
    parser.add_argument("--det-backbone-source", type=str, default=None, choices=["auto", "pretrained", "det_checkpoint"])
    parser.add_argument("--det-batch-size", type=int, default=32)
    parser.add_argument("--det-num-workers", type=int, default=2)

    parser.add_argument("--seg-data-dir", type=str, default=None)
    parser.add_argument("--seg-num-classes", type=int, default=None)
    parser.add_argument("--seg-backbone-checkpoint", type=str, default=None)
    parser.add_argument("--seg-save-preds", type=str, default=None)
    parser.add_argument("--seg-vis-dir", type=str, default=None)

    parser.add_argument("--cnt-data-root", type=str, default=None)
    parser.add_argument("--cnt-test-dir", type=str, default=None)
    parser.add_argument("--cnt-num-classes", type=int, default=None)
    aspect = parser.add_mutually_exclusive_group()
    aspect.add_argument("--cnt-keep-aspect", dest="cnt_keep_aspect", action="store_true")
    aspect.add_argument("--cnt-no-keep-aspect", dest="cnt_keep_aspect", action="store_false")
    parser.set_defaults(cnt_keep_aspect=None)
    parser.add_argument("--cnt-backbone-checkpoint", type=str, default=None)
    parser.add_argument("--cnt-backbone-source", type=str, default=None, choices=["auto", "pretrained", "ckpt"])
    parser.add_argument("--cnt-batch-size", type=int, default=32)
    parser.add_argument("--cnt-num-workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = [args.task] if args.task else _parse_tasks(args.tasks)
    if not tasks:
        raise SystemExit("No tasks selected")

    ckpt = _torch_load_cpu(args.checkpoint)
    use_full_model_eval = False
    if _is_multitask_checkpoint(ckpt):
        shared_state = ckpt["backbone"]
        use_full_model_eval = _is_full_shared_state(shared_state)

    if use_full_model_eval:
        if args.dry_run:
            print("[multitask/eval] dry-run: would run full-model evaluation")
            return 0
        return _run_full_model_eval(args, tasks)

    repo_root = _workspace_root()
    python = sys.executable
    stats_dir = _default_stats_dir(args.checkpoint, args.stats_dir)
    base_env = os.environ.copy()
    existing_pythonpath = base_env.get("PYTHONPATH", "")
    base_env["PYTHONPATH"] = f"{repo_root}:{existing_pythonpath}" if existing_pythonpath else str(repo_root)
    args.checkpoint = _maybe_write_compat_checkpoint(args.checkpoint, stats_dir=stats_dir)

    if "det" in tasks and not args.det_data_root:
        raise SystemExit("det task selected but --det-data-root is missing")
    if "seg" in tasks and not args.seg_data_dir:
        raise SystemExit("seg task selected but --seg-data-dir is missing")
    if "cnt" in tasks and not args.cnt_data_root:
        raise SystemExit("cnt task selected but --cnt-data-root is missing")

    summary: Dict[str, Any] = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "stats_dir": str(stats_dir),
        "tasks": list(tasks),
        "runs": {},
    }
    runners = {"det": _build_det_cmd, "seg": _build_seg_cmd, "cnt": _build_cnt_cmd}

    for task in tasks:
        cmd = runners[task](args, python=python, stats_dir=stats_dir)
        print(f"\n[multitask/eval] running {task}: {shlex.join(cmd)}\n")
        if args.dry_run:
            summary["runs"][task] = {"cmd": cmd, "dry_run": True}
            continue

        started = time.time()
        rc, output = _run_cmd(cmd, cwd=repo_root, env=base_env)
        elapsed = time.time() - started
        summary["runs"][task] = {
            "cmd": cmd,
            "returncode": int(rc),
            "elapsed_sec": float(elapsed),
            "artifacts": _extract_artifacts(output),
        }
        if rc != 0:
            summary_path = stats_dir / "multitask_eval_summary.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"[multitask/eval] FAILED (task={task}, rc={rc}); wrote summary to {summary_path}")
            return rc

    summary_path = stats_dir / "multitask_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[multitask/eval] done; summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

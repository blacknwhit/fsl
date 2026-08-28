from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF


# Ensure repo root (new_fscd) is on sys.path so imports like
# `object_detection.*`, `segmentation.*`, `counting.*`, `my_mod_squad.*` work
# even when running from an arbitrary cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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
    w = det_state.get("roi_heads.box_predictor.cls_score.weight")
    if hasattr(w, "shape") and len(getattr(w, "shape", [])) >= 1:
        total = int(w.shape[0])
        if total >= 2:
            return total - 1
    raise ValueError("Cannot infer det fg num_classes from det_head (missing roi_heads.box_predictor.cls_score.weight)")


def _infer_num_classes_from_conv1x1_weight(state: Mapping, weight_key: str) -> int:
    w = state.get(weight_key)
    if hasattr(w, "shape") and len(getattr(w, "shape", [])) >= 1:
        return int(w.shape[0])
    raise ValueError(f"Cannot infer num_classes from key {weight_key} in head state_dict")


def _infer_det_out_channels(det_state: Mapping, default: int = 256) -> int:
    w = det_state.get("backbone.proj.weight")
    if hasattr(w, "shape") and len(getattr(w, "shape", [])) >= 1:
        return int(w.shape[0])
    return int(default)


def _state_dict_has_lora(state_dict: Mapping) -> bool:
    return any(
        isinstance(k, str) and (".lora_a" in k.lower() or ".lora_b" in k.lower())
        for k in state_dict.keys()
    )


def _infer_plain_lora_config(ckpt: Mapping, shared_state: Mapping) -> Dict[str, object]:
    cfg = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
    lora_meta = ckpt.get("lora") if isinstance(ckpt.get("lora"), dict) else {}
    use_lora = bool(cfg.get("use_lora", False)) or bool(lora_meta) or _state_dict_has_lora(shared_state)
    return {
        "use_lora": bool(use_lora),
        "lora_rank": int(cfg.get("lora_rank", lora_meta.get("rank", 8))),
        "lora_alpha": float(cfg.get("lora_alpha", lora_meta.get("alpha", 16.0))),
        "lora_dropout": float(cfg.get("lora_dropout", lora_meta.get("dropout", 0.05))),
    }


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

    def _config_bool(key: str, default: bool) -> bool:
        if not isinstance(ckpt_config, dict) or key not in ckpt_config:
            return bool(default)
        value = ckpt_config.get(key)
        if isinstance(value, str):
            return value.strip().lower() not in {"0", "false", "no", "off"}
        return bool(value)

    def _first_tensor(suffix: str):
        preferred = (
            f"lora_moes.0.{suffix}",
            f"wrapped_blocks.0.lora_moe.{suffix}",
        )
        for key in preferred:
            value = shared_state.get(key)
            if value is not None:
                return value
        for key, value in shared_state.items():
            if isinstance(key, str) and key.endswith(f".{suffix}"):
                return value
        return None

    default_moe_k_private = _config_int("moe_k_private", 2)
    default_moe_k_shared = _config_int("moe_k_shared", 2)

    has_lora_moes = any(isinstance(k, str) and k.startswith("lora_moes.") for k in shared_state.keys())
    has_wrapped = any(isinstance(k, str) and k.startswith("wrapped_blocks.") for k in shared_state.keys())
    has_private = any(isinstance(k, str) and k.endswith(".lora_A_private") for k in shared_state.keys())
    has_shared = any(isinstance(k, str) and k.endswith(".lora_A_shared") for k in shared_state.keys())
    use_lora_moe = bool(has_lora_moes or has_wrapped)
    if not use_lora_moe:
        return {
            "use_lora_moe": False,
            "use_private_experts": False,
            "use_shared_experts": False,
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

    A_private = _first_tensor("lora_A_private")
    if hasattr(A_private, "shape") and len(getattr(A_private, "shape", [])) == 4:
        task_num = int(A_private.shape[0])
        num_experts_private = int(A_private.shape[1])
        lora_rank = int(A_private.shape[3])

    A_shared = _first_tensor("lora_A_shared")
    if hasattr(A_shared, "shape") and len(getattr(A_shared, "shape", [])) == 3:
        num_experts_shared = int(A_shared.shape[0])
        lora_rank = int(A_shared.shape[2])

    use_private_experts = _config_bool("use_private_experts", has_private)
    use_shared_experts = _config_bool("use_shared_experts", has_shared)

    gate_indices: List[int] = []
    for k in shared_state.keys():
        if not isinstance(k, str):
            continue
        if not k.endswith(".weight"):
            continue
        parts = k.split(".")
        for gate_name in ("f_gate_private", "f_gate_shared"):
            if gate_name in parts:
                pos = parts.index(gate_name) + 1
                if pos < len(parts):
                    try:
                        gate_indices.append(int(parts[pos]))
                    except Exception:
                        pass
                break
    if gate_indices:
        task_num = max(gate_indices) + 1

    return {
        "use_lora_moe": True,
        "use_private_experts": bool(use_private_experts),
        "use_shared_experts": bool(use_shared_experts),
        "task_num": int(task_num),
        "lora_rank": int(lora_rank),
        "num_experts_private": int(num_experts_private),
        "num_experts_shared": int(num_experts_shared),
        "moe_k_private": default_moe_k_private,
        "moe_k_shared": default_moe_k_shared,
    }


def _parse_tasks(text: str) -> List[str]:
    items = [s.strip().lower() for s in (text or "").split(",") if s.strip()]
    valid = {"det", "seg", "cnt"}
    bad = [t for t in items if t not in valid]
    if bad:
        raise ValueError(f"--tasks invalid: {bad}, valid: {sorted(valid)}")
    out: List[str] = []
    for t in items:
        if t not in out:
            out.append(t)
    return out


def _import_multitask_models():
    try:
        from .models import MultiTaskModel, SharedDinoV3Backbone
    except Exception:
        try:
            from models import MultiTaskModel, SharedDinoV3Backbone
        except Exception:
            module_root = Path(__file__).resolve().parent
            if str(module_root) not in sys.path:
                sys.path.insert(0, str(module_root))
            from models import MultiTaskModel, SharedDinoV3Backbone
    return MultiTaskModel, SharedDinoV3Backbone


def _default_stats_dir(checkpoint: str, stats_dir: Optional[str]) -> Path:
    if stats_dir:
        p = Path(stats_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
    ckpt = Path(checkpoint)
    p = ckpt.parent / "stats_eval"
    p.mkdir(parents=True, exist_ok=True)
    return p


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate my_mod_squad/train.py isomorphic multitask model on test sets (det/seg/cnt)"
    )
    p.add_argument("--checkpoint", type=str, required=True, help="multitask checkpoint saved by my_mod_squad/train.py")
    p.add_argument("--tasks", type=str, default="det,seg,cnt")
    p.add_argument("--stats-dir", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument(
        "--backbone-init-image-size",
        type=int,
        default=64,
        help="dummy image size used only during SharedDinoV3Backbone init to infer embed_dim (faster; does not affect eval)",
    )
    p.add_argument("--model-name", type=str, default="dinov3_vitl16")
    p.add_argument("--check-load-only", action="store_true", help="only load model and print load summary")

    # Det (match object_detection/eval.py defaults)
    p.add_argument(
        "--det-data-root",
        type=str,
        default="/nas/liyangguang103/newdataset/CD-ObjectDetection/DIOR/coco",
    )
    p.add_argument(
        "--det-ann-file",
        type=str,
        default=None,
        help="instances_test.json (default: <data-root>/annotations/instances_test.json)",
    )
    p.add_argument(
        "--det-img-dir",
        type=str,
        default=None,
        help="images/test (default: <data-root>/images/test)",
    )
    p.add_argument("--det-score-thr", type=float, default=0.0)
    p.add_argument("--det-use-coco-eval", action="store_true", help="use COCOeval if pycocotools available")
    p.add_argument("--det-batch-size", type=int, default=1)
    p.add_argument("--det-num-workers", type=int, default=2)

    # Seg (match segmentation/eval.py defaults)
    p.add_argument(
        "--seg-data-dir",
        type=str,
        default="/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/test",
    )
    p.add_argument("--seg-num-classes", type=int, default=None)

    # Cnt (match counting/eval.py defaults)
    p.add_argument(
        "--cnt-data-root",
        type=str,
        default="/nas/liyangguang103/newdataset/CD-Count/DSACA",
    )
    p.add_argument("--cnt-test-dir", type=str, default=None)
    p.add_argument("--cnt-num-classes", type=int, default=None)
    p.add_argument("--cnt-batch-size", type=int, default=4)
    p.add_argument("--cnt-num-workers", type=int, default=2)
    aspect = p.add_mutually_exclusive_group()
    aspect.add_argument("--cnt-keep-aspect", dest="cnt_keep_aspect", action="store_true")
    aspect.add_argument("--cnt-no-keep-aspect", dest="cnt_keep_aspect", action="store_false")
    p.set_defaults(cnt_keep_aspect=True)

    return p.parse_args()


@dataclass
class LoadSummary:
    shared_missing: int
    shared_unexpected: int
    det_missing_total: int
    det_expected_backbone_missing: int
    det_real_missing: int
    det_unexpected: int
    seg_missing: int
    seg_unexpected: int
    cnt_missing: int
    cnt_unexpected: int


def _build_and_load_model(args) -> tuple[object, Dict[str, object], LoadSummary]:
    ckpt = _torch_load_cpu(args.checkpoint)
    if not _is_multitask_checkpoint(ckpt):
        raise SystemExit(
            "checkpoint must be a my_mod_squad multitask checkpoint with keys: backbone/det_head/seg_head/cnt_head"
        )

    shared_state = ckpt["backbone"]
    det_state = ckpt["det_head"]
    seg_state = ckpt["seg_head"]
    cnt_state = ckpt["cnt_head"]
    ckpt_config = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else None
    assert isinstance(shared_state, dict) and isinstance(det_state, dict) and isinstance(seg_state, dict) and isinstance(cnt_state, dict)

    det_fg = _infer_fg_num_classes_from_det_state(det_state)
    seg_nc = _infer_num_classes_from_conv1x1_weight(seg_state, "decode.3.weight")
    cnt_nc = _infer_num_classes_from_conv1x1_weight(cnt_state, "decode.3.weight")
    det_out_channels = _infer_det_out_channels(det_state, default=256)

    cfg = _infer_lora_moe_config_from_shared_state(shared_state, ckpt_config)
    use_lora_moe = bool(cfg["use_lora_moe"])
    plain_lora_cfg = _infer_plain_lora_config(ckpt, shared_state)
    use_lora = bool(plain_lora_cfg["use_lora"])
    effective_lora_rank = int(cfg["lora_rank"]) if use_lora_moe else int(plain_lora_cfg["lora_rank"])

    MultiTaskModel, SharedDinoV3Backbone = _import_multitask_models()

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    shared = SharedDinoV3Backbone(
        model_name=str(args.model_name),
        image_size=int(args.backbone_init_image_size),
        checkpoint_path=None,
        use_lora=use_lora,
        use_lora_moe=use_lora_moe,
        task_num=int(cfg["task_num"]),
        lora_rank=effective_lora_rank,
        lora_alpha=float(plain_lora_cfg["lora_alpha"]),
        lora_dropout=float(plain_lora_cfg["lora_dropout"]),
        num_experts_private=int(cfg["num_experts_private"]),
        num_experts_shared=int(cfg["num_experts_shared"]),
        moe_k_private=int(cfg["moe_k_private"]),
        moe_k_shared=int(cfg["moe_k_shared"]),
        use_private_experts=bool(cfg["use_private_experts"]),
        use_shared_experts=bool(cfg["use_shared_experts"]),
        grad_checkpointing=False,
    )

    model = MultiTaskModel(
        shared=shared,
        det_num_classes=int(det_fg),
        seg_num_classes=int(seg_nc),
        cnt_num_classes=int(cnt_nc),
        image_size=int(args.image_size),
        det_out_channels=int(det_out_channels),
        det_train_backbone=True,
        seg_train_backbone=True,
        cnt_train_backbone=True,
    ).to(device)
    model.eval()

    missing_s, unexpected_s = model.shared.load_state_dict(shared_state, strict=False)

    missing_d, unexpected_d = model.detector.load_state_dict(det_state, strict=False)
    expected_prefix = "backbone.shared.backbone."
    expected_missing = [k for k in missing_d if isinstance(k, str) and k.startswith(expected_prefix)]
    real_missing = [k for k in missing_d if not (isinstance(k, str) and k.startswith(expected_prefix))]

    missing_seg, unexpected_seg = model.seg_head.load_state_dict(seg_state, strict=False)
    missing_cnt, unexpected_cnt = model.cnt_head.load_state_dict(cnt_state, strict=False)

    summary = LoadSummary(
        shared_missing=len(missing_s),
        shared_unexpected=len(unexpected_s),
        det_missing_total=len(missing_d),
        det_expected_backbone_missing=len(expected_missing),
        det_real_missing=len(real_missing),
        det_unexpected=len(unexpected_d),
        seg_missing=len(missing_seg),
        seg_unexpected=len(unexpected_seg),
        cnt_missing=len(missing_cnt),
        cnt_unexpected=len(unexpected_cnt),
    )

    meta = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_name": str(args.model_name),
        "image_size": int(args.image_size),
        "device": str(device),
        "use_lora": bool(use_lora),
        "use_lora_moe": bool(use_lora_moe),
        "moe_cfg": cfg,
        "det_fg_classes": int(det_fg),
        "det_out_channels": int(det_out_channels),
        "seg_num_classes": int(args.seg_num_classes) if args.seg_num_classes is not None else int(seg_nc),
        "cnt_num_classes": int(args.cnt_num_classes) if args.cnt_num_classes is not None else int(cnt_nc),
    }

    return model, meta, summary


@torch.no_grad()
def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=boxes1.device)

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


@torch.no_grad()
def _compute_precision_recall(
    predictions: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    iou_threshold: float = 0.5,
    score_threshold: float = 0.05,
) -> Tuple[int, int, int]:
    true_pos = 0
    false_pos = 0
    false_neg = 0

    for pred, tgt in zip(predictions, targets):
        pred_boxes = pred.get("boxes", torch.zeros((0, 4), device=pred["labels"].device))
        pred_labels = pred.get("labels", torch.zeros((0,), dtype=torch.int64, device=pred_boxes.device))
        pred_scores = pred.get("scores", torch.ones((pred_boxes.shape[0],), device=pred_boxes.device))

        keep = pred_scores >= score_threshold
        pred_boxes = pred_boxes[keep]
        pred_labels = pred_labels[keep]
        pred_scores = pred_scores[keep]

        if pred_boxes.numel():
            order = pred_scores.argsort(descending=True)
            pred_boxes = pred_boxes[order]
            pred_labels = pred_labels[order]

        gt_boxes = tgt["boxes"]
        gt_labels = tgt["labels"]
        matched_gt = torch.zeros((gt_boxes.shape[0],), dtype=torch.bool, device=gt_boxes.device)

        for pb, pl in zip(pred_boxes, pred_labels):
            candidates_mask = (gt_labels == pl) & (~matched_gt)
            if not candidates_mask.any():
                false_pos += 1
                continue

            candidates = gt_boxes[candidates_mask]
            ious = _box_iou(pb.unsqueeze(0), candidates).squeeze(0)
            max_iou, max_pos = ious.max(dim=0)
            if max_iou >= iou_threshold:
                global_idx = candidates_mask.nonzero(as_tuple=False)[max_pos].item()
                matched_gt[global_idx] = True
                true_pos += 1
            else:
                false_pos += 1

        false_neg += (~matched_gt).sum().item()

    return true_pos, false_pos, false_neg


def _eval_det(model, *, device: torch.device, stats_dir: Path, args) -> dict:
    from object_detection.dataset import CocoDetectionDataset, collate_fn

    try:
        from pycocotools.coco import COCO  # type: ignore
        from pycocotools.cocoeval import COCOeval  # type: ignore
    except Exception:
        COCO = None
        COCOeval = None

    data_root = Path(args.det_data_root)
    ann_file = Path(args.det_ann_file) if args.det_ann_file else (data_root / "annotations" / "instances_test.json")
    img_dir = Path(args.det_img_dir) if args.det_img_dir else (data_root / "images" / "test")

    ds = CocoDetectionDataset(str(ann_file), str(img_dir), transform=lambda img, tgt: (TF.to_tensor(img), tgt))
    loader = DataLoader(
        ds,
        batch_size=int(args.det_batch_size),
        shuffle=False,
        num_workers=int(args.det_num_workers),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    use_coco = bool(args.det_use_coco_eval and COCO is not None and COCOeval is not None)
    coco_results: List[Dict] = []
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

                    keep = scores >= float(args.det_score_thr)
                    boxes = boxes[keep]
                    labels = labels[keep]
                    scores = scores[keep]

                    for box, label, score in zip(boxes, labels, scores):
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
                    {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in t.items()}
                    for t in targets
                ]
                tp, fp, fn = _compute_precision_recall(outputs, targets_dev, score_threshold=float(args.det_score_thr))
                total_tp += int(tp)
                total_fp += int(fp)
                total_fn += int(fn)

            if (idx + 1) % 50 == 0:
                print(f"[Det] processed {idx+1}/{len(loader)}")

    split_name = ann_file.stem

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
            print("[Det][Warn] no predictions produced; skipping COCOeval")

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
                "score_thr": float(args.det_score_thr),
                "num_predictions": int(len(coco_results)),
                "use_coco_eval": True,
            }
        )

        pred_path = stats_dir / f"predictions_det_{split_name}.json"
        with pred_path.open("w", encoding="utf-8") as f:
            json.dump(coco_results, f)
        print(f"[Det] saved predictions: {pred_path}")
    else:
        precision = float(total_tp / max(total_tp + total_fp, 1))
        recall = float(total_tp / max(total_tp + total_fn, 1))
        metrics = {
            "task": "detection",
            "ann_file": str(ann_file),
            "img_dir": str(img_dir),
            "score_thr": float(args.det_score_thr),
            "precision@0.5": precision,
            "recall@0.5": recall,
            "tp": int(total_tp),
            "fp": int(total_fp),
            "fn": int(total_fn),
            "use_coco_eval": False,
        }

    metrics_path = stats_dir / f"metrics_det_{split_name}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[Det] saved metrics: {metrics_path}")
    return metrics


def _eval_seg(model, *, device: torch.device, stats_dir: Path, args, num_classes: int) -> dict:
    from segmentation.dataset import SegmentationDataset
    from segmentation.utils import per_class_iou_from_confusion, update_confusion_matrix

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def _transform(img, mask):
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=mean, std=std)
        mask_tensor = torch.as_tensor(TF.pil_to_tensor(mask), dtype=torch.long).squeeze(0)
        return img, mask_tensor

    ds = SegmentationDataset(args.seg_data_dir, transform=_transform, image_size=int(args.image_size))
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model.eval()
    conf = torch.zeros((int(num_classes), int(num_classes)), dtype=torch.int64)

    with torch.no_grad():
        for idx, (imgs, masks) in enumerate(loader):
            if idx % 10 == 0:
                print(f"[Seg] processed {idx}/{len(loader)}")
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

    per_class_iou, miou = per_class_iou_from_confusion(conf)
    split_name = Path(args.seg_data_dir).name or "seg"
    metrics = {
        "task": "segmentation",
        "data_dir": str(Path(args.seg_data_dir)),
        "num_classes": int(num_classes),
        "image_size": int(args.image_size),
        "ignore_indices": [255, 11],
        "per_class_iou": [(None if (v != v) else float(v.item())) for v in per_class_iou],
        "miou": (None if (float(miou.item()) != float(miou.item())) else float(miou.item())),
    }
    metrics_path = stats_dir / f"metrics_seg_{split_name}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[Seg] saved metrics: {metrics_path}")
    return metrics


class NormalizeTransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, density):
        if not torch.is_tensor(img):
            img = TF.to_tensor(img)
        img = TF.normalize(img, mean=self.mean, std=self.std)
        return img, density


def _eval_cnt(model, *, device: torch.device, stats_dir: Path, args, num_classes: int) -> dict:
    from counting.dataset import DSACADensityH5Dataset

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    root = Path(args.cnt_data_root)
    test_dir = Path(args.cnt_test_dir) if args.cnt_test_dir else (root / "test_data_class8")

    ds = DSACADensityH5Dataset(
        str(test_dir),
        num_classes=int(num_classes),
        transform=NormalizeTransform(mean, std),
        image_size=int(args.image_size),
        keep_aspect=bool(args.cnt_keep_aspect),
    )

    loader_kwargs = dict(
        batch_size=int(args.cnt_batch_size),
        shuffle=False,
        num_workers=int(args.cnt_num_workers),
        pin_memory=True,
    )
    if int(args.cnt_num_workers) > 0:
        loader_kwargs["persistent_workers"] = True

    loader = DataLoader(ds, **loader_kwargs)

    sum_abs = torch.zeros(int(num_classes), dtype=torch.float64)
    sum_sq = torch.zeros(int(num_classes), dtype=torch.float64)
    n = 0

    model.eval()
    with torch.no_grad():
        for idx, (imgs, dens) in enumerate(loader):
            if (idx + 1) % 50 == 0:
                print(f"[Cnt] processed {idx+1}/{len(loader)}")
            imgs = imgs.to(device, non_blocking=True)
            dens = dens.to(device, non_blocking=True)

            _, pred_counts = model.forward_cnt(imgs)
            gt_counts = dens.flatten(2).sum(dim=2)

            err = (pred_counts - gt_counts).to(torch.float64)
            sum_abs += err.abs().sum(dim=0).cpu()
            sum_sq += (err * err).sum(dim=0).cpu()
            n += int(imgs.shape[0])

    mae = sum_abs / max(1, n)
    rmse = (sum_sq / max(1, n)).sqrt()

    split_name = test_dir.name or "count"
    metrics = {
        "task": "counting",
        "data_root": str(root),
        "test_dir": str(test_dir),
        "num_classes": int(num_classes),
        "image_size": int(args.image_size),
        "keep_aspect": bool(args.cnt_keep_aspect),
        "per_class_mae": [float(x.item()) for x in mae],
        "per_class_rmse": [float(x.item()) for x in rmse],
        "mae": float(mae.mean().item()),
        "rmse": float(rmse.mean().item()),
        "n_images": int(n),
    }
    metrics_path = stats_dir / f"metrics_cnt_{split_name}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[Cnt] saved metrics: {metrics_path}")
    return metrics


def main() -> int:
    args = parse_args()

    tasks = _parse_tasks(args.tasks)
    stats_dir = _default_stats_dir(args.checkpoint, args.stats_dir)

    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device.index)

    model, meta, load_summary = _build_and_load_model(args)

    print("[LoadSummary]")
    print(
        "  shared: "
        f"missing={load_summary.shared_missing}, unexpected={load_summary.shared_unexpected}"
    )
    print(
        "  detector: "
        f"missing_total={load_summary.det_missing_total}, expected_backbone_missing={load_summary.det_expected_backbone_missing}, "
        f"real_missing={load_summary.det_real_missing}, unexpected={load_summary.det_unexpected}"
    )
    print(
        "  seg_head: "
        f"missing={load_summary.seg_missing}, unexpected={load_summary.seg_unexpected}"
    )
    print(
        "  cnt_head: "
        f"missing={load_summary.cnt_missing}, unexpected={load_summary.cnt_unexpected}"
    )

    if args.check_load_only:
        out = {
            "meta": meta,
            "load_summary": load_summary.__dict__,
        }
        out_path = stats_dir / "load_only_summary.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[LoadSummary] wrote: {out_path}")
        return 0

    results: Dict[str, object] = {
        "meta": meta,
        "tasks": tasks,
        "load_summary": load_summary.__dict__,
        "results": {},
    }

    # det
    if "det" in tasks:
        results["results"]["det"] = _eval_det(model, device=device, stats_dir=stats_dir, args=args)

    # seg
    if "seg" in tasks:
        seg_nc = int(args.seg_num_classes) if args.seg_num_classes is not None else int(meta["seg_num_classes"])
        results["results"]["seg"] = _eval_seg(model, device=device, stats_dir=stats_dir, args=args, num_classes=seg_nc)

    # cnt
    if "cnt" in tasks:
        cnt_nc = int(args.cnt_num_classes) if args.cnt_num_classes is not None else int(meta["cnt_num_classes"])
        results["results"]["cnt"] = _eval_cnt(model, device=device, stats_dir=stats_dir, args=args, num_classes=cnt_nc)

    summary_path = stats_dir / "eval_train_model_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Done] wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

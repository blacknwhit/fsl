import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
from collections.abc import Mapping

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from dataset import CocoDetectionDataset, collate_fn
from models import DinoV3FasterRCNN
from utils import compute_precision_recall

try:
    from pycocotools.coco import COCO  # type: ignore
    from pycocotools.cocoeval import COCOeval  # type: ignore
except Exception:
    COCO = None
    COCOeval = None

# Ensure repo root on sys.path so lora_multitask imports succeed when called via subprocess
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def build_transform():
    def _transform(img, target):
        img = TF.to_tensor(img)
        return img, target

    return _transform


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DINOv3 Faster R-CNN baseline")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/nas/liyangguang103/newdataset/CD-ObjectDetection/DIOR/coco",
        help="DIOR COCO root",
    )
    parser.add_argument("--ann-file", type=str, default=None, help="instances_val.json or instances_test.json")
    parser.add_argument("--img-dir", type=str, default=None, help="images/val or images/test")
    parser.add_argument("--checkpoint", type=str, required=True, help="path to trained checkpoint")
    parser.add_argument("--num-classes", type=int, default=None, help="foreground classes (auto if None)")
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--model-name", type=str, default="dinov3_vitl16")
    parser.add_argument(
        "--backbone-checkpoint",
        type=str,
        default="/nas/liyangguang103/old_fscd/CD-FSOD/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    )
    parser.add_argument("--out-channels", type=int, default=256)
    parser.add_argument("--score-thr", type=float, default=0.00)
    parser.add_argument("--use-coco-eval", action="store_true", help="use pycocotools COCOeval if available")

    # 新增：backbone 权重来源控制
    parser.add_argument(
        "--backbone-source",
        type=str,
        default="auto",
        choices=["auto", "pretrained", "det_checkpoint"],
        help=(
            "backbone weights source: "
            "'auto' = use det checkpoint if it contains backbone weights else use --backbone-checkpoint; "
            "'pretrained' = always use --backbone-checkpoint; "
            "'det_checkpoint' = always use backbone from --checkpoint"
        ),
    )

    # 统计目录：默认保存到 checkpoint 同目录下的 stats/
    parser.add_argument(
        "--stats-dir",
        type=str,
        default=None,
        help="directory to save evaluation artifacts (predictions/metrics). Default: <checkpoint_dir>/stats",
    )

    # 仍保留手动指定 json 的入口（若不传，将自动保存到 stats-dir）
    parser.add_argument("--save-json", type=str, default=None, help="save COCO-format predictions to json")

    parser.add_argument(
        "--check-load-only",
        action="store_true",
        help="only build model and load checkpoint, print load summary, then exit (no dataset/eval)",
    )

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def _infer_fg_num_classes_from_det_state(det_state: Mapping) -> int | None:
    # Torchvision FasterRCNN stores cls_score weights with shape [num_classes_total, ...]
    # where num_classes_total includes background.
    w = det_state.get("roi_heads.box_predictor.cls_score.weight")
    if hasattr(w, "shape") and len(getattr(w, "shape", [])) >= 1:
        total = int(w.shape[0])
        if total >= 2:
            return total - 1
    return None


def _extract_state_dict(ckpt_obj):
    # 兼容 save_checkpoint(...) 保存的 {"model": state_dict, ...} 或直接 state_dict
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        return ckpt_obj["model"]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    return None


def _is_multitask_checkpoint(ckpt_obj) -> bool:
    return (
        isinstance(ckpt_obj, dict)
        and isinstance(ckpt_obj.get("backbone"), dict)
        and isinstance(ckpt_obj.get("det_head"), dict)
    )


def _state_dict_has_backbone(state_dict: dict) -> bool:
    # 经验规则：检测常见 backbone key 命名（按你项目里 DinoV3FasterRCNN 的实现可再补充）
    backbone_markers = (
        "backbone.",
        ".backbone.",
        "dino",
        "dinov3",
        "vit",
        "transformer",
    )
    for k in state_dict.keys():
        if any(m in k.lower() for m in backbone_markers):
            # 再收紧一点：必须显式包含 backbone/主干相关字段
            if "backbone" in k.lower() or "dinov3" in k.lower() or k.lower().startswith("backbone."):
                return True
    return False


def _state_dict_has_lora(state_dict: Mapping) -> bool:
    # Deprecated: keep for backward compatibility; prefer _state_dict_has_simple_lora.
    for k in state_dict.keys():
        if isinstance(k, str) and (".lora_a" in k.lower() or ".lora_b" in k.lower()):
            return True
    return False


def _state_dict_has_simple_lora(state_dict: Mapping) -> bool:
    # Our single-task eval injects LoRA into dinov3 FFNs, which creates keys like:
    #   blocks.0.mlp.fc1.lora_A / blocks.0.mlp.fc1.lora_B / ...
    for k in state_dict.keys():
        if not isinstance(k, str):
            continue
        kl = k.lower()
        if kl.startswith("blocks.") and (".lora_a" in kl or ".lora_b" in kl):
            return True
    return False


def _state_dict_has_lora_moe(state_dict: Mapping) -> bool:
    # my_mod_squad checkpoints often store LoRA-MoE under wrapped_blocks.*.lora_moe.*
    for k in state_dict.keys():
        if not isinstance(k, str):
            continue
        kl = k.lower()
        if kl.startswith("wrapped_blocks.") and ".lora_moe." in kl:
            return True
    return False


def _maybe_inject_lora(backbone_module, checkpoint_obj) -> bool:
    """Inject LoRA modules into dinov3 FFNs if checkpoint backbone contains LoRA params.

    Returns True iff injection succeeded (or not needed). False when LoRA was
    detected but injection/import failed.
    """

    if not isinstance(checkpoint_obj, dict):
        return True
    sd = checkpoint_obj.get("backbone")
    if not isinstance(sd, Mapping):
        return True
    if not _state_dict_has_simple_lora(sd):
        if _state_dict_has_lora_moe(sd):
            print(
                "[Eval][Info] Detected LoRA-MoE-style keys in checkpoint backbone (wrapped_blocks.*.lora_moe.*). "
                "Single-task eval only supports simple LoRA (blocks.*.lora_A/B). Skipping LoRA injection."
            )
        return True

    try:
        from lora_multitask.lora import LoRAConfig, inject_lora_into_dinov3_ffn
    except Exception as e:
        print(f"[Eval][Error] Detected LoRA backbone weights but failed to import LoRA utilities: {e}")
        return False

    lora_meta = checkpoint_obj.get("lora") if isinstance(checkpoint_obj.get("lora"), dict) else {}
    cfg = LoRAConfig(
        rank=int(lora_meta.get("rank", 8)),
        alpha=float(lora_meta.get("alpha", 16.0)),
        dropout=float(lora_meta.get("dropout", 0.0)),
    )
    try:
        replaced = inject_lora_into_dinov3_ffn(backbone_module, cfg=cfg)
        print(f"[Eval] Injected LoRA into dinov3 FFN (replaced_linear={replaced})")
        return True
    except Exception as e:
        print(f"[Eval][Error] Failed to inject LoRA modules into backbone: {e}")
        return False


def _filter_state_dict_for_model(state_dict: Mapping, model_state: Mapping) -> tuple[dict, int]:
    """Keep only keys present in model_state and with matching tensor shapes.

    Returns: (filtered_state_dict, dropped_count)
    """

    filtered: dict = {}
    dropped = 0
    for k, v in state_dict.items():
        if k not in model_state:
            dropped += 1
            continue
        ref = model_state[k]
        # Only keep tensor-like entries with matching shape (avoid hard errors)
        if hasattr(v, "shape") and hasattr(ref, "shape") and tuple(v.shape) == tuple(ref.shape):
            filtered[k] = v
        else:
            dropped += 1
    return filtered, dropped


def _prepare_multitask_backbone_state(backbone_state: Mapping, target_state: Mapping) -> tuple[dict, dict]:
    """Prepare multitask checkpoint['backbone'] for loading into the raw dinov3 backbone module."""

    # my_mod_squad saves SharedDinoV3Backbone.state_dict(), which prefixes raw backbone weights with 'backbone.'
    has_prefixed = any(isinstance(k, str) and k.startswith("backbone.") for k in backbone_state.keys())
    if has_prefixed:
        candidate = {k[len("backbone.") :]: v for k, v in backbone_state.items() if isinstance(k, str) and k.startswith("backbone.")}
    else:
        candidate = {k: v for k, v in backbone_state.items() if isinstance(k, str)}

    filtered, dropped = _filter_state_dict_for_model(candidate, target_state)
    report = {
        "provided": int(len(backbone_state)),
        "candidate": int(len(candidate)),
        "kept": int(len(filtered)),
        "dropped": int(dropped),
        "used_prefixed": bool(has_prefixed),
    }
    return filtered, report


def _strip_known_prefixes(k: str, prefixes: tuple[str, ...]) -> str:
    out = k
    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if out.startswith(p):
                out = out[len(p) :]
                changed = True
    return out


def _prepare_det_head_state_for_detector(det_state: Mapping, detector_state: Mapping) -> tuple[dict, dict]:
    """Prepare multitask det_head state for loading into a torchvision-style detector.

    Multitask checkpoints may store det_head as detector.state_dict(), which includes backbone weights.
    For single-task eval we load backbone separately, so here we keep only rpn/roi_heads (and any
    other keys that match by name+shape).

    Returns:
      - filtered_state: keys remapped+filtered to match detector_state
      - report: summary stats for logging
    """

    # Common wrappers/prefixes seen in various save checkpoints.
    prefix_candidates = (
        "module.",
        "model.",
        "detector.",
        "det_head.",
    )

    # 1) strip prefixes
    remapped: dict = {}
    for k, v in det_state.items():
        if not isinstance(k, str):
            continue
        kk = _strip_known_prefixes(k, prefix_candidates)
        remapped[kk] = v

    # 2) Prefer loading only head-related keys to avoid misleading "missing backbone".
    # For our DINOv3 detector backbone wrapper, `backbone.proj.*` is trainable and MUST be loaded,
    # otherwise features into RPN/ROI will be random.
    head_prefixes = ("rpn.", "roi_heads.", "backbone.proj.")
    head_only = {k: v for k, v in remapped.items() if k.startswith(head_prefixes)}
    candidate = head_only if head_only else remapped

    # 3) Filter by existence+shape
    filtered, dropped = _filter_state_dict_for_model(candidate, detector_state)

    report = {
        "provided": int(len(det_state)),
        "after_strip": int(len(remapped)),
        "head_candidates": int(len(head_only)),
        "kept": int(len(filtered)),
        "dropped": int(dropped),
    }
    return filtered, report


def main():
    args = parse_args()
    device = torch.device(args.device)

    # 统计目录（默认：checkpoint 同级 stats/）
    ckpt_path = Path(args.checkpoint)
    stats_dir = Path(args.stats_dir) if args.stats_dir else (ckpt_path.parent / "stats")
    stats_dir.mkdir(parents=True, exist_ok=True)

    # 先加载 checkpoint，用于判断 backbone 权重来源
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    is_multitask = _is_multitask_checkpoint(checkpoint)
    if is_multitask:
        det_state = checkpoint["det_head"]
        has_backbone = True
    else:
        det_state = _extract_state_dict(checkpoint)
        if det_state is None:
            raise ValueError(f"Unrecognized checkpoint format: {args.checkpoint}")
        has_backbone = _state_dict_has_backbone(det_state)

    # In load-check mode, try to infer num_classes from checkpoint to avoid dataset loading.
    if args.check_load_only and args.num_classes is None and isinstance(det_state, Mapping):
        inferred = _infer_fg_num_classes_from_det_state(det_state)
        if inferred is not None:
            args.num_classes = inferred
            print(f"[LoadCheck] inferred --num-classes={args.num_classes} from checkpoint")
    if args.check_load_only and args.num_classes is None:
        raise SystemExit("--check-load-only requires --num-classes (or a checkpoint with box_predictor cls_score)")

    if args.backbone_source == "pretrained":
        backbone_ckpt = args.backbone_checkpoint
        print("[Eval] backbone-source=pretrained: will load pretrained backbone checkpoint.")
    elif args.backbone_source == "det_checkpoint":
        backbone_ckpt = None
        print("[Eval] backbone-source=det_checkpoint: will NOT load pretrained backbone; use weights from det checkpoint.")
    else:  # auto
        if has_backbone:
            backbone_ckpt = None
            print("[Eval] backbone-source=auto: detected backbone weights in det checkpoint; will use finetuned backbone.")
        else:
            backbone_ckpt = args.backbone_checkpoint
            print("[Eval] backbone-source=auto: no backbone weights detected; will load pretrained backbone checkpoint.")

    # Resolve num_classes (foreground). For normal eval, infer from dataset if not provided.
    num_classes = args.num_classes
    if num_classes is None:
        data_root = Path(args.data_root)
        # Allow passing DIOR root or coco root
        if not (data_root / "annotations").exists() and (data_root / "coco" / "annotations").exists():
            data_root = data_root / "coco"
        ann_file = Path(args.ann_file) if args.ann_file else data_root / "annotations" / "instances_val.json"
        img_dir = Path(args.img_dir) if args.img_dir else data_root / "images" / "val"
        ds = CocoDetectionDataset(str(ann_file), str(img_dir), transform=build_transform())
        num_classes = ds.num_classes

    model = DinoV3FasterRCNN(
        num_classes=int(num_classes),
        model_name=args.model_name,
        image_size=args.image_size,
        checkpoint_path=backbone_ckpt,
        out_channels=args.out_channels,
        freeze_backbone=True,
    ).to(device)

    if is_multitask:
        lora_injected = True
        if args.backbone_source in {"auto", "det_checkpoint"} and isinstance(checkpoint.get("backbone"), dict):
            # IMPORTANT: LoRA multitask checkpoints store LoRA params inside checkpoint["backbone"].
            # Inject LoRA modules before loading backbone state_dict, otherwise almost all backbone weights will be skipped.
            lora_injected = _maybe_inject_lora(model.detector.backbone.backbone, checkpoint)

        if args.backbone_source in {"auto", "det_checkpoint"} and isinstance(checkpoint.get("backbone"), dict):
            print("[Eval] Loading multitask backbone weights from --checkpoint...")
            target_sd = model.detector.backbone.backbone.state_dict()
            backbone_state_f, rep = _prepare_multitask_backbone_state(checkpoint["backbone"], target_sd)
            missing_b, unexpected_b = model.detector.backbone.backbone.load_state_dict(backbone_state_f, strict=False)
            total_keys = len(target_sd)
            provided_keys = int(rep["candidate"])
            matched = max(total_keys - len(missing_b), 0)
            matched_pct = (100.0 * matched / max(total_keys, 1))
            print(
                "[Eval][LoadSummary] backbone: "
                f"provided={provided_keys}, total={total_keys}, matched≈{matched} ({matched_pct:.1f}%), "
                f"missing={len(missing_b)}, unexpected={len(unexpected_b)}"
            )
            if matched_pct < 50.0:
                print(
                    "[Eval][Warn] Backbone matched <50%; this often means checkpoint format/key prefixes do not match. "
                    "If this is a my_mod_squad checkpoint, ensure you're using my_mod_squad/eval.py (compat rewrite)."
                )

        # 如果检测到 LoRA 参数但未成功注入，直接报错，避免静默输出空框
        if not lora_injected and _state_dict_has_lora(checkpoint.get("backbone", {})):
            raise RuntimeError(
                "Detected LoRA weights in checkpoint backbone but failed to inject. "
                "Make sure PYTHONPATH includes repo root and lora_multitask is importable, "
                "and run with Python>=3.10."
            )

        detector_sd = model.detector.state_dict()
        det_state_f, rep = _prepare_det_head_state_for_detector(det_state, detector_sd)
        print(
            "[Eval][LoadSummary] det_head: "
            f"provided={rep['provided']}, head_candidates={rep['head_candidates']}, "
            f"kept={rep['kept']}, dropped={rep['dropped']}"
        )

        # Load into submodules to avoid reporting backbone keys as missing.
        proj_state = {k[len('backbone.proj.') :]: v for k, v in det_state_f.items() if k.startswith('backbone.proj.')}
        rpn_state = {k[len('rpn.') :]: v for k, v in det_state_f.items() if k.startswith('rpn.')}
        roi_state = {k[len('roi_heads.') :]: v for k, v in det_state_f.items() if k.startswith('roi_heads.')}

        missing_proj, unexpected_proj = model.detector.backbone.proj.load_state_dict(proj_state, strict=False)
        missing_rpn, unexpected_rpn = model.detector.rpn.load_state_dict(rpn_state, strict=False)
        missing_roi, unexpected_roi = model.detector.roi_heads.load_state_dict(roi_state, strict=False)

        print(
            "[Eval][LoadSummary] det_head.backbone.proj: "
            f"provided={len(proj_state)}, missing={len(missing_proj)}, unexpected={len(unexpected_proj)}"
        )
        print(
            "[Eval][LoadSummary] det_head.rpn: "
            f"provided={len(rpn_state)}, missing={len(missing_rpn)}, unexpected={len(unexpected_rpn)}"
        )
        print(
            "[Eval][LoadSummary] det_head.roi_heads: "
            f"provided={len(roi_state)}, missing={len(missing_roi)}, unexpected={len(unexpected_roi)}"
        )

        if (
            len(missing_proj)
            + len(unexpected_proj)
            + len(missing_roi)
            + len(unexpected_roi)
            + len(missing_rpn)
            + len(unexpected_rpn)
        ) > 0:
            print(
                "[Eval][Warn] det_head did not fully match. "
                "Most commonly this is due to num-classes mismatch (cls_score/bbox_pred shapes)."
            )
    else:
        missing, unexpected = model.load_state_dict(det_state, strict=False)
        if missing or unexpected:
            print(f"Loaded checkpoint with missing keys: {len(missing)}, unexpected: {len(unexpected)}")
            # 如果用户期望用 det_checkpoint 的 backbone，但缺失 backbone 相关 key，给明显提示
            if args.backbone_source in {"auto", "det_checkpoint"}:
                miss_lower = [k.lower() for k in missing]
                if any(("backbone" in k) for k in miss_lower):
                    print(
                        "[Eval][Warn] Missing backbone keys while expecting finetuned backbone from det checkpoint. "
                        "This checkpoint may not contain backbone weights or key names do not match."
                    )

    if args.check_load_only:
        print("[LoadCheck] done (no dataset/eval)")
        return

    model.eval()

    data_root = Path(args.data_root)
    # Allow passing DIOR root or coco root
    if not (data_root / "annotations").exists() and (data_root / "coco" / "annotations").exists():
        data_root = data_root / "coco"
    ann_file = Path(args.ann_file) if args.ann_file else data_root / "annotations" / "instances_val.json"
    img_dir = Path(args.img_dir) if args.img_dir else data_root / "images" / "val"

    ds = CocoDetectionDataset(str(ann_file), str(img_dir), transform=build_transform())
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    use_coco = args.use_coco_eval and COCO is not None and COCOeval is not None
    coco_results: List[Dict] = []

    total_tp = total_fp = total_fn = 0

    with torch.no_grad():
        for idx, (images, targets) in enumerate(loader):
            images = [img.to(device, non_blocking=True) for img in images]
            outputs = model(images)

            if use_coco:
                for out, tgt in zip(outputs, targets):
                    image_id = int(tgt["image_id"])
                    boxes = out["boxes"].cpu()
                    labels = out["labels"].cpu()
                    scores = out["scores"].cpu()

                    keep = scores >= args.score_thr
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
                tp, fp, fn = compute_precision_recall(outputs, targets_dev, score_threshold=args.score_thr)
                total_tp += tp
                total_fp += fp
                total_fn += fn

            if (idx + 1) % 50 == 0:
                print(f"Processed {idx+1}/{len(loader)}")

    split_name = ann_file.stem  # e.g. instances_test / instances_val

    if use_coco:
        coco_gt = COCO(str(ann_file))

        # DIOR 的 COCO 标注可能缺少这些顶层字段，pycocotools.loadRes 会直接 KeyError
        coco_gt.dataset.setdefault("info", {})
        coco_gt.dataset.setdefault("licenses", [])

        coco_eval = None
        if coco_results:
            coco_dt = coco_gt.loadRes(coco_results)
            coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        else:
            print(
                "[Eval][Warn] No predictions were produced (coco_results is empty). "
                "Skipping COCOeval to avoid pycocotools.loadRes([]) crash."
            )

        # 1) 保存 predictions（默认保存到 stats-dir）
        if args.save_json:
            save_path = Path(args.save_json)
        else:
            save_path = stats_dir / f"predictions_{split_name}.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(coco_results, f)
        print(f"Saved predictions to {save_path}")

        # 2) 保存 COCO 指标到 stats-dir
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
        stats = [float(x) for x in getattr(coco_eval, "stats", [])] if coco_eval is not None else []  # len==12
        metrics = {name: stats[i] if i < len(stats) else None for i, name in enumerate(metric_names)}
        metrics.update(
            {
                "ann_file": str(ann_file),
                "img_dir": str(img_dir),
                "checkpoint": str(ckpt_path),
                "score_thr": float(args.score_thr),
                "num_predictions": int(len(coco_results)),
            }
        )

        metrics_path = stats_dir / f"metrics_{split_name}.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics to {metrics_path}")

    else:
        precision = total_tp / max(total_tp + total_fp, 1)
        recall = total_tp / max(total_tp + total_fn, 1)
        print(f"Precision@0.5: {precision:.4f}, Recall@0.5: {recall:.4f}")

        metrics = {
            "precision@0.5": float(precision),
            "recall@0.5": float(recall),
            "tp": int(total_tp),
            "fp": int(total_fp),
            "fn": int(total_fn),
            "ann_file": str(ann_file),
            "img_dir": str(img_dir),
            "checkpoint": str(ckpt_path),
            "score_thr": float(args.score_thr),
        }
        metrics_path = stats_dir / f"metrics_{split_name}.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()

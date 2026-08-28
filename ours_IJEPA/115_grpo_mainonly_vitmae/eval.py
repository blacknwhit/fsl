from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


def _torch_load_cpu(path: str):
    # Keep compatibility across torch versions (weights_only introduced later).
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


def _maybe_write_compat_checkpoint(checkpoint_path: str, *, stats_dir: Path) -> str:
    """
    Single-task eval scripts expect checkpoint['backbone'] to be the raw backbone state_dict
    (keys like 'blocks.0...', without a leading 'backbone.' prefix).

    Newer my-mod-squad training saves checkpoint['backbone'] = model.shared.state_dict(),
    which prefixes the frozen backbone weights with 'backbone.' and also includes LoRA-MoE keys.

    This function rewrites a minimal compat checkpoint (in stats_dir) if needed.
    """
    ckpt_obj = _torch_load_cpu(checkpoint_path)
    if not _is_multitask_checkpoint(ckpt_obj):
        return checkpoint_path

    backbone_state = ckpt_obj.get("backbone")
    assert isinstance(backbone_state, dict)

    # If the checkpoint stores the shared IJEPA backbone state_dict(), encoder weights are under 'backbone.'
    has_prefixed = any(k.startswith("backbone.") for k in backbone_state.keys())
    if not has_prefixed:
        return checkpoint_path

    stripped_backbone = {k[len("backbone.") :]: v for k, v in backbone_state.items() if k.startswith("backbone.")}
    if not stripped_backbone:
        # Fallback: if we can't extract anything meaningful, don't rewrite.
        return checkpoint_path

    compat = dict(ckpt_obj)
    compat["backbone"] = stripped_backbone
    compat["compat_note"] = "auto-generated for single-task eval: stripped 'backbone.' prefix from shared state_dict"
    compat_path = stats_dir / "compat_multitask_ckpt.pt"

    import torch

    torch.save(compat, compat_path)
    print(f"[multitask/eval] wrote compat checkpoint: {compat_path}")
    return str(compat_path)


def _parse_tasks(text: str) -> List[str]:
    items = [s.strip().lower() for s in (text or "").split(",") if s.strip()]
    valid = {"det", "seg", "cnt"}
    bad = [t for t in items if t not in valid]
    if bad:
        raise ValueError(f"--tasks contains invalid task(s): {bad}, valid: {sorted(valid)}")
    # preserve order while de-duping
    out: List[str] = []
    for t in items:
        if t not in out:
            out.append(t)
    return out


def _default_stats_dir(checkpoint: str, stats_dir: Optional[str]) -> Path:
    if stats_dir:
        p = Path(stats_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
    ckpt = Path(checkpoint)
    p = ckpt.parent / "stats"
    p.mkdir(parents=True, exist_ok=True)
    return p


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


def _extract_artifacts(output: str) -> Dict[str, str]:
    artifacts: Dict[str, str] = {}
    for line in output.splitlines():
        s = line.strip()
        if s.startswith("Saved metrics to "):
            artifacts["metrics"] = s[len("Saved metrics to ") :].strip()
        elif s.startswith("Saved predictions to "):
            artifacts["predictions"] = s[len("Saved predictions to ") :].strip()
    return artifacts


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate multitask checkpoint via single-task eval scripts (det/seg/cnt)")
    p.add_argument("--checkpoint", type=str, required=True, help="multitask checkpoint from multitask/train.py")
    p.add_argument("--tasks", type=str, default="det,seg,cnt", help="comma-separated: det,seg,cnt")
    p.add_argument("--stats-dir", type=str, default=None, help="shared stats dir (default: <checkpoint_dir>/stats)")
    p.add_argument("--device", type=str, default=None, help="override device for all tasks (e.g. cuda:0)")
    p.add_argument("--image-size", type=int, default=None, help="override image-size for all tasks")
    p.add_argument("--model-name", type=str, default=None, help="override model-name for all tasks")
    p.add_argument("--dry-run", action="store_true", help="print commands and exit")
    p.add_argument(
        "--check-load-only",
        action="store_true",
        help="only construct model(s) and load checkpoint(s), print load summary, then exit (no dataset/eval)",
    )
    p.add_argument(
        "--check-full-load-only",
        action="store_true",
        help=(
            "ONLY verify loading the full multitask model produced by my_mod_squad/train.py, "
            "including LoRA-MoE when present. Does NOT run dataset eval. "
            "This bypasses compat checkpoint rewriting and does not call single-task eval scripts."
        ),
    )

    p.add_argument(
        "--eval-full-model",
        action="store_true",
        help=(
            "Evaluate the training-time isomorphic multitask model (MultiTaskModel), including LoRA-MoE/task routing. "
            "Runs eval loops in-process (no compat rewrite; no subprocess single-task eval scripts)."
        ),
    )

    # Detection args (match object_detection/eval.py)
    p.add_argument("--det-data-root", type=str, default=None)
    p.add_argument("--det-ann-file", type=str, default=None)
    p.add_argument("--det-img-dir", type=str, default=None)
    p.add_argument("--det-num-classes", type=int, default=None)
    p.add_argument("--det-out-channels", type=int, default=None)
    p.add_argument("--det-score-thr", type=float, default=None)
    p.add_argument("--det-use-coco-eval", action="store_true")
    p.add_argument("--det-backbone-checkpoint", type=str, default=None)
    p.add_argument("--det-backbone-source", type=str, default=None, choices=["auto", "pretrained", "det_checkpoint"])
    p.add_argument("--det-batch-size", type=int, default=32)
    p.add_argument("--det-num-workers", type=int, default=None)

    # Seg args (match segmentation/eval.py)
    p.add_argument("--seg-data-dir", type=str, default=None)
    p.add_argument("--seg-num-classes", type=int, default=None)
    p.add_argument("--seg-backbone-checkpoint", type=str, default=None)
    p.add_argument("--seg-save-preds", type=str, default=None)
    p.add_argument("--seg-vis-dir", type=str, default=None)

    # Count args (match counting/eval.py)
    p.add_argument("--cnt-data-root", type=str, default=None)
    p.add_argument("--cnt-test-dir", type=str, default=None)
    p.add_argument("--cnt-num-classes", type=int, default=None)
    aspect = p.add_mutually_exclusive_group()
    aspect.add_argument("--cnt-keep-aspect", dest="cnt_keep_aspect", action="store_true")
    aspect.add_argument("--cnt-no-keep-aspect", dest="cnt_keep_aspect", action="store_false")
    p.set_defaults(cnt_keep_aspect=None)
    p.add_argument("--cnt-backbone-checkpoint", type=str, default=None)
    p.add_argument("--cnt-backbone-source", type=str, default=None, choices=["auto", "pretrained", "ckpt"])
    p.add_argument("--cnt-batch-size", type=int, default=None)
    p.add_argument("--cnt-num-workers", type=int, default=None)

    return p.parse_args()


def _infer_fg_num_classes_from_det_state(det_state: Dict) -> int | None:
    w = det_state.get("roi_heads.box_predictor.cls_score.weight")
    if hasattr(w, "shape") and len(getattr(w, "shape", [])) >= 1:
        total = int(w.shape[0])
        if total >= 2:
            return total - 1
    return None


def _infer_num_classes_from_conv1x1_weight(state: Dict, weight_key: str) -> int | None:
    w = state.get(weight_key)
    if hasattr(w, "shape") and len(getattr(w, "shape", [])) >= 1:
        return int(w.shape[0])
    return None


def _infer_lora_moe_config_from_shared_state(
    shared_state: Dict,
    ckpt_config: Dict[str, object] | None = None,
) -> Dict[str, object]:
    # Heuristic: LoRA-MoE checkpoints have lora_moes.* params and wrapped_blocks.* wrappers.
    def _config_int(key: str, default: int) -> int:
        if not isinstance(ckpt_config, dict):
            return int(default)
        value = ckpt_config.get(key)
        try:
            value = int(value)
        except (TypeError, ValueError):
            return int(default)
        return int(value) if int(value) >= 1 else int(default)

    default_moe_k_private = _config_int("moe_k_private", 2)
    default_moe_k_shared = _config_int("moe_k_shared", 2)

    has_lora_moes = any(isinstance(k, str) and k.startswith("lora_moes.") for k in shared_state.keys())
    has_wrapped = any(isinstance(k, str) and k.startswith("wrapped_blocks.") for k in shared_state.keys())
    use_lora_moe = bool(has_lora_moes or has_wrapped)
    if not use_lora_moe:
        return {
            "use_lora_moe": False,
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

    A_private = shared_state.get("lora_moes.0.lora_A_private")
    if hasattr(A_private, "shape") and len(getattr(A_private, "shape", [])) == 4:
        task_num = int(A_private.shape[0])
        num_experts_private = int(A_private.shape[1])
        lora_rank = int(A_private.shape[3])

    A_shared = shared_state.get("lora_moes.0.lora_A_shared")
    if hasattr(A_shared, "shape") and len(getattr(A_shared, "shape", [])) == 3:
        num_experts_shared = int(A_shared.shape[0])
        lora_rank = int(A_shared.shape[2])

    # Infer task_num from f_gate_private indices if needed
    gate_indices: List[int] = []
    for k in shared_state.keys():
        if not isinstance(k, str):
            continue
        if k.startswith("lora_moes.0.f_gate_private.") and k.endswith(".weight"):
            parts = k.split(".")
            if len(parts) >= 4:
                try:
                    gate_indices.append(int(parts[3]))
                except Exception:
                    pass
    if gate_indices:
        task_num = max(gate_indices) + 1

    # moe_k is not recoverable from tensor shapes alone, so prefer the saved checkpoint config.
    return {
        "use_lora_moe": True,
        "task_num": int(task_num),
        "lora_rank": int(lora_rank),
        "num_experts_private": int(num_experts_private),
        "num_experts_shared": int(num_experts_shared),
        "moe_k_private": default_moe_k_private,
        "moe_k_shared": default_moe_k_shared,
    }


def _import_multitask_models():
    module_dir = Path(__file__).resolve().parent
    workspace_root = Path(__file__).resolve().parents[2]
    for root in (module_dir, workspace_root):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    module = importlib.import_module("ours_IJEPA.115_grpo_mainonly_vitmae.models")
    return module.MultiTaskModel, module.SharedViTMAEBackbone


def _check_full_multitask_load_only(checkpoint_path: str, *, device: str | None, model_name: str | None, image_size: int | None) -> None:
    import torch

    ckpt = _torch_load_cpu(checkpoint_path)
    if not _is_multitask_checkpoint(ckpt):
        raise SystemExit(f"--check-full-load-only requires a multitask checkpoint (with backbone/det_head/seg_head/cnt_head): {checkpoint_path}")

    shared_state = ckpt["backbone"]
    det_state = ckpt["det_head"]
    seg_state = ckpt["seg_head"]
    cnt_state = ckpt["cnt_head"]
    ckpt_config = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else None

    assert isinstance(shared_state, dict) and isinstance(det_state, dict) and isinstance(seg_state, dict) and isinstance(cnt_state, dict)

    # Infer model hyperparams from checkpoint heads
    det_fg = _infer_fg_num_classes_from_det_state(det_state)
    seg_nc = _infer_num_classes_from_conv1x1_weight(seg_state, "decode.3.weight")
    cnt_nc = _infer_num_classes_from_conv1x1_weight(cnt_state, "decode.3.weight")
    if det_fg is None:
        raise SystemExit("Could not infer det num-classes from checkpoint det_head (missing roi_heads.box_predictor.cls_score.weight)")
    if seg_nc is None:
        raise SystemExit("Could not infer seg num-classes from checkpoint seg_head (missing decode.3.weight)")
    if cnt_nc is None:
        raise SystemExit("Could not infer cnt num-classes from checkpoint cnt_head (missing decode.3.weight)")

    cfg = _infer_lora_moe_config_from_shared_state(shared_state, ckpt_config)
    use_lora_moe = bool(cfg["use_lora_moe"])

    # Defaults consistent with train.py
    name = model_name or "ijepa_vit_huge_patch16"
    img_size = int(image_size) if image_size is not None else 448
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MultiTaskModel, SharedViTMAEBackbone = _import_multitask_models()

    shared = SharedViTMAEBackbone(
        model_name=name,
        image_size=img_size,
        checkpoint_path=None,
        use_lora_moe=use_lora_moe,
        task_num=int(cfg["task_num"]),
        lora_rank=int(cfg["lora_rank"]),
        num_experts_private=int(cfg["num_experts_private"]),
        num_experts_shared=int(cfg["num_experts_shared"]),
        moe_k_private=int(cfg["moe_k_private"]),
        moe_k_shared=int(cfg["moe_k_shared"]),
        grad_checkpointing=False,
    )

    model = MultiTaskModel(
        shared=shared,
        det_num_classes=int(det_fg),
        seg_num_classes=int(seg_nc),
        cnt_num_classes=int(cnt_nc),
        image_size=img_size,
        det_train_backbone=True,
        seg_train_backbone=True,
        cnt_train_backbone=True,
    ).to(dev)
    model.eval()

    # 1) Load shared (this includes backbone weights + LoRA-MoE when present)
    missing_s, unexpected_s = model.shared.load_state_dict(shared_state, strict=False)
    total_s = len(model.shared.state_dict())
    matched_s = max(total_s - len(missing_s), 0)
    print(
        "[FullLoadCheck] shared: "
        f"use_lora_moe={use_lora_moe}, total={total_s}, matched≈{matched_s} ({(100.0*matched_s/max(total_s,1)):.1f}%), "
        f"missing={len(missing_s)}, unexpected={len(unexpected_s)}"
    )

    # 2) Load detector head/state (rpn/roi_heads/proj; shared weights are already loaded above)
    missing_d, unexpected_d = model.detector.load_state_dict(det_state, strict=False)
    expected_prefix = "backbone.shared."
    expected_missing = [k for k in missing_d if isinstance(k, str) and k.startswith(expected_prefix)]
    real_missing = [k for k in missing_d if not (isinstance(k, str) and k.startswith(expected_prefix))]
    print(
        "[FullLoadCheck] detector: "
        f"provided={len(det_state)}, missing={len(missing_d)} (expected_backbone_missing={len(expected_missing)}, real_missing={len(real_missing)}), "
        f"unexpected={len(unexpected_d)}"
    )
    if real_missing:
        print(
            "[FullLoadCheck][Warn] detector has real missing keys (not from backbone.shared.*). "
            "This likely indicates an arch mismatch (e.g. num-classes or out_channels)."
        )

    # 3) Load seg/cnt heads
    missing_seg, unexpected_seg = model.seg_head.load_state_dict(seg_state, strict=False)
    missing_cnt, unexpected_cnt = model.cnt_head.load_state_dict(cnt_state, strict=False)
    print(f"[FullLoadCheck] seg_head: missing={len(missing_seg)}, unexpected={len(unexpected_seg)}")
    print(f"[FullLoadCheck] cnt_head: missing={len(missing_cnt)}, unexpected={len(unexpected_cnt)}")

    # 4) Quick sanity: ensure key training-time LoRA-MoE tensors are present when enabled
    if use_lora_moe:
        has_priv = any(k.startswith("lora_moes.0.lora_A_private") for k in model.shared.state_dict().keys())
        has_shared = any(k.startswith("lora_moes.0.lora_A_shared") for k in model.shared.state_dict().keys())
        print(f"[FullLoadCheck] lora_moe_keys_present: private={has_priv}, shared={has_shared}")

    print("[FullLoadCheck] done (no dataset/eval)")


def _build_full_multitask_model_from_ckpt(
    checkpoint_path: str,
    *,
    device: str | None,
    model_name: str | None,
    image_size: int | None,
) -> tuple[object, dict, dict, dict, dict, torch.device]:
    ckpt = _torch_load_cpu(checkpoint_path)
    if not _is_multitask_checkpoint(ckpt):
        raise SystemExit(
            f"--eval-full-model requires a multitask checkpoint (with backbone/det_head/seg_head/cnt_head): {checkpoint_path}"
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
    if det_fg is None:
        raise SystemExit("Could not infer det num-classes from checkpoint det_head (missing roi_heads.box_predictor.cls_score.weight)")
    if seg_nc is None:
        raise SystemExit("Could not infer seg num-classes from checkpoint seg_head (missing decode.3.weight)")
    if cnt_nc is None:
        raise SystemExit("Could not infer cnt num-classes from checkpoint cnt_head (missing decode.3.weight)")

    cfg = _infer_lora_moe_config_from_shared_state(shared_state, ckpt_config)
    use_lora_moe = bool(cfg["use_lora_moe"])

    name = model_name or "ijepa_vit_huge_patch16"
    img_size = int(image_size) if image_size is not None else 448
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        dev_index = dev.index if dev.index is not None else 0
        torch.cuda.set_device(dev_index)
        dev = torch.device(f"cuda:{dev_index}")

    MultiTaskModel, SharedViTMAEBackbone = _import_multitask_models()

    shared = SharedViTMAEBackbone(
        model_name=name,
        image_size=img_size,
        checkpoint_path=None,
        use_lora_moe=use_lora_moe,
        task_num=int(cfg["task_num"]),
        lora_rank=int(cfg["lora_rank"]),
        num_experts_private=int(cfg["num_experts_private"]),
        num_experts_shared=int(cfg["num_experts_shared"]),
        moe_k_private=int(cfg["moe_k_private"]),
        moe_k_shared=int(cfg["moe_k_shared"]),
        grad_checkpointing=False,
    )

    model = MultiTaskModel(
        shared=shared,
        det_num_classes=int(det_fg),
        seg_num_classes=int(seg_nc),
        cnt_num_classes=int(cnt_nc),
        image_size=img_size,
        det_train_backbone=True,
        seg_train_backbone=True,
        cnt_train_backbone=True,
    ).to(dev)
    model.eval()

    missing_s, unexpected_s = model.shared.load_state_dict(shared_state, strict=False)
    if missing_s or unexpected_s:
        print(f"[FullEval][Warn] shared load: missing={len(missing_s)}, unexpected={len(unexpected_s)}")

    missing_d, unexpected_d = model.detector.load_state_dict(det_state, strict=False)
    # Expected: det_state does not contain backbone.shared.* (filtered at save time)
    expected_prefix = "backbone.shared."
    expected_missing = [k for k in missing_d if isinstance(k, str) and k.startswith(expected_prefix)]
    real_missing = [k for k in missing_d if not (isinstance(k, str) and k.startswith(expected_prefix))]
    if real_missing or unexpected_d:
        print(
            "[FullEval][Warn] detector load mismatch: "
            f"real_missing={len(real_missing)}, unexpected={len(unexpected_d)}, expected_backbone_missing={len(expected_missing)}"
        )

    missing_seg, unexpected_seg = model.seg_head.load_state_dict(seg_state, strict=False)
    if missing_seg or unexpected_seg:
        print(f"[FullEval][Warn] seg_head load: missing={len(missing_seg)}, unexpected={len(unexpected_seg)}")

    missing_cnt, unexpected_cnt = model.cnt_head.load_state_dict(cnt_state, strict=False)
    if missing_cnt or unexpected_cnt:
        print(f"[FullEval][Warn] cnt_head load: missing={len(missing_cnt)}, unexpected={len(unexpected_cnt)}")

    meta = {
        "det_fg_classes": int(det_fg),
        "seg_num_classes": int(seg_nc),
        "cnt_num_classes": int(cnt_nc),
        "use_lora_moe": bool(use_lora_moe),
        "model_name": str(name),
        "image_size": int(img_size),
        "moe_cfg": dict(cfg),
    }

    return model, meta, shared_state, det_state, seg_state, cnt_state, dev


def _eval_full_model_det(
    model,
    *,
    stats_dir: Path,
    data_root: str,
    ann_file: str,
    img_dir: str,
    device: torch.device,
    score_thr: float,
    use_coco_eval: bool,
    batch_size: int,
    num_workers: int,
) -> dict:
    # Imports from existing single-task det utilities
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

    ds = CocoDetectionDataset(str(ann_file), str(img_dir), transform=lambda img, tgt: (TF.to_tensor(img), tgt))
    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    use_coco = bool(use_coco_eval and COCO is not None and COCOeval is not None)
    coco_results: List[Dict] = []
    total_tp = total_fp = total_fn = 0

    # FasterRCNN returns losses only in train mode; for inference use eval.
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
                tp, fp, fn = compute_precision_recall(outputs, targets_dev, score_threshold=float(score_thr))
                total_tp += int(tp)
                total_fp += int(fp)
                total_fn += int(fn)

            if (idx + 1) % 50 == 0:
                print(f"[FullEval][det] processed {idx+1}/{len(loader)}")

    split_name = Path(ann_file).stem
    metrics: dict

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
            print("[FullEval][det][Warn] no predictions produced; skipping COCOeval")

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
        with pred_path.open("w", encoding="utf-8") as f:
            json.dump(coco_results, f)
        print(f"[FullEval][det] saved predictions: {pred_path}")
    else:
        precision = float(total_tp / max(total_tp + total_fp, 1))
        recall = float(total_tp / max(total_tp + total_fn, 1))
        print(f"[FullEval][det] Precision@0.5: {precision:.4f}, Recall@0.5: {recall:.4f}")
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
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[FullEval][det] saved metrics: {metrics_path}")
    return metrics


def _eval_full_model_seg(model, *, stats_dir: Path, data_dir: str, device: torch.device, num_classes: int, image_size: int) -> dict:
    from segmentation.dataset import SegmentationDataset
    from segmentation.utils import per_class_iou_from_confusion, update_confusion_matrix
    import torchvision.transforms.functional as TF

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def _transform(img, mask):
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=mean, std=std)
        mask_tensor = torch.as_tensor(TF.pil_to_tensor(mask), dtype=torch.long).squeeze(0)
        return img, mask_tensor

    ds = SegmentationDataset(data_dir, transform=_transform, image_size=int(image_size))
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    model.eval()
    conf = torch.zeros((int(num_classes), int(num_classes)), dtype=torch.int64)

    with torch.no_grad():
        for idx, (imgs, masks) in enumerate(loader):
            if idx % 10 == 0:
                print(f"[FullEval][seg] processed {idx}/{len(loader)}")
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
    split_name = Path(data_dir).name or "seg"
    metrics = {
        "task": "segmentation",
        "data_dir": str(Path(data_dir)),
        "num_classes": int(num_classes),
        "image_size": int(image_size),
        "ignore_indices": [255, 11],
        "per_class_iou": [(None if (v != v) else float(v.item())) for v in per_class_iou],
        "miou": (None if (float(miou.item()) != float(miou.item())) else float(miou.item())),
    }
    metrics_path = stats_dir / f"metrics_seg_{split_name}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[FullEval][seg] saved metrics: {metrics_path}")
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
        def __init__(self, mean_, std_):
            self.mean = mean_
            self.std = std_

        def __call__(self, img, density):
            if not torch.is_tensor(img):
                img = TF.to_tensor(img)
            img = TF.normalize(img, mean=self.mean, std=self.std)
            return img, density

    root = Path(data_root)
    tdir = Path(test_dir) if test_dir else (root / "test_data_class8")
    ds = DSACADensityH5Dataset(
        str(tdir),
        num_classes=int(num_classes),
        transform=_NormalizeTransform(mean, std),
        image_size=int(image_size),
        keep_aspect=bool(keep_aspect),
    )

    loader_kwargs = dict(
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=True,
    )
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
    loader = torch.utils.data.DataLoader(ds, **loader_kwargs)

    sum_abs = torch.zeros(int(num_classes), dtype=torch.float64)
    sum_sq = torch.zeros(int(num_classes), dtype=torch.float64)
    n = 0

    model.eval()
    with torch.no_grad():
        for idx, (imgs, dens) in enumerate(loader):
            if (idx + 1) % 50 == 0:
                print(f"[FullEval][cnt] processed {idx+1}/{len(loader)}")
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
    split_name = tdir.name or "count"
    metrics = {
        "task": "counting",
        "data_root": str(root),
        "test_dir": str(tdir),
        "num_classes": int(num_classes),
        "image_size": int(image_size),
        "keep_aspect": bool(keep_aspect),
        "per_class_mae": [float(x.item()) for x in mae],
        "per_class_rmse": [float(x.item()) for x in rmse],
        "mae": float(mae.mean().item()),
        "rmse": float(rmse.mean().item()),
        "n_images": int(n),
    }
    metrics_path = stats_dir / f"metrics_cnt_{split_name}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[FullEval][cnt] saved metrics: {metrics_path}")
    return metrics


def _run_eval_full_model(args) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # Build/load the isomorphic multitask model.
    model, meta, _shared_state, _det_state, _seg_state, _cnt_state, dev = _build_full_multitask_model_from_ckpt(
        args.checkpoint,
        device=args.device,
        model_name=args.model_name,
        image_size=args.image_size,
    )

    stats_dir = _default_stats_dir(args.checkpoint, args.stats_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    tasks = _parse_tasks(args.tasks)
    results: Dict[str, object] = {"meta": meta, "tasks": tasks, "results": {}}

    # Det
    if "det" in tasks:
        ann_file = args.det_ann_file
        img_dir = args.det_img_dir
        data_root = args.det_data_root
        if not (data_root and ann_file and img_dir):
            raise SystemExit("det task selected but --det-data-root/--det-ann-file/--det-img-dir is missing")
        score_thr = float(args.det_score_thr) if args.det_score_thr is not None else 0.0
        bs = int(args.det_batch_size) if args.det_batch_size is not None else 1
        nw = int(args.det_num_workers) if args.det_num_workers is not None else 2
        results["results"]["det"] = _eval_full_model_det(
            model,
            stats_dir=stats_dir,
            data_root=data_root,
            ann_file=ann_file,
            img_dir=img_dir,
            device=dev,
            score_thr=score_thr,
            use_coco_eval=bool(args.det_use_coco_eval),
            batch_size=bs,
            num_workers=nw,
        )

    # Seg
    if "seg" in tasks:
        data_dir = args.seg_data_dir
        if not data_dir:
            raise SystemExit("seg task selected but --seg-data-dir is missing")
        # Prefer explicit num-classes override if provided, otherwise use checkpoint-inferred meta.
        seg_nc = int(args.seg_num_classes) if args.seg_num_classes is not None else int(meta["seg_num_classes"])
        img_size = int(args.image_size) if args.image_size is not None else int(meta["image_size"])
        results["results"]["seg"] = _eval_full_model_seg(
            model,
            stats_dir=stats_dir,
            data_dir=data_dir,
            device=dev,
            num_classes=seg_nc,
            image_size=img_size,
        )

    # Cnt
    if "cnt" in tasks:
        data_root = args.cnt_data_root
        if not data_root:
            raise SystemExit("cnt task selected but --cnt-data-root is missing")
        cnt_nc = int(args.cnt_num_classes) if args.cnt_num_classes is not None else int(meta["cnt_num_classes"])
        img_size = int(args.image_size) if args.image_size is not None else int(meta["image_size"])
        keep_aspect = True if args.cnt_keep_aspect is None else bool(args.cnt_keep_aspect)
        bs = int(args.cnt_batch_size) if args.cnt_batch_size is not None else 4
        nw = int(args.cnt_num_workers) if args.cnt_num_workers is not None else 2
        results["results"]["cnt"] = _eval_full_model_cnt(
            model,
            stats_dir=stats_dir,
            data_root=data_root,
            test_dir=args.cnt_test_dir,
            device=dev,
            num_classes=cnt_nc,
            image_size=img_size,
            keep_aspect=keep_aspect,
            batch_size=bs,
            num_workers=nw,
        )

    out_path = stats_dir / "multitask_full_model_eval_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[FullEval] wrote summary: {out_path}")
    return 0


def _build_det_cmd(args, *, python: str, stats_dir: Path) -> List[str]:
    cmd = [python, "object_detection/eval.py", "--checkpoint", args.checkpoint, "--stats-dir", str(stats_dir)]
    if args.check_load_only:
        cmd += ["--check-load-only"]
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
    if args.check_load_only:
        cmd += ["--check-load-only"]
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
    if args.check_load_only:
        cmd += ["--check-load-only"]
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


def main() -> int:
    args = parse_args()
    # Full training-weights load check: bypass compat rewriting and subprocess eval.
    if bool(getattr(args, "check_full_load_only", False)):
        _check_full_multitask_load_only(
            args.checkpoint,
            device=args.device,
            model_name=args.model_name,
            image_size=args.image_size,
        )
        return 0

    if bool(getattr(args, "eval_full_model", False)):
        return _run_eval_full_model(args)

    tasks = _parse_tasks(args.tasks)

    repo_root = Path(__file__).resolve().parents[2]
    python = sys.executable
    stats_dir = _default_stats_dir(args.checkpoint, args.stats_dir)

    # Ensure child eval scripts can import repo packages (e.g. lora_multitask)
    base_env = os.environ.copy()
    existing_pp = base_env.get("PYTHONPATH", "")
    base_env["PYTHONPATH"] = f"{repo_root}:{existing_pp}" if existing_pp else str(repo_root)

    # Adapt new my-mod-squad checkpoint format to the layout expected by single-task eval scripts.
    args.checkpoint = _maybe_write_compat_checkpoint(args.checkpoint, stats_dir=stats_dir)

    summary: Dict[str, object] = {
        "checkpoint": str(Path(args.checkpoint)),
        "stats_dir": str(stats_dir),
        "tasks": tasks,
        "runs": {},
    }

    runners = {
        "det": _build_det_cmd,
        "seg": _build_seg_cmd,
        "cnt": _build_cnt_cmd,
    }

    # Validate required dataset args per selected task
    if "det" in tasks and not args.det_data_root:
        raise SystemExit("det task selected but --det-data-root is missing")
    if "seg" in tasks and not args.seg_data_dir:
        raise SystemExit("seg task selected but --seg-data-dir is missing")
    if "cnt" in tasks and not args.cnt_data_root:
        raise SystemExit("cnt task selected but --cnt-data-root is missing")

    for t in tasks:
        build = runners[t]
        cmd = build(args, python=python, stats_dir=stats_dir)
        print(f"\n[multitask/eval] running {t}: {shlex.join(cmd)}\n")
        if args.dry_run:
            summary["runs"][t] = {"cmd": cmd, "dry_run": True}
            continue

        started = time.time()
        rc, out = _run_cmd(cmd, cwd=repo_root, env=base_env)
        elapsed = time.time() - started
        artifacts = _extract_artifacts(out)
        summary["runs"][t] = {
            "cmd": cmd,
            "returncode": int(rc),
            "elapsed_sec": float(elapsed),
            "artifacts": artifacts,
        }
        if rc != 0:
            summary_path = stats_dir / "multitask_eval_summary.json"
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"[multitask/eval] FAILED (task={t}, rc={rc}); wrote summary to {summary_path}")
            return rc

    summary_path = stats_dir / "multitask_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[multitask/eval] done; summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

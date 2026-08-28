from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn.functional as F


def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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


def _infer_det_out_channels(det_state: Dict) -> int | None:
    w = det_state.get("backbone.proj.weight")
    if hasattr(w, "shape") and len(getattr(w, "shape", [])) >= 1:
        return int(w.shape[0])
    return None


def _infer_lora_moe_config_from_shared_state(shared_state: Dict) -> Dict[str, object]:
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
            "moe_k_private": 2,
            "moe_k_shared": 2,
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

    return {
        "use_lora_moe": True,
        "task_num": int(task_num),
        "lora_rank": int(lora_rank),
        "num_experts_private": int(num_experts_private),
        "num_experts_shared": int(num_experts_shared),
        "moe_k_private": 2,
        "moe_k_shared": 2,
    }


def _iter_next(it, loader):
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it


def _grad_stats(params: Iterable[torch.Tensor]) -> Dict[str, float]:
    total_sq = 0.0
    param_tensors = 0
    param_elements = 0
    grad_tensors = 0
    for p in params:
        param_tensors += 1
        param_elements += int(p.numel())
        if p.grad is None:
            continue
        grad_tensors += 1
        total_sq += float(p.grad.detach().float().pow(2).sum().item())
    return {
        "grad_l2": math.sqrt(total_sq),
        "param_tensors": float(param_tensors),
        "param_elements": float(param_elements),
        "grad_tensors": float(grad_tensors),
    }


def _build_groups(model) -> Dict[str, List[torch.Tensor]]:
    lora_moe_params: List[torch.Tensor] = []
    f_gate_private_params: List[torch.Tensor] = []
    f_gate_shared_params: List[torch.Tensor] = []
    if getattr(model.shared, "lora_moes", None) is not None:
        for lora_moe in model.shared.lora_moes:
            lora_moe_params.extend([p for p in lora_moe.parameters()])
            for gate in lora_moe.f_gate_private:
                f_gate_private_params.extend([p for p in gate.parameters()])
            for gate in lora_moe.f_gate_shared:
                f_gate_shared_params.extend([p for p in gate.parameters()])

    shared_backbone_ids = {id(p) for p in model.shared.backbone.parameters()}
    lora_moe_ids = {id(p) for p in lora_moe_params}
    det_params = [p for p in model.detector.parameters() if id(p) not in shared_backbone_ids and id(p) not in lora_moe_ids]

    return {
        "f_gate_private": f_gate_private_params,
        "f_gate_shared": f_gate_shared_params,
        "det_head": det_params,
        "seg_head": list(model.seg_head.parameters()),
        "cnt_head": list(model.cnt_head.parameters()),
    }


def _to_device_det(batch, device: torch.device):
    images, targets = batch
    images = [img.to(device, non_blocking=True) for img in images]
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    return images, targets


def _to_device_seg(batch, device: torch.device):
    imgs, masks = batch
    return imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)


def _to_device_cnt(batch, device: torch.device):
    imgs, dens = batch
    imgs = imgs.to(device, non_blocking=True).float()
    dens = dens.to(device, non_blocking=True).float()
    return imgs, dens


def parse_args():
    p = argparse.ArgumentParser(description="Analyze grad norms for f_gate vs det/seg/cnt heads")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--loss-weights", type=str, default="1,1,1")
    p.add_argument("--cnt-count-loss-weight", type=float, default=1.0)
    p.add_argument("--cnt-backbone-grad-mult", type=float, default=1.0)
    p.add_argument("--num-steps", type=int, default=1)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--loss-mode", type=str, default="total", choices=["total", "main", "aux"])
    p.add_argument("--output", type=str, default=None)
    gc = p.add_mutually_exclusive_group()
    gc.add_argument("--grad-checkpointing", dest="grad_checkpointing", action="store_true")
    gc.add_argument("--no-grad-checkpointing", dest="grad_checkpointing", action="store_false")
    p.set_defaults(grad_checkpointing=False)

    # Detection dataset
    p.add_argument("--det-data-root", type=str, required=True)
    p.add_argument("--det-train-ann", type=str, default=None)
    p.add_argument("--det-train-img-dir", type=str, default=None)
    p.add_argument("--det-num-classes", type=int, default=None)
    p.add_argument("--det-out-channels", type=int, default=None)
    p.add_argument("--det-batch-size", type=int, default=2)

    # Seg dataset
    p.add_argument("--seg-train-dir", type=str, required=True)
    p.add_argument("--seg-num-classes", type=int, default=None)
    p.add_argument("--seg-batch-size", type=int, default=4)

    # Count dataset
    p.add_argument("--cnt-data-root", type=str, required=True)
    p.add_argument("--cnt-train-dir", type=str, default=None)
    p.add_argument("--cnt-num-classes", type=int, default=None)
    aspect = p.add_mutually_exclusive_group()
    aspect.add_argument("--cnt-keep-aspect", dest="cnt_keep_aspect", action="store_true")
    aspect.add_argument("--cnt-no-keep-aspect", dest="cnt_keep_aspect", action="store_false")
    p.set_defaults(cnt_keep_aspect=True)
    p.add_argument("--cnt-batch-size", type=int, default=4)

    p.add_argument("--num-workers", type=int, default=0)

    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        from my_mod_squad.datasets import build_cnt_loaders, build_det_loaders, build_seg_loaders
        from my_mod_squad.models import MultiTaskModel, SharedDinoV3Backbone
        from my_mod_squad.utils import parse_loss_weights
    except Exception:
        repo_root = Path(__file__).resolve().parents[1]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from my_mod_squad.datasets import build_cnt_loaders, build_det_loaders, build_seg_loaders
        from my_mod_squad.models import MultiTaskModel, SharedDinoV3Backbone
        from my_mod_squad.utils import parse_loss_weights

    ckpt = _torch_load_cpu(args.checkpoint)
    if not (isinstance(ckpt, dict) and "backbone" in ckpt and "det_head" in ckpt and "seg_head" in ckpt and "cnt_head" in ckpt):
        raise SystemExit(f"checkpoint is not a multitask ckpt: {args.checkpoint}")

    shared_state = ckpt["backbone"]
    det_state = ckpt["det_head"]
    seg_state = ckpt["seg_head"]
    cnt_state = ckpt["cnt_head"]

    det_fg = args.det_num_classes if args.det_num_classes is not None else _infer_fg_num_classes_from_det_state(det_state)
    seg_nc = args.seg_num_classes if args.seg_num_classes is not None else _infer_num_classes_from_conv1x1_weight(seg_state, "decode.3.weight")
    cnt_nc = args.cnt_num_classes if args.cnt_num_classes is not None else _infer_num_classes_from_conv1x1_weight(cnt_state, "decode.3.weight")
    det_out_channels = args.det_out_channels if args.det_out_channels is not None else _infer_det_out_channels(det_state)
    if det_fg is None or seg_nc is None or cnt_nc is None:
        raise SystemExit("Could not infer num-classes from checkpoint; please pass --det-num-classes/--seg-num-classes/--cnt-num-classes.")

    cfg = _infer_lora_moe_config_from_shared_state(shared_state)
    use_lora_moe = bool(cfg["use_lora_moe"])

    model_name = args.model_name or "dinov3_vitl16"
    image_size = int(args.image_size) if args.image_size is not None else 448
    dev = torch.device(args.device)
    if dev.type == "cuda" and dev.index is not None:
        torch.cuda.set_device(dev.index)

    shared = SharedDinoV3Backbone(
        model_name=model_name,
        image_size=image_size,
        checkpoint_path=None,
        use_lora_moe=use_lora_moe,
        task_num=int(cfg["task_num"]),
        lora_rank=int(cfg["lora_rank"]),
        num_experts_private=int(cfg["num_experts_private"]),
        num_experts_shared=int(cfg["num_experts_shared"]),
        moe_k_private=int(cfg["moe_k_private"]),
        moe_k_shared=int(cfg["moe_k_shared"]),
        grad_checkpointing=bool(args.grad_checkpointing),
    )

    model = MultiTaskModel(
        shared=shared,
        det_num_classes=int(det_fg),
        seg_num_classes=int(seg_nc),
        cnt_num_classes=int(cnt_nc),
        image_size=image_size,
        det_out_channels=int(det_out_channels) if det_out_channels is not None else 256,
        det_train_backbone=True,
        seg_train_backbone=True,
        cnt_train_backbone=True,
    ).to(dev)

    model.shared.load_state_dict(shared_state, strict=False)
    model.detector.load_state_dict(det_state, strict=False)
    model.seg_head.load_state_dict(seg_state, strict=False)
    model.cnt_head.load_state_dict(cnt_state, strict=False)

    det_ds, _det_val_ds, det_loader, _det_val_loader = build_det_loaders(
        data_root=args.det_data_root,
        image_size=image_size,
        batch_size=int(args.det_batch_size),
        num_workers=int(args.num_workers),
        train_ann=args.det_train_ann,
        train_img_dir=args.det_train_img_dir,
        val_ann=args.det_train_ann,
        val_img_dir=args.det_train_img_dir,
    )
    seg_ds, _seg_val_ds, seg_loader, _seg_val_loader = build_seg_loaders(
        train_dir=args.seg_train_dir,
        val_dir=args.seg_train_dir,
        image_size=image_size,
        batch_size=int(args.seg_batch_size),
        num_workers=int(args.num_workers),
    )
    cnt_ds, _cnt_val_ds, cnt_loader, _cnt_val_loader = build_cnt_loaders(
        data_root=args.cnt_data_root,
        train_dir=args.cnt_train_dir,
        val_dir=args.cnt_train_dir,
        image_size=image_size,
        num_classes=int(cnt_nc),
        keep_aspect=bool(args.cnt_keep_aspect),
        batch_size=int(args.cnt_batch_size),
        num_workers=int(args.num_workers),
    )

    w_det, w_seg, w_cnt = parse_loss_weights(args.loss_weights)

    model.train()
    det_it = iter(det_loader)
    seg_it = iter(seg_loader)
    cnt_it = iter(cnt_loader)

    stats_out: Dict[str, float] = {}
    group_sums: Dict[str, float] = {}
    group_counts: Dict[str, float] = {}

    for _ in range(int(args.num_steps)):
        model.zero_grad(set_to_none=True)
        det_batch, det_it = _iter_next(det_it, det_loader)
        seg_batch, seg_it = _iter_next(seg_it, seg_loader)
        cnt_batch, cnt_it = _iter_next(cnt_it, cnt_loader)

        det_images, det_targets = _to_device_det(det_batch, dev)
        seg_imgs, seg_masks = _to_device_seg(seg_batch, dev)
        cnt_imgs, cnt_dens = _to_device_cnt(cnt_batch, dev)
        cnt_gt_counts = cnt_dens.flatten(2).sum(dim=2)

        autocast_device = dev.type if dev.type in {"cuda", "cpu"} else "cuda"
        with torch.amp.autocast(autocast_device, enabled=bool(args.amp)):
            det_loss_dict = model.forward_det(det_images, det_targets)
            det_loss = sum(det_loss_dict.values())

            seg_logits = model.forward_seg(seg_imgs)
            seg_loss = F.cross_entropy(seg_logits, seg_masks)

            pred_dens, pred_counts = model.forward_cnt(cnt_imgs, cnt_backbone_grad_mult=float(args.cnt_backbone_grad_mult))
            dens_loss = F.mse_loss(pred_dens, cnt_dens, reduction="sum") / cnt_imgs.size(0)
            cnt_l1 = F.l1_loss(pred_counts, cnt_gt_counts)
            cnt_loss = dens_loss + float(args.cnt_count_loss_weight) * cnt_l1

            aux_loss = det_loss.new_tensor(0.0)

            main_loss = float(w_det) * det_loss + float(w_seg) * seg_loss + float(w_cnt) * cnt_loss
            if args.loss_mode == "aux":
                loss = aux_loss
            elif args.loss_mode == "main":
                loss = main_loss
            else:
                loss = main_loss + aux_loss

        loss.backward()

        stats_out["det_loss"] = stats_out.get("det_loss", 0.0) + float(det_loss.detach().item())
        stats_out["seg_loss"] = stats_out.get("seg_loss", 0.0) + float(seg_loss.detach().item())
        stats_out["cnt_loss"] = stats_out.get("cnt_loss", 0.0) + float(cnt_loss.detach().item())
        stats_out["aux_loss"] = stats_out.get("aux_loss", 0.0) + float(aux_loss.detach().item())
        stats_out["main_loss"] = stats_out.get("main_loss", 0.0) + float(main_loss.detach().item())
        stats_out["total_loss"] = stats_out.get("total_loss", 0.0) + float((main_loss + aux_loss).detach().item())

        groups = _build_groups(model)
        for name, params in groups.items():
            gs = _grad_stats(params)
            group_sums[name] = group_sums.get(name, 0.0) + gs["grad_l2"]
            group_counts[name] = group_counts.get(name, 0.0) + 1.0

    steps = float(max(int(args.num_steps), 1))
    print(f"[grad_norms] steps={int(args.num_steps)} loss_mode={args.loss_mode}")
    print(f"[loss] det={stats_out['det_loss']/steps:.6f} seg={stats_out['seg_loss']/steps:.6f} cnt={stats_out['cnt_loss']/steps:.6f}")
    print(f"[loss] main={stats_out['main_loss']/steps:.6f} aux={stats_out['aux_loss']/steps:.6f} total={stats_out['total_loss']/steps:.6f}")

    output: Dict[str, object] = {
        "loss_mode": args.loss_mode,
        "steps": int(args.num_steps),
        "loss": {k: v / steps for k, v in stats_out.items()},
        "grad_norms": {},
    }
    groups = _build_groups(model)
    for name, params in groups.items():
        avg_norm = group_sums.get(name, 0.0) / max(group_counts.get(name, 1.0), 1.0)
        gs = _grad_stats(params)
        print(f"[grad] {name}: l2={avg_norm:.6f} tensors={int(gs['grad_tensors'])}/{int(gs['param_tensors'])}")
        output["grad_norms"][name] = {
            "grad_l2": float(avg_norm),
            "param_tensors": int(gs["param_tensors"]),
            "param_elements": int(gs["param_elements"]),
            "grad_tensors": int(gs["grad_tensors"]),
        }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"[grad_norms] wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

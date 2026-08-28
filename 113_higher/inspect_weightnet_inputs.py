from __future__ import annotations

import argparse
import importlib
from typing import Iterable, Sequence

import torch

try:
    from .models import MultiTaskModel, SharedDinoV3Backbone
except Exception:
    _models = importlib.import_module("113_higher.models")
    MultiTaskModel = _models.MultiTaskModel
    SharedDinoV3Backbone = _models.SharedDinoV3Backbone


def _numel(params: Iterable[torch.nn.Parameter]) -> int:
    return int(sum(int(p.numel()) for p in params))


def _shape_str(t: torch.Tensor) -> str:
    return "x".join(str(int(x)) for x in t.shape)


def _linear_param_count(in_dim: int, out_dim: int) -> int:
    return int(in_dim) * int(out_dim) + int(out_dim)


def _count_grad_projector_param_count(in_channels: int, num_classes: int, hidden_dim: int, out_dim: int = 64) -> int:
    class_feat_dim = int(in_channels) + 1
    conv_params = class_feat_dim * int(hidden_dim) + int(hidden_dim)
    out_params = (int(hidden_dim) * int(num_classes)) * int(out_dim) + int(out_dim)
    return int(conv_params + out_params)


def _print_named_params(title: str, named_params: Sequence[tuple[str, torch.nn.Parameter]]) -> None:
    total = _numel(p for _, p in named_params)
    print(f"\n[{title}] tensors={len(named_params)} params={total}")
    for name, p in named_params:
        print(f"  - {name:70s} shape={_shape_str(p)} numel={int(p.numel())}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Inspect train.py's loss-weight MLP gradient inputs, task-head structure, "
            "and parameter counts."
        )
    )
    p.add_argument("--model-name", type=str, default="dinov3_vitl16")
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--backbone-checkpoint", type=str, default=None)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--unfreeze-backbone", action="store_true")

    p.add_argument("--det-num-classes", type=int, default=20, help="Foreground classes (without background).")
    p.add_argument("--seg-num-classes", type=int, default=11)
    p.add_argument("--cnt-num-classes", type=int, default=8)
    p.add_argument("--weight-net-cnt-grad-hidden-dim", type=int, default=32)

    p.add_argument("--use-lora-moe", action="store_true")
    p.add_argument("--lora-rank", type=int, default=8)
    p.add_argument("--num-experts-private", type=int, default=2)
    p.add_argument("--num-experts-shared", type=int, default=6)
    p.add_argument("--moe-k-private", type=int, default=2)
    p.add_argument("--moe-k-shared", type=int, default=2)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    shared = SharedDinoV3Backbone(
        model_name=str(args.model_name),
        image_size=int(args.image_size),
        checkpoint_path=args.backbone_checkpoint,
        use_lora_moe=bool(args.use_lora_moe),
        backbone_trainable=bool(args.unfreeze_backbone),
        task_num=3,
        lora_rank=int(args.lora_rank),
        num_experts_private=int(args.num_experts_private),
        num_experts_shared=int(args.num_experts_shared),
        moe_k_private=int(args.moe_k_private),
        moe_k_shared=int(args.moe_k_shared),
        grad_checkpointing=False,
    )
    model = MultiTaskModel(
        shared=shared,
        det_num_classes=int(args.det_num_classes),
        seg_num_classes=int(args.seg_num_classes),
        cnt_num_classes=int(args.cnt_num_classes),
        image_size=int(args.image_size),
    ).to(device)
    model.eval()

    shared_backbone_ids = {id(p) for p in model.shared.backbone.parameters()}
    backbone_params = [p for p in model.shared.backbone.parameters() if p.requires_grad]
    if args.use_lora_moe:
        lora_moe_params = []
        for lora_moe in model.shared.lora_moes:
            lora_moe_params.extend([p for p in lora_moe.parameters() if p.requires_grad])
        lora_moe_ids = {id(p) for p in lora_moe_params}
        det_head_params = [
            p
            for p in model.detector.parameters()
            if p.requires_grad and id(p) not in shared_backbone_ids and id(p) not in lora_moe_ids
        ]
    else:
        lora_moe_params = []
        det_head_params = [
            p for p in model.detector.parameters() if p.requires_grad and id(p) not in shared_backbone_ids
        ]
    seg_params = [p for p in model.seg_head.parameters() if p.requires_grad]
    cnt_head_params = [p for p in model.cnt_head.parameters() if p.requires_grad]

    det_last_named = [
        *[
            (f"detector.roi_heads.box_predictor.cls_score.{n}", p)
            for n, p in model.detector.roi_heads.box_predictor.cls_score.named_parameters(recurse=False)
        ],
        *[
            (f"detector.roi_heads.box_predictor.bbox_pred.{n}", p)
            for n, p in model.detector.roi_heads.box_predictor.bbox_pred.named_parameters(recurse=False)
        ],
    ]
    seg_last_named = [
        (f"seg_head.decode.3.{n}", p) for n, p in model.seg_head.decode[3].named_parameters(recurse=False)
    ]
    cnt_last_named = [
        (f"cnt_head.decode.3.{n}", p) for n, p in model.cnt_head.decode[3].named_parameters(recurse=False)
    ]

    det_last_dim = _numel(p for _, p in det_last_named)
    seg_last_dim = _numel(p for _, p in seg_last_named)
    cnt_last_dim = _numel(p for _, p in cnt_last_named)

    cnt_last_conv = model.cnt_head.decode[3]
    if not isinstance(cnt_last_conv, torch.nn.Conv2d):
        raise TypeError(f"Expected cnt_head.decode[3] to be Conv2d, got {type(cnt_last_conv).__name__}")
    if tuple(int(v) for v in cnt_last_conv.kernel_size) != (1, 1):
        raise ValueError(f"Expected cnt_head.decode[3] kernel_size=(1, 1), got {cnt_last_conv.kernel_size}")

    joint_in = det_last_dim + seg_last_dim + cnt_last_dim
    cnt_grad_hidden_dim = int(args.weight_net_cnt_grad_hidden_dim)
    joint_det_proj_params = _linear_param_count(det_last_dim, 64)
    joint_seg_proj_params = _linear_param_count(seg_last_dim, 64)
    joint_cnt_proj_params = _count_grad_projector_param_count(
        in_channels=int(cnt_last_conv.in_channels),
        num_classes=int(cnt_last_conv.out_channels),
        hidden_dim=cnt_grad_hidden_dim,
        out_dim=64,
    )
    joint_weight_net_params = (
        joint_det_proj_params
        + joint_seg_proj_params
        + joint_cnt_proj_params
        + _linear_param_count(64 * 3, 16)
        + _linear_param_count(16, 3)
    )
    per_task_shared_weight_net_params = (
        (det_last_dim * 64 + 64)
        + (seg_last_dim * 64 + 64)
        + (cnt_last_dim * 64 + 64)
        + (64 * 16 + 16)
        + (16 * 1 + 1)
    )

    print("=== Task-head structure (current code) ===")
    print("[det] detector.roi_heads.box_predictor:")
    print(model.detector.roi_heads.box_predictor)
    print("[seg] seg_head:")
    print(model.seg_head)
    print("[cnt] cnt_head:")
    print(model.cnt_head)

    print("\n=== Parameter counts (train.py selection logic) ===")
    print(f"Backbone params: {_numel(backbone_params)}")
    print(f"LoRA-MoE params: {_numel(lora_moe_params)}")
    print(f"Det head params: {_numel(det_head_params)}")
    print(f"Seg head params: {_numel(seg_params)}")
    print(f"Cnt head params: {_numel(cnt_head_params)}")
    print(f"All task-head params (det+seg+cnt): {_numel(det_head_params) + _numel(seg_params) + _numel(cnt_head_params)}")

    det_proj_params = _numel(model.detector.backbone.proj.parameters())
    det_rpn_params = _numel(model.detector.rpn.head.parameters())
    det_roi_head_params = _numel(model.detector.roi_heads.box_head.parameters())
    det_roi_pred_params = _numel(model.detector.roi_heads.box_predictor.parameters())
    print("\n[det-head breakdown]")
    print(f"  detector.backbone.proj: {det_proj_params}")
    print(f"  detector.rpn.head: {det_rpn_params}")
    print(f"  detector.roi_heads.box_head: {det_roi_head_params}")
    print(f"  detector.roi_heads.box_predictor: {det_roi_pred_params}")

    print("\n=== Gradient inputs for loss-weight network ===")
    print("train.py uses only these 'last-layer' params to build gradient vectors:")
    _print_named_params("det_last_params", det_last_named)
    _print_named_params("seg_last_params", seg_last_named)
    _print_named_params("cnt_last_params", cnt_last_named)

    print("\n[gradient vector dims]")
    print(f"det_last_dim: {det_last_dim}")
    print(f"seg_last_dim: {seg_last_dim}")
    print(f"cnt_last_dim: {cnt_last_dim}")
    print(f"joint_input_dim (det+seg+cnt): {joint_in}")
    print(f"cnt_grad_hidden_dim: {cnt_grad_hidden_dim}")

    print("\n[loss-weight net parameter counts by architecture]")
    print(f"per_task_shared phi params: {per_task_shared_weight_net_params}")
    print(f"joint phi params: {joint_weight_net_params}")


if __name__ == "__main__":
    main()

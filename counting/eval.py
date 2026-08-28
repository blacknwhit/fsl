import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from dataset import DSACADensityH5Dataset
from models import DinoV3Density


# Ensure repo root on sys.path so lora_multitask imports succeed
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _state_dict_has_lora(state_dict) -> bool:
    try:
        keys = state_dict.keys()
    except Exception:
        return False
    for k in keys:
        if isinstance(k, str) and (".lora_a" in k.lower() or ".lora_b" in k.lower()):
            return True
    return False


def _maybe_inject_lora(backbone_module, checkpoint_obj) -> bool:
    if not isinstance(checkpoint_obj, dict):
        return True
    sd = checkpoint_obj.get("backbone")
    if not isinstance(sd, dict):
        return True
    if not _state_dict_has_lora(sd):
        return True
    try:
        from lora_multitask.lora import LoRAConfig, inject_lora_into_dinov3_ffn
    except Exception as e:
        print(f"[Cnt][Error] Detected LoRA backbone weights but failed to import LoRA utilities: {e}")
        return False
    lora_meta = checkpoint_obj.get("lora") if isinstance(checkpoint_obj.get("lora"), dict) else {}
    cfg = LoRAConfig(
        rank=int(lora_meta.get("rank", 8)),
        alpha=float(lora_meta.get("alpha", 16.0)),
        dropout=float(lora_meta.get("dropout", 0.0)),
    )
    try:
        replaced = inject_lora_into_dinov3_ffn(backbone_module, cfg=cfg)
        print(f"[Cnt] Injected LoRA into dinov3 FFN (replaced_linear={replaced})")
        return True
    except Exception as e:
        print(f"[Cnt][Error] Failed to inject LoRA modules into backbone: {e}")
        return False


class NormalizeTransform:
    """Picklable transform for spawn DataLoader workers."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, density):
        # dataset 通常返回 PIL.Image + density Tensor
        if not torch.is_tensor(img):
            img = TF.to_tensor(img)
        img = TF.normalize(img, mean=self.mean, std=self.std)
        return img, density


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DINOv3 counting baseline on DSACA/VisDrone")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/nas/liyangguang103/newdataset/CD-Count/DSACA",
        help="DSACA root containing test_data_class8/",
    )
    parser.add_argument("--test-dir", type=str, default=None, help="override test split dir")
    parser.add_argument("--checkpoint", type=str, required=True, help="trained checkpoint (head-only or full model)")
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=448)
    aspect_group = parser.add_mutually_exclusive_group()
    aspect_group.add_argument("--keep-aspect", dest="keep_aspect", action="store_true")
    aspect_group.add_argument("--no-keep-aspect", dest="keep_aspect", action="store_false")
    parser.set_defaults(keep_aspect=True)
    parser.add_argument("--model-name", type=str, default="dinov3_vitl16")

    parser.add_argument(
        "--backbone-checkpoint",
        type=str,
        default="/data/xiangyuyue/ULLM-zf/fsl-20260209/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        help="path to DINOv3 backbone weights (used when checkpoint does not include backbone weights)",
    )

    parser.add_argument(
        "--backbone-source",
        type=str,
        default="auto",
        choices=["auto", "pretrained", "ckpt"],
        help=(
            "backbone weights source: "
            "'auto' = use checkpoint backbone if present else use --backbone-checkpoint; "
            "'pretrained' = always use --backbone-checkpoint (load only head from checkpoint); "
            "'ckpt' = always use backbone from --checkpoint"
        ),
    )

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--stats-dir",
        type=str,
        default=None,
        help="directory to save evaluation metrics json. Default: <checkpoint_dir>/stats",
    )

    parser.add_argument(
        "--check-load-only",
        action="store_true",
        help="only build model and load checkpoint, print load summary, then exit (no dataset/eval)",
    )
    return parser.parse_args()


def _extract_state_dict(ckpt_obj):
    # 兼容 {"model": state_dict, ...} 或直接 state_dict
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        return ckpt_obj["model"]
    if isinstance(ckpt_obj, dict):
        # 也可能直接就是 state_dict（key 像 'backbone.xxx'/'head.xxx'）
        return ckpt_obj
    return None


def _state_dict_has_backbone(state_dict: dict) -> bool:
    markers = ("backbone.", "dinov3", "dino", "vit", "transformer", "patch_embed", "blocks.")
    for k in state_dict.keys():
        if any(m in k for m in markers):
            return True
    return False


def _filter_head_only(state_dict: dict) -> dict:
    head_keys = [k for k in state_dict.keys() if k.startswith("head.")]
    return {k: state_dict[k] for k in head_keys} if head_keys else {}


def main():
    args = parse_args()
    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    stats_dir = Path(args.stats_dir) if args.stats_dir else (ckpt_path.parent / "stats")
    stats_dir.mkdir(parents=True, exist_ok=True)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # 新格式：{"backbone":..., "head":...}
    ckpt_has_backbone = isinstance(checkpoint, dict) and isinstance(checkpoint.get("backbone", None), dict)
    ckpt_has_head = isinstance(checkpoint, dict) and (
        isinstance(checkpoint.get("head", None), dict) or isinstance(checkpoint.get("cnt_head", None), dict)
    )

    # 旧格式：{"model": state_dict} 或直接 state_dict
    state_dict = _extract_state_dict(checkpoint)
    if not ckpt_has_backbone and isinstance(state_dict, dict):
        ckpt_has_backbone = _state_dict_has_backbone(state_dict)

    # backbone 来源决策
    if args.backbone_source == "pretrained":
        backbone_from_ckpt = False
    elif args.backbone_source == "ckpt":
        if not ckpt_has_backbone:
            raise ValueError("--backbone-source=ckpt but checkpoint has no backbone weights")
        backbone_from_ckpt = True
    else:  # auto
        backbone_from_ckpt = ckpt_has_backbone

    # 构建模型：如果 backbone 不来自 ckpt，就用 --backbone-checkpoint 初始化
    model = DinoV3Density(
        model_name=args.model_name,
        num_classes=args.num_classes,
        image_size=args.image_size,
        # 如果 backbone 将从 --checkpoint 加载，这里就不要触发“随机初始化”的误导性 warning
        pretrained=not backbone_from_ckpt,
        checkpoint_path=None if backbone_from_ckpt else args.backbone_checkpoint,
        freeze_backbone=False,
    ).to(device)

    # 加载权重
    loaded_any = False

    # A) 新格式：分别加载 backbone/head（优先）
    if backbone_from_ckpt and isinstance(checkpoint, dict) and isinstance(checkpoint.get("backbone"), dict):
        ok = _maybe_inject_lora(model.backbone, checkpoint)
        if not ok:
            raise RuntimeError(
                "Detected LoRA weights in checkpoint backbone but failed to inject. "
                "Run with Python>=3.10 and ensure repo root is importable (PYTHONPATH includes repo root)."
            )
        print("Loading backbone weights from --checkpoint...")
        missing, unexpected = model.backbone.load_state_dict(checkpoint["backbone"], strict=False)
        print("Loaded backbone weights from --checkpoint")
        loaded_any = True
        total_keys = len(model.backbone.state_dict())
        provided_keys = len(checkpoint["backbone"]) if isinstance(checkpoint.get("backbone"), dict) else -1
        matched = max(total_keys - len(missing), 0)
        matched_pct = (100.0 * matched / max(total_keys, 1))
        print(
            "[Cnt][LoadSummary] backbone: "
            f"provided={provided_keys}, total={total_keys}, matched≈{matched} ({matched_pct:.1f}%), "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if matched_pct < 50.0:
            print("[Cnt][Warn] Backbone matched <50%; checkpoint format may not match this eval script.")

    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("head"), dict):
        print("Loading head weights from --checkpoint...")
        missing, unexpected = model.head.load_state_dict(checkpoint["head"], strict=False)
        print("Loaded head weights from --checkpoint")
        loaded_any = True
        print(f"[Cnt][LoadSummary] head: missing={len(missing)}, unexpected={len(unexpected)}")
    elif isinstance(checkpoint, dict) and isinstance(checkpoint.get("cnt_head"), dict):
        print("Loading head weights (cnt_head) from --checkpoint...")
        missing, unexpected = model.head.load_state_dict(checkpoint["cnt_head"], strict=False)
        print("Loaded head weights (cnt_head) from --checkpoint")
        loaded_any = True
        print(f"[Cnt][LoadSummary] head(cnt_head): missing={len(missing)}, unexpected={len(unexpected)}")

    # B) 旧格式：从 state_dict 加载（可能是整模，也可能只含 head.*）
    if not loaded_any and isinstance(state_dict, dict):
        if (not backbone_from_ckpt) and ckpt_has_backbone:
            # 强制 pretrained backbone 的场景：只加载 head
            head_only = _filter_head_only(state_dict)
            if not head_only and ckpt_has_head:
                head_only = checkpoint["head"]
            if head_only:
                missing, unexpected = model.load_state_dict(head_only, strict=False)
                if missing or unexpected:
                    print(f"Loaded head-only with missing keys: {missing}, unexpected: {unexpected}")
                loaded_any = True
        else:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"Loaded state_dict with missing keys: {missing}, unexpected: {unexpected}")
            loaded_any = True

    if not loaded_any:
        raise ValueError("Could not load checkpoint: unrecognized format")

    if args.check_load_only:
        print(
            f"[LoadCheck][cnt] backbone_expected={len(model.backbone.state_dict())}, head_expected={len(model.head.state_dict())}"
        )
        print("[LoadCheck] done (no dataset/eval)")
        return

    model.eval()

    data_root = Path(args.data_root)
    test_dir = Path(args.test_dir) if args.test_dir else data_root / "test_data_class8"

    ds = DSACADensityH5Dataset(
        str(test_dir),
        num_classes=args.num_classes,
        transform=NormalizeTransform(mean, std),
        image_size=args.image_size,
        keep_aspect=args.keep_aspect,
    )

    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    loader = DataLoader(ds, **loader_kwargs)

    # 评估：用 GT density sum 得到 GT counts，预测 counts 来算 MAE/RMSE
    sum_abs = torch.zeros(args.num_classes, dtype=torch.float64)
    sum_sq = torch.zeros(args.num_classes, dtype=torch.float64)
    n = 0

    with torch.no_grad():
        for imgs, dens in loader:
            imgs = imgs.to(device, non_blocking=True)
            dens = dens.to(device, non_blocking=True)  # [B,C,H,W]

            _, pred_counts = model(imgs)  # [B,C]
            gt_counts = dens.flatten(2).sum(dim=2)  # [B,C]

            err = (pred_counts - gt_counts).to(torch.float64)
            sum_abs += err.abs().sum(dim=0).cpu()
            sum_sq += (err * err).sum(dim=0).cpu()
            n += imgs.shape[0]

    mae = sum_abs / max(1, n)
    rmse = (sum_sq / max(1, n)).sqrt()

    print("\nPer-class MAE / RMSE:")
    for c in range(args.num_classes):
        print(f"  class {c:02d}: MAE={mae[c].item():.3f}  RMSE={rmse[c].item():.3f}")
    print(f"Overall: MAE={mae.mean().item():.3f}  RMSE={rmse.mean().item():.3f}")

    split_name = test_dir.name or "count"
    metrics = {
        "task": "counting",
        "data_root": str(data_root),
        "test_dir": str(test_dir),
        "checkpoint": str(ckpt_path),
        "num_classes": int(args.num_classes),
        "image_size": int(args.image_size),
        "keep_aspect": bool(args.keep_aspect),
        "model_name": str(args.model_name),
        "backbone_source": str(args.backbone_source),
        "per_class_mae": [float(x.item()) for x in mae],
        "per_class_rmse": [float(x.item()) for x in rmse],
        "mae": float(mae.mean().item()),
        "rmse": float(rmse.mean().item()),
        "n_images": int(n),
    }
    metrics_path = stats_dir / f"metrics_cnt_{split_name}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np


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
        print(f"[Seg][Error] Detected LoRA backbone weights but failed to import LoRA utilities: {e}")
        return False
    lora_meta = checkpoint_obj.get("lora") if isinstance(checkpoint_obj.get("lora"), dict) else {}
    cfg = LoRAConfig(
        rank=int(lora_meta.get("rank", 8)),
        alpha=float(lora_meta.get("alpha", 16.0)),
        dropout=float(lora_meta.get("dropout", 0.0)),
    )
    try:
        replaced = inject_lora_into_dinov3_ffn(backbone_module, cfg=cfg)
        print(f"[Seg] Injected LoRA into dinov3 FFN (replaced_linear={replaced})")
        return True
    except Exception as e:
        print(f"[Seg][Error] Failed to inject LoRA modules into backbone: {e}")
        return False

from dataset import SegmentationDataset
from models import DinoV3Segmentation
from utils import per_class_iou_from_confusion, update_confusion_matrix


def build_transform(mean, std):
    def _transform(img, mask):
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=mean, std=std)
        mask_tensor = torch.as_tensor(TF.pil_to_tensor(mask), dtype=torch.long).squeeze(0)
        return img, mask_tensor

    return _transform


def get_color_palette(num_classes):
    """生成区分度高的颜色调色板"""
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    
    # 预定义一些高对比度的颜色
    colors = [
        [0, 0, 0],        # 0: 黑色 (背景)
        [128, 0, 0],      # 1: 深红
        [0, 128, 0],      # 2: 深绿
        [128, 128, 0],    # 3: 橄榄绿
        [0, 0, 128],      # 4: 深蓝
        [128, 0, 128],    # 5: 紫色
        [0, 128, 128],    # 6: 青色
        [128, 128, 128],  # 7: 灰色
        [255, 0, 0],      # 8: 红色
        [0, 255, 0],      # 9: 绿色
        [0, 0, 255],      # 10: 蓝色
    ]
    
    for i in range(min(num_classes, len(colors))):
        palette[i] = colors[i]
    
    # 如果类别数超过预定义颜色,自动生成
    for i in range(len(colors), num_classes):
        palette[i] = [
            (i * 67) % 256,
            (i * 113) % 256,
            (i * 197) % 256
        ]
    
    return palette


def colorize_mask(mask, num_classes):
    """将类别索引掩码转换为彩色图像"""
    palette = get_color_palette(num_classes)
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id in range(num_classes):
        color_mask[mask == class_id] = palette[class_id]
    
    return color_mask


def concatenate_images(img, gt_mask, pred_mask):
    """将原图、GT mask和预测mask横向拼接成一张图"""
    # 确保所有图像尺寸一致
    h, w = img.shape[:2]
    
    # 如果预测mask尺寸不匹配,需要resize
    if pred_mask.shape[:2] != (h, w):
        pred_mask_img = Image.fromarray(pred_mask)
        pred_mask_img = pred_mask_img.resize((w, h), Image.NEAREST)
        pred_mask = np.array(pred_mask_img)
    
    # 如果GT mask尺寸不匹配,也需要resize
    if gt_mask.shape[:2] != (h, w):
        gt_mask_img = Image.fromarray(gt_mask)
        gt_mask_img = gt_mask_img.resize((w, h), Image.NEAREST)
        gt_mask = np.array(gt_mask_img)
    
    # 创建一个空白画布: [height, width*3, channels]
    combined = np.zeros((h, w * 3, 3), dtype=np.uint8)
    
    # 放置三张图
    combined[:, :w] = img              # 左边: 原图
    combined[:, w:2*w] = gt_mask       # 中间: GT mask
    combined[:, 2*w:] = pred_mask      # 右边: 预测mask
    
    return combined


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DINOv3 ViT segmentation model")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/test",
        help="test data root (images/, masks/)",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="path to trained checkpoint")
    parser.add_argument("--num-classes", type=int, default=11)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--model-name", type=str, default="dinov3_vitl16")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-preds", type=str, default=None, help="dir to save predicted masks")
    parser.add_argument("--save-images", action="store_true", help="also save RGB image/GT/pred concatenation")
    parser.add_argument(
        "--stats-dir",
        type=str,
        default=None,
        help="directory to save evaluation metrics json. Default: <checkpoint_dir>/stats",
    )

    # 新增：可视化拼接图输出目录（原图|GT|Pred）
    parser.add_argument(
        "--vis-dir",
        type=str,
        default=None,
        help="dir to save visualization images: [Original | GT | Pred] concatenated into one image",
    )

    parser.add_argument(
        "--backbone-checkpoint",
        type=str,
        default="/data/xiangyuyue/ULLM-zf/fsl-20260209/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        help="path to DINOv3 backbone weights",
    )

    parser.add_argument(
        "--check-load-only",
        action="store_true",
        help="only build model and load checkpoint, print load summary, then exit (no dataset/eval)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    if device.type == "cuda":
        # 确保“当前 device”就是你传入的那张卡，避免默认落到 cuda:0
        # torch.device("cuda") 的 index 可能是 None，此时不要调用 set_device(None)
        if device.index is not None:
            torch.cuda.set_device(device.index)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    # If checkpoint already contains backbone weights, skip loading a separate pretrained file.
    # This avoids failing on machine-specific default paths and keeps eval behavior consistent.
    has_backbone = isinstance(checkpoint, dict) and isinstance(checkpoint.get("backbone"), dict)
    backbone_ckpt = None if has_backbone else args.backbone_checkpoint

    model = DinoV3Segmentation(
        model_name=args.model_name,
        num_classes=args.num_classes,
        image_size=args.image_size,
        pretrained=backbone_ckpt is not None,
        checkpoint_path=backbone_ckpt,
    ).to(device)

    # 1) 新格式：优先分别加载 backbone/head（用于全参训练或只训head）
    loaded_any = False
    if isinstance(checkpoint, dict) and checkpoint.get("backbone") is not None:
        ok = _maybe_inject_lora(model.backbone, checkpoint)
        if not ok:
            raise RuntimeError(
                "Detected LoRA weights in checkpoint backbone but failed to inject. "
                "Run with Python>=3.10 and ensure repo root is importable (PYTHONPATH includes repo root)."
            )
        missing, unexpected = model.backbone.load_state_dict(checkpoint["backbone"], strict=False)
        loaded_any = True
        total_keys = len(model.backbone.state_dict())
        provided_keys = len(checkpoint["backbone"]) if isinstance(checkpoint.get("backbone"), dict) else -1
        matched = max(total_keys - len(missing), 0)
        matched_pct = (100.0 * matched / max(total_keys, 1))
        print(
            "[Seg][LoadSummary] backbone: "
            f"provided={provided_keys}, total={total_keys}, matched≈{matched} ({matched_pct:.1f}%), "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if matched_pct < 50.0:
            print("[Seg][Warn] Backbone matched <50%; checkpoint format may not match this eval script.")

    # - 单任务 checkpoint: {"head": ...}
    # - 多任务 checkpoint: {"seg_head": ...}
    if isinstance(checkpoint, dict) and (checkpoint.get("head") is not None or checkpoint.get("seg_head") is not None):
        head_state = checkpoint.get("head") if checkpoint.get("head") is not None else checkpoint.get("seg_head")
        missing, unexpected = model.head.load_state_dict(head_state, strict=False)
        loaded_any = True
        print(f"[Seg][LoadSummary] head: missing={len(missing)}, unexpected={len(unexpected)}")

    # 2) 旧格式：整模
    if not loaded_any and isinstance(checkpoint, dict) and checkpoint.get("model") is not None:
        missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
        loaded_any = True
        if missing or unexpected:
            print(f"Loaded model with missing keys: {missing}, unexpected: {unexpected}")

    # 3) 更旧：直接是 state_dict
    if not loaded_any:
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        if missing or unexpected:
            print(f"Loaded checkpoint with missing keys: {missing}, unexpected: {unexpected}")

    if args.check_load_only:
        print(
            f"[LoadCheck][seg] backbone_expected={len(model.backbone.state_dict())}, head_expected={len(model.head.state_dict())}"
        )
        print("[LoadCheck] done (no dataset/eval)")
        return

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    ds = SegmentationDataset(args.data_dir, transform=build_transform(mean, std), image_size=args.image_size)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

    model.eval()
    conf = torch.zeros((args.num_classes, args.num_classes), dtype=torch.int64)

    ckpt_path = Path(args.checkpoint)
    stats_dir = Path(args.stats_dir) if args.stats_dir else (ckpt_path.parent / "stats")
    stats_dir.mkdir(parents=True, exist_ok=True)

    save_dir = Path(args.save_preds) if args.save_preds else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    if vis_dir:
        vis_dir.mkdir(parents=True, exist_ok=True)

    # 获取数据集的图像和mask路径列表
    data_root = Path(args.data_dir)
    image_paths = sorted((data_root / "images").glob("*.png")) + sorted((data_root / "images").glob("*.jpg"))
    mask_paths = sorted((data_root / "masks").glob("*.png"))

    with torch.no_grad():
        for idx, (imgs, masks) in enumerate(loader):
            if idx % 10 == 0:
                print(f"Processing {idx}/{len(loader)}...")

            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            logits = model(imgs)

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            update_confusion_matrix(
                conf=conf,
                logits_or_preds=logits.detach(),
                target=masks.detach(),
                num_classes=args.num_classes,
                ignore_indices=(255, 11),
            )

            # 新增：只要设置了 --vis-dir，就保存拼接可视化图
            if vis_dir:
                raw_img_pil = Image.open(image_paths[idx]).convert("RGB")
                orig_w, orig_h = raw_img_pil.size
                raw_img = np.array(raw_img_pil)

                raw_mask = Image.open(mask_paths[idx])
                gt_mask = np.array(raw_mask).astype(np.uint8)
                color_gt = colorize_mask(gt_mask, args.num_classes)

                preds = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                preds_img = Image.fromarray(preds)
                preds_resized = preds_img.resize((orig_w, orig_h), Image.NEAREST)
                preds_resized = np.array(preds_resized).astype(np.uint8)
                color_pred = colorize_mask(preds_resized, args.num_classes)

                combined = concatenate_images(raw_img, color_gt, color_pred)
                Image.fromarray(combined).save(vis_dir / f"{idx:05d}.png")

    per_class_iou, miou = per_class_iou_from_confusion(conf)

    print("\nPer-class IoU (%):")
    for cls_id in range(args.num_classes):
        v = per_class_iou[cls_id].item()
        if v != v:  # NaN
            print(f"  class {cls_id:02d}: N/A (union=0)")
        else:
            print(f"  class {cls_id:02d}: {v * 100:.2f}")

    print(f"Mean IoU (%): {miou.item() * 100:.2f}")

    split_name = Path(args.data_dir).name or "seg"
    metrics = {
        "task": "segmentation",
        "data_dir": str(Path(args.data_dir)),
        "checkpoint": str(ckpt_path),
        "num_classes": int(args.num_classes),
        "image_size": int(args.image_size),
        "model_name": str(args.model_name),
        "ignore_indices": [255, 11],
        "per_class_iou": [
            (None if (v != v) else float(v.item()))  # NaN -> None
            for v in per_class_iou
        ],
        "miou": (None if (float(miou.item()) != float(miou.item())) else float(miou.item())),
    }
    metrics_path = stats_dir / f"metrics_seg_{split_name}.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {metrics_path}")
    if save_dir:
        print(f"Combined results saved to: {save_dir}")
        print(f"Format: [Original Image | GT Mask | Predicted Mask]")
    if vis_dir:
        print(f"Visualization images saved to: {vis_dir}")
        print("Format: [Original Image | GT Mask | Predicted Mask]")


if __name__ == "__main__":
    main()

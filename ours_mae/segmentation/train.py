import argparse
import logging
import math
from copy import deepcopy
import random
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from dataset import SegmentationDataset
from models import DinoV3Segmentation
from utils import per_class_iou_from_confusion, save_checkpoint, update_confusion_matrix


def build_transform(train: bool, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
    def _transform(img, mask):
        if train:
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask_ = TF.hflip(mask)
                mask = mask_
            if random.random() < 0.5:
                img = TF.vflip(img)
                mask_ = TF.vflip(mask)
                mask = mask_
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=mean, std=std)
        mask_tensor = torch.as_tensor(TF.pil_to_tensor(mask), dtype=torch.long).squeeze(0)
        return img, mask_tensor

    return _transform


def parse_args():
    parser = argparse.ArgumentParser(description="Train DINOv3 ViT + simple segmentation head")
    parser.add_argument(
        "--train-dir",
        type=str,
        default="/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/train_10per",
        help="train data root (images/, masks/)",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/val",
        help="val data root (images/, masks/)",
    )
    parser.add_argument("--num-classes", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--model-name", type=str, default="dinov3_vitl16")
    parser.add_argument("--no-pretrained", action="store_true", help="do not load pretrained weights")
    parser.add_argument(
        "--backbone-checkpoint",
        type=str,
        default="/nas/liyangguang103/old_fscd/CD-FSOD/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        help="path to DINOv3 checkpoint",
    )
    parser.add_argument("--save-path", type=str, default="runs/dinov3_seg_10per.pt")
    parser.add_argument("--log-file", type=str, default="runs/train_10per.log", help="path to training log file")
    parser.add_argument("--amp", action="store_true", help="use mixed precision")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:2" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--full-finetune",
        action="store_true",
        help="enable full-parameter finetuning (by default only trains segmentation head / non-backbone params)",
    )
    return parser.parse_args()


def setup_logger(log_path: str | None):
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    # console
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    # file
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(fh)
    return logger


def _configure_finetune(model: torch.nn.Module, full_finetune: bool) -> None:
    """
    full_finetune=True:  全参数训练
    full_finetune=False: 默认冻结backbone(若存在)，仅训练head/非backbone参数；若无法识别则回退为全参数训练
    """
    if full_finetune:
        for p in model.parameters():
            p.requires_grad = True
        return

    # 先全部冻结，再有选择地解冻
    for p in model.parameters():
        p.requires_grad = False

    # 优先按显式backbone冻结/非backbone解冻
    if hasattr(model, "backbone") and isinstance(getattr(model, "backbone"), torch.nn.Module):
        for name, p in model.named_parameters():
            if not name.startswith("backbone."):
                p.requires_grad = True
    else:
        # 兜底：按名字解冻常见head/分类层
        head_keywords = ("head", "seg_head", "decode_head", "classifier", "fc", "proj", "projection")
        for name, p in model.named_parameters():
            n = name.lower()
            if any(k in n for k in head_keywords):
                p.requires_grad = True

    # 如果最终没有任何可训练参数，回退为全参数训练，避免“训练不动”
    if not any(p.requires_grad for p in model.parameters()):
        for p in model.parameters():
            p.requires_grad = True


def main():
    args = parse_args()
    device = torch.device(args.device)
    logger = setup_logger(args.log_file)

    print("Loading datasets...")
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_ds = SegmentationDataset(
        args.train_dir, transform=build_transform(True, mean, std), image_size=args.image_size
    )
    val_ds = SegmentationDataset(
        args.val_dir, transform=build_transform(False, mean, std), image_size=args.image_size
    )
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    print("Creating data loaders...")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("Building model...")
    model = DinoV3Segmentation(
        model_name=args.model_name,
        num_classes=args.num_classes,
        image_size=args.image_size,
        pretrained=not args.no_pretrained,
        checkpoint_path=args.backbone_checkpoint,
    ).to(device)

    print("Model loaded successfully.")
    if args.full_finetune:
        print("Full finetune enabled: training all parameters.")
    else:
        print("Full finetune disabled: freezing backbone (training head / non-backbone parameters).")

    _configure_finetune(model, full_finetune=args.full_finetune)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler('cuda', enabled=args.amp)

    save_dir = Path(args.save_path).parent
    best_metric = -math.inf
    best_path = save_dir / "best_miou.pt"
    best_state = None
    best_epoch = None

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*50}")
        
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        
        print(f"Loading first batch...")
        for batch_idx, (imgs, masks) in enumerate(train_loader):
            if batch_idx == 0:
                print(f"First batch loaded: imgs {imgs.shape}, masks {masks.shape}")
            
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            if batch_idx == 0:
                print(f"Data moved to {device}")
            
            optimizer.zero_grad()
            
            if batch_idx == 0:
                print(f"Running forward pass...")
            
            with torch.amp.autocast('cuda', enabled=args.amp):
                logits = model(imgs)
                loss = F.cross_entropy(logits, masks)
            
            if batch_idx == 0:
                print(f"Forward pass done, loss: {loss.item():.4f}")
                print(f"Running backward pass...")
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * imgs.size(0)
            
            if batch_idx == 0:
                print(f"First iteration complete!")
            
            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / ((batch_idx + 1) * args.batch_size)
                print(f"  Batch [{batch_idx+1}/{num_batches}] Loss: {avg_loss:.4f}")
        
        avg_loss = total_loss / len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        conf = torch.zeros((args.num_classes, args.num_classes), dtype=torch.int64)
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                logits = model(imgs)
                val_loss += F.cross_entropy(logits, masks).item() * imgs.size(0)
                update_confusion_matrix(
                    conf=conf,
                    logits_or_preds=logits.detach(),
                    target=masks.detach(),
                    num_classes=args.num_classes,
                    ignore_indices=(255, 11),
                )

        val_loss /= len(val_ds)
        _, val_iou = per_class_iou_from_confusion(conf)
        val_iou = float(val_iou.item())

        logger.info(
            f"Epoch {epoch}/{args.epochs} | train loss {avg_loss:.4f} | "
            f"val loss {val_loss:.4f} | mIoU {val_iou:.4f}"
        )

        # 记录最佳mIoU所在epoch（只记录，不保存）
        if val_iou > best_metric:
            best_metric = val_iou
            best_epoch = epoch
            # 缓存当前最优权重（仅保留在内存中）
            best_state = deepcopy(model.state_dict())
            logger.info(f"New best val miou: {best_metric:.4f} (epoch {epoch})")

    # 训练完成后，仅保存一次“验证最优”的权重
    if best_state is None:
        logger.warning("No best checkpoint found; saving final model weights instead.")
    else:
        model.load_state_dict(best_state)
    logger.info(
        f"Training finished. Saving best checkpoint to {best_path} "
        f"(val_miou={best_metric:.4f}, epoch={best_epoch})"
    )
    save_checkpoint(model, optimizer, best_epoch or args.epochs, str(best_path))


if __name__ == "__main__":
    main()

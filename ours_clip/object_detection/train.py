import argparse
import logging
import math
import random
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
from torch.amp import GradScaler
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from dataset import CocoDetectionDataset, collate_fn
from models import DinoV3FasterRCNN
from utils import save_checkpoint


def build_transform(train: bool):
    def _transform(img, target):
        if train and random.random() < 0.5:
            width, _ = img.size
            img = TF.hflip(img)
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target = dict(target)
            target["boxes"] = boxes
        img = TF.to_tensor(img)
        return img, target

    return _transform


def parse_args():
    parser = argparse.ArgumentParser(description="Train DINOv3 ViT-L/16 + Faster R-CNN baseline")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/nas/liyangguang103/newdataset/CD-ObjectDetection/DIOR/coco",
        help="DIOR COCO root containing annotations/ and images/",
    )
    parser.add_argument("--train-ann", type=str, default=None, help="path to instances_train.json")
    parser.add_argument("--val-ann", type=str, default=None, help="path to instances_val.json")
    parser.add_argument("--train-img-dir", type=str, default=None, help="train images dir")
    parser.add_argument("--val-img-dir", type=str, default=None, help="val images dir")

    parser.add_argument("--num-classes", type=int, default=None, help="foreground class count (auto if None)")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--model-name", type=str, default="dinov3_vitl16")
    parser.add_argument(
        "--backbone-checkpoint",
        type=str,
        default="/nas/liyangguang103/old_fscd/CD-FSOD/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        help="path to DINOv3 backbone checkpoint",
    )
    parser.add_argument("--out-channels", type=int, default=256, help="projection channels into detector head")
    parser.add_argument("--unfreeze-backbone", action="store_true", help="train backbone as well")
    parser.add_argument("--save-path", type=str, default="runs/dinov3_det.pt")
    parser.add_argument("--log-file", type=str, default="runs/train_det.log")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def setup_logger(log_path: str | None):
    logger = logging.getLogger("train_det")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(fh)
    return logger


def _box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(0.0, br - tl)
    inter = wh[..., 0] * wh[..., 1]
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-12)


def _ap_from_pr(rec: np.ndarray, prec: np.ndarray) -> float:
    # COCO-style 101-point interpolated AP
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = prec[rec >= t].max() if np.any(rec >= t) else 0.0
        ap += p
    return ap / 101.0


@torch.no_grad()
def eval_ap50(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    score_thresh: float = 0.0,
) -> float:
    model.eval()
    preds_by_cls = {c: [] for c in range(1, num_classes + 1)}
    gts_by_cls = {c: {} for c in range(1, num_classes + 1)}

    img_counter = 0
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            img_id = int(tgt.get("image_id", torch.tensor([img_counter])).item())
            img_counter += 1

            gt_boxes = tgt["boxes"].detach().cpu().numpy()
            gt_labels = tgt["labels"].detach().cpu().numpy().astype(int)
            for box, cls in zip(gt_boxes, gt_labels):
                gts_by_cls[cls].setdefault(img_id, []).append(box)

            pred_boxes = out["boxes"].detach().cpu().numpy()
            pred_labels = out["labels"].detach().cpu().numpy().astype(int)
            pred_scores = out["scores"].detach().cpu().numpy()

            keep = pred_scores >= score_thresh
            for box, cls, score in zip(pred_boxes[keep], pred_labels[keep], pred_scores[keep]):
                preds_by_cls[cls].append((img_id, float(score), box))

    ap_list = []
    for cls in range(1, num_classes + 1):
        preds = preds_by_cls[cls]
        gts = gts_by_cls[cls]
        num_gt = sum(len(v) for v in gts.values())
        if num_gt == 0:
            continue

        preds.sort(key=lambda x: x[1], reverse=True)
        matched = {k: [False] * len(v) for k, v in gts.items()}

        tp = np.zeros(len(preds), dtype=np.float32)
        fp = np.zeros(len(preds), dtype=np.float32)

        for i, (img_id, score, pbox) in enumerate(preds):
            if img_id not in gts:
                fp[i] = 1.0
                continue
            gt_boxes = np.array(gts[img_id], dtype=np.float32)
            ious = _box_iou_np(np.array([pbox], dtype=np.float32), gt_boxes)[0]
            j = int(np.argmax(ious)) if ious.size > 0 else -1
            if j >= 0 and ious[j] >= 0.5 and not matched[img_id][j]:
                tp[i] = 1.0
                matched[img_id][j] = True
            else:
                fp[i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec = tp_cum / max(num_gt, 1)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        ap_list.append(_ap_from_pr(rec, prec))

    return float(np.mean(ap_list)) if ap_list else 0.0


def main():
    args = parse_args()
    device = torch.device(args.device)
    logger = setup_logger(args.log_file)

    data_root = Path(args.data_root)
    # Allow passing DIOR root or coco root
    if not (data_root / "annotations").exists() and (data_root / "coco" / "annotations").exists():
        data_root = data_root / "coco"
    train_ann = Path(args.train_ann) if args.train_ann else data_root / "annotations" / "instances_train.json"
    val_ann = Path(args.val_ann) if args.val_ann else data_root / "annotations" / "instances_val.json"
    train_img_dir = Path(args.train_img_dir) if args.train_img_dir else data_root / "images" / "train"
    val_img_dir = Path(args.val_img_dir) if args.val_img_dir else data_root / "images" / "val"

    print("Loading datasets...")
    train_ds = CocoDetectionDataset(str(train_ann), str(train_img_dir), transform=build_transform(True))
    val_ds = CocoDetectionDataset(str(val_ann), str(val_img_dir), transform=build_transform(False))
    num_classes = args.num_classes or train_ds.num_classes
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}, Classes: {num_classes}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print("Building model...")
    model = DinoV3FasterRCNN(
        num_classes=num_classes,
        model_name=args.model_name,
        image_size=args.image_size,
        checkpoint_path=args.backbone_checkpoint,
        out_channels=args.out_channels,
        freeze_backbone=not args.unfreeze_backbone,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(device.type, enabled=args.amp)
    autocast_device = device.type if device.type in {"cuda", "cpu"} else "cuda"

    best_metric = -math.inf
    save_dir = Path(args.save_path).parent
    best_path = save_dir / "best_ap50.pt"
    # last_path = save_dir / "last.pt"  # 不再每轮保存
    best_state = None
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device, non_blocking=True) for img in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.amp.autocast(autocast_device, enabled=args.amp):
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())

            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += losses.item()

        avg_train_loss = total_loss / max(len(train_loader), 1)

        model.train()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device, non_blocking=True) for img in images]
                targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                val_loss += sum(loss_dict.values()).item()

        avg_val_loss = val_loss / max(len(val_loader), 1)

        # 计算验证 AP50
        val_ap50 = eval_ap50(model, val_loader, device, num_classes=num_classes)

        logger.info(
            f"Epoch {epoch}/{args.epochs} | train loss {avg_train_loss:.4f} | "
            f"val loss {avg_val_loss:.4f} | val_ap50 {val_ap50:.4f}"
        )

        # 不再保存 last
        # save_checkpoint(model, optimizer, epoch, str(last_path))

        if val_ap50 > best_metric:
            best_metric = val_ap50
            best_state = deepcopy(model.state_dict())
            best_epoch = epoch
            print(f"[ckpt] new best cached (epoch={best_epoch}, val_ap50={best_metric:.4f})")

    # 训练结束后，仅保存一次“验证最优”的权重
    if best_state is not None:
        model.load_state_dict(best_state)
        save_checkpoint(model, optimizer, best_epoch or args.epochs, str(best_path))
        print(f"[ckpt] saved best_ap50 -> {best_path} (epoch={best_epoch}, val_ap50={best_metric:.4f})")


if __name__ == "__main__":
    main()

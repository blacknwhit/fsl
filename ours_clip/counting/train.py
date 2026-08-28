import argparse
import logging
import random
import time
import math
from pathlib import Path
from typing import Tuple
from datetime import datetime  # NEW

import torch
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from dataset import DSACADensityH5Dataset
from models import DinoV3Density
from utils import save_checkpoint


class DensityTransform:
    """Picklable transform for spawn DataLoader workers."""

    def __init__(self, train: bool, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        self.train = train
        self.mean = mean
        self.std = std

    def __call__(self, img: torch.Tensor, density: torch.Tensor):
        if self.train:
            if random.random() < 0.5:
                img = torch.flip(img, dims=[2])
                density = torch.flip(density, dims=[2])
            if random.random() < 0.5:
                img = torch.flip(img, dims=[1])
                density = torch.flip(density, dims=[1])
        img = TF.normalize(img, mean=self.mean, std=self.std)
        return img, density


def parse_args():
    parser = argparse.ArgumentParser(description="Train DINOv3 ViT multi-class counting with DSACA density maps")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/nas/liyangguang103/newdataset/CD-Count/DSACA",
        help="DSACA root containing train_data_class8/ val_data_class8/",
    )
    parser.add_argument("--train-dir", type=str, default=None, help="override train split dir")
    parser.add_argument("--val-dir", type=str, default=None, help="override val split dir")
    parser.add_argument("--num-classes", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--backbone-lr-mult",
        type=float,
        default=0.1,
        help="backbone lr = lr * backbone_lr_mult (head keeps lr). default: 0.1",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--count-loss-weight", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=448)
    aspect_group = parser.add_mutually_exclusive_group()
    aspect_group.add_argument(
        "--keep-aspect",
        dest="keep_aspect",
        action="store_true",
        help="resize long side to image-size then pad (default)",
    )
    aspect_group.add_argument(
        "--no-keep-aspect",
        dest="keep_aspect",
        action="store_false",
        help="resize directly to square image-size",
    )
    parser.set_defaults(keep_aspect=True)
    parser.add_argument("--model-name", type=str, default="dinov3_vitl16")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--backbone-checkpoint",
        type=str,
        default=None,
        help="path to DINOv3 checkpoint",
    )

    # === 新增：显式控制是否全参微调 ===
    ft = parser.add_mutually_exclusive_group()
    ft.add_argument(
        "--full-finetune",
        dest="full_finetune",
        action="store_true",
        help="train backbone + head (full parameter finetune)",
    )
    ft.add_argument(
        "--freeze-backbone",
        dest="full_finetune",
        action="store_false",
        help="freeze backbone, train head only",
    )
    #parser.set_defaults(full_finetune=True)

    # NEW: run directory control (default: runs/<YYYYMMDD>[/_xx])
    parser.add_argument(
        "--runs-root",
        type=str,
        default="runs",
        help="root directory to create date-named run folder (default: runs)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="override the auto-created runs/<date> directory",
    )

    # NOTE: keep these args for compatibility; actual paths will be rewritten into run dir
    parser.add_argument("--save-path", type=str, default="runs/dinov3_count.pt")
    parser.add_argument("--log-file", type=str, default="runs/train_count.log")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="log dataloader/iteration/GPU timing (adds overhead)",
    )
    parser.add_argument(
        "--profile-iters",
        type=int,
        default=0,
        help="when --profile is on, optionally stop after N iterations per epoch (0 = no limit)",
    )
    parser.add_argument(
        "--profile-warmup",
        type=int,
        default=5,
        help="ignore first N iterations for timing averages",
    )
    parser.add_argument(
        "--bench-dataloader",
        action="store_true",
        help="only iterate the dataloader and report timing; does not build the model",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=0.0,
        help="clip grad norm (0 disables). Useful for stabilizing early training.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    pm_group = parser.add_mutually_exclusive_group()
    pm_group.add_argument(
        "--pin-memory",
        dest="pin_memory",
        action="store_true",
        help="enable DataLoader pin_memory (default)",
    )
    pm_group.add_argument(
        "--no-pin-memory",
        dest="pin_memory",
        action="store_false",
        help="disable DataLoader pin_memory",
    )
    parser.set_defaults(pin_memory=True)
    parser.add_argument("--log-interval", type=int, default=50, help="print train loss every N iterations")
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def setup_logger(log_path: str | None):
    logger = logging.getLogger("train_count")
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


def _configure_finetune(model: torch.nn.Module, full_finetune: bool) -> None:
    """显式设置 backbone 是否参与训练（通过 requires_grad 控制）"""
    if not hasattr(model, "backbone"):
        return
    for p in model.backbone.parameters():
        p.requires_grad = bool(full_finetune)


def _make_date_run_dir(runs_root: str, run_dir_override: str | None) -> Path:
    """
    Create runs/<YYYYMMDD> (or use override). If already exists, append suffix _01, _02...
    """
    if run_dir_override:
        p = Path(run_dir_override)
        p.mkdir(parents=True, exist_ok=True)
        return p

    root = Path(runs_root)
    date_str = datetime.now().strftime("%Y%m%d")
    base = root / date_str
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return base

    # avoid overwrite: runs/20251218_01, _02 ...
    for i in range(1, 1000):
        cand = root / f"{date_str}_{i:02d}"
        if not cand.exists():
            cand.mkdir(parents=True, exist_ok=True)
            return cand

    raise RuntimeError(f"Failed to create unique run dir under: {root}")


def main():
    args = parse_args()

    # NEW: create date-named run folder, and force log/ckpt paths into it
    run_dir = _make_date_run_dir(args.runs_root, args.run_dir)

    # Put logs + checkpoints under run_dir
    # Use basename of --save-path as the "last" checkpoint name for backward compatibility
    save_name = Path(args.save_path).name if args.save_path else "last.pt"
    last_ckpt_path = run_dir / save_name
    best_ckpt_path = run_dir / (Path(save_name).stem + "_best_val_count_mae.pt")
    log_path = run_dir / (Path(args.log_file).name if args.log_file else "train_count.log")

    # Rewrite args so the rest of the code uses run_dir paths
    args.save_path = str(last_ckpt_path)
    args.log_file = str(log_path)

    # Speed: enable TF32 matmul on Ampere+ GPUs (safe for training in most cases).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    device = torch.device(args.device)
    logger = setup_logger(args.log_file)

    logger.info(f"[run] dir: {run_dir}")
    logger.info(f"[run] last_ckpt: {last_ckpt_path}")
    logger.info(f"[run] best_ckpt: {best_ckpt_path}")
    logger.info(f"[run] log: {log_path}")

    if device.type == "cuda":
        # Ensure current CUDA device matches args.device.
        # This avoids pin_memory thread defaulting to cuda:0 on multi-GPU systems.
        try:
            torch.cuda.set_device(device)
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True

    data_root = Path(args.data_root)
    train_dir = Path(args.train_dir) if args.train_dir else data_root / "train_data_class8"
    val_dir = Path(args.val_dir) if args.val_dir else data_root / "val_data_class8"

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    logger.info("Loading DSACA density datasets...")
    train_ds = DSACADensityH5Dataset(
        str(train_dir),
        num_classes=args.num_classes,
        transform=DensityTransform(True, mean, std),
        image_size=args.image_size,
        keep_aspect=args.keep_aspect,
    )
    val_ds = DSACADensityH5Dataset(
        str(val_dir),
        num_classes=args.num_classes,
        transform=DensityTransform(False, mean, std),
        image_size=args.image_size,
        keep_aspect=args.keep_aspect,
    )
    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    # h5py/HDF5 can hang with forked workers on some systems (esp. NFS/NAS).
    # Use spawn context when num_workers > 0.
    train_loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        drop_last=True,
    )
    val_loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
    )
    # When pin_memory is on and using CUDA, bind pinned-memory copies to the selected GPU.
    # Otherwise, PyTorch may default to cuda:0 which can cause OOM or slowdowns.
    if bool(args.pin_memory) and device.type == "cuda":
        train_loader_kwargs["pin_memory_device"] = str(device)
        val_loader_kwargs["pin_memory_device"] = str(device)
    if args.num_workers > 0:
        train_loader_kwargs["persistent_workers"] = True
        val_loader_kwargs["persistent_workers"] = True
        train_loader_kwargs["multiprocessing_context"] = "spawn"
        val_loader_kwargs["multiprocessing_context"] = "spawn"
        train_loader_kwargs["prefetch_factor"] = 4
        val_loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, **train_loader_kwargs)
    val_loader = DataLoader(val_ds, **val_loader_kwargs)

    if args.bench_dataloader:
        logger.info("[bench] Iterating train dataloader only...")
        prev_end = time.perf_counter()
        data_sum = 0.0
        iter_sum = 0.0
        steps = 0
        if bool(args.pin_memory) and device.type == "cuda":
            logger.info(f"[bench] pin_memory_device={device}")
        # Use --profile-iters if provided, else default to 200
        limit = args.profile_iters if args.profile_iters and args.profile_iters > 0 else 200
        for step, (imgs, densities) in enumerate(train_loader, start=1):
            t0 = time.perf_counter()
            data_time = t0 - prev_end
            # touch tensors to ensure transforms ran
            _ = (imgs.shape, densities.shape)
            t1 = time.perf_counter()
            iter_time = t1 - t0
            if step > args.profile_warmup:
                data_sum += data_time
                iter_sum += iter_time
                steps += 1
            prev_end = t1
            if step % max(1, args.log_interval) == 0:
                avg_data = (data_sum / steps) if steps else 0.0
                avg_iter = (iter_sum / steps) if steps else 0.0
                logger.info(f"[bench] step {step} | data {avg_data:.3f}s | iter {avg_iter:.3f}s")
            if step >= limit:
                break
        avg_data = (data_sum / steps) if steps else 0.0
        avg_iter = (iter_sum / steps) if steps else 0.0
        logger.info(f"[bench] done | avg data {avg_data:.3f}s | avg iter {avg_iter:.3f}s | steps {steps}")
        return

    logger.info("Building model...")
    if not args.backbone_checkpoint:
        raise SystemExit("--backbone-checkpoint is required unless --bench-dataloader is set")
    model = DinoV3Density(
        model_name=args.model_name,
        num_classes=args.num_classes,
        image_size=args.image_size,
        pretrained=not args.no_pretrained,
        checkpoint_path=args.backbone_checkpoint,
        freeze_backbone=not args.full_finetune,  # 显式控制
    )

    _configure_finetune(model, args.full_finetune)

    # 新增：确保模型在正确 device（否则容易出现 device/dtype 异常）
    model = model.to(device)

    # 关键：optimizer 只优化可训练参数（冻结 backbone 时不会把它加进去）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters. Check --full-finetune/--freeze-backbone flags.")

    # === 分组学习率：head 用 args.lr，backbone 用 args.lr * args.backbone_lr_mult ===
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad] if hasattr(model, "backbone") else []
    backbone_param_ids = {id(p) for p in backbone_params}

    # 除 backbone 以外的所有可训练参数都归到 head 组（包含 density head 等）
    head_params = [p for p in trainable_params if id(p) not in backbone_param_ids]

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": float(args.lr), "weight_decay": float(args.weight_decay)})
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": float(args.lr) * float(args.backbone_lr_mult),
                "weight_decay": float(args.weight_decay),
            }
        )

    optimizer = torch.optim.AdamW(param_groups)

    print(
        f"[finetune] full_finetune={args.full_finetune} "
        f"(backbone trainable={any(p.requires_grad for p in model.backbone.parameters())}) | "
        f"lr_head={args.lr:g} lr_backbone={args.lr * args.backbone_lr_mult:g}"
    )

    scaler = GradScaler("cuda", enabled=args.amp)

    logger.info(f"Starting training for {args.epochs} epochs...")

    # NEW: track best val count mae
    best_val_count_mae = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_density_loss = 0.0
        epoch_count_loss = 0.0
        epoch_total_loss = 0.0
        samples = 0

        ema_decay = 0.9
        ema_density = None
        ema_count = None
        ema_total = None

        prev_iter_end = time.perf_counter()
        data_time_sum = 0.0
        iter_time_sum = 0.0
        gpu_time_sum_ms = 0.0
        timed_steps = 0

        for step, (imgs, densities) in enumerate(train_loader, start=1):
            iter_start_wall = time.perf_counter()
            data_time = iter_start_wall - prev_iter_end
            imgs = imgs.to(device, non_blocking=True)
            densities = densities.to(device, non_blocking=True)

            # 新增：强制用 fp32 作为“入口”dtype，让 AMP 在算子内部决定 fp16/bf16
            # 可避免 half input + float bias 这类 conv2d dtype mismatch
            imgs = imgs.float()
            densities = densities.float()

            gt_counts = densities.flatten(2).sum(dim=2)  # [B,C]

            if epoch == 1 and step == 1:
                print("densities dtype:", densities.dtype)
                print("densities min/max:", densities.min().item(), densities.max().item())
                print("per-class gt_counts (first sample):", gt_counts[0].detach().cpu().numpy())
                print("total gt_count (first sample):", gt_counts[0].sum().item())

            optimizer.zero_grad(set_to_none=True)

            start_evt = end_evt = None
            if args.profile and device.type == "cuda":
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()

            with torch.amp.autocast("cuda", enabled=args.amp):
                pred_density, pred_counts = model(imgs)
                gt_counts = gt_counts

                # IMPORTANT: avoid diluting density supervision by H*W.
                # Use sum over (C,H,W) then normalize by batch size -> per-image loss.
                density_loss = F.mse_loss(pred_density, densities, reduction="sum") / imgs.size(0)
                # Count-level auxiliary loss: use L1 for more stable scale than MSE.
                count_loss = F.l1_loss(pred_counts, gt_counts)
                loss = density_loss + args.count_loss_weight * count_loss

            scaler.scale(loss).backward()

            # Optional grad clipping (AMP-safe)
            if args.grad_clip_norm and args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip_norm)

            scaler.step(optimizer)
            scaler.update()

            gpu_time_ms = 0.0
            if args.profile and device.type == "cuda" and start_evt is not None and end_evt is not None:
                end_evt.record()
                torch.cuda.synchronize()
                gpu_time_ms = float(start_evt.elapsed_time(end_evt))

            bsz = imgs.size(0)
            samples += bsz
            epoch_density_loss += density_loss.item() * bsz
            epoch_count_loss += count_loss.item() * bsz

            total_loss_item = loss.item()
            epoch_total_loss += total_loss_item * bsz

            # EMA smoothing for more readable logs
            if ema_density is None:
                ema_density = density_loss.item()
                ema_count = count_loss.item()
                ema_total = total_loss_item
            else:
                ema_density = ema_decay * ema_density + (1 - ema_decay) * density_loss.item()
                ema_count = ema_decay * ema_count + (1 - ema_decay) * count_loss.item()
                ema_total = ema_decay * ema_total + (1 - ema_decay) * total_loss_item

            if args.log_interval > 0 and step % args.log_interval == 0:
                if args.profile:
                    iter_end_wall = time.perf_counter()
                    iter_time = iter_end_wall - iter_start_wall
                    if step > args.profile_warmup:
                        data_time_sum += data_time
                        iter_time_sum += iter_time
                        gpu_time_sum_ms += gpu_time_ms
                        timed_steps += 1

                    avg_data = (data_time_sum / timed_steps) if timed_steps else 0.0
                    avg_iter = (iter_time_sum / timed_steps) if timed_steps else 0.0
                    avg_gpu = (gpu_time_sum_ms / timed_steps) if timed_steps else 0.0
                    logger.info(
                        f"  time(avg,skip{args.profile_warmup}) | data {avg_data:.3f}s | iter {avg_iter:.3f}s | gpu {avg_gpu/1000.0:.3f}s"
                    )
                logger.info(
                    f"  iter {step}/{len(train_loader)} | "
                    f"density {density_loss.item():.3e} (ema {ema_density:.3e}) | "
                    f"count {count_loss.item():.3f} (ema {ema_count:.3f}) | "
                    f"total {total_loss_item:.3f} (ema {ema_total:.3f})"
                )

            prev_iter_end = time.perf_counter()
            if args.profile and args.profile_iters and step >= args.profile_iters:
                logger.info(f"[profile] stopping early after {step} iterations (as requested)")
                break

        avg_density = epoch_density_loss / max(samples, 1)
        avg_count = epoch_count_loss / max(samples, 1)
        avg_total = epoch_total_loss / max(samples, 1)

        model.eval()
        val_density = 0.0
        val_count_l1 = 0.0
        val_count_mae = 0.0
        val_total_mae = 0.0
        val_samples = 0
        with torch.no_grad():
            for vstep, (imgs, densities) in enumerate(val_loader, start=1):
                imgs = imgs.to(device, non_blocking=True)
                densities = densities.to(device, non_blocking=True)

                # 新增：验证同样保持入口 fp32
                imgs = imgs.float()
                densities = densities.float()

                pred_density, pred_counts = model(imgs)
                gt_counts = densities.flatten(2).sum(dim=2)

                density_loss = F.mse_loss(pred_density, densities, reduction="sum") / imgs.size(0)
                count_l1 = F.l1_loss(pred_counts, gt_counts)
                count_mae = (pred_counts - gt_counts).abs().mean()

                # 总数（8类求和）MAE：更直观
                pred_total = pred_counts.sum(dim=1)
                gt_total = gt_counts.sum(dim=1)
                total_mae = (pred_total - gt_total).abs().mean()

                bsz = imgs.size(0)
                val_samples += bsz
                val_density += density_loss.item() * bsz
                val_count_l1 += count_l1.item() * bsz
                val_count_mae += count_mae.item() * bsz
                val_total_mae += total_mae.item() * bsz

                if args.profile and args.profile_iters and vstep >= max(1, args.profile_iters // 2):
                    break

        val_density /= max(val_samples, 1)
        val_count_l1 /= max(val_samples, 1)
        val_count_mae /= max(val_samples, 1)
        val_total_mae /= max(val_samples, 1)

        logger.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"train total {avg_total:.4f} | train density {avg_density:.6e} | train count L1 {avg_count:.4f} | "
            f"val density MSE {val_density:.6e} | val count MAE {val_count_mae:.4f} | val total MAE {val_total_mae:.4f}"
        )

        # NEW: save best checkpoint by val count MAE
        if val_count_mae < best_val_count_mae:
            best_val_count_mae = float(val_count_mae)
            best_epoch = int(epoch)
            logger.info(f"[ckpt] new best val count MAE: {best_val_count_mae:.4f} @ epoch {best_epoch} -> {best_ckpt_path}")
            save_checkpoint(
                model,
                optimizer,
                epoch,
                str(best_ckpt_path),
                meta={
                    "metric": "val_count_mae",
                    "val_count_mae": float(val_count_mae),
                    "val_total_mae": float(val_total_mae),
                    "val_density_mse": float(val_density),
                    "best_epoch": int(best_epoch),
                    "full_finetune": bool(args.full_finetune),
                    "freeze_backbone": (not bool(args.full_finetune)),
                },
            )

        # NEW: always save last checkpoint
        save_checkpoint(
            model,
            optimizer,
            epoch,
            str(last_ckpt_path),
            meta={
                "metric": "last",
                "val_count_mae": float(val_count_mae),
                "val_total_mae": float(val_total_mae),
                "val_density_mse": float(val_density),
                "best_val_count_mae": float(best_val_count_mae),
                "best_epoch": int(best_epoch),
                "full_finetune": bool(args.full_finetune),
                "freeze_backbone": (not bool(args.full_finetune)),
            },
        )

if __name__ == "__main__":
    main()

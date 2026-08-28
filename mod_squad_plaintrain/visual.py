from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


_NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

_STEP_RE = re.compile(
    rf"^epoch\s+(?P<epoch>\d+)\/(?P<epochs>\d+)\s+step\s+(?P<step>\d+)\s+\|\s+loss\s+(?P<loss>{_NUM})\s+\|\s+det\s+(?P<det>{_NUM})\s+seg\s+(?P<seg>{_NUM})\s+cnt\s+(?P<cnt>{_NUM})\s*$"
)

_EPOCH_RE = re.compile(
    rf"^epoch\s+(?P<epoch>\d+)\/(?P<epochs>\d+)\s+\|\s+train\s+(?P<train>{_NUM})\s+\|\s+"
    rf"val\s+det\s+(?P<val_det>{_NUM})\s+seg\s+(?P<val_seg>{_NUM})\s+miou\s+(?P<miou>{_NUM})\s+"
    rf"cnt\s+(?P<val_cnt>{_NUM})\s+dens\s+(?P<dens>{_NUM})\s+mae\s+(?P<mae>{_NUM})\s+total_mae\s+(?P<total_mae>{_NUM})\s+\|\s+"
    rf"total\s+(?P<val_total>{_NUM})\s+\|\s+best-by=(?P<best_by>\w+)\s+metric\s+(?P<metric>{_NUM})\s*$"
)


@dataclass(frozen=True)
class StepRow:
    epoch: int
    epochs: int
    step: int
    global_step: int
    loss_avg: float
    det: float
    seg: float
    cnt: float


@dataclass(frozen=True)
class EpochRow:
    epoch: int
    epochs: int
    train: float
    val_det: float
    val_seg: float
    miou: float
    val_cnt: float
    dens: float
    mae: float
    total_mae: float
    val_total: float
    best_by: str
    metric: float


def _safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def parse_log_lines(lines: Iterable[str]) -> Tuple[List[StepRow], List[EpochRow]]:
    steps: List[StepRow] = []
    epochs: List[EpochRow] = []

    current_epoch: Optional[int] = None
    epoch_offset = 0
    last_step_in_epoch = 0

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        m = _STEP_RE.match(line)
        if m:
            epoch = int(m.group("epoch"))
            epochs_total = int(m.group("epochs"))
            step = int(m.group("step"))

            if current_epoch is None:
                current_epoch = epoch
            if epoch != current_epoch:
                epoch_offset += last_step_in_epoch
                current_epoch = epoch
                last_step_in_epoch = 0

            last_step_in_epoch = max(last_step_in_epoch, step)
            global_step = epoch_offset + step

            steps.append(
                StepRow(
                    epoch=epoch,
                    epochs=epochs_total,
                    step=step,
                    global_step=global_step,
                    loss_avg=_safe_float(m.group("loss")),
                    det=_safe_float(m.group("det")),
                    seg=_safe_float(m.group("seg")),
                    cnt=_safe_float(m.group("cnt")),
                )
            )
            continue

        m = _EPOCH_RE.match(line)
        if m:
            epochs.append(
                EpochRow(
                    epoch=int(m.group("epoch")),
                    epochs=int(m.group("epochs")),
                    train=_safe_float(m.group("train")),
                    val_det=_safe_float(m.group("val_det")),
                    val_seg=_safe_float(m.group("val_seg")),
                    miou=_safe_float(m.group("miou")),
                    val_cnt=_safe_float(m.group("val_cnt")),
                    dens=_safe_float(m.group("dens")),
                    mae=_safe_float(m.group("mae")),
                    total_mae=_safe_float(m.group("total_mae")),
                    val_total=_safe_float(m.group("val_total")),
                    best_by=str(m.group("best_by")),
                    metric=_safe_float(m.group("metric")),
                )
            )
            continue

    return steps, epochs


def write_csv_steps(path: Path, rows: List[StepRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "epochs", "step", "global_step", "loss_avg", "det", "seg", "cnt"])
        for r in rows:
            w.writerow([r.epoch, r.epochs, r.step, r.global_step, r.loss_avg, r.det, r.seg, r.cnt])


def write_csv_epochs(path: Path, rows: List[EpochRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "epoch",
                "epochs",
                "train",
                "val_det",
                "val_seg",
                "miou",
                "val_cnt",
                "dens",
                "mae",
                "total_mae",
                "val_total",
                "best_by",
                "metric",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.epoch,
                    r.epochs,
                    r.train,
                    r.val_det,
                    r.val_seg,
                    r.miou,
                    r.val_cnt,
                    r.dens,
                    r.mae,
                    r.total_mae,
                    r.val_total,
                    r.best_by,
                    r.metric,
                ]
            )


def _maybe_log_scale(values: List[float], ratio_threshold: float = 1e3) -> bool:
    finite = [v for v in values if v == v and v > 0]
    if len(finite) < 2:
        return False
    vmin = min(finite)
    vmax = max(finite)
    if vmin <= 0:
        return False
    return (vmax / max(vmin, 1e-12)) >= ratio_threshold


def plot_all(
    *,
    steps: List[StepRow],
    epochs: List[EpochRow],
    outdir: Path,
    fmt: str,
    dpi: int,
    title: Optional[str],
    logy_tasks: bool,
    logy_cnt_only: bool,
) -> List[Path]:
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "matplotlib 未安装或不可用，无法画图。\n"
            "建议在当前环境安装：pip install matplotlib\n"
            f"原始错误：{e}"
        )

    saved: List[Path] = []

    if steps:
        xs = [r.global_step for r in steps]

        # 1) total loss (running avg printed by train.py)
        ys_total = [r.loss_avg for r in steps]
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(xs, ys_total, linewidth=1.0, label="train loss (avg)")
        ax.set_xlabel("global step")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        if title:
            ax.set_title(title)
        ax.legend()
        p = outdir / f"train_total_steps.{fmt}"
        fig.tight_layout()
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        saved.append(p)

        # 2) per-task losses
        ys_det = [r.det for r in steps]
        ys_seg = [r.seg for r in steps]
        ys_cnt = [r.cnt for r in steps]

        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(xs, ys_det, linewidth=1.0, label="det")
        ax.plot(xs, ys_seg, linewidth=1.0, label="seg")
        ax.plot(xs, ys_cnt, linewidth=1.0, label="cnt")
        ax.set_xlabel("global step")
        ax.set_ylabel("loss")
        ax.grid(True, alpha=0.3)
        if title:
            ax.set_title(title + " (tasks)" if title else "tasks")
        if logy_tasks or (logy_cnt_only and _maybe_log_scale(ys_cnt)) or _maybe_log_scale(ys_cnt):
            ax.set_yscale("log")
        ax.legend()
        p = outdir / f"train_tasks_steps.{fmt}"
        fig.tight_layout()
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        saved.append(p)

    if epochs:
        ex = [r.epoch for r in epochs]

        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(ex, [r.train for r in epochs], marker="o", label="train (epoch avg)")
        ax1.plot(ex, [r.val_total for r in epochs], marker="o", label="val total")
        ax1.plot(ex, [r.val_det for r in epochs], marker="o", label="val det")
        ax1.plot(ex, [r.val_seg for r in epochs], marker="o", label="val seg")
        ax1.plot(ex, [r.val_cnt for r in epochs], marker="o", label="val cnt")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax1.grid(True, alpha=0.3)
        if title:
            ax1.set_title(title + " (epoch summary)" if title else "epoch summary")
        if _maybe_log_scale([r.val_cnt for r in epochs] + [r.train for r in epochs]):
            ax1.set_yscale("log")
        ax1.legend(ncols=3)

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(ex, [r.miou for r in epochs], marker="o", label="val mIoU")
        ax2.plot(ex, [r.mae for r in epochs], marker="o", label="val MAE")
        ax2.plot(ex, [r.total_mae for r in epochs], marker="o", label="val total_MAE")
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("metric")
        ax2.grid(True, alpha=0.3)
        ax2.legend(ncols=3)

        p = outdir / f"val_epoch_summary.{fmt}"
        fig.tight_layout()
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        saved.append(p)

        # dens curve (often tiny) saved separately
        dens_vals = [r.dens for r in epochs]
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(ex, dens_vals, marker="o", label="val dens")
        ax.set_xlabel("epoch")
        ax.set_ylabel("dens")
        ax.grid(True, alpha=0.3)
        if _maybe_log_scale([abs(v) for v in dens_vals if v == v and v != 0], ratio_threshold=1e2):
            ax.set_yscale("log")
        ax.legend()
        p = outdir / f"val_density.{fmt}"
        fig.tight_layout()
        fig.savefig(p, dpi=dpi)
        plt.close(fig)
        saved.append(p)

    return saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize multitask training logs (det/seg/cnt)")
    p.add_argument("--log", type=str, required=True, help="path to train.log")
    p.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="output directory (default: <log_dir>/visual)",
    )
    p.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"], help="image format")
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--title", type=str, default=None)
    p.add_argument("--logy-tasks", action="store_true", help="force log-scale for task-loss plot")
    p.add_argument("--logy-cnt-only", action="store_true", help="auto log-scale when cnt dominates")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(str(log_path))

    outdir = Path(args.outdir) if args.outdir else (log_path.parent / "visual")

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        steps, epochs = parse_log_lines(f)

    # dump parsed csv for later use
    write_csv_steps(outdir / "parsed_steps.csv", steps)
    write_csv_epochs(outdir / "parsed_epochs.csv", epochs)

    saved = plot_all(
        steps=steps,
        epochs=epochs,
        outdir=outdir,
        fmt=args.format,
        dpi=int(args.dpi),
        title=args.title,
        logy_tasks=bool(args.logy_tasks),
        logy_cnt_only=bool(args.logy_cnt_only),
    )

    print(f"parsed steps: {len(steps)}, epochs: {len(epochs)}")
    print(f"csv: {outdir / 'parsed_steps.csv'}")
    print(f"csv: {outdir / 'parsed_epochs.csv'}")
    for p in saved:
        print(f"saved: {p}")


if __name__ == "__main__":
    main()

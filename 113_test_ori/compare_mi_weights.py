from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List

import torch


def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _split_paths(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    for item in items:
        if not item:
            continue
        parts = [p.strip() for p in re.split(r"[;,\n]", item) if p.strip()]
        out.extend(parts)
    return out


def _guess_mi_weight(path: str) -> str:
    m = re.search(r"MIloss([0-9.]+)", path, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return ""


def _guess_flags(path: str) -> str:
    flags: List[str] = []
    if re.search(r"noNOISYGATING", path, flags=re.IGNORECASE):
        flags.append("no_noisy_gating")
    return ",".join(flags)


def _summarize_metrics(ckpt: Dict) -> Dict[str, object]:
    metrics = ckpt.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}
    out: Dict[str, object] = {
        "epoch": ckpt.get("epoch"),
        "best_by": ckpt.get("best_by"),
        "best_metric": metrics.get("best_metric"),
        "val_total_loss": metrics.get("val_total_loss"),
        "val_det_loss": metrics.get("val_det_loss"),
        "val_seg_loss": metrics.get("val_seg_loss"),
        "val_seg_miou": metrics.get("val_seg_miou"),
        "val_cnt_loss": metrics.get("val_cnt_loss"),
        "val_cnt_mae": metrics.get("val_cnt_mae"),
        "val_cnt_total_mae": metrics.get("val_cnt_total_mae"),
    }
    return out


def _format_table(rows: List[Dict[str, object]]) -> str:
    headers = [
        "ckpt",
        "mi_weight",
        "flags",
        "epoch",
        "best_by",
        "best_metric",
        "val_total_loss",
        "val_det_loss",
        "val_seg_loss",
        "val_seg_miou",
        "val_cnt_loss",
        "val_cnt_mae",
        "val_cnt_total_mae",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        line = "| " + " | ".join(str(row.get(h, "")) for h in headers) + " |"
        lines.append(line)
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(description="Compare metrics across checkpoints (MI weight experiments).")
    p.add_argument("--checkpoints", type=str, nargs="+", required=True, help="list of checkpoints; supports ';' or ','")
    p.add_argument("--output", type=str, default=None, help="write markdown/json to this file (format by extension)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    ckpts = _split_paths(args.checkpoints)
    if not ckpts:
        raise SystemExit("No checkpoints provided.")

    rows: List[Dict[str, object]] = []
    for path in ckpts:
        row: Dict[str, object] = {"ckpt": path, "mi_weight": _guess_mi_weight(path), "flags": _guess_flags(path)}
        p = Path(path)
        if not p.exists():
            row["error"] = "missing"
            rows.append(row)
            continue
        ckpt = _torch_load_cpu(str(p))
        if not isinstance(ckpt, dict):
            row["error"] = "invalid_ckpt"
            rows.append(row)
            continue
        row.update(_summarize_metrics(ckpt))
        rows.append(row)

    md = _format_table(rows)
    print(md)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".json":
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(rows, f, ensure_ascii=False, indent=2)
        else:
            with out_path.open("w", encoding="utf-8") as f:
                f.write(md + "\n")
        print(f"[compare] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env bash
set -euo pipefail

# Evaluate counting model.
# Note: freeze/full-finetune flags only affect training, not eval.
# Backbone policy:
#   --backbone-source auto => use backbone from --checkpoint if present, else use --backbone-checkpoint.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="/nas/liyangguang103/anaconda3/envs/dam/bin/python"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" eval.py \
	--checkpoint /nas/liyangguang103/new_fscd/counting/runs/20260121_03full/121_10per_best_val_count_mae.pt \
	--backbone-source auto \
	--backbone-checkpoint /nas/liyangguang103/old_fscd/CD-FSOD/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth &

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" eval.py \
	--checkpoint /nas/liyangguang103/new_fscd/counting/runs/20260121_04/121_10perfreeze_best_val_count_mae.pt \
	--backbone-source auto \
	--backbone-checkpoint /nas/liyangguang103/old_fscd/CD-FSOD/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth 





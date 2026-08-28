#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# LoRA-MoE only: no plain LoRA FFN injection.
export LORA="${LORA:-0}"
export LORA_MOE="${LORA_MOE:-1}"

# Expert configuration: 3 private experts per task, 6 shared experts.
export USE_PRIVATE_EXPERTS="${USE_PRIVATE_EXPERTS:-1}"
export USE_SHARED_EXPERTS="${USE_SHARED_EXPERTS:-1}"
export NUM_EXPERTS_PRIVATE="${NUM_EXPERTS_PRIVATE:-3}"
export NUM_EXPERTS_SHARED="${NUM_EXPERTS_SHARED:-6}"
export MOE_K_PRIVATE="${MOE_K_PRIVATE:-2}"
export MOE_K_SHARED="${MOE_K_SHARED:-2}"

# Stage schedule.
export STAGE1_EPOCHS="${STAGE1_EPOCHS:-170}"
export STAGE1_VAL_LAST_K_EPOCHS="${STAGE1_VAL_LAST_K_EPOCHS:-30}"
export STAGE1_ONLY="${STAGE1_ONLY:-0}"
export STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"

# Checkpoints and post-train tests.
export SAVE_DIR="${SAVE_DIR:-runs/latest_ours_100per_loramoe_only_p3_s6_s1e170_s2e20}"
export TEST_STAGE1_CKPT="${TEST_STAGE1_CKPT:-${SAVE_DIR}/stage1_best.pt}"
export TEST_FINAL_CKPT="${TEST_FINAL_CKPT:-${SAVE_DIR}/best_combo.pt}"
export TEST_CKPT="${TEST_CKPT:-${TEST_FINAL_CKPT}}"
export RUN_TEST_AFTER_TRAIN="${RUN_TEST_AFTER_TRAIN:-1}"
export TEST_STAGE1_AFTER_TRAIN="${TEST_STAGE1_AFTER_TRAIN:-1}"
export TEST_EVAL_FULL_MODEL="${TEST_EVAL_FULL_MODEL:-1}"
export SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-$(basename "${SAVE_DIR}")}"

exec bash "${SCRIPT_DIR}/train_100per.sh" "$@"

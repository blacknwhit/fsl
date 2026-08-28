#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CKPT="${1:-${STAGE1_RESUME_CHECKPOINT:-}}"
CONTINUE_EPOCHS="${2:-${STAGE1_CONTINUE_EPOCHS:-50}}"

if [[ -z "${CKPT}" ]]; then
  echo "Usage: bash ${BASH_SOURCE[0]} /path/to/checkpoint.pt <stage1_continue_epochs>" >&2
  echo "Or set STAGE1_RESUME_CHECKPOINT=/path/to/checkpoint.pt STAGE1_CONTINUE_EPOCHS=K" >&2
  exit 2
fi
if [[ -z "${CONTINUE_EPOCHS}" ]]; then
  echo "Missing stage1_continue_epochs. Example: bash ${BASH_SOURCE[0]} ${CKPT} 10" >&2
  exit 2
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 2
fi

CKPT_ABS="$(cd "$(dirname "${CKPT}")" && pwd)/$(basename "${CKPT}")"
CKPT_PARENT_TAG="$(basename "$(dirname "${CKPT_ABS}")")"
CKPT_TAG="$(basename "${CKPT}")"
CKPT_TAG="${CKPT_TAG%.*}"
DEFAULT_SAVE_DIR="runs/latest_ours_100per_stage1_resume_${CKPT_PARENT_TAG}_${CKPT_TAG}_s1e${CONTINUE_EPOCHS}_s2e${STAGE2_EPOCHS:-20}"
RESUME_SAVE_DIR="${SAVE_DIR:-${DEFAULT_SAVE_DIR}}"
if [[ "$(basename "${RESUME_SAVE_DIR}")" != *"${CKPT_PARENT_TAG}"* ]]; then
  RESUME_SAVE_DIR="${RESUME_SAVE_DIR%/}_${CKPT_PARENT_TAG}"
fi

export STAGE1_RESUME_CHECKPOINT="${CKPT_ABS}"
export STAGE1_EPOCHS="${CONTINUE_EPOCHS}"
export STAGE1_VAL_LAST_K_EPOCHS="${CONTINUE_EPOCHS}"
export STAGE1_ONLY="${STAGE1_ONLY:-0}"
export STAGE2_EPOCHS="${STAGE2_EPOCHS:-20}"
export SAVE_DIR="${RESUME_SAVE_DIR}"
export TEST_CKPT="${TEST_CKPT:-${SAVE_DIR}/best_combo.pt}"
export TEST_STAGE1_CKPT="${TEST_STAGE1_CKPT:-${SAVE_DIR}/stage1_best.pt}"
export TEST_FINAL_CKPT="${TEST_FINAL_CKPT:-${SAVE_DIR}/best_combo.pt}"
export RUN_TEST_AFTER_TRAIN="${RUN_TEST_AFTER_TRAIN:-1}"
export TEST_STAGE1_AFTER_TRAIN="${TEST_STAGE1_AFTER_TRAIN:-1}"
export TEST_PRECHECK="${TEST_PRECHECK:-1}"
export TEST_EVAL_FULL_MODEL="${TEST_EVAL_FULL_MODEL:-1}"
export SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-$(basename "${SAVE_DIR}")}"

exec bash "${SCRIPT_DIR}/train_100per.sh"

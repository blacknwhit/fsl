#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/train_10per.sh"
TEST_SCRIPT="${SCRIPT_DIR}/test.sh"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "ERROR: base script not found: ${BASE_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${TEST_SCRIPT}" ]]; then
  echo "ERROR: test script not found: ${TEST_SCRIPT}" >&2
  exit 1
fi

# Group 3: ours (multitask) + full training data + full-parameter finetune.
export LORA_MOE="${LORA_MOE:-1}"
export UNFREEZE_BACKBONE="${UNFREEZE_BACKBONE:-1}"
export SAVE_DIR="${SAVE_DIR:-runs/onlystage2_ours_group3_full_fullft}"

# Resolve SAVE_DIR against the same effective project root used by train_10per.sh.
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT_DEFAULT="/nas/liyangguang103/new_fscd"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"
if [[ -d "${PROJECT_ROOT}" && -f "${PROJECT_ROOT}/113_test_stage2/train.py" ]]; then
  EFFECTIVE_ROOT="${PROJECT_ROOT}"
else
  EFFECTIVE_ROOT="${REPO_ROOT}"
fi
if [[ "${SAVE_DIR}" != /* ]]; then
  export SAVE_DIR="${EFFECTIVE_ROOT}/${SAVE_DIR}"
fi

# Full-data train splits.
export DET_TRAIN_ANN="${DET_TRAIN_ANN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_train.json}"
export SEG_TRAIN_DIR="${SEG_TRAIN_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/train_5500}"
export CNT_TRAIN_DIR="${CNT_TRAIN_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA/train_data_class8}"

# Keep behavior deterministic and avoid accidental sweep mode.
export RUN_CNT_GRAD_HDIM_SWEEP="${RUN_CNT_GRAD_HDIM_SWEEP:-0}"
export RUN_LKRELU_SWEEP="${RUN_LKRELU_SWEEP:-0}"

# Enable multi-GPU linear LR scaling by default; set LINEAR_LR_SCALE=0 to disable.
export LINEAR_LR_SCALE="${LINEAR_LR_SCALE:-1}"

# Auto test after training.
export AUTO_TEST_AFTER_TRAIN="${AUTO_TEST_AFTER_TRAIN:-1}"

bash "${BASE_SCRIPT}" "$@"

if [[ "${AUTO_TEST_AFTER_TRAIN}" == "1" ]]; then
  BEST_CKPT="${SAVE_DIR}/best_combo.pt"
  if [[ ! -f "${BEST_CKPT}" ]]; then
    echo "ERROR: expected checkpoint not found: ${BEST_CKPT}" >&2
    exit 3
  fi

  TS="$(date +%Y%m%d_%H%M%S)"
  export STATS_DIR="${STATS_DIR:-${SAVE_DIR}/stats_eval_best_combo_${TS}}"

  LOG_FILE_EFFECTIVE="${LOG_FILE:-${SAVE_DIR}/train.log}"
  TEST_CMD=(bash "${TEST_SCRIPT}" "${BEST_CKPT}")
  echo "[test] ${TEST_CMD[*]}"
  if [[ "${LOG_TO_FILE:-1}" == "1" ]]; then
    "${TEST_CMD[@]}" 2>&1 | tee -a "${LOG_FILE_EFFECTIVE}"
  else
    "${TEST_CMD[@]}"
  fi
fi

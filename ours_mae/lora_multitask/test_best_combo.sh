#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"

# Standalone checkpoint test entry for this MAE run's best model.
CKPT_DEFAULT="/data/xiangyuyue/ULLM-zf/fsl-20260209/ours_mae/runs/lora_full/best_combo.pt"
CKPT="${1:-${CKPT_DEFAULT}}"

if [[ -d "${PROJECT_ROOT}" && -f "${PROJECT_ROOT}/lora_multitask/eval.py" ]]; then
  cd "${PROJECT_ROOT}"
else
  echo "ERROR: invalid PROJECT_ROOT: ${PROJECT_ROOT}" >&2
  echo "Expecting: ${PROJECT_ROOT}/lora_multitask/eval.py" >&2
  exit 1
fi

if [[ "${CKPT}" != /* ]]; then
  CKPT="${PROJECT_ROOT}/${CKPT}"
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 2
fi

# Keep ours_mae package first, and include workspace root for object_detection/segmentation/counting.
WORKSPACE_ROOT_DEFAULT="$(cd "${PROJECT_ROOT}/.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${WORKSPACE_ROOT_DEFAULT}}"
export PYTHONPATH="${PROJECT_ROOT}:${WORKSPACE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DET_DATA_ROOT="${DET_DATA_ROOT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco}"
SEG_DATA_DIR="${SEG_DATA_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/test}"
CNT_DATA_ROOT="${CNT_DATA_ROOT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA}"
CNT_TEST_DIR="${CNT_TEST_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA/test_data_class8}"

DET_ANN_FILE="${DET_ANN_FILE:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_test.json}"
DET_IMG_DIR="${DET_IMG_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/images/test}"

TEST_TASKS="${TEST_TASKS:-det,seg,cnt}"
DEVICE="${DEVICE:-cuda}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"

# Use local MAE directory by default so eval does not rely on network cache.
MODEL_NAME="${MODEL_NAME:-/data/xiangyuyue/ULLM-zf/fsl-20260209/pretrained/facebook_vit-mae-large}"

DET_USE_COCO_EVAL="${DET_USE_COCO_EVAL:-1}"
DET_SCORE_THR="${DET_SCORE_THR:-0}"

PYTHON_BIN="${PYTHON_BIN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/miniconda3/envs/fsl/bin/python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. Set PYTHON_BIN=/path/to/python" >&2
  exit 1
fi

export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

TS="$(date +%Y%m%d_%H%M%S)"
CKPT_BASENAME="$(basename "${CKPT}")"
CKPT_TAG="${CKPT_BASENAME%.*}"
STATS_DIR="${STATS_DIR:-$(dirname "${CKPT}")/stats_eval_${CKPT_TAG}_${TS}}"
DRY_RUN="${DRY_RUN:-0}"
mkdir -p "${STATS_DIR}"

CMD=(
  "${PYTHON_BIN}" "${PROJECT_ROOT}/lora_multitask/eval.py"
  --checkpoint "${CKPT}"
  --tasks "${TEST_TASKS}"
  --stats-dir "${STATS_DIR}"
  --device "${DEVICE}"
  --image-size "${IMAGE_SIZE}"
  --model-name "${MODEL_NAME}"
  --det-data-root "${DET_DATA_ROOT}"
  --det-ann-file "${DET_ANN_FILE}"
  --det-img-dir "${DET_IMG_DIR}"
  --det-score-thr "${DET_SCORE_THR}"
  --seg-data-dir "${SEG_DATA_DIR}"
  --cnt-data-root "${CNT_DATA_ROOT}"
  --cnt-test-dir "${CNT_TEST_DIR}"
)

if [[ "${DET_USE_COCO_EVAL}" == "1" ]]; then CMD+=(--det-use-coco-eval); fi

{
  echo "[run] project: ${PROJECT_ROOT}"
  echo "[run] ckpt: ${CKPT}"
  echo "[run] stats: ${STATS_DIR}"
  echo -n "[run] cmd: "
  printf '%q ' "${CMD[@]}"
  echo
} | tee "${STATS_DIR}/run_meta.txt"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] command prepared, not executing eval."
  exit 0
fi

("${CMD[@]}" 2>&1 | tee "${STATS_DIR}/console.log")

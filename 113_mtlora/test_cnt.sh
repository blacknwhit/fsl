#!/usr/bin/env bash
set -euo pipefail

CKPT_DEFAULT="/data/xiangyuyue/ULLM-zf/fsl-20260209/runs/113_mtlora_10per/best_combo.pt"
CKPT="${1:-$CKPT_DEFAULT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_ROOT_DEFAULT="/data/xiangyuyue/ULLM-zf/fsl-20260209"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"

if [[ -d "${PROJECT_ROOT}" && -f "${PROJECT_ROOT}/113_mtlora/eval.py" ]]; then
  cd "${PROJECT_ROOT}"
else
  PROJECT_ROOT="${REPO_ROOT}"
  cd "${REPO_ROOT}"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/object_detection${PYTHONPATH:+:${PYTHONPATH}}"

CNT_DATA_ROOT="${CNT_DATA_ROOT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA}"
CNT_TEST_DIR="${CNT_TEST_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA/test_data_class8}"

DEVICE="${DEVICE:-cuda}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
MODEL_NAME="${MODEL_NAME:-dinov3_vitl16}"
CNT_BATCH_SIZE="${CNT_BATCH_SIZE:-16}"
CNT_NUM_WORKERS="${CNT_NUM_WORKERS:-0}"
CNT_KEEP_ASPECT="${CNT_KEEP_ASPECT:-1}"
CHECK_LOAD_ONLY="${CHECK_LOAD_ONLY:-0}"

PYTHON_BIN="${PYTHON_BIN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/miniconda3/envs/fsl/bin/python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. Set PYTHON_BIN=/path/to/python" >&2
  exit 1
fi

# Prefer system driver libcuda to avoid CUDA compat stub conflicts.
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
CKPT_BASENAME="$(basename "$CKPT")"
CKPT_TAG="${CKPT_BASENAME%.*}"
STATS_DIR="${STATS_DIR:-"$(dirname "$CKPT")/stats_eval_cnt_${CKPT_TAG}_${TS}"}"
mkdir -p "$STATS_DIR"

cmd=(
  "$PYTHON_BIN" -m "113_mtlora.eval"
  --checkpoint "$CKPT"
  --tasks "cnt"
  --stats-dir "$STATS_DIR"
  --device "$DEVICE"
  --image-size "$IMAGE_SIZE"
  --model-name "$MODEL_NAME"
  --cnt-data-root "$CNT_DATA_ROOT"
  --cnt-test-dir "$CNT_TEST_DIR"
  --cnt-batch-size "$CNT_BATCH_SIZE"
  --cnt-num-workers "$CNT_NUM_WORKERS"
)

if [[ "$CNT_KEEP_ASPECT" == "1" ]]; then
  cmd+=(--cnt-keep-aspect)
else
  cmd+=(--cnt-no-keep-aspect)
fi

if [[ "$CHECK_LOAD_ONLY" == "1" ]]; then
  cmd+=(--check-load-only)
fi

echo "[run] project: $PROJECT_ROOT"
echo "[run] ckpt: $CKPT"
echo "[run] stats: $STATS_DIR"
echo "[run] cmd: ${cmd[*]}"
echo

("${cmd[@]}" 2>&1 | tee "$STATS_DIR/console.log")

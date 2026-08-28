#!/usr/bin/env bash
set -euo pipefail

CKPT_DEFAULT="/data/xiangyuyue/ULLM-zf/fsl-20260209/runs/mod_squad_plaintrain_sharedonly_mi_10per/best_combo.pt"
CKPT="${1:-$CKPT_DEFAULT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_ROOT_DEFAULT="/data/xiangyuyue/ULLM-zf/fsl-20260209"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"
if [[ -d "${PROJECT_ROOT}" && -f "${PROJECT_ROOT}/mod_squad_plaintrain/eval.py" ]]; then
  cd "${PROJECT_ROOT}"
else
  PROJECT_ROOT="${REPO_ROOT}"
  cd "${REPO_ROOT}"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/object_detection${PYTHONPATH:+:${PYTHONPATH}}"

EVAL_SCRIPT_DEFAULT="${PROJECT_ROOT}/mod_squad_plaintrain/eval.py"
DET_DATA_ROOT="${DET_DATA_ROOT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco}"
SEG_DATA_DIR="${SEG_DATA_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/test}"
CNT_DATA_ROOT="${CNT_DATA_ROOT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA}"
CNT_TEST_DIR="${CNT_TEST_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA/test_data_class8}"
DET_ANN_FILE="${DET_ANN_FILE:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_test.json}"
DET_IMG_DIR="${DET_IMG_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/images/test}"
DEVICE="${DEVICE:-cuda}"
IMAGE_SIZE="${IMAGE_SIZE:-}"
MODEL_NAME="${MODEL_NAME:-}"
DET_USE_COCO_EVAL="${DET_USE_COCO_EVAL:-1}"
DET_SCORE_THR="${DET_SCORE_THR:-0}"
PYTHON_BIN="${PYTHON_BIN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/miniconda3/envs/fsl/bin/python}"

export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. Set PYTHON_BIN=/path/to/python" >&2
  exit 1
fi
if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  exit 2
fi

EVAL_SCRIPT="${EVAL_SCRIPT:-$EVAL_SCRIPT_DEFAULT}"
if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "Eval script not found: $EVAL_SCRIPT" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "$EVAL_SCRIPT")/.." && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
CKPT_BASENAME="$(basename "$CKPT")"
CKPT_TAG="${CKPT_BASENAME%.*}"
STATS_DIR="${STATS_DIR:-"$(dirname "$CKPT")/stats_eval_${CKPT_TAG}_${TS}"}"
mkdir -p "$STATS_DIR"

TASKS="${TASKS:-det,seg,cnt}"
CHECK_LOAD_ONLY="${CHECK_LOAD_ONLY:-0}"
CHECK_FULL_LOAD_ONLY="${CHECK_FULL_LOAD_ONLY:-0}"
EVAL_FULL_MODEL="${EVAL_FULL_MODEL:-1}"

cmd=(
  "$PYTHON_BIN" "$EVAL_SCRIPT"
  --checkpoint "$CKPT"
  --tasks "$TASKS"
  --det-data-root "$DET_DATA_ROOT"
  --det-ann-file "$DET_ANN_FILE"
  --det-img-dir "$DET_IMG_DIR"
  --seg-data-dir "$SEG_DATA_DIR"
  --cnt-data-root "$CNT_DATA_ROOT"
  --stats-dir "$STATS_DIR"
)

if [[ -n "$CNT_TEST_DIR" ]]; then cmd+=(--cnt-test-dir "$CNT_TEST_DIR"); fi
if [[ "$CHECK_LOAD_ONLY" == "1" ]]; then cmd+=(--check-load-only); fi
if [[ "$CHECK_FULL_LOAD_ONLY" == "1" ]]; then cmd+=(--check-full-load-only); fi
if [[ "$EVAL_FULL_MODEL" == "1" ]]; then cmd+=(--eval-full-model); fi
if [[ -n "$DEVICE" ]]; then cmd+=(--device "$DEVICE"); fi
if [[ -n "$IMAGE_SIZE" ]]; then cmd+=(--image-size "$IMAGE_SIZE"); fi
if [[ -n "$MODEL_NAME" ]]; then cmd+=(--model-name "$MODEL_NAME"); fi
if [[ "$DET_USE_COCO_EVAL" == "1" ]]; then cmd+=(--det-use-coco-eval); fi
if [[ -n "$DET_SCORE_THR" ]]; then cmd+=(--det-score-thr "$DET_SCORE_THR"); fi

echo "[run] project: $PROJECT_ROOT"
echo "[run] eval: $EVAL_SCRIPT"
echo "[run] ckpt: $CKPT"
echo "[run] stats: $STATS_DIR"
echo "[run] cmd: ${cmd[*]}"
echo

cd "$PROJECT_ROOT"
("${cmd[@]}" 2>&1 | tee "$STATS_DIR/console.log")

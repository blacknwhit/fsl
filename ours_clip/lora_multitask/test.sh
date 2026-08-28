#!/usr/bin/env bash
set -euo pipefail

CKPT_DEFAULT="/data/xiangyuyue/ULLM-zf/fsl-20260209/runs/lora_multitask_10per/best_combo.pt"
CKPT="${1:-$CKPT_DEFAULT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_ROOT_DEFAULT="/data/xiangyuyue/ULLM-zf/fsl-20260209"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"

if [[ -d "${PROJECT_ROOT}" && -f "${PROJECT_ROOT}/lora_multitask/eval.py" ]]; then
  cd "${PROJECT_ROOT}"
else
  PROJECT_ROOT="${REPO_ROOT}"
  cd "${REPO_ROOT}"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/object_detection${PYTHONPATH:+:${PYTHONPATH}}"

DET_DATA_ROOT="${DET_DATA_ROOT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco}"
SEG_DATA_DIR="${SEG_DATA_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/test}"
CNT_DATA_ROOT="${CNT_DATA_ROOT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA}"
CNT_TEST_DIR="${CNT_TEST_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA/test_data_class8}"

DET_ANN_FILE="${DET_ANN_FILE:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_test.json}"
DET_IMG_DIR="${DET_IMG_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/images/test}"

TEST_TASKS="${TEST_TASKS:-det,seg,cnt}"
DEVICE="${DEVICE:-cuda}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
MODEL_NAME="${MODEL_NAME:-openai/clip-vit-large-patch14}"
BACKBONE_CKPT="${BACKBONE_CKPT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/pretrained/openai_clip-vit-large-patch14}"
DET_BACKBONE_CHECKPOINT="${DET_BACKBONE_CHECKPOINT:-${BACKBONE_CKPT}}"
SEG_BACKBONE_CHECKPOINT="${SEG_BACKBONE_CHECKPOINT:-${BACKBONE_CKPT}}"
CNT_BACKBONE_CHECKPOINT="${CNT_BACKBONE_CHECKPOINT:-${BACKBONE_CKPT}}"

DET_USE_COCO_EVAL="${DET_USE_COCO_EVAL:-1}"
DET_SCORE_THR="${DET_SCORE_THR:-0}"
PYTHON_BIN="${PYTHON_BIN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/miniconda3/envs/fsl/bin/python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. Set PYTHON_BIN=/path/to/python" >&2
  exit 1
fi

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  exit 2
fi

export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

TS="$(date +%Y%m%d_%H%M%S)"
CKPT_BASENAME="$(basename "$CKPT")"
CKPT_TAG="${CKPT_BASENAME%.*}"
STATS_DIR="${STATS_DIR:-"$(dirname "$CKPT")/stats_eval_${CKPT_TAG}_${TS}"}"
mkdir -p "$STATS_DIR"

cmd=(
  "$PYTHON_BIN" -m "lora_multitask.eval"
  --checkpoint "$CKPT"
  --tasks "$TEST_TASKS"
  --stats-dir "$STATS_DIR"
  --device "$DEVICE"
  --image-size "$IMAGE_SIZE"
  --model-name "$MODEL_NAME"
  --det-data-root "$DET_DATA_ROOT"
  --det-ann-file "$DET_ANN_FILE"
  --det-img-dir "$DET_IMG_DIR"
  --det-score-thr "$DET_SCORE_THR"
  --seg-data-dir "$SEG_DATA_DIR"
  --cnt-data-root "$CNT_DATA_ROOT"
  --cnt-test-dir "$CNT_TEST_DIR"
)

if [[ -n "$DET_BACKBONE_CHECKPOINT" ]]; then cmd+=(--det-backbone-checkpoint "$DET_BACKBONE_CHECKPOINT"); fi
if [[ -n "$SEG_BACKBONE_CHECKPOINT" ]]; then cmd+=(--seg-backbone-checkpoint "$SEG_BACKBONE_CHECKPOINT"); fi
if [[ -n "$CNT_BACKBONE_CHECKPOINT" ]]; then cmd+=(--cnt-backbone-checkpoint "$CNT_BACKBONE_CHECKPOINT"); fi
if [[ "$DET_USE_COCO_EVAL" == "1" ]]; then cmd+=(--det-use-coco-eval); fi

echo "[run] project: $PROJECT_ROOT"
echo "[run] ckpt: $CKPT"
echo "[run] stats: $STATS_DIR"
echo "[run] cmd: ${cmd[*]}"
echo

("${cmd[@]}" 2>&1 | tee "$STATS_DIR/console.log")

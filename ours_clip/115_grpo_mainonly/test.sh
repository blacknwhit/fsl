#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CKPT_DEFAULT="${CKPT_DEFAULT:-${REPO_ROOT}/runs/115_mainonly_10per/best_combo.pt}"
CKPT="${1:-$CKPT_DEFAULT}"

# Project root that contains the current `115_grpo_mainonly/` package.
PROJECT_ROOT_DEFAULT="/data/xiangyuyue/ULLM-zf/fsl-20260209"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"

if [[ -d "${PROJECT_ROOT}" && -f "${PROJECT_ROOT}/115_grpo_mainonly/eval.py" ]]; then
  cd "${PROJECT_ROOT}"
else
  PROJECT_ROOT="${REPO_ROOT}"
  cd "${REPO_ROOT}"
fi

# Ensure repo packages are importable (e.g. object_detection, segmentation, counting).
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/object_detection${PYTHONPATH:+:${PYTHONPATH}}"

# Eval script location
# You can override via: EVAL_SCRIPT=/path/to/eval.py bash test.sh <ckpt>
EVAL_SCRIPT_DEFAULT="${PROJECT_ROOT}/115_grpo_mainonly/eval.py"


# Dataset paths (defaults match each single-task eval.py default)
DET_DATA_ROOT="${DET_DATA_ROOT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco}"
SEG_DATA_DIR="${SEG_DATA_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/test}"
CNT_DATA_ROOT="${CNT_DATA_ROOT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA}"
CNT_TEST_DIR="${CNT_TEST_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA/test_data_class8}"

# Detection split (IMPORTANT): default to test (not val)
DET_ANN_FILE="${DET_ANN_FILE:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_test.json}"
DET_IMG_DIR="${DET_IMG_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/images/test}"

# Optional overrides
DEVICE="${DEVICE:-"cuda"}"           # e.g. "cuda:0"
IMAGE_SIZE="${IMAGE_SIZE:-}"   # e.g. "448"
MODEL_NAME="${MODEL_NAME:-openai/clip-vit-large-patch14}"
BACKBONE_CKPT="${BACKBONE_CKPT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/pretrained/openai_clip-vit-large-patch14}"
DET_BACKBONE_CHECKPOINT="${DET_BACKBONE_CHECKPOINT:-${BACKBONE_CKPT}}"
SEG_BACKBONE_CHECKPOINT="${SEG_BACKBONE_CHECKPOINT:-${BACKBONE_CKPT}}"
CNT_BACKBONE_CHECKPOINT="${CNT_BACKBONE_CHECKPOINT:-${BACKBONE_CKPT}}"

# Detection options
DET_USE_COCO_EVAL="${DET_USE_COCO_EVAL:-1}" # 1=try COCOeval if pycocotools available
DET_SCORE_THR="${DET_SCORE_THR:-0}"           # e.g. "0.05"

THIS_DIR="${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/miniconda3/envs/fsl/bin/python}"

# Prefer system driver libcuda to avoid 803 errors from CUDA compat stubs.
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. Set PYTHON_BIN=/path/to/python" >&2
  exit 1
fi

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  exit 2
fi

EVAL_SCRIPT="${EVAL_SCRIPT:-}"
if [[ -z "$EVAL_SCRIPT" ]]; then
  if [[ -f "$EVAL_SCRIPT_DEFAULT" ]]; then
    EVAL_SCRIPT="$EVAL_SCRIPT_DEFAULT"
  else
    EVAL_SCRIPT="$THIS_DIR/eval.py"
  fi
fi

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "Eval script not found: $EVAL_SCRIPT" >&2
  echo "Set EVAL_SCRIPT=/path/to/115_grpo_mainonly/eval.py" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "$EVAL_SCRIPT")/.." && pwd)"

TS="$(date +%Y%m%d_%H%M%S)"

# 浠庢潈閲嶈矾寰勬彁鍙栧悕瀛楋細/a/b/best_total.pt -> best_total
CKPT_BASENAME="$(basename "$CKPT")"
CKPT_TAG="${CKPT_BASENAME%.*}"

STATS_DIR="${STATS_DIR:-"$(dirname "$CKPT")/stats_eval_${CKPT_TAG}_${TS}"}"
mkdir -p "$STATS_DIR"

TASKS="${TASKS:-det,seg,cnt}"
CHECK_LOAD_ONLY="${CHECK_LOAD_ONLY:-0}"
CHECK_FULL_LOAD_ONLY="${CHECK_FULL_LOAD_ONLY:-0}"
# By default, evaluate the training-time isomorphic multitask model (MultiTaskModel),
# i.e. the real model produced by 115_grpo_mainonly/train.py (including LoRA-MoE routing).
# Set EVAL_FULL_MODEL=0 to fall back to legacy single-task subprocess eval.
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

if [[ "$CHECK_LOAD_ONLY" == "1" ]]; then
  cmd+=(--check-load-only)
fi

if [[ "$CHECK_FULL_LOAD_ONLY" == "1" ]]; then
  cmd+=(--check-full-load-only)
fi

if [[ "$EVAL_FULL_MODEL" == "1" ]]; then
  cmd+=(--eval-full-model)
fi

if [[ -n "$DEVICE" ]]; then cmd+=(--device "$DEVICE"); fi
if [[ -n "$IMAGE_SIZE" ]]; then cmd+=(--image-size "$IMAGE_SIZE"); fi
if [[ -n "$MODEL_NAME" ]]; then cmd+=(--model-name "$MODEL_NAME"); fi
if [[ -n "$DET_BACKBONE_CHECKPOINT" ]]; then cmd+=(--det-backbone-checkpoint "$DET_BACKBONE_CHECKPOINT"); fi
if [[ -n "$SEG_BACKBONE_CHECKPOINT" ]]; then cmd+=(--seg-backbone-checkpoint "$SEG_BACKBONE_CHECKPOINT"); fi
if [[ -n "$CNT_BACKBONE_CHECKPOINT" ]]; then cmd+=(--cnt-backbone-checkpoint "$CNT_BACKBONE_CHECKPOINT"); fi

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

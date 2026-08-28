#!/usr/bin/env bash
set -euo pipefail

# Multitask checkpoint (your provided path)
CKPT_DEFAULT="/nas/liyangguang103/new_fscd/runs/mod_squad_10per_1_25_nomiloss/best_combo.pt"
CKPT="${1:-$CKPT_DEFAULT}"

# Eval script location
# Default uses my_mod_squad eval wrapper, which adapts multitask checkpoint format
# (e.g. strips 'backbone.' prefix) for the single-task eval scripts.
# You can override via: EVAL_SCRIPT=/path/to/my_mod_squad/eval.py bash test.sh <ckpt>
EVAL_SCRIPT_NAS_DEFAULT="/nas/liyangguang103/new_fscd/113_test/eval.py"
export PYTHONPATH=/nas/liyangguang103/new_fscd


# Dataset paths (defaults match each single-task eval.py default)
DET_DATA_ROOT="${DET_DATA_ROOT:-/nas/liyangguang103/newdataset/CD-ObjectDetection/DIOR/coco}"
SEG_DATA_DIR="${SEG_DATA_DIR:-/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/test}"
CNT_DATA_ROOT="${CNT_DATA_ROOT:-/nas/liyangguang103/newdataset/CD-Count/DSACA}"

# Detection split (IMPORTANT): default to test (not val)
DET_ANN_FILE="${DET_ANN_FILE:-${DET_DATA_ROOT}/annotations/instances_test.json}"
DET_IMG_DIR="${DET_IMG_DIR:-${DET_DATA_ROOT}/images/test}"

# Optional overrides
DEVICE="${DEVICE:-"cuda:5"}"           # e.g. "cuda:0"
IMAGE_SIZE="${IMAGE_SIZE:-}"   # e.g. "448"
MODEL_NAME="${MODEL_NAME:-}"   # e.g. "dinov3_vitl16"

# Detection options
DET_USE_COCO_EVAL="${DET_USE_COCO_EVAL:-1}" # 1=try COCOeval if pycocotools available
DET_SCORE_THR="${DET_SCORE_THR:- 0}"          # e.g. "0.05"

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/nas/liyangguang103/anaconda3/envs/dam/bin/python}"

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
  if [[ -f "$EVAL_SCRIPT_NAS_DEFAULT" ]]; then
    EVAL_SCRIPT="$EVAL_SCRIPT_NAS_DEFAULT"
  else
    EVAL_SCRIPT="$THIS_DIR/eval.py"
  fi
fi

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "Eval script not found: $EVAL_SCRIPT" >&2
  echo "Set EVAL_SCRIPT=/path/to/my_mod_squad/eval.py" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "$EVAL_SCRIPT")/.." && pwd)"

TS="$(date +%Y%m%d_%H%M%S)"

# 从权重路径提取名字：/a/b/best_total.pt -> best_total
CKPT_BASENAME="$(basename "$CKPT")"
CKPT_TAG="${CKPT_BASENAME%.*}"

STATS_DIR="${STATS_DIR:-"$(dirname "$CKPT")/stats_eval_${CKPT_TAG}_${TS}"}"
mkdir -p "$STATS_DIR"

TASKS="${TASKS:-det,seg,cnt}"
CHECK_LOAD_ONLY="${CHECK_LOAD_ONLY:-0}"
CHECK_FULL_LOAD_ONLY="${CHECK_FULL_LOAD_ONLY:-0}"
# By default, evaluate the training-time isomorphic multitask model (MultiTaskModel),
# i.e. the real model produced by my_mod_squad/train.py (including LoRA-MoE routing).
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

#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${GPU_INDEX:-7}" # e.g. 0 or 2 (maps to CUDA_VISIBLE_DEVICES)
# 强制使用指定 GPU（物理编号）。留空则使用 GPU_INDEX。
CUDA_VISIBLE_DEVICES_OVERRIDE="${CUDA_VISIBLE_DEVICES_OVERRIDE:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# NEW: compute absolute script path BEFORE any cd happens
SCRIPT_PATH="$(readlink -f "${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")")"

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Project root that contains the `multitask/` package.
PROJECT_ROOT_DEFAULT="/nas/liyangguang103/new_fscd"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"

if [[ -d "${PROJECT_ROOT}" && -f "${PROJECT_ROOT}/113_test_ori/train.py" ]]; then
  cd "${PROJECT_ROOT}"
else
  PROJECT_ROOT="${REPO_ROOT}"
  cd "${REPO_ROOT}"
fi

# Ensure local dinov3 package is on PYTHONPATH.
export PYTHONPATH="${PROJECT_ROOT}/object_detection${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/miniconda3/envs/fsl/bin/python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. Set PYTHON_BIN=python3.x" >&2
  exit 1
fi

# =========================
# Required dataset paths
# =========================

# Detection COCO-style root (expects annotations/ and images/..., or a nested coco/annotations layout).
DET_DATA_ROOT="${DET_DATA_ROOT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco}"

# Detection train/val annotations (COCO instances json).
DET_TRAIN_ANN="${DET_TRAIN_ANN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_train_10per.json}"
DET_VAL_ANN="${DET_VAL_ANN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_val.json}"

# Segmentation root (expects images/ and masks/).
SEG_TRAIN_DIR="${SEG_TRAIN_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/train_5500_10per}"
SEG_VAL_DIR="${SEG_VAL_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/val}"

# Counting DSACA root (expects train_data_class8/ and val_data_class8/ unless overridden).
CNT_DATA_ROOT="${CNT_DATA_ROOT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA}"

# Counting train dir (if set, overrides the default split directory under CNT_DATA_ROOT).
CNT_TRAIN_DIR="${CNT_TRAIN_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA/train_data_class8_10per}"

# Optional: backbone checkpoint (recommended).
BACKBONE_CKPT="${BACKBONE_CKPT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"

# =========================
# Training hyper-params
# =========================

MODEL_NAME="${MODEL_NAME:-dinov3_vitl16}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
EPOCHS="${EPOCHS:-100}"

# Validation frequency (matches my_mod_squad.train --val-every)
VAL_EVERY="${VAL_EVERY:-1}"

DET_BATCH="${DET_BATCH:-2}"
SEG_BATCH="${SEG_BATCH:-2}"
CNT_BATCH="${CNT_BATCH:-2}"

LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

# =========================
# LoRA-MoE (private + shared pools)
# =========================
LORA_MOE="${LORA_MOE:-1}" # 1=enable LoRA-MoE adapters, 0=disable (standard multitask finetune)
LORA_RANK="${LORA_RANK:-8}"
NUM_EXPERTS_PRIVATE="${NUM_EXPERTS_PRIVATE:-3}"
NUM_EXPERTS_SHARED="${NUM_EXPERTS_SHARED:-6}"
MOE_K_PRIVATE="${MOE_K_PRIVATE:-2}"
MOE_K_SHARED="${MOE_K_SHARED:-2}"
USE_MI_SHARED="${USE_MI_SHARED:-0}" # 1=--use-mi-shared, 0=--no-mi-shared
MOE_MI_LOSS_SHARED="${MOE_MI_LOSS_SHARED:-0.005}"

# Uncertainty-based automatic task weighting
# Default OFF: do not enable AWL unless explicitly requested.
USE_AUTO_WEIGHTED_LOSS="${USE_AUTO_WEIGHTED_LOSS:-0}" # 1=--use-auto-weighted-loss

# Learnable beta weights for det/seg/cnt
# Default OFF: do not enable unless explicitly requested.
USE_DYNAMIC_LOSS_WEIGHT="${USE_DYNAMIC_LOSS_WEIGHT:-0}" # 1=--dynamic-loss-weight

# Memory optimization (does NOT change losses)
GRAD_CHECKPOINTING="${GRAD_CHECKPOINTING:-1}" # 1=--grad-checkpointing, 0=--no-grad-checkpointing

# Backbone lr defaults to LR * BACKBONE_LR_MULT if BACKBONE_LR is not set.
BACKBONE_LR_MULT="${BACKBONE_LR_MULT:-0.1}"
BACKBONE_LR="${BACKBONE_LR:-}"
BACKBONE_WD="${BACKBONE_WD:-}"

DET_LR="${DET_LR:-}"
SEG_LR="${SEG_LR:-}"
CNT_LR="${CNT_LR:-}"

DET_WD="${DET_WD:-}"
SEG_WD="${SEG_WD:-}"
CNT_WD="${CNT_WD:-}"

LOSS_WEIGHTS="${LOSS_WEIGHTS:-1,1,1}" # legacy fixed det,seg,cnt (Full MAML mainly uses LOSS_WEIGHT_BIAS)
LOSS_WEIGHT_BIAS="${LOSS_WEIGHT_BIAS:-15:8:1}" # constant bias added to learned task weights (det,seg,cnt)
SAVE_DIR="${SAVE_DIR:-runs/mod_squad_10per_1_25_nomiloss}"
LOG_FILE="${LOG_FILE:-${SAVE_DIR}/train.log}"
LOG_TO_FILE="${LOG_TO_FILE:-1}" # 1=tee to LOG_FILE, 0=stdout only
LOG_APPEND="${LOG_APPEND:-1}" # 1=append, 0=overwrite

NUM_WORKERS="${NUM_WORKERS:-5}"

# Align with counting single-task train.sh
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-1}"

# Counting debug diagnostics for 113_test.train
DEBUG_CNT="${DEBUG_CNT:-1}" # 1=enable detailed counting diagnostics
DEBUG_CNT_INTERVAL="${DEBUG_CNT_INTERVAL:-20}" # print every N steps (with DEBUG_CNT=1)
DEBUG_FIRST_N_STEPS="${DEBUG_FIRST_N_STEPS:-3}" # always print first N steps per epoch (with DEBUG_CNT=1)

# NEW: snapshot helpers (save train.sh + cmd + env into SAVE_DIR)
copy_unique() {
  local src="$1"
  local dst="$2"
  if [[ ! -e "$dst" ]]; then
    cp -a "$src" "$dst"
    return 0
  fi
  local base ext dir name
  dir="$(dirname "$dst")"
  name="$(basename "$dst")"
  base="${name%.*}"
  ext="${name##*.}"
  if [[ "$ext" == "$name" ]]; then ext=""; else ext=".$ext"; fi
  for i in $(seq -w 1 999); do
    local cand="${dir}/${base}_${i}${ext}"
    if [[ ! -e "$cand" ]]; then
      cp -a "$src" "$cand"
      return 0
    fi
  done
  echo "ERROR: failed to find unique name for $dst" >&2
  exit 1
}

mkdir -p "${SAVE_DIR}"
# OLD (remove): SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
copy_unique "${SCRIPT_PATH}" "${SAVE_DIR}/train.sh.bak"
{
  echo "[time] $(date -Is)"
  echo "[pwd]  $(pwd)"
  echo "[script] ${SCRIPT_PATH}"
} > "${SAVE_DIR}/run_meta.txt"



if [[ -n "${CUDA_VISIBLE_DEVICES_OVERRIDE}" ]]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_OVERRIDE}"

  echo "================================================"
  echo "[DEBUG] CUDA_DEVICE_ORDER: $CUDA_DEVICE_ORDER"
  echo "[DEBUG] CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
  echo "================================================"
elif [[ -n "${GPU_INDEX}" ]]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
fi

# Prefer the system driver libcuda to avoid 803 errors from compat stubs.
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# When CUDA_VISIBLE_DEVICES is set, use `cuda` (index becomes 0 within the visible set).
DEVICE="${DEVICE:-cuda}"
AMP="${AMP:-1}" # 1=on, 0=off

ARGS=(
  -m 113_test_ori.train
  --model-name "${MODEL_NAME}"
  --image-size "${IMAGE_SIZE}"
  --epochs "${EPOCHS}"
  --val-every "${VAL_EVERY}"
  --device "${DEVICE}"
  --num-workers "${NUM_WORKERS}"
  --save-dir "${SAVE_DIR}"
  --loss-weights "${LOSS_WEIGHTS}"
  --det-data-root "${DET_DATA_ROOT}"
  --det-train-ann "${DET_TRAIN_ANN}"
  --det-val-ann "${DET_VAL_ANN}"
  --det-batch-size "${DET_BATCH}"
  --seg-train-dir "${SEG_TRAIN_DIR}"
  --seg-val-dir "${SEG_VAL_DIR}"
  --seg-batch-size "${SEG_BATCH}"
  --cnt-data-root "${CNT_DATA_ROOT}"
  --cnt-train-dir "${CNT_TRAIN_DIR}"
  --cnt-batch-size "${CNT_BATCH}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --backbone-lr-mult "${BACKBONE_LR_MULT}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --cnt-backbone-grad-mult 1
)



if [[ "${LORA_MOE}" == "1" ]]; then
  ARGS+=( --use-lora-moe )
  ARGS+=(
    --lora-rank "${LORA_RANK}"
    --num-experts-private "${NUM_EXPERTS_PRIVATE}"
    --num-experts-shared "${NUM_EXPERTS_SHARED}"
    --moe-k-private "${MOE_K_PRIVATE}"
    --moe-k-shared "${MOE_K_SHARED}"
    --moe-mi-loss-shared "${MOE_MI_LOSS_SHARED}"
  )
  if [[ "${USE_MI_SHARED}" == "1" ]]; then
    ARGS+=( --use-mi-shared )
  else
    ARGS+=( --no-mi-shared )
  fi
  if [[ "${USE_AUTO_WEIGHTED_LOSS}" == "1" ]]; then
    ARGS+=( --use-auto-weighted-loss )
  fi
  if [[ "${USE_DYNAMIC_LOSS_WEIGHT}" == "1" ]]; then
    ARGS+=( --dynamic-loss-weight )
  fi
fi

if [[ "${GRAD_CHECKPOINTING}" == "1" ]]; then
  ARGS+=( --grad-checkpointing )
else
  ARGS+=( --no-grad-checkpointing )
fi

if [[ -n "${BACKBONE_CKPT}" && "${BACKBONE_CKPT}" != "/path/to/dinov3_checkpoint.pth" ]]; then
  ARGS+=( --backbone-checkpoint "${BACKBONE_CKPT}" )
fi

if [[ -n "${BACKBONE_LR}" ]]; then ARGS+=( --backbone-lr "${BACKBONE_LR}" ); fi
if [[ -n "${BACKBONE_WD}" ]]; then ARGS+=( --backbone-weight-decay "${BACKBONE_WD}" ); fi

if [[ -n "${DET_LR}" ]]; then ARGS+=( --det-lr "${DET_LR}" ); fi
if [[ -n "${SEG_LR}" ]]; then ARGS+=( --seg-lr "${SEG_LR}" ); fi
if [[ -n "${CNT_LR}" ]]; then ARGS+=( --cnt-lr "${CNT_LR}" ); fi

if [[ -n "${DET_WD}" ]]; then ARGS+=( --det-weight-decay "${DET_WD}" ); fi
if [[ -n "${SEG_WD}" ]]; then ARGS+=( --seg-weight-decay "${SEG_WD}" ); fi
if [[ -n "${CNT_WD}" ]]; then ARGS+=( --cnt-weight-decay "${CNT_WD}" ); fi

if [[ "${AMP}" == "1" ]]; then ARGS+=( --amp ); fi

# NEW: dump final command + env for reproducibility
{
  echo "[python] ${PYTHON_BIN}"
  echo -n "[cmd_escaped] "
  printf '%q ' "${PYTHON_BIN}" "${ARGS[@]}"
  echo
  echo -n "[cmd_plain] "
  echo "${PYTHON_BIN} ${ARGS[*]}"
} > "${SAVE_DIR}/cmd.txt"
env | sort > "${SAVE_DIR}/env.txt"

echo "[run] ${PYTHON_BIN} ${ARGS[*]}"
if [[ "${LOG_TO_FILE}" == "1" ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  echo "[log] ${LOG_FILE}"
  if [[ "${LOG_APPEND}" == "1" ]]; then
    "${PYTHON_BIN}" "${ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
  else
    "${PYTHON_BIN}" "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
  fi
else
  exec "${PYTHON_BIN}" "${ARGS[@]}"
fi

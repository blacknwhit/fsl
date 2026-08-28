#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${GPU_INDEX:-7}" # e.g. 0 or 2 (maps to CUDA_VISIBLE_DEVICES)
# 强制使用指定 GPU（物理编号）。留空则使用 GPU_INDEX。
CUDA_VISIBLE_DEVICES_OVERRIDE="${CUDA_VISIBLE_DEVICES_OVERRIDE:-0,1,2,3,4,5,6,7}"
#CUDA_VISIBLE_DEVICES_OVERRIDE="${CUDA_VISIBLE_DEVICES_OVERRIDE:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# NEW: compute absolute script path BEFORE any cd happens
SCRIPT_PATH="$(readlink -f "${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")")"

REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Project root that contains the `lora_multitask/` package.
PROJECT_ROOT_DEFAULT="/nas/liyangguang103/new_fscd"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"

if [[ -d "${PROJECT_ROOT}" && -f "${PROJECT_ROOT}/lora_multitask/train.py" ]]; then
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

# Detection train annotations (COCO instances json).
DET_TRAIN_ANN="${DET_TRAIN_ANN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_train_5per.json}"

# Segmentation root (expects images/ and masks/).
SEG_TRAIN_DIR="${SEG_TRAIN_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/train_5500_5per}"
SEG_VAL_DIR="${SEG_VAL_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/val}"

# Counting DSACA root (expects train_data_class8/ and val_data_class8/ unless overridden).
CNT_DATA_ROOT="${CNT_DATA_ROOT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA}"

# Counting train dir (if set, overrides the default split directory under CNT_DATA_ROOT).
CNT_TRAIN_DIR="${CNT_TRAIN_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA/train_data_class8_5per}"

# Optional: backbone checkpoint (recommended).
BACKBONE_CKPT="${BACKBONE_CKPT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"

# =========================
# Training hyper-params
# =========================

MODEL_NAME="${MODEL_NAME:-dinov3_vitl16}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
EPOCHS="${EPOCHS:-100}"
VAL_WINDOW_EPOCHS="${VAL_WINDOW_EPOCHS:-50}"
VAL_START_EPOCH="${VAL_START_EPOCH:-}"

DET_BATCH="${DET_BATCH:-2}"
SEG_BATCH="${SEG_BATCH:-2}"
CNT_BATCH="${CNT_BATCH:-2}"

LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

# =========================
# LoRA (FFN) finetune
# =========================
LORA="${LORA:-1}" # 1=enable (freeze backbone weights; train LoRA+heads), 0=disable (legacy finetune)
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_LR="${LORA_LR:-}" # default: LR
LORA_WD="${LORA_WD:-0.0}"

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

LOSS_WEIGHTS="${LOSS_WEIGHTS:-15,8,1}" # det,seg,cnt
SAVE_DIR="${SAVE_DIR:-runs/lora_5per}"
LOG_FILE="${LOG_FILE:-${SAVE_DIR}/train.log}"
LOG_TO_FILE="${LOG_TO_FILE:-1}" # 1=tee to LOG_FILE, 0=stdout only
LOG_APPEND="${LOG_APPEND:-1}" # 1=append, 0=overwrite
AUTO_TEST_AFTER_TRAIN="${AUTO_TEST_AFTER_TRAIN:-1}"

NUM_WORKERS="${NUM_WORKERS:-5}"

# Align with counting single-task train.sh
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-0.1}"

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

# When CUDA_VISIBLE_DEVICES is set, use `cuda` (index becomes 0 within the visible set).
DEVICE="${DEVICE:-cuda}"
AMP="${AMP:-1}" # 1=on, 0=off

# Prefer the system driver libcuda to avoid 803 errors from compat stubs.
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# Distributed launch
# NPROC_PER_NODE:
# - auto (default): use number of visible GPUs
# - integer: use requested value, but clamp to visible GPU count when too large
NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"
LINEAR_LR_SCALE="${LINEAR_LR_SCALE:-0}" # 1=scale lr by NPROC_PER_NODE when >1

TRAIN_MODULE="${TRAIN_MODULE:-lora_multitask.train}"

count_visible_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local s="${CUDA_VISIBLE_DEVICES// /}"
    if [[ -z "$s" ]]; then
      echo 0
      return
    fi
    IFS=',' read -r -a arr <<< "$s"
    local n=0
    local x
    for x in "${arr[@]}"; do
      [[ -n "$x" ]] && ((n+=1))
    done
    echo "$n"
    return
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    local n
    n="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [[ "$n" =~ ^[0-9]+$ ]]; then
      echo "$n"
      return
    fi
  fi

  echo 0
}

VISIBLE_GPU_COUNT="$(count_visible_gpus)"
if ! [[ "${VISIBLE_GPU_COUNT}" =~ ^[0-9]+$ ]]; then
  VISIBLE_GPU_COUNT=0
fi

if [[ "${NPROC_PER_NODE}" == "auto" ]]; then
  if [[ "${VISIBLE_GPU_COUNT}" -gt 0 ]]; then
    NPROC_PER_NODE="${VISIBLE_GPU_COUNT}"
  else
    NPROC_PER_NODE=1
  fi
fi

if ! [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: NPROC_PER_NODE must be an integer or 'auto', got: ${NPROC_PER_NODE}" >&2
  exit 1
fi

if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  NPROC_PER_NODE=1
fi

if [[ "${VISIBLE_GPU_COUNT}" -gt 0 && "${NPROC_PER_NODE}" -gt "${VISIBLE_GPU_COUNT}" ]]; then
  echo "[ddp][warn] NPROC_PER_NODE=${NPROC_PER_NODE} > visible_gpus=${VISIBLE_GPU_COUNT}; clamping to ${VISIBLE_GPU_COUNT}" >&2
  NPROC_PER_NODE="${VISIBLE_GPU_COUNT}"
fi

echo "[ddp] visible_gpus=${VISIBLE_GPU_COUNT}, nproc_per_node=${NPROC_PER_NODE}"

scale_float() {
  local val="$1"
  local scale="$2"
  "${PYTHON_BIN}" - "$val" "$scale" <<'PY'
import sys
v = float(sys.argv[1])
s = float(sys.argv[2])
print(f"{v*s:.12g}")
PY
}

if [[ -z "${VAL_START_EPOCH}" ]]; then
  if [[ "${VAL_WINDOW_EPOCHS}" =~ ^[0-9]+$ ]] && [[ "${VAL_WINDOW_EPOCHS}" -gt 0 ]] && [[ "${EPOCHS}" =~ ^[0-9]+$ ]] && [[ "${EPOCHS}" -ge "${VAL_WINDOW_EPOCHS}" ]]; then
    VAL_START_EPOCH="$((EPOCHS - VAL_WINDOW_EPOCHS + 1))"
  else
    VAL_START_EPOCH=1
  fi
fi

if [[ "${LINEAR_LR_SCALE}" == "1" && "${NPROC_PER_NODE}" -gt 1 ]]; then
  LR="$(scale_float "${LR}" "${NPROC_PER_NODE}")"
  if [[ -n "${LORA_LR}" ]]; then LORA_LR="$(scale_float "${LORA_LR}" "${NPROC_PER_NODE}")"; fi
  if [[ -n "${BACKBONE_LR}" ]]; then BACKBONE_LR="$(scale_float "${BACKBONE_LR}" "${NPROC_PER_NODE}")"; fi
  if [[ -n "${DET_LR}" ]]; then DET_LR="$(scale_float "${DET_LR}" "${NPROC_PER_NODE}")"; fi
  if [[ -n "${SEG_LR}" ]]; then SEG_LR="$(scale_float "${SEG_LR}" "${NPROC_PER_NODE}")"; fi
  if [[ -n "${CNT_LR}" ]]; then CNT_LR="$(scale_float "${CNT_LR}" "${NPROC_PER_NODE}")"; fi
  echo "[ddp] linear lr scaling enabled: x${NPROC_PER_NODE}"
  echo "[ddp] scaled LR=${LR}"
fi

ARGS=(
  --model-name "${MODEL_NAME}"
  --image-size "${IMAGE_SIZE}"
  --epochs "${EPOCHS}"
  --val-start-epoch "${VAL_START_EPOCH}"
  --device "${DEVICE}"
  --num-workers "${NUM_WORKERS}"
  --save-dir "${SAVE_DIR}"
  --loss-weights "${LOSS_WEIGHTS}"
  --det-data-root "${DET_DATA_ROOT}"
  --det-train-ann "${DET_TRAIN_ANN}"
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

if [[ "${LORA}" == "1" ]]; then
  ARGS+=( --lora )
  ARGS+=( --lora-rank "${LORA_RANK}" --lora-alpha "${LORA_ALPHA}" --lora-dropout "${LORA_DROPOUT}" )
  ARGS+=( --lora-weight-decay "${LORA_WD}" )
  if [[ -n "${LORA_LR}" ]]; then ARGS+=( --lora-lr "${LORA_LR}" ); fi
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

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  RUN_CMD=(
    "${PYTHON_BIN}" -m torch.distributed.run --standalone
    --nproc_per_node "${NPROC_PER_NODE}"
    --module "${TRAIN_MODULE}"
    "${ARGS[@]}"
  )
else
  RUN_CMD=("${PYTHON_BIN}" -m "${TRAIN_MODULE}" "${ARGS[@]}")
fi

# NEW: dump final command + env for reproducibility
{
  echo "[python] ${PYTHON_BIN}"
  echo -n "[cmd_escaped] "
  printf '%q ' "${RUN_CMD[@]}"
  echo
  echo -n "[cmd_plain] "
  echo "${RUN_CMD[*]}"
} > "${SAVE_DIR}/cmd.txt"
env | sort > "${SAVE_DIR}/env.txt"

echo "[run] ${RUN_CMD[*]}"
if [[ "${LOG_TO_FILE}" == "1" ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  echo "[log] ${LOG_FILE}"
  if [[ "${LOG_APPEND}" == "1" ]]; then
    "${RUN_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
  else
    "${RUN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
  fi
else
  "${RUN_CMD[@]}"
fi

BEST_CKPT="${SAVE_DIR}/best_combo.pt"
if [[ "${AUTO_TEST_AFTER_TRAIN}" == "1" ]]; then
  if [[ ! -f "${BEST_CKPT}" ]]; then
    echo "[warn] training finished but checkpoint not found: ${BEST_CKPT}" >&2
    exit 3
  fi
  TEST_CMD=(bash "${SCRIPT_DIR}/test.sh" "${BEST_CKPT}")
  echo "[test] ${TEST_CMD[*]}"
  if [[ "${LOG_TO_FILE}" == "1" ]]; then
    "${TEST_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
  else
    "${TEST_CMD[@]}"
  fi
fi

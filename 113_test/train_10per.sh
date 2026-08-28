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

# Project root that contains the `multitask/` package.
PROJECT_ROOT_DEFAULT="/nas/liyangguang103/new_fscd"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"

if [[ -d "${PROJECT_ROOT}" && -f "${PROJECT_ROOT}/113_test/train.py" ]]; then
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

# Detection train/val/test annotations (COCO instances json).
DET_TRAIN_ANN="${DET_TRAIN_ANN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_train_10per.json}"
DET_VAL_ANN="${DET_VAL_ANN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_val.json}"
DET_TEST_ANN="${DET_TEST_ANN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_test.json}"

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

UNFREEZE_BACKBONE="${UNFREEZE_BACKBONE:-0}" # 1=unfreeze, 0=freeze
MODEL_NAME="${MODEL_NAME:-dinov3_vitl16}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-100}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-50}"
META_SPLIT="${META_SPLIT:-0.2}"
META_SEED="${META_SEED:-42}"
SEED="${SEED:-42}"
META_ALPHA="${META_ALPHA:-5e-4}"
META_BETA="${META_BETA:-1e-3}"

# Validation frequency (matches my_mod_squad.train --val-every)
VAL_EVERY="${VAL_EVERY:-1}"
SKIP_VALIDATION="${SKIP_VALIDATION:-0}" # 1=--skip-validation

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

# Learnable beta weights for det/seg/cnt
# Default OFF: do not enable unless explicitly requested.
USE_DYNAMIC_LOSS_WEIGHT="${USE_DYNAMIC_LOSS_WEIGHT:-0}" # 1=--dynamic-loss-weight

# Learnable loss weights
# Default OFF: do not enable unless explicitly requested.
# Default ON: use gradient-MLP loss weights (required by Full MAML).
LEARN_LOSS_WEIGHTS_MLP="${LEARN_LOSS_WEIGHTS_MLP:-1}" # 1=--learn-loss-weights-mlp
# Weight net architecture:
# - per_task_shared (option1)
# - joint (option2, default)
WEIGHT_NET_ARCH="${WEIGHT_NET_ARCH:-joint}"
# Joint head output activation (used only when WEIGHT_NET_ARCH=joint):
# - leakyrelu (legacy behavior)
# - sigmoid
JOINT_WEIGHT_OUT_ACT="${JOINT_WEIGHT_OUT_ACT:-leakyrelu}"
# LeakyReLU slope for joint output activation when JOINT_WEIGHT_OUT_ACT=leakyrelu.
JOINT_LEAKYRELU_SLOPE="${JOINT_LEAKYRELU_SLOPE:-0.01}"
# Dropout probability inside weight generator network(s), 0 disables.
WEIGHT_NET_DROPOUT="${WEIGHT_NET_DROPOUT:-0}"
# Gradient vector preprocessing before weight-net:
# - l2  : per-task L2 normalization (old behavior)
# - none: keep magnitude, only nan/inf sanitize (default)
GRAD_VEC_NORMALIZE="${GRAD_VEC_NORMALIZE:-l2}"
# Multi-layer gradient encoder for weight-net:
# 0 = legacy (last-layer gradients only)
# 1 = use multi-layer module projector encoder
WEIGHT_NET_USE_MULTILAYER_GRADS="${WEIGHT_NET_USE_MULTILAYER_GRADS:-0}"
WEIGHT_NET_LAYER_EMBED_DIM="${WEIGHT_NET_LAYER_EMBED_DIM:-16}"
WEIGHT_NET_TASK_EMBED_DIM="${WEIGHT_NET_TASK_EMBED_DIM:-64}"
CNT_GRAD_HIDDEN_DIM="${CNT_GRAD_HIDDEN_DIM:-64}"

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
# Prior fusion mode for learned task weights:
# - 1: multiply prior (w_final = w_net * prior)
# - 0: add prior      (w_final = w_net + prior)
LOSS_WEIGHT_PRIOR_MUL="${LOSS_WEIGHT_PRIOR_MUL:-0}"

# Detection fast-AP50 diagnostic threshold.
# 0.0 keeps all predictions (often ends up as N_images*100 due to torchvision detections_per_img cap).
DET_AP_SCORE_THR="${DET_AP_SCORE_THR:-0.0}"
SAVE_DIR="${SAVE_DIR:-runs/32}"
LOG_FILE="${LOG_FILE:-${SAVE_DIR}/train.log}"
LOG_TO_FILE="${LOG_TO_FILE:-1}" # 1=tee to LOG_FILE, 0=stdout only
LOG_APPEND="${LOG_APPEND:-1}" # 1=append, 0=overwrite
RUN_CNT_GRAD_HDIM_SWEEP="${RUN_CNT_GRAD_HDIM_SWEEP:-1}" # 1=run 16/48/96 sweep sequentially; 0=single run
CNT_GRAD_HDIM_SWEEP_VALUES="${CNT_GRAD_HDIM_SWEEP_VALUES:-16 48 96}"
RUN_LKRELU_SWEEP="${RUN_LKRELU_SWEEP:-0}" # 1=run 0.5/0.1/0.05 sweep sequentially; 0=single run with JOINT_LEAKYRELU_SLOPE
SWEEP_SAVE_ROOT="${SWEEP_SAVE_ROOT:-runs}" # parent dir for sweep runs

NUM_WORKERS="${NUM_WORKERS:-16}"

# Align with counting single-task train.sh
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-100}"
# Separate clipping for loss-weight generator (phi) in Stage1 outer update.
PHI_GRAD_CLIP_NORM="${PHI_GRAD_CLIP_NORM:-0.1}"

# Counting debug diagnostics
DEBUG_CNT="0" # force off in this launcher
DEBUG_CNT_INTERVAL="0"
DEBUG_FIRST_N_STEPS="0"

abs_path() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import os
import sys
print(os.path.abspath(os.path.normpath(sys.argv[1])))
PY
}

if [[ "${RUN_CNT_GRAD_HDIM_SWEEP}" == "1" && "${RUN_LKRELU_SWEEP}" == "1" ]]; then
  echo "ERROR: RUN_CNT_GRAD_HDIM_SWEEP=1 and RUN_LKRELU_SWEEP=1 cannot be enabled together." >&2
  exit 2
fi

if [[ "${RUN_CNT_GRAD_HDIM_SWEEP}" == "1" ]]; then
  TEST_SCRIPT="${SCRIPT_DIR}/test.sh"

  if [[ ! -f "${TEST_SCRIPT}" ]]; then
    echo "ERROR: test script not found: ${TEST_SCRIPT}" >&2
    exit 2
  fi

  SAVE_DIR_BASE_ABS="$(abs_path "${SAVE_DIR}")"
  SAVE_DIR_BASE_DIR="$(dirname "${SAVE_DIR_BASE_ABS}")"
  SAVE_DIR_BASE_NAME="$(basename "${SAVE_DIR_BASE_ABS}")"

  for HDIM in ${CNT_GRAD_HDIM_SWEEP_VALUES}; do
    if ! [[ "${HDIM}" =~ ^[0-9]+$ ]] || [[ "${HDIM}" -le 0 ]]; then
      echo "ERROR: invalid CNT_GRAD_HDIM_SWEEP value: ${HDIM}" >&2
      exit 2
    fi

    EXP_NAME="${SAVE_DIR_BASE_NAME}_${HDIM}"
    EXP_SAVE_DIR="${SAVE_DIR_BASE_DIR}/${EXP_NAME}"
    EXP_SAVE_DIR_ABS="$(abs_path "${EXP_SAVE_DIR}")"
    EXP_LOG_FILE="${EXP_SAVE_DIR_ABS}/train.log"
    CKPT_PATH_ABS="${EXP_SAVE_DIR_ABS}/best_combo.pt"

    echo "================================================"
    echo "[cnt-hdim-sweep] hidden_dim=${HDIM}"
    echo "[cnt-hdim-sweep] save_dir=${EXP_SAVE_DIR_ABS}"
    echo "================================================"

    RUN_CNT_GRAD_HDIM_SWEEP=0 \
    CNT_GRAD_HIDDEN_DIM="${HDIM}" \
    SAVE_DIR="${EXP_SAVE_DIR_ABS}" \
    LOG_FILE="${EXP_LOG_FILE}" \
    LOG_APPEND=0 \
    bash "${SCRIPT_PATH}"

    if [[ ! -f "${CKPT_PATH_ABS}" ]]; then
      echo "ERROR: expected checkpoint not found: ${CKPT_PATH_ABS}" >&2
      exit 3
    fi

    echo "[cnt-hdim-sweep] testing checkpoint: ${CKPT_PATH_ABS}"
    STATS_DIR="${EXP_SAVE_DIR_ABS}/stats_eval_${EXP_NAME}" \
    bash "${TEST_SCRIPT}" "${CKPT_PATH_ABS}"
  done

  echo "[cnt-hdim-sweep] all experiments completed."
  exit 0
fi

if [[ "${RUN_LKRELU_SWEEP}" == "1" ]]; then
  EXP_NAMES=("31_lkrelu_0.5" "31_lkrelu_0.1" "31_lkrelu_0.05")
  EXP_SLOPES=("0.5" "0.1" "0.05")
  TEST_SCRIPT="${SCRIPT_DIR}/test.sh"

  if [[ ! -f "${TEST_SCRIPT}" ]]; then
    echo "ERROR: test script not found: ${TEST_SCRIPT}" >&2
    exit 2
  fi

  for i in "${!EXP_NAMES[@]}"; do
    EXP_NAME="${EXP_NAMES[$i]}"
    EXP_SLOPE="${EXP_SLOPES[$i]}"
    EXP_SAVE_DIR="${SWEEP_SAVE_ROOT}/${EXP_NAME}"
    EXP_LOG_FILE="${EXP_SAVE_DIR}/train.log"

    echo "================================================"
    echo "[sweep] run ${i}: name=${EXP_NAME} leaky_slope=${EXP_SLOPE}"
    echo "[sweep] save_dir=${EXP_SAVE_DIR}"
    echo "================================================"

    RUN_LKRELU_SWEEP=0 \
    SAVE_DIR="${EXP_SAVE_DIR}" \
    LOG_FILE="${EXP_LOG_FILE}" \
    LOG_APPEND=0 \
    JOINT_WEIGHT_OUT_ACT="leakyrelu" \
    JOINT_LEAKYRELU_SLOPE="${EXP_SLOPE}" \
    bash "${SCRIPT_PATH}"

    CKPT_PATH="${EXP_SAVE_DIR}/best_combo.pt"
    if [[ ! -f "${CKPT_PATH}" ]]; then
      echo "ERROR: expected checkpoint not found: ${CKPT_PATH}" >&2
      exit 3
    fi

    echo "[sweep] testing checkpoint: ${CKPT_PATH}"
    STATS_DIR="${EXP_SAVE_DIR}/stats_eval_${EXP_NAME}" \
    bash "${TEST_SCRIPT}" "${CKPT_PATH}"
  done

  echo "[sweep] all experiments completed."
  exit 0
fi

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
AMP="${AMP:-0}" # 1=on, 0=off

# Distributed launch
# NPROC_PER_NODE:
# - auto (default): use number of visible GPUs
# - integer: use requested value, but clamp to visible GPU count when too large
NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"
LINEAR_LR_SCALE="${LINEAR_LR_SCALE:-1}" # 1=scale lr by NPROC_PER_NODE when >1

TRAIN_MODULE="${TRAIN_MODULE:-113_test.train}"

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

if [[ "${LINEAR_LR_SCALE}" == "1" && "${NPROC_PER_NODE}" -gt 1 ]]; then
  LR="$(scale_float "${LR}" "${NPROC_PER_NODE}")"
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
  --stage1-epochs "${STAGE1_EPOCHS}"
  --stage2-epochs "${STAGE2_EPOCHS}"
  --meta-split "${META_SPLIT}"
  --meta-seed "${META_SEED}"
  --seed "${SEED}"
  --meta-alpha "${META_ALPHA}"
  --meta-beta "${META_BETA}"
  --val-every "${VAL_EVERY}"
  --device "${DEVICE}"
  --num-workers "${NUM_WORKERS}"
  --save-dir "${SAVE_DIR}"
  --loss-weights "${LOSS_WEIGHTS}"
  --loss-weight-bias "${LOSS_WEIGHT_BIAS}"
  --weight-net-arch "${WEIGHT_NET_ARCH}"
  --joint-weight-out-act "${JOINT_WEIGHT_OUT_ACT}"
  --joint-leakyrelu-slope "${JOINT_LEAKYRELU_SLOPE}"
  --weight-net-dropout "${WEIGHT_NET_DROPOUT}"
  --grad-vec-normalize "${GRAD_VEC_NORMALIZE}"
  --weight-net-layer-embed-dim "${WEIGHT_NET_LAYER_EMBED_DIM}"
  --weight-net-task-embed-dim "${WEIGHT_NET_TASK_EMBED_DIM}"
  --weight-net-cnt-grad-hidden-dim "${CNT_GRAD_HIDDEN_DIM}"
  --det-ap-score-thr "${DET_AP_SCORE_THR}"
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
  --phi-grad-clip-norm "${PHI_GRAD_CLIP_NORM}"
  --cnt-backbone-grad-mult 1
  --select-best-from-stage2
)

if [[ "${UNFREEZE_BACKBONE}" == "1" ]]; then
  ARGS+=( --unfreeze-backbone )
fi

if [[ "${LORA_MOE}" == "1" ]]; then
  ARGS+=( --use-lora-moe )
  ARGS+=(
    --lora-rank "${LORA_RANK}"
    --num-experts-private "${NUM_EXPERTS_PRIVATE}"
    --num-experts-shared "${NUM_EXPERTS_SHARED}"
    --moe-k-private "${MOE_K_PRIVATE}"
    --moe-k-shared "${MOE_K_SHARED}"
  )
  if [[ "${USE_DYNAMIC_LOSS_WEIGHT}" == "1" ]]; then
    ARGS+=( --dynamic-loss-weight )
  fi
fi

if [[ "${LEARN_LOSS_WEIGHTS_MLP}" == "1" ]]; then
  ARGS+=( --learn-loss-weights-mlp )
fi
if [[ "${WEIGHT_NET_USE_MULTILAYER_GRADS}" == "1" ]]; then
  ARGS+=( --weight-net-use-multilayer-grads )
fi
if [[ "${LOSS_WEIGHT_PRIOR_MUL}" == "1" ]]; then
  ARGS+=( --loss-weight-prior-mul )
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
if [[ "${SKIP_VALIDATION}" == "1" ]]; then ARGS+=( --skip-validation ); fi

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

# dump final command + env for reproducibility
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
  exec "${RUN_CMD[@]}"
fi

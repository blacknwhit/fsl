#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${GPU_INDEX:-7}"
CUDA_VISIBLE_DEVICES_OVERRIDE="${CUDA_VISIBLE_DEVICES_OVERRIDE:-0,1,2,3,4,5,6,7}"
#CUDA_VISIBLE_DEVICES_OVERRIDE="${CUDA_VISIBLE_DEVICES_OVERRIDE:-0,1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$(readlink -f "${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")")"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_ROOT_DEFAULT="/nas/liyangguang103/new_fscd"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"
if [[ -d "${PROJECT_ROOT}" && -f "${PROJECT_ROOT}/mod_squad_plaintrain/train.py" ]]; then
  cd "${PROJECT_ROOT}"
else
  PROJECT_ROOT="${REPO_ROOT}"
  cd "${REPO_ROOT}"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/object_detection${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/miniconda3/envs/fsl/bin/python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. Set PYTHON_BIN=/path/to/python" >&2
  exit 1
fi

DET_DATA_ROOT="${DET_DATA_ROOT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco}"
DET_TRAIN_ANN="${DET_TRAIN_ANN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_train_10per.json}"
DET_VAL_ANN="${DET_VAL_ANN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_val.json}"
DET_TRAIN_IMG_DIR="${DET_TRAIN_IMG_DIR:-}"
DET_VAL_IMG_DIR="${DET_VAL_IMG_DIR:-}"
SEG_TRAIN_DIR="${SEG_TRAIN_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/train_5500_10per}"
SEG_VAL_DIR="${SEG_VAL_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/val}"
CNT_DATA_ROOT="${CNT_DATA_ROOT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA}"
CNT_TRAIN_DIR="${CNT_TRAIN_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA/train_data_class8_10per}"
CNT_VAL_DIR="${CNT_VAL_DIR:-}"
BACKBONE_CKPT="${BACKBONE_CKPT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"

MODEL_NAME="${MODEL_NAME:-dinov3_vitl16}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
EPOCHS="${EPOCHS:-150}"
SEED="${SEED:-42}"
VAL_EVERY="${VAL_EVERY:-1}"
VAL_LAST_N_EPOCHS="${VAL_LAST_N_EPOCHS:-50}"
SKIP_VALIDATION="${SKIP_VALIDATION:-0}"
DET_BATCH="${DET_BATCH:-32}"
SEG_BATCH="${SEG_BATCH:-32}"
CNT_BATCH="${CNT_BATCH:-32}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
DET_LR="${DET_LR:-}"
SEG_LR="${SEG_LR:-}"
CNT_LR="${CNT_LR:-}"
DET_WD="${DET_WD:-}"
SEG_WD="${SEG_WD:-}"
CNT_WD="${CNT_WD:-}"
LOSS_WEIGHTS="${LOSS_WEIGHTS:-15:8:1}"
MI_LOSS_WEIGHT="${MI_LOSS_WEIGHT:-0.005}"
LORA_MOE="${LORA_MOE:-1}"
LORA_RANK="${LORA_RANK:-8}"
NUM_EXPERTS_PRIVATE="${NUM_EXPERTS_PRIVATE:-0}"
NUM_EXPERTS_SHARED="${NUM_EXPERTS_SHARED:-6}"
MOE_K_PRIVATE="${MOE_K_PRIVATE:-0}"
MOE_K_SHARED="${MOE_K_SHARED:-2}"
GRAD_CHECKPOINTING="${GRAD_CHECKPOINTING:-0}"
DET_AP_SCORE_THR="${DET_AP_SCORE_THR:-0.0}"
SAVE_DIR="${SAVE_DIR:-runs/mod_squad_plaintrain_sharedonly_mi_10per}"
LOG_FILE="${LOG_FILE:-${SAVE_DIR}/train.log}"
LOG_TO_FILE="${LOG_TO_FILE:-1}"
LOG_APPEND="${LOG_APPEND:-1}"
NUM_WORKERS="${NUM_WORKERS:-5}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-100}"
AMP="${AMP:-0}"
DEVICE="${DEVICE:-cuda}"
NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"
LINEAR_LR_SCALE="${LINEAR_LR_SCALE:-1}"
TRAIN_MODULE="${TRAIN_MODULE:-mod_squad_plaintrain.train}"

mkdir -p "${SAVE_DIR}"
cp -a "${SCRIPT_PATH}" "${SAVE_DIR}/train.sh.bak" 2>/dev/null || true

if [[ -n "${CUDA_VISIBLE_DEVICES_OVERRIDE}" ]]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_OVERRIDE}"
elif [[ -n "${GPU_INDEX}" ]]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
fi

export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

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
    nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
    return
  fi
  echo 0
}

VISIBLE_GPU_COUNT="$(count_visible_gpus)"
if [[ "${NPROC_PER_NODE}" == "auto" ]]; then
  if [[ "${VISIBLE_GPU_COUNT}" -gt 0 ]]; then
    NPROC_PER_NODE="${VISIBLE_GPU_COUNT}"
  else
    NPROC_PER_NODE=1
  fi
fi
if [[ "${VISIBLE_GPU_COUNT}" -gt 0 && "${NPROC_PER_NODE}" -gt "${VISIBLE_GPU_COUNT}" ]]; then
  echo "[ddp][warn] NPROC_PER_NODE=${NPROC_PER_NODE} > visible_gpus=${VISIBLE_GPU_COUNT}; clamping." >&2
  NPROC_PER_NODE="${VISIBLE_GPU_COUNT}"
fi

scale_float() {
  "${PYTHON_BIN}" - "$1" "$2" <<'PY'
import sys
print(f"{float(sys.argv[1]) * float(sys.argv[2]):.12g}")
PY
}

if [[ "${LINEAR_LR_SCALE}" == "1" && "${NPROC_PER_NODE}" -gt 1 ]]; then
  LR="$(scale_float "${LR}" "${NPROC_PER_NODE}")"
  if [[ -n "${DET_LR}" ]]; then DET_LR="$(scale_float "${DET_LR}" "${NPROC_PER_NODE}")"; fi
  if [[ -n "${SEG_LR}" ]]; then SEG_LR="$(scale_float "${SEG_LR}" "${NPROC_PER_NODE}")"; fi
  if [[ -n "${CNT_LR}" ]]; then CNT_LR="$(scale_float "${CNT_LR}" "${NPROC_PER_NODE}")"; fi
fi

ARGS=(
  --model-name "${MODEL_NAME}"
  --image-size "${IMAGE_SIZE}"
  --epochs "${EPOCHS}"
  --seed "${SEED}"
  --val-every "${VAL_EVERY}"
  --val-last-n-epochs "${VAL_LAST_N_EPOCHS}"
  --device "${DEVICE}"
  --num-workers "${NUM_WORKERS}"
  --save-dir "${SAVE_DIR}"
  --loss-weights "${LOSS_WEIGHTS}"
  --mi-loss-weight "${MI_LOSS_WEIGHT}"
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
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --lora-rank "${LORA_RANK}"
  --num-experts-private "${NUM_EXPERTS_PRIVATE}"
  --num-experts-shared "${NUM_EXPERTS_SHARED}"
  --moe-k-private "${MOE_K_PRIVATE}"
  --moe-k-shared "${MOE_K_SHARED}"
)

if [[ "${LORA_MOE}" == "1" ]]; then
  ARGS+=(--use-lora-moe)
else
  ARGS+=(--no-use-lora-moe)
fi
if [[ "${GRAD_CHECKPOINTING}" == "1" ]]; then
  ARGS+=(--grad-checkpointing)
else
  ARGS+=(--no-grad-checkpointing)
fi
if [[ -n "${BACKBONE_CKPT}" ]]; then ARGS+=(--backbone-checkpoint "${BACKBONE_CKPT}"); fi
if [[ -n "${DET_TRAIN_IMG_DIR}" ]]; then ARGS+=(--det-train-img-dir "${DET_TRAIN_IMG_DIR}"); fi
if [[ -n "${DET_VAL_IMG_DIR}" ]]; then ARGS+=(--det-val-img-dir "${DET_VAL_IMG_DIR}"); fi
if [[ -n "${CNT_VAL_DIR}" ]]; then ARGS+=(--cnt-val-dir "${CNT_VAL_DIR}"); fi
if [[ -n "${DET_LR}" ]]; then ARGS+=(--det-lr "${DET_LR}"); fi
if [[ -n "${SEG_LR}" ]]; then ARGS+=(--seg-lr "${SEG_LR}"); fi
if [[ -n "${CNT_LR}" ]]; then ARGS+=(--cnt-lr "${CNT_LR}"); fi
if [[ -n "${DET_WD}" ]]; then ARGS+=(--det-weight-decay "${DET_WD}"); fi
if [[ -n "${SEG_WD}" ]]; then ARGS+=(--seg-weight-decay "${SEG_WD}"); fi
if [[ -n "${CNT_WD}" ]]; then ARGS+=(--cnt-weight-decay "${CNT_WD}"); fi
if [[ "${AMP}" == "1" ]]; then ARGS+=(--amp); fi
if [[ "${SKIP_VALIDATION}" == "1" ]]; then ARGS+=(--skip-validation); fi

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

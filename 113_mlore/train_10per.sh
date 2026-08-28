#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${GPU_INDEX:-7}"
# CUDA_VISIBLE_DEVICES_OVERRIDE="${CUDA_VISIBLE_DEVICES_OVERRIDE:-0,1,2,3,4,5,6,7}"
CUDA_VISIBLE_DEVICES_OVERRIDE="${CUDA_VISIBLE_DEVICES_OVERRIDE:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$(readlink -f "${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")")"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_ROOT_DEFAULT="/nas/liyangguang103/new_fscd"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"

if [[ -d "${PROJECT_ROOT}" && -f "${PROJECT_ROOT}/113_mlore/train.py" ]]; then
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

# Prefer system driver libcuda to avoid 803 errors from CUDA compat stubs.
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

DET_DATA_ROOT="${DET_DATA_ROOT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco}"
DET_TRAIN_ANN="${DET_TRAIN_ANN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_train_10per.json}"
DET_VAL_ANN="${DET_VAL_ANN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_val.json}"

SEG_TRAIN_DIR="${SEG_TRAIN_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/train_5500_10per}"
SEG_VAL_DIR="${SEG_VAL_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/val}"

CNT_DATA_ROOT="${CNT_DATA_ROOT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA}"
CNT_TRAIN_DIR="${CNT_TRAIN_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA/train_data_class8_10per}"

BACKBONE_CKPT="${BACKBONE_CKPT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"

MODEL_NAME="${MODEL_NAME:-dinov3_vitl16}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
EPOCHS="${EPOCHS:-150}"
SEED="${SEED:-42}"
VAL_EVERY="${VAL_EVERY:-1}"
VALIDATE_LAST_N_EPOCHS="${VALIDATE_LAST_N_EPOCHS:-20}"

DET_BATCH="${DET_BATCH:-32}"
SEG_BATCH="${SEG_BATCH:-32}"
CNT_BATCH="${CNT_BATCH:-32}"

LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-100}"

MLORE_DECODER_DIM="${MLORE_DECODER_DIM:-256}"
MLORE_RANK_LIST="${MLORE_RANK_LIST:-8,16,24,32,40,48}"
MLORE_TOPK="${MLORE_TOPK:-4}"
MLORE_TASK_RANK="${MLORE_TASK_RANK:-32}"
MLORE_PRE_SOFTMAX="${MLORE_PRE_SOFTMAX:-0}"
MLORE_LOAD_BALANCING_WEIGHT="${MLORE_LOAD_BALANCING_WEIGHT:-3e-4}"
MLORE_SELECT_LAYERS="${MLORE_SELECT_LAYERS:-23}"
MLORE_NUM_STAGES="${MLORE_NUM_STAGES:-1}"

GRAD_CHECKPOINTING="${GRAD_CHECKPOINTING:-1}"
DET_AP_SCORE_THR="${DET_AP_SCORE_THR:-0.0}"
SAVE_DIR="${SAVE_DIR:-runs/113_mlore_10per}"
LOG_FILE="${LOG_FILE:-${SAVE_DIR}/train.log}"
LOG_TO_FILE="${LOG_TO_FILE:-1}"
LOG_APPEND="${LOG_APPEND:-1}"
NUM_WORKERS="${NUM_WORKERS:-5}"
AMP="${AMP:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "${SAVE_DIR}"
{
  echo "[time] $(date -Is)"
  echo "[pwd]  $(pwd)"
  echo "[script] ${SCRIPT_PATH}"
} > "${SAVE_DIR}/run_meta.txt"

if [[ -n "${CUDA_VISIBLE_DEVICES_OVERRIDE}" ]]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_OVERRIDE}"
elif [[ -n "${GPU_INDEX}" ]]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
fi

ARGS=(
  --model-name "${MODEL_NAME}"
  --image-size "${IMAGE_SIZE}"
  --epochs "${EPOCHS}"
  --seed "${SEED}"
  --val-every "${VAL_EVERY}"
  --validate-last-n-epochs "${VALIDATE_LAST_N_EPOCHS}"
  --device "${DEVICE}"
  --num-workers "${NUM_WORKERS}"
  --save-dir "${SAVE_DIR}"
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
  --mlore-decoder-dim "${MLORE_DECODER_DIM}"
  --mlore-rank-list "${MLORE_RANK_LIST}"
  --mlore-topk "${MLORE_TOPK}"
  --mlore-task-rank "${MLORE_TASK_RANK}"
  --mlore-load-balancing-weight "${MLORE_LOAD_BALANCING_WEIGHT}"
  --mlore-select-layers "${MLORE_SELECT_LAYERS}"
  --mlore-num-stages "${MLORE_NUM_STAGES}"
)

if [[ -n "${BACKBONE_CKPT}" ]]; then ARGS+=( --backbone-checkpoint "${BACKBONE_CKPT}" ); fi
if [[ "${MLORE_PRE_SOFTMAX}" == "1" ]]; then ARGS+=( --mlore-pre-softmax ); fi
if [[ "${GRAD_CHECKPOINTING}" == "1" ]]; then
  ARGS+=( --grad-checkpointing )
else
  ARGS+=( --no-grad-checkpointing )
fi
if [[ "${AMP}" == "1" ]]; then ARGS+=( --amp ); fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  RUN_CMD=(
    "${PYTHON_BIN}" -m torch.distributed.run --standalone
    --nproc_per_node "${NPROC_PER_NODE}"
    --module "113_mlore.train"
    "${ARGS[@]}"
  )
else
  RUN_CMD=("${PYTHON_BIN}" -m "113_mlore.train" "${ARGS[@]}")
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
  "${RUN_CMD[@]}"
fi

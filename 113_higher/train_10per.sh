#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${GPU_INDEX:-7}"
CUDA_VISIBLE_DEVICES_OVERRIDE="${CUDA_VISIBLE_DEVICES_OVERRIDE:-0,1,2,3,4,5,6,7}"
#CUDA_VISIBLE_DEVICES_OVERRIDE="${CUDA_VISIBLE_DEVICES_OVERRIDE:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_ROOT_DEFAULT="/nas/liyangguang103/new_fscd"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"
if [[ -d "${PROJECT_ROOT}" && -f "${PROJECT_ROOT}/113_higher/train.py" ]]; then
  cd "${PROJECT_ROOT}"
else
  PROJECT_ROOT="${REPO_ROOT}"
  cd "${REPO_ROOT}"
fi
SWANLAB_API_KEY="${SWANLAB_API_KEY:-kmuC8F4SjMYKIXDDLVtME}"
export SWANLAB_API_KEY
export PYTHONPATH="${PROJECT_ROOT}/object_detection${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/miniconda3/envs/fsl/bin/python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. Set PYTHON_BIN=/path/to/python" >&2
  exit 1
fi

DET_DATA_ROOT="${DET_DATA_ROOT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco}"
DET_TRAIN_ANN="${DET_TRAIN_ANN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_train_10per.json}"
DET_VAL_ANN="${DET_VAL_ANN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_val.json}"

SEG_TRAIN_DIR="${SEG_TRAIN_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/train_5500_10per}"
SEG_VAL_DIR="${SEG_VAL_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/val}"

CNT_DATA_ROOT="${CNT_DATA_ROOT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA}"
CNT_TRAIN_DIR="${CNT_TRAIN_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA/train_data_class8_10per}"
CNT_VAL_DIR="${CNT_VAL_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA/val_data_class8}"

BACKBONE_CKPT="${BACKBONE_CKPT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"

UNFREEZE_BACKBONE="${UNFREEZE_BACKBONE:-0}"
MODEL_NAME="${MODEL_NAME:-dinov3_vitl16}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-90}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-50}"
META_SPLIT="${META_SPLIT:-0.2}"
META_SEED="${META_SEED:-42}"
SEED="${SEED:-42}"

DET_BATCH="${DET_BATCH:-2}"
SEG_BATCH="${SEG_BATCH:-2}"
CNT_BATCH="${CNT_BATCH:-2}"

STAGE1_INNER_LR="${STAGE1_INNER_LR:-1e-4}"
STAGE1_PHI_LR="${STAGE1_PHI_LR:-1e-3}"
STAGE2_LR="${STAGE2_LR:-1e-4}"
STAGE2_WEIGHT_DECAY="${STAGE2_WEIGHT_DECAY:-1e-4}"
BACKBONE_LR_MULT="${BACKBONE_LR_MULT:-0.1}"

LORA_MOE="${LORA_MOE:-1}"
LORA_RANK="${LORA_RANK:-8}"
NUM_EXPERTS_PRIVATE="${NUM_EXPERTS_PRIVATE:-3}"
NUM_EXPERTS_SHARED="${NUM_EXPERTS_SHARED:-6}"
MOE_K_PRIVATE="${MOE_K_PRIVATE:-2}"
MOE_K_SHARED="${MOE_K_SHARED:-2}"
GRAD_CHECKPOINTING="${GRAD_CHECKPOINTING:-1}"

DET_AP_SCORE_THR="${DET_AP_SCORE_THR:-0.0}"
SAVE_DIR="${SAVE_DIR:-runs/113_higher}"
LOG_FILE="${LOG_FILE:-${SAVE_DIR}/train.log}"
LOG_TO_FILE="${LOG_TO_FILE:-1}"
LOG_APPEND="${LOG_APPEND:-1}"
NUM_WORKERS="${NUM_WORKERS:-16}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-100}"
SKIP_VALIDATION="${SKIP_VALIDATION:-0}"
SWANLAB_ENABLE="${SWANLAB_ENABLE:-1}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-}"
SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-}"
SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-}"
SWANLAB_MODE="${SWANLAB_MODE:-}"
SWANLAB_LOGDIR="${SWANLAB_LOGDIR:-}"
TEST_AFTER_TRAIN="${TEST_AFTER_TRAIN:-1}"
TEST_SCRIPT="${TEST_SCRIPT:-${SCRIPT_DIR}/test.sh}"
TEST_CKPT="${TEST_CKPT:-${SAVE_DIR}/best_combo.pt}"

if [[ -n "${CUDA_VISIBLE_DEVICES_OVERRIDE}" ]]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_OVERRIDE}"
elif [[ -n "${GPU_INDEX}" ]]; then
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"
fi

export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
DEVICE="${DEVICE:-cuda}"
NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"
LINEAR_LR_SCALE="${LINEAR_LR_SCALE:-1}"
TRAIN_MODULE="${TRAIN_MODULE:-113_higher.train}"

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

VISIBLE_GPU_COUNT="$(count_visible_gpus)"
if [[ "${NPROC_PER_NODE}" == "auto" ]]; then
  if [[ "${VISIBLE_GPU_COUNT}" -gt 0 ]]; then
    NPROC_PER_NODE="${VISIBLE_GPU_COUNT}"
  else
    NPROC_PER_NODE=1
  fi
fi
if ! [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: NPROC_PER_NODE must be an integer or auto, got: ${NPROC_PER_NODE}" >&2
  exit 1
fi
if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  NPROC_PER_NODE=1
fi
if [[ "${VISIBLE_GPU_COUNT}" -gt 0 && "${NPROC_PER_NODE}" -gt "${VISIBLE_GPU_COUNT}" ]]; then
  NPROC_PER_NODE="${VISIBLE_GPU_COUNT}"
fi

echo "[ddp] visible_gpus=${VISIBLE_GPU_COUNT}, nproc_per_node=${NPROC_PER_NODE}"

if [[ "${LINEAR_LR_SCALE}" == "1" && "${NPROC_PER_NODE}" -gt 1 ]]; then
  STAGE1_INNER_LR="$(scale_float "${STAGE1_INNER_LR}" "${NPROC_PER_NODE}")"
  STAGE1_PHI_LR="$(scale_float "${STAGE1_PHI_LR}" "${NPROC_PER_NODE}")"
  STAGE2_LR="$(scale_float "${STAGE2_LR}" "${NPROC_PER_NODE}")"
fi

mkdir -p "${SAVE_DIR}"

ARGS=(
  --model-name "${MODEL_NAME}"
  --image-size "${IMAGE_SIZE}"
  --stage1-epochs "${STAGE1_EPOCHS}"
  --stage2-epochs "${STAGE2_EPOCHS}"
  --meta-split "${META_SPLIT}"
  --meta-seed "${META_SEED}"
  --seed "${SEED}"
  --device "${DEVICE}"
  --num-workers "${NUM_WORKERS}"
  --save-dir "${SAVE_DIR}"
  --stage1-inner-lr "${STAGE1_INNER_LR}"
  --stage1-phi-lr "${STAGE1_PHI_LR}"
  --stage2-lr "${STAGE2_LR}"
  --stage2-weight-decay "${STAGE2_WEIGHT_DECAY}"
  --backbone-lr-mult "${BACKBONE_LR_MULT}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
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
  --cnt-val-dir "${CNT_VAL_DIR}"
  --cnt-batch-size "${CNT_BATCH}"
  --cnt-keep-aspect
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
fi
if [[ "${GRAD_CHECKPOINTING}" == "1" ]]; then
  ARGS+=( --grad-checkpointing )
else
  ARGS+=( --no-grad-checkpointing )
fi
if [[ "${SKIP_VALIDATION}" == "1" ]]; then
  ARGS+=( --skip-validation )
fi
if [[ -n "${BACKBONE_CKPT}" ]]; then
  ARGS+=( --backbone-checkpoint "${BACKBONE_CKPT}" )
fi
if [[ "${SWANLAB_ENABLE}" == "1" ]]; then
  export SWANLAB_API_KEY
  ARGS+=( --use-swanlab )
  if [[ -n "${SWANLAB_PROJECT}" ]]; then
    ARGS+=( --swanlab-project "${SWANLAB_PROJECT}" )
  fi
  if [[ -n "${SWANLAB_WORKSPACE}" ]]; then
    ARGS+=( --swanlab-workspace "${SWANLAB_WORKSPACE}" )
  fi
  if [[ -n "${SWANLAB_EXPERIMENT_NAME}" ]]; then
    ARGS+=( --swanlab-experiment-name "${SWANLAB_EXPERIMENT_NAME}" )
  fi
  if [[ -n "${SWANLAB_MODE}" ]]; then
    ARGS+=( --swanlab-mode "${SWANLAB_MODE}" )
  fi
  if [[ -n "${SWANLAB_LOGDIR}" ]]; then
    ARGS+=( --swanlab-logdir "${SWANLAB_LOGDIR}" )
  fi
fi

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
  echo -n "[cmd_plain] "
  echo "${RUN_CMD[*]}"
} > "${SAVE_DIR}/cmd.txt"
env | sort > "${SAVE_DIR}/env.txt"

echo "[run] ${RUN_CMD[*]}"
if [[ "${LOG_TO_FILE}" == "1" ]]; then
  mkdir -p "$(dirname "${LOG_FILE}")"
  if [[ "${LOG_APPEND}" == "1" ]]; then
    "${RUN_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
  else
    "${RUN_CMD[@]}" 2>&1 | tee "${LOG_FILE}"
  fi
else
  "${RUN_CMD[@]}"
fi

if [[ "${TEST_AFTER_TRAIN}" == "1" ]]; then
  if [[ ! -f "${TEST_SCRIPT}" ]]; then
    echo "ERROR: test script not found: ${TEST_SCRIPT}" >&2
    exit 2
  fi
  if [[ ! -f "${TEST_CKPT}" ]]; then
    echo "ERROR: test checkpoint not found after training: ${TEST_CKPT}" >&2
    exit 2
  fi
  TEST_CMD=(bash "${TEST_SCRIPT}" "${TEST_CKPT}")
  echo "[test] ${TEST_CMD[*]}"
  if [[ "${LOG_TO_FILE}" == "1" ]]; then
    if [[ "${LOG_APPEND}" == "1" ]]; then
      "${TEST_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
    else
      "${TEST_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
    fi
  else
    "${TEST_CMD[@]}"
  fi
fi

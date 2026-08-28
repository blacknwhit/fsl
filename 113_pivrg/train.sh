#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${GPU_INDEX:-1}"
CUDA_VISIBLE_DEVICES_OVERRIDE="${CUDA_VISIBLE_DEVICES_OVERRIDE:-5}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_ROOT_DEFAULT="/nas/liyangguang103/new_fscd"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"

if [[ -d "${PROJECT_ROOT}" && -f "${PROJECT_ROOT}/113_pivrg/train.py" ]]; then
  cd "${PROJECT_ROOT}"
else
  PROJECT_ROOT="${REPO_ROOT}"
  cd "${REPO_ROOT}"
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/object_detection${PYTHONPATH:+:${PYTHONPATH}}"

PYTHON_BIN="${PYTHON_BIN:-/nas/liyangguang103/anaconda3/envs/dam/bin/python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found. Set PYTHON_BIN=/path/to/python" >&2
  exit 1
fi

# Prefer system driver libcuda to avoid CUDA compat stub conflicts.
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

DET_DATA_ROOT="${DET_DATA_ROOT:-/nas/liyangguang103/newdataset/CD-ObjectDetection/DIOR/coco}"
DET_TRAIN_ANN="${DET_TRAIN_ANN:-/nas/liyangguang103/newdataset/CD-ObjectDetection/DIOR/coco/annotations/instances_train.json}"
SEG_TRAIN_DIR="${SEG_TRAIN_DIR:-/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/train_5500}"
SEG_VAL_DIR="${SEG_VAL_DIR:-/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/val}"
CNT_DATA_ROOT="${CNT_DATA_ROOT:-/nas/liyangguang103/newdataset/CD-Count/DSACA}"
CNT_TRAIN_DIR="${CNT_TRAIN_DIR:-/nas/liyangguang103/newdataset/CD-Count/DSACA/train_data_class8}"
BACKBONE_CKPT="${BACKBONE_CKPT:-/nas/liyangguang103/old_fscd/CD-FSOD/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth}"

MODEL_NAME="${MODEL_NAME:-dinov3_vitl16}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
EPOCHS="${EPOCHS:-150}"
DET_BATCH="${DET_BATCH:-4}"
SEG_BATCH="${SEG_BATCH:-4}"
CNT_BATCH="${CNT_BATCH:-4}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
LOSS_WEIGHTS="${LOSS_WEIGHTS:-15:8:1}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-100}"
VAL_EVERY="${VAL_EVERY:-1}"
SAVE_DIR="${SAVE_DIR:-runs/113_pivrg}"
LOG_FILE="${LOG_FILE:-${SAVE_DIR}/train.log}"
LOG_TO_FILE="${LOG_TO_FILE:-1}"
LOG_APPEND="${LOG_APPEND:-1}"
NUM_WORKERS="${NUM_WORKERS:-5}"
AMP="${AMP:-0}"
DEVICE="${DEVICE:-cuda}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
AUTO_TEST_AFTER_TRAIN="${AUTO_TEST_AFTER_TRAIN:-1}"

LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_LR="${LORA_LR:-}"
LORA_WD="${LORA_WD:-0.0}"
DET_LR="${DET_LR:-}"
SEG_LR="${SEG_LR:-}"
CNT_LR="${CNT_LR:-}"
DET_WD="${DET_WD:-}"
SEG_WD="${SEG_WD:-}"
CNT_WD="${CNT_WD:-}"
DET_AP_SCORE_THR="${DET_AP_SCORE_THR:-0.0}"
CNT_COUNT_LOSS_WEIGHT="${CNT_COUNT_LOSS_WEIGHT:-1.0}"
PIVRG_BASE_DET_AP50="${PIVRG_BASE_DET_AP50:-0}"
PIVRG_BASE_SEG_MIOU="${PIVRG_BASE_SEG_MIOU:-0}"
PIVRG_BASE_CNT_MAE="${PIVRG_BASE_CNT_MAE:-0}"
PIVRG_BOUND="${PIVRG_BOUND:-2.0}"
PIVRG_MINTEMP="${PIVRG_MINTEMP:-10.0}"
PIVRG_WARMUP_EPOCHS="${PIVRG_WARMUP_EPOCHS:-10}"

mkdir -p "${SAVE_DIR}"

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
  --device "${DEVICE}"
  --num-workers "${NUM_WORKERS}"
  --save-dir "${SAVE_DIR}"
  --loss-weights "${LOSS_WEIGHTS}"
  --det-ap-score-thr "${DET_AP_SCORE_THR}"
  --cnt-count-loss-weight "${CNT_COUNT_LOSS_WEIGHT}"
  --val-every "${VAL_EVERY}"
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
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --lora-rank "${LORA_RANK}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
  --lora-weight-decay "${LORA_WD}"
  --pivrg-base-det-ap50 "${PIVRG_BASE_DET_AP50}"
  --pivrg-base-seg-miou "${PIVRG_BASE_SEG_MIOU}"
  --pivrg-base-cnt-mae "${PIVRG_BASE_CNT_MAE}"
  --pivrg-bound "${PIVRG_BOUND}"
  --pivrg-mintemp "${PIVRG_MINTEMP}"
  --pivrg-warmup-epochs "${PIVRG_WARMUP_EPOCHS}"
)

if [[ -n "${BACKBONE_CKPT}" ]]; then ARGS+=( --backbone-checkpoint "${BACKBONE_CKPT}" ); fi
if [[ -n "${LORA_LR}" ]]; then ARGS+=( --lora-lr "${LORA_LR}" ); fi
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
    --module "113_pivrg.train"
    "${ARGS[@]}"
  )
else
  RUN_CMD=("${PYTHON_BIN}" -m "113_pivrg.train" "${ARGS[@]}")
fi

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
    if [[ "${LOG_APPEND}" == "1" ]]; then
      "${TEST_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
    else
      "${TEST_CMD[@]}" 2>&1 | tee -a "${LOG_FILE}"
    fi
  else
    "${TEST_CMD[@]}"
  fi
fi

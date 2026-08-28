#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX="${GPU_INDEX:-7}"
CUDA_VISIBLE_DEVICES_OVERRIDE="${CUDA_VISIBLE_DEVICES_OVERRIDE:-0,1,2,3,4,5,6,7}"
#CUDA_VISIBLE_DEVICES_OVERRIDE="${CUDA_VISIBLE_DEVICES_OVERRIDE:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$(readlink -f "${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")")"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_ROOT_DEFAULT="/nas/liyangguang103/new_fscd"
PROJECT_ROOT="${PROJECT_ROOT:-${PROJECT_ROOT_DEFAULT}}"
if [[ -d "${PROJECT_ROOT}" && -f "${PROJECT_ROOT}/115_grpo_mainonly/train.py" ]]; then
  cd "${PROJECT_ROOT}"
else
  PROJECT_ROOT="${REPO_ROOT}"
  cd "${REPO_ROOT}"
fi

export PYTHONPATH="${PROJECT_ROOT}/object_detection${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${PYTHON_BIN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/miniconda3/envs/fsl/bin/python}"

DET_DATA_ROOT="${DET_DATA_ROOT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/train_10per}"
DET_TRAIN_ANN="${DET_TRAIN_ANN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_train_10per.json}"
DET_VAL_ANN="${DET_VAL_ANN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_val.json}"
DET_TRAIN_IMG_DIR="${DET_TRAIN_IMG_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco}"
DET_VAL_IMG_DIR="${DET_VAL_IMG_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco}"
STAGE2_DET_VAL_ANN="${STAGE2_DET_VAL_ANN:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_val.json}"
STAGE2_DET_VAL_IMG_DIR="${STAGE2_DET_VAL_IMG_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco}"
SEG_TRAIN_DIR="${SEG_TRAIN_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/train_5500_10per}"
SEG_VAL_DIR="${SEG_VAL_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/val}"
STAGE2_SEG_VAL_DIR="${STAGE2_SEG_VAL_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/val}"
CNT_DATA_ROOT="${CNT_DATA_ROOT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/train_10per}"
CNT_TRAIN_DIR="${CNT_TRAIN_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA/train_data_class8_10per}"
CNT_VAL_DIR="${CNT_VAL_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA/val_data_class8}"
STAGE2_CNT_VAL_DIR="${STAGE2_CNT_VAL_DIR:-/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA/val_data_class8}"
BACKBONE_CKPT="${BACKBONE_CKPT:-/data/xiangyuyue/ULLM-zf/fsl-20260209/pretrained/facebook_vit-mae-large}"

normalize_cnt_split_root() {
  local p="$1"
  local b
  b="$(basename "$p")"
  if [[ "$b" == "gt_density_map" || "$b" == "gt_density_map_compressed" ]]; then
    dirname "$p"
  else
    echo "$p"
  fi
}

CNT_TRAIN_DIR="$(normalize_cnt_split_root "${CNT_TRAIN_DIR}")"
CNT_VAL_DIR="$(normalize_cnt_split_root "${CNT_VAL_DIR}")"
STAGE2_CNT_VAL_DIR="$(normalize_cnt_split_root "${STAGE2_CNT_VAL_DIR}")"

UNFREEZE_BACKBONE="${UNFREEZE_BACKBONE:-0}"
MODEL_NAME="${MODEL_NAME:-facebook/vit-mae-large}"
IMAGE_SIZE="${IMAGE_SIZE:-448}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-${EPOCHS:-100}}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-50}"
STAGE2_INIT_CHECKPOINT="${STAGE2_INIT_CHECKPOINT:-}"
LOSS_WEIGHTS="${LOSS_WEIGHTS:-15,8,1}"
STAGE1_VAL_LAST_K_EPOCHS="${STAGE1_VAL_LAST_K_EPOCHS:-50}"

DET_BATCH="${DET_BATCH:-2}"
SEG_BATCH="${SEG_BATCH:-2}"
CNT_BATCH="${CNT_BATCH:-2}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

LORA_MOE="${LORA_MOE:-1}"
LORA_RANK="${LORA_RANK:-8}"
NUM_EXPERTS_PRIVATE="${NUM_EXPERTS_PRIVATE:-3}"
NUM_EXPERTS_SHARED="${NUM_EXPERTS_SHARED:-6}"
MOE_K_PRIVATE="${MOE_K_PRIVATE:-2}"
MOE_K_SHARED="${MOE_K_SHARED:-2}"
GRAD_CHECKPOINTING="${GRAD_CHECKPOINTING:-1}"

BACKBONE_LR_MULT="${BACKBONE_LR_MULT:-0.1}"
GRAD_CLIP_NORM="${GRAD_CLIP_NORM:-100}"
NUM_WORKERS="${NUM_WORKERS:-16}"
DET_AP_SCORE_THR="${DET_AP_SCORE_THR:-0.0}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"

SAVE_DIR="${SAVE_DIR:-runs/115_mainonly_10per}"
LOG_FILE="${LOG_FILE:-${SAVE_DIR}/train.log}"
LOG_TO_FILE="${LOG_TO_FILE:-1}"
LOG_APPEND="${LOG_APPEND:-1}"
DEBUG_STEP_TIMING="${DEBUG_STEP_TIMING:-0}"
DEBUG_STEP_TIMING_INTERVAL="${DEBUG_STEP_TIMING_INTERVAL:-1}"
USE_SWANLAB="${USE_SWANLAB:-1}"
SWANLAB_API_KEY="${SWANLAB_API_KEY:-kmuC8F4SjMYKIXDDLVtME}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-115_grpo_mainonly}"
SWANLAB_WORKSPACE="${SWANLAB_WORKSPACE:-}"
SWANLAB_EXPERIMENT_NAME="${SWANLAB_EXPERIMENT_NAME:-$(basename "${SAVE_DIR}")}"
SWANLAB_MODE="${SWANLAB_MODE:-}"
SWANLAB_LOGDIR="${SWANLAB_LOGDIR:-${SAVE_DIR}/swanlog}"
RUN_TEST_AFTER_TRAIN="${RUN_TEST_AFTER_TRAIN:-1}"
TEST_PRECHECK="${TEST_PRECHECK:-1}"
TEST_SCRIPT="${TEST_SCRIPT:-${SCRIPT_DIR}/test.sh}"
TEST_STAGE1_CKPT="${TEST_STAGE1_CKPT:-${SAVE_DIR}/stage1_best.pt}"
TEST_STAGE2_CKPT="${TEST_STAGE2_CKPT:-${SAVE_DIR}/best_combo.pt}"
TEST_CKPT="${TEST_CKPT:-${TEST_STAGE2_CKPT}}"
TEST_CKPT_CANDIDATES="${TEST_CKPT_CANDIDATES:-final.pt,last.pt,best_combo.pt,stage2_best.pt,stage1_best.pt}"
TEST_TASKS="${TEST_TASKS:-det,seg,cnt}"
TEST_EVAL_FULL_MODEL="${TEST_EVAL_FULL_MODEL:-1}"
TEST_DEVICE="${TEST_DEVICE:-}"
TEST_IMAGE_SIZE="${TEST_IMAGE_SIZE:-}"
TEST_MODEL_NAME="${TEST_MODEL_NAME:-}"
TEST_DET_SCORE_THR="${TEST_DET_SCORE_THR:-}"
TEST_EVAL_SCRIPT="${TEST_EVAL_SCRIPT:-}"
TEST_STATS_DIR="${TEST_STATS_DIR:-}"
TEST_DET_DATA_ROOT="${TEST_DET_DATA_ROOT:-}"
TEST_DET_ANN_FILE="${TEST_DET_ANN_FILE:-}"
TEST_DET_IMG_DIR="${TEST_DET_IMG_DIR:-}"
TEST_SEG_DATA_DIR="${TEST_SEG_DATA_DIR:-}"
TEST_CNT_DATA_ROOT="${TEST_CNT_DATA_ROOT:-}"
TEST_CNT_TEST_DIR="${TEST_CNT_TEST_DIR:-}"

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

export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

DEVICE="${DEVICE:-cuda}"
NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"
TRAIN_MODULE="${TRAIN_MODULE:-115_grpo_mainonly.train}"

count_visible_gpus() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local s="${CUDA_VISIBLE_DEVICES// /}"
    IFS=',' read -r -a arr <<< "$s"
    local n=0
    for x in "${arr[@]}"; do
      [[ -n "$x" ]] && ((n+=1))
    done
    echo "$n"
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

ARGS=(
  --model-name "${MODEL_NAME}"
  --image-size "${IMAGE_SIZE}"
  --stage1-epochs "${STAGE1_EPOCHS}"
  --stage2-epochs "${STAGE2_EPOCHS}"
  --stage1-val-last-k-epochs "${STAGE1_VAL_LAST_K_EPOCHS}"
  --loss-weights "${LOSS_WEIGHTS}"
  --device "${DEVICE}"
  --num-workers "${NUM_WORKERS}"
  --log-interval "${LOG_INTERVAL}"
  --debug-step-timing-interval "${DEBUG_STEP_TIMING_INTERVAL}"
  --save-dir "${SAVE_DIR}"
  --det-ap-score-thr "${DET_AP_SCORE_THR}"
  --det-data-root "${DET_DATA_ROOT}"
  --det-train-ann "${DET_TRAIN_ANN}"
  --det-val-ann "${DET_VAL_ANN}"
  --det-train-img-dir "${DET_TRAIN_IMG_DIR}"
  --det-val-img-dir "${DET_VAL_IMG_DIR}"
  --stage2-det-val-ann "${STAGE2_DET_VAL_ANN}"
  --stage2-det-val-img-dir "${STAGE2_DET_VAL_IMG_DIR}"
  --det-batch-size "${DET_BATCH}"
  --seg-train-dir "${SEG_TRAIN_DIR}"
  --seg-val-dir "${SEG_VAL_DIR}"
  --stage2-seg-val-dir "${STAGE2_SEG_VAL_DIR}"
  --seg-batch-size "${SEG_BATCH}"
  --cnt-data-root "${CNT_DATA_ROOT}"
  --cnt-train-dir "${CNT_TRAIN_DIR}"
  --cnt-val-dir "${CNT_VAL_DIR}"
  --stage2-cnt-val-dir "${STAGE2_CNT_VAL_DIR}"
  --cnt-batch-size "${CNT_BATCH}"
  --lr "${LR}"
  --weight-decay "${WEIGHT_DECAY}"
  --backbone-lr-mult "${BACKBONE_LR_MULT}"
  --grad-clip-norm "${GRAD_CLIP_NORM}"
  --cnt-backbone-grad-mult 1
)

if [[ -n "${STAGE2_INIT_CHECKPOINT}" ]]; then
  ARGS+=( --stage2-init-checkpoint "${STAGE2_INIT_CHECKPOINT}" )
fi

if [[ "${UNFREEZE_BACKBONE}" == "1" ]]; then
  ARGS+=( --unfreeze-backbone )
fi
if [[ "${DEBUG_STEP_TIMING}" == "1" ]]; then
  ARGS+=( --debug-step-timing )
fi
if [[ "${LORA_MOE}" == "1" ]]; then
  ARGS+=( --use-lora-moe )
  ARGS+=( --lora-rank "${LORA_RANK}" --num-experts-private "${NUM_EXPERTS_PRIVATE}" --num-experts-shared "${NUM_EXPERTS_SHARED}" --moe-k-private "${MOE_K_PRIVATE}" --moe-k-shared "${MOE_K_SHARED}" )
fi
if [[ "${GRAD_CHECKPOINTING}" == "1" ]]; then
  ARGS+=( --grad-checkpointing )
else
  ARGS+=( --no-grad-checkpointing )
fi
if [[ "${USE_SWANLAB}" == "1" ]]; then
  export SWANLAB_API_KEY
  ARGS+=( --use-swanlab --swanlab-project "${SWANLAB_PROJECT}" --swanlab-experiment-name "${SWANLAB_EXPERIMENT_NAME}" --swanlab-logdir "${SWANLAB_LOGDIR}" )
  if [[ -n "${SWANLAB_WORKSPACE}" ]]; then
    ARGS+=( --swanlab-workspace "${SWANLAB_WORKSPACE}" )
  fi
  if [[ -n "${SWANLAB_MODE}" ]]; then
    ARGS+=( --swanlab-mode "${SWANLAB_MODE}" )
  fi
fi
if [[ -n "${BACKBONE_CKPT}" ]]; then
  ARGS+=( --backbone-checkpoint "${BACKBONE_CKPT}" )
fi

mkdir -p "${SAVE_DIR}"
cp -a "${SCRIPT_PATH}" "${SAVE_DIR}/train_10per.sh.bak"

run_training() {
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
}

run_post_train_eval() {
  local ckpt_path="$1"
  local stats_dir="${TEST_STATS_DIR:-${SAVE_DIR}/stats_eval_after_train_$(date +%Y%m%d_%H%M%S)}"
  local test_device="${TEST_DEVICE:-${DEVICE}}"
  local test_image_size="${TEST_IMAGE_SIZE:-${IMAGE_SIZE}}"
  local test_model_name="${TEST_MODEL_NAME:-${MODEL_NAME}}"
  local test_det_score_thr="${TEST_DET_SCORE_THR:-${DET_AP_SCORE_THR}}"
  local test_backbone_ckpt="${TEST_BACKBONE_CKPT:-${BACKBONE_CKPT}}"
  local -a env_args=(
    "PROJECT_ROOT=${PROJECT_ROOT}"
    "PYTHON_BIN=${PYTHON_BIN}"
    "TASKS=${TEST_TASKS}"
    "EVAL_FULL_MODEL=${TEST_EVAL_FULL_MODEL}"
    "DEVICE=${test_device}"
    "IMAGE_SIZE=${test_image_size}"
    "MODEL_NAME=${test_model_name}"
    "DET_SCORE_THR=${test_det_score_thr}"
    "STATS_DIR=${stats_dir}"
  )

  if [[ -n "${test_backbone_ckpt}" ]]; then
    env_args+=("BACKBONE_CKPT=${test_backbone_ckpt}")
  fi

  if [[ "${RUN_TEST_AFTER_TRAIN}" != "1" ]]; then
    echo "[post-train] skip eval (RUN_TEST_AFTER_TRAIN=${RUN_TEST_AFTER_TRAIN})"
    return 0
  fi
  if [[ ! -f "${TEST_SCRIPT}" ]]; then
    echo "[post-train] test script not found: ${TEST_SCRIPT}" >&2
    return 2
  fi
  if [[ ! -f "${ckpt_path}" ]]; then
    echo "[post-train] checkpoint not found for eval: ${ckpt_path}" >&2
    return 2
  fi

  if [[ -n "${TEST_EVAL_SCRIPT}" ]]; then
    env_args+=("EVAL_SCRIPT=${TEST_EVAL_SCRIPT}")
  fi
  if [[ -n "${TEST_DET_DATA_ROOT}" ]]; then
    env_args+=("DET_DATA_ROOT=${TEST_DET_DATA_ROOT}")
  fi
  if [[ -n "${TEST_DET_ANN_FILE}" ]]; then
    env_args+=("DET_ANN_FILE=${TEST_DET_ANN_FILE}")
  fi
  if [[ -n "${TEST_DET_IMG_DIR}" ]]; then
    env_args+=("DET_IMG_DIR=${TEST_DET_IMG_DIR}")
  fi
  if [[ -n "${TEST_SEG_DATA_DIR}" ]]; then
    env_args+=("SEG_DATA_DIR=${TEST_SEG_DATA_DIR}")
  fi
  if [[ -n "${TEST_CNT_DATA_ROOT}" ]]; then
    env_args+=("CNT_DATA_ROOT=${TEST_CNT_DATA_ROOT}")
  fi
  if [[ -n "${TEST_CNT_TEST_DIR}" ]]; then
    env_args+=("CNT_TEST_DIR=${TEST_CNT_TEST_DIR}")
  fi

  if [[ "${TEST_PRECHECK}" == "1" ]]; then
    echo "[post-train] precheck eval load for ${ckpt_path}"
    env "${env_args[@]}" CHECK_FULL_LOAD_ONLY=1 bash "${TEST_SCRIPT}" "${ckpt_path}"
  fi

  echo "[post-train] run full eval for ${ckpt_path}"
  env "${env_args[@]}" CHECK_FULL_LOAD_ONLY=0 bash "${TEST_SCRIPT}" "${ckpt_path}"
}

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

echo "[run] ${RUN_CMD[*]}"
if run_training; then
  echo "[post-train] eval stage1 best checkpoint: ${TEST_STAGE1_CKPT}"
  run_post_train_eval "${TEST_STAGE1_CKPT}"
  if [[ "${TEST_STAGE2_CKPT}" != "${TEST_STAGE1_CKPT}" ]]; then
    echo "[post-train] eval stage2 best checkpoint: ${TEST_STAGE2_CKPT}"
    run_post_train_eval "${TEST_STAGE2_CKPT}"
  else
    echo "[post-train] skip duplicate stage2 eval (same as stage1): ${TEST_STAGE2_CKPT}"
  fi
else
  train_rc=$?
  echo "[run] training failed (rc=${train_rc}), skip eval" >&2
  exit "${train_rc}"
fi
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

TRAIN_DIR="/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/train_5500_10per"
VAL_DIR="/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/val"
BACKBONE_CKPT="/nas/liyangguang103/old_fscd/CD-FSOD/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

NUM_CLASSES=11
EPOCHS=100
BATCH_SIZE=16
LR=1e-4
WD=1e-4
IMG_SIZE=448
MODEL_NAME="dinov3_vitl16"
NUM_WORKERS=4

# 使用 GPU_ID 隔离，程序内部用 cuda:0
GPU_ID=0
DEVICE="cuda:0"

RUN_DIR="runs/dinov3_seg_freeze_all$(date +%Y%m%d_%H%M%S)"
SAVE_PATH="${RUN_DIR}/122_freeze_ckpt.pt"
LOG_FILE="${RUN_DIR}/122_freeze_train.log"
mkdir -p "${RUN_DIR}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python3 -u train.py \
  --train-dir "${TRAIN_DIR}" \
  --val-dir "${VAL_DIR}" \
  --num-classes "${NUM_CLASSES}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --weight-decay "${WD}" \
  --image-size "${IMG_SIZE}" \
  --model-name "${MODEL_NAME}" \
  --backbone-checkpoint "${BACKBONE_CKPT}" \
  --num-workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --amp \
  --save-path "${SAVE_PATH}" \
  --log-file "${LOG_FILE}" \
  2>&1 | tee "${RUN_DIR}/console.log"

echo "Full finetune done. Outputs in ${RUN_DIR}"

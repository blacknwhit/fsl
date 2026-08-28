#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=2
cd "$(dirname "$0")"

# ===== DIOR COCO paths =====
DATA_ROOT="/nas/liyangguang103/newdataset/CD-ObjectDetection/DIOR/coco"
TRAIN_ANN="/nas/liyangguang103/newdataset/CD-ObjectDetection/DIOR/coco/annotations/instances_train_10per.json"
VAL_ANN="/nas/liyangguang103/newdataset/CD-ObjectDetection/DIOR/coco/annotations/instances_val.json"
TRAIN_IMG_DIR="/nas/liyangguang103/newdataset/CD-ObjectDetection/DIOR/coco/images/train"
VAL_IMG_DIR="/nas/liyangguang103/newdataset/CD-ObjectDetection/DIOR/coco/images/val"

# ===== Backbone =====
BACKBONE_CKPT="/nas/liyangguang103/old_fscd/CD-FSOD/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
MODEL_NAME="dinov3_vitl16"
IMG_SIZE=448

# ===== Train hyperparams =====
EPOCHS=200
BATCH_SIZE=16
LR=1e-4
WD=1e-4
NUM_WORKERS=4

# 如果不确定类别数，留空让 train.py 自动从数据集读取
# NUM_CLASSES=20
NUM_CLASSES=""

# ===== Device =====
GPU_ID=1
DEVICE="cuda:0"

RUN_DIR="runs/126_unfreeze_$(date +%Y%m%d_%H%M%S)"
SAVE_PATH="${RUN_DIR}/200_checkpoint.pt"
LOG_FILE="${RUN_DIR}/train.log"
mkdir -p "${RUN_DIR}"

CMD=(python3 train.py
  --data-root "${DATA_ROOT}"
  --train-ann "${TRAIN_ANN}"
  --val-ann "${VAL_ANN}"
  --train-img-dir "${TRAIN_IMG_DIR}"
  --val-img-dir "${VAL_IMG_DIR}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --weight-decay "${WD}"
  --image-size "${IMG_SIZE}"
  --model-name "${MODEL_NAME}"
  --backbone-checkpoint "${BACKBONE_CKPT}"
  --num-workers "${NUM_WORKERS}"
  --device "${DEVICE}"
  --amp
  --save-path "${SAVE_PATH}"
  --log-file "${LOG_FILE}"
  --unfreeze-backbone
)

if [[ -n "${NUM_CLASSES}" ]]; then
  CMD+=(--num-classes "${NUM_CLASSES}")
fi

# 如需训练 backbone，取消下一行注释
# CMD+=(--unfreeze-backbone)

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}" 2>&1 | tee "${RUN_DIR}/console.log"

echo "Training done. Outputs in ${RUN_DIR}"

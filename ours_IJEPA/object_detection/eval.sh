#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

CUDA_VISIBLE_DEVICES=0 python3 eval.py \
  --checkpoint  /nas/liyangguang103/new_fscd/object_detection/runs/126_unfreeze_20260126_231724/best_ap50.pt \
  --data-root /nas/liyangguang103/newdataset/CD-ObjectDetection/DIOR/coco \
  --ann-file /nas/liyangguang103/newdataset/CD-ObjectDetection/DIOR/coco/annotations/instances_test.json \
  --img-dir /nas/liyangguang103/newdataset/CD-ObjectDetection/DIOR/coco/images/test \
  --model-name dinov3_vitl16 \
  --image-size 448 \
  --score-thr 0.0 \
  --device cuda \
  --use-coco-eval \
  --backbone-source auto \
  --backbone-checkpoint /nas/liyangguang103/old_fscd/CD-FSOD/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
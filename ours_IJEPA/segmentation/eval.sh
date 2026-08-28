#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES="0"
cd "$(dirname "$0")"

python3 eval.py \
  --checkpoint /nas/liyangguang103/new_fscd/segmentation/runs/dinov3_seg_full_all20260122_002720/best_miou.pt \
  --data-dir /nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/test \
  --num-classes 11 \
  --image-size 448 \
  --model-name dinov3_vitl16 \
  --device cuda:0 \

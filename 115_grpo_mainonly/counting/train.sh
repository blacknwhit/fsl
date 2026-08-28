#!/usr/bin/env bash
set -euo pipefail


export CUDA_VISIBLE_DEVICES=0
cd "$(dirname "$0")"

python train.py \
	--data-root /nas/liyangguang103/newdataset/CD-Count/DSACA \
	--train-dir /nas/liyangguang103/newdataset/CD-Count/DSACA/train_data_class8_10per \
	--num-classes 8 \
	--epochs 100 \
	--batch-size 16 \
	--lr 1e-4 \
	--weight-decay 1e-4 \
	--count-loss-weight 1 \
	--image-size 448 \
	--keep-aspect \
	--model-name dinov3_vitl16 \
	--backbone-checkpoint /nas/liyangguang103/old_fscd/CD-FSOD/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
	--save-path runs/121_10perfreeze.pt \
	--log-file runs/121_10perfreeze.log \
	--amp \
	--num-workers 4 \
	--log-interval 50 \
	--grad-clip-norm 0.1 \
	--device cuda:0 \


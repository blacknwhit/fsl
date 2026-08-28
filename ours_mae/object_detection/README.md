# DINOv3 ViT-L/16 Object Detection Baseline

仿照 `segmentation/` 的最小示例，实现一个目标检测 baseline：

- Backbone：`dinov3_vitl16`
- Head：torchvision `Faster R-CNN`（不依赖 mmdet/detectron2 等外部检测框架）
- 默认冻结 backbone，只训练 1x1 投影层 + RPN/ROI head

## 环境依赖
- Python 3.9+
- PyTorch（建议 CUDA）
- torchvision
- pillow
- 本地可用的 `facebookresearch/dinov3`（通过 `dinov3.hub.backbones` 导入）

## 数据组织（DIOR COCO）
你给的数据根目录是：
`/nas/liyangguang103/newdataset/CD-ObjectDetection/DIOR/coco`

需要满足：
```
coco/
  annotations/
    instances_train.json
    instances_val.json
    instances_test.json
  images/
    train/
    val/
    test/
```

## 训练
```bash
python train.py \
  --data-root /nas/liyangguang103/newdataset/CD-ObjectDetection/DIOR/coco \
  --image-size 448 \
  --model-name dinov3_vitl16 \
  --backbone-checkpoint /nas/liyangguang103/old_fscd/CD-FSOD/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --batch-size 2 \
  --epochs 12 \
  --lr 1e-4 \
  --save-path runs/dinov3_det.pt
```

如果想微调 backbone，加入 `--unfreeze-backbone`。

## 评估
```bash
python eval.py \
  --data-root /nas/liyangguang103/newdataset/CD-ObjectDetection/DIOR/coco \
  --checkpoint runs/dinov3_det.pt \
  --image-size 448 \
  --model-name dinov3_vitl16 \
  --backbone-checkpoint /nas/liyangguang103/old_fscd/CD-FSOD/models/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --score-thr 0.05 \
  --use-coco-eval
```

说明：
- 若环境中安装了 `pycocotools`，`--use-coco-eval` 会输出标准 COCO mAP。
- 否则会退化为一个简单的 `Precision/Recall@IoU=0.5` 统计（用于 sanity check）。
- 可以用 `--save-json outputs/preds.json` 导出 COCO 格式预测，方便你用官方脚本算 mAP。


from dataset import SegmentationDataset
import torch

train_ds = SegmentationDataset(
    "/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/train",
    image_size=448,
)

print(f"Dataset length: {len(train_ds)}")
print("Loading first sample...")
img, mask = train_ds[0]
print(f"Image: {img.shape}, dtype: {img.dtype}")
print(f"Mask: {mask.shape}, dtype: {mask.dtype}, min: {mask.min()}, max: {mask.max()}")
print("Success!")
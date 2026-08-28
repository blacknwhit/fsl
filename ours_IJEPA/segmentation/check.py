import numpy as np
from PIL import Image
import glob

# 检查mydataset路径
for split in ['train', 'val', 'test']:
    msk_dir = f"/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/{split}/masks"
    masks = glob.glob(f"{msk_dir}/*.png")
    
    if not masks:
        # 尝试其他扩展名
        for ext in ['*.tif', '*.tiff', '*.jpg']:
            masks.extend(glob.glob(f"{msk_dir}/{ext}"))
    
    print(f"\n{split}: {len(masks)} masks")
    
    if len(masks) == 0:
        print(f"  ⚠️ No masks found in {msk_dir}")
        continue
    
    all_vals = set()
    masks_with_255 = 0
    
    for m in masks:
        arr = np.array(Image.open(m))
        all_vals.update(np.unique(arr).tolist())
        if 255 in arr:
            masks_with_255 += 1
    
    print(f"  Values: {sorted(all_vals)}")
    if 255 in all_vals:
        print(f"  ⚠️ Contains 255! ({masks_with_255}/{len(masks)} masks)")
    else:
        print(f"  ✓ No 255 values")
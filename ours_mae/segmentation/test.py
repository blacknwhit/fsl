import os
import numpy as np
from PIL import Image
import glob

# 定义mydataset目录路径
MYDATASET_ROOT = "/nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset"


def find_and_remove_masks_with_255(img_dir, msk_dir, split_name):
    """查找并删除包含255的mask及其对应的图像"""
    print(f"\n处理 {split_name} 数据集...")
    print(f"图像目录: {img_dir}")
    print(f"Mask目录: {msk_dir}")
    
    # 获取所有mask文件
    mask_paths = []
    for ext in ("*.png", "*.tif", "*.tiff"):
        mask_paths.extend(glob.glob(os.path.join(msk_dir, ext)))
    mask_paths = sorted(mask_paths)
    
    print(f"总mask数量: {len(mask_paths)}")
    
    if len(mask_paths) == 0:
        print("  ⚠️ 没有找到mask文件,跳过")
        return
    
    to_delete = []
    total_255_pixels = 0
    
    # 检查每个mask
    for msk_path in mask_paths:
        m = np.array(Image.open(msk_path))
        if 11 in m:
            to_delete.append(msk_path)
            total_255_pixels += np.sum(m == 11)
    
    print(f"包含11的mask数量: {len(to_delete)}")
    if len(to_delete) > 0:
        print(f"占比: {len(to_delete)/len(mask_paths)*100:.2f}%")
        print(f"总共11像素数: {total_255_pixels}")
    else:
        print("  ✓ 没有需要删除的文件")
        return
    
    # 确认删除
    print(f"\n将删除 {len(to_delete)} 对图像-mask对")
    response = input("确认删除? (yes/no): ")
    
    if response.lower() != 'yes':
        print("取消删除操作")
        return
    
    # 删除mask和对应的图像
    deleted_imgs = 0
    deleted_msks = 0
    
    for msk_path in to_delete:
        # 删除mask
        if os.path.exists(msk_path):
            os.remove(msk_path)
            deleted_msks += 1
            print(f"  删除mask: {os.path.basename(msk_path)}")
        
        # 查找并删除对应的图像
        msk_name = os.path.splitext(os.path.basename(msk_path))[0]
        img_path = None
        
        for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            candidate = os.path.join(img_dir, msk_name + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break
        
        if img_path:
            os.remove(img_path)
            deleted_imgs += 1
            print(f"  删除图像: {os.path.basename(img_path)}")
        else:
            print(f"  警告: 找不到 {msk_name} 对应的图像")
    
    print(f"\n{split_name} 删除完成:")
    print(f"  已删除图像: {deleted_imgs}")
    print(f"  已删除mask: {deleted_msks}")


# 处理train、val、test数据集
for split in ['train', 'val', 'test']:
    img_dir = os.path.join(MYDATASET_ROOT, split, "images")
    msk_dir = os.path.join(MYDATASET_ROOT, split, "masks")
    
    if not os.path.exists(msk_dir):
        print(f"\n{split}: 目录不存在,跳过")
        continue
    
    find_and_remove_masks_with_255(img_dir, msk_dir, split)

print("\n全部完成!")
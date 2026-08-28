import h5py
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# 定义需要处理的根目录
data_roots = [
    '/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/nas/liyangguang103/newdataset/CD-Count/DSACA/train_data_class8',
]

def compress_h5_file(file_path, output_dir):
    # 先读取数据到内存
    with h5py.File(file_path, 'r') as f:
        # 假设之前的 key 是 density_map 和 mask
        if 'density_map' not in f or 'mask' not in f:
            return
        dset_map = f['density_map']
        dset_mask = f['mask']
        d_map = dset_map[:]
        m_mask = dset_mask[:]
        map_dtype = dset_map.dtype
        mask_dtype = dset_mask.dtype
        map_attrs = dict(dset_map.attrs)
        mask_attrs = dict(dset_mask.attrs)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(file_path).name

    # 写入到同级新目录（不覆盖原文件）
    with h5py.File(output_path, 'w') as f:
        # 转换并压缩
        dset_map_new = f.create_dataset(
            'density_map',
            data=d_map.astype(map_dtype),
            compression="gzip",
            compression_opts=4,
        )
        dset_mask_new = f.create_dataset(
            'mask',
            data=m_mask.astype(mask_dtype),
            compression="gzip",
            compression_opts=4,
        )
        for k, v in map_attrs.items():
            dset_map_new.attrs[k] = v
        for k, v in mask_attrs.items():
            dset_mask_new.attrs[k] = v

    # 验证：仅检查 density_map 是否完全一致
    with h5py.File(output_path, 'r') as f:
        if not np.array_equal(f['density_map'][:], d_map):
            raise ValueError(f"density_map mismatch after compression: {output_path}")

if __name__ == '__main__':
    for root in data_roots:
        map_dir = Path(root) / 'gt_density_map'
        out_dir = Path(root) / 'gt_density_map_compressed'
        if not map_dir.exists():
            print(f"Skipping {map_dir}, directory not found.")
            continue
            
        print(f"Processing directory: {map_dir}")
        h5_files = list(map_dir.glob('*.h5'))
        
        for h5_file in tqdm(h5_files):
            compress_h5_file(str(h5_file), out_dir)

    print("All files compressed successfully!")
import os
import shutil
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# 参数设置
NUM_CLASSES = 11    # 0-10 共11类
DEFAULT_SEED = 2025         # 固定随机种子，保证科研可复现性
DEFAULT_ROOT_DIR = (
    "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/"
    "nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset"
)
DEFAULT_TRAIN_DIR_NAME = "train_5500"
DEFAULT_INPUT_SUBSET_DIR = (
    "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/"
    "nas/liyangguang103/newdataset/CD-SematicSeg/EvLab-SS/mydataset/train_5500_10per"
)
DEFAULT_OUTPUT_SPLIT_DIR = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/seg/train_10per"
DEFAULT_RATIOS = "0.01,0.05,0.2,0.3"

# 允许的图片扩展名 (防止jpg/png混用)
IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def get_image_pixel_counts(mask_path):
    """读取Mask并计算每类的像素数"""
    try:
        # 确保以单通道读取
        mask = Image.open(mask_path)
        mask_arr = np.array(mask)
        
        # 统计该图中每个值的像素个数
        # minlength确保即使图中只有部分类，也能输出长度为NUM_CLASSES的数组
        counts = np.bincount(mask_arr.flatten(), minlength=NUM_CLASSES)
        
        # 只取前 NUM_CLASSES 个类（忽略可能的非法值或255背景）
        return counts[:NUM_CLASSES]
    except Exception as e:
        print(f"Error reading {mask_path}: {e}")
        return np.zeros(NUM_CLASSES, dtype=int)

def find_corresponding_image(mask_name, image_dir):
    """根据mask文件名查找对应的原图文件（处理可能的后缀不同）"""
    mask_stem = os.path.splitext(mask_name)[0]
    
    # 1. 尝试直接同名
    if os.path.exists(os.path.join(image_dir, mask_name)):
        return mask_name
        
    # 2. 尝试常见后缀
    for ext in IMG_EXTENSIONS:
        img_name = mask_stem + ext
        if os.path.exists(os.path.join(image_dir, img_name)):
            return img_name
            
    return None

def parse_ratios(ratios_text):
    ratios = []
    for token in ratios_text.split(','):
        token = token.strip()
        if not token:
            continue
        ratio = float(token)
        if ratio <= 0 or ratio > 1:
            raise ValueError(f"Invalid ratio {ratio}, expected 0 < ratio <= 1")
        ratios.append(ratio)
    if not ratios:
        raise ValueError("No valid ratios provided")
    return ratios


def ratio_tag(ratio):
    return int(round(ratio * 100))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate segmentation subsets at multiple ratios.")
    parser.add_argument("--root-dir", default=DEFAULT_ROOT_DIR, help="Path to segmentation dataset root")
    parser.add_argument("--train-dir-name", default=DEFAULT_TRAIN_DIR_NAME, help="Train split directory name")
    parser.add_argument("--ratios", default=DEFAULT_RATIOS, help="Comma-separated ratios, e.g. 0.01,0.05,0.2,0.3")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip generating a ratio if target output directory already has files",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Do not skip existing output directories",
    )
    parser.add_argument(
        "--split-train-valid",
        action="store_true",
        help="Split an existing subset directory into train/valid with no overlap",
    )
    parser.add_argument(
        "--input-subset-dir",
        default=DEFAULT_INPUT_SUBSET_DIR,
        help="Path to an existing subset dir containing images and masks",
    )
    parser.add_argument(
        "--output-split-dir",
        default=DEFAULT_OUTPUT_SPLIT_DIR,
        help="Output directory that will contain train/ and valid/",
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.2,
        help="Validation split ratio for --split-train-valid mode",
    )
    return parser.parse_args()


def split_existing_subset_train_valid(input_subset_dir, output_split_dir, valid_ratio, seed):
    if valid_ratio <= 0 or valid_ratio >= 1:
        raise ValueError("valid_ratio must satisfy 0 < valid_ratio < 1")

    in_mask_dir = os.path.join(input_subset_dir, "masks")
    in_img_dir = os.path.join(input_subset_dir, "images")

    mask_files = sorted(Path(in_mask_dir).glob("*.*"))
    mask_files = [f for f in mask_files if f.suffix.lower() in IMG_EXTENSIONS]
    if not mask_files:
        raise RuntimeError(f"No mask files found in {in_mask_dir}")

    mask_names = [p.name for p in mask_files]
    rng = random.Random(seed)
    rng.shuffle(mask_names)

    valid_num = int(len(mask_names) * valid_ratio)
    valid_masks = set(mask_names[:valid_num])
    train_masks = mask_names[valid_num:]

    train_img_dir = os.path.join(output_split_dir, "train", "images")
    train_mask_dir = os.path.join(output_split_dir, "train", "masks")
    valid_img_dir = os.path.join(output_split_dir, "valid", "images")
    valid_mask_dir = os.path.join(output_split_dir, "valid", "masks")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_mask_dir, exist_ok=True)
    os.makedirs(valid_img_dir, exist_ok=True)
    os.makedirs(valid_mask_dir, exist_ok=True)

    for mask_name in tqdm(mask_names, desc="Copying train/valid split"):
        dst_mask_dir = valid_mask_dir if mask_name in valid_masks else train_mask_dir
        dst_img_dir = valid_img_dir if mask_name in valid_masks else train_img_dir

        src_mask = os.path.join(in_mask_dir, mask_name)
        shutil.copy2(src_mask, os.path.join(dst_mask_dir, mask_name))

        img_name = find_corresponding_image(mask_name, in_img_dir)
        if img_name:
            src_img = os.path.join(in_img_dir, img_name)
            shutil.copy2(src_img, os.path.join(dst_img_dir, img_name))

    print("\n--- Train/Valid Split Report (Seg) ---")
    print(f"Input subset: {input_subset_dir}")
    print(f"Output dir  : {output_split_dir}")
    print(f"Total       : {len(mask_names)}")
    print(f"Train       : {len(train_masks)}")
    print(f"Valid       : {len(valid_masks)}")
    print(f"Overlap     : {len(set(train_masks).intersection(valid_masks))}")


def build_stats(mask_dir):
    file_stats = {}
    total_pixel_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    mask_files = list(Path(mask_dir).glob("*.*"))
    mask_files = [f for f in mask_files if f.suffix.lower() in IMG_EXTENSIONS]
    if len(mask_files) == 0:
        print("Error: No mask files found! Please check the path.")
        return None, None, None

    for mask_path in tqdm(mask_files, desc="Scanning masks"):
        counts = get_image_pixel_counts(mask_path)
        file_stats[mask_path.name] = counts
        total_pixel_counts += counts

    return file_stats, total_pixel_counts, mask_files


def sample_files(file_stats, total_pixel_counts, ratio, seed):
    target_pixel_counts = total_pixel_counts * ratio
    existing_classes = [c for c in range(NUM_CLASSES) if total_pixel_counts[c] > 0]
    sorted_classes = sorted(existing_classes, key=lambda c: total_pixel_counts[c])

    class_to_files = defaultdict(list)
    for fname, counts in file_stats.items():
        for c in existing_classes:
            if counts[c] > 0:
                class_to_files[c].append(fname)

    rng = random.Random(seed + ratio_tag(ratio))
    selected_files = set()
    current_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for cls_id in sorted_classes:
        target = target_pixel_counts[cls_id]
        if current_counts[cls_id] < target:
            candidates = class_to_files[cls_id][:]
            rng.shuffle(candidates)
            for fname in candidates:
                if fname in selected_files:
                    continue
                selected_files.add(fname)
                current_counts += file_stats[fname]
                if current_counts[cls_id] >= target:
                    break

    return selected_files, current_counts, target_pixel_counts, sorted_classes


def copy_subset(selected_files, mask_dir, image_dir, output_dir, ratio):
    out_img_dir = os.path.join(output_dir, "images")
    out_mask_dir = os.path.join(output_dir, "masks")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    missing_images = []
    for mask_name in tqdm(selected_files, desc=f"Copying {ratio_tag(ratio)}%"):
        src_mask = os.path.join(mask_dir, mask_name)
        dst_mask = os.path.join(out_mask_dir, mask_name)
        shutil.copy2(src_mask, dst_mask)

        img_name = find_corresponding_image(mask_name, image_dir)
        if img_name:
            src_img = os.path.join(image_dir, img_name)
            dst_img = os.path.join(out_img_dir, img_name)
            shutil.copy2(src_img, dst_img)
        else:
            missing_images.append(mask_name)

    return missing_images


def main():
    args = parse_args()
    if args.split_train_valid:
        split_existing_subset_train_valid(
            args.input_subset_dir,
            args.output_split_dir,
            args.valid_ratio,
            args.seed,
        )
        return

    root_dir = args.root_dir
    train_dir = os.path.join(root_dir, args.train_dir_name)
    mask_dir = os.path.join(train_dir, "masks")
    image_dir = os.path.join(train_dir, "images")
    ratios = parse_ratios(args.ratios)

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Source: {train_dir}")
    print(f"Ratios: {ratios}")
    print("Step 1: Scanning dataset and calculating class statistics...")

    file_stats, total_pixel_counts, mask_files = build_stats(mask_dir)
    if file_stats is None:
        return

    print("\nGlobal Pixel Counts per Class:")
    for i, count in enumerate(total_pixel_counts):
        print(f"Class {i}: {count}")

    for ratio in ratios:
        tag = ratio_tag(ratio)
        output_dir = os.path.join(root_dir, f"{args.train_dir_name}_{tag}per")
        if args.skip_existing and os.path.isdir(output_dir) and os.listdir(output_dir):
            print(f"\nSkip {tag}% because output already exists: {output_dir}")
            continue

        print(f"\nStep 2: Selecting images for {tag}% (Greedy Strategy)...")
        selected_files, current_counts, target_pixel_counts, sorted_classes = sample_files(
            file_stats,
            total_pixel_counts,
            ratio,
            args.seed,
        )

        print(f"Selection complete. Selected {len(selected_files)} / {len(mask_files)} images.")
        print(f"Step 3: Copying files to {output_dir}...")
        missing_images = copy_subset(selected_files, mask_dir, image_dir, output_dir, ratio)

        print("\n--- Final Statistics Check (Subset vs Target) ---")
        print(f"{'Class':<6} {'Target':<12} {'Actual':<12} {'Ratio':<8}")
        for c in sorted_classes:
            tgt = int(target_pixel_counts[c])
            act = int(current_counts[c])
            full = int(total_pixel_counts[c])
            real_ratio = act / full if full > 0 else 0
            print(f"{c:<6} {tgt:<12} {act:<12} {real_ratio:.2%}")

        print(f"Target Ratio was: {ratio:.2%}")
        print(f"Final Image Count: {len(selected_files)} (from {len(mask_files)} images)")
        print(f"Dataset created at: {output_dir}")
        if missing_images:
            print(f"Warning: {len(missing_images)} masks had no corresponding images found.")
            print("First 5 missing:", missing_images[:5])

if __name__ == "__main__":
    main()
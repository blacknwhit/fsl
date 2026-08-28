import os
import h5py
import shutil
import random
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# 类别定义 (对应H5中的8个通道)
CLASS_NAMES = ['People', 'Bicycle', 'Car', 'Van', 'Truck', 'Tricycle', 'Bus', 'Motor']
NUM_CLASSES = 8

DEFAULT_ROOT_DIR = (
    "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/"
    "nas/liyangguang103/newdataset/CD-Count/DSACA/train_data_class8"
)
DEFAULT_INPUT_SUBSET_DIR = (
    "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/"
    "nas/liyangguang103/newdataset/CD-Count/DSACA/train_data_class8_10per"
)
DEFAULT_OUTPUT_SPLIT_DIR = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/cnt/train_10per"
DEFAULT_RATIOS = "0.01,0.05,0.2,0.3"
DEFAULT_SEED = 2025

def get_counts_from_h5(h5_path):
    """从H5文件中读取8个类别的总数"""
    try:
        with h5py.File(h5_path, 'r') as f:
            # 这里的 'density_map' shape 是 (8, H, W)
            dmap = f['density_map'][()]
            # 对每个通道求和，得到该类别的对象计数
            counts = np.sum(dmap, axis=(1, 2))
            return counts
    except Exception as e:
        print(f"Error reading {h5_path}: {e}")
        return np.zeros(NUM_CLASSES)

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
    parser = argparse.ArgumentParser(description="Generate count subsets at multiple ratios.")
    parser.add_argument("--root-dir", default=DEFAULT_ROOT_DIR, help="Path to full count dataset directory")
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
        help="Path to an existing subset dir containing images and gt_density_map_compressed",
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

    in_img_dir = os.path.join(input_subset_dir, "images")
    in_map_dir = os.path.join(input_subset_dir, "gt_density_map_compressed")

    h5_files = sorted(Path(in_map_dir).glob("*.h5"))
    if not h5_files:
        raise RuntimeError(f"No .h5 files found in {in_map_dir}")

    stems = [p.stem for p in h5_files]
    rng = random.Random(seed)
    rng.shuffle(stems)

    valid_num = int(len(stems) * valid_ratio)
    valid_stems = set(stems[:valid_num])
    train_stems = stems[valid_num:]

    train_img_dir = os.path.join(output_split_dir, "train", "images")
    train_map_dir = os.path.join(output_split_dir, "train", "gt_density_map_compressed")
    valid_img_dir = os.path.join(output_split_dir, "valid", "images")
    valid_map_dir = os.path.join(output_split_dir, "valid", "gt_density_map_compressed")

    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_map_dir, exist_ok=True)
    os.makedirs(valid_img_dir, exist_ok=True)
    os.makedirs(valid_map_dir, exist_ok=True)

    for stem in tqdm(stems, desc="Copying train/valid split"):
        dst_map_dir = valid_map_dir if stem in valid_stems else train_map_dir
        dst_img_dir = valid_img_dir if stem in valid_stems else train_img_dir

        src_h5 = os.path.join(in_map_dir, stem + ".h5")
        shutil.copy2(src_h5, os.path.join(dst_map_dir, stem + ".h5"))

        src_img = os.path.join(in_img_dir, stem + ".jpg")
        if os.path.exists(src_img):
            shutil.copy2(src_img, os.path.join(dst_img_dir, stem + ".jpg"))

    print("\n--- Train/Valid Split Report (Count) ---")
    print(f"Input subset: {input_subset_dir}")
    print(f"Output dir  : {output_split_dir}")
    print(f"Total       : {len(stems)}")
    print(f"Train       : {len(train_stems)}")
    print(f"Valid       : {len(valid_stems)}")
    print(f"Overlap     : {len(set(train_stems).intersection(valid_stems))}")


def generate_subset_for_ratio(root_dir, ratio, seed, file_stats, total_counts, h5_count):
    img_dir = os.path.join(root_dir, "images")
    map_dir = os.path.join(root_dir, "gt_density_map_compressed")
    show_dir = os.path.join(root_dir, "gt_show")

    tag = ratio_tag(ratio)
    output_dir = os.path.join(os.path.dirname(root_dir), f"{Path(root_dir).name}_{tag}per")

    print(f"\n========== Ratio {tag}% ==========")
    print(f"Target: {output_dir}")

    target_counts = total_counts * ratio

    existing_indices = [i for i in range(NUM_CLASSES) if total_counts[i] > 0]
    sorted_indices = sorted(existing_indices, key=lambda i: total_counts[i])
    print(f"Processing Order (Rarest First): {[CLASS_NAMES[i] for i in sorted_indices]}")

    rng = random.Random(seed + tag)
    selected_stems = set()
    current_counts = np.zeros(NUM_CLASSES)

    class_to_stems = defaultdict(list)
    for stem, counts in file_stats.items():
        for i in existing_indices:
            if counts[i] > 0:
                class_to_stems[i].append(stem)

    for cls_idx in sorted_indices:
        target = target_counts[cls_idx]
        if current_counts[cls_idx] < target:
            candidates = class_to_stems[cls_idx][:]
            rng.shuffle(candidates)

            for stem in candidates:
                if stem in selected_stems:
                    continue
                selected_stems.add(stem)
                current_counts += file_stats[stem]
                if current_counts[cls_idx] >= target:
                    break

    print(f"Selected {len(selected_stems)} / {h5_count} images.")
    print("Step 3: Copying files...")

    out_img_dir = os.path.join(output_dir, "images")
    out_map_dir = os.path.join(output_dir, "gt_density_map_compressed")
    out_show_dir = os.path.join(output_dir, "gt_show")

    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_map_dir, exist_ok=True)
    os.makedirs(out_show_dir, exist_ok=True)

    for stem in tqdm(selected_stems, desc=f"Copying {tag}%"):
        src_h5 = os.path.join(map_dir, stem + ".h5")
        shutil.copy2(src_h5, os.path.join(out_map_dir, stem + ".h5"))

        src_img = os.path.join(img_dir, stem + ".jpg")
        if os.path.exists(src_img):
            shutil.copy2(src_img, os.path.join(out_img_dir, stem + ".jpg"))
        else:
            print(f"Warning: Image not found for {stem}")

        src_show_subdir = os.path.join(show_dir, stem)
        if os.path.exists(src_show_subdir):
            dst_show_subdir = os.path.join(out_show_dir, stem)
            if os.path.exists(dst_show_subdir):
                shutil.rmtree(dst_show_subdir)
            shutil.copytree(src_show_subdir, dst_show_subdir)

    print("\n--- Final Counts Report ---")
    print(f"{'Class':<10} {'Target':<10} {'Actual':<10} {'Ratio':<8}")
    for i in sorted_indices:
        tgt = int(target_counts[i])
        act = int(current_counts[i])
        full = int(total_counts[i])
        final_ratio = act / full if full > 0 else 0
        print(f"{CLASS_NAMES[i]:<10} {tgt:<10} {act:<10} {final_ratio:.2%}")

    print(f"Dataset created at: {output_dir}")


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
    ratios = parse_ratios(args.ratios)

    map_dir = os.path.join(root_dir, "gt_density_map_compressed")
    print(f"Source: {root_dir}")
    print(f"Ratios: {ratios}")

    print("\nStep 1: Scanning H5 files...")
    h5_files = list(Path(map_dir).glob("*.h5"))
    if not h5_files:
        print("Error: No .h5 files found!")
        return

    file_stats = {}
    total_counts = np.zeros(NUM_CLASSES)

    for p in tqdm(h5_files, desc="Indexing"):
        counts = get_counts_from_h5(p)
        file_stats[p.stem] = counts
        total_counts += counts

    print("\nGlobal Object Counts:")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<10}: {int(total_counts[i])}")

    for ratio in ratios:
        tag = ratio_tag(ratio)
        output_dir = os.path.join(os.path.dirname(root_dir), f"{Path(root_dir).name}_{tag}per")
        if args.skip_existing and os.path.isdir(output_dir) and os.listdir(output_dir):
            print(f"\nSkip {tag}% because output already exists: {output_dir}")
            continue
        generate_subset_for_ratio(root_dir, ratio, args.seed, file_stats, total_counts, len(h5_files))

if __name__ == "__main__":
    main()
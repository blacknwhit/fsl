import json
import os
import random
import argparse
from collections import defaultdict
from tqdm import tqdm

DEFAULT_INPUT_JSON = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_train.json"
DEFAULT_INPUT_SUBSET_JSON = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/coco/annotations/instances_train_10per.json"
DEFAULT_OUTPUT_SPLIT_DIR = "/data/xiangyuyue/ULLM-zf/fsl-20260209/data/det/train_10per"
DEFAULT_RATIOS = "0.01,0.05,0.2,0.3"
DEFAULT_SEED = 2025


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
    parser = argparse.ArgumentParser(description="Generate detection JSON subsets at multiple ratios.")
    parser.add_argument("--input-json", default=DEFAULT_INPUT_JSON, help="Path to full train instances json")
    parser.add_argument("--ratios", default=DEFAULT_RATIOS, help="Comma-separated ratios, e.g. 0.01,0.05,0.2,0.3")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip generating a ratio if output JSON already exists",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Do not skip existing output JSON",
    )
    parser.add_argument(
        "--min-one-per-class",
        action="store_true",
        default=True,
        help="Ensure each non-empty class has at least 1 bbox target",
    )
    parser.add_argument(
        "--no-min-one-per-class",
        dest="min_one_per_class",
        action="store_false",
        help="Disable min-1 bbox target for non-empty classes",
    )
    parser.add_argument(
        "--split-train-valid",
        action="store_true",
        help="Split an existing subset json into train/valid with no overlap",
    )
    parser.add_argument(
        "--input-subset-json",
        default=DEFAULT_INPUT_SUBSET_JSON,
        help="Path to existing subset json, e.g. instances_train_10per.json",
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


def split_existing_subset_train_valid(input_subset_json, output_split_dir, valid_ratio, seed):
    if valid_ratio <= 0 or valid_ratio >= 1:
        raise ValueError("valid_ratio must satisfy 0 < valid_ratio < 1")

    print(f"Loading subset JSON from: {input_subset_json}")
    with open(input_subset_json, 'r') as f:
        coco_data = json.load(f)

    images = coco_data.get("images", [])
    annotations = coco_data.get("annotations", [])

    img_ids = [img["id"] for img in images]
    rng = random.Random(seed)
    rng.shuffle(img_ids)

    valid_num = int(len(img_ids) * valid_ratio)
    valid_ids = set(img_ids[:valid_num])
    train_ids = set(img_ids[valid_num:])

    train_images = [img for img in images if img["id"] in train_ids]
    valid_images = [img for img in images if img["id"] in valid_ids]
    train_annotations = [ann for ann in annotations if ann["image_id"] in train_ids]
    valid_annotations = [ann for ann in annotations if ann["image_id"] in valid_ids]

    common_fields = {
        "info": coco_data.get("info", {}),
        "licenses": coco_data.get("licenses", []),
        "categories": coco_data.get("categories", []),
    }

    train_data = {
        **common_fields,
        "images": train_images,
        "annotations": train_annotations,
    }
    valid_data = {
        **common_fields,
        "images": valid_images,
        "annotations": valid_annotations,
    }

    train_dir = os.path.join(output_split_dir, "train")
    valid_dir = os.path.join(output_split_dir, "valid")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)

    train_json = os.path.join(train_dir, "instances_train.json")
    valid_json = os.path.join(valid_dir, "instances_valid.json")
    with open(train_json, 'w') as f:
        json.dump(train_data, f)
    with open(valid_json, 'w') as f:
        json.dump(valid_data, f)

    print("\n--- Train/Valid Split Report (Det) ---")
    print(f"Input subset: {input_subset_json}")
    print(f"Output dir  : {output_split_dir}")
    print(f"Total images: {len(images)}")
    print(f"Train images: {len(train_images)}")
    print(f"Valid images: {len(valid_images)}")
    print(f"Train anns  : {len(train_annotations)}")
    print(f"Valid anns  : {len(valid_annotations)}")
    print(f"Overlap     : {len(train_ids.intersection(valid_ids))}")


def build_indexes(annotations):
    print("Building indexes...")

    img_to_anns = defaultdict(list)
    img_to_cats = defaultdict(set)
    cat_to_imgs = defaultdict(list)
    cat_total_counts = defaultdict(int)

    for ann in tqdm(annotations, desc="Indexing Annotations"):
        img_id = ann['image_id']
        cat_id = ann['category_id']

        img_to_anns[img_id].append(ann)
        img_to_cats[img_id].add(cat_id)
        cat_total_counts[cat_id] += 1

    for img_id, cat_ids in img_to_cats.items():
        for cat_id in cat_ids:
            cat_to_imgs[cat_id].append(img_id)

    return img_to_anns, cat_to_imgs, cat_total_counts


def make_target_counts(valid_cats, cat_total_counts, ratio, min_one_per_class):
    target_counts = {}
    for cid in valid_cats:
        full = cat_total_counts[cid]
        raw_target = int(full * ratio)
        if min_one_per_class and full > 0:
            raw_target = max(1, raw_target)
        target_counts[cid] = raw_target
    return target_counts


def sample_subset(
    images,
    annotations,
    categories,
    img_to_anns,
    cat_to_imgs,
    cat_total_counts,
    ratio,
    seed,
    min_one_per_class,
):
    valid_cats = [c['id'] for c in categories if cat_total_counts[c['id']] > 0]
    sorted_cats = sorted(valid_cats, key=lambda cid: cat_total_counts[cid])
    target_counts = make_target_counts(valid_cats, cat_total_counts, ratio, min_one_per_class)

    tag = ratio_tag(ratio)
    print(f"\nTargeting {tag}% instances for {len(sorted_cats)} categories.")
    print(f"Top 3 Rarest Classes (ID): {sorted_cats[:3]}")

    rng = random.Random(seed + tag)
    selected_img_ids = set()
    current_counts = defaultdict(int)

    print("Starting Greedy Sampling...")
    for cat_id in tqdm(sorted_cats, desc=f"Sampling Categories {tag}%"):
        target = target_counts[cat_id]
        if current_counts[cat_id] < target:
            candidates = cat_to_imgs[cat_id][:]
            rng.shuffle(candidates)

            for img_id in candidates:
                if img_id in selected_img_ids:
                    continue
                selected_img_ids.add(img_id)
                for ann in img_to_anns[img_id]:
                    cid = ann['category_id']
                    current_counts[cid] += 1
                if current_counts[cat_id] >= target:
                    break

    new_data = {
        "info": {},
        "licenses": [],
        "categories": categories,
        "images": [],
        "annotations": [],
    }

    for img in images:
        if img['id'] in selected_img_ids:
            new_data['images'].append(img)

    for ann in annotations:
        if ann['image_id'] in selected_img_ids:
            new_data['annotations'].append(ann)

    return new_data, valid_cats, target_counts, current_counts, selected_img_ids

def main():
    args = parse_args()
    if args.split_train_valid:
        split_existing_subset_train_valid(
            args.input_subset_json,
            args.output_split_dir,
            args.valid_ratio,
            args.seed,
        )
        return

    ratios = parse_ratios(args.ratios)

    print(f"Loading JSON from: {args.input_json}")
    with open(args.input_json, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']

    print(f"Loaded {len(images)} images and {len(annotations)} annotations.")

    img_to_anns, cat_to_imgs, cat_total_counts = build_indexes(annotations)

    output_dir = os.path.dirname(args.input_json)

    for ratio in ratios:
        tag = ratio_tag(ratio)
        output_json = os.path.join(output_dir, f"instances_train_{tag}per.json")
        if args.skip_existing and os.path.exists(output_json):
            print(f"\nSkip {tag}% because output already exists: {output_json}")
            continue

        new_data, valid_cats, target_counts, current_counts, selected_img_ids = sample_subset(
            images,
            annotations,
            categories,
            img_to_anns,
            cat_to_imgs,
            cat_total_counts,
            ratio,
            args.seed,
            args.min_one_per_class,
        )

        new_data["info"] = coco_data.get("info", {})
        new_data["licenses"] = coco_data.get("licenses", [])

        print(f"\nSelection Finished ({tag}%). Selected {len(selected_img_ids)} images.")
        print(f"Saving to {output_json} ...")
        with open(output_json, 'w') as f:
            json.dump(new_data, f)

        print("\nValidation Report:")
        print(f"{'Cat ID':<8} {'Total Box':<12} {'Target':<10} {'Selected':<10} {'Ratio':<8}")
        print("-" * 50)

        total_ratio_accum = 0.0
        count_cats = 0

        for cat in categories:
            cid = cat['id']
            if cid not in valid_cats:
                continue

            full = cat_total_counts[cid]
            tgt = target_counts[cid]
            sel = current_counts[cid]
            sel_ratio = sel / full if full > 0 else 0

            print(f"{cid:<8} {full:<12} {tgt:<10} {sel:<10} {sel_ratio:.2%}")
            total_ratio_accum += sel_ratio
            count_cats += 1

        avg_ratio = total_ratio_accum / count_cats if count_cats > 0 else 0
        print("-" * 50)
        print(f"Average Ratio across all categories: {avg_ratio:.2%}")

if __name__ == "__main__":
    main()
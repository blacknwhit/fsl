import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict, Any

import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class CocoDetectionDataset(Dataset):
    """
    Minimal COCO-format detection dataset.

    Expects:
        ann_file: path to instances_*.json
        img_dir:  directory containing images referenced by file_name

    Returns:
        image: FloatTensor [3,H,W] in [0,1]
        target: dict with keys boxes, labels, image_id, area, iscrowd
    """

    def __init__(
        self,
        ann_file: str,
        img_dir: str,
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.ann_file = Path(ann_file)
        self.img_dir = Path(img_dir)
        self.transform = transform

        with self.ann_file.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        self.images: List[Dict[str, Any]] = coco.get("images", [])
        self.annotations: List[Dict[str, Any]] = coco.get("annotations", [])
        self.categories: List[Dict[str, Any]] = coco.get("categories", [])

        if not self.categories:
            cat_ids = sorted({ann["category_id"] for ann in self.annotations})
            self.categories = [{"id": cid, "name": str(cid)} for cid in cat_ids]

        cat_ids_sorted = sorted(cat["id"] for cat in self.categories)
        self.cat_id_to_label = {cid: idx + 1 for idx, cid in enumerate(cat_ids_sorted)}
        self.label_to_cat_id = {label: cid for cid, label in self.cat_id_to_label.items()}
        self.num_classes = len(cat_ids_sorted)

        self.image_id_to_info = {img["id"]: img for img in self.images}
        anns_by_img: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for ann in self.annotations:
            anns_by_img[ann["image_id"]].append(ann)
        self.anns_by_img = anns_by_img

        self.image_ids = [img["id"] for img in self.images]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        image_id = self.image_ids[idx]
        info = self.image_id_to_info[image_id]
        file_name = info["file_name"]
        file_path = Path(file_name)
        if file_path.is_absolute():
            img_path = file_path
        else:
            # Common COCO variants:
            # 1) file_name = "0001.jpg"            -> img_dir/0001.jpg
            # 2) file_name = "train/0001.jpg"      -> img_dir/train/0001.jpg (if img_dir=images)
            # 3) file_name = "images/train/0001.jpg" -> data_root/images/train/0001.jpg
            cand1 = self.img_dir / file_path
            cand2 = self.img_dir.parent / file_path
            cand3 = self.img_dir.parent.parent / file_path
            if cand1.exists():
                img_path = cand1
            elif cand2.exists():
                img_path = cand2
            elif cand3.exists():
                img_path = cand3
            else:
                # Fall back to cand1 for a clear error message
                img_path = cand1

        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        anns = self.anns_by_img.get(image_id, [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            bbox = ann.get("bbox")
            if bbox is None:
                continue
            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                continue

            x1 = float(x)
            y1 = float(y)
            x2 = float(x + w)
            y2 = float(y + h)

            x1 = max(0.0, min(x1, width))
            y1 = max(0.0, min(y1, height))
            x2 = max(0.0, min(x2, width))
            y2 = max(0.0, min(y2, height))
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(self.cat_id_to_label[ann["category_id"]])
            areas.append(float(ann.get("area", w * h)))
            iscrowd.append(int(ann.get("iscrowd", 0)))

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.int64)
            areas_tensor = torch.tensor(areas, dtype=torch.float32)
            iscrowd_tensor = torch.tensor(iscrowd, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            areas_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": areas_tensor,
            "iscrowd": iscrowd_tensor,
        }

        if self.transform:
            image, target = self.transform(image, target)

        if not isinstance(image, torch.Tensor):
            image = TF.to_tensor(image)

        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

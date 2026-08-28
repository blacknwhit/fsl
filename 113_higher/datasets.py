from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

from object_detection.dataset import CocoDetectionDataset, collate_fn
from segmentation.dataset import SegmentationDataset
from counting.dataset import DSACADensityH5Dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DetTransform:
    def __init__(self, train: bool):
        self.train = bool(train)

    def __call__(self, img, target):
        if self.train and random.random() < 0.5:
            width, _ = img.size
            img = TF.hflip(img)
            boxes = target["boxes"].clone()
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
            target = dict(target)
            target["boxes"] = boxes
        return TF.to_tensor(img), target


class SegTransform:
    def __init__(self, train: bool, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        self.train = bool(train)
        self.mean = mean
        self.std = std

    def __call__(self, img, mask):
        if self.train:
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() < 0.5:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
        img = TF.normalize(TF.to_tensor(img), mean=self.mean, std=self.std)
        mask_tensor = torch.as_tensor(TF.pil_to_tensor(mask), dtype=torch.long).squeeze(0)
        return img, mask_tensor


class CountTransform:
    def __init__(self, train: bool, mean: Tuple[float, float, float], std: Tuple[float, float, float]):
        self.train = bool(train)
        self.mean = mean
        self.std = std

    def __call__(self, img: torch.Tensor, density: torch.Tensor):
        if self.train:
            if random.random() < 0.5:
                img = torch.flip(img, dims=[2])
                density = torch.flip(density, dims=[2])
            if random.random() < 0.5:
                img = torch.flip(img, dims=[1])
                density = torch.flip(density, dims=[1])
        img = TF.normalize(img, mean=self.mean, std=self.std)
        return img, density


def build_det_loaders(
    *,
    data_root: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
    train_ann: str | None = None,
    val_ann: str | None = None,
    train_img_dir: str | None = None,
    val_img_dir: str | None = None,
):
    root = Path(data_root)
    if not (root / "annotations").exists() and (root / "coco" / "annotations").exists():
        root = root / "coco"
    train_ann_p = Path(train_ann) if train_ann else root / "annotations" / "instances_train.json"
    val_ann_p = Path(val_ann) if val_ann else root / "annotations" / "instances_val.json"
    train_img_p = Path(train_img_dir) if train_img_dir else root / "images" / "train"
    val_img_p = Path(val_img_dir) if val_img_dir else root / "images" / "val"

    train_ds = CocoDetectionDataset(str(train_ann_p), str(train_img_p), transform=DetTransform(True))
    val_ds = CocoDetectionDataset(str(val_ann_p), str(val_img_p), transform=DetTransform(False))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return train_ds, val_ds, train_loader, val_loader


def build_seg_loaders(
    *,
    train_dir: str,
    val_dir: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
):
    train_ds = SegmentationDataset(
        train_dir,
        transform=SegTransform(True, IMAGENET_MEAN, IMAGENET_STD),
        image_size=image_size,
    )
    val_ds = SegmentationDataset(
        val_dir,
        transform=SegTransform(False, IMAGENET_MEAN, IMAGENET_STD),
        image_size=image_size,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_ds, val_ds, train_loader, val_loader


def build_cnt_loaders(
    *,
    data_root: str,
    train_dir: str | None,
    val_dir: str | None,
    image_size: int,
    num_classes: int,
    keep_aspect: bool,
    batch_size: int,
    num_workers: int,
):
    root = Path(data_root)
    train_p = Path(train_dir) if train_dir else root / "train_data_class8"
    val_p = Path(val_dir) if val_dir else root / "val_data_class8"

    train_ds = DSACADensityH5Dataset(
        str(train_p),
        num_classes=num_classes,
        transform=CountTransform(True, IMAGENET_MEAN, IMAGENET_STD),
        image_size=image_size,
        keep_aspect=keep_aspect,
    )
    val_ds = DSACADensityH5Dataset(
        str(val_p),
        num_classes=num_classes,
        transform=CountTransform(False, IMAGENET_MEAN, IMAGENET_STD),
        image_size=image_size,
        keep_aspect=keep_aspect,
    )

    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["multiprocessing_context"] = "spawn"
        kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    return train_ds, val_ds, train_loader, val_loader


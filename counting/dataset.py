from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

try:
    import h5py  # type: ignore
except Exception:
    h5py = None


Resampling = getattr(Image, "Resampling", Image)


class DSACADensityH5Dataset(Dataset):
    """
    VisDrone2019 counting dataset preprocessed by DSACA convert.py.

    Expects:
        split_root/
          images/*.jpg|png
          gt_density_map/*.h5  (keys: density_map, mask)

    Each .h5 stores:
        density_map: [8, H, W] float32
        mask:        [8, H, W] (optional, not used by baseline)
    """

    def __init__(
        self,
        split_root: str,
        num_classes: int = 8,
        transform: Optional[Callable] = None,
        image_size: int = 448,
        keep_aspect: bool = True,
        return_mask: bool = False,
    ):
        super().__init__()
        if h5py is None:
            raise ImportError("h5py is required to read DSACA .h5 density maps")

        self.root = Path(split_root)
        self.images_dir = self.root / "images"
        compressed_maps = self.root / "gt_density_map_compressed"
        self.maps_dir = compressed_maps if compressed_maps.exists() else self.root / "gt_density_map"
        self.num_classes = num_classes
        self.transform = transform
        self.image_size = image_size
        self.keep_aspect = keep_aspect
        self.return_mask = return_mask

        image_paths = sorted(
            [p for p in self.images_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )
        density_paths = {p.stem: p for p in self.maps_dir.glob("*.h5")}

        self.samples: List[Tuple[Path, Path]] = []
        for img_path in image_paths:
            dens_path = density_paths.get(img_path.stem)
            if dens_path is not None:
                self.samples.append((img_path, dens_path))

        if not self.samples:
            raise FileNotFoundError(f"No image/h5 pairs found in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_h5(self, path: Path) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        with h5py.File(path, "r") as hf:
            dens = torch.as_tensor(hf["density_map"][()], dtype=torch.float32)
            mask = None
            if self.return_mask and "mask" in hf:
                mask = torch.as_tensor(hf["mask"][()], dtype=torch.float32)

        if dens.ndim != 3:
            raise ValueError(f"density_map must be [C,H,W], got {dens.shape}")
        if dens.shape[0] != self.num_classes:
            raise ValueError(f"Expected {self.num_classes} channels, got {dens.shape[0]} in {path.name}")

        return dens, mask

    def _resize_keep_aspect(
        self, img: Image.Image, dens: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        width, height = img.size
        scale = self.image_size / float(max(height, width))
        new_h = max(1, int(round(height * scale)))
        new_w = max(1, int(round(width * scale)))

        img_resized = img.resize((new_w, new_h), Resampling.BILINEAR)
        img_tensor = TF.to_tensor(img_resized)

        dens_b = dens.unsqueeze(0)
        dens_resized = F.interpolate(dens_b, size=(new_h, new_w), mode="bilinear", align_corners=False)
        area_scale = (height * width) / float(new_h * new_w)
        dens_resized = dens_resized * area_scale
        dens_resized = dens_resized.squeeze(0)

        mask_resized = None
        if mask is not None:
            mask_b = mask.unsqueeze(0)
            mask_resized = F.interpolate(mask_b, size=(new_h, new_w), mode="nearest").squeeze(0)

        # pad to square
        pad_h = self.image_size - new_h
        pad_w = self.image_size - new_w
        if pad_h or pad_w:
            img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h))
            dens_resized = F.pad(dens_resized, (0, pad_w, 0, pad_h))
            if mask_resized is not None:
                mask_resized = F.pad(mask_resized, (0, pad_w, 0, pad_h))

        return img_tensor, dens_resized, mask_resized

    def __getitem__(self, idx: int):
        img_path, dens_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        dens, mask = self._load_h5(dens_path)

        if self.keep_aspect:
            img_tensor, dens_resized, mask_resized = self._resize_keep_aspect(img, dens, mask)
        else:
            img_resized = img.resize((self.image_size, self.image_size), Resampling.BILINEAR)
            img_tensor = TF.to_tensor(img_resized)

            c, h, w = dens.shape
            dens_b = dens.unsqueeze(0)
            dens_resized = F.interpolate(dens_b, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            dens_resized = dens_resized * ((h * w) / float(self.image_size * self.image_size))
            dens_resized = dens_resized.squeeze(0)

            mask_resized = None
            if mask is not None:
                mask_resized = F.interpolate(mask.unsqueeze(0), size=(self.image_size, self.image_size), mode="nearest").squeeze(0)

        if self.transform:
            if self.return_mask:
                img_tensor, dens_resized, mask_resized = self.transform(img_tensor, dens_resized, mask_resized)
            else:
                img_tensor, dens_resized = self.transform(img_tensor, dens_resized)

        if self.return_mask:
            return img_tensor, dens_resized, (mask_resized if mask_resized is not None else torch.zeros_like(dens_resized))

        return img_tensor, dens_resized


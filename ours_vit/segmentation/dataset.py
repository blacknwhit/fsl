from pathlib import Path
from typing import Callable, Optional

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


Resampling = getattr(Image, "Resampling", Image)


class SegmentationDataset(Dataset):
    """
    Expects:
        root/
            images/*.jpg|png
            masks/*.png  (uint8 with class ids)
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        image_size: int = 448,
    ):
        super().__init__()
        self.root = Path(root)
        self.image_paths = sorted((self.root / "images").glob("*"))
        self.mask_paths = sorted((self.root / "masks").glob("*"))
        assert len(self.image_paths) == len(self.mask_paths), "Mismatch images/masks count"
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        # Resize consistently to match ViT input
        img = img.resize((self.image_size, self.image_size), Resampling.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Resampling.NEAREST)

        if self.transform:
            img, mask = self.transform(img, mask)
        else:
            img = TF.to_tensor(img)
            mask = torch.as_tensor(TF.pil_to_tensor(mask), dtype=torch.long).squeeze(0)

        return img, mask

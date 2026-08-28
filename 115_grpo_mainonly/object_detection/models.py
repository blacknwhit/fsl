import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from dinov3.hub import backbones as dino_backbones
except ImportError:
    dino_backbones = None

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


class DinoV3BackboneForDetection(nn.Module):
    """
    Wrap DINOv3 ViT to a spatial feature map backbone for torchvision detectors.

    - Builds local dinov3 backbone (pretrained=False), loads user checkpoint.
    - Optionally freezes backbone.
    - Converts patch tokens to a 2D feature map and applies a 1x1 projection.
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
        image_size: int = 448,
        checkpoint_path: str | None = None,
        out_channels: int = 256,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        if dino_backbones is None:
            raise ImportError("Cannot import dinov3.hub.backbones - make sure dinov3 is in sys.path")

        print(f"Initializing {model_name} backbone from local dinov3...")
        self.backbone = getattr(dino_backbones, model_name)(pretrained=False)
        print(f"Created {model_name} backbone")

        if checkpoint_path:
            print(f"Loading backbone weights from {checkpoint_path}...")
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            print("Loaded backbone checkpoint")
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        else:
            print("Warning: No checkpoint provided, using random initialization")

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            print("Freezing backbone parameters...")
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        with torch.no_grad():
            dummy = torch.randn(1, 3, image_size, image_size)
            feats = self.backbone.forward_features(dummy)["x_norm_patchtokens"]
            embed_dim = feats.shape[-1]
        print(f"Feature dimension: {embed_dim}")

        ps = self.backbone.patch_size
        if isinstance(ps, (tuple, list)):
            ph, pw = ps
        else:
            ph = pw = int(ps)
        self.patch_size = (ph, pw)

        self.proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,H,W]
        _, _, height, width = x.shape
        ph, pw = self.patch_size

        pad_h = (ph - height % ph) % ph
        pad_w = (pw - width % pw) % pw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            height = height + pad_h
            width = width + pad_w

        if self.freeze_backbone:
            self.backbone.eval()
            with torch.no_grad():
                out = self.backbone.forward_features(x)
        else:
            out = self.backbone.forward_features(x)

        patch_tokens = out["x_norm_patchtokens"]  # [B, N, C]
        batch_size, num_tokens, embed_dim = patch_tokens.shape
        h_patch = height // ph
        w_patch = width // pw

        if h_patch * w_patch != num_tokens:
            raise ValueError(
                f"Token count mismatch: H_patch*W_patch={h_patch*w_patch} vs N={num_tokens}. "
                "Ensure image sizes are multiples of patch size."
            )

        spatial_features = patch_tokens.reshape(batch_size, h_patch, w_patch, embed_dim).permute(0, 3, 1, 2)
        return self.proj(spatial_features)


class DinoV3FasterRCNN(nn.Module):
    """
    DINOv3 ViT-L/16 backbone + torchvision Faster R-CNN head.

    Foreground classes should be passed as `num_classes` (excluding background).
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = "dinov3_vitl16",
        image_size: int = 448,
        checkpoint_path: str | None = None,
        out_channels: int = 256,
        freeze_backbone: bool = True,
        anchor_sizes: tuple[tuple[int, ...], ...] = ((32, 64, 128, 256, 512),),
        aspect_ratios: tuple[tuple[float, ...], ...] = ((0.5, 1.0, 2.0),),
        image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        image_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        backbone = DinoV3BackboneForDetection(
            model_name=model_name,
            image_size=image_size,
            checkpoint_path=checkpoint_path,
            out_channels=out_channels,
            freeze_backbone=freeze_backbone,
        )

        anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)
        roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)

        # torchvision FasterRCNN expects num_classes including background
        self.detector = FasterRCNN(
            backbone,
            num_classes=num_classes + 1,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=image_size,
            max_size=image_size,
            image_mean=image_mean,
            image_std=image_std,
        )

    def forward(self, images, targets=None):
        return self.detector(images, targets)


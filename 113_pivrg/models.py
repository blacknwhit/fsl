from __future__ import annotations

from dataclasses import dataclass

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

from .lora import LoRAConfig, count_trainable_params, inject_lora_into_dinov3_ffn, mark_only_lora_as_trainable


@dataclass(frozen=True)
class TaskFeatureMeta:
    original_size: tuple[int, int]
    padded_size: tuple[int, int]
    pad_h: int
    pad_w: int


class SharedDinoV3Backbone(nn.Module):
    TASK_NAMES = ("det", "seg", "cnt")

    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
        image_size: int = 448,
        checkpoint_path: str | None = None,
        *,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        if dino_backbones is None:
            raise ImportError("Cannot import dinov3.hub.backbones - make sure dinov3 is in sys.path")

        self.model_name = str(model_name)
        self.image_size = int(image_size)
        self.use_lora = bool(use_lora)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = float(lora_dropout)

        self.backbone = getattr(dino_backbones, self.model_name)(pretrained=False)
        self.lora_config = (
            LoRAConfig(rank=self.lora_rank, alpha=self.lora_alpha, dropout=self.lora_dropout) if self.use_lora else None
        )

        if checkpoint_path:
            try:
                state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(checkpoint_path, map_location="cpu")
            self.backbone.load_state_dict(state, strict=False)

        with torch.no_grad():
            dummy = torch.randn(1, 3, self.image_size, self.image_size)
            tokens = self.backbone.forward_features(dummy)["x_norm_patchtokens"]
            self.embed_dim = int(tokens.shape[-1])

        patch_size = self.backbone.patch_size
        if isinstance(patch_size, (tuple, list)):
            self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        else:
            self.patch_size = (int(patch_size), int(patch_size))

        if self.use_lora:
            replaced = inject_lora_into_dinov3_ffn(self.backbone, cfg=self.lora_config)  # type: ignore[arg-type]
            mark_only_lora_as_trainable(self.backbone)
            trainable = count_trainable_params(self.backbone)
            if trainable == 0:
                raise RuntimeError("LoRA enabled but no trainable parameters found after injection")
            print(f"[lora] enabled: replaced_linear={replaced}, trainable_params={trainable}")

    def export_config(self) -> dict:
        return {
            "model_name": self.model_name,
            "image_size": self.image_size,
            "use_lora": self.use_lora,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
        }

    def _is_backbone_trainable(self) -> bool:
        return any(parameter.requires_grad for parameter in self.backbone.parameters())

    def _pad_to_patch(self, x: torch.Tensor) -> tuple[torch.Tensor, TaskFeatureMeta]:
        _, _, height, width = x.shape
        ph, pw = self.patch_size
        pad_h = (ph - height % ph) % ph
        pad_w = (pw - width % pw) % pw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        meta = TaskFeatureMeta(
            original_size=(int(height), int(width)),
            padded_size=(int(height + pad_h), int(width + pad_w)),
            pad_h=int(pad_h),
            pad_w=int(pad_w),
        )
        return x, meta

    def forward_task_features(self, x: torch.Tensor, *, task_name: str) -> tuple[torch.Tensor, TaskFeatureMeta]:
        if task_name not in self.TASK_NAMES:
            raise ValueError(f"Unknown task_name: {task_name}")

        x, meta = self._pad_to_patch(x)
        trainable = self._is_backbone_trainable()
        self.backbone.train(self.training and trainable)
        with torch.set_grad_enabled(self.training and trainable):
            outputs = self.backbone.forward_features(x)
            patch_tokens = outputs["x_norm_patchtokens"]

        batch_size, num_tokens, channels = patch_tokens.shape
        h_tokens = meta.padded_size[0] // self.patch_size[0]
        w_tokens = meta.padded_size[1] // self.patch_size[1]
        expected_tokens = int(h_tokens * w_tokens)
        if expected_tokens != int(num_tokens):
            raise ValueError(f"Token mismatch: expected {expected_tokens}, got {int(num_tokens)} for task '{task_name}'")

        feature_map = patch_tokens.reshape(batch_size, h_tokens, w_tokens, channels).permute(0, 3, 1, 2)
        return feature_map, meta

    def forward_features(self, x: torch.Tensor, *, task_name: str) -> dict:
        feature_map, _ = self.forward_task_features(x, task_name=task_name)
        batch_size, channels, h_tokens, w_tokens = feature_map.shape
        patch_tokens = feature_map.permute(0, 2, 3, 1).reshape(batch_size, h_tokens * w_tokens, channels)
        return {"x_norm_patchtokens": patch_tokens}


class _DetBackboneAdapter(nn.Module):
    def __init__(self, shared: SharedDinoV3Backbone, out_channels: int = 256):
        super().__init__()
        self.shared = shared
        self.out_channels = int(out_channels)
        self.proj = nn.Conv2d(shared.embed_dim, self.out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map, _ = self.shared.forward_task_features(x, task_name="det")
        return self.proj(feature_map)


class DinoSegHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, feats: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        logits = self.decode(feats)
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


class DinoCountHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, feats: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        density = self.decode(feats)
        density = F.softplus(density)
        return F.interpolate(density, size=out_size, mode="bilinear", align_corners=False)


class MultiTaskModel(nn.Module):
    TASK_ID_DET = "det"
    TASK_ID_SEG = "seg"
    TASK_ID_CNT = "cnt"

    def __init__(
        self,
        *,
        shared: SharedDinoV3Backbone,
        det_num_classes: int,
        seg_num_classes: int,
        cnt_num_classes: int,
        image_size: int = 448,
        det_out_channels: int = 256,
        det_image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        det_image_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.shared = shared
        self.image_size = int(image_size)
        self.det_out_channels = int(det_out_channels)
        self.det_num_classes = int(det_num_classes)
        self.seg_num_classes = int(seg_num_classes)
        self.cnt_num_classes = int(cnt_num_classes)

        det_backbone = _DetBackboneAdapter(shared, out_channels=self.det_out_channels)
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
        self.detector = FasterRCNN(
            det_backbone,
            num_classes=self.det_num_classes + 1,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=self.image_size,
            max_size=self.image_size,
            image_mean=det_image_mean,
            image_std=det_image_std,
        )

        self.seg_head = DinoSegHead(shared.embed_dim, self.seg_num_classes)
        self.cnt_head = DinoCountHead(shared.embed_dim, self.cnt_num_classes)

    def export_config(self) -> dict:
        config = self.shared.export_config()
        config.update(
            {
                "det_num_classes": self.det_num_classes,
                "seg_num_classes": self.seg_num_classes,
                "cnt_num_classes": self.cnt_num_classes,
                "det_out_channels": self.det_out_channels,
            }
        )
        return config

    def forward_det(self, images, targets=None):
        return self.detector(images, targets)

    def forward_seg(self, x: torch.Tensor) -> torch.Tensor:
        feature_map, meta = self.shared.forward_task_features(x, task_name=self.TASK_ID_SEG)
        logits = self.seg_head(feature_map, meta.padded_size)
        return logits[:, :, : meta.original_size[0], : meta.original_size[1]]

    def forward_cnt(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map, meta = self.shared.forward_task_features(x, task_name=self.TASK_ID_CNT)
        density = self.cnt_head(feature_map, meta.padded_size)
        density = density[:, :, : meta.original_size[0], : meta.original_size[1]]
        counts = density.flatten(2).sum(dim=2)
        return density, counts

    def forward(self, mode: str, *args, **kwargs):
        mode = str(mode).lower()
        if mode == "det":
            return self.forward_det(*args, **kwargs)
        if mode == "seg":
            return self.forward_seg(*args, **kwargs)
        if mode == "cnt":
            return self.forward_cnt(*args, **kwargs)
        raise ValueError(f"Unknown forward mode: {mode}")

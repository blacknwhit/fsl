from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from ours_IJEPA.common.vitmae_backbone import SharedIJEPABackbone

from .lora import LoRAConfig, count_trainable_params, inject_lora_into_vit_mlp, mark_only_lora_as_trainable


class SharedViTMAEBackboneWithLoRA(SharedIJEPABackbone):
    def __init__(
        self,
        model_name: str = "ijepa_vit_huge_patch16",
        image_size: int = 448,
        checkpoint_path: str | None = None,
        *,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__(
            model_name=model_name,
            image_size=image_size,
            checkpoint_path=checkpoint_path,
            pretrained=checkpoint_path is None,
        )
        self.use_lora = bool(use_lora)
        self.lora_config = (
            LoRAConfig(rank=int(lora_rank), alpha=float(lora_alpha), dropout=float(lora_dropout))
            if self.use_lora
            else None
        )
        if self.use_lora:
            replaced = inject_lora_into_vit_mlp(self.backbone, cfg=self.lora_config)  # type: ignore[arg-type]
            mark_only_lora_as_trainable(self.backbone)
            trainable = count_trainable_params(self.backbone)
            if trainable == 0:
                raise RuntimeError("LoRA enabled but no trainable params found after injection")
            print(f"[lora] enabled: replaced_linear={replaced}, trainable_params={trainable}")

    def forward_features(self, x: torch.Tensor, *, trainable_override: bool | None = None) -> dict[str, torch.Tensor]:
        if self.use_lora:
            trainable_override = None
        return super().forward_features(x, trainable_override=trainable_override)


class _DetBackboneAdapter(nn.Module):
    """Torchvision detector backbone adapter."""

    def __init__(
        self,
        shared: SharedViTMAEBackboneWithLoRA,
        out_channels: int = 256,
        *,
        trainable_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.shared = shared
        self.proj = nn.Conv2d(shared.embed_dim, out_channels, kernel_size=1)
        self.out_channels = int(out_channels)
        self.trainable_backbone = bool(trainable_backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        ph, pw = self.shared.patch_size

        pad_h = (ph - height % ph) % ph
        pad_w = (pw - width % pw) % pw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            height += pad_h
            width += pad_w

        tokens = self.shared.forward_features(x, trainable_override=self.trainable_backbone)["x_norm_patchtokens"]
        bsz, n, c = tokens.shape
        h_patch = height // ph
        w_patch = width // pw
        if h_patch * w_patch != n:
            raise ValueError(f"Token mismatch: h_patch*w_patch={h_patch*w_patch} vs N={n}")

        feat = tokens.reshape(bsz, h_patch, w_patch, c).permute(0, 3, 1, 2)
        return self.proj(feat)


class DinoSegHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, feats: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        logits = self.decode(feats)
        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)


class DinoCountHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, feats: torch.Tensor, out_size: tuple[int, int]) -> torch.Tensor:
        density = self.decode(feats)
        density = F.softplus(density)
        return F.interpolate(density, size=out_size, mode="bilinear", align_corners=False)


@dataclass(frozen=True)
class TaskOutputs:
    det_loss: torch.Tensor
    seg_loss: torch.Tensor
    cnt_loss: torch.Tensor


class MultiTaskModel(nn.Module):
    def __init__(
        self,
        shared: SharedViTMAEBackboneWithLoRA,
        det_num_classes: int,
        seg_num_classes: int,
        cnt_num_classes: int,
        image_size: int = 448,
        det_out_channels: int = 256,
        det_image_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        det_image_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        det_train_backbone: bool = True,
        seg_train_backbone: bool = True,
        cnt_train_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.shared = shared
        self.det_train_backbone = bool(det_train_backbone)
        self.seg_train_backbone = bool(seg_train_backbone)
        self.cnt_train_backbone = bool(cnt_train_backbone)

        det_backbone = _DetBackboneAdapter(
            shared,
            out_channels=det_out_channels,
            trainable_backbone=self.det_train_backbone,
        )
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))
        roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
        self.detector = FasterRCNN(
            det_backbone,
            num_classes=int(det_num_classes) + 1,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=image_size,
            max_size=image_size,
            image_mean=det_image_mean,
            image_std=det_image_std,
        )

        self.seg_head = DinoSegHead(shared.embed_dim, int(seg_num_classes))
        self.cnt_head = DinoCountHead(shared.embed_dim, int(cnt_num_classes))
        self.cnt_num_classes = int(cnt_num_classes)

    def forward_det(self, images, targets=None):
        return self.detector(images, targets)

    def forward_seg(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, h, w = x.shape
        ph, pw = self.shared.patch_size
        tokens = self.shared.forward_features(x, trainable_override=self.seg_train_backbone)["x_norm_patchtokens"]
        _, n, c = tokens.shape
        h_patch = h // ph
        w_patch = w // pw
        feat = tokens.reshape(bsz, h_patch, w_patch, c).permute(0, 3, 1, 2)
        return self.seg_head(feat, (h, w))

    @staticmethod
    def _cnt_feat_with_scaled_backbone_grad(feat: torch.Tensor, mult: float) -> torch.Tensor:
        m = float(mult)
        if m == 1.0 or not feat.requires_grad:
            return feat
        feat2 = feat.clone()
        feat2.register_hook(lambda g: g * m)
        return feat2

    def forward_cnt(self, x: torch.Tensor, *, cnt_backbone_grad_mult: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, _, orig_h, orig_w = x.shape
        ph, pw = self.shared.patch_size
        pad_h = (ph - orig_h % ph) % ph
        pad_w = (pw - orig_w % pw) % pw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        height = orig_h + pad_h
        width = orig_w + pad_w

        tokens = self.shared.forward_features(x, trainable_override=self.cnt_train_backbone)["x_norm_patchtokens"]
        _, n, c = tokens.shape
        h_patch = height // ph
        w_patch = width // pw
        if h_patch * w_patch != n:
            raise ValueError(f"Token mismatch: h_patch*w_patch={h_patch*w_patch}, num_tokens={n}")
        feat = tokens.reshape(bsz, h_patch, w_patch, c).permute(0, 3, 1, 2)
        feat = self._cnt_feat_with_scaled_backbone_grad(feat, cnt_backbone_grad_mult)

        density = self.cnt_head(feat, (height, width))
        density = density[:, :, :orig_h, :orig_w]
        counts = density.flatten(2).sum(dim=2)
        return density, counts

    def forward_seg_and_cnt(
        self,
        seg_x: torch.Tensor,
        cnt_x: torch.Tensor,
        *,
        cnt_backbone_grad_mult: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.seg_train_backbone != self.cnt_train_backbone:
            raise ValueError("forward_seg_and_cnt requires seg_train_backbone == cnt_train_backbone")
        if seg_x.dim() != 4 or cnt_x.dim() != 4:
            raise ValueError("Expected seg_x/cnt_x to be [B,3,H,W] tensors")
        if seg_x.shape[1:] != cnt_x.shape[1:]:
            raise ValueError(f"seg_x and cnt_x must have same C/H/W, got {tuple(seg_x.shape)} vs {tuple(cnt_x.shape)}")

        seg_bsz, _, h, w = seg_x.shape
        cnt_bsz = int(cnt_x.shape[0])
        ph, pw = self.shared.patch_size
        if (h % ph) or (w % pw):
            raise ValueError(f"Fused seg+cnt requires patch-aligned H/W, got {(h, w)} for patch {self.shared.patch_size}")

        x = torch.cat([seg_x, cnt_x], dim=0)
        tokens = self.shared.forward_features(x, trainable_override=self.seg_train_backbone)["x_norm_patchtokens"]
        _, n, c = tokens.shape
        h_patch = h // ph
        w_patch = w // pw
        if h_patch * w_patch != n:
            raise ValueError(f"Token mismatch: h_patch*w_patch={h_patch*w_patch}, num_tokens={n}")

        feat = tokens.reshape(seg_bsz + cnt_bsz, h_patch, w_patch, c).permute(0, 3, 1, 2)
        seg_feat = feat[:seg_bsz]
        cnt_feat = feat[seg_bsz:]

        seg_logits = self.seg_head(seg_feat, (h, w))
        cnt_feat = self._cnt_feat_with_scaled_backbone_grad(cnt_feat, cnt_backbone_grad_mult)
        density = self.cnt_head(cnt_feat, (h, w))
        counts = density.flatten(2).sum(dim=2)
        return seg_logits, density, counts

    def forward(self, task: str, *args, **kwargs):
        task_key = str(task).lower()
        if task_key == "det":
            return self.forward_det(*args, **kwargs)
        if task_key == "seg":
            return self.forward_seg(*args, **kwargs)
        if task_key == "cnt":
            return self.forward_cnt(*args, **kwargs)
        if task_key == "seg_cnt":
            return self.forward_seg_and_cnt(*args, **kwargs)
        raise ValueError(f"Unknown task: {task}")


SharedViTMAEBackbone = SharedViTMAEBackboneWithLoRA
SharedIJEPABackboneWithLoRA = SharedViTMAEBackboneWithLoRA

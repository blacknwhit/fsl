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

from ours_IJEPA.common.vitmae_backbone import SharedIJEPABackbone as BaseSharedViTMAEBackbone
from ours_IJEPA.common.vitmae_backbone import build_vitmae_embeddings

try:
    from .vit_moe_wrapper import wrap_vit_blocks_with_lora_moe
except ImportError:
    from vit_moe_wrapper import wrap_vit_blocks_with_lora_moe


class SharedViTMAEBackbone(BaseSharedViTMAEBackbone):
    """
    Shared IJEPA ViT-H/16 backbone with optional LoRA-MoE adapters.

    `use_lora_moe` controls the architecture.
    `backbone_trainable` controls whether the original backbone parameters are updated.
    """

    def __init__(
        self,
        model_name: str = "ijepa_vit_huge_patch16",
        image_size: int = 448,
        checkpoint_path: str | None = None,
        *,
        use_lora_moe: bool = False,
        backbone_trainable: bool | None = None,
        task_num: int = 3,
        lora_rank: int = 8,
        num_experts_private: int = 2,
        num_experts_shared: int = 6,
        moe_k_private: int = 2,
        moe_k_shared: int = 2,
        grad_checkpointing: bool = False,
    ) -> None:
        super().__init__(
            model_name=model_name,
            image_size=image_size,
            checkpoint_path=checkpoint_path,
            pretrained=checkpoint_path is None,
        )
        self.use_lora_moe = bool(use_lora_moe)
        self.task_num = int(task_num)
        self.grad_checkpointing = bool(grad_checkpointing)
        if backbone_trainable is None:
            backbone_trainable = not self.use_lora_moe

        if self.use_lora_moe:
            self._setup_lora_moe(
                lora_rank=int(lora_rank),
                num_experts_private=int(num_experts_private),
                num_experts_shared=int(num_experts_shared),
                moe_k_private=int(moe_k_private),
                moe_k_shared=int(moe_k_shared),
            )
        self.set_backbone_trainable(bool(backbone_trainable))

    def _setup_lora_moe(
        self,
        *,
        lora_rank: int,
        num_experts_private: int,
        num_experts_shared: int,
        moe_k_private: int,
        moe_k_shared: int,
    ) -> None:
        self.wrapped_blocks, self.lora_moes = wrap_vit_blocks_with_lora_moe(
            self.backbone.blocks,
            input_size=self.embed_dim,
            rank=lora_rank,
            num_experts_private=num_experts_private,
            num_experts_shared=num_experts_shared,
            k_private=moe_k_private,
            k_shared=moe_k_shared,
            task_num=self.task_num,
            grad_checkpointing=self.grad_checkpointing,
        )

    def _forward_features_with_lora_moe(self, x: torch.Tensor, task_id: int) -> dict[str, torch.Tensor]:
        hidden = build_vitmae_embeddings(self.backbone, x)
        for wrapped_block in self.wrapped_blocks:
            hidden = wrapped_block(hidden, task_id=task_id)
        hidden = self.backbone.norm(hidden)
        return {"x_norm_patchtokens": hidden}

    def forward_features(
        self,
        x: torch.Tensor,
        *,
        trainable_override: bool | None = None,
        task_id: int | None = None,
    ) -> dict[str, torch.Tensor]:
        trainable = bool(trainable_override) if trainable_override is not None else self._trainable()
        self.backbone.train(self.training and trainable)
        if self.use_lora_moe:
            if task_id is None:
                raise ValueError("task_id is required when use_lora_moe=True")
            with torch.set_grad_enabled(self.training):
                return self._forward_features_with_lora_moe(x, task_id)
        return super().forward_features(x, trainable_override=trainable)


class _DetBackboneAdapter(nn.Module):
    """Torchvision detector backbone adapter."""

    def __init__(self, shared: SharedViTMAEBackbone, out_channels: int = 256, *, trainable_backbone: bool = True, task_id: int = 0):
        super().__init__()
        self.shared = shared
        self.proj = nn.Conv2d(shared.embed_dim, out_channels, kernel_size=1)
        self.out_channels = int(out_channels)
        self.trainable_backbone = bool(trainable_backbone)
        self.task_id = int(task_id)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        ph, pw = self.shared.patch_size

        pad_h = (ph - height % ph) % ph
        pad_w = (pw - width % pw) % pw
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            height += pad_h
            width += pad_w

        tokens = self.shared.forward_features(
            x,
            trainable_override=self.trainable_backbone,
            task_id=self.task_id if self.shared.use_lora_moe else None,
        )["x_norm_patchtokens"]

        bsz, n, c = tokens.shape
        h_patch = height // ph
        w_patch = width // pw
        if h_patch * w_patch != n:
            raise ValueError(f"Token mismatch: h_patch*w_patch={h_patch*w_patch} vs N={n}")

        feat = tokens.reshape(bsz, h_patch, w_patch, c).permute(0, 3, 1, 2)
        return self.proj(feat)


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


@dataclass(frozen=True)
class TaskOutputs:
    det_loss: torch.Tensor
    seg_loss: torch.Tensor
    cnt_loss: torch.Tensor


class MultiTaskModel(nn.Module):
    """
    Multi-task model with shared IJEPA ViT-H/16 backbone.

    Task IDs for LoRA-MoE routing:
    - 0: Detection
    - 1: Segmentation
    - 2: Counting
    """

    TASK_ID_DET = 0
    TASK_ID_SEG = 1
    TASK_ID_CNT = 2

    def __init__(
        self,
        shared: SharedViTMAEBackbone,
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
    ):
        super().__init__()
        self.shared = shared
        self.det_train_backbone = bool(det_train_backbone)
        self.seg_train_backbone = bool(seg_train_backbone)
        self.cnt_train_backbone = bool(cnt_train_backbone)

        det_backbone = _DetBackboneAdapter(
            shared,
            out_channels=det_out_channels,
            trainable_backbone=self.det_train_backbone,
            task_id=self.TASK_ID_DET,
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
        tokens = self.shared.forward_features(
            x,
            trainable_override=self.seg_train_backbone,
            task_id=self.TASK_ID_SEG if self.shared.use_lora_moe else None,
        )["x_norm_patchtokens"]

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

        tokens = self.shared.forward_features(
            x,
            trainable_override=self.cnt_train_backbone,
            task_id=self.TASK_ID_CNT if self.shared.use_lora_moe else None,
        )["x_norm_patchtokens"]

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

    def forward(self, mode: str, *args, **kwargs):
        mode = str(mode).lower()
        if mode == "det":
            return self.forward_det(*args, **kwargs)
        if mode == "seg":
            return self.forward_seg(*args, **kwargs)
        if mode == "cnt":
            return self.forward_cnt(*args, **kwargs)
        if mode == "seg_cnt":
            return self.forward_seg_and_cnt(*args, **kwargs)
        raise ValueError(f"Unknown forward mode: {mode}")

    def forward_seg_and_cnt(
        self,
        seg_x: torch.Tensor,
        cnt_x: torch.Tensor,
        *,
        cnt_backbone_grad_mult: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.shared.use_lora_moe:
            raise ValueError(
                "forward_seg_and_cnt is not supported when using LoRA-MoE. "
                "Use forward_seg and forward_cnt separately to preserve task routing."
            )
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

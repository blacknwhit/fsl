from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

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

try:
    from .lora_moe import LoRATaskMoE
    from .dinov3_moe_wrapper import LoRAMoEBlockWrapper
except ImportError:
    from lora_moe import LoRATaskMoE
    from dinov3_moe_wrapper import LoRAMoEBlockWrapper


class SharedDinoV3Backbone(nn.Module):
    """
    Shared DINOv3 backbone with optional LoRA-MoE adapters.

    `use_lora_moe` controls the architecture.
    `backbone_trainable` controls whether the original backbone parameters are updated.
    """
    
    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
        image_size: int = 448,
        checkpoint_path: str | None = None,
        # LoRA-MoE parameters
        use_lora_moe: bool = False,
        backbone_trainable: bool | None = None,
        task_num: int = 3,
        lora_rank: int = 8,
        num_experts_private: int = 2,
        num_experts_shared: int = 6,
        moe_k_private: int = 2,
        moe_k_shared: int = 2,
        # Memory optimization (does NOT change losses): recompute activations in backward.
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        if dino_backbones is None:
            raise ImportError("Cannot import dinov3.hub.backbones - make sure dinov3 is in sys.path")

        self.backbone = getattr(dino_backbones, model_name)(pretrained=False)

        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            self.backbone.load_state_dict(state, strict=False)

        with torch.no_grad():
            dummy = torch.randn(1, 3, image_size, image_size)
            tokens = self.backbone.forward_features(dummy)["x_norm_patchtokens"]
            self.embed_dim = int(tokens.shape[-1])

        ps = self.backbone.patch_size
        if isinstance(ps, (tuple, list)):
            ph, pw = int(ps[0]), int(ps[1])
        else:
            ph = pw = int(ps)
        self.patch_size = (ph, pw)
        
        # LoRA-MoE setup
        self.use_lora_moe = use_lora_moe
        self.task_num = task_num
        self.grad_checkpointing = bool(grad_checkpointing)
        if backbone_trainable is None:
            # Preserve the old default behavior for callers that do not pass the flag:
            # plain backbone finetunes by default, LoRA-MoE defaults to frozen backbone.
            backbone_trainable = not use_lora_moe

        if use_lora_moe:
            self._setup_lora_moe(
                lora_rank=lora_rank,
                num_experts_private=num_experts_private,
                num_experts_shared=num_experts_shared,
                moe_k_private=moe_k_private,
                moe_k_shared=moe_k_shared,
            )
        self.set_backbone_trainable(bool(backbone_trainable))
    
    def _setup_lora_moe(
        self,
        lora_rank: int,
        num_experts_private: int,
        num_experts_shared: int,
        moe_k_private: int,
        moe_k_shared: int,
    ):
        """Setup LoRA-MoE wrappers around the backbone blocks."""
        # Create wrapped blocks
        self.wrapped_blocks = nn.ModuleList()
        self.lora_moes = nn.ModuleList()
        
        for block in self.backbone.blocks:
            # Create LoRA-MoE for this block
            lora_moe = LoRATaskMoE(
                input_size=self.embed_dim,
                rank=lora_rank,
                num_experts_private=num_experts_private,
                num_experts_shared=num_experts_shared,
                k_private=moe_k_private,
                k_shared=moe_k_shared,
                task_num=self.task_num,
            )
            
            # Wrap the block
            wrapper = LoRAMoEBlockWrapper(block, lora_moe, grad_checkpointing=self.grad_checkpointing)
            self.wrapped_blocks.append(wrapper)
            self.lora_moes.append(lora_moe)

    def set_backbone_trainable(self, trainable: bool) -> None:
        trainable = bool(trainable)
        for param in self.backbone.parameters():
            param.requires_grad = trainable

    def _trainable(self) -> bool:
        return any(p.requires_grad for p in self.backbone.parameters())
    
    def forward_features(self, x: torch.Tensor, *, trainable_override: bool | None = None, task_id: int | None = None) -> dict:
        """
        Forward pass through DINOv3 backbone.
        
        Args:
            x: Input images [B, 3, H, W]
            trainable_override: Override trainable setting
            task_id: Task ID for LoRA-MoE routing (required if use_lora_moe=True)
            
        Returns:
            Dict with 'x_norm_patchtokens' key
        """
        trainable = bool(trainable_override) if trainable_override is not None else self._trainable()
        self.backbone.train(self.training and trainable)
        if self.use_lora_moe:
            if task_id is None:
                raise ValueError("task_id is required when use_lora_moe=True")
            return self._forward_features_with_lora_moe(x, task_id)
        with torch.set_grad_enabled(self.training and trainable):
            return self.backbone.forward_features(x)
    
    def _forward_features_with_lora_moe(self, x: torch.Tensor, task_id: int) -> dict:
        """Forward with LoRA-MoE adapters."""
        # Prepare tokens
        x, (H, W) = self.backbone.prepare_tokens_with_masks(x, masks=None)
        
        # Get RoPE embeddings
        rope = None
        if hasattr(self.backbone, 'rope_embed') and self.backbone.rope_embed is not None:
            rope = self.backbone.rope_embed(H=H, W=W)
        
        # Apply blocks with LoRA-MoE
        for wrapped_block in self.wrapped_blocks:
            # Forward through wrapped block. Any checkpointing is handled inside the wrapper
            # (attention+MLP only) to avoid checkpointing stateful MoE aux-stat updates.
            x = wrapped_block(x, task_id=task_id, rope=rope)
        
        # Extract patch tokens (skip CLS and storage tokens)
        n_prefix = 1 + self.backbone.n_storage_tokens  # CLS + storage tokens
        
        # Apply final norm
        if hasattr(self.backbone, 'untie_cls_and_patch_norms') and self.backbone.untie_cls_and_patch_norms:
            x_norm_patch = self.backbone.norm(x[:, n_prefix:])
        else:
            x_norm = self.backbone.norm(x)
            x_norm_patch = x_norm[:, n_prefix:]
        
        return {'x_norm_patchtokens': x_norm_patch}


class _DetBackboneAdapter(nn.Module):
    """
    Torchvision detector backbone adapter.
    Returns a single feature map tensor and exposes out_channels.
    """

    def __init__(self, shared: SharedDinoV3Backbone, out_channels: int = 256, *, trainable_backbone: bool = True, task_id: int = 0):
        super().__init__()
        self.shared = shared
        self.proj = nn.Conv2d(shared.embed_dim, out_channels, kernel_size=1)
        self.out_channels = int(out_channels)
        self.trainable_backbone = bool(trainable_backbone)
        self.task_id = task_id  # Task ID for LoRA-MoE routing

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
    Multi-task model with shared DINOv3 backbone.
    
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
        shared: SharedDinoV3Backbone,
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
            task_id=self.TASK_ID_DET,  # Detection uses task_id=0
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
        """Detection forward. Uses task_id=0 for LoRA-MoE routing."""
        return self.detector(images, targets)

    def forward_seg(self, x: torch.Tensor) -> torch.Tensor:
        """Segmentation forward. Uses task_id=1 for LoRA-MoE routing."""
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
        """
        方式A：只缩放“计数分支 -> backbone”的梯度，不影响计数头参数的梯度。
        做法：对输入给 cnt_head 的 feat 做 clone，并在该 clone 上 register_hook 缩放 dL/dfeat。
        """
        m = float(mult)
        if m == 1.0:
            return feat
        # 只有在需要反传时才能挂 hook（否则会报错）
        if not feat.requires_grad:
            return feat
        feat2 = feat.clone()
        feat2.register_hook(lambda g: g * m)
        return feat2

    def forward_cnt(self, x: torch.Tensor, *, cnt_backbone_grad_mult: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Counting forward. Uses task_id=2 for LoRA-MoE routing."""
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

        # 只对"计数任务回传到backbone"的梯度做缩放；不影响cnt_head参数梯度
        feat = self._cnt_feat_with_scaled_backbone_grad(feat, cnt_backbone_grad_mult)

        density = self.cnt_head(feat, (height, width))
        density = density[:, :, :orig_h, :orig_w]
        counts = density.flatten(2).sum(dim=2)  # [B,C]
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

    # NOTE: forward_seg_and_cnt is deprecated when using LoRA-MoE
    # Each task should be forwarded separately to avoid routing conflicts
    def forward_seg_and_cnt(
        self,
        seg_x: torch.Tensor,
        cnt_x: torch.Tensor,
        *,
        cnt_backbone_grad_mult: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fused seg+cnt forward that runs the shared backbone once.
        
        WARNING: This method is NOT supported when using LoRA-MoE.
        Use forward_seg and forward_cnt separately to ensure correct task routing.

        Requirements (to keep behavior consistent with separate forwards):
        - seg_x and cnt_x must have the same [B,3,H,W] spatial shape.
        - H/W must already be patch-aligned (no implicit padding), otherwise segmentation behavior would change.
        - seg_train_backbone and cnt_train_backbone must match (train/eval mode of the shared backbone).
        """
        # LoRA-MoE requires separate forwards for correct routing
        if self.shared.use_lora_moe:
            raise ValueError(
                "forward_seg_and_cnt is not supported when using LoRA-MoE. "
                "Use forward_seg and forward_cnt separately to ensure correct task routing."
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

        # 关键：只缩放 cnt 分支回传到 backbone 的梯度（seg 不受影响）
        cnt_feat = self._cnt_feat_with_scaled_backbone_grad(cnt_feat, cnt_backbone_grad_mult)

        density = self.cnt_head(cnt_feat, (h, w))
        counts = density.flatten(2).sum(dim=2)
        return seg_logits, density, counts

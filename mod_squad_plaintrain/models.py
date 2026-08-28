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

try:
    from .dinov3_moe_wrapper import LoRAMoEBlockWrapper
    from .lora_moe import LoRATaskMoE
except ImportError:
    from dinov3_moe_wrapper import LoRAMoEBlockWrapper
    from lora_moe import LoRATaskMoE


class SharedDinoV3Backbone(nn.Module):
    """
    Shared DINOv3 backbone with optional 113_test-style LoRA-MoE adapters.

    This experiment is shared-expert-only:
    - Task-private experts are disabled (fixed to 0).
    - Task-specific routers over shared experts are kept.
    """

    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
        image_size: int = 448,
        checkpoint_path: str | None = None,
        *,
        use_lora_moe: bool = False,
        task_num: int = 3,
        lora_rank: int = 8,
        num_experts_private: int = 0,
        num_experts_shared: int = 6,
        moe_k_private: int = 0,
        moe_k_shared: int = 2,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        if dino_backbones is None:
            raise ImportError("Cannot import dinov3.hub.backbones - make sure dinov3 is in sys.path")

        self.model_name = str(model_name)
        self.image_size = int(image_size)
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

        self.use_lora_moe = bool(use_lora_moe)
        self.task_num = int(task_num)
        self.lora_rank = int(lora_rank)
        self.num_experts_private = 0
        self.num_experts_shared = int(num_experts_shared)
        self.moe_k_private = 0
        self.moe_k_shared = int(moe_k_shared)
        self.grad_checkpointing = bool(grad_checkpointing)

        # Keep constructor args for compatibility but force shared-only.
        _ = int(num_experts_private)
        _ = int(moe_k_private)

        self.wrapped_blocks = nn.ModuleList()
        self.lora_moes = nn.ModuleList()
        self._disable_inplace_layerscale()
        if self.use_lora_moe:
            self._setup_lora_moe()

    def _disable_inplace_layerscale(self) -> None:
        # Some DINOv3 variants expose LayerScale(..., inplace=True).
        # Force non-inplace behavior to keep autograd/DDP backward stable.
        for module in self.backbone.modules():
            if hasattr(module, "inplace"):
                try:
                    module.inplace = False
                except Exception:
                    pass

    def _setup_lora_moe(self) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = False

        for block in self.backbone.blocks:
            lora_moe = LoRATaskMoE(
                input_size=self.embed_dim,
                rank=self.lora_rank,
                num_experts_private=self.num_experts_private,
                num_experts_shared=self.num_experts_shared,
                k_private=self.moe_k_private,
                k_shared=self.moe_k_shared,
                task_num=self.task_num,
            )
            wrapper = LoRAMoEBlockWrapper(block, lora_moe, grad_checkpointing=self.grad_checkpointing)
            self.wrapped_blocks.append(wrapper)
            self.lora_moes.append(lora_moe)

    def _trainable(self) -> bool:
        return any(p.requires_grad for p in self.backbone.parameters())

    def clear_aux_stats(self) -> None:
        for wrapper in self.wrapped_blocks:
            wrapper.clear_aux_stats()

    def set_collect_aux_stats(self, enabled: bool) -> None:
        for wrapper in self.wrapped_blocks:
            wrapper.set_collect_aux_stats(enabled)

    def get_mi_loss_and_clear(self) -> torch.Tensor:
        loss = None
        for wrapper in self.wrapped_blocks:
            cur = wrapper.get_mi_loss_and_clear()
            loss = cur if loss is None else loss + cur
        if loss is None:
            return self.backbone.cls_token.sum() * 0.0
        return loss

    def export_model_config(self) -> dict:
        return {
            "model_name": self.model_name,
            "image_size": int(self.image_size),
            "use_lora_moe": bool(self.use_lora_moe),
            "task_num": int(self.task_num),
            "lora_rank": int(self.lora_rank),
            "num_experts_private": int(self.num_experts_private),
            "num_experts_shared": int(self.num_experts_shared),
            "moe_k_private": int(self.moe_k_private),
            "moe_k_shared": int(self.moe_k_shared),
            "shared_only": True,
        }

    def forward_features(self, x: torch.Tensor, *, trainable_override: bool | None = None, task_id: int | None = None) -> dict:
        if self.use_lora_moe:
            if task_id is None:
                raise ValueError("task_id is required when use_lora_moe=True")
            return self._forward_features_with_lora_moe(x, int(task_id))

        trainable = bool(trainable_override) if trainable_override is not None else self._trainable()
        self.backbone.train(self.training and trainable)
        with torch.set_grad_enabled(self.training and trainable):
            return self.backbone.forward_features(x)

    def _forward_features_with_lora_moe(self, x: torch.Tensor, task_id: int) -> dict:
        x, (height, width) = self.backbone.prepare_tokens_with_masks(x, masks=None)

        rope = None
        if hasattr(self.backbone, "rope_embed") and self.backbone.rope_embed is not None:
            rope = self.backbone.rope_embed(H=height, W=width)

        for wrapped_block in self.wrapped_blocks:
            x = wrapped_block(x, task_id=task_id, rope=rope)

        n_prefix = 1 + self.backbone.n_storage_tokens
        if hasattr(self.backbone, "untie_cls_and_patch_norms") and self.backbone.untie_cls_and_patch_norms:
            x_norm_patch = self.backbone.norm(x[:, n_prefix:])
        else:
            x_norm = self.backbone.norm(x)
            x_norm_patch = x_norm[:, n_prefix:]

        return {"x_norm_patchtokens": x_norm_patch}


class _DetBackboneAdapter(nn.Module):
    def __init__(
        self,
        shared: SharedDinoV3Backbone,
        out_channels: int = 256,
        *,
        trainable_backbone: bool = True,
        task_id: int = 0,
    ):
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

        if self.shared.use_lora_moe:
            tokens = self.shared.forward_features(x, task_id=self.task_id)["x_norm_patchtokens"]
        else:
            tokens = self.shared.forward_features(x, trainable_override=self.trainable_backbone)["x_norm_patchtokens"]

        bsz, num_tokens, channels = tokens.shape
        h_patch = height // ph
        w_patch = width // pw
        if h_patch * w_patch != num_tokens:
            raise ValueError(f"Token mismatch: h_patch*w_patch={h_patch*w_patch} vs N={num_tokens}")

        feat = tokens.reshape(bsz, h_patch, w_patch, channels).permute(0, 3, 1, 2)
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

    def clear_aux_stats(self) -> None:
        self.shared.clear_aux_stats()

    def set_collect_aux_stats(self, enabled: bool) -> None:
        self.shared.set_collect_aux_stats(enabled)

    def get_mi_loss_and_clear(self) -> torch.Tensor:
        return self.shared.get_mi_loss_and_clear()

    def forward_det(self, images, targets=None):
        return self.detector(images, targets)

    def forward_seg(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, height, width = x.shape
        ph, pw = self.shared.patch_size
        if self.shared.use_lora_moe:
            tokens = self.shared.forward_features(x, task_id=self.TASK_ID_SEG)["x_norm_patchtokens"]
        else:
            tokens = self.shared.forward_features(x, trainable_override=self.seg_train_backbone)["x_norm_patchtokens"]
        _, num_tokens, channels = tokens.shape
        h_patch = height // ph
        w_patch = width // pw
        if h_patch * w_patch != num_tokens:
            raise ValueError(f"Token mismatch: h_patch*w_patch={h_patch*w_patch}, num_tokens={num_tokens}")
        feat = tokens.reshape(bsz, h_patch, w_patch, channels).permute(0, 3, 1, 2)
        return self.seg_head(feat, (height, width))

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

        if self.shared.use_lora_moe:
            tokens = self.shared.forward_features(x, task_id=self.TASK_ID_CNT)["x_norm_patchtokens"]
        else:
            tokens = self.shared.forward_features(x, trainable_override=self.cnt_train_backbone)["x_norm_patchtokens"]

        _, num_tokens, channels = tokens.shape
        h_patch = height // ph
        w_patch = width // pw
        if h_patch * w_patch != num_tokens:
            raise ValueError(f"Token mismatch: h_patch*w_patch={h_patch*w_patch}, num_tokens={num_tokens}")
        feat = tokens.reshape(bsz, h_patch, w_patch, channels).permute(0, 3, 1, 2)
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
        if mode == "all":
            return self.forward_all(*args, **kwargs)
        if mode == "seg_cnt":
            return self.forward_seg_and_cnt(*args, **kwargs)
        raise ValueError(f"Unknown forward mode: {mode}")

    def forward_all(
        self,
        det_images,
        det_targets,
        seg_x: torch.Tensor,
        cnt_x: torch.Tensor,
        *,
        cnt_backbone_grad_mult: float = 1.0,
    ):
        det_out = self.forward_det(det_images, det_targets)
        seg_logits = self.forward_seg(seg_x)
        pred_dens, pred_counts = self.forward_cnt(cnt_x, cnt_backbone_grad_mult=cnt_backbone_grad_mult)
        return det_out, seg_logits, pred_dens, pred_counts

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
                "Use forward_seg and forward_cnt separately to ensure correct task routing."
            )
        if self.seg_train_backbone != self.cnt_train_backbone:
            raise ValueError("forward_seg_and_cnt requires seg_train_backbone == cnt_train_backbone")
        if seg_x.dim() != 4 or cnt_x.dim() != 4:
            raise ValueError("Expected seg_x/cnt_x to be [B,3,H,W] tensors")
        if seg_x.shape[1:] != cnt_x.shape[1:]:
            raise ValueError(f"seg_x and cnt_x must have same C/H/W, got {tuple(seg_x.shape)} vs {tuple(cnt_x.shape)}")

        seg_bsz, _, height, width = seg_x.shape
        cnt_bsz = int(cnt_x.shape[0])
        ph, pw = self.shared.patch_size
        if (height % ph) or (width % pw):
            raise ValueError(f"Fused seg+cnt requires patch-aligned H/W, got {(height, width)} for patch {self.shared.patch_size}")

        x = torch.cat([seg_x, cnt_x], dim=0)
        tokens = self.shared.forward_features(x, trainable_override=self.seg_train_backbone)["x_norm_patchtokens"]
        _, num_tokens, channels = tokens.shape
        h_patch = height // ph
        w_patch = width // pw
        if h_patch * w_patch != num_tokens:
            raise ValueError(f"Token mismatch: h_patch*w_patch={h_patch*w_patch}, num_tokens={num_tokens}")

        feat = tokens.reshape(seg_bsz + cnt_bsz, h_patch, w_patch, channels).permute(0, 3, 1, 2)
        seg_feat = feat[:seg_bsz]
        cnt_feat = feat[seg_bsz:]
        seg_logits = self.seg_head(seg_feat, (height, width))

        cnt_feat = self._cnt_feat_with_scaled_backbone_grad(cnt_feat, cnt_backbone_grad_mult)
        density = self.cnt_head(cnt_feat, (height, width))
        counts = density.flatten(2).sum(dim=2)
        return seg_logits, density, counts

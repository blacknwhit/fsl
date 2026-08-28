from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from dinov3.hub import backbones as dino_backbones
except ImportError:
    dino_backbones = None

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

try:
    from .adapters import SharedExpertBlockAdapter
except ImportError:
    from adapters import SharedExpertBlockAdapter


@dataclass(frozen=True)
class TaskFeatureMeta:
    original_size: tuple[int, int]
    padded_size: tuple[int, int]
    pad_h: int
    pad_w: int


class ParallelFFNBlockWrapper(nn.Module):
    """
    Add a block-level parallel adapter to the FFN residual branch:

        x_attn = x + ls1(attn(norm1(x)))
        u = norm2(x_attn)
        x_out = x_attn + ls2(mlp(u)) + adapter(u, task_name)
    """

    def __init__(self, block: nn.Module, adapter: SharedExpertBlockAdapter, *, grad_checkpointing: bool = False):
        super().__init__()
        self.block = block
        self.adapter = adapter
        self.grad_checkpointing = bool(grad_checkpointing)

    def _forward_attention(self, x: torch.Tensor, rope):
        norm1_out = self.block.norm1(x)
        try:
            attn_out = self.block.attn(norm1_out, rope=rope)
        except TypeError:
            try:
                attn_out = self.block.attn(norm1_out, rope)
            except TypeError:
                attn_out = self.block.attn(norm1_out)
        return x + self.block.ls1(attn_out)

    def _forward_mlp(self, x_attn: torch.Tensor):
        u = self.block.norm2(x_attn)
        return u, self.block.ls2(self.block.mlp(u))

    def forward(self, x: torch.Tensor, *, task_name: str, rope=None) -> torch.Tensor:
        if self.training and self.grad_checkpointing:

            def _attn_fn(inp: torch.Tensor) -> torch.Tensor:
                return self._forward_attention(inp, rope)

            x_attn = checkpoint(_attn_fn, x, use_reentrant=False)

            def _mlp_fn(inp: torch.Tensor):
                u_local = self.block.norm2(inp)
                return self.block.ls2(self.block.mlp(u_local))

            ffn_out = checkpoint(_mlp_fn, x_attn, use_reentrant=False)
            u = self.block.norm2(x_attn)
        else:
            x_attn = self._forward_attention(x, rope)
            u, ffn_out = self._forward_mlp(x_attn)

        adapter_out = self.adapter(u, task_name=task_name)
        return x_attn + ffn_out + adapter_out


class SharedDinoV3Backbone(nn.Module):
    TASK_NAMES = ("det", "seg", "cnt")

    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
        image_size: int = 448,
        checkpoint_path: str | None = None,
        *,
        lora_rank: int = 8,
        num_shared_experts: int = 9,
        lora_alpha: float = 32.0,
        adapter_dropout: float = 0.05,
        routing_group_size: int = 512,
        grad_checkpointing: bool = True,
    ):
        super().__init__()
        if dino_backbones is None:
            raise ImportError("Cannot import dinov3.hub.backbones - make sure dinov3 is available")
        if model_name != "dinov3_vitl16":
            raise ValueError("113_mtlora_vision currently supports only dinov3_vitl16.")

        self.model_name = str(model_name)
        self.image_size = int(image_size)
        self.lora_rank = int(lora_rank)
        self.num_shared_experts = int(num_shared_experts)
        self.lora_alpha = float(lora_alpha)
        self.adapter_dropout = float(adapter_dropout)
        self.requested_routing_group_size = int(routing_group_size)
        self.grad_checkpointing = bool(grad_checkpointing)

        self.backbone = getattr(dino_backbones, self.model_name)(pretrained=False)
        if checkpoint_path:
            try:
                state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            except TypeError:
                state = torch.load(checkpoint_path, map_location="cpu")
            self.backbone.load_state_dict(state, strict=False)

        self.embed_dim = int(getattr(self.backbone, "embed_dim", 0) or getattr(self.backbone, "num_features", 0))
        if self.embed_dim <= 0:
            with torch.no_grad():
                dummy = torch.randn(1, 3, self.image_size, self.image_size)
                tokens = self.backbone.forward_features(dummy)["x_norm_patchtokens"]
                self.embed_dim = int(tokens.shape[-1])

        patch_size = self.backbone.patch_size
        if isinstance(patch_size, (tuple, list)):
            self.patch_size = (int(patch_size[0]), int(patch_size[1]))
        else:
            self.patch_size = (int(patch_size), int(patch_size))

        self.wrapped_blocks = nn.ModuleList()
        self.actual_routing_group_size = self.requested_routing_group_size
        self._setup_adapters()
        self._freeze_backbone()

    def _setup_adapters(self) -> None:
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            raise ValueError("DINOv3 backbone does not expose 'blocks'")
        for block in blocks:
            adapter = SharedExpertBlockAdapter(
                dim=self.embed_dim,
                rank=self.lora_rank,
                num_shared_experts=self.num_shared_experts,
                task_names=self.TASK_NAMES,
                lora_alpha=self.lora_alpha,
                dropout=self.adapter_dropout,
                routing_group_size=self.requested_routing_group_size,
            )
            self.actual_routing_group_size = adapter.routing_group_size
            wrapper = ParallelFFNBlockWrapper(block, adapter, grad_checkpointing=self.grad_checkpointing)
            self.wrapped_blocks.append(wrapper)

    def _freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for wrapper in self.wrapped_blocks:
            for parameter in wrapper.adapter.parameters():
                parameter.requires_grad = True

    def export_config(self) -> dict:
        return {
            "model_name": self.model_name,
            "image_size": self.image_size,
            "lora_rank": self.lora_rank,
            "num_shared_experts": self.num_shared_experts,
            "lora_alpha": self.lora_alpha,
            "adapter_dropout": self.adapter_dropout,
            "routing_group_size": self.actual_routing_group_size,
            "grad_checkpointing": self.grad_checkpointing,
        }

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
        x, (height_tokens, width_tokens) = self.backbone.prepare_tokens_with_masks(x, masks=None)

        rope = None
        if hasattr(self.backbone, "rope_embed") and self.backbone.rope_embed is not None:
            rope = self.backbone.rope_embed(H=height_tokens, W=width_tokens)

        for wrapped_block in self.wrapped_blocks:
            x = wrapped_block(x, task_name=task_name, rope=rope)

        n_prefix = 1 + int(getattr(self.backbone, "n_storage_tokens", 0))
        if getattr(self.backbone, "untie_cls_and_patch_norms", False):
            patch_tokens = self.backbone.norm(x[:, n_prefix:])
        else:
            patch_tokens = self.backbone.norm(x)[:, n_prefix:]

        batch_size, num_tokens, channels = patch_tokens.shape
        expected_tokens = int(height_tokens * width_tokens)
        if expected_tokens != int(num_tokens):
            raise ValueError(f"Token mismatch: expected {expected_tokens}, got {int(num_tokens)}")

        feature_map = patch_tokens.reshape(batch_size, height_tokens, width_tokens, channels).permute(0, 3, 1, 2)
        return feature_map, meta

    def forward_features(self, x: torch.Tensor, *, task_name: str) -> dict:
        feature_map, _ = self.forward_task_features(x, task_name=task_name)
        batch_size, channels, height_tokens, width_tokens = feature_map.shape
        patch_tokens = feature_map.permute(0, 2, 3, 1).reshape(batch_size, height_tokens * width_tokens, channels)
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

    def forward_joint_train(
        self,
        *,
        det_images,
        det_targets,
        seg_images: torch.Tensor,
        seg_masks: torch.Tensor,
        cnt_images: torch.Tensor,
        cnt_density: torch.Tensor,
        cnt_count_loss_weight: float = 1.0,
    ) -> dict:
        det_loss = sum(self.forward_det(det_images, det_targets).values())
        seg_logits = self.forward_seg(seg_images)
        seg_loss = F.cross_entropy(seg_logits, seg_masks)
        pred_density, pred_counts = self.forward_cnt(cnt_images)
        gt_counts = cnt_density.flatten(2).sum(dim=2)
        density_loss = F.mse_loss(pred_density, cnt_density, reduction="sum") / cnt_images.size(0)
        count_l1 = F.l1_loss(pred_counts, gt_counts)
        cnt_loss = density_loss + float(cnt_count_loss_weight) * count_l1
        return {
            "det_loss": det_loss,
            "seg_loss": seg_loss,
            "cnt_loss": cnt_loss,
            "pred_density": pred_density,
            "pred_counts": pred_counts,
            "gt_counts": gt_counts,
        }

    def forward(self, mode: str, *args, **kwargs):
        mode = str(mode).lower()
        if mode == "det":
            return self.forward_det(*args, **kwargs)
        if mode == "seg":
            return self.forward_seg(*args, **kwargs)
        if mode == "cnt":
            return self.forward_cnt(*args, **kwargs)
        if mode in {"joint_train", "multitask_train"}:
            return self.forward_joint_train(*args, **kwargs)
        raise ValueError(f"Unknown forward mode: {mode}")

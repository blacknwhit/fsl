from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint

try:
    from dinov3.hub import backbones as dino_backbones
except ImportError:
    dino_backbones = None

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign


@dataclass(frozen=True)
class TaskFeatureMeta:
    original_size: tuple[int, int]
    padded_size: tuple[int, int]
    pad_h: int
    pad_w: int


class MTLoRALinear(nn.Module):
    def __init__(
        self,
        linear: nn.Module,
        *,
        shared_rank: int,
        task_rank: int,
        tasks: Sequence[str] | None,
        task_getter: Callable[[], Optional[str]],
        shared_scale: float = 4.0,
        task_scale: float = 4.0,
        dropout: float = 0.05,
        trainable_scale_shared: bool = False,
        trainable_scale_per_task: bool = False,
        shared_mode: str = "matrix",
    ):
        super().__init__()
        if not hasattr(linear, "weight"):
            raise TypeError(f"MTLoRALinear expects a weight-bearing module, got {type(linear).__name__}")

        mode = str(shared_mode).lower()
        if mode == "add":
            mode = "addition"
        if mode not in {"matrix", "matrixv2", "addition", "lora_only"}:
            raise ValueError(f"Unsupported MTLoRA shared_mode: {shared_mode}")

        self.linear = linear
        self.task_getter = task_getter
        self.shared_mode = mode
        self.shared_rank = int(shared_rank)
        self.task_rank = int(task_rank)
        self.tasks = tuple(str(task) for task in tasks) if tasks else tuple()
        self.in_features = int(linear.weight.shape[1])
        self.out_features = int(linear.weight.shape[0])
        self.lora_dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity()

        if self.shared_rank > 0:
            self.lora_shared_A = nn.Parameter(torch.empty(self.shared_rank, self.in_features))
            self.lora_shared_B = nn.Parameter(torch.zeros(self.out_features, self.shared_rank))
            if trainable_scale_shared:
                self.lora_shared_scale = nn.Parameter(torch.tensor(float(shared_scale)))
            else:
                self.lora_shared_scale = float(shared_scale)
        else:
            self.lora_shared_A = None
            self.lora_shared_B = None
            self.lora_shared_scale = float(shared_scale)

        if self.tasks and self.task_rank > 0:
            self.lora_tasks_A = nn.ParameterDict(
                {
                    task: nn.Parameter(torch.empty(self.task_rank, self.in_features))
                    for task in self.tasks
                }
            )
            self.lora_tasks_B = nn.ParameterDict(
                {
                    task: nn.Parameter(torch.zeros(self.out_features, self.task_rank))
                    for task in self.tasks
                }
            )
            if trainable_scale_per_task:
                self.lora_task_scale = nn.ParameterDict(
                    {task: nn.Parameter(torch.tensor(float(task_scale))) for task in self.tasks}
                )
            else:
                self.lora_task_scale = {task: float(task_scale) for task in self.tasks}
        else:
            self.lora_tasks_A = None
            self.lora_tasks_B = None
            self.lora_task_scale = {}

        self.reset_parameters()
        self.freeze_base_parameters()

    def reset_parameters(self) -> None:
        if self.lora_shared_A is not None:
            nn.init.kaiming_uniform_(self.lora_shared_A, a=5**0.5)
            nn.init.zeros_(self.lora_shared_B)
        if self.lora_tasks_A is not None:
            for task in self.tasks:
                nn.init.kaiming_uniform_(self.lora_tasks_A[task], a=5**0.5)
                nn.init.zeros_(self.lora_tasks_B[task])

    def freeze_base_parameters(self) -> None:
        for parameter in self.linear.parameters():
            parameter.requires_grad = False

    def set_trainable(self, bias_mode: str = "none") -> None:
        self.freeze_base_parameters()
        if self.lora_shared_A is not None:
            self.lora_shared_A.requires_grad = True
            self.lora_shared_B.requires_grad = True
            if isinstance(self.lora_shared_scale, nn.Parameter):
                self.lora_shared_scale.requires_grad = True
        if self.lora_tasks_A is not None:
            for task in self.tasks:
                self.lora_tasks_A[task].requires_grad = True
                self.lora_tasks_B[task].requires_grad = True
            if isinstance(self.lora_task_scale, nn.ParameterDict):
                for task in self.tasks:
                    self.lora_task_scale[task].requires_grad = True
        if bias_mode in {"all", "lora_only"}:
            bias = getattr(self.linear, "bias", None)
            if isinstance(bias, nn.Parameter):
                bias.requires_grad = True

    def _apply_update(self, x: torch.Tensor, A: nn.Parameter, B: nn.Parameter, scale) -> torch.Tensor:
        dropped = self.lora_dropout(x)
        after_a = F.linear(dropped, A)
        return F.linear(after_a, B) * scale

    def _get_task_scale(self, task_name: str):
        if isinstance(self.lora_task_scale, nn.ParameterDict):
            return self.lora_task_scale[task_name]
        return float(self.lora_task_scale[task_name])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)

        shared_delta = 0.0
        if self.lora_shared_A is not None and self.lora_shared_B is not None:
            shared_delta = self._apply_update(
                x,
                self.lora_shared_A,
                self.lora_shared_B,
                self.lora_shared_scale,
            )

        task_name = self.task_getter()
        if (
            task_name
            and self.lora_tasks_A is not None
            and self.lora_tasks_B is not None
            and task_name in self.lora_tasks_A
        ):
            task_delta = self._apply_update(
                x,
                self.lora_tasks_A[task_name],
                self.lora_tasks_B[task_name],
                self._get_task_scale(task_name),
            )
            if self.shared_mode == "matrix":
                return base + task_delta
            return base + shared_delta + task_delta

        return base + shared_delta


class SharedDinoV3Backbone(nn.Module):
    TASK_NAMES = ("det", "seg", "cnt")

    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
        image_size: int = 448,
        checkpoint_path: str | None = None,
        *,
        mtlora_shared_rank: int = 16,
        mtlora_task_rank: int = 2,
        mtlora_shared_scale: float = 4.0,
        mtlora_task_scale: float = 4.0,
        mtlora_dropout: float = 0.05,
        mtlora_bias: str = "none",
        mtlora_shared_mode: str = "matrix",
        mtlora_specialize_blocks: Sequence[int] = (5, 11, 17, 23),
        mtlora_trainable_scale_shared: bool = False,
        mtlora_trainable_scale_per_task: bool = False,
        mtlora_intermediate_specialization: bool = False,
        mtlora_qkv_enabled: bool = True,
        mtlora_proj_enabled: bool = True,
        mtlora_fc1_enabled: bool = True,
        mtlora_fc2_enabled: bool = True,
        grad_checkpointing: bool = True,
    ):
        super().__init__()
        if dino_backbones is None:
            raise ImportError("Cannot import dinov3.hub.backbones - make sure dinov3 is available")
        if model_name != "dinov3_vitl16":
            raise ValueError("113_mtlora currently supports only dinov3_vitl16.")

        self.model_name = str(model_name)
        self.image_size = int(image_size)
        self.mtlora_shared_rank = int(mtlora_shared_rank)
        self.mtlora_task_rank = int(mtlora_task_rank)
        self.mtlora_shared_scale = float(mtlora_shared_scale)
        self.mtlora_task_scale = float(mtlora_task_scale)
        self.mtlora_dropout = float(mtlora_dropout)
        self.mtlora_bias = str(mtlora_bias)
        self.mtlora_shared_mode = str(mtlora_shared_mode)
        self.mtlora_specialize_blocks = tuple(int(index) for index in mtlora_specialize_blocks)
        self.mtlora_trainable_scale_shared = bool(mtlora_trainable_scale_shared)
        self.mtlora_trainable_scale_per_task = bool(mtlora_trainable_scale_per_task)
        self.mtlora_intermediate_specialization = bool(mtlora_intermediate_specialization)
        self.mtlora_qkv_enabled = bool(mtlora_qkv_enabled)
        self.mtlora_proj_enabled = bool(mtlora_proj_enabled)
        self.mtlora_fc1_enabled = bool(mtlora_fc1_enabled)
        self.mtlora_fc2_enabled = bool(mtlora_fc2_enabled)
        self.grad_checkpointing = bool(grad_checkpointing)
        self._active_task_name: Optional[str] = None

        self.backbone = getattr(dino_backbones, self.model_name)(pretrained=False)
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

        self._wrap_backbone_with_mtlora()
        self._configure_trainable_parameters()

    def export_config(self) -> dict:
        return {
            "model_name": self.model_name,
            "image_size": self.image_size,
            "mtlora_shared_rank": self.mtlora_shared_rank,
            "mtlora_task_rank": self.mtlora_task_rank,
            "mtlora_shared_scale": self.mtlora_shared_scale,
            "mtlora_task_scale": self.mtlora_task_scale,
            "mtlora_dropout": self.mtlora_dropout,
            "mtlora_bias": self.mtlora_bias,
            "mtlora_shared_mode": self.mtlora_shared_mode,
            "mtlora_specialize_blocks": list(self.mtlora_specialize_blocks),
            "mtlora_trainable_scale_shared": self.mtlora_trainable_scale_shared,
            "mtlora_trainable_scale_per_task": self.mtlora_trainable_scale_per_task,
            "mtlora_intermediate_specialization": self.mtlora_intermediate_specialization,
            "mtlora_qkv_enabled": self.mtlora_qkv_enabled,
            "mtlora_proj_enabled": self.mtlora_proj_enabled,
            "mtlora_fc1_enabled": self.mtlora_fc1_enabled,
            "mtlora_fc2_enabled": self.mtlora_fc2_enabled,
            "grad_checkpointing": self.grad_checkpointing,
        }

    def _get_active_task_name(self) -> Optional[str]:
        return self._active_task_name

    def _validate_block_layout(self, block: nn.Module, block_index: int) -> None:
        required_paths = (
            ("attn", "qkv"),
            ("attn", "proj"),
            ("mlp", "fc1"),
            ("mlp", "fc2"),
        )
        for parent_name, child_name in required_paths:
            parent = getattr(block, parent_name, None)
            if parent is None:
                raise ValueError(f"DINOv3 block {block_index} is missing '{parent_name}'")
            child = getattr(parent, child_name, None)
            if child is None:
                raise ValueError(f"DINOv3 block {block_index} is missing '{parent_name}.{child_name}'")
            if not hasattr(child, "weight"):
                raise TypeError(
                    f"DINOv3 block {block_index} layer '{parent_name}.{child_name}' has no weight attribute"
                )

    def _replace_linear(
        self,
        parent: nn.Module,
        attr_name: str,
        *,
        shared_rank: int,
        task_rank: int,
        tasks: Sequence[str] | None,
    ) -> None:
        original = getattr(parent, attr_name)
        wrapped = MTLoRALinear(
            original,
            shared_rank=shared_rank,
            task_rank=task_rank,
            tasks=tasks,
            task_getter=self._get_active_task_name,
            shared_scale=self.mtlora_shared_scale,
            task_scale=self.mtlora_task_scale,
            dropout=self.mtlora_dropout,
            trainable_scale_shared=self.mtlora_trainable_scale_shared,
            trainable_scale_per_task=self.mtlora_trainable_scale_per_task,
            shared_mode=self.mtlora_shared_mode,
        )
        setattr(parent, attr_name, wrapped)

    def _wrap_backbone_with_mtlora(self) -> None:
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            raise ValueError("DINOv3 backbone does not expose 'blocks'")
        if len(blocks) != 24:
            raise ValueError(f"Expected 24 transformer blocks for dinov3_vitl16, got {len(blocks)}")

        specialize_set = set(self.mtlora_specialize_blocks)
        for block_index, block in enumerate(blocks):
            self._validate_block_layout(block, block_index)
            enable_task_specialization = self.mtlora_intermediate_specialization or block_index in specialize_set
            task_names = self.TASK_NAMES if enable_task_specialization else None

            if self.mtlora_qkv_enabled:
                self._replace_linear(
                    block.attn,
                    "qkv",
                    shared_rank=self.mtlora_shared_rank,
                    task_rank=0,
                    tasks=None,
                )
            if self.mtlora_proj_enabled:
                self._replace_linear(
                    block.attn,
                    "proj",
                    shared_rank=self.mtlora_shared_rank,
                    task_rank=self.mtlora_task_rank,
                    tasks=task_names,
                )
            if self.mtlora_fc1_enabled:
                self._replace_linear(
                    block.mlp,
                    "fc1",
                    shared_rank=self.mtlora_shared_rank,
                    task_rank=self.mtlora_task_rank,
                    tasks=task_names,
                )
            if self.mtlora_fc2_enabled:
                self._replace_linear(
                    block.mlp,
                    "fc2",
                    shared_rank=self.mtlora_shared_rank,
                    task_rank=self.mtlora_task_rank,
                    tasks=task_names,
                )

    def _configure_trainable_parameters(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        bias_mode = self.mtlora_bias.lower()
        if bias_mode not in {"none", "all", "lora_only"}:
            raise ValueError(f"Unsupported --mtlora-bias: {self.mtlora_bias}")

        for module in self.backbone.modules():
            if isinstance(module, MTLoRALinear):
                module.set_trainable("lora_only" if bias_mode == "lora_only" else "none")

        if bias_mode == "all":
            for name, parameter in self.backbone.named_parameters():
                if name.endswith("bias"):
                    parameter.requires_grad = True

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

    def _call_block(self, block: nn.Module, x: torch.Tensor, rope, *, task_name: Optional[str]):
        previous_task_name = self._active_task_name
        self._active_task_name = task_name
        try:
            try:
                return block(x, rope=rope)
            except TypeError:
                try:
                    return block(x, rope)
                except TypeError:
                    return block(x)
        finally:
            self._active_task_name = previous_task_name

    def _apply_block(self, block: nn.Module, x: torch.Tensor, rope, *, task_name: Optional[str]):
        if self.training and self.grad_checkpointing:
            def _forward(inp: torch.Tensor) -> torch.Tensor:
                return self._call_block(block, inp, rope, task_name=task_name)

            return activation_checkpoint(_forward, x, use_reentrant=False)
        return self._call_block(block, x, rope, task_name=task_name)

    def forward_task_features(self, x: torch.Tensor, *, task_name: str) -> tuple[torch.Tensor, TaskFeatureMeta]:
        if task_name not in self.TASK_NAMES:
            raise ValueError(f"Unknown task_name: {task_name}")

        x, meta = self._pad_to_patch(x)
        self._active_task_name = task_name
        try:
            x, (height_tokens, width_tokens) = self.backbone.prepare_tokens_with_masks(x, masks=None)

            rope = None
            if hasattr(self.backbone, "rope_embed") and self.backbone.rope_embed is not None:
                rope = self.backbone.rope_embed(H=height_tokens, W=width_tokens)

            for block in self.backbone.blocks:
                x = self._apply_block(block, x, rope, task_name=task_name)

            n_prefix = 1 + int(getattr(self.backbone, "n_storage_tokens", 0))
            if getattr(self.backbone, "untie_cls_and_patch_norms", False):
                patch_tokens = self.backbone.norm(x[:, n_prefix:])
            else:
                patch_tokens = self.backbone.norm(x)[:, n_prefix:]

            batch_size, num_tokens, channels = patch_tokens.shape
            expected_tokens = int(height_tokens * width_tokens)
            if expected_tokens != int(num_tokens):
                raise ValueError(
                    f"Token mismatch: expected {expected_tokens}, got {int(num_tokens)} for task '{task_name}'"
                )
            feature_map = patch_tokens.reshape(batch_size, height_tokens, width_tokens, channels).permute(0, 3, 1, 2)
            return feature_map, meta
        finally:
            self._active_task_name = None

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

    def forward(self, mode: str, *args, **kwargs):
        mode = str(mode).lower()
        if mode == "det":
            return self.forward_det(*args, **kwargs)
        if mode == "seg":
            return self.forward_seg(*args, **kwargs)
        if mode == "cnt":
            return self.forward_cnt(*args, **kwargs)
        raise ValueError(f"Unknown forward mode: {mode}")

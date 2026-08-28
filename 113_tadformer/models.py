from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional, Sequence

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


class TaskAdaptiveFilter(nn.Module):
    def __init__(self, rank: int, task_names: Sequence[str], kernel_size: int = 3):
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(f"TaskAdaptiveFilter kernel_size must be a positive odd integer, got {kernel_size}")
        self.rank = int(rank)
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.filters = nn.ModuleDict(
            {
                str(task): nn.Conv2d(
                    self.rank,
                    self.rank,
                    kernel_size=self.kernel_size,
                    padding=self.padding,
                    groups=self.rank,
                    bias=True,
                )
                for task in task_names
            }
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for conv in self.filters.values():
            nn.init.zeros_(conv.weight)
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x: torch.Tensor, *, task_name: str) -> torch.Tensor:
        if task_name not in self.filters:
            raise KeyError(f"Unknown task_name for TaskAdaptiveFilter: {task_name}")
        return self.filters[task_name](x)


class TaskPromptUpdater(nn.Module):
    def __init__(self, dim: int, rank: int = 64, dropout: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.rank = int(rank)
        if self.rank <= 0:
            raise ValueError(f"TaskPromptUpdater rank must be positive, got {rank}")
        self.prompt_norm = nn.LayerNorm(self.dim)
        self.patch_norm = nn.LayerNorm(self.dim)
        self.prompt_q = nn.Linear(self.dim, self.rank, bias=False)
        self.patch_k = nn.Linear(self.dim, self.rank, bias=False)
        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.scale = math.sqrt(float(self.rank))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.prompt_q.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.patch_k.weight, a=5**0.5)

    def forward(self, prompts: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        if prompts.numel() == 0 or patch_tokens.numel() == 0:
            return prompts

        prompt_states = self.prompt_norm(prompts)
        patch_states = self.patch_norm(patch_tokens)
        attn_logits = torch.matmul(
            self.prompt_q(prompt_states),
            self.patch_k(patch_states).transpose(1, 2),
        ) / max(self.scale, 1e-6)
        attn = torch.softmax(attn_logits, dim=-1)
        context = torch.matmul(attn, patch_states)
        return prompts + self.dropout(context)


class TADLinear(nn.Module):
    def __init__(
        self,
        linear: nn.Module,
        *,
        rank: int,
        scale: float,
        dropout: float,
        task_getter: Callable[[], Optional[str]],
        hw_getter: Callable[[], Optional[tuple[int, int]]],
        prompt_getter: Callable[[], Optional[torch.Tensor]],
        task_names: Sequence[str] | None = None,
        filter_kernel: int = 3,
        use_task_filter: bool = False,
        layer_name: str = "",
        use_prompt_tpc: bool = False,
    ):
        super().__init__()
        if not hasattr(linear, "weight"):
            raise TypeError(f"TADLinear expects a weight-bearing module, got {type(linear).__name__}")

        self.linear = linear
        self.rank = int(rank)
        self.scale = float(scale)
        self.task_getter = task_getter
        self.hw_getter = hw_getter
        self.prompt_getter = prompt_getter
        self.task_names = tuple(str(task) for task in (task_names or ()))
        self.use_task_filter = bool(use_task_filter and self.task_names)
        self.layer_name = str(layer_name)
        self.use_prompt_tpc = bool(use_prompt_tpc and self.task_names and self.layer_name == "proj")
        self.in_features = int(linear.weight.shape[1])
        self.out_features = int(linear.weight.shape[0])
        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity()

        if self.rank <= 0:
            self.lora_shared_A = None
            self.lora_shared_B = None
        else:
            self.lora_shared_A = nn.Parameter(torch.empty(self.rank, self.in_features))
            self.lora_shared_B = nn.Parameter(torch.zeros(self.out_features, self.rank))

        self.task_filters = (
            TaskAdaptiveFilter(self.rank, self.task_names, kernel_size=int(filter_kernel))
            if self.rank > 0 and self.use_task_filter
            else None
        )
        self.tpc_prompt_norm = nn.LayerNorm(self.in_features) if self.use_prompt_tpc else None
        self.tpc_patch_norm = nn.LayerNorm(self.in_features) if self.use_prompt_tpc else None

        self.reset_parameters()
        self.freeze_base_parameters()

    def reset_parameters(self) -> None:
        if self.lora_shared_A is not None:
            nn.init.kaiming_uniform_(self.lora_shared_A, a=5**0.5)
        if self.lora_shared_B is not None:
            nn.init.zeros_(self.lora_shared_B)

    def freeze_base_parameters(self) -> None:
        for parameter in self.linear.parameters():
            parameter.requires_grad = False

    def set_trainable(self) -> None:
        self.freeze_base_parameters()
        if self.lora_shared_A is not None:
            self.lora_shared_A.requires_grad = True
        if self.lora_shared_B is not None:
            self.lora_shared_B.requires_grad = True
        if self.task_filters is not None:
            for parameter in self.task_filters.parameters():
                parameter.requires_grad = True
        if self.tpc_prompt_norm is not None:
            for parameter in self.tpc_prompt_norm.parameters():
                parameter.requires_grad = True
        if self.tpc_patch_norm is not None:
            for parameter in self.tpc_patch_norm.parameters():
                parameter.requires_grad = True

    def _project_to_rank(self, x: torch.Tensor) -> torch.Tensor:
        if self.lora_shared_A is None:
            return x.new_zeros((*x.shape[:-1], 0))
        dropped = self.dropout(x)
        return F.linear(dropped, self.lora_shared_A)

    def _project_from_rank(self, x_rank: torch.Tensor) -> torch.Tensor:
        if self.lora_shared_B is None:
            return x_rank.new_zeros((*x_rank.shape[:-1], self.out_features))
        return F.linear(x_rank, self.lora_shared_B) * self.scale

    def _shared_delta(self, x: torch.Tensor) -> torch.Tensor:
        return self._project_from_rank(self._project_to_rank(x))

    def _apply_prompt_condition(self, patch_input: torch.Tensor, prompt_tokens: Optional[torch.Tensor]) -> torch.Tensor:
        if (
            not self.use_prompt_tpc
            or prompt_tokens is None
            or self.tpc_prompt_norm is None
            or self.tpc_patch_norm is None
            or prompt_tokens.numel() == 0
            or patch_input.numel() == 0
        ):
            return patch_input

        prompt_tokens = prompt_tokens.to(device=patch_input.device, dtype=patch_input.dtype)
        prompt_states = self.tpc_prompt_norm(prompt_tokens)
        patch_states = self.tpc_patch_norm(patch_input)
        attn_logits = torch.matmul(prompt_states, patch_states.transpose(1, 2)) / math.sqrt(float(self.in_features))
        attn = torch.softmax(attn_logits, dim=-1)
        prompt_context = torch.matmul(attn.transpose(1, 2), prompt_states)
        prompt_context = prompt_context / max(int(prompt_states.shape[1]), 1)
        return patch_input + prompt_context

    def _task_delta(self, x: torch.Tensor, *, task_name: str, hw_shape: tuple[int, int]) -> torch.Tensor:
        if self.task_filters is None and not self.use_prompt_tpc:
            return self._shared_delta(x)

        batch_size, num_tokens, _ = x.shape
        height, width = hw_shape
        patch_tokens = int(height) * int(width)
        prefix_tokens = int(num_tokens) - patch_tokens
        if prefix_tokens < 0:
            raise ValueError(
                f"TADLinear token mismatch for task '{task_name}': num_tokens={num_tokens}, hw={hw_shape}"
            )

        patch_input = x[:, prefix_tokens:, :]
        if patch_input.shape[1] != patch_tokens:
            raise ValueError(
                f"TADLinear patch token mismatch for task '{task_name}': "
                f"patch_tokens={patch_input.shape[1]}, expected={patch_tokens}"
            )

        patch_input = self._apply_prompt_condition(patch_input, self.prompt_getter())
        patch_rank = self._project_to_rank(patch_input)
        rank = int(patch_rank.shape[-1])

        if self.task_filters is not None:
            patch_rank_2d = patch_rank.reshape(batch_size, height, width, rank).permute(0, 3, 1, 2)
            patch_rank_2d = self.task_filters(patch_rank_2d, task_name=task_name)
            patch_rank = patch_rank_2d.permute(0, 2, 3, 1).reshape(batch_size, patch_tokens, rank)

        if prefix_tokens > 0:
            prefix_rank = patch_rank.new_zeros((batch_size, prefix_tokens, rank))
            x_rank = torch.cat((prefix_rank, patch_rank), dim=1)
        else:
            x_rank = patch_rank
        return self._project_from_rank(x_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        if self.rank <= 0:
            return base

        shared_delta = self._shared_delta(x)
        task_name = self.task_getter()
        hw_shape = self.hw_getter()
        if (self.use_task_filter or self.use_prompt_tpc) and task_name and hw_shape is not None:
            return base + shared_delta + self._task_delta(x, task_name=task_name, hw_shape=hw_shape)
        return base + shared_delta


class SharedDinoV3Backbone(nn.Module):
    TASK_NAMES = ("det", "seg", "cnt")

    def __init__(
        self,
        model_name: str = "dinov3_vitl16",
        image_size: int = 448,
        checkpoint_path: str | None = None,
        *,
        tad_rank: int = 16,
        tad_scale: float = 4.0,
        tad_dropout: float = 0.05,
        tad_specialize_blocks: Sequence[int] = (5, 11, 17, 23),
        tad_filter_kernel: int = 3,
        tad_qkv_enabled: bool = True,
        tad_proj_enabled: bool = True,
        tad_fc1_enabled: bool = True,
        tad_fc2_enabled: bool = True,
        tad_prompt_enabled: bool = True,
        tad_prompt_len: int = 1,
        tad_tpc_enabled: bool = True,
        tad_prompt_update_rank: int = 64,
        grad_checkpointing: bool = True,
    ):
        super().__init__()
        if dino_backbones is None:
            raise ImportError("Cannot import dinov3.hub.backbones - make sure dinov3 is available")
        if model_name != "dinov3_vitl16":
            raise ValueError("113_tadformer currently supports only dinov3_vitl16.")

        self.model_name = str(model_name)
        self.image_size = int(image_size)
        self.tad_rank = int(tad_rank)
        self.tad_scale = float(tad_scale)
        self.tad_dropout = float(tad_dropout)
        self.tad_specialize_blocks = tuple(int(index) for index in tad_specialize_blocks)
        self.tad_filter_kernel = int(tad_filter_kernel)
        self.tad_qkv_enabled = bool(tad_qkv_enabled)
        self.tad_proj_enabled = bool(tad_proj_enabled)
        self.tad_fc1_enabled = bool(tad_fc1_enabled)
        self.tad_fc2_enabled = bool(tad_fc2_enabled)
        self.tad_prompt_enabled = bool(tad_prompt_enabled)
        self.tad_prompt_len = int(tad_prompt_len)
        self.tad_tpc_enabled = bool(tad_tpc_enabled)
        self.tad_prompt_update_rank = int(tad_prompt_update_rank)
        self.grad_checkpointing = bool(grad_checkpointing)
        self._active_task_name: Optional[str] = None
        self._active_hw_shape: Optional[tuple[int, int]] = None
        self._active_task_prompts: Optional[torch.Tensor] = None

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

        self._build_prompt_modules()
        self._wrap_backbone_with_tad()
        self._configure_trainable_parameters()

    def export_config(self) -> dict:
        return {
            "model_name": self.model_name,
            "image_size": self.image_size,
            "tad_rank": self.tad_rank,
            "tad_scale": self.tad_scale,
            "tad_dropout": self.tad_dropout,
            "tad_specialize_blocks": list(self.tad_specialize_blocks),
            "tad_filter_kernel": self.tad_filter_kernel,
            "tad_qkv_enabled": self.tad_qkv_enabled,
            "tad_proj_enabled": self.tad_proj_enabled,
            "tad_fc1_enabled": self.tad_fc1_enabled,
            "tad_fc2_enabled": self.tad_fc2_enabled,
            "tad_prompt_enabled": self.tad_prompt_enabled,
            "tad_prompt_len": self.tad_prompt_len,
            "tad_tpc_enabled": self.tad_tpc_enabled,
            "tad_prompt_update_rank": self.tad_prompt_update_rank,
            "grad_checkpointing": self.grad_checkpointing,
        }

    def _get_active_task_name(self) -> Optional[str]:
        return self._active_task_name

    def _get_active_hw_shape(self) -> Optional[tuple[int, int]]:
        return self._active_hw_shape

    def _get_active_task_prompts(self) -> Optional[torch.Tensor]:
        return self._active_task_prompts

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

    def _build_prompt_modules(self) -> None:
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            raise ValueError("DINOv3 backbone does not expose 'blocks'")
        if len(blocks) != 24:
            raise ValueError(f"Expected 24 transformer blocks for dinov3_vitl16, got {len(blocks)}")
        if not self.tad_prompt_enabled:
            return
        if self.tad_prompt_len <= 0:
            raise ValueError(f"tad_prompt_len must be positive when prompts are enabled, got {self.tad_prompt_len}")
        if self.tad_prompt_update_rank <= 0:
            raise ValueError(
                f"tad_prompt_update_rank must be positive when prompts are enabled, got {self.tad_prompt_update_rank}"
            )

        prompt_bank = nn.Parameter(torch.empty(len(self.TASK_NAMES), self.tad_prompt_len, self.embed_dim))
        nn.init.normal_(prompt_bank, mean=0.0, std=0.02)
        setattr(self.backbone, "tad_task_prompts", prompt_bank)
        setattr(
            self.backbone,
            "tad_prompt_updaters",
            nn.ModuleList(
                [
                    TaskPromptUpdater(self.embed_dim, rank=self.tad_prompt_update_rank, dropout=self.tad_dropout)
                    for _ in range(len(blocks))
                ]
            ),
        )

    def _replace_linear(self, parent: nn.Module, attr_name: str, *, use_task_filter: bool) -> None:
        original = getattr(parent, attr_name)
        wrapped = TADLinear(
            original,
            rank=self.tad_rank,
            scale=self.tad_scale,
            dropout=self.tad_dropout,
            task_getter=self._get_active_task_name,
            hw_getter=self._get_active_hw_shape,
            prompt_getter=self._get_active_task_prompts,
            task_names=self.TASK_NAMES if use_task_filter else None,
            filter_kernel=self.tad_filter_kernel,
            use_task_filter=use_task_filter,
            layer_name=attr_name,
            use_prompt_tpc=bool(use_task_filter and attr_name == "proj" and self.tad_prompt_enabled and self.tad_tpc_enabled),
        )
        setattr(parent, attr_name, wrapped)

    def _wrap_backbone_with_tad(self) -> None:
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            raise ValueError("DINOv3 backbone does not expose 'blocks'")
        if len(blocks) != 24:
            raise ValueError(f"Expected 24 transformer blocks for dinov3_vitl16, got {len(blocks)}")

        specialize_set = set(self.tad_specialize_blocks)
        for block_index, block in enumerate(blocks):
            self._validate_block_layout(block, block_index)
            use_task_filter = block_index in specialize_set
            if self.tad_qkv_enabled:
                self._replace_linear(block.attn, "qkv", use_task_filter=False)
            if self.tad_proj_enabled:
                self._replace_linear(block.attn, "proj", use_task_filter=use_task_filter)
            if self.tad_fc1_enabled:
                self._replace_linear(block.mlp, "fc1", use_task_filter=use_task_filter)
            if self.tad_fc2_enabled:
                self._replace_linear(block.mlp, "fc2", use_task_filter=use_task_filter)

    def _configure_trainable_parameters(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        for module in self.backbone.modules():
            if isinstance(module, TADLinear):
                module.set_trainable()
            elif isinstance(module, TaskPromptUpdater):
                for parameter in module.parameters():
                    parameter.requires_grad = True

        prompt_bank = getattr(self.backbone, "tad_task_prompts", None)
        if isinstance(prompt_bank, nn.Parameter):
            prompt_bank.requires_grad = True

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

    def _trainable(self) -> bool:
        return any(parameter.requires_grad for parameter in self.backbone.parameters())

    def _get_initial_task_prompts(
        self,
        *,
        task_name: str,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not self.tad_prompt_enabled:
            return None
        prompt_bank = getattr(self.backbone, "tad_task_prompts", None)
        if prompt_bank is None:
            return None
        task_index = self.TASK_NAMES.index(task_name)
        # Materialize a per-batch prompt tensor instead of keeping an expanded
        # view into the prompt bank. This avoids version-counter surprises when
        # the same prompt participates in checkpointed multi-task forwards.
        task_prompts = prompt_bank[task_index].unsqueeze(0).repeat(batch_size, 1, 1)
        return task_prompts.to(device=device, dtype=dtype)

    def _update_task_prompts(
        self,
        *,
        block_index: int,
        task_prompts: Optional[torch.Tensor],
        block_output: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if task_prompts is None:
            return None
        prompt_updaters = getattr(self.backbone, "tad_prompt_updaters", None)
        if prompt_updaters is None:
            return task_prompts
        n_prefix = 1 + int(getattr(self.backbone, "n_storage_tokens", 0))
        patch_tokens = block_output[:, n_prefix:, :]
        return prompt_updaters[block_index](task_prompts, patch_tokens)

    def _call_block(
        self,
        block: nn.Module,
        x: torch.Tensor,
        rope,
        *,
        task_name: Optional[str],
        hw_shape: Optional[tuple[int, int]],
        task_prompts: Optional[torch.Tensor],
    ):
        previous_task_name = self._active_task_name
        previous_hw_shape = self._active_hw_shape
        previous_task_prompts = self._active_task_prompts
        self._active_task_name = task_name
        self._active_hw_shape = hw_shape
        self._active_task_prompts = task_prompts
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
            self._active_hw_shape = previous_hw_shape
            self._active_task_prompts = previous_task_prompts

    def _apply_block(
        self,
        block: nn.Module,
        x: torch.Tensor,
        rope,
        *,
        task_name: Optional[str],
        hw_shape: Optional[tuple[int, int]],
        task_prompts: Optional[torch.Tensor],
        trainable: bool,
    ):
        if self.training and self.grad_checkpointing and trainable:
            if task_prompts is not None:

                def _forward(inp: torch.Tensor, prompts: torch.Tensor) -> torch.Tensor:
                    return self._call_block(
                        block,
                        inp,
                        rope,
                        task_name=task_name,
                        hw_shape=hw_shape,
                        task_prompts=prompts,
                    )

                return activation_checkpoint(_forward, x, task_prompts, use_reentrant=False)

            def _forward(inp: torch.Tensor) -> torch.Tensor:
                return self._call_block(
                    block,
                    inp,
                    rope,
                    task_name=task_name,
                    hw_shape=hw_shape,
                    task_prompts=None,
                )

            return activation_checkpoint(_forward, x, use_reentrant=False)
        return self._call_block(
            block,
            x,
            rope,
            task_name=task_name,
            hw_shape=hw_shape,
            task_prompts=task_prompts,
        )

    def forward_task_features(
        self,
        x: torch.Tensor,
        *,
        task_name: str,
        trainable_override: bool | None = None,
    ) -> tuple[torch.Tensor, TaskFeatureMeta]:
        if task_name not in self.TASK_NAMES:
            raise ValueError(f"Unknown task_name: {task_name}")

        x, meta = self._pad_to_patch(x)
        trainable = bool(trainable_override) if trainable_override is not None else self._trainable()
        self.backbone.train(self.training and trainable)
        with torch.set_grad_enabled(self.training and trainable):
            x, (height_tokens, width_tokens) = self.backbone.prepare_tokens_with_masks(x, masks=None)
            hw_shape = (int(height_tokens), int(width_tokens))
            task_prompts = self._get_initial_task_prompts(
                task_name=task_name,
                batch_size=int(x.shape[0]),
                device=x.device,
                dtype=x.dtype,
            )

            rope = None
            if hasattr(self.backbone, "rope_embed") and self.backbone.rope_embed is not None:
                rope = self.backbone.rope_embed(H=height_tokens, W=width_tokens)

            for block_index, block in enumerate(self.backbone.blocks):
                x = self._apply_block(
                    block,
                    x,
                    rope,
                    task_name=task_name,
                    hw_shape=hw_shape,
                    task_prompts=task_prompts,
                    trainable=trainable,
                )
                task_prompts = self._update_task_prompts(
                    block_index=block_index,
                    task_prompts=task_prompts,
                    block_output=x,
                )

            n_prefix = 1 + int(getattr(self.backbone, "n_storage_tokens", 0))
            if getattr(self.backbone, "untie_cls_and_patch_norms", False):
                patch_tokens = self.backbone.norm(x[:, n_prefix:])
            else:
                patch_tokens = self.backbone.norm(x)[:, n_prefix:]

            batch_size, num_tokens, channels = patch_tokens.shape
            expected_tokens = int(height_tokens) * int(width_tokens)
            if expected_tokens != int(num_tokens):
                raise ValueError(
                    f"Token mismatch: expected {expected_tokens}, got {int(num_tokens)} for task '{task_name}'"
                )
            feature_map = patch_tokens.reshape(batch_size, height_tokens, width_tokens, channels).permute(0, 3, 1, 2)
            return feature_map, meta

    def forward_features(self, x: torch.Tensor, *, task_name: str, trainable_override: bool | None = None) -> dict:
        feature_map, _ = self.forward_task_features(
            x,
            task_name=task_name,
            trainable_override=trainable_override,
        )
        batch_size, channels, height_tokens, width_tokens = feature_map.shape
        patch_tokens = feature_map.permute(0, 2, 3, 1).reshape(batch_size, height_tokens * width_tokens, channels)
        return {"x_norm_patchtokens": patch_tokens}


class _DetBackboneAdapter(nn.Module):
    def __init__(self, shared: SharedDinoV3Backbone, out_channels: int = 256, *, trainable_backbone: bool = True):
        super().__init__()
        self.shared = shared
        self.out_channels = int(out_channels)
        self.proj = nn.Conv2d(shared.embed_dim, self.out_channels, kernel_size=1)
        self.trainable_backbone = bool(trainable_backbone)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map, _ = self.shared.forward_task_features(
            x,
            task_name="det",
            trainable_override=self.trainable_backbone,
        )
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
        det_train_backbone: bool = True,
        seg_train_backbone: bool = True,
        cnt_train_backbone: bool = True,
    ):
        super().__init__()
        self.shared = shared
        self.image_size = int(image_size)
        self.det_out_channels = int(det_out_channels)
        self.det_num_classes = int(det_num_classes)
        self.seg_num_classes = int(seg_num_classes)
        self.cnt_num_classes = int(cnt_num_classes)
        self.det_train_backbone = bool(det_train_backbone)
        self.seg_train_backbone = bool(seg_train_backbone)
        self.cnt_train_backbone = bool(cnt_train_backbone)

        det_backbone = _DetBackboneAdapter(
            shared,
            out_channels=self.det_out_channels,
            trainable_backbone=self.det_train_backbone,
        )
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

    @staticmethod
    def _cnt_feat_with_scaled_backbone_grad(feat: torch.Tensor, mult: float) -> torch.Tensor:
        scale = float(mult)
        if scale == 1.0 or not feat.requires_grad:
            return feat
        feat_scaled = feat.clone()
        feat_scaled.register_hook(lambda grad: grad * scale)
        return feat_scaled

    def export_config(self) -> dict:
        config = self.shared.export_config()
        config.update(
            {
                "det_num_classes": self.det_num_classes,
                "seg_num_classes": self.seg_num_classes,
                "cnt_num_classes": self.cnt_num_classes,
                "det_out_channels": self.det_out_channels,
                "image_size": self.image_size,
                "det_train_backbone": bool(self.det_train_backbone),
                "seg_train_backbone": bool(self.seg_train_backbone),
                "cnt_train_backbone": bool(self.cnt_train_backbone),
            }
        )
        return config

    def forward_det(self, images, targets=None):
        return self.detector(images, targets)

    def forward_seg(self, x: torch.Tensor) -> torch.Tensor:
        feature_map, meta = self.shared.forward_task_features(
            x,
            task_name=self.TASK_ID_SEG,
            trainable_override=self.seg_train_backbone,
        )
        logits = self.seg_head(feature_map, meta.padded_size)
        return logits[:, :, : meta.original_size[0], : meta.original_size[1]]

    def forward_cnt(self, x: torch.Tensor, *, cnt_backbone_grad_mult: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map, meta = self.shared.forward_task_features(
            x,
            task_name=self.TASK_ID_CNT,
            trainable_override=self.cnt_train_backbone,
        )
        feature_map = self._cnt_feat_with_scaled_backbone_grad(feature_map, cnt_backbone_grad_mult)
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

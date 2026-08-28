from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint as _checkpoint

try:
    from .lora_moe import LoRATaskMoE
except ImportError:
    from lora_moe import LoRATaskMoE


class LoRAMoEBlockWrapper(nn.Module):
    """IJEPA ViT block wrapper with a parallel LoRA-MoE FFN branch."""

    def __init__(self, block: nn.Module, lora_moe: LoRATaskMoE, *, grad_checkpointing: bool = False):
        super().__init__()
        self.block = block
        self.lora_moe = lora_moe
        self.grad_checkpointing = bool(grad_checkpointing)

    def _attention_residual(self, x: Tensor) -> tuple[Tensor, Tensor]:
        norm1_out = self.block.norm1(x)
        attn_outputs = self.block.attn(norm1_out)
        attn_out = attn_outputs[0] if isinstance(attn_outputs, tuple) else attn_outputs
        x_attn = x + self.block.drop_path(attn_out)
        ffn_in = self.block.norm2(x_attn)
        return x_attn, ffn_in

    def _ffn_branch(self, ffn_in: Tensor) -> Tensor:
        return self.block.drop_path(self.block.mlp(ffn_in))

    def forward(self, x: Tensor, task_id: int) -> Tensor:
        if self.training and self.grad_checkpointing:
            x_attn, ffn_in = _checkpoint(self._attention_residual, x, use_reentrant=False)
            ffn_out = _checkpoint(self._ffn_branch, ffn_in, use_reentrant=False)
        else:
            x_attn, ffn_in = self._attention_residual(x)
            ffn_out = self._ffn_branch(ffn_in)

        lora_out = self.lora_moe(ffn_in, task_id=task_id)
        return x_attn + ffn_out + lora_out

    def forward_without_task(self, x: Tensor) -> Tensor:
        x_attn, ffn_in = self._attention_residual(x)
        return x_attn + self._ffn_branch(ffn_in)


def wrap_vit_blocks_with_lora_moe(
    blocks: nn.ModuleList,
    *,
    input_size: int,
    rank: int,
    num_experts_private: int,
    num_experts_shared: int,
    k_private: int,
    k_shared: int,
    task_num: int,
    grad_checkpointing: bool,
) -> tuple[nn.ModuleList, nn.ModuleList]:
    wrapped_blocks = nn.ModuleList()
    lora_moes = nn.ModuleList()
    for block in blocks:
        for param in block.parameters():
            param.requires_grad = False
        lora_moe = LoRATaskMoE(
            input_size=input_size,
            rank=rank,
            num_experts_private=num_experts_private,
            num_experts_shared=num_experts_shared,
            k_private=k_private,
            k_shared=k_shared,
            task_num=task_num,
        )
        wrapped_blocks.append(
            LoRAMoEBlockWrapper(block, lora_moe, grad_checkpointing=grad_checkpointing)
        )
        lora_moes.append(lora_moe)
    return wrapped_blocks, lora_moes


class ViTMAEWithLoRAMoE(nn.Module):
    """Frozen IJEPA encoder with trainable LoRA-MoE branches."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        embed_dim: int,
        task_num: int,
        lora_rank: int,
        num_experts_private: int,
        num_experts_shared: int,
        k_private: int,
        k_shared: int,
        grad_checkpointing: bool,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embed_dim = int(embed_dim)
        self.task_num = int(task_num)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.wrapped_blocks, self.lora_moes = wrap_vit_blocks_with_lora_moe(
            self.backbone.blocks,
            input_size=self.embed_dim,
            rank=lora_rank,
            num_experts_private=num_experts_private,
            num_experts_shared=num_experts_shared,
            k_private=k_private,
            k_shared=k_shared,
            task_num=self.task_num,
            grad_checkpointing=grad_checkpointing,
        )

    def get_trainable_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for lora_moe in self.lora_moes:
            params.extend(list(lora_moe.parameters()))
        return params

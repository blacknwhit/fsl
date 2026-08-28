from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint as _checkpoint

try:
    from .lora_moe import LoRATaskMoE
except ImportError:
    from lora_moe import LoRATaskMoE


class CLIPLoRAMoEBlockWrapper(nn.Module):
    """
    Wrap a CLIPEncoderLayer and add a parallel LoRA-MoE branch on the FFN residual path.
    """

    def __init__(
        self,
        block: nn.Module,
        lora_moe: LoRATaskMoE,
        *,
        grad_checkpointing: bool = False,
    ):
        super().__init__()
        self.block = block
        self.lora_moe = lora_moe
        self.grad_checkpointing = bool(grad_checkpointing)

    def _attention_path(self, hidden_states: Tensor) -> Tensor:
        residual = hidden_states
        hidden_states = self.block.layer_norm1(hidden_states)
        hidden_states, _ = self.block.self_attn(hidden_states=hidden_states, attention_mask=None)
        return residual + hidden_states

    def _mlp_inputs_and_outputs(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        residual = hidden_states
        mlp_inputs = self.block.layer_norm2(hidden_states)
        ffn_out = self.block.mlp(mlp_inputs)
        return residual, mlp_inputs, ffn_out

    def forward(self, hidden_states: Tensor, task_id: int) -> Tensor:
        if self.training and self.grad_checkpointing:
            attn_out = _checkpoint(self._attention_path, hidden_states, use_reentrant=False)
            residual, mlp_inputs, ffn_out = _checkpoint(self._mlp_inputs_and_outputs, attn_out, use_reentrant=False)
        else:
            attn_out = self._attention_path(hidden_states)
            residual, mlp_inputs, ffn_out = self._mlp_inputs_and_outputs(attn_out)

        lora_moe_out = self.lora_moe(mlp_inputs, task_id=task_id)
        return residual + ffn_out + lora_moe_out

    def forward_without_task(self, hidden_states: Tensor) -> Tensor:
        return self.block(hidden_states, attention_mask=None)

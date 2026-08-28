from __future__ import annotations

import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint as _checkpoint

try:
    from .lora_moe import LoRATaskMoE
except ImportError:
    from lora_moe import LoRATaskMoE


def _extract_hidden_states(block_output: object) -> Tensor:
    if isinstance(block_output, (tuple, list)):
        return block_output[0]
    if isinstance(block_output, Tensor):
        return block_output
    raise TypeError(f"Unsupported transformer block output type: {type(block_output)}")


def _run_attention(attention: nn.Module, hidden_states: Tensor) -> Tensor:
    try:
        outputs = attention(hidden_states, head_mask=None, output_attentions=False)
    except TypeError:
        try:
            outputs = attention(hidden_states, output_attentions=False)
        except TypeError:
            outputs = attention(hidden_states)
    return _extract_hidden_states(outputs)


class MAELoRAMoEBlockWrapper(nn.Module):
    """
    Wrap a ViT-MAE encoder layer and add a parallel LoRA-MoE branch on the FFN residual path.
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
        normed = self.block.layernorm_before(hidden_states)
        attention_output = _run_attention(self.block.attention, normed)
        return hidden_states + attention_output

    def _mlp_inputs_and_outputs(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        mlp_inputs = self.block.layernorm_after(hidden_states)
        intermediate_output = self.block.intermediate(mlp_inputs)
        ffn_out = self.block.output(intermediate_output, hidden_states)
        return mlp_inputs, ffn_out

    def forward(self, hidden_states: Tensor, task_id: int) -> Tensor:
        if self.training and self.grad_checkpointing:
            attn_out = _checkpoint(self._attention_path, hidden_states, use_reentrant=False)
            mlp_inputs, ffn_out = _checkpoint(self._mlp_inputs_and_outputs, attn_out, use_reentrant=False)
        else:
            attn_out = self._attention_path(hidden_states)
            mlp_inputs, ffn_out = self._mlp_inputs_and_outputs(attn_out)

        lora_moe_out = self.lora_moe(mlp_inputs, task_id=task_id)
        return ffn_out + lora_moe_out

    def forward_without_task(self, hidden_states: Tensor) -> Tensor:
        try:
            outputs = self.block(hidden_states, head_mask=None, output_attentions=False)
        except TypeError:
            try:
                outputs = self.block(hidden_states, output_attentions=False)
            except TypeError:
                outputs = self.block(hidden_states)
        return _extract_hidden_states(outputs)

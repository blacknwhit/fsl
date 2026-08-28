from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint as _checkpoint

try:
    from .lora_moe import LoRATaskMoE
except ImportError:
    from lora_moe import LoRATaskMoE


class LoRAMoEBlockWrapper(nn.Module):
    """
    113_test-style block wrapper: keep original attention/MLP, add LoRA-MoE in
    parallel to FFN residual branch.
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

    def clear_aux_stats(self) -> None:
        self.lora_moe.clear_aux_stats()

    def set_collect_aux_stats(self, enabled: bool) -> None:
        self.lora_moe.set_collect_aux_stats(enabled)

    def get_mi_loss_and_clear(self) -> Tensor:
        return self.lora_moe.get_mi_loss_and_clear()

    def forward(self, x: Tensor, task_id: int, rope: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        if self.training and self.grad_checkpointing:
            def _attn_and_norm2(x_in: Tensor) -> Tuple[Tensor, Tensor]:
                norm1_out = self.block.norm1(x_in)
                try:
                    attn_out = self.block.attn(norm1_out, rope=rope)
                except TypeError:
                    if hasattr(self.block.attn, "forward"):
                        attn_out, _ = self.block.attn(norm1_out, norm1_out, norm1_out)
                    else:
                        attn_out = self.block.attn(norm1_out)
                x_attn_local = x_in + self.block.ls1(attn_out)
                ffn_in_local = self.block.norm2(x_attn_local)
                return x_attn_local, ffn_in_local

            x_attn, ffn_in = _checkpoint(_attn_and_norm2, x, use_reentrant=False)

            def _mlp_path(ffn_in_local: Tensor) -> Tensor:
                return self.block.ls2(self.block.mlp(ffn_in_local))

            ffn_out = _checkpoint(_mlp_path, ffn_in, use_reentrant=False)
        else:
            norm1_out = self.block.norm1(x)
            try:
                attn_out = self.block.attn(norm1_out, rope=rope)
            except TypeError:
                if hasattr(self.block.attn, "forward"):
                    attn_out, _ = self.block.attn(norm1_out, norm1_out, norm1_out)
                else:
                    attn_out = self.block.attn(norm1_out)
            x_attn = x + self.block.ls1(attn_out)
            ffn_in = self.block.norm2(x_attn)
            ffn_out = self.block.ls2(self.block.mlp(ffn_in))

        lora_out = self.lora_moe(ffn_in, task_id=task_id)
        return x_attn + ffn_out + lora_out

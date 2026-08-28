# DINOv3 Block Wrapper with LoRA-MoE
# Wraps the original SelfAttentionBlock and adds a parallel LoRA-MoE branch

from typing import Optional, Tuple, List
import torch.nn as nn
from torch import Tensor
from torch.utils.checkpoint import checkpoint as _checkpoint

try:
    from .lora_moe import LoRATaskMoE
except ImportError:
    from lora_moe import LoRATaskMoE


class LoRAMoEBlockWrapper(nn.Module):
    """
    Wrapper for DINOv3 SelfAttentionBlock that adds a parallel LoRA-MoE branch.
    
    The original block computes:
        x_attn = x + ls1(attn(norm1(x)))
        x_ffn = x_attn + ls2(mlp(norm2(x_attn)))
    
    This wrapper adds LoRA-MoE parallel to FFN:
        x_attn = x + ls1(attn(norm1(x)))
        ffn_in = norm2(x_attn)
        x_out = x_attn + ls2(mlp(ffn_in)) + lora_moe(ffn_in, task_id)
    
    This way, LoRA-MoE shares the same input (norm2 output) as FFN,
    and its output is added to the same residual stream.
    """
    
    def __init__(
        self,
        block: nn.Module,
        lora_moe: LoRATaskMoE,
        *,
        grad_checkpointing: bool = False,
    ):
        """
        Args:
            block: The original SelfAttentionBlock to wrap (will be frozen)
            lora_moe: The LoRA-MoE module to add (will be trainable)
        """
        super().__init__()
        self.block = block
        self.lora_moe = lora_moe
        self.grad_checkpointing = bool(grad_checkpointing)
    
    def forward(self, x: Tensor, task_id: int, rope: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        """
        Forward pass with LoRA-MoE parallel to FFN.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            task_id: Which task's router to use (0=det, 1=seg, 2=cnt)
            rope: Optional RoPE embeddings (passed to attention)
            
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        # We checkpoint only the original block's attention/MLP path for stability.

        if self.training and self.grad_checkpointing:
            def _attn_and_norm2(x_in: Tensor) -> Tuple[Tensor, Tensor]:
                # Attention: x_attn = x + ls1(attn(norm1(x)))
                norm1_out = self.block.norm1(x_in)
                try:
                    attn_out = self.block.attn(norm1_out, rope=rope)
                except TypeError:
                    if hasattr(self.block.attn, 'forward'):
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
            # --- Attention path (unchanged from original block) ---
            norm1_out = self.block.norm1(x)
            try:
                attn_out = self.block.attn(norm1_out, rope=rope)
            except TypeError:
                if hasattr(self.block.attn, 'forward'):
                    attn_out, _ = self.block.attn(norm1_out, norm1_out, norm1_out)
                else:
                    attn_out = self.block.attn(norm1_out)
            x_attn = x + self.block.ls1(attn_out)

            # --- FFN path ---
            ffn_in = self.block.norm2(x_attn)
            ffn_out = self.block.ls2(self.block.mlp(ffn_in))

        # LoRA-MoE output (parallel branch) - must run exactly once per forward.
        lora_out = self.lora_moe(ffn_in, task_id=task_id)

        # Combine: same residual stream
        return x_attn + ffn_out + lora_out
    
    def forward_without_task(self, x: Tensor, rope: Optional[Tuple[Tensor, Tensor]] = None) -> Tensor:
        """
        Forward pass without LoRA-MoE (for inference or when task_id is not available).
        This just runs the original block.
        """
        return self.block(x, rope)


def wrap_dinov3_blocks_with_lora_moe(
    blocks: nn.ModuleList,
    input_size: int = 1024,
    rank: int = 8,
    num_experts_private: int = 2,
    num_experts_shared: int = 6,
    k_private: int = 2,
    k_shared: int = 2,
    task_num: int = 3,
    use_private_experts: bool = True,
    use_shared_experts: bool = True,
) -> Tuple[nn.ModuleList, nn.ModuleList]:
    """
    Wrap all DINOv3 blocks with LoRA-MoE.
    
    Args:
        blocks: Original nn.ModuleList of SelfAttentionBlock
        input_size: Dimension of input features
        rank: LoRA rank
        num_experts_private: Number of private experts per task per block
        num_experts_shared: Number of shared experts per block
        k_private: Top-k experts for the private pool
        k_shared: Top-k experts for the shared pool
        task_num: Number of tasks
        use_private_experts: Whether to instantiate task-private experts
        use_shared_experts: Whether to instantiate task-shared experts
    Returns:
        Tuple of:
            - wrapped_blocks: nn.ModuleList of LoRAMoEBlockWrapper
            - lora_moes: nn.ModuleList of LoRATaskMoE (for easy parameter access)
    """
    wrapped_blocks = nn.ModuleList()
    lora_moes = nn.ModuleList()
    
    for block in blocks:
        # Freeze original block
        for param in block.parameters():
            param.requires_grad = False
        
        # Create LoRA-MoE for this block
        lora_moe = LoRATaskMoE(
            input_size=input_size,
            rank=rank,
            num_experts_private=num_experts_private,
            num_experts_shared=num_experts_shared,
            k_private=k_private,
            k_shared=k_shared,
            task_num=task_num,
            use_private_experts=use_private_experts,
            use_shared_experts=use_shared_experts,
        )
        
        # Wrap the block
        wrapper = LoRAMoEBlockWrapper(block, lora_moe)
        wrapped_blocks.append(wrapper)
        lora_moes.append(lora_moe)
    
    return wrapped_blocks, lora_moes


class DINOv3WithLoRAMoE(nn.Module):
    """
    DINOv3 backbone with LoRA-MoE adapters.
    
    This module wraps the DINOv3 ViT backbone, adding:
    1. LoRA-MoE parallel to each transformer block's FFN
    
    The original backbone is frozen, only LoRA-MoE adapters are trainable.
    """
    
    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int = 1024,
        num_blocks: int = 24,
        task_num: int = 3,
        lora_rank: int = 8,
        num_experts_private: int = 2,
        num_experts_shared: int = 6,
        k_private: int = 2,
        k_shared: int = 2,
        use_private_experts: bool = True,
        use_shared_experts: bool = True,
    ):
        """
        Args:
            backbone: The original DINOv3 backbone
            embed_dim: Dimension of backbone features
            num_blocks: Number of transformer blocks
            task_num: Number of tasks
            lora_rank: LoRA rank
            num_experts_private: Number of private experts per task per block
            num_experts_shared: Number of shared experts per block
            k_private: Top-k experts for the private pool
            k_shared: Top-k experts for the shared pool
            use_private_experts: Whether to instantiate task-private experts
            use_shared_experts: Whether to instantiate task-shared experts
        """
        super().__init__()
        
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.task_num = task_num
        
        # Freeze original backbone
        for param in backbone.parameters():
            param.requires_grad = False
        
        # Wrap blocks with LoRA-MoE
        # We need to replace the blocks in the backbone
        self.wrapped_blocks, self.lora_moes = wrap_dinov3_blocks_with_lora_moe(
            blocks=backbone.blocks,
            input_size=embed_dim,
            rank=lora_rank,
            num_experts_private=num_experts_private,
            num_experts_shared=num_experts_shared,
            k_private=k_private,
            k_shared=k_shared,
            task_num=task_num,
            use_private_experts=use_private_experts,
            use_shared_experts=use_shared_experts,
        )
        
        # Store config
        self.lora_rank = lora_rank
        self.num_experts_private = num_experts_private
        self.num_experts_shared = num_experts_shared
        self.k_private = k_private
        self.k_shared = k_shared
        self.use_private_experts = bool(use_private_experts)
        self.use_shared_experts = bool(use_shared_experts)
    
    def get_trainable_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters (LoRA-MoE)."""
        params = []
        for lora_moe in self.lora_moes:
            params.extend(lora_moe.parameters())
        return params
    
    def forward_features(self, x: Tensor, task_id: int) -> dict:
        """
        Forward pass through DINOv3 with LoRA-MoE.
        
        Args:
            x: Input images [batch, 3, H, W]
            task_id: Which task's router to use (0=det, 1=seg, 2=cnt)
            
        Returns:
            Dict with 'x_norm_patchtokens' key containing output features
        """
        # Get patch embeddings from the backbone
        # We need to manually replicate the backbone's forward_features
        # but use our wrapped blocks instead
        
        # Patch embedding
        x = self.backbone.patch_embed(x)
        
        # Add position embedding if exists
        if hasattr(self.backbone, 'pos_embed') and self.backbone.pos_embed is not None:
            x = x + self.backbone.pos_embed
        
        # Get RoPE if used
        rope = None
        if hasattr(self.backbone, 'rope') and self.backbone.rope is not None:
            rope = self.backbone.rope(x)
        
        # Apply blocks with LoRA-MoE
        for wrapped_block in self.wrapped_blocks:
            # Forward through wrapped block
            x = wrapped_block(x, task_id=task_id, rope=rope)
        
        # Final norm
        if hasattr(self.backbone, 'norm'):
            x = self.backbone.norm(x)
        
        return {'x_norm_patchtokens': x}

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError(f"LoRALinear expects nn.Linear, got {type(base)}")
        r = int(rank)
        if r <= 0:
            raise ValueError(f"rank must be > 0, got {rank}")

        self.base = base
        self.rank = r
        self.alpha = float(alpha)
        self.scaling = float(alpha) / float(r)
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        dtype = self.base.weight.dtype
        device = self.base.weight.device
        in_features = int(self.base.in_features)
        out_features = int(self.base.out_features)

        self.lora_A = nn.Parameter(torch.empty((r, in_features), device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty((out_features, r), device=device, dtype=dtype))

        nn.init.normal_(self.lora_A, std=0.01)
        nn.init.zeros_(self.lora_B)

    @property
    def in_features(self) -> int:
        return int(self.base.in_features)

    @property
    def out_features(self) -> int:
        return int(self.base.out_features)

    def lora_parameters(self):
        yield self.lora_A
        yield self.lora_B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.base.weight, self.base.bias)
        x_d = self.dropout(x)
        delta = F.linear(F.linear(x_d, self.lora_A), self.lora_B)
        return out + delta * self.scaling


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05


def inject_lora_into_vit_mlp(backbone: nn.Module, *, cfg: LoRAConfig) -> int:
    """
    Replace standard ViT MLP linear layers with LoRA-augmented ones.
    Supports IJEPA/timm-style blocks (`mlp.fc1`, `mlp.fc2`) and keeps
    compatibility with transformer-style FFN modules when present.
    Returns the number of Linear layers replaced.
    """
    replaced = 0
    for m in backbone.modules():
        if hasattr(m, "intermediate") and hasattr(m.intermediate, "dense"):
            if isinstance(m.intermediate.dense, nn.Linear) and not isinstance(m.intermediate.dense, LoRALinear):
                m.intermediate.dense = LoRALinear(
                    m.intermediate.dense,
                    rank=cfg.rank,
                    alpha=cfg.alpha,
                    dropout=cfg.dropout,
                )
                replaced += 1
            if hasattr(m, "output") and isinstance(m.output.dense, nn.Linear) and not isinstance(m.output.dense, LoRALinear):
                m.output.dense = LoRALinear(
                    m.output.dense,
                    rank=cfg.rank,
                    alpha=cfg.alpha,
                    dropout=cfg.dropout,
                )
                replaced += 1
        elif hasattr(m, "mlp"):
            if hasattr(m.mlp, "fc1") and isinstance(m.mlp.fc1, nn.Linear) and not isinstance(m.mlp.fc1, LoRALinear):
                m.mlp.fc1 = LoRALinear(m.mlp.fc1, rank=cfg.rank, alpha=cfg.alpha, dropout=cfg.dropout)
                replaced += 1
            if hasattr(m.mlp, "fc2") and isinstance(m.mlp.fc2, nn.Linear) and not isinstance(m.mlp.fc2, LoRALinear):
                m.mlp.fc2 = LoRALinear(m.mlp.fc2, rank=cfg.rank, alpha=cfg.alpha, dropout=cfg.dropout)
                replaced += 1

    if replaced == 0:
        raise RuntimeError("Injected 0 LoRA layers; unexpected backbone/FFN structure")
    return int(replaced)


def inject_lora_into_dinov3_ffn(backbone: nn.Module, *, cfg: LoRAConfig) -> int:
    return inject_lora_into_vit_mlp(backbone, cfg=cfg)


def mark_only_lora_as_trainable(module: nn.Module) -> None:
    module.requires_grad_(False)
    for m in module.modules():
        if isinstance(m, LoRALinear):
            for p in m.lora_parameters():
                p.requires_grad_(True)


def count_trainable_params(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))

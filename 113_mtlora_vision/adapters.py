from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_routing_group_size(dim: int, requested: int) -> int:
    dim = int(dim)
    requested = max(1, int(requested))
    if dim % requested == 0:
        return requested
    divisors = [candidate for candidate in range(min(dim, requested), 0, -1) if dim % candidate == 0]
    return divisors[0] if divisors else 1


def cv_squared(x: torch.Tensor) -> torch.Tensor:
    eps = 1e-10
    if x.numel() <= 1:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return x.float().var() / (x.float().mean().pow(2) + eps)


class SharedExpertBlockAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        rank: int,
        num_shared_experts: int,
        task_names: Sequence[str],
        *,
        lora_alpha: float = 32.0,
        dropout: float = 0.05,
        routing_group_size: int = 512,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be > 0")
        if num_shared_experts <= 0:
            raise ValueError("num_shared_experts must be > 0")
        if not task_names:
            raise ValueError("task_names must not be empty")

        self.dim = int(dim)
        self.rank = int(rank)
        self.num_shared_experts = int(num_shared_experts)
        self.task_names = tuple(str(task_name) for task_name in task_names)
        self.requested_routing_group_size = int(routing_group_size)
        self.routing_group_size = resolve_routing_group_size(self.dim, self.requested_routing_group_size)
        self.num_groups = self.dim // self.routing_group_size
        self.scaling = float(lora_alpha) / float(rank)

        self.dropout = nn.Dropout(p=float(dropout)) if float(dropout) > 0 else nn.Identity()
        self.lora_A = nn.Linear(self.dim, self.rank, bias=False)
        self.lora_B = nn.ModuleList(
            [nn.Linear(self.rank, self.dim, bias=False) for _ in range(self.num_shared_experts)]
        )
        self.routers = nn.ModuleDict(
            {
                task_name: nn.Linear(self.dim, self.num_groups * self.num_shared_experts, bias=False)
                for task_name in self.task_names
            }
        )
        self.last_blc: torch.Tensor | None = None
        self.last_blc_count = 0

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        for expert in self.lora_B:
            nn.init.zeros_(expert.weight)
        for router in self.routers.values():
            nn.init.kaiming_uniform_(router.weight, a=math.sqrt(5))

    def iter_b_matrices(self) -> Iterable[torch.Tensor]:
        for expert in self.lora_B:
            yield expert.weight

    def forward(self, x: torch.Tensor, *, task_name: str) -> torch.Tensor:
        if task_name not in self.routers:
            raise ValueError(f"Unknown task_name {task_name!r}; expected one of {self.task_names}")

        x_for_lora = x.to(self.lora_A.weight.dtype)
        hidden = self.lora_A(self.dropout(x_for_lora))

        router = self.routers[task_name]
        x_for_route = x.to(router.weight.dtype)
        route_logits = router(x_for_route)

        if route_logits.ndim == 3:
            batch_size, seq_len, _ = route_logits.shape
            route_logits = route_logits.view(batch_size, seq_len, self.num_groups, self.num_shared_experts)
        elif route_logits.ndim == 2:
            batch_size, _ = route_logits.shape
            route_logits = route_logits.view(batch_size, self.num_groups, self.num_shared_experts)
        else:
            raise ValueError(f"Unexpected route_logits rank: {route_logits.ndim}")

        route_weight = F.softmax(route_logits, dim=-1, dtype=torch.float32).to(x.dtype)
        result = torch.zeros_like(x)

        for expert_idx, expert in enumerate(self.lora_B):
            expert_out = expert(hidden).to(result.dtype)
            weight_i = route_weight[..., expert_idx]
            weight_i = weight_i.repeat_interleave(self.routing_group_size, dim=-1)
            result = result + (weight_i * expert_out * self.scaling)

        if self.training:
            mean_dims = tuple(range(route_weight.ndim - 1))
            current_blc = cv_squared(route_weight.mean(dim=mean_dims))
            self.last_blc = current_blc if self.last_blc is None else self.last_blc + current_blc
            self.last_blc_count += 1
        else:
            self.last_blc = None
            self.last_blc_count = 0
        return result


def collect_balance_loss(model: nn.Module) -> torch.Tensor | None:
    total_blc = None
    count = 0

    for module in model.modules():
        if isinstance(module, SharedExpertBlockAdapter) and module.last_blc is not None:
            total_blc = module.last_blc if total_blc is None else total_blc + module.last_blc
            module.last_blc = None
            count += int(getattr(module, "last_blc_count", 1))
            module.last_blc_count = 0

    if total_blc is None or count == 0:
        return None
    return total_blc / count


def iter_shared_expert_b_matrices(model: nn.Module) -> Iterable[torch.Tensor]:
    for module in model.modules():
        if isinstance(module, SharedExpertBlockAdapter):
            yield from module.iter_b_matrices()


def compute_spectral_regularization(model: nn.Module) -> torch.Tensor | None:
    b_matrices = list(iter_shared_expert_b_matrices(model))
    if not b_matrices:
        return None

    total_loss = torch.zeros((), device=b_matrices[0].device, dtype=b_matrices[0].dtype)
    for b_matrix in b_matrices:
        rank = b_matrix.shape[1]
        gram = torch.mm(b_matrix.t(), b_matrix)

        with torch.no_grad():
            try:
                singular_values = torch.linalg.svdvals(b_matrix.detach().float())
                sigma_mean = singular_values.mean() + 1e-8
                weights = torch.exp(-singular_values / sigma_mean).to(b_matrix.dtype)
            except Exception:
                weights = torch.ones(rank, device=b_matrix.device, dtype=b_matrix.dtype)
            scale = torch.linalg.norm(b_matrix.detach()).item() / (rank**0.5) + 1e-8

        identity = torch.eye(rank, device=b_matrix.device, dtype=b_matrix.dtype) * (scale**2)
        diff = gram - identity
        weighted_diff = diff * weights.unsqueeze(0) * weights.unsqueeze(1)
        total_loss = total_loss + (weighted_diff.pow(2).sum())

    return total_loss / len(b_matrices)

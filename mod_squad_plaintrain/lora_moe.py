import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def compute_gating(k: int, probs: torch.Tensor, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
    zeros = torch.zeros_like(probs)
    gates = zeros.scatter(1, top_k_indices, top_k_gates)

    top_k_gates_flat = top_k_gates.flatten()
    top_k_experts_flat = top_k_indices.flatten()

    nonzeros = top_k_gates_flat.nonzero().squeeze(-1)
    top_k_experts_nonzero = top_k_experts_flat[nonzeros]

    _, order = top_k_experts_nonzero.sort(0)
    expert_size = (gates > 0).long().sum(0)

    index_sorted_experts = nonzeros[order]
    batch_index = index_sorted_experts.div(k, rounding_mode="trunc")
    batch_gates = top_k_gates_flat[index_sorted_experts]
    batch_expert = top_k_experts_nonzero[order]

    return batch_gates, batch_index, expert_size, gates, index_sorted_experts, batch_expert


class LoRATaskMoE(nn.Module):
    """
    113_test-style LoRA-MoE branch, but shared-expert-only.

    - Parallel to FFN path (the wrapper handles residual composition).
    - Task-specific routers are kept.
    - Task-private experts are removed (num_experts_private is accepted for compat but ignored).
    - MI statistics are accumulated from shared-router probabilities.
    """

    def __init__(
        self,
        input_size: int,
        rank: int = 8,
        num_experts_private: int = 0,
        num_experts_shared: int = 6,
        k_private: int = 0,
        k_shared: int = 2,
        task_num: int = 3,
    ):
        super().__init__()

        self.input_size = int(input_size)
        self.rank = int(rank)
        self.task_num = int(task_num)
        self.num_experts_private = int(num_experts_private)
        self.num_experts_shared = int(num_experts_shared)
        self.k_private = int(k_private)
        self.k_shared = min(int(k_shared), self.num_experts_shared)

        if self.num_experts_shared < 1:
            raise ValueError("num_experts_shared must be >= 1")
        if self.k_shared < 1:
            raise ValueError("k_shared must be >= 1")

        # Keep private tensors for checkpoint compatibility, but disable them.
        self.lora_A_private = nn.Parameter(torch.empty(self.task_num, 0, self.input_size, self.rank))
        self.lora_B_private = nn.Parameter(torch.empty(self.task_num, 0, self.rank, self.input_size))

        # Shared experts: [E, D, R] and [E, R, D]
        self.lora_A_shared = nn.Parameter(torch.empty(self.num_experts_shared, self.input_size, self.rank))
        self.lora_B_shared = nn.Parameter(torch.zeros(self.num_experts_shared, self.rank, self.input_size))
        self._init_lora_weights()

        # Task-specific shared router.
        self.f_gate_shared = nn.ModuleList(
            [nn.Linear(self.input_size, self.num_experts_shared, bias=False) for _ in range(self.task_num)]
        )
        for gate in self.f_gate_shared:
            nn.init.zeros_(gate.weight)

        self.collect_aux_stats = True
        self.init_aux_statistics()

    def _init_lora_weights(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A_shared, a=math.sqrt(5))
        # lora_B_shared is already zero-init, matching 113_test additive-branch setup.

    def init_aux_statistics(self) -> None:
        self.mi_task_gate_terms = [[] for _ in range(self.task_num)]
        self.mi_token_count = 0

    def clear_aux_stats(self) -> None:
        self.init_aux_statistics()

    def set_collect_aux_stats(self, enabled: bool) -> None:
        self.collect_aux_stats = bool(enabled)

    def _update_mi_statistics(self, probs: torch.Tensor, task_id: int) -> None:
        if (not self.training) or (not self.collect_aux_stats):
            return
        self.mi_task_gate_terms[task_id].append(probs.sum(0))
        self.mi_token_count += int(probs.shape[0])

    def get_mi_loss_and_clear(self) -> torch.Tensor:
        has_terms = any(len(v) > 0 for v in self.mi_task_gate_terms)
        if (not has_terms) or self.mi_token_count <= 0:
            zero = self.lora_A_shared.sum() * 0.0
            self.init_aux_statistics()
            return zero

        rows = []
        for task_terms in self.mi_task_gate_terms:
            if len(task_terms) == 0:
                rows.append(self.lora_A_shared.new_zeros(self.num_experts_shared))
            elif len(task_terms) == 1:
                rows.append(task_terms[0])
            else:
                rows.append(torch.stack(task_terms, dim=0).sum(dim=0))

        mi_task_gate = torch.stack(rows, dim=0)
        joint = mi_task_gate / max(float(self.mi_token_count), 1e-8)
        p_task = joint.sum(dim=1, keepdim=True)
        p_expert = joint.sum(dim=0, keepdim=True)
        mi_loss = -(joint * torch.log(joint / (p_task * p_expert + 1e-8) + 1e-8)).sum()
        self.init_aux_statistics()
        return mi_loss

    def _top_k_gating_shared(self, x: torch.Tensor, task_id: int, skip_mask: Optional[torch.Tensor]) -> None:
        logits = self.f_gate_shared[task_id](x)
        probs = torch.softmax(logits, dim=1) + 1e-4
        if skip_mask is not None:
            probs = torch.masked_fill(probs, skip_mask, 0)

        top_k_gates, top_k_indices = probs.topk(self.k_shared, dim=1)
        (
            self.shared_batch_gates,
            self.shared_batch_index,
            self.shared_expert_size,
            _gates,
            self.shared_index_sorted_experts,
            self.shared_batch_expert,
        ) = compute_gating(self.k_shared, probs, top_k_gates, top_k_indices)
        self._update_mi_statistics(probs, task_id)

    def forward(
        self,
        x: torch.Tensor,
        task_id: int,
        skip_mask: Optional[torch.Tensor] = None,
        multiply_by_gates: bool = True,
    ) -> torch.Tensor:
        bsz, length, emb_size = x.size()
        x_flat = x.reshape(-1, emb_size)
        if skip_mask is not None:
            skip_mask = skip_mask.view(-1, 1)

        self._top_k_gating_shared(x_flat, task_id, skip_mask)

        # Grouped expert matmul avoids per-route weight expansion.
        num_routes = int(self.shared_index_sorted_experts.numel())
        shared_outputs = x_flat.new_zeros((num_routes, self.input_size))
        offset = 0
        for expert_id in range(self.num_experts_shared):
            cur_size = int(self.shared_expert_size[expert_id].item())
            if cur_size <= 0:
                continue
            sl = slice(offset, offset + cur_size)
            expert_inputs = x_flat[self.shared_batch_index[sl]]
            expert_A = self.lora_A_shared[expert_id]
            expert_B = self.lora_B_shared[expert_id]
            cur_outputs = expert_inputs.matmul(expert_A).matmul(expert_B)
            if multiply_by_gates:
                cur_outputs = cur_outputs * self.shared_batch_gates[sl, None]
            shared_outputs[sl] = cur_outputs
            offset += cur_size

        zeros = torch.zeros(
            bsz * length,
            self.input_size,
            dtype=shared_outputs.dtype,
            device=shared_outputs.device,
        )
        y = zeros.index_add(0, self.shared_batch_index, shared_outputs)
        return y.view(bsz, length, self.input_size)

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, rank={self.rank}, "
            f"num_experts_private={self.num_experts_private}, k_private={self.k_private}, "
            f"num_experts_shared={self.num_experts_shared}, k_shared={self.k_shared}, "
            f"task_num={self.task_num}"
        )

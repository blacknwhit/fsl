# LoRA-MoE: Sparsely-Gated Mixture-of-LoRA-Experts
# Based on Mod-Squad's moe.py, replacing ParallelExperts with LoRA structure
#
# Key modifications:
# 1. Replace self.experts/self.output_experts with LoRA parameters (lora_A, lora_B)
# 2. Keep all gating logic, top-k selection, and aux loss computation unchanged
# 3. Use batch matmul for efficient LoRA computation

import math
from typing import Optional

import torch
import torch.nn as nn


@torch.jit.script
def compute_gating(k: int, probs: torch.Tensor, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
    """Compute gating indices and weights for sparse MoE routing.

    Returns:
        batch_gates: (S,) gate weights for selected entries (sorted by expert)
        batch_index: (S,) token indices in [0, T) where T = num_tokens
        expert_size: (E,) number of selected tokens per expert
        gates: (T, E) dense gates (mainly for aux stats)
        index_sorted_experts: (S,) indices into top_k_gates.flatten() of selected entries (sorted by expert)
        batch_expert: (S,) expert id for each selected entry (sorted by expert)
    """
    zeros = torch.zeros_like(probs)
    gates = zeros.scatter(1, top_k_indices, top_k_gates)

    top_k_gates_flat = top_k_gates.flatten()      # (T*k,)
    top_k_experts_flat = top_k_indices.flatten()  # (T*k,)

    nonzeros = top_k_gates_flat.nonzero().squeeze(-1)  # (S,)
    top_k_experts_nonzero = top_k_experts_flat[nonzeros]  # (S,)

    _, _index_sorted_experts = top_k_experts_nonzero.sort(0)
    expert_size = (gates > 0).long().sum(0)

    index_sorted_experts = nonzeros[_index_sorted_experts]  # (S,)
    batch_index = index_sorted_experts.div(k, rounding_mode='trunc')
    batch_gates = top_k_gates_flat[index_sorted_experts]
    batch_expert = top_k_experts_nonzero[_index_sorted_experts]

    return batch_gates, batch_index, expert_size, gates, index_sorted_experts, batch_expert


class LoRATaskMoE(nn.Module):
    """
    Dual-pool LoRA-MoE with task-private experts + shared experts.

    - Private experts: task-specific LoRA adapters (not shared across tasks).
    - Shared experts: shared across tasks, with task-specific routers.
    - Routing is per-task and top-k inside each pool.
    """

    def __init__(
        self,
        input_size: int,
        rank: int = 8,
        num_experts_private: int = 2,
        num_experts_shared: int = 6,
        k_private: int = 2,
        k_shared: int = 2,
        task_num: int = 3,
    ):
        super().__init__()

        self.input_size = int(input_size)
        self.rank = int(rank)
        self.task_num = int(task_num)
        self.num_experts_private = int(num_experts_private)
        self.num_experts_shared = int(num_experts_shared)
        self.k_private = min(int(k_private), self.num_experts_private)
        self.k_shared = min(int(k_shared), self.num_experts_shared)

        if self.k_private < 1:
            raise ValueError("k_private must be >= 1")
        if self.k_shared < 1:
            raise ValueError("k_shared must be >= 1")

        # Private experts: [T, E_priv, D, R] and [T, E_priv, R, D]
        self.lora_A_private = nn.Parameter(
            torch.empty(self.task_num, self.num_experts_private, self.input_size, self.rank)
        )
        self.lora_B_private = nn.Parameter(
            torch.zeros(self.task_num, self.num_experts_private, self.rank, self.input_size)
        )

        # Shared experts: [E_shared, D, R] and [E_shared, R, D]
        self.lora_A_shared = nn.Parameter(torch.empty(self.num_experts_shared, self.input_size, self.rank))
        self.lora_B_shared = nn.Parameter(torch.zeros(self.num_experts_shared, self.rank, self.input_size))
        self._init_lora_weights()

        # Task-specific routers for private and shared pools.
        self.f_gate_private = nn.ModuleList(
            [nn.Linear(self.input_size, self.num_experts_private, bias=False) for _ in range(self.task_num)]
        )
        self.f_gate_shared = nn.ModuleList(
            [nn.Linear(self.input_size, self.num_experts_shared, bias=False) for _ in range(self.task_num)]
        )
        for gate in list(self.f_gate_private) + list(self.f_gate_shared):
            nn.init.zeros_(gate.weight)

    def _init_lora_weights(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A_private, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_shared, a=math.sqrt(5))
        # B matrices are already zeros from initialization

    def _top_k_gating(
        self,
        x: torch.Tensor,
        task_id: int,
        router: nn.ModuleList,
        k: int,
        skip_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = router[task_id](x)
        probs = torch.softmax(logits, dim=1) + 1e-4
        if skip_mask is not None:
            probs = torch.masked_fill(probs, skip_mask, 0)

        top_k_gates, top_k_indices = probs.topk(k, dim=1)
        batch_gates, batch_index, expert_size, gates, index_sorted_experts, batch_expert = compute_gating(
            k, probs, top_k_gates, top_k_indices
        )
        return batch_gates, batch_index, batch_expert, expert_size, gates, probs

    def _top_k_gating_private(self, x: torch.Tensor, task_id: int, skip_mask: Optional[torch.Tensor]) -> None:
        (
            self.private_batch_gates,
            self.private_batch_index,
            self.private_batch_expert,
            self.private_expert_size,
            _gates,
            _probs,
        ) = self._top_k_gating(x, task_id, self.f_gate_private, self.k_private, skip_mask)

    def _top_k_gating_shared(self, x: torch.Tensor, task_id: int, skip_mask: Optional[torch.Tensor]) -> None:
        (
            self.shared_batch_gates,
            self.shared_batch_index,
            self.shared_batch_expert,
            self.shared_expert_size,
            gates,
            probs,
        ) = self._top_k_gating(x, task_id, self.f_gate_shared, self.k_shared, skip_mask)

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

        self._top_k_gating_private(x_flat, task_id, skip_mask)
        self._top_k_gating_shared(x_flat, task_id, skip_mask)

        private_inputs = x_flat[self.private_batch_index]
        private_A = self.lora_A_private[task_id, self.private_batch_expert]
        private_B = self.lora_B_private[task_id, self.private_batch_expert]
        private_outputs = torch.bmm(torch.bmm(private_inputs.unsqueeze(1), private_A), private_B).squeeze(1)
        if multiply_by_gates:
            private_outputs = private_outputs * self.private_batch_gates[:, None]

        shared_inputs = x_flat[self.shared_batch_index]
        shared_A = self.lora_A_shared[self.shared_batch_expert]
        shared_B = self.lora_B_shared[self.shared_batch_expert]
        shared_outputs = torch.bmm(torch.bmm(shared_inputs.unsqueeze(1), shared_A), shared_B).squeeze(1)
        if multiply_by_gates:
            shared_outputs = shared_outputs * self.shared_batch_gates[:, None]

        zeros = torch.zeros(
            bsz * length,
            self.input_size,
            dtype=shared_outputs.dtype,
            device=shared_outputs.device,
        )
        y = zeros.index_add(0, self.private_batch_index, private_outputs)
        y = y.index_add(0, self.shared_batch_index, shared_outputs)
        return y.view(bsz, length, self.input_size)

    def extra_repr(self):
        return (
            f"input_size={self.input_size}, rank={self.rank}, "
            f"num_experts_private={self.num_experts_private}, k_private={self.k_private}, "
            f"num_experts_shared={self.num_experts_shared}, k_shared={self.k_shared}, "
            f"task_num={self.task_num}"
        )

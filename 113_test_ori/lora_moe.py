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
    - Only MI loss is supported, applied to the shared pool (optional).
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
        use_mi_loss_shared: bool = True,
        acc_aux_loss: bool = True,
    ):
        super().__init__()

        self.input_size = int(input_size)
        self.rank = int(rank)
        self.task_num = int(task_num)
        self.num_experts_private = int(num_experts_private)
        self.num_experts_shared = int(num_experts_shared)
        self.k_private = min(int(k_private), self.num_experts_private)
        self.k_shared = min(int(k_shared), self.num_experts_shared)
        self.use_mi_loss_shared = bool(use_mi_loss_shared)
        self.acc_aux_loss = bool(acc_aux_loss)

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

        self.init_aux_statistics()

    def _init_lora_weights(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A_private, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A_shared, a=math.sqrt(5))
        # B matrices are already zeros from initialization

    def init_aux_statistics(self, clear: bool = True) -> None:
        self.shared_acc_freq = 0.0
        self.shared_MI_task_gate = torch.zeros(
            self.task_num,
            self.num_experts_shared,
            device=self.lora_A_shared.device,
            dtype=torch.float32,
        )

    def _update_shared_statistics(self, probs: torch.Tensor, gates: torch.Tensor, task_id: int) -> None:
        self.shared_acc_freq = self.shared_acc_freq + (gates > 0).float().sum(0)
        task_mask = torch.zeros(self.task_num, device=probs.device, dtype=probs.dtype)
        task_mask[task_id] = 1.0
        self.shared_MI_task_gate = self.shared_MI_task_gate.to(device=probs.device, dtype=probs.dtype) + (
            task_mask[:, None] * probs.sum(0)[None, :]
        )

    def get_aux_loss_and_clear(self):
        if not isinstance(self.shared_acc_freq, torch.Tensor):
            device = self.lora_A_shared.device
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero, zero

        if not self.use_mi_loss_shared:
            device = self.shared_acc_freq.device
            zero = torch.tensor(0.0, device=device)
            self.init_aux_statistics(clear=False)
            return zero, zero, zero, zero

        tot = self.shared_acc_freq.sum() / float(self.k_shared)
        mi_norm = self.shared_MI_task_gate / (tot + 0.0001)
        p_ti = torch.sum(mi_norm, dim=1, keepdim=True) + 0.0001
        p_ei = torch.sum(mi_norm, dim=0, keepdim=True) + 0.0001
        mi_loss = -(mi_norm * torch.log(mi_norm / p_ti / p_ei + 0.0001)).sum()

        self.init_aux_statistics(clear=False)
        zero = mi_loss.new_tensor(0.0)
        return zero, zero, zero, mi_loss

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

        if self.acc_aux_loss and self.use_mi_loss_shared:
            self._update_shared_statistics(probs, gates, task_id)

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
            f"task_num={self.task_num}, use_mi_shared={self.use_mi_loss_shared}"
        )

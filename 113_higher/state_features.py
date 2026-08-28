from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class SharedExpertStateTracker(nn.Module):
    def __init__(
        self,
        model,
        *,
        matrix_proj_dim: int = 128,
        task_hidden_dim: int | None = None,
        task_state_dim: int = 128,
    ):
        super().__init__()
        if not hasattr(model.shared, "lora_moes"):
            raise ValueError("State features require LoRA-MoE shared experts.")

        lora_moes = list(model.shared.lora_moes)
        if len(lora_moes) != 24:
            raise ValueError(f"Expected 24 LoRA-MoE blocks, got {len(lora_moes)}")

        object.__setattr__(self, "_last_moe_ref", lora_moes[-1])
        self.num_experts = int(self.last_moe.num_experts_shared)
        self.rank = int(self.last_moe.rank)
        self.input_size = int(self.last_moe.input_size)
        self.matrix_param_dim = int(self.input_size * self.rank)
        self.expert_feature_dim = int(2 * matrix_proj_dim)
        self.task_input_dim = int(self.num_experts * self.expert_feature_dim)
        self.task_state_dim = int(task_state_dim)
        self.feature_dim = int(3 * self.task_state_dim)

        self.matrix_proj = nn.Linear(self.matrix_param_dim, int(matrix_proj_dim))
        self.task_projector = nn.Linear(self.task_input_dim, self.task_state_dim)

    @property
    def last_moe(self):
        return self._last_moe_ref

    @staticmethod
    def _sync_matrix(mat: torch.Tensor, use_ddp: bool, world_size: int) -> torch.Tensor:
        if not use_ddp:
            return mat
        out = mat.detach().clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        out /= float(world_size)
        return out

    def _expert_grad_matrices(
        self,
        grad_a: torch.Tensor | None,
        grad_b: torch.Tensor | None,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if grad_a is None:
            a = torch.zeros_like(self.last_moe.lora_A_shared, device=device, dtype=torch.float32)
        else:
            a = grad_a.detach().float()
        if grad_b is None:
            b = torch.zeros_like(self.last_moe.lora_B_shared, device=device, dtype=torch.float32)
        else:
            b = grad_b.detach().float()
        return a.reshape(self.num_experts, -1), b.reshape(self.num_experts, -1)

    def _encode_task(self, expert_a: torch.Tensor, expert_b: torch.Tensor) -> torch.Tensor:
        proj_a = self.matrix_proj(expert_a)
        proj_b = self.matrix_proj(expert_b)
        expert_repr = torch.cat([proj_a, proj_b], dim=-1).reshape(1, -1)
        return self.task_projector(expert_repr).squeeze(0)

    def encode_shared_grad_state(
        self,
        grad_a: torch.Tensor | None,
        grad_b: torch.Tensor | None,
        *,
        device: torch.device,
        use_ddp: bool,
        world_size: int,
    ) -> torch.Tensor:
        expert_a, expert_b = self._expert_grad_matrices(grad_a, grad_b, device=device)
        expert_a = self._sync_matrix(expert_a, use_ddp, world_size)
        expert_b = self._sync_matrix(expert_b, use_ddp, world_size)
        return self._encode_task(expert_a, expert_b)

    def extract_features(
        self,
        *,
        det_loss: torch.Tensor,
        seg_loss: torch.Tensor,
        cnt_loss: torch.Tensor,
        device: torch.device,
        use_ddp: bool,
        world_size: int,
    ) -> torch.Tensor:
        losses = (det_loss, seg_loss, cnt_loss)
        task_states = []

        for loss in losses:
            grad_targets = []
            target_names = []
            if self.last_moe.lora_A_shared.requires_grad:
                grad_targets.append(self.last_moe.lora_A_shared)
                target_names.append("a")
            if self.last_moe.lora_B_shared.requires_grad:
                grad_targets.append(self.last_moe.lora_B_shared)
                target_names.append("b")

            grad_a = None
            grad_b = None
            if grad_targets:
                grads = torch.autograd.grad(
                    loss,
                    tuple(grad_targets),
                    retain_graph=True,
                    allow_unused=True,
                )
                for name, grad in zip(target_names, grads):
                    if name == "a":
                        grad_a = grad
                    else:
                        grad_b = grad
            task_states.append(
                self.encode_shared_grad_state(
                    grad_a,
                    grad_b,
                    device=device,
                    use_ddp=use_ddp,
                    world_size=world_size,
                )
            )

        return torch.cat(task_states, dim=0)

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointWeightGenerator(nn.Module):
    """
    114_grpo-style generator body.

    Input state and network structure are kept consistent with 114_grpo_stage1:
    Linear -> GELU -> Linear -> GELU -> Linear(3), then softmax * 24.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        base_loss_weights: tuple[float, float, float] = (15.0, 8.0, 1.0),
        hidden_dim: int = 192,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        hidden_dim2 = max(self.hidden_dim // 2, 4)

        self.net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, hidden_dim2),
            nn.GELU(),
            nn.Linear(hidden_dim2, 3),
        )
        prior = torch.tensor(
            [float(base_loss_weights[0]), float(base_loss_weights[1]), float(base_loss_weights[2])],
            dtype=torch.float32,
        )
        self.register_buffer(
            "base_loss_weights",
            prior,
        )
        self.register_buffer("weight_scale", prior.sum())

    def _reshape_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        elif state.dim() != 2:
            raise ValueError(f"state must be rank-1 or rank-2, got shape {tuple(state.shape)}")
        if int(state.shape[-1]) != self.state_dim:
            raise ValueError(f"state dim mismatch: got {int(state.shape[-1])}, expected {self.state_dim}")
        return state

    def raw_weights(self, state: torch.Tensor) -> torch.Tensor:
        state = self._reshape_state(state)
        logits = self.net(state)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
        return logits.squeeze(0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.raw_weights(state)
        weights = F.softmax(logits, dim=-1) * self.weight_scale
        return torch.nan_to_num(weights, nan=0.0, posinf=1e6, neginf=-1e6)

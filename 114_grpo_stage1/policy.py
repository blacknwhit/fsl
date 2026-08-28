from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PolicyOutput:
    # 记录一次策略前向后的关键结果。
    alpha: torch.Tensor
    mu: torch.Tensor
    concentration: torch.Tensor
    delta: torch.Tensor


class DirichletWeightGenerator(nn.Module):
    # 小型策略网络：学习三任务相对偏好，并保留一个固定总强度缩放。
    def __init__(
        self,
        state_dim: int = 24,
        hidden_dim: int = 64,
        prior_weights: tuple[float, float, float] = (15.0, 8.0, 1.0),
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)

        prior = torch.tensor(prior_weights, dtype=torch.float32)
        if prior.numel() != 3 or torch.any(prior <= 0):
            raise ValueError("prior_weights must contain 3 positive numbers")
        # 用先验和定义固定总强度，默认 15+8+1=24。
        self.register_buffer("weight_scale", prior.sum())
        hidden_dim = int(hidden_dim)
        hidden_dim2 = max(hidden_dim // 2, 4)

        self.net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, hidden_dim2),
            nn.GELU(),
            nn.Linear(hidden_dim2, 4),
        )

    def forward(self, state: torch.Tensor) -> PolicyOutput:
        # 输入统一整理成 [B, state_dim]。
        if state.dim() == 1:
            state = state.unsqueeze(0)

        out = self.net(state)
        delta = out[..., :3]
        z_c = out[..., 3:]
        mu = torch.softmax(delta, dim=-1)
        concentration = F.softplus(z_c)
        alpha = torch.clamp(concentration * mu, min=1e-8)

        return PolicyOutput(
            alpha=alpha.squeeze(0),
            mu=mu.squeeze(0),
            concentration=concentration.squeeze(0),
            delta=delta.squeeze(0),
        )

    def build_dirichlet(self, state: torch.Tensor) -> PolicyOutput:
        # 语义化封装，便于外部按“构建分布”理解调用点。
        return self.forward(state)


def build_dirichlet(
    policy: DirichletWeightGenerator,
    state: torch.Tensor,
) -> tuple[torch.distributions.Dirichlet, PolicyOutput]:
    # 一次前向同时拿到分布对象和中间结果。
    output = policy.build_dirichlet(state)
    return torch.distributions.Dirichlet(output.alpha), output


def sample_candidates(alpha: torch.Tensor, num_candidates: int) -> torch.Tensor:
    # 从当前 Dirichlet 分布独立采样 K 组候选权重。
    return torch.distributions.Dirichlet(alpha).sample((int(num_candidates),))


def dirichlet_log_prob(alpha: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
    # 计算候选权重在给定 Dirichlet 下的 log-prob。
    return torch.distributions.Dirichlet(alpha).log_prob(sample)


def compute_ratio(logp_new: torch.Tensor, logp_old: torch.Tensor) -> torch.Tensor:
    # PPO/GRPO 的密度比。
    return torch.exp(logp_new - logp_old)


def compute_clipped_grpo_loss(
    logp_new: torch.Tensor,
    logp_old: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 计算 clipped GRPO 目标，同时返回 ratio 便于日志观察。
    ratio = compute_ratio(logp_new, logp_old)
    clipped_ratio = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps))
    objective = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    return -objective.mean(), ratio


def smooth_task_weights(samples: torch.Tensor, gamma: float) -> torch.Tensor:
    # 轻微向均匀分布回拉，减少极端候选权重。
    gamma = float(gamma)
    if gamma <= 0:
        return samples
    uniform = torch.full_like(samples, 1.0 / 3.0)
    return (1.0 - gamma) * samples + gamma * uniform


def scale_task_weights(weights: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    # 用固定系数放大采样/均值权重，保留相对比例。
    if not isinstance(scale, torch.Tensor):
        scale = torch.tensor(float(scale), device=weights.device, dtype=weights.dtype)
    if scale.dim() == 0:
        scale = scale.view(1)
    while scale.dim() < weights.dim():
        scale = scale.unsqueeze(0)
    return torch.clamp(weights * scale, min=1e-8)

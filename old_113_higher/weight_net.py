from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CountGradProjector(nn.Module):
    """
    Structure-aware projector for the counting head's final 1x1 conv gradients.

    Input is the flattened concatenation of:
    1. conv.weight.grad shaped [num_classes, in_channels, 1, 1]
    2. conv.bias.grad shaped [num_classes]
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        out_dim: int = 64,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.class_feat_dim = self.in_channels + 1
        self.expected_dim = self.num_classes * self.class_feat_dim

        self.class_proj = nn.Conv1d(self.class_feat_dim, hidden_dim, kernel_size=1, bias=True)
        self.act = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.out_proj = nn.Linear(hidden_dim * self.num_classes, out_dim)

    def forward(self, flat_grad: torch.Tensor) -> torch.Tensor:
        flat_grad = flat_grad.reshape(-1)
        if int(flat_grad.numel()) != self.expected_dim:
            raise ValueError(
                "CountGradProjector input dim mismatch: "
                f"got {int(flat_grad.numel())}, expected {self.expected_dim}"
            )

        weight_dim = self.num_classes * self.in_channels
        grad_w = flat_grad[:weight_dim].reshape(self.num_classes, self.in_channels)
        grad_b = flat_grad[weight_dim:].reshape(self.num_classes, 1)

        class_grads = torch.cat([grad_w, grad_b], dim=1)
        class_grads = class_grads.transpose(0, 1).unsqueeze(0)

        feat = self.class_proj(class_grads)
        feat = self.act(feat)
        feat = feat.flatten(start_dim=1)
        return self.out_proj(feat).squeeze(0)


class JointWeightGenerator(nn.Module):
    """
    Fixed weight generator matching 113_test/train_10per.sh current config:
    - joint architecture
    - last-layer grads only
    - l2 grad normalization happens outside this module
    - det/seg projected to 64-d by Linear
    - cnt projected by CountGradProjector(hidden_dim=64)
    - joint head: Linear -> LeakyReLU -> Dropout(0) -> Linear(3) -> LeakyReLU
    - final weights use additive prior bias 15:8:1
    """

    def __init__(
        self,
        *,
        det_in_dim: int,
        seg_in_dim: int,
        cnt_in_channels: int,
        cnt_num_classes: int,
        base_loss_weights: tuple[float, float, float] = (15.0, 8.0, 1.0),
    ) -> None:
        super().__init__()
        grad_embed_dim = 64
        hidden_dim = 16

        self.det_proj = nn.Linear(int(det_in_dim), grad_embed_dim)
        self.seg_proj = nn.Linear(int(seg_in_dim), grad_embed_dim)
        self.cnt_proj = CountGradProjector(
            in_channels=int(cnt_in_channels),
            num_classes=int(cnt_num_classes),
            out_dim=grad_embed_dim,
            hidden_dim=64,
        )
        self.head = nn.Sequential(
            nn.Linear(grad_embed_dim * 3, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=False),
            nn.Dropout(p=0.0),
            nn.Linear(hidden_dim, 3),
        )
        self.register_buffer(
            "base_loss_weights",
            torch.tensor([float(base_loss_weights[0]), float(base_loss_weights[1]), float(base_loss_weights[2])]),
        )

    def raw_weights(self, det_vec: torch.Tensor, seg_vec: torch.Tensor, cnt_vec: torch.Tensor) -> torch.Tensor:
        det_feat = self.det_proj(det_vec)
        seg_feat = self.seg_proj(seg_vec)
        cnt_feat = self.cnt_proj(cnt_vec)
        feat = torch.cat([det_feat, seg_feat, cnt_feat], dim=0)
        raw = self.head(feat)
        raw = F.leaky_relu(raw, negative_slope=0.01)
        return torch.nan_to_num(raw, nan=0.0, posinf=1e6, neginf=-1e6)

    def forward(self, det_vec: torch.Tensor, seg_vec: torch.Tensor, cnt_vec: torch.Tensor) -> torch.Tensor:
        raw = self.raw_weights(det_vec, seg_vec, cnt_vec)
        out = raw + self.base_loss_weights
        return torch.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)

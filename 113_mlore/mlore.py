from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint


BatchNorm2d = nn.BatchNorm2d


def _cv_squared(route_prob: torch.Tensor) -> torch.Tensor:
    route_vector = route_prob.mean(dim=0).reshape(-1)
    mean = route_vector.mean().clamp_min(1e-6)
    std = route_vector.std(unbiased=False)
    return (std / mean).pow(2)


class LoraBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int = 1, rank: int = 8):
        super().__init__()
        padding = kernel_size // 2
        self.W = nn.Conv2d(in_channels, rank, kernel_size=kernel_size, stride=1, padding=padding)
        self.M = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.W.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W.bias)
        nn.init.kaiming_uniform_(self.M.weight, a=math.sqrt(5))
        nn.init.zeros_(self.M.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.M(self.W(x))


class SpatialAtt(nn.Module):
    def __init__(self, dim: int, dim_out: int, *, im_size: int, with_feat: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim_out, kernel_size=1)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(dim_out)
        self.convsp = nn.Linear(im_size, 1)
        self.ln_sp = nn.LayerNorm(dim)
        self.conv2 = nn.Conv2d(dim, dim_out, kernel_size=1)
        self.conv3 = nn.Conv2d(dim_out, dim_out, kernel_size=1)
        self.with_feat = bool(with_feat)
        if self.with_feat:
            self.feat_linear = nn.Conv2d(dim_out * 2, dim_out * 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape

        feat = self.conv1(x)
        feat = self.ln(feat.reshape(n, -1, h * w).permute(0, 2, 1)).permute(0, 2, 1).reshape(n, -1, h, w)
        feat = self.act(feat)
        feat = self.conv3(feat)

        feat_sp = self.convsp(x.reshape(n, -1, h * w)).reshape(n, 1, -1)
        feat_sp = self.ln_sp(feat_sp).reshape(n, -1, 1, 1)
        feat_sp = self.act(feat_sp)
        feat_sp = self.conv2(feat_sp)

        n, c, h, w = feat.shape
        feat = torch.mean(feat.reshape(n, c, h * w), dim=2).reshape(n, c, 1, 1)
        feat = torch.cat([feat, feat_sp], dim=1)
        return feat


class MLoREBlock(nn.Module):
    def __init__(
        self,
        *,
        tasks: Sequence[str],
        embed_dim: int,
        rank_list: Sequence[int],
        task_rank: int,
        topk: int,
        pre_softmax: bool,
        im_size: int,
    ):
        super().__init__()
        self.tasks = tuple(tasks)
        self.rank_list = tuple(int(rank) for rank in rank_list)
        self.num_lora = len(self.rank_list)
        self.pre_softmax = bool(pre_softmax)
        self.desert_k = max(self.num_lora - int(topk), 0)

        self.lora_list = nn.ModuleList(
            [LoraBlock(embed_dim, embed_dim, kernel_size=3, rank=rank) for rank in self.rank_list]
        )
        self.share_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.activate = nn.GELU()

        self.conv1 = nn.ModuleDict({task: nn.Conv2d(embed_dim, embed_dim, kernel_size=1) for task in self.tasks})
        self.conv2 = nn.ModuleDict(
            {task: LoraBlock(embed_dim, embed_dim, kernel_size=3, rank=int(task_rank)) for task in self.tasks}
        )
        self.conv3 = nn.ModuleDict({task: nn.Conv2d(embed_dim, embed_dim, kernel_size=1) for task in self.tasks})
        self.bn = nn.ModuleDict({task: BatchNorm2d(embed_dim) for task in self.tasks})
        self.bn_all = nn.ModuleDict({task: BatchNorm2d(embed_dim) for task in self.tasks})
        self.router = nn.ModuleDict(
            {
                task: nn.ModuleList(
                    [
                        SpatialAtt(embed_dim, embed_dim // 4, im_size=im_size, with_feat=False),
                        nn.Conv2d(embed_dim // 2, self.num_lora * 2 + 1, kernel_size=1),
                    ]
                )
                for task in self.tasks
            }
        )
        self._init_residual_head()

    def _init_residual_head(self) -> None:
        # Keep the initial decoder close to the projected backbone feature.
        for task in self.tasks:
            nn.init.normal_(self.conv3[task].weight, mean=0.0, std=1e-3)
            if self.conv3[task].bias is not None:
                nn.init.zeros_(self.conv3[task].bias)

    def _sparsify(self, logits: torch.Tensor) -> torch.Tensor:
        if self.desert_k <= 0:
            return logits
        mask = torch.zeros_like(logits, dtype=torch.bool)
        indices = torch.topk(logits, self.desert_k, dim=1, largest=False).indices
        mask.scatter_(1, indices, True)
        if self.pre_softmax:
            logits = logits.masked_fill(mask, -1e10)
        else:
            logits = logits.masked_fill(mask, 0.0)
        return logits

    def _route(self, logits: torch.Tensor, stdev: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(logits) * stdev if self.training else 0.0
        if self.pre_softmax:
            logits = self._sparsify(logits + noise)
            route = torch.softmax(logits, dim=1)
        else:
            route = torch.softmax(logits + noise, dim=1)
            route = self._sparsify(route)
        return route

    def forward(self, x: torch.Tensor, task: str) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.conv1[task](x)
        route_feat = self.router[task][0](out)
        prob_all = self.router[task][1](route_feat).unsqueeze(2)
        prob_lora, prob_mix = prob_all[:, : self.num_lora * 2], prob_all[:, self.num_lora * 2 :]
        route_logits, stdev = prob_lora.chunk(2, dim=1)
        route = self._route(route_logits, stdev)

        lora_sum = torch.zeros_like(out)
        for expert_index, expert in enumerate(self.lora_list):
            gate = route[:, expert_index]
            if not torch.any(gate != 0):
                continue
            lora_sum = lora_sum + expert(out) * gate

        delta = (
            self.bn_all[task](lora_sum)
            + self.conv2[task](out) * prob_mix[:, 0]
            + self.share_conv(out.detach())
        )
        delta = self.bn[task](delta)
        delta = self.activate(delta)
        delta = self.conv3[task](delta)
        route_prob = route.squeeze(2).squeeze(-1).squeeze(-1)
        return delta, route_prob


class MLoREDecoder(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        decoder_dim: int,
        tasks: Sequence[str],
        num_layers: int,
        num_stages: int,
        rank_list: Sequence[int],
        task_rank: int,
        topk: int,
        pre_softmax: bool,
        grad_checkpointing: bool,
        im_size: int,
    ):
        super().__init__()
        self.tasks = tuple(tasks)
        self.decoder_dim = int(decoder_dim)
        self.num_layers = int(num_layers)
        self.num_stages = int(num_stages)
        self.grad_checkpointing = bool(grad_checkpointing)
        if self.num_stages < 1 or self.num_stages > 2:
            raise ValueError(f"num_stages must be 1 or 2, got {self.num_stages}")

        self.layer_projs = nn.ModuleList(
            [nn.Conv2d(embed_dim, self.decoder_dim, kernel_size=1) for _ in range(self.num_layers)]
        )
        self.stage1_blocks = nn.ModuleList(
            [
                MLoREBlock(
                    tasks=self.tasks,
                    embed_dim=self.decoder_dim,
                    rank_list=rank_list,
                    task_rank=task_rank,
                    topk=topk,
                    pre_softmax=pre_softmax,
                    im_size=im_size,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.stage2_blocks = (
            nn.ModuleList(
                [
                    MLoREBlock(
                        tasks=self.tasks,
                        embed_dim=self.decoder_dim,
                        rank_list=rank_list,
                        task_rank=task_rank,
                        topk=topk,
                        pre_softmax=pre_softmax,
                        im_size=im_size,
                    )
                    for _ in range(self.num_layers)
                ]
            )
            if self.num_stages == 2
            else None
        )
        self.task_mask = (
            nn.ModuleDict(
                {
                    task: nn.Sequential(
                        nn.Conv2d(self.decoder_dim * self.num_layers, self.decoder_dim, kernel_size=1),
                        BatchNorm2d(self.decoder_dim),
                        nn.GELU(),
                        nn.Conv2d(self.decoder_dim, self.num_layers, kernel_size=3, padding=1),
                    )
                    for task in self.tasks
                }
            )
            if self.num_layers > 1
            else None
        )

    def _run_block(self, block: MLoREBlock, x: torch.Tensor, task: str) -> tuple[torch.Tensor, torch.Tensor]:
        if not (self.training and self.grad_checkpointing):
            return block(x, task)

        def _wrapped(inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return block(inp, task)

        return _checkpoint(_wrapped, x, use_reentrant=False)

    def forward_all_task_features(
        self,
        multi_scale_features: Sequence[torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, list[dict[str, torch.Tensor]]]]:
        if len(multi_scale_features) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} feature maps, got {len(multi_scale_features)}")

        out_feat = {task: [] for task in self.tasks}
        route_stage1 = [{task: None for task in self.tasks} for _ in range(self.num_layers)]
        route_stage2 = [{task: None for task in self.tasks} for _ in range(self.num_layers)] if self.stage2_blocks is not None else []
        for layer_index, feature in enumerate(multi_scale_features):
            combined = self.layer_projs[layer_index](feature)
            for task in self.tasks:
                x_task = combined
                delta_1, route_1 = self._run_block(self.stage1_blocks[layer_index], x_task, task)
                x_task = x_task + delta_1
                route_2 = None
                if self.stage2_blocks is not None:
                    delta_2, route_2 = self._run_block(self.stage2_blocks[layer_index], x_task, task)
                    x_task = x_task + delta_2
                out_feat[task].append(x_task)
                route_stage1[layer_index][task] = route_1
                if self.stage2_blocks is not None:
                    route_stage2[layer_index][task] = route_2

        fused = {}
        for task in self.tasks:
            fused[task] = self._fuse_task_layers(out_feat[task], task)

        return fused, {"route_1_prob": route_stage1, "route_2_prob": route_stage2}

    def forward_task_features(
        self,
        multi_scale_features: Sequence[torch.Tensor],
        task: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(multi_scale_features) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} feature maps, got {len(multi_scale_features)}")

        layer_outputs = []
        route_losses = []
        for layer_index, feature in enumerate(multi_scale_features):
            x = self.layer_projs[layer_index](feature)
            delta_1, route_1 = self._run_block(self.stage1_blocks[layer_index], x, task)
            x = x + delta_1
            route_losses.append(_cv_squared(route_1))
            if self.stage2_blocks is not None:
                delta_2, route_2 = self._run_block(self.stage2_blocks[layer_index], x, task)
                x = x + delta_2
                route_losses.append(_cv_squared(route_2))
            layer_outputs.append(x)
        fused = self._fuse_task_layers(layer_outputs, task)

        lb_loss = torch.stack(route_losses).sum() if route_losses else fused.new_zeros(())
        return fused, lb_loss

    def _fuse_task_layers(self, layer_outputs: Sequence[torch.Tensor], task: str) -> torch.Tensor:
        if len(layer_outputs) == 1:
            return layer_outputs[0]
        if self.task_mask is None:
            raise RuntimeError("task_mask is not initialized for multi-layer fusion")
        layer_mask = torch.softmax(self.task_mask[task](torch.cat(layer_outputs, dim=1)), dim=1)
        fused = torch.zeros_like(layer_outputs[0])
        for layer_index, feature in enumerate(layer_outputs):
            fused = fused + layer_mask[:, layer_index : layer_index + 1] * feature
        return fused

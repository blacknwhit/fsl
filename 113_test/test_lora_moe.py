#!/usr/bin/env python3
"""
Sanity checks for the LoRA-MoE implementation.
"""

import torch
import torch.nn as nn


def test_lora_moe():
    """Test the LoRATaskMoE module."""
    from lora_moe import LoRATaskMoE

    print("=" * 60)
    print("Test LoRATaskMoE")
    print("=" * 60)

    # Config
    input_size = 1024
    rank = 8
    num_experts_private = 2
    num_experts_shared = 6
    task_num = 3
    k_private = 2
    k_shared = 2
    batch_size = 4
    seq_len = 197  # 14*14 + 1 cls token

    lora_moe = LoRATaskMoE(
        input_size=input_size,
        rank=rank,
        num_experts_private=num_experts_private,
        num_experts_shared=num_experts_shared,
        task_num=task_num,
        k_private=k_private,
        k_shared=k_shared,
    )

    print(f"input_size: {input_size}")
    print(f"lora_rank: {rank}")
    print(f"private_experts: {num_experts_private}")
    print(f"shared_experts: {num_experts_shared}")
    print(f"task_num: {task_num}")
    print(f"top_k_private: {k_private}")
    print(f"top_k_shared: {k_shared}")

    total_params = sum(p.numel() for p in lora_moe.parameters())
    trainable_params = sum(p.numel() for p in lora_moe.parameters() if p.requires_grad)
    print(f"total params: {total_params:,}")
    print(f"trainable params: {trainable_params:,}")

    x = torch.randn(batch_size, seq_len, input_size)
    for task_id in range(task_num):
        output = lora_moe(x, task_id=task_id)
        print(f"Task {task_id}: input {x.shape} -> output {output.shape}")

    print("OK: LoRATaskMoE\n")


def test_dinov3_wrapper():
    """Test the DINOv3 wrapper."""
    from lora_moe import LoRATaskMoE
    from dinov3_moe_wrapper import LoRAMoEBlockWrapper

    print("=" * 60)
    print("Test LoRAMoEBlockWrapper")
    print("=" * 60)

    embed_dim = 1024

    class MockDinoBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, num_heads=16, batch_first=True)
            self.ls1 = nn.Identity()
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            )
            self.ls2 = nn.Identity()

        def forward(self, x):
            x = x + self.ls1(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
            x = x + self.ls2(self.mlp(self.norm2(x)))
            return x

    block = MockDinoBlock(embed_dim)
    lora_moe = LoRATaskMoE(
        input_size=embed_dim,
        rank=8,
        num_experts_private=2,
        num_experts_shared=6,
        task_num=3,
        k_private=2,
        k_shared=2,
    )
    wrapped = LoRAMoEBlockWrapper(block, lora_moe)

    batch_size = 2
    seq_len = 197
    x = torch.randn(batch_size, seq_len, embed_dim)

    for task_id in range(3):
        output = wrapped(x, task_id=task_id)
        print(f"Task {task_id}: input {x.shape} -> output {output.shape}")

    print("OK: LoRAMoEBlockWrapper\n")


def test_auto_weighted_loss():
    """Test AutomaticWeightedLoss."""
    from auto_weighted_loss import AutomaticWeightedLoss

    print("=" * 60)
    print("Test AutomaticWeightedLoss")
    print("=" * 60)

    awl = AutomaticWeightedLoss(num=3)

    det_loss = torch.tensor(0.5, requires_grad=True)
    seg_loss = torch.tensor(1.2, requires_grad=True)
    cnt_loss = torch.tensor(0.8, requires_grad=True)

    total = awl([det_loss, seg_loss, cnt_loss])
    print(f"losses: det={det_loss.item():.4f}, seg={seg_loss.item():.4f}, cnt={cnt_loss.item():.4f}")
    print(f"weighted total: {total.item():.4f}")
    print(f"learned params: {awl.params.data.tolist()}")
    print(f"weights: {awl.get_weights()}")

    total.backward()
    print(f"grad params: {awl.params.grad.tolist()}")

    print("OK: AutomaticWeightedLoss\n")


def test_integration():
    """Integration test: forward + loss + backward."""
    from lora_moe import LoRATaskMoE
    from auto_weighted_loss import AutomaticWeightedLoss

    print("=" * 60)
    print("Integration test")
    print("=" * 60)

    num_blocks = 2
    embed_dim = 128
    lora_moes = nn.ModuleList([
        LoRATaskMoE(
            input_size=embed_dim,
            rank=4,
            num_experts_private=2,
            num_experts_shared=4,
            task_num=3,
            k_private=1,
            k_shared=2,
        )
        for _ in range(num_blocks)
    ])

    awl = AutomaticWeightedLoss(num=3)

    batch_size = 2
    seq_len = 10
    det_x = torch.randn(batch_size, seq_len, embed_dim)
    seg_x = torch.randn(batch_size, seq_len, embed_dim)
    cnt_x = torch.randn(batch_size, seq_len, embed_dim)

    for moe in lora_moes:
        det_x = det_x + moe(det_x, task_id=0)
    for moe in lora_moes:
        seg_x = seg_x + moe(seg_x, task_id=1)
    for moe in lora_moes:
        cnt_x = cnt_x + moe(cnt_x, task_id=2)

    det_loss = det_x.mean()
    seg_loss = seg_x.mean()
    cnt_loss = cnt_x.mean()

    main_loss = awl([det_loss, seg_loss, cnt_loss])
    total_loss = main_loss

    total_loss.backward()

    def _grad_status(param: torch.Tensor) -> str:
        if param.grad is None:
            return "none"
        if float(param.grad.detach().abs().sum().item()) == 0.0:
            return "zero"
        return "nonzero"

    total_trainable = 0
    grad_none = 0
    grad_zero = 0
    grad_nonzero = 0

    key_nonzero = {"lora_B": 0, "f_gate": 0, "lora_A": 0}

    for block_idx, moe in enumerate(lora_moes):
        for name, p in moe.named_parameters():
            if not p.requires_grad:
                continue
            total_trainable += 1
            status = _grad_status(p)
            if status == "none":
                grad_none += 1
            elif status == "zero":
                grad_zero += 1
            else:
                grad_nonzero += 1
                if "lora_B" in name:
                    key_nonzero["lora_B"] += 1
                if "f_gate" in name:
                    key_nonzero["f_gate"] += 1
                if "lora_A" in name:
                    key_nonzero["lora_A"] += 1

        with torch.no_grad():
            a_priv = 0
            b_priv = 0
            a_shared = 0
            b_shared = 0
            if moe.lora_A_private.grad is not None:
                a_priv = int((moe.lora_A_private.grad.abs().sum(dim=(0, 2, 3)) > 0).sum().item())
            if moe.lora_B_private.grad is not None:
                b_priv = int((moe.lora_B_private.grad.abs().sum(dim=(0, 2, 3)) > 0).sum().item())
            if moe.lora_A_shared.grad is not None:
                a_shared = int((moe.lora_A_shared.grad.abs().sum(dim=(1, 2)) > 0).sum().item())
            if moe.lora_B_shared.grad is not None:
                b_shared = int((moe.lora_B_shared.grad.abs().sum(dim=(1, 2)) > 0).sum().item())
            print(
                f"  block{block_idx}: private lora_A grad_nonzero={a_priv}/{moe.lora_A_private.shape[1]} | "
                f"private lora_B grad_nonzero={b_priv}/{moe.lora_B_private.shape[1]} | "
                f"shared lora_A grad_nonzero={a_shared}/{moe.lora_A_shared.shape[0]} | "
                f"shared lora_B grad_nonzero={b_shared}/{moe.lora_B_shared.shape[0]}"
            )

    print(f"trainable params: {total_trainable}")
    print(f"grad status nonzero={grad_nonzero}, zero={grad_zero}, none={grad_none}")
    print(f"key grad nonzero counts: {key_nonzero}")

    assert (key_nonzero["lora_B"] > 0) or (key_nonzero["f_gate"] > 0), (
        "No non-zero grad found for lora_B or f_gate; MoE path may be disconnected."
    )

    print("OK: Integration test\n")


def main():
    print("\n" + "=" * 60)
    print("LoRA-MoE sanity tests")
    print("=" * 60 + "\n")

    test_lora_moe()
    test_dinov3_wrapper()
    test_auto_weighted_loss()
    test_integration()

    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()

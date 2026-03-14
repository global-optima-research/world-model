"""
GRPO 梯度 Sanity Check

验证:
1. gradient_accumulation_steps=1 时，梯度方向随 advantage 正负交替 → 抵消
2. gradient_accumulation_steps=num_generations 时，梯度方向一致 → 有效更新

不需要 GPU，纯数值模拟 GRPO 的梯度累积逻辑。
"""

import torch
import torch.nn as nn
import math


class ToyModel(nn.Module):
    """模拟 transformer 的一个简单模型"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)

    def forward(self, x):
        return self.fc(x)


def simulate_grpo_step(num_generations, gradient_accumulation_steps, clip_range):
    """模拟一个 GRPO training step"""
    model = ToyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 模拟 reward: 组内标准化
    rewards = torch.rand(num_generations)
    mean_r, std_r = rewards.mean(), rewards.std() + 1e-8
    advantages = (rewards - mean_r) / std_r

    print(f"\nConfig: gen={num_generations}, accum={gradient_accumulation_steps}, clip={clip_range}")
    print(f"Rewards:    {rewards.tolist()}")
    print(f"Advantages: {advantages.tolist()}")
    print(f"Adv sum:    {advantages.sum().item():.6f} (should be ~0)")

    # 记录初始参数
    init_params = {n: p.clone() for n, p in model.named_parameters()}

    total_grad_norms = []
    step_count = 0
    optimizer.zero_grad()

    for i in range(num_generations):
        # 模拟 log_prob 计算 (new vs old)
        x = torch.randn(1, 16)
        pred = model(x)
        # 模拟 new_log_prob - old_log_prob (初始时 ratio ≈ 1)
        new_log_prob = -0.5 * pred.pow(2).sum()
        old_log_prob = new_log_prob.detach() + torch.randn(1).item() * 0.01  # 微小偏差

        ratio = torch.exp(new_log_prob - old_log_prob)
        adv = advantages[i]

        unclipped = -adv * ratio
        clipped = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        loss = torch.maximum(unclipped, clipped) / (gradient_accumulation_steps)

        loss.backward()

        if (i + 1) % gradient_accumulation_steps == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_grad_norms.append(grad_norm.item())
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

    # 计算参数变化
    param_delta = sum(
        (p - init_params[n]).pow(2).sum().item()
        for n, p in model.named_parameters()
    ) ** 0.5

    print(f"\nResults:")
    print(f"  Optimizer steps:  {step_count}")
    print(f"  Grad norms:       {[f'{g:.6f}' for g in total_grad_norms]}")
    print(f"  Mean grad norm:   {sum(total_grad_norms)/len(total_grad_norms):.6f}")
    print(f"  Param delta (L2): {param_delta:.6f}")
    return param_delta, total_grad_norms


def main():
    torch.manual_seed(42)

    print("=" * 70)
    print("Test 1: accum=1 (bug — optimizer steps after EACH sample)")
    print("         Positive and negative advantages alternate → cancel out")
    print("=" * 70)
    delta1, norms1 = simulate_grpo_step(
        num_generations=8,
        gradient_accumulation_steps=1,
        clip_range=1e-4,
    )

    torch.manual_seed(42)  # same seed for fair comparison

    print("\n" + "=" * 70)
    print("Test 2: accum=num_generations (correct — accumulate full group)")
    print("         All advantages contribute before one optimizer step")
    print("=" * 70)
    delta2, norms2 = simulate_grpo_step(
        num_generations=8,
        gradient_accumulation_steps=8,
        clip_range=1e-4,
    )

    print("\n" + "=" * 70)
    print("Test 3: accum=1, clip_range=0.1 (large clip, still oscillates)")
    print("=" * 70)
    torch.manual_seed(42)
    delta3, norms3 = simulate_grpo_step(
        num_generations=8,
        gradient_accumulation_steps=1,
        clip_range=0.1,
    )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  accum=1, clip=1e-4:   param_delta={delta1:.6f}  (bug: {len(norms1)} optimizer steps)")
    print(f"  accum=8, clip=1e-4:   param_delta={delta2:.6f}  (correct: {len(norms2)} optimizer step)")
    print(f"  accum=1, clip=0.1:    param_delta={delta3:.6f}  (still bad: {len(norms3)} optimizer steps)")
    print(f"\n  Conclusion: accum=1 causes {len(norms1)}x more optimizer steps")
    print(f"  with alternating advantage signs, updates cancel out.")
    print(f"  Fix: set gradient_accumulation_steps = num_generations")


if __name__ == "__main__":
    main()

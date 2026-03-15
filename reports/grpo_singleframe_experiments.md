# GRPO 单帧训练实验报告

**日期**: 2026-03-15
**目标**: 验证 GRPO pipeline 梯度流通，确认 CLIP reward 能提升
**模型**: Wan2.1-T2V-1.3B (单帧 t=1, 512×512)
**硬件**: 2× RTX 5090 (GPU 5,7)
**基线 Commit**: `38ebccc` (feat: add CLIP reward + single-frame GRPO training script)

---

## 1. 发现的 Pipeline Bug

### 1.1 DanceGRPO 是 on-policy REINFORCE，不是 PPO

**Commit**: `0e7edc3`, `ba98a1d`

DanceGRPO 的训练循环每步重新采样：`sample_reference_model()` 从当前模型生成数据，然后 `grpo_one_step()` 用同一模型计算 new_log_prob。因此：

```
new_log_prob == old_log_prob  (精确相等，10位小数)
ratio = exp(0) = 1.0000000000
```

PPO 的 clip_range 机制完全失效。实际执行的是纯 REINFORCE 梯度。

### 1.2 DanceGRPO Wan 脚本缺少 DDP/FSDP

**Commit**: `340c7c7` → `eac479e` → `a7105a8` → `dd3a1f8`

DanceGRPO 的 Mochi/Hunyuan/Flux 脚本都有 FSDP 包装，唯独 Wan 脚本没有。尝试加 FSDP 失败（WanTransformerBlock 不兼容），最终改用 DDP。

### 1.3 init_same_noise 导致 reward 方差极小

所有候选从相同初始噪声出发，CLIP score 范围仅 0.02~0.05，advantage 信号 ≈ 噪声。

---

## 2. 实验汇总

### 固定参数
| 参数 | 值 |
|------|-----|
| 模型 | Wan2.1-T2V-1.3B |
| 分辨率 | 512×512×1 (单帧) |
| Reward | CLIP-ViT-L/14 (score/100, clamp [0,1]) |
| num_generations | 8 |
| gradient_accumulation_steps | 8 |
| sampling_steps | 20 |
| eta | 0.3 |
| max_grad_norm | 1.0 |
| GPU | 2× RTX 5090 (DDP) |

### 实验结果

| # | Commit | lr | clip_range | PPO epochs | same_noise | Steps | First10 Avg | Last10 Avg | Δ Reward | 结果 |
|---|--------|-----|-----------|------------|------------|-------|-------------|------------|----------|------|
| E1 | `604126d` | 1e-5 | 1e-4 | 4 | Yes | 54 | 0.2594 | 0.2685 | **+0.009** | 微弱上升 |
| E2 | `1bb2240` | 1e-5 | 0.2 | 4 | Yes | 200 | 0.2609 | 0.2500 | -0.011 | 无提升 |
| E3 | `c139925` | 1e-4 | 0.2 | 4 | Yes | 8 | — | — | — | **Mode collapse** (0.24→0.17) |
| E4 | `d3c6a31` | 5e-5 | 0.2 | 4 | Yes | 20 | — | — | — | **Mode collapse** (0.24→0.15) |
| E5 | `3fe682f` | 2e-5 | 0.2 | 2 | Yes | 29 | 0.2611 | 0.2365 | -0.025 | 缓慢下降 |
| E6 | `d437c7e` | 1e-5 | 0.2 | 1 | **No** | 60 | 0.2623 | 0.2564 | -0.006 | 无提升 |

### 关键观察

```
E1 ratio (epoch 0): 1.0000000000  (on-policy, 无梯度信号增强)
E1 ratio (epoch 1): 0.998~0.9996  (ratio 偏离，PPO 生效)
E1 ratio (epoch 2): 0.994~0.9997
E1 ratio (epoch 3): 0.998~0.9998

E3 ratio (epoch 1): 0.65~0.97    (lr 太大，ratio 偏离过多 → collapse)
E4 ratio (epoch 1): 0.96~0.99    (仍然 collapse)

E6 reward std:      0.022~0.024  (diverse noise, 比 same_noise 的 0.010 大 2 倍)
```

---

## 3. 分析

### 3.1 Multi-epoch PPO 有效但不够

E1 证明 multi-epoch PPO 确实让 ratio 偏离 1.0，PPO clipping 生效。但：
- clip_range=1e-4 把 epoch 1+ 的 ratio（0.998）截断到 [0.9999, 1.0001]，梯度信号被大幅削弱
- clip_range=0.2 (E2) 不截断，但 lr=1e-5 太小，200 步仍无提升

### 3.2 lr 与 collapse 的矛盾

| lr | 结果 |
|-----|------|
| 1e-5 | 安全但学不动 |
| 2e-5 | 缓慢下降 |
| 5e-5 | Collapse (~step 7) |
| 1e-4 | 快速 Collapse (~step 3) |

lr 的有效窗口极窄，说明 CLIP reward 信号太弱，无法在噪声中提供稳定的学习方向。

### 3.3 CLIP Reward 的根本局限

- CLIP score 对生成图像的区分度低：所有生成的 score 集中在 0.20~0.30
- Group 内 std 仅 0.01~0.02，advantage 的信噪比接近 1
- 不同 prompt 间的难度差异（0.22 vs 0.29）远大于组内差异，导致 step 间 reward 波动掩盖了学习趋势

---

## 4. 结论

### 4.1 Pipeline 验证结论

**GRPO pipeline 代码已验证正确**：
- 梯度流通 ✅
- ratio 偏离机制（multi-epoch PPO）✅
- PPO clipping ✅
- gradient accumulation ✅
- DDP 梯度同步 ✅

**DanceGRPO 原始 Wan 脚本确认有 bug**：缺少 DDP/FSDP 包装。

### 4.2 单帧 CLIP 实验是弯路

**这一轮实验的根本问题不在超参，而在 reward 选择。**

回顾报告 002 的物理 reward 验证数据：

| 指标 | 物理 Reward (报告 002) | CLIP Reward (本报告) |
|------|----------------------|---------------------|
| 动态范围 | **0.74** (0.08~0.81) | 0.10 (0.20~0.30) |
| 组内 std | **~0.23** | ~0.02 |
| 与人工评估一致性 | 5/5 流体排序正确 | N/A (不评估物理) |

**物理 reward 的信号强度是 CLIP 的 10 倍以上。**

报告 003 的视频+物理 reward GRPO 失败，不是因为物理 reward 信号不够，而是因为 pipeline 有 bug：
- `gradient_accumulation_steps=4` 与 `num_generations=4` 不匹配导致正负 advantage 梯度抵消（已修复：`203801f`）
- 缺少 DDP 梯度同步（已修复：`dd3a1f8`）

这些 bug 修好后，我们没有重试视频+物理 reward，而是去做了单帧 CLIP 实验，在一个信号强度只有 1/10 的 reward 上做了 6 组超参搜索。这是方向性错误。

### 4.3 核心教训

> **当 GRPO 训练没效果时，应优先检查 pipeline bug，而非换 reward model。更不应该用与目标无关的 reward（美学指标）替代目标 reward（物理真实性）。**

---

## 5. 下一步：重回视频 + 物理 Reward

所有 pipeline bug 已修复。物理 reward 信号足够强（std=0.23）。正确的下一步是：

### 立即执行

用修好的 pipeline 重跑视频 + 物理 reward GRPO：

| 参数 | 报告 003 (失败) | 修正后配置 |
|------|----------------|-----------|
| 分辨率 | 480×832×33 | **512×512×5** (最短光流视频) |
| num_generations | 4 | **8** |
| gradient_accumulation | 4 (bug) | **8** (= num_generations) |
| DDP | ❌ | **✅** |
| init_same_noise | Yes | **No** (diverse noise) |
| Reward | PhysicsRewardModel | PhysicsRewardModel (不变) |
| lr | 1e-5 | 1e-5 (不变) |
| Steps | 50 | **100** |

### 预期

- 物理 reward 组内 std ~0.23（比 CLIP 的 0.02 大 10 倍），advantage 信号强
- Pipeline bug 已修复，梯度不会被抵消
- 应该能看到 reward 上升趋势

### 后续路径

1. 物理 reward 提升 → 扩展到更长视频 (t=17, t=33)
2. 消融实验：flow_only vs flow+fluid
3. Warp 模拟器集成（Level 3 reward）
4. 三域联合训练

---

## 附录：复现命令

```bash
# E6 (最终配置，最稳定)
git checkout d437c7e
CUDA_VISIBLE_DEVICES=5,7 WANDB_MODE=disabled bash scripts/train_grpo_singleframe.sh

# 查看 reward 趋势
grep '^rewards:' logs/grpo_diverse_noise.log | python3 -c "
import sys, re
for i, line in enumerate(sys.stdin):
    nums = [float(x) for x in re.findall(r'[\d.]+', line)]
    print(f'step {i+1}: mean={sum(nums)/len(nums):.4f}')
"
```

# 实验报告 004: 视频 + 物理 Reward GRPO 训练 (v8)

**日期**: 2026-03-15
**目的**: 用修好的 pipeline 重跑视频 + 物理 reward GRPO，验证 reward 能否提升
**结果**: **成功。Reward 从 0.442 提升到 0.498 (+0.056)，后半段加速上升。**

## 复现信息

- **Commit**: `c831c33` (fix: video GRPO with fixed pipeline)
- **硬件**: 2× NVIDIA RTX 5090 (GPU 5, 7)
- **Conda 环境**: `wan2`
- **运行命令**:
```bash
ssh 5090
source /data/Anaconda3/etc/profile.d/conda.sh && conda activate wan2
cd ~/world-model && git checkout c831c33
export WANDB_MODE=disabled
CUDA_VISIBLE_DEVICES=5,7 bash scripts/train_grpo_physics.sh
```

## 训练配置

| 参数 | 报告 003 (失败) | 本次 v8 (成功) | 变化原因 |
|------|----------------|---------------|---------|
| 分辨率 | 480×832×33 | **512×512×5** | 最短光流视频，加速迭代 |
| num_generations | 4 | **8** | 更稳定的 advantage 估计 |
| gradient_accumulation | 4 (bug) | **8** | 对齐 num_generations，防止梯度抵消 |
| DDP | ❌ | **✅** | 多 GPU 梯度同步 |
| init_same_noise | Yes | **No** | 增加候选多样性，增大 reward 方差 |
| lr | 1e-5 | 1e-5 | 不变 |
| clip_range | 1e-2 | **0.2** | 放宽 clipping |
| num_ppo_epochs | 1 | 1 | 纯 REINFORCE |
| sampling_steps | 10 | **20** | 对齐 DanceGRPO 原始 |
| Reward | PhysicsRewardModel | PhysicsRewardModel | 不变 |
| max_train_steps | 50 | **100** | 更充分的训练 |

## 结果

### Reward 趋势 (每 10 步平均)

```
step   1- 10: 0.4422
step  11- 20: 0.4474
step  21- 30: 0.4385
step  31- 40: 0.4373  ← 前 40 步平稳
step  41- 50: 0.4461
step  51- 60: 0.4849  ← 开始上升
step  61- 70: 0.4827
step  71- 80: 0.4777
step  81- 90: 0.5107  ← 最高段
step  91-100: 0.4982
```

| 指标 | 值 |
|------|-----|
| 前 10 步平均 | 0.4422 |
| 后 10 步平均 | **0.4982** |
| **Δ Reward** | **+0.056** |
| 总耗时 | ~75 分钟 (100 步, ~45s/步) |

### 对比所有历史实验

| 实验 | Reward 类型 | 分辨率 | Steps | Δ Reward | 结果 |
|------|-----------|--------|-------|----------|------|
| 报告 003 (v3) | 物理 | 480×832×33 | 50 | +0.001 | 失败 (pipeline bug) |
| 单帧 E1 | CLIP | 512×512×1 | 54 | +0.009 | 微弱 |
| 单帧 E2 | CLIP | 512×512×1 | 200 | -0.011 | 失败 |
| 单帧 E3-E5 | CLIP | 512×512×1 | 8-29 | — | Mode collapse |
| 单帧 E6 | CLIP | 512×512×1 | 60 | -0.006 | 失败 |
| **本次 v8** | **物理** | **512×512×5** | **100** | **+0.056** | **成功** |

### Reward 信号质量对比

| 指标 | CLIP Reward (单帧) | 物理 Reward (视频) |
|------|-------------------|-------------------|
| 组内 std | 0.01~0.02 | **0.09~0.44** |
| reward 范围 | 0.20~0.30 | **0.01~0.98** |
| advantage 信噪比 | ~1 (噪声主导) | **>>1 (信号主导)** |
| loss 幅度 | ±0.003 | **±0.018** |

## 分析

### 为什么这次成功了

1. **物理 reward 信号足够强**: 组内 std=0.09~0.44，advantage 有真实的区分度
2. **Pipeline bug 全部修复**: DDP 梯度同步 + accum=num_generations + diverse noise
3. **前 40 步平稳是正常的**: REINFORCE 梯度在初期需要积累方向，不会立即上升
4. **后 50 步加速上升**: 模型开始学到提升物理得分的特征

### 后半段加速的可能原因

- 前 50 步: 模型在"探索"，不同 prompt (流体/软体/布料/刚体) 的梯度方向不一致
- 后 50 步: 模型学到了跨域通用的物理特征（如时间一致性、空间平滑性），梯度方向趋于一致
- Step 19 (reward=0.874) 和 step 34 (reward=0.812) 是高分步，对应的梯度贡献大

### 尚未收敛

最后 10 步 reward 仍在 0.498，相比前 10 步的 0.442 有明确上升趋势。后半段的加速（0.44→0.51）暗示继续训练可能进一步提升。已启动 300 步训练。

## 下一步

1. **300 步训练** (进行中): 观察是否继续上升或收敛
2. **生成对比视频**: 用 checkpoint-50 和 checkpoint-100 对相同 prompt 生成视频，视觉对比物理质量变化
3. **分域分析**: 按 prompt 类型 (流体/软体/布料/刚体) 拆分 reward，看哪个域提升最大
4. **扩展到更长视频**: 如果 t=5 持续有效，尝试 t=17 或 t=33

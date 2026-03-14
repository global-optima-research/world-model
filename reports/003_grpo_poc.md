# 实验报告 003: 物理 Reward GRPO 训练 PoC

**日期**: 2026-03-14
**目的**: 验证 PhysicsRewardModel + DanceGRPO 训练 pipeline 的端到端可行性，观察 reward 是否在训练过程中提升

## 复现信息

- **Commit**: `c21d8ee` (fix: multi-epoch training loop + accelerate PoC params)
- **硬件**: 2× NVIDIA RTX 5090 (GPU 5, 7)
- **Conda 环境**: `wan2`
- **运行命令**:
```bash
ssh 5090
source /data/Anaconda3/etc/profile.d/conda.sh && conda activate wan2
cd ~/world-model && git checkout c21d8ee

# 预计算 embeddings (已完成)
CUDA_VISIBLE_DEVICES=7 python scripts/preprocess_physics_embeddings.py \
    --model_path LongLive/wan_models/Wan2.1-T2V-1.3B \
    --prompt_file scripts/physics_prompts.txt \
    --output_dir data/physics_rl_embeddings

# 训练
export WANDB_MODE=disabled
CUDA_VISIBLE_DEVICES=5,7 bash scripts/train_grpo_physics.sh
```

## 训练配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | Wan2.1-T2V-1.3B (diffusers) | ~2.8GB |
| 视频分辨率 | 480×832×33 帧 | 与 baseline 一致 |
| sampling_steps | 10 | 快速采样 (baseline 用 20) |
| num_generations | 4 | 每 prompt 生成 4 候选 |
| learning_rate | 1e-5 | |
| clip_range | 1e-2 | PPO clipping |
| gradient_accumulation_steps | 4 | |
| max_train_steps | 50 | 5 个 epoch |
| eta (SDE) | 0.3 | log-prob 计算用 |
| shift | 3 | timestep shift |
| cfg_infer | 5.0 | 推理时 CFG |
| Reward | PhysicsRewardModel (auto mode) | 光流 + 流体物理 |

## 结果

### 训练概况

- **总耗时**: 2 小时 2 分钟 (50 步, ~145s/步)
- **Checkpoint**: `outputs/grpo_physics/checkpoint-25` (step 25)
- **保存视频**: `videos/physics_0_{0-3}.mp4` (step 0 的 4 个候选)

### Reward 趋势

将 50 步的 reward 按 epoch 汇总 (每 epoch 10 步, 每步 8 个 reward 值):

| Epoch | 平均 Reward | 标准差 | 最小 | 最大 |
|-------|------------|--------|------|------|
| 0 (step 1-10) | 0.533 | 0.228 | 0.054 | 0.917 |
| 1 (step 11-20) | 0.494 | 0.208 | 0.035 | 0.875 |
| 2 (step 21-30) | 0.504 | 0.252 | 0.034 | 0.955 |
| 3 (step 31-40) | 0.502 | 0.242 | 0.040 | 0.897 |
| 4 (step 41-50) | 0.534 | 0.189 | 0.035 | 0.919 |

**结论: Reward 没有系统性提升。** 各 epoch 的平均 reward 在 0.49~0.53 之间波动，无上升趋势。

### Loss 与梯度

| 指标 | 全程表现 |
|------|---------|
| Loss | 始终在 ±0.0000 附近 (四位小数精度下为 0) |
| Grad norm | 0.012 ~ 0.039，中位数 ~0.018 |
| 每步 loss 明细 | 单样本 loss 在 ±0.07 范围内，但 accumulation 后抵消为 ~0 |

### 逐步 loss 和 grad_norm

| Step | Loss | Grad Norm | Epoch |
|------|------|-----------|-------|
| 1 | -0.0000 | 0.016 | 0 |
| 5 | -0.0000 | 0.015 | 0 |
| 10 | -0.0000 | 0.015 | 0 |
| 15 | -0.0000 | 0.020 | 1 |
| 20 | -0.0000 | 0.020 | 1 |
| 25 | 0.0000 | 0.020 | 2 |
| 30 | -0.0000 | 0.014 | 2 |
| 35 | 0.0000 | 0.015 | 3 |
| 40 | 0.0000 | 0.016 | 3 |
| 45 | -0.0000 | 0.020 | 4 |
| 50 | 0.0000 | 0.019 | 4 |

## 分析

### 为什么模型没学到?

1. **PPO clipping 过紧 (`clip_range=1e-2`)**
   - 单样本 loss 在 ±0.07 范围内是合理的
   - 但 PPO surrogate loss 被 clip 到 ±0.01，大幅度截断了梯度信号
   - 4 个 generation 的正负 advantage 进一步抵消，导致 accumulated loss ≈ 0

2. **Gradient accumulation 稀释了信号**
   - `gradient_accumulation_steps=4`，但 batch_size=1
   - 每个 accumulation step 的 loss 方向不一致（正负 advantage 交替），梯度在累积过程中互相抵消

3. **GRPO advantage 的组内归一化**
   - `use_group=True` 导致同一 prompt 的 4 个 generation 做组内标准化
   - advantage 均值为 0，正负各半，进一步使梯度方向不稳定

4. **采样随机性主导了 reward 方差**
   - 同一 prompt 不同 generation 的 reward 差异来自采样噪声，而非策略改进
   - 10 步 denoising 的质量较差，reward 方差更多反映随机性

### 对比 DanceGRPO 原始配置

| 参数 | DanceGRPO (Wan, HPSv2) | 本次 PoC | 差异 |
|------|----------------------|---------|------|
| clip_range | 1e-4 | 1e-2 | PoC 更大 100x，但仍不够 |
| num_generations | 4 | 4 | 相同 |
| sampling_steps | 1 (单帧!) | 10 | DanceGRPO 只生成图像 |
| gradient_accumulation | 4 | 4 | 相同 |
| GPU 数量 | 8 | 2 | PoC 少 4x |
| lr | 5e-6 | 1e-5 | PoC 更大 2x |

**关键发现**: DanceGRPO 原始 Wan 脚本用 `--t 1` 只生成单帧图像，不是视频。这是核心区别 — 单帧采样 (1 步) 远快于 33 帧视频 (10-20 步)，且 reward 信号更直接。

## 已解决的工程问题

本次实验解决了以下 pipeline 问题:

1. **T5 embedding 格式不兼容**: LongLive 的 Wan 模型使用自定义 T5 (`.pth`)，需用 `wan.modules.t5.T5EncoderModel` 加载
2. **模型格式不兼容**: LongLive 模型是原始 Wan 格式 (`WanModel`)，GRPO 需要 diffusers 格式 (`WanTransformer3DModel`)，已切换到 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers`
3. **HuggingFace 网络不通**: 5090 服务器无法访问 huggingface.co，改用本地 HF cache 路径
4. **Dataloader caption 类型**: collate 返回 tuple 而非 list，`isinstance` 检查需包含 tuple
5. **单 epoch 过早退出**: 20 条数据 / 2 GPU = 10 步，需多 epoch 循环
6. **Wandb 未配置**: 添加 `WANDB_MODE=disabled` 支持

## 下一步

### 方案 A: 单帧 PoC (推荐)

与 DanceGRPO 原始配置对齐，先用 `--t 1` (单帧图像) 验证 GRPO 梯度流通：
- 采样更快 (1 步 vs 10 步)
- Reward 可用 HPSv2 先验证 pipeline，再切物理 reward
- 确认 reward 能提升后，再扩展到视频

### 方案 B: 调参继续视频训练

- 去掉 `gradient_accumulation_steps` (设为 1)
- 增大 `clip_range` 到 0.1~0.2
- 增大 `num_generations` 到 8 (需更多 GPU 或更小分辨率)
- 降低分辨率到 320×576 以加速

### 方案 C: 检查梯度流

- 做一个 sanity check: 固定 prompt，手动设定高/低 reward，验证梯度方向正确
- 检查 FSDP 是否导致梯度 sync 问题

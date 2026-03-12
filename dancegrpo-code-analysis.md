# DanceGRPO 代码分析 — Wan 2.1 GRPO 实现

**日期**: 2026-03-13
**代码版本**: DanceGRPO main branch (git clone --depth 1, 2026-03-12)
**核心文件**: `fastvideo/train_grpo_wan_2_1.py`

## 1. 整体架构

```
Prompt Text
    ↓ (离线预处理)
Prompt Embeddings (T5) + Negative Embeddings
    ↓ (训练时加载)
┌─────────────────────────────────────────────┐
│              train_one_step()                │
│                                             │
│  1. repeat prompt G 次 (num_generations)     │
│  2. sample_reference_model()                │
│     ├─ 生成 G 个候选视频 (no_grad)            │
│     ├─ VAE decode → pixel video             │
│     ├─ Reward Model 打分                    │
│     └─ 返回 latents, log_probs, rewards     │
│  3. 计算 group relative advantage           │
│  4. grpo_one_step() × 每个 timestep         │
│     ├─ 重新前向 (有梯度)                     │
│     ├─ 计算 new_log_prob                    │
│     ├─ ratio = exp(new - old)               │
│     └─ clipped surrogate loss               │
│  5. backward + optimizer.step               │
└─────────────────────────────────────────────┘
```

## 2. GRPO 算法细节

### 2.1 采样阶段 (`sample_reference_model`, L263-371)

```python
# 对每个 prompt，生成 G 个候选
for index, batch_idx in enumerate(batch_indices):
    # 每个候选独立噪声 (除非 --init_same_noise)
    input_latents = torch.randn((1, 16, latent_t, latent_h, latent_w))

    with torch.no_grad():
        z, latents, batch_latents, batch_log_probs = run_sample_step(...)

    # VAE 解码
    video = vae.decode(latents)
    export_to_video(video, f"./videos/wan_2_1_{rank}_{index}.mp4")

    # Reward 打分
    reward = reward_model(first_frame, text)  # 当前: HPSv2
```

**关键**: 采样阶段 `no_grad`，只记录 latent 轨迹和 log_prob。

### 2.2 Advantage 计算 (L456-471)

```python
# Group Relative: 每个 prompt 内部归一化
if args.use_group:
    for i in range(n):
        group_rewards = rewards[i*G : (i+1)*G]
        advantages[i*G : (i+1)*G] = (group_rewards - group_rewards.mean()) / (group_rewards.std() + 1e-8)
else:
    # 全局归一化 (跨所有 GPU gather)
    advantages = (rewards - gathered_reward.mean()) / (gathered_reward.std() + 1e-8)
```

### 2.3 策略更新 (`grpo_one_step`, L225-259)

```python
# 重新前向计算 log_prob (有梯度)
transformer.train()
pred = transformer(hidden_states=latents, timestep=timesteps, ...)
z, pred_original, log_prob = flux_step(..., grpo=True, sde_solver=True)
return log_prob
```

### 2.4 Loss 计算 (L496-524)

```python
ratio = torch.exp(new_log_probs - old_log_probs)
advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)

unclipped_loss = -advantages * ratio
clipped_loss = -advantages * torch.clamp(ratio, 1-clip_range, 1+clip_range)
loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
```

PPO-style clipped surrogate objective，和 DeepSeek-R1 的 GRPO 一致。

### 2.5 Log Probability 计算 (`flux_step`, L94-133)

```python
# SDE solver: 在 denoising step 上加噪声，计算转移概率
prev_sample_mean = latents + dsigma * model_output  # 均值
std_dev_t = eta * sqrt(delta_t)                      # 标准差

# log prob = log N(prev_sample | prev_sample_mean, std_dev_t²)
log_prob = -((prev_sample - prev_sample_mean)² / (2 * std_dev_t²)) - log(std_dev_t) - log(sqrt(2π))
log_prob = log_prob.mean(dim=(1,2,3,4))  # 在空间维度上取平均
```

**核心洞察**: GRPO 需要将 diffusion 过程转化为 MDP，每个 denoising step 是一个 action。通过 SDE solver 添加噪声 (eta>0) 来使策略随机化，从而能计算 log probability。

## 3. 模型与数据

### 3.1 模型加载

```python
# 直接用 diffusers 的 Wan Transformer
from diffusers import AutoencoderKLWan, WanTransformer3DModel

transformer = WanTransformer3DModel.from_pretrained(
    "data/Wan2.1-T2V-1.3B", subfolder="transformer", torch_dtype=torch.bfloat16)

vae = AutoencoderKLWan.from_pretrained(
    "data/Wan2.1-T2V-1.3B", subfolder="vae", torch_dtype=torch.bfloat16)
```

**注意**: 这用的是 diffusers 版本的 Wan 2.1，不是 LongLive 的自定义版本。LongLive 的 CausalWanModel 额外加了 causal attention + KV cache + self-forcing。

### 3.2 数据预处理

```bash
# 预计算 prompt embeddings (避免每次训练都跑 T5)
torchrun fastvideo/data_preprocess/preprocess_wan_2_1_embeddings.py \
    --model_path ./data/Wan2.1-T2V-1.3B \
    --output_dir data/rl_embeddings \
    --prompt_dir "./assets/prompts.txt"
```

输出结构:
```
data/rl_embeddings/
├── prompt_embed/          # T5 编码后的 prompt embedding (.pt)
├── negative_prompt_embeds/ # 负向 prompt embedding (.pt)
└── videos2caption.json    # 索引文件
```

### 3.3 Reward Model

当前实现**只有 HPSv2** (Human Preference Score v2):
- 对视频第一帧提取 ViT-H-14 image features
- 对 prompt 提取 text features
- 计算 cosine similarity 作为 reward
- **仅评估图像质量和 text-image alignment，不评估物理正确性**

## 4. 训练参数 (官方 Wan 2.1 脚本)

| 参数 | 值 | 说明 |
|------|-----|------|
| pretrained_model | Wan2.1-T2V-1.3B | 基础模型 |
| h × w × t | 512 × 512 × **1** | **注意: 只生成单帧!** |
| sampling_steps | 20 | denoising 步数 |
| eta | 0.3 | SDE 噪声系数 |
| shift | 3 | timestep schedule shift |
| num_generations | 12 | 每个 prompt 生成候选数 |
| learning_rate | 1e-5 | 学习率 |
| gradient_accumulation | 24 | 梯度累积步数 |
| clip_range | 1e-4 | PPO clip 范围 |
| adv_clip_max | 5.0 | advantage 裁剪 |
| cfg_infer | 5.0 | classifier-free guidance |
| max_train_steps | 1000 | 总训练步数 |
| gradient_checkpointing | ✅ | 节省显存 |

## 5. 对接 LongLive + 物理 Reward 的关键修改点

### 5.1 Reward 替换 (最核心)

**位置**: `sample_reference_model()` L353-365

当前:
```python
if args.use_hpsv2:
    image = preprocess_val(first_frame).unsqueeze(0)
    text = tokenizer([caption]).to(device)
    hps_score = reward_model(image, text)
    all_rewards.append(hps_score)
```

替换为:
```python
if args.use_physics_reward:
    # 1. 解码完整视频 (多帧)
    # 2. 光流估计 (RAFT)
    # 3. 物理模拟 (Warp)
    # 4. 对比打分
    physics_score = physics_reward(decoded_video, caption)
    all_rewards.append(physics_score)
```

### 5.2 单帧 → 多帧视频

**当前限制**: `--t 1` 只生成单帧图像。物理 reward 需要多帧视频才能评估运动。

需要修改:
- `--t 33` (或更多) 以生成视频序列
- latent shape: `(1, 16, latent_t, latent_h, latent_w)` 自动适配
- 显存压力增大: 需要降低 `num_generations` 或分辨率

### 5.3 LongLive vs diffusers Wan 2.1

两种路径:

**路径 A: 在 DanceGRPO 框架上用 diffusers Wan 2.1**
- 优点: 代码改动最小，DanceGRPO 已经支持
- 缺点: 没有 LongLive 的 causal attention + self-forcing 优化

**路径 B: 将 LongLive 模型接入 DanceGRPO**
- 优点: 利用 LongLive 的高效长视频生成 (KV cache + causal)
- 缺点: 需要适配 LongLive 的自定义推理接口到 GRPO 框架

**推荐: 先走路径 A 快速验证，再走路径 B 做正式训练。**

### 5.4 显存估算 (8× RTX 5090, 32GB/卡)

| 组件 | 显存 |
|------|------|
| Wan 2.1 1.3B (bf16) | ~3 GB |
| VAE (bf16) | ~1 GB |
| 优化器状态 (fp32) | ~5 GB |
| 生成 latents (t=33, 512×512) | ~2 GB |
| VAE decode (多帧) | ~4 GB |
| 物理 Reward (Warp + RAFT) | ~3 GB |
| 梯度 + 激活 (gradient checkpointing) | ~10 GB |
| **总计** | **~28 GB** ✅ |

单卡可行（gradient checkpointing 必须开）。`num_generations` 需从 12 降到 4-6。

## 6. 实现路线图

```
Phase 1: 数据准备 + Reward PoC
├─ 预计算物理 prompt embeddings (20 条)
├─ 实现 PhysicsRewardModel 类
│   ├─ RAFT 光流提取
│   ├─ Warp SPH 流体模拟
│   └─ 速度场对比 reward
└─ 单独验证 reward 信号质量

Phase 2: GRPO 集成
├─ fork train_grpo_wan_2_1.py → train_grpo_physics.py
├─ 替换 HPSv2 → PhysicsRewardModel
├─ 修改 --t 1 → --t 33 (多帧视频)
├─ 调整 num_generations, 分辨率
└─ 单域 PoC (fluid only)

Phase 3: 扩展
├─ 添加软体 (FEM) 和布料 (cloth) reward
├─ 多域联合训练
└─ 消融实验
```

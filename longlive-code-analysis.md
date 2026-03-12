# LongLive 代码架构深度分析

> 基于源码阅读，分析日期：2026-03-12

---

## 整体架构

```
train.py                          # 入口：加载 config → 创建 ScoreDistillationTrainer → train()
├── configs/
│   ├── default_config.yaml       # 默认超参
│   ├── longlive_train_init.yaml  # 阶段1：初始蒸馏（21帧短视频）
│   └── longlive_train_long.yaml  # 阶段2：流式长序列训练（240帧）
├── trainer/
│   └── distillation.py           # ScoreDistillationTrainer — FSDP分布式训练循环
├── model/
│   ├── base.py                   # BaseModel → SelfForcingModel（反向模拟+生成）
│   ├── dmd.py                    # DMD（Distribution Matching Distillation）核心损失
│   ├── dmd_switch.py             # DMDSwitch（支持 prompt 切换的 DMD）
│   └── streaming_training.py     # StreamingTrainingModel — 长序列流式训练封装
├── pipeline/
│   ├── self_forcing_training.py  # SelfForcingTrainingPipeline — 训练时的逐块去噪
│   ├── streaming_training.py     # StreamingTrainingPipeline
│   ├── streaming_switch_training.py  # 支持 prompt 切换的流式训练
│   ├── causal_inference.py       # 推理时的因果生成
│   ├── switch_causal_inference.py    # 带 KV-recache 的推理
│   └── interactive_inference.py  # 交互式推理（键盘操控）
├── wan/
│   ├── modules/
│   │   ├── causal_model.py       # ⭐ CausalWanModel — 核心因果 Transformer
│   │   ├── causal_model_infinity.py  # 无限长度推理版本
│   │   ├── model.py              # 原版 Wan 模型
│   │   ├── attention.py          # 注意力实现
│   │   ├── vae.py                # Wan VAE（16通道）
│   │   ├── t5.py                 # UMT5-XXL 文本编码器
│   │   └── clip.py               # CLIP 编码器
│   └── utils/
│       ├── fm_solvers.py         # Flow Matching 求解器
│       └── fm_solvers_unipc.py   # UniPC 求解器
└── utils/
    ├── wan_wrapper.py            # Wan 模型封装（WanDiffusionWrapper/VAE/TextEncoder）
    ├── scheduler.py              # FlowMatchScheduler
    ├── dataset.py                # 文本 prompt 数据集
    └── loss.py                   # 去噪损失函数
```

---

## 核心组件详解

### 1. 三模型 DMD 蒸馏架构

LongLive 使用 **三个模型** 进行蒸馏训练：

```
┌─────────────────────────────────────────────────────────────────┐
│                    DMD 蒸馏训练架构                               │
│                                                                 │
│  Generator (Student)     Wan 1.3B Causal DiT   ← 需要训练       │
│  Real Score (Teacher)    Wan 14B 原版 DiT       ← 冻结          │
│  Fake Score (Critic)     Wan 1.3B 原版 DiT      ← 需要训练       │
│                                                                 │
│  训练交替：                                                      │
│    每 5 步: 1 步 generator loss + 5 步 critic loss               │
│    (dfake_gen_update_ratio: 5)                                  │
│                                                                 │
│  Generator Loss = DMD KL 散度梯度                                │
│    grad = fake_score(x_t) - real_score(x_t)                     │
│    loss = ||x - (x - grad)||²                                   │
│                                                                 │
│  Critic Loss = 去噪重建损失                                      │
│    fake_score 学习预测 generator 生成分布的 score                  │
└─────────────────────────────────────────────────────────────────┘
```

**关键文件**: `model/dmd.py`
- `generator_loss()`: 先用 generator 生成视频（Self-Forcing），再用 DMD loss 优化
- `critic_loss()`: 用 generator 的输出训练 critic（fake_score）
- `_compute_kl_grad()`: DMD 核心——计算 real/fake score 差异作为梯度

### 2. CausalWanModel — 因果 Transformer

**关键文件**: `wan/modules/causal_model.py`

核心设计：
- 基于 Wan 2.1 的 DiT 架构，改造为 **因果（Causal）** 版本
- 30 个 Transformer blocks
- 每帧 1560 个 token（空间分辨率 60×104 的 latent / 4 = 15×26 × 4 patches?）
- 使用 **FlexAttention**（PyTorch 原生编译的 flex_attention）+ causal mask

注意力机制:
```
CausalWanSelfAttention:
  - 支持 KV Cache（推理和流式训练时复用历史帧的 K/V）
  - local_attn_size: 局部注意力窗口（帧数），默认12帧
  - sink_size: Frame Sink 帧数，默认3帧（保留最早的3帧做全局锚点）
  - 注意力模式: Sink帧(全局) + 局部窗口(最近12帧) = 因果注意力
  - max_attention_size: 控制注意力范围的上限
```

RoPE 修改:
```
causal_rope_apply(): 支持 start_frame 偏移，使得流式生成时
每个 chunk 的位置编码能正确接续上一个 chunk
```

### 3. Self-Forcing 训练 Pipeline

**关键文件**: `pipeline/self_forcing_training.py`

训练时的视频生成流程：
```
对每个 block（3帧一组）:
  1. 从噪声开始
  2. 多步去噪（4步：1000→750→500→250）
     - 随机选一步退出（保留梯度），其余步 no_grad
     - 每步调用 generator 预测 x0
     - 非最后步：add_noise 到下一个 timestep
  3. 最终输出存入 output
  4. 用 denoised 结果（+context_noise）更新 KV Cache
  5. 移动到下一个 block
```

这就是 **Self-Forcing** 的核心——训练时用模型自己的输出作为下一步输入（而非 teacher-forcing），减少 train/inference mismatch。

### 4. 流式长序列训练

**关键文件**: `model/streaming_training.py`

```
StreamingTrainingModel:
  设置:
    chunk_size = 21 帧（固定损失计算单位）
    max_length = 240 帧（最长序列）
    min_new_frame = 18 帧（每 chunk 最少生成的新帧）

  流程:
    setup_sequence() → 初始化 KV Cache + 条件信息
    while can_generate_more():
      generate_next_chunk() →
        1. 随机选择新帧数量（18-21帧，步长3）
        2. 从前一 chunk 取 overlap 帧（0-3帧）
        3. 拼接成 21 帧的完整 chunk
        4. 调用 pipeline.generate_chunk_with_cache()
        5. 更新 previous_frames 和 current_length
      compute_generator_loss(chunk) → DMD loss
      compute_critic_loss(chunk) → denoising loss
```

### 5. 两阶段训练配置

**阶段 1: Init（短视频蒸馏）**
```yaml
# configs/longlive_train_init.yaml
real_name: Wan2.1-T2V-14B          # Teacher: 14B
fake_name: Wan2.1-T2V-1.3B         # Student/Critic: 1.3B
denoising_step_list: [1000, 750, 500, 250]  # 4步去噪
num_training_frames: 21             # 21帧（~2.6秒）
local_attn_size: 12                 # 局部注意力12帧
sink_size: 3                        # 3帧 sink
lr: 2e-6                            # 学习率
max_iters: 700                      # 700步
total_batch_size: 64                 # 全局 batch
```

**阶段 2: Long（流式长序列训练）**
```yaml
# configs/longlive_train_long.yaml
streaming_training: true
streaming_chunk_size: 21            # 每 chunk 21帧
streaming_max_length: 240           # 最长240帧（~30秒）
streaming_min_new_frame: 18         # 每次至少生成18帧新帧
lr: 1e-5                            # 学习率更高（5x）
max_iters: 3000                     # 3000步
distribution_loss: dmd_switch       # 支持 prompt 切换

# LoRA 配置
adapter:
  type: "lora"
  rank: 256
  alpha: 256
```

---

## 关键修改点分析（如果要基于此 codebase 做研究）

### 修改点 1: 时序骨干替换（Transformer → SSM/Mamba）
```
文件: wan/modules/causal_model.py
位置: CausalWanModel 类
影响: CausalWanSelfAttention → 需要替换为 Mamba-2 SSD 层
难度: ★★★★☆
原因:
  - Self-Attention 与 SSM 的接口不同（SSM 无 KV Cache，用隐状态）
  - FlexAttention + causal mask 逻辑需要完全重写
  - RoPE 位置编码需要适配（SSM 不需要 RoPE）
  - KV Cache 机制需要替换为 SSM state management
```

### 修改点 2: 生成范式替换（Diffusion → Flow Matching）
```
文件: utils/scheduler.py, pipeline/self_forcing_training.py
位置: FlowMatchScheduler + 去噪循环
影响:
  - scheduler.add_noise() / convert_x0_to_noise() 逻辑
  - 4步去噪循环可简化为 1-2 步（Flow Matching 天然少步）
  - DMD 蒸馏可能不再需要（如果 FM 本身就够快）
难度: ★★★☆☆
注意: LongLive 的 scheduler 已经用了 FlowMatchScheduler！
  denoising_loss_type: "flow" 表明它在 loss 层面已用 flow 预测。
  但生成范式仍是 DDPM 式多步去噪 + DMD 蒸馏。
```

### 修改点 3: 物理奖励注入（RLVR post-training）
```
文件: 需新增 reward model + RL trainer
位置: 在 DMD 训练之后，额外加一个 RL 阶段
影响:
  - 需要物理奖励函数（动量守恒、碰撞检测等）
  - 需要 GRPO/PPO 优化器
  - 需要物理评估 benchmark
难度: ★★★★☆
优势: 不需要修改 LongLive 的架构，只在训练后追加
```

### 修改点 4: 损失函数扩展
```
文件: model/dmd.py, utils/loss.py
位置: compute_distribution_matching_loss()
影响: 可在 DMD loss 之外叠加额外的物理一致性损失
难度: ★★☆☆☆
```

---

## 训练资源估算

基于配置文件推算：

| 阶段 | 步数 | Batch | 帧数 | 模型 | 估计时间(8×H800) |
|------|------|-------|------|------|-------------------|
| Init | 700 | 64 | 21 | 1.3B+14B+1.3B | ~1-2天 |
| Long | 3000 | 64 | 240 | 1.3B+14B+1.3B(LoRA) | ~2-3天 |
| 总计 | - | - | - | - | **~4天** |

---

## 重要发现

1. **LongLive 已经在用 Flow Matching Scheduler**：`denoising_loss_type: flow`，它的 scheduler 就是 FlowMatchScheduler。但它仍然用 4 步去噪 + DMD 蒸馏来加速，而不是直接利用 FM 的少步特性。

2. **训练不需要视频数据**：DMD 蒸馏使用反向模拟（backward simulation）生成训练样本，只需要文本 prompt。数据文件是 `prompts/vidprom_filtered_extended.txt`。

3. **LoRA 仅在长序列阶段使用**：阶段2用 rank=256 的 LoRA 微调 generator 和 critic，而非全量微调。

4. **CC-BY-NC-SA-4.0**：`causal_model.py` 使用非商业许可（继承自 Self-Forcing），但 `train.py` 等其他文件是 Apache-2.0。需要注意许可兼容性。

5. **Prompt 切换（KV-recache）是核心创新**：`DMDSwitch` 和 `StreamingSwitchTrainingPipeline` 实现了长序列中途切换 prompt 的能力，这是交互式世界模型的关键。

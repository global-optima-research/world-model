# Oasis: A Universe in a Transformer

> Decart & Etched, 2024
> 项目页: https://oasis-model.github.io/
> 代码: https://github.com/etched-ai/open-oasis

---

## 一句话总结

Oasis 是第一个完全基于 Transformer 的实时交互式游戏世界模型，通过 Spatial Autoencoder (tokenizer) + Diffusion Transformer (DiT) 的架构，将 Minecraft 游戏帧自回归地生成出来，达到实时可玩的 20+ FPS（在 Etched Sohu 芯片上可达 ~50 FPS），证明了 Transformer 可以端到端地作为游戏引擎运行。

---

## 核心贡献

1. **首个实时可交互的 AI 游戏引擎**: 不需要传统游戏引擎，纯神经网络接收玩家动作输入、生成下一帧画面，实现了"Transformer 即游戏引擎"的概念验证。

2. **DiT + Autoregressive 帧生成框架**: 将 Diffusion Transformer 与自回归时序建模结合——每一帧通过 DiT 去噪生成，帧间通过自回归方式传递上下文，解决了"高质量单帧生成"和"时序连贯性"的双重需求。

3. **轻量可玩**: 开源的 500M 参数模型（对比 Sora 等数十亿参数的视频模型）即可产生可玩质量的 Minecraft 世界，证明世界模型不一定需要巨大规模。

4. **动作条件化设计**: 提出了将离散玩家动作（键盘 + 鼠标）注入 DiT 的有效机制，使模型能精确响应玩家操控。

5. **硬件协同设计的推理优化**: 与 Etched 的 Sohu ASIC 芯片协同设计，展示了专用硬件如何将世界模型推理加速到实时交互水平。

---

## 方法详解

### 整体架构设计

Oasis 的架构分为三个核心组件：

```
玩家动作 (keyboard/mouse)
       ↓
[Action Encoder] → action embedding
       ↓
[DiT Backbone] ← 上一帧(s) 的 latent tokens (自回归输入)
       ↓                              ↑
   去噪 T 步 ──────────────────────────┘
       ↓
[Spatial Decoder (VAE Decoder)]
       ↓
   生成的当前帧 (RGB pixels)
```

**自回归 + Diffusion 的结合方式**:
- **帧间 (inter-frame)**: 自回归。前一帧（或前几帧）的 latent representation 作为条件传入当前帧的生成过程。模型维护一个滑动窗口的上下文，类似 GPT 的 causal attention，但操作对象是帧级别的 token 序列。
- **帧内 (intra-frame)**: Diffusion。当前帧从纯噪声开始，经过 T 步去噪得到干净的 latent，再解码为像素。使用 DDPM 风格的去噪过程，但步数被大幅压缩（推理时使用 DDIM 等加速采样，约 4-8 步）。

这种设计的关键洞察是：**自回归保证了时序连贯性（每帧都看到前面的帧），Diffusion 保证了单帧的生成质量（比纯自回归 token 预测的视觉质量高得多）**。

### Tokenizer 设计

Oasis 使用一个 **Spatial Autoencoder**（本质上是 VQ-VAE / VAE 变体）作为 tokenizer：

- **编码器**: 将 256x256（或更高分辨率）的 RGB 帧编码为空间 latent tokens。空间下采样倍率为 8x，即 256x256 的图像变为 32x32 的 latent grid。
- **Latent 维度**: 每个 spatial position 的 latent vector 维度为 4（类似 Stable Diffusion 的 VAE）。
- **解码器**: 将 latent tokens 解码回像素空间。
- **训练**: Tokenizer 单独预训练，使用重建 loss + KL 正则化（或 VQ loss），在 Minecraft 帧数据上微调以适配游戏画面的特征分布。

开源代码中使用的是基于 Stable Diffusion VAE 架构的变体，具体来说是 **ViT-based spatial tokenizer**，将每帧 patch 化后编码。

### 动作条件化注入

玩家动作包含两部分：
- **离散键盘动作**: W/A/S/D 移动、跳跃、攻击等（约 7-8 个离散动作）
- **连续鼠标动作**: 视角旋转的 dx/dy（连续值）

注入方式：
1. **动作编码**: 离散动作通过 embedding table 映射为向量，连续鼠标位移通过 MLP 编码。两者拼接后通过一个小型 MLP 融合为统一的 action embedding。
2. **注入位置**: Action embedding 通过 **Adaptive Layer Normalization (AdaLN)** 注入 DiT 的每一层——与 DiT 中 timestep embedding 的注入方式一致。具体地，action embedding 与 diffusion timestep embedding 相加（或拼接后投影），共同调制 DiT block 中 LayerNorm 的 scale 和 shift 参数。
3. **这种设计的优势**: AdaLN 是一种全局条件化机制，不增加序列长度（不像 cross-attention 那样需要额外的 KV），计算开销极小但影响每一层的每一个 token。

```
action_emb = MLP(concat(keyboard_embed, mouse_embed))
condition = action_emb + timestep_emb
# 在每个 DiT block 中:
scale, shift = Linear(condition)
x = LayerNorm(x) * (1 + scale) + shift
```

### 训练细节

- **数据集**: 大规模 Minecraft 游戏录像数据。具体包括：
  - 来自互联网的 Minecraft 游戏视频（YouTube 等）
  - 配对的键盘/鼠标动作记录
  - 数据量级约在数百小时到数千小时之间（具体数字未完全公开，但远超 DIAMOND 的 87 小时）

- **训练 Loss**:
  - **Diffusion Loss**: 标准的 DDPM 噪声预测 loss（epsilon-prediction 或 v-prediction）。给定干净 latent x_0，加噪到 x_t，模型预测噪声 epsilon：
    ```
    L_diffusion = E[||epsilon - epsilon_theta(x_t, t, c)||^2]
    ```
    其中 c 包含前帧 latent + action embedding + timestep。
  - **Tokenizer Loss**: 单独训练，reconstruction loss + perceptual loss + KL/VQ regularization。

- **训练规模**:
  - 完整模型（未开源）: 据报道参数量级更大
  - 开源版本: **500M 参数**
  - 训练硬件: 大规模 GPU 集群
  - 训练时长: 未详细公开

### 500M 模型的具体架构参数

开源的 500M DiT 模型（open-oasis）的架构参数：

| 参数 | 值 |
|------|-----|
| 模型总参数量 | ~500M |
| DiT Blocks 层数 | 24 层 |
| Hidden dimension | 1024 |
| Attention heads | 16 |
| Head dimension | 64 (1024/16) |
| MLP hidden dim | 4096 (4x hidden dim) |
| Patch size | 2x2 (在 latent space 上) |
| 输入分辨率 | 360x640 像素 |
| Latent 分辨率 | 45x80 (8x 下采样) |
| Latent patch 后序列长度 | ~900 tokens (45/2 * 80/2 ≈ 22*40) |
| 上下文帧数 | 前 1 帧（可扩展） |
| Diffusion 采样步数 (推理) | DDIM ~4-8 步 |
| Noise schedule | Linear / Cosine |
| Tokenizer | 基于 Stable Diffusion VAE (latent dim=4, 8x 下采样) |

DiT Block 结构（每层）：
```
Input tokens
  → AdaLN (conditioned on action + timestep)
  → Multi-Head Self-Attention (causal mask for autoregressive)
  → Residual connection
  → AdaLN
  → MLP (GELU, expand 4x)
  → Residual connection
```

---

## 实验结果

### 生成质量指标

| 指标 | Oasis (500M) | 说明 |
|------|-------------|------|
| FID (Frechet Inception Distance) | ~20-30 范围 | 在 Minecraft 帧上评估，质量接近真实游戏帧 |
| LPIPS (感知相似度) | 具有竞争力 | 时序连贯帧之间的感知距离较低 |
| FVD (Frechet Video Distance) | 优于纯自回归 token 方法 | 视频级别质量评估 |
| 视觉质量 | 主观评价高 | 生成的 Minecraft 世界视觉上非常接近真实游戏，纹理、光照、方块结构都比较准确 |

**定性观察**:
- 能正确生成 Minecraft 的方块世界、树木、水面、天空等元素
- 对玩家动作响应基本准确（移动、视角转动、挖掘等）
- 长时间游玩后会出现"漂移"——世界一致性逐渐下降
- 偶尔出现几何结构错误（方块穿模、结构突变）

### 推理速度

| 配置 | FPS | 延迟 |
|------|-----|------|
| A100 GPU (单卡, 500M) | ~12-20 FPS | ~50-80ms/帧 |
| Etched Sohu ASIC (专用芯片) | ~50+ FPS | ~20ms/帧 |
| H100 GPU (优化后) | ~20+ FPS | ~40-50ms/帧 |
| 消费级 GPU (4090) | ~8-15 FPS | 依赖去噪步数 |

注：FPS 高度依赖于 DDIM 采样步数。4 步采样 vs 8 步采样可以带来近 2x 的速度差异。

### 与 GameNGen 等的对比

| 维度 | Oasis (Decart/Etched) | GameNGen (Google) | DIAMOND |
|------|----------------------|-------------------|---------|
| 游戏 | Minecraft | DOOM | Atari (多个) |
| 架构 | DiT (Diffusion Transformer) | Augmented Stable Diffusion (U-Net) | Diffusion (U-Net) |
| 参数量 | 500M (开源) | ~1.6B (Stable Diffusion 基础) | ~100M |
| 分辨率 | 360x640 | 320x240 | 84x84 |
| FPS | 20+ (A100) / 50+ (Sohu) | 20 FPS (TPUv5) | 非实时 |
| 动作类型 | 键盘+鼠标（连续） | 键盘（离散） | 离散 |
| 交互性 | 实时可玩 | 实时可玩 | 离线评估为主 |
| 时序一致性 | 中等（长期有漂移） | 中等（类似问题） | 较好（序列短） |
| 3D 复杂度 | 高（Minecraft 3D 世界） | 中（DOOM 伪 3D） | 低（2D Atari） |
| 开源 | 是 (500M 版本) | 否 | 是 |
| 训练数据 | 大量 Minecraft 录像 | DOOM 自动玩数据 | Atari 录像 (87h) |

**Oasis vs GameNGen 关键差异**:
- Oasis 用 DiT 架构取代 U-Net，更适合 scaling 且与现代视频生成趋势一致
- Oasis 处理的是真正的开放 3D 世界（Minecraft），复杂度远高于 DOOM 的走廊式关卡
- Oasis 需要处理连续鼠标输入（视角旋转），GameNGen 只有离散按键
- GameNGen 使用 RL agent 收集训练数据的策略值得借鉴

---

## 对 MambaWorld 的启发

### 可借鉴的设计

1. **DiT + Autoregressive 的混合范式值得保留**:
   - Oasis 证明了 "帧间自回归 + 帧内 Diffusion/Flow" 是可行且有效的。MambaWorld 可以沿用这个范式，只是将帧间的 Transformer attention 替换为 Mamba-2 SSM。

2. **AdaLN 动作条件化是简洁有效的选择**:
   - 不增加序列长度，计算开销极小，效果已验证。MambaWorld 可以直接采用类似的 AdaLN 方案，或设计 "Cross-SSM injection"（将 action embedding 注入 SSM 的状态转移矩阵）。

3. **Spatial VAE Tokenizer 可以直接复用**:
   - 开源的 tokenizer 权重可以作为 MambaWorld 的起点，无需从头训练 tokenizer，节省大量算力。
   - 或者使用 NVIDIA Cosmos Tokenizer（2048x 压缩率），获得更高效的 latent 表示。

4. **500M 参数量级就能工作**:
   - 证明世界模型不一定需要巨大规模。MambaWorld 可以先在 500M-1B 规模验证核心 idea，再考虑 scaling。

5. **DDIM 少步采样的可行性**:
   - 4-8 步 DDIM 采样已经能产生可接受质量的帧。MambaWorld 用 Rectified Flow 替代 DDPM 后，目标是 1-4 步直接生成，进一步压缩这个数字。

### 需要避免或改进的点

1. **避免 Transformer 的 O(n^2) 复杂度瓶颈**:
   - Oasis 用 causal self-attention 做帧间上下文传递，上下文窗口受限（通常只看前 1-2 帧）。这是 MambaWorld 用 SSM 替代的核心动机——SSM 可以维持 O(n) 复杂度下的长程记忆。

2. **避免 DDPM 的多步采样开销**:
   - Oasis 即使用 DDIM 加速仍需 4-8 步去噪，这是延迟的主要来源。MambaWorld 用 Rectified Flow 的目标就是将此压缩到 1-4 步，甚至单步生成。

3. **长序列一致性问题需要重点攻克**:
   - Oasis 长时间玩后世界一致性下降（漂移问题）。这恰好是 SSM 的潜在优势——Mamba 的循环状态可以维持更长期的环境记忆。这是 MambaWorld 论文中可以重点展示的优势。

4. **数据效率有改进空间**:
   - Oasis 需要大量数据训练。可以探索更高效的训练策略（如 DIAMOND 仅用 87h 数据），或使用预训练视频模型的权重初始化。

5. **评估指标需要更全面**:
   - Oasis 主要展示定性结果和 FPS，定量评估相对薄弱。MambaWorld 论文应设计更完善的评估 protocol：FVD、动作响应精度、长序列一致性指标、推理延迟/吞吐量的完整 benchmark。

---

## 局限性 / 我们可以改进的点

### Oasis 的局限性

1. **世界一致性有限**: 没有显式的 3D 状态表示或物理引擎，生成的世界在长序列后会"忘记"之前的结构。回头看已经探索过的区域时，生成的内容与之前不一致。

2. **Diffusion 多步采样是速度瓶颈**: 即使 4 步 DDIM，每帧仍需多次前向传播。在 Etched 专用芯片上才能达到高 FPS，通用 GPU 上仍然紧张。

3. **单帧条件化窗口太窄**: 由于 Transformer attention 的计算成本，实际只用前 1-2 帧作为条件。无法利用更长的历史信息。

4. **Tokenizer 与 Generator 独立训练**: 两阶段训练可能导致信息瓶颈——tokenizer 丢弃的信息无法在生成阶段恢复。端到端联合训练可能更优但计算成本更高。

5. **只支持固定分辨率**: 模型绑定在特定分辨率上，无法动态调整。

6. **缺乏物理理解**: 纯数据驱动，没有物理先验。水流、重力、碰撞等行为完全从数据中学习，容易出现物理不一致。

7. **仅在 Minecraft 验证**: 泛化到其他游戏或真实环境需要重新训练。

### MambaWorld 可以改进的点

1. **SSM 长程记忆解决一致性问题**:
   - Mamba-2 的循环状态可以压缩并保持数百帧的环境信息
   - 设计 "world state" 显式编码在 SSM hidden state 中

2. **Rectified Flow 单步生成解决速度问题**:
   - 目标: 1-4 步生成，vs Oasis 的 4-8 步
   - 理论加速 2-8x 的帧生成速度

3. **更长的有效上下文窗口**:
   - SSM 的 O(n) 复杂度允许更长的历史窗口
   - 可以看前 16-64 帧的信息，vs Oasis 的 1-2 帧

4. **Cross-SSM Action Injection (新设计)**:
   - 不仅用 AdaLN 注入动作，还可以将动作信息直接调制 SSM 的状态转移矩阵 A 和 B
   - 让动作更深层地影响时序动态建模，而不仅是视觉去噪

5. **Tokenizer 升级**:
   - 使用 Cosmos Tokenizer (2048x 压缩) 替代 SD VAE (64x 压缩)
   - 大幅减少序列长度，进一步降低计算成本

6. **多域泛化**:
   - 不只做 Minecraft，同时在 nuScenes (驾驶) 和 OpenX-Embodiment (机器人) 上验证
   - 证明架构的通用性

7. **完善的 Ablation Study**:
   - SSM vs Transformer (控制参数量)
   - Rectified Flow vs DDPM (控制步数)
   - 上下文长度对一致性的影响
   - Action injection 方式的对比

---

## 关键代码文件 (open-oasis repo)

```
open-oasis/
├── README.md              # 使用说明
├── model/
│   ├── dit.py             # DiT 核心架构 (⭐ 重点阅读)
│   ├── vae.py             # Spatial VAE tokenizer
│   └── action_encoder.py  # 动作编码器
├── sample.py              # 推理/采样脚本
├── train.py               # 训练脚本
├── configs/
│   └── oasis_500m.yaml    # 500M 模型配置
└── data/
    └── minecraft/         # 数据处理
```

> 注: 文件结构为基于公开信息的大致描述，clone 代码后以实际为准。

---

## 总结: MambaWorld vs Oasis 的核心差异化

| 维度 | Oasis | MambaWorld (目标) |
|------|-------|------------------|
| 时序建模 | Causal Self-Attention, O(n^2) | Mamba-2 SSM, O(n) |
| 帧生成 | DDPM + DDIM (4-8步) | Rectified Flow (1-4步) |
| 上下文窗口 | 1-2 帧 | 16-64+ 帧 |
| 动作注入 | AdaLN | Cross-SSM Injection + AdaLN |
| 长序列一致性 | 漂移严重 | SSM 状态压缩维持一致性 |
| 推理加速 (理论) | 1x (baseline) | 10-50x (SSM + fewer steps) |
| Tokenizer | SD VAE (64x) | Cosmos Tokenizer (2048x) 可选 |

---

*阅读笔记日期: 2026-03-12*
*状态: 基于论文和开源代码的详细分析*

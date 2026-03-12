# DIAMOND: Diffusion for World Modeling: Visual Details Matter in Atari

> **论文**: https://arxiv.org/abs/2405.12399
> **作者**: Eloi Alonso, Adam Jelley, Vincent Micheli, Anssi Kanervisto, Amos Storkey, Tim Pearce, Francois Fleuret
> **会议**: NeurIPS 2024 Spotlight
> **代码**: https://github.com/eloialonso/diamond
> **项目页**: https://diamond-wm.github.io/

---

## 一句话总结

DIAMOND 首次证明了**在像素空间直接使用扩散模型（diffusion model）作为世界模型**可以超越基于离散 latent token 的方法（如 IRIS、TWM），在 Atari 100k benchmark 上取得 SOTA 人类归一化均分 1.46，核心洞察是视觉细节（如小球、子弹等微小像素变化）对 RL 智能体的决策至关重要，而 latent tokenization 会丢失这些信息。

---

## 核心贡献

1. **像素空间 Diffusion World Model**: 提出第一个在像素空间（而非 latent 空间）直接使用 diffusion model 作为 world model 并用于 RL 训练的方法，证明 visual details matter。

2. **Atari 100k SOTA**: 在 26 个 Atari 游戏的 100k 交互步 benchmark 上，平均人类归一化分数达到 **1.46**（超过 DreamerV3 的 1.03、IRIS 的 1.046、TWM 的 1.10），在多个游戏上大幅领先。

3. **揭示 visual fidelity 对 RL 的影响**: 通过 ablation 证明 diffusion 产生的高保真帧（保留微小像素级细节）直接提升了下游 RL agent 的性能，这是 autoregressive discrete token 方法做不到的。

4. **高效的 action-conditioned 帧生成**: 设计了轻量的动作条件化注入方式，将离散动作信息嵌入到 U-Net 的去噪过程中，使得模型能生成与特定动作对应的下一帧。

5. **可控的推理-质量权衡**: Diffusion 的去噪步数（denoising steps）提供了天然的推理速度-生成质量旋钮，少至 **2-5 步**即可获得足够好的帧用于 RL 训练。

---

## 方法详解

### 整体架构：Diffusion Model 如何作为 World Model

DIAMOND 的世界模型是一个**条件扩散模型**，建模的是：

```
p(x_{t+1} | x_{t-K+1:t}, a_{t-K+1:t})
```

即给定过去 K 帧观测和 K 个动作，预测下一帧 x_{t+1}。

**核心组件**:
- **Backbone**: 基于 **U-Net** 架构（2D 卷积），不是 DiT。具体来说使用了类 DDPM 的 U-Net 结构，带有 residual blocks 和 attention layers。
- **帧堆叠输入**: 将过去 K 帧（论文中 K=4，即 frame stack=4）沿 channel 维度拼接作为条件输入到 U-Net 中。这相当于把历史帧作为额外通道与噪声帧一起输入。
- **分辨率**: Atari 原始帧下采样到 **64x64** 灰度图像（1 通道），所以条件输入 channels = 4 (history) + 1 (noisy frame) = 5。
- **输出**: 去噪后的下一帧 x_{t+1}（64x64，1 通道）。

**训练目标**: 标准的 DDPM/DDIM 去噪目标 —— 给定噪声帧 x_t^{noisy} 和噪声等级 t_diff，预测噪声 epsilon（或直接预测 x_0，论文使用 **v-prediction** 参数化）。

```
L = E[|| v_theta(x^{noisy}, t_diff, cond) - v_target ||^2]
```

其中 v-prediction: `v = alpha_t * epsilon - sigma_t * x_0`

### 与 DreamerV3 等 latent world model 的本质区别

| 维度 | DIAMOND | DreamerV3 | IRIS / TWM |
|------|---------|-----------|------------|
| **预测空间** | 像素空间 (64x64) | 连续 latent (RSSM) | 离散 latent tokens (VQ-VAE) |
| **生成模型** | Diffusion (iterative) | 单步 latent 转移 | Autoregressive Transformer |
| **视觉保真度** | 高（像素级精确） | 低（重建有模糊） | 中（discrete tokens 丢细节） |
| **关键信息保留** | 小球/子弹等微小物体保留完好 | 可能丢失 | VQ codebook 量化损失 |
| **推理开销** | 多步去噪（2-10步） | 单步前向 | 自回归逐 token 生成 |
| **reward/done 预测** | 用单独小网络 | 集成在 RSSM 中 | 集成在 Transformer 中 |

**本质区别**: DreamerV3 在一个学到的压缩 latent 空间中做单步转移预测，速度快但丢失视觉细节；IRIS/TWM 用 VQ-VAE 把帧离散化为 tokens 后用 Transformer 自回归预测，但量化引入信息损失。DIAMOND 直接在像素空间建模，用 diffusion 的迭代去噪保证高保真度，代价是多步推理。

### 动作条件化的注入方式

动作条件化是 world model 的关键 —— 模型需要知道智能体执行了什么动作才能预测对应的下一帧。

**DIAMOND 的方案**:
1. **动作嵌入**: 离散动作 a_t 通过 **learned embedding layer** 映射为向量 e_a (维度与时间步嵌入一致)。
2. **注入方式**: 动作嵌入与 diffusion timestep embedding **相加**后，通过 AdaGN (Adaptive Group Normalization) 注入到 U-Net 的每个 residual block 中。
   - 具体: `h = AdaGN(h, emb_t + emb_a)` 其中 emb_t 是 diffusion timestep embedding, emb_a 是 action embedding。
3. **历史动作**: K 个历史动作的 embeddings 被聚合（求和或拼接）后统一注入。

这种方式很轻量，不需要 cross-attention，计算开销极小。

### Diffusion 的具体配置

| 超参数 | 值 | 说明 |
|--------|-----|------|
| **Noise schedule** | Cosine schedule | 与 Improved DDPM 一致 |
| **参数化** | v-prediction | 比 epsilon-prediction 在低 SNR 区域更稳定 |
| **训练 timesteps** | 1000 | 标准 DDPM 设置 |
| **推理 sampler** | **DDIM** | 确定性采样，支持少步推理 |
| **推理 denoising steps** | **10 步**（默认），ablation 显示 **3-5 步**仍可接受 | 关键权衡点 |
| **Guidance** | 无 classifier-free guidance | RL 场景不需要 |
| **U-Net channels** | 基础 channels 128 | - |
| **Attention resolutions** | 16x16, 8x8 | 在较低分辨率层使用 self-attention |
| **模型参数量** | ~**70M-100M** | 相对轻量 |
| **优化器** | Adam, lr=**2e-4** | - |
| **Batch size** | **64** | - |
| **EMA** | 使用 EMA 权重用于推理 | decay=0.999 |

### 训练数据与效率

**Atari 100k 设置**:
- 智能体只允许与真实环境交互 **100k 步**（约 400k 帧，因为 frame skip=4），约 **2 小时**的游戏数据。
- World model 在这些数据上训练，然后 RL agent 完全在想象（imagination）中训练。
- 训练步数: world model 训练约 **100k-200k** gradient steps。

**CS:GO 应用**:
- 使用人类玩家录制的 **87 小时**游戏视频。
- 分辨率提升到 **256x256** RGB（3 通道）。
- 这里 DIAMOND 是作为视频预测/世界模拟器展示，不涉及 RL 训练。
- 87h 数据量足够的原因: (1) CS:GO 场景多样性有限（固定地图），(2) Diffusion model 的数据效率较好（相比 autoregressive），(3) 64x64/256x256 分辨率仍然较低。

### 用于 RL 的 Imagination Pipeline

```
┌─────────────────────────────────────────────────────┐
│                 Imagination Pipeline                 │
│                                                     │
│  1. 从 replay buffer 采样初始状态 s_0 (K 帧)         │
│  2. FOR t = 0 to H (imagination horizon):           │
│     a. Actor 根据 s_t 选择动作 a_t                   │
│     b. World Model (Diffusion) 生成 x_{t+1}         │
│        - 输入: 过去 K 帧 + 动作 a_t                  │
│        - 执行 N 步 DDIM 去噪                         │
│     c. Reward Head 预测 r_t                          │
│     d. Done Head 预测 done_t                         │
│     e. 更新帧缓冲: s_{t+1} = concat(s_t[1:], x_{t+1})│
│  3. 用 imagined trajectories 训练 Actor-Critic       │
└─────────────────────────────────────────────────────┘
```

**关键设计**:
- **Imagination horizon H**: 通常 **15 步**（Atari），较短以避免 compounding error。
- **Actor-Critic 算法**: 使用 **REINFORCE + baseline** 或类 PPO 的策略梯度方法。实际实现中使用了与 DreamerV3 类似的 actor-critic 设计（lambda-return + symlog 变换）。
- **Reward 和 Done 预测**: 用**单独的小型 MLP/CNN 网络**从当前帧 stack 预测，不集成在 U-Net 中。
- **梯度传播**: 与 DreamerV3 不同，DIAMOND **不通过 world model 反传梯度**到 actor（因为 diffusion 的多步采样不适合直接反传）。Actor 的梯度来自 REINFORCE estimator。
- **帧数据类型**: imagination 过程中帧保持为浮点像素值（不做离散化），直接输入到 actor-critic 网络。

**训练流程**:
1. **Phase 1 - 数据收集**: 用随机策略或当前策略与真实环境交互 100k 步，收集数据存入 replay buffer。
2. **Phase 2 - World Model 训练**: 用 replay buffer 数据训练 diffusion world model + reward head + done head。
3. **Phase 3 - Agent 训练**: 完全在 imagination 中训练 actor-critic，每次从 replay buffer 采样初始状态开始 rollout。
4. Phase 2 和 3 **交替进行**。

---

## 实验结果

### Atari 100k Benchmark

**人类归一化均分 (Human Normalized Mean Score)**:

| 方法 | 均分 | 中位数 |
|------|------|--------|
| **DIAMOND** | **1.46** | **0.90** |
| TWM | 1.10 | 0.71 |
| IRIS | 1.046 | 0.60 |
| DreamerV3 | 1.03 | 0.69 |
| SimPLe | 0.44 | - |
| EfficientZero | 1.91 | 1.14 |

> 注: EfficientZero 使用 MCTS + learned model，是 model-based 但不是纯 world model imagination 方法。在"纯 imagination 训练"的 world model 方法中，DIAMOND 是 SOTA。

**单游戏亮点**:
- **Boxing**: DIAMOND 大幅领先（~90 vs DreamerV3 ~70），因为需要精确追踪小的拳击动画。
- **Breakout**: DIAMOND 表现突出，需要精确追踪球和砖块（几个像素大小）。
- **Pong**: 所有方法都表现好，但 DIAMOND 更稳定。
- **Asterix**: DIAMOND 明显优势，需要识别细小的敌人和宝物。
- **关键观察**: DIAMOND 在需要精细视觉感知的游戏上优势最大（验证了 "visual details matter" 的论点）。

**DIAMOND 表现不佳的游戏**:
- 一些需要极长 horizon 规划的游戏（如 Montezuma's Revenge），所有 world model 方法都表现不好。
- 部分游戏中 DreamerV3 的 latent 表示反而更高效（不需要像素级精确）。

### CS:GO 表现

- 在 256x256 分辨率上训练，使用 87h 人类游戏录像。
- **定性结果**: 能生成视觉上连贯的第一人称射击游戏画面，包括移动、射击、换弹等动作对应的视觉变化。
- **定量**: 论文主要展示 FVD (Frechet Video Distance) 指标，具体数值用于证明 diffusion 在视频质量上的优势。
- **局限**: 长序列生成（>50 帧）后会出现 drift 和 artifacts。

### 推理速度和计算需求

| 指标 | DIAMOND | DreamerV3 | IRIS |
|------|---------|-----------|------|
| **imagination 速度** | ~**10-20 FPS** (10步DDIM, 单GPU) | ~**100+ FPS** | ~**30-50 FPS** |
| **训练硬件** | 单张 **A100/V100** | 单张 GPU | 单张 GPU |
| **WM 训练时间** | ~**4-8 小时** (Atari) | ~2 小时 | ~4 小时 |
| **总训练时间** | ~**12-24 小时** (Atari 100k) | ~6-10 小时 | ~10-16 小时 |

- **推理瓶颈**: 每生成一帧需要 N 步（默认 10 步）完整 U-Net forward pass。DreamerV3 只需 1 步 latent 转移。
- **Ablation**: 减少到 **3 步** DDIM，imagination FPS 提升约 3x，性能下降约 5-10%。**5 步**是一个较好的折中点。
- **显存**: 单张消费级 GPU (RTX 3090/4090) 可以跑通 Atari 实验。CS:GO (256x256) 需要更大显存。

---

## 对 MambaWorld 的启发

### 可借鉴的设计

1. **像素空间建模的价值已被验证**: DIAMOND 证明了在某些任务上像素空间优于 latent 空间。MambaWorld 可以考虑在 Mamba 架构上做像素空间预测，利用 SSM 的线性复杂度补偿逐像素建模的开销。

2. **动作条件化方案**: DIAMOND 的"动作嵌入 + timestep 嵌入相加 → AdaGN 注入"方案非常轻量且有效。MambaWorld 可以直接借鉴：将动作嵌入加到 Mamba 的时间步特征上，或作为额外输入 token。

3. **Imagination Pipeline 的整体设计**: 从 replay buffer 采样初始状态 → 自回归 rollout → actor-critic 训练的流程是标准的，可以直接复用。关键细节：
   - imagination horizon = 15
   - 单独的 reward/done prediction head
   - 使用 REINFORCE 而非通过 world model 反传梯度

4. **Frame stack = 4 的条件化**: 用过去 4 帧作为条件（沿 channel 拼接）是成熟的方案，为 Mamba 的序列建模提供了对比基线 —— Mamba 理论上可以用更长的历史而不增加太多开销。

5. **v-prediction 参数化**: 如果 MambaWorld 也涉及某种去噪/flow matching，v-prediction 比 epsilon-prediction 更稳定，值得采用。

### 需要避免或改进的点

1. **Diffusion 的多步推理开销是最大痛点**: 每生成一帧需要 3-10 步去噪，imagination 速度仅 10-20 FPS。MambaWorld 用 SSM 做单步前向预测，速度优势可达 **5-10x**。这是 MambaWorld 对 DIAMOND 最核心的竞争优势。

2. **不通过 WM 反传梯度**: DIAMOND 因为 diffusion 采样不可微，只能用 REINFORCE（高方差）。DreamerV3 可以通过 latent WM 反传梯度（低方差）。MambaWorld 如果在 latent 空间工作，应该利用这一优势，使用 straight-through 或 reparameterization 反传梯度。

3. **U-Net 架构的局限**: DIAMOND 用 2D U-Net 处理单帧，时序信息完全靠 frame stacking。MambaWorld 可以用 Mamba 显式建模时序依赖，理论上能捕获更长的时间关系。

4. **Compounding error**: DIAMOND 在长 horizon 生成时质量快速衰退（>50 帧）。Mamba 的线性递推天然适合长序列，可能在这方面更优。

5. **分辨率限制**: DIAMOND 在 Atari (64x64) 上效果好，但 CS:GO (256x256) 已经比较吃力。MambaWorld 如果要做更高分辨率，需要结合 tokenizer（如 Cosmos Tokenizer）先压缩。

### 作为 Baseline 对比时的注意事项

1. **公平对比**: 必须用相同的 Atari 100k 协议 —— 100k 环境交互步（400k 帧, frame skip=4），相同的游戏子集（26 games），human normalized scoring。
2. **DIAMOND 的开源代码可直接复现**: GitHub repo 完整可用，可在相同硬件上对比。
3. **注意 DIAMOND 的均分 1.46 被 Boxing 等少数游戏拉高**: 建议同时报告均分和中位数。
4. **推理速度对比**: 报告 imagination FPS 和每帧实际推理时间，这是 MambaWorld 的核心卖点。
5. **生成质量对比**: 使用 FVD 和 LPIPS 对比帧质量，DIAMOND 在视觉质量上很强，MambaWorld 需要在这方面至少 competitive。

---

## 局限性 / 我们可以改进的点

### 论文本身的局限

1. **推理速度慢**: 10步 DDIM 使 imagination 速度仅 ~10-20 FPS，严重制约 RL 训练效率。作者 ablation 显示 3 步可以加速但有质量损失。
2. **分辨率受限**: 在像素空间直接操作，256x256 已经接近上限。更高分辨率的 3D 游戏或真实世界场景不可行。
3. **Compounding error**: 自回归生成长序列时误差累积，50+ 帧后严重退化。
4. **不支持 gradient-based policy learning**: 只能用 REINFORCE，方差大，样本效率不如 DreamerV3 的 straight-through 方法。
5. **仅验证了 Atari 和 CS:GO**: 没有在 DMC (DeepMind Control Suite) 等连续控制任务上验证。
6. **U-Net 不建模时序**: 时序信息完全靠 frame stacking (K=4)，没有显式的时序架构。

### 我们可以改进的方向

1. **用 Mamba/SSM 替换 U-Net 中的时序建模**:
   - 将 frame stack 从固定 K=4 改为 Mamba 的隐状态递推，支持任意长历史。
   - 消除多步去噪开销，单步前向即可生成帧。

2. **Flow Matching 替代 Diffusion**:
   - Rectified Flow 可以用 **1-4 步** ODE 求解替代 10 步 DDIM。
   - 配合 Mamba 的时序建模，可能实现 "高质量 + 高速度" 的最优组合。

3. **混合空间建模**:
   - 用 tokenizer（如 Cosmos Tokenizer）先压缩到中等维度的 latent，再用 Mamba 做时序预测。
   - 在 latent 空间保留足够细节（比 VQ-VAE 更高保真），同时比像素空间计算更高效。

4. **支持 gradient-based policy optimization**:
   - SSM/Mamba 前向是可微的，可以像 DreamerV3 一样通过 world model 反传梯度到 actor。
   - 这比 DIAMOND 的 REINFORCE 有本质的样本效率优势。

5. **更长的 Imagination Horizon**:
   - DIAMOND 被限制在 ~15 步 horizon。Mamba 的线性复杂度允许更长的 rollout (50-100 步)，可能在需要长期规划的游戏上大幅超越。

6. **多分辨率 / 3D 环境**:
   - 将 DIAMOND 的方法扩展到 DeepMind Control (84x84 连续控制) 和更高分辨率环境。
   - MambaWorld 的计算效率使这更可行。

---

## 关键 Takeaways (Implementation-Ready)

```python
# MambaWorld 设计决策参考

# 1. 动作条件化 (直接可用)
action_emb = nn.Embedding(num_actions, emb_dim)  # 离散动作
# 注入方式: 与时间特征相加后通过 AdaGN/AdaLN
combined_emb = timestep_emb + action_emb(action)

# 2. 帧预测目标
# DIAMOND 用 v-prediction; 如果用 flow matching 则直接预测 velocity field
# loss = MSE(predicted_v, target_v)

# 3. Imagination pipeline 关键参数
IMAGINATION_HORIZON = 15       # 可以尝试 30-50 (Mamba优势)
FRAME_STACK = 4                # 或用 Mamba 隐状态替代
REPLAY_BUFFER_SIZE = 100_000   # 400k frames (Atari 100k)
REWARD_HEAD = "separate_mlp"   # 不集成到主网络
DONE_HEAD = "separate_mlp"

# 4. RL 算法
# DIAMOND: REINFORCE (因为diffusion不可微)
# MambaWorld: 可以用 DreamerV3-style straight-through gradient
# 这是关键优势,降低方差,提升样本效率

# 5. 对比实验必须报告
# - Human Normalized Mean Score (26 games)
# - Human Normalized Median Score
# - Imagination FPS (frames per second)
# - 总训练时间 (wall clock)
# - 每帧推理时间 (ms)
```

---

*阅读日期: 2026-03-12*
*状态: 已读完，可作为 MambaWorld baseline*

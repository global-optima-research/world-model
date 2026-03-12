# Long-Context State-Space Video World Models

> Po, Nitzan, Zhang, Chen, Dao, Shechtman, Wetzstein, Huang (Stanford / Princeton / Adobe Research)
> ICCV 2025
> arXiv: https://arxiv.org/abs/2505.20171
> 项目页: https://ryanpo.com/ssm_wm/

---

## 一句话总结

本文提出了一种基于 SSM（状态空间模型）+ 稠密局部注意力的混合架构视频世界模型，通过 block-wise SSM scanning 策略在空间一致性和时序记忆之间取得平衡，实现了线性训练复杂度和常数推理开销的长上下文视频世界建模，在 Memory Maze 和 Minecraft 上显著超越次二次复杂度基线，接近全上下文因果 Transformer 的性能。

---

## 核心贡献

1. **Block-wise SSM Scanning 方案**: 提出将 token 序列按空间维度分块为 (b_h, b_w, T) 的块，每个块独立执行 SSM 扫描。通过控制块大小在不同层实现时序记忆与空间一致性的权衡——小块意味着时序上相邻 token 距离更近（仅 b_h x b_w 而非 H x W），SSM 状态能更好地保持时序信息。

2. **SSM + 局部注意力混合架构**: 在每个 Mamba 扫描层后紧跟一个帧局部注意力（Frame Local Attention）层，对同一帧内所有 token 及前 k 帧窗口内的 token 做稠密注意力，弥补 block-wise 扫描带来的空间一致性损失。

3. **长上下文训练方案 (Long-Context Training)**: 在标准 diffusion forcing 基础上混合一种新策略——保持随机长度的前缀帧完全无噪声（clean prefix），仅对后续帧加噪并计算 loss，迫使模型学习利用远距离时序信息。

4. **线性训练 + 常数推理的计算效率**: 训练时间随上下文长度线性增长；推理时每层只需维护固定长度的 KV-cache（局部注意力用）和每个块的 SSM 状态，实现恒定的每帧推理时间和显存占用，适合交互式无限长度生成。

5. **在长程记忆任务上的系统性验证**: 设计了 Spatial Retrieval（回溯轨迹复现）和 Spatial Reasoning（随机动作探索推理未见区域）两个评估任务，系统地衡量世界模型的长程记忆能力。

---

## 方法详解

### 整体架构设计

模型采用自回归帧生成范式（逐帧生成），架构类似 CogVideoX 的 DiT 风格设计。每一帧通过 VAE 编码为 latent tokens，模型在 latent 空间上操作。核心区别是用 Mamba2 (SSD) 层替换全局因果注意力，并与帧局部注意力交替堆叠。

整体流程：
- 输入视频帧 -> VAE 编码为 latent tokens (每帧 H x W 个 token)
- Token 序列按 spatial-major / time-minor 顺序展平（先排完一帧的所有空间位置，再排下一帧）
- 交替通过 Mamba2 块和 Frame Local Attention 块
- 最终解码生成下一帧

### SSM 如何替换 Causal Attention 做时序建模

**传统方案的问题**: 标准因果 Transformer 对所有历史 token 做全注意力，复杂度为 O(T^2 * H^2 * W^2)，不可扩展到长序列。

**SSM 的优势**: SSM 天然适合因果序列建模——它维护一个固定大小的隐状态，顺序处理 token，训练时可并行展开（通过 Mamba2/SSD 的对偶形式），推理时只需递推更新状态，天然是 O(1) 每步。

**Spatial-major / Time-minor 排列**: 关键设计选择是将 token 序列按"先空间后时间"排列。这确保了：
- 因果性得到保证（一帧的所有 token 排在下一帧之前，不会泄露未来信息）
- SSM 扫描时先处理完当前帧的空间信息，再转到下一帧

**问题**: 如果直接对完整的 (H, W, T) 序列做单次 SSM 扫描，时序上相邻的 token（同一空间位置、相邻帧）在展平序列中相距 H x W 个 token。这导致 SSM 的有限状态很难在如此长的间隔后仍保留有效的时序信息。

### Block-wise Scanning 具体怎么做

核心思想：将空间维度分割为更小的块，缩短时序相邻 token 在展平序列中的距离。

具体步骤：
1. 将每帧的 H x W 空间 token 按空间划分为 (H/b_h) x (W/b_w) 个块，每块大小 b_h x b_w
2. 每个空间块跨越所有 T 帧，形成 (b_h, b_w, T) 的子序列
3. 对每个块独立执行 Mamba2 扫描
4. 块内的时序相邻 token 间距为 b_h x b_w（而非 H x W），极大缩短了时序距离

**层级化块大小**: 不同层使用不同的 (b_h, b_w)：
- 浅层使用较大的块（如整帧 H x W），保留空间一致性
- 深层使用较小的块，强化时序记忆
- 这种渐进式设计让模型同时获得空间连贯性和长程时序依赖

**代价**: 不同块之间的 token 在 SSM 层中无法直接交互（空间一致性受损），这正是需要局部注意力来弥补的原因。

### 与 Dense Local Attention 的混合策略

每个 Mamba2 层后紧跟一个 Frame Local Attention 层：

- **注意力范围**: 每个 token 可以注意到同一帧内的所有 token + 前 k 帧（论文中 k=10）内的所有 token
- **分块加速**: 将帧分组为大小为 5 的 chunk，chunk 内帧之间双向注意力，同时注意力扩展到前一个 chunk。使用 FlexAttention 实现显著加速
- **因果约束**: 注意力是因果的——只能看到当前帧和之前的帧
- **作用**: 弥补 block-wise SSM 扫描导致的空间不连贯，确保相邻帧之间的细节一致性（如纹理、物体位置的连续性）

这种设计使得：
- SSM 负责长程时序记忆（压缩历史为固定大小状态）
- 局部注意力负责短程空间-时序一致性（精确的帧间对齐）

### 动作条件化如何注入

- **连续动作**: 通过一个小型 MLP 处理连续动作值，将输出加到噪声级别嵌入（noise level embedding）上
- **离散动作**: 使用可学习嵌入（learned embedding）
- **注入方式**: 通过 Adaptive Layer Normalization (AdaLN) 将动作+噪声条件注入网络各层，与 DiT 系列模型的条件注入方式一致

### 训练细节

**训练策略 — Diffusion Forcing + Long-Context Training**:
- 基础训练使用 diffusion forcing：对每一帧独立采样噪声级别，模型学习在给定含噪前文的情况下去噪当前帧
- 长上下文训练（关键创新）：以一定概率保持一段随机长度的前缀帧完全干净（零噪声），仅对后续帧加噪并计算 loss。当后续帧噪声较高时，干净的远距离前缀帧成为更有用的上下文，推动模型学习长程依赖
- 当前缀长度为 0 时，退化为标准 diffusion forcing

**数据集**:
- **Memory Maze**: 3D 迷宫环境，场景静态，用于评估长程空间记忆。使用内部 VAE 编码（因帧数多）
- **TECO Minecraft**: Minecraft 游戏录像，用于评估更复杂场景下的世界建模能力

**其他训练细节**:
- 使用 VAE 将帧压缩到 latent 空间（空间和时间维度均有压缩）
- 架构类似 CogVideoX 的 DiT 设计
- 评估限于低分辨率合成视频（作者在局限性中提及）

**注意**: 具体的模型参数量、学习率、batch size、训练步数等超参数在公开资料中未详细披露，需参阅完整论文 PDF。

---

## 实验结果

### 评估任务设计

1. **Spatial Retrieval（空间检索）**: 给模型一段上下文轨迹（如在迷宫中走了 240 帧），然后要求模型按原路返回，预测回溯路径上的帧。评估模型能否记住之前见过的场景细节。

2. **Spatial Reasoning（空间推理）**: 给模型一段上下文后，以随机动作继续生成 560 帧。评估模型能否推理出未直接观察过的区域（如迷宫中未走过的路径）。

### 关键指标和 Baseline 对比

**评估指标**: SSIM、LPIPS、PSNR（主要指标），FVD（补充材料）

**Baselines**:
- **有限上下文因果 Transformer**: 只能看到有限帧的标准注意力模型
- **纯 Mamba2**: 只使用 Mamba2 块，无局部注意力
- **Mamba2 + Frame Local Attention**: 有局部注意力但无 block-wise scanning 和长上下文训练
- **全上下文因果 Transformer**: 能看到所有历史帧的标准注意力模型（二次复杂度上界）
- **DFoT**: 双向 diffusion forcing transformer（Minecraft 上的 baseline）

**核心结果**:
- 本方法在 Memory Maze 的 Retrieval 和 Reasoning 任务上**显著优于所有次二次复杂度 baseline**
- 性能**接近全上下文因果 Transformer**（后者有二次训练和推理复杂度）
- 纯 Mamba2 + Frame Local Attention（没有 block-wise scanning）**无法回忆视觉细节**（如球的位置），验证了 block-wise scanning 的关键作用
- 在 240 帧上下文 + 560 帧生成的长程生成任务中，本方法的 **FVD 最低，甚至超过全上下文因果 Transformer**
- 在 Minecraft 上优于 DFoT 和有限上下文因果 Transformer

### 计算效率对比

| 特性 | 全上下文因果 Transformer | 本方法 (SSM + Local Attn) |
|------|------------------------|--------------------------|
| 训练复杂度 | O(T^2) 二次 | O(T) 线性 |
| 推理每帧时间 | 随序列长度线性增长 | **恒定** |
| 推理显存 | 随序列长度线性增长（KV-cache） | **恒定**（固定 KV-cache + SSM state） |
| 无限长度生成 | 不可行 | 可行 |

- 训练时间随上下文长度**线性扩展**
- 每帧推理时间和显存**恒定**，不随已生成帧数增长
- 适合交互式应用（如游戏），可实现无限长度的持续世界生成

### Ablation 研究

消融实验在 Maze Reasoning 任务上验证了：
- **Block-wise SSM scanning 是关键**: 去掉后（退化为标准 Mamba2 全序列扫描）性能显著下降
- **长上下文训练方案是关键**: 去掉 clean prefix 训练策略后，模型无法有效利用远距离上下文
- **每个组件（架构 + 训练）都不可或缺**: 架构和训练策略缺一不可，二者协同才能实现准确的长程记忆

---

## 对 MambaWorld 的启发

### 可借鉴的设计

1. **Block-wise Scanning 是核心创新**: 直接对完整帧做 SSM 扫描效果差（时序 token 间距太大），必须通过分块缩短时序距离。这是在视频世界模型中成功使用 SSM 的关键。MambaWorld 应采用类似策略。

2. **层级化块大小设计**: 浅层用大块保空间、深层用小块强时序的渐进策略值得借鉴。这比全局统一块大小更灵活。

3. **SSM + 局部注意力混合是必要的**: 纯 SSM 不够——空间一致性需要注意力来保障。混合架构（SSM 管长程、注意力管短程）是目前最佳实践。

4. **Long-Context Training 的 Clean Prefix 策略**: 简单但有效。标准 diffusion forcing 会让模型偷懒只看邻近帧；加入干净前缀后强制模型学习远距离依赖。

5. **Spatial-major / Time-minor 排列**: 确保因果性的同时让 SSM 扫描合理。这是视频 SSM 的标准 token 排列方式。

6. **AdaLN 动作注入**: 通过 MLP 编码动作 + AdaLN 注入是成熟且有效的条件化方案，可直接复用。

### 需要避免或改进的点

1. **避免纯 SSM 方案**: 论文明确证明纯 Mamba2（无 block-wise scanning、无局部注意力）在视觉细节记忆上严重不足。不要尝试用纯 SSM 做视频世界模型。

2. **注意 block-wise scanning 的空间割裂问题**: 不同块间 token 在 SSM 层无法交互，这是一个固有限制。需要足够的局部注意力层来弥补。

3. **局部注意力窗口大小 k 需要调优**: k=10 是论文的选择，但最优值可能因任务和分辨率而异。

---

## 局限性 / 我们可以改进的点

### 论文自述局限性

1. **仅在低分辨率合成视频上验证**: 实验限于 Memory Maze 和 Minecraft 等低分辨率环境，未在高分辨率真实视频上验证。扩展到更高分辨率和更复杂场景是重要的未来方向。

2. **评估环境有限**: 只在两个环境上做了实验，泛化性有待验证。

### 我们可以改进的点

1. **扩展到更高分辨率**: 论文的 block-wise scanning 在高分辨率下每块的 token 数更多，可能需要重新设计块大小策略。可以探索自适应块大小或多尺度分块。

2. **更丰富的动作空间**: 论文的动作条件化相对简单（MLP + AdaLN）。对于连续控制或高维动作空间（如机器人操控），可能需要更强大的动作编码器。

3. **结合显式空间记忆**: 本文的长程记忆完全压缩在 SSM 状态中，是隐式记忆。可以结合显式的空间记忆模块（如 neural map 或 memory bank），在需要精确回忆时检索，而非完全依赖 SSM 状态。

4. **探索双向 SSM 变体**: 论文坚持因果（单向）SSM 以保证自回归生成的因果性。但在某些应用中（如离线视频理解或规划），可以探索双向 SSM 来更充分利用上下文。

5. **SSM 状态压缩策略**: 当前块大小决定了 SSM 状态数量（块越多，总状态越多）。可以研究更高效的状态压缩或状态共享策略，进一步降低显存。

6. **与 StateSpaceDiffuser 的对比和融合**: 同期工作 StateSpaceDiffuser（arXiv 2505.22246）采用独立的 SSM 分支压缩历史 + 扩散模型生成的双分支架构。两种范式各有优劣，值得对比研究或融合。

7. **物理一致性**: 论文主要关注视觉记忆（能否记住场景），但未明确评估物理规律的一致性（如碰撞、重力）。对于世界模型的实际应用（如强化学习中的 model-based planning），物理一致性同样关键。

---

## 参考链接

- [arXiv 论文](https://arxiv.org/abs/2505.20171)
- [项目主页](https://ryanpo.com/ssm_wm/)
- [ICCV 2025 论文 PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Po_Long-Context_State-Space_Video_World_Models_ICCV_2025_paper.pdf)
- [Synced 报道](https://syncedreview.com/2025/05/28/adobe-research-unlocking-long-term-memory-in-video-world-models-with-state-space-models/)
- [Moonlight Review](https://www.themoonlight.io/en/review/long-context-state-space-video-world-models)
- [Xun Huang 博客: Towards Video World Models](https://www.xunhuang.me/blogs/world_model.html)
- [同期工作: StateSpaceDiffuser](https://arxiv.org/abs/2505.22246)

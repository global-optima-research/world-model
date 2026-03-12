# WorldMem: Long-term Consistent World Simulation with Memory

**论文**: [arXiv 2504.12369](https://arxiv.org/abs/2504.12369)
**会议**: NeurIPS 2025 (Poster)
**作者**: Zeqi Xiao 等
**代码**: [github.com/xizaoqu/WorldMem](https://github.com/xizaoqu/WorldMem)
**项目主页**: [xizaoqu.github.io/worldmem](https://xizaoqu.github.io/worldmem/)

---

## 1. 一句话总结

WorldMem 在 Diffusion Forcing 框架的基础上引入**显式外部记忆库 (Memory Bank)**，通过**状态感知的记忆注意力机制 (State-Aware Memory Attention)** 从历史帧中检索最相关的视觉信息，从而在 Minecraft 等开放世界中实现长时间、大视角变化下的 3D 空间一致性生成。

---

## 2. 核心贡献

1. **外部记忆库机制**: 提出 Memory Bank 存储历史生成帧及其状态信息（位姿 + 时间戳），打破了上下文窗口的长度限制，使模型能在任意长时间跨度后回忆并重建之前观察过的场景。

2. **状态感知记忆注意力**: 设计了 State-Aware Memory Attention，将空间位置、视角方向和时间戳作为显式状态线索嵌入到 query-key 注意力机制中，实现精准的跨视角、跨时间检索。

3. **基于置信度的记忆检索策略**: 提出 confidence-based selection + similarity deduplication 的两阶段检索方法，利用 FOV 重叠率和时间接近度筛选最相关且非冗余的记忆单元。

4. **动态世界建模**: 通过将时间戳纳入状态嵌入，模型不仅能建模静态世界，还能捕捉世界随时间的动态演化（如植物生长、雪融化等）。

5. **显著的性能提升**: 在 Minecraft 基准上，PSNR 从 baseline 的 18.04 提升到 25.32（短期）/ 19.32（长期），LPIPS 从 0.4376 降至 0.1429，rFID 从 51.28 降至 15.37。

---

## 3. 方法详解

### 3.1 整体架构

WorldMem 建立在以下基础之上：

- **Conditional Diffusion Transformer (CDiT)**: 以外部动作信号为条件，自回归地生成第一人称视角帧
- **Diffusion Forcing (DF)**: 训练范式，支持自回归扩展生成

在此基础上，WorldMem 增加了两个关键组件：
1. **Memory Bank**: 外部存储，持续积累历史生成内容
2. **Memory Block**: 嵌入 DiT 去噪循环中的记忆注意力模块

**整体流程**:
```
输入动作序列 → CDiT 生成噪声帧 → 每个 DiT Block 中:
  (1) 常规 self-attention
  (2) Memory Block: 当前帧 tokens 作为 query，记忆帧 tokens 作为 key/value
  (3) FFN
→ 去噪完成 → 生成帧存入 Memory Bank → 继续下一段生成
```

### 3.2 记忆机制的具体设计

#### 3.2.1 怎么存 (Memory Unit)

每个记忆单元包含：
- **Memory Frame**: 历史生成帧的 latent representation（经 VAE encoder 编码后的 "clear latent"，即无噪声的干净潜变量）
- **State**: 包括：
  - **位姿 (Pose)**: 相机的空间位置和朝向
  - **时间戳 (Timestamp)**: 该帧生成时的时间步

记忆帧在扩散管线中被视为 "clear frames"（干净信号），与当前正在去噪的 noisy frames 形成对比。

#### 3.2.2 怎么检索 (Memory Retrieval)

由于记忆库会随时间增长，不可能将所有记忆帧都送入注意力，因此需要高效检索。WorldMem 采用**两阶段检索**:

**第一阶段 - 置信度筛选 (Confidence-based Selection)**:

置信度分数计算公式：
```
alpha = o * w_o - d * w_t
```
其中：
- `o`: FOV (Field-of-View) 重叠率 —— 通过 Monte Carlo 方法估算当前帧与记忆帧的视野重叠比例
- `d`: 时间戳差异
- `w_o`, `w_t`: 权重系数

优先选择视野重叠大、时间距离近的记忆帧。

**第二阶段 - 相似度去重 (Similarity Deduplication)**:

使用贪心匹配算法，基于帧对相似度过滤冗余记忆单元，确保选出的记忆帧在信息上互补而非重复。

#### 3.2.3 怎么用 (Memory Attention)

记忆信息通过 **cross-attention** 注入生成过程：

- **Query**: 当前正在去噪的帧的 latent tokens
- **Key / Value**: 检索到的记忆帧的 latent tokens

关键创新 —— **State-Aware Embedding**:
- Query 和 Key 都被注入状态感知嵌入
- 采用 **Plucker embedding** 作为位姿表示：将帧级位姿转换为**逐像素的密集位置嵌入**，编码细粒度的空间信息
- 使用 **相对编码 (Relative Encoding)** 而非绝对编码：Key 帧的状态相对于 Query 帧进行归一化，这有助于模型学习空间和时间关系而非记忆绝对坐标

### 3.3 与 DiT/Diffusion 的结合方式

- Memory Block 作为额外的注意力层**嵌入每个 DiT Block 内部**（在 self-attention 之后、FFN 之前）
- 记忆帧以 clean latent 形式参与（无噪声），当前生成帧以 noisy latent 形式作为 query
- 这意味着记忆提供的是"纯净信号"——模型可以直接依赖真实的历史视觉信息，而不是压缩表示或合成抽象

### 3.4 训练细节

- **硬件**: 4 张 H100 GPU
- **收敛**: 约 500K 训练步
- **数据集**: 基于 MineDojo 收集的 Minecraft 数据集，包含多种地形（平原、草原、沙漠）和多种动作模态（移动、视角控制、事件触发）
- **多阶段训练策略** (渐进式提高难度):
  1. **Stage 1**: 小范围导航
  2. **Stage 2**: 大范围导航
  3. **Stage 3**: 垂直转向（抬头/低头）
- **损失函数**: 标准扩散去噪损失（基于 Diffusion Forcing 框架）
- **训练时记忆采样**: 实验发现不同的采样策略对性能有影响（消融实验验证）

---

## 4. 实验结果

### 4.1 主实验 (Minecraft Benchmark)

生成 100 帧未来帧，基于 600 帧记忆库初始化：

| 方法 | PSNR ↑ | LPIPS ↓ | rFID ↓ |
|------|--------|---------|--------|
| Diffusion Forcing (baseline) | 18.04 | 0.4376 | 51.28 |
| **WorldMem** | **19.32** | **0.1429** | **15.37** |

短期一致性评估:

| 方法 | PSNR ↑ |
|------|--------|
| Diffusion Forcing | ~18 |
| **WorldMem** | **25.32** |

关键发现：
- Baseline DF 在超出上下文窗口后，PSNR 和 LPIPS 迅速恶化，rFID 急剧上升 → 长时一致性严重不足
- WorldMem 在长时间跨度后仍能准确重建之前观察过的场景
- 即使在大幅视角变化下也能保持 3D 空间一致性

### 4.2 消融实验

#### 嵌入设计 (Embedding Design):
- **密集位姿嵌入 (Plucker)** 显著优于稀疏嵌入 → 逐像素的空间编码提供更丰富的几何信息
- **相对编码** 优于绝对编码 → 特别在 LPIPS 和 rFID 上有显著提升，因为便于关系推理和信息检索

#### 时间条件 (Time Condition):
- 加入时间戳条件后 PSNR 和 LPIPS 显著提升 → 时间信息帮助模型忠实地再现世界中随时间变化的事件

#### 记忆检索策略 (Memory Retrieval Strategy):
- **随机采样**: 性能极差，rFID 急剧下降，生成质量严重退化
- **置信度筛选**: 显著提升一致性和生成质量
- **置信度 + 相似度去重**: 所有指标进一步提升 → 验证了两阶段检索策略的有效性

#### 记忆上下文窗口长度:
- 增大记忆上下文窗口可以改善性能（但受计算资源限制）

---

## 5. 对 MambaWorld 的启发

### 5.1 显式记忆 vs SSM 隐状态：各自优劣

根据 WorldMem 的结果以及 [arxiv 2512.06983](https://arxiv.org/abs/2512.06983) 对记忆机制的比较研究：

| 维度 | 显式记忆 (WorldMem 式) | SSM 隐状态 (Mamba 式) |
|------|----------------------|---------------------|
| **长期保真度** | 极强 — 直接存储原始 latent，无信息压缩损失 | 有限 — 历史信息被压缩进固定大小的隐状态，早期信息逐渐衰减 |
| **检索精确性** | 强 — 可通过状态线索精确检索特定历史帧 | 弱 — 隐状态是所有历史的混合体，无法选择性回忆 |
| **计算效率** | 差 — cross-attention 的计算量随记忆库大小线性增长；检索本身也有开销 | 极好 — 推理时 O(1) 更新隐状态，无需存储或检索 |
| **存储效率** | 差 — 需要存储所有历史帧的 latent + 状态信息 | 极好 — 只需维护固定大小的隐状态向量 |
| **可扩展性** | 受记忆库大小限制，需要检索策略 | 天然支持无限长序列（理论上） |
| **可解释性** | 强 — 可以追踪模型"回忆"了哪些历史帧 | 弱 — 隐状态难以解释 |
| **3D 一致性** | 强 — 实验证明在大视角变化下表现优异 | 未知 — 缺乏针对 3D 一致性的系统验证 |

### 5.2 可借鉴的设计

1. **状态感知嵌入 (State-Aware Embedding)**:
   - Plucker embedding 提供逐像素密集空间编码 → MambaWorld 可将类似的空间编码注入 SSM 的输入或 gate 信号
   - 相对编码优于绝对编码 → 对 SSM 也可能有益，可以考虑将动作/位姿以相对量输入

2. **混合架构的可能性**:
   - SSM 处理短期连续动态（高效）+ 显式记忆处理长期场景回忆（精确）
   - 类似于人类的工作记忆（SSM 隐状态）+ 长期记忆（外部记忆库）
   - 可以让 SSM 在正常推进时高效运行，仅在检测到"回到旧地方"时触发记忆检索

3. **基于置信度的检索策略**:
   - FOV 重叠 + 时间接近度的置信度公式简单有效
   - MambaWorld 若引入记忆机制，可直接复用此检索策略

4. **渐进式训练**:
   - 从简单到复杂的多阶段训练策略值得借鉴（小范围 → 大范围 → 垂直视角）
   - 可帮助模型逐步学习空间推理能力

5. **时间戳嵌入**:
   - 将时间信息显式编码帮助模型建模世界的动态变化
   - MambaWorld 的 SSM 虽然天然对时序敏感，但显式的时间戳编码可能有助于建模非均匀时间间隔或时间跳跃

---

## 6. 局限性 / 研究机会

### 论文自身的局限

1. **计算和存储开销**: 记忆库随时间线性增长，cross-attention 的计算成本随记忆帧数量增加。虽然有检索策略缓解，但根本上仍面临可扩展性问题。

2. **仅在 Minecraft 验证**: 实验仅在 Minecraft 环境中进行，未在真实世界视频或其他游戏环境中验证泛化性。

3. **依赖位姿信息**: 需要精确的相机位姿和时间戳作为输入，这在真实世界场景中可能难以获取（需要 SLAM 或 VIO 等辅助系统）。

4. **静态场景偏好**: 虽然加入了时间戳来建模动态变化，但核心记忆机制仍然偏向假设场景大部分是静态的，对高度动态的场景（如大量 NPC 移动）效果未知。

5. **生成质量上限**: 作为基于扩散的方法，单帧生成质量受 VAE 和扩散模型能力限制，不如专门的图像生成模型。

### 研究机会

1. **SSM + 显式记忆的混合架构**: 结合 Mamba 的高效序列建模和 WorldMem 的精确历史回忆，可能是最佳平衡点。关键研究问题：何时/如何触发记忆存储和检索？

2. **无位姿记忆检索**: 探索不依赖精确位姿的记忆检索方案（如基于视觉特征的相似度检索），提升方法的通用性。

3. **记忆压缩与遗忘**: 设计智能的记忆管理策略 —— 哪些记忆该保留、哪些该压缩、哪些该遗忘？可借鉴人类记忆的"巩固"机制。

4. **多模态记忆**: 不仅存储视觉 latent，还存储语义信息、物体状态等结构化知识。

5. **跨场景迁移**: 在一个世界中学到的记忆机制能否迁移到其他环境（如从 Minecraft 到真实世界驾驶场景）？

6. **与 3D 表示的结合**: 将记忆帧与轻量级 3D 表示（如 3D Gaussian Splatting）结合，可能进一步提升空间一致性。

---

## Sources

- [WorldMem arXiv Paper](https://arxiv.org/abs/2504.12369)
- [WorldMem HTML Version](https://arxiv.org/html/2504.12369)
- [WorldMem GitHub](https://github.com/xizaoqu/WorldMem)
- [WorldMem Project Page](https://xizaoqu.github.io/worldmem/)
- [NeurIPS 2025 Poster](https://neurips.cc/virtual/2025/loc/san-diego/poster/117127)
- [OpenReview](https://openreview.net/forum?id=c6CAVKlKmU)
- [CTOL Digital Summary](https://www.ctol.digital/news/worldmem-memory-driven-video-diffusion-persistent-simulation/)
- [On Memory: A comparison of memory mechanisms in world models](https://arxiv.org/abs/2512.06983)

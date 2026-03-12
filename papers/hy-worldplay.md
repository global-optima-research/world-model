# HY-WorldPlay (HY-World 1.5) 论文阅读笔记

> 论文: WorldPlay: Towards Long-Term Geometric Consistency for Real-Time Interactive World Modeling (arXiv: 2512.14614)
> 配套论文: WorldCompass: Reinforcement Learning for Long-Horizon World Models (arXiv: 2602.09022)
> 团队: Tencent Hunyuan
> 开源仓库: https://github.com/Tencent-Hunyuan/HY-WorldPlay
> 模型权重: https://huggingface.co/tencent/HY-WorldPlay

---

## 1. 一句话总结

HY-WorldPlay 是首个同时实现**实时交互（24 FPS, 720p）**、**长期几何一致性**且**完整训练代码开源**的流式视频扩散世界模型，提供从数据处理、预训练、中训练、RL 后训练到蒸馏的全流程 pipeline。

---

## 2. 核心贡献

1. **双重动作表征（Dual Action Representation）**：将连续 3D 相机位姿与离散键盘/鼠标控制命令相结合，解决了离散动作的歧义性和连续位姿的场景尺度不一致导致的训练不稳定问题。

2. **重构上下文记忆（Reconstituted Context Memory）**：每生成一个 chunk 时，从历史 chunk 中动态重构上下文，通过时序重采样（temporal reframing）保留几何关键但时间久远的帧，有效缓解记忆衰减问题。

3. **Context Forcing 蒸馏**：针对记忆感知模型设计的新型蒸馏方法，解决了自回归学生模型与双向教师模型之间的分布不匹配问题——通过对齐师生的 memory context，保留学生模型利用长程信息的能力，在实现实时速度的同时防止误差漂移。

4. **WorldCompass RL 后训练框架**：引入 clip-level rollout 策略 + 互补奖励函数 + negative-aware fine-tuning，长序列（381 帧）复合动作的准确率从 ~20% 提升至 ~55%。

5. **完整开源训练 pipeline**：覆盖数据处理、预训练、中训练（middle-training）、RL 后训练、蒸馏全阶段，是世界模型领域最完整的开源训练框架之一。

---

## 3. 方法详解

### 3.1 整体架构

WorldPlay 基于 Diffusion Transformer (DiT) 架构，使用 3D Causal VAE + Rotary Positional Embedding（扩展到时间维度）+ Flow Matching 训练。

提供两个 pipeline 变体：

| 变体 | 骨干网络 | 参数量 | 特点 |
|------|---------|--------|------|
| **HunyuanVideo Pipeline** | HunyuanVideo 1.5 | **8B (8.3B)** | 动作控制更精确，长期记忆更好（**推荐**） |
| **WAN Pipeline** | WAN | **5B** | 显存需求更低，但动作控制和长期记忆有所妥协 |

每个 DiT block 包含：3D Self-Attention → Cross-Attention（条件输入）→ FFN。

**流式生成机制**：采用 chunk-wise 自回归生成，每个 chunk = 4 个 latent = 16 帧。生成每个 chunk 时，基于历史观测（重构的上下文记忆）进行条件生成。

### 3.2 交互控制机制

**双重动作表征（Dual Action Representation）**：

- **连续动作**：3D 相机位姿（camera pose），提供精确的空间定位信息，有利于记忆检索
- **离散动作**：键盘/鼠标命令（W/A/S/D 等移动指令 + 视角旋转），提供尺度自适应的移动语义

以往方法要么只用离散动作（语义清晰但空间歧义），要么只用连续位姿（空间精确但场景尺度差异导致训练不稳定）。WorldPlay 两者结合，互补优势。

**相机位姿获取**：
- 真实视频：使用 VIPE 模型估计连续相机位姿
- 合成视频：从 Unreal Engine / 3D 渲染管线直接提取
- 离散动作信号从连续相机轨迹推导而来，分为移动命令和视角旋转两类

### 3.3 训练 Pipeline（重点）

训练分为 **四个阶段**：

#### 阶段一：预训练（Pre-training）
- 在大规模视频数据上进行基础视频生成能力的预训练
- 骨干网络初始化自 HunyuanVideo 1.5 或 WAN 的预训练权重
- 学习基本的视频生成质量和时序一致性

#### 阶段二：中训练（Middle-training）
- 引入动作条件控制，将模型从纯视频生成转化为交互式世界模型
- 训练模型理解并响应双重动作表征（连续位姿 + 离散指令）
- 引入 Reconstituted Context Memory 机制

#### 阶段三：RL 后训练（WorldCompass）
这是最具创新性的部分，代码于 2026.3.8 开源：

**Clip-Level Rollout 策略**：
- 不是对整个长序列进行 rollout，而是在单个目标 clip 上生成多个样本进行评估
- 提供细粒度奖励信号，同时大幅提升 rollout 效率
- 特别适合长程自回归视频生成的特性

**互补奖励函数**：
- **交互跟随准确度奖励**：通过 3D foundation model 分析相机轨迹，量化模型对输入动作的遵循程度
- **视觉质量奖励**：使用 HPSv3 奖励模型评估生成画面的视觉保真度
- 两个奖励互为约束，有效抑制 reward hacking

**Negative-Aware Fine-Tuning**：
- 借鉴扩散模型 RL 的最新进展，采用负样本感知的微调范式
- 配合多种效率优化手段

**效果**：
- 长序列（381 帧）复合动作准确率：~20% → ~55%（质变级提升）
- 基础动作准确率：64.28% → 76.56%
- 视觉质量（HPSv3）：在多数条件下提升 >1.8 分

#### 阶段四：蒸馏（Context Forcing）
- **核心问题**：传统蒸馏方法在记忆感知模型上失效，因为存在根本性的分布不匹配——自回归学生模型有 memory context，而双向教师模型没有
- **解决方案**：即使给教师模型添加 memory，不匹配的 memory context 也会导致分布发散。Context Forcing 通过对齐师生的 memory context 解决此问题
- **效果**：从多步去噪蒸馏到 **4 步去噪**，实现实时推理速度

### 3.4 数据处理方案

总数据集约 **320K** 高质量视频样本，来源三条路线：

**真实视频数据（~170K）**：
1. 从公开真实视频源出发
2. 过滤：去除短视频、低质量、带水印/UI、密集人群、异常相机运动的样本
3. 使用 HunyuanVideo 1.5 的 caption 模型为每个视频片段生成结构化文本标注
4. 使用 VIPE 估计连续相机位姿

**3D 重建增强数据（~100K）**：
1. 对精选视频进行 **3D Gaussian Splatting** 重建
2. 设计新的 revisit 轨迹从 3D 场景渲染自定义视频（解决原始视频运动单调问题）
3. 使用 **Difix3D+** 修复浮动伪影
4. 产出额外约 100K 高质量真实视频片段

**合成数据（~50K）**：
1. 收集数百个 Unreal Engine 场景
2. 设计复杂的自定义轨迹进行渲染
3. 直接提取精确的相机位姿和动作信号

### 3.5 推理优化

- **去噪步数**：蒸馏后仅需 **4 步去噪**
- **混合并行**：结合 sequence parallelism 和 attention parallelism，将每个 chunk 的 token 分布到 **8 x H800 GPU** 上
  - 不同于传统的模型复制或时间维度序列并行，这种方式在 DiT backbone 和 VAE decoder 上都实现了序列并行
  - 保证每个 chunk 的计算负载均匀分布
- **最终效果**：720p, 24 FPS, 8 x H800 GPU 实时交互

---

## 4. 实验结果

### 短期生成质量
- 在 PSNR、SSIM、LPIPS 等指标上优于现有 SOTA 方法
- 视觉保真度优秀，控制准确度具有竞争力

### 长期几何一致性
- 这是 WorldPlay 拉开差距的核心场景
- 其他 baseline（如 Oasis、Matrix-Game 2.0）在长程生成中控制准确度因误差累积而快速退化
- WorldPlay 通过 Reconstituted Context Memory 保持了显著更优的稳定性
- **WorldPlay 是唯一同时实现实时交互 + 长期一致性的系统**

### WorldCompass RL 后训练效果
- 长序列复合动作准确率：~20% → ~55%（质变）
- 基础动作准确率：64.28% → 76.56%
- HPSv3 视觉质量评分提升 >1.8
- 在 VBench 和人类评估上均表现优异

### 与竞品对比
| 系统 | 实时 | 长期一致 | 开源 |
|------|------|---------|------|
| **WorldPlay** | 24 FPS | 优秀 | 完整开源 |
| Oasis (Decart/Etched) | 实时 | 差（回访场景不一致） | 部分开源 |
| Matrix-Game 2.0 | 实时 | 差 | 否 |
| GameGen 系列 | 非实时 | 中等 | 部分 |

---

## 5. 对 MambaWorld 的启发

### 5.1 训练 Pipeline 设计可借鉴

1. **四阶段渐进式训练**：预训练 → 中训练 → RL 后训练 → 蒸馏，每阶段有明确目标。MambaWorld 可参考这种渐进式设计：先学视频生成 → 再加交互控制 → 再用 RL 对齐 → 最后蒸馏加速。

2. **WorldCompass 的 Clip-Level Rollout**：针对自回归视频生成的 RL 训练，不对整条序列做 rollout，而是在单个 clip 粒度上采样+评估，显著提升效率。这对 Mamba 架构的 RL 后训练同样适用。

3. **互补奖励设计**：交互跟随 + 视觉质量双奖励互相约束防止 reward hacking，这是一个通用且有效的范式。

4. **数据增强方案**：3D Gaussian Splatting 重建 + 重新渲染 revisit 轨迹的思路非常巧妙，可以从有限的真实视频中扩展出大量带精确位姿标注的训练数据。

5. **Context Forcing 蒸馏**：如果 MambaWorld 也采用记忆感知的自回归架构，Context Forcing 中对齐师生 memory context 的思路直接适用。

### 5.2 架构优劣分析

**WorldPlay (DiT) 的优势**：
- 基于成熟的 HunyuanVideo / WAN 预训练权重，视觉质量起点高
- Attention 机制天然适合 Reconstituted Context Memory（通过 attention 检索历史帧）
- 社区生态好，工具链成熟

**WorldPlay (DiT) 的劣势 / Mamba 的机会**：
- **计算复杂度**：8B 模型需要 8 x H800 才能实时，序列并行工程复杂度高。Mamba 的线性复杂度在长序列/长 memory 场景下有天然优势。
- **Memory 机制受限**：Reconstituted Context Memory 本质是在有限的 context window 内选择性保留帧，受 attention 的 O(n^2) 限制。Mamba 的 selective state space 可以用更高效的方式编码长期记忆。
- **蒸馏依赖**：WorldPlay 必须通过 Context Forcing 蒸馏从多步压缩到 4 步才能实时。Mamba 天然推理效率更高，可能无需激进蒸馏。
- **Chunk 大小固定**：16 帧一个 chunk 的设计较为刚性。Mamba 的流式特性可支持更灵活的生成粒度。

---

## 6. 局限性 / 研究机会

1. **硬件门槛高**：实时推理需要 8 x H800 GPU，部署成本极高。这为轻量化世界模型（如基于 Mamba/SSM 的方案）留下了巨大的研究空间。

2. **场景泛化性**：虽然声称有强泛化能力，但 320K 数据集规模相对有限，对于未见过的场景类型（如极端天气、高动态场景）的泛化能力有待验证。

3. **物理一致性缺失**：WorldPlay 主要关注几何一致性（场景结构保持），但对物理规律（碰撞、重力、流体等）的建模能力有限。这是世界模型的下一个前沿问题。

4. **交互维度有限**：目前只支持相机运动控制（移动 + 旋转），不支持物体交互（抓取、推动等）。更丰富的交互能力是重要的研究方向。

5. **RL 后训练效率**：WorldCompass 虽然有效，但长序列复合动作准确率仅达 55%，说明 RL 对齐世界模型仍有很大提升空间。探索更好的奖励函数设计和 RL 算法是开放问题。

6. **评估体系不完善**：世界模型目前缺乏统一的评估 benchmark，现有指标（FVD/FID/PSNR 等）无法全面衡量交互一致性、物理合理性等关键维度。

7. **单视角限制**：支持第一人称和第三人称视角，但不支持多视角同步生成。多视角一致的世界模型是一个有价值的方向。

8. **开源但复现门槛高**：虽然完整开源了训练代码，但 320K 数据集未完全公开，且训练需要大规模计算资源，社区复现存在一定门槛。

---

## 参考链接

- WorldPlay 论文: https://arxiv.org/abs/2512.14614
- WorldCompass 论文: https://arxiv.org/abs/2602.09022
- HY-World 1.5 技术报告: https://3d-models.hunyuan.tencent.com/world/world1_5/HYWorld_1.5_Tech_Report.pdf
- GitHub: https://github.com/Tencent-Hunyuan/HY-WorldPlay
- HuggingFace: https://huggingface.co/tencent/HY-WorldPlay
- AI Films 博客解析: https://studio.aifilms.ai/blog/hunyuan-world

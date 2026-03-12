# LongLive: Real-time Interactive Long Video Generation

> Shuai Yang, Wei Huang, Ruihang Chu, Yicheng Xiao, Yuyang Zhao, Xianbang Wang, Muyang Li, Enze Xie, Ying-Cong Chen, Yao (Jason) Lu, Song Han, Yukang Chen
> NVIDIA, MIT, HKUST, PKU 等
> ICLR 2026
> 论文: https://arxiv.org/abs/2509.22622
> 代码: https://github.com/NVlabs/LongLive
> 模型: https://huggingface.co/Efficient-Large-Model/LongLive-1.3B

---

## 一句话总结

LongLive 是一个基于帧级自回归 (frame-level AR) 架构的实时交互式长视频生成框架，通过 KV-recache 实现 prompt 无缝切换、streaming long tuning 消除 train-short-test-long 的 gap、以及 short window attention + frame sink 保持长程一致性，在单张 H100 上以 20.7 FPS 生成最长 240 秒的交互式视频，同时在 VBench 短/长视频 benchmark 上均达到 SOTA。

---

## 核心贡献

1. **KV-Recache 机制**: 在 prompt 切换边界，重新计算 KV cache（用已生成帧 + 新 prompt 重建），解决了 AR 模型在交互式场景中 prompt 切换时的"语义惯性"问题，同时保持运动连续性。

2. **Streaming Long Tuning**: 提出 train-long-test-long 的训练策略——每次迭代用上一轮保存的 KV cache 生成下一个 5s clip，只对新 clip 施加 DMD (Distribution Matching Distillation) loss，逐步延长到 60s/240s，消除训练-推理的序列长度 mismatch。

3. **Short Window Attention + Frame Sink**: 将 attention 窗口从标准 5s 压缩到 2.5s（约 9 帧 local window），同时保留最初 3 帧作为 attention sink（类似 StreamingLLM 的思路），仅用 12 帧的有效窗口即可恢复 21 帧全窗口的一致性水平，推理速度提升 28%，峰值显存降低 17%。

4. **实时交互式长视频生成**: 首个支持用户在生成过程中实时切换 prompt、改变叙事方向的长视频生成系统，单卡 H100 达到 20.7 FPS（FP8 量化可达 24.8 FPS），支持最长 240s。

5. **高效微调**: 仅需 32 GPU-days（64 张 H100 训练 12 小时），使用 LoRA rank=256（仅 27% 参数可训，350M / 1.3B），即可将短 clip 模型扩展到分钟级长视频生成。

---

## 方法详解

### 整体架构设计

```
用户 Prompt 1 → [Text Encoder] → text embedding
                                      ↓
初始帧/噪声 → [Frame-level AR DiT] → 逐帧生成 5s clip (80帧@16FPS)
                    ↑ KV cache          ↓
                    ↑                   保存 KV cache
                    ↑                   ↓
用户 Prompt 2 → [KV-Recache] ← 已生成帧 + 新 prompt
                    ↓
              继续逐帧生成下一个 5s clip
                    ↓
              ... 循环至 240s ...
                    ↓
              [Decoder] → 输出视频帧 (832×480, 16FPS)
```

**基座模型**: Wan2.1-T2V-1.3B（阿里万相 2.1 的 text-to-video 1.3B 版本），原生生成 5s clip @ 16FPS @ 832×480 分辨率。

**关键改造**: 将原始 Wan2.1 的双向注意力 (bidirectional attention) 转换为因果注意力 (causal attention)，使模型支持 KV caching，从而实现高效的帧级自回归推理。

### KV-Recache 机制具体怎么工作

核心问题：在交互式生成中，用户在某帧切换 prompt，但旧 prompt 的语义信息已经编码在 KV cache 中，导致"语义惯性"——模型会继续生成旧 prompt 的内容。

**解决方案**:
1. 在 prompt 切换边界，**暂停**生成
2. 取出已生成的视频帧前缀（作为视觉上下文）
3. 将这些帧与**新 prompt** 一起送入 cross-attention 层
4. **重新计算** KV cache，替换掉包含旧 prompt 语义的缓存
5. 以新的 KV cache 继续生成后续帧

**效果（消融实验数据）**:
- Background Consistency: KV-recache = 94.81 vs 保留旧 KV cache = 94.90（几乎无损）
- Subject Consistency: KV-recache = 94.04 vs 保留旧 KV cache = 94.12（几乎无损）
- **CLIP Score（语义一致性）**: KV-recache = **27.87** vs 保留旧 KV cache = 26.53（显著提升，说明新 prompt 被正确执行）

关键 insight: KV-recache 保持了运动/背景的连续性，同时让语义快速对齐到新 prompt。

### Frame-level Autoregressive 的设计

与 Oasis 等 patch-level AR 不同，LongLive 采用**帧级别**的自回归：

- 每一"步"生成一个完整帧的所有 latent token（而非逐 patch 生成）
- 帧内使用 DiT 的 parallel 去噪，帧间因果依赖
- 这种设计使得 KV cache 的粒度是"帧"，而非 patch，cache 管理更高效
- 每帧的 latent token 通过 VAE encoder 压缩，具体 token 数量取决于 Wan2.1 的 tokenizer 设计

### 如何保持长序列一致性

#### 1. Frame-level Attention Sink（帧级注意力锚点）

借鉴 StreamingLLM 中的 attention sink 思想，但做了帧级别的适配：
- 将视频的**前 3 帧**作为 sink token，永久保留在 KV cache 中
- 所有后续帧在所有 attention 层中都能 attend 到这 3 个 sink 帧
- 这 3 帧提供稳定的全局上下文（场景布局、光照、风格等）

#### 2. Short Window Attention

- 将 local attention 窗口从标准 5s（约 21帧）压缩到 2.5s（约 9 帧）
- 有效 attention 范围 = 3 sink + 9 local = 12 帧
- 消融结果：9-local + 3-sink 的一致性分数 = **94.1**，接近 21-frame 全窗口水平，远高于无 sink 的 12-frame 窗口（90.6）

#### 3. NTK-RoPE 位置编码适配

- 使用 NTK-aware RoPE（Neural Tangent Kernel 感知的旋转位置编码）
- 支持将模型外推到超出训练长度的序列
- 5000 步训练内即可收敛到合理性能
- 理论上支持**无限长**视频生成（通过 KV-cache 相对位置编码）

### 动作/交互条件化

LongLive 的交互条件化通过**文本 prompt** 实现（非离散动作 token）：

- 用户可以在生成过程中的任意时刻切换文本 prompt
- 新 prompt 通过 cross-attention 注入模型（Wan2.1 原生的 text conditioning 机制）
- KV-recache 确保 prompt 切换的平滑过渡
- 支持的交互类型：改变场景描述、添加/移除物体、调整风格、改变动作方向

**注意**: 这与 Oasis/DIAMOND 等游戏世界模型的离散动作条件化（键盘/鼠标）不同。LongLive 更偏向"导演式"交互控制，而非"玩家式"实时操控。

### 训练细节

| 项目 | 详情 |
|------|------|
| **基座模型** | Wan2.1-T2V-1.3B |
| **教师模型** | Wan2.1-T2V-14B（用于 DMD 蒸馏） |
| **蒸馏方法** | DMD (Distribution Matching Distillation) |
| **微调方式** | LoRA, rank=256 |
| **可训参数** | 350M / 1.3B（27%） |
| **训练规模** | 64 张 H100, 12 小时 = 32 GPU-days |
| **分辨率** | 832 × 480 |
| **帧率** | 16 FPS |
| **单 clip 长度** | 5 秒（80 帧） |
| **训练最大长度** | 60 秒 → 240 秒（课程学习） |
| **位置编码** | NTK-RoPE |
| **Loss** | DMD loss（最小化学生与教师模型输出分布的 KL 散度） |

**Streaming Long Tuning 具体流程**:
1. 从短 clip（5s）模型开始
2. 每次迭代：用前一轮保存的 KV cache 生成下一个 5s clip
3. 只对新生成的 clip 计算 DMD loss（教师 Wan2.1-14B 提供监督）
4. 已生成帧排除在梯度计算之外，防止显存溢出
5. 逐步延长：5s → 10s → 30s → 60s → 240s
6. 每步都在 prompt 切换边界上训练，模拟实际交互场景

---

## 实验结果

### 短视频（5s）VBench 评测

| 模型 | 参数量 | VBench Quality | FPS | 硬件 |
|------|--------|---------------|-----|------|
| Wan2.1-T2V | 1.3B | 84.26 | 0.78 | H100 |
| SkyReels-V2 | - | 82.67 | 0.49 | H100 |
| Self-Forcing | - | - | ~20 | H100 |
| **LongLive** | **1.3B** | **84.87** | **20.7** | **H100** |

- LongLive 在短视频质量上**超过基座模型 Wan2.1**（84.87 vs 84.26），同时速度快 **26.5 倍**

### 长视频（60s 交互式）VBench-Long 评测

| 模型 | VBench-Long Quality | FPS | 速度对比 |
|------|-------------------|-----|---------|
| SkyReels-V2 | 80.49 | ~0.5 | 1x |
| Self-Forcing | 82.46 | ~20 | ~40x |
| **LongLive** | **84.38** | **20.7** | **>41x vs SkyReels** |

- 60s 场景：6 段连续 10s prompt 的交互式生成
- LongLive 质量显著领先，同时速度最快

### 消融实验：Frame Sink + Window Size

| 配置 | 有效帧数 | 一致性分数 | 推理速度 |
|------|---------|----------|---------|
| 21-frame 全窗口 | 21 | ~94.5 | 基准 |
| 12-frame 无 sink | 12 | 90.6 | +28% |
| 9-local + 3-sink | 12 | **94.1** | **+28%** |

关键发现：frame sink 以零额外计算代价恢复了 3.5 个点的一致性分数。

### 消融实验：KV-Recache

| 方法 | BG Consistency | Subject Consistency | CLIP Score |
|------|---------------|-------------------|------------|
| 保留旧 KV | 94.90 | 94.12 | 26.53 |
| **KV-Recache** | **94.81** | **94.04** | **27.87** |

KV-recache 在几乎不损失视觉一致性的前提下，CLIP score 提升 1.34 分。

### 推理效率

| 配置 | FPS | 峰值显存 |
|------|-----|---------|
| 全窗口 attention | ~16 | 基准 |
| Short window + sink | **20.7** | **-17%** |
| + FP8 量化 | **24.8** | 更低 |
| + INT8 量化 | ~24 | 更低（质量损失极小） |

### 最大生成长度

- 单卡 H100 80GB: **最长 240 秒**（4 分钟）
- 通过 NTK-RoPE 适配：理论上支持无限长度

---

## 对 MambaWorld 的启发

### KV-Recache vs SSM Hidden State：优劣对比

| 维度 | KV-Recache (LongLive) | SSM Hidden State (MambaWorld) |
|------|----------------------|------------------------------|
| **长程记忆** | 有限（window + sink），需要额外机制 | **天然优势**，循环状态持续累积 |
| **Prompt 切换** | 需暂停 + 重新计算 cache（有开销） | **可以直接修改 hidden state**，更灵活 |
| **计算复杂度** | O(n) per frame（因为 window 固定），但 window 内是 O(w²) | **O(1) per frame**，纯循环 |
| **显存使用** | KV cache 随 window 线性增长 | **固定大小** hidden state |
| **质量上界** | 受限于 window 大小，sink 只能部分弥补 | 受限于 state 压缩的信息损失 |
| **可解释性** | KV cache 可视化/分析较容易 | hidden state 是黑盒 |
| **成熟度** | 工程成熟，已有大量优化（FlashAttention 等） | 较新，工程优化空间大 |

### 可借鉴的点

1. **Frame Sink 思想 → SSM 的初始状态锚定**: LongLive 用前 3 帧作为 sink 提供全局上下文。MambaWorld 可以用类似思路，将初始帧编码为 SSM 的"锚定状态"，防止长序列遗忘。

2. **Streaming Long Tuning**: train-long-test-long 的训练策略值得直接借鉴。用保存的 hidden state 替代 KV cache，每次迭代只训练新 clip，逐步延长序列。

3. **DMD 蒸馏框架**: 用大模型（14B）做教师蒸馏小模型（1.3B），这个 pipeline 可以复用。MambaWorld 可以用 Wan2.1-14B 或 CogVideoX 做教师。

4. **LoRA 高效微调**: rank=256 的 LoRA 只训 27% 参数即可达到 SOTA，说明基座模型的知识可以高效迁移到长视频场景。MambaWorld 也应该先 pretrain 短 clip，再 LoRA 微调长视频。

5. **NTK-RoPE 外推**: 如果 MambaWorld 的混合架构中有 attention 层，NTK-RoPE 是个免费的长度外推方案。

6. **量化兼容性**: FP8/INT8 量化几乎无损（20.7→24.8 FPS），说明这类模型对量化友好。MambaWorld 的 SSM 层也应该评估量化效果。

### 需改进 / MambaWorld 的差异化机会

1. **window attention 的局限性是我们的机会**: LongLive 用 9+3=12 帧窗口，但超出窗口的信息就丢了。SSM 的循环状态理论上能无损保留全部历史——这是 MambaWorld 的核心卖点。

2. **KV-recache 的暂停开销**: prompt 切换需要暂停生成重算 cache。SSM 可以直接在 hidden state 上做条件注入，无需暂停——这是速度优势。

3. **文本 prompt vs 离散动作**: LongLive 只支持文本 prompt 交互，不支持实时键鼠操控。MambaWorld 定位游戏世界模型，需要支持更细粒度的离散动作条件化。

---

## 局限性 / 研究机会

### 论文自述局限

1. **质量上界**: 质量提升主要来自 adaptation 和 stabilization，而非绝对质量提升。受限于教师模型（Wan2.1-14B）的能力天花板——DMD 蒸馏只能逼近但不能超越教师。
2. **单一模态交互**: 仅支持文本 prompt，不支持图像条件、动作序列、布局控制等多模态交互。
3. **分辨率受限**: 832×480，距离 1080p/4K 仍有差距。

### 我们看到的研究机会

4. **长程一致性仍有 gap**: 虽然 frame sink 缓解了问题，但 240s 视频的后半段一致性仍然有退化。SSM 的循环状态机制有潜力彻底解决这个问题。

5. **非 real-time 的 prompt 切换**: KV-recache 需要暂停重算，在高频交互场景（如游戏，每帧都有动作输入）不够高效。

6. **无物理引擎**: 生成的视频不保证物理一致性（重力、碰撞等）。结合物理约束是下一步方向。

7. **训练数据局限**: 依赖 Wan2.1-14B 做蒸馏教师，而非真实长视频数据。如果有大规模真实长视频数据+标注，直接监督学习可能更好。

8. **无 3D 理解**: 纯 2D 帧生成，没有隐式 3D 表示或深度感知。

9. **固定帧率**: 16 FPS 固定，不支持可变帧率生成或更高帧率（如 30/60 FPS 游戏场景）。

10. **单 GPU 的 240s 上限**: 虽然理论上通过 NTK-RoPE 可以无限长，但实际受显存限制。多卡并行推理的支持尚未探索。

---

## 关键数字速查

| 指标 | 数值 |
|------|------|
| 参数量 | 1.3B（可训 350M） |
| 训练成本 | 32 GPU-days (64×H100, 12h) |
| 推理 FPS | 20.7 (FP16) / 24.8 (FP8) |
| 最大长度 | 240s |
| 分辨率 | 832×480 @ 16FPS |
| VBench 短 (5s) | 84.87 |
| VBench-Long (60s) | 84.38 |
| vs SkyReels-V2 速度 | >41x |
| Attention 窗口 | 9 local + 3 sink = 12 帧 |
| LoRA rank | 256 |
| 推理显存节省 | -17% (vs 全窗口) |
| 推理速度提升 | +28% (short window + sink) |

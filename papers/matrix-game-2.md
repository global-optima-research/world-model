# Matrix-Game 2.0: An Open-Source, Real-Time, and Streaming Interactive World Model

> Skywork AI, 2025
> 论文: https://arxiv.org/abs/2508.13009
> 项目页: https://matrix-game-v2.github.io/
> 代码: https://github.com/SkyworkAI/Matrix-Game
> 模型: https://huggingface.co/Skywork/Matrix-Game-2.0

---

## 一句话总结

Matrix-Game 2.0 是首个开源的实时流式交互世界模型，通过 3D Causal VAE + Multimodal DiT + Self-Forcing 蒸馏的架构，结合键鼠动作注入模块和 KV Cache 流式推理，在单张 H100 上实现 25 FPS 的分钟级高保真交互视频生成，覆盖 Minecraft、GTA5、Unreal Engine 等多样场景，是 DeepMind Genie 3 的开源替代方案。

---

## 核心贡献

1. **首个开源实时流式交互世界模型**: 打破现有交互世界模型依赖双向注意力和多步推理导致无法实时的瓶颈，实现 25 FPS 流式生成，支持分钟级长序列。

2. **Self-Forcing 蒸馏 + DMD 对齐**: 提出基于因果架构的少步蒸馏方案——先收集 ODE 对进行因果学生模型微调，再通过 Distribution Matching Distillation (DMD) 对齐训练和推理分布，解决自回归蒸馏的分布偏移问题。

3. **帧级动作注入模块**: 设计键盘/鼠标到帧的动作注入模块，嵌入到每个 DiT Block 中，实现帧级精确动作响应（键盘精度 0.94，鼠标精度 0.95）。

4. **大规模可扩展数据生产流水线**: 为 Unreal Engine 和 GTA5 构建了可扩展的数据生产系统，生成约 1200 小时高质量交互视频数据，包含帧级交互标注。

5. **KV Cache 无限长度生成**: 采用滚动 KV Cache 机制，自动管理内存并淘汰最旧 token，支持理论上无限长度的流式视频生成。

---

## 方法详解

### 整体架构

```
玩家输入 (keyboard/mouse)
       ↓
[Action Injection Module] → 动作嵌入注入每个 DiT Block
       ↓
[3D Causal VAE] → 时空压缩的 latent 表示
       ↓
[Multimodal DiT (1.8B)] ← 历史帧 latent (自回归 + KV Cache)
       ↓
[Few-step Denoising (蒸馏后)] → 2-4 步去噪
       ↓
[3D Causal VAE Decoder] → 输出视频帧 @ 25 FPS
```

### 关键创新

#### 1. 3D Causal VAE 压缩
- 采用 3D 因果 VAE 同时压缩空间和时间维度
- 因果性保证：当前帧的编码只依赖过去帧，适合流式生成

#### 2. 动作注入模块
- 去除文本分支，模型仅从视觉内容和对应动作预测下一帧
- 键盘动作和鼠标动作分别编码后注入每个 DiT Block
- 动作模块加入后总参数量为 1.8B

#### 3. 因果少步蒸馏 (Self-Forcing + DMD)
- **阶段一**: 收集 40k 组 ODE 对（teacher 多步推理的轨迹）
- **阶段二**: 用 ODE 对微调因果学生模型 6k 步
- **阶段三**: 通过 DMD-based Self-Forcing 继续训练 4k 步，对齐训练/推理分布
- 因果蒸馏的关键：条件基于过去帧生成当前帧，最小化序列延迟

#### 4. KV Cache 流式推理
- 固定长度的滚动缓存，缓存最近的 latent 和动作嵌入
- 超出容量时自动淘汰最早 token
- 消除冗余计算，支持无限长度生成

### 训练细节

- **基础模型训练**: 120k 步，学习率 2e-5，batch size 256
- **蒸馏训练**: 40k ODE 对 → 6k 步微调 → 4k 步 DMD Self-Forcing
- **训练数据**:
  - Sekai 开源数据集筛选后约 85 小时
  - GTA-driver 数据 574 小时
  - Temple Run 游戏数据 560 小时
  - 合计约 1200 小时

---

## 实验结果

### 动作控制精度（597 帧动作序列评估）
| 指标 | Matrix-Game 2.0 | Oasis |
|------|-----------------|-------|
| 键盘精度 | **0.94** | 0.18 |
| 鼠标精度 | **0.95** | 0.84 |

- 在 597 帧动作序列上，键盘和鼠标精度均超过 0.90

### GameWorld Score Benchmark（四维评估）
- **视觉质量**: 大幅优于 Oasis
- **时序质量**: 大幅优于 Oasis
- **动作可控性**: 大幅优于 Oasis
- **物理规则理解**: 优于 Oasis
- 场景一致性和动作平滑度略低于 Oasis

### 推理性能
- 单张 H100 GPU 达到 25 FPS
- 支持分钟级长序列生成
- 覆盖 32 个 Minecraft 场景 + 16 个野外场景图像

### 与 Genie 3 对比
- Matrix-Game 2.0 完全开源 vs Genie 3 受限预览
- 25 FPS vs Genie 3 约 24 FPS (720p)
- 场景稳定性尚未完全达到 Genie 3 水平

---

## 对 MambaWorld 的启发

1. **因果架构是流式世界模型的关键**: Matrix-Game 2.0 证明因果注意力 + KV Cache 是实现实时流式生成的有效范式。Mamba 的天然因果结构和线性复杂度在这个方向上有巨大优势——可以用 Mamba 替代 DiT 中的因果自注意力，获得更高效的长序列建模。

2. **Self-Forcing 蒸馏思路**: 解决自回归生成中训练-推理分布偏移的 Self-Forcing + DMD 方法同样适用于 Mamba 架构的蒸馏加速。MambaWorld 可以探索类似的少步蒸馏策略。

3. **动作注入设计参考**: 将动作嵌入注入每个模型 Block 的设计简洁有效，MambaWorld 可以借鉴类似方案将动作信号注入 Mamba Block。

4. **数据规模参考**: 1200 小时数据就能训练出效果不错的交互世界模型，说明数据质量（帧级动作标注）比数据量更重要。MambaWorld 可以优先构建高质量标注的数据流水线。

5. **KV Cache vs Mamba State**: Matrix-Game 2.0 用滚动 KV Cache 实现无限长度生成；Mamba 天然具有固定大小的隐状态，不需要 KV Cache 且内存恒定，这是 MambaWorld 的核心竞争优势。

6. **1.8B 参数量的参考**: 证明交互世界模型不需要超大规模参数，1.8B 就能达到实时交互。MambaWorld 可以在类似参数规模下探索更高效的架构。

---

## 局限性 / 研究机会

1. **场景稳定性不足**: 相比 Genie 3，Matrix-Game 2.0 在频繁切换的场景中稳定性仍有差距。长期一致性是自回归世界模型的通病，Mamba 的长程记忆能力可能在这方面有优势。

2. **仅支持游戏场景**: 当前主要在 Minecraft、GTA5、Temple Run 等游戏环境验证，向真实世界场景的泛化能力未充分验证。

3. **蒸馏依赖 ODE 对收集**: Self-Forcing 蒸馏需要先用 teacher 模型收集大量 ODE 对，增加了训练流水线复杂度。研究更直接的少步生成方法是机会。

4. **缺少几何理解**: 模型不具备 3D 几何感知能力，无法进行深度估计或 3D 重建——与 Aether 形成互补。将几何感知引入实时交互模型是重要研究方向。

5. **单 GPU 实时但成本高**: 25 FPS 需要 H100 级别 GPU，消费级硬件上的实时性仍需探索。Mamba 架构的计算效率优势可能降低这一门槛。

6. **物理规则理解有限**: 虽然有 GameWorld Benchmark 的物理评估，但物理建模仍是学习而非显式的，复杂物理交互（流体、碰撞）的准确性有待提升。

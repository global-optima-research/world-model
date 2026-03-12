# 跨论文 Gap 分析：研究机会识别

> 基于 11 篇论文的精读，综合识别未被解决的问题和可发论文的切入点
> 分析日期：2026-03-12

---

## 已读论文全景图

### 经典/基础方法
| 论文 | 时序建模 | 帧生成 | 动作注入 | 上下文 | FPS |
|------|---------|--------|---------|--------|-----|
| Oasis (2024) | Causal Attention | DDPM+DDIM 4-8步 | AdaLN | 1-2帧 | 20 (A100) |
| DIAMOND (NeurIPS'24) | Frame Stack (K=4) | DDPM+DDIM 3-10步 | AdaGN | 4帧 | 10-20 |
| Po et al. (ICCV'25) | SSM + Local Attn | Diffusion Forcing | AdaLN | 长序列 | - |

### 当前 SOTA
| 论文 | 时序建模 | 帧生成 | 动作注入 | 上下文 | FPS |
|------|---------|--------|---------|--------|-----|
| LongLive (ICLR'26) | Causal Attn + KV Cache | 4步 DMD蒸馏 | - | 12帧窗口+sink | 20.7 (H100) |
| WorldMem (NeurIPS'25) | DiT + 显式Memory Bank | Diffusion Forcing | - | 记忆检索 | - |
| Matrix-Game 2.0 | 3D Causal DiT | Self-Forcing+DMD 1步 | AdaLN | KV Cache | 25 |
| HY-WorldPlay (2026) | DiT (WAN/HunyuanVideo) | 4步 Context Forcing蒸馏 | 双重(3D位姿+键鼠) | 重构上下文记忆 | 24 (8xH800) |
| Aether (ICCV'25) | CogVideoX-5B | Diffusion | Raymap几何条件 | - | 非实时 |
| LPWM (ICLR'26 Oral) | Causal Spatiotemporal Transformer | Per-particle RGBA decode | 粒子级latent action | 因果序列 | - |

### 基础组件
| 论文 | 核心贡献 |
|------|---------|
| Mamba-2 (ICML'24) | SSD 对偶理论，SSM = 半可分矩阵，chunk-wise 算法 |
| Flow Matching (ICLR'23) | ODE 直线路径，1-4 步生成，CFM 训练简化 |

---

## 🔍 跨论文共性问题（所有 SOTA 的共同瓶颈）

### 问题 1：所有实时世界模型都依赖蒸馏才能达到实时
| 模型 | 原始步数 | 蒸馏后步数 | 蒸馏方法 | 质量损失 |
|------|---------|-----------|---------|---------|
| LongLive | ~25步 | 4步 | DMD (Wan2.1-14B teacher) | 有（CLIP score 下降） |
| Matrix-Game 2.0 | 多步 | 1步 | Self-Forcing + DMD | 有（细节损失） |
| HY-WorldPlay | 多步 | 4步 | Context Forcing | 有 |

**Gap**: 没有人尝试用**天然少步的生成范式**（如 Flow Matching / Rectified Flow）替代 Diffusion+蒸馏。所有人都在"先训 Diffusion，再蒸馏到少步"，而不是"一开始就用少步方法"。

### 问题 2：时序建模全是 Transformer，无一例外
| 模型 | 时序骨干 |
|------|---------|
| LongLive | Causal Transformer + Window Attention |
| WorldMem | DiT |
| Matrix-Game 2.0 | 3D Causal DiT |
| HY-WorldPlay | WAN DiT / HunyuanVideo DiT |
| Aether | CogVideoX (3D Transformer) |
| LPWM | Causal Spatiotemporal Transformer |

**Po et al. 是唯一用 SSM 的，但没有开源代码。**

**Gap**: SSM/Mamba 在世界模型时序建模中**零竞争**（开源领域）。

### 问题 3：长序列一致性的解决方案都是"补丁式"的
| 模型 | 长序列方案 | 本质 | 局限 |
|------|-----------|------|------|
| LongLive | Frame Sink + Window | 保留前3帧+9帧窗口 | 窗口外信息丢失 |
| WorldMem | 显式 Memory Bank | 存储历史帧+检索 | 检索延迟、存储开销大 |
| HY-WorldPlay | 重构上下文记忆 | 从历史chunk中提取 | 需要额外计算 |

**Gap**: 没有人用**原生长序列建模能力**（如 SSM 的循环状态）来解决一致性。所有方案都是在 Transformer 的窗口限制上打补丁。

### 问题 4：计算效率瓶颈普遍存在
| 模型 | 实时所需硬件 |
|------|------------|
| LongLive 1.3B | H100 (1张) |
| HY-WorldPlay | 8x H800 (!!) |
| Matrix-Game 2.0 | RTX 4090 |
| Aether 5B | 非实时 |

**Gap**: 没有人从**架构层面**解决效率问题（除了蒸馏这种后处理方式）。SSM 的 O(n) 复杂度是架构级解决方案。

---

## 🎯 识别到的研究机会（按可行性和新颖度排序）

### 机会 A：SSM + Flow Matching 世界模型（原 MambaWorld 方向）
**读完 SOTA 后的新认知**：
- ✅ 更加确认了方向的价值——所有 SOTA 都用 Transformer + Diffusion + 蒸馏，无一例外
- ✅ Po et al. 没开源，MambaWorld 将是**首个开源 SSM 世界模型**
- ✅ Flow Matching 替代 Diffusion+蒸馏的"两步合一"策略在世界模型中**完全未被探索**
- ⚠️ 新的对标对象：不再是 Oasis/DIAMOND，而是 **LongLive、Matrix-Game 2.0**

**更新后的技术方案**：
```
时序: Mamba-2 SSD (替代所有 SOTA 的 Causal Transformer)
      ↓ 优势: O(n) 复杂度 + 原生长序列记忆（无需 window/sink/memory bank）
生成: Rectified Flow 1-4步 (替代所有 SOTA 的 Diffusion + 蒸馏)
      ↓ 优势: 天然少步，无需蒸馏这个额外阶段
动作: AdaLN + Cross-SSM injection (借鉴 Matrix-Game 的 AdaLN + 新设计)
数据: Minecraft (对标 WorldMem) + 驾驶 (对标 Aether)
```

**核心叙事升级**：
> "当前所有 SOTA 世界模型都遵循 Transformer + Diffusion + 蒸馏的范式。
> 我们提出 MambaWorld，用 SSM 替代 Transformer 获得原生长序列能力，
> 用 Flow Matching 替代 Diffusion+蒸馏实现天然少步生成，
> 从架构层面而非后处理层面解决效率和一致性问题。"

**新颖度**: ★★★★★（两个核心组件的替换都未被探索）
**可行性**: ★★★★☆（需要从零训练，但规模可控 1-2B）
**Impact**: ★★★★★（如果 work，直接挑战整个范式）

---

### 机会 B：SSM + 显式记忆混合架构
**灵感来源**: WorldMem 的记忆检索 + Mamba 的隐状态是**互补**的

| 维度 | SSM 隐状态 | WorldMem 显式记忆 |
|------|-----------|-----------------|
| 存储 | 固定大小，自动压缩 | 无限增长，保留原始信息 |
| 检索 | 隐式（通过状态传播） | 显式（相似度匹配） |
| 精确回忆 | 弱 | 强 |
| 计算开销 | O(1) per step | O(M) per retrieval |

**idea**: SSM 做短期时序动态建模 + 外部记忆做长期精确回忆，组合出一个"既快又准"的长序列世界模型。

**新颖度**: ★★★★☆
**可行性**: ★★★★★（可以在 WorldMem 代码基础上改）
**Impact**: ★★★★☆

---

### 机会 C：物体中心 SSM 世界模型（SSM + LPWM 融合）
**灵感来源**: LPWM 用粒子表示物体，但时序用 Transformer。SSM 天然适合粒子的时序演化。

**idea**: 每个粒子（物体）有独立的 SSM 状态，物体间交互通过 cross-SSM attention。

```
粒子 1: SSM_1(state_1, action) → next_state_1
粒子 2: SSM_2(state_2, action) → next_state_2
物体间: Cross-Attention(state_1, state_2) → interaction
```

**优势**:
- 物体级别的可解释时序建模
- 物体数量可变（每个物体一个 SSM 通道）
- 天然支持物体级别的因果推理

**新颖度**: ★★★★★（全新范式）
**可行性**: ★★★☆☆（需要较多创新设计，风险较高）
**Impact**: ★★★★★（如果 work，是新范式）

---

### 机会 D：几何感知 + 实时交互统一
**灵感来源**: Aether 有几何感知但不能实时；Matrix-Game 能实时但没有几何理解

**Gap**: 没有人同时实现**几何感知 + 实时交互**

**idea**: 在 MambaWorld 中加入 Raymap 几何条件（借鉴 Aether），利用 SSM 效率优势实现实时几何感知世界模型。

**新颖度**: ★★★★☆
**可行性**: ★★★☆☆（Raymap + SSM 的集成需要探索）
**Impact**: ★★★★☆

---

### 机会 E：世界模型的 Scaling Law 研究
**观察**: 所有论文都在特定规模上验证，**没有人系统研究世界模型的 scaling law**

- LLM 有 Chinchilla/Kaplan scaling law
- 视频生成有初步的 scaling 研究
- 世界模型：**完全空白**

**idea**: 系统研究参数量（100M → 500M → 1B → 5B）、数据量、计算量对世界模型质量的影响规律。

**新颖度**: ★★★★☆
**可行性**: ★★★★☆（需要大量实验，但你有 8xH800）
**Impact**: ★★★★★（Scaling law 论文引用极高）

---

## 📋 最终推荐

### 第一优先级：机会 A（SSM + Flow Matching 世界模型）

读完所有 SOTA 后，这个方向不仅没有变弱，反而**更强了**：
1. 所有 SOTA 的共性瓶颈（Transformer + Diffusion + 蒸馏）恰好是我们要替换的
2. Po et al. 没开源，我们将是首个开源 SSM 世界模型
3. Flow Matching 在世界模型中完全未被探索
4. 对标对象从经典方法升级到了 LongLive/Matrix-Game 2.0

### 可组合的增强方向：

机会 A 作为核心，可以叠加 B 或 D：
- **A + B**：SSM + Flow Matching + 显式记忆 → 在长序列一致性上同时超越 LongLive 和 WorldMem
- **A + D**：SSM + Flow Matching + Raymap 几何 → 首个实时几何感知世界模型

### 不推荐作为第一篇的：
- **C（物体中心SSM）**：太新范式，风险高，适合作为第二篇探索
- **E（Scaling Law）**：需要海量实验，适合在 A 做完后用同一套代码做

---

## 关键对比实验设计（预设）

MambaWorld 论文中需要对比的 baseline（按重要性排序）：

| 优先级 | Baseline | 为什么必须对比 |
|--------|----------|--------------|
| 1 | LongLive | 最直接竞品：长序列 + 实时 |
| 2 | Matrix-Game 2.0 | 实时交互 SOTA |
| 3 | WorldMem | 长序列一致性 SOTA |
| 4 | DIAMOND | 经典 RL world model baseline |
| 5 | Oasis | 经典 DiT world model baseline |

对比维度：
- 生成质量（FVD, FID, LPIPS, PSNR）
- 长序列一致性（30s/60s/120s/240s 各时间点的指标退化曲线）
- 推理效率（FPS, 每帧延迟, 显存占用）
- 动作响应精度
- 参数量和训练成本

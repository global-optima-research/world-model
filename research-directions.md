# World Model 论文方向分析与推荐

> 基于 5 个方向的深度调研，结合你的背景（视频生成 + 模型训练经验 + 8x5090 + 8xH800）

---

## 总览：5 个方向竞争程度对比

| 方向 | 竞争程度 | 你的匹配度 | 算力匹配 | 综合推荐 |
|------|---------|-----------|---------|---------|
| Benchmark/评估 | ⚠️ 高（2025已有~10个benchmark） | ★★★☆ | ★★★★ | B |
| 效率/压缩 | ✅ 极低（几乎空白） | ★★★★ | ★★★★ | **S** |
| 语言+世界模型 | ⚠️ 中高（PhyT2V/WISA/DiffPhy已占位） | ★★★★ | ★★★★ | A |
| 垂直领域 | ✅ 低（医疗仅2篇，农业0篇） | ★★★☆ | ★★★★ | A |
| Sim-to-Real/物理一致性 | ⚠️ 中（PIN-WM/Aether/RoboScape已发） | ★★★★ | ★★★☆ | B+ |

---

## 🏆 Tier 1：最强推荐（选一个主攻）

### 方向 1：MambaWorld — SSM+Flow Matching 高效世界模型
**核心 idea**: 用 Mamba-2 替换 Transformer 做时序建模 + Rectified Flow（非 Diffusion）做 1-4 步帧生成，构建 O(n) 复杂度的交互式世界模型。

**为什么排第一**:
- **竞争极低**: Po et al. (ICCV 2025) 只用 SSM 做了时序backbone，生成端仍用标准 Diffusion。没人把 SSM+Flow Matching 结合做完整世界模型
- **完美匹配你的背景**: 视频生成经验 + 架构设计 + 模型训练
- **故事性强**: "从 O(n²) 到 O(n) 的世界模型"，审稿人容易get到价值
- **算力可行**: 8xH800 从零训练一个 1-2B 模型在 Minecraft/driving 数据上

**技术路线**:
```
1. 时序动态: Mamba-2 替换 causal attention（Po et al. 已验证可行）
2. 帧生成: Rectified Flow 替换 DDPM（天然更直, 1-4步即可）
3. 动作注入: Cross-SSM injection（新设计）
4. 训练数据: OpenX-Embodiment / Minecraft / nuScenes
5. 对比: vs Oasis(500M, Diffusion) vs DIAMOND vs Po et al.
```

**预期贡献**:
- 相同质量下推理速度 10-50x 提升（SSM 的 O(n) + Flow 的 1-4 步）
- 长序列一致性提升（SSM 天然优势）
- 首个 SSM-Flow 世界模型

**目标会议**: CVPR 2026 (DDL ~Nov 2025) 或 NeurIPS 2026 (DDL ~May 2026)

---

### 方向 2：CompressWorld — 世界模型压缩系统性研究
**核心 idea**: 第一个系统性研究世界模型压缩的工作——覆盖量化(INT4/8/FP8)、剪枝、蒸馏、稀疏化，建立完整 benchmark。

**为什么推荐**:
- **完全空白**: LLM 压缩已有数百篇论文，世界模型压缩 = 零。这是 2020 年的 LLM 效率研究
- **工程+研究结合**: 你的训练经验是核心优势
- **影响力大**: 这类 "systematic study" 论文引用率高（参考 PTQ4DiT 在 NeurIPS 2024 的影响）
- **风险低**: 实验驱动，不依赖单个 idea 是否 work

**技术路线**:
```
1. 选基线: Oasis(500M) + DIAMOND + Cosmos(fine-tuned)
2. 压缩方法:
   - PTQ: INT8/INT4/FP8, 适配 PTQ4DiT 到视频+动作条件
   - 结构化剪枝: attention heads + FFN layers
   - 知识蒸馏: Cosmos 14B → 1B → 500M → 100M
   - 稀疏化: 结构化稀疏 + 非结构化稀疏
3. 评估维度: FVD, 动作精度, 时序一致性, FPS, 显存占用
4. 发现: 哪些维度对压缩最敏感? 世界模型 vs LLM 压缩有何不同?
```

**目标会议**: NeurIPS 2026 或 ICLR 2027

---

## 🥈 Tier 2：强推荐（可作为第二篇或备选）

### 方向 3：Surgical World Model — 手术世界模型
**核心 idea**: 在 Cosmos 2.5 基础上用 AVID-style adapter 微调，构建动作条件化的手术世界模型。加入立体内窥镜深度信息 + 组织形变物理约束。

**为什么推荐**:
- 全领域仅 2 篇论文（SurgWorld Dec 2025, MeWM Jun 2025）
- 数据已有: SurgVU 840h + SLAM 4K clips + SurgiSR4K
- Adapter 微调在 8xH800 上完全可行
- 医疗 AI 是高影响力领域

**风险**: 需要一定医学领域知识，可能需要找医学合作者

**目标会议**: MICCAI 2026 或 CVPR 2026

---

### 方向 4：PhysLang — 语言参数化物理引擎驱动的视频世界模型
**核心 idea**: 不是用 LLM 做 prompt 增强（PhyT2V 已做），而是让 LLM 输出结构化物理参数（质量、摩擦力、弹性系数），通过 cross-attention 注入视频生成模型的架构层。

**为什么推荐**:
- PhyT2V (CVPR 2025) 和 DiffPhy 只是 prompt 工程层面
- 架构级物理注入是明确的 gap
- 你的视频生成背景可以直接用

**技术路线**:
```
1. LLM → 物理参数提取器（输出连续值: mass, friction, elasticity...）
2. 物理参数 → 可学习嵌入 → cross-attention 注入 DiT
3. 训练: 物理模拟器渲染 paired data（Taichi/Warp）
4. 评估: PhyGenBench + 新设计的物理参数敏感性测试
```

**目标会议**: NeurIPS 2026 或 ICLR 2027

---

### 方向 5：CompPhys-Bench — 组合物理推理基准
**核心 idea**: 测试世界模型能否处理多步因果物理链（Rube Goldberg 风格），而非单一物理现象。

**为什么推荐（谨慎推荐）**:
- Benchmark 竞争已经激烈（2025 出了 ~10 个），但 "组合物理" 角度确实没人做
- 可以和方向 4 配合：benchmark + method 联合发
- 风险: 如果只有 benchmark 没有 method，可能被认为贡献不足

---

## ⚠️ Tier 3：暂不推荐

| 方向 | 原因 |
|------|------|
| 通用 Benchmark（长序列/材料多样性）| 2025 已有 WorldModelBench, WorldSimBench, PhyGenBench, T2VPhysBench, PhyWorldBench... 竞争过于激烈 |
| Sim-to-Real 物理正则化 | PIN-WM (RSS 2025), RoboScape (NeurIPS 2025), Aether (ICCV 2025) 已占位，需要更强的差异化 |
| 农业/水下等极小众领域 | 新颖度高但数据获取难、影响力受限、reviewers 可能不买账 |
| Edge 部署世界模型 | 工程量太大，且当前世界模型质量本身还不够好，"做小"的动机不如 LLM 领域强 |

---

## 📋 推荐执行计划

### 主线：MambaWorld（方向 1）

```
第1-2周: 精读论文
  - Po et al. "Long-Context SSM Video World Models" (ICCV 2025)
  - Rectified Flow / Flow Matching 原始论文
  - Mamba-2 论文
  - Oasis / DIAMOND 论文 + 代码

第3-4周: 搭建基线
  - 复现 Oasis 或 DIAMOND（开源）
  - 在 Minecraft 数据上跑通 baseline

第5-8周: 核心开发
  - 实现 Mamba-2 时序 backbone
  - 实现 Rectified Flow 帧生成器
  - 设计 action conditioning 机制
  - 在 8xH800 上训练

第9-10周: 实验 & ablation
  - vs Oasis, DIAMOND, Po et al.
  - 推理速度 / 生成质量 / 长序列一致性
  - Ablation: SSM vs Transformer, Flow vs Diffusion

第11-12周: 写论文
```

### 副线（并行推进）：CompressWorld（方向 2）

```
第1-4周: 在 8x5090 上搭建评估框架
  - 跑通 Oasis 推理
  - 设计评估 pipeline（FVD, 动作精度, 时序一致性）

第5-8周: 逐一测试压缩方法
  - PTQ: INT8 → INT4 → FP8
  - 剪枝: 结构化 head/layer pruning
  - 蒸馏: 需要等主线的 Mamba 模型也可以作为 teacher

第9-12周: 整理结果 + 写论文
```

---

## 关键参考论文（必读清单）

### 架构 & 效率
1. Po et al., "Long-Context State-Space Video World Models" (ICCV 2025)
2. Oasis: A Universe in a Transformer (Decart, 2024)
3. DIAMOND: Diffusion for World Modeling (2024)
4. Mamba-2: Efficient State Space Models (2024)
5. Rectified Flow / Flow Matching (Lipman et al., ICLR 2023)
6. DOLLAR: Few-Step Video via Distillation (ICCV 2025) — 278.6x speedup
7. NVIDIA Cosmos Tokenizer (2025) — 2048x compression

### 物理 & 语言
8. PhyT2V (CVPR 2025) — LLM-guided physics video
9. WISA (NeurIPS 2025 Spotlight) — Mixture-of-Physical-Experts
10. Dynalang (ICML 2024 Oral) — language-grounded world model
11. PIN-WM (RSS 2025) — physics-informed world model
12. PhyGenBench (ICML 2025) — 27 physical laws benchmark

### 领域
13. SurgWorld (Dec 2025) — surgical world model via Cosmos
14. GAIA-2 (Wayve, 2025) — driving world model
15. Navigation World Models (CVPR 2025, Meta)

### 综述
16. ACM CSUR 2025 World Model Survey (Tsinghua)
17. Survey of Embodied World Models (Tsinghua, 2025)

---

*分析日期: 2026-03-12*

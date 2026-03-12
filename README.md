# World Model Research

> 目标：发一篇世界模型方向的顶会论文（NeurIPS 2026 / ICLR 2027）

## 选定方向

**可微物理模拟器驱动的世界模型后训练** — 用 DiffTaichi/Warp 的流体、软体、布料物理奖励，通过 GRPO 后训练提升视频世界模型的复杂物理一致性。

详见 → [`final-direction.md`](final-direction.md)

## 项目结构

```
.
├── final-direction.md          # ⭐ 最终研究方向（技术方案 + 时间线 + 风险评估）
├── longlive-code-analysis.md   # LongLive 代码架构深度分析
│
├── papers/                     # 论文阅读笔记（13 篇）
│   ├── reading-list.md         # 阅读清单 + 运行计划
│   ├── oasis.md                # Oasis (2024)
│   ├── diamond.md              # DIAMOND (NeurIPS 2024 Spotlight)
│   ├── mamba2-transformers-are-ssms.md  # Mamba-2 (ICML 2024)
│   ├── flow-matching.md        # Flow Matching + Rectified Flow (ICLR 2023)
│   ├── long-context-ssm-world-model.md  # Po et al. (ICCV 2025)
│   ├── longlive.md             # LongLive (ICLR 2026)
│   ├── worldmem.md             # WorldMem (NeurIPS 2025)
│   ├── hy-worldplay.md         # HY-WorldPlay (Tencent 2026)
│   ├── lpwm.md                 # LPWM (ICLR 2026 Oral)
│   ├── matrix-game-2.md        # Matrix-Game 2.0
│   ├── aether.md               # Aether (ICCV 2025 Outstanding)
│   └── newton-pisa-ssd.md      # NewtonRewards + PISA + StateSpaceDiffuser
│
├── analysis/                   # Gap 分析迭代记录
│   ├── gap-analysis-v1.md      # v1: 初始分析（过于乐观）
│   ├── gap-analysis-v2.md      # v2: 修正后发现 SSM 已有 7+ 篇
│   └── gap-analysis-v3.md      # v3: 再修正，RLVR 也有 12+ 篇
│
├── survey/                     # 初始调研（已归档）
│   ├── initial-survey.md       # 世界模型领域初始调研
│   └── research-directions-v1.md  # 早期方向推荐（已被 final-direction.md 替代）
│
└── .gitignore
```

## 硬件

- 日常开发：8× RTX 5090 (32GB)
- 训练：8× H800 (80GB)

## Codebase

基于 [LongLive](https://github.com/NVlabs/LongLive)（NVIDIA/MIT, ICLR 2026, Apache-2.0）：
- Wan 2.1 1.3B → Causal DiT + DMD 蒸馏
- 两阶段训练：Init (21帧) → Long (240帧, LoRA)
- 估计训练成本：~4天 on 8×H800

## 核心差异化

| 维度 | 现有工作 | 我们 |
|------|---------|------|
| 物理范围 | 刚体（重力/碰撞/抛体） | 流体 + 软体 + 布料 |
| 奖励来源 | VLM/光流代理 | 可微物理模拟器精确物理量 |
| 模拟器使用 | 推理时条件化 | 训练时作为奖励优化权重 |

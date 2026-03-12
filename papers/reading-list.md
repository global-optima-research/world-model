# 论文阅读清单

## 本周必读（Week 1-2）

### 1. 🔴 Po et al. — Long-Context SSM Video World Models
- **作者**: Ryan Po, Yotam Nitzan, Richard Zhang, Berlin Chen, Tri Dao, Eli Shechtman, Gordon Wetzstein, Xun Huang
- **会议**: ICCV 2025
- **论文**: https://arxiv.org/abs/2505.20171
- **项目页**: https://ryanpo.com/ssm_wm/
- **代码**: 暂未开源
- **关注点**: SSM 如何替换 causal attention 做时序建模？block-wise scanning 的具体设计？与 dense local attention 的混合策略？
- **状态**: [ ] 未读

### 2. 🔴 Oasis — A Universe in a Transformer
- **作者**: Decart & Etched
- **时间**: 2024
- **论文**: https://oasis-model.github.io/
- **代码**: https://github.com/etched-ai/open-oasis
- **关注点**: 500M 参数的架构设计？Diffusion Transformer 如何做 autoregressive 帧生成？动作条件化怎么注入？
- **状态**: [x] 已读 → 笔记见 `papers/oasis.md`
- **TODO**: clone 代码，在 5090 上跑通推理

### 3. 🔴 Mamba-2 — Transformers are SSMs
- **作者**: Tri Dao, Albert Gu
- **会议**: ICML 2024
- **论文**: https://arxiv.org/abs/2405.21060
- **代码**: https://github.com/state-spaces/mamba
- **关注点**: SSD (Structured State Space Duality) 层的实现细节？与 Transformer attention 的等价性？如何做 cross-sequence conditioning？
- **状态**: [x] 已读 (2026-03-12) → 笔记见 `papers/mamba2-transformers-are-ssms.md`

### 4. 🟡 Flow Matching for Generative Modeling
- **作者**: Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le
- **会议**: ICLR 2023
- **论文**: https://arxiv.org/abs/2210.02747
- **关注点**: Rectified Flow 的核心数学？为什么比 Diffusion 更直（更少步数）？ODE vs SDE formulation？
- **状态**: [x] 已读 (2026-03-12) → 笔记见 `papers/flow-matching.md`（含 Rectified Flow 联合笔记）

### 5. 🟡 DIAMOND — Diffusion for World Modeling
- **作者**: Eloi Alonso, Adam Jelley, Vincent Micheli 等
- **会议**: NeurIPS 2024 Spotlight
- **论文**: https://arxiv.org/abs/2405.12399
- **项目页**: https://diamond-wm.github.io/
- **代码**: https://github.com/eloialonso/diamond
- **关注点**: 仅用 87h 数据训练的效率来自哪里？Diffusion 在 world model 场景的具体 loss 设计？
- **状态**: [x] 已读 (2026-03-12) → 笔记见 `papers/diamond.md`
- **TODO**: clone 代码，在 5090 上跑通推理

---

## 🔥 SOTA 开源模型（必跑 — Week 1-2）

> 目标：跑通推理，观察实际效果和瓶颈，为 MambaWorld 找差异化切入点

### Tier 1：直接竞品（必须跑）

#### LongLive (NVIDIA/MIT) — 长序列实时世界模型
- **会议**: ICLR 2026
- **参数**: 1.3B
- **代码**: https://github.com/NVlabs/LongLive (~926 stars)
- **硬件**: A100+ (20.7 FPS on H100, 24.8 FPS with FP8)
- **亮点**: 帧级自回归 + KV-recache，240s 长视频生成
- **为什么必跑**: 最直接竞品，长序列实时生成正是 MambaWorld 要做的事
- **关注点**: KV-recache 机制 vs SSM 隐状态，哪个更好？长序列一致性如何？
- **状态**: [x] 已读 (2026-03-12) → 笔记见 `papers/longlive.md`
- **TODO**: clone 代码，在 H800 上跑通推理

#### WorldMem — 带记忆的 Minecraft 世界模型
- **会议**: NeurIPS 2025
- **参数**: DiT + Memory blocks
- **代码**: https://github.com/xizaoqu/WorldMem (~300 stars)
- **硬件**: 训练 4xH100，推理 A100+
- **亮点**: 显式记忆机制解决长期一致性
- **为什么必跑**: 与 SSM 隐状态的长记忆能力直接可比
- **关注点**: 记忆检索机制 vs SSM 循环状态，各自优劣？
- **状态**: [ ] 未跑

#### Matrix-Game 2.0 (Skywork) — 实时交互游戏世界模型
- **参数**: 1.8B
- **代码**: https://github.com/SkyworkAI/Matrix-Game
- **硬件**: RTX 4090 可跑（5090 直接能跑）
- **亮点**: 实时 25 FPS 流式交互
- **为什么必跑**: 消费级 GPU 可跑，实时交互性能对标
- **关注点**: 怎么做到 25 FPS 的？架构优化细节？
- **状态**: [ ] 未跑

#### HY-WorldPlay (腾讯混元) — 训练代码开源
- **参数**: 5B (WAN-based) / 8B (HunyuanVideo-based)
- **代码**: https://github.com/Tencent-Hunyuan/HY-WorldPlay
- **硬件**: 5B 消费级可跑，8B 需 A100+
- **亮点**: 训练代码完整开源，24 FPS
- **为什么必跑**: 唯一一个训练代码完整开源的大规模世界模型，可以学习训练 pipeline
- **关注点**: 训练 pipeline 设计、数据处理、loss 配置
- **状态**: [ ] 未跑

### Tier 2：学习架构设计

#### Aether — 几何感知统一世界模型
- **会议**: ICCV 2025 Outstanding Paper (RIWM Workshop)
- **参数**: ~5B (基于 CogVideoX-5b-I2V)
- **代码**: https://github.com/InternRobotics/Aether (~555 stars)
- **硬件**: A100 80GB
- **亮点**: 4D 重建 + 动作预测 + 目标规划统一；零样本 sim-to-real
- **状态**: [ ] 未跑

#### Cosmos Predict 2.5 (NVIDIA)
- **参数**: 2B / 14B
- **代码**: https://github.com/nvidia-cosmos/cosmos-predict2.5
- **硬件**: 2B 单卡 A100；14B 需 H100
- **亮点**: 业界标杆，Text/Image/Video2World 统一
- **状态**: [ ] 未跑

#### LPWM — 粒子世界模型
- **会议**: ICLR 2026 Oral (top 1.18%)
- **参数**: 轻量级
- **代码**: https://github.com/taldatech/lpwm
- **硬件**: 单 GPU 可训练
- **亮点**: 全新范式——基于粒子的物体中心世界模型
- **状态**: [ ] 未跑

#### Astra — 多域世界模型
- **会议**: ICLR 2026
- **参数**: 1.3B (基于 Wan2.1)
- **代码**: https://github.com/EternalEvan/Astra
- **硬件**: 推理 24GB (RTX 3090)，训练 A100
- **亮点**: 驾驶+机器人+无人机，多域验证
- **状态**: [ ] 未跑

### Tier 3：补充参考

#### 其他开源项目
- [ ] **Ctrl-World** (ICLR 2026) — https://github.com/Robert-gyj/Ctrl-World — 机器人操作
- [ ] **RoboScape** (NeurIPS 2025 Spotlight) — https://github.com/tsinghua-fib-lab/RoboScape — 物理感知，34M-544M
- [ ] **AVID** (RLC 2025) — https://github.com/microsoft/causica — 适配器方案
- [ ] **NWM** (CVPR 2025 Best Paper HM) — https://github.com/facebookresearch/nwm — 导航世界模型
- [ ] **LingBot-World** (2026) — https://github.com/Robbyant/lingbot-world — 28B MoE
- [ ] **GenieRedux** — https://github.com/insait-institute/GenieRedux — Genie 开源复现

#### 未开源但需关注
- ❌ **Po et al.** (ICCV 2025) — SSM 世界模型，无代码。**MambaWorld 将是首个开源 SSM 世界模型**
- ❌ **DWS** (ICLR 2025) — 视频模型转世界模拟器，无代码
- ❌ **Genie 2/3** (DeepMind) — 闭源

---

## 按硬件的推荐运行顺序

### 8x RTX 5090 (32GB) 上跑
```
1. Matrix-Game 2.0    → 需 24GB，5090 轻松跑，先建立直觉
2. DIAMOND            → 消费级 GPU，经典 baseline
3. HY-WorldPlay 5B   → 消费级可跑的大模型
4. LPWM               → 单 GPU 可训，新范式
5. Astra              → 24GB 推理
6. RoboScape          → 34M-544M，轻量
```

### 8x H800 (80GB) 上跑
```
1. LongLive           → 最直接竞品，H100 级别
2. WorldMem           → 需 H100 级训练
3. Aether             → 需 A100 80GB
4. Cosmos 2B/14B      → 业界标杆
5. HY-WorldPlay 8B   → 完整训练 pipeline
6. Ctrl-World         → 机器人操作
```

---

## 补充阅读

### 效率相关
- [ ] DOLLAR: Few-Step Video via Distillation (ICCV 2025) — https://arxiv.org/abs/2412.15689
- [ ] NVIDIA Cosmos Tokenizer (2025) — https://github.com/NVIDIA/Cosmos-Tokenizer
- [ ] PTQ4DiT (NeurIPS 2024) — Diffusion Transformer 量化

### 物理 & 语言
- [ ] PhyT2V (CVPR 2025) — https://arxiv.org/abs/2412.00596
- [ ] WISA (NeurIPS 2025 Spotlight) — https://arxiv.org/abs/2503.08153
- [ ] Dynalang (ICML 2024 Oral) — https://arxiv.org/abs/2308.01399

### 综述
- [ ] ACM CSUR 2025 World Model Survey (Tsinghua) — https://arxiv.org/abs/2411.14499
- [ ] Survey of Embodied World Models (Tsinghua, 2025)

---

## 阅读笔记模板

每篇论文读完后在 `papers/` 目录下创建 `{paper-name}.md`，包含：

```markdown
# 论文标题

## 一句话总结
...

## 核心贡献
1.
2.
3.

## 方法
- 架构:
- 训练:
- 数据:

## 实验结果
- 关键指标:
- vs baseline:

## 对 MambaWorld 的启发
- 可借鉴:
- 需避免:

## 局限性 / 我们可以改进的点
-
```

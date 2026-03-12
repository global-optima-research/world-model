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
- **状态**: [ ] 未读
- **TODO**: clone 代码，在 5090 上跑通推理

### 3. 🔴 Mamba-2 — Transformers are SSMs
- **作者**: Tri Dao, Albert Gu
- **会议**: ICML 2024
- **论文**: https://arxiv.org/abs/2405.21060
- **代码**: https://github.com/state-spaces/mamba
- **关注点**: SSD (Structured State Space Duality) 层的实现细节？与 Transformer attention 的等价性？如何做 cross-sequence conditioning？
- **状态**: [ ] 未读

### 4. 🟡 Flow Matching for Generative Modeling
- **作者**: Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le
- **会议**: ICLR 2023
- **论文**: https://arxiv.org/abs/2210.02747
- **关注点**: Rectified Flow 的核心数学？为什么比 Diffusion 更直（更少步数）？ODE vs SDE formulation？
- **状态**: [ ] 未读

### 5. 🟡 DIAMOND — Diffusion for World Modeling
- **作者**: Eloi Alonso, Adam Jelley, Vincent Micheli 等
- **会议**: NeurIPS 2024 Spotlight
- **论文**: https://arxiv.org/abs/2405.12399
- **项目页**: https://diamond-wm.github.io/
- **代码**: https://github.com/eloialonso/diamond
- **关注点**: 仅用 87h 数据训练的效率来自哪里？Diffusion 在 world model 场景的具体 loss 设计？
- **状态**: [ ] 未读
- **TODO**: clone 代码，在 5090 上跑通推理

---

## 下周补充阅读

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

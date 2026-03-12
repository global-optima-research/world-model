# Gap 分析 v2：诚实版

> 基于对竞争态势的深度核查，修正之前的乐观判断
> 日期：2026-03-12

---

## ❌ 之前的判断哪里错了

### "SSM 世界模型零竞争" → 错。至少 7 篇论文已存在

| 论文 | 会议 | 做了什么 |
|------|------|---------|
| R2I (Recall to Imagine) | **ICLR 2024 Oral** (top 1.2%) | S5-based 世界模型，Memory Maze 超人类 |
| Hieros | ICML 2024 | 层级 S5 世界模型，Atari 100k SOTA |
| DRAMA | ICLR 2025 | Mamba-2 MBRL agent，7M 参数 |
| Po et al. | ICCV 2025 | SSM + Local Attention，长序列视频世界模型 |
| StateSpaceDiffuser | NeurIPS 2025 | SSM 压缩历史 + Diffusion 生成，长序列 10x 提升 |
| EDELINE | ICLR 2026 Workshop | Mamba SSM 嵌入到 Diffusion 世界模型 |
| S5WM | ICLR 2025 Workshop | S5 替换 DreamerV3 的 RSSM |

**结论：SSM 用于世界模型已经是成熟方向，不是空白。**

### "Flow Matching 世界模型无人做" → 半对半错

| 论文 | 做了什么 |
|------|---------|
| **Cosmos-Predict2.5** (NVIDIA) | Flow Matching 世界基础模型，2B/14B，**最大的 FM 世界模型** |
| FLIP (ICLR 2025) | Flow-Centric 生成式规划 |
| GoalFlow (CVPR 2025) | Rectified Flow 轨迹生成 |

Cosmos 已经用了 Flow Matching。但在**交互式游戏/RL 世界模型**中确实没人用。

### "SSM + Flow Matching 组合" → 确实没人做，但...

所有 SSM 世界模型都用 Diffusion，所有 FM 世界模型都用 Transformer。
**这个交叉点确实是空白，但"组合两个已知东西"不一定够发顶会。**

---

## 顶会 Oral/Spotlight 论文的共性

| 论文 | 获奖原因 |
|------|---------|
| LPWM (ICLR 2026 Oral) | **新表示范式**（粒子），不是组件替换 |
| Aether (ICCV 2025 Outstanding) | **新统一范式**（重建+预测+规划=同一任务） |
| DIAMOND (NeurIPS 2024 Spotlight) | **新洞察**（视觉细节对 RL 至关重要） |
| R2I (ICLR 2024 Oral) | **新能力**（长期记忆在 RL 中的关键作用） |

**共性：每篇都提出了新的思考方式，而不仅仅是"组件 A 换成组件 B"。**

Reviewer 最常见的拒稿理由：
- "Just combining X and Y without new insights"（组合没有新洞察）
- "The improvement comes from using a better backbone"（只是换了更好的骨干网络）

**MambaWorld 如果只是"用 Mamba 替换 Transformer + 用 FM 替换 Diffusion"，大概率被拒。**

---

## 真正的未解决问题（社区最关心的）

### Tier A — 根本性问题，广泛认可
1. **物理一致性**：WorldModelBench 最好的模型只有 45% mIoU。扩大模型规模对物理误差"几乎没有或负面影响"。这是 #1 未解决问题。
2. **长序列稳定性**：所有方案（window attention, memory bank, frame sink）都是补丁。
3. **评估标准缺失**：FVD/FID 不能衡量"世界理解"。社区缺乏共识。

### Tier B — 重要但部分解决
4. **可控性 vs 质量权衡**：高视觉质量 ≠ 任务成功
5. **组合泛化**：训练过 A 和 B 的场景，能不能处理 A+B？

### Tier C — 新兴且竞争少
6. **用 RL/可验证奖励训练世界模型**：仅 2 篇论文（RLVR-World, GrndCtrl）
7. **世界模型 Scaling Law**：零研究

---

## 🔄 修正后的方向推荐

### 🏆 新的第一推荐：RLVR for Visual World Models

**"用可验证物理奖励接地视觉世界模型"**

**核心 idea**：
正如 RLVR 用可验证的逻辑奖励让 LLM 学会推理，
我们用可验证的物理奖励让视频世界模型学会物理规律。

```
训练流程：
1. 拿一个预训练的视频世界模型（Cosmos / Oasis）
2. 用物理模拟器（Taichi/Warp）生成 ground-truth 物理轨迹
3. 设计可验证的物理奖励：
   - 动量守恒：碰撞前后动量差 → reward
   - 重力一致性：自由落体轨迹 vs 生成轨迹 → reward
   - 碰撞响应：反弹角度/能量损耗是否符合物理 → reward
4. RL post-training（PPO/GRPO）优化世界模型的物理一致性
```

**为什么这是最佳选择**：

| 维度 | 评分 | 理由 |
|------|------|------|
| 新颖度 | ★★★★★ | 仅 2 篇相关论文（RLVR-World 做文本/网页，GrndCtrl 做几何。动态物理一致性=空白） |
| 叙事性 | ★★★★★ | "RLVR grounded LLMs in logic; we ground world models in physics" — reviewer 秒懂 |
| 你的匹配度 | ★★★★★ | 视频生成背景 + 模型训练经验 = 完美匹配 |
| 算力可行性 | ★★★★☆ | 微调而非从零训练，8xH800 绰绰有余 |
| 解决的问题 | ★★★★★ | 直击 #1 未解决问题（物理一致性） |
| 被拒风险 | ★★☆☆☆ | 不是组件替换，是训练范式创新 |

**与现有工作的差异化**：
- **vs RLVR-World**：他们做文本/网页世界模型，我们做视觉/视频世界模型
- **vs GrndCtrl**：他们做几何一致性（深度/相机位姿），我们做**动态物理一致性**（动量/碰撞/形变）——更难但更有价值
- **vs PhyT2V/DiffPhy**：他们在推理时用 LLM 推理物理然后引导生成，我们在**训练时**直接用物理奖励优化模型权重——本质区别

**目标会议**：NeurIPS 2026 (DDL ~May 2026) 或 ICLR 2027 (DDL ~Oct 2026)

---

### 🥈 修正后的 MambaWorld：重新定位为"无蒸馏实时世界模型"

如果仍想做 SSM + Flow Matching，**必须重新定位叙事**：

~~"我们用 SSM 替换 Transformer，用 FM 替换 Diffusion"~~（会被拒）

✅ **"所有实时世界模型都需要蒸馏。我们提出第一个无需蒸馏即可实时的世界模型架构。"**

关键实验：MambaWorld（无蒸馏） vs LongLive/Matrix-Game（蒸馏后）
- 如果质量持平且不需要蒸馏 → 贡献成立
- 如果质量更差 → 论文死掉

| 维度 | 评分 | 理由 |
|------|------|------|
| 新颖度 | ★★★☆☆ | SSM 和 FM 各自都有人做，组合有新颖度但不够大 |
| 叙事性 | ★★★★☆ | "无蒸馏实时"是好故事，但需要实验支撑 |
| 风险 | ★★★★☆ | FM 质量可能不如蒸馏 Diffusion |

---

### 🥉 World Model Scaling Law

"世界模型的 Chinchilla" — 零研究存在。

- 需要大量实验（100M → 5B 多个规模），你有算力
- Chinchilla 论文引用 4000+
- 风险：结果可能不 clean

---

## 对比总结

| 方向 | 新颖度 | 可行性 | Impact | 被拒风险 | 推荐 |
|------|--------|--------|--------|---------|------|
| RLVR Visual World Model | ★★★★★ | ★★★★☆ | ★★★★★ | 低 | **🏆 首选** |
| 无蒸馏实时世界模型 (SSM+FM) | ★★★☆☆ | ★★★☆☆ | ★★★★☆ | 中高 | 🥈 备选 |
| Scaling Law | ★★★★☆ | ★★★☆☆ | ★★★★★ | 中 | 🥉 第二篇 |
| CompressWorld | ★★★☆☆ | ★★★★☆ | ★★★☆☆ | 中 | 第三篇 |

---

## 如果选方向 1（RLVR），下一步行动

```
Week 1-2: 精读
  - RLVR-World (arXiv 2505.13934) — RL 训练世界模型的框架
  - GrndCtrl (arXiv 2512.01952) — 几何自监督奖励对齐
  - RLVR/GRPO for LLMs 相关论文 — 理解 RL post-training 的通用范式
  - PhyGenBench / WorldModelBench — 物理评估基准

Week 3-4: 设计物理奖励函数
  - 选择 3-5 个可验证的物理定律（动量守恒、重力、弹性碰撞等）
  - 用 Taichi/Warp 实现 differentiable 物理模拟器作为 reward model
  - 设计 reward 信号的提取 pipeline

Week 5-8: 实现 + 训练
  - 在 Cosmos 2B 或 Oasis 上实现 RL post-training
  - 用 PPO/GRPO 优化物理一致性
  - 在 PhyGenBench / 自建物理测试集上评估

Week 9-12: 实验 + 写论文
```

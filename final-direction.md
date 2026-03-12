# 最终研究方向选定

> 日期：2026-03-12
> 基于 3 轮 gap 分析 + 3 轮方向验证后的最终结论

---

## 选定方向：可微物理模拟器驱动的世界模型后训练

**一句话叙事**：
> "现有 RLVR 方法只教会视频模型球会落下、会弹跳；我们用可微物理模拟器教它理解流体如何流动、布料如何飘动、软体如何形变。"

---

## 为什么选这个方向

### 排除过程

| 方向 | 竞争状况 | 结论 |
|------|---------|------|
| SSM 世界模型 | 🔴 7+ 篇（R2I, Hieros, DRAMA, Po et al., StateSpaceDiffuser, EDELINE, S5WM） | 排除 |
| RLVR 刚体物理世界模型 | 🔴 12+ 篇（NewtonRewards, PhysRVG, PhyGDPO, PhysCorr, PISA...） | 排除 |
| SSM + Flow Matching 组合 | 🟡 组合未被做，但 reviewer 易拒（"just combining X and Y"） | 排除 |
| World Model Scaling Law | 🟡 已有 Pearce et al. (ICML 2025) 做了 agent 层面的 scaling law，视频世界模型层面仍空白但有负面结果（scaling 对物理无帮助） | 备选 |
| **超越刚体的物理 RLVR** | 🟢 **零论文** | **✅ 选定** |
| 无蒸馏实时世界模型（Shortcut Models） | 🟢 零论文，但技术风险高 | 备选 |

### 验证结果（三轮独立深度验证，2026-03-12）

#### 第一轮验证：可微物理模拟器作为视频模型训练奖励（搜索 22 篇论文）

**结论：零论文。** 没有任何工作将 DiffTaichi/Warp/Brax/PhiFlow 的可微梯度反传到视频扩散模型。

最接近的工作：
| 论文 | 做了什么 | 为什么不算 |
|------|---------|-----------|
| gradSim (ICLR 2021) | 可微模拟器+渲染器做系统辨识 | 优化物理参数，不训练生成模型 |
| PhysDreamer (ECCV 2024) | 可微 MPM 优化材质场 | 视频模型冻结，只优化材质参数 |
| DreamPhysics (AAAI 2025) | 可微 MPM + SDS | 视频模型冻结，只优化材质参数 |
| PSIVG (2026.03) | 模拟器引导视频生成 | 推理时用，不更新权重 |
| Diffusion-DRF (2026.01) | 可微奖励反传到扩散模型 | 奖励来自 VLM，不是物理模拟器 |

两条未连接的平行轨道：
- **轨道 A**：可微物理模拟器（gradSim, DiffTaichi, Warp）→ 用于机器人控制/系统辨识
- **轨道 B**：物理感知视频生成 → 用 VLM/光流/手工物理损失
- **缺失的连接**：没有人把轨道 A 的可微模拟器接入轨道 B 的视频模型训练

#### 第二轮验证：流体/软体/布料的 RLVR 后训练（搜索 8 篇 RLVR 论文 + 4 个 Awesome List）

**结论：零论文。所有 RLVR 物理论文都只做刚体。**

| 论文 | 物理类型 | 具体内容 |
|------|---------|---------|
| NewtonRewards (2025.11) | 刚体 | 自由落体/抛体/斜面滑动 |
| PhysRVG (2026.01) | 刚体 | 运动掩码/轨迹偏移/碰撞检测 |
| PISA (ICML 2025) | 刚体 | 物体下落（Kubric 模拟器生成数据） |
| PhysCorr (2025.11) | 刚体 | 几何稳定/碰撞力学 |
| PhyGDPO (Meta, 2025.12) | 刚体 | 通用 VLM 分数（偶现浮力但非目标） |
| Phys-AR (2025.04) | 刚体 | 速度/质量一致性 |
| PhyPrompt (2026.03) | 通用 | VLM 分数（不区分物理域） |
| PhysMaster (2025.10) | 通用 | DPO 人类偏好（无物理特异性） |

**重要发现**：评估 benchmark 已支持流体/软体：
- PhysicsIQ：含流体动力学、光学、磁学、热力学
- VideoPhy2：含 solid-fluid、fluid-fluid 交互
- PhyWorldBench：含"流体与粒子动力学"类别
- → **评估基础设施就绪，但无人针对这些域设计 RL 奖励**

#### 第三轮验证：训练时 vs 推理时使用模拟器

**结论：所有处理复杂物理的工作都是推理时用模拟器，训练时用的全是刚体。**

| 论文 | 训练时用模拟器？ | 推理时用模拟器？ | 物理范围 | 更新视频模型权重？ |
|------|---------------|---------------|---------|-----------------|
| WonderPlay (ICCV 2025) | ❌ | ✅ | 流体/布料/烟雾 | ❌ |
| PhysAnimator (CVPR 2025) | ❌ | ✅ | 弹性/布料 | ❌ |
| PhysMotion (2024.11) | ❌ | ✅ | MPM 连续介质 | ❌ |
| PSIVG (2026.03) | ❌ | ✅ | 通用 | ❌ |
| PhysGen (ECCV 2024) | ❌ | ✅ | 刚体 | ❌ |
| PISA (ICML 2025) | ✅ 数据生成 | ❌ | **刚体** | ✅ SFT+奖励 |
| PhysRVG (2026.01) | ✅ 奖励 | ❌ | **刚体** | ✅ GRPO |
| Force Prompting (NeurIPS 2025) | ✅ 数据生成 | ❌ | **刚体+风** | ✅ 微调 |
| PhysCtrl (NeurIPS 2025) | ✅ 数据生成 | ❌ | **刚体** | ✅ 条件训练 |

**双重空白确认**：
1. 训练时使用模拟器 + 复杂物理（流体/软体/布料）= **零论文**
2. 可微模拟器作为视频模型训练奖励 = **零论文**

---

## 技术方案

### 基础设施

- **基础模型**: LongLive 1.3B（已蒸馏好的 Wan 2.1 因果世界模型）
- **Codebase**: https://github.com/NVlabs/LongLive (Apache-2.0)
- **硬件**: 8×H800 (80GB) 用于训练，8×RTX 5090 (32GB) 用于开发/评估
- **可微模拟器**: DiffTaichi / NVIDIA Warp

### 三类复杂物理奖励（从易到难）

```
Level 1 — 流体动力学:
  模拟器: DiffTaichi SPH (Smoothed Particle Hydrodynamics) / Warp
  场景: 水流、液体倒入、烟雾扩散
  奖励信号: 速度场一致性
    - 视频帧 → 光流/运动估计 → 生成运动场
    - 模拟器 → ground truth 运动场
    - reward = -||v_generated - v_simulated||₂

Level 2 — 软体力学:
  模拟器: DiffTaichi MPM (Material Point Method)
  场景: 弹性物体形变、橡胶球挤压、果冻晃动
  奖励信号: 形变场一致性
    - 视频帧 → 形变估计（光流 + 深度变化）
    - 模拟器 → ground truth 形变场
    - reward = -||deformation_generated - deformation_simulated||₂

Level 3 — 布料模拟:
  模拟器: Warp Cloth / DiffTaichi
  场景: 旗帜飘动、布料下落、衣物褶皱
  奖励信号: 褶皱/悬垂一致性
    - 视频帧 → 表面法线估计
    - 模拟器 → ground truth 表面法线
    - reward = cosine_similarity(normals_generated, normals_simulated)
```

### 奖励提取 Pipeline

```
输入: 文本 prompt（描述物理场景）
  ↓
LongLive Generator → 生成视频（latent → 解码 → 帧序列）
  ↓
运动估计模块:
  ├─ UniMatch / RAFT → 光流（速度场代理）
  ├─ DepthAnything V2 → 深度图（3D 结构代理）
  └─ Metric3D → 表面法线（形变代理）
  ↓
物理模拟模块:
  ├─ 从 prompt 解析物理场景参数（LLM 辅助）
  ├─ DiffTaichi/Warp 运行模拟
  └─ 输出 ground truth 物理量（速度场/形变场/法线）
  ↓
奖励计算:
  reward = weighted_sum(
    velocity_consistency,     # 速度场一致性
    deformation_consistency,  # 形变一致性
    surface_normal_consistency # 表面法线一致性
  )
  ↓
RL 优化: GRPO / DPO 更新 LongLive Generator 权重
```

### RL 后训练方法

优先选择 **GRPO**（Group Relative Policy Optimization），原因：
1. DeepSeek-R1 已验证 GRPO 在可验证奖励场景的有效性
2. NewtonRewards 和 PhysRVG 都成功使用 GRPO
3. 不需要额外的 value model，显存友好
4. 与 LongLive 的 DMD 蒸馏流程正交，可叠加

备选: DPO（如果 GRPO 训练不稳定）

### 评估方案

```
现有 Benchmark:
  - PhyGenBench (物理生成评估)
  - WorldModelBench (世界模型评估)
  - PhysicsIQ (ICCV 2025 Challenge)

自建 Benchmark — ComplexPhysBench:
  - 流体场景: 20 个 prompt（倒水、烟雾、浪花...）
  - 软体场景: 20 个 prompt（果冻、橡胶球、弹性绳...）
  - 布料场景: 20 个 prompt（旗帜、窗帘、衣物...）
  - 每个 prompt 生成 N 个视频，用模拟器打分

对比 Baselines:
  - LongLive (无物理后训练)
  - NewtonRewards (刚体 RLVR)
  - PhysRVG (刚体 RLVR)
  - Cosmos-Predict2.5 (业界标杆)
  - WonderPlay (推理时模拟器)
```

---

## 与所有现有工作的差异化

### 训练时物理方法对比

| 现有工作 | 物理类型 | 方法 | 我们的优势 |
|---------|---------|------|-----------|
| NewtonRewards (2025.11) | 刚体（牛顿力学） | GRPO + 光流奖励 | 我们做流体/软体/布料 |
| PhysRVG (2026.01) | 刚体碰撞 | GRPO + 碰撞奖励 | 我们做连续介质力学 |
| PhyGDPO (Meta, 2025.12) | 刚体 | DPO + VLM 物理奖励 | 我们用可微模拟器而非 VLM |
| PISA (ICML 2025) | 刚体（自由落体） | Kubric 数据 + 奖励后训练 | 我们覆盖更广的物理域 |
| PhysCorr (2025.11) | 刚体 | DPO + PhysicsRM | 我们有精确物理度量而非 VLM 代理 |
| Phys-AR (2025.04) | 刚体 | GRPO + 速度/质量奖励 | 我们做非牛顿物理 |
| PhysMaster (2025.10) | 一般性 | DPO + 人类偏好 | 我们用可微模拟器而非人类偏好 |
| PhyPrompt (2026.03) | 通用 | GRPO + VLM 分数 | 我们有领域特异的物理奖励 |
| Force Prompting (NeurIPS 2025) | 刚体+风 | Blender 数据 + SFT | 我们覆盖流体/软体/布料 |

### 推理时物理方法对比

| 现有工作 | 物理类型 | 方法 | 我们的优势 |
|---------|---------|------|-----------|
| WonderPlay (ICCV 2025) | 流体/布料/烟雾 | 推理时模拟器条件生成 | 我们在**训练时**用模拟器优化权重 |
| PhysAnimator (CVPR 2025) | 弹性/布料 | 推理时模拟引导 | 训练后的模型不需要推理时模拟器 |
| PhysMotion (2024) | MPM 连续介质 | 推理时 MPM 引导 | 同上 |
| PSIVG (2026.03) | 通用 | 推理时 4D 重建+模拟 | 同上 |
| PhysGen (ECCV 2024) | 刚体 | 推理时动力学引导 | 训练时学到物理，推理无额外开销 |

### 可微模拟器方法对比

| 现有工作 | 用途 | 优化目标 | 我们的优势 |
|---------|------|---------|-----------|
| gradSim (ICLR 2021) | 系统辨识 | 物理参数 | 我们优化视频生成模型权重 |
| PhysDreamer (ECCV 2024) | 动态合成 | 材质场参数 | 同上 |
| DreamPhysics (AAAI 2025) | 3D 动态 | 材质场参数 | 同上 |
| PixelBrax (2025.02) | RL 策略 | 策略网络 | 我们优化视频扩散模型 |

### 核心差异总结

**三个维度的创新**：
1. **物理范围**：刚体 → 流体 + 软体 + 布料（从 Tier 1 到 Tier 2 物理）
2. **奖励来源**：VLM/光流代理 → 可微物理模拟器的精确物理量
3. **训练 vs 推理**：不是推理时加模拟器（WonderPlay），而是训练时用模拟器奖励优化模型权重

**连接两条平行轨道**：
- 轨道 A（可微模拟器生态：gradSim, DiffTaichi, Warp）+ 轨道 B（物理视频生成 RLVR）= 我们的工作

---

## 论文标题候选

- *"Beyond Rigid Bodies: Teaching World Models Complex Physics with Differentiable Simulators"*
- *"DiffPhysReward: Grounding Video World Models in Fluid, Soft-Body, and Cloth Physics"*
- *"From Bouncing Balls to Flowing Water: Scaling Physics-Aware Video Generation Beyond Newtonian Mechanics"*

---

## 目标会议

| 会议 | DDL | 适合度 |
|------|-----|-------|
| **NeurIPS 2026** | ~May 2026 | ⭐ 首选（时间紧但可行） |
| ICLR 2027 | ~Oct 2026 | 备选（更充裕） |

---

## 执行时间线

```
Week 1-2: 基础准备
  - 精读 NewtonRewards、PhysRVG、WonderPlay（了解 RLVR 和物理模拟的最佳实践）
  - 安装 DiffTaichi / Warp，跑通基础物理模拟 demo
  - 在 8xH800 上跑通 LongLive 推理

Week 3-4: 设计物理奖励函数
  - 实现流体速度场一致性奖励（Level 1）
  - 实现软体形变一致性奖励（Level 2）
  - 设计 prompt → 物理参数的自动解析 pipeline
  - 搭建奖励提取 pipeline（光流/深度/法线估计）

Week 5-8: 实现 + 训练
  - 在 LongLive 上实现 GRPO 后训练框架
  - 先用流体奖励做 proof-of-concept
  - 逐步加入软体和布料奖励
  - 调参 + 稳定训练

Week 9-10: 评估
  - PhyGenBench 评估
  - 自建 ComplexPhysBench 评估
  - 与 baseline 对比实验
  - 消融实验（不同物理域、不同奖励权重）

Week 11-12: 写论文
  - 撰写论文
  - 制作图表和可视化
  - 准备 supplementary material（视频 demo）
```

---

## 风险和应对

| 风险 | 概率 | 应对 |
|------|------|------|
| 光流/深度估计精度不足以提供可靠奖励 | 中 | 使用多种估计方法投票；简化物理场景降低精度要求 |
| 模拟器参数与真实视频差异大 | 中 | 模拟器输出做归一化；用相对排序(DPO)而非绝对分数 |
| GRPO 训练不稳定 | 中低 | 回退到 DPO；参考 NewtonRewards 的训练策略 |
| 物理改善但视觉质量下降 | 低 | 加视觉质量正则化；多目标优化 |
| 有人抢发类似工作 | 低 | 做 3 种物理域形成壁垒；NeurIPS DDL 前提交 |

---

## 备选方向

如果方向 B 进展不顺，可转向：

### 备选 1: Shortcut Models for World Models（无蒸馏实时）
- Meta 的 Shortcut Models 从未应用到视频世界模型
- 将 Shortcut 条件化（步数条件）引入 LongLive，实现无蒸馏 1-4 步生成
- 风险：可能质量不如蒸馏方案

### 备选 2: World Model Scaling Law（视频世界模型的 Chinchilla）
- 系统研究 100M→5B 参数对世界模型质量的影响
- 需要大量实验，但 8xH800 可支持
- 风险：结果可能不 clean；已有初步相关工作

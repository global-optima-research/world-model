# 执行计划

> 方向：可微物理模拟器驱动的世界模型后训练
> 日期：2026-03-12
> 目标会议：ICLR 2027 (DDL ~Oct 2026)，如进展快可冲 NeurIPS 2026 (DDL ~May 2026)

---

## 技术选型

### 可微物理模拟器：NVIDIA Warp + Newton

| 候选 | 流体 | 软体 | 布料 | 可微 | PyTorch 集成 | 维护状态 |
|------|------|------|------|------|------------|---------|
| **NVIDIA Warp** | SPH+MPM+Euler | FEM+MPM | VBD+FEM | ✅ | 一等公民 | ✅ 活跃（v1.12.0） |
| DiffTaichi | MPM+Euler+SPH | MPM | 弹簧质点 | ✅ | 间接 | ❌ 已停滞 |
| PhiFlow | Euler(+SPH) | ❌ | ❌ | ✅ | 一等公民 | ✅ 活跃 |
| Brax | ❌ | ❌ | ❌ | ✅ | JAX only | ✅ |

**选择 Warp**：唯一同时覆盖三种物理域 + 可微 + PyTorch 原生集成的框架。
由 NVIDIA + Google DeepMind + Disney Research 支持，已捐赠 Linux Foundation。
**PhiFlow** 作为纯流体场景的补充备选。

### RL 后训练框架：DanceGRPO

| 候选 | 支持视频 | 支持 GRPO | 支持 Flow Matching | 开源 | 模型支持 |
|------|---------|----------|-------------------|------|---------|
| **DanceGRPO** | ✅ | ✅ | ✅ | ✅ | HunyuanVideo, SkyReels-I2V |
| VADER | ✅ | ❌(梯度法) | ❌ | ✅ | VideoCrafter, OpenSora |
| Flow-GRPO | ❌(仅图像) | ✅ | ✅ | ✅ | SD3.5 |
| DiffusionNFT | ❌(仅图像) | ❌(NFT) | ✅ | ✅ | NVIDIA 方案 |
| VideoTuna | ✅ | ❌ | ❌ | ✅ | CogVideoX, OpenSora |

**选择 DanceGRPO**：唯一同时支持视频 + GRPO + Flow Matching 的开源框架。
需要适配到 LongLive/Wan2.1 架构（DanceGRPO 已支持 HunyuanVideo，架构类似）。

**备选路径**：如果 DanceGRPO 适配困难，可用 VADER 的 reward gradient 方法（更简单但显存更大）。

### 基础模型：LongLive 1.3B (Wan 2.1)

已分析代码架构，修改点明确。详见 `longlive-code-analysis.md`。

### 硬件方案

**所有实验默认在 8×RTX 5090 (32GB) 上运行。**

| 任务 | 显存需求 | 5090 可行？ |
|------|---------|-----------|
| LongLive 推理（1.3B） | ~8GB | ✅ 单卡即可 |
| GRPO 后训练（1.3B generator + reward） | ~20-25GB/卡 | ✅ 8卡分布式 |
| 奖励计算（Warp + RAFT + DepthAnything） | ~5-8GB | ✅ 异步在空闲卡或 CPU 上跑 |
| 14B teacher | ~30GB | ❌ 不需要（DMD 蒸馏已完成，GRPO 阶段无需 teacher） |

GRPO 后训练不需要 DMD 的三模型架构（generator + teacher + critic），只需要:
1. Generator（1.3B，需训练）
2. Reference model（1.3B 冻结副本，用于 KL 正则化）
3. Reward model（Warp 模拟器 + 运动估计，可异步计算）

总显存 ~25GB/卡，8×5090 完全足够。H800 仅在需要时备用。

---

## 分阶段执行

### Phase 0: 环境搭建（Week 1）

```
任务 0.1: LongLive 推理验证
  - 在 8xH800 上跑通 LongLive 推理
  - 生成 10 个物理场景视频（倒水、烟雾、布料、软体），观察当前物理质量
  - 建立 baseline 视觉印象

任务 0.2: Warp 环境搭建
  - pip install warp-lang
  - 跑通 3 个 demo：
    ├─ warp/examples/sim/example_cloth.py（布料）
    ├─ warp/examples/sim/example_soft_body.py（软体）
    └─ SPH 流体 demo
  - 验证可微性：跑一个简单的梯度反传 demo

任务 0.3: DanceGRPO 环境搭建
  - clone https://github.com/XueZeyue/DanceGRPO
  - 理解代码结构，特别是 reward 接口和 GRPO 训练循环
  - 评估适配 LongLive/Wan2.1 的工作量
```

### Phase 1: 奖励函数设计与验证（Week 2-3）

**这是整个项目的技术关键——如果奖励函数不 work，后面都不成立。**

```
任务 1.1: 设计物理场景 prompt 集
  - 流体 10 个: "water pouring into a glass", "smoke rising from a candle", ...
  - 软体 10 个: "jelly wobbling on a plate", "rubber ball bouncing and deforming", ...
  - 布料 10 个: "flag waving in the wind", "tablecloth being pulled off", ...
  - 用 LongLive 生成每个 prompt 的 5 个视频样本

任务 1.2: 实现运动估计模块
  - UniMatch/RAFT → 光流提取（速度场代理）
  - DepthAnything V2 → 深度图
  - 可选: 表面法线估计
  - 对 LongLive 生成的视频跑一遍，检查估计质量

任务 1.3: 实现物理模拟模块
  - 用 Warp 搭建 3 个简化物理场景:
    ├─ 场景 A: 2D 液体倒入容器（SPH，~10K 粒子）
    ├─ 场景 B: 软球落地形变（FEM soft body）
    └─ 场景 C: 布料下落悬垂（FEM cloth）
  - 输出: 每帧的速度场 / 形变场 / 表面法线

任务 1.4: 实现奖励函数 + 验证
  - reward = -||motion_generated - motion_simulated||
  - 关键验证:
    ├─ 物理正确的视频是否得高分？
    ├─ 物理错误的视频是否得低分？
    └─ 奖励是否有足够的区分度？
  - 如果区分度不够 → 调整场景/度量/归一化
  - 如果区分度足够 → 进入 Phase 2
```

### Phase 2: GRPO 训练 Pipeline（Week 4-5）

```
任务 2.1: 适配 DanceGRPO → LongLive
  - DanceGRPO 的 GRPO 训练循环 → 接入 LongLive generator
  - 关键接口:
    ├─ 采样: LongLive generator 生成 G 个候选视频
    ├─ 奖励: 物理奖励函数打分
    ├─ 优化: GRPO 更新 generator 权重
    └─ 参考模型: 冻结的 LongLive 做 KL 正则化

任务 2.2: 单物理域 Proof-of-Concept
  - 先只用流体奖励（最简单的场景）
  - 小规模训练（100-200 步），观察:
    ├─ 奖励曲线是否上升？
    ├─ 视频物理质量是否改善？
    └─ 视觉质量是否保持？
  - 如果 work → 扩展到三种物理域
  - 如果不 work → 诊断原因，考虑切换到 DPO 或 VADER

任务 2.3: 视觉质量保护
  - 加入视觉质量正则化（FID/CLIP score 不下降）
  - 调整物理奖励 vs 质量正则化的权重
```

### Phase 3: 全量训练（Week 6-8）

```
任务 3.1: 三域联合训练
  - 流体 + 软体 + 布料三种奖励联合训练
  - 训练配置:
    ├─ 模型: LongLive 1.3B generator（不需要 14B teacher，DMD 蒸馏已完成）
    ├─ 方法: GRPO (group size=4)
    ├─ 奖励: 三域物理奖励 + 视觉质量正则化（奖励计算与训练异步，不占训练卡显存）
    ├─ 硬件: 8×RTX 5090 (32GB)
    └─ 估计时间: 2-3 天

任务 3.2: 消融实验（与 3.1 并行或之后）
  - 单域 vs 多域
  - 不同 reward 权重
  - GRPO vs DPO
  - 不同 group size
  - 有/无视觉质量正则化
```

### Phase 4: 评估（Week 9-10）

```
任务 4.1: 现有 Benchmark 评估
  - PhyGenBench: 物理生成质量
  - PhysicsIQ: 流体/软体/刚体全面评估
  - VideoPhy2: solid-fluid / fluid-fluid 交互
  - PhyWorldBench: 流体与粒子动力学
  - FVD/FID/LPIPS: 视觉质量不下降

任务 4.2: 自建 ComplexPhysBench
  - 60 个 prompt（流体/软体/布料各 20 个）
  - 每个 prompt 生成 5 个视频
  - Warp 模拟器自动评分
  - 人工评估（物理正确性 + 视觉质量）

任务 4.3: Baseline 对比
  - LongLive（无后训练）
  - LongLive + NewtonRewards 复现（刚体 RLVR）
  - Cosmos-Predict2.5（业界标杆）
  - WonderPlay（推理时模拟器）
  - 维度: 流体物理正确性、软体物理正确性、布料物理正确性、刚体物理正确性（不下降）、视觉质量
```

### Phase 5: 论文撰写（Week 11-12）

```
任务 5.1: 论文结构
  - Abstract: 问题（RLVR 只做刚体）→ 方法（可微模拟器奖励）→ 结果
  - Introduction: 动机 + 贡献（3点）
  - Related Work: RLVR for video + 可微模拟器 + 物理视频生成
  - Method: 奖励设计 + GRPO 训练 + Pipeline 架构图
  - Experiments: 定量（5 个 benchmark）+ 定性（视频对比）+ 消融
  - Conclusion + Limitations

任务 5.2: 图表
  - Figure 1: 方法总览图（prompt → 生成 → 模拟器奖励 → GRPO → 更新）
  - Figure 2: 物理域对比（刚体 vs 流体 vs 软体 vs 布料）
  - Figure 3: 定性对比（我们 vs baseline 的视频帧）
  - Table 1-3: Benchmark 定量结果

任务 5.3: Supplementary
  - 视频 demo（最重要的 selling point）
  - 更多定性结果
  - 实现细节
```

---

## 关键里程碑与决策点

| 时间 | 里程碑 | 决策 |
|------|--------|------|
| Week 1 末 | LongLive 推理 + Warp demo 跑通 | 环境 OK → 继续 |
| **Week 3 末** | **奖励函数验证通过**（区分度足够） | **最关键决策点：work → 全力推进 / 不 work → 调整方案或切备选** |
| Week 5 末 | 单域 PoC 训练成功（流体奖励提升物理质量） | 训练可行 → 扩展三域 |
| Week 8 末 | 全量训练完成 | 结果好 → 冲 NeurIPS / 结果一般 → 补实验投 ICLR |
| Week 10 末 | 评估完成，数字好看 | 开始写论文 |
| Week 12 末 | 论文完成 | 提交 |

---

## 本周（Week 1）具体 TODO

### Day 1-2: LongLive 推理
```bash
# 1. 下载 Wan 2.1 权重
# 推理只需: Wan2.1-T2V-1.3B, VAE, T5 encoder（不需要 14B teacher）
# GRPO 后训练也不需要 14B teacher（DMD 蒸馏阶段已完成）

# 2. 下载 LongLive checkpoint
# 需要: longlive_init.pt (或 ode_init.pt)

# 3. 在 5090 上跑推理（单卡 32GB 足够 1.3B 推理）
cd LongLive
bash inference.sh

# 4. 生成物理场景视频
# 准备 prompt 文件，包含流体/软体/布料场景
```

### Day 3-4: Warp 搭建
```bash
pip install warp-lang

# 跑 demo
python -m warp.examples.sim.example_cloth
python -m warp.examples.sim.example_soft_body

# 验证可微性
# 写一个简单脚本: 模拟 → loss → backward → 检查梯度
```

### Day 5-6: DanceGRPO 代码阅读
```bash
git clone https://github.com/XueZeyue/DanceGRPO
# 重点阅读:
# - GRPO 训练循环
# - reward 接口设计
# - 如何接入新模型
```

### Day 7: 精读论文
```
必读（理解 GRPO 实现细节）:
- DanceGRPO 论文 (arXiv 2505.07818)
- Flow-GRPO 论文 (arXiv 2505.05470)

必读（理解物理奖励设计）:
- NewtonRewards (arXiv 2512.00425)
- PhysRVG (arXiv 2601.11087)

参考（理解推理时物理）:
- WonderPlay (arXiv 2505.18151)
```

---

## 资源清单

### 代码仓库

| 仓库 | 用途 | URL |
|------|------|-----|
| LongLive | 基础模型 | https://github.com/NVlabs/LongLive |
| DanceGRPO | GRPO 训练框架 | https://github.com/XueZeyue/DanceGRPO |
| NVIDIA Warp | 可微物理模拟器 | https://github.com/NVIDIA/warp |
| Newton | Warp 扩展（MPM等） | https://github.com/newton-physics/newton |
| PhiFlow | 备选流体模拟器 | https://github.com/tum-pbs/PhiFlow |
| VADER | 备选 RL 框架 | https://github.com/mihirp1998/VADER |
| Flow-GRPO | GRPO 参考实现 | https://github.com/yifan123/flow_grpo |

### 论文

| 论文 | 用途 | arXiv |
|------|------|-------|
| DanceGRPO | GRPO 视频实现 | 2505.07818 |
| Flow-GRPO | GRPO + Flow Matching | 2505.05470 |
| NewtonRewards | 物理奖励设计参考 | 2512.00425 |
| PhysRVG | 物理 GRPO 参考 | 2601.11087 |
| WonderPlay | 推理时物理模拟参考 | 2505.18151 |
| PISA | 物理后训练参考 | 2503.09595 |
| DiffusionNFT | 高效 RL 方案参考 | ICLR 2026 Oral |

### 评估 Benchmark

| Benchmark | 覆盖 | URL |
|-----------|------|-----|
| PhysicsIQ | 流体/固体/光学/热力学 | https://physics-iq.github.io/ |
| VideoPhy2 | solid-fluid/fluid-fluid | https://github.com/Hritikbansal/videophy |
| PhyWorldBench | 流体与粒子动力学 | https://github.com/g-jing/phy-world-bench |
| PhyGenBench | 力学/热学/光学 | https://phygenbench123.github.io/ |

### Awesome Lists

| 列表 | URL |
|------|-----|
| Awesome-RL-for-Video-Generation | https://github.com/wendell0218/Awesome-RL-for-Video-Generation |
| Awesome-Physics-aware-Generation | https://github.com/BestJunYu/Awesome-Physics-aware-Generation |
| Awesome-Physics-Cognition-Video | https://github.com/minnie-lin/Awesome-Physics-Cognition-based-Video-Generation |

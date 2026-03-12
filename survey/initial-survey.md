# World Model（世界模型）调研报告

## 1. 定义

**世界模型**是一种 AI 系统，构建对环境的结构、动态和因果关系的内部表征。与 LLM 预测下一个 token 不同，世界模型学习物理世界如何运作——包括物理规律、空间属性、物体持久性和因果关系，能预测行动后果并模拟真实场景。

Yann LeCun 认为世界模型比当前 LLM 更适合实现"人类水平智能"。

---

## 2. 关键里程碑论文

| 时间 | 论文/工作 | 贡献 |
|------|----------|------|
| 1990 | Schmidhuber | 最早提出用循环模型学习环境动态 |
| 2018 | Ha & Schmidhuber "World Models" | VAE+RNN 架构，可在"梦境"中训练策略再迁移到真实环境 |
| 2022 | LeCun "A Path Towards Autonomous Machine Intelligence" | 提出 JEPA 框架，倡导预测性世界模型 |
| 2023 | DreamerV3 (Hafner et al.) | 通用 RL 世界模型，150+ 任务超越专用方法，首次在 Minecraft 从零收集钻石 |
| 2023 | I-JEPA (Meta) | 在潜空间而非像素空间做预测 |

---

## 3. 最新进展（2024-2025）

### 3.1 视频生成世界模型

| 系统 | 机构 | 亮点 |
|------|------|------|
| **Sora 2** | OpenAI | 物理感知视频+音频生成，被称为视频领域的"GPT-3.5 时刻" |
| **Genie 3** | Google DeepMind | 从文本/图像生成可交互 3D 环境，实时可导航 |
| **NVIDIA Cosmos** | NVIDIA | 开放权重物理 AI 基础世界模型，200万+ 下载 |
| **V-JEPA 2** | Meta | 12 亿参数，100 万+ 小时视频训练，支持零样本机器人控制 |

### 3.2 机器人世界模型

- **V-JEPA 2**：机器人无需重新训练即可在全新环境操作物体
- **NVIDIA Cosmos**：大规模生成合成训练数据，降低物理测试成本
- **Waymo**：5000 万自动驾驶里程训练，生成逼真驾驶场景

### 3.3 中国公司/实验室

| 公司 | 项目 | 详情 |
|------|------|------|
| **腾讯** | HunyuanWorld 1.0/1.1 | 3D 世界模型，WAIC 发布，完全开源，支持单卡部署 |
| **字节跳动** | Seed World Model | 模拟物理规律，用于机器人训练和自动驾驶 |
| **华为** | WEWA 模型 | 云端 WE + 车端 WA(8B参数)，车端算力减少 75% |
| **极佳视界** | GigaWorld-0 | 国内首个"纯血"物理 AI 公司，视频生成+3D+物理引擎双系统 |
| **World Labs** | Marble | 李飞飞创立，融资 $2.3 亿，空间智能 3D 世界重建 |

---

## 4. 主流架构

| 架构 | 说明 | 代表 |
|------|------|------|
| VAE + RNN | 经典方案，VAE 编码观测，RNN 建模时序动态 | Ha & Schmidhuber 2018 |
| Transformer | 自回归序列预测，二次复杂度限制上下文 | - |
| Diffusion Models | 高保真视觉生成，常作为混合架构的解码器 | Sora, Cosmos |
| State Space Models (SSM) | 恒定推理速度，长序列优势 | Mamba 变体 |
| SSM + Transformer 混合 | 当前前沿，结合长期状态追踪+局部推理 | - |
| JEPA | 潜空间预测，避免像素级重建 | V-JEPA 2 |
| 3D Gaussian Splatting + 物理引擎 | 物理 3D 世界生成 | GigaWorld-0 |

---

## 5. 应用场景

- **自动驾驶**：生成合成边缘场景用于训练和安全验证（Waymo、华为、Cosmos）
- **机器人**：零样本操作、合成训练数据、sim-to-real 迁移
- **数字孪生**：工厂/仓库/道路逼真虚拟副本
- **视频/内容生成**：物理可信的可控视频合成
- **游戏开发**：从文本/图像交互式 3D 世界创建（Genie 3）
- **科学模拟**：物理现象建模

---

## 6. 开源项目

| 项目 | 说明 | 链接 |
|------|------|------|
| DreamerV3 | RL 世界模型，150+ 任务 | https://github.com/danijar/dreamerv3 |
| HunyuanWorld-1.0 | 腾讯 3D 世界生成模型 | https://github.com/Tencent-Hunyuan/HunyuanWorld-1.0 |
| NVIDIA Cosmos | 物理 AI 基础世界模型 | https://www.nvidia.com/en-us/ai/cosmos/ |
| Meta V-JEPA 2 | 视频世界模型 | https://ai.meta.com/vjepa/ |
| Navigation World Models | Meta Research, CVPR 2025 | https://github.com/facebookresearch/nwm |
| World Model Survey | 清华大学综述 + 资源 | https://github.com/tsinghua-fib-lab/World-Model |
| Awesome-World-Models | 论文/资源索引 | https://github.com/knightnemo/Awesome-World-Models |

---

## 7. 挑战与局限

1. **物理理解不足**：视觉逼真但物理推理仍有缺陷（穿模、违反重力）
2. **领域专一性**：在受限场景表现好，跨领域泛化未解决
3. **幻觉问题**：生成看似合理但物理错误的场景
4. **数据需求巨大**：需海量高质量多模态物理交互数据
5. **计算成本高**：实时高保真世界模拟需要大量算力
6. **Sim-to-Real 差距**：模拟环境训练的模型迁移到真实环境常失败

---

## 8. 总结

世界模型是 2024-2025 年 AI 领域最热的方向之一，核心价值在于让 AI 理解物理世界规律而不仅仅是语言模式。自动驾驶和机器人是最直接的落地场景，中国公司（腾讯、字节、华为）都在积极布局。当前主要技术路线正从纯 Transformer 向 SSM+Transformer 混合架构和 JEPA 潜空间预测方向演进。

---

*调研日期：2026-03-12*

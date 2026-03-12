# Flow Matching for Generative Modeling & Rectified Flow 联合阅读笔记

> **论文 1**: Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le. *"Flow Matching for Generative Modeling"*, ICLR 2023. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747)
>
> **论文 2**: Xingchao Liu, Chengyue Gong, Qiang Liu. *"Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow"*, ICLR 2023. [arXiv:2209.03003](https://arxiv.org/abs/2209.03003)
>
> **阅读日期**: 2026-03-12

---

## 1. 一句话总结

**Flow Matching** 提出了一种基于常微分方程 (ODE) 的生成模型训练框架，通过回归 **条件向量场** 而非全局向量场/score function 来实现 simulation-free 的训练，在数学上统一了 Diffusion 和 Optimal Transport 路径，同时天然支持确定性少步采样。**Rectified Flow** 则在此基础上提出 **reflow** 操作，通过迭代地将耦合 (coupling) 拉直，使得生成路径趋近直线，从而用 1-2 步 Euler 即可高质量生成。

---

## 2. 核心贡献

### Flow Matching (Lipman et al.)

1. **提出 Conditional Flow Matching (CFM) 框架**：将不可计算的全局向量场分解为条件向量场的混合，使训练变为简单的回归问题，无需模拟 ODE 或计算 divergence。
2. **统一框架**：证明 Diffusion Models (VP-SDE/VE-SDE) 和 Optimal Transport 插值都是 CFM 的特例，只是概率路径 (probability path) 的选择不同。
3. **引入 OT 路径 (Optimal Transport conditional path)**：相比 Diffusion 路径，OT 路径更直、方差更低，在少步采样下质量显著更好。
4. **Simulation-free 训练**：与 Continuous Normalizing Flows (CNFs) 不同，CFM 不需要在训练时求解 ODE，训练效率与 DDPM 相当。
5. **实验验证**：在 CIFAR-10 和 ImageNet 上取得了与 Diffusion 方法可比的 FID，同时在少步采样场景下明显优于 Diffusion。

### Rectified Flow (Liu et al.)

1. **Reflow 操作**：通过反复用模型生成 noise-data pair，然后重新训练直线回归，使流的路径越来越直。
2. **直线度 (Straightness) 度量**：形式化定义了流的"弯曲度"，并证明 reflow 单调减少弯曲度。
3. **蒸馏 + Reflow 组合**：2 次 reflow + 蒸馏 → 1-step 高质量生成。
4. **理论保证**：证明 rectified flow 是 noise-data 之间的最优传输映射的逼近。

---

## 3. 方法详解

### 3.1 核心数学框架：Flow-based ODE

**基本设定**：我们的目标是学习一个从简单先验分布 p₀（通常是标准高斯 N(0, I)）到目标数据分布 q（即 p₁）的变换。

定义一个时间依赖的向量场 vₜ : ℝᵈ → ℝᵈ，其生成的 **流 (flow)** φₜ 满足 ODE：

```
dφₜ(x)/dt = vₜ(φₜ(x)),    φ₀(x) = x,    t ∈ [0, 1]
```

如果 vₜ 是 Lipschitz 连续的，则 φₜ 存在且唯一（Picard-Lindelöf 定理）。

**概率路径 (Probability Path)**：定义 pₜ 为时间 t 时刻的概率密度，它通过连续性方程 (continuity equation) 与 vₜ 关联：

```
∂pₜ/∂t + ∇ · (pₜ vₜ) = 0
```

等价地，pₜ 是将初始密度 p₀ 通过流 φₜ 推前 (push-forward) 得到的：

```
pₜ = [φₜ]_# p₀
```

**生成过程**：训练好 vₜ 后，采样过程为：
1. 从 p₀ = N(0, I) 采样 x₀
2. 求解 ODE：dx/dt = vₜ(x)，从 t=0 到 t=1
3. 得到的 x₁ 即为生成样本

与 Diffusion 的 SDE 采样不同，这是一个**确定性** ODE，不需要注入随机噪声。

---

### 3.2 Flow Matching (FM) 目标函数

**理想目标**：如果我们知道真实的向量场 uₜ(x) 能生成路径 pₜ，我们可以用以下损失来训练神经网络 vθ：

```
L_FM(θ) = E_{t∼U[0,1], x∼pₜ} ‖vθ(t, x) - uₜ(x)‖²
```

**问题**：uₜ(x) 是未知的！我们知道 p₀ 和 p₁（数据分布），但连接它们的中间路径 pₜ 以及对应的向量场 uₜ(x) 并不唯一，而且即使选定了一条路径，直接计算 uₜ(x) 也是 intractable 的（因为 pₜ 本身就不知道）。

这就是 Conditional Flow Matching 要解决的问题。

---

### 3.3 Conditional Flow Matching (CFM) — 为什么训练变简单了

**核心思想**：不要试图直接回归全局向量场 uₜ(x)，而是通过 **条件向量场** uₜ(x | x₁) 来间接构造它。

**Step 1：定义条件概率路径**

给定一个数据样本 x₁ ∼ q(x₁)，定义从 p₀ 到以 x₁ 为中心的 delta 分布 δ(x - x₁) 的条件路径：

```
pₜ(x | x₁) = N(x | μₜ(x₁), σₜ(x₁)²I)
```

其中 μₜ 和 σₜ 满足边界条件：
- t=0: μ₀ = 0, σ₀ = 1 → pₜ=₀(x | x₁) = N(0, I) = p₀
- t=1: μ₁ = x₁, σ₁ = σ_min ≈ 0 → pₜ=₁(x | x₁) ≈ δ(x - x₁)

**Step 2：条件向量场**

对于高斯条件路径，条件流的显式形式为：

```
ψₜ(x₀ | x₁) = σₜ(x₁) · x₀ + μₜ(x₁)
```

其对应的条件向量场为：

```
uₜ(x | x₁) = [σₜ'(x₁)/σₜ(x₁)] · (x - μₜ(x₁)) + μₜ'(x₁)
```

其中 ' 表示对 t 的导数。

**Step 3：边际化得到全局向量场**

全局概率路径和向量场通过对 x₁ 积分（边际化）得到：

```
pₜ(x) = ∫ pₜ(x | x₁) q(x₁) dx₁

uₜ(x) = ∫ uₜ(x | x₁) · [pₜ(x | x₁) q(x₁) / pₜ(x)] dx₁
```

**Step 4：CFM 损失等价于 FM 损失**

**定理 (Lipman et al., Theorem 2)**：Conditional Flow Matching 损失

```
L_CFM(θ) = E_{t∼U[0,1], x₁∼q, x∼pₜ(·|x₁)} ‖vθ(t, x) - uₜ(x | x₁)‖²
```

的梯度与 L_FM(θ) 的梯度相同（即 ∇θ L_CFM = ∇θ L_FM）。

**为什么这很重要？**

- L_FM 不可计算（需要知道 uₜ(x)）
- L_CFM **可以计算**！因为 uₜ(x | x₁) 有显式公式，而且从 pₜ(x | x₁) 采样就是简单的高斯采样
- 两者的梯度相同，所以优化 L_CFM 等价于优化 L_FM

**训练的具体操作**非常简单：

```python
# 训练一步
t = uniform(0, 1)                           # 采样时间
x_1 = sample_from_data()                     # 采样真实数据
x_0 = randn_like(x_1)                        # 采样噪声
x_t = (1 - (1-sigma_min)*t) * x_0 + t * x_1  # OT 路径插值
target = x_1 - (1-sigma_min) * x_0           # 条件向量场目标
loss = ||v_theta(t, x_t) - target||^2        # 回归损失
```

对比 DDPM 的训练：

```python
# DDPM 训练一步
t = randint(0, T)
x_0 = sample_from_data()
eps = randn_like(x_0)
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * eps
loss = ||eps_theta(t, x_t) - eps||^2
```

**形式上几乎一样！** 区别在于：
1. 插值公式不同（线性 vs 非线性 noise schedule）
2. 回归目标不同（向量场 vs 噪声）
3. FM 的 t 是连续的 [0,1]，DDPM 的 t 是离散的

---

### 3.4 不同概率路径的选择

#### (a) Optimal Transport (OT) 路径 — Lipman et al. 推荐

```
μₜ(x₁) = t · x₁
σₜ(x₁) = 1 - (1 - σ_min) · t
```

对应的条件流：
```
ψₜ(x₀ | x₁) = (1 - (1-σ_min)t) · x₀ + t · x₁
```

条件向量场：
```
uₜ(x | x₁) = (x₁ - (1-σ_min) · x₀)
            = (x₁ - (1-σ_min) · x₀)   [常数，不依赖于 t！]
```

**关键性质**：OT 路径下的条件向量场是时间无关的常数！这意味着：
- 条件轨迹 ψₜ 是从 x₀ 到 x₁ 的**直线**
- 流非常平滑，Euler 离散化的误差很小
- 少步 ODE 求解就能得到高质量结果

#### (b) VP (Variance Preserving) 路径 — 对应 DDPM/Score-based

```
μₜ(x₁) = α_t · x₁      (α_t 从 0 增到 1)
σₜ(x₁) = β_t            (β_t 从 1 减到 σ_min)
```

通常取 α_t = cos(πt/2)，β_t = sin(πt/2)（余弦调度）或其他非线性调度。

条件向量场（对应 VP 路径）：
```
uₜ(x | x₁) = [β̇_t/β_t] · (x - α_t · x₁) + α̇_t · x₁
```

这个向量场是时间依赖的、非线性的，导致轨迹弯曲。

#### (c) 直观对比

```
OT 路径（直线）:          Diffusion 路径（曲线）:

x₁ ●                     x₁ ●
   /                        |  \
  /                         |   \
 /                          |    |
● x₀                       ●----● x₀

路径是直线               先快速加噪再慢慢去噪
Euler 1 步误差小         Euler 1 步误差大
```

**数学直觉**：OT 路径本质上是高斯分布之间的 Wasserstein-2 最优传输。给定 p₀ = N(0,I) 和 "p₁" = N(x₁, σ_min²I)，McCann 插值（即两个高斯之间的 W₂ 测地线）恰好就是线性插值 pₜ = N(tμ₁, ((1-t)σ₀ + tσ₁)²I)。

---

### 3.5 与 Diffusion (Score Matching) 的精确关系

Flow Matching 和 Score-based Diffusion 在数学上有精确的对应关系：

**Score Function 与向量场的转换**：

对于高斯条件路径 pₜ(x|x₁) = N(μₜ, σₜ²I)，score function 为：

```
∇_x log pₜ(x|x₁) = -(x - μₜ(x₁)) / σₜ(x₁)²
```

而边际 score 和边际向量场的关系为：

```
uₜ(x) = [σ̇ₜ · σₜ · ∇_x log pₜ(x)] + μ̇ₜ    (某些参数化下)
```

更精确地说，对于 VP 路径，Flow Matching 的向量场 vₜ(x) 可以由 score function sₜ(x) = ∇_x log pₜ(x) 表达为：

```
vₜ(x) = f_t · x + g_t² · sₜ(x) / 2        (probability flow ODE 的向量场)
```

其中 f_t, g_t 是 VP-SDE 的 drift 和 diffusion 系数。

**核心区别总结**：

| 维度 | Score Matching / DDPM | Flow Matching (CFM) |
|------|----------------------|---------------------|
| 基础方程 | SDE: dx = f(x,t)dt + g(t)dW | ODE: dx/dt = vₜ(x) |
| 训练目标 | 预测噪声 ε 或 score ∇log p | 预测向量场 vₜ |
| 需要的先验知识 | Noise schedule α_t, β_t | 概率路径 μₜ, σₜ（更灵活） |
| 默认路径 | 非线性（VP/VE schedule） | 线性（OT 路径） |
| 采样 | 需要多步（SDE 或 ODE solver） | ODE solver，天然更少步 |
| 数学统一性 | Score Matching 是 CFM 的特例 | 更通用的框架 |

---

### 3.6 Rectified Flow 的 Reflow 操作

Rectified Flow (Liu et al.) 的核心洞察是：即使条件路径是直线，**边际路径**仍然可能弯曲（因为不同数据点的条件直线路径会交叉）。

#### 基本定义

**Rectified Flow** 定义为连接随机耦合 (X₀, X₁) ∼ π（其中 X₀ ∼ p₀, X₁ ∼ p₁）的线性插值的速度场：

```
X_t = (1-t) · X₀ + t · X₁
```

速度场 v 通过最小化回归损失学习：

```
min_v E_{(X₀,X₁)∼π} ∫₀¹ ‖v(X_t, t) - (X₁ - X₀)‖² dt
```

这与 CFM 的 OT 路径在形式上完全一致（σ_min→0 时）。

#### 路径交叉问题

关键问题：不同 (x₀, x₁) 对的直线路径可能在某个 t 处交叉。当路径交叉时，在交叉点处的速度场必须"折中"（平均），导致学到的流在该点处偏离直线。

```
x₁_A ●--------→       ←--------● x₁_B
        \   ╳   /                    交叉！
         \ / \ /
          ╳   ╳
         / \ / \
        /   ╳   \
x₀_A ●--------→       ←--------● x₀_B
```

#### Reflow 操作

**Reflow** 的思想是：用已训练好的模型重新生成 (noise, data) 配对，然后用这些新配对重新训练。

**算法**：

```
Reflow 第 k 轮:
1. 有上一轮训练好的向量场 v_{k-1}
2. 采样 z₀ ~ p₀ (噪声)
3. 用 v_{k-1} 求解 ODE: dX/dt = v_{k-1}(X_t, t), X(0)=z₀ → 得到 X̂₁ = X(1)
4. 新的耦合: (z₀, X̂₁) — 注意：z₀ 和 X̂₁ 现在不再是独立的！
5. 用新耦合 (z₀, X̂₁) 重新训练直线回归 → 得到 v_k
```

**为什么 Reflow 让路径更直？**

- 初始训练：(X₀, X₁) 是独立的随机耦合 → 直线路径大量交叉
- Reflow 后：(z₀, X̂₁) 是通过 ODE 流确定性关联的 → 交叉大大减少
- 交叉减少 → 学到的向量场更接近直线 → Euler 少步误差更小

**定理 (Liu et al.)**：Reflow 单调不增加路径的 **凸性代价 (convex transport cost)**：

```
C(π_k) ≤ C(π_{k-1})
```

其中 π_k 是第 k 轮 reflow 的耦合，C(π) = E_{(X₀,X₁)∼π}[c(X₀,X₁)] 对于凸代价函数 c。

特别地，当 c(x,y) = ‖x-y‖² 时，这等价于 Wasserstein-2 代价单调递减，即 reflow 使耦合越来越接近最优传输映射。

#### 直线度 (Straightness) 度量

```
S(v) = E ∫₀¹ ‖v(X_t, t) - (X₁ - X₀)‖² dt
```

直线度为 0 当且仅当所有轨迹都是直线。Reflow 单调减小 S(v)。

#### 与蒸馏 (Distillation) 的结合

Reflow 之后，可以进一步做蒸馏来实现 1-step 生成：

```
1-Reflow → 路径更直但仍需 ~10 步
2-Reflow → 路径更直，~2-4 步
2-Reflow + Distillation → 1 步
```

蒸馏方法：用多步 ODE 求解的输出作为 teacher，训练 student 在 1 步内直接映射。

---

### 3.7 为什么 Flow Matching 天然适合少步生成 (1-4 步)

从多个角度理解：

**(1) ODE vs SDE**：Flow Matching 使用确定性 ODE，不需要 SDE 的多步随机噪声注入。Diffusion 的 DDPM 采样本质上是 SDE 的离散化，需要足够多步来维持噪声水平的正确衰减。

**(2) OT 路径的直线性**：OT 路径下条件轨迹是直线，1 步 Euler 就能精确积分条件 ODE。边际 ODE 的轨迹虽然不完全是直线，但比 Diffusion 路径直得多。

**(3) 向量场的平滑性**：OT 路径的条件向量场是常数（不依赖 t），这意味着它是 "Lipschitz 连续且有界" 的最理想情况。VP 路径的条件向量场在 t→1 时会发散（σₜ→0），导致数值不稳定。

**(4) Reflow 进一步拉直**：通过 1-2 轮 reflow，边际轨迹也趋近直线，此时 Euler 1 步的全局误差非常小。

**量化对比**（概念性）：

```
方法                    达到 FID<10 所需步数
DDPM (1000 步调度)     ~50-200 步
DDIM                   ~20-50 步
DPM-Solver (高阶)     ~10-20 步
FM + OT 路径           ~5-20 步
FM + OT + 1-Reflow     ~2-5 步
FM + OT + 2-Reflow     ~1-2 步
```

---

## 4. 实验结果

### 4.1 Flow Matching (Lipman et al.)

#### CIFAR-10 无条件生成

| 方法 | NFE (步数) | FID ↓ |
|------|-----------|-------|
| DDPM (Ho et al.) | 1000 | 3.17 |
| Score SDE (VP) | 1000 | 2.41 |
| FM-OT (本文) | 142 | **2.99** |
| FM-OT (本文) | 91 | 3.53 |
| FM-VP (本文) | 142 | 3.28 |

#### 关键观察

1. **OT 路径在少步时显著优于 VP 路径**：当 NFE 从 142 降到 ~20 时，OT 路径的 FID 劣化远小于 VP 路径。
2. **OT-CFM 在 100+ 步时与 Score SDE 质量相当**，但在 10-20 步时 OT-CFM 大幅领先。
3. **训练效率**：FM 的训练速度与 DDPM 几乎相同（同样的回归损失，只是目标和插值不同），不需要像 CNF 那样在训练时求解 ODE。

#### ImageNet 结果

- 在 ImageNet 64x64 和 128x128 上，FM-OT 在相同步数下持续优于 VP 路径
- 与 EDM (Karras et al.) 等 SOTA Diffusion 方法相比，FM-OT 在 < 50 步时有明显优势

### 4.2 Rectified Flow (Liu et al.)

#### CIFAR-10

| 方法 | 步数 | FID ↓ |
|------|------|-------|
| DDPM | 1000 | 3.17 |
| 1-Rectified Flow | 1 | ~6.18 |
| 2-Rectified Flow | 1 | ~4.85 |
| 2-Rectified Flow + Distill | 1 | ~3.36 |
| 1-Rectified Flow | 2 | ~4.1 |

#### LSUN Bedroom 256x256

| 方法 | 步数 | FID ↓ |
|------|------|-------|
| DDPM | 1000 | ~4.89 |
| 1-Rectified Flow | 100 | ~4.2 |
| 2-Rectified Flow | 2 | ~6.5 |

#### ImageNet 256x256

Rectified Flow 在配合现代架构（如 DiT）时效果更佳。后续工作 Stable Diffusion 3、FLUX 等大规模模型验证了 Flow Matching + Rectified Flow 在高分辨率图像生成上的卓越表现。

#### 质量-速度 Tradeoff 总结

```
100步:  FM ≈ Diffusion（质量持平）
 20步:  FM >> Diffusion（FM 几乎无损，Diffusion 明显劣化）
  5步:  FM 仍可用，Diffusion 基本不可用（除非配合特殊 solver）
  1步:  需要 Reflow/蒸馏，FM 可达到接受质量
```

---

## 5. 对 MambaWorld 的启发

### 5.1 将 Flow Matching 替换 Diffusion 用于帧生成

在 MambaWorld 架构中（Mamba-2 做时序建模 + 生成器做帧合成），帧生成模块可以从 DDPM 替换为 Conditional Flow Matching：

**当前典型 World Model 帧生成（DIAMOND/Oasis 风格）**：
```
输入: 上下文帧特征 c (由 Mamba/Transformer 编码)
过程: DDPM 去噪 x_T → x_{T-1} → ... → x_0, 共 T 步 (通常 T=20-50)
输出: 生成帧 x_0
```

**替换为 Flow Matching 后**：
```
输入: 上下文帧特征 c (由 Mamba-2 编码)
过程: ODE 求解 x_0 → x_1, 1-4 步 Euler
输出: 生成帧 x_1
```

### 5.2 训练的具体改动

**损失函数替换**：

```python
# 旧: DDPM 训练
def ddpm_loss(model, x_0, context):
    t = randint(0, T)  # 离散时间步
    eps = torch.randn_like(x_0)
    x_t = sqrt_alpha_bar[t] * x_0 + sqrt_one_minus_alpha_bar[t] * eps
    eps_pred = model(x_t, t, context)
    return F.mse_loss(eps_pred, eps)

# 新: Flow Matching (OT-CFM) 训练
def flow_matching_loss(model, x_1, context):
    t = torch.rand(x_1.shape[0], device=x_1.device)  # 连续 t ∈ [0,1]
    x_0 = torch.randn_like(x_1)  # 噪声
    # OT 线性插值
    t_expand = t.view(-1, 1, 1, 1)
    x_t = (1 - t_expand) * x_0 + t_expand * x_1
    # 目标: 条件向量场 = x_1 - x_0 (OT路径, sigma_min→0)
    target = x_1 - x_0
    v_pred = model(x_t, t, context)
    return F.mse_loss(v_pred, target)
```

**注意**：
- 模型架构几乎不需要改动（输入输出维度一样，只是时间嵌入从离散变连续）
- 时间嵌入：从离散 embedding table → 连续 sinusoidal/Fourier embedding
- `x_0` 和 `x_1` 的语义反过来了：FM 中 `x_0` 是噪声，`x_1` 是数据（与 DDPM 相反）

### 5.3 推理的具体改动

```python
# 旧: DDPM 推理 (50步 DDIM)
def ddpm_sample(model, context, num_steps=50):
    x = torch.randn(...)  # x_T
    for t in reversed(timesteps):
        x = ddim_step(model, x, t, context)
    return x

# 新: Flow Matching 推理 (4步 Euler)
def fm_sample(model, context, num_steps=4):
    x = torch.randn(...)  # x_0 (噪声)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = i * dt
        v = model(x, t, context)
        x = x + v * dt  # Euler 步
    return x  # x_1 (生成的帧)
```

**速度提升估算**：
- Diffusion (DDIM 50步) → FM (Euler 4步): **12.5x 推理加速**
- Diffusion (DDIM 20步) → FM (Euler 1步 + Reflow): **20x 推理加速**
- 结合 Mamba-2 的 O(n) 时序建模（vs Transformer O(n²)），总体速度提升可达 **10-50x**

### 5.4 1-step / 4-step 生成的可行性

**4-step 生成（推荐起步方案）**：
- 不需要 reflow，直接用 OT-CFM 训练即可
- 4 步 Euler 在 OT 路径下已足够好（条件轨迹是直线，4步足以逼近边际轨迹）
- 实现最简单，训练流程与 DDPM 几乎相同

**1-step 生成（进阶方案）**：
- 需要 1-2 轮 Reflow + 蒸馏
- Reflow 每轮需要：(a) 用当前模型生成所有训练数据的 noise-data pair (b) 重新训练
- 计算成本：约 2-3 倍原始训练
- 质量可能有微小下降，但速度提升巨大
- 适合实时交互式世界模型（>30 FPS）

**推荐路线**：
```
Phase 1: OT-CFM + 4步 Euler → 验证 FM 在世界模型中可行
Phase 2: 1-Reflow + 2步 Euler → 提升速度
Phase 3: 2-Reflow + 蒸馏 → 1步，实时推理
```

### 5.5 动作条件化的兼容性

Flow Matching 的条件生成与 Diffusion 完全一致——通过 cross-attention 或 AdaLN 将条件信息 c（包含过去帧、动作编码）注入即可：

```
v_θ(x_t, t, c)  代替  v_θ(x_t, t)
```

在 MambaWorld 中，c 来自 Mamba-2 backbone 输出的隐状态，包含：
- 历史帧的时序信息
- 当前动作 embedding
- 可选的语言/物理条件

注入方式可沿用 DiT 的 AdaLN-Zero 或 cross-attention，无需特殊修改。

---

## 6. 局限性 / 注意事项

### 6.1 Flow Matching 本身的局限

1. **边际路径仍然弯曲**：虽然条件路径是直线（OT路径下），但多个条件路径的混合（边际路径）不一定是直线。Reflow 可以缓解但不能完全解决。

2. **高维下的 OT 近似**：CFM 使用的是 **独立耦合** (X₀, X₁ 独立采样) + 条件 OT 路径，而非真正的全局最优传输。真正的 mini-batch OT 耦合（如 OT-CFM with batch OT）可以进一步改善，但计算开销增加。

3. **模式覆盖 vs 模式匹配**：ODE-based 方法（确定性采样）有时会出现模式覆盖 (mode covering) 而非模式匹配 (mode seeking) 的行为，可能导致生成样本略显模糊。SDE-based 方法有时在某些指标上更好。

4. **t 的采样分布**：均匀采样 t∈[0,1] 不一定是最优的。后续工作（如 Stable Diffusion 3 的 logit-normal 分布）发现调整 t 的采样权重可以显著改善质量。

### 6.2 Reflow 的额外成本

1. **Reflow 需要额外训练轮次**：每次 reflow 需要先用当前模型生成一遍数据，再重新训练。对于大模型+大数据，计算成本不可忽略。
2. **误差累积**：reflow 使用的 (z₀, X̂₁) 配对依赖于上一轮模型的质量。如果上一轮模型有偏差，误差会传递。

### 6.3 在世界模型中的特殊考虑

1. **自回归误差累积**：世界模型连续生成帧，FM 的 1-step 采样虽然快但可能引入偏差，长序列下偏差会累积。需要在 "速度 vs 保真度" 之间找平衡。建议在自回归世界模型中使用 2-4 步而非 1 步。

2. **时序一致性**：FM 的每帧独立采样（给定条件 c）。确保时序一致性仍依赖于条件编码器（Mamba-2）的质量。FM 本身不保证帧间一致。

3. **训练稳定性**：OT-CFM 训练通常比 DDPM 更稳定（损失方差更低），但在世界模型的条件生成设置中尚缺充分验证，需要实验确认。

4. **与 latent space 的配合**：如果使用 VAE 的 latent space（而非像素空间），FM 在 latent space 中的 OT 路径是否仍优于 diffusion 路径需要验证。经验上（SD3、FLUX）已证实在 latent space 中 FM 仍然有效，但最优的 t 采样策略可能需要调整。

---

## 附录：关键公式速查表

| 概念 | 公式 |
|------|------|
| Flow ODE | dφₜ/dt = vₜ(φₜ), φ₀ = x |
| 连续性方程 | ∂pₜ/∂t + ∇·(pₜvₜ) = 0 |
| FM 损失 | L_FM = E_{t,x∼pₜ} ‖vθ(t,x) - uₜ(x)‖² |
| CFM 损失 | L_CFM = E_{t,x₁∼q,x∼pₜ(·\|x₁)} ‖vθ(t,x) - uₜ(x\|x₁)‖² |
| OT 条件路径 | ψₜ(x₀\|x₁) = (1-t)x₀ + tx₁ |
| OT 条件向量场 | uₜ(x\|x₁) = x₁ - x₀ |
| OT 插值 | xₜ = (1-t)x₀ + tx₁ |
| 直线度 | S(v) = E∫₀¹ ‖v(Xₜ,t) - (X₁-X₀)‖² dt |
| Reflow 耦合 | π_k: (z₀, φ₁^{k-1}(z₀)), z₀∼p₀ |

---

*本笔记基于对两篇论文的深入阅读，重点关注了与 MambaWorld 项目相关的技术细节。后续工作如 SD3 (Esser et al., 2024)、FLUX (Black Forest Labs, 2024)、SiT (Scalable Interpolant Transformers, Ma et al., 2024) 进一步验证了 Flow Matching 在大规模生成模型中的有效性。*

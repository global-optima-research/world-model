# Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality

**作者**: Tri Dao, Albert Gu
**会议**: ICML 2024
**论文**: https://arxiv.org/abs/2405.21060
**代码**: https://github.com/state-spaces/mamba

---

## 一句话总结

Mamba-2 通过建立 **结构化状态空间模型（SSM）与注意力机制的数学对偶关系**（SSD framework），证明了 Transformer 是 SSM 的一个特例，并据此设计了一个兼具 SSM 的线性复杂度和注意力灵活性的新架构，训练速度比 Mamba-1 快 2-8 倍，性能持平或更优。

---

## 核心贡献

1. **SSD 对偶性理论框架**：证明了一类结构化 SSM（其状态转移矩阵 A 为对角+低秩结构）等价于一种半可分矩阵（semiseparable matrix）的矩阵乘法，而注意力机制本质也是矩阵乘法——两者存在严格的数学对偶。这意味着 SSM 和 attention 是同一个计算的两种不同算法实现。

2. **高效 SSD 算法**：基于矩阵分块分解（block decomposition），将 SSM 的计算拆分为 chunk 内的二次方注意力计算 + chunk 间的线性递推，实现了比 Mamba-1 快 2-8 倍的训练速度，同时保持 O(N) 的总体复杂度。

3. **Mamba-2 架构**：新的神经网络架构，引入多头 SSM（multi-head SSM，类比多头注意力），大幅缩小状态维度的同时维持甚至提升性能。架构更简洁——将 Mamba-1 的串行门控改为并行投影。

4. **统一理论视角**：将线性注意力（linear attention）、H3、Hyena、RWKV、RetNet 等一系列"高效 Transformer 替代品"纳入同一框架，揭示了它们本质上都是结构化矩阵乘法的不同限制形式。

5. **实证验证**：在语言建模任务上，Mamba-2 在多个尺度（370M, 1.3B, 2.7B）上匹配或超越 Mamba-1 和 Transformer++，训练和推理效率显著提升。

---

## 方法详解

### 3.1 SSD (Structured State Space Duality) 的核心思想

#### 状态空间模型回顾

连续时间 SSM 定义为：
```
h'(t) = A h(t) + B x(t)
y(t)  = C h(t)
```

离散化后（零阶保持，ZOH）：
```
h_t = Ā h_{t-1} + B̄ x_t
y_t = C_t h_t
```

其中 Ā = exp(ΔA)，B̄ = (ΔA)^{-1}(exp(ΔA) - I)·ΔB。

Mamba（选择性 SSM）的关键创新：让 B, C, Δ 都依赖于输入（input-dependent），即 B_t = f_B(x_t)，C_t = f_C(x_t)，Δ_t = f_Δ(x_t)。

#### 从 SSM 到矩阵乘法

展开 SSM 的递推关系，可以得到：

```
y_t = C_t^T h_t = C_t^T (Ā_t h_{t-1} + B̄_t x_t)
    = C_t^T Ā_t Ā_{t-1} ... Ā_{s+1} B̄_s x_s  (对所有 s ≤ t 求和)
```

定义矩阵 M ∈ R^{T×T}，其中：
```
M_{ts} = C_t^T (∏_{i=s+1}^{t} Ā_i) B̄_s    当 t ≥ s
M_{ts} = 0                                    当 t < s
```

则 y = Mx，这是一个下三角矩阵乘法。

#### 半可分矩阵 (Semiseparable Matrix)

关键观察：矩阵 M 是一个 **N 阶半可分矩阵**（N-semiseparable matrix），意味着 M 的任何下三角子矩阵的秩最多为 N（N 是状态维度）。

直觉：M 的 (t,s) 元素可以写成 C_t^T · (状态转移的累积乘积) · B_s，这是两个 N 维向量的内积（经过状态转移），所以任何子矩阵的秩 ≤ N。

#### 与注意力的对偶

标准（因果）注意力计算：
```
Y = softmax(QK^T / √d) · V
```

去掉 softmax 后的线性注意力：
```
Y = (QK^T ⊙ L) · V    （L 是因果掩码，下三角全 1 矩阵）
```

矩阵 M = QK^T ⊙ L 是一个秩 ≤ d 的下三角矩阵——这恰好是一个 d 阶半可分矩阵！

所以：
- **SSM → 半可分矩阵乘法**（通过展开递推）
- **线性注意力 → 半可分矩阵乘法**（通过 QK^T 因果掩码）
- **两者是同一个数学对象的不同计算方式**

#### 对偶的精确对应

| SSM 视角 | 注意力视角 |
|----------|-----------|
| 状态维度 N | 头维度 d |
| B_t（输入投影） | K_t（Key） |
| C_t（输出投影） | Q_t（Query） |
| x_t（标量输入） | V_t（Value） |
| A_t（状态转移） | 因果掩码 L（binary） |

SSM 比线性注意力更 expressive 的地方在于：A_t 提供了 **数据依赖的衰减**（data-dependent decay），而线性注意力的因果掩码是固定的 0/1。

### 3.2 SSD 的结构约束

为了让对偶性在计算上可行，Mamba-2 对状态转移矩阵 A 施加约束：

**A 必须是标量乘以单位矩阵**，即 A_t = a_t · I，其中 a_t 是标量。

这意味着：
```
∏_{i=s+1}^{t} Ā_i = (∏_{i=s+1}^{t} a_i) · I
```

累积乘积简化为标量乘积，矩阵 M 简化为：
```
M_{ts} = (∏_{i=s+1}^{t} a_i) · C_t^T B_s
```

这可以重写为：
```
M = L ⊙ (CB^T)
```

其中 L_{ts} = ∏_{i=s+1}^{t} a_i 是一个 **1-semiseparable 矩阵**（秩 1 的衰减掩码），⊙ 是逐元素乘积。

与线性注意力对比：
- 线性注意力：M = L_causal ⊙ (QK^T)，L_causal 是 0/1 掩码
- SSD：M = L_decay ⊙ (QK^T)，L_decay 是 data-dependent 的指数衰减掩码

### 3.3 SSD 的高效算法：chunk-wise 分块计算

核心思想：将序列分成长度为 c 的 chunk，每个 chunk 内用二次方注意力（O(c²)），chunk 之间用线性递推（O(N)），总复杂度 O(T·(c + N))。

**算法 1：SSD 前向传播**

输入：序列 X ∈ R^{T×P}，Q, K ∈ R^{T×N}，衰减因子 a ∈ R^T（标量）

1. **分块**：将序列分成 T/c 个长度为 c 的 chunk
2. **Chunk 内（二次方路径）**：
   - 对每个 chunk j，计算局部注意力矩阵：
   ```
   M_j = L_j ⊙ (Q_j K_j^T)    ∈ R^{c×c}
   ```
   其中 L_j 是 chunk 内的衰减掩码
   - Y_inner_j = M_j · X_j    （chunk 内贡献）

3. **Chunk 间（线性递推路径）**：
   - 计算每个 chunk 末尾的 SSM 状态：
   ```
   h_j = decay · h_{j-1} + ∑_t K_{j,t} ⊗ X_{j,t}    ∈ R^{N×P}
   ```
   - 使用状态计算跨 chunk 贡献：
   ```
   Y_cross_j = Q_j · h_{j-1}    （经过衰减调制）
   ```

4. **合并**：Y_j = Y_inner_j + Y_cross_j

**计算复杂度分析**：
- Chunk 内：O(T/c · c² · P) = O(T · c · P) — 当 c 较小时高效
- Chunk 间：O(T/c · N · P) — 线性递推
- 总计：O(T · (c + N/c) · P)，当 c = √N 时最优，为 O(T · √N · P)
- 实际实现中 c 选 64-256，平衡 GPU 利用率

#### 与 FlashAttention 的类比

SSD 的分块策略直接借鉴了 FlashAttention 的思想：
- FlashAttention：分块计算注意力，用 online softmax 在 SRAM 中累积
- SSD：分块计算 SSM，用 SSM 状态在 chunk 边界传递信息
- 两者都避免了将 T×T 矩阵实例化到 HBM

关键实现细节：
- Chunk 内的 c×c 矩阵在 SRAM（shared memory）中计算，不写回 HBM
- 状态 h ∈ R^{N×P} 在 chunk 间通过寄存器/SRAM 传递
- 使用 Triton 实现 fused kernel，一次 pass 完成前向

### 3.4 Mamba-2 与 Mamba-1 的关键区别

| 特性 | Mamba-1 | Mamba-2 |
|------|---------|---------|
| 状态转移 A | 对角矩阵 A ∈ R^{N×N}，每个状态维度独立衰减 | 标量 A = a·I，所有状态维度共享衰减率 |
| 状态维度 N | 通常 16，需要大 N 因为 A 的表达力靠 N 个独立参数 | 可以更大（64-256），因为引入了多头来补偿 |
| 头结构 | 无显式头概念 | 多头 SSM（multi-head, multi-value, multi-query 等） |
| 块结构 | Mamba Block: 串行 conv → SSM → gate | 简化块: 并行投影 → grouped SSM → norm → 输出 |
| 选择性扫描 | 依赖自定义 CUDA kernel | 基于矩阵分块的 Triton/CUDA 实现 |
| 训练速度 | 基线 | 2-8x 加速 |
| 推理速度 | 与 Mamba-2 相当（都是 O(1) per step） | 略快（更小的有效状态） |
| 归一化 | RMSNorm 在块之间 | GroupNorm 在 SSM 之后（类似 attention 后的 norm） |

#### Mamba-2 Block 结构

```
输入 x
  ↓
线性投影 → [z, x', B, C, dt]  （并行投影，不像 Mamba-1 的串行）
  ↓
x' → 短卷积 (conv1d, kernel=4)
  ↓
SSD Layer(x', B, C, dt, A)
  ↓
GroupNorm
  ↓
⊙ SiLU(z)    （门控）
  ↓
线性投影 → 输出
```

Mamba-1 Block 对比：
```
输入 x
  ↓
线性投影 → [x', z]
  ↓
x' → conv1d → SiLU → SSM投影(B,C,dt) → selective_scan → ⊙ SiLU(z) → 线性输出
```

关键区别：Mamba-2 将 B, C, dt 的投影从 SSM 内部移到外部并行化，更利于 tensor parallelism。

### 3.5 多头结构

Mamba-2 引入了类似 Transformer 的多头机制。设：
- H = 头数
- d = 每头的 key/query 维度（对应 SSM 状态维度 N per head）
- P = 每头的 value 维度
- D = 模型维度 = H × P

**Multi-Head SSM (MHS)**：
- 每个头有独立的 Q_h, K_h ∈ R^{T×d} 和 V_h ∈ R^{T×P}
- 每个头有独立的衰减标量 a_h
- 参数量：H × (d + d + P) per timestep

**Multi-Value SSM (MVS)**：
- 所有头共享 Q, K（类似 multi-query attention 的反面）
- 每个头有独立的 V_h
- 更少参数，实验表明效果几乎一样

**Multi-Query SSM (MQS)**：
- 类似 multi-query attention：一组 K, V，多组 Q
- 推理时 KV 状态可共享

实际实现中（代码层面）：
```python
# mamba_ssm/modules/mamba2.py
class Mamba2(nn.Module):
    def __init__(self, d_model, d_state=128, d_conv=4, expand=2,
                 headdim=64, ngroups=1, ...):
        self.d_inner = int(expand * d_model)  # 扩展维度
        self.nheads = self.d_inner // headdim  # 头数
        self.ngroups = ngroups  # K,V 的组数（1=multi-query, nheads=multi-head）

        # 并行投影：一次生成 z, x, B, C, dt
        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=False)
```

`ngroups` 控制头结构：
- `ngroups = nheads`：Multi-Head SSM（每头独立 K）
- `ngroups = 1`：Multi-Query SSM（所有头共享 K）
- `1 < ngroups < nheads`：Grouped-Query SSM（类似 GQA）

默认配置用 `ngroups=1`（multi-query），因为实验发现效果与 multi-head 相当，但更高效。

### 3.6 Hardware-Efficient 实现细节

**Triton Kernel 实现**（`mamba_ssm/ops/triton/ssd_combined.py`）：

1. **Fused Chunk Scan**：
   - 一个 kernel 完成整个 SSD 前向
   - 输入 Q, K, V 和衰减因子
   - 分配 shared memory 给 chunk 内的 c×c 注意力矩阵
   - 状态 h 在 chunk 间通过寄存器传递

2. **内存访问模式**：
   - Q, K 按 (batch, head, seq, dim) 布局
   - V 按 (batch, seq, head, dim) 布局（利于 coalesced access）
   - Chunk 大小 c = 256（默认），平衡计算与内存

3. **反向传播**：
   - 需要重计算 chunk 内的注意力矩阵（类似 FlashAttention 的 recompute）
   - 状态梯度通过反向递推传播

4. **数值稳定性**：
   - 衰减因子 a_t 通过 log-space 计算避免下溢：
   ```
   log_decay_cumsum = cumsum(log(a_t))
   L_{ts} = exp(log_decay_cumsum[t] - log_decay_cumsum[s])
   ```
   - 实际使用 `-softplus(param)` 参数化 log(a_t)，确保 |a_t| < 1

---

## 实验结果

### 4.1 语言建模性能

**训练设置**：
- 数据集：The Pile（类似 Chinchilla 配方）
- Tokenizer：GPT-NeoX tokenizer
- 模型尺度：370M, 1.3B, 2.7B
- 训练 token 数：300B

**Perplexity 对比**（The Pile 验证集）：

| 模型 | 370M | 1.3B | 2.7B |
|------|------|------|------|
| Transformer++ | ~12.0 | ~9.5 | ~8.7 |
| Mamba-1 | ~11.6 | ~9.2 | ~8.5 |
| Mamba-2 | ~11.5 | ~9.15 | ~8.4 |
| Mamba-2 (更大 state) | ~11.4 | ~9.1 | ~8.35 |

关键发现：
- Mamba-2 在所有尺度上略优于 Mamba-1
- 增大状态维度（128 → 256）可以进一步提升，Mamba-1 因计算瓶颈难以做到
- 与 Transformer++ 的差距随模型增大而缩小
- 在某些下游评估（如 zero-shot）上 Mamba-2 已与 Transformer++ 持平

### 4.2 速度对比

**训练吞吐量**（tokens/sec，A100 80GB）：

| 模型 | 370M | 1.3B | 2.7B |
|------|------|------|------|
| Transformer (FlashAttention-2) | 基线 | 基线 | 基线 |
| Mamba-1 | ~1.2x | ~1.3x | ~1.5x |
| Mamba-2 | ~2.5x | ~3.0x | ~3.5x |

**SSD vs Selective Scan kernel 对比**（单层，序列长度 2048）：
- Mamba-2 SSD 比 Mamba-1 selective scan 快 **2-8 倍**
- 加速比随状态维度增大而增大（Mamba-1 的 selective scan 对大 N 很慢）

**序列长度扩展**：
- Mamba-1：序列长度增大时吞吐量下降平缓（线性复杂度）
- Mamba-2：比 Mamba-1 更快（chunk-wise 算法更 hardware-friendly）
- Transformer：二次方下降

### 4.3 扩展性实验

**Scaling Laws**：
- Mamba-2 的 loss 随计算量增长的斜率与 Transformer++ 相近
- Chinchilla-optimal 配比下，Mamba-2 在 compute-matched 对比中优于 Transformer++
- 状态维度 N 的增大提供额外增益，但收益递减

**混合架构**：
- 论文探索了 Mamba-2 + 少量 attention 层的混合架构
- 2-4 层 sliding window attention + 其余 Mamba-2 层 → 在需要精确复制的任务上显著提升
- 这表明 SSM 在某些 retrieval-heavy 任务上仍有局限

**Ablation 关键发现**：
- A 的参数化：标量 A（Mamba-2）vs 对角 A（Mamba-1）→ 性能差异极小，但速度差异巨大
- 头维度 d=64 是最优 sweet spot（类似 Transformer 的 d_head=64）
- 短卷积（conv1d）仍然重要，移除会降低约 0.1 perplexity
- GroupNorm 比 RMSNorm 更优（放在 SSD 后面，类似 attention 后 norm）

---

## 对 MambaWorld 的启发

### 5.1 SSD 层用于视频世界模型的时序建模

**直接应用场景**：在 MambaWorld 中用 SSD 层替换 causal attention 做时序帧预测。

**架构设计建议**：

```
视频帧序列: [f_1, f_2, ..., f_T]  （经过 tokenizer 后）
    ↓
每帧 tokens → spatial encoder（DiT / local attention）
    ↓
帧级 embeddings → SSD 时序层（替换 causal attention）
    ↓
条件化的下一帧 latent → 解码器（Flow Matching / Diffusion）
```

**为什么 SSD 比 Mamba-1 更适合视频世界模型**：

1. **速度**：视频序列长（256-1024 帧 × 每帧多 token），SSD 的 chunk-wise 算法训练快 2-8x
2. **多头结构**：视频的不同通道/频率可以用不同的头建模（类似视频 Transformer 中多头的作用）
3. **可与 attention 混合**：个别层用 sliding window attention 处理需要精确空间对齐的操作（如物体追踪），其余用 SSD
4. **状态维度可扩展**：视频需要更大的状态来记忆场景信息，SSD 在大 N 时仍高效

**chunk 大小选择**：
- 论文默认 c=256，对 NLP 有效
- 视频场景建议 c=64-128（每帧有多个 spatial token，一个 chunk 对应几帧，符合视频的时间局部性）
- 可做 ablation：c 太小则 chunk 间递推开销大，c 太大则 chunk 内二次方计算慢

### 5.2 Cross-Sequence Conditioning（动作注入）

MambaWorld 需要将动作信号注入时序模型，使世界模型能预测"执行动作 a_t 后的下一帧"。

**方案 1：通过 B 和 Δ 注入（推荐）**

SSD 的选择性来自 B_t, C_t, Δ_t 对输入的依赖。可以让这些参数同时依赖于输入帧和动作：

```python
# 将动作 embedding 与帧 embedding 拼接/融合后再投影
x_combined = frame_embed + action_proj(action_embed)  # 或 concat + linear
B_t, C_t, dt = proj(x_combined)  # SSD 的选择性参数依赖于 action
```

这样，动作信号直接影响状态的写入（B）和读出（C），以及时间步长（Δ），从而影响状态转移的衰减率。

**方案 2：Cross-SSM 注入（类似 cross-attention）**

受 Transformer 中 cross-attention 的启发：

```python
# 主序列提供 Q (C), 动作序列提供 K (B) 和 V (X)
# 即在 SSD 中：
# C 来自帧序列
# B 来自动作序列
# X 来自动作序列
# 这实现了"用帧去查询动作信息"
```

但这需要修改 SSD kernel，实际操作比较复杂。更实用的做法是方案 1。

**方案 3：FiLM 条件化（简单有效）**

```python
# 在 SSD 层之后，用动作对输出做 FiLM 调制
gamma, beta = film_proj(action_embed)  # 学习 scale 和 shift
y = gamma * ssd_output + beta
```

**推荐方案**：方案 1 + 方案 3 组合。方案 1 在选择性参数层面注入动作信息（影响 SSM 的动态），方案 3 在输出层面做调制（直接影响预测）。Po et al. (ICCV 2025) 也采用了类似策略。

### 5.3 多头结构的选择

对 MambaWorld 的建议：

- **使用 Grouped-Query SSM (ngroups=4-8)**：
  - 比 multi-query (ngroups=1) 表达力更强
  - 比 multi-head (ngroups=nheads) 更高效
  - 视频场景中不同头可以学习到不同的时序模式（快速运动、慢速背景、物体交互等）

- **头维度 d=64-128**：
  - 论文发现 d=64 是 NLP 的 sweet spot
  - 视频场景可能需要更大状态维度来记忆空间信息，建议尝试 d=128

- **模型维度配置建议**：
  ```
  d_model = 1024
  expand = 2 → d_inner = 2048
  headdim = 64 → nheads = 32
  ngroups = 8 → 每组 4 个头共享 K
  d_state = 128 → 每头 128 维状态
  ```

---

## 局限性 / 我们可以改进的点

### 论文本身的局限

1. **标量 A 的表达力限制**：为了计算效率，Mamba-2 将对角矩阵 A 退化为标量 a·I。这意味着所有状态维度共享同一个衰减率，理论上表达力弱于 Mamba-1 的对角 A。虽然实验中差异不大（多头补偿了这个损失），但在需要不同遗忘速率的场景中可能有限制。
   - **改进方向**：在 MambaWorld 中尝试 "grouped decay"——将状态维度分成几组，每组一个衰减率。这介于标量 A 和对角 A 之间，可能在不显著增加计算的情况下提升表达力。

2. **因果掩码限制**：SSD 只支持因果（左到右）的序列建模。视频世界模型有时需要双向上下文（如空间维度上的双向交互）。
   - **改进方向**：空间维度用双向 attention 或双向 SSM（如 Vision Mamba 的做法），时序维度用因果 SSD。

3. **精确检索能力弱**：SSM（包括 Mamba-2）在需要从长序列中精确复制/检索信息的任务上弱于 attention。论文自己也承认需要混合少量 attention 层。
   - **对 MambaWorld 的影响**：物体追踪、场景一致性等需要精确空间对应的任务可能受影响。建议在模型中保留 2-4 层局部 attention。

4. **chunk 大小的权衡**：chunk 大小 c 是一个需要调优的超参数。太小 → chunk 间通信开销大；太大 → chunk 内二次方计算慢。最优 c 取决于硬件（SRAM 大小）和任务（序列长度）。
   - **改进方向**：实现 adaptive chunk size，根据序列长度和可用 SRAM 动态调整。

### MambaWorld 特有的改进点

5. **多模态条件化**：Mamba-2 论文只处理单一序列。MambaWorld 需要同时处理视频帧 + 动作 + 可能的文本描述。需要设计有效的多模态融合方案。
   - **建议**：参考 Dynalang (ICML 2024) 的语言条件化世界模型设计，在 SSD 层增加 cross-modal 条件化。

6. **变长序列的 chunk 对齐**：视频中不同 episode 长度不同，需要处理 padding 和 chunk 边界的对齐。
   - **建议**：在训练时使用 episode-aware chunking，确保 chunk 边界不跨越 episode 边界。

7. **与 Flow Matching 的结合**：SSD 输出的是确定性的下一状态预测。如何将其与 Flow Matching 生成模型结合？
   - **建议方案**：SSD 预测条件化 latent，作为 Flow Matching 的条件输入（类似 DiT 中 class embedding 的作用）。SSD 处理时序建模，Flow Matching 处理每帧的生成。

8. **状态重置策略**：在世界模型中，当进入新场景/新 episode 时需要重置 SSM 状态。
   - **建议**：训练时在 episode 边界显式重置状态 h=0；推理时提供 reset token。

---

## 关键数学公式速查

### SSD 核心公式

**离散 SSM（选择性）**：
```
h_t = a_t · h_{t-1} + B_t x_t        （a_t 是标量衰减）
y_t = C_t^T h_t
```

**等价矩阵形式**：
```
Y = M · X
M_{ts} = C_t^T · (∏_{i=s+1}^{t} a_i) · B_s    (t ≥ s)
M = L ⊙ (C B^T)     （L 是衰减掩码，⊙ 是 Hadamard 乘积）
```

**衰减掩码**：
```
L_{ts} = exp(∑_{i=s+1}^{t} log(a_i))    (t ≥ s)
L_{ts} = 0                                (t < s)
```

**分块 SSD（chunk j 内，chunk 大小 c）**：
```
内部贡献: Y_j^{inner} = (L_j ⊙ Q_j K_j^T) · V_j     ∈ R^{c×P}
状态更新: h_j = decay_j · h_{j-1} + K_j^T · diag(decay) · V_j    ∈ R^{N×P}
跨块贡献: Y_j^{cross} = diag(decay) · Q_j · h_{j-1}    ∈ R^{c×P}
总输出:   Y_j = Y_j^{inner} + Y_j^{cross}
```

**参数化**：
```
a_t = exp(-softplus(a_param))    ∈ (0, 1)    # 确保稳定衰减
Δ_t = softplus(dt_param + dt_bias)            # 时间步长
B_t = linear(x_t)                             # 输入依赖
C_t = linear(x_t)                             # 输入依赖
```

---

## 代码实现关键入口

```
mamba/                              # GitHub: state-spaces/mamba
├── mamba_ssm/
│   ├── models/
│   │   └── mixer_seq_simple.py     # 模型组装（MambaLMHeadModel）
│   ├── modules/
│   │   ├── mamba_simple.py         # Mamba-1 Block
│   │   └── mamba2.py               # Mamba-2 Block（核心模块）
│   ├── ops/
│   │   └── triton/
│   │       ├── ssd_combined.py     # SSD 前向/反向 Triton kernel
│   │       ├── ssd_chunk_scan.py   # chunk-wise scan 实现
│   │       └── ssd_chunk_state.py  # chunk 间状态传递
│   └── layers/
│       └── mamba2.py               # Mamba-2 层的包装
```

---

*阅读日期: 2026-03-12*
*状态: 基于论文原文和代码的详细笔记，WebFetch 受限未能实时抓取最新版本*

"""
对应课案章节：网络架构 / Attention（通用形式、自注意力、掩码、多头注意力）

本节知识点：
    1. Attention 的通用形式：按相关性筛选信息，接受 Q（查询）、K（键）、V（值）三组输入，
       通用公式 Attention(Q,K,V) = softmax(QKᵀ/√d_k)·V；三步走
       —— 算相似度（Q·K 点积）→ 归一化（除 √d_k 再 Softmax）→ 加权求和（对 V 加权平均）。
    2. 为什么除以 √d_k：q·k = Σ q_i k_i 的方差是 d_k，d_k 越大点积越极端、Softmax 越接近
       one-hot、梯度越小。用 d_k=8 与 d_k=512 的数值实验打印 scores.std()、熵、最大权重、
       梯度范数来证明，并说明除以 √d_k 把方差拉回 1。
    3. Cross-Attention（Q 来自一个序列，K/V 来自另一个序列，seq_q ≠ seq_k）
       与 Self-Attention（Q/K/V 来自同一个序列）的区别与形状。
    4. 自注意力：同一份输入 X 经过 W_Q / W_K / W_V 三组不同线性变换得到 Q、K、V。
    5. 手写 scaled_dot_product_attention(Q, K, V, mask=None)，支持 (batch, heads, seq, d_k)
       四维输入；验证 output/weights 形状与「每行权重和为 1」。
    6. 掩码 mask：scores.masked_fill(mask == 0, -1e9) 的原理；causal mask（下三角，逐 token
       可见性表格并验证上三角权重为 0）；padding mask（屏蔽补齐位置并验证其权重为 0）；
       为什么用 -1e9 而不是 -inf。
    7. 多头拆分的形状变化：split_heads 把 (batch, seq, d_model) →
       (batch, seq, num_heads, d_k) → (batch, num_heads, seq, d_k)；
       合并用 transpose(1,2).contiguous().view(...)；用 is_contiguous() 打印
       transpose 后为 False、contiguous() 后为 True。
    8. 手写完整 MultiHeadAttention 模块（W_Q/W_K/W_V/W_O 四个 nn.Linear，
       d_k = d_model // num_heads，assert d_model % num_heads == 0）。
    9. 注意力权重可视化：注意力热力图（行=Query，列=Key）+ 因果掩码可见性矩阵。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Deep_Learning\\02_网络架构\\05_Attention与自注意力.py'
"""

# ---------------------------------------------------------------------------
# 统一环境初始化：目录定位 + 无界面绘图后端
# ---------------------------------------------------------------------------
from pathlib import Path

_HERE = Path(__file__).resolve().parent     # 当前脚本所在目录
_ROOT = _HERE.parent                        # Deep_Learning 根目录

import matplotlib
matplotlib.use("Agg")                       # 无界面后端，避免 plt.show() 阻塞
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = _ROOT / "output"               # 所有图片统一输出到这里
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 业务依赖
# ---------------------------------------------------------------------------
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 统一随机种子（本脚本里还会多次重设，用于公平对照）
torch.manual_seed(42)
np.random.seed(42)

# CPU 版 PyTorch：显式限制线程数，减少与其他并行脚本争抢 CPU 造成的耗时抖动
torch.set_num_threads(4)

_T_START = time.perf_counter()


def section(title: str) -> None:
    """打印分节标题，让长输出的结构清晰可读。"""
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def shape(x):
    """把张量形状格式化成 tuple，便于 print 展示。"""
    return tuple(x.shape)


# ===========================================================================
# 一、通用形式：Q / K / V 与三步走
# ===========================================================================
section("一、Attention 的通用形式：Q / K / V")

print(
    r"""
【核心思想】注意力机制是「按相关性筛选信息」的操作。
    模型在处理某个位置时，不平均地看待所有信息，而是**根据相关性分配不同权重**：
    相关的多看一眼（权重大），不相关的少看甚至不看（权重小）。

【三组输入】
    向量            中文名      类比
    Query  (Q)      查询向量    「我想找什么」—— 当前要处理的位置发出的检索请求
    Key    (K)      键向量      「我是什么」  —— 每个位置对外的索引标签
    Value  (V)      值向量      「我的内容」  —— 每个位置真正携带的信息

    用图书馆检索打比方：Query 是你输入的搜索词，Key 是每本书的标题/标签，
    Value 是书的正文。先拿搜索词和每个标签比对（算相似度），
    匹配度高的书多读一点、匹配度低的少读一点，最后把读到的内容综合起来。

【通用公式】
        Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) · V            ... (1.1)

【计算过程分三步】
    1) 算相似度：Q 和 K 做点积 → 分数矩阵 scores = Q Kᵀ，
       表示「每个 Query 位置和每个 Key 位置的匹配程度」；
    2) 归一化：除以 √d_k（防止点积过大），再经过 Softmax → 注意力权重，
       让每一行变成「和为 1 的权重分布」；
    3) 加权求和：用注意力权重对 V 做加权平均 → 输出。

【形状约定】（本脚本统一使用这个记法）
        Q: (..., seq_q, d_k)
        K: (..., seq_k, d_k)
        V: (..., seq_k, d_v)        通常 d_v = d_k
        scores = Q Kᵀ: (..., seq_q, seq_k)      ← 转置的是**最后两维**
        output: (..., seq_q, d_v)               ← 查询位置有几个，输出就有几个
    注意：K 和 V 的序列长度必须相同（都是 seq_k，因为一个 Key 配一个 Value），
          而 Q 的长度 seq_q 可以不同 —— 这正是 Cross-Attention 的基础。
"""
)

# 先做一次最小可运行的演示，把「三步」用真实数字走一遍
torch.manual_seed(42)
Q_demo = torch.randn(2, 3, 4)      # (batch=2, seq_q=3, d_k=4)
K_demo = torch.randn(2, 3, 4)      # (batch=2, seq_k=3, d_k=4)
V_demo = torch.randn(2, 3, 4)      # (batch=2, seq_k=3, d_v=4)

scores_demo = torch.matmul(Q_demo, K_demo.transpose(-2, -1))   # 第 1 步：点积算相似度
weights_demo = F.softmax(scores_demo / math.sqrt(Q_demo.size(-1)), dim=-1)  # 第 2 步：缩放 + Softmax
output_demo = torch.matmul(weights_demo, V_demo)               # 第 3 步：对 V 加权求和

print("【三步走的最小演示】")
print(f"    Q {shape(Q_demo)}  K {shape(K_demo)}  V {shape(V_demo)}")
print(f"    第 1 步 scores = Q Kᵀ           → {shape(scores_demo)}   （每个 Query 对每个 Key 的相似度）")
print(f"    第 2 步 weights = softmax(scores/√d_k) → {shape(weights_demo)}")
print(f"    第 3 步 output = weights · V    → {shape(output_demo)}")
print(f"    权重矩阵第 0 行（第一个 Query 对各 Key 的注意力）：{weights_demo[0, 0].tolist()}")
print(f"    该行求和 = {weights_demo[0, 0].sum().item():.6f}（Softmax 保证每行和为 1）")

# ===========================================================================
# 二、为什么除以 √d_k
# ===========================================================================
section("二、为什么除以 √d_k（数值实验证明）")

print(
    r"""
【推导：点积的方差随 d_k 线性增长】
    假设 q、k 的每个分量都独立、均值为 0、方差为 1（标准正态），即
        E[q_i] = E[k_i] = 0,   Var(q_i) = Var(k_i) = 1

    点积 q · k = Σ_{i=1}^{d_k} q_i k_i，每一项 q_i k_i 的期望和方差：
        E[q_i k_i] = E[q_i] E[k_i] = 0
        Var(q_i k_i) = E[(q_i k_i)²] - (E[q_i k_i])² = E[q_i²] E[k_i²] = 1 × 1 = 1

    d_k 个**独立**项相加，方差相加：
        Var(q · k) = Σ_{i=1}^{d_k} Var(q_i k_i) = d_k                ... (2.1)
    即标准差  std(q · k) = √d_k。

    d_k = 512 时标准差约 22.6 —— 分数动辄 ±20 以上。

【后果：Softmax 变得极端，梯度消失】
    Softmax 的输入数值越大，输出分布越「尖锐」（越接近 one-hot）：
        softmax 的雅可比含因子 σ_i(1 - σ_i)，当某个权重趋近 1 时该因子趋近 0，
        于是「几乎没被选中的位置」收到的梯度趋于 0 —— 这些位置的参数就学不动了。
    所以 d_k 越大 → 点积数值越大 → Softmax 越极端 → 梯度越小 → 训练越困难。

【解决办法：除以 √d_k】
        缩放的乘数 1/√d_k 作用在方差上是平方：(1/√d_k)² = 1/d_k，
        于是
            Var( (q·k) / √d_k ) = d_k / d_k = 1                    ... (2.2)
        标准差回到 1，Softmax 的输入始终在合理量级，与 d_k 无关。

    下面的数值实验会逐条验证 (2.1) 和 (2.2)，并打印 Softmax 的熵、最大权重和梯度范数。
"""
)

print("【实验 1：点积的标准差是否等于 √d_k】")
print(f"    {'d_k':>6s}  {'√d_k':>8s}  {'不缩放 scores.std()':>20s}  {'缩放后 scores.std()':>20s}")
_stat_rows = []       # 保存下来给后面的「Softmax 极端程度」实验复用
for d_k in (8, 512):
    torch.manual_seed(0)
    # 用 2048 个 Query × 512 个 Key 组成足够大的样本，让统计量稳定
    q = torch.randn(2048, d_k)
    k = torch.randn(512, d_k)
    raw = q @ k.T                      # 不缩放的点积
    scaled = raw / math.sqrt(d_k)      # 缩放后的点积
    _stat_rows.append((d_k, raw, scaled))
    print(f"    {d_k:6d}  {math.sqrt(d_k):8.3f}  {raw.std().item():20.3f}  {scaled.std().item():20.3f}")
print("    → 不缩放时 d_k=8 的 std ≈ 2.83 = √8，d_k=512 的 std ≈ 22.6 = √512，完全符合 (2.1)；")
print("      缩放后两者都回到 1 附近，与 d_k 无关，符合 (2.2)。")

print("\n【实验 2：Softmax 的输出有多极端（熵、最大权重）】")
print("    熵的定义：H = -Σ p_i log p_i，越接近 one-hot 熵越小；均匀分布时熵最大 = ln(512) ≈ 6.24")
print(f"    {'d_k':>6s}  {'缩放':>6s}  {'最大权重':>10s}  {'熵':>9s}  {'有效关注位置数':>14s}")
_grad_inputs = {}
for d_k, raw, scaled in _stat_rows:
    for tag, sc in (("否", raw), ("是", scaled)):
        p = F.softmax(sc, dim=-1)                      # 每一行是一个概率分布
        max_w = p.max(dim=-1).values.mean().item()     # 平均最大权重
        ent = -(p * torch.log(p + 1e-12)).sum(dim=-1).mean().item()   # 平均熵
        # 困惑度 exp(H) 可以理解为「有效关注的位置个数」
        print(f"    {d_k:6d}  {tag:>6s}  {max_w:10.4f}  {ent:9.4f}  {math.exp(ent):14.1f}")
        # 留一份小规模切片，供下面实验 3 观察 Softmax 的梯度用
        _grad_inputs[(d_k, tag)] = sc[:64, :min(sc.size(-1), 256)].clone()
print("    → 不缩放时 d_k=512 的最大权重明显更大、熵明显更小、有效关注位置数更少：")
print("      注意力被少数几个位置「吃掉」，这就是所谓 Softmax 趋于 one-hot。")

print("\n【实验 3：极端 Softmax 导致梯度变小（直接验证「训练困难」）】")
print("    做法：把分数矩阵送进 Softmax，让 loss = (softmax(scores) @ V).sum()，")
print("          观察 loss 对**分数**的梯度范数。梯度越小，说明该位置越难被调整。")
print("    为了让 d_k=512 这组也能算得动，这里只取前 64 行、前 256 列（规律完全一样）。")
print(f"    {'d_k':>6s}  {'缩放':>6s}  {'梯度范数':>14s}")
for d_k in (8, 512):
    for tag in ("否", "是"):
        base = _grad_inputs[(d_k, tag)]          # (64, ≤256) 的分数矩阵切片
        q = base.clone().requires_grad_(True)
        p = F.softmax(q, dim=-1)                 # 注意力权重
        v = torch.randn(q.size(-1), 8)           # V 的形状 (seq_k, d_v)
        loss = (p @ v).sum()
        loss.backward()
        print(f"    {d_k:6d}  {tag:>6s}  {q.grad.norm().item():14.6f}")
print("    → 同样构造下，未缩放（分数数值大）时 Softmax 的梯度明显更小，")
print("      因为 Softmax 已接近 one-hot，其雅可比因子 σ_i(1-σ_i) → 0，梯度被「压平」。")
print("      这就是 Transformer 里 Attention 一定要除以 √d_k 的原因。")

# ===========================================================================
# 三、Cross-Attention 与 Self-Attention
# ===========================================================================
section("三、Cross-Attention vs Self-Attention")

print(
    r"""
【区别只看 Q、K、V 的来源】
    · Cross-Attention（交叉注意力）：Q 来自一个序列，K, V 来自**另一个**序列。
      典型场景是机器翻译的 Decoder：Q 是「已生成的目标语言片段」，
      K、V 是「Encoder 编码好的源语言句子」——
      即「用目标语言的当前状态，去源语言里检索相关信息」。
      形状上：seq_q ≠ seq_k 是很正常的。

    · Self-Attention（自注意力）：Q、K、V 全部来自**同一个**序列。
      即「句子里的每个词，去同一句话里找和它相关的其它词」。
      形状上：seq_q == seq_k == 序列长度。

    两者用的是**同一个**注意力公式 (1.1)，区别只在输入怎么来。
"""
)


def attention_basic(Q, K, V, mask=None):
    """最朴素的注意力实现（无多头维度），用于演示 cross / self 的形状。

    Q: (batch, seq_q, d_k)，K: (batch, seq_k, d_k)，V: (batch, seq_k, d_v)
    返回 output: (batch, seq_q, d_v)，weights: (batch, seq_q, seq_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)   # 相似度 + 缩放
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)                 # 屏蔽不可见位置
    weights = F.softmax(scores, dim=-1)                              # 归一化成权重
    output = torch.matmul(weights, V)                                # 对 V 加权求和
    return output, weights


torch.manual_seed(42)
# ---- Cross-Attention：Q 来自「目标序列」（长 4），K/V 来自「源序列」（长 7）----
cross_q = torch.randn(2, 4, 16)     # (batch=2, seq_q=4, d_k=16)
cross_k = torch.randn(2, 7, 16)     # (batch=2, seq_k=7, d_k=16)  ← 长度可以不同
cross_v = torch.randn(2, 7, 16)     # (batch=2, seq_k=7, d_v=16)
cross_out, cross_w = attention_basic(cross_q, cross_k, cross_v)
print("【Cross-Attention】Q 来自目标序列，K/V 来自源序列（长度不同）")
print(f"    Q {shape(cross_q)}   K {shape(cross_k)}   V {shape(cross_v)}")
print(f"    output  {shape(cross_out)}   ← 第一维随 Q（4 个查询位置），所以是 4")
print(f"    weights {shape(cross_w)}     ← (seq_q, seq_k) = (4, 7)，行数跟 Q、列数跟 K")
print(f"    每行权重和 = {cross_w.sum(-1)[0].tolist()}（都等于 1）")

# ---- Self-Attention：Q、K、V 都来自同一个序列 ----
torch.manual_seed(42)
self_x = torch.randn(2, 5, 16)      # (batch=2, seq_len=5, d_model=16)
self_out, self_w = attention_basic(self_x, self_x, self_x)   # 直接拿 X 当 Q、K、V
print("\n【Self-Attention】Q、K、V 来自同一个序列（这里先直接用 X 本身）")
print(f"    X {shape(self_x)}")
print(f"    output  {shape(self_out)}   ← 与输入长度相同")
print(f"    weights {shape(self_w)}     ← (seq_len, seq_len) = (5, 5)，是个方阵")
print(f"    权重矩阵第 0 行：{self_w[0, 0].tolist()}")
print("    注意：这里 Q=K=V=X，相似度就是 X 和自身各位置的点积，")
print("          但真正的自注意力不会这么用 —— 因为「查询用的表示」和「被查询的表示」")
print("          需求不同，应该用三组不同的权重投影出来。这就是下一节的内容。")

# ===========================================================================
# 四、自注意力：同一份 X 乘三个不同的权重矩阵
# ===========================================================================
section("四、自注意力：X 经过三组线性变换得到 Q / K / V")

print(
    r"""
【关键点】同一份输入 X，乘上三个**不同**的权重矩阵，得到三个**不同用途**的向量：

        Q = X · W_Q          「我作为查询时，想找什么」
        K = X · W_K          「我作为被查询对象时，提供什么索引」
        V = X · W_V          「我作为内容提供者时，贡献什么信息」

    为什么不能像上一节那样直接令 Q = K = V = X？
    因为 X 只有一套表示，而「提问」「被检索」「提供内容」是三种不同角色，
    强制共用同一套表示会让模型失去灵活性（比如一个词作为查询时关注主语，
    作为内容时又要提供自己的语义）。三组独立的线性变换让模型自己学出这三套角色。

【形状】X: (batch, seq_len, d_model)
        W_Q / W_K / W_V: (d_model, d_k)
        → Q / K / V: (batch, seq_len, d_k)

【相似度分数】scores = Q Kᵀ / √d_k，形状 (batch, seq_len, seq_len)
        第 (i, j) 个元素表示「第 i 个位置（Query）和第 j 个位置（Key）的相关程度」。

【缩放与归一化】除以 √d_k 防止点积数值过大（见第二节），再 Softmax 得到权重。

【加权求和】output = weights · V，形状 (batch, seq_len, d_v)。
        注意力权重越大，对应 token 的 Value 对输出影响越大。
"""
)

torch.manual_seed(42)
X_self = torch.randn(2, 5, 16)          # (batch=2, seq_len=5, d_model=16)
W_Q = nn.Linear(16, 8, bias=False)      # 三个**独立**的线性变换，d_model=16 → d_k=8
W_K = nn.Linear(16, 8, bias=False)
W_V = nn.Linear(16, 8, bias=False)

q_self = W_Q(X_self)                    # (2, 5, 8)
k_self = W_K(X_self)                    # (2, 5, 8)
v_self = W_V(X_self)                    # (2, 5, 8)

print("【同一份 X，三组不同权重 → 三种用途的向量】")
print(f"    X {shape(X_self)}  →  W_Q {shape(W_Q.weight)} / W_K {shape(W_K.weight)} / W_V {shape(W_V.weight)}")
print(f"    Q = X W_Q → {shape(q_self)}")
print(f"    K = X W_K → {shape(k_self)}")
print(f"    V = X W_V → {shape(v_self)}")
print(f"    Q、K、V 的数值是否相同？（应各自不同）")
print(f"        allclose(Q, K) = {torch.allclose(q_self, k_self)}")
print(f"        allclose(Q, V) = {torch.allclose(q_self, v_self)}")

sa_out, sa_w = attention_basic(q_self, k_self, v_self)
print(f"    相似度分数 scores 形状：{shape(torch.matmul(q_self, k_self.transpose(-2, -1)))}"
      f"  ← (seq_len, seq_len) 的方阵")
print(f"    注意力权重 weights 形状：{shape(sa_w)}")
print(f"    输出 output 形状：{shape(sa_out)}   ← 每个位置都被重新表示成了「相关的 V 的加权和」")
print(f"    weights[0] 第 0 行：{sa_w[0, 0].tolist()}")

# ===========================================================================
# 五、手写 scaled_dot_product_attention（课案原函数）+ 掩码
# ===========================================================================
section("五、手写 scaled_dot_product_attention（支持 mask）")


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Scaled Dot-Product Attention（课案原函数，增加了 mask 支持）。

    参数：
        Q: (batch, heads, seq_q, d_k)
        K: (batch, heads, seq_k, d_k)
        V: (batch, heads, seq_k, d_v)
        mask: 与 scores 同形状（或可广播到该形状）的 0/1 张量，
              1 = 可见（保留），0 = 屏蔽（置为 -1e9）
    返回：
        output: (batch, heads, seq_q, d_v)
        attention_weights: (batch, heads, seq_q, seq_k)
    """
    d_k = Q.size(-1)

    # 1. Q 和 K 做点积（转置最后两维），并除以 sqrt(d_k) 防止数值过大
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. 掩码：把不可见位置的分数置成 -1e9，Softmax 后这些位置权重≈0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 3. Softmax 得到注意力权重（沿最后一维 = Key 维度归一化）
    attention_weights = F.softmax(scores, dim=-1)

    # 4. 注意力权重乘以 V，得到加权求和的结果
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


torch.manual_seed(42)
Q = torch.randn(2, 4, 5, 8)      # batch=2, heads=4, seq=5, d_k=8
K = torch.randn(2, 4, 5, 8)
V = torch.randn(2, 4, 5, 8)

output, weights = scaled_dot_product_attention(Q, K, V)

print("【课案测试：Q,K,V = torch.randn(2,4,5,8)】")
print(f"    output shape: {shape(output)}")
print(f"    weights shape: {shape(weights)}")
print(f"    output.shape == (2,4,5,8) ？ {shape(output) == (2, 4, 5, 8)}")
print(f"    weights.shape == (2,4,5,5) ？ {shape(weights) == (2, 4, 5, 5)}")
print("    解读：d_v = d_k = 8，所以 output 的最后一维还是 8；")
print("          weights 的最后两维是 (seq_q, seq_k) = (5, 5)。")

row_sums = weights.sum(-1)
print("\n【验证：每个 Query 对所有 Key 的权重之和为 1】")
print(f"    weights.sum(-1) 形状：{shape(row_sums)}")
print(f"    最小值 {row_sums.min().item():.8f}，最大值 {row_sums.max().item():.8f}")
print(f"    torch.allclose(weights.sum(-1), torch.ones_like(...)) = "
      f"{torch.allclose(row_sums, torch.ones_like(row_sums))}")
print("    → True 说明 Softmax 的归一化正确：注意力权重是一组概率分布。")
print("    注意：如果 K 的维度比 V 大（d_k ≠ d_v），output 的最后一维会变成 d_v；")
print("          本脚本里 d_k = d_v，所以最后一个维度保持不变。")

# ===========================================================================
# 六、掩码 mask：causal 与 padding
# ===========================================================================
section("六、掩码 mask：causal mask 与 padding mask")

print(
    r"""
【原理】掩码的本质是「在算 Softmax 之前，把不许看的位置分数压到极小的负数」：
        scores = scores.masked_fill(mask == 0, -1e9)
    mask 里 1 表示可见、0 表示屏蔽。被置成 -1e9 的位置，经过 Softmax 后
        exp(-1e9) / Σ exp(...) ≈ 0
    所以权重几乎精确为 0 —— 相当于「这些位置完全没被看到」。

【为什么用 -1e9 而不是 -inf？】
    如果整行都被 mask 掉（全 -inf），Softmax 的分母是 exp(-inf)+...+exp(-inf) = 0，
    分子也是 0，就会算出 0/0 = **NaN**，NaN 会污染整个网络。
    用 -1e9 这种「足够小但有限的数」时，exp(-1e9) 下溢成 0，
    但分母至少还有……嗯，如果整行都是 -1e9，分子分母同时下溢为 0/0 仍然危险，
    不过实践中 softmax 的实现会先减去行内最大值，此时整行变成全 0，
    Softmax 退化为**均匀分布**（每格 1/L）而不是 NaN —— 这才是关键区别。
    所以：-1e9 让「整行全屏蔽」这种边界情况安全退化成均匀分布，-inf 会直接产生 NaN。

【两类常见的 mask】
    1) causal mask（因果/自回归掩码）：下三角矩阵，保证位置 i 只能看到 ≤ i 的位置。
       用途：语言模型的 Decoder —— 预测第 i 个词时不能偷看第 i+1 个词。
    2) padding mask：一个 batch 里的序列长度不同，通常补齐到同一长度；
       补齐出来的位置是假数据，必须屏蔽掉，否则模型会把它当成真实 token。
"""
)

# ---------------------------------------------------------------------------
# 6.1 causal mask
# ---------------------------------------------------------------------------
print("【6.1 causal mask（下三角）：逐 token 的可见性表格】")

L = 3
causal_mask = torch.tril(torch.ones(L, L))       # 下三角为 1（含对角线），上三角为 0
tokens = ["I", "love", "deep learning"][:L]
print(f"    torch.tril(torch.ones({L},{L})) 的可见性矩阵（1 = 可见，0 = 屏蔽）：")
print("        " + "".join(f"{t:>16s}" for t in tokens))
for i in range(L):
    row = "".join(f"{('✓ 可见' if causal_mask[i, j] else '✗ 屏蔽'):>14s}" for j in range(L))
    print(f"    {tokens[i]:>12s}{row}")
print()
print("    逐 token 解读（课案的 Token 位置表）：")
print("        位置 1（I）           ：只能看自己（第 1 行只有 1 个可见）")
print("        位置 2（love）        ：能看 I 和自己（第 2 行有 2 个可见）")
print("        位置 3（deep learning）：能看前面所有词和自己（第 3 行 3 个全可见）")
print("    这就是「自回归」：生成第 i 个 token 时，只能依赖已经生成出来的前 i 个 token。")

# 实际验证加 mask 后上三角权重确实为 0
torch.manual_seed(42)
Qc = torch.randn(1, 1, L, 8)
Kc = torch.randn(1, 1, L, 8)
Vc = torch.randn(1, 1, L, 8)
_, w_no_mask = scaled_dot_product_attention(Qc, Kc, Vc)
out_c, w_causal = scaled_dot_product_attention(Qc, Kc, Vc, mask=causal_mask)

np.set_printoptions(precision=4, suppress=True)
print("\n    不加 mask 的注意力权重（上三角有值，说明「偷看」了未来）：")
print(f"        {w_no_mask[0, 0].detach().numpy()}")
print("    加 causal mask 后的注意力权重（上三角精确为 0）：")
print(f"        {w_causal[0, 0].detach().numpy()}")
upper = torch.triu(w_causal[0, 0], diagonal=1)
print(f"    上三角（不含对角线）的最大值 = {upper.max().item():.3e}"
      f"   ← 全部为 0，验证通过：{torch.allclose(upper, torch.zeros_like(upper), atol=1e-7)}")
print(f"    加 mask 后每行权重和（仍应为 1）：{w_causal[0, 0].sum(-1).tolist()}")
print("    第 1 行的权重：[1.0000, 0, 0] —— 位置 1 只能看自己，Softmax 就退化成 one-hot。")

# 边界情况：整行都被屏蔽
all_masked = torch.zeros(1, 1, L, L)             # 全 0 = 全部屏蔽
_, w_all_masked = scaled_dot_product_attention(Qc, Kc, Vc, mask=all_masked)
print(f"\n    边界情况：整行全屏蔽时的权重 = {w_all_masked[0, 0, 0].tolist()}")
print(f"    是否含 NaN？{torch.isnan(w_all_masked).any().item()}   ← 用 -1e9 不会产生 NaN")
print("    整行都被屏蔽时，Softmax 退化成均匀分布（每格 1/3），而不是 NaN。")
print("    如果用的是 -inf，这里就会出现 nan = 0/0，整个网络随即崩掉。")

# ---------------------------------------------------------------------------
# 6.2 padding mask
# ---------------------------------------------------------------------------
print("\n【6.2 padding mask：屏蔽补齐出来的假 token】")

# 造一批不等长的「句子」，用 padding 补齐到同一长度
pad_len = 6
lengths = [3, 5, 6]                       # 三个样本的真实长度
tokens_pad = [
    ["我", "喜欢", "深度学习", "<pad>", "<pad>", "<pad>"],
    ["今天", "天气", "很", "好", "<pad>"],
    ["注意力", "机制", "非常", "重要", "而且", "好用"],
]
torch.manual_seed(42)
X_pad = torch.randn(len(lengths), pad_len, 16)      # (batch=3, seq=6, d_model=16)

# 构造 padding mask：(batch, 1, 1, seq_k)，1=真实 token，0=padding
# 形状里的两个 1 是为了广播到 scores 的 (batch, heads, seq_q, seq_k)
key_padding_mask = torch.zeros(len(lengths), pad_len)
for i, ln in enumerate(lengths):
    key_padding_mask[i, :ln] = 1.0                   # 前 ln 个是真 token
key_padding_mask = key_padding_mask[:, None, None, :]   # → (batch, 1, 1, seq_k)

print(f"    三个样本的真实长度：{lengths}（统一补齐到 {pad_len}）")
print("    各样本的 token 序列：")
for i, toks in enumerate(tokens_pad):
    print(f"        {i}: {' | '.join(toks)}")
print(f"    key_padding_mask 形状：{shape(key_padding_mask)}  （第 2、3 维是 1，用于广播到 heads 和 seq_q）")
print(f"    第 0 个样本的 mask：{key_padding_mask[0, 0, 0].tolist()}   ← 后 3 个是 padding")

WQ2 = nn.Linear(16, 8, bias=False)
WK2 = nn.Linear(16, 8, bias=False)
WV2 = nn.Linear(16, 8, bias=False)
Xp = X_pad[:, None]                     # 加一个 heads 维： (batch, 1, seq, d_model)
Qp, Kp, Vp = WQ2(Xp), WK2(Xp), WV2(Xp)  # (batch, 1, seq, 8)

_, w_before = scaled_dot_product_attention(Qp, Kp, Vp)                            # 无掩码
_, w_after = scaled_dot_product_attention(Qp, Kp, Vp, mask=key_padding_mask)      # 加 padding mask

print("\n    第 0 个样本（真实长度 3）在第 0 个 Query 上的注意力权重：")
print(f"        屏蔽前：{w_before[0, 0, 0].detach().numpy()}")
print(f"        屏蔽后：{w_after[0, 0, 0].detach().numpy()}")
print("        ↑ 屏蔽前 padding 位置（后 3 个）还分走了不少权重；")
print("          屏蔽后它们全部变成 0，注意力权重被「重新分配」给真实 token（和仍为 1）。")

pad_positions = (key_padding_mask[0, 0, 0] == 0)
leaked = w_after[0, 0, :, pad_positions].abs().max().item()
print(f"\n    验证：所有 Query 行在 padding 位置上的权重最大值 = {leaked:.3e}")
print(f"    是否全为 0？{leaked < 1e-7}")
print(f"    屏蔽后每行的权重和：{w_after[0, 0].sum(-1).tolist()}   ← 依然都是 1")

# ===========================================================================
# 七、多头注意力：拆分形状
# ===========================================================================
section("七、多头：split_heads 的形状变化与 contiguous")

print(
    r"""
【为什么要多头】
    一组 Q/K/V 只能学到「一种」相关性模式（比如只关注语法主谓关系）。
    多头注意力把 d_model 维**切分成 num_heads 份**，每份独立做一次注意力，
    相当于让模型同时从多个「子空间」观察序列（一个头管语法、一个头管指代、……），
    最后把各头的结果拼回去再投影。这样表达能力更强，而总计算量几乎不变
    （因为每个头的维度从 d_model 降到了 d_k = d_model / num_heads）。

【形状变换的完整链路】
    输入 X:                  (batch, seq_len, d_model)
    ① 线性投影得到 Q/K/V:     (batch, seq_len, d_model)
    ② split_heads 第一步 —— view 拆出 heads 维:
                             (batch, seq_len, num_heads, d_k)
       含义：把最后的 d_model 维按「先 head 后 d_k」的顺序切成两维。
    ③ split_heads 第二步 —— transpose 把 heads 换到前面:
                             (batch, num_heads, seq_len, d_k)
       为什么必须换？因为注意力是在**每个头内部**独立做的，
       我们希望最后两维是 (seq, d_k)，前面的 (batch, num_heads) 只是「并行批次」，
       这样 matmul / softmax 才能直接作用在最后两维上（广播语义）。
    ④ 做注意力:              (batch, num_heads, seq_len, d_k)
    ⑤ 合并 —— transpose(1,2) 换回来:
                             (batch, seq_len, num_heads, d_k)
    ⑥ 合并 —— contiguous().view():
                             (batch, seq_len, d_model)

【为什么合并时必须 contiguous()？】
    tensor.transpose() 只改变**步长（stride）**，不搬运数据，返回的是一个「视图」。
    原来的 (b, h, s, d) 在内存里是连续的，转成 (b, s, h, d) 之后，
    要按 (b, s, h, d) 的顺序遍历元素，步长就不再是「从后往前递减」的规则形状了，
    这种张量叫 non-contiguous（内存不连续）。
    view() 要求张量在内存里是连续的（它只重新解释形状，不搬数据），
    所以对 non-contiguous 的张量调用 view() 会被直接拒绝，PyTorch 的提示大意是：
        view size is not compatible with input tensor's size and stride
        （意思就是「你要的新形状和这个张量的步长对不上」）
    解决办法：.contiguous() 会按**当前逻辑顺序**把元素真正拷贝成一份连续内存，之后再 view()。

    下面会用 is_contiguous() 打印 False → True 来证明这一点，并演示一个
    ⚠ 极其危险的陷阱：如果**漏掉了 swap 轴的 transpose(1,2)** 就直接
    contiguous().view(...)，程序不会报错、形状也对，但数据被静默打乱了。
    这类 bug 排查起来非常痛苦，所以顺序一定要记牢：
        split_heads :  view → transpose(1, 2)
        combine_heads: transpose(1, 2) → contiguous() → view
    另一个选择是用 reshape()，它等价于「需要时自动 contiguous() 再 view()」，
    但显式写出 contiguous() 更能体现「这里发生了一次内存拷贝」。
"""
)


def split_heads(x: torch.Tensor, num_heads: int):
    """把 (batch, seq_len, d_model) 拆成 (batch, num_heads, seq_len, d_k)。

    两步：
        1) view  → (batch, seq_len, num_heads, d_k)   把 d_model 切成 heads × d_k
        2) transpose(1, 2) → (batch, num_heads, seq_len, d_k)   把 heads 提到前面
    """
    batch, seq_len, d_model = x.shape
    d_k = d_model // num_heads
    assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

    # 第 1 步：拆出 heads 维。d_model = num_heads * d_k，所以 .view 是合法的重解释
    x = x.view(batch, seq_len, num_heads, d_k)
    # 第 2 步：把 heads 换到 seq 前面，让最后两维是 (seq, d_k) 以便做注意力
    x = x.transpose(1, 2)
    return x


def combine_heads(x: torch.Tensor):
    """把 (batch, num_heads, seq_len, d_k) 合并回 (batch, seq_len, d_model)。

    transpose(1,2) 之后张量不再连续，必须先 contiguous() 才能 view()。
    """
    batch, num_heads, seq_len, d_k = x.shape
    # 换回 (batch, seq_len, num_heads, d_k)
    x = x.transpose(1, 2)
    # transpose 只改步长不搬数据 → 此时内存不连续，view 会报错，必须先 contiguous()
    x = x.contiguous()
    # 把 (num_heads, d_k) 两维合并回 d_model
    return x.view(batch, seq_len, num_heads * d_k)


torch.manual_seed(42)
batch, seq_len, d_model, num_heads = 2, 10, 128, 4
d_k = d_model // num_heads
x_mha = torch.randn(batch, seq_len, d_model)

print("【split_heads / combine_heads 逐步形状打印】")
print(f"    原始输入             : {shape(x_mha)}   # (batch, seq_len, d_model)")
h1 = x_mha.view(batch, seq_len, num_heads, d_k)
print(f"    ① view 拆 heads      : {shape(h1)}   # (batch, seq_len, num_heads, d_k)")
h2 = h1.transpose(1, 2)
print(f"    ② transpose(1,2)     : {shape(h2)}   # (batch, num_heads, seq_len, d_k)")
print(f"       → 与 split_heads() 的结果一致：{torch.equal(h2, split_heads(x_mha, num_heads))}")
print(f"    ③ is_contiguous()（transpose 之后，最后两维被换过）：{h2.is_contiguous()}")
print("       ↑ False！transpose 只改步长，数据在内存里还是原来的顺序。")

# 演示「不 contiguous 就 view 会报错」——用 try/except 捕获，保证脚本不会崩
try:
    _ = h2.view(batch, seq_len, d_model)        # 试图直接 view 回 d_model
    print("    ④ 直接 view 竟然成功了？")
except RuntimeError as exc:
    # 这里捕获到的就是 PyTorch 的「view size is not compatible with input tensor's size and stride」，
    # 我们只打印中文说明，绝不打印异常栈，也不打印异常类型名
    print("    ④ 直接对 non-contiguous 张量 view → 被拒绝（新形状与步长不兼容）。")
    print("       这就是为什么「跨步长改变形状」时必须先 contiguous()。")

h3 = h2.transpose(1, 2)                         # (batch, seq_len, num_heads, d_k)
print(f"    ⑤ transpose(1,2) 换回 : {shape(h3)}   # (batch, seq_len, num_heads, d_k)")
print(f"       is_contiguous() = {h3.is_contiguous()}"
      f"   ← 两次 transpose 把步长换回原样，所以又连续了")
h4 = h3.contiguous()                            # 显式声明「这里做一次内存整理」
back = h4.view(batch, seq_len, d_model)         # ⑥ 合并两个维度
print(f"    ⑥ contiguous().view() : {shape(back)}   # (batch, seq_len, d_model)")
print(f"       与原始输入完全相同：{torch.equal(back, x_mha)}"
      f"   ← 说明 transpose(1,2) → contiguous() → view() 这条链路的顺序是正确的")

# ⚠ 反面教材：少写一步 transpose，contiguous().view() 会「静默地打乱数据」！
# 原理：contiguous() 是按**当前逻辑顺序**把元素拷成连续内存的，
#       而 h2 的逻辑顺序是 (batch, heads, seq, d_k)。直接把它 view 成
#       (batch, seq, d_model) 时，内存里的排布仍是「先 heads 后 seq」，
#       于是拼出来的 (num_heads, d_k) 会被错当成 d_model 的前半段，
#       数据被重新排列 —— 结果 shape 看起来对，数值却错了，而且**不报错**。
_bad = h2.contiguous().view(batch, seq_len, d_model)
print(f"    反面教材：h2.contiguous().view(...) 形状 {shape(_bad)}（看起来正常）")
print(f"       与原始输入相同？{torch.equal(_bad, x_mha)}   ← False！数据被静默打乱了")
print("       这类 bug 最危险：不报错、形状对，只是结果不对。")
print("       结论：合并多头必须先 transpose(1,2) 把 heads 换回原位，")
print("             再用 contiguous().view(...) 合并 —— 两步的顺序不能颠倒、也不能省。")

# ===========================================================================
# 八、手写完整 MultiHeadAttention 模块
# ===========================================================================
section("八、手写 MultiHeadAttention 模块")


class MultiHeadAttention(nn.Module):
    """多头自注意力（手写实现，含完整注释）。

    结构：
        W_Q / W_K / W_V: nn.Linear(d_model, d_model)，把输入投影成三种角色的向量
        W_O            : nn.Linear(d_model, d_model)，把多头拼接后的结果再融合一次
        d_k = d_model // num_heads，每个头独立做 scaled dot-product attention
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        # 必须能整除，否则没法把 d_model 均分给每个头
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_Q = nn.Linear(d_model, d_model)   # 生成 Query 的投影
        self.W_K = nn.Linear(d_model, d_model)   # 生成 Key 的投影
        self.W_V = nn.Linear(d_model, d_model)   # 生成 Value 的投影
        self.W_O = nn.Linear(d_model, d_model)   # 多头结果融合（output projection）
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        """query: (batch, seq_q, d_model)，key/value: (batch, seq_k, d_model)。

        写成三个参数而不是单个 x，这样同一个模块既能做自注意力
        （传同一份张量三次），也能做交叉注意力（query 和 key/value 不同）。
        """
        batch = query.size(0)

        # ① 三组独立的线性投影：同一份输入 → 三种用途的向量
        Q = self.W_Q(query)      # (batch, seq_q, d_model)
        K = self.W_K(key)        # (batch, seq_k, d_model)
        V = self.W_V(value)      # (batch, seq_k, d_model)

        # ② 拆成多头： (batch, seq, d_model) → (batch, num_heads, seq, d_k)
        Q = split_heads(Q, self.num_heads)
        K = split_heads(K, self.num_heads)
        V = split_heads(V, self.num_heads)

        # ③ 每个头内部独立做缩放点积注意力
        #    scores: (batch, heads, seq_q, seq_k)，output: (batch, heads, seq_q, d_k)
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask=mask)

        # ④ 合并多头： (batch, heads, seq, d_k) → (batch, seq, d_model)
        attn_output = combine_heads(attn_output)
        attn_output = self.dropout(attn_output)

        # ⑤ 输出投影：让不同头的信息相互融合（否则各头只是简单拼接）
        output = self.W_O(attn_output)
        return output, attn_weights


# ---- 用较小的尺寸（d_model=128, heads=4）演示，保证 CPU 上很快 ----
# 说明：Transformer 论文的原始配置是 d_model=512、num_heads=8。
#       这里为了速度改成 128/4（d_k 仍是 32 不变，形状规律完全一样），
#       下面同时对 512/8 也跑一次以确认原始配置同样正确。
print("【演示配置 A：d_model=128, num_heads=4（为速度选用）】")
torch.manual_seed(42)
mha = MultiHeadAttention(d_model=128, num_heads=4)
x_demo2 = torch.randn(2, 10, 128)
out_mha, w_mha = mha(x_demo2, x_demo2, x_demo2)     # 自注意力：三个参数都传 x
print(f"    x          : {shape(x_demo2)}")
print(f"    output     : {shape(out_mha)}   ← 与输入同形状，可以层层堆叠")
print(f"    weights    : {shape(w_mha)}     # (batch, num_heads, seq_q, seq_k)")
print(f"    d_k = d_model / num_heads = 128 / 4 = {mha.d_k}")
print(f"    每个头的权重行和都为 1：{torch.allclose(w_mha.sum(-1), torch.ones_like(w_mha.sum(-1)))}")
print(f"    四个头的注意力分布各不相同（说明各头学到了不同的模式）：")
for h in range(4):
    print(f"        头 {h} 的第 0 行：{w_mha[0, h, 0].detach().numpy()}")

print("\n【演示配置 B：Transformer 论文原始配置 d_model=512, num_heads=8】")
torch.manual_seed(42)
mha_512 = MultiHeadAttention(d_model=512, num_heads=8)
x_512 = torch.randn(2, 10, 512)
out_512, w_512 = mha_512(x_512, x_512, x_512)
print(f"    x          : {shape(x_512)}")
print(f"    output     : {shape(out_512)}")
print(f"    weights    : {shape(w_512)}")
print(f"    d_k = 512 / 8 = {mha_512.d_k}（与配置 A 的 d_k=32 相同，这正是多头设计的巧妙之处：")
print("         头数变多、总维度变大，但每个头的维度保持不变，单头计算量不随 d_model 增长）")
print(f"    assert d_model % num_heads == 0 的作用：d_model=512、heads=8 时 512%8={512 % 8}，通过")

# ---- 交叉注意力：同一个模块，query 与 key/value 长度不同 ----
print("\n【同一个模块做交叉注意力：seq_q ≠ seq_k】")
torch.manual_seed(42)
q_cross = torch.randn(2, 6, 128)      # 目标序列长度 6
kv_cross = torch.randn(2, 9, 128)     # 源序列长度 9
out_cross, w_cross = mha(q_cross, kv_cross, kv_cross)
print(f"    query {shape(q_cross)}   key/value {shape(kv_cross)}")
print(f"    output  {shape(out_cross)}   ← 第一维（序列长度）跟随 query")
print(f"    weights {shape(w_cross)}     ← (seq_q=6, seq_k=9)，非方阵")
print("    → 这就是机器翻译 Decoder 里 Cross Attention 的形状：用 6 个目标位置去查 9 个源位置。")

# ---- 因果掩码配合多头注意力 ----
print("\n【多头注意力 + causal mask】")
L_c = 10
causal = torch.tril(torch.ones(L_c, L_c))
_, w_causal_mha = mha(x_demo2, x_demo2, x_demo2, mask=causal)
upper_mha = torch.triu(w_causal_mha[0, 0], diagonal=1)
print(f"    加因果掩码后，第 0 个头权重的上三角最大值 = {upper_mha.max().item():.3e}"
      f"   ← 为 0：{upper_mha.max().item() == 0.0}")
print("    每个头、每个 Query 都只能看到当前及之前的位置。")

# ===========================================================================
# 九、注意力权重可视化
# ===========================================================================
section("九、注意力权重与因果掩码可视化")

print("【图 1】句子的自注意力热力图；【图 2】causal mask 的可见性矩阵。")

# 造一个长度为 8 的「句子」，用自注意力算出权重矩阵
torch.manual_seed(42)
seq_viz = 8
words = ["我", "喜欢", "深度", "学习", "因为", "它", "很", "有趣"]
x_viz = torch.randn(1, seq_viz, 128)
mha_viz = MultiHeadAttention(d_model=128, num_heads=4)
with torch.no_grad():
    _, w_viz = mha_viz(x_viz, x_viz, x_viz)
# 把 4 个头平均，得到「整体注意力」；形状 (seq, seq)
attn_avg = w_viz[0].mean(dim=0).detach().numpy()

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 子图 1：注意力热力图
ax = axes[0]
im = ax.imshow(attn_avg, cmap="Blues", vmin=0, vmax=attn_avg.max())
ax.set_xticks(range(seq_viz))
ax.set_yticks(range(seq_viz))
ax.set_xticklabels(words, fontsize=10)
ax.set_yticklabels(words, fontsize=10)
ax.set_xlabel("Key 位置（被注意的词）", fontsize=11)
ax.set_ylabel("Query 位置（正在处理的词）", fontsize=11)
ax.set_title("自注意力权重热力图（4 个头取平均）\n行 = Query，列 = Key", fontsize=12)
# 在每个格子里标数值，方便对照 print 的输出
for i in range(seq_viz):
    for j in range(seq_viz):
        ax.text(j, i, f"{attn_avg[i, j]:.2f}", ha="center", va="center",
                fontsize=7, color="white" if attn_avg[i, j] > attn_avg.max() * 0.6 else "black")
fig.colorbar(im, ax=ax, fraction=0.046, label="注意力权重（每行和为 1）")

# 子图 2：causal mask 可见性矩阵
ax = axes[1]
L_vis = 8
causal_vis = torch.tril(torch.ones(L_vis, L_vis)).numpy()
im2 = ax.imshow(causal_vis, cmap="Greens", vmin=0, vmax=1)
ax.set_xticks(range(L_vis))
ax.set_yticks(range(L_vis))
ax.set_xticklabels(words, fontsize=10)
ax.set_yticklabels(words, fontsize=10)
ax.set_xlabel("Key 位置（能否被看到）", fontsize=11)
ax.set_ylabel("Query 位置", fontsize=11)
ax.set_title("Causal Mask 可见性矩阵（下三角）\n可 = 可见，不 = 屏蔽", fontsize=12)
for i in range(L_vis):
    for j in range(L_vis):
        # 注意：微软雅黑没有 ✓/✗ 这两个字形，画到图上会变成方框，
        # 所以这里改用中文字「可 / 不」来表示可见性。
        mark = "可" if causal_vis[i, j] else "不"
        ax.text(j, i, mark, ha="center", va="center", fontsize=13,
                color="white" if causal_vis[i, j] else "#b03030")
fig.colorbar(im2, ax=ax, fraction=0.046, ticks=[0, 1], label="可见性（1=可见，0=屏蔽）")

fig.tight_layout()
_attn_png = OUTPUT_DIR / "02网络架构_05_注意力权重与因果掩码.png"
fig.savefig(_attn_png, dpi=110)
plt.close(fig)
print(f"    图片已保存：{_attn_png}")
print("    读图方法：热力图中颜色越深表示注意力权重越大；")
print("              因果掩码图中位置 i 的行只能看到 ≤ i 的列（下三角标「可」），")
print("              上三角标「不」表示被屏蔽。")

# ===========================================================================
# 小结
# ===========================================================================
section("小结")

print(
    f"""
    1. Attention(Q,K,V) = softmax(Q Kᵀ / √d_k) · V，三步：算相似度 → 缩放归一化 → 加权求和。
    2. 点积方差 = d_k，标准差 = √d_k；不缩放时 d_k=512 的 std ≈ 22.6，
       Softmax 趋于 one-hot、梯度变小；除以 √d_k 把方差拉回 1。
    3. Cross-Attention 的 Q 与 K/V 来自不同序列（seq_q ≠ seq_k）；
       Self-Attention 的 Q/K/V 来自同一序列（权重是 seq×seq 的方阵）。
    4. 自注意力用 W_Q / W_K / W_V 三组不同线性变换把同一份 X 投影成三种角色的向量。
    5. 手写 scaled_dot_product_attention 支持 (batch, heads, seq, d_k)；
       Q,K,V = randn(2,4,5,8) 时 output {shape(output)}、weights {shape(weights)}，
       每行权重和为 1。
    6. mask 用 masked_fill(mask == 0, -1e9)：causal mask（下三角）让位置 i 只看到 ≤ i，
       可验证上三角权重精确为 0；padding mask 让补齐位置权重为 0；
       -1e9 而非 -inf，使「整行全屏蔽」退化成均匀分布而不是 NaN。
    7. 多头：split_heads 走 view → transpose(1,2)；合并走 transpose(1,2) → contiguous() → view()。
       transpose 后 is_contiguous() 为 False，view 会报错，contiguous() 后为 True。
    8. 手写 MultiHeadAttention：W_Q/W_K/W_V/W_O 四个 nn.Linear，d_k = d_model // num_heads；
       d_model=128/heads=4 与 d_model=512/heads=8 两种配置都验证通过。
    9. 本脚本实测总耗时 {time.perf_counter() - _T_START:.2f} 秒，图片输出到：
       {OUTPUT_DIR}
"""
)

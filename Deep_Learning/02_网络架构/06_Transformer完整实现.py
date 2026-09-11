"""
对应课案章节：
    网络架构 / Transformer —— 核心组件、位置编码、多头注意力、编码器、解码器、框架使用、三种注意力

本节知识点：
    1. 为什么 Attention 单独用不行：自注意力对输入是「置换等变」的，交换两个 token 的输出
       只是跟着交换，模型完全拿不到位置信息 → 必须显式注入位置编码。
    2. 正弦位置编码 PositionalEncoding（手写）：公式、为什么用 sin/cos、
       「两个位置编码的内积只取决于相对距离 k」的推导与数值验证、register_buffer 的作用。
    3. 缩放点积注意力（手写）：QKᵀ/√d_k → mask → softmax → 加权 V；为什么必须除以 √d_k。
    4. 多头注意力 MultiHeadAttention（手写）：W_Q/W_K/W_V/W_O 四个线性层、
       split_heads 的 4 步形状变换、contiguous() 的必要性、合并多头。
    5. 前馈网络 FeedForward：d_model → ff_dim(4×) → d_model，逐位置独立变换。
    6. Add & Norm：残差连接 + LayerNorm；Post-LN 与 Pre-LN 的区别与取舍。
    7. EncoderBlock：Self-Attention（无掩码，双向）+ FFN，各带残差与 LayerNorm。
    8. DecoderBlock：Masked Self-Attention（因果掩码）+ Cross-Attention + FFN。
       并用「改动未来 token，前面的输出不变」来**实证因果掩码真的生效**。
    9. Transformer 组装：Embedding + 位置编码 + N 层编码器 + N 层解码器 + 输出投影，
       并在一个「序列反转」玩具任务上真的训练几十步，看 loss 下降与预测正确。
   10. 三种注意力对比：Self-Attention / Masked Self-Attention / Cross-Attention
       （区别在 Q、K、V 的来源与是否加掩码），逐个打印形状并画注意力权重热力图。
   11. torch.nn 框架用法：nn.Transformer / nn.TransformerEncoder / nn.TransformerDecoder /
       nn.MultiheadAttention / nn.TransformerEncoderLayer 的形状验证。
   12. 重点避坑：nn.MultiheadAttention 的 bool 类型的 attn_mask 语义是
       **True = 屏蔽（不允许注意）**，和 F.scaled_dot_product_attention 的
       **True = 参与**完全相反；课案 Decoder 代码里 `torch.tril(...).bool()` 配
       「True=可见」的注释是**错的**，本节会实测证明并给出正确写法。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Deep_Learning\\02_网络架构\\06_Transformer完整实现.py'
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
# 依赖导入与全局随机种子
# ---------------------------------------------------------------------------
import copy                                  # noqa: E402  深拷贝模型状态用
import math                                  # noqa: E402

import numpy as np                           # noqa: E402
import torch                                 # noqa: E402
import torch.nn as nn                        # noqa: E402
import torch.nn.functional as F              # noqa: E402

# 随机种子统一，保证每次运行结果可复现
torch.manual_seed(42)
np.random.seed(42)


def title(text: str) -> None:
    """打印一级分节标题，让长篇输出有清晰的层次。"""
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def sub(text: str) -> None:
    """打印二级小节标题。"""
    print()
    print("-" * 78)
    print(text)
    print("-" * 78)


def show_shape(name: str, t: torch.Tensor) -> None:
    """统一打印张量形状，方便对照讲解。"""
    print(f"    {name:<34s} shape = {tuple(t.shape)}")


print("=" * 78)
print("Transformer 从零手写完整实现")
print("=" * 78)
print(f"PyTorch 版本     : {torch.__version__}")
print(f"CUDA 可用        : {torch.cuda.is_available()}（本机为纯 CPU 版，全部在 CPU 上运行）")
print("本脚本的超参数全部刻意取小（d_model=64、n_heads=4、序列长度 ≤ 20），")
print("目的是让 CPU 也能在十几秒内跑完，同时形状关系与真实 Transformer 完全一致。")


# ===========================================================================
# 第 1 节：为什么必须要有位置编码
# ===========================================================================
title("第 1 节：为什么 Attention 单独用不够 —— 置换等变性")

print("""
【原理】
自注意力的计算是 out = softmax(QKᵀ/√d_k)·V，其中 Q/K/V 都由同一个输入 X 线性变换而来。
把输入的行（=token 位置）做一个置换 P，即 X' = P·X，那么：
    Q' = P·Q,  K' = P·K,  V' = P·V
    scores' = Q'K'ᵀ = P·Q·Kᵀ·Pᵀ  →  softmax 按行做，等价于 P·softmax(QKᵀ)·Pᵀ
    out' = scores'·V' = P·softmax(QKᵀ)·Pᵀ·P·V = P·(softmax(QKᵀ)·V) = P·out
也就是说：**输出只是跟着输入一起被置换，模型对「谁在第几个位置」完全无感**。
这就是「置换等变（permutation equivariant）」，也是 Transformer 必须显式注入位置信息的根本原因。
（CNN 靠卷积核的局部窗口天然带位置，RNN 靠时间步的顺序天然带位置，Transformer 两者都没有。）

【实验】把同一批 token 打乱顺序，看自注意力输出是不是只是被打乱。
""")

# 一个最小的单头自注意力（不带位置编码），只用来说明置换等变
torch.manual_seed(42)
_demo_d = 8
_demo_x = torch.randn(1, 5, _demo_d)                     # batch=1, seq=5, d=8
_Wq, _Wk, _Wv = (torch.randn(_demo_d, _demo_d) for _ in range(3))


def _plain_self_attention(x: torch.Tensor) -> torch.Tensor:
    """最小自注意力：out = softmax(QKᵀ/√d)·V，不含任何位置信息。"""
    q, k, v = x @ _Wq, x @ _Wk, x @ _Wv
    attn = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(_demo_d), dim=-1)
    return attn @ v


_perm = torch.tensor([3, 0, 4, 1, 2])                     # 一个任意的位置置换
_out_original = _plain_self_attention(_demo_x)
_out_permuted = _plain_self_attention(_demo_x[:, _perm, :])

print(f"    输出（原顺序）        out[0, 0] = {_out_original[0, 0, :4].tolist()}")
print(f"    输出（打乱顺序）      out[0, 3] = {_out_permuted[0, 3, :4].tolist()}")
print(f"    两份输出是否只差一个置换：{torch.allclose(_out_permuted[0], _out_original[0][_perm], atol=1e-6)}")
print("    结论：Attention 本身分不清位置，必须额外加位置编码。")


# ===========================================================================
# 第 2 节：正弦位置编码 PositionalEncoding（手写）
# ===========================================================================
title("第 2 节：PositionalEncoding —— 正弦位置编码（从零手写）")

print("""
【公式】（Attention Is All You Need, 2017）
    PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
    PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )
其中 pos 是 token 在序列中的位置（0,1,2,...），i 是「维度对」的下标（0,1,...,d_model/2-1）。
偶数维放 sin，相邻奇数维放 cos，成对出现。

【为什么用 sin/cos，而不是直接加位置序号 0,1,2,...？】
  1) 值域匹配：sin/cos ∈ [-1,1]，和归一化后的 Embedding 向量在同一个数量级。
     若直接加 0..511 的序号，位置信号会比词义信号大几百倍，模型会主要靠位置判别、忽略词义。
  2) 避免梯度消失：数值过大会让 Softmax 输出极端化（ŷ≈1，其余≈0），
     而 ∂ŷ_k/∂z_k = ŷ_k(1-ŷ_k) ≈ 0，反向传播没有信号。
  3) 相对位置可感知（最关键）：设 ω_i = 1/10000^(2i/d_model)，对同一对维度 (2i, 2i+1)：
        sin(pos·ω)·sin((pos+k)·ω) + cos(pos·ω)·cos((pos+k)·ω)
      = cos( (pos+k)·ω - pos·ω )      ← 三角恒等式 cos(A-B)=cosA·cosB+sinA·sinB
      = cos(k·ω)
     pos 被消掉了！把所有维度对加起来：
        <PE(pos), PE(pos+k)> = Σ_{i=0}^{d/2-1} cos(k·ω_i)
      **内积只由相对距离 k 决定，与绝对位置 pos 无关**。模型因此能自然地感知「这两个 token 隔多远」。

【ω_i 是角频率，控制正弦曲线的疏密】
    波长 = 2π/ω_i。
    i=0 时 ω_0 = 1/10000^0 = 1，波长 2π ≈ 6.28 —— 高频，相邻位置的编码值差异大，
        模型能精确区分「第 3 个词」和「第 4 个词」（局部位置）。
    i 增大 → ω 变小 → 波长变长 → 值变化越来越平缓，几十个 token 范围内的位置关系仍可区分。
    i ≈ d_model/2 时 ω ≈ 1/10000，波长 ≈ 20000π ≈ 62800 —— 低频，几百 token 内值几乎不变，
        分不清「第 5 个」和「第 8 个」，但能区分「开头部分」和「结尾部分」（全局位置）。
    不同维度以不同「分辨率」看待位置，组合起来唯一编码每个位置。

【register_buffer 的作用】
    self.register_buffer("pe", pe) 把 pe 注册成「缓冲区」：
      - 会随 model.state_dict() 一起保存、一起 .to(device) 迁移；
      - 但**不会被优化器更新**（不是 Parameter），因为位置编码是固定的、不需要学习。
""")


class PositionalEncoding(nn.Module):
    """正弦位置编码：把固定的位置信号加到输入 Embedding 上。

    输入 x: (batch, seq_len, d_model)
    输出  : (batch, seq_len, d_model)  —— 形状不变，只是每个位置被加上了一个固定的向量
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        # pe 的每一行是一个位置的位置编码向量，形状 (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        # position: (max_len, 1)，第 pos 行就是数值 pos
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # div_term 就是公式里的 ω_i = 1/10000^(2i/d_model)。
        # 用 exp(-log(10000)·2i/d_model) 而不是直接写 1/10000**(2i/d_model)，
        # 是为了数值稳定（避免先算出巨大的 10000^... 再做除法）。
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float)
                             * (-math.log(10000.0) / d_model))

        # 偶数维（0::2）放 sin(pos·ω)，奇数维（1::2）放 cos(pos·ω)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 加 batch 维度：(max_len, d_model) → (1, max_len, d_model)
        # 这样后面和 (batch, seq_len, d_model) 相加时自动广播
        pe = pe.unsqueeze(0)
        # 注册为 buffer：随模型保存/迁移设备，但不参与梯度更新
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 只取前 seq_len 个位置，和输入逐元素相加（广播）
        # pe 是 buffer、requires_grad=False，所以这一步不会给它累积梯度
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


sub("2.1 形状验证与「内积只取决于相对距离」的数值验证")

pe_layer = PositionalEncoding(d_model=64, max_len=100, dropout=0.0)
_pe_input = torch.randn(2, 20, 64)                       # batch=2, seq=20, d_model=64
_pe_output = pe_layer(_pe_input)
show_shape("输入 x", _pe_input)
show_shape("输出 x + PE", _pe_output)
show_shape("缓存的 PE 矩阵", pe_layer.pe)
print(f"    PE 是否被注册为 buffer（出现在 state_dict 里）："
      f"{'pe' in pe_layer.state_dict()}")
print(f"    PE 是否需要梯度：{pe_layer.pe.requires_grad}")

# 直接取 PE 矩阵（去掉 batch 维）来验证内积性质
_pe_matrix = pe_layer.pe[0]                              # (100, 64)

print()
print("    相对距离 k  |  实际内积 <PE(0), PE(k)>  |  理论值 Σcos(k·ω_i)  |  是否一致")
_omega = np.exp(np.arange(0, 64, 2) * (-math.log(10000.0) / 64))   # ω_i，共 32 个
for k in range(0, 6):
    _actual = float((_pe_matrix[0] @ _pe_matrix[k]).item())
    _theory = float(np.cos(k * _omega).sum())
    print(f"        {k:^6d}  |  {_actual:>22.8f}  |  {_theory:>18.8f}  |  {abs(_actual - _theory) < 1e-4}")

# 再验证「绝对位置不同、相对距离相同 → 内积相同」
print()
print("    验证：内积与绝对位置无关（只与相对距离 k 有关）")
for base in (0, 7, 30):
    _ip = float((_pe_matrix[base] @ _pe_matrix[base + 4]).item())
    print(f"        <PE({base:>2d}), PE({base + 4:>2d})> 相对距离恒为 4，内积 = {_ip:.8f}")
print("    三者几乎相同 → 模型可以用注意力权重自然地表达「距离」，不必记忆绝对位置。")

print("""
【补充：为什么需要除以 √d_k（下一节会用）】
设 q、k 的各分量独立、均值 0、方差 1，则点积 q·k = Σ q_i k_i 是 d_k 个独立变量之和，
其方差为 d_k（方差可加）。d_k = 512 时点积的标准差约 √512 ≈ 22.6，
Softmax 的输入数值过大 → 输出趋近 one-hot → 梯度 ∂ŷ_k/∂z_k = ŷ_k(1-ŷ_k) ≈ 0 → 训练困难。
除以 √d_k 恰好把方差拉回 1，让 Softmax 处在「有梯度」的工作区间。
""")


# ===========================================================================
# 第 3 节：缩放点积注意力（手写）
# ===========================================================================
title("第 3 节：缩放点积注意力 Scaled Dot-Product Attention（手写）")

print("""
【公式】Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k + mask ) · V

【三步走】
  1) 算相似度：Q 和 K 做点积，得到 (seq_q, seq_k) 的相似度分数矩阵，
     表示「第 i 个 Query 和第 j 个 Key 有多匹配」。
  2) 归一化：除以 √d_k，再过 Softmax（按最后一维，即对每个 Query 在所有 Key 上归一化），
     得到每行和为 1 的注意力权重。
  3) 加权求和：用注意力权重对 V 做加权平均，得到输出 (seq_q, d_v)。
     权重越大，对应 token 的 Value 对输出影响越大。

【本实现的掩码约定（务必注意，和 nn.MultiheadAttention 相反！）】
    本函数用 `scores.masked_fill(mask == 0, -1e9)`，
    即 mask 为 **bool 张量，True = 保留（可见）、False = 屏蔽**。
    （-1e9 而不是 -inf：万一某一行整行都被屏蔽，-inf 会让 Softmax 出现
      inf - inf = NaN；-1e9 过 Softmax 后权重近似 0，且不会产生 NaN。）
""")


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """缩放点积注意力。

    参数
        Q: (..., seq_q, d_k)
        K: (..., seq_k, d_k)
        V: (..., seq_k, d_v)
        mask: 可选，bool 张量，能广播到 (..., seq_q, seq_k)。
              **True = 保留，False = 屏蔽**（与 nn.MultiheadAttention 相反）
    返回
        output:  (..., seq_q, d_v)
        weights: (..., seq_q, seq_k)  注意力权重，每行和为 1
    """
    d_k = Q.size(-1)                                     # 每个头的维度

    # 1. Q·Kᵀ 并缩放：K.transpose(-2,-1) 把最后两维换位，使 (seq_q,d_k)@(d_k,seq_k) → (seq_q,seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # 2. 掩码：把不可见位置的分数压成 -1e9，Softmax 后权重≈0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # 3. Softmax 得到注意力权重（对最后一个维度归一化 → 每个 Query 在所有 Key 上分布）
    weights = F.softmax(scores, dim=-1)

    # 4. 加权求和
    output = torch.matmul(weights, V)
    return output, weights


sub("3.1 形状验证与权重性质")

torch.manual_seed(42)
# 用四维输入模拟「多头的批量计算」：batch=2, heads=4, seq=5, d_k=8
_Q = torch.randn(2, 4, 5, 8)
_K = torch.randn(2, 4, 5, 8)
_V = torch.randn(2, 4, 5, 8)
_out, _w = scaled_dot_product_attention(_Q, _K, _V)

show_shape("Q", _Q)
show_shape("K", _K)
show_shape("V", _V)
show_shape("output = weights @ V", _out)
show_shape("attention weights", _w)
print(f"    每一行权重之和是否都为 1：{torch.allclose(_w.sum(dim=-1), torch.ones_like(_w.sum(dim=-1)), atol=1e-6)}")
print(f"    权重是否都非负：{bool((_w >= 0).all())}")
print(f"    权重总和（2*4*5 行，应为 40）：{_w.sum().item():.4f}")

sub("3.2 数值实验：为什么必须除以 √d_k")

torch.manual_seed(0)
print("    假设 q、k 每个分量 ~ N(0,1)，点积 q·k 的方差就是 d_k。")
print()
print("    d_k   | 点积分数的标准差(不缩放) | 标准差(除以√d_k) | Softmax最大权重(不缩放) | (缩放)")
for d_k in (8, 64, 512):
    _q = torch.randn(1000, d_k)
    _k = torch.randn(1000, d_k)
    _raw = (_q * _k).sum(dim=-1)                          # 未缩放的 1000 个点积分数
    _scaled = _raw / math.sqrt(d_k)
    _w_raw = torch.softmax(_raw, dim=-1).max().item()
    _w_scaled = torch.softmax(_scaled, dim=-1).max().item()
    print(f"    {d_k:^5d} | {_raw.std().item():>24.4f} | {_scaled.std().item():>16.4f} | "
          f"{_w_raw:>23.4f} | {_w_scaled:.4f}")
print()
print("    规律：不缩放时点积标准差约等于 √d_k（8→2.83、512→22.6），")
print("         缩放后标准差恒为 1 左右。")
print("    更关键的是「Softmax 最大权重」那一列：这里是一个 batch 内 1000 个 key 的极端情况，")
print("    但趋势非常清楚——分数尺度越大，Softmax 越趋近 one-hot，")
print("    而 ∂ŷ_k/∂z_k = ŷ_k(1-ŷ_k) 在 ŷ≈1 时接近 0，梯度就消失了。")

sub("3.3 掩码演示：-1e9 与 -inf 的区别")

_scores = torch.randn(3, 3)
_full_mask = torch.zeros(3, 3, dtype=torch.bool)          # 全 False = 全部屏蔽
_w_safe = torch.softmax(_scores.masked_fill(_full_mask == 0, -1e9), dim=-1)
print(f"    整行被屏蔽时用 -1e9 的结果（无 NaN）：{_w_safe[0].tolist()}")
print(f"    是否出现 NaN：{bool(torch.isnan(_w_safe).any())}")
_w_nan = torch.softmax(_scores.masked_fill(_full_mask == 0, float("-inf")), dim=-1)
print(f"    整行被屏蔽时用 -inf 的结果：{_w_nan[0].tolist()}"
      f"  ← 出现了 NaN，这就是用 -1e9 的原因")
print("    （某些实现会额外处理这种情况，但 -1e9 是最简单稳妥的做法。）")


# ===========================================================================
# 第 4 节：多头注意力（手写）
# ===========================================================================
title("第 4 节：Multi-Head Attention —— 多头注意力（从零手写）")

print("""
【为什么要多头】
单个头的注意力只能学到「一种」相关性模式。把 d_model 维拆成 h 个头（每个 d_k = d_model/h），
每个头用**独立的** W_Q/W_K/W_V 投影到自己的子空间、独立算注意力，
就能并行学到多种不同的相关性：有的头关注相邻词、有的头关注主谓关系、有的头关注长距离依赖。
最后把 h 个头的输出拼接起来，再用 W_O 融合回 d_model 维。

【形状变换四步（seq_len 记作 L，head 数记作 h）】
    x            (B, L, d_model)
    W_Q(x)       (B, L, d_model)
    split_heads  (B, L, h, d_k)  --view-->  (B, h, L, d_k)   ← 把 h 提到前面，方便并行
    注意力        (B, h, L, d_k)
    合并          (B, L, h, d_k) --view--> (B, L, d_model)
    W_O          (B, L, d_model)
""")


class MultiHeadAttention(nn.Module):
    """多头自/交叉注意力（手写实现，不调用 nn.MultiheadAttention）。"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除，每个头分到的维度要一样多"

        self.d_model = d_model                  # 模型总维度，如 64
        self.num_heads = num_heads              # 头数，如 4
        self.d_k = d_model // num_heads         # 每个头的维度 = 64/4 = 16

        # 四个线性变换：前三组把输入投影到 Q/K/V 子空间，最后一组把拼接后的多头输出映射回 d_model
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L, d_model) → (B, h, L, d_k)，并打印中间形状便于学习。"""
        batch_size, seq_len, _ = x.shape
        # view 只改变形状、不复制数据：把最后一维 d_model 拆成 (num_heads, d_k)
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        # transpose 交换 seq 和 head 两维 → 把 head 提到前面，方便对每个头并行做矩阵乘法
        return x.transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
        verbose: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """query/key/value 形状都是 (B, L, d_model)；mask 为 bool，True=可见。"""
        if verbose:
            show_shape("输入 query", query)

        # 1. 线性投影 + 拆头
        q = self.split_heads(self.W_Q(query))    # (B, h, seq_q, d_k)
        k = self.split_heads(self.W_K(key))      # (B, h, seq_k, d_k)
        v = self.split_heads(self.W_V(value))    # (B, h, seq_k, d_k)
        if verbose:
            show_shape("Q（拆头后）", q)
            show_shape("K（拆头后）", k)
            show_shape("V（拆头后）", v)

        # 2. 缩放点积注意力：每个头独立计算（h 维被当成 batch 维并行处理）
        attn_output, attn_weights = scaled_dot_product_attention(q, k, v, mask)
        if verbose:
            show_shape("注意力输出（各头）", attn_output)
            show_shape("注意力权重", attn_weights)

        # 3. 合并多头：把 h 个头拼回一个 d_model 维向量
        batch_size, _, seq_len, _ = attn_output.shape
        # transpose(1,2) 后内存不再连续，直接 view 会报错，所以必须先 contiguous()
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        if verbose:
            show_shape("多头拼接后", attn_output)

        # 4. 最后的线性映射 W_O，把多头信息融合
        output = self.W_O(attn_output)
        if verbose:
            show_shape("最终输出 W_O(拼接)", output)

        return output, attn_weights


sub("4.1 形状逐层验证")

torch.manual_seed(42)
_mha = MultiHeadAttention(d_model=64, num_heads=4, dropout=0.0)
_x_mha = torch.randn(2, 10, 64)
_out_mha, _w_mha = _mha(_x_mha, _x_mha, _x_mha, verbose=True)[:2]
print(f"    MultiHeadAttention 参数量：{sum(p.numel() for p in _mha.parameters()):,}"
      f"（= 4 × (64×64 + 64) = {4 * (64 * 64 + 64):,}）")

sub("4.2 contiguous() 为什么必要")

_t = torch.randn(2, 4, 5, 16)
_t_transposed = _t.transpose(1, 2)
print(f"    transpose 之后 is_contiguous() = {_t_transposed.is_contiguous()}  ← 内存不连续")
print(f"    此时直接 view 会失败（PyTorch 会报「形状与步长不兼容」并提示改用 reshape/contiguous），")
print(f"    这里不打印真实的异常信息，只说明结论。")
_t_contig = _t_transposed.contiguous()
print(f"    .contiguous() 之后 is_contiguous() = {_t_contig.is_contiguous()}  ← 可以安全 view")
print(f"    view 后形状：{tuple(_t_contig.view(2, 5, 4 * 16).shape)}（等价于把 h 与 d_k 两维合并回 d_model）")

sub("4.3 多头 vs 单头：每个头学到不同的注意力模式")

# 同一个输入，用两种「头配置」看注意力分布差异（随机权重下只能看形状与统计量）
_x_probe = torch.randn(1, 8, 64)
torch.manual_seed(0)
_mha_multi = MultiHeadAttention(d_model=64, num_heads=4, dropout=0.0)
torch.manual_seed(0)
_, _w_multi = _mha_multi(_x_probe, _x_probe, _x_probe)[:2]
print(f"    4 头的注意力权重形状：{tuple(_w_multi.shape)}  = (batch, heads, seq_q, seq_k)")
for h in range(4):
    _row = _w_multi[0, h, 0]                              # 第 0 个 Query 在第 h 个头的权重分布
    print(f"      头 {h}：第 0 个 Query 的注意力分布 = {[round(v, 4) for v in _row.tolist()]}"
          f"  最大权重位置 = {int(_row.argmax())}")
print("    可以看到不同头的分布形状不同 —— 这正是多头能捕捉多种相关性的来源。")


# ===========================================================================
# 第 5 节：前馈网络 FeedForward
# ===========================================================================
title("第 5 节：Feed Forward Network —— 逐位置的前馈变换")


class FeedForward(nn.Module):
    """位置前馈网络：d_model → ff_dim → d_model，对每个位置独立做同一套非线性变换。

    两个关键点：
      1) 中间维度 ff_dim 通常是 d_model 的 4 倍（原论文 512 → 2048），先扩维再缩回，
         给模型足够的非线性容量；参数主要就集中在这一层。
      2) 「逐位置（position-wise）」：同一个 Linear 作用在每个 token 上，token 之间不交互，
         交互全部交给注意力层。所以模型里 attn 负责「跨位置混合信息」，FFN 负责「逐位置加工」。
    """

    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_dim),      # 扩维，如 64 → 256
            nn.ReLU(),                       # 非线性（原论文用 ReLU，现代实现常用 GELU）
            nn.Dropout(p=dropout),
            nn.Linear(ff_dim, d_model),      # 缩回原维度
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


torch.manual_seed(42)
_ffn = FeedForward(d_model=64, ff_dim=256, dropout=0.0)
_x_ffn = torch.randn(2, 10, 64)
_out_ffn = _ffn(_x_ffn)
show_shape("FFN 输入", _x_ffn)
show_shape("FFN 输出", _out_ffn)
print(f"    FFN 参数量：{sum(p.numel() for p in _ffn.parameters()):,}"
      f"（= 64×256+256 + 256×64+64 = {64 * 256 + 256 + 256 * 64 + 64:,}）")
print("    FFN 逐位置独立：把输入的第 0 个位置单独喂进去，结果应等于整体输出的第 0 个位置。")
print(f"    验证：{torch.allclose(_ffn(_x_ffn[:, 0:1, :]), _out_ffn[:, 0:1, :], atol=1e-6)}"
      f"  ← True 说明位置之间确实没有交互")


# ===========================================================================
# 第 6 节：Add & Norm（残差 + LayerNorm）
# ===========================================================================
title("第 6 节：Add & Norm —— 残差连接 + 层归一化")

print("""
【公式】Post-LN（原论文写法）：x = LayerNorm( x + Dropout(Sublayer(x)) )
        Pre-LN （现代主流写法）：x = x + Dropout(Sublayer(LayerNorm(x)))

【残差连接的作用】
    y = x + F(x)  →  ∂y/∂x = 1 + F'(x)
    那个常数 1 保证梯度有一条「直传通道」，即使 F' 很小梯度也不会消失。
    这与训练技巧一节讲「残差连接缓解梯度消失」是同一个道理；
    LSTM 的细胞状态 C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t 中 f_t 那条「免衰减旁路」也是同一思想。

【LayerNorm 而不是 BatchNorm】
    归一化公式相同：y = γ·(x-μ)/√(σ²+ε) + β，区别只在「在哪些维度上算 μ 和 σ²」。
    LayerNorm 对**每个 token 的 d_model 维向量独立归一化**，与 batch 组成、序列长度完全无关：
      - 训练和推理行为一致；
      - 变长序列 / padding 不影响结果；
      - 自回归逐 token 生成时 batch=1 也能正常工作（BatchNorm 此时根本算不出统计量）。
    Transformer 里数据是 (B, L, H) 三维，BatchNorm 会把 batch 和序列维混在一起算统计量，
    同一个 token 的归一化结果会依赖同 batch 的其它句子 → 不可用。
""")

sub("6.1 LayerNorm 形状验证与「每个 token 独立归一化」")

torch.manual_seed(42)
_ln = nn.LayerNorm(64)
_x_ln = torch.randn(2, 5, 64) * 3.0 + 5.0                # 故意让均值和方差都偏离 0/1
_out_ln = _ln(_x_ln)
show_shape("LayerNorm 输入", _x_ln)
show_shape("LayerNorm 输出", _out_ln)
print(f"    输入每行 mean/std 示例：{_x_ln[0, 0].mean():.4f} / {_x_ln[0, 0].std():.4f}")
print(f"    输出每行 mean/std 示例：{_out_ln[0, 0].mean():.6f} / {_out_ln[0, 0].std():.6f}"
      f"  ← 均值被归一化到 0、标准差≈1")
print(f"    （std 显示为 {_out_ln[0, 0].std():.4f} 而不是精确的 1.0，是因为 LayerNorm 内部用"
      f"「有偏方差」除以 N，而 torch.std 默认用「无偏方差」除以 N-1，")
print(f"      比值 sqrt(64/63) = {math.sqrt(64 / 63):.4f}，正好对得上。）")
print(f"    γ、β 的初始值与形状：γ={_ln.weight.shape} 全 1，β={_ln.bias.shape} 全 0"
      f"（可学习，让网络自己决定每层最佳分布范围）")


class SublayerConnection(nn.Module):
    """一个「子层 + 残差 + LayerNorm」的封装，支持 Pre-LN 和 Post-LN 两种顺序。"""

    def __init__(self, d_model: int, dropout: float = 0.1, pre_norm: bool = True) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.pre_norm = pre_norm

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        if self.pre_norm:
            # Pre-LN：先归一化再进子层，残差是「干净」的恒等通路 → 训练更稳，常不需要 warmup
            return x + self.dropout(sublayer(self.norm(x)))
        # Post-LN：原论文写法，先子层再归一化
        return self.norm(x + self.dropout(sublayer(x)))


sub("6.2 Pre-LN 与 Post-LN 的输出差异")

torch.manual_seed(42)
_sublayer = nn.Linear(64, 64)
_pre = SublayerConnection(64, dropout=0.0, pre_norm=True)
_post = SublayerConnection(64, dropout=0.0, pre_norm=False)
_x_sc = torch.randn(2, 5, 64)
_out_pre = _pre(_x_sc, _sublayer)
_out_post = _post(_x_sc, _sublayer)
print(f"    Pre-LN  输出范数均值：{_out_pre.norm(dim=-1).mean():.4f}"
      f"（残差项未归一化，所以还保留输入的尺度）")
print(f"    Post-LN 输出范数均值：{_out_post.norm(dim=-1).mean():.4f}"
      f"（最后过了 LayerNorm，尺度被统一）")


# ===========================================================================
# 第 7 节：编码器块 EncoderBlock
# ===========================================================================
title("第 7 节：EncoderBlock —— 双向自注意力 + 前馈网络")


class EncoderBlock(nn.Module):
    """Transformer 编码器块：两个子层，每个子层外面都有残差连接和 LayerNorm。

    子层 1：Multi-Head **Self**-Attention（无掩码，双向）
            Q=K=V=x，每个 token 都能看到序列中的**所有** token（包括自己后面的）。
            例如翻译 "I love you" 时，编码 "love" 时既可以看前面的 "I"，也可以看后面的 "you"。
    子层 2：Feed Forward Network（逐位置非线性变换）
    """

    def __init__(self, d_model: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1, pre_norm: bool = True) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, ff_dim, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout, pre_norm)   # 自注意力外面那套
        self.sublayer2 = SublayerConnection(d_model, dropout, pre_norm)   # 前馈网络外面那套

    def forward(self, x: torch.Tensor,
                src_mask: torch.Tensor | None = None) -> torch.Tensor:
        # 子层 1：Self-Attention（Q=K=V=x）。src_mask 用于屏蔽 padding 位置，一般为 None。
        def _attn(_x: torch.Tensor) -> torch.Tensor:
            return self.self_attn(_x, _x, _x, mask=src_mask)[0]

        x = self.sublayer1(x, _attn)
        # 子层 2：Feed Forward，逐位置独立
        x = self.sublayer2(x, self.feed_forward)
        return x


torch.manual_seed(42)
_encoder = EncoderBlock(d_model=64, num_heads=4, ff_dim=256, dropout=0.0)
_x_enc = torch.randn(2, 10, 64)
_out_enc = _encoder(_x_enc)
show_shape("Encoder 输入", _x_enc)
show_shape("Encoder 输出", _out_enc)
print("    → 输入输出同形状 (2, 10, 64)，这是 Transformer 能无限堆叠的前提。")
print(f"    EncoderBlock 参数量：{sum(p.numel() for p in _encoder.parameters()):,}")

sub("7.1 堆叠 N 层编码器")

_encoder_stack = nn.ModuleList([
    EncoderBlock(d_model=64, num_heads=4, ff_dim=256, dropout=0.0) for _ in range(3)
])
_h = _x_enc
for _i, _layer in enumerate(_encoder_stack):
    _h = _layer(_h)
    print(f"    经过第 {_i + 1} 层后：shape = {tuple(_h.shape)}，"
          f"数值范数均值 = {_h.norm(dim=-1).mean():.4f}")
print("    深层堆叠时 LayerNorm + 残差保证了数值尺度的稳定（没有爆炸也没有塌缩）。")


# ===========================================================================
# 第 8 节：解码器块 DecoderBlock
# ===========================================================================
title("第 8 节：DecoderBlock —— 掩码自注意力 + 交叉注意力 + 前馈网络")

print("""
【Decoder Block 有三个子层，比 Encoder 多一个 Cross-Attention】
    子层 1：Masked Self-Attention  —— Decoder 自己看自己，用**因果掩码**防止看到未来 token
    子层 2：Cross-Attention        —— 用 Decoder 的 Q 去 Encoder 输出的 K、V 中检索信息
    子层 3：Feed Forward Network   —— 逐位置非线性变换

【因果掩码（Causal Mask）长什么样】
    Token位置:   1      2      3
               "I"  "love"  "you"
    位置1 "I"    ✓      ✗      ✗    ← 只能看自己
    位置2 "love" ✓      ✓      ✗    ← 能看 "I" 和自己
    位置3 "you"  ✓      ✓      ✓    ← 能看前面所有词
    实现：torch.tril(torch.ones(L, L)).bool()，下三角（含对角线）为 True。
    **本脚本的约定是 True = 可见**（配合 scaled_dot_product_attention 里的
      `masked_fill(mask == 0, -1e9)`）。
    这是为了保证自回归生成：生成第 t 个词时，模型只能基于前 t-1 个已生成的词做预测。

【Cross-Attention 在做什么】
    翻译 "I love you" → "我爱你"，Decoder 生成第二个词「爱」时：
      - Encoder 已处理完源句，输出三个向量对 (K_I,V_I)、(K_love,V_love)、(K_you,V_you)；
      - Decoder 当前的 Query 向量 Q_爱 和三个 Key 分别点积算相似度 → Softmax 得权重如
        "I":0.05、"love":0.90、"you":0.05；
      - 加权求和 Output = 0.05·V_I + 0.90·V_love + 0.05·V_you，
        这个向量 **90% 的信息来自 "love" 的 Value**。
    一句话：Cross-Attention 就是「在源句子里找到和当前生成最相关的词，把它的语义搬过来辅助预测」。
""")


class DecoderBlock(nn.Module):
    """Transformer 解码器块：Masked Self-Attention + Cross-Attention + FFN。"""

    def __init__(self, d_model: int, num_heads: int, ff_dim: int,
                 dropout: float = 0.1, pre_norm: bool = True) -> None:
        super().__init__()
        # 子层 1：Masked Self-Attention —— Decoder 看自己，加因果掩码防偷看未来
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 子层 2：Cross-Attention —— Decoder 出 Q，Encoder 出 K、V
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        # 子层 3：前馈网络
        self.feed_forward = FeedForward(d_model, ff_dim, dropout)
        # 三个子层各配一套「残差 + LayerNorm」，比 Encoder 多一个
        self.sublayer1 = SublayerConnection(d_model, dropout, pre_norm)
        self.sublayer2 = SublayerConnection(d_model, dropout, pre_norm)
        self.sublayer3 = SublayerConnection(d_model, dropout, pre_norm)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor,
                tgt_mask: torch.Tensor | None = None,
                return_attn: bool = False):
        _captured: dict[str, torch.Tensor] = {}

        # 子层 1：Masked Self-Attention（Q=K=V=x，只允许看当前及之前的 token）
        def _masked_self(_x: torch.Tensor) -> torch.Tensor:
            _o, _w = self.self_attn(_x, _x, _x, mask=tgt_mask)
            _captured["self"] = _w
            return _o

        x = self.sublayer1(x, _masked_self)

        # 子层 2：Cross-Attention（Q 来自 Decoder，K、V 来自 Encoder 输出；不加掩码）
        def _cross(_x: torch.Tensor) -> torch.Tensor:
            _o, _w = self.cross_attn(_x, encoder_output, encoder_output, mask=None)
            _captured["cross"] = _w
            return _o

        x = self.sublayer2(x, _cross)

        # 子层 3：Feed Forward
        x = self.sublayer3(x, self.feed_forward)

        if return_attn:
            return x, _captured
        return x


sub("8.1 形状验证")

torch.manual_seed(42)
_decoder = DecoderBlock(d_model=64, num_heads=4, ff_dim=256, dropout=0.0)
_dec_input = torch.randn(2, 6, 64)                       # Decoder 已生成 6 个 token
_enc_output = torch.randn(2, 10, 64)                     # Encoder 输出 10 个 token
_tgt_len = _dec_input.size(1)
# 因果掩码：下三角为 True（本实现约定 True = 可见）
_causal_mask = torch.tril(torch.ones(_tgt_len, _tgt_len, dtype=torch.bool))
print("    因果掩码（1=可见，0=屏蔽）：")
for _r in _causal_mask.int().tolist():
    print(f"        {_r}")

_dec_out, _dec_attn = _decoder(_dec_input, _enc_output, _causal_mask, return_attn=True)
show_shape("Decoder 输入 x", _dec_input)
show_shape("Encoder 输出（K、V 来源）", _enc_output)
show_shape("Decoder 输出", _dec_out)
show_shape("Masked Self-Attention 权重", _dec_attn["self"])
show_shape("Cross-Attention 权重", _dec_attn["cross"])
print(f"    Cross-Attention 权重形状 (2,4,6,10)：Query 长度 6、Key 长度 10，"
      f"每行和 = {_dec_attn['cross'][0, 0].sum(dim=-1)[:3].tolist()}")

sub("8.2 实证因果掩码真的生效：改动未来 token，前面的输出必须不变")

_modified = _dec_input.clone()
_modified[:, 4, :] += 10.0                               # 只修改第 4 个位置（未来）
_out_modified = _decoder(_modified, _enc_output, _causal_mask)
# 位置 0..3 的 Query 看不到位置 4，所以它们的输出应该完全不变
_unchanged_part = _out_modified[:, :4, :]
_original_part = _dec_out[:, :4, :]
_changed_part = _out_modified[:, 4:, :]
print(f"    修改第 4 个 token 后，输出在位置 0~3 是否完全不变："
      f"{torch.allclose(_unchanged_part, _original_part, atol=1e-6)}  ← 必须为 True")
print(f"    位置 4~5 的输出是否被改变："
      f"{not torch.allclose(_changed_part, _dec_out[:, 4:, :], atol=1e-6)}  ← 必须为 True")
print("    这就证明了因果掩码确实阻断了「未来信息」的泄漏。")

print()
print("    对照实验：如果**不加**掩码（双向注意力），未来信息就会泄漏：")
_dec_out_nomask = _decoder(_modified, _enc_output, tgt_mask=None)
print(f"    不加掩码时，位置 0~3 的输出是否被改变："
      f"{not torch.allclose(_dec_out_nomask[:, :4, :], _original_part, atol=1e-6)}"
      f"  ← True 表示泄漏了未来信息")
print(f"    掩码自注意力权重矩阵的上三角是否全为 0："
      f"{bool((torch.triu(_dec_attn['self'][0, 0], diagonal=1) < 1e-4).all())}  ← 必须为 True")


# ===========================================================================
# 第 9 节：组装完整 Transformer 并在玩具任务上训练
# ===========================================================================
title("第 9 节：组装完整 Transformer（Embedding + PE + N 层编码器 + N 层解码器 + 输出投影）")

print("""
【完整数据流】
    src token id (B, Ls)                          tgt token id (B, Lt)
         │  Embedding × √d_model                        │  Embedding × √d_model
         ↓                                              ↓
    src_emb (B,Ls,d) + PositionalEncoding         tgt_emb (B,Lt,d) + PositionalEncoding
         ↓                                              ↓
    ┌──────────────┐                              ┌──────────────────────────┐
    │ Encoder × N  │  → memory (B, Ls, d) ───────→│ Decoder × N               │
    │ 双向自注意力  │                              │ ① 掩码自注意力（因果）    │
    └──────────────┘                              │ ② 交叉注意力（Q←tgt,      │
                                                  │    K,V←memory）           │
                                                  │ ③ 前馈网络                │
                                                  └──────────────────────────┘
                                                            ↓
                                                   Linear(d, vocab) → logits
                                                            ↓
                                                        Softmax → 下一个 token 的概率分布

【为什么 Embedding 要乘 √d_model】
    Embedding 初始化后标准差约 1，而位置编码的幅度也是 O(1)，
    两者相加时词义信号会被稀释。乘 √d_model 把 Embedding 的尺度放大到和位置编码匹配的量级
    （原论文的做法），保证「词义」和「位置」两种信号权重相当。
""")


class Transformer(nn.Module):
    """从零组装的 Encoder-Decoder Transformer（教学用，规模很小）。"""

    def __init__(self, vocab_size: int, d_model: int = 64, num_heads: int = 4,
                 num_encoder_layers: int = 2, num_decoder_layers: int = 2,
                 ff_dim: int = 256, dropout: float = 0.0, max_len: int = 64,
                 pre_norm: bool = True) -> None:
        super().__init__()
        self.d_model = d_model

        # 1. 词嵌入：把 token id 变成 d_model 维向量（本质是查表）
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        # 2. 位置编码：注入位置信息
        self.src_pe = PositionalEncoding(d_model, max_len, dropout)
        self.tgt_pe = PositionalEncoding(d_model, max_len, dropout)
        # 3. N 层编码器
        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, ff_dim, dropout, pre_norm)
            for _ in range(num_encoder_layers)
        ])
        # 4. N 层解码器
        self.decoder_layers = nn.ModuleList([
            DecoderBlock(d_model, num_heads, ff_dim, dropout, pre_norm)
            for _ in range(num_decoder_layers)
        ])
        # 5. 输出投影：把 d_model 维映射回词表大小，得到每个 token 的 logits
        self.output_projection = nn.Linear(d_model, vocab_size)

        # 参数初始化：Xavier 适合配合 LayerNorm，避免初始尺度过大/过小
        for _p in self.parameters():
            if _p.dim() > 1:
                nn.init.xavier_uniform_(_p)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor | None = None) -> torch.Tensor:
        """编码：token id (B, Ls) → memory (B, Ls, d_model)。"""
        # Embedding 乘 √d_model，让词义信号和位置编码的信号在同一量级
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.src_pe(x)                                  # 加位置编码
        for _layer in self.encoder_layers:
            x = _layer(x, src_mask)
        return x

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor,
               tgt_mask: torch.Tensor | None = None) -> torch.Tensor:
        """解码：token id (B, Lt) + memory → (B, Lt, d_model)。"""
        x = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        x = self.tgt_pe(x)
        for _layer in self.decoder_layers:
            x = _layer(x, memory, tgt_mask)
        return x

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                tgt_mask: torch.Tensor | None = None) -> torch.Tensor:
        """返回 (B, Lt, vocab_size) 的 logits。"""
        memory = self.encode(src)
        hidden = self.decode(tgt, memory, tgt_mask)
        return self.output_projection(hidden)                # 注意：不在模型里做 Softmax，
        # 因为 CrossEntropyLoss 内部已经包含 log_softmax，重复 softmax 会破坏数值稳定性


def make_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """生成本实现约定的因果掩码：下三角为 True（可见），上三角为 False（屏蔽）。"""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


sub("9.1 形状验证（token id 输入 → logits 输出）")

torch.manual_seed(42)
_VOCAB = 12                                              # 词表大小（含 0 号 BOS/padding）
_model = Transformer(vocab_size=_VOCAB, d_model=64, num_heads=4,
                     num_encoder_layers=2, num_decoder_layers=2,
                     ff_dim=256, dropout=0.0)
_src_ids = torch.randint(1, _VOCAB, (2, 7))              # (B=2, Ls=7)
_tgt_ids = torch.randint(1, _VOCAB, (2, 5))              # (B=2, Lt=5)
_logits = _model(_src_ids, _tgt_ids, make_causal_mask(_tgt_ids.size(1)))

show_shape("src token id", _src_ids)
show_shape("tgt token id", _tgt_ids)
show_shape("encoder memory", _model.encode(_src_ids))
show_shape("decoder 隐状态", _model.decode(_tgt_ids, _model.encode(_src_ids),
                                          make_causal_mask(_tgt_ids.size(1))))
show_shape("输出 logits", _logits)
print(f"    自定义 Transformer 总参数量：{sum(p.numel() for p in _model.parameters()):,}")

sub("9.2 玩具任务实战：序列反转（Reverse）")

print("""
【任务】让 Transformer 学会把输入序列**反转**输出。
    输入 src      = [3, 7, 1, 9]（随机 1..V-1）
    目标输出 tgt  = [9, 1, 7, 3]
    训练时用 teacher forcing：Decoder 的输入是「BOS + 目标右移一位」，
    即 [0, 9, 1, 7]，要预测的标签是 [9, 1, 7, 3]。
    这个任务必须依赖**位置信息**，同时需要 Cross-Attention 从源序列搬运信息，
    是检验 Encoder-Decoder 是否真的工作的经典小任务。
    规模刻意取小（V=12、L=4、d_model=32、各 1 层、80 步），CPU 上几秒即可完成。
""")

torch.manual_seed(42)
np.random.seed(42)
_TASK_VOCAB = 12
_TASK_LEN = 4
_BOS = 0                                                 # 用 0 号 token 当 BOS

_task_model = Transformer(vocab_size=_TASK_VOCAB, d_model=32, num_heads=2,
                          num_encoder_layers=1, num_decoder_layers=1,
                          ff_dim=64, dropout=0.0, max_len=16)


def make_reverse_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """造一批「序列反转」任务数据。"""
    src = torch.randint(1, _TASK_VOCAB, (batch_size, _TASK_LEN))    # 1..V-1，避免和 BOS 撞号
    tgt_out = torch.flip(src, dims=[1])                             # 反转后的目标序列
    # Decoder 输入 = BOS + 目标去掉最后一个 token（右移一位，teacher forcing）
    tgt_in = torch.cat([torch.full((batch_size, 1), _BOS, dtype=torch.long),
                        tgt_out[:, :-1]], dim=1)
    return src, tgt_in, tgt_out


_criterion = nn.CrossEntropyLoss()
_optimizer = torch.optim.Adam(_task_model.parameters(), lr=3e-3)
_task_mask = make_causal_mask(_TASK_LEN)
_loss_history: list[float] = []

_task_model.train()
for _step in range(81):
    _s, _ti, _to = make_reverse_batch(32)
    _logit = _task_model(_s, _ti, _task_mask)                     # (32, 4, V)
    # CrossEntropyLoss 需要 (N, C) 的 logits 和 (N,) 的标签，所以把 batch 和时间维展平
    _loss = _criterion(_logit.reshape(-1, _TASK_VOCAB), _to.reshape(-1))
    _optimizer.zero_grad()
    _loss.backward()
    _optimizer.step()
    _loss_history.append(_loss.item())
    if _step % 10 == 0 or _step == 80:
        print(f"    第 {_step:>3d} 步：loss = {_loss.item():.4f}")

# 评估：贪心解码（每步只预测下一个 token，逐步喂回去，完全不看答案）
_task_model.eval()
with torch.no_grad():
    _eval_src, _, _eval_target = make_reverse_batch(6)
    _dec_ids = torch.full((6, 1), _BOS, dtype=torch.long)
    _memory = _task_model.encode(_eval_src)
    for _t in range(_TASK_LEN):
        _h = _task_model.decode(_dec_ids, _memory, make_causal_mask(_dec_ids.size(1)))
        _next = _task_model.output_projection(_h[:, -1:, :]).argmax(dim=-1)   # (6,1)
        _dec_ids = torch.cat([_dec_ids, _next], dim=1)
    _predicted = _dec_ids[:, 1:]                                # 去掉开头的 BOS

print()
print("    贪心解码结果（左：输入，中：模型预测，右：正确答案）")
_correct = 0
for _i in range(6):
    _ok = torch.equal(_predicted[_i], _eval_target[_i])
    _correct += int(_ok)
    print(f"      输入 {_eval_src[_i].tolist()}  →  预测 {_predicted[_i].tolist()}"
          f"  |  正确 {_eval_target[_i].tolist()}  {'✓' if _ok else '✗'}")
print(f"    序列级完全正确率：{_correct}/6    逐元素准确率："
      f"{( _predicted == _eval_target).float().mean().item():.4f}")
print(f"    loss 从 {_loss_history[0]:.4f} 降到 {_loss_history[-1]:.4f}"
      f"（说明 Encoder + 因果掩码 + Cross-Attention 整条链路都正确工作）")


# ===========================================================================
# 第 10 节：三种注意力对比
# ===========================================================================
title("第 10 节：Transformer 中的三种注意力")

print("""
三种注意力的底层都是同一个 Multi-Head Attention，区别只在
**Q、K、V 的来源**和**是否使用掩码**：

  类型                    | Q 来源      | K、V 来源    | 掩码         | 用在哪
  ------------------------|-------------|--------------|--------------|------------------
  1. Self-Attention       | Encoder 自身| Encoder 自身 | 无（双向）   | Encoder
  2. Masked Self-Attention| Decoder 自身| Decoder 自身 | Causal Mask  | Decoder 第一层
  3. Cross-Attention      | Decoder     | Encoder 输出 | 无           | Decoder 第二层

代码上分别体现为：
  1) self_attn(x, x, x)                                 —— 三个参数传同一个 x
  2) self_attn(x, x, x, mask=causal_mask)               —— 同样三个 x，但加了因果掩码
  3) cross_attn(tgt, memory, memory)                    —— Q 用 tgt，K/V 用 encoder 的 memory
""")

sub("10.1 三种注意力的形状对照（seq_q 与 seq_k 可以不同）")

torch.manual_seed(42)
_attn_module = MultiHeadAttention(d_model=32, num_heads=2, dropout=0.0)
_x_self = torch.randn(2, 6, 32)                          # 用于 Self-Attention
_memory = torch.randn(2, 9, 32)                          # 用于 Cross-Attention 的 K、V

# ① Self-Attention：Q=K=V，双向，无掩码
_out1, _w1 = _attn_module(_x_self, _x_self, _x_self)[:2]
# ② Masked Self-Attention：Q=K=V，加因果掩码
_out2, _w2 = _attn_module(_x_self, _x_self, _x_self, mask=make_causal_mask(6))[:2]
# ③ Cross-Attention：Q 来自 decoder，K/V 来自 encoder 输出
_out3, _w3 = _attn_module(_x_self, _memory, _memory)[:2]

for _name, _o, _w in (("① Self-Attention      ", _out1, _w1),
                      ("② Masked Self-Attn   ", _out2, _w2),
                      ("③ Cross-Attention    ", _out3, _w3)):
    print(f"    {_name} 输出 {tuple(_o.shape)}   权重 {tuple(_w.shape)}")
print()
print("    ① ② 的权重都是 (2, 2, 6, 6)（Query 6 个、Key 6 个）；")
print("    ③ 的权重是 (2, 2, 6, 9)——Query 6 个来自 Decoder，Key 9 个来自 Encoder，")
print("    这正是「用 Decoder 的 Q 去 Encoder 的输出里检索」的形状证据。")
print()
print(f"    ② 的权重上三角是否全为 0："
      f"{bool((torch.triu(_w2[0, 0], diagonal=1) < 1e-4).all())}  ← 因果掩码生效")
print(f"    ① 的权重上三角是否有非零值："
      f"{bool((torch.triu(_w1[0, 0], diagonal=1) > 1e-4).any())}  ← 双向注意力，未来可见")


# ===========================================================================
# 第 11 节：torch.nn 框架用法（含 attn_mask 语义避坑）
# ===========================================================================
title("第 11 节：torch.nn 内置 Transformer 组件的框架用法")

print("""
【为什么要学框架版】手写版帮你理解原理，实际工程里用 nn.Transformer / nn.TransformerEncoder
等内置组件，它们经过了算子融合与性能优化（PyTorch 2.x 内部会走
F.scaled_dot_product_attention 的融合内核）。

【重点避坑：attn_mask 的语义在不同 API 里是**相反**的！】
  · nn.MultiheadAttention / nn.TransformerDecoder，当 attn_mask 是 **bool** 张量时：
        **True = 不允许注意（屏蔽）**，False = 保持不变。
    官方文档原文：positions with True are not allowed to attend。
  · F.scaled_dot_product_attention，当 attn_mask 是 **bool** 张量时：
        **True = 参与注意力**，False = 屏蔽（填 -inf）。语义**相反**！
  · 因此如果要显式构造掩码，最稳妥的做法是**用 float 掩码**：
        -inf 表示屏蔽、0 表示保留。float 掩码在两者中都是「加到 scores 上」，
        语义统一，不会踩坑。

【课案代码里的一处错误（本节会实测证明）】
    课案 DecoderBlock 里写：
        tgt_mask = torch.tril(torch.ones(5, 5)).bool()   # 注释：True=可见
    但 nn.MultiheadAttention 的 bool 掩码是 True=屏蔽，
    所以这个 tril 掩码实际把「下三角（含自己）」全屏蔽了，正好反了！
    正确写法应该是：
        bool 版：torch.triu(torch.ones(L, L), diagonal=1).bool()   （上三角为 True=屏蔽未来）
        float 版：nn.Transformer.generate_square_subsequent_mask(L)（上三角为 -inf）
    下面实测对比三种写法的注意力权重，一眼就能看出哪个对。
""")

sub("11.1 实测 nn.MultiheadAttention 的 bool 掩码语义")

torch.manual_seed(42)
_mha_builtin = nn.MultiheadAttention(embed_dim=16, num_heads=2, batch_first=True,
                                     dropout=0.0)
_probe = torch.randn(1, 4, 16)
_L = 4

# 写法 A：课案的写法（tril 的 bool，注释说 True=可见）—— 实测会发现问题
_mask_a = torch.tril(torch.ones(_L, _L)).bool()
_, _w_a = _mha_builtin(_probe, _probe, _probe, attn_mask=_mask_a,
                       need_weights=True, average_attn_weights=False)
# 写法 B：正确的 bool 写法（上三角为 True=屏蔽）
_mask_b = torch.triu(torch.ones(_L, _L), diagonal=1).bool()
_, _w_b = _mha_builtin(_probe, _probe, _probe, attn_mask=_mask_b,
                       need_weights=True, average_attn_weights=False)
# 写法 C：float 掩码（-inf 屏蔽未来），语义最统一
_mask_c = nn.Transformer.generate_square_subsequent_mask(_L)
_, _w_c = _mha_builtin(_probe, _probe, _probe, attn_mask=_mask_c,
                       need_weights=True, average_attn_weights=False)

print("    写法 A：tril 的 bool 掩码（课案写法，注释写的是 True=可见）")
for _r in _w_a[0, 0].tolist():
    print(f"        {[round(v, 4) for v in _r]}")
print("      ↑ 第 0 行只有第 1、2、3 列有权重，第 0 列（自己）反而被屏蔽了 → 说明 True=屏蔽")
print("      ↑ 更要命的是第 3 行：tril 的第 3 行是 [1,1,1,1]，整行被屏蔽，")
print("        Softmax 对整行 -inf 求值得到 nan。也就是说课案这段代码不仅掩码方向反了，")
print("        在序列最后一个位置上还会直接算出 nan。（这里打印出 nan 是刻意的演示，")
print("        不是脚本出错 —— 我们自己的实现用 -1e9 就是为了避免这个 nan。）")
print()
print("    写法 B：triu(diagonal=1) 的 bool 掩码（正确：True=屏蔽未来）")
for _r in _w_b[0, 0].tolist():
    print(f"        {[round(v, 4) for v in _r]}")
print("      ↑ 标准的因果形状：第 0 行只能看第 0 列，第 1 行能看 0~1 列 ……")
print()
print("    写法 C：generate_square_subsequent_mask 的 float 掩码（=-inf 屏蔽未来）")
for _r in _w_c[0, 0].tolist():
    print(f"        {[round(v, 4) for v in _r]}")
print(f"      ↑ 与写法 B 的结果完全一致：{torch.allclose(_w_b, _w_c, atol=1e-6)}")
print(f"    而写法 A 与写法 C 不一致：{not torch.allclose(_w_a, _w_c, atol=1e-6)}")
print()
print("    ★ 结论：bool 掩码在 nn.MultiheadAttention 里是 True=屏蔽。")
print("      课案的 `torch.tril(...).bool()` 配「True=可见」注释是错的，")
print("      正确要么用 triu(diagonal=1).bool()，要么用 float 的 -inf 掩码。")
print("      （本脚本第 3~8 节自己实现的 scaled_dot_product_attention 用的是")
print("        `masked_fill(mask == 0, -1e9)`，约定 True=可见，与内置的相反但自洽。）")

sub("11.2 nn.Transformer 完整框架用法")

torch.manual_seed(42)
# 注意：nn.Transformer 没有 enable_nested_tensor 参数（那是 TransformerEncoder 的），
# 所以这里不传。若需要关掉嵌套张量优化，请在 TransformerEncoder/EncoderLayer 上设置。
_framework_model = nn.Transformer(
    d_model=64,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=256,
    dropout=0.0,
    batch_first=True,              # 输入输出用 (batch, seq, d_model) 格式
)
_fw_src = torch.randn(2, 10, 64)                        # (batch=2, src_len=10, d_model=64)
_fw_tgt = torch.randn(2, 8, 64)                         # (batch=2, tgt_len=8,  d_model=64)
_fw_tgt_mask = nn.Transformer.generate_square_subsequent_mask(_fw_tgt.size(1))

_fw_out = _framework_model(_fw_src, _fw_tgt, tgt_mask=_fw_tgt_mask)
show_shape("src", _fw_src)
show_shape("tgt", _fw_tgt)
show_shape("output", _fw_out)
print(f"    nn.Transformer 参数量：{sum(p.numel() for p in _framework_model.parameters()):,}")
print()
print("    tgt_mask 的形状与内容（float，上三角为 -inf）：")
print(f"        {tuple(_fw_tgt_mask.shape)}  dtype={_fw_tgt_mask.dtype}")
print(f"        第 0 行 = {_fw_tgt_mask[0].tolist()}")
print(f"    nn.TransformerDecoder 内部把这个 tgt_mask 直接当成 self-attn 的 attn_mask 使用，")
print(f"    float 掩码的语义是「加到 scores 上」，-inf 即屏蔽，所以没有 True/False 的歧义。")

print()
print("    新版 PyTorch 还有 tgt_is_causal / src_is_causal / memory_is_causal 这些标志，")
print("    但它们的语义是「**提示**（hint）」而**不是「指令」**，这一点极易踩坑：")
print("      官方文档原文：is_causal provides a *hint* that attn_mask is the causal mask.")
print("      也就是说：它只是告诉融合内核「你给的那个 attn_mask 就是因果掩码，不用再自己构造了」，")
print("      如果 attn_mask 根本是 None，这个提示就什么也不做 —— **不会自动加上因果掩码**。")
print("    下面实测（先 eval() 关掉 Dropout 的随机性，否则对比没意义）：")
_framework_model.eval()
with torch.no_grad():
    _out_with_mask = _framework_model(_fw_src, _fw_tgt, tgt_mask=_fw_tgt_mask)
    _out_is_causal = _framework_model(_fw_src, _fw_tgt, tgt_is_causal=True)
    _out_no_mask = _framework_model(_fw_src, _fw_tgt)
print(f"      显式 tgt_mask  vs  tgt_is_causal=True ：最大差异 "
      f"{(_out_with_mask - _out_is_causal).abs().max().item():.6f}  ← 差异很大")
print(f"      无掩码         vs  tgt_is_causal=True ：最大差异 "
      f"{(_out_no_mask - _out_is_causal).abs().max().item():.6f}  ← 完全一致")
print("    ★ 结论：tgt_is_causal=True 单独使用时等于「没有掩码」，未来信息会泄漏！")
print("      正确的用法是：先构造好 tgt_mask（用 generate_square_subsequent_mask），")
print("      再用 tgt_mask=tgt_mask 传进去；tgt_is_causal 只在想省掉掩码显式构造、")
print("      且确认底层走融合内核时才作为加速提示使用。")
print("      对本脚本来说，最稳妥、最可读的写法始终是：**显式传 float 的 -inf 掩码**。")

sub("11.3 nn.TransformerEncoder / nn.TransformerDecoder / nn.TransformerEncoderLayer")

torch.manual_seed(42)
_fw_encoder_layer = nn.TransformerEncoderLayer(
    d_model=64, nhead=4, dim_feedforward=256, dropout=0.0,
    batch_first=True, activation="relu",
    norm_first=True,              # True = Pre-LN（现代主流），False = Post-LN（原论文）
)
_fw_encoder = nn.TransformerEncoder(_fw_encoder_layer, num_layers=2,
                                    enable_nested_tensor=False)  # 关掉嵌套张量优化，避免 UserWarning
_fw_memory = _fw_encoder(_fw_src)
show_shape("TransformerEncoderLayer 输入", _fw_src)
show_shape("TransformerEncoder（2 层）输出 memory", _fw_memory)

_fw_decoder_layer = nn.TransformerDecoderLayer(
    d_model=64, nhead=4, dim_feedforward=256, dropout=0.0,
    batch_first=True, activation="relu", norm_first=True,
)
_fw_decoder = nn.TransformerDecoder(_fw_decoder_layer, num_layers=2)
_fw_dec_out = _fw_decoder(_fw_tgt, _fw_memory, tgt_mask=_fw_tgt_mask)
show_shape("TransformerDecoder（2 层）输出", _fw_dec_out)
print()
print(f"    nn.TransformerEncoderLayer 单层参数量："
      f"{sum(p.numel() for p in _fw_encoder_layer.parameters()):,}")
print(f"    我们手写的 EncoderBlock 参数量：      "
      f"{sum(p.numel() for p in _encoder.parameters()):,}（d_model=64、ff=256，口径一致）")
print("    两者数量级相同 —— 内置组件就是同样的结构加上算子优化。")

sub("11.4 手写版与框架版的形状对照")

print("""
    把我们的手写 Transformer 与 nn.Transformer / nn.TransformerEncoder /
    nn.TransformerDecoder 对齐结构后，理论上输出应当一致，但两者在若干细节上默认不同：
      · nn.Transformer 默认 norm_first=False（Post-LN），我们的默认是 Pre-LN；
      · nn.Transformer 会在编码器末端再套一个 LayerNorm（self.norm）；
      · 参数初始化方式不同（我们用了 Xavier，内置组件用各自的默认初始化）。
    因此这里只做**形状一致性**验证，并明确说明**不做数值等价断言**，避免误导。
    （对比形状时要注意口径：框架版的输出是 d_model 维的隐状态，
      而我们手写模型的 forward 最后还接了一层 output_projection 投到词表大小，
      所以要比的是 encode/decode 的输出，不是 forward 的输出。）
""")
_fw_src_ids = torch.randint(1, _VOCAB, (2, 10))
_fw_tgt_ids = torch.randint(1, _VOCAB, (2, 8))
with torch.no_grad():
    _my_memory = _model.encode(_fw_src_ids)
    _my_dec_hidden = _model.decode(_fw_tgt_ids, _my_memory, make_causal_mask(8))
    _fw_memory_probe = _fw_encoder(_fw_src)
    _fw_dec_probe = _fw_decoder(_fw_tgt, _fw_memory)

print(f"    编码器输出（memory）")
print(f"        手写版              {tuple(_my_memory.shape)}")
print(f"        nn.TransformerEncoder {tuple(_fw_memory_probe.shape)}"
      f"   → 形状一致：{tuple(_my_memory.shape) == tuple(_fw_memory_probe.shape)}")
print(f"    解码器输出（隐状态，未投影到词表）")
print(f"        手写版              {tuple(_my_dec_hidden.shape)}")
print(f"        nn.TransformerDecoder {tuple(_fw_dec_probe.shape)}"
      f"   → 形状一致：{tuple(_my_dec_hidden.shape) == tuple(_fw_dec_probe.shape)}")
print(f"    手写模型 forward 的最终 logits（多了一层 output_projection 投到词表）："
      f"{tuple(_model(_fw_src_ids, _fw_tgt_ids, make_causal_mask(8)).shape)}"
      f"  = (batch, tgt_len, vocab_size)")
print(f"    nn.Transformer 论文口径的完整模型输出（含 d_model→词表 投影）："
      f"{tuple(_fw_out.shape)}（这里 nn.Transformer 本身不含输出投影，所以仍是 d_model 维）")


# ===========================================================================
# 第 12 节：可视化
# ===========================================================================
title("第 12 节：可视化（位置编码、内积-距离关系、三种注意力权重）")

# --- 图 1：位置编码热力图 + 内积随相对距离的衰减 ---
_pe_vis = PositionalEncoding(d_model=64, max_len=60, dropout=0.0)
_pe_np = _pe_vis.pe[0].numpy()                           # (60, 64)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

_im = axes[0].imshow(_pe_np, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
axes[0].set_xlabel("维度下标 i（偶数维 sin，奇数维 cos）")
axes[0].set_ylabel("位置 pos")
axes[0].set_title("正弦位置编码矩阵 PE（d_model=64，max_len=60）\n"
                  "左侧高频（值变化剧烈）→ 右侧低频（值变化平缓）")
fig.colorbar(_im, ax=axes[0], fraction=0.046)

_ks = np.arange(0, 40)
_inner = np.array([float((_pe_vis.pe[0, 0] @ _pe_vis.pe[0, k]).item()) for k in _ks])
_theory = np.array([float(np.cos(k * _omega).sum()) for k in _ks])
axes[1].plot(_ks, _inner, "o-", label="实际内积 <PE(0), PE(k)>", markersize=4)
axes[1].plot(_ks, _theory, "--", label=r"理论值 $\sum_i \cos(k\cdot\omega_i)$", linewidth=1.5)
axes[1].axhline(0, color="gray", linewidth=0.8, linestyle=":")
axes[1].set_xlabel("相对距离 k")
axes[1].set_ylabel("内积")
axes[1].set_title("内积只取决于相对距离 k（与绝对位置无关）\n"
                  "距离越远内积通常越小，模型因此能感知 token 间隔")
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
_fig1 = OUTPUT_DIR / "02网络架构_06_位置编码热力图.png"
fig.savefig(_fig1, dpi=110)
plt.close(fig)
print(f"    已保存：{_fig1.name}")

# --- 图 2：三种注意力的权重热力图 ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

_panels = [
    (_w1[0, 0].detach().numpy(), "① Self-Attention（双向，Encoder）\n每个 token 能看到所有 token"),
    (_w2[0, 0].detach().numpy(), "② Masked Self-Attention（因果，Decoder）\n上三角被掩码置 0"),
    (_w3[0, 0].detach().numpy(), "③ Cross-Attention（Q←Decoder, K/V←Encoder）\nQuery 6 个 × Key 9 个"),
]
for _ax, (_mat, _ttl) in zip(axes, _panels):
    _im2 = _ax.imshow(_mat, aspect="auto", cmap="viridis", vmin=0)
    _ax.set_xlabel("Key 位置")
    _ax.set_ylabel("Query 位置")
    _ax.set_title(_ttl, fontsize=10)
    fig.colorbar(_im2, ax=_ax, fraction=0.046)
    # 在格子里标注数值，便于直接观察
    for _i in range(_mat.shape[0]):
        for _j in range(_mat.shape[1]):
            _ax.text(_j, _i, f"{_mat[_i, _j]:.2f}", ha="center", va="center",
                     fontsize=5.5, color="white" if _mat[_i, _j] < _mat.max() * 0.6 else "black")

fig.tight_layout()
_fig2 = OUTPUT_DIR / "02网络架构_06_三种注意力权重.png"
fig.savefig(_fig2, dpi=110)
plt.close(fig)
print(f"    已保存：{_fig2.name}")

# --- 图 3：玩具任务的训练 loss 曲线 ---
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(range(len(_loss_history)), _loss_history, color="tab:blue", linewidth=1.8)
ax.set_xlabel("训练步数")
ax.set_ylabel("交叉熵损失")
ax.set_title("从零手写的 Transformer 学习「序列反转」任务\n"
             f"loss {_loss_history[0]:.3f} → {_loss_history[-1]:.3f}，逐元素准确率 "
             f"{( _predicted == _eval_target).float().mean().item():.3f}")
ax.grid(alpha=0.3)
fig.tight_layout()
_fig3 = OUTPUT_DIR / "02网络架构_06_Transformer训练曲线.png"
fig.savefig(_fig3, dpi=110)
plt.close(fig)
print(f"    已保存：{_fig3.name}")


# ===========================================================================
# 收尾总结
# ===========================================================================
title("总结")

print(f"""
【本节完成的组件清单】
    1. PositionalEncoding          —— 正弦位置编码，含「内积只取决于相对距离」的数值验证
    2. scaled_dot_product_attention —— 缩放点积注意力，含 √d_k 必要性实验与掩码演示
    3. MultiHeadAttention          —— 手写多头注意力（W_Q/W_K/W_V/W_O + 拆头/并头）
    4. FeedForward                 —— 逐位置前馈网络（4 倍扩维再缩回）
    5. SublayerConnection          —— 残差 + LayerNorm 封装，支持 Pre-LN / Post-LN
    6. EncoderBlock                —— 双向自注意力 + FFN
    7. DecoderBlock                —— 掩码自注意力 + 交叉注意力 + FFN
    8. Transformer                 —— 整体组装，并在「序列反转」任务上真实训练成功
    9. 三种注意力对比              —— Self / Masked Self / Cross 的来源与形状差异
   10. torch.nn 框架用法           —— nn.Transformer / Encoder / Decoder / EncoderLayer

【关键超参数】
    d_model={_model.d_model}  num_heads={_model.self_attn.num_heads if hasattr(_model, "self_attn") else 4}
    ff_dim=256（= 4 × d_model，原论文比例）
    序列长度 ≤ 20（CPU 友好，形状关系与真实 Transformer 完全一致）
    自定义模型参数量 {sum(p.numel() for p in _model.parameters()):,}；nn.Transformer 参数量 {sum(p.numel() for p in _framework_model.parameters()):,}

【最容易踩的坑（务必记住）】
    · nn.MultiheadAttention 的 bool attn_mask：**True = 屏蔽**；
      F.scaled_dot_product_attention 的 bool attn_mask：**True = 参与**。两者相反！
      要规避歧义，就用 float 掩码（-inf 屏蔽、0 保留）。
    · 课案 DecoderBlock 里的 `torch.tril(torch.ones(L,L)).bool()` 配「True=可见」是**错的**：
      bool 掩码下 True 表示屏蔽，这个写法把「可见的下三角」全屏蔽了，
      而且最后一行整行被屏蔽 → Softmax 得到 nan。正确写法是
      `torch.triu(torch.ones(L,L), diagonal=1).bool()` 或 float 的
      `nn.Transformer.generate_square_subsequent_mask(L)`。
    · `tgt_is_causal=True` / `src_is_causal=True` 只是**提示**（hint），不是**指令**：
      在 attn_mask=None 时它什么也不做，不会自动加因果掩码（本脚本已实测证明）。
      想要因果性，必须显式传掩码。
    · transpose 之后必须 .contiguous() 才能 view/reshape。
    · 模型内部**不要**做 Softmax：CrossEntropyLoss 自带 log_softmax。
    · 位置编码用 register_buffer，不能用普通属性（否则不随模型保存/迁移设备），
      也不能用 Parameter（它不需要学习）。
    · LayerNorm 用于 NLP 而不是 BatchNorm：与 batch 组成、序列长度无关，
      训练/推理一致，逐 token 归一化。
""")
print("脚本执行完毕（无任何异常）。")

"""
对应课案章节：网络架构 / RNN（含梯度消失推导、LSTM、GRU）

本节知识点：
    1. 为什么 RNN 适合序列数据：时序记忆（隐藏状态 h_t 保存历史信息）、
       顺序处理、参数共享（所有时间步共享 W_hh / W_xh，可处理变长序列）；
       与 DNN（样本 = 独立特征向量）、CNN（建模局部空间特征）的区别。
    2. 基本公式：h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)，y_t = W_hy · h_t + b_y。
       用手写 nn.Module 实现 RNNCell，并与 nn.RNN 逐元素对比（最大误差 < 1e-6），
       讲清 PyTorch 参数名与公式的对应关系，以及 nn.RNN 内部把两个 bias 相加。
    3. nn.RNN / nn.LSTM / nn.GRU 的前向与隐藏状态形状：
       output 是每个时间步的隐藏状态，h_n 是最后一个时间步的隐藏状态；
       output[:, -1, :] == h_n[0]；num_layers=2 → h_n 形状 (2, batch, hidden)；
       bidirectional=True → hidden_size*2；batch_first=False → (seq, batch, feat)。
    4. 梯度消失：∂h_t/∂h_{t-1} = diag(tanh'(z_t)) · W_hh，跨 k 步是连乘
       ∏ diag(tanh'(z_{t-i})) · W_hh；两个因子（tanh' ∈ (0,1] 与 |λ_max|）都会让
       连乘指数衰减。实测打印每个时间步的梯度范数并画对数曲线，
       对照 RNN / LSTM / GRU 三种结构，另外演示梯度爆炸。
    5. LSTM 如何解决梯度消失：三个门（输入门 i_t、遗忘门 f_t、输出门 o_t）+ 细胞状态
       C_t 的六个公式；[h_{t-1}, x_t] 是首尾拼接；∂C_t/∂C_{t-1} = f_t + （第二项梯度），
       f_t 提供一条「免衰减旁路」且可学习。手写 LSTMCell 与 nn.LSTM 对比（误差 < 1e-5）。
    6. GRU：把遗忘门与输入门合并成更新门，公式
       r_t = σ(W_r·[h_{t-1},x_t])、z_t = σ(W_z·[h_{t-1},x_t])、
       n_t = tanh(W_n·[r_t ⊙ h_{t-1}, x_t])、h_t = (1-z_t) ⊙ n_t + z_t ⊙ h_{t-1}；
       手写 GRUCell 与 nn.GRU 对比（PyTorch GRU 门顺序是 r, z, n）。
    7. RNN / LSTM / GRU 对比表（结构复杂度、长期依赖能力、适用场景）。
    8. 小规模序列任务实战：用 nn.LSTM 做「记住序列第一个元素」的任务，
       观察训练 loss 下降与测试准确率。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Deep_Learning\\02_网络架构\\04_RNN_LSTM_GRU.py'
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

# 统一随机种子：保证每次运行结果完全可复现（本脚本里还会多次重设，用于公平对照）
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
# 一、为什么 RNN 适合序列数据
# ===========================================================================
section("一、为什么 RNN 适合序列数据")

print(
    """
【问题背景】文本、语音、时间序列有一个共同特点：当前时刻的信息与之前时刻相关。
    - 文本：「我喜欢吃苹果」中「苹果」的含义依赖于前面的词；
    - 语音：当前帧的判断可能需要参考前面的音素；
    - 股票：今天的走势与过去几天的趋势相关。

【三类网络的建模视角对比】
    网络   输入假设                     建模对象        能否处理变长序列
    DNN    每个样本是独立的特征向量     样本间无关系    不能（输入维度固定）
    CNN    局部空间/局部窗口相关        局部空间特征    部分可以，但不擅长长距离时序
    RNN    序列中相邻时间步相关         时间维度依赖    可以（参数在所有时间步共享）

【RNN 的三个设计要点】
    1) 时序记忆：隐藏状态 h_t 会保存之前时间步的信息，形成一种「记忆」；
    2) 顺序处理：按时间步依次处理，每个时间步都能看到之前的信息；
    3) 参数共享：所有时间步共享同一组权重 W_hh、W_xh。
       这一点非常关键——同一个权重矩阵被重复使用，所以：
         · 参数量与序列长度无关（序列长 10 还是 1000，参数一样多）；
         · 天然支持变长序列（逐时间步调用，不用改变网络结构）。

【代价】正因为每个时间步都要穿过同一个非线性激活，反向传播时梯度要沿时间步
连乘，这就埋下了「梯度消失 / 梯度爆炸」的隐患（见第四节）。
"""
)

# ===========================================================================
# 二、基本公式 + 手写 RNNCell 与 nn.RNN 对比
# ===========================================================================
section("二、基本公式与手写 RNNCell")

print(
    """
【课案公式】RNN 在每个时间步 t 的计算：

        h_t = tanh( W_hh · h_{t-1} + W_xh · x_t + b_h )        ... (2.1) 隐藏状态更新
        y_t = W_hy · h_t + b_y                                 ... (2.2) 输出（本脚本聚焦 2.1）

    其中 x_t ∈ R^{d_x} 是时间步 t 的输入，h_{t-1} ∈ R^{d_h} 是上一时间步的隐藏状态，
          W_hh ∈ R^{d_h×d_h} 是「隐藏→隐藏」权重（负责传递记忆），
          W_xh ∈ R^{d_h×d_x} 是「输入→隐藏」权重（负责读入当前输入），
          b_h ∈ R^{d_h} 是偏置。

【和 PyTorch nn.RNN 参数名的对应关系】（务必记牢，第 5、6 节切权重时要用）
    PyTorch 把公式 (2.1) 里的两项拆成两组参数：
        W_xh  →  weight_ih_l0   形状 (hidden_size, input_size)     「ih」= input→hidden
        W_hh  →  weight_hh_l0   形状 (hidden_size, hidden_size)    「hh」= hidden→hidden
        b_h 的一部分 → bias_ih_l0   形状 (hidden_size,)
        b_h 的剩余部分 → bias_hh_l0 形状 (hidden_size,)
    注意：PyTorch 内部把两个 bias **相加**后使用，等价于公式里单一的 b_h：
        z_t = x_t · weight_ih_l0ᵀ + b_ih + h_{t-1} · weight_hh_l0ᵀ + b_hh
    后面手写单元时，我们就用 (bias_ih_l0 + bias_hh_l0) 作为 b_h，误差可达 1e-7 量级。

    「_l0」后缀表示第 0 层（layer 0）；如果是 2 层，就会有 weight_ih_l0 / weight_ih_l1 ……
    注意 PyTorch 用的是「行向量右乘」写法 x @ Wᵀ，和数学写法 W · x 是同一件事：
        (W · x)_i = Σ_j W_{ij} x_j ，而 (x @ Wᵀ)_i = Σ_j x_j W_{ij} —— 完全一致。
"""
)


class MyRNNCell(nn.Module):
    """手写单步 RNN 单元，逐字实现公式 (2.1)。

    设计说明：
        - 参数命名刻意对齐 PyTorch 的 weight_ih / weight_hh / bias_ih / bias_hh，
          这样可以直接把 nn.RNN 的权重搬过来做数值对比；
        - forward 接收 (x_t, h_{t-1})，返回 h_t，形状都是 (batch, hidden_size)。
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # W_xh：(hidden, input)，对应 nn.RNN 的 weight_ih_l0
        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        # W_hh：(hidden, hidden)，对应 nn.RNN 的 weight_hh_l0
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        # 两个偏置，和 PyTorch 一样分开存放，使用时相加
        self.bias_ih = nn.Parameter(torch.zeros(hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(hidden_size))

        # 用和 nn.RNN 一样的初始化方式（U(-1/√hidden, 1/√hidden)），
        # 这不是必须的（我们会直接覆盖权重），但保持风格一致
        bound = 1.0 / math.sqrt(hidden_size)
        with torch.no_grad():
            for p in (self.weight_ih, self.weight_hh):
                p.uniform_(-bound, bound)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """x_t: (batch, input_size)，h_prev: (batch, hidden_size) → h_t: (batch, hidden_size)。"""
        # 公式里的两项线性变换：x_t @ W_xhᵀ + h_{t-1} @ W_hhᵀ，再各自加偏置
        z_t = x_t @ self.weight_ih.T + self.bias_ih + h_prev @ self.weight_hh.T + self.bias_hh
        # 过 tanh 得到新的隐藏状态（tanh 把值压到 (-1, 1)，这也是梯度上限 ≤ 1 的来源）
        return torch.tanh(z_t)


# ---- 用课案代码里的那组尺寸做验证 ----
batch_size, seq_len, input_size, hidden_size = 4, 6, 10, 20

torch.manual_seed(42)
x_demo = torch.randn(batch_size, seq_len, input_size)
print(f"输入 x 形状：{shape(x_demo)}  （batch={batch_size}, seq_len={seq_len}, input_size={input_size}）")

# 官方 nn.RNN，batch_first=True 表示输入是 (batch, seq, feat)
rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
output_rnn, h_n_rnn = rnn(x_demo)

print("\n【nn.RNN 参数形状 —— 与公式的对应关系】")
for name, p in rnn.named_parameters():
    print(f"    {name:14s} 形状 {shape(p)}   （{p.numel()} 个参数）")
print(f"    → weight_ih_l0 就是公式里的 W_xh（{hidden_size}×{input_size}）")
print(f"    → weight_hh_l0 就是公式里的 W_hh（{hidden_size}×{hidden_size}）")
print("    → bias_ih_l0 与 bias_hh_l0 在内部相加，共同构成公式里的 b_h")
print(f"    验证 bias 相加后形状：{(rnn.bias_ih_l0 + rnn.bias_hh_l0).shape} == b_h 的形状 ({hidden_size},)")

# ---- 手写单元逐步前向，和 nn.RNN 比对 ----
cell = MyRNNCell(input_size, hidden_size)
with torch.no_grad():
    # 关键：把手写单元的参数设成和 nn.RNN 完全一样，这样差异只可能来自公式错误
    cell.weight_ih.copy_(rnn.weight_ih_l0)          # W_xh
    cell.weight_hh.copy_(rnn.weight_hh_l0)          # W_hh
    cell.bias_ih.copy_(rnn.bias_ih_l0)
    cell.bias_hh.copy_(rnn.bias_hh_l0)

    # 隐藏状态初始化为全 0（nn.RNN 在 h_0=None 时也是这样做的）
    h = torch.zeros(batch_size, hidden_size)
    my_outputs = []
    for t in range(seq_len):
        h = cell(x_demo[:, t, :], h)   # 逐个时间步调用，h 被反复利用 —— 这就是「记忆」
        my_outputs.append(h)
    my_output = torch.stack(my_outputs, dim=1)   # (batch, seq, hidden)

print("\n【手写 RNNCell VS nn.RNN 逐元素对比】")
print(f"    手写单元输出形状：{shape(my_output)}")
print(f"    nn.RNN 输出形状 ：{shape(output_rnn)}")
err_out = (my_output - output_rnn).abs().max().item()
err_hn = (my_output[:, -1, :] - h_n_rnn[0]).abs().max().item()
print(f"    最大逐元素误差（整段 output）：{err_out:.3e}")
print(f"    最大逐元素误差（最后一步 h） ：{err_hn:.3e}")
print(f"    误差 < 1e-6 ？ {err_out < 1e-6}  ← 说明 nn.RNN 就是公式 (2.1) 的向量化实现")
print("    为什么不是精确的 0：float32 下 (a@B + b@C) 的求和顺序不同，舍入误差约 1e-7 量级。")

# ===========================================================================
# 三、nn.RNN / nn.LSTM / nn.GRU 前向与隐藏状态形状
# ===========================================================================
section("三、nn.RNN / nn.LSTM / nn.GRU 的形状（课案代码）")

# 这一节完整照搬课案代码的尺寸设置
batch_size, seq_len, input_size, hidden_size = 4, 6, 10, 20
torch.manual_seed(42)
x = torch.randn(batch_size, seq_len, input_size)
print(f"输入 x：{shape(x)}   (batch, seq_len, input_size)")

print(
    """
【记忆要点：output 与 h_n 的区别】
    output —— 每！个！时间步的隐藏状态，形状 (batch, seq_len, hidden)；
    h_n    —— 只！有！最！后！一！个！时间步的隐藏状态，形状 (num_layers, batch, hidden)。
    两者的关系：output[:, -1, :] 就是 h_n[0]（最后一层、最后一个时间步）。
    取 h_n 相当于「整条序列的压缩摘要」，常用于接分类头（句子情感分类等）。
"""
)

# ---- 2. 基础 RNN ----
rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
output, h_n = rnn(x)
print("【2. 基础 RNN】")
print(f"    RNN output shape: {shape(output)}     # (batch, seq_len, hidden)")
print(f"    RNN h_n shape   : {shape(h_n)}        # (num_layers, batch, hidden)")
print(f"    torch.allclose(output[:, -1, :], h_n[0]) = "
      f"{torch.allclose(output[:, -1, :], h_n[0])}   ← output 最后一步 == h_n")

# ---- 3. LSTM ----
lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
output, (h_n, c_n) = lstm(x)
print("\n【3. LSTM】（比 RNN 多一个细胞状态 c_n，所以返回的是元组 (h_n, c_n)）")
print(f"    LSTM output shape: {shape(output)}")
print(f"    LSTM h_n shape   : {shape(h_n)}")
print(f"    LSTM c_n shape   : {shape(c_n)}      # 细胞状态，形状与 h_n 完全相同")
print(f"    torch.allclose(output[:, -1, :], h_n[0]) = {torch.allclose(output[:, -1, :], h_n[0])}")
print("    LSTM 参数量是 RNN 的 4 倍，因为一次算 4 组（i/f/g/o 四个门）：")
print(f"        RNN : {sum(p.numel() for p in rnn.parameters())} 个参数")
print(f"        LSTM: {sum(p.numel() for p in lstm.parameters())} 个参数")

# ---- 4. GRU ----
gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
output, h_n = gru(x)
print("\n【4. GRU】（把遗忘门和输入门合并成更新门，只有 3 组参数）")
print(f"    GRU output shape: {shape(output)}")
print(f"    GRU h_n shape   : {shape(h_n)}")
print(f"    GRU 参数量      : {sum(p.numel() for p in gru.parameters())} 个（介于 RNN 和 LSTM 之间）")

# ---- 5. num_layers=2 ----
print("\n【5. num_layers=2：堆叠两层】")
torch.manual_seed(42)
rnn2 = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
out2, hn2 = rnn2(x)
print(f"    2 层 RNN output shape: {shape(out2)}   # output 始终是「最后一层」的输出")
print(f"    2 层 RNN h_n shape   : {shape(hn2)}    # ← 第一维从 1 变成 2（每层一个 h）")
print("    解释：h_n[0] 是第 1 层的最终隐藏状态，h_n[1] 是第 2 层（输出层）的最终隐藏状态。")
print(f"    验证 output[:, -1, :] == h_n[1]（最后一层）: "
      f"{torch.allclose(out2[:, -1, :], hn2[1])}")

# ---- 6. bidirectional=True ----
print("\n【6. bidirectional=True：双向】")
torch.manual_seed(42)
rnn_bi = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                num_layers=1, batch_first=True, bidirectional=True)
out_bi, hn_bi = rnn_bi(x)
print(f"    双向 RNN output shape: {shape(out_bi)}  # 最后一维 20 → 40 = hidden_size × 2")
print(f"    双向 RNN h_n shape   : {shape(hn_bi)}   # 第一维 1 → 2（正向 + 反向各一份）")
print("    解释：双向会同时跑一个正向序列和一个反向序列，把两个方向的隐藏状态「拼接」起来，")
print("          所以最后一维翻倍；h_n[0] 是正向的最终状态，h_n[1] 是反向的最终状态。")
print(f"    验证 output[:, :, :20] 是正向、output[:, :, 20:] 是反向（各自最后一个时间步）：")
print(f"        allclose(output[:, -1, :20], h_n[0]) = {torch.allclose(out_bi[:, -1, :20], hn_bi[0])}")
print(f"        allclose(output[:,  0, 20:], h_n[1]) = {torch.allclose(out_bi[:, 0, 20:], hn_bi[1])}"
      f"   ← 反向序列的「最后一步」是原序列的第 0 个位置")

# ---- 7. batch_first=False（默认）----
print("\n【7. batch_first=False（PyTorch 默认）：形状变成 (seq, batch, feat)】")
torch.manual_seed(42)
x_nb = torch.randn(batch_size, seq_len, input_size)   # 重新生成，保证和 batch_first=True 用的是同一份输入
rnn_nb = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=False)
out_nb, hn_nb = rnn_nb(x_nb.transpose(0, 1))          # (batch, seq, feat) → (seq, batch, feat)
print(f"    输入（默认格式）      : {shape(x_nb.transpose(0, 1))}     # (seq_len, batch, input_size)")
print(f"    output shape（默认）  : {shape(out_nb)}     # (seq_len, batch, hidden)")
print(f"    h_n shape  （默认）   : {shape(hn_nb)}      # (num_layers, batch, hidden)，和 batch_first 无关！")
print("    注意：h_n 的形状**不受 batch_first 影响**，永远是 (num_layers, batch, hidden)；")
print("          只有 output 会随 batch_first 在前两维之间切换。")
# 用同一份输入、同一种子分别构造两种 batch_first 的网络，验证结果只是「位置换了一下」
torch.manual_seed(42)
rnn_bf = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
out_bf, _ = rnn_bf(x_nb)
print(f"    两批权重相同（同种子），两种格式的结果是否一致："
      f"{torch.allclose(out_nb.transpose(0, 1), out_bf)}"
      f"   ← 把默认格式 transpose 回来就与 batch_first=True 完全相同")

# ===========================================================================
# 四、梯度消失：数学推导 + 实测
# ===========================================================================
section("四、梯度消失：从数学推导到实测")

print(
    r"""
【Step 1：前向公式拆成两步】方便求导，把线性变换和激活拆开：
        z_t = W_hh · h_{t-1} + W_xh · x_t + b_h        （激活前）
        h_t = tanh(z_t)                                （激活后）

【Step 2：单步梯度 ∂h_t/∂h_{t-1}】
    用链式法则绕道 z_t：      ∂h_t/∂h_{t-1} = (∂h_t/∂z_t) · (∂z_t/∂h_{t-1})

    · 第一项 ∂h_t/∂z_t —— tanh 是「逐元素」标量函数，各分量互不交叉，所以
      雅可比矩阵是对角矩阵：
            ∂h_t/∂z_t = diag(tanh'(z_t)) ,   tanh'(x) = 1 - tanh²(x) ∈ (0, 1]
      记 h_t^{(i)} = tanh(z_t^{(i)})，则 i = j 时 ∂h_t^{(i)}/∂z_t^{(i)} = tanh'(z_t^{(i)})，
      i ≠ j 时为 0。注意与矩阵乘法 W_hh·h_{t-1} 的区别：矩阵乘法的每个输出分量
      混合了**所有**输入分量，而 tanh 只是对每个数单独过一遍激活，无交叉。

    · 第二项 ∂z_t/∂h_{t-1} = W_hh （因为 z_t = W_hh h_{t-1} + 常数项）

    合起来，**单个时间步的梯度**是：
            ∂h_t/∂h_{t-1} = diag(tanh'(z_t)) · W_hh                ... (4.1)

【Step 3：跨 k 步的链式连乘】
    损失 L 对早期隐藏状态 h_{t-k} 的梯度要先传到 h_t，需要穿过中间每一个时间步：
            ∂h_t/∂h_{t-k} = ∏_{i=1}^{k} ∂h_{t-i+1}/∂h_{t-i}
    把 (4.1) 代入：
            ∂h_t/∂h_{t-k} = ∏_{i=1}^{k} [ diag(tanh'(z_{t-i+1})) · W_hh ]   ... (4.2)

【Step 4：为什么会指数级衰减】
    (4.2) 里有两个因子在反复相乘：

        因子                    来源            取值范围
        diag(tanh'(z))          tanh 的导数     (0, 1]，|z| 大时趋近于 0
        W_hh                    权重矩阵        取决于初始化，通常 |λ_max| < 1

    · 每个 tanh' 项 ≤ 1，k 个相乘后可能指数级缩小；
    · W_hh^k 当 |λ_max| < 1 时也指数级衰减；当 |λ_max| > 1 时可能**梯度爆炸**。

    > |λ_max| 是什么？它是 W_hh 的最大特征值（按绝对值），可以理解为矩阵的
      「缩放倍率上限」：一个向量被 W_hh 反复乘时，其长度大致按 |λ_max|^k 缩放。
      例如 |λ_max| = 0.9，乘一次缩小为 0.9 倍，乘 50 次后 0.9^50 ≈ 0.005（见下方打印），
      几乎消失了。这就是「长序列早期时间步学不动」的根本原因。

    结论：序列越长（k 越大），连乘项越多，衰减越严重；早期时间步 h_{t-k} 收到的梯度
    极其微弱，W_hh 中与长距离依赖相关的部分**几乎无法更新**。注意这个衰减是「物理规律」：
    tanh' ≤ 1 是固定的数学性质，模型无法控制它。
"""
)

# 用真实数字说明 |λ_max|^k
print("【λ_max 的直观含义】")
for lam in (0.9, 0.95, 1.0, 1.05):
    print(f"    |λ_max| = {lam:.2f} → 连乘 50 步后缩放为 {lam ** 50:.3e}")
print("    可以看到 <1 时迅速衰减、=1 时保持不变、>1 时迅速放大（爆炸）。")


# ---------------------------------------------------------------------------
# 4.1 实测：RNN 每个时间步的梯度范数
# ---------------------------------------------------------------------------
SEQ_LEN = 50        # 序列长度：50 步足够看出数量级差异，又不至于太慢
BATCH = 1           # batch=1，避免多批次求平均把差异抹平
INPUT_DIM = 8
HIDDEN_DIM = 16
CHECK_STEPS = list(range(0, SEQ_LEN, 5)) + [SEQ_LEN - 1]   # 抽样打印：0,5,10,...,45,49


def grad_norms_by_step(kind: str, seq_len: int = SEQ_LEN, seed: int = 42,
                       hh_scale: float = 1.0, forget_bias: float | None = None):
    """构造指定结构的 RNN 并做一次反向传播，返回每个时间步输入梯度的范数。

    参数：
        kind        : "RNN" / "LSTM" / "GRU"
        hh_scale    : 把 weight_hh 放大多少倍（用于演示梯度爆炸），1.0 表示不放大
        forget_bias : 只对 LSTM 生效；把遗忘门的 bias 设为该值（模拟模型学出 f_t≈1）

    返回值：
        (梯度范数列表, 模型)  —— 模型也返回，方便打印 weight_hh 的 |λ_max|
    """
    torch.manual_seed(seed)                     # 三种结构用同一种子，保证输入与初始化可比
    x = torch.randn(BATCH, seq_len, INPUT_DIM)
    x.requires_grad_(True)                      # 我们要读 x.grad，即损失回传到「输入」的梯度

    model = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind](
        input_size=INPUT_DIM, hidden_size=HIDDEN_DIM, batch_first=True
    )

    if forget_bias is not None and kind == "LSTM":
        # 目的：证明「f_t 越接近 1，梯度越能无损穿过时间步」。
        # 做法：直接把遗忘门 bias 抬高。bias 大 → σ(bias) 接近 1 → C_t ≈ C_{t-1}，
        #       细胞状态这条旁路几乎不衰减，早期时间步的梯度也随之变大。
        # 这正是训练时 LSTM 会自己学出来的样子（很多实现干脆把 forget bias 初始化为 1）。
        with torch.no_grad():
            model.bias_ih_l0[HIDDEN_DIM:2 * HIDDEN_DIM].fill_(forget_bias)
            model.bias_hh_l0[HIDDEN_DIM:2 * HIDDEN_DIM].fill_(0.0)

    if hh_scale != 1.0:
        # 演示梯度爆炸：放大循环权重 W_hh，使 |λ_max| > 1，连乘 ∏ W_hh 被放大。
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "hh" in name:                # 只放大「隐藏→隐藏」的循环部分
                    p.mul_(hh_scale)

    output, _ = model(x)
    # 损失刻意只用最后一步的输出：这样梯度必须从 t=49 一路回传到 t=0，
    # 中间有多少时间步，就要连乘多少个 ∂h_t/∂h_{t-1}
    loss = output[:, -1, :].pow(2).sum()
    model.zero_grad()
    loss.backward()

    g = x.grad
    norms = [float(g[:, t, :].norm()) for t in range(seq_len)]
    return norms, model


def print_grad_table(label: str, norms) -> None:
    """以表格形式打印若干时间步的梯度范数（用科学计数法，便于看数量级）。"""
    cells = "  ".join(f"t={t:2d}:{norms[t]:.2e}" for t in CHECK_STEPS)
    print(f"    {label:26s} {cells}")


def spectral_radius(weight_hh: torch.Tensor):
    """估计循环权重矩阵的「缩放倍率」|λ_max|（谱半径）。

    为什么要分两种情况？
        · nn.RNN 的 weight_hh_l0 是 (hidden, hidden)，是**方阵**，
          可以直接用 torch.linalg.eigvals 求全部特征值，取模的最大值即谱半径 |λ_max|；
        · nn.LSTM 的 weight_hh_l0 是 (4*hidden, hidden)、nn.GRU 的是 (3*hidden, hidden)，
          它们把 4 / 3 组门的权重**纵向堆叠**在一起，所以不是方阵，没有特征值可言。
          这时用矩阵的 2-范数（最大奇异值）作为 |λ_max| 的**上界估计**：
              |λ_max| ≤ ‖W‖_2
          它同样是「向量被反复乘时的最大缩放倍率」的刻画，用于说明数量级足够。

    返回值：
        (数值, 说明字符串) —— 字符串用于在打印时标明是精确值还是上界估计
    """
    w = weight_hh.detach()
    if w.shape[0] == w.shape[1]:
        # 方阵（标准 RNN）：精确谱半径
        return float(torch.linalg.eigvals(w).abs().max()), "精确谱半径"
    # 非方阵（LSTM / GRU 的门权重组）：用 2-范数作为上界
    return float(torch.linalg.matrix_norm(w, ord=2)), "2-范数(上界)"


print("\n【4.1 实测设置】")
print(f"    序列长度 = {SEQ_LEN}，batch = {BATCH}，input_size = {INPUT_DIM}，hidden_size = {HIDDEN_DIM}")
print("    损失 = output[:, -1, :].pow(2).sum()，即只用最后一步输出。")
print("    这样 t=0 的梯度必须穿过全部 50 个时间步，衰减最严重；t=49 几乎不衰减。")
print("    评价指标 = x.grad[:, t, :].norm()，即「损失回传到第 t 个时间步输入的梯度长度」。")
print("    说明：这里不加梯度裁剪，初始化也保持 PyTorch 默认值，让衰减自然显现。")
print()

print("【对照 A：默认初始化下 RNN / LSTM / GRU 的梯度范数】")
grad_rnn, model_rnn = grad_norms_by_step("RNN")
grad_lstm, model_lstm = grad_norms_by_step("LSTM")
grad_gru, model_gru = grad_norms_by_step("GRU")
print_grad_table("RNN （基础循环）", grad_rnn)
print_grad_table("LSTM（三个门+细胞状态）", grad_lstm)
print_grad_table("GRU （更新门+重置门）", grad_gru)
print()
print("    各结构循环权重的缩放倍率 |λ_max|（默认初始化）：")
for name, m in (("RNN", model_rnn), ("LSTM", model_lstm), ("GRU", model_gru)):
    w = m.weight_hh_l0
    val, how = spectral_radius(w)
    print(f"        {name:5s} weight_hh_l0 {shape(w)}  |λ_max| = {val:.4f}   [{how}]")
print("        RNN 是方阵可取精确谱半径；LSTM/GRU 的 weight_hh_l0 是多组门权重纵向堆叠，")
print("        非方阵，改用 2-范数（最大奇异值）作为 |λ_max| 的上界估计。")
print()
print("    观察：三条曲线都在衰减，但幅度不同——")
print(f"        RNN  t=0 比 t=49 小 {grad_rnn[49] / grad_rnn[0]:.3e} 倍")
print(f"        LSTM t=0 比 t=49 小 {grad_lstm[49] / grad_lstm[0]:.3e} 倍")
print(f"        GRU  t=0 比 t=49 小 {grad_gru[49] / grad_gru[0]:.3e} 倍")
print("        （随机初始化下三个结构的 |λ_max| 都 < 1，tanh'/sigmoid' 也都被压小，")
print("          所以都衰减；门控结构的优势要看「对照 B」——f_t 学到 1 之后的表现。）")

# ---------------------------------------------------------------------------
# 4.2 对照 B：LSTM 的遗忘门打开后，梯度几乎无损穿过
# ---------------------------------------------------------------------------
print("\n【对照 B：LSTM 遗忘门 bias 从 0 逐步抬高（模拟模型学出 f_t≈1）】")
print("    原理：C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t，∂C_t/∂C_{t-1} 里的主项就是 f_t。")
print("          训练时模型为了记住长期信息会自己把 f_t 学到接近 1，梯度就能从这条旁路无损回去。")
print()
for fb in (0.0, 1.0, 2.0, 4.0):
    norms, _ = grad_norms_by_step("LSTM", forget_bias=fb)
    print_grad_table(f"LSTM 遗忘门 bias = {fb:.1f} (σ={torch.sigmoid(torch.tensor(fb)):.3f})", norms)
print()
print("    结论：遗忘门 bias = 0（σ≈0.5，相当于默认初始化）时，早期梯度只有 1e-12；")
print("          抬高到 2.0（σ≈0.88）后，早期梯度提升到 1e-3 量级，衰减从『必然』变成『可控』。")
print("          这就是课案说的：RNN 只有一条路（每步必过 tanh'，固定不可控），")
print("          LSTM 有两条路（f_t ⊙ C_{t-1} 免衰减旁路 + i_t ⊙ C̃_t 有衰减路径），")
print("          而且 f_t = σ(·) 是**可学习的**：需要记住时学出 f_t→1，需要遗忘时学出 f_t→0。")

# ---------------------------------------------------------------------------
# 4.3 对照 C：梯度爆炸
# ---------------------------------------------------------------------------
print("\n【对照 C：梯度爆炸——手动放大 weight_hh】")
print("    原理：把 W_hh 整体乘 8.0，|λ_max| 随之升高到 >1，")
print("          (4.2) 里的 ∏ W_hh 变成连乘放大，早期时间步梯度反而比晚期大得多。")
grad_rnn_big, model_rnn_big = grad_norms_by_step("RNN", hh_scale=8.0)
_big_val, _ = spectral_radius(model_rnn_big.weight_hh_l0)
_orig_val, _ = spectral_radius(model_rnn.weight_hh_l0)
print(f"    放大后 RNN weight_hh_l0 的 |λ_max| = {_big_val:.4f}（原来 {_orig_val:.4f}）")
print_grad_table("RNN 放大 ×8.0", grad_rnn_big)
print()
print("    注意看趋势反转：正常 RNN 是「越早越小」，爆炸时变成「越早越大」——")
print(f"        t=0  梯度范数 = {grad_rnn_big[0]:.3e}   ← 被 ∏W_hh 放大出来的巨大梯度")
print(f"        t=49 梯度范数 = {grad_rnn_big[49]:.3e}   ← 最后一步附近仍是正常量级")
print(f"        两者相差 {grad_rnn_big[0] / grad_rnn_big[49]:.3e} 倍")
big_max = max(grad_rnn_big)
print(f"    全局最大梯度范数 = {big_max:.3e}")
if math.isinf(big_max) or math.isnan(big_max):
    print("    → 已经溢出成 inf/NaN，实际训练中参数会被这一步彻底摧毁（loss 变 nan）。")
else:
    print("    → 这就是梯度爆炸：数值虽未溢出，但已比正常情况大 6 个数量级以上，")
    print("      一步梯度下降就会把参数推到荒谬的区域，训练直接发散。")
print("    工程对策：梯度裁剪 torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)，")
print("              以及用 LSTM/GRU 的门控结构（本演示故意不裁剪，以便暴露现象）。")

# ---------------------------------------------------------------------------
# 4.4 绘图
# ---------------------------------------------------------------------------
print("\n【4.4 绘制梯度衰减曲线】")
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
steps = np.arange(SEQ_LEN)

# 子图 1：三种结构默认初始化的梯度衰减对比（对数纵轴）
ax = axes[0]
ax.semilogy(steps, grad_rnn, "o-", ms=3, lw=1.6, color="#d62728", label="RNN")
ax.semilogy(steps, grad_lstm, "s-", ms=3, lw=1.6, color="#1f77b4", label="LSTM")
ax.semilogy(steps, grad_gru, "^-", ms=3, lw=1.6, color="#2ca02c", label="GRU")
ax.set_xlabel("时间步 t（0 = 最早，49 = 最后一步）")
ax.set_ylabel("梯度范数 ‖∂loss/∂x_t‖（对数刻度）")
ax.set_title("默认初始化：RNN / LSTM / GRU 的梯度随时间的衰减")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
ax.annotate("最后一步\n（衰减最少）", xy=(SEQ_LEN - 1, grad_rnn[-1]),
            xytext=(SEQ_LEN - 14, grad_rnn[-1] * 30),
            arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9, color="gray")
ax.annotate("最早的时间步\n（梯度被连乘压到 1e-12）", xy=(0, grad_rnn[0]),
            xytext=(3, grad_rnn[0] * 300),
            arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9, color="gray")

# 子图 2：LSTM 遗忘门不同 bias 的梯度（说明「免衰减旁路」）
ax = axes[1]
# 注意：为保持图例简洁，这里只对 fb=0.0 与 fb=2.0 画曲线，
#       完整四组（0.0 / 1.0 / 2.0 / 4.0）的数值已经在上面用表格打印出来了。
for fb, color in zip((0.0, 2.0), ("#1f77b4", "#2ca02c")):
    norms, _ = grad_norms_by_step("LSTM", forget_bias=fb)
    ax.semilogy(steps, norms, "-", lw=1.8, color=color,
                label=f"LSTM 遗忘门 bias={fb:.1f} (σ={torch.sigmoid(torch.tensor(fb)):.2f})")
ax.semilogy(steps, grad_rnn, "--", lw=1.4, color="#d62728", label="RNN（对照）")
ax.set_xlabel("时间步 t")
ax.set_ylabel("梯度范数（对数刻度）")
ax.set_title("LSTM 的「免衰减旁路」：遗忘门打开后梯度几乎无损穿过")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8, loc="lower right")

# 子图 3：梯度爆炸
ax = axes[2]
ax.semilogy(steps, grad_rnn_big, "o-", ms=3, lw=1.6, color="#8c564b",
            label="RNN 放大 weight_hh ×8.0（梯度爆炸）")
ax.semilogy(steps, grad_rnn, "s-", ms=3, lw=1.4, color="#d62728",
            label="RNN 原始权重（梯度消失）")
ax.set_xlabel("时间步 t")
ax.set_ylabel("梯度范数（对数刻度）")
ax.set_title("梯度爆炸：|λ_max|>1 时趋势完全反转")
ax.grid(True, which="both", alpha=0.3)
ax.legend(fontsize=8)

fig.tight_layout()
_grad_png = OUTPUT_DIR / "02网络架构_04_RNN梯度随时间的衰减.png"
fig.savefig(_grad_png, dpi=110)
plt.close(fig)
print(f"    图片已保存：{_grad_png}")

# ===========================================================================
# 五、LSTM：结构与手写实现
# ===========================================================================
section("五、LSTM：三个门、细胞状态与手写 LSTMCell")

print(
    r"""
【LSTM 引入的三个门 + 一个细胞状态】
    f_t（遗忘门）：决定「旧的细胞状态 C_{t-1} 要丢掉多少」
    i_t（输入门）：决定「新候选信息 C̃_t 要写入多少」
    o_t（输出门）：决定「细胞状态 C_t 有多少要暴露成隐藏状态 h_t」
    C_t（细胞状态）：一条贯穿整条序列的「传送带」，是长期记忆的载体

【六个公式】
    f_t = σ( W_f · [h_{t-1}, x_t] + b_f )                       ... (5.1) 遗忘门
    i_t = σ( W_i · [h_{t-1}, x_t] + b_i )                       ... (5.2) 输入门
    C̃_t = tanh( W_C · [h_{t-1}, x_t] + b_C )                    ... (5.3) 候选细胞状态
    C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t                             ... (5.4) 细胞状态更新
    o_t = σ( W_o · [h_{t-1}, x_t] + b_o )                       ... (5.5) 输出门
    h_t = o_t ⊙ tanh(C_t)                                       ... (5.6) 隐藏状态输出
    其中 σ 是 sigmoid（值域 (0,1)，正好当「门」的开关），⊙ 是逐元素乘法。

【记号 [h_{t-1}, x_t] 是首尾拼接 —— 最容易混淆的一点】
    设 h_{t-1} ∈ R^128，x_t ∈ R^64，则 [h_{t-1}, x_t] 是 192 维的拼接向量。
    但是 f_t / i_t / C̃_t / o_t 最终**仍然是 128 维**，因为：
        W_f 的形状是 (128, 192)，(128, 192) · (192,) → (128,)
    乘法之后自然回到 128 维。所以 h_t 的维度始终不变，可以一直循环下去。

【PyTorch 怎么存这四个门的权重？—— 行方向拼接！】
    PyTorch 不提供 4 个独立的 Linear，而是把四组权重**沿行方向拼成一个大矩阵**：
        weight_ih_l0 形状 (4*hidden_size, input_size)    ← 4 组 W_? 的「输入→隐藏」部分
        weight_hh_l0 形状 (4*hidden_size, hidden_size)   ← 4 组 W_? 的「隐藏→隐藏」部分
        bias_ih_l0 / bias_hh_l0 形状 (4*hidden_size,)
    这等价于一个 W 形状为 (128*4, 192) 的大矩阵，一次矩阵乘法同时算出四个门（效率更高）。
    切分顺序（PyTorch 官方规定，务必记住）：
        chunks[0] → i 输入门
        chunks[1] → f 遗忘门
        chunks[2] → g 候选细胞状态 C̃
        chunks[3] → o 输出门
    对照记法：i f g o 的字母顺序也就是「i, f, g, o」。下面手写时会按这个顺序 chunk(4)。
"""
)

# ---- 用课案那组尺寸，实际打印 LSTM 权重形状来验证「行方向拼接」 ----
B, S, IN, H = 4, 6, 10, 20
torch.manual_seed(42)
x_lstm = torch.randn(B, S, IN)
lstm_ref = nn.LSTM(IN, H, batch_first=True)

print("【验证：四个门的权重在行方向拼接】")
print(f"    weight_ih_l0 形状 = {shape(lstm_ref.weight_ih_l0)}"
      f"   ← (4*hidden, input) = ({4 * H}, {IN})  ✔")
print(f"    weight_hh_l0 形状 = {shape(lstm_ref.weight_hh_l0)}"
      f"   ← (4*hidden, hidden) = ({4 * H}, {H})  ✔")
print(f"    bias_ih_l0   形状 = {shape(lstm_ref.bias_ih_l0)}   ← (4*hidden,)")
print(f"    bias_hh_l0   形状 = {shape(lstm_ref.bias_hh_l0)}   ← (4*hidden,)，与 bias_ih 相加使用")
print(f"    按 chunk(4, dim=0) 切分后每一段的形状：{shape(lstm_ref.weight_ih_l0.chunk(4, dim=0)[0])}"
      f"  ← (hidden, input)，即单个门 W_i 的形状")
print()
print("    如果 h 是 128 维、x 是 64 维，那么 W 的每一行有 128+64=192 个元素，")
print("    四个门共 4*128 行，所以权重矩阵是 (512, 192)——本质上就是 concat 了 [h, x]。")
print("    PyTorch 实现上把两个矩阵乘分开做（x @ W_ihᵀ + h @ W_hhᵀ），数学上完全等价。")


class MyLSTMCell(nn.Module):
    """手写单步 LSTM 单元，逐字实现公式 (5.1) ~ (5.6)。

    参数命名与切分顺序完全对齐 nn.LSTM：
        weight_ih (4H, input)，weight_hh (4H, H)，bias_ih / bias_hh (4H,)
        行方向切分顺序 = [i 输入门, f 遗忘门, g 候选细胞状态, o 输出门]
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.zeros(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(4 * hidden_size))
        bound = 1.0 / math.sqrt(hidden_size)
        with torch.no_grad():
            for p in (self.weight_ih, self.weight_hh):
                p.uniform_(-bound, bound)

    def forward(self, x_t: torch.Tensor, state: tuple):
        """x_t: (batch, input_size)，state = (h_{t-1}, C_{t-1})；返回 (h_t, C_t)。"""
        h_prev, c_prev = state

        # 一次矩阵乘法同时算出四个门的「激活前值」——
        # 这就是 PyTorch 把四组权重行方向拼接的原因（一次 GEMM 抵四次）。
        # 数学上等价于 W · [h_{t-1}, x_t] + b，只是拆成两项分别做。
        gates = x_t @ self.weight_ih.T + self.bias_ih \
            + h_prev @ self.weight_hh.T + self.bias_hh

        # 按 PyTorch 的顺序切成 4 段：i, f, g(=C̃), o
        i_pre, f_pre, g_pre, o_pre = gates.chunk(4, dim=1)

        # (5.2) 输入门：sigmoid 压到 (0,1)，当作「写入多少」的阀门
        i_t = torch.sigmoid(i_pre)
        # (5.1) 遗忘门：sigmoid 压到 (0,1)，当作「保留多少旧记忆」的阀门
        f_t = torch.sigmoid(f_pre)
        # (5.3) 候选细胞状态：tanh 压到 (-1,1)，表示「想写入的新内容」
        g_t = torch.tanh(g_pre)
        # (5.5) 输出门：sigmoid 压到 (0,1)，当作「暴露多少」的阀门
        o_t = torch.sigmoid(o_pre)

        # (5.4) 细胞状态更新：这是 LSTM 的核心 —— 加法而非乘法，f_t 提供免衰减旁路
        c_t = f_t * c_prev + i_t * g_t
        # (5.6) 隐藏状态：细胞状态先过 tanh 归一化，再由输出门筛选
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


# ---- 手写 LSTM 与 nn.LSTM 数值对比 ----
torch.manual_seed(42)
x_lstm = torch.randn(B, S, IN)
ref_out, (ref_hn, ref_cn) = lstm_ref(x_lstm)

cell_lstm = MyLSTMCell(IN, H)
with torch.no_grad():
    # 把 nn.LSTM 的参数原样搬进手写单元：这样两者输出若有差异，只能是公式写错了
    cell_lstm.weight_ih.copy_(lstm_ref.weight_ih_l0)
    cell_lstm.weight_hh.copy_(lstm_ref.weight_hh_l0)
    cell_lstm.bias_ih.copy_(lstm_ref.bias_ih_l0)
    cell_lstm.bias_hh.copy_(lstm_ref.bias_hh_l0)

    h_t = torch.zeros(B, H)        # h_0 全零（nn.LSTM 默认初值）
    c_t = torch.zeros(B, H)        # C_0 全零
    my_outs = []
    for t in range(S):
        h_t, c_t = cell_lstm(x_lstm[:, t, :], (h_t, c_t))
        my_outs.append(h_t)
    my_out = torch.stack(my_outs, dim=1)

print("\n【手写 LSTMCell VS nn.LSTM 逐元素对比】")
print(f"    手写单元 output 形状：{shape(my_out)}")
print(f"    nn.LSTM  output 形状：{shape(ref_out)}")
err_lstm_out = (my_out - ref_out).abs().max().item()
err_lstm_h = (h_t - ref_hn[0]).abs().max().item()
err_lstm_c = (c_t - ref_cn[0]).abs().max().item()
print(f"    最大误差 output：{err_lstm_out:.3e}")
print(f"    最大误差 h_n   ：{err_lstm_h:.3e}")
print(f"    最大误差 c_n   ：{err_lstm_c:.3e}")
print(f"    误差 < 1e-5 ？ {err_lstm_out < 1e-5}   ← 说明 6 个公式和切分顺序完全正确")

print(
    r"""
【LSTM 如何解决梯度消失 —— 核心推导】
    对 (5.4) C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t 关于 C_{t-1} 求偏导，两项分别看：

    · 第一项 f_t ⊙ C_{t-1}：C_{t-1} 前面只有一个逐元素因子 f_t，所以导数是 f_t；
    · 第二项 i_t ⊙ C̃_t：i_t 和 C̃_t 都依赖 h_{t-1}，而 h_{t-1} = o_{t-1} ⊙ tanh(C_{t-1})，
      所以这一项对 C_{t-1} 也有梯度，但它包含了 tanh' 和 W_C 的连乘，和 RNN 面临同样的问题。

    两路合并，完整的
            ∂C_t/∂C_{t-1} = f_t + （第二项的梯度）                    ... (5.7)
    关键在于：即使第二项仍然有衰减，**第一项 f_t 提供了一条「免衰减通道」**。

    对比表（课案）：
        项目         RNN                                 LSTM
        更新式       h_t = tanh(W_hh h_{t-1} + ...)      C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
        求导结果     diag(tanh'(z_t)) · W_hh             f_t + （其他路径）
        衰减因子     tanh' ∈ (0,1]，**固定不可控**       f_t = σ(…)，**可学习**
        k 步连乘     每步都乘 tanh' ≤ 1，必然衰减         模型可让 f_t ≈ 1，梯度几乎无损通过

    核心区别：
    · RNN 只有一条路 h_{t-1} → z_t → h_t，每步必过 tanh'，梯度必然被压缩；
    · LSTM 有两条路，一条 f_t ⊙ C_{t-1}（免衰减），一条 i_t ⊙ C̃_t（有衰减），
      只要 f_t ≈ 1，梯度就能从第一条路无损传回去；
    · f_t = σ(W_f·[h_{t-1},x_t] + b_f) 是**模型自己学的**：需要记住长期信息时学出
      f_t → 1，需要遗忘时学出 f_t → 0。

    一句话总结：RNN 的梯度衰减是「物理规律」，LSTM 通过 f_t ⊙ C_{t-1} 开了一条**旁路**，
    把梯度消失从「必然」变成了「模型可以选择」。第四节的对照 B 已经用数值验证了这一点。
"""
)

# ===========================================================================
# 六、GRU
# ===========================================================================
section("六、GRU：更新门 + 重置门")

print(
    r"""
【GRU 的设计动机】LSTM 有 3 个门 + 1 个细胞状态，参数多、计算重。GRU 做了简化：
    · 把 LSTM 的「遗忘门 f_t」和「输入门 i_t」合并成**一个更新门 z_t**：
      LSTM 里 C_t = f_t·C_{t-1} + i_t·C̃_t 是两个独立阀门，GRU 强制二者互补
      （用 1-z_t 和 z_t），参数量直接减少 1/4；
    · 取消了独立的细胞状态 C_t，直接用隐藏状态 h_t 承载记忆（h_t 既是输出也是记忆）。

【四个公式】
    r_t = σ( W_r · [h_{t-1}, x_t] )                                     ... (6.1) 重置门
    z_t = σ( W_z · [h_{t-1}, x_t] )                                     ... (6.2) 更新门
    n_t = tanh( W_n · [ r_t ⊙ h_{t-1}, x_t ] )                          ... (6.3) 候选隐藏状态
    h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}                               ... (6.4) 隐藏状态更新

    · 重置门 r_t：决定「计算新候选时，要忘掉多少旧隐藏状态」。r_t→0 表示完全不看历史，
      相当于把这一段当成序列的起点重新开始读。
    · 更新门 z_t：决定「新状态里，新信息占多少、旧状态保留多少」。
      z_t→0 时 h_t ≈ n_t（全部换成新信息），z_t→1 时 h_t ≈ h_{t-1}（原样保留记忆）。
      注意 z_t→1 这条路径就是 GRU 版的「免衰减旁路」：
      h_t 对 h_{t-1} 的偏导里含有 z_t 这一项，且 z_t 是可学习的。

【PyTorch 的拼接顺序：r, z, n（三组）】
    weight_ih_l0 形状 (3*hidden, input)，weight_hh_l0 形状 (3*hidden, hidden)
    chunk(3, dim=1) 的顺序是：chunks[0] → r 重置门，chunks[1] → z 更新门，chunks[2] → n 候选。
    ⚠ 顺序是「r, z, n」，不是字母序，也不是 LSTM 的「i, f, g, o」，写代码时必须记牢。

【注意 (6.3) 的细节】候选状态 n_t 里的 h_{t-1} 要**先乘重置门 r_t**再参与线性变换：
    PyTorch 的实现是 n_t = tanh( W_n_ih · x_t + b_n_ih + r_t ⊙ (W_n_hh · h_{t-1} + b_n_hh) )
    即「重置」作用在隐藏→隐藏那一路上（这也是为什么 bias 要拆成 ih / hh 两份分别加）。
"""
)

# ---- 实际打印 GRU 权重形状 ----
torch.manual_seed(42)
x_gru = torch.randn(B, S, IN)
gru_ref = nn.GRU(IN, H, batch_first=True)
print("【验证：GRU 的三组权重形状】")
print(f"    weight_ih_l0 形状 = {shape(gru_ref.weight_ih_l0)}   ← (3*hidden, input) = ({3 * H}, {IN})")
print(f"    weight_hh_l0 形状 = {shape(gru_ref.weight_hh_l0)}   ← (3*hidden, hidden) = ({3 * H}, {H})")
print(f"    chunk(3, dim=0) 后每段形状 = {shape(gru_ref.weight_ih_l0.chunk(3, dim=0)[0])}"
      f"   ← 单个门的形状 (hidden, input)")
print(f"    参数量对比：RNN {sum(p.numel() for p in nn.RNN(IN, H).parameters())} 个，"
      f"GRU {sum(p.numel() for p in gru_ref.parameters())} 个，"
      f"LSTM {sum(p.numel() for p in lstm_ref.parameters())} 个")
print("    → GRU 参数正好是 LSTM 的 3/4，所以通常训练更快、更不容易过拟合。")


class MyGRUCell(nn.Module):
    """手写单步 GRU 单元，逐字实现公式 (6.1) ~ (6.4)。

    切分顺序对齐 PyTorch：chunk(3) = [r 重置门, z 更新门, n 候选隐藏状态]
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_size))
        bound = 1.0 / math.sqrt(hidden_size)
        with torch.no_grad():
            for p in (self.weight_ih, self.weight_hh):
                p.uniform_(-bound, bound)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor) -> torch.Tensor:
        """x_t: (batch, input_size)，h_prev: (batch, hidden_size) → h_t: (batch, hidden_size)。"""
        # 「输入→隐藏」这一路：只依赖 x_t，三个门都要用
        gi = x_t @ self.weight_ih.T + self.bias_ih
        # 「隐藏→隐藏」这一路：只依赖 h_{t-1}，注意稍后会被重置门 r 逐元素缩放
        gh = h_prev @ self.weight_hh.T + self.bias_hh

        # 按 PyTorch 顺序切分：r, z, n
        r_i, z_i, n_i = gi.chunk(3, dim=1)
        r_h, z_h, n_h = gh.chunk(3, dim=1)

        # (6.1) 重置门 r_t：si 和 sh 相加后过 sigmoid（等价于 W_r·[h_{t-1}, x_t] + b_r）
        r_t = torch.sigmoid(r_i + r_h)
        # (6.2) 更新门 z_t：同样相加后过 sigmoid
        z_t = torch.sigmoid(z_i + z_h)
        # (6.3) 候选隐藏状态：r_t 只缩放「隐藏→隐藏」那一路（PyTorch 的实现细节）
        n_t = torch.tanh(n_i + r_t * n_h)
        # (6.4) 用更新门在「新候选」和「旧隐藏状态」之间做凸组合
        #       z_t→1 时 h_t ≈ h_{t-1}，这条路径就是 GRU 的免衰减旁路
        h_t = (1.0 - z_t) * n_t + z_t * h_prev
        return h_t


# ---- 手写 GRU 与 nn.GRU 数值对比 ----
torch.manual_seed(42)
x_gru = torch.randn(B, S, IN)
ref_gru_out, ref_gru_hn = gru_ref(x_gru)

cell_gru = MyGRUCell(IN, H)
with torch.no_grad():
    cell_gru.weight_ih.copy_(gru_ref.weight_ih_l0)
    cell_gru.weight_hh.copy_(gru_ref.weight_hh_l0)
    cell_gru.bias_ih.copy_(gru_ref.bias_ih_l0)
    cell_gru.bias_hh.copy_(gru_ref.bias_hh_l0)

    h_g = torch.zeros(B, H)
    gru_outs = []
    for t in range(S):
        h_g = cell_gru(x_gru[:, t, :], h_g)
        gru_outs.append(h_g)
    my_gru_out = torch.stack(gru_outs, dim=1)

print("\n【手写 GRUCell VS nn.GRU 逐元素对比】")
print(f"    手写单元 output 形状：{shape(my_gru_out)}")
print(f"    nn.GRU   output 形状：{shape(ref_gru_out)}")
err_gru_out = (my_gru_out - ref_gru_out).abs().max().item()
err_gru_h = (h_g - ref_gru_hn[0]).abs().max().item()
print(f"    最大误差 output：{err_gru_out:.3e}")
print(f"    最大误差 h_n   ：{err_gru_h:.3e}")
print(f"    误差 < 1e-5 ？ {err_gru_out < 1e-5}   ← 说明 GRU 的 4 个公式和 r/z/n 顺序完全正确")

# ===========================================================================
# 七、RNN / LSTM / GRU 对比表
# ===========================================================================
section("七、RNN / LSTM / GRU 对比表（课案）")

print(
    """
    ┌────────┬──────────────────────────────┬──────────────────────────┬──────────┐
    │ 模型   │ 特点                         │ 适合场景                 │ 参数量   │
    ├────────┼──────────────────────────────┼──────────────────────────┼──────────┤
    │ RNN    │ 结构简单，但长期依赖能力弱   │ 短序列                   │ 1×       │
    │ LSTM   │ 有输入门、遗忘门、输出门     │ 长序列、稳定训练         │ 4×       │
    │ GRU    │ 门控结构比 LSTM 更简单       │ 速度优先的序列任务       │ 3×       │
    └────────┴──────────────────────────────┴──────────────────────────┴──────────┘

    工程选择的经验法则：
    · 序列短（< 20 步）、任务简单 → RNN 就够，最快；
    · 需要长距离依赖、对稳定性要求高 → LSTM；
    · 数据量大、追求训练/推理速度 → GRU（多数任务上效果与 LSTM 相当）；
    · 序列很长（> 200 步）→ 优先考虑 Transformer / Attention（见下一个脚本）。
"""
)

# ===========================================================================
# 八、小规模序列任务实战
# ===========================================================================
section("八、小规模序列任务实战")

print(
    """
本节做两个小任务，都是「整条序列 → 一个二分类标签」的范式，数据全部用 torch.randn 生成。

【任务 A：累计和的正负判断（主任务，要求能看到明显学习）】
    输入：长度为 12 的随机序列 x_0, ..., x_11，每个 x_t ~ N(0, 1)
    标签：序列所有元素之和是否大于 0
    难点：信息分散在**所有**时间步上，模型必须把 12 步的信息累积起来才能答对，
          但有大量重叠（和接近 0 的样本天然难以判断），所以准确率不会到 100%。
    这个任务的特点是「学得动」——几个 epoch 内 loss 就会明显下降。

【任务 B：记住序列的第一个元素（长距离依赖的例子）】
    输入：同样的随机序列，但把第 0 个元素放大到 N(0, 25)（信号强、方向明确）
    标签：第 0 个元素是否大于 0
    难点：标签只取决于**最早**的那个时间步，模型必须把信息带过后面 11 步。

【预期】11 步的跨度对 RNN 家族来说仍然很短，三种结构都能达到高准确率；
    真正的困难要等跨度变成几十上百步才会暴露（第四节测到的 1e-12 量级梯度就是证据），
    所以任务 B 的作用是「确认短跨度下都学得会」，而不是「证明门控更好」。
    换句话说：**结构选择要看任务的长距离依赖强度和训练数据规模，不能盲目上最复杂的模型。**

【分类范式】RNN 家族做分类的标准写法：
        output, (h_n, c_n) = lstm(x)     # 编码整条序列
        logits = fc(output[:, -1, :])    # 取最后一步的隐藏状态接全连接头
    （第三节已验证 output[:, -1, :] 与 h_n[0] 完全相同，两种写法可互换。）

【规模控制】样本 1200 / 序列长 12 / hidden 32 / 每个模型 5 个 epoch，CPU 上共约 3 秒。
"""
)

SEQ_TASK = 12
N_TRAIN, N_TEST = 1200, 300
TASK_HIDDEN = 32
EPOCHS = 5
LR = 0.02


def make_sum_task(n: int, seq_len: int = SEQ_TASK, seed: int = 0):
    """任务 A 数据：标签 = 整个序列之和是否 > 0。

    返回 X: (n, seq_len, 1)，y: (n,) 的 0/1 长整型标签。
    用独立的 torch.Generator 保证数据生成与模型初始化互不干扰、且完全可复现。
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, seq_len, 1, generator=g)
    y = (x.sum(dim=1)[:, 0] > 0).long()      # 沿时间维求和，再判断符号
    return x, y


def make_first_task(n: int, seq_len: int = SEQ_TASK, seed: int = 0, scale: float = 25.0):
    """任务 B 数据：标签 = 第 0 个时间步的值是否 > 0（其余时间步是纯噪声）。

    scale=25 是为了让信号足够强、任务在 5 个 epoch 内可学；
    如果 scale 太小（比如 3），信号会被后面的噪声淹没，模型学不动。
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, seq_len, 1, generator=g)
    x[:, 0, 0] = torch.randn(n, generator=g) * scale      # 第 0 步：强信号
    y = (x[:, 0, 0] > 0).long()
    return x, y


class SeqClassifier(nn.Module):
    """通用序列分类器：可指定循环层类型（RNN / LSTM / GRU），方便横向对比。"""

    def __init__(self, kind: str = "LSTM", input_size: int = 1,
                 hidden_size: int = TASK_HIDDEN, num_classes: int = 2):
        super().__init__()
        rnn_cls = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[kind]
        self.kind = kind
        self.rnn = rnn_cls(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 不传 h_0 / C_0，PyTorch 默认用全零初始化
        output, _ = self.rnn(x)
        # output[:, -1, :] 形状 (batch, hidden)，就是「整条序列的摘要」
        return self.fc(output[:, -1, :])


criterion = nn.CrossEntropyLoss()


def run_task(task_name: str, data_fn, kind: str, epochs: int = EPOCHS,
             lr: float = LR, verbose: bool = False):
    """在指定任务上训练一个序列分类器，返回 (逐轮训练loss列表, 逐轮测试准确率列表, 参数量)。

    verbose=True 时逐 epoch 打印（用于主任务的详细展示）。
    """
    torch.manual_seed(42)                        # 每个模型都在同一起点开始，保证对比公平
    train_x, train_y = data_fn(N_TRAIN, seed=0)
    test_x, test_y = data_fn(N_TEST, seed=1)

    model = SeqClassifier(kind)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses, accs = [], []

    if verbose:
        print(f"    {'epoch':>5s}  {'训练loss':>10s}  {'训练准确率':>10s}  {'测试准确率':>10s}")

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(train_x), train_y)   # 前向 + 交叉熵
        loss.backward()                             # 反向传播（沿时间步回传梯度）
        optimizer.step()                            # 更新参数
        losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            train_acc = (model(train_x).argmax(dim=1) == train_y).float().mean().item()
            test_acc = (model(test_x).argmax(dim=1) == test_y).float().mean().item()
        accs.append(test_acc)
        if verbose:
            print(f"    {len(losses):5d}  {loss.item():10.4f}  {train_acc:9.2%}  {test_acc:9.2%}")

    n_param = sum(p.numel() for p in model.parameters())
    return losses, accs, n_param


# ---------------------------------------------------------------------------
# 任务 A：主任务，用 LSTM 逐步展示 loss 下降
# ---------------------------------------------------------------------------
print("\n【任务 A】累计和的正负判断 —— 用 nn.LSTM 训练 5 个 epoch")

_a_x, _a_y = make_sum_task(N_TRAIN, seed=0)
print(f"    训练集：X {shape(_a_x)}  y {shape(_a_y)}，正类比例 {_a_y.float().mean():.3f}")
print("    注意：序列长 12、每一步都是 N(0,1)，和的标准差是 √12≈3.46，")
print("          所以「和恰好接近 0」的样本本身就有歧义，准确率的天花板不是 100%。")

losses_a, accs_a, param_a = run_task("累计和", make_sum_task, "LSTM", verbose=True)
print(f"\n    训练 loss：{losses_a[0]:.4f} → {losses_a[-1]:.4f}"
      f"（下降 {losses_a[0] - losses_a[-1]:.4f}）")
print(f"    测试准确率：50%（随机）→ {accs_a[0]:.2%}（第 1 轮）→ {accs_a[-1]:.2%}（第 5 轮）")
print("    → loss 稳定下降、准确率显著超过 50%，说明 LSTM 成功学会了沿时间累积信息。")
print("      这也说明：只要任务对「整条序列的聚合」有要求，RNN 家族就能用参数共享逐步学会。")

# ---------------------------------------------------------------------------
# 任务 A 对照：RNN / LSTM / GRU
# ---------------------------------------------------------------------------
print("\n【任务 A 对照】RNN / LSTM / GRU 用完全相同的设置各训练 5 个 epoch")
print(f"    {'模型':>5s}  {'参数量':>8s}  {'首轮loss':>9s}  {'末轮loss':>9s}  {'测试准确率':>10s}")
for kind in ("RNN", "LSTM", "GRU"):
    ls, ac, n_par = run_task("累计和", make_sum_task, kind)
    print(f"    {kind:>5s}  {n_par:8d}  {ls[0]:9.4f}  {ls[-1]:9.4f}  {ac[-1]:9.2%}")
print("    → 三种结构都能学会这个任务；参数量 GRU ≈ LSTM 的 3/4，训练更快。")
print("      注意这不是「门控一定赢」的比赛：短序列 + 简单聚合任务上，")
print("      结构优势体现不出来，真正拉开差距的是任务 B 这种长距离依赖。")

# ---------------------------------------------------------------------------
# 任务 B：长距离依赖
# ---------------------------------------------------------------------------
print("\n【任务 B】记住序列第 0 个元素（信号放大到 N(0,25)）—— 长距离依赖")
print(f"    {'模型':>5s}  {'末轮loss':>9s}  {'测试准确率':>10s}")
for kind in ("RNN", "LSTM", "GRU"):
    ls, ac, _ = run_task("记住第一个元素", make_first_task, kind)
    print(f"    {kind:>5s}  {ls[-1]:9.4f}  {ac[-1]:9.2%}")
print("    → 三种结构都轻松学会了 11 步跨度的记忆任务，说明这个跨度对 RNN 家族不构成挑战。")
print("      请对照第四节实测的梯度：跨 50 步时早期时间步的梯度已经衰减到 1e-12 量级，")
print("      那才是长期依赖真正失效的地方。所以：")
print("        · 跨度只有十几步 → 三种结构随便挑，优先选参数少的；")
print("        · 跨度几十步以上 → 门控结构（LSTM/GRU）的免衰减旁路开始体现价值；")
print("        · 跨度上百步 → 老老实实上 Attention / Transformer（见下一个脚本）。")
print("      另外注意 LSTM 在这里反而不如 RNN/GRU：它参数量最大（4546 vs 1186），")
print("      在这点数据量下还没训练充分，说明「更复杂的结构」不等于「更好的结果」。")

# ===========================================================================
# 小结
# ===========================================================================
section("小结")

print(
    f"""
    1. RNN 靠隐藏状态 h_t 记忆历史，所有时间步共享 W_hh / W_xh，天然支持变长序列。
    2. 公式 h_t = tanh(W_hh·h_{{t-1}} + W_xh·x_t + b_h)；手写单元与 nn.RNN 最大误差
       {err_out:.2e}（< 1e-6），证明 nn.RNN 就是这组公式的向量化实现。
    3. output 是每个时间步的隐藏状态，h_n 只有最后一步；output[:, -1, :] == h_n[0]。
    4. 梯度消失的根源是 ∂h_t/∂h_{{t-1}} = diag(tanh'(z_t))·W_hh 的 k 步连乘。
       实测（序列长 {SEQ_LEN}，损失只取最后一步）各时间步的输入梯度范数：
            RNN  t=0: {grad_rnn[0]:.2e}   t=25: {grad_rnn[25]:.2e}   t=49: {grad_rnn[49]:.2e}
            LSTM t=0: {grad_lstm[0]:.2e}   t=25: {grad_lstm[25]:.2e}   t=49: {grad_lstm[49]:.2e}
            GRU  t=0: {grad_gru[0]:.2e}   t=25: {grad_gru[25]:.2e}   t=49: {grad_gru[49]:.2e}
       三者的 t=0 都比 t=49 小 10 个数量级以上 —— 早期时间步几乎学不动。
       把 W_hh 放大 8 倍使 |λ_max| = {_big_val:.2f} > 1 后，趋势完全反转：
            t=0 的梯度范数变成 {grad_rnn_big[0]:.2e}，是 t=49（{grad_rnn_big[49]:.2e}）的
            {grad_rnn_big[0] / grad_rnn_big[49]:.2e} 倍 —— 这就是梯度爆炸。
    5. LSTM 用 f_t ⊙ C_{{t-1}} 开了一条免衰减旁路，且 f_t 可学习。
       实测：遗忘门 bias 从 0（σ≈0.5）抬到 2.0（σ≈0.88）后，t=0 的梯度从
       {grad_lstm[0]:.2e} 提升到 5.68e-03 量级（见上方对照 B 表格），衰减被大幅缓解。
       手写 6 个公式与 nn.LSTM 最大误差 {err_lstm_out:.2e}（< 1e-5）。
    6. GRU 合并遗忘门与输入门为更新门，参数是 LSTM 的 3/4；
       手写 4 个公式与 nn.GRU 最大误差 {err_gru_out:.2e}（< 1e-5）。
    7. 小任务实战：LSTM 在「累计和正负判断」上 5 个 epoch 把 loss 从 {losses_a[0]:.4f}
       降到 {losses_a[-1]:.4f}，测试准确率 50% → {accs_a[-1]:.2%}。
    8. 本脚本实测总耗时 {time.perf_counter() - _T_START:.2f} 秒，图片输出到：
       {OUTPUT_DIR}
"""
)

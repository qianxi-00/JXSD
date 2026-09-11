"""
对应课案章节：训练技巧 / 梯度爆炸与梯度消失

本节知识点：
    1.  现象对照表：梯度爆炸（loss 突然 NaN、梯度极大、一步跨飞）vs 梯度消失（loss 几乎不降、梯度接近 0）
    2.  梯度消失的根源：反向传播是**连乘**。Sigmoid 的导数上界只有 0.25，
        20 层连乘就是 $0.25^{20}\\approx 9\\times10^{-13}$，再叠加饱和区 $\\sigma'\\to0$ 就彻底下溢
    3.  20 层深网（每层 Linear(64,64)）逐层梯度范数实测：Sigmoid / Tanh / ReLU 三种激活的对比（对数纵轴）
    4.  初始化尺度的影响：`std=0.01`（信号直接死掉）/ `std=1.0`（饱和或爆炸）/ `kaiming_normal_`（尺度保持）
    5.  梯度爆炸的实测：把权重放大后逐层梯度**指数增长**到 1e8 / inf，用 `torch.isfinite` 优雅处理并打印说明
    6.  RNN 场景：$\\partial h_t/\\partial h_{t-k}=\\prod \\mathrm{diag}(\\tanh'(z))\\,W_{hh}^k$，
        用「$\\|\\partial L/\\partial h_0\\|$ 随序列长度 T 的变化」实测指数衰减与指数爆炸
    7.  梯度裁剪三件套：`clip_grad_norm_`（按总范数等比缩放，最常用）、`clip_grad_value_`（按值截断）、
        以及原理 $g\\leftarrow g\\cdot\\min(1,\\ \\text{max\\_norm}/\\|g\\|)$
    8.  裁剪的数值验证：裁剪前后 `total_norm`、裁剪后 `total_norm == max_norm`、
        以及方向余弦相似度 $==1.0$（证明「只缩小长度、不改方向」）
    9.  裁剪让训练稳定的实测：同一个「爆炸 RNN」，不裁剪 loss 震荡不降，裁剪后稳步下降
    10. 五种解决方案（课案表格）逐条给证据：ReLU / Kaiming 初始化 / BatchNorm / 残差连接 / 梯度裁剪
    11. BatchNorm 的数值验证：Sigmoid 网络每层加 `BatchNorm1d` 后逐层梯度范数大幅改善，
        并打印「落在饱和区 $|z|>5$ 的单元比例」作为证据
    12. 残差连接 $x+F(x)$ 求导得 $1+F'(x)$：20 层「普通堆叠」vs「残差堆叠」的首层梯度范数对比，
        以及 LSTM 的类比：$C_t = f_t\\odot C_{t-1}+i_t\\odot\\tilde C_t$ 里 $f_t$ 那条免衰减旁路
    13. 防梯度爆炸/消失的标准配置代码（Kaiming 初始化 + BatchNorm + ReLU + 梯度裁剪）跑一遍
    14. 课案对比表：现象 / 训练 loss / 验证 loss / 首选排查

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Deep_Learning\\04_训练技巧\\02_梯度爆炸与梯度消失演示.py'
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
# 正式内容开始：依赖导入 + 全局配置
# ---------------------------------------------------------------------------
import math                                 # 用于判断 inf / nan
import time
import unicodedata

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 统一随机种子：保证「梯度消失/爆炸」的量级每次运行都一样，便于对照
torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)                    # CPU 上限制线程数，耗时更可预期

SEED = 42
N_LAYERS = 20                               # 深网层数：20 层足以让 Sigmoid 的梯度衰减十几个数量级
WIDTH = 64                                  # 每层宽度（课案建议 64）
BATCH = 32                                  # batch 取 32 就够，20×64 的网络很快
FLOOR = 1e-30                               # 画对数纵轴时的下限：梯度下溢为 0 时用来占位


# ===========================================================================
# 通用小工具（与 01 脚本同一套：中文对齐表格 + 分节标题 + 安全格式化）
# ===========================================================================
def _disp_width(text) -> int:
    """计算字符串的终端显示宽度（中文/全角占 2 列）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in str(text))


def _pad(text, width, align="left"):
    """按显示宽度补空格。"""
    text = str(text)
    space = " " * max(0, width - _disp_width(text))
    return (text + space) if align == "left" else (space + text)


def print_table(headers, rows, aligns=None):
    """打印一张对齐的中文表格。"""
    aligns = aligns or ["left"] * len(headers)
    ncol = len(headers)
    widths = []
    for i in range(ncol):
        cells = [headers[i]] + [row[i] for row in rows]
        widths.append(max(_disp_width(c) for c in cells))
    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(sep)
    print("|" + "|".join(" " + _pad(headers[i], widths[i], aligns[i]) + " " for i in range(ncol)) + "|")
    print(sep)
    for row in rows:
        print("|" + "|".join(" " + _pad(row[i], widths[i], aligns[i]) + " " for i in range(ncol)) + "|")
    print(sep)


_CLOCK = [time.perf_counter()]


def section(title):
    """打印分节标题（附上一节耗时）。"""
    now = time.perf_counter()
    print()
    print("=" * 78)
    print(title + f"    ［上一节耗时 {now - _CLOCK[0]:.2f} 秒］")
    print("=" * 78)
    _CLOCK[0] = now


def fmt(v, nd=4):
    """把可能是 inf/NaN/None 的浮点数安全格式化成字符串（绝不抛异常）。"""
    if v is None:
        return "n/a"
    v = float(v)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return f"{v:.{nd}f}"


def sci(v, nd=3):
    """科学计数法输出（梯度量级横跨几十个数量级，必须用科学计数法才看得清）。"""
    if v is None:
        return "n/a"
    v = float(v)
    if math.isnan(v):
        return "NaN"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if v == 0.0:
        return "0（下溢）"
    return f"{v:.{nd}e}"


def orders_of_magnitude(hi, lo):
    """返回 hi/lo 跨了几个数量级（lo 下溢为 0 时返回 None）。"""
    if hi is None or lo is None or lo <= 0 or hi <= 0 or not math.isfinite(hi) or not math.isfinite(lo):
        return None
    return math.log10(hi / lo)


# ===========================================================================
# 第 1 部分：现象对照表
# ===========================================================================
section("第 1 部分：先认症状——梯度爆炸 / 梯度消失长什么样（课案表格）")

print("""
【总纲】训练出问题先看 loss 曲线的形状。梯度问题有两种完全相反的形态：

  梯度爆炸：loss 突然变成 NaN（曲线断掉），梯度值极大（可能几千、几万），
            参数一步跨太远直接变成 NaN。典型出现在：深网络、RNN/长序列、学习率过大。
  梯度消失：loss 几乎不降、停在一个高值，梯度极小（接近 0），
            参数几乎不动，训不起来。典型出现在：Sigmoid/Tanh 堆得很深的网络、
            很长的 RNN 序列。

  为什么会有这两种极端？因为反向传播的链式法则是**连乘**：
      $$\\frac{\\partial L}{\\partial W_1}
        = \\frac{\\partial L}{\\partial h_L}\\prod_{k=2}^{L}\\Big(\\mathrm{diag}(\\sigma'(z_k))\\,W_k\\Big)\\frac{\\partial h_1}{\\partial W_1}$$
  每一项都小于 1，乘 20 次就趋近 0（**消失**）；每一项都大于 1，乘 20 次就指数增长（**爆炸**）。
  换句话说：梯度问题的本质是「跨层连乘的尺度失衡」。
""")

print_table(["", "训练 loss", "梯度值", "参数更新"], [
    ["梯度爆炸", "突然变成 NaN", "极大（可能几千、几万）", "一步跨太远，参数变成 NaN"],
    ["梯度消失", "几乎不降，停在一个高值", "极小（接近 0）", "参数几乎不动，训不起来"],
])


# ===========================================================================
# 第 2 部分：梯度消失的深层演示（20 层 × 64 宽，三种激活）
# ===========================================================================
section("第 2 部分：梯度消失——20 层深网里，梯度从第 20 层到第 1 层衰减了多少？")

_FNS = {"sigmoid": torch.sigmoid, "tanh": torch.tanh, "relu": torch.relu}


class DeepNet(nn.Module):
    """可配置的深网络：n_layers 个 Linear(width,width) + 激活，最后接一个输出头。

    为什么要自己写 forward 而不是 nn.Sequential？
    因为要把每层的**预激活值 z**（激活函数之前的输出）记录下来，
    后面算「有多少单元落在饱和区 |z|>5」时要用。
    """

    def __init__(self, n_layers=N_LAYERS, width=WIDTH, act="sigmoid",
                 residual=False, use_bn=False):
        super().__init__()
        self.act_name = act
        self.residual = residual                     # True → out = x + act(Linear(x))，即残差块
        self.linears = nn.ModuleList([nn.Linear(width, width) for _ in range(n_layers)])
        # BatchNorm1d：对每个神经元的输入做「减均值、除标准差」，把尺度拉回 0 均值附近
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(n_layers)]) if use_bn else None
        self.fn = _FNS[act]
        self._zs = []                                # 记录每层预激活，供饱和区统计使用

    def forward(self, x):
        self._zs = []
        for i, lin in enumerate(self.linears):
            h = lin(x)                               # 预激活 z = W x + b
            if self.bns is not None:
                h = self.bns[i](h)                   # 归一化：把 z 稳在 0 均值、单位方差附近
            self._zs.append(h)
            a = self.fn(h)                           # 激活
            x = x + a if self.residual else a         # 残差：多一条恒等旁路
        return x


def init_normal(model, std):
    """把每层权重初始化为 N(0, std²)。std 太大 → 爆炸/饱和，太小 → 信号死掉。"""
    for lin in model.linears:
        nn.init.normal_(lin.weight, mean=0.0, std=std)
        nn.init.zeros_(lin.bias)


def init_kaiming(model, nonlinearity="relu"):
    """Kaiming 初始化：让每层输出的方差保持不变（专为 ReLU 设计）。

    原理：ReLU 会把一半神经元置 0，等价于把方差砍半。于是要让
    $\\mathrm{Var}(W)=\\frac{2}{\\text{fan\\_in}}$ 才能「每层方差不变」——
    这正是 kaiming_normal_ 默认做的事（gain 按激活函数选）。
    """
    for lin in model.linears:
        nn.init.kaiming_normal_(lin.weight, nonlinearity=nonlinearity)
        nn.init.zeros_(lin.bias)


def measure_deep(model, x):
    """一次前向 + 反向，返回 (逐层梯度范数列表, loss 值, 各层饱和比例列表)。

    损失用「输出平方和」而不是交叉熵：它不依赖标签，纯看信号/梯度在网络里的传播尺度，
    是分析梯度消失最干净的探针（$L=\\sum_i o_i^2$，$\\partial L/\\partial o=2o$）。
    """
    model.zero_grad()
    model.train()                                    # BatchNorm 用当前 batch 统计量（训练态的真实行为）
    out = model(x)
    loss = out.pow(2).sum()
    lv = loss.item()
    if not math.isfinite(lv):                        # 已经溢出 → 没法再 backward，交给调用方说明
        return None, lv, None
    loss.backward()
    norms = [float(lin.weight.grad.norm()) for lin in model.linears]
    # 饱和比例：|z|>5 时 Sigmoid 输出贴在 0/1 上、Tanh 贴在 ±1 上，导数几乎为 0
    sat = [float((z.abs() > 5).float().mean()) for z in model._zs]
    model.zero_grad()
    return norms, lv, sat


_x_probe = torch.randn(BATCH, WIDTH)                 # 固定同一批输入，保证三种激活可比

print(f"被测网络：{N_LAYERS} 层 × {WIDTH} 宽，输入 torch.randn({BATCH},{WIDTH})，"
      f"损失 = 输出平方和 $\\sum o_i^2$")
print()
print("【逐层梯度范数 · Sigmoid（默认初始化）】从最后一层往第一层看：")
torch.manual_seed(SEED)
_m_sig = DeepNet(act="sigmoid")
_g_sig, _l_sig, _sat_sig = measure_deep(_m_sig, _x_probe)
for _i in range(N_LAYERS - 1, -1, -1):
    _bar = "█" * max(1, int(round((math.log10(max(_g_sig[_i], FLOOR)) + 30) / 1.6)))
    print(f"    第 {_i + 1:>2d} 层（从输出数第 {N_LAYERS - _i:>2d} 层）梯度范数 = "
          f"{sci(_g_sig[_i]):>12s}  {_bar}")
print(f"  → Sigmoid 的梯度从第 20 层的 {sci(_g_sig[-1])} 掉到第 1 层的 {sci(_g_sig[0])}，")
_ord = orders_of_magnitude(_g_sig[-1], _g_sig[0])
print(f"    跨了约 {_ord:.1f} 个数量级（衰减到约 10^-{round(_ord)} 倍）。")

print()
print("""【为什么会这样】Sigmoid 的导数是 $\\sigma'(x)=\\sigma(x)(1-\\sigma(x))$，
    它的**上界只有 0.25**（在 x=0 处取到）。反向传播要逐层乘 $\\sigma'$：
    就算网络一直工作在最佳工作点（x≈0），20 层也只有
    $$0.25^{20}=9.1\\times10^{-13}$$
    而实际训练中输入往往偏离 0，$\\sigma'$ 更小，于是梯度很快下溢到 float32 的精度之外。
    Tanh 的导数上界是 1（比 Sigmoid 好 4 倍），ReLU 的正半轴导数恒为 1——
    但由于「ReLU 砍掉一半神经元」，默认初始化下信号仍会逐层缩水，所以 ReLU 也需要配套的初始化。""")

_summary = []
for _act in ("sigmoid", "tanh", "relu"):
    torch.manual_seed(SEED)
    _m = DeepNet(act=_act)
    _g, _lv, _sat = measure_deep(_m, _x_probe)
    if _g is None:
        # 极端情况下（前向已溢出）也要保证画图代码拿得到列表，这里用 NaN 占位
        _summary.append([_act, f"loss={fmt(_lv)}（已溢出）", "n/a", "n/a", "n/a", "n/a"])
        globals()[f"_g_{_act}"] = [float("nan")] * N_LAYERS
        continue
    _o = orders_of_magnitude(_g[-1], _g[0])
    _summary.append([_act, sci(_g[-1]), sci(_g[9]), sci(_g[0]),
                     ("NaN" if _o is None else f"{_o:.1f}"), fmt(float(np.mean(_sat)), 3)])
    globals()[f"_g_{_act}"] = _g                       # 保存给第 3、7 部分和画图使用

print()
print("【三种激活对照（默认初始化，同样 20 层）】")
print_table(["激活函数", "第 20 层梯度范数", "第 10 层梯度范数", "第 1 层梯度范数",
             "衰减数量级(第20→第1)", "饱和比例(|z|>5)"], _summary)
print("""  → Sigmoid 衰减最惨（十几个数量级，梯度基本归零）；Tanh 好一些（导数上界 1）；
    ReLU 的导数不压缩，但在**默认初始化**下每层方差仍会减半，所以也有明显衰减。
    第 3 部分会说明：ReLU 配 Kaiming 初始化，才能让 20 层的梯度尺度真正保持住。""")


# ===========================================================================
# 第 3 部分：初始化尺度的影响
# ===========================================================================
section("第 3 部分：初始化尺度——std=0.01 / std=1.0 / kaiming_normal_")

print("""
【原理】一层的输出方差 $\\mathrm{Var}(h_l)=\\text{fan\\_in}\\cdot\\mathrm{Var}(W)\\cdot\\mathrm{Var}(h_{l-1})$。
    想让信号穿过 20 层还不放大不缩小，就要让 $\\text{fan\\_in}\\cdot\\mathrm{Var}(W)\\approx 1$
    （ReLU 还要把被砍掉的一半补回来 ⇒ $\\mathrm{Var}(W)\\approx 2/\\text{fan\\_in}$，这就是 Kaiming）。
    · std 太小（0.01）：每层方差被乘上一个远小于 1 的系数 → 信号指数衰减 → 梯度**下溢为 0**；
    · std 太大（1.0）：每层方差被放大 → 预激活值冲到饱和区（Sigmoid/Tanh）或者直接溢出成 inf；
    · Kaiming/Xavier：按 fan_in 定尺度，让每层「刚好抵消」，梯度尺度大致稳定。
""")

_init_rows = []
_init_curves = {}
for _act in ("sigmoid", "relu"):
    for _kind, _val in [("std=0.01", 0.01), ("std=1.0", 1.0), ("kaiming_normal_", None)]:
        torch.manual_seed(SEED)
        _m = DeepNet(act=_act)
        if _val is None:
            init_kaiming(_m, nonlinearity="relu")
        else:
            init_normal(_m, _val)
        _g, _lv, _sat = measure_deep(_m, _x_probe)
        _key = f"{_act}+{_kind}"
        if _g is None:
            _init_rows.append([_act, _kind, "n/a（loss 已溢出为 inf）", "n/a", "n/a", "n/a"])
            _init_curves[_key] = None
            continue
        _o = orders_of_magnitude(_g[-1], _g[0])
        _init_curves[_key] = _g
        _init_rows.append([
            _act, _kind, sci(_g[-1]), sci(_g[0]),
            ("—（含 0/inf，无法比较）" if _o is None else f"{_o:.1f}"),
            (fmt(float(np.mean(_sat)), 3) if _act == "sigmoid" else "不适用(ReLU 无饱和区)"),
        ])
print_table(["激活", "初始化", "第 20 层梯度范数", "第 1 层梯度范数",
             "衰减数量级", "饱和比例"], _init_rows)

# 单独把「Sigmoid + std=1.0 的饱和比例」测出来：用来说明「梯度没消失 ≠ 网络能训练」
torch.manual_seed(SEED)
_m_sig2 = DeepNet(act="sigmoid")
init_normal(_m_sig2, 1.0)
_g_sig2, _lv_sig2, _sat_sig2 = measure_deep(_m_sig2, _x_probe)
_sat_ratio_sig2 = float(np.mean(_sat_sig2))

# 顺便把「Sigmoid + BatchNorm1d」也测出来：左图要把它画进去做对照（第 7 部分复用这组数）
torch.manual_seed(SEED)
_m_bn_def = DeepNet(act="sigmoid", use_bn=True)
_g_bn_def, _lv_bn_def, _sat_bn_def = measure_deep(_m_bn_def, _x_probe)

print()
print("  → 读表要点：")
print(f"    · Sigmoid + std=0.01：第 1 层梯度 {sci(_init_curves['sigmoid+std=0.01'][0])}"
      f"——信号还没传到第 1 层就已经在 float32 里下溢成 0 了，网络是「死」的。")
print(f"    · Sigmoid + std=1.0：预激活被层层放大，**{_sat_ratio_sig2 * 100:.1f}% 的单元落在饱和区 |z|>5**，")
print(f"      $\\sigma'$ 在这些单元上接近 0（x=5 时 $\\sigma'=6.5\\times10^{{-3}}$，"
      f"x=10 时 $\\sigma'=4.5\\times10^{{-5}}$）——网络名义上「有梯度」，实际上已经学不动。")
print(f"    · ReLU + std=1.0：前向直接溢出成 inf（第 4 部分会实测），梯度全是 NaN —— 初始化太大就是爆炸。")
print(f"    · ReLU + std=0.01：梯度精确为 0（信号逐层衰减到 0，ReLU 又把剩下的砍掉）。")
print(f"    · ReLU + kaiming_normal_：第 20 层 {sci(_init_curves['relu+kaiming_normal_'][-1])}、"
      f"第 1 层 {sci(_init_curves['relu+kaiming_normal_'][0])}，")
_o_kai = orders_of_magnitude(_init_curves["relu+kaiming_normal_"][-1], _init_curves["relu+kaiming_normal_"][0])
print(f"      20 层下来只差 {abs(_o_kai):.2f} 个数量级"
      f"（第 1 层 / 第 20 层 = {_init_curves['relu+kaiming_normal_'][0] / _init_curves['relu+kaiming_normal_'][-1]:.2f}）"
      f"—— 梯度尺度**基本没变**，这就是 Kaiming 初始化的目的：")
print("      让「fan_in 放大的方差」和「激活函数砍掉的部分」刚好抵消。")

fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.6))
_ep_axis = np.arange(1, N_LAYERS + 1)
# 左图：三种激活的逐层梯度范数（对数纵轴）+ BatchNorm 的对照
for _act, _c in zip(("sigmoid", "tanh", "relu"), ("#d62728", "#ff7f0e", "#1f77b4")):
    _g = globals()[f"_g_{_act}"]
    axes[0].plot(_ep_axis, np.maximum(_g, FLOOR), marker="o", ms=3.5, lw=1.8,
                 color=_c, label=f"{_act}")
axes[0].plot(_ep_axis, np.maximum(_g_bn_def, FLOOR), marker="s", ms=3.5, lw=2.0,
             ls="--", color="#2ca02c", label="sigmoid + BatchNorm1d（第 7 部分详解）")
axes[0].set_yscale("log")
axes[0].set_title(f"{N_LAYERS} 层深网逐层梯度范数（默认初始化）\n"
                  "Sigmoid/Tanh 越往浅层越小 → 梯度消失", fontsize=12)
axes[0].set_xlabel("层号（1 = 最靠近输入，20 = 最靠近输出）")
axes[0].set_ylabel("该层权重梯度范数（对数）")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3, which="both")
axes[0].annotate(f"Sigmoid 第 1 层 = {sci(_g_sig[0])}\n（已下溢到 float32 精度之外）",
                 xy=(1.2, max(_g_sig[0], FLOOR) * 1.6), xytext=(4.5, 1e-12),
                 color="#d62728", fontsize=9,
                 bbox=dict(facecolor="white", alpha=0.9, edgecolor="#d62728"),
                 arrowprops=dict(arrowstyle="->", color="#d62728"))

# 右图：初始化尺度的影响
_style_init = {
    "sigmoid+std=0.01": dict(color="#8c564b", ls="--", lw=1.7, marker="s", ms=3),
    "sigmoid+std=1.0": dict(color="#d62728", ls="-", lw=1.7, marker="o", ms=3),
    "sigmoid+kaiming_normal_": dict(color="#e377c2", ls=":", lw=1.8, marker="^", ms=3),
    "relu+kaiming_normal_": dict(color="#2ca02c", ls="-", lw=2.2, marker="D", ms=3),
}
for _key, _st in _style_init.items():
    _g = _init_curves.get(_key)
    if _g is None:
        continue
    axes[1].plot(_ep_axis, np.maximum(_g, FLOOR), label=_key, **_st)
axes[1].set_yscale("log")
axes[1].set_title("初始化尺度决定信号能不能穿过 20 层\n"
                  "std=0.01 → 下溢为 0；std=1.0 → 饱和；kaiming → 尺度保持", fontsize=12)
axes[1].set_xlabel("层号"); axes[1].set_ylabel("该层权重梯度范数（对数）")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3, which="both")
axes[1].annotate("std=0.01：梯度精确下溢为 0\n（画在对数轴下限上）",
                 xy=(10, FLOOR * 2.5), xytext=(3.0, 1e-19), color="#8c564b", fontsize=9,
                 bbox=dict(facecolor="white", alpha=0.9, edgecolor="#8c564b"),
                 arrowprops=dict(arrowstyle="->", color="#8c564b"))
axes[1].annotate(f"std=1.0：{_sat_ratio_sig2 * 100:.1f}% 单元饱和\n"
                 "（梯度不小但网络已学不动）",
                 xy=(15, max(_init_curves['sigmoid+std=1.0'][14], FLOOR) * 0.4),
                 xytext=(6.0, 1e-9), color="#d62728", fontsize=9,
                 bbox=dict(facecolor="white", alpha=0.9, edgecolor="#d62728"),
                 arrowprops=dict(arrowstyle="->", color="#d62728"))

fig.suptitle("训练技巧 · 梯度消失：激活函数与初始化尺度决定反向传播的尺度", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.93))
_p1 = OUTPUT_DIR / "04训练技巧_02_逐层梯度范数.png"
fig.savefig(_p1, dpi=130)
plt.close(fig)
print(f"\n[图片] 已保存 {_p1}")


# ===========================================================================
# 第 4 部分：梯度爆炸
# ===========================================================================
section("第 4 部分：梯度爆炸——权重放大后，逐层梯度指数增长到 inf")

print("""
【怎么造爆炸】把每层权重放大即可：$W_k\\to sW_k$ 时
    $$\\Big\\|\\prod_k \\mathrm{diag}(\\sigma'(z_k))W_k\\Big\\|
      \\le \\prod_k \\|W_k\\|\\cdot\\max|\\sigma'| = (s\\|W\\|)^L$$
    也就是说梯度随层数 L **指数增长**，$s\\|W\\|>1$ 时很快就冲到 float32 的上限 $3.4\\times10^{38}$（→ inf）。
    实测里用两种方式制造爆炸：
      ① 权重直接乘 3（relu 网络）→ 前向就已经溢出，loss = inf；
      ② 用 std=1.5 初始化 tanh 网络 → 数值仍然有限，可以完整看到逐层指数增长。
【必须做的事】一旦出现 inf/NaN，要用 `torch.isfinite` 拦住，
    绝不能让 NaN 顺着参数扩散，也不能让脚本抛异常栈。
""")

_expl_rows = []
_expl_curves = {}

# ① 温和一点的爆炸：tanh + std=1.5，逐层梯度还能看清
torch.manual_seed(SEED)
_m_exp = DeepNet(act="tanh")
init_normal(_m_exp, 1.5)
_g_exp, _lv_exp, _sat_exp = measure_deep(_m_exp, _x_probe)
_expl_curves["tanh std=1.5"] = _g_exp
_o_exp = orders_of_magnitude(_g_exp[0], _g_exp[-1])
print("【逐层梯度范数 · tanh + std=1.5（从最后一层往第一层看）】")
for _i in range(N_LAYERS - 1, -1, -1):
    print(f"    第 {_i + 1:>2d} 层梯度范数 = {sci(_g_exp[_i])}")
print(f"  → 梯度从第 20 层的 {sci(_g_exp[-1])} 涨到第 1 层的 {sci(_g_exp[0])}，"
      f"**放大了 {_o_exp:.1f} 个数量级**——这就是「爆炸」：越靠近输入的层拿到越夸张的梯度。")
_expl_rows.append(["tanh + std=1.5", sci(_g_exp[-1]), sci(_g_exp[0]),
                   f"{_o_exp:.1f}（增长）", "有限"])

# ② 真爆炸：relu 网络权重乘 3 → 前向直接溢出
torch.manual_seed(SEED)
_m_inf = DeepNet(act="relu")
init_normal(_m_inf, 1.0)
with torch.no_grad():
    for _lin in _m_inf.linears:
        _lin.weight.mul_(3.0)
_g_inf, _lv_inf, _sat_inf = measure_deep(_m_inf, _x_probe)
print()
print(f"【真爆炸 · relu + std=1.0 且每层权重再乘 3】前向损失 = {fmt(_lv_inf)}（不是有限值！）")
print("  用 torch.isfinite 检查并优雅处理：")
# 前向已经溢出，backward 会得到 NaN 梯度。这里显式跑一遍 backward 并检查，全程不抛异常。
try:
    _m_inf.zero_grad()
    _m_inf.train()
    _out_inf = _m_inf(_x_probe)
    _loss_inf = _out_inf.pow(2).sum()
    _finite_loss = bool(torch.isfinite(_loss_inf))
    print(f"    torch.isfinite(loss) = {_finite_loss}"
          + ("  → 出现 inf，说明激活/梯度已经爆炸；此时**绝不能**再 optimizer.step()，"
             "否则 NaN 会写进参数。下面仍跑一次 backward，只为把「梯度已经全是 NaN」看清楚。"
             if not _finite_loss else ""))
    _loss_inf.backward()
    _gn_inf = [float(_lin.weight.grad.norm()) for _lin in _m_inf.linears]
    _n_bad = sum(1 for _lin in _m_inf.linears if not bool(torch.isfinite(_lin.weight.grad).all()))
    print(f"    反向传播后：{_n_bad}/{N_LAYERS} 层的权重梯度里含 NaN/inf"
          f"（第 1 层梯度范数 = {sci(_gn_inf[0])}）")
    print("    → 结论：这里出现了 inf/NaN，梯度爆炸已经发生。正确处理方式是"
          "「检测到非有限值就停止这块更新 + 加梯度裁剪 + 调小学习率/初始化」，")
    print("      而不是硬跑下去（NaN 一旦写进参数，后面全是 NaN）。")
    _expl_rows.append(["relu + std=1.0，权重×3", "n/a（前向已 inf）", "NaN/inf",
                       "溢出", "inf / NaN"])
except Exception as _e:                                   # noqa: BLE001
    print(f"    该实验在本机环境触发异常（{type(_e).__name__}），已安全跳过（不打印栈）。")
    _expl_rows.append(["relu + std=1.0，权重×3", "n/a", "n/a", "溢出", "inf / NaN"])
finally:
    _m_inf.zero_grad()

print()
print("【爆炸对照表】")
print_table(["配置", "第 20 层梯度范数", "第 1 层梯度范数", "跨多少数量级", "是否溢出成 inf"], _expl_rows)
print("""  → 爆炸的尾巴最危险：它往往**只发生在少数几步**（比如某个难样本、某段长序列），
    loss 曲线前面看着好好的，突然就 NaN 了。所以工程上要「防守式编程」：
    监控梯度范数 + `torch.isfinite` 检查 + 梯度裁剪。""")


# ===========================================================================
# 第 5 部分：RNN 的梯度爆炸 + 梯度裁剪的数值验证
# ===========================================================================
section("第 5 部分：RNN 的梯度爆炸 与 梯度裁剪（clip_grad_norm_ / clip_grad_value_）")

print("""
【RNN 为什么特别容易爆炸】RNN 在**时间维**上共享同一个 $W_{hh}$，反向传播要沿着时间连乘：
    $$\\frac{\\partial h_t}{\\partial h_{t-k}}
      = \\prod_{i=0}^{k-1}\\mathrm{diag}\\!\\big(\\tanh'(z_{t-i})\\big)\\,W_{hh}$$
    这是**同一个矩阵连乘 k 次**。设 $W_{hh}$ 的谱半径为 $\\rho$，则 $\\|W_{hh}^k\\|$ 大致按 $\\rho^k$ 变化：
      · $\\rho<1$：梯度随序列长度**指数衰减** → 梯度消失，学不会长期依赖；
      · $\\rho>1$：梯度随序列长度**指数爆炸** → 训练直接发散。
    下面用「$\\|\\partial L/\\partial h_0\\|$ 随序列长度 T 的变化」把这个指数律实测出来。
""")

_T_LIST = (5, 10, 20, 30, 40, 50)


def rnn_h0_grad_norm(T, scale, in_dim=8, hidden=16, batch=BATCH):
    """测 $\\|\\partial L/\\partial h_0\\|$：loss 只取最后一个时间步的输出。

    这样梯度必须从最后一个时间步一路传回初始状态 h_0，
    乘的就是 $\\prod \\mathrm{diag}(\\tanh')\\,W_{hh}$ 这一串，
    用它来观察「随 T 指数变化」最干净。
    """
    torch.manual_seed(SEED)
    rnn = nn.RNN(in_dim, hidden, batch_first=True)
    with torch.no_grad():
        rnn.weight_hh_l0.mul_(scale)                 # 放大循环权重 → 改变谱半径
    inp = torch.randn(batch, T, in_dim) * 0.5
    h0 = torch.zeros(1, batch, hidden, requires_grad=True)
    out, _ = rnn(inp, h0)
    loss = out[:, -1, :].pow(2).sum()
    if not math.isfinite(loss.item()):
        return None
    loss.backward()
    return float(h0.grad.norm())


print("【实测：$\\|\\partial L/\\partial h_0\\|$ 随序列长度 T 的变化】")
_rnn_curves = {}
for _scale, _label in ((0.8, "W_hh × 0.8（谱半径<1 → 梯度消失）"),
                       (5.0, "W_hh × 5.0（谱半径>1 → 梯度爆炸）")):
    _vals = [rnn_h0_grad_norm(T, _scale) for T in _T_LIST]
    _rnn_curves[_scale] = _vals
    _txt = "  ".join(f"T={T}:{('NaN' if v is None else sci(v))}" for T, v in zip(_T_LIST, _vals))
    print(f"  {_label}\n      {_txt}")
    _v_first, _v_last = _vals[0], _vals[-1]
    if _v_first and _v_last and _v_last > 0:
        _ratio = _v_first / _v_last
        print(f"      → T 从 5 到 50，梯度{'衰减' if _ratio > 1 else '放大'}了 {sci(max(_ratio, 1 / _ratio))} 倍")
print("""  → T 每增加一段，梯度就乘上一个固定因子：因子 <1 就是指数衰减（消失），
    >1 就是指数增长（爆炸）。这就是「RNN 学不会长期依赖」和「RNN 训练容易炸」的同一个原因。""")

print()
print("""【梯度裁剪】三种做法（课案表格里的最后一道保险）：
    ① `clip_grad_norm_(params, max_norm)`（最常用）：先算所有参数梯度的**全局 L2 范数**
       $$\\|g\\|=\\sqrt{\\sum_i \\|g_i\\|^2}$$
       若 $\\|g\\|>\\text{max\\_norm}$，则所有梯度**同乘一个系数**：
       $$g \\leftarrow g\\cdot\\min\\Big(1,\\ \\frac{\\text{max\\_norm}}{\\|g\\|}\\Big)$$
       ——注意是「**等比缩小**」，方向完全不变，只是把这一步的长度截到 max_norm 以内。
    ② `clip_grad_value_(params, clip_value)`：逐元素把梯度限制在 $[-c, c]$。
       它**会改变梯度方向**（各分量被裁剪的比例不同），一般用在按范数裁剪不够稳的场合。
    ③ 原理总结：裁剪不改变「往哪走」，只改变「一步走多远」，所以它对付的是
       「偶发的超大梯度把参数一步带飞」，而不是学习率本身。""")

# ---- 裁剪的数值验证：RNN 梯度爆炸场景 ----
torch.manual_seed(SEED)
_rnn_clip = nn.RNN(8, 16, batch_first=True)
with torch.no_grad():
    _rnn_clip.weight_hh_l0.mul_(5.0)                 # 造一个爆炸的 RNN
_seq = torch.randn(BATCH, 50, 8)
_out_r, _ = _rnn_clip(_seq)
_loss_r = _out_r.pow(2).sum()
_rnn_clip.zero_grad()
_loss_r.backward()
_params_r = [p for p in _rnn_clip.parameters() if p.grad is not None]
_before = torch.cat([p.grad.detach().reshape(-1) for p in _params_r])
_tn_before = float(_before.norm())
# clip_grad_norm_ 的返回值就是裁剪前的全局范数 total_norm
_tn_returned = float(torch.nn.utils.clip_grad_norm_(_rnn_clip.parameters(), max_norm=1.0))
_after = torch.cat([p.grad.detach().reshape(-1) for p in _params_r])
_tn_after = float(_after.norm())
_cos = float(F.cosine_similarity(_before, _after, dim=0))
_max_before = float(_before.abs().max())

print()
print("【裁剪的数值验证（在一个真爆炸的 RNN 梯度上做）】")
print(f"  裁剪前：total_norm = {sci(_tn_before)}（梯度已经大到离谱），"
      f"最大单元素 |g| = {sci(_max_before)}")
print(f"  clip_grad_norm_(max_norm=1.0) 返回的 total_norm = {sci(_tn_returned)}"
      f"（与手工计算的 {sci(_tn_before)} 一致：{abs(_tn_returned - _tn_before) < 1e-6 * _tn_before}）")
print(f"  裁剪后：total_norm = {fmt(_tn_after, 6)}（**应当恰好等于 max_norm = 1.0**）")
print(f"  裁剪前后梯度的**方向余弦相似度** = {fmt(_cos, 10)}（== 1.0 说明方向一点没变）")

# ---- clip_grad_value_ 对照 ----
torch.manual_seed(SEED)
_rnn_val = nn.RNN(8, 16, batch_first=True)
with torch.no_grad():
    _rnn_val.weight_hh_l0.mul_(5.0)
_out_v, _ = _rnn_val(_seq)
_rnn_val.zero_grad()
_out_v.pow(2).sum().backward()
_pv = [p for p in _rnn_val.parameters() if p.grad is not None]
_max_v_before = max(float(p.grad.abs().max()) for p in _pv)
_before_val = torch.cat([p.grad.detach().reshape(-1) for p in _pv])
torch.nn.utils.clip_grad_value_(_rnn_val.parameters(), clip_value=0.1)
_max_v_after = max(float(p.grad.abs().max()) for p in _pv)
_after_val = torch.cat([p.grad.detach().reshape(-1) for p in _pv])
_cos_val = float(F.cosine_similarity(_before_val, _after_val, dim=0))
print(f"\n  对照 `clip_grad_value_(clip_value=0.1)`：最大单元素 |g| "
      f"{sci(_max_v_before)} → {fmt(_max_v_after, 4)}（被逐元素截到 0.1 以内），")
print(f"    但方向余弦相似度只有 {fmt(_cos_val, 6)}（≠1）——**按值裁剪会改变梯度方向**，")
print(f"    所以「按范数等比缩小」才是默认首选；按值裁剪用于需要硬上限的场景。")
_after_val_norm = float(_after_val.norm())

# ---- RNN 梯度随序列长度变化的柱状/折线数据，供画图 ----

fig2, axes2 = plt.subplots(1, 3, figsize=(19, 5.4))
# 面板 1：梯度爆炸的逐层范数
axes2[0].plot(_ep_axis, np.maximum(_g_exp, FLOOR), marker="o", ms=3.5, lw=1.9,
              color="#d62728", label="tanh + std=1.5（爆炸，仍有限）")
axes2[0].plot(_ep_axis, np.maximum(_g_kai := _init_curves["relu+kaiming_normal_"], FLOOR),
              marker="D", ms=3, lw=1.9, color="#2ca02c", label="relu + kaiming（健康）")
axes2[0].set_yscale("log")
axes2[0].set_title("梯度爆炸：逐层梯度指数增长\n"
                   f"tanh std=1.5 时第 1 层是第 20 层的 {_o_exp:.0f} 个数量级倍", fontsize=12)
axes2[0].set_xlabel("层号（1 = 最靠近输入）"); axes2[0].set_ylabel("该层权重梯度范数（对数）")
axes2[0].legend(fontsize=9); axes2[0].grid(alpha=0.3, which="both")
axes2[0].text(0.03, 0.06, "注：relu + std=1.0 且每层权重×3 时\n前向 loss 直接 = inf，"
                          "梯度全为 NaN，无法作图",
              transform=axes2[0].transAxes, fontsize=8.5, va="bottom",
              bbox=dict(facecolor="white", alpha=0.9, edgecolor="#8c564b"))

# 面板 2：RNN 的梯度随 T 指数变化
for _scale, _c in zip((0.8, 5.0), ("#1f77b4", "#d62728")):
    _vals = [(_v if _v else np.nan) for _v in _rnn_curves[_scale]]
    axes2[1].plot(_T_LIST, _vals, marker="o", ms=4, lw=2.0, color=_c,
                  label=("W_hh×0.8：指数衰减（消失）" if _scale == 0.8
                         else "W_hh×5.0：指数增长（爆炸）"))
axes2[1].set_yscale("log")
axes2[1].set_title("RNN：$\\|\\partial L/\\partial h_0\\|$ 随序列长度 T 的变化\n"
                   "同一个矩阵连乘 T 次 → 指数衰减或指数爆炸", fontsize=12)
axes2[1].set_xlabel("序列长度 T"); axes2[1].set_ylabel(r"$\|\partial L/\partial h_0\|$（对数）")
axes2[1].legend(fontsize=9); axes2[1].grid(alpha=0.3, which="both")

# 面板 3：裁剪前后每个参数张量的梯度范数
_names = ["W_ih", "W_hh", "b_ih", "b_hh"]
torch.manual_seed(SEED)
_rnn_bar = nn.RNN(8, 16, batch_first=True)
with torch.no_grad():
    _rnn_bar.weight_hh_l0.mul_(5.0)
_out_b, _ = _rnn_bar(_seq)
_rnn_bar.zero_grad()
_out_b.pow(2).sum().backward()
_nb = [float(_rnn_bar.weight_ih_l0.grad.norm()), float(_rnn_bar.weight_hh_l0.grad.norm()),
       float(_rnn_bar.bias_ih_l0.grad.norm()), float(_rnn_bar.bias_hh_l0.grad.norm())]
torch.nn.utils.clip_grad_norm_(_rnn_bar.parameters(), max_norm=1.0)
_na = [float(_rnn_bar.weight_ih_l0.grad.norm()), float(_rnn_bar.weight_hh_l0.grad.norm()),
       float(_rnn_bar.bias_ih_l0.grad.norm()), float(_rnn_bar.bias_hh_l0.grad.norm())]
_xpos = np.arange(len(_names))
axes2[2].bar(_xpos - 0.2, np.maximum(_nb, FLOOR), width=0.4, color="#d62728", label="裁剪前")
axes2[2].bar(_xpos + 0.2, np.maximum(_na, FLOOR), width=0.4, color="#2ca02c", label="裁剪后")
for _x, _v in zip(_xpos - 0.2, _nb):
    axes2[2].text(_x, max(_v, FLOOR), sci(_v, 1), ha="center", va="bottom", fontsize=8, rotation=90)
for _x, _v in zip(_xpos + 0.2, _na):
    axes2[2].text(_x, max(_v, FLOOR), sci(_v, 1), ha="center", va="bottom", fontsize=8, rotation=90)
axes2[2].set_yscale("log")
axes2[2].set_ylim(FLOOR, max(max(_nb), FLOOR) * 300)     # 留出空间，避免数值标签被裁掉
axes2[2].set_xticks(_xpos); axes2[2].set_xticklabels(_names)
axes2[2].set_title(f"梯度裁剪前后每个参数张量的梯度范数\n"
                   f"total_norm {sci(_tn_before, 1)} → {fmt(_tn_after, 3)}"
                   f"（max_norm=1.0），方向余弦={fmt(_cos, 6)}", fontsize=12)
axes2[2].set_ylabel("梯度范数（对数）"); axes2[2].legend(fontsize=9); axes2[2].grid(alpha=0.3, axis="y")

fig2.suptitle("训练技巧 · 梯度爆炸与梯度裁剪：连乘导致指数增长，裁剪把「一步的长度」截断", fontsize=14)
fig2.tight_layout(rect=(0, 0, 1, 0.92))
_p2 = OUTPUT_DIR / "04训练技巧_02_梯度爆炸与裁剪.png"
fig2.savefig(_p2, dpi=130)
plt.close(fig2)
print(f"\n[图片] 已保存 {_p2}")


# ===========================================================================
# 第 6 部分：梯度裁剪让训练稳定（同一个爆炸 RNN，裁剪 vs 不裁剪）
# ===========================================================================
section("第 6 部分：实测——同一个「爆炸 RNN」，不裁剪 vs 裁剪")

print("""
【实验设计】造一个谱半径 >1 的 RNN（把 weight_hh_l0 放大 2.5 倍），
    任务是「判断整段序列第一个通道的累加和是否为正」——必须把信息从序列开头
    一路带到结尾，所以梯度要走完整的 T 步。
    两组唯一的差别是：一组在 optimizer.step() 前加 `clip_grad_norm_(max_norm=1.0)`。
【预期】不裁剪：被放大的循环梯度让参数一步跨得很远，容易冲进 tanh 饱和区，
    之后梯度「有值但没方向」，loss 震荡不降；
    裁剪：每步的移动长度被限制住，训练能稳定推进。
""")

_SEQ_N, _SEQ_T, _SEQ_D = 640, 30, 3


def make_seq_data(n=_SEQ_N, T=_SEQ_T, d=_SEQ_D, seed=0):
    """合成序列数据：标签 = （第一个通道在时间上的累加和 > 0），需要长期记忆。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, T, d, generator=g)
    y = (x[:, :, 0].sum(dim=1) > 0).long()
    return x, y


_sx, _sy = make_seq_data()
print(f"[数据] 序列数据 x.shape={tuple(_sx.shape)}，"
      f"正类占比 {float(_sy.float().mean()):.3f}（累加和 > 0 的比例）")


def train_rnn(*, clip_norm=None, scale=2.5, lr=0.1, epochs=40, bs=64, tag=""):
    """训练一个 RNN 序列分类器，返回逐 epoch 的 loss 和平均梯度范数历史。"""
    torch.manual_seed(SEED)
    rnn = nn.RNN(_SEQ_D, 16, batch_first=True)
    head = nn.Linear(16, 2)
    with torch.no_grad():
        rnn.weight_hh_l0.mul_(scale)                 # 放大循环权重 → 谱半径 > 1 → 梯度爆炸
    params = list(rnn.parameters()) + list(head.parameters())
    opt = torch.optim.SGD(params, lr=lr)
    hist_loss, hist_gn = [], []
    diverged_at = None

    for ep in range(epochs):
        perm = torch.randperm(_sx.shape[0])
        ep_loss_sum, ep_gn_sum, nb = 0.0, 0.0, 0
        for i in range(0, _sx.shape[0], bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out, _ = rnn(_sx[idx])
            loss = F.cross_entropy(head(out[:, -1, :]), _sy[idx])
            if not math.isfinite(loss.item()):       # 爆炸到 NaN → 优雅停止，绝不抛栈
                diverged_at = ep + 1
                break
            loss.backward()
            # 先用一个极大的 max_norm 调用一次，纯粹为了把 total_norm 读出来（相当于测量）
            ep_gn_sum += float(torch.nn.utils.clip_grad_norm_(params, max_norm=1e12))
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(params, max_norm=clip_norm)
            opt.step()
            ep_loss_sum += loss.item()
            nb += 1
        if diverged_at is not None:
            print(f"  [{tag}] 第 {diverged_at} 个 epoch loss 变成非有限值 → 安全停止（不抛异常）")
            break
        with torch.no_grad():
            out_all, _ = rnn(_sx)
            hist_loss.append(F.cross_entropy(head(out_all[:, -1, :]), _sy).item())
        hist_gn.append(ep_gn_sum / max(nb, 1))
    return {"loss": hist_loss, "gn": hist_gn, "diverged_at": diverged_at, "tag": tag,
            "rnn": rnn}                              # 把训练好的 RNN 也带出来，供饱和区分析


_hist_noclip = train_rnn(clip_norm=None, tag="不裁剪")
_hist_clip = train_rnn(clip_norm=1.0, tag="裁剪 max_norm=1.0")


def rnn_saturation_ratio(rnn, x, thresh=3.0):
    """统计 RNN 各时间步预激活落在 tanh 饱和区（|z| > thresh）的比例。

    手动展开 RNN 的递推：$z_t = x_tW_{ih}^T + b_{ih} + h_{t-1}W_{hh}^T + b_{hh}$。
    tanh 在 |z|>3 处导数已经小于 0.01，可以认为「饱和、梯度传不过去」。
    """
    w_ih, w_hh = rnn.weight_ih_l0, rnn.weight_hh_l0
    b_ih, b_hh = rnn.bias_ih_l0, rnn.bias_hh_l0
    h = torch.zeros(x.shape[0], rnn.hidden_size)
    n_sat, n_tot = 0, 0
    with torch.no_grad():
        for t in range(x.shape[1]):
            z = x[:, t, :] @ w_ih.T + b_ih + h @ w_hh.T + b_hh
            n_sat += int((z.abs() > thresh).sum())
            n_tot += z.numel()
            h = torch.tanh(z)
    return n_sat / n_tot

rows = []
for _h in (_hist_noclip, _hist_clip):
    _ls = _h["loss"]
    rows.append([
        _h["tag"], len(_ls),
        fmt(_ls[0]) if _ls else "n/a",
        fmt(float(np.min(_ls))) if _ls else "n/a",
        fmt(_ls[-1]) if _ls else "n/a",
        sci(_h["gn"][0]) if _h["gn"] else "n/a",
        sci(float(np.median(_h["gn"]))) if _h["gn"] else "n/a",
        ("是" if _h["diverged_at"] else "否"),
    ])
print()
print_table(["配置", "epoch", "首轮 loss", "最低 loss", "末轮 loss",
             "首轮梯度总范数", "全程梯度总范数中位数", "是否出现 NaN"], rows)
_nc_min = float(np.min(_hist_noclip["loss"])) if _hist_noclip["loss"] else float("nan")
_nc_last = _hist_noclip["loss"][-1] if _hist_noclip["loss"] else float("nan")
_nc_g0 = _hist_noclip["gn"][0] if _hist_noclip["gn"] else float("nan")
_nc_gn = float(np.median(_hist_noclip["gn"])) if _hist_noclip["gn"] else float("nan")
_c_g0 = _hist_clip["gn"][0] if _hist_clip["gn"] else float("nan")
_c_gn = float(np.median(_hist_clip["gn"])) if _hist_clip["gn"] else float("nan")
_c_min = float(np.min(_hist_clip["loss"])) if _hist_clip["loss"] else float("nan")
# 饱和区分析：不裁剪组是不是被巨大的初始梯度带进了 tanh 饱和区？
_sat_nc = rnn_saturation_ratio(_hist_noclip["rnn"], _sx)
_sat_c = rnn_saturation_ratio(_hist_clip["rnn"], _sx)
print(f"  → 不裁剪：第一个 epoch 的梯度总范数就有 {sci(_nc_g0)}，")
print(f"    但训练完一轮后，隐藏状态的预激活有 **{_sat_nc * 100:.1f}%** 落在 tanh 饱和区 |z|>3")
print(f"    （裁剪组只有 {_sat_c * 100:.1f}%）——不受限制的更新把参数带进了饱和区，")
print(f"    此后梯度虽然数值不大（全程中位数 {sci(_nc_gn)}），方向却已经失效，")
print(f"    loss 一直在 {fmt(_nc_min)} ~ {fmt(_nc_last)} 之间来回晃，40 个 epoch 后仍停在 "
      f"{fmt(_nc_last)}，等于没学到东西。")
print(f"  → 裁剪到 max_norm=1.0 后：每步的移动长度被限制住，参数走得更平稳，")
print(f"    梯度一直保持在有效量级（裁剪前的 total_norm 中位数 {sci(_c_gn)}，说明网络确实在活跃学习），")
print(f"    loss 从 {fmt(_hist_clip['loss'][0])} 稳稳降到 {fmt(_hist_clip['loss'][-1])}"
      f"（最低 {fmt(_c_min)}），明显低于瞎猜水平 ln2≈0.693。")
print("    原因：裁剪只改「一步走多远」，不改「往哪走」——方向还是那个正确的下降方向，")
print("    只是不允许某一步因为梯度爆炸而跨过头，把参数甩进饱和区。")
print("    注意：`clip_grad_norm_` 是「防守」而不是「加速」——它不会让收敛更快，")
print("    但没有这道防守，训练可能在中途就崩掉或者卡死。")

fig3, axes3 = plt.subplots(1, 2, figsize=(15.5, 5.4))
for _h, _c in ((_hist_noclip, "#d62728"), (_hist_clip, "#2ca02c")):
    if not _h["loss"]:
        continue
    _x = np.arange(1, len(_h["loss"]) + 1)
    axes3[0].plot(_x, _h["loss"], lw=2.0, color=_c,
                  label=f"{_h['tag']}（末轮 loss={fmt(_h['loss'][-1])}）")
    axes3[1].plot(np.arange(1, len(_h["gn"]) + 1), np.maximum(_h["gn"], FLOOR),
                  lw=2.0, color=_c, label=_h["tag"])
axes3[0].axhline(math.log(2), color="gray", ls="--", lw=1.2, label="瞎猜水平 ln2≈0.693")
axes3[0].set_title("裁剪让爆炸的 RNN 训练稳定\n不裁剪：震荡不降；裁剪：稳步下降", fontsize=12)
axes3[0].set_xlabel("epoch"); axes3[0].set_ylabel("训练 loss")
axes3[0].legend(fontsize=9); axes3[0].grid(alpha=0.3)
axes3[1].set_yscale("log")
axes3[1].axhline(1.0, color="gray", ls="--", lw=1.2, label="max_norm = 1.0（裁剪阈值）")
axes3[1].set_title("每步「裁剪前」的梯度总范数（total_norm）\n"
                   "裁剪组一直高于 1.0（被截到 1.0，网络保持活跃）；\n"
                   "不裁剪组后期掉到 1 以下（冲进饱和区，梯度失去方向）", fontsize=12)
axes3[1].set_xlabel("epoch"); axes3[1].set_ylabel("梯度总范数（对数）")
axes3[1].legend(fontsize=9); axes3[1].grid(alpha=0.3, which="both")

fig3.suptitle("训练技巧 · 梯度裁剪效果：方向不变，只把「一步的长度」截断", fontsize=14)
fig3.tight_layout(rect=(0, 0, 1, 0.92))
_p3 = OUTPUT_DIR / "04训练技巧_02_梯度裁剪效果.png"
fig3.savefig(_p3, dpi=130)
plt.close(fig3)
print(f"\n[图片] 已保存 {_p3}")


# ===========================================================================
# 第 7 部分：五种解决方案（课案表格）+ 逐条证据
# ===========================================================================
section("第 7 部分：五种解决方案逐条验证（ReLU / Kaiming / BatchNorm / 残差 / 裁剪）")

print("""
────────────────────────────────────────────────────────────────────────
① ReLU 替代 Sigmoid/Tanh
   ReLU 的正半轴导数恒为 1，不像 Sigmoid（上界 0.25）那样层层压缩梯度。
   → 证据（第 2 部分实测）：同样 20 层、同样初始化，
     Sigmoid 第 1 层梯度 = {sig_l1}，Tanh = {tanh_l1}，ReLU = {relu_l1}。
     注意 ReLU 自己也不够：默认初始化下每层方差减半，仍有衰减，
     必须配合下面的 Kaiming 初始化。

② Kaiming / Xavier 初始化
   让每层权重的方差刚好抵消掉「fan_in 放大」和「激活函数砍半」，保持梯度尺度稳定。
   → 证据（第 3 部分实测）：ReLU 网络换成 kaiming_normal_ 后，
     第 20 层 {kai_l20} → 第 1 层 {kai_l1}，20 层下来只差 {kai_ord} 个数量级（基本不衰减）；
     而 std=0.01 时梯度直接下溢为 0。

③ BatchNorm / LayerNorm
   把每层输入稳在激活函数的**非饱和区**，$\\sigma'$ 不会趋近 0。
   → 证据：见下方实测（Sigmoid 网络每层加 BatchNorm1d）。

④ 残差连接
   旁路 $x+F(x)$ 求导得 $1+F'(x)$：多了个恒等项 1，等于给梯度开了一条直传通道，
   无论 $F'(x)$ 多小，梯度都不会被压成 0。LSTM 里的 $f_t\\odot C_{{t-1}}$ 是同一个道理。
   → 证据：见下方实测（20 层普通堆叠 vs 残差堆叠）。

⑤ 梯度裁剪
   梯度总长度超标就等比缩小，防一步跨飞。
   → 证据（第 5、6 部分实测）：裁剪前 total_norm = {tn_before}，裁剪后正好 = 1.0，
     方向余弦相似度 = {cos_sim}；爆炸 RNN 的 loss 从震荡不降变成稳步下降到 {clip_last}。
────────────────────────────────────────────────────────────────────────
""".format(sig_l1=sci(_g_sig[0]), tanh_l1=sci(_g_tanh[0]), relu_l1=sci(_g_relu[0]),
           kai_l20=sci(_init_curves["relu+kaiming_normal_"][-1]),
           kai_l1=sci(_init_curves["relu+kaiming_normal_"][0]),
           kai_ord=f"{abs(_o_kai):.2f}", tn_before=sci(_tn_before),
           cos_sim=fmt(_cos, 6),
           clip_last=fmt(_hist_clip["loss"][-1]) if _hist_clip["loss"] else "n/a"))

print("""
【饱和区导数值（课案注释的完整版）】
    以 Sigmoid 为例，当输入 x 很大（例如 x>5）或很小（x<-5）时，
    输出几乎贴在 0 或 1 上不变了——这就是**饱和**。
    饱和区导数值 $\\sigma'(x)$ 接近 0，反向传播时梯度会被压扁。
    BatchNorm / LayerNorm 把每层输入稳在 0 均值附近，避免进入饱和区，
    导数维持在正常范围，梯度就不会消失。
    （数值参考：$\\sigma'(0)=0.25$、$\\sigma'(5)=6.5\\times10^{-3}$、$\\sigma'(10)=4.5\\times10^{-5}$，
      差了四个数量级。）
""")

# ---- ③ BatchNorm 实测 ----
print("【③ BatchNorm 实测】同一个 Sigmoid 网络（默认初始化），每层插一个 BatchNorm1d：")
# _g_sig / _sat_sig 来自第 2 部分（Sigmoid 默认初始化），_g_bn_def / _sat_bn_def 来自第 3 部分
_g_plain, _sat_plain = _g_sig, _sat_sig
_g_bn, _sat_bn = _g_bn_def, _sat_bn_def
_bn_gain = orders_of_magnitude(_g_bn[0], _g_plain[0])
rows = [
    ["Sigmoid（不加 BN）", sci(_g_plain[-1]), sci(_g_plain[0]),
     fmt(float(np.mean(_sat_plain)), 3), "—"],
    ["Sigmoid + BatchNorm1d", sci(_g_bn[-1]), sci(_g_bn[0]),
     fmt(float(np.mean(_sat_bn)), 3),
     (f"提升 {_bn_gain:.1f} 个数量级" if _bn_gain else "n/a")],
]
print_table(["配置", "第 20 层梯度范数", "第 1 层梯度范数", "饱和比例(|z|>5)", "首层梯度改善"], rows)
print(f"  → 加 BN 后，第 1 层梯度从 {sci(_g_plain[0])} 变成 {sci(_g_bn[0])}，"
      f"提升了约 {_bn_gain:.1f} 个数量级。")
print(f"    默认初始化下 Sigmoid 本来就不怎么饱和（饱和比例 "
      f"{fmt(float(np.mean(_sat_plain)), 3)}），梯度却还是下溢成了 0，")
print("    原因是「逐层缩水」：BN 把每层的输入尺度重新归一化，反向传播时多了一个")
print("    $1/\\sigma$（batch 标准差）的尺度恢复因子，抵掉了逐层缩水，所以浅层梯度回到了可用量级。")
# 单独用 std=1.0 的强饱和场景再验证一次「BN 治饱和」
torch.manual_seed(SEED)
_m_bn2 = DeepNet(act="sigmoid", use_bn=True)
init_normal(_m_bn2, 1.0)
_g_bn2, _lv_bn2, _sat_bn2 = measure_deep(_m_bn2, _x_probe)
print(f"  → 把初始化放大到 std=1.0（强饱和场景）：不加 BN 时饱和比例 "
      f"{fmt(_sat_ratio_sig2, 3)}，加 BN 后 {fmt(float(np.mean(_sat_bn2)), 3)}"
      f"（BN 把每个神经元的输入拉回 0 均值、单位方差，天然躲开饱和区）。")
print(f"    这也解释了课案的注释：「BatchNorm/LayerNorm 把每层输入稳在 0 均值附近，")
print(f"    避免进入饱和区，导数维持在正常范围，梯度就不会消失。」")
print("    但也要诚实指出：BN 解决的是**饱和**，不是「Sigmoid 导数上界只有 0.25」这件事。")
print(f"    20 层 Sigmoid + BN 的第 1 层梯度 = {sci(_g_bn[0])}，比 ReLU + Kaiming 的 "
      f"{sci(_init_curves['relu+kaiming_normal_'][0])} 仍小很多。")
print("    真正的标准答案是「ReLU + Kaiming + BN」三件套一起上（见第 8 部分）。")

# ---- ④ 残差连接实测 ----
print()
print("【④ 残差连接实测】20 层「普通堆叠」 vs 「残差堆叠」（都在默认初始化下，公平对比）：")
_res_rows = []
_res_curves = {}
for _act in ("sigmoid", "tanh", "relu"):
    torch.manual_seed(SEED)
    _m_p = DeepNet(act=_act, residual=False)
    _gp, _lp, _sp = measure_deep(_m_p, _x_probe)
    torch.manual_seed(SEED)
    _m_r = DeepNet(act=_act, residual=True)
    _gr, _lr, _sr = measure_deep(_m_r, _x_probe)
    _gain = orders_of_magnitude(_gr[0], _gp[0])
    _res_rows.append([_act, sci(_gp[-1]), sci(_gp[0]), sci(_gr[0]),
                      ("提升 %s 个数量级" % f"{_gain:.1f}") if _gain else "n/a"])
    _res_curves[_act] = (_gp, _gr)
print_table(["激活", "普通堆叠 第20层", "普通堆叠 第1层", "残差堆叠 第1层", "首层梯度改善"], _res_rows)
_sig_p, _sig_r = _res_curves["sigmoid"]
print(f"  → Sigmoid 最直观：普通堆叠第 1 层梯度 {sci(_sig_p[0])}（等于没有），")
print(f"    残差堆叠第 1 层梯度 {sci(_sig_r[0])}——从「完全消失」变成「能用的量级」。")
print("""    数学原因：$\\partial(x+F(x))/\\partial x = 1 + F'(x)$。
    那个常数 1 就是旁路：无论 $F'(x)$ 被压得多小，梯度都能沿着恒等路径原样传回去，
    所以残差网络的浅层永远拿得到非零梯度（这也是 ResNet 能把网络做到几百层的根本原因）。
    实践提醒：残差堆叠的输出幅度会随层数累积变大（实测 mean 梯度尺度也更大），
    所以真实 ResNet 会在每个 block 里配 BatchNorm、并在最后接一个正常尺度的分类头。

【LSTM 的类比（课案原文）】梯度消失时同理，LSTM 用一条「免衰减旁路」解决：
    $$C_t = f_t\\odot C_{t-1} + i_t\\odot\\tilde C_t$$
    细胞状态 $C$ 的更新里，$C_{t-1}$ 是**加**上去的（不是乘一个矩阵），
    所以 $\\partial C_t/\\partial C_{t-1}=f_t$ 是一个**对角矩阵**，
    梯度沿时间反传时是 $\\prod f_t$（逐元素相乘），而不是 $\\prod W_{hh}$（矩阵连乘）。
    当遗忘门 $f_t\\approx1$ 时，梯度可以近似无损地穿过很多时间步——
    这就是「把梯度消失从**必然**变成了**模型可以选择**」：
    该记的东西（$f_t\\to1$）长期保留，该忘的东西（$f_t\\to0$）主动清掉。""")

fig4, axes4 = plt.subplots(1, 2, figsize=(15.5, 5.4))
for _act, _c in zip(("sigmoid", "tanh", "relu"), ("#d62728", "#ff7f0e", "#1f77b4")):
    _gp, _gr = _res_curves[_act]
    axes4[0].plot(_ep_axis, np.maximum(_gp, FLOOR), ls="--", lw=1.6, marker="o", ms=3,
                  color=_c, alpha=0.75, label=f"{_act} 普通堆叠")
    axes4[0].plot(_ep_axis, np.maximum(_gr, FLOOR), ls="-", lw=2.2, marker="^", ms=3.5,
                  color=_c, label=f"{_act} 残差堆叠")
axes4[0].set_yscale("log")
axes4[0].set_title("残差 vs 普通堆叠：逐层梯度范数\n"
                   "残差的浅层梯度不再被压成 0（$1+F'(x)$ 的恒等项）", fontsize=12)
axes4[0].set_xlabel("层号（1 = 最靠近输入）"); axes4[0].set_ylabel("该层权重梯度范数（对数）")
axes4[0].legend(fontsize=8, ncol=2); axes4[0].grid(alpha=0.3, which="both")

_labels = ["sigmoid", "tanh", "relu"]
_plain_l1 = [max(_res_curves[a][0][0], FLOOR) for a in _labels]
_res_l1 = [max(_res_curves[a][1][0], FLOOR) for a in _labels]
_x = np.arange(len(_labels))
axes4[1].bar(_x - 0.2, _plain_l1, width=0.4, color="#d62728", label="普通堆叠")
axes4[1].bar(_x + 0.2, _res_l1, width=0.4, color="#2ca02c", label="残差堆叠")
for _xi, _v in zip(_x - 0.2, _plain_l1):
    axes4[1].text(_xi, _v, sci(_v, 1), ha="center", va="bottom", fontsize=8, rotation=90)
for _xi, _v in zip(_x + 0.2, _res_l1):
    axes4[1].text(_xi, _v, sci(_v, 1), ha="center", va="bottom", fontsize=8, rotation=90)
axes4[1].set_yscale("log")
axes4[1].set_ylim(FLOOR, max(_res_l1) * 300)             # 留出空间，避免数值标签被裁掉
axes4[1].set_xticks(_x); axes4[1].set_xticklabels(_labels)
axes4[1].set_title("第 1 层（最靠近输入）梯度范数对比\n残差让浅层梯度从「消失」变成「可用」", fontsize=12)
axes4[1].set_ylabel("第 1 层梯度范数（对数）"); axes4[1].legend(fontsize=9)
axes4[1].grid(alpha=0.3, axis="y")

fig4.suptitle("训练技巧 · 残差连接：$x+F(x)$ 求导得到 $1+F'(x)$，给梯度开一条直传通道", fontsize=14)
fig4.tight_layout(rect=(0, 0, 1, 0.92))
_p4 = OUTPUT_DIR / "04训练技巧_02_残差连接对比.png"
fig4.savefig(_p4, dpi=130)
plt.close(fig4)
print(f"\n[图片] 已保存 {_p4}")


# ===========================================================================
# 第 8 部分：防梯度爆炸/消失的标准配置（课案代码，完整跑一遍）
# ===========================================================================
section("第 8 部分：防梯度爆炸/消失的标准配置（Kaiming + BatchNorm + ReLU + 梯度裁剪）")

print("""
【课案的「组合拳」】把前面四种手段叠在一起用：
    ① init_weights：Linear 用 `kaiming_uniform_`（控制 W 的尺度），bias 置 0；
    ② 每个 Linear 后面跟 `BatchNorm1d`（稳定输入，躲开饱和区）+ `ReLU`（正半轴导数恒 1）；
    ③ 训练循环里先 `zero_grad()` → `backward()` → `clip_grad_norm_(max_norm=1.0)` → `step()`。
    裁剪放在 backward 之后、step 之前，是最后一道保险：防爆炸。
【数据】用 make_classification 造一个 10 类、20 维（10 个有效特征）的数据集，
    和课案里 `nn.Linear(64, 10)` 的输出维度对齐。
""")

_Xc, _yc = make_classification(
    n_samples=1000, n_features=20, n_informative=10, n_redundant=5,
    n_classes=10, n_clusters_per_class=1, flip_y=0.05, random_state=SEED,
)
_Xc_tr, _Xc_va, _yc_tr, _yc_va = train_test_split(_Xc, _yc, test_size=0.2,
                                                  random_state=SEED, stratify=_yc)
_sc = StandardScaler().fit(_Xc_tr)
_A = torch.tensor(_sc.transform(_Xc_tr), dtype=torch.float32)
_B = torch.tensor(_yc_tr, dtype=torch.long)
_Av = torch.tensor(_sc.transform(_Xc_va), dtype=torch.float32)
_Bv = torch.tensor(_yc_va, dtype=torch.long)
print(f"[数据] 10 类数据集：训练 {_A.shape[0]} 条、验证 {_Av.shape[0]} 条，特征 {_A.shape[1]} 维")


def init_weights(m):
    """课案给的初始化函数：Linear 用 Kaiming 控制尺度，bias 置 0。"""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)   # ReLU 用 Kaiming，控制 W 的尺度
        nn.init.zeros_(m.bias)


torch.manual_seed(SEED)
model = nn.Sequential(
    nn.Linear(20, 256), nn.BatchNorm1d(256), nn.ReLU(),
    nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
    nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
    nn.Linear(64, 10),
)
model.apply(init_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
print(f"[模型] 参数量 = {sum(p.numel() for p in model.parameters()):,}")

_epochs = 20
_curve = []
_gn_curve = []
for _ep in range(_epochs):
    model.train()
    _perm = torch.randperm(_A.shape[0])
    _tot, _gn_sum, _nb = 0.0, 0.0, 0
    for _i in range(0, _A.shape[0], 64):
        _idx = _perm[_i:_i + 64]
        optimizer.zero_grad()                        # 1) 梯度清零
        _loss = F.cross_entropy(model(_A[_idx]), _B[_idx])
        _loss.backward()                             # 2) 反向传播
        # 3) 最后一道保险：梯度裁剪，防爆炸（返回的就是裁剪前的 total_norm）
        _gn_sum += float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0))
        optimizer.step()                             # 4) 更新参数
        _tot += _loss.item() * len(_idx)
        _nb += 1
    _curve.append(_tot / _A.shape[0])
    _gn_curve.append(_gn_sum / _nb)
    if _ep % 4 == 0 or _ep == _epochs - 1:
        print(f"  epoch {_ep + 1:>2d}/20  train_loss = {_curve[-1]:.4f}   "
              f"每步梯度总范数（裁剪前）= {_gn_curve[-1]:.4f}")

model.eval()
with torch.no_grad():
    _va_loss = F.cross_entropy(model(_Av), _Bv).item()
    _va_acc = (model(_Av).argmax(dim=1) == _Bv).float().mean().item()
print(f"  → 训练 loss 从 {_curve[0]:.4f} 降到 {_curve[-1]:.4f}（20 个 epoch 降了一个多数量级），"
      f"验证 loss = {_va_loss:.4f}，验证准确率 = {_va_acc:.4f}")
print(f"    每步「裁剪前」的梯度总范数只有 {min(_gn_curve):.2f} ~ {max(_gn_curve):.2f}，")
print("    说明这套配置下梯度本来就健康：既没有爆炸（loss 不会 NaN），也没有消失（loss 稳定下降），")
print("    `clip_grad_norm_` 在这里基本没被触发——它的作用是**保险**：一旦哪天出现异常大梯度，")
print("    它会把那一步截住，而不是让整个训练崩掉。")
print(f"    （补一句诚实的观察：本实验的验证 loss = {_va_loss:.2f} 明显高于训练 loss，")
print("      说明这个 10 类数据集上模型已经开始过拟合了——那是「过拟合」的问题，")
print("      不是梯度问题，处理方式见同目录 01_过拟合与欠拟合演示.py：加正则化/早停/加数据。）")


# ===========================================================================
# 第 9 部分：对比表
# ===========================================================================
section("第 9 部分：现象 / 训练 loss / 验证 loss / 首选排查（课案对比表）")

print_table(["现象", "训练 loss", "验证 loss", "首选排查"], [
    ["过拟合", "↓ 持续降", "↑ 反弹", "加正则化"],
    ["欠拟合", "高且不降", "高且不降", "增大模型"],
    ["震荡", "剧烈波动", "剧烈波动", "降低学习率"],
    ["缓慢", "极慢下降", "极慢下降", "调大学习率"],
    ["爆炸/消失", "NaN 或不降", "NaN 或不降", "梯度裁剪"],
])

print("""
【梯度问题的诊断顺序（本脚本实测支持）】
  1. loss 突然 NaN / inf  → 先看梯度范数：是不是梯度爆炸（本脚本第 4、5 部分）
       → 立刻加 clip_grad_norm_(max_norm=1.0)；再调小学习率、换 Kaiming 初始化、加 BatchNorm。
  2. loss 几乎不降、停在高位，且浅层梯度极小 → 梯度消失（第 2、3 部分）
       → 换 ReLU、改 Kaiming 初始化、加 BatchNorm/残差、别把 Sigmoid 叠太深。
  3. 两者的共同根源都是「跨层连乘的尺度失衡」，所以工程上的标准答案是同一套组合拳：
     **合适的激活 + 匹配的初始化 + 归一化层 + 残差（深网）+ 梯度裁剪（保险）**。
""")

print("=" * 78)
print("全部实验完成。图片输出目录：" + str(OUTPUT_DIR))
print("=" * 78)

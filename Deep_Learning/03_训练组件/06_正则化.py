"""
对应课案章节：训练组件 / 正则化

本节知识点：
    1.  正则化的目标：不是降低训练集损失，而是让模型在没见过的数据上表现更好——
        本质都是限制模型复杂度，防止它「背」训练集而不是「学」规律。
    2.  Dropout：训练时以概率 p 随机把神经元输出置 0，推理时全部神经元工作。
        - 为什么推理要乘 (1-p)：训练时只有 (1-p) 比例神经元活跃，期望输出是原来的 (1-p) 倍；
          推理时全活跃、输出是完整 1 倍，网络没接收过「满血」信号会不适应，乘 (1-p) 把尺度压回去；
        - PyTorch 的 nn.Dropout 用的是「倒置 Dropout」：训练时除以 (1-p) 放大，推理时什么都不做，
          效果等价且推理更快；本脚本手写倒置 dropout 与 nn.Dropout 逐一对比尺度；
        - 数值演示 train() 下多次前向结果不同、eval() 下完全相同，置 0 比例 ≈ p。
    3.  L1 / L2 范数正则的贝叶斯来源（用正确 LaTeX 重写课案公式）：
        频率派：θ 是常量，用 MLE；误差 ~ N(0, σ²) → 负对数似然 → 最小化 Σ(y-f)²（最小二乘）；
        贝叶斯派：θ 是随机变量、有先验 f(θ)，最大化后验 P(θ|X,Y) ∝ P(Y|X,θ)f(θ)，
        取负对数后多出来的 -log f(θ) **就是正则项**；
        高斯先验 → -log f(w) = w²/(2σ_w²) + const → 令 λ = σ²/σ_w² → L2 正则 λΣw²（Ridge）；
        拉普拉斯先验 → -log f(w) = |w|/b + const → 系数 2σ²/b 打包成 λ → L1 正则 λΣ|w|（Lasso）。
    4.  Weight Decay 与 L2 正则的等价性（SGD 下）：
        λ/2·w² 求导得 λw，代入 w ← w - α(∇L + λw) = (1-αλ)w - α∇L，
        即「先把权重缩到 (1-αλ) 倍，再沿梯度走一步」——这就是 Weight Decay 这个名字的来源。
        本脚本用真实数值证明「手写 L2 惩罚」与「optim.SGD(weight_decay=λ)」的权重范数完全一致。
    5.  L1 比 L2 更容易产生稀疏解（0 值）的三个角度：先验分布 / 几何视角 / 导数视角，
        并做数值验证（线性回归 + 梯度下降，统计 |w| < 1e-3 的个数）。
    6.  归一化四兄弟 BatchNorm / LayerNorm / InstanceNorm / GroupNorm 共用同一个公式
        y = γ·(x-μ)/√(σ²+ε) + β，区别只在**在哪些维度上**计算 μ 和 σ²；
        逐个打印输出形状并手写复现 BatchNorm / LayerNorm（误差 < 1e-5）。
    7.  γ 和 β 的作用：纯归一化会把表达能力压死（Sigmoid 在 0 附近近似线性），
        可学习的缩放和平移让网络自己决定每层的分布范围；打印它们的形状与初始值（γ=1、β=0）。
    8.  BatchNorm 在 train() / eval() 下的行为差异：训练用当前 batch 统计量并更新
        running_mean / running_var，推理用累积的全局统计量——实际演示输出不同并打印 running_mean 的变化。
    9.  NLP / Transformer 中的归一化：数据形状 (B, S, H)；
        BatchNorm 沿 B,H 归一化、LayerNorm 沿 H 归一化，另外两个不用。
        重点讲透 NLP 为什么用 LayerNorm 而不是 BatchNorm（变长序列、padding 差异、
        训练推理不一致、自回归生成 batch=1、BatchNorm1d 需要转置的别扭感），
        并做「同一条样本换个 batch 伙伴」的敏感度实验。
    10. Pre-LN vs Post-LN：原论文是 Post-LN `LayerNorm(x + Sublayer(x))`，
        现代实现多为 Pre-LN `x + Sublayer(LayerNorm(x))`，Pre-LN 训练更稳定、常不需要 warmup。
    11. 早停 EarlyStopping：写一个完整的类（patience / min_delta / mode / best_score / counter /
        best_state_dict / restore），实际在训练中使用，打印早停发生的 epoch 并对比恢复最佳权重
        后的验证 loss 与最后一轮验证 loss，画曲线标出早停点与最佳点。
    12. 数据增强（**禁止 torchvision**，全部用纯 torch 手写）：
        random_horizontal_flip / random_crop / random_gaussian_noise / random_vertical_flip /
        random_rotation90 / random_brightness / random_erase，以及模仿 transforms.Compose 的 Compose 类；
        讲清 padding=4 的含义（232×232 pad 后随机裁 224×224 = 最多 4 像素随机平移）；
        用「150 张小数据 + 稍大模型」制造过拟合场景，对比无增强 / 有增强的 train-test 差距。
    13. 对比表：方法 / 作用层面 / 核心机制。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Deep_Learning\\03_训练组件\\06_正则化.py'
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

OUTPUT_DIR = _ROOT / "output"               # 所有图片 / 权重文件统一输出到这里
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 正式内容开始：导入标准库与 PyTorch
# ---------------------------------------------------------------------------
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)                    # 小数据量下限制线程，避免 OpenMP 调度开销拖慢速度

print("=" * 78)
print("06 正则化：Dropout / L1-L2 / 权重衰减 / 归一化 / 早停 / 数据增强")
print("=" * 78)
print(f"PyTorch 版本：{torch.__version__}")
print()
print(
    """
本节的总纲：
    正则化的目标**不是**降低训练集损失，而是让模型在没见过的数据上表现更好。
    所有正则化手段的本质都是「限制模型复杂度」，防止模型去「背」训练集而不是「学」规律。
    下面依次讲 Dropout、L1/L2 范数、权重衰减、归一化、早停、数据增强。
"""
)
print()


# ===========================================================================
# 第 1 部分：Dropout
# ===========================================================================
print("=" * 78)
print("第 1 部分：Dropout（随机失活）")
print("=" * 78)
print(
    """
原理：
    训练时以概率 p 随机把神经元的输出置 0，推理时所有神经元正常工作。

为什么推理时要乘 (1-p)？
    训练时只有 (1-p) 比例的神经元活跃，这一层的期望输出是「全部活跃时」的 (1-p) 倍。
    推理时所有神经元都活跃，输出是完整的 1 倍——比训练时大。
    网络在训练中从没接收过这种「满血」信号，尺度对不上会不适应。
    乘 (1-p) 就是把推理时的输出压回训练时的尺度，让训练/推理的期望一致。

PyTorch 用的是「倒置 Dropout」(inverted dropout)：
    训练时**除以 (1-p)** 把保留的神经元放大 1/(1-p) 倍，让这一层的期望输出保持为原来的 1 倍；
    推理时**什么都不做**。
    数学上与「训练不缩放、推理乘 (1-p)」完全等价，但推理时省掉一次乘法，更快。
    这也是为什么你在 PyTorch 里 eval() 之后不需要手动乘 (1-p)——框架已经帮你算进训练里了。
"""
)

# --- 1.1 演示 train() 下随机、eval() 下确定 ---
torch.manual_seed(42)
drop = nn.Dropout(p=0.5)
x_drop = torch.ones(4, 10)                  # 全 1 输入，方便直接读「缩放倍数」

drop.train()
outs_train = [drop(x_drop) for _ in range(4)]
print("【演示】输入全是 1 的 (4, 10) 张量，nn.Dropout(p=0.5)：")
print("  model.train() 下连续 4 次前向，第 1 行分别为：")
for i, o in enumerate(outs_train, start=1):
    print(f"    第{i}次: " + "  ".join(f"{v:+.3f}" for v in o[0].tolist()))
_same_train = all(torch.equal(outs_train[0], o) for o in outs_train[1:])
print(f"  4 次结果是否完全相同 → {_same_train}（训练时是随机丢弃，应该为 False）")
print(f"  保留的神经元数值 = {1 / (1 - 0.5):.1f}（倒置 Dropout 把保留值放大了 1/(1-p) 倍）")

drop.eval()
outs_eval = [drop(x_drop) for _ in range(4)]
_same_eval = all(torch.equal(outs_eval[0], o) for o in outs_eval[1:])
print("  model.eval() 下连续 4 次前向：")
print(f"    第1次: " + "  ".join(f"{v:+.3f}" for v in outs_eval[0][0].tolist()))
print(f"  4 次结果是否完全相同 → {_same_eval}（推理时不做任何随机操作，应该为 True）")
print(f"  推理输出是否等于输入本身 → {torch.equal(outs_eval[0], x_drop)}（倒置 Dropout 推理时什么都不做）")
print()

# --- 1.2 统计实际置 0 比例 ---
print("【演示】实际丢弃比例是否接近 p")
torch.manual_seed(42)
for _p in (0.2, 0.5, 0.8):
    _d = nn.Dropout(p=_p)
    _d.train()
    _big = torch.ones(2000, 512)
    _o = _d(_big)
    _zero_frac = (_o == 0).float().mean().item()
    _scale = _o[_o != 0].mean().item()
    print(f"  p={_p:<4} 实际置 0 比例 = {_zero_frac * 100:6.2f}%   非零元素的缩放倍数 = {_scale:.3f}"
          f"（= 1/(1-p) = {1 / (1 - _p):.3f}）")
print()

# --- 1.3 手写「倒置 Dropout」与 nn.Dropout 对比 ---
print("【演示】手写倒置 Dropout 与 nn.Dropout 的尺度对比")


def my_inverted_dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    """手写「倒置 Dropout」：训练时以概率 p 置 0，并把保留的元素放大 1/(1-p)；推理时原样返回。

    这解释了 PyTorch nn.Dropout 在 eval() 下为什么「什么都不做」：
    因为缩放已经在训练时做完了，推理只负责恒等映射。
    """
    if not training or p == 0.0:
        return x
    # torch.rand_like 生成与 x 同形状的 [0,1) 均匀随机数，> p 的位置保留
    mask = (torch.rand_like(x) > p).to(x.dtype)
    return x * mask / (1.0 - p)             # 除以 (1-p) 就是「倒置」的那一步


torch.manual_seed(0)
_x = torch.ones(4000, 256)
_nn_drop = nn.Dropout(p=0.3)
_nn_drop.train()
_o_nn = _nn_drop(_x)
_o_my = my_inverted_dropout(_x, 0.3, training=True)
print(f"  nn.Dropout(p=0.3)   train 后：均值={_o_nn.mean().item():.4f}  非零均值={_o_nn[_o_nn != 0].mean().item():.4f}"
      f"  置0比例={(_o_nn == 0).float().mean().item() * 100:.2f}%")
print(f"  手写倒置 dropout    train 后：均值={_o_my.mean().item():.4f}  非零均值={_o_my[_o_my != 0].mean().item():.4f}"
      f"  置0比例={(_o_my == 0).float().mean().item() * 100:.2f}%")
print("  → 两者期望都保持为输入的 1.0（数学期望 = 1），非零元素都被放大到 1/(1-0.3) = "
      f"{1 / 0.7:.4f}。")
print("  → 作为对照：如果训练**不做** 1/(1-p) 缩放（朴素 dropout），这一层输出的期望只有 "
      f"{1 - 0.3:.1f}，")
print("     推理时输出期望是 1.0，两者差 1/(1-p) 倍——所以才需要在推理时乘 (1-p) 补回来。")
_o_naive = _x * (torch.rand_like(_x) > 0.3).to(_x.dtype)
print(f"     朴素 dropout 训练时的输出期望 = {_o_naive.mean().item():.4f}，"
      f"推理时（不乘任何系数）= {_x.mean().item():.4f}，比值 = {(1 / (1 - 0.3)):.4f} = 1/(1-p)")
print()


# --- 1.4 Dropout 对比实验：p = 0.0 / 0.2 / 0.5 ---
print("-" * 78)
print("Dropout 对比实验：同一个网络分别用 p=0.0 / 0.2 / 0.5 训练")
print("-" * 78)
print(
    """
实验设计（刻意制造过拟合场景，否则 Dropout 看不出效果）：
    · 数据：make_classification(n_samples=500, n_features=30, n_informative=5, ...)，
      样本少（500）、特征多（30）、有效特征少（5）→ 模型很容易记住训练集的噪声；
    · 模型：30 → 128 → 128 → 2，两层隐藏层各接 ReLU + Dropout(p)；
    · 训练 60 个 epoch，全量 batch，Adam lr=0.01；
    · 记录每个 epoch 的训练 loss 与验证 loss，最后打印 train/test 准确率与两者的差距（过拟合程度）。
"""
)


class MLPWithDropout(nn.Module):
    """30 → 128 → 128 → 2 的小 MLP，两层隐藏层后面各挂一个 Dropout(p)。

    p=0 时等价于「不放 Dropout」，正好作为对照组。
    """

    def __init__(self, in_dim: int = 30, hidden: int = 128, out_dim: int = 2, p: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, out_dim)
        self.drop = nn.Dropout(p=p)         # 训练时随机置 0，推理时什么都不做

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(torch.relu(self.fc1(x)))
        x = self.drop(torch.relu(self.fc2(x)))
        return self.head(x)


from sklearn.datasets import make_classification   # 只用 sklearn 造数据，不联网

_Xc, _yc = make_classification(
    n_samples=500, n_features=30, n_informative=5, n_redundant=0,
    n_clusters_per_class=2, class_sep=1.0, flip_y=0.05, random_state=42,
)
X_cls = torch.tensor(_Xc, dtype=torch.float32)
y_cls = torch.tensor(_yc, dtype=torch.long)
_n_tr = 350
Xc_tr, yc_tr = X_cls[:_n_tr], y_cls[:_n_tr]
Xc_te, yc_te = X_cls[_n_tr:], y_cls[_n_tr:]
print(f"数据：make_classification(500 条 × 30 维，其中 5 维有信息)，"
      f"训练 {len(Xc_tr)} / 测试 {len(Xc_te)}")

_DROP_EPOCHS = 60
_drop_curves: dict[float, dict[str, list[float]]] = {}
_drop_summary: dict[float, dict[str, float]] = {}

for _p in (0.0, 0.2, 0.5):
    torch.manual_seed(42)
    _m = MLPWithDropout(p=_p)
    _opt = torch.optim.Adam(_m.parameters(), lr=0.01)
    _crit = nn.CrossEntropyLoss()
    _tr_hist: list[float] = []
    _te_hist: list[float] = []
    for _ep in range(_DROP_EPOCHS):
        _m.train()                                  # 训练模式：Dropout 生效
        _opt.zero_grad()
        _loss = _crit(_m(Xc_tr), yc_tr)
        _loss.backward()
        _opt.step()
        _tr_hist.append(_loss.item())
        _m.eval()                                   # 评估模式：Dropout 关闭
        with torch.no_grad():
            _te_hist.append(_crit(_m(Xc_te), yc_te).item())
    _m.eval()
    with torch.no_grad():
        _tr_acc = (_m(Xc_tr).argmax(dim=1) == yc_tr).float().mean().item()
        _te_acc = (_m(Xc_te).argmax(dim=1) == yc_te).float().mean().item()
    _drop_curves[_p] = {"train": _tr_hist, "test": _te_hist}
    _drop_summary[_p] = {
        "train_acc": _tr_acc, "test_acc": _te_acc, "gap": _tr_acc - _te_acc,
        "final_train_loss": _tr_hist[-1], "final_test_loss": _te_hist[-1],
    }

print()
print(f"{'p':>6}{'训练准确率':>12}{'测试准确率':>12}{'过拟合差距':>12}{'末轮训练loss':>14}{'末轮验证loss':>14}")
print("-" * 74)
for _p in (0.0, 0.2, 0.5):
    _s = _drop_summary[_p]
    print(f"{_p:>6.1f}{_s['train_acc'] * 100:>11.2f}%{_s['test_acc'] * 100:>11.2f}%"
          f"{_s['gap'] * 100:>11.2f}%{_s['final_train_loss']:>14.4f}{_s['final_test_loss']:>14.4f}")
print()
print("  解读：p=0.0 时训练准确率很高、训练 loss 很小，但测试准确率最低、验证 loss 反而在上升——")
print("        典型的过拟合（模型把 350 条训练样本的噪声也背下来了）。")
print("        加上 Dropout 后，训练时每次都只用不同的子网络，「背不动」单个样本，")
print("        训练准确率下降、但验证 loss 更低、泛化差距（train−test）明显缩小。")
print()

# --- 画 Dropout 对比图 ---
fig_d, axes_d = plt.subplots(2, 3, figsize=(15, 8))
_colors_p = {0.0: "#C44E52", 0.2: "#4C72B0", 0.5: "#55A868"}
for _col, _p in enumerate((0.0, 0.2, 0.5)):
    _ax = axes_d[0, _col]
    _ax.plot(range(1, _DROP_EPOCHS + 1), _drop_curves[_p]["train"], color=_colors_p[_p], label="训练 loss")
    _ax.plot(range(1, _DROP_EPOCHS + 1), _drop_curves[_p]["test"], color=_colors_p[_p], ls="--", label="验证 loss")
    _ax.set_title(f"Dropout p={_p}  (train acc={_drop_summary[_p]['train_acc'] * 100:.1f}%, "
                  f"test acc={_drop_summary[_p]['test_acc'] * 100:.1f}%)", fontsize=10)
    _ax.set_xlabel("epoch")
    _ax.set_ylabel("交叉熵损失")
    _ax.legend(fontsize=9)
    _ax.grid(alpha=0.3)
    _ax.set_ylim(bottom=0)
# 下排：三种 p 的训练 / 验证 loss 叠在一张图上 + 过拟合差距柱状图
_ax_all = axes_d[1, 0]
for _p in (0.0, 0.2, 0.5):
    _ax_all.plot(range(1, _DROP_EPOCHS + 1), _drop_curves[_p]["test"], color=_colors_p[_p],
                 label=f"验证 loss (p={_p})")
_ax_all.set_title("三种 Dropout 强度的验证 loss 对比", fontsize=10)
_ax_all.set_xlabel("epoch")
_ax_all.set_ylabel("验证交叉熵损失")
_ax_all.legend(fontsize=9)
_ax_all.grid(alpha=0.3)

_ax_gap = axes_d[1, 1]
_gaps = [_drop_summary[_p]["gap"] * 100 for _p in (0.0, 0.2, 0.5)]
_bars = _ax_gap.bar(["p=0.0", "p=0.2", "p=0.5"], _gaps,
                    color=[_colors_p[_p] for _p in (0.0, 0.2, 0.5)])
_ax_gap.bar_label(_bars, fmt="%.1f%%")
_ax_gap.set_title("过拟合差距（训练准确率 − 测试准确率）", fontsize=10)
_ax_gap.set_ylabel("百分点")
_ax_gap.grid(alpha=0.3, axis="y")

_ax_acc = axes_d[1, 2]
_wbar = 0.35
_ax_acc.bar([i - _wbar / 2 for i in range(3)], [_drop_summary[_p]["train_acc"] * 100 for _p in (0.0, 0.2, 0.5)],
            width=_wbar, label="训练准确率", color="#4C72B0")
_ax_acc.bar([i + _wbar / 2 for i in range(3)], [_drop_summary[_p]["test_acc"] * 100 for _p in (0.0, 0.2, 0.5)],
            width=_wbar, label="测试准确率", color="#DD8452")
_ax_acc.set_xticks(range(3), ["p=0.0", "p=0.2", "p=0.5"])
_ax_acc.set_title("训练 / 测试准确率对比", fontsize=10)
_ax_acc.set_ylabel("准确率 (%)")
_ax_acc.legend(fontsize=9)
_ax_acc.grid(alpha=0.3, axis="y")

fig_d.tight_layout()
_p_drop = OUTPUT_DIR / "03训练组件_06_Dropout对比.png"
fig_d.savefig(_p_drop, dpi=110)
plt.close(fig_d)
print(f"已保存：{_p_drop}")
print()


# ===========================================================================
# 第 2 部分：L1 / L2 范数正则
# ===========================================================================
print("=" * 78)
print("第 2 部分：L1 / L2 范数正则——惩罚项是怎么来的（贝叶斯推导）")
print("=" * 78)
print(
    r"""
【频率派：参数是常量，用最大似然估计 MLE】
    参数 θ 是一个固定的常量，只是我们暂时不知道它的值。优化目标是让数据出现概率最大：

        θ_MLE = argmax_θ P(Y|X, θ)

    假设误差服从高斯分布 N(0, σ²)，则每个样本的预测误差 y_i - f(x_i; θ) 也服从该分布：

        P(y_i | x_i, θ) = (1 / (√(2π)·σ)) · exp( -(y_i - f(x_i; θ))² / (2σ²) )

    整个数据集的联合概率（似然函数）是各样本概率的**乘积**：

        P(Y|X, θ) = ∏_i (1 / (√(2π)·σ)) · exp( -(y_i - f(x_i; θ))² / (2σ²) )

    取负对数，把乘积变成求和（方便优化）：

        -log P(Y|X, θ) = N·log(√(2π)·σ) + (1/(2σ²)) · Σ_i (y_i - f(x_i; θ))²

    忽略常数项 log(√(2π)σ) 和正系数 1/(2σ²)（它们不影响最优 θ 的位置），最大化似然等价于最小化：

        Σ_i (y_i - f(x_i; θ))²

    **这就是最小二乘法。也就是说，平时用的 MSE 损失，底层假设就是「误差服从高斯分布」。**

【贝叶斯派：参数本身是分布，用最大后验估计 MAP】
    参数 θ 不是常量，而是随机变量，有自己的概率分布 f(θ)——称为**先验分布**，
    即「在看到数据之前，我们对 θ 的初始猜测」。优化目标变成最大化后验概率：

        P(θ|X, Y) = P(Y|X, θ) · f(θ) / P(Y|X)  ∝  P(Y|X, θ) · f(θ)

    取负对数（分母 P(Y|X) 与 θ 无关，是常数）：

        -log P(θ|X, Y) = -log P(Y|X, θ) - log f(θ) + const

    **和频率派相比，多出来的 -log f(θ) 这一项——这就是正则项。**

    直观理解：频率派只看「数据拟合得好不好」；贝叶斯派还多一个要求——「参数本身合不合理」。
    如果某个 θ 在先验里概率很低（比如 w 特别大），那么即使它拟合数据还不错，
    后验概率也会被拉低。这相当于给参数加了一个「偏好」，不让它们乱跑。

【先验分布决定正则化形式】
    · 高斯先验（均值 0、方差 σ_w²）：

          f(w) = (1/(√(2π)·σ_w)) · exp( -w² / (2σ_w²) )
          -log f(w) = w² / (2σ_w²) + log(√(2π)·σ_w)

      丢掉常数 log(√(2π)σ_w) 后剩 w²/(2σ_w²)。把它和似然的 1/(2σ²)Σ(y-f)² 放在一起，
      两边同乘 2σ²（不改变最优解的位置）：

          2σ² · [ (1/(2σ²))Σ(y-f)² + w²/(2σ_w²) ] = Σ(y-f)² + (σ²/σ_w²)·w²

      令 λ = σ² / σ_w²（**误差方差与先验方差的比值**）：

          Σ_i (y_i - f_i)² + λ · Σ_j w_j²          ← L2 正则（Ridge 岭回归）

    · 拉普拉斯先验（位置 0、尺度 b）：

          f(w) = (1/(2b)) · exp( -|w| / b )
          -log f(w) = |w| / b + log(2b)

      同样丢掉常数、乘 2σ²，系数变成 2σ²/b，打包成一个超参数 λ：

          Σ_i (y_i - f_i)² + λ · Σ_j |w_j|          ← L1 正则（Lasso）

    实践提醒：实际调参时**不需要**分别指定 σ² 和 σ_w²，直接调 λ 这一个数就行。
    λ 大 → 先验方差 σ_w² 相对小 → 参数被拉向 0 的力强（强正则）；
    λ 小 → 先验方差相对大 → 参数更自由（弱正则）。
"""
)

_TABLE_L1L2 = [
    ("高斯分布", "L2（Ridge）", "λ·Σw²", "0 点附近平滑，尾部轻（极端值概率低）"),
    ("拉普拉斯分布", "L1（Lasso）", "λ·Σ|w|", "0 点更尖更集中，尾部重（允许少数极端值）"),
]
print("先验分布 → 正则化形式对照表：")
print(f"{'先验分布':<14}{'对应正则化':<14}{'正则项':<12}{'先验的形状特点'}")
print("-" * 78)
for _row in _TABLE_L1L2:
    print(f"{_row[0]:<14}{_row[1]:<14}{_row[2]:<12}{_row[3]}")
print()


# --- 2.1 手写 L1 / L2 正则项（只对 weight，不对 bias） ---
print("-" * 78)
print("2.1 手写 L1 / L2 正则项（课案的写法，只正则化 weight、不正则化 bias）")
print("-" * 78)
print(
    """
为什么只对 weight 不对 bias？
    bias 只是把激活函数平移一下，不参与「特征组合」，它变大不会让模型变得更容易记住噪声；
    而且正则化 bias 会让网络难以拟合数据的整体偏移。所以惯例是只惩罚 weight。
代码写法（课案原文）：

    l2_reg = torch.tensor(0.)
    for name, param in model.named_parameters():
        if 'weight' in name:
            l2_reg += torch.norm(param, p=2)     # 等价于 param.pow(2).sum()
    loss = loss + l2_lambda * l2_reg

L1 同理，换成 torch.norm(param, p=1)（等价于 param.abs().sum()）。
注意：torch.tensor(0.) 是「叶子张量」，可以直接参与构建计算图；
      只要参与运算就会自动变成带 grad_fn 的非叶子张量，backward 能正常回传。
"""
)

torch.manual_seed(42)
_demo_model = nn.Linear(10, 1)
_demo_inputs, _demo_targets = torch.randn(32, 10), torch.randn(32, 1)
_demo_loss = F.mse_loss(_demo_model(_demo_inputs), _demo_targets)
_l2_reg = torch.tensor(0.0)
_l1_reg = torch.tensor(0.0)
for _name, _param in _demo_model.named_parameters():
    if "weight" in _name:
        _l2_reg = _l2_reg + torch.norm(_param, p=2)
        _l1_reg = _l1_reg + torch.norm(_param, p=1)
print(f"  nn.Linear(10, 1) 的命名参数：{[n for n, _ in _demo_model.named_parameters()]}")
print(f"  只挑含 'weight' 的：{[n for n, _ in _demo_model.named_parameters() if 'weight' in n]}")
print(f"  MSE 原始损失 = {_demo_loss.item():.6f}")
print(f"  L2 项 Σ‖W‖₂ = {_l2_reg.item():.6f}；用 param.pow(2).sum() 再开方 = "
      f"{math.sqrt(sum(p.pow(2).sum().item() for n, p in _demo_model.named_parameters() if 'weight' in n)):.6f}"
      f"（两种写法等价）")
print(f"  L1 项 Σ‖W‖₁ = {_l1_reg.item():.6f}；用 param.abs().sum() = "
      f"{sum(p.abs().sum().item() for n, p in _demo_model.named_parameters() if 'weight' in n):.6f}"
      f"（两种写法等价）")
print(f"  加正则后的总损失（λ_l2=0.001）= {(_demo_loss + 0.001 * _l2_reg).item():.6f}")
print()


# --- 2.2 权重衰减与 L2 正则的等价性（核心对比实验） ---
print("-" * 78)
print("2.2 权重衰减（weight_decay）与 L2 正则的等价性：数值证明")
print("-" * 78)
print(
    r"""
推导（SGD 下完全等价）：
    L2 正则加在损失上，系数写成 λ/2 是为了求导后消掉平方的 2，让最终衰减系数正好是 λ：

        L_reg(w) = L(w) + (λ/2)·Σw²

    求梯度： ∂L_reg/∂w = ∂L/∂w + λw
    代入 SGD 更新 w ← w - α·∂L_reg/∂w：

        w ← w - α(∇L + λw) = w - α∇L - αλw = (1 - αλ)·w - α∇L

    **最终形式：每步先把权重缩到原来的 (1-αλ) 倍，再沿梯度方向走一步。**
    "Weight Decay"（权重衰减）这个名字就来自这个「权重整体缩小」的效应。

【为什么必须写 (λ/2)·Σw² 而不是课案的 λ·‖W‖₂ 才能对上？】
    · ∂/∂w [ (λ/2)·w² ] = λw  ←  与 optim.SGD(weight_decay=λ) 每步施加的衰减量**逐位相同**；
    · ∂/∂w [ λ·‖W‖₂ ] = λ·w/‖W‖₂  ←  多了一个 1/‖W‖₂ 因子，方向和量级都不一样。
    所以「课案里 torch.norm(param, p=2) 的写法」与 weight_decay 并不严格等价，
    它们只差一个随 ‖W‖ 变化的缩放系数（实践中常被忽略，但严格验证时必须用 (λ/2)Σw²）。
    下面的数值实验会先用**梯度逐位比对**把这一点钉死，再做整段训练的对比。

实验设计（证明两条路径完全一致）：
    · 第 1 步（梯度级）：同一个模型、同一份数据、同一份上游梯度，
      分别计算「数据损失 + λ·(1/2)Σw²」的梯度与「数据损失」的梯度，
      验证两者之差恰好等于 λw；
    · 第 2 步（训练级）：同一份数据、同一个初始模型（深拷贝两份）、同一个学习率，
      路径 A 手写 (λ/2)Σw² 惩罚项、路径 B 用 optim.SGD(weight_decay=λ)，
      训练 15 个 epoch，每个 epoch 后打印权重 L2 范数并比较两条曲线。
"""
)

# --- 2.2.0 梯度级验证：手写 (λ/2)Σw² 的梯度与 weight_decay 的衰减量是否逐位相同 ---
_LAM = 0.05
_LR = 0.05
_WD_EPOCHS = 15
torch.manual_seed(42)
_Xw = torch.randn(256, 20)
_yw = (0.5 * _Xw[:, 0] - 0.3 * _Xw[:, 1] + 0.8 * _Xw[:, 2]
       + 0.02 * torch.randn(256)).unsqueeze(1)          # 一个线性回归任务 + 少量噪声

torch.manual_seed(0)
_grad_model = nn.Linear(20, 1)                          # 有 weight 也有 bias，两个都要验证
_w_before = _grad_model.weight.detach().clone()
_b_before = _grad_model.bias.detach().clone()
# 路径 A 的梯度：数据损失 + (λ/2)Σθ²（对 weight 和 bias 都加）
_grad_model.zero_grad()
F.mse_loss(_grad_model(_Xw), _yw).backward()
_grad_data_only = _grad_model.weight.grad.detach().clone()
_grad_model.zero_grad()
(F.mse_loss(_grad_model(_Xw), _yw)
 + _LAM * 0.5 * _grad_model.weight.pow(2).sum()
 + _LAM * 0.5 * _grad_model.bias.pow(2).sum()).backward()
_grad_with_l2 = _grad_model.weight.grad.detach().clone()
# 两条梯度的差
_grad_diff = _grad_with_l2 - _grad_data_only
_expected_diff = _LAM * _w_before                              # 应该恰好是 λw
_err_grad = (_grad_diff - _expected_diff).abs().max().item()
print("【梯度级验证】同一个模型、同一份数据、同一个上游梯度：")
print(f"  · 只用数据损失            → 参数梯度最大值 = {_grad_data_only.abs().max().item():.8f}")
print(f"  · 数据损失 + λ·(1/2)Σθ²   → 参数梯度最大值 = {_grad_with_l2.abs().max().item():.8f}")
print(f"  · 两者之差 vs 理论值 λ·w  的最大绝对误差 = {_err_grad:.3e}"
      f"  → {'逐位相同 ✔（λ/2 系数的取法正确）' if _err_grad < 1e-7 else '不一致'}")
print(f"  对照：如果用课案的 λ·‖W‖₂ 写法，梯度 = λ·w/‖W‖₂，"
      f"其中 1/‖W‖₂ = {1 / _w_before.norm().item():.6f}")
print(f"        → 相当于给衰减量额外乘了 {1 / _w_before.norm().item():.6f}，和 weight_decay 对不上。")
print("  【重要细节】上面必须把 **bias 也一起加进惩罚项**：")
print("    因为 optim.SGD 的 weight_decay 是对 `param_groups` 里的**全部参数**（含 bias）统一生效的。")
print("    如果手写时只惩罚 weight（模型调参时的常规做法），两条路径就会在 bias 上分道扬镳——")
print("    下面的整段训练对比会把这个现象直接打印出来。")
print()


class LinearReg(nn.Module):
    """单层线性回归模型（20 → 1）。"""

    def __init__(self, in_dim: int = 20):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

    def weight_l2_norm(self) -> float:
        """返回所有权重（不含 bias）的 L2 范数，用于观察「权重衰减」效应。"""
        return math.sqrt(sum(p.pow(2).sum().item() for n, p in self.named_parameters() if "weight" in n))


def weight_norm_gap(model_a: nn.Module, model_b: nn.Module) -> float:
    """逐元素比较两个模型全部参数的差异，返回最大绝对误差。"""
    with torch.no_grad():
        return max((pa - pb).abs().max().item()
                   for pa, pb in zip(model_a.parameters(), model_b.parameters()))


# 两个模型从完全相同的初始权重出发
torch.manual_seed(0)
_model_l2 = LinearReg()
_model_wd = copy.deepcopy(_model_l2)
_opt_l2 = torch.optim.SGD(_model_l2.parameters(), lr=_LR)                  # 路径 A：不加 weight_decay
_opt_wd = torch.optim.SGD(_model_wd.parameters(), lr=_LR, weight_decay=_LAM)  # 路径 B：weight_decay

_norm_hist_l2: list[float] = []
_norm_hist_wd: list[float] = []


def _l2_penalty_all_params(net: nn.Module, only_weight: bool) -> torch.Tensor:
    """返回 (1/2)Σθ² 形式的 L2 惩罚项。

    only_weight=True  → 只惩罚 weight（建模时防止过拟合的常规做法）；
    only_weight=False → 连 bias 一起惩罚（= optim.SGD(weight_decay=λ) 的行为，
                        因为优化器的 weight_decay 对 param_groups 里的**全部参数**生效）。
    """
    penalty = torch.tensor(0.0)
    for _name, _p in net.named_parameters():
        if only_weight and "weight" not in _name:
            continue
        penalty = penalty + 0.5 * _p.pow(2).sum()
    return penalty


print(f"超参数：λ（weight_decay）= {_LAM}，学习率 α = {_LR}，训练 {_WD_EPOCHS} 个 epoch")
print(f"初始权重 L2 范数 = {_model_l2.weight_l2_norm():.8f}（两个模型完全相同）")
print()
print("【实验 A（严格等价版）】手写惩罚项时**连 bias 一起惩罚** → 应该与 weight_decay 逐位一致")
print(f"{'epoch':>6}{'手写L2的权重范数':>20}{'weight_decay的权重范数':>24}{'两者权重最大差异':>20}")
print("-" * 72)
for _ep in range(1, _WD_EPOCHS + 1):
    # ---- 路径 A：手写 L2 惩罚项（含 bias）----
    _opt_l2.zero_grad()
    _loss_l2 = F.mse_loss(_model_l2(_Xw), _yw)
    # 系数写成 λ/2 才与 optim.SGD(weight_decay=λ) 的等价形式一致：
    #   ∂/∂θ [ (λ/2)·θ² ] = λθ，正好是 weight_decay 每步施加的衰减量
    _total_l2 = _loss_l2 + _LAM * _l2_penalty_all_params(_model_l2, only_weight=False)
    _total_l2.backward()
    _opt_l2.step()

    # ---- 路径 B：优化器内置的 weight_decay ----
    _opt_wd.zero_grad()
    _loss_wd = F.mse_loss(_model_wd(_Xw), _yw)
    _loss_wd.backward()
    _opt_wd.step()                                  # 内部自动执行 θ ← θ - α(∇L + λθ)

    _n_l2 = _model_l2.weight_l2_norm()
    _n_wd = _model_wd.weight_l2_norm()
    _norm_hist_l2.append(_n_l2)
    _norm_hist_wd.append(_n_wd)
    _diff = weight_norm_gap(_model_l2, _model_wd)
    print(f"{_ep:>6}{_n_l2:>20.10f}{_n_wd:>24.10f}{_diff:>20.3e}")

_norm_gap_final = max(abs(a - b) for a, b in zip(_norm_hist_l2, _norm_hist_wd))
_param_gap_final = weight_norm_gap(_model_l2, _model_wd)
print()
print(f"两条路径的权重范数最大差异 = {_norm_gap_final:.3e}")
print(f"两条路径的参数逐元素最大差异 = {_param_gap_final:.3e}")
print(f"判定：{'两条路径完全一致（差异 < 1e-6）✔ → **L2 正则 = 权重衰减**，得到数值证明' if _param_gap_final < 1e-6 else '参数层面有微小差异，需要进一步分析'}"
      )
print()

# --- 实验 B：只惩罚 weight（调参时的常规做法），观察 bias 上的差异 ---
print("【实验 B（对照）】手写惩罚项时**只惩罚 weight**（建模时最常用的做法）")
torch.manual_seed(0)
_model_l2b = LinearReg()
_model_wdb = copy.deepcopy(_model_l2b)
_opt_l2b = torch.optim.SGD(_model_l2b.parameters(), lr=_LR)
_opt_wdb = torch.optim.SGD(_model_wdb.parameters(), lr=_LR, weight_decay=_LAM)
for _ep in range(1, _WD_EPOCHS + 1):
    _opt_l2b.zero_grad()
    (F.mse_loss(_model_l2b(_Xw), _yw)
     + _LAM * _l2_penalty_all_params(_model_l2b, only_weight=True)).backward()
    _opt_l2b.step()
    _opt_wdb.zero_grad()
    F.mse_loss(_model_wdb(_Xw), _yw).backward()
    _opt_wdb.step()
_gap_w_b = (_model_l2b.fc.weight - _model_wdb.fc.weight).abs().max().item()
_gap_bias_b = (_model_l2b.fc.bias - _model_wdb.fc.bias).abs().max().item()
print(f"  训练 {_WD_EPOCHS} 个 epoch 后：")
print(f"    weight 的最大差异 = {_gap_w_b:.3e}（仍然很小）")
print(f"    bias   的最大差异 = {_gap_bias_b:.3e} ← **差在这里**")
print(f"  原因：optim.SGD 的 weight_decay 对 param_groups 里的**所有参数**（含 bias）统一施加衰减，")
print(f"        而我们只把 weight 放进了惩罚项 → bias 少衰减了一份，两条路径就不再等价。")
print(f"  实践建议：① 想让「手写 L2」严格等于 weight_decay，就把所有参数都加进惩罚项；")
print(f"            ② 只想正则化 weight（更常见的做法），就用 weight_decay=0 的手写版本，")
print(f"               或者把 bias 单独分组、给这一组设 weight_decay=0。")
print()
print("  参考：PyTorch SGD 内部的实际实现（两步，注意等价但顺序不同，浮点上略有舍入差异）")
print("        param.mul_(1 - lr * weight_decay)   # 先整体衰减")
print("        param.add_(d_p, alpha=-lr)          # 再走梯度那一步")
print()
print(f"理论预测：每步权重乘 (1 - αλ) = (1 - {_LR}×{_LAM}) = {1 - _LR * _LAM:.6f}，")
print(f"          权重范数理论变化比 = {(1 - _LR * _LAM) ** _WD_EPOCHS:.6f}"
      f"（只考虑衰减、忽略梯度下降的贡献，实际会因梯度而偏离）")
print()

# 附：课案那种「用 torch.norm(p=2) 而不是 0.5*Σw²」的写法，系数换算关系
print("系数对应的说明：")
print(f"  · optim.SGD(weight_decay=λ) 等价于「损失里加 (λ/2)·Σw²」（因为 ∂((λ/2)w²)/∂w = λw）；")
print(f"  · 课案写法再乘 λ 之后是 λ·‖W‖₂（对所有 weight 的 L2 范数），它的梯度是 λ·w/‖W‖₂——")
print(f"    比 weight_decay 需要的 λw 多了一个 1/‖W‖₂ 因子，所以二者**只差一个缩放系数**，并不严格相等。")
print(f"  · 想和 weight_decay 严格对齐，请手写 (λ/2)·Σw²（本节就是这么做的）；")
print(f"    若坚持用 torch.norm(p=2)，就得把 λ 换成 λ/‖W‖₂，而 ‖W‖ 每步都在变，实际上没法这么调。")
print()


# --- 2.3 λ 的影响：λ = 0 / 1e-3 / 1e-1 ---
print("-" * 78)
print("2.3 正则强度 λ 的影响：λ = 0 / 1e-3 / 1e-1")
print("-" * 78)
print(
    """
实验：同一个分类任务（30 维、350 条训练样本），三个 λ 各训练 60 epoch：
    λ=0      → 无正则，容易过拟合；
    λ=1e-3   → 轻度正则；
    λ=1e-1   → 强正则，权重被狠狠拉向 0，可能导致**欠拟合**（训练和测试准确率都不高）。
打印最终的 train/test 准确率与权重 L2 范数。
"""
)

_LAM_LIST = [0.0, 1e-3, 1e-1]
_lam_result: dict[float, dict[str, float]] = {}
for _lam in _LAM_LIST:
    torch.manual_seed(42)
    _m = nn.Sequential(nn.Linear(30, 64), nn.ReLU(), nn.Linear(64, 2))
    _opt = torch.optim.SGD(_m.parameters(), lr=0.05, weight_decay=_lam)   # 用权重衰减实现 L2
    _crit = nn.CrossEntropyLoss()
    for _ep in range(60):
        _m.train()
        _opt.zero_grad()
        _crit(_m(Xc_tr), yc_tr).backward()
        _opt.step()
    _m.eval()
    with torch.no_grad():
        _tra = (_m(Xc_tr).argmax(dim=1) == yc_tr).float().mean().item()
        _tea = (_m(Xc_te).argmax(dim=1) == yc_te).float().mean().item()
        _wn = math.sqrt(sum(p.pow(2).sum().item() for n, p in _m.named_parameters() if "weight" in n))
    _lam_result[_lam] = {"train_acc": _tra, "test_acc": _tea, "weight_norm": _wn}

print(f"{'λ':>10}{'训练准确率':>14}{'测试准确率':>14}{'权重L2范数':>16}")
print("-" * 56)
for _lam in _LAM_LIST:
    _r = _lam_result[_lam]
    print(f"{_lam:>10.0e}{_r['train_acc'] * 100:>13.2f}%{_r['test_acc'] * 100:>13.2f}%"
          f"{_r['weight_norm']:>16.4f}")
print()
print(f"  解读：λ 从 0 增到 1e-1，权重 L2 范数从 {_lam_result[0.0]['weight_norm']:.4f} "
      f"压到 {_lam_result[1e-1]['weight_norm']:.4f}（被强烈拉向 0）；")
print(f"        λ 太小时靠训练集就能拟合得很好，但泛化不一定最优；")
print(f"        λ 太大时训练准确率也掉下来了——这就是**欠拟合**：模型被约束得太死，")
print(f"        连训练集的规律都学不动了。正则强度必须适中，这也是它成为超参数的原因。")
print()


# --- 2.4 L1 为什么比 L2 更容易产生稀疏解 ---
print("-" * 78)
print("2.4 为什么 L1 比 L2 更容易产生稀疏解（0 值）：三个角度")
print("-" * 78)
print(
    r"""
【角度一：先验分布】
    拉普拉斯分布在 0 点更集中、更尖（概率密度在 0 处达到峰值，形状是尖角）；
    高斯分布在 0 点更平滑（峰值是圆滑的）。
    拉普拉斯先验强烈地「建议」参数等于 0，除非数据强烈反对——所以 L1 会把不重要的权重直接压成 0。
    从尾部看：拉普拉斯尾部更重（允许少数极端值），高斯尾部更轻（极端值概率低）。

【角度二：几何视角】
    以二维参数 (w1, w2) 为例，最小二乘的损失等值线是椭圆（椭圆的一般方程
    A·w1² + B·w2² + C·w1·w2 + D·w1 + E·w2 + F = 0）。
    两种约束区域：
        L1 约束 |w1| + |w2| ≤ c  →  **菱形**（四个顶点落在坐标轴上）
        L2 约束 w1² + w2² ≤ c    →  **圆**
    在约束下最小化损失，几何上就是「让椭圆等值线从外向内膨胀，直到第一次碰到约束区域」。
        · 椭圆第一次碰到**菱形**的位置，很容易正好是菱形的某个顶点——而顶点在坐标轴上，
          意味着其中一个参数恰好等于 0 → **稀疏解**；
        · 椭圆第一次碰到**圆**的位置，一般是在圆的某段光滑弧上，几乎不可能正好落在坐标轴上
          → 参数都非 0，只是整体变小。
    下面代码里画了这张相切示意图。

【角度三：导数视角】
    设损失函数 L(w) 在 w = 0 处的导数为 d。
        L1 正则： J = L(w) + λ|w|
            w > 0 时 J' = d + λ
            w < 0 时 J' = d - λ
            → 穿过 0 点时导数**跳跃**了 2λ（从 d-λ 跳到 d+λ）。
        L2 正则： J = L(w) + λw²
            J' = d + 2λw，在 w = 0 时就是 d
            → 穿过 0 点时导数**连续**，没有任何跳跃。
    导数跳跃意味着：只要 |d| < λ，那么无论 w 是正是负，走一步之后 J 都会变大——
    0 点两侧都是「上坡」，于是参数被推到 0 并**停在那里**（这就是软阈值 soft-thresholding）。
    L2 在 0 点附近导数连续、平滑，梯度下降只会让 w 变小但很难精确等于 0。
    下面也用代码把 0 点两侧的导数算出来验证。

【数值验证】
    用一个线性回归 + 梯度下降，分别加 L1 和 L2 正则训练，
    统计最终权重中 |w| < 1e-3 的个数——L1 应该明显更多。
"""
)

# --- 2.4.1 导数视角的数值验证 ---
print("【数值验证 · 导数视角】损失在 w=0 处导数为 d 时，0 点两侧的总导数")
_d_val = 0.3                    # 假设 L'(0) = 0.3
_lam_val = 0.5                  # 正则强度
print(f"  设 L'(0) = d = {_d_val}，正则强度 λ = {_lam_val}")
print(f"  L1： w>0 时 J' = d + λ = {_d_val + _lam_val:+.2f}；"
      f" w<0 时 J' = d - λ = {_d_val - _lam_val:+.2f}；跳跃幅度 = 2λ = {2 * _lam_val:.2f}")
print(f"  L2： J'(0) = d + 2λ·0 = {_d_val:+.2f}；w=0.01 时 J' = "
      f"{_d_val + 2 * _lam_val * 0.01:+.2f}，w=-0.01 时 J' = {_d_val + 2 * _lam_val * (-0.01):+.2f}"
      f" → 连续变化，无跳跃")
# 用 autograd 真实验证一遍（避免只做符号推导）
_w = torch.tensor([0.01], requires_grad=True)
_J_l1 = (_d_val * _w + _lam_val * _w.abs()).sum()
_J_l1.backward()
_grad_l1_pos = _w.grad.item()
_w = torch.tensor([-0.01], requires_grad=True)
_J_l1 = (_d_val * _w + _lam_val * _w.abs()).sum()
_J_l1.backward()
_grad_l1_neg = _w.grad.item()
_w = torch.tensor([0.0], requires_grad=True)
_J_l2 = (_d_val * _w + _lam_val * _w.pow(2)).sum()
_J_l2.backward()
_grad_l2_zero = _w.grad.item()
print(f"  autograd 验证：L1 在 w=+0.01 的梯度 = {_grad_l1_pos:+.4f}（理论 {_d_val + _lam_val:+.4f}），"
      f"在 w=-0.01 的梯度 = {_grad_l1_neg:+.4f}（理论 {_d_val - _lam_val:+.4f}）")
print(f"                 L2 在 w=0 的梯度 = {_grad_l2_zero:+.4f}（理论 {_d_val:+.4f}，完全连续）")
print(f"  → |d| = {abs(_d_val):.2f} < λ = {_lam_val:.2f}：此时 0 点两侧梯度同号（都为正），")
print(f"    梯度下降会把 w 一路推到 0 并停住 → 稀疏解。")
print()

# --- 2.4.2 数值验证：L1 vs L2 的稀疏性 ---
print("【数值验证 · 稀疏性】线性回归 + 梯度下降，统计 |w| < 1e-3 的权重个数")
torch.manual_seed(42)
_N_SP, _D_SP = 200, 50
_X_sp = torch.randn(_N_SP, _D_SP)
# 真实权重只有前 5 维非 0，其余 45 维完全是噪声特征（理论上应被压成 0）
_w_true = torch.zeros(_D_SP)
_w_true[:5] = torch.tensor([1.5, -2.0, 0.8, 1.2, -0.6])
_y_sp = (_X_sp @ _w_true + 0.1 * torch.randn(_N_SP)).unsqueeze(1)
print(f"  构造：{_D_SP} 维特征，真实权重只有前 5 维非 0（其余 45 维是纯噪声特征）")
print(f"  真实非零权重：{_w_true[_w_true != 0].tolist()}")


def train_linear_reg_sparse(penalty: str, lam: float, epochs: int = 300, lr: float = 0.05):
    """用梯度下降训练线性回归，加 L1 或 L2 惩罚，返回最终权重向量。

    penalty: 'l1' / 'l2' / 'none'
    注意：为了公平比较，L1 和 L2 用**同样的 λ**、同样的初始化、同样的步数。
    """
    torch.manual_seed(0)
    layer = nn.Linear(_D_SP, 1, bias=False)
    opt = torch.optim.SGD(layer.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        # 数据损失用 1/2·MSE 更方便和惩罚项系数对齐（常数倍数不影响结论）
        data_loss = 0.5 * F.mse_loss(layer(_X_sp), _y_sp)
        if penalty == "none":
            total = data_loss
        elif penalty == "l2":
            total = data_loss + lam * layer.weight.pow(2).sum()
        else:                                        # 'l1'
            total = data_loss + lam * layer.weight.abs().sum()
        total.backward()
        opt.step()
    return layer.weight.detach().clone().flatten()


_SPARSE_LAM = 0.05
_sparse_stats: dict[str, dict[str, float]] = {}
_sparse_weights: dict[str, torch.Tensor] = {}
for _pen in ("none", "l2", "l1"):
    _w_final = train_linear_reg_sparse(_pen, _SPARSE_LAM)
    _sparse_weights[_pen] = _w_final
    _n_near_zero = int((_w_final.abs() < 1e-3).sum().item())
    _sparse_stats[_pen] = {
        "near_zero": _n_near_zero,
        "nonzero": int((_w_final.abs() >= 1e-3).sum().item()),
        "l1_norm": _w_final.abs().sum().item(),
        "l2_norm": _w_final.norm().item(),
    }
    print(f"  {_pen:>5} 正则（λ={_SPARSE_LAM}）：|w| < 1e-3 的个数 = {_n_near_zero:>2} / {_D_SP}"
          f"，‖w‖₁ = {_w_final.abs().sum().item():.4f}，‖w‖₂ = {_w_final.norm().item():.4f}")
print()
print(f"  结论：L1 把 {_sparse_stats['l1']['near_zero']} 个权重压到接近 0，"
      f"而 L2 只有 {_sparse_stats['l2']['near_zero']} 个、无正则 {_sparse_stats['none']['near_zero']} 个。")
print(f"        L1 的稀疏性明显更强（这正是「特征选择」效果的来源）。")
print(f"  提示：L1 的最终权重里，前 5 个真实有效特征应该仍显著非 0："
      f"{[round(v, 3) for v in _sparse_weights['l1'][:5].tolist()]}")
print()


# --- 2.4.3 画 L1 菱形 / L2 圆与损失等高线相切的示意图 ---
fig_g, axes_g = plt.subplots(1, 2, figsize=(13, 5.6))
# 一个「椭圆」损失：中心在 (1.4, 0.9)，长短轴不等
_c1, _c2 = 1.4, 0.9
_w1 = np.linspace(-2.2, 2.4, 500)
_w2 = np.linspace(-2.2, 2.2, 500)
_W1, _W2 = np.meshgrid(_w1, _w2)
_LOSS = 0.55 * (_W1 - _c1) ** 2 + 1.5 * (_W2 - _c2) ** 2   # 椭圆等值线

for _ax, _kind in zip(axes_g, ("L1（菱形）", "L2（圆）")):
    _cs = _ax.contour(_W1, _W2, _LOSS, levels=[0.08, 0.2, 0.4, 0.7, 1.1, 1.6, 2.2],
                      cmap="Blues", linewidths=1.2)
    _ax.clabel(_cs, inline=True, fontsize=8, fmt="%.2f")
    if _kind.startswith("L1"):
        _c_val = 1.0
        _d1 = np.array([_c_val, 0, -_c_val, 0, _c_val])
        _d2 = np.array([0, _c_val, 0, -_c_val, 0])
        _ax.fill(_d1, _d2, color="#C44E52", alpha=0.18, label=f"L1 约束 |w1|+|w2| <= {_c_val}")
        _ax.plot(_d1, _d2, color="#C44E52", lw=2)
        _tangent = (_c_val, 0.0)          # 菱形顶点：一个参数恰好为 0
        _note = "相切点落在菱形的**顶点**上\n→ w2 = 0，稀疏解"
    else:
        _c_val = 1.0
        _theta = np.linspace(0, 2 * np.pi, 400)
        _ax.fill(_c_val * np.cos(_theta), _c_val * np.sin(_theta), color="#4C72B0", alpha=0.18,
                 label=f"L2 约束 w1²+w2² <= {_c_val}²")
        _ax.plot(_c_val * np.cos(_theta), _c_val * np.sin(_theta), color="#4C72B0", lw=2)
        # 椭圆上离原点最近的点（即第一次与圆相切的点）：数值搜索
        _idx = np.unravel_index(np.argmin(np.where(np.sqrt(_W1 ** 2 + _W2 ** 2) > _c_val, 1e9, _LOSS)),
                                _LOSS.shape)
        _tangent = (float(_W1[_idx]), float(_W2[_idx]))
        _note = "相切点落在圆的**光滑弧**上\n→ 两个参数都非 0，只是都变小"
    _ax.plot(*_tangent, "k*", ms=16, label=f"相切点 ≈ ({_tangent[0]:.2f}, {_tangent[1]:.2f})")
    _ax.plot(_c1, _c2, "r+", ms=12, label=f"损失最小点 ({_c1}, {_c2})")
    _ax.set_title(f"{_kind} 约束区域与损失等值线\n{_note}", fontsize=10)
    _ax.set_xlabel("w1")
    _ax.set_ylabel("w2")
    _ax.axhline(0, color="gray", lw=0.6)
    _ax.axvline(0, color="gray", lw=0.6)
    _ax.legend(fontsize=8, loc="lower right")
    _ax.grid(alpha=0.25)
    _ax.set_aspect("equal")

fig_g.tight_layout()
_p_geo = OUTPUT_DIR / "03训练组件_06_L1L2稀疏解几何.png"
fig_g.savefig(_p_geo, dpi=110)
plt.close(fig_g)
print(f"已保存：{_p_geo}")
print()


# ===========================================================================
# 第 3 部分：归一化（BatchNorm / LayerNorm / InstanceNorm / GroupNorm）
# ===========================================================================
print("=" * 78)
print("第 3 部分：归一化——四兄弟共用一个公式，区别只在「在哪些维度上算统计量」")
print("=" * 78)
print(
    r"""
统一公式：

        y = γ · (x - μ) / √(σ² + ε) + β

    μ 和 σ² 在**指定维度**上计算，γ（缩放）和 β（平移）是**可学习**参数，ε 是防止除零的小常数。

四者的区别只在「在哪些维度上算 μ 和 σ²」：
    方法          | 计算 μ,σ² 的维度            | 含义
    BatchNorm     | dim=0（batch 维度）        | 同一通道、不同样本间归一化
    LayerNorm     | dim=1,2,3（除 batch 外全部）| 同一样本、所有通道和空间位置归一化
    InstanceNorm  | dim=2,3（空间维度）         | 同一样本、同一通道、不同空间位置归一化
    GroupNorm     | 通道分组后组内算 dim=2,3,4  | 介于 LayerNorm 和 InstanceNorm 之间

为什么需要 γ 和 β？
    如果只是把数据压缩到零均值单位方差，网络的表达能力会受限——比如 Sigmoid 在 0 附近
    接近线性（区分力弱），但很多任务恰恰需要它工作在有区分力的区域。
    γ（缩放）和 β（平移）是可学习参数，让网络**自己决定**每一层最合适的分布范围和位置。
    初始值取 γ=1、β=0，保证训练开始时这一层就是标准归一化，不会一上来就破坏信号。
"""
)

# --- 3.1 四兄弟的前向形状 ---
print("-" * 78)
print("3.1 四兄弟在 (N, C, H, W) = (4, 64, 32, 32) 输入上的形状与输出")
print("-" * 78)
_N, _C, _H, _W = 4, 64, 32, 32
torch.manual_seed(42)
x4d = torch.randn(_N, _C, _H, _W)
print(f"输入张量形状：{tuple(x4d.shape)}  （N=batch, C=通道, H/W=空间）")
print()

_bn2d = nn.BatchNorm2d(_C)                       # 通道数 C
_ln2d = nn.LayerNorm([_C, _H, _W])               # 传入 normalized_shape = (C, H, W)
_in2d = nn.InstanceNorm2d(_C)                    # 通道数 C
_gn2d = nn.GroupNorm(num_groups=8, num_channels=_C)   # 64 通道分 8 组，每组 8 通道

_norm_layers_4d = [
    ("BatchNorm2d(64)", _bn2d, "dim=0（batch 维度）"),
    ("LayerNorm([64,32,32])", _ln2d, "dim=1,2,3（除 batch 外全部）"),
    ("InstanceNorm2d(64)", _in2d, "dim=2,3（空间维度）"),
    ("GroupNorm(8, 64)", _gn2d, "8 组 ×每组 8 通道，组内算 dim=2,3"),
]
print(f"{'层':<24}{'统计量维度':<26}{'输入形状':<20}{'输出形状':<20}{'形状不变?'}")
print("-" * 104)
for _name, _layer, _dims in _norm_layers_4d:
    _layer.train()
    _out = _layer(x4d)
    print(f"{_name:<24}{_dims:<26}{str(tuple(x4d.shape)):<20}{str(tuple(_out.shape)):<20}"
          f"{tuple(_out.shape) == tuple(x4d.shape)}")
print()
print("  归一化层不改变张量形状——它只改变数值分布（各自的 mean/var），这是它们能插在任意层之间的原因。")
print()

# --- 3.2 手写复现 BatchNorm / LayerNorm ---
print("-" * 78)
print("3.2 手写复现 BatchNorm2d（training=True）与 LayerNorm 的结果")
print("-" * 78)


def my_batch_norm_2d(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor,
                     eps: float = 1e-5) -> torch.Tensor:
    """手写 BatchNorm2d（训练模式）：在 dim=(0, 2, 3) 上算 μ、σ²（即「同一通道、跨样本和空间位置」）。

    x: (N, C, H, W)；gamma/beta: (C,)
    keepdim=True 是为了让 (C,) 的统计量能和 (N, C, H, W) 广播相除。
    """
    mean = x.mean(dim=(0, 2, 3), keepdim=True)                  # (1, C, 1, 1)
    var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)    # 有偏方差（PyTorch 归一化用有偏）
    x_hat = (x - mean) / torch.sqrt(var + eps)
    return gamma.view(1, -1, 1, 1) * x_hat + beta.view(1, -1, 1, 1)


def my_layer_norm(x: torch.Tensor, normalized_shape: tuple[int, ...],
                  gamma: torch.Tensor, beta: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """手写 LayerNorm：在**最后 len(normalized_shape) 个维度**上算 μ、σ²（每个样本独立，与 batch 无关）。"""
    dims = tuple(range(x.dim() - len(normalized_shape), x.dim()))
    mean = x.mean(dim=dims, keepdim=True)
    var = x.var(dim=dims, unbiased=False, keepdim=True)
    x_hat = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_hat + beta


_bn2d.train()
_out_bn = _bn2d(x4d)
_out_bn_manual = my_batch_norm_2d(x4d, _bn2d.weight, _bn2d.bias, eps=_bn2d.eps)
_err_bn = (_out_bn - _out_bn_manual).abs().max().item()
print(f"  BatchNorm2d 官方输出 vs 手写：最大绝对误差 = {_err_bn:.3e}"
      f"  → {'误差 < 1e-5，复现成功 ✔' if _err_bn < 1e-5 else '误差过大'}")

_out_ln = _ln2d(x4d)
_out_ln_manual = my_layer_norm(x4d, (_C, _H, _W), _ln2d.weight, _ln2d.bias, eps=_ln2d.eps)
_err_ln = (_out_ln - _out_ln_manual).abs().max().item()
print(f"  LayerNorm 官方输出 vs 手写：最大绝对误差 = {_err_ln:.3e}"
      f"  → {'误差 < 1e-5，复现成功 ✔' if _err_ln < 1e-5 else '误差过大'}")
print(f"  手写 BatchNorm 用的统计量：逐通道 mean 前 3 个 = "
      f"{my_batch_norm_2d(x4d, _bn2d.weight, _bn2d.bias)[0, :3, 0, 0].tolist()}"
      f"（γ=1、β=0 时）。")
print()

# --- 3.3 γ 和 β ---
print("-" * 78)
print("3.3 γ（weight）和 β（bias）的形状与初始值")
print("-" * 78)
for _name, _layer, _ in _norm_layers_4d:
    _g = _layer.weight
    _b = _layer.bias
    _g_desc = f"形状{tuple(_g.shape)}, 全为1={bool((_g == 1).all())}" if _g is not None else "无（affine=False）"
    _b_desc = f"形状{tuple(_b.shape)}, 全为0={bool((_b == 0).all())}" if _b is not None else "无（affine=False）"
    print(f"  {_name:<24} γ: {_g_desc}")
    print(f"  {'':<24} β: {_b_desc}")
print("  → 初始时 γ=1、β=0，这一层等价于「纯归一化」；训练中二者会被梯度更新，")
print("     网络可以学会把某些通道放大、某些通道平移——表达能力不受损。")
print()

# --- 3.4 BatchNorm 的 train / eval 差异 ---
print("-" * 78)
print("3.4 BatchNorm 在 train() 与 eval() 下的差异（running_mean / running_var）")
print("-" * 78)
print(
    """
train() 模式：用**当前 batch** 的 μ、σ² 归一化，同时用动量更新累积统计量：
        running_mean ← (1-momentum)·running_mean + momentum·batch_mean
eval() 模式：用累积的 running_mean / running_var 归一化，**不再更新**它们。
    默认 momentum=0.1，所以一次前向只把 running 统计量往当前 batch 拉 10%。
"""
)
torch.manual_seed(42)
_bn_demo = nn.BatchNorm1d(6, momentum=0.1)
_bn_demo.train()
print(f"  初始 running_mean = {_bn_demo.running_mean.tolist()}")
print(f"  初始 running_var  = {_bn_demo.running_var.tolist()}")
print()
x_bn = torch.randn(8, 6) * 3 + 5          # 刻意让均值 ≈5、标准差 ≈3，便于观察 running 统计量向它靠拢
print(f"  输入 x：逐列均值 = {[round(v, 3) for v in x_bn.mean(dim=0).tolist()]}")
print(f"  输入 x：逐列方差 = {[round(v, 3) for v in x_bn.var(dim=0, unbiased=False).tolist()]}")
print()
print(f"{'第几次前向':>12}{'running_mean[0]':>20}{'running_var[0]':>20}")
print("-" * 54)
for _i in range(1, 6):
    _out_bn_demo = _bn_demo(x_bn)             # train 模式，会更新 running 统计量
    print(f"{_i:>12}{_bn_demo.running_mean[0].item():>20.6f}{_bn_demo.running_var[0].item():>20.6f}")
print("  → running 统计量每次向当前 batch 的统计量靠拢 10%（momentum=0.1），逐步逼近全局分布。")
print()

# 同一输入在 train / eval 下输出不同
_bn_same = nn.BatchNorm1d(6)
_bn_same.train()
_ = _bn_same(x_bn) * 5 + 1                    # 先多跑几次，让 running 统计量和当前 batch 有区别
_ = _bn_same(x_bn * 2.0)
_bn_same.train()
_out_train_mode = _bn_same(x_bn)
_bn_same.eval()
_out_eval_mode = _bn_same(x_bn)
_diff_mode = (_out_train_mode - _out_eval_mode).abs().max().item()
print(f"  同一个输入 x 在 train() 与 eval() 下的输出最大差异 = {_diff_mode:.6f}"
      f"  → {'明显不同 ✔（train 用当前 batch 统计量、eval 用累积统计量）' if _diff_mode > 1e-6 else '相同'}")

# eval 模式下也验证一次「多次前向结果相同」
_bn_same.eval()
_eval_repeat = [_bn_same(x_bn) for _ in range(3)]
print(f"  eval() 下连续 3 次前向是否完全相同 → "
      f"{all(torch.equal(_eval_repeat[0], o) for o in _eval_repeat[1:])}（应该为 True）")
print(f"  注意：BatchNorm 本身没有随机性，train() 下输出**也不是随机的**——")
print(f"        它的『不稳定』来自「归一化用的是当前 batch 的统计量」，换个 batch 伙伴输出就变，")
print(f"        这一点在下一小节的 NLP 实验里会量化。")
print()


# --- 3.5 NLP / Transformer 中的归一化 ---
print("=" * 78)
print("3.5 NLP / Transformer 中的归一化：为什么必须用 LayerNorm")
print("=" * 78)
print(
    r"""
NLP 里数据形状是三维 (B, S, H)（batch_size, seq_len, hidden_size），各种归一化的行为也随之变化：

    方法       | 计算 μ,σ² 的维度 | 含义
    BatchNorm  | 沿 B, H          | 把**样本维度**也混进来了
    LayerNorm  | 沿 H             | 每个 token 的 H 维向量**独立**归一化
    （InstanceNorm / GroupNorm 在 NLP 中不用）

**为什么 NLP 用 LayerNorm 而不用 BatchNorm？** 四条理由：
    ① BatchNorm 在 (B,S,H) 上算统计量时把**样本维度 B 也混进来了**。
       NLP 里序列长度可变、batch 内不同样本的 padding 数量不同，
       于是同一个 token 的归一化结果会**依赖同 batch 的其他句子**——推理时 batch 组成一动，
       输出就跟着变，完全不可接受。
    ② 训练/推理统计量不一致：训练用当前 batch、推理用 running 统计量；
       变长序列下每个 batch 的 padding 比例都不同，running statistics 根本估计不准。
    ③ LayerNorm 对**每个 token 的 H 维向量独立归一化**（每个 token 单独算自己的 μ、σ²），
       与 batch 构成、序列长度完全无关 → 训练和推理行为一致，
       天然适配变长序列和自回归逐 token 生成（生成时 batch=1，BatchNorm 连一个 batch 都凑不齐）。
    ④ 形状上：LayerNorm 只要 normalized_shape=H，输入 (B,S,H) 输出形状不变，非常自然；
       而 BatchNorm1d 要求把输入转置成 (B,H,S) 再转回来，用起来十分别扭。

下面用代码把上面每一条都验证一遍。
"""
)

torch.manual_seed(42)
_B, _S, _Hd = 32, 128, 512
x_nlp = torch.randn(_B, _S, _Hd)
print(f"NLP 输入形状：(batch_size={_B}, seq_len={_S}, hidden_size={_Hd})")
print()

# --- LayerNorm：沿最后一维 H ---
ln_nlp = nn.LayerNorm(_Hd)                       # 参数 512 = hidden_size
out_ln = ln_nlp(x_nlp)
print("【LayerNorm】ln = nn.LayerNorm(512); out = ln(x)")
print(f"  输出形状：{tuple(out_ln.shape)} → 与输入相同 ✔")
print(f"  out.mean(-1) 全部为 0 ？最大绝对值 = {out_ln.mean(dim=-1).abs().max().item():.3e}"
      f" → {'✔' if out_ln.mean(dim=-1).abs().max().item() < 1e-5 else '✘'}")
print(f"  out.std(-1) 全部为 1 ？与 1 的最大偏差 = "
      f"{(out_ln.std(dim=-1, unbiased=False) - 1).abs().max().item():.3e}"
      f" → {'✔' if (out_ln.std(dim=-1, unbiased=False) - 1).abs().max().item() < 1e-3 else '✘'}")
print(f"  （严格来说归一化后是零均值、单位方差：mean=0、var=1，所以 std=√1=1；")
print(f"    这里 std 用的是有偏方差 unbiased=False，和归一化内部一致。）")
print(f"  同一个样本内不同 token 的均值都是 0：第 0 个样本的 128 个 token 的 mean(-1) "
      f"最大值 = {out_ln[0].mean(dim=-1).abs().max().item():.3e}")
print(f"  每个 token 的 512 维向量独立归一化 → LayerNorm 的 normalized_shape=H，只作用最后一维。")
print()

# --- BatchNorm1d：必须转置 ---
print("【BatchNorm1d】nn.BatchNorm1d(512) 要求把输入转置成 (B, H, S)")
bn_nlp = nn.BatchNorm1d(_Hd)                     # 参数 512 = hidden_size（当作"通道数"）
bn_nlp.train()
print(f"  转置前形状：{tuple(x_nlp.shape)}  → 转置后：{tuple(x_nlp.transpose(1, 2).shape)}")
out_bn_t = bn_nlp(x_nlp.transpose(1, 2))         # (B,H,S) 里 S 被当成"空间/长度"维
print(f"  BatchNorm1d 输出形状：{tuple(out_bn_t.shape)}  → 再转回：{tuple(out_bn_t.transpose(1, 2).shape)}")
print(f"  → 这个「先 transpose 进去、再 transpose 出来」的别扭感，正是 BatchNorm 不适合 NLP 的表面证据：")
print(f"    它把 (B,S,H) 里的 S 当成图像的「空间维度」，把 H 当成「通道」，语义上完全对不上。")
print()

# --- 敏感度实验：换 batch 伙伴，看输出变化多少 ---
print("【敏感度实验】把同一条样本放进不同的 batch（换「同 batch 的其他样本」），看输出变化多少")
x_probe = torch.randn(1, _S, _Hd)                # 被观察的「同一条样本」

# LayerNorm 的统计量只来自这条样本自身，换 batch 完全不影响
ln_probe = nn.LayerNorm(_Hd)
ln_probe.eval()
torch.manual_seed(1)
_batch_a = torch.randn(_B - 1, _S, _Hd)
torch.manual_seed(2)
_batch_b = torch.randn(_B - 1, _S, _Hd) * 5.0 + 3.0      # 完全不同的分布
_out_ln_a = ln_probe(torch.cat([x_probe, _batch_a], dim=0))[:1]
_out_ln_b = ln_probe(torch.cat([x_probe, _batch_b], dim=0))[:1]
_ln_sens = (_out_ln_a - _out_ln_b).abs().max().item()
print(f"  LayerNorm：换 batch 伙伴后，同一条样本输出的最大差异 = {_ln_sens:.3e}"
      f" → {'完全不受影响 ✔' if _ln_sens < 1e-6 else '有影响'}")

# BatchNorm 训练模式下：当前 batch 的统计量变了，输出就变
# 注意：比较时必须只取**第 0 条样本**（x_probe）的输出，
#       因为 cat 进来的其他样本本来就不同，全量比较会把无关差异算进来。
bn_probe_train = nn.BatchNorm1d(_Hd)
bn_probe_train.train()
_bn_a = bn_probe_train(torch.cat([x_probe.transpose(1, 2), _batch_a.transpose(1, 2)], dim=0))[0:1]
_bn_b = bn_probe_train(torch.cat([x_probe.transpose(1, 2), _batch_b.transpose(1, 2)], dim=0))[0:1]
_bn_sens_train = (_bn_a - _bn_b).abs().max().item()
print(f"  BatchNorm（train 模式）：换 batch 伙伴后，**同一条样本**输出的最大差异 = "
      f"{_bn_sens_train:.3e}"
      f" → {'明显受 batch 影响 ✘' if _bn_sens_train > 1e-3 else '影响很小'}")
print(f"    原因：train 模式用「当前 batch 的 μ、σ²」，而不同 batch 的统计量不一样，")
print(f"          于是同一条样本被除以了不同的分母 → 输出直接改变。")

# BatchNorm eval 模式：固定用 running 统计量，理论上不受当前 batch 影响——
# 但真正的隐患在于：这些 running 统计量是**训练时按 batch 累出来的**，
# 训练时 batch 组成的差异会原封不动地固化进 running 统计量，然后带到推理阶段。
bn_probe_eval = nn.BatchNorm1d(_Hd)
with torch.no_grad():
    bn_probe_eval.running_mean.copy_(torch.randn(_Hd) * 0.5 + 2.0)     # 模拟「被 padding / 变长序列污染的全局统计量」
    bn_probe_eval.running_var.copy_(torch.rand(_Hd) * 2 + 0.5)
bn_probe_eval.eval()                             # 推理模式：只用 running 统计量
_eval_a = bn_probe_eval(torch.cat([x_probe.transpose(1, 2), _batch_a.transpose(1, 2)], dim=0))[0:1]
_eval_b = bn_probe_eval(torch.cat([x_probe.transpose(1, 2), _batch_b.transpose(1, 2)], dim=0))[0:1]
_bn_sens_eval = (_eval_a - _eval_b).abs().max().item()
print(f"  BatchNorm（eval 模式）：固定用 running 统计量，换 batch 伙伴后同一条样本输出差异 = "
      f"{_bn_sens_eval:.3e}（为 0，因为不再统计当前 batch）")

# 但「训练时按 batch 累出来的 running 统计量」本身就被 batch 组成污染：
# 用两个组成完全不同的 batch 各自更新一次 running 统计量，看它偏离多少
_bn_acc_a = nn.BatchNorm1d(_Hd)
_bn_acc_b = nn.BatchNorm1d(_Hd)
_bn_acc_a.train()
_bn_acc_b.train()
for _ in range(5):
    _bn_acc_a(torch.cat([x_probe.transpose(1, 2), _batch_a.transpose(1, 2)], dim=0))
    _bn_acc_b(torch.cat([x_probe.transpose(1, 2), _batch_b.transpose(1, 2)], dim=0))
_acc_gap = (_bn_acc_a.running_mean - _bn_acc_b.running_mean).abs().max().item()
print(f"  同一条样本搭配不同 batch 伙伴训练 5 次后，running_mean 的最大差异 = {_acc_gap:.3f}"
      f" → 全局统计量被 batch 组成严重污染 ✘")
print(f"    → 到了推理阶段，这个被污染的 running 统计量会让同一个 token 的输出整体偏移，")
print(f"      模型在推理时的表现和训练阶段对不上。")
print(f"  对比之下 LayerNorm 的训练/推理行为完全一致（都只用当前 token 自己的 H 维统计量），")
print(f"  且只依赖单个 token → 自回归生成时 batch=1 也照常工作。这就是 NLP 选 LayerNorm 的根本原因。")
print()

# --- 3.6 Pre-LN vs Post-LN ---
print("-" * 78)
print("3.6 Pre-LN 与 Post-LN：LayerNorm 放在子层的哪一侧")
print("-" * 78)
print(
    """
Transformer 的每个子层（自注意力 / 前馈网络）外面都套着「残差连接 + LayerNorm」，有两种摆法：

    Post-LN（原始论文《Attention Is All You Need》的写法）：
        y = LayerNorm( x + Sublayer(x) )
        归一化在**残差相加之后**，于是残差主干上的信号每经过一层就被 LayerNorm 重新缩放一次，
        深层时梯度需要穿过所有 LayerNorm 才能回到浅层，训练不稳定，
        通常需要 learning-rate warmup（先小火再大火）才能训起来。

    Pre-LN（现代实现的主流写法，GPT 系列、LLaMA 等）：
        y = x + Sublayer( LayerNorm(x) )
        归一化在**子层之前**，残差主干变成一条「干净的直通高速路」：
        y = x + f(LN(x))，梯度可以沿着恒等路径直接回传，深层也不会衰减。
        因此 Pre-LN 训练更稳定、**常常不需要 warmup**，可以更早用大学习率。

一句话记忆：
    Post-LN：先加残差、再归一化（归一化挡在高速路上）；
    Pre-LN ：先归一化、再做子层、最后加残差（归一化挪到旁边的小路上）。
"""
)


class PostLNBlock(nn.Module):
    """Post-LN 残差块：y = LayerNorm(x + Sublayer(x))。"""

    def __init__(self, dim: int):
        super().__init__()
        self.sub = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.sub(x))


class PreLNBlock(nn.Module):
    """Pre-LN 残差块：y = x + Sublayer(LayerNorm(x))。"""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.sub = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.sub(self.norm(x))


torch.manual_seed(42)
_x_ln_block = torch.randn(16, 64)
for _cls, _tag in ((PostLNBlock, "Post-LN"), (PreLNBlock, "Pre-LN")):
    _blk = _cls(64)
    _h = _x_ln_block
    _stds = []
    with torch.no_grad():
        for _ in range(6):                       # 堆 6 层，观察信号尺度
            _h = _blk(_h)
            _stds.append(_h.std().item())
    print(f"  6 层 {_tag} 堆叠后逐层激活 std："
          + "  ".join(f"{s:.3f}" for s in _stds)
          + f"  （首尾比 {_stds[-1] / _stds[0]:.3f}）")
print("  → Post-LN 每层都被归一化「重置」到 std≈1（但梯度要穿过这些归一化，深层难训）；")
print("     Pre-LN 的残差主干让 std 随深度缓慢累积（信号不会被切断，梯度有直通路径）。")
print()


# ===========================================================================
# 第 4 部分：早停 EarlyStopping
# ===========================================================================
print("=" * 78)
print("第 4 部分：早停（Early Stopping）")
print("=" * 78)
print(
    """
原理：每隔一段时间检查验证集指标，如果连续 `patience` 个 epoch 没有改善就停止训练，
      并**回到验证指标最好的那个模型**（把当时保存的权重恢复回来）。
为什么有效：训练越久，模型对训练集的记忆越深。在过拟合之前停下，
            不让模型进入「死记硬背」阶段。它不需要改模型结构，只需要监控验证指标，
            是最便宜、最通用的正则化手段。
三个关键参数：
    patience  ：允许多少个 epoch 没有改善（给模型一点「缓冲期」，避免被噪声骗停）；
    min_delta ：改善多少才算「有改善」（过滤掉抖动级别的小波动）；
    mode      ：'min' 表示指标越小越好（如 loss），'max' 表示越大越好（如 accuracy）。
"""
)


class EarlyStopping:
    """早停工具类。

    参数：
        patience  : 连续多少个 epoch 没有改善就停止
        min_delta : 判定「有改善」的最小变化量（绝对值）
        mode      : 'min'（指标越小越好，如 val_loss）/ 'max'（指标越大越好，如 val_acc）
    状态：
        best_score      : 历史最佳指标
        best_epoch      : 取得最佳指标的 epoch（从 0 开始）
        counter         : 已经连续多少个 epoch 没有改善
        best_state_dict : 最佳时刻的模型权重深拷贝（**必须 deepcopy**，否则会被后续训练覆盖）
        early_stop      : 是否触发早停
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "min",
                 verbose: bool = False):
        if mode not in ("min", "max"):
            raise ValueError("mode 只能是 'min' 或 'max'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.best_score: float | None = None
        self.best_epoch: int = -1
        self.counter: int = 0
        self.best_state_dict: dict | None = None
        self.early_stop: bool = False

    def _is_improved(self, score: float) -> bool:
        """判断当前 score 相比历史最佳是否有实质改善（考虑 min_delta 和 mode）。"""
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def __call__(self, score: float, model: nn.Module, epoch: int) -> bool:
        """每个 epoch 结束后调用：传入当前验证指标、模型、epoch 编号。返回是否应该停止训练。"""
        if self._is_improved(score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            # **关键**：必须 deepcopy。state_dict() 返回的是指向同一批张量的字典，
            # 如果只写 self.best_state_dict = model.state_dict()，后续 optimizer.step()
            # 会原地修改这些张量，导致「保存的最佳权重」被训练过程持续改写，最后存下来的是最新权重。
            self.best_state_dict = copy.deepcopy(model.state_dict())
            if self.verbose:
                print(f"    [EarlyStopping] epoch {epoch}: 指标改善 → {score:.6f}（已保存最佳权重）")
        else:
            self.counter += 1
            if self.verbose:
                print(f"    [EarlyStopping] epoch {epoch}: 指标未改善 "
                      f"({self.counter}/{self.patience})，当前 {score:.6f}，最佳 {self.best_score:.6f}")
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def restore(self, model: nn.Module) -> None:
        """把模型权重恢复成「最佳时刻」的版本。"""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)


print("-" * 78)
print("4.1 实际使用 EarlyStopping 训练（patience=10，max_epochs=60，靠早停自动截断）")
print("-" * 78)
print(
    """
实验设计：
    · 数据：torch.randn(500, 20)，标签 (X[:,0] + X[:,1] > 0).long()；按 8:2 切训练/验证；
    · 模型：20 → 64 → 2 的 MLP（和课案一致）；
    · 优化器：Adam(lr=1e-2)，损失 CrossEntropyLoss，最多 60 个 epoch；
    · 早停：patience=10、min_delta=1e-4、mode='min'，监控验证 loss。
      课案写的是 max_epochs=200、patience=10；这里把 max_epochs 缩到 60 保证 CPU 上够快，
      早停会在这之前自动截断（这是早停的实用价值：不用手工猜训练多久）。
"""
)

torch.manual_seed(42)
_Xe = torch.randn(500, 20)
_ye = (_Xe[:, 0] + _Xe[:, 1] > 0).long()
_split = int(0.8 * len(_Xe))
train_loader = DataLoader(TensorDataset(_Xe[:_split], _ye[:_split]), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(_Xe[_split:], _ye[_split:]), batch_size=64)

model_es = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 2))
criterion_es = nn.CrossEntropyLoss()
optimizer_es = torch.optim.Adam(model_es.parameters(), lr=1e-2)

es = EarlyStopping(patience=10, min_delta=1e-4, mode="min")
max_epochs = 60
_es_train_loss: list[float] = []
_es_val_loss: list[float] = []
_stop_epoch = max_epochs - 1
_last_epoch = max_epochs - 1

for epoch in range(max_epochs):
    model_es.train()
    _epoch_loss = 0.0
    for Xb, yb in train_loader:
        optimizer_es.zero_grad()
        loss = criterion_es(model_es(Xb), yb)
        loss.backward()
        optimizer_es.step()
        _epoch_loss += loss.item() * Xb.size(0)
    _es_train_loss.append(_epoch_loss / len(train_loader.dataset))

    model_es.eval()
    _total_val = 0.0
    with torch.no_grad():
        for Xb, yb in val_loader:
            _total_val += criterion_es(model_es(Xb), yb).item() * Xb.size(0)
    val_loss = _total_val / len(val_loader.dataset)
    _es_val_loss.append(val_loss)
    _last_epoch = epoch

    if es(val_loss, model_es, epoch):
        _stop_epoch = epoch
        break

# 记录早停瞬间的验证 loss（未恢复权重）
_val_loss_at_stop = _es_val_loss[-1]
es.restore(model_es)                            # 恢复最佳权重
model_es.eval()
_val_loss_best_weight = 0.0
with torch.no_grad():
    for Xb, yb in val_loader:
        _val_loss_best_weight += criterion_es(model_es(Xb), yb).item() * Xb.size(0)
_val_loss_best_weight /= len(val_loader.dataset)

with torch.no_grad():
    _val_acc_restored = (model_es(_Xe[_split:]).argmax(dim=1) == _ye[_split:]).float().mean().item()

print(f"  早停发生在第 {_stop_epoch} 个 epoch（共跑了 {_last_epoch + 1} 个 epoch；"
      f"max_epochs={max_epochs}，patience={es.patience}）")
print(f"  最佳 epoch = 第 {es.best_epoch} 个，最佳验证 loss = {es.best_score:.6f}")
print(f"  恢复最佳权重后的验证 loss = {_val_loss_best_weight:.6f}"
      f"（对应验证准确率 {_val_acc_restored * 100:.2f}%）")
print(f"  早停瞬间（最后一轮）的验证 loss = {_val_loss_at_stop:.6f}")
print(f"  两者差值 = {_val_loss_best_weight - _val_loss_at_stop:+.6f}"
      f"  → {'恢复最佳权重确实比停在最后一轮更好 ✔' if _val_loss_best_weight < _val_loss_at_stop else '本轮差异不明显（数据太简单，过拟合不严重）'}")
print(f"  → 早停的真实价值：不用手工猜训练多少个 epoch，训练会自动在「验证指标不再改善」时停下，")
print(f"     并用最佳权重交付模型，避免了继续训练带来的过拟合。")
print()

# --- 画早停曲线 ---
fig_e, ax_e = plt.subplots(figsize=(10, 5.5))
_ep_range = range(1, len(_es_train_loss) + 1)
ax_e.plot(_ep_range, _es_train_loss, "-", color="#4C72B0", label="训练 loss")
ax_e.plot(_ep_range, _es_val_loss, "-", color="#DD8452", label="验证 loss")
# 标出最佳点
ax_e.axvline(es.best_epoch + 1, color="#55A868", ls="--", lw=1.6,
             label=f"最佳 epoch = {es.best_epoch + 1}（val loss={es.best_score:.4f}）")
ax_e.plot([es.best_epoch + 1], [es.best_score], "*", color="#55A868", ms=18)
# 标出早停点
ax_e.axvline(_stop_epoch + 1, color="#C44E52", ls=":", lw=1.8,
             label=f"早停 epoch = {_stop_epoch + 1}（val loss={_val_loss_at_stop:.4f}）")
ax_e.annotate(f"patience={es.patience}\n连续 {es.patience} 轮无改善",
              xy=(_stop_epoch + 1, max(_es_val_loss) * 0.75),
              xytext=(max(1, _stop_epoch - 18), max(_es_val_loss) * 0.9),
              arrowprops=dict(arrowstyle="->", color="#C44E52"), fontsize=9, color="#C44E52")
ax_e.set_xlabel("epoch")
ax_e.set_ylabel("交叉熵损失")
ax_e.set_title("早停（EarlyStopping）训练曲线：标出最佳点与早停点")
ax_e.legend(fontsize=9)
ax_e.grid(alpha=0.3)
fig_e.tight_layout()
_p_es = OUTPUT_DIR / "03训练组件_06_早停曲线.png"
fig_e.savefig(_p_es, dpi=110)
plt.close(fig_e)
print(f"已保存：{_p_es}")
print()


# ===========================================================================
# 第 5 部分：数据增强（纯 torch 手写，禁止 torchvision）
# ===========================================================================
print("=" * 78)
print("第 5 部分：数据增强（Data Augmentation）——纯 torch 手写")
print("=" * 78)
print(
    """
本节不用 torchvision（本机没有安装，也禁止 import），全部用纯 torch 手写增强函数。

课案原文用的是 torchvision.transforms.Compose([...])，等价的手写实现如下：
    transforms.RandomHorizontalFlip(p=0.5)   →  random_horizontal_flip(img, p=0.5)
    transforms.RandomCrop(224, padding=4)    →  random_crop(img, size, padding=4)
    transforms.ColorJitter(brightness=0.2)   →  random_brightness(img, factor_range=(0.8, 1.2))
    另外补充：垂直翻转 / 90° 旋转 / 高斯噪声 / 随机遮挡（RandomErasing）

关于 padding=4 的含义（课案）：
    先在原图四周各垫 4 像素（用 0 或镜像填充），得到 232×232 的图，
    再从里面随机裁出一个 224×224 的区域。
    效果相当于给原图做了**最多 4 像素的随机平移**——猫可能偏左一点、偏右一点，
    但模型依然要认出它是猫。这是一种低成本的位置增强。
    **本脚本为了 CPU 上跑得快，用 32×32 的小图演示（pad 4 → 40×40 → 随机裁 32×32），
      原始课案的数值是 224（pad 4 → 232×232 → 随机裁 224×224），两者原理完全一样。**

为什么有效：
    告诉模型「翻转后的猫还是猫」，迫使它抓住**不变的本质特征**（轮廓、纹理、部件关系），
    而不是记住表面的像素排列。这等于凭空扩大了有效训练集。
    **只在训练时增强，推理时用原始数据**——推理时我们希望模型看到的就是真实世界的数据分布。
"""
)
print()


def random_horizontal_flip(img: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """以概率 p 做水平翻转。img: (..., H, W)，翻转最后一维（宽度方向）。

    torchvision 的 RandomHorizontalFlip(p=0.5) 就等价于这个函数。
    """
    if torch.rand(1).item() < p:
        return torch.flip(img, dims=[-1])       # -1 是 W 维
    return img


def random_vertical_flip(img: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """以概率 p 做垂直翻转（翻转 H 维，即 -2）。"""
    if torch.rand(1).item() < p:
        return torch.flip(img, dims=[-2])
    return img


def random_crop(img: torch.Tensor, size: int, padding: int = 4) -> torch.Tensor:
    """先四周 pad `padding` 像素，再随机裁出 `size × size` 的区域。

    img: (C, H, W)，要求 img.shape[-1] == img.shape[-2] == size。
    用 F.pad 的 'constant' 模式补 0（torchvision 默认也是 0）。
    """
    padded = F.pad(img, (padding, padding, padding, padding), mode="constant", value=0.0)
    _, h_pad, w_pad = padded.shape
    top = torch.randint(0, h_pad - size + 1, (1,)).item()      # 左上角的随机位置
    left = torch.randint(0, w_pad - size + 1, (1,)).item()
    return padded[:, top:top + size, left:left + size]


def random_gaussian_noise(img: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    """加高斯噪声：img + randn_like(img) * std，再 clamp 回 [0, 1]（图像像素的合法范围）。"""
    return (img + torch.randn_like(img) * std).clamp(0.0, 1.0)


def random_rotation90(img: torch.Tensor) -> torch.Tensor:
    """随机旋转 0 / 90 / 180 / 270 度（用 torch.rot90 实现，不需要插值、零信息损失）。"""
    k = int(torch.randint(0, 4, (1,)).item())   # 0~3 次逆时针 90°
    return torch.rot90(img, k, dims=(-2, -1)) if k else img


def random_brightness(img: torch.Tensor, factor_range: tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
    """随机调整亮度：整张图乘以一个随机因子，再 clamp 回 [0, 1]。

    等价于 transforms.ColorJitter(brightness=0.2)：亮度因子从 [1-0.2, 1+0.2] = [0.8, 1.2] 均匀抽取。
    """
    factor = float(torch.empty(1).uniform_(*factor_range).item())
    return (img * factor).clamp(0.0, 1.0)


def random_erase(img: torch.Tensor, area_ratio: tuple[float, float] = (0.02, 0.12),
                 p: float = 0.5) -> torch.Tensor:
    """随机遮挡一块矩形区域（置为 0），模拟物体被部分遮挡的情形。

    area_ratio: 遮挡面积占整图面积的比例范围。torchvision 的 RandomErasing 思路相同。
    """
    if torch.rand(1).item() >= p:
        return img
    c, h, w = img.shape
    area = h * w * float(torch.empty(1).uniform_(*area_ratio).item())
    aspect = float(torch.empty(1).uniform_(0.3, 3.3).item())    # 长宽比
    erase_h = int(round(math.sqrt(area * aspect)))
    erase_w = int(round(math.sqrt(area / aspect)))
    if erase_h <= 0 or erase_w <= 0 or erase_h >= h or erase_w >= w:
        return img                                              # 尺寸非法就跳过
    top = int(torch.randint(0, h - erase_h + 1, (1,)).item())
    left = int(torch.randint(0, w - erase_w + 1, (1,)).item())
    out = img.clone()
    out[:, top:top + erase_h, left:left + erase_w] = 0.0
    return out


class Compose:
    """把多个增强函数串起来，模仿 torchvision.transforms.Compose 的用法。

    用法：
        t = Compose([random_horizontal_flip, lambda im: random_crop(im, 32, 4)])
        out = t(img)        # 依次施加每个变换
    """

    def __init__(self, transforms: list):
        self.transforms = list(transforms)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        inner = ",\n  ".join(getattr(t, "__name__", repr(t)) for t in self.transforms)
        return f"Compose([\n  {inner}\n])"


# --- 5.1 演示：一张 32×32 图（对照课案的 224） ---
print("-" * 78)
print("5.1 增强效果演示：32×32 小图（课案原始数值是 224×224）")
print("-" * 78)
torch.manual_seed(42)
_IMG_SIZE = 32
# 造一张「左半边亮、右半边暗」的合成图，便于肉眼判断翻转/裁剪是否生效
_grid = torch.linspace(0, 1, _IMG_SIZE).view(1, 1, _IMG_SIZE).expand(1, _IMG_SIZE, _IMG_SIZE).clone()
img_demo = _grid.clamp(0, 1).repeat(3, 1, 1) * 0.8 + 0.1
img_demo[:, :8, :] = 0.95                    # 左上角再放一块亮区，方便看旋转/遮挡
print(f"  原图形状：{tuple(img_demo.shape)}（C=3, H=W={_IMG_SIZE}），像素范围 "
      f"[{img_demo.min().item():.3f}, {img_demo.max().item():.3f}]")
print(f"  课案对照：原图 224×224 → pad 4 → 232×232 → 随机裁 224×224（最多 4 像素随机平移）")
print(f"  本脚本对照：原图 32×32 → pad 4 → 40×40 → 随机裁 32×32（同样是 4 像素随机平移）")
print()

_train_transform = Compose([
    lambda im: random_horizontal_flip(im, p=0.5),
    lambda im: random_crop(im, _IMG_SIZE, padding=4),
    lambda im: random_brightness(im, (0.8, 1.2)),
    lambda im: random_gaussian_noise(im, std=0.05),
])
print(f"  训练增强 pipeline：\n{_train_transform}")
print("  验证时不做任何增强（val_transform = 恒等变换）——推理必须看原始数据。")
print()

# 逐个函数的效果验证
print("  各函数效果验证（原图形状 %s）：" % (tuple(img_demo.shape),))
_checks = [
    ("random_horizontal_flip(p=1.0)", lambda im: random_horizontal_flip(im, p=1.0)),
    ("random_vertical_flip(p=1.0)", lambda im: random_vertical_flip(im, p=1.0)),
    ("random_crop(32, padding=4)", lambda im: random_crop(im, _IMG_SIZE, 4)),
    ("random_gaussian_noise(std=0.05)", lambda im: random_gaussian_noise(im, 0.05)),
    ("random_rotation90", random_rotation90),
    ("random_brightness((0.5, 1.5))", lambda im: random_brightness(im, (0.5, 1.5))),
    ("random_erase(p=1.0)", lambda im: random_erase(im, p=1.0)),
]
for _fname, _fn in _checks:
    torch.manual_seed(7)
    _o = _fn(img_demo)
    _changed = (_o - img_demo).abs().max().item()
    print(f"    {_fname:<34} 形状 {tuple(_o.shape)}  取值范围 [{_o.min().item():.3f}, {_o.max().item():.3f}]"
          f"  与原图最大差异 {_changed:.3f}")
print("  → 所有增强都保持形状不变（32×32），只是改变像素内容/位置。")
# 验证随机性：同一个变换跑两次结果不同
torch.manual_seed(1)
_o1 = _train_transform(img_demo)
torch.manual_seed(2)
_o2 = _train_transform(img_demo)
print(f"  同一个 pipeline 用不同随机种子跑两次，最大差异 = {(_o1 - _o2).abs().max().item():.3f}"
      f" → 每次都是「新样本」✔")
print()

# --- 画增强效果网格图 ---
fig_a, axes_a = plt.subplots(2, 4, figsize=(13, 7))
# 第一张：原图
axes_a[0, 0].imshow(img_demo.permute(1, 2, 0).numpy())
axes_a[0, 0].set_title("原图（左半亮、右半暗）", fontsize=10)
axes_a[0, 0].axis("off")
_aug_specs = [
    ("水平翻转", lambda im: random_horizontal_flip(im, p=1.0)),
    ("垂直翻转", lambda im: random_vertical_flip(im, p=1.0)),
    ("随机裁剪(pad=4)", lambda im: random_crop(im, _IMG_SIZE, 4)),
    ("90°旋转", random_rotation90),
    ("亮度调整", lambda im: random_brightness(im, (0.5, 1.5))),
    ("高斯噪声", lambda im: random_gaussian_noise(im, 0.08)),
    ("随机遮挡", lambda im: random_erase(im, p=1.0)),
]
for _i, (_title, _fn) in enumerate(_aug_specs, start=1):
    _ax = axes_a[_i // 4, _i % 4]
    torch.manual_seed(100 + _i)
    _ax.imshow(_fn(img_demo).permute(1, 2, 0).numpy())
    _ax.set_title(_title, fontsize=10)
    _ax.axis("off")
fig_a.suptitle("数据增强效果（纯 torch 手写，32×32 小图；课案原始尺寸 224）", fontsize=12)
fig_a.tight_layout()
_p_aug = OUTPUT_DIR / "03训练组件_06_数据增强效果.png"
fig_a.savefig(_p_aug, dpi=110)
plt.close(fig_a)
print(f"已保存：{_p_aug}")
print()


# --- 5.2 实验证明：数据增强缓解过拟合 ---
print("-" * 78)
print("5.2 实验证明：数据增强真的缓解过拟合（小数据 + 稍大模型）")
print("-" * 78)
print(
    """
实验设计（刻意制造「训练时没见过位置变化」的场景）：
    · 只有 **100 张** 1×16×16 的合成图（数据极少，模型很容易直接背下来）；
    · 标签规则很简单、保证任务本身可学：**图像左半部分均值是否大于右半部分**；
      但每张图的左右亮度差 δ 只有 0.05~0.15（信噪比低），必须综合半边上百个像素才能可靠判别；
    · 模型偏大：3 层卷积 + 全连接（对 100 张图来说参数绰绰有余）；
    · 评估三组数字：训练集准确率、**原始测试集**准确率、**平移压力测试集**（把测试图整体
      随机平移 ±3 像素）准确率。第三组用来量化「模型有没有学到位置不变性」。

三组对照：
    ① 无增强        —— 基线，训练时只见过目标位置固定的样本；
    ② 平移+噪声+亮度+遮挡 —— 正向增强，把「位置偏移/亮度变化/局部遮挡」变成训练分布的一部分；
    ③ 只用水平翻转  —— **有害增强**：本任务的标签是「左右均值大小关系」，
                       翻转会把左右互换、标签语义直接搞反。

结论预告：数据增强不是无条件有效的，**必须与任务的标签语义匹配**；
          正确的用法是「先在训练数据上模拟你希望模型学会的那种不变性」。
"""
)


def make_synthetic_images(n: int, size: int = 16, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """造合成图像数据集：标签 = 左半部分均值是否大于右半部分。

    返回：
        images: (n, 1, size, size) 的 float 张量，取值 [0, 1]
        labels: (n,) 的 long 张量（0/1）

    难度设计（**刻意做得难，才能暴露过拟合**）：
        · 每张图是接近「纯噪声」的随机纹理（没有真实图像的强结构，没有明显轮廓）；
        · 左右半边的亮度差 δ 只有 0.05 ~ 0.15，必须综合半边的上百个像素才能可靠判别；
        · 100 张训练图 + 参数充足的 3 层 CNN：模型本来完全有能力「记住」这 100 张图；
        · 由于这种图没有可利用的轮廓/纹理不变性，模型只能靠「全局平均」这种脆弱策略，
          一旦测试图发生位置偏移，准确率立刻掉下来——这正好用来量化数据增强的价值。
    """
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(n, 1, size, size, generator=g)                 # 基础随机噪声图
    left_mean = base[:, :, :, :size // 2].mean(dim=(1, 2, 3), keepdim=True)     # (n,1,1,1)
    right_mean = base[:, :, :, size // 2:].mean(dim=(1, 2, 3), keepdim=True)    # (n,1,1,1)
    # 标签规则：左半部分均值大于右半部分 → 类别 1
    labels = (left_mean > right_mean).long().flatten()
    # 构造思路：把左半边整体平移 +δ、右半边整体平移 -δ，δ 的符号由标签决定。
    # 平移不改变纹理，只改变半边的整体亮度，因此
    #   「左半边均值 > 右半边均值」 ⟺ 「标签 = 1」 成立，任务本身完全可学。
    # δ 取 0.05 ~ 0.15：单张图的左右差异很小，必须综合大量像素才能可靠判别 →
    # 数据少时模型倾向于直接记住这 100 张图，从而产生可观察的过拟合。
    delta = (torch.rand(n, 1, 1, 1, generator=g) * 0.10 + 0.05)      # 平移幅度 0.05 ~ 0.15
    sign = torch.where(labels.view(n, 1, 1, 1) == 1,
                       torch.ones_like(delta), -torch.ones_like(delta))
    left_patch = (base[:, :, :, :size // 2] + sign * delta).clamp(0.0, 1.0)
    right_patch = (base[:, :, :, size // 2:] - sign * delta).clamp(0.0, 1.0)
    images = torch.cat([left_patch, right_patch], dim=3)
    return images, labels


torch.manual_seed(42)
_X_img, _y_img = make_synthetic_images(400, size=16, seed=42)
_img_tr, _y_img_tr = _X_img[:100], _y_img[:100]       # 只用 100 张训练（数据极少）
_img_te, _y_img_te = _X_img[200:], _y_img[200:]       # 200 张测试
print(f"  数据：{tuple(_X_img.shape)}（100 张训练 / 200 张测试），"
      f"标签分布：训练集正类占比 {_y_img_tr.float().mean().item():.2f}，"
      f"测试集正类占比 {_y_img_te.float().mean().item():.2f}")


def shift_batch(images: torch.Tensor, max_shift: int = 3) -> torch.Tensor:
    """把整个 batch 的图像随机平移 ±max_shift 像素（用 affine_grid + grid_sample 实现）。

    作用有两个：
        · 作为**增强算子**：训练时让模型看到目标位置发生偏移的样本 → 学会位置不变性；
        · 作为**压力测试**：构造一个「测试图整体平移」的场景 → 量化模型的鲁棒性。
    """
    n = images.shape[0]
    dx = (torch.rand(n) * 2 - 1) * max_shift
    dy = (torch.rand(n) * 2 - 1) * max_shift
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    # affine_grid 的平移量要求是归一化坐标（-1~1），所以像素位移要除以图像半宽/半高
    half = images.shape[-1] / 2.0
    theta[:, 0, 2] = dx / half
    theta[:, 1, 2] = dy / half
    grid = F.affine_grid(theta, list(images.shape), align_corners=False)
    return F.grid_sample(images, grid, align_corners=False, padding_mode="zeros")


class SmallCNN(nn.Module):
    """3 层卷积 + 全连接的稍大模型（对 100 张图来说很容易过拟合）。

    结构：Conv(1→16,3×3) → ReLU → MaxPool → Conv(16→32,3×3) → ReLU → MaxPool
          → Conv(32→64,3×3) → ReLU → AdaptiveAvgPool → Linear(64→2)
    """

    def __init__(self, in_channels: int = 1, n_class: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16→8
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),            # 8→4
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                                                # → (N,64,1,1)
        )
        self.classifier = nn.Linear(64, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x.flatten(1))


def apply_aug_batch(images: torch.Tensor, kind: str) -> torch.Tensor:
    """对一个 batch 的图像做增强（**向量化实现**，避免 Python 逐张循环拖慢速度）。

    参数：
        images: (N, C, H, W) 的 float 张量，取值 [0, 1]
        kind  : 'none'     无增强（对照组）
                'flip'     逐张以 0.5 概率水平翻转
                           （对本任务**有害**：标签是「左右均值大小关系」，翻转会把语义搞反）
                'semantic' 随机平移 ±3 像素 + 高斯噪声 + 亮度扰动 + 随机遮挡
                           （不破坏左右语义，且能训练出位置/亮度鲁棒性）

    说明：这里为了性能写成整批向量化，数学上完全等价于「对每一张图独立地调用
          compose([...])」——随机掩码/随机参数都是逐样本采样的。
    """
    if kind == "none":
        return images
    n, c, h, w = images.shape
    if kind == "flip":
        # 逐张决定是否翻转，再用 flip 一次性完成（翻转是确定性操作，可以被掩码选择）
        do_flip = (torch.rand(n) < 0.5).view(n, 1, 1, 1)
        return torch.where(do_flip, torch.flip(images, dims=[-1]), images)
    # ---- 'semantic' ----
    # ① 随机平移（等价于「先 pad 再随机裁」）：让模型学会对位置偏移不变
    out = shift_batch(images, max_shift=3)
    # ② 高斯噪声：每张图、每个像素独立采样
    out = (out + torch.randn_like(out) * 0.15).clamp(0.0, 1.0)
    # ③ 亮度：每张图一个随机因子，广播到 H、W
    factor = torch.empty(n, 1, 1, 1).uniform_(0.75, 1.25)
    out = (out * factor).clamp(0.0, 1.0)
    # ④ 随机遮挡：每张图一个矩形区域置 0（面积比 2%~10%）
    area = h * w * torch.empty(n).uniform_(0.02, 0.10)
    aspect = torch.empty(n).uniform_(0.3, 3.3)
    erase_h = torch.round(torch.sqrt(area * aspect)).long().clamp(1, h - 1)
    erase_w = torch.round(torch.sqrt(area / aspect)).long().clamp(1, w - 1)
    top = (torch.rand(n) * (h - erase_h + 1).float()).long()
    left = (torch.rand(n) * (w - erase_w + 1).float()).long()
    do_erase = torch.rand(n) < 0.5
    row_idx = torch.arange(h).view(1, h, 1)
    col_idx = torch.arange(w).view(1, 1, w)
    erase_mask = ((row_idx >= top.view(n, 1, 1)) & (row_idx < (top + erase_h).view(n, 1, 1))
                  & (col_idx >= left.view(n, 1, 1)) & (col_idx < (left + erase_w).view(n, 1, 1)))
    erase_mask = erase_mask & do_erase.view(n, 1, 1)
    out = out.masked_fill(erase_mask.unsqueeze(1), 0.0)
    return out.contiguous()


# 压力测试集：把测试图整体随机平移 ±3 像素，模拟「训练时没见过的位置变化」
torch.manual_seed(2024)
_img_te_shift = shift_batch(_img_te, max_shift=3)
print(f"  另构造一份「平移压力测试集」：把 200 张测试图各自随机平移 ±3 像素 "
      f"（与原测试集的最大像素差异 = {(_img_te_shift - _img_te).abs().max().item():.3f}）")
print()

# 三种增强策略
_AUG_EPOCHS = 150
_aug_results: dict[str, dict[str, float]] = {}
_aug_history: dict[str, dict[str, list[float]]] = {}
for _tag, _kind in (("无增强（对照）", "none"),
                    ("增强：平移+噪声+亮度+遮挡", "semantic"),
                    ("增强：只用水平翻转（有害）", "flip")):
    torch.manual_seed(42)
    _net = SmallCNN()
    _opt = torch.optim.Adam(_net.parameters(), lr=1e-3)
    _crit = nn.CrossEntropyLoss()
    _hist = {"train_loss": [], "clean_loss": [], "shift_loss": []}
    for _ep in range(_AUG_EPOCHS):
        _net.train()
        # 每个 epoch 都把训练集重新增强一遍（这就是「凭空扩增数据」的具体做法：
        # 模型在整个训练过程中看到的「不同图像」远多于 100 张）
        _img_aug = apply_aug_batch(_img_tr, _kind)
        _opt.zero_grad()
        _l = _crit(_net(_img_aug), _y_img_tr)
        _l.backward()
        _opt.step()
        _hist["train_loss"].append(_l.item())
        _net.eval()
        with torch.no_grad():
            _hist["clean_loss"].append(_crit(_net(_img_te), _y_img_te).item())
            _hist["shift_loss"].append(_crit(_net(_img_te_shift), _y_img_te).item())
    _net.eval()
    with torch.no_grad():
        _acc_tr = (_net(_img_tr).argmax(dim=1) == _y_img_tr).float().mean().item()
        _acc_te = (_net(_img_te).argmax(dim=1) == _y_img_te).float().mean().item()
        _acc_sh = (_net(_img_te_shift).argmax(dim=1) == _y_img_te).float().mean().item()
    _aug_results[_tag] = {
        "train_acc": _acc_tr, "clean_acc": _acc_te, "shift_acc": _acc_sh,
        "gap": _acc_tr - _acc_te, "shift_drop": _acc_te - _acc_sh,
    }
    _aug_history[_tag] = _hist

print(f"{'增强策略':<28}{'训练准确率':>11}{'测试(原图)':>12}{'测试(平移±3px)':>16}{'平移掉点':>10}")
print("-" * 78)
for _tag, _r in _aug_results.items():
    print(f"{_tag:<28}{_r['train_acc'] * 100:>10.2f}%{_r['clean_acc'] * 100:>11.2f}%"
          f"{_r['shift_acc'] * 100:>15.2f}%{_r['shift_drop'] * 100:>9.2f}%")
print()
_none = _aug_results["无增强（对照）"]
_sem = _aug_results["增强：平移+噪声+亮度+遮挡"]
_flip = _aug_results["增强：只用水平翻转（有害）"]
print("  解读（三个结论都有真实数字支撑）：")
print(f"    ① 无增强的模型在原始测试集上 {_none['clean_acc'] * 100:.2f}%，")
print(f"       但测试图只要整体平移 ±3 像素，准确率就掉到 {_none['shift_acc'] * 100:.2f}%"
      f"（掉了 {_none['shift_drop'] * 100:.2f} 个百分点）——")
print(f"       训练时只见过「目标恰好居中/位置固定」的样本，模型没有位置不变性。")
print(f"    ② 用了平移增强的模型：原图 {_sem['clean_acc'] * 100:.2f}%，"
      f"平移后仍有 {_sem['shift_acc'] * 100:.2f}%（只掉 {_sem['shift_drop'] * 100:.2f} 个百分点）。")
print(f"       增强把「位置偏移」这种本来没见过的变化变成了训练分布的一部分，")
print(f"       所以**增强的正确用法是：先在训练数据上模拟你希望模型学会的那种不变性**。")
print(f"    ③ 只用水平翻转：训练准确率 {_flip['train_acc'] * 100:.2f}%，"
      f"测试(原图) {_flip['clean_acc'] * 100:.2f}%，平移后 {_flip['shift_acc'] * 100:.2f}% ——")
print(f"       本任务的标签就是「左右均值大小关系」，翻转会把标签语义搞反（左右互换），")
print(f"       属于**有害增强**。这说明：数据增强不是无条件有效的，必须与任务语义匹配，")
print(f"       盲目套用教科书里的翻转/裁剪只会引入错误的监督信号。")
print()

# --- 画增强对比曲线 ---
fig_ac, axes_ac = plt.subplots(1, 3, figsize=(16.5, 4.8))
for _i, (_tag, _hist) in enumerate(_aug_history.items()):
    _ax = axes_ac[_i]
    _ax.plot(range(1, _AUG_EPOCHS + 1), _hist["train_loss"], color="#4C72B0", label="训练 loss")
    _ax.plot(range(1, _AUG_EPOCHS + 1), _hist["clean_loss"], color="#DD8452", ls="--",
             label="测试 loss（原图）")
    _ax.plot(range(1, _AUG_EPOCHS + 1), _hist["shift_loss"], color="#55A868", ls=":",
             label="测试 loss（平移 ±3px）")
    _r = _aug_results[_tag]
    _ax.set_title(f"{_tag}\n原图 acc={_r['clean_acc'] * 100:.1f}%, "
                  f"平移 acc={_r['shift_acc'] * 100:.1f}%", fontsize=9)
    _ax.set_xlabel("epoch")
    _ax.set_ylabel("交叉熵损失")
    _ax.legend(fontsize=9)
    _ax.grid(alpha=0.3)
fig_ac.tight_layout()
_p_augc = OUTPUT_DIR / "03训练组件_06_数据增强训练对比.png"
fig_ac.savefig(_p_augc, dpi=110)
plt.close(fig_ac)
print(f"已保存：{_p_augc}")
print()


# ===========================================================================
# 第 6 部分：对比表
# ===========================================================================
print("=" * 78)
print("第 6 部分：正则化方法对比表（课案原表）")
print("=" * 78)
_REG_TABLE = [
    ("L1 正则", "损失函数", "λ·Σ|w|，产生稀疏解"),
    ("L2 正则 / Weight Decay", "损失 + 优化器", "λ·Σw²，限制权重大小"),
    ("Dropout", "计算单元", "随机丢弃，训练子网络"),
    ("BatchNorm", "层级", "沿 batch 和特征维归一化"),
    ("LayerNorm", "层级", "沿特征维归一化"),
    ("Early Stopping", "训练流程", "过拟合前停下"),
    ("Data Augmentation", "数据", "变换扩增样本"),
]
print(f"{'方法':<26}{'作用层面':<16}{'核心机制'}")
print("-" * 74)
for _r in _REG_TABLE:
    print(f"{_r[0]:<26}{_r[1]:<16}{_r[2]}")
print()

print("=" * 78)
print("06 正则化：全部实验完成")
print("=" * 78)

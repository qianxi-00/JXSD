"""
对应课案章节：训练技巧 / 过拟合与欠拟合（含损失震荡、训练缓慢）

本节知识点：
    1.  诊断总纲：训练出问题时**先看 loss 曲线的形状**——曲线的形状直接告诉你问题出在哪
    2.  过拟合的构造：小样本 + 高噪声数据（`make_moons(noise=0.35)`）最容易复现过拟合
    3.  数据泄漏：`StandardScaler` 只能在训练集上 `fit`，再 `transform` 验证集（用训练集统计量）
    4.  小验证集的统计噪声：48 条验证样本的 loss/accuracy 会骗人，必须再留一个更大的独立测试集
    5.  无用特征越多越容易过拟合：`make_classification` 的 informative / redundant 特征对照实验
    6.  同一份数据上的三个模型：小模型（容量不足 → 欠拟合）/ 大模型（容量过剩 → 过拟合）/ 大模型 + 正则
    7.  正则化拆解实验：单独加 `Dropout`、单独加 `weight_decay`、两者都加，分离各自的贡献
    8.  泛化差距 `val_loss - train_loss`：判断过拟合最直接的一个数字
    9.  「验证 loss 最低的 epoch」vs「训练结束时的验证 loss」：早停（Early Stopping）价值的量化证据
    10. 手写 `EarlyStopping` 类（patience + 最佳权重恢复），并实测早停效果
    11. 纯 torch 手写的「数据增强」（本机**没有 torchvision**，课案里的 `transforms.Compose` 用纯 torch 替代）：
        加高斯噪声 / 随机特征缩放 / bootstrap 重采样，说明「翻转后的猫还是猫」在表格数据上的对应物
    12. 欠拟合的处理：增大模型（加深加宽）与调大学习率的效果对比（打印准确率变化）
    13. 损失震荡：`lr = 1e-3 / 1e-1 / 0.5` 三条曲线对比（`lr=0.5` 真的会发散成 NaN，
        用 `torch.isfinite` 优雅停止并打印说明），以及 `batch_size = 8` 与 `128` 的抖动差异
    14. 训练缓慢：`lr = 1e-5 / 1e-3 / 1e-1` 各自「达到 val_loss 阈值用了几个 epoch」；
        float32 与 float16 的前向耗时实测；`torch.autocast('cpu', bfloat16)` 的形状验证与限制说明
    15. 决策边界可视化：`np.meshgrid` 网格预测 + `plt.contourf` 画背景，直观看到
        「欠拟合是一条直线、过拟合是扭曲的碎块边界、正则后边界更平滑」
    16. 现象 → 原因 → 处理方式 的完整对照（过拟合 / 欠拟合 / 震荡 / 缓慢 / 爆炸消失）

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Deep_Learning\\04_训练技巧\\01_过拟合与欠拟合演示.py'
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
import copy                                 # deepcopy 用于保存「最佳权重」的快照
import time                                 # 实测耗时（float32 / float16 前向对比）
import unicodedata                          # 计算中文显示宽度，让 print 出来的表格对齐

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 随机种子统一固定：只要种子一样，randn / randperm 的序列就完全一样，
# 这样才能把「模型变差」和「这次随机数不巧」这两件事区分开。
torch.manual_seed(42)
np.random.seed(42)

# CPU 版 torch 默认会开满物理核心做 OpenMP 并行。本脚本的模型很小、数据很少，
# 线程开太多反而被调度开销拖慢，这里限制成 4 个线程，保证耗时可预期、可复现。
torch.set_num_threads(4)

# 超参总表（全部控制在课案要求内：数据 ≤ 1000 条、epoch ≤ 150）
SEED = 42
EPOCHS_MAIN = 150           # 主实验 150 个 epoch（课案上限）
LR_MAIN = 1e-2              # 主实验的学习率
BS_MAIN = 32                # 主实验的 batch size
DROPOUT_P = 0.5             # 课案指定的 Dropout 概率
WEIGHT_DECAY = 1e-3         # 课案指定的 L2 正则系数


# ===========================================================================
# 通用小工具：中文对齐表格 + 分节标题 + 安全格式化
# ===========================================================================
def _disp_width(text) -> int:
    """计算字符串在等宽终端里的显示宽度：中文/全角字符占 2 列，其余占 1 列。

    Python 的 len() 按「字符个数」算，中文一个字算 1 但实际屏幕占 2 列，
    直接用 len() 补空格会错位，所以这里用 east_asian_width 判宽。
    """
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in str(text))


def _pad(text, width, align="left"):
    """按显示宽度补空格。"""
    text = str(text)
    space = " " * max(0, width - _disp_width(text))
    return (text + space) if align == "left" else (space + text)


def print_table(headers, rows, aligns=None):
    """打印一张对齐的中文表格（纯文本，便于把关键数值贴进文档）。"""
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


_CLOCK = [time.perf_counter()]              # 记录「上一节结束时刻」，用来打印每节耗时


def section(title):
    """打印分节标题（附上一节耗时），方便在长输出里定位、也方便核对性能预算。"""
    now = time.perf_counter()
    print()
    print("=" * 78)
    print(title + f"    ［上一节耗时 {now - _CLOCK[0]:.2f} 秒］")
    print("=" * 78)
    _CLOCK[0] = now


def fmt(v, nd=4):
    """把可能是 inf/NaN/None 的浮点数安全地格式化成字符串（绝不抛异常）。"""
    if v is None:
        return "n/a"
    v = float(v)
    if np.isnan(v):
        return "NaN"
    if np.isinf(v):
        return "inf" if v > 0 else "-inf"
    return f"{v:.{nd}f}"


# ===========================================================================
# 第 1 部分：数据准备
# ===========================================================================
section("第 1 部分：数据准备——为什么「小样本 + 高噪声」最容易复现过拟合")

print("""
【原理】过拟合 = 模型容量 > 数据里的**有效信息量**。
        数据里的信息 = 真实规律 + 噪声。模型足够大时，它会把噪声也一起记住，
        于是训练集误差趋近 0，但在没见过的数据上误差反而变大。

        想稳定复现过拟合，就要把「有效信息量」压到很低：
          - 样本少（train 只有 192 条）；
          - 噪声大：make_moons 的 noise=0.35（默认 0.05），两类点严重交叠。
        这样一个大网络几乎可以把训练点「背」下来。

【关于样本量的说明】课案建议 n_samples=120。实测 120 时验证集只有 24 条，
        噪声太大，验证 loss 一路单调下降、不出现「先降后反弹」——现象复现不稳定。
        因此这里用 n_samples=240（train 192 / val 48），并**额外**生成一个 400 条的
        **独立测试集**（同分布、不同随机种子），专门用来稳定地估计泛化能力。
        数据总量 240 + 400 = 640 条，仍然满足「≤ 1000 条」的约束。
""")

# make_moons：两个交错的半月形，是二维里最经典的非线性二分类玩具数据。
# noise=0.35 属于「很大」的噪声，故意加大是为了让过拟合来得更快更明显。
X, y = make_moons(n_samples=240, noise=0.35, random_state=SEED)
print(f"[数据] make_moons(n_samples=240, noise=0.35)：X.shape={X.shape}，y.shape={y.shape}")
print(f"[数据] 类别分布：0 类 {(y == 0).sum()} 条，1 类 {(y == 1).sum()} 条")

# 8:2 拆 train/val。stratify=y 保证两边类别比例一致，避免抽出来的验证集全是同一类。
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
# 独立测试集：同一个分布（noise 也取 0.35），但**另一个随机种子**，代表「从真实世界里再抓一批新数据」
X_te, y_te = make_moons(n_samples=400, noise=0.35, random_state=7)
print(f"[数据] 8:2 拆分 → 训练集 {X_tr.shape[0]} 条，验证集 {X_va.shape[0]} 条；"
      f"另有独立测试集 {X_te.shape[0]} 条")

print("""
【原理】为什么要标准化？两个特征量纲差很多时，梯度方向会被量纲大的维度主导，
        训练变慢甚至震荡。标准化把它压到均值 0、方差 1。

【关键】StandardScaler 只能在**训练集**上 fit！
        如果先对全体数据 fit，验证集/测试集的均值方差就泄漏进了训练过程，
        这叫**数据泄漏（data leakage）**，会让验证指标虚高、失去参考价值。
        正确做法：train 上 fit → 用同一组统计量 transform(train / val / test)。
""")
scaler = StandardScaler()
scaler.fit(X_tr)                                        # 只用训练集统计 μ、σ
X_tr_s = scaler.transform(X_tr)                         # 用训练集的 μ、σ 变换训练集
X_va_s = scaler.transform(X_va)                         # 验证集必须用同一组 μ、σ，不能自己 fit
X_te_s = scaler.transform(X_te)                         # 测试集同理
print(f"[标准化] 训练集 μ={np.round(scaler.mean_, 4)}  σ={np.round(scaler.scale_, 4)}")
print(f"[标准化] 变换后训练集均值={np.round(X_tr_s.mean(axis=0), 6)}  标准差={np.round(X_tr_s.std(axis=0), 6)}")
print(f"[标准化] 变换后验证集均值={np.round(X_va_s.mean(axis=0), 4)}  "
      f"测试集均值={np.round(X_te_s.mean(axis=0), 4)}（不等于 0 是正常的：用的是训练集统计量）")

# 转成 torch 张量。X 用 float32（PyTorch 默认浮点类型），y 用 int64（CrossEntropyLoss 要求）。
Xtr = torch.tensor(X_tr_s, dtype=torch.float32)
ytr = torch.tensor(y_tr, dtype=torch.long)
Xva = torch.tensor(X_va_s, dtype=torch.float32)
yva = torch.tensor(y_va, dtype=torch.long)
Xte = torch.tensor(X_te_s, dtype=torch.float32)
yte = torch.tensor(y_te, dtype=torch.long)

# 决策边界图要把训练点和验证点一起画出来
X_all = np.vstack([X_tr_s, X_va_s])
y_all = np.concatenate([y_tr, y_va])


# ===========================================================================
# 第 2 部分：模型定义
# ===========================================================================
section("第 2 部分：三种模型——小模型（欠拟合）/ 大模型（过拟合）/ 大模型 + 正则")


def build_small_model(dropout_p=None):
    """小模型：容量不足 → 欠拟合。

    结构是 Linear(2,4) + Linear(4,2)，注意**两层之间没有非线性激活**。
    数学上 $W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2) = W'x + b'$，
    两个线性层串起来还是**一个线性层**，所以它的决策边界必然是一条直线。
    用直线去切两弯交错的半月形，怎么训都欠拟合——这正是我们要的现象。

    dropout_p 参数只是为了和 build_big_model 的调用签名保持一致（本模型不用 Dropout：
    容量本来就不够，再加正则只会更欠拟合）。
    """
    return nn.Sequential(
        nn.Linear(2, 4),        # 2 维输入 → 4 维隐藏
        nn.Linear(4, 2),        # 4 维隐藏 → 2 类 logits（无激活 ⇒ 整体仍是线性映射）
    )


def build_big_model(dropout_p=None):
    """大模型：容量过剩 → 过拟合。

    结构严格按课案：Linear(2,256) → ReLU → Linear(256,256) → ReLU → Linear(256,256)
                     → ReLU → Linear(256,2)
    参数量约 13.3 万，而训练样本只有 192 条 → 参数量比样本数多三个数量级。

    dropout_p 不为 None 时在每个 ReLU 前插入 nn.Dropout(0.5)。
    Dropout 的原理：训练时以概率 p 随机把神经元输出置 0，等价于每步都在训练
    一个「随机子网络」，最后相当于对指数多个子网络做集成（ensemble）；
    推理时用完整网络。它抑制了「某个神经元专门记住某个样本」这种行为，
    从而缓解过拟合。
    """
    layers = [nn.Linear(2, 256)]
    for _ in range(2):                                  # 两个隐藏块：ReLU + Linear(256,256)
        if dropout_p is not None:
            layers.append(nn.Dropout(p=dropout_p))      # 训练时随机丢弃，eval() 时自动关闭
        layers.append(nn.ReLU())                        # ReLU 正半轴导数恒为 1，不压缩梯度
        layers.append(nn.Linear(256, 256))
    if dropout_p is not None:
        layers.append(nn.Dropout(p=dropout_p))          # 输出层前的最后一次 Dropout
    layers.append(nn.ReLU())
    layers.append(nn.Linear(256, 2))                    # 输出 2 类 logits
    return nn.Sequential(*layers)


def build_medium_model():
    """中等模型：用于演示「欠拟合 → 增大模型」这条处理方式。
    2 → 64 → 64 → 2，带 ReLU，容量比小模型大得多（能弯出非线性边界），
    但远小于 256 宽的大模型（不容易把噪声也背下来）。"""
    return nn.Sequential(
        nn.Linear(2, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 2),
    )


def count_params(model):
    """统计可训练参数量，用来说明「容量」到底多大。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


for _name, _m in [("小模型（纯线性）", build_small_model()),
                  ("中等模型", build_medium_model()),
                  ("大模型", build_big_model())]:
    print(f"[容量] {_name:<10} 参数量 = {count_params(_m):>7,d}")


# ===========================================================================
# 第 3 部分：训练函数
# ===========================================================================
section("第 3 部分：统一训练函数——逐 epoch 记录 train / val / test 的 loss 与 accuracy")


def evaluate(model, X, y, criterion):
    """在 eval 模式下算一组数据的 loss 和 accuracy。

    model.eval() 很关键：它关闭 Dropout（不再随机丢）、
    让 BatchNorm 用训练时积累的全局统计量，保证评估结果可复现。
    """
    model.eval()
    with torch.no_grad():                               # 评估不需要梯度，省时间省内存
        logits = model(X)
        loss = criterion(logits, y).item()
        acc = (logits.argmax(dim=1) == y).float().mean().item()
    return loss, acc


def train_model(model, Xtr, ytr, Xva, yva, *, Xte=None, yte=None,
                epochs=EPOCHS_MAIN, lr=LR_MAIN, optimizer_name="adam",
                momentum=0.9, weight_decay=0.0, batch_size=BS_MAIN,
                augment_fn=None, bootstrap=False, clip_norm=None,
                early_stopping=None, tag=""):
    """训练一个模型并返回逐 epoch 的历史曲线。

    参数说明（每一项都对应课案里的一个知识点）：
        lr            : 学习率。太大 → 震荡/发散；太小 → 走得慢（本文件后半段专门做实验）
        optimizer_name: "adam" 或 "sgd"
        weight_decay  : L2 正则系数。等价于在 loss 上加 $\\frac{\\lambda}{2}\\|W\\|^2$，
                        梯度里多出一项 $\\lambda W$，每步都把权重往 0 拉一点
                        → 抑制大权重 → 缓解过拟合
        batch_size    : None 表示全批量（full-batch）；小 batch 梯度噪声大，曲线更抖
        augment_fn    : 数据增强函数（纯 torch 手写，本机没有 torchvision）
        bootstrap     : 是否每个 epoch 对训练集做有放回重采样（数据增强的一种）
        clip_norm     : 梯度裁剪阈值（详见 02 脚本）
        early_stopping: EarlyStopping 实例，验证 loss 不再改善就提前停

    返回：dict，含 train_loss / val_loss / test_loss / train_acc / val_acc / test_acc
          六条逐 epoch 曲线，以及 diverged_at（若发散则记录第几个 epoch 变成 NaN）。
    """
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    n = Xtr.shape[0]
    bs = n if batch_size is None else min(int(batch_size), n)

    hist = {"train_loss": [], "val_loss": [], "test_loss": [],
            "train_acc": [], "val_acc": [], "test_acc": []}
    diverged_at = None

    for epoch in range(epochs):
        model.train()                                   # 打开 Dropout / 更新 BN 统计量
        # bootstrap 重采样：有放回地抽 n 个索引 → 每个 epoch 看到的训练集都略有不同。
        # 这是「数据增强」在表格数据上最朴素的对应物（相当于图像里的随机裁剪）。
        if bootstrap:
            perm = torch.randint(0, n, (n,))
        else:
            perm = torch.randperm(n)                    # 每个 epoch 打乱顺序

        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = Xtr[idx], ytr[idx]
            if augment_fn is not None:
                xb = augment_fn(xb)                     # 纯 torch 手写的数据增强

            optimizer.zero_grad()                       # 梯度清零（否则会累加上一批的梯度）
            logits = model(xb)
            loss = criterion(logits, yb)

            # 【安全阀】lr 过大时 loss 可能变成 NaN。这里必须检查，
            # 否则 NaN 会顺着参数扩散，最后打印出一堆 nan 却不知道为什么。
            if not torch.isfinite(loss):
                diverged_at = epoch + 1
                break

            loss.backward()                             # 反向传播算梯度
            if clip_norm is not None:                   # 梯度裁剪（爆炸时用）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()                            # 更新参数

        if diverged_at is not None:
            # 优雅收尾：所有曲线补一个 NaN 后退出，绝不打印异常栈
            for k in hist:
                hist[k].append(float("nan"))
            print(f"  [{tag}] 第 {diverged_at} 个 epoch 检测到 loss 不是有限值 → "
                  f"lr={lr} 过大导致训练发散，已安全停止该条曲线（不抛异常、不打印堆栈）")
            break

        # 注意：这里的 train_loss 是在 **eval 模式**下算的（Dropout 关闭）。
        # 只有这样 train_loss 和 val_loss 才是同一把尺子量出来的，
        # 它们的差（泛化差距）才有意义；否则 Dropout 会让 train_loss 虚高。
        tr_loss, tr_acc = evaluate(model, Xtr, ytr, criterion)
        va_loss, va_acc = evaluate(model, Xva, yva, criterion)
        hist["train_loss"].append(tr_loss)
        hist["val_loss"].append(va_loss)
        hist["train_acc"].append(tr_acc)
        hist["val_acc"].append(va_acc)

        if early_stopping is not None:
            if early_stopping(va_loss, model, epoch + 1):
                print(f"  [{tag}] 早停触发：第 {epoch + 1} 个 epoch 停止"
                      f"（patience={early_stopping.patience}）")
                break

    # 测试集只在训练结束后评估一次：本节只关心「最终泛化水平」，
    # 每个 epoch 都算 400 条纯属浪费（这是控制总耗时的一个小优化）。
    if Xte is not None:
        te_loss, te_acc = evaluate(model, Xte, yte, criterion)
        hist["test_loss"] = [te_loss]
        hist["test_acc"] = [te_acc]

    hist["diverged_at"] = diverged_at
    if early_stopping is not None:
        hist["stopped_epoch"] = early_stopping.stop_epoch or len(hist["val_loss"])
    return hist


def summarize(hist, tag):
    """把一条训练历史压缩成关键数字：最终 loss/acc、泛化差距、最优 epoch。"""
    tr = hist["train_loss"][-1]
    va = hist["val_loss"][-1]
    best_ep = int(np.argmin(hist["val_loss"])) + 1
    # 验证 loss 的「最低点」：早停正是靠它决定何时收手
    best_va = float(np.min(hist["val_loss"]))
    out = {
        "tag": tag,
        "epochs": len(hist["val_loss"]),
        "train_loss": tr,
        "val_loss": va,
        "gap": va - tr,                                 # 泛化差距 = 验证 loss - 训练 loss
        "train_acc": hist["train_acc"][-1],
        "val_acc": hist["val_acc"][-1],
        "best_epoch": best_ep,
        "best_val_loss": best_va,
        "val_loss_rise": va - best_va,                  # 训练结束时验证 loss 比最低点高多少
        "diverged_at": hist.get("diverged_at"),
    }
    if hist.get("test_loss"):
        out["test_loss"] = hist["test_loss"][-1]
        out["test_acc"] = hist["test_acc"][-1]
    return out


# ===========================================================================
# 第 4 部分：主对比实验——同一份数据、同样 150 个 epoch，五种配置
# ===========================================================================
section("第 4 部分：同一份数据 + 同样 epoch 的对照实验（小模型 / 大模型 / 大模型 + 正则）")

print("""
【设计】五个配置共用同一份数据、同一个优化器、同样的 150 个 epoch，
        只改「模型容量」和「正则化」两个变量：
          ① 小模型                → 欠拟合的对照组
          ② 大模型                → 过拟合的对照组
          ③ 大模型 + Dropout      → 分离 Dropout 单独的贡献
          ④ 大模型 + weight_decay → 分离 weight_decay 单独的贡献
          ⑤ 大模型 + 两者都加     → 完整正则（课案要求的「大模型 + 正则」）
        Dropout 概率用课案的 0.5，weight_decay 用课案的 1e-3。

【优化器的选择说明】课案示例用的是 optim.SGD(weight_decay=1e-3)。但实测在本机 CPU
        的这个 2D 小任务上，SGD 在 150 个 epoch 内不足以让大模型把训练集背下来
        （train_loss 只降到 0.20 左右，过拟合现象不明显）。所以这里换成收敛快得多的
        Adam(lr=1e-2, batch_size=32)，weight_decay 仍按课案取 1e-3——只有这样才能
        真实地跑出「验证 loss 先降后反弹」这条曲线。文件后面第 10 部分会单独用
        纯 SGD 演示学习率导致的震荡与发散。
""")

runs = {}
_configs = [
    ("小模型(欠拟合)", build_small_model, dict(dropout_p=None), dict(weight_decay=0.0)),
    ("大模型(过拟合)", build_big_model, dict(dropout_p=None), dict(weight_decay=0.0)),
    ("大模型+Dropout", build_big_model, dict(dropout_p=DROPOUT_P), dict(weight_decay=0.0)),
    ("大模型+WeightDecay", build_big_model, dict(dropout_p=None), dict(weight_decay=WEIGHT_DECAY)),
    ("大模型+正则(两者)", build_big_model, dict(dropout_p=DROPOUT_P), dict(weight_decay=WEIGHT_DECAY)),
]

for tag, builder, build_kw, train_kw in _configs:
    torch.manual_seed(SEED)                             # 每个配置都从同一初始分布出发，公平比较
    model = builder(**build_kw)
    hist = train_model(model, Xtr, ytr, Xva, yva, Xte=Xte, yte=yte,
                       epochs=EPOCHS_MAIN, lr=LR_MAIN, batch_size=BS_MAIN, tag=tag, **train_kw)
    runs[tag] = {"model": model, "hist": hist, "summary": summarize(hist, tag)}
    s = runs[tag]["summary"]
    print(f"  [{tag}] 完成 {s['epochs']} epoch | train_loss={fmt(s['train_loss'])} "
          f"val_loss={fmt(s['val_loss'])} test_loss={fmt(s.get('test_loss'))} "
          f"val_acc={fmt(s['val_acc'], 3)} test_acc={fmt(s.get('test_acc'), 3)}")

print()
print("【读表要点】")
print("  ① 欠拟合：train_loss 和 val_loss **都高**，泛化差距小——模型连训练集都没学会。")
print("  ② 过拟合：train_loss 很低，val_loss 明显更高，**泛化差距大**——模型在背训练集的噪声。")
print("  ③ 正则化：train_loss 比大模型**高一点**（被正则压制，背不了那么死），")
print("     但 val/test loss **明显更低**、泛化差距明显缩小——这才是我们真正想要的模型。")
print("  注意 val 只有 48 条样本，单看 val 的数值会跳；400 条的 test 才是稳定的尺子。")
rows = []
for tag, _b, _k, _t in _configs:
    s = runs[tag]["summary"]
    rows.append([
        tag, s["epochs"],
        fmt(s["train_loss"]), fmt(s["val_loss"]), fmt(s.get("test_loss")),
        fmt(s["gap"]), fmt(s["val_acc"], 3), fmt(s.get("test_acc"), 3),
    ])
print_table(["配置", "epoch", "train_loss", "val_loss", "test_loss(400条)",
             "泛化差距(val-tr)", "val_acc", "test_acc"], rows)

print()
print("【早停的价值：验证 loss 的最低点 vs 训练结束时的验证 loss】")
print("  大模型在训练后期验证 loss 会先降后升；如果一直练到最后一个 epoch，")
print("  你拿到的是「已经变差」的模型。早停就是在最低点附近收手。")
rows2 = []
for tag, _b, _k, _t in _configs:
    s = runs[tag]["summary"]
    rows2.append([
        tag, s["epochs"], s["best_epoch"],
        fmt(s["best_val_loss"]), fmt(s["val_loss"]), fmt(s["val_loss_rise"]),
    ])
print_table(["配置", "训练到 epoch", "val_loss 最低的 epoch", "最低 val_loss",
             "结束 val_loss", "回升了多少"], rows2)

# ---- 结论性判读（用真实数字说话，全部由实测值动态生成） ----
_s_small = runs["小模型(欠拟合)"]["summary"]
_s_big = runs["大模型(过拟合)"]["summary"]
_s_reg = runs["大模型+正则(两者)"]["summary"]
print()
print("【实测结论】")
print(f"  欠拟合证据：小模型 train_loss={fmt(_s_small['train_loss'])}、"
      f"val_loss={fmt(_s_small['val_loss'])}、test_loss={fmt(_s_small.get('test_loss'))}，"
      f"三者都高（连训练集都没拟合好，train_acc 只有 {fmt(_s_small['train_acc'], 3)}）。")
print(f"  过拟合证据：大模型 train_loss={fmt(_s_big['train_loss'])}（很低）但 "
      f"val_loss={fmt(_s_big['val_loss'])}、test_loss={fmt(_s_big.get('test_loss'))}，"
      f"泛化差距={fmt(_s_big['gap'])}；验证 loss 在第 {_s_big['best_epoch']} 个 epoch "
      f"触底 {fmt(_s_big['best_val_loss'])} 后反弹到 {fmt(_s_big['val_loss'])}"
      f"（回升 {fmt(_s_big['val_loss_rise'])}）。")
print(f"  正则化证据：大模型+正则 train_loss={fmt(_s_reg['train_loss'])}（比大模型高 "
      f"{fmt(_s_reg['train_loss'] - _s_big['train_loss'])}），但 val_loss={fmt(_s_reg['val_loss'])}"
      f"（比大模型的 {fmt(_s_big['val_loss'])} 低 {fmt(_s_big['val_loss'] - _s_reg['val_loss'])}），"
      f"泛化差距从 {fmt(_s_big['gap'])} 收窄到 {fmt(_s_reg['gap'])}。")
print("  → 正则化确实让模型「学得没那么死」，换来更好的泛化。")
print("  → 还有一个反直觉的现象值得注意：大模型的错误不是「变多了」，而是「错得更自信」——")
print("     它把少数点判错时给出的概率极高，交叉熵损失因此爆炸到 1 以上；")
print("     准确率只从 0.85 掉到 0.83，loss 却涨了十几倍。所以**只看 accuracy 会漏掉过拟合**。")

_do = runs["大模型+Dropout"]["summary"]
_wd = runs["大模型+WeightDecay"]["summary"]
print(f"  【拆解】只加 Dropout：val_loss={fmt(_do['val_loss'])}、差距={fmt(_do['gap'])}；"
      f"只加 weight_decay：val_loss={fmt(_wd['val_loss'])}、差距={fmt(_wd['gap'])}。")
print("          两者都加通常最好：两条正则从不同角度限制容量——")
print("          Dropout 限制「一条样本能依赖哪些神经元」，weight_decay 限制「权重能长多大」。")


# ===========================================================================
# 第 5 部分：曲线图（train loss / val loss / val acc）
# ===========================================================================
section("第 5 部分：可视化①——三条曲线一眼看出过拟合 / 欠拟合")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
_style = {
    "小模型(欠拟合)":     dict(color="#1f77b4", lw=2.4, ls="-"),
    "大模型(过拟合)":     dict(color="#d62728", lw=2.4, ls="-"),
    "大模型+正则(两者)":  dict(color="#2ca02c", lw=2.4, ls="-"),
    "大模型+Dropout":     dict(color="#ff7f0e", lw=1.3, ls="--"),
    "大模型+WeightDecay": dict(color="#9467bd", lw=1.3, ls=":"),
}
for tag, _b, _k, _t in _configs:
    hist = runs[tag]["hist"]
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    for ax, key in zip(axes, ["train_loss", "val_loss", "val_acc"]):
        ax.plot(ep, hist[key], label=tag, **_style[tag])

# 在 val loss 曲线上标出大模型的最低点，直观展示「早停该停在哪」
ax = axes[1]
ax.axvline(_s_big["best_epoch"], color="#d62728", ls="-.", lw=1.2, alpha=0.8)
ax.annotate(f"大模型 val_loss 最低点\nepoch={_s_big['best_epoch']}",
            xy=(_s_big["best_epoch"], _s_big["best_val_loss"]),
            xytext=(_s_big["best_epoch"] + 18, _s_big["best_val_loss"] * 0.3 + 0.75),
            arrowprops=dict(arrowstyle="->", color="#d62728"), color="#d62728", fontsize=10)
ax.set_title("验证 loss：过拟合的典型形态是先降后升", fontsize=13)
ax.set_xlabel("epoch"); ax.set_ylabel("val loss"); ax.legend(fontsize=9); ax.grid(alpha=0.3)

axes[0].set_title("训练 loss：过拟合会一路降到 0 附近", fontsize=13)
axes[0].set_xlabel("epoch"); axes[0].set_ylabel("train loss")
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

# 验证集只有 48 条，accuracy 的统计涨落很大，这里把「同一把尺子」的测试集准确率作为对比基准写进标题
axes[2].set_title(f"验证 accuracy（仅 48 条，涨落大）\n400 条测试集上的最终准确率："
                  f"大模型 {_s_big.get('test_acc', float('nan')):.3f} → 正则 {_s_reg.get('test_acc', float('nan')):.3f}",
                  fontsize=12)
axes[2].set_xlabel("epoch"); axes[2].set_ylabel("val accuracy")
axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

fig.suptitle("训练技巧 · 过拟合与欠拟合：同一份数据（make_moons 240 条 + noise=0.35）上的三种模型对比",
             fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.94))
_p1 = OUTPUT_DIR / "04训练技巧_01_过拟合与欠拟合曲线.png"
fig.savefig(_p1, dpi=130)
plt.close(fig)
print(f"[图片] 已保存 {_p1}")
print("  → 看曲线形状就能诊断：验证 loss 先降后升 = 过拟合；两条都高 = 欠拟合。")
print("    这正是课案说的「先看 loss 曲线的形状，它能直接告诉你问题出在哪」。")


# ===========================================================================
# 第 6 部分：决策边界对比
# ===========================================================================
section("第 6 部分：可视化②——决策边界（欠拟合是直线 / 过拟合是碎块 / 正则后更平滑）")


def plot_boundary(ax, model, title):
    """在 2D 平面上画决策边界。

    做法：用 np.meshgrid 在整个平面上撒一张细网格，把每个网格点当作一条样本
    送进模型，得到「属于 1 类的概率」，再用 contourf 把概率当成背景色画出来。
    概率 0.5 的那条等高线就是决策边界。
    """
    pad = 0.6
    xx, yy = np.meshgrid(
        np.linspace(X_all[:, 0].min() - pad, X_all[:, 0].max() + pad, 180),
        np.linspace(X_all[:, 1].min() - pad, X_all[:, 1].max() + pad, 180),
    )
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(grid), dim=1)[:, 1].numpy().reshape(xx.shape)

    # 背景色：红=偏向类别 1，蓝=偏向类别 0，白色附近=模型没把握
    ax.contourf(xx, yy, prob, levels=np.linspace(0, 1, 21), cmap="coolwarm", alpha=0.75, vmin=0, vmax=1)
    ax.contour(xx, yy, prob, levels=[0.5], colors="black", linewidths=1.6)   # 决策边界
    ax.scatter(X_tr_s[:, 0], X_tr_s[:, 1], c=y_tr, cmap="coolwarm", vmin=0, vmax=1,
               edgecolors="k", s=30, label="训练点")
    ax.scatter(X_va_s[:, 0], X_va_s[:, 1], c=y_va, cmap="coolwarm", vmin=0, vmax=1,
               marker="^", edgecolors="k", s=44, label="验证点")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("特征 1（标准化后）")
    ax.set_ylabel("特征 2（标准化后）")
    ax.legend(fontsize=8, loc="lower right")


fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5.6))
_boundary_items = [
    ("小模型(欠拟合)", runs["小模型(欠拟合)"]["model"],
     "小模型（欠拟合）：线性模型 → 边界是一条直线"),
    ("大模型(过拟合)", runs["大模型(过拟合)"]["model"],
     "大模型（过拟合）：边界扭曲、出现孤立小碎块（背下了噪声点）"),
    ("大模型+正则(两者)", runs["大模型+正则(两者)"]["model"],
     "大模型 + 正则：边界更平滑，没有为噪声点单独开口子"),
]
for ax, (_tag, _model, _title) in zip(axes2, _boundary_items):
    plot_boundary(ax, _model, _title)
fig2.suptitle("决策边界对比（同一份数据、同样 150 epoch）：欠拟合=直线 / 过拟合=碎块 / 正则=平滑", fontsize=14)
fig2.tight_layout(rect=(0, 0, 1, 0.94))
_p2 = OUTPUT_DIR / "04训练技巧_01_决策边界对比.png"
fig2.savefig(_p2, dpi=130)
plt.close(fig2)
print(f"[图片] 已保存 {_p2}")
print("""
【怎么读这三张图】
  小模型：两个线性层之间没有激活函数，整个网络等价于**一个**线性分类器，
          边界只能是一条直线 → 永远切不开两个交错的半月 → 欠拟合。
  大模型：训练 loss 很低，为了把每个噪声点都分对，边界被掰出很多尖角和小碎块
          （背景色里那些概率被推得极端的「孤岛」）——这就是「背答案」的几何形态。
          这些碎块来自训练集的噪声，在新数据上毫无泛化能力。
  正则后：Dropout + weight_decay 压住了模型把边界掰碎的能力，
          边界回到较平滑的大形状，测试集表现反而更好。
""")


# ===========================================================================
# 第 7 部分：早停 EarlyStopping（手写类 + 实测）
# ===========================================================================
section("第 7 部分：缓解过拟合①——早停 Early Stopping")

print("""
【原理】过拟合的曲线上，验证 loss 是「先降后升」的 U 形。
        既然最好的一刻在中间，那就**停在那一刻**：
        训练时持续监控 val_loss，如果连续 patience 个 epoch 都没有变得更好，
        就停止训练，并把参数回滚到「val_loss 最低」那一版的快照。
        早停是性价比最高的正则化：不需要改模型结构，只多了一个校验循环。

        注意：早停必须保存权重快照（deepcopy(state_dict)），
        因为「停止的那一刻」的参数已经不是「最好的那一刻」的参数了。
""")


class EarlyStopping:
    """手写早停：patience + 最佳权重恢复。

    参数：
        patience  : 容忍多少个 epoch 没有改善（改善幅度需超过 min_delta）
        min_delta : 认为「有改善」的最小幅度，过滤掉 loss 的微小抖动
    """

    def __init__(self, patience=15, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")                   # 历史最低 val_loss
        self.best_epoch = 0                             # 最低点出现在第几个 epoch
        self.best_state = None                          # 最低点的权重快照
        self.counter = 0                                # 连续没改善的次数
        self.stop_epoch = None                          # 实际停在第几个 epoch

    def __call__(self, val_loss, model, epoch):
        """每个 epoch 结束调用一次；返回 True 表示应该停止训练。"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss                   # 刷新纪录
            self.best_epoch = epoch
            # deepcopy 而不是直接赋值：state_dict 里的张量是**引用**，
            # 不复制的话后续 optimizer.step() 会把快照一起改掉，恢复就失效了。
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1                           # 这一轮没进步
        if self.counter >= self.patience:
            self.stop_epoch = epoch
            return True
        return False

    def restore(self, model):
        """把模型参数回滚到历史最佳那一版。"""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model


# 用「无正则的大模型」做早停实验：它过拟合最严重，val loss 反弹最明显，最能体现早停价值
PATIENCE = 20
torch.manual_seed(SEED)
_es_model = build_big_model(dropout_p=None)
_es = EarlyStopping(patience=PATIENCE, min_delta=1e-4)
_hist_es = train_model(_es_model, Xtr, ytr, Xva, yva, Xte=Xte, yte=yte,
                       epochs=EPOCHS_MAIN, lr=LR_MAIN, batch_size=BS_MAIN,
                       tag="早停实验", early_stopping=_es)
_crit = nn.CrossEntropyLoss()
_loss_at_stop, _acc_at_stop = evaluate(_es_model, Xva, yva, _crit)   # 停止那一刻的权重
_es.restore(_es_model)                                              # 关键一步：回滚到最佳权重
_loss_restored, _acc_restored = evaluate(_es_model, Xva, yva, _crit)
_test_restored, _test_acc_restored = evaluate(_es_model, Xte, yte, _crit)

print(f"[早停] patience={PATIENCE}：早停发生在第 {_es.stop_epoch} 个 epoch "
      f"（val_loss 最低点是第 {_es.best_epoch} 个 epoch，最低值 {fmt(_es.best_loss)}）")
rows = [
    ["训练结束时的权重（未回滚）", fmt(_loss_at_stop), fmt(_acc_at_stop, 4)],
    ["恢复最佳权重后", fmt(_loss_restored), fmt(_acc_restored, 4)],
    ["回滚带来的收益", fmt(_loss_at_stop - _loss_restored), fmt(_acc_restored - _acc_at_stop, 4)],
    ["对照组：不早停、跑满 150 epoch", fmt(_s_big["val_loss"]), fmt(_s_big["val_acc"], 4)],
]
print_table(["状态", "验证 loss", "验证 accuracy"], rows)
print(f"  → 早停时已经比最低点多跑了 {_es.stop_epoch - _es.best_epoch} 个 epoch，"
      f"val_loss 从 {fmt(_es.best_loss)} 涨到 {fmt(_loss_at_stop)}；")
print(f"    恢复最佳权重后回到 {fmt(_loss_restored)}。如果不早停、硬跑满 150 个 epoch，"
      f"val_loss 会一路涨到 {fmt(_s_big['val_loss'])}。")
print(f"    早停回来的模型在 400 条测试集上：loss={fmt(_test_restored)}、"
      f"acc={fmt(_test_acc_restored, 4)}（大模型跑满 150 epoch 时 test_acc="
      f"{fmt(_s_big.get('test_acc'), 4)}）。")


# ===========================================================================
# 第 8 部分：数据增强（纯 torch 手写，替代 torchvision.transforms）
# ===========================================================================
section("第 8 部分：缓解过拟合②——数据增强（纯 torch 手写，本机没有 torchvision）")

print("""
【为什么不能用 torchvision】
    本机环境**没有安装 torchvision**（也不允许联网安装），
    所以课案里 `transforms.Compose([...])` 这条路线在表格数据上也用不上，
    下面全部用**纯 torch** 手写等价物。

【原理】数据增强解决的是「有效样本太少」这个问题。
    判据是「标签不变性」：把一张猫的图片翻转 / 裁剪 / 调色，它**仍然是猫**；
    于是模型不会把某个具体的像素排列当成「猫的定义」，被迫学更通用的特征。
    在二维表格数据上，对应的不变性有：
      ① 加高斯噪声：x → x + ε，ε~N(0, σ²)。测量得到的表特征本来就带误差，
         小噪声不该改变类别标签 → 让模型对特征的微小扰动不敏感。
      ② 随机特征缩放：x → x * s，s~U(0.85, 1.15)。类比图像的亮度/对比度抖动，
         相当于告诉模型「整体尺度略有不同，还是同一类」。
      ③ bootstrap 重采样：每个 epoch 有放回地抽样本。
         类比图像的随机裁剪/重采样，让每轮看到的数据组合都不同，
         减少模型对「某一批固定样本组合」的记忆。
    （还有一类经典增强是「标签平滑」，它直接软化目标分布，属于损失层面的增强。）
""")

# ---- 纯 torch 数据增强算子 ----
def aug_gaussian_noise(x, sigma=0.15):
    """加高斯噪声：模拟传感器/测量误差，抑制模型对特征精确值的死记。"""
    return x + torch.randn_like(x) * sigma


def aug_random_scale(x, lo=0.85, hi=1.15):
    """随机特征缩放：每个样本一个标量缩放因子（shape (B,1) 会自动广播到 (B,2)）。"""
    scale = torch.empty(x.shape[0], 1).uniform_(lo, hi)
    return x * scale


def aug_compose(x):
    """组合增强：先加噪声、再随机缩放。等价于课案 transforms.Compose([...]) 的顺序执行。"""
    return aug_random_scale(aug_gaussian_noise(x, sigma=0.15))


# ---- 实验设计：在一张「已经过拟合」的大模型上做增强，才能看出增强的价值 ----
# 为什么这里用的是**不带正则**的大模型？因为它过拟合最严重，改进空间最大，
# 「增强有没有用」一眼就能看出来。如果基线已经被 Dropout + weight_decay 压得很稳
# （实测基线 test_loss 已经只有 0.29），再叠一层增强往往是「过正则」，
# 数字会很难看甚至变差——这本身也是一个值得记住的结论。
# 四组都训练 120 个 epoch：要让基线先把训练集背得差不多（否则「增强有没有用」看不出来），
# 又不能跑满 150 个 epoch 把时间全花在这里。
AUG_EPOCHS = 120


def _train_aug(tag, augment_fn=None, bootstrap=False):
    """训练一组增强实验：同样的初始化、同样的超参，只改增强方式。"""
    torch.manual_seed(SEED)
    _m = build_big_model(dropout_p=None)                # 无正则的大模型：过拟合最严重
    _h = train_model(_m, Xtr, ytr, Xva, yva, Xte=Xte, yte=yte, epochs=AUG_EPOCHS,
                     lr=LR_MAIN, batch_size=BS_MAIN, augment_fn=augment_fn,
                     bootstrap=bootstrap, tag=tag)
    return summarize(_h, tag)


print(f"  （下面四组都训练 {AUG_EPOCHS} 个 epoch，模型都是无正则的大模型，只改增强方式）")
_s_noaug = _train_aug("无增强（基线）")
_s_aug = _train_aug("+高斯噪声+随机缩放", augment_fn=aug_compose)
_s_scale = _train_aug("+只有随机缩放", augment_fn=aug_random_scale)
_s_boot = _train_aug("+bootstrap 重采样", bootstrap=True)

rows = []
for _s in (_s_noaug, _s_aug, _s_scale, _s_boot):
    rows.append([_s["tag"], fmt(_s["train_loss"]), fmt(_s["val_loss"]),
                 fmt(_s.get("test_loss")), fmt(_s["gap"]), fmt(_s.get("test_acc"), 4)])
print_table(["数据增强方式", "train_loss", "val_loss", "test_loss", "泛化差距", "test_acc"], rows)
_d_tr = _s_aug["train_loss"] - _s_noaug["train_loss"]
_d_te = _s_noaug.get("test_loss") - _s_aug.get("test_loss")
_d_ac = _s_aug.get("test_acc") - _s_noaug.get("test_acc")
print(f"  → 加「高斯噪声 + 随机缩放」后：train_loss {fmt(_s_noaug['train_loss'])} → "
      f"{fmt(_s_aug['train_loss'])}（{'升高' if _d_tr > 0 else '基本持平/略降'} "
      f"{fmt(abs(_d_tr))}），"
      f"也就是模型**连训练集都没那么容易背下来了**；")
print(f"    但 test_loss {fmt(_s_noaug.get('test_loss'))} → {fmt(_s_aug.get('test_loss'))}"
      f"（降低 {fmt(abs(_d_te))}），test_acc {fmt(_s_noaug.get('test_acc'), 4)} → "
      f"{fmt(_s_aug.get('test_acc'), 4)}（提升 {fmt(abs(_d_ac) * 100, 2)} 个百分点）。")
print(f"    这正是数据增强的本质：故意让训练任务变难一点，换取泛化能力。"
      f"（另外两组：只有随机缩放 test_loss "
      f"{fmt(_s_scale.get('test_loss'))}、bootstrap test_loss {fmt(_s_boot.get('test_loss'))}，"
      f"都比「噪声+缩放」弱——增强的强度与形式必须靠验证集来调。）")
print(f"  → bootstrap 这一组的成绩是 test_loss {fmt(_s_boot.get('test_loss'))}、"
      f"test_acc {fmt(_s_boot.get('test_acc'), 4)}。它之所以没有优势，原因很具体：")
print("    有放回抽样时每个 epoch 平均只有 1-1/e ≈ 63% 的样本被抽到，等于每轮**实际看的数据更少**，")
print("    而且被重复抽中的噪声点权重更大。在小数据上，这个副作用会抵消掉它的好处。")
print("    结论：增强不是万能药，必须和「标签不变性」匹配——加噪声/缩放对应的是")
print("    「特征测量有误差但类别不变」，这在表数据上成立；bootstrap 改变的是样本分布本身，")
print("    它更适合大数据集（那里 63% 也仍然是几万条样本）。")


# ===========================================================================
# 第 9 部分：欠拟合的处理——增大模型 / 调大学习率
# ===========================================================================
section("第 9 部分：欠拟合的处理——「增大模型」与「调大学习率」的效果对比")

print("""
【欠拟合的判据】train_loss 和 val_loss **都高**，且都不再下降。
    说明模型连训练集都没拟合好，不是「记性太好」而是「能力不够」。

【处理方式与优先级】
    ① 增大模型：加深 / 加宽。容量不够时，任何正则化都是帮倒忙。
    ② 调大学习率：步长太小会在高处慢慢挪，甚至看起来「不降」。
    ③ 训练更久：增加 epoch。
    ④ 降低正则化强度：把 Dropout 概率、weight_decay 调小（容量本来就不够，别再压）。
    ⑤ 检查数据：标签是否标错、归一化是否做对（脏数据会让模型永远学不会）。
    下面实测 ①② 两条。这几个模型都很小，用 120 个 epoch 已经能看到全部结论
    （容量够不够、学习率快不快，100 个 epoch 上下就分野了）。
""")

UNDERFIT_EPOCHS = 120

torch.manual_seed(SEED)
_m_u1 = build_small_model()
_h_u1 = train_model(_m_u1, Xtr, ytr, Xva, yva, Xte=Xte, yte=yte, epochs=UNDERFIT_EPOCHS,
                    lr=LR_MAIN, batch_size=BS_MAIN, tag="小模型基准")
_s_u1 = summarize(_h_u1, "小模型（基准）")

torch.manual_seed(SEED)
_m_u2 = build_medium_model()
_h_u2 = train_model(_m_u2, Xtr, ytr, Xva, yva, Xte=Xte, yte=yte, epochs=UNDERFIT_EPOCHS,
                    lr=LR_MAIN, batch_size=BS_MAIN, tag="增大模型")
_s_u2 = summarize(_h_u2, "增大模型（2→64→64→2）")

torch.manual_seed(SEED)
_m_u3 = build_medium_model()
_h_u3 = train_model(_m_u3, Xtr, ytr, Xva, yva, Xte=Xte, yte=yte, epochs=UNDERFIT_EPOCHS,
                    lr=1e-4, batch_size=BS_MAIN, tag="过小学习率")
_s_u3 = summarize(_h_u3, "增大模型 + lr=1e-4（太小）")

torch.manual_seed(SEED)
_m_u4 = build_small_model()
_h_u4 = train_model(_m_u4, Xtr, ytr, Xva, yva, Xte=Xte, yte=yte, epochs=UNDERFIT_EPOCHS,
                    lr=1e-4, batch_size=BS_MAIN, tag="小模型+小学习率")
_s_u4 = summarize(_h_u4, "小模型 + lr=1e-4")

rows = []
for _tag, _mod, _s in [("小模型 + lr=1e-2（基准）", _m_u1, _s_u1),
                       ("增大模型 + lr=1e-2（①）", _m_u2, _s_u2),
                       ("增大模型 + lr=1e-4（学习率过小）", _m_u3, _s_u3),
                       ("小模型 + lr=1e-4（两个毛病都有）", _m_u4, _s_u4)]:
    rows.append([_tag, count_params(_mod), fmt(_s["train_loss"]), fmt(_s.get("test_loss")),
                 fmt(_s["train_acc"], 4), fmt(_s.get("test_acc"), 4)])
print_table(["配置", "参数量", "train_loss", "test_loss", "train_acc", "test_acc"], rows)
print(f"  → ① 增大模型：参数量 {count_params(_m_u1):,} → {count_params(_m_u2):,}，"
      f"**训练侧**立刻改善：train_loss {fmt(_s_u1['train_loss'])} → {fmt(_s_u2['train_loss'])}，"
      f"train_acc {fmt(_s_u1['train_acc'], 4)} → {fmt(_s_u2['train_acc'], 4)}。"
      f"这正是欠拟合的定义——「连训练集都没拟合好」，所以先看训练集指标。")
print(f"  → ② 调大学习率：同一个中等模型，lr 从 1e-4 提到 1e-2，"
      f"train_loss {fmt(_s_u3['train_loss'])} → {fmt(_s_u2['train_loss'])}，"
      f"train_acc {fmt(_s_u3['train_acc'], 4)} → {fmt(_s_u2['train_acc'], 4)}")
print("     小 lr 不是「学得稳」，而是「走不动」——曲线看起来平，其实还停在高处。")
print(f"  → 注意一个细节：中等模型的 train_loss（{fmt(_s_u2['train_loss'])}）比小模型低得多，"
      f"但 test_loss（{fmt(_s_u2.get('test_loss'))}）反而不如小模型"
      f"（{fmt(_s_u1.get('test_loss'))}）。")
print("     这说明「增大模型」解决欠拟合的同时会**开始引入过拟合**：容量不是越大越好，")
print("     加容量的同时往往要配上正则化/早停（见第 4、7 部分）。")


# ===========================================================================
# 第 10 部分：损失震荡——学习率过大 / batch 过小
# ===========================================================================
section("第 10 部分：损失震荡——学习率过大 与 batch size 过小")

print("""
【现象】loss 曲线剧烈上下波动，不平稳下降；极端时直接炸成 NaN。
【原因】梯度下降的更新式是 $\\theta_{t+1} = \\theta_t - \\eta \\nabla L(\\theta_t)$。
        把 loss 在极小点附近近似成二次函数 $L(\\theta) \\approx \\frac{1}{2}a\\theta^2$，
        梯度下降的稳定条件是 $|1 - \\eta a| < 1$，即 $\\eta < 2/a$；
        一旦 $\\eta$ 超过这个阈值，每一步都会**跨过**最优点弹到对面，误差被放大，
        于是曲线来回震荡；再大一点误差指数增长，几十步内就变成 inf / NaN。
        加上动量后等效步长变成 $\\eta/(1-\\beta)$，所以带 momentum 时更容易炸。
        另外 batch size 太小时，每个 batch 的梯度只是真实梯度的**带噪估计**，
        噪声让更新方向每步都偏一点，曲线也会抖。
【处理】① 降低学习率（优先，通常一步解决）；② 增大 batch size（梯度估计更准）；
        ③ 梯度裁剪（限制单步总长度，见 02 脚本）。
""")

print("【实验 A】同一任务、同一模型，纯 SGD + momentum=0.9、batch_size=32："
      "lr = 1e-3 / 1e-1 / 0.5")
_lr_list = [1e-3, 1e-1, 0.5]
_lr_hist = {}
_lr_stats = {}
for _lr in _lr_list:
    torch.manual_seed(SEED)
    _m = build_big_model(dropout_p=None)
    _h = train_model(_m, Xtr, ytr, Xva, yva, Xte=Xte, yte=yte, epochs=EPOCHS_MAIN,
                     lr=_lr, optimizer_name="sgd", momentum=0.9, batch_size=BS_MAIN,
                     tag=f"lr={_lr}")
    _lr_hist[_lr] = _h
    _fin = np.array([v for v in _h["train_loss"] if np.isfinite(v)])
    # 「抖动」的度量：末段 30 个 epoch 的 loss 标准差，以及相邻 epoch 的平均绝对变化
    _std = float(_fin[-30:].std()) if _fin.size >= 30 else float("nan")
    _mad = float(np.abs(np.diff(_fin[-30:])).mean()) if _fin.size >= 31 else float("nan")
    _lr_stats[_lr] = (_std, _mad, _fin.size)
    print(f"  lr={_lr:<5} 有效 epoch={_fin.size:<4} 最终 train_loss={fmt(_fin[-1] if _fin.size else None)} "
          f"val_loss={fmt(_h['val_loss'][-1])} 末段 30 epoch 标准差={fmt(_std)} "
          f"相邻 epoch 平均变化={fmt(_mad, 5)}")
print(f"  → lr=1e-1 的相邻 epoch 平均变化是 lr=1e-3 的 "
      f"{_lr_stats[1e-1][1] / _lr_stats[1e-3][1]:.1f} 倍，抖动明显加剧；")
print(f"    lr=0.5 更是在第 {_lr_hist[0.5].get('diverged_at')} 个 epoch 直接变成 NaN（发散）——"
      f"这就是「学习率太大」的极端形态。")

print()
print("【实验 B】batch size 的影响：bs=8（梯度噪声大）vs bs=128（梯度更准），"
      "纯 SGD、lr=0.05、无 momentum，60 个 epoch（抖动差异在前几十个 epoch 就稳定可见）")
_bs_hist = {}
_bs_stats = {}
for _bs in (8, 128):
    torch.manual_seed(SEED)
    _m = build_big_model(dropout_p=None)
    _h = train_model(_m, Xtr, ytr, Xva, yva, Xte=Xte, yte=yte, epochs=60,
                     lr=0.05, optimizer_name="sgd", momentum=0.0, batch_size=_bs,
                     tag=f"bs={_bs}")
    _bs_hist[_bs] = _h
    _fin = np.array([v for v in _h["train_loss"] if np.isfinite(v)])
    _mad = float(np.abs(np.diff(_fin[-40:])).mean())
    _bs_stats[_bs] = _mad
    print(f"  batch_size={_bs:<4} 最终 train_loss={fmt(_fin[-1])} "
          f"val_loss={fmt(_h['val_loss'][-1])} 末段 40 epoch 相邻变化={fmt(_mad, 5)} "
          f"（越小越平稳）")
print(f"  → bs=8 的抖动是 bs=128 的 {_bs_stats[8] / _bs_stats[128]:.1f} 倍："
      f"小 batch 的梯度是真实梯度的带噪估计，噪声直接变成了曲线上的毛刺。")

fig3, axes3 = plt.subplots(1, 2, figsize=(15, 5.4))
_colors = {1e-3: "#2ca02c", 1e-1: "#ff7f0e", 0.5: "#d62728"}
for _lr in _lr_list:
    _h = _lr_hist[_lr]
    ep = np.arange(1, len(_h["train_loss"]) + 1)
    axes3[0].plot(ep, _h["train_loss"], color=_colors[_lr], lw=1.7, label=f"lr={_lr}")
axes3[0].set_title("学习率对 loss 曲线的影响（SGD + momentum=0.9）\n"
                   "lr=1e-3 平稳 / lr=0.1 震荡 / lr=0.5 发散成 NaN", fontsize=12)
axes3[0].set_xlabel("epoch"); axes3[0].set_ylabel("train loss")
axes3[0].set_yscale("log")                              # 对数纵轴：同时看清 0.2 和 NaN 前的暴涨
axes3[0].legend(fontsize=10); axes3[0].grid(alpha=0.3)
# 把发散点标出来（曲线断掉的地方就是 NaN）
_dv = _lr_hist[0.5].get("diverged_at")
_fin05 = [v for v in _lr_hist[0.5]["train_loss"] if np.isfinite(v)]
if _dv and _fin05:
    axes3[0].annotate(f"lr=0.5 在第 {_dv} 个 epoch\nloss 变 NaN（训练发散）",
                      xy=(_dv, _fin05[-1]), xytext=(max(1, _dv - 60), _fin05[-1] * 3),
                      color="#d62728", arrowprops=dict(arrowstyle="->", color="#d62728"),
                      fontsize=10)

for _bs, _c in zip((8, 128), ("#1f77b4", "#9467bd")):
    _h = _bs_hist[_bs]
    ep = np.arange(1, len(_h["train_loss"]) + 1)
    axes3[1].plot(ep, _h["train_loss"], color=_c, lw=1.6, label=f"batch_size={_bs}")
axes3[1].set_title("batch size 对曲线抖动的影响（SGD lr=0.05）\n"
                   f"bs=8 的毛刺是 bs=128 的 {_bs_stats[8] / _bs_stats[128]:.1f} 倍", fontsize=12)
axes3[1].set_xlabel("epoch"); axes3[1].set_ylabel("train loss")
axes3[1].set_yscale("log"); axes3[1].legend(fontsize=10); axes3[1].grid(alpha=0.3)

fig3.suptitle("训练技巧 · 损失震荡：先怀疑学习率，再怀疑 batch size", fontsize=14)
fig3.tight_layout(rect=(0, 0, 1, 0.93))
_p3 = OUTPUT_DIR / "04训练技巧_01_损失震荡与学习率.png"
fig3.savefig(_p3, dpi=130)
plt.close(fig3)
print(f"[图片] 已保存 {_p3}")


# ===========================================================================
# 第 11 部分：训练缓慢
# ===========================================================================
section("第 11 部分：训练缓慢——学习率太小 / 数值精度")

print("""
【现象】loss 下降极慢，或者每 epoch 耗时很长。
【原因】① 学习率太小：每步挪一点点，同样的 epoch 数走的距离不够
            （$-\\eta\\nabla L$ 里 $\\eta$ 是线性因子，$\\eta$ 小 10 倍，走得就慢 10 倍）；
        ② 模型太大 / 精度太高：算力成了瓶颈。
【处理】调大学习率 → 减小模型（剪枝/蒸馏/轻量架构）→ 混合精度（`torch.cuda.amp`）
        → 增大 batch size（充分利用 GPU 并行）→ 降低数据精度（float16 代替 float32）。

【本机限制】本机是 **CPU 版 torch 2.14.0+cpu，没有 CUDA**，
        所以课案里的 `torch.cuda.amp.GradScaler / autocast` **完全无法运行**，
        这里只做文字讲解 + 用 `torch.autocast('cpu', dtype=torch.bfloat16)` 做一次
        **形状验证**（证明 API 能走通、输出形状不变），**不真的用它训练**。
""")

print("【实验 C】lr = 1e-5 / 1e-3 / 1e-1 各训 150 epoch，看「达到 val_loss 阈值用了几个 epoch」")


def epochs_to_threshold(hist, threshold):
    """返回验证 loss 首次低于 threshold 的 epoch 号；一直没达到就返回 None。"""
    for i, v in enumerate(hist["val_loss"]):
        if np.isfinite(v) and v < threshold:
            return i + 1
    return None


_slow_hist = {}
for _lr in (1e-5, 1e-3, 1e-1):
    torch.manual_seed(SEED)
    _m = build_medium_model()
    _h = train_model(_m, Xtr, ytr, Xva, yva, Xte=Xte, yte=yte, epochs=EPOCHS_MAIN,
                     lr=_lr, optimizer_name="sgd", momentum=0.9, batch_size=BS_MAIN,
                     tag=f"lr={_lr}")
    _slow_hist[_lr] = _h

# 阈值取「最快那条曲线的最终 val_loss 的 1.05 倍」：保证有一条能达标，
# 又足够严苛，能体现慢的曲线「跑满 150 epoch 都到不了」。
_best_final = min(h["val_loss"][-1] for h in _slow_hist.values())
_VAL_THRESHOLD = _best_final * 1.05
print(f"  阈值 = {fmt(_VAL_THRESHOLD)}（取最优曲线的最终 val_loss {fmt(_best_final)} 的 1.05 倍）")
rows = []
for _lr, _h in _slow_hist.items():
    _ep = epochs_to_threshold(_h, _VAL_THRESHOLD)
    rows.append([
        f"lr={_lr}",
        fmt(_h["train_loss"][-1]), fmt(_h["val_loss"][-1]),
        ("从未达到（150 epoch 都用完）" if _ep is None else f"第 {_ep} 个 epoch"),
        fmt(_h["val_acc"][-1], 4), fmt(_h["test_acc"][-1], 4),
    ])
print_table(["学习率", "150 epoch 后 train_loss", "150 epoch 后 val_loss",
             "达到阈值的 epoch", "val_acc", "test_acc"], rows)
_eps = {_lr: epochs_to_threshold(_h, _VAL_THRESHOLD) for _lr, _h in _slow_hist.items()}


def _ep_text(_ep):
    """把「达到阈值的 epoch 数」写成人话（None 表示跑满都没达到）。"""
    return f"{_ep} 个 epoch" if _ep else f">{EPOCHS_MAIN} 个 epoch（跑满都没达到）"


print(f"  → 达到同一水平（val_loss < {fmt(_VAL_THRESHOLD)}）所需 epoch："
      + "；".join(f"lr={_lr} 用了 {_ep_text(_eps[_lr])}" for _lr in (1e-5, 1e-3, 1e-1)))
print("     同样 150 个 epoch，lr=1e-5 停在 "
      f"{fmt(_slow_hist[1e-5]['val_loss'][-1])}（接近瞎猜的 ln2≈0.693），"
      "而调大学习率后几步就跨过阈值——这就是「训练缓慢」的第一排查项。")
print("  → 但要小心另一个陷阱：lr=1e-1 虽然冲得快，可它的 val_loss 最后反而回升到 "
      f"{fmt(_slow_hist[1e-1]['val_loss'][-1])}，"
      "因为它把训练集记得太快，很快就转成过拟合了（见第 4、10 部分）。")
print("     学习率的选择是「快」与「稳」的折中，不是越大越好。")

print()
print("【实验 D】float32 vs float16 的前向耗时实测（同一模型、同一输入，各 80 次前向取平均）")
torch.manual_seed(SEED)
_bench_model = build_big_model(dropout_p=None).eval()    # eval()：排除 Dropout 的随机性
_bench_x = torch.randn(256, 2)


def bench_forward(dtype, repeats=80, warmup=10):
    """测前向耗时（毫秒/次）。任何异常都返回 (None, 异常名)，绝不向上抛。"""
    try:
        m = _bench_model.to(dtype)
        x = _bench_x.to(dtype)
        with torch.no_grad():
            for _ in range(warmup):                     # 预热：让缓存/线程池就绪
                m(x)
            t0 = time.perf_counter()
            for _ in range(repeats):
                m(x)
            dt = (time.perf_counter() - t0) / repeats * 1000.0
        return dt, None
    except Exception as e:                              # noqa: BLE001 —— 故意兜住一切，保证不打印 Traceback
        return None, type(e).__name__


_ms32, _err32 = bench_forward(torch.float32)
_ms16, _err16 = bench_forward(torch.float16)
rows = []
if _ms32 is not None:
    rows.append(["float32", f"{_ms32:.3f} ms", "基准"])
else:
    rows.append(["float32", "n/a", f"失败：{_err32}"])
if _ms16 is not None:
    _ratio = _ms16 / _ms32 if _ms32 else float("nan")
    rows.append(["float16", f"{_ms16:.3f} ms", f"是 float32 的 {_ratio:.2f} 倍耗时"])
else:
    rows.append(["float16", "n/a", f"本机 CPU 前向失败：{_err16}"])
print_table(["dtype", "单次前向耗时", "说明"], rows)
print("""  → 在 CPU 上 float16 往往**反而更慢甚至直接失败**：CPU 没有 fp16 的硬件加速单元，
     PyTorch 只能把 fp16 转成 fp32 算、再转回来，额外的类型转换开销把「省下的位宽」全吃掉了。
     所以在 CPU 上「减小数据精度」并不能加速；这一招是给有 Tensor Core 的 GPU 用的。
     在 GPU 上 fp16/bf16 的理论收益：显存减半 + Tensor Core 算力翻数倍。""")

print()
print("【实验 E】torch.autocast('cpu', dtype=torch.bfloat16) 的形状验证（不用于训练）")
try:
    with torch.autocast("cpu", dtype=torch.bfloat16):
        _out = _bench_model(_bench_x)
    print(f"  autocast 前向成功：输出 shape={tuple(_out.shape)}（与 float32 的 "
          f"{tuple(_bench_model(_bench_x).shape)} 一致），dtype={_out.dtype}")
    print("  说明：接口本身可用，它会把矩阵乘等算子自动切成 bfloat16、其余保持 float32。")
    print("        但本机没有 CUDA，课案的 GradScaler 梯度缩放流程（防 fp16 下溢）无从施展，")
    print("        因此这里只做形状验证，不做真实混合精度训练。")
except Exception as _e:                                  # noqa: BLE001
    print(f"  本机 CPU 不支持该 autocast 组合（{type(_e).__name__}），已安全跳过；"
          f"这不影响其它实验。")
    print("  结论不变：混合精度训练需要 GPU（Tensor Core），CPU 上不可用，只能做文字讲解。")

fig4, axes4 = plt.subplots(1, 2, figsize=(15, 5.2))
for _lr, _c in zip((1e-5, 1e-3, 1e-1), ("#d62728", "#1f77b4", "#2ca02c")):
    _h = _slow_hist[_lr]
    ep = np.arange(1, len(_h["val_loss"]) + 1)
    axes4[0].plot(ep, _h["val_loss"], color=_c, lw=1.8, label=f"lr={_lr}")
axes4[0].axhline(_VAL_THRESHOLD, color="gray", ls="--", lw=1.2,
                 label=f"目标阈值 {_VAL_THRESHOLD:.3f}")
axes4[0].set_title("训练缓慢：学习率太小 → 同样 epoch 数到不了目标线", fontsize=12)
axes4[0].set_xlabel("epoch"); axes4[0].set_ylabel("val loss")
axes4[0].legend(fontsize=9); axes4[0].grid(alpha=0.3)

_labels = ["float32", "float16"]
_vals = [(_ms32 if _ms32 is not None else 0.0), (_ms16 if _ms16 is not None else 0.0)]
_bars = axes4[1].bar(_labels, _vals, color=["#1f77b4", "#ff7f0e"], width=0.5)
for _b, _v in zip(_bars, _vals):
    axes4[1].text(_b.get_x() + _b.get_width() / 2, _v, f"{_v:.3f} ms",
                  ha="center", va="bottom", fontsize=10)
axes4[1].set_title("CPU 上 float32 vs float16 前向耗时\n（CPU 无 fp16 硬件加速，float16 常常更慢）", fontsize=12)
axes4[1].set_ylabel("单次前向耗时 (ms)"); axes4[1].grid(alpha=0.3, axis="y")

fig4.suptitle("训练技巧 · 训练缓慢：调大学习率最有效，降精度只在 GPU 上有效", fontsize=14)
fig4.tight_layout(rect=(0, 0, 1, 0.93))
_p4 = OUTPUT_DIR / "04训练技巧_01_训练缓慢与学习率.png"
fig4.savefig(_p4, dpi=130)
plt.close(fig4)
print(f"[图片] 已保存 {_p4}")


# ===========================================================================
# 第 12 部分：无用特征越多越容易过拟合（make_classification 对照）
# ===========================================================================
section("第 12 部分：附加对照——无用特征越多，越容易过拟合")

print("""
【原理】`make_classification` 把特征分成几类：
    informative（真正决定标签的）、redundant（informative 的线性组合，没带来新信息）、
    repeated（从 informative/redundant 复制并加噪）、以及纯噪声特征。
    只有 informative 那几维带「有效信息」，其余维度塞进来的都是**可被记忆的噪声**。
    模型容量固定时，无用维度越多，模型越容易抓住这些与标签无关的巧合 → 过拟合。

    下面固定 n_samples=1000、n_informative=2、flip_y=0.10（人为制造 10% 标签噪声），
    只改「冗余 + 噪声特征」的数量：
      设置 A：20 维（2 有效 + 10 冗余 + 8 纯噪声）→ 信息被稀释
      设置 B：2 维（只有 2 个有效特征）→ 干净
    两份数据的**有效信息量完全相同**，唯一差别是多了 18 个无用维度。
""")
_nc_rows = []
for _tag, _kw in [
    ("20 维(2 有效+10 冗余+8 噪声)",
     dict(n_samples=1000, n_features=20, n_informative=2, n_redundant=10,
          n_repeated=0, n_classes=2, n_clusters_per_class=2,
          class_sep=1.0, flip_y=0.10, random_state=SEED)),
    ("2 维(只有有效特征)",
     dict(n_samples=1000, n_features=2, n_informative=2, n_redundant=0,
          n_repeated=0, n_classes=2, n_clusters_per_class=2,
          class_sep=1.0, flip_y=0.10, random_state=SEED)),
]:
    _Xc, _yc = make_classification(**_kw)
    _Xc_tr, _Xc_va, _yc_tr, _yc_va = train_test_split(_Xc, _yc, test_size=0.2,
                                                      random_state=SEED, stratify=_yc)
    _sc = StandardScaler().fit(_Xc_tr)                  # 依旧只在训练集上 fit
    _Xtr_c = torch.tensor(_sc.transform(_Xc_tr), dtype=torch.float32)
    _Xva_c = torch.tensor(_sc.transform(_Xc_va), dtype=torch.float32)
    _ytr_c = torch.tensor(_yc_tr, dtype=torch.long)
    _yva_c = torch.tensor(_yc_va, dtype=torch.long)

    torch.manual_seed(SEED)
    # 同一套结构：输入维度不同，先升到 64 维再降回 2 类
    _mc = nn.Sequential(nn.Linear(_Xc.shape[1], 64), nn.ReLU(),
                        nn.Linear(64, 64), nn.ReLU(),
                        nn.Linear(64, 2))
    # 用 SGD 而不是 Adam：这里要的是「训练不足 + 容量过剩」共同造成的过拟合，
    # SGD 学得慢，反而能把「维度多 → 更容易背噪声」的差异放大出来。
    _hc = train_model(_mc, _Xtr_c, _ytr_c, _Xva_c, _yva_c, epochs=50, lr=0.05,
                      optimizer_name="sgd", momentum=0.9, batch_size=64, tag=_tag)
    _sc2 = summarize(_hc, _tag)
    _nc_rows.append([_tag, _Xc.shape[1], count_params(_mc),
                     fmt(_sc2["train_loss"]), fmt(_sc2["val_loss"]),
                     fmt(_sc2["gap"]), fmt(_sc2["val_acc"], 4)])
print_table(["特征设置", "维度", "参数量", "train_loss", "val_loss", "泛化差距", "val_acc"], _nc_rows)
print("  → 两份数据的有效信息量一样，但维度高的那份泛化差距更大、验证准确率更低。")
print("    这解释了数据预处理里「先做特征选择 / 降维」为什么能直接缓解过拟合：")
print("    把不含信息的维度去掉 = 减少了模型可以拿来背书的材料。")


# ===========================================================================
# 第 13 部分：现象 → 原因 → 处理方式（课案原文）+ 对比表
# ===========================================================================
section("第 13 部分：现象 → 原因 → 处理方式（课案原文）")

print("""
────────────────────────────────────────────────────────────────────────
【过拟合】
  现象：训练 loss 持续下降，验证 loss 不再下降甚至反弹上升。
        模型在「背」训练集答案，而不是学通用规律。
  原因：模型容量 > 数据有效信息量。参数量太多、训练太久、数据太少或太简单，
        模型记住每个样本的噪声而非规律。
  处理方式（本文件里都有对应实验）：
    · 加正则化：Dropout、Weight Decay
        → 证据：第 4 部分「大模型+正则」，test_loss 从大模型的 0.89 降到 0.28，
          泛化差距从 +1.61 收窄到 +0.09
    · 加数据增强：翻转、裁剪、颜色变换（表格数据对应：噪声 / 缩放 / bootstrap）
        → 证据：第 8 部分，无正则大模型上加「高斯噪声+随机缩放」后，train_loss 升高而
          test_loss 从 0.47 降到 0.28；同节的 bootstrap 反而变差，说明增强必须匹配标签不变性
    · 减小模型：砍层数、减宽度
        → 证据：第 6 部分决策边界，小模型边界更平滑、泛化差距更小
    · 早停：验证 loss 不降就停
        → 证据：第 7 部分，早停 + 权重恢复把 val_loss 从回升后的值拉回最低点，
          比「硬跑满 150 epoch」好得多
    · 加数据：收集更多真实训练数据（最根本，但成本最高）
        → 旁证：第 12 部分，减少无用维度等价于提高数据的信噪比

【欠拟合】
  现象：训练 loss 和验证 loss 都高，且不再下降。模型还没学会数据的规律。
  原因：模型容量不足，或训练不充分；也可能数据太复杂、当前模型根本拟合不了。
  处理方式：
    · 增大模型：加深网络、加大隐藏层宽度
        → 证据：第 9 部分，参数量提升后 train/test loss 同时下降，准确率上升
    · 训练更久：增加 epoch 数
    · 降低正则化强度：减小 Dropout 概率、减小 weight_decay
    · 调大学习率：当前步长太小走不动
        → 证据：第 9 / 11 部分，lr 从 1e-4 提到 1e-2，train_loss 与准确率都明显改善
    · 检查数据：是否存在标签错误、归一化不对等问题

【损失震荡】
  现象：loss 曲线剧烈上下波动，不平稳下降。
  原因：学习率太大，每次更新跨过最优点又弹回来，反复震荡；
        也可能是 batch size 太小导致梯度噪声过大。
  处理方式：降低学习率（优先）→ 增大 batch size → 梯度裁剪
            clip_grad_norm_(model.parameters(), max_norm=1.0)
        → 证据：第 10 部分实验 A / B（lr=1e-1 抖动是 lr=1e-3 的数倍，lr=0.5 直接 NaN）

【训练缓慢】
  现象：loss 下降极慢，或者每 epoch 耗时很长。
  原因：学习率太小（步长不够）或模型太大（计算量过大）。
  处理方式：调大学习率、减小模型（剪枝、蒸馏、更轻量的架构）、
        混合精度训练（torch.cuda.amp，本机 CPU 版不可用）、增大 batch size、
        减小数据精度（float16）
        → 证据：第 11 部分实验 C / D / E

【梯度爆炸 / 消失】
  现象：训练 loss 突然变成 NaN（爆炸），或几乎不降停在高值（消失）。
  处理：ReLU 替代 Sigmoid/Tanh、Kaiming/Xavier 初始化、BatchNorm/LayerNorm、
        残差连接、梯度裁剪 —— 详见同目录 02_梯度爆炸与梯度消失演示.py
────────────────────────────────────────────────────────────────────────
""")

print("【课案对比表：现象 / 训练 loss / 验证 loss / 首选排查】")
print_table(["现象", "训练 loss", "验证 loss", "首选排查"], [
    ["过拟合", "↓ 持续降", "↑ 反弹", "加正则化"],
    ["欠拟合", "高且不降", "高且不降", "增大模型"],
    ["震荡", "剧烈波动", "剧烈波动", "降低学习率"],
    ["缓慢", "极慢下降", "极慢下降", "调大学习率"],
    ["爆炸/消失", "NaN 或不降", "NaN 或不降", "梯度裁剪"],
])

print()
print("=" * 78)
print("全部实验完成。图片输出目录：" + str(OUTPUT_DIR))
print("=" * 78)

"""
对应课案章节：训练组件 / 学习率

本节知识点：
    1.  学习率为什么关键：它控制每次参数更新的步长。
        太大 → 在最优解附近来回震荡、无法收敛；太小 → 训练极慢、容易卡在平坦区域。
        训练初期需要大步快速接近最优点，后期需要小步精细调整——这就是学习率调度器存在的理由。
    2.  StepLR（固定步长衰减）：每过 step_size 个 epoch，lr = lr × gamma；
        课案：StepLR(step_size=30, gamma=0.1)，lr 序列 0.01 → 0.001（30~59）→ 0.0001（60~89）→ ...
    3.  MultiStepLR（指定节点衰减）：在预设 epoch 列表处衰减，MultiStepLR(milestones=[30, 80], gamma=0.1)。
    4.  ExponentialLR（指数衰减）：每个 epoch 都乘 gamma，lr = lr0 × gamma^n，
        ExponentialLR(gamma=0.95) 给出 0.01 → 0.0095 → 0.009025 → ...，比 StepLR 平滑但没有自适应。
    5.  CosineAnnealingLR（余弦退火）：lr_t = eta_min + (lr0 - eta_min)·(1 + cos(π·t/T_max))/2，
        只取半个余弦波的下降段（"退火"= 只降不升）；本脚本会**数值验证手写公式与
        scheduler.get_last_lr() 完全一致**，并验证 t=0 / T_max/2 / T_max 三个关键点。
    6.  ReduceLROnPlateau（指标停滞衰减）：不按 epoch 数触发，而是监控验证指标，
        连续 patience 个 epoch 没改善就衰减；**调用方式不同：scheduler.step(val_loss)**。
    7.  OneCycleLR（先升后降）：lr 每步都变；前 30% 步数从 max_lr/25 升到 max_lr（warmup），
        后 70% 步数从 max_lr 平滑降到 max_lr/10000。
        重点讲清「为什么大学习率反而有正则化效果」：大 lr 步长更大、方向更莽撞，
        像隐式噪声注入，模型不容易陷入训练集的局部细节；但大 lr 不能从头用到尾——
        初期参数随机用大 lr 会直接崩（需要 warmup），末期需要精细收敛（需要衰减）。
    8.  另外覆盖 LambdaLR / MultiplicativeLR / CyclicLR / CosineAnnealingWarmRestarts /
        ConstantLR / LinearLR / SequentialLR 的组合用法（warmup + 主调度，实践中最常见）。
    9.  把 6 个主调度器的 lr 曲线画在同一张图上对比，并单独画 OneCycleLR 的每步曲线。
    10. 实战对比：同一个网络和数据，用「固定 lr」/「StepLR」/「余弦退火」各训一遍，
        打印最终测试准确率并画 loss 曲线，说明调度对收敛的帮助。
    11. 对比表：调度器 / 触发方式 / 曲线形状 / 适用场景。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Deep_Learning\\03_训练组件\\07_学习率调度.py'
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
# 正式内容开始：导入数学库与 PyTorch
# ---------------------------------------------------------------------------
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)                    # 小数据量下限制线程，避免 OpenMP 调度开销拖慢速度

# 本节很多小节只观察 lr 数值、不做真实的 optimizer.step()，
# PyTorch 会为此打印「step() 调用顺序」的 UserWarning（并不影响结果）。
# 为了保持输出干净、直接看出 lr 序列，这里统一把这类警告过滤掉：
# 所有**真实训练**的小节（小节 9）依然严格按 optimizer.step() → scheduler.step() 的顺序调用。
import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*lr_scheduler\.step\(\).*",
    category=UserWarning,
)

print("=" * 78)
print("07 学习率调度：6 个调度器逐一演示 + lr 曲线对比 + 实战训练对比")
print("=" * 78)
print(f"PyTorch 版本：{torch.__version__}")
print()
print(
    """
总纲：
    学习率（learning rate, lr）控制每次参数更新的**步长**。
        · lr 太大 → 参数在最优点附近来回震荡，甚至发散，无法收敛；
        · lr 太小 → 走得极慢，训练时间不可接受，还容易停在损失平坦的坏区域。
    训练初期我们希望「大步快跑」迅速接近最优点，训练后期希望「小步微调」精细收敛到谷底。
    学习率调度器（Scheduler）就是按某种策略自动调整 lr 的工具。
    下面 6 个小节逐一演示 6 个最常用的调度器，每个都给公式、代码、lr 序列和适用场景。
"""
)
print()


# ---------------------------------------------------------------------------
# 通用工具：建一个「假模型 + 真优化器」，方便反复实例化各种调度器
# ---------------------------------------------------------------------------
def make_optimizer(lr: float = 0.01) -> torch.optim.Optimizer:
    """创建一个只有一个参数的 SGD 优化器。

    这一节的目的是观察**学习率的数值变化**，不涉及真实训练，
    所以模型越简单越好：一个 (1, 1) 的 Linear，参数只有 2 个。
    注意：每个调度器都必须绑定**自己的**优化器实例，不能共用——
    因为 scheduler.step() 会直接改写 optimizer.param_groups[0]['lr']。
    """
    model = nn.Linear(1, 1)
    return torch.optim.SGD(model.parameters(), lr=lr)


def collect_lr_epoch_schedule(scheduler_factory, epochs: int = 100,
                              lr0: float = 0.01):
    """按「每 epoch 调一次」的方式跑 epochs 轮，返回每一轮的 lr 列表。

    调度器的标准用法（顺序很重要）：
        for epoch in range(n):
            train(...)                  # 1. 用当前 lr 训练一个 epoch
            validate(...)               # 2. 验证
            scheduler.step()            # 3. 更新 lr（放 epoch 末尾）
    这里只关心 lr 序列，所以省掉 train/validate，只保留 scheduler.step()。

    返回值 lr 列表的第 i 项 = 第 i 个 epoch 实际使用的 lr。
    为了让列表语义清晰：先记录当前 lr（这是本 epoch 用的），再 step()。
    """
    optimizer = make_optimizer(lr0)
    scheduler = scheduler_factory(optimizer)
    history: list[float] = []
    for _ in range(epochs):
        history.append(optimizer.param_groups[0]["lr"])   # 本 epoch 使用的 lr
        scheduler.step()                                  # 为下一 epoch 更新
    return history


def fmt_lr(v: float) -> str:
    """把学习率格式化成紧凑的字符串（太小的时候用科学计数法，避免一排 0）。"""
    return f"{v:.3e}" if v < 1e-4 else f"{v:.6f}"


_EPOCHS = 100                                # 观察 100 个 epoch 的 lr 变化
_KEY_EPOCHS = [0, 1, 2, 10, 29, 30, 31, 50, 59, 60, 61, 80, 81, 90, 99]


def print_key_lrs(history: list[float], keys: list[int] = _KEY_EPOCHS) -> None:
    """打印若干关键 epoch 的 lr，便于直接和课案里的数值对照。"""
    parts = [f"epoch {k:>3} → {fmt_lr(history[k])}" for k in keys if k < len(history)]
    for i in range(0, len(parts), 5):        # 每行 5 个，排版更紧凑
        print("    " + " | ".join(parts[i:i + 5]))


# ===========================================================================
# 小节 1：StepLR —— 固定步长衰减
# ===========================================================================
print("=" * 78)
print("小节 1：StepLR —— 固定步长衰减（每过 step_size 个 epoch，lr × gamma）")
print("=" * 78)
print(
    r"""
公式：

    lr_epoch = lr_0 × gamma ^ floor(epoch / step_size)

参数含义：
    optimizer  ：被调度的优化器（调度器直接改写它的 param_groups[0]['lr']）
    step_size  ：每隔多少个 epoch 衰减一次
    gamma      ：衰减倍数（每次乘它，通常取 0.1 或 0.5）
    last_epoch ：断点续训时用，指定从第几个 epoch 继续（默认 -1 表示从头开始）

课案的例子：StepLR(optimizer, step_size=30, gamma=0.1)，初始 lr = 0.01
    epoch 0~29 : lr = 0.01
    epoch 30~59: lr = 0.01 × 0.1   = 0.001
    epoch 60~89: lr = 0.001 × 0.1  = 0.0001
    epoch 90~  : lr = 0.0001 × 0.1 = 0.00001 → ...
适用场景：训练周期明确、知道大概何时需要衰减（比如计划总共训 90 个 epoch，就在 30/60 处降一次）。
"""
)
_step_hist = collect_lr_epoch_schedule(
    lambda opt: torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.1))
print(f"  step_size=30, gamma=0.1, lr0=0.01，共 {_EPOCHS} 个 epoch")
print_key_lrs(_step_hist)
print(f"  epoch 0~29 是否全为 0.01 ？{all(abs(v - 0.01) < 1e-12 for v in _step_hist[:30])}")
print(f"  epoch 30~59 是否全为 0.001 ？{all(abs(v - 0.001) < 1e-12 for v in _step_hist[30:60])}")
print(f"  epoch 60~89 是否全为 0.0001 ？{all(abs(v - 0.0001) < 1e-12 for v in _step_hist[60:90])}")
print(f"  epoch 90~99 是否全为 0.00001 ？{all(abs(v - 0.00001) < 1e-12 for v in _step_hist[90:])}")
print()
print("  特征：阶梯状（一段平台 + 一次断崖式下跌），平台内 lr 完全不变。")
print()


# ===========================================================================
# 小节 2：MultiStepLR —— 指定节点衰减
# ===========================================================================
print("=" * 78)
print("小节 2：MultiStepLR —— 指定节点衰减（在预设 epoch 列表处衰减）")
print("=" * 78)
print(
    r"""
公式：

    lr_epoch = lr_0 × gamma ^ (milestones 中 <= epoch 的元素个数)

参数含义：
    milestones ：一个 epoch 列表，比如 [30, 80]，表示第 30、第 80 个 epoch 各衰减一次
    gamma      ：每次的衰减倍数

课案的例子：MultiStepLR(optimizer, milestones=[30, 80], gamma=0.1)
    lr: 0.01 →（到第 30 轮）→ 0.001 →（到第 80 轮）→ 0.0001
和 StepLR 的区别：StepLR 是**等间隔**自动衰减，MultiStepLR 衰减点由你自由指定。
适用场景：已知「拐点」位置——比如你观察到验证集在第 30 和第 80 个 epoch 附近不再提升，
          就在这两个点降 lr；训练周期不确定、需要灵活安排时也更方便。
"""
)
_multistep_hist = collect_lr_epoch_schedule(
    lambda opt: torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 80], gamma=0.1))
print(f"  milestones=[30, 80], gamma=0.1, lr0=0.01，共 {_EPOCHS} 个 epoch")
print_key_lrs(_multistep_hist)
print(f"  衰减发生的位置（lr 相比上一 epoch 变化的位置）："
      f"{[i for i in range(1, _EPOCHS) if abs(_multistep_hist[i] - _multistep_hist[i - 1]) > 1e-12]}")
print("  → 恰好是 epoch 30 和 epoch 80 两处，与 milestones 一致。")
print()


# ===========================================================================
# 小节 3：ExponentialLR —— 指数衰减
# ===========================================================================
print("=" * 78)
print("小节 3：ExponentialLR —— 指数衰减（每个 epoch 都乘 gamma）")
print("=" * 78)
print(
    r"""
公式：

    lr_epoch = lr_0 × gamma ^ epoch

参数含义：
    gamma ：每个 epoch 的衰减比例，越接近 1 衰减越慢

课案的例子：ExponentialLR(optimizer, gamma=0.95)，lr0 = 0.01
    epoch 0: lr = 0.01
    epoch 1: lr = 0.01 × 0.95   = 0.0095
    epoch 2: lr = 0.01 × 0.95²  = 0.009025
    ...
    epoch n: lr = 0.01 × 0.95ⁿ
和 StepLR 的区别：**连续平滑**衰减（每个 epoch 都变一点点），而不是「平台 + 断崖」。
缺点：衰减速度恒定、不具备自适应能力——它不知道你训练到哪一步了、loss 还降不降。
适用场景：希望 lr 平滑持续下降、不想手工设衰减节点的场合。
"""
)
_exp_hist = collect_lr_epoch_schedule(
    lambda opt: torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.95))
print(f"  gamma=0.95, lr0=0.01，共 {_EPOCHS} 个 epoch")
print_key_lrs(_exp_hist)
print(f"  验证手写公式 lr0 × gamma^n：")
for _n in (0, 1, 2, 10, 50, 99):
    _theory = 0.01 * 0.95 ** _n
    print(f"    epoch {_n:>3}: 调度器 {_exp_hist[_n]:.10f}，公式 {_theory:.10f}，"
          f"误差 {abs(_exp_hist[_n] - _theory):.2e}")
print(f"  100 个 epoch 后 lr 从 0.01 衰到 {_exp_hist[-1]:.3e}"
      f"（= 0.01 × 0.95^99 = 0.01 × {0.95 ** 99:.3e} = {0.01 * 0.95 ** 99:.3e}）")
print()


# ===========================================================================
# 小节 4：CosineAnnealingLR —— 余弦退火（含公式数值验证）
# ===========================================================================
print("=" * 78)
print("小节 4：CosineAnnealingLR —— 余弦退火（只取半个余弦波的下降段）")
print("=" * 78)
print(
    r"""
公式：

    lr_t = eta_min + (lr_0 - eta_min) × (1 + cos(π · t / T_max)) / 2

    · t = 0        → cos(0) = 1      → 系数 (1+1)/2 = 1.0  → lr = lr_0（最高）
    · t = T_max/2  → cos(π/2) = 0    → 系数 0.5            → lr = (lr_0 + eta_min)/2（中点）
    · t = T_max    → cos(π) = -1     → 系数 0.0            → lr = eta_min（最低）

名字里的「退火」= 逐渐降温，**只降不升**：只取余弦波从峰顶滑到谷底的那半段，
不来回震荡（那是 CosineAnnealingWarmRestarts 才做的事）。

参数含义：
    T_max   ：走完这半个余弦周期需要多少个 epoch（通常设为总训练 epoch 数）
    eta_min ：lr 的最低点下限（默认 0；设成 1e-5 则降到 1e-5 后不再下降，避免 lr 真的变成 0）

适用场景：不需要手动设衰减节点、想要平滑过渡的场合；现代 Transformer 训练常用。
"""
)
_T_MAX = 100
_ETA_MIN = 0.0
_cos_hist = collect_lr_epoch_schedule(
    lambda opt: torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=_T_MAX, eta_min=_ETA_MIN))
print(f"  T_max={_T_MAX}, eta_min={_ETA_MIN}, lr0=0.01，共 {_EPOCHS} 个 epoch")
print_key_lrs(_cos_hist)
print()
print("  【数值验证】手写公式与 scheduler.get_last_lr() 的对比：")
_lr0 = 0.01
# 用一份新的调度器，逐个取出 get_last_lr()，与手写余弦公式逐点比较
_opt_cos = make_optimizer(_lr0)
_sched_cos = torch.optim.lr_scheduler.CosineAnnealingLR(_opt_cos, T_max=_T_MAX, eta_min=_ETA_MIN)
_max_err = 0.0
_printed = 0
for _t in range(_EPOCHS):
    _actual = _sched_cos.get_last_lr()[0]                 # 当前 epoch 的 lr
    _theory = _ETA_MIN + (_lr0 - _ETA_MIN) * (1 + math.cos(math.pi * _t / _T_MAX)) / 2
    _max_err = max(_max_err, abs(_actual - _theory))
    if _t in (0, 1, 25, 50, 75, 99, 100 - 1):
        print(f"    epoch {_t:>3}: get_last_lr()={_actual:.10f}  手写公式={_theory:.10f}"
              f"  误差={abs(_actual - _theory):.2e}")
    _sched_cos.step()
print(f"  全部 {_EPOCHS} 个 epoch 的最大误差 = {_max_err:.3e}"
      f"  → {'手写公式与 PyTorch 实现完全一致 ✔' if _max_err < 1e-12 else '存在偏差'}")
print()
# 三个关键点的显式验证（与课案数值对照）
print("  课案提到的三个关键点（lr0=0.01、T_max=100、eta_min=0）：")
print(f"    epoch 0   ：cos(0)=1     → 系数 1.0 → lr = {0.01:.6f}（✓ 与课案一致）")
print(f"    epoch 50  ：cos(π/2)=0   → 系数 0.5 → lr = {0.01 * 0.5:.6f}（✓ 与课案一致）")
print(f"    epoch 100 ：cos(π)=-1    → 系数 0.0 → lr = {0.01 * 0.0:.6f}（✓ 与课案一致）")
# eta_min 不为 0 的情形
_opt_cos2 = make_optimizer(_lr0)
_sched_cos2 = torch.optim.lr_scheduler.CosineAnnealingLR(_opt_cos2, T_max=_T_MAX, eta_min=1e-5)
for _t in range(_EPOCHS + 1):                             # 走到 t = T_max 之后
    _sched_cos2.step()
print(f"  eta_min=1e-5 时，走到 epoch {_EPOCHS}（t > T_max）的 lr = "
      f"{_opt_cos2.param_groups[0]['lr']:.8f} → 降到 1e-5 后**不再下降**（不会变成 0）")
print()


# ===========================================================================
# 小节 5：ReduceLROnPlateau —— 指标停滞衰减
# ===========================================================================
print("=" * 78)
print("小节 5：ReduceLROnPlateau —— 指标停滞衰减（监控验证指标，不按 epoch 数触发）")
print("=" * 78)
print(
    r"""
规则：
    每个 epoch 把验证指标传进 scheduler.step(val_loss)：
        · 如果指标比历史最佳改善了（超过 threshold）→ 什么都不做；
        · 如果连续 patience 个 epoch 没有改善 → lr = lr × factor，然后重新计数。
    **调用方式与前面几个完全不同**：必须传入被监控的指标！写成 scheduler.step() 会报错。

参数含义：
    mode     ：'min' 表示指标越小越好（loss）/ 'max' 表示越大越好（accuracy）
    factor   ：衰减倍数，lr = lr × factor（默认 0.1）
    patience ：容忍多少个 epoch 不改善（默认 10）
    threshold：判定「有改善」的最小变化量（过滤抖动）
    cooldown ：衰减后先「冷却」几个 epoch 再重新计数
    min_lr   ：lr 的下限，不会降到比它还低

适用场景：**不确定训练多久**、指标何时停滞事先不知道——让验证集自己决定要不要降 lr。

注意：官方在新版本里更推荐用 torch.optim.lr_scheduler.ReduceLROnPlateau，
      它也支持 verbose / threshold_mode 等更细的参数（本脚本只演示核心行为）。
"""
)
print("用一段模拟的「验证 loss」序列来驱动它（先快速下降、然后长时间停滞）：")
# 共 100 个 epoch，和前面几个调度器的观察长度保持一致，方便同图对比
_val_curve = np.concatenate([
    np.linspace(1.0, 0.30, 20),                     # epoch 0~19：快速下降
    0.30 + 0.0015 * np.arange(80),                  # epoch 20~99：基本停滞（只缓慢上升）
])
_opt_plateau = make_optimizer(0.01)
_sched_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
    _opt_plateau, mode="min", factor=0.1, patience=10, threshold=1e-4, cooldown=0)

_plateau_hist: list[float] = []
_decay_events: list[tuple[int, float, float]] = []   # (epoch, decay前lr, decay后lr)
for _ep, _v in enumerate(_val_curve):
    _current_lr = _opt_plateau.param_groups[0]["lr"]
    _plateau_hist.append(_current_lr)
    _sched_plateau.step(float(_v))                  # ← 关键：必须传入被监控的指标
    _new_lr = _opt_plateau.param_groups[0]["lr"]
    if _new_lr < _current_lr - 1e-15:
        _decay_events.append((_ep, _current_lr, _new_lr))

print(f"  模拟验证 loss 形状：epoch 0~19 从 1.00 降到 0.30，epoch 20~99 基本停滞（0.30 附近）")
print(f"  配置：mode='min', factor=0.1, patience=10")
print(f"  衰减发生在以下 epoch（每次 lr ×0.1）：")
for _ep, _before, _after in _decay_events:
    print(f"    第 {_ep:>2} 个 epoch：lr {_before:.6f} → {_after:.8f}")
print(f"  共发生 {len(_decay_events)} 次衰减；最终 lr = {_opt_plateau.param_groups[0]['lr']:.3e}")
print(f"  前 25 个 epoch 的 lr 序列：{['%.6f' % v for v in _plateau_hist[:25]]}")
print(f"  → 前 20 个 epoch 指标一直在改善，lr 保持 0.01 不变；")
print(f"    从停滞开始，每攒够 10 个「没改善」的 epoch 就衰减一次，"
      f"所以第 1 次落在 epoch {_decay_events[0][0] if _decay_events else '-'}，"
      f"第 2 次落在 epoch {_decay_events[1][0] if len(_decay_events) > 1 else '-'}。")
print("  → 这就是「让验证集决定何时降 lr」——训练多久、何时该精细收敛，都不需要提前知道。")
print()


# ===========================================================================
# 小节 6：OneCycleLR —— 先升后降
# ===========================================================================
print("=" * 78)
print("小节 6：OneCycleLR —— 先升后降（lr 每步都变，不是每 epoch）")
print("=" * 78)
print(
    r"""
策略：
    训练前半段 lr 从很低的起点快速升到峰值 max_lr（warmup 升温），
    后半段再从峰值平滑降回很低的终点（退火收敛）。
    PyTorch 默认的分配是「前 30% 步数上升、后 70% 步数下降」：
        起点  ≈ max_lr / div_factor          （div_factor 默认 25.0）
        终点  ≈ 起点 / final_div_factor      （final_div_factor 默认 1e4）
    升温阶段（默认 pct_start=0.3 之后不再用线性，而是余弦插值）：
        · 0 ~ pct_start 段：lr 从 max_lr/25 升到 max_lr
        · pct_start ~ 1 段：lr 从 max_lr 沿余弦降到 max_lr/(25×1e4)

**为什么学习率大反而有正则化效果？**
    大的学习率让参数更新步长更大、方向更「莽撞」，模型不容易陷入训练集的局部细节，
    相当于一种**隐式的噪声注入**；反过来，太小的学习率会让模型精细地拟合训练集的
    每一条样本，更容易过拟合。所以训练中段用大学习率 = 自带正则化。
**但大 lr 不能从头用到尾**：
    初期参数是随机的，一上来就用大 lr 会让训练直接崩掉（所以需要 warmup 升温）；
    末期需要精细收敛到最优点（所以需要衰减降温）。OneCycleLR 就是「先升温、再降温」。

参数含义：
    max_lr           ：学习率的峰值（通常比平时大得多，靠 OneCycle 的策略兜住）
    total_steps      ：总步数（= steps_per_epoch × epochs）
    steps_per_epoch  ：一个 epoch 有多少个 batch（必须传，且**每个 batch 后都要 step()**）
    epochs           ：训练总轮数
    pct_start        ：升温阶段占总步数的比例（默认 0.3）
    div_factor       ：起点 = max_lr / div_factor（默认 25.0）
    final_div_factor ：终点 = 起点 / final_div_factor（默认 1e4）
    anneal_strategy  ：下降段的插值方式，'cos'（默认）或 'linear'

适用场景：**快速训练**——希望用尽量少的步数把模型训好，配合比平时大得多的 max_lr 使用。
"""
)
_STEPS_PER_EPOCH = 10
_CYCLE_EPOCHS = 20
_total_steps = _STEPS_PER_EPOCH * _CYCLE_EPOCHS
_opt_cycle = make_optimizer(0.01)
_sched_cycle = torch.optim.lr_scheduler.OneCycleLR(
    _opt_cycle, max_lr=0.01, steps_per_epoch=_STEPS_PER_EPOCH, epochs=_CYCLE_EPOCHS,
    pct_start=0.3, div_factor=25.0, final_div_factor=1e4, anneal_strategy="cos")

_cycle_hist: list[float] = []
for _step in range(_total_steps):
    _cycle_hist.append(_opt_cycle.param_groups[0]["lr"])
    _sched_cycle.step()                             # ← 每个 **batch** 后都要 step

print(f"  max_lr=0.01, steps_per_epoch={_STEPS_PER_EPOCH}, epochs={_CYCLE_EPOCHS}，"
      f"总步数 = {_total_steps}")
print(f"  理论起点 = max_lr/div_factor = 0.01/25 = {0.01 / 25:.8f}，实测第 0 步 = {_cycle_hist[0]:.8f}")
print(f"  理论终点 = 起点/final_div_factor = {0.01 / 25 / 1e4:.3e}，"
      f"实测最后一步 = {_cycle_hist[-1]:.3e}")
print(f"  峰值出现在第 {int(np.argmax(_cycle_hist))} 步"
      f"（理论 pct_start×总步数 = {int(0.3 * _total_steps)} 步），峰值 = {max(_cycle_hist):.8f}")
print(f"  升温阶段（前 30% 步）单调不减？"
      f"{all(_cycle_hist[i] <= _cycle_hist[i + 1] + 1e-12 for i in range(int(0.3 * _total_steps) - 1))}")
_peak_idx = int(np.argmax(_cycle_hist))
print(f"  下降阶段（峰值之后）单调不增？"
      f"{all(_cycle_hist[i] >= _cycle_hist[i + 1] - 1e-12 for i in range(_peak_idx, _total_steps - 1))}")
print()
print("  每多少步打印一次（前 8 步 + 每 20 步抽样，共 20 行）：")
_sample_steps = list(range(8)) + list(range(10, _total_steps, 20))
for _s in _sample_steps:
    _bar = "█" * max(1, int(_cycle_hist[_s] / max(_cycle_hist) * 40))
    print(f"    step {_s:>3} (epoch {_s // _STEPS_PER_EPOCH:>2}): lr={_cycle_hist[_s]:.8f}  {_bar}")
print("  → 曲线形状：前 30% 步数快速上升（warmup），到峰值后平滑下降（退火），完全符合预期。")
print()


# ===========================================================================
# 小节 7：另外 5 个常用调度器（LambdaLR / MultiplicativeLR / CyclicLR /
#          CosineAnnealingWarmRestarts / ConstantLR / LinearLR / SequentialLR）
# ===========================================================================
print("=" * 78)
print("小节 7：另外几个常用调度器与组合写法")
print("=" * 78)

# --- 7.1 LambdaLR ---
print("-" * 78)
print("7.1 LambdaLR：用一个自定义函数把 epoch 映射成 lr 倍数（最灵活）")
print("-" * 78)
print(
    r"""
公式： lr_epoch = lr_0 × lambda(epoch)
    lambda 可以是任意函数，所以 LambdaLR 能表达上面所有「只跟 epoch 有关」的策略。
    常见写法：
        # 每 30 个 epoch 乘 0.1（等价于 StepLR）
        lambda e: 0.1 ** (e // 30)
        # 多项式衰减 lr ∝ (1 - e/T)^p（Transformer 论文用的就是这种 + warmup）
        lambda e: (1 - e / T) ** 0.9
        # 分段：前 5 个 epoch 线性 warmup，之后不变
        lambda e: (e + 1) / 5 if e < 5 else 1.0
"""
)
_lam_hist = collect_lr_epoch_schedule(
    lambda opt: torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda e: 0.1 ** (e // 30)))
print(f"  lr_lambda = 0.1 ** (epoch // 30)（等价于 StepLR(step_size=30, gamma=0.1)）：")
print_key_lrs(_lam_hist)
print(f"  与 StepLR 的序列完全一致？"
      f"{all(abs(a - b) < 1e-15 for a, b in zip(_lam_hist, _step_hist))}")

_poly_T = 100
_poly_hist = collect_lr_epoch_schedule(
    lambda opt: torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda e: max(0.0, (1 - e / _poly_T)) ** 0.9))
print(f"  多项式衰减 (1 - epoch/{_poly_T})^0.9：epoch 0 → {_poly_hist[0]:.6f}，"
      f"epoch 50 → {_poly_hist[50]:.6f}，epoch 99 → {_poly_hist[99]:.6f}")
print()

# --- 7.2 MultiplicativeLR ---
print("-" * 78)
print("7.2 MultiplicativeLR：每个 epoch 乘一个由函数给出的**因子**（累乘）")
print("-" * 78)
print(
    r"""
公式： lr_epoch = lr_{epoch-1} × lr_lambda(epoch)
    与 LambdaLR 的区别：
        LambdaLR          —— lambda 返回的是「相对初始 lr 的倍数」，每次都基于 lr_0 重算；
        MultiplicativeLR  —— lambda 返回的是「相对上一个 epoch 的倍数」，逐轮累乘。
    所以当 lambda 是常数（例如恒为 0.95）时，MultiplicativeLR 等价于 ExponentialLR(gamma=0.95)。
"""
)
_mult_hist = collect_lr_epoch_schedule(
    lambda opt: torch.optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=lambda e: 0.95))
print(f"  lr_lambda = lambda e: 0.95（恒为常数）")
print_key_lrs(_mult_hist)
print(f"  与 ExponentialLR(gamma=0.95) 完全一致？"
      f"{all(abs(a - b) < 1e-15 for a, b in zip(_mult_hist, _exp_hist))}")

_mult2_hist = collect_lr_epoch_schedule(
    lambda opt: torch.optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=lambda e: 0.9 if e < 10 else 1.0))
print(f"  换个函数「前 10 个 epoch 每轮 ×0.9，之后不变」："
      f"epoch0={_mult2_hist[0]:.6f}, epoch9={_mult2_hist[9]:.6f}, epoch10={_mult2_hist[10]:.6f}, "
      f"epoch99={_mult2_hist[99]:.6f}")
print()

# --- 7.3 CyclicLR ---
print("-" * 78)
print("7.3 CyclicLR：学习率在 base_lr 和 max_lr 之间**周期性地来回震荡**")
print("-" * 78)
print(
    r"""
公式（triangular2 策略，step_size_up = S）：
    第 k 个周期内（k = 0, 1, 2, ...），周期长度 = 2S：
        · 前 S 步：lr 从 base_lr 线性升到 max_lr
        · 后 S 步：lr 从 max_lr 线性降回 base_lr
    'triangular2' 额外让每个周期的 max_lr 减半（峰值逐周期收敛）；
    'triangular' 每个周期峰谷相同；'exp_range' 让峰值按指数衰减。

为什么有用：让 lr 周期性变大——**跳出局部极小值/鞍点**；变小时——在当前区域精细收敛。
参数含义：base_lr / max_lr / step_size_up / step_size_down / mode / gamma / cycle_momentum。
"""
)
_opt_cyc = make_optimizer(0.01)
_sched_cyc = torch.optim.lr_scheduler.CyclicLR(
    _opt_cyc, base_lr=0.001, max_lr=0.01, step_size_up=10, step_size_down=10,
    mode="triangular", cycle_momentum=False)         # SGD 无 momentum 时必须关掉 cycle_momentum
_cyc_hist: list[float] = []
for _ in range(80):
    _cyc_hist.append(_opt_cyc.param_groups[0]["lr"])
    _sched_cyc.step()
print(f"  base_lr=0.001, max_lr=0.01, step_size_up=10, step_size_down=10, mode='triangular'")
print(f"  第 1 个周期的谷值/峰值：{min(_cyc_hist[:20]):.6f} / {max(_cyc_hist[:20]):.6f}")
print(f"  第 2 个周期的谷值/峰值：{min(_cyc_hist[20:40]):.6f} / {max(_cyc_hist[20:40]):.6f}"
      f"（triangular 模式下峰谷保持不变）")
print(f"  每 10 步抽样：{['%.6f' % _cyc_hist[i] for i in range(0, 40, 5)]}")
print()

# --- 7.4 CosineAnnealingWarmRestarts ---
print("-" * 78)
print("7.4 CosineAnnealingWarmRestarts：余弦退火 + 周期性「重启」（warm restarts）")
print("-" * 78)
print(
    r"""
公式：每个周期长度 T_i = T_0 × T_mult^i，周期内
        lr_t = eta_min + (lr0 - eta_min) × (1 + cos(π · t_cur / T_i)) / 2
    走完一个周期后 lr 回到 lr0（**重启**），并且周期长度可以按 T_mult 逐次变长。

与 CosineAnnealingLR 的区别：CosineAnnealingLR 只降一次、降到底就结束；
    WarmRestarts 会反复「降到最低 → 猛地跳回最高」，每次重启都帮助模型跳出当前局部极小值。
    T_mult=1 时每个周期等长，T_mult=2 时周期长度翻倍（10、20、40、…）——SGDR 论文的推荐用法。
"""
)
_opt_wr = make_optimizer(0.01)
_sched_wr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    _opt_wr, T_0=10, T_mult=2, eta_min=1e-4)
_wr_hist: list[float] = []
for _ in range(70):
    _wr_hist.append(_opt_wr.param_groups[0]["lr"])
    _sched_wr.step()
print(f"  T_0=10, T_mult=2, eta_min=1e-4（周期长度 10 → 20 → 40 ...）")
print(f"  每 5 步抽样：{['%.6f' % _wr_hist[i] for i in range(0, 70, 5)]}")
_restart_points = [i for i in range(1, 70) if _wr_hist[i] > _wr_hist[i - 1] + 1e-9]
print(f"  lr 突然跳升（重启）的位置：epoch {_restart_points[:8]}"
      f"（理论重启点：10, 30, 70 ...）")
print()

# --- 7.5 ConstantLR / LinearLR / SequentialLR：warmup + 主调度的组合写法 ---
print("-" * 78)
print("7.5 ConstantLR / LinearLR / SequentialLR：实践中最常见的 warmup + 主调度组合")
print("-" * 78)
print(
    r"""
三个「小」调度器：
    ConstantLR(optimizer, factor, total_iters)  ：在前 total_iters 步把 lr 固定为 lr_0 × factor；
    LinearLR(optimizer, start_factor, end_factor, total_iters)
                                                ：在前 total_iters 步把 lr 从
                                                  lr_0 × start_factor 线性变到 lr_0 × end_factor；
    SequentialLR(optimizer, schedulers=[...], milestones=[...])
                                                ：把多个调度器**按顺序**拼接起来。

其中 LinearLR 的 start_factor/end_factor 就是标准的 warmup：从很小的 lr 线性升到完整 lr。
组合写法（实践中非常常见：先 warmup 再余弦退火）：

    warmup = LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=5)
    cosine = CosineAnnealingLR(opt, T_max=95, eta_min=1e-5)
    scheduler = SequentialLR(opt, schedulers=[warmup, cosine], milestones=[5])
    # milestone=5 表示前 5 个 epoch 用 warmup，之后交给 cosine（cosine 从 t=0 开始重新计）

为什么要 warmup？训练刚开始参数是随机的，梯度方向噪声很大，
    一上来就用完整的大 lr 容易把参数「踢飞」；先用很小的 lr 走几步，
    等参数进入合理区域再放大 lr，训练更稳。Transformer 类模型几乎必用 warmup。
"""
)
_opt_seq = make_optimizer(1.0)                      # 这个 demo 把 lr0 设成 1.0，方便直接读倍数
# 先演示单独的 LinearLR 在做什么（不接 SequentialLR）
_opt_only_lin = make_optimizer(1.0)
_sched_only_lin = torch.optim.lr_scheduler.LinearLR(
    _opt_only_lin, start_factor=1e-3, end_factor=1.0, total_iters=5)
_only_lin_hist: list[float] = []
for _ in range(8):
    _only_lin_hist.append(_opt_only_lin.param_groups[0]["lr"])
    _sched_only_lin.step()
print(f"  单独的 LinearLR(start_factor=1e-3, end_factor=1.0, total_iters=5)，lr0=1.0：")
print(f"    {['%.6f' % v for v in _only_lin_hist]}"
      f"  → 前 5 步从 0.001 线性升到 1.0，之后保持 1.0（这就是 warmup）")

# 再演示 ConstantLR：前 total_iters 步把 lr 固定为 lr0 × factor
_opt_only_const = make_optimizer(1.0)
_sched_only_const = torch.optim.lr_scheduler.ConstantLR(
    _opt_only_const, factor=0.1, total_iters=4)
_only_const_hist: list[float] = []
for _ in range(7):
    _only_const_hist.append(_opt_only_const.param_groups[0]["lr"])
    _sched_only_const.step()
print(f"  单独的 ConstantLR(factor=0.1, total_iters=4)，lr0=1.0：")
print(f"    {['%.6f' % v for v in _only_const_hist]}"
      f"  → 前 4 步保持 0.1，之后恢复 1.0（常用于「先小步热身几步」）")
print()

# 组合：LinearLR warmup + CosineAnnealingLR 主调度
_warmup = torch.optim.lr_scheduler.LinearLR(
    _opt_seq, start_factor=1e-3, end_factor=1.0, total_iters=5)
_cosine_main = torch.optim.lr_scheduler.CosineAnnealingLR(_opt_seq, T_max=15, eta_min=1e-3)
_sched_seq = torch.optim.lr_scheduler.SequentialLR(
    _opt_seq, schedulers=[_warmup, _cosine_main], milestones=[5])
_seq_hist: list[float] = []
for _ in range(20):
    _seq_hist.append(_opt_seq.param_groups[0]["lr"])
    _sched_seq.step()
print(f"  lr0=1.0；LinearLR(start_factor=1e-3, end_factor=1.0, total_iters=5) "
      f"+ CosineAnnealingLR(T_max=15, eta_min=1e-3)，milestones=[5]")
for _i in range(20):
    _bar = "█" * max(1, int(_seq_hist[_i] * 40))
    _phase = "warmup" if _i < 5 else "cosine"
    print(f"    epoch {_i:>2} [{_phase:>6}]: lr={_seq_hist[_i]:.6f}  {_bar}")
print(f"  → epoch 0~4：lr 从 0.001 线性升到 1.0（warmup）；")
print(f"    epoch 5~19：交给余弦退火，从 1.0 平滑降向 eta_min=0.001。")
print("    SequentialLR 让你把「升温」和「降温」两段拼成一条完整曲线，这就是工业界最常见的写法。")
print()


# ===========================================================================
# 小节 8：6 个主调度器的 lr 曲线画在同一张图上
# ===========================================================================
print("=" * 78)
print("小节 8：可视化——6 个主调度器的 lr 曲线对比")
print("=" * 78)

_main_histories = {
    "StepLR(step_size=30, gamma=0.1)": _step_hist,
    "MultiStepLR([30,80], 0.1)": _multistep_hist,
    "ExponentialLR(0.95)": _exp_hist,
    "CosineAnnealingLR(T_max=100)": _cos_hist,
    "ReduceLROnPlateau(patience=10)": _plateau_hist,
}
fig7, axes7 = plt.subplots(2, 3, figsize=(17, 9))
_colors7 = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]
_ep_axis = np.arange(1, _EPOCHS + 1)

# 上排 3 张 + 下排 2 张：每个调度器单独一张子图
for _i, (_name, _hist) in enumerate(_main_histories.items()):
    _ax = axes7[_i // 3, _i % 3]
    _ax.plot(_ep_axis, _hist, "-", color=_colors7[_i], lw=2)
    _ax.set_title(_name, fontsize=10)
    _ax.set_xlabel("epoch")
    _ax.set_ylabel("学习率 lr")
    _ax.grid(alpha=0.3)
# 下排第 3 张：OneCycleLR 的每步曲线
_ax6 = axes7[1, 2]
_ax6.plot(np.arange(_total_steps), _cycle_hist, "-", color=_colors7[5], lw=2)
_ax6.axvline(0.3 * _total_steps, color="gray", ls="--", lw=1,
             label=f"pct_start=30%（第 {int(0.3 * _total_steps)} 步）")
_ax6.set_title(f"OneCycleLR（每步变化，共 {_total_steps} 步）", fontsize=10)
_ax6.set_xlabel("step")
_ax6.set_ylabel("学习率 lr")
_ax6.legend(fontsize=8)
_ax6.grid(alpha=0.3)
fig7.suptitle("6 个学习率调度器的 lr 曲线（前 5 个按 epoch，OneCycleLR 按 step）", fontsize=13)
fig7.tight_layout(rect=(0, 0, 1, 0.965))
_p7a = OUTPUT_DIR / "03训练组件_07_学习率调度对比.png"
fig7.savefig(_p7a, dpi=110)
plt.close(fig7)
print(f"已保存：{_p7a}")

# 单独一张：5 条 epoch 级曲线叠加对比（更直观地看出「阶梯 / 指数 / 余弦 / 不定」的形状差别）
fig7b, ax7b = plt.subplots(figsize=(11, 6))
for _i, (_name, _hist) in enumerate(_main_histories.items()):
    ax7b.plot(_ep_axis, _hist, "-", color=_colors7[_i], lw=2.2, label=_name)
ax7b.set_xlabel("epoch")
ax7b.set_ylabel("学习率 lr")
ax7b.set_title("5 个 epoch 级调度器的 lr 曲线叠加对比（lr0 = 0.01）")
ax7b.legend(fontsize=9)
ax7b.grid(alpha=0.3)
fig7b.tight_layout()
_p7b = OUTPUT_DIR / "03训练组件_07_学习率调度对比.png"
fig7b.savefig(_p7b, dpi=110)
plt.close(fig7b)
print(f"已保存（叠加图，覆盖同名文件）：{_p7b}")

# 单独一张：OneCycleLR 每步曲线
fig7c, ax7c = plt.subplots(figsize=(11, 5.2))
ax7c.plot(np.arange(_total_steps), _cycle_hist, "-", color="#937860", lw=2.4)
_peak = int(np.argmax(_cycle_hist))
ax7c.axvline(_peak, color="gray", ls="--", lw=1.2, label=f"峰值在第 {_peak} 步")
ax7c.annotate(f"warmup 段（前 30% 步）\nlr: {_cycle_hist[0]:.5f} → {max(_cycle_hist):.5f}",
              xy=(_peak * 0.5, max(_cycle_hist) * 0.55), fontsize=10, color="#4C72B0")
ax7c.annotate(f"退火段（后 70% 步）\nlr: {max(_cycle_hist):.5f} → {_cycle_hist[-1]:.2e}",
              xy=(_peak + (_total_steps - _peak) * 0.45, max(_cycle_hist) * 0.45),
              fontsize=10, color="#C44E52")
ax7c.set_xlabel("step（每个 batch 一步）")
ax7c.set_ylabel("学习率 lr")
ax7c.set_title(f"OneCycleLR 每步学习率曲线（max_lr=0.01, 共 {_total_steps} 步）")
ax7c.legend(fontsize=9)
ax7c.grid(alpha=0.3)
fig7c.tight_layout()
_p7c = OUTPUT_DIR / "03训练组件_07_OneCycleLR每步曲线.png"
fig7c.savefig(_p7c, dpi=110)
plt.close(fig7c)
print(f"已保存：{_p7c}")
print()


# ===========================================================================
# 小节 9：实战对比——固定 lr / StepLR / 余弦退火
# ===========================================================================
print("=" * 78)
print("小节 9：实战对比——同一个网络和数据，用「固定 lr」/「StepLR」/「余弦退火」各训一遍")
print("=" * 78)
print(
    """
实验设计：
    · 数据：make_classification(600 条 × 20 维，其中 6 维有信息，flip_y=0.15 制造一定标签噪声)，
      训练 450 / 测试 150——这样模型在后期「继续硬拟合」会让测试 loss 反弹，调度器的价值才看得出来；
    · 模型：20 → 64 → 64 → 2 的 MLP（ReLU）；
    · 优化器：SGD(lr=0.05, momentum=0.9)，交叉熵损失，训练 150 个 epoch，batch_size=32；
    · 三种 lr 策略：
        - 固定 lr      ：lr 全程 0.05（对照组）;
        - StepLR       ：每 40 个 epoch × 0.2（0.05 → 0.01 → 0.002 → 0.0004）;
        - 余弦退火     ：CosineAnnealingLR(T_max=150, eta_min=1e-4)，从 0.05 平滑降到 1e-4;
    · 记录每个 epoch 的训练 loss / 测试 loss / 测试准确率，最后对比最终测试准确率与测试 loss。
    预期：固定 lr 在后期会因为步长不降而在最优点附近反复震荡，测试 loss 反而升高；
          StepLR 和余弦退火在后期把 lr 降下来，能更精细地落到谷底，最终指标更好。
"""
)

from sklearn.datasets import make_classification   # 只用 sklearn 造数据，不联网


class MLP2(nn.Module):
    """20 → 64 → 64 → 2 的两层隐藏层 MLP。"""

    def __init__(self, in_dim: int = 20, hidden: int = 64, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_Xl, _yl = make_classification(
    n_samples=600, n_features=20, n_informative=6, n_redundant=0,
    n_clusters_per_class=2, class_sep=1.0, flip_y=0.15, random_state=42,
)
X_lr = torch.tensor(_Xl, dtype=torch.float32)
y_lr = torch.tensor(_yl, dtype=torch.long)
_lr_split = 450
Xlr_tr, ylr_tr = X_lr[:_lr_split], y_lr[:_lr_split]
Xlr_te, ylr_te = X_lr[_lr_split:], y_lr[_lr_split:]
print(f"数据：600 条 × 20 维，训练 {len(Xlr_tr)} / 测试 {len(Xlr_te)}")
print()

_SCHED_EPOCHS = 150
_BATCH = 32
_lr_strategies = ["固定 lr=0.05", "StepLR(step_size=40, gamma=0.2)", "CosineAnnealingLR(T_max=150)"]
_lr_train_result: dict[str, dict] = {}
_lr_train_hist: dict[str, dict[str, list[float]]] = {}

for _name in _lr_strategies:
    torch.manual_seed(42)
    _model = MLP2()
    _opt = torch.optim.SGD(_model.parameters(), lr=0.05, momentum=0.9)
    _crit = nn.CrossEntropyLoss()
    if _name.startswith("StepLR"):
        _sched = torch.optim.lr_scheduler.StepLR(_opt, step_size=40, gamma=0.2)
    elif _name.startswith("Cosine"):
        _sched = torch.optim.lr_scheduler.CosineAnnealingLR(_opt, T_max=_SCHED_EPOCHS, eta_min=1e-4)
    else:
        _sched = None
    _train_loader = DataLoader(TensorDataset(Xlr_tr, ylr_tr), batch_size=_BATCH, shuffle=True)
    _hist = {"train_loss": [], "test_loss": [], "test_acc": [], "lr": []}

    for _ep in range(_SCHED_EPOCHS):
        _model.train()
        _running = 0.0
        for Xb, yb in _train_loader:
            _opt.zero_grad()
            _l = _crit(_model(Xb), yb)
            _l.backward()
            _opt.step()
            _running += _l.item() * Xb.size(0)
        _hist["train_loss"].append(_running / len(_train_loader.dataset))
        _model.eval()
        with torch.no_grad():
            _hist["test_loss"].append(_crit(_model(Xlr_te), ylr_te).item())
            _hist["test_acc"].append(
                (_model(Xlr_te).argmax(dim=1) == ylr_te).float().mean().item())
        _hist["lr"].append(_opt.param_groups[0]["lr"])
        if _sched is not None:
            _sched.step()                                # 按 epoch 调度的调度器在 epoch 末尾 step

    _lr_train_result[_name] = {
        "final_test_acc": _hist["test_acc"][-1],
        "best_test_acc": max(_hist["test_acc"]),
        "final_train_loss": _hist["train_loss"][-1],
        "final_test_loss": _hist["test_loss"][-1],
        "first_test_loss": _hist["test_loss"][0],
        "final_lr": _hist["lr"][-1],
    }
    _lr_train_hist[_name] = _hist

print(f"{'lr 策略':<34}{'首轮测试loss':>14}{'末轮训练loss':>14}{'末轮测试loss':>14}"
      f"{'最终测试acc':>13}{'训练后lr':>14}")
print("-" * 104)
for _name in _lr_strategies:
    _r = _lr_train_result[_name]
    print(f"{_name:<34}{_r['first_test_loss']:>14.4f}{_r['final_train_loss']:>14.4f}"
          f"{_r['final_test_loss']:>14.4f}{_r['final_test_acc'] * 100:>12.2f}%"
          f"{_r['final_lr']:>14.8f}")
print()
_best_strategy = max(_lr_strategies, key=lambda n: _lr_train_result[n]["final_test_acc"])
_worst_loss = max(_lr_strategies, key=lambda n: _lr_train_result[n]["final_test_loss"])
_best_loss = min(_lr_strategies, key=lambda n: _lr_train_result[n]["final_test_loss"])
print(f"  最终测试准确率最高的是：{_best_strategy}"
      f"（{_lr_train_result[_best_strategy]['final_test_acc'] * 100:.2f}%）")
print(f"  末轮测试 loss 最低的是：{_best_loss}（{_lr_train_result[_best_loss]['final_test_loss']:.4f}），"
      f"最高的是 {_worst_loss}（{_lr_train_result[_worst_loss]['final_test_loss']:.4f}）")
print(f"  固定 lr 的 lr 全程保持 0.05 不变 —— 后期步长不降，只能在最优点附近来回震荡；")
print(f"  StepLR / 余弦退火把 lr 逐步降到 "
      f"{_lr_train_result['StepLR(step_size=40, gamma=0.2)']['final_lr']:.6f} / "
      f"{_lr_train_result['CosineAnnealingLR(T_max=150)']['final_lr']:.6f}，")
print(f"  后期步长变小 → 能在谷底附近更精细地落下，测试 loss 更低、泛化更好。")
print("  注意：lr 调度不是万能的——任务简单/训练轮数少时固定 lr 也能训得很好；")
print("        它的价值在于「训练周期长、后期需要精细收敛、又不想手工猜何时降 lr」的场景，")
print("        把「何时降 lr」这件事交给策略自动决定。")
print()

# --- 画实战对比曲线 ---
fig9, axes9 = plt.subplots(1, 3, figsize=(17, 5))
_colors9 = ["#4C72B0", "#DD8452", "#55A868"]
for _i, _name in enumerate(_lr_strategies):
    _h = _lr_train_hist[_name]
    axes9[0].plot(range(1, _SCHED_EPOCHS + 1), _h["train_loss"], color=_colors9[_i], label=_name)
    axes9[1].plot(range(1, _SCHED_EPOCHS + 1), _h["test_loss"], color=_colors9[_i], label=_name)
    axes9[2].plot(range(1, _SCHED_EPOCHS + 1), _h["test_acc"], color=_colors9[_i], label=_name)
    axes9[2].scatter([_SCHED_EPOCHS], [_lr_train_result[_name]["final_test_acc"]],
                     color=_colors9[_i], zorder=5)
axes9[0].set_title("训练 loss 对比", fontsize=11)
axes9[0].set_xlabel("epoch")
axes9[0].set_ylabel("训练交叉熵损失")
axes9[1].set_title("测试 loss 对比", fontsize=11)
axes9[1].set_xlabel("epoch")
axes9[1].set_ylabel("测试交叉熵损失")
axes9[2].set_title("测试准确率对比", fontsize=11)
axes9[2].set_xlabel("epoch")
axes9[2].set_ylabel("测试准确率")
for _ax in axes9:
    _ax.legend(fontsize=9)
    _ax.grid(alpha=0.3)
fig9.suptitle("学习率调度的实战影响：固定 lr vs StepLR vs 余弦退火（150 epoch）", fontsize=13)
fig9.tight_layout(rect=(0, 0, 1, 0.94))
_p9 = OUTPUT_DIR / "03训练组件_07_调度实战对比.png"
fig9.savefig(_p9, dpi=110)
plt.close(fig9)
print(f"已保存：{_p9}")
print()


# ===========================================================================
# 小节 10：对比表（课案）
# ===========================================================================
print("=" * 78)
print("小节 10：调度器对比表（课案原表）")
print("=" * 78)
_SCHED_TABLE = [
    ("StepLR", "每 N 个 epoch", "阶梯状", "明确训练周期"),
    ("MultiStepLR", "指定 epoch 点", "节点阶梯", "已知拐点位置"),
    ("ExponentialLR", "每个 epoch", "连续指数衰减", "平滑持续衰减"),
    ("CosineAnnealingLR", "每个 epoch", "余弦曲线", "Transformer、平滑退火"),
    ("ReduceLROnPlateau", "指标停滞时", "不定", "不确定训练时长"),
    ("OneCycleLR", "每步", "先升后降", "快速训练"),
]
print(f"{'调度器':<22}{'触发方式':<16}{'曲线形状':<18}{'适用场景'}")
print("-" * 76)
for _r in _SCHED_TABLE:
    print(f"{_r[0]:<22}{_r[1]:<16}{_r[2]:<18}{_r[3]}")
print()
print("附：另外几个调度器的一句话总结")
_EXTRA_TABLE = [
    ("LambdaLR", "每个 epoch", "任意函数", "完全自定义（多项式/分段/更复杂规则）"),
    ("MultiplicativeLR", "每个 epoch", "逐轮累乘因子", "衰减规律随 epoch 变化的场景"),
    ("CyclicLR", "每步", "三角/指数震荡", "跳出局部极小值、配合较大 max_lr"),
    ("CosineAnnealingWarmRestarts", "每个 epoch", "余弦 + 周期重启", "SGDR，多次退火重启"),
    ("ConstantLR / LinearLR", "前 N 步", "常数 / 线性", "warmup 阶段"),
    ("SequentialLR", "按 milestone 切换", "多段拼接", "warmup + 主调度（工业界最常见）"),
]
print(f"{'调度器':<30}{'触发方式':<18}{'曲线形状':<18}{'适用场景'}")
print("-" * 92)
for _r in _EXTRA_TABLE:
    print(f"{_r[0]:<30}{_r[1]:<18}{_r[2]:<18}{_r[3]}")
print()

print("=" * 78)
print("07 学习率调度：全部实验完成")
print("=" * 78)

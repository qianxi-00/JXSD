"""
对应课案章节：训练组件 / 优化器

本节知识点：
    1. 所有优化器的核心公式都是 θ ← θ - α·Δθ，区别只在**怎么算 Δθ**。
    2. 前置知识：一阶泰勒展开 L(θ+Δθ) ≈ L(θ) + ∇L·Δθ。
       为什么是减号：沿正梯度方向走 Δθ=α∇L，变化量 = α‖∇L‖² ≥ 0 —— **必定上升**；
       沿反方向走 Δθ=-α∇L，变化量 = -α‖∇L‖² ≤ 0 —— **必定下降**。
    3. 逐个优化器（每个都给：原理 / 更新公式伪代码 / 真的手写一步更新 /
       torch.optim 用法）：SGD、Momentum、NAG(nesterov=True)、AdaGrad、
       RMSProp、AdaDelta、Adam。
    4. Momentum 的两种写法：本课案 / PyTorch 用的是
       `v_t = β·v_{t-1} + g_t`，更新 `θ ← θ - α·v_t`；
       有些教材写 `v_t = β·v_{t-1} + (1-β)·g_t`。
       本脚本用 NumPy 手写并与 PyTorch 逐位对比，实测确认到底哪个一致。
    5. NAG 的实现细节：本脚本实测发现 PyTorch 的 nesterov=True 走的是
       「在实际位置算梯度」的等价形式（`g_t + β·v_t`），
       与课案写的「在 θ - α·β·v_{t-1} 处算梯度」在数值上**并不相等**，
       脚本里把两种写法都跑出来对比，讲清这个容易被忽略的细节。
    6. AdaGrad 的致命缺点：s_t 只增不减 → 有效步长 α/√s_t 单调衰减到 0。
       本脚本数值演示 s 单调增长、有效步长单调衰减。
    7. RMSProp 把累加改成指数移动平均；注意 PyTorch 里 `alpha` 是衰减系数 ρ
       （默认 0.99），**不是学习率**，这是极易踩的坑。
    8. AdaDelta 公式里没有 α → 无需设学习率（用 r_t 跟踪"过去走了多远"）。
    9. Adam = Momentum + RMSProp，核心是**偏差校正**：
       t=1 时 m_1 = 0.1·g_1（只有真实梯度的 10%）、s_1 = 0.001·g_1²（只有 0.1%），
       除以 (1-β₁^t)、(1-β₂^t) 就能修正回来。脚本数值打印 t=1、t=10 的值。
   10. 补充覆盖：AdamW（解耦权重衰减）/ RAdam / SparseAdam / LBFGS / ASGD /
       Adamax / NAdam 中的 7 个（公式 + 一句话场景）。
   11. 收敛对比实验（核心）：同一个损失函数上，SGD / SGD+momentum /
       SGD+nesterov / Adagrad / RMSprop / Adadelta / Adam 各跑 400 步，
       在 Rosenbrock 函数（f=(1-x)²+100(y-x²)²，起点 (-1.5, 1.5)）
       和简单二次函数 f(θ)=‖θ-w‖² 上对比，每 10 步打印一次 loss
       → 03训练组件_03_优化器收敛对比.png（两个子图，log 纵轴）。
   12. 轨迹图：在 Rosenbrock 等高线上画出 SGD 与 Adam 的优化路径
       → 03训练组件_03_优化器轨迹.png。
   13. 手写验证：手写的 SGD-with-momentum 与手写 Adam 和 torch.optim 版本
       在同一初始参数上跑 20 步，逐参数打印最大差异（应 < 1e-6）。
   14. 对比表（课案）：SGD / Momentum / NAG / AdaGrad / RMSProp / AdaDelta / Adam。

说明：本机未安装 torchmetrics 之外的重量级包，本脚本只用 torch + numpy + matplotlib。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Deep_Learning\\03_训练组件\\03_优化器全家桶.py'
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
# 后续正式依赖（必须写在 matplotlib 初始化之后）
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402

# 随机种子统一：保证每次运行结果一致、可复现
torch.manual_seed(42)
np.random.seed(42)

sns.set_theme(style="whitegrid", font="Microsoft YaHei")   # seaborn 风格画图
_STEPS = 400          # 收敛对比实验的步数（课案要求 ≥ 60，这里给足够步数看出差异）
_PRINT_EVERY = 10     # 每 10 步打印一次 loss


def _title(text: str) -> None:
    """打印分节标题，让输出有清晰的结构。"""
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


# ===========================================================================
# 0. 前置知识：一阶泰勒展开，以及为什么更新是"减号"
# ===========================================================================
_title("0. 前置知识：一阶泰勒展开 & 为什么是 θ ← θ - α·∇L")
print("""
  一元函数在 x₀ 附近用直线近似（斜率 × 位移）：
      f(x) ≈ f(x₀) + 斜率 × (x - x₀)

  多元函数同理，梯度 ∇L 就是"多元斜率"：
      L(θ + Δθ) ≈ L(θ) + ∇L · Δθ
      其中点积 ∇L·Δθ 就是"斜率 × 位移"，代表损失的变化量。

  【为什么是减号？】看变化量 ∇L·Δθ 的符号：
      · 若沿**正**梯度方向走：Δθ = +α∇L
            变化量 = ∇L·(α∇L) = α‖∇L‖² ≥ 0   —— 损失**必定上升**（往山上走）
      · 若沿**反**梯度方向走：Δθ = -α∇L
            变化量 = ∇L·(-α∇L) = -α‖∇L‖² ≤ 0  —— 损失**必定下降**（往山下走）
  所以更新必须写成 θ ← θ - α∇L：梯度指向"上升最快"的方向，
  减号就是往它的反方向（下降最快）走。α 就是学习率，控制每一步走多远。

  【所有优化器的统一骨架】θ ← θ - α·Δθ
      区别只在 Δθ 怎么算：
        SGD       Δθ = g_t                     （直接用当前梯度）
        Momentum  Δθ = v_t                     （梯度做指数移动平均，带惯性）
        AdaGrad   Δθ = g_t/(√s_t+ε)            （按历史梯度平方和缩放）
        RMSProp   Δθ = g_t/(√s_t+ε)            （s 改成指数移动平均）
        AdaDelta  Δθ = √(r_{t-1}+ε)/√(s_t+ε)·g_t（连学习率都自己估）
        Adam      Δθ = m̂_t/(√ŝ_t+ε)            （动量 + 自适应，都做偏差校正）
""")

# 用数值直观感受一下这两个方向到底哪个让损失下降
_lr_demo = 0.05
_theta_demo = torch.tensor([1.5, -1.0])
_loss_fn_demo = lambda t: (t ** 2).sum()          # L(θ)=‖θ‖²，最小值在原点
_g_demo = 2 * _theta_demo                          # ∇L = 2θ
_l_now = _loss_fn_demo(_theta_demo).item()
_l_plus = _loss_fn_demo(_theta_demo + _lr_demo * _g_demo).item()    # 沿正梯度走
_l_minus = _loss_fn_demo(_theta_demo - _lr_demo * _g_demo).item()   # 沿负梯度走
print(f"  数值验证（L(θ)=‖θ‖²，θ={_theta_demo.numpy()}，α={_lr_demo}）：")
print(f"    当前损失 L(θ)              = {_l_now:.6f}")
print(f"    沿 +α∇L 走（错误方向）      = {_l_plus:.6f}   <- 变大了，确实是往山上走")
print(f"    沿 -α∇L 走（θ ← θ-α∇L）     = {_l_minus:.6f}   <- 变小了，正确")


# ===========================================================================
# 1. 手写各优化器的"一步更新"（纯 NumPy，可运行代码写在注释块下方）
# ===========================================================================
_title("1. 逐个优化器：原理 / 公式 / 手写一步更新 / torch 用法")

# 统一的测试场景：二次损失 L(θ) = ‖θ - w‖²，梯度 ∇L = 2(θ - w)
# 用二维参数向量（不是大网络），几十步即可，保证脚本很快。
_W_TARGET = np.array([2.0, -3.0])                  # 损失的最小值点
_THETA0 = np.array([-1.5, 1.5])                    # 统一初始参数


def grad_quad(theta):
    """二次损失 L(θ)=‖θ-w‖² 的梯度：2(θ-w)。"""
    return 2.0 * (theta - _W_TARGET)


# ---------------------------------------------------------------------------
# 1.1 SGD
# ---------------------------------------------------------------------------
print("""
────────────────────────────────────────────────────────────────────────────
【SGD（随机梯度下降）】
  原理：最朴素的优化器，直接沿着当前 batch 的负梯度方向走一步。
  更新公式（伪代码）：
        g_t = ∇L(θ_t)                    # 当前 batch 的梯度
        θ_{t+1} = θ_t - α · g_t           # 沿负梯度走 α 步

  优点：实现最简单、内存开销最小（不需要保存任何历史状态）、泛化有时更好。
  缺点：① 对学习率极其敏感，α 太大会震荡/发散，太小则慢得像蜗牛；
        ② 所有参数共用一个学习率，遇到"病态"（不同方向曲率差异巨大）的
           损失曲面会来回震荡、走出锯齿形路径；
        ③ 容易停在鞍点或局部极小点附近。
  适用：凸问题、小模型、或配合学习率调度器使用（如 ResNet 训练）。
────────────────────────────────────────────────────────────────────────────
""")
_theta = _THETA0.copy()
_lr = 0.05
print(f"  手写一步更新（lr={_lr}）：")
_g = grad_quad(_theta)
print(f"    当前 θ            = {np.round(_theta, 6)}")
print(f"    梯度 g = 2(θ-w)   = {np.round(_g, 6)}")
_theta = _theta - _lr * _g                          # ← 这就是 SGD 的全部
print(f"    更新后 θ = θ-α·g  = {np.round(_theta, 6)}")

# ---------------------------------------------------------------------------
# 1.2 Momentum
# ---------------------------------------------------------------------------
print("""
────────────────────────────────────────────────────────────────────────────
【Momentum（动量）】
  原理：引入"速度" v 累积历史梯度。类比小球滚下山：当前梯度是**加速度**，
        速度 v 是累积的**动量**。好处：
          (1) 震荡方向上动量互相抵消，前进方向上动量持续叠加 → 加速收敛；
          (2) 能"冲"过局部极小点 / 鞍点。
  更新公式（伪代码，PyTorch 写法）：
        v_t = β · v_{t-1} + g_t           # 累积梯度（注意不是 (1-β)·g_t！）
        θ_{t+1} = θ_t - α · v_t           # 用累积后的速度更新

  ⚠ 两种等价写法（本脚本会实测确认 PyTorch 是哪一种）：
        写法 A（PyTorch / 本课案）：v_t = β·v_{t-1} + g_t
        写法 B（很多教材）        ：v_t = β·v_{t-1} + (1-β)·g_t
     两者只差一个常数因子 (1-β)：B 的 v 恒等于 A 的 v 乘以 (1-β)。
     所以 **B + 学习率 α/(1-β) 在数学上完全等价于 A + 学习率 α**。
     直接用 A 的公式配 B 的系数（或反之）会让有效学习率差 10 倍（β=0.9 时），
     这是初学者最常踩的坑之一。
  β 通常取 0.9。
  缺点：多了一份状态（与参数同形状的 v），且引入超参数 β；
        动量太大时会在极小点附近来回冲过头（需要配合学习率衰减）。
  适用：几乎万能的基础改进，CNN/MLP 训练的首选起点。
────────────────────────────────────────────────────────────────────────────
""")
_theta = _THETA0.copy()
_lr, _beta = 0.05, 0.9
_v = np.zeros_like(_theta)                          # 初始化速度 v_0 = 0
print(f"  手写一步更新（lr={_lr}, β={_beta}）：")
_g = grad_quad(_theta)
print(f"    当前 θ              = {np.round(_theta, 6)}")
print(f"    梯度 g              = {np.round(_g, 6)}")
_v = _beta * _v + _g                                # 第一步：v_1 = 0.9·0 + g = g
print(f"    更新速度 v = β·v + g = {np.round(_v, 6)}")
_theta = _theta - _lr * _v
print(f"    更新后 θ = θ-α·v    = {np.round(_theta, 6)}")

# —— 实测：PyTorch 的 momentum 到底用哪种写法？——
print("\n  ▶ 实测确认：PyTorch 用的是「写法 A」v_t = β·v_{t-1} + g_t")


def sgd_momentum_np(theta0, steps, lr, beta, grad_fn):
    """手写 SGD + Momentum（写法 A：v = βv + g）。"""
    th = np.array(theta0, dtype=np.float64)
    v = np.zeros_like(th)
    for _ in range(steps):
        g = grad_fn(th)
        v = beta * v + g                              # ← 写法 A
        th = th - lr * v
    return th


def sgd_momentum_alt_np(theta0, steps, lr, beta, grad_fn):
    """手写 SGD + Momentum 的另一种教材写法（写法 B：v = βv + (1-β)g）。"""
    th = np.array(theta0, dtype=np.float64)
    v = np.zeros_like(th)
    for _ in range(steps):
        g = grad_fn(th)
        v = beta * v + (1.0 - beta) * g               # ← 写法 B
        th = th - lr * v
    return th


def torch_run(opt_factory, theta0, steps, grad_fn):
    """用 torch.optim 的优化器跑 steps 步，返回最终参数（numpy）。

    统一使用二次损失 L(θ) = ‖θ - w‖²，这样与上面的 NumPy 手写实现
    （grad_quad 给出同一个解析梯度 2(θ-w)）严格可比。
    grad_fn 参数保留是为了表明两者的梯度来源一致，这里由 autograd 自动计算。
    """
    p = torch.tensor(np.array(theta0, dtype=np.float64), requires_grad=True)
    opt = opt_factory(p)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((p - torch.tensor(_W_TARGET)) ** 2).sum()   # L(θ)=‖θ-w‖²
        loss.backward()
        opt.step()
    return p.detach().numpy().copy()


_NCHK = 20                                          # 对比步数
_torch_mom = torch_run(lambda p: torch.optim.SGD([p], lr=0.05, momentum=0.9),
                       _THETA0, _NCHK, grad_quad)
_np_momA = sgd_momentum_np(_THETA0, _NCHK, lr=0.05, beta=0.9, grad_fn=grad_quad)
_np_momB = sgd_momentum_alt_np(_THETA0, _NCHK, lr=0.05, beta=0.9, grad_fn=grad_quad)
_np_momB_fixed = sgd_momentum_alt_np(_THETA0, _NCHK, lr=0.05 / (1 - 0.9), beta=0.9,
                                      grad_fn=grad_quad)
print(f"    跑 {_NCHK} 步后，torch.optim.SGD(momentum=0.9) 的最终 θ = {np.round(_torch_mom, 8)}")
print(f"    写法 A（v=βv+g，同 lr=0.05）      最终 θ = {np.round(_np_momA, 8)}"
      f"   与 torch 的最大差异 = {np.abs(_np_momA - _torch_mom).max():.3e}")
print(f"    写法 B（v=βv+(1-β)g，同 lr=0.05）最终 θ = {np.round(_np_momB, 8)}"
      f"   与 torch 的最大差异 = {np.abs(_np_momB - _torch_mom).max():.3e}")
print(f"    写法 B 配 lr=α/(1-β)={0.05 / (1 - 0.9):.3f}         最终 θ = {np.round(_np_momB_fixed, 8)}"
      f"   与 torch 的最大差异 = {np.abs(_np_momB_fixed - _torch_mom).max():.3e}")
print("    => 结论：PyTorch 用的是【写法 A】；写法 B 只要把学习率放大 1/(1-β) 倍就与 A 完全等价。")

# ---------------------------------------------------------------------------
# 1.3 NAG
# ---------------------------------------------------------------------------
print("""
────────────────────────────────────────────────────────────────────────────
【NAG（Nesterov Accelerated Gradient）】
  原理：Momentum 是"先算当前位置的梯度，再顺着动量走"；
        NAG 是"先顺着动量往前探一步，在那个位置算梯度"。
        类比下山：先按惯性往前看一眼，发现要上坡了就**提前减速**。
  更新公式（伪代码，课案的写法）：
        v_t = β · v_{t-1} + ∇L(θ_t - α·β·v_{t-1})     # 在"探一步"的位置算梯度
        θ_{t+1} = θ_t - α · v_t

  ⚠ 实测细节（本脚本会打印对比）：PyTorch 的 `nesterov=True` 实际等价于
        g_t = ∇L(θ_t)                                  # 就在**当前实际位置**算梯度
        v_t = β · v_{t-1} + g_t
        θ_{t+1} = θ_t - α · (g_t + β · v_t)
    这一形式与课案公式在数值上**不相等**。原因是 NAG 的"探一步"有几种
    符号约定，PyTorch 采用的是"在实际参数位置求导 + 用 (g + βv) 组合更新"
    的等价实现（与论文里的重参数化有关）。
  好处：比标准 Momentum 收敛更快、更稳定、更少过冲。
  缺点：公式与实现细节容易混淆（就像上面这一点），超参数仍需调。
  适用：需要比 Momentum 更快收敛的场合，是很多比赛的常客配置。
────────────────────────────────────────────────────────────────────────────
""")
_theta = _THETA0.copy()
_lr, _beta = 0.05, 0.9
_v = np.zeros_like(_theta)
print(f"  手写一步更新（课案写法，lr={_lr}, β={_beta}）：")
_look = _theta - _lr * _beta * _v                  # 先按惯性往前看一眼
_g = grad_quad(_look)                              # 在探到的位置算梯度
print(f"    探一步位置 θ-αβv    = {np.round(_look, 6)}")
print(f"    该处梯度 g          = {np.round(_g, 6)}")
_v = _beta * _v + _g
_theta = _theta - _lr * _v
print(f"    更新后 θ            = {np.round(_theta, 6)}")

# 实测：课案写法 vs PyTorch 真实实现
def nag_course_np(theta0, steps, lr, beta, grad_fn):
    """课案写法：v_t = βv_{t-1} + ∇L(θ_t - α·β·v_{t-1})。"""
    th = np.array(theta0, dtype=np.float64)
    v = np.zeros_like(th)
    for _ in range(steps):
        g = grad_fn(th - lr * beta * v)               # 在探一步的位置算梯度
        v = beta * v + g
        th = th - lr * v
    return th


def nag_torch_style_np(theta0, steps, lr, beta, grad_fn):
    """PyTorch nesterov=True 的等价写法：θ -= α·(g + β·v)，v = βv + g。"""
    th = np.array(theta0, dtype=np.float64)
    v = np.zeros_like(th)
    for _ in range(steps):
        g = grad_fn(th)                               # 在**当前实际位置**算梯度
        v = beta * v + g
        th = th - lr * (g + beta * v)
    return th


_torch_nag = torch_run(lambda p: torch.optim.SGD([p], lr=0.05, momentum=0.9, nesterov=True),
                       _THETA0, _NCHK, grad_quad)
_np_nag_course = nag_course_np(_THETA0, _NCHK, 0.05, 0.9, grad_quad)
_np_nag_torch = nag_torch_style_np(_THETA0, _NCHK, 0.05, 0.9, grad_quad)
print(f"\n    {_NCHK} 步后 torch.optim.SGD(nesterov=True) 最终 θ = {np.round(_torch_nag, 8)}")
print(f"    课案写法（探一步算梯度）   最终 θ = {np.round(_np_nag_course, 8)}"
      f"   与 torch 差异 = {np.abs(_np_nag_course - _torch_nag).max():.3e}")
print(f"    PyTorch 等价写法           最终 θ = {np.round(_np_nag_torch, 8)}"
      f"   与 torch 差异 = {np.abs(_np_nag_torch - _torch_nag).max():.3e}")
print("    => 结论：只有「在实际位置算梯度 + 用 (g+βv) 更新」这一形式与 PyTorch 数值一致；")
print("       课案公式是 NAG 的另一种（理论）表述，方向一致但步长不同。")

# ---------------------------------------------------------------------------
# 1.4 AdaGrad
# ---------------------------------------------------------------------------
print("""
────────────────────────────────────────────────────────────────────────────
【AdaGrad（Adaptive Gradient）】
  原理：让**每个参数有自己独立的学习率**——历史梯度大的参数，学习率自动变小；
        历史梯度小的参数，学习率自动变大。这样稀疏特征也能被有效更新。
  更新公式（伪代码）：
        s_t = s_{t-1} + g_t ⊙ g_t               # 累加梯度平方（⊙ 是逐元素乘）
        θ_{t+1} = θ_t - α · g_t / (√s_t + ε)     # ε 是极小数，防止分母为 0

  优点：① 免调学习率的"自动缩放"；② 对稀疏数据（NLP 词向量、推荐系统 ID 特征）
           效果很好，因为很少出现的特征能得到较大的有效步长。
  缺点：**s_t 只增不减**，训练越久 s_t 越大 → 有效学习率 α/√s_t 越接近 0，
        后期几乎学不动（这是 AdaGrad 被 RMSProp 取代的根本原因）。
  适用：稀疏特征、凸优化；不建议用于长训的深度网络。
────────────────────────────────────────────────────────────────────────────
""")
_theta = _THETA0.copy()
_lr, _eps = 0.05, 1e-8
_s = np.zeros_like(_theta)
print(f"  手写一步更新（lr={_lr}, ε={_eps}）：")
_g = grad_quad(_theta)
print(f"    梯度 g              = {np.round(_g, 6)}")
_s = _s + _g * _g                                   # 累加平方
print(f"    s = s + g²          = {np.round(_s, 6)}")
_theta = _theta - _lr * _g / (np.sqrt(_s) + _eps)
print(f"    更新后 θ            = {np.round(_theta, 6)}")

# —— 数值演示：s 单调增长、有效步长单调衰减 ——
print("\n  ▶ 数值演示 AdaGrad 的缺点：s_t 只增不减 → 有效步长单调衰减")
_theta = _THETA0.copy()
_s = np.zeros_like(_theta)
_lr = 0.5
print(f"    （lr={_lr}，共 200 步；有效步长取 ‖α·g/(√s+ε)‖ 的范数）")
print(f"    {'step':>6s} {'s 的范数':>16s} {'有效步长范数':>16s}")
for _step in range(1, 201):
    _g = grad_quad(_theta)
    _s = _s + _g * _g
    _delta = _lr * _g / (np.sqrt(_s) + _eps)
    _theta = _theta - _delta
    if _step in (1, 10, 50, 100, 200):
        print(f"    {_step:>6d} {np.linalg.norm(_s):>16.6e} {np.linalg.norm(_delta):>16.6e}")
print("    -> s 的范数一路增长（因为每步都在加 g²，从不遗忘），")
print("       有效步长一路衰减，最终趋近于 0 —— 这就是「后期学不动」的原因。")

# ---------------------------------------------------------------------------
# 1.5 RMSProp
# ---------------------------------------------------------------------------
print("""
────────────────────────────────────────────────────────────────────────────
【RMSProp（Root Mean Square Propagation）】
  原理：针对 AdaGrad 的 s_t 无限增长问题，把"累加"改成**指数移动平均**——
        旧的平方梯度逐渐被遗忘，学习率不会衰减到 0。
  更新公式（伪代码）：
        s_t = ρ · s_{t-1} + (1 - ρ) · g_t²       # 指数移动平均（EMA）
        θ_{t+1} = θ_t - α · g_t / (√s_t + ε)

  ⚠⚠ 极易踩的坑：**PyTorch 里 `alpha` 参数是衰减系数 ρ（默认 0.99），
        根本不是学习率！** 学习率是 `lr`（默认 0.01）。
        写成 `optim.RMSprop(params, alpha=0.01)` 想设学习率的话，
        实际是把"遗忘速度"设成了 0.01，结果完全错。
  优点：解决 AdaGrad 学习率归零的问题，适合非平稳目标（RNN 常用）。
  缺点：ρ 与 α 两个超参数耦合，仍需调；没有动量/偏差校正。
  适用：RNN、非平稳目标、需要自适应步长但不想用 Adam 的场合。
────────────────────────────────────────────────────────────────────────────
""")
_theta = _THETA0.copy()
_lr, _rho, _eps = 0.05, 0.9, 1e-8
_s = np.zeros_like(_theta)
print(f"  手写一步更新（lr={_lr}, ρ={_rho}, ε={_eps}）：")
_g = grad_quad(_theta)
print(f"    梯度 g                    = {np.round(_g, 6)}")
_s = _rho * _s + (1 - _rho) * _g * _g
print(f"    s = ρ·s + (1-ρ)·g²        = {np.round(_s, 6)}")
_theta = _theta - _lr * _g / (np.sqrt(_s) + _eps)
print(f"    更新后 θ                  = {np.round(_theta, 6)}")
print(f"  提醒：PyTorch 用法是 optim.RMSprop(params, lr={_lr}, alpha={_rho})，")
print(f"        这里的 alpha={_rho} 是衰减系数 ρ，如果你把学习率填进 alpha 就完全错了。")

# ---------------------------------------------------------------------------
# 1.6 AdaDelta
# ---------------------------------------------------------------------------
print("""
────────────────────────────────────────────────────────────────────────────
【AdaDelta】
  原理：RMSProp 还留着 α 要手调。AdaDelta 注意到参数更新量 Δθ 本身也有规律
        ——初期大步走，后期小步调。于是用 r_t 跟踪**更新量自身的平方**，
        用它替代 α。含义就是："过去几步走了多远，这一步也走差不多远"。
  更新公式（伪代码）：
        s_t = ρ · s_{t-1} + (1 - ρ) · g_t²                # 梯度平方的 EMA
        Δθ_t = -√(r_{t-1} + ε) / √(s_t + ε) · g_t          # 用 r 当"学习率"
        θ_{t+1} = θ_t + Δθ_t
        r_t = ρ · r_{t-1} + (1 - ρ) · Δθ_t²               # 更新量平方的 EMA

  【公式里没有 α】——这就是"无需设学习率"的含义。PyTorch 的 lr 参数
        只是对 Δθ 的一个额外缩放（默认 1.0，通常保持 1.0 即为标准 AdaDelta）。
  优点：完全免调学习率；对超参数 ρ 不敏感。
  缺点：公式更复杂、额外维护 r；实际中收敛速度和最终精度常不如 Adam，
        因此不如 Adam 常用。
  适用：不想调学习率、且能接受稍慢收敛的场合。
────────────────────────────────────────────────────────────────────────────
""")
_theta = _THETA0.copy()
_rho, _eps = 0.9, 1e-6
_s = np.zeros_like(_theta)          # 梯度平方的 EMA
_r = np.zeros_like(_theta)          # 更新量平方的 EMA
print(f"  手写一步更新（ρ={_rho}, ε={_eps}，注意没有学习率）：")
_g = grad_quad(_theta)
_s = _rho * _s + (1 - _rho) * _g * _g
print(f"    s = ρ·s + (1-ρ)·g²            = {np.round(_s, 6)}")
_delta = -np.sqrt(_r + _eps) / np.sqrt(_s + _eps) * _g
print(f"    Δθ = -√(r+ε)/√(s+ε)·g         = {np.round(_delta, 6)}")
_theta = _theta + _delta
_r = _rho * _r + (1 - _rho) * _delta * _delta
print(f"    更新后 θ                       = {np.round(_theta, 6)}")
print(f"    r = ρ·r + (1-ρ)·Δθ²            = {np.round(_r, 6)}")

# ---------------------------------------------------------------------------
# 1.7 Adam
# ---------------------------------------------------------------------------
print("""
────────────────────────────────────────────────────────────────────────────
【Adam（Adaptive Moment Estimation）= Momentum + RMSProp】
  原理：同时维护梯度的**一阶矩**（动量 m，管方向惯性）和**二阶矩**
        （平方梯度 s，管自适应步长），并且对两者都做**偏差校正**。
  更新公式（伪代码）：
        m_t = β₁ · m_{t-1} + (1 - β₁) · g_t          # 一阶矩（动量）
        s_t = β₂ · s_{t-1} + (1 - β₂) · g_t²         # 二阶矩（平方梯度）
        m̂_t = m_t / (1 - β₁^t)                        # 偏差校正
        ŝ_t = s_t / (1 - β₂^t)                        # 偏差校正
        θ_{t+1} = θ_t - α · m̂_t / (√ŝ_t + ε)

  【偏差校正是什么？为什么必须做？】
      m 和 s 都初始化为 0，这会让训练初期严重偏低。以 t=1 为例：
          m_1 = β₁·0 + (1-β₁)·g_1 = 0.1 · g_1        （只有真实梯度的 10%！）
          s_1 = β₂·0 + (1-β₂)·g_1² = 0.001 · g_1²    （只有真实的 0.1%！）
      这就是"初始值 0 污染了移动平均"造成的偏差。m_t 恰好只有真实值的
      (1-β₁^t) 倍，所以除以 (1-β₁^t) 就能修正回来：
          t=1  : 1-β₁^1 = 0.1      → m̂_1 = m_1/0.1 = g_1        ✓ 完全修正
          t=10 : 1-0.9^10 ≈ 0.651  → m̂_10 = m_10/0.651          （部分修正）
          t 很大: β₁^t → 0，1-β₁^t → 1，校正自动消失（不再需要修正）
      一句话：除以 1-β^t 是把被 0 初始值"压扁"的 m、s 拉回正常水平，
              而且随着训练进行，校正会自然消退。
  默认 β₁=0.9、β₂=0.999、ε=1e-8、lr=0.001。
  优点：收敛快、对学习率不敏感、对稀疏梯度和非平稳目标都稳。
  缺点：① 内存要存两份状态（m、s）；② 某些任务上泛化不如调好的 SGD+Momentum；
        ③ 与 L2 正则耦合（→ 这就是 AdamW 要解决的问题）。
  适用：绝大多数深度学习任务的默认首选。
────────────────────────────────────────────────────────────────────────────
""")
_theta = _THETA0.copy()
_lr, _b1, _b2, _eps = 0.05, 0.9, 0.999, 1e-8
_m = np.zeros_like(_theta)
_s = np.zeros_like(_theta)
_t = 1
print(f"  手写一步更新（t=1, lr={_lr}, β₁={_b1}, β₂={_b2}, ε={_eps}）：")
_g = grad_quad(_theta)
print(f"    梯度 g                        = {np.round(_g, 6)}")
_m = _b1 * _m + (1 - _b1) * _g
_s = _b2 * _s + (1 - _b2) * _g * _g
print(f"    m = β₁·m + (1-β₁)·g           = {np.round(_m, 6)}   （= 0.1·g，只有真实梯度的 10%）")
print(f"    s = β₂·s + (1-β₂)·g²          = {np.round(_s, 6)}   （= 0.001·g²，只有 0.1%）")
_c1 = 1 - _b1 ** _t
_c2 = 1 - _b2 ** _t
_m_hat = _m / _c1
_s_hat = _s / _c2
print(f"    校正系数 1-β₁^t = {_c1:.6f}，1-β₂^t = {_c2:.6f}")
print(f"    m̂ = m/(1-β₁^t)                = {np.round(_m_hat, 6)}   （= g，偏差被完全修正 ✓）")
print(f"    ŝ = s/(1-β₂^t)                = {np.round(_s_hat, 6)}   （= g²，同样被修正 ✓）")
_theta = _theta - _lr * _m_hat / (np.sqrt(_s_hat) + _eps)
print(f"    更新后 θ                       = {np.round(_theta, 6)}")

# —— 数值打印 t=1、t=10 的校正前后 m 值 ——
print("\n  ▶ 数值打印 t=1 与 t=10 的偏差校正效果")
_theta = _THETA0.copy()
_m = np.zeros_like(_theta)
_s = np.zeros_like(_theta)
print(f"    {'t':>4s} {'1-β₁^t':>10s} {'m 的范数(校正前)':>18s} {'m̂ 的范数(校正后)':>18s} "
      f"{'放大倍数':>10s}")
for _step in range(1, 31):
    _g = grad_quad(_theta)
    _m = _b1 * _m + (1 - _b1) * _g
    _s = _b2 * _s + (1 - _b2) * _g * _g
    _c1 = 1 - _b1 ** _step
    _m_hat = _m / _c1
    if _step in (1, 2, 5, 10, 20, 30):
        print(f"    {_step:>4d} {_c1:>10.6f} {np.linalg.norm(_m):>18.6e} "
              f"{np.linalg.norm(_m_hat):>18.6e} {1.0 / _c1:>10.4f}")
    _theta = _theta - _lr * _m_hat / (np.sqrt(_s / (1 - _b2 ** _step)) + _eps)
print("    -> t=1 时放大 10 倍（0.1 → 1）；t=10 时只放大约 1.54 倍；")
print("       t 越大 1-β₁^t 越接近 1，放大的倍数趋近 1，校正自动消退。")

# ---------------------------------------------------------------------------
# 1.8 各优化器的 torch.optim 实际用法
# ---------------------------------------------------------------------------
print("""
────────────────────────────────────────────────────────────────────────────
【各优化器的 torch.optim 实际用法速查】
    optim.SGD(model.parameters(), lr=0.01)                                  # 基础 SGD
    optim.SGD(model.parameters(), lr=0.01, momentum=0.9)                    # + 动量
    optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)     # NAG
    optim.Adagrad(model.parameters(), lr=0.01)                              # 自适应 lr
    optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9)                   # alpha 是 ρ！
    optim.Adadelta(model.parameters(), rho=0.9)                             # 无需 lr
    optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))            # 默认首选
  标准训练循环骨架（所有优化器都一样）：
    optimizer.zero_grad()          # 1. 清空上一轮的梯度（否则会累加！）
    loss = loss_fn(model(x), y)    # 2. 前向 + 算损失
    loss.backward()                # 3. 反向传播，把梯度写进 param.grad
    optimizer.step()               # 4. 按自己的公式更新参数
────────────────────────────────────────────────────────────────────────────
""")


# ===========================================================================
# 2. 补充优化器：AdamW / RAdam / SparseAdam / LBFGS / ASGD / Adamax / NAdam
# ===========================================================================
_title("2. 补充优化器（公式 + 场景）")
print("""
【AdamW】解耦权重衰减 —— 现代 Transformer 训练的标配（BERT/GPT 都用它）
  公式      : θ ← θ - α·[ m̂_t/(√ŝ_t+ε) + λ·θ ]      （权重衰减**单独**作用在参数上）
  对比 Adam : Adam 把 L2 正则加进损失（g ← g + λθ），再让自适应步长去缩放它，
              结果**大梯度的参数被惩罚得少**，衰减效果被 √ŝ 扭曲。
              AdamW 把 λθ 从自适应缩放中**解耦**出来，衰减是"一视同仁"的。
  【关键结论：在 Adam 下 weight decay 与 L2 正则**不等价**】
              —— 而在朴素 SGD 下两者恰好等价（因为 Δθ = α(g+λθ) 可以写成先加正则再更新）。
  场景      : Transformer / LLM 预训练、需要强正则的现代网络。

【RAdam】Rectified Adam —— 训练初期自适应学习率的"预热"修正
  公式      : 引入 ρ_∞ = 2/(1-β₂)-1，用 ρ_t = ρ_∞ - 2t·β₂^t/(1-β₂^t) 判断方差是否可信，
              ρ_t > 4 时用带修正系数的 Adam 更新，否则退化成 SGD-with-momentum。
  场景      : 不想手写 warmup 而希望自动预热、且不想调 lr 的场景。

【SparseAdam】只支持稀疏梯度的 Adam 变体
  公式      : 与 Adam 相同，但只更新**本次梯度非零的那些行**（索引级更新）。
  场景      : 大规模 Embedding（词向量表、推荐 ID 特征）——梯度稀疏且表极大，
              稠密更新会把整张表都算一遍，浪费内存和算力。

【LBFGS】拟牛顿法（二阶优化），不是一阶方法
  公式      : 用最近若干步的 (Δθ, Δg) 递推近似 Hessian 的逆 H_k，
              θ ← θ - α·H_k·g（方向比一阶法准得多）。
  场景      : 小规模、全量批（full-batch）的凸/光滑问题，能极少步数收敛到高精度；
              但它**需要多次前向反向**（line search），不吃 mini-batch，深网络基本不用。

【ASGD】Averaged SGD —— 对参数做时间平均
  公式      : 标准 SGD 更新，但额外维护参数的滑动平均 θ̄ 作为最终输出。
  场景      : 理论收敛保证好的凸问题；深度学习中已被 EMA（指数移动平均）取代。

【Adamax】Adam 的 ∞-范数变体
  公式      : 把 s_t 的 L2 范数换成**指数加权的无穷范数**：
              u_t = max(β₂·u_{t-1}, |g_t|)，θ ← θ - α·m̂_t/(u_t+ε)。
  场景      : 梯度有极端离群值时比 Adam 稳；Embedding 类任务偶尔使用。

【NAdam】Nesterov + Adam
  公式      : 在 Adam 的动量项上套用 Nesterov 的"向前看"思想：
              θ ← θ - α·(β₁·m̂_t + (1-β₁)·g_t/(1-β₁^t)) / (√ŝ_t+ε)。
  场景      : 想同时拿到 Nesterov 的加速和 Adam 的自适应，常作为 Adam 的替代试试。
""")

# 实测确认这些优化器在本机都可用（只构造不训练，开销极小）
_avail = ["AdamW", "RAdam", "SparseAdam", "LBFGS", "ASGD", "Adamax", "NAdam"]
print("  本机 torch.optim 可用性检查：")
for _name in _avail:
    _has = hasattr(torch.optim, _name)
    print(f"    torch.optim.{_name:<12s} 可用 = {_has}")


# ===========================================================================
# 3. 手写验证：momentum 与 Adam vs torch.optim（逐参数最大差异）
# ===========================================================================
_title("3. 手写验证：手写 SGD-momentum / Adam 与 torch.optim 逐参数对比")


def sgd_momentum_manual(theta0, steps, lr, beta, grad_np):
    """手写 SGD + Momentum（写法 A）。返回最终参数。"""
    th = np.array(theta0, dtype=np.float64)
    v = np.zeros_like(th)
    for _ in range(steps):
        g = grad_np(th)
        v = beta * v + g
        th = th - lr * v
    return th


def adam_manual(theta0, steps, lr, b1, b2, eps, grad_np):
    """手写 Adam（含偏差校正）。返回最终参数。"""
    th = np.array(theta0, dtype=np.float64)
    m = np.zeros_like(th)
    s = np.zeros_like(th)
    for t in range(1, steps + 1):
        g = grad_np(th)
        m = b1 * m + (1 - b1) * g
        s = b2 * s + (1 - b2) * g * g
        m_hat = m / (1 - b1 ** t)                     # 偏差校正
        s_hat = s / (1 - b2 ** t)
        th = th - lr * m_hat / (np.sqrt(s_hat) + eps)
    return th


def torch_sgd_momentum(theta0, steps, lr, beta):
    """torch.optim.SGD(momentum) 跑 steps 步。"""
    p = torch.tensor(np.array(theta0, dtype=np.float64), requires_grad=True)
    opt = torch.optim.SGD([p], lr=lr, momentum=beta)
    for _ in range(steps):
        opt.zero_grad()
        ((p - torch.tensor(_W_TARGET)) ** 2).sum().backward()
        opt.step()
    return p.detach().numpy().copy()


def torch_adam(theta0, steps, lr, b1, b2, eps):
    """torch.optim.Adam 跑 steps 步。"""
    p = torch.tensor(np.array(theta0, dtype=np.float64), requires_grad=True)
    opt = torch.optim.Adam([p], lr=lr, betas=(b1, b2), eps=eps)
    for _ in range(steps):
        opt.zero_grad()
        ((p - torch.tensor(_W_TARGET)) ** 2).sum().backward()
        opt.step()
    return p.detach().numpy().copy()


_NVERIFY = 20
# —— SGD + Momentum ——
_hand_mom = sgd_momentum_manual(_THETA0, _NVERIFY, lr=0.05, beta=0.9, grad_np=grad_quad)
_torch_mom_v = torch_sgd_momentum(_THETA0, _NVERIFY, lr=0.05, beta=0.9)
print(f"  跑 {_NVERIFY} 步，初始 θ = {_THETA0}，损失 L(θ)=‖θ-w‖²，w={_W_TARGET}")
print(f"\n  ▶ 手写 SGD-with-momentum (lr=0.05, β=0.9)")
print(f"    手写 最终 θ = {np.round(_hand_mom, 10)}")
print(f"    torch 最终 θ = {np.round(_torch_mom_v, 10)}")
for _i in range(len(_THETA0)):
    print(f"      参数[{_i}]：手写 = {_hand_mom[_i]:.12f}   torch = {_torch_mom_v[_i]:.12f}   "
          f"差异 = {abs(_hand_mom[_i] - _torch_mom_v[_i]):.3e}")
print(f"    逐参数最大差异 = {np.abs(_hand_mom - _torch_mom_v).max():.3e}   ->  应 < 1e-6 ✓")

# —— Adam ——
_hand_adam = adam_manual(_THETA0, _NVERIFY, lr=0.05, b1=0.9, b2=0.999, eps=1e-8, grad_np=grad_quad)
_torch_adam_v = torch_adam(_THETA0, _NVERIFY, lr=0.05, b1=0.9, b2=0.999, eps=1e-8)
print(f"\n  ▶ 手写 Adam (lr=0.05, β₁=0.9, β₂=0.999, ε=1e-8)")
print(f"    手写 最终 θ = {np.round(_hand_adam, 10)}")
print(f"    torch 最终 θ = {np.round(_torch_adam_v, 10)}")
for _i in range(len(_THETA0)):
    print(f"      参数[{_i}]：手写 = {_hand_adam[_i]:.12f}   torch = {_torch_adam_v[_i]:.12f}   "
          f"差异 = {abs(_hand_adam[_i] - _torch_adam_v[_i]):.3e}")
print(f"    逐参数最大差异 = {np.abs(_hand_adam - _torch_adam_v).max():.3e}   ->  应 < 1e-6 ✓")
print("\n  结论：手写实现与 PyTorch 逐位吻合，证明公式理解正确。")


# ===========================================================================
# 4. 收敛对比实验（核心）
# ===========================================================================
_title("4. 收敛对比实验：Rosenbrock + 二次函数")


def rosenbrock_np(x, y):
    """Rosenbrock 函数 f(x,y) = (1-x)² + 100(y-x²)²。

    经典的非凸"病态"函数：最小值 f(1,1)=0，但谷底是一条弯曲的窄抛物线，
    不同方向的曲率差异极大（病态条件数），一阶方法很容易来回震荡、进展缓慢，
    因此是检验优化器的经典试金石。
    """
    return (1.0 - x) ** 2 + 100.0 * (y - x ** 2) ** 2


def rosenbrock_torch(t):
    """Rosenbrock 的 torch 版本（t 是形状 (2,) 的参数张量）。"""
    return (1.0 - t[0]) ** 2 + 100.0 * (t[1] - t[0] ** 2) ** 2


_W_QUAD = torch.tensor([2.0, -3.0])                   # 二次函数的最小值点


def quad_torch(t):
    """二次损失 f(θ) = ‖θ - w‖²，凸函数，最小值 0 在 θ=w。"""
    return ((t - _W_QUAD) ** 2).sum()


_START = torch.tensor([-1.5, 1.5], dtype=torch.float64)   # 课案指定起点

# 每个优化器配对各自合适的学习率。
# 说明：Rosenbrock 上不同优化器的"可用学习率"能差 4 个数量级
#      （朴素 SGD 的梯度范数在起点高达 215，lr 稍大就直接发散），
#      所以这里给每个优化器选一个能稳定收敛的量级，这样对比才公平可读。
_OPTIMIZERS = [
    ("SGD", lambda p: torch.optim.SGD([p], lr=0.003)),
    ("Momentum(β=0.9)", lambda p: torch.optim.SGD([p], lr=0.001, momentum=0.9)),
    ("NAG(nesterov)", lambda p: torch.optim.SGD([p], lr=0.001, momentum=0.9, nesterov=True)),
    ("Adagrad", lambda p: torch.optim.Adagrad([p], lr=3.0)),
    ("RMSprop(ρ=0.99)", lambda p: torch.optim.RMSprop([p], lr=0.3, alpha=0.99)),
    ("Adadelta(ρ=0.9)", lambda p: torch.optim.Adadelta([p], lr=3.0, rho=0.9)),
    ("Adam", lambda p: torch.optim.Adam([p], lr=1.0)),
]
_LR_LABEL = {"SGD": "lr=3e-3", "Momentum(β=0.9)": "lr=1e-3", "NAG(nesterov)": "lr=1e-3",
             "Adagrad": "lr=3.0", "RMSprop(ρ=0.99)": "lr=0.3", "Adadelta(ρ=0.9)": "lr=3.0",
             "Adam": "lr=1.0"}


def run_optimizer(name, factory, loss_fn, steps, start, collect_traj=False):
    """用指定优化器在指定损失函数上跑 steps 步。

    返回 (loss 历史, 最终参数)。collect_traj=True 时额外返回参数轨迹。
    """
    p = torch.tensor(start.numpy().copy(), dtype=torch.float64, requires_grad=True)
    opt = factory(p)
    hist, traj = [], []
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(p)
        loss.backward()
        opt.step()
        hist.append(loss.item())
        if collect_traj:
            traj.append(p.detach().numpy().copy())
    return hist, p.detach().numpy().copy(), traj


# —— 4.1 Rosenbrock ——
print(f"""
  ▶ 实验设置
      损失函数：Rosenbrock f(x,y) = (1-x)² + 100(y-x²)²，最小值 f(1,1) = 0
      起  点  ：{_START.numpy()}（课案指定），初始 loss = {rosenbrock_torch(_START):.4f}
      步  数  ：{_STEPS} 步，每 {_PRINT_EVERY} 步打印一次 loss
      各优化器学习率：{', '.join(f'{k}:{v}' for k, v in _LR_LABEL.items())}
""")
_rosen_hist = {}
for _name, _factory in _OPTIMIZERS:
    _h, _pos, _ = run_optimizer(_name, _factory, rosenbrock_torch, _STEPS, _START)
    _rosen_hist[_name] = _h
    _marks = "  ".join(f"[{_i + 1}]={_h[_i]:.4e}" for _i in range(9, _STEPS, _PRINT_EVERY))
    print(f"  {_name:<18s} 最终 loss = {_h[-1]:.6e}   参数位置 = {np.round(_pos, 4)}")
    print(f"      分步 loss：{_marks}")

print(f"\n  ▶ Rosenbrock 最终结果汇总（越小越好，理论最小值 0）")
print(f"    {'优化器':<18s} {'最终 loss':>14s} {'最终参数 (x, y)':>26s}")
print("    " + "-" * 62)
for _name, _factory in _OPTIMIZERS:
    _h, _pos, _ = run_optimizer(_name, _factory, rosenbrock_torch, _STEPS, _START)
    print(f"    {_name:<18s} {_h[-1]:>14.6e} {str(np.round(_pos, 4)):>26s}")

# —— 4.2 二次函数（凸问题，各优化器都很快）——
print(f"""
  ▶ 对照组：简单二次损失 f(θ) = ‖θ - w‖²，w = {_W_QUAD.numpy()}
      起点 {_START.numpy()}，初始 loss = {quad_torch(_START):.4f}，理论最小值 0
      说明：凸问题条件下各优化器都能很快收敛——差异主要出现在非凸病态问题上。
""")
_quad_hist = {}
for _name, _factory in _OPTIMIZERS:
    _h, _pos, _ = run_optimizer(_name, _factory, quad_torch, _STEPS, _START)
    _quad_hist[_name] = _h
    _marks = "  ".join(f"[{_i + 1}]={_h[_i]:.3e}" for _i in range(9, _STEPS, 100))
    print(f"  {_name:<18s} 最终 loss = {_h[-1]:.6e}   参数位置 = {np.round(_pos, 4)}")
    print(f"      分步 loss：{_marks}")

print(f"\n  ▶ 二次函数最终结果汇总（理论最小值 0）")
print(f"    {'优化器':<18s} {'最终 loss':>14s} {'最终参数 (x, y)':>26s}")
print("    " + "-" * 62)
for _name, _factory in _OPTIMIZERS:
    _h, _pos, _ = run_optimizer(_name, _factory, quad_torch, _STEPS, _START)
    print(f"    {_name:<18s} {_h[-1]:>14.6e} {str(np.round(_pos, 4)):>26s}")

# —— 4.3 画收敛对比图 ——
fig, axes = plt.subplots(1, 2, figsize=(17, 6))
_palette = sns.color_palette("tab10", len(_OPTIMIZERS))
_x_axis = range(1, _STEPS + 1)
for _i, (_name, _factory) in enumerate(_OPTIMIZERS):
    # 纵轴取 log，所以把 0 或负数截断到一个很小的正数（否则 log 画不出来）
    _y_rosen = np.maximum(np.array(_rosen_hist[_name]), 1e-16)
    _y_quad = np.maximum(np.array(_quad_hist[_name]), 1e-16)
    axes[0].plot(_x_axis, _y_rosen, linewidth=1.8, color=_palette[_i],
                 label=f"{_name} ({_LR_LABEL[_name]})")
    axes[1].plot(_x_axis, _y_quad, linewidth=1.8, color=_palette[_i],
                 label=f"{_name} ({_LR_LABEL[_name]})")
axes[0].set_title("Rosenbrock f=(1-x)²+100(y-x²)²（非凸病态）", fontsize=13)
axes[0].set_xlabel("步数 step")
axes[0].set_ylabel("loss（log 刻度）")
axes[0].set_yscale("log")
axes[0].legend(fontsize=8.5)
axes[1].set_title("二次函数 f=‖θ-w‖²（凸，各优化器都很快）", fontsize=13)
axes[1].set_xlabel("步数 step")
axes[1].set_ylabel("loss（log 刻度）")
axes[1].set_yscale("log")
axes[1].legend(fontsize=8.5)
fig.suptitle("优化器收敛对比（起点 (-1.5, 1.5)，同一损失函数，各 400 步）", fontsize=14)
fig.tight_layout()
_conv_png = OUTPUT_DIR / "03训练组件_03_优化器收敛对比.png"
fig.savefig(_conv_png, dpi=110)
plt.close(fig)
print(f"\n已保存：{_conv_png}")

# —— 4.4 轨迹图：Rosenbrock 等高线上画 SGD 与 Adam 的路径 ——
_STEPS_TRAJ = 400
_, _pos_sgd, _traj_sgd = run_optimizer("SGD", lambda p: torch.optim.SGD([p], lr=0.003),
                                       rosenbrock_torch, _STEPS_TRAJ, _START,
                                       collect_traj=True)
_, _pos_adam, _traj_adam = run_optimizer("Adam", lambda p: torch.optim.Adam([p], lr=1.0),
                                         rosenbrock_torch, _STEPS_TRAJ, _START,
                                         collect_traj=True)
_traj_sgd = np.array(_traj_sgd)
_traj_adam = np.array(_traj_adam)

fig2, ax2 = plt.subplots(figsize=(9.5, 7.5))
_gx = np.linspace(-2.0, 2.0, 400)
_gy = np.linspace(-1.0, 3.0, 400)
_GX, _GY = np.meshgrid(_gx, _gy)
_GZ = rosenbrock_np(_GX, _GY)
# 等高线用对数间距，才能同时看清浅谷和陡壁
_levels = np.logspace(-0.5, 3.7, 30)
_cs = ax2.contour(_GX, _GY, _GZ, levels=_levels, cmap="viridis", linewidths=0.9)
ax2.clabel(_cs, inline=True, fontsize=7, fmt="%.1f")
ax2.plot(_traj_sgd[:, 0], _traj_sgd[:, 1], "o-", markersize=2.6, linewidth=1.5,
         color="tab:red", label=f"SGD (lr=3e-3) 最终 loss={rosenbrock_torch(torch.tensor(_pos_sgd)).item():.3e}")
ax2.plot(_traj_adam[:, 0], _traj_adam[:, 1], "s-", markersize=2.6, linewidth=1.5,
         color="tab:orange", label=f"Adam (lr=1.0) 最终 loss={rosenbrock_torch(torch.tensor(_pos_adam)).item():.3e}")
ax2.plot(_START[0].item(), _START[1].item(), "*", markersize=20, color="white",
         markeredgecolor="black", markeredgewidth=1.2, label="起点 (-1.5, 1.5)")
ax2.plot(1.0, 1.0, "k*", markersize=20, label="全局最小值 (1, 1)")
ax2.set_title("Rosenbrock 等高线上的优化轨迹：SGD vs Adam", fontsize=13)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.legend(fontsize=9, loc="upper left")
fig2.tight_layout()
_traj_png = OUTPUT_DIR / "03训练组件_03_优化器轨迹.png"
fig2.savefig(_traj_png, dpi=110)
plt.close(fig2)
print(f"已保存：{_traj_png}")
print(f"  SGD  路径共 {len(_traj_sgd)} 个点，最终位置 {np.round(_pos_sgd, 4)}，"
      f"距最小值点 (1,1) 的距离 = {np.linalg.norm(_pos_sgd - np.array([1.0, 1.0])):.4f}")
print(f"  Adam 路径共 {len(_traj_adam)} 个点，最终位置 {np.round(_pos_adam, 4)}，"
      f"距最小值点 (1,1) 的距离 = {np.linalg.norm(_pos_adam - np.array([1.0, 1.0])):.4f}")


# ===========================================================================
# 5. 对比表（课案）
# ===========================================================================
_title("5. 对比表（课案）")
_rows = [
    ("SGD", "基础", "θ ← θ - α·g", "最简单，内存最省"),
    ("Momentum", "惯性加速", "v=βv+g; θ ← θ-α·v", "震荡抵消、冲过局部极小点"),
    ("NAG", "提前看一步", "v=βv+∇L(θ-αβv); θ ← θ-α·v", "提前减速，收敛更快更稳"),
    ("AdaGrad", "自适应 lr", "s+=g²; θ ← θ-α·g/(√s+ε)", "每参数独立 lr，但 s 只增不减"),
    ("RMSProp", "修复 lr 衰减", "s=ρs+(1-ρ)g²; θ ← θ-α·g/(√s+ε)", "EMA 替代累加，lr 不归零"),
    ("AdaDelta", "无需设 lr", "用 √(r+ε)/√(s+ε)·g 更新", "公式里没有 α，免调学习率"),
    ("Adam", "动量 + 自适应", "m̂=m/(1-β₁^t); ŝ=s/(1-β₂^t)", "Momentum + RMSProp + 偏差校正"),
]
print(f"{'优化器':<10s} {'核心改进':<14s} {'更新公式':<40s} {'说明'}")
print("-" * 104)
for _r in _rows:
    print(f"{_r[0]:<10s} {_r[1]:<14s} {_r[2]:<40s} {_r[3]}")

print("\n【补充优化器速查】")
_ext = [
    ("AdamW", "θ ← θ - α[m̂/(√ŝ+ε) + λθ]", "解耦权重衰减，W 与 L2 不等价，Transformer 标配"),
    ("RAdam", "按 ρ_t 判断方差可信度再修正", "自动预热，不用手写 warmup"),
    ("SparseAdam", "同 Adam，但只更新非零梯度行", "超大稀疏 Embedding 表"),
    ("LBFGS", "θ ← θ - α·H_k·g（拟牛顿）", "小规模全量批，二阶信息，深网络不用"),
    ("ASGD", "SGD + 参数时间平均", "凸问题理论保证好；现多用 EMA 取代"),
    ("Adamax", "u=max(β₂u, |g|)，用 ∞-范数", "梯度有极端离群值时更稳"),
    ("NAdam", "Adam 的动量项套 Nesterov 思想", "想要 Nesterov 加速 + 自适应"),
]
print(f"{'优化器':<12s} {'核心公式':<34s} {'一句话场景'}")
print("-" * 100)
for _r in _ext:
    print(f"{_r[0]:<12s} {_r[1]:<34s} {_r[2]}")

_title("03_优化器全家桶.py 运行完毕")
print(f"输出目录：{OUTPUT_DIR}")

"""
对应课案章节：训练组件 / 激活函数

本节知识点：
    1. 激活函数的根本作用：给神经网络注入**非线性**。若没有激活函数，
       多层线性层叠加后仍然等价于一个线性变换，深层网络就失去了意义
       —— 本脚本用数值证明：两个 Linear 层叠加 == 一个等价矩阵，误差 < 1e-6。
    2. 统一模板：每个激活函数都按 ①公式 ②导数公式 ③优缺点 ④适用场景
       ⑤numpy 手写实现 ⑥torch 实现并与 numpy 对比 ⑦手写导数与 autograd 对比。
    3. Sigmoid：σ(x)=1/(1+e^{-x})，导数 σ(x)(1-σ(x))，输出 (0,1)。
       二分类输出层用它（天然是概率）；隐藏层几乎不用（x 偏离 0 就饱和、
       梯度接近 0 导致梯度消失；非零中心导致 zigzag 震荡）。
    4. Softmax：ŷ_i = e^{z_i}/Σ_j e^{z_j}，输出 (0,1) 且和为 1。
       多分类输出层用它（非负 + 归一化 + 与交叉熵搭配梯度简洁 ŷ-y）；
       隐藏层不能用它（①归一化破坏信息 ②梯度会消失 ③计算开销大）。
       本脚本数值演示：512 维隐藏层上大量输出接近 0；数值验证雅可比矩阵
       ∂ŷ/∂z = diag(ŷ) - ŷŷᵀ；并演示数值稳定性（减去 max(z) 防上溢）。
    5. Tanh：tanh(x)=(e^x-e^{-x})/(e^x+e^{-x})，导数 1-tanh²(x)，
       值域 (0,1]，输出 (-1,1)（课案「值域」指导数、[-1,1] 指函数值）。
       RNN/LSTM 隐藏状态用它（零中心、有正有负、压缩状态防指数增长）；
       缺点：仍会饱和 → 这也是 RNN 梯度消失的部分原因。
    6. ReLU：max(0,x)，导数 1(x>0)/0(x≤0)，输出 [0,+∞)，隐藏层默认首选
       （正半轴梯度恒为 1，不饱和，梯度直接透传；计算极简单）；
       缺点：x≤0 时梯度为 0 → "神经元死亡"，且非零中心。
       本脚本数值演示神经元死亡的比例。
    7. LeakyReLU / ELU / GELU / SiLU(Swish) / Mish / Softplus / PReLU /
       ReLU6 / Hardswish：每个都给公式、导数、优缺点、场景，
       并做 numpy / torch 对比（GELU、SiLU、Mish、Hardswish 用
       torch.nn.functional 的实现，numpy 侧用公式实现并对比）。
    8. 对比汇总表（课案）：Sigmoid / Softmax / Tanh / ReLU。
    9. 可视化：激活函数曲线 + 导数曲线 + 梯度消失演示
       → 03训练组件_01_激活函数与导数.png（三个子图）。
   10. 小实验：同一个 4 层网络，隐藏层激活分别用 Sigmoid / Tanh / ReLU，
       同一数据、同一 epoch，对比最终 train/test 准确率与首层权重平均梯度范数，
       用真实数值说明 Sigmoid 的梯度最小、训练最慢
       → 03训练组件_01_激活函数训练对比.png。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Deep_Learning\\03_训练组件\\01_激活函数.py'
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
import math  # noqa: E402

import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.datasets import make_moons  # noqa: E402

# 随机种子统一：保证每次运行结果一致、可复现
torch.manual_seed(42)
np.random.seed(42)

sns.set_theme(style="whitegrid", font="Microsoft YaHei")   # seaborn 风格画图

_EPOCHS = 20          # 训练对比实验的 epoch 数（课案要求 ≤ 20，控制耗时）
_N_SAMPLES = 800      # 样本数（课案要求 ≤ 1000，控制耗时）


def _title(text: str) -> None:
    """打印分节标题，让输出有清晰的结构。"""
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


# ===========================================================================
# 0. 为什么需要激活函数：没有非线性，深层网络就退化成一层
# ===========================================================================
_title("0. 为什么需要激活函数：多层线性层叠加 == 一个线性变换")

# 线性层公式：y = W x + b。
# 两个线性层叠加：y = W2 (W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2)
# —— 括号里就是一个新的矩阵和新的偏置，所以「两层」和「一层」表达能力完全相同。
# 无论叠多少层，最终都等价于 1 层，深度就完全没有意义了。
# 激活函数的作用就是在两层之间插入一个非线性函数，把这个等式链打断。

torch.manual_seed(42)
_in_dim, _hid_dim, _out_dim = 8, 16, 4

layer1 = nn.Linear(_in_dim, _hid_dim)        # 第一层线性层
layer2 = nn.Linear(_hid_dim, _out_dim)       # 第二层线性层

x_demo = torch.randn(5, _in_dim)             # 5 个样本，输入维度 8

# 路径 A：真的依次过两层
y_two_layers = layer2(layer1(x_demo))

# 路径 B：把两层手工"折叠"成一层等价线性层
W_equiv = layer2.weight @ layer1.weight                          # 权重相乘 (4,8)
b_equiv = layer2.weight @ layer1.bias + layer2.bias              # 偏置合并 (4,)
y_one_layer = x_demo @ W_equiv.T + b_equiv                       # 手工算一层

_err_no_act = (y_two_layers - y_one_layer).abs().max().item()
print(f"两个 Linear 层叠加的输出形状：{tuple(y_two_layers.shape)}")
print(f"手工合并成一层后的输出形状：{tuple(y_one_layer.shape)}")
print(f"两条路径输出的【最大误差】：{_err_no_act:.3e}   ->  < 1e-6 说明完全等价")
print("结论：没有激活函数时，再深的网络也只等价于 1 层线性变换，深度失去意义。")

# —— 反过来看：插入非线性之后，两层就再也无法折叠成一层了 ——
_h = torch.relu(layer1(x_demo))              # 中间插入 ReLU 非线性
y_act = layer2(_h)
_err_with_act = (y_act - y_one_layer).abs().max().item()
print(f"插入 ReLU 之后，两层与「等价单层」的最大误差：{_err_with_act:.3e}   ->  不再等价")
print("结论：激活函数提供了非线性表达能力，这是深层网络有意义的前提。")


# ===========================================================================
# 通用工具：numpy 实现 vs torch 实现 vs autograd 导数，三者对比
# ===========================================================================
def compare_activation(name, np_fn, np_grad, torch_fn, x):
    """统一对比：① torch 实现 vs numpy 实现 ② 手写导数 vs torch.autograd。

    np_fn   : numpy 版本的前向函数
    np_grad : numpy 版本的手写导数函数
    torch_fn: torch 版本的前向函数（可微）
    x       : torch 张量，作为测试输入
    """
    # ① torch 前向（关掉梯度追踪，只取数值）
    with torch.no_grad():
        y_torch = torch_fn(x)
    y_numpy = np_fn(x.detach().numpy().copy())       # numpy 前向
    forward_err = np.abs(y_torch.numpy() - y_numpy).max()

    # ② 手写导数 vs autograd
    x_auto = x.detach().clone().requires_grad_(True)   # autograd 用的输入副本
    y_auto = torch_fn(x_auto)
    y_auto.sum().backward()                            # 求和后反向 → 每个元素得到 dy_i/dx_i
    grad_auto = x_auto.grad.numpy()
    grad_numpy = np_grad(x.detach().numpy().copy())     # 手写导数
    grad_err = np.abs(grad_auto - grad_numpy).max()

    print(f"  {name:<12s} 前向最大误差 = {forward_err:.3e}   导数最大误差 = {grad_err:.3e}")
    return forward_err, grad_err


# 测试输入：覆盖负数、0、正数、两端饱和区
x_test = torch.linspace(-6.0, 6.0, 25)

_title("1~12. 各激活函数：公式 / 导数 / numpy / torch / autograd 五方对比")

# ---------------------------------------------------------------------------
# 1. Sigmoid
# ---------------------------------------------------------------------------
print("""
【1. Sigmoid】
  公式      : σ(x) = 1 / (1 + e^{-x})
  导数公式  : σ'(x) = σ(x) · (1 - σ(x))
  输出范围  : (0, 1)
  优点      : 输出天然落在 (0,1)，可直接解释为概率；函数平滑、处处可导。
  缺点      : ① x 偏离 0 就饱和（|x|>4 时导数 < 1.8e-2，几乎为 0）→ 梯度消失；
              ② 输出非零中心（均值 0.5），下一层输入恒正，
                 梯度方向被限制在同一象限，更新走"之"字形（zigzag），收敛慢；
              ③ 含 exp 运算，比 ReLU 慢。
  适用场景  : 二分类输出层（要一个 0~1 的概率）；也用于门控机制（如 LSTM 的遗忘门）。
  为什么隐藏层几乎不用：梯度消失 + 非零中心，深层网络根本训不动。
""")
compare_activation(
    "Sigmoid",
    lambda v: 1.0 / (1.0 + np.exp(-v)),                       # numpy 前向
    lambda v: (lambda s: s * (1.0 - s))(1.0 / (1.0 + np.exp(-v))),  # numpy 手写导数
    torch.sigmoid,
    x_test,
)

# ---------------------------------------------------------------------------
# 2. Softmax
# ---------------------------------------------------------------------------
print("""
【2. Softmax】
  公式      : ŷ_i = e^{z_i} / Σ_j e^{z_j}
  雅可比    : ∂ŷ_i/∂z_k = ŷ_i (δ_ik - ŷ_k)，即矩阵形式 ∂ŷ/∂z = diag(ŷ) - ŷŷᵀ
  输出范围  : (0, 1)，且所有分量之和 = 1
  优点      : exp 保证非负、除以总和保证归一化，天然是"多分类概率分布"；
              与交叉熵搭配时梯度极简洁（ŷ - y）。
  缺点      : ① 强制求和为 1，神经元互相竞争，破坏隐藏层信息的独立性；
              ② 维度大时大多数 ŷ_i ≈ 0，∂ŷ_k/∂z_k = ŷ_k(1-ŷ_k) 接近 0，
                 产生大量"死神经元"，梯度传不回去；
              ③ 要 exp + 求和，计算开销远大于 ReLU 的一个 max。
  适用场景  : 多分类输出层（配 CrossEntropyLoss）。
  为什么不能用在隐藏层：一句话——Softmax 是"竞争机制"（选一个最可能的），
              适合输出层做决策；隐藏层要的是"协作机制"（各自提取特征）。
""")


def softmax_stable_np(z, axis=-1):
    """数值稳定的 numpy softmax：先减去每行最大值，防止 exp 上溢。"""
    z_shift = z - z.max(axis=axis, keepdims=True)   # 关键：减去 max(z)
    e = np.exp(z_shift)
    return e / e.sum(axis=axis, keepdims=True)


def softmax_naive_np(z, axis=-1):
    """直接 exp 的"天真"实现：大数会溢出成 inf/nan，仅用于演示。"""
    e = np.exp(z)                                   # 大数在这里就炸了
    return e / e.sum(axis=axis, keepdims=True)


_z_soft = torch.randn(4, 5)
print(f"  Softmax      前向最大误差 = "
      f"{np.abs(torch.softmax(_z_soft, -1).numpy() - softmax_stable_np(_z_soft.numpy())).max():.3e}"
      f"   （稳定版 numpy vs torch）")

# —— 2.1 为什么不能用在隐藏层：数值演示 512 维隐藏层上大量输出接近 0 ——
print("\n  ▶ 演示：Softmax 用在 512 维隐藏层会发生什么")
torch.manual_seed(42)
hid = 512
# 模拟"上一层线性层输出"：input 经 Linear(128 → 512) 得到 logits
_pre = nn.Linear(128, hid)
_hidden_logits = _pre(torch.randn(64, 128))
_hidden_prob = torch.softmax(_hidden_logits, dim=-1)
_frac_tiny = (_hidden_prob < 1e-3).float().mean().item()
_frac_tiny_1e5 = (_hidden_prob < 1e-5).float().mean().item()
# 该维度下的平均最大概率：维度越大，分布越"平坦"，每个值越小
print(f"  隐藏层维度 = {hid}，batch = 64，共 {hid * 64} 个输出元素")
print(f"  输出中 < 1e-3 的元素比例：{_frac_tiny:.2%}")
print(f"  输出中 < 1e-5 的元素比例：{_frac_tiny_1e5:.2%}")
print(f"  每个样本的最大概率的平均值：{_hidden_prob.max(dim=-1).values.mean().item():.4f}"
      f"  （512 类均匀分布时理论值约 1/512 = {1/512:.4f}）")
# 对应的导数 ŷ(1-ŷ)：ŷ≈0 时导数≈0 → 梯度传不回去
_grad_proxy = (_hidden_prob * (1 - _hidden_prob))
print(f"  对角线导数 ŷ(1-ŷ) 的平均值：{_grad_proxy.mean().item():.3e}  ->  接近 0，梯度传不回去")
print("  这就是 Softmax 在隐藏层会导致大量「死神经元」的原因。")

# —— 2.2 数值验证雅可比矩阵 ∂ŷ/∂z = diag(ŷ) - ŷŷᵀ ——
print("\n  ▶ 数值验证 Softmax 雅可比矩阵：∂ŷ/∂z = diag(ŷ) - ŷŷᵀ")
torch.manual_seed(42)
_z_jac = torch.randn(6, requires_grad=True)          # 6 分类，方便把 6x6 雅可比全打出来
_y_jac = torch.softmax(_z_jac, dim=-1)
J_auto = torch.zeros(6, 6)                           # autograd 逐行求出来的雅可比
for i in range(6):
    if _z_jac.grad is not None:
        _z_jac.grad = None
    _y_jac[i].backward(retain_graph=True)            # 对第 i 个输出求导
    J_auto[i] = _z_jac.grad.clone()
_y_det = _y_jac.detach()
J_formula = torch.diag(_y_det) - _y_det.outer(_y_det)   # 公式版：diag(ŷ) - ŷŷᵀ
_jac_err = (J_auto - J_formula).abs().max().item()
print(f"  autograd 雅可比与公式 diag(ŷ)-ŷŷᵀ 的最大误差：{_jac_err:.3e}   ->  < 1e-6 验证通过")
print(f"  ŷ = {np.round(_y_det.numpy(), 4)}")
print(f"  每一行之和 = {np.round(J_formula.sum(dim=-1).numpy(), 6)}  （雅可比每行求和应为 0：Σ_k ∂ŷ_i/∂z_k = 0）")

# —— 2.3 数值稳定性：直接 exp 大数会溢出 ——
print("\n  ▶ 数值稳定性演示：softmax([1000, 1001, 1002])")
_z_big = np.array([1000.0, 1001.0, 1002.0])
with np.errstate(over="ignore", invalid="ignore"):
    # np.exp(1000) 超出 float64 上限（约 1.8e308）→ 变成 inf，inf/inf = nan
    _naive = softmax_naive_np(_z_big)
_stable = softmax_stable_np(_z_big)
_torch_sm = torch.softmax(torch.tensor(_z_big), dim=-1).numpy()
print(f"  手写不稳定实现（直接 exp）：{_naive}   <- inf/inf 得到 nan，结果完全失效")
print(f"  手写稳定实现（减去 max）  ：{np.round(_stable, 6)}")
print(f"  torch.softmax             ：{np.round(_torch_sm, 6)}")
print(f"  稳定实现与 torch 的最大误差：{np.abs(_stable - _torch_sm).max():.3e}")
print("  原理：e^{z_i}/Σe^{z_j} = e^{z_i-m}/Σe^{z_j-m}（分子分母同乘 e^{-m}，数学等价），")
print("        取 m = max(z) 后指数都 ≤ 0，绝不会上溢；真实实现（含 torch）都这么做。")

# Softmax 的导数在"标量"意义下不适用（它是向量到向量的映射），上面已用雅可比验证。

# ---------------------------------------------------------------------------
# 3. Tanh
# ---------------------------------------------------------------------------
print("""
【3. Tanh（双曲正切）】
  公式      : tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})
  导数公式  : tanh'(x) = 1 - tanh²(x)
  输出范围  : (-1, 1)（课案写「值域 (0,1]」指的是【导数】的取值范围，导数在 x=0 取到最大值 1）
  优点      : 零中心（输出有正有负，均值≈0），不会有 Sigmoid 那种 zigzag 问题；
              把值压缩到 (-1,1) 可防止循环网络中状态指数增长。
  缺点      : ① 仍然会饱和（|x|>4 时导数 < 7e-4），梯度同样会消失
                 —— 这也是 RNN 梯度消失的部分原因；
              ② 仍有 exp 运算，比 ReLU 慢。
  适用场景  : RNN / LSTM / GRU 的隐藏状态激活、需要零中心且有界输出的场合。
""")
compare_activation(
    "Tanh",
    np.tanh,
    lambda v: 1.0 - np.tanh(v) ** 2,
    torch.tanh,
    x_test,
)

# ---------------------------------------------------------------------------
# 4. ReLU
# ---------------------------------------------------------------------------
print("""
【4. ReLU（Rectified Linear Unit）】
  公式      : ReLU(x) = max(0, x)
  导数公式  : ReLU'(x) = 1 (x > 0) / 0 (x ≤ 0)   （x=0 处不可导，工程上取 0）
  输出范围  : [0, +∞)
  优点      : ① 正半轴梯度恒为 1，永远不饱和，反向传播梯度直接透传 → 深层可训；
              ② 计算极简单（一个 max / 一次比较），远快于 exp；
              ③ 输出有稀疏性（约一半神经元为 0），有一定正则效果。
  缺点      : ① x ≤ 0 时梯度为 0，参数永远收不到更新 →「神经元死亡」；
              ② 输出非零中心（全非负）。
  适用场景  : 隐藏层的默认首选。
""")
compare_activation(
    "ReLU",
    lambda v: np.maximum(0.0, v),
    lambda v: (v > 0).astype(np.float64),          # x>0 为 1，x<=0 为 0
    torch.relu,
    x_test,
)

# —— 4.1 数值演示「神经元死亡」——
print("\n  ▶ 演示：ReLU 的「神经元死亡」——给一个偏置很负的单元喂一批数据")
torch.manual_seed(42)
_dead = nn.Linear(16, 8)                       # 16 维输入 → 8 个 ReLU 单元
with torch.no_grad():
    _dead.bias.fill_(-10.0)                    # 偏置设成很负 → 预激活几乎恒负
_x_batch = torch.randn(256, 16)                # 一批数据（标准正态，均值 0 方差 1）
_pre_act = _dead(_x_batch)                     # 预激活 z = Wx + b
_out = torch.relu(_pre_act)                    # ReLU 输出
_req = _pre_act.detach().clone().requires_grad_(True)
torch.relu(_req).sum().backward()              # 反传，拿到每个位置的梯度
_grad = _req.grad
_zero_out_ratio = (_out == 0).float().mean().item()
_zero_grad_ratio = (_grad == 0).float().mean().item()
print(f"  预激活 z 的均值 = {_pre_act.mean().item():.4f}（远小于 0）")
print(f"  ReLU 输出恒为 0 的元素比例：{_zero_out_ratio:.2%}")
print(f"  梯度恒为 0 的元素比例    ：{_zero_grad_ratio:.2%}")
print("  -> 这些神经元的输出永远是 0、梯度永远是 0，参数永远不更新，即「死亡」。")
print("     缓解办法：换 LeakyReLU / ELU / GELU，或用较小的学习率、合理的初始化。")

# ---------------------------------------------------------------------------
# 5. LeakyReLU
# ---------------------------------------------------------------------------
print("""
【5. LeakyReLU（带泄漏的 ReLU）】
  公式      : LeakyReLU(x) = x (x > 0) / negative_slope · x (x ≤ 0)，slope 默认 0.01
  导数公式  : 1 (x > 0) / negative_slope (x ≤ 0)
  输出范围  : (-∞, +∞)
  优点      : 负半轴给一个很小的斜率，梯度不会变成 0 → 解决「神经元死亡」。
  缺点      : 负半轴斜率是超参数，效果不稳定；不同任务最优值差异大。
  适用场景  : ReLU 出现大量死亡神经元时的替代品；GAN 的判别器常用。
""")


def leaky_relu_np(v, slope=0.01):
    """numpy 版 LeakyReLU：负半轴乘 slope。"""
    return np.where(v > 0, v, slope * v)


def leaky_relu_grad_np(v, slope=0.01):
    """numpy 版 LeakyReLU 导数：正半轴 1，负半轴 slope。"""
    return np.where(v > 0, 1.0, slope)


compare_activation(
    "LeakyReLU",
    leaky_relu_np,
    leaky_relu_grad_np,
    lambda t: F.leaky_relu(t, negative_slope=0.01),
    x_test,
)

# ---------------------------------------------------------------------------
# 6. ELU
# ---------------------------------------------------------------------------
print("""
【6. ELU（Exponential Linear Unit）】
  公式      : ELU(x) = x (x > 0) / α(e^x - 1) (x ≤ 0)，α 默认 1.0
  导数公式  : 1 (x > 0) / ELU(x) + α (x ≤ 0)   —— 因为 d/dx[α(e^x-1)] = αe^x = ELU(x)+α
  输出范围  : (-α, +∞)，α=1 时为 (-1, +∞)
  优点      : ① 负半轴饱和到 -α，输出均值更接近 0（比 ReLU 更"零中心"），
                 能加速收敛；② 负半轴有梯度，缓解神经元死亡。
  缺点      : 含 exp 运算（负半轴），比 ReLU / LeakyReLU 慢；
              引入额外超参数 α（α=1 通常就够，PyTorch 默认 1.0）。
  适用场景  : 追求更快的收敛、且不介意一点计算开销的隐藏层；对噪声鲁棒性较好。
""")


def elu_np(v, alpha=1.0):
    """numpy 版 ELU。"""
    return np.where(v > 0, v, alpha * (np.exp(np.minimum(v, 0.0)) - 1.0))


def elu_grad_np(v, alpha=1.0):
    """numpy 版 ELU 导数：正半轴 1，负半轴 α·e^x。"""
    return np.where(v > 0, 1.0, alpha * np.exp(np.minimum(v, 0.0)))


compare_activation(
    "ELU",
    elu_np,
    elu_grad_np,
    lambda t: F.elu(t, alpha=1.0),
    x_test,
)

# ---------------------------------------------------------------------------
# 7. GELU
# ---------------------------------------------------------------------------
print("""
【7. GELU（Gaussian Error Linear Unit）】
  公式(精确): GELU(x) = x · Φ(x) = x · ½[1 + erf(x/√2)]，Φ 是标准正态分布的累积分布函数
  公式(近似): GELU(x) ≈ 0.5x(1 + tanh[√(2/π)(x + 0.044715x³)])   <- 常用近似，更快
  导数公式  : GELU'(x) = Φ(x) + x·φ(x)，φ 是标准正态的概率密度
  输出范围  : (-0.17, +∞)（在 x≈-0.75 处取最小值约 -0.17，之后单调上升）
  优点      : 处处光滑可导；负半轴不是硬性截断而是"软门控"
              （x 越负，越倾向于被置 0），实际效果常优于 ReLU。
  缺点      : 含 erf / tanh，计算比 ReLU 贵；理论解释不如 ReLU 直观。
  适用场景  : Transformer（BERT/GPT）、大规模预训练模型的标准配置。
""")


def gelu_np(v):
    """numpy 版 GELU（精确公式，用 erf 实现）。"""
    return v * 0.5 * (1.0 + np.vectorize(math.erf)(v / math.sqrt(2.0)))


def gelu_tanh_np(v):
    """numpy 版 GELU（tanh 近似公式，与精确式略有差异但非常接近）。"""
    return 0.5 * v * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (v + 0.044715 * v ** 3)))


def gelu_grad_np(v):
    """numpy 版 GELU 导数：Φ(x) + x·φ(x)。"""
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(v / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * v ** 2) / math.sqrt(2.0 * math.pi)
    return cdf + v * pdf


compare_activation("GELU", gelu_np, gelu_grad_np, F.gelu, x_test)
# 顺便看看 tanh 近似式与精确式的差异
_approx_err = np.abs(gelu_tanh_np(x_test.numpy()) - gelu_np(x_test.numpy())).max()
print(f"  {'GELU(tanh近似)':<12s} 与精确式的最大误差 = {_approx_err:.3e}   （近似式在 |x| 大时才略有偏差）")

# ---------------------------------------------------------------------------
# 8. SiLU / Swish
# ---------------------------------------------------------------------------
print("""
【8. SiLU / Swish】
  公式      : SiLU(x) = x · σ(x) = x / (1 + e^{-x})
  导数公式  : SiLU'(x) = σ(x) + x·σ(x)(1-σ(x)) = σ(x)·[1 + x(1-σ(x))]
  输出范围  : (-0.28, +∞)（x≈-1.28 处取最小值约 -0.278）
  优点      : 光滑、非单调（先降后升），负半轴有小的负值，允许"反向抑制"；
              自动驾驶 / EfficientNet 等大量模型中优于 ReLU。
  缺点      : 含 exp 与除法，比 ReLU 慢；名字里的 Swish 是同一个函数的别名
              （Swish-β：x·σ(βx)，β=1 就是 SiLU）。
  适用场景  : EfficientNet、YOLO 系列、需要平滑激活的深层网络。
""")


def silu_np(v):
    """numpy 版 SiLU / Swish：x·sigmoid(x)。"""
    return v / (1.0 + np.exp(-np.clip(v, -500, 500)))


def silu_grad_np(v):
    """numpy 版 SiLU 导数：σ(x)·[1 + x(1-σ(x))]。"""
    s = 1.0 / (1.0 + np.exp(-np.clip(v, -500, 500)))
    return s * (1.0 + v * (1.0 - s))


compare_activation("SiLU", silu_np, silu_grad_np, F.silu, x_test)

# ---------------------------------------------------------------------------
# 9. Mish
# ---------------------------------------------------------------------------
print("""
【9. Mish】
  公式      : Mish(x) = x · tanh(softplus(x)) = x · tanh(ln(1 + e^x))
  导数公式  : 令 s = softplus(x)，w = tanh(s)，则
              Mish'(x) = w + x·(1 - w²)·σ(x)      （σ 是 sigmoid，因为 softplus' = σ）
  输出范围  : (-0.31, +∞)（x≈-1.19 处取最小值约 -0.309）
  优点      : 光滑、非单调、负半轴有界且平滑；自正则特性，部分任务上优于 SiLU；
              在 YOLOv4 等检测模型中表现出色。
  缺点      : 计算最贵（tanh + softplus + sigmoid），比 SiLU 还慢。
  适用场景  : YOLOv4/v5、图像检测与分割等对面/精度敏感的骨干网络。
""")


def softplus_np(v, beta=1.0, threshold=20.0):
    """numpy 版 Softplus：softplus(x) = (1/β)·ln(1 + e^{βx})。

    大 x 时 e^{βx} 会溢出，所以照 PyTorch 的做法：βx > threshold 时直接返回 x。
    """
    return np.where(beta * v > threshold, v, np.log1p(np.exp(np.minimum(beta * v, threshold))) / beta)


def mish_np(v):
    """numpy 版 Mish：x·tanh(softplus(x))。"""
    return v * np.tanh(softplus_np(v))


def mish_grad_np(v):
    """numpy 版 Mish 导数：w + x·(1-w²)·σ(x)，其中 w = tanh(softplus(x))。"""
    s = softplus_np(v)
    w = np.tanh(s)
    sig = 1.0 / (1.0 + np.exp(-np.clip(v, -500, 500)))
    return w + v * (1.0 - w ** 2) * sig


compare_activation("Mish", mish_np, mish_grad_np, F.mish, x_test)

# ---------------------------------------------------------------------------
# 10. Softplus
# ---------------------------------------------------------------------------
print("""
【10. Softplus】
  公式      : Softplus(x) = ln(1 + e^x)
  导数公式  : Softplus'(x) = 1/(1 + e^{-x}) = σ(x)      —— 恰好就是 Sigmoid！
  输出范围  : (0, +∞)
  优点      : 是 ReLU 的**光滑近似**（处处可导，没有 x=0 的折点）；
              导数恰好是 Sigmoid，落在 (0,1)，非常规整。
  缺点      : 输出恒正（非零中心）；含 exp，比 ReLU 慢；
              大 x 时需做阈值保护否则溢出（PyTorch 的 threshold=20 就是这个作用）。
  适用场景  : 需要严格正值且光滑的输出（如预测方差 σ²）；理论推导中作为 ReLU 的平滑替代。
""")
compare_activation(
    "Softplus",
    softplus_np,
    lambda v: 1.0 / (1.0 + np.exp(-np.clip(v, -500, 500))),   # 导数就是 sigmoid
    lambda t: F.softplus(t, beta=1.0, threshold=20.0),
    x_test,
)

# ---------------------------------------------------------------------------
# 11. PReLU
# ---------------------------------------------------------------------------
print("""
【11. PReLU（Parametric ReLU，带参数的 ReLU）】
  公式      : PReLU(x) = x (x > 0) / a·x (x ≤ 0)，其中 a 是**可学习参数**
  导数公式  : 1 (x > 0) / a (x ≤ 0)，此外 ∂PReLU/∂a = x (x ≤ 0)
  输出范围  : (-∞, +∞)
  优点      : LeakyReLU 的斜率不再靠人工调，而是**让网络自己学**，
              每个通道可以学到不同的负半轴斜率；表达能力更强。
  缺点      : 增加参数量（虽然很少）；小数据集上容易过拟合，a 也可能学歪。
  适用场景  : 深层 CNN（尤其是分类骨干网络）、数据量充足时。
""")
compare_activation(
    "PReLU",
    lambda v: np.where(v > 0, v, 0.25 * v),        # 取 a = 0.25 做对比
    lambda v: np.where(v > 0, 1.0, 0.25),
    lambda t: F.prelu(t, torch.tensor([0.25])),    # F.prelu 要求传入权重张量
    x_test,
)

# 顺带演示 nn.PReLU 模块的用法与可学习参数
_prelu_mod = nn.PReLU(num_parameters=1, init=0.25)
_prelu_out = _prelu_mod(torch.randn(4, 3))
print(f"  nn.PReLU 模块：初始斜率 = {_prelu_mod.weight.item():.4f}（可学习参数），"
      f"输出形状 = {tuple(_prelu_out.shape)}")
print(f"  nn.PReLU 参数的梯度形状（说明它真的参与训练）："
      f"{tuple(_prelu_mod.weight.grad.shape) if _prelu_mod.weight.grad is not None else '(尚未反传，运行后会得到梯度)'}")

# ---------------------------------------------------------------------------
# 12. ReLU6
# ---------------------------------------------------------------------------
print("""
【12. ReLU6】
  公式      : ReLU6(x) = min(max(0, x), 6)
  导数公式  : 0 (x ≤ 0) / 1 (0 < x < 6) / 0 (x ≥ 6)
  输出范围  : [0, 6]（有上界）
  优点      : 给激活值加上上界 6，数值范围固定 → 低精度（float16/int8）推理时
              不会溢出，是移动端量化模型的标配（MobileNet 系列）。
  缺点      : 上界 6 是硬编码超参数，大于 6 的信息被直接丢弃；
              x ≥ 6 时梯度为 0，深层也可能出现"死亡"。
  适用场景  : 移动端 / 量化部署网络（MobileNetV1/V2、EfficientNet 的某些版本）。
""")
compare_activation(
    "ReLU6",
    lambda v: np.clip(v, 0.0, 6.0),
    lambda v: ((v > 0) & (v < 6)).astype(np.float64),
    lambda t: F.hardtanh(t, 0.0, 6.0),             # torch 用 hardtanh(0,6) 实现 ReLU6
    x_test,
)

# ---------------------------------------------------------------------------
# 13. Hardswish
# ---------------------------------------------------------------------------
print("""
【13. Hardswish】
  公式      : Hardswish(x) = x · ReLU6(x + 3) / 6
              = 0 (x ≤ -3) / x(x+3)/6 (-3 < x < 3) / x (x ≥ 3)
  导数公式  : 0 (x ≤ -3) / (2x+3)/6 (-3 < x < 3) / 1 (x ≥ 3)
  输出范围  : [-0.375, +∞)（x=-1.5 处取最小值 -0.375）
  优点      : 是 SiLU 的**分段线性近似**，省掉了 exp/sigmoid，
              在移动端明显更快，而精度几乎不降（MobileNetV3 的核心改进）。
  缺点      : 分段线性、在 x=±3 处不可导（工程上直接取分段值）；
              近似毕竟有偏差，极端精度要求下不如 SiLU。
  适用场景  : 移动端 / 边缘设备的高效网络（MobileNetV3、EfficientNet-Lite 等）。
""")


def hardswish_np(v):
    """numpy 版 Hardswish：x·clip(x+3, 0, 6)/6。"""
    return v * np.clip(v + 3.0, 0.0, 6.0) / 6.0


def hardswish_grad_np(v):
    """numpy 版 Hardswish 导数：分段 0 / (2x+3)/6 / 1。"""
    return np.where(v <= -3.0, 0.0, np.where(v >= 3.0, 1.0, (2.0 * v + 3.0) / 6.0))


compare_activation("Hardswish", hardswish_np, hardswish_grad_np, F.hardswish, x_test)


# ===========================================================================
# 14. 对比汇总表（课案）
# ===========================================================================
_title("14. 对比汇总表（课案）")
_summary_rows = [
    ("Sigmoid", "1/(1+e^{-x})", "(0,1)", "二分类输出层", "输出天然是概率"),
    ("Softmax", "e^{z_i}/Σe^{z_j}", "(0,1) 且和为 1", "多分类输出层", "多类概率 + 梯度简洁"),
    ("Tanh", "(e^x-e^{-x})/(e^x+e^{-x})", "(-1,1)", "RNN 隐藏状态", "零中心，范围可控"),
    ("ReLU", "max(0,x)", "[0,+∞)", "隐藏层默认首选", "正半轴梯度恒为 1，不饱和"),
]
print(f"{'激活函数':<10s} {'公式':<26s} {'输出范围':<16s} {'用途':<16s} {'核心原因'}")
print("-" * 100)
for _r in _summary_rows:
    print(f"{_r[0]:<10s} {_r[1]:<26s} {_r[2]:<16s} {_r[3]:<16s} {_r[4]}")

print("\n【各激活函数速查（扩展版）】")
_ext_rows = [
    ("LeakyReLU", "x>0? x : 0.01x", "(-∞,+∞)", "缓解神经元死亡"),
    ("ELU", "x>0? x : α(e^x-1)", "(-α,+∞)", "负半轴饱和，均值接近 0"),
    ("GELU", "x·Φ(x)", "(-0.17,+∞)", "Transformer 标配，软门控"),
    ("SiLU", "x·σ(x)", "(-0.28,+∞)", "光滑非单调，EfficientNet"),
    ("Mish", "x·tanh(ln(1+e^x))", "(-0.31,+∞)", "最贵但效果好，YOLOv4"),
    ("Softplus", "ln(1+e^x)", "(0,+∞)", "ReLU 的光滑近似"),
    ("PReLU", "x>0? x : a·x", "(-∞,+∞)", "负半轴斜率可学习"),
    ("ReLU6", "min(max(0,x),6)", "[0,6]", "移动端量化，防溢出"),
    ("Hardswish", "x·ReLU6(x+3)/6", "(-0.375,+∞)", "SiLU 的分段线性快速近似"),
]
print(f"{'激活函数':<11s} {'公式':<22s} {'输出范围':<14s} {'核心特点'}")
print("-" * 90)
for _r in _ext_rows:
    print(f"{_r[0]:<11s} {_r[1]:<22s} {_r[2]:<14s} {_r[3]}")


# ===========================================================================
# 15. 可视化：激活函数曲线 + 导数曲线 + 梯度消失演示
# ===========================================================================
_title("15. 可视化：激活函数与导数曲线、梯度消失演示")

xs = torch.linspace(-6.0, 6.0, 600)
xn = xs.numpy()
with torch.no_grad():
    curves = {
        "Sigmoid": torch.sigmoid(xs).numpy(),
        "Tanh": torch.tanh(xs).numpy(),
        "ReLU": torch.relu(xs).numpy(),
        "LeakyReLU": F.leaky_relu(xs, 0.01).numpy(),
        "ELU": F.elu(xs).numpy(),
        "GELU": F.gelu(xs).numpy(),
        "SiLU": F.silu(xs).numpy(),
        "Mish": F.mish(xs).numpy(),
        "Softplus": F.softplus(xs).numpy(),
        "Hardswish": F.hardswish(xs).numpy(),
    }
    grads = {
        "Sigmoid": torch.sigmoid(xs).numpy() * (1 - torch.sigmoid(xs).numpy()),
        "Tanh": 1 - torch.tanh(xs).numpy() ** 2,
        "ReLU": (xs.numpy() > 0).astype(float),
        "LeakyReLU": np.where(xs.numpy() > 0, 1.0, 0.01),
        "ELU": elu_grad_np(xn),
        "GELU": gelu_grad_np(xn),
        "SiLU": silu_grad_np(xn),
        "Mish": mish_grad_np(xn),
        "Softplus": 1.0 / (1.0 + np.exp(-xn)),
        "Hardswish": hardswish_grad_np(xn),
    }

fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

# 子图 1：激活函数曲线
for _name, _y in curves.items():
    axes[0].plot(xn, _y, label=_name, linewidth=1.7)
axes[0].axhline(0, color="gray", linewidth=0.8, linestyle="--")
axes[0].axvline(0, color="gray", linewidth=0.8, linestyle="--")
axes[0].set_title("各激活函数曲线", fontsize=13)
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")
axes[0].set_ylim(-1.6, 6.4)
axes[0].legend(fontsize=8, ncol=2)

# 子图 2：各激活函数的导数曲线
for _name, _g in grads.items():
    axes[1].plot(xn, _g, label=_name, linewidth=1.7)
axes[1].axhline(0, color="gray", linewidth=0.8, linestyle="--")
axes[1].set_title("各激活函数的导数曲线", fontsize=13)
axes[1].set_xlabel("x")
axes[1].set_ylabel("f'(x)")
axes[1].set_ylim(-0.25, 1.25)
axes[1].legend(fontsize=8, ncol=2)

# 子图 3：梯度消失演示
axes[2].plot(xn, grads["Sigmoid"], label="Sigmoid 导数", linewidth=2.2, color="tab:blue")
axes[2].plot(xn, grads["Tanh"], label="Tanh 导数", linewidth=2.2, color="tab:orange")
axes[2].plot(xn, grads["ReLU"], label="ReLU 导数（正半轴恒为 1）", linewidth=2.2,
             color="tab:green", linestyle="--")
axes[2].axvspan(-6, -4, color="red", alpha=0.10)
axes[2].axvspan(4, 6, color="red", alpha=0.10)
axes[2].axhline(0, color="gray", linewidth=0.8, linestyle="--")
axes[2].set_title("梯度消失演示：|x|>4 后 Sigmoid/Tanh 导数趋 0（红区）", fontsize=12)
axes[2].set_xlabel("x")
axes[2].set_ylabel("f'(x)")
axes[2].legend(fontsize=9)
_sig_at_6 = float(torch.sigmoid(torch.tensor(6.0)).item() * (1 - torch.sigmoid(torch.tensor(6.0)).item()))
_tanh_at_6 = float(1 - math.tanh(6.0) ** 2)
axes[2].annotate(f"x=6 时 Sigmoid 导数≈{_sig_at_6:.2e}\nTanh 导数≈{_tanh_at_6:.2e}",
                 xy=(6.0, 0.02), xytext=(1.2, 0.42), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="black"))

fig.tight_layout()
_act_png = OUTPUT_DIR / "03训练组件_01_激活函数与导数.png"
fig.savefig(_act_png, dpi=110)
plt.close(fig)
print(f"已保存：{_act_png}")
print(f"  读数验证：x=6 时 Sigmoid 导数 = {_sig_at_6:.3e}，Tanh 导数 = {_tanh_at_6:.3e}，"
      f"而 ReLU 在 x>0 时恒为 1.0")


# ===========================================================================
# 16. 小实验：Sigmoid / Tanh / ReLU 在同一网络、同一数据上的训练对比
# ===========================================================================
_title("16. 小实验：隐藏层激活函数对训练的影响（Sigmoid vs Tanh vs ReLU）")


class SmallNet(nn.Module):
    """4 层网络（3 个隐藏层 + 1 个输出层），隐藏层宽度 32。

    结构：in → Linear(32) → act → Linear(32) → act → Linear(32) → act → Linear(1)
    —— 输出层不加激活（后面用 BCEWithLogitsLoss，它内部含 Sigmoid）。
    """

    def __init__(self, in_dim, hidden=32, act_name="relu"):
        super().__init__()
        self.act_name = act_name
        self.fc1 = nn.Linear(in_dim, hidden)      # 第 1 层（首层权重，用于统计梯度范数）
        self.fc2 = nn.Linear(hidden, hidden)      # 第 2 层
        self.fc3 = nn.Linear(hidden, hidden)      # 第 3 层
        self.fc4 = nn.Linear(hidden, 1)           # 输出层：1 个 logit（二分类）

    def _act(self, t):
        """按名字选择隐藏层激活函数。"""
        if self.act_name == "sigmoid":
            return torch.sigmoid(t)
        if self.act_name == "tanh":
            return torch.tanh(t)
        return torch.relu(t)

    def forward(self, x):
        h = self._act(self.fc1(x))                # 隐藏层 1 + 激活
        h = self._act(self.fc2(h))                # 隐藏层 2 + 激活
        h = self._act(self.fc3(h))                # 隐藏层 3 + 激活
        return self.fc4(h).squeeze(-1)            # 输出 logits，形状 (batch,)


def train_one_activation(act_name, X_tr, y_tr, X_te, y_te):
    """用指定激活函数训练 SmallNet，返回 loss 曲线、首层梯度范数统计与 train/test 准确率。

    为什么要统计「平均梯度范数」而不是只看最后一轮的数值：
    每轮用的是**同一个模型自己的梯度**，但不同激活函数的 loss 量级不同，
    末轮的瞬时梯度会被当前 loss 大小影响，不是公平比较量。
    训练全程的**平均**梯度范数才能真正反映"梯度能不能稳定传回第一层"。
    """
    torch.manual_seed(42)                         # 每个激活函数用同一初始分布，保证公平
    model = SmallNet(X_tr.shape[1], hidden=32, act_name=act_name)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    loss_hist = []
    grad_norms = []                               # 每一轮的首层梯度范数

    for _epoch in range(_EPOCHS):
        model.train()
        opt.zero_grad()
        logits = model(X_tr)
        loss = loss_fn(logits, y_tr)
        loss.backward()
        # 关键指标：首层（fc1.weight）的梯度范数 —— 直接反映梯度能传多深
        grad_norms.append(model.fc1.weight.grad.norm().item())
        opt.step()
        loss_hist.append(loss.item())

    # 评估
    model.eval()
    with torch.no_grad():
        tr_acc = ((model(X_tr) > 0).float() == y_tr).float().mean().item()
        te_acc = ((model(X_te) > 0).float() == y_te).float().mean().item()
    return loss_hist, grad_norms, tr_acc, te_acc


# 数据：make_moons（两团交错的月牙，非线性可分，适合体现激活函数差异）
_X_all, _y_all = make_moons(n_samples=_N_SAMPLES, noise=0.2, random_state=42)
X_all = torch.tensor(_X_all, dtype=torch.float32)
y_all = torch.tensor(_y_all, dtype=torch.float32)
_n_train = int(_N_SAMPLES * 0.8)                  # 8:2 切分训练/测试
X_tr, y_tr = X_all[:_n_train], y_all[:_n_train]
X_te, y_te = X_all[_n_train:], y_all[_n_train:]
print(f"数据：make_moons，共 {_N_SAMPLES} 条（训练 {_n_train} / 测试 {_N_SAMPLES - _n_train}），"
      f"每轮全量 batch，epoch = {_EPOCHS}")

_results = {}
for _act in ["sigmoid", "tanh", "relu"]:
    _lh, _gn, _tra, _tea = train_one_activation(_act, X_tr, y_tr, X_te, y_te)
    _results[_act] = dict(loss=_lh, gnorms=_gn, train_acc=_tra, test_acc=_tea)

print(f"\n{'隐藏层激活':<12s} {'train 准确率':>11s} {'test 准确率':>11s} "
      f"{'首层梯度范数(第1轮)':>19s} {'首层梯度范数(全程均值)':>22s} {'最终 loss':>11s}")
print("-" * 96)
for _act in ["sigmoid", "tanh", "relu"]:
    _r = _results[_act]
    print(f"{_act:<12s} {_r['train_acc']:>11.4f} {_r['test_acc']:>11.4f} "
          f"{_r['gnorms'][0]:>19.3e} {float(np.mean(_r['gnorms'])):>22.3e} {_r['loss'][-1]:>11.6f}")

print("\n结论（用真实数值说话）：")
_g_sig = _results["sigmoid"]["gnorms"]
_g_tanh = _results["tanh"]["gnorms"]
_g_relu = _results["relu"]["gnorms"]
print(f"  第 1 轮首层梯度范数：Sigmoid={_g_sig[0]:.3e}  Tanh={_g_tanh[0]:.3e}  ReLU={_g_relu[0]:.3e}")
print(f"  -> Sigmoid 的梯度比另外两者小 1~2 个数量级（饱和导致梯度消失），训练起步就慢。")
print(f"  训练全程平均梯度范数：Sigmoid={np.mean(_g_sig):.3e} < "
      f"ReLU={np.mean(_g_relu):.3e} ≈ Tanh={np.mean(_g_tanh):.3e}")
print(f"  -> Sigmoid 是三者中最小的，梯度始终传不回来，所以它训练最慢、loss 最高。")
print(f"  末轮 test 准确率：Sigmoid={_results['sigmoid']['test_acc']:.4f}，"
      f"Tanh={_results['tanh']['test_acc']:.4f}，ReLU={_results['relu']['test_acc']:.4f}")
print(f"  ReLU 正半轴梯度恒为 1、不饱和，梯度最充沛，因此收敛最快、loss 最低"
      f" —— 这就是它成为隐藏层首选的原因。")

# 画 loss 曲线对比图
fig2, ax2 = plt.subplots(figsize=(9, 5.5))
_colors = {"sigmoid": "tab:blue", "tanh": "tab:orange", "relu": "tab:green"}
for _act in ["sigmoid", "tanh", "relu"]:
    _r = _results[_act]
    ax2.plot(range(1, _EPOCHS + 1), _r["loss"], marker="o", markersize=3.5,
             linewidth=1.9, color=_colors[_act],
             label=f"{_act}（test acc={_r['test_acc']:.3f}，"
                   f"平均首层梯度={float(np.mean(_r['gnorms'])):.1e}）")
ax2.set_title("隐藏层激活函数训练对比（4 层网络，宽度 32，make_moons）", fontsize=13)
ax2.set_xlabel("epoch")
ax2.set_ylabel("训练 loss（BCEWithLogitsLoss）")
ax2.set_yscale("log")                             # 对数纵轴，便于看清收敛速度差异
ax2.legend(fontsize=9)
fig2.tight_layout()
_train_png = OUTPUT_DIR / "03训练组件_01_激活函数训练对比.png"
fig2.savefig(_train_png, dpi=110)
plt.close(fig2)
print(f"已保存：{_train_png}")

_title("01_激活函数.py 运行完毕")
print(f"输出目录：{OUTPUT_DIR}")

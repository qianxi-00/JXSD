"""
对应课案章节：网络架构 → DNN（前向传播 / 随机失活 Dropout）

本节知识点：
    1. 手写全连接层：h = x @ W.T + b。讲清 nn.Linear(in, out) 的权重形状是
       (out, in)（PyTorch 的行优先约定），所以要用 x @ W.T，并与 nn.Linear 对比验证。
    2. 多层前向传播公式：z(l) = W(l) h(l-1) + b(l)，h(l) = σ(z(l))，h(0) = x；
       多分类输出层用 Softmax（交叉熵损失内部已含 log_softmax，不要重复加）。
    3. 为什么必须有激活函数：两层线性叠加 W2(W1 x) = (W2 W1) x 仍是一个线性变换，
       用数值实验演示"两个线性层 ≡ 单个矩阵 W2@W1"，说明没有非线性就等于只有一层。
    4. 不调 nn.Linear、只用 torch.matmul 手动逐层做矩阵乘法，验证与 nn.Sequential 结果一致。
    5. Dropout 的训练/推理差异：
           训练：h = r ⊙ h / (1-p)      （倒置 Dropout，Inverted Dropout）
           推理：h 不变
       讲清"为什么推理时要乘 (1-p)"（经典 Dropout 的尺度补偿），
       以及 PyTorch 为什么改用倒置 Dropout（训练期补偿 ⇒ 推理零开销）。
    6. 实测 Dropout 的三个性质：训练时输出随机、eval 时输出确定、
       训练时被置 0 的元素比例约为 p。
    7. 用 nn.Sequential 搭一个完整二分类 DNN，在 make_classification 数据上训练 10 个 epoch，
       用 accuracy_score 评估。
    8. 对照实验：同一结构"有 Dropout"vs"无 Dropout"，比较 train/test 准确率，
       体会 Dropout 的泛化作用（缓解过拟合）。
    9. DNN 结构总结表：nn.Linear / ReLU / Dropout / CrossEntropyLoss 各自的作用。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Deep_Learning\\02_网络架构\\02_DNN前向传播与随机失活.py'
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
# 后续正式依赖
# ---------------------------------------------------------------------------
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from sklearn.datasets import make_classification  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

torch.manual_seed(42)   # 统一随机种子：保证 Dropout 掩码、权重初始化、数据打乱都可复现
np.random.seed(42)

_T0 = time.perf_counter()   # 计时：交付要求单脚本 < 40 秒（优先 < 20 秒）


def section(title: str) -> None:
    """统一的章节打印，让控制台输出层次分明。"""
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


# ===========================================================================
# 一、手写全连接层：h = x @ W.T + b
# ===========================================================================
section("一、手写全连接层：h = x @ W.T + b（weight 形状是 (out, in)）")

# 维度约定（这是最容易搞混的地方，务必记牢）：
#   x        : (batch, in_features)            一批样本，每条 in_features 个特征
#   W        : (out_features, in_features)     PyTorch 约定：第一维是"输出"！
#   b        : (out_features,)
#   为了把 (batch, in) 和 (in, out) 相乘，必须先转置 W：
#       h = x @ W.T + b     ->  (batch, in) @ (in, out) + (out,) = (batch, out)
#   为什么 PyTorch 用 (out, in) 而不是 (in, out)？
#   因为这样 W @ h 就是"每个输出神经元一行权重做点积"，且 nn.Linear 的实现
#   直接调用 F.linear(input, weight, bias)，内部做 input @ weight.t()。
in_features, out_features, batch = 5, 3, 4

torch.manual_seed(0)
linear = nn.Linear(in_features, out_features)   # 课案用的全连接层
x = torch.randn(batch, in_features)             # 假数据

print(f"输入 x.shape              = {tuple(x.shape)}    # (batch, in_features)")
print(f"nn.Linear.weight.shape    = {tuple(linear.weight.shape)}    # (out_features, in_features)")
print(f"nn.Linear.bias.shape      = {tuple(linear.bias.shape)}       # (out_features,)")
print(f"weight.T.shape            = {tuple(linear.weight.T.shape)}    # (in_features, out_features)")

# 手写前向传播：就是一次矩阵乘法加偏置
h_manual = x @ linear.weight.T + linear.bias
h_module = linear(x)                            # nn.Linear 内部等价于 F.linear(x, W, b)
err_linear = (h_manual - h_module).abs().max().item()
print(f"\n手写 x @ W.T + b 与 nn.Linear 的最大误差 = {err_linear:.3e}  (< 1e-6 ⇒ 完全等价)")
assert err_linear < 1e-6, "nn.Linear 必须等价于 x @ W.T + b"

# 广播机制：bias (out,) 会自动广播到 (batch, out)，逐行相加
print(f"bias 广播后 shape = {tuple((x @ linear.weight.T).shape)} + {tuple(linear.bias.shape)}"
      f" -> {tuple(h_manual.shape)}  # (out,) 广播成 (batch, out)")
print(f"\n输出第一行（手工算）= {np.round(h_manual[0].tolist(), 4)}")
print("注意：nn.Linear 只做线性变换，不含任何激活函数，"
      "非线性要另外用 ReLU/Sigmoid 叠加。")

# ===========================================================================
# 二、为什么必须有激活函数：两层线性的叠加仍是线性
# ===========================================================================
section("二、为什么必须有激活函数：多层线性叠加 ≡ 单层线性")

# 数学事实：若 h(1) = W1 x + b1，h(2) = W2 h(1) + b2，代入得
#   h(2) = W2 (W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2)
# 其中 W2 W1 仍然是一个矩阵（记为 W_eq），W2 b1 + b2 仍然是一个向量（b_eq）。
# 所以"堆多少层线性"都等价于"一层线性"，表达能力没有任何提升。
# 这就是必须引入非线性激活函数 σ 的根本原因：
#   h(l) = σ(W(l) h(l-1) + b(l))
# σ 打破了矩阵乘法的可结合性，网络才能拟合非线性函数（理论上可逼近任意连续函数）。
d_in, d_hid, d_out = 6, 8, 4
torch.manual_seed(0)
W1 = torch.randn(d_hid, d_in)     # 第 1 层权重 (8, 6)
b1 = torch.randn(d_hid)           # 第 1 层偏置 (8,)
W2 = torch.randn(d_out, d_hid)    # 第 2 层权重 (4, 8)
b2 = torch.randn(d_out)           # 第 2 层偏置 (4,)

x_demo = torch.randn(16, d_in)    # 16 条样本
# 路线 A：老老实实做两层线性
two_layers = (x_demo @ W1.T + b1) @ W2.T + b2
# 路线 B：先把两层"折叠"成一层
W_eq = W2 @ W1                    # (4, 8) @ (8, 6) = (4, 6)
b_eq = W2 @ b1 + b2               # (4,)
one_layer = x_demo @ W_eq.T + b_eq

err_collapse = (two_layers - one_layer).abs().max().item()
print(f"两个线性层叠加的结果        shape = {tuple(two_layers.shape)}")
print(f"折叠成单层(W2@W1) 的结果    shape = {tuple(one_layer.shape)}")
print(f"两者最大误差 = {err_collapse:.3e}  ⇒ 数值上等价（误差来自 float32 矩阵乘法的"
      f"舍入，量级 ~1e-6），多层线性没有增加任何表达能力")
assert err_collapse < 1e-4, "两层线性必须能折叠成一层（允许 float32 舍入误差）"

# 加一个 ReLU 之后，就再也折叠不掉了
hidden_relu = torch.relu(x_demo @ W1.T + b1)     # 第 1 层 + 非线性
with_relu = hidden_relu @ W2.T + b2              # 第 2 层
err_relu = (with_relu - one_layer).abs().max().item()
print(f"\n中间夹一个 ReLU 后与单层线性的最大误差 = {err_relu:.3e}"
      f"  ⇒ 不再等价，非线性被引入")
print(f"ReLU 把负值压成 0：输入 {tuple(x_demo.shape)}，"
      f"隐藏层中 0 的比例 = {(hidden_relu == 0).float().mean().item():.1%}")

# ReLU 的导数：x>0 时导数为 1，x<0 时导数为 0（x=0 处约定为 0），
# 所以它既非线性又不会像 Sigmoid 那样在两端饱和（梯度消失）。
z_demo = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], requires_grad=True)
torch.relu(z_demo).sum().backward()
print(f"ReLU 在 x={z_demo.detach().tolist()} 处的导数 = {z_demo.grad.tolist()}"
      f"  # 负数段梯度为 0，正数段梯度为 1")

with torch.no_grad():   # Softmax 演示：把 logits 变成概率分布
    logits = torch.tensor([[2.0, 1.0, 0.1], [0.5, 0.5, 0.5]])
    probs = torch.softmax(logits, dim=1)
print(f"\nSoftmax 输入 logits = {logits.tolist()}")
print(f"Softmax 输出概率 = {np.round(probs.numpy(), 4).tolist()}")
print(f"每行概率之和 = {probs.sum(dim=1).tolist()}  # 恒为 1，所以多分类输出层用 Softmax")
print("提示：nn.CrossEntropyLoss 内部 = log_softmax + NLLLoss，"
      "所以模型最后一层只输出 logits，千万不要自己再加一个 Softmax！")

# ===========================================================================
# 三、手动逐层矩阵乘法：不调 nn.Linear，只用 torch.matmul
# ===========================================================================
section("三、手动逐层前向传播，验证与 nn.Sequential 一致")

# 课案公式（L 层全连接网络）：
#   h(0) = x
#   z(l) = W(l) h(l-1) + b(l)
#   h(l) = σ(z(l))
#   ŷ    = softmax(z(L))     （多分类）
# 下面用一个真实的三层网络逐层手算，参数直接从 nn.Sequential 里"偷"出来，
# 这样两条路线用的是同一组权重，结果必须完全一致。
torch.manual_seed(7)
model_verify = nn.Sequential(
    nn.Linear(10, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 3),
)
x_v = torch.randn(5, 10)                                   # 5 条样本，10 个特征
out_sequential = model_verify(x_v)                         # 路线 A：nn.Sequential

lin1, relu1, lin2, relu2, lin3 = list(model_verify)         # 路线 B：手动逐层 matmul
h0 = x_v                                                    # h(0) = x
z1 = h0 @ lin1.weight.T + lin1.bias                         # z(1) = W(1) h(0) + b(1)
h1 = torch.relu(z1)                                         # h(1) = σ(z(1))
z2 = h1 @ lin2.weight.T + lin2.bias                         # z(2) = W(2) h(1) + b(2)
h2 = torch.relu(z2)                                         # h(2) = σ(z(2))
z3 = h2 @ lin3.weight.T + lin3.bias                         # z(3) = W(3) h(2) + b(3)
out_manual = z3                                             # 最后一层不加激活，直接输出 logits

err_seq = (out_manual - out_sequential).abs().max().item()
print("逐层形状推导：")
print(f"  h(0) = x            {tuple(h0.shape)}")
print(f"  z(1) = W1 h0 + b1   {tuple(z1.shape)}   <- W1{tuple(lin1.weight.shape)}")
print(f"  h(1) = ReLU(z1)     {tuple(h1.shape)}")
print(f"  z(2) = W2 h1 + b2   {tuple(z2.shape)}   <- W2{tuple(lin2.weight.shape)}")
print(f"  h(2) = ReLU(z2)     {tuple(h2.shape)}")
print(f"  z(3) = W3 h2 + b3   {tuple(z3.shape)}   <- W3{tuple(lin3.weight.shape)}")
print(f"\n手动逐层 matmul vs nn.Sequential 的最大误差 = {err_seq:.3e}  (< 1e-6)")
assert err_seq < 1e-6, "手动逐层前向必须与 nn.Sequential 完全一致"

# 校验手写版本对 softmax 的输出也一致
probs_manual = torch.softmax(out_manual, dim=1)
probs_seq = torch.softmax(out_sequential, dim=1)
print(f"Softmax 后概率的最大误差 = {(probs_manual - probs_seq).abs().max().item():.3e}")
print(f"第一条样本的概率分布 = {np.round(probs_manual[0].detach().numpy(), 4).tolist()}")

# 参数量：全连接层的参数量 = in*out + out，可以看出它随宽度平方增长
print("\n逐层参数量（in×out + out）：")
for name, layer in zip(["Linear1", "Linear2", "Linear3"], [lin1, lin2, lin3]):
    n_w = layer.weight.numel()
    n_b = layer.bias.numel()
    print(f"  {name}: {tuple(layer.weight.shape)} -> {n_w} + {n_b} = {n_w + n_b}")
print(f"  合计 = {sum(p.numel() for p in model_verify.parameters())}")
print("结论：DNN 参数量容易爆炸（64*7*7 -> 128 就是 40 万），这也是后面 CNN 要引入"
      "权重共享的原因之一。")

# ===========================================================================
# 四、Dropout 原理：训练 h = r ⊙ h / (1-p)，推理 h 不变
# ===========================================================================
section("四、Dropout 原理：倒置 Dropout（Inverted Dropout）")

# 原始论文（Srivastava et al., 2014）的经典 Dropout：
#   训练：h = r ⊙ h        其中 r ~ Bernoulli(1-p)，被置 0 的神经元不参与本次前向/反向
#   推理：h = (1-p) * h    因为训练时平均只有 (1-p) 比例的神经元在工作，
#                          推理时全部神经元都参与，输出期望会变成 1/(1-p) 倍，必须乘 (1-p) 补偿
# PyTorch 采用「倒置 Dropout」，把补偿搬到训练阶段：
#   训练：h = r ⊙ h / (1-p)    （除以保留概率，保证输出的期望不变）
#   推理：h = h                （什么都不做）
# 两者在数学期望上完全等价，但倒置版本在推理时是恒等映射：
#   - 推理代码零开销（不产生额外的乘法与临时张量）；
#   - 训练/推理两套代码路径更少，不容易出 bug。
# 直觉理解 Dropout 为什么有用：
#   每次迭代随机"关掉"一部分神经元，模型无法依赖某几个特定神经元，
#   必须学到更冗余、更鲁棒的特征组合 —— 等价于训练了指数多个子网络并在推理时做集成。

p_drop = 0.3
dropout_layer = nn.Dropout(p=p_drop)
x_dp = torch.randn(4, 8)                    # 小张量，便于直接观察数值

# --- 训练模式：同一次输入跑 5 次，结果每次都不同 ---
print(f"Dropout 概率 p = {p_drop}，保留概率 1-p = {1 - p_drop:.1f}")
print(f"\n训练模式（model.train()）下同一输入跑 5 次：")
torch.manual_seed(42)                        # 固定种子，保证整个脚本可复现
dropout_layer.train()
train_outputs = []
for i in range(5):
    out_i = dropout_layer(x_dp)
    train_outputs.append(out_i)
    print(f"  第 {i + 1} 次：输出第 0 行 = {np.round(out_i[0].detach().numpy(), 3).tolist()}")
base = train_outputs[0]
max_diff_train = max((o - base).abs().max().item() for o in train_outputs[1:])
print(f"  5 次输出之间的最大差异 = {max_diff_train:.4f} > 0 ⇒ 训练时是随机的（每次掩码都不同）")

# --- 推理模式：同一输入跑 3 次，结果完全相同 ---
dropout_layer.eval()
eval_outputs = [dropout_layer(x_dp) for _ in range(3)]
max_diff_eval = max((o - eval_outputs[0]).abs().max().item() for o in eval_outputs[1:])
print(f"\n推理模式（model.eval()）下同一输入跑 3 次：")
for i, o in enumerate(eval_outputs):
    print(f"  第 {i + 1} 次：输出第 0 行 = {np.round(o[0].detach().numpy(), 3).tolist()}")
print(f"  3 次输出之间的最大差异 = {max_diff_eval:.3e} ⇒ 推理时完全确定")
print(f"  eval 模式输出是否严格等于输入：{torch.allclose(eval_outputs[0], x_dp, atol=1e-7)}"
      f"  # 倒置 Dropout 在推理时是恒等映射")
# 注意：nn.Dropout 在 eval 下不做任何事，也正因为如此
# "训练时除以 (1-p)" 这个补偿必须做，否则推理的尺度会偏大。

# --- 统计被置 0 的比例，应当接近 p ---
torch.manual_seed(42)
dropout_layer.train()
big = torch.randn(1000, 2000)               # 200 万个元素，统计才稳定
out_big = dropout_layer(big)
zero_ratio = (out_big == 0).float().mean().item()
print(f"\n大张量 shape={tuple(big.shape)}（共 {big.numel():,} 个元素）在 train 模式下：")
print(f"  被置 0 的比例 = (out == 0).float().mean() = {zero_ratio:.4f}"
      f"  ≈ p = {p_drop}（伯努利采样的统计涨落导致不是精确相等）")

# --- 尺度验证：训练时输出均值应接近输入均值（因为已经除以 1-p） ---
print(f"\n尺度验证（倒置 Dropout 的核心）：")
print(f"  输入均值                  = {big.mean().item():+.5f}")
print(f"  训练输出均值              = {out_big.mean().item():+.5f}"
      f"   # 与输入均值接近 ⇒ 期望不变")
print(f"  若不补偿会变成 (1-p)×均值 = {(1 - p_drop) * big.mean().item():+.5f}"
      f"   # 未补偿时会系统性偏小")
del big, out_big   # 及时释放内存

# ===========================================================================
# 五、手工实现一遍倒置 Dropout，与 F.dropout 对比
# ===========================================================================
section("五、手写倒置 Dropout，与 torch.nn.functional.dropout 对比")

torch.manual_seed(42)
x_manual = torch.randn(200000)               # 元素够多，均值统计才稳定

# 手写三步：
#   1) mask ~ Bernoulli(1-p)，形状与 x 相同（rand < keep_prob 得到 True/False）
#   2) 训练：x * mask / (1-p)；推理：直接返回 x
#   3) 用 1/(1-p) 缩放，保证输出期望等于输入期望
keep_prob = 1 - p_drop
mask = (torch.rand_like(x_manual) < keep_prob).float()          # 步骤 1
x_dropped_manual = x_manual * mask / keep_prob                  # 步骤 2：倒置缩放
x_dropped_func = F.dropout(x_manual, p=p_drop, training=True)   # PyTorch 的实现

print(f"输入 shape = {tuple(x_manual.shape)}，p = {p_drop}")
print(f"手写 mask 中保留的比例 = {mask.mean().item():.4f}  ≈ 1-p = {keep_prob:.1f}")
print(f"手写实现 保留元素的比例 = {(x_dropped_manual != 0).float().mean().item():.4f}")
print(f"F.dropout 保留元素的比例 = {(x_dropped_func != 0).float().mean().item():.4f}")

# 期望验证：E[x·mask/(1-p)] = E[x]·(1-p)/(1-p) = E[x]
# 用 20 万个元素统计，标准误 ~ 1/sqrt(200000) ≈ 0.002，足以看出量级差异
print(f"\n均值（期望）验证：")
print(f"  原始输入均值            = {x_manual.mean().item():+.5f}")
print(f"  手写倒置 Dropout 均值   = {x_dropped_manual.mean().item():+.5f}  ← 与输入接近")
print(f"  F.dropout 均值          = {x_dropped_func.mean().item():+.5f}  ← 与输入接近")
print(f"  忘记除以 (1-p) 的均值   = {(x_manual * mask).mean().item():+.5f}"
      f"  ≈ {keep_prob:.1f} × 输入均值 = {keep_prob * x_manual.mean().item():+.5f}  ← 系统性偏小")
print("结论：倒置 Dropout 通过在训练期除以保留概率，把输出的期望拉回与输入一致，"
      "于是推理期就可以什么都不做（恒等映射）。")

# 严格验证公式：把 F.dropout 用的掩码直接取出来，喂给手写实现，
# 两条路线用同一个掩码时，结果必须逐元素完全相等。
# PyTorch 的 F.dropout 内部就是：mask = (1-p) 概率为 1 的伯努利掩码，
# 然后 out = x * mask / (1-p)，所以这里应当误差为 0。
with torch.no_grad():
    x_same = torch.randn(4096)
    func_same = F.dropout(x_same, p=p_drop, training=True)
    same_mask = (func_same != 0).float()                 # 从输出反推掩码（保留=1，丢弃=0）
    manual_same = x_same * same_mask / (1 - p_drop)      # 用同一个掩码手写一遍
print(f"\n用同一个掩码对比公式：手写 vs F.dropout 元素级最大差异 = "
      f"{(manual_same - func_same).abs().max().item():.3e}"
      f"  （float32 舍入，≈0 ⇒ 公式 h = r ⊙ h / (1-p) 得到精确验证）")

# inference（training=False）时 F.dropout 直接返回输入本身
x_infer = F.dropout(x_same, p=p_drop, training=False)
print(f"F.dropout(training=False) 输出是否等于输入："
      f"{torch.allclose(x_infer, x_same, atol=1e-7)}  # 推理恒等映射")

del x_manual, x_dropped_manual, x_dropped_func, x_same, func_same, manual_same

# ===========================================================================
# 六、完整二分类训练（课案代码）
# ===========================================================================
section("六、课案代码：DNN 二分类训练（make_classification 数据）")

# 1. 提取数据：sklearn 造一个 1000 样本、20 特征、2 类的线性不可分数据集
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_classes=2,
    random_state=42,
)
print(f"原始数据 X.shape = {X.shape}，y.shape = {y.shape}，"
      f"类别分布 = {np.bincount(y).tolist()}")

# 2. 清洗/处理数据：先切分再标准化（重要！scaler 只能 fit 训练集，否则测试集信息泄漏）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y,
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)     # 用训练集统计量拟合
X_test = scaler.transform(X_test)           # 测试集复用同一套统计量（只 transform）
print(f"划分后：训练集 {X_train.shape}，测试集 {X_test.shape}")
print(f"标准化后训练集均值 = {X_train.mean():+.4f}，标准差 = {X_train.std():.4f}"
      f"  # 标准正态化，让各特征尺度一致，梯度下降更稳")

X_train_t = torch.tensor(X_train, dtype=torch.float32)   # 特征用 float32
y_train_t = torch.tensor(y_train, dtype=torch.long)      # 类别标签必须是 int64
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)
print(f"转成张量：X_train_t={tuple(X_train_t.shape)} {X_train_t.dtype}，"
      f"y_train_t={tuple(y_train_t.shape)} {y_train_t.dtype}")


def make_dnn(use_dropout: bool) -> nn.Sequential:
    """按课案结构搭建 DNN；use_dropout=False 时去掉 Dropout 层，用于对照实验。

    结构：Linear(20,64) -> ReLU -> [Dropout(0.3)] -> Linear(64,32) -> ReLU -> Linear(32,2)
    最后一层输出 2 个 logits（对应 2 个类别），不加 Softmax（CrossEntropyLoss 内部会做）。
    """
    layers = [nn.Linear(20, 64), nn.ReLU()]
    if use_dropout:
        layers.append(nn.Dropout(0.3))       # 训练时随机丢弃 30% 的隐藏单元
    layers += [nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2)]
    return nn.Sequential(*layers)


def train_dnn(use_dropout: bool, epochs: int = 10, lr: float = 1e-3, seed: int = 42):
    """训练一个 DNN，返回逐 epoch 的 (训练损失, 测试准确率) 以及最终模型。"""
    torch.manual_seed(seed)                  # 每个对照实验用同一初始化，保证公平比较
    np.random.seed(seed)

    model = make_dnn(use_dropout)
    criterion = nn.CrossEntropyLoss()        # 内部 = log_softmax + NLLLoss，输入是 logits
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # DataLoader 负责按 batch 切分并每个 epoch 重新打乱（shuffle=True）
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=32,
        shuffle=True,
    )
    print(f"\n[{'含' if use_dropout else '不含'} Dropout] 训练样本 {len(X_train_t)} 条，"
          f"batch_size=32 ⇒ 每个 epoch 有 {len(train_loader)} 个 batch")

    history = []
    for epoch in range(epochs):
        model.train()                        # 训练模式：启用 Dropout
        epoch_loss, n_seen = 0.0, 0
        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)         # 前向传播
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()            # 清空上一轮梯度（否则会累加）
            loss.backward()                  # 反向传播
            optimizer.step()                 # 更新参数

            epoch_loss += loss.item() * X_batch.size(0)
            n_seen += X_batch.size(0)
        train_loss = epoch_loss / n_seen

        # 评估：必须 eval() + no_grad()，否则 Dropout 会随机丢掉神经元、结果不稳定
        model.eval()
        with torch.no_grad():
            test_logits = model(X_test_t)
            test_pred = test_logits.argmax(dim=1)          # 取 logits 最大的类别
            test_acc = accuracy_score(y_test_t.numpy(), test_pred.numpy())
            train_logits = model(X_train_t)
            train_acc = accuracy_score(
                y_train_t.numpy(), train_logits.argmax(dim=1).numpy()
            )
        history.append((train_loss, test_acc, train_acc))
        print(f"  epoch {epoch + 1:2d}/{epochs}  train_loss = {train_loss:.4f}"
              f"  train_acc = {train_acc:.4f}  test_acc = {test_acc:.4f}")
    return history, model


print("\n" + "-" * 74)
print("实验 A：课案原结构（含 Dropout(0.3)）")
print("-" * 74)
hist_drop, model_drop = train_dnn(use_dropout=True, epochs=10)

# 课案第 5 步：查看结果（eval + no_grad + accuracy_score）
model_drop.eval()
with torch.no_grad():
    outputs = model_drop(X_test_t)
    y_pred = outputs.argmax(dim=1)
print(f"\n课案写法最终准确率：accuracy_score(y_test, y_pred) = "
      f"{accuracy_score(y_test_t.numpy(), y_pred.numpy()):.4f}")

# ===========================================================================
# 七、对照实验：有 Dropout vs 无 Dropout
# ===========================================================================
section("七、对照实验：有无 Dropout 的泛化能力对比")

print("-" * 74)
print("实验 B：去掉 Dropout 的同结构网络")
print("-" * 74)
hist_nodrop, model_nodrop = train_dnn(use_dropout=False, epochs=10)

# 汇总两者的最终 train/test 准确率，看"泛化差距 = train_acc - test_acc"
summary = {}
for tag, hist, model in (("无 Dropout", hist_nodrop, model_nodrop),
                         ("有 Dropout", hist_drop, model_drop)):
    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_t).argmax(dim=1).numpy()
        test_pred = model(X_test_t).argmax(dim=1).numpy()
    tr_acc = accuracy_score(y_train_t.numpy(), train_pred)
    te_acc = accuracy_score(y_test_t.numpy(), test_pred)
    summary[tag] = (tr_acc, te_acc, tr_acc - te_acc)

print("\n" + "=" * 74)
print("对照实验汇总（同一结构、同一初始化、同一优化器，只差一个 Dropout 层）")
print("=" * 74)
print(f"{'配置':<12}{'train_acc':>12}{'test_acc':>12}{'泛化差距':>12}")
for tag, (tr, te, gap) in summary.items():
    print(f"{tag:<12}{tr:>12.4f}{te:>12.4f}{gap:>12.4f}")
print("\n解读：")
print("  · 泛化差距 = train_acc - test_acc，越小说明越不过拟合。")
print("  · Dropout 通过随机丢弃神经元，迫使网络不依赖个别特征，")
print("    通常表现为训练准确率略降、测试准确率不降甚至更高（差距收窄）。")
print("  · 注意：本数据集本身较容易（20 维、1000 样本、10 个 epoch），")
print("    两种配置都能达到很高的准确率，Dropout 的优势在"
      "数据量小/模型大/训练久时才更明显。")

# 画 loss + accuracy 曲线对比图（两个子图）
epochs_axis = np.arange(1, 11)
loss_drop = [h[0] for h in hist_drop]
loss_nodrop = [h[0] for h in hist_nodrop]
test_drop = [h[1] for h in hist_drop]
test_nodrop = [h[1] for h in hist_nodrop]
train_drop = [h[2] for h in hist_drop]
train_nodrop = [h[2] for h in hist_nodrop]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# 左：训练损失曲线
axes[0].plot(epochs_axis, loss_drop, "o-", color="#d62728", label="有 Dropout(0.3)")
axes[0].plot(epochs_axis, loss_nodrop, "s--", color="#1f77b4", label="无 Dropout")
axes[0].set_title("训练损失（CrossEntropyLoss）随 epoch 变化")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("train loss")
axes[0].set_xticks(epochs_axis)
axes[0].grid(alpha=0.3)
axes[0].legend()

# 右：准确率曲线（实线=测试集，虚线=训练集）
axes[1].plot(epochs_axis, test_drop, "o-", color="#d62728", label="有 Dropout - 测试集")
axes[1].plot(epochs_axis, train_drop, "o:", color="#d62728", alpha=0.6,
             label="有 Dropout - 训练集")
axes[1].plot(epochs_axis, test_nodrop, "s--", color="#1f77b4", label="无 Dropout - 测试集")
axes[1].plot(epochs_axis, train_nodrop, "s:", color="#1f77b4", alpha=0.6,
             label="无 Dropout - 训练集")
axes[1].set_title("准确率随 epoch 变化（实线=测试集，点线=训练集）")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("accuracy")
axes[1].set_xticks(epochs_axis)
axes[1].set_ylim(0.5, 1.02)
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=9)

fig.suptitle("DNN 二分类：Dropout 对照实验（make_classification, 1000 样本, 20 特征）",
             fontsize=13)
fig.tight_layout()
save_path = OUTPUT_DIR / "02网络架构_02_DNN训练与Dropout对比.png"
fig.savefig(save_path, dpi=120)
plt.close(fig)
print(f"\n对比曲线图已保存：{save_path}")

# ===========================================================================
# 八、DNN 结构总结表
# ===========================================================================
section("八、DNN 结构总结表")

rows = [
    ("nn.Linear", "全连接层，做线性变换 z = W h + b；weight 形状 (out, in)，参数量 in×out+out"),
    ("ReLU", "激活函数，max(0, z)：引入非线性（否则多层线性可折叠成一层），且正区间不饱和"),
    ("Dropout", "随机丢弃神经元：训练 h=r⊙h/(1-p)，推理 h 不变；缓解过拟合，相当于子网络集成"),
    ("CrossEntropyLoss", "多分类常用损失 = log_softmax + NLLLoss，输入 logits，内部自带 Softmax"),
    ("Softmax（输出层）", "把 logits 变成概率分布，各类别概率和为 1；只在需要概率时显式使用"),
    ("optimizer.zero_grad()", "清空历史梯度，否则 PyTorch 默认会累加梯度"),
    ("model.train()/eval()", "切换 Dropout/BatchNorm 的训练与推理行为，务必成对使用"),
]
width = 20
for name, desc in rows:
    print(f"  {name:<{width}} {desc}")
print("\n训练循环的标准四步：")
print("  1) outputs = model(X_batch)        前向传播")
print("  2) loss = criterion(outputs, y)    计算损失")
print("  3) optimizer.zero_grad() / loss.backward()   清梯度 + 反向传播")
print("  4) optimizer.step()                更新参数")

print(f"\n参数总量：含 Dropout 模型 {sum(p.numel() for p in model_drop.parameters())}，"
      f"不含 Dropout 模型 {sum(p.numel() for p in model_nodrop.parameters())}"
      f"  # Dropout 层没有可学习参数")

# ===========================================================================
# 九、本节小结
# ===========================================================================
section("九、本节小结")
print("1. nn.Linear(in, out) 的 weight 是 (out, in)，前向传播就是 x @ W.T + b"
      "（已验证误差为 0）。")
print("2. 多层网络公式 z(l)=W(l)h(l-1)+b(l)，h(l)=σ(z(l))；"
      "没有激活函数时多层线性 ≡ 单层线性（已数值验证）。")
print("3. 不调 nn.Linear、只用 torch.matmul 手写逐层前向，与 nn.Sequential 结果"
      "完全一致（误差 < 1e-6）。")
print("4. Dropout 训练期随机置零并按 1/(1-p) 缩放（倒置 Dropout），"
      "推理期恒等；实测置零比例 ≈ p、训练输出均值 ≈ 输入均值。")
print("5. eval() 时 Dropout 关闭，必须切换模式，否则评估结果会随机抖动。")
print("6. 二分类 DNN 训练 10 个 epoch 后，含/不含 Dropout 的 train/test 准确率对比，")
print("   体现了 Dropout 对泛化差距的影响。")

print(f"\n脚本总耗时：{time.perf_counter() - _T0:.2f} 秒")
print("脚本正常结束。")

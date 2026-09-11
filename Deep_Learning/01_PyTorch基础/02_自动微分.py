"""
对应课案章节：PyTorch / 自动微分

本节知识点：
    1.  自动微分（Autograd）要解决什么问题：手推解析梯度在深层网络里不可行，交给框架按链式法则做
    2.  `requires_grad=True` 的含义；课案例子 y = x^2 + 3x + 1，梯度解析解 2x + 3 = [5, 7, 9]
    3.  计算图（computational graph）：`grad_fn` 是什么，`next_functions` 指向谁，叶子节点 vs 非叶子节点
    4.  `requires_grad_(True)`：原地（in-place）打开/关闭梯度追踪；为什么只有浮点张量能开梯度
    5.  `grad` 的**累加**特性：连续两次 backward 梯度翻倍（数值证明）+ `zero_grad()` 的必要性
    6.  `retain_graph=True`：为什么默认 backward 之后计算图就被释放，什么场景需要保留
    7.  `torch.no_grad()` 与 `detach()` 的区别：detach 共享内存但脱离图，no_grad 是「上下文不建图」
    8.  用 `torch.no_grad()` 做推理：省显存、避免建图；以及推理时必须 `.eval()` 的另一半原因
    9.  线性回归的解析梯度手算（∇w = 2Xᵀ(Xw+b−y)/n）与 autograd 结果对比，验证自动微分正确性
    10. `torch.autograd.grad`：只求梯度不累加到 .grad，还能一次求多个输入的梯度
    11. 高阶导数：`create_graph=True` 让梯度本身也可导，从而得到二阶导（含解析解对比）
    12. 极简反向传播对比实验：用 numpy 手写两层网络的前向 + 反向，和 autograd 结果比最大误差
    13. 画图：y = x² 与其 autograd 求出的导数曲线、两次 backward 导致梯度累加的柱状对比

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Deep_Learning\\01_PyTorch基础\\02_自动微分.py'
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
# 正式内容开始
# ---------------------------------------------------------------------------
import numpy as np
import torch
import warnings

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)

print("=" * 78)
print("02 自动微分：requires_grad / backward / grad_fn / no_grad / detach / 高阶导数")
print("=" * 78)
print(f"PyTorch 版本：{torch.__version__}    CUDA 可用：{torch.cuda.is_available()}")
print()


# ===========================================================================
# 【1】课案例子：一元多项式求导，与解析解对比
# ===========================================================================
print("-" * 78)
print("【1】课案例子：y = x^2 + 3x + 1，梯度应为 2x + 3 = [5, 7, 9]")
print("-" * 78)

# ---- 原理 ----
# 自动微分的核心是**链式法则**（chain rule）：
#     若 y = f(u), u = g(x)，则 dy/dx = (dy/du) · (du/dx)
# 对任意复杂的复合函数，只要每一步基本运算的局部导数已知，
# 从输出往输入反向逐层相乘，就能得到对任意输入的导数。这就是「反向模式自动微分」。
#
# 本节的函数：y = x^2 + 3x + 1
#     解析导数：dy/dx = 2x + 3
#     代入 x = [1, 2, 3] → [5, 7, 9]
# 这是检验 autograd 是否正确的最简例子：结果必须和手算完全一致。
#
# 为什么需要 requires_grad=True？
#     PyTorch 默认不记录任何运算（因为记录计算图有内存和时间开销）。
#     只有显式标记 requires_grad=True 的**叶子张量**，它参与过的运算才会被记录进计算图，
#     backward() 才会为它计算并填充 .grad。这相当于告诉框架：「它是我要优化的参数」。

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 前向计算：每一步基本运算都会在计算图里留下一个节点（grad_fn）
y = x ** 2 + 3 * x + 1

# 标量化：backward() 只能从**标量**出发（因为梯度本质是「输出对输入的偏导」，
# 输出必须是单个数值才有定义）。y 是向量，所以用 .sum() 把所有分量加起来。
# 若有 y = [y1, y2, y3]，则 d(sum(y))/dx_i = dy_i/dx_i，所以对 x 的梯度正好就是每个 y_i 自己的导数。
loss = y.sum()

# 反向传播：从 loss 出发，沿计算图反向走，用链式法则把局部导数逐层相乘
loss.backward()

# 解析解
analytic = 2 * torch.tensor([1.0, 2.0, 3.0]) + 3

print(f"x                      = {x.tolist()}")
print(f"y = x**2 + 3x + 1      = {y.tolist()}    （1+3+1=5, 4+6+1=11, 9+9+1=19）")
print(f"loss = y.sum()         = {loss.item():.1f}    ← backward 必须从标量出发")
print(f"x.grad（autograd 结果） = {x.grad.tolist()}")
print(f"解析解 2x + 3           = {analytic.tolist()}")
print(f"两者完全一致：{torch.equal(x.grad, analytic)}")
print()


# ===========================================================================
# 【2】计算图：grad_fn / next_functions / 叶子节点
# ===========================================================================
print("-" * 78)
print("【2】计算图：grad_fn 记录「我由哪个操作产生」")
print("-" * 78)

# ---- 原理 ----
# PyTorch 采用「动态图（define-by-run）」：代码执行到哪，图就建到哪。
# 每个由运算产生的张量都会带一个 grad_fn 属性，指向产生它的那个「反向函数节点」。
#     叶子节点（leaf）：由用户直接创建的张量（torch.tensor / randn / nn.Parameter）。
#                       只有叶子节点的 .grad 会在 backward 后长期保留。
#     非叶子节点：运算的中间结果。它们的 .grad 默认**不保留**（backward 后被清掉），
#                 因为中间量太多，全存下来会吃掉大量内存；需要时用 retain_grad() 单独保留。
# grad_fn.next_functions 是一个元组，指向「上游」的输入节点（梯度要往那边传），
# 顺着 next_functions 一路走，就能把整张计算图打印出来——这是调试梯度问题的常用手段。

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
c = a * b          # 乘法节点
d = c + 1.0        # 加法节点
e = d ** 2         # 幂节点（最后的输出）

print(f"a.is_leaf={a.is_leaf}（叶子）, a.grad_fn={a.grad_fn}  ← 叶子节点没有 grad_fn")
print(f"b.is_leaf={b.is_leaf}（叶子）, b.grad_fn={b.grad_fn}")
print(f"c = a * b  → c.is_leaf={c.is_leaf}, c.grad_fn={c.grad_fn}")
print(f"d = c + 1  → d.is_leaf={d.is_leaf}, d.grad_fn={d.grad_fn}")
print(f"e = d ** 2 → e.is_leaf={e.is_leaf}, e.grad_fn={e.grad_fn}")

# 顺着 next_functions 把图的拓扑结构打印出来（只打印节点名，不打印内存地址以免刷屏）
print("\n从输出 e 出发，沿 next_functions 反向遍历计算图：")
print(f"  e  ← {type(e.grad_fn).__name__}")
node = e.grad_fn
depth = 1
while node is not None and depth <= 3:
    nexts = node.next_functions
    names = [type(fn).__name__ if fn is not None else "None(叶子/累加器)" for fn, _ in nexts]
    print(f"  {'    ' * depth}↑ next_functions: {names}")
    # 取第一个上游节点继续往下走，把链条展示完
    node = nexts[0][0] if nexts and nexts[0][0] is not None else None
    depth += 1

e.backward()
print(f"\n反向传播后：a.grad = {a.grad.item():.1f}, b.grad = {b.grad.item():.1f}")
print(f"  手算核对：e = (ab+1)^2, ∂e/∂a = 2(ab+1)·b = 2*(6+1)*3 = 42；∂e/∂b = 2(ab+1)·a = 2*7*2 = 28")
print(f"  中间节点 c = a*b 是叶子吗：{c.is_leaf}，它的 .grad 默认不保留（访问会得到 None）")
print(f"  中间节点 d = c+1 是叶子吗：{d.is_leaf}，它的 .grad 同样不保留（非叶子节点默认如此）")
print("  → 原因：中间结果太多，全存下来会吃掉大量内存；需要时用 .retain_grad() 单独指定。")

# 想保留非叶子节点的梯度，用 retain_grad()
a2 = torch.tensor(2.0, requires_grad=True)
c2 = a2 * 3.0
c2.retain_grad()                # 显式要求保留这个中间结果的梯度
(c2 ** 2).backward()
print(f"\n对中间张量调用 .retain_grad() 后：c2.grad = {c2.grad.item():.1f}（不再为 None）")
print()


# ===========================================================================
# 【3】requires_grad_(True) 与「只有浮点张量能求梯度」
# ===========================================================================
print("-" * 78)
print("【3】requires_grad_(True)：就地打开梯度追踪")
print("-" * 78)

# ---- 原理 ----
# requires_grad 可以在创建时指定，也可以用 requires_grad_(True) 事后原地修改。
# 注意「原地」：带下划线后缀的方法（requires_grad_ / zero_ / add_）都表示 in-place 修改自身。
# 为什么整数张量不能开梯度？因为梯度是连续量的变化率，需要浮点表示；
# 整数张量上做微小扰动没有意义（离散跳变），所以 PyTorch 直接禁止。

w = torch.randn(3)
print(f"创建时：w.requires_grad = {w.requires_grad}, w.grad_fn = {w.grad_fn}")

w.requires_grad_(True)          # 原地打开
print(f"after requires_grad_(True)：w.requires_grad = {w.requires_grad}")

out = (w * 2).sum()
out.backward()
print(f"w.grad = {[round(v, 4) for v in w.grad.tolist()]}（应全为 2.0：d(2Σw)/dw = 2）")

# 冻结参数：迁移学习里常把预训练层的 requires_grad 设成 False，只训最后几层
w.requires_grad_(False)
print(f"after requires_grad_(False)：w.requires_grad = {w.requires_grad}（参数被冻结，不再更新）")

# 整数张量无法求梯度（用 try/except 捕获，只打印中文说明）
int_t = torch.tensor([1, 2, 3])
try:
    int_t.requires_grad_(True)
    print("整数张量竟然能开梯度？不应该发生")
except RuntimeError:
    print("对整数张量调用 requires_grad_(True)：PyTorch 会报错。")
    print("  → 已捕获。原因：梯度是连续变化率，必须用浮点表示；需要时先 .float() 转成浮点。")

# 检查整个计算图是否需要梯度（调试「模型参数没被训练」时的第一招）
print(f"\n(w*2).sum() 是否追踪梯度：{out.requires_grad}")
print(f"w.detach().requires_grad 状态：{w.detach().requires_grad}（detach 后不追踪）")
print()


# ===========================================================================
# 【4】grad 的累加特性 + zero_grad() 的必要性（数值证明）
# ===========================================================================
print("-" * 78)
print("【4】grad 累加特性：连续两次 backward → 梯度翻倍（这才是 zero_grad 的原因）")
print("-" * 78)

# ---- 原理 ----
# PyTorch 的 .grad 是**累加**（+=）而不是覆盖（=）。设计原因：
#   1) 多任务/多损失场景需要把不同 loss 的梯度加起来（loss = loss1 + loss2 时很方便）；
#   2) 梯度累积（gradient accumulation）技巧：显存小、放不下大 batch 时，
#      连续做几个小 batch 的 backward（不 zero_grad），攒够梯度再 step 一次，
#      效果等价于大 batch，这正是靠累加实现的。
# 但**普通训练循环**里，如果每轮不清零，梯度就会一轮轮累加，参数更新量爆炸。
# 所以课案那三行里的第一行 `optimizer.zero_grad()` 绝对不能少。
#
# 本节用数值证明：同一份计算图 backward 两次，梯度就变成两倍。

x_acc = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

y1 = (x_acc ** 2).sum()             # Σx²，解析梯度 2x = [2, 4, 6]
y1.backward()
grad_after_first = x_acc.grad.clone()
print(f"第一次 backward 后 x.grad = {grad_after_first.tolist()}（解析解 2x = [2,4,6]）")

# 关键：这里**故意不调用 zero_grad()**，直接再来一次
y2 = (x_acc ** 2).sum()
y2.backward()
grad_after_second = x_acc.grad.clone()
print(f"第二次 backward 后 x.grad = {grad_after_second.tolist()}（变成 2 倍！）")
print(f"grad 是否等于第一次的 2 倍：{torch.allclose(grad_after_second, 2 * grad_after_first)}")

# 两种清零方式（等价）：
#   x.grad.zero_()        —— 直接把已有梯度置 0
#   x.grad = None         —— 丢掉梯度对象，下次 backward 重新分配（更省内存，PyTorch 推荐）
x_acc.grad.zero_()                  # in-place 置零
print(f"x.grad.zero_() 后 x.grad = {x_acc.grad.tolist()}  ← 清零了")

y3 = (x_acc ** 2).sum()
y3.backward()
print(f"清零后再 backward：x.grad = {x_acc.grad.tolist()}  ← 回到 [2,4,6]，不再累加")

x_acc.grad = None                   # 清空的另一种写法（推荐）
print(f"把 x.grad 设为 None 后：x.grad = {x_acc.grad}")

# 用优化器演示「忘了 zero_grad 会怎样」——参数更新量翻倍
param = torch.tensor([1.0], requires_grad=True)
optimizer_demo = torch.optim.SGD([param], lr=0.1)

loss_a = (param ** 2).sum()
loss_a.backward()
optimizer_demo.step()                                   # 正确流程：backward → step
print(f"\n正确流程：param 从 1.0 → {param.item():.4f}（梯度 2.0，lr 0.1，所以减 0.2）")

loss_b = (param ** 2).sum()
loss_b.backward()                                       # 这里 .grad 已经因为上一步是 None 而重新计算
optimizer_demo.step()
print(f"再走一轮（上一轮 step 不会自动清梯度，但 zero_grad 缺失会累积）：param → {param.item():.4f}")

# 显式演示「不清零」的后果：连续两次 backward 再 step，步长翻倍
param2 = torch.tensor([1.0], requires_grad=True)
opt2 = torch.optim.SGD([param2], lr=0.1)
(param2 ** 2).sum().backward()                          # 第一次：grad = 2.0
(param2 ** 2).sum().backward()                          # 故意不清零：grad 累加成 4.0
print(f"连续两次 backward 未清零：param2.grad = {param2.grad.item():.1f}（应为 2.0+2.0=4.0）")
opt2.step()
print(f"opt2.step() 后 param2 = {param2.item():.4f}（更新量 0.4，是不清零导致的 2 倍）")
print()


# ===========================================================================
# 【5】retain_graph=True
# ===========================================================================
print("-" * 78)
print("【5】retain_graph=True：默认 backward 后计算图会被释放")
print("-" * 78)

# ---- 原理 ----
# 反向传播需要用到前向时保存的中间结果（例如乘法节点的两个输入）。
# 为了省内存，PyTorch 在 backward 结束后会**释放**这些中间缓存（free_graph）。
# 所以再对同一个 loss 调一次 backward 会报「Trying to backward through the graph a second time」。
# 什么时候需要 retain_graph=True？
#   1) 想多次反传同一个 loss（如对同一 loss 求不同输入的梯度）；
#   2) 需要求高阶导数时，梯度计算图本身还要保留（见第 10 节，用 create_graph=True）。
# 注意：retain_graph=True 会持续占内存，确认不再反传后应让它自然释放。

p = torch.tensor([1.0, 2.0], requires_grad=True)
q = (p ** 2).sum()

q.backward()                                    # 第一次：正常
print(f"第一次 backward 成功，p.grad = {p.grad.tolist()}")

# 第二次对同一张图 backward —— 默认会失败
try:
    q.backward()
    print("第二次 backward 竟然成功了？不应该发生")
except RuntimeError as exc:
    # 只打印异常类型和一句中文说明，绝不输出异常栈
    print(f"第二次对同一张图 backward：抛出 {type(exc).__name__}")
    print("  → 已捕获。原因：默认 backward 后计算图已被释放，中间缓存没了，无法再次反传。")

# 正确做法一：用 retain_graph=True
p2 = torch.tensor([1.0, 2.0], requires_grad=True)
q2 = (p2 ** 2).sum()
q2.backward(retain_graph=True)                  # 保留计算图
g_first = p2.grad.clone()
q2.backward(retain_graph=True)                  # 第二次也能成功
g_second = p2.grad.clone()
print(f"\nretain_graph=True 时：第一次 p2.grad = {g_first.tolist()}，第二次 = {g_second.tolist()}（累加成 2 倍）")
print(f"第二次仍是第一次的 2 倍：{torch.allclose(g_second, 2 * g_first)}")

# 正确做法二：每次重新做一遍前向（最常见、最省内存）
p3 = torch.tensor([1.0, 2.0], requires_grad=True)
g_a = torch.autograd.grad((p3 ** 2).sum(), p3)[0]     # autograd.grad 不写入 .grad，见第 9 节
g_b = torch.autograd.grad((p3 ** 2).sum(), p3)[0]     # 重新前向 → 新图
print(f"\n用 torch.autograd.grad 各自重新前向：g_a = {g_a.tolist()}, g_b = {g_b.tolist()}（都是 [2,4]，不累加）")
print()


# ===========================================================================
# 【6】detach() 与 torch.no_grad() 的区别
# ===========================================================================
print("-" * 78)
print("【6】detach() vs torch.no_grad()：都「不建图」，但不是一回事")
print("-" * 78)

# ---- 原理 ----
# 两者都能阻止梯度追踪，但作用层面不同：
#
#   tensor.detach()
#       —— 返回一个**新的张量对象**，它和原张量**共享同一块底层内存**（storage），
#          但 requires_grad=False，且 grad_fn=None，即从计算图上「剪断」。
#          后续用这个新张量做的运算都不会被记录。
#          常用于：把张量转成 numpy、记录指标（loss 值）、或者做「目标值」防止梯度回流。
#
#   with torch.no_grad():
#       —— 是一个**上下文管理器**，管的是「这一段代码里所有的运算都不建图」。
#          不产生新张量，原来的张量还是原来的张量（requires_grad 依旧是 True），
#          只是在这一段里执行的操作不会被记录。
#          常用于：评估/推理、参数手动更新（如 optimizer.step 内部就包在 no_grad 里）、
#                  以及计算指标时避免建图吃内存。
#
# 一句话：detach 剪断**一个张量**的图；no_grad 关掉**一段代码**的图。

t = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
u = t * 2                       # 逐元素乘，u 是向量，方便下面用下标演示共享内存
print(f"u.requires_grad = {u.requires_grad}, u.grad_fn = {u.grad_fn}")

d = u.detach()
print(f"u.detach() → requires_grad={d.requires_grad}, grad_fn={d.grad_fn}")
print(f"  共享内存验证：u.data_ptr()={u.data_ptr()}，d.data_ptr()={d.data_ptr()}，相同={u.data_ptr() == d.data_ptr()}")
d[0] = 999.0                                       # 修改 detach 出来的张量……
print(f"  修改 d[0]=999 后 u[0]={u[0].item():.1f}  ← 原张量也跟着变了，证明共享内存（这是坑！）")

# no_grad 上下文：不产生新张量，原张量身份不变
v = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"\n进入 no_grad 之前：v.requires_grad = {v.requires_grad}")
with torch.no_grad():
    w_nograd = v * 5
    print(f"no_grad 内部：v.requires_grad = {v.requires_grad}（张量本身没变）")
    print(f"  (v*5).requires_grad = {w_nograd.requires_grad}, grad_fn = {w_nograd.grad_fn}  ← 这个运算没被记录")
w_out = v * 5
print(f"离开 no_grad 之后再做同样运算：(v*5).requires_grad = {w_out.requires_grad}, grad_fn = {w_out.grad_fn}")

# 用 no_grad 做推理：更省内存
net = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 2))
sample = torch.randn(3, 4)

pred_with_graph = net(sample)                       # 建图
with torch.no_grad():                               # 不建图
    pred_no_graph = net(sample)
print(f"\n推理对比：建图输出 requires_grad={pred_with_graph.requires_grad}，no_grad 输出 requires_grad={pred_no_graph.requires_grad}")
print(f"两者数值完全一致：{torch.allclose(pred_with_graph, pred_no_graph)}")

# 评估模式 eval() 与 no_grad() 是两件事，缺一不可：
#   eval()     —— 切换层的**行为**（Dropout 关闭、BatchNorm 用滑动统计量）
#   no_grad()  —— 关闭**梯度记录**（省内存、提速）
# 它们互不替代，推理时应同时使用。
net.eval()
with torch.no_grad():
    _ = net(sample)
print("推理规范写法：model.eval() + with torch.no_grad():（一个管行为，一个管梯度）")
print()


# ===========================================================================
# 【7】用 autograd 验证线性回归的解析梯度
# ===========================================================================
print("-" * 78)
print("【7】线性回归解析梯度 vs autograd 自动微分")
print("-" * 78)

# ---- 原理 ----
# 线性回归：ŷ = Xw + b，损失用均方误差
#     L = (1/(2n)) · Σ_i (ŷ_i − y_i)²
# 对 w 和 b 求偏导（链式法则 + 求和）：
#     令残差 r = ŷ − y (形状 (n,1))
#     ∂L/∂w = (1/n) · Xᵀ r        （形状 (d,1)）
#     ∂L/∂b = (1/n) · Σ r          （标量，等于 r.mean()）
# 这里系数取 1/2n 是为了求导后正好消掉平方的 2，得到干净的 1/n。
# 下面同时用「手算解析式」和「autograd」求梯度，验证两者一致 —— 这就是自动微分的正确性证据。

n_samples, n_features = 20, 3
X_np = np.random.randn(n_samples, n_features).astype(np.float64)
w_true = np.array([[1.5], [-2.0], [0.5]])
b_true = 0.7
y_np = X_np @ w_true + b_true + 0.01 * np.random.randn(n_samples, 1)   # 加一点噪声

X_t = torch.tensor(X_np)
y_t = torch.tensor(y_np)
w = torch.zeros(n_features, 1, dtype=torch.float64, requires_grad=True)
b = torch.zeros(1, dtype=torch.float64, requires_grad=True)

# 前向
y_hat = X_t @ w + b
residual = y_hat - y_t
loss = (residual ** 2).mean()          # 等价于 (1/n)Σr²，梯度会带系数 2/n
loss.backward()

# 手算解析梯度（上面 loss 用的是 mean(r²)，所以 ∂L/∂w = (2/n)·Xᵀr）
residual_np = residual.detach().numpy()
grad_w_manual = 2.0 * X_np.T @ residual_np / n_samples
grad_b_manual = 2.0 * residual_np.mean()

print(f"X shape={tuple(X_t.shape)}, y shape={tuple(y_t.shape)}, w shape={tuple(w.shape)}")
print(f"loss = mean((Xw+b−y)²) = {loss.item():.8f}")
print(f"手算解析梯度 grad_w = {[round(float(v), 8) for v in grad_w_manual.ravel()]}")
print(f"autograd 的 w.grad = {[round(v, 8) for v in w.grad.ravel().tolist()]}")
print(f"手算解析梯度 grad_b = {grad_b_manual.item():.8f}")
print(f"autograd 的 b.grad = {b.grad.item():.8f}")
print(f"w 的最大误差 = {np.abs(w.grad.numpy() - grad_w_manual).max():.3e}（float64 下应在 1e-15 量级）")
print(f"b 的绝对误差 = {abs(b.grad.item() - grad_b_manual.item()):.3e}")
print("结论：autograd 的梯度与手推解析式完全吻合，可以放心把它当成「永不写错的反向传播」。")
print()


# ===========================================================================
# 【8】torch.autograd.grad
# ===========================================================================
print("-" * 78)
print("【8】torch.autograd.grad：只取梯度，不写进 .grad")
print("-" * 78)

# ---- 原理 ----
# tensor.backward() 会把梯度**累加**到各个叶子张量的 .grad 上（副作用）。
# torch.autograd.grad(outputs, inputs, ...) 则**直接返回**梯度，不修改 .grad，特点：
#     · 不会污染参数已有的 .grad（写元学习、梯度惩罚、对抗样本时很重要）
#     · 可以一次对多个 inputs 求梯度（返回一个元组）
#     · 常用参数：
#         grad_outputs —— 当 outputs 不是标量时，必须提供「上游梯度」（等价于对输出做加权求和）
#         retain_graph  —— 是否保留图，求多次梯度时用
#         create_graph  —— 是否对「梯度」本身再建图（求高阶导用）
#         allow_unused  —— 允许某些 input 与 output 无关（此时返回 None 而不是报错）

z = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
f = (z ** 3).sum()                       # Σz³，解析梯度 3z² = [3, 12, 27]

(g_z,) = torch.autograd.grad(f, z)       # 注意返回的是元组，取 [0]
print(f"z.grad 在 grad() 之后仍为：{z.grad}（没有被写入）")
print(f"torch.autograd.grad 返回的梯度 = {g_z.tolist()}，解析解 3z² = [3, 12, 27]")

# 一次求多个输入的梯度
m = torch.tensor([2.0], requires_grad=True)
n = torch.tensor([5.0], requires_grad=True)
out_mn = (m * n + m ** 2).sum()          # ∂/∂m = n + 2m = 5+4 = 9；∂/∂n = m = 2
g_m, g_n = torch.autograd.grad(out_mn, [m, n])
print(f"\n一次求两个输入的梯度：∂out/∂m = {g_m.item():.1f}（=n+2m=9）, ∂out/∂n = {g_n.item():.1f}（=m=2）")

# 非标量输出必须给 grad_outputs（指定每个输出分量的权重）
vec_out = torch.tensor([1.0, 2.0, 3.0], requires_grad=True) ** 2      # (3,)
weights = torch.tensor([1.0, 1.0, 1.0])
(g_vec,) = torch.autograd.grad(vec_out, vec_out, grad_outputs=weights)  # 给全 1 权重 ≡ sum
print(f"\n非标量输出 + grad_outputs=全 1：得到 {g_vec.tolist()}（≡ 对 sum 求导）")

(gu,) = torch.autograd.grad(vec_out, vec_out, grad_outputs=torch.tensor([1.0, 0.0, 0.0]))
print(f"grad_outputs=[1,0,0]：得到 {gu.tolist()}（只对第 0 个分量求导）")

# allow_unused：允许某个输入和输出无关（注意上一次 grad 调用已释放计算图，这里用 out_mn2 重新前向）
m2 = torch.tensor([2.0], requires_grad=True)
n2 = torch.tensor([5.0], requires_grad=True)
unrelated = torch.tensor([1.0], requires_grad=True)
out_mn2 = (m2 * n2 + m2 ** 2).sum()
(g_used, g_unused) = torch.autograd.grad(out_mn2, [m2, unrelated], allow_unused=True)
print(f"\nallow_unused=True：与输出无关的输入梯度返回 {g_unused}（None，而不是报错）")
print(f"                  与输出有关的输入梯度 g_used = {g_used.item():.1f}（=n+2m=9）")
print()


# ===========================================================================
# 【9】高阶导数：create_graph=True
# ===========================================================================
print("-" * 78)
print("【9】高阶导数：create_graph=True 让「梯度」本身也可导")
print("-" * 78)

# ---- 原理 ----
# 一阶导只是「把 backward 走一遍」。如果想再对一阶导求导（二阶导、海森矩阵、梯度惩罚 WGAN-GP），
# 就必须让「求梯度的过程」本身也被记录成一张图，这就是 create_graph=True 的作用。
# 之后对 first_grad 再调用 backward / autograd.grad，就能得到二阶导。

# 例子：f(x) = x³ → f'(x) = 3x² → f''(x) = 6x
xh = torch.tensor([2.0], requires_grad=True)
fh = xh ** 3
(fh_first,) = torch.autograd.grad(fh, xh, create_graph=True)     # 保留导数图
print(f"f(x)=x³ 在 x=2 处：一阶导 f'(2) = {fh_first.item():.4f}（解析 3x²=12）")

(fh_second,) = torch.autograd.grad(fh_first, xh)                 # 对一阶导再求导
print(f"                  二阶导 f''(2) = {fh_second.item():.4f}（解析 6x=12）")

# 更复杂的例子：f(x) = sin(x)·x²（x 用 numpy 生成再转张量）
xhs = torch.tensor([0.5, 1.0, 1.5], dtype=torch.float64, requires_grad=True)
fhs = (torch.sin(xhs) * xhs ** 2).sum()
(g1,) = torch.autograd.grad(fhs, xhs, create_graph=True)
(g2,) = torch.autograd.grad(g1.sum(), xhs)

# 解析解：d/dx (sin x · x²) = cos x · x² + 2x·sin x
#          d²/dx² = −sin x·x² + 2x·cos x + 2 sin x + 2x cos x = −x² sin x + 4x cos x + 2 sin x
x_np_h = xhs.detach().numpy()
an1 = np.cos(x_np_h) * x_np_h ** 2 + 2 * x_np_h * np.sin(x_np_h)
an2 = -(x_np_h ** 2) * np.sin(x_np_h) + 4 * x_np_h * np.cos(x_np_h) + 2 * np.sin(x_np_h)

print(f"\nf(x)=sin(x)·x²，x = {[round(v, 2) for v in x_np_h.tolist()]}")
print(f"  一阶导 autograd = {[round(v, 6) for v in g1.tolist()]}")
print(f"  一阶导 解析解   = {[round(v, 6) for v in an1.tolist()]}    最大误差 {np.abs(g1.detach().numpy() - an1).max():.2e}")
print(f"  二阶导 autograd = {[round(v, 6) for v in g2.tolist()]}")
print(f"  二阶导 解析解   = {[round(v, 6) for v in an2.tolist()]}    最大误差 {np.abs(g2.detach().numpy() - an2).max():.2e}")

# 用 backward(create_graph=True) 也能做（区别：会写进 .grad）。
# 注意：PyTorch 会对这种写法提示「可能造成参数与梯度的循环引用、进而内存泄漏」，
#      这是已知的设计取舍说明而非脚本错误，这里把它捕获掉，保持输出干净；
#      工程上更推荐上面用的 torch.autograd.grad(create_graph=True) 写法。
xh2 = torch.tensor([3.0], requires_grad=True)
fh2 = xh2 ** 3
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    fh2.backward(create_graph=True)
first = xh2.grad.clone()                     # = 3x² = 27
xh2.grad = None                              # 关键：清掉一阶导，否则二阶导会和它累加（.grad 是累加的）
(first.sum()).backward()                     # 对一阶导再 backward → 得到 f''=6x
print(f"\nbackward(create_graph=True) 路线：x=3 时 f'={first.item():.2f}(解析 3x²=27)，f''={xh2.grad.item():.2f}(解析 6x=18)")
xh2.grad = None                              # 断开参数与梯度的循环引用，避免内存泄漏（官方建议）
print()


# ===========================================================================
# 【10】极简反向传播对比实验：numpy 手算 vs autograd
# ===========================================================================
print("-" * 78)
print("【10】手写反向传播对比实验：numpy 手算梯度 vs autograd（要求最大误差 < 1e-6）")
print("-" * 78)

# ---- 原理 ----
# 这是本节最有说服力的一段：亲手把两层网络的梯度推一遍，再用 autograd 核对。
# 网络结构（回归任务，MSE 损失）：
#     输入 X (n, d)
#     H1 = X  @ W1 + b1        (n, h)        —— 第 1 层线性
#     A1 = relu(H1)            (n, h)        —— 激活
#     H2 = A1 @ W2 + b2        (n, 1)        —— 第 2 层线性（输出）
#     L  = mean((H2 − y)²)                    —— MSE 损失
#
# 反向传播（链式法则逐层往回）：
#     dL/dH2 = 2(H2 − y)/n                     形状 (n,1)
#     dL/dW2 = A1ᵀ · dL/dH2                    形状 (h,1)
#     dL/db2 = Σ_rows dL/dH2                   标量
#     dL/dA1 = dL/dH2 · W2ᵀ                    形状 (n,h)
#     dL/dH1 = dL/dA1 ⊙ 1[H1>0]                ReLU 的导数就是「正数为 1，否则 0」
#     dL/dW1 = Xᵀ · dL/dH1                     形状 (d,h)
#     dL/db1 = Σ_rows dL/dH1                   形状 (h,)
# 全部用 float64 计算，这样理论误差可以小到 1e-15 量级，远优于要求的 1e-6。

np.random.seed(42)
n_bt, d_in, h_hid = 16, 5, 7

Xb = np.random.randn(n_bt, d_in)
yb = np.random.randn(n_bt, 1)
W1_np = np.random.randn(d_in, h_hid) * 0.5
b1_np = np.zeros((1, h_hid))
W2_np = np.random.randn(h_hid, 1) * 0.5
b2_np = np.zeros((1, 1))

# ---------------- numpy 手写前向 ----------------
H1_np = Xb @ W1_np + b1_np
A1_np = np.maximum(H1_np, 0.0)                  # ReLU
H2_np = A1_np @ W2_np + b2_np
diff_np = H2_np - yb
L_np = (diff_np ** 2).mean()

# ---------------- numpy 手写反向 ----------------
dH2 = 2.0 * diff_np / n_bt                      # dL/dH2
dW2 = A1_np.T @ dH2                             # dL/dW2 = A1ᵀ dH2
db2 = dH2.sum(axis=0, keepdims=True)            # dL/db2
dA1 = dH2 @ W2_np.T                             # dL/dA1
dH1 = dA1 * (H1_np > 0).astype(np.float64)      # ReLU 导数：正数为 1，否则 0
dW1 = Xb.T @ dH1                                # dL/dW1 = Xᵀ dH1
db1 = dH1.sum(axis=0, keepdims=True)            # dL/db1

print("两层网络结构：X → Linear(d→h) → ReLU → Linear(h→1) → MSE")
print(f"形状：X{tuple(Xb.shape)}  W1{tuple(W1_np.shape)}  b1{tuple(b1_np.shape)}  W2{tuple(W2_np.shape)}  b2{tuple(b2_np.shape)}")
print(f"numpy 手算 loss = {L_np.item():.10f}")

# ---------------- autograd 版本（完全同样的结构） ----------------
X_t2 = torch.tensor(Xb, dtype=torch.float64)
y_t2 = torch.tensor(yb, dtype=torch.float64)
W1_t = torch.tensor(W1_np, dtype=torch.float64, requires_grad=True)
b1_t = torch.tensor(b1_np, dtype=torch.float64, requires_grad=True)
W2_t = torch.tensor(W2_np, dtype=torch.float64, requires_grad=True)
b2_t = torch.tensor(b2_np, dtype=torch.float64, requires_grad=True)

H1_t = X_t2 @ W1_t + b1_t
A1_t = torch.relu(H1_t)
H2_t = A1_t @ W2_t + b2_t
L_t = ((H2_t - y_t2) ** 2).mean()
L_t.backward()

print(f"autograd   loss = {L_t.item():.10f}   （两者 loss 之差 {abs(L_t.item() - L_np.item()):.2e}）")

# ---------------- 逐项对比 ----------------
pairs = [
    ("dL/dW1", dW1, W1_t.grad.numpy()),
    ("dL/db1", db1, b1_t.grad.numpy()),
    ("dL/dW2", dW2, W2_t.grad.numpy()),
    ("dL/db2", db2, b2_t.grad.numpy()),
]
print("\n梯度逐项对比（numpy 手写 vs autograd）：")
max_err = 0.0
for name, manual, auto in pairs:
    err = float(np.abs(manual - auto).max())
    max_err = max(max_err, err)
    print(f"  {name:<8} 形状={str(tuple(manual.shape)):<10} 手算={float(manual.ravel()[0]):+.10f}  autograd={float(auto.ravel()[0]):+.10f}  最大误差={err:.3e}")

print(f"\n>>> 四组梯度的全局最大误差 = {max_err:.3e}")
print(f">>> 是否满足 < 1e-6 的要求：{max_err < 1e-6}")
print("结论：autograd 的结果和我们手推的解析梯度一致，验证了自动微分的正确性。")
print("      这正是框架能放心用于深层网络的原因——反向传播不用人写，但数学和手算完全一样。")
print()


# ===========================================================================
# 【11】绘图：y = x² 与其导数曲线 + 梯度累加演示
# ===========================================================================
print("-" * 78)
print("【11】绘图：y = x² 及导数曲线，以及梯度累加的柱状对比")
print("-" * 78)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
fig.suptitle("自动微分：导数曲线与梯度累加特性", fontsize=15, fontweight="bold")

# ---- 子图 1：y = x² 与 dy/dx = 2x ----
ax = axes[0]
x_line = torch.linspace(-3, 3, 200, requires_grad=True)
y_line = x_line ** 2
(dy_dx,) = torch.autograd.grad(y_line.sum(), x_line)     # autograd 求出的导数
x_np_line = x_line.detach().numpy()
y_np_line = y_line.detach().numpy()
dy_np = dy_dx.detach().numpy()

ax.plot(x_np_line, y_np_line, color="#1f77b4", lw=2.2, label=r"$y = x^2$")
ax.plot(x_np_line, dy_np, color="#d62728", lw=2.0, ls="--", label=r"$\mathrm{d}y/\mathrm{d}x = 2x$（autograd）")
# 标出 x=1,2,3 三个点（课案例子里用的输入）
for xi in (1.0, 2.0, 3.0):
    ax.plot([xi], [xi ** 2], "o", color="#1f77b4", ms=6)
    ax.plot([xi], [2 * xi], "s", color="#d62728", ms=5)
    ax.annotate(f"({xi:.0f}, {2 * xi:.0f})", (xi, 2 * xi), textcoords="offset points",
                xytext=(4, -14), fontsize=8.5, color="#d62728")
ax.axhline(0, color="#999999", lw=0.8)
ax.axvline(0, color="#999999", lw=0.8)
ax.set_title("① $y = x^2$ 与 autograd 算出的导数", fontsize=11)
ax.set_xlabel("x", fontsize=10)
ax.set_ylabel("数值", fontsize=10)
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.25)

# ---- 子图 2：梯度累加（两次 backward 梯度翻倍） ----
ax = axes[1]
xs_bar = np.array([1.0, 2.0, 3.0])
bar_w = 0.35
positions = np.arange(3)
ax.bar(positions - bar_w / 2, grad_after_first.numpy(), bar_w,
       label="第一次 backward 后", color="#4c72b0", edgecolor="white")
ax.bar(positions + bar_w / 2, grad_after_second.numpy(), bar_w,
       label="第二次 backward 后（未 zero_grad）", color="#c44e52", edgecolor="white")
for i in range(3):
    ax.text(positions[i] - bar_w / 2, grad_after_first[i].item() + 0.3,
            f"{grad_after_first[i].item():.0f}", ha="center", fontsize=9, color="#4c72b0")
    ax.text(positions[i] + bar_w / 2, grad_after_second[i].item() + 0.3,
            f"{grad_after_second[i].item():.0f}", ha="center", fontsize=9, color="#c44e52")
ax.set_xticks(positions, [f"x={v:.0f}\n(解析解 2x)" for v in xs_bar], fontsize=9)
ax.set_title("② grad 会累加：不清零就翻倍", fontsize=11)
ax.set_ylabel("x.grad", fontsize=10)
ax.set_ylim(0, 15.5)
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.25, axis="y")
ax.text(0.98, 0.62, "所以\noptimizer.zero_grad()\n不能省", transform=ax.transAxes,
        ha="right", fontsize=9.5, color="#c00000")

# ---- 子图 3：detach / no_grad 对计算图的影响 ----
ax = axes[2]
ax.axis("off")
ax.set_title("③ detach() 与 no_grad() 的作用位置", fontsize=11)
ax.text(0.02, 0.90, "计算图：x → □ → □ → loss", fontsize=11, transform=ax.transAxes, color="#1f4e79")
ax.text(0.02, 0.74, "detach()：剪断「某一个张量」与图的关系", fontsize=10, transform=ax.transAxes)
ax.text(0.06, 0.64, "· 返回新张量对象，requires_grad=False", fontsize=9, transform=ax.transAxes, color="#404040")
ax.text(0.06, 0.55, "· 与原张量共享内存 → 改一个另一个也变（坑）", fontsize=9, transform=ax.transAxes, color="#404040")
ax.text(0.06, 0.46, "· 常用于：转 numpy、记录 loss 数值、构造 target", fontsize=9, transform=ax.transAxes, color="#404040")
ax.text(0.02, 0.30, "no_grad()：关掉「一段代码」的图", fontsize=10, transform=ax.transAxes)
ax.text(0.06, 0.20, "· 是上下文管理器，不动张量本身", fontsize=9, transform=ax.transAxes, color="#404040")
ax.text(0.06, 0.11, "· 常用于：推理评估、手动更新参数、算指标", fontsize=9, transform=ax.transAxes, color="#404040")
ax.text(0.02, 0.01, "推理规范：model.eval() + with torch.no_grad():", fontsize=9.5,
        transform=ax.transAxes, color="#c00000")

plt.tight_layout(rect=[0, 0, 1, 0.93])
save_path = OUTPUT_DIR / "02_自动微分_导数曲线.png"
plt.savefig(save_path, dpi=110, bbox_inches="tight")
plt.close(fig)

print(f"图已保存：{save_path}")
print(f"文件存在：{save_path.exists()}，大小：{save_path.stat().st_size / 1024:.1f} KB")
print()

# ===========================================================================
# 小结
# ===========================================================================
print("=" * 78)
print("小结")
print("=" * 78)
print("· requires_grad=True 的叶子张量才会被记录进计算图，backward() 后梯度写进 .grad")
print("· backward() 只能从标量出发；非标量要用 grad_outputs 指定上游权重")
print("· grad_fn 记录「我由哪个运算产生」，next_functions 指向更上游，可据此打印整张图")
print("· .grad 是累加的：不清零就翻倍 → 训练循环第一行必须 optimizer.zero_grad()")
print("· retain_graph=True 保留计算图以便再次反传；默认反传后图被释放")
print("· detach() 剪断单个张量（共享内存，不建图）；no_grad() 关闭一段代码的建图")
print("· torch.autograd.grad 只返回梯度、不写 .grad；支持多输入 / grad_outputs / allow_unused")
print("· create_graph=True 让梯度本身可导 → 可求二阶导、高阶导")
print(f"· 手写 numpy 反向传播与 autograd 的梯度最大误差 = {max_err:.2e}（< 1e-6，验证正确性）")
print("· 课案三行：optimizer.zero_grad() → loss.backward() → optimizer.step()")
print("=" * 78)
print("脚本执行完毕，退出码 0。")

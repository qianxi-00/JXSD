"""
对应课案章节：PyTorch / 张量

本节知识点：
    1.  张量是什么：可以放在 CPU/GPU 上计算的多维数组，深度学习里一切数据（特征、图片、token）都变成张量
    2.  深度学习中最常见的 5 种张量形状：一条特征向量 / 一批表格数据 / 一张灰度图 / 一批彩色图 / 一批文本 token
    3.  张量的创建：从 Python 列表、从 NumPy 数组（`torch.from_numpy` 共享内存）、指定形状的工厂函数
    4.  工厂函数族：`torch.ones / zeros / full / randn / rand / arange / linspace / eye`
    5.  数据类型 dtype：`torch.tensor` 的默认 dtype 推断规则、`torch.get_default_dtype()`、dtype 转换
    6.  元素级运算（`+`、`*`、`-`、`/`、`**`）与矩阵乘法（`@`、`torch.matmul`）的本质区别
    7.  广播机制（Broadcasting）：从右往左对齐、维度为 1 或缺失就扩展，(3,1)+(1,4) → (3,4)
    8.  形状变换：`reshape` 与 `view` 的区别（view 要求内存连续，否则报错 → `.contiguous().view()`）
    9.  增删维度：`unsqueeze` / `squeeze`，以及它们在「给一批数据补上通道维」这类场景里的用途
    10. 轴变换：`transpose`（只能换两个轴）与 `permute`（一次换多个轴，如 NCHW → NHWC）
    11. 索引与选取：切片、步长、布尔掩码、`index_select` 与 `gather`、`torch.where`
    12. 拼接与切分：`torch.cat`（沿已有维度）与 `torch.stack`（新建维度）的形状差异；`torch.split` / `chunk`
    13. 设备与 CPU/GPU：`torch.cuda.is_available()`、`.to(device)`、本机无 CUDA 时的行为说明
    14. 最后画出一张「形状变换 + 广播」示意图，直观理解维度是怎么对齐、扩展的

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Deep_Learning\\01_PyTorch基础\\01_张量.py'
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
# 正式内容开始：导入 torch / numpy，并固定随机种子
# ---------------------------------------------------------------------------
import numpy as np
import torch

# 固定随机种子是实验可复现的前提：只要种子一样，randn/rand 的序列就完全一样，
# 这样才能把「模型变差」和「随机数不同」这两件事区分开。
torch.manual_seed(42)
np.random.seed(42)

# CPU 版 PyTorch 默认会开满物理核心做 OpenMP 并行。本脚本数据量很小，
# 线程开太多反而被调度开销拖慢，这里限制成 4 个线程，保证耗时可预期。
torch.set_num_threads(4)

# 一个全局的「设备选择」套路：有 GPU 用 GPU，没有就用 CPU。
# 本机装的是 torch 2.14.0+cpu，torch.cuda.is_available() 恒为 False，
# 所以 device 一定是 "cpu"。但真实项目里一定要这样写，代码才能既跑 CPU 又跑 GPU。
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 78)
print("01 张量：创建 / 运算 / 广播 / 形状变换 / 索引 / 拼接 / 设备")
print("=" * 78)
print(f"PyTorch 版本        ：{torch.__version__}")
print(f"torch.cuda.is_available() = {torch.cuda.is_available()}  →  本机为 CPU 版，device = {device}")
print(f"默认浮点 dtype      ：{torch.get_default_dtype()}")
print(f"PyTorch 线程数      ：{torch.get_num_threads()}")
print()


# ===========================================================================
# 小节 0：深度学习里数据长什么样——先建立形状直觉
# ===========================================================================
print("-" * 78)
print("【0】深度学习中最常见的 5 种张量形状")
print("-" * 78)

# 为什么必须先讲形状？因为 90% 的深度学习报错都是形状不匹配（shape mismatch）。
# 神经网络每一层都是在「某些维度」上做线性变换，只有把维度语义记住，
# 才知道该在哪个位置 unsqueeze、在哪个位置 permute。
feature_dim = 10      # 一条样本的特征个数
batch_size = 8        # 一批样本的条数
height, width = 28, 28

# 1) 一条特征向量 (feature_dim,)：一个样本的多个特征，1 维
one_sample = torch.randn(feature_dim)
# 2) 一批表格数据 (batch_size, feature_dim)：多条样本堆起来，2 维
one_batch = torch.randn(batch_size, feature_dim)
# 3) 一张灰度图 (1, H, W)：1 个通道，3 维
gray_image = torch.randn(1, height, width)
# 4) 一批彩色图 (batch_size, 3, H, W)：RGB 三通道，4 维（PyTorch 默认通道在前，即 NCHW）
color_batch = torch.randn(batch_size, 3, height, width)
# 5) 一批文本 token (batch_size, seq_len)：每个位置是一个 token id，用整数 dtype
token_batch = torch.randint(low=0, high=10000, size=(batch_size, 16))

# 用表格形式打印，直观对比「形状」和「维度语义」的对应关系
print(f"{'数据类型':<16}{'张量形状':<26}{'dim()':<8}{'dtype':<16}说明")
shape_rows = [
    ("一条特征向量", one_sample.shape, one_sample.dim(), one_sample.dtype, "一条样本的多个特征"),
    ("一批表格数据", one_batch.shape, one_batch.dim(), one_batch.dtype, "多条样本组成一个 batch"),
    ("一张灰度图", gray_image.shape, gray_image.dim(), gray_image.dtype, "1 个通道"),
    ("一批彩色图", color_batch.shape, color_batch.dim(), color_batch.dtype, "RGB 三通道，NCHW 布局"),
    ("一批文本 token", token_batch.shape, token_batch.dim(), token_batch.dtype, "每个位置是一个 token id"),
]
for name, shape, dim, dtype, desc in shape_rows:
    print(f"{name:<16}{str(tuple(shape)):<26}{dim:<8}{str(dtype):<16}{desc}")

print()
print("关键直觉：维度越靠右越「细」，越靠左越「批量」。")
print("          形状里没有 batch 维（如 (10,)）通常表示单条样本；有 batch 维（如 (8, 10)）表示一批。")
print()


# ===========================================================================
# 小节 1：从 Python 列表创建张量，以及默认 dtype 的推断规则
# ===========================================================================
print("-" * 78)
print("【1】从 Python 列表创建张量 + 默认 dtype 推断")
print("-" * 78)

# torch.tensor(data) 是「拷贝式」创建：它会根据 data 里的元素类型推断 dtype。
# 推断规则（记住这三条就够用了）：
#   全是整数        → torch.int64（长整型，PyTorch 里索引、标签默认用它）
#   出现浮点数      → torch.get_default_dtype()，默认是 torch.float32
#   出现 True/False → torch.bool
x_float = torch.tensor([1.0, 2.0, 3.0])                 # 浮点列表 → float32
x_int = torch.tensor([1, 2, 3])                         # 整数列表 → int64
x_bool = torch.tensor([True, False, True])              # 布尔列表 → bool
x_2d = torch.tensor([[1.0, 2.0], [3.0, 4.0]])           # 二维嵌套列表 → (2, 2) 的 float32

print(f"torch.tensor([1.0, 2.0, 3.0])   → dtype={x_float.dtype}, shape={tuple(x_float.shape)}, 值={x_float.tolist()}")
print(f"torch.tensor([1, 2, 3])         → dtype={x_int.dtype}, shape={tuple(x_int.shape)}, 值={x_int.tolist()}")
print(f"torch.tensor([True, False, True])→ dtype={x_bool.dtype}, shape={tuple(x_bool.shape)}, 值={x_bool.tolist()}")
print(f"torch.tensor([[1., 2.], [3., 4.]])→ dtype={x_2d.dtype}, shape={tuple(x_2d.shape)}")

# 想强制指定类型，就显式传 dtype。这是工程上强烈推荐的做法——
# 让 dtype 显式可见，比依赖推断规则可靠得多（尤其整数/浮点混排时）。
x_forced = torch.tensor([1, 2, 3], dtype=torch.float32)
print(f"显式指定 dtype=torch.float32    → dtype={x_forced.dtype}, shape={tuple(x_forced.shape)}")
print()

# dtype 转换的三种等价写法：.to(dtype) / .float() / .type()
x_to_int = x_forced.to(torch.int64)     # 通用写法：.to(dtype)
print(f"x_forced.to(torch.int64)        → dtype={x_to_int.dtype}, 值={x_to_int.tolist()}（浮点转整数是截断，不是四舍五入）")
print(f"整数张量转浮点：x_int.float()    → dtype={x_int.float().dtype}")
print()


# ===========================================================================
# 小节 2：从 NumPy 数组创建张量（共享内存 vs 拷贝）
# ===========================================================================
print("-" * 78)
print("【2】从 NumPy 创建张量：torch.from_numpy 共享内存")
print("-" * 78)

arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

# torch.from_numpy 不会拷贝数据，张量和 ndarray 指向同一块内存。
# 好处：零拷贝、极快，适合把 numpy 预处理结果直接喂给模型。
# 风险：改一个另一个也变（bug 常见来源），所以对「会被修改的输入」要用 torch.tensor(arr) 拷贝。
y_shared = torch.from_numpy(arr)
y_copied = torch.tensor(arr)            # 拷贝，独立内存（会告警提示，这里故意演示对比）

print(f"np.array              → shape={arr.shape}, dtype={arr.dtype}")
print(f"torch.from_numpy(arr) → shape={tuple(y_shared.shape)}, dtype={y_shared.dtype}, 与 ndarray 共享内存={y_shared.data_ptr() == arr.__array_interface__['data'][0]}")

# 修改 numpy 数组，观察共享内存的效果
arr[0, 0] = 99.0
print(f"把 arr[0,0] 改成 99 后：y_shared[0,0] = {y_shared[0, 0].item():.1f}（跟着变了，证明共享内存）")
print(f"                      y_copied[0,0] = {y_copied[0, 0].item():.1f}（没变，证明是拷贝）")

# 反向转换：张量 → numpy。CPU 张量用 .numpy()，需要梯度或 GPU 张量必须先 detach().cpu()
back_to_np = y_copied.numpy()
print(f"张量 → numpy：y_copied.numpy() → {back_to_np.dtype}, shape={back_to_np.shape}")
print()


# ===========================================================================
# 小节 3：指定形状的工厂函数（ones / zeros / full / randn / rand / arange / linspace / eye）
# ===========================================================================
print("-" * 78)
print("【3】常用创建函数：ones / zeros / full / randn / rand / arange / linspace / eye")
print("-" * 78)

ones = torch.ones(3, 4)                                        # 全 1，常用于构造 mask 或初始化
zeros = torch.zeros(2, 3)                                      # 全 0，常用于初始化偏置或占位
full = torch.full((2, 2), fill_value=7.0)                      # 用同一个值填满
randn = torch.randn(3, 3)                                      # 标准正态 N(0,1)，神经网络权重初始化最常用
rand = torch.rand(2, 5)                                        # [0,1) 均匀分布，常用于随机 mask、dropout
arange = torch.arange(0, 10, 2)                                # 等差数列（不含终点），类似 range
linspace = torch.linspace(0.0, 1.0, 5)                         # 等间隔取 5 个点（含终点），画图坐标轴常用
eye = torch.eye(4)                                             # 单位矩阵，线性代数里构造恒等变换
ones_like = torch.ones_like(randn)                             # 形状/dtype 与 randn 一致的全 1 张量

print(f"ones(3, 4)          → shape={tuple(ones.shape)}, dtype={ones.dtype}（全 1，默认 float32）")
print(f"zeros(2, 3)         → shape={tuple(zeros.shape)}")
print(f"full((2,2), 7.0)    → shape={tuple(full.shape)}, 值={full.tolist()}")
print(f"randn(3, 3)         → shape={tuple(randn.shape)}，均值≈{randn.mean().item():+.4f}，标准差≈{randn.std().item():.4f}（应接近 0 和 1）")
print(f"rand(2, 5)          → shape={tuple(rand.shape)}，最小值={rand.min().item():.4f}，最大值={rand.max().item():.4f}（范围 [0,1)）")
print(f"arange(0, 10, 2)    → shape={tuple(arange.shape)}, 值={arange.tolist()}, dtype={arange.dtype}")
print(f"linspace(0, 1, 5)   → shape={tuple(linspace.shape)}, 值={[round(v, 3) for v in linspace.tolist()]}")
print(f"eye(4)              → shape={tuple(eye.shape)}，对角线之和（迹）={eye.trace().item():.1f}")
print(f"ones_like(randn)    → shape={tuple(ones_like.shape)}, dtype={ones_like.dtype}（形状和 dtype 都跟着 randn）")

# 想生成整数张量，必须显式给 dtype，否则 randint 之外的都是浮点
randint = torch.randint(low=0, high=10, size=(3, 4))
print(f"randint(0, 10, (3,4)) → dtype={randint.dtype}, 值=\n{randint}")
print()


# ===========================================================================
# 小节 4：元素级运算 vs 矩阵乘法（最容易混淆的两个概念）
# ===========================================================================
print("-" * 78)
print("【4】元素级运算（+ / * / **）与矩阵乘法（@）的区别")
print("-" * 78)

a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

# 元素级（element-wise）：两个张量形状相同，对应位置一一运算，形状不变。
add = a + b
mul = a * b          # 这是哈达玛积（Hadamard product），不是矩阵乘法！
pow_ = a ** 2
div = b / a

# 矩阵乘法：遵守线性代数规则 (m, n) @ (n, p) = (m, p)
# 要求：左操作数的最后一维 必须等于 右操作数的倒数第二维
matmul = a @ b

print("a =", a.tolist())
print("b =", b.tolist())
print(f"元素级相加 a + b       → shape={tuple(add.shape)}，值={add.tolist()}")
print(f"元素级相乘 a * b       → shape={tuple(mul.shape)}，值={mul.tolist()}   ← 不是矩阵乘法！")
print(f"元素级平方 a ** 2      → shape={tuple(pow_.shape)}，值={pow_.tolist()}")
print(f"矩阵乘法   a @ b       → shape={tuple(matmul.shape)}，值={matmul.tolist()}   ← 行×列再求和")

# 用形状不同的例子强调矩阵乘法的形状规则
m1 = torch.randn(4, 3)
m2 = torch.randn(3, 5)
print(f"\n矩阵乘法形状规则：(4, 3) @ (3, 5) → {tuple((m1 @ m2).shape)}，即 (m, n) @ (n, p) = (m, p)")

# 形状不匹配时 PyTorch 会报错。这里用 try/except 捕获，只打印中文说明，绝不吐出异常栈。
try:
    _ = torch.randn(4, 3) @ torch.randn(4, 5)     # 内维 3 != 4，非法
    print("(4,3)@(4,5) 竟然成功了？不应该发生")
except RuntimeError:
    print("尝试 (4, 3) @ (4, 5)：内维 3 != 4，PyTorch 会报错。")
    print("  → 已用 try/except 捕获，这类「形状不匹配」是深度学习最常见的错误之一。")
print()


# ===========================================================================
# 小节 5：广播机制（Broadcasting）——最省内存的「隐式扩展」
# ===========================================================================
print("-" * 78)
print("【5】广播机制：从右往左对齐，维度为 1 或缺失就扩展")
print("-" * 78)

# 广播的三条规则（一定要背下来）：
#   规则 1：从最右边（最后一维）开始，一维一维往左对齐比较。
#   规则 2：如果某一维两边相等 → 没问题，直接逐元素计算。
#           如果某一维有一边是 1 → 把长度为 1 的那一边「复制」扩展成另一边。
#           如果某一维有一边「缺失」（维度数少）→ 视作在前面补 1，再按上一条扩展。
#   规则 3：两边都不等且都不为 1 → 报错。
# 关键点：广播不真的复制数据，只是逻辑上让计算「以为」形状一样，所以非常省内存。
col = torch.tensor([[1.0], [2.0], [3.0]])       # shape (3, 1)
row = torch.tensor([[10.0, 20.0, 30.0, 40.0]])  # shape (1, 4)

print(f"col shape = {tuple(col.shape)}  （3 行 1 列）")
print(f"row shape = {tuple(row.shape)}  （1 行 4 列）")

broadcast_sum = col + row                       # (3,1) + (1,4) → (3,4)
broadcast_mul = col * row
print(f"(3,1) + (1,4) → shape={tuple(broadcast_sum.shape)}   ← 广播的经典例子")
print("广播后的结果矩阵：")
print(broadcast_sum)

# 再举一个「缺失维度」的例子：2 维 + 1 维，右边 1 维被对齐到最后一维
mat = torch.ones(2, 3)                          # (2, 3)
vec = torch.tensor([1.0, 2.0, 3.0])             # (3,)  → 视作 (1, 3) 扩展成 (2, 3)
print(f"\n(2,3) + (3,) → shape={tuple((mat + vec).shape)}   ← (3,) 视作 (1,3) 后扩展")
print(f"结果：\n{mat + vec}")

# 深度学习中广播的真实用途：标准化（每个特征减自己的均值、除自己的标准差）
# X 是 (batch, feature)，mean/std 是 (feature,)，广播后每个样本的每个特征都被对应处理。
X = torch.randn(5, 4)
mu = X.mean(dim=0)                              # 沿 batch 维求均值 → shape (4,)
sigma = X.std(dim=0)                            # 沿 batch 维求标准差 → shape (4,)
X_norm = (X - mu) / (sigma + 1e-8)              # (5,4) 与 (4,) 广播
print(f"\n标准化示例：X shape={tuple(X.shape)}, mu shape={tuple(mu.shape)}, sigma shape={tuple(sigma.shape)}")
print(f"X_norm = (X - mu) / (sigma + 1e-8) → shape={tuple(X_norm.shape)}")
print(f"标准化后每个特征的均值 ≈ {[round(v, 6) for v in X_norm.mean(dim=0).tolist()]}（应接近 0）")
print(f"标准化后每个特征的标准差 ≈ {[round(v, 4) for v in X_norm.std(dim=0).tolist()]}（应接近 1）")

# 演示广播失败的情形（同样只打印中文说明）
try:
    _ = torch.randn(3, 4) + torch.randn(3, 5)   # 最后一维 4 vs 5，都不为 1 → 失败
    print("形状 (3,4)+(3,5) 竟然成功了？不应该发生")
except RuntimeError:
    print("\n尝试 (3, 4) + (3, 5)：最后一维 4 != 5 且都不为 1，广播失败。")
    print("  → 已捕获。记住：广播只在「相等」或「有一边是 1」时才成立。")
print()


# ===========================================================================
# 小节 6：形状变换 reshape / view / flatten
# ===========================================================================
print("-" * 78)
print("【6】形状变换：reshape / view / flatten，以及 -1 的用法")
print("-" * 78)

# reshape 和 view 的**共同点**：都要求「元素个数不变」，即变换前后 numel() 相同。
# reshape 和 view 的**区别**：
#   view  —— 只能作用在「内存连续（contiguous）」的张量上，它不拷贝数据，只是重新解释内存布局，
#            速度最快；如果张量是从大张量切片/转置得到的（不连续），view 会直接报错。
#   reshape—— 更宽容：能 view 就 view（不拷贝），不能 view 就自动拷贝一份新的再变形。
# 结论：日常写代码用 reshape 更安全；追求极致性能、且明确知道内存连续时用 view。
z = torch.arange(12)                    # 0..11，shape (12,)
print(f"原始 z         → shape={tuple(z.shape)}, numel={z.numel()}, 内存连续={z.is_contiguous()}")

z_reshaped = z.reshape(3, 4)
z_viewed = z.view(3, 4)
print(f"z.reshape(3,4) → shape={tuple(z_reshaped.shape)}")
print(f"z.view(3,4)    → shape={tuple(z_viewed.shape)}")
print(f"reshape 结果与 view 结果是否共享内存：{z_reshaped.data_ptr() == z_viewed.data_ptr()}（都是同一块 arange 内存）")
print("reshape 后的内容（行优先填充，即最后一维变化最快）：")
print(z_reshaped)

# 用 -1 让 PyTorch 自己推算这一维：12 个元素、指定第一维 3 → 第二维必然是 4
z_auto = z.reshape(3, -1)
print(f"\nz.reshape(3, -1) → shape={tuple(z_auto.shape)}（-1 表示「这一维你自己算」）")

# 展平：把 (2,3,4) 压成 (2,12)，常用于卷积特征图送入全连接层之前
feat_map = torch.randn(2, 3, 4)
flat = feat_map.reshape(2, -1)
print(f"卷积特征图 {tuple(feat_map.shape)} → reshape(2, -1) → {tuple(flat.shape)}（保留 batch 维，其余展平）")
print(f"完全展平 feat_map.flatten() → shape={tuple(feat_map.flatten().shape)}（所有维压成 1 维）")

# 演示 view 在非连续张量上会失败——这正是要记住的坑
non_contig = torch.arange(12).reshape(3, 4).t()      # 转置后内存不再连续
print(f"\n转置得到 non_contig → shape={tuple(non_contig.shape)}, 内存连续={non_contig.is_contiguous()}")
try:
    _ = non_contig.view(-1)                          # 不连续 → view 失败
    print("non_contig.view(-1) 竟然成功了？不应该发生")
except RuntimeError:
    print("对非连续张量调用 .view(-1)：PyTorch 会报错。")
    print("  → 已捕获。两种正确写法：① .reshape(-1)（自动拷贝）；② .contiguous().view(-1)（显式拷贝成连续再 view）。")

fixed_by_contiguous = non_contig.contiguous().view(-1)
print(f".contiguous().view(-1) → shape={tuple(fixed_by_contiguous.shape)}，值={fixed_by_contiguous.tolist()}")
print(f"（转置过的值按行优先重新排布，所以顺序是 0,4,8,1,5,9,2,6,10,3,7,11）")
print()


# ===========================================================================
# 小节 7：增加 / 删除维度 unsqueeze / squeeze
# ===========================================================================
print("-" * 78)
print("【7】unsqueeze / squeeze：增删长度为 1 的维度")
print("-" * 78)

# 为什么需要这两个操作？
#   unsqueeze：很多算子要求输入有 batch 维或通道维。单张图片 (28,28) 要变成 (1,1,28,28)
#              才能送进只接受 4 维输入的卷积层；一条样本 (10,) 要变成 (1,10) 才能算 batch 统计量。
#   squeeze  ：反向操作，把无意义的长度为 1 的维度去掉，避免出现 (1,1,1,10) 这种脏形状。
img = torch.randn(28, 28)
print(f"单张灰度图          → shape={tuple(img.shape)}, dim()={img.dim()}")

img_4d = img.unsqueeze(0).unsqueeze(0)          # 在前面依次补 batch 维和通道维
print(f"unsqueeze(0).unsqueeze(0) → shape={tuple(img_4d.shape)}  ← (1, 1, H, W)，可以送进卷积层了")

# unsqueeze 的位置决定新维插在哪里；负数索引表示从右边数
t = torch.randn(2, 3)
print(f"\n原始 t                    → shape={tuple(t.shape)}")
print(f"t.unsqueeze(0)            → shape={tuple(t.unsqueeze(0).shape)}   ← 插在最前面，变成「1 个 batch」")
print(f"t.unsqueeze(1)            → shape={tuple(t.unsqueeze(1).shape)}   ← 插在中间")
print(f"t.unsqueeze(-1)           → shape={tuple(t.unsqueeze(-1).shape)}  ← 插在最后，变成「每个元素一个通道」")

# squeeze 只删长度为 1 的维度；指定 dim 时如果那一维不是 1，则原样返回（不报错）
dirty = torch.randn(1, 3, 1, 5)
print(f"\n脏形状 {tuple(dirty.shape)}：")
print(f"  .squeeze()        → shape={tuple(dirty.squeeze().shape)}   ← 删掉所有长度为 1 的维度")
print(f"  .squeeze(0)       → shape={tuple(dirty.squeeze(0).shape)}   ← 只删第 0 维")
print(f"  .squeeze(1)       → shape={tuple(dirty.squeeze(1).shape)}   ← 第 1 维长度是 3，不是 1，所以不变")
print()


# ===========================================================================
# 小节 8：轴变换 transpose / permute
# ===========================================================================
print("-" * 78)
print("【8】transpose 与 permute：交换轴的顺序")
print("-" * 78)

# 为什么需要换轴？
#   PyTorch 卷积默认 NCHW（batch, channel, H, W），
#   而 matplotlib.imshow 和很多第三方库要 NHWC（batch, H, W, channel）。
#   这就是 permute 最经典的用途。
# transpose(dim0, dim1)：一次只交换两个轴，且结果**不连续**（内存布局变了）。
# permute(*dims)       ：一次性按给定顺序重排所有轴，等价于多次 transpose。

mat = torch.arange(6).reshape(2, 3)
print(f"原始 mat          → shape={tuple(mat.shape)}")
print(mat)
print(f"mat.transpose(0,1) → shape={tuple(mat.transpose(0, 1).shape)}（2x3 变 3x2，行变列）")
print(mat.transpose(0, 1))
print(f"转置后内存连续={mat.transpose(0, 1).is_contiguous()}  ← 所以转置后想 view 必须先 contiguous()")

nchw = torch.randn(2, 3, 28, 28)                # 一批 2 张 3 通道 28x28 彩色图
nhwc = nchw.permute(0, 2, 3, 1)                 # 把通道维换到最后
print(f"\nNCHW {tuple(nchw.shape)} --permute(0,2,3,1)--> NHWC {tuple(nhwc.shape)}")
print(f"permute 只改「怎么看」，不改数据本身：元素个数不变 {nchw.numel()} == {nhwc.numel()}")

# 用 .contiguous() 让 permute 后的张量变成内存连续，这样就能安全地 view / 送进某些算子
nhwc_c = nhwc.contiguous()
print(f"permute 后 .contiguous()：内存连续={nhwc_c.is_contiguous()}，可用 .view 展平 → {tuple(nhwc_c.view(2, -1).shape)}")

# 3 维张量的 permute：batch_first=True 的常见需求
seq = torch.randn(10, 32, 64)                   # (seq_len, batch, hidden) 老式 RNN 输入布局
seq_bf = seq.permute(1, 0, 2)                   # → (batch, seq_len, hidden) batch_first 布局
print(f"\n(seq_len=10, batch=32, hidden=64) --permute(1,0,2)--> {tuple(seq_bf.shape)}")
print()


# ===========================================================================
# 小节 9：索引与切片（基础切片 / 步长 / 布尔掩码 / index_select / gather / where）
# ===========================================================================
print("-" * 78)
print("【9】索引：切片 / 步长 / 布尔掩码 / index_select / gather / where")
print("-" * 78)

t = torch.arange(24).reshape(4, 6)              # 4 行 6 列
print(f"t shape={tuple(t.shape)}：")
print(t)

# 9.1 基础切片：t[行范围, 列范围]，和 numpy 完全一致
print(f"\nt[0]        → shape={tuple(t[0].shape)}，值={t[0].tolist()}（取第 0 行，维度降了 1）")
print(f"t[0:2]      → shape={tuple(t[0:2].shape)}（取前 2 行，切片保留维度）")
print(f"t[:, 0]     → shape={tuple(t[:, 0].shape)}，值={t[:, 0].tolist()}（取第 0 列）")
print(f"t[1:3, 2:5] → shape={tuple(t[1:3, 2:5].shape)}（行列同时切）")
print(f"t[::2, ::3] → shape={tuple(t[::2, ::3].shape)}（步长切片：行每 2 个取 1，列每 3 个取 1）")
print(f"t[-1]       → 值={t[-1].tolist()}（负索引从后往前，-1 是最后一行）")

# 9.2 整数数组索引（fancy indexing）
rows = torch.tensor([0, 2, 3])
picked = t[rows]
print(f"\n用整数张量选行：t[torch.tensor([0,2,3])] → shape={tuple(picked.shape)}（选了 3 行）")

# 9.3 布尔掩码：最常用于「按条件筛样本」或「只保留正类」
mask = t % 3 == 0                               # 逐元素比较，得到同形状的 bool 张量
print(f"\n布尔掩码 (t % 3 == 0) → shape={tuple(mask.shape)}, dtype={mask.dtype}")
print(f"掩码为 True 的个数 = {int(mask.sum())}")
selected = t[mask]                              # 注意：布尔索引会把结果压成 1 维！
print(f"t[mask] → shape={tuple(selected.shape)}，值={selected.tolist()}（布尔索引会压平成 1 维）")

# 用掩码做「二分类标签」的经典写法
logits = torch.randn(6)
labels = (logits > 0).long()                    # bool → int64，作为分类标签
print(f"\nlogits  = {[round(v, 3) for v in logits.tolist()]}")
print(f"labels  = {labels.tolist()}（logits > 0 得到 bool，再 .long() 变成 0/1 标签）")

# 9.4 torch.where：按条件从两个张量里挑元素，等价于三目运算符
a_w = torch.tensor([1.0, -2.0, 3.0, -4.0])
clamped = torch.where(a_w > 0, a_w, torch.zeros_like(a_w))    # 负数变 0，这就是 ReLU 的朴素实现
print(f"\ntorch.where(a>0, a, 0) = {clamped.tolist()}  ← 手写 ReLU，和 F.relu 结果一致")

# 9.5 index_select：沿指定维度按索引取（比 fancy indexing 更明确、可读性更强）
src = torch.arange(12).reshape(3, 4)
idx = torch.tensor([2, 0])
print(f"\nindex_select 源 src shape={tuple(src.shape)}")
print(f"src.index_select(dim=0, index=[2,0]) → shape={tuple(src.index_select(0, idx).shape)}，值=\n{src.index_select(0, idx)}")
print(f"src.index_select(dim=1, index=[3,1]) → shape={tuple(src.index_select(1, torch.tensor([3, 1])).shape)}")

# 9.6 gather：每个位置按自己的索引取值，形状由 index 决定。
# 最实用的场景：从 logits 里取出「真实类别的预测分数」（即交叉熵的 pick 步骤）。
logits_3 = torch.tensor([[0.2, 0.9, 0.4],
                         [0.7, 0.1, 0.6]])      # (batch=2, num_classes=3)
target = torch.tensor([[1], [0]])               # (batch=2, 1) 真实类别索引
picked_score = logits_3.gather(dim=1, index=target)
print(f"\ngather 示例：logits shape={tuple(logits_3.shape)}, target={target.tolist()}")
print(f"gather(dim=1, index=target) → shape={tuple(picked_score.shape)}，值={picked_score.tolist()}")
print("  → 第 0 个样本取出类别 1 的分数 0.9，第 1 个样本取出类别 0 的分数 0.7，这就是交叉熵里的「挑出正确类」步骤。")
print()


# ===========================================================================
# 小节 10：拼接与切分 cat / stack / split / chunk
# ===========================================================================
print("-" * 78)
print("【10】torch.cat（沿已有维度拼）与 torch.stack（新建维度）")
print("-" * 78)

p1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])     # (2, 2)
p2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])     # (2, 2)

# cat：沿「已经存在」的维度拼接，维度数不变，被拼的那一维长度相加。
cat_dim0 = torch.cat([p1, p2], dim=0)           # (2,2)+(2,2) → (4,2)
cat_dim1 = torch.cat([p1, p2], dim=1)           # → (2,4)

# stack：先「新建」一个维度，再把张量摞上去，维度数 +1。
stack_dim0 = torch.stack([p1, p2], dim=0)       # 2 个 (2,2) → (2,2,2)
stack_dim1 = torch.stack([p1, p2], dim=1)       # → (2,2,2)，但摆放顺序不同

print(f"p1 shape={tuple(p1.shape)}, p2 shape={tuple(p2.shape)}")
print(f"torch.cat([p1,p2], dim=0)   → shape={tuple(cat_dim0.shape)}   ← 维度数不变，第 0 维相加 2+2=4")
print(f"torch.cat([p1,p2], dim=1)   → shape={tuple(cat_dim1.shape)}   ← 第 1 维相加 2+2=4")
print(f"torch.stack([p1,p2], dim=0) → shape={tuple(stack_dim0.shape)} ← 维度数 +1，新建了一维")
print(f"torch.stack([p1,p2], dim=1) → shape={tuple(stack_dim1.shape)} ← 同样是 +1，只是新维插在中间")
print(f"\nstack(dim=0) 的内容（两个矩阵摞成一个「批次」）：\n{stack_dim0}")

# 一句话记忆：形状相同、想变成「一批」→ stack；形状在某一维不同、想接起来 → cat。
# 实战对照：
#   - 把 3 个 batch 的中间激活缓存起来最后一起算 → torch.cat(activations, dim=0)
#   - 把单样本 (10,) 组成一批 → torch.stack(list_of_1d, dim=0) 得到 (batch, 10)

# 切分：split 按「每块多大」切，chunk 按「切成几块」切
big = torch.arange(20).reshape(5, 4)
parts = torch.split(big, split_size_or_sections=2, dim=0)         # 每块 2 行 → 2,2,1
chunks = torch.chunk(big, chunks=3, dim=0)                        # 切成 3 块 → 2,2,1
print(f"\nbig shape={tuple(big.shape)}")
print(f"torch.split(big, 2, dim=0) → {len(parts)} 块，形状={[tuple(p.shape) for p in parts]}（最后一块可能不满）")
print(f"torch.chunk(big, 3, dim=0) → {len(chunks)} 块，形状={[tuple(c.shape) for c in chunks]}（均分，除不尽时前面的块更大）")

# 用 stack 把「一批一维样本」组成矩阵，这是 Dataset 返回单样本、collate 组装 batch 的本质
samples = [torch.randn(5) for _ in range(4)]     # 4 条样本，每条 5 个特征
batch_t = torch.stack(samples, dim=0)
print(f"\n4 条 (5,) 样本 --stack(dim=0)--> {tuple(batch_t.shape)}  ← 这正是 DataLoader 组装 batch 的核心动作")
print()


# ===========================================================================
# 小节 11：设备与 CPU/GPU（本机无 CUDA，重点讲清 CPU 版行为）
# ===========================================================================
print("-" * 78)
print("【11】设备管理：CPU / CUDA，以及 .to(device) 的用法")
print("-" * 78)

# 张量的 .device 属性记录它在哪块硬件上。CPU 张量打印 "cpu"，
# GPU 张量打印 "cuda:0"（第 0 号 GPU）。
t_cpu = torch.randn(3, 3)
print(f"t_cpu.device = {t_cpu.device}")
print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
print(f"torch.cuda.device_count() = {torch.cuda.device_count()}")

if torch.cuda.is_available():
    # 真实 GPU 机器上会走这一支：把张量和模型都搬到同一块卡上
    t_gpu = t_cpu.to("cuda")
    print(f"已移动到 GPU：{t_gpu.device}")
else:
    # 本机是 torch 2.14.0+cpu（CPU 专用构建），这一支一定会执行。
    # 说明：CPU 版 PyTorch 里没有 CUDA 相关算子，torch.cuda.is_available() 恒为 False，
    #       .to("cuda") 会直接抛 RuntimeError。所以这里用 .to("cpu") 演示等价代码路径。
    print("\n本机情况说明：")
    print("  · 安装的是 CPU 版 torch（版本号带 +cpu），编译时就没链接 CUDA 运行时，")
    print("    因此 torch.cuda.is_available() 恒为 False，torch.cuda.device_count() 恒为 0。")
    print("  · 此时若写 .to('cuda') 会直接抛异常（实测：Torch not compiled with CUDA enabled）。")
    print("  · 但写法上应当保持「设备无关」：先算 device，再 .to(device)，这样同一份代码在")
    print("    GPU 机器上无需改动就能自动加速——这就是下面演示的写法。")

    t_dev = t_cpu.to(device)                    # device 已在上方算好，本机就是 "cpu"
    print(f"\n  设备无关写法：t_cpu.to(device) → device={t_dev.device}，形状不变 {tuple(t_dev.shape)}")
    print(f"  .to('cpu') 是幂等的：对 CPU 张量再调一次也不会拷贝报错，device 仍为 {t_cpu.to('cpu').device}")

# 用 try/except 演示 .to("cuda") 在 CPU 版上的行为（只打印中文说明，不吐异常栈）
# 注意：CPU 版 torch 抛的是 AssertionError 而不是 RuntimeError，
#       所以这里捕获宽泛的 Exception，保证任何版本下都不会把异常栈打到屏幕上。
try:
    _ = t_cpu.to("cuda")
    print("\n（意外）本机竟然能移动到 CUDA。")
except Exception as exc:
    # 只打印中文说明：不打印异常类型名，也不打印异常消息，保证输出里没有任何异常痕迹
    print("\n补充演示：在 CPU 版上执行 .to('cuda') 会被拒绝（本机 torch 未编译 CUDA 支持）。")
    print("  → 已用 try/except 捕获。正确做法永远是先判断 torch.cuda.is_available() 再决定 device。")

# dtype 和 device 可以一起搬：.to(device=..., dtype=...)
t_moved = t_cpu.to(device=device, dtype=torch.float64)
print(f"\n同时指定设备和类型：.to(device='{device}', dtype=torch.float64) → device={t_moved.device}, dtype={t_moved.dtype}")
print()


# ===========================================================================
# 小节 12：绘图——形状变换与广播示意图
# ===========================================================================
print("-" * 78)
print("【12】绘图：形状变换与广播示意图")
print("-" * 78)

fig = plt.figure(figsize=(15, 9))
fig.suptitle("张量：形状变换与广播机制示意图", fontsize=16, fontweight="bold")

# ---- 子图 1：reshape 变换链 (12,) → (3,4) → (3,2,2) ----
ax1 = fig.add_subplot(2, 3, 1)
ax1.set_title("① reshape 变换链（元素总数守恒 = 12）", fontsize=11)
ax1.axis("off")
chain = [
    ("原始 (12,)", torch.arange(12)),
    ("reshape(3, 4)", torch.arange(12).reshape(3, 4)),
    ("reshape(3, 2, 2)", torch.arange(12).reshape(3, 2, 2)),
]
y_pos = 0.92
for label, tensor in chain:
    ax1.text(0.02, y_pos, f"{label}   shape={tuple(tensor.shape)}", fontsize=9, transform=ax1.transAxes)
    y_pos -= 0.09
    ax1.text(0.06, y_pos, str(tensor.tolist())[:64], fontsize=7,
             transform=ax1.transAxes, color="#1f4e79")
    y_pos -= 0.16
ax1.text(0.02, y_pos - 0.02,
         "要点：变换前后 numel() 必须相等；\nreshape 允许拷贝，view 要求内存连续。",
         fontsize=9, transform=ax1.transAxes, color="#c00000")

# ---- 子图 2：unsqueeze / squeeze 增删维度 ----
ax2 = fig.add_subplot(2, 3, 2)
ax2.set_title("② unsqueeze / squeeze", fontsize=11)
ax2.axis("off")
steps = [
    ("(28, 28)", "单张灰度图"),
    ("unsqueeze(0) → (1, 28, 28)", "补通道维"),
    ("unsqueeze(0) → (1, 1, 28, 28)", "补 batch 维，可进卷积层"),
    ("squeeze() → (28, 28)", "去掉所有长度为 1 的维"),
]
y_pos = 0.88
for shape_text, note in steps:
    ax2.text(0.03, y_pos, shape_text, fontsize=10, transform=ax2.transAxes,
             color="#1f4e79", fontweight="bold")
    ax2.text(0.09, y_pos - 0.07, f"← {note}", fontsize=8, transform=ax2.transAxes, color="#404040")
    y_pos -= 0.20
ax2.text(0.03, y_pos + 0.03, "要点：squeeze 只删长度为 1 的维，\n指定 dim 时该维不是 1 就原样返回。",
         fontsize=9, transform=ax2.transAxes, color="#c00000")

# ---- 子图 3：permute NCHW → NHWC ----
ax3 = fig.add_subplot(2, 3, 3)
ax3.set_title("③ permute：NCHW → NHWC", fontsize=11)
ax3.axis("off")
ax3.text(0.05, 0.85, "原始 (N, C, H, W) = (2, 3, 28, 28)", fontsize=10, transform=ax3.transAxes)
ax3.annotate("", xy=(0.50, 0.62), xytext=(0.20, 0.72),
             xycoords="axes fraction", textcoords="axes fraction",
             arrowprops=dict(arrowstyle="->", color="#c00000", lw=2))
ax3.text(0.30, 0.66, "permute(0, 2, 3, 1)", fontsize=9, color="#c00000", transform=ax3.transAxes)
ax3.text(0.05, 0.48, "结果 (N, H, W, C) = (2, 28, 28, 3)", fontsize=10, transform=ax3.transAxes,
         color="#1f4e79", fontweight="bold")
ax3.text(0.05, 0.30, "轴顺序：[0]批 [1]通道 [2]高 [3]宽\n→ 换成 [0]批 [2]高 [3]宽 [1]通道",
         fontsize=8.5, transform=ax3.transAxes, color="#404040")
ax3.text(0.05, 0.06, "要点：permute 后内存不连续，\n想 view 需先 .contiguous()。",
         fontsize=9, transform=ax3.transAxes, color="#c00000")

# ---- 子图 4：(3,1) + (1,4) 广播结果热力图 ----
ax4 = fig.add_subplot(2, 3, 4)
col_demo = torch.tensor([[1.0], [2.0], [3.0]])
row_demo = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
bc = col_demo + row_demo
im4 = ax4.imshow(bc.numpy(), cmap="YlGnBu", aspect="auto")
ax4.set_title("④ 广播：(3,1) + (1,4) → (3,4)", fontsize=11)
ax4.set_xticks(range(4), [f"列{j}" for j in range(4)], fontsize=8)
ax4.set_yticks(range(3), [f"行{i}" for i in range(3)], fontsize=8)
for i in range(3):
    for j in range(4):
        ax4.text(j, i, f"{bc[i, j].item():.0f}", ha="center", va="center", fontsize=10, color="#08306b")
fig.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04).set_label("广播后数值", fontsize=8)
ax4.set_xlabel("(1,4) 沿行方向复制 3 次", fontsize=9)
ax4.set_ylabel("(3,1) 沿列方向复制 4 次", fontsize=9)

# ---- 子图 5：广播规则「从右往左对齐」示意 ----
ax5 = fig.add_subplot(2, 3, 5)
ax5.set_title("⑤ 广播规则：从右往左对齐", fontsize=11)
ax5.axis("off")
ax5.text(0.03, 0.90, "规则：从最后一维开始向左对齐比较", fontsize=9.5, transform=ax5.transAxes,
         fontweight="bold", color="#1f4e79")
rows_txt = [
    ("A  :", "[  3,   1  ]", ""),
    ("B  :", "[  1,   4  ]", ""),
    ("对齐：", "  3 vs 1 → 扩成 3        1 vs 4 → 扩成 4", ""),
    ("结果：", "[  3,   4  ]", ""),
]
y_pos = 0.88
for label, value, _ in rows_txt:
    ax5.text(0.03, y_pos, label, fontsize=10, transform=ax5.transAxes)
    ax5.text(0.22, y_pos, value, fontsize=10, transform=ax5.transAxes, color="#1f4e79")
    y_pos -= 0.13
ax5.text(0.03, y_pos - 0.05,
         "三条充要条件：\n  ① 该维两边相等 → 直接算\n  ② 有一边是 1 → 复制扩展成另一边\n  ③ 维度数少的一方 → 左边补 1 再比\n否则报错（4 vs 5 这种）。",
         fontsize=8.5, transform=ax5.transAxes, color="#404040")
ax5.text(0.03, 0.05, "广播不真正复制数据，只改「读数的步长」，所以省内存。",
         fontsize=8.5, transform=ax5.transAxes, color="#c00000")

# ---- 子图 6：cat vs stack 形状对比 ----
ax6 = fig.add_subplot(2, 3, 6)
ax6.set_title("⑥ cat 沿已有维拼接 vs stack 新建维", fontsize=11)
ax6.axis("off")
cmp_lines = [
    ("输入：p1 (2,2) , p2 (2,2)", "两个同形状矩阵", "#000000"),
    ("torch.cat(dim=0)", f"→ {tuple(torch.cat([p1, p2], 0).shape)}   维度数不变（2+2 相加）", "#1f4e79"),
    ("torch.cat(dim=1)", f"→ {tuple(torch.cat([p1, p2], 1).shape)}", "#1f4e79"),
    ("torch.stack(dim=0)", f"→ {tuple(torch.stack([p1, p2], 0).shape)}  维度数 +1（新建一维）", "#c00000"),
    ("torch.stack(dim=1)", f"→ {tuple(torch.stack([p1, p2], 1).shape)}", "#c00000"),
]
y_pos = 0.86
for text, note, color in cmp_lines:
    ax6.text(0.02, y_pos, text, fontsize=9.5, transform=ax6.transAxes, color=color, fontweight="bold")
    ax6.text(0.40, y_pos, note, fontsize=8.5, transform=ax6.transAxes, color=color)
    y_pos -= 0.15
ax6.text(0.02, y_pos - 0.04,
         "记忆法：\n  · 形状相同、要变成「一批」→ stack\n  · 某一维长度不同、要接起来 → cat",
         fontsize=9, transform=ax6.transAxes, color="#404040")

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_path = OUTPUT_DIR / "01_张量_形状变换与广播.png"
plt.savefig(save_path, dpi=110, bbox_inches="tight")
plt.close(fig)                                  # 关闭图形释放内存；本脚本绝不调用 plt.show()

print(f"示意图已保存：{save_path}")
print(f"文件存在：{save_path.exists()}，大小：{save_path.stat().st_size / 1024:.1f} KB")
print()

# ===========================================================================
# 小结
# ===========================================================================
print("=" * 78)
print("小结")
print("=" * 78)
print("· 创建：torch.tensor(列表) / torch.from_numpy(共享内存) / ones·zeros·randn·arange·linspace·eye")
print("· 默认 dtype：整数→int64，浮点→float32（由 torch.get_default_dtype() 决定）")
print("· 运算：元素级用 + * - / **（形状不变），矩阵乘法用 @（(m,n)@(n,p)→(m,p)）")
print("· 广播：从右往左对齐，维度相等或其中之一为 1 才能广播；(3,1)+(1,4)→(3,4)")
print("· 变形：reshape 宽容（必要时拷贝），view 要求内存连续，否则先 .contiguous()")
print("· 维度：unsqueeze 补 1 维、squeeze 删长度 1 的维、transpose 换两轴、permute 换多轴")
print("· 索引：切片保留维度，整数/布尔索引会降维或压平；index_select、gather 按索引取值")
print("· 拼接：cat 沿已有维（维度数不变），stack 新建维（维度数 +1）")
print(f"· 设备：torch.cuda.is_available()={torch.cuda.is_available()}（CPU 版），写 .to(device) 保持设备无关")
print("=" * 78)
print("脚本执行完毕，退出码 0。")

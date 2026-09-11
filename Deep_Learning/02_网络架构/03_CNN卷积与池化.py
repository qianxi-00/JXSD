"""
对应课案章节：网络架构 → CNN（卷积 / 池化）

本节知识点：
    1. CNN 的核心三点：局部连接（每个输出只看一小块区域）、权重共享（同一个卷积核在整张图上滑动）、
       空间特征提取（保留 H×W 的空间结构，而不是像 DNN 那样先展平）。
    2. 二维卷积公式：(Y)_{i,j} = Σ_m Σ_n (K)_{m,n} · X_{i+m, j+n} + b，
       其中 kh、kw 是卷积核高宽，(K)_{m,n} 是核在 (m,n) 处的权重，b 是偏置。
       —— 用双重循环手写滑动窗口实现，并和 nn.Conv2d 对比验证。
    3. 输出尺寸公式：H_out = (H + 2×padding - kh) / stride + 1
       （向下取整，要求能整除或按 PyTorch 的向下取整规则）。必须支持 stride 与 padding。
    4. 多通道卷积：输入 (batch, C_in, H, W)，卷积核 (C_out, C_in, kh, kw)，
       每个输出通道 = 所有输入通道卷积结果之和。逐层打印形状。
    5. 池化：最大池化 max、平均池化 mean，公式里 p_h, p_w 是池化窗口高宽，s 是步长。
       手写 unfold 版最大池化（不双重循环）并与 nn.MaxPool2d 对比（误差应为 0）。
    6. nn.AdaptiveAvgPool2d：指定"输出尺寸"而不是"窗口大小"，
       把任意输入尺寸压到固定尺寸，分类头常用它把 (N, C, H, W) 压成 (N, C, 1, 1)，
       这样网络就能接受任意分辨率的输入。
    7. 课案 CNN 结构：Conv(1,32,3,p=1)+ReLU+MaxPool(2) → Conv(32,64,3,p=1)+ReLU+MaxPool(2)
       → Flatten → Linear(64*7*7,128) → ReLU → Linear(128,10)；
       输入 (4,1,28,28) 输出 (4,10)，并逐层推导 28→28→14→14→7 的形状变化。
    8. 参数量计算：手算每层参数量并与 sum(p.numel()) 对比，
       说明权重共享让卷积参数量远小于等价的全连接层。
    9. 感受野：堆叠 3×3 卷积能逐步扩大感受野，且参数比单个大核更少。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Deep_Learning\\02_网络架构\\03_CNN卷积与池化.py'
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
from matplotlib.patches import Rectangle

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

torch.manual_seed(42)   # 统一随机种子
np.random.seed(42)

_T0 = time.perf_counter()   # 计时：交付要求单脚本 < 40 秒（优先 < 20 秒）


def section(title: str) -> None:
    """统一的章节打印，让控制台输出层次分明。"""
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


# ===========================================================================
# 一、手写二维卷积（双重循环滑动窗口）
# ===========================================================================
section("一、手写二维卷积：双重循环滑动窗口")

# 卷积公式（课案）：
#     (Y)_{i,j} = Σ_{m=0}^{kh-1} Σ_{n=0}^{kw-1} (K)_{m,n} · X_{i+m, j+n} + b
# 直觉：把卷积核当成一个"小窗口"，扣在输入特征图上，逐位置做"对应元素相乘再求和"，
#       然后按 stride 滑动到下一个位置。
# 三个关键参数：
#   kernel_size (kh, kw)：窗口大小，决定每个输出"看多大范围"
#   stride      ：滑动步长，步长越大输出的 H/W 越小
#   padding     ：在输入四周补 0 的圈数，用来控制输出尺寸（padding=kh//2 时尺寸不变）
# 输出尺寸公式（务必记住）：
#   H_out = (H + 2*padding - kh) / stride + 1
#   W_out = (W + 2*padding - kw) / stride + 1
#   例：28 + 2*1 - 3 = 27，27/1 + 1 = 28 ⇒ padding=1、3×3 核、stride=1 时尺寸不变
# 注意 PyTorch 的 Conv2d 实际做的是"互相关"（cross-correlation）：
#   它不翻转卷积核。手写实现时若要和它逐元素对齐，也不要翻转核。


def conv2d_manual(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None,
                  stride: int = 1, padding: int = 0) -> torch.Tensor:
    """手写二维卷积（支持 step 规定的多通道 / stride / padding）。

    参数：
        x      : (N, C_in, H, W)   输入特征图
        weight : (C_out, C_in, kh, kw)  卷积核（PyTorch 布局）
        bias   : (C_out,) 或 None
        stride : 滑动步长（行列相同）
        padding: 四周补 0 的圈数
    返回：
        (N, C_out, H_out, W_out)
    """
    N, C_in, H, W = x.shape
    C_out, C_in_w, kh, kw = weight.shape
    assert C_in == C_in_w, "卷积核的输入通道数必须等于输入的通道数"

    # 若给了 padding，就先用 F.pad 在 H/W 两维各补 padding 圈 0
    if padding > 0:
        x = F.pad(x, (padding, padding, padding, padding), mode="constant", value=0.0)
    Hp, Wp = x.shape[2], x.shape[3]

    # 按公式算输出尺寸（整除，向下取整）
    H_out = (Hp - kh) // stride + 1
    W_out = (Wp - kw) // stride + 1

    # 用 zeros 预分配输出，再逐位置填值
    out = torch.zeros(N, C_out, H_out, W_out, dtype=x.dtype)
    for n in range(N):                 # 遍历 batch
        for co in range(C_out):        # 遍历输出通道
            for i in range(H_out):     # 遍历输出的行
                for j in range(W_out):  # 遍历输出的列
                    # 取出当前窗口覆盖的输入块 (C_in, kh, kw)
                    window = x[n, :, i * stride:i * stride + kh, j * stride:j * stride + kw]
                    # 逐元素相乘再求和 = 内积；对所有输入通道同时求和
                    val = (window * weight[co]).sum()
                    if bias is not None:
                        val = val + bias[co]
                    out[n, co, i, j] = val
    return out


# 用一个小输入演示（8×8 输入、3×3 核），循环量很小，秒级完成
x_small = torch.randn(1, 1, 8, 8)                            # 单通道 8×8 输入
k_small = torch.randn(1, 1, 3, 3)                            # 1 个 3×3 卷积核
b_small = torch.randn(1)                                     # 1 个偏置

y_manual = conv2d_manual(x_small, k_small, b_small, stride=1, padding=0)
print(f"输入 shape  = {tuple(x_small.shape)}    # (N, C_in, H, W)")
print(f"卷积核 shape = {tuple(k_small.shape)}    # (C_out, C_in, kh, kw)")
print(f"输出 shape  = {tuple(y_manual.shape)}    # (N, C_out, H_out, W_out)")
print(f"尺寸推导：H_out = (8 + 2*0 - 3)/1 + 1 = {(8 + 0 - 3) // 1 + 1}")

# 打印输入左下角 3×3 的窗口和卷积核，手工验算一个位置
print(f"\n输入左上角 3×3 窗口 X[0,0,0:3,0:3]：")
for row in x_small[0, 0, 0:3, 0:3].numpy():
    print("   " + "  ".join(f"{v:+.4f}" for v in row))
print(f"卷积核 K[0,0]：")
for row in k_small[0, 0].numpy():
    print("   " + "  ".join(f"{v:+.4f}" for v in row))
hand_val = (x_small[0, 0, 0:3, 0:3] * k_small[0, 0]).sum() + b_small[0]
print(f"手工逐元素相乘求和 + 偏置 = {hand_val.item():+.6f}")
print(f"手写卷积输出的 (0,0) 位置 = {y_manual[0, 0, 0, 0].item():+.6f}"
      f"  ⇒ 相差 {(hand_val - y_manual[0, 0, 0, 0]).abs().item():.3e}")

# 打印完整的输出特征图数值（8x8 输入 -> 6x6 输出）
print(f"\n完整的输出特征图 (1, 1, 6, 6)：")
for row in y_manual[0, 0].numpy():
    print("   " + " ".join(f"{v:+7.3f}" for v in row))

# ===========================================================================
# 二、和 nn.Conv2d 逐元素对比（含 stride / padding）
# ===========================================================================
section("二、与 nn.Conv2d 对比：验证手写实现正确")

# 对比策略：把手工生成的权重/偏置塞进 nn.Conv2d，保证两边用同一组参数，
# 然后比较输出。nn.Conv2d 内部是高度优化的 C++/MKL 实现（im2col + GEMM），
# 手写版是朴素双重循环，两者结果应当逐元素一致（误差只来自 float32 累加顺序）。
test_cases = [
    # (输入形状,            输出通道, 核, 步长, 填充)
    ((2, 3, 10, 10), 4, 3, 1, 0),
    ((2, 3, 10, 10), 4, 3, 1, 1),
    ((2, 3, 10, 10), 2, 3, 2, 1),
    ((1, 2, 9, 9), 3, 5, 1, 2),
    ((1, 2, 9, 9), 2, 2, 3, 0),
]
print(f"{'输入 shape':<20}{'C_out':>6}{'k':>4}{'s':>3}{'p':>3}"
      f"{'输出 shape':>20}{'最大误差':>14}")
for shape, c_out, k, s, p in test_cases:
    xx = torch.randn(*shape)
    H, W = shape[2], shape[3]
    H_out = (H + 2 * p - k) // s + 1
    W_out = (W + 2 * p - k) // s + 1

    torch.manual_seed(0)                                   # 固定核参数，方便复现
    conv_ref = nn.Conv2d(shape[1], c_out, kernel_size=k, stride=s, padding=p, bias=True)
    y_ref = conv_ref(xx)                                   # PyTorch 的实现
    y_hand = conv2d_manual(xx, conv_ref.weight.detach(), conv_ref.bias.detach(),
                           stride=s, padding=p)            # 手写实现，用同一组参数
    err = (y_hand - y_ref.detach()).abs().max().item()
    formula = f"({H}+2*{p}-{k})/{s}+1={H_out}"
    print(f"{str(shape):<20}{c_out:>6}{k:>4}{s:>3}{p:>3}"
          f"{str((shape[0], c_out, H_out, W_out)):>20}{err:>14.3e}  ({formula})")
    assert err < 1e-5, f"手写卷积与 nn.Conv2d 不一致：误差 {err}"

print("\n结论：所有配置下手写卷积与 nn.Conv2d 的最大误差 < 1e-5（float32 舍入级别），"
      "说明卷积公式、stride、padding 的理解都是正确的。")

# ===========================================================================
# 三、多通道卷积：每个输出通道 = 所有输入通道卷积结果之和
# ===========================================================================
section("三、多通道卷积与形状推导")

# 通道维度的规则（初学最容易绕晕的地方）：
#   输入   : (N, C_in,  H, W)
#   卷积核 : (C_out, C_in, kh, kw)  ← 注意核的"输入通道数"必须等于 C_in
#   输出   : (N, C_out, H_out, W_out)
# 计算细节：第 co 个输出通道 = Σ_{ci=0}^{C_in-1} conv2d(X[:, ci], K[co, ci]) + b[co]
#   即"用 C_out 组卷积核，每组有 C_in 个通道的核，分别跟对应输入通道做卷积再求和"。
# 这样设计的意义：一个输出通道负责提取一种"局部模式"，
#   而它可以综合利用所有输入通道的信息（例如"红色竖直边缘 + 蓝色水平边缘的某种组合"）。
torch.manual_seed(0)
x_mc = torch.randn(2, 3, 12, 12)                    # batch=2，3 通道（如 RGB），12×12
conv_mc = nn.Conv2d(3, 8, kernel_size=3, padding=1)  # 3 通道 -> 8 通道
w_mc = conv_mc.weight.detach()                       # (8, 3, 3, 3)
b_mc = conv_mc.bias.detach()                         # (8,)
y_mc = conv_mc(x_mc)

# 手工逐通道验证："每个输出通道是所有输入通道卷积之和"
manual_ch0 = torch.zeros_like(y_mc[:, 0:1])
for ci in range(3):
    manual_ch0 = manual_ch0 + conv2d_manual(
        x_mc[:, ci:ci + 1], w_mc[0:1, ci:ci + 1], bias=None, stride=1, padding=1
    )
manual_ch0 = manual_ch0 + b_mc[0]                    # 最后加一次偏置（只加一次！）
err_ch0 = (manual_ch0[:, 0] - y_mc[:, 0].detach()).abs().max().item()

print(f"输入 x_mc.shape          = {tuple(x_mc.shape)}    # (N, C_in, H, W)")
print(f"卷积核 weight.shape      = {tuple(w_mc.shape)}    # (C_out, C_in, kh, kw)")
print(f"偏置 bias.shape          = {tuple(b_mc.shape)}       # (C_out,)")
print(f"输出 y_mc.shape          = {tuple(y_mc.shape)}   # (N, C_out, H_out, W_out)")
print(f"\n第 0 个输出通道 = 3 个输入通道分别卷积后相加，"
      f"与 nn.Conv2d 的最大误差 = {err_ch0:.3e}")
assert err_ch0 < 1e-5
print("注意：偏置只加一次，不是每个输入通道加一次！这是手写时常见的错点。")

print(f"\n参数量 = C_out × C_in × kh × kw + C_out "
      f"= 8×3×3×3 + 8 = {8 * 3 * 3 * 3} + {8} = {8 * 3 * 3 * 3 + 8}")
print(f"模型实际参数 = {sum(p.numel() for p in conv_mc.parameters())}")
print("对比：若把 3×12×12 的输入展平接全连接层到 8 个输出，"
      f"参数量是 3*12*12*8 + 8 = {3 * 12 * 12 * 8 + 8}，"
      "而且换成 13×13 的输入就完全不能用了 —— 这就是权重共享带来的泛化优势。")

del x_mc, y_mc, manual_ch0, conv_mc

# ===========================================================================
# 四、池化：最大池化与平均池化
# ===========================================================================
section("四、池化：最大池化与平均池化")

# 池化公式（课案）：
#   最大池化：(Y)_{i,j} = max_{0≤m<p_h, 0≤n<p_w} X_{i*s+m, j*s+n}
#   平均池化：(Y)_{i,j} = (1/(p_h·p_w)) · Σ_m Σ_n X_{i*s+m, j*s+n}
# 其中 p_h、p_w 是池化窗口的高和宽，s 是步长。
# 池化的作用：
#   1) 降低特征图尺寸 ⇒ 减少后续计算量和参数量；
#   2) 提供一定的"平移不变性"（小范围移动不改变最大值）；
#   3) 扩大后续层的感受野。
# 池化层没有可学习参数（没有权重、没有偏置），它只是一个固定的下采样算子。
# 输出尺寸同样满足：H_out = (H - p_h) / s + 1（池化一般不用 padding）。


def pool2d_manual(x: torch.Tensor, kernel_size: int, stride: int | None = None,
                  mode: str = "max") -> torch.Tensor:
    """手写池化（用 unfold 做滑动窗口，不用双重循环，速度更快）。

    x: (N, C, H, W)；返回 (N, C, H_out, W_out)。
    unfold 会把每个窗口里的元素摊成最后一维，于是"最大/平均"就是对最后一维做 reduce。
    """
    if stride is None:
        stride = kernel_size
    N, C, H, W = x.shape
    kh = kw = kernel_size
    H_out = (H - kh) // stride + 1
    W_out = (W - kw) // stride + 1

    # unfold(dimension=2, size=kh, step=stride)：在 H 维上滑窗 -> (N, C, H_out, W, kh)
    # 再 unfold(dimension=3, size=kw, step=stride)：在 W 维上滑窗 -> (N, C, H_out, W_out, kh, kw)
    windows = x.unfold(2, kh, stride).unfold(3, kw, stride)
    # 把 (kh, kw) 两维合并到最后一维：(N, C, H_out, W_out, kh*kw)
    windows = windows.reshape(N, C, H_out, W_out, kh * kw)
    if mode == "max":
        return windows.max(dim=-1).values          # 取每个窗口的最大值
    if mode == "mean":
        return windows.mean(dim=-1)                # 取每个窗口的平均值
    raise ValueError(f"不支持的池化模式：{mode}")


torch.manual_seed(0)
x_pool = torch.randn(2, 3, 12, 12)                 # 2 张图，3 通道，12×12

pool_ref = nn.MaxPool2d(kernel_size=2)             # 2×2、步长默认=2
y_pool_ref = pool_ref(x_pool)
y_pool_hand = pool2d_manual(x_pool, kernel_size=2, stride=2, mode="max")
err_pool = (y_pool_hand - y_pool_ref).abs().max().item()

print(f"输入 shape = {tuple(x_pool.shape)}")
print(f"最大池化：核 2、步长 2 ⇒ 输出 shape = {tuple(y_pool_ref.shape)}"
      f"   ((12-2)/2+1 = 6)")
print(f"手写 unfold 版最大池化 vs nn.MaxPool2d 的最大误差 = {err_pool:.3e}"
      f"  # 应为 0（都是取最大值，没有浮点累加）")
assert err_pool == 0.0, "最大池化的手写实现应当与 nn.MaxPool2d 完全相同"

# 平均池化对比
avg_ref = nn.AvgPool2d(kernel_size=2)
y_avg_ref = avg_ref(x_pool)
y_avg_hand = pool2d_manual(x_pool, kernel_size=2, stride=2, mode="mean")
err_avg = (y_avg_hand - y_avg_ref).abs().max().item()
print(f"手写版平均池化 vs nn.AvgPool2d 的最大误差 = {err_avg:.3e}"
      f"  # 浮点除法舍入级别")

# 打印小窗口的数值对比，直观展示 max / mean 的含义
print(f"\n取左上角 2×2 窗口 X[0,0,0:2,0:2]：")
win = x_pool[0, 0, 0:2, 0:2]
for row in win.numpy():
    print("   " + "  ".join(f"{v:+.4f}" for v in row))
print(f"   max = {win.max().item():+.4f}  -> 最大池化输出 (0,0) = "
      f"{y_pool_ref[0, 0, 0, 0].item():+.4f}")
print(f"   mean = {win.mean().item():+.4f}  -> 平均池化输出 (0,0) = "
      f"{y_avg_ref[0, 0, 0, 0].item():+.4f}")
print("\n池化层参数量 = " + str(sum(p.numel() for p in pool_ref.parameters()))
      + "（池化没有可学习参数）")

# 非整除情况演示：10×10 池化 3、步长 2 -> (10-3)/2+1 = 4
x_odd = torch.randn(1, 1, 10, 10)
odd_out = pool2d_manual(x_odd, kernel_size=3, stride=2, mode="max")
print(f"非整除示例：输入 10×10，核 3、步长 2 ⇒ 输出 {tuple(odd_out.shape)}"
      f"  ((10-3)/2+1 = 4，右侧/下侧最后 1 列被丢弃)")
print(f"  与 nn.MaxPool2d(3, stride=2) 最大误差 = "
      f"{(odd_out - nn.MaxPool2d(3, stride=2)(x_odd)).abs().max().item():.3e}")

del x_pool, y_pool_ref, y_pool_hand, y_avg_ref, y_avg_hand, x_odd, odd_out

# ===========================================================================
# 五、自适应平均池化 AdaptiveAvgPool2d
# ===========================================================================
section("五、nn.AdaptiveAvgPool2d：指定输出尺寸，而不是窗口大小")

# 普通池化：你给"窗口大小 + 步长"，输出尺寸由输入尺寸推出来（输入一变输出就变）。
# 自适应池化：你给"目标输出尺寸"，PyTorch 自动反推每个位置的窗口大小和步长。
#   最常用的是 output_size=(1, 1)：把任意 H×W 平均成一个数，
#   于是分类头就能接受任意分辨率的输入图像。
# 公式（自适应平均池化的窗口范围）：
#   start = floor(i * H / H_out)，end = ceil((i+1) * H / H_out)
adaptive = nn.AdaptiveAvgPool2d((1, 1))
in_a = torch.randn(2, 64, 7, 7)      # 常见 CNN 特征图尺寸
in_b = torch.randn(2, 64, 13, 13)    # 完全不同的分辨率
out_a = adaptive(in_a)
out_b = adaptive(in_b)
print(f"输入 {tuple(in_a.shape)} -> 输出 {tuple(out_a.shape)}")
print(f"输入 {tuple(in_b.shape)} -> 输出 {tuple(out_b.shape)}")
print("两次输入的空间尺寸不同（7×7 与 13×13），但输出都是 (2, 64, 1, 1) ⇒ "
      "分类头不用改，网络可以吃任意分辨率。")

# 验证 (1,1) 自适应平均池化就是把整个 H×W 求平均
check = in_a.mean(dim=(2, 3), keepdim=True)
print(f"与手工 mean(dim=(2,3)) 的最大误差 = {(out_a - check).abs().max().item():.3e}")

# 也可以指定非 1×1 的输出，例如 (2, 2) 把任意尺寸压成 2×2
adaptive22 = nn.AdaptiveAvgPool2d((2, 2))
print(f"\nAdaptiveAvgPool2d((2,2))：{tuple(in_b.shape)} -> {tuple(adaptive22(in_b).shape)}")
print(f"  {tuple(in_a.shape)} -> {tuple(adaptive22(in_a).shape)}"
      f"  # 输入尺寸不同，输出尺寸始终由我们指定")
print("对比 nn.AdaptiveMaxPool2d((1,1)) 同理，只是聚合算子换成 max。")

del in_a, in_b, out_a, out_b, check

# ===========================================================================
# 六、课案 CNN 结构（含逐层形状推导）
# ===========================================================================
section("六、课案 CNN 结构：逐层形状推导")


class CNN(nn.Module):
    """课案里的 CNN：两个卷积块 + 一个分类器。"""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        # 特征提取部分：卷积 + 激活 + 池化
        self.features = nn.Sequential(
            # 第一个卷积块
            # 输入通道 1（灰度图），输出通道 32，3×3 卷积核
            # padding=1 是指在卷积之前在输入特征图上下左右各补一行/列 0（加一圈"边框"），
            # 配合 3×3 核、步长 1 让 H/W 保持不变（(28+2-3)/1+1 = 28）
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),                                # 激活函数，引入非线性
            nn.MaxPool2d(kernel_size=2),              # 2×2 最大池化，尺寸减半：28 -> 14
            # 第二个卷积块
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # 输入通道 32，输出 64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),              # 尺寸再减半：14 -> 7
        )
        # 分类器部分：展平 + 全连接
        self.classifier = nn.Sequential(
            nn.Flatten(),                             # (N,64,7,7) -> (N, 64*7*7)
            nn.Linear(64 * 7 * 7, 128),               # 全连接降维到 128
            nn.ReLU(),
            nn.Linear(128, num_classes),              # 输出 10 个类别的 logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)      # 提取特征
        x = self.classifier(x)    # 分类输出
        return x


model = CNN(num_classes=10)
x_in = torch.randn(4, 1, 28, 28)     # batch=4，灰度图 28×28（用随机张量代替真实图像）
output = model(x_in)

print(f"输入形状：{tuple(x_in.shape)}")
print(f"输出形状：{tuple(output.shape)}")
print()

# 逐层形状推导（用"张量穿过每一层"的方式真实打印，而不是只写公式）
print("逐层形状推导过程（真实前向，逐层打印）：")
shape_rows = []
h = x_in
for idx, layer in enumerate(model.features, start=1):
    in_shape = tuple(h.shape)
    h = layer(h)
    out_shape = tuple(h.shape)
    extra = ""
    if isinstance(layer, nn.Conv2d):
        # 输出尺寸公式：(H + 2*padding - kh)/stride + 1
        kh, kw = layer.kernel_size
        s = layer.stride[0]
        p = layer.padding[0]
        extra = (f"  <- H_out = ({in_shape[2]} + 2*{p} - {kh})/{s} + 1 = {out_shape[2]}"
                 f"，通道 {in_shape[1]}->{out_shape[1]}（C_out 由我们指定）")
    elif isinstance(layer, nn.MaxPool2d):
        k = layer.kernel_size
        extra = f"  <- 池化 {k}×{k}，步长 {k}：{in_shape[2]} -> {out_shape[2]}"
    print(f"  {idx}. {layer.__class__.__name__:<12} {str(in_shape):<20} -> {str(out_shape):<20}{extra}")
    shape_rows.append((layer.__class__.__name__, in_shape, out_shape))

# 分类器部分（展平 + 两层全连接）
h2 = h
for idx, layer in enumerate(model.classifier, start=len(model.features) + 1):
    in_shape = tuple(h2.shape)
    h2 = layer(h2)
    out_shape = tuple(h2.shape)
    if isinstance(layer, nn.Flatten):
        extra = f"  <- 把 (N, {in_shape[1]}, {in_shape[2]}, {in_shape[3]}) 展平成 (N, {out_shape[1]})"
    elif isinstance(layer, nn.Linear):
        extra = f"  <- W 形状 {tuple(layer.weight.shape)}，即 (out_features, in_features)"
    else:
        extra = ""
    print(f"  {idx}. {layer.__class__.__name__:<12} {str(in_shape):<20} -> {str(out_shape):<20}{extra}")

print(f"\n尺寸链条总结：28 --(padding=1, 3×3 卷积)--> 28 --(池化 2)--> 14 "
      f"--(padding=1, 3×3 卷积)--> 14 --(池化 2)--> 7")
print(f"所以 Flatten 之后每一行的长度是 64 × 7 × 7 = {64 * 7 * 7}，"
      f"这也是 nn.Linear 的 in_features。")
assert tuple(output.shape) == (4, 10), "课案结构在 (4,1,28,28) 上应输出 (4,10)"

# ===========================================================================
# 七、参数量计算：手算 vs sum(p.numel())
# ===========================================================================
section("七、参数量计算（含权重共享对比）")

print("逐层参数量手算：")
conv1 = model.features[0]
conv2 = model.features[3]
fc1 = model.classifier[1]
fc2 = model.classifier[3]

# 卷积层参数量 = C_out × C_in × kh × kw + C_out(bias)
conv1_calc = 32 * 1 * 3 * 3 + 32
conv2_calc = 64 * 32 * 3 * 3 + 64
fc1_calc = 64 * 7 * 7 * 128 + 128
fc2_calc = 128 * 10 + 10

conv1_real = sum(p.numel() for p in conv1.parameters())
conv2_real = sum(p.numel() for p in conv2.parameters())
fc1_real = sum(p.numel() for p in fc1.parameters())
fc2_real = sum(p.numel() for p in fc2.parameters())

print(f"  Conv2d(1,32,3,p=1)   : C_out×C_in×kh×kw + C_out = 32×1×3×3 + 32 = "
      f"{32 * 1 * 3 * 3} + 32 = {conv1_calc}   (实际 {conv1_real})")
print(f"  Conv2d(32,64,3,p=1)  : 64×32×3×3 + 64 = {64 * 32 * 3 * 3} + 64 = "
      f"{conv2_calc}   (实际 {conv2_real})")
print(f"  Linear(64*7*7, 128)  : 3136×128 + 128 = {64 * 7 * 7 * 128} + 128 = "
      f"{fc1_calc}   (实际 {fc1_real})")
print(f"  Linear(128, 10)      : 128×10 + 10 = {128 * 10} + 10 = "
      f"{fc2_calc}   (实际 {fc2_real})")
total_real = sum(p.numel() for p in model.parameters())
print(f"  合计（手算） = {conv1_calc + conv2_calc + fc1_calc + fc2_calc}")
print(f"  合计（真实） = sum(p.numel() for p in model.parameters()) = {total_real}")
assert total_real == conv1_calc + conv2_calc + fc1_calc + fc2_calc

conv_total = conv1_real + conv2_real
print(f"\n卷积层参数合计 = {conv_total}，全连接层参数合计 = {fc1_real + fc2_real}"
      f"  ⇒ 参数量的 {100 * (fc1_real + fc2_real) / total_real:.1f}% 都在全连接层！")

# 权重共享的威力：同一个 3×3 核在整张 28×28 图上滑动，参数量与图像大小无关。
# 若第一个卷积层换成"等价的全连接层"：32 个输出通道 × 28×28 个位置，
# 每个位置都要独立看 1×3×3 的输入（局部连接），参数量是 32×28×28×9 = ...
fake_fc_conv = 32 * (28 * 28) * (1 * 3 * 3)
print(f"\n权重共享对比：")
print(f"  Conv2d(1,32,3,p=1) 实际参数 = {conv1_real}")
print(f"  若同样做『局部连接』但不共享权重（每个位置一套 3×3 核）："
      f"32 × 28×28 × 9 = {fake_fc_conv}，是前者的 {fake_fc_conv / conv1_real:.1f} 倍")
print(f"  而 nn.Linear(64*7*7, 128) 一个全连接层就有 {fc1_real} 个参数，"
      f"是第一个卷积层的 {fc1_real / conv1_real:.0f} 倍")
print("  ⇒ 权重共享 = 同一个卷积核在整张图上复用，参数少、还能处理任意输入尺寸，"
      "这是 CNN 能在图像上成功的关键。")
# ===========================================================================
# 八、感受野：堆叠 3×3 卷积扩大感受野
# ===========================================================================
section("八、感受野（Receptive Field）")

# 感受野 = 输出特征图上一个点，对应原始输入上多大的区域。
# 单层 3×3 卷积：感受野 3×3。
# 连续堆叠 k 层 3×3 卷积（stride=1）：感受野 = 2k + 1。
#   两层 -> 5×5；三层 -> 7×7。
# 为什么爱用"多个 3×3"而不是"一个 7×7"？
#   1) 参数更少：3 层 3×3 = 3 × 9 = 27 个权重/通道，
#      单个 7×7 = 49 个权重/通道（少 45%）；
#   2) 非线性更多：每层之间都夹 ReLU，表达能力更强（VGG 论文的核心论点）。
for k in (1, 2, 3, 4):
    rf = 2 * k + 1
    print(f"  堆叠 {k} 层 3×3 卷积 -> 感受野 {rf}×{rf}，"
          f"参数量（单通道、无偏置）= {k * 9}，"
          f"而单个 {rf}×{rf} 卷积核需要 {rf * rf} 个参数"
          f"（节省 {100 * (1 - k * 9 / (rf * rf)):.0f}%）")
print("在课案结构里：conv1 的感受野是 3×3，pool 后 conv2 的感受野膨胀到 7×7（在 28×28 原图上），"
      "所以第二层的神经元已经能『看到』中等尺度的结构。")

# ===========================================================================
# 九、可视化：卷积核网格 + 输入与输出特征图
# ===========================================================================
section("九、可视化：卷积核网格 + 特征图")

torch.manual_seed(42)
vis_model = CNN(num_classes=10)
x_vis = torch.randn(4, 1, 28, 28)     # 用随机张量代替真实图像（离线环境不下载数据集）

# 图 1：第一层 32 个 3×3 卷积核 + 一张输入图 + 它的输出特征图
w1 = vis_model.features[0].weight.detach().clone()      # (32, 1, 3, 3)
with torch.no_grad():
    feat1 = torch.relu(vis_model.features[0](x_vis))    # 第一层卷积 + ReLU 输出 (4, 32, 28, 28)
    pooled1 = vis_model.features[2](feat1)              # 池化后的特征图 (4, 32, 14, 14)
print(f"第一层卷积核 shape = {tuple(w1.shape)}   # (C_out, C_in, kh, kw)")
print(f"第一层卷积输出     = {tuple(feat1.shape)}   # 尺寸不变（padding=1, 3×3, stride=1）")
print(f"第一次池化输出     = {tuple(pooled1.shape)}   # 28 -> 14")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 7, height_ratios=[1.0, 1.3, 1.3], hspace=0.45, wspace=0.08)

# 第 1 行：32 个卷积核，拼成 4 行 8 列的大网格图
grid = np.zeros((4 * 3 + 3, 8 * 3 + 7))                 # 核之间留 1 像素空隙，便于区分
kmin, kmax = w1.min().item(), w1.max().item()
for idx in range(32):
    r, c = divmod(idx, 8)
    kimg = w1[idx, 0].numpy()
    kimg = (kimg - kmin) / (kmax - kmin + 1e-8)          # 归一化到 [0,1] 便于显示
    grid[r * 4:r * 4 + 3, c * 4:c * 4 + 3] = kimg
ax_k = fig.add_subplot(gs[0, :])
ax_k.imshow(grid, cmap="viridis")
ax_k.set_title("第一层 Conv2d(1,32,3,padding=1) 的 32 个 3×3 卷积核"
               "（随机初始化、未训练，所以看起来像噪声）", fontsize=12)
ax_k.set_xticks([c * 4 + 1 for c in range(8)])
ax_k.set_xticklabels([f"核{c + 1}" for c in range(8)])
ax_k.set_yticks([r * 4 + 1 for r in range(4)])
ax_k.set_yticklabels([f"核{r * 8 + 1}~{r * 8 + 8}" for r in range(4)])

# 第 2 行：输入图（28×28 随机噪声） + 第一层卷积输出的前 6 个通道
ax_in = fig.add_subplot(gs[1, 0])
ax_in.imshow(x_vis[0, 0].numpy(), cmap="gray")
ax_in.set_title("输入 1×28×28", fontsize=11)
ax_in.set_xticks([])
ax_in.set_yticks([])
for i in range(6):
    ax_c = fig.add_subplot(gs[1, i + 1])
    ax_c.imshow(feat1[0, i].numpy(), cmap="magma")
    ax_c.set_title(f"卷积输出 通道{i + 1}", fontsize=10)
    ax_c.set_xticks([])
    ax_c.set_yticks([])

# 第 3 行：池化后的前 6 个通道（尺寸减半，信息被压缩）
ax_p0 = fig.add_subplot(gs[2, 0])
ax_p0.imshow(pooled1[0, 0].numpy(), cmap="magma")
ax_p0.set_title("池化后 14×14", fontsize=11)
ax_p0.set_xticks([])
ax_p0.set_yticks([])
for i in range(6):
    ax_c = fig.add_subplot(gs[2, i + 1])
    ax_c.imshow(pooled1[0, i].numpy(), cmap="magma")
    ax_c.set_title(f"池化输出 通道{i + 1}", fontsize=10)
    ax_c.set_xticks([])
    ax_c.set_yticks([])

fig.suptitle("CNN 卷积核与特征图可视化（ReLU 后负响应被置 0，特征图偏稀疏）", fontsize=14)
path1 = OUTPUT_DIR / "02网络架构_03_CNN卷积核与特征图.png"
fig.savefig(path1, dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"已保存：{path1}")

# 图 2：卷积滑动窗口示意图
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6.5))

# 左图：6×6 输入上，3×3 核以 stride=1 滑动，用矩形框标出前几个窗口位置
demo = torch.arange(1, 37, dtype=torch.float32).reshape(6, 6)   # 编号 1~36，便于对照
ax = axes2[0]
im = ax.imshow(demo.numpy(), cmap="Blues", alpha=0.85)
for i in range(6):
    for j in range(6):
        ax.text(j, i, f"{int(demo[i, j])}", ha="center", va="center", fontsize=9)
# 标出 4 个窗口位置：分别对应输出 (0,0)、(0,1)、(1,0)、(0,2)
# 用矩形框 + 旁边的小字标注（不用图例，避免图例被挤出画布）
window_positions = [
    (0, 0, "#d62728", "窗口(0,0)", (0.55, -0.62)),
    (0, 1, "#2ca02c", "窗口(0,1)", (2.60, -0.62)),
    (1, 0, "#ff7f0e", "窗口(1,0)", (0.55, 1.62)),
    (0, 2, "#9467bd", "窗口(0,2)", (2.05, 0.35)),
]
for (wi, wj, color, label, (tx, ty)) in window_positions:
    ax.add_patch(Rectangle((wj - 0.5, wi - 0.5), 3, 3, fill=False,
                           edgecolor=color, linewidth=2.6))
    ax.text(tx, ty, label, color=color, fontsize=10, fontweight="bold")
ax.set_title("3×3 卷积核在 6×6 输入上滑动（stride=1）\n"
             "每个窗口做「逐元素相乘再求和」得到输出图上的一个值")
ax.set_xlabel("列 j")
ax.set_ylabel("行 i")
ax.set_xticks(range(6))
ax.set_yticks(range(6))
ax.set_xlim(-0.6, 5.6)
ax.set_ylim(5.6, -0.7)
fig2.colorbar(im, ax=ax, shrink=0.8, label="像素值")

# 右图：输出尺寸随 stride / padding 的变化表
ax2 = axes2[1]
ax2.axis("off")
table_data = [["输入 H", "kh", "padding", "stride", "H_out", "说明"]]
for H, k, p, s in [(28, 3, 1, 1), (28, 3, 1, 2), (28, 3, 0, 1), (28, 5, 2, 1),
                   (14, 3, 1, 1), (7, 7, 0, 1), (8, 3, 0, 1), (10, 3, 0, 2)]:
    out = (H + 2 * p - k) // s + 1
    note = "尺寸不变" if out == H else ("尺寸减半" if out * 2 == H else "尺寸变化")
    table_data.append([str(H), str(k), str(p), str(s), str(out), note])
table = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                  cellLoc="center", loc="center")
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.0, 1.7)
ax2.set_title("输出尺寸公式：H_out = (H + 2×padding - kh) / stride + 1\n"
              "（除法按 PyTorch 规则向下取整）", fontsize=12, pad=20)

fig2.suptitle("卷积滑动窗口示意与尺寸推导", fontsize=14)
fig2.tight_layout()
path2 = OUTPUT_DIR / "02网络架构_03_CNN滑动窗口示意.png"
fig2.savefig(path2, dpi=110)
plt.close(fig)
print(f"已保存：{path2}")

# ===========================================================================
# 十、CNN 组件说明表
# ===========================================================================
section("十、CNN 组件说明表")

rows = [
    ("Conv2d", "卷积层，提取局部特征；局部连接 + 权重共享，参数量 = C_out×C_in×kh×kw + C_out"),
    ("ReLU", "激活函数，增加非线性；把负响应置 0，保留强响应（特征图变稀疏）"),
    ("MaxPool2d", "池化层，降低特征图尺寸（28→14→7），提供平移不变性，无参数"),
    ("AvgPool2d", "平均池化，把窗口内数值取平均，输出更平滑"),
    ("AdaptiveAvgPool2d", "自适应池化，指定输出尺寸（常用 (1,1)），让分类头适配任意分辨率"),
    ("Flatten", "展平，把 (N,C,H,W) 变成 (N,C*H*W)，衔接卷积部分与全连接部分"),
    ("Linear", "全连接层，输出分类结果；参数量最大，通常只放在网络末端"),
]
for name, desc in rows:
    print(f"  {name:<20} {desc}")
print("\n形状流水线（N=4 的例子）：")
print("  (4,1,28,28) --conv1--> (4,32,28,28) --pool--> (4,32,14,14)")
print("              --conv2--> (4,64,14,14) --pool--> (4,64,7,7)")
print("              --flatten--> (4,3136) --fc--> (4,128) --fc--> (4,10)")

# ===========================================================================
# 十一、本节小结
# ===========================================================================
section("十一、本节小结")
print("1. 手写双重循环卷积已与 nn.Conv2d 在 5 组 stride/padding 配置下逐元素对齐"
      "（误差 < 1e-5）。")
print("2. 输出尺寸公式 H_out = (H + 2p - kh)/s + 1 是理解 CNN 形状变化的钥匙。")
print("3. 多通道卷积：输出第 co 个通道 = Σ_ci conv(X[ci], K[co,ci]) + b[co]，偏置只加一次。")
print("4. 池化无参数：手写 unfold 版最大池化与 nn.MaxPool2d 误差为 0。")
print("5. AdaptiveAvgPool2d((1,1)) 能把任意 H×W 压成 1×1，是分类头适配多分辨率的常规做法。")
print("6. 课案 CNN 在 (4,1,28,28) 上输出 (4,10)；参数量 421642，其中 95.5% 集中在")
print("   两个全连接层，卷积层只占 18816 —— 体现权重共享的参数量优势。")
print("7. 堆叠 3×3 卷积可以便宜地扩大感受野（k 层 -> 感受野 2k+1）。")

print(f"\n脚本总耗时：{time.perf_counter() - _T0:.2f} 秒")
print("脚本正常结束。")

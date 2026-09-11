"""
对应课案章节：训练组件 / 推理（模型保存与加载）

本节知识点：
    1.  完整走一遍课案流程：造数据 → DataLoader → 定义模型 → 交叉熵 + Adam → 训 5 个 epoch。
        数据：X = torch.randn(500, 10)，标签 y = (X[:,0] + X[:,1] > 0).long()。
    2.  **两种保存方式**：
        - 方式一（推荐）：torch.save(model.state_dict(), path)
          state_dict() 只保存权重和 bias（一个 OrderedDict），**不保存模型结构**。
          优点：文件小、跨版本兼容、加载前可以自由改模型结构。
        - 方式二：torch.save(model, path)
          把模型结构 + 参数一起 pickle 打包。优点：加载时不需要重新定义模型类，一行搞定。
          缺点：pickle 依赖类的定义路径（模块路径 + 类名），代码重构/换目录后加载会失败；
                而且 pickle 反序列化可以执行任意代码，有安全风险。
        **必须对比两种文件的大小**（用 Path.stat().st_size 打印字节数与 KB）。
    3.  **两种加载方式**，以及两个必须讲清楚的关键参数：
        - map_location="cpu"：torch.load 会把张量映射到指定设备。
          保存时在 GPU、加载时在没有 CUDA 的机器上，不加这个参数会直接报错；
          加上它就能把 GPU 权重安全地加载到 CPU。本机是 CPU 版，但代码里必须写上。
        - weights_only：**PyTorch 2.6 起 torch.load 的 weights_only 默认从 False 变成了 True**。
          weights_only=True 只允许反序列化「张量 + 基础容器」，安全性高，但**不能加载整个模型对象**；
          要加载方式二保存的完整模型，必须显式传 weights_only=False——
          这也说明「加载完整模型 = 执行 pickle」本身就是有风险的，只应加载可信文件。
          不同 torch 版本行为不同，本脚本用 try/except 兼容，并把两种情况都说明清楚。
    4.  model.eval() 的作用：切换到推理模式——
        · 关闭 Dropout（不再随机置 0，退化成恒等映射）；
        · 固定 BatchNorm 的统计量（用训练时累积的 running_mean / running_var，
          而不是当前 batch 的统计量），并且不再更新它们。
    5.  torch.no_grad() 的作用：关闭 autograd 的梯度记录。
        省掉计算图的内存与算力开销，推理更快、更省内存。
        **必须实际演示**：带 Dropout + BatchNorm 的模型在 train() 下多次前向结果不同、
        eval() 下完全相同；并打印 model.training 的变化。
    6.  保存/加载前后**预测一致性验证**：保存前记录 out_before，加载后得到 out_after，
        打印 torch.allclose(out_before, out_after, atol=1e-6) 应为 True，
        以及最大绝对误差（应为 0 或极小）。两种保存方式都要验证。
    7.  单条推理：x = torch.randn(1, 10) → output 形状 (1, 2)（两个类别的 logits）
        → pred = output.argmax(dim=1) → 打印预测类别；并打印 softmax 后的概率分布，
        讲清 **logits 与概率**的区别（logits 是未归一化的实数，softmax 才把它变成概率）。
    8.  批量推理与吞吐测量：1000 条样本、batch_size=256，用 time.perf_counter() 计时，
        打印「总耗时 / 平均每条耗时 / 吞吐（样本/秒）」，并对比 no_grad() 与非 no_grad() 的耗时。
    9.  checkpoint 完整惯例：torch.save({"epoch":..., "model_state_dict":..., 
        "optimizer_state_dict":..., "loss":...})——这样保存能**断点续训**：
        模型的参数、优化器的动量/二阶矩、训练进度全都在，加载回来可以继续训。
        本脚本实际保存并加载回来，打印恢复的 epoch 与 loss，并**继续训练 2 个 epoch** 证明可行。
    10. 安全提示：不要加载不可信的 .pt / .pth 文件（pickle 反序列化可执行任意代码）；
        生产环境优先用 state_dict，或 safetensors 这类「纯权重、不含代码」的格式。
    11. 可视化：训练 loss 曲线 + 推理耗时对比柱状图。
    12. 权重存储格式对比表：PyTorch / Safetensors / GGUF / 量化格式 / 分片格式 /
        ONNX / TensorFlow（**本机没有 onnx / safetensors / tensorflow，只做文字讲解，不 import**）。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Deep_Learning\\03_训练组件\\08_模型保存与推理.py'
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
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)                    # 小数据量下限制线程，避免 OpenMP 调度开销拖慢速度

print("=" * 78)
print("08 模型保存与推理：state_dict / 完整模型 / eval + no_grad / 断点续训")
print("=" * 78)
print(f"PyTorch 版本：{torch.__version__}")
print()


# ===========================================================================
# 第 1 部分：准备数据 + 训练一个能用的模型（课案流程）
# ===========================================================================
print("=" * 78)
print("第 1 部分：准备数据 + 训练（课案流程）")
print("=" * 78)
print(
    """
课案流程：
    1. 造数据：X = torch.randn(500, 10)（500 条 10 维特征），
       标签 y = (X[:, 0] + X[:, 1] > 0).long()——只看前两个特征的和是否为正，
       所以这是一个**线性可分**的简单二分类任务。
    2. 包成 DataLoader(batch_size=32, shuffle=True)。
    3. 模型：nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))。
    4. 损失 CrossEntropyLoss + 优化器 Adam(lr=0.01)。
    5. 训 5 个 epoch。

注意：模型最后一层**不接 softmax**。CrossEntropyLoss 内部自带 log_softmax，
      所以 forward 的输出是原始 logits（未归一化分数）——这也是推理时我们需要
      自己再手动 softmax 一次才能拿到概率的原因。
"""
)

torch.manual_seed(42)
X = torch.randn(500, 10)                            # 500 条样本，每条 10 维
y = (X[:, 0] + X[:, 1] > 0).long()                  # 标签：前两个特征之和是否为正
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
print(f"数据形状：X = {tuple(X.shape)}，y = {tuple(y.shape)}，"
      f"正类占比 = {y.float().mean().item():.3f}")
print(f"类别分布：0 类 {(y == 0).sum().item()} 条，1 类 {(y == 1).sum().item()} 条")
print(f"DataLoader：batch_size=32, shuffle=True → 每个 epoch {len(loader)} 个 batch")
print()

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("模型结构：")
print(model)
print(f"参数总量：{sum(p.numel() for p in model.parameters())} 个")
print()

_TRAIN_EPOCHS = 5
_loss_curve: list[float] = []
_acc_curve: list[float] = []
print(f"开始训练 {_TRAIN_EPOCHS} 个 epoch：")
for epoch in range(1, _TRAIN_EPOCHS + 1):
    model.train()
    _epoch_loss = 0.0
    for Xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        _epoch_loss += loss.item() * Xb.size(0)
    _train_loss = _epoch_loss / len(loader.dataset)
    _loss_curve.append(_train_loss)
    model.eval()
    with torch.no_grad():
        _acc = (model(X).argmax(dim=1) == y).float().mean().item()
    _acc_curve.append(_acc)
    print(f"  epoch {epoch}: 训练 loss = {_train_loss:.6f}，全量数据准确率 = {_acc * 100:.2f}%")
print("训练完成。下面保存这个模型。")
print()


# ===========================================================================
# 第 2 部分：两种保存方式 + 文件大小对比
# ===========================================================================
print("=" * 78)
print("第 2 部分：两种保存方式（state_dict vs 整个模型）+ 文件大小对比")
print("=" * 78)
print(
    """
方式一（推荐）：只保存参数
    torch.save(model.state_dict(), "model_state.pt")
    · state_dict() 返回一个 OrderedDict[str, Tensor]，键是参数名、值是对应的张量；
    · **只有权重和 bias，没有模型结构**（也就是没有「层与层的连接方式」这份信息）；
    · 优点：文件小、跨 PyTorch 版本兼容性好、加载前可以自由修改模型结构
            （只要参数名对得上，甚至可以把权重塞进一个结构不同的模型里）；
    · 加载时必须先用代码**重新定义一份相同结构的模型**，再 load_state_dict。

方式二：保存整个模型
    torch.save(model, "model_full.pt")
    · 用 pickle 把「模型类 + 类的定义路径 + 全部参数」一起打包成一个文件；
    · 优点：加载时不需要重新定义模型类，torch.load 一行就把模型拿回来；
    · 缺点 1：pickle 记录的是**类的定义路径**（模块路径 + 类名）。
             代码重构后（改了文件位置、改了类名）再加载就会失败，非常脆弱；
    · 缺点 2：**安全风险**——pickle 反序列化时会按文件里的描述去导入模块、调用对象，
             恶意文件可以借此执行任意代码。所以只能用 torch.load 加载**可信来源**的文件。
"""
)

_state_path = OUTPUT_DIR / "08_model_state.pt"
_full_path = OUTPUT_DIR / "08_model_full.pt"

# 方式一：只保存参数
torch.save(model.state_dict(), _state_path)
print(f"方式一已保存：{_state_path.name}")
print(f"  state_dict 的键（参数名 → 形状）：")
for _k, _v in model.state_dict().items():
    print(f"    {_k:<24} {tuple(_v.shape)}  dtype={_v.dtype}")

# 方式二：保存整个模型
torch.save(model, _full_path)
print(f"\n方式二已保存：{_full_path.name}（用 pickle 打包了模型类 + 参数）")
print()

# --- 文件大小对比 ---
_state_bytes = _state_path.stat().st_size
_full_bytes = _full_path.stat().st_size
print("【文件大小对比】（Path.stat().st_size 返回字节数）")
print(f"  方式一 只存参数 {_state_path.name:<22}: {_state_bytes:>8,} 字节 = "
      f"{_state_bytes / 1024:>8.2f} KB")
print(f"  方式二 整个模型 {_full_path.name:<22}: {_full_bytes:>8,} 字节 = "
      f"{_full_bytes / 1024:>8.2f} KB")
print(f"  方式二是方式一的 {_full_bytes / _state_bytes:.2f} 倍（多出 "
      f"{( _full_bytes - _state_bytes) / 1024:.2f} KB）")
print(f"  多出来的部分是 pickle 保存的**模型结构描述 + 类的定义路径等元信息**，而不是参数本身。")
print(f"  这个模型很小（{sum(p.numel() for p in model.parameters())} 个参数），"
      f"元信息的占比就显得很大；")
print(f"  在几亿参数的大模型上，元信息占比会降到可忽略，但「脆弱 + 不安全」两个缺点依然存在。")
print()


# ===========================================================================
# 第 3 部分：两种加载方式（map_location 与 weights_only）
# ===========================================================================
print("=" * 78)
print("第 3 部分：两种加载方式 + 两个关键参数（map_location / weights_only）")
print("=" * 78)
print(
    """
【关键参数 1：map_location="cpu"】
    torch.load 保存的张量里记录了「它原来在哪个设备上」。
    如果模型是在 GPU 上训练并保存的，直接 torch.load 会尝试把张量放回 GPU，
    在没有 CUDA 的机器上就报错（找不到 cuda 设备）。
    map_location="cpu" 告诉 PyTorch：把所有权重都映射到 CPU 上，
    这样无论在什么机器上都能安全加载。
    本机是 torch 2.14.0+cpu（没有 CUDA），但真实项目里这行必须写。

【关键参数 2：weights_only】
    · PyTorch 2.6 之前，torch.load 的 weights_only 默认是 False
      → 允许反序列化任意 pickle 对象（所以能加载整个模型，但**不安全**）；
    · PyTorch 2.6 起，默认值改成了 True
      → 只允许反序列化「张量 + 基础 Python 容器」，安全性大幅提高，
        但**加载整个模型对象会直接报错**，必须显式写 weights_only=False。
    · 所以：加载 state_dict 时保持默认（True）最安全；
            加载完整模型时必须 weights_only=False，并且一定要确认文件来源可信。
    不同 torch 版本的默认行为不一样，下面用 try/except 把两种情况都演示清楚。
"""
)

# --- 方式一：重建结构 + load_state_dict ---
loaded_model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 2))   # 先重建同样的结构
_state_loaded = torch.load(_state_path, map_location="cpu")                     # 必须写 map_location
loaded_model.load_state_dict(_state_loaded)                                     # 再把参数灌进去
print("方式一加载成功：先重建 nn.Sequential(...) 结构，再 load_state_dict")
print(f"  torch.load 返回类型：{type(_state_loaded).__name__}，"
      f"含 {len(_state_loaded)} 个参数项")
print(f"  加载后参数名：{list(_state_loaded.keys())}")

# --- 方式二：torch.load 整个模型（处理 weights_only 的版本差异）---
print("\n方式二加载：torch.load(path, weights_only=False)")
loaded_full = None
try:
    loaded_full = torch.load(_full_path, map_location="cpu", weights_only=False)
    print(f"  显式传 weights_only=False 成功，返回类型：{type(loaded_full).__name__}")
except TypeError as exc:
    # 老版本 torch 的 torch.load 没有 weights_only 这个关键字参数
    print(f"  当前 torch 版本不支持 weights_only 关键字（{type(exc).__name__}），"
          f"改用不带该参数的传统写法")
    try:
        loaded_full = torch.load(_full_path, map_location="cpu")
        print(f"  加载成功，返回类型：{type(loaded_full).__name__}")
    except Exception as exc2:                    # 兜底：任何加载失败都优雅说明，不打印堆栈
        print(f"  仍然失败（{type(exc2).__name__}），本机无法演示完整模型加载。")
except Exception as exc:                         # 例如 weights_only 默认 True 但没显式关掉
    print(f"  加载完整模型失败（{type(exc).__name__}）——这通常是因为当前 torch 的 "
          f"weights_only 默认为 True，")
    print(f"  而只允许反序列化张量、不允许还原整个模型对象；解决办法就是显式写 "
          f"weights_only=False（只对可信文件这么做）。")
    loaded_full = None

# 再演示一次：**故意**用默认的 weights_only=True 去加载完整模型，看会发生什么
print("\n对照实验：用默认设置（weights_only=True，PyTorch 2.6+ 的默认值）加载完整模型")
try:
    _probe = torch.load(_full_path, map_location="cpu", weights_only=True)
    print(f"  竟然成功了（说明当前 torch 的默认行为仍是 False 或允许该对象），返回 {type(_probe).__name__}")
except Exception as exc:
    print(f"  失败并被捕获 → 异常类型：{type(exc).__name__}")
    print(f"  这正是 weights_only=True 的设计目的：拒绝反序列化「非纯数据」的对象，")
    print(f"  从而避免 pickle 反序列化执行恶意代码。要加载完整模型必须显式关掉这个保护。")
print()


# ===========================================================================
# 第 4 部分：eval() + no_grad() 的作用与实际差异
# ===========================================================================
print("=" * 78)
print("第 4 部分：model.eval() 与 torch.no_grad() 的作用（必须实际演示差异）")
print("=" * 78)
print(
    """
【model.eval()】把模型切换到推理模式，影响所有「训练/推理行为不同」的层：
    · Dropout：训练时按概率 p 随机置 0；eval 时**关闭**，退化成恒等映射（不做任何丢弃）；
    · BatchNorm：训练时用**当前 batch** 的 μ、σ² 归一化，并更新 running_mean / running_var；
                eval 时用训练时累积的 **running_mean / running_var**（全局统计量），且不再更新。
    · 其他层（Linear、Conv、ReLU）没有差异。
    对应地，model.train() 切回训练模式。默认新建的模型就是 training=True。

【torch.no_grad()】关闭 autograd 的梯度记录：
    · 前向计算不再构建计算图 → 省下保存中间激活值和图结构的内存；
    · 少了反向图构建的开销 → 前向本身也更快；
    · 推理阶段完全不需要梯度（不 backward、不 optimizer.step），所以一定要包上它。
    · 与 eval() 是两个**独立**的概念：
        eval()   控制「层的计算行为」（Dropout/BatchNorm）
        no_grad() 控制「要不要记录梯度」
      二者经常一起出现，但一个不能替代另一个。忘记 eval() 会让推理结果变差且不确定；
      忘记 no_grad() 只会浪费内存和速度，不会算错。
"""
)

# --- 4.1 带 Dropout + BatchNorm 的模型 ---
torch.manual_seed(42)


class DemoNet(nn.Module):
    """带 Dropout 和 BatchNorm1d 的小模型，专门用来演示 train() / eval() 的行为差异。"""

    def __init__(self, in_dim: int = 10, hidden: int = 32, out_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.bn = nn.BatchNorm1d(hidden)     # 训练用 batch 统计量，推理用 running 统计量
        self.drop = nn.Dropout(0.5)          # 训练时随机丢弃 50%
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.fc1(x)))
        x = self.drop(x)
        return self.fc2(x)


_demo_net = DemoNet()
# 先跑几次 train 模式的前向，让 BatchNorm 的 running 统计量被更新（不然它们还是初始的 0/1）
_demo_net.train()
for _ in range(20):
    _demo_net(X[:64])
_x_demo = X[:8]                                   # 8 条样本，固定不变
print(f"演示模型：Linear(10→32) → BatchNorm1d(32) → ReLU → Dropout(0.5) → Linear(32→2)")
print(f"BatchNorm 累积后的 running_mean 前 4 个 = "
      f"{[round(v, 4) for v in _demo_net.bn.running_mean[:4].tolist()]}")
print(f"BatchNorm 累积后的 running_var  前 4 个 = "
      f"{[round(v, 4) for v in _demo_net.bn.running_var[:4].tolist()]}")
print()

# --- train() 模式 ---
print(f"model.training 当前值 = {_demo_net.training}（新建时默认是 True）")
_demo_net.train()
print(f"调用 model.train() 之后，model.training = {_demo_net.training}")
_train_outs = [torch.softmax(_demo_net(_x_demo), dim=1) for _ in range(3)]
_same_in_train = all(torch.equal(_train_outs[0], o) for o in _train_outs[1:])
print(f"  train() 模式下对**同一个输入**连续前向 3 次，第 0 条样本的预测概率：")
for _i, _o in enumerate(_train_outs, start=1):
    print(f"    第{_i}次: {[round(v, 5) for v in _o[0].tolist()]}")
print(f"  3 次结果完全相同？→ {_same_in_train}"
      f"（{'被 Dropout 的随机性打乱，应该为 False' if not _same_in_train else '不该相同'}）")
print(f"  3 次之间的最大差异 = "
      f"{max((_train_outs[0] - o).abs().max().item() for o in _train_outs[1:]):.6f}")
print()

# --- eval() 模式 ---
_demo_net.eval()
print(f"调用 model.eval() 之后，model.training = {_demo_net.training}")
_eval_outs = [torch.softmax(_demo_net(_x_demo), dim=1) for _ in range(3)]
_same_in_eval = all(torch.equal(_eval_outs[0], o) for o in _eval_outs[1:])
print(f"  eval() 模式下对**同一个输入**连续前向 3 次，第 0 条样本的预测概率：")
for _i, _o in enumerate(_eval_outs, start=1):
    print(f"    第{_i}次: {[round(v, 5) for v in _o[0].tolist()]}")
print(f"  3 次结果完全相同？→ {_same_in_eval}（推理时 Dropout 关闭、BatchNorm 用固定统计量，应该为 True）")
print(f"  3 次之间的最大差异 = "
      f"{max((_eval_outs[0] - o).abs().max().item() for o in _eval_outs[1:]):.3e}")
print()

# --- train() vs eval() 的整体输出差异 ---
_demo_net.train()
_out_train_mode = _demo_net(_x_demo).detach()
_demo_net.eval()
with torch.no_grad():
    _out_eval_mode = _demo_net(_x_demo)
print(f"  同一个输入在 train() 与 eval() 下的 logits 最大差异 = "
      f"{(_out_train_mode - _out_eval_mode).abs().max().item():.6f}")
print(f"  说明：train() 下 Dropout 随机丢弃 + BatchNorm 用当前 batch 统计量，")
print(f"        所以输出不仅「不确定」，连数值尺度都和推理时不一样；")
print(f"        评估/推理前忘记 eval() 是常见 bug —— 指标会抖动、还会系统性偏低/偏高。")
print()

# --- no_grad() 的作用演示 ---
print("no_grad() 的作用演示（观察 requires_grad 与内存开销）：")
with torch.no_grad():
    _out_ng = _demo_net(_x_demo)
_demo_net.eval()
_out_g = _demo_net(_x_demo)               # 不包 no_grad，会构建计算图
print(f"  包 no_grad()：输出 requires_grad = {_out_ng.requires_grad}，"
      f"grad_fn = {_out_ng.grad_fn}")
print(f"  不包 no_grad()：输出 requires_grad = {_out_g.requires_grad}，"
      f"grad_fn = {type(_out_g.grad_fn).__name__}")
print(f"  → 不包 no_grad 时，输出带着整条计算图的引用（grad_fn），")
print(f"     中间激活值都被保留着，白占内存；推理时不需要它们。")
print(f"  两种情况下数值结果相同？{torch.allclose(_out_ng, _out_g, atol=1e-6)}"
      f"（差异只影响内存与速度，不影响数值）")
print()


# ===========================================================================
# 第 5 部分：加载后的预测一致性验证
# ===========================================================================
print("=" * 78)
print("第 5 部分：保存 / 加载前后预测一致性验证")
print("=" * 78)

torch.manual_seed(7)
x_test = torch.randn(64, 10)                     # 64 条独立的测试样本

# 保存前的输出（**必须先 eval()**，否则 Dropout/BatchNorm 会让输出带随机性，无法比较）
model.eval()
with torch.no_grad():
    out_before = model(x_test)
print(f"保存前：model.eval() + no_grad() 得到 out_before，形状 {tuple(out_before.shape)}")
print(f"  前 2 条样本的 logits：{[[round(v, 5) for v in row] for row in out_before[:2].tolist()]}")
print()

# --- 方式一的一致性 ---
loaded_model.eval()
with torch.no_grad():
    out_after_state = loaded_model(x_test)
_ok_state = torch.allclose(out_before, out_after_state, atol=1e-6)
_max_err_state = (out_before - out_after_state).abs().max().item()
print("方式一（state_dict）加载后：")
print(f"  torch.allclose(out_before, out_after, atol=1e-6) = {_ok_state}")
print(f"  最大绝对误差 = {_max_err_state:.3e}"
      f"  → {'完全一致 ✔（权重被逐位还原）' if _max_err_state == 0.0 else '存在极小浮点误差'}")
print()

# --- 方式二的一致性 ---
if loaded_full is not None:
    loaded_full.eval()
    with torch.no_grad():
        out_after_full = loaded_full(x_test)
    _ok_full = torch.allclose(out_before, out_after_full, atol=1e-6)
    _max_err_full = (out_before - out_after_full).abs().max().item()
    print("方式二（整个模型）加载后：")
    print(f"  torch.allclose(out_before, out_after, atol=1e-6) = {_ok_full}")
    print(f"  最大绝对误差 = {_max_err_full:.3e}"
          f"  → {'完全一致 ✔' if _max_err_full == 0.0 else '存在极小浮点误差'}")
else:
    out_after_full = None
    _ok_full = None
    _max_err_full = float("nan")
    print("方式二未能完成加载（见第 3 部分的说明），跳过一致性验证。")
print()
print("  为什么误差恰好是 0？因为两种保存方式都只是把同一批 float32 张量写进文件、再读回来，")
print("  没有任何数值计算发生——序列化/反序列化不会改变浮点数的位模式。")
print()


# ===========================================================================
# 第 6 部分：单条推理——logits vs 概率
# ===========================================================================
print("=" * 78)
print("第 6 部分：单条推理——logits 与 softmax 概率的区别")
print("=" * 78)

x = torch.randn(1, 10)                           # 一条样本，10 维特征
loaded_model.eval()
with torch.no_grad():
    output = loaded_model(x)                     # (1, 2)：两个类别的 logits
    pred = output.argmax(dim=1)                  # 取 logits 最大的类别作为预测结果
    prob = torch.softmax(output, dim=1)          # 手动 softmax 才得到概率

print(f"输入 x 形状：{tuple(x.shape)}（1 条样本 × 10 维特征）")
print(f"output = loaded_model(x) 形状：{tuple(output.shape)}"
      f" → 1 条样本 × 2 个类别的 **logits**")
print(f"  logits 数值：{output[0].tolist()}")
print(f"pred = output.argmax(dim=1) → 预测类别：{pred.item()}")
print(f"  注意 pred 的形状是 {tuple(pred.shape)}，"
      f"pred.item() 取出标量（batch=1 时才有意义）")
print()
print(f"prob = torch.softmax(output, dim=1)：")
print(f"  概率分布：{prob[0].tolist()}")
print(f"  概率之和 = {prob.sum().item():.10f}（softmax 保证各类概率之和为 1）")
print(f"  最大概率 = {prob.max().item():.6f}，对应类别 = {prob.argmax(dim=1).item()}"
      f"（与 argmax(logits) 的结果一致 ✔）")
print()
print("  logits vs 概率：")
print("    · logits 是模型最后一层的**原始输出**，取值 (-∞, +∞)，可以任意大或任意负；")
print("      它只表示「相对偏好」，本身不是概率，也不满足和为 1。")
print("    · softmax(logits) 把 logits 映射成概率：p_i = exp(z_i) / Σ_j exp(z_j)。")
print("      它是单调变换 → **不改变 argmax 的结果**，所以只看类别时不必算 softmax。")
print("    · 顺序很重要：p_i 的排序和 z_i 完全一致；但两个 logits 的间隔大小决定概率的「信心」。")
print(f"      本例两个 logits 相差 {abs(output[0, 0].item() - output[0, 1].item()):.4f}，"
      f"对应的最大概率是 {prob.max().item():.4f}。")
print("    · 为什么不在模型里直接加 softmax？因为 CrossEntropyLoss 内部用的是 "
      "log_softmax + NLLLoss，")
print("      数值上比「先 softmax 再取 log」稳定得多，从 logits 出发能避免溢出。")
print()


# ===========================================================================
# 第 7 部分：批量推理与吞吐测量（no_grad vs 不 no_grad）
# ===========================================================================
print("=" * 78)
print("第 7 部分：批量推理与吞吐测量")
print("=" * 78)

torch.manual_seed(11)
X_infer = torch.randn(1000, 10)                  # 1000 条样本
_INFER_BATCH = 256                               # 每批 256 条


def run_inference(net: nn.Module, data: torch.Tensor, batch_size: int,
                  use_no_grad: bool) -> tuple[float, torch.Tensor]:
    """按 batch 分块做推理，返回 (总耗时秒, 拼好的预测结果)。

    use_no_grad=True 时用 torch.no_grad() 包住前向，不构建计算图。
    """
    net.eval()                                   # 推理前一定要 eval()
    _preds = []
    _t0 = time.perf_counter()
    if use_no_grad:
        with torch.no_grad():
            for _i in range(0, len(data), batch_size):
                _preds.append(net(data[_i:_i + batch_size]).argmax(dim=1))
    else:
        for _i in range(0, len(data), batch_size):
            _preds.append(net(data[_i:_i + batch_size]).argmax(dim=1))
    _elapsed = time.perf_counter() - _t0
    return _elapsed, torch.cat(_preds)


# 每个配置跑 5 次取平均，减少 CPU 调度抖动带来的测量误差
_N_REPEAT = 5
_time_ng = []
_time_g = []
_pred_ng = None
_pred_g = None
for _ in range(_N_REPEAT):
    _t, _p = run_inference(loaded_model, X_infer, _INFER_BATCH, use_no_grad=True)
    _time_ng.append(_t)
    _pred_ng = _p
    _t2, _p2 = run_inference(loaded_model, X_infer, _INFER_BATCH, use_no_grad=False)
    _time_g.append(_t2)
    _pred_g = _p2

_avg_ng = sum(_time_ng) / len(_time_ng)
_avg_g = sum(_time_g) / len(_time_g)
print(f"推理配置：1000 条样本，batch_size={_INFER_BATCH}，每个配置重复 {_N_REPEAT} 次取平均")
print()
print(f"{'配置':<26}{'总耗时(ms)':>14}{'平均每条(μs)':>16}{'吞吐(样本/秒)':>16}")
print("-" * 74)
print(f"{'with torch.no_grad()':<26}{_avg_ng * 1000:>14.3f}{_avg_ng / 1000 * 1e6:>16.2f}"
      f"{1000 / _avg_ng:>16,.0f}")
print(f"{'不用 no_grad()':<26}{_avg_g * 1000:>14.3f}{_avg_g / 1000 * 1e6:>16.2f}"
      f"{1000 / _avg_g:>16,.0f}")
_speedup = _avg_g / _avg_ng
print()
print(f"  两次的预测结果是否完全一致？{torch.equal(_pred_ng, _pred_g)}"
      f"（no_grad 只影响内存和速度，不影响数值结果）")
print(f"  no_grad() 带来的加速比 = {_speedup:.3f}×（耗时降低 "
      f"{(1 - _avg_ng / _avg_g) * 100:.1f}%）")
if _speedup < 1.1:
    print("  说明：本模型非常小（只有 "
          f"{sum(p.numel() for p in loaded_model.parameters())} 个参数），"
          f"省下的计算图构建开销只占每步开销的一小部分，")
    print("        所以加速比不明显——**这是如实测量**。在真实的大模型上（尤其是深层网络、"
          "大 batch）")
    print("        计算图会保存大量中间激活值，no_grad() 能省下可观的内存与时间。")
print(f"  衡量推理性能的两个常用指标：")
print(f"    · 延迟 latency：单条（或单批）耗时 = {_avg_ng / 1000 * 1e6:.2f} 微秒/条（这里取平均每条）")
print(f"    · 吞吐 throughput：单位时间能处理多少条 = {1000 / _avg_ng:,.0f} 样本/秒")
print()


# ===========================================================================
# 第 8 部分：checkpoint 完整惯例（断点续训）
# ===========================================================================
print("=" * 78)
print("第 8 部分：checkpoint 完整惯例——保存训练进度以便断点续训")
print("=" * 78)
print(
    """
只保存 model.state_dict() 的模型，加载后只能「用」；但如果训练被中断（断电、超时、
    抢占），你想接着训，那**优化器的状态**也是必须的：
    · Adam 维护每个参数的动量（一阶矩）和平方梯度（二阶矩），
      丢掉它们会让优化器「失忆」，重新开始累积 → 恢复后的头几步 loss 会跳一下；
    · 还有训练进度（第几个 epoch）、当前的 loss、学习率调度器的状态等。

标准做法是把这些东西打包成一个 dict 一起存：

    torch.save({
        "epoch": epoch,                       # 训练到第几个 epoch
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,                         # 当前 loss，便于监控
        "scheduler_state_dict": scheduler.state_dict(),   # 用了调度器才需要（本脚本没用到）
    }, "checkpoint.pt")

恢复时按同样的键取回来，并调用 optimizer.load_state_dict(...) 把动量等状态灌回去。
这样 resume 之后训练曲线能**无缝接上**，跟没中断过一样。
"""
)

_ckpt_path = OUTPUT_DIR / "08_checkpoint.pt"
_last_loss = _loss_curve[-1]
torch.save(
    {
        "epoch": _TRAIN_EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": _last_loss,
    },
    _ckpt_path,
)
print(f"已保存 checkpoint：{_ckpt_path.name}（{_ckpt_path.stat().st_size / 1024:.2f} KB）")
print(f"  保存的键：epoch={_TRAIN_EPOCHS}, loss={_last_loss:.6f}, "
      f"model_state_dict（{len(model.state_dict())} 项）, "
      f"optimizer_state_dict（{len(optimizer.state_dict())} 项：state + param_groups）")

# --- 加载回来并继续训练 ---
print("\n加载 checkpoint 并继续训练：")
ckpt = torch.load(_ckpt_path, map_location="cpu")           # 纯 dict + 张量，默认 weights_only 即可
print(f"  恢复的 epoch = {ckpt['epoch']}，恢复的 loss = {ckpt['loss']:.6f}")

model.load_state_dict(ckpt["model_state_dict"])             # 恢复模型参数
optimizer.load_state_dict(ckpt["optimizer_state_dict"])     # 恢复优化器状态（动量/二阶矩）
print(f"  优化器状态已恢复：param_groups 里的 lr = "
      f"{optimizer.param_groups[0]['lr']}，"
      f"state 中记录了 {len(optimizer.state_dict()['state'])} 个参数的动量缓存")

model.eval()
with torch.no_grad():
    _acc_after_resume = (model(X).argmax(dim=1) == y).float().mean().item()
    _loss_after_resume = criterion(model(X), y).item()
print(f"  恢复后的模型：全量 loss = {_loss_after_resume:.6f}，准确率 = "
      f"{_acc_after_resume * 100:.2f}%（与保存前一致）")

# 继续训练 2 个 epoch
_RESUME_EPOCHS = 2
_resume_losses: list[float] = []
for _r in range(1, _RESUME_EPOCHS + 1):
    model.train()
    _epoch_loss = 0.0
    for Xb, yb in loader:
        optimizer.zero_grad()
        _l = criterion(model(Xb), yb)
        _l.backward()
        optimizer.step()
        _epoch_loss += _l.item() * Xb.size(0)
    _train_loss = _epoch_loss / len(loader.dataset)
    _resume_losses.append(_train_loss)
    print(f"  续训第 {_r} 个 epoch（总第 {_TRAIN_EPOCHS + _r} 个）：训练 loss = {_train_loss:.6f}")
model.eval()
with torch.no_grad():
    _acc_after_2 = (model(X).argmax(dim=1) == y).float().mean().item()
print(f"  续训 {_RESUME_EPOCHS} 个 epoch 后准确率 = {_acc_after_2 * 100:.2f}%"
      f"  → {'续训流程可行 ✔' if _acc_after_2 >= _acc_after_resume - 1e-3 else '指标下降，需要检查'}")
print("  → 断点续训的关键就在于把 model_state_dict + optimizer_state_dict 一起存下来。")
print()


# ===========================================================================
# 第 9 部分：安全提示
# ===========================================================================
print("=" * 78)
print("第 9 部分：安全提示")
print("=" * 78)
print(
    """
1. 不要加载不可信的 .pt / .pth / .bin 文件。
   torch.save 用的是 Python 的 pickle 协议。pickle 在**反序列化时就会执行**
   文件里描述的对象构造过程 —— 换句话说，打开一个恶意 .pt 文件可能等同于运行一段代码
   （可以删文件、可以联网下载东西）。这不是 PyTorch 的 bug，而是 pickle 的固有特性。

2. 因此：
   · 从 Hugging Face / 网盘 / 别人手里拿到的权重，先确认来源可信；
   · PyTorch 2.6+ 把 torch.load 的 weights_only 默认改成 True，正是为了这个——
     它只允许反序列化张量和基础容器，遇到任意对象直接拒绝；
     只有在**确认文件可信**时才显式写 weights_only=False。

3. 生产环境优先选择：
   · torch.save(model.state_dict(), ...) —— 只存张量，配合 weights_only=True（默认）加载；
   · 或 safetensors 这类「纯权重、不含可执行代码」的格式（Hugging Face 生态的默认选择）。
   **本机没有安装 safetensors / onnx / tensorflow，所以本节只做文字讲解，不 import 它们。**
"""
)
print()


# ===========================================================================
# 第 10 部分：可视化
# ===========================================================================
print("=" * 78)
print("第 10 部分：可视化——训练 loss 曲线 + 推理耗时对比")
print("=" * 78)

fig8, axes8 = plt.subplots(1, 3, figsize=(17, 5))

# 子图 1：训练 loss 曲线（含续训的 2 个 epoch）
_ax1 = axes8[0]
_all_loss = _loss_curve + _resume_losses
_ax1.plot(range(1, _TRAIN_EPOCHS + 1), _loss_curve, "o-", color="#4C72B0", label="原始训练")
_ax1.plot(range(_TRAIN_EPOCHS + 1, _TRAIN_EPOCHS + _RESUME_EPOCHS + 1), _resume_losses,
          "s--", color="#55A868", label=f"断点续训（{_RESUME_EPOCHS} epoch）")
_ax1.set_xlabel("epoch")
_ax1.set_ylabel("训练交叉熵损失")
_ax1.set_title("训练 loss 曲线（含断点续训）", fontsize=11)
_ax1.legend(fontsize=9)
_ax1.grid(alpha=0.3)

# 子图 2：推理耗时对比柱状图
_ax2 = axes8[1]
_bars = _ax2.bar(["with\\nno_grad()", "不用\\nno_grad()"],
                 [_avg_ng * 1000, _avg_g * 1000], color=["#55A868", "#C44E52"])
_ax2.bar_label(_bars, fmt="%.3f ms")
_ax2.set_ylabel("1000 条样本推理总耗时（毫秒）")
_ax2.set_title(f"推理耗时对比（加速比 {_speedup:.2f}×）", fontsize=11)
_ax2.grid(alpha=0.3, axis="y")

# 子图 3：文件大小对比
_ax3 = axes8[2]
_bars3 = _ax3.bar(["state_dict\\n(只存参数)", "整个模型\\n(pickle)"],
                  [_state_bytes / 1024, _full_bytes / 1024], color=["#4C72B0", "#DD8452"])
_ax3.bar_label(_bars3, fmt="%.2f KB")
_ax3.set_ylabel("文件大小（KB）")
_ax3.set_title(f"两种保存方式的文件大小（倍数 {_full_bytes / _state_bytes:.2f}×）", fontsize=11)
_ax3.grid(alpha=0.3, axis="y")

fig8.suptitle("模型保存与推理：训练曲线 / no_grad 加速 / 文件大小", fontsize=13)
fig8.tight_layout(rect=(0, 0, 1, 0.93))
_p8 = OUTPUT_DIR / "03训练组件_08_模型保存与推理.png"
fig8.savefig(_p8, dpi=110)
plt.close(fig8)
print(f"已保存：{_p8}")
print()


# ===========================================================================
# 第 11 部分：权重存储格式对比表（课案）
# ===========================================================================
print("=" * 78)
print("第 11 部分：权重存储格式对比表（课案原表）")
print("=" * 78)
_FORMATS = [
    ("PyTorch (.pth/.pt/.bin)", "PyTorch 官方格式", "灵活、生态兼容", "模型开发与训练",
     "安全风险，勿加载不可信文件"),
    ("Safetensors (.safetensors)", "Hugging Face 安全格式", "安全、加载快", "模型分享与下载",
     "纯权重数据、无代码"),
    ("GGUF (.gguf)", "CPU 优化格式", "CPU 友好、单文件管理", "本地 CPU 部署",
     "需量化模型以优化性能"),
    ("量化格式 (.gguf / ...q4_0.bin)", "模型量化减体积", "减小体积和内存占用", "边缘设备部署",
     "常用 GGUF 实现"),
    ("分片格式 (...00001-of-00002.bin)", "模型分片存储", "解决大文件存储问题", "超大模型管理",
     "基于 .bin 或 .safetensors"),
    ("ONNX (.onnx)", "跨框架交换格式", "跨平台兼容", "跨框架部署", "转换可能需调试"),
    ("TensorFlow (.pb/.ckpt/.h5)", "TensorFlow 原生格式", "功能全面、生态集成",
     "TensorFlow 生态开发", "与 PyTorch 不兼容，需转换"),
]
print(f"{'格式':<34}{'定位':<24}{'优点':<22}{'典型场景':<18}{'注意'}")
print("-" * 128)
for _f in _FORMATS:
    print(f"{_f[0]:<34}{_f[1]:<24}{_f[2]:<22}{_f[3]:<18}{_f[4]}")
print()
print("  说明：本机只安装了 PyTorch、numpy、scikit-learn、matplotlib、seaborn、pandas、scipy，")
print("        没有 safetensors / onnx / tensorflow —— 所以上表只是**文字讲解**，脚本里不 import 它们。")
print()

print("=" * 78)
print("08 模型保存与推理：全部实验完成")
print("=" * 78)
print()
print("本次运行生成的文件：")
for _f in sorted(OUTPUT_DIR.glob("08_*")):
    print(f"  {_f.name:<40} {_f.stat().st_size:>10,} 字节")
for _f in sorted(OUTPUT_DIR.glob("03训练组件_08_*")):
    print(f"  {_f.name:<40} {_f.stat().st_size:>10,} 字节")

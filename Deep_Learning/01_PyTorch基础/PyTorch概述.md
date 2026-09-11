# PyTorch 概述

> 对应课案章节：**PyTorch / 概述**（课案行 154–436）
> 配套代码：本目录下 `01_张量.py`、`02_自动微分.py`、`03_数据集与DataLoader.py`、`04_权重存储与精度类型.py`

---

## 1. PyTorch 是什么

PyTorch 是当前最常用的深度学习框架之一。它的设计目标可以概括成一句话：

> **用接近普通 Python 的写法，把一个「能自动求导的多维数组计算器」用起来。**

它不是一个「把所有东西都封装好的黑箱」，而是把深度学习训练拆成几个职责清晰的部件，让你能逐层看清训练到底在做什么。这一点对「从零理解深度学习」尤其重要——你随时可以 `print` 出张量形状、`print` 出梯度，甚至自己手写反向传播和框架的结果对拍。

### 1.1 核心特点（课案表格）

| 特点 | 说明 |
| --- | --- |
| Tensor | 支持 CPU/GPU 的多维数组 |
| Autograd | 自动构建计算图并计算梯度 |
| nn.Module | 用来定义神经网络结构 |
| DataLoader | 用来批量读取数据 |
| Optimizer | 根据梯度更新模型参数 |

这五个概念基本覆盖了一个训练脚本的全部：**数据 → 模型 → 损失 → 梯度 → 更新**。

---

## 2. 五个核心抽象：各自负责什么

理解 PyTorch，关键是分清「谁负责什么、不负责什么」。下面逐个说明。

### 2.1 Tensor（张量）——数据的统一载体

张量就是**可以放到 GPU 上计算的多维数组**。文本、图片、音频、表格，进入模型前统统变成张量。

深度学习里常见的形状约定：

| 数据类型 | 张量形状示例 | 说明 |
| --- | --- | --- |
| 一条特征向量 | $(d,)$ | 一条样本的多个特征 |
| 一批表格数据 | $(B, d)$ | 多条样本组成一个 batch |
| 一张灰度图 | $(1, H, W)$ | 1 个通道 |
| 一批彩色图 | $(B, 3, H, W)$ | RGB 三通道（PyTorch 默认 **NCHW**，通道在前） |
| 一批文本 token | $(B, L)$ | 每个位置是一个 token id |

**职责**：存数据、做数值运算（元素级运算、矩阵乘法、广播、形状变换、索引）。
**不负责**：不负责求导（那是 Autograd）、不负责定义网络（那是 nn.Module）。

实战中最核心的两条形状规则：

$$
\text{矩阵乘法}: (m, n) \times (n, p) \rightarrow (m, p)
\qquad
\text{广播}: (3,1) + (1,4) \rightarrow (3,4)
$$

自己动手验证这些形状规则，就是在 `01_张量.py` 里做的事。

### 2.2 Autograd（自动微分）——训练引擎

只要一个张量被标记为 `requires_grad=True`，PyTorch 就会**在执行运算的同时**记录一张计算图；调用 `backward()` 时，它沿图反向走一遍，用链式法则把每一步的局部导数乘起来：

$$
\frac{\partial L}{\partial x}
= \frac{\partial L}{\partial y}\cdot\frac{\partial y}{\partial x}
\qquad\text{（链式法则，逐层反向相乘）}
$$

课案里的最小例子：

$$
y = x^2 + 3x + 1 \quad\Longrightarrow\quad \frac{\mathrm{d}y}{\mathrm{d}x} = 2x + 3
$$

代入 $x = [1, 2, 3]$，梯度就是 $[5, 7, 9]$。

**职责**：记录计算图、计算梯度、把梯度累加进 `leaf.grad`。
**关键陷阱**：`.grad` 是**累加**的（`+=`），不是覆盖。所以训练循环必须显式清零：

```python
optimizer.zero_grad()   # 清空上一轮梯度
loss.backward()         # 反向传播：计算当前梯度
optimizer.step()        # 根据梯度更新参数
```

这三行是**每一次参数更新的固定动作**，缺一不可。

### 2.3 nn.Module（网络结构）——把参数和计算打包

`nn.Module` 是「层」和「模型」的统一基类。它的职责是：

1. **注册参数**：`nn.Parameter` 会被自动登记，从而能被 `model.parameters()` 找到并交给优化器；
2. **注册子模块**：`self.fc1 = nn.Linear(...)` 之后，参数路径自动变成 `fc1.weight`；
3. **注册 buffer**：不需要梯度但要随模型存取的状态（如 BatchNorm 的 `running_mean`）；
4. **定义前向**：在 `forward()` 里写清楚数据怎么流。

```python
class TinyNet(nn.Module):
    def __init__(self, n_in=6, n_hidden=8, n_out=3):
        super().__init__()                     # 必须调用，否则参数注册机制不生效
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_out)

    def forward(self, x):                      # 只定义前向；反向由 autograd 自动推导
        return self.fc2(self.act(self.fc1(x)))
```

线性层的数学形式（$W$ 的形状是 $(d_{\text{out}}, d_{\text{in}})$）：

$$
\mathbf{y} = \mathbf{x}W^{\top} + \mathbf{b}
$$

**职责**：定义「有哪些参数」和「怎么算前向」。
**不负责**：不负责更新参数（那是 Optimizer）、不负责读数据（那是 DataLoader）。
`model.train()` / `model.eval()` 切换的是 Dropout、BatchNorm 的**行为**，与梯度无关。

### 2.4 DataLoader（数据管线）——把样本喂成 batch

职责分工非常清楚：

- `Dataset`：只回答「第 $i$ 条样本长什么样」，约定实现 `__len__` 与 `__getitem__`；
- `DataLoader`：只回答「给我一批一批的数据」，负责 `batch_size`、`shuffle`、`num_workers`、`collate_fn`、`pin_memory`。

batch 个数的通用公式：

$$
\text{len(dataloader)} =
\begin{cases}
\left\lceil \dfrac{n}{B} \right\rceil, & \text{drop\_last=False}\\[6pt]
\left\lfloor \dfrac{n}{B} \right\rfloor, & \text{drop\_last=True}
\end{cases}
$$

例如 $n=100$、$B=16$：`drop_last=False` 得到 $6$ 个满 batch（$16 \times 6 = 96$）加 $1$ 个只有 $4$ 条的残 batch，共 $7$ 个 batch；`drop_last=True` 则丢掉那 $4$ 条，只剩 $6$ 个 batch。

**职责**：采样、分批、并行读取、把多条样本拼成矩形张量。
**不负责**：不负责定义模型、不负责求梯度。

### 2.5 Optimizer（参数更新）——优化器

优化器持有参数引用，按梯度更新参数。以最基础的 SGD 为例：

$$
\theta_{t+1} = \theta_t - \eta \, g_t
\qquad
g_t = \frac{\partial L}{\partial \theta_t}
$$

Adam 则在此之上维护梯度的一阶矩 $m_t$ 与二阶矩 $v_t$ 的滑动平均，并做偏差修正：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t,\qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^{\,t}},\qquad
\hat{v}_t = \frac{v_t}{1-\beta_2^{\,t}},\qquad
\theta_{t+1} = \theta_t - \eta\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

**职责**：`zero_grad()` 清零、`step()` 按梯度更新参数、维护自身状态（动量等）。
**不负责**：不计算梯度（梯度由 `loss.backward()` 写进 `.grad`）。
注意：优化器状态（Adam 的一阶/二阶矩）也属于「必须存盘」的内容，否则断点续训会退化。

---

## 3. PyTorch 标准工作流程

深度学习项目通常可以按照固定流程组织代码：

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 1. 准备数据
X = torch.randn(100, 10)
y = 3 * X + 2                      # 这里用线性关系造一个可学的目标

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 2. 定义模型
model = nn.Sequential(
    nn.Linear(10, 32),
    # nn.ReLU()
)

# 3. 定义损失函数和优化器
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. 训练模型
for epoch in range(5):
    for X_batch, y_batch in dataloader:
        outputs = model(X_batch)
        loss = loss_func(outputs, y_batch)

        optimizer.zero_grad()      # 清空上一轮梯度
        loss.backward()            # 反向传播，计算当前梯度
        optimizer.step()           # 根据梯度更新参数

    print("epoch:", epoch, "loss:", loss.item())
```

> **说明（本仓库的执行约定）**：上面是课案原文的流程骨架。实际跑训练时要注意两点：
> 其一，`nn.CrossEntropyLoss()` 期望的 `y` 是 `long` 类型的类别索引（形状 $(B,)$），
> 而这里 `y = 3*X+2` 是回归目标，配 `CrossEntropyLoss` 在真实训练中会报形状/类型错误——
> 回归任务应换成 `nn.MSELoss()`；分类任务则应把标签改成类别 id。
> 其二，训练前要 `model.train()`、评估时用 `model.eval()` + `with torch.no_grad():`。
> 本目录的 `03_数据集与DataLoader.py` 给出了一个形状、类型都对得上的完整可跑版本。

流程的五个阶段和各部件的对应关系：

$$
\underbrace{\text{数据}}_{\text{Dataset/DataLoader}}
\rightarrow
\underbrace{\text{模型}}_{\texttt{nn.Module}}
\rightarrow
\underbrace{\text{损失}}_{\texttt{nn.*Loss}}
\rightarrow
\underbrace{\text{梯度}}_{\text{Autograd}}
\rightarrow
\underbrace{\text{更新}}_{\text{Optimizer}}
$$

---

## 4. 与静态图框架的对比

课案提到的「动态图 / 静态图」是理解 PyTorch 设计哲学的关键。

| 对比维度 | PyTorch（动态图，define-by-run） | 静态图框架（如 TF1 的 Graph、部分编译器栈） |
| --- | --- | --- |
| 图的构建时机 | 运算执行时**同时**建图，图随代码走 | 先声明完整计算图，再灌数据执行（define-and-run） |
| 调试体验 | 和普通 Python 一样：可断点、可 `print`、可 `pdb` | 需要专门的会话/调试工具，出错栈难读 |
| 条件与循环 | 直接用 `if` / `for`，分支只为本次执行建图 | 需要专门的 `tf.cond` / `tf.while_loop` |
| 变长输入 | 天然支持（每批形状可以不同） | 通常要靠 padding/占位符提前固定形状 |
| 性能优化空间 | 逐算子执行，单步开销略大（可用 `torch.compile` 补） | 整图优化、算子融合、部署时更极致 |
| 部署友好度 | 需导出 ONNX / TorchScript 等中间格式 | 图本身就是可序列化的产物 |
| 学习曲线 | 低，接近 NumPy | 高，需要理解「图 / 会话 / 占位符」 |

**结论**：PyTorch 用「牺牲一点极致静态优化」换来了「极低的调试门槛」，这也是它在研究与教学场景成为主流的原因。近年的 `torch.compile`、`torch.export` 等机制，则是在保留动态图易用性的同时，把静态图那部分性能收益补回来。

---

## 5. CPU 与 GPU 的差异

张量的 `.device` 决定它在哪块硬件上算。跨设备的铁律是：**参与同一次运算的张量必须在同一设备上**。

| 维度 | CPU | GPU（CUDA） |
| --- | --- | --- |
| 典型内存 | 几十 GB，便宜 | 8–80 GB 显存，昂贵 |
| 并行度 | 几十个核 | 上千个 CUDA Core |
| 适合 | 小模型、数据预处理、调试 | 大规模矩阵运算（训练/推理） |
| 代码写法 | `.to("cpu")` | `.to("cuda")` |
| 设备无关写法 | `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` | 同上 |

标准写法是**先算设备、再统一搬运**：

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)                 # 模型搬到设备
for X_batch, y_batch in dataloader:
    X_batch = X_batch.to(device)         # 数据搬到同一设备
    y_batch = y_batch.to(device)
    outputs = model(X_batch)
```

**本机环境的实测结论**：安装的是 `torch 2.14.0+cpu`（CPU 专用构建），

- `torch.cuda.is_available()` 恒为 `False`；
- `torch.cuda.device_count()` 恒为 `0`；
- 此时调用 `.to("cuda")` 会直接抛异常（实测为 `AssertionError: Torch not compiled with CUDA enabled`），
  所以在 `01_张量.py` 里我们用 `try/except` 捕获后以中文说明，并用 `.to("cpu")` 演示等价代码路径。

其他和 GPU 相关的参数在本机上的取舍：

- `pin_memory`：锁页内存可以加速 CPU→GPU 的拷贝，**没有 GPU 时设 `True` 只会多一次拷贝开销**，故取 `False`；
- `num_workers`：与 GPU 无关，它并行的是「数据读取」。Windows 上 `num_workers > 0` 会以 **spawn** 方式启动子进程，
  子进程会重新导入主模块，因此必须配合 `if __name__ == "__main__":` 保护（详见 `03_数据集与DataLoader.py`）。

---

## 6. 本目录脚本导览

| 脚本 | 对应章节 | 主要内容 | 产出的图 |
| --- | --- | --- | --- |
| `01_张量.py` | 张量 | 创建（列表 / NumPy / 工厂函数）、默认 dtype 与转换、元素级运算 vs 矩阵乘法 `@`、**广播**（`(3,1)+(1,4)→(3,4)`）、`reshape` vs `view`（非连续 → `.contiguous().view()`）、`unsqueeze/squeeze`、`transpose/permute`、切片 / 布尔掩码 / `index_select` / `gather`、`cat` vs `stack`、设备与 CPU/GPU | `01_张量_形状变换与广播.png` |
| `02_自动微分.py` | 自动微分 | `requires_grad`、`y=x^2+3x+1` 对应解析梯度 $2x+3$、`grad_fn` / `next_functions` 计算图遍历、`grad` 累加与 `zero_grad()` 必要性、`retain_graph`、`detach()` vs `no_grad()`、`torch.autograd.grad`、`create_graph=True` 高阶导数、**numpy 手写两层网络反向传播与 autograd 对拍（最大误差 < 1e-6）** | `02_自动微分_导数曲线.png` |
| `03_数据集与DataLoader.py` | 数据 | 自定义 `Dataset`（`__init__`/`__len__`/`__getitem__`）、batch 迭代、`drop_last` 对残 batch 的影响（100/16 = 6 满 + 1 残）、`shuffle` 顺序对比、Windows 上 `num_workers` 的 spawn 陷阱、变长序列自定义 `collate_fn`（padding + mask）、`TensorDataset`、`len(dataset)` 与 `len(dataloader)` 的 $\lceil n/B \rceil$ 关系、`random_split` / `Subset` 切分、**手写 `MyDataLoader` 与官方对比**、多进程与真实训练循环 | `03_数据集与DataLoader_batch与padding.png` |
| `04_权重存储与精度类型.py` | 权重存储 / 精度类型 | `state_dict`（OrderedDict、只有参数与 buffer、不含结构）、`load_state_dict(strict=True/False)` 与 `missing_keys`/`unexpected_keys`、真实存盘 + `torch.equal` 校验、`torch.save(model)` vs `state_dict`、完整 checkpoint（model+optimizer+scheduler+epoch+config）、格式对比表、**FP32/FP16/BF16/INT8 的位数结构与内存/误差实测**、FP16 下溢与溢出、BF16 精度损失、INT8 仿射量化、`torch.autocast` 混合精度 | `04_精度类型_内存与范围对比.png` |

配套工具（非课案知识点）：

| 文件 | 作用 |
| --- | --- |
| `../verify_all.py` | 一键运行 `Deep_Learning` 下所有脚本并汇总返回码 / 耗时 / 输出尾部 |
| `../output/` | 所有图片与权重文件的统一输出目录 |

> 备注：本目录的脚本全部使用标准 `import matplotlib.pyplot as plt`，**不需要任何环境兼容补丁**。
> （历史上本机 venv 的 `kiwisolver` 曾缺少 `__init__.py` 导致 matplotlib 无法导入，该问题已通过
> 补回官方 wheel 里的 `__init__.py` 根治，原先临时的内存兼容模块已删除。）

所有脚本的运行方式（PowerShell，注意中文路径用 `&` 加引号）：

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Deep_Learning\01_PyTorch基础\01_张量.py'
```

---

## 7. 学习路径建议

1. **先建立形状直觉**（`01_张量.py`）：绝大多数深度学习报错最终都归结为「形状不对」。把 `(d,)`、`(B,d)`、`(1,H,W)`、`(B,3,H,W)`、`(B,L)` 五组形状和广播规则记牢。
2. **再理解梯度从哪来**（`02_自动微分.py`）：亲手把两层网络的正反向推一遍，再和 autograd 对拍，就能彻底破除「反向传播是黑魔法」的感觉。
3. **然后打通数据管线**（`03_数据集与DataLoader.py`）：`Dataset` 管单样本、`DataLoader` 管 batch，这个分工一旦看透，训练代码就只剩五行的循环体。
4. **最后关心工程问题**（`04_权重存储与精度类型.py`）：权重怎么存、精度怎么选，决定了模型能不能复现、能不能部署、要花多少显存。

---

## 8. 一句话总结

> PyTorch 把深度学习训练拆成 **Tensor（数据）/ Autograd（梯度）/ nn.Module（结构）/ DataLoader（喂数据）/ Optimizer（更新）** 五件事，
> 每件事都有一个明确的负责人；训练循环就是把这五件事按固定节奏串起来的那五行代码。

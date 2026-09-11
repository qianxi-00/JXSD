# 深度学习课案 · 代码实现（Deep_Learning）

本目录把「深度学习」课案的**全部知识点**实现为可运行、带详细中文讲解注释的代码与文档。
每个 `.py` 脚本都遵循同一个写作范式：**先讲原理（公式 / 直觉 / 为什么这样设计），再写代码，最后用
`print` 打印形状与关键数值**，并在需要时把实验结论画成图保存到 `output/`。

- 全部代码使用 **PyTorch 2.14.0+cpu（纯 CPU 版）**，不依赖任何 GPU。
- 全部数据由 `torch.randn` / `torch.randint` 或 `scikit-learn` 现场生成，**不联网、不下载任何数据集或预训练权重**。
- 每个脚本单次运行时间都控制在 **40 秒以内**（绝大多数在 20 秒以内）。
- 全目录中文：docstring、注释、`print` 输出、Markdown 文档。

---

## 一、目录结构

```
Deep_Learning/
├── README.md                              本文件：目录树 + 知识点索引 + 运行说明
├── verify_all.py                          一键运行本目录全部脚本并汇总返回码与耗时
├── VERIFY_REPORT.md                       verify_all.py 的完整实测输出
│
├── 01_PyTorch基础/
│   ├── PyTorch概述.md                     课案「PyTorch → 概述」章节文档
│   ├── 01_张量.py                         创建 / 运算 / 广播 / 形状变换 / 索引 / 拼接 / 设备
│   ├── 02_自动微分.py                     requires_grad / backward / grad / zero_grad / detach / 计算图
│   ├── 03_数据集与DataLoader.py           自定义 Dataset + DataLoader + batch 迭代 + collate_fn
│   └── 04_权重存储与精度类型.py           state_dict / load_state_dict / FP32·FP16·BF16 / dtype 与内存
│
├── 02_网络架构/
│   ├── 01_Embedding词嵌入.py              nn.Embedding 输入输出、padding_idx、词向量相似度
│   ├── 02_DNN前向传播与随机失活.py        手写全连接前向、Dropout 训练/推理差异、完整二分类训练
│   ├── 03_CNN卷积与池化.py                手写卷积滑动窗口 + nn.Conv2d/MaxPool2d/自适应池化 + 形状推导
│   ├── 04_RNN_LSTM_GRU.py                 nn.RNN/LSTM/GRU 前向、手写 RNN/LSTM/GRU 单元、梯度消失演示
│   ├── 05_Attention与自注意力.py          缩放点积注意力手写 + 掩码 + 多头拆分形状
│   └── 06_Transformer完整实现.py          位置编码、多头注意力、FFN、编码器块、解码器块、
│                                          因果掩码、nn.Transformer 框架用法、三种注意力对比
│
├── 03_训练组件/
│   ├── 01_激活函数.py                     Sigmoid / Softmax / Tanh / ReLU / 变体，公式+导数+numpy/torch 对比
│   ├── 02_损失函数.py                     BCE / CE / MSE / MAE / SmoothL1 / NLL / Focal，含手写实现
│   ├── 03_优化器全家桶.py                 SGD / Momentum / NAG / AdaGrad / RMSProp / AdaDelta / Adam
│   │                                      手写一步更新 + torch.optim 实测 + Rosenbrock 收敛对比
│   ├── 04_评估指标.py                     Accuracy / Precision / Recall / F1 / AUC / MSE / MAE
│   │                                      手写公式 + sklearn 对照
│   ├── 05_参数初始化.py                   zeros_/ones_/constant_/normal_/uniform_/trunc_normal_/eye_/
│   │                                      xavier_/kaiming_/orthogonal_，含逐层信号衰减实验
│   ├── 06_正则化.py                       Dropout / L1·L2 / weight_decay / BatchNorm·LayerNorm /
│   │                                     早停 / 纯 torch 数据增强（不使用 torchvision）
│   ├── 07_学习率调度.py                   StepLR / MultiStepLR / ExponentialLR / CosineAnnealingLR /
│   │                                      ReduceLROnPlateau / OneCycleLR 的 lr 曲线与打印
│   └── 08_模型保存与推理.py               state_dict 与整模型两种保存方式、eval() + no_grad()、
│                                          加载一致性验证、checkpoint 断点续训、精度格式对比
│
├── 04_训练技巧/
│   ├── 01_过拟合与欠拟合演示.py           小模型 / 大模型 / 加正则三组对比曲线、决策边界、
│   │                                      损失震荡、训练缓慢
│   ├── 02_梯度爆炸与梯度消失演示.py       逐层梯度范数、梯度爆炸与裁剪、残差连接、五种解决方案
│   └── 训练技巧总结.md                    过拟合/欠拟合/损失震荡/训练缓慢/梯度问题的现象·诊断·解决
│
└── output/                                运行时自动生成：所有图片（.png）与权重文件（.pt）
```

---

## 二、知识点索引（课案章节 → 文件）

| 课案大纲章节 | 对应文件 | 关键内容 |
|---|---|---|
| **概述** · 概念 | 本 README + `01_PyTorch基础/PyTorch概述.md` | 深度学习与机器学习的区别对比表、特征工程 vs 自动学习特征 |
| **概述** · 推导 | `01_PyTorch基础/02_自动微分.py`、`03_训练组件/02_损失函数.py` | 前向传播 → 损失计算 → 反向传播（链式法则）→ 参数更新；信息量 / 信息熵 / KL 散度 / 交叉熵；**交叉熵 + Softmax 的梯度 = ŷ − y** 的完整推导与数值验证 |
| **PyTorch** · 概述 | `01_PyTorch基础/PyTorch概述.md` | Tensor / Autograd / nn.Module / DataLoader / Optimizer 五大抽象、标准工作流程 |
| **PyTorch** · 张量 | `01_PyTorch基础/01_张量.py` | 创建、元素级运算 vs 矩阵乘法、广播、reshape/view/contiguous、unsqueeze/squeeze、索引与布尔掩码、cat vs stack、permute、设备 |
| **PyTorch** · 自动微分 | `01_PyTorch基础/02_自动微分.py` | requires_grad、grad_fn、backward、grad 累加与 zero_grad、retain_graph、detach vs no_grad、高阶导数；**numpy 手算梯度与 autograd 对比** |
| **PyTorch** · 数据 | `01_PyTorch基础/03_数据集与DataLoader.py` | `__len__`/`__getitem__`、batch_size / shuffle / num_workers / drop_last、TensorDataset、collate_fn 变长 padding、手写 MyDataLoader |
| **PyTorch** · 权重存储 | `01_PyTorch基础/04_权重存储与精度类型.py`、`03_训练组件/08_模型保存与推理.py` | `.pth/.pt/.bin`、Safetensors、GGUF、量化、分片、ONNX、TF 格式对比（文字讲解）；state_dict vs 整模型保存 |
| **PyTorch** · 精度类型 | `01_PyTorch基础/04_权重存储与精度类型.py` | FP32 / FP16 / BF16 / INT8 的位宽·动态范围·内存占用、float16 下溢与溢出实测、混合精度说明 |
| **网络架构** · Embedding | `02_网络架构/01_Embedding词嵌入.py` | 查表本质、one-hot 的两个问题、`F.one_hot @ W` 等价性验证、padding_idx、词向量余弦相似度与 PCA 可视化 |
| **网络架构** · DNN 前向传播 | `02_网络架构/02_DNN前向传播与随机失活.py` | 逐层公式 `z⁽ˡ⁾=W⁽ˡ⁾h⁽ˡ⁻¹⁾+b⁽ˡ⁾`、手写矩阵乘法与 nn.Sequential 对照、无激活函数时多层退化为线性 |
| **网络架构** · 随机失活 | `02_网络架构/02_DNN前向传播与随机失活.py`、`03_训练组件/06_正则化.py` | 训练 `h = r ⊙ h/(1−p)`、推理 `h` 不变；倒置 Dropout；训练/推理差异实测；有无 Dropout 对比 |
| **网络架构** · CNN 卷积 | `02_网络架构/03_CNN卷积与池化.py` | `Y(i,j)=ΣΣ K(m,n)·X(i+m,j+n)+b`、手写滑动窗口、输出尺寸公式、多通道、参数共享、感受野 |
| **网络架构** · CNN 池化 | `02_网络架构/03_CNN卷积与池化.py` | 最大池化 / 平均池化公式、手写池化、自适应池化 `AdaptiveAvgPool2d` |
| **网络架构** · RNN 基本公式 | `02_网络架构/04_RNN_LSTM_GRU.py` | `h_t=tanh(W_hh·h_{t-1}+W_xh·x_t+b_h)`、`y_t=W_hy·h_t+b_y`、手写 RNNCell 与 nn.RNN 对照、隐藏状态形状 |
| **网络架构** · RNN 梯度消失 | `02_网络架构/04_RNN_LSTM_GRU.py`、`04_训练技巧/02_梯度爆炸与梯度消失演示.py` | 单步 `∂h_t/∂h_{t-1}=diag(tanh'(z_t))·W_hh`、跨 k 步连乘、\|λ_max\| 的作用、**逐时间步梯度范数实测与绘图** |
| **网络架构** · LSTM | `02_网络架构/04_RNN_LSTM_GRU.py` | 三门 + 细胞状态公式、`[h_{t-1},x_t]` 拼接维度说明、`∂C_t/∂C_{t-1} = f_t + …` 的**免衰减旁路**、手写 LSTMCell 与 nn.LSTM 权重切分对照 |
| **网络架构** · GRU | `02_网络架构/04_RNN_LSTM_GRU.py` | 更新门/重置门公式、手写 GRUCell、与 LSTM 的参数与速度对比 |
| **网络架构** · Attention 通用形式 | `02_网络架构/05_Attention与自注意力.py` | Q/K/V 三组向量、`softmax(QKᵀ/√d_k)·V` 三步、Cross vs Self 的区别 |
| **网络架构** · 自注意力 | `02_网络架构/05_Attention与自注意力.py` | `Q=XW_Q, K=XW_K, V=XW_V`、√d_k 缩放的方差推导与实测、掩码、多头拆分形状、contiguous 的必要性 |
| **网络架构** · Transformer 核心组件 | `02_网络架构/06_Transformer完整实现.py` | Input Embedding、Positional Encoding、Multi-Head Self-Attention、Add & Norm、FFN、Masked MHA、Cross MHA、Linear+Softmax |
| **网络架构** · 位置编码 | `02_网络架构/06_Transformer完整实现.py` | sin/cos 公式、`ω_i=1/10000^(2i/d_model)`、**内积只取决于相对距离**的推导与数值验证、频率分辨率、register_buffer |
| **网络架构** · 多头注意力 | `02_网络架构/06_Transformer完整实现.py`、`05_Attention与自注意力.py` | W_Q/W_K/W_V/W_O、拆头/并头四步形状变换、每头关注不同模式 |
| **网络架构** · 编码器 / 解码器 | `02_网络架构/06_Transformer完整实现.py` | EncoderBlock（双向自注意力 + FFN + 残差 + LayerNorm）、DecoderBlock（掩码自注意力 + 交叉注意力 + FFN）、Pre-LN vs Post-LN、因果掩码生效性实证 |
| **网络架构** · 框架使用 | `02_网络架构/06_Transformer完整实现.py` | `nn.Transformer` / `nn.TransformerEncoder` / `nn.TransformerDecoder` / `nn.TransformerEncoderLayer` 形状验证；**`attn_mask` 语义坑**与 `is_causal` 只是「提示」的实测证明 |
| **网络架构** · 三种注意力 | `02_网络架构/06_Transformer完整实现.py` | Self / Masked Self / Cross 的 Q·K·V 来源、掩码、所在层，形状与权重热力图对比 |
| **训练组件** · 激活函数 | `03_训练组件/01_激活函数.py` | Sigmoid / Softmax / Tanh / ReLU / LeakyReLU / ELU / GELU / SiLU / Mish / Softplus / PReLU / ReLU6 / Hardswish 的公式·导数·优缺点·场景，numpy 与 torch 双向对比，曲线与梯度消失图 |
| **训练组件** · 损失函数 | `03_训练组件/02_损失函数.py` | BCE / BCEWithLogits / CE / MSE / MAE / SmoothL1 / NLL / KLDiv / CosineEmbedding / Triplet / Focal，公式 + 手写 + torch 对照 + 梯度验证 |
| **训练组件** · 优化器 | `03_训练组件/03_优化器全家桶.py` | SGD / Momentum / NAG / AdaGrad / RMSProp / AdaDelta / Adam(+AdamW 等) 的更新公式**手写实现**与 `torch.optim` 对照，Rosenbrock 收敛曲线与轨迹图 |
| **训练组件** · 评估指标 | `03_训练组件/04_评估指标.py` | 混淆矩阵、Accuracy、Precision、Recall、F1、AUC（排序公式推导 + 三重实现验证）、MSE、MAE、RMSE、R²、MAPE；手写与 sklearn 逐项对比 |
| **训练组件** · 参数初始化 | `03_训练组件/05_参数初始化.py` | 常数 / 随机 / 截断正态 / 单位阵 / LeCun / Kaiming / Xavier / 正交，方差推导（`1/n_in`、`2/n_in`、`2/(n_in+n_out)`）、逐层信号衰减与梯度消失/爆炸实验 |
| **训练组件** · 正则化 | `03_训练组件/06_正则化.py` | Dropout、L1/L2（含贝叶斯 MAP 推导：高斯先验→L2、拉普拉斯先验→L1）、weight_decay 与 L2 的等价性证明、稀疏解三个角度、BatchNorm / LayerNorm / InstanceNorm / GroupNorm（**NLP 为什么用 LayerNorm**）、早停 EarlyStopping、纯 torch 数据增强 |
| **训练组件** · 学习率 | `03_训练组件/07_学习率调度.py` | StepLR / MultiStepLR / ExponentialLR / CosineAnnealingLR / ReduceLROnPlateau / OneCycleLR（+LambdaLR、CyclicLR、warmup 组合），每步 lr 打印 + 同图对比 |
| **训练组件** · 推理 | `03_训练组件/08_模型保存与推理.py` | `torch.save(model.state_dict())` 与 `torch.save(model)`、两种加载方式、`eval()` + `no_grad()`、加载后预测一致性验证、批量推理吞吐、checkpoint 断点续训 |
| **训练技巧** · 过拟合 | `04_训练技巧/01_过拟合与欠拟合演示.py` | 现象（训练 loss 降、验证 loss 反弹）、原因（容量 > 数据有效信息量）、处理（正则化 / 数据增强 / 减小模型 / 早停 / 加数据）与对应实验 |
| **训练技巧** · 欠拟合 | `04_训练技巧/01_过拟合与欠拟合演示.py` | 现象（训练与验证 loss 都高且不降）、原因（容量不足 / 训练不充分）、处理（增大模型 / 训练更久 / 降正则 / 调大学习率 / 查数据）与对应实验 |
| **训练技巧** · 损失震荡 | `04_训练技巧/01_过拟合与欠拟合演示.py` | 学习率过大 / batch 过小导致震荡，梯度裁剪；不同 lr 与 batch size 的曲线实测 |
| **训练技巧** · 训练缓慢 | `04_训练技巧/01_过拟合与欠拟合演示.py` | 学习率过小 / 模型过大；混合精度、增大 batch、降低数据精度（含 CPU 上 fp16 反而更慢的实测） |
| **训练技巧** · 梯度爆炸/消失 | `04_训练技巧/02_梯度爆炸与梯度消失演示.py` | 现象对照表、逐层梯度范数、五种解决方案（ReLU / Kaiming·Xavier / BatchNorm·LayerNorm / 残差连接 / 梯度裁剪）全部带实测证据 |
| **训练技巧** · 总结 | `04_训练技巧/训练技巧总结.md` | 现象·诊断·解决速查表、Bias-Variance 分解、诊断决策树 |

---

## 三、环境说明

### 3.1 使用的解释器

```powershell
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Deep_Learning\<子目录>\<脚本>.py'
```

- Python **3.12.12**，虚拟环境位于 `F:\ProGram\Python_Base\.venv`。
- **不要使用 `uv run`**（可能触发 `uv sync` 改动环境），也不要 `pip install` / `uv add` 任何东西。
- 路径含中文，PowerShell 中必须用引号并用 `&` 调用。

### 3.2 已安装的依赖（本项目只使用这些）

| 包 | 版本 | 用途 |
|---|---|---|
| torch | 2.14.0+cpu | 全部张量 / 自动微分 / 网络 / 优化器（**纯 CPU，`torch.cuda.is_available()` 为 False**） |
| numpy | 2.5.2 | 手写公式实现、数值验证 |
| pandas | 3.0.5 | 可用但本目录仅少量使用 |
| scipy | 1.18.0 | 可用 |
| matplotlib | 3.11.1 | 全部绘图（Agg 无界面后端） |
| seaborn | 0.13.2 | 美化绘图（ROC / 热力图等） |
| scikit-learn | 1.9.0 | 数据生成（`make_moons`/`make_blobs`/`make_classification`/`load_iris`）与指标对照 |
| tqdm / joblib | — | 进度条与并行工具 |

### 3.3 明确**不使用**的依赖

`torchvision`、`torchaudio`、`torchtext`、`tensorboard`、`plotly`、`transformers`、
`pytorch-lightning`、`torchmetrics` 均**未安装**，因此：

- 课案里用 `torchvision.transforms` 做的数据增强，改为**纯 torch 手写**（`random_horizontal_flip` /
  `random_crop` / `random_gaussian_noise` / `Compose` 等，见 `03_训练组件/06_正则化.py`）。
- 课案里用 `torchmetrics` 计算的评估指标，改为**手写公式 + `sklearn.metrics` 双向对照**
  （见 `03_训练组件/04_评估指标.py`）。
- 没有 GPU，`torch.cuda.amp` 混合精度只做文字讲解与形状验证。

### 3.4 关于 `matplotlib` 与曾经的 `kiwisolver` 问题（已修复，无需补丁）

> **现状：环境已修复，本目录不需要任何兼容补丁。**
> 之前本机虚拟环境的 `kiwisolver` 包缺少 `__init__.py`（`site-packages\kiwisolver\` 下只剩
> `_cext.cp312-win_amd64.pyd`），Python 把它当成「命名空间包」，`kiwisolver.__version__` 不存在，
> matplotlib 3.11 在启动的 `_check_versions()` 阶段就会失败，导致所有 `import matplotlib.pyplot`
> 的脚本都跑不起来。该问题已通过**从 kiwisolver 1.5.0 官方 wheel 补回缺失的 `__init__.py`** 根治。
> 现在实测：
>
> ```
> import kiwisolver         -> 1.5.0（Variable / Constraint / Solver / Strength 等都在）
> import matplotlib         -> 3.11.1
> import matplotlib.pyplot  -> OK
> import matplotlib._layoutgrid -> OK（tight_layout / constrained_layout 正常）
> ```
>
> 在此之前，本目录曾经用一个临时模块在**进程内存里**给 `kiwisolver` 补 `__version__` 来兜住这个问题；
> 既然环境已经根治，**那个临时模块已经删除**，20 个知识点脚本里的兼容导入、`sys.path` 注入
> 和相关注释也一并清除，现在它们就是标准的 `import matplotlib.pyplot as plt`。

本目录每个绘图脚本的开头都是这样一段：

```python
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
```

可以随时用下面一行确认环境是否健康（不需要本目录里的任何辅助模块）：

```powershell
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -c "import kiwisolver, matplotlib; print(kiwisolver.__version__, matplotlib.__version__); import matplotlib.pyplot as plt; print('pyplot OK')"
```

> 如果将来重装环境后 `kiwisolver` 再次出问题，症状会是 `AttributeError: module 'kiwisolver' has no
> attribute '__version__'`；那时的正确做法是**把 `kiwisolver/__init__.py` 补回去**（或重装该包），
> 而不是在业务脚本里写兼容代码。

### 3.5 绘图与输出约定

- 统一使用 `matplotlib.use("Agg")` 无界面后端，**脚本中绝不调用 `plt.show()`**，避免阻塞。
- 中文字体：`matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]`，
  同时 `axes.unicode_minus = False` 保证负号正常显示。
- 所有图片与权重文件统一保存到 `Deep_Learning/output/`，路径用
  `pathlib.Path(__file__).resolve().parent.parent / "output"` 计算并自动创建，
  **不依赖当前工作目录**。
- 随机种子统一：`torch.manual_seed(42)`、`np.random.seed(42)`，保证结果可复现。

---

## 四、运行方式

### 4.1 一键运行全部脚本（推荐）

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Deep_Learning\verify_all.py'
```

`verify_all.py` 会：

1. 递归收集本目录下所有 `.py`（跳过 `verify_all.py` 自己，免得无限递归）；
2. 用 `subprocess.run([sys.executable, script], capture_output=True, text=True, encoding="utf-8",
   timeout=300)` 逐个运行，并给子进程注入 `PYTHONIOENCODING=utf-8` / `PYTHONUTF8=1`
   （**Windows 中文环境的编码陷阱**：子进程 stdout 默认跟随 ANSI 代码页 GBK，父进程用 UTF-8 解码会失败）
   与 `OMP_NUM_THREADS=4` / `MKL_NUM_THREADS=4`（避免 CPU 版 PyTorch 线程过度并行）；
3. 打印每个脚本的返回码、耗时与**最后 15 行输出**；
4. 打印耗时排行，并以
   `共 N 个脚本，成功 N 个，失败 0 个` 收尾。

`verify_all.py` 的完整实测输出保存在 `VERIFY_REPORT.md`。

### 4.2 单独运行某个脚本

```powershell
# 建议先设置 UTF-8，否则中文输出在部分终端会显示为乱码
$env:PYTHONIOENCODING='utf-8'

& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Deep_Learning\01_PyTorch基础\01_张量.py'
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Deep_Learning\02_网络架构\06_Transformer完整实现.py'
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Deep_Learning\03_训练组件\03_优化器全家桶.py'
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Deep_Learning\04_训练技巧\02_梯度爆炸与梯度消失演示.py'
```

---

## 五、推荐学习路径

1. **打基础**：`01_PyTorch基础/01_张量.py` → `02_自动微分.py` → `03_数据集与DataLoader.py`。
   把 `03_自动微分.py` 里「numpy 手算梯度 vs autograd」的实验跑通，训练循环的三行代码就彻底明白了。
2. **理解网络**：`02_网络架构/02_DNN前向传播与随机失活.py` → `03_CNN卷积与池化.py`
   → `04_RNN_LSTM_GRU.py`（重点看梯度范数表）→ `05_Attention与自注意力.py`
   → `06_Transformer完整实现.py`（先看形状打印，再看「序列反转」任务的贪心解码结果）。
3. **掌握训练**：`03_训练组件/01_激活函数.py` → `02_损失函数.py` → `03_优化器全家桶.py`
   → `04_评估指标.py` → `05_参数初始化.py` → `06_正则化.py` → `07_学习率调度.py`
   → `08_模型保存与推理.py`。
4. **排错能力**：`04_训练技巧/01_过拟合与欠拟合演示.py` → `02_梯度爆炸与梯度消失演示.py`
   → 阅读 `训练技巧总结.md` 的诊断决策树。

---

## 六、交付自检结论

- 全部 **20 个**知识点脚本均使用 `.venv` 解释器**真实运行过**，返回码 **20/20 为 0**，
  **无任何 Traceback / Error 输出**，所有脚本的 stderr 都为空（即没有任何字体告警）。
- `verify_all.py` 的最终汇总为 **「共 20 个脚本，成功 20 个，失败 0 个」**，
  合计耗时 **130.12 秒**，最慢单脚本 **22.83 秒**（全部 < 40 秒），详见 `VERIFY_REPORT.md`。
- 本目录**不依赖任何环境兼容补丁**：20 个脚本全部使用标准 `import matplotlib.pyplot as plt`，
  运行时不需要 `sys.path` 注入，也不会导入本目录之外的任何自定义模块。
- 所有文件均创建在 `Deep_Learning/` 目录内，**未改动工作区其它任何文件**
  （`pyproject.toml` / `uv.lock` 在工作树里显示为已修改，但那是一次并行的后端开发任务加入
  `celery` / `flask` / `lightgbm` / `pytest` 等依赖造成的，与本目录无关）。

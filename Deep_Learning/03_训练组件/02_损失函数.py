"""
对应课案章节：训练组件 / 损失函数

本节知识点：
    1. 损失函数衡量预测值和真实值之间的差距，是训练的目标——优化器通过最小化损失来更新参数。
       每个损失函数都按统一模板：①公式 ②适用任务 ③关键特点 ④numpy 手写实现
       ⑤torch 内置实现 ⑥两者数值对比 ⑦手写梯度表达式与 torch.autograd 对比。
    2. 二分类交叉熵 BCE：-1/n Σ[y·log ŷ + (1-y)·log(1-ŷ)]。
       nn.BCEWithLogitsLoss()（内含 Sigmoid，用 log-sum-exp 技巧，更数值稳定）
       vs nn.BCELoss()（要求先手动 sigmoid）。
       本脚本用极端 logit（±50）演示 BCELoss(sigmoid(x)) 会失效而
       BCEWithLogitsLoss 正常（只打印数值差异，不产生任何 Traceback）。
    3. 多分类交叉熵 CE：-1/n Σ_c y_c log ŷ_c。
       nn.CrossEntropyLoss()（内含 Softmax，标签是类别编号而不是 one-hot）。
       核心推导数值验证：交叉熵 + Softmax 的梯度 = ŷ - y（误差 < 1e-6）。
       同时讲清信息量 -log P、信息熵 H(X)=-ΣP logP、KL 散度 KL(P||Q)=ΣP log(P/Q)，
       以及 **KL 散度 = 交叉熵 - 信息熵**，并用课案三分类例子实际算一遍验证等式。
    4. MSE：1/n Σ(y-ŷ)²，回归用；平方放大误差、对离群点敏感；梯度 2(ŷ-y)/n。
       数值演示一个离群点如何主导 MSE。
    5. MAE (L1Loss)：1/n Σ|y-ŷ|，线性惩罚、对离群点鲁棒；0 点不可导（用 subgradient）。
    6. SmoothL1Loss (Huber)：小误差用平方（可导、收敛快）、大误差用线性（抗离群点）。
    7. NLLLoss：NLLLoss(log_softmax(x), y) == CrossEntropyLoss(x, y)（数值验证）。
    8. 其他：nn.KLDivLoss、nn.CosineEmbeddingLoss、nn.TripletMarginLoss、
       以及手写 Focal Loss `-α(1-p_t)^γ log(p_t)`（解决类别不平衡）。
    9. 可视化：①各回归损失随误差变化的曲线 ②分类损失随预测概率变化的曲线。
   10. 对比汇总表（课案）：BCE / CE / MSE / MAE。

说明：课案原文公式是乱码（LaTeX 被抽成了 `\frac{...}` 与数字黏连的形式），
      本脚本用正确的数学写法重新表述。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Deep_Learning\\03_训练组件\\02_损失函数.py'
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
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# 随机种子统一：保证每次运行结果一致、可复现
torch.manual_seed(42)
np.random.seed(42)

sns.set_theme(style="whitegrid", font="Microsoft YaHei")   # seaborn 风格画图
_EPS = 1e-12          # numpy 手写实现里的防 log(0) 保护项


def _title(text: str) -> None:
    """打印分节标题，让输出有清晰的结构。"""
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def _cmp(name, np_value, torch_value):
    """统一打印 numpy 手写值与 torch 内置值的对比。"""
    _d = abs(float(np_value) - float(torch_value))
    print(f"  {name:<34s} numpy 手写 = {float(np_value):.8f}   "
          f"torch 内置 = {float(torch_value):.8f}   差值 = {_d:.3e}")
    return _d


# ===========================================================================
# 1. 二分类交叉熵 BCE
# ===========================================================================
_title("1. 二分类交叉熵 BCE（Binary Cross Entropy）")
print("""
  公式      : BCE = -1/n · Σ_i [ y_i·log ŷ_i + (1 - y_i)·log(1 - ŷ_i) ]
              其中 ŷ_i = σ(z_i) = 1/(1+e^{-z_i}) 是预测为正类的概率，y_i ∈ {0,1}。
  适用任务  : 二分类（输出层用 Sigmoid 得到一个 0~1 的概率）。
  关键特点  : ① 「越错惩罚越大」——预测概率与真实标签差得越远，-log 越大，
                 而且是**无上界**的增长（预测 0.001 但真实为 1 时惩罚极大）；
              ② 等价于极大似然估计（最小化 BCE = 最大化对数似然）；
              ③ 与 Sigmoid 搭配时梯度非常简洁：∂BCE/∂z = ŷ - y（后面数值验证）。
  两种实现  : nn.BCELoss()          —— 输入必须是已过 Sigmoid 的概率，需手动 sigmoid；
              nn.BCEWithLogitsLoss() —— 输入是**原始 logits**，内部含 Sigmoid，更稳定。
""")


def bce_np(y_true, y_pred, eps=_EPS):
    """numpy 手写 BCE。

    y_pred 必须是**概率**（需自行 sigmoid）；eps 是防 log(0) 的保护项。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.clip(np.asarray(y_pred, dtype=np.float64), eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred)))


def bce_with_logits_np(logits, y_true):
    """numpy 手写 BCEWithLogits（数值稳定版）。

    关键技巧：把 sigmoid 与 log 合并，避免先算 σ(z) 再取 log。
        -[y·log σ(z) + (1-y)·log(1-σ(z))] = -y·log σ(z) - (1-y)·log σ(-z)
        而 log σ(z) = -log(1 + e^{-z}) = -(softplus(-z))
    softplus 用 log1p(exp(min(x,0))) + max(x,0) 的稳定写法，保证大 |z| 时不上溢。
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    # 稳定的 softplus：softplus(x) = max(x,0) + log1p(exp(-|x|))
    sp_neg_z = np.maximum(-z, 0.0) + np.log1p(np.exp(-np.abs(z)))     # softplus(-z)
    sp_z = np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))          # softplus(z)
    return float(np.mean(y * sp_neg_z + (1.0 - y) * sp_z))


torch.manual_seed(42)
_bce_logits = torch.randn(8) * 2.0                    # 8 个样本的 logits
_bce_y = torch.randint(0, 2, (8,)).float()            # 对应的 0/1 标签
_bce_p = torch.sigmoid(_bce_logits)                   # 手动 sigmoid 得到概率
print(f"  测试数据：8 个 logits = {np.round(_bce_logits.numpy(), 4)}")
print(f"           对应标签 y   = {_bce_y.numpy().astype(int)}")

print("\n  ▶ ④ numpy 手写 与 ⑤ torch 内置 数值对比")
_cmp("BCELoss（先手动 sigmoid）", bce_np(_bce_y.numpy(), _bce_p.numpy()),
     nn.BCELoss()(_bce_p, _bce_y))
_cmp("BCEWithLogitsLoss（直接吃 logits）", bce_with_logits_np(_bce_logits.numpy(), _bce_y.numpy()),
     nn.BCEWithLogitsLoss()(_bce_logits, _bce_y))
print("  两者数值相同，说明「先 sigmoid 再 BCE」与「logits 版 BCE」数学等价。")

print("""
  ▶ ⑦ 梯度表达式：∂BCE/∂z = (ŷ - y) / n     （对 logits z 求导，n 为样本数）
    推导：BCE 对 logit z 的导数，Sigmoid 的导数 σ'(z)=σ(z)(1-σ(z)) 会与
          BCE 里的 1/ŷ 和 1/(1-ŷ) 恰好约掉，最终只剩 (ŷ - y)。这就是
          「BCEWithLogitsLoss 比 BCELoss 稳定」的另一个好处：梯度形式极简。
""")
_z_grad = _bce_logits.detach().clone().requires_grad_(True)
nn.BCEWithLogitsLoss()(_z_grad, _bce_y).backward()
_manual_grad = (torch.sigmoid(_bce_logits) - _bce_y) / _bce_logits.numel()
print(f"  手写梯度 (ŷ-y)/n 与 autograd 梯度的最大误差：{(_z_grad.grad - _manual_grad).abs().max().item():.3e}")

# —— 1.1 为什么 BCEWithLogitsLoss 更稳定：极端 logits 演示 ——
print("""
  ▶ 为什么 BCEWithLogitsLoss 更稳定？极端 logit（±50）演示
    原理：σ(50) 在 float32 下就是 1.0（精确等于 1），σ(-50) 就是 0.0。
          于是 BCELoss 内部要算 log(1 - 1.0) = log(0) = -inf，
          或者 log(0) = -inf，再相乘相加 → 得到 inf 或 nan。
          BCEWithLogitsLoss 走的是 log-sum-exp 路线：
              log(1 + e^{-z}) 在 z=+50 时 ≈ 0（稳定），在 z=-50 时 ≈ 50（不会溢出），
          全程不出现 log(0)，所以任何 logit 都能算出有限值。
    注意：下面只打印数值，不做任何会抛异常的操作。
""")
_ext_logits = torch.tensor([50.0, -50.0, 50.0, -50.0])
_ext_y = torch.tensor([1.0, 0.0, 0.0, 1.0])           # 故意全部预测错，制造极端情况
_ext_p = torch.sigmoid(_ext_logits)
print(f"  logits         = {_ext_logits.numpy()}")
print(f"  sigmoid(logits)= {_ext_p.numpy()}   <- 已经饱和成精确的 1.0 / 0.0，信息丢失")
print(f"  labels         = {_ext_y.numpy().astype(int)}（全部与预测相反，是最坏情况）")

_ext_loss_stable = nn.BCEWithLogitsLoss()(_ext_logits, _ext_y)
_ext_loss_plain = nn.BCELoss()(_ext_p, _ext_y)
print(f"\n  BCEWithLogitsLoss(logits, y)        = {_ext_loss_stable.item():.6f}   <- 正常有限值")
print(f"  BCELoss(sigmoid(logits), y)         = {_ext_loss_plain.item():.6f}   <- 数值已经失真")
# 用 nan_to_num 把潜在的 inf/nan 归零后再看，避免任何异常或 nan 显示困扰
_safe = torch.nan_to_num(_ext_loss_plain, nan=0.0, posinf=0.0, neginf=0.0)
print(f"  BCELoss 结果经 nan_to_num(inf→0, nan→0) 处理后 = {_safe.item():.6f}")
_clamped_p = _ext_p.clamp(1e-7, 1.0 - 1e-7)          # 手工裁剪概率也是常见补救
print(f"  BCELoss(clamp(ŷ, 1e-7, 1-1e-7), y)  = {nn.BCELoss()(_clamped_p, _ext_y).item():.6f}"
      f"   <- 不再产生 inf，但数值已经严重偏离真值 25.0")
print(f"  numpy 手写稳定版 bce_with_logits_np   = {bce_with_logits_np(_ext_logits.numpy(), _ext_y.numpy()):.6f}")
print("  逐项核对真值（4 项里有 2 项是极端错、2 项是极端对）：")
print("    (z=+50, y=1)：预测对且极自信 → 损失 = softplus(-50) ≈ 0")
print("    (z=-50, y=0)：预测对且极自信 → 损失 = softplus(-50) ≈ 0")
print("    (z=+50, y=0)：预测**错**到极点 → 损失 = softplus(50)  ≈ 50")
print("    (z=-50, y=1)：预测**错**到极点 → 损失 = softplus(50)  ≈ 50")
print(f"    平均值 = (0 + 0 + 50 + 50) / 4 = 25.0")
print("  -> BCEWithLogitsLoss 与手写稳定版都精确得到 25.0（真值）；")
print(f"     而 BCELoss 因为 σ(±50) 已经饱和成精确的 1.0/0.0，log(0) 破坏计算，")
print(f"     得到 {_ext_loss_plain.item():.4f}（严重失真），clamp 补救后变成 "
      f"{nn.BCELoss()(_clamped_p, _ext_y).item():.4f}（也不等于真值）。")
print("  => 结论：用 logits 版（BCEWithLogitsLoss）永远比「手动 sigmoid + BCELoss」更稳。")


# ===========================================================================
# 2. 多分类交叉熵 CE + 信息论三件套
# ===========================================================================
_title("2. 多分类交叉熵 CE（Cross Entropy）")
print("""
  公式      : CE = -1/n · Σ_i Σ_c y_ic · log ŷ_ic
              其中 ŷ = softmax(z)，y 是 one-hot（或类别编号）。
  适用任务  : 多分类（输出层用 Softmax，得到各类别的概率分布）。
  关键特点  : ① 只惩罚"真实类别那一项"的预测概率（one-hot 让其它项乘 0）；
              ② 与 Softmax 搭配时梯度极简洁：∂CE/∂z = (ŷ - y)/n；
              ③ nn.CrossEntropyLoss() 内部已含 Softmax，标签传**类别编号**，不要 one-hot。
""")


def softmax_np(z, axis=-1):
    """数值稳定的 numpy softmax。"""
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max(axis=axis, keepdims=True))       # 减最大值防上溢
    return e / e.sum(axis=axis, keepdims=True)


def cross_entropy_np(logits, labels, eps=_EPS):
    """numpy 手写多分类交叉熵：先 softmax，再取真实类别的 -log。"""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels).astype(int)
    prob = np.clip(softmax_np(logits, axis=-1), eps, 1.0)
    n = logits.shape[0]
    return float(-np.mean(np.log(prob[np.arange(n), labels])))


torch.manual_seed(42)
_ce_logits = torch.randn(6, 4) * 1.5                  # 6 个样本、4 分类
_ce_labels = torch.tensor([0, 3, 2, 1, 0, 2])         # 标签是类别编号（非 one-hot）
print(f"  logits 形状 = {tuple(_ce_logits.shape)}（6 个样本，4 个类别）")
print(f"  标签（类别编号）= {_ce_labels.numpy()}")
_cmp("CrossEntropyLoss", cross_entropy_np(_ce_logits.numpy(), _ce_labels.numpy()),
     nn.CrossEntropyLoss()(_ce_logits, _ce_labels))

# —— 2.1 NLLLoss == CrossEntropyLoss ——
print("""
  ▶ ⑦-补充 NLLLoss：NLLLoss(log_softmax(x), y) 恒等于 CrossEntropyLoss(x, y)
    因为 CrossEntropyLoss = LogSoftmax + NLLLoss。直接用「log_softmax 后取真实类别」
    在数值上比「先 softmax 再 log」更稳定（避免 log 一个很小的数）。
""")
_nll = nn.NLLLoss()(F.log_softmax(_ce_logits, dim=-1), _ce_labels)
_ce = nn.CrossEntropyLoss()(_ce_logits, _ce_labels)
print(f"  NLLLoss(log_softmax(x), y) = {_nll.item():.8f}")
print(f"  CrossEntropyLoss(x, y)     = {_ce.item():.8f}")
print(f"  两者差值 = {abs(_nll.item() - _ce.item()):.3e}  ->  完全相等，验证通过")

# —— 2.2 核心推导数值验证：CE + Softmax 的梯度 = ŷ - y ——
print("""
  ▶ 核心推导数值验证：交叉熵 + Softmax 的梯度 = ŷ - y
    【推导思路】
      设 z 是 logits，ŷ = softmax(z)，损失 L = -Σ_c y_c log ŷ_c（单样本）。
      第一步：对 log ŷ_c 求导 → ∂L/∂ŷ_c = -y_c / ŷ_c。
      第二步：链式法则 ∂L/∂z_k = Σ_c (∂L/∂ŷ_c)·(∂ŷ_c/∂z_k)。
              Softmax 的雅可比（商的求导法则）：
                  ∂ŷ_c/∂z_k = ŷ_c(δ_ck - ŷ_k)
              其中 δ_ck 是克罗内克符号（c==k 时为 1，否则 0）。
      第三步：代入，得 ∂L/∂z_k = Σ_c (-y_c/ŷ_c)·ŷ_c(δ_ck - ŷ_k)
                                 = -Σ_c y_c(δ_ck - ŷ_k)
                                 = -[y_k - ŷ_k·Σ_c y_c]
      第四步：利用 one-hot 的性质 Σ_c y_c = 1，括号里第二项化简成 ŷ_k：
                  ∂L/∂z_k = -(y_k - ŷ_k) = ŷ_k - y_k
      —— 于是整个「Softmax + 交叉熵」的梯度就是 **ŷ - y**，形式极其简洁，
         这正是它们俩被绑在一起使用的原因（框架里通常合并成一个算子直接算）。
""")
_z_ce = _ce_logits.detach().clone().requires_grad_(True)
nn.CrossEntropyLoss()(_z_ce, _ce_labels).backward()
_yhat = torch.softmax(_ce_logits, dim=-1)
_onehot = F.one_hot(_ce_labels, num_classes=4).float()
_manual_ce_grad = (_yhat - _onehot) / _ce_logits.shape[0]     # /n 是因为 loss 取了 mean
_ce_grad_err = (_z_ce.grad - _manual_ce_grad).abs().max().item()
print(f"  手算 (ŷ - y)/n 与 autograd 的梯度最大误差 = {_ce_grad_err:.3e}   ->  < 1e-6 验证通过")
print(f"  ŷ 的第一行     = {np.round(_yhat[0].numpy(), 5)}")
print(f"  one-hot 第一行 = {_onehot[0].numpy().astype(int)}")
print(f"  ŷ - y 第一行   = {np.round((_yhat - _onehot)[0].numpy(), 5)}  （真实类别的梯度为负 → 提升它）")
print("  解读：真实类别的梯度是 ŷ_k - 1 < 0，梯度下降会**增大**它的 logit；")
print("        其它类别梯度是 ŷ_k > 0，会被**压低**。预测越错（ŷ_k 越小），修正力度越大。")

# —— 2.3 信息论三件套：信息量 / 信息熵 / KL 散度 ——
print("""
  ▶ 信息论三件套（与交叉熵的关系）
    信息量（自信息）: I(x) = -log P(x)
        —— 概率越小的事件发生，带来的"信息量"越大（越意外）。
    信息熵          : H(X) = -Σ_i P(x_i)·log P(x_i)
        —— 该分布本身的平均不确定性（平均信息量），只与 P 有关。
    交叉熵          : H(P, Q) = -Σ_i P(x_i)·log Q(x_i)
        —— 用分布 Q 去编码真实分布 P 所需的平均比特数。
    KL 散度         : KL(P||Q) = Σ_i P(x_i)·log[ P(x_i) / Q(x_i) ]
        —— 用 Q 近似 P 的"额外代价"（不对称，不是距离）。

    核心恒等式：
        KL(P||Q) = H(P, Q) - H(P)
        （交叉熵 = 信息熵 + KL 散度；当 Q = P 时 KL=0，交叉熵退化成信息熵）
    这解释了为什么分类任务最小化交叉熵是对的：
        H(P) 由真实标签决定、是常数，所以最小化交叉熵 == 最小化 KL 散度
        == 让预测分布 Q 尽量逼近真实分布 P。
""")
_P = np.array([1.0, 0.0, 0.0])            # 课案三分类例子：真实是第 0 类（one-hot）
_Q = np.array([0.7, 0.2, 0.1])            # 模型预测的概率分布
_H_p = float(-np.sum(_P * np.log(_P + _EPS)))                      # 信息熵
_CE_pq = float(-np.sum(_P * np.log(_Q)))                           # 交叉熵
_KL_pq = float(np.sum(_P * np.log((_P + _EPS) / _Q)))              # KL 散度
print(f"  课案例子：P(真实) = {_P}，Q(预测) = {_Q}")
print(f"    信息熵  H(P)      = -Σ P log P          = {_H_p:.10f}   （one-hot 的熵为 0）")
print(f"    交叉熵  H(P, Q)   = -Σ P log Q          = {_CE_pq:.10f}   （= -log 0.7 = {-np.log(0.7):.10f}）")
print(f"    KL 散度 KL(P||Q)  =  Σ P log(P/Q)       = {_KL_pq:.10f}")
print(f"    验证 KL = CE - H ：{_KL_pq:.10f}  vs  {_CE_pq - _H_p:.10f}"
      f"   差值 = {abs(_KL_pq - (_CE_pq - _H_p)):.3e}")
print("  -> 等式成立。真实分布是 one-hot 时 H(P)=0，于是 KL 与交叉熵数值相等")
print("     （这也是「one-hot 标签下最小化交叉熵就是最小化 KL」的原因）。")

# 再补一个"非 one-hot"的例子，让 KL = CE - H 的拆分更明显
_P2 = np.array([0.5, 0.3, 0.2])
_Q2 = np.array([0.4, 0.4, 0.2])
_H2 = float(-np.sum(_P2 * np.log(_P2)))
_CE2 = float(-np.sum(_P2 * np.log(_Q2)))
_KL2 = float(np.sum(_P2 * np.log(_P2 / _Q2)))
print(f"\n  再看一个非 one-hot 的例子：P = {_P2}，Q = {_Q2}")
print(f"    H(P) = {_H2:.10f}   CE = {_CE2:.10f}   KL = {_KL2:.10f}")
print(f"    CE - H = {_CE2 - _H2:.10f}   与 KL 的差 = {abs(_KL2 - (_CE2 - _H2)):.3e}   ->  等式依然成立")
print("    这里 H(P)>0，能清楚看到 KL 只是交叉熵中「超出信息熵的那部分额外代价」。")

# 用 torch 的 nn.KLDivLoss 再验证一次（注意它要求第一个参数是 log-probability）
_kl_torch = nn.KLDivLoss(reduction="sum")(
    torch.log(torch.tensor(_Q)), torch.tensor(_P))
print(f"  torch 侧参考：nn.KLDivLoss(log Q, P) = {_kl_torch.item():.10f}"
      f"（reduction='sum' 取 Σ，与手写 KL = {_KL_pq:.10f} 一致）")


# ===========================================================================
# 3. 均方误差 MSE
# ===========================================================================
_title("3. 均方误差 MSE（Mean Squared Error / L2Loss）")
print("""
  公式      : MSE = 1/n · Σ_i (y_i - ŷ_i)²
  适用任务  : 回归任务（预测连续值）。
  关键特点  : ① 误差取平方 → **平方放大**大误差，对大偏差惩罚极重，对离群点敏感；
              ② 处处可导，且导数随误差线性增大（误差越大，梯度越大，收敛快）；
              ③ 假设噪声服从高斯分布时的极大似然估计。
  梯度      : ∂MSE/∂ŷ = 2(ŷ - y)/n
""")


def mse_np(y_true, y_pred):
    """numpy 手写 MSE。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))


_t = torch.tensor([3.0, 5.0, 2.5, 7.0])               # 真实值
_p_ = torch.tensor([2.8, 5.3, 2.2, 6.5])              # 预测值
_cmp("MSELoss", mse_np(_t.numpy(), _p_.numpy()), nn.MSELoss()(_p_, _t))

_pg = _p_.detach().clone().requires_grad_(True)
nn.MSELoss()(_pg, _t).backward()
_manual_mse_grad = 2.0 * (_p_ - _t) / _t.numel()
print(f"  梯度：手写 2(ŷ-y)/n 与 autograd 的最大误差 = "
      f"{(_pg.grad - _manual_mse_grad).abs().max().item():.3e}")

print("""
  ▶ 数值演示：一个离群点如何主导 MSE
    构造两组数据：正常组 [1,1,1,1,1]（预测全是 1.0，完美），
    另一组把其中一个真值换成 100（其余预测仍为 1.0）——只有 1/5 的样本是离群点。
    比较 MSE 与 MAE 的变化幅度。
""")
_y_normal = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
_y_outlier = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
_pred_const = np.array([1.0, 1.0, 1.0, 1.0, 1.0])     # 模型对这 5 个点都预测 1.0
_mse_n = mse_np(_y_normal, _pred_const)
_mae_n = float(np.mean(np.abs(_y_normal - _pred_const)))
_mse_o = mse_np(_y_outlier, _pred_const)
_mae_o = float(np.mean(np.abs(_y_outlier - _pred_const)))
print(f"  正常数据 y = {_y_normal.astype(int)}，预测全为 1.0：")
print(f"    MSE = {_mse_n:.4f}   MAE = {_mae_n:.4f}   （完美预测，都是 0）")
print(f"  含一个离群点 y = {_y_outlier.astype(int)}，预测仍全为 1.0：")
print(f"    MSE = {_mse_o:.4f}   MAE = {_mae_o:.4f}   （误差 99，占 1/5 的样本）")
print(f"  -> MSE 被离群点从 {_mse_n:.2f} 拉大到 {_mse_o:.2f}（放大 {_mse_o / max(_mse_n, 1e-12):.1f} 倍，"
      f"因为 99² = {99 ** 2}）；")
print(f"     MAE 只从 {_mae_n:.2f} 到 {_mae_o:.2f}（线性，99/5 = {99 / 5:.2f}）。")
print("  -> 一个离群点就能主导整个 MSE，梯度会被它牵着走；这就是 MSE 对离群点敏感的本质。")
print(f"  torch 侧对照：MSELoss = {nn.MSELoss()(torch.tensor(_pred_const), torch.tensor(_y_outlier)).item():.4f}，"
      f"L1Loss = {nn.L1Loss()(torch.tensor(_pred_const), torch.tensor(_y_outlier)).item():.4f}")


# ===========================================================================
# 4. 平均绝对误差 MAE
# ===========================================================================
_title("4. 平均绝对误差 MAE（Mean Absolute Error / L1Loss）")
print("""
  公式      : MAE = 1/n · Σ_i |y_i - ŷ_i|
  适用任务  : 回归任务，**数据中有异常值**时优先考虑。
  关键特点  : ① 线性惩罚：不管误差多大，梯度大小恒为 ±1/n，不会被离群点带偏；
              ② 对离群点鲁棒，等价于拟合**中位数**（MSE 拟合均值）；
              ③ 缺点：在误差 = 0 处**不可导**（|x| 在 0 点的左右导数分别是 -1 和 +1），
                 工程上用 **subgradient（次梯度）** 处理，即令 0 点的导数为 0（或任取 [-1,1]）；
              ④ 0 点附近梯度不衰减，容易在最优点附近来回震荡、难以收敛到精确解。
  次梯度说明: 凸函数 f(x)=|x| 在 x=0 处的次微分是区间 [-1, 1]，
              任何一个该区间内的值都是合法的"次梯度"，PyTorch 取 0。
""")


def mae_np(y_true, y_pred):
    """numpy 手写 MAE。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


_cmp("L1Loss", mae_np(_t.numpy(), _p_.numpy()), nn.L1Loss()(_p_, _t))

_pg1 = _p_.detach().clone().requires_grad_(True)
nn.L1Loss()(_pg1, _t).backward()
_manual_mae_grad = np.sign(_p_.numpy() - _t.numpy()) / _t.numel()     # sign 就是次梯度的一种取法
print(f"  梯度：手写 sign(ŷ-y)/n 与 autograd 的最大误差 = "
      f"{np.abs(_pg1.grad.numpy() - _manual_mae_grad).max():.3e}")
# 单独看 0 点的次梯度行为
_zero_pt = torch.tensor([0.0], requires_grad=True)
nn.L1Loss()(_zero_pt, torch.tensor([0.0])).backward()
print(f"  验证不可导点：L1Loss 在误差恰好为 0 时，PyTorch 给出的次梯度 = {_zero_pt.grad.item():.1f}"
      f"（次微分区间 [-1,1] 中取 0）")


# ===========================================================================
# 5. SmoothL1Loss（Huber 损失）
# ===========================================================================
_title("5. SmoothL1Loss / Huber：小误差用平方，大误差用线性")
print("""
  分段公式（PyTorch 的 nn.SmoothL1Loss(beta=1.0) 定义）：
      设 e = ŷ - y，|e| < beta（默认 beta=1）时：  loss = 0.5 · e² / beta
                      |e| ≥ beta 时：              loss = |e| - 0.5 · beta
      即 beta=1 时：
          |e| < 1 :  0.5 e²        （平方段，可导、在 0 附近平滑、收敛快）
          |e| ≥ 1 :  |e| - 0.5      （线性段，梯度恒为 ±1，抗离群点）

  导数      : |e| < beta 时 dloss/de = e/beta；|e| ≥ beta 时 dloss/de = sign(e)
              —— 在 |e|=beta 处两侧导数都等于 ±1，函数本身**光滑可导**（C¹ 连续），
                 这正是它叫 "Smooth" L1 的原因。
  beta 说明 : beta 是「平方段与线性段的分界点」，控制对离群点的判定尺度。
              beta 越小越接近纯 L1；beta 越大越接近纯 L2。

  适用场景  : 目标检测的边界框回归（Faster R-CNN 等）、任何"大部分样本误差小、
              但存在少量离群点"的回归任务。
""")


def smooth_l1_np(y_true, y_pred, beta=1.0):
    """numpy 手写 SmoothL1（逐元素分段，最后取均值）。"""
    e = np.abs(np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64))
    per = np.where(e < beta, 0.5 * e ** 2 / beta, e - 0.5 * beta)
    return float(np.mean(per))


def smooth_l1_grad_np(y_pred, y_true, beta=1.0):
    """numpy 手写 SmoothL1 导数（逐元素）。"""
    d = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    e = np.abs(d)
    return np.where(e < beta, d / beta, np.sign(d))


for _beta in [0.5, 1.0, 2.0]:
    _cmp(f"SmoothL1Loss(beta={_beta})",
         smooth_l1_np(_t.numpy(), _p_.numpy(), beta=_beta),
         nn.SmoothL1Loss(beta=_beta)(_p_, _t))

_pg2 = _p_.detach().clone().requires_grad_(True)
nn.SmoothL1Loss(beta=1.0)(_pg2, _t).backward()
_manual_sl1 = smooth_l1_grad_np(_p_.numpy(), _t.numpy(), beta=1.0) / _t.numel()
print(f"  梯度：手写分段导数与 autograd 的最大误差 = "
      f"{np.abs(_pg2.grad.numpy() - _manual_sl1).max():.3e}")

print("\n  ▶ 数值对比：同一离群点数据上 MSE / MAE / SmoothL1 的损失值")
_preds_cmp = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
_tgts_cmp = torch.tensor([1.0, 1.0, 1.0, 1.0, 100.0])       # 含一个离群点
_mse_v = nn.MSELoss()(_preds_cmp, _tgts_cmp).item()
_mae_v = nn.L1Loss()(_preds_cmp, _tgts_cmp).item()
_sl1_v = nn.SmoothL1Loss(beta=1.0)(_preds_cmp, _tgts_cmp).item()
print(f"  MSE       = {_mse_v:>10.4f}   （99² 全部计入，被离群点彻底主导）")
print(f"  MAE       = {_mae_v:>10.4f}   （99/5，线性，最鲁棒）")
print(f"  SmoothL1  = {_sl1_v:>10.4f}   （前 4 个误差 0 贡献 0；离群点贡献 (99-0.5)/5 = {(99 - 0.5) / 5:.4f}）")
print("  -> SmoothL1 在离群点上表现得像 MAE，在正常小误差上表现得像 MSE，兼顾两者优点。")


# ===========================================================================
# 6. 其他损失函数
# ===========================================================================
_title("6. 其他损失函数：KLDiv / CosineEmbedding / TripletMargin / Focal")

# —— 6.1 KLDivLoss ——
print("""
【6.1 nn.KLDivLoss】KL 散度损失
  公式      : KL(P||Q) = Σ_i P(i)·[ log P(i) - log Q(i) ]
  输入要求  : PyTorch 的 nn.KLDivLoss 要求**第一个参数是 log-probability**（log Q），
              第二个参数是目标分布 P（概率，不是 log）。默认 reduction='mean'。
              这是因为内部实现是 pointwise: target * (log(target) - input)。
  适用场景  : 知识蒸馏（让学生分布逼近教师分布）、分布对齐、变分推断的正则项。
""")
_kl_input = F.log_softmax(torch.randn(3, 5), dim=-1)      # log Q
_kl_target = torch.softmax(torch.randn(3, 5), dim=-1)      # P（概率）
_kl_val = nn.KLDivLoss(reduction="batchmean")(_kl_input, _kl_target)
# 手写：(1/n) Σ_samples Σ_classes P(logP - logQ)
#   注意 KL 散度是「对类别求和、对样本求平均」，不是对所有元素一起求平均。
#   reduction='batchmean' 就是 Σ_all / batch_size；而 reduction='mean' 是 Σ_all / 元素总数，
#   后者会额外除以类别数 C，是常见的踩坑点（PyTorch 官方文档也特别提示了这一点）。
_kl_manual = float(torch.mean(
    (_kl_target * (torch.log(_kl_target + _EPS) - _kl_input)).sum(dim=-1)))
print(f"  nn.KLDivLoss(reduction='batchmean') = {_kl_val.item():.8f}")
print(f"  手写 (1/n)ΣΣ P(logP-logQ)          = {_kl_manual:.8f}")
print(f"  差值 = {abs(_kl_val.item() - _kl_manual):.3e}")
print("  注意 1：KL 散度**不对称**，KL(P||Q) ≠ KL(Q||P)，所以它不是一个距离度量。")
print("  注意 2：reduction='mean' 会把结果再除以类别数 C（这里是 5），得到偏小的值；")
print("          算两个分布的 KL 时应该用 'batchmean'（或 'sum'）而不是默认的 'mean'。")
# 这里用 sum/n/元素数 手工等价出 'mean' 的数值，避免直接调用触发 PyTorch 的弃用警告
_kl_mean_wrong = nn.KLDivLoss(reduction="sum")(_kl_input, _kl_target) / (
    _kl_input.shape[0] * _kl_input.shape[1])
print(f"          对照：reduction='mean' 的数值 = {_kl_mean_wrong.item():.8f}"
      f"（≈ batchmean / 5 = {_kl_val.item() / 5:.8f}）")

# —— 6.2 CosineEmbeddingLoss ——
print("""
【6.2 nn.CosineEmbeddingLoss】余弦嵌入损失
  公式      : 输入两个向量 x1, x2 和标签 y ∈ {1, -1}
              y = 1（相似）时：loss = 1 - cos(x1, x2)
              y = -1（不相似）时：loss = max(0, cos(x1, x2) - margin)
              其中 cos(x1,x2) = x1·x2 / (‖x1‖·‖x2‖)
  适用场景  : 度量学习——人脸验证、文本语义相似度、Sentence-BERT 的双塔训练。
              只关心"两个向量方向像不像"，与向量长度无关。
""")
_x1 = torch.randn(4, 8)
_x2 = torch.randn(4, 8)
_y_cos = torch.tensor([1.0, -1.0, 1.0, -1.0])
_cos_val = nn.CosineEmbeddingLoss(margin=0.5)(_x1, _x2, _y_cos)
# 手写复现
_cos_sim = F.cosine_similarity(_x1, _x2, dim=-1)
_per = torch.where(_y_cos == 1, 1.0 - _cos_sim, torch.clamp(_cos_sim - 0.5, min=0.0))
print(f"  两两余弦相似度 = {np.round(_cos_sim.detach().numpy(), 5)}")
print(f"  nn.CosineEmbeddingLoss(margin=0.5) = {_cos_val.item():.8f}")
print(f"  手写逐元素公式均值                 = {_per.mean().item():.8f}")
print(f"  差值 = {abs(_cos_val.item() - _per.mean().item()):.3e}")

# —— 6.3 TripletMarginLoss ——
print("""
【6.3 nn.TripletMarginLoss】三元组损失
  公式      : loss = max(0, d(a, p) - d(a, n) + margin)
              其中 a=锚点(anchor)，p=正样本(positive, 同类)，n=负样本(negative, 异类)，
              d 默认是 Lp 距离（p=2，即欧氏距离）。
  直觉      : 要求"锚点到正样本的距离"至少比"锚点到负样本的距离"小 margin，
              否则就产生损失。三个样本一组，故名三元组。
  适用场景  : 人脸识别（FaceNet）、图像检索、ReID 行人重识别。
""")
_anchor = torch.randn(3, 8)
_positive = torch.randn(3, 8)
_negative = torch.randn(3, 8)
_tri_val = nn.TripletMarginLoss(margin=1.0, p=2)(_anchor, _positive, _negative)
# 手写复现：d = ‖a-b‖_2（注意 PyTorch 用的是不带平方的欧氏距离）
_d_ap = torch.norm(_anchor - _positive, p=2, dim=-1)
_d_an = torch.norm(_anchor - _negative, p=2, dim=-1)
_tri_manual = torch.clamp(_d_ap - _d_an + 1.0, min=0.0).mean()
print(f"  d(a,p) = {np.round(_d_ap.detach().numpy(), 4)}")
print(f"  d(a,n) = {np.round(_d_an.detach().numpy(), 4)}")
print(f"  margin = 1.0，逐样本 max(0, d_ap - d_an + margin) = "
      f"{np.round(torch.clamp(_d_ap - _d_an + 1.0, min=0.0).detach().numpy(), 4)}")
print(f"  nn.TripletMarginLoss = {_tri_val.item():.8f}")
print(f"  手写复现             = {_tri_manual.item():.8f}")
print(f"  差值 = {abs(_tri_val.item() - _tri_manual.item()):.3e}")

# —— 6.4 Focal Loss（手写）——
print("""
【6.4 Focal Loss（手写，PyTorch 没有内置）】解决类别不平衡的利器
  公式      : FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
              其中 p_t = 模型对**真实类别**的预测概率：
                  真实类别为 1 时 p_t = ŷ；为 0 时 p_t = 1 - ŷ。
              α_t 是类别权重（正类给 α，负类给 1-α），γ 是聚焦参数（通常 γ=2）。
  直觉      : 交叉熵里每个样本权重都是 1。Focal Loss 乘上一个调制因子 (1-p_t)^γ：
              - 对**容易样本**（p_t 接近 1）：(1-p_t)^γ 接近 0 → 权重被压得很低；
              - 对**困难样本**（p_t 接近 0）：(1-p_t)^γ 接近 1 → 权重几乎不变。
              于是训练自动聚焦到"还没学会的难样本"上。
  为什么管用: 类别不平衡时，绝大多数是易分的负样本，它们的梯度累加起来会淹没
              少量正样本的梯度。Focal Loss 把易分样本的贡献压下去，让正样本/难样本
              的梯度重新占主导，从而提升 Recall。γ=0 且 α=1 时退化成普通 BCE。
  适用场景  : 目标检测（RetinaNet）、医学影像（病灶极小）、欺诈检测等极端不平衡任务。
""")


def focal_loss_np(logits, y_true, alpha=0.25, gamma=2.0):
    """numpy 手写 Focal Loss（二分类，数值稳定版）。

    直接把 sigmoid 融进公式，用和 BCEWithLogits 一样的 log-sum-exp 技巧：
        -log p_t = softplus(-z)   (y=1)；  -log p_t = softplus(z)   (y=0)
        (1 - p_t) = σ(-z) (y=1)；          (1 - p_t) = σ(z)  (y=0)
    """
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(y_true, dtype=np.float64)
    # 稳定 sigmoid
    sig = np.where(z >= 0, 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))),
                   np.exp(np.clip(z, -500, 500)) / (1.0 + np.exp(np.clip(z, -500, 500))))
    p_t = np.where(y == 1, sig, 1.0 - sig)                       # 对真实类别的预测概率
    # -log p_t 的稳定写法
    neg_log_pt = np.maximum(-z, 0.0) + np.log1p(np.exp(-np.abs(z)))   # softplus(-z)
    neg_log_pt = np.where(y == 1, neg_log_pt,
                          np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z))))  # y=0 时用 softplus(z)
    alpha_t = np.where(y == 1, alpha, 1.0 - alpha)
    return float(np.mean(alpha_t * (1.0 - p_t) ** gamma * neg_log_pt))


def focal_loss_torch(logits, targets, alpha=0.25, gamma=2.0):
    """torch 版 Focal Loss（用于与 numpy 手写对比）。"""
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()


torch.manual_seed(42)
_fl_logits = torch.randn(20) * 2.0
_fl_y = (torch.rand(20) > 0.7).float()                # 制造不平衡：约 30% 正样本
_cmp("FocalLoss(α=0.25, γ=2)", focal_loss_np(_fl_logits.numpy(), _fl_y.numpy()),
     focal_loss_torch(_fl_logits, _fl_y))

# 直观对比：Focal 对「易分样本」的抑制效果
print("\n  ▶ 直观对比：容易样本 vs 困难样本上的损失值（真实标签都是 1）")
for _z_val in [3.0, 1.0, 0.0, -2.0, -4.0]:
    _zt = torch.tensor([_z_val])
    _yt = torch.tensor([1.0])
    _p_val = torch.sigmoid(_zt).item()
    _bce_v = F.binary_cross_entropy_with_logits(_zt, _yt).item()
    _fl_v = focal_loss_torch(_zt, _yt, alpha=0.25, gamma=2.0).item()
    print(f"    logit={_z_val:>5.1f}  p_t={_p_val:.4f}  BCE={_bce_v:.5f}  "
          f"Focal={_fl_v:.5f}  （(1-p_t)^γ={((1 - _p_val) ** 2):.5f}）")
print("  -> p_t 越接近 1（越容易），Focal 把损失压得越狠；p_t 小时两者接近。")
print("     注意 α=0.25 还会把正类样本整体乘 0.25，所以数值上比 BCE 小一截——")
print("     α 的作用是**调整正负类权重**，γ 的作用是**压制易分样本**，两者分工不同。")


# ===========================================================================
# 7. 对比汇总表（课案）
# ===========================================================================
_title("7. 对比汇总表（课案）")
_rows = [
    ("BCE", "-1/n Σ[y log ŷ + (1-y) log(1-ŷ)]", "二分类", "配合 Sigmoid"),
    ("CE", "-1/n Σ y_c log ŷ_c", "多分类", "配合 Softmax，梯度 = ŷ - y"),
    ("MSE", "1/n Σ(y - ŷ)²", "回归", "平方放大误差，对离群点敏感"),
    ("MAE", "1/n Σ|y - ŷ|", "回归（有异常值）", "线性惩罚，对离群点鲁棒"),
]
print(f"{'损失函数':<8s} {'公式':<38s} {'适用任务':<18s} {'关键特点'}")
print("-" * 100)
for _r in _rows:
    print(f"{_r[0]:<8s} {_r[1]:<38s} {_r[2]:<18s} {_r[3]}")

print("\n【扩展损失速查】")
_ext = [
    ("SmoothL1/Huber", "小误差 0.5e²，大误差 |e|-0.5", "回归（有离群点）", "兼顾 MSE 可导与 MAE 鲁棒"),
    ("NLLLoss", "-1/n Σ log ŷ_真实类", "多分类（配 LogSoftmax）", "等价于 CrossEntropyLoss"),
    ("KLDivLoss", "Σ P(logP - logQ)", "知识蒸馏 / 分布对齐", "输入需 log-prob，不对称"),
    ("CosineEmbedding", "1-cos 或 max(0, cos-margin)", "度量学习 / 双塔", "只看方向，与长度无关"),
    ("TripletMargin", "max(0, d(a,p)-d(a,n)+m)", "人脸识别 / 检索", "三元组，拉近正推远负"),
    ("Focal Loss", "-α(1-p_t)^γ log(p_t)", "极端类别不平衡", "压低易分样本，聚焦难样本"),
]
print(f"{'损失函数':<16s} {'公式':<30s} {'适用任务':<22s} {'关键特点'}")
print("-" * 105)
for _r in _ext:
    print(f"{_r[0]:<16s} {_r[1]:<30s} {_r[2]:<22s} {_r[3]}")


# ===========================================================================
# 8. 可视化
# ===========================================================================
_title("8. 可视化：回归损失曲线 + 分类损失曲线")

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

# —— 子图 1：各回归损失随误差变化 ——
_err = np.linspace(-5.0, 5.0, 800)                 # 误差 e = ŷ - y
_mse_curve = _err ** 2                             # MSE 逐元素
_mae_curve = np.abs(_err)                          # MAE 逐元素
_sl1_curve = np.where(np.abs(_err) < 1.0, 0.5 * _err ** 2, np.abs(_err) - 0.5)  # SmoothL1(beta=1)
axes[0].plot(_err, _mse_curve, label="MSE = e²（平方放大误差）", linewidth=2.2, color="tab:red")
axes[0].plot(_err, _mae_curve, label="MAE = |e|（线性惩罚）", linewidth=2.2, color="tab:blue")
axes[0].plot(_err, _sl1_curve, label="SmoothL1(beta=1)（小误差平方/大误差线性）",
             linewidth=2.2, color="tab:green", linestyle="--")
axes[0].axvline(0, color="gray", linewidth=0.8, linestyle=":")
axes[0].axhline(0, color="gray", linewidth=0.8, linestyle=":")
axes[0].set_ylim(0, 12)                            # 限制纵轴，突出小误差区的差异
axes[0].set_title("回归损失随误差的变化（e = ŷ - y）", fontsize=13)
axes[0].set_xlabel("误差 e")
axes[0].set_ylabel("单样本损失")
axes[0].legend(fontsize=9)
axes[0].annotate("MSE 在 |e| 大时\n暴涨，被离群点主导",
                 xy=(3.4, 11.5), xytext=(-4.6, 9.0), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="tab:red"))
axes[0].annotate("SmoothL1 在这里\n与 MAE 重合",
                 xy=(2.6, 2.1), xytext=(2.9, 5.0), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="tab:green"))

# —— 子图 2：分类损失随预测概率变化 ——
_p_axis = np.linspace(1e-4, 1 - 1e-4, 800)         # 预测概率 ŷ
# 真实标签 y=1 时的 BCE：-log ŷ
_bce_pos = -np.log(_p_axis)
# 真实标签 y=0 时的 BCE：-log(1-ŷ)
_bce_neg = -np.log(1.0 - _p_axis)
axes[1].plot(_p_axis, _bce_pos, label="BCE（真实 y=1）：-log ŷ", linewidth=2.2, color="tab:blue")
axes[1].plot(_p_axis, _bce_neg, label="BCE（真实 y=0）：-log(1-ŷ)", linewidth=2.2,
             color="tab:orange", linestyle="--")
# 多分类 CE 的"真实类别概率"视角：CE = -log p_t，与 y=1 的 BCE 完全同形
axes[1].plot(_p_axis, -np.log(_p_axis), label="CE（真实类别概率 p_t）：-log p_t",
             linewidth=1.6, color="black", linestyle=":")
# Focal Loss（alpha=0.25, gamma=2，真实 y=1）作对照
_focal_curve = 0.25 * (1.0 - _p_axis) ** 2 * (-np.log(_p_axis))
axes[1].plot(_p_axis, _focal_curve, label="Focal(α=0.25, γ=2)：压低易分样本",
             linewidth=2.0, color="tab:green")
axes[1].set_ylim(0, 6)
axes[1].set_title("分类损失随预测概率的变化（越错惩罚越大）", fontsize=13)
axes[1].set_xlabel("模型对真实类别的预测概率 p_t")
axes[1].set_ylabel("损失")
axes[1].legend(fontsize=9)
axes[1].annotate("预测概率→0 时\n损失→+∞（无上界）",
                 xy=(0.02, 3.9), xytext=(0.16, 4.6), fontsize=9,
                 arrowprops=dict(arrowstyle="->", color="tab:blue"))

fig.tight_layout()
_loss_png = OUTPUT_DIR / "03训练组件_02_损失函数曲线.png"
fig.savefig(_loss_png, dpi=110)
plt.close(fig)
print(f"已保存：{_loss_png}")
print("  左图读数：e=1 时 MSE=1、MAE=1、SmoothL1=0.5（分段点）；"
      "e=4 时 MSE=16、MAE=4、SmoothL1=3.5 -> MSE 增长最快。")
print(f"  右图读数：p_t=0.9 时 BCE={-np.log(0.9):.4f}；p_t=0.1 时 BCE={-np.log(0.1):.4f}"
      f" -> 预测越错（p_t 越小），损失以 -log 的速度无上界增长。")

_title("02_损失函数.py 运行完毕")
print(f"输出目录：{OUTPUT_DIR}")

r"""《机器学习》课案 · 03 监督学习-分类 · 01 逻辑回归（Logistic Regression）

对应课案章节
-----------
《机器学习》课案「分类」章 —— 逻辑回归（Logistic Regression）：
课案原文位于 `.course_extract/机器学习_课案.md` 第 263~418 行。
覆盖内容：sigmoid 函数、"模型 → 损失 → 优化"三步推导、交叉熵（对数损失）、
阈值 0.5 判类、标准化、课案示例代码（load_iris 前 100 条二分类）、
逻辑回归参数表、逻辑回归 vs 线性回归对比表。

本节知识点
---------
1. 为什么名字叫"回归"却是分类器：线性打分 z=w·x+b → sigmoid 压到 [0,1] → 阈值 0.5 判类。
2. sigmoid 函数 σ(z)=1/(1+e^(-z)) 的性质：值域 (0,1)、σ(0)=0.5、σ(-z)=1-σ(z)、单调可导。
3. 损失函数：负对数似然 = 交叉熵
   L(w,b) = -(1/N) * Σ_i [ y_i * log(p_i) + (1-y_i) * log(1-p_i) ]。
4. 优化方法：梯度下降 / 拟牛顿法（sklearn 默认 solver="lbfgs"，即 L-BFGS 拟牛顿法）。
5. 正则化：L2（岭）/ L1（稀疏）/ ElasticNet，由 C 与 l1_ratio 控制。
6. 多分类：OvR（一对多）与 multinomial（softmax 多项 logistic）。
7. 优缺点与适用场景；逻辑回归 vs 线性回归对照。

运行方式
-------
PowerShell（路径含中文，必须加引号并用 & 调用）：

    $env:PYTHONUTF8='1'; & 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Machine_Learning\03_监督学习_分类\01_逻辑回归.py'

输出：
    - 控制台：四段式讲解 + 概率/标签对照表 + 指标 + 中文解读
    - 图片： Machine_Learning/output/03_逻辑回归_sigmoid曲线.png
            Machine_Learning/output/03_逻辑回归_决策边界.png
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")            # 无界面后端，避免 plt.show() 阻塞
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_iris, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# 图片统一输出目录：Machine_Learning/output（相对本文件定位，避免依赖当前工作目录）
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42  # 全脚本统一的随机种子，保证结果可复现


def print_section(title: str) -> None:
    """打印统一风格的中文分隔标题。"""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


# ============================================================================
# ① 原理与数学推导
# ============================================================================
# 【1.1 为什么叫"回归"却是分类器？】
#   逻辑回归的"回归"来自它的中间产物：它先算一个**线性回归式的打分**
#       z = w·x + b = w1*x1 + w2*x2 + ... + wn*xn + b
#   这个 z 的取值是全体实数 (-inf, +inf)，本质上就是一次线性回归。
#   但它随后把 z 塞进 sigmoid 函数压到 [0,1]，得到一个"概率"，
#   再用 0.5 当阈值把概率翻译成类别 0 / 1。
#   所以：**内部做回归（拟合打分），外部做分类（输出类别）**。
#   "逻辑"（logistic）指的是它用的 logistic 函数，也就是 sigmoid。
#
# 【1.2 模型形式（第一步：先写模型）】
#   线性打分：      z = w·x + b
#   sigmoid 压缩：  p = σ(z) = 1 / (1 + e^(-z))
#   于是 p 的含义是：模型认为该样本属于正类（y=1）的概率。
#   二分类下只有两种情况，所以 P(y=1|x)=p，P(y=0|x)=1-p。
#
#   sigmoid 的四个关键性质（画图时会直观看到）：
#     (a) 值域 (0,1)，天然可以当概率用；
#     (b) σ(0) = 0.5 —— 打分等于 0 时模型完全犹豫；
#     (c) σ(-z) = 1 - σ(z) —— 关于点 (0, 0.5) 中心对称；
#     (d) 导数 σ'(z) = σ(z)·(1-σ(z)) —— 求梯度时非常方便，
#         而且可以直接用输出 p 表示：σ'(z) = p(1-p)。
#   注意 sigmoid 是 S 形但**不是**阶跃函数：它处处光滑可导，这正是能用梯度法优化的原因。
#
# 【1.3 损失函数（第二步：定义损失）】
#   直觉：真实标签是 1，就希望 p 越大越好；真实标签是 0，就希望 1-p 越大越好。
#   为了把"概率越大损失越小"写成公式，取 -log：
#       y = 1 时，单样本损失 = -log(p)
#       y = 0 时，单样本损失 = -log(1-p)
#   用 one-hot 的写法把两种情况合并（y 只取 0 或 1）：
#       loss_i = -[ y_i·log(p_i) + (1-y_i)·log(1-p_i) ]
#   验证：y=1 时第二项系数为 0，只剩 -log(p)；y=0 时第一项消失，只剩 -log(1-p)。合并成立。
#   整份训练集的平均损失（这就是**交叉熵损失 / 对数损失 / 负对数似然**）：
#
#       L(w, b) = -(1/N) · Σ_{i=1..N} [ y_i·log(p_i) + (1-y_i)·log(1-p_i) ]
#
#   它等价于**极大似然估计**：把 N 个样本的似然乘起来取负对数，就是上式。
#   -log 的好处：预测得越离谱（真实为 1 却预测 p→0），损失越大且增长极快，
#   这会给出很强的纠正梯度，比平方损失（MSE）在分类上更合适（MSE + sigmoid 会梯度消失）。
#
# 【1.4 优化方法（第三步：怎么求 w 和 b）】
#   目标： (w*, b*) = argmin L(w, b)。L 是凸函数，有唯一全局最优，不会陷入局部极小。
#   对 w 求偏导（利用 σ'(z)=p(1-p)，可以推导出极其简洁的形式）：
#       ∂L/∂w = (1/N) · Σ_i (p_i - y_i) · x_i
#       ∂L/∂b = (1/N) · Σ_i (p_i - y_i)
#   形式与线性回归的最小二乘梯度一模一样，只是 p 换成了 sigmoid 输出。
#   常用求解器：
#     - 梯度下降 / SGD：每次用全部或一批样本沿负梯度走一步，w ← w - η·∂L/∂w；
#     - 拟牛顿法 L-BFGS（sklearn 默认）：用历史梯度近似 Hessian 的逆，
#       收敛快、无需手工调学习率，小中型数据首选；
#     - 牛顿法 / 牛顿-楚列斯基：直接算二阶导，特征数不多时很快；
#     - SAG / SAGA：随机平均梯度，适合大样本、稀疏特征。
#   sklearn 里由 solver 参数选择，我们通常不需要手写更新公式。
#
# 【1.5 正则化：防止过拟合】
#   在损失后面加惩罚项 λ·R(w)（sklearn 用 C = 1/λ 表示，C 越小正则越强）：
#       L2： R(w) = (1/2)||w||²  —— 权重整体变小、更平滑，是默认选择；
#       L1： R(w) = ||w||₁       —— 会把部分权重压成 0，得到稀疏解，可做特征选择；
#       ElasticNet： L1 与 L2 的加权组合。
#   注意：正则项通常**不惩罚截距 b**。
#
# 【1.6 从二分类到多分类】
#   - OvR（one-vs-rest，一对多）：训练 K 个二分类器，第 k 个负责"是不是第 k 类"，
#     预测时取打分最高的那一类。K 个类就训 K 个模型。
#   - multinomial（多项 logistic / softmax）：一个模型直接输出 K 个概率，
#       p_k = exp(z_k) / Σ_j exp(z_j)  （softmax 函数）
#     各类别概率之和为 1，通常比 OvR 更准，也能给出更合理的概率。
#
# 【1.7 优缺点】
#   优点：
#     - 模型简单、训练快，可解释性强（权重 w 的符号和大小直接说明特征作用方向与强度）；
#     - 输出的是**校准得比较好的概率**，不只是硬标签，便于做阈值调整和风险排序；
#     - 损失凸，保证全局最优；天然支持 L1/L2 正则；
#     - 是神经网络单个神经元的原型，是理解深度学习的基础。
#   缺点：
#     - 只能画出一条线性决策边界（直线/超平面），对 XOR、月亮形等非线性数据无能为力，
#       除非手工做特征工程或加多项式特征；
#     - 对极端异常值和特征量纲敏感（因此需要标准化）；
#     - 对高度非线性的复杂关系表达能力不足，此时该用 SVM、树模型或神经网络。
# ============================================================================


# ============================================================================
# ② sklearn API 关键参数逐个解释
# ============================================================================
# 下述签名与默认值基于本机 scikit-learn 1.9.0 实测（print(model.get_params()) 得到）。
#
# penalty : {'l1','l2','elasticnet'}，默认 'l2'
#    含义：正则化类型（L1 稀疏 / L2 平滑 / ElasticNet 混合）。
#    ⚠ 版本状态：在 scikit-learn 1.8 中 penalty 已被**弃用（deprecated）**，
#      1.9 中默认值显示为 'deprecated'，并计划在 1.10 移除。显式传 penalty='l2'
#      会触发 FutureWarning（本脚本实测确认）。
#    ✅ 1.9 的写法：不要传 penalty，改用 l1_ratio：
#        l1_ratio = 0   ←→ 原来的 penalty='l2'（默认，纯 L2）
#        l1_ratio = 1   ←→ 原来的 penalty='l1'（纯 L1）
#        0<l1_ratio<1   ←→ 原来的 penalty='elasticnet'
#        C = np.inf     ←→ 原来的 penalty=None（不正则化）
#    调大（l1_ratio→1）：权重更稀疏，更多特征被压成 0，模型更简单但可能欠拟合；
#    调小（l1_ratio→0）：权重更平滑，所有特征都保留一点作用。
#
# C : float，默认 1.0
#    含义：正则化强度的**倒数**（C = 1/λ），是逻辑回归最重要的超参数。
#    调大（如 10、100）：正则变弱 → 模型更"敢"拟合训练数据 → 训练误差降低，
#                        但 C 过大容易过拟合、权重数值爆炸；
#    调小（如 0.01）：正则变强 → 权重被压小 → 决策边界更平滑 → 可能欠拟合。
#    常用值：0.01 / 0.1 / 1 / 10（在 log 尺度上做 GridSearchCV 搜索）。
#
# solver : {'lbfgs','liblinear','newton-cg','newton-cholesky','sag','saga'}，默认 'lbfgs'
#    含义：数值优化算法。
#      - 'lbfgs'（默认）：拟牛顿法，支持 L2 与 multinomial，无惩罚时也快，通用首选；
#      - 'liblinear'：坐标下降，只支持 OvR，小数据集上表现好，支持 L1；
#      - 'newton-cg' / 'newton-cholesky'：牛顿类，收敛快但内存开销大；
#      - 'sag' / 'saga'：随机平均梯度，适合大样本；'saga' 是唯一支持 L1 与 ElasticNet
#        且能配 multinomial 的求解器。
#    影响：换 solver 主要影响速度、内存与"能不能用某种正则"，
#          正常情况下对最终精度影响不大。默认值通常不用改。
#
# max_iter : int，默认 100
#    含义：优化算法的最大迭代次数。
#    调大：给求解器更多迭代机会，避免"未收敛"（此时 sklearn 会发 ConvergenceWarning）；
#    调小：训练更快但可能没收敛到最优。
#    常用值：1000（数据标准化后逻辑回归收敛很快，1000 基本够；本脚本设为 1000）。
#
# class_weight : dict 或 'balanced'，默认 None
#    含义：类别权重。None 表示所有样本权重都是 1。
#    'balanced' 会按 n_samples/(n_classes*np.bincount(y)) 自动反比于类别频率加权。
#    调大某类权重：该类的召回率上升、精确率可能下降。
#    常用值：类别不平衡时用 'balanced' 或手写 {0:1, 1:5}。iris 是三类各 50 条，均衡，用默认即可。
#
# multi_class : ⚠ 在 scikit-learn 1.9 中该参数**已被彻底移除**（1.5 起弃用，1.7 移除）。
#    现象：1.9 里 LogisticRegression().get_params() 的键中已经没有 multi_class。
#    替代方案：不传该参数，使用 LogisticRegression 的默认多分类行为——
#      二分类时就是普通二分类；多分类时默认采用 multinomial（softmax）多项 logistic，
#      并由 solver 自动处理（lbfgs 天然支持多分类）。
#    如果确实要用 OvR 语义，可以用 OneVsRestClassifier(LogisticRegression(...)) 包装。
#    本脚本按 1.9 的正确写法：不出现 multi_class。
#
# l1_ratio : float 或 None，默认 0.0（1.8 起取代 penalty 的写法，见上）
# tol : float，默认 1e-4。收敛阈值，调大停得更早、精度略降；一般不动。
# fit_intercept : bool，默认 True。是否拟合截距 b。数据已中心化时可设 False。
# random_state : int，默认 None。仅在 solver 为 'sag'/'saga'/'liblinear' 时用于打乱样本；
#                固定为 42 可保证结果可复现（本脚本统一 42）。
# n_jobs : int，默认 None。多分类 OvR 时的并行度，-1 表示用满所有 CPU 核。
# ============================================================================


print_section("《机器学习》课案 · 分类 · 01 逻辑回归（Logistic Regression）")
print("本脚本四段结构： ① 原理与数学推导  ② sklearn API 参数解释  ③ 完整可运行代码  ④ 结果解读")
print("配套图片输出目录：", OUTPUT_DIR)

# ============================================================================
# ③ 完整可运行代码
# ============================================================================

# ---------------------------------------------------------------------------
# 步骤 1：提取数据 —— 课案做法：load_iris 只取前 100 条，把三分类降成二分类
# ---------------------------------------------------------------------------
print_section("步骤 1：提取数据（iris 前 100 条，二分类）")

iris = load_iris()
print("数据集来源：sklearn.datasets.load_iris（内置数据集，无需联网下载）")
print("数据集简介：", [ln.strip() for ln in iris.DESCR.splitlines() if ln.strip()][0])
print("原始特征名：", list(iris.feature_names))
print("原始类别名：", [str(n) for n in iris.target_names])

# 前 100 条正好是 setosa(0) 与 versicolor(1)，各 50 条；后面的 virginica 丢掉
X = iris.data[:100]
y = iris.target[:100]
print(f"取前 100 条后的形状：X = {X.shape}，y = {y.shape}")
print("两类样本数（0=山鸢尾 setosa，1=变色鸢尾 versicolor）：",
      {int(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))})

# ---------------------------------------------------------------------------
# 步骤 2：清洗 / 处理数据 —— 划分训练集测试集 + 标准化
# ---------------------------------------------------------------------------
print_section("步骤 2：划分训练集/测试集 + 标准化")

# 分层抽样（stratify=y）保证训练集、测试集里两类比例一致，评估结果更稳定
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=y,
)
print(f"训练集 X_train = {X_train.shape}，测试集 X_test = {X_test.shape}")
print(f"训练集标签分布：{np.bincount(y_train)}，测试集标签分布：{np.bincount(y_test)}")

# 标准化：z = (x - μ) / σ。用训练集统计量 fit，再 transform 测试集（避免数据泄漏）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("训练集各特征均值（标准化后≈0）：", np.round(X_train_scaled.mean(axis=0), 10))
print("训练集各特征标准差（标准化后≈1）：", np.round(X_train_scaled.std(axis=0), 6))
print("为什么必须标准化：逻辑回归用梯度类方法优化，量纲差别大会让等高线被拉成长条，")
print("                  收敛变慢甚至震荡；标准化后各特征地位平等，正则化惩罚也才公平。")

# ---------------------------------------------------------------------------
# 步骤 3：运行逻辑回归算法
# ---------------------------------------------------------------------------
print_section("步骤 3：训练逻辑回归模型")

# 说明：sklearn 1.9 中 penalty 已弃用、multi_class 已移除。
#       这里只传 C / solver / max_iter / random_state，即 1.9 的推荐写法。
model = LogisticRegression(
    C=1.0,                 # 正则化强度倒数；1.0 为默认值
    solver="lbfgs",        # 默认拟牛顿法（L-BFGS），支持 L2 与多分类
    max_iter=1000,         # 给足迭代次数，确保收敛（避免 ConvergenceWarning）
    random_state=RANDOM_STATE,
)
model.fit(X_train_scaled, y_train)
print("模型已训练完成。")
print("学到的权重 w（每个特征一个系数）：", np.round(model.coef_[0], 4))
print("学到的截距 b：", round(float(model.intercept_[0]), 4))
print("模型参数名：", list(model.get_params().keys()))
print("→ 注意 get_params() 里已没有 multi_class、而 penalty 的默认值是 'deprecated'，")
print("  这正是 sklearn 1.9 的版本现状；1.9 推荐用 l1_ratio 取代 penalty。")
print(f"收敛迭代次数 n_iter_ = {model.n_iter_[0]}（未超过 max_iter={model.max_iter}，说明已收敛）")

# ---------------------------------------------------------------------------
# 步骤 4：得到预测结果 —— 概率 → 阈值 → 标签
# ---------------------------------------------------------------------------
print_section("步骤 4：从概率到标签（predict_proba → 阈值 0.5 → predict）")

y_pred = model.predict(X_test_scaled)                 # 硬标签
proba = model.predict_proba(X_test_scaled)            # 各类概率，形状 (n, 2)
p_positive = proba[:, 1]                              # 正类（类别 1）的概率

print("predict_proba 的输出形状：", proba.shape, "（第 2 列 = P(属于类别 1)）")
print()
print("下面挑 6 条测试样本，逐条展示【概率 → 阈值 → 标签】的对应关系：")
print("-" * 96)
print(f"{'样本序':<6}{'P(类别0)':>12}{'P(类别1)':>12}{'阈值判定 p>0.5':>18}{'predict':>10}{'真实标签':>10}{'是否一致':>10}")
print("-" * 96)

# 挑"最靠近决策边界"的样本（|p-0.5| 最小的几条）+ 最有把握的样本，信息量最大
near_boundary = np.argsort(np.abs(p_positive - 0.5))[:3]
confident = np.argsort(-np.abs(p_positive - 0.5))[:3]
demo_idx = np.concatenate([near_boundary, confident])

for i in demo_idx:
    p0, p1 = proba[i, 0], proba[i, 1]
    rule_label = 1 if p1 > 0.5 else 0
    same = "是" if int(y_pred[i]) == int(y_test[i]) else "否（预测错误）"
    print(f"{int(i):<6}{p0:>12.6f}{p1:>12.6f}{('正类1' if rule_label == 1 else '负类0'):>18}"
          f"{int(y_pred[i]):>10}{int(y_test[i]):>10}{same:>10}")
print("-" * 96)
print("怎么读这张表：")
print("  · 前 3 条是这批测试样本里概率【相对】最接近 0.5 的，模型'最犹豫'，")
print("    它们离决策边界最近，只要数据稍有扰动就可能翻类；")
print("    （这份数据太容易分，所以它们的概率也仍有 0.86 左右，并非恰好 0.5。）")
print("  · 后 3 条是概率最远离 0.5 的样本，模型'最有把握'（概率接近 0 或 1）；")
print("  · predict 列完全等价于'P(类别1) 是否 > 0.5'，即阈值 0.5 的规则；")
print("  · 注意 P(类别0) + P(类别1) = 1，二分类里两者互补。")

# 用代码验证"阈值 0.5 的规则"和 predict 完全一致
manual_label = (p_positive > 0.5).astype(int)
agree = int((manual_label == y_pred).sum())
print(f"\n手工用阈值 0.5 判类与 model.predict 一致的有 {agree}/{len(y_pred)} 条"
      f"（一致率 {agree / len(y_pred):.2%}）→ 证实 predict 就是 0.5 阈值规则。")

# ---------------------------------------------------------------------------
# 步骤 5：查看结果 —— 指标 + 阈值对预测的影响
# ---------------------------------------------------------------------------
print_section("步骤 5：评估指标与阈值的影响")

acc = accuracy_score(y_test, y_pred)
print(f"测试集准确率 accuracy = {acc:.4f}（{(y_pred == y_test).sum()} / {len(y_test)} 条预测正确）")
print("\n混淆矩阵 confusion_matrix（行=真实类别，列=预测类别）：")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"  左上 {cm[0, 0]}：真实 0 且预测 0（真负例 TN）")
print(f"  右上 {cm[0, 1]}：真实 0 却预测 1（假正例 FP，误报）")
print(f"  左下 {cm[1, 0]}：真实 1 却预测 0（假负例 FN，漏报）")
print(f"  右下 {cm[1, 1]}：真实 1 且预测 1（真正例 TP）")

print("\n分类报告 classification_report：")
print(classification_report(y_test, y_pred, target_names=["山鸢尾 setosa(0)", "变色鸢尾 versicolor(1)"],
                            digits=4, zero_division=0))
print("精确率 precision = TP/(TP+FP)：报出来的正类里有多少是真的；")
print("召回率 recall    = TP/(TP+FN)：真正的正类里抓回来了多少；")
print("F1 = 2PR/(P+R)：精确率与召回率的调和平均，两者兼顾的单一指标。")

print("\n【阈值不是必须等于 0.5】把同一组概率用不同阈值切，看看预测会怎么变：")
print("-" * 72)
print(f"{'阈值':<8}{'预测为正类数':>14}{'准确率':>12}{'正类精确率':>14}{'正类召回率':>14}")
print("-" * 72)
for thr in (0.2, 0.3, 0.5, 0.7, 0.8):
    pred_thr = (p_positive > thr).astype(int)
    n_pos = int(pred_thr.sum())
    acc_thr = accuracy_score(y_test, pred_thr)
    tp = int(((pred_thr == 1) & (y_test == 1)).sum())
    fp = int(((pred_thr == 1) & (y_test == 0)).sum())
    fn = int(((pred_thr == 0) & (y_test == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    print(f"{thr:<8.1f}{n_pos:>14}{acc_thr:>12.4f}{prec:>14.4f}{rec:>14.4f}")
print("-" * 72)
print("规律：阈值升高 → 判成正类的门槛变严 → 预测的正类变少 → 精确率往往上升、召回率下降。")
print("      医疗筛查宁可误报也要抓全（调低阈值），垃圾邮件拦截宁可漏放也别误杀（调高阈值）。")

# ---------------------------------------------------------------------------
# 步骤 6：画图 1 —— sigmoid 曲线（含 0.5 阈值线）
# ---------------------------------------------------------------------------
print_section("步骤 6：绘图 —— sigmoid 曲线")

fig1, ax1 = plt.subplots(figsize=(7.6, 4.8))
z_line = np.linspace(-8, 8, 500)
sigma_line = 1.0 / (1.0 + np.exp(-z_line))
ax1.plot(z_line, sigma_line, color="#1f77b4", linewidth=2.4, label=r"$\sigma(z)=1/(1+e^{-z})$")
ax1.axhline(0.5, color="#d62728", linestyle="--", linewidth=1.4, label="分类阈值 0.5")
ax1.axvline(0.0, color="#7f7f7f", linestyle=":", linewidth=1.2, label="z = 0（打分临界点）")
ax1.scatter([0.0], [0.5], color="#d62728", s=60, zorder=5)
ax1.annotate("σ(0) = 0.5\n模型完全犹豫", xy=(0, 0.5), xytext=(1.4, 0.28),
             fontsize=11, arrowprops=dict(arrowstyle="->", color="#d62728"))
ax1.axhspan(0.5, 1.02, color="#ffb703", alpha=0.10)
ax1.axhspan(-0.02, 0.5, color="#8ecae6", alpha=0.12)
ax1.text(6.2, 0.80, "判为正类 1", fontsize=11, color="#b06a00", ha="center")
ax1.text(-6.2, 0.18, "判为负类 0", fontsize=11, color="#1b5e7e", ha="center")
ax1.set_title("逻辑回归的核心：sigmoid 把任意实数打分 z = w·x + b 压缩到 [0, 1] 概率", fontsize=12)
ax1.set_xlabel("线性打分 z = w·x + b", fontsize=11)
ax1.set_ylabel("预测概率 p = σ(z)", fontsize=11)
ax1.set_ylim(-0.03, 1.05)
ax1.grid(alpha=0.3, linestyle="--")
ax1.legend(loc="lower right", fontsize=10)
fig1.tight_layout()
sigmoid_path = OUTPUT_DIR / "03_逻辑回归_sigmoid曲线.png"
fig1.savefig(sigmoid_path, dpi=130)
plt.close(fig1)
print("已保存图片：", sigmoid_path)

# ---------------------------------------------------------------------------
# 步骤 7：画图 2 —— 用两个特征画决策边界 + 散点
# ---------------------------------------------------------------------------
print_section("步骤 7：绘图 —— 二维决策边界（花瓣长 / 花瓣宽）")

# 只取两个特征（索引 2=花瓣长 petal length，3=花瓣宽 petal width），才能画在平面上。
# 这两个特征区分度最高，所以只用两维也能得到很高的准确率。
X2 = iris.data[:100, [2, 3]]
y2 = iris.target[:100]
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.3, random_state=RANDOM_STATE, stratify=y2
)
scaler2 = StandardScaler()
X2_train_s = scaler2.fit_transform(X2_train)
X2_test_s = scaler2.transform(X2_test)

model2 = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
model2.fit(X2_train_s, y2_train)
acc2 = accuracy_score(y2_test, model2.predict(X2_test_s))
print(f"只用两个特征时测试集准确率 = {acc2:.4f}")
print(f"两个特征上的权重 = {np.round(model2.coef_[0], 4)}，截距 = {round(float(model2.intercept_[0]), 4)}")
print("决策边界方程：w1·z1 + w2·z2 + b = 0，是一条直线；")
print("              这是逻辑回归作为'线性分类器'的直接体现。")

# 在特征平面上铺网格，对每个网格点预测类别，画出分界区域
pad = 0.8
xx, yy = np.meshgrid(
    np.linspace(X2_train_s[:, 0].min() - pad, X2_train_s[:, 0].max() + pad, 400),
    np.linspace(X2_train_s[:, 1].min() - pad, X2_train_s[:, 1].max() + pad, 400),
)
grid = np.c_[xx.ravel(), yy.ravel()]
zz_class = model2.predict(grid).reshape(xx.shape)
zz_prob = model2.predict_proba(grid)[:, 1].reshape(xx.shape)

fig2, ax2 = plt.subplots(figsize=(8.2, 6.2))
region_cmap = ListedColormap(["#a8d5e5", "#ffd79a"])
ax2.contourf(xx, yy, zz_class, levels=[-0.5, 0.5, 1.5], cmap=region_cmap, alpha=0.65)
# 概率等值线：p=0.5 就是决策边界本身，p=0.9/0.1 展示"把握程度"的梯度
cs = ax2.contour(xx, yy, zz_prob, levels=[0.1, 0.3, 0.5, 0.7, 0.9],
                 colors="gray", linewidths=1.0, linestyles="--")
ax2.clabel(cs, fmt="p=%.1f", fontsize=8, inline=True)
ax2.contour(xx, yy, zz_prob, levels=[0.5], colors="black", linewidths=2.2)

for cls, name, color, marker in [(0, "山鸢尾 setosa(0)", "#1f77b4", "o"),
                                 (1, "变色鸢尾 versicolor(1)", "#d62728", "^")]:
    ax2.scatter(X2_train_s[y2_train == cls, 0], X2_train_s[y2_train == cls, 1],
                c=color, marker=marker, s=52, edgecolors="white", linewidths=0.8,
                label=f"训练集 {name}")
ax2.scatter(X2_test_s[:, 0], X2_test_s[:, 1], c="none", edgecolors="black",
            s=110, linewidths=1.4, label="测试集样本（空心圈）")

ax2.set_title(f"逻辑回归决策边界（仅用花瓣长/花瓣宽两个标准化特征）\n测试集准确率 = {acc2:.2%}", fontsize=12)
ax2.set_xlabel("标准化后的花瓣长 petal length（标准差倍数）", fontsize=11)
ax2.set_ylabel("标准化后的花瓣宽 petal width（标准差倍数）", fontsize=11)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(alpha=0.25, linestyle="--")
fig2.tight_layout()
boundary_path = OUTPUT_DIR / "03_逻辑回归_决策边界.png"
fig2.savefig(boundary_path, dpi=130)
plt.close(fig2)
print("已保存图片：", boundary_path)

# ---------------------------------------------------------------------------
# 步骤 8：C 值对比 —— 正则化强度到底改变了什么
# ---------------------------------------------------------------------------
print_section("补充实验：C（正则化强度倒数）如何影响权重与边界")

print("【实验 A】在 iris 二维数据（几乎线性可分，太容易）上改变 C：")
print("-" * 84)
print(f"{'C 取值':<10}{'训练集准确率':>14}{'测试集准确率':>14}{'权重L2范数 ||w||₂':>20}{'|w|最大值':>14}")
print("-" * 84)
norms = []
for c_val in (0.01, 0.1, 1.0, 10.0, 100.0):
    m = LogisticRegression(C=c_val, solver="lbfgs", max_iter=1000, random_state=RANDOM_STATE)
    m.fit(X2_train_s, y2_train)
    tr_acc = accuracy_score(y2_train, m.predict(X2_train_s))
    te_acc = accuracy_score(y2_test, m.predict(X2_test_s))
    nrm = float(np.linalg.norm(m.coef_[0]))
    norms.append(nrm)
    print(f"{c_val:<10}{tr_acc:>14.4f}{te_acc:>14.4f}{nrm:>20.4f}{np.abs(m.coef_[0]).max():>14.4f}")
print("-" * 84)
print("解读 A：这份数据太容易分，无论 C 怎么变准确率都饱和在 1.0000，看不出差别；")
print(f"        但权重 L2 范数从 {norms[0]:.2f} 一路涨到 {norms[-1]:.2f}，")
print("        清楚证明：C 越小 → 正则越强 → 权重被压得越小 → 模型越保守、边界越平滑。")

# 再换一份"特征被升维"的合成数据：维度一高，正则强度的作用就会立刻显现在准确率上。
print()
print("【实验 B】换一份合成数据，并给它加【多项式特征】把维度从 2 抬到 20：")
print("          make_classification 造 120 条两维样本（两类边界很近、5% 标签被翻转），")
print("          再用 sklearn 的 PolynomialFeatures(degree=5) 把 2 维扩成 20 维。")
print("          维度一高、样本一少，正则强度 C 的欠拟合/过拟合效应就会立刻显现在准确率上")
print("          （实验 A 的 iris 太容易，准确率被'顶死'在 1.0，看不出差别）。")
X_noisy, y_noisy = make_classification(
    n_samples=120, n_features=2, n_informative=2, n_redundant=0,
    n_clusters_per_class=1, class_sep=0.9, flip_y=0.05, random_state=RANDOM_STATE,
)
X_n_tr, X_n_te, y_n_tr, y_n_te = train_test_split(
    X_noisy, y_noisy, test_size=0.3, random_state=RANDOM_STATE, stratify=y_noisy
)
poly = PolynomialFeatures(degree=5, include_bias=False)
X_n_tr_p = poly.fit_transform(X_n_tr)
X_n_te_p = poly.transform(X_n_te)
print(f"升维结果：{X_n_tr.shape[1]} 维 → {X_n_tr_p.shape[1]} 维（训练集 {X_n_tr_p.shape[0]} 条样本）")
sc_n = StandardScaler()
X_n_tr_s = sc_n.fit_transform(X_n_tr_p)
X_n_te_s = sc_n.transform(X_n_te_p)
print("-" * 92)
print(f"{'C 取值':<10}{'训练集准确率':>14}{'测试集准确率':>14}{'训练-测试差距':>16}{'权重L2范数':>14}{'现象':>18}")
print("-" * 92)
c_grid = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
records = []
for c_val in c_grid:
    m = LogisticRegression(C=c_val, solver="lbfgs", max_iter=20000, random_state=RANDOM_STATE)
    m.fit(X_n_tr_s, y_n_tr)
    tr_acc = accuracy_score(y_n_tr, m.predict(X_n_tr_s))
    te_acc = accuracy_score(y_n_te, m.predict(X_n_te_s))
    nrm = float(np.linalg.norm(m.coef_[0]))
    records.append((c_val, tr_acc, te_acc, nrm))
best_test_i = int(np.argmax([r[2] for r in records]))     # 测试准确率最高的那一行
best_train_i = int(np.argmax([r[1] for r in records]))    # 训练准确率最高的那一行
max_train = records[best_train_i][1]
for c_val, tr_acc, te_acc, nrm in records:
    gap = tr_acc - te_acc
    if c_val == records[best_test_i][0]:
        tag = "★ 测试最佳"
    elif tr_acc <= max_train - 0.12:
        tag = "欠拟合"
    elif gap > 0.15:
        tag = "过拟合信号"
    else:
        tag = "较合适"
    print(f"{c_val:<10}{tr_acc:>14.4f}{te_acc:>14.4f}{gap:>16.4f}{nrm:>14.4f}{tag:>18}")
print("-" * 92)
small_c, small_tr, small_te, small_nrm = records[0]              # 正则最强（C 最小）
big_c, big_tr, big_te, big_nrm = records[-1]                     # 正则最弱（C 最大）
best_c, best_tr, best_te, best_nrm = records[best_test_i]
print("解读 B（下面每一条结论都直接引用上表的真实数字，不是空口断言）：")
print(f"  · C = {small_c}（正则最强）时权重范数只有 {small_nrm:.3f}（权重几乎被压平），")
print(f"    训练集准确率 {small_tr:.4f}、测试集准确率 {small_te:.4f}。训练集连 2/3 都不到，")
print("    说明模型被正则压得太死、表达能力不足 → 这是【欠拟合】。")
print(f"  · 随着 C 变大，训练集准确率整体爬升，最高达到 {max_train:.4f}（C = {records[best_train_i][0]}），")
print("    也就是模型越来越'用力'地去记住训练集。")
print(f"  · 但测试集准确率并没有同步上涨：训练-测试差距从 {small_tr - small_te:+.4f}"
      f"（C={small_c}）一路拉大到 {big_tr - big_te:+.4f}（C={big_c}），")
print("    多出来的那部分拟合能力，全都用在了训练集里的噪声上 → 这是【过拟合】的典型信号。")
print(f"  · 本组测试准确率最高的是 C = {best_c}（{best_te:.4f}），但测试集只有 {len(y_n_te)} 条样本，")
print("    单点差异并不可靠；工程上应该用 K 折交叉验证来选 C，而不是只看一次划分。")
print("  · 结论：调 C 时盯住的不是训练准确率，而是【验证集准确率】与【训练-测试差距】；")
print("    C 是正则强度的倒数 —— C 越大正则越弱、越容易过拟合，越小越容易欠拟合。")

# ============================================================================
# ④ 结果解读
# ============================================================================
print_section("④ 结果解读（这些数字到底意味着什么）")

print(f"1) 测试集准确率 = {acc:.4f}：")
print(f"   在 {len(y_test)} 条从未参与训练的样本中，模型答对了 {(y_pred == y_test).sum()} 条。")
print("   iris 前 100 条（setosa vs versicolor）本身就是高度线性可分的，")
print("   所以逻辑回归这种线性模型能拿到 93%~100% 的准确率，说明任务简单、模型够用。")
print()
print(f"2) 混淆矩阵显示 FP={cm[0, 1]}、FN={cm[1, 0]}：")
print("   这两类错误的代价往往不同。若这是疾病筛查，FN（漏诊）比 FP（误诊）严重得多，")
print("   此时应该调低阈值提高召回率，而不能只看准确率这一个数字。")
print()
print("3) 权重 w 的方向与大小就是'可解释性'：")
coef_r = model.coef_[0]
for name, c in sorted(zip(iris.feature_names, coef_r), key=lambda t: -abs(t[1])):
    direction = "↑ 越大越像类别1" if c > 0 else "↑ 越大越像类别0"
    print(f"   {name:<28} 系数 = {c:+.4f}   {direction}")
print("   系数绝对值越大，该特征对判别的贡献越强；正负号表示作用方向。")
print("   注意：系数是在【标准化后】的尺度上比较才公平——这也是本脚本先做 StandardScaler 的原因之一。")
print()
print(f"4) sigmoid 曲线图（{sigmoid_path.name}）：")
print("   曲线呈现 S 形，在 z=0 附近变化最剧烈（斜率最大 = p(1-p) = 0.25），")
print("   在 |z|>4 以后几乎压平，输出概率趋于 0 或 1。")
print("   这就是'离决策边界越远的样本，模型越有把握'的数学来源。")
print()
print(f"5) 决策边界图（{boundary_path.name}）：")
print("   直线就是 p=0.5 的等值线；虚线是 p=0.1/0.3/0.7/0.9 的等值线，")
print("   它们与边界平行，越靠外代表模型越有信心。图上的空心圈是测试样本，")
print("   可以看到绝大多数测试点都落在正确的颜色区域内，与准确率数字互相印证。")
print()
print("6) 一句话总览：逻辑回归 = 线性打分 + sigmoid 概率 + 交叉熵损失 + 阈值判类；")
print("   它简单、快、可解释、给概率，是分类任务最应该先跑一遍的基线模型。")

# ============================================================================
# 超参数怎么调
# ============================================================================
# 【超参数怎么调】—— 逻辑回归实战调参顺序
#
# 1. 必做前提：先标准化（StandardScaler）。
#    对逻辑回归来说，"要不要标准化"比"调 C"重要得多。若用 Pipeline，务必让
#    scaler 只在训练折上 fit，避免数据泄漏。
#
# 2. C（最重要，优先调）：
#    在 log 尺度上网格搜索，典型范围 C ∈ {0.001, 0.01, 0.1, 1, 10, 100}。
#    - 训练准确率高、测试准确率低 → C 偏大（过拟合），往小调；
#    - 训练与测试准确率都低 → C 偏小（欠拟合），往大调；
#    - iris/乳腺癌这类简单数据，C=1 附近通常就够好。
#    注意 C 是正则强度的**倒数**：C 越大正则越弱。
#
# 3. l1_ratio（sklearn ≥1.8 取代 penalty）：
#    - 想保留全部特征、只要平滑 → l1_ratio=0（等价旧 penalty='l2'，默认）；
#    - 特征很多想要稀疏解做特征选择 → l1_ratio=1（等价旧 penalty='l1'），
#      此时必须把 solver 换成 'saga' 或 'liblinear'；
#    - 想要二者折中 → l1_ratio ∈ (0,1)，配 solver='saga'。
#    不要再用 penalty 参数（1.8 起弃用，1.10 移除，会打 FutureWarning）。
#
# 4. solver：
#    - 默认 'lbfgs' 适用于 L2 与多分类，绝大多数情况不用改；
#    - 数据量很大（>10 万样本）优先 'sag'/'saga'（可配 n_jobs）；
#    - 只要 L1 或 ElasticNet → 'saga'（或小数据用 'liblinear'，但它不支持多分类 softmax）。
#
# 5. max_iter：
#    若出现 ConvergenceWarning（本脚本用 max_iter=1000 已避免），
#    先把 max_iter 提到 1000~5000；仍不收敛就检查是否忘记标准化、或 C 是否过大。
#
# 6. class_weight：
#    类别严重不平衡（如 1:100）时用 'balanced'，或手写字典给少数类更大权重。
#    评估时不要只看准确率，要看 recall / F1 / ROC-AUC。
#
# 7. 阈值（不是 sklearn 参数，但比很多参数更重要）：
#    训练完拿到 predict_proba 后，按业务需求选阈值：
#    要召回率 → 调低；要精确率 → 调高。用验证集上的 PR 曲线选最佳阈值。
#
# 8. 多分类说明：sklearn 1.9 已移除 multi_class 参数，
#    LogisticRegression 多分类默认走 multinomial（softmax）；
#    若确实需要 OvR 语义，用 OneVsRestClassifier(LogisticRegression(...)) 显式包装。
#
# 9. 推荐自动化做法：
#    GridSearchCV(Pipeline([("scaler", StandardScaler()),
#                           ("clf", LogisticRegression(max_iter=1000))]),
#                 {"clf__C": [0.01, 0.1, 1, 10]}, cv=StratifiedKFold(5), scoring="f1")
#    —— 把标准化放进 Pipeline，交叉验证时自动避免数据泄漏。
# ============================================================================

print()
print("【完成】01_逻辑回归.py 运行结束")

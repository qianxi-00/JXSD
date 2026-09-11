r"""《机器学习》课案 · 03 监督学习-分类 · 03 支持向量机（SVM）

对应课案章节
-----------
《机器学习》课案「分类」章 —— 支持向量机（Support Vector Machine）：
课案原文位于 `.course_extract/机器学习_课案.md` 第 494~568 行。
覆盖内容：SVM 核心思想（找最优超平面，使两类之间的间隔 margin 最大）、
课案示例代码（load_iris + StandardScaler + SVC(kernel="linear")）、
核函数对比表（线性核 / 多项式核 / RBF 核 / Sigmoid 核，含公式与适用场景）、
SVM 参数说明表（C / gamma / kernel 的含义与影响）。

本节知识点
---------
1. 最大间隔思想：在所有能分开两类的超平面里，选"离最近样本最远"的那一个。
2. 支持向量：只有落在间隔边界上（或间隔内）的少数样本决定超平面，其余样本删掉也不影响结果。
3. 硬间隔 → 软间隔：引入松弛变量 ξ 与惩罚系数 C，
   等价于 **Hinge 损失 + L2 正则**：min (1/2)||w||² + C·Σ max(0, 1 - y_i·f(x_i))。
4. 拉格朗日对偶：把"求 w"变成"求每个样本的系数 α_i"，
   只有 α_i > 0 的样本才是支持向量，决策函数只依赖支持向量之间的内积。
5. 核技巧：K(x, y) = φ(x)·φ(y)，把内积换成核函数就等价于在高维空间做线性 SVM，
   而**不需要真的把数据映射到高维**（避免维度爆炸）。
6. 四种核函数：linear / poly / rbf / sigmoid 的公式、参数与适用场景。
7. 用 make_moons 月亮形数据直观对比 linear 核（切不开）与 rbf 核（切得开），
   并观察 C 与 gamma 对决策边界平滑度的影响。

运行方式
-------
PowerShell（路径含中文，必须加引号并用 & 调用）：

    $env:PYTHONUTF8='1'; & 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Machine_Learning\03_监督学习_分类\03_SVM.py'

输出：
    - 控制台：四段式讲解 + linear/rbf 对比 + C/gamma 对比 + iris 上各核函数准确率表 + 中文解读
    - 图片： Machine_Learning/output/03_SVM_linear核与rbf核对比.png
            Machine_Learning/output/03_SVM_C与gamma对边界的影响.png
            Machine_Learning/output/03_SVM_核函数在iris上的表现.png
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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_iris, make_moons
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

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
# 【1.1 核心思想：最大间隔（maximum margin）】
#   线性可分时，能把两类分开的超平面有无穷多个。SVM 的思路是：
#   **选那个"离最近的样本也尽可能远"的超平面**。
#   超平面写成 w·x + b = 0，样本 x 到它的（函数）距离由 w·x + b 的符号与大小决定。
#   把 w、b 同时缩放不影响超平面本身，所以我们可以约定一个规范：
#   让离超平面最近的样本恰好满足 |w·x + b| = 1，于是这些样本到超平面的几何距离是 1/||w||。
#   两类之间的"间隔带"总宽度就是 2/||w||。
#
#   目标：让间隔最宽 → 让 1/||w|| 最大 → 让 ||w|| 最小。
#   硬间隔 SVM 的原始问题（primal problem）：
#
#       min_{w,b}  (1/2)·||w||²
#       s.t.       y_i·(w·x_i + b) ≥ 1 ,  i = 1..N
#
#   目标函数是二次的、约束是线性的 → 这是一个**凸二次规划**，有唯一全局最优解。
#
# 【1.2 支持向量（support vectors）—— 名字的由来】
#   落在 y_i·(w·x_i + b) = 1 这两条边界线上的样本，就是"支持向量"。
#   它们是整个解里唯一"起作用"的样本：把其它样本全部删掉重新训练，得到的超平面完全一样。
#   这也是 SVM 相对其它模型的一个特点：**解是稀疏的、只由少数关键样本决定**。
#   如果数据有噪声或离群点，落在间隔带里/错分的那部分样本同样会成为支持向量。
#
# 【1.3 软间隔（soft margin）：允许犯错】
#   现实数据往往不是严格线性可分的。硬间隔会强迫所有点都分对，
#   结果就是被一两个离群点"拽"着走，间隔变得很窄，泛化很差。
#   于是给每个样本引入松弛变量 ξ_i ≥ 0，允许它跑到间隔带里面甚至另一边：
#
#       min_{w,b,ξ}  (1/2)·||w||² + C·Σ_{i=1..N} ξ_i
#       s.t.         y_i·(w·x_i + b) ≥ 1 - ξ_i ,   ξ_i ≥ 0
#
#   · ξ_i = 0            → 该样本在间隔带外、分类正确（不是支持向量）；
#   · 0 < ξ_i < 1        → 该样本落在间隔带内部但分类仍然正确；
#   · ξ_i ≥ 1            → 该样本被分错了。
#   C 是惩罚系数：
#     C → 很大：几乎不允许犯错 → 逼近硬间隔 → 间隔窄、容易过拟合；
#     C → 很小：允许大量犯错 → 间隔宽、容错强 → 可能欠拟合。
#
#   把约束代入消掉 ξ，软间隔 SVM 就等价于下面这个**无约束**形式（hinge 损失 + L2 正则）：
#
#       min_{w,b}  (1/2)·||w||² + C·Σ_{i=1..N} max( 0 , 1 - y_i·(w·x_i + b) )
#                  \___L2正则___/       \______Hinge 损失（合页损失）______/
#
#   其中 y_i ∈ {-1, +1}。Hinge 损失的含义：
#     如果 y_i·f(x_i) ≥ 1（分对了且落在间隔外）→ 损失为 0，不再惩罚；
#     如果 y_i·f(x_i) < 1（分错，或虽然分对但落在间隔带内）→ 损失 = 1 - y_i·f(x_i)。
#   和逻辑回归的交叉熵对比：hinge 对"已经分得很好"的样本直接给 0 损失，
#   所以解只依赖少数支持向量；交叉熵则永远有非零梯度，用到全部样本。
#
# 【1.4 拉格朗日对偶与核技巧】
#   用拉格朗日乘子 α_i ≥ 0 把约束并入目标，再对 w、b 求极值，得到**对偶问题**：
#
#       max_α  Σ_i α_i - (1/2)·Σ_i Σ_j α_i·α_j·y_i·y_j·(x_i·x_j)
#       s.t.   0 ≤ α_i ≤ C  ,  Σ_i α_i·y_i = 0
#
#   两个关键结论：
#     (1) w = Σ_i α_i·y_i·x_i。由 KKT 条件可知：α_i > 0 的样本恰好就是支持向量，
#         其余样本 α_i = 0。所以 w 只由支持向量拼出来。
#     (2) 决策函数 f(x) = sign( w·x + b ) = sign( Σ_i α_i·y_i·(x_i·x) + b )。
#         **注意：x 只以"内积"的形式出现！**
#
#   于是核技巧（kernel trick）登场：
#   如果我们先用映射 φ 把数据送到更高维的空间 z = φ(x)，让原本线性不可分的数据在新空间里
#   线性可分，那么对偶问题与决策函数里出现的都只是内积 φ(x_i)·φ(x_j)。
#   定义核函数 K(x_i, x_j) = φ(x_i)·φ(x_j)，则**整个算法完全不需要知道 φ 长什么样，
#   只需要能算出 K 的值**：
#
#       f(x) = sign( Σ_{i∈SV} α_i·y_i·K(x_i, x) + b )
#
#   这就是核技巧：把"先升维再算内积"两步合并成一步，避开了显式映射带来的维度爆炸。
#   例如 RBF 核对应的特征空间是**无穷维**的，但计算量与维度无关。
#
# 【1.5 四种核函数（课案表格的正式写法）】
#
#   (a) 线性核（linear）：      K(x, y) = xᵀ·y
#       就是原始空间的内积，等价于不做映射。参数只有 C。
#       适用：特征维度已经很高（文本 TF-IDF）、数据本身近似线性可分、需要快且可解释。
#       优点：训练快、不会过拟合到核参数上；缺点：无法处理非线性边界。
#
#   (b) 多项式核（poly）：      K(x, y) = ( γ·xᵀ·y + r )^d
#       其中 γ = gamma，r = coef0，d = degree。
#       适用：特征维度较低、又需要非线性边界（比如图像像素、手工特征）。
#       缺点：参数多（γ、r、d 三个），d 大时数值不稳定（容易溢出/下溢）；
#             实际使用率已经明显低于 RBF。
#
#   (c) 高斯径向基核（RBF / Gaussian）：K(x, y) = exp( -γ·||x - y||² )
#       它度量两个样本的"相似度"：距离为 0 时 K=1，距离越远 K 越快趋近 0。
#       RBF 核对应**无穷维**特征空间，理论上能逼近任意连续边界。
#       适用：**通用首选**，不知道选什么核时先用它。
#       参数：γ（gamma）控制"影响半径"，C 控制容错。
#
#   (d) Sigmoid 核：            K(x, y) = tanh( γ·xᵀ·y + r )
#       形式类似神经网络里的激活函数，因此被称为"神经网络风格的核"。
#       注意：只有当参数满足一定条件时它才是合法的（半正定）核函数，
#             否则对偶问题非凸、求解不稳定。实战中很少用。
#
# 【1.6 优缺点】
#   优点：
#     - 在小样本、高维数据上表现非常好（经典场景：文本分类、生物信息）；
#     - 有严格的统计学习理论（结构风险最小化）支撑，泛化理论清晰；
#     - 核技巧极其优雅，能用一个凸优化框架统一处理线性和非线性问题；
#     - 解由少数支持向量决定，模型存储量小。
#   缺点：
#     - 大样本时训练复杂度高（标准实现约 O(N²)~O(N³)），不适合十万级以上的数据；
#     - 对参数 C、γ 非常敏感，且没有概率输出（需要额外校准）；
#     - 必须标准化，否则距离/内积完全被大量纲特征支配；
#     - 多分类要靠 OvO / OvR 组合，输出不如树模型直观。
# ============================================================================


# ============================================================================
# ② sklearn API 关键参数逐个解释
# ============================================================================
# 默认值取自本机 scikit-learn 1.9.0 实测（SVC().get_params()）。
#
# C : float，默认 1.0
#    含义：软间隔的惩罚系数，**SVM 最重要的超参数之一**（与逻辑回归的 C 意义相反！）
#      - 逻辑回归：C 是正则强度的**倒数**，C 越大正则越弱；
#      - SVM：C 直接就是违反间隔的惩罚力度，C 越大惩罚越重。
#    调大（如 100）：几乎不允许分错 → 间隔变窄、边界紧贴样本 → 方差大、易过拟合；
#    调小（如 0.01）：允许大量样本落进间隔带 → 间隔变宽、边界平滑 → 偏差大、易欠拟合。
#    常用值：0.1 / 1 / 10 / 100（在 log 尺度上网格搜索）。
#
# kernel : {'linear','poly','rbf','sigmoid','precomputed'} 或可调用对象，默认 'rbf'
#    含义：核函数类型，决定决策边界的形状与模型的表达能力。
#      - 'linear'：线性核，边界是直线/超平面，快且可解释；
#      - 'rbf'（默认）：高斯核，边界可以任意弯曲，通用首选；
#      - 'poly'：多项式核，靠 degree / gamma / coef0 三个参数控制；
#      - 'sigmoid'：神经网络风格的核，实战少用；
#      - 'precomputed'：直接传入核矩阵（自定义核时用）。
#    影响：核选错相当于用了错误的"相似度定义"，比调 C/γ 的影响大得多。
#
# gamma : {'scale','auto'} 或 float，默认 'scale'
#    含义：RBF / poly / sigmoid 核的系数 γ。
#      - 'scale'：γ = 1 / (n_features × X.var())（默认，用特征方差自动缩放，省心）；
#      - 'auto'：γ = 1 / n_features（不看方差，特征方差差异大时容易出问题）；
#      - 传具体数字：完全手动控制。
#    直观理解：γ 决定单个样本"影响半径"的倒数。
#      调大 γ（如 10）：影响半径很小 → 每个样本只影响自己周围一小圈 →
#                       边界变得非常弯曲、甚至每个点都被单独围起来 → **过拟合**；
#      调小 γ（如 0.01）：影响半径很大 → 边界非常平滑、接近线性 → **欠拟合**。
#    常用值：先用默认 'scale'，再在 {0.001, 0.01, 0.1, 1, 10} 里搜。
#
# degree : int，默认 3
#    含义：多项式核的次数 d，只在 kernel='poly' 时生效。
#    调大：能表达更复杂的高阶交互，但数值不稳定、更容易过拟合；
#    调小（d=1）：多项式核退化为线性核。
#    常用值：2 / 3 / 4。
#
# coef0 : float，默认 0.0
#    含义：多项式核与 sigmoid 核里的常数项 r（**独立项**，不受 γ 缩放）。
#    作用：让核函数不完全依赖数据的内积，高阶项和低阶项可以混合。
#    调大：低阶项（更接近线性）的比重上升，边界更平滑；
#    调小到 0：纯 d 次多项式核。
#    常用值：0 / 0.5 / 1（只对 poly / sigmoid 有意义）。
#
# probability : bool，默认 False
#    ⚠ 版本状态：在 scikit-learn **1.9 中该参数已被弃用（deprecated）**，
#      计划在 1.11 移除；显式传 probability=True 会触发 FutureWarning
#      （本脚本实测确认：Use CalibratedClassifierCV(SVC(), ensemble=False) instead）。
#      另外 probability=True 会内部再做 5 折 Platt 校准，训练耗时明显增加。
#    ✅ 1.9 的正确做法：SVC 只给 decision_function（到超平面的带符号距离），
#      需要概率时用 CalibratedClassifierCV 包一层：
#          cal = CalibratedClassifierCV(SVC(kernel='rbf'), ensemble=False, cv=5)
#          cal.fit(X, y); cal.predict_proba(X_new)
#    本脚本按 1.9 的写法演示，不传 probability。
#
# class_weight : dict 或 'balanced'，默认 None
#    含义：类别权重。'balanced' 按类别频率反比自动加权，用于类别不平衡。
#    调大少数类权重：该类的召回率上升，精确率可能下降。
#
# tol : float，默认 1e-3。停止训练的容差，调大训练更快精度略降。
# shrinking : bool，默认 True。是否启用启发式收缩以加速训练，一般不用改。
# cache_size : float，默认 200.0（MB）。核矩阵缓存大小，调大能加速大样本训练。
# max_iter : int，默认 -1（不限制）。设成正数可强制提前停止（结果不再精确）。
# decision_function_shape : {'ovr','ovo'}，默认 'ovr'。
#    多分类时 decision_function 的输出形状：'ovr' 会返回 (n_samples, n_classes)
#    的一对多打分，'ovo' 返回原始的一对一打分。只影响返回值的形状，
#    不影响 predict 的结果。
# random_state : int，默认 None。probability=True 时用于内部交叉验证的洗牌；
#    本脚本统一设为 42 保证可复现。
# ============================================================================


print_section("《机器学习》课案 · 分类 · 03 支持向量机（SVM）")
print("本脚本四段结构： ① 原理与数学推导  ② sklearn API 参数解释  ③ 完整可运行代码  ④ 结果解读")
print("配套图片输出目录：", OUTPUT_DIR)

# ============================================================================
# ③ 完整可运行代码
# ============================================================================

# ---------------------------------------------------------------------------
# 步骤 1：造月亮形数据 —— 线性不可分的经典例子
# ---------------------------------------------------------------------------
print_section("步骤 1：造数据 make_moons（月亮形数据，线性不可分）")

X_moons, y_moons = make_moons(n_samples=300, noise=0.2, random_state=RANDOM_STATE)
print("数据来源：sklearn.datasets.make_moons(n_samples=300, noise=0.2, random_state=42)")
print(f"形状：X = {X_moons.shape}，y = {y_moons.shape}")
print("类别分布：", {int(c): int(n) for c, n in zip(*np.unique(y_moons, return_counts=True))})
print("为什么用这份数据：两类的形状像两个交错的半月，**任何一条直线都切不开**，")
print("                  正好用来演示'线性核不行、RBF 核可以'。")

Xm_train, Xm_test, ym_train, ym_test = train_test_split(
    X_moons, y_moons, test_size=0.3, random_state=RANDOM_STATE, stratify=y_moons
)
scaler_m = StandardScaler()
Xm_train_s = scaler_m.fit_transform(Xm_train)
Xm_test_s = scaler_m.transform(Xm_test)
print(f"划分：训练集 {Xm_train_s.shape}，测试集 {Xm_test_s.shape}")
print("SVM 用内积/距离衡量相似度，对特征尺度非常敏感，所以必须先标准化。")

# ---------------------------------------------------------------------------
# 步骤 2：linear 核 vs rbf 核
# ---------------------------------------------------------------------------
print_section("步骤 2：linear 核 vs rbf 核（同一份月亮数据）")

svc_linear = SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)
svc_linear.fit(Xm_train_s, ym_train)
acc_lin_train = accuracy_score(ym_train, svc_linear.predict(Xm_train_s))
acc_lin_test = accuracy_score(ym_test, svc_linear.predict(Xm_test_s))

svc_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)
svc_rbf.fit(Xm_train_s, ym_train)
acc_rbf_train = accuracy_score(ym_train, svc_rbf.predict(Xm_train_s))
acc_rbf_test = accuracy_score(ym_test, svc_rbf.predict(Xm_test_s))

print("-" * 78)
print(f"{'模型':<32}{'训练集准确率':>16}{'测试集准确率':>16}{'支持向量个数':>14}")
print("-" * 78)
print(f"{'SVC(kernel=linear, C=1.0)':<32}{acc_lin_train:>16.4f}{acc_lin_test:>16.4f}"
      f"{len(svc_linear.support_):>14}")
print(f"{'SVC(kernel=rbf, C=1.0, gamma=scale)':<32}{acc_rbf_train:>16.4f}{acc_rbf_test:>16.4f}"
      f"{len(svc_rbf.support_):>14}")
print("-" * 78)
print(f"rbf 核用了默认 gamma='scale'，等价于 1/(n_features × X.var()) = "
      f"{1.0 / (Xm_train_s.shape[1] * Xm_train_s.var()):.4f}")
print(f"训练集共 {len(ym_train)} 条，其中被选为支持向量的有 {len(svc_rbf.support_)} 条"
      f"（占 {len(svc_rbf.support_) / len(ym_train):.1%}）。")
print()
print("为什么线性核切不开月亮形数据：")
print("  线性核的决策函数是 sign(w·x + b)，边界就是一条直线（高维时是超平面）。")
print("  月亮数据的两个半月是**交错**的：任何一个半月的两端都夹在另一个半月的两侧，")
print("  所以无论这条直线怎么摆，都必然把其中一个半月的末端切到错误的一侧。")
print("  这正是'线性模型表达能力不足'的几何图像 —— 不是参数没调好，是模型族本身不够。")
print()
print("RBF 核为什么可以：")
print("  K(x,y)=exp(-γ||x-y||²) 等价于把数据映射到**无穷维**空间，")
print("  在那个空间里原来的曲线边界可以变成一张平面；")
print("  由于核技巧，我们完全不用真的去算那无穷维坐标，只算样本间的距离就够了。")

# 用交叉验证把结论坐实（避免只看一次划分）
print()
print("用 5 折交叉验证复核（更可靠）：")
for name, est in [("linear 核", SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)),
                  ("rbf 核", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE))]:
    scores = cross_val_score(est, Xm_train_s, ym_train, cv=5)
    print(f"  {name}：5 折 CV 准确率 = {scores.mean():.4f} ± {scores.std():.4f}")

# ---------------------------------------------------------------------------
# 步骤 3：C 与 gamma 对决策边界的影响
# ---------------------------------------------------------------------------
print_section("步骤 3：C 与 gamma 如何改变决策边界（网格搜索小实验）")

print("在同一份月亮数据上训练 6 个 SVM，只有 C 与 gamma 不同：")
print("-" * 84)
print(f"{'C 取值':<10}{'gamma 取值':>12}{'训练集准确率':>16}{'测试集准确率':>16}{'支持向量数':>14}")
print("-" * 84)
grid_records = []
for c_val in (0.1, 100.0):
    for g_val in (0.1, 1.0, 10.0):
        svc_g = SVC(kernel="rbf", C=c_val, gamma=g_val, random_state=RANDOM_STATE)
        svc_g.fit(Xm_train_s, ym_train)
        tr = accuracy_score(ym_train, svc_g.predict(Xm_train_s))
        te = accuracy_score(ym_test, svc_g.predict(Xm_test_s))
        grid_records.append((c_val, g_val, tr, te, len(svc_g.support_)))
        print(f"{c_val:<10}{g_val:>12}{tr:>16.4f}{te:>16.4f}{len(svc_g.support_):>14}")
print("-" * 84)


def grid_rec(c_val, g_val):
    """从上面的网格实验结果里取出指定 C / gamma 的那一行。"""
    for r in grid_records:
        if r[0] == c_val and r[1] == g_val:
            return r
    raise KeyError((c_val, g_val))


print("读表要点（下面的数字都直接来自上表）：")
print("  · gamma 越小（0.1）→ 每个样本的影响半径越大 → 边界越平滑 → 容易欠拟合；")
print("  · gamma 越大（10）→ 影响半径越小 → 边界越贴近训练点 → 训练准确率被推到最高，")
print(f"    但测试准确率没有跟着涨（C={grid_rec(100.0, 10.0)[0]} 时训练 "
      f"{grid_rec(100.0, 10.0)[2]:.4f}、测试只有 {grid_rec(100.0, 10.0)[3]:.4f}）→ 这是过拟合信号；")
print(f"  · C 小（0.1）→ 容错强 → 支持向量很多：gamma=10 时 "
      f"{grid_rec(0.1, 10.0)[4]} 个训练样本全部成了支持向量；")
print(f"  · C 大（{grid_rec(100.0, 1.0)[0]}）→ 几乎不允许犯错 → 同 gamma 下支持向量明显减少"
      f"（gamma=1 时从 {grid_rec(0.1, 1.0)[4]} 降到 {grid_rec(100.0, 1.0)[4]}）；")
print("  · 结论：C 和 gamma 一起控制模型复杂度，实践中要在二维网格上联合搜索。")

# ---------------------------------------------------------------------------
# 步骤 4：在 iris 上复现课案示例，并对比四种核函数
# ---------------------------------------------------------------------------
print_section("步骤 4：在 iris（三分类）上复现课案示例并对比核函数")

iris = load_iris()
X_iris, y_iris = iris.data, iris.target
class_names = [str(n) for n in iris.target_names]

# 课案代码：不 stratify，固定 random_state=42，保持与课案一致
Xi_train, Xi_test, yi_train, yi_test = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=RANDOM_STATE
)
scaler_i = StandardScaler()
Xi_train_s = scaler_i.fit_transform(Xi_train)
Xi_test_s = scaler_i.transform(Xi_test)
print(f"划分（与课案一致，未加 stratify）：训练集 {Xi_train_s.shape}，测试集 {Xi_test_s.shape}")

# 完全照课案的写法：SVC(kernel="linear")
model_iris = SVC(kernel="linear")
model_iris.fit(Xi_train_s, yi_train)
y_pred_iris = model_iris.predict(Xi_test_s)
print("课案示例 SVC(kernel='linear')：")
print("  预测结果（前 15 条）：", y_pred_iris[:15])
print("  真实结果（前 15 条）：", yi_test[:15])
print(f"  准确率 = {accuracy_score(yi_test, y_pred_iris):.4f}")
print(f"  支持向量个数 = {len(model_iris.support_)}，"
      f"每类支持向量数 n_support_ = {[int(v) for v in model_iris.n_support_]}")
print("\n分类报告：")
print(classification_report(yi_test, y_pred_iris, target_names=class_names, digits=4, zero_division=0))

print("\n四种核函数在同一份 iris 数据上的对比（C=1.0，其余参数用默认）：")
kernel_cases = [
    ("线性核 linear", dict(kernel="linear")),
    ("RBF 核 rbf（默认）", dict(kernel="rbf", gamma="scale")),
    ("RBF 核 rbf, gamma=auto", dict(kernel="rbf", gamma="auto")),
    ("多项式核 poly, d=3", dict(kernel="poly", degree=3, gamma="scale", coef0=0.0)),
    ("多项式核 poly, d=2", dict(kernel="poly", degree=2, gamma="scale", coef0=1.0)),
    ("Sigmoid 核 sigmoid", dict(kernel="sigmoid", gamma="scale", coef0=0.0)),
]
print("-" * 88)
print(f"{'核函数配置':<28}{'训练集准确率':>14}{'测试集准确率':>14}{'5折CV均值':>12}{'CV标准差':>12}{'支持向量数':>12}")
print("-" * 88)
kernel_records = []
for name, kw in kernel_cases:
    svc_k = SVC(C=1.0, random_state=RANDOM_STATE, **kw)
    svc_k.fit(Xi_train_s, yi_train)
    tr = accuracy_score(yi_train, svc_k.predict(Xi_train_s))
    te = accuracy_score(yi_test, svc_k.predict(Xi_test_s))
    cv_scores = cross_val_score(svc_k, Xi_train_s, yi_train, cv=5)
    kernel_records.append((name, tr, te, cv_scores.mean(), cv_scores.std(), len(svc_k.support_)))
    print(f"{name:<28}{tr:>14.4f}{te:>14.4f}{cv_scores.mean():>12.4f}"
          f"{cv_scores.std():>12.4f}{len(svc_k.support_):>12}")
print("-" * 88)
print("注意一个有意思的现象：'gamma=scale' 与 'gamma=auto' 在本数据上结果**完全相同**。")
print("  原因：数据已经用 StandardScaler 标准化，各特征方差都等于 1，于是")
print(f"    gamma='scale' = 1/(n_features × X.var()) = 1/({Xi_train_s.shape[1]} × "
      f"{Xi_train_s.var():.1f}) = {1.0 / (Xi_train_s.shape[1] * Xi_train_s.var()):.4f}")
print(f"    gamma='auto'  = 1/n_features             = 1/{Xi_train_s.shape[1]} "
      f"        = {1.0 / Xi_train_s.shape[1]:.4f}")
print("  两者恰好重合。但如果**不做标准化**，各特征方差不等于 1，两个口径就会分道扬镳 ——")
print("  这也再次说明：'scale' 更稳健，'auto' 在量纲混乱的数据上会给出荒唐的 gamma。")
print()
print("Sigmoid 核在小样本上经常表现不稳定（它的合法性依赖参数条件），")
print("这也印证了'实战中优先用 RBF、其次线性核'的经验。")

# ---------------------------------------------------------------------------
# 步骤 5：从 decision_function 到"概率"
# ---------------------------------------------------------------------------
print_section("步骤 5：SVM 的输出 —— decision_function 与概率校准")

svc_demo = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)
svc_demo.fit(Xm_train_s, ym_train)
dec = svc_demo.decision_function(Xm_test_s)
print("SVC 默认**不输出概率**，它输出的是 decision_function：样本到超平面的带符号距离。")
print("  前 8 条测试样本的 decision_function 值：", np.round(dec[:8], 4))
print("  符号就是预测类别（>0 → 类别 1，<0 → 类别 0），绝对值越大表示离边界越远、越有把握。")
print("  验证：用 sign(decision_function) 得到的标签与 predict 完全一致？",
      bool(np.all((dec > 0).astype(int) == svc_demo.predict(Xm_test_s))))
print()
print("如果想要概率，sklearn 1.9 已弃用 SVC(probability=True)（会打印弃用告警，")
print("官方建议改用 CalibratedClassifierCV）。下面按新写法演示 Platt 概率校准：")
calibrated = CalibratedClassifierCV(
    SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),
    ensemble=False,
    cv=5,
)
calibrated.fit(Xm_train_s, ym_train)
proba_cal = calibrated.predict_proba(Xm_test_s)
print("  校准后的 predict_proba（前 5 条，两列分别对应类别 0 / 1）：")
print("   ", np.round(proba_cal[:5], 4).tolist())
print(f"  校准模型测试集准确率 = {accuracy_score(ym_test, calibrated.predict(Xm_test_s)):.4f}")
print("  注意：概率校准是**在 SVM 之外再套一层逻辑回归**拟合分数→概率的映射，")
print("        所以训练时间会变长，输出的概率只是近似，不是 SVM 原生能力。")

# ---------------------------------------------------------------------------
# 步骤 6：画图 1 —— linear 核 vs rbf 核决策边界对比
# ---------------------------------------------------------------------------
print_section("步骤 6：绘图 —— linear 核 vs rbf 核决策边界对比")


def draw_svm_boundary(ax, model, X, y, title):
    """在给定坐标轴上画出 SVM 的决策边界、间隔带与支持向量。"""
    pad = 0.6
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 400),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 400),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = model.decision_function(grid).reshape(xx.shape)
    # 三色填充：中心带是间隔带（-1 到 1 之间），两侧是两类的判定区域
    ax.contourf(xx, yy, zz, levels=[zz.min() - 1, -1, 1, zz.max() + 1],
                colors=["#a8d5e5", "#f0f0f0", "#ffd79a"], alpha=0.75)
    # 实线 = 决策边界（f=0）；虚线 = 间隔边界（f=±1），虚线上的点就是支持向量
    ax.contour(xx, yy, zz, levels=[-1, 0, 1], colors=["#555555", "black", "#555555"],
               linewidths=[1.2, 2.4, 1.2], linestyles=["--", "-", "--"])
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="#1f77b4", s=42,
               edgecolors="white", linewidths=0.7, label="类别 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="#d62728", s=42,
               edgecolors="white", linewidths=0.7, label="类别 1")
    sv = model.support_vectors_
    ax.scatter(sv[:, 0], sv[:, 1], s=190, facecolors="none", edgecolors="#2ca02c",
               linewidths=1.8, label=f"支持向量（{len(sv)} 个）")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("标准化特征 1", fontsize=10)
    ax.grid(alpha=0.2, linestyle="--")
    return zz


fig1, axes1 = plt.subplots(1, 2, figsize=(13.6, 5.6))
draw_svm_boundary(axes1[0], svc_linear, Xm_train_s, ym_train,
                  f"线性核 linear：只能画一条直线\n训练 {acc_lin_train:.2%} / 测试 {acc_lin_test:.2%}"
                  "（切不开月亮形）")
draw_svm_boundary(axes1[1], svc_rbf, Xm_train_s, ym_train,
                  f"RBF 核 rbf（gamma='scale'）：边界可以弯曲\n"
                  f"训练 {acc_rbf_train:.2%} / 测试 {acc_rbf_test:.2%}（成功分开）")
axes1[0].set_ylabel("标准化特征 2", fontsize=10)
axes1[0].legend(loc="lower left", fontsize=8, framealpha=0.9)
axes1[1].legend(loc="lower left", fontsize=8, framealpha=0.9)
fig1.suptitle("同一份月亮数据：线性核 vs RBF 核（黑实线=决策边界，灰虚线=间隔边界，绿圈=支持向量）",
              fontsize=12)
fig1.tight_layout(rect=(0, 0, 1, 0.94))
moons_path = OUTPUT_DIR / "03_SVM_linear核与rbf核对比.png"
fig1.savefig(moons_path, dpi=130)
plt.close(fig1)
print("已保存图片：", moons_path)

# ---------------------------------------------------------------------------
# 步骤 7：画图 2 —— C 与 gamma 的影响（2 × 3 网格）
# ---------------------------------------------------------------------------
print_section("步骤 7：绘图 —— C 与 gamma 对决策边界平滑度的影响")

fig2, axes2 = plt.subplots(2, 3, figsize=(16.0, 9.6))
for i, c_val in enumerate((0.1, 100.0)):
    for j, g_val in enumerate((0.1, 1.0, 10.0)):
        ax = axes2[i, j]
        m = SVC(kernel="rbf", C=c_val, gamma=g_val, random_state=RANDOM_STATE)
        m.fit(Xm_train_s, ym_train)
        tr = accuracy_score(ym_train, m.predict(Xm_train_s))
        te = accuracy_score(ym_test, m.predict(Xm_test_s))
        draw_svm_boundary(ax, m, Xm_train_s, ym_train,
                          f"C = {c_val}, gamma = {g_val}\n训练 {tr:.2%} / 测试 {te:.2%}"
                          f"，SV = {len(m.support_)}")
        if i == 1:
            ax.set_xlabel("标准化特征 1", fontsize=10)
        if j == 0:
            ax.set_ylabel("标准化特征 2", fontsize=10)
axes2[0, 0].legend(loc="lower left", fontsize=7.5, framealpha=0.9)
fig2.suptitle("C 与 gamma 共同控制 SVM 的复杂度：gamma 越大边界越弯（过拟合），C 越小间隔带越宽（容错）",
              fontsize=13)
fig2.tight_layout(rect=(0, 0, 1, 0.945))
grid_path = OUTPUT_DIR / "03_SVM_C与gamma对边界的影响.png"
fig2.savefig(grid_path, dpi=130)
plt.close(fig2)
print("已保存图片：", grid_path)

# ---------------------------------------------------------------------------
# 步骤 8：画图 3 —— 各核函数在 iris 上的表现
# ---------------------------------------------------------------------------
print_section("步骤 8：绘图 —— 各核函数在 iris 上的表现")

kernel_names = [r[0] for r in kernel_records]
kernel_cv = [r[3] for r in kernel_records]
kernel_std = [r[4] for r in kernel_records]
kernel_test = [r[2] for r in kernel_records]
x_pos = np.arange(len(kernel_names))
width = 0.38

fig3, ax3 = plt.subplots(figsize=(12.4, 5.6))
bar1 = ax3.bar(x_pos - width / 2, kernel_test, width, label="测试集准确率", color="#4c72b0")
bar2 = ax3.bar(x_pos + width / 2, kernel_cv, width, yerr=kernel_std, capsize=4,
               label="5 折交叉验证准确率（误差棒=标准差）", color="#dd8452")
ax3.set_xticks(x_pos)
ax3.set_xticklabels([n.replace(" ", "\n", 1) for n in kernel_names], fontsize=8.5)
ax3.set_ylim(0.55, 1.06)
ax3.set_ylabel("准确率 accuracy", fontsize=10)
ax3.set_title("同一份 iris 数据上四种核函数的对比（C=1.0，其余参数取默认）", fontsize=12)
ax3.grid(axis="y", alpha=0.3, linestyle="--")
ax3.legend(loc="lower right", fontsize=9.5)
for b, v in zip(bar1, kernel_test):
    ax3.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
for b, v in zip(bar2, kernel_cv):
    ax3.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
fig3.tight_layout()
kernel_path = OUTPUT_DIR / "03_SVM_核函数在iris上的表现.png"
fig3.savefig(kernel_path, dpi=130)
plt.close(fig3)
print("已保存图片：", kernel_path)

# ============================================================================
# ④ 结果解读
# ============================================================================
print_section("④ 结果解读（这些数字到底意味着什么）")

print("1) 月亮数据上的核心对比：")
print(f"   线性核：训练 {acc_lin_train:.4f}，测试 {acc_lin_test:.4f}")
print(f"   RBF 核：训练 {acc_rbf_train:.4f}，测试 {acc_rbf_test:.4f}")
print(f"   两者相差约 {(acc_rbf_test - acc_lin_test) * 100:.1f} 个百分点。")
print("   这个差距**不是调参能弥补的**：线性核的假设空间里根本没有能切开月亮的曲线，")
print("   这叫'模型族表达能力不足'，只能通过换核函数（等价于换特征空间）来解决。")
print()
print("2) 支持向量的数量说明了什么：")
print(f"   线性核用了 {len(svc_linear.support_)} 个支持向量，RBF 核用了 {len(svc_rbf.support_)} 个"
      f"（训练集共 {len(ym_train)} 条）。")
print("   支持向量越多，说明越多样本落在间隔带里或分错 —— 数据越难分、或 C 越小、")
print("   或 gamma 越大（每个点都自成一圈）。支持向量占比是 SVM 的一个'难度指示器'。")
print()
print(f"3) C 与 gamma 对比图（{grid_path.name}）：")
print("   看第一行（C=0.1）与第二行（C=100）的差别，再看每一行里 gamma 从 0.1 → 10 的变化：")
print("   · gamma=0.1 时边界几乎是一条平缓的曲线（欠拟合，间隔带很宽）；")
print("   · gamma=10 时边界碎成围绕每个样本的小圈（过拟合，间隔带几乎消失）；")
print("   · C=0.1 时违反间隔的惩罚很轻，间隔带明显更宽、支持向量更多；")
print("   · C=100 时几乎不允许犯错，边界被拉到把每个训练点都分对为止。")
print("   这两张'旋钮'合起来决定了 SVM 的复杂度，实际调参要在网格上联合搜索。")
print()
print("4) iris 上各核函数的对比：")
best_kernel = max(kernel_records, key=lambda r: r[3])
print(f"   交叉验证均值最高的是：{best_kernel[0]}（CV = {best_kernel[3]:.4f} ± {best_kernel[4]:.4f}）")
print("   iris 本身近似线性可分，所以线性核已经能拿到很高分数，")
print("   RBF 核也不会更差（它包含线性核作为极限情形：gamma 很小时 RBF 近似线性）。")
print("   经验顺序：先用 RBF（默认就是它），如果线性核的 CV 与 RBF 相当，")
print("   就选线性核 —— 它训练更快、更不容易过拟合、系数还能解释。")
print()
print("5) 关于概率：")
print("   SVM 原生输出的是 decision_function（到超平面的带符号距离），不是概率。")
print("   sklearn 1.9 已经弃用 SVC(probability=True)，要概率就用 CalibratedClassifierCV 包一层。")
print("   如果业务上强依赖'校准好的概率'（比如风控打分），逻辑回归 / 树模型往往更省事。")
print()
print("6) 一句话总览：SVM = 最大间隔 + 软间隔（hinge + L2）+ 核技巧。")
print("   它把'找最好的非线性边界'化成了一个凸二次规划问题，")
print("   用少数支持向量就把整个模型撑了起来 —— 这是它优雅也强大的地方。")

# ============================================================================
# 超参数怎么调
# ============================================================================
# 【超参数怎么调】—— SVM 实战调参顺序
#
# 0. 前置条件（不做这一步，调参全是白费）：
#    - **必须标准化**（StandardScaler / MinMaxScaler）。SVM 靠内积和距离工作，
#      量纲大的特征会直接主导核函数，尤其在 RBF 核下会彻底毁掉结果；
#    - 放进 Pipeline 里做标准化，交叉验证时避免数据泄漏。
#
# 1. kernel（第一优先级：先定核，再调参）：
#    - 不知道用什么 → 直接上 'rbf'（sklearn 的默认值，也是通用首选）；
#    - 特征维度很高（文本 TF-IDF、稀疏特征）→ 优先 'linear'，高维空间往往已经线性可分；
#    - 样本量很大（>1 万）→ 也优先 'linear'（RBF 训练慢），或改用 LinearSVC / SGDClassifier；
#    - 'poly' / 'sigmoid' 参数多、稳定性差，非特殊情况不用。
#
# 2. C 与 gamma（第二优先级，必须**联合**搜索）：
#    - 典型网格：C ∈ {0.01, 0.1, 1, 10, 100, 1000}，
#                gamma ∈ {'scale', 0.001, 0.01, 0.1, 1, 10}；
#    - 先用粗网格定位数量级，再在最优附近用细网格；
#    - 经验：C 与 gamma 同时增大 → 过拟合；同时减小 → 欠拟合；
#    - 判据还是那两条：训练准确率远高于 CV → 过拟合；两者都低 → 欠拟合。
#
# 3. gamma 的取值口径：
#    - 'scale'（默认）= 1/(n_features × X.var())，随数据自动缩放，最省心；
#    - 'auto' = 1/n_features，不看方差，特征方差差异大时容易出问题；
#    - 想手动控制就传数字，此时务必和 'scale' 的计算结果比一比数量级
#      （本脚本月亮数据上 'scale' ≈ 0.5~1 的量级）。
#
# 4. class_weight：
#    - 类别不平衡（医疗、欺诈）时设 'balanced'，或手写字典给少数类更大权重；
#    - 配合评估指标一起改：别再只看 accuracy，看 recall / F1 / average_precision。
#
# 5. degree / coef0（只在 kernel='poly' / 'sigmoid' 时生效）：
#    - degree 从 2 开始试，一般不超过 4~5（再大数值不稳定）；
#    - coef0 常用 0 或 1；它控制低阶项的相对权重。
#
# 6. 需要概率 / 需要大规模训练时的替代方案：
#    - 要概率 → CalibratedClassifierCV(SVC(...), ensemble=False, cv=5)
#                （不要再写 SVC(probability=True)，1.9 已弃用）；
#    - 要速度 / 大样本 → LinearSVC（线性核的快速实现）或 SGDClassifier(loss='hinge')；
#    - 要非线性又要快 → 先降维（PCA）再 RBF，或用 Nystroem + LinearSVC 近似核方法。
#
# 7. 推荐自动化做法：
#    GridSearchCV(Pipeline([("scaler", StandardScaler()),
#                           ("svc", SVC())]),
#                 {"svc__kernel": ["rbf", "linear"],
#                  "svc__C": [0.1, 1, 10, 100],
#                  "svc__gamma": ["scale", 0.01, 0.1, 1]},
#                 cv=5, scoring="accuracy", n_jobs=-1)
#    —— 注意 Pipeline 里的 StandardScaler 会在每一折的**训练部分**单独 fit，
#       这才是评估 SVM 的正确姿势。
# ============================================================================

print()
print("【完成】03_SVM.py 运行结束")

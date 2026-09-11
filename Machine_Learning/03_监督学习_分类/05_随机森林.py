r"""《机器学习》课案 · 03 监督学习-分类 · 05 随机森林（Random Forest）

对应课案章节
-----------
《机器学习》课案「分类」章 —— 随机森林（Random Forest）：
课案原文位于 `.course_extract/机器学习_课案.md` 第 632~684 行。
覆盖内容：随机森林核心思想（构建多棵决策树并集成它们的预测结果，
以提高准确性和稳定性）、课案配图 "Bagging"、
课案示例代码（load_iris + RandomForestClassifier(n_estimators=100, random_state=42)）。

本节知识点
---------
1. Bagging（Bootstrap AGGregating）三步走：
   ① 自助采样 bootstrap：从 N 个训练样本中**有放回**地抽 N 个，得到一份"略有不同"的数据集；
   ② 在这份数据上**独立、并行**地训练一个基学习器（这里是决策树）；
   ③ 汇总：分类用**投票**（多数表决），回归用**平均**。
2. 为什么自助采样能"造出差异"：每个样本被抽中的概率是 1-1/N ≈ 63.2%，
   剩下约 36.8% 的样本**一次都没被抽到** —— 这些"袋外样本"就是 OOB 估计的来源。
3. 随机森林在 Bagging 之上再加一层随机：每次分裂只在**随机抽出的 max_features 个特征**里
   挑最优分裂。这一层随机让树与树之间"去相关"，是随机森林比普通 Bagging 更强的关键。
4. 袋外估计（OOB score）：用每棵树的袋外样本给它打分，再汇总，
   相当于**免费的一次交叉验证**，不用额外划分验证集。
5. 为什么随机森林主要降低**方差**、而不太增加偏差：
   B 棵两两相关系数为 ρ、方差为 σ² 的树取平均后，
   方差 = ρσ² + (1-ρ)σ²/B。
   B 增大只能压掉第二项，真正决定下界的是相关系数 ρ ——
   所以要靠"特征随机"把 ρ 压下来（而 ρ 降不下来时，堆再多树也没用）。
6. 与单棵决策树的对比：精度更高、稳定得多（换个随机种子结果几乎不变），
   代价是可解释性下降（不再有单一 if-else 规则，只能看特征重要性 / 部分依赖）。

运行方式
-------
PowerShell（路径含中文，必须加引号并用 & 调用）：

    $env:PYTHONUTF8='1'; & 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Machine_Learning\03_监督学习_分类\05_随机森林.py'

输出：
    - 控制台：四段式讲解 + 单树 vs 森林对比 + OOB 解释 + n_estimators 曲线 + 中文解读
    - 图片： Machine_Learning/output/03_随机森林_树数量与准确率.png
            Machine_Learning/output/03_随机森林_单树与森林对比.png
            Machine_Learning/output/03_随机森林_Bagging自助采样.png
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

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


def resolve_max_features(max_features, n_features: int) -> int:
    """把 max_features 的写法（'sqrt' / 'log2' / 比例 / 整数）换算成实际特征个数。

    scikit-learn 1.9 的 RandomForestClassifier 已不再暴露 max_features_ 属性，
    所以这里按它内部的换算规则自己算一遍，方便把结果打印出来。
    """
    if max_features == "sqrt":
        return max(1, int(np.sqrt(n_features)))
    if max_features == "log2":
        return max(1, int(np.log2(n_features)))
    if isinstance(max_features, float):
        return max(1, int(max_features * n_features))
    if max_features is None:
        return n_features
    return int(max_features)


# ============================================================================
# ① 原理与数学推导
# ============================================================================
# 【1.1 出发点：一棵决策树的两个毛病】
#   上一节我们看到，单棵决策树有两个致命缺点：
#     (1) 极易过拟合：不加限制时长到每个叶子只剩 1 个样本，训练误差必然为 0；
#     (2) 极不稳定（高方差）：训练数据稍微变一点，整棵树的结构就可能完全不同。
#   随机森林的思路非常朴素：**既然一棵树不靠谱，那就种一大片森林，让大家投票。**
#   这就是集成学习里的 Bagging（Bootstrap AGGregating）。
#
# 【1.2 Bagging 的三步】
#   第 1 步：自助采样（bootstrap sampling）
#       从 N 个训练样本中**有放回**地随机抽 N 个，得到一份新的训练集 D_b。
#       因为是"有放回"，D_b 里必然有重复样本、也必然漏掉一些样本。
#       关键计算：某个特定样本在 N 次抽取中**一次都没被抽到**的概率是
#           (1 - 1/N)^N  →  N→∞ 时收敛到 e^(-1) ≈ 0.368
#       反过来，被抽到的概率约 1 - 0.368 = **0.632**，而且其中还有重复。
#       所以每份 D_b 大约只覆盖了原数据 63.2% 的**不同**样本 —— 这就是"多样性"的来源。
#   第 2 步：并行训练
#       B 份自助采样集 → 训练 B 个基学习器。彼此之间**完全独立**，可以并行。
#   第 3 步：汇总
#       分类任务：每个基学习器投一票，得票最多的类别获胜（多数表决）；
#       回归任务：取 B 个预测值的平均。
#
# 【1.3 随机森林在 Bagging 上又加了一层随机】
#   只用 Bagging 还不够，因为如果数据里有一个极强的特征，
#   那么所有树都会在根节点附近用同一个特征分裂 → 树与树长得非常像 → 投票失去意义。
#   随机森林的补丁是：**每次分裂时，只在随机抽出的 max_features 个特征里挑最优**。
#   于是每棵树看到的"可用特征子集"都不同，树与树之间的相关性被强行压低。
#   这就是"随机"两个字在随机森林名字里的含义，也是它比普通 Bagging（max_features=None）更强的关键。
#   （sklearn 里分类任务 max_features 默认 'sqrt'，回归任务默认 1.0=全部。）
#
# 【1.4 为什么有效：偏差-方差分解】
#   设有 B 棵**两两相关系数**为 ρ、单棵方差为 σ² 的树，把它们平均后：
#
#       平均后的方差 = ρ·σ² + (1 - ρ)·σ² / B
#                     \_下界_/   \__可以被 B 压掉的部分__/
#
#   这个公式把随机森林的秘密讲透了：
#     · 第二项 (1-ρ)σ²/B 随 B 增大而趋于 0 —— 所以树越多，方差越小、结果越稳；
#     · 第一项 ρσ² **与 B 无关**，它决定了方差的下界：只要树之间还有相关性，
#       再怎么堆树也压不下去。所以真正要紧的是**把 ρ 弄小** ——
#       这正是 max_features 特征随机要做的事；
#     · 平均**不会显著增大偏差**：因为每棵树都是在同一份数据分布上训练的，
#       平均只是把"各自的随机误差"抵消掉，不会系统性地偏离真实规律。
#     · 这就是"随机森林主要降方差、几乎不增偏差"的数学解释。
#   附带好处：Bagging 之后模型"很难过拟合" —— 增加树的数量几乎不会让测试误差变差，
#   只会让它趋于平稳（本脚本的 n_estimators 曲线就是这个现象的实证）。
#
# 【1.5 袋外估计（OOB, Out-Of-Bag score）—— 免费的验证集】
#   每棵树 b 都有一批"没被它的自助采样抽到"的样本（约占 36.8%），这些就是它的袋外样本。
#   于是可以对**每一个训练样本**做这样的评估：
#       找出所有"袋外样本包含它"的树，让这些树对它投票，得到预测；
#       把该样本的预测与真实标签比较。
#   所有训练样本都这样过一遍，得到的准确率就是 oob_score_。
#   它等价于一次"近似留一法"的交叉验证，**不需要额外划分验证集、不浪费训练数据**，
#   是随机森林独有的一件称手工具（Bagging 家族都有，前提是 bootstrap=True）。
#   注意：只有当 bootstrap=True 且样本量足够时 OOB 才有意义；
#        样本很少时 OOB 会偏乐观（每棵树的袋外样本太少，统计不稳）。
#
# 【1.6 优缺点】
#   优点：
#     - 精度通常显著高于单棵决策树，且**非常稳健**（换随机种子结果几乎不变）；
#     - 几乎不需要精细调参：n_estimators 大一点总是更稳（代价只是时间）；
#     - 不需要标准化，能处理混合类型的特征，能捕捉特征交互；
#     - 自带 OOB 估计，可以在不额外划分数据的情况下估计泛化误差；
#     - 支持并行（n_jobs=-1），训练时间随核数下降；
#     - 能给出特征重要性，便于做特征筛选。
#   缺点：
#     - **可解释性差**：几百棵树没法变成一句人话规则，只能用重要性 / 部分依赖图近似；
#     - 模型体积大、预测比单棵树慢（要跑 B 棵树）；
#     - 对**高基数类别特征**的特征重要性有偏（偏向取值多的特征）；
#     - 外推能力差（树模型的通病）：预测值永远落在训练集标签的范围内；
#     - 在极度不平衡的数据上仍需配合 class_weight / 阈值调整。
# ============================================================================


# ============================================================================
# ② sklearn API 关键参数逐个解释
# ============================================================================
# 默认值取自本机 scikit-learn 1.9.0 实测（RandomForestClassifier().get_params()）。
#
# n_estimators : int，默认 100
#    含义：森林里树的数量 B。
#    调大：方差下降、结果更稳（曲线趋于平台），但训练与预测时间线性增长；
#    调小：训练快，但随机性带来的波动还没被平均掉，准确率偏低、不稳定。
#    常用值：100（默认，够用）/ 300 / 500；数据量很大时 100~300 通常已经够。
#    重要经验：**这个参数"越大越好、只是更慢"，不存在"调太大导致过拟合"的问题**。
#
# max_depth : int 或 None，默认 None
#    含义：每棵树的最大深度。注意随机森林里**单棵树可以长得比较深**：
#          反正有 Bagging 平均在兜底，不需要像单棵树那样靠剪枝来防过拟合。
#    调小：单棵树更弱、偏差更大，但整体可能仍不错（也是一种正则化）；
#    调大/None（默认）：单棵树更强、方差更大，靠平均来抵消。
#    常用值：None（默认）；数据噪声极大时可以试 10~20。
#
# max_features : int / float / {'sqrt','log2'} 或 None，默认 'sqrt'
#    含义：每次分裂时随机考虑的特征个数，**随机森林最关键的参数**（它控制 ρ）。
#      - 'sqrt'：√n_features 个（分类默认值，例如 4 个特征 → 2 个）；
#      - 'log2'：log₂(n_features) 个（更激进，树之间更不相关、单棵更弱）；
#      - float：比例（如 0.5 表示一半特征）；
#      - None / 1.0：用全部特征（退化成普通 Bagging，树之间相关性最高）。
#    调小：树之间更"去相关"、ρ 变小 → 集成效果更好；但单棵树更弱 → 偏差上升；
#    调大：单棵树更强 → 偏差下降；但树之间更像 → ρ 变大 → 方差下不去。
#    调参起点：分类先用 'sqrt'，再试 'log2' 和 0.3~0.5 的比例，选 CV 最好的。
#
# min_samples_leaf : int 或 float，默认 1
#    含义：每个叶子至少保留多少样本。这是随机森林里**最有效的防过拟合旋钮**。
#    调大（如 3、5）：单棵树更平滑 → 偏差上升、方差下降；
#    调小（1，默认）：允许叶子只有一个样本（但 Bagging 会兜底）。
#    常用值：1 / 2 / 3 / 5。类别不平衡或噪声大时可以调到 3~5。
#
# bootstrap : bool，默认 True
#    含义：是否使用自助采样（有放回抽样）。
#    True（默认）：每棵树用一份 bootstrap 样本，袋外样本可用 → 支持 oob_score；
#    False：每棵树用**全部**训练数据（只靠 max_features 制造多样性）→
#           sklearn 官方把这个变体叫 "ExtraTrees 之外的全数据 Bagging"，
#           此时 oob_score 不可用（没有袋外样本）。
#    经验：几乎总是保持 True。
#
# oob_score : bool，默认 False
#    含义：是否计算袋外估计（oob_score_）。要求 bootstrap=True。
#    调成 True：训练结束后可以直接读 model.oob_score_，相当于免费的验证分数；
#              代价是要多做一轮预测（轻微变慢）。
#    常用值：做模型选择时设 True，非常划算（本脚本就开着它）。
#
# n_jobs : int，默认 None
#    含义：并行训练用的 CPU 核数。None = 1；-1 = 用满所有核。
#    影响：树的训练相互独立，并行几乎线性加速。大森林必备。
#    常用值：-1。
#
# random_state : int 或 None，默认 None
#    含义：随机种子，同时控制"自助采样"和"特征抽样"两处随机。
#    影响：固定为 42 保证每次结果一致。值得注意的是，随机森林对种子的敏感度
#          远低于单棵决策树 —— 你可以用几组不同种子跑一遍，会发现准确率几乎不变。
#
# class_weight : dict 或 'balanced' 或 'balanced_subsample'，默认 None
#    'balanced'：按整体类别频率反比加权；
#    'balanced_subsample'：按**每棵树的 bootstrap 样本**内的频率反比加权（推荐用于不平衡数据）。
#
# max_samples : int 或 float，默认 None
#    含义：自助采样时每棵树抽多少样本。None 表示和训练集一样多（N 个）。
#    只在 bootstrap=True 时生效。
#    调小（如 0.5）：每棵树看到的数据更少 → 树之间差异更大（ρ 更小），
#                    但单棵树更弱 → 偏差上升。是"偏差-方差"的又一根旋钮。
#
# criterion : {'gini','entropy','log_loss'}，默认 'gini'。基学习器（树）的分裂标准。
# warm_start : bool，默认 False。True 时再次 fit 会在已有树上继续加树（增量训练）。
# ccp_alpha : float，默认 0.0。每棵树的代价复杂度剪枝系数，一般不用设。
# ============================================================================


print_section("《机器学习》课案 · 分类 · 05 随机森林（Random Forest）")
print("本脚本四段结构： ① 原理与数学推导  ② sklearn API 参数解释  ③ 完整可运行代码  ④ 结果解读")
print("配套图片输出目录：", OUTPUT_DIR)

# ============================================================================
# ③ 完整可运行代码
# ============================================================================

# ---------------------------------------------------------------------------
# 步骤 1：提取数据（与上一节完全相同的 iris 数据）
# ---------------------------------------------------------------------------
print_section("步骤 1：提取数据（load_iris 全量，三分类）")

iris = load_iris()
X = iris.data
y = iris.target
feature_names = list(iris.feature_names)
class_names = [str(n) for n in iris.target_names]
print("数据集来源：sklearn.datasets.load_iris（内置数据集，无需联网下载）")
print("特征名：", feature_names)
print("类别名：", class_names)
print(f"样本形状：X = {X.shape}，y = {y.shape}")
print("各类样本数：", {class_names[i]: int(n) for i, n in enumerate(np.bincount(y))})

# ---------------------------------------------------------------------------
# 步骤 2：同一次划分 —— 保证单棵树与随机森林用的是同一份数据
# ---------------------------------------------------------------------------
print_section("步骤 2：划分训练集/测试集（与课案一致，random_state=42）")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE
)
print(f"训练集 X_train = {X_train.shape}，测试集 X_test = {X_test.shape}")
print("训练集各类数量：", np.bincount(y_train), "，测试集各类数量：", np.bincount(y_test))
print("注意：单棵决策树与随机森林**共用这一次划分**，")
print("      只有数据完全相同，两者的准确率才有可比性。")
print("      随机森林同样不需要标准化（它内部还是决策树，只看特征的大小关系）。")

# ---------------------------------------------------------------------------
# 步骤 3：单棵决策树 vs 随机森林
# ---------------------------------------------------------------------------
print_section("步骤 3：单棵决策树 vs 随机森林（同一份数据、同一次划分）")

tree_clf = DecisionTreeClassifier(random_state=RANDOM_STATE)
tree_clf.fit(X_train, y_train)
tree_train_acc = accuracy_score(y_train, tree_clf.predict(X_train))
tree_test_acc = accuracy_score(y_test, tree_clf.predict(X_test))
tree_cv = cross_val_score(tree_clf, X_train, y_train, cv=5)

rf_clf = RandomForestClassifier(
    n_estimators=100,          # 100 棵树
    max_features="sqrt",       # 每次分裂随机考虑 √4 = 2 个特征（分类默认值）
    oob_score=True,            # 打开袋外估计
    bootstrap=True,            # 自助采样
    n_jobs=-1,                 # 用满所有 CPU 核
    random_state=RANDOM_STATE,
)
rf_clf.fit(X_train, y_train)
rf_train_acc = accuracy_score(y_train, rf_clf.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf_clf.predict(X_test))
rf_cv = cross_val_score(rf_clf, X_train, y_train, cv=5)

print("-" * 96)
print(f"{'模型':<36}{'训练集准确率':>16}{'测试集准确率':>16}{'5折CV均值':>14}{'CV标准差':>12}")
print("-" * 96)
print(f"{'单棵决策树 DecisionTree(不限制深度)':<36}{tree_train_acc:>16.4f}"
      f"{tree_test_acc:>16.4f}{tree_cv.mean():>14.4f}{tree_cv.std():>12.4f}")
print(f"{'随机森林 RandomForest(100 棵)':<36}{rf_train_acc:>16.4f}"
      f"{rf_test_acc:>16.4f}{rf_cv.mean():>14.4f}{rf_cv.std():>12.4f}")
print("-" * 96)
print(f"随机森林的袋外估计 oob_score_ = {rf_clf.oob_score_:.4f}")
print("oob_score_ 是什么：")
print("  每棵树的 bootstrap 采样大约会漏掉 36.8% 的训练样本，这些样本叫这棵树的'袋外样本'。")
print("  对每一个训练样本，收集所有'把它当袋外样本'的树，让它们投票得到预测，")
print("  再与真实标签比较 —— 所有训练样本过一遍得到的准确率，就是 oob_score_。")
print("  它相当于**一次免费的交叉验证**：不用额外划分验证集，也不浪费任何训练数据。")
print(f"  本例中 OOB = {rf_clf.oob_score_:.4f}，与 5 折交叉验证均值 {rf_cv.mean():.4f} 基本一致；")
print(f"  它比测试集准确率 {rf_test_acc:.4f} 略低是正常的：测试集只有 {len(y_test)} 条且这份数据太容易，")
print("  单次划分的测试分数偏乐观，而 OOB 用到了全部 105 条训练样本，估计更保守也更可靠。")
print()
tree_depths = [int(t.get_depth()) for t in rf_clf.estimators_]
print(f"森林里每棵树的深度分布：最小 {min(tree_depths)} 层，最大 {max(tree_depths)} 层，"
      f"平均 {np.mean(tree_depths):.1f} 层（前 10 棵分别是 {tree_depths[:10]}）")
print("可以看到单棵树都长得挺深，但平均之后整体依然很稳 ——")
print("这正是 Bagging 的价值：**允许基学习器'弱而不同'，靠平均来消除随机误差**。")
print(f"每棵树每次分裂随机考虑 max_features='sqrt'，"
      f"共 {rf_clf.n_features_in_} 个特征 → 实际每次考虑 "
      f"{resolve_max_features('sqrt', rf_clf.n_features_in_)} 个特征。")
print(f"森林各特征的重要性：{np.round(rf_clf.feature_importances_, 4).tolist()}")

print("\n随机森林的分类报告：")
print(classification_report(y_test, rf_clf.predict(X_test), target_names=class_names,
                            digits=4, zero_division=0))

# 稳定性对比：换不同的数据划分，看两种模型的准确率波动
# （iris 太容易，两种模型在任何划分上都是满分，看不出稳定性差异，所以改用带噪声的合成数据）
print("稳定性对比：同一份数据、12 种不同的 train_test_split 划分，看准确率波动。")
print("（iris 太容易，两种模型在任何划分上都是满分，用它做这个实验没有意义；")
print("  这里换一份带噪声的合成数据：make_classification 300×10，25% 标签被翻转。）")
X_st, y_st = make_classification(
    n_samples=300, n_features=10, n_informative=5, n_redundant=0,
    n_clusters_per_class=1, class_sep=1.0, flip_y=0.25, random_state=RANDOM_STATE,
)
print("-" * 76)
print(f"{'划分种子':<12}{'单棵决策树':>18}{'随机森林':>18}{'森林相对提升':>18}")
print("-" * 76)
tree_accs, rf_accs = [], []
for seed in range(12):
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(
        X_st, y_st, test_size=0.3, random_state=seed, stratify=y_st
    )
    t = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(Xs_tr, ys_tr)
    f = RandomForestClassifier(n_estimators=100, max_features="sqrt",
                               random_state=RANDOM_STATE, n_jobs=-1).fit(Xs_tr, ys_tr)
    a_t = accuracy_score(ys_te, t.predict(Xs_te))
    a_f = accuracy_score(ys_te, f.predict(Xs_te))
    tree_accs.append(a_t)
    rf_accs.append(a_f)
    print(f"{seed:<12}{a_t:>18.4f}{a_f:>18.4f}{a_f - a_t:>+18.4f}")
print("-" * 76)
print(f"{'均值':<12}{np.mean(tree_accs):>18.4f}{np.mean(rf_accs):>18.4f}"
      f"{np.mean(rf_accs) - np.mean(tree_accs):>+18.4f}")
print(f"{'标准差':<12}{np.std(tree_accs):>18.4f}{np.std(rf_accs):>18.4f}")
print(f"{'最差':<12}{min(tree_accs):>18.4f}{min(rf_accs):>18.4f}")
print(f"{'最好':<12}{max(tree_accs):>18.4f}{max(rf_accs):>18.4f}")
print("单棵树的准确率在不同划分之间明显摆动，随机森林既更高、也更集中 ——")
print("这就是'降低方差'在数字上的样子：均值上升、标准差下降。")

# ---------------------------------------------------------------------------
# 步骤 4：n_estimators 从 1 到 100 —— 树多了会趋于稳定
# ---------------------------------------------------------------------------
print_section("步骤 4：树的数量 n_estimators 从 1 增加到 100")

print("用 warm_start=True 增量加树（不用每次从头训练），记录训练/测试/OOB 准确率：")
print("说明：树少于 10 棵时，会有一部分训练样本'对森林里所有树都是袋外样本'，")
print("      这类样本的 OOB 预测根本不存在（sklearn 会打印告警），")
print("      所以 1~9 棵这一段只用独立森林统计训练/测试准确率，OOB 从 10 棵起才记录。")
print("-" * 84)
print(f"{'树的数量':<10}{'训练集准确率':>16}{'测试集准确率':>16}{'袋外估计 OOB':>16}{'说明':>16}")
print("-" * 84)
report_points = {1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}

n_axis = list(range(1, 101))
n_train_acc, n_test_acc, n_oob = [], [], []
oob_start = 10  # 从 10 棵树起才开始计算袋外估计


def n_note(n_val):
    """根据树的数量给出中文说明。"""
    if n_val <= 5:
        return "波动很大"
    if n_val <= 30:
        return "快速上升"
    return "基本稳定"


# 第 1 段：1~9 棵，不启用 OOB
for n in range(1, oob_start):
    rf_small = RandomForestClassifier(
        n_estimators=n, max_features="sqrt", bootstrap=True,
        oob_score=False, n_jobs=-1, random_state=RANDOM_STATE,
    )
    rf_small.fit(X_train, y_train)
    a_tr = accuracy_score(y_train, rf_small.predict(X_train))
    a_te = accuracy_score(y_test, rf_small.predict(X_test))
    n_train_acc.append(a_tr)
    n_test_acc.append(a_te)
    n_oob.append(float("nan"))
    if n in report_points:
        print(f"{n:<10}{a_tr:>16.4f}{a_te:>16.4f}{'—':>16}{n_note(n):>16}")

# 第 2 段：10~100 棵，开启 oob_score 并配合 warm_start 增量加树
rf_inc = RandomForestClassifier(
    n_estimators=oob_start,
    max_features="sqrt",
    oob_score=True,
    bootstrap=True,
    warm_start=True,     # 关键：再次 fit 时在已有树上继续加，而不是重训
    n_jobs=-1,
    random_state=RANDOM_STATE,
)
for n in range(oob_start, 101):
    rf_inc.set_params(n_estimators=n)
    rf_inc.fit(X_train, y_train)
    a_tr = accuracy_score(y_train, rf_inc.predict(X_train))
    a_te = accuracy_score(y_test, rf_inc.predict(X_test))
    a_oob = float(rf_inc.oob_score_)
    n_train_acc.append(a_tr)
    n_test_acc.append(a_te)
    n_oob.append(a_oob)
    if n in report_points:
        print(f"{n:<10}{a_tr:>16.4f}{a_te:>16.4f}{a_oob:>16.4f}{n_note(n):>16}")
print("-" * 84)
n_oob_arr = np.array(n_oob)
print(f"树从 1 棵增加到 10 棵：袋外估计从 {n_oob_arr[9]:.4f} 起，"
      f"这一段（10~20 棵）的波动是 {np.std(n_oob_arr[9:20]):.4f}；")
print(f"树从 50 棵增加到 100 棵：袋外估计稳定在 {np.nanmin(n_oob_arr[49:]):.4f} ~ "
      f"{np.nanmax(n_oob_arr[49:]):.4f}，波动只有 {np.std(n_oob_arr[49:]):.4f}。")
print(f"（测试集准确率在本数据上从 5 棵起就一直是 {n_test_acc[-1]:.4f}，已经封顶，")
print("  看不出趋势；所以这里改用袋外估计 OOB 来观察'先上升、后趋于平台'的过程。）")
print("结论：森林的准确率随树数增加**先上升、后趋于平台**；")
print("      过了平台再加树只是浪费时间，并不会让模型变差（这一点和单棵树完全不同）。")
n_test_arr = np.array(n_test_acc)
print(f"本组实验里测试集准确率从第 "
      f"{n_axis[int(np.argmax(n_test_arr >= n_test_arr.max() - 1e-12))]} 棵树起"
      f"就达到 {n_test_arr.max():.4f} 并一直保持不动。")

# ---------------------------------------------------------------------------
# 步骤 5：max_features 对去相关的影响
# ---------------------------------------------------------------------------
print_section("步骤 5：max_features 如何影响集成效果（去相关实验）")

print("在同样的 100 棵树下，只改 max_features（每次分裂随机考虑几个特征）：")
print("-" * 84)
print(f"{'max_features':<16}{'实际特征数':>12}{'训练集准确率':>16}{'测试集准确率':>16}{'袋外估计':>14}")
print("-" * 84)
mf_records = []
for mf in ["sqrt", "log2", 1.0, 0.5]:
    rf_mf = RandomForestClassifier(
        n_estimators=100, max_features=mf, oob_score=True,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    rf_mf.fit(X_train, y_train)
    a_tr = accuracy_score(y_train, rf_mf.predict(X_train))
    a_te = accuracy_score(y_test, rf_mf.predict(X_test))
    mf_records.append((mf, a_tr, a_te, float(rf_mf.oob_score_)))
    print(f"{str(mf):<16}{resolve_max_features(mf, rf_mf.n_features_in_):>12}"
          f"{a_tr:>16.4f}{a_te:>16.4f}"
          f"{float(rf_mf.oob_score_):>14.4f}")
print("-" * 84)
print("注意：iris 上四种取法结果完全一样。原因有两条：")
print("  ① 只有 4 个特征，'sqrt' 和 'log2' 换算下来都等于 2，'0.5' 也是 2；")
print("  ② iris 太容易分，无论怎么去相关，测试集都是满分，差异被'顶死'了。")
print("所以下面换回带噪声的合成数据，把 max_features 的真实作用跑出来（用 OOB 比较）：")
X_mf, y_mf = make_classification(
    n_samples=300, n_features=10, n_informative=5, n_redundant=0,
    n_clusters_per_class=1, class_sep=1.0, flip_y=0.25, random_state=RANDOM_STATE,
)
Xmf_tr, Xmf_te, ymf_tr, ymf_te = train_test_split(
    X_mf, y_mf, test_size=0.3, random_state=RANDOM_STATE, stratify=y_mf
)
print("-" * 88)
print(f"{'max_features':<16}{'实际特征数':>12}{'训练集准确率':>16}{'测试集准确率':>16}"
      f"{'袋外估计 OOB':>16}")
print("-" * 88)
mf_noisy_records = []
for mf in ["sqrt", "log2", 0.5, 0.2, 1.0]:
    rf_mf2 = RandomForestClassifier(
        n_estimators=200, max_features=mf, oob_score=True,
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    rf_mf2.fit(Xmf_tr, ymf_tr)
    a_tr = accuracy_score(ymf_tr, rf_mf2.predict(Xmf_tr))
    a_te = accuracy_score(ymf_te, rf_mf2.predict(Xmf_te))
    oob = float(rf_mf2.oob_score_)
    mf_noisy_records.append((mf, resolve_max_features(mf, rf_mf2.n_features_in_), a_tr, a_te, oob))
    print(f"{str(mf):<16}{resolve_max_features(mf, rf_mf2.n_features_in_):>12}"
          f"{a_tr:>16.4f}{a_te:>16.4f}{oob:>16.4f}")
print("-" * 88)
best_mf = max(mf_noisy_records, key=lambda r: r[4])
worst_mf = min(mf_noisy_records, key=lambda r: r[4])
print(f"OOB 最高的是 max_features={best_mf[0]}（{best_mf[4]:.4f}），"
      f"最低的是 max_features={worst_mf[0]}（{worst_mf[4]:.4f}），"
      f"相差 {best_mf[4] - worst_mf[4]:+.4f}。")
print("max_features 的核心作用是**给树'去相关'**：")
print("  · = 1.0（用全部 10 个特征）时，每棵树都倾向于在根节点附近挑同一批最强特征，")
print(f"    树与树高度相似（相关系数 ρ 大），投票的效果被削弱 → OOB 只有 "
      f"{[r for r in mf_noisy_records if r[0] == 1.0][0][4]:.4f}；")
print("  · = 'sqrt'（3 个特征）/'log2'（3 个特征）/0.5（5 个）时，每棵树看到的候选特征不同，")
print(f"    树之间差异变大（ρ 小），集成的方差下界 ρσ² 更低 → OOB 升到 "
      f"{best_mf[4]:.4f}；")
print("  · 但也不能太小：= 0.2（2 个特征）时单棵树太弱，偏差上升，OOB 又掉回去 ——")
print("    这正是 max_features 上'偏差-方差'的平衡点。")
print("  回想方差公式 方差 = ρσ² + (1-ρ)σ²/B：B 只能压掉第二项，")
print("  真正决定下限的是 ρ —— 这就是 max_features 存在的意义。")

# ---------------------------------------------------------------------------
# 步骤 6：画图 1 —— n_estimators vs 准确率
# ---------------------------------------------------------------------------
print_section("步骤 6：绘图 —— 树的数量与准确率")

fig1, ax1 = plt.subplots(figsize=(9.2, 5.4))
ax1.plot(n_axis, n_train_acc, "-", color="#d62728", linewidth=1.8, label="训练集准确率")
ax1.plot(n_axis, n_test_acc, "-", color="#1f77b4", linewidth=1.8, label="测试集准确率")
ax1.plot(n_axis, n_oob, "--", color="#2ca02c", linewidth=1.8, label="袋外估计 OOB 准确率")
ax1.axvspan(1, 10, color="#d62728", alpha=0.07)
ax1.axvspan(30, 100, color="#2ca02c", alpha=0.07)
ax1.text(4.0, 0.72, "树少：\n波动大", ha="center", fontsize=10, color="#a01c1c")
ax1.text(66, 0.72, "树多：\n曲线趋于平台", ha="center", fontsize=10, color="#1a6b34")
ax1.set_title("随机森林：树的数量 n_estimators 与准确率的关系", fontsize=12)
ax1.set_xlabel("树的数量 n_estimators", fontsize=11)
ax1.set_ylabel("准确率 accuracy", fontsize=11)
ax1.set_xlim(0, 101)
ax1.grid(alpha=0.3, linestyle="--")
ax1.legend(loc="lower right", fontsize=10)
# 这里用手工边距而不是 tight_layout()：三根线 + 两行标注时 tight_layout 偶尔会
# 报 "bottom and top margins cannot be made large enough"，手工指定更稳。
fig1.subplots_adjust(left=0.10, right=0.97, top=0.91, bottom=0.11)
n_est_path = OUTPUT_DIR / "03_随机森林_树数量与准确率.png"
fig1.savefig(n_est_path, dpi=130)
plt.close(fig1)
print("已保存图片：", n_est_path)

# ---------------------------------------------------------------------------
# 步骤 7：画图 2 —— 单棵树 vs 森林（特征重要性 + 决策边界）
# ---------------------------------------------------------------------------
print_section("步骤 7：绘图 —— 单棵决策树与随机森林的对比")

fig2, (ax2a, ax2b, ax2c) = plt.subplots(1, 3, figsize=(17.4, 5.2))

# (a) 特征重要性对比（分组条形图）
x_pos = np.arange(len(feature_names))
width = 0.38
ax2a.bar(x_pos - width / 2, tree_clf.feature_importances_, width,
         label="单棵决策树", color="#dd8452")
ax2a.bar(x_pos + width / 2, rf_clf.feature_importances_, width,
         label="随机森林（100 棵）", color="#4c72b0")
ax2a.set_xticks(x_pos)
ax2a.set_xticklabels([n.replace(" (cm)", "") for n in feature_names], fontsize=9)
ax2a.set_ylabel("特征重要性", fontsize=10)
ax2a.set_title("(a) 特征重要性对比", fontsize=11)
ax2a.grid(axis="y", alpha=0.3, linestyle="--")
ax2a.legend(fontsize=9)

# (b)(c) 决策边界对比：用两个特征便于画在平面上
X2 = iris.data[:, [2, 3]]
y2 = iris.target
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.3, random_state=RANDOM_STATE, stratify=y2
)
tree_vis = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(X2_train, y2_train)
rf_vis = RandomForestClassifier(n_estimators=100, max_features="sqrt",
                                random_state=RANDOM_STATE, n_jobs=-1).fit(X2_train, y2_train)

pad = 0.9
xx, yy = np.meshgrid(
    np.linspace(X2_train[:, 0].min() - pad, X2_train[:, 0].max() + pad, 400),
    np.linspace(X2_train[:, 1].min() - pad, X2_train[:, 1].max() + pad, 400),
)
grid_points = np.c_[xx.ravel(), yy.ravel()]
region_cmap = ListedColormap(["#a8d5e5", "#ffd79a", "#c4a3d4"])
point_colors = ["#1f77b4", "#e08a00", "#6a3d9a"]

vis_acc = {}
for ax, mdl, name in [(ax2b, tree_vis, "单棵决策树（不限制深度）"),
                      (ax2c, rf_vis, "随机森林（100 棵）")]:
    zz = mdl.predict(grid_points).reshape(xx.shape)
    a = accuracy_score(y2_test, mdl.predict(X2_test))
    vis_acc[name] = a
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5, 2.5], cmap=region_cmap, alpha=0.6)
    for cls in range(3):
        ax.scatter(X2_train[y2_train == cls, 0], X2_train[y2_train == cls, 1],
                   c=point_colors[cls], s=34, edgecolors="white", linewidths=0.6,
                   label=class_names[cls])
    ax.set_title(f"{'(b)' if name.startswith('单棵') else '(c)'} {name}\n"
                 f"2 特征下测试集准确率 = {a:.2%}", fontsize=11)
    ax.set_xlabel("花瓣长 petal length (cm)", fontsize=10)
    ax.grid(alpha=0.2, linestyle="--")
ax2b.set_ylabel("花瓣宽 petal width (cm)", fontsize=10)
ax2b.legend(loc="upper left", fontsize=8)

fig2.suptitle("单棵决策树 vs 随机森林：森林的边界更平滑、更少出现细碎的小块（高方差被平均掉了）",
              fontsize=12)
fig2.tight_layout(rect=(0, 0, 1, 0.92))
compare_path = OUTPUT_DIR / "03_随机森林_单树与森林对比.png"
fig2.savefig(compare_path, dpi=130)
plt.close(fig2)
print("已保存图片：", compare_path)
print(f"只用两个特征时：单棵树测试准确率 = {vis_acc['单棵决策树（不限制深度）']:.4f}，"
      f"随机森林 = {vis_acc['随机森林（100 棵）']:.4f}")

# ---------------------------------------------------------------------------
# 步骤 8：画图 3 —— Bagging 自助采样到底抽到了什么
# ---------------------------------------------------------------------------
print_section("步骤 8：绘图 —— Bagging 自助采样示意")

n_samples = X_train.shape[0]
rng = np.random.RandomState(RANDOM_STATE)
draws = rng.randint(0, n_samples, size=n_samples)      # 有放回抽 N 个
counts = np.bincount(draws, minlength=n_samples)
unique_ratio = float((counts > 0).mean())
print(f"训练集一共 {n_samples} 条样本，做一次自助采样（有放回抽 {n_samples} 次）：")
print(f"  被抽到的**不同**样本数 = {int((counts > 0).sum())} 条（占 {unique_ratio:.2%}）")
print(f"  一次都没被抽到的样本数 = {int((counts == 0).sum())} 条（占 {1 - unique_ratio:.2%}）")
print(f"  被重复抽到多次的样本数 = {int((counts > 1).sum())} 条")
print(f"  理论上限：1 - 1/e = {1 - 1 / np.e:.4f}，即约 63.2% 的不同样本 + 约 36.8% 的袋外样本。")

# 唯一比例随样本量 N 收敛到 1-1/e
sizes = [5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
ratio_curve = []
for s in sizes:
    trials = [float((np.bincount(rng.randint(0, s, size=s), minlength=s) > 0).mean())
              for _ in range(200)]
    ratio_curve.append(float(np.mean(trials)))

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14.6, 5.0))
ax3a.bar(np.arange(1, n_samples + 1), counts, color="#4c72b0", width=0.85)
ax3a.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.3, label="被抽中 1 次")
ax3a.text(n_samples * 0.26, max(counts) * 0.88,
          f"未抽到：{int((counts == 0).sum())} 条\n（{1 - unique_ratio:.1%}）",
          fontsize=10.5, color="#a01c1c")
ax3a.set_title(f"一次自助采样中各训练样本被抽中的次数（N = {n_samples}）", fontsize=11)
ax3a.set_xlabel("训练样本编号", fontsize=10)
ax3a.set_ylabel("被抽中的次数", fontsize=10)
ax3a.grid(axis="y", alpha=0.3, linestyle="--")
ax3a.legend(fontsize=9)

ax3b.plot(sizes, ratio_curve, "o-", color="#2ca02c", markersize=5, label="实测：不同样本占比")
ax3b.axhline(1 - 1 / np.e, color="#d62728", linestyle="--", linewidth=1.4,
             label=f"理论极限 1 - 1/e = {1 - 1 / np.e:.4f}")
ax3b.set_xscale("log")
ax3b.set_title("自助采样被覆盖的样本比例随样本量 N 收敛到 63.2%", fontsize=11)
ax3b.set_xlabel("样本量 N（对数刻度）", fontsize=10)
ax3b.set_ylabel("被抽到的不同样本占比", fontsize=10)
ax3b.set_ylim(0.4, 1.02)
ax3b.grid(alpha=0.3, linestyle="--")
ax3b.legend(fontsize=9)

fig3.suptitle("Bagging 的基石：有放回抽样 → 每棵树看到约 63.2% 的不同样本 + 约 36.8% 的袋外样本",
              fontsize=12)
fig3.tight_layout(rect=(0, 0, 1, 0.93))
bagging_path = OUTPUT_DIR / "03_随机森林_Bagging自助采样.png"
fig3.savefig(bagging_path, dpi=130)
plt.close(fig3)
print("已保存图片：", bagging_path)

# ============================================================================
# ④ 结果解读
# ============================================================================
print_section("④ 结果解读（这些数字到底意味着什么）")

print("1) 单棵树 vs 随机森林（同一份 iris、同一次划分）：")
print(f"   单棵决策树：训练 {tree_train_acc:.4f}，测试 {tree_test_acc:.4f}，"
      f"5 折 CV = {tree_cv.mean():.4f} ± {tree_cv.std():.4f}")
print(f"   随机森林  ：训练 {rf_train_acc:.4f}，测试 {rf_test_acc:.4f}，"
      f"5 折 CV = {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")
print("   在 iris 这种'太容易'的数据上两者都接近满分，差距看不出多少；")
print("   真正的差距体现在**稳定性**上（见下面第 2 点）和难数据上（噪声越大，森林优势越明显）。")
print()
print("2) 换 12 种不同划分的稳定性对比（带噪声合成数据）：")
print(f"   单棵决策树：均值 {np.mean(tree_accs):.4f}，标准差 {np.std(tree_accs):.4f}，"
      f"范围 {min(tree_accs):.4f} ~ {max(tree_accs):.4f}")
print(f"   随机森林  ：均值 {np.mean(rf_accs):.4f}，标准差 {np.std(rf_accs):.4f}，"
      f"范围 {min(rf_accs):.4f} ~ {max(rf_accs):.4f}")
print(f"   森林平均提升 {np.mean(rf_accs) - np.mean(tree_accs):+.4f}，"
      f"标准差从 {np.std(tree_accs):.4f} 降到 {np.std(rf_accs):.4f}。")
print("   这是随机森林最实在的价值：**既提高准确率，又把单棵树的'高方差'平均掉了**。")
print("   在工程上这意味着：结果可复现、上线后不会因为数据轻微变动就大幅抖动。")
print()
print("3) 袋外估计 OOB：")
print(f"   oob_score_ = {rf_clf.oob_score_:.4f}，5 折交叉验证均值 = {rf_cv.mean():.4f}，"
      f"两者基本一致。")
print(f"   它比测试集准确率 {rf_test_acc:.4f} 略低是正常的：测试集只有 {len(y_test)} 条且这份数据太容易，")
print("   单次划分的测试分数偏乐观；而 OOB 用到了全部训练样本，估计更保守、更可靠。")
print("   这说明 OOB 是一个可信的泛化误差估计，而且**不需要单独切出验证集**。")
print("   在数据稀缺的项目里，这个性质非常值钱：所有数据都能参与训练，")
print("   同时还能拿到一个近似交叉验证的评估分数。")
print()
print(f"4) n_estimators 曲线（{n_est_path.name}）：")
print("   三根线（训练 / 测试 / OOB）在树很少（1~30 棵）时上下抖动，过了 30 棵就基本黏在一起：")
print(f"   50~100 棵区间测试集准确率始终是 {n_test_acc[-1]:.4f}（本数据太容易，早就封顶），")
print(f"   OOB 在同期稳定在 {np.nanmin(n_oob_arr[49:]):.4f} ~ {np.nanmax(n_oob_arr[49:]):.4f}，"
      f"波动仅 {np.std(n_oob_arr[49:]):.4f}；")
print(f"   而 10~20 棵这一段 OOB 的波动是 {np.std(n_oob_arr[9:20]):.4f}，明显更大。")
print("   这就是随机森林的'免费午餐'性质：**加树不会让模型变差，只是更慢**；")
print("   与单棵决策树'越深越糟'形成鲜明对照。")
print()
print(f"5) 单树与森林对比图（{compare_path.name}）：")
print("   (a) 特征重要性：森林给出的重要性更'平滑'（各特征都分摊到一些），")
print("       单棵树则更极端（往往是某个特征一枝独秀）。")
print("    · 这不是因为森林'看不清'，而是因为每次分裂只用随机的一部分特征，")
print("      再加上上百棵树的平均，重要性估计的方差更小、更可靠；")
print("   (b)(c) 决策边界：单棵树的边界有明显的小块/锯齿（高方差），")
print("       森林的边界平滑得多 —— 这正是'平均'在几何上的样子。")
print()
print(f"6) Bagging 采样图（{bagging_path.name}）：")
print(f"   一次自助采样只覆盖 {unique_ratio:.1%} 的不同样本，剩下 {1 - unique_ratio:.1%} 完全没抽到；")
print(f"   理论值是 1-1/e = {1 - 1 / np.e:.4f}，即约 63.2% 被覆盖、约 36.8% 成为袋外样本。")
print("   右图显示这个比例随 N 增大稳定收敛到该极限。")
print("   '每棵树看到的数据都不一样'正是集成多样性的来源，")
print("   而'没抽到的那约 36.8%'又顺手变成了免费的验证集（OOB）。")
print()
print("7) 一句话总览：随机森林 = Bagging（自助采样 + 并行训练 + 投票） + 特征随机。")
print("   它用'平均'换来了低方差，用'特征随机'压低了相关性下界，")
print("   代价是牺牲掉决策树最大的优点 —— 可解释性。")

# ============================================================================
# 超参数怎么调
# ============================================================================
# 【超参数怎么调】—— 随机森林实战调参顺序
#
# 好消息：随机森林是"最不怕调参"的模型之一，默认参数往往就已经很强。
# 调参顺序建议如下（前两个几乎总是先动）：
#
# 1. n_estimators（先给足，别抠）：
#    - 直接上 300~500，或者用 OOB / 验证曲线确认曲线已经进入平台即可；
#    - 记住：**加树不会过拟合**，只是变慢；曲线进入平台后再加就是浪费算力；
#    - 大数据时可以先用 100 棵调其它参数，最后再把 n_estimators 提到 500。
#
# 2. max_features（第二优先级，随机森林的灵魂参数）：
#    - 分类起点 'sqrt'；再试 'log2'、0.3、0.5、None(=1.0)；
#    - 判据是 CV 分数：调小 → 树更去相关（方差下界降低）但单棵更弱（偏差上升）；
#    - 特征数很少（如 iris 只有 4 个）时，'sqrt' 和 'log2' 都是 2，效果相同。
#
# 3. min_samples_leaf / min_samples_split（防过拟合）：
#    - 从默认 1 开始，试 2 / 3 / 5；噪声大或样本少时收益明显；
#    - 调大 → 每棵树更平滑，整体偏差上升、方差下降。
#
# 4. max_depth：
#    - 默认 None（单棵树长到底）。这是刻意的：Bagging 平均本身就是正则化，
#      不需要靠剪枝防过拟合；
#    - 只有在训练时间吃不消、或数据噪声极大时才设 10~20。
#
# 5. max_samples（bootstrap 采样比例）：
#    - 默认 None（=全部 N）。调小（如 0.5~0.8）能进一步增大树之间的差异（ρ 更小），
#      在大数据上还能显著提速，值得一试。
#
# 6. oob_score=True（强烈建议打开）：
#    - 它给你一个免费的泛化误差估计，可以直接用来选 max_features / min_samples_leaf，
#      省掉一次交叉验证的计算量；
#    - 注意：必须是 bootstrap=True；样本量太小时 OOB 会偏乐观。
#
# 7. class_weight：
#    - 类别不平衡用 'balanced_subsample'（按每棵树的 bootstrap 样本自适应），
#      通常比 'balanced' 更合适；同时把评估指标换成 F1 / recall。
#
# 8. n_jobs=-1：
#    - 树的训练完全独立，并行几乎线性加速，一定打开。
#
# 9. 想再进一步（课案后续"集成学习"章会展开）：
#    - 换个基学习器 → ExtraTreesClassifier（分裂阈值也随机，更快、方差更低）；
#    - 换 boosting 思路 → GradientBoosting / HistGradientBoosting / XGBoost / LightGBM
#      （串行纠错，通常精度更高，但更容易过拟合、要更小心调参）。
#
# 10. 推荐自动化做法：
#     GridSearchCV(RandomForestClassifier(n_estimators=300, oob_score=True,
#                                          n_jobs=-1, random_state=42),
#                  {"max_features": ["sqrt", "log2", 0.5],
#                   "min_samples_leaf": [1, 2, 3],
#                   "max_depth": [None, 10, 20]},
#                  cv=5, scoring="accuracy", n_jobs=-1)
#     注意 n_jobs 在 GridSearchCV 里是"平行折"，在随机森林里是"平行树"，
#     两层都开 -1 可能会抢核，数据小的时候把 GridSearchCV 的 n_jobs 设为 1 反而更快。
# ============================================================================

print()
print("【完成】05_随机森林.py 运行结束")

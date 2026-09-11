r"""《机器学习》课案 · 03 监督学习-分类 · 04 决策树（Decision Tree）

对应课案章节
-----------
《机器学习》课案「分类」章 —— 决策树（Decision Tree）：
课案原文位于 `.course_extract/机器学习_课案.md` 第 570~630 行。
覆盖内容：决策树核心思想（对特征做一系列是/否判断）、
课案示例代码（load_iris + DecisionTreeClassifier(max_depth=3, random_state=42)）、
决策树分裂标准表（基尼系数 / 信息熵 / 分类误差，含公式与说明）。

本节知识点
---------
1. 递归划分（recursive partitioning）：不断选一个特征、选一个阈值，把当前节点一分为二，
   直到满足停止条件。
2. 三种特征选择标准（不纯度 impurity）：
   基尼系数 Gini = 1 - Σ p_i²、信息熵 Entropy = -Σ p_i·log₂(p_i)、
   分类误差 Error = 1 - max(p_i)，以及三者的对比与选用理由。
3. CART 算法：sklearn 用的是 CART，**生成的是二叉树**（每次分裂只分两支），
   并且是**贪心分裂**（每个节点只看当前最优，不回头，不保证全局最优）。
4. 特征重要性 feature_importances_ 的计算口径：该特征带来的不纯度下降、
   按到达节点的样本数加权，最后归一化到和为 1。
5. 预剪枝（pre-pruning）与后剪枝（post-pruning，代价复杂度剪枝 ccp_alpha）。
6. 决策树的优缺点：可解释、无需标准化、天然多分类；但极易过拟合、对数据扰动敏感。

运行方式
-------
PowerShell（路径含中文，必须加引号并用 & 调用）：

    $env:PYTHONUTF8='1'; & 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Machine_Learning\03_监督学习_分类\04_决策树.py'

输出：
    - 控制台：四段式讲解 + 特征重要性 + 树规模 + 预剪枝/后剪枝实验 + 中文解读
    - 图片： Machine_Learning/output/03_决策树_特征重要性.png
            Machine_Learning/output/03_决策树_树结构.png
            Machine_Learning/output/03_决策树_剪枝与复杂度.png
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")            # 无界面后端，避免 plt.show() 阻塞
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import numpy as np
from sklearn.datasets import load_iris, make_classification
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

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
# 【1.1 模型形式：一棵树就是一堆"如果……那么……"】
#   决策树把特征空间切成一个个矩形区域，每个区域给一个类别标签。
#   从根节点出发，每个内部节点是一个判断："特征 j 是否 ≤ 阈值 t？"
#   答"是"走左子树，答"否"走右子树，一路走到叶节点，叶节点的多数类就是预测结果。
#   例（iris 上训练出来的真实规则）：
#       如果 花瓣宽 ≤ 0.8   → 山鸢尾
#       否则如果 花瓣长 ≤ 4.95 → 变色鸢尾
#       否则                → 维吉尼亚鸢尾
#   这种"白盒"特性是决策树最大的卖点：**规则可以被人类直接读懂和审查**。
#
# 【1.2 第一步：怎么衡量一个节点"纯不纯"？（三种不纯度指标）】
#   设当前节点里有 K 个类别，第 i 类样本占比为 p_i（Σp_i = 1）。
#
#   (a) 基尼系数（Gini impurity）—— sklearn 的默认标准
#           Gini(t) = 1 - Σ_{i=1..K} p_i²
#       直觉：从该节点里随机抽两个样本，它们类别不一致的概率。
#       完全纯净（只有一类，p=1）时 Gini = 1 - 1 = 0；
#       二分类各占一半时 Gini = 1 - (0.25 + 0.25) = 0.5，取到最大值。
#       优点：只涉及平方运算，**不需要算对数**，计算快，是实践中的首选。
#
#   (b) 信息熵（Entropy / information entropy）
#           Entropy(t) = - Σ_{i=1..K} p_i · log₂(p_i)        （约定 0·log0 = 0）
#       直觉：该节点类别分布的"混乱程度"，也就是把它确定下来需要的平均信息量（比特）。
#       完全纯净时 Entropy = 0；二分类各占一半时 Entropy = 1（bit），取到最大值。
#       优点：对类别分布的变化更敏感，理论上和信息增益（ID3/C4.5 的核心）一脉相承；
#       缺点：有对数运算，比 Gini 稍慢；且对不平衡的类别分布更"敏感"（有时是缺点）。
#
#   (c) 分类误差（Classification error / misclassification impurity）
#           Error(t) = 1 - max_i(p_i)
#       直觉：若直接把这个节点判成多数类，会错多大比例。
#       完全纯净时 Error = 0；二分类各占一半时 Error = 0.5。
#       缺点：它是**分段线性**的，对"把 40%/60% 改善成 30%/70%"这种纯度提升常常
#             不敏感（很多候选分裂的不纯度下降都是 0），会导致树长得很差。
#       所以在**建树阶段很少用**；但它常出现在**剪枝**的准则里（因为剪枝关心的是错误率）。
#
#   三者的数值对比（二分类，正类占比 p）：
#       p      Gini = 2p(1-p)     Entropy         Error = 1-max(p,1-p)
#       0.0    0.00               0.000           0.00
#       0.1    0.18               0.469           0.10
#       0.3    0.42               0.881           0.30
#       0.5    0.50               1.000           0.50
#   可以看到：三者的**变化趋势一致**，但 Error 在 p 偏离 0.5 时明显更小、更"迟钝"，
#   这就是它不适合建树的根本原因。sklearn 只提供 'gini' 与 'entropy' 两种 criterion，
#   并没有提供分类误差 —— 正是这个原因。
#
# 【1.3 第二步：怎么选分裂点？（贪心 + 不纯度下降）】
#   对当前节点的每一次候选分裂（特征 j，阈值 t），把这个节点分成左右两部分 L、R，
#   定义**不纯度下降（impurity decrease）**：
#
#       ΔI = I(parent) - (n_L/n)·I(L) - (n_R/n)·I(R)
#
#   其中 n 是父节点样本数，n_L、n_R 是左右子节点样本数。
#   CART 的做法：遍历所有特征的所有候选阈值，**挑 ΔI 最大的那个分裂**。
#
#   · "贪心"：每个节点只看眼前最优，选完就往下走，绝不回头重新考虑。
#     好处：训练快，复杂度约 O(n_features · n_samples · log(n_samples))；
#     代价：得到的树不保证是全局最优的树（找全局最优是 NP 难问题）。
#   · "二叉树"：sklearn 的 DecisionTreeClassifier 用的是 CART，
#     无论特征有多少个取值，每次分裂**只产生左右两个孩子**（阈值是"≤ t"）。
#     对比：ID3 / C4.5 可以做多路分裂（一个特征一次分出多支），
#     好处是树更矮，坏处是分裂后每支样本变少、统计不稳。sklearn 只实现了 CART。
#
# 【1.4 停止条件：预剪枝（pre-pruning）】
#   在实际动手前就限制树的生长，参数有：
#     max_depth          ：最大深度，最直观的"刹车"；
#     min_samples_split  ：一个节点至少要有多少样本才允许继续分裂；
#     min_samples_leaf   ：分裂后每个叶子至少要有多少样本；
#     max_features       ：每次分裂只考虑随机抽出的部分特征（随机森林的关键）；
#     min_impurity_decrease：不纯度下降至少要达到多少才值得分裂；
#     max_leaf_nodes     ：限制叶子总数。
#   优点：训练快、直接得到一棵较小的树，不容易过拟合；
#   缺点："目光短浅" —— 某次分裂当下看收益很小，但它可能是后续关键分裂的前提，
#         预剪枝会把它砍掉，导致**欠拟合**。所以预剪枝常常陷入"要么太早停、要么太晚停"。
#
# 【1.5 另一种思路：后剪枝（post-pruning）+ 代价复杂度剪枝】
#   先让树自由长到很复杂，再**自底向上**把"对泛化没贡献"的子树剪掉。
#   sklearn 用的是**代价复杂度剪枝（minimal cost-complexity pruning）**，
#   由 Breiman 在 CART 里提出。定义子树 T 的代价复杂度：
#
#       R_α(T) = R(T) + α · |T|
#
#   其中 R(T) 是训练误差（叶子节点的分类误差之和），|T| 是叶子节点数（树的复杂度），
#   α ≥ 0 是复杂度惩罚系数，就是 sklearn 的 **ccp_alpha**。
#   α = 0 时不惩罚，退化成原始的完全生长树；α 越大，叶子越多被罚得越重，树被剪得越短。
#   最小代价复杂度剪枝的结论很漂亮：**存在一串"最弱环节"待剪子树**，
#   把它们按 α 从小到大依次剪掉，就能得到一系列嵌套的树。
#   sklearn 用 `cost_complexity_pruning_path` 直接给出这串候选 α 值，
#   剩下的工作就是：对每个 α 训练一棵树，用交叉验证挑出最好的那个 α。
#   后剪枝的优点是"先看清楚再决定"，通常比预剪枝得到更准的树，代价是训练更慢。
#
# 【1.6 特征重要性 feature_importances_ 怎么来的】
#   对树里的每一个分裂节点，记录：
#       该节点的样本数占比（n_node / n_total）× 该次分裂的不纯度下降 ΔI
#   把同一个特征在所有节点上的这个量加起来，得到该特征的"总贡献"，
#   最后**归一化**（所有特征加起来等于 1）。
#   两个必须知道的注意点：
#     (1) 它是"训练集上的、基于不纯度"的重要性，**偏向取值多的特征**
#         （比如 ID 号这种高基数特征会被误判为很重要）；
#     (2) 它天然带**随机性**（尤其当 max_features < n_features 时），
#         换一个 random_state 数值就会变，所以别把小数点后第三位当真。
#   更好的替代：permutation_importance（打乱某个特征看性能掉多少），
#   但计算更贵，本脚本用自带的 feature_importances_ 并说明其局限。
#
# 【1.7 优缺点】
#   优点：
#     - 可解释性极强：训练完就是一串人可以读的 if-else 规则，方便和业务专家对齐；
#     - **不需要标准化 / 归一化**！因为它只比较"大小关系"（阈值），不看数值距离；
#     - 天然支持多分类，也能自动处理特征交互（不需要手工做交叉项）；
#     - 能处理数值型和（编码后的）类别型特征，对单调变换不敏感；
#     - 训练与预测都很快。
#   缺点：
#     - **极易过拟合**：不加限制时长到每个叶子只有 1 个样本，训练准确率必然 100%；
#     - **高方差、不稳定**：训练数据稍微变一点（甚至只是换个 random_state），
#       整棵树的结构就可能完全不同 —— 这正是随机森林要解决的问题；
#     - 决策边界只能是**与坐标轴平行**的阶梯形状，对斜向边界要用很多台阶去逼近；
#     - 对类别不平衡敏感（可用 class_weight='balanced' 缓解）。
# ============================================================================


# ============================================================================
# ② sklearn API 关键参数逐个解释
# ============================================================================
# 默认值取自本机 scikit-learn 1.9.0 实测（DecisionTreeClassifier().get_params()）。
#
# criterion : {'gini','entropy','log_loss'}，默认 'gini'
#    含义：衡量节点不纯度（也就是选择分裂点）的标准。
#      - 'gini'：基尼系数 1-Σp_i²，不算对数、最快，默认值，绝大多数情况够用；
#      - 'entropy'：信息熵 -Σp_i·log₂p_i，稍慢，有时会得到更"平衡"的树；
#      - 'log_loss'：交叉熵口径（sklearn 1.1 起支持），主要用于回归树 / 概率校准场景。
#    实测差异：在本脚本的 iris 上，'gini' 与 'entropy' 的准确率几乎一样
#             （见下面的对比表）；不同只在树的结构细节上。
#    选择建议：默认 'gini'；若你希望树的划分更依赖"信息增益"的理论解释，用 'entropy'。
#
# max_depth : int 或 None，默认 None（不限制）
#    含义：树的最大深度，**最常用也最有效的预剪枝参数**。
#    调小（如 2、3）：树矮、规则少、偏差大 → 容易欠拟合；
#    调大（如 20）或 None：树高、规则多、方差大 → 容易过拟合。
#    常用值：3 / 5 / 7 / 10（iris 这种简单数据 max_depth=2 或 3 就够）。
#
# min_samples_split : int 或 float，默认 2
#    含义：一个**内部节点**至少要有多少样本才允许继续分裂。
#    传 int：绝对样本数；传 float（0~1）：占训练集总数的比例。
#    调大（如 20）：样本少的节点不再分裂 → 树更小、更保守；
#    调小（=2，默认）：几乎不限制 → 树可以长得很大。
#
# min_samples_leaf : int 或 float，默认 1
#    含义：分裂之后，**每个叶子节点**至少要留下多少样本。
#    这个参数比 min_samples_split 更"有效"：因为它直接保证了每个叶子的统计稳定性，
#    不会出现"1 个样本独占一个叶子"的极端情况。
#    调大（如 5、10）：叶子更"厚实"、边界更平滑 → 抗噪、可能欠拟合；
#    调小（=1，默认）：允许叶子只有 1 个样本 → 过拟合风险最大。
#    常用值：分类任务 1~20；样本少时用 1~5。
#
# max_features : int / float / {'sqrt','log2'} 或 None，默认 None
#    含义：每次分裂时**只考虑**多少个特征（从全部特征里随机抽）。
#      - None：考虑全部特征（单棵决策树的默认，会挑到当前最优特征）；
#      - 'sqrt'：√n_features 个（随机森林分类的默认值）；
#      - 'log2'：log₂(n_features) 个；
#      - float（0,1]：占全部特征的比例。
#    调小：每次分裂的候选特征变少 → 树更多样、单棵更弱，但整体方差下降；
#          这是**随机森林去相关的关键机制**，对单棵树则是引入随机性。
#    调大（=None）：单棵树更强（训练误差更低），但也更容易过拟合、更不稳定。
#
# ccp_alpha : float（≥0），默认 0.0
#    含义：代价复杂度剪枝的惩罚系数 α（见 ①1.5 节的 R_α(T) = R(T) + α·|T|）。
#    调大：惩罚叶子数量 → 树被剪得更短、更简单 → 可能欠拟合；
#    调小（0.0，默认）：不剪枝 → 得到完全生长的树。
#    实际用法：先跑 cost_complexity_pruning_path 拿到一串候选 α，
#             再对每个 α 用交叉验证评估，挑验证分数最高的那个。
#    注意：ccp_alpha 做的是"后剪枝"，和 max_depth 这类"预剪枝"可以叠加使用，
#          但叠加时两者的效果会互相影响，建议一次只调一个维度。
#
# random_state : int 或 None，默认 None
#    含义：随机种子。DecisionTreeClassifier 的随机性来自"当多个分裂的增益完全相同时，
#          随机挑选一个特征"（以及 max_features < n_features 时的随机抽样）。
#    影响：固定为 42 可以保证每次运行得到完全相同的树；不固定则结果会在小范围内抖动。
#          这也是"决策树不稳定"这一缺点的直接证据。
#
# splitter : {'best','random'}，默认 'best'
#    含义：'best' 在所有候选里挑最优分裂；'random' 随机挑一个最优附近的分裂。
#    'random' 会进一步增加随机性（更抗过拟合，但单棵树更弱），实战中很少改。
#
# class_weight : dict 或 'balanced'，默认 None
#    含义：类别权重，'balanced' 按类别频率反比自动加权，用于类别不平衡场景。
#
# min_impurity_decrease : float，默认 0.0
#    含义：只有不纯度下降 ≥ 这个值，分裂才被接受。调大能让树更早停止生长。
# ============================================================================


print_section("《机器学习》课案 · 分类 · 04 决策树（Decision Tree）")
print("本脚本四段结构： ① 原理与数学推导  ② sklearn API 参数解释  ③ 完整可运行代码  ④ 结果解读")
print("配套图片输出目录：", OUTPUT_DIR)

# ============================================================================
# ③ 完整可运行代码
# ============================================================================

# ---------------------------------------------------------------------------
# 步骤 1：提取数据
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
print("提醒：决策树**不需要标准化**！它只比较特征的大小关系（阈值），")
print("      所以对单调变换（加减乘除、取对数）不敏感，这是它相对 SVM/KNN 的一大优势。")

# ---------------------------------------------------------------------------
# 步骤 2：划分训练集 / 测试集
# ---------------------------------------------------------------------------
print_section("步骤 2：划分训练集/测试集（与课案一致）")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE
)
print(f"训练集 X_train = {X_train.shape}，测试集 X_test = {X_test.shape}")
print("训练集各类数量：", np.bincount(y_train), "，测试集各类数量：", np.bincount(y_test))

# ---------------------------------------------------------------------------
# 步骤 3：运行决策树算法（完全按课案写法）
# ---------------------------------------------------------------------------
print_section("步骤 3：训练决策树 DecisionTreeClassifier(max_depth=3, random_state=42)")

model = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)
model.fit(X_train, y_train)
print("模型已训练完成（决策树的训练是贪心地递归分裂，不需要迭代优化）。")
print(f"树的深度 get_depth()      = {model.get_depth()}")
print(f"叶子个数 get_n_leaves()   = {model.get_n_leaves()}（每个叶子对应一条 if-else 规则）")
print(f"实际用到的特征数 n_features_in_ = {model.n_features_in_}")
print(f"类别 classes_             = {[str(c) for c in model.classes_]}")

# ---------------------------------------------------------------------------
# 步骤 4：预测与评估
# ---------------------------------------------------------------------------
print_section("步骤 4：预测与评估")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("预测结果（前 15 条）：", y_pred[:15])
print("真实结果（前 15 条）：", y_test[:15])
print(f"测试集准确率 = {acc:.4f}（{(y_pred == y_test).sum()} / {len(y_test)} 条正确）")
print(f"训练集准确率 = {accuracy_score(y_train, model.predict(X_train)):.4f}")

print("\n分类报告：")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4, zero_division=0))
print("精确率 precision：预测成该类的样本里有多少真的是该类；")
print("召回率 recall   ：该类真实样本里有多少被找出来；")
print("F1 = 2PR/(P+R)  ：两者的调和平均。")

# ---------------------------------------------------------------------------
# 步骤 5：特征重要性（并解释为什么某个特征最重要）
# ---------------------------------------------------------------------------
print_section("步骤 5：特征重要性 feature_importances_")

importances = model.feature_importances_
print("feature_importances_ 的原始值：", np.round(importances, 4).tolist())
print(f"所有重要性之和 = {importances.sum():.4f}（sklearn 已自动归一化到 1.0）")
print()
print("-" * 76)
print(f"{'排名':<6}{'特征名':<28}{'重要性':>12}{'占比':>12}{'是否被用到':>14}")
print("-" * 76)
order = np.argsort(-importances)
for rank, idx in enumerate(order, start=1):
    used = "是" if importances[idx] > 0 else "否（本树未用到）"
    print(f"{rank:<6}{feature_names[idx]:<28}{importances[idx]:>12.4f}"
          f"{importances[idx]:>12.1%}{used:>14}")
print("-" * 76)

top_idx = int(order[0])
print(f"最重要的特征是【{feature_names[top_idx]}】，重要性 = {importances[top_idx]:.4f}。")
print("为什么是它？")
print("  · 从生物学上讲，花瓣的尺寸比花萼更能区分鸢尾花的品种，")
print("    setosa 的花瓣又短又窄，与另外两类有一道很明显的鸿沟；")
print("  · 从算法上讲，feature_importances_ 统计的是'该特征带来的不纯度下降、")
print("    按到达节点的样本数加权'。谁能在**靠近根节点**的位置、")
print("    用**一次分裂**就把大量样本分开，谁的得分就高；")
print("  · 花瓣宽/花瓣长正好能在根节点附近一次把 setosa 摘出去，所以排名靠前。")
print()
print("⚠ 使用 feature_importances_ 的两个重要提醒：")
print("  ① 它偏向**取值多**的特征（高基数特征容易被高估），对类别型特征要小心；")
print("  ② 它基于训练集的不纯度下降，天然带随机性（尤其 max_features 不为 None 时），")
print("     换 random_state 数值就会变，不要把小数第三位当结论。")
print("  更稳健的替代方案是 permutation_importance（打乱某特征看性能掉多少），")
print("  代价是计算量大约为 特征数 × 重复次数 次额外预测。")

# ---------------------------------------------------------------------------
# 步骤 6：不限制深度 vs max_depth=3 —— 过拟合的直接证据
# ---------------------------------------------------------------------------
print_section("步骤 6：完全生长的树 vs max_depth=3（看过拟合）")

print("同一份数据、同一个 random_state，只改 max_depth：")
print("-" * 92)
print(f"{'max_depth':<12}{'树深度':>8}{'叶子数':>8}{'训练集准确率':>14}{'测试集准确率':>14}"
      f"{'两者差距':>12}{'5折CV均值':>12}")
print("-" * 92)
depth_records = []
for md in [1, 2, 3, 5, 8, None]:
    dt = DecisionTreeClassifier(max_depth=md, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    tr = accuracy_score(y_train, dt.predict(X_train))
    te = accuracy_score(y_test, dt.predict(X_test))
    cv = cross_val_score(dt, X_train, y_train, cv=5).mean()
    depth_records.append((md, dt.get_depth(), dt.get_n_leaves(), tr, te, cv))
    label = "None(不限制)" if md is None else str(md)
    print(f"{label:<12}{dt.get_depth():>8}{dt.get_n_leaves():>8}{tr:>14.4f}{te:>14.4f}"
          f"{tr - te:>12.4f}{cv:>12.4f}")
print("-" * 92)
unlimited = depth_records[-1]
limited = [r for r in depth_records if r[0] == 3][0]
print("读表要点：")
print(f"  · max_depth=None（完全生长）时训练集准确率 = {unlimited[3]:.4f}（满分），"
      f"叶子数 {unlimited[2]} 个；")
print(f"    max_depth=3 时训练集 {limited[3]:.4f}，叶子数只有 {limited[2]} 个 ——")
print("    树大了一倍多，训练集上的收益却几乎为零，说明多出来的那些分裂是在拟合噪声；")
print(f"  · 但要注意：本数据上完全生长树的**测试集**准确率也是 {unlimited[4]:.4f}，")
print("    iris 只有 150 条、4 个特征，三类在花瓣维度上分得非常开，")
print("    **这份数据太容易，过拟合现象在这张表里体现不出来**；")
print("  · 所以下一步（步骤 7）会换一份带噪声的数据，把过拟合的'剪刀差'真正做出来。")
print(f"  · 本组数据上 5 折 CV 均值最高的是 max_depth="
      f"{max(depth_records, key=lambda r: r[5])[0]}"
      f"（CV = {max(r[5] for r in depth_records):.4f}）。")

# 顺带对比 gini 与 entropy 两种分裂标准
print()
print("再对比 criterion='gini'（默认）与 'entropy'（信息熵），其余参数相同：")
print("-" * 76)
print(f"{'criterion':<14}{'训练集准确率':>16}{'测试集准确率':>16}{'叶子数':>10}{'5折CV均值':>14}")
print("-" * 76)
for crit in ["gini", "entropy"]:
    dt = DecisionTreeClassifier(criterion=crit, max_depth=3, random_state=RANDOM_STATE)
    dt.fit(X_train, y_train)
    tr = accuracy_score(y_train, dt.predict(X_train))
    te = accuracy_score(y_test, dt.predict(X_test))
    cv = cross_val_score(dt, X_train, y_train, cv=5).mean()
    print(f"{crit:<14}{tr:>16.4f}{te:>16.4f}{dt.get_n_leaves():>10}{cv:>14.4f}")
print("-" * 76)
print("结论：两种标准在本数据上准确率几乎一样，只有树的结构细节不同。")
print("      gini 不算对数、更快，是默认值；entropy 更适合需要信息增益解释的场合。")

# ---------------------------------------------------------------------------
# 步骤 7：补充实验 A —— 换带噪声的数据，把过拟合真正做出来
# ---------------------------------------------------------------------------
print_section("步骤 7：补充实验 A —— 带噪声数据上的过拟合证据")

print("为什么必须补这一步：iris 三类在花瓣维度上分得太开，完全生长的树在测试集上")
print("也几乎不犯错，过拟合看不出来。真实业务数据几乎都有噪声（标注错误、异常值、")
print("特征不足），那时决策树的过拟合就会非常明显。")
print()
print("用 make_classification 造 300 条、10 个特征、25% 标签被随机翻转的数据：")
X_noisy, y_noisy = make_classification(
    n_samples=300, n_features=10, n_informative=5, n_redundant=0,
    n_clusters_per_class=1, class_sep=1.0, flip_y=0.25, random_state=RANDOM_STATE,
)
Xn_train, Xn_test, yn_train, yn_test = train_test_split(
    X_noisy, y_noisy, test_size=0.3, random_state=RANDOM_STATE, stratify=y_noisy
)
print(f"  训练集 {Xn_train.shape}，测试集 {Xn_test.shape}")
print("-" * 94)
print(f"{'max_depth':<12}{'树深度':>8}{'叶子数':>8}{'训练集准确率':>14}{'测试集准确率':>14}"
      f"{'两者差距':>12}{'5折CV均值':>12}")
print("-" * 94)
noisy_records = []
for md in [1, 2, 3, 5, 8, None]:
    dt = DecisionTreeClassifier(max_depth=md, random_state=RANDOM_STATE)
    dt.fit(Xn_train, yn_train)
    tr = accuracy_score(yn_train, dt.predict(Xn_train))
    te = accuracy_score(yn_test, dt.predict(Xn_test))
    cv = cross_val_score(dt, Xn_train, yn_train, cv=5).mean()
    noisy_records.append((md, dt.get_depth(), dt.get_n_leaves(), tr, te, cv))
    label = "None(不限制)" if md is None else str(md)
    print(f"{label:<12}{dt.get_depth():>8}{dt.get_n_leaves():>8}{tr:>14.4f}{te:>14.4f}"
          f"{tr - te:>12.4f}{cv:>12.4f}")
print("-" * 94)
n_lim = [r for r in noisy_records if r[0] == 3][0]
n_unl = noisy_records[-1]
print("读表要点（这才是教科书里的过拟合长相）：")
print(f"  · max_depth=3：训练集 {n_lim[3]:.4f}，测试集 {n_lim[4]:.4f}，"
      f"差距 {n_lim[3] - n_lim[4]:+.4f}，叶子仅 {n_lim[2]} 个；")
print(f"  · max_depth=None：训练集被顶到 {n_unl[3]:.4f}（满分！），"
      f"测试集反而掉到 {n_unl[4]:.4f}，差距 {n_unl[3] - n_unl[4]:+.4f}；")
print("  · 训练集准确率一路升到 100%，测试集却先升后降 —— 这两条线的'剪刀差'")
print("    就是过拟合的定义：模型开始记住训练集里的**噪声标签**，而这些噪声在新样本上不成立；")
print(f"  · 5 折 CV 均值最高的是 max_depth={max(noisy_records, key=lambda r: r[5])[0]}"
      f"（CV = {max(r[5] for r in noisy_records):.4f}），远高于完全生长树的 "
      f"{n_unl[5]:.4f}。")
print("  · 教训：'训练集 100% 准确率'永远不是好消息，它只说明树足够深、能记住每一个样本。")

# ---------------------------------------------------------------------------
# 步骤 8：补充实验 B —— 后剪枝（代价复杂度剪枝 ccp_alpha）在噪声数据上的效果
# ---------------------------------------------------------------------------
print_section("步骤 8：补充实验 B —— 后剪枝（代价复杂度剪枝 ccp_alpha）")

path = DecisionTreeClassifier(random_state=RANDOM_STATE).cost_complexity_pruning_path(
    Xn_train, yn_train
)
# 路径里可能出现重复的 α（对应同一棵树），用掩码去重，同时保持与 impurities 一一对齐
_keep = np.concatenate([[True], np.diff(path.ccp_alphas) > 0])
ccp_alphas = path.ccp_alphas[_keep]
ccp_impurities = path.impurities[_keep]
print("cost_complexity_pruning_path 给出的候选 α 序列（来自【最弱环节】剪枝）：")
print("  ", np.round(ccp_alphas, 5).tolist())
print("  对应的树不纯度（α 越大，允许的总不纯度越高 = 树越简单）：")
print("  ", np.round(ccp_impurities, 4).tolist())
print()
print("这些 α 的含义：把 α 从小到大依次用上，就会得到一串**嵌套**的树：")
print("  每加一个 α，就剪掉当前'最不值钱'的那棵子树（叶子数随之下降）。")
print("-" * 96)
print(f"{'ccp_alpha':<12}{'树深度':>8}{'叶子数':>8}{'训练集准确率':>14}{'测试集准确率':>14}"
      f"{'两者差距':>12}{'5折CV均值':>12}")
print("-" * 96)
prune_records = []
for a in ccp_alphas:
    dt = DecisionTreeClassifier(ccp_alpha=float(a), random_state=RANDOM_STATE)
    dt.fit(Xn_train, yn_train)
    tr = accuracy_score(yn_train, dt.predict(Xn_train))
    te = accuracy_score(yn_test, dt.predict(Xn_test))
    cv = cross_val_score(dt, Xn_train, yn_train, cv=5).mean()
    prune_records.append((float(a), dt.get_depth(), dt.get_n_leaves(), tr, te, cv))
    print(f"{float(a):<12.5f}{dt.get_depth():>8}{dt.get_n_leaves():>8}{tr:>14.4f}"
          f"{te:>14.4f}{tr - te:>12.4f}{cv:>12.4f}")
print("-" * 96)
best_prune = max(prune_records, key=lambda r: r[5])
print(f"5 折 CV 均值最高的是 ccp_alpha = {best_prune[0]:.5f}"
      f"（CV = {best_prune[5]:.4f}，叶子数 {best_prune[2]}，训练集 {best_prune[3]:.4f}）；")
print(f"完全不剪枝（ccp_alpha=0）时 CV = {prune_records[0][5]:.4f}，叶子数 {prune_records[0][2]}。")
print(f"α 取到最大（{prune_records[-1][0]:.5f}）时树被剪成只有 1 个根节点，"
      f"CV 掉到 {prune_records[-1][5]:.4f} —— 这是剪过头的欠拟合。")
print()
print("后剪枝的完整套路（推荐照这个流程做）：")
print("  ① 先训练一棵完全不限制的树（max_depth=None）；")
print("  ② 用 cost_complexity_pruning_path 拿到候选 α 序列；")
print("  ③ 对每个 α 训练一棵树，用 cross_val_score 评估；")
print("  ④ 取 CV 均值最高（或'均值 - 1 倍标准差'最高）的那个 α，")
print("     再用**全部训练数据**重新训练最终模型。")
print("  它比预剪枝更稳（先让树看够数据再决定剪哪里），代价是训练时间变长。")

# ---------------------------------------------------------------------------
# 步骤 9：画图 1 —— 特征重要性条形图
# ---------------------------------------------------------------------------
print_section("步骤 9：绘图 —— 特征重要性条形图（iris 上的 max_depth=3 模型）")

fig1, ax1 = plt.subplots(figsize=(8.8, 5.2))
sorted_idx = np.argsort(importances)          # 从小到大，条形图从下往上画
bar_colors = plt.get_cmap("viridis")(np.linspace(0.25, 0.85, len(sorted_idx)))
bars = ax1.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx],
                color=bar_colors, edgecolor="white")
for b, v in zip(bars, importances[sorted_idx]):
    ax1.text(v + 0.008, b.get_y() + b.get_height() / 2, f"{v:.4f}  ({v:.1%})",
             va="center", fontsize=10)
ax1.set_xlim(0, max(importances) * 1.30 if max(importances) > 0 else 1.0)
ax1.set_title("决策树特征重要性（max_depth=3，各特征重要性之和 = 1）\n"
              "数值 = 该特征带来的不纯度下降、按样本数加权后归一化", fontsize=11.5)
ax1.set_xlabel("特征重要性 feature_importances_", fontsize=11)
ax1.set_ylabel("特征名", fontsize=11)
ax1.grid(axis="x", alpha=0.3, linestyle="--")
fig1.tight_layout()
importance_path = OUTPUT_DIR / "03_决策树_特征重要性.png"
fig1.savefig(importance_path, dpi=130)
plt.close(fig1)
print("已保存图片：", importance_path)

# ---------------------------------------------------------------------------
# 步骤 10：画图 2 —— 树结构图（plot_tree）
# ---------------------------------------------------------------------------
print_section("步骤 10：绘图 —— 树结构图（sklearn.tree.plot_tree）")

fig2, ax2 = plt.subplots(figsize=(24.0, 13.0))   # 图要大，否则中文/阈值文字会重叠
plot_tree(
    model,
    ax=ax2,
    filled=True,                 # 按多数类着色，颜色越深越纯
    feature_names=feature_names,  # 显示特征名
    class_names=class_names,      # 显示类别名
    rounded=True,
    fontsize=8,
    precision=3,
)
ax2.set_title("决策树结构（max_depth=3）——每个方框内：分裂条件 / gini 不纯度 / 样本数 / 类别分布 / 预测类别",
              fontsize=15)
fig2.tight_layout()
tree_path = OUTPUT_DIR / "03_决策树_树结构.png"
fig2.savefig(tree_path, dpi=110, bbox_inches="tight")
plt.close(fig2)
print("已保存图片：", tree_path)

print("\n把上面这棵树翻译成人能读懂的规则（用 sklearn 自带的 export_text 口径描述）：")

# 手工把树导出成中文伪代码，便于课堂讲解（不依赖 export_text 的英文输出）
def describe_tree(tree_model, node_id=0, prefix=""):
    """把训练好的树递归翻译成中文 if-else 伪代码。"""
    lines = []
    left = tree_model.tree_.children_left[node_id]
    right = tree_model.tree_.children_right[node_id]
    n_node_samples = tree_model.tree_.n_node_samples[node_id]
    value = tree_model.tree_.value[node_id][0]
    if left == right:  # 叶节点
        cls = class_names[int(np.argmax(value))]
        dist = np.round(value / value.sum(), 3).tolist()
        lines.append(f"{prefix}→ 预测：{cls}（本叶子 {n_node_samples} 条样本，类别分布 {dist}）")
        return lines
    feat = feature_names[tree_model.tree_.feature[node_id]]
    thr = tree_model.tree_.threshold[node_id]
    lines.append(f"{prefix}如果 {feat} ≤ {thr:.3f}：  （本节点共 {n_node_samples} 条样本）")
    lines.extend(describe_tree(tree_model, left, prefix + "    "))
    lines.append(f"{prefix}否则（{feat} > {thr:.3f}）：")
    lines.extend(describe_tree(tree_model, right, prefix + "    "))
    return lines


for line in describe_tree(model):
    print("   ", line)

# ---------------------------------------------------------------------------
# 步骤 11：画图 3 —— 剪枝与复杂度（带噪声数据上，准确率随 max_depth / ccp_alpha 变化）
# ---------------------------------------------------------------------------
print_section("步骤 11：绘图 —— 剪枝与模型复杂度（带噪声数据）")

# 注意：这里用步骤 7 造的**带噪声数据**，因为 iris 上训练集和测试集都会顶到 1.0，
#       画出来两条线重合、看不出任何信息。
depth_axis = list(range(1, 16))
depth_train, depth_test, depth_cv = [], [], []
for md in depth_axis:
    dt = DecisionTreeClassifier(max_depth=md, random_state=RANDOM_STATE)
    dt.fit(Xn_train, yn_train)
    depth_train.append(accuracy_score(yn_train, dt.predict(Xn_train)))
    depth_test.append(accuracy_score(yn_test, dt.predict(Xn_test)))
    depth_cv.append(cross_val_score(dt, Xn_train, yn_train, cv=5).mean())
depth_full = DecisionTreeClassifier(max_depth=None, random_state=RANDOM_STATE).fit(Xn_train, yn_train)
depth_full_train = accuracy_score(yn_train, depth_full.predict(Xn_train))
depth_full_test = accuracy_score(yn_test, depth_full.predict(Xn_test))
depth_axis_full = [int(depth_full.get_depth())]

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(15.4, 5.4))
ax3a.plot(depth_axis, depth_train, "o-", color="#d62728", markersize=4, label="训练集准确率")
ax3a.plot(depth_axis, depth_test, "s-", color="#1f77b4", markersize=4, label="测试集准确率")
ax3a.plot(depth_axis, depth_cv, "^-", color="#2ca02c", markersize=4, label="5 折交叉验证准确率")
ax3a.scatter(depth_axis_full, [depth_full_train], marker="*", s=210, color="#d62728",
             zorder=5, label="完全生长树（训练集）")
ax3a.scatter(depth_axis_full, [depth_full_test], marker="*", s=210, color="#1f77b4",
             zorder=5, label="完全生长树（测试集）")
ax3a.annotate("训练集 100%\n测试集反而掉下来", xy=(depth_axis_full[0], depth_full_train),
              xytext=(1.3, 0.925), fontsize=9.5, color="#8c1b1b",
              arrowprops=dict(arrowstyle="->", color="#8c1b1b"))
ax3a.set_title("预剪枝：准确率随 max_depth 的变化（★ = 完全不限制深度）", fontsize=11.5)
ax3a.set_xlabel("max_depth（预剪枝）", fontsize=10)
ax3a.set_ylabel("准确率 accuracy", fontsize=10)
ax3a.grid(alpha=0.3, linestyle="--")
ax3a.legend(loc="lower left", fontsize=8.5)

pr_alpha = [r[0] for r in prune_records]
pr_train = [r[3] for r in prune_records]
pr_test = [r[4] for r in prune_records]
pr_cv = [r[5] for r in prune_records]
pr_leaves = [r[2] for r in prune_records]
ax3b.plot(pr_alpha, pr_train, "o-", color="#d62728", markersize=4, label="训练集准确率")
ax3b.plot(pr_alpha, pr_test, "s-", color="#1f77b4", markersize=4, label="测试集准确率")
ax3b.plot(pr_alpha, pr_cv, "^-", color="#2ca02c", markersize=4, label="5 折交叉验证准确率")
ax3b.axvline(best_prune[0], color="#2ca02c", linestyle=":", linewidth=1.3,
             label=f"CV 最优 α = {best_prune[0]:.4f}")
ax3b2 = ax3b.twinx()
ax3b2.plot(pr_alpha, pr_leaves, "v--", color="#9467bd", markersize=4, label="叶子数（右轴）")
ax3b2.set_ylabel("叶子节点个数", fontsize=10, color="#9467bd")
ax3b2.tick_params(axis="y", labelcolor="#9467bd")
ax3b.set_title("后剪枝：准确率随 ccp_alpha 的变化（α 越大树越小）", fontsize=11.5)
ax3b.set_xlabel("ccp_alpha（代价复杂度惩罚系数）", fontsize=10)
ax3b.set_ylabel("准确率 accuracy", fontsize=10)
ax3b.grid(alpha=0.3, linestyle="--")
lines_a, labels_a = ax3b.get_legend_handles_labels()
lines_b, labels_b = ax3b2.get_legend_handles_labels()
ax3b.legend(lines_a + lines_b, labels_a + labels_b, loc="upper right", fontsize=8)

fig3.suptitle("决策树的两种剪枝（带噪声数据）：预剪枝限制 max_depth，后剪枝用 ccp_alpha 剪掉最弱子树",
              fontsize=12.5)
fig3.tight_layout(rect=(0, 0, 1, 0.93))
prune_path = OUTPUT_DIR / "03_决策树_剪枝与复杂度.png"
fig3.savefig(prune_path, dpi=130)
plt.close(fig3)
print("已保存图片：", prune_path)

# ============================================================================
# ④ 结果解读
# ============================================================================
print_section("④ 结果解读（这些数字到底意味着什么）")

print(f"1) max_depth=3 的测试集准确率 = {acc:.4f}，树深度 = {model.get_depth()}，"
      f"叶子 = {model.get_n_leaves()} 个。")
print(f"   也就是说：整棵模型只用 {model.get_n_leaves()} 条 if-else 规则，"
      f"就把 {len(y_test)} 条测试样本分类到 {acc:.2%} 的正确率。")
print("   这种'规则数量'和'可读性'是决策树相对 SVM、神经网络最大的优势。")
print()
print("2) 特征重要性给出的结论：")
for rank, idx in enumerate(order, start=1):
    print(f"   第{rank}名 {feature_names[idx]:<24} {importances[idx]:.4f}（{importances[idx]:.1%}）")
print("   注意 'petal'（花瓣）系列明显比 'sepal'（花萼）系列重要 ——")
print("   这与植物学常识一致：setosa 的花瓣和另外两类差别极大。")
print("   也就是说，模型学到的东西是**可以用领域知识解释**的，这就是白盒模型的价值。")
print()
print("3) 过拟合的直接证据（要用带噪声的数据才看得见）：")
print(f"   iris 上：max_depth=None 训练 {unlimited[3]:.4f} / 测试 {unlimited[4]:.4f}；")
print(f"            max_depth=3    训练 {limited[3]:.4f} / 测试 {limited[4]:.4f}")
print("            → 这份数据太容易，两种深度都接近满分，**看不出过拟合**；")
print(f"   噪声数据上：max_depth=3    训练 {n_lim[3]:.4f} / 测试 {n_lim[4]:.4f}"
      f"（差距 {n_lim[3] - n_lim[4]:+.4f}，叶子 {n_lim[2]} 个）")
print(f"               max_depth=None 训练 {n_unl[3]:.4f} / 测试 {n_unl[4]:.4f}"
      f"（差距 {n_unl[3] - n_unl[4]:+.4f}，叶子 {n_unl[2]} 个）")
print(f"            → 训练集被顶到 {n_unl[3]:.4f}，测试集反而从 {n_lim[4]:.4f} 掉到 "
      f"{n_unl[4]:.4f}，这才是过拟合的标准长相。")
print("   结论：决策树不加限制时一定会长到训练误差为 0（每个叶子只剩 1 个样本），")
print("         **判断过拟合永远看'训练与验证的差距'，而不是看训练准确率本身。**")
print()
print(f"4) 树结构图（{tree_path.name}）：")
print("   每个方框里依次是：分裂条件 / gini 不纯度 / 样本数 / 各类样本分布 / 预测类别。")
print("   颜色越深表示该节点越纯（gini 越接近 0）；叶子节点的颜色就是它给出的预测。")
print("   注意根节点的那一次分裂：它一定是最能降低不纯度的那一刀 —— 这就是贪心分裂的起点。")
print()
print(f"5) 剪枝与复杂度图（{prune_path.name}，基于带噪声数据）：")
print("   左图：max_depth 从 1 增大时训练集准确率一路升到 100%（红★），")
print("         而测试集与 CV 准确率在深度 2~3 附近见顶后就开始下滑 —— 越深越差；")
print("   右图：ccp_alpha 从 0 增大时叶子数（紫虚线）单调下降，")
print(f"         准确率在 α ≈ {best_prune[0]:.5f} 处达到最好（CV = {best_prune[5]:.4f}，"
      f"叶子 {best_prune[2]} 个），")
print(f"         再往下剪（α = {prune_records[-1][0]:.5f}）树只剩 1 个根节点，"
      f"CV 掉到 {prune_records[-1][5]:.4f} —— 这就是剪过头的欠拟合。")
print("   两张图讲的是同一件事：**模型复杂度要停在一个恰到好处的位置**。")
print()
print("6) 一句话总览：决策树 = 递归划分 + 贪心选最优分裂 + 不纯度（gini/entropy）度量。")
print("   它不需要标准化、天然多分类、可解释性无敌，但**极不稳定、极易过拟合**；")
print("   正是为了治这两个毛病，才有了下一节的随机森林。")

# ============================================================================
# 超参数怎么调
# ============================================================================
# 【超参数怎么调】—— 决策树实战调参顺序
#
# 0. 前置说明：决策树不需要标准化 / 归一化，所以 Pipeline 里不需要 scaler。
#    这一步就省掉了 KNN / SVM 最麻烦的环节。
#
# 1. max_depth（第一优先级，最直观）：
#    - 从 3 开始，往上试 3 / 5 / 7 / 10 / None；
#    - 判据：训练准确率 − CV 准确率 > 0.05 就要考虑减小深度；
#    - 本脚本 iris 上：max_depth=3 的 CV 均值与完全生长的树持平甚至更好，
#      说明这份数据根本不需要深树。
#
# 2. min_samples_leaf（第二优先级，比 min_samples_split 更有用）：
#    - 它保证每个叶子有足够样本，直接抑制"1 个样本一个叶子"的过拟合；
#    - 从 1 开始，试 2 / 5 / 10 / 20；分类任务常用 1~20；
#    - 与 max_depth 有替代关系：想让树变简单，两个都可以调，但**一次只调一个**。
#
# 3. min_samples_split：
#    - 通常设为 min_samples_leaf 的 2 倍左右（如 leaf=5 → split=10）；
#    - 单独调的收益一般不如 min_samples_leaf 明显。
#
# 4. max_features：
#    - 单棵决策树默认 None（全部特征），保持这个值即可；
#    - 只有在做随机森林 / ExtraTrees 时才需要调小（分类默认 'sqrt'）；
#    - 对单棵树把 max_features 调小，只是引入随机性、降低单棵强度，一般不划算。
#
# 5. ccp_alpha（后剪枝，推荐用"路径 + 交叉验证"自动挑）：
#    path = DecisionTreeClassifier(random_state=42).cost_complexity_pruning_path(X_tr, y_tr)
#    for a in path.ccp_alphas: 用 cross_val_score 评估 DecisionTreeClassifier(ccp_alpha=a)
#    选 CV 均值最大的 α，再用全量训练集重新训练一次。
#    注意：α=0 是完全生长的树；α 太大树会退化成只有一个根节点（永远预测多数类）。
#
# 6. criterion：
#    - 默认 'gini'；如果想让树更"平衡"或需要信息增益的解释，可试 'entropy'；
#    - 本脚本实测两者在本数据上准确率几乎相同，不值得花时间纠结。
#
# 7. class_weight：
#    - 类别不平衡时用 'balanced'；同时把评估指标换成 F1 / recall，别只看 accuracy。
#
# 8. random_state：
#    - 一定要固定（本脚本固定 42），否则每次训练的树结构都不一样；
#    - 反过来，如果你想知道模型稳不稳，可以换几个 random_state 看 CV 分数的波动 ——
#      波动大正是决策树"高方差"的体现，也是随机森林存在的理由。
#
# 9. 推荐自动化做法：
#    GridSearchCV(DecisionTreeClassifier(random_state=42),
#                 {"max_depth": [3, 5, 7, None],
#                  "min_samples_leaf": [1, 2, 5, 10],
#                  "criterion": ["gini", "entropy"]},
#                 cv=5, scoring="accuracy", n_jobs=-1)
#    —— 决策树训练非常快，网格搜索完全负担得起。
# ============================================================================

print()
print("【完成】04_决策树.py 运行结束")

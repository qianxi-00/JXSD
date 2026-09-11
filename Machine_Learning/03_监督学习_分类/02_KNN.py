r"""《机器学习》课案 · 03 监督学习-分类 · 02 K-近邻（KNN）

对应课案章节
-----------
《机器学习》课案「分类」章 —— K-近邻（KNN）：
课案原文位于 `.course_extract/机器学习_课案.md` 第 419~492 行。
覆盖内容：KNN 核心思想（算距离 → 找最近的 K 个邻居 → 投票定类）、
课案示例代码（load_iris 全量三分类 + 标准化 + n_neighbors=3）、
KNN 距离度量表（欧氏 / 曼哈顿 / 切比雪夫 / 闵可夫斯基，含公式与适用场景）、
以及课案特别强调的"闵可夫斯基距离是总公式，p=1 是曼哈顿、p=2 是欧氏、
p 越大越接近切比雪夫"这一结论。

本节知识点
---------
1. 懒惰学习（lazy learning）：训练阶段几乎不做计算，把训练集整份记住；
   预测时才现算距离，因此"训练快、预测慢、内存占用大"。
2. 分类流程：计算待预测样本与所有训练样本的距离 → 取距离最小的 K 个邻居 →
   按邻居标签投票（多数表决）或加权投票（权重 = 1/距离）→ 得到预测类别。
3. 四种距离度量：欧氏、曼哈顿、切比雪夫、闵可夫斯基（含 p 的三种取值关系）。
4. 为什么 KNN 必须做标准化：距离是各维度差值的直接叠加，量纲大的特征会主导距离。
5. K 的取值对模型复杂度的影响：K 小 → 决策边界碎、方差大 → 过拟合；
   K 大 → 边界平滑、偏差大 → 欠拟合。

运行方式
-------
PowerShell（路径含中文，必须加引号并用 & 调用）：

    $env:PYTHONUTF8='1'; & 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Machine_Learning\03_监督学习_分类\02_KNN.py'

输出：
    - 控制台：四段式讲解 + 不同 K 的准确率表 + 距离度量对比 + 中文解读
    - 图片： Machine_Learning/output/03_KNN_K值与准确率曲线.png
            Machine_Learning/output/03_KNN_决策边界.png
            Machine_Learning/output/03_KNN_距离度量与权重对比.png
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
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

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
# 【1.1 一句话原理】
#   KNN 的核心思想：**近朱者赤，近墨者黑**。
#   要判断一个新样本属于哪一类，就看看它周围最近的 K 个已标注邻居都是什么类，
#   然后少数服从多数。
#
# 【1.2 懒惰学习（lazy learning）—— KNN 最特别的地方】
#   大多数算法（逻辑回归、SVM、决策树）都有明确的"训练"阶段：
#   在 fit() 里反复迭代，把知识压缩成一组参数（权重 / 支持向量 / 树结构），
#   训练完之后原始训练数据就可以丢掉了。
#   KNN 恰好相反：
#     - fit() 阶段什么都不算，只是**把整份训练集原封不动地记下来**（复杂度 O(1)）；
#     - predict() 阶段才现算：每个待预测样本都要和**全部 N 个训练样本**算距离，
#       再排序取前 K 个 —— 复杂度 O(N·d)，N 大时非常慢；
#     - 因此它必须**常驻保存全部训练数据**，内存开销大。
#   这种"不学习、把事推到预测时再做"的策略就叫**懒惰学习 / 基于实例的学习**
#   （instance-based learning）。对比：逻辑回归、SVM、决策树属于**急切学习**
#   （eager learning），先用训练数据换一个模型。
#   一句话：KNN 是"用空间（内存、预测时间）换时间（训练时间）"的算法。
#
# 【1.3 预测流程（四步）】
#   第 1 步：算距离。对待预测样本 x，算出它与每一个训练样本 x_i 的距离 d(x, x_i)。
#   第 2 步：取邻居。把距离从小到大排序，取前 K 个训练样本作为"邻居"。
#   第 3 步：投票。看这 K 个邻居的标签：
#            多数表决（uniform）：出现次数最多的类别获胜；
#            加权投票（distance）：每个邻居的票数权重 = 1 / d(x, x_i)，
#                               离得越近的话越有分量。
#   第 4 步：输出。得票最高的类别就是预测结果（分类）；
#            回归任务则取 K 个邻居目标值的平均（KNeighborsRegressor）。
#
#   注意 K=1 时的特例：新样本直接被贴上最近那个邻居的标签。
#   此时训练集上的准确率永远是 100%（每个训练点最近的邻居就是它自己，距离 0），
#   这既是"零训练误差"，也是严重过拟合的直接证据。
#
# 【1.4 四种距离度量（课案表格的正式写法）】
#
#   (a) 欧氏距离（Euclidean distance）—— L2 范数，最常用
#           d(x, y) = sqrt( Σ_{i=1..n} (x_i - y_i)^2 )
#       几何意义：两点之间的**直线距离**。
#       适用：连续变量、各特征尺度相同（身高、体重、年龄、温度、价格）。
#       注意：不做标准化时，量纲大的特征会主导整个距离。
#
#   (b) 曼哈顿距离（Manhattan / cityblock distance）—— L1 范数
#           d(x, y) = Σ_{i=1..n} |x_i - y_i|
#       几何意义：每个维度上的差值直接相加，像在城市街区里只能横走或竖走。
#       适用：高维数据、城市路径距离；对个别维度的巨大差异不如欧氏敏感（更稳健）。
#
#   (c) 切比雪夫距离（Chebyshev distance）—— L∞ 范数
#           d(x, y) = max_{i=1..n} |x_i - y_i|
#       几何意义：所有维度里，差得最大的那一个维度决定距离。
#       适用：棋盘移动类问题（国际象棋的国王可以横、竖、斜各走一格，
#             所以任意两格之间的最少步数就是切比雪夫距离）。
#
#   (d) 闵可夫斯基距离（Minkowski distance）—— 统一的总公式
#           d_p(x, y) = ( Σ_{i=1..n} |x_i - y_i|^p )^(1/p)     , p ≥ 1
#       它是上面三种距离的**父类**：
#           p = 1        →  d_1 = Σ|x_i - y_i|                    = 曼哈顿距离
#           p = 2        →  d_2 = sqrt(Σ(x_i - y_i)^2)             = 欧氏距离
#           p → +∞       →  d_∞ = max_i|x_i - y_i|                 = 切比雪夫距离
#       p 越大，距离就越被"差得最大的那个维度"支配；p 越小，各维度贡献越平均。
#       sklearn 里用 metric="minkowski" + p=... 来使用它。
#
# 【1.5 为什么 KNN 必须做标准化？（本脚本最重要的工程要点）】
#   KNN 完全依赖距离，而距离是**各维度差值的直接叠加**，没有任何"权重学习"来帮忙。
#   举例：iris 的花瓣长单位是 cm，数值范围 1.0~6.9；如果某个特征范围是 0~10000，
#   那么后者的差值会完全淹没前者 —— 相当于把其它维度全部忽略了。
#   标准化后每个特征都变成"偏离均值多少倍标准差"，量纲统一、地位平等，
#   距离才真正反映"样本之间的相似程度"。
#   结论：**只要是基于距离的算法（KNN、K-Means、SVM-RBF），第一步永远是标准化。**
#
# 【1.6 K 的取值如何影响模型复杂度（本脚本用图直观展示）】
#   K = 1：只用最近的一个邻居 → 决策边界紧贴每一个训练点、非常破碎 →
#          训练准确率 100%，测试准确率偏低 → **高方差、过拟合**。
#   K 增大：邻居范围变大，相当于在更大邻域上做平滑 → 边界变平滑 → 方差下降；
#          但 K 太大时会把很远的、甚至别的类的点也拉进来投票 →
#          边界过度平滑、细节丢失 → **高偏差、欠拟合**。
#   极端的 K = N（全部训练样本）：无论输入什么，都预测"训练集里最多的那一类"，
#          模型完全丧失分辨能力。
#   经验：二分类取奇数 K 以避免平票；K 一般从 sqrt(N) 附近开始搜
#        （N=105 时约 10），常用范围 3~20，用交叉验证确定。
#
# 【1.7 优缺点】
#   优点：
#     - 思想极其直观，几乎不需要数学推导，无需训练即可使用；
#     - 天然支持多分类，也能做回归（取邻居均值）；
#     - 非参数方法，不对数据分布做强假设，能拟合任意形状的决策边界。
#   缺点：
#     - 预测慢（要算 N 次距离），数据量大时不可接受；
#     - 必须保存全部训练集，内存开销大；
#     - 对特征尺度、无关特征、噪声标签极其敏感（一个错标样本就能带偏一片区域）；
#     - **维度灾难**：维度升高后样本变得稀疏，"最近邻"也不再近，效果急剧下降；
#     - 类别不平衡时，多数类会垄断投票结果（可用 weights="distance" 缓解）。
# ============================================================================


# ============================================================================
# ② sklearn API 关键参数逐个解释
# ============================================================================
# 默认值取自本机 scikit-learn 1.9.0 实测（KNeighborsClassifier().get_params()）。
#
# n_neighbors : int，默认 5
#    含义：邻居个数 K，**KNN 最重要的超参数**。
#    调小（K=1）：决策边界贴合训练点 → 训练准确率高、方差大 → 过拟合；
#    调大（K=50）：边界平滑 → 偏差大 → 欠拟合。
#    常用值：3 / 5 / 7 / 11 / 15（二分类优先奇数，避免平票）；
#            也可以从 sqrt(n_samples) 附近开始搜索。
#
# weights : {'uniform','distance'} 或可调用对象，默认 'uniform'
#    含义：投票权重。
#      - 'uniform'：所有邻居一票等权（简单多数表决）；
#      - 'distance'：权重 = 1/距离，越近的邻居票越重，能缓解"远处邻居瞎投票"；
#      - 也可以传自定义函数，输入距离数组返回权重数组。
#    影响：'distance' 通常在不平衡数据或 K 偏大时更稳，但对重复点（距离为 0）
#          要小心（sklearn 会给距离 0 的点同等权重）。
#    常用值：先试 'uniform'，无效再换 'distance'。
#
# metric : str 或可调用对象，默认 'minkowski'
#    含义：距离度量方式。常用取值：
#      - 'minkowski'：闵可夫斯基距离（配合 p 参数使用，是默认值）；
#      - 'euclidean'：欧氏距离（等价于 minkowski + p=2）；
#      - 'manhattan' / 'cityblock'：曼哈顿距离（等价于 p=1）；
#      - 'chebyshev'：切比雪夫距离（等价于 p=∞）；
#      - 'cosine' / 'hamming' 等也支持（文本、稀疏特征常用 cosine）。
#    影响：换 metric 相当于换了"什么叫相似"的定义，是**领域知识**的体现，不是调参。
#    常用值：数值特征默认 'minkowski'；文本向量用 'cosine'。
#
# p : int，默认 2
#    含义：当 metric='minkowski' 时的幂次。
#      p=1 → 曼哈顿距离；p=2 → 欧氏距离（默认）；p→∞ → 切比雪夫距离。
#    注意：sklearn 里想用切比雪夫就直接 metric='chebyshev'（p 不能真写无穷大）。
#    调大 p：越来越只看"差得最大的那个维度"，对离群维度更敏感；
#    调小 p：各维度差异被平均，对大量小差异更敏感。
#
# algorithm : {'auto','ball_tree','kd_tree','brute'}，默认 'auto'
#    含义：**只在算距离时用于加速的索引结构**，不改变预测结果（理论上）。
#      - 'brute'：老老实实算所有距离，O(N·d)，最慢但最稳；
#      - 'kd_tree'：按维度切分空间的二叉搜索树，低维（d<20）时快；
#      - 'ball_tree'：用超球体分割，维度略高时比 kd_tree 好；
#      - 'auto'：让 sklearn 根据数据自动挑（默认，绝大多数情况不用改）。
#    影响：只影响**速度**，不影响精度（浮点误差级别差别）；维度很高时只能退回 'brute'。
#
# n_jobs : int，默认 None
#    含义：并行计算用的 CPU 核数。None 表示 1；-1 表示用满所有核。
#    影响：KNN 预测要算大量距离，容易并行，大数据集上 n_jobs=-1 能明显加速。
#    常用值：-1（本机多核）。
#
# leaf_size : int，默认 30
#    含义：ball_tree / kd_tree 的叶子节点样本数上限。
#    调小：树更深、构建慢但查询快；调大：树更浅、构建快但查询慢。
#    只在 algorithm 用树结构时生效，默认 30 一般不用改。
# ============================================================================


print_section("《机器学习》课案 · 分类 · 02 K-近邻（KNN）")
print("本脚本四段结构： ① 原理与数学推导  ② sklearn API 参数解释  ③ 完整可运行代码  ④ 结果解读")
print("配套图片输出目录：", OUTPUT_DIR)

# ============================================================================
# ③ 完整可运行代码
# ============================================================================

# ---------------------------------------------------------------------------
# 步骤 1：提取数据 —— 课案做法：load_iris 全量（3 类 150 条）
# ---------------------------------------------------------------------------
print_section("步骤 1：提取数据（load_iris 全量，三分类任务）")

iris = load_iris()
X = iris.data
y = iris.target
class_names = [str(n) for n in iris.target_names]
print("数据集来源：sklearn.datasets.load_iris（内置数据集，无需联网下载）")
print("特征名：", list(iris.feature_names))
print("类别名：", class_names)
print(f"样本形状：X = {X.shape}，y = {y.shape}")
print("各类样本数：", {class_names[i]: int(n) for i, n in enumerate(np.bincount(y))})
print("三分类：山鸢尾 setosa(0) / 变色鸢尾 versicolor(1) / 维吉尼亚鸢尾 virginica(2)")

# ---------------------------------------------------------------------------
# 步骤 2：清洗 / 处理数据 —— 划分 + 标准化
# ---------------------------------------------------------------------------
print_section("步骤 2：划分训练集/测试集 + 标准化")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)
print(f"训练集 X_train = {X_train.shape}，测试集 X_test = {X_test.shape}")
print("训练集各类数量：", np.bincount(y_train), "，测试集各类数量：", np.bincount(y_test))

# 标准化：KNN 基于距离，不做标准化则量纲大的特征会垄断距离计算
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print("标准化前，四个特征的取值范围（最大值 - 最小值）：",
      np.round(X_train.max(axis=0) - X_train.min(axis=0), 2))
print("标准化后，四个特征的取值范围：",
      np.round(X_train_s.max(axis=0) - X_train_s.min(axis=0), 2))
print("为什么必须标准化：")
print("  ① KNN 全靠距离判断相似度，而距离 = 各维度差值的直接叠加，没有任何权重学习；")
print("  ② 若不标准化，取值范围大的特征会淹没取值范围小的特征（相当于把后者丢掉）；")
print("  ③ 标准化后每个特征都是'偏离均值几倍标准差'，量纲统一、地位平等。")
print("  结论：凡是用到距离的算法（KNN / K-Means / SVM-RBF），第一步永远是标准化。")

# ---------------------------------------------------------------------------
# 步骤 3：运行 KNN 算法 —— 先复现课案里的 n_neighbors=3
# ---------------------------------------------------------------------------
print_section("步骤 3：训练 KNN 模型（先复现课案的 n_neighbors=3）")

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_s, y_train)
print("KNeighborsClassifier(n_neighbors=3) 已 fit 完成。")
print("注意：KNN 的 fit() 不做任何计算，只是把训练集存进模型里 —— 这就是'懒惰学习'。")
print("  模型保存的样本数 n_samples_fit_ =", model.n_samples_fit_)
print("  训练阶段耗时几乎为 0，真正的工作全部发生在 predict()。")

# ---------------------------------------------------------------------------
# 步骤 4：得到预测结果
# ---------------------------------------------------------------------------
print_section("步骤 4：预测与评估")

y_pred = model.predict(X_test_s)
acc = accuracy_score(y_test, y_pred)
print("预测结果（前 15 条）：", y_pred[:15])
print("真实结果（前 15 条）：", y_test[:15])
print(f"测试集准确率 = {acc:.4f}（{(y_pred == y_test).sum()} / {len(y_test)} 条正确）")

print("\n分类报告：")
print(classification_report(y_test, y_pred, target_names=class_names, digits=4, zero_division=0))
print("精确率 precision：预测成该类的样本里，有多少真的是该类（查准）；")
print("召回率 recall   ：该类真实样本里，有多少被找出来了（查全）；")
print("F1 = 2PR/(P+R)  ：两者的调和平均。")

print("\n顺便看看在训练集上的表现（KNN 在训练集上通常接近满分，因为训练点自己就是邻居）：")
train_acc_k3 = accuracy_score(y_train, model.predict(X_train_s))
print(f"  K=3 时训练集准确率 = {train_acc_k3:.4f}，测试集准确率 = {acc:.4f}")

# ---------------------------------------------------------------------------
# 步骤 5：不同 K 的准确率表（观察过拟合 / 欠拟合）
# ---------------------------------------------------------------------------
print_section("步骤 5：不同 K 值的准确率对照表")

print("说明：对每个 K 都用同一份训练集重新 fit（KNN 训练极快），记录训练集与测试集准确率。")
print("注意：这里最大只取 K=75，因为后面要用 5 折交叉验证，每折的训练部分只剩 84 条样本，")
print("      K 必须小于每折的样本数，否则'找不到那么多邻居'会直接报错。")
print("-" * 78)
print(f"{'K 取值':<8}{'训练集准确率':>16}{'测试集准确率':>16}{'两者差距':>14}{'说明':>18}")
print("-" * 78)

k_list = [1, 2, 3, 5, 7, 9, 11, 15, 20, 25, 30, 40, 50, 75]
k_train_acc, k_test_acc = [], []
_raw_rows = []
for k in k_list:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train_s, y_train)
    tr = accuracy_score(y_train, knn_k.predict(X_train_s))
    te = accuracy_score(y_test, knn_k.predict(X_test_s))
    k_train_acc.append(tr)
    k_test_acc.append(te)
    _raw_rows.append((k, tr, te))
_max_te = max(k_test_acc)
for k, tr, te in _raw_rows:
    gap = tr - te
    if k == 1:
        note = "训练集满分→过拟合"
    elif te <= _max_te - 0.08:
        note = "测试掉太多→欠拟合"
    elif gap > 0.08:
        note = "差距偏大→过拟合倾向"
    else:
        note = "较为均衡"
    print(f"{k:<8}{tr:>16.4f}{te:>16.4f}{gap:>14.4f}{note:>18}")
print("-" * 78)
print("读表要点：")
print("  · K=1 时训练集准确率必然是 1.0000（每个训练点的最近邻就是它自己，距离为 0），")
print("    但测试集准确率不是最高 —— 这就是最典型的过拟合；")
print("  · K 逐渐增大，训练集准确率下降（边界不再贴合每个点），测试集准确率先升后降；")
print("  · K 大到 50/75 时，邻居里混进了别的类，两类准确率都被拉低 —— 欠拟合。")

best_k_test = k_list[int(np.argmax(k_test_acc))]
best_te = max(k_test_acc)
print(f"\n在这组候选里，测试集准确率最高的 K = {best_k_test}（准确率 {best_te:.4f}）。")

# 用 5 折交叉验证再挑一次 K：比"单次划分的测试准确率"更可靠
print("\n用 5 折交叉验证（在训练集内部做，不碰测试集）重新评估各 K，更可靠：")
print("-" * 62)
print(f"{'K 取值':<10}{'5折CV准确率均值':>18}{'标准差':>12}{'均值-1倍标准差':>18}")
print("-" * 62)
cv_mean, cv_std = [], []
for k in k_list:
    scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train_s, y_train, cv=5)
    cv_mean.append(scores.mean())
    cv_std.append(scores.std())
    print(f"{k:<10}{scores.mean():>18.4f}{scores.std():>12.4f}{scores.mean() - scores.std():>18.4f}")
print("-" * 62)
best_cv_idx = int(np.argmax(cv_mean))
print(f"交叉验证均值最高的 K = {k_list[best_cv_idx]}（{cv_mean[best_cv_idx]:.4f} ± {cv_std[best_cv_idx]:.4f}）")
print("为什么更信交叉验证：单次 train_test_split 的结果受划分随机性影响大，")
print("                    比如测试集只有 45 条，错 1 条准确率就掉 2.2 个百分点；")
print("                    5 折交叉验证把数据轮换用 5 次，估计更稳定。")

# ---------------------------------------------------------------------------
# 步骤 6：四种距离度量 / 两种投票权重的对比
# ---------------------------------------------------------------------------
print_section("步骤 6：距离度量与投票方式对比（K=5 固定，只换距离/权重）")

metric_cases = [
    ("欧氏距离 minkowski,p=2", dict(metric="minkowski", p=2)),
    ("曼哈顿距离 minkowski,p=1", dict(metric="minkowski", p=1)),
    ("曼哈顿距离 manhattan", dict(metric="manhattan")),
    ("切比雪夫距离 chebyshev", dict(metric="chebyshev")),
    ("闵可夫斯基 p=3", dict(metric="minkowski", p=3)),
    ("闵可夫斯基 p=10", dict(metric="minkowski", p=10)),
]
print("【按距离度量分组，weights='uniform'（等权投票）】")
print("-" * 76)
print(f"{'距离度量':<30}{'训练集准确率':>16}{'测试集准确率':>16}{'5折CV均值':>14}")
print("-" * 76)
for name, kw in metric_cases:
    m = KNeighborsClassifier(n_neighbors=5, weights="uniform", **kw)
    m.fit(X_train_s, y_train)
    tr = accuracy_score(y_train, m.predict(X_train_s))
    te = accuracy_score(y_test, m.predict(X_test_s))
    cv = cross_val_score(m, X_train_s, y_train, cv=5).mean()
    print(f"{name:<30}{tr:>16.4f}{te:>16.4f}{cv:>14.4f}")
print("-" * 76)
print("为什么这些结果几乎一样：iris 各特征标准化后尺度相同、样本量大、类别分得开，")
print("所以换距离度量对结果影响很小；换成量纲混乱或高维稀疏的数据，差别就会很明显。")

weights_cases = [("uniform 等权投票", "uniform"), ("distance 按 1/距离 加权", "distance")]
metric_for_weight = [("minkowski p=2（欧氏）", dict(metric="minkowski", p=2)),
                     ("minkowski p=1（曼哈顿）", dict(metric="minkowski", p=1))]
weights_result = []
print("\n【换投票权重（weights）】")
print("-" * 86)
print(f"{'距离度量':<26}{'投票权重':<24}{'测试集准确率':>16}{'5折CV均值':>16}")
print("-" * 86)
for mname, mkw in metric_for_weight:
    for wname, w in weights_cases:
        m = KNeighborsClassifier(n_neighbors=5, weights=w, **mkw)
        m.fit(X_train_s, y_train)
        te = accuracy_score(y_test, m.predict(X_test_s))
        cv = cross_val_score(m, X_train_s, y_train, cv=5).mean()
        weights_result.append((mname, wname, te, cv))
        print(f"{mname:<26}{wname:<24}{te:>16.4f}{cv:>16.4f}")
print("-" * 86)
print("weights='distance' 的含义：邻居的票按 1/距离 加权，越近的邻居说话越算数。")
print("在类别不平衡、或 K 取得偏大时，它通常比等权投票更稳。")

# ---------------------------------------------------------------------------
# 步骤 7：画图 1 —— K 值与准确率曲线（训练集 / 测试集两条线）
# ---------------------------------------------------------------------------
print_section("步骤 7：绘图 —— K 值与准确率曲线")

k_axis = np.arange(1, 51)          # 1~50，逐点计算，曲线才平滑
curve_train, curve_test = [], []
for k in k_axis:
    knn_k = KNeighborsClassifier(n_neighbors=int(k))
    knn_k.fit(X_train_s, y_train)
    curve_train.append(accuracy_score(y_train, knn_k.predict(X_train_s)))
    curve_test.append(accuracy_score(y_test, knn_k.predict(X_test_s)))

fig1, ax1 = plt.subplots(figsize=(8.6, 5.2))
ax1.plot(k_axis, curve_train, "o-", color="#d62728", markersize=3.5, linewidth=1.8,
         label="训练集准确率")
ax1.plot(k_axis, curve_test, "s-", color="#1f77b4", markersize=3.5, linewidth=1.8,
         label="测试集准确率")
ax1.axvline(best_k_test, color="#2ca02c", linestyle="--", linewidth=1.3,
            label=f"测试集最优 K = {best_k_test}")
# 标出三个典型区域
ax1.axvspan(0.5, 2.5, color="#d62728", alpha=0.08)
ax1.axvspan(30.5, 50.5, color="#1f77b4", alpha=0.08)
ax1.text(1.6, 0.60, "K 太小\n过拟合\n（高方差）", ha="center", fontsize=10, color="#a01c1c")
ax1.text(40.5, 0.60, "K 太大\n欠拟合\n（高偏差）", ha="center", fontsize=10, color="#14507a")
ax1.set_title("KNN 的 K 值 vs 准确率：训练集与测试集的经典剪刀差", fontsize=12)
ax1.set_xlabel("邻居个数 K（n_neighbors）", fontsize=11)
ax1.set_ylabel("准确率 accuracy", fontsize=11)
ax1.set_xlim(0.5, 50.5)
ax1.set_ylim(0.55, 1.02)
ax1.grid(alpha=0.3, linestyle="--")
ax1.legend(loc="upper right", fontsize=10)
fig1.tight_layout()
curve_path = OUTPUT_DIR / "03_KNN_K值与准确率曲线.png"
fig1.savefig(curve_path, dpi=130)
plt.close(fig1)
print("已保存图片：", curve_path)

# ---------------------------------------------------------------------------
# 步骤 8：画图 2 —— 决策边界（K=1 / K=15 / K=50 三个子图对比）
# ---------------------------------------------------------------------------
print_section("步骤 8：绘图 —— 决策边界（K=1 / K=15 / K=50 对比）")

# 取 iris 后两个特征（花瓣长、花瓣宽），它们区分度最高，便于在平面上展示
X2 = iris.data[:, [2, 3]]
y2 = iris.target
X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.3, random_state=RANDOM_STATE, stratify=y2
)
scaler2 = StandardScaler()
X2_train_s = scaler2.fit_transform(X2_train)
X2_test_s = scaler2.transform(X2_test)

pad = 0.9
xx, yy = np.meshgrid(
    np.linspace(X2_train_s[:, 0].min() - pad, X2_train_s[:, 0].max() + pad, 400),
    np.linspace(X2_train_s[:, 1].min() - pad, X2_train_s[:, 1].max() + pad, 400),
)
grid_points = np.c_[xx.ravel(), yy.ravel()]
region_cmap = ListedColormap(["#a8d5e5", "#ffd79a", "#c4a3d4"])
point_colors = ["#1f77b4", "#e08a00", "#6a3d9a"]

fig2, axes2 = plt.subplots(1, 3, figsize=(16.2, 5.4), sharey=True)
vis_accs = {}
for ax, k_val in zip(axes2, [1, 15, 50]):
    knn_vis = KNeighborsClassifier(n_neighbors=k_val)
    knn_vis.fit(X2_train_s, y2_train)
    zz = knn_vis.predict(grid_points).reshape(xx.shape)
    acc_vis = accuracy_score(y2_test, knn_vis.predict(X2_test_s))
    vis_accs[k_val] = acc_vis
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5, 2.5], cmap=region_cmap, alpha=0.65)
    for cls in range(3):
        ax.scatter(X2_train_s[y2_train == cls, 0], X2_train_s[y2_train == cls, 1],
                   c=point_colors[cls], s=34, edgecolors="white", linewidths=0.6,
                   label=class_names[cls])
    ax.scatter(X2_test_s[:, 0], X2_test_s[:, 1], c="none", edgecolors="black",
               s=80, linewidths=1.1, label="测试样本")
    ax.set_title(f"K = {k_val}    测试集准确率 = {acc_vis:.2%}", fontsize=12)
    ax.set_xlabel("标准化花瓣长", fontsize=10)
    ax.grid(alpha=0.2, linestyle="--")
axes2[0].set_ylabel("标准化花瓣宽", fontsize=10)
axes2[0].legend(loc="upper left", fontsize=8, framealpha=0.9)
fig2.suptitle("K 值决定决策边界的【复杂度】：K=1 边界破碎（过拟合）→ K=50 边界过度平滑（欠拟合）",
              fontsize=13)
fig2.tight_layout(rect=(0, 0, 1, 0.94))
boundary_path = OUTPUT_DIR / "03_KNN_决策边界.png"
fig2.savefig(boundary_path, dpi=130)
plt.close(fig2)
print("已保存图片：", boundary_path)
for k_val in (1, 15, 50):
    knn_vis = KNeighborsClassifier(n_neighbors=k_val).fit(X2_train_s, y2_train)
    print(f"  K = {k_val:>2}：测试集准确率 = "
          f"{accuracy_score(y2_test, knn_vis.predict(X2_test_s)):.4f}")
print("看图要点：K=1 时每个训练点周围都有一小块'自己的领地'，边界参差不齐；")
print("          K 增大后这些小块被抹平，边界变得光滑；K=50 时边界已基本失去细节。")

# ---------------------------------------------------------------------------
# 步骤 9：画图 3 —— 距离度量 / 投票方式对比柱状图
# ---------------------------------------------------------------------------
print_section("步骤 9：绘图 —— 距离度量与投票方式对比")

metric_names = [n for n, _ in metric_cases]
metric_cv = []
for _, kw in metric_cases:
    m = KNeighborsClassifier(n_neighbors=5, weights="uniform", **kw)
    metric_cv.append(cross_val_score(m, X_train_s, y_train, cv=5).mean())

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14.6, 5.0))
bars = ax3a.bar(range(len(metric_names)), metric_cv, color="#4c72b0", width=0.6)
ax3a.set_xticks(range(len(metric_names)))
ax3a.set_xticklabels([n.replace(" ", "\n", 1) for n in metric_names], fontsize=9)
ax3a.set_ylim(min(metric_cv) - 0.03, 1.005)
ax3a.set_title("不同距离度量下的 5 折交叉验证准确率（K=5，等权投票）", fontsize=11)
ax3a.set_ylabel("5 折 CV 准确率均值", fontsize=10)
ax3a.grid(axis="y", alpha=0.3, linestyle="--")
for b, v in zip(bars, metric_cv):
    ax3a.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}",
              ha="center", fontsize=9)

weight_labels = [f"{mn}\n{wn}" for mn, wn, _, _ in weights_result]
weight_cv = [cv for _, _, _, cv in weights_result]
bars2 = ax3b.bar(range(len(weight_labels)), weight_cv,
                 color=["#4c72b0", "#dd8452", "#55a868", "#c44e52"], width=0.6)
ax3b.set_xticks(range(len(weight_labels)))
ax3b.set_xticklabels(weight_labels, fontsize=8.5)
ax3b.set_ylim(min(weight_cv) - 0.05, 1.005)
ax3b.set_title("weights='uniform' vs 'distance' 的 5 折交叉验证准确率（K=5）", fontsize=11)
ax3b.set_ylabel("5 折 CV 准确率均值", fontsize=10)
ax3b.grid(axis="y", alpha=0.3, linestyle="--")
for b, v in zip(bars2, weight_cv):
    ax3b.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)

fig3.suptitle("距离度量与投票权重：换的是【什么叫相似】，属于领域知识而非盲目调参", fontsize=12)
fig3.tight_layout(rect=(0, 0, 1, 0.93))
metric_path = OUTPUT_DIR / "03_KNN_距离度量与权重对比.png"
fig3.savefig(metric_path, dpi=130)
plt.close(fig3)
print("已保存图片：", metric_path)

# ============================================================================
# ④ 结果解读
# ============================================================================
print_section("④ 结果解读（这些数字到底意味着什么）")

print(f"1) 课案默认配置 K=3 的测试集准确率 = {acc:.4f}：")
print(f"   在 {len(y_test)} 条测试样本中答对 {(y_pred == y_test).sum()} 条。")
print("   iris 三分类在标准化后本身就很好分，KNN 这种简单方法也能轻松跑到 90% 以上，")
print("   说明'距离近的样本类别相似'这个假设在 iris 上成立。")
print()
print("2) 训练集准确率与测试集准确率的对比才是关键：")
print(f"   K=1  → 训练集 {k_train_acc[0]:.4f}，测试集 {k_test_acc[0]:.4f}"
      f"（差距 {k_train_acc[0] - k_test_acc[0]:+.4f}）")
print(f"   K=3  → 训练集 {k_train_acc[2]:.4f}，测试集 {k_test_acc[2]:.4f}"
      f"（差距 {k_train_acc[2] - k_test_acc[2]:+.4f}）")
print(f"   K={best_k_test} → 训练集 {k_train_acc[k_list.index(best_k_test)]:.4f}，"
      f"测试集 {k_test_acc[k_list.index(best_k_test)]:.4f}"
      f"（差距 {k_train_acc[k_list.index(best_k_test)] - k_test_acc[k_list.index(best_k_test)]:+.4f}）")
print("   K=1 的训练准确率是 1.0000 —— 这不是好消息，而是'模型把训练集背下来了'的证据。")
print("   判断过拟合的标准从来不是'训练准确率高不高'，而是'训练与测试的差距大不大'。")
print()
print(f"3) K 与准确率曲线图（{curve_path.name}）：")
print("   红线（训练集）随 K 增大持续下降：K 越大，邻域越大，边界越平滑，")
print("   连训练点自己都照顾不到了；蓝线（测试集）先升后降，呈现一个明显的峰值。")
print("   两条线的间距在 K 很小的一段区间里明显偏大（过拟合），K 很大时两条线都低（欠拟合）。")
print("   这张'剪刀差'图是理解所有模型复杂度问题的通用模板。")
print()
print(f"4) 决策边界图（{boundary_path.name}）：")
print("   三个子图用的是同一份数据，只有 K 不同，测试集准确率分别是：")
print(f"     K = 1  → {vis_accs[1]:.4f}（边界到处是小碎块，每个训练点一小块领地 → 高方差）")
print(f"     K = 15 → {vis_accs[15]:.4f}（边界平滑且仍贴合类别分布 → 通常泛化最好）")
print(f"     K = 50 → {vis_accs[50]:.4f}（边界被抹成几大块，类别间的细节被吞掉 → 高偏差）")
print("   ⚠ 注意：本图只用了 2 个特征、测试集只有 45 条，K=1 偶然也能拿到很高的准确率。")
print("     评估模型不能只看一次划分的结果，要结合上表的整体趋势和交叉验证来判断。")
print()
print("5) 距离度量与投票权重对比：")
print("   在本数据上各距离度量的 CV 准确率非常接近，原因有二：")
print("     ① 已经标准化，各维度地位平等；② 三类在特征空间里本来就分得开。")
print("   这正好说明：**距离度量的选择应该由数据含义决定（比如城市路网用曼哈顿、")
print("   棋盘步数用切比雪夫），而不是当超参数去盲搜。**")
print()
print("6) 一句话总览：KNN = 懒惰学习 + 距离度量 + K 个邻居投票。")
print("   它没有'训练'，全靠数据本身；因此数据质量（标准化、去噪、去无关特征）")
print("   对它的影响，比任何超参数都大。")

# ============================================================================
# 超参数怎么调
# ============================================================================
# 【超参数怎么调】—— KNN 实战调参顺序
#
# 0. 前置条件（比调参重要得多）：
#    - 必须标准化！忘了标准化，其它参数怎么调都是在错的距离空间里瞎调；
#    - 去掉无关特征和噪声标签。KNN 不做特征选择，一个无关特征就会污染距离，
#      一个错标样本就能在它周围制造一片错误的决策区域；
#    - 数据量太大时先考虑降维（PCA）或换算法，因为 KNN 预测复杂度是 O(N·d)。
#
# 1. n_neighbors（K，第一优先级）：
#    - 起点：K ≈ sqrt(n_train)（例：105 条训练样本 → 约 10）；
#    - 搜索范围：1 ~ 30 在多数任务上够用；二分类取奇数避免平票；
#    - 用 cross_val_score / GridSearchCV 选，看 CV 均值（±标准差）；
#    - 判据：训练准确率远高于 CV 准确率 → K 太小；两者都低 → K 太大。
#
# 2. weights：
#    - 先试默认 'uniform'；若类别不平衡或 K 偏大导致效果不佳，换 'distance'；
#    - 'distance' 让近邻票更重，通常能小幅提升，但会让预测更慢一点点。
#
# 3. metric / p（第二优先级，但更多靠领域知识而不是搜索）：
#    - 连续数值特征、尺度已统一 → 'minkowski' + p=2（欧氏，默认）；
#    - 高维稀疏、或希望更抗离群维度 → p=1（曼哈顿）；
#    - 只关心差得最大的那一维（棋盘、库存最大缺口等）→ 'chebyshev'；
#    - 文本 / TF-IDF 向量 → 'cosine'（余弦距离，只看向量夹角不看长度）。
#
# 4. algorithm / leaf_size / n_jobs（只影响速度，不影响精度）：
#    - 小数据直接默认 algorithm='auto'；
#    - 特征维度 < 20 且样本多 → 'kd_tree' 或 'ball_tree' 更快；
#    - 维度很高（>30）→ 树结构失效，退回 'brute'，此时靠 n_jobs=-1 并行加速；
#    - n_jobs=-1 用满所有 CPU 核，官方推荐用于大数据的预测阶段。
#
# 5. 推荐自动化做法：
#    GridSearchCV(Pipeline([("scaler", StandardScaler()),
#                           ("knn", KNeighborsClassifier())]),
#                 {"knn__n_neighbors": list(range(1, 31, 2)),
#                  "knn__weights": ["uniform", "distance"],
#                  "knn__p": [1, 2]},
#                 cv=5, scoring="accuracy", n_jobs=-1)
#    —— 把 StandardScaler 放进 Pipeline，交叉验证的每一折都只在训练部分 fit，
#       从根本上避免数据泄漏。
# ============================================================================

print()
print("【完成】02_KNN.py 运行结束")

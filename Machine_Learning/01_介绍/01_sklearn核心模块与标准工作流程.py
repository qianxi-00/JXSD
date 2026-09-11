"""sklearn 核心模块与标准工作流程（《机器学习》课案 · 介绍章 配套讲解脚本）

对应课案章节
------------
《机器学习》课案 → `## 介绍` → `### sklearn 介绍` + `### sklearn 标准工作流程`
（课案中这两段共 2 个代码块，本脚本把它们扩展为一个完整、可运行、带详细中文讲解的工程示例）

本节知识点
----------
1. sklearn 是什么、设计哲学（统一 Estimator API / 组合优于继承 / fit 学参数 transform 用参数）
2. 三类对象：Estimator（估计器）、Transformer（转换器）、Predictor（预测器）
3. sklearn 核心模块清单及其用途：datasets / model_selection / preprocessing / linear_model /
   tree / ensemble / cluster / decomposition / metrics / joblib（另补充 pipeline / compose 等）
4. 关键 API 参数逐个解释：
   train_test_split(test_size / random_state / stratify / shuffle)
   StandardScaler(with_mean / with_std)
   RandomForestClassifier(n_estimators / max_depth / random_state / n_jobs)
5. sklearn 标准工作流程五步：
   ① 提取数据 → ② 清洗与划分数据 → ③ 运行算法(fit) → ④ 得到结果(predict/transform) → ⑤ 评估
6. Estimator API 三件套 fit / predict(或 transform) / score；fit_transform 与 fit().transform() 的区别
7. 为什么必须先在训练集 fit、再 transform 测试集 —— 数据泄漏（Data Leakage）与标准化必须在划分之后做
8. 结果解读：训练集 vs 测试集准确率对比、交叉验证均值、如何判断过拟合/欠拟合

运行方式
--------
PowerShell（路径含中文，必须加引号并用 & 调用）：

    $env:PYTHONUTF8='1'
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Machine_Learning\\01_介绍\\01_sklearn核心模块与标准工作流程.py'

写成一行：

    $env:PYTHONUTF8='1'; & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Machine_Learning\\01_介绍\\01_sklearn核心模块与标准工作流程.py'

说明：
* 唯一解释器是本项目 .venv；**禁止** uv run / pip install / uv add（会破坏环境）。
* 只用 sklearn 内置数据集（本脚本用 load_iris），**无需联网、无需下载数据**。
* 脚本使用 matplotlib 的 Agg 无界面后端，**不会调用 plt.show()**，图片直接保存到
  `Machine_Learning/output/01_sklearn_iris特征分布.png`；路径用 pathlib 相对 __file__ 定位，
  不依赖当前工作目录。
* 预期：退出码 0，输出无 Traceback、无 Warning，运行时间 < 30 秒。
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")            # 无界面后端，避免 plt.show() 阻塞
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import numpy as np

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 路径：一律相对 __file__ 定位，保证在任何工作目录下运行结果一致
# ---------------------------------------------------------------------------
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_PATH = OUTPUT_DIR / "01_sklearn_iris特征分布.png"

RANDOM_STATE = 42  # 统一随机种子：所有随机过程都用它，保证结果可复现


def title(text: str) -> None:
    """打印一个醒目的中文分节标题（纯格式化，无业务含义）。"""
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


# ===========================================================================
# ① 原理与讲解
# ===========================================================================
title("① 原理与讲解：sklearn 是什么、怎么设计、标准流程怎么走")

print(
    """
【1. sklearn 是什么】
scikit-learn（sklearn）是 Python 最流行的经典机器学习库，建立在 NumPy / SciPy 之上，
提供了"数据预处理 → 特征工程 → 模型训练 → 评估 → 持久化"的完整工具链。
它的定位是中小规模、单机内存内的经典机器学习（深度学习请用 PyTorch，极致性能的
梯度提升可以用 XGBoost / LightGBM，本环境都已安装）。

【2. 设计哲学（4 条，理解了就不会觉得 API 乱）】
  (a) 一致的 Estimator API：所有模型都是"估计器"，都提供 fit / predict / transform / score。
      → 换模型通常只需要改一行 import + 一个类名，其余代码完全不动。这正是 sklearn
        能配合 GridSearchCV、Pipeline 自动组合的前提。
  (b) fit 学参数，transform 用参数（参数分离）：
      → "从数据里学到的量"（均值、方差、PCA 方向、编码字典）只允许在 fit 阶段产生，
        之后必须复用，绝不能对着新数据重新学。这是防数据泄漏的机制保障。
  (c) 组合优于继承：Pipeline 把预处理和模型串成一条流水线；ColumnTransformer 对
      不同列做不同处理；GridSearchCV 能把任意估计器包起来做超参搜索。
  (d) 元信息可自省：get_params() / set_params() / clone() 让模型可以被程序自动复制与搜索。

【3. 三类对象（本脚本会全部用到）】
  ① Estimator 估计器：能从数据中学习的对象，接口是 fit(X, y=None)。
     例：RandomForestClassifier、KMeans。
  ② Transformer 转换器：把数据 X 变成新形态 X'，接口是 fit + transform + fit_transform。
     例：StandardScaler（本脚本用）、PCA、OneHotEncoder。转换器也是估计器。
  ③ Predictor 预测器：对新样本输出预测，接口是 predict（分类器额外有 predict_proba）。
     例：RandomForestClassifier.predict / .predict_proba。
  关系：Predictor ⊂ Estimator，Transformer ⊂ Estimator；ClassifierMixin / RegressorMixin
  负责给它们补上默认的 score 方法。

【4. 标准工作流程五步（本脚本正文严格按这五步写）】
  ① 提取数据        ：拿到特征矩阵 X 与标签向量 y        → load_iris(return_X_y=True)
  ② 清洗与划分数据  ：处理缺失/异常 → 【先划分】 → 再标准化 → train_test_split + StandardScaler
  ③ 运行算法        ：在【训练集】上学习参数              → model.fit(X_train, y_train)
  ④ 得到结果        ：对【测试集/新数据】预测或转换       → model.predict(X_test) / scaler.transform(X_test)
  ⑤ 评估            ：用指标衡量泛化能力                 → accuracy_score / confusion_matrix / cross_val_score

【5. 为什么"标准化"必须放在"划分"之后做？——数据泄漏（Data Leakage）】
  StandardScaler 会从数据里学两个统计量：每个特征的均值 mean_ 和标准差 scale_。
  如果先对【全量数据】做 fit_transform 再划分，那么测试集的均值/方差就参与了训练，
  模型间接"偷看"了测试集 → 这叫数据泄漏。后果有两个：
    (a) 评估结果虚高（乐观偏差），上线后性能大跌；
    (b) 训练集与测试集不再独立，"测试集"不再是没见过的新数据。
  正确顺序（本脚本采用）：
      train_test_split → scaler.fit_transform(X_train) → scaler.transform(X_test)
  一句话记忆：fit_transform 只能用在训练集上，测试集永远只用 transform。
  补充：测试集 transform 之后，它的均值不一定是 0、标准差不一定是 1 —— 这是正常的，
  因为它用的是【训练集的】均值和标准差，这也正是真实部署时处理新数据的方式。
"""
)


# ===========================================================================
# ② sklearn 核心模块与关键 API 参数逐个解释
# ===========================================================================
title("② sklearn 核心模块与关键 API 参数逐个解释")

print(
    """
【核心模块清单：模块 → 用途 → 常用类/函数】
  sklearn.datasets           内置数据集与数据生成器
                             load_iris / load_diabetes / load_wine / load_breast_cancer
                             make_regression / make_classification / make_moons / make_blobs
  sklearn.model_selection    数据划分、交叉验证、超参搜索
                             train_test_split / cross_val_score / KFold / StratifiedKFold
                             GridSearchCV / RandomizedSearchCV / learning_curve
  sklearn.preprocessing      特征缩放、编码、非线性变换
                             StandardScaler / MinMaxScaler / RobustScaler / OneHotEncoder
                             OrdinalEncoder / LabelEncoder / PolynomialFeatures
  sklearn.linear_model       线性模型家族
                             LinearRegression / Ridge / Lasso / ElasticNet
                             LogisticRegression / SGDRegressor / SGDClassifier
  sklearn.tree               决策树
                             DecisionTreeClassifier / DecisionTreeRegressor / plot_tree
  sklearn.ensemble           集成学习（Bagging / Boosting / Stacking）
                             RandomForestClassifier / ExtraTreesClassifier
                             GradientBoostingClassifier / HistGradientBoostingClassifier
                             AdaBoostClassifier / VotingClassifier / StackingClassifier
  sklearn.cluster            聚类
                             KMeans / DBSCAN / AgglomerativeClustering / SpectralClustering
  sklearn.decomposition      矩阵分解与降维
                             PCA / TruncatedSVD / NMF / FastICA / IncrementalPCA
  sklearn.metrics            评估指标
                             accuracy_score / precision_score / recall_score / f1_score
                             confusion_matrix / classification_report / roc_auc_score
                             mean_squared_error / root_mean_squared_error / r2_score
                             silhouette_score / adjusted_rand_score
  joblib                     模型持久化（保存到磁盘 / 加载回来）
                             joblib.dump(model, "m.joblib") / joblib.load("m.joblib")
  （常用补充）sklearn.pipeline / sklearn.compose      Pipeline / make_pipeline / ColumnTransformer
  （常用补充）sklearn.feature_selection               SelectKBest / RFE / VarianceThreshold
  （常用补充）sklearn.inspection                      permutation_importance

【train_test_split(X, y, test_size=0.25, random_state=None, stratify=None, shuffle=True)】
  test_size     : 测试集占比（0~1 小数）或绝对样本数（整数）。默认 0.25。
                  数据量小取 0.2~0.3；百万级数据取 0.01~0.05 就够。
  random_state  : 随机种子。固定后每次划分结果【完全相同】→ 实验可复现，必须固定。
  stratify      : 按传入数组的类别比例做【分层抽样】，分类任务写 stratify=y。
                  作用：保证训练集/测试集的类别比例与原始数据一致，避免少数类在
                  测试集里一个样本都没有，导致评估结果不可信（类别不平衡时尤其关键）。
  shuffle       : 划分前是否打乱顺序，默认 True。
                  注意：时序数据（股价、日志、传感器序列）不能随机打乱，否则会用
                  "未来"预测"过去"；应使用 shuffle=False 或 TimeSeriesSplit。
  返回          : X_train, X_test, y_train, y_test（顺序固定，别接错）。
                  实际样本数 = ceil(测试集占比 × 样本总数)。

【StandardScaler(with_mean=True, with_std=True, copy=True)】
  作用：把每个特征标准化成均值 0、标准差 1 —— z = (x - μ) / σ。
  with_mean : 是否减均值（中心化），默认 True。若输入是【稀疏矩阵】必须设 False，
              因为减均值会把 0 变成非 0，破坏稀疏结构与内存优势。
  with_std  : 是否除以标准差（缩放），默认 True。一般保持默认。
  copy      : 是否复制数据，默认 True（不动原数组）。内存紧张可设 False，但会原地改数据。
  fit 之后学到的属性（sklearn 约定：带下划线结尾的属性 = fit 之后才存在）：
              mean_（每列均值）、scale_（每列标准差）、var_（每列方差）、n_features_in_。
  为什么要标准化：
    (a) 距离型算法（KNN / K-Means / SVM / PCA）会被量纲大的特征主导，尺度不同则
        "距离"没有意义；
    (b) 梯度下降在特征尺度不一时收敛慢、易震荡；
    (c) 正则化对所有参数一视同仁，尺度不齐会导致惩罚不公平。
    反例：树模型（决策树 / 随机森林 / XGBoost）只关心分裂阈值的大小顺序，
    对单调缩放不敏感 → 可以不做标准化。本脚本为了让流程完整，仍然演示标准化，
    并在文末对比"标准化前/后"的分布。

【RandomForestClassifier(n_estimators=100, max_depth=None, random_state=None, n_jobs=None)】
  n_estimators : 森林中树的数量。越大 → 方差越小、越稳，但耗时线性增长且收益递减。
                 常用 100~1000。本脚本用 200。
  max_depth    : 单棵树最大深度，None 表示不限（一直长到叶节点纯净）。
                 这是控制【过拟合】最直接的旋钮：小 → 欠拟合；大 → 过拟合。
  random_state : 随机种子，同时控制 bootstrap 自助采样与每次分裂的特征子集抽样。
                 必须固定，否则每次结果不同。
  n_jobs       : 并行进程数。None=1；-1=用满所有 CPU 核（本例 -1，iris 很小，几乎瞬完）。
  min_samples_leaf / min_samples_split : 叶节点/分裂所需最小样本数，比 max_depth 更精细的
                 防过拟合手段。
  max_features : 每次分裂考虑的特征数，默认 "sqrt"（分类）。越小 → 树之间越独立 → 方差越低。
  oob_score    : 是否用袋外样本（out-of-bag）估计泛化误差，数据少时可设 True 白拿一个估计。
  class_weight : 类别权重，可设 "balanced"，类别不平衡时的重要参数。

【随机森林为什么有效】Bagging（自助采样 + 并行训练多棵树）+ 分裂时随机选特征子集
  → 每棵树是"低偏差、高方差"的基学习器，平均之后方差被大幅抵消，因而整体稳健、抗过拟合、
  且几乎不需要精细调参，是很好的"基线第一选择"。
"""
)


# ===========================================================================
# ③ 完整可运行代码（严格按标准工作流程五步走）
# ===========================================================================
title("③ 完整可运行代码：五步流程实战（load_iris + 随机森林）")

# --- 第 ① 步：提取数据 -----------------------------------------------------
print("\n【第 ① 步】提取数据")
iris = load_iris()
X, y = load_iris(return_X_y=True)  # return_X_y=True 直接返回 (特征矩阵, 标签向量)

print(f"  数据集形状 X.shape = {X.shape}  →  {X.shape[0]} 个样本，{X.shape[1]} 个特征")
print(f"  标签形状 y.shape   = {y.shape}  →  一维标签向量（每个样本一个类别编号）")
print(f"  特征名 feature_names = {[str(n) for n in iris.feature_names]}")
print(f"  类别名 target_names  = {[str(n) for n in iris.target_names]}")
print(f"  类别编号分布        = {np.bincount(y).tolist()}（下标即类别编号，值即样本数）")
print("  说明：X 是二维矩阵（N×P，N=150 个样本、P=4 个特征）；y 是一维向量。")
print("        sklearn 的约定：X 永远是大写二维数组，y 是一维标签。")

# --- 第 ② 步：清洗与划分数据 -----------------------------------------------
print("\n【第 ② 步】清洗与划分数据（注意顺序：先划分，再标准化！）")
# 本数据集没有缺失值，先做一次完整性检查（真实项目里这一步是数据清洗的主要工作）
print(f"  缺失值检查：np.isnan(X).sum() = {int(np.isnan(X).sum())} 个缺失值（0 表示数据完整）")

# 2.1 先划分：stratify=y 保证训练集/测试集的类别比例与原始数据一致
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,          # 30% 作测试集 → 150 × 0.3 = 45 个样本
    random_state=RANDOM_STATE,  # 固定随机种子，划分结果可复现
    stratify=y,             # 分层抽样：三类的比例在训练/测试集中保持一致
    shuffle=True,           # 分类任务打乱是安全的（时序数据则要小心）
)
print(f"  划分结果：训练集 X_train {X_train.shape}，测试集 X_test {X_test.shape}")
print(f"  训练集类别分布 = {np.bincount(y_train).tolist()}")
print(f"  测试集类别分布 = {np.bincount(y_test).tolist()}")
print("  → 因为用了 stratify=y，两边三类都是 35/35/35 与 15/15/15，比例一致。")

# 2.2 再标准化：fit 只在训练集上做，测试集只用 transform 复用训练集统计量
scaler = StandardScaler(with_mean=True, with_std=True)  # 减均值 + 除以标准差
X_train_scaled = scaler.fit_transform(X_train)  # 学训练集统计量 + 转换训练集  ✅ 正确
X_test_scaled = scaler.transform(X_test)        # 复用训练集统计量转换测试集    ✅ 正确
# ❌ 千万不要写成：X_test_scaled = scaler.fit_transform(X_test) —— 那会把测试集的
#    均值方差泄漏进流程，等于让模型提前看到了测试集的信息。

print(f"  标准化后 训练集 均值 = {np.round(X_train_scaled.mean(axis=0), 6).tolist()}")
print(f"  标准化后 训练集 标准差 = {np.round(X_train_scaled.std(axis=0), 6).tolist()}")
print(f"  标准化后 测试集 均值 = {np.round(X_test_scaled.mean(axis=0), 4).tolist()}")
print("  → 训练集均值恰好为 0、标准差恰好为 1（因为它就是被这样定义的）。")
print("  → 测试集均值只是【接近】0 而不等于 0，这是正常的、正确的！因为它用的是训练集的")
print("     均值和标准差来映射 —— 这正是真实部署时处理新数据的方式。")

# 2.3 验证 fit_transform 与 fit().transform() 完全等价，并演示"数据泄漏"的量化差别
Z_train_a = StandardScaler().fit_transform(X_train)
Z_train_b = StandardScaler().fit(X_train).transform(X_train)
print(f"  fit_transform(X) 与 fit(X).transform(X) 结果是否完全一致："
      f"{np.allclose(Z_train_a, Z_train_b)}")
print("  → 是的，fit_transform 只是「先 fit 再 transform」的语法糖（TransformerMixin 提供）。")

leak_scaler = StandardScaler().fit(X)  # 故意在全量数据上 fit，用来演示泄漏
print(f"  泄漏演示：只用训练集算出的均值 mean_ = {np.round(scaler.mean_, 4).tolist()}")
print(f"            用全量数据算出的均值 mean_ = {np.round(leak_scaler.mean_, 4).tolist()}")
print("  → 两者数值很接近（因为 iris 数据本身均匀），但【概念上完全不同】：后者用到了测试集。")
print("     数据量大、类别不平衡或数据分布有漂移时，这个差别会被放大成严重的乐观偏差。")

# --- 第 ③ 步：运行算法（fit）-----------------------------------------------
print("\n【第 ③ 步】运行算法：在训练集上 fit（学习）")
model = RandomForestClassifier(
    n_estimators=200,       # 200 棵树：足够稳，iris 上耗时仍在毫秒级
    max_depth=None,         # 不限深度（150 个样本很简单，不会因此过拟合）
    random_state=RANDOM_STATE,  # 固定种子 → 结果可复现
    n_jobs=-1,              # 用满所有 CPU 核并行训练
)
model.fit(X_train_scaled, y_train)  # ← 只在【训练集】上学习
print(f"  模型：{type(model).__name__}（n_estimators=200, max_depth=None, random_state=42, n_jobs=-1）")
print(f"  fit 完成。学到的属性：n_features_in_ = {model.n_features_in_}，"
      f"n_classes_ = {model.n_classes_}，n_estimators = {model.n_estimators}")
print(f"  训练好的树数量 len(model.estimators_) = {len(model.estimators_)}")
print("  注意：fit 只能看到 X_train / y_train，它从始至终没有见过测试集。")

# --- 第 ④ 步：得到结果（predict / transform）------------------------------
print("\n【第 ④ 步】得到结果：对测试集 predict（预测）")
y_pred = model.predict(X_test_scaled)     # 硬预测：返回类别编号
y_proba = model.predict_proba(X_test_scaled)  # 软预测：返回每个类别的概率
print(f"  测试集前 10 个样本的预测类别：{y_pred[:10].tolist()}")
print(f"  测试集前 10 个样本的真实类别：{y_test[:10].tolist()}")
print(f"  预测概率矩阵 y_proba.shape = {y_proba.shape}（45 个样本 × 3 个类别，每行和为 1）")
print(f"  第一个样本的三类概率 = {np.round(y_proba[0], 4).tolist()}"
      f" → 预测为「{iris.target_names[y_pred[0]]}」")

# --- 第 ⑤ 步：评估 ---------------------------------------------------------
print("\n【第 ⑤ 步】评估：用 metrics 里的指标衡量泛化能力")
train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
test_acc = accuracy_score(y_test, y_pred)
# 交叉验证：把训练集再切成 5 份，轮流留一份做验证，得到 5 个分数 → 比单次划分更可靠
cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    X_train_scaled,
    y_train,
    cv=5,                   # 5 折交叉验证
    scoring="accuracy",     # 明确指定指标，不依赖 score() 的默认口径
)
cm = confusion_matrix(y_test, y_pred)

print(f"  测试集准确率 accuracy_score(y_test, y_pred) = {test_acc:.4f}")
print(f"  训练集准确率（仅供对比）                    = {train_acc:.4f}")
print(f"  5 折交叉验证分数（在训练集内部做）          = {np.round(cv_scores, 4).tolist()}")
print(f"  交叉验证均值 = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}（标准差越小越稳定）")
print()
print("  混淆矩阵 confusion_matrix(y_test, y_pred)（行 = 真实类别，列 = 预测类别）：")
print(f"    类别顺序：{[str(n) for n in iris.target_names]}")
for i, row in enumerate(cm):
    print(f"    真实 {str(iris.target_names[i]):<10} → 预测分布 {row.tolist()}")
print(f"  对角线之和 = {int(np.trace(cm))}，等于正确预测数；总样本数 = {len(y_test)}，"
      f"二者相除 = {np.trace(cm) / len(y_test):.4f}，与准确率一致。")


# ===========================================================================
# ④ 结果解读
# ===========================================================================
title("④ 结果解读：这些数字到底意味着什么？")

print(
    f"""
【数字 1】测试集准确率 = {test_acc:.4f}
  含义：在 45 个"模型从未见过"的样本中，我们预测对了 {int(round(test_acc * len(y_test)))} 个。
  {test_acc:.4f} 就是模型【泛化能力】的估计 —— 这是唯一值得对外汇报的准确率。
  作为参照：iris 三分类随机瞎猜的准确率约为 1/3 ≈ 0.333，多数类基线是 1/3（三类均衡）。
  所以 {test_acc:.4f} 明显远高于基线，说明模型确实学到了东西。

【数字 2】训练集准确率 = {train_acc:.4f}
  含义：模型在"自己学过的数据"上的表现。
  随机森林在不限深度时，训练集准确率通常就是 1.0000（每棵树都把训练样本分到纯净的叶子里）。
  ⚠️ 所以 1.0000 完全【不能】说明模型好 —— 它只说明模型有能力记住训练集。

【数字 3】训练集 vs 测试集 的差距 = {train_acc - test_acc:.4f}
  这是判断"过拟合"最直观的温度计：
    · 差距很小（如 < 0.03~0.05）  → 泛化良好，模型没有死记硬背；
    · 差距很大（训练 1.00 / 测试 0.60）→ 过拟合（Overfitting）：模型记住了训练集的
      噪声与特例，换到新数据就崩。对策：加 max_depth / min_samples_leaf 限制复杂度、
      减特征、加数据、加正则。
    · 两者都低（训练 0.70 / 测试 0.68）→ 欠拟合（Underfitting）：模型太简单，连训练集
      都没学好。对策：换更复杂的模型（如从线性模型换到随机森林/提升树）、加特征、
      提高 max_depth、减小正则强度。
  本例差距 = {train_acc - test_acc:.4f}，属于"可接受"范围：随机森林的训练集 1.0 是它的
  固有特性（Bagging 的不偏性 + 充分生长的树），关键看测试集表现是否依然很高。
  更严谨的做法是配合袋外误差 oob_score=True 或交叉验证一起看。

【数字 4】交叉验证均值 = {cv_scores.mean():.4f}（± {cv_scores.std():.4f}）
  含义：把训练集切成 5 份，轮流用 4 份训练、1 份验证，得到 5 个分数再取平均。
  为什么要做：单次 train_test_split 的结果带随机性（这一刀切在哪里会影响分数），
  交叉验证用到了全部训练数据、并给出方差，是【更可靠的泛化误差估计】。
  标准差 {cv_scores.std():.4f} 很小，说明模型在不同数据子集上表现稳定、不敏感。
  ⚠️ 重要纪律：调参只能用训练集（配合交叉验证）或单独划出的验证集，
      【绝对不能】反复看测试集分数来调参 —— 那样测试集就变成了训练集的一部分，
      最终汇报的分数会严重虚高（这就是所谓的"对测试集过拟合"）。

【数字 5】混淆矩阵
  行是真实类别，列是预测类别。对角线上的数字是分对的样本数，其余都是分错的。
  本例中：setosa 被 100% 正确识别（它与其他两类线性可分）；错误主要集中在
  versicolor 与 virginica 之间（这两类在特征空间中有重叠，本身就难分）。
  混淆矩阵比一个准确率数字更有价值：它告诉你"错在哪里"，从而指导下一步优化方向
  （比如去收集更多 versicolor/virginica 的样本，或加入更有区分度的特征）。

【一句话总结如何判断模型好不好】
  看三个数的关系：测试集分数（要高）、训练-测试差距（要小）、交叉验证标准差（要小）。
  只报一个漂亮的训练集准确率，在工程上等于没有评估。
"""
)


# ===========================================================================
# 可视化：保存 iris 特征分布图（标准化前 vs 标准化后）
# ===========================================================================
title("绘图：iris 特征分布（标准化前 vs 标准化后），保存为 PNG")

feature_names_en = list(iris.feature_names)
feature_names_zh = ["花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度"]
class_names_zh = ["山鸢尾 (setosa)", "变色鸢尾 (versicolor)", "维吉尼亚鸢尾 (virginica)"]
colors = ["#4C72B0", "#DD8452", "#55A868"]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("鸢尾花（Iris）数据集：标准化前 vs 标准化后 及 类别分布", fontsize=15)

# 左上：标准化前的 4 个特征（原始量纲，单位都是 cm，但数值范围不同）
ax = axes[0, 0]
ax.boxplot([X[:, i] for i in range(X.shape[1])], tick_labels=feature_names_zh)
ax.set_title("标准化前：4 个特征的原始分布")
ax.set_ylabel("数值（cm）")
ax.grid(axis="y", alpha=0.3)

# 右上：标准化后的 4 个特征（同一坐标系，均值 0 标准差 1，可直接比较形状）
ax = axes[0, 1]
ax.boxplot([X_train_scaled[:, i] for i in range(X_train_scaled.shape[1])],
           tick_labels=feature_names_zh)
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_title("标准化后：均值 0、标准差 1（训练集）")
ax.set_ylabel("标准化数值（z-score）")
ax.grid(axis="y", alpha=0.3)

# 左下：花瓣长度按类别分开看 —— 直观解释"为什么这个任务好学"
ax = axes[1, 0]
petal_length_idx = 2
for idx, (name, color) in enumerate(zip(class_names_zh, colors)):
    ax.hist(X[y == idx, petal_length_idx], bins=10, alpha=0.6, label=name, color=color)
ax.set_title(f"按类别的「{feature_names_zh[petal_length_idx]}」分布（标准化前）")
ax.set_xlabel(f"{feature_names_zh[petal_length_idx]}（cm）")
ax.set_ylabel("样本数")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

# 右下：训练集 / 测试集的类别样本数 —— 展示 stratify=y 的效果
ax = axes[1, 1]
x_pos = np.arange(len(class_names_zh))
width = 0.38
train_counts = np.bincount(y_train, minlength=len(class_names_zh))
test_counts = np.bincount(y_test, minlength=len(class_names_zh))
ax.bar(x_pos - width / 2, train_counts, width, label=f"训练集（共 {len(y_train)}）", color="#4C72B0")
ax.bar(x_pos + width / 2, test_counts, width, label=f"测试集（共 {len(y_test)}）", color="#C44E52")
ax.set_xticks(x_pos)
ax.set_xticklabels(class_names_zh, fontsize=8)
ax.set_title("分层抽样（stratify=y）后的类别样本数")
ax.set_ylabel("样本数")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(FIGURE_PATH, dpi=120, bbox_inches="tight")
plt.close(fig)  # 关闭图形释放内存；本脚本【绝不】调用 plt.show()，避免无界面环境阻塞

print(f"  已保存图片：{FIGURE_PATH}")
print(f"  文件是否存在：{FIGURE_PATH.exists()}，大小 = {FIGURE_PATH.stat().st_size} 字节")
print("  提示：命令行脚本用 Agg 后端 + savefig 是标准做法；如需交互式查看（Jupyter/IDE），")
print("        可把 matplotlib.use('Agg') 换成 matplotlib.use('TkAgg') 并把 savefig 改为 plt.show()。")


# ===========================================================================
# 超参数怎么调
# ===========================================================================
title("附：超参数怎么调（实战经验）")

print(
    """
所谓"超参数"就是【不能通过 fit 从数据里学出来、必须由人指定】的参数，例如树的数量、
最大深度、学习率。调参的方法论如下：

【0. 先立评估标准，再动手调】
  必须有一个可信的验证口径，推荐：
      cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")
  或先 train_test_split 出独立的验证集。绝不允许用测试集当调参依据。

【1. 分清"该调什么"——先粗后细】
  · 影响泛化能力的关键旋钮（优先调）：
      RandomForestClassifier: max_depth、min_samples_leaf、min_samples_split、max_features
      提升树（XGBoost/LightGBM）: learning_rate、n_estimators、max_depth、subsample
      线性模型: 正则强度 C（逻辑回归/SVM）或 alpha（Ridge/Lasso）
  · 主要影响速度、几乎不影响精度的参数（按资源决定）：
      n_jobs、n_estimators（超过某点后收益递减，通常 300~500 足够）

【2. 调参手段：从粗到精的三级火箭】
  ① 手工/经验起点：随机森林先跑 n_estimators=200, max_depth=None 拿一个基线。
  ② GridSearchCV —— 网格搜索：把小范围候选值穷举，用交叉验证选最优。
        param_grid = {"n_estimators": [100, 200, 400], "max_depth": [None, 5, 10],
                      "min_samples_leaf": [1, 2, 4]}
        grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid,
                            cv=5, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        print(grid.best_params_, grid.best_score_)
        注意：候选组合数是【乘积】增长，参数一多就爆炸（"维度灾难"）。
  ③ RandomizedSearchCV —— 随机搜索：给定每个参数的分布，随机采样 N 组（如 n_iter=50）。
        高维空间里通常比网格搜索更快找到好点，因为不是所有参数都同等重要。
  · 更进一步的自动化：贝叶斯优化（Optuna / scikit-optimize），用已评估点的信息指导下一个
    采样位置，样本效率远高于随机搜索。

【3. 防过拟合 / 防欠拟合 的调参方向（口诀）】
  过拟合（训练高、验证低）→ 让模型更"笨"：
      ↑ min_samples_leaf / min_samples_split、↓ max_depth、↓ max_features、加正则、加数据
  欠拟合（训练低、验证低）→ 让模型更"聪明"：
      ↑ max_depth、↑ n_estimators、加特征/做特征工程、换更强的模型族
  数据不平衡 → class_weight="balanced"，并把 scoring 换成 f1 / roc_auc / average_precision。

【4. 调参纪律（踩过坑才懂）】
  (a) 所有预处理（标准化等）必须放进 Pipeline，否则 GridSearchCV 的每一折都会泄漏：
        pipe = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=42))
        GridSearchCV(pipe, param_grid, cv=5)   # 标准化会在每一折内部重新 fit，正确
  (b) 固定 random_state，否则两次搜索结果无法比较、也无法复现。
  (c) 一次只调一两个参数、观察趋势，比一次性 6 个参数网格更高效也更有信息量。
  (d) 别追小数点后第三位：交叉验证的标准差往往比你要调出来的提升还大，
      提升小于标准差的"最优参数"很可能只是噪声。
  (e) 调完记得用【之前从未参与调参】的测试集做最后一次评估，并且只做一次。
"""
)

print()
print("【完成】01_sklearn核心模块与标准工作流程.py 运行结束")

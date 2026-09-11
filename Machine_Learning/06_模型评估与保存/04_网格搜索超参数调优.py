r"""《机器学习》课案 —— 06 模型评估与保存 / 04 网格搜索超参数调优

对应课案章节
------------
《机器学习》课案 "模型评估与保存" 章 → "超参数调优" 节
（课案原文第 1442~1502 行；本章 5 个代码块中的第 4 个：
 load_iris + RandomForestClassifier + GridSearchCV(cv=5, scoring="accuracy") +
 best_params_ / best_score_ / best_estimator_ / 测试集准确率）。

本节知识点
----------
1. 参数（parameter）与超参数（hyperparameter）的区别：谁由数据学出来、谁必须人来定；
2. 网格搜索的暴力枚举本质与计算复杂度 O(组合数 × K 折 × 单次训练耗时)；
3. 为什么**必须**用交叉验证来选超参：直接看测试集分数选超参 = 数据泄漏，分数虚高；
4. cv_results_ 里每个字段的含义，以及如何整理成 pandas DataFrame 做分析；
5. best_estimator_ 与 refit 参数的关系（refit=True 时才会自动用最优参数在全量训练集上重训）；
6. n_jobs 与耗时/内存的权衡；
7. 用真实实验量化"用测试集挑超参"能带来多少虚假的乐观偏差。

运行方式（在 PowerShell 中复制执行）
------------------------------------
$env:PYTHONUTF8='1'; & 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Machine_Learning\06_模型评估与保存\04_网格搜索超参数调优.py'

产出
----
Machine_Learning/output/06_网格搜索热力图.png
Machine_Learning/output/06_网格搜索过拟合诊断.png
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")            # 无界面后端，避免 plt.show() 阻塞
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import time

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    StratifiedKFold,
    train_test_split,
)

# =============================================================================
# ① 原理与数学推导
# =============================================================================
# 1.1 参数 vs 超参数
# ------------------
#   * **参数（parameter）**：由训练算法**从数据里学出来**的量，存在模型对象里，
#     例如线性回归的 coef_ / intercept_、随机森林里每棵树的 split 阈值、
#     神经网络的权重。写 fit() 之后自动就有了，不需要人操心。
#   * **超参数（hyperparameter）**：**必须在训练开始前由人指定**的量，
#     例如随机森林的 n_estimators（几棵树）、max_depth（树多深）、
#     学习率、正则系数 C / alpha、KNN 的 k。
#     它们控制"模型的容量/复杂度"，学习算法无法自己决定，只能靠搜索 + 评估来选。
#   一句话：**参数是学出来的，超参数是搜出来的。**
#
# 1.2 网格搜索（Grid Search）的数学形式
# -------------------------------------
# 给定超参数空间 Λ = Λ_1 × Λ_2 × ... × Λ_m（笛卡尔积），网格搜索求解：
#       λ* = argmax_{λ ∈ Λ}  (1/K) · Σ_{k=1..K} Score( f_{λ, D_train\{D_k}} , D_k )
#   即：**对网格中的每一个 λ，做一次完整的 K 折交叉验证，取平均分最高的那个 λ**。
#   计算复杂度（"暴力枚举"的代价）：
#       O( |Λ| × K × T_fit )
#   其中 |Λ| = ∏|Λ_i| 是组合总数，K 是折数，T_fit 是单次训练耗时。
#   例：本脚本 |Λ| = 2(n_estimators) × 3(max_depth) = 6，K = 5，
#       所以总共要训练 6 × 5 = 30 个模型；refit=True 时再补 1 次全量重训 = 31 次。
#   网格搜索的两个致命缺点：
#       (a) **维度灾难**：参数从 2 个增到 10 个、每个取 5 个值，组合数就是 5^10 ≈ 976 万；
#       (b) 连续型超参数（如学习率 0.001~1）网格只能取离散点，可能整个错过最优区域。
#   替代方案：RandomizedSearchCV（随机采样，Berestycki 定理保证在同样预算下更容易
#   命中好区域）、HalvingGridSearchCV（逐次减半，先粗后细）、
#   贝叶斯优化（Optuna / hyperopt 等第三方库）。
#
# 1.3 为什么**必须**用交叉验证选超参（数据泄漏）
# ----------------------------------------------
# 错误做法：把数据切成 train/test，对每个候选 λ 在 train 上训练、**在 test 上打分**，
#           挑 test 分数最高的 λ，再报告这个 test 分数。
# 为什么错：
#   * 此时测试集**参与了模型选择**，它不再是"没见过的新数据"；
#   * 我们实际上在 6 个候选里挑了对这 45 个测试样本最"幸运"的那个，
#     等价于对测试集做了 6 次"偷看"，得到的分数是**乐观偏差**的估计；
#   * 候选组合越多，偏差越大（这也是"排行榜过拟合"的成因）。
# 正确做法：在**训练集内部**再用交叉验证切分（GridSearchCV 帮你做了），
#           选完超参后，测试集只用来看一次。
#   → 铁律：**任何"比较与选择"都必须发生在训练集/验证集内部。**
#
# 1.4 cv_results_ 有哪些字段
# ---------------------------
#   params                 : 该组合的参数 dict（好用但难做表格，下面会拆开成列）；
#   param_<名字>           : 把每个超参数单独拆成一列（**做 DataFrame 分析时最有用**）；
#   mean_test_score        : 该组合 K 折验证分数的**均值**，就是用来排序的分数；
#   std_test_score         : 该组合 K 折验证分数的**标准差**，衡量稳定性；
#   rank_test_score        : 名次（1 = 最好）；**允许并列**，所以可能出现两个 rank=1；
#   split<i>_test_score    : 第 i 折的具体分数（调试时看某一折为何异常）；
#   mean_train_score / std_train_score : 需要 return_train_score=True 才有；
#   mean_fit_time / std_fit_time / mean_score_time : 训练与打分耗时；
#   params 与分数的组合就是"超参数搜索的完整实验记录"，**务必保存下来**。
#
# 1.5 best_estimator_ 与 refit
# ----------------------------
#   * refit=True（默认）：网格搜索结束后，自动用 best_params_ 在**全部训练数据**上
#     重新 fit 一次，把结果放进 best_estimator_，于是可以直接 .predict()。
#     为什么值得这一次额外训练？因为 CV 里的每个模型只用了 (K-1)/K 的训练数据，
#     用全量数据重训能多利用这部分数据，通常略好。
#   * refit=False：不重训，best_estimator_ **不存在**，访问会抛 AttributeError；
#     只有 best_params_ / best_index_ / cv_results_ 可用。
#     什么场景用？当你只想拿最优超参数、打算自己用别的框架/数据重训时，可以省这一次训练。
#   * refit 也可以传一个字符串（多指标时指定用哪个指标来选最优并重训）。
#
# 1.6 n_jobs 的权衡
# -----------------
#   * n_jobs=None/1：单进程串行，最省内存、最容易调试；小数据（如 iris）时往往**最快**，
#     因为多进程有启动开销和"派发任务"的通信开销。
#   * n_jobs=-1：用满所有 CPU 核心。只有单次训练较慢、组合数较多时才划算
#     （经验阈值：整个搜索超过几秒到几十秒）。
#   * n_jobs>1 的代价：(a) 每个进程都要复制一份数据（内存 × 核数）；
#     (b) 与某些已并行/已多线程的估计器（如 n_jobs 已开满的 RF）叠加会争抢 CPU；
#     (c) 报错信息被包在子进程里，调试更难。
#   * 结论：**先用 n_jobs=1 跑通，确认逻辑无误后再按需加大**——这也是课案示例的写法。

# =============================================================================
# ② sklearn API 关键参数逐个解释
# =============================================================================
# 【GridSearchCV(estimator, param_grid, *, scoring, n_jobs, refit, cv, verbose,
#               return_train_score, error_score, pre_dispatch)】
#   estimator        : 未训练的估计器（会被反复 clone）。**不要传已经 fit 过的实例**。
#   param_grid       : dict 或 list[dict]。
#                      dict  形式：{"n_estimators": [50,100], "max_depth": [3,5,None]}
#                      list  形式：[{"kernel":["rbf"],"C":[1,10]}, {"kernel":["linear"],"C":[1,10]}]
#                             用于"不同核函数配不同参数"这类非笛卡尔积搜索。
#                      嵌套参数用双下划线：例如 Pipeline 里写 "clf__max_depth"。
#   scoring          : 指标名（"accuracy"/"f1"/"roc_auc"/"neg_mean_squared_error"…）
#                      或 dict（多指标）。传 None 用 estimator.score()。
#   cv               : 折数或切分器。分类任务传整数时**自动 StratifiedKFold**。
#   n_jobs           : 并行数，见 1.6。
#   refit=True       : 是否用最优参数在全量训练集上重训，见 1.5。
#   verbose=0        : >0 打印搜索进度；大数据上设 1~2 能看到"搜到哪了"。
#   return_train_score=False : 打开后 cv_results_ 会多出 mean_train_score，
#                      用于诊断"最优组合是不是过拟合出来的"（本脚本打开）。
#   error_score="raise" : 某组参数训练失败时是抛错还是记 nan。**生产脚本建议 "raise"**，
#                      避免某组参数悄悄全失败而你还在看"best_params_"。
#   pre_dispatch      : 一次性派发多少任务（并行时控制内存峰值），默认 "2*n_jobs"。
#   拟合后可用的属性：
#       best_params_ / best_score_ / best_index_ / best_estimator_ /
#       cv_results_ / scorer_ / n_splits_ / refit_time_ / multimetric_
#
# 【RandomForestClassifier 的关键超参数（就是我们要搜的东西）】
#   n_estimators  : 树的数量。**越多越稳、越慢，但不会过拟合**（方差随数量下降后饱和）。
#                   常搜 [50, 100, 200, 300]；这是"性价比"最低的一个参数，通常最后调。
#   max_depth     : 单棵树最大深度，控复杂度最重要的一把手。None = 不限深。
#                   常搜 [3, 5, 7, 10, None]；**它是对过拟合影响最大的参数**，应优先调。
#   min_samples_split / min_samples_leaf : 内部节点/叶子最少样本数，也是防过拟合的旋钮；
#                   常搜 [2, 5, 10]。min_samples_leaf 的效果通常比 max_depth 更平滑。
#   max_features  : 每次分裂考察的特征数，'sqrt'(默认) / 'log2' / None / 整数 / 浮点比例。
#                   它是随机森林"去相关"的核心，**越小树越独立、方差越低**。
#   bootstrap / oob_score : 是否有放回抽样 / 是否用袋外样本估泛化误差。
#   class_weight  : 不平衡分类时设 "balanced"（见 01 号脚本）。
#   criterion     : 'gini'(默认) / 'entropy' / 'log_loss'。
#   n_jobs        : 树级别的并行。**注意与 GridSearchCV 的 n_jobs 相乘**，
#                   两个都设 -1 会导致 CPU 严重超订，反而更慢。
#   random_state  : 固定随机种子。**强烈建议固定**，否则同一次网格搜索里
#                   不同折之间混入了随机性，分数比较会不可靠。
#
# 【ParameterGrid(param_grid)】把 dict 展开成所有参数组合的迭代器（len() 就是 |Λ|），
#   本脚本用它来枚举"假装用测试集挑超参"的那组实验。
#
# 【StratifiedKFold】见 03 号脚本注释；本脚本显式构造它并传给 cv，
#   好处是能把 random_state 固定下来，让网格搜索结果可复现。

# =============================================================================
# ③ 完整可运行代码
# =============================================================================
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("=" * 78)
print("06-04 超参数调优：GridSearchCV 网格搜索（含数据泄漏对照实验）")
print("=" * 78)

# --- 步骤 1：加载数据并划分（对应课案代码块 4） ------------------------------
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)
print("\n[1] 数据与划分")
print(f"    iris 共 {X.shape[0]} 个样本、{X.shape[1]} 个特征、{len(iris.target_names)} 个类别"
      f"（{', '.join(iris.target_names)}）")
print(f"    训练集 {X_train.shape[0]} 个，测试集 {X_test.shape[0]} 个（stratify=y 保持类别比例）")
print("    注意：**测试集从这里开始就被封存**，选超参只看交叉验证分数，最后才用它一次。")

# --- 步骤 2：定义超参数网格 ------------------------------------------------
model = RandomForestClassifier(random_state=RANDOM_STATE)
param_grid = {
    "n_estimators": [50, 100],          # 树的数量
    "max_depth": [3, 5, None],          # 树的最大深度，None = 不限深
}
n_combos = len(list(ParameterGrid(param_grid)))
CV_FOLDS = 5
print("\n[2] 超参数网格")
for key, values in param_grid.items():
    print(f"    {key:<14s} 候选值 = {values}")
print(f"    组合总数 |Λ| = 2 × 3 = {n_combos}")
print(f"    网格搜索要训练的模型数 = |Λ| × K = {n_combos} × {CV_FOLDS} = {n_combos * CV_FOLDS}"
      f"（refit=True 再补 1 次全量重训 = {n_combos * CV_FOLDS + 1} 次）")
print("    → 这就是「暴力枚举」的代价：组合数 × 折数 × 单次训练时间。")

# --- 步骤 3：执行网格搜索（对应课案代码块 4 的核心） ------------------------
cv_splitter = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=cv_splitter,              # 固定随机种子的分层 5 折，保证可复现
    scoring="accuracy",
    n_jobs=1,                    # 小数据单进程反而更快，且报错信息清晰
    refit=True,                  # 搜完自动用最优参数在全量训练集上重训
    return_train_score=True,     # 额外记录训练分数，用于诊断过拟合
    error_score="raise",         # 任何一组参数失败都立刻抛出，不静默吞掉
)
t0 = time.perf_counter()
grid_search.fit(X_train, y_train)
elapsed = time.perf_counter() - t0

print(f"\n[3] 网格搜索完成，耗时 {elapsed:.3f} 秒")
print(f"    最优参数 best_params_ = {grid_search.best_params_}")
print(f"    交叉验证最高准确率 best_score_ = {grid_search.best_score_:.4f}")
print(f"    最优组合在 cv_results_ 中的下标 best_index_ = {grid_search.best_index_}")
print(f"    评价器 scorer_ = {grid_search.scorer_}")
print(f"    实际折数 n_splits_ = {grid_search.n_splits_}，"
      f"refit 耗时 refit_time_ = {grid_search.refit_time_:.4f} 秒")
print(f"    → best_score_ 是**交叉验证的平均分**，不是测试集分数，也不是训练集分数；")
print(f"      它是对「这个超参数配置的泛化能力」的估计，因此可以用来和其他配置比较。")

# --- 步骤 4：用最优模型在测试集上评估（只此一次） --------------------------
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
train_acc_best = accuracy_score(y_train, best_model.predict(X_train))
print("\n[4] 用 best_estimator_ 在测试集上评估（测试集全程只用了这一次）")
print(f"    best_estimator_ = {best_model}")
print(f"    训练集准确率 = {train_acc_best:.4f}")
print(f"    测试集准确率 = {test_acc:.4f}")
print(f"    交叉验证 best_score_ = {grid_search.best_score_:.4f}")
gap_cv_test = abs(test_acc - grid_search.best_score_)
print(f"    → 三者关系解读：")
print(f"      训练 {train_acc_best:.4f} ≥ 交叉验证 {grid_search.best_score_:.4f} "
      f"（训练数据被模型见过，分数天然偏高）")
print(f"      测试 {test_acc:.4f} 与交叉验证 {grid_search.best_score_:.4f} 相差 "
      f"{gap_cv_test:.4f}，折算成样本数约 {gap_cv_test * len(y_test):.1f} 个"
      f"（测试集只有 {len(y_test)} 个样本，单样本价值 1/{len(y_test)} = "
      f"{1 / len(y_test):.4f}）。")
if gap_cv_test <= 1.5 / len(y_test):
    print("      → 差距在一个样本量级以内：交叉验证的估计与测试集结果一致，CV 可靠。")
elif gap_cv_test <= 4 / len(y_test):
    print("      → 差距约 1~4 个样本，属于小测试集固有的抽样波动（CV 用了 105 个样本评估，")
    print("        测试集只有 45 个，后者的方差天然更大）。两者量级一致（都在 0.9 上下），")
    print("        不必因为这点差别就推翻模型选择结果；但也不该只报 CV 分数而回避测试分数。")
else:
    print("      → 差距超过 4 个样本，需要警惕：可能是数据分布不一致、CV 流程存在泄漏，")
    print("        或者超参数被搜索过程「过拟合」了。")

# --- 步骤 5：把 cv_results_ 整理成 DataFrame（工程上必做） -----------------
cv_df = pd.DataFrame(grid_search.cv_results_)
cv_df["max_depth_标签"] = cv_df["param_max_depth"].apply(
    lambda v: "None(不限深)" if v is None else str(v)
)
show_cols = ["params", "max_depth_标签", "param_n_estimators",
             "mean_test_score", "std_test_score", "rank_test_score",
             "mean_train_score", "mean_fit_time"]
cv_show = cv_df[show_cols].sort_values("rank_test_score").reset_index(drop=True)
print("\n[5] cv_results_ 整理成表格（按名次排序，共"
      f" {len(cv_show)} 行 = 网格组合数）")
print(cv_show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\n    字段含义速查：")
print("      params            —— 这一行的超参数组合")
print("      mean_test_score   —— K 折验证分数的均值（用来排序的分数）")
print("      std_test_score    —— K 折验证分数的标准差（稳定性；越小越可信）")
print("      rank_test_score   —— 名次，1 为最优；**允许并列**")
print("      mean_train_score  —— K 折训练分数的均值（需要 return_train_score=True）")
print("      mean_fit_time     —— 该组合平均每折训练耗时（秒）")

best_row = cv_df.loc[grid_search.best_index_]
print(f"\n    最优组合细节：{grid_search.best_params_}")
print(f"      5 折验证分数 = "
      + ", ".join(f"{best_row[f'split{i}_test_score']:.4f}" for i in range(CV_FOLDS)))
print(f"      均值 {best_row['mean_test_score']:.4f}，标准差 {best_row['std_test_score']:.4f}")
print(f"      训练分数均值 {best_row['mean_train_score']:.4f}，"
      f"与验证分数相差 {best_row['mean_train_score'] - best_row['mean_test_score']:.4f}")
print(f"      平均每折训练耗时 {best_row['mean_fit_time']:.4f} 秒")
gap = best_row["mean_train_score"] - best_row["mean_test_score"]
print(f"      另有：训练分数的**样本粒度**是 1/{len(y_train)} = {1 / len(y_train):.4f}，"
      f"所以 {gap:.4f} 的差距实际上只相当于 {gap * len(y_train):.1f} 个训练样本。")
if gap < 0.02:
    print("      → 训练与验证分数几乎一致，说明这组超参数**没有过拟合**。")
elif gap < 0.06:
    print("      → 差距属于「轻微」区间：在只有 105 个训练样本的情况下，")
    print("        一两个样本的差别就足以造成这种量级的波动，**不必当成过拟合来治**。")
else:
    print("      → 训练分数明显高于验证分数，该组合有过拟合倾向，应加大正则"
          "（减小 max_depth、增大 min_samples_leaf）。")
deepest_rows = cv_df[cv_df["max_depth_标签"] != "3"]
if len(deepest_rows) > 0:
    tr_deep = deepest_rows["mean_train_score"].max()
    te_deep = deepest_rows["mean_test_score"].max()
    print(f"      更值得注意的对比：max_depth=5 或不限深的组合，训练分数最高达到 "
          f"{tr_deep:.4f}（背下了训练集），")
    print(f"      但它们的交叉验证分数只有 {te_deep:.4f}，**低于 max_depth=3 的 "
          f"{best_row['mean_test_score']:.4f}**。")
    print("      → 这正是「训练集满分不等于泛化好」的直接证据，也说明**必须用验证分数选超参**，")
    print("        如果误用训练分数选模型，一定会选中过拟合最深的那一个。")

# 用 DataFrame 做分析：看每个超参数维度对分数的影响（边际效应）
print("\n[6] 用 groupby 看单个超参数对分数的影响（边际效应分析）")
for col, name in [("param_n_estimators", "n_estimators"), ("max_depth_标签", "max_depth")]:
    grouped = cv_df.groupby(col, as_index=False)["mean_test_score"].mean().sort_values(
        "mean_test_score", ascending=False
    )
    def _fmt(value: object) -> str:
        """把 numpy 整数/浮点格式化成可读标签（避免出现 50.0 这种写法）；字符串原样返回。"""
        if isinstance(value, str):
            return value
        fvalue = float(value)          # type: ignore[arg-type]
        return str(int(fvalue)) if fvalue.is_integer() else f"{fvalue:g}"

    desc = "，".join(f"{_fmt(row[col])}→{row['mean_test_score']:.4f}"
                     for _, row in grouped.iterrows())
    print(f"    按 {name:<13s} 平均：{desc}")
print("    → 注意：这种「边际平均」只反映趋势，真实交互作用要看下面的热力图。")

# --- 步骤 6：数据泄漏对照实验（用测试集挑超参会虚高多少） -------------------
peek_records = []
for params in ParameterGrid(param_grid):
    estimator = RandomForestClassifier(random_state=RANDOM_STATE, **params)
    estimator.fit(X_train, y_train)
    peek_acc = accuracy_score(y_test, estimator.predict(X_test))
    peek_records.append(
        {
            "params": params,
            "测试集准确率": peek_acc,
        }
    )
peek_df = pd.DataFrame(peek_records).sort_values("测试集准确率", ascending=False)
print("\n[7] 数据泄漏对照实验：如果**用测试集来挑超参**会怎样？")
print(peek_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
peek_best = float(peek_df["测试集准确率"].max())
print(f"\n    「作弊式」做法：直接看测试集分数，挑最高的一组 → 报告 {peek_best:.4f}")
print(f"    正确做法：用交叉验证选超参（best_params_ = {grid_search.best_params_}），"
      f"再在测试集上评估一次 → {test_acc:.4f}")
print(f"    两者相差 {peek_best - test_acc:+.4f}。")
print("    → 本次差距不大，因为：iris 太小、候选只有 6 组、而且很多组合在测试集上"
      "分数相同（并列最优）。")
print("      但原理上这个偏差**必然 ≥ 0**：在 N 个候选中挑「对测试集最幸运」的那个，")
print("      得到的分数只会高于或等于诚实流程的分数。**候选越多，虚高越严重。**")
print("      在 Kaggle/竞赛里，用公开榜单反复挑模型就是同一个道理，")
print("      所以必须留一份真正封存的私有测试集。")

# =============================================================================
# ④ 结果解读
# =============================================================================
print("\n" + "=" * 78)
print("④ 结果解读：这些数字到底说明什么")
print("=" * 78)
if gap_cv_test <= 1.5 / len(y_test):
    cv_test_judgement = "两者基本一致，说明交叉验证没有骗你。"
elif gap_cv_test <= 4 / len(y_test):
    cv_test_judgement = (
        "差距约 1~4 个样本，属于小测试集固有的抽样波动：\n"
        "      交叉验证是在 105 个训练样本上估计的，测试集只有 45 个样本，\n"
        "      后者方差天然更大，因此测试分数略低是正常现象，不必据此推翻模型选择。\n"
        "      但要记住：**对外报告时应给出测试集分数，并说明 CV 分数与它的差距**，\n"
        "      只报更好看的那个数字是不诚实的。"
    )
else:
    cv_test_judgement = (
        "差距超过 4 个样本，需要警惕：可能是数据分布不一致、CV 流程泄漏，\n"
        "      或者超参数被搜索过程过拟合了。"
    )
print(f"""
【1】最优超参数 = {grid_search.best_params_}，交叉验证准确率 = {grid_search.best_score_:.4f}。
      怎么读：把训练集分成 {CV_FOLDS} 折、轮流当验证集，用这组超参数训练出的模型
      平均能答对约 {grid_search.best_score_:.1%} 的样本，各折波动 ±
      {best_row['std_test_score']:.4f}。
      这个数字是"选超参阶段"的成绩，**不是**最终对外报告的成绩。

【2】测试集准确率 = {test_acc:.4f}，与 CV 的 {grid_search.best_score_:.4f} 相差
      {gap_cv_test:.4f}（≈{gap_cv_test * len(y_test):.1f} 个测试样本，一个样本值
      {1 / len(y_test):.1%}）。
      {cv_test_judgement}
      另外要注意：CV 的 {grid_search.best_score_:.4f} 是**在训练集内部、按最优参数定义**
      做交叉验证得到的分数，它带有「从 6 个候选里挑最好」的乐观偏差，
      所以它略高于诚实的测试集分数是符合预期的。

【3】看 rank_test_score 列可以发现：本网格中好几个组合的分数非常接近甚至并列。
      这很正常，iris 是个"太简单"的数据集 —— 只有 150 个样本、3 个类别基本线性可分，
      随机森林在这种数据上怎么设超参都接近满分。
      → 教学要点：**在简单数据集上调参，往往看不出差异；调参的价值要在难数据上才体现。**
        真正的判断依据应是 std_test_score（稳定性）+ 业务代价，而不是小数第三位的高低。

【4】关于 best_estimator_：因为 refit=True，网格搜索结束后 sklearn 自动用
      best_params_ 在**全部 {X_train.shape[0]} 个训练样本**上重训了一次
      （耗时 {grid_search.refit_time_:.4f} 秒，已包含在上面的总耗时里）。
      这次重训不是浪费：CV 里每个模型只见过 (K-1)/K = {CV_FOLDS - 1}/{CV_FOLDS}
      的训练数据，全量重训能把这部分补回来，通常略好一点。
      如果设 refit=False，best_estimator_ 这个属性根本不存在（访问会抛 AttributeError），
      你只能拿到 best_params_ 自己再训练。

【5】关于耗时：本次 {n_combos} 组 × {CV_FOLDS} 折 = {n_combos * CV_FOLDS} 次训练
      总耗时 {elapsed:.3f} 秒（n_jobs=1）。
      因为数据太小，多进程的启动/通信开销反而可能更大 —— 这也是为什么课案示例
      在 iris 上不设 n_jobs=-1。**先单进程跑通，再按需并行**。

【6】调参的正确顺序（避免在错误的维度上浪费时间）：
      1) 先把数据与评估流程搭稳（分层、Pipeline、固定随机种子、确认基线分数）；
      2) 调**影响最大**的超参数：max_depth / min_samples_leaf（控复杂度）；
      3) 再调 max_features（控随机性/去相关）；
      4) 最后加 n_estimators（只影响稳定性，越大越好直到边际收益饱和）；
      5) 不平衡数据别忘 class_weight；
      6) 预算不足就换 RandomizedSearchCV，不要死磕网格。
""")

# --- 绘图 1：热力图（max_depth × n_estimators） -----------------------------
pivot = cv_df.pivot_table(
    index="max_depth_标签",
    columns="param_n_estimators",
    values="mean_test_score",
    aggfunc="mean",
)
# 固定行顺序，让"树越深"从上到下递增，便于观察趋势
row_order = [lab for lab in ["3", "5", "None(不限深)"] if lab in pivot.index]
pivot = pivot.loc[row_order]

fig, ax = plt.subplots(figsize=(7.4, 5.2))
sns.heatmap(
    pivot,
    annot=True,
    fmt=".4f",
    cmap="YlGnBu",
    vmin=float(pivot.values.min()) - 0.01,
    vmax=1.0,
    linewidths=0.6,
    linecolor="white",
    cbar_kws={"label": "交叉验证平均准确率 mean_test_score"},
    ax=ax,
)
ax.set_title(f"06 网格搜索热力图：max_depth × n_estimators\n"
             f"（5 折交叉验证平均准确率，最优 {grid_search.best_score_:.4f}）",
             fontsize=13, pad=12)
ax.set_xlabel("n_estimators（树的数量）", fontsize=11)
ax.set_ylabel("max_depth（单棵树最大深度）", fontsize=11)
fig.tight_layout()
p1 = OUTPUT_DIR / "06_网格搜索热力图.png"
fig.savefig(p1, dpi=130)
plt.close(fig)
print(f"\n[图 1] 已保存网格搜索热力图：{p1}")

# --- 绘图 2：过拟合诊断（训练分数 vs 验证分数 + 耗时） ----------------------
fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.9))
combo_labels = [f"树{int(r['param_n_estimators'])}\n深{r['max_depth_标签']}"
                for _, r in cv_df.iterrows()]
xpos = np.arange(len(cv_df))

ax0 = axes[0]
ax0.plot(xpos, cv_df["mean_train_score"], marker="o", linewidth=2, color="#8e44ad",
         label="训练分数均值")
ax0.plot(xpos, cv_df["mean_test_score"], marker="s", linewidth=2, color="#27ae60",
         label="验证分数均值（CV）")
ax0.fill_between(xpos,
                 cv_df["mean_test_score"] - cv_df["std_test_score"],
                 cv_df["mean_test_score"] + cv_df["std_test_score"],
                 color="#27ae60", alpha=0.15, label="验证分数 ±1 标准差")
ax0.set_xticks(xpos)
ax0.set_xticklabels(combo_labels, fontsize=8)
ax0.set_ylim(0.85, 1.02)
ax0.set_title("06 每个超参数组合的训练/验证分数（差距 = 过拟合程度）", fontsize=12, pad=10)
ax0.set_xlabel("超参数组合", fontsize=11)
ax0.set_ylabel("准确率", fontsize=11)
ax0.legend(fontsize=9, loc="lower right")
ax0.grid(alpha=0.3)

ax1 = axes[1]
colors = ["#c0392b" if i == grid_search.best_index_ else "#2980b9" for i in range(len(cv_df))]
ax1.bar(xpos, cv_df["mean_fit_time"] * 1000, color=colors, alpha=0.88)
for i, v in enumerate(cv_df["mean_fit_time"] * 1000):
    ax1.annotate(f"{v:.1f}", xy=(i, v), xytext=(0, 3), textcoords="offset points",
                 ha="center", fontsize=8)
ax1.set_xticks(xpos)
ax1.set_xticklabels(combo_labels, fontsize=8)
ax1.set_title("06 各组合的单折平均训练耗时（红色 = 最优组合）", fontsize=12, pad=10)
ax1.set_xlabel("超参数组合", fontsize=11)
ax1.set_ylabel("平均每折训练耗时（毫秒）", fontsize=11)
ax1.grid(alpha=0.3, axis="y")
fig.tight_layout()
p2 = OUTPUT_DIR / "06_网格搜索过拟合诊断.png"
fig.savefig(p2, dpi=130)
plt.close(fig)
print(f"[图 2] 已保存网格搜索诊断图：{p2}")

# 顺手把搜索结果存档为 CSV，便于事后复查（工程习惯；不属于课案要求）
csv_path = OUTPUT_DIR / "06_网格搜索结果.csv"
cv_show.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"[附] 已保存网格搜索结果表：{csv_path}")

# =============================================================================
# 超参数怎么调 / 使用注意
# =============================================================================
# 【随机森林调参的实战优先级（从收益最大的开始）】
# 1. max_depth / min_samples_leaf / min_samples_split：控制树的复杂度，**优先调**。
#    树太深 → 训练分高、验证分低（过拟合）；树太浅 → 两者都低（欠拟合）。
# 2. max_features：默认 'sqrt'。特征很多且相关性强时可试 0.3~0.5；
#    想让单棵树更强（但相关性更高）可试 'log2' 或更小的值比较。
# 3. class_weight：不平衡分类先试 'balanced'，常常比调树结构更有效。
# 4. n_estimators：**最后调**，从 100 加到 300/500，主要提升稳定性；
#    它的边际收益递减，且几乎不会导致过拟合，所以不值得花大预算搜。
# 5. criterion（gini/entropy）差异通常很小，不值得单独搜。
#
# 【搜索策略怎么选】
#   * 参数 ≤ 3 个、每个取值 ≤ 5 个 → 网格搜索够用（本例 6 组，< 1 秒）；
#   * 参数多 / 有连续超参数 / 预算有限 → RandomizedSearchCV(n_iter=30~100, random_state=42)，
#     理论上在相同预算下更容易命中好区域；
#   * 想要更高效率 → HalvingGridSearchCV / HalvingRandomSearchCV（逐次减半，先粗后细），
#     或第三方贝叶斯优化（Optuna 等）。
#   * 多指标同时看时，scoring 传 dict，refit 指定用哪个指标重训。
#
# 【使用注意（踩过的坑）】
#   * **绝对不能用测试集选超参**：那是数据泄漏，分数必然乐观偏高（本脚本步骤 7 已实测）。
#     测试集只能在最后看一次。
#   * 网格搜索前必须确认评估流程本身是正确的：分类要分层、预处理要放进 Pipeline、
#     随机种子要固定，否则你搜到的"最优"只是噪声。
#   * GridSearchCV 里 estimator 的 random_state 一定要固定，否则不同组合之间的分数
#     差异里混着随机波动，排名不可信。
#   * 不要给 GridSearchCV 传已移除的历史参数（例如 iid）；也不要给
#     mean_squared_error 传 squared=False（同样已移除）。
#   * cv_results_ 一定要存下来（本脚本存了 CSV）：超参数搜索的实验记录是可复现性的关键。
#   * n_jobs 设置：GridSearchCV 的 n_jobs 与 estimator 自身的 n_jobs 是**相乘**关系，
#     不要两个都设 -1，否则 CPU 超订、反而变慢。
#   * 网格搜索的结果分数都带 ±std，选择时应优先"分数接近但更简单/更快/更稳"的组合
#     （奥卡姆剃刀：在效果相近时选更简单的模型）。
print("\n【完成】04_网格搜索超参数调优.py 运行结束")

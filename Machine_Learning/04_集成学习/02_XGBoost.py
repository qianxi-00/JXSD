r"""02_XGBoost.py —— XGBoost 原理与实战（含早停、正则化、特征重要性）

对应课案章节
    《机器学习》课案 → 集成学习 → XGBoost
    （课案原文第 803~949 行一带：名词介绍 / XGBoost 的并行 / 核心公式 / 代码）

本节知识点
    1. XGBoost 的目标函数：预测损失 + 模型复杂度惩罚
           Obj = sum_i l(y_i, y_hat_i) + sum_m Omega(f_m)
           Omega(f) = gamma * T + (1/2) * lambda * sum_j w_j^2
    2. 二阶泰勒展开：把损失近似成"关于叶子权重 w 的二次函数"，
       从而直接解出最优叶子权重 w_j* = -G_j / (H_j + lambda)，
       以及分裂增益 Gain 公式（Gain > 0 才值得分裂）。
    3. 工程优化：列采样 colsample_bytree、缺失值默认方向、加权分位数近似分裂点、
       预排序 + 块结构（Block）、以及"并行只发生在找当前这棵树最佳分裂点内部"。
    4. 剪枝：gamma 是最小分裂增益门槛（"分裂的手续费"），post-pruning 从底往上砍。
    5. xgboost 3.x 的 sklearn 接口用法：不要手写 objective + num_class，
       早停参数写在构造函数里，eval_set / verbose 写在 fit() 里。

运行方式
    PowerShell（路径含中文，必须加引号并用 & 调用）：

    $env:PYTHONUTF8='1'
    & 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Machine_Learning\04_集成学习\02_XGBoost.py'

    脚本使用 Agg 无界面后端，图片保存到
    F:\ProGram\Python_Base\Machine_Learning\output\ 目录下。
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")            # 无界面后端，避免 plt.show() 阻塞
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import time
import warnings

import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# 教学脚本输出保持干净：只屏蔽"将来版本才会变"的告警噪音，不屏蔽真正的错误。
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

RANDOM_STATE = 42
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    """打印醒目的中文分节标题。"""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# =============================================================================
# ① 原理与数学推导
# =============================================================================
section("① 原理与数学推导：XGBoost 的目标函数、二阶展开、并行与剪枝")

print(
    r"""
【1】XGBoost 全称 eXtreme Gradient Boosting，可以理解为 GBDT 的"加强版 / 工程优化版"。
     它仍然是一棵树一棵树往后训练、后一棵修正前面模型的错误（Boosting 的串行本性不变），
     但把"每轮加什么树"这件事写得非常精确，并且加了正则化。

【2】核心目标函数（课案"核心公式"一节）

        Obj = sum_{i=1..n} l( y_i , y_hat_i )  +  sum_{m=1..M} Omega( f_m )
              \____ 预测损失 ____/                \____ 模型复杂度惩罚 ____/

     正则化项：  Omega(f) = gamma * T + (1/2) * lambda * sum_{j=1..T} w_j^2

     符号表：
        l(y_i, y_hat_i)  损失函数（预测错了要付出的代价）
        f_m              第 m 棵树（当前这一轮用来补错误的小树）
        Omega(f_m)       第 m 棵树的正则化项（对复杂树的惩罚）
        T                叶子节点数量（树有多少个最终分支）
        w_j              第 j 个叶子的权重（这个叶子给预测结果加多少分）
        gamma, lambda    正则化参数，gamma 惩罚"叶子太多"，lambda 惩罚"叶子权重太大"
        Obj              总目标函数：既要错误小，也不能让模型太复杂

     大白话：XGBoost 不只是追求预测错误小，它还希望模型结构简单，
     不要为了训练集上的一点点提升，把树长得特别复杂。

【3】二阶泰勒展开：把损失变成"关于叶子权重的二次函数"（XGBoost 最核心的一步）

     第 m 轮时，前面的 F_{m-1} 已经固定，记本轮新增的树为 f_m。把损失在 y_hat = F_{m-1}(x_i) 处展开：

        Obj_m ≈ sum_i [ l(y_i, F_{m-1}(x_i)) + g_i * f_m(x_i) + (1/2) * h_i * f_m(x_i)^2 ]
                + gamma * T + (1/2) * lambda * sum_j w_j^2

        其中  g_i = d l(y_i, y_hat) / d y_hat   |_{y_hat = F_{m-1}(x_i)}      （一阶导，梯度）
              h_i = d^2 l(y_i, y_hat) / d y_hat^2 |_{y_hat = F_{m-1}(x_i)}    （二阶导，曲率）
        （回归平方损失时 h_i = 1；二分类对数损失时 g_i = p_i - y_i，h_i = p_i * (1 - p_i)）

     因为在同一棵树上，落在同一个叶子里的样本共享同一个输出值 w_j，
     把样本按叶子分组（I_j 表示第 j 个叶子上的样本集合）后，目标函数变成：

        Obj_m ≈ sum_{j=1..T} [ (sum_{i in I_j} g_i) * w_j + (1/2) * (sum_{i in I_j} h_i + lambda) * w_j^2 ] + gamma * T

     记 G_j = sum_{i in I_j} g_i ，H_j = sum_{i in I_j} h_i 。这是一个关于 w_j 的**开口向上的抛物线**，
     直接令导数为 0 就得到最优叶子权重（不需要梯度下降、不用学习率迭代）：

        最优叶子权重：  w_j* = - G_j / ( H_j + lambda )

     把它代回去，得到"这棵树能带来的最小损失"：

        Obj_m* = -(1/2) * sum_{j=1..T} G_j^2 / (H_j + lambda) + gamma * T

【4】分裂增益公式（决定"要不要在这分裂"）

     假设某个节点分裂成左 L、右 R 两半，那么分裂带来的收益是：

        Gain = (1/2) * [ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G_L+G_R)^2/(H_L+H_R+lambda) ]
               - gamma

     结构上就是：  Gain = 分裂后左右两边的"纯度提升" - gamma（分裂的手续费）
        * 前一项 >= 0，恒为非负：分裂总不会让损失变大；
        * 减去 gamma 之后，Gain 可能 <= 0，那就"不值得分"，这个分裂被否决 —— 这就是 gamma 剪枝。
        * lambda 出现在分母里：lambda 越大，单个叶子的权重被压得越小，分裂收益也越平缓。

【5】XGBoost 的并行到底并行在哪里（课案"XGBoost 的并行"一节）

     先明确：XGBoost 依旧是 Boosting，树与树之间存在依赖（第 m 棵树要拟合第 m-1 轮算出来的
     梯度 g_i / h_i），所以 **树和树之间必定只能串行**。

     并行发生在 **训练每一棵树的内部**。训练一棵决策树时，模型要对每个特征、每个切分点做判断：
        - 哪个特征最适合分裂？  - 哪个切分点最好？  - 分裂之后损失下降多少？
     假设有 100 个特征，可以拆给多个线程并行算：

        线程1：计算特征 1-20      线程2：计算特征 21-40     线程3：计算特征 41-60
        线程4：计算特征 61-80     线程5：计算特征 81-100
        最后汇总：哪个特征、哪个切分点让损失下降最多

     所以一句话总结：**XGBoost 的并行 = 并行寻找"当前这一棵树"的最佳分裂点**。
     另外，XGBoost 把所有数据预先排好序并存成 Block 结构（特征预排序 + 块压缩），
     这样每次找分裂点时可以复用排序结果、支持缓存友好的列式访问，这是它工程上快的重要原因。

【6】XGBoost 相对 GBDT 补上的工程细节

    (a) 列采样 colsample_bytree / colsample_bylevel / colsample_bynode：
        每棵树（或每层 / 每个节点）只用一部分特征找分裂点。既加速，又像随机森林那样降方差。
    (b) 行采样 subsample：每棵树只用一部分样本，进一步抗过拟合。
    (c) 缺失值自动处理：XGBoost 在分裂时会试着把缺失样本全分到左边、再全分到右边，
        哪种分法的 Gain 大就用哪种，并把"默认方向"记录在树里（default direction）。
        所以数据里有 NaN 时**不需要先手动填充**，模型自己会学"空值该走哪边"。
    (d) 分裂点近似算法（approx / hist）：不精确枚举每一个取值，而是用**加权分位数草图**
        （weighted quantile sketch）按 h_i 作为权重挑出候选分裂点，大幅降低计算量。
    (e) 预排序 + Block 结构 + 多线程并行：见上面【5】。
    (f) 正则化与剪枝：Omega(f) 直接进目标函数（这是 GBDT 没有的），
        分裂可以先长出来、再按 gamma 从底向上做后剪枝（post-pruning），
        把"分裂增益 < gamma"的叶子砍掉。叶子数 T 越大，gamma*T 这一项罚得越狠。

【7】树模型不需要标准化（重要但常被忽略）
    决策树分裂只需要比较特征取值的大小顺序（x_j <= threshold ?），
    对特征做 Z-score 标准化或 Min-Max 归一化**不改变任何分裂点**，所以对结果没有影响。
    只有带 L1/L2 正则的线性模型、KNN、SVM、神经网络才需要标准化。
"""
)

# =============================================================================
# ② 库 API 关键参数逐个解释
# =============================================================================
section("② xgboost API 关键参数逐个解释：XGBClassifier（xgboost 3.4.1）")

print(
    r"""
XGBClassifier(n_estimators=100, learning_rate=0.3, max_depth=6, min_child_weight=1,
              gamma=0, subsample=1, colsample_bytree=1, reg_lambda=1, reg_alpha=0,
              eval_metric=None, early_stopping_rounds=None, tree_method=None, ...)

  n_estimators         树的数量 M（boosting 轮数）。有早停时它是一个"上限"，实际用多少看 best_iteration。
  learning_rate        学习率 eta / 收缩系数，默认 0.3，常用 0.05~0.1。
                       【联动】learning_rate 减半 → n_estimators 大约要翻倍。
  max_depth            每棵树最大深度，默认 6。它间接决定叶子数 T（T <= 2^max_depth）。
                       3~8 是常用范围，越大越容易过拟合。
  min_child_weight     叶子节点里样本的 h_i 之和（H_j）的最小值，默认 1。
                       H_j 近似"叶子里的有效样本数"，所以它相当于 sklearn 的 min_samples_leaf。
                       调大 → 不允许长出"只有很少样本"的叶子 → 抗过拟合、抗噪声。
  gamma                分裂所需的最小损失下降，默认 0。对应公式里的 gamma。
                       gamma=0 表示"只要 Gain > 0 就分裂"；设成 1~5 才开始真正剪枝。
                       它是最直接的"控制树复杂度"的参数之一（越大树越浅、叶子越少）。
  subsample            每棵树使用的样本比例，默认 1。设 0.7~0.9 降方差、加速。
  colsample_bytree     每棵树使用的特征比例，默认 1。设 0.7~0.9 是常用的抗过拟合手段。
                       （还有 colsample_bylevel / colsample_bynode，粒度更细。）
  reg_lambda           L2 正则系数，对应公式里的 lambda，默认 1。
                       它出现在 w_j* = -G_j/(H_j+lambda) 的分母上：调大 → 叶子权重变小 →
                       单棵树影响变弱 → 更保守、更不容易过拟合。
  reg_alpha            L1 正则系数，默认 0。它会让部分叶子权重被压到 0（稀疏化），
                       特征很多且希望自动筛特征时可以设一点（如 0.1~1）。
  eval_metric          验证集评估指标。二分类常用 'logloss'（对数损失）、'auc'、'error'。
                       sklearn 接口下不写也行（会按 objective 自动推断），显式写便于看曲线。
  early_stopping_rounds早停耐心值：验证集指标连续这么多轮没有改善就停止，并回滚到最优轮。
                       xgboost 3.x 要求写在**构造函数**里（旧版本是 fit() 的参数，会报错/告警）。
  tree_method          建树算法，默认 'auto'（小数据用精确贪心 exact，大数据自动切 hist）。
                       'hist' 是直方图分桶，速度快、内存省，是最常用的选择；
                       'exact' 是精确贪心枚举所有分裂点，最准但最慢；
                       'gpu_hist'（新版本为 device='cuda'）用 GPU 加速。
  verbosity            日志详细程度，0 = 静默。配合 fit(verbose=False) 一起用，训练过程完全干净。

  【xgboost 3.x + sklearn 接口的重要注意事项】
    1) 不要手写 objective="multi:softmax" + num_class=3：
       sklearn 接口会根据 y 自动推断 objective 和类别数，手动传会与之冲突或产生告警。
       课案里的示例代码是"原生接口"的写法，直接用 sklearn 包装器时应当省略这两个参数。
    2) 早停：early_stopping_rounds 放在构造函数里，eval_set=[(X_val, y_val)] 放在 fit() 里，
       同时 fit(verbose=False) 关闭每一轮的日志。
    3) 训练完的 best_iteration 是"从 0 开始"的轮序号，人话表述要 +1（best_iteration+1 棵树）。
    4) 树模型不需要标准化，直接喂原始特征即可。
"""
)

# =============================================================================
# ③ 完整可运行代码
# =============================================================================
section("③ 完整可运行代码：make_classification 1000 样本 + 验证集早停 + 特征重要性")

# ----------------------------- 3.1 造数据并三划分 -----------------------------
print("\n--- 3.1 造数据并划分成 训练集 / 验证集 / 测试集 ---")

X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=2,
    n_classes=2,
    flip_y=0.02,
    class_sep=1.0,
    random_state=RANDOM_STATE,
)
feature_names = [f"特征{i:02d}" for i in range(X.shape[1])]
print(f"总数据：X={X.shape}，正类比例={y.mean():.3f}")

# 先切出 20% 做测试集（全程不参与训练和早停），再从剩下的切 25% 做验证集。
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_full
)
print(f"训练集 {X_train.shape[0]} 条 / 验证集 {X_val.shape[0]} 条 / 测试集 {X_test.shape[0]} 条")
print("说明：验证集用于早停（决定用多少棵树），测试集只在最后评估一次，绝不参与调参。")

# ----------------------------- 3.2 训练（带早停） -----------------------------
print("\n--- 3.2 训练 XGBClassifier（早停参数写在构造函数，eval_set 写在 fit）---")

model = XGBClassifier(
    n_estimators=1000,          # 上限 1000 棵树，实际用多少由早停决定
    learning_rate=0.05,         # eta：每棵树的修正力度
    max_depth=4,                # 树深，间接决定叶子数 T <= 2^4 = 16
    min_child_weight=1,         # 叶子最小 H_j，相当于 min_samples_leaf
    gamma=0.0,                  # 分裂所需最小损失下降（本组先不分枝剪）
    subsample=0.8,              # 行采样 80%
    colsample_bytree=0.8,       # 列采样 80%
    reg_lambda=1.0,             # L2 正则 lambda，压制叶子权重
    reg_alpha=0.0,              # L1 正则，这里不启用
    eval_metric="logloss",      # 用对数损失做早停判据
    early_stopping_rounds=30,   # 验证集指标连续 30 轮不改善就停
    tree_method="hist",         # 直方图分桶建树，快且省内存
    random_state=RANDOM_STATE,
    n_jobs=2,
    verbosity=0,                # 静默，不打印 C 层日志
)

t0 = time.perf_counter()
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],  # 早停用的验证集
    verbose=False,              # 关闭"每一轮都打印一行"的日志
)
train_seconds = time.perf_counter() - t0
print(f"训练耗时：{train_seconds:.3f} 秒")

# ----------------------------- 3.3 早停结果 -----------------------------
print("\n--- 3.3 早停结果解读 ---")

best_round = int(model.best_iteration) + 1     # xgboost 的 best_iteration 从 0 开始
best_score = float(model.best_score)
print(f"best_iteration（0 起）= {model.best_iteration}，也就是最优时用了 {best_round} 棵树（上限 1000）")
print(f"best_score（验证集 logloss 最优值）= {best_score:.4f}")

y_pred = np.asarray(model.predict(X_test)).reshape(-1)
y_proba = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
test_logloss = log_loss(y_test, model.predict_proba(X_test))
print(f"测试集准确率：{acc:.4f}")
print(f"测试集 AUC：{auc:.4f}")
print(f"测试集 log loss：{test_logloss:.4f}")

# ----------------------------- 3.4 早停曲线 -----------------------------
print("\n--- 3.4 从 evals_result() 取早停曲线 ---")

evals = model.evals_result()
val_curve = np.asarray(evals["validation_0"]["logloss"], dtype=float)
rounds = np.arange(1, len(val_curve) + 1)
print(f"验证集损失曲线共 {len(val_curve)} 个点（= 实际训练轮数 + 早停耐心值）")
print(f"  第 1 轮：{val_curve[0]:.4f}")
print(f"  最低点：第 {int(np.argmin(val_curve)) + 1} 轮，值 {val_curve.min():.4f}")
print(f"  最后一轮：{val_curve[-1]:.4f}（回升是耐心轮数内继续试探的结果，模型会回滚到最低点）")

# ----------------------------- 3.5 gamma 剪枝实验 -----------------------------
print("\n--- 3.5 gamma 剪枝实验：gamma 越大，允许的分裂越少，叶子总数越少 ---")

gamma_rows = []
for gamma_value in (0.0, 1.0, 5.0, 20.0):
    m_g = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        gamma=gamma_value,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        eval_metric="logloss",
        early_stopping_rounds=30,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=2,
        verbosity=0,
    )
    m_g.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    # 统计建出来的树一共用了多少叶子：trees_to_dataframe() 里每行是一个节点，
    # 其中 "Leaf" 行就是叶子节点，数一下就知道 gamma 剪枝到底砍掉了多少叶子。
    df_trees = m_g.get_booster().trees_to_dataframe()
    n_leaves_total = int((df_trees["Feature"] == "Leaf").sum())
    gamma_rows.append(
        (gamma_value,
         int(m_g.best_iteration) + 1,
         n_leaves_total,
         accuracy_score(y_test, m_g.predict(X_test)),
         roc_auc_score(y_test, m_g.predict_proba(X_test)[:, 1]))
    )

print(f"{'gamma':>8} | {'最优树数':>10} | {'叶子总数':>10} | {'测试准确率':>12} | {'测试AUC':>10}")
print("-" * 66)
for g_val, n_tree, n_leaf, a, u in gamma_rows:
    print(f"{g_val:>8.1f} | {n_tree:>10d} | {n_leaf:>10d} | {a:>12.4f} | {u:>10.4f}")

# ----------------------------- 3.6 特征重要性 -----------------------------
print("\n--- 3.6 特征重要性（gain 口径：该特征参与的分裂带来的平均增益）---")

importances = np.asarray(model.feature_importances_, dtype=float)
order = np.argsort(importances)[::-1]
print("特征重要性 Top 8（importance_type 默认 'gain'）：")
for rank, idx in enumerate(order[:8], start=1):
    print(f"  第{rank:>2}名  {feature_names[idx]:<8} 重要性 = {importances[idx]:.4f}")

# ----------------------------- 3.7 换一份真实数据复现 -----------------------------
print("\n--- 3.7 换用 sklearn 内置 breast_cancer 数据集复现（569 样本 / 30 特征）---")

Xb, yb = load_breast_cancer(return_X_y=True)
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(
    Xb, yb, test_size=0.25, random_state=RANDOM_STATE, stratify=yb
)
model_bc = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric="logloss",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=2,
    verbosity=0,
)
model_bc.fit(Xb_tr, yb_tr, verbose=False)
acc_bc = accuracy_score(yb_te, model_bc.predict(Xb_te))
auc_bc = roc_auc_score(yb_te, model_bc.predict_proba(Xb_te)[:, 1])
print(f"breast_cancer 测试集准确率：{acc_bc:.4f}，AUC：{auc_bc:.4f}")
print("（这里没有标准化、没有缺失值填充，直接把原始特征喂给 XGBoost，效果依然很好。）")

# =============================================================================
# 绘图
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

axes[0].plot(rounds, val_curve, color="#d62728", lw=1.8, label="验证集 log loss")
axes[0].axvline(best_round, color="#1f77b4", ls="--", lw=1.5,
                label=f"早停选出的最优轮 = 第 {best_round} 轮")
axes[0].scatter([best_round], [val_curve[best_round - 1]], color="#1f77b4", zorder=5, s=45)
axes[0].set_xlabel("Boosting 轮数（树的数量）")
axes[0].set_ylabel("验证集 log loss（越小越好）")
axes[0].set_title("早停：验证损失降到最低后开始回升")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

top_n = 12
top_idx = order[:top_n][::-1]
axes[1].barh([feature_names[i] for i in top_idx], importances[top_idx], color="#2ca02c")
axes[1].set_xlabel("特征重要性（gain）")
axes[1].set_title("特征重要性 Top 12")
axes[1].grid(axis="x", alpha=0.3)

fig.suptitle(f"02 XGBoost（learning_rate=0.05 / max_depth=4 / 早停 30 轮）"
             f"  最优 {best_round} 棵树 / 测试准确率 {acc:.4f} / AUC {auc:.4f}", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUTPUT_DIR / "04_XGBoost_特征重要性与早停.png", dpi=130)
plt.close(fig)

# =============================================================================
# ④ 结果解读
# =============================================================================
section("④ 结果解读：这些数字到底在说什么")

print(
    f"""
1) 早停选出的最优轮数是 {best_round} 棵（上限设了 1000）
   —— 说明这份数据的"可学习信号"大概就值这么多棵树。
   继续加树，验证损失不再下降（最后一轮 {val_curve[-1]:.4f} 高于最低点 {val_curve.min():.4f}），
   模型会回滚到第 {best_round} 轮的状态，多出来的树等于白算但不会伤害结果。

2) best_score = {best_score:.4f} 是**验证集**上 log loss 的最优值，
   不是准确率、也不是测试集指标。早停的判据（eval_metric="logloss"）决定它是什么。
   我们的测试集 log loss 是 {test_logloss:.4f}，和验证集很接近 → 没有明显过拟合。

3) 测试集准确率 {acc:.4f}、AUC {auc:.4f}
   —— 准确率看"分对了多少"，AUC 看"排序能力"（正样本得分高于负样本的概率）。
   AUC 0.5 = 瞎猜，1.0 = 完美。{auc:.4f} 说明模型把两类分得很开，
   即使调分类阈值（0.5 之外）也不会差太多 —— 这正是 AUC 比准确率更稳健的地方。

4) gamma 剪枝实验的规律（见上面那张表）：
   叶子总数随 gamma 单调下降：{gamma_rows[0][2]} → {gamma_rows[1][2]} → {gamma_rows[2][2]} → {gamma_rows[3][2]}
   —— gamma 是"分裂的手续费"，手续费的闸门抬高后，只有 Gain > gamma 的分裂才被允许，
   树被剪得更浅更瘦（gamma={gamma_rows[3][0]:.0f} 时平均每棵树只剩
   {gamma_rows[3][2] / gamma_rows[3][1]:.1f} 个叶子）。
   注意"最优树数"并不是单调的（{gamma_rows[0][1]} → {gamma_rows[1][1]} → {gamma_rows[2][1]} → {gamma_rows[3][1]}）：
   单棵树被剪弱之后，模型反而需要更多轮才能拟合到同样的程度，这是两个相反方向作用的叠加结果。
   当 gamma 大到 {gamma_rows[3][0]:.0f} 时，测试 AUC 掉到 {gamma_rows[3][4]:.4f}（其余三组都在 0.96 以上），
   说明手续费收得过头了，模型开始欠拟合 —— gamma 属于必须试出来的参数。

5) 特征重要性（gain 口径）前三名：{feature_names[order[0]]} ({importances[order[0]]:.4f})、
   {feature_names[order[1]]} ({importances[order[1]]:.4f})、{feature_names[order[2]]} ({importances[order[2]]:.4f})
   —— gain 表示"这个特征参与的所有分裂，平均带来了多少损失下降"。
   数值是相对的，只看排名、不看绝对值；它反映的是"模型怎么用特征"，不等于因果重要性。

6) breast_cancer 上准确率 {acc_bc:.4f}、AUC {auc_bc:.4f}
   —— 30 个真实医学特征，300 棵 max_depth=3 的小树就能到很高水平，
   而且全程没有做标准化。这印证了"树模型不看特征尺度"。
"""
)

# =============================================================================
# 超参数怎么调
# =============================================================================
section("超参数怎么调（XGBoost）")

print(
    """
【第一步：定"学多久"】
    learning_rate 先定 0.05~0.1；n_estimators 设一个大值（如 1000~2000）并开早停
    （early_stopping_rounds=20~50 + eval_set）。让早停告诉你到底需要多少棵树。
    记住联动：learning_rate 减半 → 树的数量大约翻倍。

【第二步：定"单棵树多复杂"（影响最大）】
    max_depth            默认 6。一般从 3~4 起调，数据规律复杂再加到 6~8。
    min_child_weight     默认 1。噪声大 / 样本少时调到 5~50，防止长出只装几个样本的叶子。
    gamma                默认 0。要剪枝就试 0.1 / 1 / 5；
                         发现"树特别多、叶子特别碎"时优先加它。

【第三步：加随机性压过拟合】
    subsample            默认 1 → 试 0.7~0.9。
    colsample_bytree     默认 1 → 试 0.7~0.9（高维数据可更小，如 0.5）。
    这两个是性价比最高的抗过拟合参数，通常先动它们。

【第四步：正则化微调】
    reg_lambda（L2）默认 1 → 试 1 / 5 / 20；它直接压小叶子权重 w_j。
    reg_alpha（L1）默认 0 → 特征非常多、想自动稀疏化时试 0.1~1。

【工程参数】
    tree_method          'hist'（默认推荐，快且省内存）/ 'exact'（最准最慢）。
    n_jobs               线程数，-1 表示用满；只影响"找当前这棵树最佳分裂点"的并行速度，
                         不会改变树与树之间的串行依赖。

【调参顺序建议】
    learning_rate + 早停  →  max_depth / min_child_weight  →  subsample / colsample_bytree
    →  gamma / reg_lambda  →  最后再考虑把 learning_rate 调小、树数翻倍。
    每动一个参数都看验证集指标，不要一次改一堆。
"""
)

print(f"\n生成的图片：\n  {OUTPUT_DIR / '04_XGBoost_特征重要性与早停.png'}")
print("\n【完成】02_XGBoost.py 运行结束")

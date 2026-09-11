r"""《机器学习》课案 —— 06 模型评估与保存 / 03 交叉验证

对应课案章节
------------
《机器学习》课案 "模型评估与保存" 章 → "交叉验证" 节
（课案原文第 1404~1439 行；本章 5 个代码块中的第 3 个：
 load_iris + LogisticRegression(max_iter=200) + cross_val_score(cv=5, scoring="accuracy")）。

本节知识点
----------
1. 为什么需要交叉验证：单次 train_test_split 的评估结果带有很大偶然性，
   数据量越小越明显；CV 用"每个样本都当过一次验证样本"的方式给出更稳的估计；
2. K 折交叉验证的完整流程与每一步的数据划分细节；
3. 分层 K 折 StratifiedKFold：分类任务默认用它，保证每折类别比例与整体一致
   （本脚本用真实数据对比 KFold 与 StratifiedKFold 的每折类别比例）；
4. 留一法 LeaveOneOut（LOOCV）的原理、优点与巨大的计算代价；
5. 时间序列为什么必须用 TimeSeriesSplit 而不能随机 K 折（用未来数据训练、用过去数据测试
   = 典型的数据泄漏，线下分数会虚高到离谱）；
6. cross_validate 多指标评估与 return_train_score：用"训练分数远高于验证分数"诊断过拟合；
7. 手写 StratifiedKFold 循环复现 cross_val_score，破除"sklearn 是黑箱"的错觉。

运行方式（在 PowerShell 中复制执行）
------------------------------------
$env:PYTHONUTF8='1'; & 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Machine_Learning\06_模型评估与保存\03_交叉验证.py'

产出
----
Machine_Learning/output/06_交叉验证分数.png
Machine_Learning/output/06_训练与验证对比.png
Machine_Learning/output/06_分层K折对比.png
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")            # 无界面后端，避免 plt.show() 阻塞
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    KFold,
    LeaveOneOut,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_val_score,
    cross_validate,
    train_test_split,
)

# =============================================================================
# ① 原理与数学推导
# =============================================================================
# 1.1 为什么单次划分不可靠
# ------------------------
# 把数据切成一次 train/test，测试集准确率只是一个"点估计"。换一个 random_state、
# 换一种切法，分数可能相差好几个百分点。极端例子：某个难样本刚好落在测试集里，
# 分数就低；落在训练集里，分数就高。数据量越小，这种波动越大。
#
# 严格来说，我们真正关心的是**泛化误差的期望**
#       Err = E_{(x,y)~D}[ L(y, f̂(x)) ]
# 它是对整个数据分布 D 的期望；而单次留出法只是这个期望的一个**高方差、有偏**的估计：
#   * 训练集只用了 (1-1/k) 的数据（k=5 时只用了 80%），模型偏弱 → 偏悲观；
#     留出法只用一次划分，方差大。
#
# 1.2 K 折交叉验证（K-Fold Cross Validation）
# -------------------------------------------
# 流程（K=5 为例）：
#   (1) 把数据集随机均分成 K 个互不重叠、大小相近的子集（折，fold）；
#   (2) 对 i = 1..K：用第 i 折当验证集，**其余 K-1 折合起来**当训练集，训练模型，
#       在第 i 折上算指标 s_i；
#   (3) 汇总：CV 分数 = (1/K)·Σ s_i（均值），并报告标准差 std(s_i)。
# 关键性质：
#   * **每个样本恰好当过一次验证样本**，也恰好被训练 K-1 次 → 数据利用率远高于留出法；
#   * K 个模型是**相互独立的**，互相之间不共享参数（训练完即弃，最后要在全量数据上重训）；
#   * 均值是泛化误差的低方差估计，**标准差是模型稳定性的直接度量**：
#     std 大 = 模型对数据划分敏感（数据少、模型不稳、或存在难样本）。
# K 怎么选：
#   K=5 或 10 是经验默认值（K=10 偏差更小但计算量翻倍）；
#   K=n（留一法）几乎无偏，但方差大且极慢；
#   类别极不平衡时，即使 K 折分层，也要保证每折少数类样本数 ≥ 2（否则指标没意义）。
#
# 1.3 分层 K 折（StratifiedKFold）
# --------------------------------
# 普通 KFold 是**纯随机**分折，不保证每折的类别比例与整体一致。若正类只占 10%，
# 某一折可能只有 2% 正类，甚至一个都没有 —— 那一折的指标就完全不可信，
# 而且会导致 CV 分数的方差被人为放大。
# StratifiedKFold 的做法：**在每个类别内部单独切分**，再拼成折，使每折的类别比例
# 与整体基本一致。所以：
#   * 分类任务（尤其是类别不平衡时）默认用 StratifiedKFold；
#   * sklearn 的 cross_val_score / GridSearchCV 在检测到"分类器 + cv=整数"时
#     **自动使用 StratifiedKFold**（回归任务自动用 KFold，因为连续目标无法"分层"）。
#
# 1.4 留一法（Leave-One-Out, LOOCV）
# ----------------------------------
# K = n 的极端情形：每次只留 1 个样本做验证，用其余 n-1 个训练，重复 n 次。
#   * 优点：训练集几乎等于全量数据，偏差极小；且对任意 n 都能做，没有随机性。
#   * 缺点：(a) 需要训练 n 个模型，代价是 K 折的 n/K 倍；
#           (b) 验证集只有 1 个样本，单次指标只能是 0 或 1，**方差极大**
#               （有定理表明 LOOCV 的方差往往比 10 折更大）；
#           (c) 分层困难（每折只有 1 个样本）。
#   → 实践结论：**除非 n 极小（几十个），否则不用 LOOCV。**
#
# 1.5 时间序列必须用 TimeSeriesSplit（数据泄漏警告）
# --------------------------------------------------
# 时序数据里样本是有顺序、且前后相关的（今天的股价包含昨天的信息）。
# 若用随机 K 折，训练集里会混入"未来"的样本，再去预测"过去"的样本，
# 模型相当于**偷看了答案**：线下分数会高得离谱，上线后立刻崩掉。
# 这就是**数据泄漏（data leakage）**最经典的形态。
# TimeSeriesSplit 的规则：**永远用前面的数据训练、紧邻其后的数据验证**：
#       fold 1: train = [0 .. t1)        test = [t1 .. t2)
#       fold 2: train = [0 .. t2)        test = [t2 .. t3)   ...
#   * 训练集随着折数**单调扩张**（expanding window）；
#   * 折与折之间的测试集不重叠，且永远在训练集之后；
#   * 常用参数 gap：在训练集与测试集之间留一段"间隔"，防止前后样本的短期相关
#     造成的泄漏（例如用 T 日数据预测 T+1，label 里含 T+1 信息时尤其重要）。
#
# 1.6 交叉验证与"选模型"的关系
# ----------------------------
# CV 的用途有三个层次：
#   (1) 评估：估计泛化性能（本脚本的主体）；
#   (2) 选模型 / 选超参：GridSearchCV 内部就是"对每组超参做一遍 CV"（见 04 号脚本）；
#   (3) 早停 / 特征选择：也必须在训练折内部做，否则泄漏。
# 最容易犯的错误：**用测试集反复挑选模型，再报告测试集分数** —— 测试集信息被"用掉"了，
# 报出来的分数是乐观偏差的。正确做法：训练集内部再切出验证集或用 CV 选模型，
# 测试集只在最后看一次。

# =============================================================================
# ② sklearn API 关键参数逐个解释
# =============================================================================
# 【cross_val_score(estimator, X, y, *, cv, scoring, n_jobs, verbose, error_score)】
#   estimator   : **未训练的**估计器。函数会为每一折 clone() 出一个新副本，
#                 所以你传进去的对象在调用后**仍然没有 fit 过**（这点常被误解）。
#   cv          : 折数（int）或切分器对象（KFold/StratifiedKFold/TimeSeriesSplit/...）。
#                 传 int 时分类任务自动分层；传对象则完全按你的规则切。
#                 常用值：5（默认）、10；数据很少时用 3~5。
#   scoring     : 指标名字符串或可调用对象。分类常用 "accuracy"、"f1"、"roc_auc"、
#                 "precision_macro"、"neg_log_loss"；回归用 "r2"、"neg_mean_squared_error"
#                 （**sklearn 的 loss 类指标统一取负号**，因为框架内部约定"分数越大越好"）。
#                 传 None 时用 estimator 自带的 .score()。
#   n_jobs      : 并行进程数，-1 = 用满所有核心。小数据设 1 反而更快（省去进程开销）。
#   verbose     : >0 时打印进度（多进程下很有用）。
#   error_score : 某折训练失败时的取值。默认 nan 只发警告；设成 "raise" 便于调试
#                 （生产脚本里显式设成 "raise" 可以避免"某折悄悄失败、你自己不知道"）。
#   返回        : 长度为 cv 的数组，只含测试折分数。
#
# 【cross_validate(estimator, X, y, *, cv, scoring, return_train_score, return_estimator, ...)】
#   与 cross_val_score 的区别：一次跑出多个指标 + 训练分数 + 时间 + 训练好的模型。
#   scoring=列表             : 一次算多个指标，如 ["accuracy", "precision_macro", ...]；
#                              键名规则：单指标与名字同名，"precision_macro" 保持原样，
#                              callable 会被命名为 "test_score"。
#   return_train_score=True  : **额外返回训练折分数**，是诊断过拟合的关键开关。
#                              注意它会增加计算量（要额外 predict），且训练分数本身
#                              "天然偏高"，只能与验证分数**对比着看**，不能单独解读。
#   return_estimator=True    : 把每折训练好的模型也返回（可用于后续分析，代价是占内存）。
#   return_indices=True      : 返回每折用到的样本下标（调试切分逻辑时非常有用）。
#   返回                     : dict，键形如 fit_time / score_time /
#                              test_<指标名> / train_<指标名>。
#
# 【StratifiedKFold(n_splits=5, shuffle=False, random_state=None)】
#   n_splits      : 折数 K，必须 ≥ 2。
#   shuffle       : 是否在分折前打乱。**默认 False**，即按原始顺序分层切分，
#                   结果完全确定（可复现）；设 True 时必须同时指定 random_state，
#                   否则每次结果都不同。
#   random_state  : 仅 shuffle=True 时生效。
#   对比 KFold：KFold 没有 stratify 概念，参数相同但切分时不看 y。
#   注意：新版本 sklearn 已移除 KFold 的 shuffle=False 之外的历史行为差异；
#         也**不要**再使用已废弃的弃用参数（例如 GridSearchCV 的 iid）。
#
# 【LeaveOneOut()】无参数，n_splits 自动等于样本数 n。
#   注意：cross_val_score 返回的数组长度为 n，且每个值只能是 0/1（或 ±1），
#         不要误以为"分数很整齐"就是好事。
#
# 【TimeSeriesSplit(n_splits=5, *, test_size=None, max_train_size=None, gap=0)】
#   n_splits     : 折数（注意第 1 折的训练集通常很短，n_splits 不能超过样本能支撑的折数）。
#   test_size    : 每折测试集样本数，默认自动均分。
#   max_train_size: 限制训练集最大长度 → 变成**滑动窗口**（sliding window）而不是扩张窗口。
#   gap          : 训练集与测试集之间跳过的样本数，用来切断短期相关性造成的泄漏。
#
# 【LogisticRegression(max_iter=200, random_state=42, ...)】
#   max_iter=200 : 最大迭代次数。**默认 100 在未标准化的小数据上经常不收敛并刷
#                  ConvergenceWarning**；课案写 200 就是为了避开它。规范做法是
#                  放进 Pipeline 先 StandardScaler，再适度提高 max_iter。
#   solver       : 'lbfgs'(默认，支持多分类 softmax) / 'liblinear'(仅二分类，适合小数据) /
#                  'saga'(支持 L1 与 elasticnet，大数据很快)。
#   C            : 正则强度的**倒数**（C = 1/λ），C 越小正则越强、模型越简单；常用 0.1~10。
#   penalty      : 'l2'(默认) / 'l1'(特征选择效果) / 'elasticnet' / None。
#   class_weight : 'balanced' 自动按类别频率反比加权，不平衡分类的常用起手式。
#   random_state : 固定随机种子（仅对 'sag'/'saga'/'liblinear' 等有随机性的求解器生效）。
#
# 【train_test_split】见 01 号脚本注释；本脚本额外用它来对比"单次划分 vs 交叉验证"。

# =============================================================================
# ③ 完整可运行代码
# =============================================================================
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("=" * 78)
print("06-03 交叉验证：K 折 / 分层 K 折 / 留一法 / 时间序列切分 + 手写复现")
print("=" * 78)

# --- 步骤 1：加载数据（对应课案代码块 3） ------------------------------------
iris = load_iris()
X = iris.data
y = iris.target
print("\n[1] 数据概况（sklearn 内置 iris，无需联网下载）")
print(f"    X = {X.shape}（150 朵鸢尾花 × 4 个特征：花萼长/宽、花瓣长/宽）")
print(f"    y = {y.shape}，类别分布：")
for cls_idx, cls_name in enumerate(iris.target_names):
    cnt = int((y == cls_idx).sum())
    print(f"        类别 {cls_idx} = {cls_name:<12s} {cnt} 个（{cnt / len(y):.1%}）")
print(f"    → 三个类别完全均衡（各 50 个），这是一个「分层与否差别不大」的良性数据集；")
print(f"      正因为如此，稍后我们会用一个人造的不平衡数据来演示分层的必要性。")

# --- 步骤 2：先看看"单次划分"有多不稳定 --------------------------------------
print("\n[2] 先做一个对照实验：单次 train_test_split 的分数有多飘？")
single_scores = []
for seed in range(5):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    m = LogisticRegression(max_iter=200, random_state=RANDOM_STATE).fit(X_tr, y_tr)
    single_scores.append(accuracy_score(y_te, m.predict(X_te)))
print("    用 5 个不同的 random_state 各做一次单次划分，测试集准确率分别为：")
print("    " + "  ".join(f"seed={s}: {v:.4f}" for s, v in enumerate(single_scores)))
print(f"    最低 {min(single_scores):.4f}，最高 {max(single_scores):.4f}，"
      f"极差 {max(single_scores) - min(single_scores):.4f}")
print("    → 同一份数据、同一个模型，只因为切法不同，分数就差了好几个百分点。")
print("      这说明单次划分的结论**不可靠**，尤其当数据量小的时候。")
print("      交叉验证就是为了把这个「运气成分」平均掉。")

# --- 步骤 3：5 折交叉验证（对应课案代码块 3 的核心） ------------------------
model = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)
scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
print("\n[3] 5 折交叉验证 cross_val_score(cv=5, scoring='accuracy')")
print("    每一折的准确率：[" + ", ".join(f"{s:.4f}" for s in scores) + "]")
print(f"    平均准确率 mean  = {scores.mean():.4f}")
print(f"    准确率标准差 std = {scores.std():.4f}")
print(f"    → 解读：模型在 iris 上的泛化准确率约为 {scores.mean():.2%}，")
print(f"      各折之间波动 ±{scores.std():.4f}（标准差），说明表现**相当稳定**。")
print(f"      标准差是「稳定性」的度量：如果某次看到 std 达到 0.1 以上，")
print(f"      说明模型对数据划分非常敏感，这时平均值本身也不可信。")
print(f"    注意：调用后 model 本身仍然没有 fit（cross_val_score 内部用的是副本），"
      f"hasattr(model, 'coef_') = {hasattr(model, 'coef_')}")

# --- 步骤 4：cross_validate 多指标 + 训练分数 -------------------------------
scoring = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
}
cv_results = cross_validate(
    LogisticRegression(max_iter=200, random_state=RANDOM_STATE),
    X,
    y,
    cv=5,
    scoring=scoring,
    return_train_score=True,   # 关键：同时得到训练折分数，用于诊断过拟合
)
print("\n[4] cross_validate 多指标 + return_train_score=True")
print("    原始返回的键：", sorted(cv_results.keys()))
metric_names = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
rows = []
for name in metric_names:
    test_vals = cv_results[f"test_{name}"]
    train_vals = cv_results[f"train_{name}"]
    rows.append(
        {
            "指标": name,
            "训练分数均值": train_vals.mean(),
            "验证分数均值": test_vals.mean(),
            "验证分数标准差": test_vals.std(),
            "差值(训练-验证)": train_vals.mean() - test_vals.mean(),
        }
    )
cv_table = pd.DataFrame(rows).round(4)
print(cv_table.to_string(index=False))
print(f"\n    平均每折训练耗时 {cv_results['fit_time'].mean():.4f} 秒，"
      f"平均每折打分耗时 {cv_results['score_time'].mean():.4f} 秒，"
      f"总计约 {cv_results['fit_time'].sum() + cv_results['score_time'].sum():.4f} 秒")
print("\n    如何判断过拟合：**训练分数远高于验证分数 = 过拟合**（模型背下了训练数据，")
print("    但换一批数据就不行）；两者都很低 = 欠拟合（模型太简单）；两者都高且接近 = 良好。")
acc_train = cv_results["train_accuracy"].mean()
acc_test = cv_results["test_accuracy"].mean()
print(f"    本次：训练准确率 {acc_train:.4f}，验证准确率 {acc_test:.4f}，"
      f"差值仅 {acc_train - acc_test:.4f}。")
if acc_train - acc_test < 0.03:
    print("    → 两者几乎相等，且都接近 1，说明**既没有过拟合也没有欠拟合**，模型状态健康。")
    print("      这也符合预期：iris 只有 150 个样本、4 个特征、3 个线性可分的类别，")
    print("      逻辑回归这种简单模型正好合适，没有多少过拟合空间。")
else:
    print("    → 训练分数明显高于验证分数，存在过拟合倾向，可加强正则（调小 C）。")

# --- 步骤 5：手写 StratifiedKFold 循环，复现 cross_val_score -----------------
print("\n[5] 手写 StratifiedKFold 循环，验证与 cross_val_score 结果完全一致")
skf = StratifiedKFold(n_splits=5, shuffle=False)   # 与 cross_val_score(cv=5) 的内部策略一致
manual_scores = []
print("    折次 | 训练集样本数 | 验证集样本数 | 验证集类别分布        | 本折准确率")
for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X, y), start=1):
    X_tr, X_va = X[train_idx], X[valid_idx]
    y_tr, y_va = y[train_idx], y[valid_idx]
    fold_model = clone(model)          # 关键：每折都用**全新的未训练模型**
    fold_model.fit(X_tr, y_tr)
    fold_acc = accuracy_score(y_va, fold_model.predict(X_va))
    manual_scores.append(fold_acc)
    dist = "/".join(str(int((y_va == c).sum())) for c in np.unique(y))
    print(f"    {fold_idx:>4d} | {len(train_idx):>12d} | {len(valid_idx):>12d} | "
          f"{dist:>20s} | {fold_acc:.4f}")
manual_scores = np.array(manual_scores)
print(f"    手写循环：每折 = [" + ", ".join(f"{s:.4f}" for s in manual_scores) + "]")
print(f"    手写循环：均值 = {manual_scores.mean():.4f}，标准差 = {manual_scores.std():.4f}")
print(f"    sklearn  ：每折 = [" + ", ".join(f"{s:.4f}" for s in scores) + "]")
print(f"    sklearn  ：均值 = {scores.mean():.4f}，标准差 = {scores.std():.4f}")
print(f"    两者逐折是否完全相等（np.allclose）？ "
      f"{np.allclose(manual_scores, scores, atol=1e-12)}")
print("    → 结论：cross_val_score 没有任何魔法，它做的就是上面这个循环 + 自动分层。")
print(f"      验证集类别分布恒为 {int((y == 0).sum() / 5)}/{int((y == 1).sum() / 5)}/"
      f"{int((y == 2).sum() / 5)}（150/5=30，三类各 10 个），")
print("      这正是 StratifiedKFold 保证的「每折类别比例与整体一致」。")

# --- 步骤 6：KFold vs StratifiedKFold 的真实差异（用不平衡数据） -------------
X_imb, y_imb = make_classification(
    n_samples=300, n_features=10, weights=[0.9, 0.1], random_state=RANDOM_STATE
)
pos_total = int((y_imb == 1).sum())
print("\n[6] 分层到底有什么用？用一份不平衡数据实测（300 个样本，"
      f"正类 {pos_total} 个 = {pos_total / len(y_imb):.1%}）")
print("    普通 KFold 的每折正类个数：")
kf_pos = []
for tr, va in KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(X_imb, y_imb):
    kf_pos.append(int((y_imb[va] == 1).sum()))
print("        " + ", ".join(str(v) for v in kf_pos)
      + f"   → 折间差异很大（最少 {min(kf_pos)}，最多 {max(kf_pos)}）")
skf_pos = []
for tr, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE).split(X_imb, y_imb):
    skf_pos.append(int((y_imb[va] == 1).sum()))
print("    StratifiedKFold 的每折正类个数：")
print("        " + ", ".join(str(v) for v in skf_pos)
      + f"   → 每折都≈{pos_total / 5:.0f} 个，比例稳定，各折成绩可比")
kf_scores = cross_val_score(
    LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE),
    X_imb, y_imb, cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE), scoring="f1",
)
skf_scores = cross_val_score(
    LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE),
    X_imb, y_imb, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
    scoring="f1",
)
print(f"    普通 KFold 的 F1：[" + ", ".join(f"{s:.4f}" for s in kf_scores)
      + f"]  均值 {kf_scores.mean():.4f}，标准差 {kf_scores.std():.4f}")
print(f"    分层 K 折的 F1：[" + ", ".join(f"{s:.4f}" for s in skf_scores)
      + f"]  均值 {skf_scores.mean():.4f}，标准差 {skf_scores.std():.4f}")
print(f"    → 分层后折间标准差从 {kf_scores.std():.4f} 变为 {skf_scores.std():.4f}，"
      f"分数更稳定、更可信。")
print("      原因：分层的每一折都有稳定比例的少数类样本，避免某折「几乎没有正类」")
print("      导致指标剧烈抖动。")

# --- 步骤 7：留一法 LeaveOneOut ---------------------------------------------
loo = LeaveOneOut()
loo_scores = cross_val_score(
    LogisticRegression(max_iter=200, random_state=RANDOM_STATE), X, y, cv=loo, scoring="accuracy"
)
print("\n[7] 留一法 LeaveOneOut（K = n = 150，要训练 150 个模型）")
print(f"    返回的分数个数 = {len(loo_scores)}（等于样本数 n），"
      f"取值范围 = {sorted(set(loo_scores.tolist()))}")
print(f"    LOOCV 平均准确率 = {loo_scores.mean():.4f}")
print(f"    与 5 折的 {scores.mean():.4f} 相比差别很小，但计算量是 5 折的 "
      f"{len(loo_scores) / 5:.0f} 倍。")
print("    → 每个验证集只有 1 个样本，单折分数非 0 即 1，方差极大；")
print("      所以除非样本只有几十个，否则**不要用留一法**，5 折/10 折才是性价比之选。")

# --- 步骤 8：时间序列切分 TimeSeriesSplit（防数据泄漏） ----------------------
print("\n[8] 时间序列为什么不能用随机 K 折？—— TimeSeriesSplit 的折结构")
n_ts = 30
tscv = TimeSeriesSplit(n_splits=5)
print("    用 n=30 的示意数据展示每折的训练/测试下标区间：")
print("    折次 | 训练集下标范围        | 测试集下标范围")
for fold_idx, (tr_idx, te_idx) in enumerate(tscv.split(np.arange(n_ts)), start=1):
    print(f"    {fold_idx:>4d} | [{tr_idx[0]:>2d} .. {tr_idx[-1]:>2d}]"
          f"（{len(tr_idx):>2d} 个）      | [{te_idx[0]:>2d} .. {te_idx[-1]:>2d}]"
          f"（{len(te_idx)} 个）")
print("    → 训练集**永远在测试集之前**，且随折数单调扩张；测试集之间互不重叠。")
print("      若换成随机 KFold，训练集里会出现下标比测试集更大的样本（即「未来数据」），")
print("      模型相当于提前看到了答案 —— 这就是**数据泄漏**，线下分数虚高、上线必崩。")
print("      结论：时序任务一律 TimeSeriesSplit，并视情况设置 gap 切断短期相关性。")

# =============================================================================
# ④ 结果解读
# =============================================================================
print("\n" + "=" * 78)
print("④ 结果解读：这些数字到底说明什么")
print("=" * 78)
print(f"""
【1】iris 上的 5 折准确率 = {scores.mean():.4f} ± {scores.std():.4f}。
      怎么读这个"均值 ± 标准差"？
      * 均值回答"模型整体多好"：约 {scores.mean():.1%} 的样本能被正确分类；
      * 标准差回答"这个结论有多稳"：{scores.std():.4f} 意味着换一种划分方式，
        分数大致仍在 {scores.mean() - scores.std():.3f} ~ {scores.mean() + scores.std():.3f} 之间波动。
      标准差越小，越可以把均值当成"可信的泛化性能"。iris 数据简单、样本均衡，
      逻辑回归又是线性模型，所以这里既高分又低波动，属于教科书式的"健康"结果。

【2】与单次划分对比：单次划分的 5 次结果在
      {min(single_scores):.4f} ~ {max(single_scores):.4f} 之间跳（极差
      {max(single_scores) - min(single_scores):.4f}），
      而 5 折 CV 给出的是一个带稳定性标注的估计。
      → **数据量越小，交叉验证越重要**；样本成千上万时单次划分也够用，
        但 CV 仍然是"顺带调参"的前提。

【3】训练分数 {acc_train:.4f} 与验证分数 {acc_test:.4f} 几乎相同，说明没有过拟合。
      反过来，如果训练 1.0000、验证 0.8500，那就是典型的过拟合：
      模型把训练集背了下来（例如不加限制的决策树、KNN 取 k=1）。
      记住这个口诀：**训练高 + 验证低 = 过拟合；两者都低 = 欠拟合；两者都高且接近 = 好。**
      注意训练分数天然偏高（模型见过这些数据），单独看它没有意义，必须与验证分数对照。

【4】分层的作用在本例（iris）看不出来，因为三类各 50 个本来就均衡；
      但在步骤 6 的不平衡数据上，KFold 的每折正类数在
      {min(kf_pos)} ~ {max(kf_pos)} 之间乱跳，而 StratifiedKFold 稳定在
      {min(skf_pos)} ~ {max(skf_pos)}。这就是"分类任务默认分层"的实际价值：
      **让每折的评估条件可比**，从而让均值和标准差都有意义。

【5】关于"CV 分数能不能当最终成绩报告"：不能单独当。
      CV 分数是**模型选择阶段**用的（比较不同模型/超参、估计泛化能力）；
      最终对外报告应使用一个从未参与任何选择的**独立测试集**。
      如果确实没有额外数据，可以报告 CV 均值 ± 标准差，并说明它包含了选择过程的乐观偏差。

【6】实践清单（照着做就不会错）：
      * 分类 → StratifiedKFold（sklearn 传 cv=5 时自动如此）；回归 → KFold；
      * 时序 → TimeSeriesSplit，绝不随机打乱；
      * 数据量小 → K=5；数据量大且追求精确 → K=10；几乎不用 LOOCV；
      * 一切预处理（标准化、缺失值填充、特征选择）都必须放进 **Pipeline**，
        让它在每一折内部只用训练折 fit（见 05_模型保存与加载.py），否则照样泄漏；
      * 报告时给出"均值 ± 标准差"，并说明折数 K 与随机种子。
""")

# --- 绘图 1：各折分数 + 均值线 -----------------------------------------------
fig, ax = plt.subplots(figsize=(8.4, 5.2))
folds = np.arange(1, len(scores) + 1)
bars = ax.bar(folds, scores, width=0.5, color="#2980b9", alpha=0.85,
              label="各折验证准确率", zorder=3)
ax.plot(folds, scores, marker="o", color="#1f4e79", linewidth=1.8, zorder=4,
        label="分数折线")
for fold_x, fold_s in zip(folds, scores):
    ax.annotate(f"{fold_s:.4f}", xy=(fold_x, fold_s), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=10)
ax.axhline(scores.mean(), color="#c0392b", linestyle="--", linewidth=2,
           label=f"平均准确率 = {scores.mean():.4f}", zorder=5)
ax.fill_between([0.5, len(scores) + 0.5],
                scores.mean() - scores.std(), scores.mean() + scores.std(),
                color="#c0392b", alpha=0.12, zorder=1,
                label=f"均值 ± 1 标准差（±{scores.std():.4f}）")
ax.set_xlim(0.5, len(scores) + 0.5)
ax.set_ylim(max(0.0, scores.min() - 0.08), min(1.02, scores.max() + 0.08))
ax.set_xticks(folds)
ax.set_xticklabels([f"第 {i} 折" for i in folds], fontsize=11)
ax.set_title("06 逻辑回归在 iris 上的 5 折交叉验证准确率", fontsize=13, pad=12)
ax.set_xlabel("交叉验证折次", fontsize=11)
ax.set_ylabel("验证集准确率", fontsize=11)
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3, axis="y", zorder=0)
fig.tight_layout()
p1 = OUTPUT_DIR / "06_交叉验证分数.png"
fig.savefig(p1, dpi=130)
plt.close(fig)
print(f"\n[图 1] 已保存各折分数图：{p1}")

# --- 绘图 2：训练分数 vs 验证分数（过拟合诊断图） ---------------------------
fig, ax = plt.subplots(figsize=(8.6, 5.2))
xpos = np.arange(len(metric_names))
width = 0.36
train_means = [cv_results[f"train_{m}"].mean() for m in metric_names]
test_means = [cv_results[f"test_{m}"].mean() for m in metric_names]
test_stds = [cv_results[f"test_{m}"].std() for m in metric_names]
b1 = ax.bar(xpos - width / 2, train_means, width, yerr=[cv_results[f"train_{m}"].std() for m in metric_names],
            capsize=4, color="#8e44ad", alpha=0.88, label="训练折分数均值")
b2 = ax.bar(xpos + width / 2, test_means, width, yerr=test_stds, capsize=4,
            color="#27ae60", alpha=0.88, label="验证折分数均值")
for bars in (b1, b2):
    for bar in bars:
        ax.annotate(f"{bar.get_height():.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
ax.set_xticks(xpos)
ax.set_xticklabels(["准确率\naccuracy", "宏精确率\nprecision_macro",
                    "宏召回率\nrecall_macro", "宏 F1\nf1_macro"], fontsize=10)
ax.set_ylim(0.85, 1.04)
ax.set_title("06 训练分数 vs 验证分数（差距小 = 没有过拟合）", fontsize=13, pad=12)
ax.set_ylabel("分数（误差棒为折间标准差）", fontsize=11)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
p2 = OUTPUT_DIR / "06_训练与验证对比.png"
fig.savefig(p2, dpi=130)
plt.close(fig)
print(f"[图 2] 已保存训练/验证对比图：{p2}")

# --- 绘图 3：KFold vs StratifiedKFold 的每折少数类个数 ----------------------
fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8))
ax0 = axes[0]
ax0.bar(np.arange(1, 6) - 0.2, kf_pos, 0.4, color="#e67e22", alpha=0.9, label="普通 KFold")
ax0.bar(np.arange(1, 6) + 0.2, skf_pos, 0.4, color="#2980b9", alpha=0.9, label="分层 StratifiedKFold")
ax0.axhline(pos_total / 5, color="#c0392b", linestyle="--", linewidth=1.8,
            label=f"理想值 ≈ {pos_total / 5:.1f} 个")
ax0.set_xticks(np.arange(1, 6))
ax0.set_xticklabels([f"第 {i} 折" for i in range(1, 6)], fontsize=10)
ax0.set_title("06 每折中少数类样本个数（越接近虚线越公平）", fontsize=12, pad=10)
ax0.set_xlabel("折次", fontsize=11)
ax0.set_ylabel("少数类（正类）样本数", fontsize=11)
ax0.legend(fontsize=9)
ax0.grid(alpha=0.3, axis="y")

ax1 = axes[1]
ax1.plot(np.arange(1, 6), kf_scores, marker="o", linewidth=2, color="#e67e22",
         label=f"普通 KFold（std={kf_scores.std():.4f}）")
ax1.plot(np.arange(1, 6), skf_scores, marker="s", linewidth=2, color="#2980b9",
         label=f"分层 K 折（std={skf_scores.std():.4f}）")
ax1.set_xticks(np.arange(1, 6))
ax1.set_xticklabels([f"第 {i} 折" for i in range(1, 6)], fontsize=10)
ax1.set_title("06 不平衡数据上的 F1 折间波动", fontsize=12, pad=10)
ax1.set_xlabel("折次", fontsize=11)
ax1.set_ylabel("F1 分数", fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)
fig.tight_layout()
p3 = OUTPUT_DIR / "06_分层K折对比.png"
fig.savefig(p3, dpi=130)
plt.close(fig)
print(f"[图 3] 已保存分层对比图：{p3}")

# =============================================================================
# 超参数怎么调 / 使用注意
# =============================================================================
# 【交叉验证本身也有"超参数"：K、是否分层、是否打乱】
# 1. K 的选择：数据量小（< 1000）用 5 折；比较大且想更稳用 10 折；
#    极不平衡时确保"每折少数类 ≥ 2 个"，必要时减少 K。
# 2. shuffle 与 random_state：默认 shuffle=False（结果确定、可复现）。
#    若数据本身有顺序（例如按类别排好序的原始文件），**必须** shuffle=True + random_state，
#    否则每折的类别分布会严重偏斜。规律：分类 + 有序数据 → StratifiedKFold(shuffle=True)。
# 3. 时序数据：TimeSeriesSplit(n_splits=5, gap=0)，必要时 max_train_size 做滑动窗口。
# 4. 重复交叉验证 RepeatedStratifiedKFold：把"分层切分"重复 n_repeats 次，进一步降低方差，
#    代价是时间线性增长（小数据集上很划算）。
# 5. 与调参的连接：GridSearchCV/RandomizedSearchCV 内部就是"对每组超参跑一遍 CV"，
#    所以本节的 K、scoring、分层策略会直接决定调参结果的可靠性
#    （见 04_网格搜索超参数调优.py）。
# 使用注意：
#    * cross_val_score 传入的 estimator **不会被 fit**，想拿训练好的模型请用
#      cross_validate(return_estimator=True)，或自己在全量数据上重新 fit 一次。
#    * 任何预处理都要放进 Pipeline 再做 CV，否则每个折都会"偷看"全体数据的统计量。
#    * sklearn 的 loss 类指标（如 neg_mean_squared_error）是**负值**，取负号才是误差。
#    * error_score="raise" 能让某折的失败立刻暴露，而不是被写成 nan 悄悄吞掉。
#    * 报告 CV 结果时务必写清：折数 K、是否分层、是否 shuffle、随机种子、scoring 名称。
print("\n【完成】03_交叉验证.py 运行结束")

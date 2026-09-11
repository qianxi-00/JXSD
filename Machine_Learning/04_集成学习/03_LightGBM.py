r"""03_LightGBM.py —— LightGBM 原理与实战（直方图算法 / Leaf-wise / 早停曲线）

对应课案章节
    《机器学习》课案 → 集成学习 → LightGBM
    （课案原文第 951~1064 行一带：名词介绍 / 核心思想 / 直方图加速 / 代码）

本节知识点
    1. 直方图算法（Histogram-based）：把连续特征离散成固定数量的桶（默认 255 个），
       每个桶只记"样本数 / 一阶梯度 G 的和 / 二阶梯度 H 的和"，
       找分裂点时只遍历桶、不遍历样本 —— 数据量越大，加速越明显。
    2. Leaf-wise（叶子优先生长）vs Level-wise（按层生长）：
       每次只挑"分裂收益 Gain 最大"的那个叶子往下长，同样叶子数下损失更低，
       但更容易长深、更容易过拟合。
    3. num_leaves 与 max_depth 的关系：num_leaves <= 2^max_depth。
       LightGBM 官方建议 num_leaves 直接作为主复杂度参数（默认 31），
       而不要同时用 max_depth 去卡（两者会互相打架）。
    4. 分裂收益 Gain 的直观含义：Gain = 分开后的"纯度提升" − 分裂的手续费，
       Gain > 0 才值得分；Leaf-wise 每次挑 Gain 最大的叶子。
    5. GOSS（基于梯度的单边采样）与 EFB（互斥特征捆绑）两个进一步提速的技术。
    6. lightgbm 4.7.0 sklearn 接口：eval_set 已过时，应改用 eval_X / eval_y。

运行方式
    PowerShell（路径含中文，必须加引号并用 & 调用）：

    $env:PYTHONUTF8='1'
    & 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Machine_Learning\04_集成学习\03_LightGBM.py'

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
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# 教学脚本输出保持干净：只屏蔽"将来版本才会变"的告警噪音，不屏蔽真正的错误。
# 本机 lightgbm 4.7.0 会把 eval_set 标成过时参数（改用 eval_X / eval_y），此处统一压掉噪音。
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
section("① 原理与数学推导：直方图算法 / Leaf-wise / Gain / GOSS 与 EFB")

print(
    r"""
【1】LightGBM 全称 Light Gradient Boosting Machine。
     它还是 Boosting：一棵树一棵树地训练，后一棵继续修正前面模型的错误。
     区别在于它更强调**训练速度**和**大规模数据处理能力**，所以特别适合数据量大的场景。
     "轻（Light）"体现在两个地方：直方图算法（算得少）+ Leaf-wise 生长（长在刀刃上）。

【2】直方图算法（LightGBM 快的秘诀之一）

     XGBoost 的 exact 模式找分裂点要"逐个样本"排序、枚举每一个取值；
     LightGBM 先把连续特征离散化装进固定数量的桶里（默认 max_bin = 255）：

         年龄特征:  0~18 | 18~30 | 30~45 | 45~60 | 60+
                     桶0     桶1     桶2     桶3    桶4
                     ↑ 每个桶只记三样东西：桶里有多少样本、G 加总、H 加总

     之后找分裂点只遍历桶（比如 255 个），不遍历每一个样本。复杂度从 O(#样本) 降到 O(#桶)。
     因为桶的数量是常数 255，**数据量越大，这个加速越明显**（100 万行数据也只数 255 个桶）。

     代价：特征被人为离散了，理论上会损失一点点精度（实际几乎看不出来），
     换来的是数量级的速度提升 + 内存占用大幅下降（不用存原始浮点数，只存桶编号，通常用 8 位整数）。

     工业上还有两个进一步的优化（了解即可）：
       * GOSS（Gradient-based One-Side Sampling，基于梯度的单边采样）：
         梯度大的样本说明"还没学好"，全部保留；梯度小的样本随机丢弃一部分。
         这样只用一部分样本就能比较准确地估计分裂增益，进一步提速。
       * EFB（Exclusive Feature Bundling，互斥特征捆绑）：
         把"几乎不会同时取非零值"的稀疏特征（互斥特征）捆绑成一个特征，
         把上千维稀疏特征压成几十维，显著减少特征数量。

【3】分裂收益 Gain 的直观含义（课案"核心思想"一节）

         Gain = 分开左右两边后的"纯度提升" − 分裂一次的手续费

     * "纯度提升"：分裂后左右两个子节点各自内部的样本更一致了，损失下降了多少。
       用 XGBoost 那套二阶公式写就是：
         Gain = (1/2) * [ G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - (G_L+G_R)^2/(H_L+H_R+lambda) ] - gamma
     * "手续费"：分裂本身要付出的代价（引入复杂度），主要就是 gamma 那一项。
     * 结论：Gain > 0 才值得分，越大说明这个分裂点越划算。

     不需要纠结公式怎么推出来的，只要知道它干一件事：
     **给每个候选分裂点打分 —— 哪个分裂点最划算就选哪个。**

【4】Leaf-wise（叶子优先生长）vs Level-wise（按层生长）

     Level-wise（XGBoost 默认、sklearn 的 GBDT 也是这种）：
         同一层的所有节点一起分裂，长完这一层再长下一层。树很"整齐"。

               ●            第 0 层：1 个节点
             /   \
            ●     ●          第 1 层：2 个节点
           / \   / \
          ●   ● ●   ●        第 2 层：4 个节点（不管这些节点是不是都值得分裂）

     Leaf-wise（LightGBM 默认）：
         每次从当前所有叶子中挑出"分裂收益 Gain 最大"的那一个去分裂，
         不管它在第几层。长出来的树是"歪"的，但每一片叶子都是"最值得长"的。

               ●            先挑 Gain 最大的叶子长
             /   \
            ●     ●
           / \
          ●   ●          ← 继续从全部叶子里挑 Gain 最大的（可能还是这一支）

     对比结论：
         * 同样数量的叶子下，Leaf-wise 的训练损失更低（因为它把"分裂预算"花在了收益最大的地方）。
         * 但 Leaf-wise 更容易长出很深的树，**参数没调好时更容易过拟合**。
         * 所以 LightGBM 用 num_leaves 直接限制叶子数，而不是靠 max_depth。

【5】num_leaves 与 max_depth 的关系

     一棵二叉树里，叶子数 T 和深度 d 满足：   T <= 2^d
         max_depth=3 → 最多 2^3 = 8 个叶子
         max_depth=7 → 最多 2^7 = 128 个叶子
     反过来说，num_leaves=31 意味着"至少要 5 层才装得下"（2^5 = 32 >= 31）。

     官方建议：**num_leaves 才是控制复杂度的主参数，不要同时用 max_depth 去双重限制**。
     因为 Leaf-wise 会优先往深处长，max_depth 一旦卡住，就会浪费掉一部分"分裂预算"
     （有的分支还没长够就被深度上限掐死了），还可能让模型欠拟合。
     实践口诀：
         num_leaves 调到"过拟合的边界"（常用 31~255，数据越大可以越大），
         max_depth 设一个宽松的上限（如 -1 不限制，或 7~10）做安全网即可。

【6】LightGBM 与 XGBoost 的定位差异

    | 维度     | XGBoost                              | LightGBM                              |
    |----------|--------------------------------------|---------------------------------------|
    | 定位     | 稳定、通用、正则化强                  | 训练快、占内存少，适合大数据          |
    | 分裂点   | 预排序 / 加权分位数 / 直方图三种可选  | 默认就是直方图（max_bin=255）         |
    | 生长方式 | Level-wise                           | Leaf-wise（更省叶子、更易过拟合）     |
    | 小数据   | 往往更稳（Leaf-wise 在小数据上易过拟合）| 参数没调好容易过拟合                |
    | 大数据   | 慢一些、内存高                        | 优势明显（快 + 省内存）               |

    一句话：LightGBM "更轻、更快"，代价是默认配置比 XGBoost 更"激进"，需要更小心地调参。
"""
)

# =============================================================================
# ② 库 API 关键参数逐个解释
# =============================================================================
section("② lightgbm API 关键参数逐个解释：LGBMClassifier（lightgbm 4.7.0）")

print(
    r"""
LGBMClassifier(boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1,
               n_estimators=100, min_child_samples=20, subsample=1.0, colsample_bytree=1.0,
               reg_alpha=0.0, reg_lambda=0.0, min_split_gain=0.0, verbose=-1, n_jobs=None, ...)

  n_estimators        树的数量（boosting 轮数），默认 100。有早停时它是"上限"。
  learning_rate       学习率 eta，默认 0.1。越小越稳、需要越多树（与 n_estimators 联动）。
  num_leaves          一棵树最多有多少个叶子，默认 31，**LightGBM 最重要的复杂度参数**。
                      调大 → 树更强 → 更容易过拟合；调小 → 更保守。
                      经验：不超过 2^max_depth，且样本量大时才敢往 255 以上调。
  max_depth           树的最大深度，默认 -1（不限制）。建议只当"安全网"用（如 7~10），
                      主限制交给 num_leaves（两者的关系：num_leaves <= 2^max_depth）。
  min_child_samples   一个叶子最少需要多少样本，默认 20。**LightGBM 最常用的抗过拟合参数**：
                      调大 → 不允许长出"只装几个样本"的叶子 → 更平滑、更抗噪声。
                      （等价于 sklearn 的 min_samples_leaf、XGBoost 的 min_child_weight。）
  subsample           行采样比例，默认 1.0。LightGBM 里它对应别名 bagging_fraction。
                      注意：LightGBM 要求同时设置 subsample_freq > 0 才会真正启用 bagging，
                      纯 sklearn 接口下直接给 subsample < 1 一般也会生效，但写清 subsample_freq=1 更稳妥。
  colsample_bytree    列采样比例，默认 1.0，对应别名 feature_fraction。
                      每棵树只用一部分特征找分裂点，既加速又降方差（和随机森林的机制类似）。
  reg_lambda          L2 正则系数（别名 lambda_l2），默认 0.0，压制叶子权重过大。
  reg_alpha           L1 正则系数（别名 lambda_l1），默认 0.0，可让部分叶子权重变 0。
  min_split_gain      执行分裂所需的最小 Gain（别名 min_gain_to_split），默认 0.0。
                      等价于 XGBoost 的 gamma：Gain 不超过它就放弃分裂。
  verbose             日志详细程度。**设成 -1 可以完全关掉 LightGBM 的 C 层日志**
                      （否则会出现一堆 "[LightGBM] [Warning] No further splits with positive gain"
                      和 "[LightGBM] [Info] ..." 刷屏）。这是本机最需要注意的一个参数。
  n_jobs              线程数，默认自动。设 -1 用满所有核；设 1 便于复现（多线程下浮点求和不完全确定）。
  boosting_type       默认 'gbdt'；可选 'dart'（带 dropout）、'goss'（单边梯度采样，只能用 subsample=1）。
  max_bin             直方图桶的数量，默认 255。调小 → 更快更省内存，但精度略降。
  random_state        随机种子，固定后采样可复现。

  【lightgbm 4.7.0 sklearn 接口的重要注意事项】
    1) eval_set 已被标记为过时（DeprecationWarning），新写法是：
           model.fit(X_train, y_train, eval_X=(X_val,), eval_y=(y_val,), eval_names=["验证集"])
       用 eval_names 起的名字会成为 evals_result_ 里的键名。
    2) 早停用回调：callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
       其中 verbose=False 可以让早停本身也不打印 "[LightGBM] [Info] Early stopping..."。
    3) 如果关掉 verbose=-1 后仍有日志从 callbacks 里冒出来，
       可以再补一个 lgb.log_evaluation(period=0) —— period=0 表示"每 0 轮打印一次"，即不打印。
    4) 树模型不需要标准化：LightGBM 只按特征的排序找分桶边界，特征尺度无关紧要。

  【sklearn 接口 vs 原生接口的调用差异（速查）】
    * sklearn 接口：LGBMClassifier(...).fit(X, y) → 支持 predict / predict_proba，
      能直接进 sklearn 的 Pipeline / GridSearchCV；参数名用 sklearn 风格（n_estimators、subsample）。
    * 原生接口：lgb.Dataset(X, y) + lgb.train(params, dtrain, num_boost_round=...)
      + lgb.cv(...)；参数名用 LightGBM 自己的风格（num_leaves、bagging_fraction、
      feature_fraction、lambda_l2），功能更全（如自定义目标函数、交叉验证、增量训练）。
    两者底层是同一个 Booster，sklearn 接口只是包了一层。
"""
)

# =============================================================================
# ③ 完整可运行代码
# =============================================================================
section("③ 完整可运行代码：直方图 + Leaf-wise + 早停曲线")

# ----------------------------- 3.1 造数据（与 02 同规模，便于对比） -----------------------------
print("\n--- 3.1 造数据（与 02_XGBoost.py 完全同规模、同划分方式）---")

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
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train_full
)
print(f"训练集 {X_train.shape[0]} 条 / 验证集 {X_val.shape[0]} 条 / 测试集 {X_test.shape[0]} 条，"
      f"特征 {X.shape[1]} 个")

# ----------------------------- 3.2 训练（带早停） -----------------------------
print("\n--- 3.2 训练 LGBMClassifier（verbose=-1 关日志 + 回调式早停）---")

model = LGBMClassifier(
    n_estimators=1000,          # 上限 1000 轮，实际用多少由早停决定
    learning_rate=0.05,         # 学习率
    num_leaves=31,              # 主复杂度参数：一棵树最多 31 个叶子
    max_depth=-1,               # 不额外限制深度，把复杂度控制权交给 num_leaves
    min_child_samples=20,       # 叶子最少 20 个样本，抗过拟合
    subsample=0.8,              # 行采样（bagging_fraction）
    subsample_freq=1,           # 每 1 轮做一次 bagging，确保 subsample 真的生效
    colsample_bytree=0.8,       # 列采样（feature_fraction）
    reg_lambda=1.0,             # L2 正则
    reg_alpha=0.0,              # L1 正则（不启用）
    min_split_gain=0.0,         # 分裂所需最小 Gain（相当于 gamma）
    random_state=RANDOM_STATE,
    n_jobs=2,
    verbose=-1,                 # 关键：-1 = 完全关闭 LightGBM 的 C 层日志
)

t0 = time.perf_counter()
model.fit(
    X_train, y_train,
    eval_X=(X_val,),            # 4.7.0 新写法（eval_set 已过时）
    eval_y=(y_val,),
    eval_names=["验证集"],
    eval_metric="binary_logloss",
    callbacks=[
        lgb.early_stopping(stopping_rounds=20, verbose=False),  # 20 轮不改善就停，且不打印
        lgb.log_evaluation(period=0),                           # 双保险：每 0 轮打印 = 不打印
    ],
)
train_seconds = time.perf_counter() - t0
print(f"训练耗时：{train_seconds:.3f} 秒")

# ----------------------------- 3.3 早停结果 -----------------------------
print("\n--- 3.3 早停结果解读 ---")

best_round = int(model.best_iteration_)
y_pred = np.asarray(model.predict(X_test)).reshape(-1)
y_proba = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"最优迭代轮数 best_iteration_ = {best_round}（上限 1000）")
print(f"测试集准确率：{acc:.4f}")
print(f"测试集 AUC：{auc:.4f}")

# ----------------------------- 3.4 验证集损失曲线 -----------------------------
print("\n--- 3.4 从 evals_result_ 取验证集损失曲线 ---")

print(f"evals_result_ 的键（= fit 里 eval_names 起的名字）：{list(model.evals_result_.keys())}")
val_curve = np.asarray(model.evals_result_["验证集"]["binary_logloss"], dtype=float)
rounds = np.arange(1, len(val_curve) + 1)
print(f"曲线共 {len(val_curve)} 个点；第 1 轮 {val_curve[0]:.4f} → "
      f"最低点在第 {int(np.argmin(val_curve)) + 1} 轮 = {val_curve.min():.4f} → "
      f"最后一轮 {val_curve[-1]:.4f}")

# ----------------------------- 3.5 num_leaves 对比实验 -----------------------------
print("\n--- 3.5 num_leaves 对比实验 ---")
print("公平起见：固定 300 轮、关掉早停和采样，只改 num_leaves，看过度增长会怎样")
print(f"{'num_leaves':>11} | {'训练准确率':>11} | {'测试准确率':>11} | {'训练-测试差距':>14} | {'测试AUC':>9}")
print("-" * 72)
leaves_rows = []
for n_leaves in (7, 15, 31, 63, 255):
    m_lv = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=n_leaves,
        max_depth=-1,              # 不限制深度，让 num_leaves 独自决定复杂度
        min_child_samples=5,       # 故意放宽，好让"叶子太多"的坏处暴露出来
        subsample=1.0,             # 关掉采样，排除其它随机性干扰
        colsample_bytree=1.0,
        reg_lambda=0.0,
        random_state=RANDOM_STATE,
        n_jobs=2,
        verbose=-1,
    )
    m_lv.fit(X_train, y_train)
    tr_a = accuracy_score(y_train, m_lv.predict(X_train))
    te_a = accuracy_score(y_test, m_lv.predict(X_test))
    te_u = roc_auc_score(y_test, m_lv.predict_proba(X_test)[:, 1])
    leaves_rows.append((n_leaves, tr_a, te_a, te_u))
    print(f"{n_leaves:>11d} | {tr_a:>11.4f} | {te_a:>11.4f} | {tr_a - te_a:>14.4f} | {te_u:>9.4f}")

# ----------------------------- 3.6 num_leaves 和 max_depth 的关系 -----------------------------
print("\n--- 3.6 num_leaves 与 max_depth 的关系（T <= 2^d）---")
for d in (3, 5, 7, 8, 10):
    print(f"  max_depth={d:>2} → 理论上最多 {2 ** d:>5} 个叶子；"
          f"num_leaves=31 需要至少 {int(np.ceil(np.log2(31)))} 层才装得下")
print("  → 所以 max_depth=3 时 num_leaves=31 根本用不满，两个参数同时卡会互相打架。")

# ----------------------------- 3.7 直方图桶数的加速效果 -----------------------------
print("\n--- 3.7 max_bin 桶数对速度的影响（固定 300 轮，只比较纯训练耗时）---")
print(f"{'max_bin':>8} | {'训练耗时(秒)':>12} | {'测试准确率':>11} | {'测试AUC':>9}")
print("-" * 52)
bin_rows = []
for max_bin in (15, 63, 255):
    m_bin = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.1,
        num_leaves=31,
        max_depth=-1,
        max_bin=max_bin,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=2,
        verbose=-1,
    )
    t0 = time.perf_counter()
    m_bin.fit(X_train, y_train)
    dt = time.perf_counter() - t0
    a_bin = accuracy_score(y_test, m_bin.predict(X_test))
    u_bin = roc_auc_score(y_test, m_bin.predict_proba(X_test)[:, 1])
    bin_rows.append((max_bin, dt, a_bin, u_bin))
    print(f"{max_bin:>8d} | {dt:>12.3f} | {a_bin:>11.4f} | {u_bin:>9.4f}")

# =============================================================================
# 绘图
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

axes[0].plot(rounds, val_curve, color="#9467bd", lw=1.9, label="验证集 binary_logloss")
axes[0].axvline(best_round, color="#d62728", ls="--", lw=1.5,
                label=f"早停最优轮 = 第 {best_round} 轮")
axes[0].scatter([best_round], [val_curve[best_round - 1]], color="#d62728", zorder=5, s=45)
axes[0].set_xlabel("Boosting 轮数（树的数量）")
axes[0].set_ylabel("验证集 binary logloss（越小越好）")
axes[0].set_title("LightGBM 训练曲线：验证损失快速下降后趋于平稳（早停截断）")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

lv = np.asarray([r[0] for r in leaves_rows])
tr_acc = np.asarray([r[1] for r in leaves_rows])
te_acc = np.asarray([r[2] for r in leaves_rows])
axes[1].plot(range(len(lv)), tr_acc * 100, "o-", color="#1f77b4", lw=1.8, label="训练集准确率(%)")
axes[1].plot(range(len(lv)), te_acc * 100, "s-", color="#d62728", lw=1.8, label="测试集准确率(%)")
axes[1].set_xticks(range(len(lv)))
axes[1].set_xticklabels([str(v) for v in lv])
axes[1].set_xlabel("num_leaves（一棵树最多多少个叶子）")
axes[1].set_ylabel("准确率（%）")
axes[1].set_title("num_leaves 越大 → 测试集先好后掉（训练集已全为 100%，看不出问题）")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

fig.suptitle(f"03 LightGBM（直方图 max_bin=255 / Leaf-wise / 早停 20 轮）"
             f"  最优 {best_round} 轮 / 测试准确率 {acc:.4f} / AUC {auc:.4f}", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUTPUT_DIR / "04_LightGBM_训练曲线.png", dpi=130)
plt.close(fig)

# =============================================================================
# ④ 结果解读
# =============================================================================
section("④ 结果解读：这些数字到底在说什么")

print(
    f"""
1) 最优迭代轮数 {best_round}（上限 1000）
   —— 验证集损失在第 {int(np.argmin(val_curve)) + 1} 轮见底（{val_curve.min():.4f}），
   之后回升到 {val_curve[-1]:.4f}；LightGBM 在连续 20 轮（stopping_rounds）没有改善后触发早停，
   并把模型回滚到最优轮。注意 best_iteration_ 是"已经用了多少轮"，
   和 XGBoost 的 best_iteration（从 0 开始）在数值口径上不一样，比较时要用同一口径。

2) 测试集准确率 {acc:.4f}、AUC {auc:.4f}，训练耗时 {train_seconds:.3f} 秒
   —— 和 02_XGBoost.py（同数据同划分）属于同一水平：这份数据只有 1000 条，
     LightGBM 的"大数据优势"体现不出来，小数据上两者本来就在伯仲之间。
     LightGBM 真正的加速要在几万行以上才明显（直方图把 O(#样本) 降成 O(#桶)）。
     谁快谁准要看 04_三种Boosting对比.py 里同一轮次的实测结果。

3) num_leaves 对比实验（固定 300 轮、关掉早停和采样，见上面那张表）：
   * 训练集准确率五组全是 1.0000 —— 这个数据集只有 750 条训练样本，
     300 棵 Leaf-wise 的树不管每棵给 7 个还是 255 个叶子，都能把训练集背得一个不剩。
     **教训：只看训练集准确率，你根本发现不了过拟合。**
   * 测试集准确率：{leaves_rows[0][2]:.4f}（7 叶）→ {leaves_rows[1][2]:.4f} / {leaves_rows[2][2]:.4f}（15~31 叶最好）
     → {leaves_rows[3][2]:.4f} / {leaves_rows[4][2]:.4f}（63~255 叶掉下来）；
   * 测试集 AUC 则一路下滑：{leaves_rows[0][3]:.4f} → {leaves_rows[1][3]:.4f} → {leaves_rows[2][3]:.4f}
     → {leaves_rows[3][3]:.4f} → {leaves_rows[4][3]:.4f}
   —— AUC 比准确率更敏感：叶子给多了以后，模型对训练集里的噪声"越来越自信"，
     预测概率被推得离 0/1 更近，排序质量反而变差。这就是 Leaf-wise 容易过拟合的直接证据。
     结论：num_leaves 要调到"验证集/测试集最好"的位置（这份数据是 15~31），而不是越大越好。

4) max_bin 桶数实验（固定 300 轮，纯比训练耗时）：
   max_bin={bin_rows[0][0]} 耗时 {bin_rows[0][1]:.3f} 秒 → max_bin={bin_rows[1][0]} 耗时 {bin_rows[1][1]:.3f} 秒
   → max_bin={bin_rows[2][0]} 耗时 {bin_rows[2][1]:.3f} 秒（桶越多越慢，方向符合预期），
   而测试 AUC 分别是 {bin_rows[0][3]:.4f} / {bin_rows[1][3]:.4f} / {bin_rows[2][3]:.4f}，基本没变。
   —— 这正是直方图算法的价值：用"把特征离散成少量桶"换取速度提升，代价是极小的精度损失。
   桶数是常数、样本数是变量，所以**数据量越大，这个加速越明显**
   （1000 行时差距只有毫秒级，100 万行时就是数倍到数十倍的差距）。

5) num_leaves <= 2^max_depth 的关系：
   num_leaves=31 至少要 5 层（2^5=32）才装得下，所以 max_depth=3（最多 8 个叶子）
   和 num_leaves=31 是**互相矛盾**的：深度上限会先把树掐死，num_leaves 根本用不满。
   这就是"不要同时用两个参数卡复杂度"的原因。

6) 关于日志：本脚本用 verbose=-1 + callbacks=[..., lgb.log_evaluation(period=0)]
   把 LightGBM 的 C 层日志和早停提示全部关掉了。
   如果你自己跑的时候看到 "[LightGBM] [Warning] No further splits with positive gain"，
   那不是报错，是它在告诉你"当前这个叶子已经找不到 Gain > 0 的分裂点了"，
   属于正常现象；verbose=-1 就能让它闭嘴。
"""
)

# =============================================================================
# 超参数怎么调
# =============================================================================
section("超参数怎么调（LightGBM）")

print(
    """
【第一步：定"学多久"】
    learning_rate 先用 0.05~0.1，n_estimators 设大（1000+）并开早停
    （callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)] + eval_X/eval_y）。
    联动关系同样成立：learning_rate 减半 → 树的数量大约翻倍。

【第二步：定"单棵树多复杂"（最重要）】
    num_leaves          默认 31，是 LightGBM 的核心复杂度参数。常用 31~255。
                        数据量越大越敢调大；小数据（几千行）建议 15~31。
    max_depth           默认 -1。建议只做安全网（7~10），主限制交给 num_leaves，
                        因为 num_leaves <= 2^max_depth，两者同时卡会互相打架。
    min_child_samples   默认 20，最常用的抗过拟合参数。样本少/噪声大时调到 30~100。

【第三步：加随机性压过拟合】
    colsample_bytree（feature_fraction）  默认 1 → 试 0.7~0.9（高维可 0.5）。
    subsample（bagging_fraction）         默认 1 → 试 0.7~0.9，
                                          并记得 subsample_freq=1 让它真正生效。

【第四步：正则化】
    reg_lambda（lambda_l2）  默认 0 → 试 1~20，压制叶子权重。
    reg_alpha（lambda_l1）   默认 0 → 特征多时可试 0.1~1。
    min_split_gain           默认 0（相当于 XGBoost 的 gamma）→ 试 0.1~1 剪掉没价值的分裂。

【工程参数】
    verbose=-1           必开，否则日志刷屏。
    max_bin              默认 255。追求速度可降到 63/127，追求精度可升到 511。
    n_jobs               线程数；-1 用满。
    boosting_type        'gbdt'（默认）/ 'dart'（精度略高但慢）/ 'goss'（大数据提速，
                         此时 subsample 必须为 1）。

【调参顺序建议】
    learning_rate + 早停  →  num_leaves / min_child_samples  →  colsample_bytree / subsample
    →  reg_lambda / min_split_gain  →  最后考虑 learning_rate 减半、树数翻倍。
    小数据集上如果 LightGBM 打不过 XGBoost，八成是 num_leaves 太大 / min_child_samples 太小，
    先把这两个往保守方向调。
"""
)

print(f"\n生成的图片：\n  {OUTPUT_DIR / '04_LightGBM_训练曲线.png'}")
print("\n【完成】03_LightGBM.py 运行结束")

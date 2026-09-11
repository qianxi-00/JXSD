r"""01_GBDT.py —— GBDT（梯度提升决策树）原理与实战

对应课案章节
    《机器学习》课案 → 集成学习 → Boosting 家族的统一加法公式 / GBDT
    （课案原文第 687~802 行一带）

本节知识点
    1. Boosting 的统一加法模型：F_m(x) = F_{m-1}(x) + eta * f_m(x)，
       每一轮只训练"当前这一棵"小树 f_m 去修正前面模型的残差。
    2. GBDT 用回归树拟合 **负梯度**（也就是"往哪个方向走损失下降最快"）：
       回归任务里负梯度恰好等于残差 y - F_{m-1}(x)；
       分类任务里用的是对数损失的负梯度（不是原始残差）。
    3. shrinkage（收缩 / 学习率 learning_rate）：每棵树只允许修正一点点，
       η 越小单棵树影响越小，但需要的树越多。
    4. subsample 随机梯度提升（Stochastic Gradient Boosting）：
       每棵树只用一部分样本训练，降低方差、抑制过拟合。
    5. sklearn 的 train_score_ / staged_predict / staged_predict_proba
       可以还原"每一轮提升之后模型长什么样"，用来画提升过程曲线。

运行方式
    PowerShell（路径含中文，必须加引号并用 & 调用）：

    $env:PYTHONUTF8='1'
    & 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Machine_Learning\04_集成学习\01_GBDT.py'

    脚本不会弹出任何窗口（matplotlib 使用 Agg 后端），图片统一保存到
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
from sklearn.datasets import load_iris, make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# 教学脚本的输出要保持干净：只屏蔽"未来版本才会变"的告警，不屏蔽真正的错误。
# （本机 lightgbm 4.7.0 会把 eval_set 标成过时参数，此处统一压掉噪音。）
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

RANDOM_STATE = 42
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def section(title: str) -> None:
    """打印一个醒目的中文分节标题，方便对照屏幕输出与源码结构。"""
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# =============================================================================
# ① 原理与数学推导
# =============================================================================
section("① 原理与数学推导：Boosting 的加法模型 + 用负梯度方向拟合新树")

print(
    """
【1】Boosting 的统一加法模型（课案"Boosting 家族的统一加法公式"一节）

    初始模型：      F_0(x) = argmin_c  sum_i L(y_i, c)      —— 先给一个最粗糙的常数预测
    第 m 轮迭代：   F_m(x) = F_{m-1}(x) + eta * f_m(x)      —— 在旧模型上"加"一棵新小树
    最终模型：      F_M(x) = F_0(x) + eta * sum_{m=1..M} f_m(x)

    符号表：
        F_0(x)        初始模型（第一个"基础预测"，分类任务里通常是先验对数几率）
        f_m(x)        第 m 个弱模型（第 m 轮用来补错误的小树，通常是深度 3 左右的回归树）
        eta           学习率 / 收缩系数 shrinkage，每次修正的力度（0 < eta <= 1）
        M             弱模型数量（= n_estimators，一共训练多少轮）
        F_M(x)        最终模型（多轮修正后得到的最终预测）

    关键点：模型不是一步到位学出来的，而是"先有一个初始答案，再一轮一轮加小模型修正错误"。
    串行结构决定了 boosting 天然不能像随机森林那样并行训练树。

【2】前向分步算法（Forward Stagewise Additive Modeling）

    第 m 轮我们求解：
        (eta * f_m) = argmin_{f}  sum_i L( y_i , F_{m-1}(x_i) + f(x_i) )

    注意 argmin 只作用在"当前这一个 f"上，前面已经训练好的 F_{m-1} 被当成常数冻结，
    不去回头修改。所以每一轮只是一个"小优化问题"，容易求解，这也是串行可行的原因。

【3】为什么"负梯度"就是修正方向

    把 L(y_i, F_{m-1}(x_i) + f(x_i)) 在 F_{m-1}(x_i) 处做一阶泰勒展开：
        L ≈ L(y_i, F_{m-1}(x_i)) + f(x_i) * [ dL/dF 在 F_{m-1}(x_i) 处的值 ]

    要让损失变小，就应该让 f(x_i) 沿着负梯度方向取值，即
        f_m(x_i) ≈ - [ dL(y_i, F(x_i)) / dF(x_i) ] |_{F = F_{m-1}}
    这个量就叫"负梯度 / 伪残差（pseudo-residual）"。

    * 回归任务（平方损失 L = 0.5*(y - F)^2）：
        -dL/dF = y - F_{m-1}(x)  —— 负梯度恰好就是普通残差（真实值 - 当前预测值）！
        所以 GBDT 回归时，"拟合负梯度" ≡ "拟合残差"，非常直观。
    * 二分类任务（对数损失 / log loss，也就是课案里说的"偏差 deviance"）：
        L = -[ y*log(p) + (1-y)*log(1-p) ]，其中 p = sigmoid(F)
        -dL/dF = y - p      —— 是"真实标签 0/1 减去当前预测概率"，同样是一个残差的形状，
        只不过它是概率残差，不是原始数值残差。

【4】考试提分式的直观例子（课案"GBDT_考试分数例子"）

    假设真实分数 y = 100。
      第 0 轮：F_0 = 全班平均分 = 60        → 残差 = 100 - 60 = 40
      第 1 轮：让第 1 棵树去学"残差 40"，但它只学 8 成 → 预测 +32（eta=0.8 的收缩）
               F_1 = 60 + 32 = 92          → 残差 = 100 - 92 = 8
      第 2 轮：让第 2 棵树去学"残差 8"，同样只学 8 成 → 预测 +6.4
               F_2 = 92 + 6.4 = 98.4       → 残差 = 1.6
      第 3 轮：……以此类推，误差被一轮一轮"擦掉"。
    每棵树都很弱（只会补一点点），但很多棵树叠加起来就逼近了真实答案。
    学习率 eta 就是"只学几成"这个折扣系数。

【5】GBDT 与随机森林的区别

    | 维度         | 随机森林 (Bagging)                  | GBDT (Boosting)                     |
    |--------------|-------------------------------------|-------------------------------------|
    | 树之间关系   | 并行独立训练                        | 串行，后一棵依赖前一棵              |
    | 每棵树目标   | 各自独立拟合全量数据（自助采样）    | 拟合前面模型尚未修正的负梯度/残差   |
    | 组合方式     | 大家一起投票 / 取平均               | 加法累加（带学习率加权求和）        |
    | 基学习器     | 深树（强学习器，低偏差高方差）      | 浅树（弱学习器，高偏差低方差）      |
    | 主要降低     | 方差                                | 偏差                                |
    | 抗噪能力     | 较强（投票平滑掉了噪声）            | 较弱，容易把噪声也当残差学进去      |

【6】sklearn 中的三个工程细节

    * learning_rate（shrinkage）：不是梯度下降的学习率，而是"每棵树的输出乘一个系数"。
      它和 n_estimators 是**联动**的：learning_rate 调小一半，树的数量通常要翻倍。
    * subsample < 1.0：随机梯度提升。每棵树只从训练集里随机抽 subsample 比例样本，
      既加速又引入随机性，显著降低方差。
    * max_features < 1.0：每棵树的每次分裂只从一部分特征里挑最优切分，
      作用和随机森林的 feature_fraction 类似。
"""
)

# =============================================================================
# ② sklearn / 库 API 关键参数逐个解释
# =============================================================================
section("② sklearn API 关键参数逐个解释：GradientBoostingClassifier")

print(
    """
GradientBoostingClassifier(loss='log_loss', learning_rate=0.1, n_estimators=100,
                           subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                           min_samples_leaf=1, max_depth=3, max_features=None,
                           validation_fraction=0.1, n_iter_no_change=None, tol=1e-4,
                           random_state=None)

  loss                 损失函数，决定"负梯度长什么样"。
                       'log_loss'（默认，等价于旧的 'deviance'）：对数损失，
                           二分类/多分类都能用，输出概率；
                       'exponential'：指数损失，就是 AdaBoost 的损失，只适合二分类。
  learning_rate        学习率 η / 收缩系数 shrinkage，默认 0.1。
                       每棵树的贡献要乘上它。越小越稳、越不容易过拟合，但需要更多树。
                       【联动关系】learning_rate 减半 → n_estimators 大致要翻倍。
  n_estimators         弱模型数量 M，也就是 boosting 的轮数 / 树的数量，默认 100。
                       太大 → 训练集误差趋近 0 但测试误差回升（过拟合）。
  max_depth            每棵回归树的最大深度，默认 3。
                       这是 GBDT 最重要的复杂度旋钮：单棵树越深 → 越强 → 越容易过拟合。
                       实践中 3~8 最常见（"弱学习器"才符合 boosting 的初衷）。
  min_samples_leaf     叶子节点最少样本数，默认 1。
                       调大（如 20）可以防止树长出"只装 1 个样本"的叶子，是另一种抗过拟合手段。
  subsample            每棵树使用的训练样本比例，默认 1.0（用全部样本）。
                       取 (0,1) 即随机梯度提升，常用 0.8；既提速又降方差。
  max_features         每次分裂时考虑的特征比例/数量，默认 None（用全部特征）。
                       常用 'sqrt' 或 0.8，引入特征层面的随机性，进一步抗过拟合。
  validation_fraction  从训练集里再切出多少比例做内部验证集，默认 0.1，仅用于早停。
  n_iter_no_change     早停耐心值：内部验证集损失连续这么多轮没有改善就停下，默认 None（不早停）。
                       设成 20 就相当于"连续 20 轮没进步就收工"。
  tol                  早停阈值：改善幅度小于 tol 就算"没改善"，默认 1e-4。
  random_state         随机种子，固定后 subsample / max_features 的抽样可复现。
  criterion            回归树的分裂准则，默认 'friedman_mse'（Friedman 改进的均方误差），
                       它和负梯度拟合的数学推导更匹配，一般不用改。

  【注意】树模型不需要标准化 / 归一化：分裂只看特征的排序，不看数值尺度。
"""
)

# =============================================================================
# ③ 完整可运行代码
# =============================================================================
section("③ 完整可运行代码：先复现课案 iris 示例，再做 1000 样本的提升过程分析")

# ----------------------------- 3.1 课案原味示例（iris） -----------------------------
print("\n--- 3.1 复现课案 iris 示例（150 个样本，3 分类）---")

iris = load_iris()
X_iris, y_iris = iris.data, iris.target

X_tr_i, X_te_i, y_tr_i, y_te_i = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=RANDOM_STATE, stratify=y_iris
)

model_iris = GradientBoostingClassifier(
    n_estimators=100,      # M：100 棵树
    learning_rate=0.1,     # η：每棵树只修正 0.1 倍
    max_depth=3,           # 每棵树最多 3 层
    random_state=RANDOM_STATE,
)
model_iris.fit(X_tr_i, y_tr_i)
y_pred_i = model_iris.predict(X_te_i)

print("测试集预测结果（前 15 个）：", y_pred_i[:15])
print("测试集真实结果（前 15 个）：", y_te_i[:15])
print(f"iris 测试集准确率：{accuracy_score(y_te_i, y_pred_i):.4f}")
print(f"iris 训练集准确率：{accuracy_score(y_tr_i, model_iris.predict(X_tr_i)):.4f}")

# ----------------------------- 3.2 构造带噪声的数据 -----------------------------
print("\n--- 3.2 构造 1000 样本、带 10% 标签噪声的数据（用来展示过拟合）---")

# flip_y=0.1 表示把 10% 的标签随机翻转成别的类，制造"不可学习的噪声"，
# 这样树的轮数一多，模型就会开始死记硬背这些噪声，测试误差回升。
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,      # 20 个特征里只有 10 个真正有用
    n_redundant=2,
    n_classes=2,
    flip_y=0.1,            # 10% 标签噪声
    class_sep=1.0,
    random_state=RANDOM_STATE,
)
feature_names = [f"特征{i:02d}" for i in range(X.shape[1])]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)
print(f"数据形状：X={X.shape}，训练集 {X_train.shape[0]} 条，测试集 {X_test.shape[0]} 条")
print(f"正类比例：{y.mean():.3f}")

# ----------------------------- 3.3 训练 GBDT，故意不早停 -----------------------------
print("\n--- 3.3 训练 GBDT（n_estimators=300，故意不早停，观察过拟合）---")

N_ESTIMATORS = 300
model = GradientBoostingClassifier(
    loss="log_loss",        # 对数损失，分类任务的默认选择
    n_estimators=N_ESTIMATORS,
    learning_rate=0.1,
    max_depth=3,
    min_samples_leaf=1,
    subsample=0.8,          # < 1.0 → 随机梯度提升，每棵树只用 80% 样本
    max_features=None,      # 每次分裂用全部特征
    random_state=RANDOM_STATE,
)

t0 = time.perf_counter()
model.fit(X_train, y_train)
train_seconds = time.perf_counter() - t0

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"训练耗时：{train_seconds:.3f} 秒")
print(f"测试集准确率（300 棵树全用上）：{acc:.4f}")
print(f"测试集 AUC（300 棵树全用上）：{auc:.4f}")

# ----------------------------- 3.4 还原每一轮的提升过程 -----------------------------
print("\n--- 3.4 用 staged_predict_proba 还原每一轮的提升过程 ---")

# 训练集偏差曲线：sklearn 已经帮我们记好了，train_score_[i] 是第 i 轮之后的训练集损失。
train_deviance = np.asarray(model.train_score_)

# 测试集损失：staged_predict_proba 会依次吐出"只用了前 k 棵树"的预测概率。
# 这里在循环里逐个 stage 算 log loss，就得到测试集损失随轮数变化的曲线。
stage_list, test_loss_list, test_acc_list = [], [], []
for k, proba_stage in enumerate(model.staged_predict_proba(X_test), start=1):
    stage_list.append(k)
    test_loss_list.append(log_loss(y_test, proba_stage, labels=[0, 1]))
    test_acc_list.append(accuracy_score(y_test, np.argmax(proba_stage, axis=1)))

stage_arr = np.asarray(stage_list)
test_loss_arr = np.asarray(test_loss_list)
test_acc_arr = np.asarray(test_acc_list)

best_acc_idx = int(np.argmax(test_acc_arr))
best_loss_idx = int(np.argmin(test_loss_arr))
print(f"训练集损失 train_score_：第 1 轮 {train_deviance[0]:.4f} → "
      f"第 {N_ESTIMATORS} 轮 {train_deviance[-1]:.4f}（一路下降，说明模型在拼命记训练集）")
print(f"测试集损失：最低点出现在第 {best_loss_idx + 1} 轮，值为 {test_loss_arr[best_loss_idx]:.4f}；"
      f"第 {N_ESTIMATORS} 轮已升到 {test_loss_arr[-1]:.4f}")
print(f"测试集准确率：最高点出现在第 {best_acc_idx + 1} 轮，值为 {test_acc_arr[best_acc_idx]:.4f}；"
      f"第 {N_ESTIMATORS} 轮为 {test_acc_arr[-1]:.4f}")

# ----------------------------- 3.5 演示早停 -----------------------------
print("\n--- 3.5 加上内部验证集早停（validation_fraction + n_iter_no_change）---")

model_es = GradientBoostingClassifier(
    n_estimators=1000,          # 允许训练到 1000 棵
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    validation_fraction=0.1,    # 从训练集里切 10% 当内部验证集
    n_iter_no_change=20,        # 连续 20 轮没进步就停
    tol=1e-4,
    random_state=RANDOM_STATE,
)
t0 = time.perf_counter()
model_es.fit(X_train, y_train)
es_seconds = time.perf_counter() - t0
acc_es = accuracy_score(y_test, model_es.predict(X_test))
auc_es = roc_auc_score(y_test, model_es.predict_proba(X_test)[:, 1])

print(f"早停后实际训练了 {model_es.n_estimators_} 棵树（上限是 1000），耗时 {es_seconds:.3f} 秒")
print(f"早停模型测试集准确率：{acc_es:.4f}，AUC：{auc_es:.4f}")
print(f"对比：跑满 {N_ESTIMATORS} 棵树是 准确率 {acc:.4f} / AUC {auc:.4f}")
print("结论：早停只用了一小部分树就拿到基本等价的测试表现，这就是'不要盲目堆轮数'的意义。")

# ----------------------------- 3.6 特征重要性 -----------------------------
print("\n--- 3.6 特征重要性（这个模型主要靠哪些特征做判断）---")

importances = model.feature_importances_
order = np.argsort(importances)[::-1]
print("特征重要性 Top 8：")
for rank, idx in enumerate(order[:8], start=1):
    print(f"  第{rank:>2}名  {feature_names[idx]:<8} 重要性 = {importances[idx]:.4f}")

# =============================================================================
# 绘图（绝不 plt.show()，savefig 之后立刻 plt.close(fig)）
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

axes[0].plot(stage_arr, train_deviance, label="训练集 log loss（train_score_）", color="#1f77b4", lw=1.8)
axes[0].plot(stage_arr, test_loss_arr, label="测试集 log loss（staged_predict_proba）", color="#d62728", lw=1.8)
axes[0].axvline(best_loss_idx + 1, color="gray", ls="--", lw=1.2,
                label=f"测试损失最低点 = 第 {best_loss_idx + 1} 轮")
axes[0].set_xlabel("Boosting 轮数（累加的树的数量）")
axes[0].set_ylabel("对数损失 log loss（越小越好）")
axes[0].set_title("GBDT 提升过程：训练损失一路降，测试损失先降后升")
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

axes[1].plot(stage_arr, test_acc_arr * 100, color="#2ca02c", lw=1.8, label="测试集准确率(%)")
axes[1].axvline(best_acc_idx + 1, color="gray", ls="--", lw=1.2,
                label=f"测试准确率最高点 = 第 {best_acc_idx + 1} 轮")
axes[1].axhline(test_acc_arr[best_acc_idx] * 100, color="gray", ls=":", lw=1.0)
axes[1].set_xlabel("Boosting 轮数（累加的树的数量）")
axes[1].set_ylabel("测试集准确率（%）")
axes[1].set_title("轮数增加 → 先提升后过拟合")
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

fig.suptitle("01 GBDT 提升过程（make_classification 1000 样本 / 10% 标签噪声 / learning_rate=0.1 / max_depth=3）",
             fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUTPUT_DIR / "04_GBDT_提升过程.png", dpi=130)
plt.close(fig)

fig2, ax2 = plt.subplots(figsize=(8.0, 5.6))
top_n = 12
top_idx = order[:top_n][::-1]
ax2.barh([feature_names[i] for i in top_idx], importances[top_idx], color="#4c72b0")
ax2.set_xlabel("特征重要性（分裂带来的不纯度下降占比）")
ax2.set_title("01 GBDT 特征重要性 Top 12")
ax2.grid(axis="x", alpha=0.3)
fig2.tight_layout()
fig2.savefig(OUTPUT_DIR / "04_GBDT_特征重要性.png", dpi=130)
plt.close(fig2)

# =============================================================================
# ④ 结果解读
# =============================================================================
section("④ 结果解读：这些数字到底在说什么")

print(
    f"""
1) iris 小数据集上准确率 {accuracy_score(y_te_i, y_pred_i):.4f}
   —— 150 个样本、4 个特征、3 个类别，GBDT 用 100 棵浅树几乎能完全分开。
   注意这里训练/测试都接近 1.0，说明 iris 太简单，看不出过拟合，不能用来判断模型好坏。

2) 带噪声数据集上，训练损失从 {train_deviance[0]:.4f} 一路降到 {train_deviance[-1]:.4f}
   —— 训练集损失（偏差）单调下降，这是 boosting 的"本性"：
   每加一棵树都是在当前模型上做一次损失下降，所以训练集只会越来越好。

3) 但测试损失在 {best_loss_idx + 1} 轮见底（{test_loss_arr[best_loss_idx]:.4f}），之后回升到 {test_loss_arr[-1]:.4f}
   —— 这就是过拟合：多出来的那些树不是在学"规律"，而是在背那 10% 的随机标签噪声。
   方差项被这些噪声树放大了，所以测试表现变差。

4) 测试准确率最高 {test_acc_arr[best_acc_idx]:.4f}（第 {best_acc_idx + 1} 轮），
   跑满 {N_ESTIMATORS} 轮后是 {test_acc_arr[-1]:.4f}
   —— 准确率是有上限的（受 10% 标签噪声限制，理论上限约 0.9），
   所以"再多训练几轮"不可能继续涨，只会在最优值附近抖动。

5) 早停模型只用 {model_es.n_estimators_} 棵树就达到准确率 {acc_es:.4f} / AUC {auc_es:.4f}，
   而跑满 {N_ESTIMATORS} 棵是准确率 {acc:.4f} / AUC {auc:.4f}
   —— 树的数量省下 {(1 - model_es.n_estimators_ / N_ESTIMATORS) * 100:.0f}%，测试表现基本持平。
   工程上"能早停就早停"：更少的树 = 更小的模型 + 更快的预测 + 更低的过拟合风险。

6) 特征重要性前三名：{feature_names[order[0]]} ({importances[order[0]]:.4f})、
   {feature_names[order[1]]} ({importances[order[1]]:.4f})、{feature_names[order[2]]} ({importances[order[2]]:.4f})
   —— 我们造数据时只有 10 个 informative 特征，重要性高的基本落在这批"真有用"的特征上，
   说明模型确实抓到了信号，而不是纯靠噪声。
"""
)

# =============================================================================
# 超参数怎么调
# =============================================================================
section("超参数怎么调（GBDT / GradientBoostingClassifier）")

print(
    """
【联动关系，先记住这一条】
    n_estimators ↑ 与 learning_rate ↓ 是一对：learning_rate 从 0.1 降到 0.05，
    树的数量基本要翻倍才能达到同样的拟合程度。经验起点 learning_rate=0.05~0.1，
    n_estimators 先用早停（n_iter_no_change=20 + validation_fraction=0.1）自动定。

【控制模型复杂度（抗过拟合的主角）】
    max_depth            最有效的旋钮。默认 3。数据规律复杂可到 5~8；
                         只要发现训练/测试差距大，第一件事就是把它调小。
    min_samples_leaf     默认 1。样本量不大时设 5~50，避免长出不稳定的小叶子。
    max_features         默认 None。高维稀疏数据可设 'sqrt' 或 0.5~0.8。

【控制随机性（降方差、提速）】
    subsample            默认 1.0。设 0.7~0.9 即随机梯度提升，通常又准又快。
                         注意 subsample < 1 时训练损失曲线会带噪声，属正常现象。

【早停三件套】
    validation_fraction（默认 0.1）+ n_iter_no_change（如 20）+ tol（默认 1e-4）：
    训练集够大（>1 万）时几乎必开，能省掉大量无效的树。

【调参顺序建议】
    第 1 步  learning_rate=0.1 + 早停，先确定 n_estimators 的量级；
    第 2 步  调 max_depth（3 → 5 → 8）和 min_samples_leaf；
    第 3 步  加 subsample=0.8、max_features=0.8 压过拟合；
    第 4 步  learning_rate 减半、n_estimators 翻倍，通常还能再涨一点，但训练时间变长。
"""
)

print(f"\n生成的图片：\n  {OUTPUT_DIR / '04_GBDT_提升过程.png'}\n  {OUTPUT_DIR / '04_GBDT_特征重要性.png'}")
print("\n【完成】01_GBDT.py 运行结束")

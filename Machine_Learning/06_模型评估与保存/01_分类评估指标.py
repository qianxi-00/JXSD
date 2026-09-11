r"""《机器学习》课案 —— 06 模型评估与保存 / 01 分类评估指标

对应课案章节
------------
《机器学习》课案 "模型评估与保存" 章 → "评估指标" 节 → "分类评估指标" 小节
（课案原文第 1282~1348 行；本章 5 个代码块中的第 1 个：
 make_classification 造不平衡二分类数据 + RandomForestClassifier +
 accuracy_score / precision_score / recall_score / f1_score / confusion_matrix）。
课案原文的公式在文本抽取后已损坏，本文件按含义用正确的 LaTeX/文字重写。

本节知识点
----------
1. 混淆矩阵 TP / FP / FN / TN 的定义，以及 sklearn 的排列约定（行=真实标签，列=预测标签）；
2. 准确率 Accuracy、精确率 Precision、召回率 Recall、F1（精确率与召回率的调和平均）的数学推导；
3. support 的含义，macro / weighted / micro / binary 四种 average 聚合方式的区别与适用场景；
4. 类别不平衡时"85% 的准确率"为什么可能毫无意义；
5. 精确率与召回率的业务取舍（垃圾邮件过滤 vs 癌症筛查）；
6. ROC 曲线与 AUC 的几何/概率含义，分类阈值扫动带来的 Precision-Recall 权衡。

运行方式（在 PowerShell 中复制执行）
------------------------------------
$env:PYTHONUTF8='1'; & 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Machine_Learning\06_模型评估与保存\01_分类评估指标.py'

产出
----
Machine_Learning/output/06_混淆矩阵.png
Machine_Learning/output/06_roc曲线.png
Machine_Learning/output/06_阈值权衡曲线.png
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

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

# =============================================================================
# ① 原理与数学推导
# =============================================================================
# 1.1 混淆矩阵（Confusion Matrix）
# --------------------------------
# 二分类问题把每个样本按"真实标签 × 预测标签"放进 2×2 的表格。记：
#   TP (True  Positive)：真实为正、预测也为正 —— 判对了的"正类"
#   TN (True  Negative)：真实为负、预测也为负 —— 判对了的"负类"
#   FP (False Positive)：真实为负、却被预测为正 —— 误报 / 假警报（第一类错误）
#   FN (False Negative)：真实为正、却被预测为负 —— 漏报 / 漏检（第二类错误）
#
# **sklearn 的排列约定**（confusion_matrix 的返回值）：
#       C[i, j] = 真实标签为 i、预测标签为 j 的样本数
#   即 **行 = 真实标签（true label），列 = 预测标签（predicted label）**，行列都按
#   标签值升序排列。二分类时：
#                       预测为 0        预测为 1
#         真实为 0   [   TN (=C00)      FP (=C01) ]
#         真实为 1   [   FN (=C10)      TP (=C11) ]
#   这是最容易记反的地方：**左上角是 TN，右下角是 TP**，正类（标签 1）在右下角。
#
# 由混淆矩阵可以派生出一整套指标（设 P = TP + FN 为真实正类总数，
# N = TN + FP 为真实负类总数）：
#
#   准确率 Accuracy = (TP + TN) / (TP + TN + FP + FN) = 判对样本数 / 总样本数
#     优点：直观、单个数就能概括整体表现；
#     缺点：**对类别不平衡极不敏感**。若正类只占 1.5%，把所有样本都判成负类
#           也能拿到 98.5% 的准确率，但模型毫无用处（召回率 = 0）。
#
#   精确率 Precision = TP / (TP + FP)   —— "预测为正的样本里，有多少是真正"
#     又叫查准率。分母是模型"报出来的正类"总数。Precision 低 = 误报多。
#
#   召回率 Recall = TP / (TP + FN)      —— "真实正类里，有多少被找了出来"
#     又叫查全率 / 灵敏度 Sensitivity / TPR。Recall 低 = 漏报多。
#
#   F1 分数 = 2 * Precision * Recall / (Precision + Recall) = 2TP / (2TP + FP + FN)
#     即精确率与召回率的**调和平均**（harmonic mean）。调和平均对小的那一项
#     惩罚更重：只要有一项接近 0，F1 就接近 0。所以课案说 F1 是
#     "精确率(质量)和召回率(覆盖度)的几何妥协，只在两者都高的时候才高"。
#     为什么用调和平均而不是算术平均？因为算术平均会让"用 Precision=1.0、
#     Recall=0.0"这种极端配置拿到 0.5 的"及格分"，掩盖了模型完全没召回的事实。
#     更一般地，F_beta = (1+β²)·P·R / (β²·P + R)，β>1 偏向召回，β<1 偏向精确。
#
# 1.2 support 与多种 average
# --------------------------
#   support = 该类别在**真实标签**中出现的样本数（即 P 或 N）。它不参与打分，
#   只是告诉你"这一类的成绩是基于多少个样本算出来的"——样本只有 3 个的类别，
#   其精确率波动会非常大，不能与 300 个样本的类别同等看待。
#
# 多分类（或二分类也可以）把每类指标汇总成一个数，有几种平均方式：
#   * macro   ：先对**每个类别**各算一个指标，再取**算术平均**（各类权重相同）。
#               只要有一个小类表现差，macro 就会明显下滑 → 关心"小众类别也不能差"时用它。
#   * weighted：按各类 support 加权的平均（support 大的类说话更响），
#               等价于"以样本为单位"的视角 → 类别不平衡又想让多数类主导时用它。
#   * micro   ：先把所有类别的 TP/FP/FN **全局累加**，再用累加后的计数算一次指标。
#               在单标签多分类（每个样本只属于一个类）中，micro-Precision =
#               micro-Recall = micro-F1 = Accuracy，四者恒等——这是个常见面试题。
#   * binary  ：只报告"正类"（pos_label 指定，默认 1）的那一套指标，二分类最常用。
#   * samples ：多标签分类专用（一个样本可同时属于多个标签），此处不涉及。
#
# 1.3 ROC 曲线与 AUC
# ------------------
# 分类器输出的不是硬标签，而是"正类概率" score。取不同阈值 t（score ≥ t 判为正类）
# 会得到不同的混淆矩阵，于是得到一系列点：
#   TPR（真正率，= Recall） = TP / (TP + FN)     纵轴
#   FPR（假正率）           = FP / (FP + TN) = 1 - 特异度   横轴
# 把所有阈值对应的 (FPR, TPR) 连起来就是 **ROC 曲线**。它从 (0,0) 走到 (1,1)：
#   * 完美分类器经过左上角 (0,1)；
#   * 随机猜测落在对角线 y = x 上。
# AUC（曲线下面积）的**概率含义**：随机抽一个正样本和一个负样本，分类器给正样本
# 更高分数的概率就等于 AUC。AUC = 1 完美，0.5 等于瞎猜，< 0.5 说明分数方向反了。
# AUC 的优点：与阈值无关、对类别不平衡相对稳健，因此在不平衡数据上比准确率可信得多。
#
# 1.4 阈值权衡（本次可选加分图）
# ------------------------------
# 默认阈值 0.5 只是约定，不是最优。阈值升高 → 模型更"谨慎"报正类 → Precision 升、
# Recall 降；阈值降低 → 更大胆报正类 → Recall 升、Precision 降。业务上该取哪个点，
# 取决于 FP 与 FN 的代价比（见 ④ 结果解读）。

# =============================================================================
# ② sklearn API 关键参数逐个解释
# =============================================================================
# 【make_classification】造"人工可控"的分类数据（无需下载数据集，天然离线可跑）
#   n_samples=1000     : 样本总数。太小则指标方差大，太大则跑得慢；教学常用 500~5000。
#   n_features=20      : 特征总数（默认含 2 个冗余特征）。
#   n_informative       : 真正携带类别信息的特征数（默认 2），其余是噪声/冗余特征，
#                         用来模拟"有 18 个特征其实没用"的真实场景。
#   n_redundant         : 由 informative 特征线性组合出来的冗余特征数。
#   weights=[0.85,0.15]: **各类样本的比例**。这里刻意造成 85% : 15% 的类别不平衡，
#                         这正是"准确率会骗人"的实验土壤。注意 weights 会与
#                         n_clusters_per_class 等参数相互作用，实际比例以打印的分布为准。
#   flip_y=0.01         : 随机把 1% 的样本标签翻错，模拟标注噪声（默认 0.01）。
#   class_sep=1.0       : 类间可分程度，越小越难分（默认 1.0）。
#   random_state=42     : 固定随机种子，保证每次运行结果完全一致（可复现是硬要求）。
#
# 【train_test_split】
#   test_size=0.3       : 30% 作测试集。测试集只能用来"最后看一眼"，不能反复调参。
#   stratify=y          : **按 y 分层抽样**，保证训练/测试集中正类比例与原始数据一致。
#                         不平衡数据**必须**加这一项，否则可能出现测试集里正类只有个位数。
#   random_state=42     : 固定划分，便于结果复现与对比。
#   shuffle=True        : 默认打乱后再切分（时序数据要设 False 并改用 TimeSeriesSplit）。
#
# 【RandomForestClassifier】
#   n_estimators=100    : 树的数量。越多越稳但越慢；常用 100~500，教学 50~100 够用。
#   max_depth=None      : 单棵树最大深度，None=不限深（树会一直长到叶子纯净），
#                         容易过拟合；常用 3~20 或 None 交给 min_samples_leaf 约束。
#   max_features='sqrt' : 每次分裂随机考察的特征数，默认 sqrt(n_features)。
#                         它是随机森林"去相关"的关键，越小树之间越独立、方差越低。
#   class_weight=None   : 可传 "balanced" 让少数类样本获得更大权重，
#                         在不平衡数据上常能显著提升 Recall（本脚本保持默认以便先看到问题）。
#   bootstrap=True      : 有放回抽样构造每棵树的训练子集（Bagging 的核心）。
#   oob_score=False     : 是否用袋外样本估计泛化误差（开启可省一次验证，但训练更慢）。
#   n_jobs=None         : 并行核数，-1 表示用满所有核心；小数据设 1 反而更稳（避免线程开销）。
#   random_state=42     : 固定随机种子；不固定则每次跑出的指标都会抖动。
#
# 【指标函数】共同的关键参数
#   y_true, y_pred      : 真实标签、预测标签（顺序必须一一对应）。
#   average             : 'binary'(默认,二分类) | 'macro' | 'weighted' | 'micro' | 'samples'。
#                         多分类不显式指定会报错，务必写清楚。
#   pos_label=1         : 谁是"正类"。二分类默认 1；若标签是字符串要显式指定。
#   zero_division=0     : 分母为 0（例如某类一个都没预测出来）时返回 0 而不是 nan+警告。
#                         显式设置可以彻底避免 RuntimeWarning 刷屏。
#   labels / normalize  : confusion_matrix 用；normalize='true' 会按行归一化（看出召回率），
#                         'pred' 按列归一化（看出精确率），'all' 按总数归一化。
#   accuracy_score(normalize=True) : False 时返回"判对个数"。
#   classification_report(target_names=[...], digits=4, zero_division=0)
#                       : 一次性输出每类的 precision/recall/f1/support + 汇总，
#                         target_names 让报告里的 0/1 变成可读的中文类别名。
#   roc_auc_score(y_true, y_score) : 第二个参数必须是**连续分数/概率**而非硬标签；
#                         二分类写 predict_proba(X)[:, 1]；多分类用 average='macro' 等。
#
# 【绘图 API】
#   ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[...]).plot(ax=..., cmap=...)
#                       : 直接把混淆矩阵画成带数字标注的热力图；display_labels 支持中文。
#   RocCurveDisplay.from_estimator(estimator, X, y, name=..., ax=...)
#                       : 自动调用 predict_proba 遍历阈值画 ROC，并把 AUC 写进图例；
#                         比手写 roc_curve 更省事（sklearn 1.9 中该 API 稳定可用）。
#   ax.plot([0,1],[0,1]) : 画 y=x 参考线，用于直观判断"是否比瞎猜更好"。

# =============================================================================
# ③ 完整可运行代码
# =============================================================================
# 输出目录：与所有子目录共享 Machine_Learning/output（以 __file__ 为基准，避免依赖 cwd）
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42  # 全局统一随机种子

print("=" * 78)
print("06-01 分类评估指标：不平衡二分类上的 Accuracy / Precision / Recall / F1 / ROC-AUC")
print("=" * 78)

# --- 步骤 1：生成不平衡二分类数据（对应课案代码块 1） --------------------------
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    weights=[0.85, 0.15],   # 负类:正类 ≈ 85%:15%，制造类别不平衡
    random_state=RANDOM_STATE,
)

n_pos = int((y == 1).sum())
n_neg = int((y == 0).sum())
print("\n[1] 数据概况")
print(f"    特征矩阵形状 X = {X.shape}（1000 个样本 × 20 个特征）")
print(f"    类别分布：负类(0) {n_neg} 个（{n_neg / len(y):.2%}），"
      f"正类(1) {n_pos} 个（{n_pos / len(y):.2%}）")
print(f"    → 正类是少数类，占 {n_pos / len(y):.1%}。这正是「准确率会骗人」的场景。")

# --- 步骤 2：分层划分训练集 / 测试集 ------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=RANDOM_STATE,
    stratify=y,   # 关键：保持训练/测试集的正类比例一致
)
print("\n[2] 数据集划分（stratify=y 保持类别比例）")
print(f"    训练集 {X_train.shape[0]} 个样本，其中正类 {int((y_train == 1).sum())} 个"
      f"（{(y_train == 1).mean():.2%}）")
print(f"    测试集 {X_test.shape[0]} 个样本，其中正类 {int((y_test == 1).sum())} 个"
      f"（{(y_test == 1).mean():.2%}）")

# --- 步骤 3：训练随机森林 -----------------------------------------------------
model = RandomForestClassifier(random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# --- 步骤 4：预测（硬标签 + 正类概率） ----------------------------------------
y_pred = model.predict(X_test)                     # 阈值 0.5 下的硬标签
y_proba = model.predict_proba(X_test)[:, 1]        # 正类概率，ROC/AUC 必须用它

# --- 步骤 5：逐个指标计算（课案代码块 1 的核心） ------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
auc = roc_auc_score(y_test, y_proba)

tn, fp, fn, tp = cm.ravel()   # 提醒：ravel() 顺序固定为 (TN, FP, FN, TP)

print("\n[3] 混淆矩阵（sklearn 约定：行=真实标签，列=预测标签；标签顺序 [0, 1]）")
print("                     预测为 0(负类)   预测为 1(正类)")
print(f"    真实为 0(负类)   TN = {tn:>8d}      FP = {fp:>8d}")
print(f"    真实为 1(正类)   FN = {fn:>8d}      TP = {tp:>8d}")
print("    记忆口诀：对角线上（TN、TP）是判对的，反对角线上（FP、FN）是判错的；")
print("              FP 是「误报/虚警」，FN 是「漏报/漏检」。")
print(f"    校验：{tn}+{fp}+{fn}+{tp} = {tn + fp + fn + tp} = 测试集样本数 {len(y_test)}")

print("\n[4] 五个核心指标（阈值 = 0.5）")
print(f"    准确率 Accuracy  = (TP+TN)/总数 = ({tp}+{tn})/{tn + fp + fn + tp}"
      f" = {accuracy:.4f}  ({accuracy:.2%})")
print(f"    精确率 Precision = TP/(TP+FP)   = {tp}/({tp}+{fp}) = {precision:.4f}"
      f"  —— 模型报出的正类里有 {precision:.2%} 是真的")
print(f"    召回率 Recall    = TP/(TP+FN)   = {tp}/({tp}+{fn}) = {recall:.4f}"
      f"  —— 真实正类里有 {recall:.2%} 被找了出来")
print(f"    F1 分数          = 2PR/(P+R)    = 2*{precision:.4f}*{recall:.4f}"
      f"/({precision:.4f}+{recall:.4f}) = {f1:.4f}")
print(f"    ROC-AUC          = {auc:.4f}")
print(f"    手算校验 F1 = 2TP/(2TP+FP+FN) = 2*{tp}/(2*{tp}+{fp}+{fn})"
      f" = {2 * tp / (2 * tp + fp + fn):.4f}")

# --- 步骤 6：classification_report（含 support 与多种 average） ---------------
target_names = ["负类（多数类 0）", "正类（少数类 1）"]
report_text = classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=target_names,
    digits=4,
    zero_division=0,
)
print("\n[5] classification_report（逐类指标 + support + 三种 average 汇总）")
for line in report_text.rstrip().splitlines():
    print("    " + line)

# 用 output_dict=True 拿到结构化结果，做 macro / weighted / micro 的横向对比
report_dict = classification_report(
    y_test, y_pred, labels=[0, 1], target_names=target_names, output_dict=True, zero_division=0
)
macro_p = precision_score(y_test, y_pred, average="macro", zero_division=0)
macro_r = recall_score(y_test, y_pred, average="macro", zero_division=0)
macro_f = f1_score(y_test, y_pred, average="macro", zero_division=0)
weighted_p = precision_score(y_test, y_pred, average="weighted", zero_division=0)
weighted_r = recall_score(y_test, y_pred, average="weighted", zero_division=0)
weighted_f = f1_score(y_test, y_pred, average="weighted", zero_division=0)
micro_p = precision_score(y_test, y_pred, average="micro", zero_division=0)
micro_r = recall_score(y_test, y_pred, average="micro", zero_division=0)
micro_f = f1_score(y_test, y_pred, average="micro", zero_division=0)

compare_table = pd.DataFrame(
    {
        "average 方式": ["binary（只看正类）", "macro（各类算术平均）", "weighted（按 support 加权）", "micro（全局累加）"],
        "Precision": [precision, macro_p, weighted_p, micro_p],
        "Recall": [recall, macro_r, weighted_r, micro_r],
        "F1": [f1, macro_f, weighted_f, micro_f],
    }
)
compare_table[["Precision", "Recall", "F1"]] = compare_table[["Precision", "Recall", "F1"]].round(4)
print("\n[6] 同一份预测结果，四种 average 的对比")
print(compare_table.to_string(index=False))
print(f"    support：负类 {int(report_dict[target_names[0]]['support'])} 个，"
      f"正类 {int(report_dict[target_names[1]]['support'])} 个"
      f"（= 测试集里两类真实样本数，只表示「这个成绩基于多少样本」，不参与打分）")
print(f"    观察 1：macro 的 Precision({macro_p:.4f}) 明显低于 weighted({weighted_p:.4f})，")
print("            因为 macro 把只占少数样本的正类与占多数样本的负类同等对待，")
print("            正类较差的成绩被「平均」上去，暴露了小类问题。")
print(f"    观察 2：micro 的 P=R=F1=Accuracy={micro_f:.4f}，四者完全相等——")
print("            单标签分类中 micro 把所有类的 TP/FP/FN 全局累加，")
print("            化简后分子是判对总数、分母是总样本数，即准确率，这是恒等关系。")

# --- 步骤 7：与"全预测为多数类"的无脑基线对比 --------------------------------
base_pred = np.zeros_like(y_test)          # 全判负类
base_acc = accuracy_score(y_test, base_pred)
base_recall = recall_score(y_test, base_pred, pos_label=1, zero_division=0)
print("\n[7] 关键对照实验：一个「什么都不学、全部预测负类」的模型")
print(f"    它的准确率 Accuracy = {base_acc:.4f}（{base_acc:.2%}）")
print(f"    它的召回率 Recall   = {base_recall:.4f}（正类一个都没找出来）")
print(f"    它其实什么都没做，却拿到了 {base_acc:.1%} 的准确率；")
print(f"    而真正训练的随机森林准确率 {accuracy:.4f}，只比这个「废模型」高"
      f" {(accuracy - base_acc) * 100:.2f} 个百分点。")
print("    → 所以在不平衡数据上，**准确率高不代表模型有用**：")

# --- 步骤 8：阈值扫动，观察 Precision / Recall 的权衡 -------------------------
prec_curve, rec_curve, thresholds = precision_recall_curve(y_test, y_proba)
# precision_recall_curve 返回的 precision/recall 长度比 thresholds 多 1（最后一个点 recall=0）
f1_curve = 2 * prec_curve * rec_curve / np.clip(prec_curve + rec_curve, 1e-12, None)
best_idx = int(np.argmax(f1_curve[:-1]))          # 只在有阈值对应的点上找最优
best_threshold = float(thresholds[best_idx])
print("\n[8] 阈值不是必须取 0.5：扫动阈值看 Precision / Recall 如何互换")
print("    阈值      精确率    召回率    F1")
for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    p_t = precision_score(y_test, (y_proba >= t).astype(int), zero_division=0)
    r_t = recall_score(y_test, (y_proba >= t).astype(int), zero_division=0)
    f_t = f1_score(y_test, (y_proba >= t).astype(int), zero_division=0)
    print(f"    {t:>4.1f}      {p_t:>7.4f}   {r_t:>7.4f}   {f_t:.4f}")
print(f"    F1 最大的阈值 ≈ {best_threshold:.4f}，此时 Precision={prec_curve[best_idx]:.4f}、"
      f"Recall={rec_curve[best_idx]:.4f}、F1={f1_curve[best_idx]:.4f}")
print("    规律：阈值↑ → 模型更保守，报出的正类更「可信」（Precision↑）但漏得更多（Recall↓）；")
print("          阈值↓ → 模型更激进，几乎不漏（Recall↑）但误报变多（Precision↓）。")

# =============================================================================
# ④ 结果解读
# =============================================================================
print("\n" + "=" * 78)
print("④ 结果解读：这些数字到底说明什么")
print("=" * 78)
print(f"""
【1】准确率 {accuracy:.2%} 看着挺高，但它在这里几乎说明不了问题。
      原因：正类只占 {n_pos / len(y):.1%}。上面那个"全部预测负类"的废模型都能拿到
      {base_acc:.2%} 的准确率；只要把阈值抬到很高、几乎不报正类，准确率同样能冲到
      90% 以上。准确率的隐含假设是"两类同样重要、误报漏报代价相同"，
      而这个假设在不平衡数据上根本不成立。
      → 结论：**不平衡数据上应优先看 Recall / F1 / AUC / PR 曲线，而不是 Accuracy。**

【2】本次精确率 {precision:.4f}、召回率 {recall:.4f}，精确率高于召回率，
      说明模型偏保守（F1={f1:.4f}，介于两者之间且更靠近较小的一方）：
      它报出来的正类里 {precision:.1%} 是真的（误报率 {1 - precision:.1%}），
      但真实正类里只找回了 {recall:.1%}，还有 {(1 - recall) * (tp + fn):.0f} 个正类样本被漏掉（FN={fn}）。
      参照前面的阈值表可以看到，把阈值从 0.5 降到 0.3 左右，召回率会升到 0.80 附近
      而精确率只掉到 0.77 左右，F1 反而更高（≈0.79）——
      这说明 0.5 这个默认阈值对本数据来说偏高了。
      两类指标的一般规律：
      * Precision 高而 Recall 很低（例如 0.95 / 0.30）→ 模型过于保守，漏掉大量正类；
      * Recall 高而 Precision 很低（例如 0.30 / 0.95）→ 模型「宁可错杀一千」，误报满天飞。

【3】精确率和召回率谁更重要，**完全取决于业务，取决于 FP 与 FN 哪个代价更大**：
      * 垃圾邮件过滤：把正常邮件误判成垃圾邮件（FP）会让用户丢掉重要工作邮件，
        代价极高；而漏掉一封垃圾邮件（FN）只是多看一眼而已。
        → 这时 **Precision 优先**，宁可漏掉一些垃圾邮件，也不能误杀正常邮件。
      * 癌症筛查 / 地震预警 / 金融欺诈初筛：漏掉一个真正的病人（FN）可能致命，
        而把健康人叫回来复查（FP）只是多花点钱、多受一次惊吓。
        → 这时 **Recall 优先**，甚至愿意用很低的 Precision 去换高 Recall。
      * 一般来说：**FN 代价大就提召回，FP 代价大就提精确**；
        两者都重要、又说不清谁更重时，用 F1 作为折中。
        真正工程化的做法是给 FP/FN 赋具体金额或生命损失，算期望代价最小的阈值。

【4】ROC-AUC = {auc:.4f}：它衡量的是"模型把正类排在负类前面的能力"，
      与阈值无关。{auc:.4f} 说明随机抽一个正样本和一个负样本，
      模型给正样本更高分的概率约为 {auc:.1%}（0.5 才是瞎猜水平）。
      在本例这种"特征多、类别可分"的合成数据上，RF 的 AUC 通常接近 1，说明
      "排序能力"很好——问题不在排序，而在于你用 0.5 当阈值、又没有为少数类做任何补偿。
      AUC 高但 F1 一般，正是"模型排序没问题、决策阈值不合适"的典型信号。

【5】实用的改进方向（本脚本未展开，仅记录）：
      * class_weight="balanced" 或对少数类过采样（SMOTE 类方法，需装 imbalanced-learn）；
      * 用 PR 曲线（average precision）代替 ROC——极不平衡时 PR 曲线更敏感；
      * 按业务代价调阈值，而不是死守 0.5；
      * 交叉验证确认指标稳定（见本目录 03_交叉验证.py）。
""")

# --- 绘图 1：混淆矩阵 ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 5.4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
ax.set_title(f"06 随机森林混淆矩阵（不平衡二分类，测试集 {len(y_test)} 个样本）",
             fontsize=13, pad=12)
ax.set_xlabel("模型预测的类别", fontsize=11)
ax.set_ylabel("样本真实的类别", fontsize=11)
# 在每个格子里补充 TP/FP/FN/TN 的业务含义，便于一眼读懂
annot_map = {
    (0, 0): f"TN={tn}\n真负类，判对",
    (0, 1): f"FP={fp}\n误报（虚警）",
    (1, 0): f"FN={fn}\n漏报（漏检）",
    (1, 1): f"TP={tp}\n真正类，判对",
}
for (i, j), text in annot_map.items():
    # 深色格子用白字、浅色格子用深灰字，保证在任何 cmap 下都能看清
    text_color = "white" if cm[i, j] > cm.max() * 0.6 else "#333333"
    ax.text(j, i + 0.30, text, ha="center", va="center", fontsize=9, color=text_color)
fig.tight_layout()
cm_path = OUTPUT_DIR / "06_混淆矩阵.png"
fig.savefig(cm_path, dpi=130)
plt.close(fig)
print(f"[图 1] 已保存混淆矩阵：{cm_path}")

# --- 绘图 2：ROC 曲线 + AUC ---------------------------------------------------
fig, ax = plt.subplots(figsize=(6.6, 5.4))
RocCurveDisplay.from_estimator(
    model,
    X_test,
    y_test,
    name=f"随机森林（AUC = {auc:.4f}）",
    ax=ax,
    pos_label=1,
    curve_kwargs={"color": "#c0392b", "linewidth": 2},
)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.4, label="随机猜测参考线 y = x（AUC = 0.5）")
ax.set_title("06 ROC 曲线与 AUC（不平衡二分类）", fontsize=13, pad=12)
ax.set_xlabel("假正率 FPR = FP / (FP + TN)　→　误报比例", fontsize=11)
ax.set_ylabel("真正率 TPR = TP / (TP + FN)　→　召回率", fontsize=11)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
roc_path = OUTPUT_DIR / "06_roc曲线.png"
fig.savefig(roc_path, dpi=130)
plt.close(fig)
print(f"[图 2] 已保存 ROC 曲线：{roc_path}")

# --- 绘图 3（可选加分）：阈值 - 精确率/召回率 权衡曲线 ------------------------
fig, ax = plt.subplots(figsize=(8.0, 5.2))
ax.plot(thresholds, prec_curve[:-1], label="精确率 Precision", color="#2980b9", linewidth=2)
ax.plot(thresholds, rec_curve[:-1], label="召回率 Recall", color="#27ae60", linewidth=2)
ax.plot(thresholds, f1_curve[:-1], label="F1 分数", color="#c0392b", linewidth=2, linestyle="-.")
ax.axvline(0.5, color="gray", linestyle="--", linewidth=1.3, label="默认阈值 0.5")
ax.axvline(best_threshold, color="#8e44ad", linestyle=":", linewidth=1.8,
           label=f"F1 最优阈值 ≈ {best_threshold:.2f}")
ax.set_title("06 阈值扫动下的 Precision / Recall 权衡（不平衡二分类）", fontsize=13, pad=12)
ax.set_xlabel("判定为正类的概率阈值", fontsize=11)
ax.set_ylabel("指标取值", fontsize=11)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.05)
ax.legend(loc="center right", fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
thr_path = OUTPUT_DIR / "06_阈值权衡曲线.png"
fig.savefig(thr_path, dpi=130)
plt.close(fig)
print(f"[图 3] 已保存阈值权衡曲线：{thr_path}")

# =============================================================================
# 超参数怎么调 / 使用注意
# =============================================================================
# 【本节的"超参数"其实主要是评估口径，而不是模型本身】
# 1. average 怎么选：
#    二分类且只关心正类 → average="binary"（默认），并明确 pos_label；
#    多分类且各类同等重要 → "macro"；多分类但样本极不均衡 → "weighted"；
#    想得到一个和准确率同口径的总分 → "micro"。
# 2. 阈值怎么调：不要迷信 0.5。用 precision_recall_curve 或 GridSearchCV 扫阈值，
#    按业务代价（FN 与 FP 的代价比）选点；本脚本图 3 展示了这一权衡。
# 3. 模型侧常用调参顺序（详见本目录 04_网格搜索超参数调优.py）：
#    先 class_weight（不平衡数据的性价比最高的一招），再 max_depth / min_samples_leaf
#    控复杂度，最后加 n_estimators（边际收益递减，只影响稳定性不影响过拟合）。
# 4. 千万注意：
#    * 选指标、选阈值、选超参都只能用**训练集/验证集**；测试集只能最后用一次，
#      反复用测试集调参 = 数据泄漏，报出来的分数会虚高。
#    * 不平衡数据下不要只看准确率，也不要只看单一指标。
#    * zero_division 显式设成 0，避免某类没被预测到时返回 nan 并刷警告。
#    * 比较不同模型时必须用**同一份**测试集（同一 random_state 与 stratify）。
print("\n【完成】01_分类评估指标.py 运行结束")

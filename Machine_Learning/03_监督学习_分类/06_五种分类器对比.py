r"""《机器学习》课案 · 03 监督学习-分类 · 06 五种分类器对比

对应课案章节
-----------
《机器学习》课案「分类」章的总结与横向对比部分：
课案原文位于 `.course_extract/机器学习_课案.md` 第 263~684 行。
本脚本把该章讲过的五种分类器（逻辑回归 / K-近邻 / 支持向量机 / 决策树 / 随机森林）
放在**同一份数据、同一次划分、同一套评估流程**下横向对比，作为整章的收尾总结。
涉及课案中的：各算法代码示例、逻辑回归参数表、KNN 距离度量表、
SVM 核函数表与参数表、决策树分裂标准表、随机森林（Bagging）原理。

本节知识点
---------
1. **公平比较的三要素**：同一份数据、同一次 train_test_split（同 random_state）、
   同一套预处理与评估指标。任何一项不同，比较就没有意义。
2. **Pipeline 的价值**：把 StandardScaler 和分类器串成一个整体，
   交叉验证时 scaler 只会在每一折的**训练部分** fit_transform，
   测试部分只 transform —— 从根本上杜绝**数据泄漏**（data leakage）。
3. **五种算法的定位**：
   逻辑回归（线性、可解释、给概率）、KNN（懒惰学习、无参数、靠距离）、
   SVM（最大间隔、核技巧、小样本高维强）、决策树（白盒规则、无需标准化）、
   随机森林（Bagging 集成、稳健、几乎不用调参）。
4. **准确率不是唯一指标**：要看任务类型（类别是否平衡）、代价不对称性、
   概率需求、可解释性需求、推理延迟与模型体积。

运行方式
-------
PowerShell（路径含中文，必须加引号并用 & 调用）：

    $env:PYTHONUTF8='1'; & 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Machine_Learning\03_监督学习_分类\06_五种分类器对比.py'

输出：
    - 控制台：四段式讲解 + 对齐的中文对比表（准确率 / 训练耗时 / 5 折 CV）+ 中文选型总结
    - 图片： Machine_Learning/output/03_五种分类器_准确率与CV对比.png
            Machine_Learning/output/03_五种分类器_ROC与混淆矩阵.png
"""

from __future__ import annotations

import pathlib
import time


import matplotlib
matplotlib.use("Agg")            # 无界面后端，避免 plt.show() 阻塞
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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


# ============================================================================
# ① 原理与数学推导（五种分类器的一句话模型 + 损失 + 优化）
# ============================================================================
# 本脚本是整章收尾，这里把五种算法按"模型 → 损失 → 优化"的同一套框架并列，方便对照。
#
# 【1.1 逻辑回归 Logistic Regression】
#   模型：z = w·x + b，p = σ(z) = 1/(1+e^(-z))，用 0.5 阈值判类；
#   损失：交叉熵（负对数似然） L = -(1/N)Σ[ y·log p + (1-y)·log(1-p) ]；
#   优化：梯度下降 / 拟牛顿法（sklearn 默认 lbfgs）；损失是凸的，有全局最优。
#   特点：线性决策边界；输出**校准较好的概率**；系数可直接解释；
#         必须标准化；对非线性关系无能为力。
#
# 【1.2 K-近邻 KNN】
#   模型：没有参数模型！把训练集整份记住（懒惰学习），
#         预测时算待测样本与全部训练样本的距离，取最近 K 个邻居投票；
#   损失：没有显式的训练损失（K=1 时训练误差恒为 0）；
#   优化：没有优化过程，全部计算都发生在预测阶段（复杂度 O(N·d)）。
#   特点：非参数、能拟合任意形状边界；**对量纲和无关特征极其敏感，必须标准化**；
#         预测慢、内存大、遭遇维度灾难。
#
# 【1.3 支持向量机 SVM】
#   模型：找最大间隔超平面；软间隔等价于 hinge 损失 + L2 正则
#         min (1/2)||w||² + C·Σ max(0, 1 - y_i·f(x_i))；
#   损失：hinge 损失（分对了且落在间隔外 → 损失为 0，所以解只由支持向量决定）；
#   优化：拉格朗日对偶 → 凸二次规划；用核函数 K(x,y)=φ(x)·φ(y) 隐式升维；
#   特点：小样本、高维数据上很强；**必须标准化**；对 C/γ 敏感；
#         原生不给概率（要 CalibratedClassifierCV 校准）；
#         大样本训练慢（约 O(N²)~O(N³)）。
#
# 【1.4 决策树 Decision Tree】
#   模型：递归地把特征空间切成矩形，每个叶子给一个类别（一堆 if-else 规则）；
#   损失：不直接最小化某个全局损失，而是**贪心**地每次选让不纯度下降最多的分裂；
#         不纯度可选 Gini = 1-Σp_i²（默认）或 Entropy = -Σp_i·log₂p_i；
#   优化：没有梯度、没有迭代，纯贪心递归划分（CART 生成二叉树）。
#   特点：**不需要标准化**；可解释性最强；天然多分类；
#         但极易过拟合、方差极大（换个种子树就变样）。
#
# 【1.5 随机森林 Random Forest】
#   模型：Bagging —— B 棵决策树，每棵用一份 bootstrap 自助采样集训练，
#         每次分裂只在随机抽出的 max_features 个特征里选最优，最后投票；
#   损失：每个基学习器各自贪心建树（同决策树）；
#   优化：无梯度；B 棵树完全独立、可并行；
#   特点：平均后方差 = ρσ² + (1-ρ)σ²/B —— **降方差、几乎不增偏差**；
#         自带 OOB 袋外估计；不需要标准化；稳健、几乎不用调参；
#         代价是可解释性下降。
#
# 【1.6 五分钟选型口诀】
#   要可解释 / 要概率 → 逻辑回归、决策树；
#   小样本、高维（文本 / 基因）→ SVM（线性核优先）、逻辑回归；
#   中等数据、追求开箱即用的稳健精度 → 随机森林（第一选择）；
#   数据有清晰的距离含义、且已经过良好清洗 → KNN；
#   类别极不平衡 → 别只看准确率，看 F1 / recall / PR-AUC，并配 class_weight。
# ============================================================================


# ============================================================================
# ② 关键 API 与"比较方法论"参数逐个解释
# ============================================================================
# 本脚本的主角是"公平比较"的方法论，所以这里解释的是对比流程里用到的每个环节。
#
# train_test_split(test_size=0.3, random_state=42, stratify=y)
#    - random_state=42：固定划分，保证五个模型看到**完全相同**的训练/测试数据；
#    - stratify=y：分层抽样，保证训练集与测试集里正负类比例一致。
#      乳腺癌数据 569 条里恶性 212 条、良性 357 条（约 37% : 63%），
#      不分层的话测试集里正类比例可能偏离，指标就会抖；
#      分层之后五个模型的评估基础一致，比较才公平。
#    - test_size=0.3：测试集约 171 条，对二分类评估来说样本量够用。
#
# Pipeline([("scaler", StandardScaler()), ("clf", 分类器)])
#    为什么必须用 Pipeline（本脚本最重要的工程知识点）：
#      · 反例（错误做法）：先对**全部数据**做 StandardScaler.fit_transform，
#        再 train_test_split。此时测试集的均值/方差已经"泄漏"进了训练过程，
#        这叫**数据泄漏（data leakage）**，会让评估结果虚高，
#        在真实上线时被打回原形；
#      · 反例二：在交叉验证里手工先标准化再 cross_val_score —— 同样泄漏，
#        因为每一折的"训练部分"标准化时用到了该折"验证部分"的统计量；
#      · 正确做法：把 scaler 和模型装进 Pipeline，交给交叉验证统一调度。
#        每一折内部，scaler 只在训练部分 fit_transform，验证部分只 transform。
#      · 附带好处：部署时只需保存一个 Pipeline 对象，
#        预测新数据时不会忘记做同样的预处理。
#    注意：逻辑回归 / KNN / SVM 需要标准化；决策树 / 随机森林**不需要**，
#          但放进 Pipeline 一起标准化也**不会变差**（树只比较大小关系，
#          单调变换不改变分裂结果）。本脚本为了统一比较，五个模型全部套 Pipeline。
#
# cross_val_score(estimator, X, y, cv=StratifiedKFold(5), scoring="accuracy", n_jobs=-1)
#    - cv=StratifiedKFold(5)：5 折**分层**交叉验证，每折里正负类比例都与整体一致；
#      比普通 KFold 在二分类上更稳（尤其类别不平衡时）；
#    - scoring：本脚本用 "accuracy"，另外也算了 f1；
#    - n_jobs=-1：并行跑各折；
#    - 为什么必须报告**均值 ± 标准差**：均值代表"典型表现"，
#      标准差代表"稳定性"。两个模型均值相同时，标准差小的那个更值得上线
#      （因为它在不同数据划分下表现更一致）。
#
# 各分类器的关键参数（详见各自脚本，这里只列本脚本实际用到的取值与理由）
#    LogisticRegression(C=1.0, max_iter=1000)
#        C=1.0 默认正则强度；max_iter=1000 保证收敛。
#        注意 sklearn 1.9 已弃用 penalty、移除 multi_class，所以不传这两个参数。
#    KNeighborsClassifier(n_neighbors=5)
#        K=5 是默认值，也是二分类常用的起点（奇数可避免平票）。
#    SVC(kernel="rbf", C=1.0, gamma="scale")
#        RBF 是通用首选，'scale' 让 γ 随特征方差自动缩放，最省心。
#    DecisionTreeClassifier(max_depth=3, random_state=42)
#        限制深度是防过拟合最有效的一招；深度 3 得到可读性很好的小树。
#    RandomForestClassifier(n_estimators=300, max_features="sqrt", n_jobs=-1, random_state=42)
#        300 棵足够进入"平台区"；max_features='sqrt' 是分类默认（去相关）；
#        n_jobs=-1 用满所有核并行。
# ============================================================================


print_section("《机器学习》课案 · 分类 · 06 五种分类器对比")
print("本脚本四段结构： ① 原理与数学推导  ② sklearn API 参数解释  ③ 完整可运行代码  ④ 结果解读")
print("配套图片输出目录：", OUTPUT_DIR)

# ============================================================================
# ③ 完整可运行代码
# ============================================================================

# ---------------------------------------------------------------------------
# 步骤 1：加载数据 —— 乳腺癌数据集（二分类，比 iris 更接近真实问题）
# ---------------------------------------------------------------------------
print_section("步骤 1：加载数据（load_breast_cancer，二分类）")

data = load_breast_cancer()
X, y = data.data, data.target
feature_names = list(data.feature_names)
class_names = [str(n) for n in data.target_names]     # ['malignant'（恶性=0）, 'benign'（良性=1）]
print("数据集来源：sklearn.datasets.load_breast_cancer（内置数据集，无需联网下载）")
print("数据集简介：", [ln.strip() for ln in data.DESCR.splitlines() if ln.strip()][0])
print(f"样本形状：X = {X.shape}（{X.shape[0]} 条样本，{X.shape[1]} 个特征）")
print("特征举例：", feature_names[:5], "...")
print("类别名：", class_names, "（0 = malignant 恶性，1 = benign 良性）")
print("各类样本数：", {class_names[i]: int(n) for i, n in enumerate(np.bincount(y))})
print(f"正类（良性）占比 = {np.mean(y):.2%} —— 类别比例约 37% : 63%，不算严重不平衡，")
print("但已经足以说明**必须用 stratify 分层抽样**，否则测试集的正类比例会飘。")

# ---------------------------------------------------------------------------
# 步骤 2：同一次划分（公平比较的前提）
# ---------------------------------------------------------------------------
print_section("步骤 2：同一次 train_test_split（random_state=42, stratify=y）")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)
print(f"训练集 X_train = {X_train.shape}，测试集 X_test = {X_test.shape}")
print(f"训练集正类比例 = {np.mean(y_train):.4f}，测试集正类比例 = {np.mean(y_test):.4f}")
print("两者与整体比例几乎一致 —— 这就是 stratify=y（分层抽样）的作用。")
print("五个分类器将使用**完全相同的这份训练集与测试集**，")
print("这是任何横向对比成立的前提：只要划分变了，比较就没有意义。")

# ---------------------------------------------------------------------------
# 步骤 3：定义五个模型（统一装进 Pipeline）
# ---------------------------------------------------------------------------
print_section("步骤 3：把五个分类器统一装进 Pipeline")

models = [
    ("逻辑回归 LogisticRegression", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_STATE)),
    ])),
    ("K-近邻 KNeighbors(K=5)", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
    ])),
    ("支持向量机 SVC(rbf)", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)),
    ])),
    ("决策树 DecisionTree(d=3)", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE)),
    ])),
    ("随机森林 RandomForest(300)", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                       n_jobs=-1, random_state=RANDOM_STATE)),
    ])),
]
for name, pipe in models:
    clf_name = pipe.named_steps["clf"].__class__.__name__
    print(f"  · {name:<32} → Pipeline 步骤：{list(pipe.named_steps.keys())}，"
          f"分类器 = {clf_name}")
print()
print("为什么每个模型都套上 Pipeline：")
print("  如果先对全量数据 StandardScaler 再划分，测试集的统计量就'泄漏'进了训练，")
print("  这叫**数据泄漏**，会让评估结果虚高、上线后崩盘。")
print("  而 Pipeline 交给 cross_val_score 时，每一折内部 scaler 只在**训练部分** fit，")
print("  验证部分只 transform —— 这是 sklearn 里避免数据泄漏的标准姿势。")
print("  另外：决策树 / 随机森林其实不需要标准化（树只比较大小关系），")
print("  但套上 scaler 也不会让它们变差，为统一流程就都加上了。")

# ---------------------------------------------------------------------------
# 步骤 4：训练、预测、计时、交叉验证
# ---------------------------------------------------------------------------
print_section("步骤 4：逐一训练、评估（含训练耗时与 5 折交叉验证）")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = []
for name, pipe in models:
    # 计时：只统计 fit 的耗时（这是"训练耗时"的合理口径）
    t_start = time.perf_counter()
    pipe.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t_start

    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # 5 折分层交叉验证（在训练集内部做，绝不碰测试集）
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

    # ROC-AUC：需要打分。SVC 默认没有 predict_proba，用 decision_function；
    #          这里统一用 decision_function（五个模型都有），保证口径一致。
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        score = pipe.predict_proba(X_test)[:, 1]
    else:
        score = pipe.decision_function(X_test)
    auc = roc_auc_score(y_test, score)

    results.append({
        "name": name,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "auc": auc,
        "fit_seconds": fit_seconds,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "y_pred": y_pred,
        "score": score,
    })
    print(f"[已完成] {name:<32} 测试准确率 {acc:.4f}，"
          f"训练耗时 {fit_seconds:.4f} 秒，5 折 CV {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ---------------------------------------------------------------------------
# 步骤 5：打印对齐的中文对比表
# ---------------------------------------------------------------------------
print_section("步骤 5：五种分类器对比总表")

# 中文表格：模型名 / 测试集准确率 / 训练耗时(秒) / 5折CV均值±标准差
# 说明：中文是全角字符，用 str.ljust 按"字符数"补空格无法做到像素级对齐，
#       所以这里统一用" | "作为列分隔符，可读性最好。
header = (f"{'模型名称':<30} | {'测试集准确率':>10} | {'训练耗时(秒)':>12} | "
          f"{'5折CV均值':>10} | {'CV标准差':>9} | {'CV 均值±标准差':>24}")
line = "-" * len(header)
print(line)
print(header)
print(line)
for r in results:
    cv_str = f"{r['cv_mean']:.4f} ± {r['cv_std']:.4f}"
    print(f"{r['name']:<30} | {r['acc']:>10.4f} | {r['fit_seconds']:>12.4f} | "
          f"{r['cv_mean']:>10.4f} | {r['cv_std']:>9.4f} | {cv_str:>24}")
print(line)

best_acc = max(results, key=lambda r: r["acc"])
best_cv = max(results, key=lambda r: r["cv_mean"])
fastest = min(results, key=lambda r: r["fit_seconds"])
most_stable = min(results, key=lambda r: r["cv_std"])
print(f"测试集准确率最高：{best_acc['name']}（{best_acc['acc']:.4f}）")
print(f"5 折 CV 均值最高：{best_cv['name']}（{best_cv['cv_mean']:.4f}）")
print(f"训练最快        ：{fastest['name']}（{fastest['fit_seconds']:.4f} 秒）")
print(f"CV 最稳定(标准差最小)：{most_stable['name']}（±{most_stable['cv_std']:.4f}）")
print()
print("别忘了训练耗时是**高度依赖数据规模与硬件**的数字：")
print(f"  本数据只有 {X_train.shape[0]} 条样本、{X_train.shape[1]} 个特征，五个模型都在毫秒级；")
print("  样本量放大 100 倍后，SVM 会最先撑不住（复杂度约 O(N²)~O(N³)），")
print("  而随机森林靠并行（n_jobs=-1）还能撑得住。")

# 补充一张更全面的指标表（精确率 / 召回率 / F1 / ROC-AUC）
print()
print("再看更全面的指标（只有准确率是不够的，理由见第 ④ 部分）：")
header2 = (f"{'模型名称':<30} | {'精确率':>8} | {'召回率':>8} | {'F1':>8} | {'ROC-AUC':>9}")
line2 = "-" * len(header2)
print(line2)
print(header2)
print(line2)
for r in results:
    print(f"{r['name']:<30} | {r['prec']:>8.4f} | {r['rec']:>8.4f} | "
          f"{r['f1']:>8.4f} | {r['auc']:>9.4f}")
print(line2)
print("各指标含义（以正类=良性 benign 为例）：")
print("  精确率 precision = 判为良性的里面，真正是良性的比例（查准）；")
print("  召回率 recall    = 真正是良性的里面，被找出来的比例（查全）；")
print("  F1               = 精确率与召回率的调和平均；")
print("  ROC-AUC          = 随机取一对正负样本，正类得分高于负类的概率（0.5=瞎猜，1.0=完美）。")
print("  ROC-AUC 与阈值无关，衡量的是**排序能力**，比准确率更全面。")

# 打印其中一个模型的详细分类报告，作为"报告模板"示范
demo = results[0]
print()
print(f"以【{demo['name']}】为例，展示标准的分类报告模板：")
print(classification_report(y_test, demo["y_pred"], target_names=class_names,
                            digits=4, zero_division=0))

# ---------------------------------------------------------------------------
# 步骤 6：画图 1 —— 准确率与 CV 对比柱状图
# ---------------------------------------------------------------------------
print_section("步骤 6：绘图 —— 准确率与交叉验证对比")

short_names = ["逻辑回归", "K-近邻\nK=5", "支持向量机\nRBF 核", "决策树\n深度=3", "随机森林\n300 棵"]
accs = [r["acc"] for r in results]
cv_means = [r["cv_mean"] for r in results]
cv_stds = [r["cv_std"] for r in results]
aucs = [r["auc"] for r in results]
x_pos = np.arange(len(short_names))
width = 0.36

fig1, ax1 = plt.subplots(figsize=(12.2, 5.8))
b1 = ax1.bar(x_pos - width / 2, accs, width, label="测试集准确率", color="#4c72b0")
b2 = ax1.bar(x_pos + width / 2, cv_means, width, yerr=cv_stds, capsize=4,
             label="5 折交叉验证准确率（误差棒 = 标准差）", color="#dd8452")
low = min(min(accs), min(cv_means))
ax1.set_ylim(max(0.80, low - 0.05), 1.015)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(short_names, fontsize=10)
ax1.set_ylabel("准确率 accuracy", fontsize=11)
ax1.set_title(f"五种分类器在同一份数据上的对比（{X.shape[0]} 条乳腺癌样本，"
              f"同一次 train_test_split）", fontsize=12)
ax1.grid(axis="y", alpha=0.3, linestyle="--")
ax1.legend(loc="lower right", fontsize=10)
for b, v in zip(b1, accs):
    ax1.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}", ha="center", fontsize=9)
for b, v, s in zip(b2, cv_means, cv_stds):
    ax1.text(b.get_x() + b.get_width() / 2, v + s + 0.004, f"{v:.4f}", ha="center", fontsize=9)
fig1.tight_layout()
acc_path = OUTPUT_DIR / "03_五种分类器_准确率与CV对比.png"
fig1.savefig(acc_path, dpi=130)
plt.close(fig1)
print("已保存图片：", acc_path)

# ---------------------------------------------------------------------------
# 步骤 7：画图 2 —— ROC 曲线 + 随机森林混淆矩阵
# ---------------------------------------------------------------------------
print_section("步骤 7：绘图 —— ROC 曲线与混淆矩阵")

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14.6, 6.0))

colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"]
for r, c, sname in zip(results, colors, short_names):
    fpr, tpr, _ = roc_curve(y_test, r["score"])
    ax2a.plot(fpr, tpr, color=c, linewidth=2.0,
              label=f"{sname.replace(chr(10), ' ')}（AUC = {r['auc']:.4f}）")
ax2a.plot([0, 1], [0, 1], "k--", linewidth=1.2, label="随机猜测（AUC = 0.5）")
ax2a.set_xlim(-0.01, 1.01)
ax2a.set_ylim(-0.01, 1.02)
ax2a.set_xlabel("假正率 FPR = FP/(FP+TN)", fontsize=11)
ax2a.set_ylabel("真正率 TPR = TP/(TP+FN)（召回率）", fontsize=11)
ax2a.set_title("(a) ROC 曲线：曲线越靠近左上角越好", fontsize=11.5)
ax2a.grid(alpha=0.3, linestyle="--")
ax2a.legend(loc="lower right", fontsize=9)

rf_result = results[-1]     # 随机森林
ConfusionMatrixDisplay.from_predictions(
    y_test, rf_result["y_pred"],
    display_labels=class_names,
    cmap="Blues",
    colorbar=False,
    ax=ax2b,
)
ax2b.set_title(f"(b) 随机森林的混淆矩阵（测试集准确率 {rf_result['acc']:.4f}）\n"
               "行 = 真实类别，列 = 预测类别", fontsize=11.5)
ax2b.set_xlabel("预测类别", fontsize=10)
ax2b.set_ylabel("真实类别", fontsize=10)

fig2.suptitle("准确率只是一个数字：ROC 看排序能力，混淆矩阵看错误具体错在哪一类", fontsize=12.5)
fig2.tight_layout(rect=(0, 0, 1, 0.93))
roc_path = OUTPUT_DIR / "03_五种分类器_ROC与混淆矩阵.png"
fig2.savefig(roc_path, dpi=130)
plt.close(fig2)
print("已保存图片：", roc_path)

# 混淆矩阵的数字解释（注意两套口径要分清）
cm = confusion_matrix(y_test, rf_result["y_pred"])
print(f"随机森林混淆矩阵（行=真实类别，列=预测类别，类别顺序 {class_names}）：\n{cm}")
print(f"  按 sklearn 口径（正类 = 良性 benign = 1）：")
print(f"    cm[0,0] = {cm[0, 0]}：真负例 TN —— 真实恶性且预测恶性（正确识别出恶性）")
print(f"    cm[0,1] = {cm[0, 1]}：假正例 FP —— 真实恶性却预测良性（**漏诊**）")
print(f"    cm[1,0] = {cm[1, 0]}：假负例 FN —— 真实良性却预测恶性（**误诊**）")
print(f"    cm[1,1] = {cm[1, 1]}：真正例 TP —— 真实良性且预测良性（正确识别出良性）")
print("  换成医学口径（我们真正要抓的是'恶性'）：")
print(f"    恶性被正确识别 {cm[0, 0]} 条，漏诊 {cm[0, 1]} 条；良性被误判为恶性 {cm[1, 0]} 条。")
print("  ⚠ 同一个数字在两套口径下叫法完全相反，所以看混淆矩阵时**必须先说清楚谁是正类**。")

# ---------------------------------------------------------------------------
# 步骤 8：补充 —— 阈值对结果的影响（说明"准确率"之外的东西）
# ---------------------------------------------------------------------------
print_section("步骤 8：补充实验 —— 决策阈值如何改变结论")

print("乳腺癌筛查场景下，'漏诊（把恶性判成良性）'的代价远高于'误诊'。")
print("随机森林输出的是 P(良性)。默认规则是 'P(良性) > 0.5 判良性'（与 predict 的 argmax 一致）。")
print("如果把'判为良性'的门槛**调高**（比如要求 P(良性) > 0.7 才敢说良性），")
print("模型就会更倾向于判恶性，从而减少漏诊、提高恶性的召回率。下面看具体数据：")
rf_pipe = models[-1][1]
rf_proba = rf_pipe.predict_proba(X_test)[:, 1]      # P(良性)
print("-" * 88)
print(f"{'判良性的门槛':<14}{'准确率':>10}{'良性精确率':>12}{'良性召回率':>12}"
      f"{'恶性召回率':>12}{'漏诊 FN':>10}{'误诊 FP':>10}")
print("-" * 88)
thr_records = []
for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
    pred_thr = (rf_proba > thr).astype(int)
    acc_t = accuracy_score(y_test, pred_thr)
    prec_t = precision_score(y_test, pred_thr, zero_division=0)
    rec_t = recall_score(y_test, pred_thr, zero_division=0)
    # 恶性是类别 0：恶性召回率 = 真实恶性中被判为恶性的比例
    malignant_recall = float(((pred_thr == 0) & (y_test == 0)).sum() / max((y_test == 0).sum(), 1))
    fn = int(((pred_thr == 1) & (y_test == 0)).sum())
    fp = int(((pred_thr == 0) & (y_test == 1)).sum())
    thr_records.append((thr, acc_t, prec_t, rec_t, malignant_recall, fn, fp))
    print(f"{thr:<14.1f}{acc_t:>10.4f}{prec_t:>12.4f}{rec_t:>12.4f}"
          f"{malignant_recall:>12.4f}{fn:>10}{fp:>10}")
print("-" * 88)
t_low, t_high = thr_records[0], thr_records[-1]
print(f"门槛从 {t_low[0]:.1f} 调到 {t_high[0]:.1f}（越来越保守）：")
print(f"  · 恶性召回率从 {t_low[4]:.4f} 升到 {t_high[4]:.4f}，漏诊 FN 从 {t_low[5]} 条降到 {t_high[5]} 条；")
print(f"  · 代价是误诊 FP 从 {t_low[6]} 条升到 {t_high[6]} 条，整体准确率从 {t_low[1]:.4f} "
      f"变为 {t_high[1]:.4f}。")
print("  · 注意准确率并不是单调下降的（0.7 那一档反而比 0.6 高）——")
print("    因为准确率把所有错误一视同仁，它看不出'我们其实更在乎少漏诊'。")
print("  医学筛查通常宁可多误诊也不能漏诊，所以会把门槛调高；")
print("  反过来，如果误诊的代价极大（比如会引发有创检查），就该把门槛调低。")
print("  这说明：**同一个模型，换个决策阈值就是换了一个业务权衡**；")
print("  选阈值是业务决策，不是模型内部的事。")

# ============================================================================
# ④ 结果解读
# ============================================================================
print_section("④ 结果解读（这些数字到底意味着什么）")

print(f"1) 同一份数据（{X.shape[0]} 条乳腺癌样本、{X.shape[1]} 个特征）、同一次划分（random_state=42, "
      f"stratify=y）下：")
print(f"   测试集准确率从 {min(accs):.4f} 到 {max(accs):.4f}，跨度约 "
      f"{(max(accs) - min(accs)) * 100:.1f} 个百分点。")
print(f"   最好的是 {best_acc['name']}（{best_acc['acc']:.4f}）。")
print("   但这个差距是否'显著'要谨慎：测试集只有 "
      f"{len(y_test)} 条，错 1 条准确率就变 {(1 / len(y_test)) * 100:.2f} 个百分点，")
print("   所以排序不能只看一次划分 —— 这正是要看 5 折交叉验证的原因。")
print()
print("2) 交叉验证均值±标准差的真实排序（这才是选型依据）：")
for r in sorted(results, key=lambda x: -x["cv_mean"]):
    print(f"   {r['name']:<32} CV = {r['cv_mean']:.4f} ± {r['cv_std']:.4f}，"
          f"测试准确率 = {r['acc']:.4f}")
by_clf = {r["name"].split()[0]: r for r in results}
r_lr, r_knn, r_svc = by_clf["逻辑回归"], by_clf["K-近邻"], by_clf["支持向量机"]
r_dt, r_rf = by_clf["决策树"], by_clf["随机森林"]
print("   逐条解读：")
print(f"   · 逻辑回归（CV {r_lr['cv_mean']:.4f}）在这份数据上排第一 ——")
print("     乳腺癌的 30 个特征都是细胞核的形态学测量，恶性/良性在标准化之后**近似线性可分**，")
print("     线性模型已经够用；569 条样本对 30 个特征来说不算多，参数少的模型反而不容易过拟合。")
print("     这是一堂很好的课：**不要预设'复杂模型一定更强'，必须实测**。")
print(f"   · 决策树最差（CV {r_dt['cv_mean']:.4f}，标准差 ±{r_dt['cv_std']:.4f} 也是最大的）——")
print("     正是上一节讲的'单棵树高方差、易过拟合'；再加上深度限制在 3，表达能力也被压住了。")
print(f"   · 随机森林（CV {r_rf['cv_mean']:.4f}）明显好于单棵树（CV {r_dt['cv_mean']:.4f}，"
      f"提升 {r_rf['cv_mean'] - r_dt['cv_mean']:+.4f}），")
print("     印证了 Bagging 的价值：**集成把单棵树的高方差平均掉了**；")
print(f"     但它并没有超过逻辑回归（{r_rf['cv_mean']:.4f} vs {r_lr['cv_mean']:.4f}）——")
print("     因为这份数据的规律本来就接近线性，森林的额外表达能力用不上。")
print(f"   · KNN 与 SVM 居中（CV {r_knn['cv_mean']:.4f} / {r_svc['cv_mean']:.4f}），")
print("     两者都依赖距离或内积，必须先标准化；KNN 在 30 维上还要面对'维度灾难'。")
print("   结论：**均值高 + 标准差小**才是好模型；只看均值容易选到'碰运气型'的模型，")
print("         而只看一次测试集划分更容易被骗（测试集只有 "
      f"{len(y_test)} 条，错 1 条就变 {100 / len(y_test):.2f} 个百分点）。")
print()
print("3) 训练耗时（本数据规模下）意味着什么：")
for r in sorted(results, key=lambda x: x["fit_seconds"]):
    print(f"   {r['name']:<32} {r['fit_seconds'] * 1000:>9.2f} 毫秒")
r_rf_ms = r_rf["fit_seconds"] * 1000
print(f"   本数据只有 {X_train.shape[0]} 条样本，除随机森林（要训 300 棵树，约 "
      f"{r_rf_ms:.0f} 毫秒）外都在 10 毫秒以内，")
print("   这个排序在真实项目里没有实际意义 —— 但样本量放大到 10 万级时它会彻底翻转：")
print("   决策树/KNN 训练几乎不花时间（KNN 的 fit 只是把数据存起来），")
print("   SVM 会最先撑不住（复杂度约 O(N²)~O(N³)），")
print("   随机森林靠并行（n_jobs=-1）还能接受。")
print("   所以选型时**一定要结合数据规模**，不能只看小数据集上的耗时。")
print()
print(f"4) ROC-AUC 与准确率的排序并不完全一致（{roc_path.name} 左图）：")
by_auc = sorted(results, key=lambda x: -x["auc"])
by_acc = sorted(results, key=lambda x: -x["acc"])
for i, (ra, rc) in enumerate(zip(by_auc, by_acc), start=1):
    print(f"   第{i}名：AUC 榜 = {ra['name']:<32}（{ra['auc']:.4f}）"
          f" | 准确率榜 = {rc['name']}（{rc['acc']:.4f}）")
print("   可以看到中段的名次会互换（例如随机森林与 KNN 在两个榜上的顺序不同）。")
print("   原因：AUC 衡量的是**把正类排在前面的能力**，与阈值无关；")
print("         准确率只是一个固定阈值（0.5）下的快照，会被阈值和类别比例影响。")
print("   两个模型准确率相同，AUC 可能差很多；反之亦然。所以指标要看一组，不要只看一个。")
print()
print(f"5) 混淆矩阵告诉我们错误具体发生在哪（{roc_path.name} 右图）：")
print(f"   真实恶性 {int((y_test == 0).sum())} 条，真实良性 {int((y_test == 1).sum())} 条。")
print(f"   随机森林漏诊（恶性判成良性）{int(cm[0, 1])} 条，误诊（良性判成恶性）{int(cm[1, 0])} 条。")
print("   在医学场景里，漏诊的代价远大于误诊 —— 所以**这两类错误绝不能等权重看待**，")
print("   准确率这种把所有错误一视同仁的指标，本身就是有偏的。")
print()
print("6) 一句话总结：同一份数据上，不同算法的差异来自三处 ——")
print("   ① 假设空间不同（能不能表达非线性边界）；")
print("   ② 归纳偏置不同（偏好平滑还是偏好记忆、是否依赖距离）；")
print("   ③ 方差水平不同（单棵树 vs 森林）。")
print("   工程上正确的做法是：先确立一个**可信的评估协议**（固定划分 + 分层 + 交叉验证），")
print("   再让几个候选模型在完全相同的条件下比，最后才谈精度。")

print()
print("【如何根据场景选算法】")
print("  · 要可解释、要系数方向、要概率 → 逻辑回归（线性）或浅决策树（非线性规则）；")
print("  · 小样本 + 高维（文本 TF-IDF、基因表达）→ SVM 线性核 / 逻辑回归 + L1；")
print("  · 中等规模表格数据、想先要一个稳健的默认基线 → 随机森林")
print("    （注意：本脚本实测在这份乳腺癌数据上逻辑回归反而最好，")
print("      所以'先跑逻辑回归当基线，再上随机森林'才是稳妥的顺序）");
print("  · 特征有明确'距离/相似度'含义、且已清洗干净 → KNN；")
print("  · 类别极不平衡、且需要概率排序 → 逻辑回归 / 随机森林 + class_weight，")
print("    评估改用 F1、recall、PR-AUC，别用准确率；")
print("  · 有严格推理延迟 / 模型体积限制（嵌入式、端侧）→ 逻辑回归或浅决策树；")
print("  · 想要再高一点精度且愿意付出调参成本 → 梯度提升（XGBoost / LightGBM）。")

print()
print("【为什么不能只看准确率】")
print("  ① 类别不平衡时会骗人：99% 负类的数据里，全部预测负类也有 99% 准确率，但毫无价值；")
print("  ② 不同错误的代价不对称：医疗漏诊 ≫ 误诊，风控漏放 ≫ 误拦，")
print("     准确率把所有错误当成等价，无法表达这种差别；")
print("  ③ 它丢掉了概率信息：把 0.51 和 0.99 都算成'预测为正'，")
print("     而排序能力（AUC / PR-AUC）才是很多业务真正需要的东西；")
print("  ④ 它是单点估计，没有不确定性：测试集只有几百条时，")
print("     准确率的 95% 置信区间可能有 ±5 个百分点，排序很容易翻转；")
print("  ⑤ 它不告诉你模型错在哪里：必须看混淆矩阵、看分类报告，")
print("     甚至去看被错分的具体样本。")
print("  正确姿势：**准确率 + 精确率 + 召回率 + F1 + ROC-AUC（或 PR-AUC）+ 混淆矩阵**")
print("            + 交叉验证的均值与标准差，一起看。")

# ============================================================================
# 超参数怎么调
# ============================================================================
# 【超参数怎么调】—— 横向对比场景下的统一调参策略
#
# 0. 第一原则：**先把评估协议固定下来，再调参。**
#    - 固定 train_test_split(random_state=42, stratify=y)；
#    - 交叉验证一律用 StratifiedKFold(5, shuffle=True, random_state=42)；
#    - 所有预处理（标准化等）放进 Pipeline，避免数据泄漏；
#    - 只有这样，模型之间的比较才是可比的。
#
# 1. 调参预算分配（经验值）：
#    - 逻辑回归：只有 C（和 l1_ratio）值得调，搜索空间小，几十次评估就够；
#    - KNN：n_neighbors + weights + p，网格很小；
#    - SVM：C × gamma 二维网格，搜索空间最大，值得多花预算；
#    - 决策树：max_depth + min_samples_leaf + ccp_alpha；
#    - 随机森林：max_features + min_samples_leaf，n_estimators 直接给 300~500 不调。
#
# 2. 各模型的最小调参网格（可直接复制）：
#    逻辑回归   {"clf__C": [0.01, 0.1, 1, 10]}
#    KNN        {"clf__n_neighbors": [3, 5, 7, 11], "clf__weights": ["uniform", "distance"]}
#    SVM        {"clf__C": [0.1, 1, 10, 100], "clf__gamma": ["scale", 0.01, 0.1, 1]}
#    决策树     {"clf__max_depth": [3, 5, 7, None], "clf__min_samples_leaf": [1, 2, 5]}
#    随机森林   {"clf__max_features": ["sqrt", "log2", 0.5], "clf__min_samples_leaf": [1, 2]}
#
# 3. 评分函数（scoring）要跟业务对齐，不要一律用 accuracy：
#    - 类别平衡、错误等价 → "accuracy"；
#    - 关心少数类召回 → "recall"；
#    - 想要平衡 → "f1" / "f1_weighted"；
#    - 关心排序能力 → "roc_auc"；极不平衡时用 "average_precision"（PR-AUC）。
#
# 4. 用嵌套交叉验证 / 独立测试集防止"调参过拟合"：
#    - 如果在同一份测试集上反复挑模型，测试集就被你'用坏'了；
#    - 正确做法：GridSearchCV(cv=5) 只在内层训练数据上选参数，
#      最终只在一次测试集上报告结果（或用嵌套 CV 得到无偏估计）。
#
# 5. 多指标同时看：
#    GridSearchCV(..., scoring=["accuracy", "f1", "roc_auc"], refit="f1")
#    —— 一次搜索拿到三个指标的完整报告，避免"为了刷准确率牺牲召回率"。
#
# 6. 随机森林的"免费"技巧：
#    - 打开 oob_score=True，用 OOB 分数代替一次交叉验证来粗调 max_features；
#    - n_jobs=-1 用满核；数据很大时把 max_samples 调到 0.5~0.8 进一步提速。
#
# 7. 别忘了"数据和特征"永远比"模型和参数"重要：
#    在本项目里，同一次划分下五个模型的准确率差距只有几个百分点，
#    而换一套更好的特征往往能带来十几甚至几十个百分点的提升。
#    调参之前，先看特征工程和样本质量。
# ============================================================================

print()
print("【完成】06_五种分类器对比.py 运行结束")

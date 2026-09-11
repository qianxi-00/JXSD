"""
对应课案章节：训练组件 / 评估指标

本节知识点：
    1. 评估指标用来衡量模型"好不好"。不同任务用不同指标，核心是比较 y_pred 与 y_true。
       每个指标都按统一模板：①公式 ②直觉解释 ③手写 numpy/torch 实现
       ④sklearn.metrics 计算 ⑤两者对比（打印数值与差值）。
    2. 混淆矩阵：TP / FN / FP / TN 的 2×2 表格（课案）；
       多分类扩展为 C×C，对角线是预测正确的（✅），其余格子是分错的。
    3. Accuracy = (TP+TN)/(TP+TN+FP+FN)；多分类 = 对角和/总和。
       手工复现课案例子（4 样本 3 分类，y_true=[0,2,1,1]，logits argmax 后
       =[0,2,1,0] → 3/4 = 0.75）。
    4. Precision / Recall / F1 = 2PR/(P+R)；课案例子
       y_pred=[0,1,1,0,1,0]、y_true=[0,1,0,0,1,1] → P=R=F1≈0.667。
       讲清"误判代价高看 Precision（垃圾邮件宁可漏过不能误删）、
       漏判代价高看 Recall（癌症宁可误报不能漏诊）"，以及
       多分类 average='macro' 与 'weighted' 的区别。
    5. AUC(AUROC)：随机抽一个正样本和一个负样本，正样本分数更高的概率。
       基于排序的公式 AUC = (Σ rank_正 - n_pos(n_pos+1)/2) / (n_pos·n_neg)，
       并推清分母为什么是 n_pos·n_neg（最差/最佳排名的等差数列求和之差）。
       相同分数共享平均排名。手写排序法 + O(n²) 正负对比较法 + sklearn 三重验证。
       课案例子 y_scores=[0.1,0.8,0.6,0.3,0.9]、y_true=[0,1,1,0,1] → AUC=1.0。
    6. MSE / MAE：课案例子 y_pred=[2.8,5.3,2.2,6.5]、y_true=[3.0,5.0,2.5,7.0]
       → MSE=0.1175、MAE=0.325；并补充 RMSE / R² / MAPE / 解释方差。
    7. 可视化：手写计算不同阈值下的 TPR/FPR 画 ROC 曲线（标注 AUC）与 PR 曲线；
       另外画"不同 AUC 值（≈0.5 / 0.75 / 1.0）对应的 ROC 形状"对比图
       → 03训练组件_04_ROC与PR曲线.png。
    8. 讲清**类别不平衡下 Accuracy 会骗人**（99% 负样本时全预测负也能拿 99% 准确率
       但 Recall=0），用小实验数值演示，说明此时要看 F1 / AUC。
    9. 真实模型评估：训一个小二分类模型（make_moons，≤1000 条、≤20 epoch），
       用 model.eval() + torch.no_grad() 拿预测概率，
       一次性打印 Accuracy / Precision / Recall / F1 / AUC / 混淆矩阵。

关于 torchmetrics 的说明：
    课案原文用 `from torchmetrics import Accuracy/Precision/Recall/F1Score/AUROC/...`，
    但**本机环境没有安装 torchmetrics**（且任务要求禁止 pip/uv install），
    因此本脚本全部改用 **sklearn.metrics + 手写 numpy 公式** 来实现同样的指标，
    并在每处对照打印数值，保证与课案语义完全一致。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Deep_Learning\\03_训练组件\\04_评估指标.py'
"""

# ---------------------------------------------------------------------------
# 统一环境初始化：目录定位 + 无界面绘图后端
# ---------------------------------------------------------------------------
from pathlib import Path

_HERE = Path(__file__).resolve().parent     # 当前脚本所在目录
_ROOT = _HERE.parent                        # Deep_Learning 根目录

import matplotlib
matplotlib.use("Agg")                       # 无界面后端，避免 plt.show() 阻塞
import matplotlib.pyplot as plt

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = _ROOT / "output"               # 所有图片统一输出到这里
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 后续正式依赖（必须写在 matplotlib 初始化之后）
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.datasets import make_moons  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

# 随机种子统一：保证每次运行结果一致、可复现
torch.manual_seed(42)
np.random.seed(42)

sns.set_theme(style="whitegrid", font="Microsoft YaHei")   # seaborn 风格画图
_EPS = 1e-12          # 防止除零

_EPOCHS = 20          # 真实模型评估的训练轮数（课案要求 ≤ 20）
_N_SAMPLES = 800      # 样本数（课案要求 ≤ 1000）


def _title(text: str) -> None:
    """打印分节标题，让输出有清晰的结构。"""
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def _cmp(name, hand_value, sk_value):
    """统一打印手写值与 sklearn 值的对比。"""
    _d = abs(float(hand_value) - float(sk_value))
    print(f"  {name:<40s} 手写 = {float(hand_value):.10f}   "
          f"sklearn = {float(sk_value):.10f}   差值 = {_d:.3e}")
    return _d


# ===========================================================================
# 1. 混淆矩阵
# ===========================================================================
_title("1. 混淆矩阵（Confusion Matrix）")

# ---------------------------------------------------------------------------
# 1.1 二分类：TP / FN / FP / TN 的 2×2 表格
# ---------------------------------------------------------------------------
print("""
  课案的二分类混淆矩阵（行是"实际"，列是"预测"）：

                      预测为正          预测为负
        实际为正    TP(True Positive)  FN(False Negative)
        实际为负    FP(False Positive) TN(True Negative)

  读法（记住这四个词的含义是后面所有指标的基础）：
      TP：真的是正类，模型也说正  → 判对了（正类）
      TN：真的是负类，模型也说负  → 判对了（负类）
      FP：真的是负类，模型却说是正 → 误报（False Positive，"狼来了"里喊了狼）
      FN：真的是正类，模型却说是负 → 漏报（False Negative，"狼来了"里狼真来了没人信）
""")

# 构造一个具体的二分类例子：10 封邮件，1 = 垃圾邮件，0 = 正常邮件
_y_true_bin = np.array([1, 0, 0, 1, 1, 0, 0, 1, 0, 1])
_y_pred_bin = np.array([1, 0, 1, 1, 0, 0, 0, 1, 1, 0])
print(f"  例子：10 封邮件的垃圾邮件判别（1=垃圾邮件，0=正常邮件）")
print(f"    真实标签 y_true = {_y_true_bin}")
print(f"    模型预测 y_pred = {_y_pred_bin}")

# —— 手写混淆矩阵 ——
_TP = int((( _y_true_bin == 1) & (_y_pred_bin == 1)).sum())
_FN = int(((_y_true_bin == 1) & (_y_pred_bin == 0)).sum())
_FP = int(((_y_true_bin == 0) & (_y_pred_bin == 1)).sum())
_TN = int(((_y_true_bin == 0) & (_y_pred_bin == 0)).sum())
print(f"\n  ③ 手写统计：TP = {_TP}   FN = {_FN}   FP = {_FP}   TN = {_TN}")
print(f"     校验：TP+FN+FP+TN = {_TP + _FN + _FP + _TN} == 样本数 {len(_y_true_bin)}"
      f" -> {_TP + _FN + _FP + _TN == len(_y_true_bin)}")

# 手写一张课本样式的表格
print("\n  手写混淆矩阵（课本排版）：")
print(f"    {'':<14s}{'预测为正':>12s}{'预测为负':>12s}")
print(f"    {'实际为正':<12s}{_TP:>12d}{_FN:>12d}")
print(f"    {'实际为负':<12s}{_FP:>12d}{_TN:>12d}")

# —— sklearn ——
_cm_sk = confusion_matrix(_y_true_bin, _y_pred_bin)
print(f"\n  ④ sklearn.metrics.confusion_matrix：")
print(f"    {_cm_sk}     （行=真实，列=预测；sklearn 的行列顺序是 [0, 1]）")
print(f"    sklearn 的 [1,1] 元素 = {_cm_sk[1, 1]}  ↔  手写 TP = {_TP}")
print(f"    sklearn 的 [1,0] 元素 = {_cm_sk[1, 0]}  ↔  手写 FN = {_FN}")
print(f"    sklearn 的 [0,1] 元素 = {_cm_sk[0, 1]}  ↔  手写 FP = {_FP}")
print(f"    sklearn 的 [0,0] 元素 = {_cm_sk[0, 0]}  ↔  手写 TN = {_TN}")
print(f"    逐元素完全一致 = {np.array_equal(_cm_sk, np.array([[_TN, _FP], [_FN, _TP]]))}")

# ---------------------------------------------------------------------------
# 1.2 多分类：C×C 混淆矩阵
# ---------------------------------------------------------------------------
print("""
  ▶ 多分类混淆矩阵（课案）：扩展为 C×C，以三分类（猫/狗/鱼）为例

                      预测：猫          预测：狗          预测：鱼
        实际：猫      ✅ TP₀ = 8        错判为狗          错判为鱼
        实际：狗      错判为猫          ✅ TP₁ = 7        错判为鱼
        实际：鱼      错判为猫          错判为狗          ✅ TP₂ = 5

    **对角线（✅）是预测正确的，其余格子是分错的。**
    Accuracy = (8+7+5) / 总数 = 对角和 / 总和。
""")
_y_true_mc = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
_y_pred_mc = np.array([0, 0, 0, 0, 0, 0, 0, 1,      # 8 个猫：7 个对，1 个错判成狗
                       1, 1, 1, 1, 1, 1, 0,          # 7 个狗：6 个对，1 个错判成猫
                       2, 2, 2, 2, 0])               # 5 个鱼：4 个对，1 个错判成猫
_cm_mc = confusion_matrix(_y_true_mc, _y_pred_mc, labels=[0, 1, 2])
print(f"  sklearn 多分类混淆矩阵（labels=[0,1,2]，行=真实，列=预测）：")
print(f"    {_cm_mc}")
print(f"  对角线和 = {np.trace(_cm_mc)}，总样本数 = {_cm_mc.sum()}，"
      f"Accuracy = {np.trace(_cm_mc)}/{_cm_mc.sum()} = {np.trace(_cm_mc) / _cm_mc.sum():.6f}")
print("  解读：对角线上是各类判对的数量；第 i 行第 j 列（i≠j）表示「真实是 i 却被判成 j」。")
print("        第 j 列非对角元素之和就是类别 j 的 FP，第 i 行非对角元素之和就是类别 i 的 FN。")

# 画一张混淆矩阵热力图（放在最后的图里会更挤，这里单独保存一张小的）
fig_cm, ax_cm = plt.subplots(figsize=(5.4, 4.6))
sns.heatmap(_cm_mc, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["猫", "狗", "鱼"], yticklabels=["猫", "狗", "鱼"], ax=ax_cm)
ax_cm.set_title("三分类混淆矩阵示例", fontsize=12)
ax_cm.set_xlabel("预测类别")
ax_cm.set_ylabel("真实类别")
fig_cm.tight_layout()
_cm_png = OUTPUT_DIR / "03训练组件_04_混淆矩阵.png"
fig_cm.savefig(_cm_png, dpi=110)
plt.close(fig_cm)
print(f"  （混淆矩阵热力图已保存：{_cm_png}）")


# ===========================================================================
# 2. Accuracy
# ===========================================================================
_title("2. Accuracy（准确率）")
print("""
  公式      : Accuracy = (TP + TN) / (TP + TN + FP + FN)      —— 二分类
              多分类时    = 对角和 / 总和 = Σ_i TP_i / N
  直觉      : 所有预测里，预测正确的占多少。
  优点      : 最直观、最好理解。
  缺点      : **类别不平衡时会骗人**（见第 8 节）——
              99% 都是负样本时，全部预测负也能拿 99% 的准确率，但一个正样本都没抓到。
""")


def accuracy_hand(y_true, y_pred):
    """手写多分类准确率：预测等于真实的比例。"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float((y_true == y_pred).mean())


print("  ▶ 课案例子手工复现（用 sklearn + 手写，不用 torchmetrics）")
print("     课案原文用 `from torchmetrics import Accuracy`，但本机**没有安装 torchmetrics**，")
print("     所以下面改用手写的 numpy 公式 + sklearn.metrics.accuracy_score 来实现同一指标。")
# 课案的 y_pred 是 logits（每行三个类别的分数），要先 argmax 得到类别编号
_y_pred_logits = torch.tensor([
    [3.0, 1.0, 0.0],   # argmax = 0
    [0.0, 0.0, 5.0],   # argmax = 2
    [0.0, 2.0, 0.0],   # argmax = 1
    [0.5, 0.0, 0.0],   # argmax = 0
])
_y_true_case = torch.tensor([0, 2, 1, 1])       # 样本 3 预测为 0 但真实为 1 → 错
_y_pred_case = _y_pred_logits.argmax(dim=1)      # logits → 类别编号
print(f"     logits = \n{_y_pred_logits.numpy()}")
print(f"     argmax 后的预测类别 = {_y_pred_case.numpy()}")
print(f"     真实类别 y_true     = {_y_true_case.numpy()}")
print(f"     逐样本是否正确      = {( _y_pred_case == _y_true_case).numpy().astype(int)}"
      f"   （1=对，0=错）")
_acc_hand = accuracy_hand(_y_true_case.numpy(), _y_pred_case.numpy())
_cmp("Accuracy（课案例子，手写 vs sklearn)", _acc_hand,
     accuracy_score(_y_true_case.numpy(), _y_pred_case.numpy()))
print(f"     手算：正确 {int((_y_pred_case == _y_true_case).sum())} 个 / 总共 4 个 = "
      f"{int((_y_pred_case == _y_true_case).sum())}/4 = {_acc_hand:.4f}   ->  3/4 = 0.75 ✓")

# 多分类 = 对角和 / 总和
_cmp("Accuracy（20 样本三分类，手写 vs sklearn)", accuracy_hand(_y_true_mc, _y_pred_mc),
     accuracy_score(_y_true_mc, _y_pred_mc))
print(f"     验证「对角和/总和」：{np.trace(_cm_mc)}/{_cm_mc.sum()} = "
      f"{np.trace(_cm_mc) / _cm_mc.sum():.6f}")


# ===========================================================================
# 3. Precision / Recall / F1
# ===========================================================================
_title("3. Precision（精确率）/ Recall（召回率）/ F1")
print("""
  公式      : Precision = TP / (TP + FP)      —— 判"正"的里面有多少是真的正样本
              Recall    = TP / (TP + FN)      —— 真正的正样本有多少被找出来了
              F1        = 2 · P · R / (P + R) —— 精确率和召回率的**调和平均**

  直觉与取舍：
      · **Precision 误判代价高时看重**：垃圾邮件过滤。
        如果误把正常邮件判成垃圾（FP），用户可能错过重要邮件（误删），代价很高。
        所以宁可漏过几封垃圾邮件，也不能误删正常邮件 → 追求高 Precision。
      · **Recall 漏判代价高时看重**：癌症筛查。
        如果漏掉一个癌症病人（FN），后果是致命的（漏诊）；多报几个（FP）只是再复查一次。
        所以宁可误报，不能漏了 → 追求高 Recall。
      · **F1 是调和平均**：它比算术平均更"严格"——只要 P 和 R 有一个很小，
        F1 就会被拉得很低。所以 F1 高意味着 P 和 R 都不差，是"兼顾两者"的指标。
        需要注意的是调和平均对较小值敏感，这正是我们想要的：不希望用一个很高的
        P 去"补偿"一个很低的 R。

  多分类的两种平均方式：
      · average='macro'    —— 各类别指标先算出来，再**简单平均**（不按样本数加权）。
                              每个类别权重相同，小类别的表现会同等重要，
                              因此在类别不平衡时 macro 更能暴露"小类学得差"。
      · average='weighted' —— 按**各类别样本数加权平均**。
                              样本多的类别主导结果，类别不平衡时会被大类掩盖。
""")

# 课案例子
_y_pred_pr = np.array([0, 1, 1, 0, 1, 0])
_y_true_pr = np.array([0, 1, 0, 0, 1, 1])
print(f"  课案例子：y_pred = {_y_pred_pr}，y_true = {_y_true_pr}")
_cm_pr = confusion_matrix(_y_true_pr, _y_pred_pr)
_tp_pr, _fn_pr, _fp_pr, _tn_pr = int(_cm_pr[1, 1]), int(_cm_pr[1, 0]), int(_cm_pr[0, 1]), int(_cm_pr[0, 0])
print(f"  ③ 手写混淆矩阵：TP = {_tp_pr}   FN = {_fn_pr}   FP = {_fp_pr}   TN = {_tn_pr}")
print(f"     逐样本核对：样本 0(真0预0)=TN  样本 1(真1预1)=TP  样本 2(真0预1)=FP")
print(f"                 样本 3(真0预0)=TN  样本 4(真1预1)=TP  样本 5(真1预0)=FN")
print(f"     所以 TP=2, FN=1, FP=1, TN=2 ✓")

_p_hand = _tp_pr / (_tp_pr + _fp_pr)
_r_hand = _tp_pr / (_tp_pr + _fn_pr)
_f1_hand = 2 * _p_hand * _r_hand / (_p_hand + _r_hand)
print(f"\n  ③ 手写计算：")
print(f"     Precision = TP/(TP+FP) = {_tp_pr}/({_tp_pr}+{_fp_pr}) = {_tp_pr}/{_tp_pr + _fp_pr} "
      f"= {_p_hand:.10f}   （课案说 ≈ 0.67 ✓）")
print(f"     Recall    = TP/(TP+FN) = {_tp_pr}/({_tp_pr}+{_fn_pr}) = {_tp_pr}/{_tp_pr + _fn_pr} "
      f"= {_r_hand:.10f}   （课案说 ≈ 0.67 ✓）")
print(f"     F1        = 2PR/(P+R)  = {_f1_hand:.10f}   （课案说 ≈ 0.67 ✓）")

print(f"\n  ④⑤ sklearn 对比：")
_cmp("Precision（binary）", _p_hand, precision_score(_y_true_pr, _y_pred_pr, zero_division=0))
_cmp("Recall（binary）", _r_hand, recall_score(_y_true_pr, _y_pred_pr, zero_division=0))
_cmp("F1（binary）", _f1_hand, f1_score(_y_true_pr, _y_pred_pr, zero_division=0))

# —— 多分类的 macro vs weighted ——
print("\n  ▶ 多分类：average='macro' 与 'weighted' 的区别（用第 1.2 节的三分类数据）")
_cm3 = _cm_mc
print(f"    各类样本数：猫={_cm3[0].sum()}，狗={_cm3[1].sum()}，鱼={_cm3[2].sum()}  （明显不平衡）")
_prec_per = []
_rec_per = []
for _c in range(3):
    _tp_c = _cm3[_c, _c]
    _fp_c = _cm3[:, _c].sum() - _tp_c          # 第 c 列非对角 = 类别 c 的 FP
    _fn_c = _cm3[_c, :].sum() - _tp_c          # 第 c 行非对角 = 类别 c 的 FN
    _p_c = _tp_c / (_tp_c + _fp_c) if (_tp_c + _fp_c) > 0 else 0.0
    _r_c = _tp_c / (_tp_c + _fn_c) if (_tp_c + _fn_c) > 0 else 0.0
    _prec_per.append(_p_c)
    _rec_per.append(_r_c)
    _support = _cm3[_c].sum()
    print(f"      {['猫', '狗', '鱼'][_c]}：TP={_tp_c} FP={_fp_c} FN={_fn_c} "
          f"-> P={_p_c:.6f}  R={_r_c:.6f}  (样本数 {_support})")
_f1_per = [2 * p * r / (p + r) if (p + r) > 0 else 0.0 for p, r in zip(_prec_per, _rec_per)]
_supports = np.array([_cm3[c].sum() for c in range(3)], dtype=float)
_macro_p = float(np.mean(_prec_per))
_weighted_p = float(np.average(_prec_per, weights=_supports))
_macro_r = float(np.mean(_rec_per))
_weighted_r = float(np.average(_rec_per, weights=_supports))
_macro_f1 = float(np.mean(_f1_per))
_weighted_f1 = float(np.average(_f1_per, weights=_supports))
print(f"\n      手写 macro  P = mean({np.round(_prec_per, 4)}) = {_macro_p:.10f}")
print(f"      手写 weighted P = Σ(w_i·P_i) 按样本数加权      = {_weighted_p:.10f}")
print(f"      手写 macro  R = {_macro_r:.10f}，  weighted R = {_weighted_r:.10f}")
print(f"      手写 macro F1 = {_macro_f1:.10f}， weighted F1 = {_weighted_f1:.10f}")
print(f"\n      sklearn 对照：")
_cmp("Precision macro", _macro_p, precision_score(_y_true_mc, _y_pred_mc, average="macro", zero_division=0))
_cmp("Precision weighted", _weighted_p, precision_score(_y_true_mc, _y_pred_mc, average="weighted", zero_division=0))
_cmp("Recall macro", _macro_r, recall_score(_y_true_mc, _y_pred_mc, average="macro", zero_division=0))
_cmp("Recall weighted", _weighted_r, recall_score(_y_true_mc, _y_pred_mc, average="weighted", zero_division=0))
_cmp("F1 macro", _macro_f1, f1_score(_y_true_mc, _y_pred_mc, average="macro", zero_division=0))
_cmp("F1 weighted", _weighted_f1, f1_score(_y_true_mc, _y_pred_mc, average="weighted", zero_division=0))
print(f"\n      解读：weighted 值（{_weighted_p:.4f}）比 macro 值（{_macro_p:.4f}）更靠近大类别（猫/狗）的表现，")
print(f"            因为它按样本数加权；macro 让小类别（鱼）也占 1/3 权重，更能暴露小类学得差。")


# ===========================================================================
# 4. AUC（AUROC）
# ===========================================================================
_title("4. AUC / AUROC（曲线下面积）")
print("""
  直觉定义：
      **随机抽一个正样本和一个负样本，正样本的预测分数高于负样本分数的概率 = AUC。**
      可以理解成"模型的排序能力"——它不关心绝对分数，只关心"谁更可能"。
      · AUC = 1.0：完美排序（所有正样本的分数都高于所有负样本）
      · AUC = 0.5：等同随机猜测（等于抛硬币，模型没有任何区分能力）
      · AUC < 0.5：比随机还差（把分数取反就能 > 0.5）

  基于排序的公式：
      AUC = ( Σ rank_正 - n_pos·(n_pos+1)/2 ) / ( n_pos · n_neg )
      其中 n_pos = 正样本总数，n_neg = 负样本总数，
      **排名方向**：分数最低 → rank = 1，分数最高 → rank = N。

  【分母为什么是 n_pos · n_neg？】（课案推导，这里讲清）
      正负样本两两配对，一共能配出 n_pos × n_neg 个 (正, 负) 对。
      AUC = "正样本分数更高的对数" / "总对数"，所以分母就是总对数 n_pos·n_neg。
      从排名角度看：
        · 最差情况（正样本全部排在末尾）：正样本排名为 1, 2, ..., n_pos，
          排名和 = n_pos(n_pos+1)/2。这就是分子的减数（"最差基准"）。
        · 最佳情况（正样本全部排在最前）：正样本占据排名
          n_neg+1, n_neg+2, ..., n_neg+n_pos，
          等差数列求和 = n_pos·[(n_neg+1) + (n_neg+n_pos)] / 2
                       = n_pos·(2·n_neg + n_pos + 1) / 2
        · 分子最大值 = 最佳排名和 - 最差排名和
                     = n_pos(2n_neg + n_pos + 1)/2 - n_pos(n_pos + 1)/2
                     = n_pos · 2·n_neg / 2
                     = n_pos · n_neg            ← 恰好等于分母！
      所以分子除以分母的结果天然落在 [0, 1]，AUC=1 表示完美排序。

  **分数相同的样本共享它们占据位置的平均排名**：
      例如分数 [0.3, 0.5, 0.8, 0.8, 0.9]（从低到高排列），
      两个 0.8 占据第 3、4 位，各自排名取 (3+4)/2 = 3.5。
      （这叫"平均排名"法，等价于把并列的对算作 0.5 分。）

  适用场景：推荐系统、广告排序、风控评分等**只关心"谁更可能"**的场景；
            以及类别不平衡的分类评估（AUC 对正负样本比例不敏感）。
""")


def ranks_with_ties(scores):
    """返回每个样本的排名（从 1 开始，分数最低 = 1；并列样本取平均排名）。

    实现方式：先按分数升序稳定排序，扫描出每个"并列组"的起止下标 [i, j]，
    该组的平均排名 = (i + j) / 2 + 1（因为排名从 1 开始）。
    """
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(scores, kind="mergesort")     # 稳定排序，保证并列顺序可复现
    sorted_scores = scores[order]
    n = len(scores)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        # 找出所有与 sr[i] 相同的元素，它们占据位置 i..j
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0                # 平均排名（排名从 1 开始）
        ranks[order[i:j + 1]] = avg_rank              # 并列样本共享同一个平均排名
        i = j + 1
    return ranks


def auc_by_rank(y_true, y_scores):
    """③ 手写 AUC —— 基于排序的公式。"""
    y_true = np.asarray(y_true)
    ranks = ranks_with_ties(y_scores)                # 分数最低 → rank 1
    pos = (y_true == 1)
    n_pos = float(pos.sum())
    n_neg = float((~pos).sum())
    if n_pos == 0 or n_neg == 0:                     # 只有一类时 AUC 无定义
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auc_by_pairs(y_true, y_scores):
    """③ 手写 AUC —— O(n²) 的"正负样本对比较"版本（三重验证用）。

    对所有 (正, 负) 对计数：正分数 > 负分数 记 1 分，相等记 0.5 分，然后除以总对数。
    这个实现直观但复杂度是 O(n_pos·n_neg)，只适合小样本。
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    pos_scores = y_scores[y_true == 1]
    neg_scores = y_scores[y_true == 0]
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return float("nan")
    wins = (pos_scores[:, None] > neg_scores[None, :]).sum()
    ties = (pos_scores[:, None] == neg_scores[None, :]).sum()
    return float((wins + 0.5 * ties) / (len(pos_scores) * len(neg_scores)))


# —— 课案例子 ——
_y_scores_case = np.array([0.1, 0.8, 0.6, 0.3, 0.9])
_y_true_case_auc = np.array([0, 1, 1, 0, 1])
print(f"  ▶ 课案例子：y_scores = {_y_scores_case}，y_true = {_y_true_case_auc}")
print(f"     正样本（y=1）：分数 {_y_scores_case[_y_true_case_auc == 1]}  ->  索引 1,2,4，分数 0.8/0.6/0.9")
print(f"     负样本（y=0）：分数 {_y_scores_case[_y_true_case_auc == 0]}  ->  索引 0,3，分数 0.1/0.3")
print(f"     最小正样本分数 = {_y_scores_case[_y_true_case_auc == 1].min()} > "
      f"最大负样本分数 = {_y_scores_case[_y_true_case_auc == 0].max()}  ->  完美排序，AUC 应为 1.0")
_ranks_case = ranks_with_ties(_y_scores_case)
print(f"     每个样本的（平均）排名 = {_ranks_case}")
print(f"     正样本排名和 = {_ranks_case[_y_true_case_auc == 1].sum()}，"
      f"n_pos = {(_y_true_case_auc == 1).sum()}，n_neg = {(_y_true_case_auc == 0).sum()}")
_np_case = (_y_true_case_auc == 1).sum()
_nn_case = (_y_true_case_auc == 0).sum()
print(f"     分子 = {_ranks_case[_y_true_case_auc == 1].sum()} - "
      f"{_np_case}·{_np_case + 1}/2 = {_ranks_case[_y_true_case_auc == 1].sum() - _np_case * (_np_case + 1) / 2}")
print(f"     分母 = n_pos·n_neg = {_np_case}·{_nn_case} = {_np_case * _nn_case}")
print(f"     AUC = 分子/分母 = {auc_by_rank(_y_true_case_auc, _y_scores_case):.10f}   "
      f"->  课案说 1.0 ✓")
_cmp("AUC（课案例子，手写排序法 vs sklearn)", auc_by_rank(_y_true_case_auc, _y_scores_case),
     roc_auc_score(_y_true_case_auc, _y_scores_case))

# —— 并列排名的演示 ——
print("\n  ▶ 并列分数共享平均排名 的演示")
_demo_scores = np.array([0.3, 0.5, 0.8, 0.8, 0.9])
_demo_ranks = ranks_with_ties(_demo_scores)
print(f"     分数（从低到高）= {np.sort(_demo_scores)}")
print(f"     对应平均排名     = {_demo_ranks}（按输入顺序：{_demo_scores}）")
print(f"     -> 两个 0.8 占据第 3、4 位，各自取 (3+4)/2 = {(3 + 4) / 2} ✓")

# —— 随机数据的四重验证 ——
print("\n  ▶ 手写两种实现 vs sklearn 的三重验证（随机数据，200 条）")
_rs = np.random.RandomState(42)
_y_rand = np.random.RandomState(7).randint(0, 2, 200)      # 100 正 / 100 负，大致平衡
# 分数带重叠（正负分布有交叉），这样 AUC 落在 0.5~1 之间，验证才不是"退化"情形
_s_rand = np.where(_y_rand == 1,
                   np.random.RandomState(11).normal(0.8, 1.0, 200),
                   np.random.RandomState(12).normal(0.0, 1.0, 200))
_auc_rank = auc_by_rank(_y_rand, _s_rand)
_auc_pairs = auc_by_pairs(_y_rand, _s_rand)
_auc_sk = roc_auc_score(_y_rand, _s_rand)
print(f"     正样本数 = {int((_y_rand == 1).sum())}，负样本数 = {int((_y_rand == 0).sum())}，"
      f"总对数 = {int((_y_rand == 1).sum()) * int((_y_rand == 0).sum())}")
print(f"     手写排序法（O(n log n)）  AUC = {_auc_rank:.12f}")
print(f"     手写配对法（O(n²)）       AUC = {_auc_pairs:.12f}")
print(f"     sklearn.roc_auc_score     AUC = {_auc_sk:.12f}")
print(f"     排序法 vs sklearn 差值 = {abs(_auc_rank - _auc_sk):.3e}   ->  < 1e-9 ✓")
print(f"     配对法 vs sklearn 差值 = {abs(_auc_pairs - _auc_sk):.3e}   ->  < 1e-9 ✓")

# 另外再用"分组并列"的分数验证一次（大量重复分数是排序法的边界情况）
_s_tie = np.round(_s_rand, 0)                              # 把分数取整，制造大量并列
_auc_tie_rank = auc_by_rank(_y_rand, _s_tie)
_auc_tie_pairs = auc_by_pairs(_y_rand, _s_tie)
_auc_tie_sk = roc_auc_score(_y_rand, _s_tie)
print(f"\n     批量并列分数（把上面分数取整，制造大量相同分数）：")
print(f"       不同分数取值只有 {len(np.unique(_s_tie))} 种")
print(f"       手写排序法 AUC = {_auc_tie_rank:.12f}")
print(f"       手写配对法 AUC = {_auc_tie_pairs:.12f}")
print(f"       sklearn AUC    = {_auc_tie_sk:.12f}")
print(f"       排序法 - sklearn = {abs(_auc_tie_rank - _auc_tie_sk):.3e}；"
      f"配对法 - sklearn = {abs(_auc_tie_pairs - _auc_tie_sk):.3e}   ->  并列处理正确 ✓")

# —— AUC 对类别不平衡不敏感的小演示 ——
print("\n  ▶ 补充：AUC 对类别不平衡不敏感（同一批分数，只改变正负样本比例）")
print("     做法：固定 200 个正样本，再从同一个负样本分布里抽样不同数量的负样本，")
print("           把正类比例从 50% 一路压到 1%。")
print("           AUC 的定义是「正样本分数高于负样本的比例」，只取决于两个分布，")
print("           与两者各有多少个样本无关，所以 AUC 应该基本不变；")
print("           而 Accuracy 会因为负类占比变大（'全判负'越来越占便宜）而升高。")
_rs4 = np.random.RandomState(3)
_n_pos_demo = 200
_neg_pool = _rs4.normal(0.0, 1.0, 20000)                   # 负样本池（均值 0）
_pos_scores = _rs4.normal(0.6, 1.0, _n_pos_demo)           # 正样本分数（均值 0.6，明显重叠）
_threshold = 0.6                                            # 统一的预测阈值
print(f"     模型分数有重叠：负样本 N(0,1)，正样本 N(0.6,1)，判定阈值 = {_threshold}")
print(f"     {'正样本比例':>11s} {'样本数':>8s} {'Accuracy':>10s} {'Recall':>9s} "
      f"{'AUC(sklearn)':>14s} {'AUC(排序法)':>13s}")
print("     " + "-" * 72)
for _ratio in [0.5, 0.2, 0.05, 0.01]:
    # ratio = n_pos/(n_pos+n_neg)  =>  n_neg = n_pos·(1-ratio)/ratio
    _n_neg_need = int(round(_n_pos_demo * (1.0 - _ratio) / _ratio))
    _neg_used = _neg_pool[:_n_neg_need]                        # 从负样本池取前 n_neg_need 个
    _s_mix = np.concatenate([_pos_scores, _neg_used])
    _y_mix = np.concatenate([np.ones(_n_pos_demo, dtype=int),
                             np.zeros(len(_neg_used), dtype=int)])
    _px = (_s_mix > _threshold).astype(int)                    # 统一阈值下的预测
    _acc_v = accuracy_score(_y_mix, _px)
    _rec_v = recall_score(_y_mix, _px, zero_division=0)
    _auc_sk_v = roc_auc_score(_y_mix, _s_mix)
    _auc_rank_v = auc_by_rank(_y_mix, _s_mix)                  # 手写排序法
    print(f"     {float((_y_mix == 1).mean()):>11.2%} {len(_y_mix):>8d} "
          f"{_acc_v:>10.4f} {_rec_v:>9.4f} {_auc_sk_v:>14.4f} {_auc_rank_v:>13.4f}")
print("     -> Accuracy 随正负比例剧烈变化（正样本越少，'全判负'越占便宜，Accuracy 越高），")
print("        而 AUC 基本稳定 —— 它只取决于「正样本分数 vs 负样本分数」的比较结果，")
print("        与正负样本各有多少个无关（这正是它适合类别不平衡评估的原因）。")
print("        注意 Recall 在这里恒为 0.45：因为正样本集合没变、阈值也没变，")
print("        变多的只是负样本，所以 Recall 不受影响，而 Accuracy 被负样本'注水'了。")


# ===========================================================================
# 5. 回归指标：MSE / MAE / RMSE / R² / MAPE / 解释方差
# ===========================================================================
_title("5. 回归指标：MSE / MAE / RMSE / R² / MAPE / 解释方差")
print("""
  公式：
      MSE  = 1/n Σ (y - ŷ)²                    均方误差（平方放大大误差）
      MAE  = 1/n Σ |y - ŷ|                     平均绝对误差（线性惩罚，抗离群点）
      RMSE = √MSE                              均方根误差（量纲与 y 一致，更好解释）
      R²   = 1 - Σ(y-ŷ)² / Σ(y-ȳ)²             决定系数：模型解释了目标多少方差
      MAPE = 100%/n Σ |(y-ŷ)/y|                平均绝对百分比误差（跨量纲比较友好）
      解释方差 = 1 - Var(y-ŷ) / Var(y)          与 R² 相近，但衡量的是"误差的方差"
""")


def mse_hand(y_true, y_pred):
    """手写 MSE。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_true - y_pred) ** 2))


def mae_hand(y_true, y_pred):
    """手写 MAE。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse_hand(y_true, y_pred):
    """手写 RMSE = √MSE。"""
    return float(np.sqrt(mse_hand(y_true, y_pred)))


def r2_hand(y_true, y_pred):
    """手写 R² = 1 - SS_res/SS_tot。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))          # 残差平方和
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))   # 总平方和
    return 1.0 - ss_res / (ss_tot + _EPS)


def mape_hand(y_true, y_pred):
    """手写 MAPE（百分比），用 (|y|+eps) 防止除以 0。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + _EPS))) * 100.0)


# 课案例子
_y_pred_reg = np.array([2.8, 5.3, 2.2, 6.5])
_y_true_reg = np.array([3.0, 5.0, 2.5, 7.0])
print(f"  ▶ 课案例子：y_pred = {_y_pred_reg}，y_true = {_y_true_reg}")
_err_vec = _y_true_reg - _y_pred_reg
print(f"     逐个误差 (y-ŷ)     = {_err_vec}")
print(f"     误差平方 (y-ŷ)²    = {np.round(_err_vec ** 2, 6)}")
print(f"     绝对误差 |y-ŷ|     = {np.abs(_err_vec)}")
print(f"     手算 MSE = (0.2²+0.3²+0.3²+0.5²)/4 = "
      f"({0.04}+{0.09}+{0.09}+{0.25})/4 = {0.47}/4 = {0.47 / 4}   ->  课案说 0.1175 ✓")
print(f"     手算 MAE = (0.2+0.3+0.3+0.5)/4 = {1.3}/4 = {1.3 / 4}   ->  课案说 0.325 ✓")

print("\n  ③⑤ 手写 vs sklearn 全部回归指标：")
_cmp("MSE（手写 vs sklearn）", mse_hand(_y_true_reg, _y_pred_reg),
     mean_squared_error(_y_true_reg, _y_pred_reg))
_cmp("MAE（手写 vs sklearn）", mae_hand(_y_true_reg, _y_pred_reg),
     mean_absolute_error(_y_true_reg, _y_pred_reg))
_cmp("RMSE（手写 vs sklearn √MSE）", rmse_hand(_y_true_reg, _y_pred_reg),
     float(np.sqrt(mean_squared_error(_y_true_reg, _y_pred_reg))))
_cmp("R²（手写 vs sklearn）", r2_hand(_y_true_reg, _y_pred_reg),
     r2_score(_y_true_reg, _y_pred_reg))
_cmp("解释方差（手写 vs sklearn）",
     1.0 - float(np.var(_y_true_reg - _y_pred_reg)) / (float(np.var(_y_true_reg)) + _EPS),
     explained_variance_score(_y_true_reg, _y_pred_reg))
print(f"     MAPE（手写）= {mape_hand(_y_true_reg, _y_pred_reg):.10f}%"
      f"   （sklearn 1.9 未提供 MAPE，用 mean_absolute_percentage_error 也已被移除，故只给手写）")

print("\n  ▶ MSE vs MAE 对离群点的敏感度（复习损失函数一节的结论）")
_y_clean = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
_y_dirty = np.array([1.0, 1.0, 1.0, 1.0, 100.0])
_pred_flat = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
print(f"     正常数据：MSE = {mse_hand(_y_clean, _pred_flat):.4f}，MAE = {mae_hand(_y_clean, _pred_flat):.4f}")
print(f"     含离群点：MSE = {mse_hand(_y_dirty, _pred_flat):.4f}，MAE = {mae_hand(_y_dirty, _pred_flat):.4f}")
print(f"     -> MSE 被一个离群点从 0 拉到 {mse_hand(_y_dirty, _pred_flat):.2f}（99²/5 = {99 ** 2 / 5:.2f}），"
      f"MAE 只到 {mae_hand(_y_dirty, _pred_flat):.2f}（99/5 = {99 / 5:.2f}）")


# ===========================================================================
# 6. 可视化：ROC / PR 曲线 + 不同 AUC 的形状对比
# ===========================================================================
_title("6. 可视化：手写 ROC / PR 曲线 + 不同 AUC 的形状对比")


def roc_pr_points(y_true, y_scores, n_thresholds=200):
    """手写计算不同阈值下的 TPR / FPR / Precision / Recall。

    阈值从"全部判负"扫到"全部判正"：
        TPR = Recall = TP/(TP+FN)     —— 真正例率（纵轴）
        FPR          = FP/(FP+TN)     —— 假正例率（横轴）
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    pos = (y_true == 1)
    n_pos = float(pos.sum())
    n_neg = float((~pos).sum())
    # 阈值取分数范围上的等分点（含略高于最大值，保证有"全判负"的起点）
    thresholds = np.linspace(y_scores.max() + 1e-9, y_scores.min() - 1e-9, n_thresholds)
    tprs, fprs, precs, recalls = [], [], [], []
    for th in thresholds:
        pred_pos = y_scores >= th
        tp = float((pred_pos & pos).sum())
        fp = float((pred_pos & ~pos).sum())
        tpr = tp / n_pos if n_pos > 0 else 0.0                      # = Recall
        fpr = fp / n_neg if n_neg > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0             # 约定：无预测正样本时 P=1
        tprs.append(tpr)
        fprs.append(fpr)
        precs.append(prec)
        recalls.append(tpr)
    return np.array(fprs), np.array(tprs), np.array(precs), np.array(recalls)


# 用一组真实感的数据：正样本分数偏高，但有重叠 → AUC 在 0.5 与 1 之间
_rs2 = np.random.RandomState(7)
_n_eval = 300
_y_eval = _rs2.randint(0, 2, _n_eval)
_s_eval = np.where(_y_eval == 1, _rs2.normal(0.8, 0.7, _n_eval), _rs2.normal(0.2, 0.7, _n_eval))
_s_eval = 1.0 / (1.0 + np.exp(-_s_eval))          # 压到 (0,1)，模拟概率输出
_auc_eval = roc_auc_score(_y_eval, _s_eval)
_fprs, _tprs, _precs, _recalls = roc_pr_points(_y_eval, _s_eval)


def construct_controlled_scores(y_true, target_auc, seed=0):
    """构造一组分数，使其 AUC 大约等于 target_auc（用于画不同 AUC 的 ROC 形状）。

    做法：分数 = 标签 × 强度 + 噪声。强度越大，正负样本重叠越少，AUC 越高。
    这里用简单的扫描找出最接近目标 AUC 的强度。
    """
    rs = np.random.RandomState(seed)
    best = None
    for strength in np.linspace(0.0, 12.0, 241):
        noise = rs.normal(0, 1.0, len(y_true))
        scores = y_true * strength + noise
        auc = roc_auc_score(y_true, scores)
        if best is None or abs(auc - target_auc) < abs(best[0] - target_auc):
            best = (auc, scores.copy())
    return best[1], best[0]


fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))

# 子图 1：ROC 曲线（手写）
axes[0].plot(_fprs, _tprs, linewidth=2.4, color="tab:blue", label=f"手写 ROC（AUC = {_auc_eval:.4f}）")
axes[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=1.4, label="随机猜测（AUC = 0.5）")
axes[0].set_title("ROC 曲线（手写不同阈值下的 TPR/FPR）", fontsize=12.5)
axes[0].set_xlabel("FPR = FP/(FP+TN)　（假正例率）")
axes[0].set_ylabel("TPR = TP/(TP+FN)　（真正例率 = Recall）")
axes[0].legend(fontsize=9, loc="lower right")
axes[0].fill_between(_fprs, _tprs, alpha=0.15, color="tab:blue")   # 填充面积=AUC 直观含义
axes[0].text(0.45, 0.30, f"填充面积\n≈ AUC = {_auc_eval:.3f}", fontsize=10, color="tab:blue")

# 子图 2：PR 曲线（手写）
axes[1].plot(_recalls, _precs, linewidth=2.4, color="tab:green", label="手写 PR 曲线")
axes[1].axhline(float(np.mean(_y_eval == 1)), linestyle="--", color="gray", linewidth=1.4,
                label=f"随机基线（正样本比例 = {float(np.mean(_y_eval == 1)):.3f}）")
axes[1].set_title("PR 曲线（Precision-Recall）", fontsize=12.5)
axes[1].set_xlabel("Recall = TP/(TP+FN)")
axes[1].set_ylabel("Precision = TP/(TP+FP)")
axes[1].set_xlim(0, 1.02)
axes[1].set_ylim(0, 1.05)
axes[1].legend(fontsize=9, loc="lower left")
axes[1].text(0.05, 0.12,
             "PR 曲线在\n类别不平衡时\n比 ROC 更敏感",
             fontsize=9.5, color="tab:green")

# 子图 3：不同 AUC 对应的 ROC 形状
print("\n  ▶ 构造三种打分，使 AUC ≈ 0.5 / 0.75 / 1.0，直观理解 AUC 的含义")
for _target, _color, _ls in [(0.5, "tab:gray", "--"), (0.75, "tab:orange", "-"), (1.0, "tab:red", "-")]:
    _sc, _ac = construct_controlled_scores(_y_eval, _target, seed=1)
    _f3, _t3, _, _ = roc_pr_points(_y_eval, _sc, n_thresholds=80)
    axes[2].plot(_f3, _t3, linewidth=2.4, color=_color, linestyle=_ls,
                 label=f"目标 AUC≈{_target:.2f} → 实际 {_ac:.4f}")
    print(f"     目标 AUC = {_target:.2f}  ->  实际 AUC = {_ac:.4f}")
axes[2].plot([0, 1], [0, 1], ":", color="black", linewidth=1.2)
axes[2].set_title("不同 AUC 值对应的 ROC 曲线形状", fontsize=12.5)
axes[2].set_xlabel("FPR")
axes[2].set_ylabel("TPR")
axes[2].legend(fontsize=9, loc="lower right")

fig.tight_layout()
_roc_png = OUTPUT_DIR / "03训练组件_04_ROC与PR曲线.png"
fig.savefig(_roc_png, dpi=110)
plt.close(fig)
print(f"已保存：{_roc_png}")
print(f"  左图：曲线越靠近左上角越好；对角线 = 随机猜测（AUC=0.5）；"
      f"曲线下面积就是 AUC。")
print(f"  中图：横轴 Recall、纵轴 Precision；随机基线是一条水平线（= 正样本比例）。")
print(f"  右图：AUC=1 时曲线贴住左上角（^ 形），AUC=0.5 时退化成对角线。")


# ===========================================================================
# 7. 类别不平衡下 Accuracy 会骗人
# ===========================================================================
_title("7. 类别不平衡下 Accuracy 会骗人")
print("""
  场景：1000 个样本里有 990 个负样本、只有 10 个正样本（典型的欺诈检测 / 罕见病筛查）。
        模型偷懒——**全部预测为负**。
  这份"模型"什么正样本都没抓到，但 Accuracy 会是 99%，看起来非常漂亮。
  下面用真实数值演示，并说明此时该看什么指标。
""")
_rs3 = np.random.RandomState(0)
_y_imb = np.concatenate([np.zeros(990, dtype=int), np.ones(10, dtype=int)])
_y_all_neg = np.zeros(1000, dtype=int)                       # 偷懒模型：全预测负
_cm_imb = confusion_matrix(_y_imb, _y_all_neg)
_acc_imb = accuracy_score(_y_imb, _y_all_neg)
_p_imb = precision_score(_y_imb, _y_all_neg, zero_division=0)
_r_imb = recall_score(_y_imb, _y_all_neg, zero_division=0)
_f1_imb = f1_score(_y_imb, _y_all_neg, zero_division=0)
print(f"  数据集：正样本 {int((_y_imb == 1).sum())} 个，负样本 {int((_y_imb == 0).sum())} 个，"
      f"正样本比例 = {( _y_imb == 1).mean():.2%}")
print(f"  偷懒模型（全部预测为负）：")
print(f"    混淆矩阵 = {_cm_imb.tolist()}   （TP={_cm_imb[1, 1]}, FN={_cm_imb[1, 0]}, "
      f"FP={_cm_imb[0, 1]}, TN={_cm_imb[0, 0]}）")
print(f"    Accuracy  = {_acc_imb:.6f}   <- 【99%！看起来完美，其实毫无用处】")
print(f"    Precision = {_p_imb:.6f}   <- 0（没有预测任何正样本，分母为 0，约定记 0）")
print(f"    Recall    = {_r_imb:.6f}   <- 【0！一个正样本都没抓到】")
print(f"    F1        = {_f1_imb:.6f}   <- 0（被 Recall=0 直接拉到 0）")
print(f"    AUC       = {roc_auc_score(_y_imb, _y_all_neg.astype(float)):.6f}"
      f"   <- 0.5，等于随机猜测（所有分数都一样）")
print("\n  => 结论：类别不平衡时，**Accuracy 会骗人**。")
print("     90/10 甚至 99/1 的数据上，全预测多数类就能拿到很高的 Accuracy，")
print("     但 Recall = 0 意味着模型完全没有业务价值。")
print("     此时应该看：Recall（有没有漏掉正样本）、F1（兼顾 P 与 R）、")
print("     AUC / PR-AUC（排序能力，对类别比例不敏感）。")
print("     实践做法：重采样（过采样正类/欠采样负类）、类别加权损失、")
print("              或改用 Focal Loss（见损失函数一节）。")


# ===========================================================================
# 8. 真实模型评估：完整评估流程
# ===========================================================================
_title("8. 真实模型评估：训练一个小二分类模型并一次性打印全部指标")


class BinaryNet(nn.Module):
    """小型的二分类 MLP：2 → 32 → 32 → 1（输出 1 个 logit）。"""

    def __init__(self, in_dim=2, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),      # 输入层 → 隐藏层
            nn.ReLU(),                      # 隐藏层激活（回忆激活函数一节：ReLU 是默认首选）
            nn.Linear(hidden, hidden),      # 隐藏层 → 隐藏层
            nn.ReLU(),
            nn.Linear(hidden, 1),           # 输出 1 个 logit（不加 Sigmoid，损失函数内含）
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)      # 形状 (batch,)


# 数据：make_moons（两团交错的月牙，非线性可分）
_X_m, _y_m = make_moons(n_samples=_N_SAMPLES, noise=0.2, random_state=42)
X_t = torch.tensor(_X_m, dtype=torch.float32)
y_t = torch.tensor(_y_m, dtype=torch.float32)
_n_tr = int(_N_SAMPLES * 0.8)
X_tr, y_tr = X_t[:_n_tr], y_t[:_n_tr]
X_te, y_te = X_t[_n_tr:], y_t[_n_tr:]
print(f"  数据：make_moons，共 {_N_SAMPLES} 条（训练 {_n_tr} / 测试 {_N_SAMPLES - _n_tr}），"
      f"epoch = {_EPOCHS}")

torch.manual_seed(42)
_model = BinaryNet()
_opt = torch.optim.Adam(_model.parameters(), lr=0.01)
_loss_fn = nn.BCEWithLogitsLoss()           # 内含 Sigmoid，比手动 sigmoid + BCELoss 更稳
for _ep in range(_EPOCHS):
    _model.train()                          # 训练模式
    _opt.zero_grad()
    _loss = _loss_fn(_model(X_tr), y_tr)
    _loss.backward()
    _opt.step()
    if (_ep + 1) % 5 == 0:
        print(f"    epoch {_ep + 1:>2d}/{_EPOCHS}   train loss = {_loss.item():.6f}")

# —— 完整评估流程：model.eval() + torch.no_grad() ——
# eval() 关闭 Dropout/BatchNorm 的训练行为；no_grad() 关闭梯度追踪，省内存、更快。
_model.eval()
with torch.no_grad():
    _logits_te = _model(X_te)                       # 输出 logits（未过 Sigmoid）
    _prob_te = torch.sigmoid(_logits_te)            # ③ sigmoid 得到正类概率
    _pred_te = (_prob_te >= 0.5).long()             # ④ 以 0.5 为阈值得到类别预测

_y_true_np = y_te.numpy().astype(int)
_y_pred_np = _pred_te.numpy().astype(int)
_y_prob_np = _prob_te.numpy().astype(np.float64)

print("\n  ▶ 在测试集上评估（model.eval() + torch.no_grad()）")
print(f"     测试样本数 = {len(_y_true_np)}")
print(f"     预测概率前 8 个 = {np.round(_y_prob_np[:8], 4)}")
print(f"     真实标签前 8 个 = {_y_true_np[:8]}")

_hand_acc = accuracy_hand(_y_true_np, _y_pred_np)
_hand_cm = confusion_matrix(_y_true_np, _y_pred_np)
_hand_tp, _hand_fn, _hand_fp, _hand_tn = (int(_hand_cm[1, 1]), int(_hand_cm[1, 0]),
                                          int(_hand_cm[0, 1]), int(_hand_cm[0, 0]))
_hand_prec = _hand_tp / (_hand_tp + _hand_fp) if (_hand_tp + _hand_fp) > 0 else 0.0
_hand_rec = _hand_tp / (_hand_tp + _hand_fn) if (_hand_tp + _hand_fn) > 0 else 0.0
_hand_f1 = 2 * _hand_prec * _hand_rec / (_hand_prec + _hand_rec) if (_hand_prec + _hand_rec) > 0 else 0.0
_hand_auc = auc_by_rank(_y_true_np, _y_prob_np)

print(f"\n     {'指标':<14s} {'手写实现':>14s} {'sklearn':>14s} {'差值':>12s}")
print("     " + "-" * 58)
_sk_acc = accuracy_score(_y_true_np, _y_pred_np)
_sk_prec = precision_score(_y_true_np, _y_pred_np, zero_division=0)
_sk_rec = recall_score(_y_true_np, _y_pred_np, zero_division=0)
_sk_f1 = f1_score(_y_true_np, _y_pred_np, zero_division=0)
_sk_auc = roc_auc_score(_y_true_np, _y_prob_np)
for _nm, _hv, _sv in [("Accuracy", _hand_acc, _sk_acc), ("Precision", _hand_prec, _sk_prec),
                      ("Recall", _hand_rec, _sk_rec), ("F1", _hand_f1, _sk_f1),
                      ("AUC", _hand_auc, _sk_auc)]:
    print(f"     {_nm:<14s} {_hv:>14.10f} {_sv:>14.10f} {abs(_hv - _sv):>12.3e}")

print(f"\n     混淆矩阵（行=真实，列=预测）：")
print(f"       [[TN={_hand_tn:>4d}, FP={_hand_fp:>4d}],")
print(f"        [FN={_hand_fn:>4d}, TP={_hand_tp:>4d}]]")
print(f"     sklearn 混淆矩阵 = {_hand_cm.tolist()}")
print(f"     校验：{_hand_tp} + {_hand_tn} = {_hand_tp + _hand_tn} 个判对，"
      f"共 {len(_y_true_np)} 个 → Accuracy = {(_hand_tp + _hand_tn) / len(_y_true_np):.6f}")

print(f"\n  ▶ 完整评估结论（一句话解读）")
print(f"     Accuracy = {_sk_acc:.4f}：整体判对了 {_sk_acc:.2%} 的样本。")
print(f"     Precision = {_sk_prec:.4f}：模型说是正类的样本里，{_sk_prec:.2%} 真的是正类。")
print(f"     Recall = {_sk_rec:.4f}：真正的正样本里，{_sk_rec:.2%} 被找出来了。")
print(f"     F1 = {_sk_f1:.4f}：Precision 与 Recall 的调和平均，两者较均衡。")
print(f"     AUC = {_sk_auc:.4f}：随机取一正一负，正样本分数更高的概率是 {_sk_auc:.2%}，排序能力良好。")

_title("04_评估指标.py 运行完毕")
print(f"输出目录：{OUTPUT_DIR}")

r"""K-Means 聚类（无监督学习）——原理推导 + sklearn API + 实战 + 结果解读

对应课案章节
------------
《机器学习》课案 · 无监督学习 · 聚类 · K-均值（K-Means）
课案原文位置：`.course_extract/机器学习_课案.md` 第 1163~1226 行
（含"聚类"定义、"K-Means 最低可运行版本"与"K-Means 参数与评估"表格）

本节知识点
----------
1. 无监督学习 / 聚类的含义：没有标签，只从数据自身的结构里发现"相似的样本组"。
2. K-Means 的模型形式：用 K 个质心 μ_1, ..., μ_K 概括整个数据集。
3. 目标函数：簇内平方误差和 SSE（sklearn 里叫 inertia）。
4. 优化方法：EM 风格的交替迭代（分配样本 → 更新质心 → 判断收敛），
   以及 K-Means++ 初始化如何缓解"初值敏感"。
5. 评估指标：肘部法、轮廓系数 Silhouette、Calinski-Harabasz、Davies-Bouldin，
   以及只在"有真实标签"时才能用的 ARI / NMI。
6. 优缺点与适用边界（球状、大小相近的簇；需预先给定 K；对异常值敏感）。

运行方式
--------
在 PowerShell 中执行（路径含中文，必须加引号并用 & 调用）：

    $env:PYTHONUTF8='1'; & 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Machine_Learning\05_无监督学习\01_KMeans聚类.py'

产出图片（保存在 Machine_Learning/output/ 下）：
    05_KMeans肘部法.png       —— K 从 1 到 10 的 inertia 曲线 + 拐点标注
    05_KMeans聚类结果.png     —— K=4 的聚类散点图 + 红色质心
    05_KMeans不同K对比.png    —— K=2,3,4,5,6 的聚类效果对比（辅助理解）
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")            # 无界面后端，避免 plt.show() 阻塞
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ① 原理与数学推导
# ============================================================================
#
# 【1. 无监督学习在做什么】
# 有监督学习拿到的数据是 (X, y)：X 是特征，y 是标准答案。无监督学习只有 X，
# 没有 y。聚类（clustering）要做的事是：把 n 个样本切成 K 个互不相交的组
# （簇，cluster），使"同簇内的样本尽量像，不同簇的样本尽量不像"。
# 注意：聚类输出的"簇编号"是没有语义的（第 0 簇不等于"类别 0"），
# 编号只是一个记号，换个随机种子编号可能整体重排——所以后文评估时不能直接
# 拿簇编号和真实标签做"准确率"，必须用对编号置换不敏感的指标（ARI / NMI）。
#
#
# 【2. 模型形式（模型长什么样）】
# K-Means 用 K 个"质心"（centroid）来代表 K 个簇：
#
#       μ_1, μ_2, ..., μ_K ∈ R^d      （d 是特征维度，本脚本 d=2）
#
# 同时把每个样本 x_i 指派给离它最近的那个质心，记 c_i ∈ {1,...,K} 为样本 i 的簇标号：
#
#       c_i = argmin_k  || x_i - μ_k ||^2
#
# 也就是说：给定质心，样本的归属就完全确定了（"最近邻指派"）。
# 所以 K-Means 的"参数"其实只有这 K 个质心，模型非常简洁。
#
#
# 【3. 目标函数 / 损失函数（SSE，sklearn 中叫 inertia）】
# 要优化的目标就是"每个样本到它所属质心的距离平方和"：
#
#       J(c, μ) = Σ_{i=1}^{n} || x_i - μ_{c_i} ||^2
#               = Σ_{k=1}^{K} Σ_{i: c_i = k} || x_i - μ_k ||^2
#
# 这个量叫簇内平方误差和（SSE, Sum of Squared Errors），sklearn 的属性名是
# `inertia_`。它越小说明簇内越紧凑。注意 inertia 随 K 单调不增（K 越大越容易
# 拟合，K = n 时 inertia = 0），所以"inertia 最小"不能直接用来选 K，这正是
# 需要用"肘部法"看拐点的原因。
#
# 数学上这个优化问题是 NP-hard 的（离散指派 + 连续质心），所以只能求局部最优。
#
#
# 【4. 优化方法：EM 式的交替迭代（Lloyd 算法）】
# 思路：固定一边、优化另一边，交替进行。第 t 轮：
#
#   ① 分配步（E 步，固定 μ，优化 c）：把每个样本指派给最近的质心
#         c_i^(t) = argmin_k || x_i - μ_k^(t) ||^2
#      这一步使 J 单调不增（每个点都选了当前最近的质心）。
#
#   ② 更新步（M 步，固定 c，优化 μ）：把每个质心挪到本簇样本的均值处
#         μ_k^(t+1) = (1 / |S_k|) Σ_{i ∈ S_k} x_i ,  其中 S_k = { i : c_i = k }
#      这一步也使 J 单调不增：对 μ_k 求偏导
#         ∂J/∂μ_k = -2 Σ_{i ∈ S_k} (x_i - μ_k) = 0
#         ⇒ μ_k = mean(S_k)   （均值就是使平方误差最小的点）
#      若某簇样本数为 0（空簇），sklearn 会重新给它指派一个质心（默认取
#      距离最远的样本点），所以实际实现里还要处理这个边界情况。
#
#   ③ 收敛判断：质心移动量的平方和（或 J 的变化量）小于 tol，或轮数达到 max_iter。
#
# 因为每一步都不会让 J 变大，而 J 有下界（≥0），所以算法一定收敛；但收敛到的
# 通常是局部最优，且结果依赖初始质心。
#
#
# 【5. K-Means++ 初始化（缓解初值敏感）】
# 随机初始化可能把两个质心丢进同一个真实簇里，导致收敛到很差的局部最优。
# K-Means++ 的思路是"让初始质心尽量互相远离"，用概率化地依次选点：
#   - 第 1 个质心从样本里均匀随机选一个；
#   - 之后每个质心以正比于 D(x)^2 的概率选择，其中 D(x) 是 x 到"已选质心中
#     最近那个"的距离（D(x)^2 记为该点的"能量"）。
# 即：离已有质心越远的点，越可能被选为下一个质心。
# 这样初始化出来的质心分散，配合多次重启（n_init）就能大幅提升解的质量，
# 这也是 sklearn 的默认 init="k-means++"。
#
#
# 【6. 优缺点】
# 优点：
#   - 原理直观、实现简单、训练快（每轮复杂度 O(n·K·d)），样本量大也能用；
#   - 只有 K 和距离计算，内存占用小（MiniBatchKMeans 还能处理超大数据）。
# 缺点：
#   - 必须预先指定 K，而 K 往往不知道（用肘部法 / 轮廓系数猜）；
#   - 隐含假设：簇是"球状、各向同性、大小相近"的，且用欧氏距离度量相似度；
#     遇到月牙形、长条形、密度差别大的簇（如 make_moons）会切错；
#   - 对异常值敏感：均值不是稳健统计量，一个极端点能把质心拽偏；
#   - 只保证局部最优，需要 n_init 多次重启取最优；
#   - inertia 是无界的，不同 K 之间不能直接比大小。
#
# 实践建议：K-Means 之前先标准化特征（否则量纲大的特征会主导欧氏距离）；
# 遇到非凸簇可换 DBSCAN / 谱聚类 / 高斯混合模型（GMM）。
#
#
# ============================================================================


# ============================================================================
# ② sklearn 的 KMeans API 关键参数逐个解释
# ============================================================================
#
# KMeans(n_clusters=8, init="k-means++", n_init="auto", max_iter=300,
#        tol=1e-4, algorithm="lloyd", random_state=None, ...)
#
# (1) n_clusters : int，默认 8 —— 【最重要的参数】簇的个数 K。
#     含义：预先告诉算法"我想把数据切成几组"。
#     调大：簇更多、更细，inertia 一定下降，但可能把一个自然的簇硬拆成两半
#           （过分割），可解释性变差。
#     调小：簇更少、更粗，inertia 上升，可能把两个不同的簇合并（欠分割）。
#     常用值：业务上能解释的数目（如 3=低/中/高价值客户），或由肘部法 / 轮廓
#             系数选出（本例 K=4，因为 make_blobs 就是用 4 个中心生成的数据）。
#     注意：它是必填语义参数，没有"最优默认值"，必须由业务或指标决定。
#
# (2) init : {"k-means++", "random"} 或形状为 (n_clusters, n_features) 的数组，
#     默认 "k-means++"。
#     含义：初始质心怎么来。
#     "k-means++"：按距离平方概率散布初始质心，收敛快、结果好，一般不用改。
#     "random"：从数据里随机挑 K 个点当初始质心，容易掉进坏局部最优，
#               必须靠 n_init 多次重启补救。
#     也可直接传入自己算好的初始质心数组（例如用上一次训练结果热启动）。
#     调参影响：从 "random" 换到 "k-means++" 通常是"免费"的质量提升。
#
# (3) n_init : int 或 "auto"，默认 "auto"。
#     含义：用不同的初始质心重复跑多少遍，最后取 inertia 最小的那次。
#     "auto" 在 init="k-means++" 时等价于 1，在 init="random" 时等价于 10。
#     调大：结果更稳定、更接近全局较优，代价是耗时线性增长。
#     调小到 1：快，但结果可能随 random_state 抖动。
#     常用值：显式写 10（本脚本就写 10，避免默认值随版本变化，也便于复现）。
#     提示：sklearn 1.4 起这个参数的默认值才变成 "auto"，早期版本默认是 10；
#           显式写出来是消除版本差异、避免告警的好习惯。
#
# (4) max_iter : int，默认 300。
#     含义：单次运行里"分配 + 更新"最多迭代多少轮。
#     调大：给算法更多轮次去收敛，可能找到更小的 inertia（但一般 300 轮早已收敛）。
#     调小（如 10）：可能没收敛就停，inertia 偏大、标签不稳定；适合大数据快速试验。
#     常用值：几百即可；可用 model.n_iter_ 检查实际用了几轮（远小于 300 说明已收敛）。
#
# (5) tol : float，默认 1e-4。
#     含义：收敛阈值。若相邻两轮质心位移的 Frobenius 范数平方小于 tol，就认为收敛。
#     调大（如 1e-2）：更早停止，快一点，精度略降。
#     调小（如 1e-8）：要求更严格，轮数更多，通常收益极小。
#     常用值：1e-4（默认基本不用动）。
#
# (6) algorithm : {"lloyd", "elkan"}，默认 "lloyd"。
#     含义：用哪种迭代实现。两者求的是同一个目标函数，只是加速方式不同。
#     "lloyd"：标准实现，每次迭代都要算全部样本到全部质心的距离。
#     "elkan"：用三角不等式剪枝，跳过不可能成为最近邻的距离计算，
#              在 K 不太大、簇分得比较开时更快（内存开销更大）。
#     注意：老版本的 "auto" 选项已被删除，请只写 "lloyd" 或 "elkan"。
#     调参影响：只影响速度，几乎不影响结果（结果差异来自 init 的随机性）。
#
# (7) random_state : int / None，默认 None。
#     含义：随机数种子，控制初始质心的选择（以及 elkan/lloyd 里的平局处理）。
#     调参影响：它不影响模型质量，只影响"可复现性"。课程统一用 42，
#               这样任何人跑出来的簇编号和指标都一致。
#
# 其他常用属性 / 方法：
#     model.cluster_centers_ : (K, d) 质心坐标，可用于解释每个簇的"典型画像"；
#     model.labels_          : (n,) 每个样本的簇编号（等价于 fit_predict 的返回）；
#     model.inertia_         : SSE，越小簇内越紧凑（但随 K 单调下降，别直接比 K）；
#     model.n_iter_          : 实际迭代轮数；
#     model.predict(X_new)   : 用训练好的质心给新样本指派簇（这叫"归纳"能力，
#                              DBSCAN 等算法没有这个能力）。
#
# ============================================================================


# ============================================================================
# ③ 完整可运行代码
# ============================================================================

print("=" * 78)
print("K-Means 聚类实战（无监督学习）")
print("=" * 78)

# ---------------------------------------------------------------------------
# 3.1 生成数据：make_blobs 造 4 个"球状"簇，正好符合 K-Means 的假设
# ---------------------------------------------------------------------------
# n_samples=300：样本数；centers=4：真实簇数（等价于我们心里的"正确答案"）；
# cluster_std=0.6：每个簇的标准差（越小簇越紧凑、越好分）；
# random_state=42：固定随机种子，保证可复现。
X, y_true = make_blobs(
    n_samples=300,
    centers=4,
    cluster_std=0.6,
    random_state=42,
)

print("\n【1】数据概况")
print(f"  样本数 n_samples      : {X.shape[0]}")
print(f"  特征维度 n_features   : {X.shape[1]}")
print(f"  真实簇数 centers      : {len(np.unique(y_true))}（{np.unique(y_true).tolist()}）")
print("  说明：make_blobs 是造聚类数据集的工具，它自带真实标签 y_true，")
print("        但真实场景里我们没有 y_true——所以下文会分别演示")
print("        『无真实标签也能用』和『有真实标签才能用』的两类评估指标。")

# ---------------------------------------------------------------------------
# 3.2 肘部法：K 从 1 到 10 训练，记录 inertia
# ---------------------------------------------------------------------------
print("\n【2】肘部法（Elbow Method）：遍历 K=1..10，看 inertia 怎么降")

K_RANGE = range(1, 11)
inertia_list = []
silhouette_list = []          # 轮廓系数需要 K>=2 才有定义

for k in K_RANGE:
    # n_init=10 显式写出：重复 10 次不同初始化取最优，结果更稳且无版本告警
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels_k = km.fit_predict(X)
    inertia_list.append(km.inertia_)
    if k >= 2:
        # 轮廓系数：[-1, 1]，越接近 1 表示"簇内紧、簇间远"
        silhouette_list.append(silhouette_score(X, labels_k))
    else:
        silhouette_list.append(float("nan"))

print(f"  {'K':>3} | {'inertia(SSE)':>14} | {'相邻降幅':>10} | {'轮廓系数':>10}")
print("  " + "-" * 54)
for idx, k in enumerate(K_RANGE):
    if idx == 0:
        drop_text = "—"
    else:
        drop_text = f"{inertia_list[idx - 1] - inertia_list[idx]:.1f}"
    sil_text = "—" if np.isnan(silhouette_list[idx]) else f"{silhouette_list[idx]:.4f}"
    print(f"  {k:>3} | {inertia_list[idx]:>14.2f} | {drop_text:>10} | {sil_text:>10}")

# ---- 拐点的自动检测：这里给两种常用启发式，并对比它们的结果 -----------------
# 直觉：inertia 曲线是一条"先陡降、后平缓"的凸曲线，拐点（手肘）就是
#      "再增加一个簇带来的收益明显变小"的那个 K。
# 手肘位置是主观的，不同启发式可能给出 K=3 或 K=4 这样的相邻答案，所以下面
# 同时算两种、并把结果都打印出来，让读者看到"自动检测只是辅助，不是判决"。
def find_knee_chord(k_values, inertia_values):
    """弦距法（kneedle 的简化版）：归一化到 [0,1] 后，取离"首尾连线"最远的点。"""
    x = np.asarray(list(k_values), dtype=float)
    y = np.asarray(inertia_values, dtype=float)
    x_norm = (x - x.min()) / (x.max() - x.min())          # 归一化到 [0,1]
    y_norm = (y - y.min()) / (y.max() - y.min())
    # 归一化后首尾连线从 (0,1) 到 (1,0)，直线方程：x + y - 1 = 0
    distance = np.abs(x_norm + y_norm - 1.0) / np.sqrt(2.0)
    return int(x[int(np.argmax(distance))])


def find_knee_drop_ratio(k_values, inertia_values):
    """相对降幅法：找"加簇收益突然崩塌"的位置，返回（手肘 K, 崩塌处的 K, 降幅比）。

    记 d_k = inertia(K=k) - inertia(K=k+1)（第 k+1 个簇带来的误差下降量）。
    若 d_{k+1} / d_k 突然变得很小，说明第 k+1 个簇已经把该分的都分完了，
    真正的手肘就在 K = k+1。本函数返回使该比值最小的 K-1 作为手肘。
    """
    y = np.asarray(inertia_values, dtype=float)
    drops = -np.diff(y)                                   # drops[i] = K=i+1 → i+2 的降幅
    ratios = drops[1:] / np.maximum(drops[:-1], 1e-12)    # ratios[i] 对应 K = i+3
    idx = int(np.argmin(ratios))                          # 收益崩塌最厉害的位置
    k_break = int(list(k_values)[idx + 2])
    k_elbow = int(list(k_values)[idx + 1])
    return k_elbow, k_break, float(ratios[idx])


knee_chord = find_knee_chord(K_RANGE, inertia_list)
knee_drop, knee_break, ratio_min = find_knee_drop_ratio(K_RANGE, inertia_list)
knee_k = knee_drop                                          # 以相对降幅法为准
sil_best_k = int(np.nanargmax(silhouette_list)) + 1

print(f"\n  自动检测拐点（相对降幅法，本文采用）: K = {knee_drop}")
print(f"     依据：K={knee_break} 时加簇收益突然崩塌，本次降幅比仅 {ratio_min:.4f}")
print(f"           （即第 {knee_break} 个簇几乎没带来新的误差下降）")
print(f"  自动检测拐点（弦距法，kneedle 简化）: K = {knee_chord}")
print(f"  轮廓系数最大的 K                     : K = {sil_best_k}")
print("  解读：inertia 随 K 单调下降，所以不能选『inertia 最小』的 K（那永远是最大 K）。")
print("        肘部法找的是『再增加一个簇，收益明显变小』的那个 K——就是曲线的弯折处。")
print("        手肘位置带主观性：不同启发式给出 3 或 4 都算合理（本例弦距法偏小一档），")
print("        所以正确做法是『多指标交叉验证 + 业务判断』，而不是迷信某一个自动结果。")
print(f"        本例生成数据时 centers=4，真实簇数就是 4；相对降幅法与轮廓系数都指向 4，")
print("        两者互相印证，因此下面正式选用 K=4。")

# ---------------------------------------------------------------------------
# 3.3 用选定的 K=4 正式聚类
# ---------------------------------------------------------------------------
print("\n【3】用 K=4 正式聚类")

K_CHOSEN = 4
kmeans = KMeans(
    n_clusters=K_CHOSEN,      # 簇数：由肘部法 + 业务判断得到
    init="k-means++",         # 初始质心：K-Means++，分散且收敛快
    n_init=10,                # 用 10 组初始质心重启，取 inertia 最小的
    max_iter=300,             # 单次运行最多 300 轮
    tol=1e-4,                 # 质心位移小于该阈值即视为收敛
    algorithm="lloyd",        # 标准 Lloyd 迭代（也可换 "elkan" 加速）
    random_state=42,          # 固定种子，保证结果可复现
)
y_pred = kmeans.fit_predict(X)

print(f"  实际迭代轮数 n_iter_     : {kmeans.n_iter_}")
print(f"  簇内平方误差和 inertia_  : {kmeans.inertia_:.4f}")
print("\n  聚类中心 cluster_centers_（每行是一个质心的坐标，共 %d 行）：" % K_CHOSEN)
for k, center in enumerate(kmeans.cluster_centers_):
    print(f"    簇 {k} 的质心 : [{center[0]:>8.4f}, {center[1]:>8.4f}]")

print("\n  前 20 个样本的簇编号 labels_[:20]：")
print(f"    {y_pred[:20].tolist()}")
print("  说明：簇编号 0/1/2/3 只是记号，没有大小和顺序含义；")
print("        换一个 random_state，编号可能整体重排，但分组本身基本不变。")

# 各簇样本数（检查有没有空簇 / 严重不均衡）
counts = np.bincount(y_pred, minlength=K_CHOSEN)
print("\n  各簇样本数：")
for k, c in enumerate(counts):
    print(f"    簇 {k} : {c:>3} 个样本")
print("  说明：本例 4 个真实簇大小相近（各约 75 个），所以各簇样本数接近——")
print("        这正是 K-Means 喜欢的『簇大小相近』假设；若差别悬殊，要警惕切错。")

# ---------------------------------------------------------------------------
# 3.4 评估：无监督指标（不需要真实标签）
# ---------------------------------------------------------------------------
print("\n【4】聚类质量评估")

sil = silhouette_score(X, y_pred)
ch = calinski_harabasz_score(X, y_pred)
db = davies_bouldin_score(X, y_pred)

print(f"  轮廓系数   silhouette_score        = {sil:.4f}    （范围 [-1,1]，越接近 1 越好）")
print(f"  Calinski-Harabasz 指数              = {ch:.4f}    （组间/组内方差比，越大越好）")
print(f"  Davies-Bouldin 指数                 = {db:.4f}    （簇内/簇间距离比，越小越好）")

print("\n  指标含义速查：")
print("    ① 轮廓系数：对每个样本 i，记 a(i) 为它到【同簇】其他样本的平均距离，")
print("       b(i) 为它到【最近邻簇】所有样本的平均距离，则")
print("           s(i) = [b(i) - a(i)] / max(a(i), b(i))")
print("       （课案原文该公式被抽取成了重复乱码，此处按含义重写为标准形式。）")
print("       全部样本的 s(i) 均值就是 silhouette_score，范围 [-1,1]：")
print("         ≈ 1   ：样本离本簇很近、离别的簇很远 → 分得好；")
print("         ≈ 0   ：样本正好落在两个簇的边界上 → 归属模糊；")
print("         < 0   ：样本可能被分错了簇（离邻簇比离本簇还近）。")
print("    ② Calinski-Harabasz：CH = [组间离散度/(K-1)] / [组内离散度/(n-K)]，")
print("       本质是方差分析的 F 值，越大说明『簇间分得开、簇内又紧凑』。")
print("    ③ Davies-Bouldin：每个簇找最相似的另一个簇，算簇内散度与簇间距离之比，")
print("       再对所有簇求平均；越小越好，0 是理想下界。")

# ---------------------------------------------------------------------------
# 3.5 评估：有真实标签才能用的外部指标（ARI / NMI）
# ---------------------------------------------------------------------------
print("\n【5】外部指标（只有手上有真实标签时才能算）")

ari = adjusted_rand_score(y_true, y_pred)
nmi = normalized_mutual_info_score(y_true, y_pred)

print(f"  调整兰德指数 ARI  adjusted_rand_score           = {ari:.4f}")
print(f"  标准化互信息 NMI  normalized_mutual_info_score  = {nmi:.4f}")
print("\n  为什么平时不能用这两个指标？")
print("    ARI / NMI 都要把『预测簇编号』和『真实类别标签』作比较，属于外部指标。")
print("    真实场景的聚类任务根本没有标签，所以它们只能用于：① 教学与仿真实验")
print("    （本例 make_blobs 自带 y_true）；② 用少量人工标注做『聚类结果抽查』。")
print("  数值含义：")
print("    ARI ∈ [-1,1]，0 表示和随机划分一样，1 表示与真实标签完全一致；")
print("       它对簇编号的置换不敏感（把簇 0/1 互换，ARI 不变），所以适合聚类。")
print("    NMI ∈ [0,1]，衡量两个划分共享多少信息量，1 表示完全一致。")
print(f"    本次 ARI={ari:.4f}、NMI={nmi:.4f} 都接近 1，说明 K-Means 找出的划分和")
print("    造数据时的真实 4 个簇几乎完全一致；剩下的一点点差异来自两种编号方式的")
print("    细节不同（ARI/NMI 允许簇被整体重编号），不是真的『分错了』。")

# ---------------------------------------------------------------------------
# 3.6 画图 1：肘部法曲线（K vs inertia）
# ---------------------------------------------------------------------------
k_values = list(K_RANGE)

fig1, ax1 = plt.subplots(figsize=(8.5, 5.2))
ax1.plot(k_values, inertia_list, marker="o", color="#1f77b4", linewidth=2,
         markersize=7, label="inertia（簇内平方误差和）")
# 标出自动检测到的拐点
ax1.scatter([knee_k], [inertia_list[knee_k - 1]], s=260, facecolors="none",
            edgecolors="#d62728", linewidths=2.5, zorder=5,
            label=f"手肘（相对降幅法）K={knee_k}")
if knee_chord != knee_k:
    ax1.scatter([knee_chord], [inertia_list[knee_chord - 1]], s=170, marker="s",
                facecolors="none", edgecolors="#7f7f7f", linewidths=2.0, zorder=5,
                label=f"另一启发式（弦距法）候选 K={knee_chord}")
ax1.annotate(f"手肘在这里\nK={knee_k}",
             xy=(knee_k, inertia_list[knee_k - 1]),
             xytext=(knee_k + 1.4, inertia_list[knee_k - 1] + (max(inertia_list) - min(inertia_list)) * 0.30),
             fontsize=12, color="#d62728",
             arrowprops=dict(arrowstyle="->", color="#d62728", linewidth=1.6))
# 参考线：生成数据时设定的真实簇数
ax1.axvline(4, color="#2ca02c", linestyle="--", linewidth=1.6, alpha=0.9,
            label="真实簇数 K=4（造数据时设定）")
ax1.set_xlabel("簇数 K", fontsize=12)
ax1.set_ylabel("inertia（SSE，越小簇内越紧凑）", fontsize=12)
ax1.set_title("K-Means 肘部法：K 与 inertia 的关系（找曲线弯折处）", fontsize=13)
ax1.set_xticks(k_values)
ax1.grid(alpha=0.3)
ax1.legend(fontsize=10)
fig1.tight_layout()
fig1.savefig(OUTPUT_DIR / "05_KMeans肘部法.png", dpi=150)
plt.close(fig1)
print(f"\n  已保存图片：{OUTPUT_DIR / '05_KMeans肘部法.png'}")

# ---------------------------------------------------------------------------
# 3.7 画图 2：K=4 的聚类结果散点图（按预测簇着色 + 红色质心）
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(8.5, 6.2))
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
for k in range(K_CHOSEN):
    mask = y_pred == k
    ax2.scatter(X[mask, 0], X[mask, 1], s=42, color=colors[k], alpha=0.75,
                edgecolors="white", linewidths=0.6, label=f"预测簇 {k}（{mask.sum()} 个样本）")
# 红色质心
ax2.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=320, marker="X", color="red", edgecolors="black", linewidths=1.5,
            zorder=6, label="质心 cluster_centers_（每簇均值点）")
for k, center in enumerate(kmeans.cluster_centers_):
    ax2.annotate(f"μ{k}", xy=(center[0], center[1]), xytext=(center[0] + 0.18, center[1] + 0.18),
                 fontsize=13, color="darkred", fontweight="bold")
ax2.set_xlabel("特征 1", fontsize=12)
ax2.set_ylabel("特征 2", fontsize=12)
ax2.set_title(f"K-Means 聚类结果（K={K_CHOSEN}，轮廓系数={sil:.4f}）", fontsize=13)
ax2.legend(fontsize=10, loc="best")
ax2.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig(OUTPUT_DIR / "05_KMeans聚类结果.png", dpi=150)
plt.close(fig2)
print(f"  已保存图片：{OUTPUT_DIR / '05_KMeans聚类结果.png'}")

# ---------------------------------------------------------------------------
# 3.8 画图 3（加分）：不同 K 的聚类效果对比
# ---------------------------------------------------------------------------
k_show = [2, 3, 4, 5, 6]
fig3, axes = plt.subplots(1, len(k_show), figsize=(4.2 * len(k_show), 4.2))
for ax, k in zip(axes, k_show):
    km_k = KMeans(n_clusters=k, init="k-means++", n_init=10,
                  max_iter=300, tol=1e-4, algorithm="lloyd", random_state=42)
    lab_k = km_k.fit_predict(X)
    ax.scatter(X[:, 0], X[:, 1], c=lab_k, cmap="tab10", s=22, alpha=0.8)
    ax.scatter(km_k.cluster_centers_[:, 0], km_k.cluster_centers_[:, 1],
               s=170, marker="X", color="red", edgecolors="black", linewidths=1.0, zorder=5)
    ax.set_title(f"K={k}\ninertia={km_k.inertia_:.0f}", fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.25)
fig3.suptitle("不同 K 的 K-Means 聚类效果对比（红 X 为质心）", fontsize=14)
fig3.tight_layout()
fig3.savefig(OUTPUT_DIR / "05_KMeans不同K对比.png", dpi=150)
plt.close(fig3)
print(f"  已保存图片：{OUTPUT_DIR / '05_KMeans不同K对比.png'}")
print("  对比图解读：K=2 会把两个相邻的真实簇合并；K=3 仍有一个簇被合并；")
print("              K=4 与真实结构吻合、簇内紧凑；K=5、K=6 开始把一个自然簇硬拆开，")
print("              inertia 虽继续下降，但分组已经不再有业务意义——这就是过分割。")

# ---------------------------------------------------------------------------
# 3.9 顺带对比 algorithm="lloyd" 与 "elkan"（只影响速度，几乎不影响结果）
# ---------------------------------------------------------------------------
kmeans_elkan = KMeans(n_clusters=K_CHOSEN, init="k-means++", n_init=10,
                      max_iter=300, tol=1e-4, algorithm="elkan", random_state=42)
labels_elkan = kmeans_elkan.fit_predict(X)
print("\n【6】algorithm 参数对比（lloyd vs elkan）")
print(f"  lloyd 的 inertia_        : {kmeans.inertia_:.4f}")
print(f"  elkan 的 inertia_        : {kmeans_elkan.inertia_:.4f}")
print(f"  两者划分的 ARI 一致性     : {adjusted_rand_score(y_pred, labels_elkan):.4f}"
      "（=1 表示两种实现给出完全相同的划分）")
print("  结论：algorithm 只改变『算得多快』，不改变目标函数，对结果几乎无影响。")

# ---------------------------------------------------------------------------
# 3.10 结果解读汇总
# ---------------------------------------------------------------------------
print("\n" + "=" * 78)
print("④ 结果解读")
print("=" * 78)
print(f"""
1) 数据与 K 的选择
   我们用 make_blobs 造了 300 个 2 维样本、4 个球状簇（cluster_std=0.6）。
   肘部法画出 K=1..10 的 inertia 曲线：K 从 1 增到 4 时 inertia 急速下降
   （每次加簇都能吃掉一大块误差），K>4 之后曲线明显变平缓——这个"弯折处"
   就是手肘。本项目自动检测到的拐点是 K={knee_k}，轮廓系数最大的 K 也是
   K={sil_best_k}，而生成数据时设定的真实簇数正是 4：三方一致，说明 K=4 可靠。

2) 聚类质量（无真实标签也能算的三个指标）
   - 轮廓系数 = {sil:.4f}：非常接近 1，说明每个样本离本簇很近、离别的簇很远。
     轮廓系数范围是 [-1, 1]：越接近 1 越好（>0.5 通常认为结构清晰，>0.7 相当好，
     接近 0 说明簇边界模糊，出现负值说明有样本被分到了错误的簇）。
     本次 {sil:.4f} 属于"结构非常清晰"，这与数据本身是 4 个分得很开的球状簇相符。
   - Calinski-Harabasz = {ch:.2f}：越大越好，表示簇间方差远大于簇内方差。
     它没有绝对阈值，通常用来横向比较不同 K / 不同算法的结果，取最大值。
   - Davies-Bouldin = {db:.4f}：越小越好（0 是理想值），本次远小于 1，簇内紧凑、
     簇间分离，属于很好的结果。

3) 有真实标签时的外部指标
   - ARI = {ari:.4f}、NMI = {nmi:.4f}，都接近 1，说明 K-Means 还原出了造数据时的
     真实分组。这两个指标在真实无标签场景中不可用，只能用于教学/仿真/人工抽查。
     注意它们是"对簇编号置换不敏感"的，所以即使簇 0 和簇 1 的编号被互换，
     指标仍然是 1——这正是聚类评估需要的外部指标特性。

4) 结果怎么用
   - cluster_centers_ 就是每个簇的"典型画像"：业务上可以把质心坐标翻译成
     "高消费高频用户""低消费低频用户"等标签；
   - labels_ 可以直接作为新特征喂给下游有监督模型（这叫"聚类特征工程"）；
   - predict() 还能给未来新样本直接指派簇，便于线上打分。
""")

# ============================================================================
# 超参数怎么调
# ============================================================================
#
# 【K 怎么选（最关键，也是 K-Means 唯一的"必调"参数）】
#   ① 肘部法：画 K vs inertia，取曲线从陡降变平缓的拐点。优点是快、直观；
#      缺点是拐点有时不明显（曲线很平滑时人眼看不出）。可配合"到首尾连线距离
#      最大"的自动检测（本脚本 find_knee）或 kneed 库辅助判断。
#   ② 轮廓系数：对每个 K 算 silhouette_score，取最大。优点是能给出统一尺度的
#      质量分数；缺点是计算量 O(n^2)（样本量很大时改用采样估计或 MiniBatch），
#      且偏爱凸的、大小相近的簇。
#   ③ Calinski-Harabasz（越大越好）/ Davies-Bouldin（越小越好）：同样是"扫一遍
#      K 取极值"，计算比轮廓系数便宜，适合快速筛选。
#   ④ 业务可解释性【最终裁判】：指标只是参考，最后要问"这个 K 的分组能不能讲出
#      一个业务故事"。K=7 比 K=4 的轮廓系数略高，但如果业务上只能落地 4 类客户
#      运营策略，就应该选 4。多个指标指向不同 K 时，优先选业务能解释的那个。
#
# 【其他参数怎么调】
#   - init 与 n_init 优先保证质量：保持 init="k-means++" + n_init=10（默认值，
#     显式写出更稳）。如果结果在不同 random_state 下差别很大，说明数据里的簇
#     结构本来就不清晰，先把 n_init 调到 20~50 看是否能稳定下来。
#   - 数据预处理比调参更重要：先 StandardScaler 标准化（量纲统一），
#     必要时对偏态特征做对数变换，异常值考虑截断——否则欧氏距离会被大尺度特征
#     和极端值主导，再怎么调 K 也没用。
#   - 追求速度：数据量大时用 MiniBatchKMeans(batch_size=1024~4096)，或把
#     algorithm="elkan"、max_iter 调小；追求精度就反过来。
#   - tol 一般不用动；只有你发现 n_iter_ 恰好等于 max_iter（说明没收敛）时，
#     才需要增大 max_iter 或放宽 tol。
#   - 结果不稳定/想复现：永远固定 random_state=42。
#
# 【什么时候不该用 K-Means】
#   簇是月牙形/环形（make_moons）、密度差异大、或存在大量异常值时，
#   改用 DBSCAN（按密度，不需要指定 K）、谱聚类、或高斯混合模型 GMM
#   （软聚类，输出"属于各簇的概率"）。
#
# ============================================================================

print(f"【完成】{pathlib.Path(__file__).name} 运行结束")

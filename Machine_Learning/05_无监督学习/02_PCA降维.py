r"""主成分分析 PCA 降维（无监督学习）——数学推导 + sklearn API + 实战 + 结果解读

对应课案章节
------------
《机器学习》课案 · 无监督学习 · 降维 · 主成分分析（PCA）
课案原文位置：`.course_extract/机器学习_课案.md` 第 1227~1271 行
（含"降维"定义、"PCA 是最常用的线性降维方法"与 iris 的 PCA 代码示例）

本节知识点
----------
1. 降维要解决什么问题：高维数据可视化难、距离度量失效（维度灾难）、
   计算慢、特征之间高度相关带来冗余。
2. PCA 的完整数学推导：数据中心化 → 协方差矩阵 C = (1/(n-1)) XᵀX →
   对 C 做特征值分解（或对 X 做 SVD）→ 取最大的 d 个特征值对应的特征向量
   作为主成分 → 投影 Z = X W_d。
3. 为什么"方差最大 = 信息保留最多"、主成分为什么彼此正交（互不相关）、
   解释方差比例 λ_i / Σλ 的含义、以及"降维一定是有损的"。
4. 为什么 PCA 之前必须标准化（量纲/尺度问题）。
5. sklearn 参数：n_components、whiten、svd_solver、random_state。
6. 实战：iris（4 维）降到 2 维并解释主成分载荷；
   digits（1797×64）画碎石图、用 n_components=0.95 自动确定"保留 95% 方差
   需要多少维"、再降到 2 维可视化。

运行方式
--------
在 PowerShell 中执行（路径含中文，必须加引号并用 & 调用）：

    $env:PYTHONUTF8='1'; & 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Machine_Learning\05_无监督学习\02_PCA降维.py'

产出图片（保存在 Machine_Learning/output/ 下）：
    05_PCA_iris降维散点.png     —— 原始 2 特征 vs PCA 前两主成分（按鸢尾花品种着色）
    05_PCA_碎石图.png           —— digits 各主成分解释方差比例 + 累计曲线（标 95% 所需维度）
    05_PCA_digits降维散点.png   —— digits 降到 2 维的散点（按真实数字标签着色 + colorbar）
"""

from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")            # 无界面后端，避免 plt.show() 阻塞
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import warnings

# --- 第三方噪声告警的精确屏蔽（与本节知识点无关，只为保证输出干净）-------------
# NumPy 2.5 起，对数组直接赋 shape（arr.shape = ...）会发 DeprecationWarning；
# 而 sklearn 1.9 的 load_digits() 内部仍写作 images.shape = (-1, 8, 8)，
# 这是第三方包内部实现、本脚本无法修改。该警告默认被 Python 过滤规则隐藏，
# 但在 `-W error` / pytest 环境下会变成异常，因此这里只按消息精确忽略这一条，
# 不使用全局 ignore，以免掩盖我们自己代码里的任何告警。
warnings.filterwarnings(
    "ignore",
    message=r"Setting the shape on a NumPy array has been deprecated",
    category=DeprecationWarning,
)

import numpy as np
from sklearn.datasets import load_digits, load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ① 原理与数学推导
# ============================================================================
#
# 【0. 降维在做什么】
# 降维 = 把 d 维特征空间里的样本，映射到一个更低维的 d' 维空间（d' < d），
# 同时"尽可能保留重要信息"。PCA 是最经典的【线性】降维方法：它假设低维结构
# 可以通过原来特征的线性组合得到，并且把"信息量"定义为"方差"。
#
#
# 【1. 第一步：数据中心化（centering）】
# 记原始数据矩阵 X ∈ R^{n×d}（n 个样本、d 个特征），每列的均值记为 x̄ ∈ R^d。
# PCA 的第一步是减去均值，得到中心化矩阵：
#
#       X_c = X - 1_n x̄ᵀ          （每个特征列均值变为 0）
#
# 为什么要中心化？因为 PCA 关心的是"数据在某个方向上散布得有多开"（方差），
# 而方差只与偏离均值有关。不中心化的话，均值大的特征会被误认为"方差大"，
# 主成分方向会被数据的整体偏移带偏。
#
#
# 【2. 第二步：协方差矩阵】
# 中心化之后，特征之间的"共同变化程度"由协方差矩阵刻画（无偏估计用 n-1）：
#
#       C = (1 / (n - 1)) · X_cᵀ X_c        ∈ R^{d×d}
#
# 其中 C 是对称半正定矩阵；对角线元素 C_jj 是第 j 个特征的方差，
# 非对角元素 C_jk 是第 j、k 个特征的协方差（>0 同向变化，<0 反向变化）。
# 关键恒等式：所有特征的方差总和 = trace(C)。
#
#
# 【3. 第三步：特征值分解（或对 X 做 SVD）】
# 协方差矩阵是对称矩阵，一定能正交对角化：
#
#       C = V Λ Vᵀ ,   Λ = diag(λ_1, λ_2, ..., λ_d),  λ_1 ≥ λ_2 ≥ ... ≥ λ_d ≥ 0
#
# V 的每一列 v_i 是一个单位特征向量，λ_i 是对应特征值。两条重要性质：
#   - 特征向量彼此正交：v_iᵀ v_j = 0 (i≠j)，且 ‖v_i‖ = 1（V 是正交矩阵）；
#   - 把数据投影到 v_i 方向后，投影结果的【方差恰好等于 λ_i】，且不同方向
#     上的投影【互不相关】（协方差为 0）。这就是"主成分之间正交/去相关"。
#
# 为什么"方差最大"的方向就是最大特征值的方向？形式化地，我们想求解
#
#       max_w  wᵀ C w     s.t.  ‖w‖₂ = 1
#
# 用拉格朗日乘子法：L(w, λ) = wᵀ C w - λ(wᵀ w - 1)，
# 令 ∂L/∂w = 2Cw - 2λw = 0，得 Cw = λw —— 也就是说 w 必须是 C 的特征向量，
# 此时目标值 wᵀ C w = λ。所以要最大，就取【最大特征值 λ_1】对应的特征向量 v_1；
# 第二主成分是"与 v_1 正交的条件下"方差最大的方向，即 v_2，依此类推。
#
# 工程实现上更常用【奇异值分解 SVD】代替显式的特征值分解：
#
#       X_c = U S Vᵀ  ⇒  C = (1/(n-1)) V S² Vᵀ  ⇒  λ_i = s_i² / (n - 1)
#
# 好处是：不用显式算 d×d 的协方差矩阵（d 很大时省内存），且数值更稳定
# （直接对 C 求特征值会放大条件数、损失精度）。sklearn 默认走 SVD 路线。
#
#
# 【4. 第四步：取前 d' 个主成分并投影】
# 把特征值从大到小排序后，只保留前 d' 个特征向量，拼成投影矩阵
#
#       W_d = [v_1, v_2, ..., v_d']  ∈ R^{d × d'}
#
# 降维后的新坐标（主成分得分）就是：
#
#       Z = X_c W_d   ∈ R^{n × d'}
#
# Z 的第 j 列就是"第 j 主成分得分"。这等价于把每个样本 x 变成一组新特征
# z = (v_1ᵀx, v_2ᵀx, ..., v_d'ᵀx)，新特征是原特征的线性组合、且两两不相关。
#
#
# 【5. 解释方差比例：怎么衡量"保留了多少信息"】
# 因为投影到 v_i 的方差是 λ_i，总方差是 Σλ_i，所以：
#
#       第 i 个主成分的解释方差比例 = explained_variance_ratio_i = λ_i / Σ_{j=1..d} λ_j
#       前 d' 个主成分的累计解释方差比例 = Σ_{i=1..d'} λ_i / Σ_{j=1..d} λ_j
#
# 这个比例的分子分母都是方差，与量纲无关，所以是 [0,1] 之间的"信息占比"。
# 累计比例 0.95 就意味着"用 d' 维近似原来的 d 维，保留了 95% 的总方差（信息）"。
#
#
# 【6. 降维是有损的】
# 丢掉后 d-d' 个主成分，等价于丢掉了它们承载的方差：
#
#       丢弃的信息量 = Σ_{i=d'+1..d} λ_i   （占总量 1 - 累计比例）
#
# 可以证明：用 Z 重建回原空间的最佳近似是 X̂_c = Z W_dᵀ，
# 其【平方重建误差恰好等于被丢掉的特征值之和】（这就是 Eckart–Young 定理：
# 在"秩不超过 d'"的所有矩阵里，截断 SVD 是最优的低秩近似）。
# 所以"保留 95% 方差"= 重建误差平方和约占总方差 5%。这也是去噪的原理：
# 末尾那些小特征值方向往往主要是噪声，丢掉它们反而让信号更干净。
#
#
# 【7. 为什么 PCA 之前一定要标准化】
# PCA 的目标函数是"方差最大"，而方差是有量纲的。若某特征用"米"、另一特征用
# "毫米"表示同一物理量，后者方差会大 10^6 倍，主成分就会被它垄断——但这纯粹是
# 单位造成的，不是真实的信息分布。标准化（StandardScaler：减均值、除标准差）后
# 每个特征方差都为 1，总方差 = d，PCA 找的是"特征之间的相关性结构"，
# 各特征站在同一起跑线上。经验规则：
#   - 特征量纲不同 或 数值范围相差大 → 必须标准化；
#   - 量纲相同且业务上"大方差本就更重要"（如全部是同一传感器的不同通道）→
#     可以只做中心化，不做缩放（sklearn 里就是直接 PCA，或先用 MinMaxScaler 之外的
#     手动减均值）。
#
#
# 【8. 优缺点速览】
# 优点：无监督、无参数可调（除了维度）、有严格的数学解释、计算高效（SVD 是成熟的
#       数值算法）、能去相关、能去噪、压缩后可显著加速下游模型、可做二维/三维可视化。
# 缺点：只捕捉【线性】结构（流形数据如 Swiss roll 要用 t-SNE/UMAP/KernelPCA）；
#       主成分是原特征的线性组合，可解释性不如原始特征（"PC1 = 0.52×花萼长 + …"）；
#       对量纲/异常值敏感；是无监督的，方差大 ≠ 对预测任务有用，可能丢掉
#       "方差小但判别力强"的方向（此时应改用 LDA 等有监督降维）。
#
# ============================================================================


# ============================================================================
# ② sklearn 的 PCA API 关键参数逐个解释
# ============================================================================
#
# PCA(n_components=None, whiten=False, svd_solver="auto", tol=0.0,
#     iterated_power="auto", n_oversamples=10, power_iteration_normalizer="auto",
#     random_state=None)
#
# (1) n_components : int / float ∈ (0,1) / "mle" / None，默认 None。 ——【最重要】
#     含义：保留多少个主成分（也就是降到几维）。
#     · int（如 2、30）：明确保留前 d' 个主成分，输出维度就是 d'。
#       调大：保留信息更多、压缩率低，下游模型慢一点但精度通常更高（到一定程度后
#             反而下降，因为噪声也进来了）。
#       调小：压缩更狠、可视化方便，但丢的信息更多（欠拟合风险）。
#       常用值：2 或 3（画图）、原维度的 10%~30%（做特征压缩）。
#     · float ∈ (0,1)（如 0.95）：表示"保留至少这么多比例的累计方差"，由算法自动
#       决定维度数，结果存在 pca.n_components_ 里。这是最省心的用法：
#       不用猜维度，直接说"我要留 95% 的信息"。调大（0.99）维度更多、更保真；
#       调小（0.90）压缩更狠。
#       注意：用小数或 "mle" 时，svd_solver 必须是 "full" / "covariance_eigh" / "auto"
#             （"arpack"、"randomized" 不支持）。
#     · "mle"：用 MLE（最大似然估计）自动猜维度（Minka 的方法），本质是
#       "把明显大于噪声水平的特征值都留下"。同样需要 full/covariance_eigh/auto。
#       MLE 有时会低估维度，它是启发式估计，不是业务目标，慎用于生产。
#     · None：全保留，用于画碎石图看清全部 64 个主成分（本脚本 digits 就是这么做的）。
#     常用值总结：可视化 = 2；自动压缩 = 0.95；要看全谱 = None。
#
# (2) whiten : bool，默认 False。
#     含义：白化（球化）。设 True 时，会把每个主成分得分再除以 sqrt(λ_i)，
#     使输出各列方差都为 1、且互不相关（协方差矩阵变成单位阵）。
#     为什么有用：很多下游模型（KNN、SVM-RBF、KMeans）默认各方向同等重要，
#     白化后特征尺度一致、条件数更好，常常收敛更快、精度更高。
#     代价：低方差主成分（往往以噪声为主）被放大到与其他方向同权，可能放大噪声；
#     同时输出不再是"原始方差的等比缩放"，解释方差比例的直观性减弱。
#     常用值：做下游模型前处理可以试 True；做可视化 / 解释时保持 False（默认）。
#
# (3) svd_solver : {"auto", "full", "covariance_eigh", "arpack", "randomized"}，
#     默认 "auto"。含义：用哪个数值求解器算特征分解。它【只影响速度与数值稳定性，
#     不影响数学结果】（精度在数值误差范围内一致）。
#     · "auto"：默认由数据形状自动挑（本脚本都用它，最省事）。
#       规则大致是：小数据 / 要全谱 → full；n_features 大且只要少数几维 → randomized。
#     · "full"：用 scipy 的 LAPACK 全量分解，最精确，适合 d 不大（几百以内）。
#     · "covariance_eigh"：先算 d×d 协方差再特征分解，n_samples 远大于 d 时更快、
#       内存更省（sklearn 1.5+ 提供）。
#     · "arpack"：迭代法只求前 k 个特征对，适合"只要少数几维"的稀疏/大矩阵；
#       要求 0 < n_components < min(n_samples, n_features)，不收敛会报错。
#     · "randomized"：随机化 SVD，快且可近似，适合超大矩阵 + 只要前 k 维；
#       精度由 n_oversamples / iterated_power 控制，结果依赖 random_state。
#     调参影响：换 solver 不会改变"应该保留几维"的结论，只改变耗时。
#
# (4) random_state : int / None，默认 None。
#     含义：仅在 svd_solver="randomized"（或 "auto" 解析到 randomized）时用于
#     随机初始化，保证可复现。用 full/arpack 时它是确定性的、此参数无影响。
#     课程统一写 42，保证任何机器上结果一致。
#
# (5) tol : float，默认 0.0。仅 arpack / randomized 使用，是奇异值的收敛容差；
#     0.0 表示用机器精度。一般不用动。
#
# (6) iterated_power / n_oversamples：只对 randomized 生效，控制"幂迭代次数"和
#     "过采样列数"。调大更准更慢；默认值通常够用。
#
# 其他常用属性 / 方法：
#     pca.explained_variance_          : 各主成分的方差（= λ_i，降序）；
#     pca.explained_variance_ratio_    : 各主成分解释方差比例（λ_i / Σλ）；
#     pca.singular_values_             : 奇异值 s_i（λ_i = s_i²/(n-1)）；
#     pca.components_                  : (d', d) 载荷矩阵，第 i 行是第 i 主成分
#                                        在各原始特征上的系数（= v_iᵀ），
#                                        用来解释"这个主成分由哪些特征组成"；
#     pca.n_components_                : 实际保留的维度（用 0.95 或 "mle" 时很有用）；
#     pca.mean_                        : 训练时算出的特征均值（预测新数据要用同一均值）；
#     pca.fit_transform(X) / transform(X_new)：降维；inverse_transform(Z)：重建。
#     重要实践：transform 只应使用训练集 fit 出来的 PCA（否则等于数据泄漏）。
#
# ============================================================================


# ============================================================================
# ③ 完整可运行代码
# ============================================================================


def describe_principal_component(loadings, feature_names, top_n=3):
    """把主成分载荷翻译成一句中文说明：主要有哪些特征、同向还是反向。

    loadings      : 长度 d 的向量，第 i 个元素是该主成分在第 i 个原始特征上的系数。
    feature_names : 原始特征的中文名列表。
    返回          : 形如 "花瓣长度(+0.58)、花瓣宽度(+0.56)、花萼长度(+0.52) 同向贡献，与 花萼宽度(-0.27) 反向"
    """
    idx_sorted = np.argsort(-np.abs(loadings))          # 按载荷绝对值从大到小排序
    top_idx = idx_sorted[:top_n]
    parts = [f"{feature_names[i]}({loadings[i]:+.2f})" for i in top_idx]
    same_sign = np.all(loadings[top_idx] > 0) or np.all(loadings[top_idx] < 0)
    direction = "同向贡献" if same_sign else "混合方向贡献"
    rest_idx = idx_sorted[top_n:]
    rest = "、".join(f"{feature_names[i]}({loadings[i]:+.2f})" for i in rest_idx)
    return f"{'、'.join(parts)} {direction}；其余特征：{rest}"


print("=" * 78)
print("PCA 主成分分析降维实战（无监督学习）")
print("=" * 78)

# ===========================================================================
# 3.1 实验 (a)：iris 数据集 4 维 → 2 维
# ===========================================================================
print("\n【实验 A】iris 鸢尾花数据集：4 维 → 2 维")

iris = load_iris()
X_iris_raw = iris.data            # (150, 4)
y_iris = iris.target              # (150,) 三个品种：0=setosa, 1=versicolor, 2=virginica
iris_feature_names_cn = ["花萼长度", "花萼宽度", "花瓣长度", "花瓣宽度"]

print(f"  数据集规模        : {X_iris_raw.shape[0]} 个样本 × {X_iris_raw.shape[1]} 个特征")
print(f"  原始特征名        : {'、'.join(iris_feature_names_cn)}（单位都是 cm）")

# ---- 标准化：让每个特征方差为 1，避免量纲/尺度差异主导方差 ----
scaler_iris = StandardScaler()
X_iris = scaler_iris.fit_transform(X_iris_raw)
print("  标准化后各特征均值（应≈0）:", np.round(X_iris.mean(axis=0), 6).tolist())
print("  标准化后各特征标准差（应=1）:", np.round(X_iris.std(axis=0, ddof=0), 4).tolist())

# ---- 在标准化数据上手动复现 PCA 的数学步骤，验证 sklearn 的结果 ----
X_centered = X_iris - X_iris.mean(axis=0)                      # ① 中心化
cov_matrix = (X_centered.T @ X_centered) / (X_iris.shape[0] - 1)  # ② 协方差 C = XᵀX/(n-1)
eig_values, eig_vectors = np.linalg.eigh(cov_matrix)           # ③ 对称矩阵特征值分解
order = np.argsort(eig_values)[::-1]                           # 特征值从大到小
eig_values = eig_values[order]
manual_ratio = eig_values / eig_values.sum()                   # ④ 解释方差比例 λ_i / Σλ

print("\n  —— 用 numpy 手动复现 PCA 的数学步骤（对照 sklearn）——")
print("  协方差矩阵 C = XᵀX/(n-1)：")
for row in cov_matrix:
    print("     [" + "  ".join(f"{v:>7.4f}" for v in row) + "]")
print(f"  特征值 λ（降序）      : {np.round(eig_values, 4).tolist()}")
print(f"  特征值之和 = trace(C) : {eig_values.sum():.4f}")
print(f"     说明：StandardScaler 用总体标准差(ddof=0)缩放，若改用无偏协方差(ddof=1)，")
print(f"          每个特征的方差是 n/(n-1) = {X_iris.shape[0] / (X_iris.shape[0] - 1):.4f}，")
print(f"          {X_iris.shape[1]} 个特征之和 = {X_iris.shape[1] * X_iris.shape[0] / (X_iris.shape[0] - 1):.4f}，与 trace(C) 完全吻合。")
print("          这验证了一个恒等式：所有特征的方差之和 = 协方差矩阵的迹 = 全部特征值之和，")
print("          所以『解释方差比例 = λ_i / Σλ』本质上就是『这个方向占总方差的百分比』。")
print(f"  解释方差比例 λ_i/Σλ   : {np.round(manual_ratio, 4).tolist()}")

# ---- sklearn 正式降维：4 → 2 ----
pca_iris = PCA(n_components=2, svd_solver="auto", random_state=42)
Z_iris = pca_iris.fit_transform(X_iris)

print("\n  —— sklearn PCA(n_components=2) 结果 ——")
print(f"  原始数据维度   : {X_iris_raw.shape}")
print(f"  降维后数据维度 : {Z_iris.shape}")
print(f"  实际保留维数 n_components_ : {pca_iris.n_components_}")

print("\n  降维后的前 5 条数据（每行是两个主成分得分 PC1, PC2）：")
for i in range(5):
    print(f"     样本 {i}: [{Z_iris[i, 0]:>8.4f}, {Z_iris[i, 1]:>8.4f}]   真实品种 = {iris.target_names[y_iris[i]]}")

print(f"\n  各主成分解释方差比例 explained_variance_ratio_ : {np.round(pca_iris.explained_variance_ratio_, 4).tolist()}")
print(f"  各主成分方差 explained_variance_               : {np.round(pca_iris.explained_variance_, 4).tolist()}")
print("  说明：sklearn 的 explained_variance_ 是“无偏”估计，等于 λ_i（第 ② 步算出的特征值），")
print("        与上面手动算的 λ 完全吻合，说明我们对 PCA 的理解和库的实现一致。")
print(f"  累计解释方差比例（前 2 维）: {pca_iris.explained_variance_ratio_.sum():.4f}")
print(f"  ⇒ 用 2 个新特征代替原来 4 个特征，保留了全部方差的 "
      f"{pca_iris.explained_variance_ratio_.sum() * 100:.2f}%，")
print(f"     损失的信息（被丢掉的两个方向）仅占 {100 - pca_iris.explained_variance_ratio_.sum() * 100:.2f}%。")

print("\n  components_ 载荷矩阵（每行是一个主成分，每列对应一个原始特征）：")
print(f"     {'':<10}" + "".join(f"{n:>12}" for n in iris_feature_names_cn))
for i, row in enumerate(pca_iris.components_):
    print(f"     PC{i + 1:<7}" + "".join(f"{v:>12.4f}" for v in row))
print("\n  怎么读这张表：数值是“该主成分 = 各原始特征 × 该系数”里的系数（载荷）。")
print("     系数绝对值越大，说明该原始特征对这个主成分的贡献越大（越重要）；")
print("     一个主成分里多个特征同号 → 它们同向变化；反号 → 它们此消彼长。")
for i, row in enumerate(pca_iris.components_):
    print(f"     PC{i + 1} 的解释 → {describe_principal_component(row, iris_feature_names_cn)}")
print("\n  第一主成分 PC1 的中文解读：")
print("     PC1 在“花瓣长度、花瓣宽度、花萼长度”上的载荷同号且都很大（约 0.5~0.6），")
print(f"     只有“花萼宽度”是负号（{pca_iris.components_[0][1]:+.2f}）且绝对值较小")
print("     → PC1 主要刻画“花朵整体尺寸”：花越大，PC1 越大；花萼越宽，PC1 略小。")
print("\n  第二主成分 PC2 的中文解读：")
print(f"     PC2 以“花萼宽度”为主（载荷 {pca_iris.components_[1][1]:+.2f}），"
      f"另带少量“花萼长度”（{pca_iris.components_[1][0]:+.2f}），")
print(f"     而“花瓣长度”（{pca_iris.components_[1][2]:+.2f}）与“花瓣宽度”"
      f"（{pca_iris.components_[1][3]:+.2f}）的载荷几乎为 0")
print("     → 在“整体尺寸”已被 PC1 拿走之后，PC2 刻画的是“花萼相对宽窄”这一剩余变化。")
print("     两个主成分合起来就是一组正交且互不相关的综合指标：(整体尺寸, 花萼宽窄)。")

# ---- 顺便演示 n_components="mle"：让算法自动估计维度 ----
pca_mle = PCA(n_components="mle", svd_solver="full", random_state=42)
pca_mle.fit(X_iris)
print(f"\n  旁注：n_components=\"mle\" 在 iris 上自动估计出 {pca_mle.n_components_} 个主成分")
print("        （它是基于特征值谱的启发式估计，不做业务保证；实际项目里更推荐写 0.95。）")

# ===========================================================================
# 3.2 实验 (b)-1：digits 手写数字 64 维 —— 全谱 + 碎石图 + 95% 方差维度
# ===========================================================================
print("\n" + "=" * 78)
print("【实验 B】digits 手写数字数据集：64 维 → 画碎石图，并实测“保留 95% 方差”需要几维")
print("=" * 78)

digits = load_digits()
X_digits_raw = digits.data        # (1797, 64)
y_digits = digits.target          # (1797,) 0~9
print(f"  数据集规模 : {X_digits_raw.shape[0]} 个样本 × {X_digits_raw.shape[1]} 个特征")
print("  特征含义   : 每张 8×8 的手写数字灰度图被拉平成 64 个像素值（0~16）")

scaler_digits = StandardScaler()
X_digits = scaler_digits.fit_transform(X_digits_raw)
print("  已标准化：64 个像素特征均值≈0、标准差=1（否则“总是亮的像素”会主导方差）")

# 全谱 PCA：n_components=None 表示不降维，只求特征值谱，用于画碎石图
pca_full = PCA(n_components=None, svd_solver="full", random_state=42)
pca_full.fit(X_digits)
ratio_full = pca_full.explained_variance_ratio_
cum_ratio = np.cumsum(ratio_full)
print(f"\n  全谱 PCA 共得到 {len(ratio_full)} 个主成分（= 原始维度 {X_digits_raw.shape[1]}）")
print(f"  第 1 主成分解释方差比例          : {ratio_full[0]:.4f}（{ratio_full[0] * 100:.2f}%）")
print(f"  前 2 个主成分累计解释方差比例    : {cum_ratio[1]:.4f}")
print(f"  前 10 个主成分累计解释方差比例   : {cum_ratio[9]:.4f}")
print(f"  前 20 个主成分累计解释方差比例   : {cum_ratio[19]:.4f}")
print("  解读：方差高度集中在最前面几个主成分上——这就是“碎石图”里陡降的那一段，")
print("        说明 64 个像素之间存在很强的相关性（笔画相邻像素总是一起变化），")
print("        原始 64 维里大量维度是冗余的，完全可以压缩。")

# 用 n_components=0.95 实测：保留 95% 方差需要多少维
pca_95 = PCA(n_components=0.95, svd_solver="full", random_state=42)
Z_digits_95 = pca_95.fit_transform(X_digits)
n_95 = pca_95.n_components_
ratio_95 = pca_95.explained_variance_ratio_.sum()
print(f"\n  PCA(n_components=0.95) 实测结果：")
print(f"    自动确定保留维数 n_components_ = {n_95}  （原始 {X_digits_raw.shape[1]} 维）")
print(f"    实际累计解释方差比例            = {ratio_95:.4f}")
print(f"    压缩率                          = {X_digits_raw.shape[1]} → {n_95} 维，"
      f"降到原来的 {n_95 / X_digits_raw.shape[1] * 100:.1f}%")
print(f"    数据量变化                      = {Z_digits_95.nbytes / 1024:.1f} KB（原 {X_digits_raw.nbytes / 1024:.1f} KB）")

# 量化"降维是有损的"：把 95% 版本重建回 64 维，看误差，并用数学恒等式校验
X_digits_recon = pca_95.inverse_transform(Z_digits_95)
residual = X_digits - X_digits_recon
mse_95 = float(np.mean(residual ** 2))                              # 均方重建误差
residual_fro2 = float(np.sum(residual ** 2))                        # 残差 Frobenius 范数平方
dropped_eig_sum = float(np.sum(pca_full.explained_variance_[n_95:]))  # 被丢弃特征值之和 Σλ_i
n_d, d_d = X_digits.shape
print(f"    重建误差（标准化空间的 MSE）    = {mse_95:.6f}")
print(f"    残差 Frobenius 范数平方 ‖X-X̂‖²  = {residual_fro2:.4f}")
print(f"    被丢弃特征值之和 Σλ_i (i>d')    = {dropped_eig_sum:.4f}")
print(f"    校验 (n-1)·Σλ_i = {(n_d - 1) * dropped_eig_sum:.4f}"
      f"  （应等于 ‖X-X̂‖²，相对偏差 {abs(residual_fro2 - (n_d - 1) * dropped_eig_sum) / residual_fro2:.2e}）")
print(f"    被丢弃的方差比例                = {1 - ratio_95:.4f}（约占 {100 - ratio_95 * 100:.2f}%）")
print("    这正是 Eckart–Young 定理的体现：在所有秩不超过 d' 的矩阵中，截断 SVD 是使")
print("    ‖X - X̂‖_F 最小的那个（最优低秩近似），且残差的 Frobenius 范数平方恰好等于")
print("    被丢弃的奇异值平方和 = (n-1)·Σ_{i>d'} λ_i——上面已用真实数字校验通过。")
print("    注意 MSE 还要再除以元素个数 n×d，所以它的绝对大小依赖数据尺度与样本量，")
print("    跨数据集比较『重建 MSE』是没有意义的，比较『被丢弃方差比例』才有意义。")
print("    结论：降维一定是有损的，只是当丢掉的是小方差（多为噪声）方向时，")
print("          损失对下游任务的影响很小。")

# ===========================================================================
# 3.3 实验 (b)-2：digits 降到 2 维做可视化
# ===========================================================================
pca_digits_2d = PCA(n_components=2, svd_solver="auto", random_state=42)
Z_digits_2d = pca_digits_2d.fit_transform(X_digits)
print(f"\n  digits 降到 2 维后维度 : {Z_digits_2d.shape}")
print(f"  前 2 维累计解释方差比例: {pca_digits_2d.explained_variance_ratio_.sum():.4f}"
      f"（仅保留 {pca_digits_2d.explained_variance_ratio_.sum() * 100:.2f}% 的信息）")
print("  注意：2 维只保留了很小一部分方差，所以下面的散点图只能粗略看出数字的聚集趋势，")
print("        不同数字之间会有明显重叠——这就是“降到 2 维方便看，但信息损失大”的代价。")

# ===========================================================================
# 3.4 画图 1：iris 原始 2 特征 vs PCA 前两主成分
# ===========================================================================
species_colors = ["#1f77b4", "#2ca02c", "#d62728"]

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8))

# 左：只用原始前两个特征看数据（信息不全，setosa 能分开，另两类重叠）
for cls in np.unique(y_iris):
    mask = y_iris == cls
    ax1.scatter(X_iris_raw[mask, 0], X_iris_raw[mask, 1], s=46,
                color=species_colors[cls], alpha=0.8, edgecolors="white", linewidths=0.6,
                label=iris.target_names[cls])
ax1.set_xlabel("花萼长度 sepal length (cm)", fontsize=11)
ax1.set_ylabel("花萼宽度 sepal width (cm)", fontsize=11)
ax1.set_title("原始特征（只取前 2 个特征）", fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# 右：PCA 前两主成分（融合了全部 4 个特征的信息）
for cls in np.unique(y_iris):
    mask = y_iris == cls
    ax2.scatter(Z_iris[mask, 0], Z_iris[mask, 1], s=46,
                color=species_colors[cls], alpha=0.8, edgecolors="white", linewidths=0.6,
                label=iris.target_names[cls])
ax2.set_xlabel(f"第一主成分 PC1（解释方差 {pca_iris.explained_variance_ratio_[0] * 100:.1f}%）", fontsize=11)
ax2.set_ylabel(f"第二主成分 PC2（解释方差 {pca_iris.explained_variance_ratio_[1] * 100:.1f}%）", fontsize=11)
ax2.set_title(f"PCA 降到 2 维（累计保留 {pca_iris.explained_variance_ratio_.sum() * 100:.1f}% 方差）", fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

fig1.suptitle("iris 数据集：原始 2 个特征 vs PCA 前两主成分（按真实品种着色）", fontsize=14)
fig1.tight_layout()
fig1.savefig(OUTPUT_DIR / "05_PCA_iris降维散点.png", dpi=150)
plt.close(fig1)
print(f"\n  已保存图片：{OUTPUT_DIR / '05_PCA_iris降维散点.png'}")

# ===========================================================================
# 3.5 画图 2：digits 碎石图（Scree Plot）+ 累计解释方差曲线
# ===========================================================================
fig2, ax_left = plt.subplots(figsize=(12.5, 6.2))
x_axis = np.arange(1, len(ratio_full) + 1)

# 柱状：各主成分单独的解释方差比例
ax_left.bar(x_axis, ratio_full, color="#4c72b0", alpha=0.75, width=0.85,
            label="各主成分解释方差比例（λ_i / Σλ）")
ax_left.set_xlabel("主成分序号（按方差从大到小排列）", fontsize=12)
ax_left.set_ylabel("单个主成分的解释方差比例", fontsize=12, color="#4c72b0")
ax_left.tick_params(axis="y", labelcolor="#4c72b0")
ax_left.set_xticks(np.arange(0, len(ratio_full) + 1, 5))
ax_left.grid(alpha=0.3, axis="y")

# 折线：累计解释方差比例（用右侧副坐标轴，避免两条曲线量级差距太大看不清）
ax_right = ax_left.twinx()
ax_right.plot(x_axis, cum_ratio, marker="o", markersize=4, color="#d62728",
              linewidth=2, label="累计解释方差比例")
ax_right.set_ylabel("累计解释方差比例", fontsize=12, color="#d62728")
ax_right.tick_params(axis="y", labelcolor="#d62728")
ax_right.set_ylim(0, 1.05)

# 标出"保留 95% 方差"所需的主成分个数
ax_right.axhline(0.95, color="#2ca02c", linestyle="--", linewidth=1.6)
ax_right.axvline(n_95, color="#2ca02c", linestyle="--", linewidth=1.6)
ax_right.scatter([n_95], [cum_ratio[n_95 - 1]], s=180, marker="*",
                 color="#2ca02c", edgecolors="black", linewidths=0.8, zorder=6)
ax_right.annotate(f"保留 95% 方差需要 {n_95} 个主成分\n"
                  f"（{X_digits_raw.shape[1]} 维 → {n_95} 维，压缩到 {n_95 / X_digits_raw.shape[1] * 100:.1f}%）",
                  xy=(n_95, cum_ratio[n_95 - 1]),
                  xytext=(n_95 + 3.5, 0.72),
                  fontsize=11, color="#2ca02c",
                  arrowprops=dict(arrowstyle="->", color="#2ca02c", linewidth=1.5))

# 两条图例合并处理
handles_left, labels_left = ax_left.get_legend_handles_labels()
handles_right, labels_right = ax_right.get_legend_handles_labels()
ax_left.legend(handles_left + handles_right, labels_left + labels_right,
               loc="center right", fontsize=10)
ax_left.set_title("digits 数据集 PCA 碎石图：64 个主成分的解释方差比例与累计曲线", fontsize=13)
fig2.tight_layout()
fig2.savefig(OUTPUT_DIR / "05_PCA_碎石图.png", dpi=150)
plt.close(fig2)
print(f"  已保存图片：{OUTPUT_DIR / '05_PCA_碎石图.png'}")
print("  碎石图解读：柱子高度就是每个主成分“值多少信息”。")
print("     前几根柱子很高（陡坡），说明少数几个方向就概括了大部分变化；")
print("     后面几十根柱子非常矮（碎石/平台），再增加维度几乎换不来新的方差——")
print("     这些方向主要是噪声和几乎不变的像素，是可以安全丢掉的。")
print(f"     绿星标注处就是实测结论：保留 95% 方差只需 {n_95} 维，其余 "
      f"{X_digits_raw.shape[1] - n_95} 维承载的方差不足 5%。")

# ===========================================================================
# 3.6 画图 3：digits 降到 2 维的散点图（按真实数字标签着色 + colorbar）
# ===========================================================================
fig3, ax3 = plt.subplots(figsize=(9.5, 7.6))
scatter = ax3.scatter(Z_digits_2d[:, 0], Z_digits_2d[:, 1], c=y_digits,
                      cmap="tab10", s=16, alpha=0.8, vmin=-0.5, vmax=9.5)
cbar = fig3.colorbar(scatter, ax=ax3, ticks=np.arange(10))
cbar.set_label("真实数字标签（0~9）", fontsize=12)
ax3.set_xlabel(f"第一主成分 PC1（解释方差 {pca_digits_2d.explained_variance_ratio_[0] * 100:.1f}%）", fontsize=12)
ax3.set_ylabel(f"第二主成分 PC2（解释方差 {pca_digits_2d.explained_variance_ratio_[1] * 100:.1f}%）", fontsize=12)
ax3.set_title("digits 手写数字降到 2 维的分布（按真实数字着色，仅保留 "
              f"{pca_digits_2d.explained_variance_ratio_.sum() * 100:.1f}% 方差）", fontsize=13)
ax3.grid(alpha=0.3)
fig3.tight_layout()
fig3.savefig(OUTPUT_DIR / "05_PCA_digits降维散点.png", dpi=150)
plt.close(fig3)
print(f"  已保存图片：{OUTPUT_DIR / '05_PCA_digits降维散点.png'}")
print("  散点图解读：数字 0/6、1/7、3/8 等“写法相近”的手写体在 2 维平面上彼此重叠，")
print("     因为只保留 2 维丢掉了大量细节；而像 0 与 1 这种笔画差异大的数字，")
print("     会落在相对分离的角落。这说明 2 维适合“看趋势”，不适合“做判别”。")

# ===========================================================================
# 3.7 结果解读汇总
# ===========================================================================
print("\n" + "=" * 78)
print("④ 结果解读")
print("=" * 78)
print(f"""
1) 解释方差比例（explained_variance_ratio_）到底是什么
   它就是 λ_i / Σλ：第 i 个主成分承载的方差占全部方差的百分比。
   因为方差越大 = 数据在该方向变化越剧烈 = 该方向携带的"信息"越多，
   所以这个百分比可以直接读成"这个主成分保留了多少信息"。
   · iris：PC1 = {pca_iris.explained_variance_ratio_[0] * 100:.2f}%，PC2 = {pca_iris.explained_variance_ratio_[1] * 100:.2f}%。
   · digits：PC1 = {ratio_full[0] * 100:.2f}%，前 10 维累计 = {cum_ratio[9] * 100:.2f}%，
     前 20 维累计 = {cum_ratio[19] * 100:.2f}%。

2) 累计解释方差比例 与"压缩到几维"
   累计比例回答的是"用 d' 维代替原来的 d 维，保住了多少信息"。
   · iris 降到 2 维：{X_iris_raw.shape[1]} → 2 维，累计保留 {pca_iris.explained_variance_ratio_.sum() * 100:.2f}%，
     也就是说 2 维就几乎完整概括了 4 个特征（因为 4 个特征高度相关），
     这正是 PCA 的价值：去冗余。
   · digits 保留 95% 方差：{X_digits_raw.shape[1]} → {n_95} 维（压缩到 {n_95 / X_digits_raw.shape[1] * 100:.1f}%），
     实测累计 {ratio_95 * 100:.2f}%。像素维度里绝大部分是冗余的。
   · digits 降到 2 维：只保留 {pca_digits_2d.explained_variance_ratio_.sum() * 100:.2f}% 方差，
     所以二维图上类别严重重叠——降到几维取决于目的：为了看图就 2 维，
     为了给下游模型降本增效就写到 0.95。

3) 降维带来的信息损失
   降维一定是有损的：丢掉后 d-d' 个方向，就丢掉它们承载的方差。
   · iris 丢掉了 {100 - pca_iris.explained_variance_ratio_.sum() * 100:.2f}%；
   · digits 95% 版本丢掉了 {100 - ratio_95 * 100:.2f}%，实测标准化空间重建 MSE = {mse_95:.6f}；
     根据 Eckart–Young 定理，平方重建误差恰好等于被丢弃的方差（前面已用
     ‖X-X̂‖² = (n-1)·Σλ_i 校验通过），也就是说"丢掉多少方差"就是"损失多少信息"。
   关键结论：如果丢掉的是"小方差方向"，损失通常只是噪声，下游任务几乎不降精度；
   但如果某个判别力很强的方向方差很小（例如两类只在一个很小的方向上有差别），
   PCA（无监督）会把它当噪声丢掉，此时应改用有监督降维（LDA）或直接不降维。

4) 主成分怎么读（载荷 components_）
   每个主成分都是原始特征的线性组合，系数就是 components_ 的行。
   iris 的 PC1 = {pca_iris.components_[0][0]:+.2f}×花萼长 {pca_iris.components_[0][1]:+.2f}×花萼宽
                  {pca_iris.components_[0][2]:+.2f}×花瓣长 {pca_iris.components_[0][3]:+.2f}×花瓣宽，
   三个"长度/宽度"同号、且与花萼宽反号 → 可以解释为"花朵整体尺寸"这一综合指标。
   注意符号可以整体取反（特征向量方向无所谓正负），所以别纠结 PC1 的正负号，
   要看的是"哪些特征同号、谁大谁小"。

5) PCA 的典型用途
   · 可视化：把高维数据压到 2~3 维画散点，观察是否天然分群（本项目 digits 就是这么用的）；
   · 去噪：丢掉末尾低方差方向，相当于低通滤波，图像/信号处理里很常用；
   · 加速下游模型：维度降低后 KNN/SVM 的距离计算和训练都更快，内存更省；
     但要注意 PCA 之后再用 KNN 时，通常建议 whiten=True 让各方向等权；
   · 去相关：主成分彼此正交，能消除多重共线性（对线性回归的系数稳定性有帮助）；
   · 压缩存储：只存 Z 和 W_d，需要时用 inverse_transform 近似还原原数据。

6) 重要使用规范（新手最常踩的坑）
   · 必须先标准化（除非量纲一致且业务上"大方差确实更重要"）；
   · PCA 只能在【训练集】上 fit，然后 transform 训练集和测试集，
     绝不能用全量数据 fit（那是数据泄漏，会让评估指标虚高）；
   · 标准化器和 PCA 都属于"模型的一部分"，上线时要一起保存（joblib）并在推理时复用；
   · 不要用 PCA 降维后再解释"某个原始特征的重要性"——降维后特征已混在一起。
""")

# ============================================================================
# 超参数怎么调
# ============================================================================
#
# 【n_components：PCA 唯一真正需要调的参数】
#   决策顺序建议：
#   ① 先画碎石图（PCA(n_components=None)）看谱形：如果累计曲线在很靠前的位置
#      就冲到 0.9 以上，说明数据冗余大、降维收益高；
#   ② 做可视化 / 画图 → 直接 n_components=2（或 3）；
#   ③ 给下游模型压缩特征 → 写 n_components=0.95（或 0.99）让它自动定维度，
#      这是最稳、最不需要猜的用法；也可以写成 0.90 换更强的压缩；
#   ④ 若下游精度下降，就把比例从 0.95 提到 0.99，或干脆不降维，用交叉验证
#      按"下游任务指标"而不是"解释方差比例"来选维度（这才是最终裁判）；
#   ⑤ "mle" 只用于快速探测，不建议作为生产配置（它不针对你的业务目标）。
#
# 【whiten】
#   - 下游是 KNN / SVM(RBF) / KMeans / 神经网络 → 可以试 whiten=True，
#     常能提升精度并加快收敛（各方向尺度一致，条件数更好）；
#   - 做可视化、做解释、给线性回归用 → 保持 False（默认），
#     否则低方差方向（噪声）被放大，图会变得很乱、解释也失真。
#
# 【svd_solver】
#   - 维度 d 不大（几百以内）、要全谱或高精度 → "full" 或 "covariance_eigh"；
#   - 样本数 n 远大于 d → "covariance_eigh" 更省内存（先算 d×d 协方差）；
#   - d 或 n 非常大、只要前几维 → "randomized"（可配合 n_oversamples 提高精度）；
#   - 只要少数几维且需要精确解 → "arpack"。
#   - 记住：solver 只影响速度/数值稳定性，不影响"该保留几维"的结论。
#   - 用 n_components=0.95 或 "mle" 时，不能选 "arpack" / "randomized"。
#
# 【其他实践要点】
#   - 标准化：StandardScaler（方差为 1）是最常用的；若特征稀疏或想做"保留零值"的
#     缩放，用 MaxAbsScaler；做图像像素这种同量纲数据可只减均值。
#   - 随机性：只有 randomized solver 会用到随机性，写 random_state=42 保证可复现。
#   - 数据量：n 或 d 极大时先考虑 IncrementalPCA（分批 fit）或
#     TruncatedSVD（稀疏矩阵专用，不做中心化），它们能在内存装不下时工作。
#   - 非线性结构：如果数据是流形（Swiss roll、make_moons 这类卷曲结构），
#     PCA 会失效，改用 KernelPCA / t-SNE / UMAP（t-SNE/UMAP 只适合可视化，
#     不适合作为下游模型输入，因为它们不提供稳定的 transform 语义）。
#
# ============================================================================

print(f"【完成】{pathlib.Path(__file__).name} 运行结束")

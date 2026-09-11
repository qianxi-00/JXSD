"""线性回归（Linear Regression）——最小二乘推导、正规方程、梯度下降与评估指标。

对应课案章节
    监督学习 → 回归 → 线性回归
    （课案原文：https:// 课程《机器学习》"监督学习 · 回归 · 线性回归"一节，
      包含 make_regression 生成数据、train_test_split 划分、LinearRegression 拟合、
      MSE / R² 打印、散点 + 拟合线绘图的示例代码）

本节知识点
    1. 线性回归的模型形式：单样本 y = kx + b、多样本、矩阵形式 y = Xθ
    2. 最小二乘损失函数 L(θ) = ‖Xθ - y‖² 的来历
    3. 对损失求导并令导数为 0，解得正规方程（闭式解）θ* = (XᵀX)⁻¹Xᵀy
    4. sklearn 的 LinearRegression 走的是解析解（底层用最小二乘/SVD），无需手动迭代
    5. 用「手写批量梯度下降」和「SGDRegressor 随机梯度下降」两种迭代法逼近同一解，并与解析解对比
    6. 回归评估指标 MSE / RMSE / MAE / R² 的公式与含义
    7. 多特征回归（load_diabetes）中 coef_ 与 intercept_ 的解释
    8. 超参数怎么调

运行方式（PowerShell，路径含中文必须加引号并用 & 调用）
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Machine_Learning\\02_监督学习_回归\\01_线性回归.py'

    控制台中文若显示乱码，可先执行：   $env:PYTHONUTF8='1'
    图片保存到 Machine_Learning/output/ 目录，如需交互式查看，把 plt.savefig(...) 换成
    plt.show()（并去掉 matplotlib.use("Agg")）即可在窗口中旋转/缩放查看。
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
from sklearn.datasets import load_diabetes, make_regression
from sklearn.linear_model import LinearRegression, Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 0. 公共配置：输出目录用 pathlib 相对本文件定位，不依赖"当前工作目录"
# ---------------------------------------------------------------------------
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42          # 所有随机过程固定种子，保证结果可复现


def print_title(text: str) -> None:
    """打印带分隔线的中文小标题，让控制台输出层次清晰。"""
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


# ===========================================================================
# ① 原理与数学推导：模型形式 → 损失函数 → 优化方法 → 优缺点
# ===========================================================================
#
# 【1】模型形式（模型假设）
#   单个样本：
#       ŷ = k·x + b
#   其中 k 是权重（斜率，表示 x 每增加 1 个单位，y 平均变化多少），
#        b 是截距（偏置，x=0 时模型的输出），ŷ 是预测值。
#
#   多个样本（n 个样本，1 个特征）：
#       ŷ_i = k·x_i + b ,  i = 1, 2, ..., n
#
#   矩阵形式（把截距吸收进参数向量，是写推导最方便的形式）：
#       令参数向量      θ = [k, b]ᵀ
#       令设计矩阵      X = [[x_1, 1],
#                           [x_2, 1],
#                           ...,
#                           [x_n, 1]]      ← 最后一列全是 1，专门对应截距 b
#       则模型统一写成 ŷ = X·θ
#
#   推广到 p 个特征时 X 的每一行是 [x_i1, x_i2, ..., x_ip, 1]，θ 是 (p+1) 维向量。
#
# 【2】损失函数：最小二乘（Least Squares）
#   我们想让"预测值"和"真实值"尽量接近，于是用残差 e = Xθ - y 的平方和衡量误差：
#
#       L(θ) = Σ_i (ŷ_i - y_i)²  =  (Xθ - y)ᵀ(Xθ - y)  =  ‖Xθ - y‖²
#
#   为什么用平方而不是绝对值？
#     · 平方处处可导，求导方便，能直接得到闭式解；绝对值在 0 点不可导。
#     · 平方对"大误差"惩罚更重，模型会优先消除离谱的错误。
#     · 在"误差服从高斯分布"的假设下，最小二乘等价于极大似然估计（MLE）。
#   缺点：对异常值非常敏感（一个离群点可以把整条直线拽偏）。
#
# 【3】优化方法 A：求导令零 → 正规方程（闭式解）
#   对 θ 求梯度：
#       ∂L/∂θ = ∂/∂θ (θᵀXᵀXθ - 2yᵀXθ + yᵀy) = 2XᵀXθ - 2Xᵀy
#   令梯度为 0：
#       XᵀXθ = Xᵀy                       ← 这就是"正规方程"
#   若 XᵀX 可逆，两边左乘 (XᵀX)⁻¹：
#       θ* = (XᵀX)⁻¹Xᵀy                  ← 最小二乘的闭式解
#   直觉理解：L(θ) 是关于 θ 的"开口向上的二次碗"，碗底就是梯度为 0 的那一点，
#             所以这个解是全局最优解，不存在局部最优问题。
#   注意：当特征之间有强共线性（XᵀX 近似奇异）时直接求逆数值不稳定，
#         sklearn 底层用 SVD/最小二乘求解器（lstsq），比手写求逆稳健得多。
#
# 【4】优化方法 B：梯度下降（迭代解）
#   当样本量或特征数极大时，求逆的代价 O(p³) 太高，改用迭代：
#       θ ← θ - η · ∂L/∂θ = θ - η · 2Xᵀ(Xθ - y) / n
#   其中 η 是学习率。每轮用**全部样本**算梯度叫批量梯度下降（BGD），
#   每次只用 1 个样本叫随机梯度下降（SGD），用一小批叫 Mini-batch GD。
#
# 【5】优缺点
#   优点：简单、可解释（系数就是"特征每变化 1 单位对 y 的平均影响"）、训练快、是很多
#         复杂模型（逻辑回归、神经网络全连接层）的基本单元。
#   缺点：只能表达线性关系（需人工构造多项式/交互特征）；对异常值敏感；不处理共线性
#         （要靠 Ridge/Lasso 等正则化变体）；不做特征缩放时梯度下降会收敛很慢。
#
# ===========================================================================

print_title("机器学习 · 监督学习 · 回归 · 线性回归（最小二乘 / 正规方程 / 梯度下降）")

# ===========================================================================
# ② sklearn API 关键参数逐个解释
# ===========================================================================
#
# 【make_regression】生成模拟回归数据
#   n_samples      : 样本数。越大越稳定，本例 100。
#   n_features     : 特征数。本例 1（方便画二维散点 + 拟合直线）。
#   noise          : 加到目标上的高斯噪声标准差。0 = 完全线性可分（R²=1）；
#                    噪声越大 R² 越低，本例 20，故意留出"拟合不完美"的空间。
#   random_state   : 随机种子，固定 42 保证每次运行数据完全一样。
#
# 【train_test_split】划分训练集 / 测试集
#   test_size      : 测试集比例，本例 0.3（30% 测试）。
#   random_state   : 固定 42，保证"同一次划分"，才能公平比较不同模型。
#   shuffle        : 默认 True，划分前先打乱；回归任务一般不需要 stratify。
#
# 【LinearRegression】普通最小二乘（OLS）
#   fit_intercept  : 默认 True，即自动学习截距 b（设计矩阵自动补一列 1）。
#                    若数据已中心化、想强制过原点则设 False。
#   copy_X         : 默认 True，不修改传入的 X（安全，几乎不用改）。
#   n_jobs         : 仅多目标回归（y 是多列）时并行，单目标无效。
#   positive       : 默认 False；设 True 会强制所有系数 ≥ 0（当业务上"特征只能正向影响"时用）。
#   注意：LinearRegression **没有** alpha / learning_rate 这类超参数——它直接解正规方程，
#         没有需要调的"训练超参数"（这正是解析解的优点）。
#
# 【SGDRegressor】随机梯度下降版线性回归（迭代解）
#   loss            : 损失函数，"squared_error"=最小二乘（默认）。
#   penalty         : 正则化类型，"l2"（默认）/ "l1" / "elasticnet" / None。
#   alpha           : 正则化强度，越大惩罚越强、系数越接近 0；默认 0.0001。
#   learning_rate   : 学习率调度策略，"constant" / "optimal"（默认）/ "invscaling" / "adaptive"。
#   eta0            : 初始学习率（learning_rate="constant" 时就是固定学习率），默认 0.01。
#   max_iter        : 最多迭代多少轮（epoch），默认 1000。
#   tol             : 提前停止阈值：连续若干轮损失下降小于 tol 就停，默认 1e-3。
#   early_stopping  : True 时自动留出验证集做早停（防过拟合）。
#   random_state    : 固定 42，保证 SGD 的随机顺序可复现。
#   ⚠ 关键实践：SGD 对特征尺度极敏感，**必须先 StandardScaler 标准化**，否则收敛极慢甚至发散。
#
# 【StandardScaler】标准化 z = (x - μ) / σ
#   with_mean / with_std : 默认都 True（做中心化 + 缩放）。稀疏矩阵要设 with_mean=False。
#
# 【Ridge】带 L2 正则的最小二乘（拓展演示）
#   alpha : L2 惩罚强度，越大系数被压得越接近 0（但不为 0），用于缓解共线性/过拟合。
#
# ===========================================================================

# ---------------------------------------------------------------------------
# ③ 完整可运行代码
# ---------------------------------------------------------------------------
print_title("第 1 步：生成数据并划分训练集 / 测试集（对应的就是课案里的第 1、2 步）")

X, y = make_regression(
    n_samples=100,
    n_features=1,
    noise=20,
    random_state=RANDOM_STATE,
)
print(f"生成的数据：X.shape = {X.shape}，y.shape = {y.shape}")
print(f"X 的取值范围：[{X.min():.3f}, {X.max():.3f}]")
print(f"y 的取值范围：[{y.min():.3f}, {y.max():.3f}]（y 的量级决定了后面 RMSE 算好还是算差）")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=RANDOM_STATE,
)
print(f"划分完成：训练集 {X_train.shape[0]} 条，测试集 {X_test.shape[0]} 条")

# ---------------------------------------------------------------------------
# 第 2 步：方法一 —— LinearRegression（解析解 / 正规方程）
# ---------------------------------------------------------------------------
print_title("第 2 步：方法一 LinearRegression（解析解，直接解正规方程 θ* = (XᵀX)⁻¹Xᵀy）")

t0 = time.perf_counter()
ols_model = LinearRegression()
ols_model.fit(X_train, y_train)
ols_fit_time = time.perf_counter() - t0

y_pred_ols = ols_model.predict(X_test)

print("sklearn 解析解得到的参数：")
print(f"  斜率 k（coef_）      = {ols_model.coef_[0]:.6f}")
print(f"  截距 b（intercept_） = {ols_model.intercept_:.6f}")
print(f"  拟合耗时             = {ols_fit_time * 1000:.3f} 毫秒（解析解几乎瞬间完成）")
print("解读：斜率 k 表示『特征 x 每增加 1 个单位，预测的 y 平均增加 k 个单位』；")
print("      截距 b 表示『当 x = 0 时模型的预测值』，它只是直线与 y 轴的交点，不一定有业务含义。")

# --- 手工用 numpy 复现正规方程，验证"解析解"到底在算什么 -------------------
X_design = np.hstack([X_train, np.ones((X_train.shape[0], 1))])   # 补一列 1 对应截距
theta_manual = np.linalg.solve(X_design.T @ X_design, X_design.T @ y_train)  # 解正规方程
print("\n手工用 numpy 解正规方程 XᵀXθ = Xᵀy 得到：")
print(f"  k = {theta_manual[0]:.6f}，b = {theta_manual[1]:.6f}")
print("  → 与 sklearn 的结果一致，说明 LinearRegression 做的就是最小二乘的闭式解。")
print("  （这里用 np.linalg.solve 而不是显式求逆 pinv/(XᵀX)⁻¹，数值上更稳定）")

# ---------------------------------------------------------------------------
# 第 3 步：方法二 —— 手写批量梯度下降（BGD）逼近同一个解
# ---------------------------------------------------------------------------
print_title("第 3 步：方法二 手写批量梯度下降（把解析解的公式换成一轮一轮地迭代）")

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)     # 只在训练集上 fit，避免数据泄漏
X_test_std = scaler.transform(X_test)           # 测试集只用训练集的 μ、σ 做 transform


def batch_gradient_descent(
    X_std: np.ndarray,
    y_raw: np.ndarray,
    learning_rate: float = 0.02,
    n_iters: int = 200,
) -> tuple[float, float, list[float]]:
    """手写批量梯度下降，返回 (标准化空间下的斜率, 标准化空间下的截距, 每轮损失)。

    梯度推导（对 MSE 损失 L = (1/n)Σ(ŷ_i - y_i)²）：
        ∂L/∂w = (2/n) Σ (ŷ_i - y_i) · x_i
        ∂L/∂b = (2/n) Σ (ŷ_i - y_i)
    更新： w ← w - η·∂L/∂w ； b ← b - η·∂L/∂b
    """
    n = X_std.shape[0]
    w, b = 0.0, 0.0          # 从 θ = 0 出发
    loss_history: list[float] = []

    for _ in range(n_iters):
        y_hat = w * X_std.ravel() + b          # 前向计算
        residual = y_hat - y_raw               # 残差 e = ŷ - y
        loss_history.append(float(np.mean(residual ** 2)))   # 记录当前 MSE，用于画收敛曲线

        grad_w = 2.0 * np.mean(residual * X_std.ravel())      # 对 w 的梯度
        grad_b = 2.0 * np.mean(residual)                      # 对 b 的梯度

        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

    return w, b, loss_history


t0 = time.perf_counter()
w_std, b_std, loss_history = batch_gradient_descent(
    X_train_std, y_train, learning_rate=0.02, n_iters=200
)
gd_fit_time = time.perf_counter() - t0

# 把"标准化空间"的参数换算回"原始尺度"：
#   z = (x - μ)/σ ，ŷ = w·z + b = (w/σ)·x + (b - w·μ/σ)
#   ⟹  k = w/σ ， b_原始 = b - w·μ/σ
mu, sigma = scaler.mean_[0], scaler.scale_[0]
k_gd = w_std / sigma
b_gd = b_std - w_std * mu / sigma

print(f"手写 BGD（200 轮，学习率 0.02，在标准化后的特征上迭代）得到：")
print(f"  标准化空间：w = {w_std:.6f}，b = {b_std:.6f}")
print(f"  换算回原始尺度：k = {k_gd:.6f}，b = {b_gd:.6f}")
print(f"  迭代耗时 = {gd_fit_time * 1000:.2f} 毫秒")
print(f"  与解析解对比：Δk = {abs(k_gd - ols_model.coef_[0]):.6f}，"
      f"Δb = {abs(b_gd - ols_model.intercept_):.6f}")
print("解读：迭代法在有限轮数后只能『逼近』解析解，差距就是还没有完全收敛的误差；")
print("      轮数越多 / 学习率越合适，这个差距越小。")

# ---------------------------------------------------------------------------
# 第 4 步：方法三 —— SGDRegressor（sklearn 自带的随机梯度下降）
# ---------------------------------------------------------------------------
print_title("第 4 步：方法三 SGDRegressor（随机梯度下降，工业界处理大数据时更常用）")

t0 = time.perf_counter()
sgd_model = SGDRegressor(
    loss="squared_error",
    penalty=None,             # 先不加重正则，便于和解析解直接对比
    learning_rate="constant",
    eta0=0.05,
    max_iter=500,
    tol=1e-6,
    random_state=RANDOM_STATE,
)
sgd_model.fit(X_train_std, y_train.ravel())
sgd_fit_time = time.perf_counter() - t0

k_sgd = sgd_model.coef_[0] / sigma
b_sgd = sgd_model.intercept_[0] - sgd_model.coef_[0] * mu / sigma

print(f"SGDRegressor 参数：k = {k_sgd:.6f}，b = {b_sgd:.6f}，耗时 = {sgd_fit_time * 1000:.2f} 毫秒")
print(f"与解析解的差距：Δk = {abs(k_sgd - ols_model.coef_[0]):.6f}，"
      f"Δb = {abs(b_sgd - ols_model.intercept_):.6f}")
print("解读：SGD 每轮只看一个样本，梯度是『带噪声』的，所以参数会在最优点附近抖动；")
print("      它的优势在大数据：单轮成本与样本量成正比而不是与样本量的平方相关。")

y_pred_sgd = sgd_model.predict(X_test_std)

# ---------------------------------------------------------------------------
# 第 5 步：评估（MSE / RMSE / MAE / R²）
# ---------------------------------------------------------------------------
print_title("第 5 步：用四个回归指标评估三个解，并解读每个数字的含义")


def regression_report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """计算并打印 MSE / RMSE / MAE / R²，返回字典方便后续比较。"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))                       # RMSE 就是 MSE 开方
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"  [{name}] MSE = {mse:10.3f} | RMSE = {rmse:9.3f} | "
          f"MAE = {mae:9.3f} | R² = {r2:.4f}")
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}


# 手写 BGD 在测试集上的预测值（用原始尺度参数直接算，避免再涉及标准化）
y_pred_gd = k_gd * X_test.ravel() + b_gd

print("三个方法在测试集（30 条）上的表现：")
metrics_ols = regression_report("LinearRegression 解析解", y_test, y_pred_ols)
metrics_gd = regression_report("手写批量梯度下降    ", y_test, y_pred_gd)
metrics_sgd = regression_report("SGDRegressor 随机GD ", y_test, y_pred_sgd)

print("""
【指标含义与结果解读】
1) MSE（均方误差）= 平均(预测 - 真实)²，单位是 y 的平方，数值本身不直观，但对大误差惩罚最重。
2) RMSE（均方根误差）= √MSE，单位与 y 一致，最容易解释：
   本实验 y 的取值大致在几百的量级，RMSE ≈ 20 左右，
   意思是"模型预测平均会偏离真实值大约 20 个单位"，和生成数据时加的 noise=20 基本吻合，
   说明模型已经把线性部分学到位了，剩下的误差来自数据本身的噪声（理论上无法消除）。
3) MAE（平均绝对误差）：对每个样本的误差一视同仁地取绝对值再平均。
   本实验中 MAE 略小于 RMSE，这是正常现象——RMSE 被少数较大的误差"放大"了。
   如果 MAE 远小于 RMSE，说明存在少数离谱的离群点，此时用 MAE 更能反映"典型误差"。
4) R²（决定系数）= 1 - SS_res/SS_tot = 1 - Σ(y-ŷ)² / Σ(y-ȳ)²：
   · R² = 1  → 完美预测；
   · R² = 0  → 和"永远预测训练集均值"一样差；
   · R² < 0  → 比直接猜均值还差（模型完全没学到东西，或用了错误的特征）；
   · 本实验 R² 在 0.9 量级，说明模型解释了约 90% 的 y 的波动，拟合质量很好。
5) 三个方法的指标几乎相同 → 佐证它们求的是**同一个最优化问题**，只是求解路径不同：
   解析解一步到位，梯度下降一点点逼近。""")
print(f"  三者 R² 对比：解析解 {metrics_ols['R2']:.4f} / 手写BGD {metrics_gd['R2']:.4f} / "
      f"SGDRegressor {metrics_sgd['R2']:.4f}")

# ---------------------------------------------------------------------------
# 第 6 步：画图（散点 + 三条拟合线；梯度下降收敛曲线）
# ---------------------------------------------------------------------------
print_title("第 6 步：绘图（图 1 拟合对比；图 2 梯度下降的损失收敛曲线）")

fig1, ax1 = plt.subplots(figsize=(9, 6), constrained_layout=True)
ax1.scatter(X_train, y_train, s=28, c="#4C72B0", alpha=0.75, label="训练集数据点")
ax1.scatter(X_test, y_test, s=40, c="#DD8452", alpha=0.85, marker="^", label="测试集数据点")

x_line = np.linspace(X.min() - 0.3, X.max() + 0.3, 200).reshape(-1, 1)
ax1.plot(x_line, ols_model.predict(x_line), color="#C44E52", linewidth=2.4,
         label=f"LinearRegression 解析解  k={ols_model.coef_[0]:.2f}")
ax1.plot(x_line, k_gd * x_line.ravel() + b_gd, color="#55A868", linewidth=2.0,
         linestyle="--", label=f"手写批量梯度下降  k={k_gd:.2f}")
ax1.plot(x_line, k_sgd * x_line.ravel() + b_sgd, color="#8172B2", linewidth=1.8,
         linestyle=":", label=f"SGDRegressor  k={k_sgd:.2f}")

ax1.set_title("线性回归：解析解 / 梯度下降 / 随机梯度下降的拟合直线对比", fontsize=13)
ax1.set_xlabel("特征 X")
ax1.set_ylabel("目标 y")
ax1.legend(loc="best", fontsize=10)
ax1.grid(alpha=0.3)
fig1.savefig(OUTPUT_DIR / "02_线性回归_三种解拟合对比.png", dpi=120)
plt.close(fig1)
print(f"已保存图片：{OUTPUT_DIR / '02_线性回归_三种解拟合对比.png'}")
print("看图要点：三条线几乎重合 —— 说明迭代法与解析解收敛到了同一个最优解；")
print("          若把它们分开画，肉眼看不出差别，但参数的小数点后几位仍有差异（未完全收敛）。")

fig2, ax2 = plt.subplots(figsize=(9, 5), constrained_layout=True)
ax2.plot(range(1, len(loss_history) + 1), loss_history, color="#C44E52", linewidth=2)
ax2.axhline(loss_history[-1], color="#4C72B0", linestyle="--",
            label=f"第 {len(loss_history)} 轮损失 ≈ {loss_history[-1]:.1f}")
ax2.set_title("批量梯度下降的收敛过程（损失函数随迭代轮数下降）", fontsize=13)
ax2.set_xlabel("迭代轮数（epoch）")
ax2.set_ylabel("训练集 MSE 损失")
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)
fig2.savefig(OUTPUT_DIR / "02_线性回归_梯度下降收敛曲线.png", dpi=120)
plt.close(fig2)
print(f"已保存图片：{OUTPUT_DIR / '02_线性回归_梯度下降收敛曲线.png'}")
print(f"首轮损失 {loss_history[0]:.1f} → 末轮损失 {loss_history[-1]:.1f}，")
print("曲线先急剧下降后逐渐变平，说明参数已经接近最优点；若曲线还在明显下降，就说明轮数不够。")

# ---------------------------------------------------------------------------
# 第 7 步：多特征回归（load_diabetes），解释 coef_ 数组与 R²
# ---------------------------------------------------------------------------
print_title("第 7 步：多特征线性回归（load_diabetes，10 个特征）—— 读懂 coef_ 数组")

diabetes = load_diabetes()
X_d, y_d = diabetes.data, diabetes.target
print(f"糖尿病数据集：X.shape = {X_d.shape}，特征名 = {list(diabetes.feature_names)}")
print("（该数据集的 10 个特征已被作者标准化过：每一列均值为 0、平方和为 1）")

X_d_train, X_d_test, y_d_train, y_d_test = train_test_split(
    X_d, y_d, test_size=0.3, random_state=RANDOM_STATE
)
multi_model = LinearRegression().fit(X_d_train, y_d_train)
y_d_pred = multi_model.predict(X_d_test)

print(f"\n截距 intercept_ = {multi_model.intercept_:.3f}")
print("各特征的系数 coef_（按 |系数| 从大到小排序，便于看出哪个特征影响最大）：")
order = np.argsort(-np.abs(multi_model.coef_))
for idx in order:
    print(f"  {diabetes.feature_names[idx]:>4s} : {multi_model.coef_[idx]:9.3f}")
print("解读：系数为正表示该特征越大、预测的疾病进展指标越高（正相关）；")
print("      系数为负表示负相关。因为特征已被标准化到同一量纲，系数绝对值之间可以直接比较大小，")
print("      绝对值越大说明该特征对预测的影响越强。")
print("      注意：线性回归对特征间的共线性很敏感，若两个特征高度相关，系数会变得不稳定、")
print("      甚至出现与常识相反的符号，这时应改用 Ridge / Lasso。")

mse_d = mean_squared_error(y_d_test, y_d_pred)
print(f"\n测试集评估：MSE = {mse_d:.2f}，RMSE = {np.sqrt(mse_d):.2f}，"
      f"MAE = {mean_absolute_error(y_d_test, y_d_pred):.2f}，"
      f"R² = {r2_score(y_d_test, y_d_pred):.4f}")
print("解读：糖尿病数据的 R² 只有 0.4~0.5 左右，远低于上一个模拟数据集的 0.9 量级。")
print("      这很正常——真实医学数据的规律本来就不是严格线性的，线性模型只能解释一半左右的波动；")
print("      换成树模型 / 神经网络通常会更高，但也要警惕过拟合。")

fig3, ax3 = plt.subplots(figsize=(7, 7), constrained_layout=True)
ax3.scatter(y_d_test, y_d_pred, s=35, alpha=0.7, c="#4C72B0", label="测试集样本")
lim = [min(y_d_test.min(), y_d_pred.min()) - 20, max(y_d_test.max(), y_d_pred.max()) + 20]
ax3.plot(lim, lim, color="#C44E52", linestyle="--", linewidth=2, label="理想情况 y = ŷ")
ax3.set_title(f"糖尿病数据：真实值 vs 预测值（R² = {r2_score(y_d_test, y_d_pred):.3f}）", fontsize=12)
ax3.set_xlabel("真实值 y")
ax3.set_ylabel("预测值 ŷ")
ax3.legend(fontsize=10)
ax3.grid(alpha=0.3)
fig3.savefig(OUTPUT_DIR / "02_线性回归_多特征预测对比.png", dpi=120)
plt.close(fig3)
print(f"已保存图片：{OUTPUT_DIR / '02_线性回归_多特征预测对比.png'}")
print("看图要点：点越贴近红色虚线，预测越准；点分散成一条『云带』说明还有大量未被解释的波动。")

# ---------------------------------------------------------------------------
# 第 8 步（拓展）：正则化回归 Ridge —— 缓解共线性与过拟合
# ---------------------------------------------------------------------------
print_title("第 8 步（拓展）：Ridge 回归 —— 在损失里加入 L2 惩罚项 α‖θ‖²")

ridge = Ridge(alpha=1.0, random_state=RANDOM_STATE).fit(X_d_train, y_d_train)
print(f"Ridge(alpha=1.0) 测试集 R² = {ridge.score(X_d_test, y_d_test):.4f}，"
      f"OLS 测试集 R² = {multi_model.score(X_d_test, y_d_test):.4f}")
print(f"OLS 系数绝对值之和 = {np.abs(multi_model.coef_).sum():.3f}，"
      f"Ridge 系数绝对值之和 = {np.abs(ridge.coef_).sum():.3f}")
print("解读：Ridge 的系数整体更小（被『压扁』了），这是 L2 惩罚在起作用——")
print("      它用一点点偏差换取方差的大幅下降，在特征共线或样本少时通常泛化更好。")
print("      代价函数变成 L(θ) = ‖Xθ - y‖² + α‖θ‖²，闭式解为 θ* = (XᵀX + αI)⁻¹Xᵀy，")
print("      αI 让矩阵一定可逆，顺便解决了共线性导致的数值不稳定。")

# ---------------------------------------------------------------------------
# ④ 结果解读（汇总）
# ---------------------------------------------------------------------------
print_title("④ 结果解读汇总")

print(f"""
【本次实验的核心结论】
1. 数据：make_regression(n_samples=100, n_features=1, noise=20, random_state=42)，
   训练 70 条 / 测试 30 条。
2. 三种求解方式得到几乎相同的直线：
   · LinearRegression（解析解）  k = {ols_model.coef_[0]:.4f}, b = {ols_model.intercept_:.4f}
   · 手写批量梯度下降（200 轮）  k = {k_gd:.4f}, b = {b_gd:.4f}
   · SGDRegressor（500 轮）       k = {k_sgd:.4f}, b = {b_sgd:.4f}
   → 因为三者优化的是同一个目标函数（最小二乘），只是求解路径不同。
3. 测试集指标：解析解 R² = {metrics_ols['R2']:.4f}，RMSE = {metrics_ols['RMSE']:.3f}；
   与生成数据时的噪声水平 noise=20 相当 → 说明线性部分已被充分学到，
   剩余误差属于"数据本身不可消除的噪声"。
4. 用哪个？
   · 特征少、样本小（几万条以内）→ 直接用 LinearRegression，解析解又快又准，不用调参；
   · 样本量百万级、特征维度很高 → 用 SGDRegressor（可配合 partial_fit 流式训练）；
   · 特征共线 / 明显过拟合 → 用 Ridge（L2）或 Lasso（L1，还能做特征选择）。
5. 一句话总结：**线性回归 = 找到一条使"误差平方和"最小的直线**，
   解析解是"一步到位"，梯度下降是"步步逼近"。""")

# ---------------------------------------------------------------------------
# 超参数怎么调（中文小结）
# ---------------------------------------------------------------------------
print_title("超参数怎么调（线性回归实践小结）")
print("""
1. LinearRegression：没有可调的训练超参数（解析解）。
   唯一要想的是 fit_intercept（数据是否已中心化）和 positive（是否要求系数非负）。
2. SGDRegressor：
   · learning_rate + eta0：先试 learning_rate="optimal"（默认，自动衰减）；
     若手动指定，"constant" + eta0=0.01~0.1 是常用起点；
   · 判定标准：看损失曲线。损失震荡/上升 → 学习率太大（除以 10 再试）；
     损失下降太慢 → 学习率太小或轮数不够。
   · alpha：正则化强度，从 1e-4（默认）开始按 10 倍网格搜 {1e-5, 1e-4, 1e-3, 1e-2}；
   · max_iter / tol：先设大一点（1000+、tol=1e-5）确保收敛，再收紧；
   · 铁律：**一定要先标准化特征**，否则各维尺度差异会让梯度下降难以收敛。
3. 数值稳定性：特征强共线时不要手写 (XᵀX)⁻¹（可能报奇异矩阵），
   要么用 np.linalg.lstsq / solve，要么直接用 Ridge。
4. 评估口径：回归调参用交叉验证的 RMSE 或 R²（R² 便于跨数据集比较，RMSE 便于业务解释）。
""")

print(f"\n【完成】01_线性回归.py 运行结束（图片输出目录：{OUTPUT_DIR}）")

r"""《机器学习》课案 —— 06 模型评估与保存 / 02 回归评估指标

对应课案章节
------------
《机器学习》课案 "模型评估与保存" 章 → "评估指标" 节 → "回归评估指标" 小节
（课案原文第 1351~1401 行；本章 5 个代码块中的第 2 个：
 make_regression 造回归数据 + LinearRegression + MSE / RMSE / MAE / R²）。
课案原文的公式在文本抽取后已损坏（LaTeX 变成重复乱码），本文件按含义重写为正确公式。

本节知识点
----------
1. MSE、RMSE、MAE 的定义与量纲分析（谁和 y 同单位、谁好解释）；
2. R² = 1 - SS_res/SS_tot 的推导、为什么 R² 可以为负、R² = 0 到底意味着什么
   （"和永远预测训练集均值一样好"）；
3. 补充指标 MAPE（平均绝对百分比误差）的用法与致命缺陷（真值接近 0 时会爆炸）；
4. 对异常值的敏感度对比：MAE 是线性惩罚，MSE/RMSE 是平方惩罚，
   并用一次"注入单个离群点"的真实实验把差异量化出来。

运行方式（在 PowerShell 中复制执行）
------------------------------------
$env:PYTHONUTF8='1'; & 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Machine_Learning\06_模型评估与保存\02_回归评估指标.py'

产出
----
Machine_Learning/output/06_回归预测对比.png
Machine_Learning/output/06_残差分布图.png
Machine_Learning/output/06_误差敏感度对比.png
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

from sklearn.datasets import make_regression
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

# =============================================================================
# ① 原理与数学推导
# =============================================================================
# 记号：n 个测试样本，真实值 y_i，预测值 ŷ_i，残差 e_i = y_i - ŷ_i，
#       真实值均值 ȳ = (1/n)·Σ y_i。
#
# 1.1 MSE —— 均方误差（Mean Squared Error）
#         MSE = (1/n) · Σ_{i=1..n} (y_i - ŷ_i)²
#     * 量纲：y 的平方（例如目标是"万元"，MSE 的单位是"万元的平方"），
#       所以 MSE = 413 这种数字**没法直接跟业务对话**，只能用于模型之间比较。
#     * 数学性质：对 ŷ 求导令为 0，得 ŷ = ȳ —— 也就是说"最小化 MSE"的最优常数
#       预测就是均值。这正是 MSE 与高斯噪声假设（极大似然）一一对应的原因，
#       也是线性回归（最小二乘）的默认目标函数。
#     * 缺点：平方放大误差 → **对离群点极度敏感**；一个错得离谱的样本能主宰整个指标。
#
# 1.2 RMSE —— 均方根误差（Root Mean Squared Error）
#         RMSE = sqrt(MSE) = sqrt( (1/n) · Σ (y_i - ŷ_i)² )
#     * 量纲：**与 y 相同**（"万元"），因此最容易向业务方解释：
#       "平均而言，我们的预测离真实值差大约 20.3（在 ±326 的目标范围里）"。
#     * 注意：RMSE ≥ MAE 恒成立（由方差非负可证：Var(e) = MSE - MAE² ≥ 0）。
#       两者差距越大，说明误差分布越不均匀、存在少量"错得特别狠"的样本。
#     * 注意：RMSE ≠ "平均误差"。它是"误差平方的平均再开方"，比 MAE 更偏袒大误差。
#
# 1.3 MAE —— 平均绝对误差（Mean Absolute Error）
#         MAE = (1/n) · Σ |y_i - ŷ_i|
#     * 量纲与 y 相同，直观："平均每次预测偏了多少"。
#     * 数学性质：最小化 MAE 的最优常数预测是**中位数**而不是均值，
#       所以 MAE 对离群点稳健（robust）——一个天大的错误也只按线性计入。
#     * 缺点：绝对值在 0 点不可导，梯度法是次梯度、收敛略慢；
#       且它"不区分小错和大错"，若业务上大错不可接受，MAE 会麻痹你。
#
# 1.4 R² —— 决定系数（coefficient of determination）
#         SS_res = Σ (y_i - ŷ_i)²            （残差平方和，模型没解释掉的部分）
#         SS_tot = Σ (y_i - ȳ)²              （总平方和，数据本身的波动）
#         R² = 1 - SS_res / SS_tot
#     读法：模型解释掉了数据波动的多大比例。
#     * R² = 1  ：完美预测，SS_res = 0。
#     * R² = 0  ：SS_res = SS_tot，即**模型的预测和"永远预测 ȳ"一样好**。
#                 注意 ȳ 是"真实值的均值"，实践中常用"训练集均值"当基线；
#                 本脚本用 DummyRegressor(strategy="mean") 实测，正是 R²≈0。
#     * R² < 0  ：**完全可能**！因为 SS_res/SS_tot > 1 时 R² 为负 —— 模型在测试集上
#                 比"直接报均值"还差。（例如用了错误的模型、严重的过拟合、
#                 或者测试集分布与训练集不同。）R² 没有下界，可以是 -0.5，也可以是 -10。
#                 → 教学要点：R² 为负不是 bug，而是"你的模型连均值都不如"的明确信号。
#     * R² 与相关系数 r 的关系：一元线性回归中 R² = r²，但多元回归里没有这个等式；
#       R² 也不代表"因果关系"或"预测一定准"。
#     * 缺点：**加特征永不下降**（哪怕是无用特征），所以比较不同特征数的模型要看
#       调整后 R²：R²_adj = 1 - (1-R²)(n-1)/(n-p-1)，p 为特征数。
#
# 1.5 MAPE —— 平均绝对百分比误差（补充指标）
#         MAPE = (100%/n) · Σ |(y_i - ŷ_i) / y_i|
#     * 优点：无量纲，跨数据集、跨量级可比，业务方最爱（"平均偏差 5%"）。
#     * 致命缺陷：
#       (a) **y_i 接近 0 时爆炸**（除以接近 0 的数），本次数据就会遇到这个问题；
#       (b) 不对称：预测偏大时误差被低估（分母变大的 y_i 仍是真值？不，分母恒为 y_i，
#           所以 |ŷ - y| / y 对"预测偏小"惩罚更重）；
#       (c) 对 y_i = 0 无定义，需要加 eps 或改用 sMAPE。
#
# 1.6 异常值敏感度（本脚本用真实实验演示）
#     只改一个测试样本：把它抬高 400（其余 149 个样本完全不变），观察
#         MAE 的变化 ≈ 400/n        （线性，被 n 稀释，几乎不动）
#         MSE 的变化 ≈ (400² + 2·400·e)/n   （平方，立刻暴涨）
#     这就是"MAE 线性惩罚、MSE/RMSE 平方惩罚"的可测量后果。

# =============================================================================
# ② sklearn API 关键参数逐个解释
# =============================================================================
# 【make_regression】造回归数据
#   n_samples=500     : 样本数。回归指标（尤其 R²）在小样本上非常不稳。
#   n_features=5      : 特征数。注意默认只有 n_informative=10 与 n_features 取小者有效，
#                       这里 5 个特征全部有效（目标由它们线性组合 + 噪声生成）。
#   noise=20          : **加到目标上的高斯噪声标准差**（单位与 y 相同）。
#                       noise=0 时 R² 会是 1.0（完美线性可分）；
#                       这里 20 意味着"理论上最好的模型"也还剩约 20 的 RMSE，
#                       所以别指望 RMSE 小于 20 —— 这是数据的噪声下限，不是模型的锅。
#   bias=0.0          : 目标整体的偏移量（截距）。
#   random_state=42   : 固定随机种子，保证结果可复现。
#
# 【LinearRegression】普通最小二乘（OLS）
#   fit_intercept=True : 是否拟合截距。**除非数据已中心化，否则保持 True**；
#                        设成 False 会强制过原点，通常让 R² 明显变差。
#   copy_X=True        : 是否复制 X（True 更安全，避免原数据被就地修改）。
#   n_jobs=None        : 仅对多目标（y 为二维）时的矩阵运算并行有意义，单目标无效。
#   positive=False     : 是否约束系数非负。若业务上知道特征与目标同向，设为 True
#                        可提升可解释性（需要 scipy 的 nnls）。
#   tol / solver       : sklearn 1.9 中 solver 仅剩 'svd'/'cholesky'/'lsqr' 等；
#                        默认自动选。**不要再用已移除的 normalize=True 参数**。
#   注意：LinearRegression 几乎没有超参数，所以它的成绩**只反映数据和特征质量**，
#         不需要网格搜索；要调参请换 Ridge/Lasso（带 alpha 正则）。
#
# 【DummyRegressor】零模型基线（评估 R² 时必备的对照）
#   strategy="mean"    : 永远预测训练集均值 → 其 R² 应≈0；"median"/"quantile"/"constant" 同理。
#                        没有基线对照的 R² 意义有限：R²=0.3 究竟算好还是差，
#                        取决于"随手猜一个均值"能拿多少分。
#
# 【四个指标函数】（sklearn 1.9 中的正确用法）
#   mean_squared_error(y_true, y_pred)          : MSE。
#        **注意**：老课的 squared=False 参数**已被移除**，传了会直接报错。
#        要 RMSE 请用 np.sqrt(mean_squared_error(...)) 或
#        sklearn.metrics.root_mean_squared_error(y_true, y_pred)（1.4+ 新增）。
#   mean_absolute_error(y_true, y_pred)         : MAE。
#   r2_score(y_true, y_pred, multioutput="uniform_average")
#                                               : R²。多目标回归时 multioutput 决定怎么汇总
#                                                 （'raw_values' 逐目标返回 / 'variance_weighted' 按方差加权）。
#   mean_absolute_percentage_error(y_true, y_pred)
#                                               : MAPE，返回的是**小数**（0.05 = 5%），
#                                                 要乘 100 才是百分数；y_true 有 0 时会警告/返回极大值。
#   sample_weight                               : 四个函数都支持，用于给样本加权（例如近期数据权重更高）。
#
# 【绘图要点】
#   ax.scatter(y_true, y_pred) + ax.plot([lo,hi],[lo,hi]) : 真实 vs 预测散点，
#       点越贴近 y=x 越好；系统性偏离 y=x 说明模型有偏（bias）。
#   ax.hist(residuals) + ax.axvline(0)                      : 残差分布，
#       应大致以 0 为中心、近似正态；偏斜或长尾说明模型设定有问题。

# =============================================================================
# ③ 完整可运行代码
# =============================================================================
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("=" * 78)
print("06-02 回归评估指标：MSE / RMSE / MAE / R² / MAPE 与异常值敏感度实验")
print("=" * 78)

# --- 步骤 1：生成回归数据（对应课案代码块 2） ---------------------------------
X, y = make_regression(
    n_samples=500,
    n_features=5,
    noise=20,               # 目标上的高斯噪声标准差，决定了 RMSE 的理论下限
    random_state=RANDOM_STATE,
)
print("\n[1] 数据概况")
print(f"    特征矩阵 X = {X.shape}（500 个样本 × 5 个特征），目标 y = {y.shape}")
print(f"    y 的取值范围 [{y.min():.2f}, {y.max():.2f}]，均值 {y.mean():.2f}，"
      f"标准差 {y.std():.2f}")
print(f"    生成时注入的噪声标准差 noise=20 → 任何模型的 RMSE 都很难低于 20，")
print(f"    这是数据自身的噪声下限，不是模型不够好。")

# --- 步骤 2：划分训练 / 测试集 -----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE
)
print(f"\n[2] 数据集划分：训练集 {X_train.shape[0]} 个，测试集 {X_test.shape[0]} 个")

# --- 步骤 3：训练线性回归 ----------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n[3] 训练好的线性回归")
print(f"    截距 intercept_ = {model.intercept_:.4f}")
print("    各特征系数 coef_ = [" + ", ".join(f"{c:.4f}" for c in model.coef_) + "]")
print("    真实数据就是由这 5 个特征线性组合 + 高斯噪声生成的，")
print("    所以线性回归是**正确设定（correctly specified）**的模型，正常应该表现很好；")
print("    系数绝对值越大说明该特征对目标的线性影响越强（注意：特征未标准化时，")
print("    系数大小不能直接比较重要性，因为各特征的量级可能不同）。")

# --- 步骤 4：四个核心指标（对应课案代码块 2） --------------------------------
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)                       # squared=False 已被移除，必须自己开方
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("\n[4] 回归评估指标（测试集）")
print(f"    MSE  均方误差     = {mse:.4f}      （单位是 y 的平方，不直观）")
print(f"    RMSE 均方根误差   = {rmse:.4f}     （单位同 y，最好解释）")
print(f"    MAE  平均绝对误差 = {mae:.4f}      （单位同 y，对离群点稳健）")
print(f"    R²   决定系数     = {r2:.4f}")
print(f"    MAPE 平均绝对百分比误差 = {mape:.4f} = {mape:.2%}")

# --- 步骤 5：手工复算，证明公式没被"库函数黑箱化" ----------------------------
residuals = y_test - y_pred
mse_manual = float(np.mean(residuals ** 2))
mae_manual = float(np.mean(np.abs(residuals)))
ss_res = float(np.sum(residuals ** 2))
ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
r2_manual = 1.0 - ss_res / ss_tot
print("\n[5] 按公式手工复算，与 sklearn 结果对照")
print(f"    MSE  = mean(残差²)          = {mse_manual:.4f}   （sklearn: {mse:.4f}）")
print(f"    RMSE = sqrt(MSE)            = {np.sqrt(mse_manual):.4f}")
print(f"    MAE  = mean(|残差|)          = {mae_manual:.4f}   （sklearn: {mae:.4f}）")
print(f"    SS_res = Σ(y-ŷ)² = {ss_res:.2f}")
print(f"    SS_tot = Σ(y-ȳ)² = {ss_tot:.2f}")
print(f"    R² = 1 - SS_res/SS_tot = 1 - {ss_res:.2f}/{ss_tot:.2f}"
      f" = {r2_manual:.4f}   （sklearn: {r2:.4f}）")
print(f"    残差均值 = {residuals.mean():.4f}（接近 0 说明模型没有系统性高估或低估）")
print(f"    残差标准差 = {residuals.std(ddof=0):.4f}，"
      f"最大绝对残差 = {np.abs(residuals).max():.4f}")

# --- 步骤 6：基线对照 —— R² = 0 到底长什么样 ---------------------------------
dummy = DummyRegressor(strategy="mean")
dummy.fit(X_train, y_train)
y_dummy = dummy.predict(X_test)
dummy_mse = mean_squared_error(y_test, y_dummy)
dummy_mae = mean_absolute_error(y_test, y_dummy)
dummy_r2 = r2_score(y_test, y_dummy)
print("\n[6] 基线对照：一个「永远预测训练集均值」的傻瓜模型")
print(f"    它预测的常数永远是训练集均值 = {y_train.mean():.4f}")
print(f"    它的 RMSE = {np.sqrt(dummy_mse):.4f}，MAE = {dummy_mae:.4f}，R² = {dummy_r2:.4f}")
print(f"    → 它的 R² = {dummy_r2:.4f} ≈ 0，这正是 **R² = 0 的含义**：")
print("      模型的预测效果和「永远报均值」完全一样（分子 SS_res 等于分母 SS_tot）。")
print(f"      实测值是 {dummy_r2:.4f} 而不是恰好 0，是因为这里的均值来自训练集，")
print("      而 SS_tot 是按测试集自己的均值算的，两者有微小差异，完全正常。")
print(f"    → 我们的线性回归 R² = {r2:.4f}，远高于基线的 {dummy_r2:.4f}，")
print(f"      说明它确实学到了特征与目标之间的线性关系，而不是在瞎猜。")

# --- 步骤 7：R² 为负的演示 ---------------------------------------------------
# 用一个故意"学错"的模型：把训练标签打乱后再拟合，它学到的是噪声。
shuffled_y = np.random.default_rng(RANDOM_STATE).permutation(y_train)
bad_model = LinearRegression().fit(X_train, shuffled_y)
bad_pred = bad_model.predict(X_test)
bad_r2 = r2_score(y_test, bad_pred)
bad_mae = mean_absolute_error(y_test, bad_pred)
print("\n[7] 演示 R² 可以为负：故意用「打乱标签」的数据训练一个模型")
print(f"    该模型 RMSE = {np.sqrt(mean_squared_error(y_test, bad_pred)):.4f}，"
      f"MAE = {bad_mae:.4f}，R² = {bad_r2:.4f}")
print(f"    R² = {bad_r2:.4f} < 0，说明它比「直接报均值」（R²≈0）还要差。")
print("    → 结论：**R² 没有下界，可以为负**。看到负的 R² 不要以为算错了，")
print("      它是在明确告诉你：这个模型连最简单的均值基线都不如，必须换模型或查数据。")

# --- 步骤 8：MAPE 的陷阱 -----------------------------------------------------
abs_y_sorted = np.sort(np.abs(y_test))
print("\n[8] 为什么 MAPE = {:.2%} 看起来比别的指标差那么多？".format(mape))
print(f"    MAE = {mae:.2f}，而测试集里 |y| 的平均值是 {np.abs(y_test).mean():.2f}，")
print(f"    如果按「MAE / 平均|y|」粗略估算，相对误差只有 "
      f"{mae / np.abs(y_test).mean():.2%} 左右，远小于 MAPE 的 {mape:.2%}。")
print(f"    原因：MAPE 逐样本除以**该样本的真实值 y_i**。本数据的 y 有正有负、")
print(f"    而且有几个样本的真值非常靠近 0（测试集里最小的 |y| 仅 {abs_y_sorted[0]:.4f}）。")
tiny_idx = int(np.argmin(np.abs(y_test)))
print(f"    最极端的那个样本：y_true = {y_test[tiny_idx]:.4f}，y_pred = {y_pred[tiny_idx]:.4f}，")
print(f"    单样本百分比误差 = {abs(residuals[tiny_idx]) / abs(y_test[tiny_idx]):.2%}，")
print(f"    它一个人就把整体 MAPE 拉高了好几个百分点。")
print(f"    → 结论：**当真值可能接近 0 时不要用 MAPE**。可以改用 sMAPE")
print(f"      （分母换成 (|y|+|ŷ|)/2，天然有界于 0~200%），或先对目标做区间分段再算。")

# --- 步骤 9：异常值敏感度真实实验 -------------------------------------------
y_test_out = y_test.copy()
outlier_pos = int(np.argmax(np.abs(y_test)))        # 挑一个原本就靠边缘的样本
y_test_out[outlier_pos] += 400.0                    # 只把这一个真值抬高 400
mae_out = mean_absolute_error(y_test_out, y_pred)
mse_out = mean_squared_error(y_test_out, y_pred)
rmse_out = np.sqrt(mse_out)
print("\n[9] 异常值敏感度实验：只把 1 个测试样本的真值抬高 400，其余 149 个完全不动")
print(f"    样本下标 {outlier_pos}：原真值 {y_test[outlier_pos]:.4f} → "
      f"新真值 {y_test_out[outlier_pos]:.4f}（预测值不变 = {y_pred[outlier_pos]:.4f}）")
print(f"    指标        原值        注入离群点后    变化幅度")
print(f"    MAE      {mae:>9.4f}   {mae_out:>11.4f}   {(mae_out / mae - 1) * 100:>+7.2f}%")
print(f"    MSE      {mse:>9.4f}   {mse_out:>11.4f}   {(mse_out / mse - 1) * 100:>+7.2f}%")
print(f"    RMSE     {rmse:>9.4f}   {rmse_out:>11.4f}   {(rmse_out / rmse - 1) * 100:>+7.2f}%")
print(f"    MAPE     {mape:>9.4f}   "
      f"{mean_absolute_percentage_error(y_test_out, y_pred):>11.4f}   "
      f"{(mean_absolute_percentage_error(y_test_out, y_pred) / mape - 1) * 100:>+7.2f}%")
print(f"    理论解释：n = {len(y_test)}，MAE 的增加量≈400/{len(y_test)} = "
      f"{400 / len(y_test):.4f}（线性、被样本数稀释）；")
print(f"    而 MSE 的增加量≈400²/{len(y_test)} = {400 ** 2 / len(y_test):.2f}（平方放大）。")
print("    → **MAE 线性惩罚、MSE/RMSE 平方惩罚**，这一条实验数据就是最直接的证据。")
print("      若业务上「一个离谱的预测」不可接受 → 用 MSE/RMSE（它会重罚）；")
print("      若数据里本来就有很多天然离群点、不想被它们带偏 → 用 MAE（或 Huber 损失）。")

# =============================================================================
# ④ 结果解读
# =============================================================================
print("\n" + "=" * 78)
print("④ 结果解读：这些数字到底说明什么")
print("=" * 78)
y_span = float(y_test.max() - y_test.min())
print(f"""
【1】先看量级，才知道好坏——脱离量级谈 RMSE 是没有意义的。
      测试集目标 y 的范围约 [{y_test.min():.1f}, {y_test.max():.1f}]
      （跨度 {y_span:.1f}，标准差 {y_test.std():.1f}，平均绝对值 {np.abs(y_test).mean():.1f}）。
      本次 RMSE = {rmse:.2f}，MAE = {mae:.2f}：
      * RMSE 只占 y 标准差的 {rmse / y_test.std():.1%}、占目标跨度的 {rmse / y_span:.1%}；
      * MAE 只有平均 |y| 的 {mae / np.abs(y_test).mean():.1%}。
      → 用业务语言说：**平均每次预测偏离真值约 {mae:.1f} 个单位**，
        而目标本身在 ±{max(abs(y_test.min()), abs(y_test.max())):.0f} 之间波动，
        这个精度是相当好的。

【2】R² = {r2:.4f}：模型解释了测试集目标波动的 {r2:.2%}。
      作为对照，"永远报均值"的傻瓜模型 R² ≈ 0，所以 {r2:.4f} 是实打实的提升。
      本次 R² 很高并不奇怪：数据本来就是 5 个特征的**线性**组合加噪声生成的，
      而 LinearRegression 正是这个数据的"正确模型"。
      这也提示：R² 高只说明"模型设定与数据生成机制匹配"，不代表模型在真实业务上一定强。

【3】RMSE({rmse:.2f}) 与 MAE({mae:.2f}) 的比值 = {rmse / mae:.3f}。
      理论下界是 1.0（完全均匀的误差），比值越大说明"少数样本错得特别狠"。
      本次比值 {rmse / mae:.2f}，非常接近 1；再看残差最大绝对值 {np.abs(residuals).max():.2f}，
      它是残差标准差 {residuals.std():.2f} 的 {np.abs(residuals).max() / residuals.std():.1f} 倍 ——
      对 {len(y_test)} 个近似正态的残差来说，最大偏差落在 3 倍标准差以内属于正常范围。
      可以判断：**误差分布比较均匀，没有严重的离群预测**，
      RMSE 和 MAE 的结论一致、互相印证。
      如果哪次看到 RMSE 是 MAE 的两三倍，就要去查那几个残差最大的样本了。

【4】RMSE ≈ {rmse:.2f}，而生成数据时注入的噪声标准差是 20。
      两者非常接近 —— 这说明模型已经**逼近了数据的信息上限**，
      再换更复杂的模型（例如随机森林、XGBoost）也很难把 RMSE 明显压低，
      因为剩余的误差是数据自身的高斯噪声，不是模型没学好。
      这是回归任务里非常重要的一条判断准则：
      **先估算噪声下限，再判断模型还有没有提升空间**，避免做无用功。
      注意 RMSE({rmse:.2f}) 略大于 noise(20)，因为我们还要额外估计 5 个系数，
      参数估计误差会贡献一点额外的预测误差。

【5】MAPE = {mape:.2%} 与其他指标"看起来矛盾"，这是 MAPE 自身缺陷造成的：
      它的分母是每个样本的真值，而本数据的 y 有正有负、有几个真值极其接近 0，
      单样本百分比误差可以飙到几百甚至上千个百分点，一两个这样的样本就能毁掉整体 MAPE。
      → 教训：**报告指标前必须检查它对当前数据是否良定义**。
        当 |y| 可能接近 0 时，MAPE 不可用，应改用 sMAPE、MASE，
        或者干脆报告"MAE / 目标均值"这类相对误差。

【6】该向业务方报哪个指标？
      * 首选 **RMSE**：与目标同单位，一句话就能说清"平均偏多少"；
        而且它和模型优化目标（MSE）一致，不会出现"训练时优化 A、汇报时看 B"。
      * **MAE** 适合"误差有长尾、不想被极值主导"的场合，也更接近"典型误差"的直觉。
      * **R²** 适合回答"模型比瞎猜强多少"，用于模型之间、特征集之间横向比较。
      * **MAPE** 只在真值远离 0 且业务方习惯百分比时使用，并务必注明它的定义。
      * 同时报告 RMSE + MAE + R² 是最稳妥的做法，三个数互相印证，防止误判。
""")

# --- 绘图 1：真实值 vs 预测值 -------------------------------------------------
fig, ax = plt.subplots(figsize=(6.8, 6.0))
ax.scatter(y_test, y_pred, s=34, alpha=0.65, color="#2980b9",
           edgecolors="white", linewidths=0.6, label="测试集样本")
lim_lo = float(min(y_test.min(), y_pred.min())) - 20
lim_hi = float(max(y_test.max(), y_pred.max())) + 20
ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], linestyle="--", color="#c0392b",
        linewidth=1.8, label="理想参考线 y = x（完美预测）")
ax.set_xlim(lim_lo, lim_hi)
ax.set_ylim(lim_lo, lim_hi)
ax.set_title(f"06 线性回归：真实值 vs 预测值（R² = {r2:.4f}，RMSE = {rmse:.2f}）",
             fontsize=13, pad=12)
ax.set_xlabel("真实值 y", fontsize=11)
ax.set_ylabel("预测值 ŷ", fontsize=11)
ax.legend(loc="upper left", fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
p1 = OUTPUT_DIR / "06_回归预测对比.png"
fig.savefig(p1, dpi=130)
plt.close(fig)
print(f"\n[图 1] 已保存真实/预测对比图：{p1}")

# --- 绘图 2：残差分布 + 残差 vs 预测值 ---------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
ax0 = axes[0]
ax0.hist(residuals, bins=26, color="#27ae60", alpha=0.8, edgecolor="white")
ax0.axvline(0.0, color="#c0392b", linestyle="--", linewidth=1.8, label="残差为 0（完美预测）")
ax0.axvline(residuals.mean(), color="#8e44ad", linestyle=":", linewidth=1.8,
            label=f"残差均值 = {residuals.mean():.2f}")
ax0.set_title(f"06 残差分布直方图（标准差 {residuals.std():.2f}）", fontsize=12, pad=10)
ax0.set_xlabel("残差 = 真实值 - 预测值", fontsize=11)
ax0.set_ylabel("样本个数", fontsize=11)
ax0.legend(fontsize=9)
ax0.grid(alpha=0.3)

ax1 = axes[1]
ax1.scatter(y_pred, residuals, s=32, alpha=0.65, color="#e67e22",
            edgecolors="white", linewidths=0.6)
ax1.axhline(0.0, color="#c0392b", linestyle="--", linewidth=1.8)
ax1.set_title("06 残差 vs 预测值（理想情况应是无规律的带状）", fontsize=12, pad=10)
ax1.set_xlabel("预测值 ŷ", fontsize=11)
ax1.set_ylabel("残差 = y - ŷ", fontsize=11)
ax1.grid(alpha=0.3)
fig.tight_layout()
p2 = OUTPUT_DIR / "06_残差分布图.png"
fig.savefig(p2, dpi=130)
plt.close(fig)
print(f"[图 2] 已保存残差分析图：{p2}")

# --- 绘图 3：异常值敏感度对比 -------------------------------------------------
fig, ax = plt.subplots(figsize=(8.2, 5.2))
labels = ["MAE\n(线性惩罚)", "MSE\n(平方惩罚)", "RMSE\n(平方惩罚后开方)"]
before = [mae, mse, rmse]
after = [mae_out, mse_out, rmse_out]
xpos = np.arange(len(labels))
width = 0.36
bars1 = ax.bar(xpos - width / 2, before, width, label="原始测试集", color="#2980b9", alpha=0.9)
bars2 = ax.bar(xpos + width / 2, after, width, label="注入 1 个 +400 的离群点后", color="#c0392b", alpha=0.9)
for bars in (bars1, bars2):
    for bar in bars:
        ax.annotate(f"{bar.get_height():.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
ax.set_yscale("log")   # MSE 量级远大于 MAE，用对数轴才能同时看清
ax.set_xticks(xpos)
ax.set_xticklabels(labels, fontsize=10)
ax.set_title("06 异常值敏感度：只改 1 个样本（共 150 个），指标变化差多少", fontsize=13, pad=12)
ax.set_ylabel("指标取值（对数坐标）", fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
p3 = OUTPUT_DIR / "06_误差敏感度对比.png"
fig.savefig(p3, dpi=130)
plt.close(fig)
print(f"[图 3] 已保存异常值敏感度对比图：{p3}")

# =============================================================================
# 超参数怎么调 / 使用注意
# =============================================================================
# 【线性回归几乎没有超参数，所以"调参"的重点不在模型本身】
# 1. 换模型而不是调参：若 R² 明显偏低，先试 Ridge / Lasso / ElasticNet（带 alpha 正则），
#    alpha 用 GridSearchCV 调（见本目录 04_网格搜索超参数调优.py），
#    再考虑树模型（RandomForest / XGBoost / LightGBM）。
# 2. 特征工程通常比调参收益大得多：标准化、去极值、加交互项、对数变换目标（缓解长尾）。
#    注意：标准化必须放进 Pipeline，否则交叉验证时会数据泄漏（见 05_模型保存与加载.py）。
# 3. 指标选择本身就是"超参数"：
#    * loss="squared_error" 优化 MSE，等价于假设噪声是高斯分布；
#    * loss="absolute_error" 优化 MAE，等价于假设噪声是拉普拉斯分布，抗离群点；
#    * 若离群点很多又想兼顾，用 Huber 损失（sklearn 的 SGDRegressor(loss="huber")）。
# 4. 使用注意：
#    * 务必报告**基线**（DummyRegressor），否则 R² 的高低无法解读。
#    * 别往 mean_squared_error 里传 squared=False —— 该参数已移除，会直接报 TypeError。
#    * RMSE 与 y 同单位、MSE 不同，汇报时优先给 RMSE 并注明单位。
#    * MAPE 遇到 0 或接近 0 的真值会失真，先检查 np.abs(y_true).min() 再决定用不用。
#    * 测试集只用来最后评估一次；调参请用交叉验证（本目录 03 / 04 号脚本）。
#    * 残差图比单个数字更能暴露问题：若残差呈漏斗形（heteroscedasticity）或弯曲，
#      说明模型设定有问题（该做变换或加特征），而不是"再调调参就好"。
print("\n【完成】02_回归评估指标.py 运行结束")

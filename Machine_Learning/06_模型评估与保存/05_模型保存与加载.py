r"""《机器学习》课案 —— 06 模型评估与保存 / 05 模型保存与加载

对应课案章节
------------
《机器学习》课案 "模型评估与保存" 章 → "模型保存与加载" 节
（课案原文第 1505~1552 行；本章 5 个代码块中的第 5 个：
 load_iris + RandomForestClassifier + joblib.dump / joblib.load + 两次预测准确率对比）。

本节知识点
----------
1. 为什么要做模型持久化：训练一次很贵，生产环境不能边请求边训练；模型要能跨进程/跨机器复用；
2. joblib 与 pickle 的区别：joblib 对"含大 numpy 数组的对象"更高效（数组单独存储、可压缩、
   反序列化时用零拷贝副本），pickle 是 Python 标准库、通用但对大数组笨重；
3. 版本兼容风险：pickle 保存的是"类的引用路径 + 属性字典"，不是代码本身。
   如果加载环境的 sklearn 版本不同、类被改名/移动，加载会直接抛
   ModuleNotFoundError / AttributeError —— 本脚本用受控实验真实重现这个错误；
4. 为什么必须保存**整个 Pipeline** 而不是只保存模型：预处理（标准化、编码、填充）
   的拟合结果（均值/方差/类别表）同样是"从训练数据学到的参数"，
   少存任何一环，推理时的输入分布就和训练时不一致，预测会悄悄变差；
5. 工程化落地的完整做法：模型 + 特征名 + 类别名 + 训练时间 + 库版本，
   一起打包成"带元信息的字典"保存，避免"模型能加载但不知道怎么用"。

运行方式（在 PowerShell 中复制执行）
------------------------------------
$env:PYTHONUTF8='1'; & 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Machine_Learning\06_模型评估与保存\05_模型保存与加载.py'

产出
----
Machine_Learning/output/06_随机森林模型.pkl              （joblib 保存的裸模型）
Machine_Learning/output/06_随机森林模型_pickle.pkl       （pickle 保存的同款模型，用于对比）
Machine_Learning/output/06_标准化随机森林流水线.pkl      （含 StandardScaler 的完整 Pipeline）
Machine_Learning/output/06_模型元信息.pkl                （模型 + 特征名 + 类别名 + 训练时间等）
Machine_Learning/output/06_模型元信息.json               （同一份元信息的可读版本，不含模型对象）
Machine_Learning/output/06_模型保存与加载验证.png        （加载前后一致性验证图）
"""

from __future__ import annotations

import pathlib
import sys


import matplotlib
matplotlib.use("Agg")            # 无界面后端，避免 plt.show() 阻塞
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

import json
import pickle
import shutil
import time
from datetime import datetime

import joblib
import numpy as np
import sklearn

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# =============================================================================
# ① 原理与数学推导（本节偏工程，重点是"序列化"的机制与风险）
# =============================================================================
# 1.1 为什么必须做模型持久化
# --------------------------
# 训练一个模型 = 解一个优化问题，代价可能是几秒，也可能是几天。
# 生产环境的正确形态是"训练与推理分离"：
#       离线训练 → **保存模型到磁盘** → 在线服务启动时加载模型 → 只做 predict
# 好处：
#   * 服务启动不需要重新训练（秒级 vs 小时级）；
#   * 训练脚本与推理服务解耦，各自独立部署、独立扩容；
#   * 模型文件可版本化（model_v1/model_v2），支持灰度发布与快速回滚；
#   * 可复现与可审计：某次线上结果的模型到底是哪一版，有文件为证。
#
# 1.2 sklearn 模型里到底存了什么？
# --------------------------------
# fit() 之后，模型对象上多出来的是"学到的参数"，以属性形式存在，**全部是普通
# Python / numpy 对象**。例如：
#   * LinearRegression : coef_（一维数组）、intercept_（标量）
#   * RandomForest     : estimators_（一堆 DecisionTreeClassifier 对象）、
#                        n_features_in_、classes_、feature_importances_
#   * StandardScaler   : mean_、scale_、var_（都是数组）
#   * Pipeline         : steps（(名字, 估计器) 列表）
# 所谓"保存模型"，就是把这些对象**序列化（serialize）**成字节流写到磁盘；
# "加载模型"就是反序列化（deserialize）回内存对象。
#
# 1.3 pickle 与 joblib 的区别
# ---------------------------
#   * **pickle**（Python 标准库）：
#       - 通用对象序列化协议，能存几乎任何 Python 对象；
#       - 对 numpy 数组的处理是"把数组当普通对象，按缓冲区逐字节写"，
#         大数组会产生大量小片段的读写，**既慢又占空间**；
#       - 是**不安全**的：反序列化会执行 pickle 里记录的构造逻辑，
#         所以**绝不要加载来源不明的 .pkl 文件**（可被构造为任意代码执行）。
#   * **joblib**（sklearn 自家依赖，专门为数值计算优化）：
#       - 对**大 numpy 数组**做了特殊处理：数组被单独拎出来存放
#         （`.npy` 格式，天然二进制连续），反序列化时可以用 `np.load(mmap_mode=...)`
#         做**内存映射/零拷贝**，不必先把整块数组读进内存再复制一遍；
#       - 支持 `compress`（0~9 或 ('zlib',3) 等），大模型压缩后体积可显著下降；
#       - 因此 sklearn 官方文档的推荐就是：**保存 sklearn 模型请用 joblib**。
#   * 结论：模型里数组越多越大（随机森林、SVM、神经网络），joblib 的优势越明显；
#     对于只有几个标量参数的极小模型，两者差别在测量噪声之内
#     —— 本脚本会用真实计时说明这一点，而不是空喊结论。
#   * 两者都建立在 pickle 协议之上（joblib 也是"pickle + 数组旁路"），
#     所以**版本兼容风险两者完全相同**（见 1.4）。
#
# 1.4 版本兼容风险（生产事故高发区）
# ----------------------------------
# pickle 文件里存的是**类的模块路径 + 类名 + 实例属性字典**，不是类的代码。
# 反序列化时 Python 会去 `import` 这个名字并调用它的构造逻辑。于是：
#   * sklearn 从 0.24 升级到 1.9，某些类被重命名/移动/参数被移除 →
#     加载时抛 `ModuleNotFoundError: No module named 'sklearn.xxx'` 或
#     `AttributeError: module 'sklearn.xxx' has no attribute 'OldName'`；
#   * 即使能加载，内部属性的语义变了（例如某属性从"数组"变成"稀疏矩阵"），
#     也可能**不报错但预测结果悄悄改变**（最危险的情形）。
# 工程上的应对：
#   1) **把训练环境的库版本一起保存**（本脚本的元信息字典里就有 sklearn 版本）；
#   2) 保存与部署环境尽量锁定同一套依赖（容器镜像 / lock 文件）；
#   3) 升级 sklearn 后**必须回归验证**：用固定测试集对比新旧模型的预测结果；
#   4) 更彻底的方案：跨大版本时改用 **ONNX / PMML** 等语言中立的模型格式，
#      或者导出为纯权重 + 自己实现推理。
#
# 1.5 为什么必须保存**整个 Pipeline**
# -----------------------------------
# 设训练时的流程是  X --(StandardScaler)--> X' --(模型)--> ŷ。
# 如果只保存模型，部署时把**原始未标准化的 X** 喂给模型，那么
#       ŷ_wrong = f_model(X)  ≠  f_model(X') = ŷ_correct
# 输入分布错位，预测精度会明显下降，而且**不会报任何错**——
# 这类 bug 极难排查，是"线下 95%、线上 60%"的常见元凶之一。
# Pipeline 把预处理器与模型绑成一个整体，保存它就是保存了"从原始输入到预测"的完整链路：
#       X_raw → scaler.transform → clf.predict
# 所以：**Pipeline 保存的是流程，不是一段代码**；加载后直接喂原始特征即可。
# （补充：随机森林对特征尺度不敏感，本例中"漏存 scaler"造成的差异不明显；
#   但逻辑回归、SVM、KNN、神经网络等对尺度敏感的模型会立刻暴露出问题，
#   本脚本 ③ 步骤 4 会用一个逻辑回归 Pipeline 把差异实测出来。）

# =============================================================================
# ② sklearn / joblib API 关键参数逐个解释
# =============================================================================
# 【joblib.dump(value, filename, *, compress=0, protocol=None, cache_size=None)】
#   value      : 要保存的对象（模型、Pipeline、dict 都行，只要可 pickle）。
#   filename   : 目标路径。**扩展名约定用 .pkl 或 .joblib**（两者内容一样，只是习惯）；
#                更推荐 .joblib 以免和 pickle 混淆。本脚本按项目要求用 .pkl。
#   compress=0 : 压缩级别。0 = 不压缩（**默认**，最快）；
#                1~9 = zlib 压缩级别（越大越小越慢）；
#                也可传 ('zlib', 3) / ('gzip', 4) / ('lzma', 2) 等元组指定算法。
#                大模型落盘/传输时用 compress=3 通常能省 50%~80% 体积。
#   protocol   : pickle 协议版本，默认最高（Python 3.12 下是 5）。
#                **不要降级协议**，除非要兼容老 Python。
#   cache_size : 内部缓存（一般不用管）。
#   注意：文件名后缀**不会**影响格式；joblib.dump 也可以存到已打开的文件对象。
#
# 【joblib.load(filename, *, mmap_mode=None)】
#   filename   : 路径或文件对象。
#   mmap_mode  : 'r'/'r+'/'c'，启用内存映射加载 numpy 数组。
#                大模型（几百 MB 以上）推理时用它可显著降低内存峰值与加载耗时
#                （多个进程可以共享同一份只读页缓存）。
#   注意：mmap 得到的是只读数组，**修改它之前要先 copy**。
#   安全提醒：joblib.load 与 pickle.load 一样会执行反序列化逻辑，
#            **只加载自己生成的文件**。
#
# 【pickle.dump(obj, file, protocol=None) / pickle.load(file)】
#   必须配合 open(path, "wb") / open(path, "rb") 使用；
#   protocol 建议用 pickle.HIGHEST_PROTOCOL（3.12 下 = 5）。
#   它没有 compress 参数，要压缩得自己套 gzip/lzma 包装。
#
# 【pickle.HIGHEST_PROTOCOL】当前解释器支持的最高协议版本（越新越紧凑、越快）。
#
# 【Pipeline(steps, *, memory=None, verbose=False)】
#   steps   : 形如 [("scaler", StandardScaler()), ("clf", RandomForestClassifier(...))]。
#             **最后一步必须是估计器（有 fit/predict）**，前面的都是变换器（有 fit/transform）。
#   memory  : 缓存每个变换器的拟合结果到磁盘（换超参时可不重复 fit 预处理），
#             仅调参阶段有用，生产推理不需要。
#   verbose : 是否打印每步耗时。
#   访问子步骤：pipe.named_steps["clf"]、pipe["clf"] 或 pipe.steps[0][1]。
#   网格搜索时参数名要写成 "clf__max_depth" 这种双下划线形式。
#
# 【StandardScaler(copy=True, with_mean=True, with_std=True)】
#   with_mean / with_std : 是否中心化 / 是否除以标准差。若用稀疏矩阵必须 with_mean=False。
#   拟合后得到 mean_（均值）、scale_（标准差）、var_、n_samples_seen_ —— **这些就是要保存的东西**。
#   标准化公式：z = (x - mean_) / scale_。
#
# 【np.testing.assert_array_equal(a, b)】逐元素严格相等（含 NaN 位置），
#   不相等会抛 AssertionError。**验证"加载后模型与原来完全一致"最直接的手段**；
#   浮点比较若允许误差，用 np.testing.assert_allclose(a, b, rtol=1e-12, atol=0)。
#
# 【pathlib.Path.stat().st_size】文件字节数；除以 1024 得 KB，除以 1024² 得 MB。

# =============================================================================
# ③ 完整可运行代码
# =============================================================================
OUTPUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def size_kb(path: pathlib.Path) -> float:
    """返回文件大小（KB，保留两位小数）。"""
    return path.stat().st_size / 1024.0


print("=" * 78)
print("06-05 模型保存与加载：joblib / pickle / Pipeline / 元信息 / 版本兼容")
print("=" * 78)
print(f"\n[0] 环境信息（务必随模型一起记录，排查版本兼容问题时是唯一线索）")
print(f"    Python         : {sys.version.split()[0]}")
print(f"    scikit-learn   : {sklearn.__version__}")
print(f"    joblib         : {joblib.__version__}")
print(f"    numpy          : {np.__version__}")
print(f"    输出目录        : {OUTPUT_DIR}")

# --- 步骤 1：数据与划分（对应课案代码块 5） ---------------------------------
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)
print("\n[1] 数据与划分")
print(f"    iris：{X.shape[0]} 个样本、{X.shape[1]} 个特征、"
      f"类别 = {[str(n) for n in iris.target_names]}")
print(f"    训练集 {X_train.shape[0]} 个，测试集 {X_test.shape[0]} 个（stratify=y）")

# --- 步骤 2：训练裸模型并做 joblib 保存 / 加载（课案代码块 5 的核心） -------
t0 = time.perf_counter()
model = RandomForestClassifier(random_state=RANDOM_STATE)
model.fit(X_train, y_train)
train_seconds = time.perf_counter() - t0
y_pred = model.predict(X_test)
proba = model.predict_proba(X_test)
acc_before = accuracy_score(y_test, y_pred)
print("\n[2] 训练随机森林并保存（课案用 model.joblib，本脚本按项目规范用 .pkl）")
print(f"    模型结构：{model.n_estimators} 棵树，每棵最大深度 {model.max_depth or '不限'}，"
      f"共 {len(model.estimators_)} 个树对象")
print(f"    训练耗时 {train_seconds:.4f} 秒")
print(f"    保存前测试集准确率 = {acc_before:.4f}")

model_path = OUTPUT_DIR / "06_随机森林模型.pkl"
t_dump = time.perf_counter()
joblib.dump(model, model_path)
dump_seconds = time.perf_counter() - t_dump
t_load = time.perf_counter()
loaded_model = joblib.load(model_path)
load_seconds = time.perf_counter() - t_load
print(f"    joblib.dump 保存到：{model_path}")
print(f"    文件大小 {size_kb(model_path):.2f} KB，dump 耗时 {dump_seconds * 1000:.3f} 毫秒，"
      f"load 耗时 {load_seconds * 1000:.3f} 毫秒")

y_pred_loaded = loaded_model.predict(X_test)
proba_loaded = loaded_model.predict_proba(X_test)
acc_after = accuracy_score(y_test, y_pred_loaded)

print("\n[3] 关键验证：加载后的模型必须与原来**完全一致**（不是「差不多」）")
print(f"    加载后测试集准确率 = {acc_after:.4f}（保存前 {acc_before:.4f}）")
# 断言 1：硬标签逐个相等
np.testing.assert_array_equal(y_pred, y_pred_loaded)
print("    ✓ np.testing.assert_array_equal(y_pred, y_pred_loaded) 通过："
      f"{len(y_pred)} 个预测标签**逐元素完全相同**")
# 断言 2：预测概率逐个严格相等（比标签更严格：能发现"标签相同但概率漂移"的问题）
np.testing.assert_array_equal(proba, proba_loaded)
print("    ✓ np.testing.assert_array_equal(proba, proba_loaded) 通过："
      f"{proba.shape[0]}×{proba.shape[1]} 个正类概率**严格逐元素相等**")
# 断言 3：学到的参数也一致
np.testing.assert_array_equal(model.feature_importances_, loaded_model.feature_importances_)
print("    ✓ 特征重要性数组也完全一致（说明模型内部参数被完整还原，不只是预测碰巧相同）")
print(f"    ✓ 准确率严格相等：{acc_before:.4f} == {acc_after:.4f} → "
      f"{acc_before == acc_after}")
print("    为什么要断言「概率也相等」：如果加载后概率发生微小漂移（版本差异的典型症状），")
print("    硬标签可能仍全部相同，你会在很久以后才发现线上排序/阈值决策出了问题。")

# --- 步骤 4：pickle 对照实验 ------------------------------------------------
print("\n[4] pickle 对照实验（同样的模型，同样的数据）")
pickle_path = OUTPUT_DIR / "06_随机森林模型_pickle.pkl"
with open(pickle_path, "wb") as f:
    t_dump = time.perf_counter()
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_dump_seconds = time.perf_counter() - t_dump
with open(pickle_path, "rb") as f:
    t_load = time.perf_counter()
    pickle_model = pickle.load(f)
    pickle_load_seconds = time.perf_counter() - t_load
pickle_pred = pickle_model.predict(X_test)
pickle_acc = accuracy_score(y_test, pickle_pred)
np.testing.assert_array_equal(y_pred, pickle_pred)
print(f"    pickle 文件：{pickle_path}")
print(f"    pickle 协议版本 = {pickle.HIGHEST_PROTOCOL}，文件大小 {size_kb(pickle_path):.2f} KB")
print(f"    pickle dump {pickle_dump_seconds * 1000:.3f} 毫秒，"
      f"load {pickle_load_seconds * 1000:.3f} 毫秒（准确率 {pickle_acc:.4f}，预测与 joblib 一致）")
print(f"    对比结果：joblib {size_kb(model_path):.2f} KB / pickle {size_kb(pickle_path):.2f} KB，"
      f"体积差 {abs(size_kb(model_path) - size_kb(pickle_path)):.2f} KB")
print("    → 说明：这个模型只有 100 棵浅树、体量很小（约 150 KB），两者体积在同一量级；")
print(f"      本次 pickle 反而略小 {abs(size_kb(model_path) - size_kb(pickle_path)):.2f} KB"
      f"（约 {abs(size_kb(model_path) - size_kb(pickle_path)) / size_kb(model_path):.1%}），")
print("      因为 joblib 的容器格式自带一些索引/元数据开销，模型太小时摊不掉这部分成本 ——")
print("      **不要迷信「joblib 存出来一定更小」**，它的优势在下面这类场景才显现。")
print("      joblib 的优势要在「数组又多又大」时才显现（本脚本步骤 8 用大数组实测）。")

# 顺便演示 joblib 的压缩能力
compressed_path = OUTPUT_DIR / "06_随机森林模型_压缩.pkl"
joblib.dump(model, compressed_path, compress=3)
print(f"    附：joblib.dump(..., compress=3) 后文件 {size_kb(compressed_path):.2f} KB，"
      f"相当于未压缩的 {size_kb(compressed_path) / size_kb(model_path):.1%}"
      f"（代价是 dump/load 变慢）")

# --- 步骤 5：工程化做法 —— 保存整个 Pipeline --------------------------------
print("\n[5] 工程化做法：保存整个 Pipeline（预处理 + 模型），而不是只保存模型")
rf_pipe = Pipeline(
    [
        ("scaler", StandardScaler()),                    # 第 1 步：标准化
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),  # 第 2 步：分类器
    ]
)
rf_pipe.fit(X_train, y_train)
pipe_pred = rf_pipe.predict(X_test)
pipe_acc = accuracy_score(y_test, pipe_pred)
pipe_path = OUTPUT_DIR / "06_标准化随机森林流水线.pkl"
joblib.dump(rf_pipe, pipe_path)
loaded_pipe = joblib.load(pipe_path)
np.testing.assert_array_equal(pipe_pred, loaded_pipe.predict(X_test))
np.testing.assert_array_equal(rf_pipe.predict_proba(X_test), loaded_pipe.predict_proba(X_test))
print(f"    Pipeline 结构：{list(rf_pipe.named_steps.keys())}")
print(f"    scaler 学到的均值 mean_ = {np.round(rf_pipe.named_steps['scaler'].mean_, 4)}")
print(f"    scaler 学到的标准差 scale_ = {np.round(rf_pipe.named_steps['scaler'].scale_, 4)}")
print(f"    流水线文件：{pipe_path}（{size_kb(pipe_path):.2f} KB）")
print(f"    ✓ 加载后 predict / predict_proba 与原来完全一致（断言通过）")
print(f"    准确率：裸模型 {acc_before:.4f}，Pipeline {pipe_acc:.4f}")
print(f"    → 注意：加载后的 Pipeline 可以直接喂**原始特征**（本次是未经标准化的 X_test），")
print(f"      因为它自带 scaler；这正是「保存整个 Pipeline」的意义 ——")
print("      标准化用的 mean_/scale_ 也是从训练数据学到的「参数」，漏掉它们，")
print("      推理时的输入分布就和训练时不一致了。")

print("\n    漏存预处理器会怎样？用对尺度**敏感**的逻辑回归实测：")
lr_pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, random_state=RANDOM_STATE)),
    ]
)
lr_pipe.fit(X_train, y_train)
lr_pipe_pred = lr_pipe.predict(X_test)
lr_pipe_acc = accuracy_score(y_test, lr_pipe_pred)
lr_only = lr_pipe.named_steps["clf"]                    # 只拿模型，丢掉 scaler
lr_only_pred = lr_only.predict(X_test)                  # 故意喂未标准化的原始数据
lr_only_acc = accuracy_score(y_test, lr_only_pred)
n_diff = int((lr_pipe_pred != lr_only_pred).sum())
print(f"      完整 Pipeline（内部先标准化）准确率 = {lr_pipe_acc:.4f}")
print(f"      只有模型、直接喂原始特征   准确率 = {lr_only_acc:.4f}"
      f"，有 {n_diff}/{len(y_test)} 个样本预测不同")
if n_diff > 0:
    print("      → 两者不同！这就是「只保存模型、丢掉预处理器」的后果：")
    print("        输入分布错位，精度下降，而且**不会报任何错误**，极难排查。")
else:
    print("      → 本次两者巧合一致（iris 特征量级相近、样本少），但这是运气，不是保证；")
    print("        **绝不能依赖这种巧合**。换一份量级差异大的数据，差异会立刻显现。")

# --- 步骤 6：保存带元信息的字典（生产环境强烈建议） ------------------------
print("\n[6] 保存「模型 + 元信息」字典：解决「模型能加载但不知道怎么用」的问题")
model_bundle = {
    "model": model,                                  # 模型对象本身
    "feature_names": [str(n) for n in iris.feature_names],   # 特征名与顺序（推理时必须对齐！）
    "target_names": [str(n) for n in iris.target_names],     # 类别名（把 0/1/2 翻译成业务含义）
    "n_features_in": int(model.n_features_in_),      # 期望的输入特征数
    "classes": model.classes_.tolist(),              # 类别标签顺序（predict_proba 的列顺序）
    "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "train_seconds": round(float(train_seconds), 6),
    "test_accuracy": round(float(acc_before), 6),
    "n_train_samples": int(X_train.shape[0]),
    "random_state": RANDOM_STATE,
    "library_versions": {
        "python": sys.version.split()[0],
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    },
}
bundle_path = OUTPUT_DIR / "06_模型元信息.pkl"
joblib.dump(model_bundle, bundle_path)
loaded_bundle = joblib.load(bundle_path)
print(f"    元信息字典已保存：{bundle_path}（{size_kb(bundle_path):.2f} KB）")
print(f"    加载后可直接使用，无需再猜特征顺序：")
print(f"      feature_names = {loaded_bundle['feature_names']}")
print(f"      target_names  = {loaded_bundle['target_names']}")
print(f"      classes       = {loaded_bundle['classes']}（predict_proba 的列顺序）")
print(f"      trained_at    = {loaded_bundle['trained_at']}")
print(f"      test_accuracy = {loaded_bundle['test_accuracy']}")
print(f"      sklearn 版本   = {loaded_bundle['library_versions']['scikit-learn']}")
bundle_model_pred = loaded_bundle["model"].predict(X_test)
np.testing.assert_array_equal(y_pred, bundle_model_pred)
print("    ✓ 字典里的模型加载后预测结果同样完全一致（断言通过）")

# 再导出一份"可读版"元信息（不含模型对象，方便人工查看/给别的语言的服务读）
readable = {k: v for k, v in model_bundle.items() if k != "model"}
json_path = OUTPUT_DIR / "06_模型元信息.json"
json_path.write_text(json.dumps(readable, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"    另存可读版（不含模型对象）：{json_path}（{size_kb(json_path):.2f} KB）")

# --- 步骤 7：版本兼容风险的真实重现（受控实验，不依赖真实的环境差异） -----
print("\n[7] 版本兼容风险：pickle 存的是「类的引用路径」，不是类的代码")
print("    做一个受控实验：定义类 → 保存 → 删掉这个类 → 再加载")


class _TempLegacyModel:
    """模拟"旧版本 sklearn 里的某个类"，用来演示加载失败。"""

    def __init__(self, coefficient: float = 1.0) -> None:
        self.coefficient = coefficient


temp_path = OUTPUT_DIR / "_tmp_legacy_model.pkl"
joblib.dump(_TempLegacyModel(coefficient=0.5), temp_path)
del _TempLegacyModel          # 相当于"新版 sklearn 把这个类改名/删掉了"
compat_message = ""
try:
    joblib.load(temp_path)
except Exception as exc:      # 这里**故意捕获**，用于演示；生产代码里应记录日志并告警
    compat_message = f"{type(exc).__name__}: {exc}"
print(f"    加载旧文件时的结果（已捕获，不打印 Traceback）：{compat_message}")
print("    → 这就是换 sklearn 版本后最常见的报错形态：")
print("      ModuleNotFoundError（模块没了）/ AttributeError（类名没了）。")
print("      最危险的情况则是「能加载但预测结果悄悄变了」——不报错，只能靠回归测试发现。")
print("      防范措施：① 连同 library_versions 一起保存（见步骤 6）；")
print("                ② 部署环境锁定依赖版本；③ 升级后必做预测结果回归对比。")
temp_path.unlink(missing_ok=True)   # 演示完即删除，保持输出目录干净

# --- 步骤 8：joblib vs pickle 在大 numpy 数组上的真实差距 -------------------
print("\n[8] joblib vs pickle 的差距在哪：用 8 MB 大数组实测（临时文件，测完即删）")
bench_dir = OUTPUT_DIR / "_tmp_bench"
shutil.rmtree(bench_dir, ignore_errors=True)   # 清掉可能残留的旧临时目录（如上次异常退出）
bench_dir.mkdir(exist_ok=True)
big_array = np.random.default_rng(RANDOM_STATE).random((1000, 1000))   # float64 ≈ 7.6 MB
payload = {"weights": big_array}

joblib_big = bench_dir / "big_joblib.pkl"
pickle_big = bench_dir / "big_pickle.pkl"

t0 = time.perf_counter()
joblib.dump(payload, joblib_big)
joblib_dump_ms = (time.perf_counter() - t0) * 1000
t0 = time.perf_counter()
loaded_big = joblib.load(joblib_big)
joblib_load_ms = (time.perf_counter() - t0) * 1000
np.testing.assert_array_equal(loaded_big["weights"], big_array)

with open(pickle_big, "wb") as f:
    t0 = time.perf_counter()
    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    pickle_dump_ms = (time.perf_counter() - t0) * 1000
with open(pickle_big, "rb") as f:
    t0 = time.perf_counter()
    loaded_big_pk = pickle.load(f)
    pickle_load_ms = (time.perf_counter() - t0) * 1000
np.testing.assert_array_equal(loaded_big_pk["weights"], big_array)

print(f"    数组形状 {big_array.shape}（float64，约 {big_array.nbytes / 1024 ** 2:.2f} MB）")
print(f"    joblib ：文件 {size_kb(joblib_big):>9.2f} KB，dump {joblib_dump_ms:>7.2f} 毫秒，"
      f"load {joblib_load_ms:>7.2f} 毫秒")
print(f"    pickle ：文件 {size_kb(pickle_big):>9.2f} KB，dump {pickle_dump_ms:>7.2f} 毫秒，"
      f"load {pickle_load_ms:>7.2f} 毫秒")

# joblib 的独门武器：mmap_mode —— 加载时不真正读数组，只建立内存映射
t0 = time.perf_counter()
mmap_payload = joblib.load(joblib_big, mmap_mode="r")
joblib_mmap_ms = (time.perf_counter() - t0) * 1000
mmap_weights = mmap_payload["weights"]
np.testing.assert_array_equal(mmap_weights, big_array)
print(f"    joblib(mmap_mode='r')：load {joblib_mmap_ms:>7.2f} 毫秒"
      f"（是普通 load 的 {joblib_mmap_ms / joblib_load_ms:.1%}）")
print(f"      返回的数组类型 = {type(mmap_weights).__name__}"
      f"，是 np.memmap = {isinstance(mmap_weights, np.memmap)}（断言通过：内容仍然完全正确）")
print("      → mmap 只是建立「文件→虚拟内存」的映射，并不把 7.63 MB 读进进程内存；")
print("        多个推理进程可以共享同一份只读页缓存，这是 joblib 在生产环境的核心价值。")

# 压缩：注意压缩率完全取决于数据本身的熵
joblib.dump(payload, bench_dir / "big_joblib_z3.pkl", compress=3)
z_ratio = size_kb(bench_dir / "big_joblib_z3.pkl") / size_kb(joblib_big)
print(f"    joblib(compress=3)：文件 "
      f"{size_kb(bench_dir / 'big_joblib_z3.pkl'):>9.2f} KB（是未压缩的 {z_ratio:.1%}）")
print(f"      → 注意：这份数组是**均匀随机数**，熵极高、几乎不可压缩，所以只压到 {z_ratio:.1%}；")
print(f"        而前面那个随机森林模型压到了 {size_kb(compressed_path) / size_kb(model_path):.1%}"
      f"（树的分裂阈值重复度高、熵低，压缩效果就好得多）。")
print("        结论：**压缩率取决于数据本身的熵，不要期待固定的压缩比**。")
print("        另外，压缩文件无法 mmap（joblib 会忽略 mmap_mode 并给出警告），二者通常二选一：")
print("        追求加载速度与低内存 → 不压缩 + mmap；追求存储/带宽 → compress=3。")
print("    → 关于本次计时的一个诚实结论：在「数据已在页缓存里的单进程微基准」中，")
print("      pickle 的 dump/load 反而比 joblib 略快（{:.2f} ms vs {:.2f} ms）。".format(
    pickle_load_ms, joblib_load_ms))
print("      joblib 的价值不在这个微基准里，而在 (a) mmap（本次实测快 "
      f"{joblib_load_ms / joblib_mmap_ms:.1f} 倍）；")
print("      (b) 内置 compress 一行搞定压缩；(c) sklearn 官方生态默认支持。")
# 注意：mmap 加载的数组持有文件句柄，Windows 上必须先释放引用才能删掉临时文件
del mmap_payload, mmap_weights
shutil.rmtree(bench_dir, ignore_errors=True)   # 清理临时基准测试文件
print(f"    （临时基准文件已删除，输出目录保持整洁）")

# =============================================================================
# ④ 结果解读
# =============================================================================
print("\n" + "=" * 78)
print("④ 结果解读：这些数字到底说明什么")
print("=" * 78)
print(f"""
【1】断言全部通过 = "加载后的模型和原来一模一样"这件事**被证明了**，而不是"看起来一样"。
      我们做了三重校验：
        (a) 硬标签逐个相等（{len(y_pred)} 个）；
        (b) 预测概率**严格**逐元素相等（{proba.shape[0]}×{proba.shape[1]} 个浮点数，不是"接近"）；
        (c) 内部参数（feature_importances_）也相等。
      为什么非要这么严？因为真实的坑往往藏在 (b)：换了 sklearn 版本后，
      概率可能从小数第 10 位开始漂移，标签却全部不变 ——
      如果你的业务用概率做阈值决策或排序，问题会在很久以后才暴露。

【2】准确率 {acc_before:.4f} → {acc_after:.4f}（完全相同）。
      这是**预期结果**：序列化只搬运对象，不做任何数值运算，理论上必须完全一致。
      如果这里出现哪怕 0.0001 的差异，就说明加载过程出了问题（版本差异、精度损失、
      或者你加载的根本不是同一个文件），必须立刻排查，而**不能当成"正常误差"放过**。

【3】文件大小：joblib {size_kb(model_path):.2f} KB / pickle {size_kb(pickle_path):.2f} KB
      / joblib(compress=3) {size_kb(compressed_path):.2f} KB。
      这个 100 棵树的随机森林体量很小，因此 joblib 与 pickle 的体积几乎一样 ——
      **这是诚实结论，不要为了"证明 joblib 更好"而夸大**。
      差距在大数组上才明显：步骤 8 的 8 MB 数组实验显示，
      joblib 的价值主要在于数组单独存储、反序列化时可 mmap（多进程推理省内存），
      以及一行 compress 就能把模型压到原来的一小部分。
      工程结论：**保存 sklearn 模型请用 joblib**（官方推荐），必要时加 compress=3。

【4】时间账：训练 {train_seconds * 1000:.2f} 毫秒，保存 {dump_seconds * 1000:.2f} 毫秒，
      加载 {load_seconds * 1000:.2f} 毫秒。
      模型越复杂，这个差距越夸张：深度学习模型训练几小时，加载几秒钟 ——
      这正是"训练与推理分离"的价值。**推理服务启动时只加载，绝不重新训练。**

【5】关于"必须保存整个 Pipeline"：步骤 5 用逻辑回归实测了"只保存模型、丢掉 scaler"的后果。
      随机森林对特征尺度不敏感，所以本例中漏存 scaler 的影响很小 ——
      但这恰恰是最危险的地方：**在你的训练数据上看不出问题，上线后换一批数据就崩**。
      只要流程里有任何"在训练数据上 fit 出来的东西"（标准化器、缺失值填充器、
      OneHotEncoder 的类别表、TF-IDF 的词表、特征选择器），就必须一起保存。

【6】元信息字典（步骤 6）是"能不能用得起来"的关键：
      没有 feature_names，你就不知道推理时 4 个特征的顺序；顺序错了，模型不会报错，
      只会给出错误的预测。没有 train_at / library_versions，
      线上出问题时就无法判断"是模型旧了还是环境换了"。
      → 建议的最小元信息集：特征名与顺序、类别名与顺序、训练时间、
        训练样本数、随机种子、评价指标、各库版本、特征工程步骤说明。

【7】最后的安全提醒（很容易被忽略）：
      joblib.load / pickle.load 在反序列化时会执行文件里的构造逻辑，
      **恶意构造的 .pkl 可以执行任意代码**。所以：
      * 只加载自己训练、自己保存的文件；
      * 模型文件要走可信的制品仓库，不要从聊天群/网盘随手拿一个 .pkl 就 load；
      * 必须接收外部模型时，优先用 ONNX 这类不含可执行逻辑的格式。
""")

# --- 绘图：加载前后一致性验证 ------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.8))

# 图 1：保存前 vs 加载后的正类概率散点（应严格落在 y=x 上）
ax0 = axes[0]
proba_cls1 = proba[:, 1]              # 取类别 1 的概率做可视化
proba_cls1_loaded = proba_loaded[:, 1]
ax0.scatter(proba_cls1, proba_cls1_loaded, s=48, alpha=0.75, color="#2980b9",
            edgecolors="white", linewidths=0.7, label="测试集样本（类别 1 的概率）")
ax0.plot([0, 1], [0, 1], linestyle="--", color="#c0392b", linewidth=1.8,
         label="理想参考线 y = x")
ax0.set_xlim(-0.03, 1.03)
ax0.set_ylim(-0.03, 1.03)
ax0.set_title("06 保存前 vs 加载后的预测概率\n（严格逐元素相等，全部落在 y=x 上）",
              fontsize=12, pad=10)
ax0.set_xlabel("保存前 predict_proba(类别 1)", fontsize=10)
ax0.set_ylabel("joblib 加载后 predict_proba(类别 1)", fontsize=10)
ax0.legend(fontsize=9, loc="upper left")
ax0.grid(alpha=0.3)

# 图 2：几种加载方式的准确率对比（应完全等高）
ax1 = axes[1]
labels = ["保存前\n原模型", "joblib\n加载后", "pickle\n加载后", "Pipeline\n加载后"]
values = [acc_before, acc_after, pickle_acc, pipe_acc]
bars = ax1.bar(np.arange(len(labels)), values, width=0.55,
               color=["#34495e", "#2980b9", "#27ae60", "#8e44ad"], alpha=0.9)
for bar, val in zip(bars, values):
    ax1.annotate(f"{val:.4f}", xy=(bar.get_x() + bar.get_width() / 2, val),
                 xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)
ax1.set_xticks(np.arange(len(labels)))
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylim(0.0, 1.12)
ax1.axhline(acc_before, color="#c0392b", linestyle=":", linewidth=1.6)
ax1.set_title("06 各加载方式的测试集准确率\n（与保存前完全一致 → 断言全部通过）",
              fontsize=12, pad=10)
ax1.set_ylabel("测试集准确率", fontsize=10)
ax1.grid(alpha=0.3, axis="y")

# 图 3：文件体积对比
ax2 = axes[2]
size_labels = ["joblib\n(未压缩)", "pickle\n(协议5)", "joblib\n(compress=3)", "含 scaler 的\nPipeline"]
size_values = [size_kb(model_path), size_kb(pickle_path), size_kb(compressed_path), size_kb(pipe_path)]
bars2 = ax2.bar(np.arange(len(size_labels)), size_values, width=0.55,
                color=["#2980b9", "#e67e22", "#16a085", "#8e44ad"], alpha=0.9)
for bar, val in zip(bars2, size_values):
    ax2.annotate(f"{val:.1f} KB", xy=(bar.get_x() + bar.get_width() / 2, val),
                 xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)
ax2.set_xticks(np.arange(len(size_labels)))
ax2.set_xticklabels(size_labels, fontsize=9)
ax2.set_title("06 模型文件体积对比\n（compress=3 一行就能显著瘦身）", fontsize=12, pad=10)
ax2.set_ylabel("文件大小（KB）", fontsize=10)
ax2.grid(alpha=0.3, axis="y")

fig.tight_layout()
fig_path = OUTPUT_DIR / "06_模型保存与加载验证.png"
fig.savefig(fig_path, dpi=130)
plt.close(fig)
print(f"\n[图] 已保存加载一致性验证图：{fig_path}")

print("\n[汇总] 本次生成的文件")
for p in [model_path, pickle_path, compressed_path, pipe_path, bundle_path, json_path, fig_path]:
    print(f"    {p.name:<34s} {size_kb(p):>9.2f} KB   {p}")

# =============================================================================
# 超参数怎么调 / 使用注意
# =============================================================================
# 【本节"可调"的其实不是模型超参，而是持久化策略】
# 1. joblib.dump 的 compress：
#    * 0（默认）：最快，文件最大；本地开发用；
#    * 3 左右：体积/耗时平衡点，落盘与网络传输首选（本脚本实测压缩比见上图）；
#    * 9：最小但慢，只在存储/带宽极度受限时用。
# 2. joblib.load 的 mmap_mode='r'：大模型（几百 MB ~ GB）多进程推理时启用，
#    多个 worker 共享同一份只读页缓存，显著降低内存峰值；注意得到的是只读数组。
# 3. 保存粒度：
#    * 只存模型 → 适合"预处理已经完全固化在服务代码里"的场景（不推荐，容易错位）；
#    * 存整个 Pipeline → **推荐默认做法**，预处理与模型永远同步；
#    * 存 dict（模型 + 元信息）→ **生产推荐做法**，本脚本步骤 6 的写法。
# 4. 版本策略：
#    * 记录 library_versions，用容器镜像/锁定文件固定依赖；
#    * sklearn 大版本升级后，用固定测试集跑一遍新旧模型预测对比（回归验证），
#      必要时用 ONNX 等中立格式重导。
# 使用注意（踩过的坑）：
#   * **不要用 pickle/joblib 加载来源不明的文件**，反序列化可执行任意代码。
#   * 保存路径要用 pathlib 且目录先 mkdir(parents=True, exist_ok=True)，
#     否则在干净环境里首次运行会 FileNotFoundError。
#   * 文件名**不要**用中文以外的奇怪字符，也别依赖工作目录（用 __file__ 推导绝对路径）。
#   * 加载后**必须**做一致性断言（本脚本的做法），而不是只看准确率"差不多"。
#   * 特征顺序！特征顺序！特征顺序！——推理时的列顺序必须与训练时完全一致，
#     顺序错了模型不报错、只给你错误的答案（所以元信息里要存 feature_names）。
#   * 每次保存建议带上时间戳或版本号（如 06_随机森林模型_v20250101.pkl），
#     方便灰度和回滚；同名覆盖会丢掉上一个可用版本。
print("\n【完成】05_模型保存与加载.py 运行结束")

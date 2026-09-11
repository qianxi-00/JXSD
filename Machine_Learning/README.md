# 机器学习（Machine Learning）课案代码实现

本目录把《机器学习》课案的**全部知识点**逐一实现为可运行、带详细中文注释与讲解的 Python 代码，
并配以中文文档。所有脚本都只用 **scikit-learn 内置数据集**或 `make_*` 生成的数据，
**不需要联网、不需要下载任何数据集**，可以在离线环境中直接跑通。

---

## 一、环境说明

| 项目 | 说明 |
| --- | --- |
| 解释器 | `F:\ProGram\Python_Base\.venv\Scripts\python.exe`（Python 3.12.12） |
| 运行方式 | **必须**使用上面的虚拟环境解释器，不要用系统 Python，也不要执行 `uv run` |
| 关键依赖 | numpy 2.5.2 / pandas 3.0.5 / scipy 1.18.0 / matplotlib 3.11.1 / seaborn 0.13.2 / scikit-learn 1.9.0 / xgboost 3.4.1 / lightgbm 4.7.0 / joblib |
| 数据集 | 全部来自 `sklearn.datasets`：`load_iris`、`load_diabetes`、`load_wine`、`load_breast_cancer`、`load_digits`，以及 `make_regression` / `make_classification` / `make_moons` / `make_blobs` |
| 随机种子 | 所有涉及随机的步骤统一 `random_state=42`，保证结果完全可复现 |
| 绘图 | 统一 `matplotlib.use("Agg")` + 保存到 `output/`，**不调用 `plt.show()`**（不会阻塞、不会弹窗） |

### 通用运行命令（PowerShell）

路径含中文，必须用引号括起来并用 `&` 调用：

```powershell
# 运行单个脚本（示例）
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Machine_Learning\02_监督学习_回归\01_线性回归.py'

# 中文输出乱码时，先设置 UTF-8（推荐每次都加）
$env:PYTHONUTF8='1'
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Machine_Learning\02_监督学习_回归\01_线性回归.py'

# 一键运行本目录全部脚本并汇总
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Machine_Learning\verify_all.py'

# 一键运行并生成 VERIFY_REPORT.md（真实运行输出）
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Machine_Learning\verify_all.py' --report
```

### 关于绘图环境（已修复，无需任何补丁）

早期本机 `.venv` 中的第三方绘图依赖 **kiwisolver 安装不完整**（`kiwisolver/__init__.py` 丢失，
只剩编译好的 `_cext.*.pyd`），会导致 `import matplotlib` 直接抛
`AttributeError: module 'kiwisolver' has no attribute '__version__'`。

该问题**已在环境中修复**（`site-packages\kiwisolver\__init__.py` 已补回，实测
`kiwisolver.__version__ == 1.5.0`、`matplotlib.__version__ == 3.11.1`）。因此：

- 所有脚本现在都是**直接 `import matplotlib` / `import matplotlib.pyplot as plt`**，不再有任何兼容补丁代码；
- 原本为此存在的 `_compat/` 目录（含 `env_patch.py`）已**整体删除**；
- 绘图统一配置如下（每个脚本开头都有）：

```python
import matplotlib
matplotlib.use("Agg")            # 无界面后端，避免 plt.show() 阻塞
import matplotlib.pyplot as plt
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
```

> 字体候选第一个 `Microsoft YaHei` 在 Windows 上必然存在，因此不会出现 `findfont` 缺字告警；
> `SimHei` / `DejaVu Sans` 只是跨平台的兜底。

---

## 二、目录结构

```text
Machine_Learning/
├── README.md                        本文件：总览、知识点索引、运行命令、环境说明
├── verify_all.py                    一键运行本目录所有脚本并汇总（可 --report 生成报告）
├── VERIFY_REPORT.md                 verify_all.py 的真实完整输出 + 实际执行命令
├── 01_介绍/
│   ├── 机器学习基础概念.md           什么是机器学习 / 四种学习范式 / 频率派 vs 贝叶斯派 / sklearn 工作流
│   └── 01_sklearn核心模块与标准工作流程.py
├── 02_监督学习_回归/
│   └── 01_线性回归.py                最小二乘推导 / 正规方程 / 梯度下降 / MSE·RMSE·MAE·R² / 拟合图
├── 03_监督学习_分类/
│   ├── 01_逻辑回归.py                sigmoid / 交叉熵 / 概率输出 / 决策边界
│   ├── 02_KNN.py                     四种距离度量 / K 值选择 / 决策边界
│   ├── 03_SVM.py                     最大间隔 / 软间隔 C / 核函数 linear·rbf / gamma
│   ├── 04_决策树.py                  基尼·熵·分类误差 / 预剪枝 / 特征重要性 / 树结构可视化
│   ├── 05_随机森林.py                Bagging / 特征随机 / OOB / 与单棵树的对比
│   └── 06_五种分类器对比.py           同一数据同一次划分下五种分类器的横向对比
├── 04_集成学习/
│   ├── 集成学习原理.md               Bagging vs Boosting / 加法模型与前向分步 / GBDT 负梯度 / XGBoost 二阶泰勒+正则 / LightGBM 直方图+Leaf-wise
│   ├── 01_GBDT.py                    sklearn GradientBoostingClassifier + 提升过程曲线
│   ├── 02_XGBoost.py                 xgboost.XGBClassifier + eval_set 早停 + 特征重要性
│   ├── 03_LightGBM.py                lightgbm.LGBMClassifier + 早停 + 训练曲线
│   └── 04_三种Boosting对比.py         同一份数据、同一次划分下准确率/耗时/AUC 对比
├── 05_无监督学习/
│   ├── 01_KMeans聚类.py              肘部法 / 轮廓系数 / 质心可视化 / 不同 K 对比
│   └── 02_PCA降维.py                 方差解释率 / 碎石图 / iris 与 digits 二维可视化
├── 06_模型评估与保存/
│   ├── 01_分类评估指标.py             accuracy·precision·recall·F1 / 混淆矩阵 / classification_report / ROC-AUC
│   ├── 02_回归评估指标.py             MSE·RMSE·MAE·R²·MAPE / 预测对比图 / 残差
│   ├── 03_交叉验证.py                 K 折 / 分层 K 折 / cross_val_score / cross_validate
│   ├── 04_网格搜索超参数调优.py         GridSearchCV / cv_results_ / 热力图
│   └── 05_模型保存与加载.py            joblib / pickle / Pipeline 持久化与预测一致性断言
└── output/                          运行时自动生成：图片（.png）与模型文件（.pkl）
```

---

## 三、知识点索引（课案章节 → 文件）

| 课案章节 | 知识点 | 对应文件 |
| --- | --- | --- |
| 介绍 · 什么是机器学习 | 定义、与传统编程的区别、T/P/E | `01_介绍/机器学习基础概念.md` |
| 介绍 · 机器学习分类 | 监督 / 无监督 / 半监督 / 强化学习对照表 | `01_介绍/机器学习基础概念.md` |
| 介绍 · 频率派 vs 贝叶斯派 | MLE（似然→对数→优化）、MAP、贝叶斯估计与预测（积分）、两者对比 | `01_介绍/机器学习基础概念.md` |
| 介绍 · sklearn 介绍 | 核心模块清单、Estimator API（fit/predict/transform/score） | `01_介绍/机器学习基础概念.md`、`01_介绍/01_sklearn核心模块与标准工作流程.py` |
| 介绍 · sklearn 标准工作流程 | 提取数据 → 清洗/划分 → 运行算法 → 得到结果 → 查看效果 | `01_介绍/01_sklearn核心模块与标准工作流程.py` |
| 监督学习 · 回归 · 线性回归 | 模型形式、最小二乘损失、正规方程闭式解、梯度下降、MSE/RMSE/MAE/R²、拟合图 | `02_监督学习_回归/01_线性回归.py` |
| 分类 · 逻辑回归 | 为什么叫回归却是分类器、sigmoid、交叉熵损失、阈值、penalty/C | `03_监督学习_分类/01_逻辑回归.py` |
| 分类 · K-近邻 | 距离度量（欧氏/曼哈顿/切比雪夫/闵可夫斯基）、K 值影响、标准化 | `03_监督学习_分类/02_KNN.py` |
| 分类 · 支持向量机 | 最大间隔、支持向量、软间隔 C、核函数与 gamma、决策边界 | `03_监督学习_分类/03_SVM.py` |
| 分类 · 决策树 | 基尼/熵/分类误差、分裂过程、max_depth 等剪枝参数、feature_importances_ | `03_监督学习_分类/04_决策树.py` |
| 分类 · 随机森林 | Bagging 自助采样、特征随机、OOB、与单棵树对比 | `03_监督学习_分类/05_随机森林.py` |
| 集成学习 · Boosting 统一加法公式 | F₀(x)、f_m(x)、学习率 η、弱模型数 M、F_M(x) | `04_集成学习/集成学习原理.md` |
| 集成学习 · GBDT | 残差/负梯度拟合、串行纠错、与随机森林的区别 | `04_集成学习/集成学习原理.md`、`04_集成学习/01_GBDT.py` |
| 集成学习 · XGBoost | 目标函数 = 损失 + 正则、二阶泰勒、分裂增益、缺失值处理、并行找分裂点 | `04_集成学习/集成学习原理.md`、`04_集成学习/02_XGBoost.py` |
| 集成学习 · LightGBM | 直方图算法、Leaf-wise、num_leaves、Gain 判据 | `04_集成学习/集成学习原理.md`、`04_集成学习/03_LightGBM.py` |
| 集成学习 · 三者对比 | 同数据同划分下准确率 / 训练耗时 / AUC | `04_集成学习/04_三种Boosting对比.py` |
| 无监督学习 · K-Means | 目标函数、交替迭代、肘部法、轮廓系数、质心可视化 | `05_无监督学习/01_KMeans聚类.py` |
| 无监督学习 · PCA | 中心化→协方差→特征分解/SVD、方差解释率、累计解释率、二维可视化 | `05_无监督学习/02_PCA降维.py` |
| 评估 · 分类评估指标 | accuracy / precision / recall / F1 / 混淆矩阵 / classification_report / AUC-ROC | `06_模型评估与保存/01_分类评估指标.py` |
| 评估 · 回归评估指标 | MSE / RMSE / MAE / R² / MAPE | `06_模型评估与保存/02_回归评估指标.py` |
| 评估 · 交叉验证 | K 折、分层 K 折、cross_val_score、cross_validate | `06_模型评估与保存/03_交叉验证.py` |
| 评估 · 超参数调优 | GridSearchCV、best_params_、best_score_、cv_results_、热力图 | `06_模型评估与保存/04_网格搜索超参数调优.py` |
| 评估 · 模型保存与加载 | joblib.dump/load、pickle 对比、Pipeline 持久化、预测一致性断言 | `06_模型评估与保存/05_模型保存与加载.py` |

---

## 四、每个脚本的统一结构

所有算法脚本都按同一套「讲解型」结构组织，方便当作讲义直接阅读：

1. **① 原理与数学推导**：模型形式 → 损失函数 → 优化方法 → 优缺点（用中文注释写清公式含义）
2. **② sklearn API 关键参数逐个解释**：每个参数的含义、调大/调小的影响、常用值
3. **③ 完整可运行代码**：分步骤实现，每一步都有中文注释
4. **④ 结果解读**：打印指标之后，用中文解释"这些数字意味着什么"
5. **超参数怎么调**：脚本末尾给出一段实践向的调参小结

每个 `.py` 文件的文件头 docstring 都包含三段：
`对应课案章节` / `本节知识点` / `运行方式`（可直接复制的 PowerShell 命令）。

---

## 五、输出文件（`output/`）

脚本运行时自动创建 `output/` 目录，并写入：

- **图片（.png）**：拟合曲线、决策边界、混淆矩阵、ROC 曲线、聚类结果、碎石图、
  提升过程、网格搜索热力图等。文件名以所在章节号开头（如 `03_`、`04_`）便于归类。
- **模型文件（.pkl）**：`06_模型评估与保存/05_模型保存与加载.py` 用 `joblib` 保存的模型与 Pipeline。

> 想看交互式图表：把脚本里的 `plt.savefig(...)` 换成 `plt.show()`，
> 并去掉 `matplotlib.use("Agg")` 一行即可在窗口中查看。

---

## 六、自检与验证

```powershell
# 逐个运行全部脚本，打印每个脚本的返回码与最后 15 行输出，最后给出汇总
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Machine_Learning\verify_all.py' --report
```

- `verify_all.py` 会用 `subprocess.run(..., capture_output=True, encoding="utf-8", timeout=180)`
  逐个运行本目录下所有课案脚本，并自动跳过工具/隐藏目录与本文件。
- 判定标准：**返回码为 0**，且输出中不含真正的 `Traceback (most recent call last)`、
  也不含行首就是 `XxxError: …` / `XxxWarning: …` 的异常回显行。
  （脚本正文里**讲解** "Warning"、"AttributeError" 这类词属于教学内容，不会被误判为失败。）
- 最后一次真实运行的完整结果见 [`VERIFY_REPORT.md`](VERIFY_REPORT.md)。
- 汇总行格式为：`共 N 个脚本，成功 N 个，失败 0 个`。
- **最近一次验证结果：共 19 个脚本，成功 19 个，失败 0 个，总耗时约 98 秒**
  （同一套脚本多次运行耗时在 87~98 秒之间波动，均远低于 300 秒上限；
  本次为移除兼容补丁、改用环境原生 matplotlib 3.11.1 之后的回归结果）
  （19 个课案脚本 = 01 章 1 个 + 02 章 1 个 + 03 章 6 个 + 04 章 4 个 + 05 章 2 个 + 06 章 5 个，
  另有 1 个工具文件 `verify_all.py` 不参与统计）。

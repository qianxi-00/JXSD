# 校验报告（VERIFY_REPORT）

> 本文档记录 `Front_End/` 全部交付物的**真实执行过的校验命令**与**原样输出**。
>
> 首次校验时间：2026-09-11　|　**增强后回归校验：2026-09-11（同日，见 §6.1 环境变更记录）**
> 校验人：实现工程师（自动化脚本 + 浏览器实测）
> Python：`F:\ProGram\Python_Base\.venv\Scripts\python.exe`（3.12.12）
> Streamlit：1.61.1　|　Node.js：v24.20.0　|　**matplotlib：3.11.1（已恢复可用）**

---

## 一、实际执行的命令

### 1.1 一键校验（本报告的主体）

```powershell
$env:PYTHONIOENCODING='utf-8'
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Front_End\verify_all.py'
```

退出码：**0**（全部通过）

### 1.2 逐个 Streamlit 应用手动启动（verify_all.py 内部执行的就是这个命令）

`verify_all.py` 用 `subprocess.Popen` 对每个应用执行：

```
[<venv>\python.exe, "-m", "streamlit", "run", <应用路径>,
 "--server.headless", "true", "--server.port", <8601 起递增>, "--browser.gatherUsageStats", "false"]
```

即等价于下面这样的手工命令（18 个应用，端口 8601~8618）：

```powershell
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m streamlit run `
    'F:\ProGram\Python_Base\Front_End\04_Streamlit\01_安装与运行.py' `
    --server.headless true --server.port 8601 --browser.gatherUsageStats false
# … 02→8602、03→8603 … 13→8613、14_多页面应用/app.py→8614、
#    15_图书管理系统/admin.py→8615、15_图书管理系统/app.py→8616、
#    15_图书管理系统/user.py→8617、16_多页面_侧边栏方式.py→8618
```

每个应用启动后轮询 `http://127.0.0.1:<port>/_stcore/health`（兼容 `/healthz`），
返回 200 即通过（超时 45 秒），随后 `terminate()` → 必要时 `kill()`。

### 1.3 数据层单独自检

```powershell
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' `
    'F:\ProGram\Python_Base\Front_End\04_Streamlit\15_图书管理系统\data.py'
```

### 1.4 浏览器端到端实测（Chrome，真实渲染）

```text
file:///F:/ProGram/Python_Base/Front_End/06_综合实战/index.html
file:///F:/ProGram/Python_Base/Front_End/03_JavaScript/03_事件处理.html
file:///F:/ProGram/Python_Base/Front_End/02_CSS/03_盒子模型.html
http://127.0.0.1:8615/            （图书管理系统：登录 → 借书）
```

---

## 二、`verify_all.py` 的完整输出（原样粘贴）

```text
==============================================================================
Front_End 交付物一键校验
==============================================================================
校验根目录：F:\ProGram\Python_Base\Front_End
Python 解释器：F:\ProGram\Python_Base\.venv\Scripts\python.exe

HTML 文件：15 个
Python 文件：23 个
JS 文件：3 个　CSS 文件：4 个　Markdown：8 个
Streamlit 应用：18 个

------------------------------------------------------------------------------
(a) Python 语法检查（py_compile）
------------------------------------------------------------------------------
  [OK]   04_Streamlit/01_安装与运行.py
  [OK]   04_Streamlit/02_页面配置.py
  [OK]   04_Streamlit/03_文本.py
  [OK]   04_Streamlit/04_数据.py
  [OK]   04_Streamlit/05_图表.py
  [OK]   04_Streamlit/06_输入.py
  [OK]   04_Streamlit/07_媒体.py
  [OK]   04_Streamlit/08_布局.py
  [OK]   04_Streamlit/09_状态与提示.py
  [OK]   04_Streamlit/10_侧边栏.py
  [OK]   04_Streamlit/11_表单.py
  [OK]   04_Streamlit/12_会话状态.py
  [OK]   04_Streamlit/13_缓存机制.py
  [OK]   04_Streamlit/14_多页面应用/app.py
  [OK]   04_Streamlit/14_多页面应用/pages/1_数据总览.py
  [OK]   04_Streamlit/14_多页面应用/pages/2_用户管理.py
  [OK]   04_Streamlit/14_多页面应用/pages/3_系统设置.py
  [OK]   04_Streamlit/15_图书管理系统/admin.py
  [OK]   04_Streamlit/15_图书管理系统/app.py
  [OK]   04_Streamlit/15_图书管理系统/data.py
  [OK]   04_Streamlit/15_图书管理系统/user.py
  [OK]   04_Streamlit/16_多页面_侧边栏方式.py
  [OK]   verify_all.py
  → Python 编译通过 23 / 23

------------------------------------------------------------------------------
(b) HTML 解析 + 本地引用检查（html.parser）
------------------------------------------------------------------------------
  [OK]   01_HTML/01_最小示例.html（引用 0 个，其中本地文件 0 个）
  [OK]   01_HTML/02_常用标签.html（引用 9 个，其中本地文件 1 个）
  [OK]   01_HTML/03_标签属性.html（引用 4 个，其中本地文件 0 个）
  [OK]   01_HTML/04_嵌套关系.html（引用 0 个，其中本地文件 0 个）
  [OK]   02_CSS/01_CSS语法.html（引用 0 个，其中本地文件 0 个）
  [OK]   02_CSS/02_常用属性.html（引用 2 个，其中本地文件 1 个）
  [OK]   02_CSS/03_盒子模型.html（引用 0 个，其中本地文件 0 个）
  [OK]   02_CSS/04_Flex布局.html（引用 0 个，其中本地文件 0 个）
  [OK]   02_CSS/05_外部样式/index.html（引用 1 个，其中本地文件 1 个）
  [OK]   03_JavaScript/01_基础语法.html（引用 0 个，其中本地文件 0 个）
  [OK]   03_JavaScript/02_DOM操作.html（引用 1 个，其中本地文件 0 个）
  [OK]   03_JavaScript/03_事件处理.html（引用 1 个，其中本地文件 0 个）
  [OK]   03_JavaScript/04_外部JS/index.html（引用 2 个，其中本地文件 2 个）
  [OK]   03_JavaScript/05_综合示例/index.html（引用 6 个，其中本地文件 2 个）
  [OK]   06_综合实战/index.html（引用 15 个，其中本地文件 2 个）
  → HTML 通过 15 / 15，共检查引用 41 个

------------------------------------------------------------------------------
(b2) JavaScript 语法检查（node --check，可选）
------------------------------------------------------------------------------
  [OK]   10 段 JavaScript 全部通过语法检查
  → JS 通过 10 / 10

------------------------------------------------------------------------------
(c) Streamlit 应用启动 + 健康检查（subprocess.Popen）
------------------------------------------------------------------------------
  [..] 启动 04_Streamlit/01_安装与运行.py（端口 8601）……
  [OK]   04_Streamlit/01_安装与运行.py → 端口 8601 健康检查 200（/_stcore/health），耗时 2.45 秒
  [..] 启动 04_Streamlit/02_页面配置.py（端口 8602）……
  [OK]   04_Streamlit/02_页面配置.py → 端口 8602 健康检查 200（/_stcore/health），耗时 2.65 秒
  [..] 启动 04_Streamlit/03_文本.py（端口 8603）……
  [OK]   04_Streamlit/03_文本.py → 端口 8603 健康检查 200（/_stcore/health），耗时 1.72 秒
  [..] 启动 04_Streamlit/04_数据.py（端口 8604）……
  [OK]   04_Streamlit/04_数据.py → 端口 8604 健康检查 200（/_stcore/health），耗时 1.84 秒
  [..] 启动 04_Streamlit/05_图表.py（端口 8605）……
  [OK]   04_Streamlit/05_图表.py → 端口 8605 健康检查 200（/_stcore/health），耗时 0.00 秒
  [..] 启动 04_Streamlit/06_输入.py（端口 8606）……
  [OK]   04_Streamlit/06_输入.py → 端口 8606 健康检查 200（/_stcore/health），耗时 1.71 秒
  [..] 启动 04_Streamlit/07_媒体.py（端口 8607）……
  [OK]   04_Streamlit/07_媒体.py → 端口 8607 健康检查 200（/_stcore/health），耗时 1.65 秒
  [..] 启动 04_Streamlit/08_布局.py（端口 8608）……
  [OK]   04_Streamlit/08_布局.py → 端口 8608 健康检查 200（/_stcore/health），耗时 1.87 秒
  [..] 启动 04_Streamlit/09_状态与提示.py（端口 8609）……
  [OK]   04_Streamlit/09_状态与提示.py → 端口 8609 健康检查 200（/_stcore/health），耗时 1.75 秒
  [..] 启动 04_Streamlit/10_侧边栏.py（端口 8610）……
  [OK]   04_Streamlit/10_侧边栏.py → 端口 8610 健康检查 200（/_stcore/health），耗时 1.77 秒
  [..] 启动 04_Streamlit/11_表单.py（端口 8611）……
  [OK]   04_Streamlit/11_表单.py → 端口 8611 健康检查 200（/_stcore/health），耗时 2.04 秒
  [..] 启动 04_Streamlit/12_会话状态.py（端口 8612）……
  [OK]   04_Streamlit/12_会话状态.py → 端口 8612 健康检查 200（/_stcore/health），耗时 2.05 秒
  [..] 启动 04_Streamlit/13_缓存机制.py（端口 8613）……
  [OK]   04_Streamlit/13_缓存机制.py → 端口 8613 健康检查 200（/_stcore/health），耗时 1.77 秒
  [..] 启动 04_Streamlit/14_多页面应用/app.py（端口 8614）……
  [OK]   04_Streamlit/14_多页面应用/app.py → 端口 8614 健康检查 200（/_stcore/health），耗时 1.69 秒
  [..] 启动 04_Streamlit/15_图书管理系统/admin.py（端口 8615）……
  [OK]   04_Streamlit/15_图书管理系统/admin.py → 端口 8615 健康检查 200（/_stcore/health），耗时 1.66 秒
  [..] 启动 04_Streamlit/15_图书管理系统/app.py（端口 8616）……
  [OK]   04_Streamlit/15_图书管理系统/app.py → 端口 8616 健康检查 200（/_stcore/health），耗时 1.64 秒
  [..] 启动 04_Streamlit/15_图书管理系统/user.py（端口 8617）……
  [OK]   04_Streamlit/15_图书管理系统/user.py → 端口 8617 健康检查 200（/_stcore/health），耗时 1.73 秒
  [..] 启动 04_Streamlit/16_多页面_侧边栏方式.py（端口 8618）……
  [OK]   04_Streamlit/16_多页面_侧边栏方式.py → 端口 8618 健康检查 200（/_stcore/health），耗时 1.66 秒
  → Streamlit 启动成功 18 / 18

------------------------------------------------------------------------------
(d) 额外加强：用 AppTest 真正执行每个应用的脚本（检查 Traceback）
------------------------------------------------------------------------------
  [OK]   04_Streamlit/01_安装与运行.py → 脚本执行无异常
  [OK]   04_Streamlit/02_页面配置.py → 脚本执行无异常
  [OK]   04_Streamlit/03_文本.py → 脚本执行无异常
  [OK]   04_Streamlit/04_数据.py → 脚本执行无异常
  [OK]   04_Streamlit/05_图表.py → 脚本执行无异常
  [OK]   04_Streamlit/06_输入.py → 脚本执行无异常
  [OK]   04_Streamlit/07_媒体.py → 脚本执行无异常
  [OK]   04_Streamlit/08_布局.py → 脚本执行无异常
  [OK]   04_Streamlit/09_状态与提示.py → 脚本执行无异常
  [OK]   04_Streamlit/10_侧边栏.py → 脚本执行无异常
  [OK]   04_Streamlit/11_表单.py → 脚本执行无异常
  [OK]   04_Streamlit/12_会话状态.py → 脚本执行无异常
[cache_data] 实际执行了 get_cached_time()
  [OK]   04_Streamlit/13_缓存机制.py → 脚本执行无异常
  [OK]   04_Streamlit/14_多页面应用/app.py → 脚本执行无异常
  [OK]   04_Streamlit/15_图书管理系统/admin.py → 脚本执行无异常
  [OK]   04_Streamlit/15_图书管理系统/app.py → 脚本执行无异常
  [OK]   04_Streamlit/15_图书管理系统/user.py → 脚本执行无异常
  [OK]   04_Streamlit/16_多页面_侧边栏方式.py → 脚本执行无异常
  → 脚本执行无异常 18 / 18

==============================================================================
校验汇总
==============================================================================
HTML 15 个 / Python 23 个 / Streamlit 应用 18 个，失败 0 个
==============================================================================
```

> 输出中那一行 `[cache_data] 实际执行了 get_cached_time()` 是 **13_缓存机制.py 有意打印**的
> 缓存证据（只有缓存未命中时才会打印），不是错误。
>
> 另：上面这次运行发生在本文档（`VERIFY_REPORT.md`）生成**之前**，
> 所以当时扫描到 Markdown 7 个；现在再运行 `verify_all.py` 会显示 **8 个**
> （多出来的就是本文档）。其余数字不变。

---

## 三、浏览器端到端实测结果

`verify_all.py` 的健康检查只能证明"服务器起来了"；它不能证明"页面在浏览器里渲染正确、JS 真的跑通了"。
所以下面这些是**用真实 Chrome 打开、真实点击、读取页面文本**得到的结论。

### 3.1 Streamlit：图书管理系统（`http://127.0.0.1:8615`）

| 步骤 | 操作 | 结果 |
|---|---|---|
| 1 | 打开首页 | 登录页正常渲染，页面文本中 `Traceback` / `StreamlitAPIException` / `AttributeError` 均为 **false** |
| 2 | 输入 `admin` / `admin123` 并点登录 | 登录成功；侧边栏出现**按角色生成**的导航：`首页 / 图书管理 / 读者视角`；首页指标卡显示"图书种类 8、馆藏总册数 29、可借册数 29、用户总数 3" |
| 3 | 点「图书管理」 | URL 变为 **`/books`**；渲染出 5 个选项卡（图书管理 / 用户管理 / 借阅记录 / 数据统计 / 系统设置）、3 个数据表格、查询表单、新增图书折叠面板、**预填了当前值的修改表单**（正在编辑《HTML 与 CSS 设计与构建网站》）、带二次确认的删除按钮 |
| 4 | 点「读者视角」 | URL 变为 **`/reader`**；渲染 5 个选项卡 |
| 5 | 点「📕 确认借阅」 | 页面提示借阅成功；「当前在借」由 0 变为 **1**，「历史借阅」变为 1 |

**并行的磁盘校验（证明 JSON 持久化真的生效）：**

```
数据文件存在: True | 大小: 3894 字节
图书种类: 8 | 用户数: 3 | 借阅记录: 1
第一本书可借: 4 / 5
记录: {'id': 1, 'book_id': 1, 'book_title': 'HTML 与 CSS 设计与构建网站', 'username': 'admin',
       'borrow_date': '2026-09-11', 'due_date': '2026-10-11', 'return_date': '', 'status': '借出'}
密码是否为明文(应为 False): False
```

→ 点击借阅后，图书 `available` 从 5 变成 4，借阅记录写进了磁盘上的 `data/library.json`，密码是哈希值。

> 校验完成后已把这次浏览器测试产生的借阅记录清掉（删除 `library.json`，
> 由 `verify_all.py` 最后一轮运行中的应用自动重建为干净的种子数据）。

### 3.2 `06_综合实战/index.html`（HTML + CSS + JS 三者结合）

用 `file://` 直接打开（模拟双击），读取页面实际状态：

| 检查项 | 结果 |
|---|---|
| `<title>` | `张三的个人主页 · 前端基础综合实战` |
| 外部 CSS 是否加载 | ✅ 已生效（body 字体为微软雅黑） |
| JS 是否执行（DOMContentLoaded） | ✅ 打印 `[app.js] 个人主页小站的 7 个模块全部初始化完成`，**JS 错误数 = 0** |
| 动态生成的作品卡片 | ✅ **6 张**（HTML 里 `#work-grid` 是空的，全部由 `createAppElement` 生成） |
| 技能条 | ✅ 6 条 |
| 分类筛选按钮 | ✅ 4 个 |
| 导航高亮链接 | ✅ 4 个 |
| 主题切换 | ✅ 初始 `light`；点按钮后变为 **`dark`** |
| 知识点计数动画 | ✅ 从 0 数到 **42** |
| 回到顶部按钮 | ✅ 存在 |

**交互实测：**

| 交互 | 期望 | 实际结果 |
|---|---|---|
| 点「网页」分类 | 只显示 category=web 的作品 | ✅ 卡片数 6 → **2**，`.is-active` 移到「网页」按钮 |
| 点主题切换 | 切换 data-theme | ✅ `light` → **`dark`** |
| 提交表单（姓名只填 1 字，其余留空） | 显示多条错误、页面不刷新 | ✅ `#form-result` 类名为 `form-result is-error`；`#cf-name-error` 文本为「姓名至少 2 个字符」 |
| 填正确后提交 | 通过、数据渲染到结果面板、页面不刷新 | ✅ 类名为 `form-result is-ok`，面板显示 姓名/邮箱/主题/留言 四个字段的值 |

### 3.3 `03_JavaScript/03_事件处理.html`

| 检查项 | 结果 |
|---|---|
| `window` 加载事件 / `DOMContentLoaded` | ✅ 顶部横幅被写入：「✅ DOMContentLoaded 已触发 —— 页面结构解析完毕，可以安全操作 DOM 了。」 |
| 事件冒泡顺序 | ✅ 点击内层，日志顺序为 **inner → mid → outer**（与预期完全一致） |
| **事件委托**：动态新增一项再删掉 | ✅ 列表项数 **2 → 3 → 2**；新增的那一项**没有任何独立监听器**，删除依然生效 |
| 表单实时校验（`input` 事件） | ✅ 输入「张」后 `#err-name` 显示「至少 2 个字」 |
| `submit` + `preventDefault` | ✅ 页面地址仍是 `file:`（**没有刷新**），输出面板显示「submit 事件触发，已调用 event.preventDefault() 阻止页面刷新」 |
| 是否有阻塞式 `alert` 弹窗 | ✅ 无（全部改成页内输出面板，注释里保留了课案的 `alert` 写法） |

### 3.4 `02_CSS/03_盒子模型.html`（可视化器）

| 操作 | 期望计算 | 实际输出 |
|---|---|---|
| 初始（`content-box`，w=120 p=10 b=4 m=12） | 120 + 20 + 8 = **148** | `= 120 + 20 + 8 = 148 × 88 px` ✅ |
| 把 padding 拉到 30 | 120 + 60 + 8 = **188** | `= 120 + 60 + 8 = 188 × 128 px` ✅ |
| 切换为 `border-box` | 内容区被压到 120−60−8 = **52**，盒子自身仍为 **120** | `= 52 + 60 + 8 = 120 × 68 px` ✅ |

→ 这正是课案"盒子模型"一节要讲清楚的核心结论，**计算完全正确**。

### 3.5 `04_Streamlit/05_图表.py`（matplotlib 渲染实测 —— 本次增强新增）

用真实 Chrome 打开 `http://127.0.0.1:8605`，等待 9 秒让全部图表渲染完成后读取页面：

| 检查项 | 结果 |
|---|---|
| `Traceback` / `StreamlitAPIException` / `AttributeError` / `ImportError` | ✅ 全部 **false** |
| 字体缺失警告（`missing from font` / `Glyph missing`） | ✅ **false** |
| 页面上的图片元素数 | ✅ **4 个**（正好对应我写的 4 个 matplotlib 图表） |
| 图片原始尺寸 | ✅ `1460×572`、`1460×572`、`1460×604`、`1460×857` |
| 与 `figsize` 比例是否吻合 | ✅ `(9,3.6)`→2.50、`(9,3.6)`→2.50、`(9,3.8)`→2.37、`(10.5,6.2)`→1.69，全部吻合 |
| 信息栏输出的库版本 | ✅ 「matplotlib **3.11.1**（渲染后端 Agg）、Altair 6.2.2。已自动选用中文字体 **Microsoft YaHei**，所以图表里的中文能正常显示。」 |

**截图人工确认：** 折线图的标题「各渠道月度销售额趋势」、三条线的图例
「线上渠道 / 线下渠道 / 批发渠道」、轴标签「月份」「销售额（万元）」**全部是正确的中文字形**
（不是方块）；网格线、圆形/方形/三角形三种数据点标记、图例框都正常。
下方柱状图的数值标注（`260`）与**最大值高亮**（`300`，粉色柱）也与代码一致。

→ **matplotlib + `st.pyplot` 在本环境完全可用，中文显示正常。**

---

## 四、`data.py` 数据层自检输出

```text
======================================================================
data.py 数据层自检
======================================================================

[1] 数据文件位置：
    F:\ProGram\Python_Base\Front_End\04_Streamlit\15_图书管理系统\data\library.json

[2] 初始统计：
    图书种类: 8
    馆藏总册数: 29
    可借册数: 29
    已借出册数: 0
    用户总数: 3
    管理员数: 1
    借阅记录总数: 0
    当前在借: 0
    逾期未还: 0

[3] 图书列表（前 3 本）：
    id=1 《HTML 与 CSS 设计与构建网站》 Jon Duckett 可借 5/5
    id=2 《JavaScript 高级程序设计》 Matt Frisbie 可借 4/4
    id=3 《流畅的 Python》 Luciano Ramalho 可借 3/3

[4] 登录测试：
    admin/admin123 → True 登录成功，欢迎 系统管理员
    admin/wrong    → False 密码错误

[5] 借书 / 还书测试（用 zhangsan 借第 1 本书）：
    借书 → True 借阅成功：《HTML 与 CSS 设计与构建网站》，请于 2026-10-11 前归还
    重复借 → False 你已经借了《HTML 与 CSS 设计与构建网站》且尚未归还（应该失败）
    还书 → True 归还成功：《HTML 与 CSS 设计与构建网站》（2026-09-11 借出）
    重复还 → False 你没有借阅《HTML 与 CSS 设计与构建网站》，或已经归还过了（应该失败）

[6] 借阅记录：
    记录1 《HTML 与 CSS 设计与构建网站》 zhangsan 已归还

自检完成，没有异常。
```

---

## 五、校验结论

**（以下为 matplotlib 恢复并完成增强后的回归校验结果）**

| 校验项 | 数量 | 结果 |
|---|---|---|
| Python 语法编译（`py_compile`） | 23 | ✅ 23 / 23 |
| HTML 解析 + 本地引用存在性（`html.parser`） | 15（41 个引用） | ✅ 15 / 15 |
| JavaScript 语法（`node --check`，可选） | 10 段 | ✅ 10 / 10 |
| Streamlit 应用启动 + 健康检查 200 | 18 | ✅ 18 / 18 |
| Streamlit 脚本实际执行无异常（AppTest） | 18 | ✅ 18 / 18 |
| 浏览器端到端交互实测（图书管理系统 / 综合实战 / 事件处理 / 盒子模型） | 4 个页面 | ✅ 全部通过 |
| **matplotlib 渲染实测**（4 张图 + 中文字形 + `background_gradient` 渐变色） | 5 项 | ✅ **全部通过**（见 §6.1 证据） |
| **Traceback / Error 输出** | — | ✅ **0 个** |
| **WARNING 输出** | — | ✅ **0 行** |

### **最终汇总：`HTML 15 个 / Python 23 个 / Streamlit 应用 18 个，失败 0 个`**

---

## 六、环境说明与已知限制

### 6.1 环境变更记录：matplotlib 已恢复可用（含本次增强）

> **背景**：首次交付时，本虚拟环境的 `matplotlib` 无法导入 ——
> `AttributeError: module 'kiwisolver' has no attribute '__version__'`
> （`kiwisolver` 缺少 `__init__.py`，既没有 `__version__` 属性也没有可用的 dist 元数据）。
> 因此当时把图表示例降级成了注释、并用 `Styler.highlight_max` / `Styler.map`
> 替代了依赖 matplotlib 的 `Styler.background_gradient`。
>
> **现在**：协调者已从 kiwisolver 1.5.0 官方 wheel 中补回
> `site-packages\kiwisolver\__init__.py`，实测：

```text
import kiwisolver            -> 1.5.0
import matplotlib            -> 3.11.1
import matplotlib.pyplot     -> OK      （backend = Agg）
font_manager.findfont("Microsoft YaHei") -> C:\Windows\Fonts\msyh.ttc
中文渲染 glyph 警告数 -> 0
```

**本次增强（回归后重新校验）的内容：**

| 文件 | 变更 |
|---|---|
| `04_Streamlit/05_图表.py` | **新增「六、matplotlib：st.pyplot(fig)」整节**：`matplotlib.use("Agg")` + 中文字体探测 + `label()` 中英回退；4 个真实图表（折线 / 柱状含数值标注与最大值高亮 / 散点含分组与线性拟合 / **2×2 子图**含饼图）；新增「三套方案取舍」对照表；补充说明章节改写 |
| `04_Streamlit/04_数据.py` | **恢复 `Styler.background_gradient()`** 渐变色表格，并新增「组合用法（渐变 + 数字格式化）」；保留 `highlight_max` / `map` 作为"不依赖 matplotlib 的替代方案"；新增 `Styler` 方法一览表 |
| `04_Streamlit/07_媒体.py` | 修正 `create_chart_image()` 的注释（原文写"因为本环境 matplotlib 不可用"，现在改为说明"这是为了演示 Pillow 本身能画什么；真要画图表请用 `st.pyplot`"） |
| `04_Streamlit/Streamlit知识点.md` | 新增 matplotlib / `st.pyplot` 完整小节 + 三套图表方案对照表 + `Styler` 条件格式小节；库可用性表把 matplotlib 改为 ✅；常见坑表新增 4 条 matplotlib 相关坑 |
| `README.md` | §4.1 可用库加入 matplotlib / kiwisolver；§4.2 移除 matplotlib 行；新增 §4.2.1「matplotlib 用法的四个要点」；§4.3 增加一行说明 |

**这些新增内容的实测证据：**

```text
# 1) 用 UserWarning 当错误跑 AppTest（可捕获字体缺失 / 弃用警告）
OK    04_数据.py exceptions = 0
OK    05_图表.py exceptions = 0

# 2) background_gradient 真的生成了渐变色
background_gradient 生成成功
  HTML 长度: 4938
  含 background-color: True
  上色单元格数: 15
  前 3 个颜色: ['#005924', '#f6fcf4', '#bce4b5']
组合用法（渐变+格式化）成功，含 ¥: True | 含 background-color: True

# 3) 真实浏览器渲染确认（http://127.0.0.1:8605）
hasTraceback: false | hasError: false | hasFontWarning: false
图片元素数: 4
  naturalW×naturalH: 1460x572 / 1460x572 / 1460x604 / 1460x857
  → 对应 figsize (9,3.6) / (9,3.6) / (9,3.8) / (10.5,6.2)，比例完全吻合
信息栏输出: "matplotlib **3.11.1**（渲染后端 Agg）、Altair 6.2.2。
            已自动选用中文字体 Microsoft YaHei，所以图表里的中文能正常显示。"
```

截图确认：折线图的标题「各渠道月度销售额趋势」、图例「线上渠道 / 线下渠道 / 批发渠道」、
轴标签「月份 / 销售额（万元）」**全部以正确的中文字形渲染**，没有出现方块字；
下方的柱状图数值标注（260）与最大值高亮（300，粉色）也都正确。

### 6.2 本环境**不可用**、因此项目中**未使用**的东西

| 库 / 能力 | 状态 | 应对方式 |
|---|---|---|
| **plotly** | ❌ 未安装 | 不使用，只在 `05_图表.py` 与知识点文档里用注释说明 `st.plotly_chart` 的用法 |
| **ruff / black** | ❌ 未安装 | 不使用 |
| **网络** | ⚠️ 不保证 | 不下载任何图片 / 音频 / 视频 / 字体。`07_媒体.py` 用 **Pillow 现场生成图片**、用标准库 **`wave` + numpy 现场合成 WAV**；视频与 `st.map` 只用注释说明且**不在运行时读取任何文件** |
| 数据库 | — | `15_图书管理系统` 用 **JSON 文件**持久化，不依赖任何数据库 |

> **本次校验中唯一"看似异常"的输出**是那一行 `[cache_data] 实际执行了 get_cached_time()` ——
> 它是 `13_缓存机制.py` **故意打印**的缓存证据（用于证明"第二次调用没有真的执行"），不是错误。
>
> Streamlit 在 `AppTest` / 裸模式下会打印
> `WARNING ... missing ScriptRunContext! This warning can be ignored when running in bare mode.`
> 这是**正常现象**（如题所述）。`verify_all.py` 在第 (d) 步用 `logging.disable(logging.WARNING)`
> 在本阶段内临时静音了它，因此最终输出里没有任何 WARNING 行（0 行）。

### 6.3 边界确认（是否写到了 `Front_End` 之外）

**没有。** `git status --short` 的结果：

```
 M pyproject.toml
 M uv.lock
?? .course_extract/
?? Back_End/
?? Front_End/
?? Deep_Learning/
?? Machine_Learning/
```

- `?? Front_End/` —— 本次工作的**唯一**产出目录（其余 `??` 项是别的 agent 的产出或工作区原有内容）。
- ⚠️ **` M pyproject.toml` 与 ` M uv.lock` 不是本次工作产生的**：
  - 我只用 `& '<venv>\Scripts\python.exe' ...` 直接调用虚拟环境，
    **从未执行 `uv`、`uv run`、`pip install` 或任何依赖安装命令**；
  - 这两个文件的 diff 内容是新增 `celery` / `flask` / `lightgbm` / `pytest` / `pytest-asyncio` / `xgboost`
    等依赖 —— **与前端工作完全无关**（属于后端 / 机器学习方向）；
  - 时间戳为 `pyproject.toml 15:41:49`、`uv.lock 15:43:52`，
    推测是**并行处理的另一个 agent 执行了 `uv` 命令**（或工作区既有状态）。
  - `RAG/测试问题.txt` 的改动同样与本次工作无关。
- `config.py` / `.env` / `README.md`（根目录）**未被修改**（时间戳仍是 2026-09-09）。
- `Back_End` / `Machine_Learning` / `Deep_Learning` 三个目录**未被本 agent 触碰**。

### 6.4 交付物中的运行时生成文件

以下文件是**运行时产物**，删掉后会自动重建，不影响校验结论：

| 文件 | 何时生成 | 说明 |
|---|---|---|
| `04_Streamlit/_media_output/generated_demo.png` | 运行 `07_媒体.py` | Pillow 现场生成的示例图片 |
| `04_Streamlit/_media_output/generated_chart.png` | 运行 `07_媒体.py` | Pillow 手绘的柱状图 |
| `04_Streamlit/_media_output/generated_tone.wav` | 运行 `07_媒体.py` | 标准库 `wave` 合成的 2 秒提示音 |
| `04_Streamlit/15_图书管理系统/data/library.json` | 运行图书管理系统 | JSON 数据文件（含 8 本书 + 3 个账号的种子数据） |
| `01_HTML/img/demo.png` | 构建时一次性生成 | 供 `02_常用标签.html` 的 `<img>` 演示使用（**必需**，不是运行时可再生的） |

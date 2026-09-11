# 前端基础 · 课案完整实现

> 本目录把课案《前端基础》的**全部知识点**实现成了**可直接打开 / 运行、带详细中文注释与讲解**的代码 + 文档。
>
> 课案主线：**网页是什么 → HTML → CSS → JavaScript → 综合示例 → 前端框架生态 → Streamlit → 重点回顾**

---

## 一、目录树

```
Front_End/
├── README.md                    ← 本文件：总览 + 知识点索引 + 打开/运行方式
├── verify_all.py                ← 一键校验本目录所有交付物
├── VERIFY_REPORT.md             ← verify_all.py 的完整输出 + 实际执行的命令
│
├── 01_HTML/                     【课案：网页是什么 / HTML / 理解网页结构】
│   ├── 01_最小示例.html
│   ├── 02_常用标签.html
│   ├── 03_标签属性.html
│   ├── 04_嵌套关系.html
│   ├── img/demo.png             （用 Pillow 生成，供 <img> 演示使用，可离线）
│   └── HTML知识点.md
│
├── 02_CSS/                      【课案：CSS】
│   ├── 01_CSS语法.html
│   ├── 02_常用属性.html
│   ├── 03_盒子模型.html          （含可拖动滑块的盒子模型可视化器）
│   ├── 04_Flex布局.html          （含可点击切换属性的 Flex 交互演示）
│   ├── 05_外部样式/
│   │   ├── index.html
│   │   └── css/style.css
│   └── CSS知识点.md
│
├── 03_JavaScript/               【课案：JavaScript / 综合示例】
│   ├── 01_基础语法.html
│   ├── 02_DOM操作.html
│   ├── 03_事件处理.html
│   ├── 04_外部JS/
│   │   ├── index.html
│   │   ├── css/style.css
│   │   └── js/app.js
│   ├── 05_综合示例/
│   │   ├── index.html
│   │   ├── css/style.css
│   │   └── js/app.js
│   └── JavaScript知识点.md
│
├── 04_Streamlit/                【课案：Streamlit】
│   ├── 01_安装与运行.py
│   ├── 02_页面配置.py
│   ├── 03_文本.py
│   ├── 04_数据.py
│   ├── 05_图表.py
│   ├── 06_输入.py
│   ├── 07_媒体.py
│   ├── 08_布局.py
│   ├── 09_状态与提示.py
│   ├── 10_侧边栏.py
│   ├── 11_表单.py
│   ├── 12_会话状态.py
│   ├── 13_缓存机制.py
│   ├── 14_多页面应用/            （方式 1：pages/ 目录 + st.navigation）
│   │   ├── app.py
│   │   └── pages/
│   │       ├── 1_数据总览.py
│   │       ├── 2_用户管理.py
│   │       └── 3_系统设置.py
│   ├── 15_图书管理系统/          （综合实战：完整可用的系统）
│   │   ├── app.py               （入口：登录 + 角色分发 + 导航）
│   │   ├── admin.py             （管理员：图书增删改查 / 用户 / 记录 / 统计）
│   │   ├── user.py              （普通用户：浏览 / 借还 / 记录 / 个人中心）
│   │   ├── data.py              （数据层：JSON 文件持久化 + 全部业务规则）
│   │   ├── data/library.json    （数据文件，首次运行自动创建）
│   │   └── README.md
│   ├── 16_多页面_侧边栏方式.py    （方式 2：单文件 + session_state 切页）
│   ├── Streamlit知识点.md
│   └── _media_output/           （07_媒体.py 运行时生成的图片/音频，删掉会自动重建）
│
├── 05_前端框架生态/              【课案：前端框架生态】
│   └── README.md                （React / Vue / Node.js / 构建工具 / 打包流程）
│
└── 06_综合实战/                  【课案：重点回顾 —— 三者结合】
    ├── index.html               （个人主页小站：导航 + 卡片 + 表单校验 + 动态列表）
    ├── css/style.css
    └── js/app.js
```

---

## 二、知识点索引（课案章节 → 文件）

### 课案：网页是什么 / HTML

| 课案小节 | 实现文件 | 说明 |
|---|---|---|
| 网页是什么 | `01_HTML/01_最小示例.html` | 网页 = 一份 HTML 文档；最小骨架逐行拆解 |
| HTML 最小示例 | `01_HTML/01_最小示例.html` | `!DOCTYPE` / `html` / `head` / `meta charset` / `title` / `body` |
| 常用标签 | `01_HTML/02_常用标签.html` | 课案表格 8 类标签全部演示 + `br`/`ol`/语义化标签/实体字符 |
| 标签属性 | `01_HTML/03_标签属性.html` | `href` / `class` / `id` + 常见属性速查 + `data-*` + 布尔属性坑 |
| 理解网页结构 → 嵌套关系 | `01_HTML/04_嵌套关系.html` | 父子/兄弟/祖先/后代术语；**实时绘制本页真实 DOM 树** |
| 理解网页结构 → 开发者工具 | `01_HTML/04_嵌套关系.html` | 右键查看源代码 vs 右键检查的区别；Elements/Styles/Console |
| 知识点总结 | `01_HTML/HTML知识点.md` | 完整中文总结 + 速查表 + 易错点 |

### 课案：CSS

| 课案小节 | 实现文件 | 说明 |
|---|---|---|
| CSS 语法 | `02_CSS/01_CSS语法.html` | 选择器/{属性:值}/声明；三种引入方式；常用选择器；优先级与层叠 |
| 常用属性 → 文字样式 | `02_CSS/02_常用属性.html` | `color`/`font-size`/`font-family`/`text-align` 等全部实测 |
| 常用属性 → 背景与边框 | `02_CSS/02_常用属性.html` | `background-*` / `border` / `border-radius` / `box-shadow` |
| 常用属性 → 尺寸 | `02_CSS/02_常用属性.html` | `px`/`%`/`vw`/`max-width`；尺寸与 padding/border 的关系 |
| 盒子模型 | `02_CSS/03_盒子模型.html` | **可交互可视化器**：拖滑块实时看 content/padding/border/margin 与总尺寸计算 |
| Flexbox 布局 | `02_CSS/04_Flex布局.html` | 两根轴；容器/项目属性；**可点击切换** `justify-content`/`align-items`/`wrap`/`direction` |
| 在 HTML 中使用 CSS → 方式 1 | `02_CSS/01_CSS语法.html` | `<style>` 标签 |
| 在 HTML 中使用 CSS → 方式 2（推荐） | `02_CSS/05_外部样式/index.html` + `css/style.css` | 真实项目级样式表：CSS 变量 / 重置 / 组件 / 响应式 / 打印样式 |
| 知识点总结 | `02_CSS/CSS知识点.md` | 含盒子模型尺寸计算公式、Flex 速查表、坑清单 |

### 课案：JavaScript

| 课案小节 | 实现文件 | 说明 |
|---|---|---|
| 基础语法 → 变量 | `03_JavaScript/01_基础语法.html` | `let`/`const`/`var` 对比与坑 |
| 基础语法 → 数据类型 | `03_JavaScript/01_基础语法.html` | 7 种类型 + `typeof` 怪癖 + 模板字符串 |
| 基础语法 → 条件判断 | `03_JavaScript/01_基础语法.html` | `if/else if/else`、三元、`switch` 贯穿陷阱 |
| 基础语法 → 循环 | `03_JavaScript/01_基础语法.html` | `for`/`for...of`/`for...in`/`while`/`break`/`continue` + 数组方法 |
| 基础语法 → 函数 | `03_JavaScript/01_基础语法.html` | 四种定义方式、默认参数、剩余参数、作用域、闭包 |
| DOM 操作 | `03_JavaScript/02_DOM操作.html` | 课案三按钮示例 + 获取元素/改内容/改样式/改属性/增删元素 |
| 常用 DOM 操作（课案表格） | `03_JavaScript/02_DOM操作.html` | `getElementById`/`querySelector(All)`/`textContent`/`innerHTML`/`style` |
| 事件处理 | `03_JavaScript/03_事件处理.html` | `onclick` vs `addEventListener`、事件对象、冒泡/捕获可视化、**事件委托** |
| 在 HTML 中使用 JS → 方式 1 | `03_JavaScript/01_基础语法.html` | 内联 `<script>` + 点击计数器 |
| 在 HTML 中使用 JS → 方式 2（推荐） | `03_JavaScript/04_外部JS/index.html` + `js/app.js` | 含 `defer`/`async` 对比、四种常见错误排查 |
| 综合示例：三者结合 | `03_JavaScript/05_综合示例/index.html` + `css/` + `js/` | 计数器 + 主题切换 + 待办清单 + 表单校验 + localStorage |
| 知识点总结 | `03_JavaScript/JavaScript知识点.md` | 完整总结 + 经典坑清单 |

### 课案：前端框架生态

| 课案小节 | 实现文件 | 说明 |
|---|---|---|
| React / Vue / Node.js / 构建工具 | `05_前端框架生态/README.md` | 还原课案流程图；框架的"数据驱动视图"；React vs Vue；Node.js 为什么是前端工具链底座；Vite/webpack 打包八步流程；术语表 |

### 课案：Streamlit

| 课案小节 | 实现文件 | 端口 |
|---|---|---|
| 安装与运行 | `04_Streamlit/01_安装与运行.py` | 8601 |
| 页面配置 | `04_Streamlit/02_页面配置.py` | 8602 |
| 文本 | `04_Streamlit/03_文本.py` | 8603 |
| 数据 | `04_Streamlit/04_数据.py`（含 `Styler` 渐变色） | 8604 |
| 图表 | `04_Streamlit/05_图表.py`（内置图表 + Altair + **matplotlib**） | 8605 |
| 输入 | `04_Streamlit/06_输入.py` | 8606 |
| 媒体 | `04_Streamlit/07_媒体.py` | 8607 |
| 布局 | `04_Streamlit/08_布局.py` | 8608 |
| 状态与提示 | `04_Streamlit/09_状态与提示.py` | 8609 |
| 侧边栏 | `04_Streamlit/10_侧边栏.py` | 8610 |
| 表单 | `04_Streamlit/11_表单.py` | 8611 |
| 会话状态 | `04_Streamlit/12_会话状态.py` | 8612 |
| 缓存机制 | `04_Streamlit/13_缓存机制.py` | 8613 |
| 多页面应用 → 方式 1：子页面 | `04_Streamlit/14_多页面应用/app.py` + `pages/` | 8614 |
| 多页面应用 → 方式 2：侧边栏 | `04_Streamlit/16_多页面_侧边栏方式.py` | 8616 |
| 综合实战：图书管理系统 | `04_Streamlit/15_图书管理系统/app.py` | 8615 |
| 知识点总结 | `04_Streamlit/Streamlit知识点.md` | 完整总结 + 常见坑速查表 |

### 课案：重点回顾

| 课案要点 | 实现文件 |
|---|---|
| HTML 定义网页结构 | `06_综合实战/index.html` |
| CSS 控制网页样式（盒子模型 / Flexbox） | `06_综合实战/css/style.css` |
| JavaScript 实现网页交互（DOM / 事件） | `06_综合实战/js/app.js` |
| Streamlit 用 Python 快速生成网页 | `04_Streamlit/15_图书管理系统/` |

---

## 三、怎么打开 / 怎么运行

### 3.1 所有 HTML 文件：**直接用浏览器打开**

| 方式 | 操作 |
|---|---|
| 双击 | 在文件管理器里双击 `.html` 文件 |
| 浏览器打开 | `Ctrl + O`，选择文件 |
| 拖拽 | 把文件拖进浏览器窗口 |

> **无需服务器、无需联网。** 所有页面都是**自包含**的：内联样式/脚本，
> 或者配套的相对路径文件（如 `css/style.css`、`js/app.js`、`img/demo.png`）。
>
> ⚠️ **保持目录结构不变**：如果一个 HTML 引用了同目录下的 `css/` 或 `js/`，
> 把 HTML 单独拷走会导致样式/脚本加载失败。

**建议顺序：**

```
01_HTML/01_最小示例.html
  → 01_HTML/02_常用标签.html
  → 01_HTML/03_标签属性.html
  → 01_HTML/04_嵌套关系.html        （按 F12 对照 Elements 面板看）
02_CSS/01_CSS语法.html
  → 02_CSS/02_常用属性.html
  → 02_CSS/03_盒子模型.html          （拖动滑块玩一下）
  → 02_CSS/04_Flex布局.html          （点按钮切换属性）
  → 02_CSS/05_外部样式/index.html
03_JavaScript/01_基础语法.html
  → 03_JavaScript/02_DOM操作.html
  → 03_JavaScript/03_事件处理.html
  → 03_JavaScript/04_外部JS/index.html
  → 03_JavaScript/05_综合示例/index.html
06_综合实战/index.html               （三者结合的完整小站）
```

### 3.2 Streamlit 应用：必须用**工作区自带的虚拟环境**

> ⚠️ **环境约定**：只能使用 `F:\ProGram\Python_Base\.venv`（Python 3.12.12）。
> **不要**用 `uv run`（可能触发依赖同步破坏环境），**不要** pip / uv install 任何东西。

**统一命令模板（PowerShell）：**

```powershell
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m streamlit run `
    '<应用文件的完整路径>' `
    --server.headless true --server.port 86xx --browser.gatherUsageStats false
```

**逐个应用的现成命令**（端口从 8601 递增）：

```powershell
# 01 安装与运行
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m streamlit run 'F:\ProGram\Python_Base\Front_End\04_Streamlit\01_安装与运行.py' --server.headless true --server.port 8601 --browser.gatherUsageStats false

# 02 页面配置
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m streamlit run 'F:\ProGram\Python_Base\Front_End\04_Streamlit\02_页面配置.py' --server.headless true --server.port 8602 --browser.gatherUsageStats false

# 03 文本
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m streamlit run 'F:\ProGram\Python_Base\Front_End\04_Streamlit\03_文本.py' --server.headless true --server.port 8603 --browser.gatherUsageStats false

# 04 数据
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m streamlit run 'F:\ProGram\Python_Base\Front_End\04_Streamlit\04_数据.py' --server.headless true --server.port 8604 --browser.gatherUsageStats false

# 05 图表
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m streamlit run 'F:\ProGram\Python_Base\Front_End\04_Streamlit\05_图表.py' --server.headless true --server.port 8605 --browser.gatherUsageStats false

# 06 输入
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m streamlit run 'F:\ProGram\Python_Base\Front_End\04_Streamlit\06_输入.py' --server.headless true --server.port 8606 --browser.gatherUsageStats false

# 07 媒体
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m streamlit run 'F:\ProGram\Python_Base\Front_End\04_Streamlit\07_媒体.py' --server.headless true --server.port 8607 --browser.gatherUsageStats false

# 08 布局
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m streamlit run 'F:\ProGram\Python_Base\Front_End\04_Streamlit\08_布局.py' --server.headless true --server.port 8608 --browser.gatherUsageStats false

# 09 状态与提示
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m streamlit run 'F:\ProGram\Python_Base\Front_End\04_Streamlit\09_状态与提示.py' --server.headless true --server.port 8609 --browser.gatherUsageStats false

# 10 侧边栏
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m streamlit run 'F:\ProGram\Python_Base\Front_End\04_Streamlit\10_侧边栏.py' --server.headless true --server.port 8610 --browser.gatherUsageStats false

# 11 表单
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m streamlit run 'F:\ProGram\Python_Base\Front_End\04_Streamlit\11_表单.py' --server.headless true --server.port 8611 --browser.gatherUsageStats false

# 12 会话状态
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m streamlit run 'F:\ProGram\Python_Base\Front_End\04_Streamlit\12_会话状态.py' --server.headless true --server.port 8612 --browser.gatherUsageStats false

# 13 缓存机制
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m streamlit run 'F:\ProGram\Python_Base\Front_End\04_Streamlit\13_缓存机制.py' --server.headless true --server.port 8613 --browser.gatherUsageStats false

# 14 多页面应用（★ 入口是 app.py）
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m streamlit run 'F:\ProGram\Python_Base\Front_End\04_Streamlit\14_多页面应用\app.py' --server.headless true --server.port 8614 --browser.gatherUsageStats false

# 15 图书管理系统（★ 入口是 app.py）
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m streamlit run 'F:\ProGram\Python_Base\Front_End\04_Streamlit\15_图书管理系统\app.py' --server.headless true --server.port 8615 --browser.gatherUsageStats false

# 16 多页面（侧边栏方式）
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m streamlit run 'F:\ProGram\Python_Base\Front_End\04_Streamlit\16_多页面_侧边栏方式.py' --server.headless true --server.port 8616 --browser.gatherUsageStats false
```

浏览器访问 `http://localhost:<端口>`。停止服务：在终端按 `Ctrl + C`。

**图书管理系统的演示账号：**

| 角色 | 用户名 | 密码 |
|---|---|---|
| 管理员 | `admin` | `admin123` |
| 普通用户 | `user` | `user123` |
| 普通用户 | `zhangsan` | `123456` |

### 3.3 一键校验全部交付物

```powershell
& 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' 'F:\ProGram\Python_Base\Front_End\verify_all.py'
```

它会做四件事：Python 语法编译、HTML 解析与本地引用检查、
**把每个 Streamlit 应用真正启动起来做健康检查**、用 AppTest 真正执行脚本确认无异常。
完整输出见 `VERIFY_REPORT.md`。

---

## 四、环境与限制说明（重要）

### 4.1 已确认可用的库

| 库 | 版本 | 用途 |
|---|---|---|
| streamlit | 1.61.1 | 全部 Streamlit 示例 |
| pandas | 3.0.5 | 数据表格、统计、`Styler` 条件格式 |
| numpy | 2.5.2 | 数值计算、生成音频、拟合趋势线 |
| **matplotlib** | **3.11.1** | **`st.pyplot(fig)` 图表、`Styler.background_gradient` 渐变色** |
| kiwisolver | 1.5.0 | matplotlib 的依赖（曾因缺少 `__init__.py` 导致 matplotlib 无法导入，现已修复） |
| altair | 6.2.2 | 图表（Streamlit 内置图表的底层） |
| Pillow | 12.3.0 | 现场生成示例图片 |

### 4.2 明确**不可用**、因此本项目**没有使用**的东西

| 库 / 能力 | 状态 | 本项目的应对 |
|---|---|---|
| **plotly** | ❌ 未安装 | 不使用，只用注释说明用法（`st.plotly_chart`） |
| **ruff / black** | ❌ 未安装 | 不使用 |
| **网络** | ⚠️ 不保证 | 不下载任何图片/音频/视频/字体；`07_媒体.py` 用 Pillow 现场生成图片、用标准库 `wave` + numpy 现场合成 WAV；视频只用注释说明；`st.map` 只用注释说明 |
| 数据库 | — | `15_图书管理系统` 用 **JSON 文件**持久化，不依赖任何数据库 |

### 4.2.1 matplotlib 用法的四个要点（本项目 `05_图表.py` 已全部做好）

1. **`matplotlib.use("Agg")` 必须在 `import matplotlib.pyplot` 之前调用** ——
   服务器没有显示器，必须用"只出图、不弹窗"的 Agg 后端。
2. **不要 `plt.show()`** —— 交给 `st.pyplot(fig)` 渲染。
3. **要手动设置中文字体**，否则中文会变成小方块：
   ```python
   plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
   plt.rcParams["axes.unicode_minus"] = False
   ```
   本项目还做了**字体探测**：先用 `font_manager.findfont(..., fallback_to_default=False)`
   确认本机真的装了这个字体，没有就把图表标题自动换成英文（避免出现"豆腐块"）。
4. **画完就 `plt.close(fig)`** —— 否则每次重跑都会累积 figure，最终内存报警。

### 4.3 与课案原文的**有意差异**（都是为了让代码能真的跑起来 / 更健壮）

| 课案原文 | 本项目做法 | 原因 |
|---|---|---|
| `st.dataframe(df, use_container_width=True)` | `st.dataframe(df, width="stretch")` | `use_container_width` 在 Streamlit 1.61 已**弃用**，会打印弃用警告 |
| HTML/JS 示例用 `alert()` 输出 | 改成页面内的**输出面板**（注释里保留 `alert` 原始写法） | `alert` 会阻塞浏览，且自动化验证时弹窗会卡住页面 |
| `st.empty()` 示例里无条件 `time.sleep(3)` | 改成**点按钮才演示** | 否则页面每次加载都要等 3 秒；演示性等待不应该拖慢正常使用 |
| 气球/雪花/进度条无条件执行 | 改成**点按钮触发** | 否则每点一次任何按钮都会重新播放动画 |
| `10_侧边栏.py` 课案示例中 `st.dataframe({...})` | 包一层 `pd.DataFrame(...)` | 直接传字典在部分版本上会告警；显式转换更稳妥 |
| 课案只提到「Streamlit 也支持 matplotlib」 | `05_图表.py` 增加了**完整可运行的 matplotlib 示例**（折线 / 柱状 / 散点 / 子图，含中文标题与字体设置） | 课案没有给 matplotlib 的完整代码；本项目把"四个必须知道的点"都写进了代码与注释 |
| `15_图书管理系统/data.py` 用 MySQL + SQLModel | 用 **JSON 文件**持久化 | 题目要求"不依赖数据库"；JSON 在任何机器上都能直接跑 |
| `data.User.find_user(name).borrow_book(title)` | `data.borrow_book(book_id, username) -> (bool, str)` | 课案写法在书不存在时会抛 `AttributeError`；本项目要求**数据层不抛异常**，因此界面永远不会出现 Traceback |
| 图书管理让用户手输书名 | 改成**下拉选择** | 手输书名极容易打错一个字就找不到书 |
| 无登录 / 无权限 | 完整的**登录 + 角色权限** | 课案 `app.py` 只是把两个页面并排；真实系统必须有权限控制 |
| 无"改"和"删" | 完整的**增删改查** | 题目要求功能包含增删改查 |

---

## 五、文件统计

| 类别 | 数量 |
|---|---|
| HTML 文件 | 15 |
| CSS 文件 | 4 |
| JavaScript 文件 | 3 |
| Python 文件 | 23（含 `verify_all.py`） |
| Streamlit 应用 | 18 |
| Markdown 文档 | 8 |
| 资源文件 | `01_HTML/img/demo.png`、`04_Streamlit/_media_output/*`（运行时生成） |

> 精确数字以 `verify_all.py` 的汇总输出为准，见 `VERIFY_REPORT.md`。

---

## 六、写作约定

- **全中文**：注释、docstring、README、页面上的文字都是简体中文。
- **每个 HTML 文件顶部**都有中文注释块说明：
  对应课案章节 / 本节知识点 / 如何打开查看。
  HTML 用 `<!-- -->`、CSS 用 `/* */`、JS 用 `//`。
- **每个 Streamlit `.py` 开头**都有 docstring 说明：
  对应课案章节 / 本节知识点 / **运行方式（完整的 `streamlit run` 命令）**。
- 代码里是**讲解型注释**：讲清"这个标签/属性/样式/API 是干什么的、为什么这样写"，
  而不只是"这行做了什么"。
- 所有脚本里的路径一律用 `pathlib.Path(__file__).resolve().parent` 计算，
  **不依赖当前工作目录**。
- **不允许出现任何 Traceback / Error 输出** —— 这一点由 `verify_all.py` 的第 (d) 步
  （AppTest 真正执行脚本）自动保证。

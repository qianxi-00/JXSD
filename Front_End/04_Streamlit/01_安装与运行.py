"""
=====================================================================================
文件：01_安装与运行.py
对应课案章节：Streamlit → 安装与运行（以及「Streamlit」这一节的总体介绍）
本节知识点：
  1. Streamlit 是什么：用纯 Python 写网页，不需要 HTML/CSS/JS。
  2. 安装命令：pip install streamlit。
  3. 运行方式：streamlit run app.py（注意不是 python app.py！）。
  4. 常用启动参数：--server.port / --server.headless / --server.runOnSave /
     --logger.level / --client.showErrorDetails / --browser.gatherUsageStats。
  5. Streamlit 的运行模型：★ 每次交互都会从头重新执行整个脚本 ★
     （这是理解后面所有内容的前提，比任何 API 都重要）。

运行方式（在本项目里，必须用工作区自带的虚拟环境）：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m streamlit run `
        'F:\\ProGram\\Python_Base\\Front_End\\04_Streamlit\\01_安装与运行.py' `
        --server.headless true --server.port 8601 --browser.gatherUsageStats false

    或者先激活虚拟环境，然后再用简写：
    streamlit run 01_安装与运行.py

浏览器会自动打开 http://localhost:8501（本项目示例指定端口后就变成 8601）。
停止服务：在终端按 Ctrl + C。
=====================================================================================
"""

# streamlit 是唯一的必需导入。约定俗成缩写为 st。
import streamlit as st

# set_page_config 必须是【第一个】Streamlit 调用，而且整个脚本只能调用一次。
# 它设置的是浏览器标签页标题、图标、整体布局等"页面级"配置。
st.set_page_config(
    page_title="01 安装与运行",   # 浏览器标签页上的标题
    page_icon="🚀",               # 浏览器标签页上的图标（emoji 或图片路径）
    layout="centered",            # centered=内容居中（默认）；wide=铺满宽屏
)

st.title("01 安装与运行")
st.caption("对应课案：Streamlit → 安装与运行")

# ---------------------------------------------------------------------------
# 一、Streamlit 是什么
# ---------------------------------------------------------------------------
st.header("一、Streamlit 是什么")

st.markdown(
    """
前面学的 HTML / CSS / JS 是**传统前端**，要做好一个页面需要写三种语言、三个文件。

**Streamlit** 是一个 Python 库，可以**用 Python 代码直接生成网页**，
完全不需要写 HTML/CSS/JS。你只需要写 Python，它负责把每个 `st.xxx()` 调用
翻译成网页上的一个组件。
"""
)

# st.columns 把页面横向切成若干列，参数是每列的宽度比例。
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("传统前端")
    st.markdown(
        """
- 需要写 **HTML**（结构）
- 需要写 **CSS**（样式）
- 需要写 **JavaScript**（交互）
- 前后端要约定接口
        """
    )

with col_right:
    st.subheader("Streamlit")
    st.markdown(
        """
- **只写 Python**
- 交互由框架自动处理
- 数据、图表、表格有现成组件
- 适合数据应用与内部工具
        """
    )

with st.container(border=True):
    st.markdown(
        """
**课案原文的总结：**

> Streamlit 让你用纯 Python 构建网页应用，无需学习 HTML/CSS/JS。
> 从数据展示到完整后台管理系统，都可以快速实现。
> 如果要做更精美的商业网站或复杂交互还是需要传统前端，
> 但对于数据应用和内部工具，Streamlit 是最快的方式。
        """
    )

# ---------------------------------------------------------------------------
# 二、安装
# ---------------------------------------------------------------------------
st.header("二、安装")

st.markdown("官方安装命令是：")
st.code("pip install streamlit", language="bash")

st.warning(
    "**本项目的环境约定：**工作区已经装好了 Streamlit（1.61.1），"
    "**不要**再执行 pip / uv install，也不要使用 `uv run`（可能触发依赖同步而破坏环境）。"
    "请统一使用工作区自带的虚拟环境：`F:\\ProGram\\Python_Base\\.venv`。"
)

st.markdown("验证是否安装成功：")
st.code(
    "& 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -c \"import streamlit; print(streamlit.__version__)\"",
    language="powershell",
)

st.write("当前运行本页面的 Streamlit 版本是：", st.__version__)

# ---------------------------------------------------------------------------
# 三、运行
# ---------------------------------------------------------------------------
st.header("三、运行")

st.markdown("**关键点：不是 `python app.py`，而是 `streamlit run app.py`。**")
st.code(
    """# 最简写法（前提：已经激活虚拟环境，且当前目录下确实有 app.py）
streamlit run app.py

# 本项目推荐的完整写法（不激活环境，直接用绝对路径的 python）
& 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m streamlit run `
    'F:\\ProGram\\Python_Base\\Front_End\\04_Streamlit\\01_安装与运行.py' `
    --server.headless true --server.port 8601 --browser.gatherUsageStats false""",
    language="powershell",
)

st.markdown("**为什么不能写 `python app.py`？**")
st.markdown(
    """
因为 Streamlit 应用需要一个**常驻的 Web 服务**来托管你的脚本，
`python app.py` 只是把脚本跑一遍就退出了。
`streamlit run` 会启动一个 Tornado 服务器，并在浏览器每次交互时重新执行你的脚本。
"""
)

st.subheader("常用启动参数")
st.markdown(
    """
| 参数 | 作用 |
|---|---|
| `--server.port 8601` | 指定端口（默认 8501）。同时运行多个应用时必须区分 |
| `--server.headless true` | 无头模式：不尝试自动打开浏览器（服务器 / 脚本里常用） |
| `--server.runOnSave true` | 保存文件后自动重载（开发时很方便） |
| `--logger.level debug` | 提高日志级别，排查问题用 |
| `--client.showErrorDetails true` | 浏览器里显示完整的错误堆栈（开发时用） |
| `--browser.gatherUsageStats false` | 关闭使用数据上报（本项目统一加这个参数） |
| `--server.address 0.0.0.0` | 允许局域网内其它机器访问 |
"""
)

st.markdown("也可以把配置写进配置文件 `~/.streamlit/config.toml`：")
st.code(
    """[server]
port = 8601
headless = true

[browser]
gatherUsageStats = false""",
    language="toml",
)

# ---------------------------------------------------------------------------
# 四、运行模型（最重要的一节）
# ---------------------------------------------------------------------------
st.header("四、运行模型：每次交互都重新执行整个脚本")

st.error(
    "**这是理解 Streamlit 的第一前提：**\n\n"
    "用户每一次操作（点按钮、拖滑块、输入文字……）都会让 Streamlit "
    "**从上到下重新执行一遍整个脚本**，然后把结果重新渲染到页面上。"
)

st.markdown(
    """
这带来的三个直接结论：

1. **脚本就是页面的"渲染函数"** —— 你写什么顺序，页面上就是什么顺序。
2. **脚本局部变量每次都会被重置** —— 因为它们是新一轮执行里重新创建的。
   要想跨交互保留数据，必须用 `st.session_state`（见 `12_会话状态.py`）。
3. **重跑是廉价的** —— 但耗时的计算（读数据库、训练模型）必须用
   `@st.cache_data` / `@st.cache_resource` 缓存起来（见 `13_缓存机制.py`）。
"""
)

# ↑ 上一段里出现了反引号，这里用普通文本再强调一次
st.info(
    "记住这三点，后面 12（会话状态）和 13（缓存机制）就都是「顺理成章」，"
    "而不是需要死记硬背的新知识了。"
)

# ---------------------------------------------------------------------------
# 五、课案原文的第一个例子（完整复现）
# ---------------------------------------------------------------------------
st.header("五、课案原文的第一个例子")

st.markdown("课案里的 `app.py`：")
st.code(
    '''import streamlit as st

st.title("我的第一个网页")
st.write("Hello World!")
name = st.text_input("请输入你的名字")
if name:
    st.write("你好，" + name)''',
    language="python",
)

st.markdown("**下面就是它的实时复现，试着在输入框里打字：**")
st.divider()

st.title("我的第一个网页")
st.write("Hello World!")

# st.text_input 返回一个字符串：用户输入的内容。
# 还没输入时返回空字符串 ""（如果传了 value 参数，则返回该默认值）。
name = st.text_input("请输入你的名字", placeholder="比如：张三")

# if name: 利用了 Python 里"空字符串是假值"的特性
if name:
    st.write("你好，" + name)
else:
    st.caption("（左边输入框里输入内容后，这里会立刻出现问候语）")

st.divider()

# ---------------------------------------------------------------------------
# 六、本目录的学习路线
# ---------------------------------------------------------------------------
st.header("六、本目录的学习路线")

st.markdown(
    """
| 文件 | 覆盖的课案小节 |
|---|---|
| `01_安装与运行.py` | 安装与运行（本文件） |
| `02_页面配置.py` | `st.set_page_config` |
| `03_文本.py` | 文本类组件 |
| `04_数据.py` | 数据类组件（dataframe / table / metric / json） |
| `05_图表.py` | 图表类组件（line / area / bar / scatter / altair） |
| `06_输入.py` | 输入类组件（13 种） |
| `07_媒体.py` | 图片 / 音频 / 视频 |
| `08_布局.py` | columns / tabs / expander / container / empty / dialog |
| `09_状态与提示.py` | success / info / warning / error / spinner / progress / status / toast / balloons / snow |
| `10_侧边栏.py` | `st.sidebar` |
| `11_表单.py` | `st.form` + `st.form_submit_button` |
| `12_会话状态.py` | `st.session_state` 与 `key` |
| `13_缓存机制.py` | `@st.cache_data` 与 `@st.cache_resource` |
| `14_多页面应用/` | 方式 1：`pages/` 目录 + `st.navigation` |
| `15_图书管理系统/` | 综合实战：可用的图书管理系统 |
| `16_多页面_侧边栏方式.py` | 方式 2：单文件 + `session_state` 切页 |
"""
)

st.success("看完本文件，你已经知道怎么把 Streamlit 跑起来了。下一步：`02_页面配置.py`。")

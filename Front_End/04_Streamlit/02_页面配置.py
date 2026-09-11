"""
=====================================================================================
文件：02_页面配置.py
对应课案章节：Streamlit → 页面配置（st.set_page_config）
本节知识点：
  1. st.set_page_config 的四个参数：page_title / page_icon / layout /
     initial_sidebar_state。
  2. ★ 铁律：set_page_config 必须是脚本里第一个 Streamlit 调用，且只能调用一次 ★
  3. layout 的两种取值：centered（居中）vs wide（宽屏）。
  4. initial_sidebar_state 的三种取值：expanded / collapsed / auto。
  5. menu_items（额外知识点）：自定义右上角菜单。
  6. 主题配置：.streamlit/config.toml 里的 [theme] 段。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m streamlit run `
        'F:\\ProGram\\Python_Base\\Front_End\\04_Streamlit\\02_页面配置.py' `
        --server.headless true --server.port 8602 --browser.gatherUsageStats false

【本文件的特殊之处】layout 和 initial_sidebar_state 是"启动时生效一次"的配置，
页面上的下拉框只能"演示"这两个参数的含义，不能真的动态修改它们
（因为 set_page_config 一个脚本只能调用一次，而且必须最先调用）。
这一点本身就是本文件要讲的重点之一。
=====================================================================================
"""

import streamlit as st

# =============================================================================
# ★★★ 全脚本第一个 Streamlit 调用 ★★★
# -----------------------------------------------------------------------------
# set_page_config 做的事情是"配置这一次页面加载"：
# 它必须在任何其它 st.xxx 之前执行，否则 Streamlit 会直接抛异常：
#
#     StreamlitAPIException: set_page_config() can only be called once per app
#     page, and must be called as the first Streamlit command.
#
# 参数说明：
#   page_title              浏览器标签页上的标题
#   page_icon               浏览器标签页上的图标（推荐 emoji，也可以用图片路径或 URL）
#   layout                  "centered"（默认，内容居中，最大宽度约 730px）
#                           "wide"（铺满整个视口宽度，适合放表格和图表）
#   initial_sidebar_state   "expanded"（默认展开）/ "collapsed"（默认折叠）/ "auto"
#   menu_items              自定义右上角"⋮"菜单里的内容（可选）
# =============================================================================
st.set_page_config(
    page_title="02 页面配置",          # 看浏览器标签页，标题就是它
    page_icon="⚙️",                    # 看浏览器标签页左侧的小图标
    layout="centered",                 # 本页用居中布局，方便和 wide 做对比说明
    initial_sidebar_state="expanded",  # 侧边栏默认展开
    menu_items={
        # 右上角菜单里可以放自定义链接和信息
        "Get Help": "https://docs.streamlit.io",
        "Report a bug": None,          # None 表示不显示这一项
        "About": "### 前端基础 · Streamlit 章节\n`02_页面配置.py` 示例。",
    },
)

st.title("02 页面配置")
st.caption("对应课案：Streamlit → 页面配置（st.set_page_config）")

# ---------------------------------------------------------------------------
# 一、课案原文
# ---------------------------------------------------------------------------
st.header("一、课案原文")

st.code(
    '''import streamlit as st

st.set_page_config(
    page_title="我的应用",       # 浏览器标签页标题
    page_icon="🖥️",             # 浏览器标签页图标
    layout="wide",              # 布局：centered(居中) / wide(宽屏)
    initial_sidebar_state="expanded"  # 侧边栏初始状态
)

st.title("页面内容从这里开始")''',
    language="python",
)

st.info(
    "**上面这 6 行就是课案的全部内容。**"
    "本页用的也是这个函数，只是参数值略有不同（`page_icon` 换成了 ⚙️，"
    "`layout` 换成了 centered），方便你对比。"
)

# ---------------------------------------------------------------------------
# 二、参数详解（课案表格）
# ---------------------------------------------------------------------------
st.header("二、参数详解")

st.markdown(
    """
| 参数 | 可选值 | 说明 |
|---|---|---|
| `page_title` | 任意字符串 | 浏览器标签页上的标题。**不写的话默认是脚本文件名** |
| `page_icon` | 任意 emoji / 图片路径 / URL | 浏览器标签页上的图标 |
| `layout` | `"centered"` / `"wide"` | 内容居中（默认）或铺满宽屏 |
| `initial_sidebar_state` | `"expanded"` / `"collapsed"` / `"auto"` | 侧边栏初始状态：展开 / 折叠 / 自动 |
| `menu_items` | 字典 | 自定义右上角菜单（`Get Help` / `Report a bug` / `About`） |
"""
)

# ---------------------------------------------------------------------------
# 三、layout 的区别
# ---------------------------------------------------------------------------
st.header("三、layout：centered 与 wide 的区别")

st.markdown(
    """
- **`centered`（本页正在用的）**：内容居中，最大宽度约 **730px**。
  适合阅读类、表单类页面 —— 一行太长反而不好读。
- **`wide`**：内容**铺满**整个视口宽度。
  适合要放大表格、大数据图表的页面 —— 宽度就是信息量。

**怎么切换？** 改 `set_page_config` 里的 `layout` 参数，然后重新运行脚本。
"""
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**centered（约 730px）**")
    st.code(
        """<div style="
  max-width: 730px;
  margin: 0 auto;">内容</div>""",
        language="html",
    )

with col2:
    st.markdown("**wide（100% 视口）**")
    st.code(
        """<div style="
  max-width: 100%;
  margin: 0;">内容</div>""",
        language="html",
    )

st.markdown(
    "下面这条横线就是本页内容的实际宽度边界 —— 因为本页是 `centered`，"
    "所以两边会有留白。把 `layout` 改成 `wide` 重新运行，留白就会消失。"
)
st.divider()
st.caption("↑ 这条分割线的宽度 = 当前 layout 下的内容宽度")

st.subheader("感受一下宽度差异")
# 一个很宽的元素：在 centered 下会显得"挤"，在 wide 下才舒展
st.dataframe(
    {
        "列1": list(range(1, 11)),
        "列2": [f"数据-{i}" for i in range(1, 11)],
        "列3": [i * 100 for i in range(1, 11)],
        "列4": ["较长的说明文字，用来把表格撑宽一点"] * 10,
    },
    # width="stretch" 让表格占满可用宽度（旧写法是 use_container_width=True，已弃用）
    width="stretch",
)

# ---------------------------------------------------------------------------
# 四、initial_sidebar_state
# ---------------------------------------------------------------------------
st.header("四、initial_sidebar_state：侧边栏初始状态")

st.markdown(
    """
| 取值 | 效果 |
|---|---|
| `"expanded"` | 初始**展开**（默认值） |
| `"collapsed"` | 初始**折叠**（只留一个箭头，用户可点开） |
| `"auto"` | 由 Streamlit 根据屏幕宽度自动判断（手机端通常折叠） |

**注意：** 这个参数只决定**首次加载**时的状态，
用户手动点折叠/展开之后就以用户的操作优先，刷新页面才会回到这个初始值。
"""
)

# 侧边栏里的组件：用 with st.sidebar 块，或者 st.sidebar.xxx()
with st.sidebar:
    st.header("👈 这是侧边栏")
    st.markdown("本页把 `initial_sidebar_state` 设成了 `expanded`。")
    st.divider()
    st.markdown("**改一改试试：**")
    st.code('initial_sidebar_state="collapsed"', language="python")
    st.caption("改成上面这样重新运行，侧边栏一开始就是收起的。")

# ---------------------------------------------------------------------------
# 五、★ 铁律：只能是第一个调用，且只能调用一次
# ---------------------------------------------------------------------------
st.header("五、★ 铁律：第一个调用 + 只能一次")

st.error(
    "`st.set_page_config()` **必须是脚本里第一个 Streamlit 命令**，"
    "而且**整个脚本只能调用一次**。违反任意一条都会直接抛异常。"
)

st.markdown("**错误示范 1：前面有别的 st 调用**")
st.code(
    '''import streamlit as st

st.write("先写点东西")        # ❌ 这一行在前面了
st.set_page_config(page_title="X")   # → 抛 StreamlitAPIException''',
    language="python",
)

st.markdown("**错误示范 2：调用了两次**")
st.code(
    '''import streamlit as st

st.set_page_config(page_title="第一次")
st.write("中间的内容")
st.set_page_config(page_title="第二次")   # ❌ 抛 StreamlitAPIException''',
    language="python",
)

st.markdown("**正确写法**")
st.code(
    '''import streamlit as st

st.set_page_config(page_title="X", layout="wide")   # ✅ 第一行就调用
st.title("页面内容从这里开始")''',
    language="python",
)

st.markdown(
    """
**为什么有这么严格的限制？**
因为 `set_page_config` 设置的是"整个页面外壳"的属性（标题、图标、宽度、侧边栏状态），
这些属性必须在页面开始渲染之前就确定下来，中途没法改。

**多页面应用怎么办？**
用 `st.navigation` + `st.Page` 的多页面应用，
**只在入口脚本（`app.py`）里调用一次** `set_page_config`，
`pages/` 下的子页面**不要再调用** —— 详见 `14_多页面应用/app.py`。
"""
)

# ---------------------------------------------------------------------------
# 六、额外：主题配置
# ---------------------------------------------------------------------------
st.header("六、额外：主题怎么改")

st.markdown(
    """
`set_page_config` 管的是"页面外壳"，而**配色主题**要在配置文件里改。
在项目根目录（或者用户目录 `~/.streamlit/`）建一个 `.streamlit/config.toml`：
"""
)
st.code(
    """[theme]
base = "light"                 # "light" 或 "dark"
primaryColor = "#4f46e5"       # 主色（按钮、滑块、选中态）
backgroundColor = "#ffffff"    # 页面背景色
secondaryBackgroundColor = "#f0f2f6"   # 侧边栏、输入框背景
textColor = "#1f2937"          # 正文文字色
font = "sans serif"            # 字体族

[server]
port = 8602
headless = true

[browser]
gatherUsageStats = false""",
    language="toml",
)

st.markdown(
    "另外，浏览器地址栏右侧的 **⋮ 菜单 → Settings** 里也可以由用户临时切换亮/暗主题。"
)

st.divider()
st.success(
    "**本节要点回顾：**\n"
    "1. `set_page_config` 必须是第一个 Streamlit 调用，且只能调用一次。\n"
    "2. 四个常用参数：`page_title` / `page_icon` / `layout` / `initial_sidebar_state`。\n"
    "3. 多页面应用只在入口脚本里配置一次。"
)

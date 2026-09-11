"""
=====================================================================================
文件：14_多页面应用/app.py
对应课案章节：Streamlit → 多页面应用 → 方式 1：子页面（pages/ 目录 + st.navigation）
本节知识点：
  1. 为什么需要多页面：功能一多，一个文件几百上千行就没法维护了。
  2. 目录结构约定：入口 app.py + pages/ 子目录（文件名以数字前缀排序）。
  3. `st.Page(...)` 注册一个页面：可以指向【函数】，也可以指向【文件路径】。
  4. `st.navigation([...])` 注册导航：默认渲染在侧边栏，自动高亮当前页。
  5. `pg.run()` 执行当前选中的那个页面。
  6. ★★ 铁律：`st.set_page_config` 只在【入口脚本】里调用一次，
     `pages/` 下的子页面绝对不要再调用它。
  7. `st.Page` 的其它参数：title / icon / url_path / default / visibility。

目录结构（本目录）：
    14_多页面应用/
    ├── app.py                  ← 入口（本文件）：配置 + 注册导航 + 运行当前页
    └── pages/
        ├── 1_数据总览.py        ← 子页面
        ├── 2_用户管理.py        ← 子页面
        └── 3_系统设置.py        ← 子页面

运行方式（★ 注意入口是 app.py，不是 pages 里的文件）：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m streamlit run `
        'F:\\ProGram\\Python_Base\\Front_End\\04_Streamlit\\14_多页面应用\\app.py' `
        --server.headless true --server.port 8614 --browser.gatherUsageStats false
=====================================================================================
"""

import datetime

import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# ★★★ 唯一的 set_page_config：只在入口脚本里调用一次 ★★★
# -----------------------------------------------------------------------------
# pages/ 下的子页面绝对不要再调用 st.set_page_config，否则会直接抛：
#     StreamlitAPIException: set_page_config() can only be called once per app page
# =============================================================================
st.set_page_config(
    page_title="管理后台（多页面应用）",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# 首页：用【函数】定义的页面
# =============================================================================
# st.Page 的第一个参数既可以是"文件路径字符串"，也可以是一个"无参函数"。
# 后者适合"内容比较短、不值得单独开文件"的首页 / 关于页。
# =============================================================================
def home() -> None:
    """首页内容。这个函数就是"一个页面"，里面写什么，页面上就显示什么。"""
    st.title("🏠 管理后台首页")
    st.caption("对应课案：Streamlit → 多页面应用 → 方式 1：子页面")

    st.write("欢迎使用管理后台，请在左侧导航栏选择功能页面。")

    # ---- 顶部指标卡：多页面应用里，首页通常放"全局概览" ----
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("用户总数", "1,280", "5.2%")
    col2.metric("本月订单", "3,452", "12.8%")
    col3.metric("系统负载", "42%", "-3.1%")
    col4.metric("今日活跃", "8,921", "3.4%")

    st.divider()

    left, right = st.columns([2, 1])

    with left:
        st.subheader("近 30 天访问趋势")
        # 固定随机种子，保证每次刷新图形一致
        rng = np.random.default_rng(42)
        trend = pd.DataFrame(
            {
                "访问量": 1000 + np.cumsum(rng.normal(0, 80, 30)).round(0),
                "注册数": 60 + np.cumsum(rng.normal(0, 12, 30)).round(0),
            },
            index=pd.date_range("2026-01-01", periods=30, freq="D"),
        )
        st.line_chart(trend, height=320)

    with right:
        st.subheader("系统状态")
        st.success("✅ 数据库连接正常")
        st.success("✅ 缓存服务正常")
        st.warning("⚠️ 磁盘使用率 78%")
        st.info("ℹ️ 下次备份：明天 03:00")

        with st.expander("查看版本信息"):
            st.json(
                {
                    "应用版本": "v1.0.0",
                    "Streamlit": st.__version__,
                    "Python": "3.12",
                    "环境": "本地开发",
                }
            )

    st.divider()

    st.subheader("本多页面应用的结构")
    st.code(
        """14_多页面应用/
├── app.py                ← 入口：set_page_config + st.navigation + pg.run()
└── pages/
    ├── 1_数据总览.py      ← 子页面（文件名前面的数字决定导航顺序）
    ├── 2_用户管理.py
    └── 3_系统设置.py""",
        language="text",
    )
    st.info(
        "**四个页面分别用两种方式注册：**"
        "首页用的是【函数】`st.Page(home, ...)`，"
        "另外三个用的是【文件路径】`st.Page(\"pages/1_数据总览.py\", ...)`。"
        "两种方式可以混用。"
    )


# =============================================================================
# 注册导航 + 运行当前页
# =============================================================================
# st.navigation(pages, position="sidebar", expanded=False)
#   pages     —— st.Page 组成的列表（也可以传"分组字典"）
#   position  —— "sidebar"（默认，放侧边栏）/ "hidden"（隐藏导航，自己用按钮切页）
#   expanded  —— 分组导航时是否默认展开
#
# st.Page(page, title=None, icon=None, url_path=None, default=False, visibility="visible")
#   page       —— 文件路径字符串，或一个无参函数
#   title      —— 导航里显示的名字（不写则从文件名/函数名推导）
#   icon       —— 导航里的图标（emoji）
#   url_path   —— 浏览器地址栏里的路径（不写则自动生成。中文标题建议显式指定）
#   default    —— 是否是默认打开的那个页面（不写则第一个是默认）
#   visibility —— "visible" / "hidden"（hidden 的页面不出现在导航里，但仍可通过 URL 访问）
# =============================================================================
pages = [
    st.Page(home, title="首页", icon="🏠", url_path="home", default=True),
    st.Page("pages/1_数据总览.py", title="数据总览", icon="📊", url_path="overview"),
    st.Page("pages/2_用户管理.py", title="用户管理", icon="👥", url_path="users"),
    st.Page("pages/3_系统设置.py", title="系统设置", icon="⚙️", url_path="settings"),
]

# st.navigation 返回"当前被选中的那个 Page 对象"
pg = st.navigation(pages, position="sidebar")

# 在侧边栏底部加一点全局信息（导航栏是自动生成的，这里只是补充）
with st.sidebar:
    st.markdown("---")
    st.caption("当前用户：admin")
    st.caption(f"版本：v1.0.0　|　{datetime.date.today()}")
    st.caption("Starlette/Streamlit 多页面应用示例")

# ★★ 这一行才会真正执行"当前选中的页面" ★★
# 注意：一定要放在所有"全局配置"之后。
pg.run()

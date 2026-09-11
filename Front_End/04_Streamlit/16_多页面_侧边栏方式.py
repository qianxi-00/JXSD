"""
=====================================================================================
文件：16_多页面_侧边栏方式.py
对应课案章节：Streamlit → 多页面应用 → 方式 2：侧边栏
本节知识点：
  1. 不拆文件，用「侧边栏按钮 + session_state」手动切换页面，所有内容在一个 .py 里。
  2. 与方式 1 的对比：适合 2~3 个页面的简单场景，所有 Streamlit 版本通用。
  3. 核心实现：`st.session_state.page` 记住"当前在哪一页"，
     `if / elif` 根据它渲染不同内容。
  4. 三种导航控件的写法：st.button / st.radio / st.segmented_control 的取舍。
  5. 把"页面"写成函数，主流程只负责分发 —— 代码更清晰。
  6. ★ 对比总结表：方式 1（pages/ 目录 + st.navigation） vs 方式 2（单文件侧边栏）。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m streamlit run `
        'F:\\ProGram\\Python_Base\\Front_End\\04_Streamlit\\16_多页面_侧边栏方式.py' `
        --server.headless true --server.port 8616 --browser.gatherUsageStats false
=====================================================================================
"""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="16 多页面（侧边栏方式）", page_icon="📱", layout="wide")

# =============================================================================
# 一、初始化"当前页面"
# =============================================================================
# ★ 关键：把"当前在第几页"存进 session_state，这样跨脚本重跑才能记住。
if "page" not in st.session_state:
    st.session_state.page = "首页"

# 页面清单：集中定义，方便导航和分发共用
PAGES = ["首页", "数据看板", "用户列表", "系统设置"]


# =============================================================================
# 二、侧边栏导航（三种写法的对比）
# =============================================================================
with st.sidebar:
    st.title("📱 导航")
    st.markdown("---")

    # -----------------------------------------------------------------------
    # 写法 A：用按钮切换（课案原文的写法）
    # -----------------------------------------------------------------------
    st.caption("**写法 A：按钮**（课案原文）")
    for page_name in PAGES:
        # 当前页高亮：用 type="primary" 让它在视觉上"选中"
        is_current = st.session_state.page == page_name
        if st.button(
            page_name,
            key=f"nav_btn_{page_name}",              # ★ 循环里创建按钮，key 必须唯一
            type="primary" if is_current else "secondary",
            width="stretch",
        ):
            st.session_state.page = page_name
            st.rerun()

    st.markdown("---")

    # -----------------------------------------------------------------------
    # 写法 B：用单选按钮切换（最简洁，天然有"选中态"）
    # -----------------------------------------------------------------------
    st.caption("**写法 B：st.radio**（最简洁，推荐）")
    # index 参数让 radio 默认选中"当前页"，这样两种导航不会打架
    picked = st.radio(
        "选择页面",
        PAGES,
        index=PAGES.index(st.session_state.page),
        key="nav_radio",
        label_visibility="collapsed",
    )
    # radio 的值变化时，同步到 session_state
    if picked != st.session_state.page:
        st.session_state.page = picked
        st.rerun()

    st.markdown("---")

    # -----------------------------------------------------------------------
    # 写法 C：分段控件（新版特性，外观像"标签页"，适合 2~4 个页面）
    # -----------------------------------------------------------------------
    st.caption("**写法 C：st.segmented_control**（新版，外观紧凑）")
    seg = st.segmented_control(
        "分段控件导航",
        PAGES,
        default=st.session_state.page,
        key="nav_segmented",
        label_visibility="collapsed",
    )
    # segmented_control 在没选中时可能返回 None，所以要判断
    if seg and seg != st.session_state.page:
        st.session_state.page = seg
        st.rerun()

    st.markdown("---")
    st.caption(f"当前页面：**{st.session_state.page}**")
    st.caption("版本：v1.0.0")


# =============================================================================
# 三、各页面的内容（写成函数，主流程只负责分发）
# =============================================================================
def render_home() -> None:
    """首页。"""
    st.title("🏠 首页")
    st.caption("当前页面由 st.session_state.page 控制")

    st.markdown(
        """
欢迎来到首页。注意看左边的侧边栏 —— **三套导航控件**（按钮 / 单选 / 分段控件）
控制的是**同一个** `st.session_state.page`，所以点哪个都会切页。

这就是「方式 2：侧边栏」的全部原理：**用一个状态变量记住当前页，用 if/elif 分发内容。**
"""
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("用户总数", "1,280", "5.2%")
    col2.metric("本月订单", "3,452", "12.8%")
    col3.metric("系统负载", "42%", "-3.1%")

    st.divider()
    st.subheader("课案原文的代码")
    st.code(
        '''import streamlit as st

st.set_page_config(page_title="多页面演示", page_icon="📱")

# 初始化当前页面
if "page" not in st.session_state:
    st.session_state.page = "首页"

# 侧边栏导航
with st.sidebar:
    st.title("导航")
    if st.button("🏠 首页", use_container_width=True):
        st.session_state.page = "首页"
        st.rerun()
    if st.button("📊 数据", use_container_width=True):
        st.session_state.page = "数据"
        st.rerun()
    if st.button("⚙️ 设置", use_container_width=True):
        st.session_state.page = "设置"
        st.rerun()

# 根据状态渲染不同页面
st.title(f"当前页面：{st.session_state.page}")

if st.session_state.page == "首页":
    st.write("欢迎来到首页")
elif st.session_state.page == "数据":
    st.line_chart({"A": [1, 2, 3], "B": [4, 5, 6]})
elif st.session_state.page == "设置":
    st.slider("音量", 0, 100, 50)''',
        language="python",
    )
    st.info(
        "**本文件对课案原文做了两处改进：**\n"
        "1. 每个页面写成一个函数（`render_home()` 等），主流程只负责 `if/elif` 分发 —— "
        "比把内容全堆在 if 里清晰得多。\n"
        "2. 导航控件用循环生成，并给每个按钮唯一的 `key`（循环里创建组件必须这样）。"
    )


def render_dashboard() -> None:
    """数据看板页。"""
    st.title("📊 数据看板")

    # 固定随机种子，保证每次刷新图形一致
    rng = np.random.default_rng(7)

    st.subheader("销售额趋势")
    trend = pd.DataFrame(
        {
            "线上": 120 + np.cumsum(rng.normal(0, 20, 12)).round(0),
            "线下": 90 + np.cumsum(rng.normal(0, 15, 12)).round(0),
        },
        index=[f"{i}月" for i in range(1, 13)],
    )
    st.line_chart(trend, height=320)

    st.subheader("各渠道构成")
    col_pie, col_bar = st.columns(2)
    with col_pie:
        channel = pd.DataFrame(
            {"销售额": [1230, 805, 480]},
            index=["线上", "线下", "批发"],
        )
        st.bar_chart(channel)
    with col_bar:
        st.area_chart(trend, height=280)

    st.subheader("明细数据")
    st.dataframe(trend, width="stretch")

    # 提供下载
    st.download_button(
        "⬇️ 下载数据（CSV）",
        data=trend.to_csv().encode("utf-8-sig"),
        file_name="看板数据.csv",
        mime="text/csv",
    )


def render_users() -> None:
    """用户列表页。"""
    st.title("👥 用户列表")

    # 这份数据存在 session_state 里，切页再回来依然保留
    if "page2_users" not in st.session_state:
        st.session_state.page2_users = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "姓名": ["张三", "李四", "王五", "赵六"],
                "角色": ["管理员", "普通用户", "普通用户", "访客"],
                "城市": ["北京", "上海", "广州", "深圳"],
                "消费金额": [3200, 1580, 4600, 890],
            }
        )

    df = st.session_state.page2_users

    # 搜索
    keyword = st.text_input("🔍 按姓名搜索", key="page2_search")
    view = df
    if keyword.strip():
        view = df[df["姓名"].str.contains(keyword.strip(), regex=False)]

    st.caption(f"显示 {len(view)} / {len(df)} 条")

    edited = st.data_editor(
        view,
        width="stretch",
        hide_index=True,
        num_rows="dynamic",
        column_config={
            "id": st.column_config.NumberColumn("id", disabled=True),
            "消费金额": st.column_config.NumberColumn("消费金额（元）", format="¥%d"),
            "角色": st.column_config.SelectboxColumn("角色", options=["管理员", "普通用户", "访客"]),
        },
        key="page2_editor",
    )

    if st.button("💾 保存修改", type="primary"):
        # 直接把编辑后的结果写回 session_state
        st.session_state.page2_users = edited
        st.success("已保存到会话状态（切到别的页面再回来，数据还在）")
        st.rerun()


def render_settings() -> None:
    """系统设置页。"""
    st.title("⚙️ 系统设置")

    if "page2_settings" not in st.session_state:
        st.session_state.page2_settings = {"音量": 50, "语言": "简体中文", "深色模式": False}

    current = st.session_state.page2_settings

    with st.form("page2_settings_form"):
        volume = st.slider("音量", 0, 100, int(current["音量"]))
        language = st.selectbox(
            "语言",
            ["简体中文", "English"],
            index=["简体中文", "English"].index(current["语言"]),
        )
        dark = st.checkbox("深色模式", value=bool(current["深色模式"]))
        saved = st.form_submit_button("保存设置", type="primary")

    if saved:
        st.session_state.page2_settings = {"音量": volume, "语言": language, "深色模式": dark}
        st.success("设置已保存")
        st.toast("设置已保存", icon="💾")

    st.markdown("**当前设置：**")
    st.json(st.session_state.page2_settings)


# =============================================================================
# 四、主流程：根据 st.session_state.page 分发到对应的渲染函数
# =============================================================================
st.markdown(f"### 当前页面：`{st.session_state.page}`")
st.caption("对应课案：Streamlit → 多页面应用 → 方式 2：侧边栏")
st.divider()

if st.session_state.page == "首页":
    render_home()
elif st.session_state.page == "数据看板":
    render_dashboard()
elif st.session_state.page == "用户列表":
    render_users()
elif st.session_state.page == "系统设置":
    render_settings()
else:
    # 兜底分支：万一 session_state 里的值不是预期的（比如被手工改过）
    st.error(f"未知页面：{st.session_state.page}")
    if st.button("回到首页"):
        st.session_state.page = "首页"
        st.rerun()

st.divider()

# =============================================================================
# 五、两种方式的对比
# =============================================================================
st.header("两种多页面方式的对比")

st.markdown(
    """
| 对比项 | 方式 1：`pages/` 目录 + `st.navigation` | 方式 2：单文件 + 侧边栏（本文件） |
|---|---|---|
| 文件组织 | 一个页面一个文件（`pages/1_xxx.py`） | 全部在一个 `.py` 里 |
| 导航栏 | **框架自动生成**，自动高亮当前页 | 自己写按钮 / radio |
| 当前页面状态 | 框架管理 | **自己用 `st.session_state` 管理** |
| 浏览器 URL | 每个页面有独立 URL（可分享 / 可前进后退） | 只有一个 URL |
| 页面间共享数据 | 用 `st.session_state` | 用 `st.session_state` |
| 适合规模 | **3 个页面以上的正式项目** | **2~3 个页面的简单小应用** |
| 代码可维护性 | 高（文件隔离） | 页面一多就难维护 |
| 版本要求 | 需要 1.36+（`st.navigation`） | **所有 Streamlit 版本通用** |

**怎么选？**

- 页面上线后要给别人用、需要分享某个页面的链接 → **方式 1**（有独立 URL）。
- 只是自己用的小工具、就 2~3 页 → **方式 2**（写起来最快）。

> **课案原文的总结：**
> 方式 1 多文件拆分、导航自动高亮当前页，适合 3 个页面以上的正式项目；
> 方式 2 单文件，适合 2-3 个页面的简单小应用。
"""
)

st.success(
    "**本节要点回顾：**\n"
    "1. 方式 2 的原理：**用一个 `st.session_state.page` 记住当前页，用 `if/elif` 分发内容**。\n"
    "2. 导航控件三种写法：`st.button`（课案原文）/ `st.radio`（最简洁）/ "
    "`st.segmented_control`（外观紧凑）。\n"
    "3. **循环里创建的组件必须给唯一的 `key`**，否则会报 DuplicateElementId。\n"
    "4. 把每个页面写成函数，主流程只做分发 —— 代码清晰得多。\n"
    "5. 没有独立 URL 是方式 2 最大的短板：不能分享「直接打开某一页」的链接。"
)

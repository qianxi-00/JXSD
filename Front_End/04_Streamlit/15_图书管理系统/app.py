"""
=====================================================================================
文件：15_图书管理系统/app.py
对应课案章节：Streamlit → 综合实战：图书管理系统（入口：登录 + 导航）
本文件职责：**应用入口**。负责登录校验、角色分发、注册导航。

本节知识点：
  1. 登录门禁的标准写法：用 `st.session_state` 记住登录状态，未登录时只渲染登录表单。
  2. 角色权限：管理员和普通用户看到**不同的导航菜单**（这是最基础的权限控制）。
  3. `st.navigation` + `st.Page` 的动态用法：**页面列表可以根据运行时状态拼出来**。
  4. `st.set_page_config` 只在入口调用一次（子页面绝对不能再调）。
  5. 演示账号用 `st.expander` 折叠起来展示，方便第一次使用的人。
  6. `st.rerun()` 在登录/退出后立刻刷新界面。

默认账号（首次运行会自动创建）：
    管理员： admin / admin123
    普通用户： user / user123 、 zhangsan / 123456

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m streamlit run `
        'F:\\ProGram\\Python_Base\\Front_End\\04_Streamlit\\15_图书管理系统\\app.py' `
        --server.headless true --server.port 8615 --browser.gatherUsageStats false
=====================================================================================
"""

import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# 把本文件所在目录加入 sys.path，保证 `import data` 一定能成功。
# 为什么需要？因为 Streamlit 多页面应用里，子页面可能在稍微不同的上下文里被加载。
# 用 pathlib 基于 __file__ 计算绝对路径，不依赖当前工作目录。
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import data  # noqa: E402  （必须在 sys.path 调整之后导入）

# =============================================================================
# ★★★ 唯一的 set_page_config：只在入口脚本调用一次 ★★★
# =============================================================================
st.set_page_config(
    page_title="图书管理系统",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# 一、会话状态初始化
# =============================================================================
# ★ 三件套：是否已登录 / 用户名 / 角色。
#   用 `if "key" not in st.session_state` 判断，避免每次重跑都被重置。
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""
if "role" not in st.session_state:
    st.session_state.role = ""          # "admin" 或 "user"
if "display_name" not in st.session_state:
    st.session_state.display_name = ""


# =============================================================================
# 二、登录 / 注册页面（未登录时显示）
# =============================================================================
def render_login_page() -> None:
    """渲染登录 + 注册页面。这是未登录用户能看到的唯一内容。"""

    # 顶部留白，让登录框看起来居中一些
    st.write("")
    st.write("")

    left, middle, right = st.columns([1, 1.2, 1])

    with middle:
        st.title("📚 图书管理系统")
        st.caption("前端基础课案 · Streamlit 综合实战")

        # 用选项卡把「登录」和「注册」放在一起
        tab_login, tab_register = st.tabs(["🔑 登录", "📝 注册"])

        # ---------------------------------------------------------------
        # 登录表单
        # ---------------------------------------------------------------
        with tab_login:
            # st.form 把两个输入框和一个按钮打包：点提交才算一次
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("用户名", placeholder="比如：admin")
                password = st.text_input("密码", type="password", placeholder="比如：admin123")
                submitted = st.form_submit_button("登录", type="primary", width="stretch")

            if submitted:
                # data.login() 返回 (是否成功, 提示信息, 用户字典)
                ok, message, user = data.login(username, password)
                if not ok:
                    st.error(message)
                elif user is None:
                    st.error("登录失败：未取到用户信息")
                else:
                    # ★ 把登录状态写进 session_state，然后重跑让界面切换
                    st.session_state.logged_in = True
                    st.session_state.username = user.get("username", "")
                    st.session_state.role = user.get("role", "user")
                    st.session_state.display_name = user.get("name", user.get("username", ""))
                    st.toast(message, icon="👋")
                    st.rerun()

            with st.expander("演示账号（点开查看）"):
                st.markdown(
                    """
| 角色 | 用户名 | 密码 | 能做什么 |
|---|---|---|---|
| 管理员 | `admin` | `admin123` | 图书增删改查、用户管理、查看全部借阅记录、重置数据 |
| 普通用户 | `user` | `user123` | 浏览/搜索图书、借书、还书、查看自己的借阅记录 |
| 普通用户 | `zhangsan` | `123456` | 同上 |
"""
                )
                st.caption(
                    "数据文件首次运行时会自动创建（`15_图书管理系统/data/library.json`）。"
                    "如果忘记了改动，可以用管理员身份在「系统设置」里重置数据。"
                )

        # ---------------------------------------------------------------
        # 注册表单
        # ---------------------------------------------------------------
        with tab_register:
            with st.form("register_form", clear_on_submit=True):
                reg_username = st.text_input("用户名 *", placeholder="至少 2 个字符")
                reg_password = st.text_input("密码 *", type="password", placeholder="至少 6 位")
                reg_password2 = st.text_input("确认密码 *", type="password")
                reg_name = st.text_input("姓名", placeholder="留空则用用户名")
                reg_role = st.radio("注册角色", ["普通用户", "管理员"], horizontal=True)
                reg_submitted = st.form_submit_button("注册", type="primary", width="stretch")

            if reg_submitted:
                # 先做一遍界面层的校验（两次密码是否一致）
                if reg_password != reg_password2:
                    st.error("两次输入的密码不一致")
                else:
                    ok, message = data.register_user(
                        username=reg_username,
                        password=reg_password,
                        name=reg_name,
                        role="admin" if reg_role == "管理员" else "user",
                    )
                    if ok:
                        st.success(message + "，现在可以去「登录」标签页登录了。")
                    else:
                        st.error(message)

        st.divider()
        st.caption(
            f"数据文件：`{data.data_file_path()}`　"
            "（JSON 文件持久化，重启服务数据依然保留；想恢复初始数据请用管理员登录后重置）"
        )


# =============================================================================
# 三、首页（首页对两种角色显示的内容略有不同）
# =============================================================================
def render_home() -> None:
    """首页：概览指标 + 使用说明。首页是函数形式的页面（st.Page 可以直接接函数）。"""
    role_label = "管理员" if st.session_state.role == "admin" else "普通用户"
    st.title("📚 图书管理系统")
    st.caption(f"当前登录：**{st.session_state.display_name}**（{role_label}）")

    # ---- 统计指标 ----
    info = data.stats()
    row1 = st.columns(4)
    row1[0].metric("图书种类", info["图书种类"])
    row1[1].metric("馆藏总册数", info["馆藏总册数"])
    row1[2].metric("当前在借", info["当前在借"])
    row1[3].metric("逾期未还", info["逾期未还"])

    row2 = st.columns(4)
    row2[0].metric("可借册数", info["可借册数"])
    row2[1].metric("已借出册数", info["已借出册数"])
    row2[2].metric("用户总数", info["用户总数"])
    row2[3].metric("借阅记录总数", info["借阅记录总数"])

    st.divider()

    left, right = st.columns([3, 2])

    with left:
        st.subheader("馆藏分类分布")
        # 用 pandas 做一次分组统计，再画柱状图
        import pandas as pd

        books = data.list_books()
        if books:
            df = pd.DataFrame(books)
            by_category = df.groupby("category")["total"].sum().sort_values(ascending=False)
            st.bar_chart(by_category, height=300)
        else:
            st.info("馆藏里还没有图书。")

        st.subheader("最新上架的图书")
        recent = data.books_dataframe_rows(data.list_books())[-5:]
        recent.reverse()
        st.dataframe(
            pd.DataFrame(recent)[["id", "书名", "作者", "分类", "可借", "总册数", "状态"]],
            width="stretch",
            hide_index=True,
        )

    with right:
        st.subheader("快速上手")
        if st.session_state.role == "admin":
            st.markdown(
                """
你现在是**管理员**，左侧导航里有：

- **图书管理**：新增 / 修改 / 删除 / 搜索图书
- **用户管理**：查看 / 新增 / 删除用户
- **借阅记录**：查看全部借阅流水
- **数据统计**：馆藏与借阅的数据分析
- **系统设置**：重置数据、查看数据文件路径等

> 提示：管理员的增删改都会**立刻写入 JSON 文件**，刷新浏览器也不会丢。
"""
            )
        else:
            st.markdown(
                """
你现在是**普通用户**，左侧导航里有：

- **图书浏览**：搜索 / 按分类筛选 / 看可借状态
- **我的借阅**：当前在借的书、应还日期、逾期提醒、一键归还
- **借阅记录**：你的历史借阅流水
- **个人中心**：修改密码、查看账号信息

> 提示：借书后「图书浏览」里的可借册数会立刻 -1。
"""
            )

        st.subheader("本系统的模块结构")
        st.code(
            """15_图书管理系统/
├── app.py      ← 入口：登录 + 角色分发 + st.navigation（本文件）
├── admin.py    ← 管理员页面：图书增删改查 / 用户管理 / 借阅记录 / 统计
├── user.py     ← 普通用户页面：浏览 / 借还 / 我的借阅 / 个人中心
├── data.py     ← 数据层：JSON 文件持久化 + 全部业务规则
├── data/
│   └── library.json   ← 数据文件（首次运行自动创建）
└── README.md   ← 使用说明""",
            language="text",
        )

        st.info(
            "**分层设计的好处：**`data.py` 只管数据，三个界面文件只管展示。"
            "要换成数据库，只改 `data.py` 一个文件就够了。"
        )


# =============================================================================
# 四、主流程
# =============================================================================
if not st.session_state.logged_in:
    # ---------------- 未登录：只显示登录页 ----------------
    render_login_page()
else:
    # ---------------- 已登录：按角色注册导航 ----------------
    with st.sidebar:
        st.markdown(
            f"**👤 {st.session_state.display_name}**  \n"
            f"<span style='color:#888;font-size:13px;'>"
            f"{'管理员' if st.session_state.role == 'admin' else '普通用户'}"
            f" @{st.session_state.username}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("---")

    # ★ 动态页面列表：不同角色看到不同的页面集合
    if st.session_state.role == "admin":
        pages = [
            st.Page(render_home, title="首页", icon="🏠", url_path="home", default=True),
            st.Page("admin.py", title="图书管理", icon="📖", url_path="books"),
            st.Page("user.py", title="读者视角", icon="👥", url_path="reader"),
        ]
    else:
        pages = [
            st.Page(render_home, title="首页", icon="🏠", url_path="home", default=True),
            st.Page("user.py", title="我的图书馆", icon="📖", url_path="reader"),
        ]

    pg = st.navigation(pages, position="sidebar")

    # 侧边栏底部：退出登录
    with st.sidebar:
        st.markdown("---")
        if st.button("🚪 退出登录", width="stretch"):
            # 退出时把登录相关的状态全部清掉
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.role = ""
            st.session_state.display_name = ""
            st.rerun()
        st.caption("数据文件：`data/library.json`")

    # ★ 执行当前选中的页面
    pg.run()

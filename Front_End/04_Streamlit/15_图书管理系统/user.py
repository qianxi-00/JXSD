"""
=====================================================================================
文件：15_图书管理系统/user.py
对应课案章节：Streamlit → 综合实战：图书管理系统（普通用户功能）
本文件职责：读者视角的页面 —— 浏览 / 搜索图书、借书、还书、我的借阅、借阅记录、个人中心。

本节知识点：
  1. 课案原文的四个选项卡（借阅图书 / 归还图书 / 借阅记录 / 查询可借书）的完整实现。
  2. 借书表单：课案里让用户手输「书名」和「用户名」；
     本实现改成**下拉选择书籍**（避免手输错书名导致"书不存在"），
     并用当前登录用户作为默认借阅人 —— 这是实际项目里更稳妥的做法。
  3. `st.session_state` 在整个会话里共享，所以这里能直接读到登录用户。
  4. 业务操作的返回约定：`(是否成功, 提示信息)`，界面直接把提示信息显示出来，
     **不会出现任何异常堆栈**。
  5. 逾期提醒：把已逾期的书用 `st.warning` / `st.error` 高亮出来。
  6. 权限：管理员也可以访问本页（"读者视角"），用来代读者借还书。

运行方式：从入口 app.py 进入（用普通用户账号登录，如 user / user123），
         也可以单独运行本文件做调试（会提示需要登录）：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m streamlit run `
        'F:\\ProGram\\Python_Base\\Front_End\\04_Streamlit\\15_图书管理系统\\user.py' `
        --server.headless true --server.port 8618 --browser.gatherUsageStats false
=====================================================================================
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# 保证 `import data` 能找到同目录下的 data.py
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import data  # noqa: E402

# ★ 本文件是子页面，**不要**调用 st.set_page_config()


# =============================================================================
# 权限门禁
# =============================================================================
def require_login() -> None:
    """
    检查是否已登录。未登录就提示并 `st.stop()`。

    管理员也可以访问本页（用于"代读者借还书"），所以这里只要求"已登录"，
    不限制角色。
    """
    if not st.session_state.get("logged_in"):
        st.warning("请先从入口 `app.py` 登录（本页需要登录后才能访问）。")
        st.info(
            "单独运行本文件只能看到这个提示。正确用法是运行入口脚本：\n\n"
            "```\n"
            "streamlit run 15_图书管理系统/app.py\n"
            "```\n\n"
            "然后用普通用户账号登录：`user` / `user123` 或 `zhangsan` / `123456`"
        )
        st.stop()


require_login()

CURRENT_USER = st.session_state.get("username", "")
IS_ADMIN = st.session_state.get("role") == "admin"

# =============================================================================
# 页头
# =============================================================================
st.title("📖 我的图书馆")
st.caption("对应课案：Streamlit → 综合实战：图书管理系统")

mine = data.borrowed_books(CURRENT_USER)
overdue = [b for b in mine if b.get("overdue")]

head = st.columns(4)
head[0].metric("当前在借", len(mine))
head[1].metric("逾期未还", len(overdue))
head[2].metric("历史借阅", len(data.borrow_history(CURRENT_USER)))
head[3].metric(
    "可借图书种类",
    len(data.list_books(only_available=True)),
)

if overdue:
    st.error(
        f"⚠️ 你有 **{len(overdue)}** 本书已经逾期未还，请尽快归还："
        + "、".join(f"《{b['title']}》" for b in overdue)
    )

st.divider()

# =============================================================================
# 用选项卡划分功能（对应课案原文的四个 tab）
# =============================================================================
tab_borrow, tab_return, tab_history, tab_search, tab_profile = st.tabs(
    ["📕 借阅图书", "📗 归还图书", "📋 借阅记录", "🔍 查询可借书", "👤 个人中心"]
)


# =============================================================================
# TAB 1：借阅图书
# =============================================================================
with tab_borrow:
    st.subheader("借阅图书")

    st.markdown(
        """
课案原文的做法是让用户**手动输入书名和用户名**：

```python
with st.form("borrow_book", clear_on_submit=True):
    col1, col2 = st.columns(2)
    book_title = col1.text_input("书名")
    user_name = col2.text_input("用户名")
    if st.form_submit_button("借阅图书"):
        user = data.User.find_user(user_name)
        user.borrow_book(book_title)
        st.success(f"已借阅：{book_title}")
```

**本实现做了一处改进：**把「书名」从手输改成**下拉选择**。
因为手输非常容易打错一个字就找不到书；而课案原文的
`user.borrow_book(book_title)` 在书不存在时会直接抛异常
（`book.status = "借出"` 里的 `book` 是 `None`）。
**健壮的做法是：能选择就不要让用户输入。**
"""
    )

    borrowable = data.list_books(only_available=True)
    if not borrowable:
        st.warning("当前没有可借的图书（所有书都被借完了）。")
    else:
        with st.form("borrow_form", clear_on_submit=False):
            b1, b2 = st.columns([3, 2])

            book_map = {int(b["id"]): b for b in borrowable}
            pick_id = b1.selectbox(
                "选择要借的书",
                options=list(book_map.keys()),
                format_func=lambda bid: (
                    f"《{book_map[bid]['title']}》 — {book_map[bid]['author']}"
                    f"（可借 {book_map[bid]['available']} 册）"
                ),
            )
            # 借阅人默认是当前登录用户；管理员可以改成别人（代借）
            borrower = b2.text_input(
                "借阅人用户名",
                value=CURRENT_USER,
                help="默认是当前登录用户。管理员可以改成其它用户名来「代借」。",
            )

            borrow_submitted = st.form_submit_button(
                "📕 确认借阅", type="primary", width="stretch"
            )

        if borrow_submitted:
            ok, message = data.borrow_book(int(pick_id), borrower.strip())
            if ok:
                st.success(message)
                st.toast("借阅成功", icon="📕")
                st.rerun()
            else:
                st.error(message)

    st.divider()
    st.subheader("当前在借（点下面的按钮可以直接还）")

    if not mine:
        st.info("你当前没有借阅任何图书。")
    else:
        for book in mine:
            with st.container(border=True):
                c1, c2, c3 = st.columns([4, 2, 1])
                c1.markdown(f"**《{book['title']}》**　—　{book['author']}")
                c1.caption(
                    f"分类：{book.get('category', '')}　|　"
                    f"馆藏地：{book.get('place', '')}"
                )

                if book.get("overdue"):
                    c2.error(f"已逾期！应还 {book['due_date']}")
                else:
                    c2.success(f"应还日期 {book['due_date']}")
                c2.caption(f"借出日期 {book['borrow_date']}")

                if c3.button("↩️ 归还", key=f"quick_return_{book['id']}", width="stretch"):
                    ok, message = data.return_book(int(book["id"]), CURRENT_USER)
                    if ok:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)


# =============================================================================
# TAB 2：归还图书
# =============================================================================
with tab_return:
    st.subheader("归还图书")

    st.code(
        '''with st.form("return_book", clear_on_submit=True):
    col1, col2 = st.columns(2)
    book_title = col1.text_input("书名")
    user_name = col2.text_input("用户名")
    if st.form_submit_button("归还图书"):
        user = data.User.find_user(user_name)
        user.return_book(book_title)
        st.success(f"已归还：{book_title}")''',
        language="python",
    )

    if not mine:
        st.info("你当前没有需要归还的图书。")
    else:
        with st.form("return_form", clear_on_submit=False):
            r1, r2 = st.columns([3, 2])

            mine_map = {int(b["id"]): b for b in mine}
            return_id = r1.selectbox(
                "选择要归还的书",
                options=list(mine_map.keys()),
                format_func=lambda bid: (
                    f"《{mine_map[bid]['title']}》 — 借于 {mine_map[bid]['borrow_date']}"
                    f"（应还 {mine_map[bid]['due_date']}）"
                ),
            )
            # 归还人默认当前用户；管理员可代还
            returner = r2.text_input(
                "归还人用户名",
                value=CURRENT_USER,
                help="默认是当前登录用户。管理员可以改成其它用户名来「代还」。",
            )
            return_submitted = st.form_submit_button(
                "📗 确认归还", type="primary", width="stretch"
            )

        if return_submitted:
            ok, message = data.return_book(int(return_id), returner.strip())
            if ok:
                st.success(message)
                st.toast("归还成功", icon="📗")
                st.rerun()
            else:
                st.error(message)

    if IS_ADMIN:
        st.info(
            "你是管理员，可以在「图书管理 → 借阅记录」里对**任意用户**的记录执行强制归还。"
        )


# =============================================================================
# TAB 3：借阅记录
# =============================================================================
with tab_history:
    st.subheader("我的借阅记录")

    st.code(
        '''# 课案原文的写法（注意：borrow_history 是书名列表，字符串）
with st.form("query_history", clear_on_submit=False):
    user_name = st.text_input("用户名")
    submitted = st.form_submit_button("查询")
    if submitted:
        user = data.User.find_user(user_name)
        if user is None:
            st.error("用户不存在")
        else:
            history = user.borrow_history
            for book_title in history:
                st.write(f"- {book_title}")''',
        language="python",
    )

    st.markdown(
        "课案原文只记录了「书名列表」，信息量比较有限。"
        "本实现改成查询**完整的借阅记录**（借出日期、应还日期、归还日期、是否逾期）："
    )

    with st.form("history_form", clear_on_submit=False):
        h1, h2 = st.columns([2, 1])
        history_user = h1.text_input(
            "用户名",
            value=CURRENT_USER,
            help="默认查自己的记录。管理员可以改成任意用户名。",
        )
        history_submitted = h2.form_submit_button("🔍 查询", type="primary", width="stretch")

    # 不在表单里的"实时"查询：默认显示当前用户；提交后显示查询的用户
    if history_submitted:
        st.session_state.history_target_user = history_user.strip()

    target_user = st.session_state.get("history_target_user", CURRENT_USER)

    if not target_user:
        st.warning("请输入用户名。")
    elif data.find_user(target_user) is None:
        st.error(f"用户「{target_user}」不存在")
    else:
        records = data.borrow_history(target_user)
        st.caption(f"用户 **{target_user}** 共有 **{len(records)}** 条借阅记录")

        if not records:
            st.info("该用户暂无借阅记录。")
        else:
            record_df = pd.DataFrame(records)
            st.dataframe(record_df, width="stretch", hide_index=True)

            # 简单统计
            s1, s2, s3 = st.columns(3)
            s1.metric("记录总数", len(record_df))
            s2.metric("未归还", int((record_df["状态"] == "借出").sum()))
            s3.metric("逾期未还", int((record_df["是否逾期"] == "是").sum()))

            st.download_button(
                "⬇️ 导出我的借阅记录（CSV）",
                data=record_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"借阅记录_{target_user}.csv",
                mime="text/csv",
            )


# =============================================================================
# TAB 4：查询可借书（对应课案的 data.Library.list_book()）
# =============================================================================
with tab_search:
    st.subheader("查询可借图书")

    st.code(
        '''# 课案原文（在 data.py 里）
class Library:
    @staticmethod
    def list_book():
        with Session(engine) as session:
            books = session.exec(
                select(Book).where(Book.status == "在馆", Book.place == "佛山图书馆")
            ).all()
            return [{"书名": b.title, "作者": b.author,
                     "馆藏地": b.place, "状态": b.status} for b in books]

# 页面里直接展示
st.dataframe(data.Library.list_book())''',
        language="python",
    )

    st.markdown(
        "本实现用 `data.list_books(only_available=True)` 达到同样效果，"
        "并且提供关键字搜索和分类筛选。"
    )

    f1, f2, f3 = st.columns([2, 1, 1])
    kw = f1.text_input("🔍 搜索（书名 / 作者 / 出版社 / ISBN）", key="reader_search_kw")
    cat_options = data.list_categories()
    cat = f2.selectbox("分类", ["（全部）"] + cat_options, key="reader_search_cat")
    only_avail = f3.checkbox("只看可借", value=True, key="reader_search_avail")

    found = data.list_books(
        keyword=kw,
        category="" if cat == "（全部）" else cat,
        only_available=only_avail,
    )
    found_rows = data.books_dataframe_rows(found)

    st.caption(f"共找到 **{len(found_rows)}** 本")

    if found_rows:
        # 只展示读者关心的列
        show_df = pd.DataFrame(found_rows)[
            ["id", "书名", "作者", "分类", "出版社", "馆藏地", "可借", "总册数", "状态"]
        ]
        st.dataframe(show_df, width="stretch", hide_index=True)

        # 在表格下面直接借书（省得切回"借阅图书"选项卡）
        st.markdown("**快速借阅：**")
        available_books = {int(b["id"]): b for b in found if b["available"] > 0}
        if available_books:
            q1, q2 = st.columns([3, 1])
            quick_id = q1.selectbox(
                "选择要借的书",
                options=list(available_books.keys()),
                format_func=lambda bid: (
                    f"《{available_books[bid]['title']}》 — {available_books[bid]['author']}"
                ),
                key="reader_quick_borrow_select",
            )
            if q2.button("📕 立即借阅", key="reader_quick_borrow_btn", width="stretch"):
                ok, message = data.borrow_book(int(quick_id), CURRENT_USER)
                if ok:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        else:
            st.warning("上面筛选结果里没有可借的书。")
    else:
        st.info("没有符合条件的图书。")


# =============================================================================
# TAB 5：个人中心
# =============================================================================
with tab_profile:
    st.subheader("个人中心")

    me = data.find_user(CURRENT_USER)
    if me is None:
        st.error("当前用户信息读取失败，请重新登录。")
    else:
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**账号信息**")
            st.json(
                {
                    "用户名": me.get("username", ""),
                    "姓名": me.get("name", ""),
                    "角色": "管理员" if me.get("role") == "admin" else "普通用户",
                    "注册时间": me.get("created_at", ""),
                    "当前在借数量": len(me.get("borrowed", [])),
                }
            )

        with c2:
            st.markdown("**修改密码**")
            with st.form("change_password_form", clear_on_submit=True):
                old_pwd = st.text_input("原密码", type="password")
                new_pwd = st.text_input("新密码", type="password", placeholder="至少 6 位")
                new_pwd2 = st.text_input("确认新密码", type="password")
                pwd_submitted = st.form_submit_button("修改密码", type="primary", width="stretch")

            if pwd_submitted:
                if new_pwd != new_pwd2:
                    st.error("两次输入的新密码不一致")
                elif not old_pwd:
                    st.error("请输入原密码")
                else:
                    ok, message = data.change_password(CURRENT_USER, old_pwd, new_pwd)
                    if ok:
                        st.success(message + "（下次登录请使用新密码）")
                        st.toast("密码已修改", icon="🔑")
                    else:
                        st.error(message)

        st.divider()
        st.markdown("**我的借阅概览**")
        if mine:
            overview = pd.DataFrame(
                [
                    {
                        "书名": b["title"],
                        "作者": b["author"],
                        "借出日期": b["borrow_date"],
                        "应还日期": b["due_date"],
                        "是否逾期": "是" if b.get("overdue") else "否",
                    }
                    for b in mine
                ]
            )
            st.dataframe(overview, width="stretch", hide_index=True)
        else:
            st.info("当前没有在借的图书。")

        st.divider()
        st.caption(
            "提示：本系统的数据保存在 `15_图书管理系统/data/library.json`，"
            "刷新浏览器不会丢失；只有管理员在「系统设置」里执行重置才会恢复初始数据。"
        )

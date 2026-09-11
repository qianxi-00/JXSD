"""
=====================================================================================
文件：15_图书管理系统/admin.py
对应课案章节：Streamlit → 综合实战：图书管理系统（管理员功能）
本文件职责：管理员页面 —— **图书增删改查** / 用户管理 / 借阅记录 / 数据统计 / 系统设置。

本节知识点：
  1. 权限门禁：进入页面前先检查 `st.session_state.role`，不满足就 `st.stop()`。
  2. 「增删改查」在 Streamlit 里的标准实现套路：
       · 查：筛选条件 + st.dataframe
       · 增：st.form(clear_on_submit=True) + 提交后 data.add_book()
       · 改：st.selectbox 选一条 → 表单预填当前值 → 提交后 data.update_book()
       · 删：选中一条 → 二次确认 → data.delete_book()
  3. `st.stop()` 的作用：立刻停止脚本执行（不是报错），后面的代码完全不会跑。
  4. 每次写操作之后调用 `st.rerun()`，让页面立刻显示最新数据。
  5. 用 `st.tabs` 把功能分区，避免一个页面太长。
  6. 用 `st.dataframe` 的 `column_config` 优化列显示。

运行方式：从入口 app.py 进入（用管理员账号登录），
         也可以单独运行本文件做调试（会提示需要登录）：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m streamlit run `
        'F:\\ProGram\\Python_Base\\Front_End\\04_Streamlit\\15_图书管理系统\\admin.py' `
        --server.headless true --server.port 8617 --browser.gatherUsageStats false
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

# ★ 注意：本文件是子页面，**不要**调用 st.set_page_config()
#   （配置由入口 app.py 统一负责）


# =============================================================================
# 权限门禁
# =============================================================================
def require_admin() -> None:
    """
    检查当前会话是不是管理员。不是就给出提示并 `st.stop()`。

    `st.stop()` 会**立刻结束本次脚本执行**（不是抛异常），
    所以后面的代码一行都不会跑，也不会出现任何报错。
    """
    # 单独运行本文件时（没有经过 app.py），session_state 里什么都没有，
    # 这时给出一个友好的提示，而不是让页面报错。
    if not st.session_state.get("logged_in"):
        st.warning("请先从入口 `app.py` 登录（本页需要登录后才能访问）。")
        st.info(
            "单独运行本文件只能看到这个提示。正确用法是运行入口脚本：\n\n"
            "```\n"
            "streamlit run 15_图书管理系统/app.py\n"
            "```\n\n"
            "然后用管理员账号登录：`admin` / `admin123`"
        )
        st.stop()

    if st.session_state.get("role") != "admin":
        st.error("⛔ 权限不足：本页面只有管理员可以访问。")
        st.info("请用管理员账号（admin / admin123）登录，或切换到普通用户可用的页面。")
        st.stop()


require_admin()

# =============================================================================
# 页头
# =============================================================================
st.title("📖 图书管理（管理员）")
st.caption("对应课案：Streamlit → 综合实战：图书管理系统")

info = data.stats()
head = st.columns(5)
head[0].metric("图书种类", info["图书种类"])
head[1].metric("馆藏总册数", info["馆藏总册数"])
head[2].metric("可借册数", info["可借册数"])
head[3].metric("当前在借", info["当前在借"])
head[4].metric("逾期未还", info["逾期未还"])

st.divider()

# =============================================================================
# 用选项卡分四个区
# =============================================================================
tab_books, tab_users, tab_records, tab_stats, tab_system = st.tabs(
    ["📚 图书管理", "👥 用户管理", "📋 借阅记录", "📊 数据统计", "⚙️ 系统设置"]
)


# =============================================================================
# TAB 1：图书管理（增 / 删 / 改 / 查）
# =============================================================================
with tab_books:
    st.subheader("① 查询图书")

    # ---- 筛选条件：用 st.form 打包，改完一起查（避免每敲一个字就查询一次）----
    with st.form("book_search_form"):
        c1, c2, c3 = st.columns([2, 1, 1])
        keyword = c1.text_input(
            "关键字（书名 / 作者 / ISBN / 出版社）",
            placeholder="留空表示全部",
        )
        categories = data.list_categories()
        category = c2.selectbox("分类", ["（全部）"] + categories)
        only_available = c3.checkbox("只看可借")
        searched = st.form_submit_button("🔍 查询", type="primary", width="stretch")

    # 记住上次的查询条件（用 session_state），这样点别的按钮不会丢失筛选
    if searched:
        st.session_state.admin_book_query = {
            "keyword": keyword,
            "category": "" if category == "（全部）" else category,
            "only_available": only_available,
        }

    query = st.session_state.get(
        "admin_book_query",
        {"keyword": "", "category": "", "only_available": False},
    )

    books = data.list_books(
        keyword=query["keyword"],
        category=query["category"],
        only_available=query["only_available"],
    )
    rows = data.books_dataframe_rows(books)

    st.caption(
        f"查询条件：关键字=「{query['keyword'] or '（空）'}」、"
        f"分类=「{query['category'] or '（全部）'}」、"
        f"只看可借={query['only_available']} → 命中 **{len(rows)}** 本"
    )

    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
            column_config={
                "id": st.column_config.NumberColumn("id", width="small"),
                "总册数": st.column_config.NumberColumn("总册数", width="small"),
                "可借": st.column_config.NumberColumn("可借", width="small"),
                "已借出": st.column_config.NumberColumn("已借出", width="small"),
                "状态": st.column_config.TextColumn("状态", width="small"),
            },
        )
    else:
        st.info("没有符合条件的图书。")

    st.divider()

    # ---- 新增图书 ----
    st.subheader("② 新增图书")
    with st.expander("➕ 点开填写新书信息", expanded=False):
        with st.form("add_book_form", clear_on_submit=True):
            a1, a2 = st.columns(2)
            new_title = a1.text_input("书名 *", placeholder="必填")
            new_author = a2.text_input("作者 *", placeholder="必填")

            a3, a4 = st.columns(2)
            new_isbn = a3.text_input("ISBN", placeholder="可留空")
            new_category = a4.text_input("分类", value="前端开发")

            a5, a6 = st.columns(2)
            new_publisher = a5.text_input("出版社", placeholder="可留空")
            new_year = a6.number_input(
                "出版年",
                min_value=0,
                max_value=2100,
                value=2024,
                step=1,
                help="填 0 表示不填",
            )

            a7, a8 = st.columns(2)
            new_total = a7.number_input("总册数", min_value=1, max_value=999, value=1, step=1)
            new_place = a8.text_input("馆藏地", value="佛山图书馆")

            add_submitted = st.form_submit_button("➕ 添加图书", type="primary", width="stretch")

        if add_submitted:
            ok, message = data.add_book(
                title=new_title,
                author=new_author,
                isbn=new_isbn,
                category=new_category,
                publisher=new_publisher,
                year=int(new_year),
                total=int(new_total),
                place=new_place,
            )
            if ok:
                st.success(message)
                st.toast("图书已添加", icon="✅")
                st.rerun()
            else:
                st.error(message)

    st.divider()

    # ---- 修改 / 删除 ----
    st.subheader("③ 修改或删除图书")

    all_books = data.list_books()
    if not all_books:
        st.info("馆藏为空，请先添加图书。")
    else:
        # 用 selectbox 选中要操作的那本。format_func 让下拉里显示得更清楚。
        book_options = {b["id"]: b for b in all_books}
        selected_id = st.selectbox(
            "选择一本图书",
            options=list(book_options.keys()),
            format_func=lambda bid: (
                f"id={bid} ｜ 《{book_options[bid]['title']}》 "
                f"— {book_options[bid]['author']} "
                f"（可借 {book_options[bid]['available']}/{book_options[bid]['total']}）"
            ),
            key="admin_selected_book",
        )
        selected = book_options[selected_id]

        edit_col, delete_col = st.columns([3, 1])

        with edit_col:
            # ★ 表单预填当前值：用 value=/index= 参数把已有数据填进去
            with st.form("edit_book_form"):
                st.markdown(f"**正在编辑：《{selected['title']}》**")

                e1, e2 = st.columns(2)
                edit_title = e1.text_input("书名 *", value=selected["title"])
                edit_author = e2.text_input("作者 *", value=selected["author"])

                e3, e4 = st.columns(2)
                edit_isbn = e3.text_input("ISBN", value=selected.get("isbn", ""))
                edit_category = e4.text_input(
                    "分类", value=selected.get("category", "未分类")
                )

                e5, e6 = st.columns(2)
                edit_publisher = e5.text_input("出版社", value=selected.get("publisher", ""))
                edit_year = e6.number_input(
                    "出版年",
                    min_value=0,
                    max_value=2100,
                    value=int(selected.get("year", 0) or 0),
                    step=1,
                )

                e7, e8 = st.columns(2)
                edit_total = e7.number_input(
                    "总册数",
                    min_value=1,
                    max_value=999,
                    value=int(selected.get("total", 1)),
                    step=1,
                    help="改小总册数时，不能小于「已借出」的册数",
                )
                edit_place = e8.text_input("馆藏地", value=selected.get("place", "佛山图书馆"))

                save_submitted = st.form_submit_button(
                    "💾 保存修改", type="primary", width="stretch"
                )

            if save_submitted:
                ok, message = data.update_book(
                    selected_id,
                    title=edit_title.strip(),
                    author=edit_author.strip(),
                    isbn=edit_isbn.strip(),
                    category=edit_category.strip(),
                    publisher=edit_publisher.strip(),
                    year=int(edit_year),
                    total=int(edit_total),
                    place=edit_place.strip(),
                )
                if ok:
                    st.success(message)
                    st.toast("修改已保存", icon="💾")
                    st.rerun()
                else:
                    st.error(message)

        with delete_col:
            st.markdown("**删除这本书**")
            borrowed_count = selected["total"] - selected["available"]
            if borrowed_count > 0:
                st.warning(f"有 {borrowed_count} 册在借，不能删除")
            else:
                # 二次确认：勾选之后才能点删除
                confirmed = st.checkbox("我确认删除", key=f"confirm_del_{selected_id}")
                if st.button(
                    "🗑️ 删除图书",
                    disabled=not confirmed,
                    width="stretch",
                    key=f"del_book_{selected_id}",
                ):
                    ok, message = data.delete_book(selected_id)
                    if ok:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                if not confirmed:
                    st.caption("请先勾选「我确认删除」")


# =============================================================================
# TAB 2：用户管理
# =============================================================================
with tab_users:
    st.subheader("① 用户列表")

    users = data.list_users()
    if users:
        user_df = pd.DataFrame(users)
        st.dataframe(user_df, width="stretch", hide_index=True)

        uc1, uc2, uc3 = st.columns(3)
        uc1.metric("用户总数", len(users))
        uc2.metric("管理员", int((user_df["角色"] == "管理员").sum()))
        uc3.metric("有在借的用户", int((user_df["在借数量"] > 0).sum()))
    else:
        st.info("还没有任何用户。")

    st.divider()

    st.subheader("② 新增用户")
    with st.form("admin_add_user_form", clear_on_submit=True):
        u1, u2 = st.columns(2)
        nm_username = u1.text_input("用户名 *", placeholder="至少 2 个字符，登录时使用")
        nm_password = u2.text_input("密码 *", type="password", placeholder="至少 6 位")

        u3, u4 = st.columns(2)
        nm_name = u3.text_input("姓名", placeholder="留空则用用户名")
        nm_role = u4.selectbox("角色", ["普通用户", "管理员"])

        nm_submitted = st.form_submit_button("➕ 添加用户", type="primary", width="stretch")

    if nm_submitted:
        ok, message = data.register_user(
            username=nm_username,
            password=nm_password,
            name=nm_name,
            role="admin" if nm_role == "管理员" else "user",
        )
        if ok:
            st.success(message)
            st.rerun()
        else:
            st.error(message)

    st.divider()

    st.subheader("③ 删除用户")
    usernames = [u["用户名"] for u in users]
    if usernames:
        del_username = st.selectbox("选择要删除的用户", options=usernames, key="admin_del_user")
        target = data.find_user(del_username)
        if target:
            borrowed_count = len(target.get("borrowed", []))
            st.write(
                f"该用户角色：**{'管理员' if target.get('role') == 'admin' else '普通用户'}**，"
                f"当前在借 **{borrowed_count}** 本。"
            )
        confirm_del_user = st.checkbox("我确认删除该用户", key="confirm_del_user")
        if st.button(
            "🗑️ 删除用户",
            disabled=not confirm_del_user,
            key="admin_del_user_btn",
        ):
            ok, message = data.delete_user(del_username)
            if ok:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
        st.caption("⚠️ 有未归还图书的用户无法删除；系统至少保留一个管理员。")
    else:
        st.info("没有可删除的用户。")


# =============================================================================
# TAB 3：借阅记录
# =============================================================================
with tab_records:
    st.subheader("借阅流水")

    records = data.borrow_history()      # 不传参数 = 全部记录
    if not records:
        st.info("还没有任何借阅记录。可以让普通用户在「我的图书馆」里借几本书。")
    else:
        record_df = pd.DataFrame(records)

        f1, f2, f3 = st.columns(3)
        # 用 multiselect 做筛选：默认全选
        status_filter = f1.multiselect(
            "状态",
            options=sorted(record_df["状态"].unique().tolist()),
            default=sorted(record_df["状态"].unique().tolist()),
            key="admin_record_status",
        )
        user_filter = f2.text_input("按用户名筛选（包含匹配）", key="admin_record_user")
        book_filter = f3.text_input("按书名筛选（包含匹配）", key="admin_record_book")

        view = record_df.copy()
        if status_filter:
            view = view[view["状态"].isin(status_filter)]
        if user_filter.strip():
            view = view[view["用户名"].str.contains(user_filter.strip(), regex=False)]
        if book_filter.strip():
            view = view[view["书名"].str.contains(book_filter.strip(), regex=False)]

        st.caption(f"显示 {len(view)} / {len(record_df)} 条记录")
        st.dataframe(view, width="stretch", hide_index=True, height=380)

        # ---- 管理员强制归还 ----
        st.divider()
        st.subheader("强制归还（管理员）")
        active = record_df[record_df["状态"] == "借出"]
        if len(active) == 0:
            st.success("当前没有未归还的记录。")
        else:
            force_options = {
                int(row["记录号"]): row for _, row in active.iterrows()
            }
            force_id = st.selectbox(
                "选择要强制归还的记录",
                options=list(force_options.keys()),
                format_func=lambda rid: (
                    f"记录{rid} ｜《{force_options[rid]['书名']}》 "
                    f"｜ {force_options[rid]['用户名']} ｜ 应还 {force_options[rid]['应还日期']}"
                ),
                key="admin_force_return",
            )
            if st.button("↩️ 强制归还", key="admin_force_return_btn"):
                # 直接用「记录 id」归还 —— 具体的业务规则仍然只在 data.py 里实现一份
                ok, message = data.return_by_record(force_id)
                if ok:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        # 下载备份
        st.download_button(
            "⬇️ 导出借阅记录（CSV）",
            data=record_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="借阅记录.csv",
            mime="text/csv",
        )


# =============================================================================
# TAB 4：数据统计
# =============================================================================
with tab_stats:
    st.subheader("馆藏与借阅统计")

    stat_info = data.stats()
    s1, s2, s3 = st.columns(3)
    s1.metric("图书种类", stat_info["图书种类"])
    s2.metric("馆藏总册数", stat_info["馆藏总册数"])
    s3.metric("借出率", f"{stat_info['已借出册数'] / max(stat_info['馆藏总册数'], 1):.1%}")

    books_all = data.list_books()
    if books_all:
        book_df = pd.DataFrame(books_all)

        c1, c2 = st.columns(2)
        with c1:
            st.caption("按分类统计馆藏册数")
            by_category = book_df.groupby("category")["total"].sum().sort_values(ascending=False)
            st.bar_chart(by_category, height=300)

        with c2:
            st.caption("按分类统计可借册数")
            by_category_avail = (
                book_df.groupby("category")["available"].sum().sort_values(ascending=False)
            )
            st.bar_chart(by_category_avail, height=300)

        st.caption("借出情况（可借 vs 已借出）")
        # 造一份"可借 / 已借出"的对比数据
        book_df["已借出"] = book_df["total"] - book_df["available"]
        ratio_df = (
            book_df.groupby("category")[["available", "已借出"]].sum()
        )
        ratio_df.columns = ["可借", "已借出"]
        st.area_chart(ratio_df, height=280)

        st.caption("📌 借出最多的图书 Top 10")
        top_books = (
            book_df.assign(已借出=book_df["total"] - book_df["available"])
            .sort_values("已借出", ascending=False)
            .head(10)[["id", "title", "author", "category", "total", "available", "已借出"]]
        )
        top_books.columns = ["id", "书名", "作者", "分类", "总册数", "可借", "已借出"]
        st.dataframe(top_books, width="stretch", hide_index=True)
    else:
        st.info("暂无图书数据。")

    st.divider()

    records_all = data.borrow_history()
    if records_all:
        record_df_all = pd.DataFrame(records_all)
        st.caption("📌 借阅次数最多的读者 Top 10")
        top_users = (
            record_df_all.groupby("用户名")
            .size()
            .reset_index(name="借阅次数")
            .sort_values("借阅次数", ascending=False)
            .head(10)
        )
        st.dataframe(top_users, width="stretch", hide_index=True)

        st.caption("📌 每日借出趋势")
        trend = (
            record_df_all.groupby("借出日期")
            .size()
            .reset_index(name="借出次数")
            .set_index("借出日期")
            .sort_index()
        )
        st.line_chart(trend, height=260)
    else:
        st.info("暂无借阅记录，无法统计。")


# =============================================================================
# TAB 5：系统设置
# =============================================================================
with tab_system:
    st.subheader("数据文件")

    path = data.data_file_path()
    st.code(str(path), language="text")

    st.markdown(
        f"""
- **文件大小**：{path.stat().st_size:,} 字节
- **格式**：JSON（UTF-8，`ensure_ascii=False`，中文直接可读）
- **持久化说明**：本系统**不使用数据库**，所有数据都存在这个 JSON 文件里。
  重启 Streamlit、刷新浏览器、甚至重装 Python，数据都不会丢。
- **自动创建**：第一次运行时会自动创建 `data/` 目录和这个文件，并写入 8 本示例图书和 3 个账号。
"""
    )

    with st.expander("查看数据文件内容（前 4000 字符）"):
        try:
            raw = path.read_text(encoding="utf-8")
            st.code(raw[:4000] + ("\n…（内容过长已截断）" if len(raw) > 4000 else ""), language="json")
        except OSError as exc:
            st.error(f"读取数据文件失败：{exc}")

    st.divider()
    st.subheader("备份与恢复")

    backup_col, _ = st.columns([1, 2])
    try:
        backup_bytes = path.read_bytes()
        backup_col.download_button(
            "⬇️ 下载数据文件备份",
            data=backup_bytes,
            file_name="library_backup.json",
            mime="application/json",
            width="stretch",
        )
    except OSError as exc:
        st.error(f"读取数据文件失败：{exc}")

    st.divider()
    st.subheader("⚠️ 危险操作")

    with st.container(border=True):
        st.error(
            "**重置全部数据**：会把图书、用户、借阅记录全部恢复成初始的示例数据。"
            "你添加的所有内容都会丢失，**此操作不可撤销**。"
        )
        reset_confirm = st.text_input(
            "请输入 RESET 以确认（大小写敏感）",
            key="admin_reset_confirm",
            placeholder="RESET",
        )
        if st.button("💥 重置为初始数据", key="admin_reset_btn"):
            if reset_confirm.strip() != "RESET":
                st.warning("确认文字不正确，操作已取消。请输入 RESET。")
            else:
                data.reset_data()
                # 重置后把登录状态也清掉（因为账号也回到初始状态了）
                st.session_state.logged_in = False
                st.session_state.username = ""
                st.session_state.role = ""
                st.session_state.display_name = ""
                st.success("数据已重置为初始状态，请重新登录。")
                st.rerun()

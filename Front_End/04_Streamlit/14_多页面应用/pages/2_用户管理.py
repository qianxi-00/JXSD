"""
=====================================================================================
文件：14_多页面应用/pages/2_用户管理.py
对应课案章节：Streamlit → 多页面应用 → 方式 1：子页面
本节知识点：
  1. 子页面里的表单提交（课案原文的 `st.form` + `st.form_submit_button`）。
  2. 跨页面共享数据：子页面之间不能直接共享普通变量，
     要用 `st.session_state`（同一会话内所有页面共享）。
  3. 一个完整的「增删改查」迷你实现（用 session_state 当内存数据库）。
  4. 用 `st.data_editor` 做"批量编辑"。

运行方式：运行上一层的入口 app.py（见 1_数据总览.py 的说明）。
=====================================================================================
"""

import pandas as pd
import streamlit as st

# ★ 再次强调：子页面里【不要】调用 st.set_page_config()

st.title("👥 用户管理")
st.caption("对应课案：Streamlit → 多页面应用 → 方式 1：子页面")

st.markdown(
    """
这是一个带**增删改查**的用户管理页。数据存在 `st.session_state` 里，
所以**在同一次会话中切到别的页面再切回来，数据还在**；
但刷新浏览器就会重置（这是 `session_state` 的持久边界，见 `12_会话状态.py`）。
"""
)

# ---------------------------------------------------------------------------
# 一、初始化"内存数据库"
# ---------------------------------------------------------------------------
# ★ 跨页面共享数据的正确方式：st.session_state
#   同一会话里，所有子页面读写的是同一份 session_state。
if "users" not in st.session_state:
    st.session_state.users = [
        {"id": 1, "姓名": "张三", "角色": "管理员", "城市": "北京", "状态": "启用"},
        {"id": 2, "姓名": "李四", "角色": "普通用户", "城市": "上海", "状态": "启用"},
        {"id": 3, "姓名": "王五", "角色": "普通用户", "城市": "广州", "状态": "停用"},
    ]
if "next_user_id" not in st.session_state:
    st.session_state.next_user_id = 4

ROLE_OPTIONS = ["管理员", "普通用户", "访客"]
CITY_OPTIONS = ["北京", "上海", "广州", "深圳", "杭州"]
STATUS_OPTIONS = ["启用", "停用"]


def users_dataframe() -> pd.DataFrame:
    """把 session_state 里的用户列表转成 DataFrame，方便展示和统计。"""
    return pd.DataFrame(st.session_state.users)


# ---------------------------------------------------------------------------
# 二、课案原文的表单示例
# ---------------------------------------------------------------------------
st.subheader("① 课案原文的表单示例（添加用户）")

st.code(
    '''with st.form("add_user", clear_on_submit=True):
    col1, col2 = st.columns(2)
    name = col1.text_input("姓名")
    role = col2.selectbox("角色", ["普通用户", "管理员"])
    if st.form_submit_button("添加用户"):
        st.success(f"已添加：{name}（{role}）")''',
    language="python",
)

# 真实运行：课案原文的结构 + 真正的"写入"
with st.form("add_user", clear_on_submit=True):
    col1, col2 = st.columns(2)
    new_name = col1.text_input("姓名", placeholder="请输入姓名")
    new_role = col2.selectbox("角色", ROLE_OPTIONS)

    col3, col4 = st.columns(2)
    new_city = col3.selectbox("城市", CITY_OPTIONS)
    new_status = col4.selectbox("状态", STATUS_OPTIONS)

    # st.form_submit_button 必须在 with st.form 块内部
    add_submitted = st.form_submit_button("添加用户", type="primary")

if add_submitted:
    if not new_name.strip():
        st.error("姓名不能为空")
    else:
        # 写入"内存数据库"
        st.session_state.users.append(
            {
                "id": st.session_state.next_user_id,
                "姓名": new_name.strip(),
                "角色": new_role,
                "城市": new_city,
                "状态": new_status,
            }
        )
        st.session_state.next_user_id += 1
        st.success(f"已添加：{new_name.strip()}（{new_role}）")
        st.toast("用户已添加", icon="✅")
        # 重跑让下面的表格立刻显示新数据
        st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# 三、用户列表 + 搜索 + 删除
# ---------------------------------------------------------------------------
st.subheader("② 用户列表")

df_users = users_dataframe()

# 顶部指标
c1, c2, c3, c4 = st.columns(4)
c1.metric("用户总数", len(df_users))
c2.metric("管理员", int((df_users["角色"] == "管理员").sum()) if len(df_users) else 0)
c3.metric("启用中", int((df_users["状态"] == "启用").sum()) if len(df_users) else 0)
c4.metric("城市数", df_users["城市"].nunique() if len(df_users) else 0)

# 搜索 + 筛选
search_col, role_col, city_col = st.columns([2, 1, 1])
keyword = search_col.text_input("🔍 搜索（姓名包含关键字）", key="user_search")
role_filter = role_col.multiselect("角色筛选", ROLE_OPTIONS, default=ROLE_OPTIONS, key="user_role_filter")
city_filter = city_col.multiselect("城市筛选", CITY_OPTIONS, default=CITY_OPTIONS, key="user_city_filter")

# 逐层过滤。注意每次 df[...] 都返回新 DataFrame，不会改原数据。
filtered = df_users.copy()
if keyword.strip():
    # str.contains 做"包含"匹配；regex=False 表示不把关键字当正则
    filtered = filtered[filtered["姓名"].str.contains(keyword.strip(), regex=False)]
if role_filter:
    filtered = filtered[filtered["角色"].isin(role_filter)]
if city_filter:
    filtered = filtered[filtered["城市"].isin(city_filter)]

st.caption(f"筛选结果：{len(filtered)} / {len(df_users)} 条")

if len(filtered) > 0:
    st.dataframe(filtered, width="stretch", hide_index=True)

    # ---- 删除：选一个 id，然后删掉 ----
    del_col1, del_col2 = st.columns([3, 1])
    del_id = del_col1.selectbox(
        "选择要删除的用户（按 id）",
        options=filtered["id"].tolist(),
        format_func=lambda i: f"id={i} ｜ {filtered.loc[filtered['id'] == i, '姓名'].iloc[0]}",
        key="user_delete_select",
    )
    if del_col2.button("🗑️ 删除该用户", width="stretch"):
        # 用列表推导重新构造一份"过滤掉该 id"的列表
        st.session_state.users = [u for u in st.session_state.users if u["id"] != del_id]
        st.success(f"已删除 id={del_id} 的用户")
        st.rerun()
else:
    st.warning("没有符合条件的用户。")

st.divider()

# ---------------------------------------------------------------------------
# 四、批量编辑
# ---------------------------------------------------------------------------
st.subheader("③ 批量编辑（st.data_editor）")
st.markdown(
    """
`st.data_editor` 允许用户在表格里**直接修改**。
它的返回值是编辑后的 DataFrame，我们把它**写回 session_state**，
这样切到别的页面再回来，修改依然在。
"""
)

edited = st.data_editor(
    df_users,
    width="stretch",
    hide_index=True,
    num_rows="dynamic",      # 允许用户添加/删除行
    column_config={
        "id": st.column_config.NumberColumn("id", disabled=True, help="主键，不允许修改"),
        "姓名": st.column_config.TextColumn("姓名", required=True),
        "角色": st.column_config.SelectboxColumn("角色", options=ROLE_OPTIONS, required=True),
        "城市": st.column_config.SelectboxColumn("城市", options=CITY_OPTIONS, required=True),
        "状态": st.column_config.SelectboxColumn("状态", options=STATUS_OPTIONS, required=True),
    },
    key="user_data_editor",
)

save_col1, save_col2 = st.columns([1, 3])
if save_col1.button("💾 保存编辑结果", type="primary", width="stretch"):
    # to_dict("records") 把 DataFrame 转成"字典列表"，正好是我们要的结构
    st.session_state.users = edited.to_dict("records")
    # 同步 next_user_id，避免新增的行和已有 id 撞车
    if len(edited) > 0:
        st.session_state.next_user_id = int(edited["id"].max()) + 1
    st.success("已保存到会话状态（切到别的页面再回来，数据还在）")
    st.rerun()

save_col2.caption(
    "提示：这份数据存在 `st.session_state` 里，所以**同一会话内跨页面共享**；"
    "但刷新浏览器就会恢复成初始的 3 条数据。"
)

st.divider()

with st.expander("💡 这个子页面用到了哪些知识点？"):
    st.markdown(
        """
- **表单**：`st.form(clear_on_submit=True)` + `st.form_submit_button` 批量收集输入。
- **会话状态**：`st.session_state.users` —— **这是多页面之间共享数据的唯一正确方式**。
  普通变量会在切页 / 重跑时丢失。
- **数据**：`st.dataframe` / `st.data_editor` / `st.metric` / `st.column_config`。
- **pandas**：`str.contains` 模糊匹配、`isin` 集合筛选、`nunique` 去重计数、
  `to_dict("records")` 转字典列表。
- **交互**：`st.rerun()` 让增删后页面立刻刷新；`st.toast` 做轻量反馈。
"""
    )

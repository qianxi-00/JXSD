"""
=====================================================================================
文件：14_多页面应用/pages/3_系统设置.py
对应课案章节：Streamlit → 多页面应用 → 方式 1：子页面
本节知识点：
  1. 子页面里的 expander / checkbox / slider / selectbox / color_picker（课案原文结构）。
  2. 把设置项存进 `st.session_state`，实现"跨页面生效 + 跨重跑保留"。
  3. 用 `st.form` 做"配置表单"：一次改完一起保存，避免每改一项就重跑。
  4. 「重置为默认值」的标准写法。
  5. 危险操作区域的 UI 惯例（红色边框 + 二次确认）。

运行方式：运行上一层的入口 app.py（见 1_数据总览.py 的说明）。
=====================================================================================
"""

import pandas as pd
import streamlit as st

# ★ 子页面里【不要】调用 st.set_page_config()

st.title("⚙️ 系统设置")
st.caption("对应课案：Streamlit → 多页面应用 → 方式 1：子页面")

st.markdown(
    """
本页演示"设置页面"的标准做法：**用表单一次改完，点保存才生效**，
保存的结果放进 `st.session_state`，这样切到别的页面也能读到
（真实项目里则会写进数据库或配置文件）。
"""
)

# ---------------------------------------------------------------------------
# 默认设置（重置时用）
# ---------------------------------------------------------------------------
DEFAULT_SETTINGS = {
    "邮件通知": True,
    "站内通知": True,
    "短信通知": False,
    "通知频率": 15,
    "语言": "简体中文",
    "主题色": "#2196F3",
    "每页条数": 20,
    "开启调试模式": False,
}

# 初始化：只在第一次运行时把默认值写进 session_state
if "settings" not in st.session_state:
    # dict(...) 做一份浅拷贝，避免后面修改时影响到 DEFAULT_SETTINGS
    st.session_state.settings = dict(DEFAULT_SETTINGS)


# ---------------------------------------------------------------------------
# 一、课案原文的设置面板（expander + 各种组件）
# ---------------------------------------------------------------------------
st.subheader("① 课案原文的写法")

st.code(
    '''with st.expander("通知设置", expanded=True):
    st.checkbox("启用邮件通知", value=True)
    st.checkbox("启用站内通知", value=True)
    st.slider("通知频率（分钟）", 5, 60, 15)

with st.expander("显示设置"):
    st.selectbox("语言", ["简体中文", "English"])
    st.color_picker("主题色", "#2196F3")''',
    language="python",
)

st.markdown("**下面用表单把课案的结构包装起来，一次改完再保存：**")

current = st.session_state.settings

with st.form("settings_form", border=True):
    st.subheader("通知设置")
    col_n1, col_n2, col_n3 = st.columns(3)
    # value= 参数用当前设置做默认值，这样表单打开时显示的是"已保存的值"
    email_on = col_n1.checkbox("启用邮件通知", value=current["邮件通知"])
    site_on = col_n2.checkbox("启用站内通知", value=current["站内通知"])
    sms_on = col_n3.checkbox("启用短信通知", value=current["短信通知"])

    notify_freq = st.slider(
        "通知频率（分钟）",
        min_value=5,
        max_value=60,
        value=int(current["通知频率"]),
        step=5,
        help="每隔多少分钟汇总推送一次通知",
    )

    st.divider()
    st.subheader("显示设置")
    col_d1, col_d2, col_d3 = st.columns(3)
    language = col_d1.selectbox(
        "语言",
        ["简体中文", "English", "日本語"],
        index=["简体中文", "English", "日本語"].index(current["语言"]),
    )
    theme_color = col_d2.color_picker("主题色", current["主题色"])
    page_size = col_d3.number_input(
        "每页显示条数",
        min_value=5,
        max_value=200,
        value=int(current["每页条数"]),
        step=5,
    )

    st.divider()
    debug_mode = st.checkbox("开启调试模式（仅开发环境使用）", value=current["开启调试模式"])

    # 两个提交按钮：主操作 + 重置
    col_save, col_reset = st.columns([1, 1])
    save_clicked = col_save.form_submit_button("💾 保存设置", type="primary", width="stretch")
    reset_clicked = col_reset.form_submit_button("↩️ 重置为默认值", width="stretch")

if save_clicked:
    # 把表单里的值写回 session_state
    st.session_state.settings = {
        "邮件通知": email_on,
        "站内通知": site_on,
        "短信通知": sms_on,
        "通知频率": int(notify_freq),
        "语言": language,
        "主题色": theme_color,
        "每页条数": int(page_size),
        "开启调试模式": debug_mode,
    }
    st.success("设置已保存（存进了 st.session_state.settings）")
    st.toast("设置已保存", icon="💾")

if reset_clicked:
    # 重置：把默认值重新写进去，然后重跑让表单显示回默认值
    st.session_state.settings = dict(DEFAULT_SETTINGS)
    st.info("已重置为默认值")
    st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# 二、当前生效的设置
# ---------------------------------------------------------------------------
st.subheader("② 当前生效的设置")

saved = st.session_state.settings

# 用指标卡展示几个关键项
c1, c2, c3, c4 = st.columns(4)
c1.metric("通知频率", f"{saved['通知频率']} 分钟")
c2.metric("语言", saved["语言"])
c3.metric("每页条数", saved["每页条数"])
c4.metric("调试模式", "开启" if saved["开启调试模式"] else "关闭")

# 用主题色做一块实时预览
st.markdown(
    f"""<div style="background:{saved['主题色']};height:56px;border-radius:10px;
    display:flex;align-items:center;justify-content:center;color:#fff;font-weight:bold;">
    当前主题色实时预览：{saved['主题色']}</div>""",
    unsafe_allow_html=True,
)

with st.expander("查看完整的设置 JSON"):
    st.json(saved)

st.markdown(
    """
**验证「跨页面共享」：**把这里改一改保存，然后切到「首页」或「数据总览」再切回来 ——
设置仍然是保存后的值。因为 `st.session_state` 是**整个会话**共享的，
不是某一个页面私有的。
"""
)

st.divider()

# ---------------------------------------------------------------------------
# 三、危险操作区域
# ---------------------------------------------------------------------------
st.subheader("③ 危险操作区域")

st.markdown(
    """
真实系统的设置页最后通常会有一个"危险操作"区域。它的 UI 惯例是：

1. **视觉上隔离**（红色边框 / 红色标题），让用户不会误点；
2. **二次确认**（弹对话框或要求输入确认文字）；
3. **说明后果**（"此操作不可撤销"）。
"""
)

with st.container(border=True):
    st.error("⚠️ 危险操作区域")
    st.markdown(
        """
- **清空全部会话状态**：会重置本应用所有的临时状态（包括用户管理里添加的数据、设置项）。
- **此操作不可撤销。**
"""
    )

    # 用两列放"确认"和"执行"
    col_confirm, col_do = st.columns([2, 1])
    confirm_text = col_confirm.text_input(
        "请输入 RESET 以确认（大小写敏感）",
        key="danger_confirm",
        placeholder="RESET",
    )

    if col_do.button("💥 清空全部会话状态", width="stretch"):
        if confirm_text.strip() != "RESET":
            st.warning("确认文字不正确，操作已取消。请输入 RESET。")
        else:
            # 清空 session_state（用 list() 包一层，避免边遍历边删除）
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.success("已清空全部会话状态（设置和数据都恢复初始值）")
            st.rerun()

st.divider()

# ---------------------------------------------------------------------------
# 四、设置项的说明表
# ---------------------------------------------------------------------------
st.subheader("④ 设置项说明")

st.dataframe(
    pd.DataFrame(
        {
            "设置项": ["邮件通知", "站内通知", "短信通知", "通知频率", "语言", "主题色", "每页条数", "开启调试模式"],
            "类型": ["bool", "bool", "bool", "int", "str", "str(hex)", "int", "bool"],
            "默认值": ["开启", "开启", "关闭", "15", "简体中文", "#2196F3", "20", "关闭"],
            "说明": [
                "是否通过邮件推送通知",
                "是否在站内消息中心显示",
                "短信有成本，默认关闭",
                "汇总推送的间隔（分钟）",
                "界面语言",
                "主色调，影响按钮和高亮色",
                "表格分页时每页的条数",
                "调试模式会输出更多日志，生产环境必须关闭",
            ],
        }
    ),
    width="stretch",
    hide_index=True,
)

st.divider()

with st.expander("💡 这个子页面用到了哪些知识点？"):
    st.markdown(
        """
- **多页面应用**：`pages/3_系统设置.py`，由 `app.py` 注册。
- **表单**：`st.form` + 两个 `st.form_submit_button`（保存 / 重置）。
  **用表单的好处**：用户改 6 个设置项只触发一次重跑，而不是改一项重跑一次。
- **会话状态**：`st.session_state.settings` 跨页面共享、跨重跑保留；
  「重置为默认值」的写法是**把默认字典重新赋值 + `st.rerun()`**。
- **布局**：`st.expander`、`st.columns`、`st.container(border=True)`。
- **输入组件**：`checkbox` / `slider` / `selectbox` / `color_picker` / `number_input` / `text_input`。
- **UI 惯例**：危险操作用红色 `st.error` 标注 + 输入确认文字 + 二次确认。
"""
    )

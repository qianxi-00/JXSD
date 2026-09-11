"""
=====================================================================================
文件：08_布局.py
对应课案章节：Streamlit → 布局
本节知识点（课案表格里的组件，全部覆盖）：
  1. st.columns            列布局：把页面横向切成 N 列
  2. st.tabs               选项卡：切换显示不同内容
  3. st.expander           展开面板：可折叠的区域
  4. st.container          容器：把一组组件打包（可以带边框）
  5. st.empty              占位符：后续用它替换内容（做"原地更新"的关键）
  6. @st.dialog            弹出对话框（模态框）
  7. st.sidebar            侧边栏（详细内容见 10_侧边栏.py）
额外补充：
  8. st.popover            气泡弹出框（比 dialog 轻量）
  9. st.form               表单（详细内容见 11_表单.py）
  10. st.container(horizontal=True)  横向容器（新版特性）

★ 本文件的一个设计考虑：
  课案里 st.empty() 的示例中有 time.sleep(3)，会让页面每次加载都卡 3 秒。
  本文件把这类"演示性等待"都改成【点按钮才触发】，
  这样正常打开页面时是秒开的，需要看效果时再点按钮。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m streamlit run `
        'F:\\ProGram\\Python_Base\\Front_End\\04_Streamlit\\08_布局.py' `
        --server.headless true --server.port 8608 --browser.gatherUsageStats false
=====================================================================================
"""

import time

import pandas as pd
import streamlit as st

st.set_page_config(page_title="08 布局", page_icon="🧩", layout="wide")

st.title("08 布局组件")
st.caption("对应课案：Streamlit → 布局")

st.markdown(
    """
布局组件本身**不显示内容**，它们的作用是**决定其它组件放在哪里、怎么排列**。

| 组件 | 代码 | 作用 |
|---|---|---|
| 列布局 | `col1, col2 = st.columns(2)` | 将页面分为 N 列 |
| 容器 | `with st.container():` | 创建可重复调用的容器 |
| 占位符 | `placeholder = st.empty()` | 占位，后续可替换内容 |
| 展开面板 | `with st.expander("展开查看"):` | 可折叠/展开的区域 |
| 选项卡 | `tab1, tab2 = st.tabs(["标签1","标签2"])` | 选项卡切换 |
| 弹出框 | `@st.dialog("标题")` | 弹出对话框 |
| 侧边栏 | `st.sidebar.xxx()` | 在侧边栏放置组件 |
"""
)

st.divider()

# ---------------------------------------------------------------------------
# 一、st.columns 列布局
# ---------------------------------------------------------------------------
st.header("一、st.columns 列布局")

st.markdown(
    """
`col1, col2, col3 = st.columns(3)`

- 参数是一个**整数**：等分成 N 列。
- 参数是一个**列表**：按比例分配宽度。`st.columns([1, 2, 1])` = 左中右 1:2:1。
- 参数是 `gap`：列间距，`"small"`（默认）/ `"medium"` / `"large"`。
- 参数是 `vertical_alignment`：`"top"`（默认）/ `"center"` / `"bottom"`。
- **用法：**用 `with col:` 块，或者 `col.xxx()` 直接调用。
- 列里面**还可以再分列**（嵌套），但不要嵌太深，否则可读性会崩。
"""
)

st.code(
    '''col1, col2, col3 = st.columns([1, 2, 1])   # 比例 1:2:1

with col1:
    st.button("左列按钮")
with col2:
    st.button("中间列按钮（更宽）")
with col3:
    st.button("右列按钮")''',
    language="python",
)

col1, col2, col3 = st.columns([1, 2, 1])   # 比例 1:2:1

with col1:
    st.button("左列按钮", width="stretch")
with col2:
    st.button("中间列按钮（更宽）", width="stretch")
with col3:
    st.button("右列按钮", width="stretch")

st.markdown("**常见的几种用法：**")

st.caption("① 等分 N 列（st.columns(4)）")
c = st.columns(4)
for i, col in enumerate(c, start=1):
    with col:
        st.metric(f"指标 {i}", f"{i * 100}")

st.caption("② 按比例分（st.columns([2, 1])）")
c_wide, c_narrow = st.columns([2, 1])
with c_wide:
    st.info("宽度占 2/3")
with c_narrow:
    st.info("宽度占 1/3")

st.caption("③ 列间距（gap）")
c_gap = st.columns(3, gap="large")
for i, col in enumerate(c_gap, start=1):
    with col:
        st.success(f"gap=\"large\" 第 {i} 列")

st.caption("④ 垂直对齐（vertical_alignment）")
c_align = st.columns(3, vertical_alignment="bottom")
with c_align[0]:
    st.write("很短")
with c_align[1]:
    st.write("中等长度的一段文字")
with c_align[2]:
    st.write("这是最长的一段文字，用来把这一列撑高，方便观察底部对齐效果")

st.caption("⑤ 列里嵌列（嵌套）")
outer_left, outer_right = st.columns(2)
with outer_left:
    st.markdown("**左边这块又分成了两列：**")
    inner_a, inner_b = st.columns(2)
    inner_a.metric("内 A", "1")
    inner_b.metric("内 B", "2")
with outer_right:
    st.markdown("**右边是普通内容**")
    st.write("列是可以嵌套的，但一般不超过两层。")

st.divider()

# ---------------------------------------------------------------------------
# 二、st.tabs 选项卡
# ---------------------------------------------------------------------------
st.header("二、st.tabs 选项卡")

st.markdown(
    """
`tab1, tab2, tab3 = st.tabs(["标签1", "标签2", "标签3"])`

- 和浏览器的标签页一样：**同一时间只显示一个标签的内容**，但**所有标签的内容都会被计算**。
- ⚠️ **重要区别：**`st.tabs` 会执行**所有**标签里的代码；
  而 `st.radio` + `if` 只会执行**当前选中**的那一支。
  内容很重（要查数据库）时，用 radio 更省资源。
- 配合 `st.columns` 是组织复杂页面的标准做法。
"""
)

st.code(
    '''tab1, tab2, tab3 = st.tabs(["数据", "图表", "设置"])

with tab1:
    st.write("这是数据区域")
    st.dataframe({"A": [1, 2, 3], "B": [4, 5, 6]})

with tab2:
    st.write("这是图表区域")
    st.line_chart({"A": [1, 2, 3], "B": [4, 5, 6]})

with tab3:
    st.write("这是设置区域")
    st.slider("参数", 0, 100, 50)''',
    language="python",
)

tab1, tab2, tab3 = st.tabs(["数据", "图表", "设置"])

with tab1:
    st.write("这是数据区域")
    st.dataframe(pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}), width="stretch")

with tab2:
    st.write("这是图表区域")
    st.line_chart(pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}))

with tab3:
    st.write("这是设置区域")
    st.slider("参数", 0, 100, 50)

st.info(
    "**验证「所有标签都会执行」：**打开 F12 的 Console，或看运行 streamlit 的那个终端，"
    "你会发现在页面刚打开时，三个标签里的代码全都跑过了。"
    "（本页在三个标签里各放了一句 `st.write`，它们都会被求值。）"
)

st.caption("默认选中哪个标签？用 `default=` 参数（传标签名）：")
tab_x, tab_y = st.tabs(["默认选中的标签", "另一个标签"], default="另一个标签")
with tab_x:
    st.write("我是第一个标签")
with tab_y:
    st.success("我是第二个标签 —— 但因为 default 参数，页面打开时默认选中我")

st.divider()

# ---------------------------------------------------------------------------
# 三、st.expander 展开面板
# ---------------------------------------------------------------------------
st.header("三、st.expander 展开面板")

st.markdown(
    """
`with st.expander("点击展开更多信息 ▼"):`

- 默认**折叠**，用户点击标题才展开 —— 适合放**不常用但需要能查到**的内容。
- `expanded=True`：默认展开。
- `icon="🔍"`：标题左侧加一个图标。
- **注意：**expander 里的代码**总是会执行**（只是结果被隐藏了），
  所以不要在里面放耗时的操作。
"""
)

st.code(
    '''with st.expander("点击展开更多信息 ▼"):
    st.write("这些内容默认是隐藏的")
    st.write("适合放置不常用但需要的信息")
    st.code("print('Hello')")''',
    language="python",
)

with st.expander("点击展开更多信息 ▼"):
    st.write("这些内容默认是隐藏的")
    st.write("适合放置不常用但需要的信息")
    st.code("print('Hello')")

with st.expander("默认展开的面板（expanded=True）", expanded=True):
    st.write("我一打开就是展开状态。")
    st.markdown("可以用来放**使用说明**、**参数解释**这类用户第一次就该看到的内容。")

with st.expander("带图标的面板", icon="🔍"):
    st.write("`icon` 参数给标题加一个图标（传 emoji）。")

st.markdown("**实战用法：把「原始数据」折叠起来，主界面只显示结果**")
st.code(
    '''st.subheader("分析结果")
st.metric("总销售额", "¥1,230 万")

with st.expander("查看原始数据"):
    st.dataframe(raw_df)     # 需要时才展开看''',
    language="python",
)

st.divider()

# ---------------------------------------------------------------------------
# 四、st.container 容器
# ---------------------------------------------------------------------------
st.header("四、st.container 容器")

st.markdown(
    """
`st.container()` 创建一个"容器"，用来把一组组件**打包**在一起。

| 参数 | 作用 |
|---|---|
| `border=True` | 给容器加一个边框（视觉上分组，非常常用） |
| `height=300` | 固定高度，超出部分滚动 |
| `key="..."` | 给容器一个标识（新版特性，可用于定位） |
| `horizontal=True` | 容器内的组件**横向排列**（新版特性，替代 columns 的另一种选择） |

**为什么需要容器？**两个典型场景：

1. **视觉分组**：把"筛选条件"放进一个带边框的容器，把"结果"放进另一个。
2. **★ 顺序填充 ★**：先用 `container = st.container()` 占好位置，
   然后**在页面后面**再往这个容器里写内容 —— 这样就能实现"后算出来的内容显示在前面"。
"""
)

st.code(
    '''# 场景 1：视觉分组
with st.container(border=True):
    st.write("这是一个带边框的容器")
    st.write("里面的内容是一组的")''',
    language="python",
)

with st.container(border=True):
    st.write("这是一个带边框的容器")
    st.write("里面的内容是一组的")

st.markdown("**场景 2：★ 顺序填充（容器的杀手级用法）★**")
st.code(
    '''# 先在页面顶部占一个位置
header_area = st.container()

st.write("这一行在页面上出现得比下面的结果【早】，但在页面里的位置更靠下")

# ...中间做一堆计算...
result = "计算结果：42"

# 最后再往前面那个容器里写 —— 内容会出现在页面【顶部】
header_area.success(result)''',
    language="python",
)

# 真实演示
header_area = st.container()

st.write("① 这一行代码在脚本里出现得更【早】（所以它在页面上位置更靠上）。"  )
st.caption("但是下面这个绿色框，是用页面顶部的容器渲染的 —— 它的代码在最后才执行。")

# 模拟一段计算
computed_value = sum(range(1, 11))     # 1+2+...+10 = 55

# 往前面那个容器里写内容
with header_area:
    st.success(f"② 我的代码写在脚本的最后，但我显示在页面顶部。（计算结果：1+2+...+10 = {computed_value}）")

st.markdown("**横向容器（`horizontal=True`）**")
st.code(
    '''# 容器内的组件横向排列，不用写 columns
with st.container(horizontal=True):
    st.button("按钮 A")
    st.button("按钮 B")
    st.button("按钮 C")''',
    language="python",
)

with st.container(horizontal=True):
    st.button("按钮 A")
    st.button("按钮 B")
    st.button("按钮 C")

st.divider()

# ---------------------------------------------------------------------------
# 五、st.empty 占位符
# ---------------------------------------------------------------------------
st.header("五、st.empty 占位符")

st.markdown(
    """
`placeholder = st.empty()` 创建一个**空占位块**，返回一个 DeltaGenerator 对象。
之后你可以**反复**往它里面写内容 —— **新的会覆盖旧的**。

**三个典型用途：**

1. **进度提示**：一边循环一边更新同一行文字（不刷屏）。
2. **原地替换**：先把"加载中"写进去，算完再替换成结果。
3. **清空内容**：`placeholder.empty()` 把内容清掉。

⚠️ **重要限制：**`st.empty()` 只占**一个元素**的位置。
如果你往里面写了 3 个组件，它们会显示成一组，但下次再写会**整组替换**。
"""
)

st.code(
    '''placeholder = st.empty()
placeholder.write("3秒后这个文字会变化...")
time.sleep(3)
placeholder.success("内容被替换了！")      # ← 覆盖掉上面的内容''',
    language="python",
)

st.warning(
    "课案里这段代码的 `time.sleep(3)` 会让页面每次加载都卡 3 秒。"
    "本文件把它改成了**点按钮才演示**，这样正常打开页面是秒开的。"
)

col_empty_1, col_empty_2 = st.columns([1, 3])

with col_empty_1:
    if st.button("▶ 演示「原地替换」（3 秒）", width="stretch"):
        # 注意：这个 placeholder 是在按钮块里创建的，每次点击都会新建一个
        demo_placeholder = st.empty()
        demo_placeholder.info("正在处理，3 秒后这里会变…")
        time.sleep(3)
        # 新的内容会覆盖旧的内容
        demo_placeholder.success("✅ 内容被替换了！")
    else:
        st.caption("点上面的按钮看效果")

with col_empty_2:
    st.markdown("**另一个更实用的用法：进度文字原地更新（不刷屏）**")
    st.code(
        '''status_box = st.empty()

for i in range(5):
    # 每次都覆盖上一次的内容，所以页面上永远只有一行
    status_box.info(f"正在处理第 {i + 1} / 5 步…")
    time.sleep(0.5)

status_box.success("全部完成！")''',
        language="python",
    )

if st.button("▶ 演示「进度文字原地更新」（2.5 秒）"):
    status_box = st.empty()
    for i in range(5):
        # 每一次循环都覆盖上一轮的内容 —— 页面上始终只有一行
        status_box.info(f"正在处理第 {i + 1} / 5 步…")
        time.sleep(0.5)
    status_box.success("全部完成！（整个过程页面上只出现了这一行）")

st.markdown("**清空占位符：**")
st.code(
    '''box = st.empty()
box.write("一些内容")
# ...过一会儿...
box.empty()        # 把内容清空，占位块还在，但什么都不显示''',
    language="python",
)

st.divider()

# ---------------------------------------------------------------------------
# 六、@st.dialog 弹出对话框
# ---------------------------------------------------------------------------
st.header("六、@st.dialog 弹出对话框")

st.markdown(
    """
`@st.dialog("标题")` 是一个**装饰器**，把普通函数变成"点击后弹出的模态对话框"。

**用法：**

```python
@st.dialog("确认删除")           # ① 用装饰器装饰一个函数
def confirm_dialog():
    st.write("确定要删除吗？")
    if st.button("确定"):
        do_delete()
        st.rerun()              # 关掉对话框需要重跑脚本

# ② 调用这个函数 = 打开对话框
if st.button("删除"):
    confirm_dialog()
```

**关键要点：**

- 对话框函数**只有被调用时才会执行**。
- 对话框里可以放**任何** Streamlit 组件（包括图表、表单）。
- `width`：`"small"`（默认）/ `"large"`。
- `dismissible=False`：强制用户必须做选择才能关闭（做"必须确认"的场景）。
- 关闭对话框的方式：点右上角的 ✕、按 Esc、点遮罩层，或者在函数内部调用 `st.rerun()`。
"""
)

st.code(
    '''if st.button("删除数据"):
    @st.dialog("确认删除")
    def confirm_delete():
        st.write("确定要删除所有数据吗？此操作不可撤销。")
        col_ok, col_cancel = st.columns(2)
        if col_ok.button("确定删除", type="primary"):
            st.session_state.deleted = True
            st.rerun()      # 让状态变化后页面立即更新
        if col_cancel.button("取消"):
            st.rerun()
    confirm_delete()

if "deleted" in st.session_state and st.session_state.deleted:
    st.error("数据已删除")''',
    language="python",
)


# ---- 对话框 1：课案原文的"确认删除" ----
@st.dialog("确认删除")
def confirm_delete_dialog():
    """课案原文的确认删除对话框（函数定义在模块层级，更清晰）。"""
    st.write("确定要删除所有数据吗？此操作不可撤销。")
    col_ok, col_cancel = st.columns(2)

    if col_ok.button("确定删除", type="primary", width="stretch"):
        # 用 session_state 记录"用户确认了"，然后重跑脚本让页面立即更新
        st.session_state.deleted = True
        st.rerun()

    if col_cancel.button("取消", width="stretch"):
        # 直接重跑脚本，对话框会自然关闭（因为它不再被调用）
        st.rerun()


# ---- 对话框 2：一个真正的"表单弹窗" ----
@st.dialog("填写新用户信息")
def add_user_dialog():
    """演示对话框里放表单 —— 这是真实项目里最常见的弹窗用法。"""
    st.caption("这是一个完整的表单弹窗，提交后会写回主页面。")
    with st.form("dialog_form"):
        d_name = st.text_input("姓名", placeholder="请输入姓名")
        d_role = st.selectbox("角色", ["普通用户", "管理员", "访客"])
        d_age = st.number_input("年龄", min_value=1, max_value=120, value=25)
        submitted = st.form_submit_button("保存", type="primary")

    if submitted:
        if not d_name.strip():
            st.error("姓名不能为空")
        else:
            # 把结果写进 session_state，主页面读到后会显示出来。
            # 用「先判断 key 是否存在再初始化」这个标准写法，最稳妥。
            if "dialog_users" not in st.session_state:
                st.session_state.dialog_users = []
            st.session_state.dialog_users.append(
                {"姓名": d_name.strip(), "角色": d_role, "年龄": int(d_age)}
            )
            # ★ 让对话框关闭：重跑脚本，这次不再调用 add_user_dialog()
            st.rerun()


col_dlg1, col_dlg2 = st.columns(2)

with col_dlg1:
    st.subheader("示例 1：确认对话框")
    st.write("模拟一个危险操作：")
    if st.button("🗑️ 删除所有数据", type="primary", width="stretch"):
        confirm_delete_dialog()

    # 读取"用户是否确认过"。用 in 判断 + 属性访问，这是最稳妥的写法
    if st.session_state.get("deleted", False):
        st.error("数据已删除")
        if st.button("撤销删除（恢复）"):
            st.session_state.deleted = False
            st.rerun()
    else:
        st.caption("还没有删除任何数据。")

with col_dlg2:
    st.subheader("示例 2：表单对话框")
    st.write("点击按钮，在弹出的对话框里填写表单：")
    if st.button("➕ 新增用户", width="stretch"):
        add_user_dialog()

    if "dialog_users" not in st.session_state:
        st.session_state.dialog_users = []
    users = st.session_state.dialog_users
    if users:
        st.caption(f"已通过对话框添加了 {len(users)} 个用户：")
        st.dataframe(pd.DataFrame(users), width="stretch", hide_index=True)
    else:
        st.caption("还没有添加任何用户。")

st.info(
    "**对话框的状态问题：**对话框里的组件值同样由 `st.session_state` 管理。"
    "如果你希望「关闭对话框后清空表单」，可以在关闭时删掉对应的 key，"
    "或者给 `st.form` 传 `clear_on_submit=True`。"
)

st.divider()

# ---------------------------------------------------------------------------
# 七、额外：st.popover 气泡弹出框
# ---------------------------------------------------------------------------
st.header("七、额外：st.popover 气泡弹出框")

st.markdown(
    """
`st.popover("点击我")` 会弹出一个**附着在按钮旁边**的小气泡 ——
比 `@st.dialog` 轻量得多（不遮住整个页面）。

**什么时候用哪个？**

| | `@st.dialog` | `st.popover` |
|---|---|---|
| 出现方式 | 屏幕正中的模态框，带遮罩 | 按钮旁边的小气泡 |
| 是否阻断操作 | ✅ 必须处理或关闭 | ❌ 不阻断，可以同时操作页面 |
| 适合 | 确认危险操作、表单弹窗 | 筛选条件、说明提示、小设置面板 |
"""
)

st.code(
    '''with st.popover("⚙️ 筛选条件"):
    st.checkbox("只看已完成")
    st.slider("金额下限", 0, 1000, 100)''',
    language="python",
)

with st.popover("⚙️ 筛选条件"):
    st.checkbox("只看已完成")
    st.slider("金额下限", 0, 1000, 100)

with st.popover("ℹ️ 这是什么？"):
    st.markdown(
        "popover 是**不阻断操作**的气泡弹窗。\n\n"
        "点页面其它地方就会自动关掉。"
    )

st.divider()

# ---------------------------------------------------------------------------
# 八、实战：一个完整的页面骨架
# ---------------------------------------------------------------------------
st.header("八、实战：把这些布局组合成一个页面骨架")

st.markdown("真实的数据应用页面通常长这样：")
st.code(
    '''# 侧边栏放导航和筛选
with st.sidebar:
    st.title("导航")
    menu = st.radio("页面", ["总览", "明细"])

# 顶部一行指标卡
m1, m2, m3, m4 = st.columns(4)
m1.metric("总销售额", "¥1,230 万")
m2.metric("订单数", "3,452")

# 主体分标签
tab_chart, tab_table = st.tabs(["图表", "明细表"])

with tab_chart:
    left, right = st.columns([2, 1])
    with left:
        st.line_chart(sales)          # 主图占 2/3
    with right:
        with st.expander("查看说明", expanded=True):
            st.write("这张图展示了……")   # 侧栏说明占 1/3

with tab_table:
    st.dataframe(detail)''',
    language="python",
)

# 真实渲染这个骨架
with st.container(border=True):
    st.markdown("**顶部：4 个指标卡（st.columns(4)）**")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("总销售额", "¥1,230 万", delta="8.5%")
    m2.metric("订单数", "3,452", delta="12%")
    m3.metric("客单价", "¥36.0", delta="-3.2%")
    m4.metric("活跃用户", "8,921", delta="5.1%")

    st.markdown("**主体：选项卡 + 列布局（st.tabs + st.columns）**")
    frame_tab_chart, frame_tab_table = st.tabs(["图表", "明细表"])

    sales_frame = pd.DataFrame(
        {"月份": ["1月", "2月", "3月", "4月", "5月", "6月"],
         "线上": [120, 180, 150, 220, 260, 300],
         "线下": [90, 110, 140, 130, 160, 175]}
    ).set_index("月份")

    with frame_tab_chart:
        left_frame, right_frame = st.columns([2, 1])
        with left_frame:
            st.line_chart(sales_frame)
        with right_frame:
            with st.expander("查看说明", expanded=True):
                st.write("这张折线图展示了线上和线下渠道 6 个月的销售趋势。")
                st.write("线上增长明显快于线下。")

    with frame_tab_table:
        st.dataframe(sales_frame, width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# 课案原文的完整示例
# ---------------------------------------------------------------------------
st.header("课案原文的完整示例")
st.code(
    '''import streamlit as st
import time

st.title("布局组件演示")

# === 列布局 ===
st.subheader("列布局 (columns)")
col1, col2, col3 = st.columns([1, 2, 1])  # 比例 1:2:1

with col1:
    st.button("左列按钮")
with col2:
    st.button("中间列按钮（更宽）")
with col3:
    st.button("右列按钮")

# === 选项卡 ===
st.subheader("选项卡 (tabs)")
tab1, tab2, tab3 = st.tabs(["数据", "图表", "设置"])

with tab1:
    st.write("这是数据区域")
    st.dataframe({"A": [1, 2, 3], "B": [4, 5, 6]})

with tab2:
    st.write("这是图表区域")
    st.line_chart({"A": [1, 2, 3], "B": [4, 5, 6]})

with tab3:
    st.write("这是设置区域")
    st.slider("参数", 0, 100, 50)

# === 展开面板 ===
st.subheader("展开面板 (expander)")
with st.expander("点击展开更多信息 ▼"):
    st.write("这些内容默认是隐藏的")
    st.write("适合放置不常用但需要的信息")
    st.code("print('Hello')")

# === 容器 ===
st.subheader("容器 (container)")
with st.container(border=True):
    st.write("这是一个带边框的容器")
    st.write("里面的内容是一组的")

# === 占位符 ===
st.subheader("占位符 (empty)")
placeholder = st.empty()
placeholder.write("3秒后这个文字会变化...")
time.sleep(3)
placeholder.success("内容被替换了！")

# === 弹出对话框 ===
st.subheader("弹出对话框 (dialog)")
if st.button("删除数据"):
    @st.dialog("确认删除")
    def confirm_delete():
        st.write("确定要删除所有数据吗？此操作不可撤销。")
        col_ok, col_cancel = st.columns(2)
        if col_ok.button("确定删除", type="primary"):
            st.session_state.deleted = True
            st.rerun()
        if col_cancel.button("取消"):
            st.rerun()
    confirm_delete()

if "deleted" in st.session_state and st.session_state.deleted:
    st.error("数据已删除")''',
    language="python",
)

st.warning(
    "**与课案原文的一处差异：**课案的 `placeholder` 示例里有 `time.sleep(3)`，"
    "会让页面**每次加载**都等待 3 秒。本文件把它改成了点按钮才演示 —— "
    "**这是实际开发中应该做的选择**（演示性的等待不应该拖慢正常使用）。"
)

st.success(
    "**本节要点回顾：**\n"
    "1. `st.columns` 分列（等分或按比例），是数据看板的主力布局。\n"
    "2. `st.tabs` **会执行所有标签里的代码**；内容很重时改用 `st.radio` + `if`。\n"
    "3. `st.expander` 的代码**也总是会执行**，只是结果被折叠了。\n"
    "4. `st.container` 除了分组，还能实现「**先占位、后填充**」，改变内容的显示顺序。\n"
    "5. `st.empty` 是「原地更新」的关键 —— 进度提示、状态替换都靠它。\n"
    "6. `@st.dialog` 做模态框（确认、表单），`st.popover` 做不阻断的小气泡。"
)

"""
=====================================================================================
文件：04_数据.py
对应课案章节：Streamlit → 数据
本节知识点（课案表格里的 4 个组件，全部覆盖）：
  1. st.dataframe   交互表格：可排序、可筛选、可下载
  2. st.table       静态表格：一次性渲染全部数据
  3. st.metric      指标卡：显示指标和变化幅度（delta）
  4. st.json        格式化显示 JSON 数据
额外补充（实战常用）：
  5. st.data_editor + st.column_config   可编辑表格与列格式
  6. st.download_button                  下载数据为 CSV
  7. st.dataframe 的 height / hide_index / selection_mode 参数
  8. pandas Styler 给表格上色

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m streamlit run `
        'F:\\ProGram\\Python_Base\\Front_End\\04_Streamlit\\04_数据.py' `
        --server.headless true --server.port 8604 --browser.gatherUsageStats false
=====================================================================================
"""

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="04 数据", page_icon="📊", layout="wide")

st.title("04 数据组件")
st.caption("对应课案：Streamlit → 数据")

# ---------------------------------------------------------------------------
# 速查表
# ---------------------------------------------------------------------------
st.header("速查表（课案原文）")
st.markdown(
    """
| 组件 | 代码 | 作用 |
|---|---|---|
| 数据表格 | `st.dataframe(df)` | 可排序、可筛选、可下载的交互表格 |
| 静态表格 | `st.table(df)` | 一次性渲染全部数据的静态表 |
| 指标卡 | `st.metric("温度", "25°C", delta="1.5°C")` | 显示指标和变化幅度 |
| JSON | `st.json(data)` | 格式化显示 JSON 数据 |
"""
)

st.divider()

# ---------------------------------------------------------------------------
# 准备一份示例数据
# ---------------------------------------------------------------------------
# 课案里的 df 非常简单，这里先用它建立直觉，再换一份更有代表性的数据。
# 注意：为了让页面每次刷新结果一致（便于对照），这里固定了随机种子。
df = pd.DataFrame(
    {
        "姓名": ["张三", "李四", "王五"],
        "年龄": [25, 30, 28],
        "城市": ["北京", "上海", "广州"],
    }
)

# 更长的一份数据，用来展示交互表格的排序/筛选/滚动能力
np.random.seed(42)
big_df = pd.DataFrame(
    {
        "订单号": [f"NO{1000 + i}" for i in range(50)],
        "城市": np.random.choice(["北京", "上海", "广州", "深圳", "杭州"], size=50),
        "销售额": np.random.randint(1000, 20000, size=50),
        "订单数": np.random.randint(1, 20, size=50),
        "日期": pd.date_range("2026-01-01", periods=50, freq="D"),
    }
)
# 派生一列：客单价 = 销售额 / 订单数，保留两位小数
big_df["客单价"] = (big_df["销售额"] / big_df["订单数"]).round(2)

# ---------------------------------------------------------------------------
# 一、st.dataframe —— 交互表格（最常用）
# ---------------------------------------------------------------------------
st.header("一、st.dataframe：交互表格")

st.markdown(
    """
`st.dataframe` 是 Streamlit 里**最重要的数据展示组件**。它渲染出来的表格：

- 每一列都能**点击表头排序**（升序 / 降序）
- 鼠标悬停右上角会出现**搜索/筛选**按钮
- 右上角有**全屏**和**下载 CSV** 按钮
- 行多时自动出现**纵向滚动条**，列多时自动出现**横向滚动条**
"""
)

st.subheader("课案原文的简单示例")
st.code(
    '''df = pd.DataFrame({
    "姓名": ["张三", "李四", "王五"],
    "年龄": [25, 30, 28],
    "城市": ["北京", "上海", "广州"]
})

st.dataframe(df)''',
    language="python",
)
st.dataframe(df, width="stretch")

st.subheader("50 行的数据（试试点表头排序、点右上角搜索）")
st.dataframe(big_df, width="stretch", height=320)

st.markdown("**常用参数：**")
st.markdown(
    """
| 参数 | 作用 |
|---|---|
| `width="stretch"` | 占满容器宽度（**推荐**；旧的 `use_container_width=True` 已弃用） |
| `width=600` | 指定像素宽度 |
| `height=320` | 固定高度，超出部分滚动；不设则自动适应 |
| `hide_index=True` | 隐藏左侧的行号索引列 |
| `column_order=["城市", "销售额"]` | 指定列的显示顺序 |
| `column_config={...}` | 列的格式配置（见下面「进阶」一节） |
| `on_select="rerun"` + `selection_mode="multi-row"` | 让表格支持"选中行"并回调 |
"""
)

col_a, col_b = st.columns(2)
with col_a:
    st.caption("hide_index=True（隐藏索引列）")
    st.dataframe(df, hide_index=True, width="stretch")
with col_b:
    st.caption('column_order=["城市", "姓名"]（只显示这两列，且按此顺序）')
    st.dataframe(df, column_order=["城市", "姓名"], width="stretch")

st.subheader("支持「选中行」的表格")
st.markdown(
    "设置 `on_select=\"rerun\"` 之后，用户点击行会触发脚本重跑，"
    "通过返回值就能拿到选中的行。"
)
st.code(
    '''event = st.dataframe(
    df,
    on_select="rerun",              # 选中变化时重跑脚本
    selection_mode="multi-row",     # 允许多选；也可以 "single-row" / "multi-column"
)

# event.selection 是一个字典，形如 {"rows": [0, 2], "columns": []}
if event.selection.rows:
    st.write("你选中的行：")
    st.dataframe(df.iloc[event.selection.rows])''',
    language="python",
)

# 真实运行这段逻辑（返回值不是 None 时才处理）
select_event = st.dataframe(
    df,
    on_select="rerun",
    selection_mode="multi-row",
    width="stretch",
    key="demo_select_table",
)
# event.selection 是一个类似字典的对象，用 .rows 取选中的行下标
if select_event.selection and select_event.selection.rows:
    st.success(f"你选中了第 {[i + 1 for i in select_event.selection.rows]} 行")
    st.dataframe(df.iloc[select_event.selection.rows], width="stretch")
else:
    st.caption("（在上面表格里点一行试试，这里会显示你选中的内容）")

st.divider()

# ---------------------------------------------------------------------------
# 二、st.table —— 静态表格
# ---------------------------------------------------------------------------
st.header("二、st.table：静态表格")

st.markdown(
    """
`st.table(df)` 会把**全部数据一次性画出来**，没有排序、没有筛选、没有滚动。

**什么时候用它？**只在数据很少（比如 3~5 行）、而且你要的就是"一张固定的表"时。
数据一多，几万行全画出来页面会非常卡。**绝大多数情况请用 `st.dataframe`。**
"""
)

st.code(
    '''st.table(df)''',
    language="python",
)

col_left, col_right = st.columns(2)
with col_left:
    st.caption("st.table(df) —— 静态小表")
    st.table(df)
with col_right:
    st.caption("st.dataframe(df) —— 交互表格")
    st.dataframe(df, width="stretch")

st.markdown(
    """
| 对比项 | `st.dataframe` | `st.table` |
|---|---|---|
| 排序 / 筛选 / 搜索 | ✅ | ❌ |
| 滚动 | ✅ 超出高度自动滚动 | ❌ 全部铺开 |
| 下载 CSV | ✅ 右上角自带 | ❌ |
| 性能 | 好（大数据也能用） | 差（全量渲染） |
| **结论** | **默认选它** | 只在 3~5 行的小表时用 |
"""
)

st.divider()

# ---------------------------------------------------------------------------
# 三、st.metric —— 指标卡
# ---------------------------------------------------------------------------
st.header("三、st.metric：指标卡")

st.markdown(
    """
`st.metric(label, value, delta=None, delta_color="normal")`

- `label`：指标名称
- `value`：当前值（字符串或数字）
- `delta`：变化幅度。**正数显示绿色 ↑，负数显示红色 ↓**（这是默认行为）
- `delta_color`：`"normal"`（默认，正绿负红）/ `"inverse"`（反过来，正红负绿）/
  `"off"`（不染色，只显示箭头和数值）
"""
)

st.subheader("课案原文的示例")
st.code(
    '''col1, col2, col3 = st.columns(3)
col1.metric("销售额", "¥12,800", delta="8.5%")
col2.metric("订单数", "356", delta="12%")
col3.metric("客单价", "¥36.0", delta="-3.2%", delta_color="inverse")''',
    language="python",
)

col1, col2, col3 = st.columns(3)
col1.metric("销售额", "¥12,800", delta="8.5%")
col2.metric("订单数", "356", delta="12%")
col3.metric("客单价", "¥36.0", delta="-3.2%", delta_color="inverse")

st.markdown("**对照实验：同一个 delta 值，三种 `delta_color` 的效果**")
col_n, col_i, col_o = st.columns(3)
col_n.metric("delta_color='normal'（默认）", "100", delta="5", delta_color="normal")
col_i.metric("delta_color='inverse'（反色）", "100", delta="5", delta_color="inverse")
col_o.metric("delta_color='off'（不上色）", "100", delta="5", delta_color="off")

st.markdown("**负数和无 delta：**")
col_x, col_y, col_z = st.columns(3)
col_x.metric("上个月", "¥9,800", delta="-12%")
col_y.metric("本月", "¥12,800", delta="30%")
col_z.metric("无变化幅度（不传 delta）", "42")

st.info(
    "**实战技巧：**指标卡几乎总是配合 `st.columns` 使用 —— "
    "把 N 个指标横向排成一行，这是数据看板最经典的布局。"
)

st.subheader("实战：由数据算出来的真实指标")
total_sales = int(big_df["销售额"].sum())
total_orders = int(big_df["订单数"].sum())
avg_price = round(total_sales / total_orders, 2)

m1, m2, m3, m4 = st.columns(4)
# delta 也可以算出来：这里假设"上期"是当前值的 92%
m1.metric("总销售额", f"¥{total_sales:,}", delta=f"{total_sales * 0.085:,.0f}")
m2.metric("总订单数", f"{total_orders:,}", delta="12%")
m3.metric("平均客单价", f"¥{avg_price}", delta="-3.2%")
# 这里的 delta 是真实计算出来的：最后一单比第一单的差额
m4.metric(
    "单笔最高销售额",
    f"¥{int(big_df['销售额'].max()):,}",
    delta=f"{int(big_df['销售额'].max() - big_df['销售额'].min()):,}",
    help="help 参数可以加一个鼠标悬停的问号提示",
)

st.divider()

# ---------------------------------------------------------------------------
# 四、st.json —— 格式化显示 JSON
# ---------------------------------------------------------------------------
st.header("四、st.json：格式化显示 JSON")

st.markdown(
    """
`st.json(obj)` 会把 Python 的 **字典 / 列表**格式化成一个可折叠的 JSON 树。

**典型用途：**调试接口返回数据、展示配置项、查看 session_state 的内容。
"""
)

st.code(
    '''st.json({"name": "张三", "age": 25, "hobbies": ["篮球", "编程"]})''',
    language="python",
)

st.json({"name": "张三", "age": 25, "hobbies": ["篮球", "编程"]})

st.markdown("**更复杂的嵌套结构（可以点左侧三角折叠/展开）：**")
st.json(
    {
        "用户": {
            "id": 1001,
            "姓名": "张三",
            "标签": ["VIP", "老用户"],
            "地址": {"省": "广东省", "市": "佛山市", "详细": "某某路 1 号"},
        },
        "订单": [
            {"订单号": "NO1000", "金额": 12800, "状态": "已完成"},
            {"订单号": "NO1001", "金额": 3560, "状态": "待发货"},
        ],
        "统计": {"订单总数": 356, "总金额": 128000.5, "是否活跃": True},
    },
    expanded=2,   # 前 2 层默认展开
)

st.info(
    "**调试利器：**把 `st.session_state` 丢给 `st.json`，"
    "就能在页面上直接看到当前所有会话状态：\n"
    "`st.json(dict(st.session_state))`"
)

st.divider()

# ---------------------------------------------------------------------------
# 五、进阶：st.column_config 与 st.data_editor
# ---------------------------------------------------------------------------
st.header("五、进阶：列格式配置与可编辑表格")

st.subheader("st.column_config：让表格更好看")
st.markdown(
    """
`column_config` 是一个字典，键是列名，值是一种"列类型"，用来控制这一列怎么显示：

| 类型 | 用途 |
|---|---|
| `st.column_config.NumberColumn(format="%.2f")` | 数字格式（千分位、小数位） |
| `st.column_config.ProgressColumn(min_value=0, max_value=100)` | 显示成进度条 |
| `st.column_config.LinkColumn()` | 显示成可点击的链接 |
| `st.column_config.CheckboxColumn()` | 显示成勾选框 |
| `st.column_config.DateColumn(format="YYYY-MM-DD")` | 日期格式 |
| `st.column_config.TextColumn(width="medium", help="...")` | 文本，可设宽度和提示 |
| `st.column_config.BarChartColumn()` | 显示成迷你柱状图 |
"""
)

st.code(
    '''st.dataframe(
    big_df.head(10),
    column_config={
        "销售额": st.column_config.NumberColumn(
            "销售额（元）",
            format="¥%d",              # 数字格式化
            help="本单的总金额",
        ),
        "客单价": st.column_config.NumberColumn(
            "客单价",
            format="%.2f",
        ),
        "日期": st.column_config.DateColumn(
            "下单日期",
            format="YYYY-MM-DD",
        ),
        "订单数": st.column_config.ProgressColumn(
            "订单数（进度条）",
            min_value=0,
            max_value=20,
            format="%d",
        ),
    },
    hide_index=True,
    width="stretch",
)''',
    language="python",
)

st.dataframe(
    big_df.head(10),
    column_config={
        "销售额": st.column_config.NumberColumn(
            "销售额（元）",
            format="¥%d",
            help="本单的总金额",
        ),
        "客单价": st.column_config.NumberColumn("客单价", format="%.2f"),
        "日期": st.column_config.DateColumn("下单日期", format="YYYY-MM-DD"),
        "订单数": st.column_config.ProgressColumn(
            "订单数（进度条）",
            min_value=0,
            max_value=20,
            format="%d",
        ),
    },
    hide_index=True,
    width="stretch",
)

st.subheader("st.data_editor：让用户直接改数据")
st.markdown(
    """
`st.data_editor` 和 `st.dataframe` 长得几乎一样，但它**可以编辑**。
返回值是**编辑后的 DataFrame**，你可以直接拿来用。

`num_rows` 参数控制能不能增删行：`"fixed"`（默认，固定行数）/ `"dynamic"` /
`"add"`（只能加）/ `"delete"`（只能删）。
"""
)

st.code(
    '''edited = st.data_editor(
    df,
    num_rows="dynamic",       # 允许用户增删行
    column_config={
        "年龄": st.column_config.NumberColumn("年龄", min_value=0, max_value=120, step=1),
        "城市": st.column_config.SelectboxColumn(
            "城市", options=["北京", "上海", "广州", "深圳"]
        ),
    },
    width="stretch",
)
st.write("当前表格内容：", edited)''',
    language="python",
)

edited = st.data_editor(
    df,
    num_rows="dynamic",
    column_config={
        "年龄": st.column_config.NumberColumn("年龄", min_value=0, max_value=120, step=1),
        "城市": st.column_config.SelectboxColumn(
            "城市", options=["北京", "上海", "广州", "深圳"]
        ),
    },
    width="stretch",
    key="demo_data_editor",
)
st.write("当前表格内容（编辑后立刻同步）：")
st.dataframe(edited, width="stretch")

st.divider()

# ---------------------------------------------------------------------------
# 六、进阶：下载数据 & 给表格上色
# ---------------------------------------------------------------------------
st.header("六、进阶：下载数据与表格上色")

st.subheader("st.download_button：把数据下载成 CSV")
st.markdown(
    """
**注意一个坑：**DataFrame 不能直接传给 `download_button`，必须先转成
**字符串或字节**。最常用的一行是：

```python
csv = df.to_csv(index=False).encode("utf-8-sig")
```

`utf-8-sig` 会带上 BOM 头 —— **这是让 Excel 打开中文 CSV 不乱码的关键**。
"""
)

csv_bytes = big_df.head(20).to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="⬇️ 下载前 20 行数据（CSV）",
    data=csv_bytes,
    file_name="订单数据.csv",
    mime="text/csv",
    help="用 utf-8-sig 编码，Excel 直接打开不会乱码",
)
st.caption("点上面的按钮，浏览器会下载一个 `订单数据.csv` 文件。")

st.subheader("pandas Styler：给表格上色")
st.markdown(
    "`st.dataframe` 支持 pandas 的 `Styler` 对象，可以用**条件格式**高亮数据 —— "
    "不用手动画图，表格自己就把「哪里高、哪里低」讲清楚了。"
)

st.markdown("**① 渐变色（`background_gradient`）：数值越大颜色越深**")
st.code(
    '''# 销售额越高，背景越绿。cmap 是 matplotlib 的色带名（Greens / Blues / RdYlGn ...）
styled = big_df.head(15).style.background_gradient(subset=["销售额"], cmap="Greens")
st.dataframe(styled, width="stretch")''',
    language="python",
)

styled = big_df.head(15).style.background_gradient(subset=["销售额"], cmap="Greens")
st.dataframe(styled, width="stretch")

st.info(
    "**`background_gradient` 需要 matplotlib。**\n\n"
    "它内部用 matplotlib 的**色带（colormap）**把数值映射成颜色，"
    "所以环境里必须能导入 matplotlib。本项目虚拟环境已具备（matplotlib 3.11.1），"
    "因此上面的渐变色正常显示。\n\n"
    "如果某台机器上没装 matplotlib，这个调用会抛 `ImportError`；"
    "**替代方案是下面 ②③ 两种不依赖 matplotlib 的写法**，效果也够用。"
)

st.markdown("**② 高亮最大 / 最小值（`highlight_max` / `highlight_min`，不依赖 matplotlib）**")
st.code(
    '''# 只把「销售额」列里最大的那个单元格涂色
styled = big_df.head(15).style.highlight_max(subset=["销售额"], color="#bbf7d0")
st.dataframe(styled, width="stretch")

# 也可以同时高亮最大和最小
styled = (big_df.head(15).style
          .highlight_max(subset=["销售额"], color="#bbf7d0")
          .highlight_min(subset=["销售额"], color="#fecaca"))''',
    language="python",
)

styled_hl = big_df.head(15).style.highlight_max(subset=["销售额"], color="#bbf7d0")
st.dataframe(styled_hl, width="stretch")
st.caption("上面「销售额」列里最大的那个单元格被涂成了浅绿色。")

st.markdown("**③ 自定义规则（`Styler.map`，不依赖 matplotlib）**")
st.code(
    '''def mark_high_sales(value):
    """销售额超过 15000 就涂红，否则不涂。返回的就是一段 CSS。"""
    return "background-color: #fecaca; font-weight: bold" if value > 15000 else ""

st.dataframe(
    big_df.head(10).style.map(mark_high_sales, subset=["销售额"]),
    width="stretch",
)''',
    language="python",
)


def mark_high_sales(value: float) -> str:
    """销售额超过 15000 就涂红，否则不涂。返回的就是一段 CSS。"""
    return "background-color: #fecaca; font-weight: bold" if value > 15000 else ""


st.dataframe(
    big_df.head(10).style.map(mark_high_sales, subset=["销售额"]),
    width="stretch",
)

st.markdown("**④ 组合使用：渐变色 + 数字格式化**")
st.code(
    '''styled = (
    big_df.head(12).style
    .background_gradient(subset=["客单价"], cmap="Blues")   # 渐变色
    .format({"销售额": "¥{:,}", "客单价": "{:.2f}"})          # 数字格式化
)
st.dataframe(styled, width="stretch")''',
    language="python",
)

styled_combo = (
    big_df.head(12).style
    .background_gradient(subset=["客单价"], cmap="Blues")
    .format({"销售额": "¥{:,}", "客单价": "{:.2f}"})
)
st.dataframe(styled_combo, width="stretch")

st.markdown(
    """
**常用的 `Styler` 方法一览：**

| 方法 | 依赖 matplotlib | 作用 |
|---|---|---|
| `background_gradient(cmap=...)` | ✅ 需要 | 按数值大小做**渐变色**（最直观） |
| `bar(subset=..., color=...)` | ✅ 需要 | 在单元格里画**迷你条形图** |
| `highlight_max` / `highlight_min` | ❌ 不需要 | 高亮最大 / 最小值 |
| `highlight_null` | ❌ 不需要 | 高亮缺失值（数据清洗时很有用） |
| `format({...})` | ❌ 不需要 | 数字 / 日期的显示格式 |
| `map(func, subset=...)` | ❌ 不需要 | **自定义规则**（`func` 返回一段 CSS） |
| `apply(func, axis=...)` | ❌ 不需要 | 按行 / 列整体计算后再上色 |

> ⚠️ **两个实践提醒：**
> 1. `Styler` 只影响**显示**，不会改变底层数据 —— 导出的 CSV 里还是原始数值。
> 2. 数据量很大（几万行以上）时，逐格计算样式会明显变慢。
>    这种情况用 `st.column_config.ProgressColumn` / `BarChartColumn` 更划算
>    （由前端渲染，不走 Python）。
"""
)

st.divider()

# ---------------------------------------------------------------------------
# 课案原文的完整示例
# ---------------------------------------------------------------------------
st.header("课案原文的完整示例")
st.code(
    '''import streamlit as st
import pandas as pd

df = pd.DataFrame({
    "姓名": ["张三", "李四", "王五"],
    "年龄": [25, 30, 28],
    "城市": ["北京", "上海", "广州"]
})

st.subheader("交互表格 (dataframe)")
st.dataframe(df, use_container_width=True)

st.subheader("静态表格 (table)")
st.table(df)

st.subheader("指标卡 (metric)")
col1, col2, col3 = st.columns(3)
col1.metric("销售额", "¥12,800", delta="8.5%")
col2.metric("订单数", "356", delta="12%")
col3.metric("客单价", "¥36.0", delta="-3.2%", delta_color="inverse")

st.subheader("JSON")
st.json({"name": "张三", "age": 25, "hobbies": ["篮球", "编程"]})''',
    language="python",
)

st.warning(
    "**注意版本差异：**课案里写的是 `st.dataframe(df, use_container_width=True)`，"
    "这个参数在 Streamlit 1.61 里已经被标记为弃用（会在运行时打印提示）。"
    "**新写法是 `st.dataframe(df, width=\"stretch\")`** —— 本文件用的都是新写法。"
)

st.success(
    "**本节要点回顾：**\n"
    "1. 展示表格默认用 `st.dataframe`（可排序、可筛选、可下载）；"
    "`st.table` 只适合几行的小表。\n"
    "2. `st.metric` 的 `delta` 默认「正绿负红」，"
    "用 `delta_color=\"inverse\"` 可以反色（比如「客单价涨了但老板不高兴」这种场景）。\n"
    "3. `st.json` 是调试接口数据和 `session_state` 的利器。\n"
    "4. `column_config` 让表格从「能看」变成「好看」。\n"
    "5. 下载 CSV 记得 `.encode(\"utf-8-sig\")`，否则 Excel 打开中文乱码。"
)

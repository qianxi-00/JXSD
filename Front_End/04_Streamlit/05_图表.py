"""
=====================================================================================
文件：05_图表.py
对应课案章节：Streamlit → 图表
本节知识点（课案表格里的 4 个内置图表，全部覆盖）：
  1. st.line_chart     折线图
  2. st.area_chart     面积图
  3. st.bar_chart      柱状图
  4. st.scatter_chart  散点图
额外补充：
  5. st.altair_chart   Altair 声明式图表（Streamlit 内置图表底层就是它，能力更强）
  6. st.pyplot(fig)    ★ matplotlib 图表（科研/统计图最常用）
  7. 图表常用参数：x / y / color / stack / horizontal / height
  8. st.map 的用法（课案未提，这里只用文字说明，因为地图瓦片需要联网）

★ 三套图表方案怎么选（本文件三套都真实演示了）：
  ┌────────────┬───────────────────┬──────────────────────────────────────┐
  │ 方案        │ 入口               │ 什么时候用                            │
  ├────────────┼───────────────────┼──────────────────────────────────────┤
  │ 内置图表    │ st.line_chart 等   │ 快速看一下数据的形状，一行代码出图      │
  │ Altair     │ st.altair_chart    │ 要交互（悬停提示/缩放/图例筛选）、      │
  │            │                    │ 多图层、直方图、饼图                   │
  │ matplotlib │ st.pyplot(fig)     │ 要精细控制（子图、双轴、误差棒、        │
  │            │                    │ 统计图、科研出版风格）；输出是静态图片  │
  └────────────┴───────────────────┴──────────────────────────────────────┘

  ★ matplotlib 的两个必备准备工作（本文件已在代码里做好）：
    ① 渲染后端设为 "Agg"（无窗口后端）—— 服务器/Streamlit 环境里没有图形界面，
       必须用 Agg 出图，而且 `matplotlib.use("Agg")` 必须在 `import pyplot` 之前调用；
    ② 指定中文字体 —— matplotlib 默认字体 DejaVu Sans 没有中文字形，
       中文会变成一个个小方块。要设置 `plt.rcParams["font.sans-serif"]`，
       并把 `axes.unicode_minus` 设为 False（否则负号也会变方块）。
    ③ 更简单的做法是让 Streamlit 帮忙渲染：把 `fig` 交给 `st.pyplot(fig)`，
       **不要调用 `plt.show()`**（它只在本地窗口里用，服务器上会卡住）。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m streamlit run `
        'F:\\ProGram\\Python_Base\\Front_End\\04_Streamlit\\05_图表.py' `
        --server.headless true --server.port 8605 --browser.gatherUsageStats false
=====================================================================================
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# =============================================================================
# matplotlib 的准备工作
# -----------------------------------------------------------------------------
# ★ 顺序很重要：matplotlib.use("Agg") 必须在 import matplotlib.pyplot 之前调用！
#   如果在 pyplot 之后再切换后端，matplotlib 会警告"后端已经初始化过了"。
# =============================================================================
import matplotlib

# "Agg" 是一个纯光栅化、不出窗口的后端。它的名字来自 Agg 渲染库。
# 服务器环境没有显示器，用默认的交互式后端会报错或卡住，所以必须指定它。
matplotlib.use("Agg")

import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib import font_manager                  # noqa: E402


def _pick_cjk_font() -> str | None:
    """
    从候选列表里挑一个**本机真正装了**的中文字体名。

    为什么要这样挑？因为直接写 plt.rcParams["font.sans-serif"] = ["SimHei"]
    在没装这个字体的机器上不会报错，只会让中文全部变成小方块（"豆腐块"）。
    先用 font_manager.findfont(fallback_to_default=False) 探测一下，
    确认存在再使用；一个都没有就返回 None，后面把标题换成英文。

    返回：可用的中文字体名，或 None。
    """
    for name in (
        "Microsoft YaHei",      # 微软雅黑（Windows 默认有）
        "SimHei",               # 黑体（Windows 默认有）
        "SimSun",               # 宋体
        "Noto Sans CJK SC",     # Linux 常见
        "Source Han Sans SC",   # 思源黑体
        "PingFang SC",          # macOS 常见
        "WenQuanYi Micro Hei",  # Linux 常见
    ):
        try:
            # fallback_to_default=False：找不到就抛异常，而不是悄悄回退到 DejaVu Sans
            font_manager.findfont(
                font_manager.FontProperties(family=name),
                fallback_to_default=False,
            )
            return name
        except Exception:      # noqa: BLE001 —— 找不到是正常情况，继续试下一个
            continue
    return None


CJK_FONT = _pick_cjk_font()

if CJK_FONT:
    # 把中文字体放在字体族列表的最前面，matplotlib 会优先用它。
    # 后面保留 DejaVu Sans 作为"兜底"（它负责数学符号等）。
    plt.rcParams["font.sans-serif"] = [CJK_FONT, "DejaVu Sans"]
    # 默认的 unicode_minus 用的是 DejaVu 里的"−"（U+2212），中文字体里可能没有，
    # 会让负号显示成方块。设为 False 表示用普通的 ASCII 连字符 "-"。
    plt.rcParams["axes.unicode_minus"] = False
    # 顺便调一下默认字号，让图在网页里看起来更舒服
    plt.rcParams["font.size"] = 10


def label(zh: str, en: str) -> str:
    """
    中文优先的文案函数：有中文字体就用中文，没有就退回英文。

    这样这段代码在任何机器上都不会出现"豆腐块"，也不需要 `# -*- coding` 之类的技巧。
    """
    return zh if CJK_FONT else en


st.set_page_config(page_title="05 图表", page_icon="📈", layout="wide")

st.title("05 图表组件")
st.caption("对应课案：Streamlit → 图表")

# ---------------------------------------------------------------------------
# 速查表
# ---------------------------------------------------------------------------
st.header("速查表（课案原文）")
st.markdown(
    """
| 组件 | 代码 | 作用 |
|---|---|---|
| 折线图 | `st.line_chart(data)` | 折线图 |
| 面积图 | `st.area_chart(data)` | 面积图 |
| 柱状图 | `st.bar_chart(data)` | 柱状图 |
| 散点图 | `st.scatter_chart(data)` | 散点图 |
"""
)

st.info(
    "**本文件会演示三套图表方案**，方便你对比：\n\n"
    "| 方案 | 入口 | 特点 |\n"
    "|---|---|---|\n"
    "| **Streamlit 内置图表** | `st.line_chart` 等 | 一行代码出图，底层是 Altair/Vega-Lite |\n"
    "| **Altair** | `st.altair_chart(chart)` | 声明式，交互强（悬停/缩放/图例筛选/多图层） |\n"
    "| **matplotlib** | `st.pyplot(fig)` | 精细控制（子图、双轴、误差棒），输出**静态图片** |\n\n"
    "本环境的版本：matplotlib **"
    + matplotlib.__version__
    + "**（渲染后端 `"
    + matplotlib.get_backend()
    + "`）、Altair **"
    + alt.__version__
    + "**。"
    + (
        f" 已自动选用中文字体 **{CJK_FONT}**，所以图表里的中文能正常显示。"
        if CJK_FONT
        else " ⚠️ 本机没有找到中文字体，图表标题已自动改用英文（避免出现方块字）。"
    )
)

st.divider()

# ---------------------------------------------------------------------------
# 准备示例数据
# ---------------------------------------------------------------------------
# 课案原文用的是 np.random.randn(50, 3) —— 3 列各 50 个标准正态随机数。
# 为了让每次刷新看到的图一致，这里固定随机种子。
np.random.seed(42)

chart_data = pd.DataFrame(
    np.random.randn(50, 3),
    columns=["A列", "B列", "C列"],
)
# 再做一份更像真实业务的数据：6 个月的销售数据
months = ["1月", "2月", "3月", "4月", "5月", "6月"]
sales_data = pd.DataFrame(
    {
        "月份": months,
        "线上": [120, 180, 150, 220, 260, 300],
        "线下": [90, 110, 140, 130, 160, 175],
        "批发": [60, 70, 65, 80, 95, 110],
    }
).set_index("月份")

# 散点图用的数据：两个变量，看它们的相关性
scatter_data = pd.DataFrame(
    {
        "广告投入": np.random.randint(10, 100, size=60),
        "销售额": np.random.randint(100, 1000, size=60),
        "地区": np.random.choice(["华北", "华东", "华南"], size=60),
    }
)
# 让销售额和广告投入呈现正相关（这样散点图看起来才有意义）
scatter_data["销售额"] = (scatter_data["广告投入"] * 8 + np.random.randint(-80, 80, size=60)).clip(lower=50)

st.divider()

# ---------------------------------------------------------------------------
# 一、st.line_chart —— 折线图
# ---------------------------------------------------------------------------
st.header("一、st.line_chart 折线图")

st.markdown(
    """
**用途：**展示数据随**时间**的变化趋势。这是最常用的图表类型。

- 传入 DataFrame 时，**索引（index）自动作为 x 轴**，
  **每一列是一条线**，列名就是图例。
- 图例可以点击，用来**显示/隐藏**某条线。
- 鼠标悬停会显示具体数值。
"""
)

st.subheader("课案原文的示例")
st.code(
    '''chart_data = pd.DataFrame(
    np.random.randn(50, 3),
    columns=["A列", "B列", "C列"]
)

st.line_chart(chart_data)''',
    language="python",
)
st.line_chart(chart_data)

st.subheader("更像真实业务的例子（6 个月销售趋势）")
st.code(
    '''sales_data = pd.DataFrame({
    "月份": ["1月", "2月", "3月", "4月", "5月", "6月"],
    "线上": [120, 180, 150, 220, 260, 300],
    "线下": [90, 110, 140, 130, 160, 175],
    "批发": [60, 70, 65, 80, 95, 110],
}).set_index("月份")      # ★ 设成索引，它才会自动变成 x 轴

st.line_chart(sales_data)''',
    language="python",
)
st.line_chart(sales_data)

st.markdown("**常用参数：**")
st.markdown(
    """
| 参数 | 作用 |
|---|---|
| `x="列名"` | 指定哪一列作为 x 轴（不指定则用索引） |
| `y="列名"` 或 `y=["列1", "列2"]` | 指定要画哪些列 |
| `color="列名"` | 按某列分组着色（长表格式时用） |
| `x_label` / `y_label` | 坐标轴标题 |
| `width="stretch"` / `height=320` | 宽高 |

**⚠️ 最常见的坑：x 轴不对。**
`st.line_chart` 默认拿 **DataFrame 的 index** 当 x 轴。
如果你的日期/月份是一个**普通列**而不是索引，就要么 `.set_index("日期")`，
要么显式写 `st.line_chart(df, x="日期", y=["线上", "线下"])`。
"""
)

st.subheader("用 x / y 参数显式指定（不设索引的写法）")
st.code(
    '''st.line_chart(sales_data.reset_index(), x="月份", y=["线上", "线下"], y_label="销售额")''',
    language="python",
)
st.line_chart(sales_data.reset_index(), x="月份", y=["线上", "线下"], y_label="销售额（万元）")

st.divider()

# ---------------------------------------------------------------------------
# 二、st.area_chart —— 面积图
# ---------------------------------------------------------------------------
st.header("二、st.area_chart 面积图")

st.markdown(
    """
**用途：**和折线图几乎一样，只是线下方面积被填充了颜色。
**适合表达"累积量"或"各部分加起来是总量"**（比如各渠道销售额合计）。

- **默认是堆叠（stack）的** —— 各列依次往上垒，看的是总量和构成。
- 想改成"各自独立的面积"，传 `stack=False`。
"""
)

st.code(
    '''st.area_chart(sales_data)               # 默认堆叠：看总量与构成
st.area_chart(sales_data, stack=False)  # 不堆叠：看各条线自己的量''',
    language="python",
)

st.subheader("堆叠面积图（默认 stack=True）")
st.area_chart(sales_data)

st.subheader("非堆叠面积图（stack=False）")
st.area_chart(sales_data, stack=False)

st.subheader("课案原文的示例（随机数据）")
st.area_chart(chart_data)

st.divider()

# ---------------------------------------------------------------------------
# 三、st.bar_chart —— 柱状图
# ---------------------------------------------------------------------------
st.header("三、st.bar_chart 柱状图")

st.markdown(
    """
**用途：**比较**不同类别之间**的大小（比如各城市销售额）。

- 默认**按类别聚合**（多列会堆叠）。
- `horizontal=True` 变成横向条形图 —— 类别名字很长时更好看。
- `sort=True`（默认）会按值排序；传 `sort=False` 保持原顺序。
- `stack=False` 让多列并排显示（分组柱状图），而不是堆叠。
"""
)

st.code(
    '''st.bar_chart(sales_data)                          # 纵向，堆叠
st.bar_chart(sales_data, stack=False)             # 纵向，分组并排
st.bar_chart(sales_data, horizontal=True)         # 横向条形
st.bar_chart(sales_data, sort=False)              # 不排序''',
    language="python",
)

c1, c2 = st.columns(2)
with c1:
    st.caption("默认：堆叠柱状图")
    st.bar_chart(sales_data)
with c2:
    st.caption("stack=False：分组并排")
    st.bar_chart(sales_data, stack=False)

st.caption("horizontal=True：横向条形图（类别名长的时候更好读）")
st.bar_chart(sales_data, horizontal=True, height=280)

st.subheader("课案原文的示例（随机数据）")
st.bar_chart(chart_data)

st.divider()

# ---------------------------------------------------------------------------
# 四、st.scatter_chart —— 散点图
# ---------------------------------------------------------------------------
st.header("四、st.scatter_chart 散点图")

st.markdown(
    """
**用途：**看**两个数值变量之间的关系**（相关性）。

- `x` / `y`：指定横纵坐标的列。
- `color`：按某列分组着色（分类变量）。
- `size`：按某列的数值决定点的大小（气泡图）。
"""
)

st.code(
    '''st.scatter_chart(scatter_data, x="广告投入", y="销售额")
st.scatter_chart(scatter_data, x="广告投入", y="销售额", color="地区")
st.scatter_chart(scatter_data, x="广告投入", y="销售额", color="地区", size="销售额")''',
    language="python",
)

st.caption("基础散点图：看广告投入与销售额的关系")
st.scatter_chart(scatter_data, x="广告投入", y="销售额")

st.caption("按地区着色（color 参数）")
st.scatter_chart(scatter_data, x="广告投入", y="销售额", color="地区")

st.caption("按地区着色 + 按销售额决定点大小（气泡图）")
st.scatter_chart(scatter_data, x="广告投入", y="销售额", color="地区", size="销售额")

st.subheader("课案原文的示例（随机数据，无 x/y 参数）")
st.scatter_chart(chart_data)

st.divider()

# ---------------------------------------------------------------------------
# 五、进阶：st.altair_chart
# ---------------------------------------------------------------------------
st.header("五、进阶：st.altair_chart（能力更强）")

st.markdown(
    """
Streamlit 的内置图表**底层就是 Altair**，所以内置图表能做的它都能做，
而内置图表做不到的（多图层、双轴、自定义 tooltip、图例位置、参考线……）它也能做。

**Altair 的核心思想是"声明式"：**你不写"怎么画"，而是描述
「用什么数据、什么图形、x 映射到哪个字段、y 映射到哪个字段」，
由 Altair 生成出 Vega-Lite 规范交给浏览器渲染。

```python
alt.Chart(数据)              # 1. 用哪份数据
   .mark_bar()              # 2. 用什么图形（bar/line/point/area/arc...）
   .encode(                 # 3. 字段映射（哪个列 → 哪个视觉通道）
       x="月份:N",           #    N=名义型（分类），Q=定量，T=时间，O=有序
       y="销售额:Q",
       color="渠道:N",
   )
   .properties(width=600, height=300)
```

`st.altair_chart(chart, width="stretch")` 把它渲染出来（`width="stretch"` = 自适应宽度）。
"""
)

st.subheader("示例 1：带数据标签的柱状图")
st.code(
    '''base = alt.Chart(sales_reset).encode(
    x=alt.X("月份:N", sort=None, title="月份"),
    y=alt.Y("线上:Q", title="线上销售额（万元）"),
)

bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
    color=alt.Color("线上:Q", scale=alt.Scale(scheme="blues"), legend=None)
)
# 在柱子顶部叠加文字标签（Altair 用 + 号把多个图层叠起来）
labels = base.mark_text(dy=-8, fontWeight="bold").encode(text="线上:Q")

st.altair_chart((bars + labels).properties(height=320), width="stretch")''',
    language="python",
)

sales_reset = sales_data.reset_index()
base = alt.Chart(sales_reset).encode(
    x=alt.X("月份:N", sort=None, title="月份"),
    y=alt.Y("线上:Q", title="线上销售额（万元）"),
)
bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
    color=alt.Color("线上:Q", scale=alt.Scale(scheme="blues"), legend=None)
)
# Altair 用 + 号把多个图层叠加起来
labels = base.mark_text(dy=-8, fontWeight="bold").encode(text="线上:Q")
st.altair_chart((bars + labels).properties(height=320), width="stretch")

st.subheader("示例 2：带参考线的折线图 + 交互式缩放")
st.code(
    '''# 把宽表转成"长表"（Altair 的 color 分组更习惯长表格式）
long_df = sales_reset.melt(id_vars="月份", var_name="渠道", value_name="销售额")

line = (
    alt.Chart(long_df)
    .mark_line(point=True)                   # point=True 把每个数据点画出来
    .encode(
        x=alt.X("月份:N", sort=None),
        y=alt.Y("销售额:Q"),
        color=alt.Color("渠道:N", legend=alt.Legend(title="销售渠道")),
        tooltip=["月份", "渠道", "销售额"],   # ★ 自定义鼠标悬停提示
    )
)
# 平均值参考线：水平虚线
rule = (
    alt.Chart(long_df)
    .mark_rule(strokeDash=[6, 4], color="red")
    .encode(y="mean(销售额):Q")
)

st.altair_chart((line + rule).properties(height=340).interactive(), width="stretch")''',
    language="python",
)

long_df = sales_reset.melt(id_vars="月份", var_name="渠道", value_name="销售额")
line = (
    alt.Chart(long_df)
    .mark_line(point=True)
    .encode(
        x=alt.X("月份:N", sort=None),
        y=alt.Y("销售额:Q"),
        color=alt.Color("渠道:N", legend=alt.Legend(title="销售渠道")),
        tooltip=["月份", "渠道", "销售额"],
    )
)
rule = (
    alt.Chart(long_df)
    .mark_rule(strokeDash=[6, 4], color="red")
    .encode(y="mean(销售额):Q")
)
st.altair_chart((line + rule).properties(height=340).interactive(), width="stretch")

st.subheader("示例 3：直方图（内置图表做不到）")
st.code(
    '''hist = (
    alt.Chart(scatter_data)
    .mark_bar()
    .encode(
        x=alt.X("销售额:Q", bin=alt.Bin(maxbins=15), title="销售额区间"),
        y=alt.Y("count():Q", title="频次"),
    )
    .properties(height=300)
)
st.altair_chart(hist, width="stretch")''',
    language="python",
)
hist = (
    alt.Chart(scatter_data)
    .mark_bar()
    .encode(
        x=alt.X("销售额:Q", bin=alt.Bin(maxbins=15), title="销售额区间"),
        y=alt.Y("count():Q", title="频次"),
    )
    .properties(height=300)
)
st.altair_chart(hist, width="stretch")

st.subheader("示例 4：饼图 / 环形图")
st.code(
    '''pie_data = pd.DataFrame({
    "渠道": ["线上", "线下", "批发"],
    "销售额": [1230, 805, 480],
})

pie = (
    alt.Chart(pie_data)
    .mark_arc(innerRadius=60)                # innerRadius>0 就是环形图
    .encode(
        theta=alt.Theta("销售额:Q", stack=True),
        color=alt.Color("渠道:N"),
        tooltip=["渠道", "销售额"],
    )
    .properties(height=320)
)
st.altair_chart(pie, width="content")''',
    language="python",
)

pie_data = pd.DataFrame(
    {
        "渠道": ["线上", "线下", "批发"],
        "销售额": [1230, 805, 480],
    }
)
pie = (
    alt.Chart(pie_data)
    .mark_arc(innerRadius=60)
    .encode(
        theta=alt.Theta("销售额:Q", stack=True),
        color=alt.Color("渠道:N"),
        tooltip=["渠道", "销售额"],
    )
    .properties(height=320)
)
# 饼图不需要拉伸宽度，用 width="content" 保持它自己的尺寸
st.altair_chart(pie, width="content")

st.divider()

# ---------------------------------------------------------------------------
# 六、matplotlib：st.pyplot
# ---------------------------------------------------------------------------
st.header("六、matplotlib：st.pyplot(fig)")

st.markdown(
    """
**matplotlib** 是 Python 世界里最经典的绘图库，科研、统计、论文插图几乎都用它。

### 核心用法只有三步

```python
import matplotlib
matplotlib.use("Agg")              # ① 无窗口后端（必须在 import pyplot 之前！）

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 3.6))     # ② 创建「画布 + 坐标轴」
ax.plot(x, y, marker="o", label="线上")       #    在 ax 上画
ax.set_title("月度销售额")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)                     # ③ 把 fig 交给 Streamlit 渲染
plt.close(fig)                     #    用完关闭，释放内存
```

### ★ 四个必须知道的点

| 要点 | 说明 |
|---|---|
| **`matplotlib.use("Agg")`** | 服务器没有显示器，必须用"只出图、不弹窗"的 Agg 后端。**必须在 `import pyplot` 之前调用** |
| **不要 `plt.show()`** | `plt.show()` 是给本地窗口用的，在服务器上会卡住。**交给 `st.pyplot(fig)` 渲染** |
| **中文字体** | 默认字体没有中文字形，中文会变成方块。要设 `plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", ...]`，并把 `plt.rcParams["axes.unicode_minus"] = False`（否则负号也是方块） |
| **`fig` vs `plt`** | 现代写法是"面向对象"的：用 `fig, ax = plt.subplots()`，然后所有操作都在 `ax` 上做。`plt.plot(...)` 那种写法操作的是"当前图"，图一多就会搞混 |

### `st.pyplot` 的参数

| 参数 | 作用 |
|---|---|
| `fig` | `matplotlib.figure.Figure` 对象。**不传**的时候它渲染"当前活动的图"（不推荐，容易出错） |
| `clear_figure` | 渲染后是否清空该 fig，默认由 Streamlit 自动判断（通常会清）。**自己再 `plt.close(fig)` 更保险** |
| `width` | 宽度，默认 `"stretch"`（自适应容器宽度） |

> ⚠️ **内存提醒：**Streamlit 每次交互都重跑脚本。如果每次都 `plt.subplots()` 却不关闭 fig，
> 图表会不断累积在内存里（matplotlib 会警告 "More than 20 figures have been opened"）。
> **画完就 `plt.close(fig)`** 是个好习惯。
"""
)

st.code(
    '''import matplotlib
matplotlib.use("Agg")               # ★ 必须在 import pyplot 之前
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False      # 负号不要变方块

fig, ax = plt.subplots(figsize=(8, 3.6))
ax.plot(df["月份"], df["线上"], marker="o", linewidth=2, label="线上渠道")
ax.set_title("月度销售额")
ax.set_xlabel("月份")
ax.set_ylabel("销售额（万元）")
ax.grid(alpha=0.3)
ax.legend()

st.pyplot(fig)        # ← 交给 Streamlit 渲染（width 默认 "stretch"）
plt.close(fig)        # ← 释放内存''',
    language="python",
)

st.subheader("1）折线图（含网格、图例、数据点标记）")

fig_line, ax_line = plt.subplots(figsize=(9, 3.6))
# 索引是月份字符串，直接用列表画
months_list = sales_reset["月份"].tolist()
ax_line.plot(months_list, sales_reset["线上"], marker="o", linewidth=2,
             markersize=6, label=label("线上渠道", "Online"))
ax_line.plot(months_list, sales_reset["线下"], marker="s", linewidth=2,
             markersize=6, label=label("线下渠道", "Offline"))
ax_line.plot(months_list, sales_reset["批发"], marker="^", linewidth=2,
             markersize=6, label=label("批发渠道", "Wholesale"))
ax_line.set_title(label("各渠道月度销售额趋势", "Monthly sales by channel"))
ax_line.set_xlabel(label("月份", "Month"))
ax_line.set_ylabel(label("销售额（万元）", "Sales (10k CNY)"))
ax_line.grid(alpha=0.3)                 # 网格线：alpha 让网格淡一点，不抢主体
ax_line.legend()                        # 显示图例（用的是 plot 里的 label）
fig_line.tight_layout()                 # 自动调整子图间距，避免标签被裁掉
st.pyplot(fig_line)
plt.close(fig_line)                     # ★ 释放内存

st.subheader("2）柱状图（带数值标注，并高亮最大值）")

fig_bar, ax_bar = plt.subplots(figsize=(9, 3.6))
online = sales_reset["线上"].tolist()
bar_colors = [
    # 最大值那根柱子用强调色，其余用主题色
    "#ec4899" if value == max(online) else "#4f46e5"
    for value in online
]
bars = ax_bar.bar(months_list, online, color=bar_colors, width=0.6)
# 在每根柱子顶部标注数值
for rect, value in zip(bars, online):
    ax_bar.text(
        rect.get_x() + rect.get_width() / 2,     # x 坐标：柱子中心
        value + 4,                               # y 坐标：柱子顶部再高一点
        str(value),
        ha="center", va="bottom", fontsize=9,
    )
ax_bar.set_title(label("线上渠道月度销售额（最高值已高亮）", "Online sales (max highlighted)"))
ax_bar.set_xlabel(label("月份", "Month"))
ax_bar.set_ylabel(label("销售额（万元）", "Sales (10k CNY)"))
ax_bar.grid(axis="y", alpha=0.3)        # 只在 y 轴方向画网格线
fig_bar.tight_layout()
st.pyplot(fig_bar)
plt.close(fig_bar)

st.subheader("3）散点图（看两个变量的相关性与分组）")

fig_scatter, ax_scatter = plt.subplots(figsize=(9, 3.8))
# 按"地区"分组，每组一种颜色和标记，这样能看出"不同地区的关系是否不同"
region_styles = {
    "华北": {"color": "#4f46e5", "marker": "o"},
    "华东": {"color": "#16a34a", "marker": "s"},
    "华南": {"color": "#ec4899", "marker": "^"},
}
for region, style in region_styles.items():
    subset = scatter_data[scatter_data["地区"] == region]
    ax_scatter.scatter(
        subset["广告投入"],
        subset["销售额"],
        label=label(f"{region}地区", region),
        alpha=0.75,             # 半透明：点重叠时能看出密度
        s=45,                   # 点的大小
        **style,
    )
# 画一条"趋势线"（一次多项式拟合），说明怎么做线性拟合
coef = np.polyfit(scatter_data["广告投入"], scatter_data["销售额"], 1)
x_line = np.linspace(scatter_data["广告投入"].min(), scatter_data["广告投入"].max(), 50)
ax_scatter.plot(x_line, np.polyval(coef, x_line),
                color="#111827", linestyle="--", linewidth=1.5,
                label=label("线性拟合", "Linear fit"))
ax_scatter.set_title(label("广告投入 vs 销售额", "Ad spend vs Sales"))
ax_scatter.set_xlabel(label("广告投入（万元）", "Ad spend"))
ax_scatter.set_ylabel(label("销售额（万元）", "Sales"))
ax_scatter.grid(alpha=0.3)
ax_scatter.legend()
fig_scatter.tight_layout()
st.pyplot(fig_scatter)
plt.close(fig_scatter)

st.subheader("4）子图（subplots）：一张画布放多个图")

st.markdown(
    """
这是 matplotlib **最有价值、也是内置图表和 Altair 做不到的能力**：
用 `plt.subplots(2, 2)` 一次得到 2×2 共 4 个坐标轴，
然后分别往每个 `ax` 上画 —— 一张图里做多视角对比，是数据分析报告的标准版式。
"""
)

st.code(
    '''# plt.subplots(行数, 列数) 返回 (fig, axes)，axes 是一个二维数组
fig, axes = plt.subplots(2, 2, figsize=(10, 6))

axes[0, 0].plot(months, online);        axes[0, 0].set_title("折线")
axes[0, 1].bar(months, online);         axes[0, 1].set_title("柱状")
axes[1, 0].scatter(x, y, alpha=.7);     axes[1, 0].set_title("散点")
axes[1, 1].pie(values, labels=labels);  axes[1, 1].set_title("饼图")

fig.tight_layout()          # ★ 一定要调，否则子图标题会互相重叠
st.pyplot(fig)
plt.close(fig)''',
    language="python",
)

fig_grid, axes = plt.subplots(2, 2, figsize=(10.5, 6.2))

# 左上：折线
axes[0, 0].plot(months_list, sales_reset["线上"], marker="o", color="#4f46e5")
axes[0, 0].set_title(label("折线：趋势", "Line: trend"), fontsize=11)
axes[0, 0].grid(alpha=0.3)

# 右上：柱状
axes[0, 1].bar(months_list, sales_reset["线下"], color="#16a34a")
axes[0, 1].set_title(label("柱状：类别对比", "Bar: comparison"), fontsize=11)
axes[0, 1].grid(axis="y", alpha=0.3)

# 左下：散点
axes[1, 0].scatter(scatter_data["广告投入"], scatter_data["销售额"],
                   alpha=0.55, s=25, color="#ec4899")
axes[1, 0].set_title(label("散点：相关性", "Scatter: correlation"), fontsize=11)
axes[1, 0].grid(alpha=0.3)

# 右下：饼图（matplotlib 的饼图只需一行 pie）
channel_values = [int(sales_reset["线上"].sum()),
                  int(sales_reset["线下"].sum()),
                  int(sales_reset["批发"].sum())]
axes[1, 1].pie(
    channel_values,
    labels=[label("线上", "Online"), label("线下", "Offline"), label("批发", "Wholesale")],
    autopct="%1.1f%%",                       # 显示百分比，保留 1 位小数
    colors=["#4f46e5", "#16a34a", "#f59e0b"],
    startangle=90,
)
axes[1, 1].set_title(label("饼图：构成", "Pie: composition"), fontsize=11)

fig_grid.tight_layout()          # ★ 子图必须调，否则标题会互相重叠
st.pyplot(fig_grid)
plt.close(fig_grid)

st.subheader("5）三套方案的取舍")

st.markdown(
    """
| | Streamlit 内置图表 | Altair | **matplotlib** |
|---|---|---|---|
| 代码量 | **最少**（一行） | 中 | 中偏多 |
| 交互性 | ✅ 悬停提示、图例筛选 | ✅ **最强**（缩放/框选/自定义 tooltip） | ❌ **静态图片** |
| 精细控制 | ❌ 几乎没有 | 中（声明式） | ✅ **最强**（子图、双轴、误差棒、注释、样式） |
| 子图 / 多面板 | ❌ | ⚠️ 需要 `concat`/`hconcat` | ✅ **`plt.subplots(2,2)` 原生支持** |
| 统计图 | ❌ | 部分（直方图/箱线图要手写） | ✅ **箱线图、小提琴图、热力图、误差棒** |
| 中文支持 | ✅ 自动 | ✅ 自动 | ⚠️ **要手动设字体** |
| 适合场景 | 快速看数据形状 | 网页里的交互式图表 | **科研插图、统计报告、复杂版式** |
| Streamlit 组件 | `st.line_chart` 等 | `st.altair_chart` | **`st.pyplot`** |

**一句话选择：**
「只想快速看一眼」→ 内置图表；「要给用户交互」→ Altair；
「要精细控制 / 出统计图 / 做子图」→ **matplotlib**。
"""
)

st.divider()

# ---------------------------------------------------------------------------
# 七、补充说明：其它图表库 & st.map
# ---------------------------------------------------------------------------
st.header("七、补充说明")

st.subheader("1）其它图表库怎么用")
st.markdown(
    """
| 库 | Streamlit 组件 | 说明 |
|---|---|---|
| **Streamlit 内置** | `st.line_chart` / `area_chart` / `bar_chart` / `scatter_chart` | 底层是 Altair，最省事 |
| **Altair** | `st.altair_chart(chart)` | 声明式，交互最强，**推荐** |
| **matplotlib** | `st.pyplot(fig)` | **本环境可用（3.11.1）**，静态图片，精细控制最强 |
| plotly | `st.plotly_chart(fig)` | 交互性也很强（缩放、框选、3D）；**本环境未安装**，只用注释说明 |
| Bokeh | `st.bokeh_chart(fig)` | 需要额外安装 bokeh |
| Graphviz | `st.graphviz_chart(dot)` | 画流程图 / 关系图 |

**完整代码见本节「六、matplotlib：st.pyplot(fig)」**，那里有折线、柱状、散点、子图四个可运行示例。

**如果以后要用 plotly**，写法是这样的（本环境没有装 plotly，所以这里只给代码）：
"""
)
st.code(
    '''import plotly.express as px

fig = px.scatter(df, x="广告投入", y="销售额", color="地区",
                 title="广告投入 vs 销售额")
st.plotly_chart(fig, width="stretch")     # ← 注意是 st.plotly_chart''',
    language="python",
)

st.subheader("2）st.map 地图（需要联网，这里只讲用法）")
st.code(
    '''# DataFrame 里要有 lat / lon 两列（或者用 latitude / longitude 参数指定列名）
map_df = pd.DataFrame({
    "lat": [39.9042, 31.2304, 23.1291, 22.5431],
    "lon": [116.4074, 121.4737, 113.2644, 114.0579],
    "城市": ["北京", "上海", "广州", "深圳"],
})

st.map(map_df)                    # 打点
st.map(map_df, size=200)          # 点的大小
st.map(map_df, color="#ff0000")   # 点的颜色''',
    language="python",
)
st.caption(
    "⚠️ 本环境不保证联网，所以这里没有真的渲染地图 —— "
    "`st.map` 需要从网上加载地图瓦片，离线时会显示空白。"
)

st.divider()

# ---------------------------------------------------------------------------
# 课案原文的完整示例
# ---------------------------------------------------------------------------
st.header("课案原文的完整示例")
st.code(
    '''import streamlit as st
import pandas as pd
import numpy as np

# 生成示例数据
chart_data = pd.DataFrame(
    np.random.randn(50, 3),
    columns=["A列", "B列", "C列"]
)

st.subheader("折线图")
st.line_chart(chart_data)

st.subheader("面积图")
st.area_chart(chart_data)

st.subheader("柱状图")
st.bar_chart(chart_data)

st.subheader("散点图")
st.scatter_chart(chart_data)''',
    language="python",
)

st.success(
    "**本节要点回顾：**\n"
    "1. 四种内置图表：`line_chart`（趋势）/ `area_chart`（累积与构成）/ "
    "`bar_chart`（类别比较）/ `scatter_chart`（两变量关系）。\n"
    "2. 它们默认拿 **DataFrame 的 index 当 x 轴** —— x 轴不对时，"
    "要么 `set_index`，要么显式传 `x=` / `y=`。\n"
    "3. 需要交互（悬停提示、缩放、多图层、直方图、饼图）时用 `st.altair_chart`。\n"
    "4. 需要精细控制（子图、误差棒、统计图、科研风格）时用 **`st.pyplot(fig)`**；"
    "记得 `matplotlib.use(\"Agg\")` 要写在 `import pyplot` 之前、"
    "要手动设置中文字体、**不要调用 `plt.show()`**、画完要 `plt.close(fig)`。\n"
    "5. `width=\"stretch\"` 让图表自适应宽度（旧的 `use_container_width=True` 已弃用）。"
)

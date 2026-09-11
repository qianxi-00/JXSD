"""
=====================================================================================
文件：10_侧边栏.py
对应课案章节：Streamlit → 侧边栏
本节知识点：
  1. 侧边栏是页面左侧的导航 / 控制区域，所有组件都可以放到侧边栏里。
  2. 两种写法：`with st.sidebar:` 块 与 `st.sidebar.xxx()` 直接调用。
  3. 课案原文示例的完整复现：侧边栏导航 + 版权信息 + 文件上传。
  4. 侧边栏的定位思想：**侧边栏放"控制条件"，主区域放"结果"**。
  5. 侧边栏不能嵌套（不能在里面再开一个 sidebar）。
  6. 侧边栏的显示/隐藏由用户控制，`st.set_page_config(initial_sidebar_state=...)`
     只决定初始状态。
  7. 多页面应用里的导航也默认出现在侧边栏（见 14 / 16 节）。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m streamlit run `
        'F:\\ProGram\\Python_Base\\Front_End\\04_Streamlit\\10_侧边栏.py' `
        --server.headless true --server.port 8610 --browser.gatherUsageStats false
=====================================================================================
"""

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="10 侧边栏",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",   # 侧边栏默认展开
)

# =============================================================================
# 一、两种写法
# =============================================================================
# 写法 A：用 with st.sidebar: 块（推荐，缩进一眼看出"这些都在侧边栏里"）
# 写法 B：直接 st.sidebar.xxx(...)（少一层缩进，但一长串写下来可读性差）
# 两种完全等价，可以混用。
# =============================================================================

# ---------------------------------------------------------------------------
# 侧边栏：导航 + 控制条件（课案原文的结构）
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("导航菜单")
    st.markdown("---")

    # st.radio 放在侧边栏里，就变成"导航菜单"
    menu = st.radio(
        "选择页面",
        ["首页", "数据总览", "用户管理", "系统设置"],
        key="sidebar_menu",
    )

    st.markdown("---")

    # 侧边栏底部放一些固定信息（版本号、当前用户等）
    st.caption("当前用户：admin")
    st.caption("版本：v1.0.0")

    # 侧边栏里也可以放文件上传、滑块、按钮……任何组件都可以
    uploaded = st.file_uploader("上传数据（CSV）", type=["csv"], key="sidebar_upload")

    st.markdown("---")
    st.caption("提示：点左上角的 « 可以收起侧边栏。")

# =============================================================================
# 主页面：根据侧边栏的选择显示不同内容
# =============================================================================
st.title("10 侧边栏")
st.caption("对应课案：Streamlit → 侧边栏")

st.info(
    "**本页的导航就在左边的侧边栏里** —— 点「选择页面」切换，"
    "下面的内容会跟着变。这就是课案原文示例的完整复现。"
)

st.divider()

# ---------------------------------------------------------------------------
# 课案原文的代码
# ---------------------------------------------------------------------------
st.header("课案原文的代码")
st.code(
    '''import streamlit as st

# 侧边栏中放置组件
with st.sidebar:
    st.title("导航菜单")
    st.markdown("---")

    menu = st.radio(
        "选择页面",
        ["首页", "数据总览", "用户管理", "系统设置"]
    )

    st.markdown("---")
    st.caption(f"当前用户：admin")
    st.caption("版本：v1.0.0")

    uploaded = st.file_uploader("上传数据", type=["csv"])

# 主页面根据侧边栏选择显示不同内容
st.title(f"{menu}")

if menu == "首页":
    st.write("欢迎来到首页")
elif menu == "数据总览":
    st.line_chart({"销售额": [100, 200, 150, 300, 250]})
elif menu == "用户管理":
    st.dataframe({"姓名": ["张三", "李四"], "角色": ["管理员", "普通用户"]})
elif menu == "系统设置":
    st.slider("参数配置", 0, 100, 50)  # 范围、默认值''',
    language="python",
)

st.divider()

# ---------------------------------------------------------------------------
# 主区域：根据侧边栏的选择渲染不同内容（就是课案原文的 if/elif 分支）
# ---------------------------------------------------------------------------
st.header(f"当前页面：{menu}")

if menu == "首页":
    st.write("欢迎来到首页")
    st.markdown(
        """
这里是首页。左边侧边栏里的 `st.radio` 返回值就是 `menu` 变量，
主区域用 `if / elif` 根据它的值渲染不同内容 ——
**这就是最朴素的"单文件多页面"实现方式**（完整版见 `16_多页面_侧边栏方式.py`）。
"""
    )
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("用户总数", "1,280", "5.2%")
    col_b.metric("本月订单", "3,452", "12.8%")
    col_c.metric("系统负载", "42%", "-3.1%")

elif menu == "数据总览":
    st.write("这是数据总览页面。")
    # 课案原文用的是 {"销售额": [...]}，直接传字典给 st.line_chart 也能用
    st.line_chart({"销售额": [100, 200, 150, 300, 250]})

    st.markdown("**本页的销售额数据：**")
    st.dataframe(
        pd.DataFrame(
            {
                "月份": ["1月", "2月", "3月", "4月", "5月"],
                "销售额": [100, 200, 150, 300, 250],
            }
        ).set_index("月份"),
        width="stretch",
    )

elif menu == "用户管理":
    st.write("这是用户管理页面。")
    # 课案原文用的是 {"姓名": [...], "角色": [...]}
    st.dataframe(
        pd.DataFrame({"姓名": ["张三", "李四"], "角色": ["管理员", "普通用户"]}),
        width="stretch",
    )
    st.caption("st.dataframe 可以直接吃字典 —— 它会自动转成 DataFrame。")

elif menu == "系统设置":
    st.write("这是系统设置页面。")
    st.slider("参数配置", 0, 100, 50)   # 三个参数：标签、最小值、最大值、默认值

# ---------------------------------------------------------------------------
# 侧边栏上传的文件：在主区域处理
# ---------------------------------------------------------------------------
if uploaded is not None:
    st.divider()
    st.header("你在侧边栏上传的文件")
    st.success(f"已收到：{uploaded.name}（{uploaded.size:,} 字节）")
    try:
        uploaded_df = pd.read_csv(uploaded)
        st.dataframe(uploaded_df.head(20), width="stretch")
        st.caption(f"共 {len(uploaded_df)} 行 × {len(uploaded_df.columns)} 列")
    except Exception as exc:      # noqa: BLE001 —— 读文件必须兜底
        st.error(f"这个 CSV 读不出来：{exc}")

st.divider()

# ---------------------------------------------------------------------------
# 详解
# ---------------------------------------------------------------------------
st.header("详解：侧边栏的几个要点")

st.subheader("1）两种写法完全等价")
st.code(
    '''# 写法 A：with 块（推荐）
with st.sidebar:
    st.title("标题")
    st.slider("参数", 0, 100, 50)
    st.button("按钮")

# 写法 B：直接调用
st.sidebar.title("标题")
st.sidebar.slider("参数", 0, 100, 50)
st.sidebar.button("按钮")''',
    language="python",
)

st.subheader("2）任何组件都能放进侧边栏")
st.markdown(
    """
侧边栏**不是一种特殊的容器**，它和主区域一样可以放任何组件：
文本、图表、表格、输入、文件上传、甚至 `st.columns` 和表单。

唯一的限制：**侧边栏里不能再开一个侧边栏**（`st.sidebar.sidebar` 不存在）。
"""
)

st.subheader("3）侧边栏的定位：控制条件 vs 结果")
st.markdown(
    """
这是使用侧边栏最重要的一条经验：

| 放哪里 | 放什么 |
|---|---|
| **侧边栏** | 导航菜单、筛选条件（日期范围、地区、类别）、全局参数、用户信息 |
| **主区域** | 数据表格、图表、结果、详细内容 |

**为什么？**因为侧边栏的内容**一直可见**，用户改了条件马上就能看到主区域的变化。
如果筛选条件放在主区域里，用户往下滚看结果时就看不到筛选器了。
"""
)

st.subheader("4）改一改：看看侧边栏作为「筛选器」的样子")
st.code(
    '''with st.sidebar:
    st.header("筛选条件")
    date_range = st.date_input("日期范围", value=(start, end))
    regions = st.multiselect("地区", ["华北", "华东", "华南"], default=["华北"])
    min_amount = st.slider("金额下限", 0, 10000, 1000)
    only_vip = st.checkbox("只看 VIP 用户")

# 主区域用这些条件过滤数据
filtered = df[
    df["地区"].isin(regions)
    & (df["金额"] >= min_amount)
    & (df["日期"].between(*date_range))
]
st.dataframe(filtered)''',
    language="python",
)

# 真做一个侧边栏筛选器，作用在主区域的表格上
filter_placeholder = st.container(border=True)
filter_placeholder.markdown("**下面的表格受「侧边栏的筛选条件」控制：**")

# 这份数据放在主区域，但筛选条件从侧边栏读
sample_df = pd.DataFrame(
    {
        "城市": ["北京", "上海", "广州", "深圳", "北京", "上海", "广州", "深圳"],
        "类别": ["电子产品", "服装", "电子产品", "食品", "服装", "电子产品", "食品", "服装"],
        "金额": [3200, 800, 5600, 300, 1200, 4800, 450, 2100],
    }
)

with st.sidebar:
    st.markdown("---")
    st.subheader("🔍 筛选条件（作用在右表）")
    picked_cities = st.multiselect(
        "城市",
        ["北京", "上海", "广州", "深圳"],
        default=["北京", "上海", "广州", "深圳"],
        key="filter_cities",
    )
    picked_categories = st.multiselect(
        "类别",
        ["电子产品", "服装", "食品"],
        default=["电子产品", "服装", "食品"],
        key="filter_categories",
    )
    min_amount_filter = st.slider("金额下限", 0, 6000, 0, step=100, key="filter_amount")

# 按侧边栏的条件过滤（注意：multiselect 返回列表，用 isin 判断）
# 还要处理"一个都没选"的情况 —— 那时应该显示空表，而不是报错
filtered_df = sample_df[
    sample_df["城市"].isin(picked_cities)
    & sample_df["类别"].isin(picked_categories)
    & (sample_df["金额"] >= min_amount_filter)
]

filter_placeholder.dataframe(filtered_df, width="stretch", hide_index=True)
st.caption(
    f"原始数据 {len(sample_df)} 行 → 筛选后 {len(filtered_df)} 行"
    "（去左边的侧边栏改一改筛选条件试试）"
)

st.subheader("5）侧边栏的宽度能改吗？")
st.markdown(
    """
**不能直接在 Python 里改**（`st.sidebar` 没有 `width` 参数）。
用户可以用鼠标**拖动侧边栏右边缘**来调整宽度。

如果想改默认宽度，只能写 CSS（属于"自定义样式"的范畴）：

```python
st.markdown(
    \"\"\"<style>
    [data-testid="stSidebar"] { min-width: 320px; max-width: 320px; }
    </style>\"\"\",
    unsafe_allow_html=True,
)
```

⚠️ 这种写法依赖 Streamlit 内部的 `data-testid`，**版本升级可能会失效**，慎用。
"""
)

st.divider()

# ---------------------------------------------------------------------------
# 课案原文的完整示例（原样再贴一遍，方便对照）
# ---------------------------------------------------------------------------
st.header("课案原文的完整示例")
st.code(
    '''import streamlit as st

# 侧边栏中放置组件
with st.sidebar:
    st.title("导航菜单")
    st.markdown("---")

    menu = st.radio(
        "选择页面",
        ["首页", "数据总览", "用户管理", "系统设置"]
    )

    st.markdown("---")
    st.caption(f"当前用户：admin")
    st.caption("版本：v1.0.0")

    uploaded = st.file_uploader("上传数据", type=["csv"])

# 主页面根据侧边栏选择显示不同内容
st.title(f"{menu}")

if menu == "首页":
    st.write("欢迎来到首页")
elif menu == "数据总览":
    st.line_chart({"销售额": [100, 200, 150, 300, 250]})
elif menu == "用户管理":
    st.dataframe({"姓名": ["张三", "李四"], "角色": ["管理员", "普通用户"]})
elif menu == "系统设置":
    st.slider("参数配置", 0, 100, 50)  # 范围、默认值''',
    language="python",
)

st.success(
    "**本节要点回顾：**\n"
    "1. 两种写法等价：`with st.sidebar:`（推荐）与 `st.sidebar.xxx()`。\n"
    "2. 任何组件都能放进侧边栏，但**侧边栏不能嵌套**。\n"
    "3. 侧边栏放**控制条件**（导航、筛选、参数），主区域放**结果**。\n"
    "4. 侧边栏的初始状态由 `st.set_page_config(initial_sidebar_state=...)` 决定，"
    "之后由用户自己控制。\n"
    "5. 多页面应用的导航也默认渲染在侧边栏里（第 14 / 16 节）。"
)

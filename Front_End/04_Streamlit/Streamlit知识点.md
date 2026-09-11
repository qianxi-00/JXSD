# Streamlit 知识点总结

> 对应课案章节：**Streamlit（安装与运行 → 页面配置 → 文本 → 数据 → 图表 → 输入 → 媒体 → 布局 → 状态与提示 → 侧边栏 → 表单 → 会话状态 → 缓存机制 → 多页面应用 → 综合实战：图书管理系统）**
>
> **运行环境约定（本项目）：**必须使用工作区自带的虚拟环境，**不要**用 `uv run`，也不要安装任何东西。
>
> ```powershell
> & 'F:\ProGram\Python_Base\.venv\Scripts\python.exe' -m streamlit run `
>     '<文件路径>' --server.headless true --server.port 86xx --browser.gatherUsageStats false
> ```
>
> 端口分配：`01`→8601、`02`→8602 …… `16`→8616，`15_图书管理系统/app.py`→8615。

---

## 0. 文件索引（课案小节 → 文件）

| 课案小节 | 文件 | 运行端口 |
|---|---|---|
| 安装与运行 | `01_安装与运行.py` | 8601 |
| 页面配置 | `02_页面配置.py` | 8602 |
| 文本 | `03_文本.py` | 8603 |
| 数据 | `04_数据.py` | 8604 |
| 图表 | `05_图表.py` | 8605 |
| 输入 | `06_输入.py` | 8606 |
| 媒体 | `07_媒体.py` | 8607 |
| 布局 | `08_布局.py` | 8608 |
| 状态与提示 | `09_状态与提示.py` | 8609 |
| 侧边栏 | `10_侧边栏.py` | 8610 |
| 表单 | `11_表单.py` | 8611 |
| 会话状态 | `12_会话状态.py` | 8612 |
| 缓存机制 | `13_缓存机制.py` | 8613 |
| 多页面应用（方式 1） | `14_多页面应用/app.py` + `pages/1_数据总览.py`、`2_用户管理.py`、`3_系统设置.py` | 8614 |
| 多页面应用（方式 2） | `16_多页面_侧边栏方式.py` | 8616 |
| 综合实战：图书管理系统 | `15_图书管理系统/app.py`（+ `admin.py` / `user.py` / `data.py` / `README.md`） | 8615 |

---

## 1. Streamlit 是什么

**用纯 Python 写网页，不需要写 HTML / CSS / JS。**

| | 传统前端 | Streamlit |
|---|---|---|
| 写什么 | HTML + CSS + JS | **只写 Python** |
| 渲染 | 浏览器执行 JS | 服务端生成 + 内建前端渲染 |
| 适合 | 面向公众的产品站点 | **数据应用、内部工具、原型** |

> **课案的结论：**"从数据展示到完整后台管理系统，都可以快速实现。
> 如果要做更精美的商业网站或复杂交互还是需要传统前端，
> 但对于数据应用和内部工具，Streamlit 是最快的方式。"

---

## 2. 安装与运行

```bash
pip install streamlit        # 安装（本项目已装好，不要重复安装）
streamlit run app.py         # ★ 运行：不是 python app.py
```

**为什么不能 `python app.py`？**因为 Streamlit 应用需要一个常驻 Web 服务，
`python app.py` 只是把脚本跑一遍就退出了。

### 常用启动参数

| 参数 | 作用 |
|---|---|
| `--server.port 8601` | 指定端口（默认 8501） |
| `--server.headless true` | 无头模式，不尝试自动打开浏览器 |
| `--server.runOnSave true` | 保存后自动重载（开发时方便） |
| `--logger.level debug` | 提高日志级别 |
| `--client.showErrorDetails true` | 浏览器里显示完整错误堆栈 |
| `--browser.gatherUsageStats false` | 关闭使用数据上报 |
| `--server.address 0.0.0.0` | 允许局域网访问 |

### ★★★ 运行模型：每次交互都重新执行整个脚本 ★★★

**这是理解 Streamlit 的第一前提。** 用户每一次操作（点按钮、拖滑块、输入）
都会让 Streamlit **从上到下重新执行整个脚本**，然后重新渲染页面。

三个直接推论：

1. **脚本就是页面的"渲染函数"** —— 写什么顺序，页面就是什么顺序。
2. **脚本局部变量每次都会被重置** → 要跨交互保留数据，必须用 `st.session_state`。
3. **重跑是廉价的**，但耗时计算（查数据库、训练模型）必须用
   `@st.cache_data` / `@st.cache_resource` 缓存。

---

## 3. 页面配置 `st.set_page_config`

```python
import streamlit as st

st.set_page_config(
    page_title="我的应用",              # 浏览器标签页标题
    page_icon="🖥️",                    # 浏览器标签页图标（emoji / 图片路径 / URL）
    layout="wide",                     # "centered"（默认，约 730px）/ "wide"（铺满）
    initial_sidebar_state="expanded",  # "expanded" / "collapsed" / "auto"
    menu_items={                       # 可选：自定义右上角菜单
        "Get Help": "https://docs.streamlit.io",
        "Report a bug": None,
        "About": "说明文字",
    },
)

st.title("页面内容从这里开始")
```

> ⚠️ **铁律：`set_page_config` 必须是脚本里第一个 Streamlit 调用，且整个脚本只能调用一次。**
> 违反任意一条都会抛 `StreamlitAPIException`。
> **多页面应用只在入口脚本（`app.py`）里调用一次，子页面绝对不要调用。**

| 参数 | 可选值 | 说明 |
|---|---|---|
| `page_title` | 任意字符串 | 浏览器标签页标题（不写则默认是脚本文件名） |
| `page_icon` | emoji / 图片路径 / URL | 标签页图标 |
| `layout` | `"centered"` / `"wide"` | 内容居中 / 铺满宽屏 |
| `initial_sidebar_state` | `"expanded"` / `"collapsed"` / `"auto"` | 侧边栏初始状态 |

### 主题在配置文件里改

`.streamlit/config.toml`：

```toml
[theme]
base = "light"                 # "light" / "dark"
primaryColor = "#4f46e5"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#1f2937"
font = "sans serif"

[server]
port = 8601
headless = true

[browser]
gatherUsageStats = false
```

---

## 4. 文本组件

| 组件 | 代码 | 作用 |
|---|---|---|
| 页面标题 | `st.title("文本")` | 最大号标题，**每页只用一次** |
| 二级标题 | `st.header("文本")` | 中等标题 |
| 三级标题 | `st.subheader("文本")` | 较小标题 |
| 通用文本 | `st.write("文本")` | **万能显示**：文字/数字/列表/字典/DataFrame/图表 |
| Markdown | `st.markdown("**粗体** *斜体*")` | 渲染 Markdown（`unsafe_allow_html=True` 可直接写 HTML） |
| 代码块 | `st.code(code, language="python")` | 带语法高亮的代码（右上角有复制按钮） |
| 公式 | `st.latex(r"E = mc^2")` | LaTeX 数学公式（**字符串前加 `r`**） |
| 说明文字 | `st.caption("小字说明")` | 灰色小字 |
| 纯文本 | `st.text("纯文本")` | **不解析 Markdown** |
| 分割线 | `st.divider()` | 一条水平分割线 |

> **`st.write` 的智能之处：**传 `str` 渲染 Markdown、传 `DataFrame` 渲染交互表格、
> 传 `Series` 渲染折线图、传函数/类显示源码。**不确定用哪个组件时先用它。**
> 它还可以一次传多个参数，会横向排成一行：`st.write("总数：", 42, "条")`。

---

## 5. 数据组件

| 组件 | 代码 | 作用 |
|---|---|---|
| 交互表格 | `st.dataframe(df)` | 可排序 / 可筛选 / 可搜索 / 可下载 CSV |
| 静态表格 | `st.table(df)` | 一次性渲染全部数据，**没有滚动** |
| 指标卡 | `st.metric(label, value, delta=...)` | 显示指标和变化幅度 |
| JSON | `st.json(obj)` | 格式化显示 JSON |

> **默认用 `st.dataframe`；`st.table` 只适合 3~5 行的小表**（数据一多会非常卡）。

### 常用参数

| 参数 | 作用 |
|---|---|
| `width="stretch"` | 占满容器宽度（**推荐**；旧的 `use_container_width=True` 已弃用） |
| `height=320` | 固定高度，超出滚动 |
| `hide_index=True` | 隐藏左侧行号列 |
| `column_order=["城市", "销售额"]` | 指定列顺序 |
| `column_config={...}` | 列格式配置 |
| `on_select="rerun"` + `selection_mode="multi-row"` | 支持"选中行" |

### `st.column_config` 常用列类型

```python
st.dataframe(
    df,
    column_config={
        "销售额": st.column_config.NumberColumn("销售额", format="¥%d"),
        "进度":   st.column_config.ProgressColumn("进度", min_value=0, max_value=100),
        "网址":   st.column_config.LinkColumn("网址"),
        "完成":   st.column_config.CheckboxColumn("完成"),
        "日期":   st.column_config.DateColumn("日期", format="YYYY-MM-DD"),
        "城市":   st.column_config.SelectboxColumn("城市", options=["北京", "上海"]),
    },
    hide_index=True,
    width="stretch",
)
```

### `st.metric` 的 `delta`

- 默认 `delta_color="normal"`：**正数绿色 ↑，负数红色 ↓**
- `delta_color="inverse"`：反色（正红负绿）
- `delta_color="off"`：不上色，只显示箭头和数值

### 下载 CSV 的坑

```python
csv = df.to_csv(index=False).encode("utf-8-sig")   # ★ utf-8-sig 让 Excel 不乱码
st.download_button("下载 CSV", data=csv, file_name="数据.csv", mime="text/csv")
```

### `Styler`：条件格式（给表格上色）

`st.dataframe` 支持 pandas 的 `Styler` 对象。它只影响**显示**，不改底层数据。

```python
# ① 渐变色 —— 按数值大小映射颜色（需要 matplotlib 提供色带）
big_df.style.background_gradient(subset=["销售额"], cmap="Greens")

# ② 单元格内迷你条形图（同样需要 matplotlib）
big_df.style.bar(subset=["销售额"], color="#4f46e5")

# ③ 高亮最大 / 最小（不需要 matplotlib）
big_df.style.highlight_max(subset=["销售额"], color="#bbf7d0")

# ④ 自定义规则（func 返回一段 CSS，不需要 matplotlib）
def mark_high(v):
    return "background-color: #fecaca; font-weight: bold" if v > 15000 else ""
big_df.style.map(mark_high, subset=["销售额"])

# ⑤ 可以链式组合
(big_df.style
    .background_gradient(subset=["客单价"], cmap="Blues")
    .format({"销售额": "¥{:,}", "客单价": "{:.2f}"}))
```

| 方法 | 需要 matplotlib | 作用 |
|---|---|---|
| `background_gradient` / `bar` | ✅ | 渐变色 / 迷你条形图 |
| `highlight_max` / `highlight_min` / `highlight_null` | ❌ | 高亮极值 / 缺失值 |
| `format` | ❌ | 数字 / 日期显示格式 |
| `map` / `apply` | ❌ | 自定义规则 |

> **实践提醒：**数据量很大（几万行以上）时逐格算样式会明显变慢 ——
> 这种情况改用 `st.column_config.ProgressColumn` / `BarChartColumn` 更划算（由前端渲染）。

---

## 6. 图表

### 四种内置图表

| 组件 | 代码 | 适合 |
|---|---|---|
| 折线图 | `st.line_chart(data)` | 数据随**时间**变化 |
| 面积图 | `st.area_chart(data)` | 累积量 / 构成（默认堆叠） |
| 柱状图 | `st.bar_chart(data)` | **类别之间**比较 |
| 散点图 | `st.scatter_chart(data)` | 两个**数值**变量的关系 |

```python
st.line_chart(df)                                  # 索引当 x 轴，每列一条线
st.line_chart(df, x="月份", y=["线上", "线下"])      # 显式指定
st.area_chart(df, stack=False)                     # 不堆叠
st.bar_chart(df, horizontal=True)                  # 横向条形
st.scatter_chart(df, x="广告", y="销售额", color="地区", size="销售额")
```

> ⚠️ **最常见的坑：x 轴不对。**`st.line_chart` 默认拿 **DataFrame 的 index** 当 x 轴。
> 日期/月份是普通列时要 `.set_index("日期")`，或者显式传 `x=`。

### Altair（能力更强）

```python
import altair as alt

line = (
    alt.Chart(long_df)
    .mark_line(point=True)                        # 图形：bar/line/point/area/arc/rule/text
    .encode(
        x=alt.X("月份:N", sort=None),             # N 名义 / Q 定量 / T 时间 / O 有序
        y=alt.Y("销售额:Q"),
        color=alt.Color("渠道:N"),
        tooltip=["月份", "渠道", "销售额"],         # 自定义悬停提示
    )
    .properties(height=340)
    .interactive()                                 # 允许缩放/平移
)
st.altair_chart(line, width="stretch")
```

内置图表做不到的：多图层叠加（用 `+`）、参考线、直方图（`bin=`）、饼图（`mark_arc`）。

### matplotlib：`st.pyplot(fig)`

matplotlib 是 Python 最经典的绘图库，**科研 / 统计 / 论文插图**几乎都用它。
它和 Altair 最大的区别是：**输出是静态图片，但精细控制能力最强**。

```python
import matplotlib

matplotlib.use("Agg")          # ① 无窗口后端 —— ★ 必须在 import pyplot 之前！
import matplotlib.pyplot as plt

# ② 中文字体（不设的话中文会变成小方块）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False      # 负号也别变方块

fig, ax = plt.subplots(figsize=(9, 3.6))        # ③ 创建「画布 + 坐标轴」
ax.plot(x, y, marker="o", label="线上渠道")
ax.bar(x, y, color="#4f46e5")
ax.scatter(x, y, alpha=0.7)
ax.set_title("月度销售额"); ax.set_xlabel("月份"); ax.set_ylabel("万元")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()                              # 自动调整间距，避免标签被裁掉

st.pyplot(fig)                 # ④ 交给 Streamlit 渲染（width 默认 "stretch"）
plt.close(fig)                 # ⑤ ★ 释放内存，否则 fig 会不断累积
```

**★ 四个必须知道的点：**

| 要点 | 说明 |
|---|---|
| `matplotlib.use("Agg")` | 服务器没有显示器，必须用"只出图、不弹窗"的 Agg 后端。**必须在 `import pyplot` 之前调用** |
| **不要 `plt.show()`** | 它是给本地窗口用的，在服务器上会卡住。交给 `st.pyplot(fig)` |
| **中文字体** | 默认字体没有中文字形。要设 `rcParams["font.sans-serif"]` + `axes.unicode_minus = False` |
| **`plt.close(fig)`** | Streamlit 每次交互都重跑，不关 fig 会累积到内存里（matplotlib 会警告 "More than 20 figures"） |

**`st.pyplot` 参数：**`fig`（Figure 对象）/ `clear_figure` / `width`（默认 `"stretch"`）。

**matplotlib 最强的地方：子图（subplots）**

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 6))    # 2×2 四个坐标轴
axes[0, 0].plot(...);        axes[0, 0].set_title("折线")
axes[0, 1].bar(...);         axes[0, 1].set_title("柱状")
axes[1, 0].scatter(...);     axes[1, 0].set_title("散点")
axes[1, 1].pie(...);         axes[1, 1].set_title("饼图")
fig.tight_layout()                                  # ★ 子图必须调
st.pyplot(fig); plt.close(fig)
```

### 三套图表方案怎么选

| | Streamlit 内置图表 | Altair | **matplotlib** |
|---|---|---|---|
| 代码量 | **最少**（一行） | 中 | 中偏多 |
| 交互性 | ✅ 悬停提示、图例筛选 | ✅ **最强**（缩放/框选/自定义 tooltip） | ❌ **静态图片** |
| 精细控制 | ❌ 几乎没有 | 中（声明式） | ✅ **最强**（子图、双轴、误差棒、注释） |
| 子图 / 多面板 | ❌ | ⚠️ 要 `concat` / `hconcat` | ✅ **`plt.subplots(2,2)` 原生支持** |
| 统计图 | ❌ | 部分 | ✅ **箱线图、小提琴图、热力图、误差棒** |
| 中文支持 | ✅ 自动 | ✅ 自动 | ⚠️ **要手动设字体** |
| 适合 | 快速看数据形状 | 网页里的交互式图表 | **科研插图、统计报告、复杂版式** |
| 入口 | `st.line_chart` 等 | `st.altair_chart` | **`st.pyplot`** |

> **一句话选择：**「只想快速看一眼」→ 内置图表；「要给用户交互」→ Altair；
> 「要精细控制 / 出统计图 / 做子图」→ **matplotlib**。

### 本环境的图表库可用性

| 库 | 状态 | 说明 |
|---|---|---|
| **Streamlit 内置图表** | ✅ 可用 | 底层就是 Altair / Vega-Lite，最省事 |
| **Altair 6.2.2** | ✅ 可用 | 声明式，交互最强 |
| **matplotlib 3.11.1** | ✅ **可用** | `st.pyplot(fig)`；配合 `matplotlib.use("Agg")` + 中文字体设置使用 |
| plotly | ❌ 未安装 | 不可使用，只用注释说明用法（`st.plotly_chart`） |
| Bokeh / Graphviz | ❌ 未安装 | — |

**`st.map` 需要联网加载地图瓦片**，离线时会显示空白，本项目只用注释说明用法。

---

## 7. 输入组件（15 种）

| 组件 | 代码 | 返回值 |
|---|---|---|
| 按钮 | `st.button("点我")` | `bool` |
| 文本输入 | `st.text_input("标签")` | `str` |
| 文本域 | `st.text_area("标签")` | `str` |
| 数字输入 | `st.number_input("标签", min_value=0, max_value=100, value=50)` | `int`/`float`（**取决于 value 的类型**） |
| 滑块 | `st.slider("标签", 0, 100, 50)` | `int`/`float`；传元组默认值则返回**元组**（区间滑块） |
| 选择滑块 | `st.select_slider("标签", options=["差","中","良","优"])` | 选中值 |
| 下拉选择 | `st.selectbox("标签", ["A","B"])` | 选中值 |
| 多选 | `st.multiselect("标签", ["A","B"], default=["A"])` | `list` |
| 复选框 | `st.checkbox("勾选我")` | `bool` |
| 单选 | `st.radio("标签", ["1","2"], horizontal=True)` | 选中值 |
| 日期 | `st.date_input("选择日期")` | `datetime.date`（传元组则是区间） |
| 时间 | `st.time_input("选择时间")` | `datetime.time` |
| 文件上传 | `st.file_uploader("上传", type=["csv"])` | `UploadedFile` |
| 颜色 | `st.color_picker("选颜色", "#00FFAA")` | `str`（hex） |
| 相机 | `st.camera_input("拍照")` | `UploadedFile` |

**核心交互模型：用户输入 → 变量获取 → 页面更新。**不需要写回调。

```python
name = st.text_input("你的名字")     # ① 用户输入
st.write("你好，" + name)            # ② 变量获取 → ③ 页面更新
```

**实用技巧：**

- `format_func=lambda x: 中文映射[x]` —— **显示中文，返回代码**。
- `st.selectbox(..., accept_new_options=True)` —— 允许用户输入列表外的新值。
- 日期 + 时间拼成完整时间戳：`datetime.datetime.combine(d, t)`。
- `st.file_uploader` **一定要写 `type=`**，并且要处理"文件读不出来"的情况。

---

## 8. 媒体组件

| 组件 | 代码 | 支持的输入 |
|---|---|---|
| 图片 | `st.image("img.png", caption="说明")` | **路径 / URL / PIL.Image / numpy 数组** |
| 音频 | `st.audio("song.mp3")` | 路径 / URL / **bytes** / **numpy 数组**（需给 `sample_rate`） |
| 视频 | `st.video("video.mp4")` | 路径 / URL / bytes |

```python
st.image(img_obj, caption="PIL 对象", width="stretch")
st.image(arr, caption="numpy 数组")           # 形状 (高, 宽, 3)，uint8 0~255
st.image([p1, p2], caption=["图一", "图二"], width=260)   # 一次传列表
st.audio(audio_bytes, format="audio/mp3")     # 传 bytes
st.audio(samples, sample_rate=22050)          # 传 numpy 数组
st.video("video.mp4", start_time=10, end_time=30)
st.video("loop.mp4", autoplay=True, muted=True, loop=True)
```

> **本项目 `07_媒体.py` 的约定：绝不引用不存在的文件。**
> 图片用 **Pillow 现场生成**并保存成真实 PNG；音频用标准库 **`wave` + numpy 现场合成**；
> 视频只做文字说明。**这样在任何机器上运行都不会出现"文件找不到"的错误。**
>
> **实战经验：**凡是"读取文件"的代码，先判断文件是否存在：
> ```python
> if Path("photo.png").exists():
>     st.image(Image.open("photo.png"))
> else:
>     st.info("请先把 photo.png 放到脚本同目录下")
> ```

---

## 9. 布局组件

| 组件 | 代码 | 作用 |
|---|---|---|
| 列布局 | `col1, col2 = st.columns(2)` | 分成 N 列（整数等分 / 列表按比例） |
| 容器 | `with st.container(border=True):` | 把一组组件打包 |
| 占位符 | `ph = st.empty()` | 占位，后续**覆盖**内容 |
| 展开面板 | `with st.expander("展开查看"):` | 可折叠区域 |
| 选项卡 | `tab1, tab2 = st.tabs(["A","B"])` | 选项卡切换 |
| 弹出框 | `@st.dialog("标题")` | 模态对话框 |
| 侧边栏 | `with st.sidebar:` / `st.sidebar.xxx()` | 侧边栏 |
| 气泡弹窗 | `st.popover("点我")` | 不阻断操作的小气泡 |

### `st.columns`

```python
c1, c2, c3 = st.columns([1, 2, 1])                 # 按比例 1:2:1
st.columns(4)                                       # 等分 4 列
st.columns(3, gap="large", vertical_alignment="bottom")
```
列**可以嵌套**，但不要超过两层。

### `st.tabs` 的两个要点

1. **所有标签里的代码都会执行**（只是结果被隐藏）——所以内容很重时改用 `st.radio` + `if`。
2. `default="标签名"` 可以指定默认选中哪个。

### `st.expander`

里面的代码**也总是会执行**，只是结果被折叠了。`expanded=True` 默认展开。

### `st.container` 的杀手级用法：**先占位，后填充**

```python
header_area = st.container()      # ① 先在页面顶部占位
st.write("这一行在脚本里出现得更早，但在页面里位置更靠下")
# ...中间做一堆计算...
with header_area:                 # ② 最后往前面那个容器里写
    st.success("我的代码在最后，但显示在页面顶部")
```

### `st.empty`：原地更新

```python
ph = st.empty()
ph.info("处理中…")
time.sleep(1)
ph.success("完成了！")     # ← 覆盖掉上面的内容

ph.empty()                # 清空内容（占位块还在）
```

> ⚠️ `st.empty()` 只占**一个元素**的位置。往里面写多个组件会显示成一组，下次再写会**整组替换**。
> 它**不会**让多个组件叠在同一位置 —— 那是 `st.container(key=...)` 的用法。

### `@st.dialog`

```python
@st.dialog("确认删除")
def confirm_dialog():
    st.write("确定要删除吗？此操作不可撤销。")
    c1, c2 = st.columns(2)
    if c1.button("确定", type="primary"):
        do_delete()
        st.rerun()             # 重跑 → 不再调用它 → 对话框关闭
    if c2.button("取消"):
        st.rerun()

if st.button("删除"):          # ★ 调用这个函数 = 打开对话框
    confirm_dialog()
```

对话框**只在被调用时执行**，里面可以放任何组件（包括表单）。`dismissible=False` 强制必须做选择。

---

## 10. 状态与提示

| 组件 | 代码 | 作用 |
|---|---|---|
| 成功 | `st.success("成功！")` | 🟢 绿色框 |
| 信息 | `st.info("提示信息")` | 🔵 蓝色框 |
| 警告 | `st.warning("注意！")` | 🟡 黄色框 |
| 错误 | `st.error("出错了！")` | 🔴 红色框 |
| 异常 | `st.exception(e)` | 显示**完整堆栈**（开发用，生产慎用） |
| 加载动画 | `with st.spinner("处理中..."):` | 转圈 + 提示文字 |
| 进度条 | `bar = st.progress(0)` → `bar.progress(50, text="...")` | 显示进度 |
| 状态容器 | `with st.status("正在运行...") as s:` | 分阶段执行日志 |
| Toast | `st.toast("操作成功！", icon="✅")` | 右上角**自动消失**的通知 |
| 气球 | `st.balloons()` | 🎈 庆祝动画 |
| 雪花 | `st.snow()` | ❄️ 飘雪动画 |

### `st.spinner` vs `st.status`

| | `st.spinner` | `st.status` |
|---|---|---|
| 显示 | 一个"遮罩"式的转圈 | **可折叠的分步执行日志** |
| 结束后 | 消失 | `status.update(label=..., state="complete")` 变绿勾并自动折叠 |
| 适合 | 单个耗时操作 | **多步骤流程** |

`state` 三个取值：`"running"`（默认）/ `"complete"`（绿勾）/ `"error"`（红叉）。

### 进度条的两种写法

```python
# 写法 A（课案原文）：额外的 st.empty() 显示文字
bar = st.progress(0)
text = st.empty()
for i in range(101):
    bar.progress(i)
    text.text(f"进度：{i}%")

# 写法 B（更简洁）：text 参数直接写在进度条上
bar = st.progress(0, text="准备中…")
for i in range(101):
    bar.progress(i, text=f"处理中… {i}%")
bar.progress(100, text="完成！")
```

### ★ 重要原则

**`st.balloons()` / `st.snow()` / `st.progress()` / `st.spinner()` 这类"有副作用"的调用，
必须放在触发条件里**（如 `if st.button(...)`），否则**每次脚本重跑都会重新播放/重新等待**。

---

## 11. 侧边栏

```python
# 写法 A：with 块（推荐，缩进一眼看出"这些都在侧边栏里"）
with st.sidebar:
    st.title("导航菜单")
    menu = st.radio("选择页面", ["首页", "数据总览", "用户管理", "系统设置"])
    st.caption("当前用户：admin")
    uploaded = st.file_uploader("上传数据", type=["csv"])

# 写法 B：直接调用（完全等价）
st.sidebar.title("导航菜单")
st.sidebar.radio("选择页面", [...])

# 主区域根据侧边栏的选择渲染
if menu == "首页":
    st.write("欢迎来到首页")
elif menu == "数据总览":
    st.line_chart({"销售额": [100, 200, 150, 300, 250]})
```

**要点：**

1. 两种写法完全等价，**可以混用**。
2. **任何组件都能放进侧边栏**，但**侧边栏不能嵌套**。
3. **定位：侧边栏放「控制条件」（导航、筛选、参数），主区域放「结果」。**
4. 宽度不能直接用 Python 改（用户可拖拽），要改只能写 CSS（依赖内部 `data-testid`，慎用）。
5. 多页面应用的导航也默认渲染在侧边栏里。

---

## 12. 表单 `st.form`

**表单解决的问题：**默认每次输入变化都重跑脚本；表单把 N 个输入**打包成一次提交**。

```python
with st.form("register_form", clear_on_submit=True):
    username = st.text_input("用户名 *", placeholder="请输入用户名")
    password = st.text_input("密码 *", type="password")
    email = st.text_input("邮箱")

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("年龄", min_value=1, max_value=120, value=18)
    with col2:
        gender = st.radio("性别", ["男", "女", "其他"], horizontal=True)

    agree = st.checkbox("我已阅读并同意服务条款")

    # ★ form_submit_button 必须在 with form 块【内部】
    submitted = st.form_submit_button("注册", type="primary")

# 提交结果的判断写在 with 块【外面】
if submitted:
    if not username or not password:
        st.error("用户名和密码为必填项")
    elif not agree:
        st.warning("请先同意服务条款")
    else:
        st.success(f"注册成功！欢迎 {username}")
        st.json({"用户名": username, "邮箱": email, "年龄": age, "性别": gender})
```

### 参数

| 参数 | 默认 | 作用 |
|---|---|---|
| `key` | 必填 | 表单唯一标识。**页面上多个表单必须用不同 key** |
| `clear_on_submit` | `False` | 提交后是否清空输入框（"新增"场景很有用） |
| `enter_to_submit` | `True` | 在单行输入框里按回车是否触发提交 |
| `border` | `True` | 是否给表单画边框 |

### ★ 三条铁律

1. **`st.form_submit_button` 必须写在 `with st.form` 块内部。**
2. **表单不能嵌套**（`Forms cannot be nested`）。
3. **表单里的组件不会触发即时重跑** —— 所以：
   - 需要"实时联动"的输入（拖滑块看图表变化）→ **不要**用表单；
   - "一次性提交"的输入（注册、下单、保存配置）→ **用表单**。

### 按钮样式

`type="primary"` = 实心高亮（主操作：提交/保存/确认）；
`type="secondary"` = 空心（次要操作：取消/重置）。**一个区域里主按钮最多一个。**

### 校验经验

**把错误收集成列表一次性显示**，比"报一个改一个"体验好得多：

```python
errors = []
if len(name) < 2:   errors.append("姓名太短")
if not email_ok:    errors.append("邮箱格式错误")
if not topics:      errors.append("请至少选一个方向")

if errors:
    st.error(f"有 {len(errors)} 个问题需要修正：")
    for i, msg in enumerate(errors, 1):
        st.write(f"{i}. {msg}")
```

---

## 13. 会话状态 `st.session_state`

**为什么需要它？**因为脚本每次交互都重跑，**普通变量会被重置**。

```python
# === 初始化（★ 必须判断，否则每次重跑都会重置）===
if "count" not in st.session_state:
    st.session_state.count = 0

# === 计数器 ===
col1, col2, col3 = st.columns(3)
if col1.button("减一"):
    st.session_state.count -= 1
if col2.button("归零"):
    st.session_state.count = 0
if col3.button("加一"):
    st.session_state.count += 1
st.metric("当前计数", st.session_state.count)

# === 待办事项（注意 key 必须唯一）===
if "todos" not in st.session_state:
    st.session_state.todos = []

new_item = st.text_input("添加新事项", key="new_todo_input")
if st.button("添加"):
    if new_item:
        st.session_state.todos.append(new_item)
        st.rerun()          # 重跑一次，顺便清空输入框

for i, item in enumerate(st.session_state.todos):
    col_text, col_del = st.columns([4, 1])
    col_text.write(f"{i+1}. {item}")
    if col_del.button("删除", key=f"del_{i}"):     # ★ key 必须唯一
        st.session_state.todos.pop(i)
        st.rerun()

# === 模拟登录 ===
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if not st.session_state.logged_in:
    name = st.text_input("用户名")
    if st.button("登录"):
        st.session_state.logged_in = True
        st.session_state.user_name = name
        st.rerun()
else:
    st.success(f"已登录：{st.session_state.user_name}")
    if st.button("退出登录"):
        st.session_state.logged_in = False
        st.session_state.user_name = ""
        st.rerun()
```

### ★★ 持久边界（课案原文）★★

| 操作 | session_state |
|---|---|
| 点击按钮、拖动滑块等交互（`st.rerun()`） | ✅ **保留** |
| **刷新浏览器（F5）** | ❌ **重置** |
| 关闭标签页后重新打开 | ❌ **重置** |
| 重新执行 `streamlit run app.py` | ❌ **重置** |

> **所以 `session_state` 只在「当前标签页的连续交互中」有效。**
> 它**不是数据库** —— 需要真正持久化的数据必须写进文件 / 数据库。

### `key` 参数的两个作用

1. **用 `st.session_state.xxx` 直接读写组件值**：
   ```python
   name = st.text_input("名字", key="name_input")
   # 下面两种写法等价
   st.write(name)
   st.write(st.session_state.name_input)

   # ★ 程序化设置组件值：直接改 key 对应的值再重跑
   if st.button("一键改成张三"):
       st.session_state.name_input = "张三"
       st.rerun()
   ```
2. **区分同名组件**：页面上有两个"添加"按钮时必须用不同 `key`，
   否则报 `StreamlitDuplicateElementId`。**循环里创建的组件，key 必须带循环变量。**

### `st.rerun()`

立刻中止当前脚本、从头重跑。

> ⚠️ **`st.rerun()` 之后的代码不会执行！**
> 想在重跑后显示提示，用「**存消息 + 重跑 + 开头读取**」模式：
>
> ```python
> # ① 脚本靠前的位置：读出上次的消息并显示，然后清掉
> if st.session_state.get("flash_message"):
>     st.success(st.session_state.flash_message)
>     st.session_state.flash_message = None
>
> # ② 操作处：存消息再 rerun
> if st.button("保存"):
>     do_save()
>     st.session_state.flash_message = "保存成功！"
>     st.rerun()
> ```

### 常用操作速查

```python
if "key" not in st.session_state: st.session_state.key = 初始值   # 初始化
v = st.session_state.key                    # 读
v = st.session_state.get("key", 默认值)      # 读（不存在时给默认值）
st.session_state.key = 新值                  # 写
if "key" in st.session_state: ...            # 判断
del st.session_state["key"]                  # 删单个
for k in list(st.session_state.keys()):      # 清空全部
    del st.session_state[k]
st.json(dict(st.session_state))              # ★ 调试：一眼看清所有状态
```

### 回调函数写法（进阶）

```python
def increment():
    st.session_state.count += 1

st.button("加一", on_click=increment)   # 不需要 if，状态在重跑【之前】就改好了
```
> 回调里**不能调用 `st.write` 之类的渲染函数**（那时还没有渲染上下文）。

---

## 14. 缓存机制

**解决什么问题？**脚本每次交互都重跑 → 耗时的操作会被反复执行。

| 装饰器 | 缓存什么 | 返回什么 | 是否要求可序列化 |
|---|---|---|---|
| `@st.cache_data` | **数据**（DataFrame / list / dict / 数值） | **数据的副本** | ✅ 要求 |
| `@st.cache_resource` | **资源**（连接 / 模型 / 锁 / 客户端） | **同一个对象**（`id` 相同） | ❌ 不要求 |

```python
# 缓存数据加载（只加载一次，后续从缓存读取）
@st.cache_data(ttl=3600)          # ttl=3600 表示 1 小时后缓存过期
def load_data():
    time.sleep(2)                 # 模拟耗时加载
    return pd.DataFrame({...})

st.title("缓存演示")

if st.button("加载数据"):
    with st.spinner("首次加载需要 2 秒，后续秒开..."):
        df = load_data()
    st.dataframe(df)
    st.caption("再次点击按钮，数据瞬间显示（命中缓存）")

if st.button("清除缓存"):
    load_data.clear()
    st.success("缓存已清除，下次加载将重新计算")
    st.rerun()
```

```python
# 缓存资源：全局单例、不可序列化的对象
@st.cache_resource
def get_db_connection(db_url="sqlite:///demo.db"):
    return create_engine(db_url)          # SQLAlchemy Engine 不能被 pickle

@st.cache_resource
def get_lock():
    return threading.Lock()               # 锁也必须是全局唯一的
```

### 怎么选

> 你返回的东西**能不能被"复制一份"而没有任何问题**？
> **能 → `cache_data`；不能（连接、模型、锁）→ `cache_resource`。**
> 不确定时**先用 `cache_data`**。

### ★ 怎么证明缓存生效

**在被缓存的函数里打印一行日志**，观察第二次还会不会打印：

```python
@st.cache_data
def load_data(n: int):
    print(f"[cache_data] 实际执行了 load_data(n={n})")   # ← 只有第一次会打印
    time.sleep(2)
    return pd.DataFrame({...})
```

### 参数

| 参数 | 作用 |
|---|---|
| `ttl` | 缓存存活时间（秒 / `"1h"` / `timedelta`），到点自动失效 |
| `max_entries` | 最多缓存多少份结果（超出按 LRU 淘汰） |
| `show_spinner` | 缓存未命中时是否显示转圈 |
| `persist` | 是否持久化到磁盘（**仅 `cache_data`**） |
| `hash_funcs` | 自定义不可哈希参数的处理方式 |

### 清除缓存

```python
load_data.clear()           # 清空【这一个函数】的缓存
st.cache_data.clear()       # 清空所有 cache_data 缓存
st.cache_resource.clear()   # 清空所有 cache_resource 缓存
```

### 两个经典坑

**坑 1：传了不可哈希的参数**（list / dict / DataFrame / 自定义对象）

```python
# ❌ UnhashableParamError
@st.cache_data
def process(items: list): ...

# ✅ 办法一：参数名后加下划线（★ 危险：该参数不参与缓存键！）
@st.cache_data
def process_a(items_): ...          # 不管传什么列表都返回第一次的结果

# ✅ 办法二：转成可哈希类型（推荐）
@st.cache_data
def process_b(items: tuple): ...
process_b((1, 2, 3))
```

**坑 2：函数里有副作用**（被缓存的函数只在未命中时执行）

```python
# ❌ 永远返回第一次调用时的时间
@st.cache_data
def get_now():
    return time.time()

# ✅ 缓存"取数"，时间戳留在外面
@st.cache_data
def get_data():
    return fetch_from_db()

fetched_at = time.time()
data = get_data()
```

> **原则：被 `@st.cache_*` 装饰的函数必须是「纯函数」** —— 只根据参数算结果，不产生副作用。

### `session_state` vs `cache`

| | `st.session_state` | `@st.cache_*` |
|---|---|---|
| 解决什么 | **状态丢失** | **重复计算** |
| 作用范围 | **单个用户的单个标签页** | **所有用户、所有会话共享** |
| 生命周期 | 刷新浏览器就没 | 进程活着就一直在（可设 `ttl`） |

---

## 15. 多页面应用

### 方式 1：`pages/` 目录 + `st.navigation`（推荐 ≥3 个页面）

```
myapp/
├── app.py                       ← 入口：配置 + 注册导航 + pg.run()
└── pages/
    ├── 1_数据总览.py             ← 文件名前面的数字决定导航顺序
    ├── 2_用户管理.py
    └── 3_系统设置.py
```

```python
# app.py（入口）
import streamlit as st

st.set_page_config(page_title="管理后台", page_icon="🏠", layout="wide")

# 首页用【函数】定义
def home():
    st.title("管理后台首页")
    col1, col2, col3 = st.columns(3)
    col1.metric("用户总数", "1,280", "5.2%")

# 注册导航：首页用函数，子页用文件路径
pg = st.navigation([
    st.Page(home, title="首页", icon="🏠", url_path="home", default=True),
    st.Page("pages/1_数据总览.py", title="数据总览", icon="📊", url_path="overview"),
    st.Page("pages/2_用户管理.py", title="用户管理", icon="👥", url_path="users"),
    st.Page("pages/3_系统设置.py", title="系统设置", icon="⚙️", url_path="settings"),
])

pg.run()      # ★ 执行当前选中的页面
```

**`st.Page` 参数：**

| 参数 | 作用 |
|---|---|
| `page` | 文件路径字符串 **或一个无参函数** |
| `title` | 导航里显示的名字 |
| `icon` | 导航里的图标 |
| `url_path` | 浏览器地址栏路径（中文标题建议显式指定） |
| `default` | 是否默认打开（不写则第一个是默认） |
| `visibility` | `"visible"` / `"hidden"`（hidden 不出现在导航，但可通过 URL 访问） |

**`st.navigation` 参数：**`position="sidebar"`（默认）/ `"hidden"`（自己写导航）/ `"top"`；
`expanded` 控制分组导航是否默认展开。也可以传字典做**分组导航**。

> ⚠️ **`st.set_page_config` 只在入口调用一次，子页面绝对不要调用。**
>
> ⚠️ **页面之间不能共享普通变量** —— 跨页面共享数据必须用 `st.session_state`。

### 方式 2：单文件 + 侧边栏（适合 2~3 个页面）

```python
import streamlit as st

st.set_page_config(page_title="多页面演示", page_icon="📱")

if "page" not in st.session_state:      # ① 初始化当前页面
    st.session_state.page = "首页"

with st.sidebar:                        # ② 侧边栏导航
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

st.title(f"当前页面：{st.session_state.page}")     # ③ 按状态分发

if st.session_state.page == "首页":
    st.write("欢迎来到首页")
elif st.session_state.page == "数据":
    st.line_chart({"A": [1, 2, 3], "B": [4, 5, 6]})
elif st.session_state.page == "设置":
    st.slider("音量", 0, 100, 50)
```

### 两种方式对比

| 对比项 | 方式 1（`pages/` + `st.navigation`） | 方式 2（单文件 + 侧边栏） |
|---|---|---|
| 文件组织 | 一页一文件 | 全部在一个 `.py` |
| 导航栏 | **框架自动生成**，自动高亮 | 自己写按钮 / radio |
| 当前页状态 | 框架管理 | 自己用 `session_state` |
| **浏览器 URL** | **每页独立 URL（可分享、可前进后退）** | 只有一个 URL |
| 适合 | **3 个页面以上的正式项目** | **2~3 个页面的简单小应用** |
| 版本要求 | 需要 1.36+ | 所有版本通用 |

> **课案原文的总结：**方式 1 多文件拆分、导航自动高亮当前页，适合 3 个页面以上的正式项目；
> 方式 2 单文件，适合 2-3 个页面的简单小应用。

### 动态页面（按权限显示不同导航）

```python
if st.session_state.role == "admin":
    pages = [st.Page(home, title="首页"),
             st.Page("admin.py", title="图书管理")]
else:
    pages = [st.Page(home, title="首页"),
             st.Page("user.py", title="我的图书馆")]

pg = st.navigation(pages)
pg.run()
```

### `st.stop()`：权限门禁

```python
if not st.session_state.get("logged_in"):
    st.warning("请先登录")
    st.stop()                # ★ 立刻结束脚本（不是报错），后面的代码一行都不跑
```

---

## 16. 综合实战：图书管理系统

见 `15_图书管理系统/README.md`（含完整的功能清单、数据文件结构、业务规则、
与课案原文的差异说明）。

**核心设计：**

```
app.py / admin.py / user.py     ← 界面层：只负责"显示"和"收集输入"
            │
            ▼
        data.py                  ← 数据层：JSON 持久化 + 全部业务规则（不抛异常）
            │
            ▼
    data/library.json            ← 存储层：JSON 文件，重启不丢
```

**关键点：**

- 数据层所有函数返回 `(是否成功, 提示信息)`，**绝不抛异常** → 界面永远不会出现 Traceback。
- 密码用 `hashlib.sha256` 哈希存储，**不存明文**。
- 借还书要**同时**更新「图书可借册数」「用户借阅列表」「借阅记录」三处，保证一致。
- 用 `threading.Lock` 保护"读—改—写"三步，避免并发写坏文件。

---

## 17. 常见坑速查

| 坑 | 现象 | 解决 |
|---|---|---|
| `set_page_config` 不是第一个调用 | `StreamlitAPIException` | 移到最前面；多页面只在入口调一次 |
| 普通变量跨重跑丢失 | 计数器永远不动 | 用 `st.session_state` + `if "key" not in ...` 初始化 |
| `st.rerun()` 之后的代码不执行 | 提示不显示 | 用「存消息 + 重跑 + 开头读取」模式 |
| 循环里创建的组件 key 重复 | `StreamlitDuplicateElementId` | `key=f"item_{i}"` |
| `form_submit_button` 写在 form 外面 | `StreamlitAPIException: must be used inside an st.form()` | 移进 `with st.form` 块内（**这是硬性错误，不是警告**） |
| `st.line_chart` 的 x 轴不对 | 横轴变成行号 | `.set_index("日期")` 或显式传 `x=` |
| `use_container_width` 警告 | 运行时打印弃用提示 | 改用 `width="stretch"` |
| 下载 CSV 用 Excel 打开乱码 | 中文变问号 | `.encode("utf-8-sig")` |
| `st.table` 渲染几万行 | 页面卡死 | 改用 `st.dataframe` |
| `st.image("a.png")` 文件不存在 | `FileNotFoundError` | 先 `Path("a.png").exists()` 判断 |
| 缓存函数里有副作用 | 时间/随机数永远是第一次的值 | 被缓存的函数必须是纯函数 |
| `@st.cache_data` 传 list/dict | `UnhashableParamError` | 转成 tuple，或用 `hash_funcs` |
| `st.balloons()` 无条件执行 | 每点一次任何按钮都放气球 | 放进 `if st.button(...)` 里 |
| 表单里放"实时联动"的组件 | 拖滑块页面没反应 | 实时联动**不要**用表单 |
| `st.sidebar` 里再开 sidebar | 不存在 | 侧边栏不能嵌套 |
| `st.tabs` 里放了很重的查询 | 打开页面就很慢 | 每个标签的代码都会执行；改用 `st.radio` + `if` |
| `matplotlib.use("Agg")` 写在 `import pyplot` 之后 | 警告"后端已初始化" | 必须**在 import pyplot 之前**调用 |
| matplotlib 图里中文变方块 | `口口口`，并提示 glyph missing | 设 `rcParams["font.sans-serif"]` + `axes.unicode_minus=False` |
| 用了 `plt.show()` | 服务器上卡住 / 报错 | 改用 `st.pyplot(fig)`，**不要 `plt.show()`** |
| 只 `plt.subplots()` 不 `plt.close(fig)` | 内存上涨，警告 "More than 20 figures" | 画完就 `plt.close(fig)`（或让 `st.pyplot` 自动清理） |

---

## 18. 重点回顾

- **Streamlit 用 Python 直接生成网页**，不写 HTML/CSS/JS，
  适合数据应用和内部工具。
- **运行模型是理解一切的前提**：每次交互都重跑整个脚本 →
  所以需要 `session_state`（存状态）和 `cache`（省计算）。
- **`set_page_config` 必须是第一个调用，且只能一次**（多页面只在入口调）。
- **表单**把 N 个输入打包成一次提交；**提交按钮必须在 form 内部**。
- **`session_state` 的持久边界**：交互保留，刷新浏览器就重置 —— 它不是数据库。
- **`cache_data` 存数据（返回副本），`cache_resource` 存资源（返回同一对象）**；
  证明缓存生效的方法是在函数里打印日志。
- **多页面有两种方式**：`pages/` + `st.navigation`（有独立 URL，适合正式项目）
  和单文件 + 侧边栏 + `session_state`（适合 2~3 页的小应用）。
- **图表有三套方案**：内置图表（`st.line_chart` 等，一行出图）/ Altair
  （`st.altair_chart`，交互最强）/ **matplotlib（`st.pyplot(fig)`，精细控制最强）**。
  用 matplotlib 要记住四件事：**`use("Agg")` 写在 import 之前、设中文字体、
  不要 `plt.show()`、画完 `plt.close(fig)`**。
- **综合实战**验证了一件事：只写 Python，就能做出带登录、权限、增删改查、
  数据持久化的完整管理系统。

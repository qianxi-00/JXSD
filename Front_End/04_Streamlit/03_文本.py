"""
=====================================================================================
文件：03_文本.py
对应课案章节：Streamlit → 文本
本节知识点（课案表格里的 9 个组件，全部覆盖）：
  1. st.title        页面标题（最大号，每页一般只用一次）
  2. st.header       二级标题
  3. st.subheader    三级标题
  4. st.write        万能显示：文字 / 数字 / 列表 / 字典 / DataFrame / 图表……
  5. st.markdown     渲染 Markdown 格式（粗体、斜体、列表、表格、链接、代码……）
  6. st.code         带语法高亮的代码块
  7. st.latex        LaTeX 数学公式
  8. st.caption      灰色小字说明
  9. st.text         纯文本（不解析 Markdown）
  10. st.divider     水平分割线（课案表格里也有它）

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m streamlit run `
        'F:\\ProGram\\Python_Base\\Front_End\\04_Streamlit\\03_文本.py' `
        --server.headless true --server.port 8603 --browser.gatherUsageStats false
=====================================================================================
"""

import pandas as pd
import streamlit as st

# set_page_config 必须是第一个 Streamlit 调用
st.set_page_config(page_title="03 文本", page_icon="📝", layout="wide")

st.title("03 文本组件")
st.caption("对应课案：Streamlit → 文本")

st.markdown(
    """
本节把课案表格里的 9 个文本组件逐个演示一遍。
**每个组件上面是说明 + 代码，下面是它的真实渲染结果**，方便你直接对照。
"""
)

# ---------------------------------------------------------------------------
# 课案的组件速查表
# ---------------------------------------------------------------------------
st.header("速查表（课案原文）")

st.markdown(
    """
| 组件 | 代码 | 作用 |
|---|---|---|
| 页面标题 | `st.title("文本")` | 最大号标题，每页只用一次 |
| 二级标题 | `st.header("文本")` | 中等标题 |
| 三级标题 | `st.subheader("文本")` | 较小标题 |
| 通用文本 | `st.write("文本")` | 万能显示，支持文字/数字/DataFrame |
| Markdown | `st.markdown("**粗体** *斜体*")` | 渲染 Markdown 格式 |
| 代码块 | `st.code("print('hello')", language="python")` | 带语法高亮的代码 |
| 公式 | `st.latex(r"E = mc^2")` | LaTeX 数学公式 |
| 说明文字 | `st.caption("小字说明")` | 灰色小字，用于注释 |
| 纯文本 | `st.text("纯文本")` | 纯文本 |
| 分割线 | `st.divider()` | 一条水平分割线 |
"""
)

st.divider()

# ---------------------------------------------------------------------------
# 一、三种标题
# ---------------------------------------------------------------------------
st.header("一、三种标题：title / header / subheader")

st.markdown("它们的关系类似 HTML 里的 `<h1>` / `<h2>` / `<h3>`：字号依次减小。")
st.code(
    '''st.title("① 这是一级标题 (title)")
st.header("② 这是二级标题 (header)")
st.subheader("③ 这是三级标题 (subheader)")''',
    language="python",
)

st.markdown("**渲染结果：**")
st.title("① 这是一级标题 (title)")
st.header("② 这是二级标题 (header)")
st.subheader("③ 这是三级标题 (subheader)")

st.warning(
    "**注意：**`st.title` 虽然可以调用多次，但按惯例**一个页面只用一次**，"
    "用来表示页面的主题（就像 HTML 里一个页面只写一个 `<h1>`）。"
    "本页最上面的那个 `st.title` 才是它作为「页面标题」的用法，"
    "这里再调用只是为了演示效果。"
)

st.divider()

# ---------------------------------------------------------------------------
# 二、st.write —— 万能显示
# ---------------------------------------------------------------------------
st.header("二、st.write：万能显示")

st.markdown(
    """
`st.write` 是 Streamlit 里最"聪明"的函数：**你给它什么，它就用最合适的方式显示出来**，
不需要你记住一大堆专用函数。课案原文说它"万能显示"，就是这个意思。
"""
)

st.code(
    '''st.write("文字：hello world")
st.write("数字：", 12345)
st.write("列表：", [1, 2, 3, 4, 5])
st.write("字典：", {"name": "张三", "age": 25})
st.write("DataFrame：", df)''',
    language="python",
)

st.markdown("**渲染结果：**")
st.write("文字：hello world")
st.write("数字：", 12345)
st.write("列表：", [1, 2, 3, 4, 5])
st.write("字典：", {"name": "张三", "age": 25})

# 直接传入 DataFrame
df = pd.DataFrame({"姓名": ["张三", "李四"], "分数": [95, 87]})
st.write("DataFrame：", df)

st.markdown(
    """
**`st.write` 能自动处理的类型（部分）：**

| 传入类型 | 渲染成 |
|---|---|
| `str` | Markdown 文本 |
| `int` / `float` | 数字 |
| `list` / `tuple` / `dict` | 结构化展示 |
| `pandas.DataFrame` | 交互表格 |
| `pandas.Series` | 折线图 |
| `matplotlib` / `altair` / `plotly` 图表对象 | 图表 |
| 函数 / 类 | 源码高亮 |
"""
)

st.info(
    "**小技巧：**`st.write` 可以一次传多个参数，它们会**横向排成一行**："
    "`st.write(\"总数：\", 42, \"条\")`。"
)
st.write("试试多参数：", "总数 =", 42, " 条，平均分 =", 91.0)

st.divider()

# ---------------------------------------------------------------------------
# 三、st.markdown
# ---------------------------------------------------------------------------
st.header("三、st.markdown：渲染 Markdown")

st.markdown("课案原文：`st.markdown(\"**粗体**、*斜体*、\\`代码\\`\")`")
st.code(
    '''st.markdown("支持 **粗体**、*斜体*、`代码`")''',
    language="python",
)

st.markdown("**渲染结果：**")
st.markdown("支持 **粗体**、*斜体*、`代码`")

st.markdown("**Markdown 常用语法速查（都在下面真实渲染出来）：**")

st.markdown(
    """
# 一级标题（#）
## 二级标题（##）
### 三级标题（###）

**粗体**、*斜体*、~~删除线~~、`行内代码`

- 无序列表项 1
- 无序列表项 2
  - 嵌套项（前面加两个空格）

1. 有序列表项 1
2. 有序列表项 2

> 引用块：用 > 开头

[这是一个链接](https://www.python.org)

| 表头1 | 表头2 |
|---|---|
| 单元格 | 单元格 |

---

上面这条横线是用 `---` 写的（等同于 `st.divider()`）。
"""
)

st.markdown("**还支持直接写 HTML（加 `unsafe_allow_html=True`）：**")
st.code(
    '''st.markdown(
    "你选的颜色：<span style='color:#e74c3c'>████</span> #e74c3c",
    unsafe_allow_html=True,
)''',
    language="python",
)
st.markdown(
    "你选的颜色：<span style='color:#e74c3c;font-size:20px;'>████</span> "
    "<code>#e74c3c</code>　（这段 HTML 是真实生效的）",
    unsafe_allow_html=True,
)
st.warning(
    "**安全提示：**`unsafe_allow_html=True` 会真的把字符串当 HTML 渲染。"
    "如果内容是用户输入的，就可能被注入脚本（XSS）。"
    "**只有当 HTML 是你自己写死的时候才用它。**"
)

st.divider()

# ---------------------------------------------------------------------------
# 四、st.code
# ---------------------------------------------------------------------------
st.header("四、st.code：带语法高亮的代码块")

st.markdown(
    """
`st.code(body, language="python")`

- `body`：要显示的代码字符串。
- `language`：语言名，决定高亮规则，例如 `"python"` / `"javascript"` / `"html"` /
  `"css"` / `"sql"` / `"bash"` / `"json"` / `"yaml"` / `"toml"`。
- `line_numbers=True`：显示行号（可选）。
- 代码块右上角有一个**复制按钮**，用户一键就能复制。
"""
)

st.code(
    '''def hello():
    print('Hello World')''',
    language="python",
)

st.markdown("同一段代码换个 `language` 参数，高亮就完全不同：")
col_py, col_js = st.columns(2)
with col_py:
    st.caption('language="python"')
    st.code("const x = 1;\nconsole.log(x);", language="python")
with col_js:
    st.caption('language="javascript"')
    st.code("const x = 1;\nconsole.log(x);", language="javascript")

st.markdown("**带行号的写法：**")
st.code(
    """import streamlit as st

st.title("带行号的代码块")
st.write("line_numbers=True")""",
    language="python",
    line_numbers=True,
)

st.divider()

# ---------------------------------------------------------------------------
# 五、st.latex
# ---------------------------------------------------------------------------
st.header("五、st.latex：LaTeX 数学公式")

st.markdown(
    """
`st.latex(r"...")` 会把字符串当成 LaTeX 数学公式渲染出来。

⚠️ **建议在字符串前加 `r`**（raw string），否则反斜杠 `\\` 会被 Python 当成转义字符，
公式就会渲染失败。这就是课案写 `st.latex(r"E = mc^2")` 的原因。
"""
)

st.markdown("**课案原文的积分公式：**")
st.code(
    '''st.latex(r"\\int_a^b f(x) dx = F(b) - F(a)")''',
    language="python",
)
st.latex(r"\int_a^b f(x) dx = F(b) - F(a)")

st.markdown("**更多例子：**")
col_a, col_b = st.columns(2)
with col_a:
    st.caption("质能方程（课案里的另一个例子）")
    st.latex(r"E = mc^2")
    st.caption("二次方程求根公式")
    st.latex(r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}")
with col_b:
    st.caption("求和符号")
    st.latex(r"\sum_{i=1}^{n} i = \frac{n(n+1)}{2}")
    st.caption("矩阵")
    st.latex(r"\begin{bmatrix} a & b \\ c & d \end{bmatrix}")

st.info(
    "**提示：**如果只是想在 Markdown 文本里内联一个公式，"
    "用 `st.markdown` 配合 `$...$`（行内）或 `$$...$$`（独立一行）也可以。"
)
st.markdown(r"例如：行内公式 $a^2 + b^2 = c^2$ 可以直接写在 Markdown 里。")

st.divider()

# ---------------------------------------------------------------------------
# 六、st.caption 与 st.text
# ---------------------------------------------------------------------------
st.header("六、st.caption 与 st.text")

col_cap, col_txt = st.columns(2)

with col_cap:
    st.subheader("st.caption")
    st.markdown("灰色小字，用于补充说明、注释、数据来源等。")
    st.code(
        'st.caption("这是一段说明小字")',
        language="python",
    )
    st.caption("这是一段说明小字 —— 我比正文小一号，而且是灰色的")

with col_txt:
    st.subheader("st.text")
    st.markdown(
        "**纯文本**：不做 Markdown 渲染，`**粗体**` 这类符号会原样显示。"
        "适合显示不希望被解析的原始内容。"
    )
    st.code(
        'st.text("**这段不会变粗** *也不会变斜体*")',
        language="python",
    )
    st.text("**这段不会变粗** *也不会变斜体*")

st.markdown("**对比一下 markdown 和 text 对同一串内容的处理：**")
col_md, col_tx = st.columns(2)
with col_md:
    st.caption("st.markdown 的结果")
    st.markdown("**加粗**、*斜体*、`代码`、[链接](https://www.python.org)")
with col_tx:
    st.caption("st.text 的结果")
    st.text("**加粗**、*斜体*、`代码`、[链接](https://www.python.org)")

st.divider()

# ---------------------------------------------------------------------------
# 七、st.divider
# ---------------------------------------------------------------------------
st.header("七、st.divider：分割线")

st.markdown("上面你用到的每一处横线，都是它画出来的。")
st.code('st.divider()', language="python")
st.write("上半部分内容")
st.divider()
st.write("下半部分内容（上面这条线就是 st.divider() 画的）")

st.divider()

# ---------------------------------------------------------------------------
# 八、课案原文的完整示例
# ---------------------------------------------------------------------------
st.header("八、课案原文的完整示例（逐行对照）")

st.code(
    '''import streamlit as st

st.title("Streamlit 文本组件演示")
st.header("这是二级标题")
st.subheader("这是三级标题")

# st.write 万能显示：文字、数字、DataFrame、图表都能直接丢给它
st.write("文字：hello world")
st.write("数字：", 12345)
st.write("列表：", [1, 2, 3, 4, 5])
st.write("字典：", {"name": "张三", "age": 25})

# 直接传入 DataFrame
import pandas as pd
df = pd.DataFrame({"姓名": ["张三", "李四"], "分数": [95, 87]})
st.write("DataFrame：", df)

st.markdown("支持 **粗体**、*斜体*、`代码`")
st.code("def hello():\\n    print('Hello World')", language="python")
st.latex(r"\\int_a^b f(x) dx = F(b) - F(a)")
st.caption("这是一段说明小字")
st.text("纯文本")

st.divider()''',
    language="python",
)

st.success(
    "**本节要点回顾：**\n"
    "1. `st.title` / `st.header` / `st.subheader` = 三级标题，title 每页只用一次。\n"
    "2. `st.write` 是万能的，不确定用哪个组件时先用它。\n"
    "3. `st.markdown` 支持完整 Markdown 语法，甚至可以直接写 HTML（慎用）。\n"
    "4. `st.latex` 的字符串前面记得加 `r`。\n"
    "5. `st.caption` 是灰色小字，`st.text` 是不解析 Markdown 的纯文本。"
)

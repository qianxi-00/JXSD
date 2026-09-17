# -*- coding: utf-8 -*-
"""一次性课案提取脚本 —— Typora 导出的课案 HTML → 纯文本 + 代码块文本。

**不是**流水线的一环，也不被任何模块 import（全仓只有 ``VERIFY_REPORT.md``
第 9 节用文字提到过它）：只在「要核课案原文」时手工跑一次。
`docs/课案全文提取.txt` 与 `docs/课案代码块提取.txt` 就是它产物的副本（改过名）——
重跑会覆盖 `OUT` / `OUTC`，**不会**自动更新 docs/ 下那两份。

输入 / 输出
    | 方向 | 路径 |
    |---|---|
    | 输入 | `SRC`：Typora 导出的课案 HTML（本机 DSH 附件目录里的那一份） |
    | 输出 1 | `OUT`：可读全文（标题转 `#`、代码转围栏、表格转竖线分隔的行） |
    | 输出 2 | `OUTC`：只留代码块，供 grep 配置项 / 环境变量 / API 形状 |

⚠️ **本脚本现在直接跑不起来**：两个输出路径都在仓库根的 ``_html_parse/`` 下，
而该目录**当前不存在**（产物已被移进 ``docs/``），
两次 ``open(OUT, "w")`` 没有 ``mkdir`` 兜底 ⇒ 会以 ``FileNotFoundError`` 结束。
要重跑就先建目录，或把 OUT / OUTC 指到你要的位置（`SRC` 本机目前还在）。

已保留的原始说明（英文 docstring 原文）
    Extract readable text + code from the Typora-exported HTML courseware (v2).
    - skips mermaid diagram CSS noise, keeps diagram node/edge labels as text
    - unwraps CodeMirror code fences into plain ``` blocks

踩过的坑
    · **Typora 的代码块不是 ``<pre><code>``**：它是 CodeMirror，每行一个
      ``<pre class="CodeMirror-line">``（新版是 ``<div class="cm-line">``）。
      直接取外层 ``text_content()`` 会得到一整坨没换行的字符串 —— 所以
      `extract_code()` 要按行拼回 ``\\n``，两条分支对应 Typora 的新旧版式。
    · **mermaid 图是 SVG，直接走文本会捞到一堆样式噪声**（字体名、路径、颜色值）。
      所以 `walk()` 遇到 ``md-diagram-panel`` 就只取 ``<text>`` / ``<span>`` 里的
      节点标签、按出现顺序用 ``->`` 串起来，而不是把 SVG 全 dump 出来。
    · ``&nbsp;`` / 零宽空格会让标题和表头对不上（同一个词看起来一样却 grep 不到），
      `norm()` 统一清掉并合并连续空格。
    · **导入即执行**：本文件是脚本式写法，顶层就跑 `walk()` 并写文件，
      没有 ``if __name__ == "__main__"`` 保护，也没有自检 —— 别 import 它。
"""
import re
from lxml import html as LH

# 课案 HTML 的来源（DSH 附件目录里的固定哈希文件名）。
# ⚠️ 这是本机绝对路径且不可参数化：附件被清理或换机器后这里必然读不到，
#    重新跑之前先把 SRC 改成你手头那份 HTML 的实际路径。
SRC = r"F:\ProGramApp\DSH\attachments\v1\files\e3\e30e09c2e27346090d8d2c3fdfaaea807072e0f2f53db596bb278907e067bedd\自媒体Agent.html"
# 产物落在仓库根的 _html_parse/ 下（不是 docs/；docs/ 里那两份是改过名的副本）。
# ⚠️ **该目录当前不存在**，而这两次 open() 没有 mkdir 兜底 —— 直接跑会
#    FileNotFoundError。重跑前先建目录，或把这两个常量指到别处。
OUT = r"F:\ProGram\Python_Base\_html_parse\courseware.txt"
OUTC = r"F:\ProGram\Python_Base\_html_parse\courseware_code_only.txt"

# 以二进制读再交给 lxml：让 lxml 自己按 <meta charset> 判编码，
# 比先 decode 成 str 再解析更不容易在混合编码的导出文件上踩坑
with open(SRC, "rb") as f:
    data = f.read()

doc = LH.fromstring(data)
# Typora 把正文放在 <div id="write">；万一导出版式变了就退到 <body>，
# 都取不到时干脆用整个文档 —— 保证后面 walk() 一定有个根可走
write = doc.xpath('//div[@id="write"]')
root = write[0] if write else (doc.body if doc.body is not None else doc)

lines = []
# 这些标签的内容一律丢：样式/脚本/矢量图，留着只会污染提取文本
SKIP = {"style", "script", "svg", "noscript"}


def norm(t: str) -> str:
    """清洗文本：去不换行空格与零宽空格、合并连续空格、去首尾空白。

    中文与英文排版混排时 Typora 会插 ``&nbsp;`` 与零宽字符，
    不清理会让同一个词在提取文本里 grep 不到。

    Args:
        t: 原始文本。

    Returns:
        str: 清洗后的单行文本。
    """
    t = t.replace("\u00a0", " ").replace("\u200b", "")
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def txt(el) -> str:
    """取一个元素的可见文本并 `norm()` 清洗。

    Args:
        el: lxml 元素。

    Returns:
        str: 清洗后的文本（可能为空串）。
    """
    return norm(el.text_content())


def extract_code(pre) -> str:
    """把 Typora 的 CodeMirror 代码块还原成多行纯文本。

    Typora 导出 HTML 时不用 ``<pre><code>``，而是把每行代码放进一个独立的
    ``<pre class="CodeMirror-line">``（旧版）或 ``<div class="cm-line">``（新版）——
    这正是本函数有两条分支的原因。若两条都没命中（既不是 CodeMirror 也不是 cm-line），
    就退回整块的 ``text_content()``，宁可换行丢了也别返回空。

    Args:
        pre: 代码块容器元素（class 含 ``md-fences``，或本身就是 ``pre``）。

    Returns:
        str: 代码原文，行间用 ``\\n`` 连接；取不到内容时是空串。
    """
    ln = pre.xpath('.//pre[contains(@class,"CodeMirror-line")]')
    if ln:
        return "\n".join(x.text_content() for x in ln)
    cml = pre.xpath('.//*[contains(@class,"cm-line")]')
    if cml:
        return "\n".join(x.text_content() for x in cml)
    return pre.text_content()


def diagram_labels(panel):
    """Compact text of a rendered mermaid diagram: node labels in document order.

    只取 SVG 里的 ``<text>`` / ``<span>``（节点与连线上的标签），
    按文档顺序用 ``->`` 串起来。**不取样式噪声** —— 直接 dump SVG 会捞到
    字体名、path 坐标、颜色值，那些对「核课案讲了什么」毫无用处。

    Args:
        panel: class 含 ``md-diagram-panel`` 的容器元素。

    Returns:
        str: 形如 ``开始 -> 判断 -> 结束``；一个标签都没取到时返回**空串**
        （调用方据此决定整段图要不要输出）。
    """
    labels = []
    for t in panel.xpath(".//*[local-name()='text'] | .//*[local-name()='span']"):
        s = norm(t.text_content())
        # 同一标签在 SVG 里常被拆成多个 <text>/<tspan> 重复出现，去重后再拼
        if s and s not in labels:
            labels.append(s)
    return " -> ".join(labels) if labels else ""


def walk(el):
    """递归遍历 HTML，把每种标签翻译成一到多行 Markdown 文本追加进 `lines`。

    这是整个脚本的核心分发器：标题 → ``#``、段落 → 原文、代码块 → 围栏、
    表格 → ``| a | b |``、列表项 → ``-``、引用 → ``>``、mermaid 图 → ``[DIAGRAM]``。
    **没有匹配到任何规则的标签继续往下递归**，所以 ``div`` / ``section`` 这类
    容器不需要显式处理。

    Args:
        el: 当前 lxml 元素。

    Returns:
        None: 结果通过模块级列表 `lines` 累积（脚本式写法，不是纯函数）。
    """
    tag = el.tag
    # 注释节点等的 tag 不是 str；SKIP 里的标签直接整棵砍掉
    if not isinstance(tag, str) or tag in SKIP:
        return
    cls = el.get("class") or ""

    if "md-diagram-panel" in cls:
        lab = diagram_labels(el)
        if lab:
            # 前后各留一个空行，让图在产物里是独立段落（也便于按 [DIAGRAM] grep）
            lines.append("")
            lines.append("[DIAGRAM] " + lab)
            lines.append("")
        return

    # `md-fences` 是 Typora 的代码块容器；裸 `pre` 是兜底
    if "md-fences" in cls or tag == "pre":
        lines.append("")
        lines.append("```")
        # rstrip：CodeMirror 每行自带换行，不清掉围栏前会多出空行
        lines.append(extract_code(el).rstrip("\n"))
        lines.append("```")
        lines.append("")
        return

    if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
        lines.append("")
        # ``h3`` → ``###``：直接拿标签名的数字部分当层级，与原文层级一致
        lines.append("#" * int(tag[1]) + " " + txt(el))
        lines.append("")
        return

    if tag == "p":
        t = txt(el)
        for s in el.xpath(".//img/@src"):
            # typora uses data: URIs for some, skip those
            if not s.startswith("data:"):
                lines.append("[IMG] " + s)
        if t:
            lines.append(t)
        return

    if tag == "li":
        lines.append("- " + txt(el))
        return

    if tag == "table":
        # 只按行拼 ``| cell | cell |``，**不补 Markdown 的分隔行**（``|---|``）——
        # 产物是给人/grep 读的，不是给 Markdown 渲染器用的
        for tr in el.xpath(".//tr"):
            lines.append("| " + " | ".join(txt(c) for c in tr.xpath("./td|./th")) + " |")
        return

    if tag == "blockquote":
        lines.append("> " + txt(el))
        return

    if tag == "hr":
        lines.append("\n---\n")
        return

    for child in el:
        walk(child)


walk(root)
# 压掉 3 个以上连续换行：上面每条规则都在自己前后补空行，
# 叠起来会产出大片空行，让产物行数与课案行号对不上
res = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
with open(OUT, "w", encoding="utf-8") as f:
    f.write(res)

# code-only companion (useful for grepping config/env/API surface)
# 从**已生成的全文**里再抠代码块，而不是重走一遍 DOM：
# 这样两份产物天然一致，不会出现「全文里有、代码块里没有」的漂移
blocks = re.findall(r"```\n(.*?)\n```", res, re.S)
with open(OUTC, "w", encoding="utf-8") as f:
    f.write("\n\n".join(f"===== BLOCK {i+1} =====\n{b}" for i, b in enumerate(blocks)))

print("text chars:", len(res), "lines:", res.count("\n") + 1)
print("code blocks:", len(blocks), "code chars:", sum(len(b) for b in blocks))

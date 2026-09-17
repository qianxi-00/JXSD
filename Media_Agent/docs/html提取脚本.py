# -*- coding: utf-8 -*-
"""Extract readable text + code from the Typora-exported HTML courseware (v2).

- skips mermaid diagram CSS noise, keeps diagram node/edge labels as text
- unwraps CodeMirror code fences into plain ``` blocks
"""
import re
from lxml import html as LH

SRC = r"F:\ProGramApp\DSH\attachments\v1\files\e3\e30e09c2e27346090d8d2c3fdfaaea807072e0f2f53db596bb278907e067bedd\自媒体Agent.html"
OUT = r"F:\ProGram\Python_Base\_html_parse\courseware.txt"
OUTC = r"F:\ProGram\Python_Base\_html_parse\courseware_code_only.txt"

with open(SRC, "rb") as f:
    data = f.read()

doc = LH.fromstring(data)
write = doc.xpath('//div[@id="write"]')
root = write[0] if write else (doc.body if doc.body is not None else doc)

lines = []
SKIP = {"style", "script", "svg", "noscript"}


def norm(t: str) -> str:
    t = t.replace("\u00a0", " ").replace("\u200b", "")
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def txt(el) -> str:
    return norm(el.text_content())


def extract_code(pre) -> str:
    ln = pre.xpath('.//pre[contains(@class,"CodeMirror-line")]')
    if ln:
        return "\n".join(x.text_content() for x in ln)
    cml = pre.xpath('.//*[contains(@class,"cm-line")]')
    if cml:
        return "\n".join(x.text_content() for x in cml)
    return pre.text_content()


def diagram_labels(panel):
    """Compact text of a rendered mermaid diagram: node labels in document order."""
    labels = []
    for t in panel.xpath(".//*[local-name()='text'] | .//*[local-name()='span']"):
        s = norm(t.text_content())
        if s and s not in labels:
            labels.append(s)
    return " -> ".join(labels) if labels else ""


def walk(el):
    tag = el.tag
    if not isinstance(tag, str) or tag in SKIP:
        return
    cls = el.get("class") or ""

    if "md-diagram-panel" in cls:
        lab = diagram_labels(el)
        if lab:
            lines.append("")
            lines.append("[DIAGRAM] " + lab)
            lines.append("")
        return

    if "md-fences" in cls or tag == "pre":
        lines.append("")
        lines.append("```")
        lines.append(extract_code(el).rstrip("\n"))
        lines.append("```")
        lines.append("")
        return

    if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
        lines.append("")
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
res = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
with open(OUT, "w", encoding="utf-8") as f:
    f.write(res)

# code-only companion (useful for grepping config/env/API surface)
blocks = re.findall(r"```\n(.*?)\n```", res, re.S)
with open(OUTC, "w", encoding="utf-8") as f:
    f.write("\n\n".join(f"===== BLOCK {i+1} =====\n{b}" for i, b in enumerate(blocks)))

print("text chars:", len(res), "lines:", res.count("\n") + 1)
print("code blocks:", len(blocks), "code chars:", sum(len(b) for b in blocks))

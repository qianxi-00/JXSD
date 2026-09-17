"""把 Markdown 里的 mermaid 图抽出来，做成一个能在浏览器里**真渲染**的校验页。

为什么需要它：mermaid 语法错误在 Markdown 预览里往往只表现为"这块不显示"或
"显示成一段带箭头的纯文本"——不真渲染一遍很难发现。而架构图恰恰是最容易写错的
（节点先引用后定义、标签里混了 `**` 加粗、标签里出现裸的 ASCII 双引号、
子图连子图导致布局塌成一整块空白……这些都是实测踩过的）。

用法：
    .venv\\Scripts\\python.exe RAG\\script\\check_mermaid.py                    # 默认查 RAG/README.md
    .venv\\Scripts\\python.exe RAG\\script\\check_mermaid.py 某文档.md
    .venv\\Scripts\\python.exe RAG\\script\\check_mermaid.py --mermaid-js <mermaid.min.js 路径>

产物：`.dsh_tmp/mermaid/check.html`（连同 mermaid.min.js 一起），
用浏览器打开它，页面顶部会写出"渲染成功 N/M 张"以及每张图对应的小标题。

mermaid.min.js 从哪来：本机 DSH 的 Web 前端里带了一份（默认路径见 DEFAULT_MERMAID_JS）。
换机器时用 `--mermaid-js` 指一份本地 mermaid，或从任意 CDN 下一份放旁边。
"""

import argparse
import html
import re
import shutil
from pathlib import Path

# --- 路径引导：与 app\main.py 同一套三段式样板 ---
_BASE = Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent

DEFAULT_README = _BASE / "RAG" / "README.md"
OUT_DIR = _BASE / ".dsh_tmp" / "mermaid"
# DSH Web 前端自带的 mermaid（本机实测可用；换机器时用参数指定）
DEFAULT_MERMAID_JS = Path(r"F:\ProGramApp\DSH\profiles\web\node_modules\mermaid\dist\mermaid.min.js")

PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>mermaid 渲染校验</title>
<style>
  body {{ background:#fff; color:#111; font-family: system-ui, sans-serif; margin: 20px; }}
  section {{ margin-bottom: 28px; border-top: 1px solid #ddd; padding-top: 10px; }}
  h3 {{ font-size: 14px; color: #444; }}
  #result {{ font-size: 14px; white-space: pre-wrap; padding: 8px; background: #f5f5f5; }}
  .err {{ color: #b00; }}
  .ok {{ color: #060; }}
</style>
</head>
<body>
<div id="result">渲染中…</div>
{blocks}
<script src="mermaid.min.js"></script>
<script>
  const problems = [];
  window.addEventListener('error', e => problems.push('window.onerror: ' + e.message));
  (async () => {{
    try {{
      mermaid.initialize({{ startOnLoad: false, securityLevel: 'loose' }});
      await mermaid.run({{ querySelector: '.mermaid' }});
      const rendered = document.querySelectorAll('.mermaid svg').length;
      const total = document.querySelectorAll('.mermaid').length;
      const lines = [`渲染成功 ${{rendered}}/${{total}} 张`, ...problems];
      if (rendered !== total) lines.push('有图没渲染成 svg —— 对照下面各节，哪一节还是源码文本就是哪张图有问题');
      const box = document.getElementById('result');
      box.textContent = lines.join('\\n');
      box.className = rendered === total ? 'ok' : 'err';
    }} catch (err) {{
      const box = document.getElementById('result');
      box.textContent = '渲染抛异常: ' + err;
      box.className = 'err';
    }}
  }})();
</script>
</body>
</html>
"""


def extract_blocks(text: str) -> tuple[list[str], list[str]]:
    """抽出所有 ```mermaid 块，并为每块找到"它前面最近的一行标题"用于对号入座。"""
    blocks, titles = [], []
    for match in re.finditer(r"```mermaid\n(.*?)```", text, re.S):
        blocks.append(match.group(1))
        heading = ""
        for line in text[: match.start()].splitlines()[::-1]:
            if line.strip().startswith("#"):
                heading = line.strip()
                break
        titles.append(heading)
    return blocks, titles


def main() -> int:
    parser = argparse.ArgumentParser(description="真渲染校验 Markdown 里的 mermaid 图")
    parser.add_argument("target", nargs="?", default=str(DEFAULT_README), help="要检查的 Markdown")
    parser.add_argument("--mermaid-js", default=str(DEFAULT_MERMAID_JS), help="本地 mermaid.min.js 路径")
    parser.add_argument("--out", default=str(OUT_DIR), help="产物目录")
    args = parser.parse_args()

    target = Path(args.target)
    if not target.is_absolute():
        target = _BASE / target
    if not target.exists():
        print(f"[fail] 找不到 {target}")
        return 1

    blocks, titles = extract_blocks(target.read_text(encoding="utf-8"))
    if not blocks:
        print("[fail] 该文件里没有 mermaid 块")
        return 1

    mermaid_js = Path(args.mermaid_js)
    if not mermaid_js.exists():
        print(f"[fail] 找不到 mermaid.min.js：{mermaid_js}（用 --mermaid-js 指定）")
        return 1
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(mermaid_js, out_dir / "mermaid.min.js")

    sections = "\n".join(
        f'<section><h3>{html.escape(titles[i])}（第 {i + 1} 张）</h3>'
        f'<pre class="mermaid">{html.escape(blocks[i])}</pre></section>'
        for i in range(len(blocks))
    )
    (out_dir / "check.html").write_text(PAGE.format(blocks=sections), encoding="utf-8")

    print(f"{target}: 找到 {len(blocks)} 个 mermaid 块")
    print(f"产物: {out_dir / 'check.html'}")
    print("用浏览器打开它，页顶会显示「渲染成功 N/M 张」")
    for i, title in enumerate(titles, 1):
        print(f"  {i}. {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

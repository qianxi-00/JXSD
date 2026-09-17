# -*- coding: utf-8 -*-
r"""
Agent 课程 Notebook 工具链：percent 源 ↔ .ipynb 转换 / 结构检查 / 覆盖率审计
================================================================
为什么需要这个文件（而不是手写 .ipynb JSON）：

    .ipynb 是 JSON，手写/手改极易出错（cell 结构、id、metadata、换行），
    而且 diff 时几乎不可读。所以本项目的做法是：

        作者写 **percent 格式的 .py**（Jupytext 通用约定）
            ↓  nbtool.py py2nb
        生成 .ipynb（唯一入库产物）

    percent 格式长这样（`# %%` = 一个 code cell，`# %% [markdown]` = markdown cell）：

        # %% [markdown]
        # # 标题
        # 正文……
        #
        # | 列1 | 列2 |

        # %%
        print("hello")

    markdown cell 里的每一行都必须是注释（`#` 开头），`#` 后面**最多**去掉一个空格，
    这样 `#   - 子项` 会得到 `  - 子项`（保留缩进），符合 Markdown 的嵌套写法。

子命令：

    py2nb    SRC [DST]        单个转换；DST 省略时写成同名 .ipynb
    py2nb    --all            批量转换 scratch 源目录下的全部 .py
    nb2py    NB [DST]         反向还原成 percent 源（用于批量编辑 / 纯文本审阅）
    strip    PATH... | --all  清空执行输出与 execution_count（提交前统一跑）
    check    PATH... | --all  结构检查（内核名 / 首格 / 运行条件 / 引导格 / 语法 / 绝对路径泄漏）
    coverage                  覆盖率审计：162 个源 .py 是否都被 notebook 吸收

运行方式（仓库根目录下）：
    & .\.venv\Scripts\python.exe Agent\_tools\nbtool.py check --all
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

import nbformat

sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文报错

REPO_ROOT = Path(__file__).resolve().parents[2]     # Agent/_tools/nbtool.py → 仓库根
AGENT = REPO_ROOT / "Agent"
SRC_ROOT = Path(r"F:\ProGramApp\DSH_Temporary\agent_nb\src")   # percent 源（仓库外暂存区）
PY_SOURCE = AGENT / "_py_source"                    # 归档后的原 .py 课案
MANIFEST = AGENT / "_tools" / "notebook_manifest.json"

# 内核名必须是这个：机器默认的 python3 指向 Anaconda（依赖冲突），
# .venv 自带的 spec 又用裸 python（落到 Store 占位符），两个都不能用。
KERNEL_NAME = "python-base-agent"
KERNEL_DISPLAY = "Python (Python_Base .venv)"
PY_VERSION = "3.12.12"

CELL_RE = re.compile(r"^# %%(?:\s*\[(?P<tags>[^\]]*)\])?\s*$")
# 引导格的特征：把仓库根塞进 sys.path 并 chdir —— 没有它 notebook 里 import config 必失败
BOOTSTRAP_MARKERS = ("sys.path.insert", "config.py")
# 运行条件区块的标记（模板强制要求三档之一）
TIER_MARKERS = ("🟢", "🟡", "🔴")

CODE_NB = "code"
MD_NB = "markdown"


# ================================================================
# percent 源解析
# ================================================================
def parse_percent(text: str) -> list[tuple[str, list[str]]]:
    """把 percent 格式的 .py 解析成 [(cell 类型, 行列表), ...]。

    规则：
      · `# %%` 开启一个 code cell；`# %% [markdown]` 开启一个 markdown cell；
      · 文件必须**以 cell 标记开头**，第一个标记之前的非空行会直接报错
        （避免作者把文件头 docstring 放在标记外面而被静默丢掉）；
      · markdown cell 的每一行都必须是注释。
    """
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if lines and lines[0].startswith("\ufeff"):      # 去 BOM
        lines[0] = lines[0].lstrip("\ufeff")

    cells: list[tuple[str, list[str]]] = []
    kind: str | None = None
    buf: list[str] = []

    for raw in lines:
        m = CELL_RE.match(raw)
        if m:
            if kind is not None:
                cells.append((kind, buf))
            tags = (m.group("tags") or "").lower()
            kind = MD_NB if "markdown" in tags else CODE_NB
            buf = []
            continue
        if kind is None:
            if raw.strip():
                raise ValueError(
                    f"文件必须以 '# %%' 开头；这一行不在任何 cell 里：{raw[:70]!r}"
                )
            continue
        buf.append(raw)

    if kind is not None:
        cells.append((kind, buf))
    return cells


def md_uncomment(lines: list[str]) -> str:
    """markdown cell：每行必须是 `#` 注释，去掉 `#` 与**最多一个**空格。"""
    out: list[str] = []
    for i, ln in enumerate(lines, start=1):
        if not ln.strip():
            out.append("")
            continue
        if not ln.lstrip().startswith("#"):
            raise ValueError(f"markdown cell 第 {i} 行不是注释：{ln[:70]!r}")
        body = ln.lstrip()[1:]
        if body.startswith(" "):
            body = body[1:]
        out.append(body)
    while out and not out[0].strip():      # 去掉首尾空行
        out.pop(0)
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def cells_to_notebook(cells: list[tuple[str, list[str]]]) -> nbformat.NotebookNode:
    """把解析结果组装成 nbformat 4 的 notebook（含正确的 kernelspec）。"""
    nb = nbformat.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": KERNEL_DISPLAY,
            "language": "python",
            "name": KERNEL_NAME,
        },
        "language_info": {"name": "python", "version": PY_VERSION},
    }
    for kind, buf in cells:
        if kind == MD_NB:
            src = md_uncomment(buf)
            if not src.strip():
                continue                    # 空 markdown cell 直接丢
            nb.cells.append(nbformat.v4.new_markdown_cell(src))
        else:
            src = "\n".join(buf).strip("\n")
            nb.cells.append(nbformat.v4.new_code_cell(src))
    nbformat.validate(nb)
    return nb


def notebook_to_percent(nb: nbformat.NotebookNode) -> str:
    """反向：notebook → percent 源。与 py2nb 往返稳定。"""
    out: list[str] = []
    for cell in nb.cells:
        if cell.cell_type == MD_NB:
            out.append("# %% [markdown]")
            for ln in cell.source.split("\n"):
                out.append("#" if not ln.strip() else "# " + ln)
        elif cell.cell_type == CODE_NB:
            out.append("# %%")
            out.append(cell.source.rstrip("\n"))
        else:
            raise ValueError(f"不支持的 cell 类型：{cell.cell_type}")
        out.append("")
    return "\n".join(out).rstrip("\n") + "\n"


# ================================================================
# 读 / 写（统一 LF，避免 Windows 上写出 CRLF 让 diff 全是噪声）
# ================================================================
def write_notebook(nb: nbformat.NotebookNode, path: Path) -> None:
    """写出 .ipynb：用 nbformat 序列化（保证格式正确），但**按 LF 落盘**。"""
    text = nbformat.writes(nb, version=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def read_notebook(path: Path) -> nbformat.NotebookNode:
    with open(path, encoding="utf-8") as f:
        return nbformat.read(f, as_version=4)


def disp(path: Path) -> str:
    """显示用路径：在仓库内就显示相对路径，不在仓库内（如 scratch 暂存区）就显示绝对路径。

    踩坑记录：一开始直接写 `path.relative_to(REPO_ROOT)`，一旦目标是仓库外的
    scratch 目录就抛 `ValueError: ... is not in the subpath of ...` —— 而且是在
    **文件已经写成功之后**才炸，看起来像转换失败，其实只是打印挂了。
    """
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def all_notebooks() -> list[Path]:
    """仓库里全部的课程 notebook（排除归档区与 checkpoint）。"""
    found = []
    for p in AGENT.rglob("*.ipynb"):
        if "_py_source" in p.parts or ".ipynb_checkpoints" in p.parts:
            continue
        found.append(p)
    return sorted(found)


def all_scratch_sources() -> list[Path]:
    if not SRC_ROOT.exists():
        return []
    return sorted(p for p in SRC_ROOT.rglob("*.py") if not p.name.startswith("_"))


def target_of(src: Path) -> Path:
    """scratch 源 → 仓库里的 notebook 路径。

    scratch 结构镜像仓库：<SRC_ROOT>/<章>/<名>.py → <AGENT>/<章>/<名>.ipynb
    """
    rel = src.relative_to(SRC_ROOT)
    return (AGENT / rel).with_suffix(".ipynb")


# ================================================================
# 子命令实现
# ================================================================
def cmd_py2nb(args: argparse.Namespace) -> int:
    if args.all:
        sources = all_scratch_sources()
        if not sources:
            print(f"没有找到 scratch 源：{SRC_ROOT}")
            return 1
    else:
        sources = [Path(args.src)]

    failed = 0
    for src in sources:
        try:
            cells = parse_percent(src.read_text(encoding="utf-8"))
            nb = cells_to_notebook(cells)
            if args.all:
                dst = target_of(src)
            else:
                dst = Path(args.dst) if args.dst else src.with_suffix(".ipynb")
            write_notebook(nb, dst)
            n_md = sum(1 for c in nb.cells if c.cell_type == MD_NB)
            n_py = sum(1 for c in nb.cells if c.cell_type == CODE_NB)
            print(f"  ✓ {disp(dst)}  （markdown {n_md} + code {n_py}）")
        except Exception as exc:      # noqa: BLE001 —— 批量模式下要跑完再汇总
            failed += 1
            print(f"  ✗ {src}：{type(exc).__name__}: {exc}")
    if failed:
        print(f"\n失败 {failed} 个")
        return 1
    print(f"\n共转换 {len(sources)} 个 notebook")
    return 0


def cmd_nb2py(args: argparse.Namespace) -> int:
    nb = read_notebook(Path(args.nb))
    text = notebook_to_percent(nb)
    dst = Path(args.dst) if args.dst else Path(args.nb).with_suffix(".py")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8", newline="\n")
    print(f"  ✓ {dst}  （{len(nb.cells)} 个 cell）")
    return 0


def cmd_strip(args: argparse.Namespace) -> int:
    targets = all_notebooks() if args.all else [Path(p) for p in args.paths]
    changed = 0
    for p in targets:
        nb = read_notebook(p)
        dirty = False
        for cell in nb.cells:
            if cell.cell_type == CODE_NB:
                if cell.get("outputs"):
                    cell["outputs"] = []
                    dirty = True
                if cell.get("execution_count") is not None:
                    cell["execution_count"] = None
                    dirty = True
        nb.metadata.pop("widgets", None)
        if dirty:
            write_notebook(nb, p)
            changed += 1
    print(f"  共检查 {len(targets)} 个，清掉输出的 {changed} 个")
    return 0


def check_notebook(path: Path) -> list[str]:
    """结构检查。返回问题列表（空 = 通过）。"""
    problems: list[str] = []
    rel = disp(path)
    try:
        nb = read_notebook(path)
    except Exception as exc:      # noqa: BLE001
        return [f"{rel}: 无法解析（{type(exc).__name__}: {exc}）"]

    ks = (nb.metadata.get("kernelspec") or {})
    if ks.get("name") != KERNEL_NAME:
        problems.append(f"{rel}: kernelspec.name={ks.get('name')!r}，应为 {KERNEL_NAME!r}"
                        "（用错内核会跑到 Anaconda / Store 占位符上）")

    if nb.get("nbformat") != 4:
        problems.append(f"{rel}: nbformat={nb.get('nbformat')}，应为 4")

    if not nb.cells:
        return [f"{rel}: 没有任何 cell"]

    # 首格必须是 markdown 标题
    first = nb.cells[0]
    if first.cell_type != MD_NB:
        problems.append(f"{rel}: 首格不是 markdown（应是 `# 标题`）")
    elif not first.source.lstrip().startswith("# "):
        problems.append(f"{rel}: 首格 markdown 不是一级标题（应以 `# ` 开头）")

    md_text = "\n".join(c.source for c in nb.cells if c.cell_type == MD_NB)
    code_text = "\n".join(c.source for c in nb.cells if c.cell_type == CODE_NB)

    if "运行条件" not in md_text:
        problems.append(f"{rel}: 缺少「运行条件」区块")
    elif not any(t in md_text for t in TIER_MARKERS):
        problems.append(f"{rel}: 「运行条件」区块没有三档标记（🟢 离线 / 🟡 需模型 / 🔴 需外部服务）")

    if not all(m in code_text for m in BOOTSTRAP_MARKERS):
        problems.append(f"{rel}: 缺少环境引导格（要 chdir 到仓库根 + sys.path.insert，"
                        "否则 `from config import settings` 必失败）")

    n_md = sum(1 for c in nb.cells if c.cell_type == MD_NB)
    n_code = sum(1 for c in nb.cells if c.cell_type == CODE_NB)
    if n_code == 0:
        problems.append(f"{rel}: 没有 code cell")
    if n_md < 3:
        problems.append(f"{rel}: markdown cell 只有 {n_md} 个，讲解过少")

    # 每个 code cell 必须能编译；顺带查绝对路径泄漏
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != CODE_NB or not cell.source.strip():
            continue
        try:
            ast.parse(cell.source)
        except SyntaxError as exc:
            problems.append(f"{rel}: 第 {i} 个 cell 语法错误 → {exc.msg}（行 {exc.lineno}）")
        for bad in ("F:\\ProGram", "F:/ProGram", "C:\\Users\\QianXi"):
            if bad in cell.source:
                problems.append(f"{rel}: 第 {i} 个 cell 泄漏绝对路径 {bad!r}（应基于 Path.cwd() 推导）")

    # 有输出说明忘了 strip
    if any(c.get("outputs") for c in nb.cells if c.cell_type == CODE_NB):
        problems.append(f"{rel}: 还带着执行输出（提交前请跑 `nbtool.py strip --all`）")

    # markdown 被「多加一层 #」的静默损坏。
    # 起因：某个补输出的脚本把 percent 源的 `buf`（**已经是**带 `# ` 的注释行）
    # 又加了一次 `# `，三轮下来正文从 `# 标题` 变成 `# # # # 标题`。
    # ast.parse 查不出、关键字检查也查不出，但 notebook 里标题会带着一串字面 `#` 显示。
    # 单行 `# # xxx` 可能是作者有意（markdown 里 `# # x` 是「内容是 # x 的 H1」），
    # 所以只在**成规模出现**时才报（≥3 行基本可以断定是系统性损坏）。
    doubled = [ln for c in nb.cells if c.cell_type == MD_NB
               for ln in c.source.split("\n") if ln.startswith("# # ")]
    if len(doubled) >= 3:
        problems.append(f"{rel}: {len(doubled)} 行 markdown 以 '# # ' 开头，"
                        f"疑似被多加了一层 '#' 前缀（例：{doubled[0][:50]!r}）")

    return problems


def cmd_check(args: argparse.Namespace) -> int:
    targets = all_notebooks() if args.all else [Path(p) for p in args.paths]
    if not targets:
        print("没有找到 notebook（Agent/**/*.ipynb，已排除 _py_source）")
        return 1
    bad = 0
    for p in targets:
        problems = check_notebook(p)
        if problems:
            bad += 1
            for msg in problems:
                print(f"  ✗ {msg}")
    print(f"\n检查 {len(targets)} 个 notebook：通过 {len(targets) - bad}，有问题 {bad}")
    return 1 if bad else 0


# ---------------- 覆盖率审计 ----------------
# 模板强制要求从 notebook 里删掉的两类样板行（见 `_nb_template.md` 第 6 节第 1、2 条）。
# 它们「不在 notebook 里」是**预期行为**；算进覆盖率分母只会让每个文件永远差几个点，
# 反而掩盖真正被丢掉的代码。所以这里直接不计入指纹。
TEMPLATE_REMOVED_PREFIXES = (
    'if __name__ ==',
    'sys.stdout.reconfigure(',
)


def code_fingerprint(text: str) -> list[str]:
    """抽出一个 .py 的「有效代码行」：去掉注释、空行、docstring 与模板必删行。

    ⚠️ 踩坑记录 1：一开始只把 docstring 的**首行**行号记进排除集合，
    结果多行 docstring 的后续行全被当成代码行 —— 覆盖率一律偏低 30~40 个百分点，
    看起来像「转换时把代码弄丢了」，其实只是统计口径错了。
    正确做法是记下 docstring 节点的**完整行区间**（lineno ~ end_lineno）。

    ⚠️ 踩坑记录 2：本函数会把连续空白压成一个空格，所以比对时**必须**拿
    `normalized_lines()` 处理过的 notebook 文本，不能直接比原文 ——
    否则 `x = 1  # 注释`（注释前两个空格）永远匹配不上。
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                first = body[0]
                end = getattr(first, "end_lineno", first.lineno) or first.lineno
                docstring_lines.update(range(first.lineno, end + 1))

    out: list[str] = []
    for i, line in enumerate(text.split("\n"), start=1):
        if i in docstring_lines:
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        norm = re.sub(r"\s+", " ", stripped)
        if norm.startswith(TEMPLATE_REMOVED_PREFIXES):
            continue
        out.append(norm)
    return out


def source_root(manifest: dict) -> Path:
    """源 .py 的根目录：清单里写了 `_源根` 且该目录存在就用它，否则退回 Agent/。

    归档前 .py 在 `Agent/<章>/`，归档后在 `Agent/_py_source/<章>/` ——
    所以清单里的 sources 一律写成「相对源根」的形式，两种形态都能用。
    """
    name = manifest.get("_源根") or ""
    candidate = AGENT / name
    return candidate if name and candidate.is_dir() else AGENT


def normalized_lines(text: str) -> set[str]:
    """把一段代码切成「规格化行集合」，供指纹比对。

    必须两边都规格化：`code_fingerprint` 会把连续空白压成一个空格，
    如果拿它去比**未规格化**的原文，那 `x = 1  # 注释`（注释前两个空格）
    就永远匹配不上 —— 覆盖率会莫名其妙掉一大截。
    """
    out: set[str] = set()
    for line in text.split("\n"):
        s = line.strip()
        if s:
            out.add(re.sub(r"\s+", " ", s))
    return out


def cmd_coverage(args: argparse.Namespace) -> int:
    if not MANIFEST.exists():
        print(f"缺少映射清单：{MANIFEST}")
        return 1
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    notebooks = manifest["notebooks"]
    excluded: dict[str, str] = manifest.get("_排除") or {}
    src_root = source_root(manifest)

    # 收集每个 notebook 的代码文本（存成「规格化行集合」，与指纹同一口径）
    nb_code: dict[str, set[str]] = {}
    for rel_nb in [n["path"] for n in notebooks]:
        p = AGENT / rel_nb
        if not p.exists():
            nb_code[rel_nb] = set()
            continue
        nb = read_notebook(p)
        nb_code[rel_nb] = normalized_lines(
            "\n".join(c.source for c in nb.cells if c.cell_type == CODE_NB)
        )

    # 扫描源文件（相对源根；跳过 __pycache__、下划线开头的内部目录、tmp_* 运行产物）
    source_files = sorted(
        p for p in src_root.rglob("*.py")
        if "__pycache__" not in p.parts
        and not any(part.startswith("_") for part in p.relative_to(src_root).parts)
        and "tmp_" not in str(p.relative_to(src_root))
    )

    declared: dict[str, str] = {}
    for entry in notebooks:
        for src in entry["sources"]:
            declared[src] = entry["path"]

    print(f"源根：{src_root}")
    print(f"清单声明 {len(declared)} 个源文件；实际扫描到 {len(source_files)} 个\n")

    orphans = [p for p in source_files
               if str(p.relative_to(src_root)).replace("\\", "/") not in declared]

    rows = []
    missing_nb = []
    for src_rel, nb_rel in sorted(declared.items()):
        p = src_root / src_rel
        if not p.exists():
            missing_nb.append(src_rel)
            continue
        lines = code_fingerprint(p.read_text(encoding="utf-8"))
        if not lines:
            rows.append((src_rel, nb_rel, 0, 0.0, "无有效代码行"))
            continue
        hay = nb_code.get(nb_rel, set())
        hit = sum(1 for ln in lines if ln in hay)
        ratio = hit / len(lines)
        note = "" if ratio >= args.fail_under else "← 低于阈值"
        rows.append((src_rel, nb_rel, len(lines), ratio, note))

    worst = sorted((r for r in rows if r[4]), key=lambda r: r[3])[:25]
    if args.filter:
        # 指定了过滤词就只列匹配的行，且**不受「最差 25 个」限制** ——
        # 否则 subagent 查自己那几个文件时，它们常常不在最差 25 里，
        # 看不到数字就只能自己重写一遍覆盖率脚本（实测发生过）。
        picked = [r for r in rows if args.filter in r[0].replace("\\", "/")]
        if not picked:
            print(f"没有匹配 {args.filter!r} 的源文件")
        else:
            print(f"匹配 {args.filter!r} 的源文件（{len(picked)} 个）：")
            print("  {:<50} {:<40} {:>6} {:>7}".format("源文件", "目标 notebook", "代码行", "覆盖"))
            for src_rel, nb_rel, n, ratio, note in sorted(picked, key=lambda r: r[3]):
                print(f"  {src_rel:<50} {nb_rel:<40} {n:>6} {ratio:>6.1%} {note}")
    elif worst:
        print("覆盖率最低的 25 个源文件：")
        print("  {:<50} {:<40} {:>6} {:>7}".format("源文件", "目标 notebook", "代码行", "覆盖"))
        for src_rel, nb_rel, n, ratio, note in worst:
            print(f"  {src_rel:<50} {nb_rel:<40} {n:>6} {ratio:>6.1%} {note}")
    else:
        print("所有源文件覆盖率均达标 ✔")

    if orphans:
        print(f"\n⚠️ 未被任何 notebook 引用的源文件（{len(orphans)} 个）：")
        for p in orphans:
            rel = str(p.relative_to(src_root)).replace("\\", "/")
            reason = excluded.get(rel)
            print(f"    {rel}" + (f"\n        已登记为排除项：{reason}" if reason else "   ← 需要处理"))
    else:
        print("\n✔ 没有孤儿源文件")

    if missing_nb:
        print(f"\n⚠️ 清单里声明但源根下不存在的文件（{len(missing_nb)} 个）：")
        for s in missing_nb:
            print(f"    {s}")

    unaccounted = [p for p in orphans
                   if str(p.relative_to(src_root)).replace("\\", "/") not in excluded]
    total = len(rows)
    ok = sum(1 for r in rows if not r[4])
    print(f"\n合计：声明 {total} 个源文件，覆盖达标 {ok}，未达标 {total - ok}；"
          f"孤儿 {len(orphans)}（已登记排除 {len(orphans) - len(unaccounted)}，待处理 {len(unaccounted)}）")
    return 0 if (not unaccounted and ok == total and not missing_nb) else 1

    return 0 if (not orphans and ok == total and not missing_nb) else 1


# ---------------- 预期输出核对 ----------------
# 匹配 markdown 里的 `### 预期输出` + ```text 围栏。
#
# ⚠️ 中间允许夹说明段落（`.*?` + DOTALL）：第一版要求标题**紧挨着**围栏
# （`### 预期输出\n+```text`），结果「标题 → 一段解释 → 代码块」这种写法
# 整段静默漏检 —— 01_langgraph/02 的 subagent 自己发现并挪动了说明段才暴露出来。
# 漏检比误报危险得多：它让「没核对」看起来像「核对通过」。
EXPECT_RE = re.compile(r"###\s*预期输出[^\n]*\n.*?```text\n(.*?)```", re.DOTALL)

# 「本段输出不确定」的标注词。写 notebook 的人已经把易变内容标出来了
# （时间戳、随机 UUID、模型自己的措辞……），核对器必须认这些标注，
# 否则会把「本来就每次不同」的段落当成错误，一天到晚误报。
#
# 这些词是**从实际 notebook 里抄来的**，不是拍脑袋想的：第一版只写了
# 「每次运行不同 / 随机 / 非确定」，结果 02_langchain/02 那格明明标着
# 「这一格所有内容都由模型决定，是本课最不稳定的一格」却仍被判为不一致。
# 只收「明确声明输出不由代码决定」的措辞 —— 宁可漏判，
# 也不要用宽泛的词把真正的错误盖掉。
VOLATILE_RE = re.compile(
    r"每次(运行|执行)?[^\n]{0,8}(不同|不一样|会变)|随机|非确定|不确定|时间戳|会变|"
    r"视[^\n]{0,8}而定|实测值|本机跑出来|因机器而异|以你运行时为准|"
    r"由模型决定|最不稳定|不稳定|重跑一次|别逐字比对"
)

# **否定用法**要先剔除，否则会把「讲确定性」的句子反过来当成「声明不确定」。
# 实测踩到的例子（`03_deepagents/01` 的 subagent 报上来的）：
#   它在确定性输出的那一格写「所以它不带『不确定』声明」——
#   那三个字反而让整段被判定为「已声明非确定」而**跳过比对**。
#   看起来通过，其实根本没核。这是核对器最危险的一类错：把「没核」伪装成「核过了」。
VOLATILE_NEGATION_RE = re.compile(
    r"(?:不带|不含|不是|并非|没有|无需|不需要|不谈|不算|非)[^\n。；]{0,8}"
    r"(?:不确定|非确定|随机|会变|每次[^\n。；]{0,6}(?:不同|不一样)|时间戳|实测值)"
)


def is_declared_volatile(region: str) -> bool:
    """这段预期输出是否被**明确声明**为「不确定」。

    先剔掉否定用法再匹配（见 VOLATILE_NEGATION_RE 的说明）。
    """
    return bool(VOLATILE_RE.search(VOLATILE_NEGATION_RE.sub("", region)))


def block_region(text: str, start: int, end: int) -> str:
    """取「这一段的预期输出」的说明范围：从它自己的小标题，到下一个同级/更高级标题之前。

    用于按段判断「这段输出是不是被声明为不确定」。范围里包含标题、```text 块本身、
    以及紧跟其后的解释性段落（实际写法里「每次不同」这类说明通常写在块后面）。

    ⚠️ 两个位置都得算准，第一版两个都写错、直接把整段判成「无说明」：
      · 向后找下一个标题必须从 `end`（围栏结束）之后开始找 —— 从 `start` 开始的话，
        开头的 `### 预期输出` 自己就命中了 `^#{2,3}\\s`，`end` 立刻等于 `start`，范围成空；
      · 向前找本段标题用 `rfind("###", 0, start)`，取不到就退回 0。
    """
    head = text.rfind("###", 0, start)
    if head == -1:
        head = 0
    m = re.search(r"(?m)^#{2,3}\s", text[end:])
    stop = end + m.start() if m else len(text)
    return text[head:stop]


def norm_lines(text: str) -> list[str]:
    """把输出切成「非空且已去首尾空白」的行列表 —— 比对时忽略缩进与空行差异。"""
    return [ln.strip() for ln in text.split("\n") if ln.strip()]


def is_subsequence(needle: list[str], hay: list[str]) -> tuple[bool, list[str]]:
    """needle 的每一行是否**按顺序**出现在 hay 里。返回 (是否全中, 缺失的行)。"""
    pos, missing = 0, []
    for want in needle:
        found = False
        while pos < len(hay):
            if hay[pos] == want:
                pos += 1
                found = True
                break
            pos += 1
        if not found:
            pos = 0          # 没找到就不推进游标，避免后续行被连带判失败
            missing.append(want)
    return (not missing), missing


def cell_output_text(cell) -> str:
    parts = []
    for out in cell.get("outputs", []):
        kind = out.get("output_type")
        if kind == "stream":
            parts.append(out.get("text", ""))
        elif kind in ("execute_result", "display_data"):
            parts.append(out.get("data", {}).get("text/plain", ""))
        elif kind == "error":
            parts.append(f"{out.get('ename')}: {out.get('evalue')}")
    return "".join(parts)


def cmd_verify(args: argparse.Namespace) -> int:
    """执行 notebook，核对每个「预期输出」块是否真的能在实跑输出里找到。

    为什么需要它：模板要求每格输出后面写「预期输出」，但**写的人可能凭记忆编**。
    人工核对 32 个 notebook 不现实，所以这里自动化：把每段「预期输出」当成
    行序列，检查它是不是**紧邻的上一个 code cell** 实际输出的子序列。
    （用子序列而不是全等，是因为输出里常混着时间戳、路径、对象地址等易变内容。）
    """
    from nbclient import NotebookClient

    targets = all_notebooks() if args.all else [Path(p) for p in args.paths]
    if not targets:
        print("没有找到 notebook")
        return 1

    total_blocks = total_ok = skipped = 0
    bad_files = 0
    for path in targets:
        rel = disp(path)
        nb = read_notebook(path)
        client = NotebookClient(
            nb, timeout=args.timeout, kernel_name=KERNEL_NAME, allow_errors=True,
            startup_timeout=120,
            resources={"metadata": {"path": str(path.parent)}},
        )
        try:
            client.execute()
        except Exception as exc:      # noqa: BLE001
            print(f"  ✗ {rel}: 执行失败 {type(exc).__name__}: {str(exc)[:120]}")
            bad_files += 1
            continue

        last_out: list[str] = []
        problems = []
        for cell in nb.cells:
            if cell.cell_type == CODE_NB:
                text = cell_output_text(cell)
                if text.strip():
                    last_out = norm_lines(text)
                continue
            # 按「段」判定不确定，而不是按整格。
            # 踩坑记录（subagent 反馈出来的设计缺陷）：第一版用 `VOLATILE_RE.search(cell.source)`
            # 命中就跳过整格。可一个 markdown 格里完全可能同时放「确定性输出」和
            # 「模型输出」两段 —— 那样确定性那段也被一起跳过，真错误就被盖掉了。
            # 更糟的是会出现「假通过」：格子里若有句「不同版本这一串名字会变」讲的是
            # 中间件工具名，与模型输出无关，却让整格蒙混过关。
            # 所以改成：找出每段「预期输出」自己的说明范围（它的标题 → 下一个标题之前），
            # 只有那段范围里写了「不确定」才跳过。
            for block in EXPECT_RE.finditer(cell.source):
                want = norm_lines(block.group(1))
                if not want:
                    continue
                if is_declared_volatile(block_region(cell.source, block.start(), block.end())):
                    skipped += 1
                    continue
                total_blocks += 1
                ok, missing = is_subsequence(want, last_out)
                if ok:
                    total_ok += 1
                else:
                    problems.append((want, missing))

        if problems:
            bad_files += 1
            print(f"  ✗ {rel}: {len(problems)} 段「预期输出」与实跑不符")
            for want, missing in problems[:2]:
                print(f"      期望首行：{want[0][:90]}")
                for m in missing[:3]:
                    print(f"      实际输出里找不到：{m[:90]}")
        else:
            print(f"  ✓ {rel}")

    print(f"\n核对 {len(targets)} 个 notebook：「预期输出」共 {total_blocks} 段，"
          f"与实跑一致 {total_ok} 段，不一致 {total_blocks - total_ok} 段；"
          f"另有 {skipped} 段已声明为非确定输出（跳过比对）；有问题的文件 {bad_files} 个")
    return 1 if bad_files else 0


# ================================================================
# CLI
# ================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Agent 课程 notebook 工具链")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("py2nb", help="percent 源 → .ipynb")
    p1.add_argument("src", nargs="?")
    p1.add_argument("dst", nargs="?")
    p1.add_argument("--all", action="store_true", help=f"批量转换 {SRC_ROOT}")
    p1.set_defaults(func=cmd_py2nb)

    p2 = sub.add_parser("nb2py", help=".ipynb → percent 源")
    p2.add_argument("nb")
    p2.add_argument("dst", nargs="?")
    p2.set_defaults(func=cmd_nb2py)

    p3 = sub.add_parser("strip", help="清空执行输出")
    p3.add_argument("paths", nargs="*")
    p3.add_argument("--all", action="store_true")
    p3.set_defaults(func=cmd_strip)

    p4 = sub.add_parser("check", help="结构检查")
    p4.add_argument("paths", nargs="*")
    p4.add_argument("--all", action="store_true")
    p4.set_defaults(func=cmd_check)

    p5 = sub.add_parser("coverage", help="覆盖率审计")
    p5.add_argument("--fail-under", type=float, default=0.95,
                    help="单个源文件的代码行覆盖率阈值（默认 0.95）")
    p5.add_argument("--filter", default="",
                    help="只看路径包含该子串的源文件（如 03_deepagents），会列出全部匹配项而不是最差 25 个")
    p5.set_defaults(func=cmd_coverage)

    p6 = sub.add_parser("verify", help="执行并按「预期输出」块核对真实输出")
    p6.add_argument("paths", nargs="*")
    p6.add_argument("--all", action="store_true")
    p6.add_argument("--timeout", type=int, default=900, help="单格超时秒数（默认 900）")
    p6.set_defaults(func=cmd_verify)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=====================================================================================
文件：verify_all.py
作用：**一键校验 `Front_End/` 目录下的所有交付物**。

它做四件事（前三件是必须的，第四件是额外加强）：

  (a) 用 `py_compile` 编译本目录下所有 `.py` 文件 —— 检查 Python 语法。
  (b) 用标准库 `html.parser` 解析所有 `.html` 文件 —— 检查能被解析，
      并检查 HTML 里 `href` / `src` 引用的【本地相对路径文件确实存在】。
  (c) 对每一个 Streamlit 应用，用 `subprocess.Popen` 真正把它启动起来，
      轮询 `http://127.0.0.1:<port>/_stcore/health`（兼容 `/healthz`）
      直到返回 200 或 45 秒超时，然后结束进程。端口从 **8601** 开始递增，
      每个应用一个端口。
  (d) 【额外】用 `streamlit.testing.v1.AppTest` 在进程内把每个应用真正跑一遍，
      断言脚本执行过程中 **没有抛出任何异常**。
      —— 因为健康检查只能证明"服务器起来了"，不能证明"脚本跑通了"。
      另外，如果系统里装了 Node.js，还会用 `node --check` 校验所有 JS 语法。

最后打印汇总：「HTML N 个 / Python N 个 / Streamlit 应用 N 个，失败 0 个」。

运行方式（必须用工作区自带的虚拟环境）：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' `
        'F:\\ProGram\\Python_Base\\Front_End\\verify_all.py'

退出码：0 = 全部通过；1 = 有失败项。
=====================================================================================
"""

from __future__ import annotations

import html.parser
import os
import py_compile
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# 让中文输出在 Windows 控制台 / 重定向到文件时都不乱码
# ---------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

# ---------------------------------------------------------------------------
# 基本路径：全部基于本文件所在目录计算，不依赖当前工作目录
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

# 端口从 8601 开始递增，每个 Streamlit 应用一个端口
BASE_PORT = 8601
# 健康检查超时（秒）
HEALTH_TIMEOUT = 45

# ---------------------------------------------------------------------------
# 已知的"故意失效"的引用：
#   01_HTML/02_常用标签.html 里有一张故意写错路径的 <img>，用来演示 alt 文字的效果。
#   这是教学需要，不是 bug，所以显式列在白名单里。
# ---------------------------------------------------------------------------
ALLOW_MISSING_REFS = {
    "01_HTML/02_常用标签.html::img/这只图片并不存在.png",
}

# HTML 属性名的合法格式（用来发现"引号写错导致属性名被截断"这类问题）
VALID_ATTR_NAME = re.compile(r"^[A-Za-z_:][-A-Za-z0-9_:.]*$")
# 需要检查本地文件是否存在的属性
CHECKED_ATTRS = ("href", "src")
# 这些前缀表示"不是本地相对路径"，跳过检查
SKIP_PREFIXES = (
    "http://", "https://", "//", "mailto:", "tel:", "data:",
    "javascript:", "#", "ftp:",
)


# =============================================================================
# 一、HTML 解析器：收集 href / src，并检查属性名是否合法
# =============================================================================
class LinkCollector(html.parser.HTMLParser):
    """收集 HTML 里的 (标签, 属性, 值)，同时记录非法的属性名。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        # 列表元素：(标签名, 属性名, 属性值)
        self.refs: list[tuple[str, str, str]] = []
        # 列表元素：(标签名, 非法属性名)
        self.bad_attr_names: list[tuple[str, str]] = []
        # 记录解析过程中遇到的错误（html.parser 基本不会报错，因为 HTML 容错性强）
        self.errors: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            # ① 检查属性名是否合法 —— 能发现「引号写错导致属性被截断」这类问题
            if not VALID_ATTR_NAME.match(name):
                self.bad_attr_names.append((tag, name))

            # ② 收集 href / src
            if name in CHECKED_ATTRS and value:
                self.refs.append((tag, name, value.strip()))

    # 自闭合标签（<img ... />）走这个回调
    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def error(self, message: str) -> None:      # pragma: no cover - 极少触发
        self.errors.append(message)


def check_html_file(path: Path) -> tuple[list[str], int, int]:
    """
    解析一个 HTML 文件并校验它的本地引用。

    返回：(问题列表, 检查的引用数, 属性数)
    """
    problems: list[str] = []
    text = path.read_text(encoding="utf-8", errors="replace")

    parser = LinkCollector()
    try:
        parser.feed(text)
        parser.close()
    except Exception as exc:                    # noqa: BLE001 - 解析失败要报告而不是崩掉
        problems.append(f"HTML 解析失败：{type(exc).__name__}: {exc}")
        return problems, 0, 0

    for message in parser.errors:
        problems.append(f"解析器报错：{message}")

    # ---- 检查非法属性名 ----
    for tag, name in parser.bad_attr_names:
        problems.append(
            f"<{tag}> 上出现了不合法的属性名 {name!r}"
            f"（通常是因为属性值里的引号没配对，或用了中文引号）"
        )

    # ---- 检查本地引用是否存在 ----
    checked = 0
    for tag, attr, value in parser.refs:
        lowered = value.lower()
        if not value or lowered.startswith(SKIP_PREFIXES):
            continue

        # 去掉查询串和锚点：a.css?v=1#x  →  a.css
        clean = value.split("?", 1)[0].split("#", 1)[0]
        if not clean:
            continue

        # 绝对路径（以 / 开头）在直接用浏览器打开文件时本来就无效，
        # 也没有"本地文件"可查，所以跳过。
        if clean.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", clean):
            continue

        rel_key = f"{path.relative_to(ROOT).as_posix()}::{clean}"
        if rel_key in ALLOW_MISSING_REFS:
            # 已知的、故意写错的演示引用 —— 跳过
            continue

        target = (path.parent / clean).resolve()
        checked += 1
        if not target.exists():
            problems.append(
                f"<{tag} {attr}=\"{value}\"> 引用的本地文件不存在：{clean}"
            )

    return problems, checked, len(parser.refs)


# =============================================================================
# 二、JS 语法检查（可选，需要系统里有 Node.js）
# =============================================================================
def check_javascript(html_files: list[Path], js_files: list[Path], tmp_dir: Path) -> tuple[list[str], int]:
    """
    用 `node --check` 校验所有 JS 语法。

    包括两类：
      ① .html 里的内联 <script> 块（用正则抽出来，先剥掉 HTML 注释避免误匹配）；
      ② 独立的 .js 文件。

    如果系统里没有 Node.js，就跳过（返回 0 个检查项），不影响整体结论。
    """
    node = shutil.which("node")
    if node is None:
        return [], 0

    problems: list[str] = []
    checked = 0

    html_comment = re.compile(r"<!--.*?-->", re.S)
    script_re = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.S | re.I)

    jobs: list[tuple[str, str]] = []            # (显示名, JS 源码)
    for path in html_files:
        text = html_comment.sub("", path.read_text(encoding="utf-8", errors="replace"))
        for index, body in enumerate(script_re.findall(text)):
            if body.strip():
                jobs.append((f"{path.relative_to(ROOT).as_posix()}（内联 script #{index}）", body))
    for path in js_files:
        jobs.append((path.relative_to(ROOT).as_posix(), path.read_text(encoding="utf-8", errors="replace")))

    for index, (label, source) in enumerate(jobs):
        tmp_file = tmp_dir / f"js_check_{index}.js"
        tmp_file.write_text(source, encoding="utf-8")
        result = subprocess.run(
            [node, "--check", str(tmp_file)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        checked += 1
        if result.returncode != 0:
            detail = (result.stderr or "").strip().splitlines()
            head = " | ".join(detail[:4]) if detail else "未知错误"
            problems.append(f"JS 语法错误：{label} → {head}")

    return problems, checked


# =============================================================================
# 三、Streamlit 应用：启动 + 健康检查
# =============================================================================
def wait_for_health(port: int, process: subprocess.Popen, timeout: int) -> tuple[bool, str, float]:
    """
    轮询健康检查接口，直到返回 200 或超时。

    返回：(是否健康, 使用的端点或错误信息, 耗时秒数)
    """
    endpoints = ("/_stcore/health", "/healthz")
    start = time.perf_counter()

    while time.perf_counter() - start < timeout:
        # 进程提前退出说明启动失败，不用再等了
        if process.poll() is not None:
            return False, f"进程提前退出（退出码 {process.returncode}）", time.perf_counter() - start

        for endpoint in endpoints:
            url = f"http://127.0.0.1:{port}{endpoint}"
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status == 200:
                        return True, endpoint, time.perf_counter() - start
            except (urllib.error.URLError, OSError, ValueError):
                # 还没起来 / 端口未监听 —— 继续轮询
                pass
        time.sleep(0.4)

    return False, f"{timeout} 秒内未就绪", time.perf_counter() - start


def stop_process(process: subprocess.Popen) -> None:
    """先礼貌地 terminate，等 8 秒还不退出就 kill。"""
    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            process.kill()
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            pass
    except OSError:
        pass


def run_streamlit_app(app_path: Path, port: int, log_dir: Path) -> tuple[bool, str, str]:
    """
    启动一个 Streamlit 应用并做健康检查。

    返回：(是否成功, 摘要信息, 详细日志)
    """
    stderr_file = log_dir / f"stderr_{port}.log"
    # 每个应用一个独立的 stderr 文件，启动失败时把内容打印出来方便定位
    with stderr_file.open("w+", encoding="utf-8", errors="replace") as err_handle:
        command = [
            sys.executable, "-m", "streamlit", "run", str(app_path),
            "--server.headless", "true",
            "--server.port", str(port),
            "--browser.gatherUsageStats", "false",
        ]
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=err_handle,
                cwd=str(app_path.parent),      # 以应用所在目录为工作目录
            )
        except OSError as exc:
            return False, f"启动失败：{exc}", ""

        try:
            healthy, endpoint, elapsed = wait_for_health(port, process, HEALTH_TIMEOUT)
        finally:
            stop_process(process)

        err_handle.flush()
        try:
            err_handle.seek(0)
            stderr_text = err_handle.read()
        except OSError:
            stderr_text = ""

    if healthy:
        summary = f"端口 {port} 健康检查 200（{endpoint}），耗时 {elapsed:.2f} 秒"
        return True, summary, stderr_text

    summary = f"端口 {port} 启动失败：{endpoint}（耗时 {elapsed:.2f} 秒）"
    return False, summary, stderr_text


# =============================================================================
# 四、Streamlit 应用：用 AppTest 在进程内真正跑一遍脚本
# =============================================================================
def run_app_test(app_path: Path) -> tuple[bool, str]:
    """
    用 `streamlit.testing.v1.AppTest` 执行脚本，检查有没有抛异常。

    为什么需要这一步？因为 `/_stcore/health` 在服务器起来时就返回 200，
    而脚本是在浏览器建立会话后才执行的。所以健康检查通过 **不代表脚本没报错**。
    AppTest 会把脚本真正跑一遍，`at.exception` 里就是所有未捕获的异常。

    返回：(是否成功, 摘要信息)
    """
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError as exc:                  # pragma: no cover
        return True, f"跳过（本环境的 streamlit 没有 AppTest：{exc}）"

    try:
        app_test = AppTest.from_file(str(app_path), default_timeout=60)
        app_test.run()
    except Exception as exc:                    # noqa: BLE001 - 测试框架本身出错也要报告
        return False, f"AppTest 执行失败：{type(exc).__name__}: {exc}"

    exceptions = list(app_test.exception)
    if exceptions:
        detail = "; ".join(
            (item.value or "").strip().splitlines()[0] if item.value else "未知异常"
            for item in exceptions
        )
        return False, f"脚本抛出 {len(exceptions)} 个异常：{detail}"

    return True, "脚本执行无异常"


# =============================================================================
# 五、主流程
# =============================================================================
def main() -> int:
    print("=" * 78)
    print("Front_End 交付物一键校验")
    print("=" * 78)
    print(f"校验根目录：{ROOT}")
    print(f"Python 解释器：{sys.executable}")
    print()

    # 启动 Streamlit 时 PYTHONIOENCODING 保证子进程日志不乱码
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    failures: list[str] = []
    tmp_root = Path(tempfile.mkdtemp(prefix="fe_verify_"))
    pycache_dir = tmp_root / "pycache"
    js_dir = tmp_root / "js"
    log_dir = tmp_root / "logs"
    for folder in (pycache_dir, js_dir, log_dir):
        folder.mkdir(parents=True, exist_ok=True)

    try:
        # -----------------------------------------------------------------
        # 收集文件
        # -----------------------------------------------------------------
        html_files = sorted(
            path for path in ROOT.rglob("*.html")
            if "_media_output" not in path.parts
        )
        py_files = sorted(path for path in ROOT.rglob("*.py"))
        js_files = sorted(path for path in ROOT.rglob("*.js"))
        css_files = sorted(path for path in ROOT.rglob("*.css"))
        md_files = sorted(path for path in ROOT.rglob("*.md"))

        # Streamlit 应用：
        #   · 只找 04_Streamlit 目录下的 .py
        #   · 排除 pages/ 子目录里的文件（它们是子页面，由入口注册）
        #   · 排除纯数据/工具模块（data.py）
        streamlit_apps: list[Path] = []
        streamlit_dir = ROOT / "04_Streamlit"
        if streamlit_dir.is_dir():
            for path in sorted(streamlit_dir.rglob("*.py")):
                if "pages" in path.relative_to(streamlit_dir).parts:
                    continue
                if path.name in {"data.py"}:
                    continue
                streamlit_apps.append(path)

        print(f"HTML 文件：{len(html_files)} 个")
        print(f"Python 文件：{len(py_files)} 个")
        print(f"JS 文件：{len(js_files)} 个　CSS 文件：{len(css_files)} 个　Markdown：{len(md_files)} 个")
        print(f"Streamlit 应用：{len(streamlit_apps)} 个")
        print()

        # =================================================================
        # (a) Python 语法检查
        # =================================================================
        print("-" * 78)
        print("(a) Python 语法检查（py_compile）")
        print("-" * 78)
        py_ok = 0
        for index, path in enumerate(py_files):
            rel = path.relative_to(ROOT).as_posix()
            cfile = pycache_dir / f"{index}_{path.stem}.pyc"
            try:
                py_compile.compile(str(path), cfile=str(cfile), doraise=True)
                py_ok += 1
                print(f"  [OK]   {rel}")
            except py_compile.PyCompileError as exc:
                failures.append(f"[Python] {rel} 编译失败")
                print(f"  [FAIL] {rel}")
                print(f"         {str(exc).strip().splitlines()[-1]}")
            except OSError as exc:
                failures.append(f"[Python] {rel} 无法编译：{exc}")
                print(f"  [FAIL] {rel} → {exc}")
        print(f"  → Python 编译通过 {py_ok} / {len(py_files)}")
        print()

        # =================================================================
        # (b) HTML 解析 + 本地引用检查
        # =================================================================
        print("-" * 78)
        print("(b) HTML 解析 + 本地引用检查（html.parser）")
        print("-" * 78)
        html_ok = 0
        total_refs = 0
        for path in html_files:
            rel = path.relative_to(ROOT).as_posix()
            problems, checked, refs = check_html_file(path)
            total_refs += refs
            if problems:
                failures.append(f"[HTML] {rel} 有 {len(problems)} 个问题")
                print(f"  [FAIL] {rel}")
                for problem in problems:
                    print(f"         · {problem}")
            else:
                html_ok += 1
                print(f"  [OK]   {rel}（引用 {refs} 个，其中本地文件 {checked} 个）")
        print(f"  → HTML 通过 {html_ok} / {len(html_files)}，共检查引用 {total_refs} 个")
        print()

        # =================================================================
        # (b2) JS 语法检查（可选，需要 Node.js）
        # =================================================================
        print("-" * 78)
        print("(b2) JavaScript 语法检查（node --check，可选）")
        print("-" * 78)
        js_problems, js_checked = check_javascript(html_files, js_files, js_dir)
        if js_checked == 0:
            print("  [SKIP] 系统里没有 Node.js，跳过 JS 语法检查（不影响其它结论）")
        else:
            for problem in js_problems:
                failures.append(f"[JS] {problem}")
                print(f"  [FAIL] {problem}")
            if not js_problems:
                print(f"  [OK]   {js_checked} 段 JavaScript 全部通过语法检查")
            print(f"  → JS 通过 {js_checked - len(js_problems)} / {js_checked}")
        print()

        # =================================================================
        # (c) Streamlit 应用启动 + 健康检查
        # =================================================================
        print("-" * 78)
        print("(c) Streamlit 应用启动 + 健康检查（subprocess.Popen）")
        print("-" * 78)
        app_ok = 0
        for index, app_path in enumerate(streamlit_apps):
            port = BASE_PORT + index
            rel = app_path.relative_to(ROOT).as_posix()
            print(f"  [..] 启动 {rel}（端口 {port}）……", flush=True)
            success, summary, stderr_text = run_streamlit_app(app_path, port, log_dir)
            if success:
                app_ok += 1
                print(f"  [OK]   {rel} → {summary}")
            else:
                failures.append(f"[Streamlit] {rel} 启动失败")
                print(f"  [FAIL] {rel} → {summary}")
                # 启动失败时把 stderr 打出来，方便定位
                if stderr_text.strip():
                    print("         ---- 子进程 stderr ----")
                    for line in stderr_text.strip().splitlines()[-25:]:
                        print(f"         {line}")
                    print("         ------------------------")
        print(f"  → Streamlit 启动成功 {app_ok} / {len(streamlit_apps)}")
        print()

        # =================================================================
        # (d) 额外：AppTest 真正执行脚本，检查有没有异常
        # =================================================================
        print("-" * 78)
        print("(d) 额外加强：用 AppTest 真正执行每个应用的脚本（检查 Traceback）")
        print("-" * 78)
        # AppTest 在"裸模式"下运行脚本，Streamlit 会为每次执行打印一行
        #   WARNING ... missing ScriptRunContext! This warning can be ignored ...
        # 这是**正常现象**（消息自己也说了可以忽略），但它会污染校验报告。
        # 用 logging.disable() 在【本阶段内】临时静音 WARNING 及以下级别 ——
        # 注意只给 logger 设 level 是没用的，Streamlit 的子 logger 已显式设过级别，
        # logging.disable 作用于 logging 模块全局，一定能生效。
        # 阶段结束后会恢复原来的设置。
        import logging

        _previous_disable_level = logging.root.manager.disable
        logging.disable(logging.WARNING)

        apptest_ok = 0
        try:
            for app_path in streamlit_apps:
                rel = app_path.relative_to(ROOT).as_posix()
                success, summary = run_app_test(app_path)
                if success:
                    apptest_ok += 1
                    print(f"  [OK]   {rel} → {summary}")
                else:
                    failures.append(f"[AppTest] {rel} {summary}")
                    print(f"  [FAIL] {rel} → {summary}")
        finally:
            # 恢复日志设置，不影响后续输出
            logging.disable(_previous_disable_level)

        print(f"  → 脚本执行无异常 {apptest_ok} / {len(streamlit_apps)}")
        print()

        # =================================================================
        # 汇总
        # =================================================================
        print("=" * 78)
        print("校验汇总")
        print("=" * 78)
        if failures:
            print(f"发现 {len(failures)} 个问题：")
            for index, item in enumerate(failures, start=1):
                print(f"  {index}. {item}")
            print()

        print(
            f"HTML {len(html_files)} 个 / Python {len(py_files)} 个 / "
            f"Streamlit 应用 {len(streamlit_apps)} 个，失败 {len(failures)} 个"
        )
        print("=" * 78)

        return 0 if not failures else 1

    finally:
        # 清理临时目录
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

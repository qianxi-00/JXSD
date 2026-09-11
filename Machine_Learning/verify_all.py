"""Machine_Learning 目录一键验证脚本：逐个运行本目录下所有课案脚本并汇总结果。

对应课案章节
    不直接对应课案知识点，属于本目录的工程化配套工具（自检 / 回归测试用）。

本节知识点
    1. 用 pathlib 递归发现本目录下所有 .py 脚本（自动跳过工具/隐藏目录和本文件）
    2. 用 subprocess.run 隔离运行每个脚本：捕获 stdout / stderr、限制超时、拿到返回码
    3. 统一把子进程的输出按 UTF-8 解码（PYTHONUTF8=1 + encoding="utf-8"），避免中文乱码
    4. 用"返回码 + 是否出现真正的 Traceback/异常回显行"双重判定成功与否
       （注意区分"脚本报错"和"脚本正文里讲解 Warning 这个词"，后者不算失败）
    5. 汇总打印「共 N 个脚本，成功 N 个，失败 0 个」，并在有失败时以非 0 退出码结束

运行方式（PowerShell，路径含中文必须加引号并用 & 调用）
    只打印汇总：
        & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Machine_Learning\\verify_all.py'
    同时把完整输出写入 VERIFY_REPORT.md（UTF-8）：
        & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Machine_Learning\\verify_all.py' --report

    说明：请用虚拟环境里的解释器运行本脚本，脚本内部用 sys.executable 调用子脚本，
          因此子脚本也会使用同一个解释器（不会误用系统 Python）。
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import pathlib
import re
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# 基本配置
# ---------------------------------------------------------------------------
ML_DIR = pathlib.Path(__file__).resolve().parent          # Machine_Learning 目录
REPORT_PATH = ML_DIR / "VERIFY_REPORT.md"                 # 报告输出位置
TAIL_LINES = 15                                           # 每个脚本只展示最后 15 行输出
TIMEOUT_SECONDS = 180                                     # 单个脚本的超时时间（秒）

# 这些名字开头的目录/文件不参与验证：
#   以 _ 开头     —— 工具 / 私有目录
#   以 . 开头     —— 隐藏目录，如 .ipynb_checkpoints、__pycache__
SKIP_DIR_PREFIXES = ("_", ".")
SKIP_FILE_PREFIXES = ("_", ".")
SKIP_FILE_NAMES = {"verify_all.py"}

# 判定真实异常的两个精确模式（比"关键字包含"可靠得多）：
#   1) Python 崩溃时一定会打印的固定首行；
#   2) 真正的异常/警告回显行 —— 行首就是异常类名加冒号（如 "ValueError: xxx"、
#      "sklearn.exceptions.ConvergenceWarning: xxx"），或 "C:\...\file.py:12: FutureWarning: ..."。
# 这样设计是为了避免误伤：很多脚本会在正文里**讲解** "AttributeError"、"DeprecationWarning"
# 这类词（作为教学内容），那属于正常输出，不能当成失败。
TRACEBACK_RE = re.compile(r"Traceback \(most recent call last\)")
ERROR_LINE_RE = re.compile(
    r"^\s*(?:\S+\.py:\d+:\s*)?[A-Za-z_][\w.]*(?:Error|Exception|Warning|Interrupt)\s*:"
)
# 仅作提示（不参与判定）的关键字：出现在正文里说明"这里提到了某种警告"
MENTION_KEYWORDS = ("Warning", "警告", "Traceback")


def discover_scripts() -> list[pathlib.Path]:
    """递归发现所有待验证脚本，按相对路径排序，保证每次运行顺序一致。"""
    scripts: list[pathlib.Path] = []
    for path in ML_DIR.rglob("*.py"):
        rel_parts = path.relative_to(ML_DIR).parts
        # 跳过工具目录（以 _ 或 . 开头）下的文件
        if any(part.startswith(SKIP_DIR_PREFIXES) for part in rel_parts[:-1]):
            continue
        name = path.name
        if name in SKIP_FILE_NAMES or name.startswith(SKIP_FILE_PREFIXES):
            continue
        scripts.append(path)
    return sorted(scripts, key=lambda p: str(p.relative_to(ML_DIR)).lower())


def build_env() -> dict[str, str]:
    """构造子进程环境变量：强制 UTF-8 输出 + 无界面绘图后端。"""
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"            # 让子进程 stdout 用 UTF-8，中文不乱码
    env["PYTHONIOENCODING"] = "utf-8"  # 双保险
    env["MPLBACKEND"] = "Agg"          # 无界面后端，避免任何弹窗阻塞
    env.pop("PYTHONPATH", None)        # 避免外部 PYTHONPATH 干扰
    return env


def run_one(script: pathlib.Path, env: dict[str, str]) -> dict[str, object]:
    """运行单个脚本，返回结构化结果。"""
    rel = script.relative_to(ML_DIR)
    cmd = [sys.executable, str(script)]
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=TIMEOUT_SECONDS,
            env=env,
            cwd=str(ML_DIR),          # 故意用一个"中立"的工作目录，检验脚本不依赖 cwd
        )
        returncode = completed.returncode
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        timed_out = False
    except subprocess.TimeoutExpired as exc:                 # pragma: no cover - 超时分支
        returncode = -1
        stdout = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = f"超时：脚本运行超过 {TIMEOUT_SECONDS} 秒已被强制终止"
        timed_out = True

    elapsed = time.perf_counter() - start
    combined = stdout + "\n" + stderr
    lines = combined.splitlines()
    traceback_hits = [ln for ln in lines if TRACEBACK_RE.search(ln)]
    error_hits = [ln for ln in lines if ERROR_LINE_RE.match(ln)]
    mention_lines = [ln for ln in lines
                     if any(kw in ln for kw in MENTION_KEYWORDS)
                     and ln not in error_hits]
    ok = (returncode == 0) and not traceback_hits and not error_hits and not timed_out

    return {
        "rel": rel,
        "cmd": cmd,
        "returncode": returncode,
        "elapsed": elapsed,
        "stdout": stdout,
        "stderr": stderr,
        "traceback_hits": traceback_hits,
        "error_hits": error_hits,
        "mention_lines": mention_lines,
        "timed_out": timed_out,
        "ok": ok,
        "tail": (stdout.rstrip().splitlines() or [""])[-TAIL_LINES:],
        "stderr_tail": (stderr.rstrip().splitlines() or [""])[-TAIL_LINES:],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="逐个运行 Machine_Learning 目录下所有课案脚本并汇总结果（全中文输出）"
    )
    parser.add_argument("--report", action="store_true",
                        help="把完整输出同时写入 Machine_Learning/VERIFY_REPORT.md")
    args = parser.parse_args()

    out: list[str] = []

    def emit(text: str = "") -> None:
        """既打印到控制台，也收集到 out 列表（用于写报告）。"""
        print(text)
        out.append(text)

    scripts = discover_scripts()
    started_at = _dt.datetime.now()

    emit("=" * 88)
    emit("《机器学习》课案代码目录 · 一键验证报告")
    emit("=" * 88)
    emit(f"虚拟环境解释器 : {sys.executable}")
    emit(f"Python 版本    : {sys.version.split()[0]}")
    emit(f"工作目录       : {ML_DIR}")
    emit(f"开始时间       : {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    emit(f"待运行脚本数   : {len(scripts)} 个（自动跳过工具/隐藏目录与本脚本）")
    emit(f"单脚本超时限制 : {TIMEOUT_SECONDS} 秒")
    emit("")
    emit("对每个脚本实际执行的命令（cwd 固定为 Machine_Learning/）：")
    for i, script in enumerate(scripts, 1):
        emit(f"  [{i:02d}] & '{sys.executable}' '{script}'")
    emit("")

    results: list[dict[str, object]] = []
    t_all = time.perf_counter()

    for i, script in enumerate(scripts, 1):
        rel = script.relative_to(ML_DIR)
        emit("=" * 88)
        emit(f"[{i}/{len(scripts)}] 运行：{rel}")
        emit("-" * 88)
        print(f"    运行中…… {rel}", flush=True)   # 进度提示（报告里不重复记录）

        result = run_one(script, build_env())
        results.append(result)

        emit(f"返回码：{result['returncode']}    耗时：{result['elapsed']:.2f} 秒    "
             f"判定：{'✅ 通过' if result['ok'] else '❌ 失败'}")
        emit(f"输出最后 {TAIL_LINES} 行：")
        for line in result["tail"]:                      # type: ignore[union-attr]
            emit(f"    | {line}")
        if result["stderr_tail"] and result["stderr"].strip():   # type: ignore[union-attr]
            emit("标准错误最后若干行：")
            for line in result["stderr_tail"]:           # type: ignore[union-attr]
                emit(f"    ! {line}")
        if result["traceback_hits"]:
            emit("⚠ 输出中出现真正的 Traceback（脚本报错了）：")
            for line in result["traceback_hits"][:5]:        # type: ignore[union-attr]
                emit(f"    ! {line}")
        if result["error_hits"]:
            emit("⚠ 输出中出现真正的异常/警告回显行：")
            for line in result["error_hits"][:10]:           # type: ignore[union-attr]
                emit(f"    ! {line}")
        if result["mention_lines"]:
            emit(f"（提示，不算失败）输出中有 {len(result['mention_lines'])} 行正文里提到了"
                 f" Warning / Traceback 等字样，属于讲解内容：")
            for line in result["mention_lines"][:3]:         # type: ignore[union-attr]
                emit(f"    ~ {line.strip()[:110]}")
        emit("")

    total_elapsed = time.perf_counter() - t_all
    ok_count = sum(1 for r in results if r["ok"])
    fail_count = len(results) - ok_count
    slow = [r for r in results if r["elapsed"] > 30]      # type: ignore[operator]

    emit("=" * 88)
    emit("汇总")
    emit("=" * 88)
    emit(f"共 {len(results)} 个脚本，成功 {ok_count} 个，失败 {fail_count} 个")
    emit(f"总耗时：{total_elapsed:.2f} 秒（要求 < 300 秒）")
    emit("")
    emit("逐脚本结果一览（中文列宽无法用等宽对齐，故用 | 分隔）：")
    emit("  序号 | 脚本 | 返回码 | 耗时(秒) | 结果")
    for i, r in enumerate(results, 1):
        flag = "通过" if r["ok"] else "失败"
        emit(f"  {i:02d} | {r['rel']} | {r['returncode']} | {r['elapsed']:.2f} | {flag}")
    if fail_count:
        emit("")
        emit("失败脚本清单（请修复后重跑）：")
        for r in results:
            if not r["ok"]:
                emit(f"  - {r['rel']}（返回码 {r['returncode']}）")
    if slow:
        emit("")
        emit("⚠ 以下脚本单次运行超过 30 秒，建议优化：")
        for r in slow:
            emit(f"  - {r['rel']}：{r['elapsed']:.2f} 秒")   # type: ignore[str-format]

    emit("")
    emit(f"结束时间：{_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    emit("=" * 88)

    if args.report:
        header = [
            "# 验证报告 VERIFY_REPORT.md",
            "",
            "本文件由 `verify_all.py --report` 自动生成，内容为**真实执行**的完整输出，未做人工修饰。",
            "",
            "## 实际执行的命令",
            "",
            "```powershell",
            f"& '{sys.executable}' '{ML_DIR / 'verify_all.py'}' --report",
            "```",
            "",
            "## 验证环境",
            "",
            f"- 解释器：`{sys.executable}`（Python {sys.version.split()[0]}）",
            f"- 工作目录：`{ML_DIR}`",
            f"- 生成时间：{started_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "- 子进程环境变量：`PYTHONUTF8=1`、`PYTHONIOENCODING=utf-8`、`MPLBACKEND=Agg`",
            "",
            "## 完整输出",
            "",
            "```text",
        ]
        footer = ["```", ""]
        REPORT_PATH.write_text("\n".join(header + out + footer), encoding="utf-8")
        print(f"\n完整报告已写入：{REPORT_PATH}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

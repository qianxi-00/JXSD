"""
一键运行 Deep_Learning 目录下所有示例脚本，并汇总每个脚本的返回码与输出。

对应课案章节：
    全部章节（本脚本是工程化自检工具，不讲解具体知识点）

本节知识点：
    1. 用 `subprocess.run([...], capture_output=True, text=True)` 批量驱动子进程，
       这在工程上等价于「跑一遍全部单元测试」，是深度学习项目交付前的常规动作。
    2. Windows 中文环境的编码陷阱：Python 子进程的 stdout 默认跟随系统 ANSI 代码页
       （中文 Windows 上是 GBK）。若父进程用 encoding="utf-8" 解码就会
       UnicodeDecodeError。解决办法是给子进程注入 PYTHONIOENCODING=utf-8 / PYTHONUTF8=1，
       强制子进程用 UTF-8 输出。
    3. CPU 版 PyTorch 的线程控制：设置 OMP_NUM_THREADS 等环境变量，
       避免多个脚本串行运行时 OpenMP 线程过度并行反而拖慢速度。
    4. 超时保护：timeout=300 秒，防止某个脚本卡死导致整个自检流程挂住。

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Deep_Learning\\verify_all.py'
    （在 PowerShell 中建议先执行 $env:PYTHONIOENCODING='utf-8' 以便终端正确显示中文）
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 目录定位：一律用 __file__ 推算，绝不依赖当前工作目录
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent          # Deep_Learning 根目录
OUTPUT_DIR = ROOT / "output"                    # 图片 / 权重文件的统一输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 每个脚本最多跑 300 秒；正常脚本都在 40 秒以内，这里只是兜底
TIMEOUT_SECONDS = 300
# 每个脚本打印输出的最后多少行（题目要求 15 行）
TAIL_LINES = 15


def collect_scripts() -> list[Path]:
    """收集本目录下所有需要运行的 .py 脚本。

    规则：
        - 递归搜索所有子目录（如 01_PyTorch基础/、02_网络架构/ ...）
        - 跳过 verify_all.py 自己（否则会无限递归调用自己）
        - 跳过 __pycache__ 等隐藏目录
        - 按路径排序，保证每次运行顺序一致、输出可复现
    """
    scripts: list[Path] = []
    for path in sorted(ROOT.rglob("*.py")):
        # rglob 会命中 __pycache__ 里的东西吗？不会（里面只有 .pyc），但保险起见还是过滤
        if "__pycache__" in path.parts:
            continue
        if path.name == "verify_all.py":
            continue
        scripts.append(path)
    return scripts


def build_child_env() -> dict[str, str]:
    """构造子进程环境变量：既解决编码问题，又控制 CPU 线程数。"""
    env = os.environ.copy()

    # --- 编码：强制子进程用 UTF-8 写 stdout/stderr，父进程才能用 utf-8 解码 ---
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    # --- 线程：CPU 版 PyTorch 默认会开满所有核心做 OpenMP 并行，
    #     脚本是串行跑的，线程开太多反而增加调度开销。这里限成 4 线程。---
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")

    # --- matplotlib 在无显示环境下必须用 Agg 后端（脚本内部已经设置，这里再兜一层）---
    env.setdefault("MPLBACKEND", "Agg")

    return env


def run_one(script: Path, env: dict[str, str]) -> dict:
    """运行单个脚本，返回 {路径, 返回码, 耗时, 输出尾部, 错误尾部}。"""
    rel = script.relative_to(ROOT)
    started = time.perf_counter()

    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",      # 极端情况下也不让解码错误把自检流程打断
            timeout=TIMEOUT_SECONDS,
            env=env,
            cwd=str(ROOT),          # 故意用 Deep_Learning 作为工作目录，验证脚本不依赖 cwd
        )
        elapsed = time.perf_counter() - started
        return {
            "script": script,
            "rel": rel,
            "returncode": result.returncode,
            "elapsed": elapsed,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
            "timeout": False,
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - started
        # TimeoutExpired 的 stdout/stderr 可能是 bytes（取决于是否设置了 encoding）
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", errors="replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", errors="replace")
        return {
            "script": script,
            "rel": rel,
            "returncode": -1,
            "elapsed": elapsed,
            "stdout": out,
            "stderr": err + f"\n[超时] 超过 {TIMEOUT_SECONDS} 秒仍未结束，已强制终止。",
            "timeout": True,
        }


def tail(text: str, n: int = TAIL_LINES) -> str:
    """取文本最后 n 行（题目要求打印最后 15 行输出）。"""
    lines = text.rstrip("\n").splitlines()
    if not lines:
        return "    <无输出>"
    return "\n".join("    " + line for line in lines[-n:])


def main() -> int:
    scripts = collect_scripts()
    env = build_child_env()

    print("=" * 78)
    print("Deep_Learning 目录脚本一键自检（verify_all.py）")
    print("=" * 78)
    print(f"Deep_Learning 根目录 : {ROOT}")
    print(f"使用解释器           : {sys.executable}")
    print(f"Python 版本          : {sys.version.split()[0]}")
    print(f"待运行脚本数量       : {len(scripts)}")
    print(f"单脚本超时上限       : {TIMEOUT_SECONDS} 秒")
    print(f"输出目录             : {OUTPUT_DIR}")
    print("=" * 78)
    print()

    records: list[dict] = []
    for index, script in enumerate(scripts, start=1):
        print("-" * 78)
        print(f"[{index}/{len(scripts)}] 运行：{script.relative_to(ROOT)}")
        print("-" * 78)
        record = run_one(script, env)
        records.append(record)

        status = "成功" if record["returncode"] == 0 else "失败"
        print(f"返回码：{record['returncode']}（{status}）  耗时：{record['elapsed']:.2f} 秒")
        print(f"--- 最后 {TAIL_LINES} 行输出 ---")
        print(tail(record["stdout"]))

        # 有 stderr 就单独展示；正常脚本不应有任何 stderr 内容
        if record["stderr"].strip():
            print(f"--- 最后 {TAIL_LINES} 行标准错误 ---")
            print(tail(record["stderr"]))
        print()

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------
    total = len(records)
    ok_list = [r for r in records if r["returncode"] == 0]
    fail_list = [r for r in records if r["returncode"] != 0]
    total_elapsed = sum(r["elapsed"] for r in records)

    print("=" * 78)
    print("耗时排行（本次实测）")
    print("=" * 78)
    for record in sorted(records, key=lambda r: r["elapsed"], reverse=True):
        flag = "OK  " if record["returncode"] == 0 else "FAIL"
        print(f"  [{flag}] {record['elapsed']:6.2f}s  {record['rel']}")
    print(f"  合计耗时：{total_elapsed:.2f} 秒")
    print()

    if fail_list:
        print("=" * 78)
        print("失败脚本清单")
        print("=" * 78)
        for record in fail_list:
            print(f"  - {record['rel']}（返回码 {record['returncode']}）")
        print()

    print("=" * 78)
    print(f"共 {total} 个脚本，成功 {len(ok_list)} 个，失败 {len(fail_list)} 个")
    print("=" * 78)

    # 退出码：全部成功返回 0，否则返回失败个数（便于外部 CI 判断）
    return 0 if not fail_list else len(fail_list)


if __name__ == "__main__":
    sys.exit(main())

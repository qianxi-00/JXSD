"""verify_all.py —— 一键运行 Back_End 下所有可运行脚本并汇总结果
================================================================
对应课案章节：后端开发基础（全章节自检工具）

**它做什么？**
    1. 扫描 Back_End 目录下所有 .py 文件（跳过 __pycache__ / data 等运行时目录）；
    2. 按文件类型选择正确的运行方式：
         · FastAPI 示例（直接运行会 uvicorn.run 阻塞）→ 加 `--check` 走快速自检；
         · test_*.py  → 用 pytest 运行（直接运行它们没有任何输出）；
         · conftest.py / __init__.py → 跳过（它们是配置/包标识，不是可执行脚本）；
         · 其他脚本 → 直接用虚拟环境解释器运行；
    3. 逐个捕获输出，打印「返回码 + 最后 15 行输出」；
    4. 最后跑一遍 pytest 做交叉验证，并打印汇总。

**为什么用 subprocess 而不是 import？**
    每个示例脚本都是"独立程序"（有 `if __name__ == "__main__"`），运行它们能验证：
        · 解释器能正常导入所有依赖；
        · 脚本的路径计算不依赖当前工作目录（这里故意把 cwd 设成 Back_End）；
        · 输出没有 Traceback、退出码为 0。

本节知识点：
    1. subprocess.run 的 capture_output / text / encoding / timeout 参数
    2. 用 sys.executable 保证"用同一个解释器"运行子脚本（而不是碰运气找 python）
    3. 环境变量注入：PYTHONIOENCODING=utf-8，保证子进程输出是 UTF-8
    4. 表驱动地扫描与执行，最后统一汇总退出码

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\verify_all.py'
    （输出会被写入 Back_End/VERIFY_REPORT.md 的素材，见文件末尾提示）
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time

# ------------------------------------------------------------
# 路径：本文件位于 Back_End/verify_all.py
# ------------------------------------------------------------
BACK_END = pathlib.Path(__file__).resolve().parent
ROOT = BACK_END.parent
DATA_DIR = BACK_END / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

VENV_PYTHON = sys.executable            # 当前解释器就是虚拟环境里的 python.exe
PER_SCRIPT_TIMEOUT = 180                # 单个脚本最长运行 180 秒（防止意外阻塞）

# 不扫描的目录（运行时产物 / 缓存 / 虚拟环境）
SKIP_DIRS = {"__pycache__", "data", ".pytest_cache", ".ruff_cache", ".venv"}

# 需要跳过、不单独运行的文件名
SKIP_NAMES = {
    "__init__.py",      # 包标识，运行它没有任何意义
    "conftest.py",      # pytest 配置与 fixture 定义，单独运行不收集任何测试
}

# 这些目录下的脚本"直接运行会启动服务器并阻塞"，统一改用 --check 快速自检
SERVICE_DIRS = {"06_FastAPI", "10_综合实战"}


def discover_scripts() -> list[pathlib.Path]:
    """扫描 Back_End 下所有 .py 文件（按路径排序，保证每次运行顺序一致）。"""
    scripts: list[pathlib.Path] = []
    for path in sorted(BACK_END.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.name in SKIP_NAMES:
            continue
        if path.resolve() == pathlib.Path(__file__).resolve():
            continue                     # 不要递归运行自己
        scripts.append(path)
    return scripts


def decide_mode(script: pathlib.Path) -> tuple[str, list[str]]:
    """决定用什么方式运行脚本，返回 (模式说明, 额外的命令行参数)。"""
    if script.name.startswith("test_") or script.name.endswith("_test.py"):
        # 测试文件：用 pytest 运行才有意义（直接执行只会定义函数，没有任何输出）
        return "pytest", ["-m", "pytest", str(script), "-q", "--no-header"]
    if any(part in SERVICE_DIRS for part in script.parts) and script.name != "verify_api.py":
        # FastAPI 示例 / book_api 入口：加 --check 走不阻塞的自检模式
        return "FastAPI --check", ["--check"]
    return "直接运行", []


def run_script(script: pathlib.Path, mode: str, extra_args: list[str]) -> dict:
    """运行一个脚本并收集结果。

    关键参数：
        capture_output=True  捕获 stdout/stderr（不刷屏，便于统一汇总）
        text=True            以文本而非字节返回
        encoding="utf-8"     明确编码，避免 Windows 默认 GBK 造成乱码
        timeout=...          超时保护：任何会阻塞的脚本都会被杀掉并记为失败
        cwd=BACK_END         故意用一个"中立"的工作目录，验证脚本不依赖 CWD
        env                  注入 PYTHONIOENCODING，让子进程输出 UTF-8
    """
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    if mode.startswith("pytest"):
        # pytest 模式：命令已经是完整的 ["-m", "pytest", ...]
        cmd = [VENV_PYTHON, *extra_args]
    else:
        cmd = [VENV_PYTHON, str(script), *extra_args]

    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=PER_SCRIPT_TIMEOUT,
            cwd=str(BACK_END),
            env=env,
        )
        cost = time.perf_counter() - started
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "cost": cost,
            "timeout": False,
            "cmd": cmd,
        }
    except subprocess.TimeoutExpired as exc:
        cost = time.perf_counter() - started
        return {
            "returncode": -1,
            "stdout": (exc.stdout or "") if isinstance(exc.stdout, str) else "",
            "stderr": f"[超时] 脚本运行超过 {PER_SCRIPT_TIMEOUT} 秒被强制终止",
            "cost": cost,
            "timeout": True,
            "cmd": cmd,
        }
    except Exception as exc:                     # 极端情况（解释器都起不来）
        cost = time.perf_counter() - started
        return {
            "returncode": -2,
            "stdout": "",
            "stderr": f"[启动失败] {type(exc).__name__}: {exc}",
            "cost": cost,
            "timeout": False,
            "cmd": cmd,
        }


def tail_lines(text: str, count: int = 15) -> list[str]:
    """取输出的最后 count 行（去掉尾部空行）。"""
    lines = text.rstrip("\n").splitlines()
    return lines[-count:] if len(lines) > count else lines


def main() -> int:
    print("#" * 78)
    print("# Back_End 全量脚本自检（verify_all.py）")
    print("#" * 78)
    print(f"解释器      ：{VENV_PYTHON}")
    print(f"Python 版本 ：{sys.version.split()[0]}")
    print(f"扫描目录    ：{BACK_END}")
    print(f"工作目录    ：{BACK_END}（所有子脚本都在这个「中立」目录下运行，用于验证它们不依赖当前工作目录）")
    print(f"单脚本超时  ：{PER_SCRIPT_TIMEOUT} 秒")
    print()

    scripts = discover_scripts()
    print(f"发现 {len(scripts)} 个可运行脚本：")
    for path in scripts:
        mode, _ = decide_mode(path)
        print(f"  · {path.relative_to(BACK_END).as_posix():<52} [{mode}]")
    print()

    results: list[dict] = []
    for index, script in enumerate(scripts, start=1):
        rel = script.relative_to(BACK_END).as_posix()
        mode, extra = decide_mode(script)
        print("=" * 78)
        print(f"[{index}/{len(scripts)}] {rel}    （模式：{mode}）")
        print("=" * 78)

        result = run_script(script, mode, extra)
        result["script"] = rel
        result["mode"] = mode
        results.append(result)

        status = "成功 ✓" if result["returncode"] == 0 else f"失败 ✗（返回码 {result['returncode']}）"
        print(f"  运行方式：{' '.join(result['cmd'])}")
        print(f"  返回码  ：{result['returncode']}   {status}   耗时 {result['cost']:.2f} 秒")

        out_lines = tail_lines(result["stdout"], 15)
        print(f"  ---- 标准输出最后 {len(out_lines)} 行 ----")
        for line in out_lines:
            print(f"  | {line}")

        err_lines = tail_lines(result["stderr"], 15)
        if err_lines:
            print(f"  ---- 标准错误最后 {len(err_lines)} 行 ----")
            for line in err_lines:
                print(f"  ! {line}")
        print()

    # ============================================================
    # 汇总
    # ============================================================
    total = len(results)
    failed = [r for r in results if r["returncode"] != 0]
    success = total - len(failed)

    print("#" * 78)
    print("# 脚本自检汇总")
    print("#" * 78)
    print(f"{'脚本':<56}{'模式':<16}{'返回码':>6}{'耗时':>9}")
    print("-" * 78)
    for r in results:
        mark = "" if r["returncode"] == 0 else "  ✗"
        print(f"{r['script']:<56}{r['mode']:<16}{r['returncode']:>6}{r['cost']:>8.2f}s{mark}")
    print("-" * 78)

    # ---------- 附加：pytest 交叉验证 ----------
    print()
    print("#" * 78)
    print("# 附加验证：pytest（07_测试框架 + 10_综合实战/book_api）")
    print("#" * 78)
    pytest_targets = [
        str(BACK_END / "07_测试框架"),
        str(BACK_END / "10_综合实战" / "book_api"),
    ]
    pytest_result = run_script(BACK_END / "07_测试框架" / "test_calculator.py", "pytest",
                               ["-m", "pytest", *pytest_targets, "-q", "--no-header"])
    py_lines = tail_lines(pytest_result["stdout"] + pytest_result["stderr"], 12)
    for line in py_lines:
        print(f"  | {line}")
    pytest_ok = pytest_result["returncode"] == 0
    print(f"  pytest 返回码：{pytest_result['returncode']}   "
          f"{'全部通过 ✓' if pytest_ok else '存在失败 ✗'}   耗时 {pytest_result['cost']:.2f} 秒")

    # ---------- 附加：FastAPI 接口全量校验 ----------
    print()
    print("#" * 78)
    print("# 附加验证：06_FastAPI/verify_api.py（TestClient 逐个示例校验接口）")
    print("#" * 78)
    api_result = run_script(BACK_END / "06_FastAPI" / "verify_api.py", "直接运行", [])
    api_lines = tail_lines(api_result["stdout"], 8)
    for line in api_lines:
        print(f"  | {line}")
    api_ok = api_result["returncode"] == 0
    print(f"  verify_api 返回码：{api_result['returncode']}   "
          f"{'接口全部通过 ✓' if api_ok else '存在失败 ✗'}   耗时 {api_result['cost']:.2f} 秒")

    # ---------- 最终结论 ----------
    print()
    print("#" * 78)
    print("# 最终结论")
    print("#" * 78)
    print(f"共 {total} 个脚本，成功 {success} 个，失败 {len(failed)} 个。")
    if failed:
        print("\n失败清单：")
        for r in failed:
            print(f"  ✗ {r['script']}（返回码 {r['returncode']}）")
            for line in tail_lines(r["stderr"], 5):
                print(f"      {line}")
    print(f"pytest 附加验证：{'通过 ✓' if pytest_ok else '失败 ✗'}")
    print(f"verify_api 接口校验：{'通过 ✓' if api_ok else '失败 ✗'}")

    # 从 verify_api 的输出里提取"实际发出的请求数"，让结论更具体
    api_requests = ""
    for line in api_result["stdout"].splitlines():
        if "实际发出的请求" in line:
            api_requests = line.split("：")[-1].strip()
            break

    all_ok = not failed and pytest_ok and api_ok
    print()
    if all_ok:
        print("全部自检通过 ✓（脚本 0 失败，pytest 全绿"
              + (f"，{api_requests} 个接口请求全部符合预期）" if api_requests else "）"))
        print("提示：把本文件的完整输出保存到 Back_End/VERIFY_REPORT.md 即可作为交付证据。")
    else:
        print("自检未全部通过 ✗ 请查看上面的失败明细。")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
r"""
Agent 课程 Notebook 批量无头执行器
================================================================
对每个 notebook 起一个独立内核，用 nbclient 从头跑到尾，按三档判定：

    PASS       全跑通，没有异常
    PASS-降级  跑完了，但某处打印了「跳过 / 降级 / 未就绪」——
               说明该 notebook 的**前置条件没满足**（缺密钥、缺服务），
               这是设计好的降级路径，不算失败
    TRACEBACK  某个 cell 抛异常（落盘完整输出）
    TIMEOUT    cell 超过超时上限还没跑完

车道安排（照搬 Agent_jxsd/run_all.py 里已经验证过的经验）：
    · **串行车道**：会起常驻服务的章（05_mcp / 07_protocols / 01_langgraph 的接口版），
      它们要占 8000 / 9000 / 2024 端口，并发必然互撞；
    · **4 条并行车道**：其余 notebook 平均分。

用法（仓库根目录下）：
    & .\.venv\Scripts\python.exe Agent\_tools\run_notebooks.py                 # 全量
    & .\.venv\Scripts\python.exe Agent\_tools\run_notebooks.py 01_langgraph    # 只跑某章
    & .\.venv\Scripts\python.exe Agent\_tools\run_notebooks.py --list          # 只列计划
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

sys.stdout.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[2]
AGENT = REPO_ROOT / "Agent"
OUT_DIR = Path(r"F:\ProGramApp\DSH_Temporary\agent_nb")
LOG = OUT_DIR / "run_nb.log"
RESULT_JSON = OUT_DIR / "run_nb_result.json"
FAIL_DIR = OUT_DIR / "run_nb_failures"

KERNEL_NAME = "python-base-agent"

# 起常驻服务 / 占端口的章 → 串行
SERIAL_CHAPTERS = {"05_mcp", "07_protocols"}
# 单个 notebook 里出现这些片段，也归到串行车道
SERIAL_MARKERS = ("uvicorn.run", "mcp.run(", "langgraph dev", "--port 2024")

# 降级标记：这些字样出现在输出里就说明走了「前置条件没满足」的分支
DEGRADE_MARKERS = ("[跳过]", "[降级]", "未就绪", "跳过：", "请在 .env")

# 慢章给足时间（评估类要跑很多轮模型调用）
SLOW_MARKERS = ("06_langfuse",)
TIMEOUT_DEFAULT = 900
TIMEOUT_SLOW = 2700

lock = threading.Lock()
results: list[dict] = []


# ================================================================
# 扫描与分类
# ================================================================
def all_notebooks(chapter: str | None = None) -> list[Path]:
    out = []
    for p in sorted(AGENT.rglob("*.ipynb")):
        if "_py_source" in p.parts or ".ipynb_checkpoints" in p.parts:
            continue
        rel = p.relative_to(AGENT)
        if chapter and rel.parts[0] != chapter:
            continue
        out.append(p)
    return out


def resolve_targets(target: str | None) -> list[Path]:
    """解析位置参数：可以是**章节目录名**（跑整章），也可以是**单个 notebook**。

    为什么要支持单个：多个 subagent 会在同一章里并发改造不同的 notebook，
    如果每人都跑一次 `run_notebooks.py <章>`，就会把同章其它 notebook 也带着跑一遍
    —— 既浪费模型额度，又会因为抢端口 / 抢 tmp 工作目录而互相干扰。
    所以每个 subagent 只跑自己那一个。

    路径的两种写法都接受（踩过的坑）：
        `Agent\\01_langgraph\\01_xx.ipynb`  —— 相对仓库根
        `01_langgraph\\01_xx.ipynb`         —— 相对 Agent/
    一开始只实现了后者，于是任务书里写的 `Agent\\...` 形式被拼成 `Agent/Agent/...`，
    报「没有找到 notebook」—— 有 5 个 subagent 先后踩到并反馈。
    """
    if not target:
        return all_notebooks()
    if target.endswith(".ipynb"):
        p = Path(target)
        candidates: list[Path] = []
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(REPO_ROOT / p)    # 相对仓库根
            candidates.append(AGENT / p)        # 相对 Agent/
        for c in candidates:
            if c.exists():
                return [c]
        return []
    return all_notebooks(target)



def classify(path: Path) -> str:
    rel = path.relative_to(AGENT).as_posix()
    chapter = path.relative_to(AGENT).parts[0]
    if chapter in SERIAL_CHAPTERS:
        return "serial"
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:      # noqa: BLE001
        return "parallel"
    if any(m in text for m in SERIAL_MARKERS):
        return "serial"
    return "parallel"


def timeout_of(path: Path) -> int:
    rel = path.relative_to(AGENT).as_posix()
    return TIMEOUT_SLOW if any(m in rel for m in SLOW_MARKERS) else TIMEOUT_DEFAULT


# ================================================================
# 单个 notebook 的执行
# ================================================================
def exec_one(path: Path) -> dict:
    rel = path.relative_to(AGENT).as_posix()
    tmo = timeout_of(path)
    t0 = time.time()
    status, detail, text = "?", "", ""

    try:
        nb = nbformat.read(str(path), as_version=4)
        client = NotebookClient(
            nb,
            timeout=tmo,
            kernel_name=KERNEL_NAME,          # 显式指定，别用 metadata 里的（错了也不报）
            allow_errors=False,
            startup_timeout=120,
            resources={"metadata": {"path": str(path.parent)}},   # 内核 cwd = notebook 所在目录
        )
        client.execute()
        text = "\n".join(
            str(o.get("text", ""))
            for c in nb.cells if c.cell_type == "code"
            for o in c.get("outputs", [])
        )
        status = "PASS"
        if any(m in text for m in DEGRADE_MARKERS):
            status = "PASS-降级"
            detail = "（走了前置条件降级分支）"
    except CellTimeoutError as exc:
        status, detail = "TIMEOUT", str(exc)[:200]
    except CellExecutionError as exc:
        status, detail = "TRACEBACK", str(exc)[-1500:]
    except Exception as exc:      # noqa: BLE001 —— 起内核失败等
        status, detail = "RUNNER-ERR", f"{type(exc).__name__}: {exc}"
    dur = time.time() - t0

    if status not in ("PASS", "PASS-降级"):
        FAIL_DIR.mkdir(parents=True, exist_ok=True)
        safe = rel.replace("/", "__").replace("\\", "__")
        (FAIL_DIR / f"{status}__{safe}.txt").write_text(
            f"== {rel} ==\n{detail}\n\n---- 输出 tail ----\n{text[-8000:]}",
            encoding="utf-8",
        )

    with lock:
        results.append({"notebook": rel, "status": status, "seconds": round(dur, 1),
                        "timeout": tmo, "detail": detail[:300]})
        print(f"[{status:10s}] {dur:7.1f}s  {rel}{detail and '  ' + detail[:80]}", flush=True)
    return results[-1]


def lane(items: list[Path], name: str) -> None:
    for p in items:
        try:
            exec_one(p)
        except BaseException as exc:      # noqa: BLE001
            with lock:
                results.append({"notebook": str(p), "status": "LANE-ERR", "seconds": 0,
                                "timeout": 0, "detail": f"{type(exc).__name__}: {exc}"})
                print(f"[LANE-ERR  ] {name} {p}: {exc}", flush=True)
        if name == "serial":
            time.sleep(2.0)      # 给端口释放留间隙


# ================================================================
# 主流程
# ================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("chapter", nargs="?", help="只跑某一章，如 01_langgraph")
    ap.add_argument("--lanes", type=int, default=4)
    ap.add_argument("--list", action="store_true", help="只列计划不执行")
    args = ap.parse_args()

    # 环境：中文 Windows 必须 UTF-8；本机 Clash 会拦回环请求，必须绕开
    os.environ["PYTHONUTF8"] = "1"
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"

    files = resolve_targets(args.chapter)
    if not files:
        print(f"没有找到 notebook：{args.chapter!r}")
        return 1
    if len(files) == 1:
        print(f"单个 notebook：{files[0].relative_to(AGENT)}")

    serial = [f for f in files if classify(f) == "serial"]
    parallel = [f for f in files if classify(f) == "parallel"]

    print(f"计划：串行 {len(serial)} | 并行 {len(parallel)} | 共 {len(files)}")
    if args.list:
        for f in files:
            print(f"  [{classify(f):8s}] {f.relative_to(AGENT)}")
        return 0

    # 单笔记本模式：日志 / 结果 / 失败落盘都带 notebook 名，
    # 否则同一章的几个 subagent 并发跑时会互相覆盖对方的现场。
    global LOG, RESULT_JSON, FAIL_DIR
    if len(files) == 1:
        safe = files[0].relative_to(AGENT).as_posix().replace("/", "__")
        LOG = OUT_DIR / f"run_nb.{safe}.log"
        RESULT_JSON = OUT_DIR / f"run_nb.{safe}.json"
        FAIL_DIR = OUT_DIR / "run_nb_failures" / safe

    if FAIL_DIR.exists():
        for old in FAIL_DIR.glob("*.txt"):
            old.unlink()

    t0 = time.time()
    threads = [threading.Thread(target=lane, args=(serial, "serial"), daemon=True)]
    for i in range(args.lanes):
        threads.append(threading.Thread(target=lane, args=(parallel[i::args.lanes], f"p{i}"),
                                        daemon=True))
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    ok = sum(1 for r in results if r["status"] in ("PASS", "PASS-降级"))
    bad = [r for r in results if r["status"] not in ("PASS", "PASS-降级")]
    lines = ["=" * 88,
             f"本轮 {len(results)} 个 notebook | PASS {ok} | 异常 {len(bad)} | "
             f"总耗时 {(time.time() - t0) / 60:.1f} 分钟",
             "=" * 88]
    for r in sorted(results, key=lambda x: x["notebook"]):
        lines.append(f"[{r['status']:10s}] {r['seconds']:7.1f}s  {r['notebook']}")
    if bad:
        lines += ["", "---- 异常清单（完整输出见 run_nb_failures/）----"]
        for r in bad:
            lines.append(f"  [{r['status']}] {r['notebook']}")
            if r["detail"]:
                lines.append(f"      {r['detail'][:300]}")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(lines), encoding="utf-8")
    RESULT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(lines[:3]))
    print(f"日志：{LOG}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

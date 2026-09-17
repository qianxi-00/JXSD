"""四条线路的最终验收脚本：每条线路打一次真接口，把结果落成一份证据 JSON。

为什么单独写成脚本（而不是临时敲命令）：
    - 目标验收要能**一次跑完四条线路**并留下可比对的记录；
    - ④ 融合线路要等 Text-to-SQL（约 8~12s），整轮接近 1 分钟，用脚本才好控制超时；
    - 结果落盘后可以直接贴进报告，不必从控制台回抄（控制台是 GBK，中文容易花）。

用法：.venv/Scripts/python.exe RAG/script/acceptance_4routes.py
（先起服务：`$env:PGGSSENCMODE="disable"; .venv/Scripts/python.exe RAG/app/main.py`）
产物：`.dsh_tmp/acceptance_4routes.json`（gitignored 的临时目录，只作留证用）

⚠️ **这个脚本的 `[OK]` 只证明"链路跑通了"，不证明"答对了"** —— 这是 2026-09-17 全量实测抓到的教训：
它只检查「没报错 + 该出的字段出了 + 答案非空」，于是 fusion 那条
「黄帅今年高铁票一共报销了多少钱？」被判 `[OK]`，而当时答案是 **1691 元（错）**：
黄帅两张票是 G626（高铁，162.2 元）与 Z766（直达特快，1528.8 元），问"高铁票"的正解是 **162.2 元**，
错因是 PG `tickets` 表没有车次字段、只能按 `ticket_type='train'` 求和（见 README §8 N23）。
**要判对错，必须与数据级 ground truth 对照**（或跑 `script/run_stage_eval.py` 用标注评估集），
别把这里的 `[OK]` 当成质量结论。
"""

import json
import time
import urllib.request
from pathlib import Path

BASE = Path(r"F:\ProGram\Python_Base")
ENDPOINT = "http://127.0.0.1:8099/api/chat"

# 每条线路用最能体现它长处的问题：
CASES = [
    ("basic", "何海燕的火车票票号是多少？", []),  # 单人单票事实型
    ("agentic", "于慧的机票票号是多少？", []),  # 需要工具与子代理核验
    ("graph", "董文的航班是从哪到哪的？", []),  # 关系型（图里确实有）
    (  # 聚合型：同时触发票据 + 图谱 + Text-to-SQL
        "fusion",
        "黄帅今年高铁票一共报销了多少钱？",
        [],
    ),
    (  # 多轮：用 ② 做一次追问（①③ 是单轮线路，实测追问会走保守回复）
        "agentic",
        "那乐艳的呢？",
        [
            {"role": "user", "content": "万宁的火车票票号是多少？"},
            {"role": "assistant", "content": "万宁的火车票票号是 T20230702063302。"},
        ],
    ),
]


def ask(mode: str, question: str, history: list[dict]) -> dict:
    payload = json.dumps(
        {"question": question, "mode": mode, "history": history}, ensure_ascii=False
    ).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT, data=payload, headers={"content-type": "application/json"}
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=300) as resp:  # noqa: S310 固定本机地址
            body = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 验收脚本要如实记录失败
        return {
            "mode": mode,
            "question": question,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(time.perf_counter() - started, 2),
        }
    return {
        "mode": mode,
        "question": question,
        "ok": True,
        "elapsed_s": round(time.perf_counter() - started, 2),
        "cache_hit": body.get("cache_hit"),
        "system_error": body.get("system_error"),
        "sources": len(body.get("sources") or []),
        "has_sql": bool((body.get("extra") or {}).get("sql")),
        "communities": len((body.get("extra") or {}).get("communities") or []),
        "answer_head": (body.get("answer") or "").replace("\n", " ")[:160],
    }


records = []
for mode, question, history in CASES:
    row = ask(mode, question, history)
    records.append(row)
    flag = "OK " if row.get("ok") else "FAIL"
    print(
        f"[{flag}] {mode:<8} {row['elapsed_s']:>6}s src={row.get('sources', '-')} "
        f"sql={row.get('has_sql', '-')} 社区={row.get('communities', '-')} "
        f"err={row.get('system_error') or '-'}"
    )
    print(f"        Q: {question}")
    print(f"        A: {row.get('answer_head') or row.get('error')}")

out = BASE / ".dsh_tmp" / "acceptance_4routes.json"
out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nwritten: {out}")

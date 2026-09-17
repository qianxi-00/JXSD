"""探针：Ragas 评判模型的"输出不完整"到底是预算不足还是思考吃预算？

真机现象（`langfuse_evaluation.py` 复跑，36 条数据集）：**Faithfulness 失败 26 条**刷
`Ragas 指标 ... 打分失败: The output is incomplete due to a max_tokens length limit`，
而把 `llm_factory(max_tokens=4096)` 从默认 1024 提上去**失败比例没变**。

本探针用一段"和评判同量级"的长提示词 + JSON 输出要求，比较三种配置：
    A. max_tokens=1024（ragas 默认）
    B. max_tokens=4096（只加预算）
    C. max_tokens=4096 + thinking 关闭（路由那套开关）
看点：reasoning_len 是否吃掉大头、finish_reason 是否 length、content 是否能解析成 JSON。

结论（本机实测）：C 才治本 —— reasoning 0 字、completion 仅 264 token、JSON 完整；
所以 `_ragas_bundle()` 现在同时传 `max_tokens` 与 `extra_body={"thinking": ...}`。
**换评判模型/换端点后重跑本探针**。

用法：.venv\\Scripts\\python.exe RAG\\script\\probe_ragas_judge_budget.py
"""

import json
import sys
from pathlib import Path

BASE = Path(r"F:\ProGram\Python_Base")
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "RAG"))

from config import settings  # noqa: E402
from openai import OpenAI  # noqa: E402

# 与 Faithfulness 评判同量级：一段证据 + 一个回答 + 要求逐句判定并输出 JSON
EVIDENCE = (
    "票据原文：乘机人 董文（DONGWEN），航班 G55004，日期 Jan01，舱位 G，"
    "自 ERLIBANJICHANG 至 XINZHENGJICHANG，座位 42C，登机口 C18，登机时间 22:20，"
    "来源文件 data/flight/ticket-091.png。另：票号 ETKT8762369777769/1。"
) * 3
PROMPT = (
    "请判断下面这段回答是否完全由给定证据支持，并输出 JSON："
    '{"statements": [{"statement": "…", "verdict": "supported|unsupported", "reason": "…"}], '
    '"score": 0.0}\n\n'
    f"【证据】\n{EVIDENCE}\n\n"
    "【回答】\n董文的航班是从二里半机场到新郑机场，航班号 G55004，登机口 C18，"
    "座位 42C，登机时间 22:20，票号 ETKT8762369777769/1。\n"
)

CASES = [
    ("A. max_tokens=1024（ragas 默认）", 1024, None),
    ("B. max_tokens=4096（当前设置）", 4096, None),
    ("C. max_tokens=4096 + 关思考", 4096, {"thinking": {"type": "disabled"}}),
]

client = OpenAI(
    api_key=settings.llm.api_key, base_url=settings.llm.base_url, timeout=settings.llm.timeout
)

records = []
for label, max_tokens, extra_body in CASES:
    kwargs = {
        "model": settings.llm.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body
    try:
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        usage = getattr(resp, "usage", None)
        content = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None) or ""
        # JSON 能不能解析，是"输出是否完整"的直接判据（ragas 就是这么判失败的）
        try:
            json.loads(content)
            json_ok = True
        except Exception:  # noqa: BLE001
            json_ok = False
        row = {
            "case": label,
            "max_tokens": max_tokens,
            "finish_reason": resp.choices[0].finish_reason,
            "reasoning_len": len(reasoning),
            "content_len": len(content),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "json_parsable": json_ok,
            "extra_body": extra_body,
        }
    except Exception as exc:  # noqa: BLE001 探针如实记录
        row = {"case": label, "error": f"{type(exc).__name__}: {exc}"[:200]}
    records.append(row)
    print(
        f"{label:<34} finish={row.get('finish_reason')} reasoning={row.get('reasoning_len')} "
        f"content={row.get('content_len')} completion={row.get('completion_tokens')} "
        f"json_ok={row.get('json_parsable')} {row.get('error', '')}"
    )

out = BASE / ".dsh_tmp" / "ragas_judge_probe.json"
out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"written: {out}")

r"""探针：验证当前 LLM 端点**认不认**"关思考"参数，以及短任务的预算够不够。

为什么需要这个脚本（一次真实的翻车）：
    路由（`llm/chat.py::route_query`）必须立刻拿到正文，所以显式发了
    `{"enable_thinking": False}`。**离线用例全绿**（它们 monkeypatch 了客户端，
    只能断言"发了哪个字段"），但真机端到端复跑发现这个字段被端点**静默忽略**：
    域外问题（"今天杭州天气怎么样？"）思考 453~1111 字，把 `max_tokens=256` 吃光，
    `finish_reason=length`、正文为空 ⇒ 路由静默退化成"走 RAG + 直接检索"。

    换成这个端点真正认的 `{"thinking": {"type": "disabled"}}` 后：
    reasoning=0 字、正文 40 字、只花 14 token。

**换端点 / 换模型 / 升级 SDK 之后都该重跑一次本脚本**，因为：
    - `enable_thinking` 是 DashScope/Qwen 系参数；
    - `thinking` / `reasoning_effort` 是另一派（Anthropic / OpenAI 新风格）；
    - 到底哪个生效只有实测知道，文档不保证。

用法：
    .venv\Scripts\python.exe RAG\script\probe_llm_thinking.py
    .venv\Scripts\python.exe RAG\script\probe_llm_thinking.py --question "自定义问题" --runs 3

判定标准（看输出表）：
    - "生效"的标志：`reasoning=0` 且 `content>0` —— 思考真被关掉，且正文出得来；
    - "被忽略"的标志：`reasoning>0` 或 `content=0`（`finish=length` 说明预算被思考吃光）；
    - 若所有候选参数都被忽略，就得靠加大 `ROUTER_MAX_TOKENS` 兜住思考，
      并按"域外问题也要留够预算"来取值（本轮实测 1024 仍不够，不建议走这条路）。
"""

import argparse
import json
import sys
from pathlib import Path

# --- 路径引导：与 app\main.py 同一套三段式样板（详见那边的注释）---
_BASE = Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
sys.path.insert(0, str(_BASE))
sys.path.insert(0, str(_BASE / "RAG"))

from config import settings  # noqa: E402
from core.logger import logger  # noqa: E402
from core.prompts import ROUTER_PROMPT  # noqa: E402
from openai import OpenAI  # noqa: E402

# 域外问题：思考最长、最容易把预算吃光 —— 拿它当"压力样本"。
DEFAULT_QUESTION = "今天杭州天气怎么样？"

# 每个候选配置都要单独试：标签只用于输出，方便对照。
CASES: list[tuple[str, int, dict | None]] = [
    ("不发任何思考参数", 256, None),
    ("enable_thinking=False（路由旧写法，实测被忽略）", 256, {"enable_thinking": False}),
    ("enable_thinking=False + 预算 1024", 1024, {"enable_thinking": False}),
    ("thinking={'type':'disabled'}（路由现写法）", 256, {"thinking": {"type": "disabled"}}),
    ("reasoning_effort='none'", 256, {"reasoning_effort": "none"}),
]


def probe_once(client: OpenAI, question: str, max_tokens: int, extra_body: dict | None) -> dict:
    """跑一次调用，把"能不能出正文"的关键字段全记下来。"""
    kwargs = {
        "model": settings.llm.model,
        "messages": [{"role": "user", "content": ROUTER_PROMPT.format(question=question)}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:  # noqa: BLE001 探针要如实记录 400 之类的拒绝
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}

    message = resp.choices[0].message
    choice = resp.choices[0]
    usage = getattr(resp, "usage", None)
    reasoning = getattr(message, "reasoning_content", None) or ""
    return {
        "ok": True,
        "finish_reason": getattr(choice, "finish_reason", None),
        "content_len": len(message.content or ""),
        "content": (message.content or "")[:80],
        "reasoning_len": len(reasoning),
        # completion_tokens 是最直观的成本证据：关掉思考后从 256 掉到 ~14。
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="验证端点认不认'关思考'参数")
    parser.add_argument("--question", default=DEFAULT_QUESTION, help="探针问题（默认取域外问题当压力样本）")
    parser.add_argument("--runs", type=int, default=2, help="每个配置重复几次（模型输出有随机性）")
    parser.add_argument("--json", default="", help="把原始结果写到该路径（UTF-8 JSON）")
    args = parser.parse_args()

    client = OpenAI(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        timeout=settings.llm.timeout,
    )
    logger.info(f"[探针] 端点={settings.llm.base_url} 模型={settings.llm.model} 问题={args.question!r}")

    records = []
    for label, max_tokens, extra_body in CASES:
        for run in range(args.runs):
            row = probe_once(client, args.question, max_tokens, extra_body)
            row.update({"case": label, "run": run, "max_tokens": max_tokens, "extra_body": extra_body})
            records.append(row)
            verdict = "?"
            if row["ok"]:
                verdict = "生效" if (row["reasoning_len"] == 0 and row["content_len"] > 0) else "被忽略/失败"
            print(
                f"{label:<48} run{run} -> {verdict} | finish={row.get('finish_reason')} "
                f"content={row.get('content_len')} reasoning={row.get('reasoning_len')} "
                f"completion_tokens={row.get('completion_tokens')} {row.get('error', '')}"
            )

    if args.json:
        out = Path(args.json)
        out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"[探针] 原始结果已写入 {out}")

    ok_any = any(r.get("ok") and r.get("content_len") for r in records)
    return 0 if ok_any else 1


if __name__ == "__main__":
    raise SystemExit(main())

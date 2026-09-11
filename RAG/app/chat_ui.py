# --- 路径引导：确保能导入 Python_Base 根目录的 config 与 RAG 内部包 ---
import sys as _sys
from pathlib import Path as _Path

_BASE = _Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
_sys.path.insert(0, str(_BASE))          # Python_Base 根（config.py）
_sys.path.insert(0, str(_BASE / "RAG"))  # RAG 根（core/llm/pipeline 等包）
"""Chainlit 前端界面:AI 聊天问答,可视化展示 RAG 全链路执行过程

功能:
- 侧边设置(齿轮)中可切换 流式 / 非流式 输出;
- 每一步链路(缓存 -> Agent 路由 -> Query 改写 -> 四路召回 -> 重排序 -> LLM 思考 -> 回答)
  以可视化 Step 呈现,可清晰看到是否命中缓存、改写策略、召回内容与分数。
"""

import json

import chainlit as cl
from chainlit.input_widget import Switch

from core.prompts import REWRITE_LABELS
from pipeline.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()


def _fmt_date(value) -> str:
    if not value:
        return "无"
    return f"{value // 10000:04d}-{value // 100 % 100:02d}-{value % 100:02d}"


def _fmt_amount(value) -> str:
    if value is None:
        return "无"
    return f"{value / 100:.2f}元"


def _rows_table(rows: list[dict], score_key: str) -> str:
    if not rows:
        return "_无_"
    header = "| # | 来源 | 类型 | 票号 | 相关人/买方 | 日期 | 金额 | 路线 | 交易对方 | 分数 |"
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for i, r in enumerate(rows, 1):
        score = r.get(score_key)
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "无"
        lines.append(
            f"| {i} | {r.get('source_file') or '无'} | {r.get('ticket_type') or '无'}"
            f" | {r.get('ticket_no') or '无'} | {r.get('person') or '无'}"
            f" | {_fmt_date(r.get('date_int'))} | {_fmt_amount(r.get('amount_fen'))}"
            f" | {r.get('route') or '无'} | {r.get('counterparty') or '无'} | {score_text} |"
        )
    return "\n".join(lines)


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("stream_mode", True)
    await cl.ChatSettings(
        [
            Switch(id="stream_mode", label="流式输出", initial=True),
        ]
    ).send()
    await cl.Message(
        content=(
            "您好!我是财务票据智能问答助手 🎫\n\n"
            "支持三类票据问答:✈️ 登机牌 / 🧾 发票 / 🚄 火车票\n\n"
            "在右上角 ⚙️ 设置里可以切换 **流式输出** 开关;\n"
            "每次问答会展示完整链路:\n"
            "缓存 → Agent 路由(含 query 改写策略)→ Query 改写 → 四路召回 → 重排序 → LLM 思考 → 回答。"
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict):
    stream_mode = bool(settings.get("stream_mode", True))
    cl.user_session.set("stream_mode", stream_mode)
    mode = "流式" if stream_mode else "非流式"
    await cl.Message(content=f"⚙️ 已切换为 **{mode}输出** 模式,后续回答将按此模式生成。").send()


@cl.on_message
async def on_message(message: cl.Message):
    question = message.content.strip()
    stream_mode = bool(cl.user_session.get("stream_mode", True))

    answer_msg: cl.Message | None = None
    final: dict = {}

    async for ev in pipeline.run_events(question, stream=stream_mode):
        t = ev["type"]

        if t == "cache_hit":
            detail = {
                "命中方式": "精确缓存(10分钟内重复提问)" if ev["mode"] == "exact" else "预设问答相似度命中",
                "匹配问题": ev.get("matched_question") or "—",
                "相似度": ev.get("similarity") if ev.get("similarity") is not None else "—",
                "来源": [s.get("source_file") for s in ev.get("sources", [])] or "缓存答案无来源",
            }
            async with cl.Step(name="① 缓存层:命中 ✅", type="tool") as step:
                step.output = json.dumps(detail, ensure_ascii=False, indent=2)

        elif t == "route":
            if ev["route"] == "direct":
                text = "LLM 路由判断:**DIRECT**,简单问题,无需检索,直接回答"
            else:
                label = REWRITE_LABELS.get(ev.get("rewrite") or "direct", "直接检索")
                text = (
                    f"LLM 路由判断:**RAG**,涉及知识库内容,开始检索票据\n\n"
                    f"query 改写策略:**{label}**(召回阶段将保留直接检索)"
                )
            async with cl.Step(name="② Agent 路由", type="llm") as step:
                step.output = text

        elif t == "rewrite":
            items = "\n".join(f"{i}. {q}" for i, q in enumerate(ev["rewritten"], 1))
            async with cl.Step(name="③ Query 改写", type="tool") as step:
                step.output = (
                    f"改写策略:**{ev['label']}**\n\n生成检索 query {len(ev['rewritten'])} 条:\n{items}\n\n"
                    f"_改写检索将与直接检索的结果做并集召回_"
                )

        elif t == "retrieve":
            sections = [
                f"**直接检索 · 向量召回 {len(ev['direct_vector_hits'])} 条**",
                _rows_table(ev["direct_vector_hits"], "vector_score"),
                f"\n**直接检索 · BM25 召回 {len(ev['direct_keyword_hits'])} 条**",
                _rows_table(ev["direct_keyword_hits"], "keyword_score"),
            ]
            if ev.get("rewrite_vector_hits") or ev.get("rewrite_keyword_hits"):
                sections += [
                    f"\n**改写检索({ev['label']}) · 向量召回 {len(ev['rewrite_vector_hits'])} 条**",
                    _rows_table(ev["rewrite_vector_hits"], "vector_score"),
                    f"\n**改写检索 · BM25 召回 {len(ev['rewrite_keyword_hits'])} 条**",
                    _rows_table(ev["rewrite_keyword_hits"], "keyword_score"),
                ]
            sections.append(f"\n**合并去重后:{ev['merged_count']} 条**")
            async with cl.Step(name="④ 双路召回 + 合并去重", type="retrieval") as step:
                step.output = "\n".join(sections)

        elif t == "rerank":
            kept = ev["kept"]
            body = (
                f"候选 {ev['merged_count']} 条 → 重排取 top_k={ev['top_k']} →"
                f" 相关度阈值 {ev['relevance_p']} 过滤后 **保留 {len(kept)} 条**\n\n"
                f"{_rows_table(kept, 'rerank_score')}"
            )
            async with cl.Step(name="⑤ 重排序(Reranker)", type="rerank") as step:
                step.output = body

        elif t == "thinking":
            async with cl.Step(name="LLM 思考过程", type="thinking") as step:
                step.output = ev["text"]

        elif t == "token":
            if answer_msg is None:
                answer_msg = cl.Message(content="")
            await answer_msg.stream_token(ev["text"])

        elif t == "no_evidence":
            async with cl.Step(name="⚠️ 无可靠依据", type="tool") as step:
                step.output = "召回内容全部低于相关度阈值,触发保守回复,不调用 LLM 生成。"

        elif t == "error":
            if answer_msg is None:
                answer_msg = cl.Message(content="")
            await answer_msg.stream_token(ev["message"])

        elif t == "done":
            final = ev

    if answer_msg is not None:
        if final.get("sources") and final.get("conservative") is not True:
            cite = "\n\n---\n📎 **参考票据**\n" + "\n".join(
                f"- `{s.get('source_file')}`({s.get('ticket_type')},票号 {s.get('ticket_no') or '无'})"
                for s in final["sources"]
            )
            await answer_msg.stream_token(cite)
        await answer_msg.send()

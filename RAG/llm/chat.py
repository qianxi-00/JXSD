"""LLM 调用:Agent 路由 / query 改写 / 非流式 / 流式输出(OpenAI 兼容接口)

流式接口逐段产出 {"reasoning": ...}(思考内容)或 {"content": ...}(回答内容)。
"""

from collections.abc import AsyncIterator

import re

from openai import AsyncOpenAI, OpenAI

from config import settings
from core.logger import logger
from core.prompts import (
    BACKTRACK_PROMPT,
    DIRECT_QA_PROMPT,
    HYDE_PROMPT,
    RAG_ANSWER_PROMPT,
    ROUTER_PROMPT,
    SUBQUERY_PROMPT,
    SYSTEM_PROMPT,
)

REWRITE_PROMPTS = {
    "hyde": HYDE_PROMPT,
    "subquery": SUBQUERY_PROMPT,
    "backtrack": BACKTRACK_PROMPT,
}


def _extra_body() -> dict:
    if settings.llm.enable_thinking:
        return {"enable_thinking": True}
    return {}


def _extract_reasoning(obj) -> str | None:
    reasoning = getattr(obj, "reasoning_content", None)
    if not reasoning:
        reasoning = (getattr(obj, "model_extra", None) or {}).get("reasoning_content")
    return reasoning


def _messages(question: str, context: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": RAG_ANSWER_PROMPT.format(context=context, question=question)},
    ]


def route_query(question: str) -> tuple[bool, str]:
    """Agent 路由:返回 (是否需要 RAG 检索, query 改写策略 hyde/subquery/backtrack/direct)"""
    client = OpenAI(api_key=settings.llm.api_key, base_url=settings.llm.base_url)
    try:
        resp = client.chat.completions.create(
            model=settings.llm.model,
            messages=[{"role": "user", "content": ROUTER_PROMPT.format(question=question)}],
            temperature=0,
            max_tokens=64,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning(f"[路由] LLM 调用失败,默认走 RAG+直接检索: {exc}")
        return True, "direct"
    logger.debug(f"[路由] LLM 原始输出: {text}")

    need_rag, rewrite = True, "direct"
    m_route = re.search(r'"route"\s*:\s*"([^"]+)"', text, re.I)
    m_rewrite = re.search(r'"rewrite"\s*:\s*"([^"]+)"', text, re.I)
    route_value = (m_route.group(1).upper() if m_route else (text.upper() if text.upper() in ("RAG", "DIRECT") else "RAG"))
    need_rag = route_value != "DIRECT"
    if m_rewrite:
        rewrite_value = m_rewrite.group(1).lower()
        for method in ("hyde", "subquery", "backtrack", "direct"):
            if method in rewrite_value:
                rewrite = method
                break
    return need_rag, rewrite


def rewrite_query(question: str, method: str) -> list[str]:
    """按改写策略生成用于检索的 query 列表;direct 或失败时返回空列表(仅保留直接检索)"""
    prompt = REWRITE_PROMPTS.get(method)
    if not prompt or not question.strip():
        return []
    client = OpenAI(api_key=settings.llm.api_key, base_url=settings.llm.base_url)
    try:
        resp = client.chat.completions.create(
            model=settings.llm.model,
            messages=[{"role": "user", "content": prompt.format(question=question)}],
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            extra_body=_extra_body(),
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning(f"[改写] {method} 失败,回退仅直接检索: {exc}")
        return []
    if not text:
        return []
    if method == "subquery":
        subs = [ln.strip(" \t0123456789.、()()•-–") for ln in text.splitlines() if ln.strip()]
        return [s for s in subs if s][:3]
    return [text]


def generate_answer(question: str, context: str) -> tuple[str, str | None]:
    """非流式输出:基于检索上下文一次性返回完整回答,返回 (回答, 思考内容)"""
    client = OpenAI(api_key=settings.llm.api_key, base_url=settings.llm.base_url)
    resp = client.chat.completions.create(
        model=settings.llm.model,
        messages=_messages(question, context),
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
        extra_body=_extra_body(),
    )
    message = resp.choices[0].message
    return message.content or "", _extract_reasoning(message)


def generate_direct_answer(question: str) -> tuple[str, str | None]:
    """非流式输出:不经过检索,由 LLM 直接回答,返回 (回答, 思考内容)"""
    client = OpenAI(api_key=settings.llm.api_key, base_url=settings.llm.base_url)
    resp = client.chat.completions.create(
        model=settings.llm.model,
        messages=[{"role": "user", "content": DIRECT_QA_PROMPT.format(question=question)}],
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
        extra_body=_extra_body(),
    )
    message = resp.choices[0].message
    return message.content or "", _extract_reasoning(message)


async def stream_answer(question: str, context: str):
    """流式输出:基于检索上下文,逐段产出 {"reasoning": ...} 或 {"content": ...}"""
    client = AsyncOpenAI(api_key=settings.llm.api_key, base_url=settings.llm.base_url)
    stream = await client.chat.completions.create(
        model=settings.llm.model,
        messages=_messages(question, context),
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
        stream=True,
        extra_body=_extra_body(),
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = _extract_reasoning(delta)
        if reasoning:
            yield {"reasoning": reasoning}
        if delta and delta.content:
            yield {"content": delta.content}
    await stream.close()


async def stream_direct_answer(question: str):
    """流式输出:不经过检索,由 LLM 直接回答,逐段产出 {"reasoning": ...} 或 {"content": ...}"""
    client = AsyncOpenAI(api_key=settings.llm.api_key, base_url=settings.llm.base_url)
    stream = await client.chat.completions.create(
        model=settings.llm.model,
        messages=[{"role": "user", "content": DIRECT_QA_PROMPT.format(question=question)}],
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
        stream=True,
        extra_body=_extra_body(),
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = _extract_reasoning(delta)
        if reasoning:
            yield {"reasoning": reasoning}
        if delta and delta.content:
            yield {"content": delta.content}
    await stream.close()

"""四条 RAG 线路的注册表与统一入口（前端选择器、HTTP `mode` 参数都打到这里）。

## 为什么需要这一层

在本模块之前，"走哪条链路"是**写死在调用点**的：
`app/main.py` 只调基础篇的 `RAGPipeline`，`agentic/` 与 `graph_rag/` 各有一套自己的入口
（前者只能从 Python 直接调、后者是另一个端口的服务）。于是：
    - 前端四个入口无从谈起（用户看到的永远是基础链路）；
    - 三条线路各自的缓存键混在一起，换线路会命中旧线路的答案（见 `core/cache.py` 的 route 参数）；
    - 想横向对比线路优劣，得写四种不同的调用代码。

本模块把"线路"变成一个**数据**（`ModeSpec`）而不是四段 if：新增线路只要加一条注册表项，
前端下拉框、`GET /api/modes`、日志里的 mode 名字都会自动跟上。

## 统一契约（四条线路都返回这个形状）

    {
      "mode": "basic",               # 线路键
      "question": "...",
      "answer": "...",               # 给用户看的正文（可能含引用编号）
      "sources": [...],              # 可核对的票据来源（没有则空列表）
      "extra": {...},                # 线路专属明细（图谱证据、SQL、核验提示…）
      "cache_hit": "exact" | "preset" | None,
      "elapsed_s": 1.23,
      "system_error": "" | "…",      # 非空表示"没查成"，与"没查到"区分
      "events": [...]                # 链路步骤（前端渲染 Step 用）
    }

## 惰性 import 是刻意的

`agentic/` 会拉起 deepagents/langgraph（import 就要几百毫秒），
`graph_rag/` 会连 Neo4j —— 只问基础线路的用户不该为它们付启动成本。
所以四个 runner 一律写成函数，import 放在函数体第一行。
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from core.logger import logger
from core.routes import (
    ROUTE_AGENTIC,
    ROUTE_BASIC,
    ROUTE_FUSION,
    ROUTE_GRAPH,
    ROUTE_KEYS,
    ROUTE_LABELS,
    ROUTE_SUMMARIES,
    ROUTE_SUPPORTS_HISTORY,
    cache_scope,
)

@dataclass(frozen=True)
class ModeSpec:
    """一条线路的静态描述。`runner` 的签名统一为 (question, history) -> dict。"""

    key: str
    runner: Callable[[str, list[dict] | None], dict]
    # 是否为流式线路：True 表示 runner 会返回事件流（目前只有基础线路是真流式）
    streaming: bool = False


def _run_coroutine_sync(coro):
    """在没有"正在运行的 loop"的线程上跑完协程，返回结果。

    为什么需要它：基础线路的底层是异步的（`RAGPipeline.run_and_collect`），
    而 `answer()` 是**同步**入口。直接 `asyncio.run()` 在两种情况下会炸：
        - 被一个 `async def` 的 FastAPI 端点调用（事件循环已在跑）→
          `RuntimeError: asyncio.run() cannot be called from a running event loop`
          （真机实测踩到过，见 README 的端到端复跑清单）；
        - 被 Chainlit 的回调调用（同样是 async 上下文）。
    所以这里判断一下：当前线程没有 loop 就直接跑；有 loop 就把协程丢到
    **新线程**里跑（新线程没有 loop，`asyncio.run` 合法）。

    为什么不用 nest_asyncio 之类：那要新增依赖，而且本质是给已有 loop 打补丁；
    新开一个线程的语义干净得多，代价只是多一次线程创建（本函数不是热路径）。
    """
    import asyncio
    import threading

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # 没有正在跑的 loop：当前线程直接跑
        return asyncio.run(coro)

    # 有正在跑的 loop：换一个线程跑，并把异常/结果带回调用线程
    box: dict = {}

    def _worker():
        try:
            box["result"] = asyncio.run(coro)
        except BaseException as exc:  # noqa: BLE001 原样带回给调用方
            box["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box.get("result")


def _normalize_history(history: Any) -> list[dict]:
    """把前端传来的 history 收敛成 `[{"role": "user"|"assistant", "content": str}, ...]`。

    为什么要"收敛"：Chainlit 回传的是它自己的消息对象（或 dict），
    而 `finance_agent` 只认 role/content 两个字段。这里做一次清洗，
    让四条线路都拿到同一种结构；非法条目直接丢掉而不是报错 ——
    历史有问题不该让整次问答失败。
    """
    if not history:
        return []
    cleaned: list[dict] = []
    for item in history:
        if isinstance(item, dict):
            role, content = item.get("role"), item.get("content")
        else:
            role = getattr(item, "role", None)
            content = getattr(item, "content", None)
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content})
    return cleaned


# ---------------------------------------------------------------- 四条线路的 runner
def _run_basic(question: str, history: list[dict]) -> dict:
    """基础篇单轮检索。**刻意不用 history**（它就是"单轮"线路）。

    历史在这里不是"忘了传"，而是这条线路的定义：它的路由/改写/召回全部只针对
    当前这一句。追问里的代词（"那乐艳的呢？"）没有解析者，实测会抽不到过滤条件
    （filter 为空）、重排最高分 0.098 < 0.22 → 走保守回复。
    要支持多轮请选 Agentic 或融合线路（`ROUTE_SUPPORTS_HISTORY`）。
    """
    pipeline = _basic_pipeline()
    # `run_and_collect` 是 async（内部消费异步事件流）；这里用 `_run_coroutine_sync`
    # 而不是裸 `asyncio.run`，因为本函数可能被 async 端点/Chainlit 回调调到
    # ——那种情况下事件循环已经在跑，裸 asyncio.run 会直接抛 RuntimeError（实测踩过）。
    collected = _run_coroutine_sync(pipeline.run_and_collect(question))
    return {
        "answer": collected.get("answer", ""),
        "sources": collected.get("sources") or [],
        "cache_hit": collected.get("cache_hit"),
        "extra": {
            "route": collected.get("route"),
            "rewrite": collected.get("rewrite"),
            "conservative": collected.get("conservative"),
            "error": collected.get("error"),
        },
    }


_BASIC_PIPELINE = None


def _basic_pipeline():
    """进程内复用同一个 RAGPipeline（它持有 Redis 客户端与预热的 BM25 索引）。

    每次调用都新建会把 BM25 语料重新加载一遍（本机实测百毫秒级），
    而且会让"缓存命中"看起来忽高忽低（不同实例的 preset 矩阵各自加载）。
    """
    global _BASIC_PIPELINE
    if _BASIC_PIPELINE is None:
        from pipeline.rag_pipeline import RAGPipeline

        _BASIC_PIPELINE = RAGPipeline()
    return _BASIC_PIPELINE


_ANSWER_CACHE = None


def _answer_cache():
    """进程内复用的 AnswerCache（③④ 线路读写缓存用）。

    为什么不每次 `AnswerCache()`：它的 `_ensure_preset_loaded()` 会在首次 lookup 时
    把预设问法向量从 Redis 拉回来并**做一次 numpy 归一化**；每次请求新建实例 =
    每次问一句都重做一遍（矩阵只有 5 行，但这是纯浪费，且让"缓存命中"的耗时看起来忽高忽低）。
    与 `_basic_pipeline` 同样的取舍：进程级单例，单进程串行没问题，多 worker 各自一份也正确
    （Redis 是共享的，缓存语义不受影响）。
    注意它只被 ③④ 用；① 的缓存由 `RAGPipeline` 自己持有，② 由 `answer_financial_question` 管。
    """
    global _ANSWER_CACHE
    if _ANSWER_CACHE is None:
        from core.cache import AnswerCache

        _ANSWER_CACHE = AnswerCache()
    return _ANSWER_CACHE


def _cached_route(question: str, history: list[dict], route: str, produce):
    """给"不依赖多轮、或本轮没有历史"的线路套一层**线路作用域**缓存。

    课案流程图把"QA 缓存命中 → 直接返回"画在**所有**线路的最前面；本轮之前只有
    ①（`RAGPipeline` 自带）与 ②（`answer_financial_question` 内部）接了缓存，
    ③④ 每次都真跑 —— ④ 的聚合问题要跑三路取证（约 24s），答案却往往与上一句一字不差。

    三条纪律：
    - **history 非空就既不查也不写**：缓存键只含问题文本，"那乐艳的呢？"在不同上下文里
      含义不同。只跳过"读"是**不够的** —— 那样会把"依赖历史的答案"存成"只看问题文本的答案"，
      之后有人单问同一句就直接命中它（等于把上一轮的语境偷偷复用）。与 ② 的
      `use_cache=not history`（读写一起关）同一口径。
    - **作用域按线路分**：③=graph、④=fusion（`core/routes.py::cache_scope`），
      四线路互不串答案。
    - **命中时不返回 `extra` 明细**：`extra`（图谱社区/关系/SQL 结果）是本次链路的产物，
      上次那次的明细与本次问题只有"文本相同"这一层关系；给前端一个 `from_cache=True`
      比把上次的社区摘要冒充成本次证据要诚实（① 命中缓存时同样不给链路明细）。
    """
    cache = _answer_cache()
    use_cache = not history
    if use_cache:
        cached = cache.lookup(question, route=route)
        if cached and cached.get("answer"):
            logger.info(f"[线路] mode={route} 命中缓存({cached.get('cache_hit')})，跳过检索与生成")
            return {
                "answer": cached["answer"],
                "sources": cached.get("sources") or [],
                "cache_hit": cached.get("cache_hit"),
                "extra": {"from_cache": True},
            }

    payload = produce()
    if use_cache and payload.get("answer"):
        # route 必须与上面的 lookup 一致，否则"存了但读不到"（永不命中且不报错）
        cache.store(question, payload["answer"], payload.get("sources") or [], route=route)
    return {**payload, "cache_hit": None}


def _run_agentic(question: str, history: list[dict]) -> dict:
    """优化篇 Agentic RAG：Deep Agents 主控 + 工具 + 子代理核验。支持多轮。"""
    from agentic.finance_agent import answer_financial_question

    # history 传**副本**：answer_financial_question 会就地往里追加本轮问答，
    # 直接传调用方的列表会让前端 session 里的历史被静默改写。
    working_history = list(history)
    # info 出参拿回"这次是缓存命中还是真跑了 Agent"——返回值是 str，装不下这个信息。
    info: dict = {}
    answer = answer_financial_question(
        question,
        working_history,
        route=ROUTE_AGENTIC,
        info=info,
        # 多轮时**必须关缓存**：精确缓存键只有问题文本，而"那乐艳的呢？"在不同
        # 上下文里含义不同，缓存会把上一轮语境的答案答给下一轮同样的问题（串答案）。
        use_cache=not working_history,
    )
    return {
        "answer": answer,
        # Agentic 线路的来源写在回答正文里（证据文件路径），这里不额外造 sources，
        # 避免前端显示一份"看起来像结构化来源、其实是空的"列表。
        "sources": [],
        "cache_hit": info.get("cache_hit"),
        "extra": {"history_len": len(working_history), "from_cache": info.get("from_cache")},
    }


def _run_graph(question: str, history: list[dict]) -> dict:
    """GraphRAG 线路：社区摘要 + 实体多跳。**不吃 history**（见 ROUTE_SUPPORTS_HISTORY）。

    接缓存（线路作用域 `graph`）：图谱问答的答案只取决于问题与图谱状态，重复问一句
    没有理由再跑一遍"社区召回 + 多跳遍历 + LLM 生成"。
    """

    def produce() -> dict:
        from graph_rag.retriever import answer_query, retrieve_hierarchical

        subgraph = retrieve_hierarchical(question, top_k=5, max_hops=2, max_nodes=10) or {}
        return {
            "answer": answer_query(question, subgraph),
            "sources": [],
            "extra": {
                "communities": subgraph.get("communities") or [],
                "nodes": subgraph.get("nodes") or [],
                "relationships": subgraph.get("relationships") or [],
            },
        }

    return _cached_route(question, history, ROUTE_GRAPH, produce)


def _run_fusion(question: str, history: list[dict]) -> dict:
    """融合线路（本项目的第四种编排）：三路取证 + 分层门控 + 数字核验。支持多轮。

    接缓存（线路作用域 `fusion`）：这是四条线路里最贵的一条（三路取证 + 可能一次
    Text-to-SQL，聚合问题实测约 24s），而"同一句话问第二遍"完全可复用上次答案。
    ⚠️ 多轮时 `_cached_route` 会自动跳过缓存（历史变了，答案就未必一样）。
    """

    def produce() -> dict:
        from pipeline.fusion import answer_fusion

        result = answer_fusion(question, history=history)
        payload = result.as_dict()
        return {
            "answer": payload["answer"],
            "sources": payload["tickets"],
            "extra": {
                "communities": payload["communities"],
                "relationships": payload["relationships"],
                "sql": payload["sql"],
                "unmatched_numbers": payload["unmatched_numbers"],
                "queries": payload["queries"],
                "filter_expr": payload["filter_expr"],
                "rewrite": payload["rewrite"],
                "system_error": payload["system_error"],
            },
        }

    return _cached_route(question, history, ROUTE_FUSION, produce)


MODES: dict[str, ModeSpec] = {
    ROUTE_BASIC: ModeSpec(ROUTE_BASIC, _run_basic, streaming=True),
    ROUTE_AGENTIC: ModeSpec(ROUTE_AGENTIC, _run_agentic),
    ROUTE_GRAPH: ModeSpec(ROUTE_GRAPH, _run_graph),
    ROUTE_FUSION: ModeSpec(ROUTE_FUSION, _run_fusion),
}


def list_modes() -> list[dict]:
    """给前端/`GET /api/modes` 用的线路清单（顺序即展示顺序）。"""
    return [
        {
            "key": key,
            "label": ROUTE_LABELS[key],
            "summary": ROUTE_SUMMARIES[key],
            "supports_history": ROUTE_SUPPORTS_HISTORY[key],
            "streaming": MODES[key].streaming,
        }
        for key in ROUTE_KEYS
    ]


def normalize_mode(mode: str | None) -> str:
    """把外部传进来的 mode 收敛成合法线路键。

    三种输入都接受（前端/脚本来源不同，宽容一点比让用户看到"切了没用"好）：
        - 线路键本身（"fusion"）；
        - 中文显示名（"④ 融合线路（三合一）"）—— Chainlit 的 Select 用的是
          label→value 映射，前端在个别版本/配置下会回传 label，这里兜住；
        - 前端可能加的后缀形式（"④ 融合线路（三合一）（支持多轮）"）—— 按"包含显示名"再兜一次。

    未知值一律落到基础线路并记 warning：前端传错（或旧版本前端不传）时，
    用户还能拿到一个能用的答案，而不是 400 或 500。
    """
    if not mode:
        return ROUTE_BASIC
    if mode in MODES:
        return mode
    # 显示名（含前端拼的多轮后缀）→ 线路键
    for key in ROUTE_KEYS:
        label = ROUTE_LABELS[key]
        if mode == label or label in mode:
            logger.info(f"[线路] mode={mode!r} 按显示名解析为 {key!r}")
            return key
    logger.warning(f"[线路] 未知 mode={mode!r}，按基础线路处理；可选值：{list(MODES)}")
    return ROUTE_BASIC


def answer(question: str, mode: str | None = None, history: list[dict] | None = None) -> dict:
    """统一入口：按 `mode` 分发到对应线路，并保证返回形状一致。

    这里**不做缓存**：三条新线路的缓存要么在各自链路里（基础线路），
    要么由 Runner 决定（Agentic 走 AnswerCache 的 route 作用域）。
    在这一层再套一层缓存，会让"缓存命中"这件事有两个真相，评估时对不上账。
    """
    key = normalize_mode(mode)
    cleaned = _normalize_history(history)
    started = time.perf_counter()
    logger.info(f"[线路] mode={key} 问题={question[:40]!r} 历史 {len(cleaned)} 条")

    try:
        payload = MODES[key].runner(question, cleaned) or {}
    except Exception as exc:  # noqa: BLE001 单条线路失败不能把服务打挂
        # 与各线路内部的降级不同，这里兜的是"意料之外"：连 runner 都没跑起来
        # （import 失败、依赖缺失、Neo4j 客户端初始化炸了…）。
        # 返回可读文案 + system_error，让前端能明确区分"没查到"与"没查成"。
        logger.error(f"[线路] mode={key} 执行失败: {exc}")
        return {
            "mode": key,
            "question": question,
            "answer": f"该线路执行失败（{key}）：{exc}",
            "sources": [],
            "extra": {},
            "cache_hit": None,
            "elapsed_s": round(time.perf_counter() - started, 2),
            "system_error": f"{type(exc).__name__}: {exc}",
            "events": [{"type": "error", "mode": key, "message": str(exc)}],
        }

    return {
        "mode": key,
        "question": question,
        "answer": payload.get("answer", ""),
        "sources": payload.get("sources") or [],
        "extra": payload.get("extra") or {},
        "cache_hit": payload.get("cache_hit"),
        "elapsed_s": round(time.perf_counter() - started, 2),
        "system_error": "",
        # 步骤事件：目前只有基础线路会吐细粒度事件（它的事件流本来就有 7 种），
        # 其余线路给一条"完成"事件，前端据此渲染统一的 Step 列表。
        "events": [{"type": "done", "mode": key}],
    }


async def answer_events(
    question: str,
    mode: str | None = None,
    history: list[dict] | None = None,
    stream: bool = True,
) -> AsyncIterator[dict]:
    """异步事件流入口：基础线路**真流式**，其余线路把结果包成一条 token 事件。

    `stream` 只对基础线路有意义（它的 LLM 调用有流式/非流式两条路径，
    对应页面上的"流式输出"开关）；另外三条线路的耗时都在生成之前的检索/工具环节，
    这个开关对它们没有可切换的东西。

    为什么不让四条都真流式：Agentic（工具循环）与融合（多路取证）在生成之前
    要跑好几秒的检索/工具调用，真流式也只能在最后那一步开始吐字；
    与其为它们各写一套流式协议，不如统一成"事件流"这一种形状，
    前端无论哪条线路都按同一套渲染。
    """
    key = normalize_mode(mode)
    cleaned = _normalize_history(history)

    if key == ROUTE_BASIC:
        # 基础线路复用既有事件流（start/route/rewrite/retrieve/rerank/token/done）
        pipeline = _basic_pipeline()
        async for event in pipeline.run_events(question, stream=stream):
            event["mode"] = key
            yield event
        return

    yield {"type": "start", "mode": key, "question": question}
    result = answer(question, key, cleaned)
    # 事件类型叫 `mode` 而不是 `route`：基础线路的 `route` 事件带的是
    # `route: "rag"|"direct"`（LLM 路由判断的结果），而这里表达的是"用户在设置里选了哪条线路"。
    # 两者复用同一个类型名会直接崩前端 —— 实测踩过：UI 的 route 分支读 `ev["route"]`，
    # 拿到没有该字段的 `mode` 事件就 KeyError，整条消息处理中断（页面上只有提问没有回答）。
    yield {
        "type": "mode",
        "mode": key,
        "label": ROUTE_LABELS[key],
        "summary": ROUTE_SUMMARIES[key],
    }
    if result["system_error"]:
        yield {"type": "error", "mode": key, "message": result["system_error"]}
    yield {"type": "token", "mode": key, "text": result["answer"]}
    yield {
        "type": "done",
        "mode": key,
        "answer": result["answer"],
        "sources": result["sources"],
        "cache_hit": result["cache_hit"],
        "elapsed_s": result["elapsed_s"],
        # `extra` 必须传给前端：②③④ 线路没有细粒度事件，页面上的"证据构成"那一 Step
        # 完全靠它渲染（见 app/chat_ui.py::_render_evidence_step）。
        "extra": result["extra"],
        # 说明：conservative 只在基础线路的 done 里有（这是既有契约），
        # 三条新线路靠 system_error + sources 判空让前端得到等价信息。
    }


def cache_key_for(mode: str) -> str:
    """该线路在 Redis 精确缓存里的作用域（供需要自己读写缓存的调用方使用）。"""
    return cache_scope(normalize_mode(mode))

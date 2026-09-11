"""RAG 流水线(事件化):
缓存层 -> Agent 路由(路由 + query 改写策略) -> 直接检索与改写检索双路并集召回
-> 合并去重 -> 重排序 -> 拼接元数据 -> LLM 生成

run_events 逐步产出事件供前端可视化:
  start / cache_hit / route / rewrite / retrieve / rerank / thinking / no_evidence / token / done
"""

import asyncio
import time
from collections.abc import AsyncIterator

from pymilvus.exceptions import MilvusException

from config import settings
from core.cache import AnswerCache
from core.logger import logger
from core.prompts import NO_EVIDENCE_REPLY, REWRITE_LABELS
from llm.chat import (
    generate_answer,
    generate_direct_answer,
    rewrite_query,
    route_query,
    stream_answer,
    stream_direct_answer,
)
from retrieval.keyword_retrieval import keyword_search
from retrieval.rerank import rerank
from retrieval.vector_retrieval import vector_search


def _error_reply(reason: str = "generic") -> str:
    if reason == "milvus":
        return (
            "⚠️ 知识库服务(Milvus)当前不可用,无法完成检索。"
            "请在数据库目录下执行 `powershell -File F:\\DockerDesktopData\\start-dbs.ps1` 启动服务后重试。"
        )
    if reason == "rerank":
        return "⚠️ 重排序服务(Reranker)暂时不可用,请稍后重试,或检查网络与 API Key 配置。"
    if reason == "llm":
        return "⚠️ 大模型服务暂时不可用,请稍后重试,或检查网络与 API Key 配置。"
    return "⚠️ 服务内部出现异常,请稍后重试,详情请查看 logs 目录下的日志。"


def _chunk_text(text: str, size: int = 6) -> list[str]:
    return [text[i : i + size] for i in range(0, len(text), size)]


def _format_date(value) -> str:
    if not value:
        return "无"
    return f"{value // 10000:04d}-{value // 100 % 100:02d}-{value % 100:02d}"


def _format_amount(value) -> str:
    if value is None:
        return "无"
    return f"{value / 100:.2f}元"


def _format_record(index: int, row: dict) -> str:
    metadata = (
        f"[{index}] 类型:{row.get('ticket_type') or '无'}"
        f" | 票号:{row.get('ticket_no') or '无'}"
        f" | 相关人/买方:{row.get('person') or '无'}"
        f" | 日期:{_format_date(row.get('date_int'))}"
        f" | 金额:{_format_amount(row.get('amount_fen'))}"
        f" | 路线:{row.get('route') or '无'}"
        f" | 交易对方:{row.get('counterparty') or '无'}"
    )
    return f"{metadata}\n内容:{row.get('semantic_text') or '无'}"


def _build_context(rows: list[dict]) -> str:
    return "\n\n".join(_format_record(i + 1, r) for i, r in enumerate(rows))


def _to_source(row: dict) -> dict:
    return {
        "source_file": row.get("source_file"),
        "ticket_type": row.get("ticket_type"),
        "ticket_no": row.get("ticket_no"),
        "rerank_score": row.get("rerank_score"),
    }


def _view_row(row: dict, score_key: str) -> dict:
    return {
        "source_file": row.get("source_file"),
        "ticket_type": row.get("ticket_type"),
        "ticket_no": row.get("ticket_no"),
        "person": row.get("person"),
        "date_int": row.get("date_int"),
        "amount_fen": row.get("amount_fen"),
        "route": row.get("route"),
        "counterparty": row.get("counterparty"),
        "score": row.get(score_key),
    }


class RAGPipeline:
    """将缓存、路由、改写、检索与生成组合为完整的问答流程"""

    def __init__(self) -> None:
        self.cache = AnswerCache()

    def retrieve_union(
        self, question: str, rewritten: list[str]
    ) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
        """直接检索 + 改写检索各自做双路召回,按 id 合并去重,
        返回 (直接向量, 直接关键词, 改写向量, 改写关键词, 合并去重结果)"""
        vec_hits = vector_search(question, top_n=settings.retrieval.top_n)
        kw_hits = keyword_search(question, top_n=settings.retrieval.top_n)

        rw_vec: list[dict] = []
        rw_kw: list[dict] = []
        for query in rewritten:
            rw_vec.extend(vector_search(query, top_n=settings.retrieval.top_n))
            rw_kw.extend(keyword_search(query, top_n=settings.retrieval.top_n))

        merged: list[dict] = []
        seen: set[str] = set()
        for row in vec_hits + kw_hits + rw_vec + rw_kw:
            if row["id"] not in seen:
                seen.add(row["id"])
                merged.append(row)
        return vec_hits, kw_hits, rw_vec, rw_kw, merged

    async def _stream_llm(self, agen) -> AsyncIterator[dict]:
        """包装 LLM 流式输出:先汇总思考内容(thinking),再逐段输出回答(token)"""
        thinking: list[str] = []
        thinking_flushed = False
        collected: list[str] = []
        async for part in agen:
            if "reasoning" in part:
                thinking.append(part["reasoning"])
                continue
            if thinking and not thinking_flushed:
                thinking_flushed = True
                yield {"type": "thinking", "text": "".join(thinking)}
            yield {"type": "token", "text": part["content"]}
            collected.append(part["content"])
        if thinking and not thinking_flushed:
            yield {"type": "thinking", "text": "".join(thinking)}
        yield {"type": "llm_done", "answer": "".join(collected)}

    async def run_events(self, question: str, stream: bool = True) -> AsyncIterator[dict]:
        """执行完整流程并逐事件产出(供可视化/UI 消费)"""
        question = question.strip()
        yield {"type": "start", "question": question}

        t0 = time.perf_counter()
        cached = self.cache.lookup(question)
        logger.info(f"[缓存] 未命中,耗时 {time.perf_counter() - t0:.2f}s" if cached is None
                    else f"[缓存] 命中({cached['cache_hit']}),耗时 {time.perf_counter() - t0:.2f}s")
        if cached:
            yield {
                "type": "cache_hit",
                "mode": cached["cache_hit"],
                "similarity": cached.get("similarity"),
                "matched_question": cached.get("question"),
                "answer": cached.get("answer", ""),
                "sources": cached.get("sources", []),
            }
            for piece in _chunk_text(cached.get("answer", "")):
                yield {"type": "token", "text": piece}
            yield {
                "type": "done",
                "answer": cached.get("answer", ""),
                "sources": cached.get("sources", []),
                "cache_hit": cached["cache_hit"],
                "route": None,
                "rewrite": None,
                "conservative": False,
            }
            return

        t1 = time.perf_counter()
        need_rag, method = route_query(question)
        route = "rag" if need_rag else "direct"
        logger.info(f"[路由] {route} | 改写策略={REWRITE_LABELS[method]} | 耗时 {time.perf_counter() - t1:.2f}s")
        yield {"type": "route", "route": route, "rewrite": method}

        if route == "direct":
            t2 = time.perf_counter()
            try:
                answer = ""
                if stream:
                    async for ev in self._stream_llm(stream_direct_answer(question)):
                        if ev["type"] == "llm_done":
                            answer = ev["answer"]
                        else:
                            yield ev
                else:
                    answer, reasoning = generate_direct_answer(question)
                    if reasoning:
                        yield {"type": "thinking", "text": reasoning}
                    yield {"type": "token", "text": answer}
            except Exception as exc:
                logger.error(f"[直接回答] LLM 调用失败: {exc}")
                yield {"type": "error", "message": _error_reply("llm")}
                return
            logger.info(f"[直接回答] 完成,长度 {len(answer)} 字符 | 耗时 {time.perf_counter() - t2:.2f}s")
            self.cache.store(question, answer, [])
            yield {"type": "done", "answer": answer, "sources": [], "cache_hit": None, "route": "direct", "conservative": False}
            return

        try:
            rewritten = rewrite_query(question, method) if method != "direct" else []
        except Exception as exc:
            logger.warning(f"[改写] 失败,回退仅直接检索: {exc}")
            rewritten = []
        if rewritten:
            logger.info(f"[改写] 策略={REWRITE_LABELS[method]},生成 {len(rewritten)} 条检索 query: {rewritten}")
            yield {"type": "rewrite", "method": method, "label": REWRITE_LABELS[method], "rewritten": rewritten}
        else:
            logger.info("[改写] 未生成改写 query,仅使用直接检索")

        try:
            t3 = time.perf_counter()
            vec_hits, kw_hits, rw_vec, rw_kw, merged = self.retrieve_union(question, rewritten)
        except MilvusException as exc:
            logger.error(f"[召回] Milvus 不可用: {exc}")
            yield {"type": "error", "message": _error_reply("milvus")}
            return
        except Exception as exc:
            logger.error(f"[召回] 检索失败: {exc}")
            yield {"type": "error", "message": _error_reply("generic")}
            return
        logger.info(
            f"[召回] 直接检索:向量 {len(vec_hits)}/BM25 {len(kw_hits)};"
            f"改写检索:向量 {len(rw_vec)}/BM25 {len(rw_kw)};去重后 {len(merged)} 条"
            f" | 耗时 {time.perf_counter() - t3:.2f}s"
        )
        logger.debug(f"[召回-直接向量] {[(r['source_file'], r.get('vector_score')) for r in vec_hits]}")
        logger.debug(f"[召回-直接BM25] {[(r['source_file'], r.get('keyword_score')) for r in kw_hits]}")
        logger.debug(f"[召回-改写向量] {[(r['source_file'], r.get('vector_score')) for r in rw_vec]}")
        logger.debug(f"[召回-改写BM25] {[(r['source_file'], r.get('keyword_score')) for r in rw_kw]}")
        yield {
            "type": "retrieve",
            "method": method,
            "label": REWRITE_LABELS[method],
            "rewritten": rewritten,
            "direct_vector_hits": [_view_row(r, "vector_score") for r in vec_hits],
            "direct_keyword_hits": [_view_row(r, "keyword_score") for r in kw_hits],
            "rewrite_vector_hits": [_view_row(r, "vector_score") for r in rw_vec],
            "rewrite_keyword_hits": [_view_row(r, "keyword_score") for r in rw_kw],
            "merged_count": len(merged),
        }

        try:
            t4 = time.perf_counter()
            top_rows = rerank(question, merged, top_k=settings.rerank.top_k)
        except Exception as exc:
            logger.error(f"[重排] 失败: {exc}")
            yield {"type": "error", "message": _error_reply("rerank")}
            return
        logger.info(
            f"[重排] 候选 {len(merged)} -> 保留 {len(top_rows)} 条"
            f" (top_k={settings.rerank.top_k}, 相关度>={settings.rerank.relevance_p}) | 耗时 {time.perf_counter() - t4:.2f}s"
        )
        yield {
            "type": "rerank",
            "kept": [_view_row(r, "rerank_score") for r in top_rows],
            "merged_count": len(merged),
            "top_k": settings.rerank.top_k,
            "relevance_p": settings.rerank.relevance_p,
        }

        if not top_rows:
            logger.info("[生成] 召回均低于相关度阈值,走保守回复,不调用 LLM")
            yield {"type": "no_evidence"}
            for piece in _chunk_text(NO_EVIDENCE_REPLY):
                yield {"type": "token", "text": piece}
            self.cache.store(question, NO_EVIDENCE_REPLY, [])
            yield {"type": "done", "answer": NO_EVIDENCE_REPLY, "sources": [], "cache_hit": None, "route": "rag", "conservative": True}
            return

        t5 = time.perf_counter()
        try:
            if stream:
                answer = ""
                async for ev in self._stream_llm(stream_answer(question, _build_context(top_rows))):
                    if ev["type"] == "llm_done":
                        answer = ev["answer"]
                    else:
                        yield ev
            else:
                answer, reasoning = generate_answer(question, _build_context(top_rows))
                if reasoning:
                    yield {"type": "thinking", "text": reasoning}
                yield {"type": "token", "text": answer}
        except Exception as exc:
            logger.error(f"[生成] LLM 调用失败: {exc}")
            yield {"type": "error", "message": _error_reply("llm")}
            return
        logger.info(f"[生成] 完成,长度 {len(answer)} 字符 | 耗时 {time.perf_counter() - t5:.2f}s")

        sources = [_to_source(r) for r in top_rows]
        self.cache.store(question, answer, sources)
        yield {"type": "done", "answer": answer, "sources": sources, "cache_hit": None, "route": "rag", "conservative": False}

    async def run_and_collect(self, question: str) -> dict:
        """消费事件流,组装为最终的问答结果(供非流式接口使用)"""
        result = {
            "question": question.strip(),
            "answer": "",
            "sources": [],
            "cache_hit": None,
            "route": None,
            "rewrite": None,
        }
        async for ev in self.run_events(question, stream=False):
            t = ev["type"]
            if t == "token":
                result["answer"] += ev["text"]
            elif t == "cache_hit":
                result["cache_hit"] = ev["mode"]
                result["sources"] = ev.get("sources", [])
                result["answer"] = ev.get("answer", "")
                if ev.get("similarity") is not None:
                    result["similarity"] = ev["similarity"]
                if ev.get("matched_question"):
                    result["matched_question"] = ev["matched_question"]
            elif t == "route":
                result["route"] = ev["route"]
                result["rewrite"] = ev.get("rewrite")
            elif t == "done":
                result["answer"] = ev["answer"]
                result["sources"] = ev["sources"]
                result["conservative"] = ev.get("conservative", False)
            elif t == "error":
                result["answer"] += ev["message"]
                result["error"] = True
        return result

    async def run_stream(self, question: str) -> AsyncIterator[str]:
        """兼容接口:仅产出回答文本的流式生成器"""
        async for ev in self.run_events(question, stream=True):
            if ev["type"] == "token":
                yield ev["text"]

    def run(self, question: str) -> dict:
        """同步便捷入口(脚本/测试用)"""
        return asyncio.run(self.run_and_collect(question))

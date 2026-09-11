"""关键词召回:BM25 算法(jieba 分词),语料来自 Milvus tick 集合"""

import functools
import time

import jieba
from rank_bm25 import BM25Okapi

from config import settings
from core.logger import logger


def _tokenize(text: str) -> list[str]:
    return [t for t in jieba.lcut(text.lower()) if t.strip()]


@functools.lru_cache(maxsize=1)
def _load_index():
    """加载语料并构建 BM25 索引(进程内缓存,数据更新后重启进程生效)"""
    from core.database import get_milvus_client
    from retrieval.vector_retrieval import OUTPUT_FIELDS

    t0 = time.perf_counter()
    client = get_milvus_client()
    rows = client.query(
        collection_name=settings.milvus.collection,
        filter='ticket_type in ["flight", "invoice", "train"]',
        output_fields=OUTPUT_FIELDS,
        limit=10000,
    )
    docs = [r.get("semantic_text") or "" for r in rows]
    bm25 = BM25Okapi([_tokenize(d) for d in docs])
    logger.info(f"[BM25] 语料加载完成,共 {len(rows)} 篇 | 耗时 {time.perf_counter() - t0:.2f}s")
    return rows, bm25


def keyword_search(query: str, top_n: int | None = None) -> list[dict]:
    """BM25 关键词召回,返回带 keyword_score 的记录列表"""
    t0 = time.perf_counter()
    rows, bm25 = _load_index()
    scores = bm25.get_scores(_tokenize(query))
    order = sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)
    top_n = top_n or settings.retrieval.top_n

    hits = []
    for i in order[:top_n]:
        if scores[i] <= 0:
            break
        row = dict(rows[i])
        row["keyword_score"] = scores[i]
        hits.append(row)
    logger.info(f"[BM25召回] {len(hits)} 条 | 耗时 {time.perf_counter() - t0:.2f}s")
    logger.debug(f"[BM25召回] {[{ 'source': r['source_file'], 'score': r['keyword_score']} for r in hits]}")
    return hits

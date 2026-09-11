"""重排序模型:调用 SiliconFlow rerank 接口"""

import time

import requests

from config import settings
from core.logger import logger


def rerank(query: str, documents: list[dict], top_k: int | None = None) -> list[dict]:
    """对候选记录(含 semantic_text)按与 query 的相关性重排序,
    返回前 top_k 条且相关度 >= relevance_p 的记录"""
    if not documents:
        return []
    t0 = time.perf_counter()
    top_k = top_k or settings.rerank.top_k

    resp = requests.post(
        f"{settings.rerank.base_url.rstrip('/')}/rerank",
        headers={"Authorization": f"Bearer {settings.rerank.api_key}"},
        json={
            "model": settings.rerank.model,
            "query": query,
            "documents": [d.get("semantic_text") or "" for d in documents],
            "top_n": min(top_k, len(documents)),
        },
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"rerank 请求失败 HTTP {resp.status_code}: {resp.text[:300]}")

    results = resp.json()["results"][:top_k]
    ranked = []
    for item in results:
        row = dict(documents[item["index"]])
        row["rerank_score"] = item.get("relevance_score")
        if row["rerank_score"] is not None and row["rerank_score"] >= settings.rerank.relevance_p:
            ranked.append(row)
    logger.info(
        f"[重排] 候选 {len(documents)} 条,返回 {len(results)} 条,阈值 {settings.rerank.relevance_p} 后保留 {len(ranked)} 条"
        f" | 最高分 {max((item.get('relevance_score') or 0) for item in results):.3f} | 耗时 {time.perf_counter() - t0:.2f}s"
    )
    return ranked

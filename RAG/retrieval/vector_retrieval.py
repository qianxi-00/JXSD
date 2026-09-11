"""向量召回:基于 Milvus 向量检索"""

import time

from config import settings
from core.logger import logger
from core.database import get_milvus_client
from retrieval.embedding import embed_query

OUTPUT_FIELDS = [
    "id",
    "ticket_type",
    "ticket_no",
    "person",
    "date_int",
    "amount_fen",
    "route",
    "counterparty",
    "semantic_text",
    "source_file",
]


def vector_search(query: str, top_n: int | None = None) -> list[dict]:
    """将 query 向量化后在 Milvus 中检索,返回带 vector_score 的记录列表"""
    t0 = time.perf_counter()
    client = get_milvus_client()
    vector = embed_query(query)
    res = client.search(
        collection_name=settings.milvus.collection,
        data=[vector],
        anns_field="vec",
        limit=top_n or settings.retrieval.top_n,
        output_fields=OUTPUT_FIELDS,
    )
    hits = []
    for hit in res[0]:
        row = dict(hit.get("entity", {}))
        row["id"] = hit["id"]
        row["vector_score"] = hit.get("distance")
        hits.append(row)
    logger.info(f"[向量召回] {len(hits)} 条 | 耗时 {time.perf_counter() - t0:.2f}s")
    logger.debug(f"[向量召回] {[{ 'source': r['source_file'], 'score': r['vector_score']} for r in hits]}")
    return hits

"""向量召回:基于 Milvus 向量检索

在 RAG 链路里的位置
==================
双路召回里的**向量路**(另一路是 `retrieval/keyword_retrieval.py` 的 BM25)。
主链路 `pipeline/rag_pipeline.py::_recall` 会为每条 query(原问题 + 改写 query)
各调一次本函数,再把两路结果按 id 合并去重。

对应课案:`RAG 基础篇` 向量检索 / `RAG 优化篇` 混合召回(多路并集)一节。

集合与字段口径(换库前必读)
==========================
集合名取自 `settings.milvus.collection`(本机是 `tick`,标准答案是"票据一票一记录"),
向量字段固定叫 `vec`,由 `data_process/tick_extract.py` 建表时定义:
`FLOAT_VECTOR dim=1024` + `AUTOINDEX` + **`metric_type="COSINE"`**(`tick_extract.py:254-257`)。
两个直接推论:
1. **维度是 1024,与 `EMBEDDING_SIZE` 绑定** —— 换 embedding 模型必须重建集合并重算全部向量
   (维度校验在 `retrieval/embedding.py::_assert_dimensions`);
2. **`hit["distance"]` 在 COSINE 下其实是"相似度"**(越大越相似,取值约 -1~1),
   名字叫 distance 是老习惯。所以下面把它改名成 `vector_score` 交给上层,
   **不要**按"距离越小越好"去理解(那会得出完全相反的结论)。
   另外 README 提过:只有当向量已归一化时 IP 才等价于 COSINE,本项目**不做归一化**。
"""

import time

from config import settings
from core.logger import logger
from core.database import get_milvus_client
from retrieval.embedding import embed_query

# 检索时要 Milvus 回传的标量字段。这份清单是"一次召回取齐下游所需"的契约:
# - 结构化字段(ticket_type / ticket_no / person / date_int / amount_fen / route /
#   counterparty):拼证据元数据、给前端展示、评估时比对;
# - semantic_text:重排接口喂的就是它(`retrieval/rerank.py::_request`),少了它重排直接空转;
# - source_file / id:溯源与去重(合并去重用 id)。
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
    # ocr_text 必须取回来:课案口径里 semantic_text 只用于生成向量,
    # ocr_text 才是交给大模型的完整证据;不取的话生成阶段只能拿检索文本当证据。
    "ocr_text",
    "source_file",
]


def vector_search(
    query: str, top_n: int | None = None, filter_expr: str | None = None
) -> list[dict]:
    """将 query 向量化后在 Milvus 中检索,返回带 vector_score 的记录列表。

    filter_expr 是 Milvus 标量过滤表达式(pipeline.filters.build_milvus_filter 生成);
    传空表示不过滤——票据的 date_int / amount_fen / person 等结构化字段做召回阶段硬过滤。

    参数与返回约定:
    - `top_n` 走 `top_n or settings.retrieval.top_n`:传 `None`(或 0 这类假值)时用
      `RETRIEVAL_TOP_N`(本机 10)。注意**这是过滤之后的条数**,因为标量过滤下推到了 Milvus 服务端
      —— 与 BM25 那路"先打分后过滤"的实现不同,但两者语义一致(都返回"满足条件的 top_n 条"),
      所以上层可以放心把两路结果直接并集;
    - `filter_expr` 传 `None`/`""` 都表示不过滤:Milvus 的空表达式就是无条件;
    - 返回的是**一行一个 dict** 的新字典(不是 Milvus 内部对象),额外带
      `vector_score = hit.distance`(COSINE 相似度,越大越相关);
    - **每条结果必有 `id`**:合并去重靠它(`rag_pipeline._recall` 用 `row["id"]`),
      所以下面即使 `entity` 里没带主键也要显式补上,否则去重会 KeyError。
    - 结果数量可能少于 `top_n`:库里符合条件的行本来就不够时,少返回是正常的,不报错。
      过滤条件零命中的情况由上层 `rag_pipeline.retrieve_union` 负责回退,不在这一层处理。
    """
    t0 = time.perf_counter()
    client = get_milvus_client()
    # query 只有一条,所以下面 data=[vector] 是"单元素批次";embed_query 失败(网关 4xx/超时)
    # 会直接抛异常,由上层捕获后转成对用户的"知识库/服务不可用"提示。
    vector = embed_query(query)
    res = client.search(
        collection_name=settings.milvus.collection,
        data=[vector],
        anns_field="vec",
        limit=top_n or settings.retrieval.top_n,
        # 过滤条件下推到服务端:先在标量字段上筛掉不合格的行,再在剩下的行里做向量检索。
        # 这比"取回 top_n 再在本地过滤"正确得多 —— 否则过滤会把召回直接筛空。
        filter=filter_expr or "",
        output_fields=OUTPUT_FIELDS,
    )
    hits = []
    # res 是"每条 query 一组结果"的嵌套列表(nq 个),这里只发了 1 条 query ⇒ 取 res[0]。
    # 改成批量检索时记得同步改这里,否则会把别的 query 的命中混进来。
    for hit in res[0]:
        # 复制成新 dict:既能把 Milvus 返回对象统一成上层认识的普通字典,
        # 也避免后续(加 vector_score、BM25 那路加 keyword_score)污染底层对象。
        row = dict(hit.get("entity", {}))
        row["id"] = hit["id"]
        row["vector_score"] = hit.get("distance")
        hits.append(row)
    logger.info(
        f"[向量召回] {len(hits)} 条 | 过滤: {filter_expr or '无'} | 耗时 {time.perf_counter() - t0:.2f}s"
    )
    # 注意:f-string 的副作用是**无论日志级别是否开启都会先构造出这个列表**。
    # 行数很少时无所谓;若将来把 top_n 调到几百,这里会变成每查一次的固定开销。
    logger.debug(f"[向量召回] {[{ 'source': r['source_file'], 'score': r['vector_score']} for r in hits]}")
    return hits

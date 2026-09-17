"""关键词召回:BM25 算法(jieba 分词),语料来自 Milvus tick 集合

在 RAG 链路里的位置
==================
双路召回里的**关键词路**(另一路是 `retrieval/vector_retrieval.py` 的稠密向量)。
向量路擅长语义相近但用词不同的查询,BM25 擅长"票号、地名、人名"这类**字面必须对上**的
关键词 —— 两者并集去重后一起交给重排(`pipeline/rag_pipeline.py::_recall`)。

对应课案:`RAG 优化篇` 混合召回 / BM25 稀疏检索一节(课案自身对"票据流程要不要走 BM25"
有相反口径,本仓按"走双路"实现,差异已登记在 README)。

本文件最关键的一个坑(已有回归用例)
==================================
`BM25Okapi` 的词频平滑会让**出现在过半文档里的词得到负 IDF**
(`rank_bm25.BM25Okapi._calc_idf`:`idf = log(N-df+0.5) - log(df+0.5)`,为负时被替换成
`epsilon × 平均 IDF`,而平均 IDF 本身也可能是负的 ⇒ 命中行的总分仍为负)。
本项目语料只有 300 篇票据、且"高铁票/发票"这类词的 df 很高,因此**负分是常态**。
所以:**不能用"分数 > 0"判命中**(那样真实命中的行会被整批静默丢掉),
必须用"查询词与文档词有重叠"判命中、只用 BM25 分数排序。
回归用例:`tests/test_keyword_retrieval.py::test_common_terms_are_not_dropped_when_bm25_score_is_negative`。

另外一个反直觉的细节:`rank_bm25` 对**查询里不在语料词表内的词**贡献 0 分
(`BM25Okapi.get_scores` 里是 `self.idf.get(q) or 0`),所以 OOV 词不会报错、只是不参与打分。
(以上引用的是 `rank_bm25` 0.2.2 的实现,升级依赖时行号会漂移,按函数名找即可。)
"""

import functools
import time

import jieba
from rank_bm25 import BM25Okapi

from config import settings
from core.logger import logger
from pipeline.filters import row_matches_filters


def _tokenize(text: str) -> list[str]:
    """把文本切成 BM25 用的词序列:jieba 精确切词 → 转小写 → 丢掉纯空白 token。

    这里有两个必须守住的约定:
    1. **语料与查询必须用同一个函数切词**(下面 `_load_index` 与 `keyword_search` 都调它)。
       两侧切法不一致时不会报错,只是匹配率悄悄腰斩 —— 这是最典型的"能跑但不对";
    2. **先在整体上转小写再切词**:库里路线字段是拼音/英文站名(如 `Hefei-Wulumuqi`),
       大小写不一致会让同一条路线匹配不上。

    也因为它只是"切词",没有停用词表:像"的""是多少"这类高频词会留在查询词集合里,
    它们分数很低、排不到前面,但会让"词重叠判命中"更容易成立(这是刻意的取舍 ——
    宁可多召回一点交给重排去筛,也不要漏召回)。
    """
    return [t for t in jieba.lcut(text.lower()) if t.strip()]


@functools.lru_cache(maxsize=1)
def _load_index():
    """加载语料并构建 BM25 索引(进程内缓存,数据更新后重启进程生效)

    返回 `(rows, bm25, tokenized)` 三元组:原始行、BM25 索引、以及**与 rows 同序**的
    分词结果(供下面的"词重叠判命中"直接查,不用重复分词)。

    生命周期与失效(重要):
    - `lru_cache(maxsize=1)` + 无参 ⇒ 每个进程只建一次。`tick_extract.py` 入库 / 
      `embed_tickets.py` 回填向量之后,**已运行的进程看不到新数据**,必须重启
      (或显式 `_load_index.cache_clear()`)。评估/调试时"召回旧数据"多半就是这个原因;
    - 构建代价 = 一次全量查询 + 300 篇 jieba 分词(首次调用还要加载 jieba 词典),
      因此第一次关键词召回的耗时明显高于后续(日志里的 `[BM25] 语料加载完成` 就是它)。

    两处硬编码(换库/扩容时要看):
    - 过滤条件写死三种票据类型,与 `data_process/embed_tickets.py` 同一口径 ——
      库里真出现第四类票据时,这些行**不会进 BM25 语料**(向量路也搜不到,只有结构化 SQL 查得到);
    - `limit=10000` 是兜底:Milvus 的 query 默认上限 16384,语料超过 1 万篇时这里会
      **静默截断**(不报错,只是后半部分永远搜不到),届时要改成分页或按主键迭代。

    另注:`docs` 取的是 `semantic_text`(与做向量、喂 rerank 的文本同源)。
    空 `semantic_text` 的行分词结果是空列表 ⇒ 永远无法通过"词重叠"判定,
    等于自动被排除(与 `embed_tickets.py` 跳过空文本的口径一致)。
    """
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
    tokenized = [_tokenize(d) for d in docs]
    # 注意:语料为空时 BM25Okapi 会直接除零(`BM25._initialize` 算 avgdl 时除以 corpus_size;
    # 语料非空但所有文本都是空串时,则在 `BM25Okapi._calc_idf` 算 average_idf 时除以空词表)——
    # 也就是"库还没灌数据"时这里抛的是 ZeroDivisionError,不是可读的业务错误。
    if not any(tokenized):
        logger.warning(f"[BM25] 语料为空或全为空文本({len(rows)} 篇),跳过建索引,关键词路返回空")
        return rows, None, tokenized
    bm25 = BM25Okapi(tokenized)
    logger.info(f"[BM25] 语料加载完成,共 {len(rows)} 篇 | 耗时 {time.perf_counter() - t0:.2f}s")
    return rows, bm25, tokenized


def keyword_search(
    query: str, top_n: int | None = None, filters: dict | None = None
) -> list[dict]:
    """BM25 关键词召回,返回带 keyword_score 的记录列表。

    filters 是 pipeline.filters.extract_ticket_filters 抽出的结构化条件。
    语料与 BM25 索引在进程内缓存,过滤条件下推不到 Milvus,因此在打分后的
    选取阶段逐行过滤(row_matches_filters),与向量召回的过滤口径一致。

    与向量路的差别(读代码时别被绕住):
    - 向量路把过滤下推到 Milvus,**先筛后搜**;这里只能**先对全库打分、再逐行筛**,
      所以 `top_n` 是在过滤之后才计数的(见下面的循环),两路返回的语义才对齐;
    - 返回的 `keyword_score` 是**原始 BM25 分数,可能为负**(见模块 docstring 的说明),
      它只用于排序/展示,不要拿它和 `vector_score`(COSINE,0~1 附近)比大小。
    """
    t0 = time.perf_counter()
    rows, bm25, tokenized = _load_index()
    if bm25 is None:
        # 库还没灌数据(见 _load_index 的除零说明):关键词路直接返回空,
        # 由向量路或主流程的"零命中"分支去处理,不要在这里崩。
        logger.warning("[BM25] 索引为空,关键词召回返回空结果")
        return []
    # 判命中用 set(词重叠),打分用 list —— 别把 set 直接喂给 get_scores:
    # BM25 是按"查询词频"累加贡献的,set 会丢掉重复词、改变打分。
    query_tokens = set(_tokenize(query))
    scores = bm25.get_scores(_tokenize(query))

    # 命中的判定用"词重叠",排序用 BM25 分数。
    # 不用 scores <= 0 判定:BM25Okapi 在语料很小、或某个词出现在过半文档时,
    # 该词的 IDF 为负,真实命中的行也会得到负分,用正负判定会静默漏召回。
    #
    # 这行的执行顺序也是刻意的:先 `sorted(..., reverse=True)` 把全部文档按分数降序排好,
    # 再用 if 过滤掉没有词重叠的行 —— 结果列表天然保持"分数从高到低",
    # 下游(bm25 打分列 / 日志)依赖这个顺序。
    # 复杂度是 O(N log N + N),N=语料篇数(当前 300,可接受;到十万级要改成 Top-K 堆)。
    order = [
        i
        for i in sorted(range(len(rows)), key=lambda i: scores[i], reverse=True)
        if query_tokens & set(tokenized[i])
    ]
    top_n = top_n or settings.retrieval.top_n

    hits = []
    for i in order:
        row = rows[i]
        # filters=None/{} 都表示不过滤(extract_ticket_filters 抽取失败时会返回 {})。
        if filters and not row_matches_filters(row, filters):
            continue
        # 必须复制再挂 keyword_score:`rows` 是进程内缓存的**同一批 dict**,
        # 直接写会把分数污染进缓存,下一次调用、甚至别的调用方都会看到上一次的分数。
        row = dict(row)
        row["keyword_score"] = scores[i]
        hits.append(row)
        # 满额就停:过滤是在这里才生效的,所以计数必须在 append 之后做。
        if len(hits) >= top_n:
            break
    logger.info(
        f"[BM25召回] {len(hits)} 条 | 过滤: {filters or '无'} | 耗时 {time.perf_counter() - t0:.2f}s"
    )
    logger.debug(f"[BM25召回] {[{ 'source': r['source_file'], 'score': r['keyword_score']} for r in hits]}")
    return hits

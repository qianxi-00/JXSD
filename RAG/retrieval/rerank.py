"""重排序模型:调用 SiliconFlow rerank 接口

在 RAG 链路里的位置
==================
双路召回(向量 + BM25)按 id 合并去重后的**统一收口段**:把几十条候选交给 reranker 精排,
再卡阈值决定"哪些能当证据喂给大模型"。`pipeline/rag_pipeline.py` 在它之后只剩两件事:
拼接上下文 → 调 LLM;一条都没过阈值就**不调 LLM**,直接走保守回复(`NO_EVIDENCE_REPLY`)。
(`agentic/finance_agent.py` 的工具也调它,所以这里是两个链路的共享收口。)

模型与阈值的绑定关系(换模型必读)
================================
当前 reranker 是 `BAAI/bge-reranker-v2-m3` @ SiliconFlow,`RERANK_RELEVANCE_P=0.22`
(在 `.env` 里;注意 `config.py` 里那个 `relevance_p = 0.65` 是**旧栈 Qwen3-Reranker-4B 的默认值**,
真正生效的以 `.env` 为准)。**不同 reranker 的 `relevance_score` 尺度完全不同**:
旧栈命中票据能到 0.65 以上,而 bge-reranker 命中票据只有 0.51~0.54、负例全部 ≤0.0074 ——
沿用旧阈值会把正确答案整条丢掉。换 reranker 必须用标注集重新标定(见下"两个出口"与
`script/calibrate_rerank.py`),**不能照抄别的模型的值**。

对外两个出口(为什么要有两个)
============================
- `rerank_scores`:返回**全部候选**的原始相关度分数(不过阈值、不截断),给阈值标定用;
- `rerank`:在 `rerank_scores` 基础上做 `RERANK_RELEVANCE_P` 门控 + `top_k` 截断,给主流程用。

阈值标定需要看**完整的分数分布**(平台在哪、F1 拐点在哪),所以标定脚本
(`script/calibrate_rerank.py`)走 `rerank_scores`;主流程走 `rerank`。

对应课案:`RAG 优化篇` 重排序(rerank)+ 阈值标定一节。
"""

import time

import requests

from config import settings
from core.logger import logger


def _request(query: str, documents: list[dict]) -> list[dict]:
    """调 rerank 接口,返回接口原样的 results 列表。

    请求体里的三个约定,换服务商时最容易踩:
    - `documents` 取的是每条记录的 **`semantic_text`**(与做向量化、建 BM25 语料同源)。
      注意这与"生成阶段拿 `ocr_text` 当证据"是**两处不同口径**:重排按检索文本打分,
      生成按 OCR 全文作答。改 `semantic_text` 的生成模板(课案的结构化摘要模板)前先想清楚这一点;
    - `top_n=len(documents)` —— 明确要求**给全部候选打分**,不是"只要前 N 条"。
      `rerank_scores`(标定用)依赖这个行为才能拿到完整分数分布;
    - `timeout=60` 是**写死**的(没走 `settings.*`)。换慢网关或候选数暴涨时要记得调这里,
      否则表现是"重排这一路超时失败",而上层会把它转成给用户看的"重排服务不可用"提示。

    失败处理:非 200 直接抛 `RuntimeError`,并把响应体**截断到 300 字符**拼进消息
    (网关的错误页动辄几 KB,全量塞进异常会淹没日志)。上层
    (`pipeline/rag_pipeline.py`、`agentic/finance_agent.py`)捕获后转成用户可见文案。

    另外:这里用 `resp.json()["results"]` 直接取键 —— 网关若返回别的结构(如包了一层 `data`),
    抛的是 `KeyError` 而不是可读的错误消息,排查时先怀疑**接口契约变了**。
    """
    resp = requests.post(
        f"{settings.rerank.base_url.rstrip('/')}/rerank",
        headers={"Authorization": f"Bearer {settings.rerank.api_key}"},
        json={
            "model": settings.rerank.model,
            "query": query,
            "documents": [d.get("semantic_text") or "" for d in documents],
            "top_n": len(documents),
        },
        timeout=60,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"rerank 请求失败 HTTP {resp.status_code}: {resp.text[:300]}")
    return resp.json()["results"]


def rerank_scores(query: str, documents: list[dict]) -> list[dict]:
    """返回全部候选记录(原字段 + `rerank_score`),顺序与接口返回一致。

    不做阈值门控、不做 top_k 截断 —— 阈值标定需要看到完整的分数分布,
    主流程的取舍交给 `rerank()`。

    实现上的两个要点:
    - **接口返回的是候选下标**(`index`),不是原样的记录,所以这里用
      `documents[item["index"]]` 回填原记录(原字段一并保留,标定时才能看到票号/人名等标签)。
      也因此"接口按什么顺序返回"对正确性没有影响 —— 但本函数**刻意保持接口返回顺序不排序**,
      顺序的显式处理统一放在 `rerank()` 里做,标定脚本拿到的就是网关的原始输出;
    - `relevance_score` 缺失时填 **`None` 而不是 0**:0 会被当成"相关度为零的合法分数"参与
      阈值比较,而 `None` 表示"接口没给这个字段"(契约异常),`rerank()` 会把它排到最后并过滤掉。
    """
    if not documents:
        return []
    results = _request(query, documents)
    scored = []
    for item in results:
        row = dict(documents[item["index"]])
        row["rerank_score"] = item.get("relevance_score")
        scored.append(row)
    return scored


def rerank(query: str, documents: list[dict], top_k: int | None = None) -> list[dict]:
    """对候选记录(含 semantic_text)按与 query 的相关性重排序,
    返回前 top_k 条且相关度 >= relevance_p 的记录

    三步语义(顺序很重要,别调换):
    1. **按分数显式降序排序**(见下面的注释,不能依赖接口返回顺序);
    2. **截断到 `top_k`**(默认 `RERANK_TOP_K`,本机 8);
    3. **再过 `relevance_p` 阈值**(默认 `RERANK_RELEVANCE_P`,本机 0.22)。
       因为阈值是卡在"前 top_k 条"之内的,所以返回条数**可能少于 top_k** ——
       这正是主流程要的语义:"最多给 LLM 8 条证据,且每条都够相关";
       反过来"先过阈值再截断"会退化成"只要够相关就凑满 8 条",证据质量反而更差。

    调用方怎么用返回值:`rag_pipeline` 拿它拼上下文;**返回空列表是正常控制流**
    (不是错误),上层会不调 LLM 直接回保守话术,所以别在这里替它兜底成"至少返回一条"。
    """
    if not documents:
        return []
    t0 = time.perf_counter()
    top_k = top_k or settings.rerank.top_k

    # 显式按分数降序再截断：接口"降序返回"只是约定，不排序就截断会在 top_k 小于
    # 候选数时静默取错前 K 条（单元测试的伪数据恰好降序，钉不住这个假设）。
    #
    # `None` 用负无穷兜底:它表示"接口没给 relevance_score",参与比较时不能被当成 0
    #(那样会排到真实低分记录前面),排到最后再由阈值那步自然滤掉。
    scored = sorted(
        rerank_scores(query, documents),
        key=lambda row: row["rerank_score"] if row["rerank_score"] is not None else float("-inf"),
        reverse=True,
    )
    results = scored[:top_k]
    ranked = [
        row
        for row in results
        if row["rerank_score"] is not None and row["rerank_score"] >= settings.rerank.relevance_p
    ]
    # 这行日志是"重排到底有没有起作用"的主要证据(line 形如:
    # `[重排] 候选 38 条,返回 8 条,阈值 0.22 后保留 3 条 | 最高分 0.531 | 耗时 1.2s`)。
    # 注意 `最高分` 统计的是**截断后**的 results,不是全部候选,别拿它当"全库最高分"。
    # `default=0`:接口返回 `{"results": []}` 时 results 为空序列,没有 default 会让
    # max() 抛 ValueError —— 那会把"这次重排没结果"升级成"重排整段失败"。
    logger.info(
        f"[重排] 候选 {len(documents)} 条,返回 {len(results)} 条,阈值 {settings.rerank.relevance_p} 后保留 {len(ranked)} 条"
        f" | 最高分 {max((row.get('rerank_score') or 0 for row in results), default=0):.3f} | 耗时 {time.perf_counter() - t0:.2f}s"
    )
    return ranked

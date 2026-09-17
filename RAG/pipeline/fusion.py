"""融合线路：把基础篇、Agentic、GraphRAG 三条线路的**长处**放进同一次问答。

设计出发点（先说清"为什么融合不是简单地串起来"）：

| 线路 | 独门长处 | 单用时的短板 |
|---|---|---|
| 基础篇单轮检索 | 双路召回（向量+BM25）+ 重排门控，票据原文级证据最准；缓存与保守回复兜底 | 只认当前这一句；不含关系型知识；不含精确聚合 |
| Agentic RAG | 会自己定检索式、调 Text-to-SQL 拿**精确数值**、子代理逐份核验证据 | 慢；检索仍只在票据向量库里，问"谁和谁有关系"答不出 |
| GraphRAG | 社区摘要给出**全局/关系型**视角，实体多跳补出票据里没写在一起的关联 | 图只覆盖部分票据，覆盖不到时只会说"图谱里没有" |

融合线路的做法（三路取证 → 分层门控 → 一次生成 → 数字核验）：

    1. **票据证据**（基础篇）：按抽取到的条件做向量 + BM25 双路召回，合并去重后
       统一 cross-encoder 重排，过 `RERANK_RELEVANCE_P` 门控；
    2. **图谱证据**（GraphRAG）：抽实体 → 分层检索（社区摘要 + 多跳子图）；
    3. **结构化证据**（优化篇 Text-to-SQL）：问题是"聚合/计数/金额合计"意图时，
       走 `text_to_sql` 拿精确数值（不经 LLM 转述，直接给模型一个权威数字）；
    4. **分层门控而不是混排**：三类证据的分数**量纲不同**（重排分数负例贴 0，
       社区相似度普遍 0.6~0.7），直接混排会让图谱证据永远压过票据（或反之）。
       所以各按各的阈值门控，进提示词时**标注来源类型**；
    5. **裁决口径写进提示词**：数值类以【结构化统计】为准，票号/日期/人名等事实以
       【票据原文证据】为准，【图谱】只用于关系与整体情况；
    6. **数字核验**：生成后把答案里的数字逐个回查证据文本，证据里找不到的
       单独列出来提示"无可核对来源"（见 `verify_numbers`）。

诚实说明本线路的边界：
    - 它**不是**课案里的东西。课案只有基础篇/优化篇两条线（GraphRAG 在优化篇后半），
      融合是**本项目自己设计的第四种编排**；每一步用的都是课案里的既有能力，
      但"怎么组合、阈值怎么分层、冲突怎么裁决"是设计选择，没有官方依据。
    - 不做 agent 循环（那是线路②的活）：这里是**固定编排**，行为可预测、可回归测试；
      代价是它不会像 Agent 那样在第一次检索不理想时自己改写重试。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from config import settings
from core.logger import logger
from core.prompts import NO_EVIDENCE_REPLY, REWRITE_LABELS
from llm.chat import generate_answer, rewrite_query, route_query
from pipeline.filters import build_milvus_filter, extract_ticket_filters
from retrieval.keyword_retrieval import keyword_search
from retrieval.rerank import rerank
from retrieval.vector_retrieval import vector_search

# 聚合意图判定词表（强信号）：命中就值得多花一次 Text-to-SQL。
_AGGREGATE_STRONG = ("一共", "总共", "合计", "总计", "总和", "统计", "几张", "有多少张", "总金额")
# 弱信号：单独出现不算聚合（"票号是多少？"里也有"多少"），必须再配一个钱/数量词。
_AGGREGATE_WEAK = ("多少", "花了")
_MONEY_WORDS = ("钱", "元", "金额", "费用", "报销", "合计", "价格", "票价")
# 图谱证据的门控阈值：社区相似度落在 0.6~0.7 区间（bge-m3 量纲），
# 0.5 是"低于它基本就是随机社区"的经验下限。
GRAPH_SCORE_FLOOR = 0.5
# 图谱证据最多带几段社区摘要进提示词：再多会把票据原文挤出上下文，且社区之间高度重复。
GRAPH_MAX_COMMUNITIES = 3
# 票据证据条数上限（与 Agentic 线路的 MAX_EVIDENCE 对齐，便于两条线路横向对比）。
MAX_TICKET_EVIDENCE = 5
# 数字回查用的正则：抓 1,234.56 / 1691 / 498.90 这类金额与计数。
# 前后不能是数字或小数点，避免 "T20230702063302" 被切出一段当金额；
# 单个数字（如引用编号 [票据1] 里的 1）由 verify_numbers 单独过滤掉。
_NUMBER_RE = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![\d.])")

# 单路取证的**有界等待**秒数（真机踩坑后加的）。
# 为什么要它：`_sql_evidence` 走 psycopg 连 PG，而在本机沙箱里 libpq 的 GSS/SSPI
# 协商会**永久卡住**（实测：15 秒有界探针不返回、`connect_timeout` 对它无效，见 README §8.1）。
# 没有这道闸时，问一句聚合问题就会把整条融合线路挂满 10 分钟
# —— 票据与图谱证据明明都拿到了，却因为一条**可选**路径卡死而答不出来。
# 20 秒的取值依据：正常 SQL 往返实测 7.6~11.3s（含一次 LLM 生成 SQL），留约 2 倍余量。
SQL_EVIDENCE_TIMEOUT_S = 20.0


def _call_with_timeout(fn, seconds: float):
    """在**有界时间**内跑 fn；超时返回 (False, None)，异常返回 (False, exc)，成功 (True, 值)。

    实现是线程 + `join(timeout)`：`fn` 卡在系统调用里时无法被安全中断，
    所以超时后那个线程会继续挂着（daemon=True，不阻止进程退出）——
    这是刻意的取舍：要的是"这次请求还能返回"，而不是"杀掉那个卡住的调用"。

    为什么不用 signal.alarm / asyncio.wait_for：前者在 Windows 上不可用，
    后者要求被等待的是协程，而 psycopg 的阻塞连接是纯同步的。
    """
    import threading

    box: dict = {}

    def _worker():
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 原样带回给调用方判断
            box["error"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=seconds)
    if thread.is_alive():
        return False, None
    if "error" in box:
        return False, box["error"]
    return True, box.get("value")


@dataclass
class FusionResult:
    """融合线路的产出：答案 + **分类证据明细**（前端与评估都要能看到"这句话依据什么"）。"""

    question: str
    answer: str
    tickets: list[dict] = field(default_factory=list)
    communities: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    sql: dict | None = None
    unmatched_numbers: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    filter_expr: str = ""
    rewrite: str = "direct"
    elapsed_s: float = 0.0
    system_error: str = ""

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "tickets": self.tickets,
            "communities": self.communities,
            "relationships": self.relationships,
            "sql": self.sql,
            "unmatched_numbers": self.unmatched_numbers,
            "queries": self.queries,
            "filter_expr": self.filter_expr,
            "rewrite": self.rewrite,
            "elapsed_s": self.elapsed_s,
            "system_error": self.system_error,
        }


def needs_aggregate(question: str) -> bool:
    """问题是否像"要算一个数"（决定是否值得多花一次 Text-to-SQL）。

    刻意用**词表**而不是让 LLM 判断：这一步要快且确定，而 LLM 判定会在
    最不该失败的地方多引入一次网络调用与不确定性。漏判的代价只是"少一次精确统计"，
    误判的代价只是一次多余的结构化查询，都不是致命的。

    两档信号的原因：`多少` 在事实型问题里也常见（"票号是多少？"），
    单凭它就会给每条事实问题都加一次 SQL 调用（实测约 8 秒）。
    所以弱信号必须再配一个"钱/数量"词才算聚合意图。
    """
    if any(hint in question for hint in _AGGREGATE_STRONG):
        return True
    return any(hint in question for hint in _AGGREGATE_WEAK) and any(
        word in question for word in _MONEY_WORDS
    )


def _recall_tickets(queries: list[str], filters: dict) -> tuple[list[dict], str]:
    """票据证据：双路召回（向量 + BM25）→ 主键合并去重。返回 (候选, 过滤表达式)。"""
    filter_expr = build_milvus_filter(filters)
    recalled: dict[str, dict] = {}
    for text in queries:
        try:
            # `or None`：空串会被 Milvus 当成"语法错误的表达式"（与 rag_pipeline 同一处坑）
            hits = vector_search(text, top_n=settings.retrieval.top_n, filter_expr=filter_expr or None)
        except Exception as exc:  # noqa: BLE001 单条 query 失败不拖垮整轮
            logger.error(f"[融合] 向量召回失败,跳过该路: {text!r} -> {exc}")
            hits = []
        for row in hits:
            recalled.setdefault(str(row.get("id")), row)
        try:
            hits = keyword_search(text, top_n=settings.retrieval.top_n, filters=filters or None)
        except Exception as exc:  # noqa: BLE001 BM25 侧同理
            logger.error(f"[融合] BM25 召回失败,跳过该路: {text!r} -> {exc}")
            hits = []
        for row in hits:
            recalled.setdefault(str(row.get("id")), row)
    return list(recalled.values()), filter_expr


def _graph_evidence(question: str) -> dict:
    """图谱证据：分层检索（社区摘要 + 实体多跳）。任何异常都降级成空证据。

    为什么敢降级：图谱是**补充**证据，票据证据才是主证据。Neo4j 没起时
    整个融合线路不该跟着挂掉 —— 这是"融合"与"单跑 GraphRAG"的重要区别。
    """
    try:
        from graph_rag.retriever import retrieve_hierarchical

        return retrieve_hierarchical(question, top_k=5, max_hops=2, max_nodes=10) or {}
    except Exception as exc:  # noqa: BLE001 图谱不可用不阻断融合
        logger.warning(f"[融合] 图谱检索不可用,本次只用票据与结构化证据: {exc}")
        return {"communities": [], "nodes": [], "relationships": []}


def _sql_evidence(question: str) -> dict | None:
    """结构化证据：Text-to-SQL 拿精确数值。失败/超时/无结果都返回 None（不影响其他证据）。

    整段包在**有界等待**里（见 `_call_with_timeout`）：PG 若卡住（本机沙箱实测会），
    这里超时后按"没有结构化证据"处理，票据与图谱证据照常出答案，
    而不是让用户的提问挂满十分钟。
    """
    from agentic.text_to_sql import query_ticket_db

    def _invoke():
        return query_ticket_db.invoke({"question": question})

    ok, payload = _call_with_timeout(_invoke, SQL_EVIDENCE_TIMEOUT_S)
    if not ok:
        if payload is None:
            logger.warning(
                f"[融合] Text-to-SQL 超过 {SQL_EVIDENCE_TIMEOUT_S}s 未返回，本次不用结构化证据"
                "（PG 卡住时就是这样：查 README §8.1 的 PGGSSENCMODE 那一条）"
            )
        else:
            logger.warning(f"[融合] Text-to-SQL 失败,跳过结构化证据: {payload}")
        return None
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        logger.info(f"[融合] 结构化查询未给出可用结果: {str(payload)[:120]}")
        return None
    return payload


def verify_numbers(answer: str, evidence_text: str) -> list[str]:
    """把答案里的数字回查证据文本，返回"证据里找不到"的那些。

    这是融合线路相对其他三条的**额外一道闸**：模型在拼多来源上下文时，
    有可能把上一轮的数字、或训练语料里的常见数字写进答案。
    做法刻意保守——只做**字符串包含**判断（去掉千分位逗号后比对），
    宁可误报"无来源"，也不做数值容差匹配（那会漏掉 1691 vs 16910 这类错误）。

    单字符数字（含引用编号 [票据1] 里的 1）一律跳过：它们几乎总是序号，
    报出来只会淹没真正需要复核的数字。

    已知局限：证据写 `¥1,691.00` 而答案写 `1691` 能对上（去逗号 + 去尾零），
    但证据写 `2025年10月14日` 而答案把日期拆成 `10月14日` 时会误报；
    因此**它只作为提示，不用于改写答案的正文结论**。
    """
    haystack = evidence_text.replace(",", "")
    unmatched: list[str] = []
    for raw in _NUMBER_RE.findall(answer):
        plain = raw.replace(",", "")
        if len(plain) < 2 and "." not in plain:
            continue  # 单字符数字：序号/引用编号，不是需要核对的数值
        if plain in haystack:
            continue
        # 末尾零差异（1691.00 vs 1691）
        normalized = plain.rstrip("0").rstrip(".") if "." in plain else plain
        if normalized and normalized in haystack:
            continue
        unmatched.append(raw)
    return unmatched


def _select_communities(graph: dict) -> list[dict]:
    """按分数下限 + 条数上限挑选真正要进提示词的社区摘要。

    单独抽成函数是为了让"证据里到底放了几段"与"前端看到几段"来自**同一个列表**——
    两处各算一次的话，UI 显示 5 段、模型实际只看到 3 段，
    排查"为什么它没答出第 5 段里的信息"时会白白浪费很多时间。
    """
    rows = [
        item
        for item in (graph.get("communities") or [])
        if (item.get("score") or 0) >= GRAPH_SCORE_FLOOR
    ]
    return rows[:GRAPH_MAX_COMMUNITIES]


def _build_context(question: str, tickets: list[dict], communities: list[dict], graph: dict, sql: dict | None) -> str:
    """把三类证据拼成一段带**来源标签**的上下文（标签是裁决口径的载体）。"""
    sections = [f"用户问题：{question}"]

    if tickets:
        blocks = [
            f"[票据{index}] {json.dumps(row, ensure_ascii=False, default=str)}"
            for index, row in enumerate(tickets, start=1)
        ]
        sections.append("【票据原文证据】（事实与票号以此为准）\n" + "\n".join(blocks))

    if communities:
        lines = [
            f"[图社区{item.get('community_id')}] {item.get('summary') or ''}"
            for item in communities
        ]
        sections.append("【图谱社区摘要】（用于关系与全局视角，不作为数字依据）\n" + "\n".join(lines))

    relations = graph.get("relationships") or []
    if relations:
        lines = [
            f"{row.get('source')} --{row.get('relation')}--> {row.get('target')}" for row in relations
        ]
        sections.append("【图谱关系】（多跳推断，需与票据证据互相印证）\n" + "\n".join(lines))

    if sql:
        rows = sql.get("rows") or []
        sections.append(
            "【结构化统计】（数值以此为准，由 SQL 直接算出，未经模型转述）\n"
            f"SQL: {sql.get('sql')}\n结果: {json.dumps(rows, ensure_ascii=False, default=str)}"
        )

    sections.append(
        "回答要求：\n"
        "1. 只依据上面的证据作答，证据里没有的信息不要推测；\n"
        "2. 数值问题以【结构化统计】为准，票号/日期/人名等事实以【票据原文证据】为准，"
        "【图谱】只用于说明关系与整体情况；\n"
        "3. 多来源冲突时按上面第 2 条的优先级取舍，并说明依据来源；\n"
        "4. 每个结论后用 [票据N] / [图社区N] / [结构化统计] 标注来源编号。"
    )
    return "\n\n".join(sections)


def answer_fusion(question: str, history: list[dict] | None = None) -> FusionResult:
    """跑一次融合问答。`history` 只用于把上一轮语境拼进检索文本（见下）。"""
    started = time.perf_counter()
    result = FusionResult(question=question, answer="")

    # 多轮：把上一轮的用户问题拼进检索文本。融合线路不做代词消解模型（那是线路②的活），
    # 只用最朴素的办法——"那乐艳的呢？"单独检索必然召不回东西，把上一轮的实体
    # （万宁）与当前问题拼起来就能召回。只取上一轮，避免历史越长越跑偏。
    retrieval_question = question
    if history:
        previous_user = next(
            (item.get("content", "") for item in reversed(history) if item.get("role") == "user"),
            "",
        )
        if previous_user and previous_user != question:
            retrieval_question = f"{previous_user} {question}"
            logger.info(f"[融合] 多轮：检索文本拼接上一轮问题 -> {retrieval_question!r}")

    # ① 路由与改写（基础篇能力）：决定是否值得检索、用哪种改写策略
    try:
        need_rag, method = route_query(question)
    except Exception as exc:  # noqa: BLE001 路由失败按"要检索 + 直接检索"兜底
        logger.warning(f"[融合] 路由失败,按 RAG+直接检索 兜底: {exc}")
        need_rag, method = True, "direct"
    result.rewrite = method

    if not need_rag:
        # 直答分支：与基础篇一致——闲聊/算术问题不去查库，省一次检索与一次重排
        try:
            answer, _ = generate_answer(question, "（本次判定无需检索知识库，请直接回答。）")
        except Exception as exc:  # noqa: BLE001 直答失败也要有可读反馈
            logger.error(f"[融合] 直答失败: {exc}")
            answer = "生成回答时发生错误，请稍后重试。"
            result.system_error = f"直答失败: {exc}"
        result.answer = answer
        result.elapsed_s = round(time.perf_counter() - started, 2)
        return result

    try:
        rewritten = [] if method == "direct" else rewrite_query(question, method)
    except Exception as exc:  # noqa: BLE001 改写失败退化成只用原问题
        logger.warning(f"[融合] 改写失败,退化为直接检索: {exc}")
        rewritten = []
    queries = list(dict.fromkeys([retrieval_question, *[q for q in rewritten if q and q.strip()]]))
    result.queries = queries
    logger.info(f"[融合] 改写策略={REWRITE_LABELS.get(method, method)}，检索文本 {len(queries)} 条")

    # ② 三路取证
    filters = extract_ticket_filters(retrieval_question)
    candidates, filter_expr = _recall_tickets(queries, filters)
    result.filter_expr = filter_expr

    tickets: list[dict] = []
    if candidates:
        try:
            tickets = rerank(retrieval_question, candidates, top_k=MAX_TICKET_EVIDENCE)
        except Exception as exc:  # noqa: BLE001 重排失败要显式记成系统故障
            logger.error(f"[融合] 重排失败: {exc}")
            result.system_error = f"重排失败: {exc}"
    result.tickets = tickets

    graph = _graph_evidence(retrieval_question)
    communities = _select_communities(graph)
    result.communities = communities
    result.relationships = graph.get("relationships") or []

    sql = _sql_evidence(question) if needs_aggregate(question) else None
    result.sql = sql

    # ③ 三类证据全空 → 分"没查到"与"没查成"两种情况（与 Agentic 线路同一口径）
    if not tickets and not communities and not sql:
        if result.system_error:
            result.answer = "检索服务暂时不可用，本次未能获取任何资料，请稍后重试或联系管理员。"
        else:
            result.answer = NO_EVIDENCE_REPLY
        result.elapsed_s = round(time.perf_counter() - started, 2)
        return result

    # ④ 生成
    context = _build_context(question, tickets, communities, graph, sql)
    try:
        answer, _ = generate_answer(question, context)
    except Exception as exc:  # noqa: BLE001 生成失败不能伪装成"没查到"
        logger.error(f"[融合] 生成失败: {exc}")
        result.answer = "生成回答时发生错误，请稍后重试。"
        result.system_error = f"生成失败: {exc}"
        result.elapsed_s = round(time.perf_counter() - started, 2)
        return result

    # ⑤ 数字核验：只提示，不改写正文结论（见 verify_numbers 的局限说明）
    unmatched = verify_numbers(answer, context)
    if unmatched:
        logger.warning(f"[融合] 答案里有 {len(unmatched)} 个数字在证据中找不到: {unmatched}")
        answer = (
            f"{answer}\n\n> ⚠️ 核验提示：以下数字未在本次证据中找到可核对来源 —— "
            f"{', '.join(unmatched)}。请人工复核。"
        )
    result.answer = answer
    result.unmatched_numbers = unmatched
    result.elapsed_s = round(time.perf_counter() - started, 2)
    logger.info(
        f"[融合] 完成：票据 {len(tickets)} 条 / 社区 {len(communities)} 段 "
        f"/ 关系 {len(result.relationships)} 条 / 结构化 {'有' if sql else '无'}，"
        f"耗时 {result.elapsed_s}s"
    )
    return result

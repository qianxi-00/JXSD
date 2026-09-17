"""Agentic RAG 测试:对照优化篇课案「Deep Agents RAG / 安全保障 / 上下文管理」。

课案主流程:
QA 缓存与 FAQ 命中后直接返回 -> 未命中时 DeepAgent 选检索策略并把文本放进
search_queries -> 工具用**原问题**抽票据条件、逐条召回、按票据主键去重、
重排序并按阈值筛选 -> 至多 5 份证据写入 /retrieved/ -> 主 Agent 为每份文件
委派一次 evidence-analyst -> 证据不足时最多再检索一轮 -> 综合带来源的回答。

离线用例全部 monkeypatch 外部依赖(检索/重排/后端/Agent),真实调用放到集成用例。
"""

import pytest

from agentic import finance_agent


@pytest.fixture(autouse=True)
def _reset_retrieval_attempt():
    """每个用例前清零"本轮第几次检索"。

    为什么需要它：这是**模块级状态**（课案流程图那条"否，且未重检 / 否，已重检"分支
    靠它判定），生产上由两个问答入口 `reset_retrieval_attempt()` 清零；
    而本文件很多用例直接调 `search_financial_docs.invoke(...)`（不经过入口），
    于是计数会在用例之间累积 —— 第 2 个用例起就会拿到"已重检"的收尾文案，
    表现为与文案无关的用例莫名其妙地失败。夹具把"每个用例都是新的一次问答"这件事显式化。
    """
    finance_agent.reset_retrieval_attempt()
    yield
    finance_agent.reset_retrieval_attempt()


def doc(row_id, **kw):
    base = {
        "id": row_id,
        "ticket_type": "train",
        "ticket_no": f"NO-{row_id}",
        "person": "张三",
        "date_int": 20250305,
        "amount_fen": 43600,
        "route": "北京-上海",
        "source_file": f"data/train/{row_id}.png",
        "semantic_text": "张三 高铁票 北京 上海 436元",
        "ocr_text": "张三 高铁票 北京南 上海虹桥 436.00元",
    }
    base.update(kw)
    return base


class FakeBackend:
    def __init__(self):
        self.files: list[tuple[str, bytes]] = []

    def upload_files(self, files):
        self.files.extend(files)
        return []


class FakeCache:
    """AnswerCache 的替身。

    ⚠ 签名必须与真类保持一致（`route` 是线路作用域，见 core/cache.py）：
    替身少一个参数，改真类时就会以 TypeError 的形式炸在**这里**而不是产品代码里，
    看上去像"代码坏了"，其实是替身没跟上。
    """

    def __init__(self, hit=None):
        self.hit = hit
        self.stored: list[tuple] = []
        self.routes: list[str] = []

    def lookup(self, question, route=""):
        self.routes.append(route)
        return self.hit

    def store(self, question, answer, sources, route=""):
        self.routes.append(route)
        self.stored.append((question, answer, sources))


class TestDocumentHelpers:
    def test_document_text_prefers_ocr(self):
        assert finance_agent.document_text(doc("t1")) == "张三 高铁票 北京南 上海虹桥 436.00元"

    def test_document_text_falls_back_to_semantic(self):
        assert finance_agent.document_text(doc("t1", ocr_text="")) == "张三 高铁票 北京 上海 436元"

    def test_document_metadata_is_a_copy(self):
        row = doc("t1")
        metadata = finance_agent.document_metadata(row)
        metadata["person"] = "李四"
        assert row["person"] == "张三"

    def test_document_key_prefers_id(self):
        assert finance_agent.document_key(doc("t1")) == "t1"

    def test_document_key_falls_back_to_source_file(self):
        row = doc("t1")
        row.pop("id")
        assert finance_agent.document_key(row) == "data/train/t1.png"

    def test_evidence_markdown_has_source_fields_and_text(self):
        markdown = finance_agent.evidence_markdown(doc("t1"))
        assert "# 财务证据" in markdown
        assert "来源：data/train/t1.png" in markdown
        assert "票据字段：" in markdown
        assert "436.00元" in markdown


class TestFilterFallback:
    """原问题抽不出过滤条件时的两条兜底（多轮追问真机踩到的场景）。

    实测背景：追问"那乐艳的呢？"被模型当成 `original_query` 传进工具 —— 这句话里
    没有姓名主体，抽不出任何条件。真机对照两次跑：一次模型第二次工具调用改写出了
    "乐艳的火车票…"（于是条件与重排 query 都有实义，答出票号），一次两次都只传追问原话
    （于是全库不过滤 + 重排 0.098，答"检索两次都没命中"）。

    ⚠ 关键实测（`RAG/script/probe_rerank_query.py`）：**光把过滤条件补齐并不够** ——
    即使只召回乐艳那一张票，用追问原话做重排 query 仍然打分 0.098 < 0.22、保留 0 条；
    换成"乐艳 火车票 票号"或拼接文本才保留 1 条。所以兜底有两条：条件 + 重排 query。
    """

    def test_falls_back_to_search_query_for_filters(self, monkeypatch):
        monkeypatch.setattr(finance_agent, "vector_search", lambda *a, **kw: [])
        monkeypatch.setattr(finance_agent, "rerank", lambda *a, **kw: [])
        detail = finance_agent.retrieve_evidence(
            "那乐艳的呢？", ["乐艳 火车票 票号", "乐艳 差旅 报销凭证"]
        )
        assert detail["filters_from"].startswith("search_query:")
        # 注意：这种连写句式只能抽出票据类型，抽不出人名词（人名词要"乐艳的火车票"）。
        # 这不是本兜底的缺陷 —— 所以还需要下面那条重排 query 兜底。
        assert detail["filters"]["ticket_type"] == "train"
        assert 'ticket_type == "train"' in detail["filter_expr"]

    def test_rerank_query_falls_back_to_concatenation(self, monkeypatch):
        """原问题无实义时，重排 query 必须拼接改写文本 —— 否则打分恒约 0.1、必留 0 条。"""
        captured = {}

        def fake_rerank(query, documents, top_k=None):
            captured["query"] = query
            return documents

        monkeypatch.setattr(finance_agent, "vector_search", lambda *a, **kw: [{"id": "t1"}])
        monkeypatch.setattr(finance_agent, "rerank", fake_rerank)
        detail = finance_agent.retrieve_evidence("那乐艳的呢？", ["乐艳 火车票 票号"])
        assert captured["query"] == "那乐艳的呢？ 乐艳 火车票 票号"
        assert detail["rerank_query"] == "那乐艳的呢？ 乐艳 火车票 票号"

    def test_original_query_still_wins_when_it_has_filters(self, monkeypatch):
        """原问题能抽出条件时**不改行为**：两条兜底都不触发，重排仍用原问题。"""
        captured = {}

        def fake_vector(query, top_n=None, filter_expr=None):
            captured.setdefault("expr", filter_expr)
            return [{"id": "t1"}]

        def fake_rerank(query, documents, top_k=None):
            captured["rerank_query"] = query
            return documents

        monkeypatch.setattr(finance_agent, "vector_search", fake_vector)
        monkeypatch.setattr(finance_agent, "rerank", fake_rerank)
        detail = finance_agent.retrieve_evidence(
            "乐艳的火车票票号是多少？",
            ["某个改写成 HyDE 段落、里面写了别人的名字 张伟 的文本"],
        )
        assert detail["filters_from"] == "original_query"
        # 用的是原问题抽出的条件，而不是改写文本里那个"张伟"
        assert detail["filters"]["person"] == "乐艳"
        assert "张伟" not in (captured["expr"] or "")
        # 重排也用原问题（课案口径：按用户的问题排序，而不是按改写者的意图）
        assert captured["rerank_query"] == "乐艳的火车票票号是多少？"

    def test_no_fallback_available_keeps_empty_filters(self, monkeypatch):
        """连检索文本也抽不出条件时，保持"不过滤"（不能凭空造条件）。"""
        monkeypatch.setattr(finance_agent, "vector_search", lambda *a, **kw: [])
        monkeypatch.setattr(finance_agent, "rerank", lambda *a, **kw: [])
        detail = finance_agent.retrieve_evidence("那它的呢？", ["它 那个"])
        assert detail["filters"] == {}
        assert detail["filters_from"] == ""


class TestRetrievalAttempt:
    """课案流程图「主 Agent 验证证据是否充分?」的两条否分支必须有**代码级状态**。

    课案原话（优化篇正文 :9）："若证据不足，可改进查询后最多再调用一次检索工具；
    第二次仍不足时明确说明不足，不能补答或编造。"

    光靠提示词不够：模型看到的两次空召回结果**长得一模一样**，它没有依据判断
    "这是第一次还是第二次"，实测会出现反复改查询重试。所以工具在第二次及以后的
    空召回时回一句**明确写着"已重新检索过"**的文案，把状态直接告诉它。
    """

    def _empty_recall(self, monkeypatch):
        monkeypatch.setattr(finance_agent, "backend", FakeBackend())
        monkeypatch.setattr(finance_agent, "vector_search", lambda *a, **kw: [doc("t1")])
        monkeypatch.setattr(finance_agent, "rerank", lambda *a, **kw: [])

    def test_first_attempt_returns_plain_no_evidence(self, monkeypatch):
        self._empty_recall(monkeypatch)
        result = finance_agent.search_financial_docs.invoke(
            {"original_query": "不存在的问题", "search_queries": ["q1"]}
        )
        assert result == finance_agent.NO_EVIDENCE_ANSWER
        assert finance_agent.last_retrieval()["retrieval_attempt"] == 1

    def test_second_attempt_returns_recheck_wording(self, monkeypatch):
        self._empty_recall(monkeypatch)
        finance_agent.search_financial_docs.invoke(
            {"original_query": "不存在的问题", "search_queries": ["q1"]}
        )
        result = finance_agent.search_financial_docs.invoke(
            {"original_query": "不存在的问题", "search_queries": ["q2 改进后的检索文本"]}
        )
        assert result == finance_agent.RECHECK_NO_EVIDENCE_ANSWER
        assert result != finance_agent.NO_EVIDENCE_ANSWER
        # 文案里必须明确"已重新检索过"，否则模型无从判断该收尾
        assert "重新检索" in result
        assert finance_agent.last_retrieval()["retrieval_attempt"] == 2

    def test_third_attempt_still_recheck_wording(self, monkeypatch):
        """第三次怎么走？—— **被硬上限拦下**（见下面的用例）。这里先钉住"拦下之前"的行为：
        第 3 次调用不再进检索管线，所以它既不是"首次口径"也不再消耗检索预算。"""
        called = {"n": 0}

        def counting_vector(*a, **kw):
            called["n"] += 1
            return [doc("t1")]

        monkeypatch.setattr(finance_agent, "backend", FakeBackend())
        monkeypatch.setattr(finance_agent, "vector_search", counting_vector)
        monkeypatch.setattr(finance_agent, "rerank", lambda *a, **kw: [])
        for _ in range(3):
            result = finance_agent.search_financial_docs.invoke(
                {"original_query": "不存在的问题", "search_queries": ["q1"]}
            )
        assert result == finance_agent.RETRIEVAL_LIMIT_ANSWER
        # 关键：第 3 次**一次向量召回都没发生**（省下 embedding + rerank 的调用与费用）
        assert called["n"] == 2
        # 计数停在硬上限，不会因为被拦下而继续膨胀
        assert finance_agent._retrieval_attempt == finance_agent.MAX_RETRIEVAL_ATTEMPTS

    def test_hard_cap_does_not_raise(self, monkeypatch):
        """硬上限**不抛异常**：抛了会被 ToolRetryMiddleware 当可重试失败再重试两次，
        反而把调用次数放大（这正是加硬上限要避免的事）。"""
        monkeypatch.setattr(finance_agent, "backend", FakeBackend())
        monkeypatch.setattr(finance_agent, "vector_search", lambda *a, **kw: [])
        monkeypatch.setattr(finance_agent, "rerank", lambda *a, **kw: [])
        finance_agent._retrieval_attempt = finance_agent.MAX_RETRIEVAL_ATTEMPTS
        result = finance_agent.search_financial_docs.invoke(
            {"original_query": "不存在的问题", "search_queries": ["q1"]}
        )
        assert isinstance(result, str) and result == finance_agent.RETRIEVAL_LIMIT_ANSWER

    def test_recall_failure_wording_wins_over_attempt(self, monkeypatch):
        """系统故障（全路召回抛异常）优先于重检口径 —— "没查成"不能说成"没查到"。"""
        monkeypatch.setattr(finance_agent, "backend", FakeBackend())

        def boom(*a, **kw):
            raise RuntimeError("Milvus 挂了")

        monkeypatch.setattr(finance_agent, "vector_search", boom)
        for _ in range(2):
            result = finance_agent.search_financial_docs.invoke(
                {"original_query": "张三的火车票", "search_queries": ["q1"]}
            )
        assert result == finance_agent.RECALL_UNAVAILABLE_ANSWER

    def test_reset_function_clears_counter(self, monkeypatch):
        self._empty_recall(monkeypatch)
        finance_agent.search_financial_docs.invoke(
            {"original_query": "不存在的问题", "search_queries": ["q1"]}
        )
        finance_agent.reset_retrieval_attempt()
        result = finance_agent.search_financial_docs.invoke(
            {"original_query": "不存在的问题", "search_queries": ["q1"]}
        )
        assert result == finance_agent.NO_EVIDENCE_ANSWER

    def test_question_entry_resets_counter(self):
        """问答入口必须清零：否则第二个问题一开口就被判成"已重检"。"""
        finance_agent._retrieval_attempt = 5
        finance_agent.answer_financial_question(
            "万宁的火车票", [], slow_path=lambda q, h: "答", cache=FakeCache()
        )
        assert finance_agent._retrieval_attempt == 0

    def test_eval_entry_resets_counter(self, monkeypatch):
        finance_agent._retrieval_attempt = 5
        monkeypatch.setattr(finance_agent, "AnswerCache", lambda: FakeCache())
        finance_agent.answer_query_agentic("万宁的火车票", use_cache=False, runner=lambda q: "答")
        assert finance_agent._retrieval_attempt == 0


class TestSearchFinancialDocs:
    @pytest.fixture
    def patched(self, monkeypatch):
        backend = FakeBackend()
        monkeypatch.setattr(finance_agent, "backend", backend)
        calls = {"vector": [], "rerank": []}

        def fake_vector_search(query, top_n=None, filter_expr=None):
            calls["vector"].append({"query": query, "top_n": top_n, "filter_expr": filter_expr})
            return {
                "q1": [doc("t1"), doc("t2")],
                "q2": [doc("t2"), doc("t3")],
            }[query]

        def fake_rerank(query, documents, top_k=None):
            calls["rerank"].append({"query": query, "documents": documents, "top_k": top_k})
            return documents[:2]

        monkeypatch.setattr(finance_agent, "vector_search", fake_vector_search)
        monkeypatch.setattr(finance_agent, "rerank", fake_rerank)
        return backend, calls

    def test_writes_evidence_files_and_returns_paths(self, patched):
        backend, calls = patched
        result = finance_agent.search_financial_docs.invoke(
            {"original_query": "张三2025年的火车票", "search_queries": ["q1", "q2"]}
        )
        assert "已写入证据文件" in result
        assert len(backend.files) == 2
        paths = [path for path, _ in backend.files]
        assert all(path.startswith("/retrieved/") and path.endswith(".md") for path in paths)
        assert all(path in result for path in paths)

    def test_filter_is_built_from_original_query(self, patched):
        _, calls = patched
        finance_agent.search_financial_docs.invoke(
            {"original_query": "张三2025年的火车票", "search_queries": ["q1"]}
        )
        assert calls["vector"][0]["filter_expr"] == (
            'ticket_type == "train" and person == "张三"'
            " and date_int >= 20250101 and date_int <= 20251231"
        )

    def test_rerank_uses_original_query(self, patched):
        _, calls = patched
        finance_agent.search_financial_docs.invoke(
            {"original_query": "张三2025年的火车票", "search_queries": ["q1", "q2"]}
        )
        assert calls["rerank"][0]["query"] == "张三2025年的火车票"

    def test_dedupes_across_queries_by_document_key(self, patched):
        _, calls = patched
        finance_agent.search_financial_docs.invoke(
            {"original_query": "张三2025年的火车票", "search_queries": ["q1", "q2"]}
        )
        ids = [d["id"] for d in calls["rerank"][0]["documents"]]
        assert ids == ["t1", "t2", "t3"], "重复票据(t2)只应出现一次"

    def test_duplicate_queries_are_collapsed(self, patched):
        _, calls = patched
        finance_agent.search_financial_docs.invoke(
            {"original_query": "张三2025年的火车票", "search_queries": ["q1", "q1", "  "]}
        )
        assert [c["query"] for c in calls["vector"]] == ["q1"]

    def test_returns_no_evidence_when_threshold_filters_everything(self, monkeypatch):
        backend = FakeBackend()
        monkeypatch.setattr(finance_agent, "backend", backend)
        monkeypatch.setattr(
            finance_agent, "vector_search", lambda *a, **kw: [doc("t1")]
        )
        monkeypatch.setattr(finance_agent, "rerank", lambda *a, **kw: [])
        result = finance_agent.search_financial_docs.invoke(
            {"original_query": "不存在的问题", "search_queries": ["q1"]}
        )
        assert result == finance_agent.NO_EVIDENCE_ANSWER
        assert backend.files == []

    def test_all_recall_failures_report_system_error_not_no_evidence(self, monkeypatch):
        """全部检索文本都抛异常 ⇒ 系统故障，绝不返回"未检索到资料"。

        区分点：`NO_EVIDENCE_ANSWER` 会被用户读成"知识库里没这张票"，
        于是一次向量库故障就变成了业务结论。这里断言两句文案不同，
        且工具**不抛异常**（抛异常会让 Agent 进错误处理分支）。
        """
        backend = FakeBackend()
        monkeypatch.setattr(finance_agent, "backend", backend)

        def boom(*args, **kwargs):
            raise RuntimeError("Milvus 连接被拒绝")

        monkeypatch.setattr(finance_agent, "vector_search", boom)
        result = finance_agent.search_financial_docs.invoke(
            {"original_query": "张三2025年的火车票", "search_queries": ["q1", "q2"]}
        )
        assert result == finance_agent.RECALL_UNAVAILABLE_ANSWER
        assert result != finance_agent.NO_EVIDENCE_ANSWER
        assert backend.files == []
        detail = finance_agent.last_retrieval()
        assert detail["recall_failed"] is True
        assert detail["failed_queries"] == ["q1", "q2"]

    def test_partial_recall_failure_still_answers(self, monkeypatch):
        """只挂一路 ⇒ 可降级，用剩下的召回结果照常回答（不能升级成系统故障）。"""
        backend = FakeBackend()
        monkeypatch.setattr(finance_agent, "backend", backend)

        def half_broken(query, top_n=None, filter_expr=None):
            if query == "q1":
                raise RuntimeError("这一路挂了")
            return [doc("t2")]

        monkeypatch.setattr(finance_agent, "vector_search", half_broken)
        monkeypatch.setattr(finance_agent, "rerank", lambda *a, **kw: [doc("t2")])
        result = finance_agent.search_financial_docs.invoke(
            {"original_query": "张三2025年的火车票", "search_queries": ["q1", "q2"]}
        )
        assert result != finance_agent.RECALL_UNAVAILABLE_ANSWER
        assert "已写入证据文件" in result
        assert finance_agent.last_retrieval()["recall_failed"] is False

    def test_no_queries_is_not_a_recall_failure(self, monkeypatch):
        """模型没给出任何检索文本 ⇒ 不是基础设施故障，不能误判成系统故障。"""
        backend = FakeBackend()
        monkeypatch.setattr(finance_agent, "backend", backend)
        monkeypatch.setattr(
            finance_agent,
            "vector_search",
            lambda *a, **kw: pytest.fail("没有检索文本时不该调用 vector_search"),
        )
        result = finance_agent.search_financial_docs.invoke(
            {"original_query": "张三2025年的火车票", "search_queries": ["  ", ""]}
        )
        assert result == finance_agent.NO_EVIDENCE_ANSWER
        assert finance_agent.last_retrieval()["recall_failed"] is False


class TestUnifiedQuestionEntry:
    def test_cache_hit_short_circuits_slow_path(self):
        cache = FakeCache(hit={"cache_hit": "preset", "answer": "预设答案", "sources": []})
        called = []
        answer = finance_agent.answer_financial_question(
            "张三今年高铁票一共报销了多少钱？",
            [],
            cache=cache,
            slow_path=lambda q, h: called.append(q) or "不该被调用",
        )
        assert answer == "预设答案"
        assert called == []

    def test_slow_path_result_is_cached_and_history_updated(self):
        cache = FakeCache()
        history: list[dict] = []
        answer = finance_agent.answer_financial_question(
            "张三今年高铁票一共报销了多少钱？",
            history,
            cache=cache,
            slow_path=lambda q, h: "合计 836.00 元",
        )
        assert answer == "合计 836.00 元"
        assert cache.stored and cache.stored[0][1] == "合计 836.00 元"
        assert history[-1] == {"role": "assistant", "content": "合计 836.00 元"}
        assert history[-2]["role"] == "user"


class TestAgentConstruction:
    @pytest.fixture
    def captured(self, monkeypatch):
        captured = {}

        def fake_create(**kwargs):
            captured.update(kwargs)
            return "fake-agent"

        monkeypatch.setattr(finance_agent, "create_deep_agent", fake_create)
        return captured

    def test_build_finance_agent_wires_tools_and_subagent(self, captured):
        agent = finance_agent.build_finance_agent()
        assert agent == "fake-agent"
        assert finance_agent.search_financial_docs in captured["tools"]
        assert captured["system_prompt"] == finance_agent.RAG_SYSTEM_PROMPT
        subagents = captured["subagents"]
        assert len(subagents) == 1
        assert subagents[0]["name"] == "evidence-analyst"
        assert "read_file" in subagents[0]["system_prompt"]

    def test_build_finance_agent_accepts_extra_tools_and_middleware(self, captured):
        sentinel_tool = object()
        sentinel_middleware = object()
        finance_agent.build_finance_agent(
            middleware=[sentinel_middleware], extra_tools=[sentinel_tool], checkpointer="cp"
        )
        assert sentinel_tool in captured["tools"]
        assert captured["middleware"] == [sentinel_middleware]
        assert captured["checkpointer"] == "cp"

    def test_production_agent_adds_guardrails_and_side_effect_tools(self, captured):
        finance_agent.build_production_agent()
        names = [getattr(t, "name", None) for t in captured["tools"]]
        assert {"delete_index", "update_config"} <= set(names)
        middleware_kinds = {type(m).__name__ for m in captured["middleware"]}
        assert {
            "ContextEditingMiddleware",
            "SummarizationMiddleware",
            "ModelCallLimitMiddleware",
            "HumanInTheLoopMiddleware",
            "ModelRetryMiddleware",
            "ToolRetryMiddleware",
        } <= middleware_kinds

    def test_side_effect_tools_do_not_really_execute(self):
        assert "未删除" in finance_agent.delete_index.invoke({"index_name": "tick"})
        assert "未修改" in finance_agent.update_config.invoke({"key": "k", "value": 1})


class TestHitlInterruptSemantics:
    """审批的**语义**必须被钉住：`interrupt_on` 的布尔值由 langchain 归一化，
    语义一变（升级依赖时）就会静默失效 —— 这类安全配置不能只断言"中间件存在"。

    实测（langchain 1.4.0）：`True` 归一化成
    `{"allowed_decisions": ["approve", "edit", "reject", "respond"]}`，
    `False` 的条目**直接不出现在归一化结果里**（即不拦截）。
    """

    def test_write_tools_require_approval(self):
        config = finance_agent.review_middleware.interrupt_on
        for tool_name in ("delete_index", "update_config"):
            assert tool_name in config, f"{tool_name} 必须走人工审批"
            assert "approve" in config[tool_name]["allowed_decisions"]

    def test_read_only_tools_are_not_intercepted(self):
        config = finance_agent.review_middleware.interrupt_on
        for tool_name in ("search_financial_docs", "query_ticket_db"):
            assert tool_name not in config, f"{tool_name} 是只读工具，不该进审批名单"

    def test_approval_does_not_grant_permissions(self):
        """审批只是"这一句话放不放行"，不给调用者任何权限（课案 2692）。"""
        config = finance_agent.review_middleware.interrupt_on
        assert all("allowed_decisions" in value for value in config.values())
        assert all(
            set(value["allowed_decisions"]) <= {"approve", "edit", "reject", "respond"}
            for value in config.values()
        )


class TestAgentAnswerExtraction:
    def test_final_agent_answer_reads_last_message(self):
        class Msg:
            content = "最终回答"

        assert finance_agent.final_agent_answer({"messages": [{"role": "user"}, Msg()]}) == "最终回答"

    def test_update_history_appends_turn(self):
        history: list[dict] = []
        finance_agent.update_history(history, "问题", "答案")
        assert history == [
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "答案"},
        ]


class TestAnswerQueryAgentic:
    """给评估脚本用的入口:返回 answer / docs / candidate_ids / queries 等字段"""

    def test_returns_evaluation_shape_with_injected_runner(self, monkeypatch):
        monkeypatch.setattr(finance_agent, "AnswerCache", lambda: FakeCache())
        result = finance_agent.answer_query_agentic(
            "张三的火车票", use_cache=False, runner=lambda q: "合计 436.00 元"
        )
        assert result["answer"] == "合计 436.00 元"
        assert result["from_cache"] is False
        assert "candidate_ids" in result and "queries" in result
        # 评估结果必须显式带 recall_failed：报告据此把"检索基础设施挂了"的样本
        # 从准确率里剔出去，否则环境故障会被记成链路能力问题。
        assert result["recall_failed"] is False

    def test_cache_hit_marks_from_cache(self, monkeypatch):
        monkeypatch.setattr(
            finance_agent, "AnswerCache", lambda: FakeCache(hit={"cache_hit": "exact", "answer": "缓存答案", "sources": []})
        )
        result = finance_agent.answer_query_agentic("张三的火车票", use_cache=True)
        assert result["from_cache"] is True
        assert result["answer"] == "缓存答案"
        # 命中缓存 = 没发生检索，所以不可能是"检索失败"（否则会把缓存样本算成环境故障）
        assert result["recall_failed"] is False

    def test_agentic_entry_flags_all_recall_failures(self, monkeypatch):
        """评估入口要把"全路召回失败"透出来，与链路异常（error 字段）区分开。"""
        monkeypatch.setattr(finance_agent, "AnswerCache", lambda: FakeCache())

        def boom(*args, **kwargs):
            raise RuntimeError("Milvus 连接被拒绝")

        monkeypatch.setattr(finance_agent, "vector_search", boom)

        def fake_runner(query):
            # 模拟 Agent：真跑链路时会调用工具，工具返回系统故障文案
            finance_agent.search_financial_docs.invoke(
                {"original_query": query, "search_queries": ["q1", "q2"]}
            )
            return finance_agent.RECALL_UNAVAILABLE_ANSWER

        result = finance_agent.answer_query_agentic("张三的火车票", use_cache=False, runner=fake_runner)
        assert result["recall_failed"] is True
        # 关键区分：这是"检索失败"（recall_failed）而不是"链路抛异常"（error）
        assert result["error"] is None


class TestCacheRouteScope:
    """线路作用域：四条 RAG 线路的缓存必须互不串答案。"""

    def test_route_scope_is_forwarded_to_cache(self):
        """lookup 与 store 收到的 route 必须一致 —— 不一致会"存了但读不到"，
        而且**不报错**，只表现为缓存永远 miss，极难归因。"""
        cache = FakeCache()
        finance_agent.answer_financial_question(
            "万宁的火车票", [], slow_path=lambda q, h: "答案", cache=cache, route="fusion"
        )
        assert cache.routes == ["fusion", "fusion"]

    def test_default_route_keeps_legacy_key(self):
        """不传 route 时必须是空串：基础线路的老键（含 seed_details 播的 60 条明细）继续有效。"""
        cache = FakeCache()
        finance_agent.answer_financial_question(
            "万宁的火车票", [], slow_path=lambda q, h: "答案", cache=cache
        )
        assert cache.routes == ["", ""]

    def test_info_out_param_reports_cache_hit(self):
        """info 出参把"命中缓存"告诉编排层（返回值是 str，装不下这个信息）。"""
        cache = FakeCache(hit={"cache_hit": "exact", "answer": "缓存答案", "sources": []})
        info: dict = {}
        answer = finance_agent.answer_financial_question(
            "万宁的火车票", [], cache=cache, route="agentic", info=info
        )
        assert answer == "缓存答案"
        assert info == {"cache_hit": "exact", "from_cache": True}

    def test_info_out_param_reports_miss_on_slow_path(self):
        cache = FakeCache()
        info: dict = {}
        finance_agent.answer_financial_question(
            "万宁的火车票", [], slow_path=lambda q, h: "新答案", cache=cache, info=info
        )
        assert info == {"cache_hit": None, "from_cache": False}

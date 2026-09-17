"""RAG 流水线测试:召回过滤接入、零命中回退、事件与来源字段契约。

全部外部依赖(缓存/向量/关键词/重排/LLM)在测试内替换,不访问网络与数据库。
"""

import pytest

from pipeline import rag_pipeline


class FakeCache:
    def __init__(self):
        self.stored: list[tuple] = []

    def lookup(self, question):
        return None

    def store(self, question, answer, sources):
        self.stored.append((question, answer, sources))


class FakeSearch:
    """记录调用参数,按轮次返回预设结果的假检索函数"""

    def __init__(self, rounds):
        # rounds: list[list[dict]] — 第 N 次调用返回第 N 组结果(不足时取最后一组)
        self.rounds = rounds
        self.calls: list[dict] = []

    def __call__(self, query, **kwargs):
        self.calls.append({"query": query, **kwargs})
        index = min(len(self.calls) - 1, len(self.rounds) - 1)
        return self.rounds[index]


def _row(row_id="ticket_a", **kw):
    base = {
        "id": row_id,
        "ticket_type": "invoice",
        "ticket_no": "INV1",
        "person": "张三",
        "date_int": 20250601,
        "amount_fen": 200000,
        "route": "",
        "counterparty": "腾讯科技有限公司",
        "source_file": f"data/invoice/{row_id}.png",
        "semantic_text": "发票内容",
    }
    base.update(kw)
    return base


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.setattr(rag_pipeline, "AnswerCache", FakeCache)
    return rag_pipeline.RAGPipeline()


def _install_searches(monkeypatch, vector, keyword):
    monkeypatch.setattr(rag_pipeline, "vector_search", vector)
    monkeypatch.setattr(rag_pipeline, "keyword_search", keyword)


class TestRetrieveUnionFilters:
    def test_filter_pushed_to_both_paths(self, pipeline, monkeypatch):
        vector = FakeSearch([[_row()]])
        keyword = FakeSearch([[_row()]])
        _install_searches(monkeypatch, vector, keyword)

        result = pipeline.retrieve_union("金额超过1000元的发票", [])

        assert result["filters"] == {
            "ticket_type": "invoice",
            "amount_min_fen": 100000,
            "amount_min_operator": ">",
        }
        assert result["filter_expr"] == 'ticket_type == "invoice" and amount_fen > 100000'
        assert vector.calls[0]["filter_expr"] == result["filter_expr"]
        assert keyword.calls[0]["filters"] == result["filters"]
        assert result["filter_fallback"] is False

    def test_filter_applied_to_rewritten_queries(self, pipeline, monkeypatch):
        vector = FakeSearch([[_row()]])
        keyword = FakeSearch([[_row()]])
        _install_searches(monkeypatch, vector, keyword)

        pipeline.retrieve_union("张三2025年的火车票", ["假设答案文本"])

        # 直接检索 + 改写检索各一次,过滤条件在两条路径上都生效
        assert [c["query"] for c in vector.calls] == ["张三2025年的火车票", "假设答案文本"]
        assert {c["filter_expr"] for c in vector.calls} == {
            'ticket_type == "train" and person == "张三"'
            " and date_int >= 20250101 and date_int <= 20251231"
        }

    def test_no_conditions_no_filter(self, pipeline, monkeypatch):
        vector = FakeSearch([[_row()]])
        keyword = FakeSearch([[_row()]])
        _install_searches(monkeypatch, vector, keyword)

        result = pipeline.retrieve_union("你好", [])

        assert result["filters"] == {}
        assert result["filter_expr"] == ""
        assert vector.calls[0]["filter_expr"] is None
        assert keyword.calls[0]["filters"] is None

    def test_zero_hit_falls_back_to_unfiltered(self, pipeline, monkeypatch):
        # 第 1 轮(带过滤)空结果,第 2 轮(不带过滤)有结果
        vector = FakeSearch([[], [_row()]])
        keyword = FakeSearch([[], [_row()]])
        _install_searches(monkeypatch, vector, keyword)

        result = pipeline.retrieve_union("张三2024年的发票", [])

        assert result["filter_fallback"] is True
        assert len(result["merged"]) == 1
        assert vector.calls[0]["filter_expr"] is not None
        assert vector.calls[1]["filter_expr"] is None
        assert keyword.calls[1]["filters"] is None

    def test_no_fallback_when_no_filters(self, pipeline, monkeypatch):
        vector = FakeSearch([[]])
        keyword = FakeSearch([[]])
        _install_searches(monkeypatch, vector, keyword)

        result = pipeline.retrieve_union("你好", [])

        assert result["filter_fallback"] is False
        assert len(vector.calls) == 1

    def test_merge_dedupes_by_id(self, pipeline, monkeypatch):
        shared = _row("ticket_same")
        vector = FakeSearch([[shared, _row("ticket_v")]])
        keyword = FakeSearch([[dict(shared), _row("ticket_k")]])
        _install_searches(monkeypatch, vector, keyword)

        result = pipeline.retrieve_union("你好", [])

        ids = [r["id"] for r in result["merged"]]
        assert ids == ["ticket_same", "ticket_v", "ticket_k"]


class TestSourceAndEventFields:
    def test_view_row_carries_ticket_id(self):
        view = rag_pipeline._view_row(_row("ticket_x"), "rerank_score")
        assert view["id"] == "ticket_x"

    def test_to_source_carries_ticket_id(self):
        source = rag_pipeline._to_source(_row("ticket_x"))
        assert source["id"] == "ticket_x"
        assert source["source_file"] == "data/invoice/ticket_x.png"


class TestRunEventsFilterContract:
    @pytest.fixture
    def patched(self, pipeline, monkeypatch):
        vector = FakeSearch([[_row("ticket_a"), _row("ticket_b")]])
        keyword = FakeSearch([[_row("ticket_c")]])
        _install_searches(monkeypatch, vector, keyword)
        monkeypatch.setattr(rag_pipeline, "route_query", lambda q: (True, "direct"))
        monkeypatch.setattr(rag_pipeline, "rewrite_query", lambda q, m: [])
        monkeypatch.setattr(
            rag_pipeline, "rerank", lambda q, docs, top_k=None: [dict(docs[0], rerank_score=0.9)]
        )
        monkeypatch.setattr(
            rag_pipeline, "generate_answer", lambda q, ctx: ("回答内容 [1]", None)
        )
        return pipeline

    async def test_retrieve_event_exposes_filters(self, patched):
        events = [ev async for ev in patched.run_events("金额超过1000元的发票", stream=False)]
        retrieve = next(ev for ev in events if ev["type"] == "retrieve")
        assert retrieve["filters"]["ticket_type"] == "invoice"
        assert retrieve["filter_expr"] == 'ticket_type == "invoice" and amount_fen > 100000'
        assert retrieve["filter_fallback"] is False
        assert "id" in retrieve["direct_vector_hits"][0]

    async def test_done_event_sources_carry_ids(self, patched):
        events = [ev async for ev in patched.run_events("金额超过1000元的发票", stream=False)]
        done = next(ev for ev in events if ev["type"] == "done")
        assert done["sources"][0]["id"] == "ticket_a"

    async def test_stage_events_carry_elapsed_s(self, patched):
        """课案要求记录关键阶段耗时,事件里必须带 elapsed_s"""
        events = [ev async for ev in patched.run_events("金额超过1000元的发票", stream=False)]
        for kind in ("route", "retrieve", "rerank"):
            event = next(ev for ev in events if ev["type"] == kind)
            assert isinstance(event["elapsed_s"], float), event


class TestCacheWriteIsolation:
    """`use_cache=False`（评估口径）必须同时关掉**读和写**。

    只关读的话，评估跑出来的答案会写进 Redis 精确缓存（TTL 10 分钟），
    这段时间内真人问同一句会直接拿到评估那一跑生成的答案。
    """

    @pytest.fixture
    def patched(self, pipeline, monkeypatch):
        _install_searches(monkeypatch, FakeSearch([[_row("ticket_a")]]), FakeSearch([[_row("ticket_a")]]))
        monkeypatch.setattr(rag_pipeline, "route_query", lambda q: (True, "direct"))
        monkeypatch.setattr(rag_pipeline, "rewrite_query", lambda q, m: [])
        monkeypatch.setattr(
            rag_pipeline, "rerank", lambda q, docs, top_k=None: [dict(docs[0], rerank_score=0.9)]
        )
        monkeypatch.setattr(rag_pipeline, "generate_answer", lambda q, ctx: ("回答", None))
        return pipeline

    async def test_eval_mode_does_not_write_cache(self, patched):
        async for _ in patched.run_events("发票金额是多少", stream=False, use_cache=False):
            pass
        assert patched.cache.stored == []

    async def test_normal_mode_still_writes_cache(self, patched):
        async for _ in patched.run_events("发票金额是多少", stream=False, use_cache=True):
            pass
        assert patched.cache.stored, "正常链路必须继续写缓存（否则精确缓存层就废了）"

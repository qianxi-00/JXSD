"""评估记录器测试:把 RAG 流水线的事件流映射成评估 record。

课案「系统评估 · 第二步:记录每个阶段的真实输出」要求保存:路由结果、
filters 抽取结果、Milvus 过滤表达式、关键词/向量召回结果、合并候选、
重排顺序、最终回答、返回类型(cache/faq/rag/llm)与耗时 ——
这里是那一步的落点,产出 evaluation/stage_metrics.py 需要的 record 结构。
"""

import pytest

from evaluation.run_recorder import record_from_events


def events_for_rag_run() -> list[dict]:
    return [
        {"type": "start", "question": "张三2025年的火车票一共多少钱?"},
        {"type": "route", "route": "rag", "rewrite": "direct"},
        {
            "type": "retrieve",
            "method": "direct",
            "label": "直接检索",
            "rewritten": [],
            "filters": {"ticket_type": "train", "person": "张三", "date_start": 20250101, "date_end": 20251231},
            "filter_expr": 'ticket_type == "train" and person == "张三"',
            "filter_fallback": False,
            "direct_vector_hits": [{"id": "t1"}, {"id": "t2"}],
            "direct_keyword_hits": [{"id": "t2"}, {"id": "t3"}],
            "rewrite_vector_hits": [],
            "rewrite_keyword_hits": [{"id": "t4"}],
            "merged_count": 4,
        },
        {"type": "rerank", "kept": [{"id": "t1"}, {"id": "t3"}], "merged_count": 4, "top_k": 8, "relevance_p": 0.65},
        {"type": "token", "text": "共 2 张"},
        {"type": "done", "answer": "共 2 张,合计 836.00 元", "sources": [{"id": "t1"}], "cache_hit": None, "route": "rag", "conservative": False},
    ]


SAMPLE = {
    "question": "张三2025年的火车票一共多少钱?",
    "task_type": "structured_aggregation",
    "expected_route": "search",
    "expected_filter": {"ticket_type": "train", "person": "张三"},
    "expected_ticket_ids": ["t1", "t3"],
    "ground_truth": "共 2 张,合计 836.00 元",
    "need_citation": True,
}


class TestRecordFromRagEvents:
    def test_maps_route_filters_and_ids(self):
        record = record_from_events(events_for_rag_run(), SAMPLE, elapsed_s=1.5)
        assert record["actual_route"] == "rag"
        assert record["actual_filters"]["person"] == "张三"
        assert record["filter_expr"] == 'ticket_type == "train" and person == "张三"'
        assert record["filter_fallback"] is False
        assert record["reranked_ids"] == ["t1", "t3"]
        assert record["answer"] == "共 2 张,合计 836.00 元"
        assert record["elapsed_s"] == pytest.approx(1.5)
        assert record["question"] == SAMPLE["question"]
        assert record["expected_ticket_ids"] == ["t1", "t3"]

    def test_candidate_ids_dedupe_in_recall_order(self):
        record = record_from_events(events_for_rag_run(), SAMPLE, elapsed_s=1.0)
        # 合并顺序 = 直接向量 + 直接关键词 + 改写向量 + 改写关键词,按 id 去重
        assert record["candidate_ids"] == ["t1", "t2", "t3", "t4"]

    def test_error_event_marks_error_route(self):
        events = [{"type": "start", "question": "q"}, {"type": "error", "message": "服务不可用"}]
        record = record_from_events(events, SAMPLE, elapsed_s=0.2)
        assert record["actual_route"] == "error"
        assert "服务不可用" in record["answer"]

    def test_no_evidence_is_still_a_retrieval_run(self):
        events = [
            {"type": "route", "route": "rag", "rewrite": "direct"},
            {
                "type": "retrieve",
                "filters": {},
                "filter_expr": "",
                "filter_fallback": True,
                "direct_vector_hits": [],
                "direct_keyword_hits": [],
                "rewrite_vector_hits": [],
                "rewrite_keyword_hits": [],
                "merged_count": 0,
            },
            {"type": "rerank", "kept": [], "merged_count": 0, "top_k": 8, "relevance_p": 0.65},
            {"type": "no_evidence"},
            {"type": "done", "answer": "无法作答", "sources": [], "cache_hit": None, "route": "rag", "conservative": True},
        ]
        record = record_from_events(events, SAMPLE, elapsed_s=0.5)
        # 该检索的问题确实走了检索,只是没有达标证据,回退标记要被记下来
        assert record["actual_route"] == "rag"
        assert record["filter_fallback"] is True
        assert record["reranked_ids"] == []


class TestRecordFromCacheAndDirect:
    def test_cache_hit_record(self):
        events = [
            {"type": "cache_hit", "mode": "preset", "quality": None, "answer": "缓存答案", "sources": []},
            {"type": "token", "text": "缓存答案"},
            {"type": "done", "answer": "缓存答案", "sources": [], "cache_hit": "preset", "route": None, "conservative": False},
        ]
        record = record_from_events(events, SAMPLE, elapsed_s=0.1)
        assert record["cache_hit"] == "preset"
        assert record["actual_route"] == "cache"
        assert record["answer"] == "缓存答案"

    def test_direct_route_record(self):
        events = [
            {"type": "route", "route": "direct", "rewrite": "direct"},
            {"type": "token", "text": "1+1=2"},
            {"type": "done", "answer": "1+1=2", "sources": [], "cache_hit": None, "route": "direct", "conservative": False},
        ]
        record = record_from_events(events, SAMPLE, elapsed_s=0.3)
        assert record["actual_route"] == "direct"
        assert record["answer"] == "1+1=2"
        assert record["candidate_ids"] == []

    def test_stage_timings_are_collected(self):
        events = events_for_rag_run()
        events[2]["elapsed_s"] = 0.4
        events[3]["elapsed_s"] = 0.6
        record = record_from_events(events, SAMPLE, elapsed_s=1.5)
        assert record["stage_timings"] == {"retrieve": 0.4, "rerank": 0.6}

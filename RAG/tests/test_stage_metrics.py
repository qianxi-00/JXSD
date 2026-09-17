"""分阶段指标测试:对照基础篇课案「系统评估 · 第三步:按阶段计算指标」。

课案要求逐段评估,而不是只看最终回答:
- 路由准确率:该搜索的问题是否进入 RAG,不该搜索的是否直接回答;
- 筛选字段准确率:票据类型、人员、日期范围、金额范围、路线、票号是否抽取正确;
- Recall@K:多路召回的候选里是否包含期望票据;
- Rerank Hit@K:重排序后前 K 是否包含期望票据;
- 答案准确性:金额、日期、人员、票据数量等关键事实是否正确。

每条 record 的字段约定见 evaluation/stage_metrics.py 模块 docstring。
"""

import pytest

from evaluation.stage_metrics import (
    answer_fact_accuracy,
    extract_facts,
    fact_hit_rate,
    filter_field_accuracy,
    recall_at_k,
    rerank_hit_at_k,
    route_accuracy,
    summarize,
)


def record(**kw) -> dict:
    base = {
        "question": "问题",
        "task_type": "retrieval",
        "expected_route": "search",
        "actual_route": "rag",
        "expected_filter": {},
        "actual_filters": {},
        "expected_ticket_ids": [],
        "candidate_ids": [],
        "reranked_ids": [],
        "ground_truth": "",
        "answer": "",
    }
    base.update(kw)
    return base


class TestRouteAccuracy:
    def test_perfect(self):
        records = [
            record(expected_route="search", actual_route="rag"),
            record(expected_route="direct", actual_route="direct"),
        ]
        assert route_accuracy(records) == pytest.approx(1.0)

    def test_half(self):
        records = [
            record(expected_route="search", actual_route="rag"),
            record(expected_route="direct", actual_route="rag"),
        ]
        assert route_accuracy(records) == pytest.approx(0.5)

    def test_cache_route_counts_as_direct_answer(self):
        # 命中缓存时不再进入检索,应视作"没有走检索"
        records = [record(expected_route="search", actual_route="cache")]
        assert route_accuracy(records) == pytest.approx(0.0)

    def test_records_without_expectation_are_skipped(self):
        records = [record(expected_route=None, actual_route="rag")]
        assert route_accuracy(records) == pytest.approx(0.0)

    def test_empty_records(self):
        assert route_accuracy([]) == pytest.approx(0.0)


class TestFilterFieldAccuracy:
    def test_perfect(self):
        records = [
            record(expected_filter={"ticket_type": "train", "person": "张三"},
                   actual_filters={"ticket_type": "train", "person": "张三"}),
        ]
        report = filter_field_accuracy(records)
        assert report["overall"]["accuracy"] == pytest.approx(1.0)
        assert report["fields"]["ticket_type"]["accuracy"] == pytest.approx(1.0)

    def test_partial(self):
        records = [
            record(expected_filter={"ticket_type": "train", "person": "张三"},
                   actual_filters={"ticket_type": "train", "person": "李四"}),
        ]
        report = filter_field_accuracy(records)
        assert report["fields"]["ticket_type"]["accuracy"] == pytest.approx(1.0)
        assert report["fields"]["person"]["accuracy"] == pytest.approx(0.0)
        assert report["overall"]["accuracy"] == pytest.approx(0.5)

    def test_missing_field_counts_as_miss(self):
        records = [record(expected_filter={"person": "张三"}, actual_filters={})]
        report = filter_field_accuracy(records)
        assert report["fields"]["person"]["accuracy"] == pytest.approx(0.0)

    def test_unexpected_extra_field_is_reported(self):
        # 期望没有条件但抽出了条件(过抽取)也要能看见
        records = [record(expected_filter={}, actual_filters={"ticket_type": "train"})]
        report = filter_field_accuracy(records)
        assert report["fields"]["ticket_type"]["false_positive"] == 1

    def test_only_counts_fields_that_appear(self):
        records = [record(expected_filter={"person": "张三"}, actual_filters={"person": "张三"})]
        report = filter_field_accuracy(records)
        assert set(report["fields"]) == {"person"}


class TestRetrievalStageMetrics:
    def test_recall_at_k(self):
        records = [
            record(expected_ticket_ids=["t1", "t2"], candidate_ids=["t1", "t9"]),
            record(expected_ticket_ids=["t3"], candidate_ids=["t8"]),
        ]
        # 第一条 1/2,第二条 0 → 平均 0.25
        assert recall_at_k(records, k=5) == pytest.approx(0.25)

    def test_rerank_hit_at_k(self):
        records = [
            record(expected_ticket_ids=["t1"], reranked_ids=["t1", "t2"]),
            record(expected_ticket_ids=["t3"], reranked_ids=["t2"]),
        ]
        assert rerank_hit_at_k(records, k=5) == pytest.approx(0.5)

    def test_refusal_samples_count_as_hit_when_nothing_retrieved(self):
        records = [record(expected_ticket_ids=[], candidate_ids=[])]
        assert recall_at_k(records, k=5) == pytest.approx(1.0)


class TestAnswerFacts:
    """关键事实的口径:金额/日期/票号/数量都归一化成带前缀的字符串,
    这样"436"既能是金额也能是数量时不会混淆。"""

    def test_extracts_amounts_dates_ids_and_counts(self):
        text = "发票INV20250101开票日期2025-11-04,总金额39,802.91元,共2张。"
        facts = extract_facts(text)
        assert "金额:39802.91" in facts
        assert "日期:2025-11-04" in facts
        assert "票号:INV20250101" in facts
        assert "数量:2" in facts

    def test_amount_format_variants_normalize(self):
        assert "金额:43802.91" in extract_facts("金额 43802.91 元")
        assert "金额:1234.50" in extract_facts("合计 1,234.50元")
        assert "金额:436.00" in extract_facts("票价为436元")

    def test_date_format_variants_normalize(self):
        assert "日期:2025-11-04" in extract_facts("开票日期2025年11月4日")
        assert "日期:2025-11-04" in extract_facts("开票日期 2025/11/04")
        assert "日期:2025-11-04" in extract_facts("开票日期 2025-11-04")

    def test_hit_rate_all_present(self):
        answer = "合计 1,234.50 元,日期 2025-11-04"
        assert fact_hit_rate(answer, {"金额:1234.50", "日期:2025-11-04"}) == 1.0

    def test_hit_rate_partial(self):
        assert fact_hit_rate("合计 1,234.50 元", {"金额:1234.50", "日期:2025-11-04"}) == pytest.approx(0.5)

    def test_hit_rate_without_facts_is_one(self):
        assert fact_hit_rate("随便说说", set()) == 1.0

    def test_hit_rate_with_no_fact_in_answer(self):
        assert fact_hit_rate("不知道", {"金额:1234.50"}) == pytest.approx(0.0)

    def test_answer_fact_accuracy_averages_records(self):
        records = [
            record(ground_truth="合计 436.00 元", answer="共 436.00 元"),
            record(ground_truth="合计 436.00 元", answer="不知道"),
        ]
        assert answer_fact_accuracy(records) == pytest.approx(0.5)


class TestSummarize:
    def test_summary_contains_all_stage_metrics(self):
        records = [
            record(
                expected_route="search",
                actual_route="rag",
                expected_filter={"person": "张三"},
                actual_filters={"person": "张三"},
                expected_ticket_ids=["t1"],
                candidate_ids=["t1"],
                reranked_ids=["t1"],
                ground_truth="合计 436.00 元",
                answer="合计 436.00 元",
            )
        ]
        summary = summarize(records, k=5)
        assert summary["count"] == 1
        assert summary["route_accuracy"] == pytest.approx(1.0)
        assert summary["filter_field_accuracy"]["overall"]["accuracy"] == pytest.approx(1.0)
        assert summary["recall_at_k"] == pytest.approx(1.0)
        assert summary["rerank_hit_at_k"] == pytest.approx(1.0)
        assert summary["answer_fact_accuracy"] == pytest.approx(1.0)

    def test_summary_on_empty_records(self):
        summary = summarize([], k=5)
        assert summary["count"] == 0
        assert summary["route_accuracy"] == pytest.approx(0.0)
        assert summary["recall_at_k"] == pytest.approx(0.0)

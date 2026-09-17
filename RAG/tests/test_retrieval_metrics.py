"""排序指标测试:对照优化篇课案的 retrieval_metrics 模块。

课案要求(「RAG 评估」与「微调 · 评估指标」共用同一套口径):
- 检索与精排都是排序问题,用 Recall@K / MRR / MAP / nDCG@K,而不是准确率或 F1;
- 同一 doc_id 在结果里重复出现只计一次;
- Recall@K 在 relevant 为空(拒答类样本)时:没检索到任何票据记 1.0,否则 0.0。

课案原文给的例子:query 有两个相关文档 d1、d2,模型返回前 4 个是 [d3, d1, d5, d2],
则 Recall@4 = 1.0、MRR = 0.5、AP = 0.5、nDCG@4 ≈ 0.65。
"""

import pytest

from evaluation.retrieval_metrics import (
    average_precision,
    mrr,
    ndcg_at_k,
    recall_at_k,
    validate_k,
)

COURSE_RETRIEVED = ["d3", "d1", "d5", "d2"]
COURSE_RELEVANT = ["d1", "d2"]


class TestValidateK:
    def test_accepts_positive(self):
        validate_k(1)
        validate_k(10)

    def test_rejects_zero_and_negative(self):
        with pytest.raises(ValueError):
            validate_k(0)
        with pytest.raises(ValueError):
            validate_k(-1)


class TestRecallAtK:
    def test_course_example(self):
        assert recall_at_k(COURSE_RETRIEVED, COURSE_RELEVANT, k=4) == pytest.approx(1.0)

    def test_partial_recall(self):
        assert recall_at_k(["d1", "x", "y"], COURSE_RELEVANT, k=3) == pytest.approx(0.5)

    def test_k_truncates(self):
        assert recall_at_k(["d3", "d1", "d2"], COURSE_RELEVANT, k=1) == pytest.approx(0.0)

    def test_duplicates_counted_once(self):
        assert recall_at_k(["d1", "d1", "d1"], COURSE_RELEVANT, k=3) == pytest.approx(0.5)

    def test_empty_relevant_with_nothing_retrieved_is_one(self):
        # 拒答类样本:期望什么都没检索到
        assert recall_at_k([], [], k=5) == pytest.approx(1.0)

    def test_empty_relevant_with_something_retrieved_is_zero(self):
        assert recall_at_k(["d1"], [], k=5) == pytest.approx(0.0)

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            recall_at_k(COURSE_RETRIEVED, COURSE_RELEVANT, k=0)


class TestMRR:
    def test_course_example(self):
        assert mrr(COURSE_RETRIEVED, COURSE_RELEVANT, k=4) == pytest.approx(0.5)

    def test_first_position(self):
        assert mrr(["d2", "x"], COURSE_RELEVANT, k=2) == pytest.approx(1.0)

    def test_no_hit(self):
        assert mrr(["x", "y"], COURSE_RELEVANT, k=2) == pytest.approx(0.0)

    def test_k_truncation_drops_hit(self):
        assert mrr(["d3", "d1"], COURSE_RELEVANT, k=1) == pytest.approx(0.0)


class TestNDCG:
    def test_course_example(self):
        # dcg = 1/log2(3) + 1/log2(5);ideal = 1/log2(2) + 1/log2(3)
        assert ndcg_at_k(COURSE_RETRIEVED, COURSE_RELEVANT, k=4) == pytest.approx(0.65, abs=0.01)

    def test_perfect_ranking_is_one(self):
        assert ndcg_at_k(["d1", "d2"], COURSE_RELEVANT, k=2) == pytest.approx(1.0)

    def test_duplicate_ids_counted_once(self):
        # 课案自带的边界用例
        assert ndcg_at_k(["A", "A"], ["A"], k=2) == pytest.approx(1.0)

    def test_no_relevant_ids_returns_zero(self):
        assert ndcg_at_k(["x"], [], k=2) == pytest.approx(0.0)

    def test_no_hit_returns_zero(self):
        assert ndcg_at_k(["x", "y"], COURSE_RELEVANT, k=2) == pytest.approx(0.0)


class TestAveragePrecision:
    def test_course_example(self):
        # 命中位置 2 与 4:Precision@2=1/2、Precision@4=2/4,平均 = 0.5
        assert average_precision(COURSE_RETRIEVED, COURSE_RELEVANT, k=4) == pytest.approx(0.5)

    def test_perfect_ranking_is_one(self):
        assert average_precision(["d1", "d2"], COURSE_RELEVANT, k=2) == pytest.approx(1.0)

    def test_empty_relevant_returns_zero(self):
        assert average_precision(["x"], [], k=2) == pytest.approx(0.0)

    def test_duplicates_do_not_double_count_hits(self):
        # 课案口径:重复 doc_id 不再计命中,但它仍占用一个排名位,
        # 因此把后面的相关文档挤到了第 3 位 → AP = (1/1 + 2/3) / 2 = 5/6
        assert average_precision(["d1", "d1", "d2"], COURSE_RELEVANT, k=3) == pytest.approx(5 / 6)

    def test_duplicate_after_all_hits_keeps_full_score(self):
        assert average_precision(["d1", "d2", "d1"], COURSE_RELEVANT, k=3) == pytest.approx(1.0)

    def test_denominator_uses_min_of_relevant_and_k(self):
        # 只有 1 个相关文档、k=2 时,命中第 1 位就是满分
        assert average_precision(["d1", "d2"], ["d1", "d2", "d9"], k=2) == pytest.approx(1.0)

"""阈值标定测试:对照基础篇课案「评测集生成 / 阈值搜索」。

课案方法:
1. 正例 = 与 query 同义/近义的说法,负例 = 难负例(语义相近但不相关);
2. 把 query、正例、负例都向量化并归一化,点积得到相似度分数;
3. 对候选阈值逐个算 accuracy / precision / recall / F1,取 F1 最高的阈值。

返回值顺序与课案实现一致:`accuracy, precision, recall, f1`。
"""

import pytest

from evaluation.threshold import (
    build_pos_neg_scores,
    calculate_metrics,
    f1_plateau,
    find_optimal_threshold,
    select_hard_negatives,
)


class TestCalculateMetrics:
    def test_perfect_separation(self):
        scores = [{"query": "q", "pos": [0.9], "neg": [0.1]}]
        accuracy, precision, recall, f1 = calculate_metrics(0.5, scores)
        assert (accuracy, precision, recall, f1) == (1.0, 1.0, 1.0, 1.0)

    def test_all_wrong(self):
        scores = [{"query": "q", "pos": [0.1], "neg": [0.9]}]
        assert calculate_metrics(0.5, scores) == (0.0, 0.0, 0.0, 0.0)

    def test_mixed_counts(self):
        # 阈值 0.5:pos 中 1 个命中 1 个漏;neg 中 1 个误报 1 个正确拒绝
        scores = [{"query": "q", "pos": [0.9, 0.4], "neg": [0.6, 0.1]}]
        accuracy, precision, recall, f1 = calculate_metrics(0.5, scores)
        assert accuracy == pytest.approx(0.5)
        assert precision == pytest.approx(0.5)
        assert recall == pytest.approx(0.5)
        assert f1 == pytest.approx(0.5)

    def test_threshold_is_inclusive(self):
        # 分数恰好等于阈值算命中(score >= threshold)
        scores = [{"query": "q", "pos": [0.5], "neg": [0.5]}]
        accuracy, precision, recall, f1 = calculate_metrics(0.5, scores)
        assert (precision, recall) == (0.5, 1.0)

    def test_aggregates_across_queries(self):
        scores = [
            {"query": "q1", "pos": [0.9], "neg": [0.1]},
            {"query": "q2", "pos": [0.9], "neg": [0.9]},
        ]
        # tp=2, tn=1, fp=1, fn=0
        accuracy, precision, recall, f1 = calculate_metrics(0.5, scores)
        assert accuracy == pytest.approx(0.75)
        assert precision == pytest.approx(2 / 3)
        assert recall == pytest.approx(1.0)
        assert f1 == pytest.approx(2 * (2 / 3) * 1.0 / (2 / 3 + 1.0))

    def test_empty_dataset_does_not_divide_by_zero(self):
        # 课案原实现在空输入上会 ZeroDivisionError,这里要求退化为全 0
        assert calculate_metrics(0.5, []) == (0.0, 0.0, 0.0, 0.0)


class TestFindOptimalThreshold:
    def test_finds_separating_threshold(self):
        scores = [
            {"query": "q1", "pos": [0.9, 0.8], "neg": [0.2, 0.3]},
            {"query": "q2", "pos": [0.85], "neg": [0.15]},
        ]
        best, metrics, all_metrics = find_optimal_threshold(scores, step=0.05)
        assert metrics["f1"] == pytest.approx(1.0)
        # 最高负例是 0.3,最小正例是 0.8:任何 (0.3, 0.8] 的阈值都能完全分开
        assert 0.30 <= best <= 0.75

    def test_returns_metrics_for_every_threshold(self):
        scores = [{"query": "q", "pos": [0.9], "neg": [0.1]}]
        _, _, all_metrics = find_optimal_threshold(scores, threshold_range=(0.0, 1.0), step=0.25)
        assert [m["threshold"] for m in all_metrics] == [0.0, 0.25, 0.5, 0.75]
        for m in all_metrics:
            assert set(m) == {"threshold", "accuracy", "precision", "recall", "f1"}

    def test_best_is_argmax_of_f1(self):
        scores = [
            {"query": "q1", "pos": [0.9, 0.4], "neg": [0.6, 0.35]},
            {"query": "q2", "pos": [0.55], "neg": [0.5]},
        ]
        best, metrics, all_metrics = find_optimal_threshold(scores, step=0.05)
        assert metrics["f1"] == max(m["f1"] for m in all_metrics)
        assert best == metrics["threshold"]

    def test_prefers_plateau_center_over_left_edge(self):
        """F1 平台很宽时返回平台中点。

        标定结果是要写进 .env 当生产阈值的:平台左端点紧贴悬崖
        (再小 0.01 就掉 precision),对噪声与数据漂移不鲁棒。
        """
        # 最高负例 0.3、最低正例 0.9 → (0.3, 0.9] 内所有阈值共享同一个最高 F1
        scores = [{"query": "q", "pos": [0.9], "neg": [0.3]}]
        best, metrics, all_metrics = find_optimal_threshold(scores, step=0.05)
        plateau = [m["threshold"] for m in all_metrics if m["f1"] == pytest.approx(1.0)]
        assert len(plateau) > 1, "用例前提:必须存在一个宽平台"
        assert best != plateau[0], f"不应返回平台左端点 {plateau[0]}"
        assert best == pytest.approx(plateau[len(plateau) // 2])
        assert best == metrics["threshold"]

    def test_all_zero_f1_still_returns_a_threshold(self):
        """连一个阈值都分不开时也要给一个值,且落在搜索区间中间而不是下界。"""
        scores = [{"query": "q", "pos": [0.1], "neg": [0.9]}]
        best, metrics, _ = find_optimal_threshold(scores, threshold_range=(0.4, 0.6), step=0.05)
        assert metrics["f1"] == 0.0
        assert 0.4 <= best <= 0.6

    def test_accepts_custom_range_and_step(self):
        scores = [{"query": "q", "pos": [0.9], "neg": [0.1]}]
        _, _, all_metrics = find_optimal_threshold(scores, threshold_range=(0.4, 0.6), step=0.1)
        assert [m["threshold"] for m in all_metrics] == pytest.approx([0.4, 0.5]) or [
            round(m["threshold"], 6) for m in all_metrics
        ] == [0.4, 0.5]


class TestF1Plateau:
    """平台函数:标定报告要告诉人"当前值是否已经在平台内"。"""

    def test_returns_all_max_tied_points_in_order(self):
        metrics = [
            {"threshold": 0.5, "f1": 0.9},
            {"threshold": 0.6, "f1": 1.0},
            {"threshold": 0.7, "f1": 1.0},
            {"threshold": 0.8, "f1": 0.8},
        ]
        assert [m["threshold"] for m in f1_plateau(metrics)] == [0.6, 0.7]

    def test_single_max_gives_single_point(self):
        metrics = [{"threshold": 0.5, "f1": 0.9}, {"threshold": 0.6, "f1": 1.0}]
        assert [m["threshold"] for m in f1_plateau(metrics)] == [0.6]

    def test_empty_input_returns_empty(self):
        assert f1_plateau([]) == []

    def test_matches_find_optimal_threshold(self):
        """两处不能各算一套:find_optimal_threshold 取的就是这个平台的中点。"""
        scores = [{"query": "q", "pos": [0.9], "neg": [0.3]}]
        best, _, all_metrics = find_optimal_threshold(scores, step=0.05)
        plateau = f1_plateau(all_metrics)
        assert best == pytest.approx(plateau[len(plateau) // 2]["threshold"])


class TestBuildPosNegScores:
    @staticmethod
    def fake_embed(text: str) -> list[float]:
        table = {
            "query": [1.0, 0.0],
            "同义问法": [1.0, 0.0],
            "相似但无关": [0.0, 1.0],
            "两倍长向量": [2.0, 0.0],
        }
        return table[text]

    def test_cosine_similarity_normalized(self):
        items = [{"query": "query", "pos": ["同义问法"], "neg": ["相似但无关"]}]
        scores = build_pos_neg_scores(items, self.fake_embed)
        assert scores[0]["pos"] == [pytest.approx(1.0)]
        assert scores[0]["neg"] == [pytest.approx(0.0)]

    def test_non_unit_vectors_are_normalized(self):
        items = [{"query": "query", "pos": ["两倍长向量"], "neg": []}]
        scores = build_pos_neg_scores(items, self.fake_embed)
        assert scores[0]["pos"] == [pytest.approx(1.0)]

    def test_preserves_query_text(self):
        items = [{"query": "query", "pos": ["同义问法"], "neg": []}]
        assert build_pos_neg_scores(items, self.fake_embed)[0]["query"] == "query"


class TestSelectHardNegatives:
    """难负例挖掘:语义相近但不相关(课案 hn_mine 的 API 适配版)

    参数沿用课案:num_negatives 取几条、range_min/range_max 在哪段相似度区间里挑、
    max_score 负例相似度上限、absolute_margin 负例要比正例低多少。
    """

    CORPUS = [
        ("正例文档", 0.95),
        ("太像的候选", 0.85),
        ("较像的候选", 0.70),
        ("一般的候选", 0.60),
        ("无关候选", 0.20),
    ]

    def test_excludes_positive_and_respects_margin(self):
        picked = select_hard_negatives(
            self.CORPUS,
            positive="正例文档",
            positive_score=0.95,
            num_negatives=5,
            range_min=0,
            range_max=10,
            max_score=0.8,
            absolute_margin=0.1,
        )
        # "太像的候选" 0.85 高于 max_score(且高于 0.95-0.1),被排除
        assert picked == ["较像的候选", "一般的候选", "无关候选"]

    def test_range_min_skips_closest_candidates(self):
        picked = select_hard_negatives(
            self.CORPUS,
            positive="正例文档",
            positive_score=0.95,
            num_negatives=5,
            range_min=2,
            range_max=10,
            max_score=0.8,
            absolute_margin=0.1,
        )
        assert picked == ["一般的候选", "无关候选"]

    def test_range_max_bounds_window(self):
        picked = select_hard_negatives(
            self.CORPUS,
            positive="正例文档",
            positive_score=0.95,
            num_negatives=5,
            range_min=0,
            range_max=1,
            max_score=0.8,
            absolute_margin=0.1,
        )
        # 只看相似度最高的 1 个候选(太像的候选),被 max_score 过滤掉
        assert picked == []

    def test_num_negatives_caps_result(self):
        picked = select_hard_negatives(
            self.CORPUS,
            positive="正例文档",
            positive_score=0.95,
            num_negatives=1,
            range_min=0,
            range_max=10,
            max_score=0.8,
            absolute_margin=0.1,
        )
        assert picked == ["较像的候选"]

    def test_margin_check_skipped_without_positive_score(self):
        picked = select_hard_negatives(
            self.CORPUS,
            positive="正例文档",
            positive_score=None,
            num_negatives=5,
            range_min=0,
            range_max=10,
            max_score=0.8,
            absolute_margin=0.1,
        )
        assert picked == ["较像的候选", "一般的候选", "无关候选"]

    def test_returns_empty_when_corpus_only_has_positive(self):
        picked = select_hard_negatives(
            [("正例文档", 0.95)],
            positive="正例文档",
            positive_score=0.95,
            num_negatives=3,
        )
        assert picked == []

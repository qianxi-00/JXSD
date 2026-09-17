"""Langfuse 评估脚本的纯逻辑测试(优化篇「实验评估脚本」)。

script/ 不是包,这里用 importlib 按文件路径加载脚本模块,
只测不联网的部分:p95、条目级评估器、运行级批次汇总。
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "script" / "langfuse_evaluation.py"


def load_script_module():
    spec = importlib.util.spec_from_file_location("langfuse_evaluation", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return load_script_module()


class TestP95:
    def test_single_value(self, script):
        assert script.p95([1.2]) == pytest.approx(1.2)

    def test_nearest_rank(self, script):
        values = [float(i) for i in range(1, 101)]  # 1..100
        # nearest-rank: 第 ceil(0.95*100)=95 个 → 下标 94 → 95.0
        assert script.p95(values) == pytest.approx(95.0)

    def test_p95_below_max_for_long_tail(self, script):
        values = [1.0] * 19 + [100.0]
        assert script.p95(values) < 100.0


class TestItemEvaluator:
    def _evaluate(self, script, output, metadata=None):
        class Item:
            input = {"question": "q"}

        return script.item_evaluator(input=Item(), output=output, metadata=metadata or {})

    def test_recall_and_runtime_metrics(self, script):
        evaluations = self._evaluate(
            script,
            {
                "question": "张三的火车票",
                "response": "共 436.00 元",
                "retrieved_ids": ["t1", "t2"],
                "latency_s": 3.5,
                "search_queries": 2,
            },
            metadata={"relevant_ids": ["t1", "t9"]},
        )
        by_name = {e.name: e.value for e in evaluations}
        assert by_name["Recall@K"] == pytest.approx(0.5)
        assert by_name["latency_s"] == pytest.approx(3.5)
        assert by_name["search_queries"] == pytest.approx(2.0)

    def test_missing_metadata_gives_zero_recall(self, script):
        evaluations = self._evaluate(
            script,
            {"question": "q", "response": "a", "retrieved_ids": ["t1"], "latency_s": 1.0, "search_queries": 1},
        )
        # relevant 为空:检索到了东西就算未命中
        by_name = {e.name: e.value for e in evaluations}
        assert by_name["Recall@K"] == pytest.approx(0.0)


class TestBatchRunEvaluator:
    @staticmethod
    def item(output, evaluations=()):
        return SimpleNamespace(output=output, evaluations=list(evaluations))

    def test_batch_aggregates(self, script):
        class Ev:
            def __init__(self, name, value):
                self.name = name
                self.value = value

        results = [
            self.item(
                {"latency_s": 2.0, "search_queries": 1, "agent_error": None},
                [Ev("Recall@K", 1.0), Ev("latency_s", 2.0)],
            ),
            self.item(
                {"latency_s": 4.0, "search_queries": 3, "agent_error": "Boom"},
                [Ev("Recall@K", 0.0), Ev("latency_s", 4.0)],
            ),
        ]
        by_name = {e.name: e.value for e in script.batch_run_evaluator(item_results=results)}
        assert by_name["batch_items"] == pytest.approx(2.0)
        assert by_name["batch_mean_latency_s"] == pytest.approx(3.0)
        assert by_name["batch_p95_latency_s"] == pytest.approx(4.0)
        assert by_name["batch_mean_search_queries"] == pytest.approx(2.0)
        assert by_name["batch_agent_error_count"] == pytest.approx(1.0)
        assert by_name["batch_mean_Recall@K"] == pytest.approx(0.5)

    def test_skips_runtime_metrics_in_metric_means(self, script):
        class Ev:
            def __init__(self, name, value):
                self.name = name
                self.value = value

        results = [
            self.item({"latency_s": 1.0, "search_queries": 1}, [Ev("latency_s", 1.0), Ev("search_queries", 1.0)])
        ]
        names = {e.name for e in script.batch_run_evaluator(item_results=results)}
        # latency_s / search_queries 已经按 output 汇总过,不能再造同名 batch_mean_*
        assert "batch_mean_latency_s" in names
        assert "batch_mean_search_queries" in names

    def test_empty_results(self, script):
        evaluations = script.batch_run_evaluator(item_results=[])
        assert [e.name for e in evaluations] == ["batch_items"]
        assert evaluations[0].value == 0


class TestRagasBundleDegradation:
    def test_returns_dict_even_without_ragas(self, script):
        # ragas 装了就返回四个指标,没装就返回空字典 —— 两种情况都不能抛异常
        bundle = script._ragas_bundle()
        assert isinstance(bundle, dict)
        assert set(bundle) <= {"Context Precision", "Context Recall", "Faithfulness", "Answer Relevancy"}

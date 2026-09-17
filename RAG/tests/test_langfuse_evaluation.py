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

    def test_token_metrics_present(self, script):
        """T5：条目级要能看到 token 用量（课案要求质量与成本一起看）。"""
        evaluations = self._evaluate(
            script,
            {
                "question": "q", "response": "a", "retrieved_ids": [], "latency_s": 1.0,
                "search_queries": 1,
                "tokens": {"llm_calls": 3, "input_tokens": 1200, "output_tokens": 340, "total_tokens": 1540},
                "cost": None,
            },
        )
        by_name = {e.name: e.value for e in evaluations}
        assert by_name["tokens_in"] == pytest.approx(1200)
        assert by_name["tokens_out"] == pytest.approx(340)
        assert by_name["tokens_total"] == pytest.approx(1540)
        assert by_name["llm_calls"] == pytest.approx(3)
        # 没填单价 ⇒ 不出 cost 这一条（报 0 会被读成"成本为零"）
        assert "cost" not in by_name

    def test_cost_metric_only_when_known(self, script):
        """填了单价（cost 非 None）时才出 cost。"""
        evaluations = self._evaluate(
            script,
            {
                "question": "q", "response": "a", "retrieved_ids": [], "latency_s": 1.0,
                "search_queries": 1, "tokens": {}, "cost": 0.0123,
            },
        )
        assert {e.name: e.value for e in evaluations}["cost"] == pytest.approx(0.0123)

    def test_missing_tokens_field_is_tolerated(self, script):
        """老记录/注入 runner 的离线样本没有 tokens 字段 ⇒ 记 0，不能 KeyError。"""
        evaluations = self._evaluate(
            script, {"question": "q", "response": "a", "retrieved_ids": [], "latency_s": 1.0, "search_queries": 0}
        )
        by_name = {e.name: e.value for e in evaluations}
        assert by_name["tokens_total"] == pytest.approx(0.0)
        assert by_name["llm_calls"] == pytest.approx(0.0)


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

    def test_batch_aggregates_tokens_and_cost(self, script):
        """T5：批次侧的 token/成本来自**条目级指标的自动汇总**（而不是另写一套）。

        所以这条用例的构造要点是：条目级 evaluations 里得有 tokens_* / cost
        —— 那正是 `item_evaluator` 产出的东西。
        """
        class Ev:
            def __init__(self, name, value):
                self.name = name
                self.value = value

        results = [
            self.item(
                {"latency_s": 2.0, "search_queries": 1, "agent_error": None},
                [Ev("tokens_in", 100.0), Ev("tokens_out", 10.0), Ev("tokens_total", 110.0),
                 Ev("llm_calls", 2.0), Ev("cost", 0.002)],
            ),
            self.item(
                {"latency_s": 4.0, "search_queries": 2, "agent_error": None},
                [Ev("tokens_in", 300.0), Ev("tokens_out", 30.0), Ev("tokens_total", 330.0),
                 Ev("llm_calls", 4.0), Ev("cost", 0.006)],
            ),
        ]
        by_name = {e.name: e.value for e in script.batch_run_evaluator(item_results=results)}
        assert by_name["batch_mean_tokens_in"] == pytest.approx(200.0)
        assert by_name["batch_mean_tokens_out"] == pytest.approx(20.0)
        assert by_name["batch_mean_tokens_total"] == pytest.approx(220.0)
        assert by_name["batch_mean_llm_calls"] == pytest.approx(3.0)
        assert by_name["batch_mean_cost"] == pytest.approx(0.004)

    def test_batch_omits_cost_when_unpriced(self, script):
        """没填单价 ⇒ 条目级不出 cost ⇒ 批次侧也就没有成本指标。

        （宁可没有，也不报一个会被读成"零成本"的数。）
        """
        class Ev:
            def __init__(self, name, value):
                self.name = name
                self.value = value

        results = [
            self.item(
                {"latency_s": 2.0, "search_queries": 1, "agent_error": None},
                [Ev("tokens_total", 15.0)],
            ),
        ]
        by_name = {e.name: e.value for e in script.batch_run_evaluator(item_results=results)}
        assert by_name["batch_mean_tokens_total"] == pytest.approx(15.0)
        assert "batch_mean_cost" not in by_name

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


class TestJudgeModelIsSeparate:
    """Ragas 的评判模型必须走 `eval_llm_target()`（课案：评估模型与生成模型分开配置）。

    为什么专门钉这条：脚本的日志级别在生产运行时可能只放 WARNING/ERROR，
    "这次到底用哪个模型评判的"从终端看不出来；而它直接决定评估结论可不可信
    （用生成模型给自己打分 ⇒ 系统性偏高）。这里用打桩把 llm_factory 的入参抓下来。
    """

    def _install_fakes(self, script, monkeypatch, captured: dict):
        import ragas.embeddings as ragas_emb
        import ragas.llms as ragas_llms

        class FakeLLM:
            def __init__(self, model, **kwargs):
                captured["model"] = model
                captured["factory_kwargs"] = kwargs

        class FakeMetric:
            def __init__(self, **kwargs):
                captured.setdefault("metrics", []).append(type(self).__name__)

        def fake_factory(model, **kwargs):
            return FakeLLM(model, **kwargs)

        class FakeEmbeddings:
            def __init__(self, **kwargs):
                captured["embeddings"] = kwargs

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                captured.setdefault("clients", []).append(kwargs)

        monkeypatch.setattr(ragas_llms, "llm_factory", fake_factory)
        monkeypatch.setattr(ragas_emb, "OpenAIEmbeddings", FakeEmbeddings)
        monkeypatch.setattr("openai.AsyncOpenAI", FakeAsyncOpenAI)
        # 四指标类在 metric 模块里 import，这里用假类替掉 collections 里的同名符号
        import ragas.metrics.collections as coll

        for name in ("AnswerRelevancy", "ContextPrecision", "ContextRecall", "Faithfulness"):
            monkeypatch.setattr(coll, name, FakeMetric, raising=False)
        monkeypatch.setattr(script, "_ragas_metrics", None)

    def test_llm_factory_receives_eval_model(self, script, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(script.settings.eval_llm, "model", "judge-model-z")
        monkeypatch.setattr(script.settings.eval_llm, "max_tokens", 4096)
        monkeypatch.setattr(script.settings.eval_llm, "api_key", "")
        monkeypatch.setattr(script.settings.eval_llm, "base_url", "")
        monkeypatch.setattr(script.settings.llm, "api_key", "gen-key")
        monkeypatch.setattr(script.settings.llm, "base_url", "https://gen.example/v1")
        self._install_fakes(script, monkeypatch, captured)

        script._ragas_bundle()
        assert captured["model"] == "judge-model-z", "评判模型必须来自 EVAL_LLM_*"
        # 密钥/网关允许逐项回退到生成侧
        client_kwargs = captured["clients"][0]
        assert client_kwargs["api_key"] == "gen-key"
        assert client_kwargs["base_url"] == "https://gen.example/v1"
        # 关思考的开关要一直带着（换回思考模型时不加它会大面积截断，实测过）
        assert captured["factory_kwargs"]["extra_body"] == {"thinking": {"type": "disabled"}}
        assert captured["factory_kwargs"]["max_tokens"] == 4096

    def test_falls_back_to_generation_model_when_unset(self, script, monkeypatch):
        captured: dict = {}
        monkeypatch.setattr(script.settings.eval_llm, "model", "")
        monkeypatch.setattr(script.settings.llm, "model", "gen-model-w")
        monkeypatch.setattr(script.settings.llm, "max_tokens", 2048)
        self._install_fakes(script, monkeypatch, captured)

        script._ragas_bundle()
        assert captured["model"] == "gen-model-w"
        assert captured["factory_kwargs"]["max_tokens"] == 2048

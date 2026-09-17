"""大模型评分解析测试:对照基础篇课案「系统评估 · 第四步」。

课案的 EVALUATION_PROMPT 要求模型输出:
    Feedback: {{针对标准的反馈}} [RESULT] {{1 到 5 的整数}}
这里锁住解析行为:必须能取出 [RESULT] 后的 1~5 整数,取不到或越界返回 None。

后半段（`TestJudgeUsesEvalModel`）钉的是**评判模型与生成模型分家**：
优化篇课案要求两者分开配置，避免同一个模型给自己打分偏高（自我偏好）。
"""

import pytest

from config import settings as cfg_settings
from evaluation import llm_judge
from evaluation.llm_judge import parse_result_score


class TestParseResultScore:
    def test_course_format(self):
        text = "Feedback: 回答与参考答案一致,金额与张数都正确 [RESULT] 5"
        assert parse_result_score(text) == 5

    def test_lowercase_and_spacing(self):
        assert parse_result_score("feedback: ok\n[result]   3") == 3

    def test_full_width_brackets(self):
        assert parse_result_score("Feedback: 部分正确【RESULT】4") == 4

    def test_score_with_suffix(self):
        assert parse_result_score("[RESULT] 2分") == 2

    def test_missing_marker_returns_none(self):
        assert parse_result_score("Feedback: 我觉得还行,给 4 分") is None

    def test_out_of_range_returns_none(self):
        assert parse_result_score("[RESULT] 6") is None
        assert parse_result_score("[RESULT] 0") is None

    def test_empty_returns_none(self):
        assert parse_result_score("") is None

    def test_prefers_first_marker(self):
        assert parse_result_score("[RESULT] 4 ... [RESULT] 5") == 4


class Recorder:
    """假 OpenAI 客户端：记下 create 的入参，回一段可解析的评分文本。"""

    def __init__(self):
        self.kwargs: dict = {}
        self.init_kwargs: dict = {}

    def create(self, **kwargs):
        self.kwargs = kwargs
        msg = type("M", (), {"content": "Feedback: ok [RESULT] 5"})()
        return type("R", (), {"choices": [type("C", (), {"message": msg})()]})()


@pytest.fixture
def recorder(monkeypatch):
    rec = Recorder()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            rec.init_kwargs = kwargs
            self.chat = type("Ch", (), {"completions": rec})()

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    return rec


class TestJudgeUsesEvalModel:
    def test_uses_eval_model_when_configured(self, recorder, monkeypatch):
        """配了 EVAL_LLM_MODEL 就必须用它，而不是生成模型。"""
        monkeypatch.setattr(cfg_settings.eval_llm, "model", "judge-model-x")
        monkeypatch.setattr(cfg_settings.eval_llm, "temperature", 0.0)
        result = llm_judge.score_answer(question="q", answer="a", ground_truth="gt")
        assert result["score"] == 5
        assert recorder.kwargs["model"] == "judge-model-x"
        assert recorder.kwargs["temperature"] == 0.0

    def test_eval_credentials_fall_back_per_field(self, recorder, monkeypatch):
        """只配模型、不配密钥/网关时，逐项回退到生成侧配置（同一家网关的常见用法）。"""
        monkeypatch.setattr(cfg_settings.eval_llm, "model", "judge-model-x")
        monkeypatch.setattr(cfg_settings.eval_llm, "api_key", "")
        monkeypatch.setattr(cfg_settings.eval_llm, "base_url", "")
        monkeypatch.setattr(cfg_settings.llm, "api_key", "gen-key")
        monkeypatch.setattr(cfg_settings.llm, "base_url", "https://gen.example/v1")
        llm_judge.score_answer(question="q", answer="a", ground_truth="gt")
        assert recorder.init_kwargs["api_key"] == "gen-key"
        assert recorder.init_kwargs["base_url"] == "https://gen.example/v1"

    def test_warns_when_falling_back_to_generation_model(self, recorder, monkeypatch, caplog=None):
        """未配置时必须**留痕**：否则"评判用的就是生成模型"这件事完全看不出来。"""
        monkeypatch.setattr(cfg_settings.eval_llm, "model", "")
        monkeypatch.setattr(cfg_settings.llm, "model", "gen-model-y")
        logged: list[str] = []
        monkeypatch.setattr(llm_judge.logger, "warning", lambda msg, *a, **kw: logged.append(str(msg) % a if a else str(msg)))
        llm_judge.score_answer(question="q", answer="a", ground_truth="gt")
        assert recorder.kwargs["model"] == "gen-model-y"
        assert any("EVAL_LLM_MODEL" in m and "自我偏好" in m for m in logged), logged

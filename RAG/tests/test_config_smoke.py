"""配置加载冒烟测试：验证根目录 config.py / .env 能被 RAG 子项目正常读取。

这是既有行为的回归护栏（characterization test），不引入新行为。
"""

from config import settings


def test_flat_and_grouped_settings_load():
    """扁平字段与 RAG 分组字段都能访问"""
    assert settings.llm is not None
    assert settings.retrieval is not None
    assert isinstance(settings.retrieval.top_n, int) and settings.retrieval.top_n > 0
    assert isinstance(settings.rerank.top_k, int) and settings.rerank.top_k > 0
    assert 0.0 < settings.rerank.relevance_p < 1.0
    assert 0.0 < settings.redis.sim_threshold < 1.0
    assert settings.redis.exact_ttl > 0
    assert settings.embedding.embedding_size > 0


def test_llm_and_embedding_are_api_mode():
    """本项目约定全部走外部 API：base_url 与 model 必须在 .env 里配置"""
    assert settings.llm.base_url.startswith("http"), "LLM_BASE_URL 未配置"
    assert settings.llm.model, "LLM_MODEL 未配置"
    assert settings.embedding.base_url.startswith("http"), "EMBEDDING_BASE_URL 未配置"
    assert settings.embedding.model, "EMBEDDING_MODEL 未配置"
    assert settings.rerank.base_url.startswith("http"), "RERANK_BASE_URL 未配置"
    assert settings.rerank.model, "RERANK_MODEL 未配置"


def test_milvus_collection_configured():
    assert settings.milvus.db_name, "MILVUS_DB_NAME 未配置"
    assert settings.milvus.collection, "MILVUS_COLLECTION 未配置"


class TestEvalLlmTarget:
    """评判模型的目标解析（课案：评估模型与生成模型分开配置）。

    `eval_llm_target()` 是 Ragas 与 `evaluation/llm_judge.py` **共用的**回退规则入口 ——
    写在两处必然漂移，所以这里把三种情形都钉住。
    """

    def test_configured_uses_eval_model(self, monkeypatch):
        from config import eval_llm_target

        monkeypatch.setattr(settings.eval_llm, "model", "judge-x")
        monkeypatch.setattr(settings.eval_llm, "api_key", "ev-key")
        monkeypatch.setattr(settings.eval_llm, "base_url", "https://ev.example/v1")
        target = eval_llm_target()
        assert target["model"] == "judge-x"
        assert target["api_key"] == "ev-key"
        assert target["base_url"] == "https://ev.example/v1"
        assert target["from_fallback"] is False

    def test_credentials_fall_back_per_field(self, monkeypatch):
        from config import eval_llm_target

        monkeypatch.setattr(settings.eval_llm, "model", "judge-x")
        monkeypatch.setattr(settings.eval_llm, "api_key", "")
        monkeypatch.setattr(settings.eval_llm, "base_url", "")
        monkeypatch.setattr(settings.llm, "api_key", "gen-key")
        monkeypatch.setattr(settings.llm, "base_url", "https://gen.example/v1")
        target = eval_llm_target()
        assert target["model"] == "judge-x"  # 模型不跟着回退
        assert target["api_key"] == "gen-key"
        assert target["base_url"] == "https://gen.example/v1"

    def test_unconfigured_falls_back_to_generation_model(self, monkeypatch):
        from config import eval_llm_target

        monkeypatch.setattr(settings.eval_llm, "model", "")
        monkeypatch.setattr(settings.llm, "model", "gen-y")
        monkeypatch.setattr(settings.llm, "max_tokens", 4096)
        target = eval_llm_target()
        assert target["model"] == "gen-y"
        assert target["from_fallback"] is True, "回退必须是显式标记，调用方才能打 WARNING"

    def test_env_example_documents_the_keys(self):
        """`.env.example` 必须列出这几个键（换机器时靠它知道要配什么）。"""
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        text = (root / ".env.example").read_text(encoding="utf-8")
        for key in ("EVAL_LLM_MODEL", "EVAL_LLM_API_KEY", "EVAL_LLM_BASE_URL"):
            assert key in text, f".env.example 缺少 {key}"

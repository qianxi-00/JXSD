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

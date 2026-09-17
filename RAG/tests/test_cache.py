"""缓存层测试:精确缓存键的作用域与归一化、降级行为、预设相似度缓存阈值。

课案要点(基础篇「缓存与FAQ」):
- Redis QA 缓存是问题级短缓存,键里**不**放会话 history;
- 键需要包含模型名、集合名与两个阈值,配置变化后旧答案不能复用;
- 查询先做空白归一化,同一问题不同空白应命中同一条缓存;
- Redis 异常时降级返回 None(继续走 RAG),不抛出。
"""

import numpy as np
import pytest

from config import settings
from core import cache as cache_module
from core.cache import AnswerCache


class FakeRedis:
    def __init__(self):
        self.data: dict[str, str] = {}
        self.set_calls: list[tuple] = []
        self.ttls: list[int] = []
        self.fail = False

    def get(self, key):
        if self.fail:
            raise RuntimeError("redis down")
        return self.data.get(key)

    def set(self, key, value, ex=None):
        if self.fail:
            raise RuntimeError("redis down")
        self.set_calls.append((key, value, ex))
        self.ttls.append(ex)
        self.data[key] = value

    def exists(self, *keys):
        return sum(1 for k in keys if k in self.data)

    def pipeline(self):
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, redis):
        self.redis = redis
        self.ops: list[tuple] = []

    def set(self, key, value):
        self.ops.append((key, value))
        return self

    def execute(self):
        for key, value in self.ops:
            self.redis.data[key] = value
        return [True] * len(self.ops)


@pytest.fixture
def fake_redis(monkeypatch):
    redis = FakeRedis()
    monkeypatch.setattr(cache_module, "get_redis_client", lambda: redis)
    return redis


@pytest.fixture
def cache(fake_redis):
    return AnswerCache()


class TestNormalizeAndScope:
    def test_whitespace_normalized(self, cache):
        a = cache._exact_key("  张三   的 火车票 ")
        b = cache._exact_key("张三 的 火车票")
        assert a == b

    def test_different_query_different_key(self, cache):
        assert cache._exact_key("张三的火车票") != cache._exact_key("李四的火车票")

    def test_key_scoped_by_llm_model(self, cache, monkeypatch):
        before = cache._exact_key("张三的火车票")
        monkeypatch.setattr(settings.llm, "model", "another-model")
        assert cache._exact_key("张三的火车票") != before

    def test_key_scoped_by_collection(self, cache, monkeypatch):
        before = cache._exact_key("张三的火车票")
        monkeypatch.setattr(settings.milvus, "collection", "another_collection")
        assert cache._exact_key("张三的火车票") != before

    def test_key_scoped_by_thresholds(self, cache, monkeypatch):
        before = cache._exact_key("张三的火车票")
        monkeypatch.setattr(settings.rerank, "relevance_p", 0.5)
        assert cache._exact_key("张三的火车票") != before

        monkeypatch.setattr(settings.redis, "sim_threshold", 0.7)
        assert cache._exact_key("张三的火车票") != before


class TestStoreAndLookup:
    def test_store_then_lookup_hits_exact(self, cache):
        cache.store("张三的火车票", "答案是 436 元", [{"id": "ticket_a"}])
        hit = cache.lookup("张三的火车票")
        assert hit["cache_hit"] == "exact"
        assert hit["answer"] == "答案是 436 元"
        assert hit["sources"] == [{"id": "ticket_a"}]

    def test_whitespace_variant_hits_same_entry(self, cache):
        cache.store("张三的火车票", "答案", [])
        assert cache.lookup("   张三的火车票   ")["answer"] == "答案"

    def test_internal_whitespace_collapsed(self, cache):
        cache.store("张三 的 火车票", "答案", [])
        assert cache.lookup("张三    的   火车票")["answer"] == "答案"

    def test_store_sets_configured_ttl(self, cache, fake_redis):
        cache.store("张三的火车票", "答案", [])
        assert fake_redis.ttls == [settings.redis.exact_ttl]

    def test_store_skips_empty_question_or_answer(self, cache, fake_redis):
        cache.store("   ", "答案", [])
        cache.store("张三的火车票", "", [])
        assert fake_redis.set_calls == []

    def test_lookup_miss_returns_none(self, cache, monkeypatch):
        # 预设缓存也关掉,确认纯未命中路径
        monkeypatch.setattr(cache, "_lookup_preset", lambda q: None)
        assert cache.lookup("没有缓存的问题") is None

    def test_lookup_degrades_on_redis_error(self, cache, fake_redis):
        fake_redis.fail = True
        assert cache.lookup("张三的火车票") is None

    def test_store_degrades_on_redis_error(self, cache, fake_redis):
        fake_redis.fail = True
        cache.store("张三的火车票", "答案", [])  # 不应抛出

    def test_exact_hit_short_circuits_preset(self, cache, monkeypatch):
        called = []
        monkeypatch.setattr(cache, "_lookup_preset", lambda q: called.append(q))
        cache.store("张三的火车票", "精确答案", [])
        hit = cache.lookup("张三的火车票")
        assert hit["cache_hit"] == "exact"
        assert called == []


class TestPresetLookup:
    def _ready_cache(self, cache, monkeypatch, vector, threshold_hit: float):
        """构造两条预设问答,并让查询向量与第一条的相似度为给定值"""
        cache._preset_questions = ["预设问题一", "预设问题二"]
        cache._preset_answers = ["预设答案一", "预设答案二"]
        matrix = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        cache._preset_matrix = matrix
        monkeypatch.setattr("retrieval.embedding.embed_query", lambda q: vector)
        return cache

    def test_hit_above_threshold(self, cache, monkeypatch):
        vec = np.array([1.0, 0.0], dtype=np.float32)
        self._ready_cache(cache, monkeypatch, vec, 1.0)
        monkeypatch.setattr(settings.redis, "sim_threshold", 0.85)
        hit = cache._lookup_preset("任意问题")
        assert hit["cache_hit"] == "preset"
        assert hit["answer"] == "预设答案一"
        assert hit["similarity"] == 1.0

    def test_miss_below_threshold(self, cache, monkeypatch):
        # 与两条预设的相似度都是 0.6 左右,低于阈值即未命中
        vec = np.array([0.6, 0.6], dtype=np.float32)
        self._ready_cache(cache, monkeypatch, vec, 0.6)
        monkeypatch.setattr(settings.redis, "sim_threshold", 0.85)
        assert cache._lookup_preset("任意问题") is None

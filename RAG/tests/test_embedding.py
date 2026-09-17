"""embedding 请求构造测试:不同厂商对 `dimensions` 参数的支持不一样。

实测(2026-09-17,SiliconFlow):
- `Qwen/Qwen3-Embedding-4B` 接受 `dimensions`(可以做 MRL 截断);
- 同门的 `BAAI/bge-m3` 原生就是 1024 维,传 `dimensions` 直接
  `400 20015 The parameter is invalid`。

所以 `dimensions` 必须"显式开启才下发";同时返回值维度必须与 `EMBEDDING_SIZE` 一致 ——
Milvus 集合是按这个维度建的,维度不一致会把坏向量静默写进去。
"""

import pytest

from retrieval import embedding as embedding_mod


class FakeEmbeddings:
    """记录调用参数的假 embeddings 客户端。"""

    def __init__(self, dim: int = 1024, fail_times: int = 0):
        self.dim = dim
        self.fail_times = fail_times
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("模拟网络错误")

        class _Item:
            def __init__(self, index, vector):
                self.index = index
                self.embedding = vector

        class _Resp:
            def __init__(self, items):
                self.data = items

        texts = kwargs["input"]
        return _Resp([_Item(i, [0.1] * self.dim) for i, _ in enumerate(texts)])


class FakeClient:
    def __init__(self, embeddings: FakeEmbeddings):
        self.embeddings = embeddings


@pytest.fixture
def fake_api(monkeypatch):
    """替换 OpenAI 客户端与重试等待,测试不碰网络也不睡觉。"""
    monkeypatch.setattr(embedding_mod.time, "sleep", lambda _s: None)

    def install(dim: int = 1024, fail_times: int = 0) -> FakeEmbeddings:
        fake = FakeEmbeddings(dim=dim, fail_times=fail_times)
        monkeypatch.setattr(embedding_mod, "get_embedding_client", lambda: FakeClient(fake))
        return fake

    return install


def test_empty_input_returns_empty_without_calling_api(fake_api):
    fake = fake_api()
    assert embedding_mod.embed_texts([]) == []
    assert fake.calls == []


def test_dimensions_is_not_sent_by_default(fake_api):
    """默认不下发 dimensions —— bge-m3 这类模型会因为这个参数直接 400。"""
    fake = fake_api()
    embedding_mod.embed_texts(["票据"])
    assert "dimensions" not in fake.calls[0], fake.calls[0]


def test_dimensions_is_sent_when_explicitly_enabled(fake_api, monkeypatch):
    """需要 MRL 截断的模型(Qwen3-Embedding-v4 等)可以显式开启。"""
    fake = fake_api()
    monkeypatch.setattr(embedding_mod.settings.embedding, "send_dimensions", True)
    embedding_mod.embed_texts(["票据"])
    assert fake.calls[0]["dimensions"] == embedding_mod.settings.embedding.embedding_size


def test_dimension_mismatch_raises_with_actionable_message(fake_api):
    """模型返回的维度与配置不一致时必须报错,不能把坏向量写进 Milvus。"""
    fake_api(dim=768)
    with pytest.raises(RuntimeError) as excinfo:
        embedding_mod.embed_texts(["票据"])
    message = str(excinfo.value)
    assert "768" in message and str(embedding_mod.settings.embedding.embedding_size) in message


def test_dimension_mismatch_is_not_retried(fake_api):
    """维度不符是配置问题,重试没有意义 —— 只应该请求一次。"""
    fake = fake_api(dim=768)
    with pytest.raises(RuntimeError):
        embedding_mod.embed_texts(["票据"])
    assert len(fake.calls) == 1


def test_transient_error_is_retried_then_raises(fake_api):
    fake = fake_api(fail_times=3)
    with pytest.raises(RuntimeError) as excinfo:
        embedding_mod.embed_texts(["票据"])
    assert "模拟网络错误" in str(excinfo.value)
    assert len(fake.calls) == 3


def test_transient_error_recovers(fake_api):
    fake = fake_api(fail_times=1)
    vectors = embedding_mod.embed_texts(["票据"])
    assert len(vectors) == 1 and len(vectors[0]) == 1024
    assert len(fake.calls) == 2


def test_batch_preserves_input_order(fake_api, monkeypatch):
    monkeypatch.setattr(embedding_mod, "get_embedding_client", lambda: FakeClient(FakeEmbeddings()))
    vectors = embedding_mod.embed_texts(["甲", "乙", "丙"])
    assert len(vectors) == 3
    assert embedding_mod.embed_query("甲") == vectors[0]

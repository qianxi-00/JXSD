"""LLM 客户端构造测试：**必须显式带超时**。

为什么要专门钉住这件事：OpenAI SDK 的默认超时是 600 秒、默认还会重试 3 次，
一旦网关断连（本项目用的私有网关在负载下会断），单次调用就能挂十几分钟。
实测踩过：建图任务卡了 46 分钟没有任何输出，Web 端表现为页面一直转圈。
所以超时不是"可选的调优项"，而是必须显式设置的失败边界。
"""

from config import settings
from graph_rag import builder
from llm import chat

CAPTURED: list[dict] = []


class FakeOpenAI:
    def __init__(self, **kwargs):
        CAPTURED.append(kwargs)
        self.chat = self
        self.completions = self

    def create(self, **kwargs):  # pragma: no cover - 只用来满足属性访问
        raise AssertionError("测试不应真的发请求")


class FakeAsyncOpenAI(FakeOpenAI):
    pass


def test_sync_client_carries_timeout(monkeypatch):
    CAPTURED.clear()
    monkeypatch.setattr(chat, "OpenAI", FakeOpenAI)
    chat._client()
    assert CAPTURED[0]["timeout"] == settings.llm.timeout
    assert CAPTURED[0]["base_url"] == settings.llm.base_url


def test_async_client_carries_timeout(monkeypatch):
    CAPTURED.clear()
    monkeypatch.setattr(chat, "AsyncOpenAI", FakeAsyncOpenAI)
    chat._async_client()
    assert CAPTURED[0]["timeout"] == settings.llm.timeout


def test_graph_builder_client_carries_timeout(monkeypatch):
    CAPTURED.clear()
    monkeypatch.setattr(builder, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(builder, "_llm_client", None)
    builder.get_llm_client()
    assert CAPTURED[0]["timeout"] == settings.llm.timeout


def test_timeout_is_configured_and_bounded():
    """超时必须是**配置项**且有合理上界：太短会误杀正常的长回答，太长等于没有。"""
    assert 30 <= settings.llm.timeout <= 600

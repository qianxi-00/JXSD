"""建图 LLM 出口测试:`call_llm` 的重试与失败语义。

背景(实测踩到):建图要跑几分钟、几十次 LLM 调用,私有网关偶尔
`RemoteProtocolError: Server disconnected without sending a response` ——
没有重试的话,一次抖动就白跑整轮(实测两次重建都在抽取/摘要阶段挂掉)。
"""

import pytest

from graph_rag import builder


class FakeCompletions:
    def __init__(self, outcomes: list):
        self.outcomes = outcomes
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes[min(self.calls - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeClient:
    def __init__(self, outcomes):
        self.completions = FakeCompletions(outcomes)
        self.chat = self


@pytest.fixture
def fake_llm(monkeypatch):
    monkeypatch.setattr(builder.time, "sleep", lambda _s: None)

    def install(outcomes):
        client = FakeClient(outcomes)
        monkeypatch.setattr(builder, "get_llm_client", lambda: client)
        return client

    return install


def test_returns_content_on_first_success(fake_llm):
    # 原样返回(判空时才 strip),JSON 解析交给 load_json_object
    client = fake_llm([FakeResponse("  实体：张三  ")])
    assert builder.call_llm("sys", "user") == "  实体：张三  "
    assert client.completions.calls == 1


def test_retries_transient_connection_error(fake_llm):
    """网关偶发断连必须重试 —— 否则整轮建图白跑。"""
    client = fake_llm([ConnectionError("Server disconnected"), FakeResponse("恢复后的内容")])
    assert builder.call_llm("sys", "user") == "恢复后的内容"
    assert client.completions.calls == 2


def test_gives_up_after_max_attempts(fake_llm):
    client = fake_llm([ConnectionError("Server disconnected")])
    with pytest.raises(RuntimeError) as excinfo:
        builder.call_llm("sys", "user")
    assert "Server disconnected" in str(excinfo.value)
    assert client.completions.calls == builder.LLM_MAX_ATTEMPTS


def test_empty_content_is_retried_then_raises(fake_llm):
    """空内容同样是"这次调用废了",要重试而不是把空串当成结果。"""
    client = fake_llm([FakeResponse(""), FakeResponse("有效内容")])
    assert builder.call_llm("sys", "user") == "有效内容"
    assert client.completions.calls == 2


def test_json_mode_passes_response_format(fake_llm, monkeypatch):
    captured = {}
    client = fake_llm([FakeResponse('{"ok": 1}')])
    original = client.completions.create

    def spy(**kwargs):
        captured.update(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(client.completions, "create", spy)
    builder.call_llm("sys", "user", json_mode=True, temperature=0.3)
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["temperature"] == 0.3
    assert captured["model"] == builder.settings.llm.model

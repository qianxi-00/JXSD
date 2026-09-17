"""生产链路追踪接线测试（优化篇课案「系统监控与部署 · Langfuse 运行质量」）。

课案要求：一次问答对应一条 Trace，模型/检索/证据分析形成 Span，并写入
Session / User 标识以便按会话与用户聚合；同一个 agent 只需在调用时附加 callback。

本项目原来的缺口：`invoke_slow_agent` 裸调 `agent.invoke(...)`，全仓 `CallbackHandler` 零命中 ——
Langfuse 只用在离线实验评估，生产链路一条 trace 都没有。

这里钉住三件事：没配凭证不阻断主链路、配了要真的挂上、Session/User 标识要透传。
"""

from langchain_core.messages import AIMessage

from agentic import finance_agent


class FakeAgent:
    """记录 invoke 的入参与 config。"""

    def __init__(self):
        self.calls: list[dict] = []

    def invoke(self, payload, config=None, **kwargs):
        self.calls.append({"payload": payload, "config": config, "kwargs": kwargs})
        return {"messages": [*payload["messages"], AIMessage(content="答案")]}


def fake_handler():
    return object()


def install(monkeypatch, handler=None) -> FakeAgent:
    agent = FakeAgent()
    monkeypatch.setattr(finance_agent, "agent", agent)
    monkeypatch.setattr(finance_agent, "_langfuse_handler", None)
    monkeypatch.setattr(finance_agent, "langfuse_callbacks", lambda: [handler or fake_handler()])
    return agent


def test_no_callbacks_when_langfuse_not_configured(monkeypatch):
    """没配密钥时不能报错、也不能挂空回调 —— 本地/离线跑必须照常工作。"""
    monkeypatch.setattr(finance_agent.settings, "langfuse_public_key", "")
    monkeypatch.setattr(finance_agent.settings, "langfuse_secret_key", "")
    monkeypatch.setattr(finance_agent, "_langfuse_handler", "已初始化")  # 即使有残留也不能用
    assert finance_agent.langfuse_callbacks() == []


def test_callbacks_attached_to_invoke_config(monkeypatch):
    agent = install(monkeypatch)
    finance_agent.invoke_slow_agent("于强的机票是从哪到哪的?", [])
    config = agent.calls[0]["config"]
    assert config and len(config["callbacks"]) == 1


def test_session_and_user_metadata_passed(monkeypatch):
    agent = install(monkeypatch)
    finance_agent.invoke_slow_agent("问题", [], session_id="finance-session-42", user_id="user-17")
    metadata = agent.calls[0]["config"]["metadata"]
    assert metadata["langfuse_session_id"] == "finance-session-42"
    assert metadata["langfuse_user_id"] == "user-17"


def test_metadata_omitted_when_no_session_or_user(monkeypatch):
    agent = install(monkeypatch)
    finance_agent.invoke_slow_agent("问题", [])
    assert "metadata" not in agent.calls[0]["config"]


def test_thread_id_reaches_the_agent_config(monkeypatch):
    """`thread_id` 必须真的进到 `agent.invoke(config=...)` 里，而不是只停在 tracing_config。

    这是"跨请求恢复"那条路的接口：调用方给了 thread_id，langgraph 才找得到对应的检查点
    （前提是 Agent 带了 checkpointer，见 build_production_agent 的 WARNING）。
    没有这条用例的话，`tracing_config(thread_id=...)` 就是个没人跑的死参数。
    """
    agent = install(monkeypatch)
    finance_agent.invoke_slow_agent("问题", [], thread_id="conv-42")
    assert agent.calls[0]["config"]["configurable"] == {"thread_id": "conv-42"}
    # 与 Langfuse 的 session 是**两个字段**：混用一个会导致"改了追踪标识就悄悄改了线程"
    assert "langfuse_session_id" not in (agent.calls[0]["config"].get("metadata") or {})


def test_thread_id_works_without_langfuse(monkeypatch):
    """追踪关掉时也要能把 thread_id 传下去（两件事本来无关）。"""
    agent = FakeAgent()
    monkeypatch.setattr(finance_agent, "agent", agent)
    monkeypatch.setattr(finance_agent, "langfuse_callbacks", list)
    finance_agent.invoke_slow_agent("问题", [], thread_id="conv-7")
    assert agent.calls[0]["config"] == {"configurable": {"thread_id": "conv-7"}}


def test_no_thread_id_keeps_config_without_configurable(monkeypatch):
    """不给 thread_id 时**不能**凭空造一个 —— langgraph 会去找一个并不存在的线程。"""
    agent = install(monkeypatch)
    finance_agent.invoke_slow_agent("问题", [])
    assert "configurable" not in agent.calls[0]["config"]


def test_handler_creation_failure_is_fail_open(monkeypatch):
    """Langfuse 挂不上不能拖垮问答 —— 只警告、继续跑。"""
    monkeypatch.setattr(finance_agent.settings, "langfuse_public_key", "pk-x")
    monkeypatch.setattr(finance_agent.settings, "langfuse_secret_key", "sk-x")
    monkeypatch.setattr(finance_agent, "_langfuse_handler", None)

    def boom():
        raise RuntimeError("langfuse 连不上")

    monkeypatch.setattr(finance_agent, "_build_langfuse_handler", boom)
    assert finance_agent.langfuse_callbacks() == []


def test_payload_still_carries_history(monkeypatch):
    agent = install(monkeypatch)
    history = [{"role": "user", "content": "上一轮"}, {"role": "assistant", "content": "上一答"}]
    finance_agent.invoke_slow_agent("新问题", history)
    messages = agent.calls[0]["payload"]["messages"]
    assert [m["content"] for m in messages] == ["上一轮", "上一答", "新问题"]

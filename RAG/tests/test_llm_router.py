"""路由调用测试：**路由必须真的拿到模型输出**，不能静默退化成默认值。

实测背景（2026-09-17 换到 DeepSeek `deepseek-flash`）：它是思考模型，
`max_tokens=64` 时思考会把预算吃光、正文恒为空 —— 而 `route_query` 里
解析不到 JSON 就"兜底成走 RAG + 直接检索"，**看起来一切正常，实际路由全废**。
日志里只会留下 `[路由] LLM 原始输出: `（空的）。

第二轮实测（`RAG/script/probe_llm_thinking.py`（当时是两支一次性探针，已合并落成该脚本））发现光给预算并不够，
而且**第一版修法用错了字段**：`{"enable_thinking": False}` 被这个端点静默忽略，
域外问题思考 465~1111 字，256 的预算照样吃光（3/3 全空）；
真正生效的是 `{"thinking": {"type": "disabled"}}`（reasoning=0、只花 14 token）。

所以这里钉三件事：路由必须显式关思考（用对的字段）、预算足够出正文、
空返回时的告警要带上能定位原因的线索。
"""

from types import SimpleNamespace

from llm import chat


class Recorder:
    def __init__(self, content: str, reasoning: str = ""):
        self.kwargs: dict = {}
        self._content = content
        self._reasoning = reasoning

    def create(self, **kwargs):
        self.kwargs = kwargs
        message = SimpleNamespace(content=self._content, reasoning_content=self._reasoning)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")]
        )


def install(monkeypatch, content: str, reasoning: str = "") -> Recorder:
    recorder = Recorder(content, reasoning)

    class FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=recorder)

    monkeypatch.setattr(chat, "OpenAI", FakeClient)
    return recorder


def test_router_disables_thinking_without_mutating_global_setting(monkeypatch):
    recorder = install(monkeypatch, '{"route": "DIRECT", "rewrite": "direct"}')
    need_rag, rewrite = chat.route_query("1+1等于几?")
    assert (need_rag, rewrite) == (False, "direct")
    # 必须带上这个端点**真正认**的关思考开关：只发 enable_thinking 时它被静默忽略，
    # 域外问题的思考会吃光 256 的预算、正文为空（实测 3/3），路由静默退化。
    assert recorder.kwargs["extra_body"]["thinking"] == {"type": "disabled"}
    # 不能顺手把全局配置改掉：这只影响这一次调用
    assert chat.settings.llm.enable_thinking in (True, False)


def test_router_budget_is_enough_for_a_thinking_model(monkeypatch):
    recorder = install(monkeypatch, '{"route": "RAG", "rewrite": "hyde"}')
    chat.route_query("万宁的火车票票号是多少?")
    assert recorder.kwargs["max_tokens"] >= 256, (
        "预算太小的话思考模型会把 token 全花在思考上,正文为空 -> 路由静默退化"
    )


def test_router_keeps_qwen_style_switch_for_gateway_compat(monkeypatch):
    """`enable_thinking` 对当前端点无效，但要保留给 Qwen 系网关（两边都不报 400）。"""
    recorder = install(monkeypatch, '{"route": "RAG", "rewrite": "direct"}')
    chat.route_query("万宁的火车票票号是多少?")
    assert recorder.kwargs["extra_body"]["enable_thinking"] is False


def test_empty_content_warning_carries_diagnosable_context(monkeypatch):
    """空正文的告警必须带 finish_reason 与思考长度 —— 否则下一个看日志的人
    分不清"预算被思考吃光"和"模型真的什么都没说"。

    这里直接盯 `logger.warning` 调用（而不是 caplog）：项目的 logger 由
    `core/logger.py` 自己配 handler、不向 root 传播，caplog 抓不到。
    """
    from types import SimpleNamespace as NS

    warnings: list[str] = []
    monkeypatch.setattr(chat.logger, "warning", lambda msg, *a, **kw: warnings.append(str(msg)))

    recorder = install(monkeypatch, "", reasoning="x" * 465)

    # 让 fake 响应带上 finish_reason=length，模拟"预算被思考吃光"的真实现象
    def create(**kwargs):
        recorder.kwargs = kwargs
        message = NS(content="", reasoning_content="x" * 465)
        return NS(choices=[NS(message=message, finish_reason="length")])

    monkeypatch.setattr(recorder, "create", create)

    assert chat.route_query("今天杭州天气怎么样？") == (True, "direct")
    assert len(warnings) == 1
    assert "finish_reason=length" in warnings[0]
    assert "465" in warnings[0]
    assert str(chat.ROUTER_MAX_TOKENS) in warnings[0]


def test_router_parses_rewrite_strategy(monkeypatch):
    install(monkeypatch, '{"route": "RAG", "rewrite": "subquery"}')
    assert chat.route_query("对比一下两个人的票") == (True, "subquery")


def test_empty_content_still_falls_back_gracefully(monkeypatch):
    """真返回空内容时保持原兜底行为（不能抛异常拖垮整条链路）。"""
    install(monkeypatch, "")
    assert chat.route_query("随便问问") == (True, "direct")


def test_thinking_flag_follows_config_for_generation(monkeypatch):
    """生成路径仍然跟随配置开关，没被路由的特例改坏。"""
    recorder = install(monkeypatch, "答案在这里")
    monkeypatch.setattr(chat.settings.llm, "enable_thinking", True)
    chat.generate_answer("问题", "上下文")
    assert recorder.kwargs["extra_body"] == {"enable_thinking": True}

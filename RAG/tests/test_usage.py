"""token 用量采集与成本估算的测试（T5：优化篇「质量与成本一起看」）。

覆盖三件事：
    1. `TokenUsageCallback._extract` 对**三种真实形状**的提取（实测探针拿到的两种 + 取不到）；
    2. 多次调用累加、以及"取不到用量但确实调用过"时 `llm_calls` 仍然 +1；
    3. `estimate_cost` 的算术（含缓存命中价）与"没配单价 ⇒ None"。

形状依据：`.dsh_tmp/probe_usage_shape.py` 对 deepseek-chat 的真实调用输出。
"""

from types import SimpleNamespace

from agentic.usage import TokenUsageCallback, estimate_cost
from config import settings


def llm_result(usage_metadata=None, token_usage=None, response_metadata_usage=None):
    """构造一个形状接近 LangChain LLMResult 的假对象。"""
    message = SimpleNamespace(
        usage_metadata=usage_metadata,
        response_metadata={"token_usage": response_metadata_usage} if response_metadata_usage else {},
    )
    return SimpleNamespace(
        generations=[[SimpleNamespace(message=message)]],
        llm_output={"token_usage": token_usage} if token_usage else {},
    )


class TestExtract:
    def test_new_style_usage_metadata(self):
        """实测形状：{'input_tokens': 11, 'output_tokens': 40, 'input_token_details': {'cache_read': 0}}"""
        cb = TokenUsageCallback()
        cb.on_llm_end(llm_result(usage_metadata={
            "input_tokens": 11, "output_tokens": 40, "total_tokens": 51,
            "input_token_details": {"cache_read": 0},
        }))
        assert (cb.input_tokens, cb.output_tokens, cb.cached_input_tokens) == (11, 40, 0)
        assert cb.calls == 1

    def test_new_style_with_cache_read(self):
        cb = TokenUsageCallback()
        cb.on_llm_end(llm_result(usage_metadata={
            "input_tokens": 100, "output_tokens": 5, "input_token_details": {"cache_read": 64},
        }))
        assert cb.cached_input_tokens == 64
        assert cb.as_dict()["total_tokens"] == 105

    def test_old_style_llm_output(self):
        """老式形状：llm_output['token_usage']（部分 provider 只给这个）。"""
        cb = TokenUsageCallback()
        cb.on_llm_end(llm_result(token_usage={
            "prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10,
            "prompt_cache_hit_tokens": 2,
        }))
        assert (cb.input_tokens, cb.output_tokens, cb.cached_input_tokens) == (7, 3, 2)

    def test_response_metadata_fallback(self):
        """第三级回退：只有 message.response_metadata 里有 token_usage。"""
        cb = TokenUsageCallback()
        cb.on_llm_end(llm_result(response_metadata_usage={"prompt_tokens": 4, "completion_tokens": 1}))
        assert (cb.input_tokens, cb.output_tokens) == (4, 1)

    def test_missing_usage_still_counts_the_call(self):
        """取不到用量时**调用次数照常 +1** —— 否则"这家不上报用量"会长得像"没调模型"。"""
        cb = TokenUsageCallback()
        cb.on_llm_end(llm_result())
        cb.on_llm_end(llm_result())
        assert cb.calls == 2
        assert cb.as_dict()["total_tokens"] == 0

    def test_accumulates_across_calls(self):
        cb = TokenUsageCallback()
        cb.on_llm_end(llm_result(usage_metadata={"input_tokens": 10, "output_tokens": 5}))
        cb.on_llm_end(llm_result(usage_metadata={"input_tokens": 20, "output_tokens": 7}))
        d = cb.as_dict()
        assert (d["llm_calls"], d["input_tokens"], d["output_tokens"], d["total_tokens"]) == (2, 30, 12, 42)


class TestEstimateCost:
    def _prices(self, monkeypatch, in_price=0.0, out_price=0.0, cached=0.0):
        monkeypatch.setattr(settings.usage, "price_input_per_million", in_price)
        monkeypatch.setattr(settings.usage, "price_output_per_million", out_price)
        monkeypatch.setattr(settings.usage, "price_cached_input_per_million", cached)

    def test_returns_none_when_prices_unset(self, monkeypatch):
        self._prices(monkeypatch)
        assert estimate_cost({"input_tokens": 1000, "output_tokens": 1000}) is None

    def test_basic_math(self, monkeypatch):
        """输入 2 元/百万、输出 8 元/百万：100 万输入 + 10 万输出 = 2 + 0.8 = 2.8"""
        self._prices(monkeypatch, in_price=2.0, out_price=8.0)
        cost = estimate_cost({"input_tokens": 1_000_000, "output_tokens": 100_000})
        assert cost == 2.8

    def test_cached_input_uses_its_own_price(self, monkeypatch):
        """命中缓存的输入按 caches 单价算：80 万缓存(0.5) + 20 万普通(2.0) + 0 输出"""
        self._prices(monkeypatch, in_price=2.0, out_price=8.0, cached=0.5)
        cost = estimate_cost({
            "input_tokens": 1_000_000, "output_tokens": 0, "cached_input_tokens": 800_000,
        })
        assert cost == 0.2 * 2.0 + 0.8 * 0.5  # 0.4 + 0.4 = 0.8

    def test_cached_never_exceeds_input(self, monkeypatch):
        """缓存命中数不可能超过输入总量（上游偶尔给不一致的数），多出来的按普通价算。"""
        self._prices(monkeypatch, in_price=2.0, out_price=8.0, cached=0.5)
        cost = estimate_cost({"input_tokens": 100, "output_tokens": 0, "cached_input_tokens": 999})
        assert cost == 100 * 2.0 / 1_000_000

    def test_zero_prices_fall_back_to_input_price_for_cache(self, monkeypatch):
        """没单独配缓存价时按普通输入价算（保守偏高，不会低估成本）。"""
        self._prices(monkeypatch, in_price=1.0, out_price=1.0, cached=0.0)
        cost = estimate_cost({"input_tokens": 1_000_000, "output_tokens": 0, "cached_input_tokens": 500_000})
        assert cost == 1.0

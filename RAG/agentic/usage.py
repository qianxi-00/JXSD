"""采集一轮问答里的 **token 用量与成本估算**（优化篇课案「优化闭环 · 质量与成本一起看」）。

为什么需要它：评估脚本此前只统计**延迟**与**检索次数**，没有 token —— 于是回答不了
"把 top_k 调大 / 多跑一轮重检之后，成本涨了多少"。而课案明确要求质量与成本**一起看**。

采集方式：**LangChain 回调**（`on_llm_end`）。为什么不用别的路子：

    · 回调拿到的 `usage_metadata` 是**逐次 LLM 调用**的权威数据，而且 LangChain 会自动把
      回调传播给子代理 —— 主 Agent + evidence-analyst 子代理 + 路由/改写等所有调用都在内，
      这正是"一轮问答的真实成本"；
    · 去 Langfuse 反查 trace 需要额外 API 往返，且要在评估脚本里做聚合，容易与本次运行错位；
    · 直接数 prompt 字符只能估算，和账单对不上。

⚠️ 实测形状（探针 `.dsh_tmp/probe_usage_shape.py`，deepseek-chat 真实调用）：
    generation.message.usage_metadata = {'input_tokens': 11, 'output_tokens': 40,
                                         'total_tokens': 51, 'input_token_details': {'cache_read': 0}}
    response.llm_output['token_usage'] = {'completion_tokens': 40, 'prompt_tokens': 11,
                                          'total_tokens': 51, 'prompt_cache_hit_tokens': 0, ...}
两种都存在，但**不是所有 provider/版本都给**，所以抽取按"新式 → 老式 → response_metadata"
三级回退，取不到就记 0 并把 `calls` 照常 +1（"3 次调用但 0 token"本身就是"这家没上报用量"的信号，
比静默记 0 更有用）。

成本：**必须由使用者填单价**（`USAGE_PRICE_*`，见 `.env.example`）。刻意不内置价目表 ——
价格随厂商调整、且本项目走的是网关（模型名未必是官方名），内置一张表迟早变成错误信息。
单价没填时 `estimate_cost()` 返回 None，指标里就不出现 cost（宁可不报，也不报错数）。
"""

from __future__ import annotations

from langchain_core.callbacks import BaseCallbackHandler

from config import settings


class TokenUsageCallback(BaseCallbackHandler):
    """把一轮问答里所有 LLM 调用的 token 累加起来。

    用法：构造一个实例 → 挂进 invoke 的 `config={"callbacks": [...]}` →
    跑完读 `as_dict()`。实例是**每次问答一份**（不要跨问答复用，否则数字会累加）。
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_input_tokens = 0

    # ---------------------------------------------------------------- 抽取
    @staticmethod
    def _extract(response) -> tuple[int, int, int]:  # noqa: ANN001
        """从 LangChain 的 LLMResult 里抽出 (输入, 输出, 命中缓存的输入)。

        三级回退的理由见模块 docstring；三级都取不到就返回全 0（调用方据此判断"这家没上报"）。
        """
        generations = getattr(response, "generations", None) or []
        if generations and generations[0]:
            message = getattr(generations[0][0], "message", None)
            usage = getattr(message, "usage_metadata", None) or {}
            if usage:
                details = usage.get("input_token_details") or {}
                return (
                    int(usage.get("input_tokens") or 0),
                    int(usage.get("output_tokens") or 0),
                    int(details.get("cache_read") or 0),
                )
        # 老式：llm_output / response_metadata 里的 token_usage（OpenAI 风格键名）
        for holder in (getattr(response, "llm_output", None) or {},):
            usage = (holder or {}).get("token_usage") or {}
            if usage:
                return (
                    int(usage.get("prompt_tokens") or 0),
                    int(usage.get("completion_tokens") or 0),
                    int(usage.get("prompt_cache_hit_tokens") or 0),
                )
        if generations and generations[0]:
            message = getattr(generations[0][0], "message", None)
            usage = (getattr(message, "response_metadata", {}) or {}).get("token_usage") or {}
            if usage:
                return (
                    int(usage.get("prompt_tokens") or 0),
                    int(usage.get("completion_tokens") or 0),
                    int(usage.get("prompt_cache_hit_tokens") or 0),
                )
        return 0, 0, 0

    # ---------------------------------------------------------------- 回调
    def on_llm_end(self, response, **kwargs) -> None:  # noqa: ANN001, ARG002
        """每次 LLM 调用结束都进这里。

        `calls` **无条件 +1**：即使这次取不到 usage，也要留下"发生过一次调用"的痕迹，
        否则"这家 provider 不上报用量"会表现成"这一轮根本没调模型"。
        """
        self.calls += 1
        prompt, completion, cached = self._extract(response)
        self.input_tokens += prompt
        self.output_tokens += completion
        self.cached_input_tokens += cached

    # ---------------------------------------------------------------- 出口
    def as_dict(self) -> dict:
        return {
            "llm_calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            # 命中提示缓存的输入 token（DeepSeek 会给 prompt_cache_hit_tokens）：
            # 计入成本时这部分通常更便宜，所以单独留一列而不是并进 input_tokens。
            "cached_input_tokens": self.cached_input_tokens,
        }


def estimate_cost(usage: dict) -> float | None:
    """按配置的单价估算成本；单价没配（全 0）时返回 None。

    公式：`((未命中缓存的输入) × 输入单价 + (命中缓存输入) × 缓存单价 + 输出 × 输出单价) / 1e6`。
    缓存单价没单独配时按普通输入单价算（保守偏高，不会低估）。
    """
    price_in = settings.usage.price_input_per_million
    price_out = settings.usage.price_output_per_million
    price_cached = settings.usage.price_cached_input_per_million or price_in
    if not (price_in or price_out):
        return None
    input_tokens = int(usage.get("input_tokens") or 0)
    cached = int(usage.get("cached_input_tokens") or 0)
    # 缓存命中数按定义不可能超过输入总量。上游偶尔给出不自洽的数（缓存 > 输入），
    # 这时**整份按普通输入价**算 —— 保守偏高，不会把成本报低（报低会让人以为"很便宜"，
    # 那是比不报成本更糟的误导）。
    if cached < 0 or cached > input_tokens:
        cached = 0
    billable_input = input_tokens - cached
    output_tokens = int(usage.get("output_tokens") or 0)
    cost = (
        billable_input * price_in
        + cached * price_cached
        + output_tokens * price_out
    ) / 1_000_000
    return round(cost, 6)


def usage_summary(usage: dict) -> dict:
    """把 token 用量与成本打成一个字典（评估记录与批次指标共用的形状）。"""
    return {**usage, "cost": estimate_cost(usage), "currency": settings.usage.currency}

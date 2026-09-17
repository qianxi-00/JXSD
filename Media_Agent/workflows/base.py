# -*- coding: utf-8 -*-
"""工作流基类 —— 与 ``workflows/__init__.py`` 功能一致，提供别名兼容

课案出处：自媒体课案 → 项目架构 → 工作流基类

为什么两个文件都留着
    课案的 ``workflows/__init__.py`` 是 LLM 调用的真入口（``llm_call(prompt, temperature)``），
    而 ``base.py`` 里又写了一份「同功能 + 别名兼容」的实现，
    两边的 ``llm_call`` 签名其实**并不一致**（一个是 prompt 优先，一个是 model 优先）。

    这里不去复制两份逻辑：``base.py`` 只做**转发**，
    真实现始终在 ``__init__.py``，避免两处各改一半造成行为漂移。
    额外补一个 ``safe_llm_call(model, prompt)``，
    给「手里已经有模型实例、不想再取一次缓存」的调用方用。
"""

from workflows import clear_model_cache, get_model, llm_call

__all__ = ["get_model", "llm_call", "safe_llm_call", "clear_model_cache"]


def safe_llm_call(model, prompt: str, fallback: str = "") -> str:
    """用**已构造好的**模型实例调一次，失败返回提示文本而不抛异常。

    与 ``llm_call`` 的区别只有一个：模型由调用方自己传进来。
    适合在一个节点里连续多次调用、想省掉 ``get_model`` 查缓存开销的场景。

    Args:
        model: ``get_model()`` 返回的 LangChain 聊天模型实例。
        prompt: 提示词。
        fallback: 失败时的返回值；不传则返回 ``[LLM调用失败: ...]``。

    Returns:
        模型输出文本；失败时为提示文本（不会是异常）。
    """
    from langchain_core.messages import HumanMessage

    try:
        response = model.invoke([HumanMessage(content=prompt)])
        content = response.content
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)
    except Exception as exc:  # noqa: BLE001 —— 同上，节点不该因单次调用失败而中断
        return fallback if fallback else f"[LLM调用失败: {exc}]"

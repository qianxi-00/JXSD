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

import sys
from pathlib import Path

# ---- 路径引导：必须在 import 项目内模块（workflows 包）之前执行 ----
# 直接 `python workflows/base.py` 时 sys.path[0] 是 workflows/ 目录，
# 找不到 workflows 包本身，会报 ModuleNotFoundError: No module named 'workflows'。
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from workflows import clear_model_cache, get_model, llm_call  # noqa: E402

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


if __name__ == "__main__":
    print("=== 工作流基类自检（离线）===")

    import inspect

    import workflows
    from config import settings

    # 1) 转发关系：这三个名字必须与 workflows/__init__.py 里的是**同一个对象**，
    #    而不是各写一份 —— 本文件存在的意义就是避免「两处各改一半」。
    assert llm_call is workflows.llm_call
    assert get_model is workflows.get_model
    assert clear_model_cache is workflows.clear_model_cache
    print("  转发对象同一性（llm_call / get_model / clear_model_cache）  OK")

    # 2) safe_llm_call 是「model 优先」，与 llm_call 的「prompt 优先」不同；
    #    这个差异是课案兼容的既有约定，改了会让调用方静默错位。
    sig = inspect.signature(safe_llm_call)
    assert list(sig.parameters) == ["model", "prompt", "fallback"], sig
    assert sig.parameters["fallback"].default == ""
    print(f"  safe_llm_call 签名 {sig}  OK")

    # 3) 模型不可用时返回中文提示文本，绝不抛异常（工作流节点靠它不中断）
    class _BrokenModel:
        def invoke(self, messages):
            raise RuntimeError("模拟：密钥无效")

    out = safe_llm_call(_BrokenModel(), "任意提示")
    assert out.startswith("[LLM调用失败:"), out
    print(f"  模型不可用 → 提示文本而非异常  OK  {out}")

    # 4) 「没配密钥」那条降级在转出的 llm_call 里（本文件只转发、不复制那个判断）。
    #    临时把密钥取空，全程离线，不会发起真实请求。
    settings.media_llm_api_key = lambda: ""
    try:
        out = llm_call("任意提示")
    finally:
        del settings.media_llm_api_key
    assert out.startswith("[LLM未配置]"), out
    print("  缺 API_KEY → [LLM未配置]（走 llm_call）  OK")

    print("\n全部自检通过")

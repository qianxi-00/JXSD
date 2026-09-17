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

与课案的落地差异
    | 项 | 课案原文 | 本实现 | 原因 |
    |---|---|---|---|
    | ``llm_call`` 从哪来 | ``llm_call = safe_llm_call``（别名直接指向 ``safe_llm_call``） | 从 ``workflows`` 转发真入口 ``llm_call`` | 课案的别名与它自己正文里 **11 处** ``llm_call(prompt, temperature=...)`` 调用**签名互斥** —— ``safe_llm_call`` 的首形参是 ``model``，照课案走必 ``TypeError`` |
    | ``get_model`` | ``ChatOpenAI(api_key=settings.api_key, ...)`` 直连根扁平配置 | 转发的 ``get_model`` 里按 ``MEDIA_LLM_PROVIDER`` 解析密钥 / 地址 / 模型名 | 自媒体链路要能单独换服务商，而不动 RAG 与 Agent 课案共用的根 ``API_KEY`` / ``BASE_URL`` |
    | 超时参数名 | ``request_timeout=90`` | ``timeout=90`` | ``init_chat_model`` 走的是 ``ChatOpenAI`` 的 ``timeout`` 别名（实测：传进去落到 ``request_timeout``，值 90.0） |
    | ``list[dict]`` 响应 | 直接返回 ``response.content`` | 内容是 list 时按分块拼成字符串 | OpenAI 兼容服务会返回 ``[{"type": "text", "text": ...}]`` 这类分块，直接交给下游会被当成结构化数据 |
    | 路径引导 | 无（靠外层注入 ``PYTHONPATH``） | 顶部把项目根插进 ``sys.path`` | 直接 ``python workflows/base.py`` 时 ``sys.path[0]`` 是 ``workflows/``，会 ``ModuleNotFoundError: No module named 'workflows'``；自检里能过是因为 ``verify_all.py`` 给子进程注了 ``PYTHONPATH`` |
    | 离线自检 | 无 | 末尾 ``if __name__ == "__main__":`` 4 项断言 | 本文件全仓 0 引用、改坏不会有任何红线；已登记进 ``verify_all.py`` 的 ``MODULE_SELF_CHECKS`` |
    | 绝对路径 | 无 | 无 | —— |

踩过的坑
    · **``settings`` 是普通类实例，不是 pydantic 模型**（``config.py`` 的 ``class Settings``
      只继承 ``object``，靠 ``__getattr__`` 把未知属性转发给 ``_CoreSettings``）——
      所以自检里能直接 ``settings.media_llm_api_key = lambda: ""`` 打桩；换成 pydantic 模型
      这一步会抛 ``ValueError``。打完桩**必须 ``del`` 还原**：实例属性会盖住类方法，
      不删的话后面所有断言都会一直走「没配密钥」那条分支。
    · 验证「转发」要用 ``is`` 比对象同一性，别只比签名 —— 签名相同、实现各写一份的
      「假转发」只有 ``is`` 能抓出来，而它正是本文件存在的意义。
    · 两个 ``__all__`` 并不相同：``workflows`` 是 ``get_model / llm_call / clear_model_cache``，
      本文件多一个 ``safe_llm_call`` —— 只有``from workflows.base import *`` 才看得到这个差别。
"""

import sys
from pathlib import Path

# ---- 路径引导：必须在 import 项目内模块（workflows 包）之前执行 ----
# 直接 `python workflows/base.py` 时 sys.path[0] 是 workflows/ 目录，
# 找不到 workflows 包本身，会报 ModuleNotFoundError: No module named 'workflows'。
#
# 为什么用 parent.parent：__file__ 是 workflows/base.py，上一级是 workflows/ 包目录，
# 再上一级才是项目根 Media_Agent（也就是 `workflows` 包的父目录）。
# 注意 .venv 的 python_base_root.pth 里只有仓库根 F:\ProGram\Python_Base
# （实测内容确为该一行），够不到 Media_Agent，所以这里必须自己算。
#
# 顺带说明「为什么自检里看着能过」：verify_all.py 的 _child_env() 会给子进程注入
# PYTHONPATH=Media_Agent，把引导这件事替本文件做了 —— 手动直跑才暴露问题。
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
# 幂等：sys.path 里已有就跳过（重复 insert 只会让这个列表越堆越长）。
# 放队首是为了让项目内的模块优先于同名第三方包命中。
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# 只转发、不复制实现：真实现全在 workflows/__init__.py（理由见模块 docstring 的差异表）。
from workflows import clear_model_cache, get_model, llm_call  # noqa: E402

# 比 workflows.__all__ 多一个 safe_llm_call：它只在本文件定义。
# 注意 __all__ 只约束 `from workflows.base import *`，对 `import workflows.base` 无影响。
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
        ``model`` 传了 ``None`` 或已失效的实例也一样 —— ``invoke`` 抛出的
        ``AttributeError`` 同样落在下面的 ``except`` 里。
    """
    # 延迟导入：只有真要发请求时才需要它（与 workflows/__init__.py 的 llm_call 写法一致）。
    from langchain_core.messages import HumanMessage

    try:
        # 请求形状：单条 HumanMessage；正文从 response.content 取。
        response = model.invoke([HumanMessage(content=prompt)])
        content = response.content
        if isinstance(content, list):
            # 少数 OpenAI 兼容服务把正文按块返回（如 [{"type": "text", "text": "..."}]）。
            # 直接交给下游会被当成结构化数据，所以在这里压成纯字符串。
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)
    except Exception as exc:  # noqa: BLE001 —— 同上，节点不该因单次调用失败而中断
        return fallback if fallback else f"[LLM调用失败: {exc}]"


# 自检块：必须**纯离线**（verify_all.py 第 1 层在零密钥状态下跑它，不能发网络请求），
# 判据是退出码 —— 任何一条断言不成立就抛 AssertionError，子进程退出码随之为非 0。
if __name__ == "__main__":
    print("=== 工作流基类自检（离线）===")

    # 这几个 import 放块内：正常 import 本模块时不该有副作用，
    # 也不该把「读 config（进而读根 .env）」强加给所有调用方。
    import inspect

    import workflows
    from config import settings

    # 1) 转发关系：这三个名字必须与 workflows/__init__.py 里的是**同一个对象**，
    #    而不是各写一份 —— 本文件存在的意义就是避免「两处各改一半」。
    #    用 is 而不是比签名：签名相同、实现各写一份的「假转发」只有对象同一性能抓出来。
    assert llm_call is workflows.llm_call
    assert get_model is workflows.get_model
    assert clear_model_cache is workflows.clear_model_cache
    print("  转发对象同一性（llm_call / get_model / clear_model_cache）  OK")

    # 2) safe_llm_call 是「model 优先」，与 llm_call 的「prompt 优先」不同；
    #    这个差异是课案兼容的既有约定，改了会让调用方静默错位。
    #    连默认值一起钉住：本文件除了转发没有别的逻辑，签名就是它对外唯一的契约。
    sig = inspect.signature(safe_llm_call)
    assert list(sig.parameters) == ["model", "prompt", "fallback"], sig
    assert sig.parameters["fallback"].default == ""
    print(f"  safe_llm_call 签名 {sig}  OK")

    # 3) 模型不可用时返回中文提示文本，绝不抛异常（工作流节点靠它不中断）
    #    用「一调用就抛」的假模型打桩：不联网、不需要真密钥，专盯 except 分支 ——
    #    真模型失败往往是超时或限流，跑一次要几十秒，不适合放进离线自检。
    class _BrokenModel:
        def invoke(self, messages):
            raise RuntimeError("模拟：密钥无效")

    out = safe_llm_call(_BrokenModel(), "任意提示")
    assert out.startswith("[LLM调用失败:"), out
    print(f"  模型不可用 → 提示文本而非异常  OK  {out}")

    # 4) 「没配密钥」那条降级在转出的 llm_call 里（本文件只转发、不复制那个判断）。
    #    临时把密钥取空，全程离线，不会发起真实请求。
    #
    #    这里能直接给实例赋值，是因为 settings 是**普通类实例**（config.py 的
    #    `class Settings` 只继承 object），实例属性会盖住同名类方法；
    #    换成 pydantic 模型这一步会直接抛 ValueError。
    #    del 必须放 finally：不还原的话实例属性会一直盖着真方法，
    #    之后任何一次 llm_call 都会走「未配置」分支。
    settings.media_llm_api_key = lambda: ""
    try:
        out = llm_call("任意提示")
    finally:
        del settings.media_llm_api_key
    assert out.startswith("[LLM未配置]"), out
    print("  缺 API_KEY → [LLM未配置]（走 llm_call）  OK")

    # 收尾这行是给人看的标记；verify_all.py 第 1 层只认退出码 0
    #（本块没有显式 sys.exit，正常跑完即为 0，断言失败则是 1）。
    print("\n全部自检通过")

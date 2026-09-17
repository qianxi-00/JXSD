# -*- coding: utf-8 -*-
"""自媒体 Agent —— 工作流包（LangGraph 编排层）

课案出处：自媒体课案 → 各模块的 LangGraph 实现

本包是「前端 views/ → 工作流 workflows/ → 能力 tools/」三层里的中间层：
每个模块一个文件，内部用 ``StateGraph`` 把若干节点串起来，节点里调 LLM 或工具。

目录约定（与课案一致）
    __init__.py   本文件，LLM 调用的统一入口（llm_call / get_model）
    base.py       工作流基类：模型构造 + 带错误兜底的调用
    positioning.py   账号定位（串行 3 节点）
    hot_topic.py     热点监控（Send 并行抓取 → LLM 筛选 → 选题建议）
    replicate.py     内容复刻（下载 → ASR 提文案 → 拆解 → 仿写 → 标题）
    video.py         口播视频（提词器 / 数字人）
    mashup.py        视频剪辑（DeepAgents + video-use 技能）
    review.py        数据复盘（抖音采集 → 漏斗诊断 → 内容评估 → 优化策略）

LLM 从哪来（重要，别重复配密钥）
    默认统一走根目录 ``config.py`` 的扁平字段：
        settings.api_key     → 密钥
        settings.base_url    → OpenAI 兼容接口地址（当前是中转站）
        settings.model_name  → 模型名
    自媒体链路可用 ``MEDIA_LLM_MODEL`` 单独覆盖模型名（见 settings.media_llm_model()），
    但不重复配置密钥与地址。

    要**整条链路换服务商**时设 ``MEDIA_LLM_PROVIDER=deepseek``，
    改为读根 .env 里已存在的 ``DEEPSEEK_API_KEY`` / ``DEEPSEEK_BASE_URL`` /
    ``DEEPSEEK_MODEL``（见 settings.media_llm_api_key() / media_llm_base_url()）。
    这样换端点的动作只落在 Media_Agent 内 —— 根 .env 的 API_KEY/BASE_URL 是
    RAG 与 Agent 课案共用的，直接改它会连带影响别的子项目。

为什么用 ``init_chat_model`` 而不是课案的 ``ChatOpenAI(...)``
    本仓库已有的 Agent 课案实现（``Agent/03_deepagents/*_jxsd.py`` 等十余处）
    统一使用 ``init_chat_model``，这里保持一致，避免同仓库两套写法。
"""

import sys

from langchain.chat_models import init_chat_model

from config import settings

# Windows 控制台默认 GBK，本模块会打印中文日志
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

__all__ = ["get_model", "llm_call", "clear_model_cache"]


# 模型实例按 temperature 缓存：LangGraph 每个节点都会要一个模型，
# 不复用的话一次工作流要反复构造十几次客户端。
_MODEL_CACHE: dict[float, object] = {}


def get_model(temperature: float = 0.5, use_deepagent_model: bool = False):
    """取一个 LangChain 聊天模型实例（按 temperature 缓存复用）。

    Args:
        temperature: 采样温度。创作类用 0.7~0.8，分析诊断类用 0.3~0.4。
        use_deepagent_model: 为 True 时用 ``MEDIA_DEEPAGENT_MODEL``
            （视频剪辑那个 DeepAgent 链路很长，允许单独指定更稳的模型）。

    Returns:
        可直接 ``.invoke(...)`` 的聊天模型实例。
    """
    key = round(float(temperature), 2)
    if use_deepagent_model:
        # DeepAgent 用独立的缓存键，避免和普通节点串用同一个实例
        key = -key if key != 0 else -0.01
    if key not in _MODEL_CACHE:
        model_name = (
            settings.media_deepagent_model()
            if use_deepagent_model
            else settings.media_llm_model()
        )
        _MODEL_CACHE[key] = init_chat_model(
            model_provider="openai",
            model=model_name,
            api_key=settings.media_llm_api_key(),
            base_url=settings.media_llm_base_url(),
            temperature=temperature,
            timeout=90,      # 90 秒超时：长 prompt + 复杂推理需要更久
            max_retries=1,
        )
    return _MODEL_CACHE[key]


def clear_model_cache() -> None:
    """清空模型缓存（改完 .env 想立刻生效时用）。"""
    _MODEL_CACHE.clear()


def llm_call(prompt: str, temperature: float = 0.5, fallback: str = "") -> str:
    """调用 LLM 并返回纯文本 —— **绝不抛异常**。

    课案里每个节点都直接调这个函数；工作流节点一旦抛异常整条图就断了，
    而自媒体场景里「某一步失败」是常态（网络、限流、额度），
    所以这里把异常收成一段可读的中文提示，让下游节点自己判断。

    Args:
        prompt: 提示词。
        temperature: 采样温度。
        fallback: 指定时，失败返回它；不指定则返回 ``[LLM调用失败: ...]`` 文本。

    Returns:
        模型输出的文本；失败时为提示文本（不会是异常）。
    """
    from langchain_core.messages import HumanMessage

    if not settings.media_llm_api_key():
        msg = (
            "[LLM未配置] 根目录 .env 里没有 API_KEY。\n"
            "自媒体链路默认复用根配置的 API_KEY / BASE_URL / MODEL_NAME 三项；\n"
            "若设了 MEDIA_LLM_PROVIDER=deepseek，则改用 DEEPSEEK_API_KEY / "
            "DEEPSEEK_BASE_URL / DEEPSEEK_MODEL。"
        )
        return fallback if fallback else msg

    try:
        model = get_model(temperature=temperature)
        response = model.invoke([HumanMessage(content=prompt)])
        content = response.content
        # 有些模型会返回 list[dict]（多模态分块），统一压成字符串
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)
    except Exception as exc:  # noqa: BLE001 —— 故意的：节点不能因为 LLM 失败整条挂掉
        return fallback if fallback else f"[LLM调用失败: {exc}]"

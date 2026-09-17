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

与课案的落地差异
    | 项 | 课案原文 | 本实现 | 原因 |
    |---|---|---|---|
    | ``llm_call`` 签名 | 课案靠 ``base.py`` 的 ``llm_call = safe_llm_call`` 取别名，首形参是 ``model`` | 真入口在本文件，签名为 ``llm_call(prompt, temperature, fallback)`` | 课案正文 11 处都写 ``llm_call(prompt, temperature=...)``，别名那条路必 ``TypeError``；详见 ``base.py`` 的差异表 |
    | 实例复用 | 每个节点各自 ``ChatOpenAI(...)`` 新建 | 按 ``temperature`` 缓存复用（``_MODEL_CACHE``） | 一次工作流要取十几次模型，不复用就反复构造客户端、重复建连接池 |
    | 超时 / 重试参数名 | ``request_timeout=90`` / ``max_retries=1`` | ``timeout=90`` / ``max_retries=1``（值不变） | ``init_chat_model`` 认的是 ``ChatOpenAI.timeout`` 别名（实测传进去落到 ``request_timeout``，值 90.0） |
    | 缺密钥 | 无此分支 | 先查 ``settings.media_llm_api_key()``，为空直接返回中文提示文本 | 项目验收底线是「零密钥也必须全绿」，不能任由构造模型时抛出去 |
    | 响应为 ``list`` | 直接返回 ``response.content`` | 按分块拼成字符串 | OpenAI 兼容服务有返回 ``[{"type": "text", "text": ...}]`` 的情况 |
    | 绝对路径 | 无 | 无 | —— |

踩过的坑
    · **``temperature`` 要先归一化再当缓存键**：``round(float(t), 2)`` 让 ``0.7`` 与
      ``0.7000000001`` 命中同一个实例，否则浮点噪声会白白多造一堆客户端。
    · DeepAgent 那一路的缓存键用``取负``，而不是再开一个 dict —— 但 ``-0.0 == 0.0``
      （实测 ``hash`` 也相等），直接取负会让 ``temperature=0`` 与普通节点**撞同一个槽位**，
      所以零值特判成 ``-0.01``。这是反直觉的一处，别「顺手简化」掉。
    · ``max_retries`` 不写时 ``ChatOpenAI`` 的默认值是 ``None``（实测），
      会落到 openai SDK 自己的默认 2 次重试；这里显式钉成 1 —— 链路本来就有
      ``fallback`` 降级，快速失败比慢慢重试划算（否则最坏要等三次 90 秒超时）。
    · 缓存是**进程级**的：Streamlit 的 rerun 不会重执行 import，
      改完 ``.env`` 想立刻生效得显式调 ``clear_model_cache()``。
    · 本文件没有 ``if __name__ == "__main__":`` 自检块，所以**不在** ``verify_all.py``
      的 ``MODULE_SELF_CHECKS`` 里 —— 那份清单只收带 ``__main__`` 自检的模块。
"""

import sys

# 只导入工厂函数，不在这里实例化任何模型 —— 没配密钥时「导入本包」本身也必须能成功。
from langchain.chat_models import init_chat_model

from config import settings

# Windows 控制台默认 GBK，本模块会打印中文日志
# hasattr 守卫是必要的：stdout 被换成非标准对象（IDE、pytest 的捕获器）时没有 reconfigure，
# 直接调会 AttributeError。errors="replace" 则保证个别字符编不出来时不会把整行日志打断。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 包对外公开的三个名字；base.py 会再从本模块转发它们（外加它自己定义的 safe_llm_call）。
__all__ = ["get_model", "llm_call", "clear_model_cache"]


# 模型实例按 temperature 缓存：LangGraph 每个节点都会要一个模型，
# 不复用的话一次工作流要反复构造十几次客户端。
# 键 = round(temperature, 2)；DeepAgent 那一路再取负区分（见 get_model()）。
# 进程级：同一进程内所有节点共享，Streamlit 的 rerun 不会重放 import，所以它一直有效。
_MODEL_CACHE: dict[float, object] = {}


def get_model(temperature: float = 0.5, use_deepagent_model: bool = False):
    """取一个 LangChain 聊天模型实例（按 temperature 缓存复用）。

    Args:
        temperature: 采样温度。创作类用 0.7~0.8，分析诊断类用 0.3~0.4。
        use_deepagent_model: 为 True 时用 ``MEDIA_DEEPAGENT_MODEL``
            （视频剪辑那个 DeepAgent 链路很长，允许单独指定更稳的模型）。

    Returns:
        可直接 ``.invoke(...)`` 的聊天模型实例。
        **同一组 ``(temperature, use_deepagent_model)`` 返回的是同一个实例** ——
        命中缓存就直接给，不会每次重新建客户端。
    """
    key = round(float(temperature), 2)
    if use_deepagent_model:
        # DeepAgent 用独立的缓存键，避免和普通节点串用同一个实例
        # 用「取负」而不是再开一个 dict：改动面最小。
        # 但零值必须特判 —— -0.0 == 0.0 且 hash(-0.0) == hash(0.0)（实测两者都成立），
        # 直接写 key = -key 会让 temperature=0 与普通节点命中同一个槽位，
        # 这里取 -0.01 正是为了躲开那一格。
        key = -key if key != 0 else -0.01
    if key not in _MODEL_CACHE:
        # 模型名 / 密钥 / 地址三项都走 config 的解析方法，业务代码不直接读 .env：
        # 设了 MEDIA_LLM_PROVIDER=deepseek 时它们会整体切到 DEEPSEEK_* 那一组。
        model_name = (
            settings.media_deepagent_model()
            if use_deepagent_model
            else settings.media_llm_model()
        )
        _MODEL_CACHE[key] = init_chat_model(
            model_provider="openai",  # 走 OpenAI 兼容协议：中转站与 DeepSeek 都是这个形状
            model=model_name,
            api_key=settings.media_llm_api_key(),
            base_url=settings.media_llm_base_url(),
            temperature=temperature,
            timeout=90,      # 90 秒超时：长 prompt + 复杂推理需要更久（课案原值，参数名随 API 改）
            max_retries=1,   # 1 次重试：不写时 ChatOpenAI 默认是 None，会落到 openai SDK 默认的 2 次（实测）；链路自带 fallback，快速失败更划算
        )
    return _MODEL_CACHE[key]


def clear_model_cache() -> None:
    """清空模型缓存（改完 .env 想立刻生效时用）。

    注意它只清**模型实例**：密钥 / 地址 / 模型名的来源 ``settings`` 是 ``@lru_cache``
    的进程级单例（见 ``config.py`` 的 ``get_settings()``），``.env`` 的新值不会因此被重读。
    真要连配置一起换掉，还得清 ``get_settings`` 的缓存并重启进程。

    Returns:
        None。
    """
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
        失败分两种：没配密钥 → ``[LLM未配置] ...``；调用出错 → ``[LLM调用失败: ...]``。
        指定了 ``fallback`` 时两种都返回 ``fallback``。
    """
    # 延迟导入（与 base.safe_llm_call 同一写法）：只在真要发请求时才需要它。
    from langchain_core.messages import HumanMessage

    # 先查密钥再动模型：零密钥是常态（verify_all.py 离线跑、CI 没有密钥），
    # 这里必须**早退**成一段可读文本，而不是让 get_model 在构造时把异常抛给节点。
    if not settings.media_llm_api_key():
        msg = (
            "[LLM未配置] 根目录 .env 里没有 API_KEY。\n"
            "自媒体链路默认复用根配置的 API_KEY / BASE_URL / MODEL_NAME 三项；\n"
            "若设了 MEDIA_LLM_PROVIDER=deepseek，则改用 DEEPSEEK_API_KEY / "
            "DEEPSEEK_BASE_URL / DEEPSEEK_MODEL。"
        )
        return fallback if fallback else msg

    try:
        # 请求形状：单条 HumanMessage；正文从 response.content 取。
        # get_model 命中缓存时不建新客户端，所以这一步在节点里几乎是零成本。
        model = get_model(temperature=temperature)
        response = model.invoke([HumanMessage(content=prompt)])
        content = response.content
        # 有些模型会返回 list[dict]（多模态分块），统一压成字符串
        # 直接返回会是 Python 字面量而不是正文，下游按字符串处理的地方会读到一堆键名。
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        return str(content)
    # 兜住一切：网络超时、限流、额度用尽、模型名写错、base_url 不通……
    # 工作流节点正是靠这段不中断（课案里每个节点都直接调本函数）。
    except Exception as exc:  # noqa: BLE001 —— 故意的：节点不能因为 LLM 失败整条挂掉
        return fallback if fallback else f"[LLM调用失败: {exc}]"

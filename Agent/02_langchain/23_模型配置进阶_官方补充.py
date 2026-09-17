# -*- coding: utf-8 -*-
r"""
LangChain 官方补充篇：模型配置进阶（非课案内容，故不带 _jxsd 后缀）
================================================================
来源与定位：
    本文件对照 LangChain **官方文档** /oss/python/langchain/models.mdx
    （含 messages.mdx 的「内容块 / 多模态」一节），补缺口表 **第 10 项**。

课案 02_langchain/01_模型 讲了「怎么把模型接上」；这一篇讲的是**接上之后的工程细节**：
    · 模型参数：temperature / max_tokens / timeout / max_retries 各自管什么；
    · 连接韧性：重试策略（官方默认 max_retries=6、指数退避 + 抖动，
      只重试网络错误 / 429 / 5xx，401 与 404 不重试）；
    · 限流：`InMemoryRateLimiter` 挂在模型上，避免把自己的额度打穿；
    · 用量核算：`usage_metadata` 的完整字段（含推理 token 与缓存命中）；
    · 内容块与多模态：`content_blocks` 的形态，以及本机网关的现实支持情况。

⚠️ 本文件有两个**实测得出的重要结论**（官方文档不会告诉你，因为它们是本机网关的脾气）：
    A. **中转网关可能无视 `max_tokens`**：本机设 `max_tokens=16` 让它写长文，
       实际返回了 4540 个输出 token、`finish_reason=stop` —— 参数被接受但没被执行。
       所以"限制输出长度"不能只依赖这个参数，要在提示词里约束或自己截断；
    B. **图片内容块在本机不可用**：带 image 块的请求直接 `OpenAIConnectionError:
       Connection error`（同一端点纯文本调用正常）—— 无法确认是网关不支持还是链路问题，
       结论就是"本机不具备多模态验证条件"，多模态那部分缺口维持待补。

⚠️ 本文件需要真实模型（且会故意触发一次超时错误，属预期行为）。

运行方式（项目根目录下）：
    uv run Agent/02_langchain/23_模型配置进阶_官方补充.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import time

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.rate_limiters import InMemoryRateLimiter

from config import settings


def build_model(**overrides):
    """按 .env 里的配置造一个模型实例，允许覆盖任意模型参数。"""
    kwargs = {
        "model_provider": "openai",
        "model": settings.model_name,
        "api_key": settings.api_key,
        "base_url": settings.base_url,
        "max_retries": 0,      # 演示里关掉自动重试，避免每次都等 6 轮退避
    }
    kwargs.update(overrides)
    return init_chat_model(**kwargs)


# 用量累加器：核算成本的基础（官方字段见 Demo 5）
USAGE_TOTAL = {"input_tokens": 0, "output_tokens": 0, "reasoning": 0, "cache_read": 0}


def collect_usage(response) -> dict:
    """把一次响应的 usage_metadata 累加进来，并返回本次明细。"""
    usage = getattr(response, "usage_metadata", None) or {}
    details_out = usage.get("output_token_details") or {}
    details_in = usage.get("input_token_details") or {}
    USAGE_TOTAL["input_tokens"] += usage.get("input_tokens", 0)
    USAGE_TOTAL["output_tokens"] += usage.get("output_tokens", 0)
    USAGE_TOTAL["reasoning"] += details_out.get("reasoning", 0)
    USAGE_TOTAL["cache_read"] += details_in.get("cache_read", 0)
    return usage


# ================================================================
# Demo 1：模型参数与响应元数据
# ================================================================
# 官方 models.mdx 的参数表：temperature（随机性）、max_tokens（输出上限）、
# timeout（秒）、max_retries（默认 6）。它们通过 init_chat_model 的 **kwargs 传入。
# 除了参数，响应对象本身也带着情报：response_metadata（模型名/结束原因/token 用量）
# 与 content_blocks（内容块列表，LangChain 1.x 的标准形态）。
def demo_1_parameters_and_metadata() -> None:
    print("=" * 70)
    print("Demo 1：模型参数与响应元数据")
    print("=" * 70)

    model = build_model(temperature=0.2, timeout=60)
    response = model.invoke("用一句话说明 temperature 参数的作用。")

    print(f"  回答：{str(response.content)[:90]}…")
    print(f"  content_blocks：{response.content_blocks}")
    metadata = response.response_metadata or {}
    print(f"  模型名：{metadata.get('model_name')}｜结束原因：{metadata.get('finish_reason')}")
    print(f"  token 用量：{collect_usage(response)}")
    print(
        "  ↑ 三个值得记住的字段：\n"
        "    · response_metadata.finish_reason：**是不是被截断**看它（stop=正常结束）；\n"
        "    · content_blocks：LangChain 1.x 的标准内容形态（文本/推理/工具调用/图片…）；\n"
        "    · usage_metadata：含推理 token 与缓存命中，是成本核算的唯一依据。"
    )


# ================================================================
# Demo 2：max_tokens 的实测真相（网关不一定听你的）
# ================================================================
def demo_2_max_tokens_reality() -> None:
    print("\n" + "=" * 70)
    print("Demo 2：max_tokens 的实测真相 —— 中转网关可能无视它")
    print("=" * 70)

    model = build_model(max_tokens=16)
    try:
        # 只要 300 字就够证明问题：max_tokens=16 若真生效，输出会被截到十几个 token。
        # （早先要求"至少两千字"时更慢更费额度，结论完全一样，所以缩短。）
        response = model.invoke("用大约 300 字介绍 LangGraph 的持久化机制。")
    except Exception as exc:  # noqa: BLE001
        print(f"  本次调用失败（网关抖动/额度问题，非代码问题）：{type(exc).__name__}: {str(exc)[:100]}")
        print("  重跑一次通常即可；本 Demo 要演示的是参数是否被执行，换个时段再验也行。")
        return
    usage = collect_usage(response)
    text = str(response.content)
    print(f"  请求参数 max_tokens=16，实际返回：{len(text)} 字符 / {usage.get('output_tokens')} 输出 token")
    print(f"  finish_reason：{(response.response_metadata or {}).get('finish_reason')}")
    if usage.get("output_tokens", 0) > 100:
        print(
            "  ⚠️ 结论：**本机网关没有执行 max_tokens**（参数被接受但不生效）——\n"
            "     换端点/直连官方 API 时通常有效，但**永远不要假设它一定生效**。\n"
            "     要控长度就三管齐下：提示词里限字数 + 自己截断 + 用 finish_reason 监控。"
        )
    else:
        print("  ✔ 本次 max_tokens 生效（输出被压到阈值内）")


# ================================================================
# Demo 3：连接韧性 —— timeout 与 max_retries
# ================================================================
# 官方语义（models.mdx「Connection resilience」）：
#   · max_retries 默认 6，用**指数退避 + 抖动**重发；
#   · 只对网络错误、429（限流）、5xx 重试；401 / 404 这类客户端错误不重试；
#   · 长任务建议 max_retries=10~15 并配 checkpointer（失败可续）。
# 本 Demo 用一个**必然超时**的小 timeout 证明超时参数真的生效
# （max_retries=0，否则会等满 6 轮退避，演示就太慢了）。
def demo_3_resilience() -> None:
    print("\n" + "=" * 70)
    print("Demo 3：连接韧性 —— timeout 生效验证（故意造一次超时）")
    print("=" * 70)

    model = build_model(timeout=0.001)     # 1 毫秒：必然超时
    started = time.time()
    try:
        model.invoke("你好")
        print("  居然没超时？（不符合预期）")
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - started
        print(f"  预期内失败：{type(exc).__name__}（耗时 {elapsed:.2f}s）")
        print(f"  异常文本：{str(exc)[:100]}")
    print(
        "  ↑ timeout 确实被传给了底层 HTTP 客户端（1 毫秒必然失败）。\n"
        "    生产建议：网络不稳就给 timeout=120、max_retries=10~15，并配 checkpointer；\n"
        "    注意**不是所有错误都重试**（401/404 直接失败，免得白等 6 轮）。"
    )


# ================================================================
# Demo 4：限流 —— InMemoryRateLimiter（零模型调用，直接测节流器）
# ================================================================
# 官方把限流器挂在**模型实例**上：`init_chat_model(..., rate_limiter=limiter)`，
# 之后每次模型调用（含 agent 内部的）都会先向它申请令牌。
# 本 Demo 不调模型，直接测限流器本身的节流效果 —— 结论干净、还不花钱。
#   参数（实测签名）：requests_per_second / check_every_n_seconds / max_bucket_size
def demo_4_rate_limiter() -> None:
    print("\n" + "=" * 70)
    print("Demo 4：限流器 —— 每秒 2 次，连要 4 个令牌要等多久？")
    print("=" * 70)

    limiter = InMemoryRateLimiter(
        requests_per_second=2,       # 每秒放行 2 次
        check_every_n_seconds=0.05,  # 检查间隔（越小越精确、越费 CPU）
        max_bucket_size=1,           # 桶容量：允许的突发量
    )
    started = time.time()
    stamps = []
    for index in range(4):
        limiter.acquire()            # 向限流器申请一次放行（同步版）
        stamps.append(time.time() - started)
        print(f"  第 {index + 1} 个令牌拿到，累计 {stamps[-1]:.2f}s")
    gaps = [round(stamps[i + 1] - stamps[i], 2) for i in range(len(stamps) - 1)]
    print(f"  相邻间隔：{gaps} 秒（2 req/s 的稳态间隔应约 0.5 秒）")
    print(
        "  ↑ 用法：`init_chat_model(..., rate_limiter=limiter)` —— 挂上之后，\n"
        "    agent 里每一次模型调用都会先申请令牌，天然把并发压在你允许的速率内。\n"
        "    `max_bucket_size` 是突发容量：设 1 表示不许攒额度、必须匀速。"
    )


# ================================================================
# Demo 5：token 用量核算（跑两次，看累计）
# ================================================================
def demo_5_token_accounting() -> None:
    print("\n" + "=" * 70)
    print("Demo 5：token 用量核算（成本核算的基础）")
    print("=" * 70)

    # 先清零：USAGE_TOTAL 是模块级累加器，前面几个 Demo 也往里加过
    # （不清零就会打印出「本次 2 次调用」却包含前面所有调用的总量 —— 本文件踩过）
    for key in USAGE_TOTAL:
        USAGE_TOTAL[key] = 0

    model = build_model()
    rounds = 0
    for question in ("一句话解释什么是向量检索。", "一句话解释什么是重排序。"):
        response = model.invoke(question)
        rounds += 1
        usage = collect_usage(response)
        print(f"  提问：{question}")
        print(f"    输入 {usage.get('input_tokens')} / 输出 {usage.get('output_tokens')} "
              f"/ 其中推理 {((usage.get('output_token_details') or {}).get('reasoning'))} "
              f"/ 缓存命中 {((usage.get('input_token_details') or {}).get('cache_read'))}")
    print(f"\n  本段累计（{rounds} 次调用）：{USAGE_TOTAL}")
    print(
        "  ↑ 注意两点（都是本机实测观察到的）：\n"
        "    ① **推理 token 计入输出**：本机模型输出里很大一部分是 reasoning，成本要算进去；\n"
        "    ② **缓存命中会显著省钱**：input_token_details.cache_read 命中越多、输入越便宜。\n"
        "    把 collect_usage 这种累加器挂在生产链路里，才能回答「这个功能一次多少钱」。"
    )


# ================================================================
# Demo 6：内容块与多模态的现实
# ================================================================
# 是否真的发一次图片块请求。默认开启 —— 实测它是**可捕获的异常**，不会带崩脚本：
#     图片块调用失败：OpenAIConnectionError: Connection error.（纯文本调用同一端点正常）
# 注意：如果直接跑本文件时曾在 Demo 6 处"整跑中断、日志为空"，那多半不是这里的问题，
# 而是 **PowerShell 的 `*>` / 管道重定向会缓冲输出**，进程被中断时缓冲丢光导致的假象。
# 排查这类脚本建议让 Python 自己写日志（见 README 排障一节）。
MULTIMODAL_LIVE_TEST = True


def demo_6_multimodal_reality() -> None:
    print("\n" + "=" * 70)
    print("Demo 6：多模态（内容块）在本机的现实")
    print("=" * 70)

    # 一个 1x1 的透明 PNG（base64），用来最小化测试图片输入
    tiny_png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    message = HumanMessage(content=[
        {"type": "text", "text": "这张图片是什么颜色？一句话回答。"},
        {"type": "image", "base64": tiny_png, "mime_type": "image/png"},
    ])
    print("  内容块的标准写法（跨厂商通用，支持视觉的模型都能吃）：")
    print(f"    HumanMessage(content=[{{'type': 'text', ...}}, "
          f"{{'type': 'image', 'base64': '<base64>', 'mime_type': 'image/png'}}])")

    if not MULTIMODAL_LIVE_TEST:
        print("\n  （已跳过真实调用：MULTIMODAL_LIVE_TEST=False）")
        print("  实测结论：本机网关对图片块**不可用** —— OpenAIConnectionError: Connection error，")
        print("    而同一端点的纯文本调用正常。")
    else:
        model = build_model()
        try:
            response = model.invoke([message])
            print(f"\n  ✔ 网关接受了图片块：{str(response.content)[:80]}")
        except Exception as exc:  # noqa: BLE001
            print(f"\n  ✘ 图片块调用失败：{type(exc).__name__}: {str(exc)[:120]}")
            print("    （这是**可捕获的异常**，脚本会继续往下跑，不影响其它 Demo）")

    print(
        "  ↑ 结论：多模态相关缺口在本仓库继续挂「待补（要支持视觉的端点）」。\n"
        "    代码写法本身是对的（内容块是 LangChain 1.x 的跨厂商标准），\n"
        "    换一个支持视觉的模型端点即可跑通，无需改代码。"
    )


if __name__ == "__main__":
    demo_1_parameters_and_metadata()
    demo_2_max_tokens_reality()
    demo_3_resilience()
    demo_4_rate_limiter()
    demo_5_token_accounting()
    demo_6_multimodal_reality()
    print("\n全部 Demo 执行完毕。")


# ================================================================
# 实测结论 / 未收录清单 / 踩坑提示
# ================================================================
# 1. 官方事实（models.mdx）：
#    - 常用参数：temperature / max_tokens / timeout / max_retries（默认 6）；
#    - 重试策略：指数退避 + 抖动；只重试网络错误、429、5xx；401/404 不重试；
#    - 长任务建议 max_retries=10~15 并配 checkpointer；
#    - 内容块：`response.content_blocks` 是标准形态，跨厂商通用；
#    - 限流：`langchain_core.rate_limiters.InMemoryRateLimiter` 挂在模型实例上。
# 2. 本机实测（这一跑的真实观察）：
#    - Demo 1：参数被接受；response_metadata 含 model_name / finish_reason / token_usage；
#      content_blocks 返回 [{'type': 'text', 'text': ...}]；usage 含 reasoning 与 cache_read；
#    - Demo 2：**max_tokens=16 未被网关执行** —— 多次实测分别返回 4574 / 6147 / 4729
#      输出 token 且 finish_reason=stop（提示词缩短到 300 字后仍是数千 token）；
#    - Demo 3：timeout=0.001 立即失败（证明参数确实传到 HTTP 层）；
#    - Demo 4：requests_per_second=2 时，4 个令牌按约 0.5 秒间隔发放；
#    - Demo 5：两次调用可累计出 input/output/reasoning/cache_read 四个口径；
#    - Demo 6：图片块请求报 OpenAIConnectionError（纯文本正常）→ 本机无多模态条件。
# 3. 未收录（官方还有、本文件没做的）：
#    - **模型档位/动态切模型**：本文件没重复造轮子 —— 动态换模型见
#      `20_上下文工程_官方补充.py` Demo 1（request.override(model=...)）与
#      `11_内置中间件_官方补充.py`（ModelFallbackMiddleware 失败降级）；
#    - **异步调用 / 批量 / 流式的模型层细节**：流式见 `19_事件流v3_官方补充.py`；
#    - **provider 专属参数**（如 ChatOpenAI 的 use_responses_api）：换厂商时再查对应集成页；
#    - **多模态输入输出**：本机链路不支持，维持待补。
# 4. 踩坑提示：
#    A. **别假设 max_tokens 生效**：中转网关可能忽略它（本机实测忽略）；
#       控长度要靠提示词 + 自截断 + 监控 finish_reason；
#    B. **推理 token 计入输出**：本机模型输出里 reasoning 占大头，估算成本时漏了它就会低估；
#    C. `max_retries` 默认 6：调试/演示时务必显式设 0，否则一次失败要等满退避序列；
#       （反过来，生产里调大它 + 配 checkpointer 才是抗网络抖动的正解）
#    D. 限流器挂在**模型实例**上，不是挂在 agent 上：同一实例被 agent 与工具共用时，
#       限流对它们**统一生效**；换实例就换了一份额度；
#    E. `max_bucket_size` 决定突发能力：设 1 = 严格匀速，设大 = 允许短时突发（容易撞限额）。

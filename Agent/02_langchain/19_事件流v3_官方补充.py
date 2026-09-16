# -*- coding: utf-8 -*-
r"""
LangChain 官方补充篇：事件流 v3（非课案内容，故不带 _jxsd 后缀）
================================================================
来源与定位：
    本文件对照 LangChain **官方文档** /oss/python/langchain/event-streaming.mdx，
    补上课案 07_流式 章之后的另一半：**官方现在推荐新项目用 v3 事件流**。

官方原文（第一段就给了结论）：
    「For most application and frontend use cases, use **Event Streaming** through
      `stream_events(..., version="v3")`. Event Streaming returns a run object with
      typed projections, so each projection can be consumed independently instead of
      parsing stream-mode tuples.」

课案 07_流式 讲的是 `stream_mode`（values / updates / messages / custom 等），
那套仍然可用；v3 的差别是**把每种数据做成独立投影**，不必再去解析元组：

    投影                          用途
    ----------------------------  ------------------------------------------
    for event in stream           最原始的协议事件（全信封、所有通道）
    stream.messages               模型消息流，每次 LLM 调用一个
    message.text                  文本增量（逐 token）
    message.reasoning             推理内容增量（模型支持时才有）
    message.tool_calls            工具调用入参的增量与最终结果
    message.output                模型调用完成后的完整消息对象
    stream.values                 agent 状态快照
    stream.output                 最终状态
    stream.subgraphs              嵌套子图运行
    stream.subagents              **命名**子代理的运行（可拿到内层消息）
    stream.extensions             自定义 transformer 投影
    stream.tool_calls             工具执行生命周期（入参/输出增量/输出/错误）

⚠️ 两条本机实测要点（官方文档没放在显眼处）：
    A. v3 目前是**实验性协议**：本地 langchain 1.4.0 / langgraph 1.2.11 调用时会打印
       `LangChainBetaWarning: The v3 streaming protocol on Pregel is experimental` ——
       能在生产用，但要预期 API 变动；
    B. **投影是单次消费的**：同一个 stream 上先把 `stream.messages` 抽干，
       再读 `stream.tool_calls` 会拿到空结果。多个投影要么各开一个新 stream，
       要么用 `stream.interleave(...)` 一次交织消费（Demo 2 实测对比了这两种写法）。

⚠️ 本文件需要真实模型（流式内容来自模型）。

运行方式（项目根目录下）：
    uv run Agent/02_langchain/19_事件流v3_官方补充.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错
import warnings

from langchain.agents import create_agent
from langchain.agents.middleware import ToolErrorMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.tools import ToolException
from config import settings

# v3 是实验性协议，每次调用都会打 Beta 警告；这里统一静音以免刷屏，
# 但**请知道它的存在**（升级 langgraph 时优先回归测试本文件）。
warnings.filterwarnings("ignore", category=Warning, module="langgraph")

model = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

INPUT = {"messages": [{"role": "user", "content": "北京天气如何？用一句话回答。"}]}


@tool
def get_weather(city: str) -> str:
    """查询城市天气。"""
    return f"{city}：晴，25℃"


@tool
def risky_lookup(keyword: str) -> str:
    """查询一个可能不存在的东西（演示工具报错在流里怎么体现）。"""
    # ToolException 会被框架转成错误 ToolMessage（普通异常会直接中断运行，
    # 原因见 02_langchain/16_测试与护栏_官方补充.py Demo 4 的实测）
    raise ToolException(f"下游服务查不到 {keyword!r}")


def build_weather_agent():
    return create_agent(model=model, tools=[get_weather])


# ================================================================
# Demo 1：基础 —— 逐 token 文本、finalize 的消息对象、token 用量
# ================================================================
def demo_1_basic_projections() -> None:
    print("=" * 70)
    print("Demo 1：stream.messages —— 逐 token 文本 + 完整消息 + token 用量")
    print("=" * 70)

    agent = build_weather_agent()
    stream = agent.stream_events(INPUT, version="v3")

    for index, message in enumerate(stream.messages, start=1):
        print(f"\n  第 {index} 次模型调用（节点：{message.node}）")
        # .text 是可迭代的增量流；边收边打印就是打字机效果
        pieces = []
        for delta in message.text:
            pieces.append(delta)
        text = "".join(pieces)
        print(f"    文本增量拼接：{text[:90]!r}" if text else "    （本次没有文本输出，只产出了工具调用）")

        # .output 是 finalize 后的完整 AIMessage
        final = message.output
        usage = getattr(final, "usage_metadata", None)
        if usage:
            print(f"    token 用量：输入 {usage.get('input_tokens')} / 输出 {usage.get('output_tokens')}"
                  f" / 合计 {usage.get('total_tokens')}")
            details = usage.get("output_token_details") or {}
            if details.get("reasoning"):
                print(f"    其中推理 token：{details['reasoning']}（该模型会产出 reasoning）")

        # reasoning 投影：模型不吐推理内容时这里是空的
        reasoning = "".join(delta for delta in message.reasoning)
        print(f"    reasoning 投影长度：{len(reasoning)} 字符")

        # 工具调用投影（本条消息发起的调用）
        finalized = message.tool_calls.get() if hasattr(message.tool_calls, "get") else None
        if finalized:
            print(f"    本条消息发起的工具调用：{[(c['name'], c['args']) for c in finalized]}")

    print(f"\n  最终状态里的最后一条消息：{str(stream.output['messages'][-1].content)[:90]}")
    print(
        "  ↑ 与课案 07 的 stream_mode='messages' 相比，v3 把「文本 / 推理 / 工具调用 / 完整消息 /\n"
        "    token 用量」拆成了同一对象上的不同属性，前端按需取用，不必再解析元组。"
    )


# ================================================================
# Demo 2：投影的消费规则（本文件最值钱的一课）
# ================================================================
# 实测现象：
#   ① 先 for m in stream.messages（抽干）→ 再 list(stream.tool_calls) → **空**；
#   ② 单开一个新 stream 只读 tool_calls → 正常拿到；
#   ③ stream.interleave("messages", "tool_calls", "values") → 一次拿到所有投影的事件。
# 结论：投影是**同一条底层事件流的多个视图**，谁先被抽干谁就把事件消费掉了。
def demo_2_consumption_rules() -> None:
    print("\n" + "=" * 70)
    print("Demo 2：投影的单次消费规则（错误写法 vs 正确写法）")
    print("=" * 70)

    agent = build_weather_agent()

    # ---- 错误写法：同一个 stream 上依次消费两个投影 ----
    stream = agent.stream_events(INPUT, version="v3")
    message_count = sum(1 for _ in stream.messages)
    leftover_calls = len(list(stream.tool_calls))
    print(f"  错误写法（先 messages 后 tool_calls）：messages={message_count} 条，"
          f"tool_calls={leftover_calls} 个 ← 被抽干了")

    # ---- 正确写法 A：每个投影开一个新 stream（最简单）----
    stream_a = agent.stream_events(INPUT, version="v3")
    calls_a = list(stream_a.tool_calls)
    print(f"  正确写法 A（各开新 stream）：tool_calls={len(calls_a)} 个")

    # ---- 正确写法 B：interleave 一次交织消费（推荐，只跑一次图）----
    stream_b = agent.stream_events(INPUT, version="v3")
    counts: dict[str, int] = {}
    for kind, _payload in stream_b.interleave("messages", "tool_calls", "values"):
        counts[kind] = counts.get(kind, 0) + 1
    print(f"  正确写法 B（interleave）：{counts}")
    print(
        "  ↑ 记这一条就够了：**想同时要多种投影，就用 interleave**；\n"
        "    各开新 stream 虽然简单，但每一路都会把图重跑一遍（模型要重复付费）。"
    )


# ================================================================
# Demo 3：工具执行生命周期 —— 入参、输出、以及 error 字段
# ================================================================
def demo_3_tool_lifecycle() -> None:
    print("\n" + "=" * 70)
    print("Demo 3：stream.tool_calls —— 工具执行生命周期（含失败）")
    print("=" * 70)

    agent = create_agent(model=model, tools=[get_weather, risky_lookup])

    # ---- A. 正常调用：注意要把流**消费完**再读 output ----
    stream = agent.stream_events(INPUT, version="v3")
    calls = list(stream.tool_calls)      # 先抽干，事件才会 finalize
    for call in calls:
        print(f"  ✔ 正常调用：{call.tool_name}({call.input})")
        print(f"     输出类型：{type(call.output).__name__}，内容：{str(call.output)[:60]}")
        print(f"     error 字段：{call.error}")
    print(
        "     实测提醒：在 for 循环里边收边读 `call.output` 会拿到 **None** ——\n"
        "     投影是按生命周期推进的，必须把流消费完（或先 list() 收集）后再读最终值。"
    )

    # ---- B. 工具失败但没有错误处理中间件 ----
    print("\n  --- B. 工具抛 ToolException，但**没有**错误处理中间件 ---")
    stream = agent.stream_events(
        {"messages": [{"role": "user", "content": "查一下 keyword='不存在的记录'"}]},
        version="v3",
    )
    try:
        list(stream.tool_calls)
        print("     竟然没抛异常？（不符合预期）")
    except Exception as exc:  # noqa: BLE001
        print(f"     运行直接中断：{type(exc).__name__}: {str(exc)[:70]}")
    print(
        "     实测：本版本里 `ToolException` 也会被 ToolNode 的默认处理**原样抛出**，\n"
        "     整个运行中断 —— 流里自然也就看不到任何「工具失败」事件。"
    )

    # ---- C. 同一个失败工具，挂上 ToolErrorMiddleware ----
    print("\n  --- C. 同一个失败工具，挂 ToolErrorMiddleware ---")

    def on_error(exc: Exception, request) -> str:
        """把工具异常转成模型能读懂的说明（返回 None 则让异常继续往外抛）。

        注意签名是**两个参数**：(exc, request) —— request.tool_call 里有工具名/入参/id，
        需要按工具分流处理时用它（写成单参数会报 TypeError，本文件实测踩过）。
        """
        print(f"     [on_error] 捕获 {type(exc).__name__}：{str(exc)[:50]}")
        return f"工具执行失败（{type(exc).__name__}）。请换个关键词或直接说明无法查询。"

    safe_agent = create_agent(
        model=model,
        tools=[get_weather, risky_lookup],
        middleware=[ToolErrorMiddleware(on_error)],
    )
    stream = safe_agent.stream_events(
        {"messages": [{"role": "user", "content": "查一下 keyword='不存在的记录'"}]},
        version="v3",
    )
    for call in list(stream.tool_calls):
        print(f"     失败调用：{call.tool_name}({call.input})")
        print(f"       输出：{str(call.output)[:70]}")
        print(f"       error 字段：{call.error}")
    print(
        "  ↑ 对照很清楚：**不挂中间件 → 运行中断；挂上 → 失败变成一条普通的工具事件**，\n"
        "    流能正常跑完、模型也能读到失败原因并换个办法。\n"
        "    这也是官方错误处理四分类里「LLM 可修复」那一类的落地方式\n"
        "    （中间件细节见 02_langchain/11_内置中间件_官方补充.py Demo 1）。"
    )


# ================================================================
# Demo 4：子代理投影 —— 命名子代理的运行能单独看
# ================================================================
# 官方说明：内层 agent 通过**包裹工具**被调用时，它的事件落在嵌套命名空间；
# 你在 create_agent(name=...) 里给的名字就是流里的标识，而 .cause 是派发它的那次工具调用。
def demo_4_subagents() -> None:
    print("\n" + "=" * 70)
    print("Demo 4：stream.subagents —— 命名子代理单独成流")
    print("=" * 70)

    inner = create_agent(
        model=model,
        tools=[get_weather],
        system_prompt="你只回答天气问题，一句话以内。",
        name="weather_expert",          # ← 这个名字决定它在流里的标识
    )

    @tool
    def ask_weather_expert(question: str) -> str:
        """把天气问题转给天气专家代理。"""
        result = inner.invoke({"messages": [{"role": "user", "content": question}]})
        return str(result["messages"][-1].content)

    outer = create_agent(model=model, tools=[ask_weather_expert])
    stream = outer.stream_events(
        {"messages": [{"role": "user", "content": "帮我问天气专家：上海天气怎么样？"}]},
        version="v3",
    )

    # 内联消费：边拿子代理句柄边读它自己的投影（不要在循环外先 list()，理由见 Demo 2）
    inner_texts: list[str] = []
    for sub in stream.subagents:
        cause = getattr(sub, "cause", None)
        # 实测：cause 是个 dict（形如 {'type': 'toolCall', 'tool_call_id': ...}），
        # 里面**没有**工具名字段，要拿工具名得回到消息里按 tool_call_id 反查
        cause_id = cause.get("tool_call_id") if isinstance(cause, dict) else None
        print(f"  子代理 name={sub.name!r}，由工具调用 id={cause_id} 派发")
        for message in sub.messages:
            text = "".join(delta for delta in message.text) or str(message.output.content)
            inner_texts.append(text[:80])
            print(f"    内层模型输出：{text[:80]!r}")

    if not inner_texts:
        print("  内层 messages 为空 —— 说明句柄拿到了但内层消息没被投影出来（见文末实测结论）")
    print(
        "  ↑ 子代理投影的价值：多 Agent 系统里，主代理与子代理的输出能**分别**喂给前端 ——\n"
        "    用户既能看到「主代理在等子代理」，也能看到子代理自己说了什么。"
    )


if __name__ == "__main__":
    demo_1_basic_projections()
    demo_2_consumption_rules()
    demo_3_tool_lifecycle()
    demo_4_subagents()
    print("\n全部 Demo 执行完毕。")


# ================================================================
# 实测结论 / 未收录清单 / 踩坑提示
# ================================================================
# 1. 实测结论（langchain 1.4.0 / langgraph 1.2.11，本机）：
#    - `agent.stream_events(input, version="v3")` 返回 `GraphRunStream`，可用投影：
#      abort / extensions / interleave / interrupted / interrupts / lifecycle /
#      messages / output / subagents / subgraphs / tool_calls / values；
#    - 调用会打印 LangChainBetaWarning（v3 是实验性协议），已在文件里统一静音；
#    - Demo 1：message.text 能拼出完整文本；message.output.usage_metadata 给出
#      输入/输出/合计 token 与推理 token（本机模型实测 reasoning=208）；
#    - Demo 2：同一 stream 上先消费 messages 后，tool_calls 拿到 **0** 个（被抽干）；
#      新开 stream 或 interleave 都能正常拿到（interleave 实测交织出 7 个事件）；
#    - Demo 3：stream.tool_calls 能拿到 tool_name / input / output / error。
#      三个实测细节：① 正常调用的 output 是 **ToolMessage**，但必须把流消费完再读
#      （循环里边收边读是 None）；② 工具抛 ToolException 且**没挂**错误中间件时，
#      运行直接中断、流里看不到失败事件；③ 挂上 ToolErrorMiddleware 后同一失败
#      变成 `output=None, error='下游服务查不到 ...'` —— 失败成为可观测的一等事件；
#    - Demo 4：stream.subagents 能按 name= 对齐子代理；**内联消费**（边拿句柄边读
#      sub.messages）能拿到内层 2 条模型消息（1 次工具调用 + 1 次文本），
#      而先 list() 收集句柄再读则为空 —— 与 Demo 2 的消费规则同源；
#      cause 是形如 {'type': 'toolCall', 'tool_call_id': ...} 的 dict，**不含工具名**。
# 2. 未收录（官方还有、本文件没做的）：
#    - **stream.extensions**（自定义 transformer 投影）：要自己实现 transformer 插件，
#      属于前端/协议层扩展，本课未涉及；
#    - **stream.subgraphs**（普通嵌套子图的运行流）：与 subagents 投影的差别是
#      「未命名的子图」；课案 01_langgraph 09_子图 已讲子图本身，需要时同理可读；
#    - **异步接口 `astream_events`**：本文件只用同步写法，异步用法与 v2 一致；
#    - **v2 `astream_events` / stream_mode 的对照迁移**：课案 07_流式 已覆盖 stream_mode，
#      迁移到 v3 时按本文件的投影表逐项替换即可。
# 3. 踩坑提示：
#    A. **投影单次消费**：想同时要多种投影 → 用 `interleave(...)`；各开新 stream 会让
#       图重跑（模型重复计费），只适合偶尔调试；
#    B. v3 的 `message.output.content` 可能是**内容块列表**（text / tool_call 等），
#       不是纯字符串 —— 本文件取文本时用了 `"".join(message.text)`，别直接对 content 做字符串操作；
#    C. v3 是实验性协议，升级 langgraph 后请优先回归本文件；
#    D. 工具报错要用 `ToolException`（会被转成错误消息并出现在 error/output 上）；
#       普通异常会直接中断运行，流里也就看不到「工具失败」这一事件；
#    E. 子代理要出现在 `stream.subagents` 里，必须给内层 `create_agent(name=...)` 命名；
#       不命名则是普通子图（落在 subgraphs 投影）。读内层消息同样要**内联消费**；
#    F. `ToolErrorMiddleware(on_error)` 的 `on_error` 是**两参数** `(exc, request)` ——
#       写成单参数会在工具报错时抛 TypeError（本文件实测踩过）。

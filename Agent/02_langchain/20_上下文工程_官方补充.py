# -*- coding: utf-8 -*-
r"""
LangChain 官方补充篇：上下文工程总纲（非课案内容，故不带 _jxsd 后缀）
================================================================
来源与定位：
    本文件对照 LangChain **官方文档** /oss/python/langchain/context-engineering.mdx
    （概念篇 concepts/context.mdx），补上课案缺的那张「总纲图」。

官方第一章就把话挑明了：
    「When agents fail, it's usually because the LLM call inside the agent took the
      wrong action... LLMs fail for one of two reasons: ① 模型能力不够；② **该给的上下文没给**。
      More often than not - it's actually the second reason.」
    所以官方把「提供正确信息与工具、且格式正确」称为 AI 工程师的第一职责。

官方给的三类控制 × 三种数据源（本文件的核心索引表）：

    控制类型（官方叫法）        管什么                                瞬时/持久
    --------------------------  ------------------------------------  ----------
    Model Context               单次模型调用看到什么（提示词/消息/工具/  瞬时
                                模型/响应格式）
    Tool Context                工具能读到与写出什么（state/store/context）持久
    Life-cycle Context          模型调用与工具调用**之间**发生什么      持久
                                （摘要、护栏、日志…）

    数据源                      别名              作用域        本仓库对应演示
    --------------------------  ----------------  ------------  --------------------------
    Runtime Context             静态配置          会话级        16_测试与护栏 Demo 4
    State                       短期记忆          会话级        课案 05/06 + 12_记忆 补充篇
    Store                       长期记忆          跨会话        课案 05/06、deepagents 11_记忆

机制：官方原话「LangChain middleware is the mechanism under the hood」——
所有上下文控制最终都落在中间件钩子上（课案 10_中间件_钩子 已打底）。

**本文件只实现课案与既有补充篇尚未覆盖的 4 格**（其余格子给出索引，不重复造）：

    # | 格子 | 本文件 | 已覆盖的对照
    1 | Model × Tools：按调用方动态裁剪工具集         | Demo 1 | 课案只讲了静态工具列表
    2 | Model × Response Format：按状态动态切换输出格式 | Demo 2 | 课案 08 讲的是固定 schema
    3 | Tool × State/Store：工具读三种数据源、写长期记忆 | Demo 3 | 16 补充篇只读了 context
    4 | Life-cycle：模型调用的耗时与工具调用计数（可观测）| Demo 4 | 摘要/护栏见 12、16 补充篇

⚠️ 本文件需要真实模型。

缺口表对应：`Agent/官方文档缺口对照.md` 的 **LangChain 第 7 项**（上下文工程总纲）。

运行方式（项目根目录下）：
    uv run Agent/02_langchain/20_上下文工程_官方补充.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import time

from langchain.agents import create_agent
from langchain.agents.middleware import after_model, before_model, wrap_model_call
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime, tool
from langgraph.graph import MessagesState
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field

from config import settings

model = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


def final_text(result: dict) -> str:
    return str(result["messages"][-1].content)


# ================================================================
# Demo 1：Model Context × Tools —— 按调用方动态裁剪工具集
# ================================================================
# 官方 Model Context 一节把 "Tools" 单列为一项可控内容：**给模型看哪些工具**是可调的。
# 现实需求：同一个 agent 服务不同套餐的用户 —— 免费用户不该看到「删除」这类危险工具。
# 做法：在 wrap_model_call 里用 request.override(tools=...) 换一份工具列表。
# 好处：工具根本没进提示词，模型**不可能**调用它（比"调了再拒绝"更安全）。
class CallerInfo(BaseModel):
    """本次调用的上下文：谁在调、什么套餐。

    注意：不管用 dataclass 还是 pydantic 模型，只要 invoke 时传了 context，
    langgraph 序列化都会打 `PydanticSerializationUnexpectedValue` 告警
    （实测两种写法都有，属无害噪音 —— 16_测试与护栏_官方补充.py 里记录过同一现象）。
    """

    user_id: str
    # 默认给最低权限（fail-closed）：拿不到身份时不该默认拿到高权限
    plan: str = "free"     # "free" / "pro"

# 记录模型每次调用实际看到的工具名，用来证明裁剪生效
seen_tools: list[list[str]] = []


@wrap_model_call
def limit_tools_by_plan(request, handler):
    """按套餐裁剪工具集：免费用户看不到 delete_record。"""
    # 取不到 context（未注入）或没声明 plan 时**按最低权限处理** ——
    # 安全相关的默认值必须 fail-closed：宁可少给工具，也不能默认放行。
    plan = getattr(request.runtime.context, "plan", None) or "free"
    allowed = [t for t in request.tools if not (plan == "free" and t.name == "delete_record")]
    seen_tools.append([t.name for t in allowed])
    return handler(request.override(tools=allowed))


@tool
def search_records(keyword: str) -> str:
    """按关键词查询记录。"""
    return f"查到 2 条包含 {keyword!r} 的记录"


@tool
def delete_record(record_id: str) -> str:
    """删除一条记录（危险操作）。"""
    return f"已删除 {record_id}"


def demo_1_dynamic_tools() -> None:
    print("=" * 70)
    print("Demo 1：Model Context × Tools —— 按套餐动态裁剪工具集")
    print("=" * 70)

    agent = create_agent(
        model=model,
        tools=[search_records, delete_record],
        middleware=[limit_tools_by_plan],
        context_schema=CallerInfo,
    )

    for plan in ("free", "pro"):
        seen_tools.clear()
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "帮我查一下 keyword=订单"}]},
            context=CallerInfo(user_id=f"u-{plan}", plan=plan),
        )
        print(f"  {plan:<5} 用户 → 模型看到的工具：{seen_tools[-1] if seen_tools else '（没记录）'}")
        print(f"        回答：{final_text(result)[:70]}")

    print(
        "  ↑ 免费用户的提示词里**压根没有** delete_record —— 这是「工具级上下文控制」，\n"
        "    比「让模型别调、调了再拒绝」更可靠：不可见即不可调用。\n"
        "    override 还能换 model / messages / tool_choice / response_format（见 Demo 2）。"
    )


# ================================================================
# Demo 2：Model Context × Response Format —— 按状态动态切换输出格式
# ================================================================
# 课案 08_结构化输出 讲的是「固定的 schema」；官方这里强调的是**动态**：
# 同一个 agent，在处理不同类型请求时用不同的输出格式（甚至有时不用结构化输出）。
# 场景：首轮把用户需求整理成结构化任务单，之后自由对话即可。
class TaskTicket(BaseModel):
    """结构化任务单。"""

    goal: str = Field(description="用户目标，一句话")
    steps: list[str] = Field(description="拆解出的 3 个步骤")


format_log: list[str] = []


@wrap_model_call
def dynamic_response_format(request, handler):
    """只在这轮消息数较少（=任务刚开始）时要求结构化输出。"""
    message_count = len(request.messages)
    if message_count <= 1:
        format_log.append("启用结构化输出（TaskTicket）")
        return handler(request.override(response_format=TaskTicket))
    format_log.append("自由文本（不再要求 schema）")
    return handler(request)


def demo_2_dynamic_format() -> None:
    print("\n" + "=" * 70)
    print("Demo 2：Model Context × Response Format —— 按状态切换输出格式")
    print("=" * 70)

    agent = create_agent(
        model=model,
        tools=[],
        middleware=[dynamic_response_format],
    )
    result = agent.invoke({
        "messages": [{"role": "user", "content": "我想给团队做个每周自动汇总的机器人"}],
    })
    structured = result.get("structured_response")
    print(f"  格式决策：{format_log}")
    if structured is not None:
        print(f"  结构化结果类型：{type(structured).__name__}")
        print(f"    目标：{getattr(structured, 'goal', None)}")
        print(f"    步骤：{getattr(structured, 'steps', None)}")
    else:
        print(f"  本轮没有结构化结果，拿到的是文本：{final_text(result)[:80]}")
    print(
        "  ↑ 同一个 agent，**按上下文决定这轮要不要 schema** —— 这就是 Model Context 的\n"
        "    「瞬时」性质：改的是这一次模型调用看到的东西，state 里存的仍是普通消息。\n"
        "    实用场景：首轮抽任务单、后续自由对话；或简单问题跳过结构化输出省 token。"
    )


# ================================================================
# Demo 3：Tool Context × 三种数据源 —— 工具能读到什么、能写到哪里
# ================================================================
# 官方 Tool Context 的说法：工具能读写的三处 —— Runtime Context（静态配置）、
# State（短期记忆）、Store（长期记忆）。`ToolRuntime` 对象正好把三样都带齐了
#   实测字段：state / context / config / stream_writer / tool_call_id / store /
#            tools / execution_info / server_info
# 本 Demo 演示：一次工具调用里**同时读三种数据源**，并把结果写进长期记忆（Store）。
class TicketState(MessagesState):
    """自定义状态：**必须继承 MessagesState**（它带了 add_messages reducer）。

    踩坑记录：一开始这里写成裸 TypedDict 并自己声明 `messages: list`，
    结果丢掉了 reducer —— 消息被后续节点覆盖，agent 的 while 循环判不出终止条件，
    整个运行**无限调用模型**（实测卡死 10 分钟没有任何输出）。
    正确姿势就是继承 MessagesState，再加自己的字段。
    """

    ticket_no: str          # 自定义状态字段：票据号


@tool
def inspect_context(runtime: ToolRuntime) -> str:
    """把工具能看到的三类上下文都读出来（演示 Tool Context 的数据源）。"""
    # ① Runtime Context：本次运行的静态配置（谁在调、什么环境）
    user = getattr(runtime.context, "user_id", "（未注入）")
    # ② State：本次会话的短期记忆（含自定义字段）
    ticket = runtime.state.get("ticket_no", "（state 里没有 ticket_no）")
    message_count = len(runtime.state.get("messages", []))
    # ③ Store：跨会话的长期记忆（读写都在这里）
    namespace = ("users", user)
    previous = runtime.store.get(namespace, "last_seen") if runtime.store else None
    runtime.store.put(namespace, "last_seen", {"note": "刚刚查询过上下文"}) if runtime.store else None
    return (
        f"runtime.context.user_id={user}；"
        f"state.ticket_no={ticket}，state 里已有 {message_count} 条消息；"
        f"store 里的历史记录={previous.value if previous else '（首次）'}"
    )


def demo_3_tool_context_sources() -> None:
    print("\n" + "=" * 70)
    print("Demo 3：Tool Context —— 工具同时读 Runtime Context / State / Store")
    print("=" * 70)

    store = InMemoryStore()
    agent = create_agent(
        model=model,
        tools=[inspect_context],
        store=store,                       # 长期记忆要显式传进去
        state_schema=TicketState,          # 自定义状态字段：票据号
        context_schema=CallerInfo,
    )
    config = {"configurable": {"thread_id": "tool-context-demo"}}

    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "先看看你现在能拿到哪些上下文信息"}],
            "ticket_no": "T-2026-0917",
        },
        config,
        context=CallerInfo(user_id="u-1001", plan="pro"),
    )
    for message in result["messages"]:
        if message.type == "tool":
            print(f"  工具返回：{message.content}")
    print(f"  state 里的 ticket_no：{result.get('ticket_no')}")

    # 再跑一次：这次 Store 里已经有上次写进去的记录了（跨会话/跨轮次）
    result2 = agent.invoke(
        {"messages": [{"role": "user", "content": "再查一次上下文"}]},
        {"configurable": {"thread_id": "tool-context-demo-2"}},   # 换会话，但 Store 共享
        context=CallerInfo(user_id="u-1001", plan="pro"),
    )
    for message in result2["messages"]:
        if message.type == "tool":
            print(f"  第二次（换了 thread_id）工具返回：{message.content}")
    print(
        "  ↑ 一次工具调用把三种数据源都用上了：\n"
        "    · runtime.context —— 静态配置（会话级，由 invoke 注入）；\n"
        "    · runtime.state  —— 短期记忆（会话级，含自定义字段 ticket_no）；\n"
        "    · runtime.store  —— 长期记忆（跨会话：第二次换了 thread_id 仍读得到）。\n"
        "    这三者的边界画清楚，『该把数据放哪』就不再是拍脑袋。"
    )


# ================================================================
# Demo 4：Life-cycle Context —— 模型调用之间发生了什么（最小可观测实现）
# ================================================================
# 官方的第三类上下文：模型调用与工具调用**之间**发生的事情 —— 摘要、护栏、日志…
# 课案与补充篇已有「摘要」（12_记忆 Demo 1）和「护栏」（16_测试与护栏 Demo 3），
# 这里补最小可观测实现：用 before_model / after_model 钩子统计每次调用的耗时与
# 该轮产生的工具调用数 —— 生产排障（"为什么这次回答特别慢"）的第一手数据。
lifecycle_stats: dict[str, float] = {"calls": 0, "elapsed": 0.0, "tool_calls": 0}


@before_model
def start_timer(state, runtime):
    lifecycle_stats["_t0"] = time.time()


@after_model
def record_after_model(state, runtime):
    lifecycle_stats["calls"] += 1
    lifecycle_stats["elapsed"] += time.time() - lifecycle_stats.get("_t0", time.time())
    # 这一轮模型消息里请求了几个工具调用（按消息结构直接数，不额外调模型）
    last = state["messages"][-1] if state.get("messages") else None
    lifecycle_stats["tool_calls"] += len(getattr(last, "tool_calls", None) or [])


def demo_4_lifecycle_observability() -> None:
    print("\n" + "=" * 70)
    print("Demo 4：Life-cycle Context —— 用钩子做最小可观测")
    print("=" * 70)

    agent = create_agent(
        model=model,
        tools=[search_records],
        middleware=[start_timer, record_after_model],
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "查一下 keyword=发票 的记录"}]})
    print(f"  回答：{final_text(result)[:80]}")
    print(f"  本次运行统计：模型调用 {int(lifecycle_stats['calls'])} 次，"
          f"累计耗时 {lifecycle_stats['elapsed']:.2f} 秒，"
          f"模型请求的工具调用 {int(lifecycle_stats['tool_calls'])} 个")
    print(
        "  ↑ 这就是官方说的 Life-cycle Context：钩子挂在模型/工具调用之间，\n"
        "    改的是**持久**的东西（日志、指标、状态），而不是单次提示词。\n"
        "    生产里把这两个钩子换成上报 Langfuse/OpenTelemetry 就是全链路追踪的起点。"
    )


if __name__ == "__main__":
    demo_1_dynamic_tools()
    demo_2_dynamic_format()
    demo_3_tool_context_sources()
    demo_4_lifecycle_observability()
    print("\n全部 Demo 执行完毕。")


# ================================================================
# 实测结论 / 覆盖索引 / 踩坑提示
# ================================================================
# 1. 官方框架（context-engineering.mdx）：
#    - 三类控制：Model Context（瞬时）/ Tool Context（持久）/ Life-cycle Context（持久）；
#    - 三种数据源：Runtime Context（静态配置）/ State（短期记忆）/ Store（长期记忆）；
#    - 机制：LangChain middleware —— 所有控制最终都挂在中间件钩子上。
# 2. 本仓库覆盖索引（照着查即可，不必重看文档）：
#    Model × 提示词      → 课案 10_中间件_钩子_jxsd（@dynamic_prompt）
#    Model × 消息        → 12_记忆_持久化与中断进阶_官方补充（trim_messages / RemoveMessage）
#    Model × 工具        → 本文件 Demo 1（按调用方裁剪工具集）
#    Model × 模型        → 11_内置中间件_官方补充（ModelFallbackMiddleware）
#    Model × 输出格式    → 本文件 Demo 2 + 课案 08_结构化输出
#    Tool  × 数据源      → 本文件 Demo 3 + 16_测试与护栏 Demo 4（ToolRuntime.context）
#    Life-cycle × 摘要   → 12_记忆 补充篇 Demo 1（summarize 节点）
#    Life-cycle × 护栏   → 16_测试与护栏 Demo 3（确定性护栏短路）
#    Life-cycle × 可观测 → 本文件 Demo 4 + 课案 06_langfuse
# 3. 实测结论（本机，langchain 1.4.0）：
#    - `ModelRequest.override` 支持 model / system_message（system_prompt 已废弃）/
#      messages / tool_choice / tools / response_format / model_settings / state；
#    - `ToolRuntime` 实测字段：state / context / config / stream_writer / tool_call_id /
#      store / tools / execution_info / server_info；
#    - Demo 1：免费用户模型只看到 ['search_records']，付费用户看到两个工具 —— 裁剪生效；
#    - Demo 2：结构化输出生效，拿到 TaskTicket(goal=…, steps=[3 条])；
#    - Demo 3：一次工具调用读到三种数据源 —— context.user_id=u-1001、
#      state.ticket_no=T-2026-0917、store 首次为空；**换 thread_id 后再读，
#      store 里已能取到上次写入的 {'note': '刚刚查询过上下文'}（跨会话长期记忆生效）**；
#    - Demo 4：钩子统计出「模型调用 3 次 / 累计 72.14 秒 / 请求工具调用 2 个」。
# 4. 踩坑提示：
#    A. **自定义状态 schema 必须继承 MessagesState**，别用裸 TypedDict 自己声明
#       `messages: list` —— 丢掉 add_messages reducer 后消息会被覆盖，
#       agent 循环判不出终止条件，实测**无限调用模型卡死**（本文件 Demo 3 踩过）；
#    B. 动态裁剪工具集要在 **wrap_model_call** 里做（改 request），不要用
#       before_model 去改 state —— 前者改的是「这次调用看到什么」（瞬时），后者改的是会话状态；
#    C. `request.override(...)` 是**不可变**的：它返回新请求，原 request 不变，
#       记得把返回值交给 handler；
#    D. `response_format` 动态切换后，结构化结果落在 `result["structured_response"]`；
#       某轮不设 schema 时该键可能不存在 —— 取值要判空（本文件 Demo 2 就是这写法）；
#    E. 工具里访问 `runtime.store` 前先确认 create_agent 传了 `store=`；
#       没传时 store 为 None（本文件做了 None 兜底）；
#    F. `before_model` / `after_model` 的返回值：不打算改状态就**别返回 dict**，
#       返回 None 表示只做副作用（打点、日志）。

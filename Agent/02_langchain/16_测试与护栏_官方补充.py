# -*- coding: utf-8 -*-
r"""
LangChain 官方补充篇：测试与护栏（非课案内容，故不带 _jxsd 后缀）
================================================================
来源与定位：
    本文件对照 LangChain **官方文档**（/oss/python/langchain）补课案完全空白的
    「工程化交付」维度。官方出处：
      - test/index.mdx、test/unit-testing.mdx、test/integration-testing.mdx、test/evals.mdx
        → 怎么验证 Agent 是对的（本文件 Demo 1/2）
      - guardrails.mdx → 确定性护栏 vs 模型护栏（本文件 Demo 3）
      - runtime.mdx（+ tools.mdx 的 Access context 一节）→ Runtime Context 依赖注入（Demo 4）

课案 02_langchain 覆盖了「把 Agent 跑起来」的全套零件（模型/消息/智能体/工具/记忆/
流式/结构化输出/人工审核/中间件/多 Agent）。官方文档里还缺的 12 块，按学习价值排序：

    # | 缺口 | 官方出处 | 优先级
    1 | 测试与评估（单测/集成/轨迹评估）        | test/*.mdx | 高 ← 本文件 Demo 1/2
    2 | RAG / 语义检索                          | knowledge-base.mdx | 高（要 embeddings）
    3 | Runtime Context 与依赖注入              | runtime.mdx | 高 ← 本文件 Demo 4
    4 | Skills 渐进披露（多 Agent 第 5 种模式）  | multi-agent/skills.mdx | 高（要模型）
    5 | LangGraph 自定义工作流                   | multi-agent/custom-workflow.mdx | 高（要模型）
    6 | 事件流 v3（stream_events 类型化投影）    | event-streaming.mdx | 中（要模型）
    7 | 上下文工程总纲                          | context-engineering.mdx | 中（要模型）
    8 | Guardrails 安全护栏                      | guardrails.mdx | 中 ← 本文件 Demo 3
    9 | MCP 进阶（连接生命周期/认证/Elicitation）| mcp/connections.mdx | 中（要服务端）
    10| 模型配置进阶（多模态/限流/token 用量）   | models.mdx | 中（部分要视觉模型）
    11| 可观测与可视化调试（LangSmith/Studio）   | observability.mdx | 中低（要外部账号）
    12| Deep Agents harness 组装教程             | deep-agent-from-scratch.mdx | 中低

    其余缺口的完整对照表见 Agent/官方文档缺口对照.md。

⚠️ 一个重要发现（写代码前必须知道）：
    官方**新文档全站搜索 "LCEL" 零命中** —— 课案 15_管道.py 讲的 LCEL 已不是官方主线，
    官方「编排」的对应物是 LangGraph 的 StateGraph（multi-agent/custom-workflow.mdx：
    把 create_agent 产物当节点，与确定性步骤混编）。本文件不展开，先记录在案。

为什么本文件全离线（0 次真实模型调用）：
    测试与护栏这两件事的价值恰恰在于**可重复、可断言**——依赖真模型反而测不稳。
    官方单测文档本身推荐的 `GenericFakeChatModel` 就是零 API 的方案。

运行方式（项目根目录下）：
    uv run Agent/02_langchain/16_测试与护栏_官方补充.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from dataclasses import dataclass
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool as core_tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import PrivateAttr


# ================================================================
# 假模型工具箱：官方单测用 GenericFakeChatModel，需要「盯住模型收到什么」时用脚本模型
# ================================================================
def ai_tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    """构造一条「模型要调工具」的 AIMessage（脚本模型的一行剧本）。"""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


class ScriptedModel(ChatOpenAI):
    """按剧本依次吐消息的假模型（与 11_内置中间件_官方补充.py 里的同名类同一手法）。

    继承 ChatOpenAI、只覆写 _generate —— bind_tools / 消息校验等框架方法沿用真实现，
    唯一被替换的是「真正发 HTTP 请求」那一步，所以断网也能跑。
    _received 记录每次模型**实际收到**的消息列表（Demo 4 靠它验证注入是否生效）。
    """

    _script: list = PrivateAttr(default_factory=list)
    _cursor: int = PrivateAttr(default=0)
    _received: list = PrivateAttr(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._received.append(list(messages))
        message = self._script[self._cursor]
        self._cursor += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def make_scripted(script: list) -> ScriptedModel:
    """造一个剧本模型。api_key / base_url 传假值即可：永远不会被真正用到。"""
    model = ScriptedModel(model="scripted", api_key="offline", base_url="http://localhost:9")
    model._script = script
    return model


def extract_trajectory(result: dict) -> list[str]:
    """从结果消息里抽出「工具调用轨迹」：模型依次调了哪些工具（官方轨迹评估的核心数据）。"""
    trajectory: list[str] = []
    for message in result["messages"]:
        for call in getattr(message, "tool_calls", None) or []:
            trajectory.append(call["name"])
    return trajectory


# ================================================================
# Demo 1：官方单测模式 —— GenericFakeChatModel + 假模型跑多轮记忆
# ================================================================
# 官方 unit-testing.mdx 的核心思路：**单元测试不该依赖外部服务**。
#   - 用 GenericFakeChatModel 依次吐出预设回复（它就是个「消息列表播放器」）；
#   - 用 InMemorySaver 当 checkpointer，于是多轮对话、记忆是否保住都能断言；
#   - 全程零 API key、零网络，CI 里可以放心跑。
# 注意：GenericFakeChatModel 的 messages 是个**一次性迭代器** —— 每 invoke 一次消费一条，
#       剧本用完再 invoke 会直接 StopIteration（这是官方示例里最容易踩的小坑）。
def demo_1_fake_model_unit_test() -> None:
    print("=" * 70)
    print("Demo 1：官方单测模式 —— GenericFakeChatModel + InMemorySaver")
    print("=" * 70)

    fake = GenericFakeChatModel(
        messages=iter([
            AIMessage(content="好的，我记住了：你叫小明。"),
            AIMessage(content="你叫小明。"),
        ])
    )
    agent = create_agent(model=fake, tools=[], checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "unit-test-1"}}

    # ---- 第 1 轮 ----
    first = agent.invoke({"messages": [{"role": "user", "content": "我叫小明"}]}, config)
    assert first["messages"][-1].content == "好的，我记住了：你叫小明。", first["messages"][-1]
    print(f"  第 1 轮回复：{first['messages'][-1].content}")

    # ---- 第 2 轮（同一个 thread_id，历史应当被带上）----
    second = agent.invoke({"messages": [{"role": "user", "content": "我叫什么？"}]}, config)
    assert second["messages"][-1].content == "你叫小明。"
    # 断言「记忆生效」：第 2 轮的状态里必须还留着第 1 轮的两条消息
    contents = [str(m.content) for m in second["messages"]]
    assert "我叫小明" in contents, contents
    print(f"  第 2 轮回复：{second['messages'][-1].content}")
    print(f"  第 2 轮状态里共 {len(second['messages'])} 条消息，第 1 轮的用户消息仍在 → 记忆断言通过")
    print(
        "  ↑ 这就是官方推荐的单元测试形态：断言的是**框架行为**（记忆有没有带上、\n"
        "    顺序对不对），而不是模型说了什么漂亮话 —— 所以它永远不会 flaky。"
    )


# ================================================================
# Demo 2：轨迹断言 —— 断言 Agent「做了什么」，而不是「说了什么」
# ================================================================
# 官方 test/evals.mdx 用 agentevals 包做 trajectory match，四种模式：
#     strict（结构与顺序完全一致）/ unordered（顺序无关）
#     / subset（实际**只调**参考里的工具，不许有额外的）
#     / superset（实际**至少调齐**参考工具，允许多调）
# 本仓库没装 agentevals —— 按仓库惯例：缺包给出中文提示，同时用**本地简化版**
# 把同样的思想演示出来（真正的评估逻辑并不神秘，就是比对工具调用序列）。
def demo_2_trajectory_assertion() -> None:
    print("\n" + "=" * 70)
    print("Demo 2：轨迹断言 —— Agent 该调哪个工具、按什么顺序")
    print("=" * 70)

    try:
        import agentevals  # noqa: F401

        print("  检测到 agentevals：生产评估可用官方的 trajectory match 四模式")
    except ImportError:
        print("  （未安装 agentevals —— 装了才有官方四模式：strict/unordered/subset/superset）")
        print("  本 Demo 用本地简化版演示同样的思想；要装：uv add agentevals")

    @core_tool
    def get_weather(city: str) -> str:
        """查询指定城市的天气。"""
        return f"{city}：晴，25℃"

    # 剧本：模型发起一次工具调用，然后给出最终答复
    model = make_scripted([
        ai_tool_call("get_weather", {"city": "北京"}, "c1"),
        AIMessage(content="北京今天晴，25℃。"),
    ])
    agent = create_agent(model=model, tools=[get_weather])
    result = agent.invoke({"messages": [{"role": "user", "content": "北京天气如何？"}]})

    actual = extract_trajectory(result)
    print(f"  实际轨迹（模型依次调用的工具）：{actual}")

    # ---- 本地简化版匹配器：strict（全等）与 subset（子集）----
    def strict_match(expected: list[str], got: list[str]) -> bool:
        """顺序与内容完全一致。"""
        return expected == got

    def subset_match(expected: list[str], got: list[str]) -> bool:
        """期望的每一步都在实际序列里出现过（允许实际多调了别的工具）。"""
        remaining = list(got)
        for name in expected:
            if name not in remaining:
                return False
            remaining.remove(name)   # 每个期望项只能匹配一次
        return True

    expected = ["get_weather"]
    assert strict_match(expected, actual), (expected, actual)
    assert subset_match(expected, actual), (expected, actual)
    print(f"  strict 匹配 {expected} → 通过（顺序与内容全等）")
    print(f"  subset 匹配 {expected} → 通过（期望步骤都在实际里）")

    # ---- 反面用例：期望与实际不符时必须**能识别出来**（否则测试是假绿）----
    wrong_expected = ["search_documents"]
    assert not strict_match(wrong_expected, actual)
    assert not subset_match(wrong_expected, actual)
    print(f"  反面用例 strict/subset 匹配 {wrong_expected} → 正确判为不一致 ✔")
    print(
        "  ↑ 轨迹断言的意义：模型「换个方式问同样的问题」时，回复文本每次都不同，\n"
        "    但它**该走的流程**是稳定的 —— 把流程写成断言，才是可维护的 Agent 测试。"
    )


# ================================================================
# Demo 3：确定性护栏 —— 正则/关键词拦截，模型一次都不被调
# ================================================================
# 官方 guardrails.mdx 把护栏分两条路线：
#     确定性护栏：正则、关键词、白名单 —— 快、免费、可解释，处理「已知的坏输入」；
#     模型护栏  ：让另一个模型判安全 —— 灵活，但慢且要花钱（课案已学的 PIIMiddleware
#                 属于「确定性的模型无关护栏」，见 11_内置中间件_官方补充.py Demo 4）。
# 实现手段就是课案 10 章学过的包裹式钩子：在 wrap_model_call 里检查输入，
# 命中就**短路**（不调 handler，直接返回预设回复）—— 模型调用次数为 0 是最好的证据。
FORBIDDEN_WORDS = ("转账", "信用卡号", "身份证号")
guard_stats = {"model_calls": 0, "blocked": 0}


@wrap_model_call
def safety_guard(request, handler):
    """确定性护栏：命中违禁词就短路；否则原样放行。"""
    last_message = request.messages[-1] if request.messages else None
    text = str(getattr(last_message, "content", ""))
    if any(word in text for word in FORBIDDEN_WORDS):
        guard_stats["blocked"] += 1
        print(f"    [safety_guard] 命中违禁词，拦截（本次不调用模型）")
        # 短路：直接返回一条预设回复，handler（真正调模型的那层）根本不会被调用
        return AIMessage(content="抱歉，这个请求涉及敏感信息，我不能处理。请通过人工客服渠道办理。")
    guard_stats["model_calls"] += 1
    return handler(request)


def demo_3_deterministic_guardrail() -> None:
    print("\n" + "=" * 70)
    print("Demo 3：确定性护栏 —— 违禁词短路，模型 0 次调用")
    print("=" * 70)

    guard_stats["model_calls"] = 0
    guard_stats["blocked"] = 0
    model = make_scripted([
        AIMessage(content="好的，我来介绍一下转账的一般流程。"),   # 只有正常输入才会用到这条剧本
    ])
    agent = create_agent(model=model, tools=[], middleware=[safety_guard])

    # ---- 用例 A：正常输入 → 放行，模型被调 1 次 ----
    normal = agent.invoke({"messages": [{"role": "user", "content": "介绍一下 LangChain 是什么"}]})
    print(f"  用例 A（正常输入）→ 回复：{str(normal['messages'][-1].content)[:40]}")
    print(f"    模型调用次数：{guard_stats['model_calls']}（应该 1）")

    # ---- 用例 B：违禁输入 → 短路，模型调用次数**不增加** ----
    blocked = agent.invoke({"messages": [{"role": "user", "content": "帮我转账 100 万到这个账号"}]})
    print(f"  用例 B（含「转账」）→ 回复：{blocked['messages'][-1].content}")
    print(f"    模型调用次数：{guard_stats['model_calls']}（仍是 1 → 模型确实没被调用）")
    print(f"    拦截次数：{guard_stats['blocked']}")
    assert guard_stats["model_calls"] == 1 and guard_stats["blocked"] == 1
    print(
        "  ↑ 护栏的价值不只是「拦住了」，而是**在花钱之前就拦住**"
        "（用例 B 的模型调用为 0，累计那 1 次是用例 A 的）；\n"
        "    确定性护栏适合处理已知的坏模式，开放式风险再叠加模型护栏（官方两条路线并用）。"
    )


# ================================================================
# Demo 4：Runtime Context —— 工具怎么拿到「本次运行的上下文」
# ================================================================
# 官方 runtime.mdx 的核心：依赖注入。Runtime 对象带着五类信息：
#     context（本次运行的静态上下文：用户 ID、数据库连接…）/ store（长期记忆）
#     / stream writer（自定义流）/ 执行信息（thread_id、run_id）/ server info
# 工具里想读 context，就在签名上加 `runtime: ToolRuntime` 参数 —— 它是**保留参数**，
# 不会被当成工具入参暴露给模型（模型看不到它）。
@dataclass
class UserContext:
    """本次运行的上下文（官方叫 context_schema）：多用户/多租户的标准做法。"""

    user_id: str
    tier: str


@tool
def get_my_profile(runtime: ToolRuntime) -> str:
    """读取当前用户的档案（演示工具的依赖注入）。"""
    # runtime.context 就是 invoke(context=...) 传进来的那个对象
    return f"用户 {runtime.context.user_id}，会员等级 {runtime.context.tier}"


def demo_4_runtime_context() -> None:
    print("\n" + "=" * 70)
    print("Demo 4：Runtime Context —— 工具通过 ToolRuntime 读取注入的上下文")
    print("=" * 70)

    model = make_scripted([
        ai_tool_call("get_my_profile", {}, "c1"),
        AIMessage(content="已读取到您的档案。"),
    ])
    agent = create_agent(
        model=model,
        tools=[get_my_profile],
        context_schema=UserContext,   # 声明上下文的形状
    )

    # ---- 用例 A：正常注入 ----
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "我的档案是什么？"}]},
        context=UserContext(user_id="u-1001", tier="黄金"),
    )
    tool_outputs = [m for m in result["messages"] if m.type == "tool"]
    assert tool_outputs and "u-1001" in str(tool_outputs[0].content), tool_outputs
    print(f"  工具返回：{tool_outputs[0].content}")
    print("  ↑ 用户身份是 invoke 时注入的，工具代码里没有任何硬编码 —— 这就是依赖注入")

    # ---- 用例 B：忘了传 context 会怎样？（实测行为，防止生产里漏传）----
    model = make_scripted([
        ai_tool_call("get_my_profile", {}, "c1"),
        AIMessage(content="（这一轮不该正常完成）"),
    ])
    agent = create_agent(model=model, tools=[get_my_profile], context_schema=UserContext)
    try:
        agent.invoke({"messages": [{"role": "user", "content": "我的档案是什么？"}]})
        print("\n  用例 B（不传 context）→ 竟然跑完了（不符合预期）")
    except Exception as exc:  # noqa: BLE001
        print(f"\n  用例 B（不传 context）→ 抛 {type(exc).__name__}: {exc}")
        print(
            "  ↑ 实测行为（重要）：runtime.context 是 **None**，工具里一访问属性就 AttributeError；\n"
            "    而 ToolNode 默认只把 ToolInvocationError 转成错误消息，**其余异常一律往外抛**\n"
            "    （源码 langgraph/prebuilt/tool_node.py 的 _default_handle_tool_errors），\n"
            "    所以整个运行会中断，而不是变成一条模型能看见的错误消息。\n"
            "    想让这类异常转成消息、让模型自己补救 → 挂 ToolErrorMiddleware，\n"
            "    写法见 02_langchain/11_内置中间件_官方补充.py 的 Demo 1。"
        )


if __name__ == "__main__":
    demo_1_fake_model_unit_test()
    demo_2_trajectory_assertion()
    demo_3_deterministic_guardrail()
    demo_4_runtime_context()
    print("\n全部 Demo 执行完毕（0 次真实模型调用，离线可复现）。")


# ================================================================
# 实测结论 / 未收录清单 / 踩坑提示
# ================================================================
# 1. 实测结论：
#    - Demo 1：GenericFakeChatModel + InMemorySaver 两轮断言通过（记忆确实被带上）；
#    - Demo 2：轨迹提取 == ["get_weather"]，strict/subset 两种匹配与反面用例都符合预期；
#    - Demo 3：违禁词短路后模型调用次数保持 1（用例 A 用掉的那次），即用例 B 模型 0 调用；
#    - Demo 4：ToolRuntime 注入生效（工具拿到 u-1001/黄金）；**不传 context 时抛
#      AttributeError: 'NoneType' object has no attribute 'user_id'** —— ToolNode 的
#      _default_handle_tool_errors 只处理 ToolInvocationError，其余（**含普通 ToolException**）
#      一律 re-raise，运行中断（不是转成消息）；
#      想转消息要挂 ToolErrorMiddleware（见 11_内置中间件_官方补充.py Demo 1）。
#      另：传 context 时会打印两条 pydantic UserWarning（PydanticSerializationUnexpectedValue，
#      源自 dataclass context 的序列化探测）—— 无害噪音，不影响运行结果。
# 2. 未收录（官方还有、本文件没做的）：
#    - RAG / 语义检索（knowledge-base.mdx）：要 embeddings 接口，属应用域而非工程域；
#    - Skills 渐进披露（multi-agent/skills.mdx）与 custom-workflow：需要真实模型；
#    - 事件流 v3、上下文工程总纲、MCP 进阶（认证/Elicitation，要真实远端服务器）、
#      模型配置进阶（多模态要视觉模型）、LangSmith 可观测（要外部账号）；
#    - 完整对照表见 Agent/官方文档缺口对照.md。
# 3. 踩坑提示：
#    A. GenericFakeChatModel 的 messages 是**一次性迭代器**：剧本用尽后再 invoke 会
#       StopIteration —— 单元测试里记得按用例重建模型实例。
#    B. 工具的 `runtime: ToolRuntime` 是保留参数，不会出现在给模型的 schema 里；
#       同理别把自己的参数命名成 runtime/config（官方 tools.mdx 的保留字表）。
#    C. context 不传**不会在编译期报错**，而是在工具执行时抛 AttributeError 并**中断运行**
#       （见 Demo 4 实测）—— 生产建议在调用封装层统一注入，避免漏传。
#    D. 确定性护栏只是第一道闸：它挡不住「换个说法绕过违禁词」，开放式风险要叠模型护栏。
#    E. 轨迹断言别写成「回复文本全等」——模型措辞天然会变，那是最脆的测试；断言流程与状态。

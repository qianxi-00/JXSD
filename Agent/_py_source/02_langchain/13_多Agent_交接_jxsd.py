# -*- coding: utf-8 -*-
r"""
LangChain 多 Agent：交接（Handoff）
================================================================
课案原文（用工具控制流程，完整实现逐段保留）：

    from typing import Literal
    from langchain.agents import AgentState, create_agent
    from langchain.messages import AIMessage, ToolMessage
    from langchain.tools import tool, ToolRuntime
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import Command
    from typing_extensions import NotRequired
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import MemorySaver
    from config import setting

    # 1. 定义状态（包含当前活跃智能体的追踪器）
    class MultiAgentState(AgentState):
        active_agent: NotRequired[str]

    # 2. 创建智能体切换工具
    @tool
    def transfer_to_sales(runtime: ToolRuntime) -> Command:
        \"\"\"将对话转接给销售智能体。\"\"\"
        # 找到最后一条AI消息
        last_ai_message = next(
            msg for msg in reversed(runtime.state["messages"]) if isinstance(msg, AIMessage)
        )
        # 创建转接通知消息
        transfer_message = ToolMessage(
            content="已从客服智能体转接至销售智能体",
            tool_call_id=runtime.tool_call_id,
        )
        # 返回跳转命令
        return Command(
            goto="sales_agent",
            update={
                "active_agent": "sales_agent",
                "messages": [last_ai_message, transfer_message],
            },
            graph=Command.PARENT,
        )

    # Command 的 graph 参数有两个主要选项：
    # Command.LOCAL（默认）：goto 的目标是当前子图内的节点。
    # Command.PARENT：goto 的目标是父图内的节点。
    # create_agent 内部实际上封装了一个子图逻辑。

    （transfer_to_support 与上面完全对称，方向相反）

    # 3. 创建智能体并绑定切换工具
    sales_agent   = create_agent(model=llm, tools=[transfer_to_support], system_prompt="你是销售智能体……")
    support_agent = create_agent(model=llm, tools=[transfer_to_sales],   system_prompt="你是客服智能体……")

    # 4. 调用智能体的节点函数 / 5. 路由函数 / 6. 构建状态图（见下方代码）
    builder = StateGraph(MultiAgentState)
    builder.add_conditional_edges(START, route_initial, ["sales_agent", "support_agent"])
    builder.add_conditional_edges("sales_agent", route_after_agent, ["sales_agent", "support_agent", END])
    builder.add_conditional_edges("support_agent", route_after_agent, ["sales_agent", "support_agent", END])

    graph = builder.compile(checkpointer=MemorySaver())

--------------------------------------------------------------
handoff（交接）的原理 —— 「把对话控制权交出去」
--------------------------------------------------------------
    普通 Agent 的图只有它自己一个节点：模型 → 工具 → 模型 → …。
    多 Agent 交接要解决的是：**当前 Agent 怎么把「接下来谁说话」的决定权交出去？**

    课案用的是「工具即出口」的写法：
      1. 每个专家 Agent 手里都握着一个「转接工具」（transfer_to_xxx）；
      2. 模型判断「这问题不归我管」时，就会调用这个工具；
      3. 这个工具不返回字符串，而是返回一个 **Command** —— LangGraph 的控制指令：
         goto="sales_agent" 表示「下一站去 sales_agent 节点」；
      4. 关键在 `graph=Command.PARENT`：
         专家 Agent 自己是一张**子图**（create_agent 内部封装），
         子图的节点里没有 sales_agent / support_agent，
         所以必须声明「这个 goto 是给**父图**看的」，
         否则 LangGraph 会在子图里找不到目标而报错。

    Command 的 graph 参数两个取值（课案注释原文）：

    # | 取值            | goto 的目标位置 | 什么时候用                                   |
    # |-----------------|----------------|---------------------------------------------|
    # | Command.LOCAL   | 当前子图内的节点 | 默认值；在自己这张图里跳转                     |
    # | Command.PARENT  | 父图内的节点     | 子图要把控制权交回父图（本节的交接就是这种）      |

    另外两个容易忽略的细节：
      - `update={"messages": [last_ai_message, transfer_message]}`：
        必须把「发起 tool_call 的那条 AIMessage」和「配对的 ToolMessage」**成对**写回状态。
        只写 ToolMessage 的话，消息序列里会出现一条孤儿 ToolMessage
        （没有对应的 tool_call_id 请求），下次请求供应商接口直接 400。
      - `runtime.tool_call_id`：由框架注入，就是当前这次工具调用的编号，
        用它填 ToolMessage.tool_call_id 才能配上对。

    交接和能力边界的关系：每个专家只拿「别人的转接工具」，不拿别人的业务工具，
    所以模型只能选择「自己答」或「转出去」，不会越权去调不属于自己的工具。

课案出处：Agent 课案 → 多Agent → 交接

前置条件与运行方式：
    - 根目录 `.env` 的 `API_KEY` / `BASE_URL` / `MODEL_NAME` 已填好；
    - 不需要数据库（课案这里用的是内存版 `MemorySaver`）、不需要额外依赖；
    - 首次 invoke 之后课案原文是一个 `input()` 交互循环，
      本文件在**非交互环境**会自动改跑两轮脚本化演示（技术问题 → 价格问题），
      正好覆盖「先客服、再交接给销售」这条主线；
    - 在项目根目录 `F:\ProGram\Python_Base` 下执行（否则 import 不到 `config`）：
    uv run Agent/02_langchain/13_多Agent_交接_jxsd.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from typing import Literal

from langchain.agents import AgentState, create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import NotRequired
from config import settings

# 课案写的是 ChatOpenAI(model=setting.MODEL_NAME, ...)；本项目统一用 init_chat_model + settings
llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


# ---------- 1. 定义状态（包含当前活跃智能体的追踪器） ----------
class MultiAgentState(AgentState):
    # AgentState 已经带好了 messages；这里再加一个字段记录「现在是谁在值班」。
    # NotRequired 表示这个键可以不传（第一次进来时还没有活跃 Agent）。
    active_agent: NotRequired[str]


# ---------- 2. 创建智能体切换工具 ----------
@tool
def transfer_to_sales(
    runtime: ToolRuntime,
) -> Command:
    """将对话转接给销售智能体。"""
    # 找到最后一条AI消息
    # —— 这条就是「模型决定转接」的那条 AIMessage，它带着 tool_calls，
    #    必须和下面的 ToolMessage 一起写回状态，消息序列才完整。
    last_ai_message = next(
        msg for msg in reversed(runtime.state["messages"]) if isinstance(msg, AIMessage)
    )
    # 创建转接通知消息
    transfer_message = ToolMessage(
        content="已从客服智能体转接至销售智能体",
        tool_call_id=runtime.tool_call_id,     # 框架注入的本次调用编号，必须配对
    )
    # 返回跳转命令
    return Command(
        goto="sales_agent",                    # 下一站：父图里的 sales_agent 节点
        update={
            "active_agent": "sales_agent",     # 同步更新「谁在值班」
            "messages": [last_ai_message, transfer_message],
        },
        graph=Command.PARENT,                  # 目标在父图，不在 create_agent 那张子图里
    )


# Command 的 graph 参数有两个主要选项：
# Command.LOCAL（默认）：goto 的目标是当前子图内的节点。
# Command.PARENT：goto 的目标是父图内的节点。
# create_agent 内部实际上封装了一个子图逻辑。
#
# ⚠️ 为什么必须写 Command.PARENT（不写会怎样）：
#   这段代码是写在**工具函数**里的，而工具是在专家 Agent 那张**子图**里被调用的。
#   用默认的 Command.LOCAL，LangGraph 会去子图里找名为 "sales_agent" 的节点 ——
#   子图里只有 model / tools 这些节点，找不到就报错（或静默不跳转，交接失效）。
#   加上 graph=Command.PARENT，等于告诉框架：「这个 goto 请到外面那层父图去执行」。


@tool
def transfer_to_support(
    runtime: ToolRuntime,
) -> Command:
    """将对话转接给客服智能体。"""
    # 找到最后一条AI消息
    last_ai_message = next(
        msg for msg in reversed(runtime.state["messages"]) if isinstance(msg, AIMessage)
    )
    # 创建转接通知消息
    transfer_message = ToolMessage(
        content="已从销售智能体转接至客服智能体",
        tool_call_id=runtime.tool_call_id,
    )
    # 返回跳转命令
    return Command(
        goto="support_agent",
        update={
            "active_agent": "support_agent",
            "messages": [last_ai_message, transfer_message],
        },
        graph=Command.PARENT,
    )


# ---------- 3. 创建智能体并绑定切换工具 ----------
# 注意每个 Agent 手里的工具：销售拿着「转客服」，客服拿着「转销售」——
# 谁也拿不到别人的业务工具，这就是交接模式的能力边界。
sales_agent = create_agent(
    model=llm,
    tools=[transfer_to_support],
    system_prompt="你是一名销售智能体。负责处理销售咨询。如果用户询问技术问题或售后支持，请转接给客服智能体。",
)

support_agent = create_agent(
    model=llm,
    tools=[transfer_to_sales],
    system_prompt="你是一名客服智能体。负责处理技术问题。如果用户询问价格或购买事宜，请转接给销售智能体。",
)


# ---------- 4. 定义调用智能体的节点函数 ----------
def call_sales_agent(state: MultiAgentState) -> Command:
    """调用销售智能体的节点。"""
    # 子图返回的若是一个 Command（转接工具产生的），它会一路冒泡成节点的返回值，
    # 于是父图就按 Command 里的 goto 跳走 —— 交接就是这么发生的。
    response = sales_agent.invoke(state)
    return response


def call_support_agent(state: MultiAgentState) -> Command:
    """调用客服智能体的节点。"""
    # 与 call_sales_agent 完全对称：节点函数只做转发，
    # 「跳去哪」由子图里的 Command 决定 —— 所以这里没有 if/else 路由代码。
    response = support_agent.invoke(state)
    return response


# ---------- 5. 定义路由函数（判断是结束还是继续） ----------
def route_after_agent(
    state: MultiAgentState,
) -> Literal["sales_agent", "support_agent", "__end__"]:
    """根据活跃智能体进行路由，如果智能体没有调用工具则结束。"""
    messages = state.get("messages", [])

    # 检查最后一条消息：如果是不带工具调用的AI消息，说明对话结束
    # （模型给出了纯文本答复 = 它认为这一轮说完了；如果带 tool_calls，
    #   那可能是转接工具，需要按 active_agent 再跑一轮）
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and not last_msg.tool_calls:
            return "__end__"

    # 否则路由到当前活跃的智能体
    active = state.get("active_agent", "sales_agent")
    return active if active else "sales_agent"


def route_initial(
    state: MultiAgentState,
) -> Literal["sales_agent", "support_agent"]:
    """初始路由：根据状态中的活跃智能体决定，默认销售智能体。"""
    return state.get("active_agent") or "sales_agent"


# ---------- 6. 构建状态图 ----------
builder = StateGraph(MultiAgentState)
builder.add_node("sales_agent", call_sales_agent)
builder.add_node("support_agent", call_support_agent)

# 起始边：根据初始状态进行条件路由
builder.add_conditional_edges(START, route_initial, ["sales_agent", "support_agent"])

# 智能体节点后的边：检查是否结束或跳转到另一个智能体
builder.add_conditional_edges("sales_agent", route_after_agent, ["sales_agent", "support_agent", END])
builder.add_conditional_edges("support_agent", route_after_agent, ["sales_agent", "support_agent", END])

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)
# 配置线程 ID (用于区分不同的对话会话)
config = {"configurable": {"thread_id": "user_conversation_1"}}


# ---------- 7. 便捷函数：抽出一轮的「谁值班 / 有没有发生转接」，便于观察 ----------
def describe_turn(result: dict) -> None:
    """打印本轮轨迹里的人机消息与转接痕迹。"""
    print("  本次对话内部通信：")
    for msg in result["messages"]:
        kind = type(msg).__name__
        if isinstance(msg, AIMessage) and msg.tool_calls:
            print(f"    {kind:<12} 要求调用 {[c['name'] for c in msg.tool_calls]} ← 交接动作")
        else:
            print(f"    {kind:<12} {str(msg.content)[:70]}")

    transfers = [
        msg for msg in result["messages"]
        if isinstance(msg, ToolMessage) and "转接" in str(msg.content)
    ]
    if transfers:
        print(f"  ✅ 发生了交接：{transfers[-1].content}")
    else:
        print("  本轮没有发生交接（当前值班的 Agent 自己处理完了）。")


if __name__ == "__main__":
    # 课案原文的第一问：技术问题 → 应当由客服接手
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "你好，我的账户登录有问题，能帮忙吗？"}]},
        config,
    )
    # 打印结果消息（课案用 msg.pretty_print()，这里用更紧凑的一行式，便于贴进报告）
    for msg in result["messages"]:
        msg.pretty_print()
    print("=== 多轮对话已启动 (输入 'exit' 或 'quit' 退出) ===")

    # 规范第 5.4 条：交互循环先探一次 stdin。
    # 本机实测坑：某些环境 isatty() 返回 True，但 input() 立刻抛 EOFError，
    # 所以只判 isatty() 不够 —— 真正的兜底是下面 while 循环外那层 try/except，
    # 一旦读到 EOFError 就把 interactive_ok 翻回 False，改跑自动演示
    # （不会卡死、不会 traceback）。非交互环境（重定向输入 / 后台运行）直接跳过交互分支。
    interactive_ok = False
    if sys.stdin.isatty():
        interactive_ok = True          # 先假定能交互，真读不到时由下面的 except 翻回来
        try:
            # ---------- 课案原文的交互循环 ----------
            while True:
                # 1. 获取用户输入
                user_input = input("\nYou: ")

                # 2. 检查退出条件
                if user_input.lower() in ["exit", "quit"]:
                    print("=== 对话结束 ===")
                    break

                # 3. 运行图
                result = graph.invoke({"messages": [{"role": "user", "content": user_input}]}, config)

                print('本次对话内部通信')
                for msg in result["messages"]:
                    print(msg)

                # 4. 打印最后一条 AI 回复
                last_msg = result["messages"][-1]
                print(f"\nAI: {last_msg.content}")
        except EOFError:
            # 标准输入被判为终端却读不到内容 —— 退回到脚本化演示
            print("\n[提示] 读不到交互输入（无可用控制台），改用自动演示。\n")
            interactive_ok = False

    if not interactive_ok:
        # ---------- 非交互环境的等价演示（规范第 5.4 条：避免 input() 卡死 / EOFError） ----------
        print("\n[非交互环境] 自动演示两轮，代替手工输入。\n")
        describe_turn(result)

        # 第二轮故意问销售问题：客服应当调用 transfer_to_sales 把对话交出去
        # 两轮用的是同一份 config（同 thread_id），所以第二轮模型能看到第一轮的历史 ——
        # 这也说明交接发生在**同一个会话**里，只是「值班的人」换了。
        print("\n--- 第二轮：用户改问价格（预期触发 交接 → sales_agent） ---")
        before = len(result["messages"])
        result = graph.invoke(
            {"messages": [{"role": "user", "content": "你们这个套餐多少钱？我想买，能给我介绍下价格吗？"}]},
            config,
        )
        # 只打印本轮新增的消息，避免把历史全刷一遍
        # （result["messages"] 是累计的完整历史，所以用 before 切片取本轮增量）
        for msg in result["messages"][before:]:
            msg.pretty_print()
        describe_turn({"messages": result["messages"][before:]})
        print("\nAI:", result["messages"][-1].content)


# ================================================================
# 实测结论 / 与本课案的差异 / 踩坑提示
# ================================================================
# 1. 实测结论：本文件在非交互环境（重定向输入）下走自动演示，
#    第一轮「账户登录有问题」由客服接住；第二轮「套餐多少钱」客服调用 transfer_to_sales，
#    控制台会打印「✅ 发生了交接：已从客服智能体转接至销售智能体」。
#    注意模型是否发动转接属于模型判断，偶发不调工具时重跑一次即可（见 README 的实测提示）。
# 2. 与课案的差异（都不是逻辑改动）：
#      - `from config import setting` → `from config import settings`；
#      - 课案是死循环 + input()，本文件加了非交互兜底（重定向输入时自动演示两轮），
#        交互终端里仍然是课案原来的循环；
#      - 课案正文里的 `sales_agent` / `support_agent`（下划线）与图节点名
#        `"sales_agent"` / `"support_agent"` 是两套名字，本文件原样保留，别改错对象。
# 3. 踩坑提示 A —— 忘了成对回填消息（本节第一大坑）：
#    `update={"messages": [last_ai_message, transfer_message]}` 里两条必须都在。
#    只写 ToolMessage 会出现「没有对应 tool_call 的孤儿 ToolMessage」，
#    下一次请求供应商接口直接 400；只写 AIMessage 则模型永远等不到工具结果，循环卡死。
# 4. 踩坑提示 B —— `runtime.tool_call_id` 不能自己编：
#    它由框架注入，就是这次工具调用的编号；随手写 "call_1" 之类的字符串会配对失败。
# 5. 踩坑提示 C —— 路由函数里那个「最后一条是纯文本 AIMessage 就结束」的判断
#    是图能停下来的唯一出口；删掉它，两个 Agent 会互相转接直到撞上递归上限。
# 6. 踩坑提示 D —— Command 的 graph 参数写错会静默失效：
#    写成默认的 Command.LOCAL 时，goto 的目标会被拿到子图里去找，
#    结果要么报节点不存在，要么交接不发生。

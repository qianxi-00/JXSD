# -*- coding: utf-8 -*-
r"""
LangChain 内置中间件：七个开箱即用的中间件
================================================================
本节要讲什么（课案原文用 7 段独立代码列出这 7 个内置中间件，本节把每一段都跑一遍并打印结果）：

    每个 Demo 的固定结构 = 「它解决什么问题 → 关键参数什么意思 → 跑一遍看得到的现象」。
    记住一句话：中间件不是新概念，它们全都是 10_中间件_钩子_jxsd.py 里那六个钩子的
    **官方封装**（比如 ModelRetryMiddleware = wrap_model_call + 重试循环，
    TodoListMiddleware = before_agent 注入工具和提示词 + state 里的 todos 字段）。
    所以本节的重点不是「会用」，而是「知道轮子已经造好了，别自己再写一遍」。

    Demo 1：SummarizationMiddleware — 自动摘要（解决：历史太长撑爆上下文）
    Demo 2：HumanInTheLoopMiddleware — 工具执行前审批（课案："见上文人工审核"）
    Demo 3：ModelCallLimitMiddleware — 限制模型调用次数（解决：Agent 死循环烧钱）
    Demo 4：ModelRetryMiddleware — 模型失败自动重试（解决：模型接口偶发超时）
    Demo 5：ToolRetryMiddleware — 工具失败自动重试（解决：外部服务偶发不可用）
    Demo 6：TodoListMiddleware — 自动创建待办清单（解决：多步任务没有规划）
    Demo 7：ContextEditingMiddleware — 清理旧工具结果（解决：工具返回太长占满上下文）

课案总表（原样保留）：

    # | 中间件 | 运行机制与适用场景 | 触发后可见效果 | 关键参数 |
    # |---|---|---|---|
    # | SummarizationMiddleware | 每次调用模型前检查消息数或 token 数；达到阈值后，用指定模型把较早历史压缩成一条带 lc_source="summarization" 标记的摘要消息，同时保留最近上下文。适合长对话，摘要会额外调用一次模型。 | 原来的多条历史消息被"摘要消息 + 最近消息"替代，Agent 仍能继续回答。 | model、trigger、keep、summary_prompt |
    # | HumanInTheLoopMiddleware | 模型生成指定工具调用后、工具真正执行前暂停图，等待人工执行 approve、reject、edit 或 respond。适合删除、转账、发消息等有副作用的操作；必须配置 checkpointer，并用同一个 thread_id 恢复。 | 第一次调用返回 interrupt，工具尚未执行；提交审核决定后才继续运行。 | interrupt_on、description_prefix、checkpointer |
    # | ModelCallLimitMiddleware | 在每次模型调用前检查本次运行或整个 thread 的累计调用次数，防止 Agent 无限循环或成本失控。它限制的是 Agent 内部模型调用次数，不是接口 QPS 限流。 | 达到上限时可直接结束并追加限制提示，或抛出 ModelCallLimitExceededError。 | run_limit、thread_limit、exit_behavior |
    # | ModelRetryMiddleware | 包裹模型调用；遇到指定异常后按照退避策略重新调用同一个模型请求。max_retries=2 表示首次调用失败后最多再试 2 次，总尝试次数最多为 3。 | 临时超时等异常被自动吸收；后续尝试成功时 Agent 正常返回，全部失败时继续、抛错或返回自定义提示。 | max_retries、retry_on、on_failure、initial_delay、backoff_factor、max_delay、jitter |
    # | ToolRetryMiddleware | 包裹工具调用；工具抛出指定异常时重试同一次工具请求，可只作用于部分工具。适合网络查询、数据库读取等偶发失败操作，不适合对非幂等写操作盲目重试。 | 同一次工具调用会连续尝试，成功结果作为 ToolMessage 返回模型；重试耗尽后继续、抛错或返回自定义提示。 | max_retries、tools、retry_on、on_failure、退避参数 |
    # | TodoListMiddleware | 向 Agent 注入 write_todos 工具、任务规划提示词和 todos 状态；模型可创建或更新结构化待办项。适合三步以上的复杂任务，不应强迫简单问答创建待办。 | 消息中出现 write_todos 工具调用，最终状态可通过 result["todos"] 读取。 | system_prompt、tool_description |
    # | ContextEditingMiddleware | 每次模型调用前复制当前消息列表，并按策略清理旧工具结果；默认的 ClearToolUsesEdit 会把较早 ToolMessage 内容替换成占位符。它只缩短本次发给模型的上下文，不会永久删除 checkpoint 中的原始消息。 | 模型看到的旧工具结果变为 [cleared] 或自定义占位符，最近若干条工具结果保持完整。 | edits、token_count_method；策略参数 trigger、keep、clear_at_least、exclude_tools、placeholder |

一句话选型：
    上下文太长 → Summarization / ContextEditing；怕烧钱 → ModelCallLimit；
    外部服务不稳 → ModelRetry / ToolRetry；任务多步 → TodoList；
    工具有副作用 → HumanInTheLoop。

参数写法的两条通用规律（课案表格里那串参数名，看这两条就够了）：
    - 阈值类参数都是「元组 (单位, 数值)」：`trigger=("messages", 6)` 是消息条数，
      也可以写成 `("tokens", 3000)`；`keep` 同理，表示「保留最近多少」。
    - 退避类参数在 ModelRetry / ToolRetry 上完全同名同义：
      initial_delay（首次重试前等几秒）→ backoff_factor（每轮再乘几倍）
      → max_delay（封顶）→ jitter（加随机抖动，防止大量请求同时重试打垮服务）。

课案出处：Agent 课案 → 中间件 → 内置中间件

前置条件与运行方式：
    - 根目录 `.env` 的 `API_KEY` / `BASE_URL` / `MODEL_NAME` 已填好；
    - 不需要数据库、不需要额外依赖（7 个中间件都在 langchain 包内）；
    - 本文件会**连着跑 7 个 Demo**，模型调用次数较多（Demo 1 的摘要本身还要多调一次模型），
      整轮明显比别的小节慢；某个 Demo 单独失败不影响后续 Demo；
    - Demo 2 会在**当前工作目录**写出 output.txt（课案原文行为）；
    - 在项目根目录 `F:\ProGram\Python_Base` 下执行（否则 import 不到 `config`）：
    uv run Agent/02_langchain/11_内置中间件_jxsd.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import builtins
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelRequest,
    ModelRetryMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
    ToolRetryMiddleware,
    wrap_model_call,
)
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import PrivateAttr
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


# ================================================================
# Demo 1：SummarizationMiddleware —— 历史太长就自动摘要
# ================================================================
# 课案原文：
#     trigger=("messages", 6)  # 达到 6 条消息就摘要
#     keep=("messages", 2)     # 将旧消息总结为一条摘要消息，同时保留最近两条原始消息
#
# 机制：每次调用模型**之前**检查一次。阈值一到，就把「老历史」交给摘要模型压成一条，
# 并给这条消息打上 lc_source="summarization" 标记（便于程序识别）。
# 副作用：触发摘要的那一轮会**多花一次模型调用**（摘要本身也要调模型）。
def demo_1_summarization() -> None:
    print("=" * 70)
    print("Demo 1：SummarizationMiddleware —— 自动摘要")
    print("=" * 70)

    agent = create_agent(
        model=llm,
        middleware=[
            SummarizationMiddleware(
                model=llm,                  # 用哪个模型写摘要（一般用便宜的小模型）
                trigger=("messages", 6),    # 达到 6 条消息就摘要
                keep=("messages", 2),       # 旧消息压成一条摘要，同时保留最近两条原文
                                            # 为什么还要 keep：摘要会丢细节，
                                            # 最近两条往往是当前话题的上下文，丢了模型就答偏。
            )
        ],
    )

    # 课案原文的 7 条历史：6 条「人机对话」+ 1 条新提问 = 触发阈值（>6）
    # 这个条数是**故意凑出来的**：trigger=("messages", 6) 是「达到 6 条就摘要」，
    # 而 invoke 时框架会先把新提问并进来，于是本轮请求实际带 7 条 → 刚好越线 → 必触发。
    # 想验证「不触发」的样子，把 history 删掉两条再跑即可。
    history = [
        HumanMessage(content="我计划去北京三天。"),
        AIMessage(content="可以安排故宫、长城和颐和园。"),
        HumanMessage(content="第一天想去故宫。"),
        AIMessage(content="建议提前预约上午场。"),
        HumanMessage(content="第二天去八达岭长城。"),
        AIMessage(content="可以乘坐高铁到八达岭长城站。"),
        HumanMessage(content="请根据前面的讨论给我一个简短建议。"),
    ]
    print(f"送入 {len(history)} 条历史消息（超过 trigger=6，本轮会先压缩再回答）")

    result = agent.invoke({"messages": history})
    # 关键现象：返回的消息数比送进去的还多。因为「压缩」不是删消息，而是**追加**一条摘要消息，
    # 只不过后续请求不会再带原始旧消息 —— 想验证这点，可以在下面循环里看那条 lc_source 标记。
    print(f"\n返回 {len(result['messages'])} 条消息（比送进去的还多，因为摘要那条是新增的）：")
    for index, value in enumerate(result["messages"], start=1):
        # 摘要消息带 lc_source 标记，据此把它和普通消息区分开
        # （additional_kwargs 是 LangChain 给消息挂「框架侧元数据」的地方，
        #   模型看不到它，程序可以靠它识别特殊消息）
        mark = " ← 摘要消息" if value.additional_kwargs.get("lc_source") == "summarization" else ""
        print(f"  [{index}] {type(value).__name__:<12}{mark}")
        print(f"       {str(value.content)[:110]}")


# ================================================================
# Demo 2：HumanInTheLoopMiddleware —— 工具执行前审批
# ================================================================
# 课案这一段只写了「见上文人工审核」，完整实现见 09_人工审核_jxsd.py。
# 这里为了「每段都跑一遍」，做一个非交互的浓缩版：只演示 approve 一条路径，
# 三种决策（approve / reject / edit）的完整对照请看 09_人工审核_jxsd.py。
def demo_2_human_in_the_loop() -> None:
    print("\n" + "=" * 70)
    print("Demo 2：HumanInTheLoopMiddleware —— 工具执行前审批（浓缩版，完整版见 09_人工审核_jxsd.py）")
    print("=" * 70)

    @tool
    def write_file(content: str) -> str:
        """将内容写入 output.txt。"""
        Path("output.txt").write_text(content, encoding="utf-8")
        return "已写入 output.txt"

    agent = create_agent(
        model=llm,
        tools=[write_file],
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"write_file": {"allowed_decisions": ["approve", "reject"]}}
            )
        ],
        system_prompt="你是文件助手。用户要求写文件时，必须调用 write_file 工具；调用后等待人工审核。",
        # 审批必须能"暂停后恢复"，所以必须有 checkpointer
        checkpointer=MemorySaver(),
    )

    config = {"configurable": {"thread_id": "demo2-hitl"}}
    result = agent.invoke(
        {"messages": [("user", "请用 write_file 工具把「你好世界」写入 output.txt")]},
        config=config,
        version="v2",
    )

    if not result.interrupts:
        # 本机模型偶发「把 tool_call 说成文本」，此时不会产生中断 —— 给中文提示而不是报错
        # （判断依据：没有 interrupt 说明 HITL 中间件根本没被触发，而不是审核逻辑出错）
        print("  本轮模型没有发起工具调用（本机模型偶发行为），未产生中断。")
        print("  模型输出：", str(result.value["messages"][-1].content)[:80])
        print("  重新运行本 Demo 即可看到中断效果。")
        return

    # result.value 才是「真正的状态字典」：result 本身是 version="v2" 的 GraphOutput 包装，
    # value 里装的才是 messages —— 这是 LangChain/LangGraph 1.x 的新返回结构，老教程里没有。
    print("  第一次 invoke 返回中断：", result.interrupts[0].value["action_requests"])
    print("  工具尚未执行 → approval 决定：approve")
    # Command(resume=...) 就是「带着人工决定，从检查点原地复活」；
    # 必须复用同一个 config（thread_id），否则找不到那个被冻结的检查点。
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config=config,
        version="v2",
    )
    print("  恢复后 AI：", result.value["messages"][-1].content)
    print("  output.txt 是否生成：", Path("output.txt").exists())


# ================================================================
# Demo 3：ModelCallLimitMiddleware —— 限制模型调用次数
# ================================================================
def demo_3_model_call_limit() -> None:
    print("\n" + "=" * 70)
    print("Demo 3：ModelCallLimitMiddleware —— 限制模型调用次数")
    print("=" * 70)

    agent = create_agent(
        model=llm,
        middleware=[
            ModelCallLimitMiddleware(
                thread_limit=1     # 整个 thread 只允许调用 1 次模型
                # run_limit 是「单次 invoke 内」的限额，thread_limit 是「跨多轮对话」的累计限额
            )
        ],
        checkpointer=MemorySaver(),
    )
    config = {"configurable": {"thread_id": "limit-demo"}}
    # 两次 invoke 用的是**同一个** config —— 这正是 Demo 3 的全部要点：
    # thread_limit 统计的是这个 thread 的累计值，换个 thread_id 就又能调一次模型了。

    first = agent.invoke(
        {"messages": [{"role": "user", "content": "用一句话介绍 LangChain"}]},
        config=config,
    )
    print("第一次:", first["messages"][-1].content)

    # 同一个 thread 已调用过一次模型，第二次在调用模型前直接结束
    # （所以返回值里没有新的 AIMessage，只有中间件追加的限制提示 —— 这不算 traceback，是设计行为）
    second = agent.invoke(
        {"messages": [{"role": "user", "content": "再说一句"}]},
        config=config,
    )
    print("第二次:", second["messages"][-1].content)
    print("  ↑ 第二次没有真的问模型，而是被中间件拦下并追加了限制提示（exit_behavior 默认 'end'）")


# ================================================================
# Demo 4：ModelRetryMiddleware —— 模型失败自动重试
# ================================================================
# 课案原文：定义一个「前两次必定超时」的模型子类，用来观察重试过程。
# 说明：_generate 是 LangChain 聊天模型的同步底层实现，invoke 最终会走到这里，
#       因此在它里面抛异常 = 模拟一次模型调用失败。
class FlakyChatOpenAI(ChatOpenAI):
    """前两次调用固定超时，第三次才请求真实模型。"""

    _attempts: int = PrivateAttr(default=0)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._attempts += 1
        print(f"  模型尝试第 {self._attempts} 次")
        if self._attempts < 3:
            raise TimeoutError("模拟模型超时")
        return super()._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )


def demo_4_model_retry() -> None:
    print("\n" + "=" * 70)
    print("Demo 4：ModelRetryMiddleware —— 模型失败自动重试")
    print("=" * 70)

    model = FlakyChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )
    agent = create_agent(
        model=model,
        middleware=[
            ModelRetryMiddleware(
                max_retries=2,          # 初始调用失败后，最多再重试几次
                retry_on=(TimeoutError,),  # 哪些异常需要重试
                initial_delay=0.0,      # 第一次重试前等待多少秒
                backoff_factor=0.0,     # 每次重试的等待时间增长倍数
            )
        ],
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "回复：重试成功"}]})
    print("最终回答:", result["messages"][-1].content)
    # initial_delay / backoff_factor 都设成 0，是为了让教学时不必等退避时间；
    # 生产环境恰恰**不能**设 0 —— 对方服务正超时的时候立刻重试会把它打死（雪崩），
    # 标准做法是 initial_delay=1.0 起步、backoff_factor=2.0 翻倍、配 jitter 打散。
    print("  ↑ 第 1、2 次抛 TimeoutError 被中间件吞掉，第 3 次成功 → 总尝试 3 次 = 1 + max_retries")


# ================================================================
# Demo 5：ToolRetryMiddleware —— 工具失败自动重试
# ================================================================
# 注意课案用的是模块级变量 + global：重试的是**同一次工具调用**，
# 所以计数器会跨重试累加（第 1、2 次抛异常，第 3 次返回真结果）。
attempts = 0


@tool
def get_weather(city: str) -> str:
    """查询城市天气。"""
    global attempts
    attempts += 1
    print(f"  工具尝试第 {attempts} 次")
    if attempts < 3:
        raise ConnectionError("模拟天气服务暂时不可用")
    return f"{city}：晴，26℃"


def demo_5_tool_retry() -> None:
    global attempts
    print("\n" + "=" * 70)
    print("Demo 5：ToolRetryMiddleware —— 工具失败自动重试")
    print("=" * 70)

    attempts = 0   # 重置计数器，保证本 Demo 可重复运行
    agent = create_agent(
        model=llm,
        tools=[get_weather],
        middleware=[
            ToolRetryMiddleware(
                max_retries=2,
                # tools 是**白名单**：不写就作用于所有工具。
                # 生产里通常要写 —— 只给「幂等的读操作」开重试，写操作不能碰（见文末踩坑 C）。
                tools=["get_weather"],        # 只对指定工具生效（按工具名过滤）
                retry_on=(ConnectionError,),  # 只重试这类异常
                initial_delay=0.0,
                backoff_factor=0.0,
            )
        ],
        system_prompt="查询天气时必须且只调用一次 get_weather 工具。",
    )

    result = agent.invoke({"messages": [{"role": "user", "content": "北京天气如何"}]})

    if attempts == 0:
        # attempts 还是 0 = 模型压根没发起工具调用，重试链一次都没进。
        # 这是本机模型的偶发行为，跟中间件无关；下面的 next() 也会因此抛 StopIteration，
        # 所以必须在这里提前 return（少了这句，脚本会在最后一行崩掉）。
        print("  本轮模型没有发起工具调用（本机模型偶发行为），重试逻辑未触发，请重新运行。")
        print("  模型输出：", str(result["messages"][-1].content)[:80])
        return

    # 从消息列表里挑出 get_weather 那条 ToolMessage：
    # 中间件重试了 3 次，但**只有成功那一次**会产生 ToolMessage —— 前两次的异常被吞掉了，
    # 消息列表里不会留下痕迹。这正是「包裹式钩子对模型透明」的直观证据。
    tool_result = next(
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.name == "get_weather"
    )
    # 打印出来的是「第 3 次成功」的结果：前两次的 ConnectionError 在消息列表里查不到任何痕迹，
    # 因为包裹式钩子把异常吞在了模型看不到的那一层。
    print("工具结果:", tool_result.content)
    print("  ↑ 前两次 ConnectionError 被中间件吸收并重试，第 3 次成功，模型只看到成功结果")


# ================================================================
# Demo 6：TodoListMiddleware —— 自动创建待办清单
# ================================================================
# 机制：这个中间件会「偷偷」给 Agent 注入一个 write_todos 工具 + 一段规划提示词，
#       模型调它就把结构化待办写进 state 的 todos 字段。
#       （所以模型眼里它和普通工具没区别，区别是它由中间件自带、不占你的 tools 列表。）
def demo_6_todo_list() -> None:
    print("\n" + "=" * 70)
    print("Demo 6：TodoListMiddleware —— 自动创建待办清单")
    print("=" * 70)

    agent = create_agent(
        model=llm,
        middleware=[TodoListMiddleware()]
        # 注意 tools=[] 是空的：write_todos 这个工具是中间件**自己注入**的，
        # 不需要（也不能）写进 tools 列表。这正是中间件的价值 —— 能力以插件形式加进来。
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "请规划：调研竞品、设计接口、编写测试。",
                }
            ]
        }
    )

    todos = result.get("todos") or []
    # 用 .get 而不是 result["todos"]：todos 这个键只有在 TodoListMiddleware 启用、
    # 且模型真的调了 write_todos 之后才会出现在 state 里，直接下标会 KeyError。
    if not todos:
        print("  本轮模型没有调用 write_todos（本机模型偶发行为），请重新运行。")
        print("  模型输出：", str(result["messages"][-1].content)[:80])
        return

    for index, todo in enumerate(todos, start=1):
        # 课案只打印 content；这里额外把 status 也带上（pending / in_progress / completed），
        # 因为「状态」才是待办清单区别于普通文本的地方。
        # 本机模型可能把多行内容塞进一条，这里截断显示。
        content = str(todo["content"]).replace("\n", " ")[:70]
        print(f"  {index}. [{todo['status']}] {content}")

    print(f"\n  result['todos'] 共 {len(todos)} 条（这就是「结构化待办」——程序可直接读）")


# ================================================================
# Demo 7：ContextEditingMiddleware —— 清理旧工具结果
# ================================================================
# 课案的注释：「第一个 wrap_model_call 是最外层，清理后再进入 show_model_context」。
# 下面用 @wrap_model_call 把「真正发给模型的消息」打印出来，让清理效果看得见：
#   - 清理只影响 **本次请求**（request.messages）；
#   - state["messages"]（完整状态）保持原样，历史不会被永久删除。
@wrap_model_call
def show_model_context(request: ModelRequest, handler):
    """课案注释里提到的 show_model_context：打印本次请求实际带上的上下文。

    它是一个**包裹式钩子**（不是节点）：被 @wrap_model_call 装饰后就成了中间件对象，
    可以直接放进 create_agent(middleware=[...]) —— 这正是 10 节讲的用法。
    这里把它和 ContextEditingMiddleware 串在一起，是为了让「清理」这件看不见的事变得可见。
    """
    print(f"  [show_model_context] 本次发给模型 {len(request.messages)} 条消息：")
    # 注意打印的是 request.messages，也就是「即将发给供应商接口的那一份」，
    # 和 state["messages"] 不是同一个对象 —— 这是本 Demo 的核心区别所在。
    for index, message in enumerate(request.messages, start=1):
        body = str(message.content).replace("\n", " ")[:46]
        print(f"      [{index}] {type(message).__name__:<12} {body!r}")
    # 洋葱模型里「往里走一层」：调 handler 才真正发起模型请求。
    # 想演示短路，把这一行换成 return AIMessage(...) 就行（不发请求直接给假答复）。
    return handler(request)


def demo_7_context_editing() -> None:
    print("\n" + "=" * 70)
    print("Demo 7：ContextEditingMiddleware —— 清理旧工具结果")
    print("=" * 70)

    agent = create_agent(
        model=llm,
        middleware=[
            # 第一个 wrap_model_call 是最外层，清理后再进入 show_model_context
            # 为什么顺序是这样：包裹式中间件是洋葱嵌套，列表**第一个在最外层**，
            # 所以「清理上下文」先执行，再轮到 show_model_context 打印 ——
            # 打印出来的就是清理后的结果，否则这个演示看不到任何效果。
            ContextEditingMiddleware(
                edits=[
                    ClearToolUsesEdit(
                        trigger=1,  # 当整个对话上下文超过约 1 个 token 时，开始清理旧工具结果。
                                    # —— 课案原文的极端值，等于「总是清理」，只为演示效果
                        keep=1,     # 最近 1 个工具结果永远保留，不清理
                                    # （保留最近的是为了不让模型丢失「刚查到的东西」）
                    )
                ]
            ),
            show_model_context,
        ],
    )

    # 手工拼一段「3 次工具调用 + 3 条工具结果」的历史：
    # 为什么是 3 次而不是 1 次 —— 因为 keep=1，只有存在**多条**旧工具结果时才看得出
    # 「前面的被清、最后一条留着」这个对比效果。全是手工构造的假数据，不会真的联网查天气。
    history = [
        HumanMessage(content="查询北京天气"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "get_weather", "args": {"city": "北京"}, "id": "call_1", "type": "tool_call"}
            ],
        ),
        ToolMessage(content="北京：晴，25℃", name="get_weather", tool_call_id="call_1"),
        HumanMessage(content="查询上海天气"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "get_weather", "args": {"city": "上海"}, "id": "call_2", "type": "tool_call"}
            ],
        ),
        ToolMessage(content="上海：小雨，22℃", name="get_weather", tool_call_id="call_2"),
        HumanMessage(content="查询深圳天气"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "get_weather", "args": {"city": "深圳"}, "id": "call_3", "type": "tool_call"}
            ],
        ),
        ToolMessage(content="深圳：多云，28℃", name="get_weather", tool_call_id="call_3"),
        HumanMessage(content="只根据已有结果总结三地天气，不要再次调用工具。"),
    ]

    result = agent.invoke({"messages": history})

    print("\n  ---- 完整状态 result['messages']（未被永久删除） ----")
    # 上下两段对照着看，Demo 7 的结论就出来了：
    #   上面 show_model_context 打印的「本次发给模型 N 条」里，旧工具结果已变成占位符；
    #   下面这份 result['messages']（也就是 state / checkpoint 里的原始消息）**一条没少**。
    # 一句话：ContextEditing 是「每次请求时的投影裁剪」，不是「删数据」。
    for index, value in enumerate(result["messages"], start=1):
        body = str(value.content).replace("\n", " ")[:46]
        print(f"      [{index}] {type(value).__name__:<12} {body!r}")


if __name__ == "__main__":
    # 7 个 Demo 刻意「顺序执行、互不 try/except 包住」：
    # 每个 Demo 都是独立的 agent + 独立的 thread_id，互不共享状态；
    # 只有 Demo 1 和 Demo 3 各自需要 checkpointer，且都在 Demo 内部自己建。
    demo_1_summarization()
    demo_2_human_in_the_loop()
    demo_3_model_call_limit()
    demo_4_model_retry()
    demo_5_tool_retry()
    demo_6_todo_list()
    demo_7_context_editing()
    print("\n全部 Demo 执行完毕。")


# ================================================================
# 实测结论 / 与本课案的差异 / 踩坑提示
# ================================================================
# 1. 实测结论：7 个 Demo 在本机连续跑通。其中「模型愿不愿意按剧本演」是不确定的，
#    所以 Demo 2 / 5 / 6 都带了一句「本轮模型没有发起工具调用（本机模型偶发行为）」
#    的判断 —— 这不是中间件失效，重跑一次即可（原因见 Agent/README.md「关于本机模型的实测提示」）。
#    可以无条件复现的是 Demo 4（自己造的 FlakyChatOpenAI 必抛超时）和 Demo 7（历史是手工拼的）。
# 2. 与课案的差异（都不是逻辑改动）：
#      - Demo 1 的 model：课案是 `model=model`（新建的 ChatOpenAI）；
#        本文件传的是同一个 llm 对象，语义相同（摘要模型一般建议换更便宜的小模型）。
#      - Demo 2：课案原文只有一句「见上文人工审核」，本文件补了一个**只走 approve** 的
#        浓缩版，三种决策的完整对照在 09_人工审核_jxsd.py。
#        另：课案表格里 HITL 的决策有 approve / reject / edit / **respond** 四种，
#        本 Demo 的 allowed_decisions 只声明了 approve / reject，所以 respond 在本文件里演示不到。
#      - Demo 5：课案用的是模块级变量 attempts + global，本文件保留同样写法，
#        但在 Demo 开头重置了一次计数器，保证脚本可重复运行。
#      - Demo 7：课案只有 ContextEditingMiddleware 一个中间件（那段注释
#        「第一个 wrap_model_call 是最外层」在课案里其实**没有对应的第二个中间件**）；
#        本文件补了一个 show_model_context（见该函数上方的注释），
#        把「清理前 vs 清理后到底发了什么给模型」变成肉眼可见。
#      - 全部 Demo 的 `from config import setting` → `from config import settings`。
# 3. 踩坑提示 A —— Demo 3 的 thread_limit=1 必须配 checkpointer：
#    「跨轮累计调用次数」这件事本身就是状态，没 checkpointer 就没有 thread 可累计。
# 4. 踩坑提示 B —— Demo 7 的 trigger=1 是**故意的极端值**（课案原文如此）：
#    意思是「几乎立刻开始清理」，只为让清理效果一眼可见；
#    生产里应当按真实 token 数设（如 trigger=("tokens", 10000)），否则每轮都在清。
# 5. 踩坑提示 C —— Demo 5 不适合照搬到写操作：ToolRetryMiddleware 重试的是**同一次调用**，
#    对「扣款 / 发消息」这类非幂等操作盲目重试等于重复执行（课案表格里也专门标了这句）。

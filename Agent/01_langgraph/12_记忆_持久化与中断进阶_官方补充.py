# -*- coding: utf-8 -*-
r"""
LangGraph 官方补充篇③：记忆管理、持久化粒度与中断进阶（非课案内容）
================================================================
来源与定位：
    本文件对照 LangGraph **官方文档**继续补课案的空白，是 01_langgraph 补充系列的第 3 篇：
      - add-memory.mdx（Manage short-term memory）→ 记忆「怎么不爆」（Demo 1）
      - checkpointers.mdx（Durability modes）     → 检查点写几次由你定（Demo 2）
      - interrupts.mdx（Common patterns / Rules） → 多中断并行 + 工具内中断 + 规则（Demo 3/4）
    前两篇：10_控制流与函数式API_官方补充.py、11_容错与测试_官方补充.py。

本文件对应缺口表（Agent/官方文档缺口对照.md）里的第 7、10、11 项：

    # | 缺口 | 官方出处 | 本文件
    7 | 短期记忆的上下文管理（trim / delete / summarize） | add-memory.mdx | Demo 1
    10| 中断进阶（多中断并行 / 工具内中断 / 中断规则）    | interrupts.mdx | Demo 3/4
    11| Durability modes（"exit"/"async"/"sync"）          | checkpointers.mdx | Demo 2
    9 | 子图持久化三种作用域                              | use-subgraphs.mdx | 未收录（见文末说明）

和课案的关系：
    课案 01_langgraph 的 02/03/04 讲了「记忆怎么存」（checkpointer / Store），
    06/07 讲了「单点中断 + 审批」，08 讲了时间旅行。本文件补的是同一批概念的**另一半**：
    记忆怎么不爆、检查点写多勤、并行中断怎么恢复、工具里怎么暂停。

运行方式（项目根目录下，**全离线、0 次模型调用**）：
    uv run Agent/01_langgraph/12_记忆_持久化与中断进阶_官方补充.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt
from pydantic import PrivateAttr


# ================================================================
# Demo 1：记忆「怎么不爆」—— 裁剪 / 删除 / 摘要三种手法
# ================================================================
# 课案讲了记忆怎么**存**；官方 add-memory.mdx 还讲了记忆怎么**不爆**（对话一长就撑爆上下文）：
#     裁剪（trim_messages）：按 token 数砍掉旧消息，本次请求变短，state 不动；
#     删除（RemoveMessage）：按 id 精确删掉 state 里的消息（配合摘要用）；
#     摘要（summarize 节点）：把早期对话压成一段文字存进自定义 state 字段，原文删掉。
#
#    ⚠️ 三者与「内置中间件」的关系：02_langchain/11 讲的 SummarizationMiddleware /
#       ContextEditingMiddleware 就是前两者的**官方封装**（生产直接用）；
#       这里手写一遍是为了看懂机制 —— 知道封装里干了什么，出问题才排得动。
class ChatState(MessagesState):
    """MessagesState 自带 messages 字段（用 add_messages reducer）；
    这里再挂一个 summary 字段，用来存压缩后的早期对话。"""

    summary: str


def trim_before_model(state: ChatState) -> dict:
    """节点里演示裁剪：只看不改 —— 计算「如果只保留最近 N 条会是什么样」。"""
    trimmed = trim_messages(
        state["messages"],
        strategy="last",          # 从最新往回保留
        token_counter=len,        # 教学用「条数」当 token 计数（真实项目用模型或估算函数）
        max_tokens=3,             # 最多保留 3 条
        start_on="human",         # 保证裁剪后第一条是 human（否则模型看到半截对话）
        include_system=True,      # 系统提示永远保留
        allow_partial=False,      # 不切半条消息
    )
    print(f"    [trim] 原始 {len(state['messages'])} 条 → 裁到 {len(trimmed)} 条")
    for message in trimmed:
        print(f"        {message.type:<7} {str(message.content)[:28]}")
    return {}   # 只演示，不改状态


def summarize_and_delete(state: ChatState) -> dict:
    """官方 summarize 模式：把早期对话压成摘要写进 state，再把原文删掉。"""
    old_messages = state["messages"][:2]          # 假设最早的 2 条要被压缩掉
    summary = state.get("summary") or ""
    new_summary = (summary + " / " if summary else "") + "用户之前问过背景问题并得到了答复"
    # RemoveMessage(id=...) 是唯一能从 state 里**删**消息的手段（reducer 只支持增，不支持删）
    removals = [RemoveMessage(id=m.id) for m in old_messages]
    print(f"    [summarize] 删除 {len(removals)} 条旧消息，摘要字段更新为：{new_summary}")
    return {"messages": removals, "summary": new_summary}


memory_graph = (
    StateGraph(ChatState)
    .add_node("trim", trim_before_model)
    .add_node("summarize", summarize_and_delete)
    .add_edge(START, "trim")
    .add_edge("trim", "summarize")
    .add_edge("summarize", END)
    .compile(checkpointer=InMemorySaver())
)


# ================================================================
# Demo 2：Durability modes —— 检查点写几次，由你决定
# ================================================================
# 官方 checkpointers.mdx 的 durability 参数有三档（课案没提过）：
#     "sync"  ：每个 super-step 都**同步**落盘（写完才继续）—— 最稳，最慢；
#     "async" ：每步**异步**落盘（不阻塞执行）—— 折中，崩溃时可能丢最后一步；
#     "exit"  ：只在**运行结束时**落一次盘 —— 最快，但中途崩溃等于没跑过、
#               也无法中断恢复/时间旅行（因为这些都要靠中间检查点）。
#   默认值 None（等于按 checkpointer 的默认策略走）。
class StepState(MessagesState):
    step: int


def make_step_node() -> object:
    def step_node(state: StepState) -> dict:
        return {"step": state.get("step", 0) + 1}

    return step_node


# 注：Demo 2 里每种 durability 都**新建一个图**（各自独立的 InMemorySaver），
# 这样三档的检查点历史互不干扰，数出来的数量才干净。


# ================================================================
# Demo 3：多中断并行 —— 一次恢复多个
# ================================================================
# 课案 06/07 讲的是「单个节点中断 → Command(resume=值) 恢复」。
# 官方 interrupts.mdx 的 Common patterns 里还有一幕：**并行分支各自中断**，
# 这时一次 invoke 会返回**多个** Interrupt，恢复要用「中断 id → 决定」的字典。
class ParallelState(MessagesState):
    a: str
    b: str


def branch_a(state: ParallelState) -> dict:
    decision = interrupt({"branch": "A", "ask": "A 分支要放行吗？"})
    return {"a": f"A：{decision}"}


def branch_b(state: ParallelState) -> dict:
    decision = interrupt({"branch": "B", "ask": "B 分支要放行吗？"})
    return {"b": f"B：{decision}"}


parallel_graph = (
    StateGraph(ParallelState)
    .add_node("a", branch_a)
    .add_node("b", branch_b)
    .add_edge(START, "a")
    .add_edge(START, "b")      # 两条并行分支，各自都会中断
    .add_edge("a", END)
    .add_edge("b", END)
    .compile(checkpointer=InMemorySaver())
)


# ================================================================
# Demo 4：工具内中断 + 中断的三条规则
# ================================================================
# 官方 interrupts.mdx 明确支持「在工具内部 interrupt」：工具执行到一半需要人给输入
# （要多少预算、选哪个方案、确认收货地址…），这时整个运行会暂停，resume 的值
# **作为 interrupt() 的返回值**回到工具里继续跑。
#
# 三条必须记住的规则（官方 Rules of interrupts）：
#     1. **不能用 try/except 包住 interrupt()** —— 暂停是靠抛异常实现的，
#        被吞掉之后框架再也恢复不了这次运行；
#     2. 中断点之前的**副作用必须幂等** —— 恢复时节点/工具会从头重放，
#        发邮件、扣款这类操作要做成可重复执行；
#     3. 同一个节点里多次 interrupt 时，**顺序和次数必须稳定**（别用随机/时间条件控制），
#        否则恢复时对不上号。
class ScriptedModel(ChatOpenAI):
    """按剧本依次吐消息的假模型（同 11_内置中间件_官方补充.py 的手法）。

    这里是工具内中断的演示 —— 需要「模型先要求调工具」这一步，用剧本模型最稳。
    """

    _script: list = PrivateAttr(default_factory=list)
    _cursor: int = PrivateAttr(default=0)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = self._script[self._cursor]
        self._cursor += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


@tool
def request_budget(purpose: str) -> str:
    """申请一笔预算（金额需要人工确认）。"""
    # 工具执行到一半停下问人：resume 传进来的值就是 interrupt 的返回值
    amount = interrupt({"question": f"「{purpose}」需要多少预算？"})
    return f"「{purpose}」已批准预算 {amount} 元"


def build_budget_agent():
    model = ScriptedModel(model="scripted", api_key="offline", base_url="http://localhost:9")
    model._script = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "request_budget",
                "args": {"purpose": "团建"},
                "id": "c1",
                "type": "tool_call",
            }],
        ),
        AIMessage(content="预算已办妥。"),
    ]
    return create_agent(model=model, tools=[request_budget], checkpointer=InMemorySaver())


if __name__ == "__main__":
    # ---------- Demo 1 ----------
    print("=" * 70)
    print("Demo 1：记忆不爆的三种手法 —— 裁剪 / 删除+摘要")
    print("=" * 70)
    history = [
        SystemMessage(content="你是记账助手", id="s1"),
        HumanMessage(content="我昨天花了 30 元吃午饭", id="m1"),
        AIMessage(content="记下了：午饭 30 元", id="m2"),
        HumanMessage(content="今天午饭花了 45 元", id="m3"),
        AIMessage(content="记下了：午饭 45 元", id="m4"),
        HumanMessage(content="这周一共花了多少？", id="m5"),
    ]
    config = {"configurable": {"thread_id": "memory-demo"}}
    result = memory_graph.invoke({"messages": history, "summary": ""}, config)
    print(f"  处理后的消息：{[str(m.content)[:16] for m in result['messages']]}")
    print(f"  summary 字段：{result['summary']}")
    print(
        "  ↑ 裁剪（trim）只影响「本次发给模型的内容」，不动 state；\n"
        "    删除（RemoveMessage）+ 摘要才真正压缩 state —— 课案讲过的 SummarizationMiddleware\n"
        "    与 ContextEditingMiddleware 就是这两招的封装版（生产直接用封装，懂原理才排得动）。"
    )

    # ---------- Demo 2 ----------
    print("\n" + "=" * 70)
    print("Demo 2：Durability modes —— 检查点写几次由你定")
    print("=" * 70)
    for mode in ("sync", "async", "exit"):
        graph = (
            StateGraph(StepState)
            .add_node("one", make_step_node())
            .add_node("two", make_step_node())
            .add_node("three", make_step_node())
            .add_edge(START, "one")
            .add_edge("one", "two")
            .add_edge("two", "three")
            .add_edge("three", END)
            .compile(checkpointer=InMemorySaver())
        )
        run_config = {"configurable": {"thread_id": f"durability-{mode}"}}
        out = graph.invoke({"messages": [], "step": 0}, run_config, durability=mode)
        checkpoints = len(list(graph.get_state_history(run_config)))
        print(f"  durability={mode:<6} 最终 step={out['step']}，落盘检查点数量={checkpoints}")
    print(
        "  ↑ 实测：sync / async 都写了 5 个检查点，而 exit **只写 1 个**（运行结束才落盘）。\n"
        "    选型：要中断恢复/时间旅行 → sync 或 async（这俩数量相同，差别在写入时机：\n"
        "    sync 写完才继续、async 异步写）；纯批处理、崩了重跑即可 → exit 最快。"
    )

    # ---------- Demo 3 ----------
    print("\n" + "=" * 70)
    print("Demo 3：多中断并行 —— 用「中断 id → 决定」的字典一次恢复")
    print("=" * 70)
    multi_config = {"configurable": {"thread_id": "parallel-interrupt"}}
    first = parallel_graph.invoke({"messages": [], "a": "", "b": ""}, multi_config)
    interrupts = first.get("__interrupt__", [])
    print(f"  第一次 invoke 返回 {len(interrupts)} 个中断：")
    for item in interrupts:
        print(f"    id={item.id[:12]}… value={item.value}")
    decisions = {item.id: f"{item.value['branch']} 分支已放行" for item in interrupts}
    resumed = parallel_graph.invoke(Command(resume=decisions), multi_config)
    print(f"  字典恢复后：a={resumed['a']!r} b={resumed['b']!r}")
    print(
        "  ↑ 单个中断时 Command(resume=值) 就够；**并行多中断**必须用\n"
        "    Command(resume={中断id: 值}) 把每个决定送回对应的分支。"
    )

    # ---------- Demo 4 ----------
    print("\n" + "=" * 70)
    print("Demo 4：工具内中断 —— 工具执行到一半问人要输入")
    print("=" * 70)
    agent = build_budget_agent()
    agent_config = {"configurable": {"thread_id": "tool-interrupt"}}
    first = agent.invoke({"messages": [{"role": "user", "content": "帮我办个团建"}]}, agent_config)
    print(f"  第一次 invoke 返回中断：{[i.value for i in first.get('__interrupt__', [])]}")
    second = agent.invoke(Command(resume=5000), agent_config)
    tool_messages = [m for m in second["messages"] if m.type == "tool"]
    print(f"  恢复后工具返回：{[str(m.content) for m in tool_messages]}")
    print(f"  最终回复：{second['messages'][-1].content}")
    print(
        "  ↑ resume 的值（5000）**成了工具里 interrupt() 的返回值**，工具继续跑完。\n"
        "    规则：① 别用 try/except 包 interrupt（暂停靠抛异常实现）；\n"
        "          ② 中断点之前的副作用必须幂等（恢复会重放）；\n"
        "          ③ 同一节点内多次 interrupt 的顺序/次数要稳定。"
    )

    print("\n全部 Demo 执行完毕（0 次模型调用，离线可复现）。")


# ================================================================
# 实测结论 / 未收录清单 / 踩坑提示
# ================================================================
# 1. 实测结论（langgraph 1.2.11，本机）：
#    - Demo 1：trim_messages(strategy="last", token_counter=len, max_tokens=3,
#      start_on="human", include_system=True) 把 6 条裁到 3 条且保留了 SystemMessage；
#      RemoveMessage 按 id 删除成功，自定义 summary 字段正常写入。
#    - Demo 2：durability="sync"/"async" 各写 5 个检查点，durability="exit" 只写 1 个。
#    - Demo 3：并行两分支各产生 1 个 Interrupt（共 2 个），用 {id: 值} 字典一次恢复成功。
#    - Demo 4：工具内 interrupt() 生效 —— 暂停返回 {'question': '「团建」需要多少预算？'}，
#      Command(resume=5000) 后工具返回「「团建」已批准预算 5000 元」。
# 2. 未收录（官方还有、本文件没做的）：
#    - **子图持久化三种作用域**（use-subgraphs.mdx 的 per-invocation / per-thread / stateless）：
#      实测发现父图自己的 checkpointer 会把子图状态一起存进父检查点，两种模式在
#      「父图返回值」上看不出差异；要观察差异得下钻子图命名空间（subgraphs=True 的快照），
#      本机未稳定复现出差异 → 只登记不写成 Demo，避免"注释里写的行为"和实测不一致。
#    - 长期记忆策略分类（semantic / episodic / procedural）与 Store 语义搜索：需要 embeddings。
#    - 可观测性（LangSmith / Studio）：需要外部账号。
# 3. 踩坑提示：
#    A. RemoveMessage 是**唯一**能从 state 删消息的手段（add_messages 只会追加）；
#       删除后 state 里就真没了，历史消息要用它前先确认这是你要的语义（时间旅行仍可回到旧检查点）。
#    B. trim_messages 的 token_counter 参数：教学用 len 最直观，生产要换成真实计数
#       （传模型实例，或用官方的近似计数函数），否则"3 条"在长文本场景下可能已经超窗。
#    C. durability="exit" 与「中断恢复/时间旅行」互斥：没有中间检查点，就没法从半路恢复。
#    D. 并行多中断的恢复字典必须用 **Interrupt.id** 当键（不是分支名、不是节点名）。
#    E. interrupt() 不能被 try/except 包住 —— 这一条踩了会表现为「resume 后再也走不动」，
#       排查时优先检查是不是把 interrupt 吞进了异常处理。

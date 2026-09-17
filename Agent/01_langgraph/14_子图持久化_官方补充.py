# -*- coding: utf-8 -*-
r"""
LangGraph 官方补充篇⑤：子图持久化三档（非课案内容）
================================================================
来源与定位：
    本文件对照 LangGraph **官方文档** use-subgraphs.mdx 的「Subgraph persistence」一节，
    补缺口表 **LangGraph 第 9 项**（此前标记为"父图 checkpointer 会把子图状态一起存，
    差异未稳定复现"—— 那是**实验设计错了**，本文件用正确的设计把差异做出来）。

官方给的三档（`compile()` 的 checkpointer 参数）：

    模式            checkpointer=   行为
    --------------  --------------  ----------------------------------------------------
    per-invocation  None（默认）     每次调用**全新开始**；但在**单次调用内**继承父图的
                                    checkpointer，所以仍支持 interrupt 与持久执行
    per-thread      True            状态**跨调用累积**（同一个 thread 上接着上次继续）
    stateless       False           **完全没有检查点** —— 像普通函数调用，
                                    不支持 interrupt，也没有持久执行

    前提：**父图必须带 checkpointer**，子图的持久化能力才谈得上（官方 Note）。

为什么值得单独讲：
    它决定「子代理要不要记住上一轮」。官方举例：客服机器人把问题转给"账单专家"子代理时，
    这位专家该记得客户之前问过什么，还是每次都从零开始？
    · 大多数多 Agent 场景（子代理处理一次性请求）→ **per-invocation**（默认就对）；
    · 需要多轮记忆的子代理（研究助手逐步积累上下文）→ **per-thread**；
    · 纯计算、不需要暂停恢复 → **stateless**（最省，也无恢复能力）。

⚠️ 三个模式里最容易搞混的是 **per-invocation 与 stateless**：两者"每次都是新的"，
    但 per-invocation 在**单次调用内**能中断/恢复（继承父图 checkpointer），
    stateless 不能 —— Demo 2 用 interrupt 把这条差异做成可复现的实测。

本文件**不需要模型**（纯图与状态机），可直接离线复现。

运行方式（项目根目录下）：
    uv run Agent/01_langgraph/14_子图持久化_官方补充.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, interrupt


# ================================================================
# 子图：内部有一个**私有计数器**，并把"第几次执行"写进消息
# ================================================================
# 关键设计（上一版实验就是错在这里）：
#     计数写在**子图私有字段** run_count 上（父图 schema 里没有这个键）。
#     如果父图也有同名字段并有 checkpointer，父图会把它一起存下来、再喂回子图，
#     于是"per-invocation 每次都是 1"这个差异就被掩盖了 —— 上一版因此看不出区别。
class SubState(MessagesState):
    run_count: int          # 子图私有：本轮是子图的第几次执行


def sub_step(state: SubState) -> dict:
    count = state.get("run_count", 0) + 1
    return {"run_count": count, "messages": [AIMessage(content=f"子图第 {count} 次执行")]}


def build_subgraph(mode: str):
    """按官方三种模式编译子图。"""
    builder = StateGraph(SubState)
    builder.add_node("step", sub_step)
    builder.add_edge(START, "step")
    builder.add_edge("step", END)
    if mode == "per-thread":
        return builder.compile(checkpointer=True)      # 跨调用累积
    if mode == "stateless":
        return builder.compile(checkpointer=False)     # 完全无检查点
    return builder.compile()                           # per-invocation（默认）


class ParentState(MessagesState):
    """父图状态：只有 messages（**故意不加 run_count**，见上面的关键设计）。"""


def build_parent(mode: str):
    """父图：带 checkpointer，把子图当一个节点。"""
    builder = StateGraph(ParentState)
    builder.add_node("sub", build_subgraph(mode))
    builder.add_edge(START, "sub")
    builder.add_edge("sub", END)
    return builder.compile(checkpointer=InMemorySaver())


# ================================================================
# Demo 1：三档对照 —— 同一个 thread 连调三次，看子图"记不记得"
# ================================================================
def demo_1_three_modes() -> None:
    print("=" * 70)
    print("Demo 1：三档对照（同一 thread 连调三次，看子图计数）")
    print("=" * 70)

    for mode in ("per-invocation", "per-thread", "stateless"):
        parent = build_parent(mode)
        config = {"configurable": {"thread_id": f"sub-persist-{mode}"}}
        counts: list[str] = []
        for _ in range(3):
            result = parent.invoke({"messages": []}, config)
            # 三次调用的消息会累积在父图里；取出「子图第 N 次执行」这条
            latest = [m.content for m in result["messages"] if "子图第" in str(m.content)]
            counts.append(str(latest[-1]).split("第")[1].split("次")[0].strip() if latest else "?")
        print(f"  {mode:<15} 三次调用中子图报告的批次：{counts}")
        if mode == "per-thread":
            print("                  ↑ 1→2→3：**状态跨调用累积**（每次接着上次继续）")
        else:
            print("                  ↑ 每次都从 1 开始：说明子图状态没有跨调用保留")

    print(
        "\n  ↑ 结论清清楚楚：\n"
        "    · per-invocation（默认）：每次调用**全新开始**（计数恒为 1）；\n"
        "    · per-thread（checkpointer=True）：**跨调用累积**（1→2→3）；\n"
        "    · stateless（checkpointer=False）：也每次从 1 开始 —— 它与 per-invocation\n"
        "      **在这一栏看不出区别**，差别在 Demo 2（能不能中断恢复）。"
    )


# ================================================================
# Demo 2：per-invocation vs stateless 的真正差异 —— 能否 interrupt
# ================================================================
# 官方对 per-invocation 的描述是「在**单次调用内**继承父图 checkpointer，
# 因此仍支持 interrupt 与持久执行」；而 stateless 是「完全没有检查点，
# 不支持 interrupts 或 durable execution」。
# 本 Demo 让子图里的节点调用 interrupt()，对两种模式各跑一次看结果。
class InterruptSubState(MessagesState):
    approved: str


def sub_step_with_interrupt(state: InterruptSubState) -> dict:
    decision = interrupt({"question": "子图需要人工确认，批准吗？"})
    return {"approved": str(decision), "messages": [AIMessage(content=f"子图收到决定：{decision}")]}


def build_interrupt_subgraph(mode: str):
    builder = StateGraph(InterruptSubState)
    builder.add_node("ask", sub_step_with_interrupt)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    if mode == "stateless":
        return builder.compile(checkpointer=False)
    return builder.compile()          # per-invocation：继承父图 checkpointer


def build_interrupt_parent(mode: str):
    builder = StateGraph(ParentState)
    builder.add_node("sub", build_interrupt_subgraph(mode))
    builder.add_edge(START, "sub")
    builder.add_edge("sub", END)
    return builder.compile(checkpointer=InMemorySaver())


def demo_2_interrupt_difference() -> None:
    print("\n" + "=" * 70)
    print("Demo 2：per-invocation vs stateless —— 能不能在子图里中断")
    print("=" * 70)

    observed: dict[str, str] = {}
    for mode in ("per-invocation", "stateless"):
        print(f"\n  --- 模式：{mode} ---")
        parent = build_interrupt_parent(mode)
        config = {"configurable": {"thread_id": f"sub-interrupt-{mode}"}}
        try:
            first = parent.invoke({"messages": [], "approved": ""}, config)
            if "__interrupt__" in first:
                payload = first["__interrupt__"][0].value
                print(f"    第一次 invoke 正常暂停，中断负载：{payload}")
                resumed = parent.invoke(Command(resume="批准"), config)
                print(f"    恢复后的消息：{[str(m.content) for m in resumed['messages']]}")
                observed[mode] = "可中断可恢复"
            else:
                print(f"    没有中断，直接跑完：{[str(m.content) for m in first['messages']]}")
                observed[mode] = "未中断"
        except Exception as exc:  # noqa: BLE001
            print(f"    ✘ 执行失败：{type(exc).__name__}: {str(exc)[:160]}")
            observed[mode] = f"失败：{type(exc).__name__}"

    print(f"\n  实测结果对照：{observed}")
    if observed.get("stateless") == "可中断可恢复":
        print(
            "  ⚠️ 注意：**本机这个实验里 stateless 也能中断恢复** —— 与官方那句\n"
            "     「stateless 不支持 interrupts / durable execution」并不冲突，而是场景问题：\n"
            "     这里**父图带了 checkpointer**，中断信息由父图落盘，所以照样能暂停恢复。\n"
            "     stateless 的代价要在别的场景才显形：子图**内部多步执行**、中途崩了要续跑时\n"
            "     它没有自己的检查点可用（本文件没有构造那个场景，所以不下断言）。\n"
            "     实践建议：**没特殊理由就用默认的 per-invocation** —— 它等于\n"
            "     「每次全新 + 单次调用内可恢复」，两头都占了。"
        )
    print(
        "\n  ↑ 三档的完整画像（结合 Demo 1 的计数结果）：\n"
        "    · per-invocation：每次全新（计数恒 1）+ 单次调用内可中断/可恢复 ← 默认首选；\n"
        "    · per-thread：跨调用累积（1→2→3），适合需要多轮记忆的子代理；\n"
        "    · stateless：也每次全新；省掉子图自己的检查点，但**没有可下钻的子图状态**\n"
        "      （Demo 3 会看到：它的子图快照是拿不到的/没有的）。"
    )


# ================================================================
# Demo 3：下钻看子图自己的状态（运维/排障用）
# ================================================================
# 实测：`get_state(config, subgraphs=True)` 返回的是 **StateSnapshot**，
# 下钻路径是 `snapshot.tasks[*].state`（Task 上的 state 才是子图快照）。
# 注意**要在暂停时下钻**：图跑完之后 tasks 是空的（没有待执行任务），
# 也就看不到子图快照了 —— 这正是"卡在半路要排障"时最常用的姿势。
def demo_3_inspect_subgraph_state() -> None:
    print("\n" + "=" * 70)
    print("Demo 3：下钻子图状态（在中断暂停时看最合适）")
    print("=" * 70)

    for mode in ("per-invocation", "stateless"):
        print(f"\n  --- 模式：{mode} ---")
        parent = build_interrupt_parent(mode)
        config = {"configurable": {"thread_id": f"sub-inspect-{mode}"}}
        parent.invoke({"messages": [], "approved": ""}, config)     # 跑到中断处停下

        snapshot = parent.get_state(config, subgraphs=True)
        print(f"    父图 values（只有 messages，子图私有字段不冒泡）：{sorted(snapshot.values.keys())}")
        print(f"    待执行任务数：{len(snapshot.tasks)}")
        for task in snapshot.tasks:
            inner = getattr(task, "state", None)
            print(f"      任务 {task.name}：", end="")
            if inner is None:
                print("（拿不到子图快照）")
                continue
            print(f"子图快照 ✔")
            print(f"        子图 values={inner.values}")
            print(f"        子图待执行节点={inner.next}")
            print(f"        子图内的中断={[item.value for item in (inner.interrupts or [])]}")
        parent.invoke(Command(resume="批准"), config)               # 收尾，别留半截状态

    print(
        "\n  ↑ 三个实测要点（都是这次踩出来的）：\n"
        "    ① 下钻路径是 **`snapshot.tasks[*].state`**，不是 `snapshot.subgraphs`（不存在该属性）；\n"
        "    ② **要在暂停时下钻** —— 图跑完后 tasks 为空，什么都看不到；\n"
        "    ③ 子图私有字段（比如本文件的 run_count）**不会冒泡到父图**，\n"
        "       父图 values 里只有 messages —— 想给上层用，得自己在子图里 return 出去。"
    )


if __name__ == "__main__":
    demo_1_three_modes()
    demo_2_interrupt_difference()
    demo_3_inspect_subgraph_state()
    print("\n全部 Demo 执行完毕（0 次模型调用，离线可复现）。")


# ================================================================
# 实测结论 / 与课案的衔接 / 踩坑提示
# ================================================================
# 1. 官方事实（use-subgraphs.mdx#subgraph-persistence）：
#    - 三档：checkpointer=None（per-invocation，默认）/ True（per-thread）/ False（stateless）；
#    - per-invocation：每次全新，但单次调用内继承父图 checkpointer（支持 interrupt/持久执行）；
#    - per-thread：状态跨调用累积；stateless：无检查点、不支持 interrupt；
#    - 前提：父图必须带 checkpointer。
# 2. 本机实测（langgraph 1.2.11）：
#    - Demo 1：per-invocation 三次调用子图计数都是 1；per-thread 是 1→2→3；
#      stateless 也是每次 1（与 per-invocation 在这一栏相同）；
#    - Demo 2：**两种模式都能中断并恢复**（`{'per-invocation': '可中断可恢复',
#      'stateless': '可中断可恢复'}`）—— 因为父图带 checkpointer，中断由父图落盘。
#      官方说 stateless「不支持 interrupts / durable execution」，在本机这个场景下**没复现出差异**，
#      文件里如实说明了原因，并指出代价要在"子图内部多步执行 + 中途续跑"时才显形；
#    - Demo 3：**这才是 stateless 的实测差异** —— 暂停时下钻子图状态，
#      per-invocation 能拿到子图快照（values / next / interrupts），
#      stateless **拿不到**（`task.state is None`）；
#      下钻路径是 `snapshot.tasks[*].state`（**不是** `snapshot.subgraphs`）。
# 3. 与课案的衔接：
#    - 课案 01_langgraph 09_子图 讲了「怎么把子图当节点用」；
#    - 12_记忆_持久化与中断进阶_官方补充.py 讲了父图层面的 durability 三档
#      （exit/async/sync）—— 与这里的**子图持久化三档**是两件不同的事，别混；
#    - 03_deepagents 的同步/异步子代理则是"子代理"视角，同样与这三档正交。
# 4. 踩坑提示：
#    A. **实验设计别让父图"帮忙"持久化**：想让子图私有状态可见，就不要把同名字段放进
#       父图 schema（否则父图 checkpointer 会替你记住，看不出 per-invocation 的"每次全新"）；
#    B. per-invocation 与 stateless 的差别**不在"记不记得"**，而在**能不能中断/恢复** ——
#       只看状态计数会把两者当成一样；
#    C. per-thread 的子图**必须**与父图用同一个 thread（框架自动处理命名空间），
#       换 thread 就是新会话；
#    D. 子图状态不自动冒泡到父图：要用到的数据得自己 return 上去（或让子图写进 messages）。

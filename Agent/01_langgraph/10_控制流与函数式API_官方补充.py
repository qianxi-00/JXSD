# -*- coding: utf-8 -*-
r"""
LangGraph 官方补充篇①：控制流三件套与函数式 API（非课案内容，故不带 _jxsd 后缀）
================================================================
来源与定位：
    本文件对照 LangGraph **官方文档**（/oss/python/langgraph）补课案没讲的知识点。
    官方出处：
      - use-graph-api.mdx     → 「Map-Reduce and the send API」「Combine control flow and
                                state updates with Command」「Create and control loops」
      - graph-api.mdx         → Command / Send 的概念定义
      - functional-api.mdx、use-functional-api.mdx → 函数式 API（@entrypoint / @task）
      - choosing-apis.mdx     → 两种 API 的选型
      - workflows-agents.mdx  → 六种经典工作流模式

课案 01_langgraph（00~09）把「图 API」主干讲完了：StateGraph / 记忆 / 中断 / 时间旅行 /
子图 / 流式。但对官方文档仍有结构性缺口：

    课案已讲                      官方文档还有（本文件补哪些）
    ----------------------------  ------------------------------------------------
    条件边（add_conditional_edges） ✔ 已讲 → 控制流三件套里的第 1 件
    Command(resume=...) 恢复中断    → Command 还有 update / goto 两个字段【本文件 Demo 2】
    （无）                         → Send API：Map-Reduce 动态并行【Demo 1】
    （无）                         → 函数式 API @entrypoint/@task【Demo 3】

LangGraph 12 大缺口速览（官方文档 vs 课案，按学习价值排序；本文件补前 3 项）：

    # | 缺口 | 官方出处 | 优先级
    1 | Send API（Map-Reduce 并行 fan-out）      | use-graph-api.mdx | 高 ← 本文件 Demo 1
    2 | Command(goto)（控制流+状态更新合一）      | use-graph-api.mdx | 高 ← 本文件 Demo 2
    3 | 函数式 API（@entrypoint/@task）           | functional-api.mdx | 高 ← 本文件 Demo 3
    4 | 六个经典工作流模式（orchestrator-worker 等）| workflows-agents.mdx | 高
    5 | 容错：RetryPolicy / 节点超时 / 错误处理    | fault-tolerance.mdx | 高 → 见 11_官方补充
    6 | 测试 LangGraph 应用（三种 pytest 模式）    | test.mdx | 高 → 见 11_官方补充
    7 | 短期记忆的上下文管理（trim/delete/summarize）| add-memory.mdx | 中
    8 | 长期记忆策略分类 + Store 语义搜索           | concepts/memory.mdx | 中
    9 | 子图持久化作用域（三种模式）               | use-subgraphs.mdx | 中
    10| 中断进阶（多中断并行 / 工具内中断 / 规则）  | interrupts.mdx | 中
    11| Durability modes 与自定义 checkpointer     | checkpointers.mdx | 中
    12| 可观测性与本地开发服务器（Studio/dev）      | observability.mdx | 中

为什么本文件全部离线（0 次模型调用）：
    控制流是**纯 Python 机制**——Send 怎么扇出、Command 怎么跳转、@task 怎么复用结果，
    这些行为的对错与模型无关，用固定数据才能确定性复现（也才好排查）。生产里
    "「产出主题」「写笑话」这类节点当然应该换成模型调用，本文件用固定值替代并在注释里标出。

两个容易搞错的官方写法（本地实测 + 官方文档核对）：
      - **批量默认策略的正确 API 是 `set_node_defaults(...)`**，本地 1.2.11 就有：
            StateGraph.set_node_defaults(retry_policy=RetryPolicy(max_attempts=2), timeout=5)
        它给图里**所有**节点设默认值（单节点 add_node 传的值优先），在 compile() 时生效。
        写成 `add_node_defaults` 会 hasattr 为 False —— 那是名字记错了，不是版本不支持。
      - 官方的 **list-form edge 是「多起点」**：`add_edge(["a", "b"], "c")` 表示 a、b 都完成后
        才走 c（本地同样支持）；而把列表当**终点**的 `add_edge(START, ["a", "b"])` 不支持
        （实测 TypeError: unhashable type: 'list'）—— 并行扇出请用 Send，或写两条 add_edge。

运行方式（项目根目录下）：
    uv run Agent/01_langgraph/10_控制流与函数式API_官方补充.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import operator
import time
from typing import Annotated, Literal

from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt


# ================================================================
# Demo 1：Send API —— Map-Reduce 与并行 fan-out
# ================================================================
# 要解决的问题：课案 01_基础图 教的是「一个节点跑完轮到下一个」的串行图。
#   现实里经常遇到「一批互相独立的任务」——10 个文档各写摘要、5 个城市各查天气。
#   用串行图要写循环；用 Send 则是**运行时按数据量动态扇出**：有几条数据就起几个
#   同名节点实例，并行跑，结果靠 reducer 自动汇总。
#
# 关键点（三个，缺一个就跑不对）：
#   1. 汇总字段必须是 `Annotated[list, operator.add]` 这种**带 reducer** 的声明；
#      否则多个 worker 的结果会互相覆盖，只剩最后一条（课案 00_框架总览讲过
#      「覆盖 vs 追加」的区别，这里是它最重要的实际用途）。
#   2. fan-out 函数返回 `[Send("节点名", 私有输入), ...]`，挂在条件边上。
#   3. worker 收到的 state **不是**图的完整 state，而是 Send 第二参传进去的私有输入
#      —— 每个 worker 各拿一份，所以并行时不会互相踩。
class MapReduceState(TypedDict):
    """Map-Reduce 的公共状态。"""

    subjects: list[str]                          # 待处理清单（map 的输入）
    jokes: Annotated[list[str], operator.add]    # worker 产出（reduce 汇总）
    best: str                                    # 最终挑选结果


def generate_topics(state: MapReduceState) -> dict:
    """官方示例里这一步是「让模型生成主题」；本文件用固定清单保证可复现。"""
    print("[generate_topics] 产出 3 个主题（生产里这步是模型调用）")
    return {"subjects": ["狮子", "大象", "企鹅"]}


def fan_out_to_workers(state: MapReduceState) -> list[Send]:
    """条件边 = 分拣站：为 subjects 里每条数据各派一个 write_joke 实例。"""
    print(f"[fan_out] 把 {len(state['subjects'])} 个主题分发给 worker（并行执行）")
    # Send(节点名, 该实例的输入状态)：有几条就起几个实例，互不干扰
    return [Send("write_joke", {"subject": s}) for s in state["subjects"]]


def write_joke(state: dict) -> dict:
    """worker 节点：注意它收到的 state 只有 Send 传进来的 {"subject": ...}。"""
    subject = state["subject"]
    print(f"    [write_joke] 处理：{subject}")
    # 返回 {"jokes": [...]}：经 operator.add 追加进公共状态，而不是覆盖
    return {"jokes": [f"《{subject}》：冷笑话一则"]}


def pick_best(state: MapReduceState) -> dict:
    """reduce 之后的收口节点：这时才能看到**全部** worker 的产出。"""
    print(f"[pick_best] 汇总到 {len(state['jokes'])} 条结果，挑第一条")
    # 生产里这步一般也是模型（打分选优）；这里固定取第一条
    return {"best": state["jokes"][0]}


map_reduce_graph = (
    StateGraph(MapReduceState)
    .add_node("generate_topics", generate_topics)
    .add_node("write_joke", write_joke)
    .add_node("pick_best", pick_best)
    .add_edge(START, "generate_topics")
    # 条件边的第三个参数是「可能去向」的声明（返回 Send 时也可省略，写上更清晰）
    .add_conditional_edges("generate_topics", fan_out_to_workers, ["write_joke"])
    .add_edge("write_joke", "pick_best")   # 所有 worker 都完成后才轮到 pick_best
    .add_edge("pick_best", END)
    .compile()
)


# ================================================================
# Demo 2：Command(goto=...) —— 控制流的另一半
# ================================================================
# 课案里的 Command 只用过 `Command(resume=...)`（06/07 章把人工决定送回被冻结的图）。
# 官方文档里 Command 有三个字段，另两个课案没讲：
#     Command(update={...}, goto="节点名")  // 一个返回值里**同时**改状态 + 定去向
#     Command(resume=...)                  // 课案已讲（中断恢复）
# 什么时候用它替代条件边？——当「往哪走」和「状态怎么改」本来就是同一件事的时候：
# 条件边得写成两个函数（一个改状态、一个做路由），Command 一次返回搞定。
#
# ⚠️ 官方明确的警告：Command 只**新增动态边**，不会取消静态边。
#    如果 decide 同时用 add_edge 连了别的节点，那么 goto 的目标和静态边的目标
#    **都会被执行**。所以下面的 decide 故意不声明任何静态出边。
class RouteState(TypedDict):
    mode: str
    visited: Annotated[list[str], operator.add]


def decide(state: RouteState) -> Command[Literal["fast_path", "slow_path"]]:
    """返回类型注解里用 Literal 声明可能的去向（官方推荐写法，便于静态检查与画图）。"""
    if state["mode"] == "fast":
        # 一次返回：把走过的路径写进状态，同时决定下一步去 fast_path
        return Command(update={"visited": ["decide → fast_path"]}, goto="fast_path")
    return Command(update={"visited": ["decide → slow_path"]}, goto="slow_path")


def fast_path(state: RouteState) -> dict:
    return {"visited": ["fast_path 执行完毕"]}


def slow_path(state: RouteState) -> dict:
    return {"visited": ["slow_path 执行完毕"]}


route_graph = (
    StateGraph(RouteState)
    .add_node("decide", decide)
    .add_node("fast_path", fast_path)
    .add_node("slow_path", slow_path)
    .add_edge(START, "decide")
    # 关键：这里没有 decide 的任何静态出边 —— 去向 100% 由 Command(goto=...) 决定
    .add_edge("fast_path", END)
    .add_edge("slow_path", END)
    .compile()
)


# 循环也可以用 Command 写（官方 use-graph-api 的「Create and control loops」一节）
# 本 Demo 的 tick 节点自己跳自己，n 到 3 就 goto=END —— 图上一条循环边都没有。
class LoopState(TypedDict):
    n: int
    log: Annotated[list[str], operator.add]


def tick(state: LoopState) -> Command[Literal["tick", "__end__"]]:
    n = state.get("n", 0) + 1
    if n >= 3:
        return Command(update={"n": n, "log": [f"第 {n} 次：到点了，结束"]}, goto=END)
    return Command(update={"n": n, "log": [f"第 {n} 次：继续"]}, goto="tick")


loop_graph = (
    StateGraph(LoopState)
    .add_node("tick", tick)
    .add_edge(START, "tick")
    .compile()
)


# ================================================================
# Demo 3：函数式 API —— @entrypoint / @task
# ================================================================
# 官方有**两种**建模方式，课案 100% 只教了图 API（StateGraph）：
#     图 API      —— 显式画节点和边，状态机思路，适合复杂分支/多人协作/可视化调试；
#     函数式 API  —— 就是普通 Python 函数 + 两个装饰器，适合计算型流程与快速原型。
#   两者**共用同一个运行时**（同一套 checkpointer / interrupt / 流式），可以混用。
#   （选型细节见官方 choosing-apis.mdx）
#
#   两个装饰器：
#     @task        = 可持久化的函数：它的返回值会写进 checkpoint，**恢复时不重算**
#     @entrypoint  = 可持久化的流程入口：支持 checkpointer、interrupt、stream
#
# 本 Demo 把「函数式 API + 人工中断 + 重放不重算」串成一条：
#   第一次 invoke 跑到 interrupt 暂停 → resume 继续 → 前面那个昂贵的 @task
#   **不会**再执行第二次（这就是 @task 相对普通函数最本质的差别）。
task_calls = {"n": 0}


@task
def expensive_double(x: int) -> int:
    """模拟昂贵计算：调用次数会被打印出来，用来证明「恢复时有没有重算」。"""
    task_calls["n"] += 1
    print(f"    [task expensive_double] 第 {task_calls['n']} 次真正执行（模拟耗时计算）")
    time.sleep(0.3)
    return x * 2


@entrypoint(checkpointer=MemorySaver())
def review_flow(inp: dict) -> dict:
    """函数式 API 的入口：普通函数写法，靠装饰器获得持久化与中断能力。"""
    # .result() 同步取回 task 结果；task 的结果此刻已经进 checkpoint 了
    data = expensive_double(inp["x"]).result()
    # interrupt() 与图 API 里用法完全一样（课案 06/07 讲过），返回值就是 resume 时传进来的东西
    decision = interrupt({"question": "翻倍完成，放行吗？", "data": data})
    return {"data": data, "decision": decision}


if __name__ == "__main__":
    # ---------- Demo 1 ----------
    print("=" * 70)
    print("Demo 1：Send API —— Map-Reduce 并行扇出")
    print("=" * 70)
    mr_result = map_reduce_graph.invoke({"subjects": [], "jokes": [], "best": ""})
    print("\n最终状态：")
    print("  subjects:", mr_result["subjects"])
    for index, joke in enumerate(mr_result["jokes"], start=1):
        print(f"  jokes[{index}]:", joke)
    print("  best    :", mr_result["best"])
    print(
        "  ↑ 3 条结果**同时**在 jokes 里 —— 这就是 reducer(operator.add) 的作用；\n"
        "    如果把 MapReduceState 里的 jokes 改成裸 list（没有 Annotated 声明），\n"
        "    三个 worker 在**同一 super-step** 写同一字段会直接抛 InvalidUpdateError\n"
        "    （LastValue 通道报 At key 'jokes': Can receive only one value per step），图直接失败。\n"
        "    注意区分：跨 super-step 的串行写入才是「后写覆盖先写」。"
    )

    # ---------- Demo 2 ----------
    print("\n" + "=" * 70)
    print("Demo 2：Command(goto=...) —— 边改状态边决定去向")
    print("=" * 70)
    for mode in ("fast", "slow"):
        result = route_graph.invoke({"mode": mode, "visited": []})
        print(f"  mode={mode!r} 最终状态: visited={result['visited']}")
    print(
        "  ↑ decide 节点上**没有任何静态出边**，去向全靠 Command(goto=...) 动态决定；\n"
        "    对比课案的条件边写法：那需要单独再写一个路由函数，Command 把它和改状态合并了。"
    )

    loop_result = loop_graph.invoke({"n": 0, "log": []})
    print(f"\n  循环演示（Command 自跳，n 到 3 停）：n={loop_result['n']} log={loop_result['log']}")
    print("  ↑ 图上没有 tick→tick 的循环边，循环是 Command(goto='tick') 自己跳出来的")

    # ---------- Demo 3 ----------
    print("\n" + "=" * 70)
    print("Demo 3：函数式 API —— @entrypoint / @task + 中断 + 重放不重算")
    print("=" * 70)
    config = {"configurable": {"thread_id": "func-demo-1"}}

    print("\n--- 第一次 invoke：跑到 interrupt() 就暂停 ---")
    first = review_flow.invoke({"x": 21}, config)
    # 实测形态：返回的是一个 dict，里面有 __interrupt__ 键（与图 API 的中断返回一致）
    print("  返回类型:", type(first).__name__)
    print("  返回内容:", first)
    print("  ↑ 图停在这里，还没返回最终结果；interrupt() 收到的值存在 __interrupt__ 里")

    print("\n--- resume：用 Command(resume=...) 带着决定恢复 ---")
    second = review_flow.invoke(Command(resume="放行"), config)
    print("  返回内容:", second)
    print(
        f"  expensive_double 累计真正执行 {task_calls['n']} 次"
        " ← 恢复时**没有重算**（结果直接从 checkpoint 复用，这就是 @task 的价值）"
    )

    print("\n--- 换个 thread_id：一切重新开始 ---")
    other = review_flow.invoke({"x": 5}, {"configurable": {"thread_id": "func-demo-2"}})
    print("  返回内容:", other)
    print(
        f"  expensive_double 累计真正执行 {task_calls['n']} 次"
        " ← 新线程会重算（@task 的复用是**按 thread 隔离**的）"
    )
    print("\n全部 Demo 执行完毕（0 次模型调用，离线可复现）。")


# ================================================================
# 实测结论 / 未收录清单 / 踩坑提示
# ================================================================
# 1. 实测结论（langgraph 1.2.11，本机）：
#    - Demo 1：三个 worker 的产出都在 jokes 里（reducer 生效），pick_best 能看到全部 3 条；
#    - Demo 2：decide 无静态出边时 Command(goto=...) 正常跳转；tick 执行 3 次（自跳 2 次）后 goto=END；
#    - Demo 3：第一次 invoke 返回 {'__interrupt__': [Interrupt(value=..., id=...)]}；
#      resume 后 expensive_double **只执行过 1 次**（重放不重算）；换 thread 后变成第 2 次。
# 2. 两个容易搞错的官方写法（本地实测 + 官方文档核对）：
#    - **批量默认策略的 API 是 `set_node_defaults`（本地 1.2.11 就有）**，不是
#      add_node_defaults（那是名字记错，不是版本问题）：它在 compile() 时对图内所有节点生效，
#      单节点 add_node 传的策略优先；
#    - 官方 list-form edge 的列表是**起点**（add_edge(["a","b"], "c")：多起点都完成才走 c，
#      本地支持）；把列表当**终点**（add_edge(START, ["a","b"])）不支持 → TypeError:
#      unhashable type: 'list'。并行扇出请用 Send，或写两条独立 add_edge。
# 3. 未收录（官方还有、本文件没做的）：
#    - workflows-agents.mdx 的六种工作流模式：prompt chaining / parallelization / routing /
#      orchestrator-worker / evaluator-optimizer / agents。前三个本文件已覆盖机制，
#      evaluator-optimizer（结构化输出打分 + 反馈循环）需要真实模型，留给 08_结构化输出
#      与后续练习；orchestrator-worker 就是 Demo 1。
#    - 容错（RetryPolicy / 超时 / 错误处理）与测试（三种 pytest 模式）→ 见同目录
#      11_容错与测试_官方补充.py。
#    - 记忆工程化（trim/delete/summarize）→ 同目录 12_记忆_持久化与中断进阶_官方补充.py；
#    - 长期记忆策略分类与 Store 语义搜索 → 同目录 13_长期记忆_官方补充.py；
#    - 子图持久化三种作用域 → 同目录 14_子图持久化_官方补充.py；
#    - 可观测性（LangSmith / Studio）→ 见 Agent/官方文档缺口对照.md（本仓库不用 LangSmith）。
# 4. 踩坑提示：
#    A. Send 的 worker 收到的是**私有输入**（Send 第二参），不是完整 state；
#       想在 worker 里读公共字段必须由 fan-out 函数显式塞进去。
#    B. 并行写同一个字段必须带 reducer：同一 super-step 内写多次会抛 InvalidUpdateError
#       （不是"互相覆盖"——覆盖只发生在跨 super-step 的串行写入，Demo 1 注释里有反例）。
#    C. Command 只加动态边：同时存在静态出边时**两条路都会跑**（官方警告，已在 Demo 2 注释标明）。
#    D. Command 的去向建议用 `Command[Literal[...]]` 注解声明；goto 到 END 时用 END 常量
#       （它的节点名就是 "__end__"）。
#    E. interrupt() 不能被 try/except 包住，否则框架的暂停信号会被吞掉（官方 interrupts 规则）。
#    F. @task 只能在 @entrypoint 内调用；@task 的复用按 thread_id 隔离（Demo 3 已实测）。

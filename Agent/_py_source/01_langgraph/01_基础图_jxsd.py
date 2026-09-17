# -*- coding: utf-8 -*-
"""
LangGraph 基本概念：状态图（StateGraph）的四个元素
================================================================
LangGraph 把工作流抽象为一张**图（Graph）**，由四种元素组成：

  # | 概念           | 说明                                              |
  # |----------------|---------------------------------------------------|
  # | StateGraph     | 图本身，所有节点和边的容器                        |
  # | 节点（Node）   | 执行具体逻辑的函数，接收 state 返回更新           |
  # | 边（Edge）     | 连接两个节点，前一个输出直接传给后一个            |
  # | 条件边         | 根据返回值动态选择下一个节点                      |
  #    （Conditional Edge）

对应到本文件的代码：
  - StateGraph → MyState（流转的数据类型）+ builder（图的容器）
  - 节点       → step_one / step_two / big / small 各司其职
  - 普通边     → builder.add_edge()：固定 A → B
  - 条件边     → builder.add_conditional_edges()：运行时按返回值决定去向

数据怎么流动？记住一条规则：
  **节点不返回「完整状态」，只返回「增量」**。
  增量会被 LangGraph 合并（reduce）进全局状态，再传给下一个节点。
  合并规则由状态字段的类型决定：
    - 普通字段（`count: int`）      → 覆盖：后写的值直接盖掉先写的
    - Annotated + operator.add 字段 → 追加：新列表拼接到旧列表后面（见第 5 节）

课案出处：Agent 课案 → langgraph → 基本概念
运行方式：
    uv run Agent/01_langgraph/01_基础图_jxsd.py
前置条件：
    - 依赖：`langgraph`（本项目已 uv sync 装好），无第三方服务。
    - 配置：**本文件不读 .env、不连大模型**——`from config import settings` 不需要，
      整节只演示图本身的调度，所以是 01_langgraph 里唯一能离线秒跑的文件之一。
"""

import operator
import sys
from typing import Annotated, TypedDict

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，防中文/emoji 报错

from langgraph.graph import END, START, StateGraph

# ============================================================
# 1. 定义状态（图中流转的数据）
# ============================================================
class MyState(TypedDict):
    """课案原文的状态定义。

    TypedDict 只描述「有哪些字段、什么类型」，不产生运行时校验，
    它的作用是给编辑器 / LangGraph 提供字段清单和合并规则。
    """

    count: int  # 计数器：普通字段，节点返回什么就是什么（覆盖式）
    log: str  # 日志：普通字段，每个节点覆盖写入


# ============================================================
# 2. 定义节点函数（接收 state，返回更新的 state）
# ============================================================
def step_one(state: MyState) -> dict:
    """步骤一：count + 1，并把日志重置为「执行了步骤一」。"""
    # 注意只返回要改的字段。log 这里是覆盖，不是拼接——
    # 想拼接（保留历史）就得用 Annotated + operator.add，见第 5 节。
    return {"count": state["count"] + 1, "log": "执行了步骤一"}


def step_two(state: MyState) -> dict:
    """步骤二：count × 2，并把新日志拼在旧日志后面（手工拼字符串）。"""
    return {"count": state["count"] * 2, "log": state["log"] + " → 步骤二"}


def choose_path(state: MyState) -> str:
    """条件边的路由函数。

    ⚠️ 路由函数只做一件事：**返回「下一个节点的名字」**（字符串）。
    它不执行跳转，也不碰状态——真正的跳转由 LangGraph 按
    add_conditional_edges 里给的映射表完成。
    """
    # 条件边：根据 count 值决定走哪条路
    return "big" if state["count"] > 10 else "small"


def big_handler(state: MyState) -> dict:
    """大数分支：count > 10 时走这里。"""
    return {"log": f"count={state['count']}, 太大了"}


def small_handler(state: MyState) -> dict:
    """小数分支：count <= 10 时走这里。"""
    return {"log": f"count={state['count']}, 很小"}


# ============================================================
# 3. 构建图（课案原文的组装流程）
# ============================================================
builder = StateGraph(MyState)

builder.add_node("step_one", step_one)  # 添加节点：节点名 "step_one" → 函数 step_one
builder.add_node("step_two", step_two)
builder.add_node("big", big_handler)
builder.add_node("small", small_handler)

builder.add_edge(START, "step_one")  # 普通边：START → step_one（START 是图的虚拟入口）
builder.add_edge("step_one", "step_two")  # 普通边：step_one → step_two
builder.add_edge("big", END)  # big → 结束（END 是图的虚拟出口）
builder.add_edge("small", END)  # small → 结束

# ---------- 3.1 条件边：本节的绝对重点 ----------
# add_conditional_edges 的三个参数，课案里给了「列表」写法，
# 这里特意用「映射字典」写法，把两者的区别讲透：
#
#   builder.add_conditional_edges("step_two", choose_path, {"big": "big", "small": "small"})
#                                 ↑ 源节点    ↑ 路由函数   ↑ 路由函数返回值 → 真实节点名
#
# 三种参数形式对比（同一个意思，写法不同）：
#   ① 列表：add_conditional_edges("step_two", choose_path, ["big", "small"])
#      → 路由函数返回 "big"，就必须存在一个**同名节点** "big"；适合「返回值=节点名」
#   ② 映射字典（推荐）：add_conditional_edges("step_two", choose_path, {"big": "big", "small": "small"})
#      → 左边是路由函数的返回值，右边是真正要跳的节点名；
#        返回值可以和节点名不一样，例如 {"big": "handle_large"}，语义更自由
#   ③ 不传第三个参数：LangGraph 按路由函数的类型标注/返回值自行推断目标
#      → 少写代码，但图结构不够一目了然，教学里不推荐
#
# 第三个参数还有一个隐藏作用：**声明这笔分叉可能去哪些地方**，
# 于是 graph.get_graph() 才能把条件边画出来（否则图是残缺的）。
builder.add_conditional_edges(
    "step_two",
    choose_path,  # 路由函数，返回目标节点名
    {"big": "big", "small": "small"},  # 返回值 → 节点映射
)

graph = builder.compile()  # 编译：把 builder 变成可执行的图（编译后不能再改结构）


# ============================================================
# 4. 运行（课案原文的 invoke）
# ============================================================
def run_course_example() -> None:
    """课案原文示例：count=1 → +1=2 → ×2=4 → small → "count=4, 很小"。"""
    print("=" * 72)
    print("① 课案原文示例：invoke({'count': 1, 'log': ''})")
    print("=" * 72)
    print("  数据流：count=1 —step_one→ 2 —step_two→ 4 —choose_path→ small")
    result = graph.invoke({"count": 1, "log": ""})
    print("  返回：", result)
    # 预期输出：{'count': 4, 'log': 'count=4, 很小'}
    print()


def run_big_branch() -> None:
    """换个初值走 big 分支：count=6 → +1=7 → ×2=14 > 10 → big。"""
    print("=" * 72)
    print("② 换一个初值，让条件边走另一条分叉：invoke({'count': 6, 'log': ''})")
    print("=" * 72)
    print("  数据流：count=6 —step_one→ 7 —step_two→ 14 > 10 —choose_path→ big")
    result = graph.invoke({"count": 6, "log": ""})
    print("  返回：", result)
    # 预期输出：{'count': 14, 'log': 'count=14, 太大了'}
    print()


def show_graph_structure() -> None:
    """把图的结构打印出来——让「节点/普通边/条件边」看得见摸得着。"""
    print("=" * 72)
    print("③ 图的结构（get_graph()）：普通边与条件边的区别一目了然")
    print("=" * 72)
    net = graph.get_graph()
    for edge in sorted(net.edges, key=lambda e: (e.source, e.target)):
        kind = "条件边" if edge.conditional else "普通边"
        print(f"  [{kind}] {edge.source:<10} → {edge.target}")
    print()


# ============================================================
# 5. 补充：Annotated + operator.add —— 「追加式」字段
# ============================================================
# 上面 MyState 里的 log 是**覆盖式**的：step_one 写 "执行了步骤一"，
# step_two 写 "执行了步骤一 → 步骤二"——之所以还能看到历史，
# 是因为我们在节点里**手工**把它拼了起来（state["log"] + " → 步骤二"）。
#
# 一旦节点变多，每个节点都要手工拼接，又啰嗦又容易漏。
# LangGraph 提供了「声明式」的解法：用 Annotated 给字段挂一个**合并函数**（reducer），
# 让框架自动完成合并。
class ChatState(TypedDict):
    """演示追加式字段的状态。

    | 写法                                  | 合并行为                | 典型用途          |
    |---------------------------------------|-------------------------|-------------------|
    | title: str                            | 覆盖：新值盖旧值        | 当前状态类字段    |
    | history: Annotated[list, operator.add]| 追加：新旧列表拼接      | 对话历史、执行日志|
    | messages: Annotated[list, add_messages]| 追加 + 按 id 去重/更新 | LangGraph 内置    |

    注意 `operator.add` 作用在 list 上就是 `[] + []`（列表拼接），
    作用在 int/str 上则是数值相加 / 字符串相连——所以 reducer 选错类型会出怪事。
    """

    title: str  # 普通字段：覆盖
    history: Annotated[list[str], operator.add]  # 追加式字段：自动 concat


def node_a(state: ChatState) -> dict:
    """节点 A：只返回「我这一条」日志，不操心拼接。"""
    # 对比上面 step_two 的手工拼接——这里只写自己新增的部分
    return {"title": "A 处理后的标题", "history": ["经过节点A"]}


def node_b(state: ChatState) -> dict:
    """节点 B：同样只返回自己那一条。"""
    return {"title": "B 处理后的标题", "history": ["经过节点B"]}


annotated_builder = StateGraph(ChatState)
annotated_builder.add_node("a", node_a)
annotated_builder.add_node("b", node_b)
annotated_builder.add_edge(START, "a")
annotated_builder.add_edge("a", "b")
annotated_builder.add_edge("b", END)
annotated_graph = annotated_builder.compile()


def run_annotated_demo() -> None:
    print("=" * 72)
    print("④ 补充：Annotated + operator.add 的「追加式」字段")
    print("=" * 72)
    result = annotated_graph.invoke({"title": "初始标题", "history": ["图开始执行"]})
    print("  返回：", result)
    # 预期输出：
    #   title   → "B 处理后的标题"（覆盖式：A 写的被 B 盖掉了，只剩最后一句）
    #   history → ['图开始执行', '经过节点A', '经过节点B']（追加式：一条都没丢）
    print()
    print("  对照结论：同一个状态里，title 走覆盖、history 走追加；")
    print("  差别**只来自字段的声明方式**，节点函数本身的写法完全一样。")
    print()


if __name__ == "__main__":
    run_course_example()
    run_big_branch()
    show_graph_structure()
    run_annotated_demo()


# ============================================================
# 6. 实测结论 · 与本课案的差异 · 踩坑提示
# ============================================================
# 【实测结论】课案那句注释 `print(result) # count=1 → +1=2 → ×2=4 → small → "count=4, 很小"`
# 本机结果完全一致：{'count': 4, 'log': 'count=4, 很小'}。
# 换初值 6 时（6+1=7，7×2=14 > 10）也确实走 big，返回 {'count': 14, 'log': 'count=14, 太大了'}。
# 第 ④ 节额外实证了 reducer：同一个 ChatState 里 title 被覆盖、history 被追加，
# 差别**只来自字段声明方式**，节点函数写法一模一样。
#
# 【与本课案的差异】
#   1. 课案的条件边示例用的是 `{"big": "big", "small": "small"}` 映射字典写法，
#      本文件保持原样；但第 3.1 节额外补了列表写法 ["big", "small"] 的对比，
#      因为两者在「返回值能不能不同于节点名」这一点上行为不同，最容易踩坑。
#   2. 课案原文**没有** Annotated + operator.add 这一段（第 5 节），
#      它是本文件补的：讲清「覆盖 vs 追加」这对 reducer 概念，
#      为后面 02/03 节的 MessagesState（messages 靠 add_messages 追加）提前铺路。
#   3. 课案的例子里 MyState 只有 count / log 两个普通字段，
#      所以每次都是覆盖；想保留历史就得像第 5 节那样声明 reducer。
#      这不是课案写错，而是「基础概念一节只讲最朴素的形态」。
#
# 【踩坑提示】
#   1. 节点函数返回的必须是 **dict（增量）**，不是完整的 state。
#      返回整个 state 不会报错，但会让 reducer 语义变得难以预料。
#   2. `builder.compile()` 之后图的结构就冻结了：再 add_node / add_edge 不会生效，
#      也不会报错——要改结构必须改 builder 后重新 compile。
#   3. 条件边的路由函数**只返回名字**。它不执行跳转、也改不了状态；
#      真正的跳转由 LangGraph 查 add_conditional_edges 的映射表完成（第 3.1 节）。
#   4. 状态字段没声明却返回同名键，会被 LangGraph 静默丢弃（TypedDict 不做运行时校验），
#      排查「数据莫名不见了」时先回来核对 schema。
#   5. `graph.get_graph()` 画出的边依赖第三个参数：不传映射/列表时，
#      条件边在图上是**残缺**的——这是「打印图结构」时最常见的困惑来源。

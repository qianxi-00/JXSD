# -*- coding: utf-8 -*-
"""
LangGraph 基础：状态图（StateGraph）
================================================================
LangGraph 的核心思想：把智能体建模为一张「图」。
- 节点（Node）  ：一个 Python 函数，接收状态、返回对状态的更新
- 边（Edge）    ：决定节点之间的跳转关系（固定边 / 条件边）
- 状态（State） ：在整个图中流转的共享数据（通常用 TypedDict 定义）
- 入口/出口    ：START / END，标记图的起点和终点

运行方式：
    uv run 01_langgraph/01_基础图.py
"""

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


# ---------- 1. 定义状态 ----------
class State(TypedDict):
    """
    状态就是一个 TypedDict，所有节点共享。
    - messages：普通覆盖式字段（后写的值直接覆盖先写的）
    - history ：使用 Annotated + operator.add 实现「追加式」字段
                每个节点返回的列表会被拼接（reduce）到原有列表后面
    """
    messages: str
    history: Annotated[list[str], operator.add]


# ---------- 2. 定义节点（就是普通函数） ----------
def node_a(state: State) -> dict:
    """节点 A：接收当前状态，返回对状态的「增量更新」（而不是全量状态）"""
    print(f"[节点A] 收到消息：{state['messages']}")
    # 只需要返回要更新的字段
    return {"messages": "A 已处理", "history": ["经过节点A"]}


def node_b(state: State) -> dict:
    """节点 B：可以读到前面节点写入的数据"""
    print(f"[节点B] 收到消息：{state['messages']}")
    return {"messages": "B 已处理", "history": ["经过节点B"]}


# ---------- 3. 条件边：根据状态动态决定下一个节点 ----------
def route(state: State) -> str:
    """返回值是「下一个节点」的名字，也可以返回 END 表示结束"""
    if "B" in state["messages"]:
        return END
    return "b"


# ---------- 4. 组装图 ----------
builder = StateGraph(State)

# 注册节点
builder.add_node("a", node_a)
builder.add_node("b", node_b)

# 固定边：START -> a
builder.add_edge(START, "a")
# 条件边：a 之后走 route 函数决定去向
builder.add_conditional_edges("a", route, ["b", END])
# 固定边：b -> END
builder.add_edge("b", END)

# 编译：把 builder 变成可执行的图
graph = builder.compile()

# ---------- 5. 执行 ----------
if __name__ == "__main__":
    # Windows 控制台默认 GBK 编码，强制切换为 UTF-8 防止中文乱码
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    # invoke：一次性执行完整张图，返回最终状态
    result = graph.invoke({"messages": "你好"})
    print("最终状态：", result)

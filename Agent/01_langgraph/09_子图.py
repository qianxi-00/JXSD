# -*- coding: utf-8 -*-
"""
LangGraph 子图（Subgraph）
================================================================
把复杂大图拆成多个小图（子图），每个子图负责独立功能，
最后组合成大图——这是管理复杂 Agent 的标准做法。

要点：
    1. 子图先 compile()，然后作为一个节点 add_node 进父图
    2. 父图和子图可以共享同一个 State，数据无缝传递
    3. 子图同样可以有自己的 checkpointer / 中断 / 记忆

运行方式：
    uv run 01_langgraph/09_子图.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    question: str   # 用户问题
    plan: str       # 子图产出的计划
    answer: str     # 最终答案


# ---------- 子图：规划器 ----------
def planner(state: State) -> dict:
    """规划节点：假装调用 LLM 生成执行计划"""
    print(f"[规划器] 分析问题：{state['question']}")
    return {"plan": f"1. 检索资料 2. 归纳要点（针对：{state['question']}）"}


planner_builder = StateGraph(State)
planner_builder.add_node("planner", planner)
planner_builder.add_edge(START, "planner")
planner_builder.add_edge("planner", END)

# 子图先编译，再当节点使用
planner_graph = planner_builder.compile()


# ---------- 父图 ----------
def executor(state: State) -> dict:
    """执行节点：读取子图写入 plan，产出答案"""
    print(f"[执行器] 按计划执行：{state['plan']}")
    return {"answer": "（已按计划完成的答案）"}


def reviewer(state: State) -> dict:
    print(f"[审核] 检查答案：{state['answer']}")
    return {}


main_builder = StateGraph(State)
# 关键：子图作为一个普通节点加进父图
main_builder.add_node("planner", planner_graph)
main_builder.add_node("executor", executor)
main_builder.add_node("reviewer", reviewer)
main_builder.add_edge(START, "planner")
main_builder.add_edge("planner", "executor")
main_builder.add_edge("executor", "reviewer")
main_builder.add_edge("reviewer", END)

main_graph = main_builder.compile()

if __name__ == "__main__":
    result = main_graph.invoke({"question": "LangGraph 是什么？"})
    print("最终结果：", result)

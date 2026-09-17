# -*- coding: utf-8 -*-
"""
LangGraph 时间旅行（Time Travel）
================================================================
检查点保存了每一轮的完整状态，因此可以：
    1. get_state_history(config) 遍历全部历史快照
    2. 选中某一个历史快照，从那里「改写历史」重新执行

用途：调试 Agent 行为、对比不同决策分支的结果、回滚重来。

运行方式：
    uv run 01_langgraph/08_时间旅行.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


def step_1(state: dict) -> dict:
    return {"value": state["value"] + 1, "log": ["step_1"]}


def step_2(state: dict) -> dict:
    return {"value": state["value"] * 10, "log": ["step_2"]}


builder = StateGraph(dict)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", END)

graph = builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "tt-1"}}

# 正常执行一轮
result = graph.invoke({"value": 1, "log": []}, config)
print("正常执行结果：", result)

# ---------- 1. 查看全部历史快照 ----------
snapshots = list(graph.get_state_history(config))
for snap in snapshots:
    print(f"step={snap.metadata.get('step')}  值={snap.values}")

# ---------- 2. 选一个历史快照，从那里重新开始 ----------
# 选 step_1 之后的快照（value==2），修改输入后重新执行
# 注意：起始快照的 values 可能为 None，需要用 (s.values or {}) 防御
old_snapshot = [s for s in snapshots if (s.values or {}).get("value") == 2][0]
new_config = graph.update_state(
    old_snapshot.config,
    values={"value": 100},  # 改写历史状态
)

# 从这个快照继续执行（None 表示从当前位置继续跑后面的节点）
result = graph.invoke(None, new_config)
print("时间旅行后的结果：", result)

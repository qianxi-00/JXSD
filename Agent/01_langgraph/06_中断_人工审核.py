# -*- coding: utf-8 -*-
"""
LangGraph 中断（Interrupt）：人工审核 Human-in-the-loop
================================================================
敏感操作（删库、转账、发邮件……）不应让 Agent 自作主张。
做法：在节点里调用 interrupt()，图会在那里暂停，
把「待审核信息」抛给人类；人类通过 Command 决定：
    - 同意   ：Command(resume=True)  → 从断点继续执行
    - 拒绝   ：Command(resume=错误反馈) → 节点内可以拿到反馈并走拒绝分支

关键点：
    1. 编译时必须传 checkpointer（中断状态要靠检查点保存）
    2. interrupt() 会在重放时返回 Command(resume=...) 传入的值

运行方式：
    uv run 01_langgraph/06_中断_人工审核.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    task: str        # 待执行的任务
    approved: bool   # 是否被人工批准


def human_review(state: State) -> dict:
    """
    人工审核节点。
    interrupt(payload) 暂停图执行，payload 会出现在
    __interrupt__ 事件里，展示给人类审核。
    """
    answer = interrupt(
        {"question": f"要执行任务「{state['task']}」，请确认是否批准？(yes/no)"}
    )
    # 恢复执行后，interrupt() 的返回值就是 Command(resume=...) 传的值
    return {"approved": answer in (True, "yes")}


def execute(state: State) -> dict:
    """执行节点：只有批准了才会真正走到这里"""
    if state["approved"]:
        print(f"✅ 已批准，正在执行：{state['task']}")
    else:
        print(f"❌ 人类已拒绝，任务「{state['task']}」不会执行")
    return {}


builder = StateGraph(State)
builder.add_node("human_review", human_review)
builder.add_node("execute", execute)
builder.add_edge(START, "human_review")
# 条件边：批准 → 执行；拒绝 → 直接结束
builder.add_conditional_edges(
    "human_review",
    lambda s: "execute" if s.get("approved") else END,
    ["execute", END],
)
builder.add_edge("execute", END)

# 中断必须配合 checkpointer
graph = builder.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "review-1"}}

    # 第一次 invoke：会停在 interrupt 处
    result = graph.invoke({"task": "删除生产数据库里的测试表"}, config)
    # 停在断点时，结果里带 __interrupt__ 信息
    print("暂停，等待审核：", result["__interrupt__"])

    # 人类审核：拒绝（也可以给文字反馈，让 Agent 调整后重试）
    result = graph.invoke(Command(resume="no"), config)
    print("拒绝后结果：", result)

    # ---------- 再来一次，这次批准 ----------
    config = {"configurable": {"thread_id": "review-2"}}
    graph.invoke({"task": "清理过期缓存"}, config)
    result = graph.invoke(Command(resume="yes"), config)
    print("批准后结果：", result)

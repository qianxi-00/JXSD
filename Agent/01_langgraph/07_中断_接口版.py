# -*- coding: utf-8 -*-
"""
LangGraph 中断：接口（HTTP）版本
================================================================
实际项目中，人工审核不是在脚本里 resume，而是：
    1. 用户调接口发起任务 → 图跑到 interrupt 处挂起
    2. 审核人调另一个接口 → Command(resume=...) 恢复执行

启动服务：
    uv run Agent/01_langgraph/07_中断_接口版.py     # 默认 8000 端口

接口：
    POST /task   发起任务（会在审核处暂停，返回 __interrupt__）
    POST /review 提交审核结果（resume）
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from typing import TypedDict

from fastapi import FastAPI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

app = FastAPI(title="人工审核示例")


class State(TypedDict):
    task: str
    approved: bool


def human_review(state: State) -> dict:
    answer = interrupt({"question": f"是否批准任务：{state['task']}？"})
    return {"approved": answer in (True, "yes", "True")}


def execute(state: State) -> dict:
    print(f"执行任务：{state['task']}")
    return {}


builder = StateGraph(State)
builder.add_node("human_review", human_review)
builder.add_node("execute", execute)
builder.add_edge(START, "human_review")
builder.add_conditional_edges(
    "human_review",
    lambda s: "execute" if s.get("approved") else END,
    ["execute", END],
)
builder.add_edge("execute", END)

graph = builder.compile(checkpointer=MemorySaver())


class TaskRequest(BaseModel):
    thread_id: str
    task: str


class ReviewRequest(BaseModel):
    thread_id: str
    approved: bool


@app.post("/task")
def create_task(req: TaskRequest):
    """发起任务；如果触发审核，返回中断信息给前端展示审核弹窗"""
    config = {"configurable": {"thread_id": req.thread_id}}
    result = graph.invoke({"task": req.task}, config)
    if "__interrupt__" in result:
        return {"status": "waiting_review", "interrupt": result["__interrupt__"]}
    return {"status": "done"}


@app.post("/review")
def review(req: ReviewRequest):
    """提交审核结果，恢复被中断的图"""
    config = {"configurable": {"thread_id": req.thread_id}}
    result = graph.invoke(Command(resume=req.approved), config)
    return {"status": "done", "result": {k: v for k, v in result.items() if k != "__interrupt__"}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)

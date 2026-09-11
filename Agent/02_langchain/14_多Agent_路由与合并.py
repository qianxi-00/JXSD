# -*- coding: utf-8 -*-
"""
LangChain 多 Agent：路由与合并
================================================================
「路由 + 合并」模式：
    - 路由（fan-out）：一个调度节点按任务拆分，分发给多个专家并行处理
    - 合并（fan-in) ：汇总节点收集所有专家的结果，合并成最终答案

典型场景：多维度分析（同一问题让「技术 / 商业 / 风险」三个专家同时分析）。

运行方式：
    uv run 02_langchain/14_多Agent_路由与合并.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import operator
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


class State(TypedDict):
    question: str
    # Annotated + add：多个专家的输出会「合并」到同一个列表里（fan-in 关键）
    opinions: Annotated[list[str], operator.add]
    summary: str


# ---------- 路由：拆任务 ----------
def router(state: State) -> dict:
    """调度节点：准备分发给专家的任务"""
    print(f"[路由] 分发任务：{state['question']}")
    return {}


# ---------- 三个专家（并行节点） ----------
def tech_expert(state: State) -> dict:
    r = llm.invoke(f"从技术角度一句话评价：{state['question']}")
    return {"opinions": [f"技术专家：{r.content}"]}


def biz_expert(state: State) -> dict:
    r = llm.invoke(f"从商业角度一句话评价：{state['question']}")
    return {"opinions": [f"商业专家：{r.content}"]}


def risk_expert(state: State) -> dict:
    r = llm.invoke(f"从风险角度一句话评价：{state['question']}")
    return {"opinions": [f"风险专家：{r.content}"]}


# ---------- 合并：汇总 ----------
def merge(state: State) -> dict:
    joined = "\n".join(state["opinions"])
    r = llm.invoke(f"综合以下多方意见，给出一句总结：\n{joined}")
    return {"summary": r.content}


builder = StateGraph(State)
builder.add_node("router", router)
builder.add_node("tech", tech_expert)
builder.add_node("biz", biz_expert)
builder.add_node("risk", risk_expert)
builder.add_node("merge", merge)

builder.add_edge(START, "router")
# fan-out：router 之后三条边 = 三个专家并行执行
builder.add_edge("router", "tech")
builder.add_edge("router", "biz")
builder.add_edge("router", "risk")
# fan-in：三条汇入边，等全部专家完成后再进 merge
builder.add_edge("tech", "merge")
builder.add_edge("biz", "merge")
builder.add_edge("risk", "merge")
builder.add_edge("merge", END)

graph = builder.compile()

if __name__ == "__main__":
    result = graph.invoke({"question": "用 Rust 重写核心服务是否值得？", "opinions": []})
    print("\n".join(result["opinions"]))
    print("综合结论：", result["summary"])

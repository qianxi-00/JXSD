# -*- coding: utf-8 -*-
"""
LangChain 多 Agent：交接（Handoff）
================================================================
交接模式：Agent A 判断问题属于 Agent B 的领域，
就把「对话控制权」移交给 B（A 停止，B 接管后续对话）。
适合客服转接、工单分诊场景。

LangChain 1.4 中的标准实现：把每个专家做成图上的节点，
用「分诊节点 + 条件边」完成控制权移交（等效于 handoff 工具）。

运行方式：
    uv run Agent/02_langchain/13_多Agent_交接.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


# ---------- 各自的工具 ----------
@tool
def query_bill() -> str:
    """查询用户账单"""
    return "本月账单：话费 99 元"


@tool
def repair_network() -> str:
    """远程检修网络"""
    return "网络已重置，恢复正常"


# ---------- 1. 两个专家（各自带工具的智能体） ----------
billing_agent = create_react_agent(
    llm,
    [query_bill],
    prompt="你是账务客服，只处理账单问题，回答要简短。",
)

network_agent = create_react_agent(
    llm,
    [repair_network],
    prompt="你是网络客服，只处理网络问题，回答要简短。",
)


# ---------- 2. 分诊台：决定把对话移交给谁 ----------
def supervisor(state: MessagesState):
    """前台：只判断交给谁（这一步就是「交接」的决策点）"""
    response = llm.invoke(
        [
            ("system", "你是分诊台。判断用户问题属于哪类，只回复一个词：billing（账务）或 network（网络）"),
            *state["messages"],
        ]
    )
    target = "billing_agent" if "billing" in response.content.lower() else "network_agent"
    print(f"[分诊台] 交接给 -> {target}")
    return {"messages": [response]}


def handoff_billing(state: MessagesState):
    """交接到账务专家：以独立身份处理整段对话"""
    result = billing_agent.invoke({"messages": state["messages"]})
    return {"messages": [result["messages"][-1]]}


def handoff_network(state: MessagesState):
    """交接到网络专家"""
    result = network_agent.invoke({"messages": state["messages"]})
    return {"messages": [result["messages"][-1]]}


# ---------- 3. 组装图：分诊台按领域移交控制权 ----------
builder = StateGraph(MessagesState)
builder.add_node("supervisor", supervisor)
builder.add_node("billing_agent", handoff_billing)
builder.add_node("network_agent", handoff_network)
builder.add_edge(START, "supervisor")
# 条件边：分诊结果决定控制权移交给哪个专家
builder.add_conditional_edges(
    "supervisor",
    lambda s: "billing_agent" if any("billing" in m.content.lower() for m in [s["messages"][-1]]) else "network_agent",
    ["billing_agent", "network_agent"],
)
builder.add_edge("billing_agent", END)
builder.add_edge("network_agent", END)

graph = builder.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    # 例 1：网络问题 → 移交给网络专家
    r1 = graph.invoke(
        {"messages": [("user", "我家断网了，帮我修一下")]},
        config={"configurable": {"thread_id": "ho-1"}},
    )
    print("最终回复：", r1["messages"][-1].content)

    # 例 2：账单问题 → 移交给账务专家
    r2 = graph.invoke(
        {"messages": [("user", "帮我查一下这个月话费")]},
        config={"configurable": {"thread_id": "ho-2"}},
    )
    print("最终回复：", r2["messages"][-1].content)

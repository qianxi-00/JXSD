# -*- coding: utf-8 -*-
"""
LangGraph 短期记忆：内存中的 Checkpointer
================================================================
默认情况下图执行完状态就丢了，无法多轮对话。
给图挂上 checkpointer（检查点）后，每一轮的完整状态都会被保存下来，
同一 thread_id 的对话就能接续上下文——这就是「短期记忆」。

- MemorySaver：把检查点保存在内存里，适合开发调试，重启即失
- 生产环境请用 langgraph-checkpoint-postgres 提供的 PostgresSaver

运行方式：
    uv run 01_langgraph/02_短期记忆_内存.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台 UTF-8，防止中文乱码

from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from config import settings

# 初始化大模型（OpenAI 兼容接口）
llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


def chat(state: MessagesState) -> dict:
    """
    对话节点：MessagesState 是 LangGraph 内置状态，
    其中 messages 字段自动按「追加消息」的方式合并。
    """
    response = llm.invoke(state["messages"])
    # 返回的新消息会被追加到 messages 列表末尾
    return {"messages": [response]}


# ---------- 组装图 ----------
builder = StateGraph(MessagesState)
builder.add_node("chat", chat)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

# 关键点：编译时传入 checkpointer，开启短期记忆
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    # thread_id：会话 ID。相同 thread_id 共享一份记忆，
    # 换一个 thread_id 就是一条全新的、没有记忆的对话。
    config = {"configurable": {"thread_id": "demo-1"}}

    # 第一轮：告诉模型自己是谁
    r1 = graph.invoke({"messages": [("user", "你好，我叫小明，请记住我")]}, config)
    print("AI：", r1["messages"][-1].content)

    # 第二轮：同一 thread_id，模型记得上一轮内容
    r2 = graph.invoke({"messages": [("user", "我叫什么名字？")]}, config)
    print("AI：", r2["messages"][-1].content)

    # 第三轮：换一个 thread_id，记忆清空，模型不认识小明
    r3 = graph.invoke(
        {"messages": [("user", "我叫什么名字？")]},
        {"configurable": {"thread_id": "demo-2"}},
    )
    print("AI（新会话）：", r3["messages"][-1].content)

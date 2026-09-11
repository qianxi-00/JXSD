# -*- coding: utf-8 -*-
"""
LangGraph 流式输出
================================================================
大模型响应慢，全量等待体验差。LangGraph 支持 4 种流式模式：

    stream_mode="values"   每个节点执行后，流出完整状态值
    stream_mode="updates"  每个节点执行后，只流出该节点的增量更新
    stream_mode="messages" 流出 LLM 的 token（打字机效果）
    stream_mode="custom"   节点内自定义 get_stream_writer() 推送内容

多种模式可同时开启，事件会带 (模式, 数据) 元组一起流出。

运行方式：
    uv run 01_langgraph/05_流式输出.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, MessagesState, StateGraph
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
    streaming=True,  # 开启流式
)


def chat(state: MessagesState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


builder = StateGraph(MessagesState)
builder.add_node("chat", chat)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)
graph = builder.compile()


def stream_values():
    """values：每步流出完整状态"""
    print("===== stream_mode=values =====")
    for chunk in graph.stream(
        {"messages": [("user", "用一句话介绍上海")]},
        stream_mode="values",
    ):
        print("完整状态最后一条消息：", chunk["messages"][-1].content)


def stream_updates():
    """updates：每步只流出节点增量"""
    print("===== stream_mode=updates =====")
    for chunk in graph.stream(
        {"messages": [("user", "用一句话介绍北京")]},
        stream_mode="updates",
    ):
        print("节点增量：", chunk)


def stream_messages():
    """messages：token 级流式，打字机效果"""
    print("===== stream_mode=messages =====")
    for mode, chunk in graph.stream(
        {"messages": [("user", "讲一个 50 字左右的笑话")]},
        stream_mode=["messages", "updates"],
    ):
        if mode == "messages":
            # 新版 LangGraph 的 messages 流返回 (消息块, 元数据) 元组
            if isinstance(chunk, tuple):
                chunk = chunk[0]
            # chunk 是 AIMessageChunk，一个 token
            content = getattr(chunk, "content", "")
            if isinstance(content, list):  # 部分模型的 content 是分块列表
                content = "".join(str(c) for c in content)
            if content:
                print(content, end="", flush=True)
        else:
            print()  # 节点结束时换行


if __name__ == "__main__":
    stream_values()
    stream_updates()
    stream_messages()

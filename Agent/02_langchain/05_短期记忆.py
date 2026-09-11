# -*- coding: utf-8 -*-
"""
LangChain 智能体：短期记忆（thread_id 隔离）
================================================================
create_agent 的智能体同样支持 checkpointer 短期记忆：
    agent.compile(checkpointer=...) 之后，
    相同 thread_id 的对话自动带上历史上下文。

运行方式：
    uv run 02_langchain/05_短期记忆.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="你是一个记账助手，帮用户记录并回答账目问题。",
    # LangChain 1.x 支持在 create_agent 里直接配 checkpointer
    checkpointer=MemorySaver(),
)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "bill-001"}}

    r1 = agent.invoke({"messages": [("user", "今天午饭花了 30 元")]}, config)
    print("AI：", r1["messages"][-1].content)

    r2 = agent.invoke({"messages": [("user", "我今天一共花了多少钱？")]}, config)
    print("AI：", r2["messages"][-1].content)

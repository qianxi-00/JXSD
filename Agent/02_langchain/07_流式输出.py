# -*- coding: utf-8 -*-
"""
LangChain 智能体：流式输出
================================================================
两种流式：
    1. 消息流（token 打字机）：stream_mode="messages"
    2. 步骤流：stream_mode="updates"，看到每个节点/工具的进度

运行方式：
    uv run 02_langchain/07_流式输出.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
    streaming=True,
)


@tool
def get_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


agent = create_agent(model=llm, tools=[get_time])

if __name__ == "__main__":
    print("===== token 流式 =====")
    for token, metadata in agent.stream(
        {"messages": [("user", "现在几点了？顺便问一下你是什么模型")]},
        stream_mode="messages",
    ):
        # 只打印 AI 的 token（过滤掉工具消息）
        if token.content and metadata.get("langgraph_node") == "model":
            print(token.content, end="", flush=True)
    print()

    print("===== 步骤流式 =====")
    for update in agent.stream(
        {"messages": [("user", "现在几点了？")]},
        stream_mode="updates",
    ):
        print(update)

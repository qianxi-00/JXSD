# -*- coding: utf-8 -*-
"""
DeepAgents 记忆（MemoryMiddleware）
================================================================
DeepAgents 的记忆 = 一个特殊的「记忆文件夹」：
Agent 自己决定什么值得记，通过 write_file 写进 /memories/ 目录，
每次新会话开始时自动把记忆注入系统提示词。

记忆跨会话生效的前提：后端要持久化（StoreBackend / FilesystemBackend）。

memory=["xxx.md"]：指定记忆文件的说明，引导 Agent 记什么。

运行方式：
    uv run 03_deepagents/11_记忆.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain.chat_models import init_chat_model
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

agent = create_deep_agent(
    model=llm,
    backend=StateBackend(),  # 生产环境换 StoreBackend 让记忆真正持久化
    system_prompt="你是贴心的私人助手。",
    # 记忆说明：告诉 Agent 应该记住哪些方面的信息
    memory=["用户偏好.md：记录用户的称呼、喜好、忌讳，每次会话结束前更新"],
)

if __name__ == "__main__":
    # 第一轮：告知偏好 → Agent 写入 /memories/
    r1 = agent.invoke(
        {"messages": [("user", "我叫小明，喜欢简洁回答，讨厌表情包")]},
        config={"configurable": {"thread_id": "mem-1"}, "recursion_limit": 50},
    )
    print("会话1 AI：", r1["messages"][-1].content)
    print("记忆文件：", list(r1.get("files", {}).keys()))

    # 第二轮（新 thread_id）：记忆被注入 → Agent 直接按偏好行事
    r2 = agent.invoke(
        {"messages": [("user", "介绍一下你自己")]},
        config={"configurable": {"thread_id": "mem-2"}, "recursion_limit": 50},
    )
    print("会话2 AI：", r2["messages"][-1].content)

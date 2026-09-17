# -*- coding: utf-8 -*-
"""
DeepAgents 基础：create_deep_agent
================================================================
DeepAgents = 「深度智能体」框架，对标 Claude Code 的架构。
它不只是一个会调工具的 Agent，而是自带一整套「智能体操作系统」：

    - 虚拟文件系统（backend）：ls / read_file / write_file / edit_file
    - Shell 执行器（execute）：执行命令
    - 子智能体（task）       ：把任务委派给专门的子 Agent
    - 任务规划（todo）       ：自动维护任务清单
    - 长上下文管理           ：自动摘要、裁剪历史

一个 create_deep_agent 调用即可获得以上全部能力。

运行方式：
    uv run 03_deepagents/01_智能体.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


@tool
def internet_search(query: str) -> str:
    """联网搜索。query：搜索关键词"""
    # 演示用假数据；生产环境可接 Tavily / SerpAPI 等
    return f"搜索「{query}」的结果：LangGraph 是用于构建智能体的图编排框架……"


if __name__ == "__main__":
    agent = create_deep_agent(
        model=llm,
        tools=[internet_search],
        system_prompt="你是一个研究助手。先搜索资料，把笔记写入文件再总结。",
    )

    result = agent.invoke(
        {"messages": [("user", "调研一下 LangGraph 是什么，写成调研笔记")]},
        config={"recursion_limit": 50},  # 深度智能体步数多，放宽步数上限
    )
    print(result)
    print("AI：", result["messages"][-1].content)

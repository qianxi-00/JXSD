# -*- coding: utf-8 -*-
"""
DeepAgents 子智能体（SubAgents）
================================================================
主 Agent 可以通过 task 工具把子任务委派给子 Agent：
每个子 Agent 有独立的系统提示词、独立工具、独立上下文窗口。

好处：
    - 上下文隔离：子 Agent 的大量中间信息不会污染主 Agent 的上下文
    - 职责单一：每个子 Agent 只擅长一件事

用法：subagents=[{"name": ..., "description": ..., "system_prompt": ..., "tools": ...}]

运行方式：
    uv run 03_deepagents/12_子智能体.py
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
def fake_search(keyword: str) -> str:
    """搜索资料。keyword：关键词"""
    return f"「{keyword}」相关资料：LangGraph……（模拟搜索结果）"


research_agent = create_deep_agent(
    model=llm,
    tools=[fake_search],
    system_prompt="你是总编，调研任务委派给 researcher 子智能体，然后汇总成一句话结论。",
    subagents=[
        {
            "name": "researcher",
            "description": "资料调研专家，擅长搜索和整理资料",  # 给主 Agent 看的「外包说明」
            "system_prompt": "你是调研专家，用搜索工具收集资料，输出要点列表。",
            "tools": [fake_search],   # 子 Agent 独立工具集
            "model": llm,             # 子 Agent 可指定不同模型（传实例最稳妥）
        },
        # 可以继续添加更多子智能体，如 coder / writer / reviewer …
    ],
)

if __name__ == "__main__":
    result = research_agent.invoke(
        {"messages": [("user", "调研一下 LangGraph 并给我一个简短结论")]},
        config={"recursion_limit": 80},
    )
    print("AI：", result["messages"][-1].content)

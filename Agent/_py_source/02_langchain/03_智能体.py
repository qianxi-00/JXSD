# -*- coding: utf-8 -*-
"""
LangChain 基础：智能体（create_agent）
================================================================
create_agent 是 LangChain 1.x 封装好的智能体：
内部自动实现了「模型 ↔ 工具」的循环（ReAct 模式）：
    用户提问 → 模型判断是否需要工具 → 调工具 → 结果回传 → 再生成回复

只要三样东西：
    model   ：大模型
    tools   ：工具列表
    prompt  ：系统提示词（人设 / 规则）

运行方式：
    uv run 02_langchain/03_智能体.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


# ---------- 1. 定义工具 ----------
@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。city：城市名称，如「上海」"""
    # 演示用：实际项目中这里会调真实天气 API
    return f"{city} 晴，25 度，适合出行"


# ---------- 2. 创建智能体 ----------
agent = create_agent(
    model=llm,
    tools=[get_weather],                       # 工具列表
    system_prompt="你是一个天气助手，回答前先查天气工具。",  # 人设
)

if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [("user", "上海今天天气怎么样？适合出去玩吗？")]}
    )

    # 打印完整执行轨迹：可以看到模型发起了工具调用，拿到结果后组织回答
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            print(f"[工具结果] {msg.content}")
        else:
            print(f"[{msg.type}] {msg.content}")

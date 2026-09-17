# -*- coding: utf-8 -*-
"""
Function Call ⑤：DeepAgents 版智能体（agent_deepagents.py）
================================================================
对比三个版本：
    - OpenAI SDK  ：手写完整循环，理解原理用
    - LangChain   ：create_agent，工具循环自动化，轻量够用
    - DeepAgents  ：在工具循环之上，再加文件系统 / 子智能体 / 任务规划，
                   适合复杂、长程、多步任务

运行方式：
    uv run 04_function_call/agent_deepagents.py
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
def get_weather(city: str) -> str:
    """查询指定城市的实时天气，包括天气状况和温度"""
    weather_map = {"上海": "晴 25 度", "北京": "多云 18 度", "广州": "阵雨 30 度"}
    return weather_map.get(city, f"{city} 天气未知")


agent = create_deep_agent(
    model=llm,
    tools=[get_weather],
    system_prompt=(
        "你是生活助手。多城市查询时先列任务清单（todo），"
        "把结果整理成表格文件保存，最后汇报。"
    ),
)

if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [("user", "对比一下上海、北京、广州三地天气，整理成表格")]},
        config={"recursion_limit": 50},
    )
    print("AI 最终回答：", result["messages"][-1].content)

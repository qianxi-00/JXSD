# -*- coding: utf-8 -*-
"""
Function Call ④：LangChain 版智能体（agent_langchain.py）
================================================================
对比 agent_openai.py 的手写循环，LangChain 一行搞定：
    create_agent(model, tools, system_prompt)

工具定义也从「函数 + 手写 JSON Schema」简化为一个 @tool 装饰器——
docstring 与类型注解自动生成描述。

运行方式：
    uv run 04_function_call/agent_langchain.py
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
)


# ---------- 用 @tool 定义工具：描述自动生成，无需手写 JSON Schema ----------
@tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气，包括天气状况和温度"""
    weather_map = {"上海": "晴 25 度", "北京": "多云 18 度", "广州": "阵雨 30 度"}
    return weather_map.get(city, f"{city} 天气未知")


@tool
def get_current_time() -> str:
    """获取当前的日期和时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


agent = create_agent(
    model=llm,
    tools=[get_weather, get_current_time],
    system_prompt="你是一个生活助手，查询天气和时间请使用工具。",
)

if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [("user", "上海现在天气怎么样？现在几点了？")]}
    )
    print("AI 最终回答：", result["messages"][-1].content)

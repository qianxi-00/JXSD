# -*- coding: utf-8 -*-
"""
LangChain 多 Agent：MCP 子智能体
================================================================
一个主 Agent 下挂多个「领域专家」子 Agent，每个子 Agent 通过
MCP 接入不同工具服务器（如：一个接天气服务、一个接数据库服务）。

架构：
    主 Agent（路由 + 汇总）
      ├── 子Agent：天气专家（tools 来自 MCP 天气服务）
      └── 子Agent：数据库专家（tools 来自 MCP 数据库服务）

MCP 工具加载：langchain-mcp-adapters 的 MultiServerMCPClient
    - stdio：拉起本地子进程（如 uv run xxx.py）
    - http ：连接已运行的 MCP 服务

运行方式：
    uv run 02_langchain/12_多Agent_MCP子Agent.py
    （先把 05_mcp/01_服务端.py 跑起来，或用 stdio 方式自动拉起）
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

# MCP 服务端脚本路径：基于本文件位置解析，避免受运行目录影响
import sys
from pathlib import Path  # noqa: E402

MCP_SERVER = str(Path(__file__).resolve().parents[1] / "05_mcp" / "01_服务端.py")


async def main():
    # ---------- 1. 连接 MCP 服务，加载工具 ----------
    # 0.3 版本不再支持 async with 上下文管理器，直接实例化
    client = MultiServerMCPClient(
        {
            # 方式 A：stdio——由适配器自动拉起本地 MCP 服务子进程
            "weather": {
                "command": "uv",
                "args": ["run", MCP_SERVER],
                "transport": "stdio",
            },
            # 方式 B：http——连接已经在运行的服务
            # "weather": {
            #     "url": "http://127.0.0.1:8000/mcp",
            #     "transport": "streamable_http",
            # },
        }
    )
    weather_tools = await client.get_tools()  # 把 MCP 工具转成 LangChain 工具

    # ---------- 2. 每个领域一个子 Agent ----------
    weather_agent = create_agent(
        model=llm,
        tools=weather_tools,
        system_prompt="你是天气专家，只负责回答天气相关的问题。",
    )

    # ---------- 3. 主 Agent：决定找哪个专家 ----------
    # 把子 Agent 包装成一个普通工具（接收问题字符串，返回专家的回答）
    from langchain_core.tools import tool  # noqa: E402

    @tool
    async def ask_weather_agent(question: str) -> str:
        """咨询天气专家。question：要问的问题"""
        # MCP 工具是异步的，内部子 Agent 必须用 ainvoke
        result = await weather_agent.ainvoke({"messages": [("user", question)]})
        return result["messages"][-1].content

    main_agent = create_agent(
        model=llm,
        tools=[ask_weather_agent],
        system_prompt="你是总管，天气问题交给 ask_weather_agent 工具处理。",
    )

    result = await main_agent.ainvoke(
        {"messages": [("user", "上海适合穿短袖吗？")]}
    )
    print("AI：", result["messages"][-1].content)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

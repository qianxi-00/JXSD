# -*- coding: utf-8 -*-
"""
MCP ⑥：LangChain / DeepAgents 调用 MCP 工具
================================================================
langchain-mcp-adapters 一行把 MCP 服务变成 LangChain 工具：

    client = MultiServerMCPClient({...})   # 多个 MCP 服务的配置
    tools = await client.get_tools()        # 自动转换格式

之后 create_agent(tools=tools) 或 create_deep_agent(tools=tools) 即用。
一个 Agent 可以同时接多个 MCP 服务（stdio / http 混用）。

运行方式：
    uv run 05_mcp/06_agent调用_langchain.py
"""

import asyncio
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding="utf-8")

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


async def main():
    # 0.3 版本不再支持 async with 上下文管理器，直接实例化
    client = MultiServerMCPClient(
        {
            # 本地 stdio 服务
            "life": {
                "command": "uv",
                "args": ["run", str(Path(__file__).resolve().parent / "01_服务端.py")],
                "transport": "stdio",
            },
            # 远程 HTTP 服务（先启动：uv run 05_mcp/01_服务端.py http）
            # "remote": {
            #     "url": "http://127.0.0.1:8000/mcp",
            #     "transport": "streamable_http",
            # },
        }
    )
    tools = await client.get_tools()
    print("MCP 工具列表：", [t.name for t in tools])

    # 方式 A：LangChain 智能体
    # from langchain.agents import create_agent
    # agent = create_agent(model=llm, tools=tools)

    # 方式 B：DeepAgents 智能体
    agent = create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt="你是生活助手，回答天气等问题时使用工具。",
    )

    # MCP 工具是异步的，必须用 ainvoke 调用智能体
    result = await agent.ainvoke(
        {"messages": [("user", "上海天气怎么样？顺便算一下 3+5")]},
        config={"recursion_limit": 50},
    )
    print("AI：", result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())

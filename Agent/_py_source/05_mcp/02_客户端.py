# -*- coding: utf-8 -*-
"""
MCP ②：客户端（fastmcp.Client）
================================================================
客户端连接 MCP 服务，列出 / 调用工具：
    - StdioTransport ：以子进程方式拉起服务脚本（本地）
    - StreamableHttpTransport：连接远程 HTTP 服务

用法：
    async with Client(transport) as client:
        await client.list_tools()
        await client.call_tool("get_weather", {"city": "上海"})

运行方式：
    uv run 05_mcp/02_客户端.py
"""

import asyncio
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding="utf-8")

from fastmcp import Client
from fastmcp.client.transports import StdioTransport, StreamableHttpTransport


async def demo_stdio():
    """本地 stdio：自动拉起 01_服务端.py 子进程"""
    transport = StdioTransport(
        command="uv", args=["run", str(Path(__file__).resolve().parent / "01_服务端.py")],
    )
    async with Client(transport) as client:
        tools = await client.list_tools()
        print("可用工具：", [t.name for t in tools])
        result = await client.call_tool("get_weather", {"city": "上海"})
        print("调用结果：", result)


async def demo_http():
    """远程 HTTP：先启动服务（uv run 05_mcp/01_服务端.py http）"""
    transport = StreamableHttpTransport(url="http://127.0.0.1:8000/mcp")
    async with Client(transport) as client:
        result = await client.call_tool("add", {"a": 1, "b": 2})
        print("HTTP 调用结果：", result)


if __name__ == "__main__":
    asyncio.run(demo_stdio())
    # 需要 HTTP 服务已启动，否则注释掉下一行
    # asyncio.run(demo_http())

# -*- coding: utf-8 -*-
"""
MCP ⑨：权限——带认证的客户端
================================================================
完整调用链：
    1. 找认证服务登录，拿 JWT（POST /login）
    2. MCP 客户端带上 Bearer Token 连接 MCP 服务
    3. 服务端校验 Token，合法才放行

运行顺序：
    1. uv run 05_mcp/08_权限_服务端.py http     （窗口 1）
    2. uv run 05_mcp/09_权限_客户端.py          （窗口 2）
"""

import asyncio
import sys
sys.stdout.reconfigure(encoding="utf-8")

import httpx
from fastmcp import Client
from fastmcp.client.auth import BearerAuth
from fastmcp.client.transports import StreamableHttpTransport

MCP_URL = "http://127.0.0.1:8000/mcp"


def login(username: str, password: str) -> str:
    """向认证服务换取 JWT"""
    resp = httpx.post(
        "http://127.0.0.1:9000/login",
        json={"username": username, "password": password},
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


async def main():
    # 1. 登录拿令牌
    token = login("admin", "123456")
    print("拿到 JWT：", token[:40], "...")

    # 2. 带 Bearer Token 连接 MCP 服务（fastmcp 3.x 客户端必须异步使用）
    transport = StreamableHttpTransport(url=MCP_URL, auth=BearerAuth(token))
    async with Client(transport) as client:
        tools = await client.list_tools()
        print("工具列表：", [t.name for t in tools])

        result = await client.call_tool("get_weather", {"city": "上海"})
        print("调用结果：", result)


if __name__ == "__main__":
    asyncio.run(main())

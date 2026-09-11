# -*- coding: utf-8 -*-
"""
MCP ⑤：OpenAI Agent 调用 MCP 工具（05_agent调用_openai.py）
================================================================
原生 OpenAI SDK 接 MCP：手动做「MCP 工具 → OpenAI Function Call 格式」的转换。

流程：
    1. MCP 客户端 list_tools 拿到工具定义
    2. 转成 OpenAI 的 tools 格式（type=function / name / description / parameters）
    3. 模型返回 tool_calls 后，用 MCP 客户端 call_tool 执行
    4. 结果回传模型

运行方式：
    uv run 05_mcp/05_agent调用_openai.py
"""

import asyncio
from pathlib import Path
import json
import sys
sys.stdout.reconfigure(encoding="utf-8")

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from openai import OpenAI
from config import settings

client_llm = OpenAI(api_key=settings.api_key, base_url=settings.base_url)


def mcp_tool_to_openai(tool) -> dict:
    """把 MCP 工具定义转换成 OpenAI Function Call 描述格式"""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


async def main():
    transport = StdioTransport(command="uv", args=["run", str(Path(__file__).resolve().parent / "01_服务端.py")])
    async with Client(transport) as mcp:
        # 1. 拉取 MCP 工具并转换格式
        mcp_tools = await mcp.list_tools()
        openai_tools = [mcp_tool_to_openai(t) for t in mcp_tools]
        print("接入的 MCP 工具：", [t.name for t in mcp_tools])

        messages = [
            {"role": "system", "content": "你是生活助手，优先使用工具回答。"},
            {"role": "user", "content": "上海天气怎么样？"},
        ]

        # 2. Function Call 循环（同 04_function_call/agent_openai.py）
        for _ in range(10):
            resp = client_llm.chat.completions.create(
                model=settings.model_name,
                messages=messages,
                tools=openai_tools,
            )
            msg = resp.choices[0].message
            messages.append(msg.model_dump())

            if not msg.tool_calls:
                print("AI：", msg.content)
                break

            for call in msg.tool_calls:
                # 3. 用 MCP 客户端执行工具（不再本地找函数）
                result = await mcp.call_tool(
                    call.function.name,
                    json.loads(call.function.arguments),
                )
                print(f"[MCP 工具 {call.function.name}] {result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result),
                })


if __name__ == "__main__":
    asyncio.run(main())

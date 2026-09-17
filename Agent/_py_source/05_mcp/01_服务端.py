# -*- coding: utf-8 -*-
"""
MCP ①：服务端（FastMCP）
================================================================
MCP（Model Context Protocol）：AI 的「USB 接口」标准。
把工具做成 MCP 服务，任何支持 MCP 的客户端（Claude / Cursor /
LangChain / DeepAgents…）都能即插即用，不用为每个框架重写工具。

FastMCP 三大装饰器：
    @mcp.tool     工具：让模型「做事」（函数）
    @mcp.resource 资源：让模型「读数据」（URI 定位，只读）
    @mcp.prompt   提示词：可复用的提示词模板

传输方式：
    stdio  ：标准输入输出（本地子进程，如 Claude Desktop）
    http   ：Streamable HTTP（网络服务，生产推荐）

运行方式：
    uv run 05_mcp/01_服务端.py            # 默认 stdio 模式（供子进程调用）
    uv run 05_mcp/01_服务端.py http       # HTTP 模式，监听 8000 端口
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

# 创建 MCP 服务实例。name 会显示在客户端的工具列表里
mcp = FastMCP(name="生活服务")


# ---------- 工具 ----------
@mcp.tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气。city：城市名称，如「上海」"""
    weather_map = {"上海": "晴 25 度", "北京": "多云 18 度", "广州": "阵雨 30 度"}
    return weather_map.get(city, f"{city} 天气未知")


@mcp.tool
def add(a: int, b: int) -> int:
    """计算两个整数的和"""
    return a + b


def run_stdio():
    """stdio 模式：被客户端以子进程方式拉起（如 02_langchain/12 的用法）"""
    mcp.run()  # 默认 stdio


def run_http():
    """HTTP 模式：作为独立网络服务运行，客户端通过 URL 连接"""
    mcp.run(transport="http", host="127.0.0.1", port=8000, path="/mcp")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "http":
        run_http()
    else:
        run_stdio()

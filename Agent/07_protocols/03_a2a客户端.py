# -*- coding: utf-8 -*-
"""
A2A 协议 ②：客户端（发现并调用远程 Agent）
================================================================
A2A 客户端流程：
    1. 拉取远程 Agent 的名片（agent.json）：知道它叫什么、能干什么
    2. 按 A2A 任务格式发送消息，接收流式/非流式响应
    3. 多个 A2A Agent 可以互相调用，形成 Agent 网络

官方 SDK：
    uv add a2a-sdk
    from a2a.client import A2AClient / A2ACardResolver

运行方式：
    先启动服务端（见 02_a2a服务端.py），再运行本文件：
    uv run 07_protocols/03_a2a客户端.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

AGENT_URL = "http://localhost:10000"

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("本示例依赖 a2a-sdk（uv add a2a-sdk），演示流程如下：")

    print(
        """
# ---------- A2A 客户端伪代码 ----------
from a2a.client import A2ACardResolver, A2AClient
import httpx

async def main():
    async with httpx.AsyncClient() as httpx_client:
        # 1. 拉取远程 Agent 的「名片」（能力发现）
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=AGENT_URL)
        card = await resolver.get_agent_card()
        print("发现智能体：", card.name, card.description)

        # 2. 创建客户端并发送任务
        client = A2AClient(httpx_client=httpx_client, agent_card=card)
        response = await client.send_message(
            {
                "message": {
                    "role": "user",
                    "parts": [{"type": "text", "text": "现在几点了？"}],
                }
            }
        )
        print("远程 Agent 回复：", response)

# A2A 的意义：不管对端是 LangGraph 还是 CrewAI 写的 Agent，
# 只要它说 A2A 协议，你都能用同样方式发现并调用它。
"""
    )

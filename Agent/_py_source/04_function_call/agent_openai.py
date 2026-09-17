# -*- coding: utf-8 -*-
"""
Function Call ③：原生 OpenAI SDK 实现智能体（agent_openai.py）
================================================================
不用任何框架，手写完整的 Function Call 循环，理解底层原理：

    1. 把「用户问题 + 工具描述」发给模型
    2. 模型若需要工具，返回 tool_calls（函数名 + 参数）
    3. 程序执行对应函数，把结果作为 ToolMessage 追加进消息列表
    4. 再发给模型，让它根据工具结果生成最终回答
    5. （模型可能连续调多个工具，循环直到不再返回 tool_calls）

对比：LangChain 的 create_agent 内部就是自动做了这个循环。

运行方式：
    uv run 04_function_call/agent_openai.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from openai import OpenAI
from config import settings

from tools import TOOL_REGISTRY
from tool_desc import TOOLS

# OpenAI 兼容客户端（DeepSeek / 通义 / vLLM 都能这样接）
client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

messages = [
    {"role": "system", "content": "你是一个生活助手，查询天气和时间请使用工具。"},
    {"role": "user", "content": "上海现在天气怎么样？现在几点了？"},
]

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    # ---------- Function Call 主循环 ----------
    for _ in range(10):  # 防御：限制最多 10 轮，避免死循环
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=messages,
            tools=TOOLS,             # 把工具描述发给模型
            tool_choice="auto",      # 模型自己决定是否调工具
        )
        msg = response.choices[0].message
        messages.append(msg.model_dump())

        # 模型没有要求调工具 → 说明已有最终答案，退出循环
        if not msg.tool_calls:
            print("AI 最终回答：", msg.content)
            break

        # 执行模型要求的每一个工具调用
        for call in msg.tool_calls:
            name = call.function.name
            import json
            args = json.loads(call.function.arguments)  # 参数是 JSON 字符串
            print(f"[模型请求调用] {name}({args})")

            # 从注册表找到真正的 Python 函数并执行
            result = TOOL_REGISTRY[name](**args)
            print(f"[工具执行结果] {result}")

            # 工具结果回传给模型（tool_call_id 必须对应）
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": result,
            })

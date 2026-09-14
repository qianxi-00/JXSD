# -*- coding: utf-8 -*-
"""
DeepAgents 流式输出
================================================================
和 LangGraph 一样，deep agent 也用 stream + stream_mode 流式：
    "messages"：token 级流式（打字机）
    "updates" ：每个节点/工具步骤的增量
    "values"  ：每步的完整状态（可以看到 todo / files 的变化）

运行方式：
    uv run 03_deepagents/02_流式输出.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
    streaming=True,
)

agent = create_deep_agent(
    model=llm,
    tools=[],
    system_prompt="你是写作助手，先在文件里列提纲，再输出正文。",
)

if __name__ == "__main__":
    print("===== token 流式 =====")
    input_data = {"messages": [{"role": "user", "content": "如何制作披萨"}]}

    # updates：每完成一个节点就输出增量
    # for chunk in agent.stream(input_data, stream_mode="messages"):
    #     print(chunk[0].content, end="")

    # 多模式流式：每个事件是 (mode, data) 元组；
    # messages 模式的 data 又是 (消息块, 元数据) 二元组
    for mode, data in agent.stream(
        {"messages": [("user", "写一篇 100 字的短文介绍秋天")]},
        stream_mode=["messages", "updates"],
        config={"recursion_limit": 50},
    ):
        if mode == "messages":
            chunk = data[0] if isinstance(data, tuple) else data
            content = getattr(chunk, "content", "")
            if isinstance(content, list):
                content = "".join(str(c) for c in content)
            if content:
                print(content, end="", flush=True)
        else:
            # 打印每个步骤的节点增量，观察 Agent 的行动轨迹
            print(f"\n[步骤] {data}")

# -*- coding: utf-8 -*-
"""
DeepAgents 后端（Backend）①：StateBackend —— 状态内虚拟文件系统
================================================================
DeepAgents 的文件工具（write_file / read_file / ls…）并不直接写磁盘，
而是写到「backend」里。默认 backend 就是 StateBackend：
文件内容存在图的 State 中（files 字段），只存在于本次会话。

适合：任务过程中的中间产物（草稿、笔记、计划），会话结束即丢弃。

运行方式：
    uv run 03_deepagents/03_后端_State.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from deepagents import create_deep_agent
from deepagents.backends import StateBackend  # 默认后端，显式写出便于理解
from langchain.chat_models import init_chat_model
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

agent = create_deep_agent(
    model=llm,
    backend=StateBackend(),  # 不传 backend 时默认就是它
    system_prompt="你是笔记助手，把要点写入 notes.md。",
)

if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [("user", "把 LangGraph 的三个核心概念写进 notes.md")]},
        config={"recursion_limit": 50},
    )

    # 文件就存在 State 里，可以自己读取
    files = result.get("files", {})
    print("会话内虚拟文件：", list(files.keys()))
    for name, content in files.items():
        print(f"----- {name} -----")
        print(content)

# -*- coding: utf-8 -*-
"""
DeepAgents 后端（Backend）④：LocalShellBackend —— 本机 Shell 执行
================================================================
让 execute 工具直接在本机执行 shell 命令（等同于给 Agent 开了终端）。

⚠️ 安全警告：本机执行 = 完全信任 Agent，只用于受控环境，
   生产环境请用 Docker / 云沙箱（见 08_后端_Sandbox.py）。

运行方式：
    uv run 03_deepagents/06_后端_LocalShell.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from deepagents import create_deep_agent
from deepagents.backends import DEFAULT_EXECUTE_TIMEOUT, LocalShellBackend
from langchain.chat_models import init_chat_model
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

if __name__ == "__main__":
    # root_dir：命令的工作目录；timeout：单条命令超时秒数
    backend = LocalShellBackend(
        root_dir=".",
        timeout=DEFAULT_EXECUTE_TIMEOUT,  # 单条命令默认超时时间
    )

    agent = create_deep_agent(
        model=llm,
        backend=backend,
        system_prompt="你是运维助手，不要删除任何东西。",
    )

    # result = agent.invoke(
    #     {"messages": [("user", "看看当前目录有哪些文件，统计一下 Python 文件数量")]},
    #     config={"recursion_limit": 50},
    # )
    # print("AI：", result["messages"][-1].content)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "创建一个py文件，py文件里面的内容是创建一个txt文件，并执行py文件"}]})
    print(result["messages"][-1].content)

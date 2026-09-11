# -*- coding: utf-8 -*-
"""
DeepAgents 后端（Backend）⑥：Sandbox —— 云沙箱执行
================================================================
LocalShell 在本机执行命令太危险，生产环境把 execute 放进沙箱：
    - 命令在隔离容器 / 云端虚拟机里运行
    - 即使 Agent 执行了危险命令，也伤不到宿主机

课案使用 Opensandbox 系列包：
    uv add deepagents-opensandbox opensandbox opensandbox-server
并需先运行沙箱服务端（课案建议在 WSL/Linux 中运行）：
    opensandbox-server

运行方式：
    1. （可选，Linux/WSL）启动沙箱服务：opensandbox-server
    2. uv run 03_deepagents/08_后端_Sandbox.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from deepagents import create_deep_agent
from deepagents.backends import LangSmithSandbox  # deepagents 内置沙箱适配
from langchain.chat_models import init_chat_model
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

if __name__ == "__main__":
    # LangSmithSandbox 需要一个已创建的 LangSmith 沙箱实例：
    #   from langsmith.sandbox import Sandbox
    #   sandbox = Sandbox.create(...)  # 需配置 LANGSMITH_API_KEY
    #   backend = LangSmithSandbox(sandbox)
    # 未配置 LangSmith 时会直接抛出缺少凭据的异常——这是预期的。

    # Opensandbox 版本（需安装 deepagents-opensandbox 并启动服务端）：
    #   uv add deepagents-opensandbox opensandbox opensandbox-server
    #   （在 WSL/Linux 中启动服务端）opensandbox-server
    #   from deepagents_opensandbox import OpensandboxBackend
    #   backend = OpensandboxBackend(server_url="http://localhost:8000")

    # 由于沙箱需要云端/本地服务端支撑，本文件仅做导入与说明，不实际执行。
    print("沙箱后端说明输出完毕（真实执行需要 LangSmith 或 Opensandbox 服务端）")

# -*- coding: utf-8 -*-
"""
DeepAgents 后端（Backend）③：FilesystemBackend —— 真实磁盘
================================================================
让智能体的文件工具直接读写真实磁盘目录（可以限制根目录做沙箱）。

用途：让 Agent 直接处理你电脑上的项目 / 文档。

安全提示：给 Agent 指定 root_dir 限定作用范围，避免它乱写系统目录。

运行方式：
    uv run 03_deepagents/05_后端_Filesystem.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

if __name__ == "__main__":
    # 智能体只能看到这个目录（相对路径的根）
    workdir = Path("tmp_deepagents_fs")
    workdir.mkdir(exist_ok=True)
    (workdir / "hello.txt").write_text("你好，DeepAgents！", encoding="utf-8")

    backend = FilesystemBackend(root_dir=str(workdir))

    agent = create_deep_agent(
        model=llm,
        backend=backend,
        system_prompt="你是文件助手，操作前先 ls 看看目录里有什么。",
    )

    result = agent.invoke(
        {"messages": [("user", "读一下 hello.txt，再创建 bye.txt 写上再见")]},
        config={"recursion_limit": 50},
    )
    print("AI：", result["messages"][-1].content)
    print("磁盘上现在有：", [p.name for p in workdir.iterdir()])

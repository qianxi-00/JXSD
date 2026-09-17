# -*- coding: utf-8 -*-
"""
ACP 协议：Agent Client Protocol（编辑器 ↔ 智能体）
================================================================
ACP 让「编辑器/IDE（如 Zed）」与「本地智能体进程」用统一协议通信：
编辑器出界面，Agent 进程干活，跨编辑器复用同一个智能体。

DeepAgents 提供 ACP 适配：
    uv add deepagents-acp
    deepagents-acp --deepagents ACP_SERVER_PATH

把本文件作为 ACP 服务运行后，编辑器里即可与智能体对话。

运行方式：
    uv add deepagents-acp   # 需先安装
    deepagents-acp --deepagents 07_protocols/acp智能体.py
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
)

# DeepAgents 的 ACP 适配器要求模块暴露一个 `agent` 对象（编译后的图）
agent = create_deep_agent(
    model=llm,
    tools=[],
    system_prompt=(
        "你是编码助手。可以读写文件、规划任务；"
        "回答尽量给出可直接执行的修改步骤。"
    ),
)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("本文件由 deepagents-acp 加载运行（不要直接执行）")
    print("启动命令：deepagents-acp --deepagents 07_protocols/acp智能体.py")

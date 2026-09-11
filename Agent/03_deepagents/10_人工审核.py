# -*- coding: utf-8 -*-
"""
DeepAgents 人工审核（interrupt_on）
================================================================
给工具配置 interrupt_on，调用前挂起等待人类批准：

    interrupt_on={
        "工具名": {"allowed_decisions": ["approve", "edit", "reject"]}
    }

    approve：批准        edit：改参数后执行
    reject：拒绝（模型收到反馈后自行调整）

必须配 checkpointer（中断状态要持久化）。

运行方式：
    uv run 03_deepagents/10_人工审核.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


@tool
def send_email(to: str, subject: str) -> str:
    """发送邮件（危险操作，需要人工批准）"""
    return f"已发送邮件给 {to}，主题：{subject}"


agent = create_deep_agent(
    model=llm,
    tools=[send_email],
    system_prompt="你是邮件助手，帮用户发送邮件。",
    checkpointer=MemorySaver(),
    interrupt_on={
        "send_email": {
            "allowed_decisions": ["approve", "reject"],
            "description": "发送邮件前必须经过用户确认",
        },
    },
)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "email-1"}}

    # 第一次执行：模型要发邮件 → 挂起
    result = agent.invoke(
        {"messages": [("user", "给 boss@company.com 发一封邮件，主题：请假")]},
        config,
    )
    print("挂起等待审核：", result.get("__interrupt__"))

    # 批准（演示）。拒绝的话：
    # Command(resume={"decisions": [{"type": "reject", "message": "拒绝理由"}]})
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config,
    )
    print("AI：", result["messages"][-1].content)

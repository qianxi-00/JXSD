# -*- coding: utf-8 -*-
"""
LangChain 智能体：人工审核（Human-in-the-loop）
================================================================
工具调用前先经人类批准。LangChain 1.4 用 HumanInTheLoopMiddleware 配置：

    HumanInTheLoopMiddleware(
        interrupt_on={
            工具名: {"allowed_decisions": ["approve", "edit", "reject", "respond"]}
        }
    )

    approve：批准执行
    reject ：拒绝，模型会收到拒绝反馈并自行调整
    edit   ：修改参数后执行
    respond：人类直接代替工具给出回复

恢复执行的格式：
    Command(resume={"decisions": [{"type": "approve"}]})
    Command(resume={"decisions": [{"type": "reject", "message": "拒绝理由"}]})

必须配 checkpointer（中断状态要靠检查点保存）。

运行方式：
    uv run Agent/02_langchain/09_人工审核.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
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
def delete_file(filename: str) -> str:
    """删除指定文件（危险操作，需要人工批准）"""
    return f"文件 {filename} 已删除"


@tool
def read_file(filename: str) -> str:
    """读取文件内容（安全操作，无需批准）"""
    return f"{filename} 的内容……"


agent = create_agent(
    model=llm,
    tools=[delete_file, read_file],
    system_prompt="你是文件管理助手。",
    checkpointer=MemorySaver(),  # 人工审核必须配 checkpointer
    middleware=[
        # 危险工具调用前挂起，等待人类决策
        HumanInTheLoopMiddleware(
            interrupt_on={
                "delete_file": {
                    "allowed_decisions": ["approve", "reject"],
                    "description": "删除文件属于危险操作，需要用户确认",
                },
            }
        ),
    ],
)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "hitl-1"}}

    # 第一次执行：模型要删文件 → 挂起等待审核
    result = agent.invoke(
        {"messages": [("user", "帮我把 tmp.txt 删掉")]}, config
    )
    print("挂起等待审核：", result.get("__interrupt__"))

    # ---------- 分支 1：拒绝 ----------
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "reject", "message": "不许删！文件还要用"}]}),
        config,
    )
    print("拒绝后 AI：", result["messages"][-1].content)

    # ---------- 分支 2：批准（换一个 thread_id 重新演示） ----------
    config = {"configurable": {"thread_id": "hitl-2"}}
    agent.invoke({"messages": [("user", "帮我把 tmp.txt 删掉")]}, config)
    result = agent.invoke(
        Command(resume={"decisions": [{"type": "approve"}]}),
        config,
    )
    print("批准后 AI：", result["messages"][-1].content)

# -*- coding: utf-8 -*-
"""
A2A 协议 ①：服务端（把 Agent 发布为 A2A 服务）
================================================================
A2A（Agent2Agent）：不同公司/框架做的 Agent 互相协作的开放协议。
一个 A2A 服务端 = 一张「智能体名片」（Agent Card）+ 标准任务接口：

    GET  /.well-known/agent.json   名片（能力、地址、认证方式）
    POST /tasks/send              提交任务（A2A 任务标准格式）

a2a_auto_wrapper 可以零改动把现有 Agent（LangGraph / CrewAI…）
包装成 A2A 服务：
    uv add a2a_auto_wrapper
    a2a-auto-wrap --module 07_protocols/a2a服务端 --url http://localhost:10000

运行方式：
    uv add a2a_auto_wrapper    # 需先安装
    uv run 07_protocols/02_a2a服务端.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.tools import tool
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


@tool
def get_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# a2a_auto_wrapper 会寻找 `agent` 对象并发布
agent = create_agent(
    model=llm,
    tools=[get_time],
    system_prompt="你是时间助手，回答时间问题请使用工具。",
)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print("本文件由 a2a_auto_wrapper 包装发布（不要直接执行）")
    print("发布命令：a2a-auto-wrap --module 07_protocols.02_a2a服务端 --url http://localhost:10000")
    print("发布后可访问 http://localhost:10000/.well-known/agent.json 查看名片")

# -*- coding: utf-8 -*-
"""
LangChain 基础：消息（Message）
================================================================
对话的底层单位是消息。LangChain 常用 4 种角色：

    system    系统设定：规定 AI 的人设 / 行为规则（通常只出现一次，放最前）
    user      用户输入
    assistant AI 的回复（有的供应商叫 assistant，LangChain 统一封装）
    tool      工具执行结果（回传给模型的 ToolMessage）

消息列表 = 对话的完整上下文，模型每次都是对着整个列表重新生成回复。
「记忆」的本质：不断把新消息 append 到列表里，再交给模型。

运行方式：
    uv run 02_langchain/02_消息.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.chat_models import init_chat_model
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

if __name__ == "__main__":
    messages = [
        # 系统消息：设定人设
        ("system", "你是一个简洁的编程助手，每次回答不超过 50 字。"),
        # 用户消息
        ("user", "什么是函数调用？"),
    ]

    response = llm.invoke(messages)
    print("AI：", response.content)

    # ---------- 多轮对话：把 AI 的回复也放进列表 ----------
    messages.append(response)          # 记住 AI 刚说了什么
    messages.append(("user", "展开讲讲"))  # 用户追问
    response = llm.invoke(messages)
    print("AI（追问）：", response.content)

    # ---------- 工具消息示例 ----------
    # 调用工具后，要把 ToolMessage（工具结果）放进列表回传给模型：
    # from langchain_core.messages import ToolMessage
    # messages.append(ToolMessage(content="查询结果：上海 25 度", tool_call_id="call_xxx"))

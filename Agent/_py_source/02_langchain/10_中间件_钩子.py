# -*- coding: utf-8 -*-
"""
LangChain 中间件：钩子（Hooks）
================================================================
中间件（Middleware）= 在智能体循环的各个环节插入自定义逻辑。
钩子模型：before_model / after_model / modify_model_request / wrap_model_call 等。

    before_model        每次调模型之前（可改状态）
    after_model         每次调模型之后（可改状态、可 interrupt 审核）
    modify_model_request 修改发给模型的请求（改消息、改参数）
    wrap_model_call     包裹模型调用（重试、日志、限流等）

课案示例：用钩子实现「调用日志 + 每轮注入动态时间」。

运行方式：
    uv run 02_langchain/10_中间件_钩子.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from datetime import datetime

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    before_model,
    after_model,
    dynamic_prompt,
)
from langchain.chat_models import init_chat_model
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


# ---------- 钩子 1：before_model（调模型前） ----------
def log_before(state: AgentState, runtime) -> dict | None:
    """记录调用次数；也可以在这里清洗/裁剪消息"""
    count = state.get("call_count", 0) + 1
    print(f"[before_model] 第 {count} 次调用模型")
    return {"call_count": count}


# ---------- 钩子 2：after_model（调模型后） ----------
def log_after(state: AgentState, runtime) -> dict | None:
    print(f"[after_model] 最新消息：{state['messages'][-1].content[:50]}")
    return None  # 不修改状态


# ---------- 钩子 3：dynamic_prompt（动态提示词） ----------
@dynamic_prompt
def inject_time(request):
    """每次调模型前动态生成系统提示词，注入当前时间"""
    return f"你是时间助手。当前时间：{datetime.now()}，回答要简短。"


# ---------- 组装中间件 ----------
agent = create_agent(
    model=llm,
    tools=[],
    middleware=[
        before_model(log_before),
        after_model(log_after),
        inject_time,
    ],
)

if __name__ == "__main__":
    result = agent.invoke({"messages": [("user", "现在是几点？")]})
    print("AI：", result["messages"][-1].content)

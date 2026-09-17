# -*- coding: utf-8 -*-
"""
LangChain 内置中间件全家桶
================================================================
LangChain 1.x 内置了 7 个常用中间件，覆盖上下文管理三大痛点：

    ① SummarizationMiddleware  消息太多 → 自动摘要压缩历史
    ② ModelCallLimitMiddleware 限制调用次数 → 防止死循环 / 控制成本
    ③ ModelRetryMiddleware     模型调用失败 → 自动重试
    ④ ToolRetryMiddleware      工具执行失败 → 自动重试
    ⑤ TodoListMiddleware       任务清单 → 让 Agent 规划多步任务
    ⑥ ContextEditingMiddleware 上下文清理 → 把老工具结果替换成占位符
    ⑦ HumanInTheLoopMiddleware 人工审核（见 09_人工审核.py）

运行方式：
    uv run 02_langchain/11_内置中间件.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.agents import create_agent
from langchain.agents.middleware import (
    ContextEditingMiddleware,
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    SummarizationMiddleware,
    TodoListMiddleware,
    ToolRetryMiddleware,
)
from langchain.chat_models import init_chat_model
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

agent = create_agent(
    model=llm,
    tools=[],
    system_prompt="你是简洁助手。",
    middleware=[
        # ① 历史消息超过一定 token 时，自动摘要成一条总结消息
        SummarizationMiddleware(
            model=llm,                # 用哪个模型来写摘要
            trigger=("tokens", 4000),  # 超过 4000 token 触发摘要
            keep=("messages", 4),     # 摘要时保留最近 4 条原文
        ),

        # ② 模型最多调 10 次，防止工具死循环烧钱（thread 级限流）
        ModelCallLimitMiddleware(thread_limit=10),

        # ③ 模型接口失败自动重试，最多 2 次，指数退避
        ModelRetryMiddleware(max_retries=2),

        # ④ 工具失败自动重试，最多 3 次
        ToolRetryMiddleware(max_retries=3),

        # ⑤ 任务清单：多步任务自动规划 TODO
        TodoListMiddleware(),

        # ⑥ 上下文清理：旧的超大工具结果替换为占位符，省 token
        ContextEditingMiddleware(),
    ],
)

if __name__ == "__main__":
    result = agent.invoke({"messages": [("user", "帮我总结一下 LangChain 的核心概念")]})
    print("AI：", result["messages"][-1].content)

# -*- coding: utf-8 -*-
"""
DeepAgents 后端（Backend）⑤：ContextHubBackend —— 上下文枢纽
================================================================
新版 ContextHubBackend 是基于 LangSmith Hub 的共享文件后端：
    ContextHubBackend(identifier="owner/repo-name")
不同会话挂到同一个 Hub 仓库上，即可互相读写文件。

⚠️ 它依赖 LangSmith 账号（需配置 LANGSMITH_API_KEY 等环境变量），
   本文件给出一个不依赖 LangSmith 的等效演示：
   用 StoreBackend + 固定命名空间充当「共享枢纽」——
   相同命名空间的多个会话共享同一份文件，语义与 ContextHub 一致。

场景：主 Agent 与多个子 Agent 之间传递中间产物、
     多个独立会话共享同一份数据。

运行方式：
    uv run Agent/03_deepagents/07_后端_ContextHub.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from deepagents import create_deep_agent
from deepagents.backends import StoreBackend
from langchain.chat_models import init_chat_model
from langgraph.store.postgres import PostgresStore
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

if __name__ == "__main__":
    # 如需使用真正的 ContextHubBackend（LangSmith Hub 版）：
    #   from deepagents.backends import ContextHubBackend
    #   hub = ContextHubBackend(identifier="-/team-alpha")
    # 并配置 LANGSMITH_API_KEY 环境变量。

    with PostgresStore.from_conn_string(settings.pg_uri) as store:
        store.setup()

        # 用固定命名空间模拟「共享枢纽」：所有挂到该命名空间的会话共享文件
        hub = StoreBackend(store=store, namespace=lambda _rt: ("hub", "team-alpha"))

        agent = create_deep_agent(
            model=llm,
            backend=hub,
            system_prompt="你是团队协作助手，把结论写入共享文件。",
        )

        # 会话 A 写入文件
        r1 = agent.invoke(
            {"messages": [("user", "把「需求已确认」写入 shared/requirements.md")]},
            config={"configurable": {"thread_id": "session-A"}, "recursion_limit": 50},
        )
        print("会话 A：", r1["messages"][-1].content)

        # 会话 B（同一个枢纽）读取文件
        r2 = agent.invoke(
            {"messages": [("user", "读取 shared/requirements.md 告诉我写了什么")]},
            config={"configurable": {"thread_id": "session-B"}, "recursion_limit": 50},
        )
        print("会话 B：", r2["messages"][-1].content)

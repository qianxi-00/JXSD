# -*- coding: utf-8 -*-
"""
DeepAgents 后端（Backend）②：StoreBackend —— 跨会话的文件系统
================================================================
StateBackend 的文件随会话消失。StoreBackend 把文件写入
LangGraph Store（PostgresStore），按命名空间隔离，跨会话可用。

典型用途：用户的项目文件、长期沉淀的笔记。

新版 API：
    StoreBackend(
        store=my_store,
        namespace=callable,  # 接收 Runtime，返回命名空间元组
                             # 例：lambda rt: (rt.server_info.user.identity, "filesystem")
    )

依赖：PostgreSQL（见 config.py 的 pg_uri；需要 langgraph 库）

运行方式：
    uv run Agent/03_deepagents/04_后端_Store.py
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
    with PostgresStore.from_conn_string(settings.pg_uri) as store:
        store.setup()

        # namespace 是可调用对象：根据运行时上下文动态生成命名空间。
        # 这里演示用固定命名空间，实际可按 user_id / thread_id 划分。
        backend = StoreBackend(store=store, namespace=lambda _rt: ("user-1001", "filesystem"))

        agent = create_deep_agent(
            model=llm,
            backend=backend,
            system_prompt="你是文档助手，可以把重要文档写入文件系统。",
        )

        result = agent.invoke(
            {"messages": [("user", "创建一份 readme.md，写上：这是用户的项目说明")]},
            config={"configurable": {"thread_id": "store-demo"},
                    "recursion_limit": 50},
        )
        print("AI：", result["messages"][-1].content)

        # 换一个新会话（新 thread_id），文件依然在——这就是 StoreBackend 的意义
        result2 = agent.invoke(
            {"messages": [("user", "读取 readme.md 并告诉我内容")]},
            config={"configurable": {"thread_id": "store-demo-2"},
                    "recursion_limit": 50},
        )
        print("新会话 AI：", result2["messages"][-1].content)

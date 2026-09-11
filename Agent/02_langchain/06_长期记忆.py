# -*- coding: utf-8 -*-
"""
LangChain 智能体：长期记忆（InMemoryStore / PostgresStore）
================================================================
长期记忆 = 跨会话共享的记忆（不依赖 thread_id）。
在 create_agent 里通过 store 参数注入，配合中间件或工具读写。

课案示例场景：智能体记住用户的姓名偏好，新会话里直接使用。

存储结构（三元组）：
    (命名空间, key, value)
    如 ("memories", "1001") → {"name": "小红", "style": "简洁"}

运行方式：
    uv run 02_langchain/06_长期记忆.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.store.memory import InMemoryStore
from langchain_core.tools import tool
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

store = InMemoryStore()  # 生产环境换成 langgraph.store.postgres.PostgresStore


# ---------- 用工具的方式读写长期记忆 ----------
@tool
def save_user_memory(key: str, value: str) -> str:
    """保存用户的长期记忆。key：记忆条目名（如 name）；value：内容"""
    # 智能体调用工具时通过 store 上下文拿到注入的 store
    store.put(("memories", "1001"), key, {"value": value})
    return f"已保存记忆 {key}={value}"


@tool
def read_user_memory(key: str) -> str:
    """读取用户的长期记忆。key：记忆条目名"""
    item = store.get(("memories", "1001"), key)
    return item.value["value"] if item else "没有这条记忆"


# ---------- 预先写一条记忆，模拟历史会话存下的偏好 ----------
store.put(("memories", "1001"), "style", {"value": "回复风格：简洁，不要废话"})

agent = create_agent(
    model=llm,
    tools=[save_user_memory, read_user_memory],
    system_prompt="你是私人助手。回答前先用 read_user_memory 查用户偏好并遵守。",
    store=store,
)

if __name__ == "__main__":
    # 新会话（新 thread_id），但偏好依然生效——这就是「长期」的含义
    result = agent.invoke(
        {"messages": [("user", "介绍一下 LangChain")]},
        config={"configurable": {"thread_id": "session-A", "user_id": "1001"}},
    )
    print("AI：", result["messages"][-1].content)

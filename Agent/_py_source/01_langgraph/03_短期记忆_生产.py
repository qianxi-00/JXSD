# -*- coding: utf-8 -*-
"""
LangGraph 短期记忆：生产环境使用 PostgreSQL 持久化
================================================================
MemorySaver 数据存内存，重启就丢。生产环境用 PostgresSaver：
检查点写入 PostgreSQL，服务重启后记忆依然存在。

前置准备：
    1. 启动 PostgreSQL（Docker 一行命令）：
       docker run -e POSTGRES_PASSWORD=postgres -d --name postgres -p 5432:5432 postgres:18
    2. 进入容器建库：
       docker exec -it postgres psql -U postgres -c "CREATE DATABASE langgraph;"
    3. .env 中配置 PG_URI=postgresql://<用户名>:<口令>@127.0.0.1:5432/<库名>
       （用户名 / 口令 / 库名属于敏感信息，只写在本地 .env 里，不要提交进仓库）

依赖：uv add langgraph-checkpoint-postgres psycopg

运行方式：
    uv run 01_langgraph/03_短期记忆_生产.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


def chat(state: MessagesState) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("chat", chat)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

if __name__ == "__main__":
    # PostgresSaver 建议用连接池（PostgresSaver.from_conn_string 内部创建）
    with PostgresSaver.from_conn_string(settings.pg_uri) as checkpointer:
        # 首次使用前必须执行：自动在数据库里建好检查点相关的表
        checkpointer.setup()

        graph = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": "user-1001"}}

        # 由于数据已经落库，即使进程重启，
        # 只要 thread_id 相同，历史对话依然可以被读出来
        result = graph.invoke({"messages": [("user", "你好，我叫小红，记住我")]}, config)
        print("AI：", result["messages"][-1].content)

        result = graph.invoke({"messages": [("user", "我叫什么？")]}, config)
        print("AI：", result["messages"][-1].content)

        # 查看某个会话的完整历史快照
        snapshot = graph.get_state(config)
        print("当前状态包含", len(snapshot.values["messages"]), "条消息")

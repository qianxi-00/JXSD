# -*- coding: utf-8 -*-
"""
LangGraph 长期记忆：跨会话的用户信息（Store）
================================================================
checkpointer 的记忆按 thread_id 隔离，只对单个会话有效；
Store 的记忆按「命名空间」全局共享，可以跨会话——这就是「长期记忆」。

典型场景：记住用户偏好。用户在会话 A 里说过喜欢简洁回复，
换到会话 B 里，智能体依然记得。

- PostgresStore：基于 PostgreSQL 的生产级 Store
- put / get / search：写入、读取、按前缀搜索记忆条目

前置准备同 03_短期记忆_生产.py（需要 PostgreSQL）。

运行方式：
    uv run 01_langgraph/04_长期记忆.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langgraph.store.postgres import PostgresStore
from config import settings

if __name__ == "__main__":
    with PostgresStore.from_conn_string(settings.pg_uri) as store:
        # 首次使用前先建表
        store.setup()

        # ---------- 写入长期记忆 ----------
        # put(命名空间, 键, 值)
        # 命名空间用元组表示层级，例如 ("users", "1001") = 用户 1001 的记忆空间
        store.put(("users", "1001"), "preference", {"reply_style": "简洁，少说废话"})
        store.put(("users", "1001"), "profile", {"name": "小红", "city": "上海"})

        # ---------- 读取 ----------
        item = store.get(("users", "1001"), "preference")
        print("读取记忆：", item.value)

        # ---------- 搜索：按命名空间前缀列出所有条目 ----------
        items = list(store.search(("users", "1001")))
        for it in items:
            print(f"记忆[{it.key}] = {it.value}")

        # ---------- 删除 ----------
        # store.delete(("users", "1001"), "preference")
        print("长期记忆演示完成，数据已持久化到 PostgreSQL")

# -*- coding: utf-8 -*-
"""
DeepAgents 记忆（MemoryMiddleware）
================================================================
DeepAgents 的记忆 = 一个特殊的「记忆文件夹」：
Agent 自己决定什么值得记，通过 write_file 写进 /memories/ 目录，
每次新会话开始时自动把记忆注入系统提示词。

记忆跨会话生效的前提：后端要持久化（StoreBackend / FilesystemBackend）。

memory=["xxx.md"]：指定记忆文件的说明，引导 Agent 记什么。

运行方式：
    uv run 03_deepagents/11_记忆.py
"""

import sys

from deepagents.backends.utils import create_file_data
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

sys.stdout.reconfigure(encoding="utf-8")

from deepagents import create_deep_agent
from deepagents.backends import StateBackend, StoreBackend, CompositeBackend
from langchain.chat_models import init_chat_model
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

# agent = create_deep_agent(
#     model=llm,
#     backend=StateBackend(),  # 生产环境换 StoreBackend 让记忆真正持久化
#     system_prompt="你是贴心的私人助手。",
#     # 记忆说明：告诉 Agent 应该记住哪些方面的信息
#     memory=["用户偏好.md：记录用户的称呼、喜好、忌讳，每次会话结束前更新"],
# )

if __name__ == "__main__":
#     # 第一轮：告知偏好 → Agent 写入 /memories/
#     r1 = agent.invoke(
#         {"messages": [("user", "我叫小明，喜欢简洁回答，讨厌表情包")]},
#         config={"configurable": {"thread_id": "mem-1"}, "recursion_limit": 50},
#     )
#     print("会话1 AI：", r1["messages"][-1].content)
#     print("记忆文件：", list(r1.get("files", {}).keys()))
#
#     # 第二轮（新 thread_id）：记忆被注入 → Agent 直接按偏好行事
#     r2 = agent.invoke(
#         {"messages": [("user", "介绍一下你自己")]},
#         config={"configurable": {"thread_id": "mem-2"}, "recursion_limit": 50},
#     )
#     print("会话2 AI：", r2["messages"][-1].content)
    with (
        PostgresStore.from_conn_string(settings.pg_uri) as store,
        PostgresSaver.from_conn_string(settings.pg_uri) as checkpointer,
    ):
        store.setup()
        checkpointer.setup()

        # namespace 是可调用对象：根据运行时上下文动态生成命名空间。
        # 这里演示用固定命名空间，实际可按 user_id / thread_id 划分。
        backend = StoreBackend(store=store, namespace=lambda _rt: ("user-1001", "filesystem"))

        agent = create_deep_agent(
            model=llm,
            backend=backend,
            system_prompt="你是文档助手，可以把重要文档写入文件系统。",
        )

        existing = store.get(("my-agent",), "/memories/AGENTS.md")
        if existing is None:
            store.put(
                ("my-agent",),
                "/memories/AGENTS.md",
                create_file_data("""回复风格 回复简洁，不超过三句话"""),
            )
            print("初始化默认记忆文件")
        else:
            print("记忆文件已存在，保留现有内容")

        agent = create_deep_agent(
            model=llm,
            memory=["/memories/AGENTS.md"],
            system_prompt="你的记忆文件在 /memories/AGENTS.md。每次回复前先 read_file 读取记忆，学到新信息后用 edit_file 更新该文件。",
            backend=CompositeBackend(
                default=StateBackend(),
                routes={
                    "/memories/": StoreBackend(
                        namespace=lambda rt: ("my-agent",),
                    ),

                },
            ),
            store=store,  # 长期记忆
            checkpointer=checkpointer,  # 短期记忆（对话历史）
        )

        # Thread 1：Agent 学到新偏好，自动写入记忆
        config1 = {"configurable": {"thread_id": "1"}}
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "我叫小明，我喜欢长篇大论"}]},
            config=config1,
        )
        for m in result["messages"]:
            print(f"m.content:{m.content}")
            print(f"m.name:{m.name}")
            print("=" * 50)

        # 验证是否修改了
        mem = store.get(("my-agent",), "/memories/AGENTS.md")
        print(mem.value['content'])
        print("----" * 50)

        result = agent.invoke(
            {"messages": [{"role": "user", "content": "先把我刚刚说的话重复一边，写一篇咖啡店小红书帖子"}]},
            config=config1,
        )

        print(result["messages"][-1].content)

        print("----" * 50)

        # Thread 2：Agent 读取记忆，应用之前的偏好
        config2 = {"configurable": {"thread_id": "2"}}
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "先把我刚刚说的话重复一边，写一篇咖啡店小红书帖子"}]},
            config=config2,
        )
        print(result["messages"][-1].content)
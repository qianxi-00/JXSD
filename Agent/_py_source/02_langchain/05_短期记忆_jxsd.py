# -*- coding: utf-8 -*-
r"""
LangChain 智能体：短期记忆（checkpointer + thread_id）
================================================================
课案原文（用 PostgreSQL 落库的 checkpointer 保存会话状态）：

    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.postgres import PostgresSaver
    from conf import settings

    DB = "postgresql://<用户名>:<口令>@127.0.0.1:5432/langgraph"
    model = ChatOpenAI(model=settings.model_name, api_key=settings.api_key, base_url=settings.base_url)
    with PostgresSaver.from_conn_string(DB) as checkpointer:
        checkpointer.setup()                       # 首次运行建表
        agent = create_agent(model=model, tools=[], checkpointer=checkpointer)
        agent.invoke({"messages": [{"role": "user", "content": "我叫张三"}]},
                     {"configurable": {"thread_id": "1"}})
        result = agent.invoke({"messages": [{"role": "user", "content": "我叫什么？"}]},
                              {"configurable": {"thread_id": "1"}})
        print(result["messages"][-1].content)      # 你叫张三

本项目按铁律 3（不硬编码连接串）把 `DB` 字面量换成 `settings.pg_uri`。

短期记忆的三个关键点：

    1. 「记忆」存在 checkpointer 里，不占我们的代码。每次 invoke 只传**本轮新消息**，
       历史消息由框架按 thread_id 自动取回来拼在开头 —— 这就是它和 02_消息 里
       「自己 append 到列表」的本质区别。
    2. config 里的 `thread_id` 是会话号：同一个 id 才共享历史，换一个 id 就是全新会话。
       所以「多用户隔离」只需要给每个用户一个 thread_id。
    3. checkpointer 决定「记在哪、活多久」：

    # | checkpointer        | 存储位置     | 进程重启后         | 适用场景                 |
    # |---------------------|-------------|-------------------|--------------------------|
    # | InMemorySaver       | 内存字典     | 丢                | 本地调试、单元测试        |
    # | PostgresSaver       | PostgreSQL  | 还在（可续聊）     | 生产环境（本节演示）      |
    # | SqliteSaver         | 本地文件     | 还在（单机）       | 单机小工具               |

前置准备（本机已就绪，无需重复操作）：
    - PostgreSQL 已运行，且库名与 `settings.pg_uri` 一致（本项目为 langgraph）；
    - `checkpointer.setup()` 会自动建表，重复执行是幂等的。

课案出处：Agent 课案 → langChain → 核心组件 → 短期记忆

前置条件与运行方式：
    - 根目录 `.env` 的 `API_KEY` / `BASE_URL` / `MODEL_NAME` 已填好，且 `PG_URI` 已配；
    - `PG_URI` 为空或连不上时，`main()` 会打印中文提示并直接返回，**不会抛 traceback**；
    - 在项目根目录 `F:\ProGram\Python_Base` 下执行（否则 import 不到根目录的 `config`）：
    uv run Agent/02_langchain/05_短期记忆_jxsd.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.postgres import PostgresSaver
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

# 课案在这里写死了 DB 连接串；本项目从 .env 经 config.py 读取（连接串内含口令，绝不能进代码）
DB_URI = settings.pg_uri


def main() -> None:
    # ---------- 0. 前置检查：连接串没配就直接给中文提示，不抛异常 ----------
    if not DB_URI:
        print("[跳过] settings.pg_uri 为空：请在项目根目录 .env 里配置 PG_URI=")
        print("       postgresql://<用户>:<口令>@127.0.0.1:5432/langgraph")
        return

    try:
        # from_conn_string 是上下文管理器：退出时自动归还连接
        with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
            # ---------- 1. 课案原样：首次运行建表 ----------
            # 幂等操作，表已存在时不会报错，也不会清空已有数据。
            checkpointer.setup()

            # ---------- 2. 把 checkpointer 交给 create_agent ----------
            # 注意 checkpointer 是挂在 agent（底层图）上的，不是挂在某次 invoke 上。
            agent = create_agent(model=llm, tools=[], checkpointer=checkpointer)

            # thread_id="1" 是课案原文用的会话号；
            # config 的位置参数是 invoke 的第二个参数，也可以写成 config=... 关键字。
            config = {"configurable": {"thread_id": "1"}}

            # ---------- 3. 第一轮：只发「我叫张三」 ----------
            # 我们并没有把历史消息带上，但框架会把该 thread 的历史自动拼进去。
            agent.invoke(
                {"messages": [{"role": "user", "content": "我叫张三"}]},
                config,
            )

            # ---------- 4. 第二轮：只发「我叫什么？」 ----------
            # 这一轮之所以能答出来，全靠 checkpointer 按 thread_id 取回了第一轮的消息。
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "我叫什么？"}]},
                config,
            )
            print("===== 4. 课案原文：同一 thread 的跨轮记忆 =====")
            print("AI：", result["messages"][-1].content)   # 预期：你叫张三

            # ---------- 5. 证据：状态里到底存了什么 ----------
            # get_state 读的就是 checkpointer 里该 thread 的最新快照。
            snapshot = agent.get_state(config)
            print("\n===== 5. checkpointer 里存的状态 =====")
            print(f"  该 thread 共 {len(snapshot.values['messages'])} 条消息（不是 1 条！）：")
            for index, msg in enumerate(snapshot.values["messages"], start=1):
                print(f"    [{index}] {msg.type:<7} {str(msg.content)[:40]}")

            # ---------- 6. 换一个 thread_id = 换一个会话（记忆互不干扰） ----------
            # 这是「短期记忆」的边界：只认 thread_id，不认人。
            other_config = {"configurable": {"thread_id": "1-demo-isolated"}}
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "我叫什么？"}]},
                other_config,
            )
            print("\n===== 6. 换 thread_id 后（应当答不出名字） =====")
            print("AI：", result["messages"][-1].content)

            # ---------- 7. 落库证明 ----------
            # 上面所有状态都写在 PostgreSQL 里，进程重启后只要 thread_id 相同就能续聊，
            # 这正是 PostgresSaver 比 InMemorySaver 值钱的地方（见 01_langgraph/03_短期记忆_生产.py）。
            # 想亲眼验证：把本文件再跑一次，第 4 步仍能答出「你叫张三」——
            # 因为 thread_id 没变，历史是从数据库里捞回来的，不是内存里的。
            print("\n===== 7. 存储位置 =====")
            print("  本轮会话已写入 PostgreSQL，连接串来自 settings.pg_uri（不回显口令）")

    except Exception as exc:  # 连接不上数据库时给出可操作的中文提示，而不是一堆 traceback
        print("[跳过] 连接 PostgreSQL 失败：", type(exc).__name__)
        print("       请确认本机 PostgreSQL 已启动、库 langgraph 已创建、.env 的 PG_URI 正确。")
        print("       详情：", str(exc)[:200])


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
云技能：把 SKILL.md 存进 PostgreSQL，用 CompositeBackend 做多用户隔离
================================================================
课案出处：Agent 课案 → skills → 云技能

本节要讲什么：

    1. 【为什么上云】本地技能放在磁盘上「一台机器一份、改一次全体生效」，
       但多用户场景下每个人要有自己的技能库 —— 靠 StoreBackend 的 namespace 分区实现；
    2. 【后端路由】CompositeBackend 的三段式写法：default 走内存、
       routes 把 "/skills/" 前缀交给 StoreBackend（落到 PostgreSQL）；
    3. 【两层存储的分工】PostgresStore（长期数据，按 namespace 分区）
       vs PostgresSaver（会话检查点，按 thread_id 分区），各管什么；
    4. 【与课案的两处必要差异】连接串改用 settings.pg_uri（不硬编码口令）、
       store.put 的 key 要去掉路由前缀（否则技能发现数为 0）——
       这两条都有实测依据，见文件头下方「★ 与课案原文的两处【必要改动】」；
    5. 【隔离验证】同一个 agent 代码，只换 namespace，就能看到两个用户
       彼此发现不到对方的技能。

课案原文开头这段话就是本节的全部动机：

    DeepAgents 的 skills 机制原生支持两步加载：启动时只加载 SKILL.md 的
    `description`（省 Token），Agent 匹配到技能后才 `read_file` 拉取完整提示词。
    配合 `StoreBackend` 即可实现云端多用户隔离。

也就是说：
    - 本地技能（03 小节）放在磁盘上，一台机器一份，改一次全体生效；
    - 云技能把 SKILL.md 存到 LangGraph 的 Store（长期记忆）里，
      按 namespace 分区 —— user-001 和 user-002 各有各的技能库，互相看不见。

三个后端各自的位置（本节的骨架）：

    CompositeBackend(
        default = StateBackend()      # 其它所有路径：走内存（会话内有效）
        routes  = {"/skills/": StoreBackend(namespace=...)}   # /skills/ 前缀 → 走 Store
    )

    于是模型看到的 "/skills/sql-gen/SKILL.md"，实际存在 PostgreSQL 的
    store 表里，namespace = ("user-001",)。

★ 与课案原文的两处【必要改动】（不改就跑不通 / 违反规范）★

    1) 连接串：课案硬编码
           DB = "postgresql://<用户>:<口令>@<主机>:<端口>/langgraph"
       （课案原文里这一行写的是 postgres/postgres/localhost:5432 的字面量，
         本文件按规范脱敏成占位符，代码里一律不出现连接串字面量）
       规范第 1 节铁律 3 明令禁止硬编码口令与连接串，本文件改用
           settings.pg_uri
       （.env 里的 PG_URI，本机已实测可用，库名 langgraph）。

    2) store.put 的 key：课案写的是 "/skills/code-review/SKILL.md"。
       在本机 deepagents 0.7.13 里，CompositeBackend 命中路由 "/skills/" 后
       会把前缀【剥掉】再交给 StoreBackend（见 backends/composite.py 的
       _route_for_path），所以 StoreBackend 里的 key 必须是【相对于路由前缀】
       的 "/code-review/SKILL.md"。
       写全路径 "/skills/code-review/SKILL.md" 的后果是：模型虚拟路径变成
       "/skills/skills/code-review/SKILL.md"，而 skills=["/skills/"] 一个技能都发现不了
       （实测：发现技能数 0）。这一点课案的版本和本机版本的语义不同，
       本文件按本机实测的正确写法来。

运行前置条件：
    - .env 里的 PG_URI 可用（本机 PostgreSQL 17 + langgraph 库，已实测）；
    - settings.api_key 可用（模型 grok-4.6，已实测）。

运行方式：
    uv run Agent/08_skills/04_云技能_jxsd.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，防中文/emoji 报错

import uuid

from langchain.chat_models import init_chat_model

from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


# ================================================================
# 课案原文（DB 那行做了脱敏，实际代码里必须用 settings.pg_uri）
# ================================================================
COURSE_CODE = '''from langchain.agents import create_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.backends.utils import create_file_data
from langgraph.store.postgres import PostgresStore
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_openai import ChatOpenAI
from config import settings

DB = "postgresql://<用户>:<口令>@<主机>:<端口>/langgraph"   # ← 课案是硬编码，已脱敏
model = ChatOpenAI(model=settings.model_name, api_key=settings.api_key, base_url=settings.base_url)

with (
    PostgresStore.from_conn_string(DB) as store,
    PostgresSaver.from_conn_string(DB) as checkpointer,
):
    store.setup()
    checkpointer.setup()

    # 为不同用户预存不同的 skill
    store.put(("user-001",), "/skills/code-review/SKILL.md", create_file_data("""---
name: code-review
description: Python 代码审查，检查规范性和性能问题
---


你是 Python 专家，审查以下代码的规范性、性能和安全性。
"""))

    store.put(("user-001",), "/skills/sql-gen/SKILL.md", create_file_data("""---
name: sql-gen
description: 根据自然语言生成 SQL 语句
---


根据用户的自然语言描述生成对应的 SQL 语句。
"""))

    agent = create_deep_agent(
        model=model,
        skills=["/skills/"],
        backend=CompositeBackend(
            default=StateBackend(),
            routes={
                "/skills/": StoreBackend(
                    namespace=lambda rt: ("user-001",),  # 本地用固定值
                ),
            },
        ),
        store=store,
        checkpointer=checkpointer,
    )
    config = {"configurable": {"thread_id": "1"}}
    result = agent.invoke({"messages": [{"role": "user", "content": "用技能sql-gen查询用户总数"}]}, config)
    for m in result["messages"]:
        print(m)
'''


# ================================================================
# 两个技能的内容：与课案一致，保留 create_file_data("""...""") 的写法
# ================================================================
# 注意 name 必须等于技能目录名（code-review / sql-gen），否则中间件会告警
CODE_REVIEW_SKILL = """---
name: code-review
description: Python 代码审查，检查规范性和性能问题
---


你是 Python 专家，审查以下代码的规范性、性能和安全性。
"""

SQL_GEN_SKILL = """---
name: sql-gen
description: 根据自然语言生成 SQL 语句
---


根据用户的自然语言描述生成对应的 SQL 语句。
"""

# 额外造一个「只有 user-002 有」的技能，用来演示多用户隔离
REPORT_SKILL = """---
name: weekly-report
description: 撰写正式工作周报时使用。当用户要求写周报、总结本周工作时使用。
---


按「本周完成 / 下周计划 / 风险与求助」三段写，总字数 200 字以内。
"""

USER_001 = "user-001"
USER_002 = "user-002"


# ================================================================
# 1. 课案原文与两处必要改动
# ================================================================
def section_course() -> None:
    print("=" * 78)
    print("1. 课案原文（连接串已脱敏；store key 见下方说明）")
    print("=" * 78)
    print(COURSE_CODE)
    print("-" * 78)
    # 用 ''' 做定界符：正文里要出现 create_file_data("""...""")，用 """ 会被提前截断
    print(
        f'''
本文件相对课案原文的改动：

    [改 1] 连接串
        课案：DB = "postgresql://<用户>:<口令>@<主机>:<端口>/langgraph"（硬编码，本文件已脱敏）
        本文件：settings.pg_uri
        原因：规范铁律 —— 口令/连接串不写进代码。当前 pg_uri 指向
              {settings.pg_uri.split('@')[-1] if '@' in settings.pg_uri else settings.pg_uri}
              （用户名口令部分不打印）

    [改 2] store.put 的 key 要去掉路由前缀
        课案：store.put(ns, "/skills/code-review/SKILL.md", ...)
        本文件：store.put(ns, "/code-review/SKILL.md", ...)
        原因：CompositeBackend 命中 "/skills/" 路由后会把前缀剥掉再交给
              StoreBackend（本机 deepagents 0.7.13 实测）。
              用课案的写法，模型虚拟路径会变成 /skills/skills/code-review/SKILL.md，
              而 skills=["/skills/"] 会发现 0 个技能 —— 本文件在注释里保留了
              这个坑的说明，代码走正确写法。

    [保持] create_file_data("""...""") 的写法原样保留 ——
           它把字符串包成 LangGraph Store 认识的文件结构
           （content / encoding / created_at / modified_at）。
'''
    )


# ================================================================
# 2. 写入云端技能
# ================================================================
def section_seed_store(store) -> None:
    print("=" * 78)
    print("2. 把技能写进 PostgreSQL 的 Store（按 namespace 隔离）")
    print("=" * 78)

    from deepagents.backends.utils import create_file_data

    # store.put(namespace, key, value)
    #   namespace：分区键，相当于「谁的技能库」；这里用 ("user-001",) 固定值，
    #              真实项目里可以从鉴权信息里取，实现一用户一技能库。
    #   key      ：文件在 StoreBackend 里的路径 —— 相对于 CompositeBackend 的路由前缀。
    store.put((USER_001,), "/code-review/SKILL.md", create_file_data(CODE_REVIEW_SKILL))
    print(f"  [{USER_001}] + /code-review/SKILL.md")
    store.put((USER_001,), "/sql-gen/SKILL.md", create_file_data(SQL_GEN_SKILL))
    print(f"  [{USER_001}] + /sql-gen/SKILL.md")

    # user-002 只有一个技能 —— 用来证明两个用户的技能库互相看不见
    store.put((USER_002,), "/weekly-report/SKILL.md", create_file_data(REPORT_SKILL))
    print(f"  [{USER_002}] + /weekly-report/SKILL.md")

    print()
    print("  写入结果（直接读 Store，验证真的落库了）：")
    for user in (USER_001, USER_002):
        items = store.search((user,))
        print(f"    namespace=({user!r},) → {[item.key for item in items]}")


# ================================================================
# 3. 用 CompositeBackend + StoreBackend 跑 agent
# ================================================================
def build_agent(store, checkpointer, user_id: str):
    """按用户组装 agent：同一个 model / 同一套代码，只有 namespace 不同。"""
    from deepagents import create_deep_agent
    from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

    return create_deep_agent(
        model=llm,
        skills=["/skills/"],  # 技能的【父目录】——云技能同样遵守这条规则
        backend=CompositeBackend(
            default=StateBackend(),  # 非 /skills/ 的路径留在内存里（会话结束即失效）
            routes={
                # 命中 "/skills/" 的路径全部交给 StoreBackend，落到 PostgreSQL
                "/skills/": StoreBackend(
                    # namespace 是「运行时 → 命名空间」的函数。
                    # 课案注释写的是「本地用固定值」；真实项目里改成
                    # lambda rt: (rt.context.user_id,) 之类，就能按登录用户隔离。
                    namespace=lambda rt: (user_id,),
                    store=store,
                ),
            },
        ),
        store=store,                 # 让 agent 的长期记忆工具也用这个 store
        checkpointer=checkpointer,   # 会话检查点：多轮对话能续上
    )


def section_run(store, checkpointer) -> None:
    print()
    print("=" * 78)
    print("3. 真跑一次：让 agent 用云端的 sql-gen 技能")
    print("=" * 78)

    agent = build_agent(store, checkpointer, USER_001)

    # thread_id 用 uuid：每次运行都是全新会话。
    # 课案写的是固定 "1"，但那样第二次运行会命中 PostgreSQL 里上一轮的检查点，
    # 而 SkillsMiddleware 的逻辑是「state 里已有 skills_metadata 就不再加载」，
    # 结果就是重跑时技能加载被跳过、行为不稳定。教学脚本要可重复运行，故用 uuid。
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print(f"  thread_id = {config['configurable']['thread_id']}")
    print("  用户输入：用技能 sql-gen 查询用户总数")
    print()

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "用技能sql-gen查询用户总数"}]},
        config,
    )

    # 只打印最后一条 AI 回答 + 工具调用轨迹（课案是 for m in result["messages"]: print(m)，
    # 那样会把每条消息对象全量打出来，几十行噪音；这里保留等价信息但可读）
    print("  ── 工具调用轨迹 ──")
    for message in result["messages"]:
        if type(message).__name__ == "AIMessage":
            for call in getattr(message, "tool_calls", None) or []:
                print(f"    模型 → {call.get('name')}  参数 {call.get('args')}")
    print()
    print("  ── 最终回答 ──")
    print("  " + "-" * 74)
    for line in str(result["messages"][-1].content).splitlines():
        print("  | " + line)
    print("  " + "-" * 74)


# ================================================================
# 4. 多用户隔离验证：同一个 agent 代码，换 namespace 就换技能库
# ================================================================
def section_isolation(store, checkpointer) -> None:
    print()
    print("=" * 78)
    print("4. 多用户隔离验证：同一份代码，只换 namespace")
    print("=" * 78)

    from deepagents.middleware.skills import SkillsMiddleware

    from deepagents.backends import CompositeBackend, StateBackend, StoreBackend

    # 两个用户各建一套 backend + middleware，唯一的差别就是 namespace。
    # 这里每个 user_id 都在循环内【立刻】用掉（before_agent 马上执行），
    # 所以下面的 lambda 直接闭包捕获 user_id 是安全的；如果把这些对象先存起来、
    # 出了循环再统一调用，闭包就会全部读到最后一个 user_id —— 「循环里建 lambda」
    # 的经典坑，写成 lambda rt, u=user_id: (u,) 才保险。
    for user_id in (USER_001, USER_002):
        backend = CompositeBackend(
            default=StateBackend(),
            routes={"/skills/": StoreBackend(namespace=lambda rt, u=user_id: (u,), store=store)},
        )
        middleware = SkillsMiddleware(backend=backend, sources=["/skills/"])
        update = middleware.before_agent({}, None, None)  # type: ignore[arg-type]
        names = [s["name"] for s in (update["skills_metadata"] if update else [])]
        print(f"  namespace=({user_id!r},) 发现的技能 → {names}")

    print(
        """
  结论：两份数据存在同一张 PostgreSQL 表里，靠 namespace 分区；
        user-002 无论怎么问，都不可能看到 user-001 的 code-review / sql-gen。
        这就是课案说的「配合 StoreBackend 即可实现云端多用户隔离」——
        隔离的粒度不是机器，也不是目录，而是 store 的 namespace。

  顺带说明另外两层的分工：
        PostgresSaver（checkpointer）存的是【会话消息】，
        按 thread_id 分区，负责「多轮对话记得住」；
        PostgresStore（store）存的是【跨会话的长期数据】，
        按 namespace 分区，本节的技能文件就放在这一层。
        两者可以共用一个库（本机都在 langgraph 库里，表名不同）。
"""
    )


# ================================================================
# 5. 前置检查 + 主流程
# ================================================================
def preflight() -> bool:
    """检查 PostgreSQL 是否可用；不可用就打印中文提示并返回 False。"""
    if not settings.pg_uri:
        print("！.env 里没有配置 PG_URI，请在 F:\\ProGram\\Python_Base\\.env 中补上，")
        print("  形如：PG_URI=postgresql://<用户>:<口令>@127.0.0.1:5432/langgraph")
        return False
    return True


if __name__ == "__main__":
    section_course()

    if not preflight():
        sys.exit(0)

    # 延迟 import：把「缺包」也纳入可读提示的范围
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
        from langgraph.store.postgres import PostgresStore
    except ImportError as exc:
        print()
        print(f"！缺少依赖：{exc}")
        print("  请在项目根目录执行：uv add langgraph-checkpoint-postgres langgraph-checkpoint")
        sys.exit(0)

    # 一个 with 同时开两个连接上下文（课案的写法）：
    #   store        —— 长期记忆 / 技能文件
    #   checkpointer —— 会话检查点
    try:
        with (
            PostgresStore.from_conn_string(settings.pg_uri) as store,
            PostgresSaver.from_conn_string(settings.pg_uri) as checkpointer,
        ):
            # setup() 必须各调一次：自动建表（幂等，重复调用没问题）
            store.setup()
            checkpointer.setup()

            section_seed_store(store)
            section_run(store, checkpointer)
            section_isolation(store, checkpointer)
    except Exception as exc:  # noqa: BLE001 —— 教学脚本不允许抛 traceback
        print()
        print("！连接 PostgreSQL 失败：", f"{type(exc).__name__}: {str(exc)[:200]}")
        print("  排查顺序：")
        print("    1. PostgreSQL 是否已启动（默认 127.0.0.1:5432）")
        print("    2. langgraph 库是否已建：CREATE DATABASE langgraph;")
        print("    3. .env 里的 PG_URI 用户名/口令/库名是否正确")
        sys.exit(0)

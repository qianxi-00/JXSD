# -*- coding: utf-8 -*-
"""
DeepAgents 记忆（memory + CompositeBackend + PostgresStore）
================================================================
DeepAgents 的记忆不走「向量库检索」那套，而是**通过文件系统实现** ——
Agent 把记忆当作文件来读写，backend 决定这些文件存在哪、谁能访问。

本节要讲什么
    1. DeepAgents 记忆的机制：**记忆就是文件**（不是向量库），
       `memory=[...]` 声明哪些路径算记忆，MemoryMiddleware 把内容拼进系统提示词；
    2. 短期记忆 vs 长期记忆分别由哪个参数负责
       （`checkpointer=` 管对话历史，`store=` + `memory=` 管记忆文件）；
    3. 为什么要 CompositeBackend 而不是裸 StoreBackend
       —— 只有 `/memories/` 需要持久化，框架内部文件留在内存更干净；
    4. **本文件的核心坑**：课案预存记忆用的 Store key 多了一层 `/memories/`，
       导致 Agent 在 `/memories/memories/` 里找文件（第六节完整复盘）。

一、两种记忆的区别（课案总表）
    | 记忆类型 | 机制 | 生命周期 |
    |---|---|---|
    | 短期记忆 | checkpoint 自动保存 messages，同一 thread_id 内多轮对话共享 | 同 thread |
    | 长期记忆 | memory=["/memories/..."] + StoreBackend，文件持久化到 PostgreSQL | 跨 thread、跨会话 |

    对应到代码就是两个参数各管一摊：
        checkpointer=checkpointer   → 短期记忆（对话历史）
        store=store + memory=[...]  → 长期记忆（记忆文件）

二、核心流程（课案原文三步）
    1. 用 `memory=["/memories/AGENTS.md"]` 告诉 Agent 哪些文件是记忆
    2. 用 `CompositeBackend` 把 `/memories/` 路由到 `StoreBackend`（跨线程持久化）
    3. Agent 通过 `read_file` / `edit_file` 工具读写记忆文件

    第 1 步的 memory 参数由 `MemoryMiddleware` 消费：它会把记忆文件的内容
    读出来、拼进**系统提示词**（所以 Agent 一开场就"记得"），
    同时把文件路径写进提示词，告诉模型「学到新东西要 update 这个文件」。

三、为什么要 CompositeBackend 而不是直接用 StoreBackend
    因为只有 `/memories/` 需要持久化，其余路径（框架内部的
    `/large_tool_results/`、`/conversation_history/`）留在内存里更干净 ——
    这一点和 09_后端_Composite_jxsd.py 的「建议」段是同一个道理。

四、记忆作用域怎么选（课案表）
    | 作用域 | namespace | 谁共享 | 典型用途 |
    |---|---|---|---|
    | Agent 级 | (assistant_id,) | 所有用户共享同一个 Agent 的记忆 | Agent 自学优化、累积知识 |
    | 用户级 | (user.identity,) | 每个用户独立 | 用户偏好、个人历史 |
    | 组织级 | (org_id,) | 整个组织共享 | 合规策略、公共知识库 |

    课案给的本地版用固定 namespace `("my-agent",)` 替代 `rt.server_info`，
    原因见 04_后端_Store_jxsd.py 文件头第三节：`server_info` 只有部署到
    LangGraph Server / LangSmith 时才由平台注入，本地跑拿不到。

五、课案代码里那个数据库字面量已被替换（规范铁律第 3 条）
    课案原文：  DB = "postgresql://<用户名>:<口令>@127.0.0.1:5432/langgraph"
    本项目：    DB = settings.pg_uri
    连接串里内嵌账号口令，写进代码等于把凭据提交进仓库，一律只从 .env 读。

六、课案代码里的一个真实 bug：预存记忆的 Store key 多了一层 `/memories/`
    课案原文是这么预存记忆的：

        existing = store.get(("my-agent",), "/memories/AGENTS.md")
        if existing is None:
            store.put(("my-agent",), "/memories/AGENTS.md", create_file_data("回复风格 回复简洁，不超过三句话"))

    但 CompositeBackend 的路由语义是**把前缀剥掉再转给目标后端**：
        调用方路径 /memories/AGENTS.md  →  StoreBackend 收到 /AGENTS.md
    而 StoreBackend 是**拿路径直接当 Store key** 的（04 文件里实测：
    写 /bubble_sort.py，Store 里的 key 就是 /bubble_sort.py）。

    两边一凑，结果就是：
        - 课案预存的 key `/memories/AGENTS.md`，在 Agent 眼里出现在 `/memories/memories/AGENTS.md`
        - Agent 按提示词去 read_file("/memories/AGENTS.md") → File not found
        - 它在 `/memories/memories/` 下翻到文件，再用 edit_file 改，
          又因为 read_file 的返回带着行号而改不动，最后白白烧掉几十步

    本文件跑第一版时实测就是这个现象（输出里能看到 File '/AGENTS.md' not found
    和 ls 返回 ['/memories/memories/']）。所以下面把预存 key 改成 `/AGENTS.md`
    —— **Store key 是「路由前缀剥掉之后」的路径**，这样才对得上。
    课案原样保留在注释里，方便对照。

    顺带一提：记忆命名空间也换成了 ("jxsd-my-agent",)，
    避免和同目录的 11_记忆.py（它用的是 ("my-agent",)）写同一个 PostgreSQL Store 互相覆盖。

课案出处：Agent 课案 → deepAgents → 记忆

运行方式：
    uv run Agent/03_deepagents/11_记忆_jxsd.py

前置条件：
    - .env 里的 api_key / base_url / model_name 可调通（实测可用）
    - PostgreSQL 已启动，settings.pg_uri 指向的库里已建好 langgraph 库（实测可用）
    - 依赖：langgraph-checkpoint-postgres / psycopg（项目 venv 已装）
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.backends.utils import create_file_data
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from config import settings

# 课案这里是写死的连接串；本项目一律走配置（口令只存在于 .env 里）
DB = settings.pg_uri

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

# 记忆文件的路径（Agent 视角的路径），也是交给 MemoryMiddleware 的 sources
MEMORY_PATH = "/memories/AGENTS.md"

# Agent 访问 /memories/AGENTS.md 时，CompositeBackend 会剥掉 /memories/ 前缀，
# StoreBackend 实际拿到的路径（= Store 里的 key）就是 /AGENTS.md。
# 预存记忆必须用这个 key，否则 Agent 读不到（详见文件头第六节）。
MEMORY_KEY = "/AGENTS.md"

# 记忆的命名空间。课案本地版写死 ("my-agent",)；生产环境改成
# lambda rt: (rt.server_info.user.identity,) 就能做到「每个用户一份记忆」。
# 这里换个名字，避免和同目录 11_记忆.py 共用同一个 Store 命名空间互相覆盖。
MEMORY_NAMESPACE = ("jxsd-my-agent",)

# 演示开关：为 True 时每次运行先清空本文件自己的记忆，让
# 「初始化默认记忆 → Agent 学到新偏好 → 改写记忆 → 新 thread 读记忆」这条主线
# 每次都完整走一遍（不清空的话，第二次跑只会打印「记忆文件已存在，保留现有内容」）。
# 生产环境当然不能清 —— 不清空才是记忆的意义所在。
# 想观察课案的 else 分支，把这个常量改成 False 再跑一次即可。
RESET_MEMORY_ON_START = True

if __name__ == "__main__":
    if not DB:
        print("[跳过] 未配置 settings.pg_uri（.env 里的 PG_URI），无法演示长期记忆。")
        sys.exit(0)

    # 两个上下文管理器一起开：一个 Store（长期记忆），一个 Checkpointer（短期记忆）。
    # PostgresStore / PostgresSaver 都是连接池封装，用 with 保证连接被释放。
    with (
        PostgresStore.from_conn_string(DB) as store,
        PostgresSaver.from_conn_string(DB) as checkpointer,
    ):
        # 首次使用前必须建表：store 建 store 相关表，checkpointer 建 checkpoint 相关表。
        # 重复调用是幂等的，所以每次跑都调一遍没问题。
        store.setup()
        checkpointer.setup()

        # ---------- 1. 预存记忆（长期记忆的初始内容） ----------
        # 演示用：先清空本文件命名空间下的旧记忆，保证每次跑都是同一个起点。
        # （只清 MEMORY_NAMESPACE 这一个命名空间，不碰别的数据）
        if RESET_MEMORY_ON_START:
            for item in store.search(MEMORY_NAMESPACE):
                store.delete(MEMORY_NAMESPACE, item.key)
            print(f"已清空 namespace={MEMORY_NAMESPACE} 下的历史记忆（演示用）")

        # 先查一次，避免每次都把用户已经改过的记忆覆盖回默认值。
        # 注意用的是 MEMORY_KEY（/AGENTS.md）而不是 MEMORY_PATH（/memories/AGENTS.md）——
        # 上面文件头第六节解释了这个坑。
        existing = store.get(MEMORY_NAMESPACE, MEMORY_KEY)
        if existing is None:
            store.put(
                MEMORY_NAMESPACE,
                MEMORY_KEY,
                # create_file_data 生成的是 backend 认识的 FileData 结构
                # （content / encoding / created_at / modified_at），
                # 直接 put 一个裸字符串 backend 读不出来。
                create_file_data("""回复风格 回复简洁，不超过三句话"""),
            )
            print("初始化默认记忆文件")
        else:
            print("记忆文件已存在，保留现有内容")

        # 打印一下 Store 里真实的 key，方便把「路径」和「Store key」对上号
        print(f"Store namespace={MEMORY_NAMESPACE} 里的 key：{[i.key for i in store.search(MEMORY_NAMESPACE)]}")

        # 记下「更新前」的记忆内容，跑完 Thread 1 后好做对比
        _before = store.get(MEMORY_NAMESPACE, MEMORY_KEY)
        before_text = _before.value["content"] if _before else ""

        agent = create_deep_agent(
            model=llm,
            # 声明哪些文件算「记忆」：MemoryMiddleware 会把它们注入系统提示词
            memory=[MEMORY_PATH],
            # 课案原文的 system_prompt 是：
            #   "你的记忆文件在 /memories/AGENTS.md。每次回复前先 read_file 读取记忆，
            #    学到新信息后用 edit_file 更新该文件。"
            # 这里在原文后面补了两句「工具用法」提示，原因是实测踩到的坑：
            #   read_file 的返回**每行前面带行号**（形如 `1  正文`），
            #   模型会把带行号的整行当成 old_string 丢给 edit_file，
            #   而磁盘/Store 里的真实内容是没有行号的 →
            #   于是反复报 "String not found in file"，记忆一直改不动。
            # 把格式说清楚、并给一条 write_file 兜底路径，更新就能成功。
            system_prompt=(
                "你的记忆文件在 /memories/AGENTS.md。每次回复前先 read_file 读取记忆，"
                "学到新信息后用 edit_file 更新该文件。\n"
                "工具用法提醒：read_file 返回的每一行前面有形如 `1  ` 的行号，"
                "edit_file 的 old_string 必须填**去掉行号后的原文**；"
                "edit_file 一旦报 String not found，就别再重试，"
                "直接用 write_file 把整理好的**新记忆全文（不带行号）**写进去 —— 这样最稳。"
            ),
            backend=CompositeBackend(
                # 兜底留在内存，只有 /memories/ 走持久化
                default=StateBackend(),
                routes={
                    "/memories/": StoreBackend(
                        # 生产改成 lambda rt: (rt.server_info.user.identity,)
                        namespace=lambda rt: MEMORY_NAMESPACE,
                    ),
                },
            ),
            store=store,                # 长期记忆（记忆文件的物理存储）
            checkpointer=checkpointer,  # 短期记忆（对话历史）
        )

        # ---------- 2. Thread 1：Agent 学到新偏好，自动写入记忆 ----------
        # 注意 thread_id 特意用了 "jxsd-mem-1" 这种带前缀的名字：
        # 话题历史存在同一张 PostgreSQL checkpoint 表里，是按 thread_id 索引的，
        # 如果这里也用课案的 "1"，就会和同目录 11_记忆.py 的历史串在一起
        # （实测第一版就串了：thread 1 一开场就冒出上一份代码留下的「我叫张三」）。
        config1 = {"configurable": {"thread_id": "jxsd-mem-1"}}
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "我叫xx，我喜欢长篇大论"}]},
            config={**config1, "recursion_limit": 60},
        )
        # 把**完整消息轨迹**打出来（课案也是这么做的）：
        # 这里能直接看到 Agent 到底有没有 read_file / edit_file / write_file 记忆文件，
        # 比只看最后一条回答信息量大得多。
        for m in result["messages"]:
            print(f"m.content:{m.content}")
            print(f"m.name:{m.name}")
            print("=" * 50)

        # 验证是否修改了
        # 注意这里绕开 Agent 直接读 Store —— 这是「记忆真的落库了」的硬证据
        mem = store.get(MEMORY_NAMESPACE, MEMORY_KEY)
        after_text = mem.value["content"] if mem else ""
        print(after_text)
        print(f"记忆是否被更新：{'是' if after_text != before_text else '否（本轮模型没有改写记忆文件）'}")
        print("----" * 50)

        # 同一 thread 的第二问：这次要的是「应用刚学到的偏好」。
        # 课案原文的任务，用来观察它有没有按记忆里的「喜欢长篇大论」来写。
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "先把我刚刚说的话重复一边，写一篇咖啡店小红书帖子"}]},
            config={**config1, "recursion_limit": 60},
        )

        print(result["messages"][-1].content)

        print("----" * 50)

        # ---------- 3. Thread 2：**新会话**读取记忆，应用之前的偏好 ----------
        # 这里是最关键的一步：thread_id 换了，短期记忆（对话历史）是空的，
        # 但 Agent 一上来仍然知道「用户喜欢长篇大论」—— 因为记忆走的是 Store，
        # 由 MemoryMiddleware 注入系统提示词，跟 thread 无关。
        config2 = {"configurable": {"thread_id": "jxsd-mem-2"}}
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "先把我刚刚说的话重复一边，写一篇咖啡店小红书帖子"}]},
            config={**config2, "recursion_limit": 60},
        )
        print(result["messages"][-1].content)

        print("\n" + "=" * 60)
        # 下面这一段是「看得见的部分」——只陈述**机制**上必然成立的事，
        # 不保证模型一定照做（模型服从度有波动，见下面的实测观察）
        print("结论（看得见的部分）：")
        print(f"  1) 记忆文件的物理落点：PostgresStore 的 namespace={MEMORY_NAMESPACE}，key='{MEMORY_KEY}'")
        print(f"  2) 记忆文件当前内容：{after_text!r}")
        print("  3) thread 2 是**全新会话**（thread_id 不同，对话历史为空），")
        print("     但 MemoryMiddleware 会把上面这份记忆文件注入它的系统提示词 ——")
        print("     所以「跨会话记住用户偏好」这件事在机制上是成立的，跟 thread_id 无关。")
        print()
        # 「实测观察」段：把不确定性明说，避免学生把模型的行为波动当成代码 bug
        print("实测观察（务必自己跑一遍看）：")
        print("  - 模型是否**真的按记忆行事**、是否**真的改写记忆文件**，取决于模型服从度，会有波动；")
        print("    本文件用 grok-4.6 实测：有时能成功改写记忆（输出里出现")
        print("    Successfully replaced 1 instance(s)），有时它会连着几次 edit_file 失败就放弃。")
        print("  - 最常见的失败原因是 read_file 返回的内容**带行号**，模型把带行号的整行"
              "当成 old_string 传给 edit_file；")
        print("    上面的 system_prompt 已经专门提醒过这一点，但提示词不是强约束。")
        print("  - 想干净复现，把 RESET_MEMORY_ON_START 置 True 再跑（默认就是 True）。")

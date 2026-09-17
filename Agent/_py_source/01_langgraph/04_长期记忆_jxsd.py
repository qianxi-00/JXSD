# -*- coding: utf-8 -*-
"""
LangGraph 长期记忆：跨会话的用户信息（Store）
================================================================
**短期记忆 vs 长期记忆**（课案概念，一张表说清）：

  # | 维度     | 短期记忆（checkpointer）        | 长期记忆（Store）                  |
  # |----------|---------------------------------|------------------------------------|
  # | 存什么   | 一个会话的**完整状态**          | 一条条**事实/偏好**（key-value）   |
  # | 隔离键   | thread_id（按会话隔离）         | namespace 元组（按业务维度隔离）   |
  # | 编译参数 | compile(checkpointer=...)       | compile(store=...)                 |
  # | 访问方式 | 框架自动读写，节点无感知        | 节点里手动 `runtime.store.xxx`     |
  # | 生效范围 | 同一个 thread_id 内             | **跨 thread_id、跨会话、跨用户**   |
  # | 典型场景 | 多轮对话上下文                  | 「用户偏好」「用户档案」这类要记住的事实 |

一句话：**checkpointer 管「这次聊了什么」，Store 管「这个用户是谁」。**
两者可以同时挂在一张图上（compile(checkpointer=..., store=...)），互不干扰。

**namespace 的隔离作用（本节重点）**：
`ns = ("memories", "u1")` 这种**元组**是 Store 里的层级路径，等价于文件系统的目录。
同一个 key 放在不同 namespace 下就是两条互不可见的记忆：
    ("memories", "u1") → 用户 u1 的记忆空间
    ("memories", "u2") → 用户 u2 的记忆空间（搜 u1 的内容永远搜不到）
这行代码就是多租户隔离的全部实现——比「在 value 里塞一个 user_id 再过滤」可靠得多。

**课案原文的一处笔误**：课案写的是 `from config import setting`（少了 s），
本项目统一改成 `from config import settings`；同时把课案硬编码的
`postgresql://<用户名>:<口令>@127.0.0.1:5432/langgraph` 换成 `settings.pg_uri`。

课案出处：Agent 课案 → langgraph → 核心组件 → 长期记忆
运行方式：
    uv run Agent/01_langgraph/04_长期记忆_jxsd.py
前置条件：
    - 依赖：`langgraph` + `langgraph-checkpoint-postgres`（PostgresStore 在这个包里）
      + `psycopg` 驱动，本项目已 uv sync 装好。
    - 数据库：需要一个可连的 PostgreSQL，连接串写在根目录 .env 的 PG_URI；
      库和 03_短期记忆_生产_jxsd.py 共用同一个库即可 ——
      LangGraph 会自己建独立的 store 表，与 checkpoints 表互不影响。
      首次运行由 `store.setup()` 自动建表（幂等）。
    - 配置：.env 里同时要有 MODEL_NAME / API_KEY / BASE_URL（本文件要真实调用大模型）。
    - 前置服务没起来时本文件**不抛 traceback**：连接串为空打印中文提示后 sys.exit(0)，
      连不上则捕获 psycopg.OperationalError 打印排查步骤。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，防中文/emoji 报错

from typing import TypedDict

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime  # 节点中通过 runtime.store 访问长期记忆
from langgraph.store.postgres import PostgresStore  # 长期记忆存储

from config import settings  # 课案原文是 `from config import setting`，此处已修正


class State(TypedDict):
    """课案原文的状态：只有一个 messages 字段。

    ⚠️ 注意这里用的是**普通 list 字段（覆盖式）**，而不是 MessagesState：
    节点返回 `{"messages": [...]}` 会把整个列表**覆盖**掉，历史不累积。
    这在本节恰好是想要的效果——长期记忆不靠会话上下文，靠 Store；
    所以这张图**不需要 checkpointer**，也就不需要 thread_id。
    （对比 02/03：那边用 MessagesState + checkpointer，历史由框架累积。）
    """

    messages: list  # 图节点之间流转的消息列表


# 初始化 LLM 模型：课案本节用 ChatOpenAI 写法，参数全部来自 settings
llm = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


# ============================================================
# 0. 前置检查：连接串必须来自 .env
# ============================================================
def require_pg_uri() -> str:
    """检查 settings.pg_uri，为空时打印中文提示并优雅退出（不抛异常）。"""
    uri = (settings.pg_uri or "").strip()
    if not uri:
        print("=" * 74)
        print("未配置 PostgreSQL 连接串，无法演示 PostgresStore 长期记忆。")
        print("=" * 74)
        print("请在项目根目录 .env 中配置（连接串含账号口令，不要写进代码）：")
        print("    PG_URI=postgresql://<用户名>:<口令>@<主机>:5432/<库名>")
        print("共享同一个库即可——LangGraph 会自己建独立的 store 表，")
        print("和短期记忆的 checkpoints 表互不影响。")
        sys.exit(0)
    return uri


# ============================================================
# 1. 节点函数：通过 runtime.store 读写长期记忆
# ============================================================
def make_chat_node(store_ns: tuple[str, str]):
    """生成对话节点。

    节点签名 `(state, runtime: Runtime)` 是 LangGraph 的**依赖注入**写法：
    只要第二个参数标注为 Runtime，LangGraph 就会在调用时自动注入运行时对象，
    我们从 `runtime.store` 拿到编译图时传入的那个 Store。
    （没有标注 Runtime 的第二个参数不会被注入——类型标注在这里是「开关」。）

    参数 store_ns 是本节点使用的 namespace，普通函数里可以直接闭包捕获；
    真实项目里一般写成 `("memories", user_id)`，user_id 从请求上下文取。
    """

    def chat(state: State, runtime: Runtime) -> dict:
        text = state["messages"][-1]["content"]  # 当前用户输入
        ns = store_ns  # namespace：按 user_id 隔离数据

        # ---- 1.1 检索：从长期记忆中取相关历史 ----
        # search(命名空间前缀, query=检索词, limit=N)
        # ⚠️ 实测要点（langgraph-checkpoint-postgres 3.1.2）：
        #    没给 PostgresStore 配向量索引时，query 参数会被**静默忽略**，
        #    SQL 退化成 `... WHERE prefix = ns ORDER BY updated_at DESC LIMIT N`，
        #    也就是「取这个 namespace 下最近更新的 N 条」，不是语义检索。
        #    要做真正的语义检索，创建 Store 时要带 index 配置：
        #        PostgresStore.from_conn_string(uri, index={
        #            "dims": 1024,                       # 向量维度
        #            "embed": some_embeddings_object,    # 嵌入模型
        #            "fields": ["data"],                 # 对 value 里哪个字段做向量化
        #        })
        #    数据量小时「最近 N 条」也够用，所以我们这里保持课案写法。
        mems = runtime.store.search(ns, query=text, limit=3)
        info = "；".join([m.value["data"] for m in mems]) or "暂无"

        # ---- 1.2 写入：把当前输入存进长期记忆 ----
        # put(命名空间, key, value)
        # key 用原文，意味着「同一句话」会覆盖同一条记忆；
        # 生产里建议用 uuid4 / 时间戳做 key，把内容放进 value。
        runtime.store.put(ns, text, {"data": text})

        # ---- 1.3 把检索到的记忆塞进系统提示词 ----
        response = llm.invoke(
            [
                {
                    "role": "system",
                    "content": f"你是一个拥有长期记忆的助手，根据以上记忆，回答客户的问题，记忆为{info}",
                },
                {"role": "user", "content": text},
            ]
        )
        return {"messages": [{"role": "ai", "content": response.content}]}

    return chat


# ============================================================
# 2. compile(store=store) 与 compile(checkpointer=...) 的区别
# ============================================================
# 课案原文：`graph = builder.compile(store=store)   # 只传 store，不传 checkpointer`
#
# 两个参数作用完全不同，别混：
#   compile(checkpointer=...)  → 开启**短期记忆**：框架自动把每一步状态存/取，
#                                节点代码里看不见它；必须传 thread_id 才能定位会话。
#   compile(store=...)         → 注入**长期记忆**：框架只把它挂到 runtime 上，
#                                存什么、取什么、什么时候存，全部由节点自己决定。
# 所以：只传 store 的图 invoke 时**不需要 config**（没有会话概念）；
#       只传 checkpointer 的图 invoke 时**必须传 config**（否则报错）。
#
# 两者也可以一起传：compile(checkpointer=cp, store=store) —— 既记会话，又记事实。


# ============================================================
# 3. 主流程
# ============================================================
def main() -> None:
    db_uri = require_pg_uri()

    # from_conn_string 返回上下文管理器，用 with 自动管理连接
    with PostgresStore.from_conn_string(db_uri) as store:
        store.setup()  # 首次运行创建数据库表（幂等）

        ns_u1 = ("memories", "u1")
        ns_u2 = ("memories", "u2")

        # ---------- 3.1 先手工演示 namespace 的隔离作用 ----------
        print("=" * 74)
        print("① namespace 隔离：不同元组 = 互不可见的独立记忆空间")
        print("=" * 74)
        # 清掉上一次运行的残留，保证每次跑出来的结果一致（只动本节自己的 namespace）
        for key in [it.key for it in store.search(ns_u1, limit=100)]:
            store.delete(ns_u1, key)
        for key in [it.key for it in store.search(ns_u2, limit=100)]:
            store.delete(ns_u2, key)

        store.put(ns_u1, "profile", {"data": "u1 的档案"})
        print(f"  在 {ns_u1} 里写入 1 条；在 {ns_u2} 里写入 0 条")
        print(f"  搜 {ns_u1} → {[it.value['data'] for it in store.search(ns_u1)]}")
        print(f"  搜 {ns_u2} → {[it.value['data'] for it in store.search(ns_u2)]}   ← 空，搜不到 u1 的东西")
        print()

        # ---------- 3.2 课案原文流程：构建图并跑两轮 ----------
        print("=" * 74)
        print("② 课案原文流程：第一次写入记忆，第二次自动检索到")
        print("=" * 74)
        builder = StateGraph(State)
        builder.add_node("chat", make_chat_node(ns_u1))  # 节点归属 u1 的记忆空间
        builder.add_edge(START, "chat")
        builder.add_edge("chat", END)
        graph = builder.compile(store=store)  # 只传 store，不传 checkpointer

        # 第一次对话：写入记忆
        graph.invoke({"messages": [{"role": "user", "content": "我叫张三"}]})
        print("  第一次对话已执行：『我叫张三』→ 已写入长期记忆")

        # 第二次对话：自动检索到 "我叫张三"
        result = graph.invoke({"messages": [{"role": "user", "content": "我是谁"}]})
        print("  第二次对话返回：", result["messages"][-1]["content"])  # 预期答出「张三」

        print("\n  当前 u1 的记忆空间里有：")
        for it in store.search(ns_u1, limit=100):
            print(f"      key={it.key!r:<14} value={it.value}")
        print()

        # ---------- 3.3 对照：换个 namespace，同一张图也「不认识」你 ----------
        print("=" * 74)
        print("③ 对照：把 namespace 换成 u2，同一张图立刻「失忆」")
        print("=" * 74)
        builder2 = StateGraph(State)
        builder2.add_node("chat", make_chat_node(ns_u2))
        builder2.add_edge(START, "chat")
        builder2.add_edge("chat", END)
        graph2 = builder2.compile(store=store)
        result2 = graph2.invoke({"messages": [{"role": "user", "content": "我是谁"}]})
        print("  u2 问『我是谁』→", result2["messages"][-1]["content"])
        print("  ↑ 同一个 Store、同一张图结构，只因 namespace 不同就查不到 u1 的记忆。")
        print()

        print("=" * 74)
        print("小结：checkpointer 管「这次会话聊了什么」，Store 管「这个用户是谁」；")
        print("      namespace 元组是多租户隔离的关键，一个字段都不要省。")
        print("=" * 74)


if __name__ == "__main__":
    import psycopg

    try:
        main()
    except psycopg.OperationalError as exc:
        print("=" * 74)
        print("无法连接 PostgreSQL，请先确认服务已启动。")
        print("=" * 74)
        print(f"  错误信息：{str(exc).splitlines()[0]}")
        print("  启动命令见 03_短期记忆_生产_jxsd.py 的文件头说明。")
        sys.exit(0)

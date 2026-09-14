# -*- coding: utf-8 -*-
"""
LangGraph 短期记忆（二）：PostgreSQL 生产版
================================================================
`02_短期记忆_内存_jxsd.py` 讲清了 checkpoint 概念，但它把检查点存在进程内存里，
进程一退就全没了。生产环境必须落库——本节用 PostgreSQL 把同一张图跑成「不会失忆」。

**前置准备**（课案原文，Docker 方式）：
    # 账号密码 postgres/postgres，数据库名字：postgres
    docker run -e POSTGRES_PASSWORD=<你的口令> -d --name postgres -p 5432:5432 postgres:18
    # 进入容器建库（LangGraph 的检查点表都建在这个库里）
    docker exec -it postgres psql -U postgres -c "CREATE DATABASE langgraph;"
    # 安装依赖
    uv add langgraph-checkpoint-postgres

**本项目的配置差异（重要）**：
课案里的连接串是硬编码的 `postgresql://<用户名>:<口令>@127.0.0.1:5432/langgraph`，
本项目一律改用 `settings.pg_uri`（值来自根目录 `.env` 的 `PG_URI`）。
连接串里内嵌账号口令，**写进代码就等于把凭据提交进仓库**，任何情况下都不要这样做。
本文件在启动时会检查 `settings.pg_uri` 是否为空，为空则打印中文提示并优雅退出。

**本节实现课案的两个完整代码块**：
    ① `with PostgresSaver.from_conn_string(...)` —— 最简写法，适合脚本 / 小流量
    ② 生产环境最佳实践：`psycopg_pool.ConnectionPool(min_size/max_size)`
       + `try/finally: pool.close()` —— 适合常驻服务
    外加课案结尾提到的「定期清理旧 checkpoint」，并解释为什么必须清理。

**①和②到底差在哪？**
    | 维度       | ① from_conn_string              | ② ConnectionPool                    |
    |------------|----------------------------------|-------------------------------------|
    | 连接管理   | 内部临时建池，with 退出即关闭    | 自己建池，进程活多久池就活多久      |
    | 连接数     | 默认 1 条（够脚本用）            | min_size=5 / max_size=20，可扛并发  |
    | 复用       | 每次 with 都要重新握手           | 连接复用，省掉 TCP + 认证开销       |
    | 适用场景   | 一次性脚本、Demo、定时任务       | FastAPI / 常驻服务、高并发          |
    | 收尾       | with 自动关闭                    | **必须自己 finally: pool.close()**  |

课案出处：Agent 课案 → langgraph → 核心组件 → 短期记忆（完整示例 / 生产环境最佳实践）
运行方式：
    uv run Agent/01_langgraph/03_短期记忆_生产_jxsd.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，防中文/emoji 报错

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from psycopg_pool import ConnectionPool

from config import settings

# 大模型：课案本节的原文写法就是 ChatOpenAI(model=..., api_key=..., base_url=...)，
# 这里保持与课案一致（参数值仍然全部来自 settings，不硬编码）。
# 其他小节统一用 init_chat_model，两种写法等价，init_chat_model 更方便换模型供应商。
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
        print("未配置 PostgreSQL 连接串，无法演示短期记忆落库。")
        print("=" * 74)
        print("请在项目根目录 .env 中配置（连接串含账号口令，不要写进代码）：")
        print("    PG_URI=postgresql://<用户名>:<口令>@<主机>:5432/<库名>")
        print()
        print("并确认本机 PostgreSQL 已启动、且库已建好：")
        print('    docker exec -it postgres psql -U postgres -c "CREATE DATABASE langgraph;"')
        print("配好后再运行本文件。内存版概念演示见 02_短期记忆_内存_jxsd.py。")
        sys.exit(0)
    return uri


# ============================================================
# 1. 节点函数与图（①② 共用同一张图，只是 checkpointer 不同）
# ============================================================
def chat(state: MessagesState) -> dict:
    """课案原文写法：一个节点，把模型回复追加进 messages。

    对比②里的 lambda 写法：
        builder.add_node("chat", lambda s: {"messages": [llm.invoke(s["messages"])]})
    两者完全等价，具名函数可读性更好、也好加日志——本文件统一用具名函数。
    """
    return {"messages": [llm.invoke(state["messages"])]}


def build_graph(checkpointer):
    """构建「一问一答」的图，并挂上传入的 checkpointer。

    checkpointer 要求 invoke 时传 config（至少含 thread_id），
    否则 LangGraph 不知道「该读哪份记忆」，会直接报 ValueError。
    """
    builder = StateGraph(MessagesState)
    builder.add_node("chat", chat)
    builder.add_edge(START, "chat")
    builder.add_edge("chat", END)
    # 编译图（带 PostgreSQL checkpoint）
    return builder.compile(checkpointer=checkpointer)


def count_checkpoints(thread_id: str) -> int:
    """只读查询：这个 thread_id 在 checkpoints 表里落了几行。

    用来实证「记忆真的写进数据库了」，不修改任何数据。
    """
    import psycopg

    with psycopg.connect(settings.pg_uri) as conn:
        row = conn.execute(
            "SELECT count(*) FROM checkpoints WHERE thread_id = %s", (thread_id,)
        ).fetchone()
        return row[0] if row else 0


# ============================================================
# 2. 课案代码块①：with PostgresSaver.from_conn_string(...)
# ============================================================
def demo_with_conn_string(db_uri: str) -> None:
    print("=" * 74)
    print("① 课案代码块一：with PostgresSaver.from_conn_string(settings.pg_uri)")
    print("=" * 74)
    print("  连接串已从 .env 读取（课案硬编码的 postgresql://<用户名>:<口令>@... 已替换）")
    print()

    # from_conn_string 返回的是**上下文管理器**：内部帮你建好连接池，
    # with 块结束时自动关闭。适合脚本 / 定时任务这类「跑完就走」的场景。
    with PostgresSaver.from_conn_string(db_uri) as checkpointer:
        checkpointer.setup()  # 首次运行创建必要的表（幂等，重复执行不会出错）

        graph = build_graph(checkpointer)
        config = {"configurable": {"thread_id": "user1_session22"}}

        result = graph.invoke({"messages": [{"role": "user", "content": "你好"}]}, config=config)
        print("  AI：", result["messages"][-1].content)
        print()

        # 第二轮：同 thread_id —— checkpointer 会把上轮的消息从 PG 里读回来一起送模型
        result = graph.invoke({"messages": [{"role": "user", "content": "我上一句说了什么？"}]}, config=config)
        print("  AI：", result["messages"][-1].content)

        snapshot = graph.get_state(config)
        print(f"\n  本轮会话共 {len(snapshot.values['messages'])} 条消息，next={snapshot.next}")
        print(f"  数据库 checkpoints 表里该 thread_id 落库行数：{count_checkpoints('user1_session22')}")
        print("  ↑ 行数不为 0 就是「记忆真的写进了 PostgreSQL」的直接证据。")
    # with 退出：连接池已自动关闭
    print("  with 块结束，连接池已自动关闭。")
    print()


# ============================================================
# 3. 课案代码块②：生产环境最佳实践（ConnectionPool）
# ============================================================
def demo_connection_pool(db_uri: str) -> None:
    print("=" * 74)
    print("② 课案代码块二（生产最佳实践）：psycopg_pool.ConnectionPool")
    print("=" * 74)
    print("  min_size=5 / max_size=20：低负载时保底 5 条连接，高负载最多扩到 20 条。")
    print("  为什么生产要用池？每次新建连接都要 TCP 三次握手 + 认证，")
    print("  池化后这些开销只在进程启动时付一次，请求路径上直接拿现成连接。")
    print()

    # open=True（默认）时，构造 ConnectionPool 就会立刻建立 min_size 条连接。
    pool = ConnectionPool(db_uri, min_size=5, max_size=20)
    try:
        # 注意：这里直接复用池（PostgresSaver 接受连接池对象）
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()  # 创建必要的表（幂等）

        graph = build_graph(checkpointer)

        config = {"configurable": {"thread_id": "user1_session22"}}
        result = graph.invoke({"messages": [{"role": "user", "content": "你好"}]}, config=config)
        print("  AI：", result["messages"][-1].content)

        # 池的连接数会随负载动态伸缩：get_stats() 是观察它的窗口
        stats = pool.get_stats()
        print(
            f"\n  连接池状态：pool_size={stats.get('pool_size')}"
            f"  pool_available={stats.get('pool_available')}"
            f"  requests_waiting={stats.get('requests_waiting')}"
        )
        print("  （pool_size 会从 min_size=5 起步，按需增长到 max_size=20）")
    finally:
        # ⚠️ 生产环境最容易漏的一行：不关池，进程退出时连接会留在数据库侧，
        #    直到 PG 自己超时回收。常驻服务里还会造成连接泄漏，最终打满 max_connections。
        pool.close()
        print("\n  finally: pool.close() 已执行，连接池正常释放。")
    print()


# ============================================================
# 4. 定期清理旧 checkpoint（课案结尾提到，这里讲透「为什么」）
# ============================================================
def demo_cleanup(db_uri: str) -> None:
    print("=" * 74)
    print("③ 定期清理旧 checkpoint（避免数据膨胀）")
    print("=" * 74)
    print("  为什么要清理？每执行**一步**就落一个检查点，一行不省。")
    print("  同一个 thread_id 聊 100 轮，就会攒下几百行 checkpoint + writes + blobs；")
    print("  长期不清理，表会膨胀到拖慢查询、占满磁盘。")
    print()

    # -------- 课案原文的清理 SQL（原样保留，便于对照） --------
    # # 定期清理旧 checkpoint（避免数据膨胀）
    # # 可以通过定时任务执行：
    # DELETE FROM checkpoint WHERE created_at < NOW() - INTERVAL '30 days'
    #
    # ⚠️ 本机实测的版本差异（langgraph-checkpoint-postgres 3.1.2）：
    #   1. 表名是 `checkpoints`（**复数**），不是 `checkpoint`；
    #   2. checkpoints 表**没有 created_at 列**。
    #      实测列清单：thread_id / checkpoint_ns / checkpoint_id /
    #                   parent_checkpoint_id / type / checkpoint / metadata
    #      也就是说课案这句 SQL 在本版本上直接跑会报「列不存在」，
    #      需要按自己用的版本调整（升级 LangGraph 前先 SHOW COLUMNS 确认一遍）。
    #   3. 清理不能只删 checkpoints：checkpoint_writes、checkpoint_blobs 里
    #      也存着同一批 thread_id 的明细数据，要一起删，否则留下孤儿数据。
    print("  课案的清理 SQL（原样保留，注意上面的版本差异说明）：")
    print("      DELETE FROM checkpoint WHERE created_at < NOW() - INTERVAL '30 days'")
    print()

    import psycopg

    with psycopg.connect(db_uri) as conn:
        # -------- 4.1 只读盘点：看看现在库里攒了多少 --------
        print("  当前库存盘点（只读查询，不改数据）：")
        rows = conn.execute(
            "SELECT thread_id, count(*) AS n FROM checkpoints "
            "GROUP BY thread_id ORDER BY n DESC LIMIT 5"
        ).fetchall()
        for thread_id, n in rows:
            print(f"      thread_id={thread_id:<20} checkpoints={n} 行")
        print()

        # -------- 4.2 可实际执行的清理：按 thread_id 删（本文件只删自己刚建的演示线程）--------
        # 生产里常见两种策略：
        #   a) 按时间：只保留最近 N 天（需要自己记录时间戳，见上面的版本差异说明）
        #   b) 按会话：业务上已结束 / 已归档的 thread_id，整条删除（↓ 就是这种）
        demo_thread = "jxsd-cleanup-demo"
        # 先造一点数据出来，让删除有东西可删
        with PostgresSaver.from_conn_string(db_uri) as cp:
            cp.setup()
            g = build_graph(cp)
            g.invoke(
                {"messages": [{"role": "user", "content": "这条会话马上会被清理掉"}]},
                config={"configurable": {"thread_id": demo_thread}},
            )
        before = conn.execute(
            "SELECT count(*) FROM checkpoints WHERE thread_id = %s", (demo_thread,)
        ).fetchone()[0]
        print(f"  演示线程 {demo_thread} 清理前：checkpoints={before} 行")

        # 三张表一起删，避免孤儿数据
        for table in ("checkpoints", "checkpoint_writes", "checkpoint_blobs"):
            conn.execute(f"DELETE FROM {table} WHERE thread_id = %s", (demo_thread,))
        conn.commit()

        after = conn.execute(
            "SELECT count(*) FROM checkpoints WHERE thread_id = %s", (demo_thread,)
        ).fetchone()[0]
        print(f"  演示线程 {demo_thread} 清理后：checkpoints={after} 行")
        print("  ↑ 这就是「定时任务里该跑的清理动作」，接到 cron / APScheduler 即可。")
    print()


if __name__ == "__main__":
    db_uri = require_pg_uri()

    # 连不上数据库时给中文提示 + 优雅退出，而不是甩一坨 psycopg 的 traceback
    import psycopg

    try:
        demo_with_conn_string(db_uri)
        demo_connection_pool(db_uri)
        demo_cleanup(db_uri)
    except psycopg.OperationalError as exc:
        print("=" * 74)
        print("无法连接 PostgreSQL，请先确认服务已启动。")
        print("=" * 74)
        print(f"  错误信息：{str(exc).splitlines()[0]}")
        print()
        print("  启动命令（课案原文，Docker 方式）：")
        print("      docker run -e POSTGRES_PASSWORD=<你的口令> -d --name postgres -p 5432:5432 postgres:18")
        print('      docker exec -it postgres psql -U postgres -c "CREATE DATABASE langgraph;"')
        print("  并检查 .env 里的 PG_URI 主机 / 端口 / 库名是否正确。")
        sys.exit(0)

    print("=" * 74)
    print("小结：同一张图，checkpointer 从 InMemorySaver 换成 PostgresSaver，")
    print("      代码几乎不用改，记忆就从「进程内」升级成「跨进程持久化」。")
    print("=" * 74)

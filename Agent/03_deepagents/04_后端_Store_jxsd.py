# -*- coding: utf-8 -*-
"""
DeepAgents 运行环境②：StoreBackend —— 用户的长期档案柜
================================================================
StateBackend 是「本次会话的草稿本」，StoreBackend 是「用户的长期档案柜」。
它把文件保存在 LangGraph Store（`BaseStore`）里，而不是图的 state 里。

本节要讲什么
    1. 它**解决什么问题**：StateBackend 的文件跟着 `thread_id` 走，
       用户换个会话就「失忆」；StoreBackend 把文件搬进 Store，
       让记忆跨 thread、跨会话、跨用户地活下来 —— 这就是「长期记忆」的底座；
    2. `namespace` 是它的灵魂：一个「运行时对象 → 字符串元组」的函数，
       决定这批文件落在 Store 的哪个抽屉里，也是「谁能看到谁」的隔离边界；
    3. 课案注释里「部署到 LangSmith 时 store 由平台自动注入」到底是什么意思
       —— 本文件第三节逐句拆开解释（本地跑和上平台跑的区别）；
    4. 什么时候选它：用户偏好、历史资料、跨 thread 的项目资料；
       什么时候**不**选它：进程内临时草稿（那是 StateBackend 的活）。

一、和 StateBackend 的区别（课案原话）
    - StateBackend：跟着 `thread_id` 走
    - StoreBackend：可以跨 `thread_id` 读取
    适合存放：
        - 用户偏好
        - 用户历史资料
        - 长期记忆
        - 多次任务之间共享的信息
        - 跨 thread 的项目资料

二、`namespace` 是它的灵魂
    StoreBackend 的构造签名是：
        StoreBackend(*, namespace: Callable[[Runtime], tuple[str, ...]], store: BaseStore | None = None)
    `namespace` 是**一个函数**，接收运行时对象 `Runtime`，返回一个字符串元组，
    决定「这批文件放在 Store 的哪个抽屉里」：

        namespace=lambda rt: ("memories", "1")            # 本文件：写死一个抽屉
        namespace=lambda rt: (rt.server_info.user.identity,)   # 生产：按登录用户隔离

    源码里对 namespace 的每一段都做了字符白名单校验（字母数字和 - _ . @ + : ~），
    目的是防止 `*` `?` `[]` 这类通配符被注入进 store 的查询里。

三、课案注释里的那句「平台自动注入」，是本文件要重点解释的地方
    课案原文：
        agent = create_deep_agent(
            model=model,
            backend=StoreBackend(namespace=lambda rt: ("memories","1")),
            store=InMemoryStore(),
        )
        # 部署到 LangSmith 时改用：namespace=lambda rt: (rt.server_info.user.identity,)
        # 且 store 由平台自动注入，无需手动传

    拆开看是两件事：

    1）`namespace=lambda rt: (rt.server_info.user.identity,)` 为什么本地写不了？
       `rt` 是 `langgraph.runtime.Runtime`，它的 `server_info` 字段**只有跑在
       LangGraph Server / LangSmith Deployment 上时才存在**（由服务端注入，
       里面装着 user.identity、assistant_id、thread_id 等运行时身份信息）。
       本地用 `agent.invoke(...)` 直跑时根本没有「服务器」，访问 rt.server_info
       会直接报错。所以本地只能退化成「固定元组」或自己从 config 里取 user_id。

    2）`store` 为什么「由平台自动注入，无需手动传」？
       本地跑的时候，图是我们自己 compile 的，Store 对象得自己造、自己传：
           create_deep_agent(..., store=InMemoryStore())   # 或者 PostgresStore
       而部署到 LangSmith Deployment 后，平台的 `langgraph.json` / 部署配置里
       已经声明了 Store（生产一般配 PostgresStore），运行时由图服务器自动
       compile 进图里。此时 StoreBackend 内部走的是 LangGraph 的 `get_store()`
       从运行上下文里取 Store —— **取得到，所以调用方不用传**；
       再手动传一个反而可能和平台配置冲突。

    一句话总结：
        本地 = 自己造 Store + 写死 namespace；
        上平台 = Store 由平台给 + namespace 用 rt.server_info 做真人级隔离。

四、Store 用什么实现
    - `InMemoryStore`：内存版，进程退出即丢，只适合演示（本文件用它，零外部依赖）
    - `PostgresStore`：生产版，落 PostgreSQL，重启不丢（见 11_记忆_jxsd.py）

课案出处：Agent 课案 → deepAgents → 运行环境 → ② StoreBackend

运行方式：
    uv run Agent/03_deepagents/04_后端_Store_jxsd.py

前置条件：.env 里的 api_key / base_url / model_name 可调通（实测可用）。
         本文件用 InMemoryStore，**不需要** PostgreSQL。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from deepagents import create_deep_agent
from deepagents.backends import StoreBackend
from langchain.chat_models import init_chat_model
from langgraph.store.memory import InMemoryStore
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

# 本地演示用内存 Store：零外部依赖，但进程退出就没了。
# 生产环境把这一行换成 PostgresStore.from_conn_string(settings.pg_uri) 即可。
store = InMemoryStore()

agent = create_deep_agent(
    model=llm,
    # namespace 决定文件落在 Store 的哪个抽屉：
    # 这里写死 ("memories", "1")，让所有 thread 都读写同一个抽屉，
    # 从而直观地演示「跨 thread」这件事。
    # 生产环境改成：namespace=lambda rt: (rt.server_info.user.identity,)（原因见文件头第三节）
    backend=StoreBackend(namespace=lambda rt: ("memories", "1")),
    # 本地必须自己把 store 交给图；部署到 LangSmith 时这行去掉 —— store 由平台自动注入
    store=store,
)

if __name__ == "__main__":
    # ---------- ① 第一个 thread：让 Agent 写一个文件 ----------
    # 注意课案的措辞是「创建一个py文件，写冒泡排序」，
    # 在 StoreBackend 下这个 py 文件只是被**写进 Store**，不会被真的执行。
    print("===== ① thread_id='store-A'：创建文件 =====")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "创建一个py文件，写冒泡排序"}]},
        config={"configurable": {"thread_id": "store-A"}, "recursion_limit": 50},
    )
    print(result["messages"][-1].content)

    # ---------- ② 换一个 thread：文件依然在（这就是 StoreBackend 的意义） ----------
    # 课案原文第二问就是「当前root目录有哪些文件」，用来验证跨 thread 可见。
    print("\n===== ② thread_id='store-B'（新会话）：文件还在吗？=====")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "当前root目录有哪些文件"}]},
        config={"configurable": {"thread_id": "store-B"}, "recursion_limit": 50},
    )
    print(result["messages"][-1].content)

    # ---------- ③ 绕过 Agent，直接翻 Store 看数据到底存在哪 ----------
    # 这一步是「证据」：证明文件确实写在 Store 的 ("memories","1") 命名空间下，
    # 而不是写在 state 里、也不是写在磁盘上。
    print("\n===== ③ 直接查 Store，验证物理落点 =====")
    # search(ns) 不带 query 时就是「列出这个命名空间下的全部条目」。
    # 注意：StoreBackend 是**拿路径直接当 key** 的，所以这里 key 长成 /xxx.py 的样子。
    items = store.search(("memories", "1"))
    if not items:
        # 模型偶发不发 tool_calls（本机模型已知波动，见 README），此时 Store 是空的，
        # 打印这句提示避免学生误以为代码写错了
        print("Store 里暂无数据（模型本轮可能没调用 write_file）")
    for item in items:
        print(f"  key = {item.key}")
        # Item.value 就是当初写进去的 FileData 字典（content/encoding/created_at/modified_at）
        value = item.value or {}
        # 只截前 200 字并压掉换行，免得一屏被源码刷满
        content = str(value.get("content", ""))[:200].replace("\n", " ")
        print(f"  content 前 200 字 = {content}")
    print(f"\n共 {len(items)} 条记录，位于 namespace ('memories', '1')")

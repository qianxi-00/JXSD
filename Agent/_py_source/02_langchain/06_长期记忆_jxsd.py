# -*- coding: utf-8 -*-
r"""
LangChain 智能体：长期记忆（store + ToolRuntime）
================================================================
课案原文（和 LangGraph 一样，要通过 `runtime.store` 手动读写；
LangChain 中在自定义工具里通过 `ToolRuntime` 访问）：

    from langchain.tools import ToolRuntime
    from langgraph.store.postgres import PostgresStore

    @tool
    def remember(info: str, runtime: ToolRuntime) -> str:
        \"\"\"记住用户信息\"\"\"
        runtime.store.put(("memories", "u1"), "info", {"data": info})
        return f"已记住: {info}"

    @tool
    def recall(query: str, runtime: ToolRuntime) -> str:
        \"\"\"回忆用户之前的信息\"\"\"
        mems = runtime.store.search(("memories", "u1"), query=query, limit=3)
        return "；".join([m.value["data"] for m in mems]) or "暂无"

    with PostgresStore.from_conn_string("postgresql://...") as store:
        store.setup()
        store.put(("memories", "u1"), "info", {"data": "用户叫张三，25岁"})   # 模拟历史记忆
        agent = create_agent(model=model, tools=[remember, recall], store=store)
        result = agent.invoke({"messages": [{"role": "user", "content": "我叫什么？多大？"}]})
        print(result["messages"][-1].content)

本项目把那段连接串字面量换成 `settings.pg_uri`（铁律 3）。

短期记忆 vs 长期记忆：

    # | 维度       | 短期记忆（checkpointer）        | 长期记忆（store）                     |
    # |-----------|--------------------------------|--------------------------------------|
    # | 存放内容   | 整个会话的消息历史              | 你主动写进去的事实 / 偏好 / 知识       |
    # | 作用域     | thread_id（会话级）             | namespace（按用户、按业务自定义）      |
    # | 谁写入     | 框架自动写                      | **必须自己写**（工具或中间件里 put）   |
    # | 谁读出来   | 框架自动拼进 messages           | **必须自己读**（工具里 search / get）  |
    # | 跨会话     | 换 thread_id 就没了             | 换 thread_id 依然在                   |

为什么长期记忆要「手动」？因为「什么值得长期记住」是业务判断，
框架替你决定不了：用户说「我讨厌香菜」是该记的，说「今天几号」就不该记。
所以课案把它做成两个工具，让**模型自己决定**何时记住、何时回忆。

存储结构是三元组（namespace, key, value）：

    namespace = ("memories", "u1")   # 命名空间，元组形式，第一段通常是类别，第二段是用户/租户
    key       = "info"               # 条目名，同一个 namespace 下唯一，重复 put 是覆盖
    value     = {"data": "..."}      # 任意 JSON（dict），这里约定用一个 data 字段装正文

课案出处：Agent 课案 → langChain → 核心组件 → 长期记忆

前置条件与运行方式：
    - 根目录 `.env` 的 `API_KEY` / `BASE_URL` / `MODEL_NAME` 已填好；
    - 长期记忆依赖 PostgreSQL，`PG_URI` 必须已配且库 `langgraph` 已建；
      `PG_URI` 为空或连不上时 `main()` 打印中文提示后直接返回，**不抛 traceback**；
    - 在项目根目录 `F:\ProGram\Python_Base` 下执行（否则 import 不到 `config`）：
    uv run Agent/02_langchain/06_长期记忆_jxsd.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from langgraph.store.postgres import PostgresStore
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


# ---------- 1. 课案原文的两个工具：写记忆 / 读记忆 ----------
# 关键点：ToolRuntime 是「注入参数」，不是模型要填的参数 ——
# 所以工具 schema 里只会出现 info / query，模型看不到 runtime 这一项。
# runtime 上挂着 store、state、config 等运行时上下文，是工具访问长期记忆的唯一入口。
@tool
def remember(info: str, runtime: ToolRuntime) -> str:
    """记住用户信息"""
    # namespace 固定为 ("memories", "u1")：课案用一个假用户 id 演示
    runtime.store.put(("memories", "u1"), "info", {"data": info})
    return f"已记住: {info}"


@tool
def recall(query: str, runtime: ToolRuntime) -> str:
    """回忆用户之前的信息"""
    # search = 按自然语言查询；配了 embedding 的 store 会做语义检索，
    # 没配 embedding 时退化为「列出该 namespace 下的条目」（本项目即如此）。
    # 所以下面这句「or "暂无"」不是多余的：search 返回空列表时 join 出来是空串，
    # 模型会收到一个空白 ToolMessage，容易误判成「记忆功能坏了」。
    mems = runtime.store.search(("memories", "u1"), query=query, limit=3)
    return "；".join([m.value["data"] for m in mems]) or "暂无"


def main() -> None:
    if not settings.pg_uri:
        print("[跳过] settings.pg_uri 为空：长期记忆的 PostgresStore 需要 PostgreSQL。")
        print("       请在项目根目录 .env 里配置 PG_URI=postgresql://<用户>:<口令>@127.0.0.1:5432/langgraph")
        return

    try:
        # ---------- 2. 课案原文：把 store 挂到 agent 上 ----------
        with PostgresStore.from_conn_string(settings.pg_uri) as store:
            store.setup()   # 首次运行建表（幂等）

            # 模拟「上一个会话早就存下来的记忆」：
            # 注意这一步发生在 agent 运行之前，说明记忆是先于对话存在的。
            store.put(("memories", "u1"), "info", {"data": "用户叫张三，25岁"})

            # 课案原文是 create_agent(model=model, tools=[remember, recall], store=store)。
            # 这里只多补了一句 system_prompt：本机模型比较"随性"，
            # 不明确要求它「先查记忆再回答」时经常直接凭人设作答，演不出长期记忆的效果。
            agent = create_agent(
                model=llm,
                tools=[remember, recall],
                store=store,
                system_prompt=(
                    "你是私人助理。回答任何关于用户个人信息（姓名、年龄、住址、喜好）的问题前，"
                    "必须先调用 recall 工具查询长期记忆，不要凭空回答；"
                    "用户要求你记住某事时调用 remember 工具。"
                ),
            )

            # ---------- 3. 课案原文调用：让模型自己决定用 recall 工具 ----------
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "我叫什么？多大？"}]}
            )
            print("===== 3. 模型凭长期记忆回答（没有历史消息） =====")
            print("AI：", result["messages"][-1].content)
            print("\n  完整轨迹（期望能看到模型主动调用了 recall）：")
            for index, msg in enumerate(result["messages"], start=1):
                print(f"    [{index}] {msg.type:<7} {str(msg.content)[:60]}")

            called_recall = any(
                getattr(m, "tool_calls", None) and any(c["name"] == "recall" for c in m.tool_calls)
                for m in result["messages"]
            )
            if not called_recall:
                # 本机模型偶发「不调工具直接作答」，这不是代码问题，提示一下避免误判
                print("  ⚠ 本轮模型没有调用 recall（本机模型偶发行为），上面的回答不代表长期记忆失效。")
                print("    下方第 4 步直接读写 store，可以无条件验证记忆确实存着。")

            # ---------- 4. 直接验证 store 的增删改查（不经过模型） ----------
            # 第 3 步依赖模型「愿意调工具」，这一步绕开模型直接读写 store ——
            # 教学上很重要：把「存储是否正常」和「模型是否听话」两件事拆开验证，
            # 否则模型不发 tool_calls 时你会误以为是长期记忆坏了。
            print("\n===== 4. 直接操作 store =====")
            store.put(("memories", "u1"), "hobby", {"data": "用户喜欢打羽毛球"})
            item = store.get(("memories", "u1"), "hobby")
            print("  get(('memories','u1'), 'hobby') →", item.value if item else None)

            mems = store.search(("memories", "u1"), query="用户的爱好", limit=3)
            print("  search(query='用户的爱好') →", [(m.key, m.value["data"]) for m in mems])

            # ---------- 5. 换 namespace = 换一个人的记忆 ----------
            # 长期记忆按 namespace 隔离，和 thread_id 无关：
            # 也就是「换会话仍在」，这正是「长期」二字的含义。
            store.put(("memories", "u2"), "info", {"data": "用户叫李四，30岁"})
            print("\n===== 5. namespace 隔离 =====")
            print("  u1 →", [m.value["data"] for m in store.search(("memories", "u1"), limit=10)])
            print("  u2 →", [m.value["data"] for m in store.search(("memories", "u2"), limit=10)])

            # ---------- 6. 让模型真的「写入」一条新记忆 ----------
            # 注意这次 invoke 没有传 checkpointer，所以模型手里**没有**第 3 步的对话历史；
            # 它仍然能被记住，全靠 store 跨会话存在 —— 这就是长期记忆与短期记忆的分界。
            print("\n===== 6. 让模型调用 remember 工具写新记忆 =====")
            result = agent.invoke(
                {"messages": [{"role": "user", "content": "记住：我住在南昌，平时用 Python 写后端。"}]}
            )
            print("AI：", result["messages"][-1].content)
            print("  写入后 u1 的全部记忆：",
                  [m.value["data"] for m in store.search(("memories", "u1"), limit=10)])

    except Exception as exc:
        print("[跳过] 连接 PostgreSQL 失败：", type(exc).__name__)
        print("       请确认本机 PostgreSQL 已启动、库 langgraph 已创建、.env 的 PG_URI 正确。")
        print("       详情：", str(exc)[:200])


if __name__ == "__main__":
    main()

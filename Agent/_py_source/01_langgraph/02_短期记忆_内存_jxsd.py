# -*- coding: utf-8 -*-
"""
LangGraph 短期记忆（一）：内存版 Checkpointer
================================================================
> 课案此处直接上 PostgreSQL（PostgresSaver）。这里先用**内存版**把 checkpoint
> 这个概念本身讲透，不引入任何外部依赖；生产版（连接池 + 落库）见
> `03_短期记忆_生产_jxsd.py`。

**为什么需要 checkpointer？**
默认情况下，`graph.invoke()` 跑完就结束了，状态随函数返回，进程里不留痕迹。
于是每次调用都是一个「失忆的新 Agent」，不可能多轮对话。
给图挂上 checkpointer（检查点）之后，LangGraph 会在**每一步执行后**把完整状态
存下来。第二次带着同一个 `thread_id` 调用时，它会先把历史状态读出来，
和新输入合并，再继续跑——这就是「短期记忆」。

**三个必须记住的点**：
  1. `thread_id` 是记忆的隔离键：同一个 thread_id 共享一份记忆，
     换一个 thread_id 就是一条全新的、没有记忆的会话。
  2. `compile(checkpointer=...)` 只是「装上了记忆装置」；
     真正决定读哪份记忆的是 **invoke 时传的 config**，
     所以带 checkpointer 的图**每次调用都必须传 config**（至少含 thread_id）。
  3. MemorySaver / InMemorySaver 把检查点存在**进程内存**里：
     开发调试够用，**进程一退出就全没了**（本文件最后一节会实测这一点）。
     → 生产环境用 PostgresSaver，见 03。

课案出处：Agent 课案 → langgraph → 核心组件 → 短期记忆
运行方式：
    uv run Agent/01_langgraph/02_短期记忆_内存_jxsd.py
前置条件：
    - 依赖：`langgraph`（含内置 checkpointer 模块），本项目已 uv sync 装好。
    - 配置：根目录 .env 配好 MODEL_NAME / API_KEY / BASE_URL（本文件要走 5 次真实对话）。
    - 外部服务：**不需要 PostgreSQL**——这正是本节课案分开讲内存版的原因；
      落库版见 03_短期记忆_生产_jxsd.py（需要 settings.pg_uri）。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，防中文/emoji 报错

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver, MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph

from config import settings

# 大模型统一走 config.settings（密钥不落代码，只从 .env 读）
llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


# ============================================================
# 1. 节点函数：一轮对话
# ============================================================
def chat(state: MessagesState) -> dict:
    """对话节点。

    MessagesState 是 LangGraph 内置状态，只有一个字段：
        messages: Annotated[list[AnyMessage], add_messages]
    `add_messages` 这个 reducer 会**追加**新消息；
    更妙的是，如果新消息带 id（比如人工改写历史），它会**原地更新**而不是重复追加。

    这就是为什么课程里的对话节点写起来这么短：
    只要把 llm 的回复包成列表返回，历史消息由框架维护。
    """
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # 追加到 messages 末尾（不是覆盖）


# ============================================================
# 2. 组装图
# ============================================================
builder = StateGraph(MessagesState)
builder.add_node("chat", chat)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

# 关键点：编译时传入 checkpointer，开启短期记忆。
# MemorySaver 和 InMemorySaver 是**同一个类**的两个名字
# （MemorySaver 是历史遗留别名，新代码推荐写 InMemorySaver）。
print("MemorySaver is InMemorySaver →", MemorySaver is InMemorySaver)
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)


def make_config(thread_id: str) -> dict:
    """生成 checkpointer 需要的 config。

    形如 {"configurable": {"thread_id": "..."}}——thread_id 就是「会话 ID」。
    多用户系统里，这个值一般直接取业务侧的 user_id + 会话序号。
    """
    return {"configurable": {"thread_id": thread_id}}


# ============================================================
# 3. 实测：thread_id 就是记忆的隔离墙
# ============================================================
def demo_thread_isolation() -> None:
    print("=" * 74)
    print("① thread_id 隔离：同一个 thread_id 记得住，换一个就失忆")
    print("=" * 74)
    config = make_config("user1_session22")

    # 第一轮：告诉模型自己是谁
    r1 = graph.invoke({"messages": [{"role": "user", "content": "你好，我叫张三"}]}, config=config)
    print("  第 1 轮 →", r1["messages"][-1].content)

    # 第二轮：同一 thread_id，历史消息被自动读出来一起送给模型
    r2 = graph.invoke({"messages": [{"role": "user", "content": "我叫什么名字？"}]}, config=config)
    print("  第 2 轮 →", r2["messages"][-1].content)  # 预期能答出「张三」
    print(f"  此时该会话共 {len(r2['messages'])} 条消息（2 轮 × (提问+回答)）")

    # 第三轮：换一个 thread_id —— 全新的记忆空间，模型不认识张三
    other = make_config("user2_session01")
    r3 = graph.invoke({"messages": [{"role": "user", "content": "我叫什么名字？"}]}, config=other)
    print("  换 thread_id →", r3["messages"][-1].content)  # 预期「不知道 / 你没告诉我」
    print()


# ============================================================
# 4. 实测：get_state(config) 看「记忆快照」
# ============================================================
def demo_get_state() -> None:
    print("=" * 74)
    print("② get_state(config)：查看某个会话当前的记忆快照")
    print("=" * 74)
    snapshot = graph.get_state(make_config("user1_session22"))
    print(f"  values（当前状态）   ：{len(snapshot.values['messages'])} 条消息")
    for m in snapshot.values["messages"]:
        print(f"      [{m.type:<6}] {str(m.content)[:44]}")
    # next：这张快照之后「还欠着哪些节点没跑」。
    # 图已经跑到 END，所以是空元组 ()。
    print(f"  next（待执行节点）   ：{snapshot.next}   ← () 表示已到 END，没活干了")
    print(f"  config（快照定位）   ：{snapshot.config}")
    print(f"  metadata（元信息）   ：{snapshot.metadata}")
    print()

    # get_state_history：把这条 thread 上的所有检查点**从新到旧**列出来
    history = list(graph.get_state_history(make_config("user1_session22")))
    print(f"③ get_state_history：该 thread 共 {len(history)} 个检查点（从新到旧）")
    for i, snap in enumerate(history):
        print(
            f"      [{i}] 消息数={len(snap.values.get('messages', [])):<3}"
            f" next={str(snap.next):<16} source={snap.metadata.get('source')}"
        )
    print("  说明：每执行一步（输入 / 节点 / 循环）都会落一个检查点，")
    print("        这就是「时间旅行」（08_时间旅行_jxsd.py）能回退的物理基础。")
    print()


# ============================================================
# 5. 实测：不传 config 会怎样（说清「为什么每次都要传」）
# ============================================================
def demo_missing_config() -> None:
    print("=" * 74)
    print("④ 反面教材：带 checkpointer 的图不传 config")
    print("=" * 74)
    try:
        graph.invoke({"messages": [{"role": "user", "content": "你好"}]})
        print("  居然没报错？（当前版本行为可能已变）")
    except Exception as exc:
        # 预期抛错：Checkpointer requires one or more of the following 'configurable'
        # keys: thread_id, checkpoint_ns, checkpoint_id
        print(f"  报错类型：{type(exc).__name__}")
        print(f"  报错信息：{str(exc).splitlines()[0]}")
        print("  原因：checkpointer 不知道「该读哪份记忆」，必须靠 config 里的 thread_id 指路。")
    print()


# ============================================================
# 6. 实测：进程内存在，进程一退就没
# ============================================================
def demo_restart_loses_memory() -> None:
    print("=" * 74)
    print("⑤ 内存版的天花板：进程重启即丢")
    print("=" * 74)
    # 用「换一个全新的 InMemorySaver」来模拟另一次进程启动：
    # 旧的 checkpointer 对象只活在当前进程的内存里，新进程里是空的。
    restarted = builder.compile(checkpointer=InMemorySaver())
    r = restarted.invoke(
        {"messages": [{"role": "user", "content": "我叫什么名字？"}]},
        config=make_config("user1_session22"),  # 故意用同一个 thread_id
    )
    print("  新进程（同一个 thread_id）→", r["messages"][-1].content)
    print("  ↑ 同一个 thread_id、同一个问题，但记忆没了 —— 因为检查点只存在于内存。")
    print()
    print("  结论：MemorySaver / InMemorySaver 适合开发调试；")
    print("        要让记忆活过进程重启，把 checkpointer 换成 PostgresSaver 即可，")
    print("        代码几乎不用改 —— 这正是 checkpointer 抽象的价值。")
    print("        生产写法见 03_短期记忆_生产_jxsd.py。")
    print()


if __name__ == "__main__":
    demo_thread_isolation()
    demo_get_state()
    demo_missing_config()
    demo_restart_loses_memory()


# ============================================================
# 7. 实测结论 · 与本课案的差异 · 踩坑提示
# ============================================================
# 【实测结论】
#   1. 第 ① 节：同一个 thread_id 连问两轮，模型能答出「张三」；
#      换成 user2_session01 再问同一句，模型答不知道 —— 隔离生效。
#   2. 第 ② 节：get_state(config).next 在跑到 END 之后是**空元组 ()**；
#      get_state_history 返回的检查点数量 = 「每执行一步落一个」，
#      这就是 08_时间旅行_jxsd.py 能回退的物理基础。
#   3. 第 ④ 节：带 checkpointer 的图不传 config，实测抛的是
#      ValueError: Checkpointer requires one or more of the following 'configurable' keys:
#      thread_id, checkpoint_ns, checkpoint_id —— 课案原话「checkpointer 要求传入 config，
#      至少包含 thread_id」说的就是这条报错。
#   4. 第 ⑤ 节：换一个全新的 InMemorySaver 再问「我叫什么名字」，
#      模型确实答不出来 —— 记忆只活在进程内存里。
#
# 【与本课案的差异】
#   1. **最大的差异**：课案《短期记忆》一节是直接上 PostgreSQL（PostgresSaver）的，
#      没有内存版这一节。本文件是**补出来的前置课**：先用零依赖的方式把
#      checkpoint / thread_id / config 三个概念讲透，再进 03 落库，
#      避免学员第一次接触就被 Docker + 连接串 + 建表三件事同时劝退。
#   2. 课案原文写 `with PostgresSaver.from_conn_string(DB_URI)`；本文件对应的是
#      最朴素的 `InMemorySaver()`，编译参数名一样（checkpointer=），换回去即可。
#   3. 课案的连接串是硬编码字面量，本项目的落库版（03）统一改用 settings.pg_uri。
#
# 【踩坑提示】
#   1. **config 是 invoke 的参数，不是 compile 的参数**。compile 只管「装记忆装置」，
#      读哪份记忆由 invoke 时传的 config 决定 —— 这是最容易记反的一点。
#   2. MemorySaver 和 InMemorySaver 是同一个类（本文件第 79 行直接打印 True 验证），
#      新代码统一写 InMemorySaver，看到 MemorySaver 不必以为是另一套 API。
#   3. 内存版的检查点跟着进程走。用 uvicorn --reload 或多 worker 跑开发服务时，
#      每次重载/换 worker 都会「失忆」，现象是「上一句还聊得好好的，下一句就忘了」——
#      这不是模型问题，是 checkpointer 选错了。
#   4. 同一个 checkpointer 实例可以编译出多张图，它们**共享**同一份存储；
#      thread_id 相同的两张图会互相看到对方的检查点，起名时务必带业务前缀。

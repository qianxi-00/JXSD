# -*- coding: utf-8 -*-
"""
LangGraph 流式输出（Streaming）：三种 stream_mode 对比
================================================================
大模型逐 token 生成，如果等全部生成完再返回，用户要盯着空屏好几秒。
LangGraph 支持把「图执行过程中的中间产物」实时吐出来，这就是流式输出。

课案本节演示的是 `stream_mode="messages"`（token 级、打字机效果），
并在注释里提到了另外两种模式的写法。本文件把**三种模式都实现成可运行的演示**，
依次跑一遍做对比：

  # | 模式       | 每次流出什么                        | 典型用途                        |
  # |------------|-------------------------------------|---------------------------------|
  # | "messages" | LLM 的**单个 token** + 元数据        | 打字机效果（本课案主推）        |
  # | "updates"  | 每个节点执行完的**增量更新**（dict） | 做进度条 /「正在执行 XX 节点」  |
  # | "values"   | 每个节点执行后的**完整状态**         | 实时展示完整上下文快照          |
  # | "custom"   | 节点内 get_stream_writer() 推的内容  | 自定义进度（本文件不展开）      |

**messages 模式的两个关键细节**：
  1. 它流出的是 `(message_chunk, metadata)` **二元组**，
     所以写 `for message_chunk, metadata in graph.stream(..., stream_mode="messages")`。
  2. `metadata["langgraph_node"]` 告诉你这段 token 是**哪个节点**产出的。
     图里一旦有多个节点调模型（多 Agent 场景很常见），
     就必须靠它把 token 分流——否则几个节点的输出会糊在一起。
     → 课案原句「只输出 three 节点产生的 LLM 内容」就是干这个的。

课案出处：Agent 课案 → langgraph → 核心组件 → 流式输出
运行方式：
    uv run Agent/01_langgraph/05_流式输出_jxsd.py
前置条件：
    - 依赖：`langgraph` + `langchain-openai`（本项目已 uv sync 装好）。
    - 配置：根目录 .env 配好 MODEL_NAME / API_KEY / BASE_URL。
    - 外部服务：不需要数据库。但本文件会**真实调用大模型 3 次**
      （三种 stream_mode 各跑一遍整张图，每次都走到 step_three 调模型），
      其中 messages 模式那次要等 500 字长文逐 token 吐完，是本节最慢的一段。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，防中文/emoji 报错

from typing import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph

from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

# 课案原文写死在 step_three 里的提问词。抽成常量只为「打印出来的提问」和
# 「真正发给模型的提问」保持同一个来源，避免改了一处忘了另一处。
ARTICLE_PROMPT = "请写出一篇500字的文章"


# ============================================================
# 1. 状态与三个节点（课案原文）
# ============================================================
class State(TypedDict):
    """课案原文状态：只有一个 text 字段，节点逐段往后拼接。"""

    text: str


def step_one(state: State) -> dict:
    return {"text": state["text"] + " → 步骤一"}


def step_two(state: State) -> dict:
    return {"text": state["text"] + " → 步骤二"}


def step_three(state: State) -> dict:
    """三个节点里唯一调 LLM 的节点，也是 messages 模式唯一产出 token 的节点。

    ⚠️ 注意这里用的是 `llm.invoke()`（同步、非流式调用）。
    那为什么外层 `stream_mode="messages"` 还能看到一个个 token？
    因为 messages 模式是**在 LangGraph 层面拦截回调事件**的：
    只要节点里发生的是 LangChain 的模型调用，LangGraph 就能拿到
    on_chat_model_stream 事件并逐个吐出 chunk，节点代码本身不用改。
    —— 这是 messages 模式最好用的一点：**不用为了流式去改业务代码**。
    """
    response = llm.invoke(
        [
            {"role": "user", "content": ARTICLE_PROMPT},
        ]
    )
    return {"text": response.content + " → 步骤三"}


# ============================================================
# 2. 构建图（课案原文的三节点直链）
# ============================================================
builder = StateGraph(State)
builder.add_node("one", step_one)
builder.add_node("two", step_two)
builder.add_node("three", step_three)
builder.add_edge(START, "one")
builder.add_edge("one", "two")
builder.add_edge("two", "three")
builder.add_edge("three", END)
graph = builder.compile()


# ============================================================
# 3. 模式一：messages（课案主推，打字机效果）
# ============================================================
def stream_messages() -> None:
    """课案原文实现：只输出 three 节点产生的 LLM 内容。"""
    print("=" * 74)
    print('① stream_mode="messages" —— token 级流式，打字机效果')
    print("=" * 74)
    print(f"  提问：{ARTICLE_PROMPT}")
    print("  输出：", end="", flush=True)

    emitting_nodes: set[str] = set()  # 记录「哪些节点产出过 token」，用于讲清过滤器的价值
    first_metadata: dict | None = None

    for message_chunk, metadata in graph.stream(
        {"text": "开始"},
        stream_mode="messages",
    ):
        # metadata 里带这批 token 的来路信息，第一次拿到时打印出来看看有什么
        if first_metadata is None:
            first_metadata = metadata
            print(f"\n  [调试] 第一个 chunk 的 metadata 键：{sorted(metadata.keys())}\n")

        emitting_nodes.add(metadata.get("langgraph_node"))

        # 只输出 three 节点产生的 LLM 内容
        if metadata.get("langgraph_node") == "three":
            content = message_chunk.content

            if isinstance(content, str) and content:
                print(content, end="", flush=True)

    print("\n")
    print(f"  [观察] 本次产出 token 的节点集合：{emitting_nodes}")
    print("  ↑ 本例只有 three 节点调模型，所以过滤器看着「没起作用」；")
    print("    一旦图里有多个节点调模型（多 Agent 常见），")
    print("    metadata['langgraph_node'] 就是唯一能把 token 分流的手段。")
    print(f"  [观察] metadata 样例：langgraph_node={first_metadata.get('langgraph_node')!r} "
          f"langgraph_step={first_metadata.get('langgraph_step')!r}")
    print()


# ============================================================
# 4. 模式二：updates（课案注释里被注释掉的第一种）
# ============================================================
def stream_updates() -> None:
    """课案注释原文：

    # updates：每完成一个节点就输出该节点的增量
    # for chunk in graph.stream({"text": "开始"}, stream_mode="updates"):
        # print(chunk)  # {'one': {'text': '开始 → 步骤一'}}  →  {'two': {...}}
    """
    print("=" * 74)
    print('② stream_mode="updates" —— 每完成一个节点，流出该节点的增量')
    print("=" * 74)
    print("  输出形如 {\"节点名\": {该节点返回的增量}}，节点没返回值时是 {\"节点名\": None}")
    for chunk in graph.stream({"text": "开始"}, stream_mode="updates"):
        node_name = next(iter(chunk))
        delta = chunk[node_name]
        if isinstance(delta, dict) and "text" in delta:
            # 课案原样是 print(chunk)；这里三个节点的增量都带 text，
            # 而 three 节点那段是 LLM 写的 500 字长文，全打出来会把对比结果刷没，
            # 所以只截前 60 字——流出的数据结构本身没有改动。
            shown = str(delta["text"])[:60].replace("\n", " ")
            print(f"      {node_name:<6} → text 前 60 字：{shown}…")
        else:
            print(f"      {node_name:<6} → {delta}")
    print("\n  用途：做「正在执行 XXX 节点」的进度提示最合适——")
    print("        它只给增量，不需要把整个状态搬来搬去。")
    print()


# ============================================================
# 5. 模式三：values（课案注释里被注释掉的第二种）
# ============================================================
def stream_values() -> None:
    """课案注释原文：

    # values：每完成一个节点就输出完整的当前状态
    # for chunk in graph.stream({"text": "开始"}, stream_mode="values"):
        # print(chunk["text"])  # 开始 → 开始 → 步骤一 → 开始 → 步骤一 → 步骤二
    """
    print("=" * 74)
    print('③ stream_mode="values" —— 每完成一个节点，流出完整的当前状态')
    print("=" * 74)
    print("  第一个 chunk 是**输入状态**（还没跑任何节点），之后每跑完一个节点再来一个：")
    for i, chunk in enumerate(graph.stream({"text": "开始"}, stream_mode="values")):
        # 课案原样是 print(chunk["text"])；同理这里只截前 70 字防刷屏
        text = str(chunk["text"]).replace("\n", " ")
        print(f"      [{i}] {text[:70]}…")
    print("\n  用途：需要「每步之后的完整快照」时用它（比如把中间状态渲染到前端）；")
    print("        代价是数据量大——状态越大，重复传输越浪费，这也是它和 updates 的核心差别。")
    print()


if __name__ == "__main__":
    print("三种模式依次各跑一遍，对比它们「流出的东西」有什么不同：\n")
    stream_messages()  # 课案原文：长文 + 打字机
    stream_updates()  # 课案注释：updates
    stream_values()  # 课案注释：values
    print("=" * 74)
    print("小结：messages 给 token（体验最好）、updates 给增量（最省）、values 给全量（最全）。")
    print("      三者可以同时开：stream_mode=['messages', 'updates']，")
    print("      这时每个事件是 (模式名, 数据) 二元组，按模式名分流即可。")
    print("=" * 74)


# ============================================================
# 6. 实测结论 · 与本课案的差异 · 踩坑提示
# ============================================================
# 【实测结论】三者「流出的东西」确实不同，本机跑出来的形状如下：
#   - messages：事件是 (message_chunk, metadata) 二元组。
#     metadata 里带 langgraph_node / langgraph_step / checkpoint_ns 等键
#     （第 ① 节会打印第一个 chunk 的键清单，可以直接核对）。
#     本图只有 three 节点调模型，所以 metadata['langgraph_node'] 恒为 "three"。
#   - updates ：事件是 {"节点名": 该节点返回的增量} 单键字典，节点没返回值时为 {"节点名": None}。
#   - values  ：事件是**完整状态**，而且第一个 chunk 是 invoke 的**输入状态**
#     （还没跑任何节点），所以三个节点会给出 4 个 chunk，不是 3 个。
#     这一点最容易数错，是「为什么多了一条」的答案。
#
# 【与本课案的差异】
#   1. 课案本节只实现了 `stream_mode="messages"`；updates / values 在课案源码里是
#      **被注释掉的备选写法**（本文件第 4、5 节把它们恢复成可运行的函数，
#      并把课案注释原文抄在 docstring 里对照）。
#   2. 课案原样是 `print(chunk)` / `print(chunk["text"])`，会把 500 字长文整段刷屏；
#      本文件只截前 60 / 70 字并替换换行。**流出的数据结构没有改动**，
#      只是显示做了截断（注释里已标明）。
#   3. 课案把提问词 "请写出一篇500字的文章" 写死在 step_three 里；
#      本文件抽成 ARTICLE_PROMPT 常量，保证「打印的提问」和「真正发出去的提问」
#      同源，不会改了一处忘了另一处。
#
# 【踩坑提示】
#   1. messages 模式的**去重陷阱**：同一个 token 可能因为回调层层冒泡而被重复投递，
#      组 UI 时务必按 metadata 里的节点 + run id 分流，不要直接无脑拼接。
#   2. `message_chunk.content` 不一定总是字符串（部分模型会给出 content block 列表），
#      所以要像第 ① 节那样先 `isinstance(content, str)` 再打印，否则会 TypeError。
#   3. `stream_mode` 同时开多个时返回的是 (模式名, 数据) 二元组，
#      和单开时的返回结构**不一样**，切换时记得同步改解包代码。
#   4. updates 模式拿不到 token 级粒度，values 模式拿不到「哪些是新增的」——
#      想要打字机效果只能用 messages，这是它们的分工，不是优劣。

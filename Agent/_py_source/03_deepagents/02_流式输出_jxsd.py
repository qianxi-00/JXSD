# -*- coding: utf-8 -*-
"""
DeepAgents 流式输出（stream / stream_mode）
================================================================
本节讲「怎么把 DeepAgent 的思考过程一点点吐出来」，也就是打字机效果 +
行动轨迹可视化。

本节要讲什么
    1. `agent.stream(input, stream_mode=...)` 的四种模式各吐出什么、
       分别用在什么场景（课案只演示了其中一种，本文件把主要的三种都跑一遍）；
    2. 课案代码里的一处**注释与参数对不上**（注释写 updates、参数写 messages），
       借此把「token 级」和「节点级」这两种流式的区别讲清楚；
    3. 多模式同时订阅时，事件会从裸数据变成 `(mode, data)` 二元组，
       调用方必须自己按 mode 分发 —— 这是最容易写错的一步；
    4. 拿到 token 级流式的前提：模型侧要开 streaming。

一、stream_mode 到底有哪几种（和 LangGraph 完全一致）
    LangGraph 的 `graph.stream(input, stream_mode=...)` 支持多种模式，
    DeepAgents 编译出来的就是一张 LangGraph 图，所以用法一模一样：

    | stream_mode | 每次吐出什么 | 典型用途 |
    |---|---|---|
    | "messages" | (消息块, 元数据) —— **token 级**增量，模型吐一个字就出来一个 | 打字机效果 |
    | "updates"  | {节点名: 该节点的状态增量} —— **节点级**，一步一个 | 看 Agent 调了哪个工具 |
    | "values"   | 每一步之后的**完整状态** | 调试、观察 todo / files 的变化 |
    | "custom"   | 节点内部用 get_stream_writer() 主动写的自定义事件 | 自定义进度条 |

    也可以传一个列表同时订阅多种模式，此时每个事件是 `(mode, data)` 二元组，
    需要自己按 mode 分发（本文件第二段演示）。

二、课案代码与它的一个小笔误
    课案原文（deepAgents → 流式输出）：

        # updates：每完成一个节点就输出增量
        for chunk in agent.stream(input_data, stream_mode="messages"):
            print(chunk[0].content, end="")

    这里注释写的是 updates，参数用的却是 "messages" —— 注释是从别处复制过来的。
    按参数为准：**"messages" 是 token 级流式**，`chunk[0]` 才是真正的消息块对象，
    `chunk[1]` 是元数据（含 langgraph_node 字段，告诉你这个 token 是哪个节点吐的）。
    「每完成一个节点输出增量」对应的其实是 `stream_mode="updates"`，
    本文件把两种都跑一遍，对照着看就清楚了。

三、一个实践要点
    要拿到 token 级流式，模型必须支持流式返回。本文件在建模型时显式传了
    `streaming=True`；大多数 OpenAI 兼容网关默认就开，写上更保险。

课案出处：Agent 课案 → deepAgents → 流式输出

运行方式：
    uv run Agent/03_deepagents/02_流式输出_jxsd.py

前置条件：.env 里的 api_key / base_url / model_name 可调通（实测可用）。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
    streaming=True,          # 显式开启流式，保证 token 级增量能吐出来
)

# 课案这里没传 system_prompt，本文件也不传 —— 用框架默认的深度智能体提示词，
# 顺便看看「不写人设」时它默认是什么风格。
agent = create_deep_agent(model=llm)

input_data = {"messages": [{"role": "user", "content": "如何制作披萨"}]}


def _chunk_text(chunk) -> str:
    """把消息块的内容安全地转成字符串。

    不同模型的 content 可能是 str，也可能是 [{"type": "text", "text": ...}] 这样的块列表，
    甚至夹杂 reasoning 块。统一拍平成字符串再打印，避免 TypeError。
    """
    # 取出 content；用 getattr 兜底是因为有些流式块（如纯 reasoning 块）根本没有 content
    content = getattr(chunk, "content", "")
    # 形态 1：普通文本模型，content 就是字符串，直接用
    if isinstance(content, str):
        return content
    # 形态 2：多模态 / 带推理的模型，content 是「内容块数组」，
    #         必须逐块取出 text 再拼起来，否则 print 出来是一坨 Python 对象
    if isinstance(content, list):
        parts = []
        for item in content:
            # 数组里可能混着裸字符串
            if isinstance(item, str):
                parts.append(item)
            # 也可能混着 {"type": "text", "text": "..."} 字典；
            # 用 .get("text", "") 而不是 item["text"] —— 非文本块（图片等）没有 text 键
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    # 形态 3：其他类型（数字/None 等），兜底转字符串；空值返回空串而不是 "None"
    return str(content) if content else ""


if __name__ == "__main__":
    # ---------- 1. 课案原样：stream_mode="messages"（token 级打字机） ----------
    print("===== ① stream_mode='messages'：token 级流式 =====")
    for chunk in agent.stream(
        input_data,
        stream_mode="messages",
        config={"recursion_limit": 50},
    ):
        # chunk 是 (消息块, 元数据) 二元组，课案取 chunk[0] 就是这个消息块
        message_chunk, metadata = chunk[0], chunk[1]
        text = _chunk_text(message_chunk)
        if text:
            print(text, end="", flush=True)
    print("\n")

    # ---------- 2. stream_mode="updates"：节点级增量（这才是课案注释里说的那个） ----------
    print("===== ② stream_mode='updates'：节点级增量（看 Agent 的行动轨迹） =====")
    for chunk in agent.stream(
        {"messages": [("user", "写一篇 60 字短文介绍秋天，写到 autumn.md 里再读出来")]},
        stream_mode="updates",
        config={"recursion_limit": 50},
    ):
        # dict 的 key 是节点名（model / tools / 各中间件钩子），
        # value 是这个节点写回 state 的增量。
        for node_name, update in chunk.items():
            messages = update.get("messages", []) if isinstance(update, dict) else []
            if not messages:
                # 有的中间件钩子节点只写非消息字段，这里照样把节点名报出来，
                # 否则「每完成一个节点就输出增量」会看起来什么都没发生。
                print(f"[{node_name}] （非消息增量）")
                continue
            for message in messages:
                # 工具调用：模型这一轮决定调用哪些工具、传什么参数
                for call in getattr(message, "tool_calls", None) or []:
                    print(f"[{node_name}] → 调用工具 {call.get('name')} 参数={call.get('args')}")
                # 工具返回：ToolMessage 自带 name 字段，说明是哪个工具返回的
                if type(message).__name__ == "ToolMessage":
                    preview = str(message.content)[:80].replace("\n", " ")
                    print(f"[{node_name}] ← 工具 {message.name} 返回：{preview}…")
                elif getattr(message, "content", ""):
                    # AI 的正文（模型决定不再调工具、直接回答时就是它）
                    preview = str(message.content)[:60].replace("\n", " ")
                    print(f"[{node_name}] 💬 {preview}…")

    # ---------- 3. 多模式同时订阅：事件变成 (mode, data) 二元组 ----------
    print("\n===== ③ stream_mode=['messages','updates']：一次订阅多种 =====")
    for mode, data in agent.stream(
        {"messages": [("user", "用一句话说明什么是递归")]},
        stream_mode=["messages", "updates"],
        config={"recursion_limit": 50},
    ):
        # 传列表订阅多模式后，事件统一变成 (模式名, 数据) 二元组，
        # 所以这里必须按模式名分发 —— 这正是多模式订阅最容易写错的地方：
        # 照着单模式的写法写 chunk[0]，就会把字符串 "messages" 当成消息块。
        if mode == "messages":
            # messages 模式下 data 是 (消息块, 元数据) 二元组；
            # 用 isinstance 判断是防御性写法，兼容某些版本直接给消息块的情况
            message_chunk = data[0] if isinstance(data, tuple) else data
            text = _chunk_text(message_chunk)
            if text:
                # end="" + flush：token 级流式要的就是「不换行、立刻可见」
                print(text, end="", flush=True)
        else:
            # updates 模式这里只报节点名，避免输出太吵
            print(f"\n[步骤] {list(data.keys())}")
    print()

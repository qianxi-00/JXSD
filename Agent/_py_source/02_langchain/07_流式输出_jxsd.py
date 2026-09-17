# -*- coding: utf-8 -*-
r"""
LangChain 智能体：流式输出（stream）
================================================================
课案原文（用 stream_mode="updates" 看「每一步的产出」）：

    @tool
    def get_weather(city: str) -> str:
        \"\"\"获取指定城市的天气。\"\"\"
        return f"{city}永远是晴天！"

    model = ChatOpenAI(model=settings.model_name, api_key=settings.api_key, base_url=settings.base_url)
    agent = create_agent(model=model, tools=[get_weather])

    for chunk in agent.stream(
        {"messages": [{"role": "user", "content": "北京天气怎么样？"}]},
        stream_mode="updates",
    ):
        for step, data in chunk.items():
            msgs = data.get("messages", [])
            if msgs:
                last = msgs[-1]
                text = last.content_blocks if hasattr(last, "content_blocks") else str(last.content)
                print(f"[{step}] {text}")

为什么需要流式？智能体一次 invoke 可能要跑好几秒（模型思考 → 调工具 → 再思考），
全部跑完再一次性返回，用户会觉得「卡住了」。流式把中间过程一点点吐出来。

stream_mode 的几种取值（同一份 agent，换个参数就能切换观察粒度）：

    # | stream_mode | 每次吐出什么                       | 典型用途                     |
    # |-------------|-----------------------------------|------------------------------|
    # | "values"    | 状态的**完整快照**                 | 想看全局状态怎么长出来的       |
    # | "updates"   | 本次只被**改动**的那部分（默认值）   | 看节点/工具的执行顺序（课案用它）|
    # | "messages"  | (消息分片, 元数据) 二元组，**逐 token** | 打字机效果、前端流式渲染     |
    # | "custom"    | 你自己在节点里 emit 的自定义数据     | 自定义进度条                 |
    # | "debug"     | 最啰嗦的调试事件                    | 排查框架内部行为             |

两条容易踩的坑：

    1. `stream_mode="updates"` 吐的是 `{节点名: 该节点的输出}`，
       所以课案要写两层循环：外层是每一步，内层 `.items()` 拆出节点名和数据。
    2. `stream_mode="messages"` 吐的是**分片**而不是完整消息，
       而且工具调用的参数也会以分片形式混进来；所以要先按
       `metadata["langgraph_node"] == "model"` 过滤，再判断 `content` 非空。

课案出处：Agent 课案 → langChain → 核心组件 → 流式输出

前置条件与运行方式：
    - 根目录 `.env` 的 `API_KEY` / `BASE_URL` / `MODEL_NAME` 已填好（流式依赖供应商接口支持
      `stream=True`，当前模型实测支持）；不需要数据库；
    - 本节会把同一个问题问 3 遍（updates / messages / values 各一次），
      所以模型调用次数是 3 次，比别的小节稍慢；
    - 在项目根目录下执行（否则 import 不到 `config`）：
    uv run Agent/02_langchain/07_流式输出_jxsd.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
    # 显式打开流式：让底层走流式 HTTP 接口，增量才会一点点回来。
    # 不打开也能用 stream_mode="messages"（框架会临时切换），
    # 但显式声明语义更清楚，也是本项目 07_流式输出.py 的一贯写法。
    streaming=True,
)


# ---------- 1. 课案原文的天气工具 ----------
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city}永远是晴天！"


agent = create_agent(model=llm, tools=[get_weather])


if __name__ == "__main__":
    # 同一份输入复用三次：三个 stream_mode 观察的是**同一次执行的不同侧面**，
    # 传同一个 question 才好横向对比（注意是三次独立运行，不是同一次跑三遍）。
    question = {"messages": [{"role": "user", "content": "北京天气怎么样？"}]}

    # ---------- 2. 课案原文：stream_mode="updates"（按步骤流） ----------
    print("===== 2. 课案原文：stream_mode='updates' —— 看每一步 =====")
    for chunk in agent.stream(question, stream_mode="updates"):
        # chunk 形如 {"model": {"messages": [AIMessage(...)]}} 或 {"tools": {...}}
        for step, data in chunk.items():
            msgs = data.get("messages", [])
            if msgs:
                last = msgs[-1]
                # content_blocks 是 LangChain 1.x 的结构化内容块（文本 + 工具调用一起）；
                # 老版本 / 纯文本消息没有这个属性，所以用 hasattr 兜底。
                text = last.content_blocks if hasattr(last, "content_blocks") else str(last.content)
                print(f"[{step}] {text}")

    # ---------- 3. stream_mode="messages"：逐 token 打字机 ----------
    print("\n===== 3. stream_mode='messages' —— 打字机效果 =====")
    for token, metadata in agent.stream(question, stream_mode="messages"):
        # 过滤：只要模型节点吐出来的正文；工具节点的分片和空分片都跳过。
        if token.content and metadata.get("langgraph_node") == "model":
            print(token.content, end="", flush=True)   # flush 保证立刻显示，不被行缓冲吞掉
    print()   # 收尾换行

    # ---------- 4. stream_mode="values"：看状态快照的演变 ----------
    print("\n===== 4. stream_mode='values' —— 状态快照 =====")
    for snapshot in agent.stream(question, stream_mode="values"):
        # 每个快照都是「此刻的完整状态」，所以消息条数会单调增长：1 → 2 → 3 → 4
        print(f"  快照消息数：{len(snapshot['messages'])}，"
              f"最新一条：{type(snapshot['messages'][-1]).__name__}")

    # ---------- 5. 异步流式（Web 服务里常用） ----------
    # FastAPI / WebSocket 场景下用 astream，边收边推给前端：
    #     async for token, metadata in agent.astream(question, stream_mode="messages"):
    #         await websocket.send_text(token.content)
    # 本节故意**只演示同步版**：逻辑与异步版一模一样，只是 for / async for 之差，
    # 真跑异步还得额外起 asyncio 事件循环，反而盖住了「流式」这个主题。
    print("\n===== 5. 异步流式 =====")
    print("  agent 同时提供 ainvoke / astream，接口与同步版一一对应（见文件头注释示例）")

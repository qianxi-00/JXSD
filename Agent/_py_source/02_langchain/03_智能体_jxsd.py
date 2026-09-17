# -*- coding: utf-8 -*-
"""
LangChain 核心组件（三）：智能体（create_agent）
================================================================
课案原文（最小可用示例，只要一个模型就能跑）：

    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI
    from config import setting

    model = ChatOpenAI(model=setting.MODEL_NAME, api_key=setting.API_KEY, base_url=setting.BASE_URL)
    agent = create_agent(model=model, tools=[])
    result = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
    print(result["messages"][-1].content)

`create_agent` 干的事：把一个「裸模型」包装成一张 LangGraph 状态图：

    用户输入 → [模型 → 判断要不要调工具 → 调工具 → 结果回传] × N → 输出

这张图就是 ReAct 循环（Reason + Act）。它带来的三个变化：

    1. 输入输出都变成「消息列表」：进 `{"messages": [...]}`，出 `result["messages"]`，
       最后一条就是最终答复（用 [-1] 取）。
    2. 模型可以自己决定调用工具、拿到结果后继续推理，直到给出纯文本回复才停。
    3. 因为底层是 LangGraph 图，所以天然支持 checkpointer（短期记忆，见 05）、
       store（长期记忆，见 06）、middleware（钩子，见 10）。

关键参数（本项目用得到的）：

    # | 参数            | 作用                                                 | 本节是否用 |
    # | model          | 必填。大模型对象（init_chat_model 的返回值）           | 是        |
    # | tools          | 工具列表。传 [] 就是「不挂任何工具」的纯聊天智能体      | 是        |
    # | system_prompt  | 系统提示词（人设 / 规则），等价于一条 SystemMessage     | 是        |
    # | middleware     | 中间件列表，见 10 / 11                                | 否        |
    # | response_format| 结构化输出，见 08                                     | 否        |
    # | checkpointer   | 短期记忆（线程级持久化），见 05                        | 否        |
    # | store          | 长期记忆（跨线程共享），见 06                          | 否        |
    # | state_schema   | 自定义状态结构，见 13 多 Agent 交接                    | 否        |

课案出处：Agent 课案 → langChain → 核心组件 → 智能体

前置条件：
    - 根目录 `.env` 的 `API_KEY` / `BASE_URL` / `MODEL_NAME` 已填好；
    - 不需要数据库。本节的天气工具是写死的假数据，不联网。

运行方式（在项目根目录下执行，否则 import 不到 `config`）：
    uv run Agent/02_langchain/03_智能体_jxsd.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from config import settings

# 课案这里写的是 ChatOpenAI(model=..., api_key=..., base_url=...)；
# 本项目统一用 init_chat_model，参数全部来自 settings（见 01_模型_jxsd.py 的对比表）。
llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


# ---------- 1. 挂一个工具，用来观察 ReAct 循环 ----------
@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。city：城市名称，如「上海」"""
    # 演示用：真实项目里这里会去调天气 API
    return f"{city} 晴，25 度，适合出行"


# ---------- 2. 课案原文的最小智能体：不挂工具 ----------
# tools=[] 时图上没有「工具节点」，模型只能直接回答，
# 这也解释了为什么它和裸模型 invoke 的效果几乎一样。
plain_agent = create_agent(model=llm, tools=[])

# ---------- 3. 带工具 + 人设的智能体 ----------
agent = create_agent(
    model=llm,
    tools=[get_weather],                                    # 工具列表
    system_prompt="你是一个天气助手，回答前先调用天气工具查询。",   # 人设 / 规则
)


def dump_trace(result: dict) -> None:
    """把一次 invoke 的完整执行轨迹打印出来，看清 ReAct 循环的每一步。"""
    for index, msg in enumerate(result["messages"], start=1):
        if isinstance(msg, ToolMessage):
            # ToolMessage = 工具执行结果，是「外部世界」回传给模型的信息
            print(f"  [{index}] {msg.type:<7} ← {msg.name} 工具返回：{msg.content}")
        elif msg.type == "ai" and getattr(msg, "tool_calls", None):
            # AIMessage 带 tool_calls = 模型决定「先别回答，我要调工具」
            for call in msg.tool_calls:
                print(f"  [{index}] {msg.type:<7} → 模型要求调用 {call['name']}，参数 {call['args']}")
        else:
            print(f"  [{index}] {msg.type:<7} {msg.content}")


if __name__ == "__main__":
    # ---------- 4. 课案原文调用 ----------
    # 两个 agent 共用同一个 llm 对象是安全的：模型对象无状态（见 01_模型_jxsd.py 第 7 步），
    # 「记得什么」由 agent 这张图的状态决定，不写在模型里。
    result = plain_agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
    print("===== 4. 课案最小示例（tools=[]） =====")
    print("返回类型：", type(result).__name__)
    print("消息条数：", len(result["messages"]))
    print("最后一条：", result["messages"][-1].content)

    # ---------- 5. 观察 ReAct 循环：模型 → 工具 → 模型 ----------
    print("\n===== 5. 带工具的智能体：完整轨迹 =====")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "上海今天天气怎么样？适合出去玩吗？"}]}
    )
    dump_trace(result)
    print("\n最终答复：", result["messages"][-1].content)

    # ---------- 6. 结果字典里都有什么 ----------
    # 除了 messages，智能体还可能往状态里写别的键（比如 11 节的 todos），
    # 这里只打印键名，保证输出稳定。
    print("\n===== 6. 状态字典的键 =====")
    print(" ", list(result.keys()))
    print("  当前消息数：", len(result["messages"]))

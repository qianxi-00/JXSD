# -*- coding: utf-8 -*-
"""
Function Call ⑤（课案完整版）：DeepAgents 版智能体（agent_deepagents_jxsd.py）
================================================================
课案原话：「DeepAgents 封装了 LangChain 的 create_agent，内置文件系统、
任务拆解等中间件。对工具调用来说，API 完全一致，只是换成 create_deep_agent」。

所以要看清这个文件，关键是分清「一样」和「不一样」：

    一样的地方（课案重点强调的）
        - 工具定义：还是 @tool 装饰器，一个字都不用改；
        - 调用方式：还是 agent.invoke({"messages": [...]})，连返回结构都一样；
        - 模型对象：还是同一个 ChatOpenAI / init_chat_model。

    不一样的地方（课案的对比表里写的「额外能力」）
        - 内置文件系统：ls / read_file / write_file / edit_file 这类工具**自动就有了**，
          不用自己定义，模型可以直接读写文件；
        - 内置任务拆解：write_todos 工具，模型会把复杂任务拆成待办清单逐步勾掉；
        - 内置子 Agent 委派：task 工具，可把子任务外包给临时的子 Agent；
        - 内置上下文压缩：对话太长时自动摘要，避免撑爆上下文窗口。

一句话（课案原文）：`create_deep_agent` = `create_agent` + 全套预装中间件。

**本节要讲什么**（两个演示，正文章节一一对应）：
  ① 课案原文的最小示例：同一批 @tool 工具、同一个题库，只把 create_agent 换成
     create_deep_agent，验证「API 完全一致」这句话到底成不成立；
  ② 用上课案对比表里说的「额外能力」：把 FilesystemMiddleware / TodoListMiddleware
     装进来的内置工具清单打印出来（write_file / write_todos …），
     再让模型真的调一次 write_file —— 这些工具**我们一个都没定义**。

课案出处：Agent 课案 → 工具调用 → function call → deepagents（含对比表）

运行方式：
    uv run Agent/04_function_call/agent_deepagents_jxsd.py
前置条件：
    - 依赖：`deepagents` + `langchain` + `langchain-core`，本项目已 uv sync 装好。
      缺 deepagents 时本文件会在 import 阶段就报 ModuleNotFoundError
      （不像 01_langgraph 的对照实验那样有兜底），安装命令：uv add deepagents。
    - 配置：根目录 .env 配好 MODEL_NAME / API_KEY / BASE_URL ——
      统一 `from config import settings` 读取；课案原文用 ChatOpenAI 直构造。
    - 网络：需要 OpenAI 兼容接口。本文件会**真实调用大模型多次**，会产生 token 费用；
      演示 2 还带 `config={"recursion_limit": 50}`，中间件会多跑好几轮。
    - 磁盘：演示 2 里 write_file 写进的是 DeepAgents 的**状态后端**（默认 StateBackend，
      存在图状态里），**不会真的在你磁盘上建文件**，所以放心跑，不会污染仓库。
    - 不需要数据库、不需要起服务。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

from config import settings

# ================================================================
# 一、工具定义（课案原样）—— 和 LangChain 那节完全一样
# ================================================================
# 「工具定义不变」是课案这一节的核心结论：
# 从 create_agent 换到 create_deep_agent，你的工具函数一行都不用动。

@tool
def add_tool(a: float, b: float) -> float:
    """返回 a + b 的结果"""
    return a + b


@tool
def sub_tool(a: float, b: float) -> float:
    """返回 a - b 的结果"""
    return a - b


@tool
def mul_tool(a: float, b: float) -> float:
    """返回 a * b 的结果"""
    return a * b


@tool
def div_tool(a: float, b: float) -> float:
    """返回 a / b 的结果"""
    return a / b


# ---------- 模型 ----------
# 课案原文：model = ChatOpenAI(model=setting.MODEL_NAME, api_key=..., base_url=...)
# 本项目统一用 init_chat_model（CONVENTIONS 第 3 节），效果等价
model = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

# 同一批工具，三个版本共用（OpenAI 原生 / LangChain / DeepAgents）
tools = [add_tool, sub_tool, mul_tool, div_tool]


# ================================================================
# 二、唯一区别：create_agent → create_deep_agent
# ================================================================
# 课案原文：
#     agent = create_deep_agent(
#         model=model,
#         tools=[add_tool, sub_tool, mul_tool, div_tool],
#         system_prompt="你是一个助手，可以帮助用户进行数学计算。",
#     )
agent = create_deep_agent(
    model=model,
    tools=tools,
    system_prompt="你是一个助手，可以帮助用户进行数学计算。",
)


def print_messages(result: dict) -> None:
    """
    课案原文的打印方式：
        for msg in result["messages"]:
            print(f"{msg.type}: {msg}")

    这里保持课案的结构，做了两点可读性处理（实测必需）：
        1. 把「模型提出工具调用」那一步展开成一行，否则 tool_calls
           混在整条消息对象里根本看不出来；
        2. 折叠**完全重复**的消息：本机模型有时会把同一批调用反复提好几遍，
           原样打印会刷屏几十行，看不出主线索。
    """
    printed = set()          # 去重指纹：重复出现的同一动作只打第一次

    for msg in result["messages"]:
        calls = getattr(msg, "tool_calls", None) or []
        # 给「带工具调用的 ai 消息」和「工具结果消息」各算一个指纹
        if calls:
            fingerprint = ("call", tuple((c["name"], str(c["args"])) for c in calls))
        elif getattr(msg, "name", None):
            fingerprint = ("tool", msg.name, str(msg.content))
        else:
            fingerprint = None
        if fingerprint is not None:
            if fingerprint in printed:
                continue     # 重复动作，跳过（第一次已经打过了）
            printed.add(fingerprint)

        print(f"{msg.type}: {msg.content if msg.content else ''}")
        # 关键：DeepAgents 内置的工具（write_todos / write_file ...）也会以
        # ToolMessage 的形式出现在这条消息链里——它们不是你定义的，
        # 是中间件自动加进来的，这正是「额外能力」的直观证据
        for call in calls:
            print(f"    → 提出调用 {call['name']}({call['args']})")
        if getattr(msg, "name", None):
            # ToolMessage.name = 是哪个工具的执行结果
            print(f"    ← 工具 {msg.name} 的执行结果")
        print("-" * 50)


# ================================================================
# 三、跑起来
# ================================================================
if __name__ == "__main__":
    print("=" * 62)
    print("演示 1：课案原样 —— 只问算数，看 API 是否真的完全一致")
    print("=" * 62)
    result = agent.invoke({"messages": [{"role": "user", "content": "2+4*6"}]})
    print_messages(result)
    # 预期轨迹与前面几节相同：mul_tool(4,6)=24 → add_tool(2,24)=26 → 最终答案 26
    # 唯一多出来的，是 DeepAgents 自己那套中间件在背后运转（本轮用不上就不会出现）
    # ⚠️ 本机实测：当前模型有时只调一次 mul_tool 就收工，甚至给出「结果是 24」这种
    #    漏掉加法的错误答案。要分清责任——协议和循环都是对的（工具确实被调用了、
    #    结果也确实回填了），错在模型自己的推理。这也是选模型时要实测的原因之一。

    print()
    print("=" * 62)
    print("演示 2：用上课案对比表里说的「额外能力」（任务拆解 + 文件系统）")
    print("=" * 62)
    # 先看看 create_deep_agent 到底白送了哪些工具（这是课案对比表里
    # 「额外能力：内置文件系统、任务拆解」的可核对证据）：
    # 下面这些工具我们一个都没定义，全是中间件装进来的
    print("create_deep_agent 自动装上的内置工具：")
    try:
        from deepagents.middleware import FilesystemMiddleware
        from langchain.agents.middleware import TodoListMiddleware
        # FilesystemMiddleware 负责文件系统那一套
        print(f"  文件系统（FilesystemMiddleware）：{[t.name for t in FilesystemMiddleware().tools]}")
        # TodoListMiddleware 负责任务拆解：write_todos
        print(f"  任务拆解（TodoListMiddleware）  ：{[t.name for t in TodoListMiddleware().tools]}")
    except Exception as exc:                      # noqa: BLE001 —— 中间件属于框架内部实现，版本变动时不该炸掉整个演示
        print(f"  （当前 deepagents 版本取不到中间件清单：{exc}）")
    print()

    # 注意这条 prompt：里面提到的 write_file 我们**一个都没定义**——
    # 它是 create_deep_agent 预装中间件自带的工具。这跟 create_agent 是本质差别：
    # create_agent 遇到这种要求只能干瞪眼，因为它手里根本没有文件工具。
    #
    # 实测经验（多次跑下来结论很一致）：
    #   - 把要用的工具名点出来 → 模型基本都会照做，能稳定看到中间件工具被执行；
    #   - 一句话里塞三件事（列清单 + 算数 + 写文件）→ 模型经常只在正文里
    #     「复述」它打算调用哪个工具，却不真的发 tool_calls，甚至直接收工。
    #   所以这里刻意拆成一步一个小任务，保证演示能跑出结果。
    result = agent.invoke(
        {"messages": [{"role": "user", "content":
                       "请调用 write_file 工具，把 '2+4*6=26' 写入 result.md"}]},
        # 中间件会多跑好几轮，默认递归上限容易撞到，这里放宽到 50
        # （与本目录既有 deepagents 示例一致）
        config={"recursion_limit": 50},
    )
    print_messages(result)
    # 补充说明：write_file 写进的是 DeepAgents 的**状态后端**（默认 StateBackend，
    # 存在图状态里），并不会真的在你磁盘上建文件——所以放心跑，不会污染仓库。
    # 想落到真实磁盘，得换成 FilesystemBackend / 自定义 backend。
    #
    # 想再看「任务拆解」write_todos 的实战，把上面的 prompt 换成
    #   "请用 write_todos 工具把「计算 2+4*6」拆成待办清单"
    # 即可；该工具确实已注册（见上面的内置工具清单），只是当前模型有时
    # 会拿 write_file 代替它来记清单，属于模型偏好问题，不是配置问题。

    print()
    print("=" * 62)
    print("课案《LangChain create_agent vs DeepAgents create_deep_agent》对比表")
    print("（原文以注释形式保留在本文件末尾）")
    print("=" * 62)
    print("  create_agent      —— 工具定义 @tool / 调用 agent.invoke() / 无额外能力 / 精准控制")
    print("  create_deep_agent —— 工具定义完全相同 / 调用方式完全相同")
    print("                       额外：内置文件系统、任务拆解、子Agent委派、上下文压缩")
    print("                       适用：快速搭建，想立刻干活")
    print()
    print("三个版本对照小结：")
    print("  agent_openai_jxsd.py     —— 手写循环，理解原理")
    print("  agent_langchain_jxsd.py  —— create_agent / bind_tools，工具循环自动化")
    print("  agent_deepagents_jxsd.py —— create_deep_agent，在工具循环之上再送一套中间件")


# ================================================================
# 六、课案原文的对比表与一句话总结（以注释形式原样保留）
# ================================================================
# |                | LangChain create_agent            | DeepAgents create_deep_agent        |
# |----------------|-----------------------------------|-------------------------------------|
# | 工具定义        | @tool 装饰器                       | 完全相同                             |
# | 调用方式        | agent.invoke({"messages": [...]}) | 完全相同                             |
# | 额外能力        | 无                                | 内置文件系统、任务拆解、子Agent委派、上下文压缩 |
# | 适用场景        | 精准控制                           | 快速搭建，想立刻干活                    |
#
# 课案原文的一句话总结：
#     create_deep_agent = create_agent + 全套预装中间件。
#     工具调用写法一模一样，只是多了开箱即用的能力。

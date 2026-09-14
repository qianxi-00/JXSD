# -*- coding: utf-8 -*-
"""
DeepAgents 子智能体（SubAgent）
================================================================
主 Agent 可以通过 `task` 工具把子任务委派给子 Agent。
每个子 Agent 有独立的系统提示词、独立的工具集、**独立上下文窗口**。

本节要讲什么
    1. 它**解决什么问题**：一个 Agent 干所有活的三个代价
       —— 上下文被中间过程撑爆、提示词什么都要管却什么都不精、模型无法按任务换；
       子智能体把「翻资料」这类脏活关进独立上下文，只把结论带回主线；
    2. `SubAgent` 有哪些字段、哪些必填（name 唯一、description 决定委派时机）；
    3. 主 Agent 侧看不到「子 Agent 的内部过程」，只看得到一个 `task` 工具
       —— 本文件用内省把工具清单打出来验证这一点；
    4. 课案原文里三个子智能体**重名**的 bug（第 3 节），实测直接 ValueError。

一、为什么需要子智能体
    1. 上下文隔离：子 Agent 翻十几篇资料产生的中间信息，不会塞进主 Agent 的上下文；
       它只把最终结论回传一句话，主 Agent 的上下文窗口始终干净。
    2. 职责单一：每个子 Agent 只擅长一件事，提示词可以写得很窄、很准。
    3. 模型/工具可不同：调研用便宜模型、写代码用强模型，按需分配（见 SubAgent 的 model 字段）。

二、SubAgent 的字段（deepagents.middleware.subagents.SubAgent，一个 TypedDict）
    | 字段 | 必需 | 说明 |
    |---|---|---|
    | name | 是 | 唯一标识。主 Agent 调 task 时用它选人 —— **不能重名** |
    | description | 是 | 干什么用的。主 Agent 靠这句话决定「这个活该不该外包出去」 |
    | system_prompt | 否 | 子 Agent 的人设与工作规范 |
    | tools | 否 | 子 Agent 专属工具集；不填则继承主 Agent 的工具 |
    | model | 否 | 覆盖主 Agent 的模型，格式 'provider:model-name' |
    | middleware | 否 | 追加中间件（限流、日志等） |
    | interrupt_on | 否 | 给这个子 Agent 单独配人工审核（需要 checkpointer） |
    | skills | 否 | 这个子 Agent 能用的技能目录 |

    `SubAgent` 是 TypedDict，所以 `SubAgent(name=..., description=...)` 和
    直接写 `{"name": ..., "description": ...}` 完全等价，课案两种写法都能见到。

三、课案原文里的一个「坑」——本文件必须讲清楚
    课案（deepAgents → 子智能体）写的是：

        subagents=[
            SubAgent(name="researcher", description="深度调研一个课题，返回结构化报告", system_prompt="你是调研专家。回应的第一句加上'[调研中]'。"),
            SubAgent(name="researcher", description="深度调研一个课题，返回结构化报告", system_prompt="你是调研专家。回应的第一句加上'[调研中]'。"),
            SubAgent(name="researcher", description="深度调研一个课题，返回结构化报告", system_prompt="你是调研专家。回应的第一句加上'[调研中]'。"),
        ],

    **三个 SubAgent 连名字都一模一样**，明显是复制粘贴没改。实测这样写会直接报错：

        ValueError: Duplicate subagent name 'researcher'; each subagent must have a unique name.

    所以本文件的可运行代码改成三个**不同职责**的子 Agent
    （researcher / writer / reviewer），其余用法与课案一致。
    课案原文原样保留在下面的注释里，方便对照。

四、主 Agent 看到的工具
    配了 subagents 之后，主 Agent 的工具清单里会多出一个 `task`，
    参数是 (子 Agent 名字, 任务描述)。你可以在运行时内省看到它（本文件会打印）。

课案出处：Agent 课案 → deepAgents → 子智能体

运行方式：
    uv run Agent/03_deepagents/12_子智能体_jxsd.py

前置条件：.env 里的 api_key / base_url / model_name 可调通（实测可用）。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from deepagents import SubAgent, create_deep_agent
from langchain.chat_models import init_chat_model
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

# ------------------------------------------------------------------
# 课案原文（注意三个 researcher 重名，直接跑会 ValueError: Duplicate subagent name）
# ------------------------------------------------------------------
# agent = create_deep_agent(
#     model=model,
#     system_prompt="你是项目经理，复杂任务委派给子Agent执行。",
#     subagents=[
#         SubAgent(
#             name="researcher",
#             description="深度调研一个课题，返回结构化报告",
#             system_prompt="你是调研专家。回应的第一句加上'[调研中]'。",
#         ),
#         SubAgent(
#             name="researcher",
#             description="深度调研一个课题，返回结构化报告",
#             system_prompt="你是调研专家。回应的第一句加上'[调研中]'。",
#         ),
#         SubAgent(
#             name="researcher",
#             description="深度调研一个课题，返回结构化报告",
#             system_prompt="你是调研专家。回应的第一句加上'[调研中]'。",
#         )
#     ],
# )
# result = agent.invoke({"messages": [{"role": "user", "content": "调研一下Python异步编程"}]})
# for m in result["messages"]:
#     print(m)

# SubAgent：声明式子 Agent，主 Agent 通过 task 工具自动委派
# 注意三个子 Agent 的**名字各不相同**（researcher / writer / reviewer）——
# 课案原文三个都叫 researcher，实测会直接抛
# ValueError: Duplicate subagent name（详见文件头第三节）。
# description 是写给**主 Agent 的模型**看的：它靠这句话决定「这个活该不该外包给谁」，
# 所以写得越具体，委派越准（本文件三条 description 分别对应调研/写作/审查三种活）。
agent = create_deep_agent(
    model=llm,
    system_prompt="你是项目经理，复杂任务委派给子Agent执行。",
    subagents=[
        SubAgent(
            name="researcher",
            description="深度调研一个课题，返回结构化报告",
            # 让子 Agent 在自己的回答里留个记号，方便在消息轨迹里认出
            # 「这段是子 Agent 产出的」——课案用 '[调研中]' 就是这个目的。
            system_prompt="你是调研专家。回应的第一句加上'[调研中]'。",
        ),
        SubAgent(
            name="writer",
            description="把调研要点改写成通俗易懂的科普短文",
            # 每个子 Agent 有独立上下文：它只看得到 task 传进去的那段任务描述，
            # 看不到主 Agent 的整段对话 —— 这正是「上下文隔离」的落地方式。
            system_prompt="你是科普作者。回应的第一句加上'[写作中]'。输出不超过 200 字。",
        ),
        SubAgent(
            name="reviewer",
            description="审查文案的事实准确性与逻辑漏洞，指出问题并给出修改建议",
            # 给子 Agent 留「可辨识的记号」（[审稿中]）纯粹是为了教学观察：
            # 在 ToolMessage 里一眼认出这段产出是哪个子 Agent 写的
            system_prompt="你是严格的审稿人。回应的第一句加上'[审稿中]'。只挑毛病，不要重写全文。",
        ),
    ],
)


def _list_tools(agent_) -> list[str]:
    """内省：列出主 Agent 挂载的工具名（配了 subagents 之后应该能看到 task）。"""
    try:
        return sorted(agent_.nodes["tools"].bound.tools_by_name)
    except Exception:                      # noqa: BLE001 —— 内省失败不影响主流程
        return []


if __name__ == "__main__":
    print("主 Agent 的工具清单（注意 task）：")
    for name in _list_tools(agent):
        print(f"  - {name}")
    print()

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "调研一下Python异步编程"}]},
        # 子 Agent 会再各跑一轮，步数需求比单 Agent 大，放宽上限
        config={"recursion_limit": 80},
    )

    # 课案是 for m in result["messages"]: print(m) —— 把**完整消息轨迹**打出来。
    # 这里面藏着子 Agent 的产出（以 ToolMessage 的形式回到主 Agent），
    # 比只看最后一条结论信息量大得多。
    print("=" * 60)
    print("完整消息轨迹：")
    print("=" * 60)
    for m in result["messages"]:
        print(m)
        print("-" * 60)

    print("\n最终回答：")
    print(result["messages"][-1].content)

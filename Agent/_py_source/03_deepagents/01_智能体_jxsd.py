# -*- coding: utf-8 -*-
"""
DeepAgents 智能体：create_deep_agent 最小示例
================================================================
本节是 DeepAgents 的「第一眼」：用最少的代码把它跑起来，先看到效果，
再回头理解它到底比一个裸的 LangChain Agent 多了什么。

本节要讲什么
    1. `create_deep_agent(model=..., system_prompt=...)` 最小的两行怎么用
       —— 课案 h5「智能体」的代码一共只有 5 行，本文件第一段就是它；
    2. 这个框架到底「深度」在哪：一次调用就白送一整套中间件栈（见下一节），
       所以本文件专门用内省把默认工具清单真的打印出来，而不是靠嘴说；
    3. 默认 backend 是 StateBackend（文件只活在 LangGraph state 里，不落盘）
       —— 这是后面 03~09 七种后端的起点，本文件先把「默认值长什么样」定下来；
    4. 调用姿势的两个参数：`{"messages": [...]}` 的入参格式、
       `config={"recursion_limit": 50}` 为什么要放宽。

一、DeepAgents 是什么
    DeepAgents（`deepagents` 包）是 LangChain 官方推出的「深度智能体」框架，
    设计目标对标 Claude Code 那一类「自己开终端、自己读写文件、自己拆任务」
    的编码智能体。它不是又一个 Agent 封装，而是在基础 Agent 之上
    预置了一整套「智能体操作系统」中间件（middleware）：

        SkillsMiddleware          技能：按需加载的 SKILL.md 操作手册
        FilesystemMiddleware      虚拟文件系统：ls / read_file / write_file /
                                  edit_file / glob / grep / delete
        SubAgentMiddleware        子智能体：用 task 工具把活外包出去
        SummarizationMiddleware   长上下文管理：历史过长时自动摘要压缩
        MemoryMiddleware          记忆：把 /memories/ 下的文件注入系统提示词
        HumanInTheLoopMiddleware  人工审核：危险工具调用前挂起等用户批准

    一个 `create_deep_agent(...)` 调用即可获得以上全部能力。

二、它和 LangChain `create_agent` 的关系（本节的第二个知识点）
    看 deepagents 的源码（`deepagents/graph.py` 里 `create_deep_agent` 的最后一段）：

        deepagent_middleware = [SkillsMiddleware(...), FilesystemMiddleware(...),
                                SubAgentMiddleware(...), 摘要中间件, ...]   # 先拼中间件栈
        ...
        return create_agent(                      # ← 最终调用的就是 LangChain 的 create_agent
            model,
            system_prompt=final_system_prompt,
            tools=_tools,
            middleware=deepagent_middleware,
            ...
        )

    也就是说：

        langchain.agents.create_agent  = 发动机（模型 + 工具 + 中间件组成的基础 Agent 循环）
        deepagents.create_deep_agent   = 装好全套配件的整车（预置了上面那套中间件栈）

    你完全可以自己用 `create_agent` 手动拼出同样的中间件栈，只是 deepagents
    帮你把默认值配好了。所以学 deepagents 有个捷径：**看不懂的行为，就去翻
    langchain.agents.middleware 里的对应中间件**。

三、默认拿到哪些工具（本文件用内省真实打印出来）
    默认 backend 是 `StateBackend()`（见 deepagents/graph.py：
    `backend = backend if backend is not None else StateBackend()`），
    所以文件工具操作的是一个「住在 LangGraph state 里的虚拟文件系统」，
    不碰真实磁盘。工具清单里会出现 `execute`，但**非沙箱后端调用它只会返回错误提示**
    （源码注释原文：For non-sandbox backends, the `execute` tool will return an error message.），
    真正能执行命令的是 06 LocalShellBackend / 08 Sandbox。

课案出处：Agent 课案 → deepAgents → 智能体

运行方式：
    uv run Agent/03_deepagents/01_智能体_jxsd.py

前置条件：根目录 .env 里的 api_key / base_url / model_name 可调通（实测可用），
         不依赖数据库、不依赖任何外部服务。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from config import settings

# 课案原文用的是 ChatOpenAI(model=..., api_key=..., base_url=...)；
# 本项目规范（CONVENTIONS 第 3 节）要求统一走 init_chat_model，参数同样来自 settings。
llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


def _list_tools(agent) -> list[str]:
    """内省：把编译后图里 tools 节点挂载的工具名列出来。

    这只是「看一眼白送了什么」的调试手段，业务代码不需要这么写。
    路径 agent.nodes["tools"].bound.tools_by_name 属于 langgraph 内部结构，
    版本升级可能变，所以用 getattr 层层兜底，拿不到就返回空列表。
    """
    try:
        tools_by_name = agent.nodes["tools"].bound.tools_by_name
        return sorted(tools_by_name)
    except Exception:                      # noqa: BLE001 —— 内省失败不该影响主流程
        return []


if __name__ == "__main__":
    # ---------- 1. 创建智能体（就是课案那两行） ----------
    # system_prompt 决定「人设」，此处照搬课案：全栈工程师、擅长 Python。
    # 注意这里没有传 tools、没有传 backend、没有传 checkpointer —— 全部有默认值。
    agent = create_deep_agent(
        model=llm,
        system_prompt="你是一个全栈工程师，擅长 Python。",
    )

    # ---------- 2. 先看看它自带哪些工具 ----------
    print("create_deep_agent 默认挂载的工具：")
    for name in _list_tools(agent):
        print(f"  - {name}")
    print()

    # ---------- 3. 执行任务 ----------
    # 注意 invoke 的入参格式：{"messages": [...]}，和 LangGraph 一致。
    # 课案写的是 {"role": "user", "content": ...} 的字典形式；
    # 元组形式 ("user", "...") 是 LangChain 的简写，两者等价。
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "写个冒泡排序"}]},
        # recursion_limit：深度智能体一步任务会走很多轮（写文件→读文件→改文件→回答），
        # 默认 25 步容易触发 GraphRecursionError，这里放宽到 50。
        config={"recursion_limit": 50},
    )

    # result["messages"] 是**完整的消息轨迹**（Human / AI / Tool 全在里面），
    # 课案只取了最后一条 = Agent 的最终回答。
    print("=" * 60)
    print("最终回答：")
    print(result["messages"][-1].content)

    # ---------- 4. 顺带看一眼 Agent 在虚拟文件系统里留了什么 ----------
    # default backend 是 StateBackend，文件存在 state 的 files 字段里 —— 不落盘。
    files = result.get("files") or {}
    print()
    print(f"本轮共 {len(result['messages'])} 条消息；StateBackend 里留下 {len(files)} 个文件")
    for path in files:
        print(f"  - {path}")

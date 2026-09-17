# -*- coding: utf-8 -*-
"""
DeepAgents 运行环境①：StateBackend —— Agent 的草稿纸
================================================================
DeepAgents 通过 `ls` / `read_file` / `write_file` / `edit_file` / `glob` /
`grep` / `delete` 这些工具把「文件系统」暴露给 Agent，但所有操作都不是直接
落盘，而是交给一个**可插拔的后端（backend）** 转发。内置后端一共 7 种：

| 后端 | 说明 | 适用场景 |
|---|---|---|
| StateBackend（默认） | 存入 LangGraph state，同 thread 跨轮持久化，不跨 thread | Agent 草稿纸、中间结果暂存 |
| StoreBackend | 存入 LangGraph Store，跨 thread 持久化 | 长期记忆 |
| FilesystemBackend | 对接真实磁盘，仅文件操作 | 本地项目、CI/CD |
| LocalShellBackend | = FilesystemBackend + execute，可在宿主机执行任意 shell 命令 | 本地开发 CLI（仅限受控环境） |
| Sandbox | = FilesystemBackend + execute，但代码在隔离容器内运行，不触碰宿主机 | 生产环境、多租户、不可信代码 |
| ContextHubBackend | 存入 LangSmith Hub 仓库，持久化 + 版本历史 | LangSmith 原生方案，无需单独 Store |
| CompositeBackend | 路由分发：按路径前缀把不同目录分发到不同后端 | 混合策略 |

本节讲第①个，也是**默认**的那个。

本节要讲什么
    1. StateBackend 是**默认后端**：不传 `backend=` 时 deepagents 用的就是它
       （源码原句 `backend = backend if backend is not None else StateBackend()`），
       本文件把「不传」和「显式传」两种写法都建一遍做对照；
    2. 它的文件住在图的 State 里（`files` 字段），随 checkpoint 一起保存，
       **同 thread 跨轮在、换 thread 就没了** —— 本文件用三轮对话实测这一点；
    3. 所以它必须配 checkpointer，否则第二轮 invoke 时 state 是全新的；
    4. 什么时候选它：Agent 的草稿纸、执行计划、中间结果暂存；
       什么时候**不**选它：任何需要跨会话/跨用户记住的东西（那要 StoreBackend）。

一、StateBackend 是什么
    文件内容存在图的 State 里（`files` 字段），随 checkpoint 一起保存。
    源码注释原话：*Files persist within a conversation thread but not across threads.*
    —— 同一个 `thread_id` 里跨轮保留，换 `thread_id` 就看不到。

    适合存放（课案原话）：
        - Agent 的执行计划
        - 临时分析结果
        - 调研过程中的草稿
        - 当前任务的中间文件

    能力对照（课案表格）：
        | 能力 | 是否支持 |
        |---|---|
        | 同一个 thread 跨轮保留 | 是 |
        | 不同 thread 共享 | 否 |
        | 写入真实磁盘 | 否 |
        | 适合长期用户记忆 | 不太适合 |

二、默认 backend 就是它（本文件要验证的第一个结论）
    deepagents 源码里写得很直白：
        backend = backend if backend is not None else StateBackend()
    所以下面这两行**完全等价**，本文件两种都建一遍做对照：

        agent = create_deep_agent(model=llm, checkpointer=checkpointer)
        agent = create_deep_agent(model=llm, backend=StateBackend(), checkpointer=checkpointer)

三、为什么必须配 checkpointer
    StateBackend 把文件写进 state，而 state 的「跨轮记忆」是靠 checkpoint 存下来的。
    不传 checkpointer，第二轮 invoke 时 state 是全新的，文件自然就没了。
    本文件用 `InMemorySaver`（内存版，进程退出即丢）；
    生产环境换成 `PostgresSaver` 即可（见同目录 11_记忆_jxsd.py）。

课案出处：Agent 课案 → deepAgents → 运行环境 → ① StateBackend

运行方式：
    uv run Agent/03_deepagents/03_后端_State_jxsd.py

前置条件：.env 里的 api_key / base_url / model_name 可调通（实测可用），无需数据库。

【非交互环境说明】
    课案是 `while True: input("你说： ")` 的交互式聊天循环。本项目规范要求：
    非交互终端（管道/重定向）下自动改用预设问题，避免卡死或 EOFError。
    所以下面用 sys.stdin.isatty() 做了分流，交互时行为和课案完全一致。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

checkpointer = InMemorySaver()

# 课案原文：
#     # 默认就是 StateBackend，无需显式指定
#     agent = create_deep_agent(model=model, checkpointer=checkpointer)
#     # 等价于
#     # agent = create_deep_agent(model=model, backend=StateBackend(), checkpointer=checkpointer)
# 这里按「显式指定」的写法建，方便一眼看到用的是哪个后端。
agent = create_deep_agent(
    model=llm,
    backend=StateBackend(),
    checkpointer=checkpointer,
)

# ---------- 演示用的对话脚本 ----------
# 同一个 thread（thread_id="1"）内的三轮对话，专门用来观察 state 里的 files 变化：
#   第 1 轮：让 Agent 写文件 → state["files"] 里出现新条目
#   第 2 轮：让它 ls → 证明文件在同一 thread 内跨轮还在
#   第 3 轮：**换一个 thread_id** 再 ls → 证明 StateBackend 不跨 thread
PRESET_QUESTIONS = [
    "把「LangGraph 三大核心：State、Node、Edge」写进 notes.md",
    "当前目录有哪些文件？把内容读给我听",
]

config = {"configurable": {"thread_id": "1"}}
other_thread_config = {"configurable": {"thread_id": "2"}}


def _show_files(result: dict, tag: str) -> None:
    """打印本轮结束后 StateBackend 里留下的虚拟文件。"""
    files = result.get("files") or {}
    print(f"    [{tag}] state 里的文件：{sorted(files)}")


def _ask(question: str, cfg: dict) -> dict:
    """问一轮，打印回答，并顺手把 state 里的文件列出来。"""
    # 入参格式和 LangGraph 一致：{"messages": [...]}；cfg 里带 thread_id，
    # 它决定这次调用读写哪一份 checkpoint（也就是哪一份「虚拟文件系统」）。
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        # recursion_limit 放宽到 50：深度智能体一步任务会走很多轮
        # （写文件 → 读文件 → 改文件 → 回答），默认 25 步容易撞 GraphRecursionError
        config={**cfg, "recursion_limit": 50},
    )
    # 最后一次 AI 消息就是最终回答；前面的 ToolMessage 是过程，这里不打印
    print(f"AI：{result['messages'][-1].content}")
    return result


def run_interactive() -> bool:
    """交互模式：和课案一模一样。

    :return: True 表示真的和用户聊过；False 表示一开始就读到 EOF
             （有些「看着像终端但 stdin 已关闭」的环境会这样），此时回退到预设脚本。
    """
    first_round = True
    while True:
        try:
            content = input("你说： ")
        except (EOFError, KeyboardInterrupt):
            # 课案是裸的 input()；这里兜住两种中断：
            #   EOFError —— stdin 已关闭（管道/重定向/某些「看着像终端」的环境）
            #   KeyboardInterrupt —— 用户按了 Ctrl+C
            # 返回值区分「压根没聊过」和「聊过几轮才结束」，让 __main__ 决定要不要回退到预设脚本。
            print("\n（输入结束）")
            return not first_round
        first_round = False
        if content.strip().lower() in {"exit", "quit", "q"}:
            print("退出。")
            return True
        # 交互时用同一个 config（thread_id="1"），所以文件在同一 thread 内跨轮保留
        _show_files(_ask(content, config), "thread-1")


def run_preset_demo() -> None:
    """非交互模式：跑预设脚本，把三个结论验证一遍。"""
    print("===== ① 同一 thread 第 1 轮：写文件 =====")
    _show_files(_ask(PRESET_QUESTIONS[0], config), "thread-1")

    print("\n===== ② 同一 thread 第 2 轮：文件还在（跨轮保留）=====")
    _show_files(_ask(PRESET_QUESTIONS[1], config), "thread-1")

    print("\n===== ③ 换 thread_id=2：StateBackend 不跨 thread =====")
    # 同样的 state 结构，但 checkpointer 按 thread_id 分开存，
    # 所以新线程里 Agent 看到的是一张空目录。
    _show_files(_ask("当前目录有哪些文件？", other_thread_config), "thread-2")

    print("\n结论：StateBackend 的文件跟着 thread_id 走 —— 同线程跨轮在，换线程就没了。")


if __name__ == "__main__":
    # 交互终端就走课案那套 while + input；否则（管道 / 重定向 / CI）跑预设脚本。
    if sys.stdin.isatty() and run_interactive():
        pass
    else:
        run_preset_demo()

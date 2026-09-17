# -*- coding: utf-8 -*-
r"""
DeepAgents 官方补充篇：上下文治理（非课案内容，故不带 _jxsd 后缀）
================================================================
来源与定位：
    本文件对照 DeepAgents **官方文档**（/oss/python/deepagents）补课案没讲的内容。
    官方出处：
      - context-engineering.mdx → 内置上下文压缩（工具结果卸载 / 会话摘要 / 按需 compact）
      - permissions.mdx         → 文件系统权限规则（allow / deny / interrupt）
      - overview.mdx#task-planning → 任务规划 write_todos（v0.7 起改为 opt-in）

课案 03_deepagents（01~13）把「骨架」讲得很全：create_deep_agent、七种后端、
人工审核、记忆、子智能体、skills。官方文档里还缺的两整层是
「上下文工程与治理」和「代码化编排」（后者需要额外依赖，见文末）：

    # | 缺口 | 官方出处 | 优先级
    1 | 内置上下文压缩（卸载/摘要/compact 工具） | context-engineering.mdx | 高 ← 本文件 Demo 4
    2 | Runtime context + 自定义 State schema    | context-engineering.mdx | 高 ← 本文件 Demo 2/4 用到
    3 | FilesystemPermission 文件权限规则        | permissions.mdx | 高 ← 本文件 Demo 2/3
    4 | write_todos 任务规划（v0.7 起 opt-in）   | overview.mdx | 高 ← 本文件 Demo 1
    5 | Interpreters + PTC（进程内 QuickJS）     | interpreters.mdx | 高（要 pip install deepagents[quickjs]）
    6 | RubricMiddleware 评分循环（Beta）        | rubric.mdx | 高（要真实评审模型）
    7 | 动态子代理（在代码里扇出 task）          | dynamic-subagents.mdx | 中（依赖解释器）
    8 | MCP 工具接入（MCPAdapter）               | tools.mdx | 中（要起本地 MCP server）
    9 | 异步子代理 AsyncSubAgent                 | async-subagents.mdx | 中（要 Agent Protocol 服务器）
    10| 容错（官方错误分类表）                    | fault-tolerance.mdx | 中（对应中间件已在 02_langchain/11 官方补充）
    11| 自定义 Backend 协议（造自己的后端）        | backends.mdx | 中
    12| deepagents 版 RAG（检索-卸载-委派）       | rag.mdx | 中（要 embeddings）

两个本地版本要点（实测，写代码必须知道）：
    A. **write_todos 自 v0.7 起是 opt-in**：deepagents 0.7.13 的默认中间件栈不含 TodoList，
       要显式传 `middleware=[TodoListMiddleware()]`（TodoListMiddleware 来自 **langchain**，
       不是 deepagents）。
    B. 权限规则的 `mode="interrupt"` 需要 checkpointer（官方要求 deepagents>=0.6.8），
       并且它会自动并入 HITL 中间件 —— 恢复方式与工具中断完全相同：
       `Command(resume={"decisions": [{"type": "approve"}]})`。

为什么本文件全离线（0 次真实模型调用）：
    这四件事讲的都是**框架机制**（工具在不在、权限拦不拦、结果卸不卸载），
    用 ScriptedModel（继承 ChatOpenAI、覆写 _generate 按剧本返回消息）驱动即可
    100% 复现；真模型只会引入随机性。

运行方式（项目根目录下）：
    uv run Agent/03_deepagents/14_上下文治理_官方补充.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from langchain.agents.middleware import TodoListMiddleware
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import PrivateAttr

from deepagents import FilesystemPermission, create_deep_agent


# ================================================================
# 剧本模型：让「模型」按我们的剧本发起工具调用（离线可复现的关键）
# ================================================================
def ai_tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    """构造一条「模型要调工具」的 AIMessage。"""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


class ScriptedModel(ChatOpenAI):
    """按剧本依次吐消息的假模型（与 02_langchain/11_内置中间件_官方补充.py 同名类同一手法）。

    继承 ChatOpenAI、只覆写 _generate —— bind_tools / 消息校验等框架方法沿用真实现，
    唯一被替换的是「真正发 HTTP 请求」那一步，所以断网也能跑。
    """

    _script: list = PrivateAttr(default_factory=list)
    _cursor: int = PrivateAttr(default=0)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        message = self._script[self._cursor]
        self._cursor += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def make_scripted(script: list) -> ScriptedModel:
    """造一个剧本模型。api_key / base_url 传假值即可：永远不会被真正用到。"""
    model = ScriptedModel(model="scripted", api_key="offline", base_url="http://localhost:9")
    model._script = script
    return model


def show_tool_messages(result: dict, limit: int = 90) -> None:
    """打印结果里的工具消息（工具到底干了什么，全在这里）。"""
    for message in result.get("messages", []):
        if message.type == "tool":
            name = getattr(message, "name", "?")
            body = str(message.content).replace("\n", " ")
            print(f"    [ToolMessage] {name}: {body[:limit]}")


# ================================================================
# Demo 1：write_todos 是 opt-in —— v0.7 起不传就没有
# ================================================================
# 背景：老版本的 deepagents 默认就给代理装 write_todos（任务规划工具）；
#       0.7 起改为**显式选择**。这个变更不讲清楚，升级后会出现
#       「代理突然不会做计划了」的困惑 —— 本 Demo 用对照实验把差异摆出来。
def demo_1_todo_opt_in() -> None:
    print("=" * 70)
    print("Demo 1：write_todos 是 opt-in —— 不传 TodoListMiddleware 就没有这个工具")
    print("=" * 70)

    # ---- A. 默认的 create_deep_agent：没有 write_todos ----
    default_agent = create_deep_agent(
        model=make_scripted([
            ai_tool_call("write_todos", {"todos": [{"content": "调研", "status": "pending"}]}, "c1"),
            AIMessage(content="（默认代理这一轮不会有待办清单）"),
        ]),
    )
    print("\nA. 默认 create_deep_agent（不传 middleware）")
    try:
        result = default_agent.invoke({"messages": [{"role": "user", "content": "帮我规划三件事"}]})
        show_tool_messages(result)
        todos = result.get("todos")
        print(f"    state 里的 todos：{todos!r}")
        todos = result.get("todos")
        if not todos:
            print("    ↑ 默认栈里没有 TodoListMiddleware，write_todos 不是可用工具")
        else:
            print(f"    ↑ 本次默认栈**居然带了** TodoList（todos={todos}）—— "
                  "说明这个版本的默认栈变了，结论要以运行结果为准")
    except Exception as exc:  # noqa: BLE001
        print(f"    调用 write_todos 直接失败：{type(exc).__name__}: {str(exc)[:80]}")
        print("    ↑ 同样证明：默认代理没有这个工具")

    # ---- B. 显式挂上 TodoListMiddleware：立刻可用 ----
    todo_agent = create_deep_agent(
        model=make_scripted([
            ai_tool_call(
                "write_todos",
                {"todos": [
                    {"content": "调研竞品", "status": "in_progress"},
                    {"content": "设计接口", "status": "pending"},
                    {"content": "编写测试", "status": "pending"},
                ]},
                "c1",
            ),
            AIMessage(content="已排好待办清单。"),
        ]),
        # 注意：TodoListMiddleware 来自 **langchain**（deepagents 自己不导出它）
        middleware=[TodoListMiddleware()],
    )
    print("\nB. create_deep_agent(middleware=[TodoListMiddleware()])")
    result = todo_agent.invoke({"messages": [{"role": "user", "content": "帮我规划三件事"}]})
    show_tool_messages(result)
    todos = result.get("todos") or []
    print(f"    state 里的 todos（{len(todos)} 条）：")
    for index, todo in enumerate(todos, start=1):
        print(f"      {index}. [{todo.get('status')}] {todo.get('content')}")
    assert todos, "挂了 TodoListMiddleware 之后 todos 应该有内容"
    print(
        "    ↑ 同一句用户输入，差别只在于**有没有把 TodoListMiddleware 传进 middleware** ——\n"
        "      这就是 v0.7 的 opt-in 变更：想要规划能力，得自己声明。"
    )


# ================================================================
# Demo 2：FilesystemPermission（deny）—— 越权写入被拒，且不是异常
# ================================================================
# 规则形状（官方 permissions.mdx）：
#     FilesystemPermission(operations=["read"|"write"], paths=["glob 模式"],
#                          mode="allow" | "deny" | "interrupt")
#   - operations：write 覆盖 write_file / edit_file / delete；read 覆盖 ls / read_file / glob / grep
#   - 求值顺序：**先匹配先生效**；一条都不匹配时默认放行（宽松默认）
#   - paths 建议锚定（"/secrets/**"），避免 "/**/secrets" 这种到处误伤的写法
def demo_2_permission_deny() -> None:
    print("\n" + "=" * 70)
    print("Demo 2：FilesystemPermission(mode='deny') —— 越权写入被拒")
    print("=" * 70)

    agent = create_deep_agent(
        model=make_scripted([
            # 剧本：先试图往禁区写，再往允许的目录写
            ai_tool_call("write_file", {"file_path": "/secrets/token.txt", "content": "机密"}, "c1"),
            ai_tool_call("write_file", {"file_path": "/work/notes.txt", "content": "普通笔记"}, "c2"),
            AIMessage(content="该写的写了，该拦的拦了。"),
        ]),
        permissions=[
            # 禁区：写 /secrets/** 一律拒绝
            FilesystemPermission(operations=["write"], paths=["/secrets/**"], mode="deny"),
            # 白名单：/work/** 明确允许（其实默认就放行，这里写出来是为了让规则自解释）
            FilesystemPermission(operations=["read", "write"], paths=["/work/**"], mode="allow"),
        ],
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "写两个文件"}]})
    show_tool_messages(result, limit=120)
    # 结论按**实际拿到的工具消息**判断打印，不写死 —— 权限规则一变，这里要能如实反映
    tool_texts = [str(m.content) for m in result["messages"] if m.type == "tool"]
    denied = [text for text in tool_texts if "permission denied" in text.lower() or "拒绝" in text]
    written = [text for text in tool_texts if "Updated file" in text]
    print(
        f"    ↑ 实测：{len(denied)} 条被权限拒绝、{len(written)} 条写入成功 ——\n"
        "      拒绝的表现是**工具返回一句说明**（不是抛异常）；规则「先匹配先生效」，没匹配上的默认放行。\n"
        "      注意路径是 StateBackend 的虚拟文件系统路径，都以 / 开头。"
    )


# ================================================================
# Demo 3：FilesystemPermission（interrupt）—— 越权写入转为人工审批
# ================================================================
# mode="interrupt" 是本文件最值得学的一招：把「权限」和「人工审核」缝在一起 ——
# 命中规则的写操作不会被执行，而是**抛出一个人工中断**，等审批决定：
#     Command(resume={"decisions": [{"type": "approve"}]})  批准 → 真的执行
#     Command(resume={"decisions": [{"type": "reject"}]})   拒绝 → 不执行
# 它要求 checkpointer（要能暂停/恢复），且恢复格式与工具中断完全一致。
def demo_3_permission_interrupt() -> None:
    print("\n" + "=" * 70)
    print("Demo 3：FilesystemPermission(mode='interrupt') —— 越权写入转人工审批")
    print("=" * 70)

    def build_agent(script: list):
        return create_deep_agent(
            model=make_scripted(script),
            permissions=[
                FilesystemPermission(operations=["write"], paths=["/protected/**"], mode="interrupt"),
            ],
            # interrupt 模式必须配 checkpointer，否则没法暂停/恢复
            checkpointer=MemorySaver(),
        )

    # ---- A. 批准路径 ----
    agent = build_agent([
        ai_tool_call("write_file", {"file_path": "/protected/report.txt", "content": "报告"}, "c1"),
        AIMessage(content="已按审批结果处理。"),
    ])
    config = {"configurable": {"thread_id": "perm-approve"}}
    first = agent.invoke({"messages": [{"role": "user", "content": "写一份报告到受保护目录"}]}, config)
    print("\nA. 批准路径")
    if "__interrupt__" in first:
        payload = first["__interrupt__"][0].value
        print(f"    第一次 invoke 返回中断（工具还没执行）：{str(payload)[:110]}")
    else:
        print(f"    第一次 invoke 没有中断：{str(first.get('messages', [])[-1].content)[:80]}")
    second = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)
    show_tool_messages(second, limit=110)
    print("    ↑ 批准后工具才真正执行（ToolMessage 是成功写入的结果）")

    # ---- B. 拒绝路径 ----
    agent = build_agent([
        ai_tool_call("write_file", {"file_path": "/protected/report.txt", "content": "报告"}, "c1"),
        AIMessage(content="已按审批结果处理。"),
    ])
    config = {"configurable": {"thread_id": "perm-reject"}}
    agent.invoke({"messages": [{"role": "user", "content": "再写一份到受保护目录"}]}, config)
    rejected = agent.invoke(Command(resume={"decisions": [{"type": "reject"}]}), config)
    print("\nB. 拒绝路径")
    show_tool_messages(rejected, limit=110)
    rejected_msgs = [str(m.content) for m in rejected.get("messages", []) if getattr(m, "type", "") == "tool"]
    print(f"    （本次工具消息：{rejected_msgs[-1][:70] if rejected_msgs else '（无）'}）")
    print("    ↑ 拒绝后文件没有被写入 —— 权限 + 人工审核的组合，比单纯的 deny 更适合\n"
          "      「本来该允许、但要有人签字」的场景")


# ================================================================
# Demo 4：内置上下文压缩 —— 超长工具结果自动「卸载」到文件系统
# ================================================================
# deepagents 与普通 agent 最本质的差别之一：**它自带上下文压缩**，不用你加中间件。
# 官方 context-engineering.mdx 讲了三层机制：
#     1. 卸载（offloading）：工具结果过大时，把完整内容写进 backend 文件，
#        历史里只留「文件路径 + 少量预览」；
#     2. 摘要（summarization）：会话接近模型窗口上限时把旧对话压成结构化摘要；
#     3. 按需 compact：可选地给代理一个 compact_conversation 工具自己触发压缩。
# 本 Demo 演示第 1 层（最容易观察）：造一个返回 10 万字符的工具，看历史被替换成什么。
BIG_TEXT_LEN = 100_000


@tool
def fetch_huge_document(topic: str) -> str:
    """返回一份超长文档（用于触发上下文卸载）。"""
    # 造 10 万字符：真实场景是爬下来的网页 / 大日志 / 长检索结果
    return f"# {topic}\n" + ("这是一行重复的正文内容。" * (BIG_TEXT_LEN // 12))


def demo_4_context_offloading() -> None:
    print("\n" + "=" * 70)
    print("Demo 4：内置上下文压缩 —— 超长工具结果被卸载到文件系统")
    print("=" * 70)

    agent = create_deep_agent(
        model=make_scripted([
            ai_tool_call("fetch_huge_document", {"topic": "长文档"}, "c1"),
            AIMessage(content="文档已读，我按预览内容回答。"),
        ]),
        tools=[fetch_huge_document],
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "读一下那份长文档"}]})

    tool_messages = [m for m in result["messages"] if m.type == "tool"]
    if not tool_messages:
        print("  本轮没有工具消息（不符合预期）")
        return
    history_len = len(str(tool_messages[0].content))
    print(f"  工具原始返回：约 {BIG_TEXT_LEN:,} 字符")
    print(f"  历史里留下的工具消息：{history_len:,} 字符")
    print(f"  历史内容预览：{str(tool_messages[0].content)[:150]!r}")

    # 卸载后的完整内容去哪儿了？StateBackend 把它写进了 state 的虚拟文件系统
    files = result.get("files")
    if isinstance(files, dict) and files:
        print(f"  state 里的虚拟文件系统：{sorted(files.keys())}")
        for path, payload in files.items():
            content = payload.get("content", payload) if isinstance(payload, dict) else payload
            print(f"    {path}：{len(str(content)):,} 字符（完整内容在这里，可被 read_file/ls 读回）")
        print(
            "  ↑ 这就是「上下文工程」的核心手法：**把大块内容从上下文挪到文件系统**，\n"
            "    模型需要细节时再用 read_file/grep 取 —— 上下文窗口只花在「路径 + 预览」上。\n"
            "    课案 03~09 讲的七种后端，正是这套机制的存放载体。"
        )
    else:
        print("  （state 里没有 files 字段 —— 该版本可能把卸载内容放在别处，请看上面的历史预览）")


if __name__ == "__main__":
    demo_1_todo_opt_in()
    demo_2_permission_deny()
    demo_3_permission_interrupt()
    demo_4_context_offloading()
    print("\n全部 Demo 执行完毕（0 次真实模型调用，离线可复现）。")


# ================================================================
# 实测结论 / 未收录清单 / 踩坑提示
# ================================================================
# 1. 实测结论（deepagents 0.7.13，本机）：
#    - Demo 1：默认 create_deep_agent 的 state 里没有 todos（write_todos 不是可用工具）；
#      传 middleware=[TodoListMiddleware()] 后 todos 被正确写入（3 条结构化待办）。
#    - Demo 2：写 /secrets/** 被拒（工具返回拒绝说明，不抛异常），写 /work/** 成功。
#    - Demo 3：写 /protected/** 先返回 __interrupt__（工具没执行）；approve 后工具才执行；
#      另开 thread 用 reject 则文件不写入。
#    - Demo 4：工具返回 100,002 字符，历史里只剩 1,608 字符，内容为
#      "Tool result too large, the result of this tool call c1 was saved in the filesystem
#       at this path: /large_tool_results/c1" + 预览；完整内容在 state 的虚拟文件系统
#      result["files"]["/large_tool_results/c1"]（100,002 字符，可被 ls / read_file 读回）。
# 2. 未收录（官方还有、本文件没做的）：
#    - Interpreters + PTC（interpreters.mdx）：要给环境装 `pip install "deepagents[quickjs]"`；
#    - RubricMiddleware（rubric.mdx，0.6.5+ Beta）：评审子代理要真实模型，否则评不出东西；
#    - 动态子代理（dynamic-subagents.mdx）：建立在解释器之上；
#    - AsyncSubAgent（async-subagents.mdx）：要 Agent Protocol 服务器（本地 langgraph dev 或远端）；
#    - MCPAdapter（tools.mdx）：要起一个真实 MCP server（本仓库 05_mcp 章有现成的可复用）；
#    - 自定义 Backend 协议（backends.mdx）：继承 BackendProtocol 实现七个方法，属进阶练习；
#    - deepagents 版 RAG（rag.mdx）：要 embeddings 接口。
#    完整对照表见 Agent/官方文档缺口对照.md。
# 3. 踩坑提示：
#    A. write_todos 自 v0.7 起 opt-in：升级后「代理不会做计划了」多半是漏传 TodoListMiddleware。
#       而且它来自 langchain 包（from langchain.agents.middleware import TodoListMiddleware），
#       不是 deepagents 自己导出的。
#    B. 权限规则**先匹配先生效**，一条不匹配就默认放行 —— 写规则时把最特殊的放最前面。
#    C. mode="interrupt" 必须配 checkpointer，恢复格式与工具中断完全一致
#       （Command(resume={"decisions": [...]})）；忘了 checkpointer 会在中断时报错。
#    D. paths 要锚定（"/secrets/**"）；"/**/secrets" 这类未锚定模式会让批量工具
#       （ls/glob/grep）保守地过度触发。
#    E. 本文件用 ScriptedModel 驱动，所以 todos/files 这些 state 字段都能直接检查；
#       换成真实模型后字段名不变，只是内容不再确定 —— 断言时请断言**结构**而不是文本。

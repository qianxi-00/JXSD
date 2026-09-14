# -*- coding: utf-8 -*-
"""
A2A 协议 · 客户端：DeepAgents 总调度员（A2AClient + langgraph dev 暴露为 A2A）
================================================================
课案原文：DeepAgents 通过 `A2AClient` 调用 CrewAI A2A 服务，
自身通过 `langgraph dev` 暴露 A2A。

A2A 解决什么问题、谁和谁通信：
    A2A（Agent-to-Agent）管的是 **Agent ↔ Agent** 这条边 ——
    让「用不同框架、由不同团队/厂商写的智能体」能互相下单、互相交货。
    本文件里通信的双方就是：
        客户端（本文件）→ DeepAgents 总调度员（另一个 Agent）
        DeepAgents 总调度员 → CrewAI 数据分析师（又一个 Agent）
    注意区分三张网（课案表格）：
        A2A = Agent ↔ Agent（协作）    ← 本节
        ACP = Agent ↔ 编辑器（集成，见 01/02）
        MCP = Agent ↔ 工具（调外设）

    传输层上 A2A = **JSON-RPC 2.0 over HTTP**，所以报文格式和 02 节学的那套完全一样
    （有 id 是请求、无 id 是通知、错误码也是 -32700/-32600/-32601/-32602…），
    只是从「stdio 一行一个 JSON」换成了「HTTP POST 一个 JSON」。

本节要讲什么
    1. 调度员怎么「把调另一个 Agent」变成一件 LLM 能理解的事：
       把 A2A 调用封成一个 `@tool`（`call_crewai_analyst`）——
       对模型来说它和「查天气」没有区别，这就是 A2A 的价值所在；
    2. `langgraph.json` 是干什么的：它把 `graph = agent` 这个变量
       注册成 HTTP 服务上的一张「图」，`assistant_id` 就是它的键名；
    3. `langgraph dev --port 2024` 怎么把图暴露出去（为什么端口是 2024）；
    4. 工具里为什么必须 try/except 而不是往外抛异常。

整体架构（两个 A2A 服务 + 一个调度员）：
        ┌──────────────────────────────┐        ┌───────────────────────────┐
        │  DeepAgents 总调度员          │  A2A   │  CrewAI 数据分析师         │
        │  deepagent_a2a.py            │ ─────► │  crewai_a2a_server.py     │
        │  langgraph dev --port 2024   │ :2025  │  serve_crewai_agent(:2025)│
        └──────────────────────────────┘        └───────────────────────────┘
                   ▲
                   │ langgraph_sdk（HTTP，见 05_a2a互相通信_jxsd.py）
              a2a_client.py

关键设计：**调度员把「调另一个 Agent」封装成一个 @tool**。
    对 LLM 来说，`call_crewai_analyst` 和「查天气」「读文件」没有任何区别，
    它只是在需要的时候选这个工具；至于工具背后是本地函数还是跨进程、跨框架的
    A2A 调用，模型完全不用知道 —— 这就是 A2A 的价值：
    把「另一个框架写的 Agent」变成一个可被调用的普通工具/服务。

本文件对应课案「A2A → DeepAgents端」的三块内容：
    ① deepagent_a2a.py（58 行）：调度员 Agent + call_crewai_analyst 工具
    ② `# CrewAI 分析师的 A2A 客户端（端口 2025）` 那 8 行 json（langgraph.json）
    ③ `pip install "langgraph-cli[inmem]"` 那 7 行（把图暴露成 HTTP 服务）

⚠️ 本机现实：`a2a_auto_wrapper` 没装 → A2AClient 不可用。
   降级做法：把课案源码原文打印出来，然后
     · 用 `langgraph_sdk`（本机**已装**）真去连一次 2024 端口，
       没服务就打印中文启动指引，不抛 traceback；
     · 用一个「假 A2AClient」验证 call_crewai_analyst 的降级分支
       （课案里那段 try/except 是有意为之：A2A 服务没起时要给出可读提示）。

课案出处：Agent 课案 → 协议 → A2A → DeepAgents端

运行方式：
    uv run Agent/07_protocols/04_a2a客户端_jxsd.py

前置条件：
    - **能直接跑**：缺 a2a_auto_wrapper 时走降级演示（打印课案源码 + 用假 client
      验证 try/except 分支 + 用 langgraph_sdk 探一次 2024 端口），不抛 traceback；
    - 想走完整真流程才需要：
      ① `uv add a2a_auto_wrapper langgraph-api langgraph-sdk deepagents crewai`
      ② `uv add "langgraph-cli[inmem]"`（提供 `langgraph dev` 命令）
      ③ 先把 CrewAI 服务端跑起来：`python crewai_a2a_server.py`（:2025）
      ④ 再起调度员：`langgraph dev --port 2024 --no-browser`
      ⑤ 两端都起来后，本文件才会走到「连接成功」分支。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，防中文/emoji 报错

import asyncio
import json
import platform
from pathlib import Path

from config import settings

# ================================================================
# 0. 依赖探测
# ================================================================
# | 包               | 作用                                          | 本机 |
# | a2a_auto_wrapper | A2AClient（调度员调 CrewAI 的那只手）          | 缺   |
# | deepagents       | create_deep_agent（总调度员本体）              | 已装 |
# | langgraph_sdk    | 通过 HTTP 调 2024 端口的调度员（05 节要用）     | 已装 |
# | langgraph_cli    | `langgraph dev` 命令本身                       | 缺   |
_HAS_A2A_WRAPPER = False
_HAS_DEEPAGENTS = False
_HAS_LANGGRAPH_SDK = False
_HAS_LANGGRAPH_CLI = False
_IMPORT_ERRORS: list[str] = []

try:
    from a2a_auto_wrapper import A2AClient  # type: ignore[import-not-found]

    _HAS_A2A_WRAPPER = True
except ImportError as exc:
    _IMPORT_ERRORS.append(f"a2a_auto_wrapper：{exc}")

try:
    from deepagents import create_deep_agent

    _HAS_DEEPAGENTS = True
except ImportError as exc:
    _IMPORT_ERRORS.append(f"deepagents：{exc}")

try:
    from langgraph_sdk import get_client

    _HAS_LANGGRAPH_SDK = True
except ImportError as exc:
    _IMPORT_ERRORS.append(f"langgraph_sdk：{exc}")

try:
    from langgraph_cli import cli  # type: ignore[import-not-found]  # noqa: F401

    _HAS_LANGGRAPH_CLI = True
except ImportError as exc:
    _IMPORT_ERRORS.append(f"langgraph-cli：{exc}")

# 两个服务的地址（课案固定端口：调度员 2024，CrewAI 分析师 2025）
DEEPAGENT_PORT = 2024
CREWAI_ANALYST_URL = "http://localhost:2025/"

# ================================================================
# 1. 课案原文（对照用）：deepagent_a2a.py 的 58 行
# ================================================================
# 课案出处：Agent 课案 → 协议 → A2A → DeepAgents端
#
# ```python
# """DeepAgents A2A 服务：总调度员，可自主调用 CrewAI 分析师
#
# 通过 langgraph dev --port 2024 暴露为 A2A 服务，加载入口见 langgraph.json。
# """
# from a2a_auto_wrapper import A2AClient
# from langchain_openai import ChatOpenAI
# from langchain_core.tools import tool
# from deepagents import create_deep_agent
#
# from config import setting
#
# # CrewAI 分析师的 A2A 客户端（端口 2025）
# crewai_client = A2AClient(base_url="http://localhost:2025/")
#
# @tool
# async def call_crewai_analyst(query: str) -> str:
#     """需要数据分析或专业报告时调用 CrewAI 分析师。
#
#     Args:
#         query: 交给分析师的完整问题描述，包含待分析的数据
#
#     Returns:
#         分析师返回的分析报告文本
#     """
#     try:
#         return await crewai_client.send_message(query)
#     except Exception as e:
#         return (
#             f"调用 CrewAI 分析师失败：{e}。"
#             "请确认 crewai_a2a_server.py 已在 2025 端口启动后重试。"
#         )
#
# agent = create_deep_agent(
#     model=ChatOpenAI(
#         model=setting.MODEL_NAME,
#         api_key=setting.API_KEY,
#         base_url=setting.BASE_URL,
#         max_retries=3,
#         timeout=60,
#     ),
#     system_prompt="你是总调度员。遇到数据分析任务时主动调用 call_crewai_analyst。",
#     tools=[call_crewai_analyst],
# )
#
# graph = agent  # langgraph dev 加载入口（langgraph.json 中 deep_researcher 指向此处）
# ```
#
# 三处值得抄下来的细节：
# | 写法                              | 为什么                                                |
# | `@tool` + **async def**           | LangGraph/LangChain 支持异步工具；A2A 调用是网络 IO，   |
# |                                   | 写成 async 才不会把事件循环堵住                        |
# | docstring 里的 Args/Returns       | **这就是给 LLM 看的工具说明书**，写得越清楚，模型越会   |
# |                                   | 在正确的时机调用它                                     |
# | try/except 里返回可读提示          | 工具**不要往外抛异常**：抛了会变成 Agent 运行失败；      |
# |                                   | 返回一句人能看懂的失败原因，模型还能自己决定要不要重试   |
COURSE_DEEPAGENT_PY = '''"""DeepAgents A2A 服务：总调度员，可自主调用 CrewAI 分析师

通过 langgraph dev --port 2024 暴露为 A2A 服务，加载入口见 langgraph.json。
"""
from a2a_auto_wrapper import A2AClient
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from deepagents import create_deep_agent

from config import settings                    # 课案原文是 from config import setting

# CrewAI 分析师的 A2A 客户端（端口 2025）
crewai_client = A2AClient(base_url="http://localhost:2025/")


@tool
async def call_crewai_analyst(query: str) -> str:
    """需要数据分析或专业报告时调用 CrewAI 分析师。

    Args:
        query: 交给分析师的完整问题描述，包含待分析的数据

    Returns:
        分析师返回的分析报告文本
    """
    try:
        return await crewai_client.send_message(query)
    except Exception as e:
        return (
            f"调用 CrewAI 分析师失败：{e}。"
            "请确认 crewai_a2a_server.py 已在 2025 端口启动后重试。"
        )


agent = create_deep_agent(
    model=ChatOpenAI(
        model=settings.model_name,             # 课案原文 setting.MODEL_NAME
        api_key=settings.api_key,              # 课案原文 setting.API_KEY
        base_url=settings.base_url,            # 课案原文 setting.BASE_URL
        max_retries=3,
        timeout=60,
    ),
    system_prompt="你是总调度员。遇到数据分析任务时主动调用 call_crewai_analyst。",
    tools=[call_crewai_analyst],
)

graph = agent   # langgraph dev 加载入口（langgraph.json 中 deep_researcher 指向此处）
'''

# ================================================================
# 2. 课案原文（对照用）：langgraph.json（8 行配置）
# ================================================================
# 课案原话：`langgraph dev` 加载入口见 langgraph.json。
#
# 字段逐条解释：
# | 字段           | 说明                                                            |
# | python_version | 让 langgraph-cli 起的运行时用哪个 Python（本项目是 3.13，         |
# |                | 课案写 3.11 —— 只影响 CLI 建的隔离环境，不影响你 .venv 里的代码） |
# | dependencies   | 依赖清单，`["./"]` = 就用当前目录这个包，不做额外安装             |
# | graphs         | **最重要**：名字 → "文件路径:变量名"。这里 deep_researcher 指向    |
# |                | ./deepagent_a2a.py 的 graph 变量（就是文件末尾那个 graph = agent）|
# | env            | 环境变量文件，.env 里的 api_key/base_url 由此注入               |
COURSE_LANGGRAPH_JSON = '''{
  "python_version": "3.11",
  "dependencies": ["./"],
  "graphs": {
    "deep_researcher": "./deepagent_a2a.py:graph"
  },
  "env": ".env"
}'''


def build_local_langgraph_json() -> str:
    """把课案那份 langgraph.json 改写成「本机可直接用」的版本。

    差别只有两处（都为了让学员复制即用）：
        · python_version 取当前解释器真实版本，而不是课案的 3.11；
        · graphs 的路径用绝对路径指向本文件所在目录下的 deepagent_a2a.py。
    """
    this_dir = Path(__file__).resolve().parent
    payload = {
        # platform.python_version() → "3.13.5"；langgraph.json 只认 "x.y"
        "python_version": ".".join(platform.python_version().split(".")[:2]),
        "dependencies": ["./"],
        "graphs": {
            # 真实部署时这个文件就叫 deepagent_a2a.py（就是上面那段课案代码）
            "deep_researcher": str(this_dir / "deepagent_a2a.py") + ":graph",
        },
        "env": ".env",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ================================================================
# 3. 课案原文（对照用）：把图暴露成 HTTP 服务（langgraph dev）
# ================================================================
# 课案原文：
# ```bash
# pip install "langgraph-cli[inmem]"
#
# # Linux / macOS
# LANGCHAIN_TRACING_V2=false langgraph dev --port 2024 --no-browser
# # Windows PowerShell
# $env:PYTHONUTF8="1"; $env:LANGCHAIN_TRACING_V2="false"; .venv\Scripts\langgraph dev --port 2024 --no-browser
# ```
#
# 三个参数/变量的含义：
# | 内容                        | 说明                                                  |
# | langgraph-cli[inmem]        | CLI 本体 + inmem（内存版运行时），本地开发不需要起数据库 |
# | --port 2024                 | 监听端口，和 CrewAI 的 2025 错开，这是本文档的约定      |
# | --no-browser                | 别自动开浏览器（服务器/无桌面环境必须加，否则卡住）      |
# | LANGCHAIN_TRACING_V2=false  | 关掉 LangSmith 上报，避免没配 key 时刷一堆告警          |
# | $env:PYTHONUTF8="1"         | Windows 下让日志里的中文不乱码                         |
#
# 跑起来之后：
#     http://localhost:2024/ok            → 健康检查
#     http://localhost:2024/docs          → 自动生成的 API 文档
#     assistant_id = "deep_researcher"    → 就是 langgraph.json 里 graphs 的键名
def print_dev_server_hint() -> None:
    print("【启动 DeepAgents 侧服务（课案原文，端口 2024）】")
    print('    pip install "langgraph-cli[inmem]"    # 本项目请用：uv add "langgraph-cli[inmem]"')
    print("    # Linux / macOS")
    print("    LANGCHAIN_TRACING_V2=false langgraph dev --port 2024 --no-browser")
    print("    # Windows PowerShell")
    print('    $env:PYTHONUTF8="1"; $env:LANGCHAIN_TRACING_V2="false"; '
          '.venv\\Scripts\\langgraph dev --port 2024 --no-browser')
    print()


# ================================================================
# 4. 调度员本体：按课案原文组装（A2AClient 可用时）
# ================================================================
def build_crewai_client():
    """课案那行 `crewai_client = A2AClient(base_url="http://localhost:2025/")`。"""
    return A2AClient(base_url=CREWAI_ANALYST_URL)


async def call_crewai_analyst_impl(client, query: str) -> str:
    """call_crewai_analyst 的真实函数体 —— 课案那段 try/except 原样搬过来。

    单独抽成模块级函数，是为了让「降级演示」能直接 await 它本体，
    验证「2025 端口没起服务」时返回的是**一句人话**而不是异常。
    """
    try:
        # send_message 背后就是一次 A2A 的 message/send（HTTP POST JSON-RPC）
        return await client.send_message(query)
    except Exception as e:
        # 工具里吞掉异常、返回可读文本：模型看到这句话还能自己决定重试或换个说法
        return (
            f"调用 CrewAI 分析师失败：{e}。"
            "请确认 crewai_a2a_server.py 已在 2025 端口启动后重试。"
        )


def build_dispatcher(crewai_client):
    """组装总调度员：一个 @tool + create_deep_agent。

    注意工具定义写在函数内部，是为了能用外部传入的 crewai_client
    （课案是模块级写法，两者等价；本文件要同时演示「真 client」和「假 client」）。
    """
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI

    @tool
    async def call_crewai_analyst(query: str) -> str:
        """需要数据分析或专业报告时调用 CrewAI 分析师。

        Args:
            query: 交给分析师的完整问题描述，包含待分析的数据

        Returns:
            分析师返回的分析报告文本
        """
        return await call_crewai_analyst_impl(crewai_client, query)

    # 调度员本体：模型用课案原文的 ChatOpenAI 三件套；
    # max_retries / timeout 也是课案原文，都跟「A2A 是网络调用」这件事有关。
    agent = create_deep_agent(
        model=ChatOpenAI(
            model=settings.model_name,
            api_key=settings.api_key,
            base_url=settings.base_url,
            max_retries=3,   # 网络抖动时自动重试 3 次，A2A 调用尤其需要
            timeout=60,      # 对方 LLM 可能想很久，60 秒够用
        ),
        # 提示词里点名「遇到数据分析任务时主动调用 call_crewai_analyst」，
        # 否则模型可能自己硬答、不去用那个工具
        system_prompt="你是总调度员。遇到数据分析任务时主动调用 call_crewai_analyst。",
        tools=[call_crewai_analyst],
    )
    return agent


class FakeA2AClient:
    """降级用的假 A2AClient：验证 call_crewai_analyst 的 try/except 分支。

    它故意抛 ConnectionError —— 这正是「2025 端口没起服务」时 A2AClient
    会抛的东西。学员能看到课案那段 try/except 到底在防什么。
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    async def send_message(self, query: str) -> str:
        raise ConnectionError(f"无法连接 {self.base_url}（Connection refused）")


async def probe_crewai_tool(client, label: str) -> None:
    """直接 await 工具函数体，看它拿到什么。

    绕开 LLM 直接调工具，是为了把「工具内部逻辑」和「模型决策」拆开看：
    A2A 调用失败时，工具返回的是一句话，而不是抛异常。
    调的就是 build_dispatcher 里那个 @tool 包着的同一个实现，不存在"抄一遍"。
    """
    print(f"[{label}] 直接调用工具函数（不经过模型）")
    result = await call_crewai_analyst_impl(
        client, "分析销售数据：1月100万 2月120万 3月90万，做趋势预测"
    )
    print("    返回：" + result)
    print()


# ================================================================
# 5. 降级演示：langgraph_sdk 真连一次 2024 端口
# ================================================================
async def probe_langgraph_server() -> None:
    """用 langgraph_sdk 探测 2024 端口。

    langgraph_sdk 是**纯 HTTP 客户端**（本机已装），所以哪怕没装
    langgraph-cli、没起服务，也能跑这一段 —— 我们要的就是「连不上时
    给出人话指引」而不是甩一坨 traceback。
    """
    print("【探测 DeepAgents 侧服务】http://localhost:2024")
    if not _HAS_LANGGRAPH_SDK:
        print("    langgraph_sdk 未安装，跳过。安装：uv add langgraph-sdk")
        print()
        return

    client = get_client(url=f"http://localhost:{DEEPAGENT_PORT}")
    try:
        # threads.create() 会真的发一次 HTTP 请求
        thread = await client.threads.create()
        print(f"    连接成功！已创建会话线程 thread_id = {thread['thread_id']}")
        print("    （05_a2a互相通信_jxsd.py 会用这个 thread_id 提交问题）")
    except Exception as exc:
        print(f"    连不上（{type(exc).__name__}）：{str(exc)[:120]}")
        print("    → 这是**预期结果**：DeepAgents 侧服务还没启动。")
        print("      按下面三步启动整套 A2A 链路后，本文件会走到「连接成功」分支：")
        print("        ① uv add a2a_auto_wrapper langgraph-api langgraph-sdk deepagents crewai")
        print("        ② .venv\\Scripts\\python crewai_a2a_server.py          "
              "# CrewAI 分析师，:2025")
        print("        ③ langgraph dev --port 2024 --no-browser              "
              "# 总调度员，:2024")
    print()


# ================================================================
# 6. 入口
# ================================================================
def main() -> None:
    print("=" * 66)
    print("A2A 客户端：DeepAgents 总调度员（调 CrewAI 分析师 + 自身暴露为 A2A）")
    print("=" * 66)
    print()

    # ---------- 第 1 步：课案原文 ----------
    print("【课案原文】deepagent_a2a.py（58 行）：")
    for line in COURSE_DEEPAGENT_PY.splitlines():
        print("    " + line)
    print()

    print("【课案原文】langgraph.json（8 行，CrewAI/调度员的加载入口配置）：")
    for line in COURSE_LANGGRAPH_JSON.splitlines():
        print("    " + line)
    print()
    print("【本机可用版】langgraph.json（版本号与路径都换成动态生成）：")
    for line in build_local_langgraph_json().splitlines():
        print("    " + line)
    print()

    print_dev_server_hint()

    # ---------- 第 2 步：依赖情况 ----------
    print("【前置条件检查】")
    for err in _IMPORT_ERRORS:
        print(f"    - 缺失：{err}")
    print("      安装：uv add a2a_auto_wrapper langgraph-api langgraph-sdk deepagents crewai")
    print('      另需：uv add "langgraph-cli[inmem]"    # 提供 langgraph dev 命令')
    print()

    # ---------- 第 3 步：A2AClient / create_deep_agent ----------
    # 依赖齐全就组装「真调度员」；缺 a2a_auto_wrapper 就换成两件降级的事。
    if _HAS_A2A_WRAPPER and _HAS_DEEPAGENTS:
        print("【依赖检查】a2a_auto_wrapper + deepagents 均已就绪，组装真实调度员。")
        client = build_crewai_client()
        agent = build_dispatcher(client)
        print(f"    调度员已就绪：{type(agent).__name__}，工具 = [call_crewai_analyst]")
        print(f"    它会通过 A2A 调用 {CREWAI_ANALYST_URL}（需先起 crewai_a2a_server.py）")
        print()
    else:
        # 降级分支刻意不「装样子」去调远程服务，只做两件有意义的事：
        #   ① 用假 client 直接 await 工具函数体 → 验证课案那段 try/except 真的能兜住失败；
        #   ② 交给第 4 步的 langgraph_sdk 去真连 :2024，连不上就给中文指引。
        print("【降级演示】a2a_auto_wrapper 未安装 → 无法 new 出真的 A2AClient。")
        print("            改成两件事：① 用假 client 验证工具里的 try/except 分支；")
        print("                      ② 用 langgraph_sdk（已装）真连一次 2024 端口看提示。")
        print()
        asyncio.run(probe_crewai_tool(FakeA2AClient(CREWAI_ANALYST_URL),
                                      "假 A2AClient · 未起 2025 服务"))
        # deepagents 装了只能说明「这段代码语法可用」，不代表能跑通 ——
        # 真正跑起来还需要 :2025 有服务，所以这里只报告事实、不发起调用。
        if _HAS_DEEPAGENTS:
            print("【依赖检查】deepagents 已装 → 课案那段 create_deep_agent(...) 语法本身可用；")
            print("            只是 model 换成 settings 三件套、@tool 里的 client 换成假 client。")
            print("            真正跑起来需要 2025 端口有服务，本机不具备，故不发起调用。")
            print()

    # ---------- 第 4 步：探测 2024 端口 ----------
    asyncio.run(probe_langgraph_server())

    print("=" * 66)
    print("小结：调度员 = 一个普通 DeepAgent + 一个「调远程 Agent」的 @tool。")
    print("      · A2AClient 把 A2A 的 HTTP JSON-RPC 藏进了 send_message()")
    print("      · create_deep_agent 出的图交给 langgraph.json 注册，")
    print("        langgraph dev --port 2024 一跑，它就同时成了 A2A 服务端")
    print("      · 三方串联（客户端 → 调度员 → 分析师）见 05_a2a互相通信_jxsd.py")
    print("=" * 66)


if __name__ == "__main__":
    main()

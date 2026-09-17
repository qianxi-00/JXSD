# -*- coding: utf-8 -*-
"""
ACP 协议 · 实操：把 Deep Agent 暴露成 ACP 服务端（stdio 模式）
================================================================
课案原话：ACP 协议定义了 Agent 与代码编辑器/IDE 之间的标准通信方式，
使得自定义的 Deep Agent 可以在任何支持 ACP 的客户端中使用。

一句话理解 ACP（Agent Client Protocol）：
    它之于「编辑器 ↔ 智能体」的关系，就等于 LSP（Language Server Protocol）
    之于「编辑器 ↔ 语言服务器」的关系 —— 编辑器只出界面，智能体进程干活，
    两边用同一套协议说话，于是同一个智能体可以插到 Zed / PyCharm / VS Code /
    Neovim 里复用，不用为每个编辑器写一次适配。

本节要讲什么
    1. ACP **解决什么问题 / 谁和谁通信**：编辑器（客户端）↔ 智能体（服务端）。
       编辑器负责界面、快捷键、文件上下文，智能体负责真正干活；
       ACP 是它们之间的「插座标准」。注意它**不是**给 Agent 之间通信用
       （那是 A2A，见 03~05），也**不是**给 Agent 调工具用（那是 MCP）；
    2. 怎么把一个自己写的 Deep Agent 挂上这个插座：`AgentServerACP(agent)`
       + `await run_agent(server)`，默认走 **stdio**（stdin 读请求、stdout 写响应），
       一行都不用自己写 JSON-RPC；
    3. 编辑器侧要做什么配置：PyCharm/JetBrains 的 `~/.jetbrains/acp.json`
       怎么把「用哪个解释器、跑哪个脚本」注册进去；
    4. 报文长什么样（本节末尾的降级演示会手工构造一段），
       为下一节 02_acp原理_jxsd.py 的真实收发做铺垫。

本节的三个知识点（和课案 h4 小标题一一对应）：
    1. 安装：`pip install deepagents-acp`（本项目用 uv，见下方 DEPS 提示）
    2. 快速上手：课案 `# test.py` —— 用 create_deep_agent 造一个 Agent，
       再用 AgentServerACP + run_agent(server) 把它挂到 stdio 上
    3. PyCharm 配置：写 `C:\\Users\\<用户名>\\.jetbrains\\acp.json` 注册自定义 Agent

和相邻小节的关系：
    下一节 `02_acp原理_jxsd.py` 会把这里「被 AgentServerACP 封装掉」的 JSON-RPC
    报文一层层拆开，用纯标准库真的收发一遍。先看本节「怎么用」，
    再看下一节「底下到底在传什么」。

运行前置条件：
    - `deepagents`（本项目已装）
    - `acp` + `deepagents-acp`（**本机 venv 没装**）→ 走「降级演示」分支：
      打印安装命令 + 一条模拟的 ACP 会话报文，不会抛 traceback。
    - `settings.api_key` / `base_url` / `model_name`（根目录 .env，实测可用）

运行方式：
    uv run Agent/07_protocols/01_acp智能体_jxsd.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，防中文/emoji 报错

import asyncio
import json

from config import settings

# ================================================================
# 0. 依赖探测：课案的三件套里，后两件本机没装
# ================================================================
# 铁律 1.2：模块层面永远不许因为缺包而 import 失败，
# 所以所有第三方 import 都用 try/except 兜住，
# 把「缺什么」记在布尔开关里，等到 __main__ 再决定走真流程还是降级演示。
#
# | 包              | 作用                                    | 本机 |
# | deepagents      | 造 Deep Agent（create_deep_agent）       | 已装 |
# | acp             | 官方 Python SDK，提供 run_agent()        | 缺   |
# | deepagents-acp  | 把 DeepAgents 接进 ACP 的适配器          | 缺   |
_HAS_DEEPAGENTS = False
_HAS_ACP_SDK = False
_HAS_DEEPAGENTS_ACP = False
_IMPORT_ERRORS: list[str] = []

try:
    from deepagents import create_deep_agent

    _HAS_DEEPAGENTS = True
except ImportError as exc:  # pragma: no cover - 本机已装
    _IMPORT_ERRORS.append(f"deepagents：{exc}")

try:
    from acp import run_agent  # type: ignore[import-not-found]

    _HAS_ACP_SDK = True
except ImportError as exc:
    _IMPORT_ERRORS.append(f"acp：{exc}")

try:
    from deepagents_acp.server import AgentServerACP  # type: ignore[import-not-found]

    _HAS_DEEPAGENTS_ACP = True
except ImportError as exc:
    _IMPORT_ERRORS.append(f"deepagents-acp：{exc}")

# 课案 test.py 里这一行还额外用到了 LocalShellBackend：
#     backend=LocalShellBackend(root_dir=".", virtual_mode=True, inherit_env=True)
# 注释说得很清楚：inherit_env=True 让「执行 shell 命令」时继承当前 Python 进程
# 的所有环境变量（否则子进程 PATH/代理/密钥全是空的，装依赖、跑构建都会失败）。
# 它在 deepagents 0.7.x 里位于 deepagents.backends 子包，单独 try 一层。
_HAS_LOCAL_SHELL_BACKEND = False
try:
    from deepagents.backends import LocalShellBackend

    _HAS_LOCAL_SHELL_BACKEND = True
except ImportError as exc:
    _IMPORT_ERRORS.append(f"deepagents.backends.LocalShellBackend：{exc}")

# 「真流程」的开关：三件套齐了才走 run_agent(server)，否则降级
_ACP_READY = _HAS_DEEPAGENTS and _HAS_ACP_SDK and _HAS_DEEPAGENTS_ACP


# ================================================================
# 1. 课案原文（对照用）：这段就是课案 `# test.py` 的 27 行
# ================================================================
# 课案出处：Agent 课案 → 协议 → ACP → 实操 → 快速上手
#
# ```python
# # test.py
# import asyncio
# from acp import run_agent
# from deepagents import create_deep_agent
# from langgraph.checkpoint.memory import MemorySaver
# from deepagents_acp.server import AgentServerACP
# from langchain_openai import ChatOpenAI
# from conf import settings
#
#
# async def main() -> None:
#     model = ChatOpenAI(model=settings.model_name, api_key=settings.api_key, base_url=settings.base_url)
#     agent = create_deep_agent(
#         model=model,
#         system_prompt="你是一个编程助手",
#         checkpointer=MemorySaver(),
#         # 可自定义：tools、subagents、middleware、skills 等
#         backend=LocalShellBackend(root_dir=".", virtual_mode=True, inherit_env=True),  # 让执行 shell 命令时继承当前 Python 进程的所有环境变量。
#     )
#
#     server = AgentServerACP(agent)
#     await run_agent(server)  # 默认 stdio：stdin 读取 → stdout 写回
#
#
# if __name__ == "__main__":
#     asyncio.run(main())
# ```
#
# 课案到本项目的两处改写（规范第 3 节）：
#     ① `from conf import settings` → `from config import settings`
#     ② `ChatOpenAI(model=..., api_key=..., base_url=...)` 这种显式三参数的写法
#        在本节是课案原文明确演示的形态，允许保留；但参数必须来自 settings，
#        一个都不许硬编码进代码。
COURSE_TEST_PY = '''# test.py
import asyncio
from acp import run_agent
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import MemorySaver
from deepagents_acp.server import AgentServerACP
from langchain_openai import ChatOpenAI
from config import settings          # 课案原文是 from conf import settings


async def main() -> None:
    model = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )
    agent = create_deep_agent(
        model=model,
        system_prompt="你是一个编程助手",
        checkpointer=MemorySaver(),
        # 可自定义：tools、subagents、middleware、skills 等
        backend=LocalShellBackend(root_dir=".", virtual_mode=True, inherit_env=True),
    )

    server = AgentServerACP(agent)
    await run_agent(server)      # 默认 stdio：stdin 读取 → stdout 写回


if __name__ == "__main__":
    asyncio.run(main())
'''


# ================================================================
# 2. 课案原文（对照用）：PyCharm 的 acp.json 8 行配置
# ================================================================
# 课案出处：Agent 课案 → 协议 → ACP → 实操 → pycharm配置
#
# 步骤（课案原文）：
#     1. 创建/编辑 C:\\Users\\<用户名>\\.jetbrains\\acp.json（内容见下）
#     2. 重启 PyCharm，打开 AI Chat，在 Agent 下拉菜单里会看到新增的
#        "My Deep Agent"（带自定义 agent 图标），选中即可使用
#     3. PyCharm 会通过 **stdio** 启动你的 Python 脚本
#
# 课案字段表：
# | 字段    | 说明                                                   |
# | command | Python 解释器的绝对路径，Windows 下必须带 .exe 后缀     |
# | args    | 启动参数，这里传入你的 ACP 服务端脚本路径               |
#
# ⚠️ 照抄课案会踩的坑（务必按注释改成本机真实路径）：
#     - 课案里写的是 `C:\\Users\\13261\\anaconda3\\python.exe`，那是课案作者的机器；
#       本机应填 `F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe`
#       ——**必须用装了 deepagents-acp 的那个解释器**，用错解释器 PyCharm 会
#       以「ModuleNotFoundError: No module named 'acp'」静默退出。
#     - JSON 里反斜杠是转义字符，Windows 路径必须写成双反斜杠 `\\\\`
#       （下面 COURSE_ACP_JSON 用 Python 原始字符串 r'''...''' 保存，
#        所以文件里看到的仍是课案原样的双反斜杠，别被这里骗了）。
COURSE_ACP_JSON = r'''{
    "agent_servers": {
        "My Deep Agent": {
            "command": "C:\\Users\\13261\\anaconda3\\python.exe",
            "args": ["C:\\Users\\13261\\Documents\\project\\ai_llm_jiaoan\\test.py"]
        }
    }
}'''


def build_local_acp_json() -> str:
    """把课案那份 acp.json 改写成「本机可直接用」的版本。

    课案用固定路径演示，这里用 sys.executable / __file__ 动态生成，
    免得学员复制过去还要手改路径。注意 Windows 路径在 JSON 里是 `\\\\`。
    """
    payload = {
        "agent_servers": {
            "My Deep Agent": {
                # sys.executable = 当前正在跑本文件的解释器，
                # 也就是我们要求的 F:\ProGram\Python_Base\.venv\Scripts\python.exe
                "command": sys.executable,
                # __file__ = 本文件绝对路径（ACP 服务端脚本）
                "args": [str(__file__)],
            }
        }
    }
    # ensure_ascii=False：让中文（"My Deep Agent" 的键名可自定义）原样输出
    return json.dumps(payload, ensure_ascii=False, indent=4)


# ================================================================
# 3. 客户端的支持情况（课案表格搬进注释）
# ================================================================
# | 客户端          | 说明                                |
# | Zed             | 原生支持 ACP 外部 Agent             |
# | JetBrains IDEs  | 通过 AI Assistant 支持 ACP          |
# | VS Code         | 通过 vscode-acp 插件                |
# | Neovim          | 通过 ACP 兼容插件                   |
#
# 也就是说：同一个 test.py，注册进哪个编辑器就能在哪个编辑器里用，
# 这正是「协议」的价值 —— 适配一次，处处可用。


# ================================================================
# 4. 真流程：造 Agent → 挂到 stdio（等价于课案 test.py 的 main）
# ================================================================
def build_agent():
    """按课案 test.py 组装 Deep Agent。

    与课案的两点差异（都是为了「本机没装 acp 时也能看出结构」）：
        - model 直接用课案的 ChatOpenAI + settings 三件套；
        - backend 只有在 LocalShellBackend 导入成功时才传，
          否则 create_deep_agent 用默认 backend（否则这里会 NameError）。
    """
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.memory import MemorySaver

    model = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )

    kwargs = {
        "model": model,
        "system_prompt": "你是一个编程助手",
        # MemorySaver：把会话检查点放在内存里，进程退出即丢。
        # 生产环境换成 PostgresSaver（见 Agent/01_langgraph/03_短期记忆_生产.py）。
        "checkpointer": MemorySaver(),
        # 可自定义：tools、subagents、middleware、skills 等
    }
    if _HAS_LOCAL_SHELL_BACKEND:
        # virtual_mode=True：文件操作被限制在 root_dir 下，路径穿越会被拦；
        # inherit_env=True：执行 shell 命令时继承当前 Python 进程的所有环境变量。
        kwargs["backend"] = LocalShellBackend(
            root_dir=".", virtual_mode=True, inherit_env=True
        )
    return create_deep_agent(**kwargs)


async def serve_acp() -> None:
    """课案 test.py 的真身：把 Agent 交给 AgentServerACP，挂到 stdio 上。"""
    server = AgentServerACP(build_agent())
    # run_agent 默认走 stdio：从 stdin 读 JSON-RPC 请求，往 stdout 写响应/通知。
    # 注意：一旦跑到这里，stdout 就被协议独占了，
    #      你自己 print 的任何东西都会污染协议流，所以调试信息只能走 stderr。
    await run_agent(server)


# ================================================================
# 5. 降级演示：缺包时让学员照样看清「ACP 长什么样」
# ================================================================
def print_install_hint() -> None:
    print("【前置条件检查】本机 venv 缺少 ACP 相关依赖：")
    for err in _IMPORT_ERRORS:
        # 把原始 ImportError 文本打出来，学生能直接看到「缺的是哪个模块名」
        print(f"    - {err}")
    print()
    print("请任选一种方式安装（本项目统一用 uv，不往全局环境装）：")
    print("    uv add deepagents-acp        # 会连带装上 acp（deepagents-acp 的依赖）")
    print("    # 或：uv add acp deepagents-acp")
    print("    # 或：F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe -m pip install deepagents-acp")
    print()


def print_protocol_demo() -> None:
    """手工构造一段 ACP 报文，让学员在缺包时也能看到协议的样子。

    这里演示的是「一次完整会话」的最小报文序列（第 02 节会真的收发一遍）：
        ① initialize      握手，交换协议版本与能力
        ② session/new     建会话，拿到 sessionId
        ③ session/prompt  发用户消息
        ④ session/update  通知（无 id）流式推回复，最后一条带 stopReason
    """
    transcript = [
        # ---- ① initialize：整个会话的第一条消息，必须先握手 ----
        # 客户端把自己的协议版本和能力报出来；「能力」是后面能不能用某个功能的依据
        # （例如 fs.readTextFile=true 表示「我可以帮你读文件」）。
        (
            "→ ① 握手（客户端 → 服务端）",
            {
                "jsonrpc": "2.0",      # 固定字符串 "2.0"，不是数字 2.0
                "id": 1,               # 有 id = 请求，服务端必须回一条同 id 的响应
                "method": "initialize",  # 方法名，ACP 一共只有 6 个核心方法
                "params": {
                    "protocolVersion": 1,
                    "clientCapabilities": {"fs": {"readTextFile": True, "writeTextFile": True}},
                },
            },
        ),
        # ---- ① 的应答：id 必须与请求一致（这里是 1），靠它把响应配回请求 ----
        # 服务端同样报出自己的能力；双方之后只在「共同支持」的能力上协作。
        (
            "← ① 握手应答（服务端 → 客户端）",
            {
                "jsonrpc": "2.0",
                "id": 1,               # 和上面那条请求的 id 一一对应，这就是配对规则
                "result": {            # 成功用 result；失败则是 error，二者只出现一个
                    "protocolVersion": 1,
                    "agentCapabilities": {
                        "loadSession": True,
                        "promptCapabilities": {"image": False, "embeddedContext": True},
                    },
                },
            },
        ),
        # ---- ② session/new：建一个会话，后续所有消息都要带这个 sessionId ----
        # cwd 是「工作目录」，告诉 Agent 这个项目在哪；mcpServers 是可选的外部工具来源。
        (
            "→ ② 新建会话",
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "session/new",
                "params": {"cwd": r"F:\ProGram\Python_Base", "mcpServers": []},
            },
        ),
        # 服务端生成 sessionId 并返回；下一节的真实收发里，这个 id 会一路带在 params 里
        ("← ② 会话 ID", {"jsonrpc": "2.0", "id": 2, "result": {"sessionId": "sess_0001"}}),
        # ---- ③ session/prompt：把用户的话发过去（这是唯一会「触发 Agent 干活」的方法）----
        # prompt 是**内容块数组**而不是裸字符串，所以可以夹带文件、图片等结构化内容
        (
            "→ ③ 发送用户消息（prompt 是内容块数组，可夹带文件）",
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "session/prompt",
                "params": {
                    "sessionId": "sess_0001",
                    "prompt": [{"type": "text", "text": "帮我看看这个项目怎么跑起来"}],
                },
            },
        ),
        # ---- ④ session/update：**没有 id**，所以是通知（notification）----
        # 判定规则就一条：有没有 id。没有 id = 单向通知，客户端收到就渲染，不需要回复。
        # Agent 的流式输出全靠它：一次回复被切成很多条这样的通知，一条一条推过来。
        (
            "← ④ 流式推送（**通知：没有 id**，客户端不回复）",
            {
                "jsonrpc": "2.0",
                "method": "session/update",     # 注意：这条没有 "id" 字段
                "params": {
                    "sessionId": "sess_0001",
                    # sessionUpdate 是通知的「子类型」，决定客户端怎么渲染这一块
                    "update": {
                        "sessionUpdate": "agent_message_chunk",   # 回复正文的一个片段
                        "content": {"type": "text", "text": "先执行 uv sync 安装依赖，"},
                    },
                },
            },
        ),
        # 再来一块：同一次回复的第 2 个片段，客户端按到达顺序拼接成完整回答
        (
            "← ④ 再来一块（同一次回复被切成很多块）",
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": "sess_0001",
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": "然后 uv run main.py。"},
                    },
                },
            },
        ),
        # ---- 最后才回 ③ 那条请求的响应（id=3）----
        # 顺序很关键：**先把所有 session/update 推完，最后才回带 id 的响应**，
        # 响应里的 stopReason 说明这一轮为什么结束（end_turn = 正常说完）。
        (
            "← ③ 最后一个带 id 的响应：这一轮结束 + 结束原因",
            {"jsonrpc": "2.0", "id": 3, "result": {"stopReason": "end_turn"}},
        ),
    ]

    print("【降级演示】手工构造的 ACP 会话报文（真实协议格式，仅内容为示意）：")
    for title, message in transcript:
        # 用 separators 去掉空格，输出更接近真实线路上的样子（一行一个 JSON 对象）
        print(f"    {title}")
        print("        " + json.dumps(message, ensure_ascii=False, separators=(",", ":")))
    print()
    # 这一节的降级演示只做到「看懂报文」；真正的收发验证放在下一节，
    # 所以这里先把判定规则钉死，下一节再逐条用真代码验证。
    print("记住三条规则（下一节会逐条验证）：")
    print("    ① 请求/响应成对，靠 id 配对；")
    print("    ② 通知没有 id，服务端不回复（session/update 就是通知）；")
    print("    ③ 一行一个完整 JSON 对象，用 \\n 分隔 —— 所以是「行分隔 JSON-RPC」。")
    print()


def probe_model() -> None:
    """真调一次大模型，证明 settings 三件套可用（规范第 5 节要求留验证证据）。

    这里刻意不用 DeepAgent 去调：DeepAgent 带 shell/file 工具，会真的在磁盘上
    执行命令，作为「跑一遍看输出」的验收动作太重；直接问一句话就足以证明
    api_key / base_url / model_name 通。
    """
    print("【模型连通性自检】settings.model_name =", settings.model_name)
    if not settings.api_key:
        print("    settings.api_key 为空，跳过自检（请检查根目录 .env）")
        return
    try:
        from langchain.chat_models import init_chat_model

        llm = init_chat_model(
            model_provider="openai",
            model=settings.model_name,
            api_key=settings.api_key,
            base_url=settings.base_url,
        )
        reply = llm.invoke("只回复两个字：可用")
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        print("    模型回复：", text.replace("\n", " ")[:60])
    except Exception as exc:  # 网络/额度问题不该让演示崩掉
        print(f"    模型调用失败（{type(exc).__name__}）：{exc}")


def main() -> None:
    # 整个 main 就是「教学设计」本身：先把课案原文摊开给学员看，
    # 再看本机有没有装依赖 —— 装了走真实链路，没装就用手工报文演示协议长什么样。
    print("=" * 66)
    print("ACP 实操：把 Deep Agent 暴露为 ACP 服务端（stdio 模式）")
    print("=" * 66)
    print()

    # ---------- 第 1 步：看课案原文 ----------
    # 逐行打印课案代码，保证读者不用回去翻 HTML 就能对照下面的真实流程
    print("【课案原文】test.py（27 行，本项目已把 conf → config）：")
    for line in COURSE_TEST_PY.splitlines():
        print("    " + line)
    print()

    # ---------- 第 2 步：PyCharm / JetBrains 注册配置 ----------
    # 先看课案那份（路径是课案作者的机器），再看本机可用版（路径动态生成）
    print("【课案原文】C:\\Users\\<用户名>\\.jetbrains\\acp.json：")
    for line in COURSE_ACP_JSON.splitlines():
        print("    " + line)
    print()
    print("【本机可用版】把上面的路径换成动态生成的（可直接复制）：")
    for line in build_local_acp_json().splitlines():
        print("    " + line)
    print()

    # ---------- 第 3 步：装没装？ ----------
    # 三件套齐全 → 真的起 stdio 服务端（会阻塞等编辑器连接，Ctrl+C 退出）；
    # 缺任何一件 → 降级成手工报文演示，让学员照样能看清协议。
    print_install_hint()
    if _ACP_READY:
        print("【依赖检查】acp + deepagents-acp + deepagents 均已就绪，启动真实 ACP 服务端。")
        print("            （注意：stdio 服务端会阻塞等待编辑器的 JSON-RPC 请求，Ctrl+C 退出）")
        print()
        try:
            asyncio.run(serve_acp())
        except KeyboardInterrupt:
            # 服务端本来就是长驻进程，手动中断属于正常退出路径，不算失败
            print("\n已手动中断 ACP 服务端。")
            return
        except Exception as exc:
            # 起不来时给出最可能的两个原因，而不是甩 traceback
            print(f"启动失败（{type(exc).__name__}）：{exc}")
            print("常见原因：解释器不是装 deepagents-acp 的那个；或编辑器未按协议发起 initialize。")
            return
    else:
        print("【降级演示】未安装 acp / deepagents-acp，无法真的 run_agent(server)。")
        print("            下面用「手工构造报文」的方式展示协议交互，安装后本文件会自动切到真流程。")
        print()
        print_protocol_demo()

    # ---------- 第 4 步：确认大模型侧配置没问题 ----------
    # 无论走真流程还是降级，都要证明 settings 三件套可用（规范要求留验证证据）
    probe_model()
    print()
    # 小结把「协议解决什么问题」再收一句口，并指向下一节的真实收发
    print("=" * 66)
    print("小结：ACP = 编辑器与 Agent 之间的一层「LSP 式」标准协议。")
    print("      你只写 create_deep_agent(...) + run_agent(server)，")
    print("      报文封装、stdio 读写、会话管理都由 deepagents-acp 代劳。")
    print("      想知道它到底在传什么 → 继续看 02_acp原理_jxsd.py。")
    print("=" * 66)


if __name__ == "__main__":
    main()

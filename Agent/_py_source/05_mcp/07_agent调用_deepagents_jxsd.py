# -*- coding: utf-8 -*-
"""
MCP ⑦ Agent 调用 · DeepAgents（create_deep_agent）
================================================================
课案这一节只给了一行差异说明：

    和 LangChain 版的唯一区别：create_agent → create_deep_agent，
    额外获得文件系统、任务拆解等内置能力。

本节把这「一行」补成一个完整、可读、能跑的示例，并讲清 DeepAgents 接 MCP 到底差在哪。

【差异到底在哪】

    # | | LangChain | DeepAgents |
    # |---|---|---|
    # | MCP 客户端 | MultiServerMCPClient | **完全相同** |
    # | 获取工具 | client.get_tools() | **完全相同** |
    # | 创建 Agent | create_agent(model, tools) | create_deep_agent(model, tools) |
    # | 额外能力 | 无 | 内置文件系统、任务拆解、子Agent委派 |

关键结论：**DeepAgents 对 MCP 没有任何特殊要求。**
因为 MCP 工具经由 langchain-mcp-adapters 之后已经是标准 LangChain BaseTool，
而 create_deep_agent 接受的就是 BaseTool 列表 —— 协议层和框架层是解耦的，
MCP 在 LangChain 侧「转一次」，下游谁用都一样。

【多出来的能力是什么】

create_deep_agent 会在你给的 tools 之外，自动挂上一整套内置工具（middleware 注入）。
课案的说法是「文件系统、任务拆解、子Agent委派」，本机 deepagents 0.7.13 实测的
内置工具清单如下（下面代码里会**自动打印**出来，版本变了也不用改注释）：

    ls / read_file / write_file / edit_file / delete / glob / grep  —— 文件系统（虚拟工作区）
    execute                                                         —— 在沙箱里执行 shell 命令
    task                                                            —— 把子任务委派给子 Agent（上下文隔离）

注意 0.7.13 里**已经没有 write_todos 这个工具**了，任务拆解改由系统提示词里的
规划规范驱动（不再是一个可调用工具）。所以「课案表格 = 概念」，具体工具名要以你装到的
版本为准 —— 这正是下面那段自动探测代码的用处。

代价：工具变多了，每轮请求的 tool schema 也更长，**token 消耗明显更高**。
所以同一句「2+4*6」，普通 create_agent 一般一轮就结束，
DeepAgents 可能会多绕几步。简单任务用 create_agent 就够了，
DeepAgents 的价值在「需要多步骤 + 需要留下中间产出（文件）的复杂任务」。

课案出处：Agent 课案 → MCP协议 → Agent调用 → DeepAgents

运行方式（本文件自己起 MCP 服务端，跑完自动关闭）：
    uv run Agent/05_mcp/07_agent调用_deepagents_jxsd.py

本机实测结论（这几条正是「课案那行差异说明」展开后的真实样子）：
    ① MCP 客户端代码与 06 一字不差 —— 换框架不用改 MCP 那两行，这是本文件最想证明的事；
    ② 工具清单**不要照抄本注释的数字**：DeepAgents 的内置工具随版本变化，
       本文件下面的代码会把「来自 MCP 的」和「内置的」分两组**自动打印**出来，
       以那次输出为准（文件头的清单也是这么来的，所以版本升级不会让注释变成假话）；
    ③ 课案说的「任务拆解」在 0.7.13 里**已经不是可调用工具**（没有 write_todos 了），
       改由系统提示词里的规划规范驱动 —— 所以「课案表格 = 概念，工具名要看装的版本」；
    ④ Agent 跑 completions 的轮次比普通 create_agent 多，recursion_limit 用默认值容易撞
       GraphRecursionError，本文件显式放大到 50；
    ⑤ 「推回去」只保证**步数**够了，不保证算对。本机跑这一版时出现过两种情况：
       第一种：模型只调一次 mul 就把 2+4*6 答成 28，被推回去后又调了一次 mul，最终仍答 28；
       第二种：正常拆成 mul + add 两步、答 26。
       结论 —— **流程完整性校验是必要的，但它治不了算术错误**；
       真要保证结果正确，得再叠一层结果校验（比如拿表达式自己算一遍比对）。
"""

import asyncio
from pathlib import Path
import socket
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

# 标准库 + 第三方依赖。
# 注意这里**没有** import create_deep_agent：deepagents 属于「未必每台机器都装」的库，
# 放到 _load_create_deep_agent() 里 try/except 延迟导入，缺包时给中文提示而不是崩栈。
import threading

from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import settings

# 标准库 + 第三方依赖；deepagents 刻意不在这里 import（见下面 _load 函数）。
SERVER_SCRIPT = Path(__file__).resolve().parent / "01_服务端_jxsd.py"
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8000
MCP_URL = f"http://{HTTP_HOST}:{HTTP_PORT}/mcp"

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


def _load_create_deep_agent():
    """按规范第 5 条：缺包时不能让模块 import 就炸，要打印中文提示并降级退出。

    deepagents 属于「课案里用到、但未必每台机器都装了」的第三方库，
    所以放在函数里 try/except，而不是模块顶部裸 import。
    """
    # 探测端口：外面已经有服务端就直接连；返回 None 表示「不是我起的」。
    try:
        from deepagents import create_deep_agent
        return create_deep_agent
    except ImportError:
        print("❌ 未安装 deepagents，无法运行本节示例。")
        print("   安装命令： uv add deepagents")
        print("   或者直接看 06_agent调用_langchain_jxsd.py，那个只需要 langchain-mcp-adapters。")
        return None


# 缺包时给中文提示 + 安装命令，并指向 06（那个只需要 langchain-mcp-adapters）。
def _port_in_use(host: str, port: int) -> bool:
    sock = socket.socket()
    sock.settimeout(0.5)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def start_server_in_thread():
    """后台线程起 MCP 服务端；端口被占就直接连已有的。"""
    # importlib 从文件路径加载 01，避开「文件名以数字开头」和「触发 __main__ 自检」两个坑；
    # 注册进 sys.modules 后，pydantic / dataclass 才能正常解析它里面的模型。
    if _port_in_use(HTTP_HOST, HTTP_PORT):
        print(f"ℹ️  {HTTP_HOST}:{HTTP_PORT} 已有 MCP 服务在运行，直接连它。")
        return None

    import importlib.util

    import uvicorn

    spec = importlib.util.spec_from_file_location("mcp_server_01", SERVER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # 和 01/02 一样用 importlib 加载，避开文件名与 __main__ 自检两个坑。
    sys.modules["mcp_server_01"] = module
    spec.loader.exec_module(module)

    app = module.mcp.http_app(transport="streamable-http", path="/mcp")
    server = uvicorn.Server(
        uvicorn.Config(app, host=HTTP_HOST, port=HTTP_PORT, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            return server, thread
        time.sleep(0.1)
    print(f"❌ MCP 服务端启动失败：{HTTP_HOST}:{HTTP_PORT} 无法监听。")
    return None


def _text_of(content) -> str:
    """DeepAgents / LangChain 1.x 的 message.content 可能是 str，也可能是内容块列表。

    取最后一条回答时两种形态都要能处理，否则会打出 "[{'type': 'text', ...}]"。
    """
    # 内容块列表是 LangChain 1.x 之后的新形态（多模态 / 推理内容都用它表示），
    # 所以取值必须两种形态都兼容，否则打印出来是一串 dict。
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                # 两种形态都要处理：str 直接用，list 则把 text 块拼起来。
                parts.append(str(block))
        return "".join(parts)
    return str(content)


async def main(create_deep_agent) -> None:
    # ① MCP 客户端：和 06 LangChain 版**一字不差**
    # 配置里只有一个 MCP 服务，和 06 的写法**一字不差** —— 这是本文件要证明的核心。
    client = MultiServerMCPClient(
        {
            "math_tools": {
                "url": MCP_URL,
                "transport": "streamable_http",
            }
        }
    )

    # ② 获取工具：也一字不差
    tools = await client.get_tools()
    print("获取到的 MCP 工具:", [t.name for t in tools])

    # ③ 唯一的区别：create_agent → create_deep_agent
    # 唯一的区别就在这一行：create_agent → create_deep_agent。
    agent = create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "你是一个计算助手。必须调用工具做每一步运算，禁止心算；"
            "表达式里有几步运算就调几次工具，算完后直接给出最终数字。"
        ),
    )

    # DeepAgents 到底比普通 Agent 多了哪些工具？直接查编译后的图。
    # LangGraph 编译出来的图里有一个名为 "tools" 的节点（ToolNode），
    # 它的 .bound.tools_by_name 就是「这个 Agent 真正能调用的全部工具」。
    # 把结果分成「你给的 MCP 工具」和「它内置的工具」两组，差异一眼看清。
    # 用集合差集把「你给的」和「它自带的」分开，版本升级导致工具清单变化也能一眼看出。
    mcp_names = {t.name for t in tools}
    all_names = _collect_tool_names(agent)
    builtin = sorted(all_names - mcp_names)
    print(f"\nAgent 可用的全部工具 {len(all_names)} 个：")
    print(f"  · 来自 MCP（你提供的）：{sorted(mcp_names)}")
    print(f"  · 内置（DeepAgents 自动挂上的）：{builtin}")

    print("\n调用 agent...（DeepAgents 会先拆任务，比普通 Agent 多几轮，请稍等）")
    # recursion_limit：DeepAgents 的循环比普通 Agent 长（内置工具 + 子Agent 委派），
    # 默认上限容易撞到 GraphRecursionError，工程里一般显式放大。
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "2+4*6"}]},
        config={"recursion_limit": 50},
    )

    # ⚠️ 流程完整性校验（和 06 完全一样的写法，说明这个套路与框架无关）：
    # 表达式至少需要几步运算，就至少该发生几次工具调用；少调了就把历史带上再推一次。
    # 本机实测：不加这段，模型经常只调一次 mul 就把 2+4*6 答成 24。
    # 注意这里和 06 的差别：**校验逻辑一字未改**，只有 agent 对象换了实现。
    required_steps = sum("2+4*6".count(op) for op in "+-*/")     # = 2
    for push_back in range(1, 4):        # 最多推 3 次，避免模型死活不改导致无限循环
        # tool_calls 分散在各条 AIMessage 上，逐条累加才是「实际发生了几次工具调用」
        made = sum(len(getattr(m, "tool_calls", None) or []) for m in result["messages"])
        if made >= required_steps:
            break                        # 步数达标就收工，不再多问模型一次
        print(f"⚠️  只发生了 {made} 次工具调用，2+4*6 至少需要 {required_steps} 步 —— "
              f"把历史带上再推它一次（第 {push_back} 次）。")
        # 一定要把已有历史整段带上再补 user 消息，
        # 否则模型会丢掉中间结果从零重算；config 也要重传（recursion_limit 不会自动继承）
        # 重推时 config 必须一起传：recursion_limit 不会自动继承上一次调用。
        result = await agent.ainvoke(
            {
                "messages": list(result["messages"]) + [{
                    "role": "user",
                    "content": f"你只调用了 {made} 次工具，还有运算没算完"
                               f"（至少需要 {required_steps} 步）。"
                               "请继续调用工具算出剩余步骤，禁止心算，算完给出最终数字。",
                }]
            },
            config={"recursion_limit": 50},
        )

    # 打印轨迹时把每次调用标上「MCP」或「内置」——
    # 这是 DeepAgents 最容易被误解的地方：工具列表变长之后，读者要看得出「模型调的是谁」
    print("\n--- 消息轨迹（注意有没有内置工具出现） ---")
    for msg in result["messages"]:
        kind = type(msg).__name__
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                # mcp_names 是「你通过 MultiServerMCPClient 提供的那些工具名」的集合
                origin = "MCP" if tc["name"] in mcp_names else "内置"
                # 内置工具（如 write_file）的 args 可能很长，截断到 90 字避免刷屏
                print(f"  [{kind}][{origin}] 调用 {tc['name']}，参数 {str(tc['args'])[:90]}")
        elif kind == "ToolMessage":
            # 工具返回值同样截断：文件内容类的返回动不动几千字符
            print(f"  [{kind}] 返回 {_text_of(msg.content)[:90]}")
        elif getattr(msg, "content", None):
            print(f"  [{kind}] {_text_of(msg.content)[:90]}")

    print("\n最终答案:")
    # content 可能是 str 也可能是内容块列表，统一走 _text_of 取出纯文本
    print(_text_of(result["messages"][-1].content))


def _collect_tool_names(agent) -> set[str]:
    """从编译后的 LangGraph 图里读出「这个 Agent 能调用的全部工具名」。

    实现要点：CompiledStateGraph.nodes["tools"] 是 ToolNode，
    它的 .bound.tools_by_name 是 {工具名: BaseTool} 字典。

    不同 deepagents 版本的内部结构会变，所以这里做「能读到就读，
    读不到就返回空集合」的宽松处理 —— 探测失败绝不能中断教学演示。
    """
    # 读不到就返回空集合：不同 deepagents 版本的内部结构会变，
    # 探测失败绝不能中断教学演示。
    names: set[str] = set()
    try:
        tool_node = agent.nodes["tools"]
        names.update(tool_node.bound.tools_by_name.keys())
    except Exception:
        pass
    # 读不到工具清单时返回空集合，最多让「内置工具」那一行显示不全，不影响主流程。
    return names


if __name__ == "__main__":
    create_deep_agent = _load_create_deep_agent()
    if create_deep_agent is None:
        sys.exit(0)                       # 缺包时正常退出，不抛 traceback

    # 缺包时用 sys.exit(0) 正常退出：这是「环境没装好」，不是脚本出错。
    started = start_server_in_thread()
    try:
        asyncio.run(main(create_deep_agent))
    finally:
        if started:
            server, thread = started
            # 缺包走 sys.exit(0)，不抛 traceback —— 环境没装好不是代码错误。
            server.should_exit = True
            thread.join(timeout=10)
            print("\n✅ 本文件启动的 MCP 服务端已关闭。")

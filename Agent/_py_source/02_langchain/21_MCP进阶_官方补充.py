# -*- coding: utf-8 -*-
r"""
LangChain 官方补充篇：MCP 进阶（非课案内容，故不带 _jxsd 后缀）
================================================================
来源与定位：
    本文件对照 LangChain **官方文档**：
      - /oss/python/langchain/mcp/index.mdx        （MCP 总览与传输方式）
      - /oss/python/langchain/mcp/connections.mdx  （连接生命周期 / 多服务端 / 命名空间）
      - /oss/python/langchain/mcp/auth.mdx         （认证）
      - /oss/python/deepagents/tools.mdx#mcp-tools （DeepAgents 侧接入 MCP）
    对应缺口表里 **LangChain 第 9 项**和 **DeepAgents 第 8 项**（一次覆盖两格）。

与课案 05_mcp 章的关系：
    课案讲的是 **MCP 协议本身**：服务端怎么写、客户端怎么连、资源 / 提示词 / 工具三原语、
    JWT 权限与认证、部署与调试 —— 那部分不再重复。
    本文件补的是**连接层与工程接入**：连接生命周期、多服务端聚合与命名空间、
    把 MCP 工具接进 create_agent 与 create_deep_agent、以及返回值的真实形态。

⚠️ 本机现实（很重要，决定了本文件为什么这么写）：
    官方**新** API 是 `langchain.mcp.MCPAdapter`（要求 `langchain[mcp]>=1.4.0`，beta）。
    本机实测它**不可用**：
        ImportError: No module named 'fastmcp.client.group'
    原因是本机 fastmcp 是 3.4.7，而 `langchain.mcp` 需要 fastmcp 4.x
    （本地 banner 已在提示 "Update available: 4.0.4"）。
    升级 fastmcp 会改动 pyproject.toml / uv.lock —— 而这两个文件正在被别的改动占用，
    所以本文件改用**已安装的经典适配器** `langchain-mcp-adapters`（课案 06/07 章同款），
    并在每处标注官方新 API 的对应写法，等依赖升级后照着换即可。

本文件的运行设计（尽量自包含、少占资源）：
    1. 服务端脚本**现场生成到临时目录**（不往仓库塞演示文件），结束自动清理；
    2. Demo 1~3 用 **stdio 传输**拉起服务端子进程 —— 不占端口、不走网络，
       因此不会遇到课案 05_mcp 章那种 8000 端口冲突与代理拦截问题；
    3. Demo 4 改用**进程内 HTTP 服务端**（127.0.0.1:8770，避开课案常驻服务的 8000）：
       因为实测 **stdio 传输 + create_deep_agent 会卡死**，HTTP 才正常
       （详见文末实测结论第 2 条）；用完在 finally 里关掉 uvicorn。

运行方式（项目根目录下，需真实模型）：
    uv run Agent/02_langchain/21_MCP进阶_官方补充.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import asyncio
import os
import tempfile
from pathlib import Path

# 本机开着 Clash 等系统代理时，127.0.0.1 的回环请求会被代理接管（课案排障一节记录过，
# 表现为 McpError: Session terminated / 502）。Demo 4 要连本机进程内的 HTTP 服务端，
# 所以这里给进程设上 NO_PROXY，保证回环直连。
# 追加式写法：环境里若已有 NO_PROXY（用户自己设的白名单），把回环补进去，
# 而不是因为 setdefault 而整体失效。
for _proxy_key in ("NO_PROXY", "no_proxy"):
    _existing = os.environ.get(_proxy_key, "")
    if "127.0.0.1" not in _existing:
        os.environ[_proxy_key] = (_existing + "," if _existing else "") + "127.0.0.1,localhost"

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import settings

model = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

# 两个演示服务端的源码：现场写到临时目录再以子进程拉起
# 脚本开头把日志压到 ERROR：fastmcp 启动时会打 INFO 日志
# （"Starting MCP server ... with transport 'stdio'"），演示里纯属噪音。
WEATHER_SERVER = '''
from fastmcp import FastMCP
import logging

logging.getLogger("fastmcp").setLevel(logging.ERROR)   # 压掉它的启动 INFO 日志

mcp = FastMCP("weather-server")


@mcp.tool
def get_weather(city: str) -> str:
    """查询指定城市的天气。"""
    return f"{city}：晴，25℃"


@mcp.tool
def search(query: str) -> str:
    """天气服务里的检索（故意与另一个服务端同名，用来演示命名冲突）。"""
    return f"[天气服务] 命中：{query}"


@mcp.resource("weather://cities")
def supported_cities() -> str:
    """本服务支持的城市清单。"""
    return "北京、上海、广州、深圳"


if __name__ == "__main__":
    # show_banner=False：关掉 FastMCP 的启动 banner（否则会刷满输出，编码还可能错乱）
    mcp.run(show_banner=False)
'''

DOCS_SERVER = '''
from fastmcp import FastMCP
import logging

logging.getLogger("fastmcp").setLevel(logging.ERROR)   # 压掉它的启动 INFO 日志

mcp = FastMCP("docs-server")


@mcp.tool
def search(query: str) -> str:
    """文档服务里的检索（与天气服务同名）。"""
    return f"[文档服务] 命中：{query}"


@mcp.prompt
def explain_topic(topic: str) -> str:
    """生成一段用于讲解某主题的提示词。"""
    return f"请用三句话解释 {topic}，面向初学者。"


if __name__ == "__main__":
    # show_banner=False：关掉 FastMCP 的启动 banner（否则会刷满输出，编码还可能错乱）
    mcp.run(show_banner=False)
'''


def build_servers(root: Path) -> tuple[Path, Path]:
    """把两个演示服务端写到临时目录，返回脚本路径。"""
    weather = root / "weather_server.py"
    docs = root / "docs_server.py"
    weather.write_text(WEATHER_SERVER, encoding="utf-8")
    docs.write_text(DOCS_SERVER, encoding="utf-8")
    return weather, docs


def stdio_connection(script: Path) -> dict:
    """stdio 连接的配置形状（官方新 API 里叫 transport，经典适配器同理）。"""
    return {"command": sys.executable, "args": [str(script)], "transport": "stdio"}


# ---- Demo 4 专用的「进程内 HTTP 服务端」 --------------------------------------
# 为什么 Demo 4 不用 stdio：实测 **create_deep_agent + stdio 工具会卡死**
# （240 秒不返回，而同样的工具交给 create_agent 秒回，见文末实测结论第 2 条）。
# 课案 05_mcp/07_agent调用_deepagents_jxsd.py 走的是 streamable-http，已验证可用，
# 这里照搬那套：FastMCP 的 http_app + uvicorn 跑在后台线程里，客户端用 URL 连。
# 端口特意避开课案常驻服务占用的 8000，改用 8770。
HTTP_PORT = 8770
HTTP_URL = f"http://127.0.0.1:{HTTP_PORT}/mcp"


def start_inprocess_mcp_server():
    """在后台线程里起一个进程内 MCP 服务端（返回 uvicorn.Server，用于收尾关闭）。"""
    import threading
    import time as _time

    import uvicorn
    from fastmcp import FastMCP

    mcp = FastMCP("weather-http-server")

    @mcp.tool
    def get_weather(city: str) -> str:
        """查询指定城市的天气。"""
        return f"{city}：晴，25℃"

    app = mcp.http_app(transport="streamable-http", path="/mcp")
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=HTTP_PORT, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):          # 最多等 10 秒，轮询到端口真正就绪
        if server.started:
            return server
        _time.sleep(0.1)
    raise RuntimeError(f"进程内 MCP 服务端启动失败（127.0.0.1:{HTTP_PORT} 未监听）")


def tool_result_text(result) -> str:
    """把 MCP 工具的返回值转成人看的文本。

    ⚠️ 实测要点：MCP 工具返回的是 **content block 列表**（形如
    [{'type': 'text', 'text': '...', 'id': ...}]），不是普通字符串 ——
    直接当 str 用会打印出一大坨字典。
    """
    if isinstance(result, list):
        parts = []
        for item in result:
            if isinstance(item, dict):
                parts.append(str(item.get("text", item)))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(result)


# ================================================================
# Demo 1：连接生命周期 —— 发现工具 → 构建 agent → 调用
# ================================================================
# 官方 connections.mdx 的核心结论：
#   「Discovery happens inside the context, but the tools it returns hold the client,
#     so they stay callable after the context exits.」
# 即：用 async with 打开适配器 → 在上下文内发现工具 → 出了上下文工具**照样能调**，
# 因为工具本身持有客户端（每次调用时自己开一次会话，用完就放）。
#
# 经典适配器的写法与官方新 API 的对照：
#     经典（本文件）：client = MultiServerMCPClient({...}); tools = await client.get_tools()
#     官方新 API   ：async with MCPAdapter(target) as adapter: tools = await adapter.list_tools()
async def demo_1_lifecycle(weather_script: Path) -> None:
    print("=" * 70)
    print("Demo 1：连接生命周期 —— 发现工具后即可脱离上下文使用")
    print("=" * 70)

    client = MultiServerMCPClient({"weather": stdio_connection(weather_script)})
    tools = await client.get_tools()
    print(f"  发现 {len(tools)} 个工具：{[t.name for t in tools]}")

    # 工具拿在手里后，不需要保持连接上下文（工具每次调用自己开会话）
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="你可以调用天气工具。回答简洁，一句话以内。",
    )
    # ⚠️ MCP 工具是**异步**实现：必须走异步入口 ainvoke（同步 invoke 会报错）
    result = await agent.ainvoke({"messages": [{"role": "user", "content": "北京天气怎么样？"}]})
    print(f"  agent 回答：{str(result['messages'][-1].content)[:100]}")
    print(
        "  ↑ 关键点：工具在**发现时**建立连接，在**每次调用时**各自开会话、用完即放 ——\n"
        "    所以长跑的 agent 不会攥着一条空闲连接。这也是官方强调的\n"
        "    「one session per invocation」语义。"
    )


# ================================================================
# Demo 2：多服务端聚合 + 同名工具的命名空间问题
# ================================================================
# 官方 connections.mdx 专门讲了这件事：
#   用 MCPConfig 聚合多个服务端时，**每个工具会自动带上配置键前缀**
#   （`weather_search` / `docs_search`），所以同名工具不会撞车。
# 本机经典适配器**不会**自动加前缀 —— 两个 search 会重名（实测见输出），
# 迁移到官方新 API 时这一点是**行为差异**，本 Demo 演示怎么手工补救。
async def demo_2_multiple_servers(weather_script: Path, docs_script: Path) -> None:
    print("\n" + "=" * 70)
    print("Demo 2：多服务端聚合 —— 同名工具的冲突与手工命名空间")
    print("=" * 70)

    client = MultiServerMCPClient({
        "weather": stdio_connection(weather_script),
        "docs": stdio_connection(docs_script),
    })
    tools = await client.get_tools()
    names = [t.name for t in tools]
    print(f"  两个服务端合计发现 {len(tools)} 个工具：{names}")
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        print(f"  ⚠️ 重名工具：{sorted(duplicates)} ← 经典适配器不会自动加服务端前缀")
        print("     官方新 API（MCPConfig）会自动加前缀（weather_search / docs_search）；")
        print("     用经典适配器就得自己改名字，否则模型看到两个同名工具会随机选一个。")

    # 手工打前缀的**正确**做法：按服务端分别取工具（get_tools(server_name=...)），
    # 而不是去猜工具描述属于谁 —— 实测踩过：靠描述里的关键词判断会把
    # 「文档服务里的检索（与天气服务同名）」误判成天气服务，两个工具前缀撞在一起。
    renamed = []
    for server_name in ("weather", "docs"):
        for tool in await client.get_tools(server_name=server_name):
            tool.name = f"{server_name}_{tool.name}"
            renamed.append(tool)
    print(f"  按服务端手工加前缀后：{[t.name for t in renamed]}")

    # 分别调用两个同名工具，证明它们确实是两个不同的实现
    for tool in renamed:
        if tool.name.endswith("search"):
            result = await tool.ainvoke({"query": "LangGraph"})
            print(f"    {tool.name} → {tool_result_text(result)}")
    print(
        "  ↑ 多服务端接入的第一道坎不是连接，而是**命名**：\n"
        "    工具名是给模型看的唯一标识，重名等于让模型掷骰子；\n"
        "    正确姿势是按 server_name 分别取工具再加前缀，别靠描述文本猜来源。"
    )


# ================================================================
# Demo 3：MCP 的三类原语在客户端侧怎么取
# ================================================================
# MCP 有三个原语：工具（tools）、资源（resources）、提示词（prompts）。
# 课案 05_mcp 讲了服务端怎么定义；这里看客户端怎么取（经典适配器的三个方法）。
async def demo_3_resources_and_prompts(weather_script: Path, docs_script: Path) -> None:
    print("\n" + "=" * 70)
    print("Demo 3：资源与提示词 —— 客户端侧的取法")
    print("=" * 70)

    client = MultiServerMCPClient({
        "weather": stdio_connection(weather_script),
        "docs": stdio_connection(docs_script),
    })

    # 资源：只读数据（这里用自定义 URI 方案 weather://cities）
    try:
        resources = await client.get_resources("weather")
        print(f"  weather 服务端的资源（{len(resources)} 个）：")
        for item in resources:
            print(f"    uri={getattr(item, 'uri', item)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  取资源失败：{type(exc).__name__}: {str(exc)[:120]}")

    # 提示词：服务端预置的提示模板
    try:
        prompt = await client.get_prompt("docs", "explain_topic", arguments={"topic": "MCP 协议"})
        text = getattr(prompt, "messages", prompt)
        print(f"  docs 服务端的提示词 explain_topic → {str(text)[:120]}")
    except Exception as exc:  # noqa: BLE001
        print(f"  取提示词失败：{type(exc).__name__}: {str(exc)[:120]}")

    print(
        "  ↑ 三类原语的客户端取法（经典适配器）：\n"
        "    tools     → await client.get_tools()          （能直接交给 agent）\n"
        "    resources → await client.get_resources(server)（只读数据，要自己喂给模型）\n"
        "    prompts   → await client.get_prompt(server, name, arguments=...)\n"
        "    官方新 API 里资源与提示词同样由 MCPAdapter 暴露。"
    )


# ================================================================
# Demo 4：把 MCP 工具交给 DeepAgents（覆盖缺口表 DeepAgents 第 8 项）
# ================================================================
# 官方 deepagents/tools.mdx#mcp-tools 的原文结论很简单：
#   「Load tools from any MCP server and pass them directly to create_deep_agent.」
# 也就是说 DeepAgents 自己**没有** MCP 适配层（本地实测：deepagents 包里
# 没有任何 MCP 符号），它只是接收 LangChain 工具 —— 所以 Demo 1 拿到的工具
# 原样塞进去即可，同时白拿 deepagents 的文件系统、子代理等能力。
async def demo_4_deepagents() -> None:
    print("\n" + "=" * 70)
    print("Demo 4：MCP 工具 → create_deep_agent（DeepAgents 第 8 项）")
    print("=" * 70)

    from deepagents import create_deep_agent

    client = MultiServerMCPClient({"weather": {"url": HTTP_URL, "transport": "streamable_http"}})
    tools = await client.get_tools()

    agent = create_deep_agent(
        model=model,
        tools=tools,          # ← MCP 工具直接传进去，和普通工具没有区别
        system_prompt="你可以调用 MCP 天气工具。回答一句话即可。",
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "上海天气如何？"}]},
        config={"recursion_limit": 20},   # 兜底：防止意外的长循环把演示拖死
    )
    print(f"  发现 {len(tools)} 个 MCP 工具：{[t.name for t in tools]}")
    print(f"  deep agent 回答：{str(result['messages'][-1].content)[:100]}")
    print(
        "  ↑ DeepAgents 不重复造 MCP 适配层：**工具就是工具**。\n"
        "    所以「MCP + 深度智能体」的组合成本极低 —— 接完协议，文件系统/子代理/\n"
        "    上下文压缩全部白拿（课案 03_deepagents 章讲的那些能力）。\n"
        "    ⚠️ 但传输方式有讲究：本 Demo 走 HTTP 而不是 stdio，原因见文末实测结论第 2 条。"
    )


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mcp_adv_") as tmp:
        root = Path(tmp)
        weather_script, docs_script = build_servers(root)
        print(f"演示服务端已生成到临时目录：{root}\n")
        await demo_1_lifecycle(weather_script)
        await demo_2_multiple_servers(weather_script, docs_script)
        await demo_3_resources_and_prompts(weather_script, docs_script)

        # Demo 4 用进程内 HTTP 服务端（stdio 与 deepagents 组合会卡死，见文末结论）
        print(f"\n（Demo 4 启动进程内 HTTP 服务端：{HTTP_URL}）")
        server = start_inprocess_mcp_server()
        try:
            await demo_4_deepagents()
        finally:
            server.should_exit = True      # 收尾：让 uvicorn 优雅退出
    print("\n全部 Demo 执行完毕（临时目录已清理，HTTP 服务端已关闭）。")


if __name__ == "__main__":
    asyncio.run(main())


# ================================================================
# 实测结论 / 与课案的衔接 / 踩坑提示
# ================================================================
# 1. 本机实测（langchain-mcp-adapters 0.3.2 / fastmcp 3.4.7 / deepagents 0.7.13）：
#    - 官方新 API `langchain.mcp.MCPAdapter` **不可用**：导入即报
#      `No module named 'fastmcp.client.group'`（需要 fastmcp 4.x，本机是 3.4.7）；
#    - 经典适配器 MultiServerMCPClient 可用：`await client.get_tools()` 拿到工具，
#      `await tool.ainvoke({...})` 调用成功（实测本文件服务端的 get_weather / search
#      都能正常调用，返回值是 content block 列表）；
#    - **MCP 工具的返回值是 content block 列表**（[{'type': 'text', 'text': ...}]），
#      不是字符串 —— 直接 str() 会打出一坨字典（本文件提供了 tool_result_text 转换）；
#    - MCP 工具是异步实现，agent 必须走 `ainvoke`（同步 invoke 不适用）；
#    - `await client.get_tools(server_name="docs")` 可按服务端分别取工具 ——
#      这是「手工加命名空间前缀」的正确姿势（靠工具描述猜来源会误判，实测踩过）；
#    - `get_resources()` 返回 Blob（含 uri/mimetype/data），
#      `get_prompt()` 返回**消息列表**（[HumanMessage(...)]），不是字符串；
#    - deepagents 包里**没有任何 MCP 符号**：官方路线就是把工具传进去。
# 2. ★ 重要发现：**stdio 传输 + create_deep_agent 会卡死**
#    - 同样一个 stdio MCP 工具：交给 `create_agent` 秒回（Demo 1）；
#      交给 `create_deep_agent` 则 **240 秒不返回**（起两次实测均如此，已加超时保护）；
#    - 换成 **streamable_http 传输**（Demo 4 的写法）立刻正常返回；
#    - 结论：DeepAgents 接 MCP 时**用 HTTP 传输**；课案 05_mcp/07 用的正是 HTTP，
#      与本文件结论一致。stdio 为什么冲突未再深挖（属依赖内部行为，不值得占课时）。
# 3. 与课案 05_mcp 章的衔接：
#    课案 = 协议本身（服务端/客户端/三原语/JWT 权限/部署/调试）；
#    本文件 = 连接层与工程接入（生命周期、多服务端命名空间、接 agent 与 deep agent）。
#    权限认证部分课案已用 07/08/09 三个文件讲透（JWT + scope + 服务端校验），
#    对应官方 mcp/auth.mdx；本文件不重复，那一套需要起 9000/8000 两个常驻服务。
# 4. 未收录（官方还有、本文件没做的）：
#    - 官方新 API 的 **MCPConfig 自动前缀**与 **ClientGroup**（每服务端独立连接、
#      可混协议代际）：都要 fastmcp 4.x，本机依赖不具备，接口写法已在文件里标注；
#    - **OAuth / bearer 认证的客户端侧**（mcp/auth.mdx）：课案 05_mcp 用 JWT 讲了同类机制，
#      本文件保持"自包含"没有起常驻服务；
#    - **部署侧共享连接池与缓存**（connections.mdx 的 Scale a deployment）：
#      属于服务化部署话题，课案 11_部署 讲的是另一条线。
# 5. 踩坑提示：
#    A. MCP 工具返回值是 **content block 列表**，喂给模型前最好自己转文本；
#    B. MCP 工具是异步的：`agent.invoke()` 会失败，用 `await agent.ainvoke()`；
#    C. 多服务端必须处理**同名工具**：经典适配器不会自动加前缀（官方新 API 会），
#       重名会让模型随机挑一个 —— 用 get_tools(server_name=...) 分别取名再加前缀；
#    D. stdio 传输下服务端是**子进程**：脚本路径要用绝对路径，
#       并且记得进程会在客户端退出时被回收（本文件用临时目录 + 自动清理）；
#    E. fastmcp 的启动 banner 与 INFO 日志这样压：
#       `mcp.run(show_banner=False)` + 服务端脚本里
#       `logging.getLogger("fastmcp").setLevel(logging.ERROR)` ——
#       注意是**fastmcp 这个 logger**：设 root 级别**无效**（fastmcp 用自己的 handler
#       且不向 root 传播；实测设了 root 级别日志照打，换成它的 logger 才安静）；
#    F. 进程内 HTTP 服务端记得收尾 `server.should_exit = True`，
#       否则 uvicorn 线程会吊着端口不放（本文件在 finally 里做）；
#    G. 连本机回环服务端前设 `NO_PROXY=127.0.0.1,localhost`：
#       系统代理会接管回环请求（课案排障一节记录过），本文件已在模块顶部设好。

# -*- coding: utf-8 -*-
"""
MCP ① 快速开始 · 服务端（FastMCP）
================================================================
MCP（Model Context Protocol，模型上下文协议）是一种**标准化的工具调用协议**，
可以把它理解成「AI 世界的 USB 接口」：

    工具作者   ——按 MCP 规范写一次服务端——→  任何支持 MCP 的客户端都能直接插上
    客户端     ——Claude Desktop / Cursor / LangChain / DeepAgents / 自研 Agent

没有 MCP 之前，每换一个 Agent 框架就要把工具函数重抄一遍；有了 MCP，
「工具」变成了一个独立进程/独立服务，框架只负责连上来。

本节（快速开始 · 服务端）要讲清三件事：

    1. 如何用 FastMCP 起一个 MCP 服务端 —— 三行核心代码：
           from fastmcp import FastMCP
           mcp = FastMCP("演示 🚀")
           @mcp.tool
           def add(a: float, b: float) -> float: ...
    2. **函数的类型注解 + docstring 就是给模型看的工具说明书**。
       FastMCP 会自动把签名转成 JSON Schema（参数名/类型/必填），
       把 docstring 转成 description。注解写错 → 模型调用参数就错。
    3. 四种传输方式怎么启动（本文件最后一节把课案那张对比表完整保留）。

    # | 传输 | 启动写法 | 特点 |
    # |---|---|---|
    # | stdio | mcp.run(transport="stdio") | 本地子进程，客户端拉起本脚本，走标准输入输出 |
    # | http | mcp.run(transport="http", port=8000, host="0.0.0.0") | 普通请求/响应，类 REST，不支持流式 |
    # | streamable-http | mcp.run(transport="streamable-http", ...) | 全双工，可流式推送中间状态（生产推荐） |
    # | sse | mcp.run(transport="sse", port=8001, host="0.0.0.0") | 半双工，**已弃用**，被 streamable-http 取代 |

课案出处：Agent 课案 → MCP协议 → 快速开始 → 服务端
          Agent 课案 → MCP协议 → 四种协议

运行方式（无参数 = 自检演示，跑完自动退出，不会卡住）：
    uv run Agent/05_mcp/01_服务端_jxsd.py

真要起一个常驻服务给别的窗口/别的文件连时：
    uv run Agent/05_mcp/01_服务端_jxsd.py http     # streamable-http，127.0.0.1:8000/mcp
    uv run Agent/05_mcp/01_服务端_jxsd.py sse      # 已弃用的 sse，127.0.0.1:8001/sse
    uv run Agent/05_mcp/01_服务端_jxsd.py stdio    # stdio，交给客户端当子进程拉起
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from fastmcp import FastMCP


# ---------- 1. 创建服务实例 ----------
# name 会出现在客户端的服务列表里（课案里叫 "演示 🚀"，emoji 是故意加的：
# 用来验证从服务端 → 传输 → 客户端整条链路的 UTF-8 编码是否正常）。
mcp = FastMCP("演示 🚀")


# ---------- 2. 用 @mcp.tool 把普通函数变成 MCP 工具 ----------
# 装饰器做的事情：
#   1) 读函数签名 → 生成 JSON Schema {"a": {"type": "number"}, "b": {...}, "required": [...]}
#   2) 读 docstring  → 生成工具的 description（模型就是靠这段话判断「什么时候该调它」）
#   3) 注册进服务端的工具表，客户端 list_tools() 就能看到
# 课案写法是 @mcp.tool（不带括号）；想改显示名可以写 @mcp.tool(title="加法")
@mcp.tool
def add(a: float, b: float) -> float:
    """两数相加
    :param a:第一个数字
    :param b:第二个数字
    :return: a+b的结果
    """
    return a + b


@mcp.tool
def sub(a: float, b: float) -> float:
    """两数相减
    :param a:第一个数字
    :param b:第二个数字
    :return: a-b的结果
    """
    return a - b


@mcp.tool
def mul(a: float, b: float) -> float:
    """两数相乘
    :param a:第一个数字
    :param b:第二个数字
    :return: a*b的结果
    """
    return a * b


@mcp.tool
def div(a: float, b: float) -> float:
    """两数相除
    :param a:第一个数字
    :param b:第二个数字
    :return: a/b的结果
    """
    # 除零是「工具里唯一的业务异常」。MCP 会把抛出的异常包成 is_error=True 的结果
    # 回给客户端（而不是让整条连接崩掉），模型看到错误信息后可以自己纠正参数重试。
    if b == 0:
        raise ValueError("除数不能为 0")
    return a / b


# ---------- 3. 四种协议（课案 4332-4395 原文搬进注释） ----------
# 数据传输方式：
#   - 单工：单向传输信息
#   - 半双工：一次请求，多次返回，不可打断
#   - 全双工：一次请求，多次返回，可添加新的请求参数
#
# 协议：
#   1. stdio：本地运行，直接开启子进程，运行 py
#   2. http
#      - 普通 HTTP 请求/响应模式，类似于传统的 REST API
#      - 每个请求都是完整的响应，不支持流式传输
#      - 适合：简单、快速的返回
#   3. streamable-http：全双工
#      - 支持流式传输的 HTTP 模式
#      - 可以逐步返回结果，实时推送中间状态
#      - 适合耗时操作、长响应场景（如大文件处理、复杂计算）
#      - 客户端可以边接收边处理，不需要等待完整响应
#   4. sse：半双工，已弃用，被 streamable-http 取代
#      - 客户端需要等待处理完成，一次请求，多次返回，不可打断
#
# 对应的启动代码就是课案那四行（注意 sse 那行已经不建议用了）：
#     # mcp.run(transport="stdio")
#     # mcp.run(transport="http", port=8000, host="0.0.0.0")
#     mcp.run(transport="streamable-http", port=8000, host="0.0.0.0")
#     # mcp.run(transport="sse", port=8001, host="0.0.0.0")  # 弃用
#
# 选哪个？
#   - 本机给 Claude Desktop / LangChain 用  → stdio（省去端口和鉴权，客户端帮你拉进程）
#   - 部署成网络服务 / 多客户端共享         → streamable-http（可流式、可挂中间件、可多进程）
#   - 老客户端兼容                          → 才考虑 sse，新项目别用

# 本文件对外服务的地址（与 02_客户端_jxsd.py 约定一致）
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8000
MCP_PATH = "/mcp"


def run_stdio() -> None:
    """方式一：stdio —— 不监听端口，靠标准输入输出和父进程说话。

    客户端（如 02_客户端_jxsd.py 里的 StdioTransport）会用
    `python 01_服务端_jxsd.py` 把本脚本当**子进程**拉起来，
    然后通过 stdin/stdout 收发 JSON-RPC。所以这个模式在终端里
    单独跑起来会「看起来卡住」——它在等父进程发消息，这是正常的。
    """
    mcp.run(transport="stdio")


def run_http() -> None:
    """方式二：http —— 普通请求/响应，类 REST，不支持流式。"""
    mcp.run(transport="http", host=HTTP_HOST, port=HTTP_PORT, path=MCP_PATH)


def run_streamable_http() -> None:
    """方式三：streamable-http —— 全双工、可流式（课案默认推荐，也是本套文件默认）。"""
    mcp.run(transport="streamable-http", host=HTTP_HOST, port=HTTP_PORT, path=MCP_PATH)


def run_sse() -> None:
    """方式四：sse —— 半双工，已弃用，只作教学对照。"""
    mcp.run(transport="sse", host=HTTP_HOST, port=8001)


def self_check() -> None:
    """无参数运行时的自检演示：后台线程起 streamable-http，自己连自己看一眼。

    这样学员「只运行一个文件」就能确认服务端真的起来了、工具真的注册进去了，
    跑完还能自动退出，不会把终端卡在服务进程上。
    真正要常驻服务时请用 `... http` / `... sse` / `... stdio` 参数。
    """
    import threading
    import time

    import uvicorn
    from fastmcp import Client

    # mcp.http_app() 把 FastMCP 实例包成一个标准 ASGI 应用，
    # 于是可以交给任意 ASGI 服务器（uvicorn / hypercorn）跑，也能挂 --workers 多进程。
    # 这正是 11_部署_jxsd.py 生产环境那一节的基础。
    app = mcp.http_app(transport="streamable-http", path=MCP_PATH)

    server = uvicorn.Server(
        uvicorn.Config(app, host=HTTP_HOST, port=HTTP_PORT, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)   # daemon：主线程退出时自动收摊
    thread.start()

    # 等服务真正就绪（最多 10 秒）。直接 sleep 固定秒数是不可靠的写法。
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)

    # 起不来最常见的原因就是端口被占用（比如上一个窗口的服务端没关干净）。
    # 这里**不能**直接抛异常：自检失败也要给出人能看懂的中文提示。
    if not server.started:
        print(f"❌ 服务未能启动：{HTTP_HOST}:{HTTP_PORT} 可能已被占用。")
        print("   请换端口，或先关掉占用该端口的程序。")
        return

    print(f"✅ 服务端已启动：http://{HTTP_HOST}:{HTTP_PORT}{MCP_PATH}（传输：streamable-http）")

    # 局部 import asyncio：只有自检这条路才需要事件循环，常驻服务那几条路用不上
    import asyncio

    async def _probe() -> None:
        # 课案原文写法就是直接给 URL：Client("http://localhost:8000/mcp")
        # FastMCP 会按 URL 自动选出 streamable-http 传输。
        async with Client(f"http://{HTTP_HOST}:{HTTP_PORT}{MCP_PATH}") as client:
            # 自检的核心就是这两件事：工具注册上了没有（list_tools）、
            # 工具能跑通没有（call_tool）。两件都过，说明服务端真的可用。
            tools = await client.list_tools()
            print(f"   已注册工具 {len(tools)} 个：")
            for t in tools:
                # inputSchema 就是 FastMCP 从类型注解自动生成的 JSON Schema，
                # 模型靠它知道该传什么参数。description 来自函数 docstring 首行。
                params = list((t.inputSchema or {}).get("properties", {}).keys())
                print(f"     - {t.name}({', '.join(params)})：{t.description}")
            # 挑 div 而不是 add：因为 div 有除零分支，能顺带验证「工具返回值取法」这条路
            result = await client.call_tool("div", {"a": 10, "b": 4})
            print(f"   调用 div(10, 4) = {result.content[0].text}")

    asyncio.run(_probe())

    # 收摊：先让 uvicorn 停，再等线程结束。顺序反了会打印到一半日志就断掉
    server.should_exit = True       # 通知 uvicorn 退出
    thread.join(timeout=10)         # 等它真的退完；daemon 线程不等也不会阻塞主线程，但日志会乱序
    print("✅ 服务端已关闭，演示结束。")

    # 自检跑完顺带把「怎么当常驻服务用」再讲一遍：
    # 学员看完输出就知道下一步该敲哪条命令，不用回去翻文件头
    print()
    print("要把本文件当常驻服务用，请选择传输方式：")
    print(f"    uv run Agent/05_mcp/01_服务端_jxsd.py http   → http://{HTTP_HOST}:{HTTP_PORT}{MCP_PATH}")
    print("    uv run Agent/05_mcp/01_服务端_jxsd.py sse    → http://127.0.0.1:8001/sse（已弃用）")
    print("    uv run Agent/05_mcp/01_服务端_jxsd.py stdio  → 交给客户端当子进程拉起")


# ---------- 4. 入口：命令行第一个参数决定用哪种传输 ----------
# 设计意图：不带参数 = 自检（跑完就退，适合学员第一次运行）；
#           带参数   = 常驻服务（适合另开窗口起服务、再跑 02/05/06/07 去连它）。
# 这就是「同一个文件既能当示例又能当服务端」的做法。
if __name__ == "__main__":
    # 不传参数时 mode 为空串，落到最后的 else 走自检分支
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if mode == "http":
        run_http()
    elif mode in ("streamable-http", "streamable_http"):
        # 两种写法都认：课案写作 "streamable-http"，Python 侧习惯下划线，别让学员踩这个坑
        run_streamable_http()
    elif mode == "sse":
        run_sse()
    elif mode == "stdio":
        run_stdio()
    else:
        self_check()

# -*- coding: utf-8 -*-
"""
MCP ② 快速开始 · 客户端（fastmcp.Client）
================================================================
服务端起好了，客户端只需要一个 URL 就能连上去。课案原文只有六行：

    client = Client("http://localhost:8000/mcp")

    async def main():
        async with client:
            tools = await client.list_tools()
            for tool in tools:
                print(tool)
            result = await client.call_tool("add", {"a": 1, "b": 2})
            print(result)

    if __name__ == "__main__":
        asyncio.run(main())

本节把这段代码「跑通 + 补全」，并顺带讲清客户端最容易踩的三个点：

    1. **一切都是异步的**：Client 必须 `async with` 进入上下文，且
       FastMCP 3.x 起**不支持同步用法**。所以入口永远是 `asyncio.run(main())`。
    2. **客户端只认「工具名 + 参数字典」**：`call_tool("add", {"a": 1, "b": 2})`。
       参数名必须和服务端函数签名一致，写错会得到 is_error=True 的结果。
    3. **返回的不是裸值，是 CallToolResult**：
           result.content[0].text  → 文本（MCP 协议层给的就是文本）
           result.data             → FastMCP 帮你按返回类型反序列化后的 Python 值
       课案里直接 print(result) 看到的是完整对象，别被吓到。

想用 stdio（服务端当子进程）时，客户端侧换成：

    from fastmcp.client.transports import StdioTransport
    transport = StdioTransport(command="python", args=["01_服务端_jxsd.py"])

URL 与传输的对应关系（课案 4332-4395「四种协议」）：
    http://host:8000/mcp   → streamable-http（推荐）
    http://host:8001/sse   → sse（已弃用）
    stdio                  → 没有 URL，用 StdioTransport 拉子进程

课案出处：Agent 课案 → MCP协议 → 快速开始 → 客户端
          参考 Agent 课案 → MCP协议 → 四种协议

运行方式（本文件会自己把 01 的服务端拉起来，跑完自动关闭）：
    uv run Agent/05_mcp/02_客户端_jxsd.py

若你已经另开窗口常驻了 01 的服务端，本文件检测到端口被占用就会直接连它，不会报错：
    uv run Agent/05_mcp/01_服务端_jxsd.py http    # 窗口 1
    uv run Agent/05_mcp/02_客户端_jxsd.py         # 窗口 2
"""

import asyncio
from pathlib import Path
import socket
import sys

sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import threading
import time

import uvicorn
from fastmcp import Client

# 与服务端约定的地址，和 01_服务端_jxsd.py 保持一致
SERVER_SCRIPT = Path(__file__).resolve().parent / "01_服务端_jxsd.py"
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8000
MCP_URL = f"http://{HTTP_HOST}:{HTTP_PORT}/mcp"


# ---------- 1. 自检脚手架：让本文件能单独跑起来 ----------
# 这一段（端口探测 + 后台起服务端）在每个客户端文件里都有一份，
# 目的是让学员「只运行一个文件」就能看到完整效果，不用先开两个终端。
def port_in_use(host: str, port: int) -> bool:
    """探测端口是否已被监听：用来判断「外面是不是已经有一个服务端在跑」。"""
    sock = socket.socket()
    sock.settimeout(0.5)         # 短超时：探测不该让脚本卡住
    try:
        sock.connect((host, port))   # 能连上就说明有人监听
        return True
    except OSError:
        return False                 # 连不上不是错误，只是「没人监听」
    finally:
        sock.close()


def start_server_in_thread():
    """把 01 的服务端在本进程的后台线程里拉起来，返回 (uvicorn.Server, Thread) 或 None。

    为什么要自己起服务端？—— 学员「只运行一个文件」就得看到完整效果，
    否则还要开两个终端，体验很差。关键是跑完能干净地关掉，不卡死终端。
    """
    if port_in_use(HTTP_HOST, HTTP_PORT):
        print(f"ℹ️  检测到 {HTTP_HOST}:{HTTP_PORT} 已有服务在运行，直接连它（不再另起服务端）。")
        return None              # None 的语义是「不是我起的」，退出时不能去关它

    # 动态加载 01 的模块：文件名以数字开头，没法 `import 01_服务端_jxsd`，
    # 用 importlib 从文件路径加载是最干净的做法（不会触发它 __main__ 里的自检）。
    import importlib.util

    spec = importlib.util.spec_from_file_location("mcp_server_01", SERVER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mcp_server_01"] = module     # 注册进 sys.modules，dataclass/pydantic 才认得出它
    spec.loader.exec_module(module)

    # 同一个 FastMCP 实例既能 mcp.run()，也能交出 ASGI 应用 —— 后者才能塞进线程里跑
    app = module.mcp.http_app(transport="streamable-http", path="/mcp")
    server = uvicorn.Server(
        uvicorn.Config(app, host=HTTP_HOST, port=HTTP_PORT, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)   # daemon：主线程退出即收摊
    thread.start()

    for _ in range(100):        # 最多等 10 秒
        # 轮询等就绪：uvicorn 把「服务真的开始监听」暴露成 server.started，
        # 比写死 time.sleep(2) 可靠 —— 机器慢时 2 秒不够，快时白等。
        if server.started:
            return server, thread
        time.sleep(0.1)

    print(f"❌ 服务端启动失败：{HTTP_HOST}:{HTTP_PORT} 无法监听。")
    return None


async def demo_http() -> None:
    """① 课案原文写法：直接给 URL，FastMCP 按 URL 自动选传输。"""
    print("=" * 64)
    print("① HTTP 连接（课案原文写法：Client(\"http://localhost:8000/mcp\")）")
    print("=" * 64)

    # async with 进入时会：建连 → initialize 握手 → 协商协议版本与能力；
    # 退出时自动关闭连接。不写 async with 手动 new 出来的 client 是不完整的。
    async with Client(MCP_URL) as client:
        # --- list_tools：拿到服务端所有工具的「定义」（不含执行） ---
        tools = await client.list_tools()
        print(f"可用工具 {len(tools)} 个：")
        for tool in tools:
            params = ", ".join((tool.inputSchema or {}).get("properties", {}).keys())
            print(f"  - {tool.name}({params})：{tool.description}")
            # 注释掉课案原始的 print(tool)：它会打印整个 Tool 对象（含完整 JSON Schema），
            # 信息量太大，上面这行挑重点更好读。想看全貌就把下面这行的注释放开。
            # print(tool)

        # --- call_tool：真正执行一个工具 ---
        # 期望输出：add(1, 2) = 3.0
        result = await client.call_tool("add", {"a": 1, "b": 2})
        print(f"\nadd(1, 2) = {result.content[0].text}")
        print(f"  · result.data（反序列化后的 Python 值）= {result.data!r}")
        print(f"  · result.is_error = {result.is_error}")

        # 服务端抛异常时不会断连，而是回一个 is_error=True 的结果。
        # raise_on_error=False 让客户端把错误当「普通返回值」拿回来，而不是抛 Python 异常。
        # 期望输出：is_error=True，内容=Error calling tool 'div': 除数不能为 0
        # 注意：此时**服务端**会在自己的控制台打一段红色 traceback —— 那是服务端的日志，
        #       属于正常现象，连接没有断，后面 stdio 演示照样能跑。
        print("\n下面这条会故意触发服务端异常（服务端控制台会打红色日志，属正常）：")
        bad = await client.call_tool("div", {"a": 1, "b": 0}, raise_on_error=False)
        print(f"div(1, 0) 的返回：is_error={bad.is_error}，内容={bad.content[0].text}")


async def demo_stdio() -> None:
    """② stdio 连接：客户端用 StdioTransport 把服务端脚本当子进程拉起来。

    和 HTTP 的区别：不需要端口、不需要鉴权、进程生命周期跟着客户端走。
    Claude Desktop 这类桌面客户端用的就是这个模式。
    """
    print()
    print("=" * 64)
    print("② stdio 连接（服务端当子进程，无需端口）")
    print("=" * 64)

    from fastmcp.client.transports import StdioTransport

    # 用当前解释器直接跑脚本，避免依赖 uv 在 PATH 里。
    # ⚠️ 第二个参数 "stdio" 必须带上：不传参数时 01 会走「自检演示」分支去抢 8000 端口，
    #    而 stdio 模式下服务端**绝对不能往 stdout 打印任何东西**——
    #    stdout 是 JSON-RPC 的通道，多打印一行字就会把协议流打断。
    transport = StdioTransport(
        command=sys.executable, args=[str(SERVER_SCRIPT), "stdio"]
    )
    # 把 transport 交给 Client 而不是 URL —— 这是 stdio 与 HTTP 在客户端侧唯一的写法差异
    async with Client(transport) as client:
        tools = await client.list_tools()
        print(f"通过 stdio 拿到工具：{[t.name for t in tools]}")
        result = await client.call_tool("mul", {"a": 8, "b": 2})
        print(f"mul(8, 2) = {result.content[0].text}")


# ---------- 2. 主流程：两种传输各走一遍 ----------
async def main() -> None:
    # 顺序有讲究：先跑 HTTP（顺带验证自己起的服务端可用），再跑 stdio（独立子进程，不受影响）
    await demo_http()
    await demo_stdio()


# ---------- 3. 入口：按需拉起服务端 → 跑两个演示 → 关掉自己起的服务 ----------
if __name__ == "__main__":
    # started 为 None 有两种可能：① 端口已被占用（用了别人的服务端）；② 启动失败。
    # 两种情况下都不该由本文件去关它，所以后面只判 `if started`。
    started = start_server_in_thread()

    try:
        asyncio.run(main())
    finally:
        # 只有「我们自己起的」服务端才需要关；外面已经在跑的那个不能动。
        if started:
            server, thread = started
            server.should_exit = True     # 通知 uvicorn 优雅退出
            thread.join(timeout=10)       # 等线程真正结束，否则最后一行提示可能先于日志打印
            print("\n✅ 本文件启动的服务端已关闭。")

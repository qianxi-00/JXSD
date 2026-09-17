# -*- coding: utf-8 -*-
"""
MCP ⑪ 部署（开发环境 + 生产环境）
================================================================
课案原文：MCP 服务可通过 HTTP 协议部署，支持**直接运行**和 **ASGI 应用**两种方式。

    # | 方式 | 做法 | 适用 |
    # |---|---|---|
    # | 开发环境 | `python server.py` 直接跑（内置 uvicorn） | 快速开发、内部工具 |
    # | 生产环境 | `mcp.http_app()` 造 ASGI 应用 → uvicorn 多进程 | 正式部署、要自定义中间件 |

为什么生产要换成 ASGI 应用？因为直接 `mcp.run()` 相当于「框架帮你起了一个 uvicorn」，
你没法控制它：加不了自己的中间件（CORS、限流、访问日志）、开不了多进程、
也不方便挂到已有的 FastAPI 应用上。`mcp.http_app()` 把 FastMCP 还原成一个
**标准 Starlette/ASGI 应用**，之后所有 ASGI 生态的能力就都能用了。

本文件做三件事：
    ① 打印 server.py（开发环境）与 app.py（生产环境）两份完整代码；
    ② **真的**用 mcp.http_app() 造出 ASGI 应用并短暂起一次，验证它能跑通（不碰 Docker）；
    ③ 把课案那段 Dockerfile 原样保留，并逐行讲清楚每一行在干什么。

课案出处：Agent 课案 → MCP协议 → 部署 → 开发环境 / 生产环境

运行方式（不需要 Docker；会短暂占用 8024 端口，跑完自动关闭）：
    uv run Agent/05_mcp/11_部署_jxsd.py

本机实测结论：
    ① `mcp.http_app(transport="streamable-http")` 的返回类型是
       `fastmcp.server.http.StarletteWithLifespan` —— 一个标准 Starlette/ASGI 应用；
       正因为如此，「uvicorn app:app」才成立，也才挂得上 CORS、限流这类 ASGI 中间件
       （这就是课案「生产环境支持自定义中间件」那句话的技术前提）；
    ② 自检里真起一次服务并连上去：拿到工具 ['process_data']，
       调用返回「已处理: 课案-生产环境示例」—— 证明课案那段 app.py 是可运行的写法，
       不是抄来跑不通的伪代码；
    ③ 本文件**不碰 Docker**：Dockerfile 只作为「课案原文 + 逐行讲解」打印出来，
       真要构建镜像按第 ③ 节的两条命令（docker build / docker run）手工执行即可。
"""

import asyncio
import socket
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import threading

from fastmcp import FastMCP

HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8024          # 生产/开发示例用 8000，本文件自检避开它
MCP_PATH = "/mcp"


# ================================================================
# 开发环境：server.py —— 最简单，直接跑
# ================================================================
# 课案原文（# pip install fastmcp 之后 `python server.py`）：
SERVER_PY = '''# server.py
from fastmcp import FastMCP


mcp = FastMCP("数学工具 🚀")


@mcp.tool
def add(a: float, b: float) -> float:
    """两数相加"""
    return a + b


@mcp.tool
def multiply(a: float, b: float) -> float:
    """两数相乘"""
    return a * b


if __name__ == "__main__":
    mcp.run(transport="streamable-http", host="0.0.0.0", port=8000)
'''

# 运行：
#     python server.py
#     # 服务地址: http://localhost:8000/mcp/
#
# 补充（课案没写但很常用）：把 transport 换成 stdio，就是**本地调试**形态 ——
# 不监听端口，由调试客户端（如 02_客户端_jxsd.py 的 StdioTransport、
# 或者 IDE 里的 MCP 插件）当子进程拉起：
#     mcp.run(transport="stdio")
#
#   # | 形态 | 一行代码 | 调试手段 |
#   # |---|---|---|
#   # | HTTP（开发） | mcp.run(transport="streamable-http", ...) | MCP Inspector 填 URL 连 |
#   # | stdio（本地调试） | mcp.run(transport="stdio") | 客户端/IDE 当子进程拉起 |
#
# host="0.0.0.0" 表示监听所有网卡（容器里必须这么写，
# 写 127.0.0.1 的话容器外一律连不上）。本地单机调试建议写 127.0.0.1 更安全。

# ================================================================
# 生产环境：app.py —— 造 ASGI 应用，交给 uvicorn
# ================================================================
# 课案原文：
APP_PY = '''# app.py
from fastmcp import FastMCP


mcp = FastMCP("生产服务")


@mcp.tool
def process_data(input: str) -> str:
    """处理数据"""
    return f"已处理: {input}"


# 创建 ASGI 应用（默认使用 streamable-http 传输）
app = mcp.http_app(transport="streamable-http")
'''

# 使用 Uvicorn 运行（课案原文）：
#     # 安装
#     pip install uvicorn[standard]
#
#     # 单进程
#     uvicorn app:app --host 0.0.0.0 --port 8000
#
#     # 多进程（生产推荐）
#     uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
#
# 注意 `app:app` 的含义：**模块名 : 变量名**。
# 即「文件 app.py 里那个叫 app 的对象」，uvicorn 会 import 这个模块再取属性。
# 所以 mcp.http_app() 的返回值一定要赋给名为 app 的模块级变量。

# ================================================================
# Dockerfile（课案原文，一字未改）
# ================================================================
DOCKERFILE = '''FROM python:3.13-slim

WORKDIR /app
COPY app.py .
RUN pip install fastmcp uvicorn

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
'''

# ---------- 逐行讲解 ----------
# FROM python:3.13-slim
#     基础镜像。slim 版去掉了编译工具链和文档，体积小很多；
#     代价是「需要编译的包」（如某些带 C 扩展的库）装不上，
#     真遇到就改用 python:3.13（完整版）或多阶段构建。
#     课案选 3.13 是因为 FastMCP / MCP SDK 要求 Python ≥ 3.10，3.13 是当时的稳定版。
#
# WORKDIR /app
#     设置容器内的工作目录，后续 COPY / RUN / CMD 都在这个目录下执行。
#     等价于先 mkdir -p /app 再 cd /app，而且目录不存在会自动建。
#
# COPY app.py .
#     把宿主机的 app.py 复制到容器的 /app/app.py（`.` 指 WORKDIR）。
#     只复制这一个文件 —— 容器里不需要整个项目，服务本身就是单文件。
#
# RUN pip install fastmcp uvicorn
#     在**构建阶段**装依赖，装完的结果会被打进镜像层。
#     放在 COPY 之后是为了让缓存失效范围最小：改代码不会导致重装依赖。
#     生产上更推荐锁版本（pip install "fastmcp==3.4.7"）或 COPY requirements.txt 后 pip install -r。
#
# EXPOSE 8000
#     **声明**容器会监听 8000 端口。注意它只是文档性质的声明，
#     真正对外暴露要靠 `docker run -p 8000:8000`。
#
# CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
#     容器启动时的默认命令，用 exec 数组写法（不是 shell 字符串）——
#     这样 uvicorn 会成为 PID 1，能正确接收 docker stop 发来的 SIGTERM 并优雅退出。
#     --host 0.0.0.0 是**必须**的：写 127.0.0.1 的话端口只在容器内部回环，
#     宿主机的 -p 映射根本连不上（这是 Docker 新手最常见的坑）。
#
# 课案原文的 CMD 是**单进程**的。要开多进程（生产推荐），
# 在 CMD 里加 --workers 即可 —— 注意 MCP 服务端如果用「内存态」保存会话，
# 多进程下每个 worker 各存一份，会互相看不见；此时应改用 stateless_http=True
# 或者把状态放到 Redis 这类外部存储里：
DOCKERFILE_WORKERS = '''CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
'''

# 构建与运行（课案原文）：
#     docker build -t mcp-server .
#     docker run -d -p 8000:8000 mcp-server
#
#   docker build -t mcp-server .   → 用当前目录的 Dockerfile 构建，镜像打标签 mcp-server
#   -d                             → 后台运行
#   -p 8000:8000                   → 宿主机 8000 ←→ 容器 8000
#
# 客户端连接：http://<服务器IP>:8000/mcp（transport 选 streamable-http）

# ================================================================
# 自检：真的把 ASGI 应用造出来跑一次（不涉及 Docker）
# ================================================================
# ---------- 四、自检用的最小服务端 ----------
# 和上面 APP_PY 里的写法完全一致，只是为了在本文件里**真跑一次** mcp.http_app()，
# 证明那套写法可用（全程不涉及 Docker）。
mcp = FastMCP("生产服务")


@mcp.tool
def process_data(input: str) -> str:
    """处理数据"""
    return f"已处理: {input}"


# 端口探测与 01/02 里同理：连不上说明没人监听，不算错误。
def _port_open(host: str, port: int) -> bool:
    sock = socket.socket()
    sock.settimeout(0.5)
    try:
        sock.connect((host, port))
        # 用 7 行代码把「端口探测」这件事讲清楚，比直接抄 01/02 的版本短，适合部署章节的节奏。
        return True
    except OSError:
        return False
    finally:
        sock.close()


# 用 01/02 同一套写法真起一次服务，再用 fastmcp.Client 连上去调一次工具做验证。
async def _probe(url: str) -> None:
    from fastmcp import Client
# process_data 的入参名就叫 input —— 和课案 app.py 保持一致，便于对照。

    async with Client(url) as client:
        tools = await client.list_tools()
        print(f"  可用工具：{[t.name for t in tools]}")
        result = await client.call_tool("process_data", {"input": "课案-生产环境示例"})
        print(f"  process_data 返回：{result.content[0].text}")


if __name__ == "__main__":
    # ①~③ 先把两份源码和 Dockerfile 原文打印出来（课案原文，一字未改）。
    print("=" * 64)
    print("① 开发环境：server.py")
    print("=" * 64)
    print(SERVER_PY)
    print("运行：")
    print("    python server.py")
    print("    # 服务地址: http://localhost:8000/mcp/\n")

    # ② 生产环境：app.py + uvicorn，重点是让学员看清「uvicorn app:app」里的 app:app 是什么。
    print("=" * 64)
    print("② 生产环境：app.py + uvicorn")
    print("=" * 64)
    print(APP_PY)
    print("运行：")
    print("    pip install uvicorn[standard]")
    print("    uvicorn app:app --host 0.0.0.0 --port 8000                  # 单进程")
    print("    uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4      # 多进程（生产推荐）\n")

    # ③ Dockerfile：
    # FROM/WORKDIR/COPY/RUN/EXPOSE/CMD 六行，逐行讲解见上面 # 逐行讲解 那段。
    print("=" * 64)
    print("③ Dockerfile（课案原文）")
    print("=" * 64)
    print(DOCKERFILE)
    print("多进程版本只需改 CMD 一行：")
    print(DOCKERFILE_WORKERS)
    print("构建与运行：")
    print("    docker build -t mcp-server .")
    print("    docker run -d -p 8000:8000 mcp-server\n")

    # ---------- 真跑一次 ASGI 应用，证明 app.py 那套写法可用 ----------
    # ④ 自检：端口被占就跳过（不影响上面的部署说明），否则真起一次、连一次、关掉。
    print("=" * 64)
    print(f"④ 自检：把 mcp.http_app() 造出来的 ASGI 应用真起一次（{HTTP_HOST}:{HTTP_PORT}）")
    print("=" * 64)
# 这一段和 APP_PY 里的代码逐行对应，是「课案原文真的能跑」的证据。

    if _port_open(HTTP_HOST, HTTP_PORT):
        print(f"ℹ️  {HTTP_HOST}:{HTTP_PORT} 已被占用，跳过自检（不影响上面的部署说明）。")
        sys.exit(0)

    import uvicorn

    # app.py 里 `app = mcp.http_app(transport="streamable-http")` 就是这个对象
    app = mcp.http_app(transport="streamable-http")
    print(f"  app 类型：{type(app).__module__}.{type(app).__name__}   ← 标准 ASGI 应用")
    print(f"  等价启动命令：uvicorn <模块>:app --host 0.0.0.0 --port {HTTP_PORT}")

    config = uvicorn.Config(app, host=HTTP_HOST, port=HTTP_PORT, log_level="warning")
    # uvicorn.Server + 后台线程 + 轮询 started，与 01/02 完全同一套路。
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # 轮询等就绪：uvicorn 真正开始监听后会把 server.started 置位，比写死 sleep 可靠。
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)

    # 起来了才验证；没起来就直说「跳过自检」，不制造假成功。
    if server.started:
        asyncio.run(_probe(f"http://{HTTP_HOST}:{HTTP_PORT}{MCP_PATH}"))
        server.should_exit = True
        thread.join(timeout=10)
        print("  ✅ ASGI 应用验证通过并已关闭（全程未使用 Docker）。")
    # 注意这一支也要显式报错：端口被占不是「验证通过」，不能让学员误判。
    else:
        print("  ❌ 端口无法监听，跳过自检。")

    print()
    print("小结：开发用 `python server.py` 一行搞定；生产用 app.py + uvicorn 多进程 + Docker。")

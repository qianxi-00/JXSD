# -*- coding: utf-8 -*-
"""
MCP ⑩ 权限 · 客户端（携带 Bearer Token 调用）
================================================================
课案原文只有这几行：

    from fastmcp.client.auth import BearerAuth

    TOKEN = "eyJhbGciOiJIUzI1NiIs..."   # 从认证服务获取的 JWT

    async def main():
        client = Client("http://localhost:8000/mcp", auth=BearerAuth(TOKEN))
        async with client:
            tools = await client.list_tools()
            ...

本节把 TOKEN 那一行的「从认证服务获取」真正补出来，跑通完整链路：

    ┌─────────────┐  ①POST /login（用户名+口令）   ┌──────────────┐
    │   客户端     │ ─────────────────────────────→ │  认证服务(08) │
    │             │ ←───────────────────────────── │   8022       │
    │             │  ②返回 JWT（sub/scope/exp）     └──────────────┘
    │             │
    │             │  ③Authorization: Bearer <JWT>  ┌──────────────┐
    │             │ ─────────────────────────────→ │ MCP 服务端(09)│
    │             │ ←───────────────────────────── │   8023       │
    └─────────────┘  ④验签通过，返回工具结果         └──────────────┘

注意 ③ 那一步：客户端**只是把令牌带上**，它无法自己定义权限 ——
scope 是认证服务签进 payload 的，客户端改一个字符签名就对不上。
这就是课案强调的「客户端无权自行定义权限」。

课案「权限」一节的总纲，先把三个角色记住，再看本文件就好懂了：

    # | 角色 | 职责 |
    # |---|---|---|
    # | 认证服务（外部系统，本套文件是 08） | 签发令牌，定义用户身份（sub） |
    # | 服务端（本套文件是 09） | 验证令牌签名 → 提取声明 → 决定是否放行 |
    # | 客户端（本文件） | 持有令牌，请求时放在 Authorization 头中，无权自行定义权限 |

为什么客户端**必须**无权定义权限？因为客户端是不可信的一端：
只要令牌能在客户端侧被「加工」，任何调用方都能给自己加上 write 权限。
所以整套设计是：**签发权在 08，校验权在 09，客户端只有「拿」和「带」两件事**。

`required_scopes` 干什么（本文件三个反例就是围着它转的）：
    09 里写的是 `JWTVerifier(required_scopes=["read"])`，意思是
    「所有调用至少要带 read 权限」。判断依据是令牌 payload 里的 `scope` 声明
    （**空格分隔的字符串**，不是数组）。
    于是 09 会做两道检查，顺序固定：
        1) 先验签 + 校验时间（exp 是否过期、iat 是否在未来）→ 不过就 401；
        2) 再看 scope 是否覆盖 required_scopes           → 不够就 403。
    本文件的反例二撞的是第 2 道，反例三撞的是第 1 道。

课案出处：Agent 课案 → MCP协议 → 权限 → 客户端

运行方式（本文件会自己把 08 认证服务和 09 MCP 服务端都起起来，跑完全部关闭）：
    uv run Agent/05_mcp/10_权限_客户端_jxsd.py

本机实测结论（跑通后应当看到的正是这四条）：
    ① 合法令牌（alice / pass123，scope="read execute"）→ 通过，能列出 add / get_weather；
    ② 口令错误 → 08 返回 access_token=None，客户端拿不到令牌，等同匿名，被 401 拦下；
    ③ 令牌合法但 scope 为空（bob）→ 验签能过，但过不了 required_scopes=["read"]，被拒；
    ④ 用**正确密钥**签一个 exp 在 60 秒前的令牌 → 签名没问题，仍然 401。
       这一条最能说明「服务端一行鉴权代码都没写，时间校验是 JWTVerifier 自动做的」。
"""

import asyncio
import importlib.util
from pathlib import Path
import socket
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import threading

# 本文件要同时对接两个服务，所以地址分成两组常量，且都必须和 08 / 09 保持一致：
# 改端口时三处一起改（08 的 AUTH_PORT、09 的 MCP_PORT、这里的两个常量），否则会连不上。
AUTH_HOST = "127.0.0.1"
AUTH_PORT = 8022               # 与 08_权限_认证服务_jxsd.py 一致
MCP_HOST = "127.0.0.1"
MCP_PORT = 8023                # 与 09_权限_服务端_jxsd.py 一致
MCP_PATH = "/mcp"
AUTH_URL = f"http://{AUTH_HOST}:{AUTH_PORT}"
MCP_URL = f"http://{MCP_HOST}:{MCP_PORT}{MCP_PATH}"


# ---------- 1. 加载兄弟模块：文件名以数字开头，没法直接 import ----------
def load_sibling(filename: str, module_name: str):
    """按文件名加载同目录模块（文件名以数字开头，无法直接 import）。

    sys.modules 判断同时干两件事：① 缓存（08 被 09 和本文件都要用，只该真执行一次）；
    ② 让 pydantic / dataclass 在解析类型时能在 sys.modules 里找到这个模块。
    """
    if module_name in sys.modules:
        return sys.modules[module_name]

    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module      # 先注册再 exec，否则模块内的自引用会找不到自己
    spec.loader.exec_module(module)
    return module


# ---------- 2. 端口探测：判断 08 / 09 是不是已经在别的窗口跑着 ----------
def port_open(host: str, port: int) -> bool:
    """能连上就说明端口已被监听。用短超时，别让探测本身把脚本拖慢。"""
    # 短超时探测：只是为了判断 08 / 09 是不是已经在外面跑着。
    sock = socket.socket()
    sock.settimeout(0.5)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


# ---------- 3. 通用后台启动器：08 认证服务与 09 MCP 服务端共用一份 ----------
def start_uvicorn_in_thread(app, host: str, port: int):
    """通用：把任意 ASGI 应用跑在后台线程，返回 (server, thread)；已占用则返回 None。

    用 uvicorn.Server（而不是 uvicorn.run）是为了拿到 server.started 与 should_exit 两个开关：
    started 用来「等就绪再往下走」，should_exit 用来「跑完干净地关掉」。
    直接写 time.sleep(2) 等启动是不可靠的——机器慢的时候 2 秒不够，快的时候白等。
    """
    if port_open(host, port):
        print(f"ℹ️  {host}:{port} 已有服务在运行，直接连它。")
        return None                        # None = 「不是我起的」，调用方据此决定要不要关它

    import uvicorn

    server = uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)   # daemon：主线程退出即收摊
    thread.start()
    for _ in range(100):                   # 最多轮询 10 秒（100 × 0.1s）
        if server.started:
            return server, thread
        time.sleep(0.1)
    print(f"❌ {host}:{port} 启动失败。")
    return None


# ---------- 4. ① 登录认证服务，换回 JWT（课案 TOKEN = "eyJ..." 的真实来源） ----------
def login(username: str, password: str) -> str | None:
    """① 向认证服务登录，换回 JWT（课案里 TOKEN = "eyJ..." 那一行的真实来源）。

    用 requests 而不是 httpx，是为了贴近课案里 curl 的写法，读起来最直观。
    """
    import requests

    # 注意 08 的 /login 走的是「宽松错误」写法：口令错也返回 HTTP 200，
    # 只是 access_token 为 None。所以判断成功与否必须看字段，不能看状态码。
    # 08 的 /login 走「宽松错误」写法：口令错也返回 200，只是 access_token 为 None，
    # 所以成功与否必须看字段，不能看 HTTP 状态码。
    resp = requests.post(
        f"{AUTH_URL}/login",
        json={"username": username, "password": password},
        timeout=10,
    )
    data = resp.json()
    if not data.get("access_token"):
        print(f"   登录失败（{username}）：{data.get('message')}")
        return None
    return data["access_token"]


# ---------- 5. ③④ 带令牌调用 MCP 服务端的统一入口（正例与三个反例都走它） ----------
async def try_connect(label: str, token: str | None) -> None:
    """用给定令牌连一次 MCP 服务端，打印成功/被拒的结论。

    把「连一次」抽成函数，是为了让正例和三个反例的代码只有 token 一处不同 ——
    教学上更容易看出「变量只有令牌，结论却完全不同」。
    """
    # 延迟 import：这几个类只有真正连服务端时才需要，放模块顶部会拖慢纯打印流程
    from fastmcp import Client
    from fastmcp.client.auth import BearerAuth
    from fastmcp.client.transports import StreamableHttpTransport

    if token is None:
        # 没有令牌就不带 auth —— 等价于「匿名调用」，应当被中间件拦下
        transport = StreamableHttpTransport(url=MCP_URL)
    else:
        # BearerAuth 会把令牌塞进每个请求的 Authorization 头：
        #     Authorization: Bearer eyJhbGciOi...
        transport = StreamableHttpTransport(url=MCP_URL, auth=BearerAuth(token))

    try:
        # 注意 401 是在 async with 进入（initialize 握手）时就发生的，
        # 不等你调 list_tools —— 这就是「根本进不到工具函数」的含义
        async with Client(transport) as client:
            tools = await client.list_tools()
            print(f"   ✅ {label}：通过，可用工具 {[t.name for t in tools]}")
            result = await client.call_tool("add", {"a": 3, "b": 5})
            print(f"      add(3, 5) = {result.content[0].text}")
    except Exception as exc:
        # 只打异常类型 + 第一行：MCP 客户端的异常里常带一大段 HTML/JSON，
        # 全打出来会盖住后面几条反例的结论
        print(f"   ⛔ {label}：被拒（{type(exc).__name__}: {str(exc).splitlines()[0][:90]}）")


# ---------- 6. 主流程：一个正例 + 三个反例，把「谁说了算」演示清楚 ----------
async def main() -> None:
    # 早于 09 加载 08：09 内部也要 import 08，先加载能让两边拿到同一个模块实例
    auth_service = load_sibling("08_权限_认证服务_jxsd.py", "mcp_auth_service_08")

    # ---------- ② 登录拿令牌 ----------
    print("=" * 64)
    print("② 向认证服务登录，拿 JWT")
    print("=" * 64)
    token = login("alice", "pass123")
    if not token:
        print("❌ 拿不到令牌，后续演示无法继续。")
        return
    # 只打前 40 个字符：令牌是敏感凭据，整条打出来等于把凭据写进日志
    print(f"   拿到 JWT：{token[:40]}...（共 {len(token)} 字符）")
    # 顺手解一下 payload 给学员看：JWT 的 payload 只是 Base64，谁都能解开读
    print(f"   payload：{auth_service.verify_token(token)}")

    # ---------- ③④ 带令牌调用 ----------
    print()
    print("=" * 64)
    print("③④ 携带 Bearer Token 调用 MCP 服务端")
    print("=" * 64)
    await try_connect("合法令牌（alice, scope='read execute'）", token)

    # ---------- 反例一：错误口令 → 拿不到令牌 ----------
    # 想说明的是：拿不到令牌 ≠ 服务端拦你，而是**认证服务这一关就签不出来**。
    # 两道关口的分工，正是「签发权 / 校验权分离」的落地。
    print()
    print("=" * 64)
    print("反例一：口令错误，认证服务不签发令牌")
    print("=" * 64)
    bad_token = login("alice", "wrong-password")
    await try_connect("匿名/无令牌", bad_token)

    # ---------- 反例二：scope 不足 → 被 required_scopes 拦下 ----------
    print()
    print("=" * 64)
    print("反例二：令牌合法但没有 read 权限（required_scopes=['read']）")
    print("=" * 64)
    # 直接调认证服务的签发函数，故意给一个空 scope。
    # 客户端**无法**自己改令牌里的 scope —— 改了签名就对不上，所以只能从签发端控。
    no_scope_token = auth_service.create_token("bob", scopes="")
    await try_connect("无 scope 令牌", no_scope_token)

    # ---------- 反例三：令牌过期 → exp 校验直接拒绝 ----------
    print()
    print("=" * 64)
    print("反例三：令牌已过期（exp 在过去）")
    print("=" * 64)
    import jwt as pyjwt

    # 用**正确的密钥**签一个已经过期的令牌：签名能过，时间过不了。
    # 这是本文件最关键的一条反例：证明 exp 校验不是「业务代码写的」，而是 JWTVerifier 自动做的。
    expired = pyjwt.encode(
        {
            "sub": "alice",
            "scope": "read execute",
            "exp": int(time.time()) - 60,      # 60 秒前就过期了
            "iat": int(time.time()) - 3600,    # 签发时间放在 1 小时前，时间顺序自洽
        # exp 在 60 秒前、iat 在 1 小时前：时间先后自洽，唯一的问题就是「已经过期」。
        },
        auth_service.SECRET_KEY,
        algorithm=auth_service.ALGORITHM,
    )
    await try_connect("已过期令牌", expired)
    print("   ↑ 课案那句话的实证：JWTVerifier 底层用 pyjwt 自动校验 exp，服务端无需写代码。")


# ---------- 7. 启动顺序：先 08（发令牌的）再 09（验令牌的），退出时反序关闭 ----------
if __name__ == "__main__":
    auth_module = load_sibling("08_权限_认证服务_jxsd.py", "mcp_auth_service_08")
    server_module = load_sibling("09_权限_服务端_jxsd.py", "mcp_protected_server_09")

    # 先把认证服务起在 8022，再把 MCP 服务端起在 8023；
    # 顺序不能反：本文件用的密钥来自 08，09 也是从 08 取同一个密钥，08 必须先可用。
    # 两个都用 uvicorn.Server + 后台线程，跑完用 should_exit 干净地关掉。
    # 顺序不能反：09 的密钥来自 08，08 必须先起来；关闭时则反序（先关 09 再关 08）。
    auth_started = start_uvicorn_in_thread(auth_module.app, AUTH_HOST, AUTH_PORT)
    mcp_app = server_module.mcp.http_app(transport="streamable-http", path=MCP_PATH)
    mcp_started = start_uvicorn_in_thread(mcp_app, MCP_HOST, MCP_PORT)

    print(f"认证服务：{AUTH_URL}      MCP 服务端：{MCP_URL}\n")

    try:
        asyncio.run(main())
    finally:
        # 反序关闭：先关 09（对外服务）再关 08（依赖它的认证服务），避免关闭窗口期内 401 噪音
        for started, name in ((mcp_started, "MCP 服务端"), (auth_started, "认证服务")):
            if started:                     # None 表示那是外面已经在跑的服务，绝不能动它
                server, thread = started
                server.should_exit = True
                thread.join(timeout=10)
                print(f"✅ {name}已关闭。")

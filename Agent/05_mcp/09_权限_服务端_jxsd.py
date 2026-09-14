# -*- coding: utf-8 -*-
"""
MCP ⑨ 权限 · 服务端（JWTVerifier 校验 Bearer Token）
================================================================
课案原文：适用于内部微服务，认证服务和 MCP 服务器共享同一密钥。

    verifier = JWTVerifier(
        public_key="your-shared-secret-key-minimum-32-chars",  # 共享密钥
        algorithm="HS256"                                      # HMAC 签名算法
    )
    mcp = FastMCP(name="Protected API", auth=verifier)

就这两段。`auth=` 参数就是课案说的「校验 Bearer token 的中间件」：
FastMCP 会自动往 ASGI 应用上挂一层认证中间件，
客户端请求头里没有合法的 `Authorization: Bearer <token>` 就直接 401，
**根本进不到工具函数**——所以业务代码里一行鉴权都不用写。

JWTVerifier 底层使用 pyjwt 库，**自动校验时间**（课案原文）：

    # | 声明 | 校验规则 | 结果 |
    # |---|---|---|
    # | exp | 过期 → 令牌拒绝 | 401 |
    # | iat | 在未来 → 令牌拒绝（防时钟偏移攻击） | 401 |
    # | 签名 | 与共享密钥不匹配 | 401 |

服务端无需额外代码处理。

细粒度权限：`required_scopes=["read"]` 表示「所有调用至少要带 read 权限」。
scope 来自令牌 payload 里的 `scope` 声明（空格分隔的字符串）。
想做到「某个工具要 write 才能调」，两种做法：
    1) 再起一个 FastMCP 实例、换一个 required_scopes 不同的 verifier，挂不同工具；
    2) 在工具函数内部读请求上下文里的 token 信息自行判断。

课案出处：Agent 课案 → MCP协议 → 权限 → 服务端

运行方式（本文件自己起服务端，并用 08 签发的令牌自检一次，跑完自动关闭）：
    uv run Agent/05_mcp/09_权限_服务端_jxsd.py
"""

import asyncio
import importlib.util
from pathlib import Path
import socket
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import threading

from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import JWTVerifier

# 本文件监听 8023（认证服务在 8022）
MCP_HOST = "127.0.0.1"
MCP_PORT = 8023
MCP_PATH = "/mcp"


def load_sibling(filename: str, module_name: str):
    """从同目录按文件名加载模块。

    本目录下所有文件名都以数字开头（`08_权限_认证服务_jxsd.py`），
    不是合法的 Python 标识符，没法直接 `import`，
    所以用 importlib 从文件路径加载 —— 这是处理这类文件名的标准做法。

    注册进 sys.modules 很重要：pydantic / dataclass 在解析类型时
    会回头去 sys.modules 找模块，不注册会在某些场景下报奇怪的错。
    同时 sys.modules 也充当缓存：10_权限_客户端_jxsd.py 里会先加载本文件、
    本文件再加载 08，加了这层判断就只会真正执行一次，不会拿到两份实例。
    """
    # 先查 sys.modules：10 → 09 → 08 这条加载链上谁都可能先把 08 拉起来，
    # 有这层缓存才能保证「只有一个认证服务模块实例」。
    if module_name in sys.modules:
        return sys.modules[module_name]

    path = Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    # 路径拼接用 Path(__file__).resolve().parent，保证从任何工作目录运行都能找到同目录文件。
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# 认证服务模块：这里只为拿它的 SECRET_KEY。
# 「认证服务与 MCP 服务端共享同一密钥」这句话，落到代码上就是这一行 import。
auth_service = load_sibling("08_权限_认证服务_jxsd.py", "mcp_auth_service_08")
SECRET_KEY = auth_service.SECRET_KEY

# ================================================================
# 创建校验器（课案的「中间件」）
# ================================================================
# 演示用对称密钥（HS256）：public_key 这个参数名看着别扭 ——
# 对称算法没有公私钥之分，这里传的就是那把共享密钥，课案原文也是这么写的。
#
# 生产环境建议换成 RS256：
#     JWTVerifier(public_key=open("public.pem").read(), algorithm="RS256")
# 或让服务端自动去认证服务拉公钥（JWKS）：
#     JWTVerifier(jwks_uri="https://auth.example.com/.well-known/jwks.json")
# 好处是服务端只拿得到**公钥**，即使被拖库也无法伪造令牌。
verifier = JWTVerifier(
    public_key=SECRET_KEY,
    algorithm="HS256",
    required_scopes=["read"],       # 所有调用至少要有 read 权限
)

# name 会显示在客户端/Inspector 里；auth=verifier 就是挂上认证中间件
mcp = FastMCP(name="Protected API", auth=verifier)


@mcp.tool
def add(a: float, b: float) -> float:
    """两数相加"""
    # 注意这里没有任何鉴权代码 —— 请求能走到这一行，就说明令牌已经验过了
    return a + b


@mcp.tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气"""
    weather_map = {"上海": "晴 25 度", "北京": "多云 18 度"}
    return weather_map.get(city, f"{city} 天气未知")


def start_server_in_thread():
    """后台线程起带认证的 MCP 服务端；端口被占则直接连已有的。"""
    # 起服务前先探测端口：外面已经有带认证的服务在跑就直接连它。
    sock = socket.socket()
    sock.settimeout(0.5)
    try:
        sock.connect((MCP_HOST, MCP_PORT))
        print(f"ℹ️  {MCP_HOST}:{MCP_PORT} 已有服务在运行，直接连它。")
        return None
    # config 里 log_level="warning" 是为了压掉 uvicorn 的访问日志，让教学输出干净。
    except OSError:
        pass
    finally:
        sock.close()

    import uvicorn

    # 带认证的服务**只能以 HTTP 运行**：stdio 模式没有 HTTP 请求头，
    # 塞不进 Authorization，中间件拿不到令牌，所有调用都会 401。
    # 交给 uvicorn 跑，是为了拿到 started / should_exit 两个开关（等就绪 + 干净关闭）。
    app = mcp.http_app(transport="streamable-http", path=MCP_PATH)
    server = uvicorn.Server(
        uvicorn.Config(app, host=MCP_HOST, port=MCP_PORT, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # 起不来就返回 None；此时调用方仍会尝试连接，外面可能本来就有服务。
    for _ in range(100):
        if server.started:
            return server, thread
        time.sleep(0.1)
    print(f"❌ MCP 服务端启动失败：{MCP_HOST}:{MCP_PORT} 无法监听。")
    return None


async def self_check(url: str) -> None:
    """三种情况各走一遍：无令牌 / 合法令牌 / 伪造令牌。"""
    from fastmcp import Client
    from fastmcp.client.auth import BearerAuth
    from fastmcp.client.transports import StreamableHttpTransport

    # --- ① 不带令牌：应在握手阶段就被 401 拦下 ---
    # 反例一：不带 Authorization 头。FastMCP 挂的认证中间件会在握手阶段就 401，
    # 请求根本进不到工具函数 —— 这也是「业务代码里一行鉴权都不用写」的原因。
    print("① 不带令牌连接：")
    try:
        async with Client(url) as client:
            await client.list_tools()
        print("   ❌ 竟然通过了 —— 认证中间件没生效！")
    except Exception as exc:
        print(f"   ✅ 已被拒绝：{type(exc).__name__}（HTTP 401 Unauthorized）")

    # --- ② 用 08 认证服务签发的合法令牌 ---
    # 正例：令牌由 08 用同一把密钥签发，服务端验签必然通过。
    print("\n② 用认证服务（08）签发的合法令牌连接：")
    token = auth_service.create_token("alice")
    transport = StreamableHttpTransport(url=url, auth=BearerAuth(token))
    async with Client(transport) as client:
        tools = await client.list_tools()
        print(f"   ✅ 通过，可用工具：{[t.name for t in tools]}")
        result = await client.call_tool("add", {"a": 3, "b": 5})
        print(f"   add(3, 5) = {result.content[0].text}")

    # --- ③ 用别的密钥伪造一个令牌：签名对不上，必须拒绝 ---
    # 反例二：令牌结构完全合法（sub/scope/exp/iat 齐全），
    # 但签名用的是攻击者猜的密钥 —— 验签失败，服务端必须拒绝。
    print("\n③ 用错误的密钥伪造令牌连接：")
    import jwt as pyjwt

    forged = pyjwt.encode(
        {
            "sub": "hacker",
            "scope": "read execute",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        # 把拒绝路径也写成断言：如果伪造令牌通过了，说明密钥配置有问题，必须显式报出来。
        },
        # 把伪造令牌的 payload 也写全，是为了证明「拒绝的原因是签名，不是字段缺失」。
        "wrong-secret-key-that-attacker-guessed",
        algorithm="HS256",
    )
    try:
        transport = StreamableHttpTransport(url=url, auth=BearerAuth(forged))
        async with Client(transport) as client:
            await client.list_tools()
        print("   ❌ 伪造令牌竟然通过了 —— 密钥配置有问题！")
    except Exception as exc:
        print(f"   ✅ 已被拒绝：{type(exc).__name__}（签名校验失败）")


if __name__ == "__main__":
    # 服务端起来后显式把 required_scopes 打出来，方便对照下面三种情况的结论。
    started = start_server_in_thread()
    url = f"http://{MCP_HOST}:{MCP_PORT}{MCP_PATH}"
    print(f"带认证的 MCP 服务端：{url}（required_scopes=['read']）\n")

    try:
        asyncio.run(self_check(url))
    finally:
        if started:
            server, thread = started
            # 退出时反序关闭并打印结果，方便学员确认端口确实被释放了。
            server.should_exit = True
            thread.join(timeout=10)
            print("\n✅ 本文件启动的 MCP 服务端已关闭。")

    print()
    print("配套：10_权限_客户端_jxsd.py 演示「登录 → 拿令牌 → 调用」的完整链路。")

# -*- coding: utf-8 -*-
"""
MCP ⑧ 权限 · 认证服务（独立签发 JWT）
================================================================
课案原文：服务端通过 JWT 令牌**验证**客户端的身份和权限，
令牌由外部认证服务颁发，客户端不能自行定义权限。

核心原则（课案原表）：

    # | 角色 | 职责 |
    # |---|---|---|
    # | 认证服务（外部系统） | 签发令牌，定义用户身份（sub） |
    # | 服务端 | 验证令牌签名 → 提取声明 → 决定是否放行 |
    # | 客户端 | 持有令牌，请求时放在 Authorization 头中，无权自行定义权限 |

为什么必须拆成三个角色？
    如果 MCP 服务端自己签发令牌，那么「谁能用哪个工具」就由服务端自己说了算，
    客户端只要连上服务端就能自己给自己发一张通行证 —— 权限形同虚设。
    拆开之后：**签发权在认证服务，校验权在服务端，持有权在客户端**，三者互相制衡。

JWT 结构：header.payload.signature，用 . 分隔的三段 Base64URL
    header    {"alg": "HS256", "typ": "JWT"}
    payload   {"sub": 用户, "exp": 过期时间, "iat": 签发时间, ...}
    signature HMAC-SHA256(header.payload, SECRET_KEY)

注意 payload 只是 Base64 **编码**，不是加密 —— 任何人都能解开看到内容。
所以 JWT 里绝不能放口令、手机号这类敏感数据；它的安全性来自「改一个字签名就对不上」。

课案原文的依赖安装命令（本项目 venv 已装好，无需再装）：
    # pip install fastapi uvicorn pyjwt

课案出处：Agent 课案 → MCP协议 → 权限 → 认证服务

运行方式（本文件自己把认证服务起在 8022，用 requests 真登录一次，跑完自动关闭）：
    uv run Agent/05_mcp/08_权限_认证服务_jxsd.py

配套文件：
    服务端（校验令牌）：09_权限_服务端_jxsd.py
    客户端（携带令牌）：10_权限_客户端_jxsd.py
"""

import datetime
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import jwt
from fastapi import FastAPI
from pydantic import BaseModel


# ================================================================
# 一、共享密钥
# ================================================================
# 课案原文：
#     SECRET_KEY = "your-shared-secret-key-minimum-32-chars"  # 共享密钥
#
# ⚠️ 绝不能把密钥硬编码进代码 —— 提交进仓库就等于把签发权公开了。
#    正确做法：从环境变量读，本地开发用一份演示默认值兜底。
#    生产环境请在部署平台（K8s Secret / Docker Secret / 云厂商密钥管理）里配置：
#        $env:MCP_JWT_SECRET = "<足够长的随机串>"        # Windows PowerShell
#        export MCP_JWT_SECRET="<足够长的随机串>"        # Linux
#    生成随机串： python -c "import secrets; print(secrets.token_urlsafe(48))"
#
# ⚠️ 认证服务（本文件）和 MCP 服务端（09）**必须用同一个密钥**，
#    否则服务端验签必然失败 —— 这就是课案说的「内部微服务共享同一密钥」。
SECRET_KEY_ENV = "MCP_JWT_SECRET"
SECRET_KEY = os.environ.get(SECRET_KEY_ENV) or "demo-shared-secret-key-minimum-32-chars"
SECRET_KEY_FROM_ENV = bool(os.environ.get(SECRET_KEY_ENV))

# HS256 = HMAC-SHA256，对称签名：签发和校验用同一个密钥。
# 若要做成「认证服务持私钥签发、服务端只用公钥校验」，换成 RS256 即可
# （JWTVerifier 支持 public_key 传公钥，或 jwks_uri 自动拉取 JWKS）。
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 1     # 课案：过期时间 1 小时后令牌失效

app = FastAPI(title="MCP 认证服务")


# ================================================================
# 二、模拟用户数据库
# ================================================================
# 课案原文：
#     USERS = {
#         "alice": {"password": "pass123"},
#         "bob":   {"password": "pass456"},
#     }
#
# 再一次强调：真实项目里**永远不要明文存口令**，要存 bcrypt/argon2 的哈希值。
# 这里保留明文只为让课案示例能一眼看懂，口令本身也是明显的演示用弱口令。
USERS = {
    "alice": {"password": "pass123"},
    "bob":   {"password": "pass456"},
}


class LoginRequest(BaseModel):
    """登录请求体：FastAPI 会自动把 JSON 反序列化成这个模型并做类型校验"""
    username: str
    password: str


# ================================================================
# 三、签发 JWT
# ================================================================
def create_token(user_id: str, scopes: str = "read execute") -> str:
    """签发 JWT 令牌

    课案原文的 payload 只有 sub / exp / iat 三项：

        payload = {
            "sub": user_id,
            "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=1),  # 过期时间
            "iat": datetime.datetime.utcnow(),                                # 签发时间
        }

    这里补一个 scope 字段，原因是：09_权限_服务端_jxsd.py 里
    `JWTVerifier(required_scopes=["read"])` 是靠 payload 里的 **scope 声明**做权限判断的。
    没有 scope，服务端只会校验「签名对不对」，无法区分 read / write 权限。
    MCP 官方规范里 scope 是**空格分隔的字符串**（不是数组），这也是最容易写错的一处。

    ⚠️ 另一个坑：课案用的 `datetime.datetime.utcnow()` 在 Python 3.12+ 已标记废弃
       （它返回的是「不带时区的 UTC 时间」，容易和本地时间混用出错）。
       现代写法是 `datetime.datetime.now(datetime.timezone.utc)`，下面用的是后者。
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "sub": user_id,                                          # subject：用户唯一标识
        "scope": scopes,                                         # 权限声明（空格分隔）
        "exp": now + datetime.timedelta(hours=TOKEN_EXPIRE_HOURS),  # 过期时间
        "iat": now,                                              # 签发时间
    }
    # jwt.encode 内部会做 Base64URL + HMAC 签名，返回 "xxx.yyy.zzz" 形式的字符串
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(token: str) -> dict:
    """校验 JWT：验签 + 校验 exp/iat，返回 payload；不合法会抛 jwt 异常。

    服务端（09）用的是 FastMCP 的 JWTVerifier，底层同样是 pyjwt；
    这个函数只是方便你在本文件里手工验证一下签发结果。
    """
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])


# ================================================================
# 四、登录接口
# ================================================================
@app.post("/login")
def login(req: LoginRequest):
    """用户登录 → 认证服务查库 → 签发令牌

    课案原文这里 import 了 HTTPException 但实际走的是「返回错误体」的宽松写法：
        if not user or user["password"] != req.password:
            return {"access_token": None, "token_type": None, "message": "用户名或密码错误"}
    这里保持课案写法（客户端拿到 200 + access_token=None，自己判断）。
    """
    # 课案的宽松写法：口令错也返回 HTTP 200，只在 body 里给 access_token=None；
    # 严格做法是 raise HTTPException(401)，但那样客户端还要处理异常分支，示例更难读。
    user = USERS.get(req.username)
    if not user or user["password"] != req.password:
        return {"access_token": None, "token_type": None, "message": "用户名或密码错误"}

    token = create_token(req.username)
    return {"access_token": token, "token_type": "bearer", "message": "登录成功"}


# ================================================================
# 五、把服务真的起起来，并用 requests 走一遍登录
# ================================================================
AUTH_HOST = "127.0.0.1"
AUTH_PORT = 8022          # 课案用的是 9000；这里避开常用端口，减少和本机其他服务撞车的概率
AUTH_URL = f"http://{AUTH_HOST}:{AUTH_PORT}"


def start_auth_service_in_thread(port: int = AUTH_PORT):
    """后台线程起 uvicorn，返回 (server, thread)；端口被占则返回 None（表示已有服务）。"""
    # socket 只用来「问一句端口有没有人听」，用完立刻关闭。
    import socket

    sock = socket.socket()
    sock.settimeout(0.5)
    try:
        sock.connect((AUTH_HOST, port))
        print(f"ℹ️  {AUTH_HOST}:{port} 已有服务在运行，直接用它。")
        # HS256 是对称算法：签发与校验用同一把密钥，所以 08 和 09 必须共享它。
        return None
    except OSError:
        pass
    finally:
        sock.close()

    import threading
    import time

    import uvicorn

    # uvicorn.Server 比 uvicorn.run() 更适合「跑在后台线程」的场景：
    # 它把 server.started 和 server.should_exit 暴露出来，
    # 于是我们既能等它就绪，也能干净地让它停下来 —— 不会卡住终端。
    server = uvicorn.Server(
        uvicorn.Config(app, host=AUTH_HOST, port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    for _ in range(100):          # 最多等 10 秒
        # 轮询等就绪；server.started 是 uvicorn 真正开始监听后置位的标志。
        if server.started:
            return server, thread
        time.sleep(0.1)
    print(f"❌ 认证服务启动失败：{AUTH_HOST}:{port} 无法监听。")
    return None


def demo_login_flow() -> None:
    """用 requests 真登录：Alice / Bob / 错误口令 三种情况各走一遍。"""
    import requests

    print(f"密钥来源：{'环境变量 ' + SECRET_KEY_ENV if SECRET_KEY_FROM_ENV else '内置演示默认值（生产请改用环境变量）'}")
    print(f"认证服务地址：{AUTH_URL}")
    print()

    # 课案里的 curl 等价写法：
    #   curl -X POST http://localhost:9000/login \
    #     -H "Content-Type: application/json" \
    #     -d '{"username":"alice","password":"pass123"}'
    # 三种情况各走一遍：两个合法用户 + 一次错误口令，
    # 用来对比「什么时候签发、什么时候不签发」。
    for username, password in [("alice", "pass123"), ("bob", "pass456"), ("alice", "wrong")]:
        resp = requests.post(
            f"{AUTH_URL}/login",
            json={"username": username, "password": password},
            timeout=10,
        )
        data = resp.json()
        print(f"POST /login  {username}/{password}  → HTTP {resp.status_code}")
        print(f"   message      : {data.get('message')}")
        if data.get("access_token"):
            token = data["access_token"]
            # 令牌是敏感凭据：日志里只打前 30 个字符，绝不整条打印
            print(f"   access_token : {token[:30]}...（共 {len(token)} 字符，只展示前缀）")
            print(f"   payload 解出来: {verify_token(token)}")
        else:
            print(f"   access_token : None（登录失败，不签发令牌）")
        print()


if __name__ == "__main__":
    started = start_auth_service_in_thread()
    try:
        demo_login_flow()
    finally:
        if started:
            # 只有「我们自己起的」才需要关；外面已经在跑的那个不能动。
            server, thread = started
            server.should_exit = True
            thread.join(timeout=10)
            print("✅ 认证服务已关闭。")

    # 无参数运行时走「起服务 → 真登录 → 关服务」，跑完自动收摊，不会卡住终端。
    print()
    print("说明：本文件无参数运行时走的是「起服务 → 真登录 → 关服务」的自检流程。")
    print("      09_权限_服务端_jxsd.py 与 10_权限_客户端_jxsd.py 会用 importlib")
    print("      直接加载本模块，自己把认证服务跑起来，**不需要手动常驻**。")
    print("      只有想在浏览器/Postman 里手工调 /login 时，才需要把它单独跑着。")

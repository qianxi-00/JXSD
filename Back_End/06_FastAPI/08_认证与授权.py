"""FastAPI 认证与授权：JWT（HS256）+ 密码哈希 + 角色权限
================================================================
对应课案章节：后端开发基础 → Web框架 → FastAPI（认证与授权）

先分清两个词（面试常问）：
    · **认证 Authentication**：验证"你是谁" —— 登录时校验用户名密码。
    · **授权 Authorization**：验证"你能做什么" —— 这个接口是不是只有管理员能调。
    对应到 HTTP 状态码：401 = 未认证（没带 token / token 无效）；403 = 已认证但无权限。

本节知识点：
    1. 认证的五个步骤：用户登录 → 服务器验密 → 签发 Token → 客户端携带 → 服务器验签
    2. 为什么密码必须哈希存储：数据库泄露时明文密码等于全泄露
       —— 用标准库 hashlib.pbkdf2_hmac + 随机盐（本环境不装 passlib/bcrypt）
    3. JWT 的结构：header.payload.signature，用密钥做 HMAC-SHA256 签名防篡改
    4. OAuth2PasswordBearer：FastAPI 提供的 token 提取器
       —— 它只负责从 `Authorization: Bearer xxx` 里把 token 抠出来，不负责验证
    5. Form 表单登录（OAuth2 密码模式的协议要求），tokenUrl 必须指向登录接口
    6. 依赖注入做鉴权：get_current_user（认证）+ require_role(...)（授权）
    7. Token 过期（exp）与篡改检测，以及为什么 JWT 签发后无法主动撤销
    8. 生产注意：密钥必须来自配置、必须用 HTTPS、敏感操作要二次校验

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\08_认证与授权.py' --check
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\08_认证与授权.py'
    启动后可在 http://127.0.0.1:8108/docs 里点 Authorize 按钮直接试。
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import os
import pathlib
import secrets
import sys
from typing import Annotated

import jwt
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field

HERE = pathlib.Path(__file__).resolve().parent
BACK_END = HERE.parent
ROOT = HERE.parents[1]
DATA_DIR = BACK_END / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PORT = 8108

# ============================================================
# 一、安全配置
# ============================================================
# 生产环境必须从配置中心/环境变量读取，绝不能硬编码在代码里。
# 这里优先读环境变量 JWT_SECRET，没设置就用一个开发默认值（并在启动时提醒）。
JWT_SECRET = os.getenv("JWT_SECRET", "dev-only-secret-change-me-in-production")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

app = FastAPI(
    title="08 认证与授权",
    description="后端开发基础 · FastAPI 示例：JWT 认证 + 角色授权",
    version="1.0.0",
)

# OAuth2PasswordBearer 是"token 提取器"：
#   · tokenUrl="token" 告诉 /docs 页面去哪里换 token（会显示 Authorize 按钮）
#   · 请求时自动读取 Authorization: Bearer <token>
#   · 没有或格式不对 → 自动返回 401
#   · 它**不验证** token 的合法性，验证由 jwt.decode 负责
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# ============================================================
# 二、密码哈希（标准库实现，替代 passlib/bcrypt）
# ============================================================
def hash_password(password: str, salt: bytes | None = None) -> str:
    """用 PBKDF2-HMAC-SHA256 生成密码哈希，返回 "算法$迭代次数$盐$哈希" 格式。

    为什么不能直接存明文 / 直接 md5？
        · 明文：库一泄露，用户密码直接暴露（而且用户到处复用密码）；
        · 单纯 md5/sha256：太快了，攻击者每秒能试上亿次，配合彩虹表秒破。
        · PBKDF2 通过**大量迭代 + 随机盐**把每次尝试的成本抬高几万倍，
          并且同一个密码在不同用户那里哈希结果也不同（盐不同），彩虹表失效。
    """
    if salt is None:
        salt = secrets.token_bytes(16)          # 每个用户一个独立随机盐
    iterations = 120_000                        # 迭代次数：越高越安全，也越慢
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """校验密码：用同样的盐和迭代次数重算，再比对。

    用 hmac.compare_digest 而不是 == 的原因：
        字符串比较是"短路"的，比较耗时会随匹配长度变化，
        攻击者可以用统计方法（时序攻击）逐字节猜出正确哈希。
        compare_digest 是恒定时间比较，堵住这个信道。
    """
    try:
        algorithm, iterations, salt_hex, digest_hex = stored.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        )
        return hmac.compare_digest(digest.hex(), digest_hex)
    except (ValueError, TypeError):
        return False


# ============================================================
# 三、"用户表"（真实项目里应该是数据库表）
# ============================================================
class UserInDB(BaseModel):
    username: str
    hashed_password: str
    role: str = "user"
    disabled: bool = False


def build_user_table() -> dict[str, UserInDB]:
    """初始化内存用户表。

    演示账号（密码都是明文写在注释里，方便教学；真实项目绝不会有这张表）：
        alice / alice123    普通用户
        root  / root123     管理员
        tom   / tom123      被禁用的账号
    """
    return {
        "alice": UserInDB(username="alice", hashed_password=hash_password("alice123"), role="user"),
        "root": UserInDB(username="root", hashed_password=hash_password("root123"), role="admin"),
        "tom": UserInDB(username="tom", hashed_password=hash_password("tom123"),
                        role="user", disabled=True),
    }


USERS: dict[str, UserInDB] = build_user_table()


class Token(BaseModel):
    """OAuth2 标准响应格式，access_token + token_type 是协议固定字段。"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="有效期（秒）")


class UserOut(BaseModel):
    username: str
    role: str


# ============================================================
# 四、签发与校验 Token
# ============================================================
def create_access_token(username: str, role: str, expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    """签发 JWT。

    payload 里放什么？
        sub  主题（这里放用户名）—— JWT 规范字段
        role 自定义字段，用于鉴权
        exp  过期时间（时间戳）—— JWT 规范字段，PyJWT 会自动校验
        iat  签发时间
    注意：**payload 只是 Base64 编码，不是加密**！任何人都能解开看内容，
         所以绝不能放密码、身份证号等敏感信息。它的安全性来自"签名防篡改"。
    """
    now = dt.datetime.now(dt.timezone.utc)
    payload = {
        "sub": username,
        "role": role,
        "iat": now,
        "exp": now + dt.timedelta(minutes=expires_minutes),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """校验并解码 JWT。

    jwt.decode 会依次检查：
        1) 签名是否正确（用同一个密钥重算 HMAC）→ 不对就抛 InvalidSignatureError
        2) 算法是否在允许列表里 → 防止 "alg=none" 这类降级攻击
        3) exp 是否过期 → 过期抛 ExpiredSignatureError
    把异常统一转成 401，并带上 WWW-Authenticate 头（OAuth2 规范要求）。
    """
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 已过期，请重新登录",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token 无效（{type(exc).__name__}）",
            headers={"WWW-Authenticate": "Bearer"},
        ) from None


# ============================================================
# 五、登录接口（OAuth2 密码模式）
# ============================================================
@app.post("/token", response_model=Token, summary="登录换取 Token", tags=["认证"])
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> Token:
    """登录接口。

    为什么用 Form 而不是 JSON？
        这是 OAuth2 密码模式（RFC 6749）的协议规定：
        客户端必须以 application/x-www-form-urlencoded 提交 username / password。
        前端用 URLSearchParams 提交即可（见文件末尾注释）。

    安全细节：
        · 用户名不存在与密码错误返回**同一个提示**（"用户名或密码错误"），
          否则等于告诉攻击者"这个用户名是存在的"，方便他做用户名枚举。
    """
    user = USERS.get(form_data.username)
    if user is None or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if user.disabled:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="账号已被禁用")

    token = create_access_token(user.username, user.role)
    return Token(access_token=token, expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60)


# ============================================================
# 六、认证依赖 + 授权依赖
# ============================================================
async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> UserInDB:
    """认证依赖：从 token 里解析出当前用户。

    能走到这里说明 Authorization 头格式正确（OAuth2PasswordBearer 已经检查过了），
    但 token 本身的合法性还要自己验。
    """
    payload = decode_token(token)
    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Token 缺少 sub 字段")
    user = USERS.get(username)
    if user is None:
        # token 合法但用户已被删除（JWT 无状态带来的典型问题）
        raise HTTPException(status_code=401, detail="用户不存在或已被删除")
    if user.disabled:
        raise HTTPException(status_code=403, detail="账号已被禁用")
    return user


def require_role(*roles: str):
    """授权依赖工厂：生成一个"检查角色"的依赖。

    用法：`user: UserInDB = Depends(require_role("admin"))`
    这种"返回依赖函数的函数"是 FastAPI 里实现参数化依赖的标准写法。
    """

    async def checker(user: Annotated[UserInDB, Depends(get_current_user)]) -> UserInDB:
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"需要 {'/'.join(roles)} 角色，当前角色为 {user.role}",
            )
        return user

    return checker


@app.get("/profile", response_model=UserOut, summary="受保护资源：当前用户信息", tags=["受保护"])
async def read_profile(user: Annotated[UserInDB, Depends(get_current_user)]) -> UserOut:
    """任何已登录用户都能访问。"""
    return UserOut(username=user.username, role=user.role)


@app.get("/admin/users", summary="仅管理员可访问", tags=["受保护"])
async def admin_list_users(
    user: Annotated[UserInDB, Depends(require_role("admin"))],
) -> dict:
    """授权示例：只有 admin 角色能看用户列表。

    普通用户访问会得到 403（而不是 401）—— 因为他已经证明了自己是谁，只是权限不够。
    """
    return {
        "operator": user.username,
        "users": [{"username": u.username, "role": u.role, "disabled": u.disabled}
                  for u in USERS.values()],
    }


@app.get("/me/orders", summary="数据隔离：只能看自己的订单", tags=["受保护"])
async def my_orders(user: Annotated[UserInDB, Depends(get_current_user)]) -> dict:
    """授权不仅是"能不能进门"，还包括"能看哪些数据"（水平越权防护）。

    最常见的漏洞写法是 `GET /orders?username=xxx` —— 攻击者改个参数就能看别人的订单。
    正确做法：查询条件里的用户身份**永远取自 token**，绝不从前端参数取。
    """
    fake_orders = {"alice": [{"id": 1, "amount": 99}], "root": [{"id": 2, "amount": 199}]}
    return {"username": user.username, "orders": fake_orders.get(user.username, []),
            "说明": "查询条件取自 token 中的身份，天然防水平越权"}


# ============================================================
# 自检
# ============================================================
def run_self_check() -> None:
    from fastapi.testclient import TestClient

    print("=" * 72)
    print("FastAPI 认证与授权 · 自检")
    print("=" * 72)
    print(f"JWT 算法：{JWT_ALGORITHM}    有效期：{ACCESS_TOKEN_EXPIRE_MINUTES} 分钟")
    print(f"密钥来源：{'环境变量 JWT_SECRET' if os.getenv('JWT_SECRET') else '内置开发默认值（生产必须改）'}")
    print()

    global USERS
    USERS = build_user_table()          # 每次自检重建用户表，保证可重复运行
    client = TestClient(app)

    # ---------- 1. 密码哈希 ----------
    print("[1] 密码哈希（PBKDF2-HMAC-SHA256 + 随机盐）")
    h1 = hash_password("alice123")
    h2 = hash_password("alice123")
    print(f"    同一次调用两次的哈希是否相同：{h1 == h2}（应为 False，因为盐是随机的）")
    print(f"    哈希串格式：{h1[:46]}…（pbkdf2_sha256$迭代次数$盐$哈希）")
    assert h1 != h2, "随机盐应当让相同密码产生不同哈希"
    assert verify_password("alice123", h1), "正确密码应当校验通过"
    assert not verify_password("wrong", h1), "错误密码应当校验失败"
    print("    ✓ 正确密码通过、错误密码失败、相同密码哈希不同（盐生效）")

    # ---------- 2. 登录 ----------
    def login(username: str, password: str) -> tuple[int, dict]:
        resp = client.post("/token", data={"username": username, "password": password})
        return resp.status_code, (resp.json() if resp.content else {})

    code, body = login("alice", "alice123")
    print("\n[2] 登录换取 Token")
    print(f"    POST /token  username=alice  状态码 {code}（期望 200）")
    print(f"    token_type={body.get('token_type')}  expires_in={body.get('expires_in')} 秒")
    assert code == 200 and body["token_type"] == "bearer"
    alice_token = body["access_token"]
    print(f"    access_token（前 60 字符）：{alice_token[:60]}…")

    code, body = login("alice", "wrong-password")
    print("\n[3] 密码错误")
    print(f"    状态码 {code}（期望 401），提示：{body.get('detail')}")
    assert code == 401

    code, body = login("not-exist", "whatever")
    print(f"\n[4] 用户名不存在")
    print(f"    状态码 {code}（期望 401），提示：{body.get('detail')}（与密码错误提示完全一致，防用户名枚举）")
    assert code == 401 and body["detail"] == "用户名或密码错误"

    code, body = login("tom", "tom123")
    print("\n[5] 被禁用的账号")
    print(f"    状态码 {code}（期望 403），提示：{body.get('detail')}")
    assert code == 403

    # ---------- 3. 受保护资源 ----------
    print("\n[6] 访问受保护资源 /profile")
    resp = client.get("/profile")
    print(f"    不带 Token：状态码 {resp.status_code}（期望 401），响应 {resp.text[:70]}")
    assert resp.status_code == 401
    assert resp.headers.get("WWW-Authenticate") == "Bearer"

    resp = client.get("/profile", headers={"Authorization": f"Bearer {alice_token}"})
    print(f"    带合法 Token：状态码 {resp.status_code}（期望 200），响应 {resp.text}")
    assert resp.status_code == 200 and resp.json()["username"] == "alice"

    # ---------- 4. Token 篡改与过期 ----------
    print("\n[7] Token 安全性验证")
    tampered = alice_token[:-4] + ("aaaa" if not alice_token.endswith("aaaa") else "bbbb")
    resp = client.get("/profile", headers={"Authorization": f"Bearer {tampered}"})
    print(f"    篡改签名后：状态码 {resp.status_code}（期望 401），提示 {resp.json().get('detail')}")
    assert resp.status_code == 401

    resp = client.get("/profile", headers={"Authorization": "Bearer not.a.jwt"})
    print(f"    随便编一个：状态码 {resp.status_code}（期望 401），提示 {resp.json().get('detail')}")
    assert resp.status_code == 401

    expired = create_access_token("alice", "user", expires_minutes=-1)   # 已过期 1 分钟
    resp = client.get("/profile", headers={"Authorization": f"Bearer {expired}"})
    print(f"    已过期的 Token：状态码 {resp.status_code}（期望 401），提示 {resp.json().get('detail')}")
    assert resp.status_code == 401

    wrong_secret = jwt.encode({"sub": "hacker", "role": "admin",
                               "exp": dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=5)},
                              "another-secret", algorithm="HS256")
    resp = client.get("/profile", headers={"Authorization": f"Bearer {wrong_secret}"})
    print(f"    别人用其他密钥签的：状态码 {resp.status_code}（期望 401），提示 {resp.json().get('detail')}")
    assert resp.status_code == 401
    print("    ✓ 签名校验能挡住：改内容、乱编、过期、换密钥 —— 四种攻击都不行")

    # ---------- 5. 授权 ----------
    print("\n[8] 授权（角色权限）")
    resp = client.get("/admin/users", headers={"Authorization": f"Bearer {alice_token}"})
    print(f"    alice(user) 访问 /admin/users：状态码 {resp.status_code}（期望 403），"
          f"提示 {resp.json().get('detail')}")
    assert resp.status_code == 403

    code, body = login("root", "root123")
    assert code == 200
    root_token = body["access_token"]
    resp = client.get("/admin/users", headers={"Authorization": f"Bearer {root_token}"})
    print(f"    root(admin) 访问 /admin/users：状态码 {resp.status_code}（期望 200）")
    print(f"    响应：{resp.text[:150]}…")
    assert resp.status_code == 200 and resp.json()["operator"] == "root"
    print("    ✓ 401（未认证）与 403（已认证但无权限）语义清晰")

    # ---------- 6. 数据隔离 ----------
    print("\n[9] 数据隔离（防水平越权）")
    resp = client.get("/me/orders", headers={"Authorization": f"Bearer {alice_token}"})
    print(f"    alice 查看自己的订单：{resp.json()}")
    resp = client.get("/me/orders", headers={"Authorization": f"Bearer {root_token}"})
    print(f"    root  查看自己的订单：{resp.json()}")
    print("    ✓ 用户身份来自 Token，不接受前端传入的用户名参数")

    # ---------- 7. JWT 的可读性 ----------
    header_b64, payload_b64, _ = alice_token.split(".")
    import base64
    import json as _json

    def _b64(s: str) -> dict:
        return _json.loads(base64.urlsafe_b64decode(s + "=" * (-len(s) % 4)))

    print("\n[10] JWT 结构（Base64 只是编码，不是加密）")
    print(f"    header  = {_b64(header_b64)}")
    print(f"    payload = {_b64(payload_b64)}")
    print("    ✓ 任何人都能解开 payload 看内容，所以里面不能放敏感信息")

    print("\n" + "=" * 72)
    print("自检全部通过 ✓")
    print("=" * 72)
    print("知识点回顾：")
    print("1. 认证=你是谁（401），授权=你能做什么（403），两者分开实现")
    print("2. 密码用 PBKDF2 + 随机盐哈希存储，比对用 hmac.compare_digest 防时序攻击")
    print("3. JWT = Base64(header).Base64(payload).签名；payload 可读不可改")
    print("4. OAuth2PasswordBearer 只提取 token，验证必须自己做（jwt.decode）")
    print("5. 用依赖注入串联：get_current_user（认证）→ require_role（授权）")
    print("6. 查询数据时，用户身份只从 token 取，绝不信前端传的参数")
    print("7. 前端登录示例（OAuth2 密码模式要求表单格式）：")
    print("   fetch('/token', {method:'POST',")
    print("     headers:{'Content-Type':'application/x-www-form-urlencoded'},")
    print("     body:new URLSearchParams({username:'alice', password:'alice123'})})")
    print()


def main() -> None:
    print()
    print("#" * 72)
    print("# FastAPI 认证与授权（后端开发基础 · Web框架章节）")
    print("#" * 72)
    print(f"解释器：{sys.executable}")
    print(f"计划监听端口：{PORT}")

    if "--check" in sys.argv:
        print()
        print("提示：--check 模式，执行自检，不启动服务器、不阻塞。")
        run_self_check()
    else:
        print(f"启动 uvicorn：http://127.0.0.1:{PORT}/docs")
        print("演示账号：alice/alice123（普通用户）、root/root123（管理员）、tom/tom123（已禁用）")
        print("按 Ctrl+C 停止。")
        uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")


if __name__ == "__main__":
    main()

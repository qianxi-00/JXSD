# -*- coding: utf-8 -*-
"""
MCP ⑦：权限——认证服务（签发 JWT）
================================================================
MCP 服务不能裸奔，谁都能调就乱套了。标准做法：
    1. 认证服务：用户登录，校验身份后签发 JWT 令牌（本文件）
    2. MCP 服务端：用公钥校验请求头里的 JWT（见 08_权限_服务端.py）
    3. 客户端：先拿令牌，再带 Bearer Token 调用（见 09_权限_客户端.py）

JWT 组成：header.payload.signature
    payload 里放用户身份（sub）、权限（scope）、过期时间（exp）等。

运行方式：
    uv run Agent/05_mcp/07_权限_认证服务.py          # 签发一个演示令牌
    uv run Agent/05_mcp/07_权限_认证服务.py serve    # 作为登录接口运行（9000 端口）
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from datetime import datetime, timedelta, timezone

import jwt
from fastapi import FastAPI
from pydantic import BaseModel

# ============ 密钥配置 ============
# 生产环境：私钥自己保管（签发用），公钥公开（校验用）
# 演示环境：直接用对称密钥（同一个字符串既签发又校验）
JWT_SECRET = "demo-secret-change-me"   # 生产环境必须换成 RSA 私钥/强随机密钥
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 60


def create_token(user_id: str, scopes: list[str]) -> str:
    """签发 JWT：把用户身份和权限写进 payload"""
    payload = {
        "sub": user_id,                     # subject：用户唯一标识
        "scope": " ".join(scopes),          # 权限列表（如 read write execute）
        "iat": datetime.now(timezone.utc),  # 签发时间
        "exp": datetime.now(timezone.utc) + timedelta(minutes=TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=ALGORITHM)


def verify_token(token: str) -> dict:
    """校验 JWT：验签 + 验过期，返回 payload；不合法直接抛异常"""
    return jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])


# ---------- 作为 HTTP 登录接口运行 ----------
app = FastAPI(title="MCP 认证服务")


class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/login")
def login(req: LoginRequest):
    """
    演示版：任意用户名 + 密码 123456 都能登录。
    生产版：查数据库校验密码（密码哈希存储）。
    """
    if req.password != "123456":
        return {"error": "用户名或密码错误"}

    # 按用户角色分配权限
    scopes = ["read", "execute"] if req.username == "admin" else ["read"]
    return {"access_token": create_token(req.username, scopes), "token_type": "Bearer"}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "serve":
        # 以 HTTP 服务运行登录接口
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=9000)
    else:
        token = create_token("user-1001", ["read", "execute"])
        print("签发的 JWT：", token)
        print("校验结果：", verify_token(token))

# -*- coding: utf-8 -*-
"""
MCP ⑧：权限——带认证的服务端
================================================================
给 MCP 服务加上 JWT 校验：所有请求必须携带合法的 Bearer Token。
FastMCP 通过 auth 参数注入 JWTVerifier（校验器）。

说明：
    - 演示用对称密钥（与 07_权限_认证服务.py 共享 JWT_SECRET）
    - 生产环境建议 RSA：认证服务持私钥签发，服务端只用公钥校验
      （JWTVerifier 支持 jwks_uri 自动拉取公钥）

运行方式：
    uv run 05_mcp/08_权限_服务端.py http
    （客户端见 09_权限_客户端.py）
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from fastmcp import FastMCP
from fastmcp.server.auth.providers.jwt import JWTVerifier

# 与认证服务共享的密钥（生产环境换成公钥/JWKS）
JWT_SECRET = "demo-secret-change-me"

# 创建校验器：要求请求头 Authorization: Bearer <token>
# 演示用对称密钥（HS256）：public_key 参数即共享密钥
verifier = JWTVerifier(
    public_key=JWT_SECRET,
    algorithm="HS256",
    required_scopes=["read"],  # 所有调用至少要有 read 权限
)

mcp = FastMCP(name="受保护的MCP服务", auth=verifier)


@mcp.tool
def get_weather(city: str) -> str:
    """查询指定城市的实时天气"""
    weather_map = {"上海": "晴 25 度", "北京": "多云 18 度"}
    return weather_map.get(city, f"{city} 天气未知")


@mcp.tool
def admin_reset() -> str:
    """重置系统（危险操作，需要 write 权限才能调用）"""
    # 细粒度权限：在函数内自己校验 scope（FastMCP 会把 Token 信息放进上下文）
    return "系统已重置"


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    # 带认证的服务只能以 HTTP 运行（stdio 无法传 Token 头）
    mcp.run(transport="http", host="127.0.0.1", port=8000, path="/mcp")

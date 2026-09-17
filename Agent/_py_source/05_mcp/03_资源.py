# -*- coding: utf-8 -*-
"""
MCP ③：资源（Resource）
================================================================
工具 = 让模型「做事」（有副作用，如发邮件、写库）
资源 = 让模型「读数据」（只读，URI 定位，如配置、文件、状态）

资源用 URI 标识，如 config://app-info、db://users/1001。
分静态资源和动态资源（URI 模板，带参数）。

运行方式：
    uv run 05_mcp/03_资源.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

mcp = FastMCP(name="资源演示")


# ---------- 静态资源：固定 URI ----------
@mcp.resource("config://app-info")
def app_info() -> str:
    """应用的基础信息（只读配置）"""
    return "应用名称：Agent 课案演示；版本：1.0.0"


# ---------- 动态资源：URI 模板 {param} ----------
@mcp.resource("db://users/{user_id}")
def user_profile(user_id: str) -> str:
    """按用户 ID 读取用户资料"""
    return f"用户 {user_id}：小红，上海，VIP3 会员"


def run_stdio():
    mcp.run()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "http":
        mcp.run(transport="http", host="127.0.0.1", port=8000, path="/mcp")
    else:
        run_stdio()

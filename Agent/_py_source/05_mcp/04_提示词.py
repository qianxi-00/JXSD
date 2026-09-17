# -*- coding: utf-8 -*-
"""
MCP ④：提示词（Prompt）
================================================================
@mcp.prompt 定义可复用的提示词模板，客户端可以拉取使用。
用途：把团队沉淀的优质 Prompt 集中管理、统一分发、随时更新。

运行方式：
    uv run 05_mcp/04_提示词.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from fastmcp import FastMCP

mcp = FastMCP(name="提示词演示")


# ---------- 简单提示词 ----------
@mcp.prompt
def code_review(code: str) -> str:
    """代码审查提示词模板：参数会自动填充进模板"""
    return f"请审查以下代码，从 可读性 / 性能 / 安全 三个角度给出意见：\n\n{code}"


# ---------- 多消息提示词（返回消息列表，可包含角色） ----------
@mcp.prompt
def translate(text: str, target_lang: str = "英文") -> list:
    """翻译提示词：system 设定角色，user 给任务"""
    return [
        {"role": "system", "content": "你是专业翻译，译文自然流畅。"},
        {"role": "user", "content": f"把下面的内容翻译成{target_lang}：\n{text}"},
    ]


def run_stdio():
    mcp.run()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "http":
        mcp.run(transport="http", host="127.0.0.1", port=8000, path="/mcp")
    else:
        run_stdio()

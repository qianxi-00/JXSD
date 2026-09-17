# -*- coding: utf-8 -*-
"""
LangChain 基础：模型（init_chat_model）
================================================================
LangChain 用 init_chat_model 统一封装各家大模型。
好处：换模型供应商只改参数，业务代码完全不变。

    init_chat_model(
        model_provider="openai",     # 供应商：openai / anthropic / google_genai ...
        model="deepseek-chat",       # 模型名
        api_key=..., base_url=...,   # OpenAI 兼容接口相关参数
    )

课案约定：密钥从 conf.settings 读取，不硬编码。

运行方式：
    uv run 02_langchain/01_模型.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.chat_models import init_chat_model
from config import settings

# 初始化模型：所有 OpenAI 兼容服务（DeepSeek、通义、Kimi、vLLM……）都这样接
llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
    temperature=0.7,  # 采样温度，越高越随机
)

if __name__ == "__main__":
    # ---------- 最简单调用：一句话进，一句话出 ----------
    response = llm.invoke("用一句话解释什么是 Agent")
    print("普通回复：", response.content)

    # ---------- 带系统提示词（角色设定） ----------
    response = llm.invoke(
        [
            ("system", "你是一位毒舌程序员，回答简短刻薄但专业。"),
            ("user", "Python 好学吗？"),
        ]
    )
    print("带角色回复：", response.content)

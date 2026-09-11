# -*- coding: utf-8 -*-
"""
Langfuse ①：基础追踪（Tracing）
================================================================
Agent 上线后是个黑盒：它为什么调了这个工具？花了多少钱？
Langfuse = LLM 应用的可观测平台，自动记录：
    每次模型调用（输入 / 输出 / token 数 / 耗时 / 成本）
    每次工具调用（参数 / 结果）
    完整的调用链路树（trace）

两种接入方式：
    1. @observe 装饰器：自动把普通函数变成可追踪的 span
    2. CallbackHandler：LangChain 专用回调，自动追踪整条链路

前置：.env 配置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST

运行方式：
    uv run 06_langfuse/01_追踪.py
    （去 Langfuse 控制台查看 trace）
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.chat_models import init_chat_model
from langfuse import Langfuse, get_client, observe
from langfuse.langchain import CallbackHandler
from config import settings

# ---------- 方式 1：@observe 装饰器（任意 Python 函数） ----------
lf = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    host=settings.langfuse_host,
)


@observe  # 这个函数会被自动追踪：每次调用生成一个 trace
def generate_answer(question: str) -> str:
    """普通业务函数，加个装饰器就有完整追踪"""
    llm = init_chat_model(
        model_provider="openai",
        model=settings.model_name,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )
    return llm.invoke(question).content


def demo_observe():
    print("AI：", generate_answer("一句话解释什么是 Langfuse"))
    get_client().flush()  # 立刻上传（否则可能等批量刷新）


# ---------- 方式 2：LangChain 回调（自动追踪整条链） ----------
def demo_callback():
    llm = init_chat_model(
        model_provider="openai",
        model=settings.model_name,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )
    handler = CallbackHandler()  # LangChain 全链路自动埋点
    response = llm.invoke("什么是 RAG？一句话回答。", config={"callbacks": [handler]})
    print("AI：", response.content)
    get_client().flush()  # 立即上传（v4 用 get_client().flush()）


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    demo_observe()
    demo_callback()
    print("已上传到 Langfuse，去控制台查看：", settings.langfuse_host)

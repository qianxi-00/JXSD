# -*- coding: utf-8 -*-
"""
Langfuse ②：会话（Session）与用户（User）
================================================================
单条 trace 只能看到一次调用；要分析「某个用户的完整旅程」、
「某次会话的连续对话」，需要打上元数据标签：

    trace_id  ：一次请求/一次图执行
    session_id：一次会话（一次聊天窗口）
    user_id   ：一个用户（跨会话聚合）

LangChain 接入：CallbackHandler 会自动从 config["configurable"] 里读
session_id / user_id，无需手动设置。

运行方式：
    uv run 06_langfuse/02_会话和用户.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.chat_models import init_chat_model
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    session_id = "session-20260909-001"
    user_id = "user-1001"

    # 模拟同一用户、同一会话里的两轮对话
    handler = CallbackHandler()
    config = {
        "callbacks": [handler],
        "configurable": {
            "session_id": session_id,  # 同一会话共享
            "user_id": user_id,        # 归属到用户
        },
    }

    r1 = llm.invoke("你好，我是小明", config=config)
    r2 = llm.invoke("帮我写一句朋友圈文案，关于加班", config=config)

    get_client().flush()
    print("两轮对话已上传；在 Langfuse 里可按用户/会话筛选查看")

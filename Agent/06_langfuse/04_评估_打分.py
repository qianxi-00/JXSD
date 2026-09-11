# -*- coding: utf-8 -*-
"""
Langfuse ④：评估——打分（Score）
================================================================
追踪只能看到「发生了什么」，评估回答「好不好」。
两种打分方式：

    1. 人工打分：在 Langfuse 控制台对某条 trace 点赞/打分/标注
    2. 代码打分：程序里对 trace 打分（用户反馈、业务指标、LLM 评审）

v4 API：在 @observe 函数内直接调用 score_current_trace()
自动打到「当前这次调用」的 trace 上，无需手动拿 trace_id。

打分后可以在 Langfuse 看板上分析：哪类问题得分低、哪个版本更好。

运行方式：
    uv run Agent/06_langfuse/04_评估_打分.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.chat_models import init_chat_model
from langfuse import Langfuse, observe
from config import settings

lf = Langfuse(
    public_key=settings.langfuse_public_key,
    secret_key=settings.langfuse_secret_key,
    host=settings.langfuse_host,
)

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


@observe
def answer(question: str) -> str:
    """带追踪的业务函数：内部收集业务指标并打分"""
    response = llm.invoke(question)

    # ---------- 业务指标打分示例：回复长度是否达标 ----------
    length = len(response.content)
    lf.score_current_trace(
        name="response_length_ok",
        value=1.0 if 20 <= length <= 200 else 0.0,
        comment=f"回复长度 {length} 字",
    )
    return response.content


@observe
def with_user_feedback(question: str) -> str:
    """模拟「用户点了踩」：把点赞/点踩上报为 trace 分数"""
    response = llm.invoke(question)
    feedback = "thumb_up"  # 实际来自前端用户的点赞按钮
    lf.score_current_trace(
        name="user_feedback",
        value=feedback,          # 类别型分数直接传字符串
        data_type="CATEGORICAL",
    )
    return response.content


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    answer("什么是 Function Call？一句话回答")
    with_user_feedback("什么是 MCP？一句话回答")
    lf.flush()
    print("打分已上传，在 Langfuse 的 trace 详情和看板中查看")

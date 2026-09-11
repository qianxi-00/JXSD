# -*- coding: utf-8 -*-
"""
LangChain 智能体：结构化输出（with_structured_output）
================================================================
让模型严格按 Pydantic 模型输出 JSON，而不是自由文本。
用途：信息抽取、打分、路由分类、表单填充……
底层原理：Function Call。模型按 schema 生成参数，LangChain 解析校验。

用法：
    structured_llm = llm.with_structured_output(MyModel)
    result = structured_llm.invoke(...)   # 直接得到 MyModel 实例

运行方式：
    uv run 02_langchain/08_结构化输出.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from pydantic import BaseModel, Field

from langchain.chat_models import init_chat_model
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


# ---------- 1. 定义输出结构 ----------
class MovieReview(BaseModel):
    """影评结构"""
    title: str = Field(description="电影名称")
    score: int = Field(description="评分 1-10")
    reason: str = Field(description="一句话评价理由")
    tags: list[str] = Field(description="类型标签，如['科幻','剧情']")


class RouteDecision(BaseModel):
    """路由决策结构"""
    intent: str = Field(description="用户意图：chat / order / complaint")


if __name__ == "__main__":
    # ---------- 2. 信息抽取 ----------
    reviewer = llm.with_structured_output(MovieReview)
    review = reviewer.invoke("评价一下《流浪地球2》：视觉震撼，剧情稍散，8分吧")
    print("结构化结果：")
    print("  片名：", review.title)
    print("  评分：", review.score)
    print("  理由：", review.reason)
    print("  标签：", review.tags)

    # ---------- 3. 意图路由（客服分流常用） ----------
    router = llm.with_structured_output(RouteDecision)
    decision = router.invoke("我要投诉你们物流太慢了！")
    print("路由决策：", decision.intent)

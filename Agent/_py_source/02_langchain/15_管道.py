# -*- coding: utf-8 -*-
"""
LangChain 管道（LCEL 链式编程）
================================================================
LCEL（LangChain Expression Language）用 `|` 把组件串成管道：

    prompt | model | parser

数据像水流一样从左流向右，上一个组件的输出是下一个的输入。
支持 batch（并行批量）、stream（流式）、ainvoke（异步）等统一接口。

运行方式：
    uv run 02_langchain/15_管道.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

if __name__ == "__main__":
    # ---------- 1. 三段管道：提示词模板 → 模型 → 解析器 ----------
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是{domain}专家，回答不超过 30 字。"),
            ("user", "{question}"),
        ]
    )
    parser = StrOutputParser()  # 把 AIMessage 解析成纯字符串

    chain = prompt | llm | parser

    result = chain.invoke({"domain": "数据库", "question": "什么是索引？"})
    print("管道结果：", result)

    # ---------- 2. 管道支持流式 ----------
    print("----- 流式 -----")
    for token in chain.stream({"domain": "后端", "question": "什么是消息队列？"}):
        print(token, end="", flush=True)
    print()

    # ---------- 3. 管道支持批量并行 ----------
    results = chain.batch(
        [
            {"domain": "前端", "question": "什么是虚拟 DOM？"},
            {"domain": "运维", "question": "什么是容器？"},
        ]
    )
    for r in results:
        print("批量结果：", r)

    # ---------- 4. 管道串联管道（子链复用） ----------
    analyzer = prompt | llm | parser
    summary_chain = analyzer | (lambda text: f"【分析摘要】{text}")
    print(summary_chain.invoke({"domain": "AI", "question": "什么是 RAG？"}))

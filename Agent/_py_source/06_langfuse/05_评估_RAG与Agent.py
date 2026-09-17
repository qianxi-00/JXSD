# -*- coding: utf-8 -*-
"""
Langfuse ⑤：评估——RAG 评估 与 Agent 评估（Experiment）
================================================================
用 Langfuse 的实验（Experiment）能力批量评估：

    run_experiment(
        name=实验名,
        data=[{"input":..., "expected_output":...}, ...],  # 测试用例
        task=被测任务,            # (*, item) -> 输出
        evaluators=[评分函数],    # (input, output, expected_output, metadata) -> 分数
    )

    RAG 评估：问答对数据集 + 检索流程，用 ragas 指标评估
        - faithfulness（忠实度：回答是否基于检索内容）
        - answer_relevancy（答案相关性）
    Agent 评估：自定义评分函数，检查 Agent 是否按预期行动

运行方式：
    uv run Agent/06_langfuse/05_评估_RAG与Agent.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from langchain.chat_models import init_chat_model
from langfuse import Langfuse
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

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    # ---------- 1. 准备测试用例（也可换成控制台 Dataset 里的 items） ----------
    data = [
        {"input": {"question": "LangChain 是什么？"},
         "expected_output": {"keywords": "LLM 应用开发框架"}},
        {"input": {"question": "LangGraph 是什么？"},
         "expected_output": {"keywords": "智能体图编排框架"}},
    ]

    # ---------- 2. 定义被测任务（换成你的 RAG/Agent 流程） ----------
    def my_rag_task(*, item, **kwargs):
        """被评估的任务：输入问题，返回回答（这里用裸 LLM 模拟）"""
        question = item["input"]["question"]
        answer = llm.invoke(question).content
        return {"answer": answer}

    # ---------- 3. 定义评分函数 ----------
    def relevance_scorer(*, input, output, expected_output, **kwargs):
        """关键词命中率：expected 的关键词是否出现在回答里"""
        keywords = expected_output["keywords"]
        answer = output["answer"]
        hit = all(k in answer for k in keywords.split())
        return {"name": "keyword_hit", "value": 1.0 if hit else 0.0}

    # ---------- 4. 跑实验 ----------
    try:
        result = lf.run_experiment(
            name="baseline-v1",
            data=data,
            task=my_rag_task,
            evaluators=[relevance_scorer],
        )
        print("实验完成！结果概览：", result)
    except Exception as e:
        print("实验执行失败（检查 Langfuse 密钥是否配置）：", e)
    finally:
        lf.flush()

    # RAG 评估进阶：ragas 库提供 faithfulness / answer_relevancy 等标准指标，
    # 把 ragas 的 score 函数包成 evaluator 传入 run_experiment 即可。

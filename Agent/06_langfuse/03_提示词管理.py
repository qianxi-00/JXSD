# -*- coding: utf-8 -*-
"""
Langfuse ③：提示词管理（Prompt Management）
================================================================
提示词不该硬编码在代码里（改一句要发版）。
Langfuse 提供提示词托管：
    - 控制台创建/修改提示词，带版本管理
    - 代码里 get_prompt 拉取最新版本
    - 支持 label（production / staging / latest）切换灰度

好处：产品/运营在界面上改提示词，代码零改动即时生效。

运行方式：
    1. 在 Langfuse 控制台创建名为「sql-expert」的 text prompt
    2. uv run 06_langfuse/03_提示词管理.py
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

    # ---------- 1. 拉取托管提示词 ----------
    # get_prompt(名字, label/version)；label="production" 取生产版
    try:
        prompt = lf.get_prompt("sql-expert", label="production")
        prompt_text = prompt.compile(variables={"dialect": "MySQL"})  # 编译变量
    except Exception as e:
        # 没配置时兜底：本地默认提示词（演示用）
        print(f"未找到托管提示词（{e}），使用本地默认")
        prompt_text = "你是 MySQL 专家，把用户需求翻译成 SQL，只输出 SQL。"

    # ---------- 2. 使用提示词 ----------
    response = llm.invoke(
        f"{prompt_text}\n\n需求：查询每个部门的平均工资，只看平均工资大于 1 万的部门"
    )
    print("AI：", response.content)

    # ---------- 3. （可选）代码里创建/更新提示词 ----------
    # lf.create_prompt(
    #     name="sql-expert",
    #     prompt="你是 {dialect} 专家，把用户需求翻译成 SQL，只输出 SQL。",
    #     labels=["production"],
    # )
    lf.flush()

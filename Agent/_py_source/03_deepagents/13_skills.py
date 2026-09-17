# -*- coding: utf-8 -*-
"""
DeepAgents Skills（技能）
================================================================
Skill = 预先写好的「操作手册」（Markdown 文件），平时不占上下文，
只有任务匹配到时才会被 Agent 按需加载（渐进式披露）。

结构（一个文件夹 = 一个技能）：
    my-skill/
        SKILL.md     必须的入口文件，含 frontmatter（name/description）
        （其他参考文件，Agent 需要时再读）

SKILL.md 示例：
    ---
    name: report-writer
    description: 撰写正式周报时使用本技能
    ---
    （正文：周报的写作步骤和格式要求）

用法：create_deep_agent(skills=["技能目录路径", ...])
依赖：uv add deepagents-skills（如果使用独立 skills 包）
     本示例直接用本地技能文件夹，把 skills 指向 13_skill_demo 目录。

运行方式：
    uv run 03_deepagents/13_skills.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

if __name__ == "__main__":
    # ---------- 1. 动态创建一个技能目录（实际项目中提前准备好） ----------
    skill_dir = Path("tmp_skill_report_writer")
    skill_dir.mkdir(exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: report-writer
description: 撰写正式工作周报时使用本技能
---

# 周报撰写规范

1. 结构固定为三段：本周完成 / 下周计划 / 风险与求助
2. 每条用「动宾短语」开头，例如「完成了 XX 模块开发」
3. 总字数控制在 200 字以内
""",
        encoding="utf-8",
    )

    # ---------- 2. 创建带技能的 Deep Agent ----------
    agent = create_deep_agent(
        model=llm,
        tools=[],
        system_prompt="你是职场写作助手。",
        skills=[str(skill_dir)],  # 传入技能目录，Agent 会按需加载
    )

    result = agent.invoke(
        {"messages": [("user", "帮我写本周周报：完成了登录模块，下周做支付")]},
        config={"recursion_limit": 50},
    )
    print("AI：", result["messages"][-1].content)

# -*- coding: utf-8 -*-
"""
DeepAgents 后端（Backend）⑦：CompositeBackend —— 组合后端
================================================================
不同类型的文件放不同地方：
    - 临时文件（草稿、中间结果）→ StateBackend（快、随会话丢弃）
    - 持久文件（最终成果）      → FilesystemBackend（落盘保留）

CompositeBackend 按路径前缀路由到不同后端。

运行方式：
    uv run 03_deepagents/09_后端_Composite.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain.chat_models import init_chat_model
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

if __name__ == "__main__":
    out_dir = Path("tmp_deepagents_composite")
    out_dir.mkdir(exist_ok=True)

    # 路由规则：/final/** 走磁盘，其他走 State
    backend = CompositeBackend(
        default=StateBackend(),
        routes={"/final/": FilesystemBackend(root_dir=str(out_dir))},
    )

    agent = create_deep_agent(
        model=llm,
        backend=backend,
        system_prompt="草稿写临时文件，最终成果写入 /final/ 目录。",
    )

    result = agent.invoke(
        {"messages": [("user", "把最终报告写到 /final/report.md，内容是：项目验收通过")]},
        config={"recursion_limit": 50},
    )
    print("AI：", result["messages"][-1].content)
    print("磁盘文件：", [p.name for p in out_dir.iterdir()])

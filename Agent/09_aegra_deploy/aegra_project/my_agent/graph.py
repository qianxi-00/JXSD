# -*- coding: utf-8 -*-
"""
Aegra 项目的 Graph 定义（由 09_aegra_deploy/02_项目骨架_jxsd.py 生成）
================================================================
aegra.json 的 graphs 字段指向这里："./my_agent/graph.py:graph"
即「文件路径 : 模块级变量名」，所以下面那个 `graph = ...` 是**必须叫 graph** 的。

配置**不另起一套**：直接用 Python_Base 根目录那份 config.py / .env。
下面的循环会从本文件往上找，第一个同时含 config.py + pyproject.toml 的目录
就是 Python_Base 根目录，把它插进 sys.path，`from config import settings` 才能生效。
（aegra.json 的 "dependencies": ["./"] 只把**本项目**根目录加进 sys.path，
够不到 Python_Base 根目录，所以这里要自己补一句。这就是那个字段的真实作用。）

⚠️ 放到服务器上独立部署时，Python_Base 根目录不在场，按课案原样做即可：
   把根目录的 config.py 和 .env 一起带过去（**仍然只有这一份配置**），不用另写 conf.py。
"""

import sys
from pathlib import Path

# 向上找到 Python_Base 根目录（含 config.py + pyproject.toml 的那一层）
for _p in Path(__file__).resolve().parents:
    if (_p / "config.py").exists() and (_p / "pyproject.toml").exists():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from config import settings  # 根目录统一配置，不另建一套

# 课案用的是 ChatOpenAI 直连写法（部署项目自成一体，不依赖 Python_Base 的
# init_chat_model 封装），但三个参数一律来自 settings，绝不硬编码。
llm = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


def chatbot(state: MessagesState):
    """最简单的单节点对话：把最近 6 条消息丢给模型

    为什么只取 [-6:]：MessagesState 是无限增长的，全量传进去 token 会越来越贵。
    这里截最近 6 条做上下文窗口，是课案里最朴素的「滑动窗口记忆」写法。
    （真正要长期记忆时，应换成 checkpoint + 摘要，见 01_langgraph 那几节。）
    """
    return {"messages": [llm.invoke(state["messages"][-6:])]}


# 图的组装：一个节点、两条边，START → chatbot → END
# 用链式写法（课案原样）而不是 builder.add_node 逐行写。
graph = (
    StateGraph(MessagesState)
    .add_node("chatbot", chatbot)
    .add_edge(START, "chatbot")
    .add_edge("chatbot", END)
    .compile()
)

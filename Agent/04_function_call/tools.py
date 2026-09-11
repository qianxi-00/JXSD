# -*- coding: utf-8 -*-
"""
Function Call ①：工具函数定义（tools.py）
================================================================
Function Call 是 Agent 的基石：模型本身只会「说话」，
要让它「做事」（查库、发邮件、算数……），必须提供工具。
本文件定义智能体可用的工具函数，被三个版本的示例共用：
    - agent_openai.py     （原生 OpenAI SDK）
    - agent_langchain.py  （LangChain）
    - agent_deepagents.py （DeepAgents）

运行方式：本文件只是库，不需要直接运行。
"""

import json
import random


# ---------- 普通函数版（给原生 OpenAI SDK 用） ----------
def get_weather(city: str) -> str:
    """
    查询指定城市的天气。

    :param city: 城市名称，如「上海」
    :return: 天气描述字符串
    """
    weather_map = {
        "上海": ("晴", 25),
        "北京": ("多云", 18),
        "广州": ("阵雨", 30),
    }
    desc, temp = weather_map.get(city, ("未知", random.randint(0, 35)))
    return json.dumps({"city": city, "weather": desc, "temperature": temp},
                      ensure_ascii=False)


def get_current_time() -> str:
    """获取当前时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 工具注册表：模型返回工具名后，程序在这里查到真正要执行的函数
TOOL_REGISTRY = {
    "get_weather": get_weather,
    "get_current_time": get_current_time,
}

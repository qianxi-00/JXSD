# -*- coding: utf-8 -*-
"""
Function Call ②：工具描述（tool_desc.py）
================================================================
Function Call 的关键第二件套：把工具「翻译」成模型能理解的 JSON Schema。
模型看不到函数代码，只看到这份描述；描述写得好坏直接决定调用准确率。

OpenAI 格式三要素：
    type        固定为 "function"
    function.name        函数名（模型据此指名调用）
    function.description 干什么用（模型据此判断何时该调用）
    function.parameters  参数的 JSON Schema（模型据此填参）

运行方式：本文件只是库，不需要直接运行。
"""

# 查天气工具的描述（与 tools.get_weather 对应）
GET_WEATHER_DESC = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市的实时天气，包括天气状况和温度",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，例如：上海",
                },
            },
            "required": ["city"],
        },
    },
}

# 查时间工具的描述（无参数）
GET_TIME_DESC = {
    "type": "function",
    "function": {
        "name": "get_current_time",
        "description": "获取当前的日期和时间",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

# 发给 API 的 tools 参数：工具描述列表
TOOLS = [GET_WEATHER_DESC, GET_TIME_DESC]

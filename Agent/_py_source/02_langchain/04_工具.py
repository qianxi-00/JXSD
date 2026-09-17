# -*- coding: utf-8 -*-
"""
LangChain 基础：工具（Tool）
================================================================
工具 = 让模型能「动手做事」的 Python 函数。
@tool 装饰器会自动从函数签名和 docstring 生成 JSON Schema 描述，
模型据此决定什么时候调、传什么参数。

三种定义方式：
    1. @tool 装饰器（推荐，最常用）
    2. StructuredTool.from_function
    3. Pydantic 模型定义（参数复杂 / 需要校验时）

注意：docstring 越清晰，模型调用越准——它就是写给模型看的说明书。

运行方式：
    uv run 02_langchain/04_工具.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from pydantic import BaseModel, Field

from langchain_core.tools import StructuredTool, tool


# ---------- 方式 1：@tool 装饰器 ----------
@tool
def add(a: int, b: int) -> int:
    """计算两个整数的和。a：第一个数；b：第二个数"""
    return a + b


# ---------- 方式 2：StructuredTool ----------
def search_db(keyword: str, limit: int = 10) -> str:
    """模拟数据库搜索"""
    return f"在数据库中搜索「{keyword}」，返回 {limit} 条结果"


search_tool = StructuredTool.from_function(
    func=search_db,
    name="search_db",
    description="在业务数据库中搜索关键词",
)


# ---------- 方式 3：Pydantic 模型（参数复杂 / 需要校验） ----------
class OrderQuery(BaseModel):
    """订单查询入参"""
    order_id: str = Field(description="订单编号，格式 ORD-xxxx")
    with_detail: bool = Field(default=False, description="是否返回明细")


@tool(args_schema=OrderQuery)
def query_order(order_id: str, with_detail: bool = False) -> str:
    """根据订单编号查询订单信息"""
    return f"订单 {order_id} 状态：已发货（{'含明细' if with_detail else '不含明细'}）"


if __name__ == "__main__":
    # 工具直接调用 = 普通函数
    print(add.invoke({"a": 1, "b": 2}))

    # 查看 LangChain 自动生成的 JSON Schema（这就是发给模型的工具描述）
    print(add.tool_call_schema.model_json_schema())

    # 模拟模型的工具调用：传入 tool_call，拿到 ToolMessage
    result = add.tool_call_schema.model_validate({"a": 3, "b": 5})
    print("按 schema 解析参数：", add.invoke(result.model_dump()))

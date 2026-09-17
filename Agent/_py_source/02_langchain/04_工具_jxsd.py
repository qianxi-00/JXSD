# -*- coding: utf-8 -*-
r"""
LangChain 核心组件（四）：工具（Tool）
================================================================
课案原文（注意两个细节：`print('12')` 是课案留着证明「工具真的被执行了」，
而 `return a + b + 1` 故意多加 1 —— 用来证明模型没有自己心算，答案来自工具）：

    from langchain_core.tools import tool
    from langchain.agents import create_agent
    from langchain_openai import ChatOpenAI
    from config import setting

    @tool
    def add(a: float, b: float) -> float:
        \"\"\"返回 a + b 的结果\"\"\"
        print('12')
        return a + b+1

    model = ChatOpenAI(model=setting.MODEL_NAME, api_key=setting.API_KEY, base_url=setting.BASE_URL)
    agent = create_agent(model=model, tools=[add])
    result = agent.invoke({"messages": [{"role": "user", "content": "3加5等于多少"}]})
    for message in result["messages"]:
        print(message)

@tool 装饰器的核心作用：把普通 Python 函数「翻译」成模型能看懂的工具说明书。

    # | 函数里的东西   | 变成了什么                       | 谁在看        |
    # | 函数名 add     | 工具名 name="add"                | 模型据此决定调哪个 |
    # | docstring      | 工具描述 description（就是提示词！） | 模型据此决定要不要调 |
    # | 类型注解 a: float | JSON Schema 的 type: number     | 供应商接口校验参数 |
    # | 参数名 a        | JSON Schema 的 properties 键名    | 模型据此填 args    |
    # | 默认值 b: float = 1 | Schema 里标记为非必填（required 不含它） | 模型可以省略该参数 |

所以三条实用结论：

    1. docstring 是写给模型看的提示词，不是写给人看的注释 —— 越具体，调用越准；
    2. 类型注解决定参数的 JSON 类型，写错注解（比如该 float 写成 str）模型就会传错；
    3. 返回给模型的是函数的返回值本身，模型看不到函数体，也看不到 print 的内容
       （print 只出现在你自己的控制台）。

课案出处：Agent 课案 → langChain → 核心组件 → 工具

前置条件：
    - 根目录 `.env` 的 `API_KEY` / `BASE_URL` / `MODEL_NAME` 已填好；
    - 不需要数据库。两个工具都是本地函数，不联网。

运行方式（在项目根目录下执行，否则 import 不到 `config`）：
    uv run Agent/02_langchain/04_工具_jxsd.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import json

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from config import settings


# ---------- 1. 课案原文的工具定义 ----------
@tool
def add(a: float, b: float) -> float:
    """返回 a + b 的结果"""
    # 这一行是课案原文：工具每被调用一次就会在控制台打印一次 "12"，
    # 它是「工具确实被执行了」的最直接证据（模型是看不到这个 print 的）。
    print("12")
    # 注意这里是 +1：故意算错，用来证明最终答案来自工具，而不是模型自己心算的。
    return a + b + 1


# ---------- 2. 再定义一个参数更多的工具，观察 schema 怎么长出来 ----------
@tool
def book_ticket(city: str, count: int = 1, seat: str = "二等座") -> str:
    """预订火车票。city：目的地城市；count：张数，默认 1 张；seat：席别，默认二等座"""
    return f"已为你预订 {city} 的 {seat} {count} 张"


llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

agent = create_agent(model=llm, tools=[add])


if __name__ == "__main__":
    # ---------- 3. 工具对象自省：看看 @tool 到底生成了什么 ----------
    # 「自省」= 不问模型、直接问工具对象本身，把上面那张对照表的右列读出来。
    # 这三行是排查「模型为什么没调我的工具」的第一现场：name 必须唯一、description 必须具体。
    print("===== 3. @tool 从函数里抽出来的东西 =====")
    print("工具名 name           ：", add.name)
    print("工具描述 description  ：", add.description)      # ← 就是 docstring
    print("入参模型 args_schema  ：", add.args_schema.model_json_schema())

    # 模型真正看到的就是这段 JSON Schema（发给供应商接口的 tools 字段）
    print("\n发给模型的完整工具 schema：")
    print(json.dumps(add.tool_call_schema.model_json_schema(), ensure_ascii=False, indent=2))

    print("\n带默认值的工具 schema（count / seat 不在 required 里）：")
    schema = book_ticket.tool_call_schema.model_json_schema()
    print("  properties:", list(schema["properties"].keys()))
    print("  required  :", schema.get("required"))
    print("  类型注解 →  JSON 类型：",
          {k: v.get("type") for k, v in schema["properties"].items()})

    # ---------- 4. 工具首先是普通函数：可以直接 invoke ----------
    print("\n===== 4. 直接调用工具（不经过模型） =====")
    print("add.invoke({'a': 3, 'b': 5}) =", add.invoke({"a": 3, "b": 5}))   # 3+5+1 = 9

    # ---------- 5. 让智能体调用它：验证答案来自工具 ----------
    print("\n===== 5. 交给智能体调用（课案原文写法） =====")
    result = agent.invoke({"messages": [{"role": "user", "content": "3加5等于多少"}]})
    for message in result["messages"]:
        print(f"  {type(message).__name__:<14} {str(message.content)[:70]}")

    # 如果模型自己心算，答案会是 8；实际拿到 9，说明数字确实来自 add 的返回值。
    print("\n最终答复：", result["messages"][-1].content)

    # ---------- 6. 带默认值的参数，模型可以省略 ----------
    print("\n===== 6. 让智能体用带默认值的工具 =====")
    agent2 = create_agent(model=llm, tools=[book_ticket])
    result2 = agent2.invoke({"messages": [{"role": "user", "content": "帮我订一张去北京的票"}]})
    print("最终答复：", result2["messages"][-1].content)

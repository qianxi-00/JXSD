# -*- coding: utf-8 -*-
"""
Function Call ①（课案完整版）：工具函数定义集合（tools_jxsd.py）
================================================================
课案原文的 tools.py 只有 4 个数学工具；本文件在**原样保留**它们的基础上，
把课案《参数类型》小节里出现的 8 种 JSON Schema 参数类型，逐个落成
**真实可执行的 Python 工具函数**，让「Schema 怎么写」和「函数怎么收参」对得上。

为什么要拆成「函数」和「描述」两个文件？
    - 模型看不到函数代码，它只能读到一份 JSON 描述（见同目录 tool_desc_jxsd.py）；
    - 程序真正执行的却是这里的 Python 函数。
    两边靠**函数名**对齐：模型返回的 tool_call.function.name == "mul_tool"，
    程序再按这个名字从 TOOL_REGISTRY 里取出真函数来执行。
    名字对不上，就会出现「模型说要查天气，程序却找不到 get_weather」这类错误。

与相邻文件的关系：
    tools_jxsd.py       —— 工具本体（本文件，纯 Python，不依赖任何框架）
    tool_desc_jxsd.py   —— 工具描述（JSON Schema）+ list_tools / call_tool
    agent_openai_jxsd.py    —— 用原生 OpenAI SDK 手写主循环调用这些工具
    agent_langchain_jxsd.py —— 换成 @tool 装饰器 + create_agent 自动调用
    agent_deepagents_jxsd.py—— 同一个任务换成 create_deep_agent

课案出处：Agent 课案 → 工具调用 → function call → 实例 / 参数类型

运行方式：
    uv run Agent/04_function_call/tools_jxsd.py    # 直接运行 = 本地逐个自测，不联网、不花钱
前置条件：
    - 依赖：**只用 Python 标准库（json）**，不 import 任何框架、不 import config，
      所以它是本章唯一「零依赖、零配置、零费用」的文件，可放心反复跑。
    - 不需要 .env、不需要 API Key、不需要数据库。
    - 后三个文件（agent_openai / agent_langchain / agent_deepagents）才是真调模型的部分，
      它们需要根目录 .env 配好 MODEL_NAME / API_KEY / BASE_URL。

课案《概念》小节的两句定义（整章的总纲，后续四个文件都在实现它）：
    「Function Call 是大模型调用外部工具的标准化协议，现在改名为 Tool Call。」
    「Function Call 允许大模型根据用户输入自动决定是否需要调用外部工具，
      并生成相应的函数调用。这是实现 Agent 功能的核心机制。」
    —— 拆开看，这里面藏着三个「谁」：
       谁决定 → 大模型自己（不是我们写 if/else 判断）；
       决定什么 → 调哪个工具、参数填什么（以结构化 JSON 的形式给出，不是文字描述）；
       谁执行 → **本文件的 Python 函数**（模型只写「申请书」，真正干活的是程序）。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import json


# ================================================================
# 一、课案《实例》里的数学四则工具（课案原文，一字未改）
# ================================================================
# 课案原话：「下面是一个完整的实战案例，展示如何使用 Function Call
# 实现一个数学计算 Agent。」
# 这 4 个函数刻意**不带类型注解、不带 @tool 装饰器**，就是最普通的 Python 函数——
# 目的是先讲清最朴素的形态：工具本质上只是「一个能被程序按名字调用的函数」，
# 框架提供的装饰器只是帮你自动生成描述而已。

def add_tool(a, b):
    """
    返回a+b的结果
    :param a:第一个数字
    :param b:第二个数字
    :return: a+b的结果
    """
    return a + b


def sub_tool(a, b):
    """
    返回a-b的结果
    :param a:第一个数字
    :param b:第二个数字
    :return: a-b的结果
    """
    return a - b


def mul_tool(a, b):
    """
    返回a*b的结果
    :param a:第一个数字
    :param b:第二个数字
    :return: a*b的结果
    """
    return a * b


def div_tool(a, b):
    """
    返回a/b的结果
    :param a:第一个数字
    :param b:第二个数字
    :return: a/b的结果
    """
    return a / b


# ================================================================
# 二、课案《参数类型》小节：8 种 JSON Schema 类型对应的真实函数
# ================================================================
# 课案《参数类型》小节开头原话：「OpenAI 的 tools 参数使用 JSON Schema 来定义函数参数，
# 支持以下类型」。这句里有两个信息点，后面每个函数都会用到：
#   1. 参数用的是 **JSON Schema** 这套公开标准，不是 OpenAI 自创格式 ——
#      所以 type / properties / required / items / enum 这些关键字的含义
#      可以去 JSON Schema 规范里查，模型侧也是按这套标准解析的；
#   2. 「支持以下类型」= 模型能被约束到的取值范围。类型写得越准，
#      模型乱填的概率越低（写 number 它会给你 4.5，写 integer 它才只给 4）。
# 课案《参数类型》小节给出了 8 个 Schema 片段（string / number / integer /
# boolean / array / object / null / 完整示例），但片段只是「描述」，
# 没有对应的函数——这里把它们逐一落成真函数，方便对照：
#
#   | 课案小节        | JSON Schema 类型              | 本文件的工具函数            |
#   |----------------|-------------------------------|----------------------------|
#   | 1. string      | "string"                      | format_username_tool       |
#   | 2. number      | "number"（整数或浮点）         | calc_price_tool            |
#   | 3. integer     | "integer"                     | check_stock_tool           |
#   | 4. boolean     | "boolean"                     | toggle_feature_tool        |
#   | 5. array       | "array" + items               | summarize_tags_tool        |
#   | 6. object      | "object" + properties         | describe_user_tool（含嵌套）|
#   | 7. null        | ["string", "null"]            | set_nickname_tool          |
#   | 8. 完整示例     | 多类型组合（create_user）       | create_user_tool           |
#   | 课案 agent.py 的 enum 片段 | "string" + enum   | convert_temperature_tool   |
#
# 为什么每个函数都带类型注解（a: str / price: float）？因为**注解不是给 Python 看的**——
# Python 运行时不校验它。真正的用途在 agent_langchain_jxsd.py：@tool 装饰器靠
# 形参名 + 注解 + docstring 自动生成那份 JSON Schema，注解就是「properties 的 type」。
# 所以本文件的注解和 tool_desc_jxsd.py 里手写的 Schema 是**一回事的两种写法**。
#
# 命名沿用课案的工具后缀约定：一律以 `_tool` 结尾。
# tool_desc_jxsd.list_tools() 正是靠这个后缀（dir() + endswith("_tool")）扫出全部工具的。
# ⚠️ 反过来说：本文件里**任何以 _tool 结尾的顶层名字都会被自动当成工具发出去**，
#    包括意外的临时函数——加函数前先想清楚它该不该被模型看见。

# ---------- 2.1 string：字符串类型 ----------
def format_username_tool(username: str) -> str:
    """
    规范化用户名：去掉首尾空格、转小写、首字母大写
    :param username: 用户名（string 类型）
    :return: 规范化后的用户名
    """
    # 真实业务里「规范化」比「原样返回」更能看出参数确实传进来了
    return username.strip().lower().capitalize()


# ---------- 2.2 number：数字类型（整数或浮点数） ----------
def calc_price_tool(price: float) -> str:
    """
    计算商品的含税价格（税率 13%）
    :param price: 商品价格（number 类型，整数或小数都可以）
    :return: 含税价格的文字说明
    """
    total = round(price * 1.13, 2)
    return f"价格 {price} 元，含税（13%）后 {total} 元"


# ---------- 2.3 integer：整数类型 ----------
def check_stock_tool(count: int) -> str:
    """
    检查库存是否充足
    :param count: 商品数量（integer 类型，只接受整数）
    :return: 库存状态说明
    """
    if count >= 10:
        return f"库存 {count} 件，充足"
    return f"库存 {count} 件，偏少，建议补货"


# ---------- 2.4 boolean：布尔值 ----------
def toggle_feature_tool(is_enabled: bool) -> str:
    """
    开启或关闭某个功能开关
    :param is_enabled: 是否启用（boolean 类型，只能是 true / false）
    :return: 开关状态说明
    """
    return f"功能已{'开启' if is_enabled else '关闭'}"


# ---------- 2.5 array：数组类型（需指定 items 元素类型） ----------
def summarize_tags_tool(tags: list[str]) -> str:
    """
    统计标签：去重后按字典序输出
    :param tags: 标签列表（array 类型，元素是 string）
    :return: 标签统计结果
    """
    unique = sorted(set(tags))
    return f"共传入 {len(tags)} 个标签，去重后 {len(unique)} 个：{'、'.join(unique)}"


# ---------- 2.6 object：对象类型（properties 里再套 properties 就是嵌套对象） ----------
def describe_user_tool(user: dict) -> str:
    """
    把用户信息渲染成一句话（user 里可以带嵌套的 address 对象）
    :param user: 用户信息对象，形如 {"name": "张三", "age": 28,
                 "address": {"city": "南昌", "zip": "330000"}}
    :return: 用户描述
    """
    # 对象参数到了 Python 这边就是 dict，取值时务必用 .get 兜底：
    # 模型偶尔会漏字段，直接 user["age"] 会 KeyError 把主循环炸掉
    name = user.get("name", "匿名用户")
    age = user.get("age", "未知")
    address = user.get("address") or {}
    city = address.get("city", "未填写")
    zip_code = address.get("zip", "未填写")
    return f"{name}，{age} 岁，所在城市 {city}，邮编 {zip_code}"


# ---------- 2.7 null：空值（常用于可选字段） ----------
def set_nickname_tool(optional_field: str | None = None) -> str:
    """
    设置昵称，允许传空（null）
    :param optional_field: 昵称，可以为 null（type 写成 ["string", "null"]）
    :return: 设置结果说明
    """
    if optional_field is None:
        return "昵称已清空（模型传入的是 null）"
    return f"昵称已设置为：{optional_field}"


# ---------- 2.8 enum：枚举（课案 agent.py 里那处 enum 片段的正确用法） ----------
def convert_temperature_tool(value: float, unit: str = "celsius") -> str:
    """
    温度单位换算（摄氏度 <-> 华氏度）
    :param value: 温度数值（number 类型）
    :param unit: 传入数值的单位，只能是 "celsius" 或 "fahrenheit"（enum 枚举，string 类型）
    :return: 换算结果
    """
    # enum 的作用就是**把取值限定在固定几个里**，模型只能从中挑一个；
    # 函数这边仍然要按普通字符串处理，不能假设模型一定守规矩
    if unit == "celsius":
        return f"{value}°C = {round(value * 9 / 5 + 32, 1)}°F"
    if unit == "fahrenheit":
        return f"{value}°F = {round((value - 32) * 5 / 9, 1)}°C"
    return f"不认识单位 {unit}，只支持 celsius / fahrenheit"


# ---------- 2.9 课案《8. 完整示例》的 create_user，落成真函数 ----------
def create_user_tool(name: str, email: str, age: int = 0,
                     is_active: bool = True, tags: list[str] | None = None) -> str:
    """
    创建用户（对应课案《参数类型 → 8. 完整示例》里的 create_user）
    :param name: 用户名（string）
    :param email: 邮箱（string）
    :param age: 年龄（integer）
    :param is_active: 是否激活（boolean）
    :param tags: 标签（array，元素是 string）
    :return: 创建结果的 JSON 字符串
    """
    user = {
        "name": name,
        "email": email,
        "age": age,
        "is_active": is_active,
        "tags": tags or [],
    }
    # 返回 JSON 字符串而不是 dict：工具结果最终要拼进 messages 的 content 字段，
    # 那里只接受字符串
    return json.dumps({"已创建用户": user}, ensure_ascii=False)


# ================================================================
# 三、工具注册表：模型报出函数名后，程序靠它找到真函数
# ================================================================
# 课案原话（关键点说明 3）：「正确处理 message.tool_calls，记录到历史并调用实际工具」。
# 白名单式注册表还有一个安全作用：模型只能调到这里列出的函数，
# 不可能通过伪造名字让程序执行任意代码（比如 getattr 到 os.system）。

# 数学四则工具：课案的「实例」只用这 4 个
MATH_TOOL_NAMES = ["add_tool", "sub_tool", "mul_tool", "div_tool"]

# 参数类型示范工具：课案《参数类型》小节那 8 种类型 + enum
PARAM_TOOL_NAMES = [
    "format_username_tool",
    "calc_price_tool",
    "check_stock_tool",
    "toggle_feature_tool",
    "summarize_tags_tool",
    "describe_user_tool",
    "set_nickname_tool",
    "convert_temperature_tool",
    "create_user_tool",
]

TOOL_REGISTRY = {
    "add_tool": add_tool,
    "sub_tool": sub_tool,
    "mul_tool": mul_tool,
    "div_tool": div_tool,
    "format_username_tool": format_username_tool,
    "calc_price_tool": calc_price_tool,
    "check_stock_tool": check_stock_tool,
    "toggle_feature_tool": toggle_feature_tool,
    "summarize_tags_tool": summarize_tags_tool,
    "describe_user_tool": describe_user_tool,
    "set_nickname_tool": set_nickname_tool,
    "convert_temperature_tool": convert_temperature_tool,
    "create_user_tool": create_user_tool,
}


# ================================================================
# 四、本地自测：直接运行本文件时，把每个工具都真调一遍
# ================================================================
# 意义：先把「函数本身没问题」这件事验证掉。
# 后面主循环出问题时，才能确定是「模型没调工具」而不是「函数写错了」。

if __name__ == "__main__":
    print("=" * 62)
    print("一、课案《实例》的数学四则工具")
    print("=" * 62)
    print(f"add_tool(2, 3)   -> {add_tool(2, 3)}")      # 5
    print(f"sub_tool(10, 4)  -> {sub_tool(10, 4)}")     # 6
    print(f"mul_tool(4, 6)   -> {mul_tool(4, 6)}")      # 24
    print(f"div_tool(24, 6)  -> {div_tool(24, 6)}")     # 4.0

    print()
    print("=" * 62)
    print("二、课案《参数类型》的 8 种类型 → 真实函数")
    print("=" * 62)
    print(f"[string ] format_username_tool('  ZHANGsan ') -> {format_username_tool('  ZHANGsan ')}")
    print(f"[number ] calc_price_tool(100)                 -> {calc_price_tool(100)}")
    print(f"[integer] check_stock_tool(3)                  -> {check_stock_tool(3)}")
    print(f"[boolean] toggle_feature_tool(True)            -> {toggle_feature_tool(True)}")
    print(f"[array  ] summarize_tags_tool(['b', 'a', 'b']) -> {summarize_tags_tool(['b', 'a', 'b'])}")
    print(f"[object ] describe_user_tool(嵌套 address)     -> "
          f"{describe_user_tool({'name': '张三', 'age': 28, 'address': {'city': '南昌', 'zip': '330000'}})}")
    print(f"[null   ] set_nickname_tool(None)              -> {set_nickname_tool(None)}")
    print(f"[enum   ] convert_temperature_tool(25, 'celsius') -> {convert_temperature_tool(25, 'celsius')}")
    print(f"[多类型 ] create_user_tool(...)                -> "
          f"{create_user_tool('张三', 'zhangsan@example.com', 28, True, ['学生', '篮球'])}")

    print()
    print("=" * 62)
    print("三、工具注册表 TOOL_REGISTRY")
    print("=" * 62)
    print(f"数学工具：{MATH_TOOL_NAMES}")
    print(f"参数示范工具：{PARAM_TOOL_NAMES}")
    print(f"注册表共 {len(TOOL_REGISTRY)} 个工具，"
          f"名字与函数是否全部对齐：{all(globals()[n] is f for n, f in TOOL_REGISTRY.items())}")
    print()
    print("下一步：看 tool_desc_jxsd.py（工具描述 / JSON Schema），再看 agent_openai_jxsd.py（主循环）。")


# ================================================================
# 五、实测结论 · 与本课案的差异 · 踩坑提示
# ================================================================
# 【实测结论】直接运行本文件，13 个工具全部一次通过，输出与注释里的预期值一致：
#   add_tool(2, 3) = 5 / sub_tool(10, 4) = 6 / mul_tool(4, 6) = 24 / div_tool(24, 6) = 4.0
#   format_username_tool('  ZHANGsan ') → 'Zhangsan'（strip → lower → capitalize 三步都生效）
#   summarize_tags_tool(['b','a','b']) → 共 3 个，去重后 2 个：a、b
#   最后那行 `all(globals()[n] is f for n, f in TOOL_REGISTRY.items())` 打印 True，
#   证明注册表里 13 个名字与函数对象**全部对齐**（没写错名字、没漏注册）。
#
# 【与本课案的差异】
#   1. 课案《实例》的 tools.py 只有 4 个数学工具（add/sub/mul/div），
#      且**没有类型注解、没有 @tool 装饰器**——本文件原样保留了这 4 个函数
#      （连 docstring 格式都没改），另加 9 个参数类型示范工具。
#      加它们的原因：课案《参数类型》小节只给了 Schema 片段，没有函数，
#      学员看不出「Schema 的 string 到 Python 这边到底是什么类型」。
#   2. 课案没有 TOOL_REGISTRY 这个注册表（它的 tool_desc.call_tool 用 getattr 按名字取），
#      本文件加注册表是为了给 agent_openai_jxsd.py 的白名单式分发提供数据源，
#      同时避免「模型报个名字就能 getattr 到任意函数」的风险。
#   3. 课案的 div_tool 没有除零保护，本文件保持一致（保护放在 tool_desc.call_tool 里兜底），
#      这样「课案原文的写法会怎样出错」可以在第 ④ 节真看到，而不是被提前掩盖。
#
# 【踩坑提示】
#   1. `dir()` 扫出来的是**按字母序**的模块成员，所以 list_tools() 的顺序是字母序，
#      不是定义顺序。想固定顺序（比如让模型优先看到常用工具）得自己排序，别依赖定义次序。
#   2. 函数名是模型和程序之间**唯一的对齐键**。改了函数名忘了改 Schema，
#      现象是「模型说调用 get_weather，程序回一句『没有名为 get_weather 的工具』」——
#      排查时先比对两边名字，再看参数。
#   3. `div_tool(1, 0)` 会抛 ZeroDivisionError。本文件的工具是「裸函数」，
#      异常会直接抛给调用方；官方主循环里靠 tool_desc.call_tool 的 try/except 兜住，
#      再把错误信息当工具结果回传给模型 —— 这是 Agent 能「自愈」的机制，不是掩盖错误。
#   4. 对象参数（describe_user_tool 的 user: dict）到 Python 这边就是 dict，
#      取值务必用 .get 兜底（第 2.6 节已注明）：模型偶尔会漏字段，
#      直接 user["age"] 会 KeyError 把整个主循环炸掉。
#   5. `list[str] | None = None` 这种写法要求 Python 3.10+；
#      课案环境是 3.13，本机 .venv 同样满足，老环境上要改回 Optional[List[str]]。

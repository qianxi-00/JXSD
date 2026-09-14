# -*- coding: utf-8 -*-
"""
Function Call ②（课案完整版）：工具描述与 JSON Schema（tool_desc_jxsd.py）
================================================================
Function Call 的第二件套：把工具「翻译」成模型能读懂的 JSON Schema。
模型看不到 tools_jxsd.py 里的函数代码，它看到的只有这份描述；
**描述写得好坏，直接决定调用准确率**（少写一个 description，
模型就可能永远想不起来该调这个工具）。

本文件比课案原文多做了三件事：
    1. 课案的 list_tools() / call_tool() 原样保留（含中文键名 工具名 / 工具描述）；
    2. 课案《参数类型》小节的 8 个 Schema 片段全部保留成常量，逐个加中文注释；
    3. 把「片段」组装成「完整的 parameters」——课案里 parameters 是写死的
       {a: number, b: number}，只能给 4 个数学工具用，这里改成按工具查表。

OpenAI tools 参数的三层结构（课案的 tools 定义就是这三层）：
    {"type": "function",                        # 第一层：固定值，表示这是一个函数工具
     "function": {
         "name": "mul_tool",                    # 第二层：函数名，模型据此指名调用
         "description": "返回a*b的结果",         #         干什么用，模型据此决定何时调用
         "parameters": {                        # 第三层：参数的 JSON Schema，模型据此填参
             "type": "object",
             "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
             "required": ["a", "b"]
         },
         "strict": True                         # 开启后模型必须生成合法 JSON
     }}

课案出处：Agent 课案 → 工具调用 → function call → 实例（tool_desc.py）/ 参数类型

运行方式：
    uv run Agent/04_function_call/tool_desc_jxsd.py   # 打印全部 Schema 并演示取工具，不联网
前置条件：
    - 依赖：只用标准库（json）+ 同目录的 tools_jxsd.py，**不 import 任何框架、不 import config**。
    - 不需要 .env、不需要 API Key、不需要数据库；本文件全程不发起网络请求。
    - 因为它要 `import tools_jxsd`，所以**必须在 04_function_call 目录下运行**
      （或用 uv run 的脚本方式，见上面的运行命令）；从别的目录直接跑会 ModuleNotFoundError。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import json

import tools_jxsd


# ================================================================
# 一、课案的 tool_desc.py（原文照搬，只把 import 的模块名换成 _jxsd 版）
# ================================================================
# 课案原文：
#     import tools
#
#     def list_tools():
#         """列出tools.py中所有的工具"""
#         return [{"工具名": func, "工具描述": getattr(tools, func).__doc__}
#                 for func in dir(tools) if func.endswith("_tool")]
#
#     def call_tool(tool_name, *args, **kwargs):
#         """调用工具"""
#         return getattr(tools, tool_name)(*args, **kwargs)

def list_tools(only: list[str] | None = None) -> list[dict]:
    """
    列出 tools_jxsd.py 中所有的工具

    :param only: 可选，只列这几个工具名；不传 = 列出全部（课案原行为）
    :return: [{"工具名": ..., "工具描述": ..., "参数": ...}, ...]

    两点说明：
        1. 课案用 dir() + endswith("_tool") 扫描模块，这正是**命名后缀约定**的意义——
           只要函数叫 xxx_tool，就会被自动收集，不需要手工维护清单；
        2. "参数" 这个键是课案原文没有、但主循环需要的：课案在 agent.py 里把
           parameters 写死成 {"a": number, "b": number}，那是给 4 个数学工具专用的；
           本文件工具变多了，所以改成按工具名查表（见下方的 TOOL_PARAMETERS）。
        3. "工具描述" 取的是整个 __doc__（课案原样），因此连 :param / :return 那几行
           也一起发给了模型。实战里可以只取第一行做精简描述，这里保持课案原样子，
           方便对照「模型实际看到的是什么」。
    """
    names = [func for func in dir(tools_jxsd) if func.endswith("_tool")]
    if only is not None:
        # 排序保证顺序稳定：dir() 返回的列表是按字母序的，过滤后仍然是字母序
        names = [func for func in names if func in only]
    return [
        {
            "工具名": name,
            "工具描述": getattr(tools_jxsd, name).__doc__,
            "参数": TOOL_PARAMETERS.get(name, EMPTY_PARAMETERS),
        }
        for name in names
    ]


def call_tool(tool_name, *args, **kwargs):
    """
    调用工具：按名字从 tools_jxsd 模块里取出真函数并执行

    课案原文是 `return getattr(tools, tool_name)(*args, **kwargs)`，一行搞定。
    本文件多包了两层保护，因为**参数是模型生成的，不可信**：
        1. 名字白名单：模型可能报出一个不存在的函数名，getattr 会抛 AttributeError；
           更危险的是，不设白名单时模型若能控制名字，就可能 getattr 到 os.system 这类函数；
        2. 执行兜底：模型可能给出 div_tool(a=1, b=0) 这种非法参数。
    这两类错误都不该让主循环崩掉——把错误当成「工具执行结果」回传给模型，
    模型下一轮往往能自己纠正（这正是 Agent 自愈能力的来源）。
    """
    func = tools_jxsd.TOOL_REGISTRY.get(tool_name)
    if func is None:
        return f"没有名为 {tool_name} 的工具"
    try:
        return func(*args, **kwargs)
    except Exception as exc:                      # noqa: BLE001 —— 故意兜住全部异常，回传给模型
        return f"工具 {tool_name} 执行出错：{type(exc).__name__}: {exc}"


# ================================================================
# 二、课案《参数类型》小节的 8 个 Schema 片段（原文保留 + 逐字段中文注释）
# ================================================================
# 先明确一个容易混的点：下面这些片段描述的是**一个参数**，不是整套 parameters。
# 它们的 "name" 键不属于 JSON Schema 标准，是课案为了讲清「这个片段对应哪个参数名」
# 而人工加上的标记；真正拼进 parameters.properties 时，name 会变成键名（见第三节）。

# ---------- 1. string —— 字符串类型 ----------
SCHEMA_STRING = {
    "name": "username",
    "type": "string",
    "description": "用户名",
}
#   name        ：参数名。必须和 Python 函数的形参名一致，
#                 否则模型填对了名字、程序却接不住（call_tool 会报 unexpected keyword）
#   type        ：JSON Schema 的类型关键字，"string" = 字符串
#   description ：写给模型看的自然语言说明。这是**最影响准确率**的一行，
#                 课案这里只写了「用户名」，实战里应写得更具体，例如「用户名，2-20 个字符」

# ---------- 2. number —— 数字类型（整数或浮点数） ----------
SCHEMA_NUMBER = {
    "name": "price",
    "type": "number",
    "description": "商品价格",
}
#   number：整数和小数都收。模型传 100 还是 100.5 都合法

# ---------- 3. integer —— 整数类型 ----------
SCHEMA_INTEGER = {
    "name": "count",
    "type": "integer",
    "description": "商品数量",
}
#   integer：只收整数。价格用 number、数量用 integer，这是在「约束模型别乱填」

# ---------- 4. boolean —— 布尔值 ----------
SCHEMA_BOOLEAN = {
    "name": "is_enabled",
    "type": "boolean",
    "description": "是否启用",
}
#   boolean：只有 true / false 两个取值。参数名用 is_ / has_ 开头是惯例，
#           模型看到 is_enabled 就知道该填 true 或 false，而不是填「是」

# ---------- 5. array —— 数组类型（需指定 items 元素类型） ----------
SCHEMA_ARRAY = {
    "name": "tags",
    "type": "array",
    "items": {"type": "string"},
    "description": "标签列表",
}
#   items：数组**必须**声明元素类型，否则模型不知道里面该放字符串还是对象。
#          课案这里 items 是 string，于是模型只会生成 ["学生", "篮球"] 这种；
#          若要对象数组，就写成 "items": {"type": "object", "properties": {...}}

# ---------- 6. object —— 对象类型（需指定 properties） ----------
SCHEMA_OBJECT = {
    "name": "user",
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "description": "用户信息",
}
#   properties：对象里有哪些字段、各是什么类型。
#               Python 那边收到的是一个 dict：{"name": "张三", "age": 28}
#   嵌套对象：properties 里再写一个 type=object 的字段就是嵌套，
#            见本文件 2.10 的 SCHEMA_OBJECT_NESTED

# ---------- 7. null —— 空值（常用于可选字段） ----------
SCHEMA_NULL = {
    "name": "optional_field",
    "type": ["string", "null"],   # 可以为 string 或 null
    "description": "可选字段",
}
#   type 写成**列表**表示多选一：["string", "null"] = 要么字符串、要么 null。
#   这是 JSON Schema 表达「可选字段」的标准写法（OpenAI 严格模式下，
#   可选参数就是靠它实现的：字段仍然出现在 required 里，但允许模型传 null）。

# ---------- 8. 完整示例 —— 课案 3738-3761 的 create_user ----------
# 注意：这个片段跟前 7 个不同，它是一个**完整的 tools 元素**（含 type/function 两层），
# 可以直接塞进 client.chat.completions.create(tools=[SCHEMA_CREATE_USER]) 里用。
SCHEMA_CREATE_USER = {
    "type": "function",
    "function": {
        "name": "create_user",
        "description": "创建用户",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "用户名"},
                "age": {"type": "integer", "description": "年龄"},
                "email": {"type": "string", "description": "邮箱"},
                "is_active": {"type": "boolean", "description": "是否激活"},
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "标签",
                },
            },
            "required": ["name", "email"],
        },
    },
}
#   required：**必填字段清单**。这里只有 name / email 必填，age / is_active / tags 可省。
#             必填与可选的划分很关键：全必填会让模型硬编造参数，全可选又容易漏填。

# ---------- 补充 A：enum 枚举（课案 agent.py 里那处 enum 片段的正确写法） ----------
SCHEMA_ENUM_VALUE = {
    "name": "value",
    "type": "number",
    "description": "温度数值",
}
SCHEMA_ENUM_UNIT = {
    "name": "unit",
    "type": "string",
    "enum": ["celsius", "fahrenheit"],
    "description": "传入数值的单位",
}
#   enum：把取值**限定在固定几个选项里**，模型只能从中挑一个，能极大降低乱填。
#   ⚠️ 课案 agent.py 里把这处 enum 挂在了 number 类型上，且没有参数名：
#         "b": {"type": "number", "enum": ["celsius", "fahrenheit"]}
#      这是课案原文的笔误（枚举值是字符串，类型却写 number）。
#      正确写法是 type: "string" + enum: [...]，也就是上面 SCHEMA_ENUM_UNIT 的样子。

# ---------- 补充 B：嵌套对象（object 里再套 object） ----------
SCHEMA_OBJECT_NESTED = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "用户名"},
        "age": {"type": "integer", "description": "年龄"},
        "address": {
            "type": "object",
            "description": "地址（对象里再套一个对象）",
            "properties": {
                "city": {"type": "string", "description": "城市"},
                "zip": {"type": "string", "description": "邮编"},
            },
            "required": ["city", "zip"],
            # additionalProperties：禁止出现 Schema 里没声明的字段。
            # 它和 strict 是一对：开了 strict，每个 object 都得写上这一句，
            # 否则模型仍可能自由发挥出多余字段。
            "additionalProperties": False,
        },
    },
    "required": ["name", "age", "address"],
    "additionalProperties": False,
}


# ================================================================
# 三、把「片段」组装成「完整的 parameters」
# ================================================================
# 组装规则就一条：片段的 name → properties 的键；片段其余字段 → 该键的值。
# 也就是说片段里的 "name" 键在组装后要**去掉**，否则 "name" 会变成对象里一个
# 名叫 name 的字段，模型就会多填一个莫名其妙的参数。

def build_parameters(*fields: dict, required: list[str] | None = None) -> dict:
    """
    把若干「参数片段」组装成一个完整的 parameters（JSON Schema object）

    :param fields: 参数片段，例如 SCHEMA_STRING、SCHEMA_NUMBER
    :param required: 必填字段名列表；不传 = 所有字段都必填
    :return: {"type": "object", "properties": {...}, "required": [...], "additionalProperties": False}
    """
    properties = {}
    for field in fields:
        name = field["name"]
        # 去掉非标准的 name 键，其余字段原样保留（type / description / enum / items / properties……）
        properties[name] = {key: value for key, value in field.items() if key != "name"}
    return {
        "type": "object",
        "properties": properties,
        # 默认全必填：OpenAI 严格模式要求 required 覆盖 properties 的全部字段，
        # 「可选」改用 type: ["string", "null"] 表达（见 SCHEMA_NULL）
        "required": list(required) if required is not None else list(properties),
        "additionalProperties": False,
    }


# 没有描述的工具走这个兜底 Schema：一个不接任何参数的 object
EMPTY_PARAMETERS = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

# 课案 agent.py 给 4 个数学工具写死的参数：a、b 都是 number
SCHEMA_MATH_A = {"name": "a", "type": "number", "description": "第一个数字"}
SCHEMA_MATH_B = {"name": "b", "type": "number", "description": "第二个数字"}

# 工具名 → 完整 parameters。键必须和 tools_jxsd.py 里的函数名严格一致
TOOL_PARAMETERS = {
    # 数学四则：对应课案 agent.py 里写死的 {"a": number, "b": number}
    **{name: build_parameters(SCHEMA_MATH_A, SCHEMA_MATH_B) for name in tools_jxsd.MATH_TOOL_NAMES},
    # 课案《参数类型》8 种类型，逐个落到 tools_jxsd.py 的真函数上
    "format_username_tool": build_parameters(SCHEMA_STRING),
    "calc_price_tool": build_parameters(SCHEMA_NUMBER),
    "check_stock_tool": build_parameters(SCHEMA_INTEGER),
    "toggle_feature_tool": build_parameters(SCHEMA_BOOLEAN),
    "summarize_tags_tool": build_parameters(SCHEMA_ARRAY),
    "describe_user_tool": SCHEMA_OBJECT_NESTED,      # 对象 + 嵌套对象
    "set_nickname_tool": build_parameters(SCHEMA_NULL),
    "convert_temperature_tool": build_parameters(SCHEMA_ENUM_VALUE, SCHEMA_ENUM_UNIT),
    # 课案《8. 完整示例》的 create_user：直接取它 parameters 那一段（多类型组合）
    "create_user_tool": SCHEMA_CREATE_USER["function"]["parameters"],
}


# ================================================================
# 四、多工具声明：最终喂给 client.chat.completions.create(tools=...) 的列表
# ================================================================
# 课案 agent.py 用的是**列表推导式动态生成**（关键点说明 1：「动态生成 tools」），
# 原文长这样（注意课案这里有个语法笔误，parameters 那行的 } 后漏了逗号，
# 导致 "strict": True 被写进了 parameters 里面；本文件按正确的结构生成）：
#
#     tools = [{
#         "type": "function",
#         "function": {
#             "name": tool["工具名"],
#             "description": tool["工具描述"],
#             "parameters": {
#                 "type": "object",
#                 "properties": {"a": {"type": "number"},
#                                "b": {"type": "number", "enum": ["celsius", "fahrenheit"]}},
#                 "required": ["a", "b"],
#                 "additionalProperties": False  # 禁止额外字段，配合 strict 使用
#             }
#             "strict": True  # 开启后，模型必须生成合法 JSON   ← 课案原文此处缩进/逗号写错了
#         }
#     } for tool in tool_desc.list_tools()]
#
# 本文件把写死的 parameters 换成 tool["参数"]，这个推导式的骨架和课案完全一致。
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": tool["工具名"],
            "description": tool["工具描述"],
            "parameters": tool["参数"],
            # strict：课案原文写着 True。开启后模型必须生成合法 JSON（结构化输出约束），
            # 但要求每个 object 都带 additionalProperties: False，且 required 覆盖所有字段。
            # 本文件的 Schema 已经满足这两条，所以原样保持 True；
            # 若换成不支持 strict 的服务（部分国产中转不认这个字段），把它改成 False 即可。
            "strict": True,
        },
    }
    for tool in list_tools()
]


# ================================================================
# 五、本地自测：打印全部 Schema，并演示取工具 / 调工具
# ================================================================
if __name__ == "__main__":
    print("=" * 62)
    print("一、课案《参数类型》的 8 个 Schema 片段（原样打印，逐个对照注释看）")
    print("=" * 62)
    fragments = [
        ("1. string", SCHEMA_STRING),
        ("2. number", SCHEMA_NUMBER),
        ("3. integer", SCHEMA_INTEGER),
        ("4. boolean", SCHEMA_BOOLEAN),
        ("5. array", SCHEMA_ARRAY),
        ("6. object", SCHEMA_OBJECT),
        ("7. null", SCHEMA_NULL),
        ("补充 A. enum", SCHEMA_ENUM_UNIT),
    ]
    for title, fragment in fragments:
        # ensure_ascii=False → 中文原样输出（默认 True 会打成 \uXXXX，看不懂）
        print(f"{title}：{json.dumps(fragment, ensure_ascii=False)}")

    print()
    print("8. 完整示例（create_user，整段 tools 元素）：")
    print(json.dumps(SCHEMA_CREATE_USER, ensure_ascii=False, indent=2))

    print()
    print("=" * 62)
    print("二、list_tools()：把片段组装成每个工具的 parameters")
    print("=" * 62)
    for tool in list_tools():
        props = list(tool["参数"]["properties"])
        print(f"{tool['工具名']:<26} 参数={props} 必填={tool['参数']['required']}")

    print()
    print("=" * 62)
    print("三、TOOLS 多工具声明（最终发给模型的东西）")
    print("=" * 62)
    print(f"共 {len(TOOLS)} 个工具：{[t['function']['name'] for t in TOOLS]}")
    print("其中一个工具的完整声明（describe_user_tool，含嵌套对象）：")
    sample = next(t for t in TOOLS if t["function"]["name"] == "describe_user_tool")
    print(json.dumps(sample, ensure_ascii=False, indent=2)[:600] + " ...")

    print()
    print("=" * 62)
    print("四、call_tool()：按名字分发到真函数（含模型乱传参数时的兜底）")
    print("=" * 62)
    print(f"正常调用   call_tool('mul_tool', a=4, b=6)  -> {call_tool('mul_tool', a=4, b=6)}")
    print(f"名字写错   call_tool('mul_tool_typo', a=1)  -> {call_tool('mul_tool_typo', a=1)}")
    print(f"参数非法   call_tool('div_tool', a=1, b=0)  -> {call_tool('div_tool', a=1, b=0)}")
    print()
    print("下一步：agent_openai_jxsd.py 把这些描述发给模型，跑真正的 Function Call 主循环。")


# ================================================================
# 六、实测结论 · 与本课案的差异 · 踩坑提示
# ================================================================
# 【实测结论】直接运行本文件（不联网）可看到：
#   - 8 个 Schema 片段逐个打印，中文原样输出（ensure_ascii=False）；
#   - list_tools() 扫出 13 个工具，每个都带上了正确的 properties / required；
#   - TOOLS 列表里每个元素的 function.name 与 parameters 严格配对；
#   - 第 ④ 节三条 call_tool 演示分别命中三种路径：
#       正常调用 → 24；名字写错 → "没有名为 mul_tool_typo 的工具"；参数非法 → ZeroDivisionError。
#     **这三行输出就是 call_tool 两层保护都在工作的证据**，
#     也是本文件相对课案「getattr 一行搞定」的实际增益。
#
# 【与本课案的差异】
#   1. 课案的 list_tools() 只返回 {工具名, 工具描述} 两个键，**没有「参数」**；
#      课案在 agent.py 里把 parameters 写死成 {"a": number, "b": number}，
#      因此它的写法只能服务 4 个数学工具。本文件加了 "参数" 键 + TOOL_PARAMETERS 查表，
#      工具扩容到 13 个也不用改主循环。
#   2. 课案的 call_tool 是 `return getattr(tools, tool_name)(*args, **kwargs)` 一行。
#      本文件加了两层保护（名字白名单 + 执行兜底），原因写在 call_tool 的 docstring 里：
#      参数是模型生成的，属于不可信输入，不该让主循环崩掉。
#   3. 课案《8. 完整示例》那段 create_user 是**独立于 tools 列表**的一段 Schema 片段；
#      本文件把它接到真函数 create_user_tool 上，让它也进入 tools 列表被调用到。
#   4. 课案 agent.py 的 tools 列表推导式有一处语法笔误（parameters 那行的 } 后漏了逗号，
#      导致 "strict": True 被写进了 parameters 内部）。本文件第 4 节把原文抄下来并标了
#      出错位置，生成时用的是正确结构 —— 用课案原文那份代码时**会直接 SyntaxError**，
#      不是运行期才能发现的问题。
#
# 【踩坑提示】
#   1. `strict: True` 有硬性配套条件：每个 object 都要写 additionalProperties: False，
#      且 required 必须覆盖 properties 的全部字段。少一条，服务端会直接返回 400
#      （报错信息通常只说 schema 不合法，不会告诉你是哪一条）。
#   2. JSON Schema 里**没有 "name" 这个关键字**。本文件片段里的 "name" 是课案为了
#      标注「这个片段对应哪个参数名」而人工加的，组装时必须去掉
#      （build_parameters 第 280 行就是干这个的）；忘了去掉，模型会多填一个叫 name 的字段。
#   3. 参数名必须和 Python 形参名严格一致，否则报 unexpected keyword argument。
#      "工具名 / 工具描述 / 参数" 这几个中文键名是课案的习惯，只在本项目内部用，
#      发给模型前一定要映射成 name / description / parameters（见第 4 节）。
#   4. `__doc__` 取的是**整个 docstring**，包括 :param / :return 那几行，
#      它们会一起发给模型、一起计费。实战里可以只取第一行做精简描述，
#      本文件保持课案原样是为了让学员看到「模型实际收到了什么」。

# -*- coding: utf-8 -*-
"""
Function Call ④（课案完整版）：LangChain 版智能体（agent_langchain_jxsd.py）
================================================================
课案原话：「LangChain 提供了更简洁的 Agent 实现方式。」
对比 agent_openai_jxsd.py 那 80 多行手写循环，本文件只要 create_agent 一行——
工具循环、历史记录、错误处理它全包了。

课案原文的文件结构是两个文件：
    c_agent/day01/function_call/langchain_api/
    ├── agent.py          # Agent 主程序
    └── tools.py          # 工具定义
本目录为了对照方便，把「工具定义」和「Agent 主程序」合在一个文件里，
用 `# ---------- 分节 ----------` 隔开，内容与课案一一对应。

课案给出了**两条路线**，本文件都实现并各跑一遍：

    路线一：create_agent —— 官方推荐的「开箱即用」写法
        agent = create_agent(model=llm, tools=[...], system_prompt="...")
        result = agent.invoke({"messages": [...]})
        优点：工具循环全自动（模型说要调工具，它自己执行、自己回填、自己再问一轮）；
        代价：循环过程被封装了，看不清中间发生了什么（只能回头翻 result["messages"]）。

    路线二：bind_tools —— 把工具「绑」到模型上，循环仍然自己写
        bound = llm.bind_tools([...])          # 模型从此知道有哪些工具可用
        ai = bound.invoke(messages)            # 它会在 ai.tool_calls 里提出调用请求
        ... 自己执行、自己回填 ToolMessage、再 invoke 一轮 ...
        这其实是 agent_openai_jxsd.py 的 LangChain 写法：
        与原生 SDK 相比，arguments 已经从 JSON 字符串变成了现成的 dict。

一句话：create_agent ≈ bind_tools + 一个写好的主循环。

课案出处：Agent 课案 → 工具调用 → function call → langchain

运行方式：
    uv run Agent/04_function_call/agent_langchain_jxsd.py
前置条件：
    - 依赖：`langchain`（1.x，含 langchain.agents.create_agent）+ `langchain-openai`
      + `langchain-core`，本项目已 uv sync 装好。
    - 配置：根目录 .env 配好 MODEL_NAME / API_KEY / BASE_URL ——
      统一 `from config import settings` 读取；课案原文是 ChatOpenAI 直构造，
      本项目统一改成 init_chat_model（换服务商只改 model_provider 一个参数）。
    - 网络：需要 OpenAI 兼容接口。本文件会**真实调用大模型多次**
      （路线一两次 invoke + 路线二的循环），会产生 token 费用。
    - 不需要数据库、不需要起服务；也没有 `input()` 交互，可直接在 CI 里跑。
    - 注意：课案原文是 `langchain_api/agent.py` + `tools.py` 两个文件，
      本文件把两者合在一处（用 `# ---------- 分节 ----------` 隔开），内容一一对应。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import json

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from config import settings

# ---------- 0. 大模型 ----------
# 课案原文用 ChatOpenAI(api_key=..., base_url=..., model=..., temperature=0) 直接构造；
# 本项目的统一写法是 init_chat_model（见 CONVENTIONS 第 3 节），
# 好处是换服务商只改 model_provider 一个参数，其余代码不动
llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


# ================================================================
# 一、课案 tools.py：@tool 装饰器定义工具
# ================================================================
# 与 agent_openai_jxsd.py 里那 4 个裸函数相比，只多了两样东西：
#     1. @tool 装饰器 —— 把普通函数包装成 LangChain 的 BaseTool，
#        自动导出 name / description / args_schema（等价于手写那份 JSON Schema）；
#     2. 类型注解 a: float, b: float -> float —— 类型推断的依据，
#        没有注解，LangChain 生成不出 parameters，模型就不知道该填什么类型。
#
# 对照关系（课案 langchain 小节与「参数类型」小节是同一件事的两种写法）：
#     @tool 的 docstring        →  JSON 的 function.description
#     @tool 的类型注解 + 形参名  →  JSON 的 function.parameters（properties / required）
#     @tool 的函数名            →  JSON 的 function.name

@tool
def add_tool(a: float, b: float) -> float:
    """
    返回a+b的结果
    :param a:第一个数字
    :param b:第二个数字
    :return: a+b的结果
    """
    return a + b


@tool
def sub_tool(a: float, b: float) -> float:
    """
    返回a-b的结果
    :param a:第一个数字
    :param b:第二个数字
    :return: a-b的结果
    """
    return a - b


@tool
def mul_tool(a: float, b: float) -> float:
    """
    返回a*b的结果
    :param a:第一个数字
    :param b:第二个数字
    :return: a*b的结果
    """
    return a * b


@tool
def div_tool(a: float, b: float) -> float:
    """
    返回a/b的结果
    :param a:第一个数字
    :param b:第二个数字
    :return: a/b的结果
    """
    return a / b


# 课案原文：tools_list = [tools.add_tool, tools.sub_tool, tools.mul_tool, tools.div_tool]
tools_list = [add_tool, sub_tool, mul_tool, div_tool]

# 工具名 → 工具对象：bind_tools 路线里，模型只回传工具名，程序得自己找回工具
TOOL_MAP = {t.name: t for t in tools_list}


# ================================================================
# 二、课案 agent.py：create_agent 一行创建 Agent
# ================================================================
# create_agent 返回的是一个 LangGraph 图：model 节点 + tools 节点 + 条件边。
# 模型有 tool_calls 就走向 tools 节点执行，执行完再回到 model 节点，
# 直到模型不再要工具为止——这正是我们在 agent_openai_jxsd.py 里手写的那个循环。
agent = create_agent(
    model=llm,
    tools=tools_list,
    system_prompt="你是一个助手，可以帮助用户进行数学计算。",
)


def build_few_shot_messages() -> list[dict]:
    """
    课案的 Few-shot 历史对话（与 agent_openai_jxsd.py 里那段完全一致）

    注意这里用的是**标准 OpenAI 消息格式**（assistant 带 tool_calls、
    tool 带 tool_call_id），LangChain 能直接吃下去——它会自动转成
    AIMessage / ToolMessage。这也是 LangChain 好用的地方：
    消息格式与 OpenAI 协议基本兼容，不用手写适配层。
    """
    return [
        {"role": "user", "content": "8*2-9"},
        {"role": "assistant", "content": "要计算8*2-9。首先计算8*2。", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "mul_tool", "arguments": json.dumps({"a": 8, "b": 2})}}
        ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "16"},
        {"role": "assistant", "content": "8*2=16，现在计算16-9。", "tool_calls": [
            {"id": "call_2", "type": "function",
             "function": {"name": "sub_tool", "arguments": json.dumps({"a": 16, "b": 9})}}
        ]},
        {"role": "tool", "tool_call_id": "call_2", "content": "7"},
        {"role": "assistant", "content": "16-9=7，所以8*2-9=7"},
    ]


def print_trace(result: dict) -> None:
    """
    课案的打印方式：遍历 result["messages"]，逐条打印角色和内容

    课案原文：
        print(f"\\n最终答案:")
        for msg in result["messages"]:
            print(f"Role: {msg.type}")
            print(f"Content: {msg}")
            print("-" * 50)

    「最终答案」其实只是最后一条消息；中间那些 ai → tool → ai 才是
    create_agent 自动帮我们跑完的工具循环，翻一遍就能看清它做了什么。

    课案原文是 print(f"Content: {msg}")，直接把整条消息对象打出来——
    里面塞满了 response_metadata / token_usage / id，一屏根本看不完。
    这里精简成「角色 + 内容 + 工具调用」三样关键信息（课案想看的东西一样不少）。
    """
    print("最终答案（完整消息链）：")
    for msg in result["messages"]:
        # msg.type 取值：human / ai / tool / system
        #   ai 带 tool_calls    → 模型在要工具
        #   tool               → 框架自动执行工具后的结果回填
        #   最后一条 ai 无 tool_calls → 才是真正的自然语言答案
        print(f"Role: {msg.type}")
        if getattr(msg, "tool_calls", None):
            for call in msg.tool_calls:
                print(f"   → 提出调用 {call['name']}({call['args']})  id={call['id']}")
        if msg.content:
            print(f"Content: {msg.content}")
        print("-" * 50)


# ================================================================
# 三、路线二：bind_tools（手动循环的 LangChain 写法）
# ================================================================
def run_with_bind_tools(user_input: str, max_turns: int = 10) -> str | None:
    """
    把工具绑到模型上，然后自己写循环

    与 agent_openai_jxsd.py 的主循环逐行对照，差别只有三处：
        1. tools 不用手写 JSON Schema —— 直接传 @tool 对象，LangChain 负责转换；
        2. ai.tool_calls 里的 args **已经是 dict**，不用 json.loads（原生 SDK 给的是字符串）；
        3. 回填结果用 ToolMessage 对象，不用手拼 {"role": "tool", ...} 字典。
    循环结构、循环上限、退出条件，三者一模一样。
    """
    bound_llm = llm.bind_tools(tools_list)

    messages = [
        SystemMessage("你是一个助手，可以帮助用户进行数学计算。"),
        HumanMessage(user_input),
    ]

    for _ in range(max_turns):
        ai = bound_llm.invoke(messages)
        messages.append(ai)

        # 模型没提工具调用 → 这就是最终答案
        if not ai.tool_calls:
            print(f"最终答案: {ai.content}")
            return ai.content

        for call in ai.tool_calls:
            # LangChain 已经把工具调用规整成 {"name": ..., "args": {...}, "id": ...}
            # args 直接就是 dict，这是它比原生 SDK 省事的地方
            result = TOOL_MAP[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
            print(f"调用工具: {call['name']}, 参数: {call['args']}, 结果: {result}")
    else:
        print(f"⚠️ 已循环 {max_turns} 次仍未得到最终答案，强制结束（防御上限生效）")
    return None


# ================================================================
# 四、跑起来
# ================================================================
if __name__ == "__main__":
    print("=" * 62)
    print("路线一：create_agent（课案原样，带 Few-shot 历史）")
    print("=" * 62)
    messages = build_few_shot_messages()
    messages.append({"role": "user", "content": "2+4*6"})
    result = agent.invoke({"messages": messages})
    print_trace(result)
    # 课案对应的预期行为：agent.invoke 一次，内部自动完成「模型要工具 → 执行 → 再问」
    # 的往返，最终 result["messages"][-1].content 就是 2+4*6=26
    # ⚠️ 本机实测：带这段 Few-shot 时，当前模型（中转站 grok-4.6）常常直接文字作答、
    #    一个工具都不调——与 agent_openai_jxsd.py 演示 1 遇到的是同一个现象。

    print()
    print("=" * 62)
    print("路线一（干净版）：去掉 Few-shot，同一道题")
    print("=" * 62)
    # 与 agent_openai_jxsd.py 演示 2 的原因相同：当前模型对这段 Few-shot 并不买账，
    # 去掉示例后工具循环反而跑得干净利落
    result = agent.invoke({"messages": [{"role": "user", "content": "2+4*6"}]})
    print_trace(result)

    print()
    print("=" * 62)
    print("路线二：bind_tools（自己写循环，与原生 SDK 逐行对照）")
    print("=" * 62)
    run_with_bind_tools("2+4*6")

    print()
    print("=" * 62)
    print("两条路线对照（课案《与 OpenAI API 实现的对比》表以注释形式保留在源码里）：")
    print("=" * 62)
    print("  create_agent —— 一行创建，工具循环全自动（结果看 result['messages'] 里的 ai→tool→ai）")
    print("  bind_tools   —— 自己写循环；args 已经是 dict，比原生 SDK 少一步 json.loads")
    print("  共同点       —— JSON Schema 不用手写：@tool 把 docstring + 类型注解自动转成 parameters")
    print("  共同点       —— 初始化从 openai.Client() 变成 ChatOpenAI / init_chat_model")


# ================================================================
# 五、课案原文的对比表与关键点说明（以注释形式原样保留）
# ================================================================
# 《与 OpenAI API 实现的对比》：
# | 特性             | OpenAI API                    | LangChain                |
# |-----------------|-------------------------------|--------------------------|
# | 初始化           | openai.Client()               | ChatOpenAI()             |
# | 工具定义         | JSON 格式                      | @tool 装饰器              |
# | 创建 Agent       | 手动循环                       | create_agent()           |
# | 调用方式         | chat.completions.create()     | agent.invoke()           |
# | 处理 tool_calls  | 手动解析                       | 自动处理                  |
#
# 《关键点说明》：
#   1. @tool 装饰器：LangChain 自动将函数转换为工具定义
#   2. 类型注解：a: float, b: float -> float 帮助类型推断
#   3. create_agent：自动处理工具调用、历史记录、错误处理
#   4. messages 格式：使用标准的对话消息格式
#   5. result 返回：包含所有对话历史和最终答案

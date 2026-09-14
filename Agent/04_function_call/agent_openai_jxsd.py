# -*- coding: utf-8 -*-
"""
Function Call ③（课案完整版）：原生 OpenAI SDK 手写 Agent 主循环（agent_openai_jxsd.py）
================================================================
这是本章**最重要的一段代码**：不借助任何框架，把 Function Call 的完整回路手写一遍。
理解了这个循环，就理解了 LangChain 的 create_agent / DeepAgents 的 create_deep_agent
内部到底在替你做什么。

一次完整的 Function Call 往返（课案《概念》小节说的「模型自动决定是否调用外部工具」）：
    ① 把「system 提示 + 历史对话 + 工具描述(tools)」一起发给模型
    ② 模型若判断需要工具，不直接回答，而是返回 message.tool_calls（函数名 + JSON 参数）
    ③ 程序解析 JSON 参数、执行本地函数、把结果包成 {"role": "tool", "tool_call_id": ...} 追加进历史
    ④ 带着工具结果再发一次请求，模型据此继续推理（可能还要再调工具，也可能给最终答案）
    ⑤ 循环直到模型不再返回 tool_calls —— 此时 message.content 就是最终答案

课案《概念》小节的原话（Function Call 的正式定义，建议背下来）：
    「Function Call 是大模型调用外部工具的标准化协议，现在改名为 Tool Call。」
    「Function Call 允许大模型根据用户输入自动决定是否需要调用外部工具，
      并生成相应的函数调用。这是实现 Agent 功能的核心机制。」
拆开这三个「谁」，就是上面 ①~⑤ 的分工：
    谁决定   → 大模型（不是我们写 if/else 去猜用户想算什么）；
    决定什么 → 调哪个工具 + 参数填什么，以**结构化 JSON** 给出（不是自然语言描述）；
    谁执行   → 本文件的 Python 函数（模型只递交「申请书」，真正干活的是程序）。

四个容易踩的坑（课案《关键点说明》都点到了）：
    1. tool_call_id 必须一一对应：工具结果靠 id 认领，写错 id 模型就对不上号；
    2. assistant 那轮必须把 tool_calls 原样存回历史，否则模型不知道自己刚才调过什么；
    3. 必须设循环上限（课案用 range(10)），否则模型来回调同一个工具就死循环了；
    4. tool 角色的消息 content 只能是字符串，所以工具结果要 str() 一下。

课案原文的两处笔误（本文件已按正确写法实现，详见对应注释）：
    1. `from config import setting` → 本项目没有 setting，只有 settings；
    2. tools 列表推导式里 parameters 那行结尾少了 `}` 后的逗号，导致 "strict": True
       被写进了 parameters 字典内部。

课案出处：Agent 课案 → 工具调用 → function call → 实例（agent.py 主循环 + 运行结果示例）

运行方式：
    uv run Agent/04_function_call/agent_openai_jxsd.py
前置条件：
    - 依赖：`openai` SDK + 标准库 json，本目录的 tools_jxsd.py / tool_desc_jxsd.py
      （本项目已 uv sync 装好）。
    - 配置：根目录 .env 必须配好 MODEL_NAME / API_KEY / BASE_URL ——
      本文件统一 `from config import settings` 读取，密钥不落代码。
    - 网络：需要一个 **OpenAI 兼容**的接口（DeepSeek / 通义 / vLLM / 各类中转都行）。
      换服务商只改 .env 的 BASE_URL 与 MODEL_NAME，代码一行不用动。
    - 成本：本文件会**真实调用大模型多次**（三个演示，每个演示都可能转好几圈主循环），
      是会实际产生 token 费用的文件之一。
    - 不需要数据库、不需要起服务。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import json

# 课案原文写的是 `import openai` + `openai.Client(...)`，两种写法等价；
# 这里用 from 形式，与本目录既有的 agent_openai.py 保持一致
from openai import OpenAI

from config import settings
import tools_jxsd                      # 真正的 Python 工具函数
import tool_desc_jxsd as tool_desc     # 工具描述；课案里这个模块叫 tool_desc

# ---------- 0. 客户端 ----------
# OpenAI 兼容客户端：DeepSeek / 通义 / vLLM / 各种中转都能这样接，
# 只改 base_url 与 model 即可，代码一行不用动
client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

# ---------- 0.1 课案在这里插了一张「两个 API 怎么选」的表（原文保留）----------
# 课案原话：「tools 定义详见 https://developers.openai.com/api/docs/guides/function-calling
#   目前大部分模型还在 Chat Completions API，OpenAI 官方已明确推荐：对于所有新项目，
#   应使用 Responses API。Chat Completions API 虽然会继续得到支持，
#   但已进入维护状态，不再作为未来的发展方向。」
#
# | 特性维度 | Chat Completions API (/v1/chat/completions) | Responses API (/v1/responses) |
# |----------|--------------------------------------------|-------------------------------|
# | 核心定位 | 基础的对话补全                              | 面向智能体（Agent）的统一接口   |
# | 状态管理 | 无状态：每次请求需手动传递完整对话历史         | 有状态：通过 store: true 让服务端自动保存对话上下文 |
# | 工具与执行| 仅支持自定义函数 (function)；工具执行循环需客户端自行编写 | 支持内置工具（网页搜索、文件搜索等）及自定义函数；平台自动编排多步工具调用 |
# | 核心概念 | 消息 (Messages)：一个包含角色和内容的数组      | 条目 (Items)：更灵活的对象，可表示消息、推理过程、工具调用等多种类型 |
# | 请求/响应 | 请求体用 messages；响应是单一的 message 对象   | 请求体用 input；响应是一个包含多个 output 条目的列表 |
# | 性能与成本| 缓存利用率作为基准                            | 缓存利用率提升 40-80%，可显著降低成本 |
#
# 这张表跟本文件的关系，就在第 4 行「工具执行循环需客户端自行编写」——
# **本文件那 80 行主循环，正是 Chat Completions API 强制你手写的那部分**。
# 换成 Responses API，平台会替你编排多步工具调用，循环就不必自己写了；
# 这也是后面几节 LangChain / DeepAgents 越包越省事的同一个方向。
# 本文件仍按课案用 Chat Completions，因为「先手写一遍」才是理解 Agent 的必经之路。

# 课案原句，一字未改
system_prompt = "你是一个助手，可以帮助用户进行数学计算。"


# ================================================================
# 一、动态生成 tools 定义（课案原文写法）
# ================================================================
# 课案《关键点说明 1》：「动态生成 tools —— 通过列表推导式从 tool_desc.list_tools()
# 动态生成工具定义」。好处是新增工具时只要在 tools.py 里加一个 xxx_tool 函数，
# 这里一个字都不用改。
#
# 课案原文（注意 parameters 那行结尾的语法笔误已在本文件修正）：
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
#             }                       ← 课案原文这里少了逗号/右括号，"strict" 被写进了 parameters 内
#             "strict": True  # 开启后，模型必须生成合法 JSON
#         }
#     } for tool in tool_desc.list_tools()]
#
# 课案那时 tools.py 里只有 4 个数学工具，所以 parameters 写死 {a, b} 也没问题；
# 本文件工具更多（含 string / array / object 等类型），所以改成取 tool["参数"]。
tools = [
    {
        "type": "function",
        "function": {
            "name": tool["工具名"],
            "description": tool["工具描述"],
            "parameters": tool["参数"],
            "strict": True,      # 与 additionalProperties: False 配对使用
        },
    }
    for tool in tool_desc.list_tools()
]


# ================================================================
# 二、历史对话示例（Few-shot）—— 课案原文，一字不漏
# ================================================================
# 课案《关键点说明 2》：「Few-shot 示例 —— 在历史对话中提供示例，
# 帮助模型理解如何调用工具」。
# 这段历史本身就是一本「工具调用示范教材」，它在教模型三件事：
#     ① 复杂算式要**拆成多步**，一步只调一个工具；
#     ② 每轮 assistant 消息里要带 tool_calls，工具结果用 role="tool" 回填；
#     ③ 最终答案要基于工具结果算出来（16-9=7 → 8*2-9=7）。
#
# tool_call_id 的作用（课案《关键点说明 5》：「通过 tool_call_id 关联工具调用和结果」）：
#     一轮里模型可以同时发起多个工具调用（并行调用），回来的结果到底属于哪一个？
#     全靠 id 认领：assistant 的 tool_calls[i].id == tool 消息的 tool_call_id。
#     对不上号，模型会认为「我要的结果还没给我」，要么重调、要么报错。
def build_few_shot_history() -> list[dict]:
    """课案的 Few-shot 历史对话（8*2-9 的三段往返）"""
    return [
        {"role": "user", "content": "8*2-9"},
        # 第一轮：模型决定先算 8*2。注意 tool_calls 里的 arguments 是**JSON 字符串**，
        # 不是 dict —— 这是 OpenAI 协议的硬性规定，所以下面用 json.dumps 造示例
        {"role": "assistant", "content": "要计算8*2-9。首先计算8*2。", "tool_calls": [
            {"id": "call_1", "type": "function",
             "function": {"name": "mul_tool", "arguments": json.dumps({"a": 8, "b": 2})}}
        ]},
        # 工具结果回填：tool_call_id 必须等于上面那条的 "call_1"
        {"role": "tool", "tool_call_id": "call_1", "content": "16"},
        # 第二轮：拿到 16 之后，模型继续算 16-9
        {"role": "assistant", "content": "8*2=16，现在计算16-9。", "tool_calls": [
            {"id": "call_2", "type": "function",
             "function": {"name": "sub_tool", "arguments": json.dumps({"a": 16, "b": 9})}}
        ]},
        {"role": "tool", "tool_call_id": "call_2", "content": "7"},
        # 工具用完了，这一轮不再返回 tool_calls，而是直接给最终答案
        {"role": "assistant", "content": "16-9=7，所以8*2-9=7"},
    ]


# ================================================================
# 三、主循环 —— 课案《实例》那段完整 Agent 循环
# ================================================================
# 课案原文是裸写在脚本里的（一个 for + break），本文件把它包成函数以便复用，
# **循环体一字未改**，只补了三处防御/说明（都有注释标出）。
def run_agent_loop(history: list[dict], tools: list[dict],
                   system_prompt: str = system_prompt, max_turns: int = 10,
                   first_tool_choice: str = "auto"):
    """
    执行 Function Call 主循环

    :param history: 对话历史（会被就地修改：追加 assistant 的 tool_calls 与 tool 结果）
    :param tools: 工具描述列表（发给模型的 tools 参数）
    :param system_prompt: system 提示词
    :param max_turns: 循环上限，课案原文是写死的 range(10)
    :param first_tool_choice: 首轮的 tool_choice；课案是 auto，
           传 "required" 可强制模型第一轮必须调工具（实测需要，原因见循环内注释）
    :return: 模型的最终回答；达到上限仍未结束时返回 None
    """
    final_answer = None
    tool_calls_made = 0        # 本文件加的计数器：直观看出主循环到底转了几圈、调了几次工具
    turns_used = 0

    # 课案《关键点说明 4》：「循环控制 —— 最多循环 10 次，防止无限循环」。
    # 模型完全可能陷入「反复调同一个工具」的怪圈，没有上限就是死循环 + 一直烧 token
    # （课案原文写的是 for _ in range(10)，这里换成 turn 是为了能判断「是不是第一轮」）
    for turn in range(max_turns):
        turns_used = turn + 1
        # 课案原文固定用 auto（即「模型自己决定调不调工具」），本文件把它提成参数：
        # 实测当前模型（中转站的 grok-4.6）相当随性——有时乖乖调工具，有时直接心算，
        # 有时甚至只在正文里**复述**「调用 mul_tool with a=4, b=6」却根本不发 tool_calls。
        # 想稳定看到主循环转起来，首轮传 "required" 最干脆。
        # ⚠️ 但绝不能每一轮都用 required：模型每轮都被逼着调工具，
        #    就永远没机会输出最终答案，循环会一路撞到 max_turns 上限——
        #    这也正说明「循环上限」不是可有可无的摆设。
        tool_choice = first_tool_choice if turn == 0 else "auto"

        response = client.chat.completions.create(
            model=settings.model_name,     # 课案原文是 setting.MODEL_NAME（settings 的另一种大小写风格）
            # 每次都要把完整历史重新发一遍：Chat Completions API 是**无状态**的，
            # 服务端不记得你上一轮说了什么（这一点正是课案开头那张对比表里
            # 「Responses API 有状态 / Chat Completions 无状态」的差别）
            messages=[{"role": "system", "content": system_prompt}, *history],
            tools=tools,
            tool_choice=tool_choice,
            temperature=0,     # 温度 0：算数题要的是稳定复现，不要随机发挥
        )

        message = response.choices[0].message

        # 如果没有工具调用，说明任务结束
        if not message.tool_calls:
            print("结束")
            print(f"最终答案: {message.content}")
            final_answer = message.content
            break

        # 记录assistant的tool_calls
        # 课案用 [dict(tc) for tc in ...] 把 pydantic 对象转成普通 dict 再入历史，
        # 而不是 message.model_dump()：model_dump 会带上 refusal / annotations
        # 等一堆空字段，部分兼容服务端见到陌生字段会直接报错
        history.append({
            "role": "assistant",
            "content": message.content,
            "tool_calls": [dict(tc) for tc in message.tool_calls],
        })

        # 处理每个工具调用（模型一轮可能同时要调好几个工具，所以是 for 不是 if）
        for tool_call in message.tool_calls:
            tool_calls_made += 1
            # arguments 是模型生成的 JSON **字符串**，必须自己 json.loads 成 dict
            args = json.loads(tool_call.function.arguments)
            # 按函数名分发到真实函数执行（课案：tool_desc.call_tool）
            result = tool_desc.call_tool(tool_call.function.name, **args)
            # 工具结果必须带 tool_call_id 回到历史里，否则模型对不上号
            history.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(result)})
            # str(result)：role="tool" 的 content 只接受字符串，返回 dict/数字都要转一下
            print(f"调用工具: {tool_call.function.name}, 参数: {args}, 结果: {result}")
    else:
        # 课案原文没有这个分支（for-else 在没 break 时触发）；
        # 加上是为了让「防御上限真的被触发」这件事可见，而不是静默退出
        print(f"⚠️ 已循环 {max_turns} 次仍未得到最终答案，强制结束（防御上限生效）")

    print(f"[主循环共转 {turns_used} 圈，执行 {tool_calls_made} 次工具调用]")
    return final_answer


# ================================================================
# 四、跑起来：三个演示
# ================================================================
def demo_math_few_shot():
    """演示 1：课案《实例》原样 —— Few-shot 历史 + 数学题 2+4*6"""
    print("=" * 62)
    print("演示 1：课案《实例》原样（Few-shot 历史 + 用户问 2+4*6）")
    print("=" * 62)
    # 课案原文：
    #     user_prompt = "2+4*6"
    #     history.append({"role": "user", "content": user_prompt})
    history = build_few_shot_history()
    user_prompt = "2+4*6"
    history.append({"role": "user", "content": user_prompt})

    # 只给 4 个数学工具：课案当时的 tools.py 里就只有这 4 个，
    # list_tools(only=...) 正是为了还原这个场景（否则 13 个工具一起发，
    # 模型面对「创建用户」「查库存」这些无关工具反而更容易分心）
    math_tools = tool_desc.list_tools(only=tools_jxsd.MATH_TOOL_NAMES)
    print(f"本轮可用工具：{[t['工具名'] for t in math_tools]}")
    print()
    # 课案《运行结果示例》给出的预期输出：
    #     调用工具: mul_tool, 参数: {'a': 4, 'b': 6}, 结果: 24
    #     调用工具: add_tool, 参数: {'a': 2, 'b': 24}, 结果: 26
    #     结束
    #     最终答案: 2+4*6=26
    #
    # ⚠️ 本机实测提醒（详见演示 2）：把这段 Few-shot 一起喂给当前模型
    #    （本机接的是中转站的 grok-4.6）时，模型会**照着示例反复调 mul_tool**，
    #    甚至干脆不调工具直接作答——Few-shot 示例对它是干扰而非引导。
    #    这不是代码错，而是「提示词效果依模型而异」的真实体感，
    #    换 GPT-4o / DeepSeek 一类模型，这段示例通常就能如期生效。
    return run_agent_loop(history, _to_openai_tools(math_tools), system_prompt)


def demo_math_plain():
    """演示 2：同一道题、同一 system 提示，去掉 Few-shot + 首轮强制调工具"""
    print()
    print("=" * 62)
    print("演示 2：去掉 Few-shot，首轮 tool_choice='required'（稳定复现课案轨迹）")
    print("=" * 62)
    # 与演示 1 的差别有两处：
    #   ① history 里没有那 6 条示例消息；
    #   ② 首轮 tool_choice 传 "required"（强制模型第一轮必须调工具）。
    #
    # 为什么要加 ②？实测当前模型在 auto 下相当随性：
    #   - 有时乖乖调 mul_tool，再调 add_tool（就是课案预期的那条轨迹）；
    #   - 有时直接心算给出 26，一个工具都不调；
    #   - 有时在正文里复述「调用 mul_tool with a=4, b=6」，却根本不发 tool_calls。
    # 首轮 required 至少能保证「确实调过工具」，让主循环真的转起来。
    #
    # 但要如实说明：第二轮仍是 auto，模型可能接着调 add_tool（课案预期），
    # 也可能自己心算把 2+24 算完就收工。这一段属于模型的自由裁量，脚本管不着。
    # 课案《运行结果示例》给出的完整两步轨迹是：
    #     调用工具: mul_tool, 参数: {'a': 4, 'b': 6}, 结果: 24
    #     调用工具: add_tool, 参数: {'a': 2, 'b': 24}, 结果: 26
    #     结束
    #     最终答案: 2+4*6=26
    # 本机多次实测中，这条轨迹与「只调 mul_tool 后心算收工」两种都出现过；
    # 参数方面模型按 number 类型填成了 4.0/6.0，与课案的 4/6 数值等价，
    # 只是 JSON number 的写法不同。
    history = [{"role": "user", "content": "2+4*6"}]
    math_tools = tool_desc.list_tools(only=tools_jxsd.MATH_TOOL_NAMES)
    return run_agent_loop(history, _to_openai_tools(math_tools), system_prompt,
                          first_tool_choice="required")


def demo_param_types():
    """演示 3：课案《参数类型》小节的 9 个工具，看模型怎么填不同类型的参数"""
    print()
    print("=" * 62)
    print("演示 3：参数类型（string / integer / boolean / array / enum 真的被填对了）")
    print("=" * 62)
    param_tools = tool_desc.list_tools(only=tools_jxsd.PARAM_TOOL_NAMES)
    print(f"本轮可用工具：{[t['工具名'] for t in param_tools]}")
    openai_tools = _to_openai_tools(param_tools)

    prompts = [
        "帮我创建用户：姓名张三，邮箱 zhangsan@example.com，年龄 28，标签是「学生」和「篮球」，设置为已激活。",
        "25 摄氏度等于多少华氏度？",
    ]
    for prompt in prompts:
        print()
        print(f"[用户] {prompt}")
        # 每个问题都用一份干净的、只含 system + user 的历史
        run_agent_loop([{"role": "user", "content": prompt}], openai_tools, system_prompt)


def _to_openai_tools(tool_list: list[dict]) -> list[dict]:
    """
    把 list_tools() 的结果转成 API 需要的 tools 参数

    抽成函数是因为模块顶部那份 tools 是「全量工具」，而各演示只想给一部分工具；
    转换逻辑（就是课案那段列表推导式）只应存在一份。
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool["工具名"],
                "description": tool["工具描述"],
                "parameters": tool["参数"],
                "strict": True,
            },
        }
        for tool in tool_list
    ]


if __name__ == "__main__":
    print(f"模型：{settings.model_name}    接口：{settings.base_url}")
    print(f"模块顶部 tools 共 {len(tools)} 个（= list_tools() 扫到的全部工具）")
    print()

    demo_math_few_shot()        # 课案《实例》原样（带 Few-shot）
    demo_math_plain()           # 同一道题去掉 Few-shot，复现课案预期轨迹
    demo_param_types()          # 课案《参数类型》9 种参数的实战

    print()
    print("=" * 62)
    print("三个演示跑完。回看每次输出末尾那行 [主循环共转 N 圈…]：")
    print("  - 转 2 圈 = 模型调一次工具、拿到结果后再要一次（这就是 Function Call 的往返）；")
    print("  - 演示 1 里模型反复调同一个工具、一直撞到 10 圈上限，正是「循环上限」存在的意义。")
    print("=" * 62)

# -*- coding: utf-8 -*-
"""
MCP ⑤ Agent 调用 · openai（原生 SDK + 手写 Function Call 循环）
================================================================
课案这一节要讲的是：**不用任何 Agent 框架，用最原始的 OpenAI SDK 接 MCP**。
看懂它，你就明白 LangChain / DeepAgents 那些框架在背后帮你做了什么。

整条链路只有四步（课案原文的流程）：

    1. MCP 客户端 list_tools 拿到工具定义
    2. 转成 OpenAI 的 tools 格式（type=function / name / description / parameters）
    3. 模型返回 tool_calls 后，用 MCP 客户端 call_tool 执行
    4. 结果以 role="tool" 回填进 messages，再问一次模型

第 2 步的转换是这一节的灵魂，就一行：

    openai_tools = [
        {"type": "function",
         "function": {"name": t.name, "description": t.description, "parameters": t.inputSchema}}
        for t in mcp_tools
    ]

    # | MCP 侧字段 | OpenAI 侧字段 | 说明 |
    # |---|---|---|
    # | tool.name | function.name | 模型调用时回传的名字，必须逐字一致 |
    # | tool.description | function.description | 来自服务端 docstring，模型靠它选工具 |
    # | tool.inputSchema | function.parameters | 来自服务端类型注解生成的 JSON Schema |

反过来，模型回的结果也要转回去：
    模型的 tool_calls[i] → client.call_tool(name, json.loads(arguments))
    工具结果            → {"role": "tool", "tool_call_id": ..., "content": 文本}

**框架帮你省的就是这来回两次转换。** 下一节（LangChain）会看到它被压缩成两行。

课案出处：Agent 课案 → MCP协议 → Agent调用 → openai

运行方式（本文件自己起 MCP 服务端，跑完自动关闭）：
    uv run Agent/05_mcp/05_agent调用_openai_jxsd.py
"""

import asyncio
import json
from pathlib import Path
import socket
import sys

sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import threading
import time

from openai import AsyncOpenAI

from config import settings

# 复用 01 的四则运算服务端（load 它的模块拿到 mcp 实例，不再重写一遍工具）
SERVER_SCRIPT = Path(__file__).resolve().parent / "01_服务端_jxsd.py"
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8000                     # 与课案 Client("http://localhost:8000/mcp") 一致
MCP_URL = f"http://{HTTP_HOST}:{HTTP_PORT}/mcp"

# 课案用的是 AsyncOpenAI（因为整个循环都在 async 环境里，await 调用更自然）
openai_client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url)
MODEL = settings.model_name

# 循环上限：模型可能反复调用工具，必须设一个上限，否则出 bug 时会无限转下去烧钱
MAX_ROUNDS = 8


def mcp_tool_to_openai(tool) -> dict:
    """MCP 工具定义 → OpenAI tools 格式（本节的转换函数）"""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            # inputSchema 已经是标准 JSON Schema，两边格式完全一致，直接搬
            "parameters": tool.inputSchema,
        },
    }


# ---------- 自检脚手架：让本文件能单独跑起来（与 01/02/03 同一套路） ----------
def _port_in_use(host: str, port: int) -> bool:
    """能连上就说明端口已被监听 —— 判断外面是不是已经有一个 MCP 服务端在跑。"""
    # 探测端口用短超时：连不上是「没人监听」的正常情况，不该让脚本卡住。
    sock = socket.socket()
    sock.settimeout(0.5)
    try:
        sock.connect((host, port))
        return True
    # 探测失败只说明没人监听，绝不能因此中断脚本。
    except OSError:
        return False
    finally:
        sock.close()


def count_operations(expr: str) -> int:
    """数一数表达式里有几个二元运算符，作为「至少要调用几次工具」的下界。

    ⚠️ 生产实践补充（课案没写，但非常值得学）：
    模型会不会「偷懒少调工具」是**不可控**的 —— 本机实测：
        · 「8*2-9」有时只调 mul 就答 16；
        · 「2+4*6」有时只调 mul 就答 24，甚至编出工具从没返回过的 48。
    光靠系统提示词「请务必逐步调用工具」是压不住的。
    正确做法是**用程序算出流程下界，再检查实际执行有没有达到** ——
    与其祈祷模型自觉，不如让代码在模型偷懒时把它推回去。
    这就是 Agent 工程里说的「流程完整性校验 / guardrail」。

    这个简陋的实现只适合加减乘除的纯表达式（负数、括号会数错），
    生产里应该换成真正的表达式解析器。
    """
    return sum(expr.count(op) for op in "+-*/")


def start_server_in_thread():
    """后台线程拉起 01 的 streamable-http 服务；端口被占就直接连已有的。

    返回 None 表示「不是我起的」，退出时不能去关别人的服务端。
    """
    # 已经有服务在跑就直接用（可能是学员另开窗口起的），返回 None 交给调用方判断。
    if _port_in_use(HTTP_HOST, HTTP_PORT):
        print(f"ℹ️  {HTTP_HOST}:{HTTP_PORT} 已有 MCP 服务在运行，直接连它。")
        return None

    import importlib.util

    import uvicorn

    # 动态加载 01 的模块拿它的 mcp 实例：文件名以数字开头没法 import；
    # 从文件路径加载不会触发 01 __main__ 里的自检分支（那是另一个保护）
    # 动态加载 01 的模块：文件名以数字开头没法 import，
    # 从文件路径加载不会触发它 __main__ 里的自检分支。
    spec = importlib.util.spec_from_file_location("mcp_server_01", SERVER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mcp_server_01"] = module
    spec.loader.exec_module(module)

    # 注册进 sys.modules 后，模块内的 pydantic 模型才能被正确解析。
    app = module.mcp.http_app(transport="streamable-http", path="/mcp")
    server = uvicorn.Server(
        uvicorn.Config(app, host=HTTP_HOST, port=HTTP_PORT, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):          # 轮询等就绪，最多 10 秒
        # 起服务同样用「轮询 + started」，而不是固定 sleep。
        if server.started:
            return server, thread
        time.sleep(0.1)
    # 和 01/02/03 一样：轮询 started，起不来就打印中文原因。
    print(f"❌ MCP 服务端启动失败：{HTTP_HOST}:{HTTP_PORT} 无法监听。")
    return None


async def main() -> None:
    from fastmcp import Client

    client = Client(MCP_URL)

    async with client:
        # ---------- 步骤 1：拉取 MCP 工具定义 ----------
        mcp_tools = await client.list_tools()
        print("加载MCP工具：", [t.name for t in mcp_tools])

        # ---------- 步骤 2：转成 OpenAI 的 tools 格式 ----------
        openai_tools = [mcp_tool_to_openai(t) for t in mcp_tools]
        # 把转出来的结构打出来看一眼，这就是「框架替你做的那件事」长什么样：
        # {'type': 'function', 'function': {'name': 'add', 'description': '两数相加',
        #  'parameters': {'properties': {'a': {'type': 'number'}, 'b': {'type': 'number'}}, ...}}}
        print("转换后的 OpenAI 工具格式（以第一个为例）：")
        print(json.dumps(openai_tools[0], ensure_ascii=False, indent=2))

        # ---------- 课案原文的「手工构造多轮历史」 ----------
        # 课案在 messages 里手写了 assistant(tool_calls) / tool 两组消息，
        # 目的是让你看清「多轮工具调用的对话历史到底长什么样」——
        # 下面这段就是课案原文，默认注释掉（因为真实循环会自动生成同样的结构）。
        #
        # messages = [
        #     {"role": "system", "content": "你是一个助手，可以帮助用户进行数学计算。"},
        #     {"role": "user", "content": "8*2-9"},
        #     {"role": "assistant", "content": "要计算8*2-9。首先计算8*2。",
        #      "tool_calls": [{"id": "call_1", "type": "function",
        #                      "function": {"name": "mul", "arguments": json.dumps({"a": 8, "b": 2})}}]},
        #     {"role": "tool", "tool_call_id": "call_1", "content": "16"},
        #     {"role": "assistant", "content": "8*2=16，现在计算16-9。",
        #      "tool_calls": [{"id": "call_2", "type": "function",
        #                      "function": {"name": "sub", "arguments": json.dumps({"a": 16, "b": 9})}}]},
        #     {"role": "tool", "tool_call_id": "call_2", "content": "7"},
        #     {"role": "assistant", "content": "16-9=7，所以8*2-9=7"},
        #     {"role": "user", "content": "2+4*6"}
        # ]
        #
        # 要点：assistant 消息里的 tool_calls 和随后的 tool 消息是**成对**出现的，
        #       tool_call_id 必须一一对应，否则模型接口会直接报 400。

        # ---------- 步骤 3 & 4：真实的 Function Call 循环 ----------
        # 「8*2-9」和「2+4*6」都是两步运算，但优先级不同，
        # 两条题各跑一遍能看出模型是不是真的按顺序调工具。
        for question in ["8*2-9", "2+4*6"]:
            print("\n" + "=" * 60)
            print(f"用户：{question}")
            print("=" * 60)

            messages = [
                {
                    "role": "system",
                    # 系统提示词定规矩：**每一步**算术都要交给工具，禁止心算。
                    # 这不是废话——实测如果不写「每一步」，模型经常只调一次 mul 就
                    # 自己把 2+24 心算掉，甚至算错成 24。Agent 的输出稳定性，
                    # 一大半靠系统提示词约束，而不是靠模型自觉。
                    # 系统提示词里的四条规则缺一不可，尤其第 1、2 条：
                    # 不写「每一步都调工具」，模型会心算掉中间步骤，甚至算错。
                    "content": (
                        "你是一个助手，可以帮助用户进行数学计算。严格遵守以下规则：\n"
                        "1. 任何一次算术运算都必须调用工具完成，严禁心算或直接给出数字；\n"
                        "2. 表达式里有几步运算，就分几步调用工具，"
                        "前一步工具的结果作为下一步工具的输入；\n"
                        # 这是本节的核心循环：模型要工具 → 我们执行 → 结果回填 → 再问一次。
                        "3. 所有步骤都算完后，最后一行按「最终结果：<数字>」的格式输出；\n"
                        "4. <数字> 必须是工具返回过的值，不得自行编造。"
                    ),
                },
                {"role": "user", "content": question},
            ]

            # 流程下界：这个表达式至少要几次工具调用才算算完
            required_steps = count_operations(question)
            tool_calls_made = 0     # 实际调用了多少次
            push_backs = 0          # 因为「少调了工具」把模型推回去的次数

            # MAX_ROUNDS 是硬保险：模型陷入工具循环时由这个 for 兜住。
            for round_no in range(1, MAX_ROUNDS + 1):
                response = await openai_client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    tools=openai_tools,
                    tool_choice="auto",
                # 模型说完了但没有 tool_calls，是循环的常规退出点；这里先做完整性校验再退。
                )
                msg = response.choices[0].message

                # 没有 tool_calls 说明模型认为可以收尾了 —— 这是循环的常规退出条件
                if not msg.tool_calls:
                    # ⚠️ 但在收尾之前，先做「流程完整性校验」（见 count_operations 的注释）。
                    # 少调了工具就补一条 user 消息把它推回去继续算，最多推 2 次，
                    # 避免模型死活不改导致无限重试。
                    # 推回去的关键：先把助手那句话原样记进历史（保持对话结构完整），
                    # 再补一条 user 消息催它继续 —— 直接改 system 提示词是没用的，
                    # 因为模型只看「最近发生了什么」。
                    if tool_calls_made < required_steps and push_backs < 2:
                        push_backs += 1
                        print(
                            f"⚠️  模型只调用了 {tool_calls_made} 次工具，但 {question} "
                            f"至少需要 {required_steps} 步运算 —— 已把它推回去补齐（第 {push_backs} 次）。"
                        )
                        messages.append({"role": "assistant", "content": msg.content or ""})
                        messages.append({
                            # 把 assistant 消息原样追加：tool_calls 与随后的 tool 消息必须成对出现。
                            "role": "user",
                            "content": (
                                f"你只调用了 {tool_calls_made} 次工具，还有运算没有算完"
                                f"（{question} 至少需要 {required_steps} 步）。"
                                "请继续调用工具算出剩余步骤，禁止心算，算完后按"
                                "「最终结果：<数字>」输出。"
                            ),
                        })
                        continue

                    print(f"\n最终回答：{msg.content}")
                    break

                # 把 assistant 这条（含 tool_calls）原样追加进历史。
                # 用 model_dump() 而不是手写 dict：OpenAI SDK 的字段（refusal、
                # annotations 等）它会一并带上，避免漏字段导致下一轮 400。
                messages.append(msg.model_dump())

                for tool_call in msg.tool_calls:
                    func_name = tool_call.function.name
                    # arguments 是**字符串**（模型吐的是 JSON 文本），必须 json.loads
                    func_args = json.loads(tool_call.function.arguments)
                    print(f"[第 {round_no} 轮] 调用MCP工具：{func_name}，参数：{func_args}")

                    # 关键一行：工具不在本地，而是通过 MCP 客户端远程执行
                    tool_result = await client.call_tool(func_name, func_args)
                    func_response = (
                        tool_result.content[0].text if tool_result.content else str(tool_result)
                    )
                    print(f"           工具返回结果：{func_response}")
                    tool_calls_made += 1

                    # role="tool" 的消息必须带 tool_call_id，模型靠它把结果和请求对上
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": func_response,
                    })
            else:
                # for-else：循环跑满 MAX_ROUNDS 还没 break，说明模型陷入工具循环了
                # 调用成功与否只影响「要不要关服务」，演示本身已经跑完了，所以放在 finally 里。
                print(f"⚠️  已达最大轮次 {MAX_ROUNDS}，仍未给出最终回答，已中止。")


if __name__ == "__main__":
    started = start_server_in_thread()
    # finally 里只做收摊，不吞异常 —— 演示出错时栈信息照样要打出来。
    try:
        asyncio.run(main())
    finally:
        if started:
            server, thread = started
            server.should_exit = True
            thread.join(timeout=10)
            print("\n✅ 本文件启动的 MCP 服务端已关闭。")

# -*- coding: utf-8 -*-
"""
MCP ④ 提示词模板（Prompt）
================================================================
课案原文：Prompts 用于向客户端暴露**可复用的提示词模板**，客户端填充参数后发送给 LLM，
统一团队的提示词规范。

为什么要把 Prompt 放到服务端？
    - 一个团队里，写 Prompt 最好的人只有那么一两个；
    - 把优质 Prompt 沉淀成 MCP 服务，所有人（以及所有 Agent）拉下来就能用；
    - 改 Prompt 只改服务端一处，不用通知每个人去更新本地代码 —— 这才是它真正的价值。

和 Tool / Resource 对比（课案原表）：

    # | 装饰器 | 用途 | 客户端调用 | 返回值 |
    # |---|---|---|---|
    # | @mcp.tool | 执行操作（计算、查询、写入） | call_tool(name, args) | 执行结果 |
    # | @mcp.resource | 暴露只读数据 | read_resource(uri) | 数据内容 |
    # | @mcp.prompt | 提供提示词模板 | get_prompt(name, args) | 拼接后的消息 |

关键点：**Prompt 本身不调用模型**。它只负责「拼字符串 / 拼消息」，
返回给客户端之后由客户端自己决定发给哪个 LLM。所以这一节跑起来很快，也不花钱。

本文件实现课案的三段模板：
    ① 基础模板        code_review        返回字符串
    ② 含角色设定      writing_assistant  返回字符串，内含角色和写作要求
    ③ 带上下文的复杂  math_tutor         返回字符串，含解题要求与难度分支
并且额外补一个「返回消息列表」的多角色写法（课案没写，但实际项目很常用）。

课案出处：Agent 课案 → MCP协议 → 提示词模板（了解）

运行方式（本文件自己起服务端，跑完自动关闭）：
    uv run Agent/05_mcp/04_提示词_jxsd.py

本机实测结论（这段代码的价值全在这几条上，不跑一遍等于没学）：
    ① 四个提示词都能在 list_prompts() 里看到，description 就是各自函数的 docstring；
    ② get_prompt("code_review", {...}) 返回的是 **PromptMessage 列表**，不是字符串：
       取正文必须多走一层 —— prompt.messages[0].content.text。
       课案在客户端那段特意标注过这一点，老版本直接是 str，新版本（FastMCP 3.x）要多一层 .text；
    ③ difficulty / style 这些带默认值的参数**不用传**，服务端会用默认值填，
       所以 get_prompt("math_tutor", {"question": ...}) 也能成功；
    ④ 多角色模板（translate）返回两条消息，各自的 role 分别是 assistant / user —— 见下面第 4 段的版本坑。

为什么这一节跑得又快又不花钱：Prompt 服务端不调用任何模型，
它只是「帮你把模板拼好」，所以整段没有任何 LLM 请求，纯字符串处理。
"""

import asyncio
import socket
import sys

sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import threading
import time

import uvicorn
from fastmcp import FastMCP

# Message 是 FastMCP 3.x 里「一条消息」的载体：
# 返回多角色提示词时必须用它，不能再返回 [{"role": ..., "content": ...}] 这种 dict。
from fastmcp.prompts import Message

# 本文件监听 8011（和 03 的 8010 错开，方便两个文件同时开着对照看）
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8011
MCP_PATH = "/mcp"

mcp = FastMCP("提示词演示 🚀")


# ---------- 1. 基础提示词模板（返回字符串） ----------
# @mcp.prompt 后面不写名字时，函数名就是提示词名（code_review）。
# 想改名写 @mcp.prompt(name="代码审查")。
@mcp.prompt
def code_review(language: str, code: str) -> str:
    """代码审查提示词"""
    # 下面这段正文就是课案原文：四个审查维度写死在服务端，
    # 客户端拿到的一定是团队规定的那套，不会各写各的。
    return f"""请审查以下 {language} 代码，从以下几个方面给出建议：
1. 代码规范性
2. 性能优化
3. 安全漏洞
4. 可读性

代码：
{code}"""


# ---------- 2. 包含角色设定的提示词（返回字符串，内含角色设定） ----------
# 注意 style: str = "正式" 这个默认值：FastMCP 会把它标成「非必填参数」，
# 客户端不传 style 时就用 "正式"。—— 提示词模板的参数默认值就是这么用的。
@mcp.prompt
def writing_assistant(topic: str, style: str = "正式") -> str:
    """写作助手提示词，内含角色和写作要求"""
    return f"""你是一位{style}风格的专业写手。
请写一篇关于「{topic}」的文章，要求不少于500字。"""


# ---------- 3. 带上下文的复杂提示词（返回字符串） ----------
# 这一段体现的是「把业务规则写进模板」：难度不同，要求不同。
# 把这种分支写在服务端，客户端就永远拿到符合规范的 Prompt。
@mcp.prompt
def math_tutor(question: str, difficulty: str = "中等") -> str:
    """数学辅导提示词"""
    # 复杂模板的价值在于「业务规则放在服务端」：难度不同、要求不同，
    # 这段分支式要求写在服务端，客户端就永远拿到合规的提示词。
    return f"""你是一位耐心的数学老师，学生提出了以下问题：

{question}

难度等级：{difficulty}

要求：
- 先给出解题思路，再给出详细步骤
- 用通俗易懂的语言讲解
- 如果难度为"困难"，需要补充相关的知识点背景"""


# ---------- 4. 额外：返回「消息列表」的多角色模板（课案未写，工程里很常用） ----------
# 返回 list 时，列表里每一项都会变成一条消息。相比拼一个大字符串，
# 这样能把「角色铺垫」和「用户任务」分开，模型理解更稳。
#
# ⚠️ 版本坑（现有精简版 04_提示词.py 就是踩在这里）：
#   FastMCP 3.x 不再接受 [{"role": "system", "content": "..."}] 这种 dict，
#   会直接报 `messages[0] must be Message or str, got dict`。
#   必须用 fastmcp.prompts.Message 包装。
#
# ⚠️ 协议坑：MCP 的 PromptMessage.role **只允许 user / assistant**，没有 system。
#   所以「角色设定」要么折进正文（见上面第 2 段 writing_assistant 的写法），
#   要么用 assistant 先说一句来铺垫 —— 后者就是下面这个例子。
@mcp.prompt
def translate(text: str, target_lang: str = "英文") -> list:
    """翻译提示词：用 assistant 消息做角色铺垫，再用 user 消息给任务"""
    return [
        Message(role="assistant", content="我是专业翻译，译文自然流畅，只输出译文本身。"),
        Message(role="user", content=f"把下面的内容翻译成{target_lang}：\n{text}"),
    ]


# ---------- 5. 客户端侧：本文件自己起服务端 + 逐个把模板拉下来看 ----------
# 说明：下面这段是「自检脚手架」，和 01/02/03 里的同名函数一模一样。
# 之所以每个文件都带一份，是为了让每个文件都能**单独运行**（学员不用先开别的窗口）。
def _port_in_use(host: str, port: int) -> bool:
    """探测端口是否已被监听：判断外面是不是已经有一个同类服务在跑。"""
    sock = socket.socket()
    sock.settimeout(0.5)          # 探测用短超时，避免脚本卡在 connect 上
    try:
        sock.connect((host, port))
        return True               # 能连上就说明有人监听
    except OSError:
        return False              # 连不上不算错误，只表示「没人监听」
    finally:
        sock.close()              # 无论成败都要关，防止 socket 泄漏


def start_server_in_thread():
    """后台线程起服务端；端口被占则直接连已有的那个。

    返回 None 的语义是「不是我起的」——调用方据此决定退出时要不要关掉它。
    外面已经在跑的服务绝不能动：那可能是学员另一个窗口里正在用的服务端。
    """
    if _port_in_use(HTTP_HOST, HTTP_PORT):
        print(f"ℹ️  {HTTP_HOST}:{HTTP_PORT} 已有服务在运行，直接连它。")
        return None

    # mcp.http_app() 把 FastMCP 实例包成一个标准 ASGI 应用，再交给 uvicorn 跑。
    # 用 uvicorn.Server 而不是 uvicorn.run()，是为了拿到 started / should_exit 两个开关。
    app = mcp.http_app(transport="streamable-http", path=MCP_PATH)
    server = uvicorn.Server(
        uvicorn.Config(app, host=HTTP_HOST, port=HTTP_PORT, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):          # 轮询等待就绪，最多 10 秒（比写死 sleep 可靠）
        # 轮询等就绪（最多 10 秒）；server.started 由 uvicorn 在真正开始监听后置位。
        if server.started:
            return server, thread
        time.sleep(0.1)
    print(f"❌ 服务端启动失败：{HTTP_HOST}:{HTTP_PORT} 无法监听。")
    return None


async def main(url: str) -> None:
    from fastmcp import Client

    async with Client(url) as client:
        # --- 列出所有提示词模板 ---
        # list_prompts() 给的是「模板定义」：名字 + 描述 + 参数表，不含内容，也不花任何模型调用
        prompts = await client.list_prompts()
        print("可用提示词：")
        for p in prompts:
            print(f"  {p.name}: {p.description}")

        # --- 获取提示词（填充参数） ---
        # get_prompt 才是「填参数、拼内容」；返回的是 PromptMessage 列表
        prompt = await client.get_prompt(
            "code_review",
            {
                "language": "Python",
                "code": "def add(a,b):\n    return a+b",
            },
        )
        # FastMCP 3.x：.content 是 TextContent 对象，用 .text 取字符串
        # （课案在这里特意标注过：老版本直接是 str，新版本要多一层 .text）
        print(f"\n生成的提示词：\n{prompt.messages[0].content.text}")

        # --- 获取带角色设定的提示词 ---
        # style="幽默" 覆盖了函数签名里的默认值 "正式"；不传就用默认值
        prompt2 = await client.get_prompt(
            "writing_assistant",
            {"topic": "人工智能的未来", "style": "幽默"},
        )
        print(f"\n写作助手提示词：\n{prompt2.messages[0].content.text}")

        # --- 复杂模板：注意 difficulty 用默认值「中等」 ---
        # 只传必填参数也能成功 —— 这就是自定义提示词模板参数默认值的用途
        prompt3 = await client.get_prompt(
            "math_tutor", {"question": "为什么 0.999… = 1？"}
        )
        print(f"\n数学辅导提示词（difficulty 取默认值）：\n{prompt3.messages[0].content.text}")

        # --- 多角色模板：messages 里有两条，各带自己的 role ---
        # 这是课案没写、但工程里常用的写法：返回**消息列表**而不是一大段字符串。
        prompt4 = await client.get_prompt(
            "translate", {"text": "工具调用是 Agent 的基础能力。", "target_lang": "英文"}
        )
        print(f"\n翻译模板共 {len(prompt4.messages)} 条消息：")
        for m in prompt4.messages:
            print(f"  [{m.role}] {m.content.text}")


# ---------- 6. 入口：起服务 → 跑客户端演示 → 关掉自己起的服务 ----------
if __name__ == "__main__":
    started = start_server_in_thread()
    url = f"http://{HTTP_HOST}:{HTTP_PORT}{MCP_PATH}"

    try:
        asyncio.run(main(url))
    finally:
        # 放在 finally 里：即使演示中途抛异常，也要把服务端关掉，不然端口会一直被占着
        if started:
            server, thread = started
            server.should_exit = True     # 通知 uvicorn 退出
            thread.join(timeout=10)       # 等它真的退完再打印，避免日志乱序
            print("\n✅ 本文件启动的服务端已关闭。")

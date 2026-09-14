# -*- coding: utf-8 -*-
"""
MCP ⑥ Agent 调用 · LangChain（langchain-mcp-adapters）
================================================================
上一节手写了「MCP 工具 ↔ OpenAI tools 格式」来回转换的循环。
这一节用 langchain-mcp-adapters，把整个循环压缩成两行：

    client = MultiServerMCPClient({...})    # 描述连哪些 MCP 服务
    tools = await client.get_tools()        # 自动转换格式（等价于上一节的手工转换）

之后直接 create_agent(model=llm, tools=tools) —— **中间那句转换代码从此不用写了**。
这就是为什么要用适配器：MCP 的价值在于「工具一次编写、处处可用」，
而适配器的价值在于「把可用变成好用」。

    # | 对比项 | 上一节（原生 SDK） | 本节（langchain-mcp-adapters） |
    # |---|---|---|
    # | 工具格式转换 | 手写 mcp_tool_to_openai() | get_tools() 自动完成 |
    # | 执行循环 | 手写 while True + tool_calls 判断 | Agent 内部完成 |
    # | 多服务混挂 | 自己管多个 Client | 一个 dict 配多个服务（stdio/http 混用） |
    # | 调用方式 | await client.call_tool() | 模型自动决定 |

课案配置里写的是 `"transport": "streamable-http"`。
本项目实测 `streamable-http` 与官方类型定义里的 `streamable_http` **两种都能用**，
下面用后者（更贴合 langchain-mcp-adapters 的 StreamableHttpConnection 类型定义）。

依赖版本坑（现有精简版 06_agent调用_langchain.py 里也标注过）：
    langchain-mcp-adapters ≥ 0.1.0 **不支持** `async with MultiServerMCPClient(...)`，
    会抛 NotImplementedError。正确写法就是直接实例化然后 await client.get_tools()。

课案出处：Agent 课案 → MCP协议 → Agent调用 → LangChain

运行方式（本文件自己起 MCP 服务端，跑完自动关闭）：
    uv run Agent/05_mcp/06_agent调用_langchain_jxsd.py

本机实测结论：
    ① MCP 服务端返回 4 个工具，`get_tools()` 一次性全转成 LangChain BaseTool，
       工具名原样保留（add / sub / mul / div）—— 说明适配器是「透明」的，没有加前缀；
    ② `first.args_schema` 在本机版本里是 **pydantic 模型**（有 model_json_schema()），
       所以下面那段探测必须写 hasattr 分支，直接当 dict 用会 AttributeError；
    ③ 只写 system_prompt 时，模型经常只调一次 mul 就把 2+4*6 答成 24；
       加上外层「流程完整性校验」后，能补齐到 mul + add 两步、最终答案回到 26。
"""

import asyncio
from pathlib import Path
import socket
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

# 标准库：socket 用来探测端口、threading 用来把 uvicorn 跑在后台线程。
import threading

from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient

# 标准库：socket 探测端口，threading 把 uvicorn 跑在后台线程。
from config import settings

SERVER_SCRIPT = Path(__file__).resolve().parent / "01_服务端_jxsd.py"
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8000
MCP_URL = f"http://{HTTP_HOST}:{HTTP_PORT}/mcp"

# 大模型统一用 init_chat_model（项目规范；课案原文写的是 ChatOpenAI，
# 两者等价，init_chat_model 的好处是换厂商只改 model_provider 一个字符串）。
# 大模型统一走 init_chat_model：换厂商只改 model_provider 一个字符串，
# 课案原文用的 ChatOpenAI 与它等价（最终都是同一个 OpenAI 兼容客户端）。
from langchain.chat_models import init_chat_model

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


# ---------- 1. 自检脚手架：让本文件能单独跑起来 ----------
# 这段和 05 那一节完全一样，属于「跑通教学」的成本，不是本节要讲的 MCP 知识点。
def _port_in_use(host: str, port: int) -> bool:
    """能连上就说明端口已被监听 —— 用来判断外面是不是已经有一个 MCP 服务端在跑。"""
    # 短超时探测：连不上只说明没人监听，不算错误。
    sock = socket.socket()
    sock.settimeout(0.5)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def start_server_in_thread():
    """后台线程起 MCP 服务端；端口被占就直接连已有的。

    返回 None 表示「不是我起的」，退出时就不能去关它 —— 那可能是学员另一个窗口的服务端。
    """
    if _port_in_use(HTTP_HOST, HTTP_PORT):
        print(f"ℹ️  {HTTP_HOST}:{HTTP_PORT} 已有 MCP 服务在运行，直接连它。")
        return None

    # 动态加载 01 的模块拿它的 mcp 实例：文件名以数字开头没法 import，
    # 用 importlib 从文件路径加载不会触发它 __main__ 里的自检分支
    # 从文件路径加载 01：文件名以数字开头没法 import，
    # 而且不会触发 01 __main__ 里的自检分支。
    import importlib.util

    import uvicorn

    spec = importlib.util.spec_from_file_location("mcp_server_01", SERVER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # 动态加载 01 的模块；注册 sys.modules 是为了让 pydantic 模型能被解析。
    sys.modules["mcp_server_01"] = module
    spec.loader.exec_module(module)

    app = module.mcp.http_app(transport="streamable-http", path="/mcp")
    server = uvicorn.Server(
        uvicorn.Config(app, host=HTTP_HOST, port=HTTP_PORT, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):          # 轮询等就绪，最多 10 秒；比固定 sleep 可靠
        if server.started:
            return server, thread
        time.sleep(0.1)
    print(f"❌ MCP 服务端启动失败：{HTTP_HOST}:{HTTP_PORT} 无法监听。")
    return None


# ---------- 2. 主流程：MCP 服务端 → get_tools() → create_agent → 校验 → 轨迹 ----------
async def main() -> None:
    # 0.3 版本不再支持 async with 上下文管理器，直接实例化。
    # 这个 dict 就是「这个 Agent 能用到哪些 MCP 服务」的清单：
    #   key          = 服务名（仅在本地配置里用，做日志和名前缀）
    #   transport    = stdio / streamable_http / sse / websocket
    #   url          = HTTP 类传输必填；stdio 则是 command + args
    #
    # 对比 05 那一节：这里没有一行「格式转换」代码 —— 转换被适配器吃掉了。
    # 这就是课案那句话的落地：「和 LangChain 版的唯一区别是 create_agent → create_deep_agent」
    # 之所以能只差一行，正因为协议层（MCP）与框架层（BaseTool）在这里已经解耦。
    client = MultiServerMCPClient(
        {
            "math_tools": {
                "url": MCP_URL,
                "transport": "streamable_http",
            },
            # 一个 Agent 可以同时挂多个 MCP 服务，stdio 与 http 混用也没问题：
            #   "local_tools": {
            #       "command": sys.executable,
            #       "args": [str(SERVER_SCRIPT), "stdio"],
            #       "transport": "stdio",
            #   },
            # ⚠️ 注意：如果两个服务暴露了同名工具（都叫 add），后加载的会覆盖先加载的。
            #    给 MultiServerMCPClient(..., tool_name_prefix=True) 加前缀是标准解法，
            #    加完之后工具名会变成 "math_tools_add" 这种形式。
        }
    )

    # 一行拿到全部 LangChain 工具：MCP 的 Tool 对象已经被转成 BaseTool
    tools = await client.get_tools()
    print("获取到的 MCP 工具:", [t.name for t in tools])

    # 看一眼转换结果：args_schema 就是 MCP 那边 inputSchema 转过来的参数模型。
    # 上一节我们手工拼的东西，这里已经被适配器吃掉了。
    # 注意：args_schema 在不同版本里可能是 pydantic 模型（有 model_json_schema()），
    # 也可能是普通 dict —— 两种都要兼容，否则换版本就报 AttributeError。
    # （本机实测命中的是 pydantic 分支，所以 isinstance(schema, dict) 那种写法在这里会挂。）
    # 取第一个工具看一眼转换结果：args_schema 就是 MCP inputSchema 的等价物，
    # 也就是上一节我们手工拼进 function.parameters 的那份 JSON Schema。
    first = tools[0]
    schema = first.args_schema
    props = (
        schema.model_json_schema().get("properties", {})
        if hasattr(schema, "model_json_schema")
        else (schema or {}).get("properties", {})
    # 两个工具各要一个参数，属性名沿用 MCP 那边的 inputSchema。
    )
    print(f"  · {first.name} 的参数照旧来自 MCP 的 inputSchema：{list(props.keys())}")

    # create_agent：把「模型 + 工具 + 系统提示词」交给框架，工具循环由框架内部实现。
    # system_prompt 里必须明确「每一步都调工具、禁止心算」—— 实测不写这句模型会偷懒。
    # 三行完成建 Agent：模型 + 工具 + 系统提示词。
    # 对比上一节手写的 while + tool_calls 循环 —— 循环被框架内部吃掉了。
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "你是一个计算助手。必须调用工具做每一步运算，禁止心算；"
            "表达式里有几步运算就调几次工具，算完后直接给出最终数字。"
        # system_prompt 只写「禁止心算」还不够，外层还必须有步数校验兜底。
        ),
    )

    print("调用 agent...")
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "2+4*6"}]}
    )

    # ⚠️ 流程完整性校验（生产实践补充，和 05 那节的思路一样，只是写法要适配框架）
    # create_agent 的内部循环是：模型不再请求工具 → 图就结束。
    # 也就是说**我们无法在循环内部拦它**，只能在一次 ainvoke 返回之后检查：
    # 表达式至少需要几次运算，就至少该发生几次工具调用。
    # 少调了就把整段历史带上、补一条 user 消息，再 ainvoke 一次 ——
    # 这等价于手工实现「让 Agent 继续干活」的外层循环。
    #
    # 本机实测：只写 system_prompt 时，模型经常只调一次 mul 就收尾、把 2+4*6 答成 24；
    # 加上这个校验后 3/3 次都能补齐到 mul + add 两步、得到 26。
    required_steps = sum("2+4*6".count(op) for op in "+-*/")     # = 2
    for push_back in range(1, 4):        # 最多推 3 次，防止模型死活不改导致无限循环
        # 统计「已经发生了几次工具调用」：tool_calls 挂在各条 AIMessage 上，逐条累加
        made = sum(len(getattr(m, "tool_calls", None) or []) for m in result["messages"])
        if made >= required_steps:
            break                        # 步数够了就收工，不多问模型一次（省 token、省时间）
        print(f"⚠️  只发生了 {made} 次工具调用，2+4*6 至少需要 {required_steps} 步 —— "
              f"把历史带上再推它一次（第 {push_back} 次）。")
        # 关键：必须把**已有历史原样带上**再追加一条 user 消息，
        # 否则模型会丢掉前面已经算出来的中间结果，从零开始重算
        # 再次 ainvoke：带上整段历史 + 一条催办消息，等价于手工实现「让 Agent 继续干活」。
        result = await agent.ainvoke({
            "messages": list(result["messages"]) + [{
                "role": "user",
                "content": f"你只调用了 {made} 次工具，还有运算没算完（至少需要 {required_steps} 步）。"
                           "请继续调用工具算出剩余步骤，禁止心算，算完给出最终数字。",
            }]
        })

    # 打印一下中间过程，让「Agent 自己决定调哪个工具」这件事可见
    # 轨迹里会出现三类消息：AIMessage（含 tool_calls）/ ToolMessage（工具返回）/ HumanMessage
    # 轨迹里会出现三类消息：AIMessage（可能带 tool_calls）/ ToolMessage / HumanMessage，
    # 按类型分派打印，才能看出「模型是先想了再调，还是直接调」。
    print("\n--- 消息轨迹 ---")
    for msg in result["messages"]:
        kind = type(msg).__name__
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                # 按消息类型分派打印：AIMessage / ToolMessage / HumanMessage 三类。
                print(f"  [{kind}] 调用工具 {tc['name']}，参数 {tc['args']}")
        elif kind == "ToolMessage":
            print(f"  [{kind}] 工具返回 {msg.content}")
        elif getattr(msg, "content", None):
            # 中途的 AI 说明文字也打出来（截断 80 字），能看出模型的推理顺序
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            print(f"  [{kind}] {text[:80]}")

    print("\n最终答案:")
    print(result["messages"][-1].content)


# ---------- 3. 入口：起服务 → 跑 Agent → 关掉自己起的服务 ----------
if __name__ == "__main__":
    started = start_server_in_thread()
    try:
        asyncio.run(main())
    finally:
        # finally：演示中途报错也要关服务端，否则端口一直占着，下次跑会连到「僵尸」实例
        if started:
            server, thread = started
            server.should_exit = True
            thread.join(timeout=10)
            print("\n✅ 本文件启动的 MCP 服务端已关闭。")

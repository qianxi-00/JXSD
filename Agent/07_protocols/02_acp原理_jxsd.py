# -*- coding: utf-8 -*-
"""
ACP 协议 · 原理：把 JSON-RPC 2.0 报文自己收一发
================================================================
课案原文：IDE 和 Agent 双方通过 JSON-RPC 2.0 通信 —— Agent Client Protocol。

上一节（01_acp智能体_jxsd.py）里 `run_agent(server)` 一行就把 Agent 挂上了 stdio，
但「底下到底在传什么」被 deepagents-acp 封装掉了。本节就是把那层封装撕开：

    ① 讲透 JSON-RPC 2.0 的报文形态（请求 / 响应 / 通知 / 批量，以及它们的判定规则）
    ② 讲透 ACP 的 6 个核心方法（5 个请求 + 1 个通知）
    ③ **真的跑一遍**：
       - 内存双向管道：客户端协程 ↔ 服务端协程，走完整会话（initialize →
         session/new → session/prompt → 一串 session/update → 结果）
       - 真实子进程管道：`subprocess.Popen` 起一个 Python 子进程当 ACP 服务端，
         从它的 stdout 按行读真的 JSON —— 这就是 PyCharm 启动你脚本时发生的事

全程**只用标准库**（asyncio / json / subprocess / sys），不依赖 acp 包，
所以本机没装 acp 也能看到协议的真实收发。

和相邻小节的关系：
    01 讲「怎么把 Agent 挂上去」（用法），
    02 讲「挂上去之后传什么」（原理）——本节，
    03/04/05 换到 A2A：那是 Agent ↔ Agent，走的是「JSON-RPC over HTTP」。

课案出处：Agent 课案 → 协议 → ACP → 原理

运行方式：
    uv run Agent/07_protocols/02_acp原理_jxsd.py

前置条件：
    - **无**。本文件只用 Python **标准库**（asyncio / json / os / subprocess / sys），
      不需要 acp / deepagents-acp，也不连 LangChain、不调大模型、不连数据库；
    - 因此它是整个 07_protocols 里最「空手可跑」的一节：
      哪怕依赖全都没装，也能看到 JSON-RPC 的真实收发。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，防中文/emoji 报错

import asyncio
import json
import os
import subprocess

# ================================================================
# 1. JSON-RPC 2.0 的四种消息形态（课案「核心特征」表格搬进注释）
# ================================================================
# 课案原文：JSON-RPC 2.0 是一种轻量级的远程过程调用协议，用 JSON 格式编码
# 请求和响应，**与传输层无关**（可跑在 HTTP、stdio、WebSocket 等之上）。
#
# | 特征          | 说明                                                          |
# | 请求格式      | {"jsonrpc": "2.0", "id": 1, "method": "方法名", "params": {...}} |
# | 响应格式      | {"jsonrpc": "2.0", "id": 1, "result": {...}} 或 {"id":1,"error":{...}} |
# | 通知(Notification) | 不带 id 的请求，服务端**不回复**（如 session/update 流式推送） |
# | 批量请求      | 一次发多个请求对象组成的数组，服务端返回对应数组               |
# | 传输无关      | 只定义消息格式，不限制传输方式——stdio、HTTP、WebSocket 均可    |
#
# 四条铁律（背下来，A2A 那三节还会再见）：
#     1. `jsonrpc` 字段固定是字符串 "2.0"，不是数字 2.0；
#     2. **有 id 必有回**：请求一定收到一条同 id 的响应（result 或 error 二选一）；
#     3. **无 id 不回**：通知（notification）单向，服务端处理完就算了；
#     4. id 由**发起方**分配，只在一次会话内唯一，用来把响应配回请求。
#
# ★ 每种报文的判定规则（拿到一条 JSON，按下面四步就能判断它属于哪一种）
#   判定看的是**字段组合**，不是内容，所以规则可以穷举：
#
#   | 判定顺序 | 报文形态      | 判据（字段组合）                                    | 收到方要做什么 |
#   |---|---|---|---|
#   | 第 1 步   | 批量请求/响应 | 顶层是 **JSON 数组** `[...]`                        | 逐条按下表判定，把「有响应的」收集成数组回；数组里可能混着请求和通知 |
#   | 第 2 步   | 请求          | 对象里**有 `method` 且有 `id`**                     | 处理它，然后回一条**同 id** 的响应 |
#   | 第 3 步   | 通知          | 对象里**有 `method` 但没有 `id`**（或 id 为 null）   | 处理它，**不回任何东西**（回了就是协议错误） |
#   | 第 4 步   | 响应          | 对象里**没有 `method`，有 `id`**，并带 `result` 或 `error` | 按 id 找到等待中的请求，把 result/error 交给它 |
#
#   两个边界情况（本文件下面都真的构造出来跑过）：
#     · `id: null` 的错误响应 —— 请求连 JSON 都解析不出来（-32700 Parse error）时，
#       规范要求回一条 `"id": null` 的 error，因为此时根本不知道对方用的是哪个 id；
#     · 批量请求返回的数组**可以比请求数组短** —— 数组里的通知不产生响应条目，
#       所以「两个请求 + 一个通知」进去，出来只有两条响应。
#
#   还有一个不属于 JSON-RPC、但属于本协议传输层的规则：
#     · **行分隔**：一行 = 一个完整 JSON 对象，用 `\n` 分隔（不是长度前缀、不是大数组），
#       所以读的一方可以按行解析、按行处理，天然支持流式。
#
# JSON-RPC 标准错误码（错误响应里 error.code 用这些值）：
# | 码     | 含义           | 典型场景                             |
# | -32700 | Parse error    | 收到的根本不是合法 JSON              |
# | -32600 | Invalid Request| 缺 jsonrpc / method 字段，结构不对    |
# | -32601 | Method not found| 方法名不认识（例如拼错 session/new） |
# | -32602 | Invalid params | 参数缺失或类型不对（例如没传 cwd）    |
# | -32603 | Internal error | 服务端内部异常                        |
# | -32000 ~ -32099 | 服务端自定义错误码，ACP 用它传业务错误 |


# ================================================================
# 2. 消息构造函数：一眼看清四种形态的字段差异
# ================================================================
def rpc_request(method: str, params: dict, req_id: int) -> dict:
    """请求：有 id，等对方回。"""
    return {"jsonrpc": "2.0", "id": req_id, "method": method, "params": params}


def rpc_notification(method: str, params: dict) -> dict:
    """通知：**不带 id**，对方不回复。ACP 的 session/update / session/cancel 都是它。"""
    return {"jsonrpc": "2.0", "method": method, "params": params}


def rpc_result(req_id: int, result: dict) -> dict:
    """成功响应：id + result，两个字段就够了。"""
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def rpc_error(req_id: int | None, code: int, message: str, data=None) -> dict:
    """失败响应：id + error{code,message,data}，**不能同时有 result**。

    id 可以是 None —— 当请求的 id 都解析不出来时（例如 JSON 坏了），
    规范要求回一条 `"id": null` 的错误。
    """
    err = {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}
    if data is not None:
        err["error"]["data"] = data
    return err


# ================================================================
# 3. ACP 的 6 个核心方法（课案表格搬进注释）
# ================================================================
# 课案原文：ACP 定义的核心方法只有 6 个，其中 5 个是请求（客户端 → 服务端），
# 1 个是通知（服务端 → 客户端）：
#
# | 方法            | 一句话                                                    |
# | initialize      | 握手，交换协议版本和能力                                  |
# | session/new     | 新建会话，返回 sessionId                                  |
# | session/load    | 恢复已有会话                                              |
# | session/prompt  | 发送用户消息，Agent 处理后返回 stopReason                 |
# | session/cancel  | 取消当前会话中的执行（通知）                              |
# | session/update  | 通知：Agent 流式推送状态，客户端不要求回复。通过           |
# |                 | sessionUpdate 子类型区分：agent_message_chunk（回复文本）、  |
# |                 | agent_thought_chunk（思考链）、tool_call（发起工具调用）、   |
# |                 | tool_call_update（工具状态更新）、plan（执行计划）等        |
#
# 方向补充（课案表格没写全，但实际会用到）：
#     - 客户端 → 服务端：initialize / session/new / session/load / session/prompt
#       / session/cancel
#     - 服务端 → 客户端：session/update（通知），以及 fs/read_text_file、
#       fs/write_text_file、session/request_permission 这几个「反向请求」
#       —— 因为你要读文件、要用户批准执行命令，光靠 Agent 自己做不到。
#
# 一次典型会话的报文时序（课案原文给出的就是这段）：
#     → {"jsonrpc":"2.0","id":1,"method":"initialize","params":{...}}
#     ← {"jsonrpc":"2.0","id":1,"result":{...}}
#     → {"jsonrpc":"2.0","id":2,"method":"session/new","params":{...}}
#     ← {"jsonrpc":"2.0","id":2,"result":{...}}
#     → {"jsonrpc":"2.0","id":3,"method":"session/prompt","params":{...}}
#     ← {"jsonrpc":"2.0","method":"session/update","params":{...}}   ← 无 id，通知
#     ← {"jsonrpc":"2.0","method":"session/update","params":{...}}
#     ← {"jsonrpc":"2.0","id":3,"result":{"stop_reason":"end_turn"}}
#
# 传输层细节（课案原文）：客户端（PyCharm）通过 stdin 发 JSON-RPC 请求，
# 服务端（你的 Python 脚本）处理后从 stdout 回复。**每一行是一个完整的
# JSON 对象，用 \\n 分隔**。AgentServerACP 和 run_agent() 已经封装好了协议，
# 你只需关注 agent 的业务逻辑。


# ================================================================
# 4. 假的 ACP 服务端：纯逻辑，不依赖任何第三方包
# ================================================================
# 真服务端里，这里应该是 `AgentServerACP(deep_agent)`；为了不调模型、不联网，
# 我们用一个「固定回复 + 分块吐出」的假 Agent，只演示协议层。
FAKE_AGENT_REPLY = "先执行 uv sync 安装依赖，然后 uv run main.py 启动。"


class FakeAcpServer:
    """内存版 ACP 服务端：收请求 → 分发 → 回响应，顺手推 session/update 通知。

    它同时扮演两个角色（ACP 的现实就是这样，双向都对开）：
        - 服务端：处理 initialize / session/new / session/prompt / session/load
        - 客户端：主动向对端发 session/update 通知（流式推送）
    """

    def __init__(self, inbound: "asyncio.Queue[dict]", outbound: "asyncio.Queue[dict]") -> None:
        self.inbound = inbound  # 对端 → 我
        self.outbound = outbound  # 我 → 对端
        self.sessions: dict[str, dict] = {}
        self._next_session = 0
        self._next_call = 100  # 服务端反向请求用的 id 段，和客户端 id 错开，避免撞车

    # ---------- 4.1 发送侧 ----------
    async def send(self, message: dict) -> None:
        await self.outbound.put(message)

    async def notify(self, method: str, params: dict) -> None:
        """推一条通知出去（无 id）。"""
        await self.send(rpc_notification(method, params))

    async def _stream_message(self, session_id: str, text: str, chunk_size: int = 12) -> None:
        """把回复切成若干 agent_message_chunk 通知推出去。

        这就是「流式输出」在协议层的真实样子：**没有 id 的 session/update**
        一条接一条，客户端收到就渲染，不需要回复。
        """
        for i in range(0, len(text), chunk_size):
            # 用「无 id 的通知」一句一句推 —— 这就是协议层的流式输出：
            # 发送方不等任何回复，接收方收到一块渲染一块。
            await self.notify(
                "session/update",
                {
                    "sessionId": session_id,
                    "update": {
                        # 子类型固定用 agent_message_chunk：客户端靠它判断「这是正文片段」
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": text[i : i + chunk_size]},
                    },
                },
            )

    # ---------- 4.2 处理侧 ----------
    async def handle(self, message) -> None:
        """收到一条消息（可能是请求、通知，或批量数组）。"""
        if isinstance(message, list):
            # 批量请求：逐条处理，把「有结果的」收集成数组一次性回。
            # 通知在批量里不产生响应条目（没有 id）。
            responses = []
            for item in message:
                resp = await self._dispatch(item)
                if resp is not None:
                    responses.append(resp)
            if responses:
                await self.send(responses)
            return

        resp = await self._dispatch(message)
        if resp is not None:
            await self.send(resp)

    async def _dispatch(self, message: dict) -> dict | None:
        """单条分发。返回 None 表示「不回复」（通知，或通知类方法）。"""
        # --- 结构校验：缺 jsonrpc / method 一律 -32600 ---
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return rpc_error(None, -32600, "Invalid Request：缺少 jsonrpc 2.0 字段")

        method = message.get("method")
        params = message.get("params") or {}
        req_id = message.get("id")
        is_notification = req_id is None  # 没 id 就是通知，处理完不回

        if method == "initialize":
            # 握手：客户端报自己的 protocolVersion / 能力，服务端回自己的。
            # 双方按「都支持的能力」往下走（例如客户端不支持写文件，
            # 服务端就不该发 fs/write_text_file 反向请求）。
            client_caps = params.get("clientCapabilities", {})
            return rpc_result(
                req_id,
                {
                    "protocolVersion": min(params.get("protocolVersion", 1), 1),
                    "agentCapabilities": {
                        "loadSession": True,
                        "promptCapabilities": {
                            # 客户端连读文件都不会，就别指望它提供上下文
                            "embeddedContext": bool(client_caps.get("fs", {}).get("readTextFile")),
                            "image": False,
                        },
                    },
                },
            )

        if method == "session/new":
            cwd = params.get("cwd")
            if not cwd:
                # 参数校验失败 → -32602，而不是抛异常把连接搞崩
                return rpc_error(req_id, -32602, "Invalid params：session/new 必须带 cwd")
            self._next_session += 1
            # sessionId 由服务端分配（客户端只报 cwd），格式没有强制要求，这里用可读的 sess_0001
            session_id = f"sess_{self._next_session:04d}"
            # 建会话时就把 cwd / mcpServers 记下来，后面 session/prompt 只带 sessionId 就能找回上下文
            self.sessions[session_id] = {"cwd": cwd, "mcpServers": params.get("mcpServers", [])}
            return rpc_result(req_id, {"sessionId": session_id})

        if method == "session/load":
            session_id = params.get("sessionId")
            if session_id not in self.sessions:
                # 业务级错误用服务端自定义码（-32000 ~ -32099）
                return rpc_error(req_id, -32001, "会话不存在", {"sessionId": session_id})
            return rpc_result(req_id, {"sessionId": session_id})

        if method == "session/prompt":
            session_id = params.get("sessionId")
            if session_id not in self.sessions:
                return rpc_error(req_id, -32001, "会话不存在", {"sessionId": session_id})
            # ⚠️ 顺序很关键：ACP 的 session/prompt 生命周期要求
            #    「先推完所有 session/update，最后才回这条带 id 的响应」，
            #    响应里的 stopReason 表示本轮为什么结束：
            #    end_turn（正常说完）/ max_tokens / refusal / cancelled（被取消）
            # 先推一条「思考中」再推正文 —— 这是真实 ACP 的常见形态：
            # 一次 session/prompt 期间，服务端可以按自己的节奏推任意多条 session/update，
            # 客户端只需要按 sessionUpdate 子类型分别渲染（思考链 vs 正文）。
            await self.notify(
                "session/update",
                {
                    "sessionId": session_id,
                    "update": {
                        # agent_thought_chunk = 思考链；agent_message_chunk = 正文（见 _stream_message）
                        "sessionUpdate": "agent_thought_chunk",
                        "content": {"type": "text", "text": "（思考）学员问的是启动步骤…"},
                    },
                },
            )
            # 再把正文切成多块推出去（全部推完才轮到下面那条带 id 的响应）
            await self._stream_message(session_id, FAKE_AGENT_REPLY)
            return rpc_result(req_id, {"stopReason": "end_turn"})

        if method == "session/cancel":
            # 通知：不回。服务端收到后要把进行中的 prompt 以
            # stopReason="cancelled" 收尾，并撤掉所有待批准的 request_permission。
            print(f"        [服务端] 收到 session/cancel 通知（无 id），停止会话 {params.get('sessionId')}")
            return None

        return rpc_error(req_id, -32601, f"Method not found：{method}")

    async def run_forever(self) -> None:
        while True:
            message = await self.inbound.get()
            await self.handle(message)


class FakeAcpClient:
    """内存版 ACP 客户端（扮演 PyCharm）：发请求、按 id 收响应、收通知。"""

    def __init__(self, inbound: "asyncio.Queue[dict]", outbound: "asyncio.Queue[dict]") -> None:
        self.inbound = inbound  # 服务端 → 我
        self.outbound = outbound  # 我 → 服务端
        self.pending: "dict[int, asyncio.Future]" = {}
        self.notifications: "list[dict]" = []
        self._req_id = 0
        self._reader: "asyncio.Task | None" = None

    def _new_id(self) -> int:
        self._req_id += 1
        return self._req_id

    async def call(self, method: str, params: dict) -> dict:
        """发一个请求并**等它那条同 id 的响应**（通知会被顺路收集起来）。"""
        req_id = self._new_id()
        fut: "asyncio.Future" = asyncio.get_running_loop().create_future()
        self.pending[req_id] = fut
        await self.outbound.put(rpc_request(method, params, req_id))
        return await fut

    async def notify(self, method: str, params: dict) -> None:
        await self.outbound.put(rpc_notification(method, params))

    async def read_loop(self) -> None:
        """常驻读循环：把响应配给等待的 Future，把通知丢进 notifications 列表。

        真实客户端读的是 stdin 的每一行；这里读的是队列，
        但「一行一个 JSON 对象」的解帧逻辑完全一样。
        """
        while True:
            message = await self.inbound.get()
            if isinstance(message, list):  # 批量响应
                for item in message:
                    self._route(item)
                continue
            self._route(message)

    def _route(self, message: dict) -> None:
        if "id" in message and message["id"] is not None:
            fut = self.pending.pop(message["id"], None)
            if fut and not fut.done():
                fut.set_result(message)
                return
        # 没有 id：通知
        self.notifications.append(message)


async def demo_memory_pipe() -> None:
    """演示 ①：内存双向管道跑完整会话。"""
    print("=" * 66)
    print("演示 ① 内存管道：客户端协程 ↔ 服务端协程，完整 ACP 会话")
    print("=" * 66)

    c2s: "asyncio.Queue[dict]" = asyncio.Queue()  # 客户端 → 服务端
    s2c: "asyncio.Queue[dict]" = asyncio.Queue()  # 服务端 → 客户端
    server = FakeAcpServer(inbound=c2s, outbound=s2c)
    client = FakeAcpClient(inbound=s2c, outbound=c2s)

    server_task = asyncio.create_task(server.run_forever())

    def show(direction: str, message) -> None:
        """打印一行报文 —— 行分隔 JSON-RPC 的真实观感。"""
        text = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        print(f"    {direction} {text}")

    # ---------- ① 批量请求：先脱开读循环，直观看到「数组进、数组出」 ----------
    print("\n[1] 批量请求：一个数组里塞两条，服务端回同样结构的数组")
    print("    （JSON-RPC 规范：批量请求可以混合成功与失败，逐条独立结算）")
    batch = [
        rpc_request("initialize", {"protocolVersion": 1}, 101),
        rpc_request("session/nwe", {}, 102),  # 故意拼错方法名
    ]
    show("→", batch)
    await c2s.put(batch)
    # 读循环还没启动，直接手取 —— 省得和 read_loop 抢队列。
    # 收到的一定是**数组**（批量请求 → 批量响应），且长度可能小于请求数（里面没有通知）。
    show("←", await s2c.get())
    print()

    # ---------- 启动常驻读循环：后面的请求/响应都靠它按 id 配对 ----------
    # （真实客户端里这就是「从 stdin 按行读，读一条路由一条」的循环）
    reader_task = asyncio.create_task(client.read_loop())

    # ---------- ② initialize：握手 ----------
    # 注意下面先把「要发的报文」单独打印一遍，再用 client.call 真的发 ——
    # 打印和发送用的是同一份 params，所以看到的就是线路上跑的那条。
    print("[2] 握手：initialize（交换协议版本与能力）")
    init_params = {
        "protocolVersion": 1,
        "clientCapabilities": {"fs": {"readTextFile": True, "writeTextFile": True}},
    }
    print("    → " + json.dumps(rpc_request("initialize", init_params, 1),
                               ensure_ascii=False, separators=(",", ":")))
    # client.call 会分配 id、发出请求、并 await 那条同 id 的响应（配对逻辑在 read_loop 里）
    resp = await client.call("initialize", init_params)
    show("←", resp)

    # ---------- ③ session/new：建会话 ----------
    # cwd 用 os.getcwd()，让演示的会话落在真实工作目录上
    print("\n[3] 建会话：session/new（必须给 cwd，否则 -32602）")
    new_params = {"cwd": os.getcwd(), "mcpServers": []}
    print("    → " + json.dumps(rpc_request("session/new", new_params, 2),
                               ensure_ascii=False, separators=(",", ":")))
    resp = await client.call("session/new", new_params)
    show("←", resp)
    # sessionId 由服务端在响应里给出；后面 session/prompt / session/cancel 都要带它
    session_id = resp["result"]["sessionId"]

    # ---------- ④ 参数校验失败长什么样 ----------
    print("\n[4] 故意漏掉 cwd —— 看错误响应（JSON-RPC 错误码 -32602）")
    bad_params = {"mcpServers": []}
    print("    → " + json.dumps(rpc_request("session/new", bad_params, 3),
                               ensure_ascii=False, separators=(",", ":")))
    resp = await client.call("session/new", bad_params)
    show("←", resp)

    # ---------- ⑤ session/prompt：一轮对话 ----------
    print("\n[5] 发消息：session/prompt → 服务端先推一串 session/update 通知，最后才回响应")
    before = len(client.notifications)
    prompt_params = {
        "sessionId": session_id,
        "prompt": [{"type": "text", "text": "这个项目怎么跑起来？"}],
    }
    print("    → " + json.dumps(rpc_request("session/prompt", prompt_params, 4),
                               ensure_ascii=False, separators=(",", ":")))
    # 用 create_task 而不是 await：session/prompt 期间服务端会先推一串通知，
    # 必须让「等响应的协程」和「读循环」同时活着，通知才收得到（这正是异步协议的特征）
    prompt_task = asyncio.create_task(client.call("session/prompt", prompt_params))
    # 先歇一下让读循环把服务端推的通知全部取走，
    # 否则下面统计通知条数时可能「响应已经到了、通知还在路上」，看着像丢了消息
    await asyncio.sleep(0.05)
    resp = await prompt_task
    show("←", resp)
    # 数一下这一轮推了几条通知 —— 全部无 id，全部不需要客户端回复
    new_notifications = client.notifications[before:]
    print(f"    （这一轮共 {len(new_notifications)} 条 session/update 通知，全部无 id）")
    # 通知太多会刷屏，只挑头两条和最后一条展示（中间省略）
    for note in new_notifications[:2]:
        show("←", note)
    if len(new_notifications) > 3:
        print("    … 中间省略若干块 …")
    if new_notifications:
        show("←", new_notifications[-1])

    # ---------- ⑥ method 拼错 ----------
    print("\n[6] 方法名拼错：session/nwe → -32601 Method not found")
    print("    → " + json.dumps(rpc_request("session/nwe", {}, 5),
                               ensure_ascii=False, separators=(",", ":")))
    resp = await client.call("session/nwe", {})
    show("←", resp)

    # ---------- ⑦ session/cancel：通知 ----------
    print("\n[7] 取消：session/cancel —— 通知，服务端**不回复**")
    # 先用 show 把「要发的通知」打出来，再真的发（client.notify 不带 id）
    show("→", rpc_notification("session/cancel", {"sessionId": session_id}))
    await client.notify("session/cancel", {"sessionId": session_id})
    await asyncio.sleep(0.1)  # 给服务端一点处理时间
    print("    客户端收到的新消息数：0（0 = 果然一条都不回）")

    # 收尾：取消两个常驻任务并等它们真正退出（return_exceptions=True 避免取消异常往上冒）
    server_task.cancel()
    reader_task.cancel()
    await asyncio.gather(server_task, reader_task, return_exceptions=True)
    print()


# ================================================================
# 5. 演示 ②：真的起一个子进程，按行读写 JSON（真实 stdio）
# ================================================================
# 子进程里跑的这段脚本，就是「你的 ACP 服务端」的最小等价物：
# 循环 `for line in sys.stdin` 读一行、json.loads、按 method 分发、
# print(json.dumps(...), flush=True) 写回一行。
#
# ⚠️ 两个真实世界才会遇到的坑，这里都踩到了并处理掉：
#    1. **必须 flush**：stdout 带缓冲，不 flush 对方会一直等（看起来像死锁）。
#    2. **必须只用 stdout 传协议**：任何 print 调试输出都会污染协议流 ——
#       真实项目里日志一律走 stderr（看 child 脚本里那句
#       `print("child: got notification ...", file=sys.stderr)`，它就是为了演示这一点）。
CHILD_SERVER_SCRIPT = r'''
import json, sys

# 子进程也要显式指定编码，否则 Windows 下默认 GBK 会读不动 UTF-8 请求
sys.stdin.reconfigure(encoding="utf-8")
sys.stdout.reconfigure(encoding="utf-8")

sessions = {}
for line in sys.stdin:                      # ← 一行一个完整 JSON 对象
    line = line.strip()
    if not line:
        continue
    try:
        msg = json.loads(line)              # ← 解帧：一行 = 一条消息
    except json.JSONDecodeError:
        sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": None,
            "error": {"code": -32700, "message": "Parse error"}}) + "\n")
        sys.stdout.flush()
        continue

    method = msg.get("method")
    params = msg.get("params") or {}
    rid = msg.get("id")
    if rid is None:                         # 通知：不回复
        print("child: got notification " + str(method), file=sys.stderr, flush=True)
        continue

    if method == "initialize":
        result = {"protocolVersion": 1, "agentCapabilities": {"loadSession": True}}
    elif method == "session/new":
        sid = "sess_child_1"
        sessions[sid] = params.get("cwd")
        result = {"sessionId": sid}
    elif method == "session/prompt":
        result = {"stopReason": "end_turn"}
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": rid,
            "error": {"code": -32601, "message": "Method not found: " + str(method)}}),
            flush=True)
        continue

    # flush=True 是关键：不加它，管道缓冲会让父进程永远读不到这一行
    print(json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}, ensure_ascii=False),
          flush=True)
'''


def demo_subprocess_pipe() -> None:
    """演示 ②：subprocess + stdin/stdout 真实收发 JSON-RPC。

    这就是 PyCharm / Zed 启动你 ACP 服务端脚本时做的事：
    起进程 → 往它的 stdin 写请求 → 从它的 stdout 按行读响应。
    """
    print("=" * 66)
    print("演示 ② 真实子进程：subprocess + stdin/stdout（PyCharm 就是这么起你的脚本）")
    print("=" * 66)

    env = dict(os.environ)
    # PYTHONIOENCODING：强制子进程 stdio 用 UTF-8，避开 Windows 默认 GBK
    env["PYTHONIOENCODING"] = "utf-8"

    # sys.executable = 当前解释器（本项目的 .venv），不用去猜 python 在哪；
    # -c 直接内联那段子进程脚本，避免为了演示多建一个文件。
    # stdin/stdout 都用 PIPE：这正是「stdio 传输」的字面含义 —— 协议跑在管道上。
    proc = subprocess.Popen(
        [sys.executable, "-c", CHILD_SERVER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,   # stderr 单独开一根管子，日志才不会污染协议流
        text=True,
        encoding="utf-8",
        env=env,
    )

    def send(message: dict) -> None:
        """往子进程 stdin 写一行 JSON（带 \\n，行分隔协议）。"""
        # separators 去空格：线路上传输的就是这种紧凑形式
        line = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        print(f"    → {line}")
        assert proc.stdin is not None
        proc.stdin.write(line + "\n")
        proc.stdin.flush()  # 不 flush 子进程会一直阻塞在 readline 上

    def recv() -> dict:
        """从子进程 stdout 读一行 JSON。"""
        assert proc.stdout is not None
        line = proc.stdout.readline()
        print(f"    ← {line.strip()}")
        return json.loads(line)

    try:
        # ① 握手：按「请求 → 必须收到一条同 id 响应」的规则走
        send(rpc_request("initialize", {"protocolVersion": 1, "clientCapabilities": {}}, 1))
        recv()

        # ② 建会话：id=2，同样等一条响应
        send(rpc_request("session/new", {"cwd": os.getcwd(), "mcpServers": []}, 2))
        recv()

        # ③ 发消息：真实服务端会在响应之前推一串 session/update 通知，
        #    这里的子进程脚本简化了，直接回一条 stopReason 响应
        send(rpc_request("session/prompt", {"sessionId": "sess_child_1",
                                            "prompt": [{"type": "text", "text": "hi"}]}, 3))
        recv()

        # 通知：发出去后子进程不会回（它只往 stderr 记了一行日志）
        note = rpc_notification("session/cancel", {"sessionId": "sess_child_1"})
        print(f"    → {json.dumps(note, ensure_ascii=False, separators=(',', ':'))}")
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(note) + "\n")
        proc.stdin.flush()
        print("    ← （无响应 —— 通知的定义就是服务端不回复）")

        # 故意发一条坏 JSON，验证 -32700 Parse error 也是「一行一条响应」
        # （注意此时服务端根本解析不出 id，所以它回的那条 error 里 id 是 null）
        print('    → {"jsonrpc":"2.0","id":9,"method":  <-- 故意截断的坏 JSON')
        assert proc.stdin is not None
        proc.stdin.write('{"jsonrpc":"2.0","id":9,"method":\n')
        proc.stdin.flush()
        recv()

        # 收尾：关掉 stdin 让子进程的 for 循环自然结束，再等它退出并收 stderr 日志
        assert proc.stdin is not None
        proc.stdin.close()  # 关掉 stdin → 子进程 for 循环结束，正常退出
        proc.wait(timeout=10)
        stderr = proc.stderr.read() if proc.stderr else ""
        print(f"    子进程退出码：{proc.returncode}")
        print("    子进程 stderr（日志走这里，不污染协议流）：")
        for line in stderr.strip().splitlines():
            print(f"        {line}")
    finally:
        # 无论上面哪一步出错，都不能把子进程留在后台
        if proc.poll() is None:
            proc.kill()
    print()


# ================================================================
# 6. 入口
# ================================================================
def main() -> None:
    print("ACP 原理：JSON-RPC 2.0 的真实收发")
    print()

    # 演示 ① 用 asyncio（协议本身是异步双向流，同步写法做不出「一边等响应一边收通知」）
    asyncio.run(demo_memory_pipe())

    # 演示 ② 用同步 subprocess（这里要让学员看清「一行一行读」的手感）
    demo_subprocess_pipe()

    # 小结把「判定规则」再钉一遍 —— 这三条是本节唯一需要背下来的东西
    print("=" * 66)
    print("小结：ACP 没有魔法 —— 就是「一行一个 JSON 对象」跑在 stdin/stdout 上。")
    print("     · 有 id 的必回一条同 id 的响应（result 或 error）")
    print("     · 无 id 的是通知，服务端不回复（流式输出全靠它）")
    print("     · 服务端也能反向发请求（fs/read_text_file、session/request_permission）")
    print("     deepagents-acp 的 AgentServerACP + run_agent() 帮你把上面这些全做了，")
    print("     你只写 create_deep_agent(...)（见 01_acp智能体_jxsd.py）。")
    print("=" * 66)


if __name__ == "__main__":
    # Windows 上 asyncio 默认事件循环对 subprocess 支持不好，
    # 但本文件两个演示各用各的（asyncio 只跑内存队列），无需特殊设置。
    main()

# -*- coding: utf-8 -*-
"""
A2A 协议 · 服务端：把 CrewAI 分析师发布成 A2A 服务（AutoA2A / 2025 端口）
================================================================
课案原文：A2A（Agent-to-Agent）是 Google 推出的 Agent 间通信协议，
基于 JSON-RPC 2.0 over HTTP，实现不同框架、不同厂商的智能体之间标准化对话。

本节要讲什么
    1. A2A **解决什么问题 / 谁和谁通信**：Agent ↔ Agent。
       现实痛点：调度员是 DeepAgents 写的，分析师是 CrewAI 写的，
       两家框架互不认识 —— A2A 就是让它们能互相「下单/交货」的公共语言。
       （对照记：ACP 是编辑器 ↔ Agent，MCP 是 Agent ↔ 工具，三者互不替代。）
    2. A2A 是**怎么落在 HTTP 上的**：两个端点就够 ——
       `GET /.well-known/agent.json` 拿「名片」（能力发现），
       `POST /` 发 JSON-RPC（真正的调用）。这就是「JSON-RPC 2.0 over HTTP」。
    3. **Agent Card** 里每个字段是什么意思（谁靠它决定要不要调用你）；
    4. 一条 `message/send` 请求长什么样、响应里的 task/status/artifacts 怎么读；
    5. 怎么把 CrewAI Agent 一行发布成 A2A 服务（`serve_crewai_agent`）。

三种协议的分工（课案表格）：
    | 协议 | 通信双方        | 用途                       |
    |---|---|---|
    | A2A  | Agent ↔ Agent   | Agent 之间的对话与协作      |
    | ACP  | Agent ↔ 编辑器  | Agent 与 IDE/编辑器集成     |
    | MCP  | Agent ↔ 工具    | Agent 调用外部工具/服务     |

本节对应课案的「A2A → CrewAI端」：
    `crewai_a2a_server.py` —— 一个 29 行的文件，
    用 `serve_crewai_agent(analyst, port=2025)` 一行把 CrewAI Agent 暴露成 A2A 服务。
    DeepAgents 那边的调度员（见 04_a2a客户端_jxsd.py）再通过 A2AClient 调它。

CrewAI A2A 本地部署方案对比（课案表格）：
    | 方案              | 启动方式                            | 说明                                   |
    |---|---|---|
    | a2a_auto_wrapper  | serve_crewai_agent(agent, port=2025)| 基于 AutoA2A，一行代码启动，最简        |
    | AMP Factory       | CrewAI 企业自托管版                 | 付费，部署到自己的 AWS/Azure/GCP        |

⚠️ 本机现实：`crewai` 与 `a2a_auto_wrapper` **都没装**，所以不可能真的起 2025 端口。
   本文件按规范做了两层降级，仍然能让你看清 A2A 到底长什么样：
     ① 打印课案 29 行源码 + Agent Card（智能体名片）JSON 示例；
     ② 用**纯标准库**起一个真的 HTTP JSON-RPC 服务（随机端口、演示完自动关），
        让 urllib 真的把 `message/send` 发过去、真的收回来 ——
        「A2A = JSON-RPC 2.0 over HTTP」这句话就不再是空话。
   装了 crewai / a2a_auto_wrapper 之后，本文件会自动切到真流程。

课案出处：Agent 课案 → 协议 → A2A → CrewAI端

运行方式：
    uv run Agent/07_protocols/03_a2a服务端_jxsd.py

前置条件：
    - **无强制前置**：缺 crewai / a2a_auto_wrapper 时走纯标准库的降级演示，
      不需要网络、不需要 API Key（因为降级版回的是写死的报告，不调模型）；
    - 想走真流程（`serve_crewai_agent(analyst, port=2025)`）才需要：
      `uv add a2a_auto_wrapper langgraph-api langgraph-sdk deepagents crewai`，
      以及 .env 里可用的 api_key / base_url / model_name（CrewAI 要拿它们去建 LLM）。
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，防中文/emoji 报错

import json
import threading
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from config import settings

# ================================================================
# 0. 依赖探测：crewai / a2a_auto_wrapper 本机都没装
# ================================================================
# 铁律 1.2：模块层面永远不许因为缺包而 import 失败。
# | 包               | 作用                                      | 本机 |
# | crewai           | 造 CrewAI Agent（Agent / LLM）            | 缺   |
# | a2a_auto_wrapper | serve_crewai_agent / A2AClient（AutoA2A） | 缺   |
_HAS_CREWAI = False
_HAS_A2A_WRAPPER = False
_IMPORT_ERRORS: list[str] = []

try:
    from crewai import Agent, LLM  # type: ignore[import-not-found]

    _HAS_CREWAI = True
except ImportError as exc:
    _IMPORT_ERRORS.append(f"crewai：{exc}")

try:
    from a2a_auto_wrapper import serve_crewai_agent  # type: ignore[import-not-found]

    _HAS_A2A_WRAPPER = True
except ImportError as exc:
    _IMPORT_ERRORS.append(f"a2a_auto_wrapper：{exc}")

# 课案里的端口：CrewAI 分析师固定监听 2025
CREWAI_A2A_PORT = 2025

# ================================================================
# 1. 课案原文（对照用）：crewai_a2a_server.py 的 29 行
# ================================================================
# 课案出处：Agent 课案 → 协议 → A2A → CrewAI端
#
# ```python
# """CrewAI A2A 服务 — python crewai_a2a_server.py 启动（监听 2025 端口）"""
# from crewai import Agent, LLM
# from config import setting
#
#
# # 数据分析师 Agent：作为 A2A 服务端，等待 DeepAgents 调度员调用
# analyst = Agent(
#     role="数据分析师",
#     goal="对数据进行深度分析并生成结构化报告",
#     backstory="你是资深数据分析师，擅长数据洞察和趋势预测。",
#     llm=LLM(
#         model=setting.MODEL_NAME,
#         provider="openai",
#         base_url=setting.BASE_URL,
#         api_key=setting.API_KEY,
#     ),
# )
#
#
# if __name__ == "__main__":
#     from a2a_auto_wrapper import serve_crewai_agent
#
#     # 一行代码将 CrewAI Agent 暴露为 A2A 服务：http://localhost:2025
#     serve_crewai_agent(analyst, port=2025)
# ```
#
# 课案到本项目的两处改写（规范第 3 节）：
#     ① `from config import setting` → `from config import settings`
#        （本项目没有 conf.py / setting 单数形式，统一是 config.settings）
#     ② `setting.MODEL_NAME` 这种全大写下划线风格 → 本项目的 `settings.model_name`
#        （.env 字段是小写：API_KEY / BASE_URL / MODEL_NAME → api_key / base_url / model_name）
COURSE_SERVER_PY = '''"""CrewAI A2A 服务 — python crewai_a2a_server.py 启动（监听 2025 端口）"""
from crewai import Agent, LLM
from config import settings                 # 课案原文是 from config import setting


# 数据分析师 Agent：作为 A2A 服务端，等待 DeepAgents 调度员调用
analyst = Agent(
    role="数据分析师",
    goal="对数据进行深度分析并生成结构化报告",
    backstory="你是资深数据分析师，擅长数据洞察和趋势预测。",
    llm=LLM(
        model=settings.model_name,          # 课案原文 setting.MODEL_NAME
        provider="openai",
        base_url=settings.base_url,         # 课案原文 setting.BASE_URL
        api_key=settings.api_key,           # 课案原文 setting.API_KEY
    ),
)


if __name__ == "__main__":
    from a2a_auto_wrapper import serve_crewai_agent

    # 一行代码将 CrewAI Agent 暴露为 A2A 服务：http://localhost:2025
    serve_crewai_agent(analyst, port=2025)
'''


def build_analyst():
    """按课案原文构造 CrewAI 数据分析师——只在 crewai 装好时才会被调用。

    这里刻意保持课案的三段式（role / goal / backstory），
    因为 CrewAI 的心智模型就是「招一个员工」：给它职位、目标、履历。
    """
    return Agent(
        role="数据分析师",          # 职位：决定它「是谁」
        goal="对数据进行深度分析并生成结构化报告",   # 目标：决定它「要达成什么」
        backstory="你是资深数据分析师，擅长数据洞察和趋势预测。",   # 履历：决定它「怎么想问题」
        # LLM 的三件套全部来自 settings（.env），一个都不许硬编码
        llm=LLM(
            model=settings.model_name,
            provider="openai",
            base_url=settings.base_url,
            api_key=settings.api_key,
        ),
    )


# ================================================================
# 2. A2A 的第一块拼图：Agent Card（智能体名片）
# ================================================================
# A2A 和 ACP 最大的不同：**对方在哪、会什么，要能被「发现」**。
# 所以 A2A 规定每个服务必须在固定路径上挂一张名片：
#
#     GET  /.well-known/agent.json      （老版本，2025 年前后的教程都写这个）
#     GET  /.well-known/agent-card.json （新版官方路径，两者内容一致）
#
# 名片字段逐个解释：
# | 字段                          | 说明                                                    |
# | name / description / version  | 自我介绍，客户端拿它渲染「我发现了哪个智能体」           |
# | url                           | **A2A 服务的基础地址**，后面所有 JSON-RPC 都往这里 POST |
# | protocolVersion               | A2A 协议版本，双方不一致就别硬聊                        |
# | preferredTransport            | JSONRPC / GRPC / HTTP+JSON，多数实现是 JSONRPC          |
# | capabilities.streaming        | 是否支持流式（对应 message/stream 方法）                |
# | capabilities.pushNotifications| 是否支持服务端主动回调推送                              |
# | defaultInputModes/OutputModes | 收发的内容形态，一般是 ["text/plain"]                   |
# | skills[]                      | **能力清单**：会干什么、给样例、有哪些标签              |
# | skills[].id                   | 技能唯一标识；官方建议用「url + 技能名」算 UUID，避免撞名 |
# | skills[].examples             | 自然语言示例，调度员（LLM）主要靠它决定要不要调用        |
def build_agent_card(base_url: str) -> dict:
    """生成 CrewAI 分析师的 Agent Card。"""
    skill_name = "数据分析与趋势预测"
    return {
        # ---- 自我介绍：客户端拿这几项渲染「我发现了哪个智能体」 ----
        "name": "CrewAI 数据分析师",
        "description": "对数据进行深度分析并生成结构化报告，擅长趋势预测。",
        # uuid5：同一台服务、同一个技能名永远算出同一个 id（确定性哈希）
        "url": base_url,            # ★ A2A 服务的基础地址，后续所有 JSON-RPC 都往这里 POST
        "version": "1.0.0",         # 这个 Agent 自己的版本号（不是协议版本）
        # ---- 协议协商：双方版本不一致就别硬聊 ----
        "protocolVersion": "0.3.0",
        "preferredTransport": "JSONRPC",     # 也可 GRPC / HTTP+JSON，多数实现是 JSONRPC
        # ---- 能力位：客户端据此决定「能不能用某个方法」 ----
        "capabilities": {
            "streaming": True,       # 支持 message/stream（流式返回）
            "pushNotifications": False,   # 不支持服务端主动回调推送
        },
        "defaultInputModes": ["text/plain"],    # 我能收什么形态
        "defaultOutputModes": ["text/plain"],   # 我会吐什么形态
        # ---- skills[]：能力清单，调度员（LLM）主要靠这一节决定要不要调用你 ----
        "skills": [
            {
                # 技能唯一标识；官方建议用 url + 技能名 算 UUID，避免不同 Agent 之间撞名
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{base_url}#{skill_name}")),
                "name": skill_name,
                "description": "接收一段业务数据描述，输出趋势判断、异常点与建议。",
                "tags": ["数据分析", "趋势预测", "报告生成"],
                # examples 是**自然语言示例**：模型判断「这个活像不像它举的例子」，比 tags 更管用
                "examples": ["分析销售数据：1月100万 2月120万 3月90万，做趋势预测"],
            }
        ],
    }


# ================================================================
# 3. A2A 的第二块拼图：怎么发一条任务消息
# ================================================================
# 消息体长这样（和 ACP 的 session/prompt 神似，都是「角色 + 内容块数组」）：
#
#     {
#       "jsonrpc": "2.0", "id": 1, "method": "message/send",
#       "params": {
#         "message": {
#           "role": "user",
#           "parts": [{"kind": "text", "text": "分析销售数据…"}],
#           "messageId": "<客户端生成，用于去重/幂等>"
#         }
#       }
#     }
#
# 响应把「任务」整体返回，关键字段：
# | 字段                              | 说明                                 |
# | result.id                         | taskId，后续用 tasks/get 查进度      |
# | result.status.state               | submitted / working / completed …    |
# | result.artifacts[].parts[].text   | **Agent 的产出正文就在这里**         |
# | result.history[]                  | 往返消息历史                          |
#
# 方法名演进（不同版本教程会打架，认准你装的包）：
# | 版本   | 发消息              | 查任务            |
# | 0.2.x  | tasks/send          | tasks/get         |
# | 0.3.x  | message/send        | tasks/get         |
# | —      | message/stream（流式，配 capabilities.streaming=true） |
def build_task_message(query: str) -> dict:
    """构造一条 A2A 的 message/send 请求。"""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                # role 只能是 user / agent —— 和 OpenAI messages 的角色含义一样
                "role": "user",
                # parts 是内容块数组：文本、文件、结构化数据都能塞
                "parts": [{"kind": "text", "text": query}],
                "messageId": str(uuid.uuid4()),
            }
        },
    }


# ================================================================
# 4. 降级演示用的「真 HTTP」A2A 服务（纯标准库，零依赖）
# ================================================================
# 目的不是替代 a2a_auto_wrapper，而是用 60 行把 A2A 的传输层坐实：
#     · 名片：GET /.well-known/agent.json
#     · 调用：POST /  （body 是 JSON-RPC，Content-Type: application/json）
# 真实实现（serve_crewai_agent / AMP Factory）也是这两个端点，只是内部接了 LLM。
FAKE_ANALYST_REPORT = (
    "【趋势判断】1→2 月增长 20%，2→3 月回落 25%，属单峰波动，非持续增长。\n"
    "【异常点】3 月环比 -25% 需排查（促销结束？渠道断货？）。\n"
    "【建议】把 2 月作为活动基准，Q2 目标按 1 月水平 +10% 设定更稳。"
)


class _A2AHandler(BaseHTTPRequestHandler):
    """最小 A2A 端点实现：一个 GET 名片 + 一个 POST JSON-RPC。"""

    # 让 handler 能拿到外面构造好的名片
    agent_card: dict = {}

    def log_message(self, fmt, *args):  # 关掉 http.server 默认的 stderr 噪音
        return

    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 规定的方法名
        if self.path in ("/.well-known/agent.json", "/.well-known/agent-card.json"):
            # 能力发现：客户端启动时第一件事就是拉这张名片
            self._send_json(self.agent_card)
        else:
            self._send_json({"error": "not found", "path": self.path}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            # JSON-RPC 标准错误码 -32700
            self._send_json(
                {"jsonrpc": "2.0", "id": None,
                 "error": {"code": -32700, "message": "Parse error"}}
            )
            return

        method = request.get("method")
        req_id = request.get("id")

        if method == "message/send":
            message = (request.get("params") or {}).get("message") or {}
            # 取出用户问的文本（parts 里 kind == "text" 的那些拼起来）
            # —— A2A 的 parts 是**内容块数组**，所以要过滤 + 拼接，不能直接当字符串用
            query = " ".join(
                p.get("text", "") for p in message.get("parts", []) if p.get("kind") == "text"
            )
            # 真实服务端这里会把 query 交给 CrewAI Agent 跑；
            # 降级版直接回一段写死的报告，但**报文结构是真的**。
            self._send_json(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,       # ★ 必须回同 id：这是 JSON-RPC 的配对规则
                    "result": {
                        "id": str(uuid.uuid4()),          # taskId
                        "contextId": str(uuid.uuid4()),   # 会话上下文 id
                        # status.state 是任务的生命周期：
                        # submitted（已接收）→ working（处理中）→ completed / failed / canceled
                        # 长任务用 message/send 拿到 taskId 后，可以再 tasks/get 轮询进度
                        "status": {"state": "completed", "timestamp": "2026-01-01T00:00:00Z"},
                        # ★ 产出正文放在 artifacts 里（不是放在 status 里）——
                        #   一个任务可以产出多份 artifact，每份又由多个 part 组成
                        "artifacts": [
                            {
                                "artifactId": str(uuid.uuid4()),
                                "name": "分析报告",     # 给客户端展示用的可读名字
                                "parts": [
                                    {
                                        # kind="text" 和 A2A 请求里的 parts 对齐；
                                        # 换成 "file"/"data" 就能回文件或结构化数据
                                        "kind": "text",
                                        "text": f"[降级演示·非真实 CrewAI 输出]\n收到问题：{query}\n\n"
                                                + FAKE_ANALYST_REPORT,
                                    }
                                ],
                            }
                        ],
                        # history 原样带回往返消息，方便客户端做审计/多轮拼接
                        "history": [message],
                    },
                }
            )
            return

        # tasks/get：按 taskId 查一个**已经提交过**的任务的当前状态。
        # 这里简化成「直接回答 completed」，真实实现要维护一张 taskId → 状态 的表。
        if method == "tasks/get":
            self._send_json(
                {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "id": (request.get("params") or {}).get("id"),
                        "status": {"state": "completed"},
                    },
                }
            )
            return

        # 方法不认识 → JSON-RPC -32601
        self._send_json(
            {"jsonrpc": "2.0", "id": req_id,
             "error": {"code": -32601, "message": f"Method not found: {method}"}}
        )


def start_fake_a2a_server() -> tuple[ThreadingHTTPServer, str]:
    """起一个真的 HTTP 服务（端口 0 = 让系统分配空闲端口，避免撞车）。

    返回 (server, base_url)。跑完记得 server.shutdown()。
    """
    # port=0：让操作系统给个空闲端口。课案用的是固定 2025，
    # 但那是「真部署」才需要的稳定性；演示里用随机端口才不会跟别的进程抢。
    server = ThreadingHTTPServer(("127.0.0.1", 0), _A2AHandler)
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    _A2AHandler.agent_card = build_agent_card(base_url)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, base_url


def http_get_json(url: str) -> dict:
    """GET 一个 JSON 端点（能力发现走这里）。"""
    # noqa: S310 —— 目标是本地回环地址，不是用户可控的 URL
    with urllib.request.urlopen(url, timeout=10) as resp:  # noqa: S310 - 本地回环地址
        return json.loads(resp.read().decode("utf-8"))


def http_post_json(url: str, payload: dict) -> dict:
    """POST 一个 JSON-RPC 报文（真正的调用走这里）。"""
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    # Content-Type 必须是 application/json —— A2A 靠它区分「这是协议报文」而不是普通表单
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    # noqa: S310 —— 同上，本地回环
    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 - 本地回环地址
        return json.loads(resp.read().decode("utf-8"))


def print_install_hint() -> None:
    print("【前置条件检查】本机 venv 缺少 A2A 相关依赖：")
    for err in _IMPORT_ERRORS:
        # 原始 ImportError 文本里带着模块名，方便对号入座
        print(f"    - {err}")
    print()
    print("课案安装命令（本项目统一用 uv）：")
    print("    uv add a2a_auto_wrapper langgraph-api langgraph-sdk deepagents crewai")
    print("    # 或：F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe -m pip install "
          "a2a_auto_wrapper langgraph-api langgraph-sdk deepagents crewai")
    print("    # 注：crewai 体积不小（会带 litellm / embedchain 等一大串依赖），装前有个心理准备")
    print()


# ================================================================
# 5. 入口
# ================================================================
def main() -> None:
    print("=" * 66)
    print("A2A 服务端：把 CrewAI 数据分析师发布为 A2A 服务（课案端口 2025）")
    print("=" * 66)
    print()

    # ---------- 第 1 步：课案原文 ----------
    # 逐行打印课案源码，让读者先看到「一行代码发布成 A2A 服务」的原貌
    print("【课案原文】crewai_a2a_server.py（29 行）：")
    for line in COURSE_SERVER_PY.splitlines():
        print("    " + line)
    print()
    # PYTHONUTF8=1 是 Windows 上跑 CrewAI 的必备项（中文日志会乱码）
    print("【启动命令（课案原文）】")
    print("    $env:PYTHONUTF8='1'; .venv\\Scripts\\python crewai_a2a_server.py")
    print("    # $env:PYTHONUTF8='1' 是为了让 CrewAI 打印中文日志时不乱码")
    print()

    # ---------- 第 2 步：依赖情况 ----------
    print_install_hint()

    if _HAS_CREWAI and _HAS_A2A_WRAPPER:
        print("【依赖检查】crewai + a2a_auto_wrapper 均已就绪，按课案原文启动真实 A2A 服务。")
        print(f"            监听端口 {CREWAI_A2A_PORT}，Ctrl+C 退出。")
        print(f"            名片地址：http://localhost:{CREWAI_A2A_PORT}/.well-known/agent.json")
        print()
        analyst = build_analyst()
        try:
            # 课案原话：一行代码将 CrewAI Agent 暴露为 A2A 服务
            serve_crewai_agent(analyst, port=CREWAI_A2A_PORT)
        except KeyboardInterrupt:
            print("\n已手动停止 A2A 服务。")
        return

    # ---------- 第 3 步：降级演示 —— 先看名片 ----------
    # 两个服务端方案的取舍在输出里说明：真流程要付费/装包，降级版用标准库坐实协议
    print("【降级演示】未安装 crewai / a2a_auto_wrapper，无法真的 serve_crewai_agent(analyst, port=2025)。")
    print("            改用纯标准库起一个真 HTTP 服务，把 A2A 的传输层坐实。")
    print()
    server, base_url = start_fake_a2a_server()
    print(f"已启动降级版 A2A 服务：{base_url}（端口由系统分配，演示完自动关闭）")
    print()

    try:
        # ① 能力发现：真实 A2A 客户端第一步也是拉这张名片，才知道对方会什么、地址在哪
        card = http_get_json(f"{base_url}/.well-known/agent.json")
        print("① 能力发现：GET /.well-known/agent.json → Agent Card（智能体名片）")
        for line in json.dumps(card, ensure_ascii=False, indent=2).splitlines():
            print("    " + line)
        print()

        query = "分析销售数据：1月100万 2月120万 3月90万，做趋势预测"
        request = build_task_message(query)
        # 打印↔接收共用同一份 request 对象，所以看到的报文就是真正发出去的报文
        print("② 调用：POST / ，body 是 JSON-RPC 的 message/send")
        print("    → " + json.dumps(request, ensure_ascii=False, separators=(",", ":")))
        response = http_post_json(base_url, request)
        # 响应可能很长（artifacts 里塞了整份报告），先截 120 字符给个全貌
        print("    ← " + json.dumps(response, ensure_ascii=False, separators=(",", ":"))[:120] + " …")
        print()
        # 逐字段读响应：这是「怎么从 A2A 响应里把 Agent 的产出抠出来」的标准动作
        result = response["result"]
        print(f"    taskId    = {result['id']}")
        print(f"    state     = {result['status']['state']}")
        print("    产出正文（artifacts[0].parts[0].text）：")
        for line in result["artifacts"][0]["parts"][0]["text"].splitlines():
            print("        " + line)
        print()

        # 顺手演示一下错误路径：方法名写错会拿到 -32601
        # （A2A 复用 JSON-RPC 的标准错误码，和 02_acp原理_jxsd.py 里那套是同一份规范）
        bad = {"jsonrpc": "2.0", "id": 9, "method": "message/foo", "params": {}}
        print("③ 错误路径：方法名写错 → JSON-RPC -32601")
        print("    → " + json.dumps(bad, ensure_ascii=False, separators=(",", ":")))
        print("    ← " + json.dumps(http_post_json(base_url, bad), ensure_ascii=False,
                                   separators=(",", ":")))
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # 本地回环也可能被代理/防火墙拦掉；协议内容已经打印过，所以这里只提示不中断
        print(f"    本地 HTTP 演示失败（{type(exc).__name__}）：{exc}")
        print("    多为端口/防火墙/代理拦截 127.0.0.1 所致；协议内容仍以上面的报文为准。")
    finally:
        # 无论成功失败都要关服务，否则线程会一直挂着
        server.shutdown()
        server.server_close()
        print()
        print("降级版 A2A 服务已关闭。")
        print()

    # 收口：把「A2A 服务端到底要做哪两件事」压缩成两条，并指向下一节（客户端怎么调它）
    print("=" * 66)
    print("小结：A2A 服务端就两件事 ——")
    print("    ① 在 /.well-known/agent.json 挂一张「名片」（我是谁、会什么、在哪）")
    print("    ② 在同一地址收 JSON-RPC 的 message/send（GET 发现 + POST 调用）")
    print("    serve_crewai_agent(analyst, port=2025) 就是把这两件事自动配好；")
    print("    真实环境下这里跑的是 CrewAI 的 LLM，降级版换成了一段写死的报告。")
    print("    下一步：DeepAgents 调度员怎么调它 → 04_a2a客户端_jxsd.py")
    print("=" * 66)


if __name__ == "__main__":
    main()

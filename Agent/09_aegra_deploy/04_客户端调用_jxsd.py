# -*- coding: utf-8 -*-
"""
部署 ④：用 langgraph_sdk 调用部署好的 Agent（客户端那 30 行）
================================================================
Aegra 起好之后，客户端代码和连官方 LangSmith Deployments **完全一样** ——
这就是上一节「客户端 SDK 同款」那一格的实际含义：

    from langgraph_sdk import get_sync_client
    client = get_sync_client(url="http://localhost:2026")

    thread = client.threads.create()          # 创建会话线程（状态持久化在 PostgreSQL）
    for chunk in client.runs.stream(
        thread["thread_id"],
        "bushu",                              # 对应 aegra.json 中 graphs 的 key
        input={"messages": [{"type": "human", "content": "你好"}]},
        stream_mode="messages",
    ):
        ...

本节的核心知识点是**流式事件的数据结构**：

    · `stream_mode="messages"` 产生的事件名是 `messages/partial` / `messages/complete`
      （带斜杠的 SSE 事件名，前端按 event: 字段分发）；
    · `chunk.data` 是一个**两元素列表**：`[消息块, 元数据]`
        data[0] = {"content": ..., "type": "AIMessageChunk", ...}   真正的内容
        data[1] = {"langgraph_node": "chatbot", "tags": [...], ...}  来自哪个节点
    · ⚠️ **`content` 是「累计全文」而不是增量 token** ——
      每来一个事件，content 都是「从开头到现在」的完整字符串。
      所以想做出打字机效果，得自己记「已经打印了多少字符」，每次只打新增的那一段。
      （这是课案里专门用一句注释点出来的坑，见下面 consume_stream 里的原句保留。）

本文件的运行策略（本机大概率没起 Aegra，所以做了降级设计）：

    1. 先用 httpx 探测 http://localhost:2026/health（2 秒超时）；
    2. 通 → 走真实的 langgraph_sdk 调用；
    3. 不通 → 打印中文提示告诉你怎么把 09_aegra_deploy/aegra_project/ 跑起来，
       然后用「手工构造的假 chunk」把同一段消费逻辑真的跑一遍 ——
       这样你**看得见 SSE 事件长什么样**，而不是只看一段没跑过的代码。

    ⭐ 关键设计：在线和离线走的是**同一个 consume_stream()**，
       离线的假 chunk 只是把数据源换成了 FakeClient。
       所以你在离线模式下读懂的打印逻辑，换成真服务一个字都不用改。

课案出处：Agent 课案 → 部署 → 调用

本节要讲什么：

    1. 【那 30 行】用 langgraph_sdk 调用部署好的 Agent：threads.create() 建会话、
       runs.stream(...) 流式取结果，以及 assistant_id 到底是什么；
    2. 【核对】assistant_id 必须和 aegra.json 里 graphs 的 key 一致，
       课案「调用」写 bushu、「项目结构」写 agent，本节会读真实生成的
       aegra.json 帮你核对，并说明不一致就会 404；
    3. 【核心知识点】流式事件的数据结构：为什么 stream_mode="messages" 的事件名
       是 messages/partial（带斜杠）、为什么 chunk.data 是【两元素列表】；
    4. 【课案专门点出的坑】content 是「累计全文」而不是增量 token ——
       想做打字机效果必须自己记「已经打了多少字符」再取差量，
       本节会把「累计」和「差量」并排打印出来给你看；
    5. 【降级设计】本机没起 Aegra 也能看懂：用假 chunk 把同一段消费逻辑真跑一遍。

运行前置条件：
    · 本文件本身只依赖已装好的 httpx / langgraph_sdk，离线也能跑；
    · 想走【真实调用】分支，需要先按部署第 ② 节生成骨架、第 ③ 节把服务跑起来
      （cp .env.example .env → uv run aegra dev），并把 Docker Desktop 起着。
      探测不通不会报错，会自动降级到第 4 节的离线演示。

运行方式：
    uv run Agent/09_aegra_deploy/04_客户端调用_jxsd.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE / "aegra_project"

# 和课案一致：Aegra 默认监听 2026
BASE_URL = "http://localhost:2026"
HEALTH_URL = f"{BASE_URL}/health"

# 课案「调用」那段里写的 assistant_id 是 "bushu" ——
# 注意它**必须**和 aegra.json 里 graphs 的 key 一致，否则请求会 404。
# 课案作者的 aegra.json 里 key 就叫 bushu；而课案「项目结构」那节给的示例
# 是 "agent"。下面 register_target() 会去读真实生成的 aegra.json 帮你核对。
COURSE_ASSISTANT_ID = "bushu"


# ================================================================
# 1. 课案原文（原样保留，供对照）
# ================================================================
COURSE_SNIPPET = '''import sys

from langgraph_sdk import get_sync_client

# Windows 控制台默认 GBK 编码，遇到 emoji 会抛 UnicodeEncodeError，强制 UTF-8
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

client = get_sync_client(url="http://localhost:2026")

# 创建会话线程（状态持久化在 PostgreSQL）
thread = client.threads.create()

# 流式调用
printed = 0  # 已打印的字符数
for chunk in client.runs.stream(
    thread["thread_id"],
    "bushu",      # 对应 aegra.json 中 graphs 的 key
    input={
        "messages": [{"type": "human", "content": "你好"}]
    },
    stream_mode="messages",
):
    # messages/partial 事件的 data 是 [消息块, 元数据]
    # 注意：content 是“累计全文”而非增量 token，需自己计算差量打印
    if chunk.event == "messages/partial":
        content = chunk.data[0]["content"]
        if isinstance(content, str) and len(content) > printed:
            print(content[printed:], end="", flush=True)
            printed = len(content)
'''


def section_1_snippet() -> None:
    print("=" * 78)
    print("1. 课案原文：客户端调用（约 30 行）")
    print("=" * 78)
    for ln in COURSE_SNIPPET.rstrip("\n").splitlines():
        print("    " + ln)
    print("\n  说明：每个 graph 启动时会自动注册一个**同名默认 assistant**，")
    print("        所以 assistant_id 直接填 graph 名即可。")
    print("        同一 thread_id 连续调用就是多轮对话。")


# ================================================================
# 2. 核对 assistant_id（读真实生成的 aegra.json）
# ================================================================
def registered_graph_keys() -> list:
    """读 aegra.json，返回 graphs 里注册的所有 key（= assistant_id 候选）

    读不到就返回空列表 —— 这是探测，不抛异常。
    """
    cfg = PROJECT_DIR / "aegra.json"
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
        return list(data.get("graphs", {}).keys())
    except Exception:   # noqa: BLE001 —— 文件不存在/JSON 坏了都当成「没读到」
        return []


def section_2_assistant_id() -> str:
    """打印 assistant_id 与 aegra.json 的对应关系，返回本次实际使用的 id"""
    print("\n" + "=" * 78)
    print("2. assistant_id 从哪来：核对 aegra.json 的 graphs key")
    print("=" * 78)
    keys = registered_graph_keys()
    if keys:
        print(f"  读到 {PROJECT_DIR / 'aegra.json'}")
        print(f"  已注册的 graph（= assistant_id）：{keys}")
    else:
        print(f"  ⚪ 没读到 {PROJECT_DIR / 'aegra.json'}（先跑 02_项目骨架_jxsd.py 生成）")

    if keys and COURSE_ASSISTANT_ID in keys:
        print(f"  ✅ 课案用的 \"{COURSE_ASSISTANT_ID}\" 在已注册列表里，可直接调用")
        return COURSE_ASSISTANT_ID
    if keys:
        print(f"  ⚠️ 课案写的 \"{COURSE_ASSISTANT_ID}\" **不在**已注册列表里 → 真调用会 404。")
        print(f"     课案「调用」写 bushu、「项目结构」写 agent，两处不一致是课案的笔误；")
        print(f"     解决办法二选一：改 aegra.json 的 key，或改客户端的 assistant_id。")
        print(f"     本次演示自动改用已注册的 \"{keys[0]}\"。")
        return keys[0]
    print(f"  ⚪ 没有可用列表，本次演示按课案原值 \"{COURSE_ASSISTANT_ID}\" 走（离线演示不受影响）")
    return COURSE_ASSISTANT_ID


# ================================================================
# 3. 核心：流式消费逻辑（在线 / 离线共用同一份代码）
# ================================================================
def consume_stream(chunks, show_events: bool = False) -> None:
    """消费 `stream_mode="messages"` 的事件流，做出打字机效果

    这一段就是课案那 30 行里的循环体，**一个字都没改**，只是把
    `client.runs.stream(...)` 的结果作为参数传进来 ——
    这样在线模式（真 SSE）和离线模式（假 chunk）能跑同一份逻辑。

    show_events=True 时不打打字机，改成「一个事件一行」，把累计全文和差量并排显示，
    方便看清「content 是累计的」这件事。
    """
    printed = 0        # 已打印的字符数（课案原句：printed = 0  # 已打印的字符数）
    partial_count = 0  # 顺带统计事件数，方便观察「累计全文」是怎么长的

    for chunk in chunks:
        # messages/partial 事件的 data 是 [消息块, 元数据]
        # 注意：content 是“累计全文”而非增量 token，需自己计算差量打印
        if chunk.event == "messages/partial":
            content = chunk.data[0]["content"]
            if isinstance(content, str) and len(content) > printed:
                if show_events:
                    # 把「累计全文」和「算出来的差量」并排打出来，一眼看出为什么要 [printed:]
                    print(f"      [event] {chunk.event:<18} 累计={content!r}   →  本次只打 {content[printed:]!r}")
                else:
                    print(content[printed:], end="", flush=True)
                printed = len(content)
            partial_count += 1
        elif show_events:
            # metadata / messages/complete 等事件没有 data[0]["content"]，
            # 靠这个 else 直观说明：循环里那个 if chunk.event == 判断不是可选的。
            print(f"      [event] {chunk.event:<18} （非 partial，跳过：没有 data[0]['content']）")

    print()   # 流结束换行（打字机效果期间一直没换行）
    print(f"      ↑ 共收到 {partial_count} 个 messages/partial 事件，最终全文 {printed} 个字符")


# ================================================================
# 4. 离线演示：手工构造假 chunk
# ================================================================
class FakeChunk:
    """模拟 langgraph_sdk 的 StreamPart：只有 .event 和 .data 两个属性被用到

    真实类型是 langgraph_sdk.schema.StreamPart（NamedTuple），字段一致：
        event: str   事件名，例如 "messages/partial"
        data:  Any   事件负载，messages 模式下是 [消息块, 元数据]
    """

    def __init__(self, event: str, data):
        self.event = event
        self.data = data


def build_fake_chunks() -> list:
    """把一个回答拆成 token，再拼成「累计全文」的 messages/partial 事件序列

    ⭐ 这里是本节最值得盯着看的地方：每一次事件的 data[0]["content"] 都是
       **从头累积**的字符串，而不是那一个 token。所以课案才要
       `content[printed:]` 自己算差量。
    """
    tokens = ["你", "好", "！", "我是", "部署", "在", "Aegra", "上", "的", "助手", "。"]
    meta = {"langgraph_node": "chatbot", "tags": [], "ls_provider": "openai"}

    chunks = [
        # 真实调用时，stream 的第一个事件通常是 metadata（run 的基本信息），
        # 它没有 data[0]["content"] —— 这就是循环里必须写 `if chunk.event ==` 的原因
        FakeChunk("metadata", {"run_id": "1ef7c1a2-fake-run-id", "attempt": 1}),
    ]
    cumulative = ""
    for tok in tokens:
        cumulative += tok
        chunks.append(
            FakeChunk(
                "messages/partial",
                # data 是两元素列表：[消息块, 元数据]
                [{"content": cumulative, "type": "AIMessageChunk", "id": "lc_run--fake"}, meta],
            )
        )
    chunks.append(
        FakeChunk(
            "messages/complete",
            [{"content": cumulative, "type": "AIMessage", "id": "lc_run--fake"}, meta],
        )
    )
    return chunks


class FakeClient:
    """假的 langgraph_sdk 客户端，接口形状和真的对齐：client.threads / client.runs

    这样 consume_stream 里的调用方式（threads.create() → runs.stream(...)）
    在离线模式下也**完全一致**，学员不用在两套代码之间来回换算。
    """

    class _Threads:
        @staticmethod
        def create():
            return {"thread_id": "fake-thread-0001", "created_at": "2025-01-01T00:00:00Z"}

    class _Runs:
        @staticmethod
        def stream(thread_id, assistant_id, input, stream_mode):
            # 真客户端这里是发 HTTP 请求、逐条 yield SSE 事件；
            # 离线模式直接返回预构造好的事件列表（同样是可迭代的）
            return build_fake_chunks()

    def __init__(self):
        self.threads = self._Threads()
        self.runs = self._Runs()


def section_4_offline(assistant_id: str) -> None:
    """离线演示：用假 chunk 把课案那段差量打印逻辑真跑一遍"""
    print("\n" + "=" * 78)
    print("4. 离线演示：手工构造 messages/partial 事件，跑一遍课案那段逻辑")
    print("=" * 78)

    chunks = build_fake_chunks()
    print(f"  构造了 {len(chunks)} 个事件。开头三个的原样长这样：")
    for c in chunks[:3]:
        print(f"\n    event = {c.event!r}")
        print("    data  = " + json.dumps(c.data, ensure_ascii=False)[:150] + " ...")
    print("\n  ⚠️ 看后两个：第一个 partial 的 content 是「你」，第二个是「你好」——")
    print("     它每次都是**从开头累积**的完整字符串，不是那一个 token。")
    print("     所以课案才要 `content[printed:]` 自己算差量。")

    print("\n  ---- 下面是课案那段循环体的真实输出（打字机效果）----")
    print("      ", end="")
    client = FakeClient()
    thread = client.threads.create()          # 和真客户端同形
    consume_stream(
        client.runs.stream(
            thread["thread_id"],
            assistant_id,                      # 对应 aegra.json 中 graphs 的 key
            input={"messages": [{"type": "human", "content": "你好"}]},
            stream_mode="messages",
        ),
        show_events=False,
    )

    print("\n  ---- 再跑一次，这次一个事件一行，把「累计」和「差量」并排看 ----")
    consume_stream(FakeClient().runs.stream(
        "fake-thread-0001", assistant_id,
        input={"messages": [{"type": "human", "content": "你好"}]},
        stream_mode="messages",
    ), show_events=True)


# ================================================================
# 5. 在线模式：真的连 Aegra
# ================================================================
def health_ok() -> bool:
    """探测 Aegra 是否在跑（2 秒超时，任何异常都算「没起」）"""
    try:
        import httpx   # langgraph_sdk 的依赖，一定装了
        resp = httpx.get(HEALTH_URL, timeout=2.0)
        return resp.status_code == 200
    except Exception:   # noqa: BLE001 —— 连接失败/超时/包缺失都归为「连不上」
        return False


def section_3_health() -> bool:
    print("\n" + "=" * 78)
    print("3. 探测本机 Aegra 服务")
    print("=" * 78)
    print(f"  GET {HEALTH_URL}   （httpx，2 秒超时）")
    ok = health_ok()
    if ok:
        print('  ✅ 通！返回 {"status": "healthy"} 之类，服务已就绪')
    else:
        print("  ❌ 连不上 —— 本机现在没有 Aegra 在 2026 端口上跑")
        print("\n  想把它跑起来（完整步骤见 03_本地开发与生产部署_jxsd.py）：")
        print(f"      1) cd {PROJECT_DIR}")
        print("      2) cp .env.example .env      # 然后填上真实的 API_KEY 等（.env 不进 Git）")
        print("      3) 确认 Docker Desktop 已经启动（aegra dev 要拉 PostgreSQL 容器）")
        print("      4) 确认目录里没有别的项目留下的旧 docker-compose.yml（课案点名的坑）")
        print("      5) uv run aegra dev          # 自动：生成 compose → 拉 PostgreSQL → 迁移 → 热重载")
        print("      6) 另开一个终端：curl.exe http://localhost:2026/health")
        print("      7) 交互式文档：http://localhost:2026/docs")
        print("\n  注意 aegra 要装在独立的 3.12 环境里，不要装进 Python_Base 的 venv。")
    return ok


def section_5_online(assistant_id: str) -> None:
    """服务在跑时的真实调用 —— 与离线演示共用 consume_stream"""
    print("\n" + "=" * 78)
    print("5. 在线调用（真实 langgraph_sdk）")
    print("=" * 78)
    try:
        from langgraph_sdk import get_sync_client

        client = get_sync_client(url=BASE_URL)
        thread = client.threads.create()
        print(f"  已创建线程：{thread['thread_id']}")
        print(f"  调用 assistant_id = {assistant_id}，stream_mode = messages")
        print("  AI：", end="")
        consume_stream(
            client.runs.stream(
                thread["thread_id"],
                assistant_id,
                input={"messages": [{"type": "human", "content": "你好"}]},
                stream_mode="messages",
            ),
        )
        # 同一个 thread_id 再问一次，就是多轮对话（上下文由服务端 PostgreSQL 里的
        # checkpoint 维持，客户端不需要把历史一起发过去 —— 这正是 Threads API 的价值）
        print("\n  同一 thread 再问一次（验证多轮上下文）：")
        print("  AI：", end="")
        consume_stream(
            client.runs.stream(
                thread["thread_id"],
                assistant_id,
                input={"messages": [{"type": "human", "content": "我叫什么？"}]},
                stream_mode="messages",
            ),
        )
    except Exception as exc:   # noqa: BLE001 —— 服务端报错不该把课案脚本炸掉
        print(f"  ❌ 调用失败：{type(exc).__name__}: {exc}")
        print("     常见原因：assistant_id 和 aegra.json 的 key 不一致（会 404）；")
        print("               或服务刚起来还在跑数据库迁移，等几秒再试。")


# ================================================================
# 主流程
# ================================================================
if __name__ == "__main__":
    print("Agent 课案 · 部署 ④：客户端调用（langgraph_sdk + messages/partial 差量打印）")
    section_1_snippet()
    assistant_id = section_2_assistant_id()
    if section_3_health():
        section_5_online(assistant_id)
    else:
        section_4_offline(assistant_id)
        print("\n" + "=" * 78)
        print("小结")
        print("=" * 78)
        print("  · 客户端代码在 Aegra 和官方平台之间**完全通用**，只换 url；")
        print("  · messages/partial 的 content 是累计全文，要自己 [printed:] 算差量；")
        print("  · 事件名带斜杠（messages/partial），因为要作为 SSE 的 event 字段发给前端；")
        print("  · 循环里那个 `if chunk.event ==` 不是可选的 —— metadata 等事件没有 content。")
        print("\n  服务起来后再跑一次本文件，就会自动切到第 4 节的真实调用路径。")

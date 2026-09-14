# -*- coding: utf-8 -*-
"""
A2A 协议 · 互相通信：客户端 → DeepAgents 调度员 → CrewAI 分析师
================================================================
课案原文：启动顺序：① `python crewai_a2a_server.py` →
② `langgraph dev --port 2024 --no-browser` → ③ 运行下方客户端。
通过 `langgraph_sdk` 调用 DeepAgents，DeepAgents 自主判断是否通过 A2A 调用 CrewAI。

本节要讲什么
    1. **谁和谁通信**：三个进程、三次跳转，把 A2A 放回真实链路里看 ——
       普通 Python 脚本 →（langgraph_sdk / HTTP）→ DeepAgents 调度员
       →（A2AClient / HTTP JSON-RPC）→ CrewAI 分析师，再逐层原路返回。
       A2A 管的是后面那一跳（Agent ↔ Agent），前面那一跳是 LangGraph 平台自己的 API；
    2. **为什么顺序不能反**：调度员在模块导入时就 new 出 A2AClient 指向 :2025，
       分析师没起就会在工具里拿到 Connection refused；
    3. **客户端怎么写才稳**：`threads.create()` 先建线程、
       `assistant_id` 必须等于 `langgraph.json` 里的键名、
       `runs.wait` 的语义（等服务端整轮跑完才返回）、以及指数退避重试；
    4. **收尾为什么要 raise**：最后一次重试不再吞异常，否则函数会静默返回 None。

一次提问在三个进程之间怎么走（这才是 A2A 的「全景图」）：
    ① a2a_client.py  ──langgraph_sdk（HTTP）──►  :2024  总调度员（DeepAgents）
    ② 总调度员判断「这题要数据分析」→ 选 call_crewai_analyst 工具
    ③ 工具内部 ──A2AClient.send_message（HTTP JSON-RPC）──►  :2025  CrewAI 分析师
    ④ 分析师把报告原路返回 → 调度员润色成最终回复 → a2a_client 打印

也就是说：**同一个协议（A2A / JSON-RPC over HTTP）串起了三种技术栈**——
客户端是普通 Python 脚本，中间是 LangGraph/DeepAgents，末端是 CrewAI。
换成任何别的框架，只要它说 A2A，链路照样成立。

本文件对应课案的「A2A → 互相通信」：
    · 60 行 `a2a_client.py`（get_client + threads.create + runs.wait + 指数退避重试）
    · `$env:PYTHONUTF8="1"; .venv\\Scripts\\python a2a_client.py`

⚠️ 本机现实：2024 / 2025 两个端口都没有服务（依赖没装、也没启动）。
   降级路径：
     ① 真去连一次 2024，连不上就打印「三进程启动顺序」的中文指引；
     ② 用假 client 把 `wait_with_retry` 的**指数退避逻辑**跑通 ——
        失败两次、第三次成功，退避 1s→2s，计时打印出来，代码原样未改。

课案出处：Agent 课案 → 协议 → A2A → 互相通信

运行方式：
    # 完整链路（需先装依赖并起两个服务，见文件内提示）
    uv run Agent/07_protocols/05_a2a互相通信_jxsd.py

前置条件：
    - **能直接跑**：只要装了 langgraph_sdk（本项目已装）就能运行 ——
      连不上 2024 端口时会给中文启动指引，并用假 client 把重试逻辑跑通；
    - 想走完整真流程需要四步：
      ① `uv add a2a_auto_wrapper langgraph-api langgraph-sdk deepagents crewai`
      ② `uv add "langgraph-cli[inmem]"`（提供 `langgraph dev`）
      ③ 写两个服务文件（课案原文见 03_a2a服务端_jxsd.py、04_a2a客户端_jxsd.py）
      ④ 按 ①→②→③ 的顺序启动三个进程（顺序不能反，原因见第二节）
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，防中文/emoji 报错

import asyncio
import time

# ================================================================
# 0. 依赖探测
# ================================================================
# | 包            | 作用                                          | 本机 |
# | langgraph_sdk | get_client：通过 HTTP 调 2024 端口的调度员      | 已装 |
_HAS_LANGGRAPH_SDK = False
_IMPORT_ERRORS: list[str] = []

try:
    from langgraph_sdk import get_client

    _HAS_LANGGRAPH_SDK = True
except ImportError as exc:
    _IMPORT_ERRORS.append(f"langgraph_sdk：{exc}")

DEEPAGENT_URL = "http://localhost:2024"
# langgraph.json 里 graphs 的键名 —— assistant_id 必须和它对上，
# 否则 runs.wait 会报「assistant not found」，这是最常见的踩坑点。
ASSISTANT_ID = "deep_researcher"

# 课案里那个问题（调度员看到「分析销售数据…趋势预测」就会去调 CrewAI 分析师）
SALES_QUESTION = "分析销售数据：1月100万 2月120万 3月90万，做趋势预测"

# ================================================================
# 1. 课案原文（对照用）：a2a_client.py 的 60 行
# ================================================================
# 课案出处：Agent 课案 → 协议 → A2A → 互相通信
#
# ```python
# import asyncio
#
# from langgraph_sdk import get_client
#
#
# async def wait_with_retry(client, thread_id: str, max_attempts: int = 3) -> dict:
#     """带指数退避重试的 runs.wait 封装。
#
#     Args:
#         client: langgraph_sdk 客户端实例
#         thread_id: 会话线程 ID
#         max_attempts: 最大尝试次数（含首次）
#
#     Returns:
#         运行完成后的最终状态字典
#     """
#     for attempt in range(max_attempts):
#         try:
#             return await client.runs.wait(
#                 thread_id,
#                 assistant_id="deep_researcher",
#                 input={
#                     "messages": [
#                         {
#                             "role": "user",
#                             "content": "分析销售数据：1月100万 2月120万 3月90万，做趋势预测",
#                         }
#                     ]
#                 },
#             )
#         except Exception as e:
#             if attempt == max_attempts - 1:
#                 raise
#             wait_seconds = 2 ** attempt
#             print(f"运行失败（{e}），{wait_seconds} 秒后重试 ({attempt + 1}/{max_attempts})...")
#             await asyncio.sleep(wait_seconds)
#
#
# async def main() -> None:
#     """创建会话线程并等待 DeepAgents 调度员执行完成，打印最终回复。"""
#     client = get_client(url="http://localhost:2024")
#     thread = await client.threads.create()
#     result = await wait_with_retry(client, thread["thread_id"])
#     print(result["messages"][-1]["content"])
#
#
# if __name__ == "__main__":
#     asyncio.run(main())
# ```
#
# 课案这 60 行里藏了 4 个工程要点，值得单独拎出来：
# | 写法                              | 为什么                                                  |
# | `threads.create()` 先建线程        | LangGraph 里 thread 就是「一段会话」；同一个 thread 复用  |
# |                                   | 才有上下文记忆（对应 checkpointer 的 thread_id）          |
# | `assistant_id="deep_researcher"`  | 必须是 langgraph.json → graphs 里的**键名**，不是变量名   |
# | `runs.wait(...)`                  | 服务端把整轮跑完才返回（非流式）；要边跑边看用 runs.stream |
# | `2 ** attempt` 指数退避            | 第 1 次失败等 1s、第 2 次等 2s、第 3 次 4s…；             |
# |                                   | 避免「服务刚起、还在预热」时被客户端连环打断              |
# | `if attempt == max_attempts - 1: raise` | **最后一次不再吞异常** —— 否则重试耗尽后函数会静默返回 None，调用方拿到 None 才发现出事，排查成本更高 |
COURSE_CLIENT_PY = '''import asyncio

from langgraph_sdk import get_client


async def wait_with_retry(client, thread_id: str, max_attempts: int = 3) -> dict:
    """带指数退避重试的 runs.wait 封装。

    Args:
        client: langgraph_sdk 客户端实例
        thread_id: 会话线程 ID
        max_attempts: 最大尝试次数（含首次）

    Returns:
        运行完成后的最终状态字典
    """
    for attempt in range(max_attempts):
        try:
            return await client.runs.wait(
                thread_id,
                assistant_id="deep_researcher",
                input={
                    "messages": [
                        {
                            "role": "user",
                            "content": "分析销售数据：1月100万 2月120万 3月90万，做趋势预测",
                        }
                    ]
                },
            )
        except Exception as e:
            if attempt == max_attempts - 1:
                raise
            wait_seconds = 2 ** attempt
            print(f"运行失败（{e}），{wait_seconds} 秒后重试 ({attempt + 1}/{max_attempts})...")
            await asyncio.sleep(wait_seconds)


async def main() -> None:
    """创建会话线程并等待 DeepAgents 调度员执行完成，打印最终回复。"""
    client = get_client(url="http://localhost:2024")
    thread = await client.threads.create()
    result = await wait_with_retry(client, thread["thread_id"])
    print(result["messages"][-1]["content"])


if __name__ == "__main__":
    asyncio.run(main())
'''

# ================================================================
# 2. 启动顺序与运行命令（课案原文搬进注释）
# ================================================================
# 课案原文的启动顺序（**顺序不能反**）：
#     ① python crewai_a2a_server.py        → CrewAI 分析师，监听 2025
#     ② langgraph dev --port 2024 --no-browser → 总调度员，监听 2024
#     ③ 运行 a2a_client.py                  → 提问、等待、打印
#
# 为什么顺序不能反？因为 ② 的调度员在**模块导入时**就会 new 出
# A2AClient(base_url="http://localhost:2025/")；虽然 A2AClient 构造通常不连网，
# 但 ① 没起时，③ 一提问就会在工具里拿到 Connection refused。
#
# 运行客户端的命令（课案原文，Windows PowerShell）：
#     $env:PYTHONUTF8="1"; .venv\Scripts\python a2a_client.py
#
# 为什么要设 PYTHONUTF8？
#     Windows 下 Python 的默认 IO 编码跟随「系统区域设置」（中文系统是 GBK）。
#     CrewAI / LangGraph 的日志里有 emoji 和中文，GBK 编不出来就会
#     UnicodeEncodeError（典型现场：跑一半突然 UnicodeEncodeError: 'gbk' codec ...）。
#     $env:PYTHONUTF8="1" 让整个进程按 UTF-8 处理 stdio，一劳永逸。
#     （本文件顶部再显式 `sys.stdout.reconfigure(encoding="utf-8")` 兜底，
#       所以即使忘了设环境变量也不会崩。）
def print_startup_guide() -> None:
    """把课案的启动顺序与命令原样打出来 —— 这是本节最容易做错的一步。"""
    print("【课案原文：启动顺序（不能反）+ 运行命令】")
    print("    ① .venv\\Scripts\\python crewai_a2a_server.py          # CrewAI 分析师 :2025")
    print("    ② .venv\\Scripts\\langgraph dev --port 2024 --no-browser  # 总调度员 :2024")
    print('    ③ $env:PYTHONUTF8="1"; .venv\\Scripts\\python a2a_client.py')
    print('    # PYTHONUTF8="1"：让 stdio 走 UTF-8，避免 emoji/中文在 GBK 控制台炸掉')
    print()


# ================================================================
# 3. 课案原文的函数，原样保留（一行没改）
# ================================================================
async def wait_with_retry(client, thread_id: str, max_attempts: int = 3) -> dict:
    """带指数退避重试的 runs.wait 封装。

    Args:
        client: langgraph_sdk 客户端实例
        thread_id: 会话线程 ID
        max_attempts: 最大尝试次数（含首次）

    Returns:
        运行完成后的最终状态字典
    """
    for attempt in range(max_attempts):
        try:
            # runs.wait：服务端把**整轮**跑完才返回（非流式）。
            # 注意 assistant_id 必须是 langgraph.json → graphs 里的**键名**，
            # 不是 Python 变量名 —— 写错就拿不到图，报 assistant not found。
            return await client.runs.wait(
                thread_id,
                assistant_id=ASSISTANT_ID,   # 课案是字面量 "deep_researcher"，这里提成常量
                input={
                    "messages": [
                        {
                            "role": "user",
                            # 这句就是「让调度员自己决定要不要去调 CrewAI 分析师」的输入
                            "content": SALES_QUESTION,
                        }
                    ]
                },
            )
        except Exception as e:
            # 最后一次不再吞异常（原因见文件头第四节）——否则重试耗尽后
            # 函数会静默返回 None，调用方拿到 None 才发现出事，排查成本更高
            if attempt == max_attempts - 1:
                raise
            # 指数退避：第 1 次失败等 1s、第 2 次等 2s、第 3 次等 4s…
            # 目的：服务刚起、还在预热时，别被客户端连环打断
            wait_seconds = 2 ** attempt
            print(f"运行失败（{e}），{wait_seconds} 秒后重试 ({attempt + 1}/{max_attempts})...")
            await asyncio.sleep(wait_seconds)


# ================================================================
# 4. 降级演示：假 client，把「指数退避」这段逻辑真的跑一遍
# ================================================================
class FlakyClient:
    """失败 N 次后成功的假 client —— 专门用来验证 wait_with_retry 的重试分支。

    `.runs` 属性模仿 langgraph_sdk 的形状（client.runs.wait / client.threads.create），
    所以丢给 wait_with_retry 的那个函数**一行都不用改**。
    """

    class _Runs:
        def __init__(self, fail_times: int) -> None:
            self.fail_times = fail_times
            self.calls = 0

        async def wait(self, thread_id, assistant_id, input):  # noqa: A002 - 对齐官方签名
            self.calls += 1
            if self.calls <= self.fail_times:
                # 模拟「服务刚起、还在预热」时的连接失败
                raise ConnectionError(
                    f"Connection refused to {DEEPAGENT_URL}（{assistant_id} 尚未就绪）"
                )
            # 成功时返回的字典结构对齐课案：result["messages"][-1]["content"]
            question = input["messages"][-1]["content"]
            return {
                "messages": [
                    {"role": "user", "content": question},
                    {
                        "role": "assistant",
                        "content": (
                            "【模拟回复（假 client，非真实模型输出）】\n"
                            "已完成调度：call_crewai_analyst → A2A → :2025 数据分析师。\n"
                            "结论：1→2 月 +20%，2→3 月 -25%，属单峰波动；"
                            "建议排查 3 月环比下滑原因，Q2 目标按 1 月水平 +10% 设定。"
                        ),
                    },
                ]
            }

    class _Threads:
        def __init__(self) -> None:
            self.created = 0

        async def create(self):
            self.created += 1
            return {"thread_id": f"thread-flaky-{self.created}"}

    def __init__(self, fail_times: int) -> None:
        self.runs = FlakyClient._Runs(fail_times)
        self.threads = FlakyClient._Threads()


async def demo_retry_with_fake_client() -> None:
    """用假 client 跑 main() 的同一套流程，证明重试逻辑真的生效。"""
    print("【降级演示】用假 client 跑通 wait_with_retry 的指数退避分支")
    print("            （2024 端口没服务，但重试逻辑本身可以离线验证）")
    client = FlakyClient(fail_times=2)   # 前两次失败，第三次成功
    thread = await client.threads.create()
    print(f"    已创建线程：thread_id = {thread['thread_id']}")

    started = time.perf_counter()
    result = await wait_with_retry(client, thread["thread_id"])   # ← 课案原函数
    elapsed = time.perf_counter() - started
    print(f"    最终回复（result['messages'][-1]['content']）：")
    for line in result["messages"][-1]["content"].splitlines():
        print("        " + line)
    print(f"    共尝试 {client.runs.calls} 次，退避累计约 {elapsed:.1f} 秒"
          f"（理论：1s + 2s = 3s）")
    print()


async def demo_timeout_with_real_sdk() -> None:
    """真连一次 2024 端口：连不上就给出中文指引（不抛 traceback）。"""
    print("【探测真实服务】http://localhost:2024")
    if not _HAS_LANGGRAPH_SDK:
        print("    langgraph_sdk 未安装，跳过。安装：uv add langgraph-sdk")
        print()
        return

    client = get_client(url=DEEPAGENT_URL)
    try:
        thread = await client.threads.create()
        print(f"    连接成功！thread_id = {thread['thread_id']}")
        print("    正在提交问题并等待调度员执行（可能要几十秒，期间它会去调 :2025）…")
        result = await wait_with_retry(client, thread["thread_id"])
        # 课案最后一行：把最终回复打出来
        print(result["messages"][-1]["content"])
    except Exception as exc:
        # 连不上是**预期结果**（本机没起服务），所以给出可照做的启动清单而不是 traceback
        print(f"    连不上（{type(exc).__name__}）：{str(exc)[:140]}")
        print("    → 预期结果：本机没装依赖、也没起服务。完整链路需要：")
        print("      ① uv add a2a_auto_wrapper langgraph-api langgraph-sdk deepagents crewai")
        print('      ② uv add "langgraph-cli[inmem]"                         # 提供 langgraph dev')
        print("      ③ 写两个服务文件：crewai_a2a_server.py / deepagent_a2a.py")
        print("         （内容见 03_a2a服务端_jxsd.py、04_a2a客户端_jxsd.py 里的课案原文）")
        print("      ④ 按上面的启动顺序跑 ①②③")
        print("    上面的退避演示已经把客户端逻辑验证完了，连上服务后本文件会走成功分支。")
    print()


# ================================================================
# 5. 入口
# ================================================================
def main() -> None:
    # 结构固定：先摊开课案原文 → 再看前置条件 → 先验证客户端逻辑（降级）
    # → 最后真连一次 2024 端口。无论连不连得上，本文件都能跑完并给出结论。
    print("=" * 66)
    print("A2A 互相通信：a2a_client → DeepAgents 调度员 → CrewAI 分析师")
    print("=" * 66)
    print()

    # 逐行打印课案那 60 行客户端，方便对照后面的真实调用
    print("【课案原文】a2a_client.py（60 行）：")
    for line in COURSE_CLIENT_PY.splitlines():
        print("    " + line)
    print()

    print_startup_guide()

    print("【前置条件检查】")
    if _IMPORT_ERRORS:
        for err in _IMPORT_ERRORS:
            print(f"    - 缺失：{err}")
        print("      安装：uv add langgraph-sdk langgraph-api")
    else:
        # langgraph_sdk 是纯 HTTP 客户端，不依赖服务端存在就能 import
        print("    - langgraph_sdk：已就绪（纯 HTTP 客户端，能独立探测服务是否在线）")
    print()

    # ---------- 降级：先把客户端逻辑验证掉 ----------
    asyncio.run(demo_retry_with_fake_client())

    # ---------- 真实：连一次 2024 ----------
    asyncio.run(demo_timeout_with_real_sdk())

    print("=" * 66)
    print("小结：A2A 的完整链路 = 三次跳转、两种传输 ——")
    print("    a2a_client ──langgraph_sdk(HTTP)──► :2024 调度员")
    print("    调度员     ──A2AClient(HTTP JSON-RPC)──► :2025 CrewAI 分析师")
    print("    分析师报告 ──────────────────────────► 逐层原路返回")
    print("    每一跳都是「标准协议 + 普通 HTTP」，所以框架不同也能拼起来。")
    print("=" * 66)


if __name__ == "__main__":
    main()

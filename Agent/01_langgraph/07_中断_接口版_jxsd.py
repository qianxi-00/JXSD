# -*- coding: utf-8 -*-
"""
LangGraph 中断（二）：接口版（FastAPI）
================================================================
课案原句：**在实际项目中，使用接口进行中断。**

为什么脚本版（06）不够用？因为真实系统里「按回车继续」这个动作不是人坐在终端前敲的，
而是**审核人在前端点了一个按钮**，前端再调一个 HTTP 接口。
于是「中断现场」必须跨请求保存在服务端 —— 这正好是 checkpointer 的用武之地：

    用户 A 调 POST /start/{thread_id}   → 图跑到断点挂起，接口返回 waiting
    （等待人工审核……可能是几秒，也可能是几小时，进程甚至可能重启过）
    审核人调 POST /resume/{thread_id}   → 图从断点继续跑完，接口返回 completed

⚠️ 两个 thread_id 必须一致，它就是这份「中断现场」的取件码。

**本节的两个接口**（课案原文）：
    POST /start/{thread_id}    发起任务；若停在断点，返回 status="waiting"
    POST /resume/{thread_id}   人工审核通过，恢复执行；返回 status="completed"

**怎么跑？**
本文件不是「起个服务让你手动点」就完事——它在 `__main__` 里会**真的把 uvicorn 起起来**，
自动打一遍 /start/ 和 /resume/，把返回结果打印出来，然后自己关掉服务。
这样既能当教学演示直接跑，也能当冒烟测试用（端口用 8021，避开常见的 8000）。
    服务在子线程里跑，主线程用 requests 当客户端，测完把 server.should_exit 置位收工。
（子线程里 uvicorn 不会去装信号处理器，这是它能被塞进线程的前提。）

课案出处：Agent 课案 → langgraph → 核心组件 → 中断（第二段：接口版）
运行方式：
    uv run Agent/01_langgraph/07_中断_接口版_jxsd.py
前置条件：
    - 依赖：`langgraph` + `fastapi` + `uvicorn` + `requests`（本项目已 uv sync 装好）。
      缺包时文件头部的 import 检查会打印中文安装提示后 sys.exit(0)，不会甩 traceback。
    - 端口：占用本机 **8021**（刻意避开 8000/8080）。若被占用，
      把文件顶部的 PORT 常量改掉即可，自测脚本有超时 + 明确提示。
    - 外部服务：**不需要数据库、不需要大模型、不需要 API Key**——
      接口里的图只有 step 一个节点，没有任何模型调用。
"""

import sys
import threading
import time
from typing import TypedDict

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，防中文/emoji 报错

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

PORT = 8021  # 刻意避开 8000/8080，防止和本机已有服务撞端口

# ---------- 前置依赖检查（缺包时给中文提示，别让 import 直接炸）----------
try:
    import uvicorn
    from fastapi import FastAPI
except ImportError as exc:  # pragma: no cover - 环境缺包时才会走到
    print("=" * 74)
    print("缺少接口版所需依赖，无法运行本文件。")
    print("=" * 74)
    print(f"  缺失：{exc.name}")
    print("  安装：uv add fastapi uvicorn")
    print("  只想看中断原理的话，脚本版见 06_中断_人工审核_jxsd.py（无外部依赖）。")
    sys.exit(0)


# ============================================================
# 1. 状态、节点、图（与 06 一致，只是把「按回车」换成交互接口）
# ============================================================
class State(TypedDict):
    text: str


def step(state: State) -> dict:
    """被人工审核挡住的那个节点。"""
    print("      [服务端] step节点开始执行")
    return {"text": state["text"] + " → 已执行"}


builder = StateGraph(State)
builder.add_node("step", step)
builder.add_edge(START, "step")
builder.add_edge("step", END)

# ⚠️ 接口版**必须**用能跨请求、跨进程的 checkpointer 吗？
#    用 MemorySaver 时：中断现场存在**当前服务进程**的内存里。
#    只要服务不重启，多次 HTTP 请求之间就能接上（这已经能满足「审核人稍后点按钮」）；
#    但如果服务重启（发布、崩溃、多副本负载均衡），现场就丢了。
#    → 生产环境把这里换成 PostgresSaver(settings.pg_uri)，恢复逻辑一行都不用改。
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["step"],
)


# ============================================================
# 2. FastAPI 应用与两个接口（课案原文结构）
# ============================================================
app = FastAPI(title="LangGraph 中断接口版")


def get_config(thread_id: str):
    """把 thread_id 包装成 LangGraph 需要的 config 结构。

    抽成函数是为了在 /start 和 /resume 里**用同一个来源**生成 config——
    两边只要有一处写错（比如漏了 configurable 这层），
    就会出现「恢复时找不到现场」这种极难排查的问题。
    """
    return {
        "configurable": {
            "thread_id": thread_id
        }
    }


@app.post("/start/{thread_id}")
def start_graph(thread_id: str):
    """发起任务：图会一路跑到断点前停下。

    返回值里带上 next，前端可以据此知道「卡在哪个节点」，
    如果 next 为空就说明这个任务压根没触发中断（已经跑完了）。
    """
    config = get_config(thread_id)

    result = graph.invoke(
        {"text": "hello"},
        config=config,
    )

    snapshot = graph.get_state(config)

    return {
        "state": result,
        "next": snapshot.next,  # 待执行节点（元组 → JSON 数组）
        "status": "waiting" if snapshot.next else "completed",
    }


@app.post("/resume/{thread_id}")
def resume_graph(thread_id: str):
    """人工审核通过：从断点继续执行。

    ⚠️ 课案原文直接 invoke(None, config)。这里加了一道前置校验：
       如果这个 thread_id 没有处于中断状态（没调过 /start、或者已经恢复过了），
       invoke(None) 会拿一个空状态继续跑，行为不符合预期且难排查。
       HTTP 接口是对外边界，参数校验不能省——所以先查 snapshot.next 再放行。
    """
    config = get_config(thread_id)

    snapshot = graph.get_state(config)
    if not snapshot.next:
        return {
            "state": snapshot.values,
            "status": "no_pending_interrupt",
            "message": "该 thread_id 没有待恢复的中断（请先调用 /start/{thread_id}）",
        }

    result = graph.invoke(
        None,
        config=config,
    )

    return {
        "state": result,
        "status": "completed",
    }


@app.get("/health")
def health():
    """健康检查：自测脚本用它确认服务真的起来了（比 sleep 猜时间可靠）。"""
    return {"ok": True, "port": PORT}


# ============================================================
# 3. __main__：真的把服务起起来，自己打一遍两个接口，再关掉
# ============================================================
def _start_server_in_thread(port: int):
    """在子线程里启动 uvicorn，返回 (server, thread)。

    uvicorn.Server.run() 只在主线程安装信号处理器
    （框架内部判断 `threading.current_thread() is threading.main_thread()`），
    所以塞进子线程不会报「signal only works in main thread」。
    """
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="uvicorn-selftest", daemon=True)
    thread.start()
    return server, thread


def _wait_until_ready(port: int, timeout: float = 20.0) -> bool:
    """轮询 /health，直到服务就绪（或用完超时）。"""
    import requests

    # 为什么不用 time.sleep(2) 死等？uvicorn 在子线程里的启动耗时不可控
    # （首次导入 FastAPI 路由、端口绑定、事件循环起来都要时间），
    # 猜短了会连不上，猜长了每次自测都白等。轮询 /health 是确定性的做法。
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            # /health 是专门为自测加的最轻接口：不碰图、不碰状态，只证明进程活着
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return True
        except requests.RequestException:
            # 服务还没起来时连接会被拒绝（ConnectionError/ReadTimeout 都是它的子类），
            # 这正是预期路径：吞掉异常 → 睡 200ms → 再试，而不是让异常冒泡打断自测
            time.sleep(0.2)
    return False


def self_test() -> None:
    import requests

    print("=" * 74)
    print(f"① 在子线程里启动 uvicorn：http://127.0.0.1:{PORT}")
    print("=" * 74)
    server, thread = _start_server_in_thread(PORT)

    if not _wait_until_ready(PORT):
        print(f"  服务在 20 秒内没起来，可能端口 {PORT} 被占用。")
        print("  排查：换个端口改本文件顶部的 PORT，或用 netstat -ano | findstr 8021 看占用。")
        server.should_exit = True
        thread.join(timeout=5)
        sys.exit(0)

    print("  服务已就绪（/health 返回 200）。")
    print()

    base = f"http://127.0.0.1:{PORT}"
    thread_id = "demo-1"

    try:
        # ---------- 3.1 /start：图会停在断点 ----------
        print("=" * 74)
        print(f"② POST /start/{thread_id}  —— 发起任务，预期停在断点")
        print("=" * 74)
        resp = requests.post(f"{base}/start/{thread_id}", timeout=15)
        print(f"  HTTP {resp.status_code}")
        print(f"  返回：{resp.json()}")
        print("  ↑ status=waiting、next=['step']：图停住了，等人工审核。")
        print("    注意 text 仍是 'hello'，step 节点没跑。")
        print()

        # 断点期间再查一次状态：模拟前端「审核页面」刷新
        print("  此时再 POST 一次 /start/demo-2（另一个会话）来对比隔离性：")
        resp2 = requests.post(f"{base}/start/demo-2", timeout=15)
        print(f"  返回：{resp2.json()}")
        print("  ↑ 两个 thread_id 各自一份中断现场，互不干扰。")
        print()

        # ---------- 3.2 /resume：从断点继续 ----------
        print("=" * 74)
        print(f"③ POST /resume/{thread_id}  —— 人工审核通过，恢复执行")
        print("=" * 74)
        resp = requests.post(f"{base}/resume/{thread_id}", timeout=15)
        print(f"  HTTP {resp.status_code}")
        print(f"  返回：{resp.json()}")
        print("  ↑ status=completed、text='hello → 已执行'：断点已放行，step 跑完了。")
        print()

        # ---------- 3.3 重复 resume：校验生效 ----------
        print("=" * 74)
        print("④ 再 POST 一次 /resume  —— 校验「没有待恢复中断」的返回")
        print("=" * 74)
        resp = requests.post(f"{base}/resume/{thread_id}", timeout=15)
        print(f"  返回：{resp.json()}")
        print("  ↑ no_pending_interrupt：接口对无效恢复请求给出了明确提示，而不是静默跑飞。")
        print()
    finally:
        # ---------- 3.4 收工：让 uvicorn 退出 ----------
        print("=" * 74)
        print("⑤ 关闭服务：server.should_exit = True")
        print("=" * 74)
        server.should_exit = True
        thread.join(timeout=10)
        print(f"  uvicorn 已退出（线程存活：{thread.is_alive()}）")
        print()

    print("=" * 74)
    print("小结：脚本版靠 input() 等人按回车，接口版靠 HTTP 请求等人点按钮；")
    print("      两者背后都是同一个 checkpointer 保存的「中断现场」。")
    print("      真实项目里把 MemorySaver 换成 PostgresSaver，恢复逻辑完全不用改。")
    print("=" * 74)


if __name__ == "__main__":
    self_test()


# ============================================================
# 4. 实测结论 · 与本课案的差异 · 踩坑提示
# ============================================================
# 【实测结论】本机跑一遍自测，四步的返回值与课案描述一致：
#   ① 服务在子线程里起得来，/health 返回 {"ok": true, "port": 8021}。
#   ② POST /start/demo-1 → status="waiting"、next=["step"]、text 仍是 "hello"
#      （next 在 Python 里是元组 ('step',)，过 JSON 序列化后变成数组 ["step"]，
#       前端拿到的永远是数组，别按元组去比）。
#   ③ POST /resume/demo-1 → status="completed"、text="hello → 已执行"，
#      并且服务端日志里出现「[服务端] step节点开始执行」。
#   ④ 再 POST 一次 /resume/demo-1 → status="no_pending_interrupt"。
#   另外 /start/demo-2 与 /start/demo-1 各自一份现场，互不干扰。
#
# 【与本课案的差异】
#   1. 课案的接口版是「起服务 → 让你用浏览器/Postman 手动点」。本文件在 __main__ 里
#      **真的把 uvicorn 起起来、自动打一遍两个接口、再自己关掉**：
#      既能当教学演示直接跑，也能当冒烟测试用。图的逻辑、两个路由、恢复方式
#      与课案一字不差，多的只是这段自测脚手架。
#   2. 课案的 /resume 直接 invoke(None, config)。本文件加了一道前置校验
#      （先查 snapshot.next，没有待恢复的中断就返回 no_pending_interrupt）——
#      因为 HTTP 接口是对外边界，参数校验不能省，详见 resume_graph 的 docstring。
#   3. 课案没写 /health。本文件加它只为让自测能**确定性地**等服务就绪，
#      而不是 sleep 猜时间。
#   4. 课案的连接串/端口都是硬编码；本文件端口提成模块常量 PORT，方便避让冲突。
#
# 【踩坑提示】
#   1. uvicorn.Server.run() 只在**主线程**安装信号处理器（框架内部判断
#      `threading.current_thread() is threading.main_thread()`）。
#      如果换成 uvicorn.run(...) 而不是 Server(config).run()，塞进子线程会直接
#      报「signal only works in main thread」——这是本文件必须用 Server 对象的原因。
#   2. 收工只能靠 `server.should_exit = True` + `thread.join()`。
#      直接杀线程是不行的（Python 没有安全的 kill thread），
#      这也是为什么要把 server 和 thread 都返回出来。
#   3. MemorySaver 的中断现场在**当前进程内存**里：uvicorn --reload 重载、
#      多 worker 负载均衡、服务重启，都会让审核人点「通过」时报找不到现场。
#      生产必须换 PostgresSaver(settings.pg_uri)，此时**恢复逻辑一行都不用改**。
#   4. thread_id 是这份现场的取件码，/start 和 /resume 必须传同一个值。
#      真实系统里它通常来自业务工单号，不要用自增数字（重启后可能撞号）。
#   5. 恢复用的 invoke(None, config) 里那个 None 不是「随便填」：
#      它的语义是「不提供新输入，从保存的现场接着跑」。
#      静态断点用 None，动态 interrupt() 才用 Command(resume=值)，两者不能混。

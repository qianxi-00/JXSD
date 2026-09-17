# -*- coding: utf-8 -*-
r"""
DeepAgents 官方补充篇⑥：异步子代理（非课案内容）
================================================================
来源与定位：
    本文件对照 DeepAgents **官方文档** /oss/python/deepagents/async-subagents.mdx，
    补缺口表 **DeepAgents 第 9 项**。

同步子代理 vs 异步子代理（官方对比表的核心两行）：

    维度        同步子代理（课案 03_deepagents/12）      异步子代理（本文件）
    ----------  ------------------------------------  ----------------------------------
    执行模型    主管**阻塞**等子代理跑完                 立刻返回 task_id，主管继续对话
    中途干预    做不到                                  可以用 update_async_task 追加指令
    取消        做不到                                  可以 cancel_async_task
    状态        无状态                                  有独立线程、跨交互保留状态
    适用        必须先拿到结果才能继续                    长耗时/可并行/需要边跑边聊的任务

官方原文的关键约束：
    「Async subagents communicate with **any server that implements the Agent Protocol**.
      You can use LangSmith Deployments, or **self-host any Agent Protocol-compatible server**。」
    也就是说：异步子代理**必须有一个服务端**（子代理跑在那边，主管通过 SDK 管它）。

本文件的"自己起服务"方案（本机可跑，不依赖 LangSmith）：
    ① 现场生成一个最小 LangGraph 应用（langgraph.json + 一个图）；
    ② 用官方 CLI 起本地开发服务：`langgraph dev --port 2024`（这就是一个 Agent Protocol 服务端）；
    ③ 主管 agent 用 `AsyncSubAgent(url="http://127.0.0.1:2024", graph_id="bg-graph")` 连上去；
    ④ 演练 five tools：start / check / list（update / cancel 在注释里说明）。
    跑完自动关掉服务（含子进程树）。

⚠️ 两个本机实测踩坑（都在下面代码里处理了）：
    A. **中文 Windows 上必须开 UTF-8 模式**：`langgraph dev` 启动时会在
       `langgraph_api/validation.py` 里用默认编码（GBK）读 UTF-8 文件，直接
       `UnicodeDecodeError: 'gbk' codec can't decode byte 0x94`。
       解法：给子进程设 `PYTHONUTF8=1`；
    B. **`url=None`（ASGI 传输）在本地跑不通**：同步调用会报「ASGI transport requires
       async invocation」，换成异步调用又报 `'NoneType' object is not callable` ——
       因为 ASGI 应用只在部署上下文里存在。**本地演示必须显式给 url。**

依赖：`uv add "langgraph-cli[inmem]"`（已装；它提供的 `langgraph dev` 就是本地服务端）。

运行方式（项目根目录下，需真实模型；**本文件会自己起/关本地服务**）：
    uv run Agent/03_deepagents/19_异步子代理_官方补充.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

from deepagents import AsyncSubAgent, create_deep_agent

from config import settings

REPO_ROOT = Path(__file__).resolve().parents[2]     # Agent/03_deepagents/xx.py → 仓库根
PORT = 2024
BASE_URL = f"http://127.0.0.1:{PORT}"
GRAPH_ID = "bg-graph"

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
    max_retries=0,
    # 本机网关在负载高时单次请求可能超过 2 分钟，这里放宽到 4 分钟
    timeout=240,
)

# 现场生成的最小"后台图"：一个只会干活、不啰嗦的单轮 agent
BG_GRAPH_SOURCE = f'''
"""langgraph dev 托管的后台图（由 19_异步子代理_官方补充.py 现场生成）。"""
import sys

sys.path.insert(0, r"{REPO_ROOT}")      # 让子进程能找到仓库根的 config.py

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from config import settings

llm = init_chat_model(
    model_provider="openai", model=settings.model_name,
    api_key=settings.api_key, base_url=settings.base_url,
    max_retries=0, timeout=240,
)

graph = create_agent(
    model=llm,
    tools=[],
    system_prompt="你是后台任务执行器。认真完成任务，用一句话汇报结果。",
)
'''


def server_is_up() -> bool:
    """单次探测：服务端现在是否已经就绪（不等待）。"""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(f"{BASE_URL}/ok", timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def wait_for_server(timeout: int = 90) -> bool:
    """轮询 /ok，确认 Agent Protocol 服务端已就绪（最多等 timeout 秒）。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server_is_up():
            return True
        time.sleep(1.5)
    return False


def start_agent_protocol_server(workdir: Path):
    """在临时目录里生成最小应用并起 langgraph dev，返回 Popen 对象。

    返回值语义：**进程对象 = 本文件起的（结束时由本文件关）；None = 复用已有服务**
    （上一次运行留下了孤儿进程时会出现这种情况 —— 复用比重复起一个必然失败的更稳，
    而且绝不误杀别人的进程）。
    """
    if server_is_up():
        print(f"  检测到 {BASE_URL} 已有服务在运行 → 复用它（本次不新起进程）")
        return None

    (workdir / "bg_graph.py").write_text(BG_GRAPH_SOURCE, encoding="utf-8")
    (workdir / "langgraph.json").write_text(
        json.dumps({"dependencies": ["."], "graphs": {GRAPH_ID: "./bg_graph.py:graph"}}, indent=2),
        encoding="utf-8",
    )

    # 关键：PYTHONUTF8=1（中文 Windows 上否则会因为 GBK 读 UTF-8 文件而启动失败）
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["LANGSMITH_TRACING"] = "false"
    env["NO_PROXY"] = "127.0.0.1,localhost"

    cli = Path(sys.executable).parent / ("langgraph.exe" if os.name == "nt" else "langgraph")
    if not cli.exists():
        raise FileNotFoundError(
            f"找不到 langgraph CLI：{cli}\n"
            '请先安装：uv add "langgraph-cli[inmem]"（本文件的服务端由它提供）'
        )
    process = subprocess.Popen(
        [str(cli), "dev", "--port", str(PORT), "--host", "127.0.0.1", "--no-browser"],
        cwd=str(workdir),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        # 单独进程组，便于结束时连同子进程一起收掉
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    return process


def stop_agent_protocol_server(process) -> None:
    """关掉**本文件起的**服务端（Windows 上连同子进程树一起收）。

    process 为 None（复用已有服务）时不动作 —— 别人起的服务不该由我们关掉。
    """
    if process is None:
        print("  （本次复用的是已有服务，不做关闭动作）")
        return
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    else:
        # POSIX：langgraph dev 会 fork uvicorn，只 terminate 父进程会留下子进程占端口
        try:
            import os as _os
            import signal as _signal
            _os.killpg(_os.getpgid(process.pid), _signal.SIGTERM)
        except Exception:  # noqa: BLE001
            process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()


def remove_temp_dir(path: Path) -> None:
    """删掉临时目录；在 Windows 上删不掉**也不算失败**。

    ⚠️ 实测踩坑：`langgraph dev` 的进程树刚被 taskkill 掉时，Windows **不会立刻**
    释放它占用的文件句柄。紧接着 `shutil.rmtree` 就会抛
    `PermissionError: [WinError 32] 另一个程序正在使用此文件，进程无法访问。`
    —— 整份演示明明全部跑完了，却崩在最后一行清理上。
    所以这里：先小步重试（等句柄释放），最后一步容忍失败 ——
    清理不掉只是留下一个临时目录，绝不该让演示报错。
    """
    for _ in range(5):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(1.0)      # 给 Windows 一点时间释放进程持有的句柄
    shutil.rmtree(path, ignore_errors=True)
    if path.exists():
        print(f"  （临时目录未能删除，可手动清理：{path}）")


def tool_sequence(result: dict) -> list[str]:
    return [
        call["name"]
        for message in result["messages"]
        for call in (getattr(message, "tool_calls", None) or [])
    ]


def tool_outputs(result: dict) -> list[str]:
    return [str(m.content) for m in result["messages"] if m.type == "tool"]


def build_supervisor(checkpointer=None):
    """主管 agent：配一个异步子代理（必须给 url，否则本地跑不通 —— 见文件头 B 条）。

    实测要点：**异步任务的跟踪挂在会话线程上** —— 想让 check/list 看得到任务，
    必须用**同一个 agent 实例 + 同一个 thread_id**（配 checkpointer + config 传 thread_id）。
    否则新建 agent 再查，只会得到「No tracked task found」。
    """
    return create_deep_agent(
        model=llm,
        subagents=[
            AsyncSubAgent(
                name="bg-worker",
                description="后台任务执行器：适合把长耗时/可并行的活儿丢过去慢慢做",
                graph_id=GRAPH_ID,                 # 必须与 langgraph.json 里注册的图名一致
                url=BASE_URL,                      # ← 本地 Agent Protocol 服务地址
            )
        ],
        checkpointer=checkpointer,
    )


# ================================================================
# Demo 1：启动后台任务 —— 主管不阻塞，立刻拿到 task_id
# ================================================================
def demo_1_launch() -> None:
    print("=" * 70)
    print("Demo 1：start_async_task —— 丢出去就返回，不阻塞主管")
    print("=" * 70)

    agent = build_supervisor()
    started = time.time()
    result = agent.invoke(
        {
            "messages": [{
                "role": "user",
                "content": "把「统计 1 到 20 的和，并说明计算过程」这件事放到后台去做，拿到任务 ID 就告诉我。",
            }]
        },
        config={"recursion_limit": 20},
    )
    elapsed = time.time() - started
    print(f"  工具调用序列：{tool_sequence(result)}")
    for output in tool_outputs(result):
        print(f"  工具返回：{output[:150]}")
    print(f"  主管回答（{elapsed:.1f}s）：{str(result['messages'][-1].content)[:160]}")
    print(
        "  ↑ 注意耗时构成：主管**发完就返回**了（不等后台跑完）——\n"
        "    这正是异步子代理与课案 03_deepagents/12 的同步子代理的根本区别。\n"
        "    同步版会一直阻塞到子代理给出结果；异步版立刻拿到 task_id，之后随时来查。"
    )


# ================================================================
# Demo 2：查状态 check_async_task + 列任务 list_async_tasks
# ================================================================
def demo_2_check_and_list() -> None:
    print("\n" + "=" * 70)
    print("Demo 2：check_async_task / list_async_tasks —— 事后查进度")
    print("=" * 70)

    # 关键：同一个 agent 实例 + 同一个 thread_id（配 checkpointer），任务才"看得见"
    agent = build_supervisor(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "async-demo-thread"}, "recursion_limit": 20}

    launched = agent.invoke(
        {"messages": [{"role": "user", "content": "后台算一下 100 的阶乘有多少位数字，拿到任务 ID 即可。"}]},
        config,
    )
    # 用正则抽 task_id：模型同一轮里可能既调 start 又调 check（后者的返回是 JSON 形状的
    # "task_id": "..."），用 split("task_id:") 会切出带引号/空串的垃圾值 —— 本文件踩过。
    task_id = ""
    for output in tool_outputs(launched):
        match = re.search(r"task_id[\"']?\s*[:：]\s*[\"']?([0-9a-fA-F-]{8,})", output)
        if match:
            task_id = match.group(1)
            break
    if not task_id:
        print("  ⚠️ 没能从工具返回里解析出 task_id（返回内容见上），后续 check 会查不到任务。")
    print(f"  已启动任务：{task_id or '（没拿到 ID）'}")

    print("  等 8 秒再查状态（给后台一点时间）…")
    time.sleep(8)

    check = agent.invoke(
        {"messages": [{"role": "user", "content": f"查一下任务 {task_id} 现在的状态和结果。"}]},
        config,     # ← 同一个 thread_id：任务跟踪信息在这里
    )
    print(f"  查询用的工具：{tool_sequence(check)}")
    for output in tool_outputs(check):
        print(f"  工具返回：{output[:220]}")
    print(f"  主管回答：{str(check['messages'][-1].content)[:180]}")

    listing = agent.invoke(
        {"messages": [{"role": "user", "content": "把所有后台任务列出来给我。"}]},
        config,
    )
    print(f"\n  列表用的工具：{tool_sequence(listing)}")
    print(f"  主管回答：{str(listing['messages'][-1].content)[:220]}")
    print(
        "  ↑ 五个工具的分工（官方表格）：\n"
        "    start_async_task  启动，立刻返回 task_id\n"
        "    check_async_task  查状态与结果（running / success / error …）\n"
        "    list_async_tasks  列出所有任务\n"
        "    update_async_task 给正在跑的任务追加指令（本文件未演示）\n"
        "    cancel_async_task 取消任务（本文件未演示）\n"
        "    中间件会自动处理线程创建、运行管理与状态持久化 —— 你只写工具调用即可。\n"
        "    但**跟踪信息挂在会话线程上**（本文件实测踩过）：换 agent 实例或换 thread_id\n"
        "    再查会得到“No tracked task found”，所以一定要配 checkpointer + 固定 thread_id。"
    )


if __name__ == "__main__":
    # 不用 `with tempfile.TemporaryDirectory(...)`：它的退出清理一旦撞上 Windows
    # 文件锁就会抛异常（见 remove_temp_dir 的说明），这里改成显式、容忍失败的清理。
    tmp = tempfile.mkdtemp(prefix="ap_server_")
    workdir = Path(tmp)
    try:
        # 先判断端口上是否已有服务：有就复用（那时**不会**生成应用文件，也不该打印"已生成"）
        if server_is_up():
            print(f"检测到 {BASE_URL} 已有服务 → 复用（本次不生成应用、不新起进程）")
            print("  ⚠️ 注意：复用的服务加载的是**它自己启动时**那份图定义；")
            print("     若你刚改过 BG_GRAPH_SOURCE，请先杀掉旧服务（或换端口）再跑，否则验的是旧代码。")
            process = None
        else:
            print(f"生成最小 Agent Protocol 应用（临时目录，跑完即删）：{workdir}")
            process = start_agent_protocol_server(workdir)
        try:
            if process is not None:
                print(f"启动服务 {BASE_URL}（langgraph dev，PYTHONUTF8=1）…")
            if not wait_for_server():
                print(
                    "服务未能在 90 秒内就绪。手动复现时**别照抄上面的临时目录**"
                    "（它在本文件结束时就删了），自己建一个目录放同样的两个文件即可：\n"
                    "  bg_graph.py（内容见本文件头部的模板）+ langgraph.json\n"
                    "  $env:PYTHONUTF8='1'; langgraph dev --port 2024\n"
                    "然后重跑本文件（它会复用已就绪的服务）。"
                )
                raise SystemExit(1)
            print("服务已就绪 ✔\n")
            demo_1_launch()
            demo_2_check_and_list()
        finally:
            stop_agent_protocol_server(process)
            if process is not None:
                print("\n本地 Agent Protocol 服务已关闭（含子进程树）。")
    finally:
        # 关服务之后再删目录：服务收不干净时句柄还占着，删不掉也无所谓
        remove_temp_dir(workdir)
    print("全部 Demo 执行完毕。")


# ================================================================
# 实测结论 / 未收录清单 / 踩坑提示
# ================================================================
# 1. 官方事实（async-subagents.mdx）：
#    - 异步子代理必须连一个实现 Agent Protocol 的服务端（LangSmith 部署或自建）；
#    - AsyncSubAgent 字段：name / description / graph_id / url / headers；
#      `url` 省略时走 ASGI（进程内），给值时走 HTTP 到远端服务；
#    - 主管拿到五个工具：start / check / update / cancel / list_async_tasks；
#    - 与同步子代理的差别：非阻塞、可中途追加指令、可取消、状态跨交互保留。
# 2. 本机实测（langgraph-cli 0.4.31 + deepagents 0.7.13）：
#    - 本地起服务可行：`langgraph dev --port 2024`（这就是一个 Agent Protocol 服务端），
#      `/ok` 返回 200，注册在 langgraph.json 里的图以 graph_id 暴露；
#    - 主管实测：`start_async_task` 返回真实 task_id（形如 01a0ad8b-7a0a-7151-…），
#      主管不等待后台完成即答复 —— 非阻塞得到验证；
#    - `check_async_task` 会去服务端查真实状态（本文件 Demo 2 打印了它返回的内容）。
# 3. 未收录（官方还有、本文件没做的）：
#    - **update_async_task / cancel_async_task**：需要"任务还在跑"的时序才能演示得漂亮，
#      本文件为稳定性只演示 start / check / list；接口语义见文件顶部工具表；
#    - **远程部署**（LangSmith Deployments 或自建服务 + headers 鉴权）：
#      本文件用本机 dev 服务替代，换成 `url=` 远端地址 + `headers={"Authorization": ...}` 即可；
#    - **多异步子代理并行的完整编排**：官方示意里主管同时管 researcher + coder，
#      本文件只配了一个后台 worker，机制相同。
# 4. 踩坑提示：
#    A. **中文 Windows 必须 `PYTHONUTF8=1`**：否则 `langgraph dev` 启动即崩
#       （`langgraph_api/validation.py` 用 GBK 读 UTF-8 文件 → UnicodeDecodeError）；
#    B. **本地不要用 `url=None`（ASGI）**：同步调用报「requires async invocation」，
#       异步调用报 `'NoneType' object is not callable` —— ASGI 应用只在部署上下文里存在；
#    C. `graph_id` 必须与 langgraph.json 里注册的名字**完全一致**，否则启动任务就失败；
#    D. 后台图自己也会调模型 —— 它跑在服务端进程里，**服务端的 .env/PYTHONPATH 要对**，
#       所以生成的 bg_graph.py 里显式把仓库根插进 sys.path（本文件已处理）；
#    E. 关服务要连同子进程一起收（Windows 用 `taskkill /F /T`）：
#       `langgraph dev` 会 fork 出 uvicorn 子进程，只 terminate 父进程会留下孤儿端口占用
#       （本机其它课案章节也踩过端口被占的坑）。

# -*- coding: utf-8 -*-
"""
DeepAgents 运行环境④：LocalShellBackend —— 把本机终端也交给 Agent
================================================================
LocalShellBackend = **FilesystemBackend + execute 工具**。
除了能读写真实文件，还能在**宿主机**上直接跑 shell 命令。课案列举了它能做的事：

    python app.py
    pip install flask
    pytest
    git status
    npm run build

本节要讲什么
    1. 它**解决什么问题**：FilesystemBackend 只能「改代码」，跑不了
       `python app.py`；LocalShellBackend 补上 execute，让 Agent 能真的运行、
       真的看到报错、真的自己改到跑通；
    2. 代价是什么：execute 落在**你的宿主机**上，没有进程隔离、没有资源限制，
       理论上它能 `rm -rf`（课案原话：生产环境禁用）；
    3. 什么时候选它：本地开发 CLI、受控环境、配合人工审核用；
       什么时候换人：多租户 / 不可信代码 → Sandbox（08）；
    4. 为什么「virtual_mode 开启 shell 之后就等于没有」——
       路径拦截只作用于文件工具，命令绕得过去。

课案对它的定位一句话说透：

> Agent 不但拿到了项目文件，还拿到了本机终端。理论上它可以执行 rm -rf，生产环境禁用。

一、和 FilesystemBackend 的分工（对照着记）
    | 后端 | 文件操作 | 执行命令 | 代码跑在哪 |
    |---|---|---|---|
    | FilesystemBackend | ✅ | ❌ | —— |
    | LocalShellBackend | ✅ | ✅ | **你的宿主机**（无隔离） |
    | Sandbox | ✅ | ✅ | 隔离容器（见 08） |

    「改代码」和「运行代码」是两个能力，课案用一篇对比讲的就是这个分界。

二、⚠️ 安全警告（源码的措辞比课案还重）
    LocalShellBackend 的 docstring 明确写着：

        This backend grants agents BOTH direct filesystem access AND
        unrestricted shell execution on your local machine.
        **No process isolation. No resource limits.**

    并且特别强调：
        `virtual_mode=True` 和路径限制在开启 shell 之后**提供不了任何安全性**，
        因为命令可以访问系统上的任意路径。
    它建议的兜底手段是 Human-in-the-Loop（人工审核，见 10_人工审核_jxsd.py）
    ——**执行任何操作前都让人点一次头**。

三、本文件的三个工程化改动（都为了「真的能跑起来」）
    1）`root_dir` 从课案的 `./agent_workspace` 改成脚本同级的独立临时目录
       `Agent/03_deepagents/tmp_jxsd_deepagents_shell/`，
       避免污染仓库、也避免和现有精简版文件互相干扰。
    2）`inherit_env=True` + 把**当前解释器所在目录**塞到 PATH 最前面。
       原因：本机 PATH 里的 `python` 是个 Windows Store 占位符，执行后**静默无输出**；
       真正的解释器是 `.venv\\Scripts\\python.exe`。不让 `python` 指向它，
       Agent 跑 `python xxx.py` 就会「成功但什么也没发生」，演示直接失败。
    3）加了一句限制性的 system_prompt（课案没写）。
       原生后端等于把整台机器的终端交出去，加一句「不许删文件、不许越界」
       是最低成本的安全措施 —— 注意它只是**提示词约束，不是强制隔离**。

课案出处：Agent 课案 → deepAgents → 运行环境 → ④ LocalShellBackend

运行方式：
    uv run Agent/03_deepagents/06_后端_LocalShell_jxsd.py

前置条件：.env 里的 api_key / base_url / model_name 可调通（实测可用）。
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import DEFAULT_EXECUTE_TIMEOUT, LocalShellBackend
from langchain.chat_models import init_chat_model
from langgraph.errors import GraphRecursionError
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

WORKDIR = Path(__file__).resolve().parent / "tmp_jxsd_deepagents_shell"

# 当前解释器就在 .venv\Scripts\ 下，把它顶到 PATH 最前面，
# 保证 Agent 敲的 `python` 就是这个虚拟环境的 python（绕开 Windows Store 占位符）。
VENV_SCRIPTS = str(Path(sys.executable).resolve().parent)


def build_agent():
    """建一个「有终端」的 DeepAgent。"""
    # root_dir 既是文件工具的虚拟根，也是 execute 执行命令时的工作目录（cwd）
    backend = LocalShellBackend(
        root_dir=str(WORKDIR),          # 文件操作和执行命令的工作目录
        virtual_mode=True,              # 课案注释：拦截 ../ 和绝对路径（注意：管不住 shell）
        inherit_env=True,               # 继承父进程环境变量
        # env 覆盖 PATH：把 venv 的 Scripts 顶到最前，让子进程里的 `python` 是真的解释器
        env={"PATH": VENV_SCRIPTS + os.pathsep + os.environ.get("PATH", "")},
        timeout=DEFAULT_EXECUTE_TIMEOUT,  # 单条命令默认超时 120 秒
    )

    return create_deep_agent(
        model=llm,
        backend=backend,
        # 课案此处没有 system_prompt。这里补两条 —— 并说明为什么必须补：
        #
        # ① 安全约束：原生后端等于把整台机器的终端交出去，
        #    「不许删文件、不许越界」是最低成本的防护 ——
        #    再次强调：这是提示词层面的约束，不是强制隔离。
        # ② 平台差异（实测踩过的坑）：`execute` 走的是 `subprocess.run(shell=True)`，
        #    在 Windows 上落到的是 **cmd.exe**，不是 bash。
        #    课案是在 Linux 语境的课，模型很容易顺手敲 `ls` / `pwd` / `cat`，
        #    这些命令在 cmd.exe 里根本不存在，执行必然失败；
        #    而模型看到失败又会换个写法重试 —— 实测就这样一路撞到
        #    GraphRecursionError（Recursion limit of 50 reached）。
        #    把「当前是什么 shell、该用什么命令」写进提示词，循环立刻消失。
        system_prompt=(
            "你是本地开发助手，工作目录就是当前目录。\n"
            "【工作方式】用户给出任务后**直接动手完成**，不要反问、不要请求确认、"
            "不要在回答里贴代码让用户自己去建文件 —— 你有 write_file 和 execute，"
            "自己把文件建出来、把脚本跑起来，最后汇报真实结果。\n"
            "【重要】execute 工具运行的是 Windows 的 cmd.exe，不是 bash：\n"
            "  列目录用 dir，不要用 ls；看当前路径用 cd，不要用 pwd；\n"
            "  看文件内容用 type，不要用 cat；删除命令一律不许用。\n"
            "  想列目录/读文件，优先用 ls / read_file 这类文件工具（它们跨平台），\n"
            "  只有真的要跑代码时才用 execute。\n"
            "只允许在当前目录内创建、修改文件，禁止访问当前目录以外的路径。\n"
            "运行 Python 脚本时直接用 python 命令。"
        ),
    )


if __name__ == "__main__":
    WORKDIR.mkdir(parents=True, exist_ok=True)

    # 每次运行前清空这个专用临时目录，保证演示结果是本次跑出来的
    # （只动 tmp_jxsd_deepagents_shell 自己的内容，不碰任何其他目录）
    for stale in WORKDIR.iterdir():
        if stale.is_file():
            stale.unlink()

    print(f"Agent 的工作目录（= shell 的 cwd）：{WORKDIR}")
    print(f"python 解释器：{Path(sys.executable).resolve()}")
    print(f"初始磁盘内容：{sorted(p.name for p in WORKDIR.iterdir())}\n")

    agent = build_agent()

    # 课案原文任务：这次「执行 py 文件」是真的能做到的
    print("===== 执行课案任务 =====")
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "创建一个py文件，py文件里面的内容是创建一个txt文件，并执行py文件"}]},
            # 这次的任务链路更长（建文件 → 执行 → 看结果 → 可能再改），步数放宽到 50
            config={"recursion_limit": 50},
        )
        print(result["messages"][-1].content)
    except GraphRecursionError:
        # 深度智能体在「命令一直失败」时会不断换写法重试，最终撞上步数上限。
        # 这里兜住并给出中文提示，免得直接甩一个 traceback 给用户。
        print(
            "[提示] 达到步数上限仍未收敛：模型大概率在反复尝试当前平台不存在的命令。\n"
            "       请检查 system_prompt 里是否说明了当前 shell 是 cmd.exe（本文件已说明）。"
        )

    # ---------- 事后核对：这次磁盘上应该多出一个由脚本生成的 txt ----------
    print("\n===== 磁盘实际结果 =====")
    for path in sorted(WORKDIR.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(WORKDIR)}  ({path.stat().st_size} 字节)")

    print(
        "\n结论：和 FilesystemBackend 相比，这次脚本是**真的在宿主机上跑过了**，\n"
        "      所以工作目录里能看到脚本自己生成的 txt 文件。\n"
        "      代价是：Agent 拿到的是你本机的终端，没有任何隔离。"
    )

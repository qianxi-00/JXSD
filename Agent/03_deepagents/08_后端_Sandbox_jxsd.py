# -*- coding: utf-8 -*-
"""
DeepAgents 运行环境⑥：Sandbox —— 代码在隔离容器里跑
================================================================
LocalShellBackend 把宿主机终端交出去，生产环境不敢这么干。
Sandbox 的思路是：**文件操作 API 完全不变，但 execute 落到隔离环境里**。

一、统一沙箱接口：换 Provider 就像换数据库驱动
    DeepAgents 定义了统一的沙箱接口（`SandboxBackendProtocol`），
    由不同厂商提供实际的隔离环境。文件操作 API 不变，
    底层跑在谁家的沙箱里由你决定（课案表格）：

    | Provider | 位置 | 说明 | 需额外安装 / 配置 |
    |---|---|---|---|
    | LangSmithSandbox | 云端 | LangChain 官方云端沙箱（需付费开通） | LANGSMITH_API_KEY |
    | E2B | 云端 | 云沙箱（类似远程 Docker） | pip install e2b + API Key |
    | Daytona | 云端 | 开发环境管理平台 | pip install daytona + API Key |
    | Modal | 云端 | Modal 云函数平台 | pip install modal + Token |
    | Runloop | 云端 | Runloop 沙箱平台 | Runloop API Key |
    | 本地 VFS | 本地 | = LocalShellBackend，可直接用，不隔离进程 | 内置，无需额外安装 |

    所以「本地 VFS」那一行就是 06 那个文件 —— 它是最省事的沙箱，
    但**不隔离进程**，安全等级和最右边那列完全不同。

二、本文件包含课案里的两套代码
    ① LangSmithSandbox（云端付费）—— 课案原话就是：

           # 此代码不用跑，收费

       它的用法是「**先创建远程沙箱，再包进 backend**」：
           client  = SandboxClient()                     # 需要 LANGSMITH_API_KEY
           sandbox = client.sandbox(name="my-sandbox")   # 在云端开一台隔离环境
           agent   = create_deep_agent(model=..., backend=LangSmithSandbox(sandbox))
       注意 sandbox 是**外部资源**：用完要自己销毁，否则云端一直计费。

    ② 本地可跑的沙箱 opensandbox（Docker 方案）—— 课案步骤：
           pip install deepagents-opensandbox deepagents opensandbox opensandbox-server
           opensandbox-server init-config ~/.sandbox.toml --example docker
           opensandbox-server                                 # 启动沙箱服务，问就输 YES
       然后代码里：设 OPEN_SANDBOX_DOMAIN → SandboxSync.create(image=..., timeout=...)
       → `sandbox.commands.run("python --version")` → 包成 OpensandboxBackend → 建 Agent
       → 第 6 步 **`sandbox.kill()` 销毁沙箱，清理本地 Docker 资源**。

三、本文件的行为（规范第 5.3 条：外部依赖必须优雅降级）
    沙箱要么花钱（LangSmith 云端）、要么需要 Docker + Linux/WSL 环境 +
    额外 pip 包（opensandbox 系列，本项目 venv 里没装，且规范禁止装新依赖）。
    所以：

        - 模块 import 阶段零报错（缺包全部用 try/except 兜住）
        - `__main__` 里只做**前置条件体检 + 中文说明**，然后 sys.exit(0)
        - **绝不真去连 LangSmith**：即便 `LANGSMITH_API_KEY` 密钥存在也不会自动发起请求，
          两套演示代码都封装成函数，本文件不调用它们
        - 运行出来的是清晰的说明文字，不是 traceback

    想真的跑，按上面第二节的步骤准备环境，然后自己把对应函数调起来。

四、课案原话「# 此代码不用跑，收费」的背景（这句必须讲清楚，否则学生会困惑）
    LangSmithSandbox 落在 **LangChain 官方云端**（课案 Provider 表第一行：
    「LangChain 官方云端沙箱（需付费开通）」）。它是**付费托管服务**，
    计费口径是「沙箱实例的存在时长 + 资源占用」：
        client.sandbox(name="my-sandbox")   ← 从这一刻起就在计费
        sandbox.delete()                    ← 直到销毁才停
    所以课案直接把这 16 行标成「不用跑」——不是代码有问题，
    而是「一跑就产生真实账单」，课堂上不可能让每个学生都去开一台。
    这也是本文件把它封成 `demo_langsmith_sandbox()` **且不调用**的原因。

    与之相对，课案紧接着给了「本地可跑的沙箱」小节（opensandbox + Docker）：
        不花钱、不依赖任何云账号，在**自己的 Docker** 里起隔离容器，
        用 `SandboxSync.create(image=..., timeout=...)` 建、`sandbox.kill()` 销毁。
    两者的关系可以这么记：
        | 方案 | 谁出钱/出机器 | 隔离在哪 | 适合 |
        |---|---|---|---|
        | LangSmithSandbox（云端托管） | 付费给 LangChain | 厂商云端容器 | 生产、不想自己运维 |
        | opensandbox（自建 Docker） | 自己（本地 Docker 即可） | 本机 Docker 容器 | 学习、内网、成本敏感 |
    共同点是**对 Agent 的接口一样**（都是 Sandbox 后端，文件工具 + execute 不变），
    这正是第一节「换 Provider 就像换数据库驱动」那句话的落地。

五、Sandbox 在七种后端里的位置（课案表格：什么时候选它）
    Sandbox = FilesystemBackend + execute，但代码在隔离容器内运行，不触碰宿主机。
    适用场景（课案原话）：**生产环境、多租户、不可信代码**。
    也就是说：
        - 要跑代码但**不在乎隔离** → LocalShellBackend（06，本机终端直接给出去）；
        - 要跑代码且**必须隔离** → 就是本节，代价是外部依赖（付费云端或自建 Docker）；
        - 连代码都不用跑、只要改文件 → FilesystemBackend（05）就够，别上沙箱。
    最后一行「本地 VFS = LocalShellBackend，可直接用，**不隔离进程**」是课案
    Provider 表里的兜底选项：它是唯一不需要额外安装的「沙箱」，
    但安全性跟前面几行完全不是一个量级，别被「沙箱」两个字骗了。

课案出处：Agent 课案 → deepAgents → 运行环境 → ⑥ Sandbox（含「本地可跑的沙箱」小节）

运行方式：
    uv run Agent/03_deepagents/08_后端_Sandbox_jxsd.py

前置条件：无（本文件不会发任何网络请求，也不会启动容器）。
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from deepagents import create_deep_agent
from deepagents.backends import LangSmithSandbox   # 这个类本身是本地定义的，import 阶段不联网
from langchain.chat_models import init_chat_model
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

# LangSmith / opensandbox 的凭据都不在 config.py 里，按规范直接读环境变量
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY", "")


# ============================================================
# 课案代码 ①：LangSmithSandbox —— 先创建远程沙箱，再包进 backend（需 LANGSMITH_API_KEY）
# ============================================================
# 此代码不用跑，收费
#
# from deepagents import create_deep_agent
# from deepagents.backends import LangSmithSandbox
# from langsmith.sandbox import SandboxClient
# from langchain_openai import ChatOpenAI
# from conf import settings
#
# model = ChatOpenAI(model=settings.model_name, api_key=settings.api_key, base_url=settings.base_url)
#
# # 先创建远程沙箱，再包进 backend（需 LANGSMITH_API_KEY）
# client = SandboxClient()
# sandbox = client.sandbox(name="my-sandbox")
# agent = create_deep_agent(model=model, backend=LangSmithSandbox(sandbox))
#
# result = agent.invoke({"messages": [{"role": "user", "content": "当前目录有哪些文件"}]})
# print(result["messages"][-1].content)
def demo_langsmith_sandbox() -> None:
    """课案 ① 的可执行版本（本文件不会调用）。

    运行前提：
        - 已付费开通 LangSmith Sandbox
        - 环境变量 LANGSMITH_API_KEY 已设置
        - 云端已存在名为 my-sandbox 的沙箱
    """
    try:
        from langsmith.sandbox import SandboxClient
    except ImportError as exc:                       # pragma: no cover —— 缺包时给中文提示
        print(f"[跳过] 未能导入 langsmith.sandbox：{exc}")
        return

    # 先创建远程沙箱，再包进 backend
    # ⚠️ 注意顺序：这一行一执行，云端就开始计费（详见文件头第四节）
    client = SandboxClient()
    sandbox = client.sandbox(name="my-sandbox")
    try:
        # 把远程沙箱包成 backend —— 之后 Agent 的文件工具与 execute 都落到这台沙箱里
        agent = create_deep_agent(model=llm, backend=LangSmithSandbox(sandbox))
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "当前目录有哪些文件"}]},
            config={"recursion_limit": 50},
        )
        # 沙箱里执行命令同样会走多轮（ls → 读 → 回答），所以也放宽步数上限
        print(result["messages"][-1].content)
    finally:
        # 沙箱是云端计费资源，用完必须销毁
        try:
            sandbox.delete()
            print("已销毁远程沙箱。")
        except Exception as exc:                     # noqa: BLE001
            print(f"[警告] 沙箱销毁失败，请到 LangSmith 控制台手动清理：{exc}")


# ============================================================
# 课案代码 ②：本地可跑的沙箱（opensandbox + Docker）
# ============================================================
# 安装依赖包： pip install deepagents-opensandbox deepagents opensandbox opensandbox-server
# 生成默认配置： opensandbox-server init-config ~/.sandbox.toml --example docker
# 启动沙箱服务，选输入YES：opensandbox-server
def demo_opensandbox() -> None:
    """课案 ② 的可执行版本（本文件不会调用）。

    第 4、5 步之间对应课案原文的流程；第 6 步 sandbox.kill() 是收尾清理，
    对应课案标题「6. 销毁沙箱，清理本地 Docker 资源」。
    """
    try:
        from datetime import timedelta

        from opensandbox.sync.sandbox import SandboxSync
        from deepagents_opensandbox import OpensandboxBackend
    except ImportError as exc:                       # pragma: no cover —— 缺包时给中文提示
        print(f"[跳过] 本地沙箱依赖未安装：{exc}")
        print("       安装命令：pip install deepagents-opensandbox opensandbox opensandbox-server")
        return

    # 1. 配置本地沙箱连接（无需 API Key，直接连接本机 8080 服务）
    os.environ["OPEN_SANDBOX_DOMAIN"] = "http://localhost:8080"

    # 2. 在本地 Docker 中创建隔离沙箱容器
    sandbox = SandboxSync.create(
        image="python:3.12-slim",        # 基础 Docker 镜像，可自定义
        timeout=timedelta(seconds=300),  # 沙箱总存活时长
    )
    try:
        execution = sandbox.commands.run("python --version")
        print(execution.logs.stdout[0].text)

        # 3. 包装为 DeepAgents 兼容的执行后端
        backend = OpensandboxBackend(sandbox=sandbox)

        # 4. 创建 DeepAgent，绑定沙箱后端
        agent = create_deep_agent(
            model=llm,
            system_prompt="你是具备隔离沙箱执行权限的编程助手，所有代码和命令都在沙箱中安全运行。",
            backend=backend,
        )

        # 5. 执行任务
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "创建一个 hello.py 并运行它"}]},
            config={"recursion_limit": 50},
        )
        for message in result["messages"]:
            print(message.content)
    finally:
        # 6. 销毁沙箱，清理本地 Docker 资源
        sandbox.kill()
        print("已销毁沙箱，本地 Docker 资源已清理。")


# ============================================================
# 降级路径：体检 + 中文说明（本文件实际执行的只有这一段）
# ============================================================
if __name__ == "__main__":
    print("=" * 68)
    print("DeepAgents 后端⑥：Sandbox（隔离容器执行）")
    print("=" * 68)
    print(
        "\n本文件按课案要求**不做真实调用**，只做说明与前置条件体检。\n"
        "原因见下：\n"
    )

    # ---------- 体检 ①：云端沙箱 ----------
    # 只「报告」前置条件，不发起任何网络请求（文件头第三节说明了原因）
    print("① LangSmithSandbox（云端沙箱）")
    print("   课案原话：# 此代码不用跑，收费")
    if LANGSMITH_API_KEY:
        # 只打印长度，绝不回显密钥本身
        print(f"   环境变量 LANGSMITH_API_KEY：已设置（长度 {len(LANGSMITH_API_KEY)}）")
    else:
        print("   环境变量 LANGSMITH_API_KEY：**未设置**")
    print("   还需：LangSmith Sandbox 已付费开通 + 云端已存在沙箱实例。")
    print("   ⚠️ 本文件不会自动连接 LangSmith；确认要跑请自行调用 demo_langsmith_sandbox()。\n")

    # ---------- 体检 ②：本地 opensandbox（Docker） ----------
    # 逐项检查「自建沙箱」的两个前提：Python 侧依赖 + Docker 运行时
    print("② opensandbox（本地 Docker 沙箱）")
    missing = []
    for module_name in ("opensandbox", "deepagents_opensandbox"):
        try:
            # __import__ 而不是 import：模块名是变量，且这里只想探测「在不在」
            __import__(module_name)
            print(f"   {module_name:<24} 已安装")
        except ImportError:
            missing.append(module_name)
            print(f"   {module_name:<24} **未安装**")

    docker_ok = False
    try:
        # 只查 docker 可执行文件在不在 PATH 里，不启动任何容器
        # （shutil.which 是「查得到就算有」，比 subprocess 跑 `docker version` 更轻、更快）
        import shutil
        docker_ok = shutil.which("docker") is not None
    except Exception:                                # noqa: BLE001
        # 探测本身失败也不该让体检崩掉 —— 一律当作「不可用」继续往下走
        docker_ok = False
    print(f"   docker 可执行文件{'已找到' if docker_ok else '**未找到**'}")

    if missing or not docker_ok:
        print(
            "\n   [跳过] 本地沙箱的前置条件不满足，无法演示。\n"
            "   准备步骤（课案原文，建议在 Linux / WSL 里做）：\n"
            "       pip install deepagents-opensandbox deepagents opensandbox opensandbox-server\n"
            "       opensandbox-server init-config ~/.sandbox.toml --example docker\n"
            "       opensandbox-server        # 启动沙箱服务，交互提示选 YES\n"
            "   然后确认本机 8080 端口可用（OPEN_SANDBOX_DOMAIN=http://localhost:8080）。\n"
        )

    # ---------- 结论与替代方案 ----------
    print("=" * 68)
    print(
        "结论：两个沙箱方案都依赖外部条件（付费云端 / Docker + 额外 pip 包），\n"
        "      本项目规范禁止新增依赖，故本文件只打印说明并正常退出。\n"
        "\n不花钱也能体验「Agent 执行命令」的两条路：\n"
        "  - 06_后端_LocalShell_jxsd.py：内置的「本地 VFS」，直接能跑，但不隔离进程；\n"
        "  - 09_后端_Composite_jxsd.py：把 execute 交给沙箱、文件路由到磁盘的混合方案。\n"
        "\n真要上生产，请用 Sandbox 类后端（LangSmith / E2B / Daytona / Modal / Runloop\n"
        "或自建 Docker），并配合 10_人工审核_jxsd.py 做危险操作审批。"
    )
    sys.exit(0)

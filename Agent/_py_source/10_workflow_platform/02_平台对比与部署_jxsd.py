# -*- coding: utf-8 -*-
"""
可视化工作流平台：Coze / Dify / n8n / Langflow 对比与部署
================================================================
课案出处：Agent 课案 → 工作流 → 可视化平台 → coze / dify / n8n / langflow / 对比

本节要讲什么：

    1. 【四个平台各自是谁】Coze / Dify / n8n / Langflow 的定位差异与开源情况 ——
       开源与否决定的不是「免不免费」，而是「能不能私有化部署、能不能改源码」；
    2. 【怎么选】把课案那张对比表翻译成四条能直接照对的决策规则（见下）；
    3. 【n8n 部署】课案那段 docker run 的每个参数在干什么，
       尤其是那个「数据目录建了 /data/n8n、实际却挂在命名卷上」的细节；
    4. 【Dify 配置】arxiv 检索工具的 URL 模板逐参数拆解
       （search_query 的字段前缀、{{topic}} 模板变量、sortBy 为什么必须加）；
    5. 【Langflow 部署】开发环境（conda + uv + SSRF 开关）与
       生产环境（docker compose + 管理员口令）两套命令逐行说明，
       外加搭 RAG 之前用 ollama 拉向量模型那一步；
    6. 【本机实测】docker 命令与守护进程是否就绪、langflow 是否已装、
       7860 端口有没有被占用。

四平台一句话画像（课案「对比」那节的表格）：

    | 平台     | 定位                                                     | 开源 |
    |----------|----------------------------------------------------------|------|
    | Coze     | 面向普通用户，适合日常使用                               | 否   |
    | Dify     | 面向开发者                                               | 是   |
    | n8n      | 面向专业开发者                                           | 是   |
    | Langflow | 可视化 LangChain/LangGraph，面向专业开发者，适合定制化开发 | 是   |

怎么选（把课案的表格翻译成决策规则）：

    · 我不想写代码，只想拖几个节点做个bot给同事用         → Coze
    · 我要做一个能上线的 RAG / Agent 应用，团队有后端      → Dify
    · 我要把 100 个系统串起来做自动化（GitHub、飞书、数据库）→ n8n
    · 我已经在用 LangChain/LangGraph，想有个可视化调试界面  → Langflow

本文件把课案里三段**部署/配置命令**原样保留为常量字符串，并逐行加中文注释：

    1. n8n 的 docker 部署命令（mkdir -p /data/n8n 那段）
    2. Dify 里配置的 arxiv 论文检索工具 URL 模板
    3. Langflow 的 conda 安装 + docker compose 生产环境启动命令

运行时还会检查本机 docker 命令是否可用并给出提示 —— 这三段里有两段依赖 Docker。

★ 口令一律用占位符 ★
    课案 Langflow 生产环境那步写的是 LANGFLOW_SUPERUSER_PASSWORD=123456，
    本文件改成 LANGFLOW_SUPERUSER_PASSWORD=YOUR_LANGFLOW_SUPERUSER_PASSWORD，
    绝不把任何真实口令写进代码。

运行前置条件：无（不装任何东西，纯打印 + 本机命令探测）。

运行方式：
    uv run Agent/10_workflow_platform/02_平台对比与部署_jxsd.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，防中文/emoji 报错

import shutil
import subprocess

# ================================================================
# 课案原文命令块（常量字符串，原样保留）
# ================================================================
# 为什么把命令写成常量而不是直接在 print 里写：
#   一是课案原文可以【逐字保留】，改动只发生在注释里，对照时不会失真；
#   二是这些块会被 print_command_block() 打印 + 配一段「逐行说明」，
#      把「原文」和「讲解」分成两个来源，读者一眼能分清哪句是课案的、哪句是补充的。
#   三是有五段（n8n / dify / langflow 开发 / langflow 生产 / ollama），
#      集中放在文件头部，方便横向比较各家的安装方式。
#
# 注意：这五段都是【给别人在自己机器上执行】的原文，本文件不会真的执行它们 ——
#       本文件只做打印 + 本机环境探测（第 5 节）。
#
# --- ① n8n：课案「n8n → 安装」里的 shell 块（Linux 服务器上执行）---
N8N_INSTALL_CMD = """mkdir -p /data/n8n
# 赋予node用户（UID=1000）读写权限
sudo chown -R 1000:1000 /data/n8n
sudo chmod -R 755 /data/n8n

docker run -d --rm --name n8n-tunnel -p 5678:5678 -v n8n_test_data:/home/node/.n8n -e N8N_SECURE_COOKIE=false n8nio/n8n:latest start --tunnel
"""

# --- ② Dify：课案「dify → 论文追踪」里配置的 HTTP 请求工具 URL 模板 ---
DIFY_ARXIV_URLS = """https://export.arxiv.org/api/query?search_query=all:{{topic}}&start=0&max_results=5&sortBy=submittedDate


https://export.arxiv.org/api/query?search_query=cat:cs.AI&max_results=5
"""

# --- ③ Langflow：课案「langflow → 安装 → 开发环境」的 conda 流程 ---
LANGFLOW_DEV_CMD = """# 创建 Conda 环境
conda create -n langflow python=3.11 -y

# 激活环境
conda activate langflow

# 安装高性能依赖安装器
python -m pip install --upgrade uv

# 使用 uv 安装 Langflow
uv pip install --python "$env:CONDA_PREFIX\\python.exe" langflow

# 验证安装
langflow --version
python -m pip check

# 设置当前 PowerShell 会话的环境变量
# 完全关闭 SSRF 防护。这样 Langflow 可以访问本机数据库、Ollama、Docker 服务等，
# 但如果 Langflow 暴露到公网，会有安全风险
$env:LANGFLOW_SSRF_PROTECTION_ENABLED = "false"
$env:LANGFLOW_SSRF_ALLOWED_HOSTS = "localhost,127.0.0.1"

# 启动 Langflow
langflow run --host 127.0.0.1 --port 7860
"""

# --- ④ Langflow：课案「langflow → 启动 → 生产环境」的 docker compose 流程 ---
LANGFLOW_PROD_CMD = """git clone https://github.com/langflow-ai/langflow.git
cd langflow/docker_example
# 创建一个带有 Langflow 管理员密码的文件：.env
# 默认管理员用户名是 langflow
LANGFLOW_SUPERUSER_PASSWORD=YOUR_LANGFLOW_SUPERUSER_PASSWORD
docker compose up
# 访问 http://localhost:7860/
"""

# --- ⑤ Langflow：课案「搭建 RAG 问答」里下载本地向量模型的命令 ---
OLLAMA_CMD = """ollama pull bge-m3
ollama run bge-m3  "这是一个测试文本"


ollama tun qwen3
"""


# ================================================================
# 通用表格打印
# ================================================================
def display_width(text: str) -> int:
    """中文按 2 列算，否则中文表格会歪。"""
    return sum(2 if "\u2e80" <= ch <= "\u9fff" or "\uff00" <= ch <= "\uff60" else 1 for ch in text)


def print_table(headers: list[str], rows: list[list[str]], widths: list[int]) -> None:
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(line)
    print("| " + " | ".join(h + " " * (w - display_width(h)) for h, w in zip(headers, widths)) + " |")
    print(line)
    for row in rows:
        cells = [c.split("\n") for c in row]
        for i in range(max(len(c) for c in cells)):
            parts = [
                (c[i] if i < len(c) else "") + " " * (w - display_width(c[i] if i < len(c) else ""))
                for c, w in zip(cells, widths)
            ]
            print("| " + " | ".join(parts) + " |")
        print(line)


def print_command_block(title: str, command: str, notes: list[str]) -> None:
    """打印一段课案原文命令，然后逐行给出中文注释。"""
    print(f"  【{title}】课案原文：")
    print("  " + "-" * 74)
    for line in command.rstrip("\n").splitlines():
        print("  | " + line)
    print("  " + "-" * 74)
    print("  逐行说明：")
    for note in notes:
        print("    · " + note)
    print()


# ================================================================
# 1. 四平台对比
# ================================================================
def section_compare() -> None:
    print("=" * 78)
    print("1. 四平台对比（课案「对比」那节）")
    print("=" * 78)
    print()

    print_table(
        ["平台", "定位", "开源"],
        [
            ["Coze", "面向普通用户，适合日常使用", "否"],
            ["Dify", "面向开发者", "是"],
            ["n8n", "面向专业开发者", "是"],
            ["Langflow", "可视化 LangChain/LangGraph，\n面向专业开发者，适合定制化开发", "是"],
        ],
        [12, 48, 8],
    )
    print()
    print("  地址：Coze https://code.coze.cn/home    Dify https://cloud.dify.ai/")
    print()
    print(
        """  补充说明（课案表格之外的取舍依据）：

    · 开源与否决定的不是「免不免费」，而是「能不能私有化部署、能不能改源码」。
      Coze 闭源 → 数据必须过它的云；Dify / n8n / Langflow 都能整包丢进内网。
    · 「面向开发者」的差别在于抽象层次：
        Dify   把 RAG 链路（分段、检索、重排、引用）做成了开箱即用的节点；
        n8n    把「集成」做成了几百个现成节点，强项是跨系统自动化；
        Langflow 直接暴露 LangChain/LangGraph 的组件，能细调到代码级，
                 代价是你要懂 LangChain 的概念（Chain / Retriever / Tool）。
    · 本仓库 Agent/ 目录下那些 LangGraph 代码，配一个 Langflow 界面调试是最顺的；
      如果只是想把「每天抓 GitHub 热榜发到群里」跑起来，n8n 更省事。
"""
    )


# ================================================================
# 2. n8n 部署
# ================================================================
def section_n8n() -> None:
    print("=" * 78)
    print("2. n8n：安装（课案「n8n → 安装」）")
    print("=" * 78)
    print()
    print_command_block(
        "n8n docker 部署（在 Linux 服务器上执行）",
        N8N_INSTALL_CMD,
        [
            "mkdir -p /data/n8n —— 建数据目录。n8n 的工作流定义、凭证都存这里，"
            "所以必须落在宿主机上，不能只靠容器内的匿名卷。",
            "# 赋予node用户（UID=1000）读写权限 —— n8n 官方镜像里的 node 用户 UID 固定是 1000，"
            "容器以该用户运行；不 chown 的话容器起得来但写不进数据，工作流保存会失败。",
            "sudo chown -R 1000:1000 /data/n8n —— 把属主改成 1000:1000。",
            "sudo chmod -R 755 /data/n8n —— 属主可读写执行、其他用户可读可执行。",
            "docker run —— 注意课案这里挂的是【命名卷】n8n_test_data:/home/node/.n8n，"
            "而不是上面刚建的 /data/n8n；两者不冲突，但要知道真正的数据在命名卷里"
            "（docker volume inspect n8n_test_data 可以查到宿主机路径）。",
            "-d --rm —— 后台运行，容器退出后自动删除（--rm 与持久化数据是兼容的，"
            "因为数据在卷里；但不适合生产，生产别加 --rm）。",
            "--name n8n-tunnel —— 容器名，后面 docker logs/stop 都用它。",
            "-p 5678:5678 —— 宿主机 5678 → 容器 5678，n8n 默认端口。",
            "-e N8N_SECURE_COOKIE=false —— 允许用 http（非 https）访问。"
            "n8n 默认要求 https 才给登录，本地调试必须关掉，否则登录页一直转圈。",
            "n8nio/n8n:latest start --tunnel —— 用官方镜像启动。"
            "start --tunnel 会开一条隧道，让外部服务（如 GitHub Webhook）能回调到你本机，"
            "开发 webhook 流程时必开；纯本地用可以去掉 --tunnel。",
            "国内网络拉取慢的话，可换镜像加速或指定具体版本号代替 :latest。",
        ],
    )
    print(
        """  课案「使用（github热榜）」那条流程的思路（图看不懂也能照做）：
      Schedule Trigger（定时）→ HTTP Request（GitHub Trending 页面）
      → HTML/Code 节点提取仓库名 → 格式化 → 推送到 IM。
      这一步的价值在于演示 n8n 的核心卖点：不用写代码就能把
      「定时 + HTTP + 解析 + 通知」串起来。
"""
    )


# ================================================================
# 3. Dify
# ================================================================
def section_dify() -> None:
    print("=" * 78)
    print("3. Dify：论文追踪（课案「dify → 论文追踪」）")
    print("=" * 78)
    print()
    print("  课案在 Dify 里配置的 arxiv 检索工具 URL 模板：")
    print("  " + "-" * 74)
    for line in DIFY_ARXIV_URLS.rstrip("\n").splitlines():
        print("  | " + line)
    print("  " + "-" * 74)
    print(
        """  逐参数说明（这是 Dify 里「自定义工具 / HTTP 请求节点」的典型配置）：

    · https://export.arxiv.org/api/query —— arxiv 官方 API，
      export 子域返回 Atom XML，比 www 页面更适合程序解析，且不限流那么狠。
    · search_query=all:{{topic}} —— 双花括号是【Dify 的模板变量】语法，
      运行时由用户在对话里填的 topic 替换。all: 表示在标题+摘要+作者里全文搜；
      换成 ti: 只搜标题、au: 只搜作者、cat: 只搜分类。
    · start=0&max_results=5 —— 从第 0 条开始取 5 条。分页就改 start。
    · sortBy=submittedDate —— 按投稿时间排序（默认按相关度）。
      做「论文追踪」必须加这个，否则拿到的是一堆老论文。
    · 第二条 https://export.arxiv.org/api/query?search_query=cat:cs.AI&max_results=5
      是「只看 cs.AI 分类的最新 5 篇」的固定 URL，适合做每日推送。

  接进 Dify 之后的一般流程：
      开始节点（收 topic）→ HTTP 请求节点（上面这个 URL）→
      LLM 节点（把 XML 结果总结成中文摘要）→ 结束节点（输出）。

  小坑：arxiv 返回的是 XML，Dify 的 HTTP 节点默认按 JSON 解析会失败，
       要么在节点里选「文本」响应类型再用代码节点解析，
       要么在 URL 后面加 &x=1 之类的干扰参数骗过缓存（课案没提，但社区常用）。
"""
    )


# ================================================================
# 4. Langflow：安装 + 启动 + 生产部署 + RAG 搭建
# ================================================================
def section_langflow() -> None:
    print("=" * 78)
    print("4. Langflow：安装与启动（课案「langflow → 安装 / 启动」）")
    print("=" * 78)
    print()
    print_command_block(
        "开发环境：conda 建环境 + uv 装包 + 启动",
        LANGFLOW_DEV_CMD,
        [
            "conda create -n langflow python=3.11 -y —— 单独开一个环境。"
            "Langflow 依赖树很重，和项目主环境混装极易冲突，这是必要的隔离。",
            "conda activate langflow —— 激活环境；后面所有命令都在这个环境里执行。",
            "python -m pip install --upgrade uv —— 装 uv 作为安装器。"
            "Langflow 依赖上百个包，用 uv 比 pip 快一个数量级。",
            'uv pip install --python "$env:CONDA_PREFIX\\python.exe" langflow —— '
            "关键点：uv 默认不认 conda 环境，必须用 --python 显式指向 "
            "$env:CONDA_PREFIX（conda 当前环境的根目录）下的解释器，"
            "否则会装到别处，装完 langflow 命令找不到。",
            "langflow --version / python -m pip check —— 验证安装 + 检查依赖冲突。",
            "$env:LANGFLOW_SSRF_PROTECTION_ENABLED = \"false\" —— 关掉 SSRF 防护。"
            "Langflow 默认禁止工作流访问内网地址，但你的本地数据库、Ollama 都在内网，"
            "不关就一调一个报错。⚠ 只有在 Langflow 不暴露公网时才关。",
            "$env:LANGFLOW_SSRF_ALLOWED_HOSTS = \"localhost,127.0.0.1\" —— "
            "更保险的做法：不整体关闭，只把本机加进白名单。",
            "langflow run --host 127.0.0.1 --port 7860 —— 启动。"
            "只监听 127.0.0.1 是安全的默认；要让同事访问才改成 0.0.0.0。",
            "启动后浏览器访问 http://localhost:7860/，就能拖节点搭流程了。",
        ],
    )
    print("  课案「启动」那节就是上面这段的浓缩版（不需要重装时只跑这三行）：")
    print("      conda activate langflow")
    print('      $env:LANGFLOW_SSRF_PROTECTION_ENABLED = "false"')
    print("      langflow run --host 127.0.0.1 --port 7860")
    print()

    print_command_block(
        "生产环境：docker compose（课案「生产环境」五步）",
        LANGFLOW_PROD_CMD,
        [
            "git clone https://github.com/langflow-ai/langflow.git —— 拉源码，"
            "生产部署用的是仓库里现成的 docker_example，不是 pip 包。",
            "cd langflow/docker_example —— 切到示例目录，里面有 docker-compose.yml"
            "（含 Langflow + PostgreSQL 两个服务，生产必须外接数据库，不能用 SQLite）。",
            "创建 .env 文件 —— compose 会自动读同目录的 .env。",
            "LANGFLOW_SUPERUSER_PASSWORD=... —— 管理员口令。"
            "★ 课案原文写的是 123456，本文件按规范改成占位符 "
            "YOUR_LANGFLOW_SUPERUSER_PASSWORD；真实部署请用强口令，"
            "并且 .env 绝对不能提交进 Git。",
            "默认管理员用户名是 langflow —— 用户名不用配，是固定的。",
            "docker compose up —— 起服务；加 -d 可以后台运行（课案没加）。",
            "访问 http://localhost:7860/ —— 生产环境同样是 7860 端口，"
            "和开发环境一致，所以本机不要同时开两个。",
        ],
    )
    print_command_block(
        "搭 RAG 问答之前：用 ollama 拉本地向量模型（课案「搭建 RAG 问答」第 2 步）",
        OLLAMA_CMD,
        [
            "ollama pull bge-m3 —— 拉 BGE-M3 向量模型（多语言，中文效果好），"
            "在 Langflow 里把它配成 Embedding 节点，就不用花 OpenAI 的钱。",
            'ollama run bge-m3 "这是一个测试文本" —— 验证模型能跑。',
            "ollama tun qwen3 —— 课案原文如此，应为 `ollama run qwen3`（笔误），"
            "作用是拉一个本地大模型作为 LLM 节点；本文件保留原文以便对照。",
            "前置：先按 https://ollama.com/ 装好 ollama 并让它常驻。",
        ],
    )


# ================================================================
# 5. 本机环境探测：docker 可用吗？
# ================================================================
def section_check_docker() -> None:
    print("=" * 78)
    print("5. 本机环境检查：上面有三段命令依赖 Docker")
    print("=" * 78)

    docker_path = shutil.which("docker")
    if docker_path is None:
        print("  ✗ 本机没有找到 docker 命令。")
        print("    → n8n 那段 docker run、Langflow 生产环境的 docker compose up 都跑不了。")
        print("    → 装 Docker Desktop for Windows：https://www.docker.com/products/docker-desktop/")
        print("    → 装完重开终端，再跑本文件确认。")
        print("    → 只想用 Langflow 的话，走第 4 节的 conda 开发环境方式，不需要 Docker。")
        return

    print(f"  ✓ 找到 docker：{docker_path}")

    # `docker info` 会去连守护进程：命令存在 ≠ 守护进程在跑（Docker Desktop 没启动就是这种）
    try:
        proc = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        print(f"  ！执行 docker info 失败：{type(exc).__name__}: {exc}")
        return

    if proc.returncode == 0 and proc.stdout.strip():
        print(f"  ✓ Docker 守护进程在跑，Server 版本：{proc.stdout.strip()}")
        print("    → 第 2 节的 n8n 命令、第 4 节的 docker compose 命令都可以直接用。")
        print("    → n8n 起来后访问 http://localhost:5678 设置管理员账号。")
    else:
        print("  ！docker 命令在，但守护进程没响应（Docker Desktop 没启动？）")
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        if detail:
            print(f"    最后一行输出：{detail[-1][:160]}")
        print("    → 启动 Docker Desktop，等托盘图标变绿后重跑本文件。")

    # 顺带看看 langflow 是否已装、7860 是否已被占用（和 01 那节的探测呼应）
    print()
    print("  顺带检查 Langflow 相关的两个前提：")
    langflow_path = shutil.which("langflow")
    print(f"    langflow 命令：{langflow_path if langflow_path else '未安装（走第 4 节的 conda 安装流程）'}")

    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        listening = sock.connect_ex(("127.0.0.1", 7860)) == 0
    print(f"    127.0.0.1:7860 ：{'已在监听（Langflow 可能正在运行）' if listening else '没有服务监听（Langflow 未启动）'}")
    if not listening:
        print("    → 此时跑 01_langflow_api_jxsd.py 会走 dry-run 分支并打印中文排障提示，属预期行为。")


if __name__ == "__main__":
    section_compare()
    section_n8n()
    section_dify()
    section_langflow()
    section_check_docker()

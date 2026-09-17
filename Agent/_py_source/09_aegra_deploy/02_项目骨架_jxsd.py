# -*- coding: utf-8 -*-
"""
部署 ②：Aegra 项目骨架 —— 前置条件 / 安装 / 项目结构 / aegra.json
================================================================
本节要讲什么：

    1. 【前置条件】跑 Aegra 需要什么：Python 3.11+ / Docker / Redis，
       以及为什么这三样里后两样是「自动拉起」而不是你自己 docker run；
    2. 【安装】课案那三行 conda 命令的每一行在干什么、为什么要单独建环境，
       以及本仓库默认的 uv 等价写法；
    3. 【五个命令】aegra init / dev / up / down / serve 各自做什么，
       其中 aegra serve 在 Windows 上跑不了（不是配置问题，换参数也没用）；
    4. 【项目结构】骨架里每个文件分别是干什么的，课案的目录树长什么样，
       以及本项目相对课案的三处偏差（为什么用 .env.example、为什么不生成 conf.py）；
    5. 【本节最值钱的一条】aegra.json 里 dependencies 的语义与 langgraph.json
       完全不同 —— 一个是「加进 sys.path 的目录列表」，一个是「pip 依赖列表」；
    6. 【动手】真的在 09_aegra_deploy/aegra_project/ 下生成一整套骨架，
       打印目录树，并做一次冒烟导入（真的 import 生成的 graph.py）。

本节讲的是「动手之前的地基」，对应课案的「前置条件」和「项目结构」两节，
分三块：

    1. 环境要求与安装命令：Python 3.11+ / Docker / Redis，以及那三行 conda 安装；
    2. 五个常用命令：aegra init / dev / up / down / serve（**serve 在 Windows 上跑不了**）；
    3. 项目骨架的每个文件分别是干什么的，重点是 aegra.json 的两个字段。

⭐ 本节最值得记住的一个知识点，是 `dependencies` 字段的语义：

    | 配置文件        | dependencies 的语义                        |
    |----------------|--------------------------------------------|
    | langgraph.json | 依赖包列表（pip 语义，会去装包）              |
    | aegra.json     | **加入 sys.path 的目录列表**（不装任何东西）   |

    所以 Aegra 里的 `"dependencies": ["./"]` 是把**项目根目录塞进 sys.path**，
    目的是让 `from config import settings` 这种「同项目内互相 import」能生效。
    它和 langgraph.json 里的「声明依赖」完全是两回事，抄配置时最容易踩。

⭐ 本节不只是打印讲解，还会**真的在这台机器上生成一整套骨架**：

    09_aegra_deploy/aegra_project/
    ├── aegra.json          # 部署配置（格式兼容 langgraph.json，但它不认 JSON 注释）
    ├── .env.example        # 只放 docker-compose 要用的容器/端口变量（不进 Git 的真 .env 另说）
    ├── Dockerfile          # 生产镜像（aegra up 用）
    ├── requirements.txt
    ├── docker-compose.yml  # 课案 405-487 行那份，口令全部改成 change-me 占位
    └── my_agent/
        ├── __init__.py
        └── graph.py        # 你的 Graph 定义

    ⚠️ **不生成 conf.py**：本仓库（Python_Base）已经有一套根目录统一配置
       config.py + .env，所有 `_jxsd` 代码一律 `from config import settings`，
       不另起第二套。课案的 conf.py 原文只在运行时打印出来给你对照。

    生成完之后还会打印目录树，并做一次「冒烟导入」——
    真的把生成的 graph.py import 进来，确认它不是一个坏的骨架。

⚠️ 口令安全：本文件生成的所有文件里，数据库口令一律是 `change-me` 这类占位符。
   课案 .env 里出现过真实口令，**任何真实口令都不要写进模板、不要提交进仓库**，
   只在本地那份 .env 里填（.env 已在 .gitignore 里）。

课案出处：Agent 课案 → 部署 → 前置条件 / 项目结构 / aegra.json 配置 / my_agent/graph.py

运行方式：
    uv run Agent/09_aegra_deploy/02_项目骨架_jxsd.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import importlib
import shutil
import unicodedata
from pathlib import Path

# 本文件所在目录 = 09_aegra_deploy/；骨架生成到它下面的 aegra_project/
HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE / "aegra_project"
# Python_Base 仓库根目录。
# HERE = F:\ProGram\Python_Base\Agent\09_aegra_deploy
#        parents[0] = Agent\  →  parents[1] = F:\ProGram\Python_Base（仓库根）
# 所以这里取 parents[1]，不是 parents[2]。写错这一层，下面的冒烟导入就会
# 找不到 config 模块，报 ModuleNotFoundError: No module named 'config'。
REPO_ROOT = HERE.parents[1]


# ================================================================
# 小工具：按显示宽度对齐打印表格
# ================================================================
def _disp_width(text: str) -> int:
    """按东亚字符宽度计算字符串在终端里占的列数"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _disp_width(text))


def print_table(title: str, headers: list, rows: list) -> None:
    widths = [
        max(_disp_width(headers[i]), *(_disp_width(str(r[i])) for r in rows))
        for i in range(len(headers))
    ]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(f"\n【{title}】")
    print(line)
    print("| " + " | ".join(_pad(str(headers[i]), widths[i]) for i in range(len(headers))) + " |")
    print(line)
    for row in rows:
        print("| " + " | ".join(_pad(str(row[i]), widths[i]) for i in range(len(row))) + " |")
    print(line)


# ================================================================
# 1. 前置条件
# ================================================================
# 课案原表（注释形式保留）：
# | 要求 | 说明 |
# |---|---|
# | Python 3.11+ | Aegra 硬性要求 |
# | Docker | aegra dev / aegra up 会自动拉起 PostgreSQL（pgvector/pgvector:pg18） |
# | Redis | 仅生产模式 aegra up 使用，也是自动拉起 |
def section_1_prerequisites() -> None:
    print("=" * 78)
    print("1. 前置条件")
    print("=" * 78)
    print_table(
        "前置条件",
        ["要求", "说明"],
        [
            ["Python 3.11+", "Aegra 硬性要求（本项目课案用 3.12）"],
            ["Docker", "aegra dev / aegra up 会自动拉起 PostgreSQL（pgvector/pgvector:pg18）"],
            ["Redis", "仅生产模式 aegra up 使用，也是自动拉起"],
        ],
    )
    # 注意 Docker 和 Redis 是「自动拉起」：你不需要自己 docker run 一个 PostgreSQL，
    # Aegra 会生成 docker-compose.yml 并帮你把容器起来（下一节 03 会看到这个过程）。
    print("\n>>> Docker / Redis 都是「自动拉起」，不用自己 docker run 起数据库。")
    print(">>> 但本机必须装了 Docker 并且 Docker 引擎处于运行状态（Windows 上就是 Docker Desktop 开着）。")


# ================================================================
# 2. 安装命令
# ================================================================
# 课案原文（新建 3.12 环境）：
#     conda create -n aegra python=3.12 -y
#     conda activate aegra
#     pip install aegra-cli aegra-api langgraph langchain-openai pydantic-settings
#
# 说明几点：
#   · 课案用 conda 建独立环境，是因为 Aegra 的依赖（尤其是 langgraph 版本）
#     和你现有项目未必兼容 —— 隔离环境是最省事的做法；
#   · 三个包分工不同：aegra-cli 是命令行工具（init/dev/up/down/serve），
#     aegra-api 是运行时（FastAPI 服务本体），其余是你的 Agent 要用到的库；
#   · 如果你用 uv（本仓库的默认做法），对应的写法是：
#         uv venv --python 3.12
#         uv pip install aegra-cli aegra-api langgraph langchain-openai pydantic-settings
INSTALL_COMMANDS = [
    "conda create -n aegra python=3.12 -y",
    "conda activate aegra",
    "pip install aegra-cli aegra-api langgraph langchain-openai pydantic-settings",
]


def section_2_install() -> None:
    print("\n" + "=" * 78)
    print("2. 安装（新建 3.12 环境）")
    print("=" * 78)
    for cmd in INSTALL_COMMANDS:
        print("    " + cmd)
    print("\n  为什么单独建环境：Aegra 对 langgraph 版本有要求，")
    print("  塞进现有项目环境容易和已有依赖打架，隔离最省事。")
    print("  三个包的分工：aegra-cli = 命令行；aegra-api = 服务本体；其余是 Agent 用到的库。")
    print("  用 uv 的话等价写法：")
    print("    uv venv --python 3.12")
    print("    uv pip install aegra-cli aegra-api langgraph langchain-openai pydantic-settings")


# ================================================================
# 3. 五个常用命令
# ================================================================
# 课案原表（注释形式保留）：
# | 命令 | 说明 |
# |---|---|
# | aegra init | 生成模板项目（simple-chatbot / react-agent） |
# | aegra dev | 本地开发：自动拉起 PostgreSQL + 热重载 |
# | aegra up | 生产部署：PostgreSQL + Redis + 应用全部容器化启动 |
# | aegra down | 停止容器（加 --volumes 连数据卷一起删） |
# | aegra serve | 只启动应用，数据库自备。Windows 上跑不了，生产走 Docker / Linux |
COMMANDS_TABLE = [
    ["aegra init", "生成模板项目（simple-chatbot / react-agent）",
     "交互式选择模板，脚手架直接给你一个能跑的项目"],
    ["aegra dev", "本地开发：自动拉起 PostgreSQL + 热重载",
     "改代码不用重启，下一节 03 详细讲它自动做的四件事"],
    ["aegra up", "生产部署：PostgreSQL + Redis + 应用全部容器化启动",
     "在服务器上执行；等价于 docker compose up --build"],
    ["aegra down", "停止容器（加 --volumes 连数据卷一起删）",
     "注意 --volumes 会删数据卷 = 检查点/线程数据全没了"],
    ["aegra serve", "只启动应用，数据库自备。**Windows 上跑不了**",
     "生产走 Docker / Linux；Windows 本地开发请用 aegra dev"],
]


def section_3_commands() -> None:
    print("\n" + "=" * 78)
    print("3. 五个常用命令")
    print("=" * 78)
    print_table("aegra 常用命令", ["命令", "说明", "补充"], COMMANDS_TABLE)
    # aegra serve 在 Windows 上跑不了 —— 原因是它依赖 Unix 上的进程/信号模型
    # （Gunicorn/uvicorn worker 管理、SIGTERM 优雅退出那一套）。
    # 这不是配置问题，换参数也解决不了，本地开发统一用 aegra dev。
    print("\n>>> 重点：aegra serve 在 Windows 上跑不了（依赖 Unix 的进程/信号模型），")
    print("    本地开发一律用 aegra dev；生产部署在 Linux 上跑 Docker。")


# ================================================================
# 4. 项目目录树（课案原样）
# ================================================================
# 课案原文：
#     my_agent_project/
#     ├── aegra.json           # 部署配置（格式兼容 langgraph.json）
#     ├── .env                 # 环境变量
#     ├── conf.py              # 配置文件
#     ├── Dockerfile           # 生产镜像（aegra up 用）
#     ├── requirements.txt
#     └── my_agent/
#         └── graph.py         # 你的 Graph 定义
COURSE_TREE = """my_agent_project/
├── aegra.json           # 部署配置（格式兼容 langgraph.json）
├── .env                 # 环境变量
├── conf.py              # 配置文件
├── Dockerfile           # 生产镜像（aegra up 用）
├── requirements.txt
└── my_agent/
    └── graph.py         # 你的 Graph 定义"""

# ⚠️ 本项目的偏差（重要，别照抄课案）：
#   课案在项目根放了 conf.py，graph.py 里写 `from config import setting` ——
#   这行其实**同时有两个问题**：
#     ① 模块名对不上：文件叫 conf.py，却 import config；
#     ② 变量名对不上：课案 conf.py 里导出的是 `settings`，却 import 了 setting。
#   本仓库（Python_Base）已经有一套**根目录统一配置** config.py（from config import settings），
#   所以本项目**不再生成第二套配置**：graph.py 里向上找到 Python_Base 根目录插进 sys.path，
#   然后 `from config import settings`；课案的 conf.py 原文只在下面以字符串形式打印给你看，
#   不落盘、不参与运行。
COURSE_TREE_NOTE = """本项目生成的骨架与课案的三处偏差：
  1. 用 .env.example 而不是 .env —— 模板进仓库、真 .env 不进 Git（里面是真的密钥）
  2. **不生成 conf.py**，也不另建第二套 Settings —— 大模型/数据库配置全仓库只有一份来源：
     Python_Base 根目录的 config.py + .env；graph.py 会自己把根目录插进 sys.path
  3. 本目录的 .env.example 只放 docker-compose 要用的「容器/端口」变量
     （compose 只读同目录的 .env），其余 key 一个都不重复定义"""


def section_4_tree() -> None:
    print("\n" + "=" * 78)
    print("4. 项目目录树（课案原样）")
    print("=" * 78)
    for ln in COURSE_TREE.splitlines():
        print("    " + ln)
    print()
    for ln in COURSE_TREE_NOTE.splitlines():
        print("    " + ln)
    print("\n    ---- 课案原样的 conf.py（只打印对照，本项目不生成、不使用）----")
    for ln in CONF_PY.rstrip().splitlines():
        print("    " + ln)


# ================================================================
# 5. aegra.json 配置与字段说明
# ================================================================
# 课案原文：
#     {
#       "graphs": {
#         "agent": "./my_agent/graph.py:graph"
#       },
#       "dependencies": ["./"]
#     }
#
# 课案原表（注释形式保留）：
# | 字段 | 说明 |
# |---|---|
# | graphs | "agent名称": "文件路径:变量名" 映射，与 langgraph.json 相同 |
# | dependencies | 语义与 langgraph.json 不同：这里是加入 sys.path 的目录列表。
# |              | ["./"] 把项目根目录加进去，from conf import settings 才能生效 |
AEGRA_JSON = """{
  "graphs": {
    "agent": "./my_agent/graph.py:graph"
  },
  "dependencies": ["./"]
}"""


def section_5_aegra_json() -> None:
    print("\n" + "=" * 78)
    print("5. aegra.json：配置与字段说明")
    print("=" * 78)
    for ln in AEGRA_JSON.splitlines():
        print("    " + ln)

    print_table(
        "aegra.json 字段说明",
        ["字段", "说明"],
        [
            ["graphs", "\"agent名称\": \"文件路径:变量名\" 映射，与 langgraph.json 相同"],
            ["dependencies",
             "语义与 langgraph.json 不同：这里是加入 sys.path 的目录列表（\"./\" = 项目根）"],
        ],
    )

    print("\n  ◆ graphs 里的 \"agent\" 就是 assistant_id")
    print("      启动时每个 graph 会自动注册一个**同名默认 assistant**，")
    print("      所以客户端调用时 assistant_id 直接填这个 key 就行（下一节 04 用到）。")
    print("      ⚠️ 课案「调用」那段的 assistant_id 写的是 \"bushu\"，那是课案作者自己的项目名，")
    print("         它必须和这里的 key 一致，否则会 404。")
    print("\n  ◆ dependencies 的语义差异（本节最值钱的一条）")
    print("      langgraph.json 的 dependencies 是**pip 依赖列表**（要去装包）；")
    print("      aegra.json 的 dependencies 是**加进 sys.path 的目录列表**（什么都不装）。")
    print("      所以 [\"./\"] 的作用是：让项目根目录下的模块能被 import。")
    print("      这就是 graph.py 里 `from config import settings` 能跑通的原因 ——")
    print("      **不是**因为依赖装好了，而是因为项目根被塞进了 sys.path。")
    print("\n      （顺带提醒：aegra.json 是纯 JSON，**不支持注释**，")
    print("        所以本节生成的文件里它一个 # 都没有。）")


# ================================================================
# 6. 骨架文件内容（本节真正要落盘的东西）
# ================================================================
# 每个文件的正文都放在这个 dict 里，键 = 相对路径，值 = 文件内容。
# 生成物统一带一句「由 02_项目骨架_jxsd.py 生成」的标记，和课案 .py 文件区分开。

GRAPH_PY = '''# -*- coding: utf-8 -*-
"""
Aegra 项目的 Graph 定义（由 09_aegra_deploy/02_项目骨架_jxsd.py 生成）
================================================================
aegra.json 的 graphs 字段指向这里："./my_agent/graph.py:graph"
即「文件路径 : 模块级变量名」，所以下面那个 `graph = ...` 是**必须叫 graph** 的。

配置**不另起一套**：直接用 Python_Base 根目录那份 config.py / .env。
下面的循环会从本文件往上找，第一个同时含 config.py + pyproject.toml 的目录
就是 Python_Base 根目录，把它插进 sys.path，`from config import settings` 才能生效。
（aegra.json 的 "dependencies": ["./"] 只把**本项目**根目录加进 sys.path，
够不到 Python_Base 根目录，所以这里要自己补一句。这就是那个字段的真实作用。）

⚠️ 放到服务器上独立部署时，Python_Base 根目录不在场，按课案原样做即可：
   把根目录的 config.py 和 .env 一起带过去（**仍然只有这一份配置**），不用另写 conf.py。
"""

import sys
from pathlib import Path

# 向上找到 Python_Base 根目录（含 config.py + pyproject.toml 的那一层）
for _p in Path(__file__).resolve().parents:
    if (_p / "config.py").exists() and (_p / "pyproject.toml").exists():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        break

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph

from config import settings  # 根目录统一配置，不另建一套

# 课案用的是 ChatOpenAI 直连写法（部署项目自成一体，不依赖 Python_Base 的
# init_chat_model 封装），但三个参数一律来自 settings，绝不硬编码。
llm = ChatOpenAI(
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


def chatbot(state: MessagesState):
    """最简单的单节点对话：把最近 6 条消息丢给模型

    为什么只取 [-6:]：MessagesState 是无限增长的，全量传进去 token 会越来越贵。
    这里截最近 6 条做上下文窗口，是课案里最朴素的「滑动窗口记忆」写法。
    （真正要长期记忆时，应换成 checkpoint + 摘要，见 01_langgraph 那几节。）
    """
    return {"messages": [llm.invoke(state["messages"][-6:])]}


# 图的组装：一个节点、两条边，START → chatbot → END
# 用链式写法（课案原样）而不是 builder.add_node 逐行写。
graph = (
    StateGraph(MessagesState)
    .add_node("chatbot", chatbot)
    .add_edge(START, "chatbot")
    .add_edge("chatbot", END)
    .compile()
)
'''

INIT_PY = '''# -*- coding: utf-8 -*-
"""my_agent 包标记（由 09_aegra_deploy/02_项目骨架_jxsd.py 生成）

有 __init__.py，my_agent 才是一个可 import 的包，
Aegra 才能按 aegra.json 里的 "./my_agent/graph.py:graph" 找到图。
内容保持为空即可，不要在这里 import graph —— 那会让项目启动时多绕一层。
"""
'''

REQUIREMENTS_TXT = '''# Aegra 项目依赖（由 09_aegra_deploy/02_项目骨架_jxsd.py 生成）
# 课案原文这 8 行包名，各自作用见 05_对接Langfuse_jxsd.py 里的常量说明。
aegra-cli
aegra-api
langgraph
langchain
langchain-openai
deepagents
langfuse
pydantic-settings
'''

DOCKERFILE = '''# 生产镜像（由 09_aegra_deploy/02_项目骨架_jxsd.py 生成）
# aegra up 会用这个文件构建应用容器；逐行中文注释见 03_本地开发与生产部署_jxsd.py
FROM python:3.12-slim

WORKDIR /app

# 国内 PyPI 镜像加速
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 2026
CMD ["aegra", "serve", "--host", "0.0.0.0", "--port", "2026"]
'''

ENV_EXAMPLE = '''# Aegra 项目的 .env 模板（由 09_aegra_deploy/02_项目骨架_jxsd.py 生成）
#
# ⚠️ 这里**不是**第二套配置！大模型、数据库这些应用配置全仓库只有一份来源：
#       Python_Base 根目录的  F:\\ProGram\\Python_Base\\.env
#   本文件只负责 docker-compose 需要的「容器 / 端口」变量（compose 只读同目录的 .env），
#   其余 key 一个都不重复定义。
#
# 用法：cp .env.example .env   （.env 不进 Git；模板里全是占位符，不要填真值进模板）

# ---------- docker-compose 用：PostgreSQL 容器 ----------
POSTGRES_USER="my_agent_project"
POSTGRES_PASSWORD="change-me"
POSTGRES_DB="my_agent_project"
POSTGRES_HOST="localhost"
POSTGRES_PORT=5434
PORT=2026

# ---------- Langfuse 追踪开关（见 05_对接Langfuse_jxsd.py） ----------
# 这四行一加，代码零改动就能开启追踪；Aegra 通过 OTEL_TARGETS 决定往哪推 span。
# 真实 pk/sk 写在 Python_Base 根目录 .env 或本目录 .env，别写进模板。
OTEL_TARGETS="LANGFUSE"
LANGFUSE_BASE_URL="http://localhost:3000"
'''

# 课案原样的 conf.py —— **只打印给你看，不落盘**。
# 全仓库一套配置来源（Python_Base 根目录 config.py / .env），不另建第二套 Settings 类。
CONF_PY = '''# -*- coding: utf-8 -*-
"""【课案原文对照 · 本仓库不采用】课案原样的 conf.py
===============================================================
课案的部署项目里，配置放在项目根的 conf.py，由 graph.py 去 import。

⚠️ 本项目**不用**这个文件，统一用 Python_Base 根目录的 config.py：
       from config import settings
   理由：全仓库一套配置来源，避免每个子项目各自维护一份 .env 和 Settings 类。

另外课案 my_agent/graph.py 里写的是 `from config import setting`，
和这个文件对不上（文件名是 conf.py，类实例叫 settings），是课案笔误。

import os
from pydantic import ConfigDict
from pydantic_settings import BaseSettings

PATH = os.path.dirname(os.path.abspath(__file__))


class Settings(BaseSettings):
    api_key: str
    model_name: str
    base_url: str

    mysql_user: str
    mysql_password: str
    mysql_database: str
    mysql_host: str
    mysql_port: int
    baidu_qfan_api_key: str
    gitee_api_key: str
    dashscope_api_key: str
    dashscope_base_url: str
    langfuse_secret_key: str
    langfuse_public_key: str
    langfuse_host: str

    model_config = ConfigDict(
        extra="allow",
        env_file=f"{PATH}/.env",
        case_sensitive=False,
    )


settings = Settings()
"""
'''

# docker-compose.yml —— 课案 405-487 行那份，结构逐字保留。
# 唯一的改动：口令类默认值换成占位符 change-me（课案里那份默认口令值这里不复述，
# 任何像口令的字符串都不进仓库）。用户名/库名 my_agent_project 不是口令，原样保留。
DOCKER_COMPOSE_YML = '''# Docker Compose - PostgreSQL + Redis + API
# aegra dev  -> docker compose up postgres -d  (database only, in-memory broker)
# aegra up   -> docker compose up --build      (full stack, Redis broker)
# （由 09_aegra_deploy/02_项目骨架_jxsd.py 生成；口令默认值已改为 change-me 占位符）

services:
  postgres:
    image: pgvector/pgvector:pg18
    container_name: my_agent_project-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-my_agent_project}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-change-me}
      POSTGRES_DB: ${POSTGRES_DB:-my_agent_project}
    ports:
      - "${POSTGRES_PORT:-5434}:5432"
    volumes:
      - postgres_data:/var/lib/postgresql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-my_agent_project}"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: my_agent_project-redis
    restart: unless-stopped
    ports:
      - "6380:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

  my_agent_project:
    build: .
    container_name: my_agent_project-api
    restart: unless-stopped
    ports:
      - "${PORT:-2026}:${PORT:-2026}"
    env_file:
      - .env
    environment:
      - POSTGRES_USER=${POSTGRES_USER:-my_agent_project}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-change-me}
      - POSTGRES_HOST=postgres
      - POSTGRES_PORT=5432
      - POSTGRES_DB=${POSTGRES_DB:-my_agent_project}
      - AEGRA_CONFIG=aegra.json
      - AUTH_TYPE=${AUTH_TYPE:-noop}
      - PORT=${PORT:-2026}
      - REDIS_BROKER_ENABLED=true
      - REDIS_URL=redis://redis:6379/0
      # 启用 Langfuse 追踪；容器内 localhost 指向自身，需走宿主机网关访问 langfuse
      - OTEL_TARGETS=LANGFUSE
      - LANGFUSE_BASE_URL=http://host.docker.internal:3000
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:${PORT:-2026}/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    volumes:
      - ./src:/app/src:ro
      - ./aegra.json:/app/aegra.json:ro

volumes:
  postgres_data:
  redis_data:
'''

# 骨架文件清单：相对路径 → 内容。
# 键是【相对 aegra_project/ 的路径】，值就是上面那几个常量里写好的文件正文。
# 落盘循环（section_6_generate）只做三件事：拼路径 → 建父目录 → write_text，
# 所有「文件里到底写什么」都在这里，所以想改骨架内容只改这个 dict 的值即可。
#
# 七个文件各自的角色（对照课案的目录树看）：
#   aegra.json          部署配置。Aegra 启动时读它，知道「有哪些 graph、从哪 import」；
#                       等价于 langgraph.json 的位置，但 dependencies 语义不同（见第 5 节）。
#   requirements.txt    依赖清单。进 Docker 镜像时被 `pip install -r` 消费。
#   Dockerfile          生产镜像配方。aegra up 构建应用容器时用它（逐行讲解见 03 小节）。
#   .env.example        容器/端口变量的【模板】。刻意不生成 .env：
#                       真 .env 里是真实密钥，进仓库就漏了；用的时候 cp 一份再填。
#   docker-compose.yml  课案那份的等价物。aegra dev / up 内部就是调 docker compose。
#   my_agent/__init__.py 空包标记，让 my_agent 成为可 import 的包。
#   my_agent/graph.py   你的 Graph 定义。aegra.json 里 "./my_agent/graph.py:graph"
#                       指向的就是它的模块级变量 graph。
#
# ⚠️ 这里【刻意没有】conf.py：本仓库已经有一套根目录统一配置（config.py + .env），
#    骨架里的 graph.py 直接 from config import settings，不另起第二套。
#    课案的 conf.py 原文只在 section_4_tree() 里打印给你对照，不落盘、不参与运行。
SKELETON_FILES = {
    "aegra.json": AEGRA_JSON + "\n",
    "requirements.txt": REQUIREMENTS_TXT,
    "Dockerfile": DOCKERFILE,
    ".env.example": ENV_EXAMPLE,
    "docker-compose.yml": DOCKER_COMPOSE_YML,
    "my_agent/__init__.py": INIT_PY,
    "my_agent/graph.py": GRAPH_PY,
}


def section_6_generate() -> None:
    """真的把骨架写到硬盘上"""
    print("\n" + "=" * 78)
    print("6. 生成项目骨架（真的落盘）")
    print("=" * 78)
    print(f"  目标目录：{PROJECT_DIR}")

    for rel, content in SKELETON_FILES.items():
        path = PROJECT_DIR / rel
        path.parent.mkdir(parents=True, exist_ok=True)   # my_agent/ 需要自动创建
        # newline="\n" 保证 LF 换行：Dockerfile / docker-compose.yml 进 Linux 容器更稳
        # （Windows 默认会写成 CRLF，某些镜像的 shell 解析 CRLF 会出莫名其妙的错）
        path.write_text(content, encoding="utf-8", newline="\n")
        print(f"    [已写入] {rel:<24} {path.stat().st_size:>5} 字节")

    print("\n  ⚠️ 只生成了 .env.example，**没有**生成 .env ——")
    print("     真 .env 里是真实密钥，进仓库就漏了；用的时候 cp .env.example .env 再填。")
    print("     ⚠️ 这个 .env 不是可选的：docker-compose.yml 里写了 `env_file: - .env`，")
    print("        缺了它连 `docker compose config` 都会直接报")
    print("        「env file ...\\.env not found」，服务更起不来。")
    print("     实测：cp .env.example .env 之后 `docker compose config --quiet` 退出码为 0，")
    print("           说明生成的 docker-compose.yml 与 .env.example 是彼此匹配的。")
    print("\n  ⚠️ 应用配置（API_KEY / MODEL_NAME / PG_URI ...）**不在**这里配：")
    print("     全仓库统一读 Python_Base 根目录的  F:\\ProGram\\Python_Base\\.env")
    print("     graph.py 会向上找到含 config.py 的那层插进 sys.path，再 from config import settings。")


def section_7_tree_and_smoke() -> None:
    """打印生成后的目录树，并对生成的 graph.py 做一次冒烟导入"""
    print("\n" + "=" * 78)
    print("7. 生成结果：目录树 + 冒烟校验")
    print("=" * 78)
    print(f"{PROJECT_DIR.name}/")

    def walk(directory: Path, prefix: str = "") -> None:
        # 跳过 __pycache__：那是跑出来的字节码缓存，不属于骨架的组成部分
        entries = sorted(
            (p for p in directory.iterdir() if p.name != "__pycache__"),
            # 排序键 (is_file, name)：目录排在文件前面，同类按名字排序。
            # 不排序的话 iterdir() 的顺序由文件系统决定，每次打印的树都不一样。
            key=lambda p: (p.is_file(), p.name),
        )
        for i, entry in enumerate(entries):
            last = i == len(entries) - 1
            branch = "└── " if last else "├── "
            size = f"   # {entry.stat().st_size} 字节" if entry.is_file() else "/"
            print(f"{prefix}{branch}{entry.name}{size}")
            if entry.is_dir():
                walk(entry, prefix + ("    " if last else "│   "))

    walk(PROJECT_DIR)

    # ---- 冒烟校验：真的把生成的 graph.py import 进来 ----
    # aegra.json 的 "dependencies": ["./"] 会把项目根加进 sys.path，
    # 这里手工模拟同样的效果：项目根（找 my_agent 包） + 仓库根（找 config 模块）。
    print("\n  [冒烟校验] 尝试 import 生成的 my_agent.graph ...")
    for p in (str(REPO_ROOT), str(PROJECT_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        module = importlib.import_module("my_agent.graph")
        graph = module.graph
        nodes = sorted(graph.get_graph().nodes.keys())
        print(f"    ✅ import 成功，模块文件：{module.__file__}")
        print(f"    ✅ graph 对象：{type(graph).__name__}，节点：{nodes}")
        del sys.modules["my_agent.graph"]
        del sys.modules["my_agent"]
    except Exception as exc:   # noqa: BLE001 —— 骨架是生成物，坏掉必须看得见原因
        print(f"    ❌ 导入失败：{type(exc).__name__}: {exc}")
        print("       常见原因：仓库根目录的 config.py / .env 不完整，或缺 langchain-openai。")
        return

    print("\n  [说明] 当前骨架里的 graph.py 用的是 Python_Base 根目录的 config.py，")
    print("         所以在这里能 import 成功。真正部署到服务器时，把 Python_Base 根目录的")
    print("         config.py + .env 一起带过去（仍然只有这一份配置），不用另写 conf.py。")

    # 清掉刚才 import 产生的字节码缓存，让生成物目录保持干净
    removed = 0
    for cache in PROJECT_DIR.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
        removed += 1
    if removed:
        print(f"\n  [清理] 已删除 {removed} 个 __pycache__（冒烟导入的副产物）")


# ================================================================
# 主流程
# ================================================================
if __name__ == "__main__":
    print("Agent 课案 · 部署 ②：Aegra 项目骨架（前置条件 / 安装 / 项目结构 / aegra.json）")
    section_1_prerequisites()
    section_2_install()
    section_3_commands()
    section_4_tree()
    section_5_aegra_json()
    section_6_generate()
    section_7_tree_and_smoke()
    print(f"\n下一步：cd {PROJECT_DIR}")
    print("        cp .env.example .env   # 填上真实值（.env 不进 Git）")
    print("        uv run aegra dev       # 本地开发（详见 03_本地开发与生产部署_jxsd.py）")

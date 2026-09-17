# -*- coding: utf-8 -*-
"""
部署 ③：本地开发与生产部署 —— aegra dev / aegra up / Dockerfile
================================================================
本节要讲什么：

    1. 【本地开发】uv run aegra dev 这一条命令背后自动做的四件事
       （生成 compose → 拉起 PostgreSQL → 执行迁移 → 热重载启动服务），
       以及起来之后怎么验证服务正常（/health 与 /docs）；
    2. 【课案点名的坑】目录里若已有旧的 docker-compose.yml，aegra dev 会直接沿用，
       等于拿上一套项目的账号/端口去连库 —— 为什么这个坑特别难查，
       以及课案给的两条处理办法；
    3. 【生产部署】Dockerfile 逐行讲解（分层缓存、镜像加速源、exec 形式与 PID 1、
       --host 0.0.0.0 与 127.0.0.1 的区别），以及 aegra up / aegra down；
    4. 【本机实测】真的去查这台机器：docker CLI / docker 引擎 / compose 插件、
       aegra 命令、2026 端口占用，以及第 2 节那个坑在本机的真实前置条件 ——
       结论全部来自真实命令输出，不是照抄课案。

上一节把骨架搭好了，这一节讲「怎么把它跑起来」，分两段：

  【本地开发】uv run aegra dev
      一条命令自动完成四件事：
          生成 docker-compose.yml → 拉起 PostgreSQL → 执行数据库迁移 → 热重载启动服务
      然后：
          curl http://localhost:2026/health   →  {"status": "healthy"}
          交互式 API 文档                      →  http://localhost:2026/docs

  【生产部署】aegra up / aegra down
      Dockerfile 构建镜像，PostgreSQL + Redis + 应用全部容器化启动，
      所有服务自带健康检查和崩溃自动重启。

⚠️ 课案里明确点名的坑（本节会真的在本机帮你查一遍）：
      「如果目录里已有旧的 docker-compose.yml（比如之前 langgraph 部署留下的），
        aegra dev 会**直接用它**而不是生成新的，先删掉或改名。」
   这个坑很隐蔽：旧文件里的镜像 tag、端口、环境变量都是上一套项目的，
   aegra dev 却一声不吭地照用，最后表现为「数据库连不上 / 迁移失败」，
   排查方向很容易被带偏。

本文件的两段实测探测（结果按本机真实情况打印）：
    1. docker CLI / docker daemon / docker compose 是否就绪
    2. aegra 命令是否装了、2026 端口是否已被占用、
       以及 02 节刚生成的 docker-compose.yml 会不会触发上面那个坑

依赖：本文件只用到标准库（shutil / subprocess / socket），不需要 Docker 真的在跑。

课案出处：Agent 课案 → 部署 → 本地开发 / 生产部署

运行方式：
    uv run Agent/09_aegra_deploy/03_本地开发与生产部署_jxsd.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import shutil
import socket
import subprocess
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE / "aegra_project"


# ================================================================
# 小工具：按显示宽度对齐打印表格 + 安全执行外部命令
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


# 中文没有空格，textwrap 派不上用场；这里按「显示宽度」折行，
# 并优先在中文标点后断开，避免把「超时甚至失败」劈成两半。
_CJK_BREAK_AFTER = "。，；：、）】！？"


def wrap_cjk(text: str, width: int = 66) -> list:
    """按显示宽度折行，优先在中文标点后断开，返回行列表（不丢字符）"""
    lines, cur = [], ""
    for ch in text:
        cur += ch
        if _disp_width(cur) >= width:
            # 回退到最近的标点处断开（最多回退 20 个字符），找不到就硬断
            cut = max((i + 1 for i, c in enumerate(cur) if c in _CJK_BREAK_AFTER), default=-1)
            if cut >= len(cur) - 20:
                lines.append(cur[:cut])
                cur = cur[cut:]
            else:
                lines.append(cur)
                cur = ""
    if cur:
        lines.append(cur)

    # 避头点：中文排版里标点不能落在行首，把它并回上一行末尾。
    # 不做这一步的话，折行结果会出现「，超时甚至失败」这种以逗号开头的行，
    # 读起来像排版事故 —— 这是中文排版（而不是英文）特有的规则。
    for i in range(1, len(lines)):
        while lines[i] and lines[i][0] in _CJK_BREAK_AFTER:
            lines[i - 1] += lines[i][0]
            lines[i] = lines[i][1:]
    # 过滤掉被搬空的行（避免打印出空行）
    return [ln for ln in lines if ln]


def run_cmd(args: list, timeout: int = 10) -> tuple:
    """执行外部命令，返回 (是否成功, 输出文本)

    探测外部命令**绝不能抛异常**：命令不存在、超时、返回码非 0
    都要变成「一句人话」，而不是 traceback 甩到学员脸上。
    """
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            # errors="replace"：docker 在中文 Windows 上可能吐 GBK 字节，
            # 解码失败不要抛 UnicodeDecodeError，用替换字符顶过去即可 ——
            # 探测的目的是「能不能用」，不是「输出一个字都不能错」。
            errors="replace",
        )
        # stdout 为空时退回 stderr：docker 的版本信息、报错经常只出现在其中一边
        out = (proc.stdout or proc.stderr or "").strip()
        return proc.returncode == 0, out
    except FileNotFoundError:
        return False, f"命令不存在：{args[0]}"
    except subprocess.TimeoutExpired:
        return False, f"命令超时（{timeout}s）：{' '.join(args)}"
    except Exception as exc:   # noqa: BLE001 —— 探测失败本身就是一种结果
        return False, f"{type(exc).__name__}: {exc}"


# ================================================================
# 1. 本地开发：aegra dev 自动做的四件事
# ================================================================
# 课案原文：
#     # 在项目根目录执行（Docker 需处于运行状态）
#     uv run aegra dev
#
#     自动完成：生成 docker-compose.yml → 拉起 PostgreSQL → 执行数据库迁移 → 热重载启动服务。
DEV_STEPS = [
    ["① 生成 docker-compose.yml", "按 aegra.json / .env 里的参数写一份 compose 文件",
     "⚠️ 目录里已有同名文件时**直接沿用旧的**，不覆盖 —— 本节第 4 部分专门讲"],
    ["② 拉起 PostgreSQL", "docker compose up postgres -d，镜像 pgvector/pgvector:pg18",
     "开发模式只用数据库，不用 Redis（broker 走内存）"],
    ["③ 执行数据库迁移", "建 Aegra 自己的表（threads / runs / checkpoint 等）",
     "空库第一次跑才会真正建表，之后是幂等的"],
    ["④ 热重载启动服务", "uvicorn 监听 2026 端口，改代码自动生效",
     "所以开发时不用反复 Ctrl+C 重启"],
]


def section_1_local_dev() -> None:
    print("=" * 78)
    print("1. 本地开发：uv run aegra dev")
    print("=" * 78)
    print("  在项目根目录执行（Docker 需处于运行状态）：")
    print("      uv run aegra dev")
    print()
    print_table("aegra dev 自动完成的四件事", ["步骤", "做什么", "注意"], DEV_STEPS)

    # 验证服务正常 —— 课案原文：
    #     curl http://localhost:2026/health
    #     # → {"status": "healthy"}
    print("\n  验证服务正常：")
    print("      curl http://localhost:2026/health")
    print('      # → {"status": "healthy"}')
    # PowerShell 里 curl 是 Invoke-WebRequest 的别名，参数不兼容，
    # 想用真的 curl 得写 curl.exe；或者干脆用 Invoke-RestMethod。
    print("\n  ⚠️ Windows 提示：PowerShell 里 `curl` 是 Invoke-WebRequest 的别名，")
    print("     参数不通用。要么写 curl.exe，要么用 Invoke-RestMethod http://localhost:2026/health。")
    print("\n  交互式 API 文档：http://localhost:2026/docs")
    print("     （FastAPI 自带的 Swagger UI，可以直接在页面上试 threads / runs 接口，")
    print("       调试阶段比写客户端代码快得多。）")


# ================================================================
# 2. 那个坑：目录里已有旧的 docker-compose.yml
# ================================================================
# 课案原文：
#     注意：如果目录里已有旧的 docker-compose.yml（比如之前 langgraph 部署留下的），
#     `aegra dev` 会直接用它而不是生成新的，先删掉或改名。
#
# 为什么这个坑特别难查：
#   · aegra dev 不报错，它会用旧文件正常把容器起起来；
#   · 旧文件里的 POSTGRES_USER / POSTGRES_PASSWORD / 端口都是上一套项目的，
#     于是服务连不上库 → 表现为「迁移失败」或「连接被拒绝」；
#   · 你去翻 .env、翻 aegra.json 都是对的，因为问题根本不在那儿。
#
# 应对方式（课案给的两条，任选其一）：
#     del docker-compose.yml            # 删掉，让 aegra dev 重新生成
#     ren docker-compose.yml old.yml    # 或改名留档
def section_2_pitfall() -> None:
    print("\n" + "=" * 78)
    print("2. 课案点名的坑：目录里已有旧的 docker-compose.yml")
    print("=" * 78)
    print("  课案原文：")
    print("      如果目录里已有旧的 docker-compose.yml（比如之前 langgraph 部署留下的），")
    print("      aegra dev 会直接用它而不是生成新的，先删掉或改名。")
    print("\n  为什么难查：aegra dev 不报错，用旧文件照样把容器起起来；")
    print("  但旧文件里的账号/口令/端口是上一套项目的 → 服务连不上库 →")
    print("  你看到的是「迁移失败 / 连接被拒绝」，而 .env 和 aegra.json 看起来都没问题。")
    print("\n  处理办法（课案给的两条任选其一）：")
    print("      del docker-compose.yml            # 删掉，让 aegra dev 重新生成")
    print("      ren docker-compose.yml old.yml    # 改名留档")


# ================================================================
# 3. 生产部署：Dockerfile 逐行注释
# ================================================================
# 课案原文（Dockerfile）：
#     FROM python:3.12-slim
#
#     WORKDIR /app
#
#     # 国内 PyPI 镜像加速
#     ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
#
#     COPY requirements.txt .
#     RUN pip install --no-cache-dir -r requirements.txt
#
#     COPY . .
#
#     EXPOSE 2026
#     CMD ["aegra", "serve", "--host", "0.0.0.0", "--port", "2026"]
DOCKERFILE = '''FROM python:3.12-slim

WORKDIR /app

# 国内 PyPI 镜像加速
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 2026
CMD ["aegra", "serve", "--host", "0.0.0.0", "--port", "2026"]
'''

# 逐行中文注释：键 = Dockerfile 里的原文行，值 = 这一行在干什么 / 为什么这么写
DOCKERFILE_ANNOTATIONS = [
    ("FROM python:3.12-slim",
     "基础镜像。slim 版去掉了编译工具链等用不到的东西，体积从 ~1GB 降到 ~150MB。"
     "选 3.12 是因为 Aegra 要求 Python 3.11+，而课案整个环境就是 3.12。"),
    ("WORKDIR /app",
     "容器内的工作目录。后面所有相对路径（COPY 的 . 、requirements.txt）都以它为准，"
     "CMD 里的 aegra 也在这个目录下执行 —— 所以 aegra.json 必须在 /app 根。"),
    ("ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple",
     "换清华 PyPI 镜像。**只在构建阶段生效**（ENV 在 RUN 时可见），"
     "国内服务器上不换源，装依赖那一步经常几十秒超时甚至失败。"),
    ("COPY requirements.txt .",
     "先单独 COPY 依赖清单，再 RUN 安装 —— 这是 Docker 分层缓存的经典技巧："
     "只要 requirements.txt 没变，改业务代码时这一层直接命中缓存，不用重装依赖。"),
    ("RUN pip install --no-cache-dir -r requirements.txt",
     "--no-cache-dir 不保留 pip 下载缓存，镜像更小（容器里不需要二次安装，缓存纯浪费）。"),
    ("COPY . .",
     "把项目代码复制进镜像。注意要配合 .dockerignore 用，"
     "否则本地 .env（真实密钥！）也会被打进镜像 —— 生产环境应当用环境变量注入。"),
    ("EXPOSE 2026",
     "声明容器监听 2026 端口。这行只是**文档性质**，不会真的开端口，"
     "真正对外暴露靠 docker-compose 的 ports 或 docker run -p。"),
    ('CMD ["aegra", "serve", "--host", "0.0.0.0", "--port", "2026"]',
     "用 exec 数组形式（不是 shell 字符串），保证 aegra 是 PID 1，"
     "能直接收到 docker stop 的 SIGTERM 优雅退出。"
     "--host 0.0.0.0 是必须的：绑 127.0.0.1 的话容器外访问不到。"
     "（回想第 2 节：aegra serve 在 Windows 上跑不了，但这行是在 Linux 容器里跑的，没问题。）"),
]


def section_3_production() -> None:
    print("\n" + "=" * 78)
    print("3. 生产部署：Dockerfile（逐行中文注释）")
    print("=" * 78)
    for ln in DOCKERFILE.rstrip("\n").splitlines():
        print(("    " + ln) if ln.strip() else "")

    print("\n  ◆ 逐行说明：")
    for code, note in DOCKERFILE_ANNOTATIONS:
        print(f"\n    {code}")
        # 说明较长，按显示宽度折行，窄终端也读得下去
        for seg in wrap_cjk(note, 66):
            print("        " + seg)

    print("\n  在服务器上（项目目录内）：")
    print_table(
        "生产部署命令",
        ["命令", "说明"],
        [
            ["aegra up", "构建镜像并启动 PostgreSQL + Redis + 应用"],
            ["aegra down", "停止（加 --volumes 会把数据卷一起删）"],
        ],
    )
    print("\n  所有服务自带健康检查和崩溃自动重启（compose 里的 healthcheck + restart）。")


# ================================================================
# 4. 本机实测探测
# ================================================================
def section_4_probe() -> None:
    """真的去查本机环境，把结果打印出来 —— 结论必须基于真实输出"""
    print("\n" + "=" * 78)
    print("4. 本机环境实测探测")
    print("=" * 78)

    # ---- 4.1 docker CLI ----
    print("\n  [4.1] Docker")
    docker_path = shutil.which("docker")
    if docker_path:
        ok, ver = run_cmd(["docker", "--version"])
        print(f"        CLI        ：{'已安装' if ok else '存在但调用失败'}  {docker_path}")
        print(f"        版本       ：{ver if ok else ver}")
        # docker CLI 装了 ≠ Docker 引擎在跑 —— Windows 上必须开着 Docker Desktop。
        # 这一步是最容易被忽略的：CLI 在，`docker --version` 也正常，但一拉容器就报
        # "error during connect / cannot connect to the Docker daemon"。
        ok2, srv = run_cmd(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=15)
        if ok2:
            print(f"        引擎状态   ：✅ 运行中（Server {srv}）")
        else:
            print("        引擎状态   ：❌ 未运行 / 连不上")
            print("                     → Windows 上请先启动 Docker Desktop，等鲸鱼图标变绿再试")
        ok3, cver = run_cmd(["docker", "compose", "version"], timeout=15)
        print(f"        compose 插件：{'✅ ' + cver.splitlines()[0] if ok3 else '❌ ' + cver}")
        print("                     → aegra dev / up 内部就是调 docker compose，插件缺了跑不了")
    else:
        print("        ❌ 未找到 docker 命令")
        print("           → 装 Docker Desktop（Windows）后重开终端；aegra dev / up 都依赖它")
        print("           → 本节其余探测继续，不受影响")

    # ---- 4.2 aegra 命令 ----
    print("\n  [4.2] aegra 命令")
    aegra_path = shutil.which("aegra")
    if aegra_path:
        ok, ver = run_cmd(["aegra", "--version"], timeout=15)
        print(f"        已安装     ：{aegra_path}")
        print(f"        版本       ：{ver if ok else '（--version 不支持，可直接 aegra --help）'}")
    else:
        # 本机没装是正常情况：Aegra 要求独立的 3.12 环境，
        # 不能（也不该）装进 Python_Base 这个 venv —— 规范里明确禁止改依赖。
        print("        ❌ 未找到 aegra 命令（本机没装，属正常）")
        print("           → 按 02 节那三行在**独立环境**里装，不要装进 Python_Base 的 venv：")
        print("               conda create -n aegra python=3.12 -y")
        print("               conda activate aegra")
        print("               pip install aegra-cli aegra-api langgraph langchain-openai pydantic-settings")
        print("           → 本节其余内容（Dockerfile / 坑 / 流程）不受影响，纯讲解 + 探测")

    # ---- 4.3 2026 端口 ----
    print("\n  [4.3] 2026 端口（Aegra 默认监听）")
    # 用 socket.connect_ex 探测：比 ping / curl 快，也不依赖服务返回内容
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    in_use = sock.connect_ex(("127.0.0.1", 2026)) == 0
    sock.close()
    if in_use:
        print("        ✅ 127.0.0.1:2026 已被监听 —— 可能 Aegra 已经在跑了")
        print("           → 直接试 curl.exe http://localhost:2026/health")
    else:
        print("        ⚪ 127.0.0.1:2026 没有服务在监听")
        print("           → 说明本地还没起 Aegra；启动方式见本文件第 1 节（aegra dev）")

    # ---- 4.4 课案那个坑，在本机的真实情况 ----
    print("\n  [4.4] 课案那个坑在本机的真实情况")
    # 02 节刚刚生成过 docker-compose.yml，所以这里**必然**能演示出这个坑的前置条件
    compose = PROJECT_DIR / "docker-compose.yml"
    if compose.exists():
        size = compose.stat().st_size
        print(f"        ⚠️ {compose}")
        print(f"           已存在（{size} 字节，02 节生成的）")
        print("           → 现在直接跑 aegra dev，它会**沿用这份**而不是重新生成。")
        print("           → 这正是课案提醒的场景：确认这份 compose 就是你想要的那份；")
        print("             如果它是别的项目留下的，先 del 或 ren 掉再跑 aegra dev。")
    else:
        print(f"        ⚪ {PROJECT_DIR} 下没有 docker-compose.yml")
        print("           → 首次运行 aegra dev 会自动生成一份，不会触发这个坑")


# ================================================================
# 主流程
# ================================================================
if __name__ == "__main__":
    print("Agent 课案 · 部署 ③：本地开发与生产部署（aegra dev / aegra up / Dockerfile）")
    section_1_local_dev()
    section_2_pitfall()
    section_3_production()
    section_4_probe()
    print("\n小结：本地 aegra dev 一条命令搞定（注意旧 compose 文件的坑）；")
    print("      生产 aegra up 走 Dockerfile + compose 全容器化（aegra serve 只在 Linux 上跑）。")

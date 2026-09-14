# -*- coding: utf-8 -*-
"""
MCP ⑫ 调试工具 · MCP Inspector
================================================================
课案原文：MCP Inspector 是 Anthropic 官方出品的 MCP **可视化调试工具**，
用于测试和调试 MCP 服务端。

为什么需要它？MCP 服务端启动后是「哑」的 —— 它只在等人发 JSON-RPC，
终端里看不到任何工具列表。想确认「工具有没有注册上、参数 Schema 长什么样、
调用会不会报错」，写客户端代码太慢，Inspector 点几下就行。

它和前面几个文件的分工：

    # | 手段 | 适合场景 |
    # |---|---|---|
    # | 02_客户端_jxsd.py | 写代码验证，可重复、可进 CI |
    # | MCP Inspector（本文件） | 手工探索：看看有哪些工具/资源/提示词，随手填参数试一次 |
    # | 打印日志 | 定位服务端内部逻辑问题 |

课案四个小节：安装与启动 → 连接 MCP 服务端 → 功能使用 → 调用示例。

Inspector 支持三种传输，和课案「四种协议」那张表是同一套概念，只是换了个入口：

    # | 传输方式 | Inspector 里怎么填 | 什么时候选它 |
    # |---|---|---|
    # | Streamable HTTP（推荐） | URL 填 http://<主机>:<端口>/mcp | 服务端已经在跑，或要连远程/容器里的服务端 |
    # | SSE | URL 填 http://<主机>:<端口>/sse | 只有老服务端还在用 sse 时才选，新项目不要用 |
    # | STDIO | Command / Args 分两格填 | 本地 .py 脚本，不想占端口，让 Inspector 自己拉子进程 |

课案出处：Agent 课案 → MCP协议 → 调试工具
         → 安装与启动 / 连接 MCP 服务端 / 功能使用 / 调用示例

运行方式（纯教学文件，只打印操作步骤 + 检查本机 npx，不会真的启动 Inspector）：
    uv run Agent/05_mcp/12_调试工具_jxsd.py

本机实测结论（改脚本时请照抄这个格式，别只说「应该可以」）：
    ① npx / node 都能用 `shutil.which` 找到（本机 npx 来自 npm 全局前缀）。
    ② 本文件**不会**真的启动 Inspector：`npx @modelcontextprotocol/inspector`
       是前台常驻进程，跑在脚本里会把演示卡死，所以只打印命令、由学员自己开终端执行。
    ③ Inspector 默认监听 6274（前端 UI）+ 6277（代理），本机 6274 空闲时可直接用。
"""

import shutil
import socket
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

# 和 01_服务端_jxsd.py 约定的地址 —— Inspector 要连的就是这个
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 8000
MCP_URL = f"http://{HTTP_HOST}:{HTTP_PORT}/mcp"


# ---------- 1. 环境检查：npx / node 到底有没有、能不能用 ----------
def check_npx() -> bool:
    """检查本机有没有 npx（MCP Inspector 是通过 npm 生态分发的）。

    为什么要用 shutil.which 而不是直接 subprocess.run(["npx", ...])？
    Windows 上 npx 实际是 npx.CMD / npx.ps1，PATH 解析规则和 Linux 不同，
    shutil.which 会按 PATHEXT 正确找到可执行文件；直接跑 "npx" 反而可能 FileNotFoundError。
    """
    print("=" * 64)
    print("① 环境检查")
    print("=" * 64)

    # 两个都要查：npx 负责下载并拉起 Inspector，node 是它真正的运行时
    npx_path = shutil.which("npx")
    node_path = shutil.which("node")

    # 缺 npx 是最常见的失败原因（没装 Node.js），所以给出可操作的三条指引后直接返回 False，
    # 让后面的步骤整体跳过 —— 而不是让脚本在半路抛 FileNotFoundError
    # 缺 npx 就直接返回 False，让后面步骤整体跳过；
    # 不能半路抛 FileNotFoundError —— 那样学员看到的是一堆栈，而不是「去装 Node.js」。
    if not npx_path:
        print("❌ 未找到 npx，Inspector 无法安装/启动。")
        print("   请先安装 Node.js（自带 npm / npx）：https://nodejs.org/")
        print("   或使用 nvm-windows / fnm 管理 Node 版本。")
        print("   安装后重开终端，再运行本文件确认。")
        return False

    print(f"✅ npx 已找到：{npx_path}")
    if node_path:
        print(f"✅ node 已找到：{node_path}")

    # 真跑一次 --version，确认它不只是「存在」而是「能用」
    # （which 只证明文件在，不证明它可执行：权限、损坏的 shim、PATH 里的同名目录都能骗过 which）
    try:
        result = subprocess.run(
            [npx_path, "--version"],
            capture_output=True,
            text=True,
            timeout=60,             # npx 首次可能触网，给足 60 秒
            encoding="utf-8",
            errors="replace",       # npm 输出偶尔含非 UTF-8 字节，替换掉比抛异常好
        )
        if result.returncode == 0:
            print(f"✅ npx 版本：{result.stdout.strip()}")
        else:
            # 版本查不出来也不影响 Inspector 使用，所以只警告、不返回 False
            # 版本号查不出来只警告、不算失败：npm 源慢或离线都不影响 Inspector 的可用性。
            print(f"⚠️  npx --version 返回码 {result.returncode}：{(result.stderr or '').strip()[:200]}")
    except subprocess.TimeoutExpired:
        print("⚠️  npx --version 超时（网络或 npm 源慢），但不影响后续使用。")
    except Exception as exc:
        print(f"⚠️  执行 npx --version 失败：{type(exc).__name__}: {exc}")

    return True


# ---------- 2. 服务端在不在：Inspector「连不上」的头号原因 ----------
def check_server_running() -> bool:
    """看一眼 8000 端口有没有 MCP 服务在跑 —— Inspector 连不上多半是这个原因。"""
    sock = socket.socket()
    sock.settimeout(0.5)            # 探测用短超时：宁可在 0.5 秒内判定「没人监听」，也别卡住
    try:
        sock.connect((HTTP_HOST, HTTP_PORT))
        print(f"✅ 检测到 {MCP_URL} 已有服务在运行，可以直接用 Inspector 连它。")
        return True
    except OSError:
        # 连不上不是错误，只是「还没起服务」。这里不抛异常，而是把补救命令打出来
        print(f"ℹ️  {HTTP_HOST}:{HTTP_PORT} 当前没有服务。")
        print("   要用 Inspector 调试，请先另开一个窗口启动服务端：")
        print("       uv run Agent/05_mcp/01_服务端_jxsd.py http")
        return False
    finally:
        sock.close()                # 探测完立刻关闭，别把探测 socket 留成泄漏


# ---------- 3. 把课案四小节的原文整理成终端里的操作手册 ----------
# 说明：本文件是**纯打印型**教学文件 —— 下面这些 print 的内容就是课案原文，
# 所以整段没有「业务逻辑」，读者只要逐段对照课案的四个小节即可。
def print_guide() -> None:
    # —— 课案「安装与启动」 ——
    print()
    print("=" * 64)
    print("② 安装与启动（课案原文）")
    print("=" * 64)
    print("    npx @modelcontextprotocol/inspector")
    print()
    print("  首次运行会自动下载依赖；启动后终端会显示访问地址")
    print("  （默认 http://localhost:6274），用浏览器打开即可。")
    print()
    print("  常用变体：")
    print("    npx @modelcontextprotocol/inspector --version        # 看版本")
    print("    npx -y @modelcontextprotocol/inspector               # 跳过确认直接下载")
    print("    npx @modelcontextprotocol/inspector --help           # 看全部参数")
    print()
    # 先讲清「Inspector 是前台常驻进程」，再给命令 —— 否则学员会以为脚本卡住了。
    print("  ⚠️ 它是**前台常驻**进程：关掉终端 Inspector 就停了；")
    print("     调试期间请单开一个终端窗口跑它，别和别的命令挤在一起。")

    # —— 课案「连接 MCP 服务端」（三种传输） ——
    print()
    print("=" * 64)
    print("③ 连接 MCP 服务端（课案原文：支持三种传输方式，Streamable HTTP 推荐）")
    print("=" * 64)
    print("  在 Inspector 的连接页面里选传输方式，然后填地址：")
    print()
    # —— 传输方式对照表：和课案「四种协议」是同一套概念，只是换了个入口 ——
    print("    # | 传输方式 | Inspector 里怎么填 | 说明 |")
    print("    # |---|---|---|")
    print("    # | Streamable HTTP（推荐） | URL 填下面的地址 | 新项目一律选它 |")
    print("    # | SSE | URL 填 http://<主机>:<端口>/sse | 已弃用 |")
    print("    # | STDIO | Command / Args 分两格填 | 本地脚本，不占端口 |")
    print()
    print(f"    Streamable HTTP 的 URL： {MCP_URL}")
    print()
    # STDIO 方式：把「用哪个解释器 + 跑哪个脚本」告诉 Inspector，
    # 它就会像 02_客户端_jxsd.py 的 StdioTransport 一样把脚本拉成子进程
    print("    STDIO 方式（不用先起服务，Inspector 自己拉子进程）：")
    print(f"      Command : {sys.executable}")
    print(f"      Args    : {__file__.replace('12_调试工具_jxsd.py', '01_服务端_jxsd.py')} stdio")
    print("      （Args 里那个 stdio 参数不能省，否则会走自检分支去抢 8000 端口）")
    print()
    # 排查清单：按「最可能的原因排在前面」排序，学员从上往下试即可
    # —— 连不上时的排查清单，按「最可能的原因」排序 ——
    print("    连不上时按顺序排查：")
    print("      1. 服务端起了吗？（① 里的检测结果）")
    print("      2. URL 结尾的 /mcp 有没有漏？（漏了就 404）")
    print("      3. host 写的是 127.0.0.1 还是 0.0.0.0？跨机器访问要用实际 IP")
    print("      4. 带认证的服务要先拿令牌（见 08/09/10），Inspector 里在")
    print("         「Authentication」区选 Bearer Token 并粘贴 JWT")

    # —— 课案「功能使用」原表 ——
    # —— 配套本套课案怎么练：把 Inspector 的四个功能对到四个文件上 ——
    # STDIO 方式的关键是把解释器和脚本路径分开填，Inspector 才能拉子进程。
    print()
    print("=" * 64)
    print("④ 功能使用（课案原表）")
    print("=" * 64)
    print("    # | 功能 | 说明 |")
    print("    # |---|---|")
    print("    # | Tools 列表 | 左侧面板展示服务端所有工具的名称和描述 |")
    print("    # | 工具调用 | 点击工具，填写参数表单，点击执行即可测试 |")
    print("    # | Resources 浏览 | 浏览服务端暴露的资源（文件、数据等） |")
    print("    # | Prompts 浏览 | 查看服务端提供的提示词模板 |")
    print()
    # 这段是「配套本套课案文件怎么练」——把 Inspector 的四个功能
    # 分别对上 01 / 03 / 04 / 09 四个文件，学员照着点一遍就全会了
    print("  对着本套课案文件可以这样练：")
    print("    · 连 01_服务端_jxsd.py  → Tools 里能看到 add / sub / mul / div，")
    print("      点 div 填 a=1,b=0 执行，能直接在界面上看到红字报错「除数不能为 0」")
    print("    · 连 03_资源_jxsd.py    → Resources 里固定资源 2 个、资源模板 4 个，")
    print("      点 users://top/3 就能读到数据库查出来的 TOP3")
    print("    · 连 04_提示词_jxsd.py  → Prompts 里填参数，右侧直接渲染出拼好的提示词")
    print("    · 连 09_权限_服务端_jxsd.py → 不填 Token 会连不上，填上 08 签发的 JWT 才通")
    print()
    # 收尾提醒：Inspector 解决「手工探索」，回归验证仍要写成代码（02 那种）。
    print("  调试完记得回来：Inspector 只解决「手工探索」，")
    print("  真正要固化下来的验证还是得写成 02_客户端_jxsd.py 那样的代码。")


# ---------- 4. 主流程：查环境 → 查服务端 → 打印手册 → 给一键复现命令 ----------
if __name__ == "__main__":
    # 两个检查的返回值决定后面打印哪条建议，所以必须先拿到结果再打印手册
    has_npx = check_npx()
    print()
    server_up = check_server_running()
    print_guide()

    print()
    print("=" * 64)
    print("⑤ 一键复现命令")
    print("=" * 64)
    print("    终端 1（服务端）：")
    print("        uv run Agent/05_mcp/01_服务端_jxsd.py http")
    print("    终端 2（Inspector）：")
    print("        npx @modelcontextprotocol/inspector")
    print("    浏览器： http://localhost:6274  →  传输选 Streamable HTTP")
    print(f"            URL 填 {MCP_URL}")
    print()

    # 下面两个分支都用 sys.exit(0)：缺环境是「学员的机器还没配好」，
    # 不是脚本出错 —— 用 0 退出，避免 CI 或批处理里被当成失败
    if not has_npx:
        print("⚠️  本机缺 npx，上面这两步请先装 Node.js 再执行。")
        sys.exit(0)

    if not server_up:
        print("ℹ️  服务端当前没起，照上面「终端 1」的命令起来之后再连。")
        sys.exit(0)

    print("✅ 环境就绪：npx 可用 + 服务端在跑，直接执行「终端 2」那条命令即可开始调试。")

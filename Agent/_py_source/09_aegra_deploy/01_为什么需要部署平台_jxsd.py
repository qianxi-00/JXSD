# -*- coding: utf-8 -*-
"""
部署 ①：为什么需要「部署平台」—— License 校验的坑与开源替代 Aegra
================================================================
本节是「部署」这一章的开场：先讲清楚一件很容易踩坑的事 ——
    langgraph build 构建出来的官方 `langgraph-api` 镜像，**不是** MIT 开源的
    langgraph 库，而是 LangGraph Platform（现已改名 LangSmith Deployments），
    容器启动时会强制校验 license，验证不过**直接退出**。

所以「我本地 graph 跑得好好的，一打成镜像就起不来」几乎必然是 license 问题，
而不是代码问题。课案里那段 `ValueError: License verification failed...` 报错原文
在下面第 1 节里**逐字保留**，见到它别再怀疑自己的图。

关键认知（后面所有小节都建立在这条上）：

    LangGraph **库**是 MIT 开源的 —— 图引擎、checkpoint、streaming 全免费；
    收费的只是官方这一层「部署平台」（托管 + REST API + 多租户 + 控制台）。

也就是说，「部署运行时」这件事是可以自己搭 / 用开源方案替代的，
本节最后介绍的 **Aegra**（Apache 2.0）就是干这个的。

本节要讲什么：

    1. 【踩坑现场】langgraph build 出来的官方 langgraph-api 镜像启动即退出，
       报 `ValueError: License verification failed...` —— 原文逐字保留，
       见到它别再怀疑自己的图写错了；
    2. 【澄清误解】到底哪部分收费：LangGraph 库（MIT 开源，图引擎本体）
       vs LangGraph Platform / LangSmith Deployments（商业的部署运行时）；
    3. 【选型地图 · 表一】LangSmith 其实一个产品干了两件事 ——
       观测/追踪 与 部署运行时，而这两件事的可替代性完全不同：
       追踪可以用 Langfuse 顶掉，运行时 Langfuse 根本不做；
    4. 【能力清单 · 表二】部署运行时四要素（Threads API / Runs / 流式 /
       Checkpoint 管理）在「用库」和「用平台」两种形态下分别由谁提供；
    5. 【出路】开源替代 Aegra（Apache 2.0）：同一套 Agent Protocol、
       客户端 SDK 同款、没有 license 校验。

本节还会把课案里的两张对照表用「打印 + 注释」两种形式完整呈现：
    表一：LangSmith 的两个角色，分别能不能被 Langfuse 替代
    表二：部署运行时四要素（Threads API / Runs / 流式 / Checkpoint 管理）
          在「LangGraph 库」和「LangSmith」两种形态下分别怎么做

本节不依赖任何外部服务，纯打印，直接能跑。

课案出处：Agent 课案 → 部署 →（开头：langgraph build / LangSmith Deployments / Aegra）

运行前置条件：无（纯标准库 + 控制台打印，离线可跑，不连任何外部服务）。

运行方式：
    uv run Agent/09_aegra_deploy/01_为什么需要部署平台_jxsd.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import unicodedata


# ================================================================
# 小工具：按「显示宽度」对齐打印表格
# ================================================================
# 中文/全角字符在终端里占 2 个字符位，直接用 len() 算宽度会让表格歪掉，
# 所以先用 unicodedata.east_asian_width 判断再补齐空格。
def _disp_width(text: str) -> int:
    """按东亚字符宽度计算字符串在终端里占的列数"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    """把 text 右侧补空格到指定显示宽度"""
    return text + " " * max(0, width - _disp_width(text))


def print_table(title: str, headers: list, rows: list) -> None:
    """打印一张带标题的等宽表格（单元格过长时不做折行，保持原貌）"""
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
# 1. 课案原文：license 校验失败的报错
# ================================================================
# `langgraph build` 构建的官方 `langgraph-api` 镜像属于 LangGraph Platform
# （现改名 LangSmith Deployments），容器启动时强制校验 license，验证不过直接退出。
# 下面这段报错**逐字保留课案原文**，遇到它说明你打的是官方镜像、缺 license：
#
#     ValueError: License verification failed. Please ensure proper configuration:
#     - For local development, set a valid LANGSMITH_API_KEY for an account with LangGraph Cloud access.
#     - For production, configure the LANGGRAPH_CLOUD_LICENSE_KEY environment variable.
#
LICENSE_ERROR_TEXT = """ValueError: License verification failed. Please ensure proper configuration:
- For local development, set a valid LANGSMITH_API_KEY for an account with LangGraph Cloud access.
- For production, configure the LANGGRAPH_CLOUD_LICENSE_KEY environment variable."""


def section_1_license_error() -> None:
    """复现课案里那段 license 报错长什么样，并解释两条出路"""
    print("=" * 78)
    print("1. 课案原文：官方 langgraph-api 镜像启动时的 license 校验失败")
    print("=" * 78)
    print("容器启动时校验不过，进程直接退出，日志里就是这一段：\n")
    for ln in LICENSE_ERROR_TEXT.splitlines():
        print("    " + ln)

    # 报文里给了两条「官方」出路 —— 注意两条都指向付费/注册：
    #   - 本地开发：注册 LangSmith 账号并配置 LANGSMITH_API_KEY
    #     （且该账号要有 LangGraph Cloud 访问权限，不是随便注册一个就行）
    #   - 生产环境：需要企业版付费 license，配 LANGGRAPH_CLOUD_LICENSE_KEY
    print("\n报文里给出的两条出路（都要求注册 / 付费，不是配置错误那么简单）：")
    print("    · 本地开发：注册 LangSmith 账号并配置 LANGSMITH_API_KEY")
    print("      （账号还必须具备 LangGraph Cloud 访问权限）")
    print("    · 生产环境：需要企业版付费 license，配置 LANGGRAPH_CLOUD_LICENSE_KEY")
    print("\n>>> 结论：这不是你代码写错了，是镜像本身要求 license。")


# ================================================================
# 2. 澄清一个高频误解：到底哪部分是收费的？
# ================================================================
# 课案原文：
#     `langgraph` 库本身是 MIT 开源的，收费的只是官方这层「部署平台」。
#     开源替代方案：**Aegra**。
#
# 把「库」和「平台」拆开看，收费边界就非常清楚了：
#
# | 东西 | 是否开源 | 说明 |
# |---|---|---|
# | langgraph（pip 包） | MIT 开源，免费 | 图引擎本体：StateGraph / 节点 / 边 / 中断 |
# | langgraph-checkpoint-* | MIT 开源，免费 | 检查点存储（内存 / SQLite / PostgreSQL） |
# | langgraph-sdk | MIT 开源，免费 | 客户端 SDK（连官方平台和连 Aegra 用的是同一个） |
# | langgraph-api 镜像 / LangSmith Deployments | 商业产品 | 托管运行时 + REST API + 控制台 + 多租户 |
#
# 一句话：**你写的 Agent 代码是你的，跑 Agent 的那层「平台」才是商品。**
# 既然那层平台的对外契约只是「一套 Agent Protocol 的 HTTP API」，
# 就完全可以自己用 FastAPI + PostgreSQL 复刻 —— 这就是 Aegra 做的事。
def section_2_what_is_paid() -> None:
    print("\n" + "=" * 78)
    print("2. 到底哪部分收费？—— 库 vs 平台")
    print("=" * 78)
    print_table(
        "开源 vs 商业：逐项拆开看",
        ["组成部分", "许可", "说明"],
        [
            ["langgraph（pip 包）", "MIT 开源", "图引擎本体：StateGraph / 节点 / 边 / 中断"],
            ["langgraph-checkpoint-*", "MIT 开源", "检查点存储：内存 / SQLite / PostgreSQL"],
            ["langgraph-sdk", "MIT 开源", "客户端 SDK（连官方平台与连 Aegra 是同一个）"],
            ["langgraph-api 镜像", "商业产品", "LangGraph Platform / LangSmith Deployments"],
            ["LangSmith Deployments", "商业产品", "托管运行时 + REST API + 控制台 + 多租户"],
        ],
    )
    print("\n>>> 你写的 Agent 代码是你的；跑 Agent 的那层「平台」才是商品。")


# ================================================================
# 3. 表一：LangSmith 的两个角色，Langfuse 能替代吗？
# ================================================================
# 课案原表（注释形式保留）：
# | LangSmith 的角色 | Langfuse 能替代吗 | 开源替代 |
# |---|---|---|
# | 观测/追踪（trace、调试、评估） | ✅ 能，Langfuse 的主业（MIT 开源） | Langfuse |
# | 部署运行时（threads API、runs、流式、checkpoint 管理） | ❌ 不能，Langfuse 完全不做这块 | Aegra、或自己用 FastAPI 包一层 |
#
# 这张表是整章的「选型地图」：LangSmith 其实一个产品干了两件事，
# 而这两件事的可替代性完全不同 ——
#   · 追踪/观测：Langfuse 是同类竞品，而且 MIT 开源、可自托管，换过去很自然；
#   · 部署运行时：Langfuse **根本不做**（它是观测平台，不是运行时平台），
#     所以「我上了 Langfuse 是不是就不用 LangSmith 了」这个念头是错的，
#     这块要靠 Aegra（或自己拿 FastAPI 包一层）来补。
def section_3_table_smith_roles() -> None:
    print("\n" + "=" * 78)
    print("3. 表一：LangSmith 的角色与替代方案")
    print("=" * 78)
    print_table(
        "LangSmith 的角色与替代",
        ["LangSmith 的角色", "Langfuse 能替代吗", "开源替代"],
        [
            ["观测/追踪（trace、调试、评估）", "✅ 能，Langfuse 的主业（MIT 开源）", "Langfuse"],
            ["部署运行时（threads API、runs、流式、checkpoint 管理）",
             "❌ 不能，Langfuse 完全不做这块", "Aegra、或自己用 FastAPI 包一层"],
        ],
    )
    print("\n>>> 两个角色要分开选型：追踪用 Langfuse，运行时用 Aegra，两者互不冲突。")


# ================================================================
# 4. 表二：部署运行时四要素
# ================================================================
# 课案原表（注释形式保留）：
# | 概念 | 说明 | LangGraph vs LangSmith |
# |---|---|---|
# | Threads API | 对话会话管理。每个用户对话是一个 thread，同一 thread 内 Agent 记住上下文 |
# |             | LangGraph 库：用 thread_id 区分。LangSmith：提供 REST API 创建/查询/删除线程 |
# | Runs | 执行记录。每次 graph.invoke() 产生一个 run，记录输入/输出/状态 |
# |      | LangGraph 库：同步执行。LangSmith：异步执行 + 持久化每次 run 的结果 |
# | 流式 | Agent 执行时实时推送中间步骤（工具调用、思考过程）给前端 |
# |      | LangGraph 库：stream_mode="updates"。LangSmith：把流式结果转为 HTTP 流式响应（text/event-stream）供前端消费 |
# | Checkpoint 管理 | 状态快照。每步执行后自动保存 state，支持中断恢复和时间旅行 |
# |                 | LangGraph 库：自己搭数据库。LangSmith：替你做三件事 —— ① 云端托管 ② 持久化（重启不丢）③ 多租户（用户数据隔离） |
#
# 读懂这张表的关键：左边「概念」是**业务上必须有**的能力，
# 右边两列是「这两种形态下，同样的能力分别由谁提供」——
#   · 用 LangGraph 库：能力都在，但**零件要你自己拼**（线程 id 自己管、
#     数据库自己搭、流式响应自己包成 SSE）；
#   · 用 LangSmith/Aegra：这些零件被封装成**一套 REST API**，
#     前端只认 HTTP，Agent 变成一个有 URL 的后端服务。
def section_4_table_runtime_elements() -> None:
    print("\n" + "=" * 78)
    print("4. 表二：部署运行时四要素，两种形态下分别怎么做")
    print("=" * 78)

    rows = [
        ["Threads API", "对话会话管理。每个用户对话是一个 thread，同一 thread 内 Agent 记住上下文",
         "LangGraph 库：用 thread_id 区分\nLangSmith：提供 REST API 创建/查询/删除线程"],
        ["Runs", "执行记录。每次 graph.invoke() 产生一个 run，记录输入/输出/状态",
         "LangGraph 库：同步执行\nLangSmith：异步执行 + 持久化每次 run 的结果"],
        ["流式", "Agent 执行时实时推送中间步骤（工具调用、思考过程）给前端",
         "LangGraph 库：stream_mode=\"updates\"\nLangSmith：转为 HTTP 流式响应（text/event-stream）"],
        ["Checkpoint 管理", "状态快照。每步执行后自动保存 state，支持中断恢复和时间旅行",
         "LangGraph 库：自己搭数据库\nLangSmith：① 云端托管 ② 持久化 ③ 多租户"],
    ]

    # 四要素逐个展开讲，避免上面那张宽表在窄终端里被折行看不清
    for name, desc, cmp_ in rows:
        print(f"\n  ◆ {name}")
        print(f"      是什么：{desc}")
        for ln in cmp_.splitlines():
            print(f"      怎么做：{ln}")

    print("\n>>> 左边这些能力业务上一个都不能少；区别只是「自己拼」还是「平台给」。")


# ================================================================
# 5. 一句话总结
# ================================================================
# 课案原文：
#     一句话：**LangGraph 负责"怎么跑"（图引擎），
#     LangSmith 负责"跑在哪 + 谁在跑 + 跑的结果存哪"（运行时平台）。**
#     Aegra 把这个运行时平台用开源替代了。
#
# 这三个问句正好对应三件事：
#     "怎么跑"  → 图引擎（langgraph 库，免费）
#     "跑在哪"  → 部署形态（容器 / 服务器 / 托管）
#     "谁在跑"  → 多租户与鉴权（谁的 thread、谁的 run）
#     "结果存哪" → 持久化（PostgreSQL checkpoint + runs 记录）
ONE_SENTENCE = (
    "LangGraph 负责「怎么跑」（图引擎），LangSmith 负责「跑在哪 + 谁在跑 + 跑的结果存哪」"
    "（运行时平台）。Aegra 把这个运行时平台用开源替代了。"
)


def section_5_summary() -> None:
    print("\n" + "=" * 78)
    print("5. 一句话总结")
    print("=" * 78)
    print("  " + ONE_SENTENCE)


# ================================================================
# 6~7. 开源替代：Aegra
# ================================================================
# 课案原文：
#     地址：https://github.com/aegra/aegra （Apache 2.0）
#     Aegra 是 LangGraph Platform 的开源自托管替代：用 FastAPI + PostgreSQL
#     实现了同一套 Agent Protocol，`langgraph_sdk` 客户端代码原样可用，
#     没有任何 license 校验。
#
# 这段是整章的转折点，三个细节值得单独拎出来：
#   1. **同一套 Agent Protocol** —— 意味着协议层是兼容的，不是「类似」而是「同一套」。
#   2. **langgraph_sdk 客户端代码原样可用** —— 这是最实用的一条：
#      以后从 Aegra 迁回官方平台（或反过来），客户端那 30 行**一行都不用改**，
#      只换 url。这也是下一节 04 客户端调用能直接照抄课案的原因。
#   3. **没有任何 license 校验** —— 对应本节开头那个 ValueError，Aegra 里不存在。
AEGRA_REPO = "https://github.com/aegra/aegra"
AEGRA_LICENSE = "Apache 2.0"


def section_6_aegra() -> None:
    print("\n" + "=" * 78)
    print("6. 开源替代：Aegra")
    print("=" * 78)
    print(f"  仓库地址：{AEGRA_REPO}  （{AEGRA_LICENSE}）")
    print("  它是什么：LangGraph Platform 的开源自托管替代。")
    print("            用 FastAPI + PostgreSQL 实现同一套 Agent Protocol，")
    print("            langgraph_sdk 客户端代码原样可用，没有任何 license 校验。")
    print("\n  三个关键细节：")
    print("    ① 「同一套」Agent Protocol —— 协议层兼容，不是「类似」。")
    print("    ② langgraph_sdk 客户端代码原样可用 —— 迁回官方平台只换 url，客户端不用改。")
    print("    ③ 没有 license 校验 —— 本节开头那个 ValueError 在 Aegra 里不存在。")

    # 课案原表（注释形式保留）：
    # | 对比 | LangSmith Deployments | Aegra |
    # |---|---|---|
    # | 自托管 | 仅企业版（license key） | 免费（Apache 2.0） |
    # | 数据库 | 官方托管 | 自己的 PostgreSQL |
    # | 追踪 | 仅 LangSmith | Langfuse |
    # | 客户端 SDK | LangGraph SDK | 同款 LangGraph SDK |
    print_table(
        "Aegra vs LangSmith Deployments（课案四行对比表）",
        ["对比", "LangSmith Deployments", "Aegra"],
        [
            ["自托管", "仅企业版（license key）", "免费（Apache 2.0）"],
            ["数据库", "官方托管", "自己的 PostgreSQL"],
            ["追踪", "仅 LangSmith", "Langfuse"],
            ["客户端 SDK", "LangGraph SDK", "同款 LangGraph SDK"],
        ],
    )
    # 注意最后一行：两边的客户端 SDK 是**同款**。
    # 表格里唯一「完全一样」的一格，恰恰是学员最关心的那一格 —— 迁移成本。
    print("\n>>> 注意最后一行：两边客户端 SDK 是同款，所以迁移成本几乎为零。")
    print(">>> 下一节（02_项目骨架）就开始动手搭一个 Aegra 项目。")


# ================================================================
# 主流程
# ================================================================
if __name__ == "__main__":
    print("Agent 课案 · 部署 ①：为什么需要部署平台（LangSmith Deployments 的 license 坑 → Aegra）")
    section_1_license_error()
    section_2_what_is_paid()
    section_3_table_smith_roles()
    section_4_table_runtime_elements()
    section_5_summary()
    section_6_aegra()
    print("\n（本节纯讲解，无需任何外部服务；下一节开始生成真实项目骨架）")

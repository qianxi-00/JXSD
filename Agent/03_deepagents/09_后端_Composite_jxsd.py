# -*- coding: utf-8 -*-
"""
DeepAgents 运行环境⑦：CompositeBackend —— 按路径前缀路由到不同后端
================================================================
前面六个后端各管一摊，但真实项目里一个 Agent 往往**同时**需要：

    - 临时草稿
    - 长期记忆
    - 真实项目文件
    - 安全代码执行

CompositeBackend 就是用来把它们拼起来的：它自己**通常不负责真正存储**，
而是按路径前缀，把操作转发给对应的后端。可以理解成后端层的「路由器」。

本节要讲什么
    1. 它**解决什么问题**：前面六个后端不是「七选一」，真实项目是**要同时用**；
       CompositeBackend 让一个 Agent 按路径前缀同时拥有草稿区、长期记忆、
       真实项目文件（以及可选的沙箱执行）；
    2. 课案那句「大多数场景用 CompositeBackend」的依据是什么
       —— 框架自己会往 `/large_tool_results/`、`/conversation_history/` 写内部文件，
       default 必须是内存后端，否则项目目录会被框架的内部产物搞脏；
    3. 路由语义：**先按前缀匹配、匹配上就把前缀剥掉再转给目标后端**
       —— 这条规则决定了「Store key 到底该写什么」（11_记忆_jxsd.py 踩的坑就是这个）；
    4. 边界情况：前缀必须**结尾带斜杠**才按目录匹配；没匹配上的走 `default`。

一、课案的组合示例（本文件实现的就是它）
    CompositeBackend(
        default=StateBackend(),                  # 兜底：内部临时文件不进磁盘
        routes={
            "/workspace/": FilesystemBackend(root_dir=".", virtual_mode=True),
            "/memories/":  StoreBackend(namespace=lambda rt: ("memories", "1")),
        },
    )

    路由规则（课案注释原文）：

        /workspace/readme.md → FilesystemBackend（真实磁盘）
        /memories/user.md    → StoreBackend（跨线程持久化）
        其他所有路径         → StateBackend（内存）

二、为什么 `default` 最好用 StateBackend（课案「建议」段，这段很实用）
    课案原文：

    > **建议**：大多数场景用 `CompositeBackend`。因为 DeepAgents 内部会自动向后端
    > 写入工具结果（`/large_tool_results/`）和对话历史（`/conversation_history/`），
    > 所以 `default` 最好用 `StateBackend`（内存、自动清退），让这些内部文件不落盘；
    > `/workspace/` 路由到 `FilesystemBackend` 操作真实项目文件；
    > `/memories/` 路由到 `StoreBackend` 做长期记忆。

    这条建议不是拍脑袋：在 deepagents 的 `FilesystemMiddleware` 源码里确实能找到
    `/large_tool_results/`（超大工具结果被 offload 到文件）和
    `conversation_history` 这两个前缀。它们属于框架的内部产物，
    量可能很大、也没有长期价值，落到磁盘上只会把项目目录搞脏。

三、各后端数据到底存在哪（课案总表，收尾时对照着看）
    | 后端 | 物理存储 | 你能用资源管理器看到吗 | 生命周期 |
    |---|---|---|---|
    | StateBackend | LangGraph state（Python dict） | ❌ | 同 thread，进程死即丢 |
    | FilesystemBackend | 你的真实磁盘 | ✅ | 永久 |
    | LocalShellBackend | 你的真实磁盘 | ✅ | 永久 |
    | StoreBackend | LangGraph Store 数据库（本地 InMemoryStore / 生产 PostgreSQL） | ❌ | 跨 thread，取决于 Store |
    | ContextHubBackend | LangSmith Hub 云端仓库（Git 版本管理，每次写 = 一次 commit） | ❌（LangSmith 网页可看） | 永久 + 版本历史 |
    | Sandbox | 沙箱容器内临时文件系统 | ❌ | 沙箱销毁即丢 |
    | CompositeBackend | 取决于路由规则（以上任意组合） | — | — |

四、七种后端横向定位表（课案「运行环境」开篇那张表，收口用）
    | 后端 | 说明 | 适用场景 |
    |---|---|---|
    | StateBackend（默认） | 存入 LangGraph state，同 thread 跨轮持久化，不跨 thread | Agent 草稿纸、中间结果暂存 |
    | StoreBackend | 存入 LangGraph Store，跨 thread 持久化 | 长期记忆 |
    | FilesystemBackend | 对接真实磁盘，仅文件操作 | 本地项目、CI/CD |
    | LocalShellBackend | = FilesystemBackend + execute，可在宿主机执行任意 shell 命令 | 本地开发 CLI（仅限受控环境） |
    | Sandbox | = FilesystemBackend + execute，但代码在隔离容器内运行，不触碰宿主机 | 生产环境、多租户、不可信代码 |
    | ContextHubBackend | 存入 LangSmith Hub 仓库，持久化 + 版本历史 | LangSmith 原生方案，无需单独 Store |
    | CompositeBackend | 路由分发：按路径前缀把不同目录分发到不同后端 | 混合策略 |

    「怎么选」三句话（把上面 7 行压成决策）：
        ① 只改文件、不跑代码             → FilesystemBackend
        ② 要跑代码、且不在乎隔离         → LocalShellBackend（本地开发）
        ③ 要跑代码、且必须隔离           → Sandbox（付费云端 or 自建 Docker）
    其余的 State / Store / ContextHub 管「文件存哪」，与「能不能执行」正交，
    可以按持久化需求自由组合 —— 组合的方式就是本节的 CompositeBackend。

课案出处：Agent 课案 → deepAgents → 运行环境 → ⑦ CompositeBackend

运行方式：
    uv run Agent/03_deepagents/09_后端_Composite_jxsd.py

前置条件：.env 里的 api_key / base_url / model_name 可调通（实测可用）。
         用 InMemoryStore 演示 Store 路由，**不需要** PostgreSQL。

【本文件相对课案的一处改动】
    课案 `FilesystemBackend(root_dir=".", virtual_mode=True)` 指的是运行时当前目录。
    为不污染仓库，这里改成脚本同级的 `tmp_jxsd_deepagents_composite/`，其余不变。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend, StoreBackend
from langchain.chat_models import init_chat_model
from langgraph.store.memory import InMemoryStore
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

# /workspace/ 这条路由最终落到这个真实目录
WORKSPACE_DIR = Path(__file__).resolve().parent / "tmp_jxsd_deepagents_composite"

# 本地演示用内存 Store；生产换成 PostgresStore.from_conn_string(settings.pg_uri)
store = InMemoryStore()

agent = create_deep_agent(
    model=llm,
    backend=CompositeBackend(
        default=StateBackend(),          # 兜底：/large_tool_results/ 等内部文件留在内存
        routes={
            # 前缀要用**结尾带斜杠**的写法，才会按「目录」匹配
            "/workspace/": FilesystemBackend(root_dir=str(WORKSPACE_DIR), virtual_mode=True),
            "/memories/": StoreBackend(namespace=lambda rt: ("memories", "1")),
        },
    ),
    # 本地要自己把 store 交给图（同 04 文件里的解释：上平台后由平台自动注入）
    store=store,
    system_prompt=(
        "你是项目助手，路径规则如下："
        "真实项目文件写到 /workspace/ 下；需要长期记住的信息写到 /memories/ 下；"
        "其他临时草稿随便放。"
    ),
)

if __name__ == "__main__":
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- ① 课案任务：往 /workspace/ 写文件 → 应该落到真实磁盘 ----------
    print("===== ① 在 workspace 里创建一个 txt 文件 =====")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "在workspace里面创建一个txt文件"}]},
        config={"recursion_limit": 50},
    )
    print(result["messages"][-1].content)

    # ---------- ② 课案任务：往 /memories/ 写文件 → 应该进 Store ----------
    print("\n===== ② 在 memories 里记一条长期信息 =====")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "把「用户偏好：回答要简短」写进 /memories/prefs.md"}]},
        config={"recursion_limit": 50},
    )
    print(result["messages"][-1].content)

    # ---------- ③ 验证路由：三份数据分别去了哪 ----------
    print("\n===== ③ 路由结果验证 =====")

    print("[真实磁盘] /workspace/ →", WORKSPACE_DIR)
    disk_files = sorted(p.relative_to(WORKSPACE_DIR) for p in WORKSPACE_DIR.rglob("*") if p.is_file())
    if disk_files:
        for rel in disk_files:
            print(f"    ✅ {rel}")
    else:
        print("    （本轮没有文件落到磁盘）")

    print("[LangGraph Store] /memories/ → namespace ('memories', '1')")
    items = store.search(("memories", "1"))
    if items:
        for item in items:
            print(f"    ✅ {item.key}")
    else:
        print("    （本轮没有文件写进 Store）")

    # StateBackend 兜底：这些文件只存在于本次运行的 state 里，函数返回即丢
    state_files = sorted((result.get("files") or {}).keys())
    print("[StateBackend 兜底] 其他路径 → 内存")
    if state_files:
        for path in state_files:
            print(f"    ✅ {path}")
    else:
        print("    （本轮 state 里没有额外文件）")

    print(
        "\n结论：同一个 Agent、同一套文件工具，靠路径前缀把数据分流到了三个地方。\n"
        "      这就是课案推荐「大多数场景用 CompositeBackend」的原因。"
    )

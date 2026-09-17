# -*- coding: utf-8 -*-
"""
DeepAgents 运行环境⑤：ContextHubBackend —— 文件存进 LangSmith Hub 仓库
================================================================
ContextHubBackend 把文件存到 **LangSmith Hub 的 agent 仓库**里。
和前面几个后端的最大区别是：Hub 仓库本身是 Git 托管的 ——
**每一次写文件 = 一次 commit**，所以它自带版本历史。

本节要讲什么
    1. 它**解决什么问题**：既要持久化、又要版本历史，但又不想自己维护
       一个 Store/数据库 —— 直接借用 LangSmith Hub 的 agent 仓库；
    2. 什么时候选它：已经在用 LangSmith 的团队，想让 Agent 的产出
       像代码一样可回溯、可 diff、可回滚（课案表格：LangSmith 原生方案，无需单独 Store）；
       什么时候换人：不能联网 / 不想依赖外部账号 → StoreBackend（04）；
    3. 与本文件相邻后端的取舍：ContextHub 与 Store 都能「跨线程持久化」，
       差别就在「版本历史 + 托管」这两点上，本文件第三节给出了不依赖
       LangSmith 的等效做法。

一、物理落点对照（课案表格里的这一行）
    | 后端 | 物理存储 | 你能用资源管理器看到吗 | 生命周期 |
    |---|---|---|---|
    | ContextHubBackend | LangSmith Hub 云端仓库（Git 版本管理，每次写 = 一次 commit） | ❌（LangSmith 网页可看） | 永久 + 版本历史 |

    适用场景（课案表格）：LangSmith 原生方案，无需单独 Store。

二、构造参数
    源码签名：
        ContextHubBackend(identifier: str, *, client: Client | None = None)
    课案写法：
        ContextHubBackend("your-org/my-agent")     # owner/name 格式
    `identifier` 也可以是 `"-/name"`（`-` 表示当前登录用户的个人空间）。
    内部默认用 `langsmith.Client()`，它从环境变量里读凭据：

        LANGSMITH_API_KEY     必填，LangSmith 的 API Key
        LANGSMITH_ENDPOINT    可选，自托管 / 特定区域时用

    ⚠️ 注意：`settings`（config.py）里**没有** LangSmith 相关字段，所以这里按规范
    直接读环境变量：`os.environ.get("LANGSMITH_API_KEY", "")`。

三、本文件怎么跑（规范第 5.3 条的优雅降级）
    本机没有配置 LANGSMITH_API_KEY，所以：

        - import 阶段零报错（ContextHubBackend 本身是本地类，构造时才连 LangSmith）
        - `__main__` 里先检查密钥，**为空就打印中文指引并 sys.exit(0)**，
          绝不把 langsmith 的凭据异常 traceback 抛给用户
        - 密钥存在时才走真实的建 Agent + 调用流程（代码是完整的，不是占位符）

    想真的跑起来，需要三步：
        1. 注册 LangSmith 账号，拿到 API Key
        2. 在 PowerShell 里设置环境变量（只在当前会话生效）：
               $env:LANGSMITH_API_KEY = "lsv2_pt_xxxxxxxx"
           或者永久写入用户环境变量（改完要重开终端）
        3. 在 LangSmith 网页上建一个 agent 仓库，把下面的 HUB_IDENTIFIER
           改成 "你的组织名/仓库名"

课案出处：Agent 课案 → deepAgents → 运行环境 → ⑤ ContextHubBackend

运行方式：
    uv run Agent/03_deepagents/07_后端_ContextHub_jxsd.py

前置条件：LANGSMITH_API_KEY 环境变量（本机当前为空 → 走降级提示路径）。
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from deepagents import create_deep_agent
from deepagents.backends import ContextHubBackend
from langchain.chat_models import init_chat_model
from config import settings

# LangSmith Hub 仓库标识，owner/name 格式。用之前改成你自己的仓库。
HUB_IDENTIFIER = "your-org/my-agent"

# config.py 里没有 LangSmith 字段，按规范直接从环境变量读
langsmith_api_key = os.environ.get("LANGSMITH_API_KEY", "")

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


def run_with_context_hub() -> None:
    """有 LANGSMITH_API_KEY 时走这条真实路径。"""
    # 课案原文：需设置 LANGSMITH_API_KEY 环境变量
    # ContextHubBackend("owner/name") 内部会 new 一个 langsmith.Client()，
    # 此时才真正发起网络鉴权。
    backend = ContextHubBackend(HUB_IDENTIFIER)

    agent = create_deep_agent(model=llm, backend=backend)

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "当前目录有哪些文件"}]},
        config={"recursion_limit": 50},
    )
    print(result["messages"][-1].content)
    print(f"\n写入的文件可以在 LangSmith 网页上打开仓库 {HUB_IDENTIFIER} 查看（含 Git 提交历史）。")


if __name__ == "__main__":
    print("=" * 62)
    print("DeepAgents 后端⑤：ContextHubBackend（LangSmith Hub 云端仓库）")
    print("=" * 62)

    if not langsmith_api_key:
        # ---------- 降级路径：不抛异常，只给清晰的中文指引 ----------
        # 这一段是「缺外部账号时怎么体面地退出」的示范：
        # 说清 ① 为什么跳过 ② 想跑要做什么 ③ 有没有不依赖它的替代方案。
        print(
            "\n[跳过] 未检测到 LANGSMITH_API_KEY 环境变量，本示例不连接 LangSmith。\n"
            "\n为什么跳过：\n"
            "  ContextHubBackend 把文件写进 LangSmith Hub 的 agent 仓库，\n"
            "  构造后端时就会 new 一个 langsmith.Client() 去鉴权；\n"
            "  没有密钥会直接抛凭据异常 —— 这是外部账号依赖，不是代码问题。\n"
            "\n想真正运行，请依次完成：\n"
            "  1. 注册 https://smith.langchain.com 并创建 API Key；\n"
            "  2. 在本机设置环境变量（PowerShell，当前会话生效）：\n"
            '         $env:LANGSMITH_API_KEY = "lsv2_pt_xxxxxxxx"\n'
            "     （永久生效请用 setx，改完重开终端）\n"
            # 用 !r 把当前值打进提示里：学生能一眼看到「要改的是哪一个常量」
            f"  3. 把本文件里的 HUB_IDENTIFIER 从 {HUB_IDENTIFIER!r} 改成你自己的仓库，\n"
            '     格式为 "组织名/仓库名"，也可以是 "-/仓库名"（个人空间）；\n'
            "  4. 重新运行本文件。\n"
        )

        # 顺带说明「不依赖 LangSmith 时的替代方案」，方便对照理解
        # —— 教学上很重要：学生要能判断「这个后端值不值得引入一个外部账号」
        print(
            "不依赖 LangSmith 的等效做法（本项目 04_后端_Store_jxsd.py 用的就是它）：\n"
            "  用 StoreBackend + 固定 namespace 当共享枢纽，\n"
            "  同样能让多个会话/线程共享同一份文件，只是少了 Git 版本历史。\n"
        )
        # exit(0) 而不是非零退出：这是「按设计跳过」，在 CI 里不应算失败
        sys.exit(0)

    # ---------- 正常路径 ----------
    print(f"\n已检测到 LANGSMITH_API_KEY（长度 {len(langsmith_api_key)}），连接 Hub 仓库 {HUB_IDENTIFIER} …\n")
    try:
        run_with_context_hub()
    except Exception as exc:                      # noqa: BLE001 —— 外部服务异常要转成中文提示
        # 网络/鉴权这类外部失败不该甩 traceback，转成中文提示 + 排查清单后正常退出
        print(f"\n[失败] 连接 LangSmith Hub 出错：{type(exc).__name__}: {exc}")
        print("请检查：API Key 是否有效、HUB_IDENTIFIER 仓库是否存在、网络是否能访问 smith.langchain.com。")
        sys.exit(0)

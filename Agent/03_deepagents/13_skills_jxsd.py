# -*- coding: utf-8 -*-
"""
DeepAgents Skills（技能）：按需加载的「操作手册」
================================================================
课案出处：Agent 课案 → skills → deepagents skills（另见「云技能」小节）

本节要讲什么
    1. Skills **是什么**：把智能体的能力封装成「一个文件夹 + 一个 SKILL.md」，
       一次配置、永久复用，而不是每次往提示词里粘一大段（课案「概念」段）；
    2. 它**为什么省 Token**：「选择 → 学习 → 使用」三步渐进式披露，
       启动时只看 `name` + `description`，匹配到了才 read_file 拉正文，
       正文里提到的 `references/` 还要再等等才读（课案「原理」段）；
    3. **怎么用**：`create_deep_agent(skills=[技能父目录])` 一行接上，
       本文件 Part 1 现场造一个真实技能目录，并用 `stream_mode="updates"`
       把「启动时没有正文、匹配后才 read_file」这个证据打印出来；
    4. **云技能**：技能仓库也可以不是磁盘而是 Store
       （`skills=["/skills/"]` + CompositeBackend 路由到 StoreBackend），
       靠 namespace 做多用户隔离 —— 本文件 Part 2 会真的跑一遍（需 PostgreSQL）。

一、Skills 是什么（课案「概念」段）
    Skills 是把 AI 智能体封装成一个**可复用的组件**，以 Markdown 文件格式存在，
    功能通过**动态加载**实现。

    Skills 和传统 Prompt 的区别（课案三条）：

    | 维度 | 普通 Prompt | Skills |
    |---|---|---|
    | 配置成本 | 每次使用都要重新配置 | 一次配置，永久复用 |
    | Token 效率 | 每次调用全量加载 | 按需懒加载，只加载对应内容 |
    | 维护成本 | 跨场景反复复制粘贴 | 只改 SKILL.md，全局/项目级统一生效 |

二、Skills 的核心：一个文件夹 + 一个 SKILL.md
    目录结构（课案）：

        image-to-mermaid/
        ├── SKILL.md      # 必需：指令 + 元数据（YAML frontmatter）
        ├── scripts/      # 可选：执行脚本
        ├── references/   # 可选：文档资料（渐进式披露，用到才读）
        └── assets/       # 可选：资源（模板文件、示例图片、配置项）

    SKILL.md 基本模板：

        ---
        name: skill-name
        description: 说明这个 Skill 的功能以及使用场景
        ---

        # 在这里开始写你的内容，Markdown 格式 —— 智能体会在选择该技能时读取

    元数据字段（课案表）：

    | 字段 | 必需 | 说明 |
    |---|---|---|
    | name | 是 | Skill 名称，最长 64 字符，只允许小写字母、数字和 -，不能以 - 开头或结尾 |
    | description | 是 | 简短说明使用场景，最长 1024 字符，不能为空 |
    | trigger_keywords | 否 | 强制推荐关键词，自动触发时使用 |
    | license | 否 | 开源许可证或指向 Skill 附带的许可证文件 |
    | compatibility | 否 | 描述兼容性，最多 500 字符 |
    | metadata | 否 | 自定义键值对（作者、版本号等） |
    | allowed-tools | 否 | 允许使用的工具列表，空格分隔 |

三、「逐步加载」机制（为什么省 Token）
    1. **选择**：启动时只读每个技能的 name + description（几十个字符）
    2. **学习**：任务匹配到某个技能，才把该 SKILL.md 的正文加载进来
    3. **使用**：按指示执行，可参考其他文件（references/）或跑 scripts/

    课案的例子：一个「代码审查」Skill 里同时放了 Python 规范 200 行、
    JavaScript 规范 200 行、C++ 规范 300 行。用户说「帮我审查这段 Python 代码」时，
    AI 只需要加载 Python 那部分 —— 既快又省 Token。

四、课案代码与本文件的落地差异（重要）
    课案原文：

        model = ChatOpenAI(model=..., api_key=..., base_url=...)
        agent = create_deep_agent(
            model=model,
            backend=LocalShellBackend(root_dir=".", inherit_env=True),
            skills=["/"],          # 扫描根目录 -> 发现 /image-to-mermaid/SKILL.md
        )
        result = agent.invoke({"messages": [{"role": "user", "content": (
            "请使用 image-to-mermaid 技能，把 img.png 转换成 Mermaid 架构图代码。"
            "注意：不要直接用 read_file 读取图片文件，而是按 SKILL.md 的说明运行其中的转换脚本。"
        )}]})

    它的完整前置条件是：先下载 `haydon-image-gen.zip` 解压到 skills 目录、
    准备一张 img.png、并且脚本内部会调用 qwen-vl 视觉模型。
    这些都属于**外部下载 / 额外依赖**，本项目规范不允许新增依赖，
    所以本文件做了两处等价替换：

        | 课案 | 本文件 | 为什么 |
        |---|---|---|
        | 下载 haydon-image-gen.zip 拿 image-to-mermaid 技能 | 现场生成一个纯文本技能 report-writer | 不依赖外网下载，机制完全一样 |
        | `skills=["/"]` 扫描整个根目录 | `skills=[技能父目录]` | 扫根目录会把整台机器当技能仓库翻，既不安全也很慢 |

    **注意 `skills` 传的是技能的【父目录】，不是 SKILL.md 本身** —— 课案原话如此，
    也是本节最容易搞错的一点。

五、云技能（课案「云技能」小节）
    课案的原文总结：
    > DeepAgents 的 skills 机制原生支持两步加载：启动时只加载 SKILL.md 的 description
    > （省 Token），Agent 匹配到技能后才 read_file 拉取完整提示词。
    > 配合 StoreBackend 即可实现云端多用户隔离。

    所以技能的「仓库」也可以不是磁盘，而是 Store：
        skills=["/skills/"] + CompositeBackend(routes={"/skills/": StoreBackend(namespace=按用户)})，
    不同用户 namespace 不同 → 每个人看到自己的技能集。本文件 Part 2 会真的跑一遍。

六、云技能为什么必须挂在 CompositeBackend 上（把机制说透）
    `skills=["/skills/"]` 只是个**路径前缀**，它本身不说明文件在哪。
    真正决定「去哪读」的是 backend：
        - Part 1：backend 是 LocalShellBackend（root_dir=磁盘目录）→ 技能在本地磁盘；
        - Part 2：backend 是 CompositeBackend，`/skills/` 路由到 StoreBackend
          → 同一行 `skills=["/skills/"]`，技能却在 PostgreSQL 的 Store 里。
    这正是课案那句「原生支持两步加载 … 配合 StoreBackend 即可实现云端多用户隔离」的
    完整含义：**加载时机由 SkillsMiddleware 控制，加载位置由 backend 决定**。
    也正因如此，Part 2 预存技能时要把 **Store key 写成「剥掉路由前缀」之后的路径**
    —— 这是实测出来的（本机 deepagents 0.7.13）：

        预存 key 写成 /skills/code-review/SKILL.md
          → SkillsMiddleware 先 backend.ls("/skills/")，StoreBackend 拿到的真实目录是
            "/skills/"，返回给上层的路径被 remap 成 "/skills/skills/"
          → 结果「发现技能数 = 0」，Agent 根本看不到这个技能（静默失效，不报错）

        预存 key 写成 /code-review/SKILL.md（= /skills/ 被剥掉后的路径）
          → ls("/skills/") 返回 "/skills/code-review/"
          → SkillsMiddleware 读到 /skills/code-review/SKILL.md ✅（实测能列出 2 个技能）

    这条规则和 11_记忆_jxsd.py 第六节是**同一个坑**：经过 CompositeBackend 路由时，
    调用方路径会被剥掉前缀再交给目标后端；所以凡是「绕过 Agent 直接往 Store 里
    预存文件」的地方，key 都要按「剥掉前缀」来写。为了让 Part 2 真的能演示出
    效果，下面的预存同时写了两种 key（课案原样路径 + 正确路径），
    这样既能对照课案原文，又能真的跑出「云端技能被发现」的结果。

运行方式：
    uv run Agent/03_deepagents/13_skills_jxsd.py

前置条件：
    - Part 1（本地技能）：只需 .env 的模型配置可调通（实测可用）
    - Part 2（云技能）：额外需要 PostgreSQL（settings.pg_uri，实测可用）
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from pathlib import Path

# 下面这组 import 就是本节要用的全部依赖，按用途分四类看：
#   ① create_deep_agent       —— 建 Agent 本体
#   ② 三种 backend            —— CompositeBackend（路由）+ StateBackend（兜底）
#                               + StoreBackend（云技能仓库）/ LocalShellBackend（本地技能仓库）
#   ③ create_file_data        —— 往 Store 里存「文件」时必须用它的 FileData 结构
#   ④ PostgresStore/PostgresSaver + settings.pg_uri —— Part 2 的持久化与检查点
from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, StateBackend, StoreBackend
from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.utils import create_file_data
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

SKILLS_ROOT = Path(__file__).resolve().parent / "tmp_jxsd_deepagents_skills"

# 和 06 文件同样的原因：把当前 venv 的 Scripts 目录顶到 PATH 最前面，
# 免得技能里的脚本被 PATH 中那个「假的 python」静默吃掉。
VENV_SCRIPTS = str(Path(sys.executable).resolve().parent)


def build_local_skill() -> Path:
    """现场造一个技能：report-writer（含 references/，用来演示渐进式披露）。"""
    skill_dir = SKILLS_ROOT / "report-writer"
    (skill_dir / "references").mkdir(parents=True, exist_ok=True)

    # ---- SKILL.md：唯一必需的文件，元数据 + 指令 ----
    (skill_dir / "SKILL.md").write_text(
        """---
name: report-writer
description: 撰写正式工作周报时使用本技能。用户提到周报、日报、工作总结时加载。
---

# 周报撰写规范

1. 结构固定为三段：本周完成 / 下周计划 / 风险与求助
2. 每条用「动宾短语」开头，例如「完成了 XX 模块开发」
3. 总字数控制在 200 字以内

## 更多要求

如果用户没有特别说明，默认面向技术团队负责人汇报。
详细的行文口吻与用词禁忌见 `references/style.md`（需要时再读）。
""",
        encoding="utf-8",
    )

    # ---- references/：渐进式披露的「第二层」，只有真的用到才会被 read_file ----
    (skill_dir / "references" / "style.md").write_text(
        """# 行文风格

- 不写「我觉得」「大概」「可能」，一律给确定结论
- 不写与本周无关的背景铺垫
- 风险项必须写清「影响面 + 需要的支持」
""",
        encoding="utf-8",
    )
    return skill_dir


def part1_local_skills() -> None:
    """Part 1：技能放在真实磁盘上（对应课案的 deepagents skills 代码）。"""
    print("=" * 66)
    print("Part 1：本地技能目录 + LocalShellBackend（课案的 deepagents skills 写法）")
    print("=" * 66)

    skill_dir = build_local_skill()
    print(f"技能父目录（skills 参数传的就是它）：{SKILLS_ROOT}")
    print(f"技能目录结构：")
    for path in sorted(SKILLS_ROOT.rglob("*")):
        if path.is_file():
            print(f"    {path.relative_to(SKILLS_ROOT)}")
    print()

    agent = create_deep_agent(
        model=llm,
        # 课案用 LocalShellBackend，是因为它的技能里有转换脚本要执行；
        # 这里保持同一类后端，同时也能执行技能里可能带的 scripts/。
        backend=LocalShellBackend(
            root_dir=str(SKILLS_ROOT),
            inherit_env=True,
            env={"PATH": VENV_SCRIPTS + os.pathsep + os.environ.get("PATH", "")},
        ),
        # 课案是 skills=["/"]（扫描根目录）；这里指向技能的父目录，语义相同但更安全。
        # 关键：传父目录，SkillsMiddleware 会自己去里面找 */SKILL.md。
        skills=[str(SKILLS_ROOT)],
        system_prompt="你是职场写作助手。需要写周报时，先读技能文件按规范来写。",
    )

    # 用 updates 模式把工具调用打出来 —— 这是「渐进式披露」最直接的证据：
    # 启动时系统提示里只有 description，模型决定用技能之后才会 read_file 拉 SKILL.md 正文。
    print("执行任务（同时打印 Agent 的工具调用轨迹）：")
    # 用 updates 模式跑，是为了把「渐进式披露」的证据留在输出里：
    # 启动时的系统提示里只有 name + description，模型决定用这个技能之后
    # 才会出现 read_file 去拉 SKILL.md 正文 —— 有没有这一步，就是机制有没有生效的判据。
    for chunk in agent.stream(
        {"messages": [("user", "帮我写本周周报：完成了登录模块，下周做支付")]},
        stream_mode="updates",
        # 技能任务链路更长（选技能 → 读 SKILL.md → 可能再读 references → 写正文），放宽步数
        config={"recursion_limit": 50},
    ):
        for node_name, update in chunk.items():
            # 有的中间件钩子节点只写非消息字段，取不到 messages 就当空列表处理
            messages = update.get("messages", []) if isinstance(update, dict) else []
            for message in messages:
                # ① 模型这一轮决定调用哪些工具、传什么参数（read_file / write_file 都在这里现形）
                for call in getattr(message, "tool_calls", None) or []:
                    print(f"  [{node_name}] → {call.get('name')}({call.get('args')})")
                # ② 工具返回：只打印长度，避免把整个 SKILL.md 正文刷到屏幕上
                if type(message).__name__ == "ToolMessage":
                    print(f"  [{node_name}] ← {message.name} 返回 {len(str(message.content))} 字符")
                # ③ 其余带正文的消息 = 模型写出来的周报本身
                elif getattr(message, "content", ""):
                    print(f"\n【周报】\n{message.content}\n")

    print("注意上面是否出现 read_file 读取 SKILL.md —— 那就是「匹配到才加载」。")


def part2_cloud_skills() -> None:
    """Part 2：技能存进 Store，按用户 namespace 隔离（对应课案的「云技能」小节）。"""
    print("\n" + "=" * 66)
    print("Part 2：云技能 —— skills 指向 Store 里的 /skills/（课案「云技能」小节）")
    print("=" * 66)

    if not settings.pg_uri:
        print("[跳过] 未配置 settings.pg_uri（.env 里的 PG_URI），无法演示云端技能仓库。")
        return

    with (
        PostgresStore.from_conn_string(settings.pg_uri) as store,
        PostgresSaver.from_conn_string(settings.pg_uri) as checkpointer,
    ):
        store.setup()
        checkpointer.setup()

        # 为不同用户预存不同的 skill：这里演示 user-001 的两个技能。
        # key 的写法有讲究（详见文件头第六节，实测结论）：
        #   · "/skills/code-review/SKILL.md" —— 课案原样路径，保留下来做对照；
        #     但 CompositeBackend 会把 /skills/ 路由前缀剥掉再交给 StoreBackend，
        #     于是 StoreBackend 的真实目录成了 "/skills/"，
        #     ls("/skills/") 只回一个 "/skills/skills/"，技能发现数为 0（静默失效）。
        #   · "/code-review/SKILL.md" —— 剥掉前缀后的路径，这才是 SkillsMiddleware
        #     ls("/skills/") 时真正会去翻的那一层，写它技能才能被发现。
        # 两个 key 都写（内容相同，namespace 隔离不受影响），
        # 既能对照课案，又能真的跑出「云端技能被发现」的效果。
        store.put(
            ("user-001",),
            "/skills/code-review/SKILL.md",
            create_file_data(
                """---
name: code-review
description: Python 代码审查，检查规范性和性能问题
---

你是 Python 专家，审查以下代码的规范性、性能和安全性。
输出格式：先列问题（按严重程度排序），再给修改后的代码。
"""
            ),
        )
        # 同上，另存一份「剥掉 /skills/ 前缀」的 key —— 这一份才会被 SkillsMiddleware 找到
        store.put(
            ("user-001",),
            "/code-review/SKILL.md",
            create_file_data(
                """---
name: code-review
description: Python 代码审查，检查规范性和性能问题
---

你是 Python 专家，审查以下代码的规范性、性能和安全性。
输出格式：先列问题（按严重程度排序），再给修改后的代码。
"""
            ),
        )
        store.put(
            ("user-001",),
            "/skills/sql-gen/SKILL.md",
            create_file_data(
                """---
name: sql-gen
description: 根据自然语言生成 SQL 语句
---

根据用户的自然语言描述生成对应的 SQL 语句。
要求：只输出一段 SQL，表名用 users / orders，不要解释。
"""
            ),
        )
        # 同上：sql-gen 也补一份正确 key
        store.put(
            ("user-001",),
            "/sql-gen/SKILL.md",
            create_file_data(
                """---
name: sql-gen
description: 根据自然语言生成 SQL 语句
---

根据用户的自然语言描述生成对应的 SQL 语句。
要求：只输出一段 SQL，表名用 users / orders，不要解释。
"""
            ),
        )
        print("已向 Store 的 namespace ('user-001',) 预存 2 个技能：code-review / sql-gen")
        print("（Store 里的 key 同时写了课案原样路径与「剥掉 /skills/ 前缀」的正确路径）\n")

        agent = create_deep_agent(
            model=llm,
            # 技能仓库在 Store 里，路径前缀 /skills/
            skills=["/skills/"],
            backend=CompositeBackend(
                default=StateBackend(),
                routes={
                    "/skills/": StoreBackend(
                        # 本地用固定值；生产改成 lambda rt: (rt.server_info.user.identity,)
                        # 就能做到「每个用户看到自己的技能集」
                        namespace=lambda rt: ("user-001",),
                    ),
                },
            ),
            store=store,
            checkpointer=checkpointer,
        )

        # thread_id 用带前缀的名字，避免和同目录其他文件（它们也用 PostgresSaver
        # 往同一个库写检查点，课案里写的是 "1"）串到同一条对话历史上去。
        config = {"configurable": {"thread_id": "jxsd-skills-1"}}
        result = agent.invoke(
            {"messages": [{"role": "user", "content": "用技能sql-gen查询用户总数"}]},
            config={**config, "recursion_limit": 50},
        )
        # 和 Part 1 一样打印**完整消息轨迹**，方便确认技能有没有被真的加载：
        # 只要看到 read_file("/skills/sql-gen/SKILL.md")，就说明云技能被发现了。
        for m in result["messages"]:
            print(m)
            print("-" * 50)

        print("\n最终回答：")
        print(result["messages"][-1].content)
        print(
            "\n要点：技能文件不在磁盘上，而是存在 Store 的 ('user-001',) 命名空间里；\n"
            "      换一个 namespace 就是另一套技能 —— 这就是课案说的「云端多用户隔离」。"
        )


if __name__ == "__main__":
    part1_local_skills()
    part2_cloud_skills()

# -*- coding: utf-8 -*-
r"""
LangChain 官方补充篇：Skills 渐进披露（非课案内容，故不带 _jxsd 后缀）
================================================================
来源与定位：
    本文件对照 LangChain **官方文档** /oss/python/langchain/multi-agent/skills.mdx
    （及配套教程 skills-sql-assistant.mdx），补上官方**多 Agent 五模式**里课案缺的那个。

    官方多 Agent 五种模式      课案覆盖            本文件
    --------------------------  ------------------  ------------------
    1. subagents（子代理）       ✅ 12_多Agent_MCP子Agent
    2. handoffs（交接）          ✅ 13_多Agent_交接
    3. router（路由）            ✅ 14_多Agent_路由与合并
    4. skills（技能渐进披露）    ❌ 缺              ← 本文件
    5. custom-workflow           ❌ 缺              见 18_自定义工作流_官方补充.py

和课案 08_skills 章的区别（容易混，先说清）：
    08_skills 讲的是 **DeepAgents 内置**的 skills（SkillsMiddleware + agentskills.io 规范，
    技能目录放好后由框架自动发现与注入）。
    本文件讲的是 **LangChain create_agent 侧的手工版** —— 官方原文明确说：
    「内置 skill 支持见 Deep Agents」，LangChain 侧要自己搭一个 `load_skill` 工具。
    两条路线的关系：08 章是「用框架的能力」，本文件是「自己实现这套机制」。

核心思想（官方 Key characteristics）：
    - **提示词驱动的专业化**：技能本质是一段专业提示词 / 领域知识，不是代码；
    - **渐进披露（progressive disclosure）**：技能**按需加载**，不加载就不占上下文；
    - 与 agentskills.io / llms.txt 同构：用工具调用做文档/知识的分级披露。

⚠️ 本文件与前面几篇补充篇不同：**它需要真实模型**（技能加载与否由模型决策），
   不是纯离线。因此按仓库惯例：模型没按预期调用工具时，打印中文提示并给出重跑建议，
   而不是抛 traceback。

缺口表对应：`Agent/官方文档缺口对照.md` 的 **LangChain 第 4 项**（Skills 渐进披露）。

运行方式（项目根目录下，会调用 .env 里的模型）：
    uv run Agent/02_langchain/17_Skills渐进披露_官方补充.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import tempfile
from pathlib import Path

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from config import settings

model = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


# ================================================================
# 技能库：技能就是磁盘上的一个目录（SKILL.md + 可选附属资源）
# ================================================================
# 目录约定与课案 08_skills 章一致（agentskills.io 规范）：
#     skills/<技能名>/SKILL.md        ← 技能正文（YAML frontmatter + Markdown）
#     skills/<技能名>/assets/...      ← 附属资源（schema、模板、脚本），**用到时才读**
# 这种「技能目录 + 附属资源」的组织来自 agentskills.io 规范（与课案 08_skills 一致）；
# 而「技能正文只指向附属文件、用到时才读」这个**引用感知（reference awareness）**模式，
# 出自官方 skills.mdx 的 "Extending the pattern" 一节。
# ⚠️ 别记混：官方那篇 SQL 教程（skills-sql-assistant）用的是**内存字典**装技能正文
#    （原文结尾自述 "implemented skills as in-memory Python dictionaries"），
#    并没有 assets/schema.sql 这种文件结构 —— 本文件才是文件 + 附属资源的组织形式。
SKILLS: dict[str, str] = {
    "write_sql": """---
name: write_sql
description: SQL 查询编写专家：先给可执行 SQL，再解释思路
---

# SQL 编写技能

回答数据库问题时，严格按以下格式输出：

1. **SQL**：先给出一个可直接执行的 SQL 语句（用 ```sql 代码块）；
2. **解释**：再用不超过三句话说明思路（用到哪些表、为什么这样过滤）；
3. 不要臆造字段名；字段不确定时，先说明"需要看 schema"。

如需详细表结构，请读取附属文件：assets/schema.sql
""",
    "review_legal_doc": """---
name: review_legal_doc
description: 法务文档审阅：逐条列出风险点与修改建议
---

# 法务文档审阅技能

审阅合同时，输出一张风险清单，每条包含：

- **原文摘录**（不超过一行）
- **风险等级**（高 / 中 / 低）
- **修改建议**（一句话）

不要给笼统结论，必须逐条对应原文。
""",
}

SKILL_ASSETS: dict[tuple[str, str], str] = {
    ("write_sql", "assets/schema.sql"): """-- 简化的订单库表结构
CREATE TABLE users   (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
CREATE TABLE orders  (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL, created_at TEXT);
""",
}


def build_skill_library(root: Path) -> None:
    """把技能写到磁盘（真实项目里这些文件本来就躺在仓库里）。"""
    for name, content in SKILLS.items():
        skill_dir = root / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")
    for (name, rel), content in SKILL_ASSETS.items():
        asset = root / name / rel
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_text(content, encoding="utf-8")


def make_skill_tools(skills_root: Path) -> list:
    """按官方模式造两个工具：加载技能正文、读取技能的附属资源。"""

    @tool
    def load_skill(skill_name: str) -> str:
        """加载一个专业技能提示词。

        可用技能：
        - write_sql：SQL 查询编写专家
        - review_legal_doc：法务文档审阅

        返回该技能的完整提示词。
        """
        # 官方基础实现就是把技能正文读出来返回；技能名不合规时给出可用清单
        path = skills_root / skill_name / "SKILL.md"
        if not path.exists():
            return f"没有名为 {skill_name!r} 的技能。可用技能：{', '.join(SKILLS)}"
        return path.read_text(encoding="utf-8")

    @tool
    def read_skill_asset(skill_name: str, relative_path: str) -> str:
        """读取某个技能目录下的附属资源文件（如 assets/schema.sql）。"""
        path = (skills_root / skill_name / relative_path).resolve()
        # 安全边界：附属资源必须落在该技能目录内（防 ../ 越权读取）。
        # 用 is_relative_to 而不是字符串 startswith —— 后者会被「同名前缀目录」绕过
        # （例如 /skills/write_sql_x 以 /skills/write_sql 开头），这类越权很难察觉。
        if not path.is_relative_to((skills_root / skill_name).resolve()):
            return "路径越界，已拒绝"
        if not path.is_file():      # is_file 而不是 exists：目录也会让 exists 为真
            available = [rel for (n, rel) in SKILL_ASSETS if n == skill_name]
            return f"资源不存在。该技能可用的附属文件：{available or '（无）'}"
        return path.read_text(encoding="utf-8")

    return [load_skill, read_skill_asset]


SYSTEM_PROMPT = (
    "你是一个通用助手，可通过技能获得专业能力。\n"
    "可用技能：write_sql（SQL 编写）、review_legal_doc（法务审阅）。\n"
    "当用户的问题属于某个技能的领域时，先调用 load_skill 加载该技能的提示词，"
    "然后严格按技能里的要求作答；技能正文里提到的附属文件用 read_skill_asset 读取。"
)


def has_tool_call(result: dict, tool_name: str) -> bool:
    """检查这轮对话里模型是否真的调用了某个工具（用于兜底提示）。"""
    for message in result.get("messages", []):
        for call in getattr(message, "tool_calls", None) or []:
            if call["name"] == tool_name:
                return True
    return False


def final_text(result: dict) -> str:
    return str(result["messages"][-1].content)


# ================================================================
# Demo 1：基础模式 —— 技能按需加载
# ================================================================
def demo_1_basic_skill_loading(skills_root: Path) -> None:
    print("=" * 70)
    print("Demo 1：基础模式 —— 模型自己决定加载哪个技能")
    print("=" * 70)

    agent = create_agent(
        model=model,
        tools=make_skill_tools(skills_root),
        system_prompt=SYSTEM_PROMPT,
    )
    result = agent.invoke({
        "messages": [{"role": "user", "content": "帮我查一下每个城市的订单总金额，写个 SQL"}],
    })

    if not has_tool_call(result, "load_skill"):
        print("  ⚠️ 本轮模型没有调用 load_skill（偶发行为），重跑一次通常就有；")
        print("     也可以把 system_prompt 里的要求写得更强制。")
        print("  模型直接回答：", final_text(result)[:120])
        return

    # 打印模型加载了哪个技能（工具消息里有技能正文）
    for message in result["messages"]:
        if message.type == "tool" and getattr(message, "name", "") == "load_skill":
            first_line = str(message.content).strip().splitlines()[0]
            print(f"  ✔ 模型调用了 load_skill，读到的技能开头：{first_line[:40]}")

    answer = final_text(result)
    print(f"  最终回答（前 200 字）：\n    {answer[:200].replace(chr(10), chr(10) + '    ')}")
    if "select" in answer.lower():
        print("  ✔ 回答里出现了 SQL（技能要求的格式生效了）")
    else:
        print("  （回答里没看到 SQL —— 模型可能没完全遵守技能格式，属模型行为）")
    print(
        "  ↑ 关键：技能正文是在**模型决定要用它之后**才进入上下文的。\n"
        "    用户问的是 SQL 问题时，法务技能一个字都没进上下文 —— 这就是渐进披露。"
    )


# ================================================================
# Demo 2：渐进披露到底省了多少上下文（可量化）
# ================================================================
def demo_2_disclosure_savings() -> None:
    print("\n" + "=" * 70)
    print("Demo 2：渐进披露省了多少上下文（量化对比）")
    print("=" * 70)

    # 方案 A：把**所有技能全文**塞进 system_prompt（"全量注入"，很多项目这么干）
    full_injection = SYSTEM_PROMPT + "\n\n" + "\n\n".join(SKILLS.values())
    # 方案 B：只列技能名与用途，正文按需加载（本文件的模式）
    def description_of(body: str) -> str:
        """从 SKILL.md 的 frontmatter 里取 description（按前缀找，不靠行号下标）。"""
        for raw_line in body.splitlines():
            if raw_line.strip().startswith("description:"):
                return raw_line.split("description:", 1)[1].strip()
        return "（无描述）"

    skill_index = "\n".join(f"- {name}：{description_of(content)}" for name, content in SKILLS.items())
    lazy = SYSTEM_PROMPT + "\n" + skill_index

    a, b = len(full_injection), len(lazy)
    print(f"  技能数量：{len(SKILLS)} 个")
    print(f"  方案 A 全量注入 system_prompt：{a} 字符")
    print(f"  方案 B 只列索引、正文按需加载：{b} 字符")
    print(f"  → 本轮省下 {a - b} 字符（约 {(a - b) / a:.0%}）")
    print(
        "  ↑ 技能越多差距越大（省的是**每轮请求**都要带的那份固定开销）；\n"
        "    渐进披露的代价是多一次工具调用（多一个来回）——\n"
        "    所以技能正文大的场景收益明显，技能很短时全量注入反而更省事。"
    )


# ================================================================
# Demo 3：引用感知 —— 技能正文指向附属文件，用到时才读（第二级披露）
# ================================================================
# 官方 "Extending the pattern → Reference awareness" 讲的正是这一招：
# 技能正文只说「需要详细表结构时读 assets/schema.sql」，
# 于是**schema 也不会在加载技能时一起进上下文**，而是等模型真需要时才读。
def demo_3_reference_awareness(skills_root: Path) -> None:
    print("\n" + "=" * 70)
    print("Demo 3：引用感知 —— 技能指向附属文件，模型按需再读")
    print("=" * 70)

    agent = create_agent(
        model=model,
        tools=make_skill_tools(skills_root),
        system_prompt=SYSTEM_PROMPT,
    )
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "写个 SQL 统计每个城市的订单总金额。我不确定表结构，你先看 schema 再写。",
        }],
    })

    loaded = has_tool_call(result, "load_skill")
    read_asset = has_tool_call(result, "read_skill_asset")
    print(f"  加载技能 load_skill：{'✔' if loaded else '✗ 本轮没调'}")
    print(f"  读取附属文件 read_skill_asset：{'✔' if read_asset else '✗ 本轮没调'}")
    for message in result["messages"]:
        if message.type == "tool" and getattr(message, "name", "") == "read_skill_asset":
            print(f"    schema 内容（工具返回）：{str(message.content)[:60]}…")
    print(f"  最终回答（前 160 字）：{final_text(result)[:160]}")
    print(
        "  ↑ 两级披露：技能正文（第一级）→ 附属资源（第二级）。\n"
        "    如果模型这轮没读附属文件，是因为它觉得技能正文够了 —— 这是**模型判断**，\n"
        "    生产里可以在技能正文里把「必须先读 schema」写得更硬（或干脆用权限规则限制）。"
    )
    print(
        "\n  官方提到的另外两种扩展（本文件不展开，留作练习）：\n"
        "    · 动态工具注册：加载技能的同时注册新工具（技能正文 + 工具集一起变强）；\n"
        "    · 层级技能：技能里再定义子技能（data_science → pandas_expert / viz / stats）。"
    )


if __name__ == "__main__":
    # 技能写到临时目录：本文件自包含、可重复运行，不往仓库里塞演示文件。
    # 真实项目请把技能目录放进版本库（现成例子见 Agent/08_skills/skills/）。
    with tempfile.TemporaryDirectory(prefix="skills_demo_") as tmp:
        skills_root = Path(tmp) / "skills"
        build_skill_library(skills_root)
        print(f"技能库已生成到临时目录：{skills_root}")
        print(f"目录结构：{[str(p.relative_to(skills_root)) for p in sorted(skills_root.rglob('*')) if p.is_file()]}\n")

        demo_1_basic_skill_loading(skills_root)
        demo_2_disclosure_savings()
        demo_3_reference_awareness(skills_root)

    print("\n全部 Demo 执行完毕。")


# ================================================================
# 实测结论 / 与课案的衔接 / 踩坑提示
# ================================================================
# 1. 官方事实（multi-agent/skills.mdx）：
#    - LangChain 侧**没有** skills API（本地实测：langchain / langchain.agents /
#      agents.middleware / tools 里都没有 skills 相关符号），官方给的就是
#      「手写 load_skill 工具 + system_prompt 里列技能」这个模式；
#    - 内置 skill 支持在 DeepAgents（课案 08_skills 章）；
#    - 官方列了三种扩展：动态工具注册 / 层级技能 / 引用感知（本文件 Demo 3 演示第三种）。
# 2. 与课案的衔接：
#    - 课案 08_skills 讲 deepagents 内置 skills（框架自动发现技能目录）；
#      本文件是同一套思想的「手工实现版」，两相对照能看清框架替你做了什么；
#    - 官方多 Agent 五模式至此覆盖 4 种，剩 custom-workflow 见下一文件（18_）。
# 3. 踩坑提示：
#    A. 技能名要**写进工具 docstring 或 system_prompt**：模型不会去猜目录里有什么，
#       不列清单它就不会调 load_skill（这是渐进披露模式最常见的落地失败原因）；
#    B. 技能正文里的附属文件路径要用**技能目录内相对路径**，并在工具里做越界校验
#       （本文件 read_skill_asset 已加 startswith 检查，防 `../` 读到仓库其他文件）；
#    C. 技能不是越细越好：技能太碎会导致模型反复加载、来回多花好几轮；
#       官方建议按「团队/领域」粒度切分，而不是按单个问题切分；
#    D. 渐进披露省的是**每轮固定开销**，代价是**多一次工具往返** —— 技能很短时
#       全量注入反而更快（Demo 2 给了量化思路）；
#    E. 技能目录请进版本库并配 code review：技能正文会直接改变模型行为，
#       它和代码一样需要评审（课案 08_skills 的 code-review-skill 就是这个思路）。

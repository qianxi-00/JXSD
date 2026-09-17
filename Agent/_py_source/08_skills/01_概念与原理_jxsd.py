# -*- coding: utf-8 -*-
"""
Skills 概念与原理：一个文件夹 + 一个 SKILL.md，就是一个「可复用技能包」
================================================================
课案出处：Agent 课案 → skills → 概念 / 原理 / 目录结构 / SKILL.md

本节要讲什么（对应课案的「概念 / 原理 / 目录结构 / SKILL.md」四小节）：

    1. 【概念】Skills 是什么 —— 它不是「一段更长的提示词」，而是 AI 智能体的
       插件系统：把某个应用场景的专家知识封装成一个可复用、可安装的组件；
    2. 【区别】Skills 与传统 Prompt 的三点差别：配置成本 / Token 效率 / 维护成本。
       这三条是本节最该背下来的东西，也是后面每节都在反复印证的结论；
    3. 【原理】一个文件夹 + 一个 SKILL.md，靠「逐步加载（渐进式披露）」三步走
       —— 选择（只读 name + description）→ 学习（命中才读 SKILL.md 正文）
       → 使用（按正文指示再读某个 references/ 文件或跑 scripts/ 脚本）；
    4. 【结构】目录结构：最小结构只有 SKILL.md；完整结构多出 scripts/、
       references/、assets/ 三个可选目录，它们的分工各不相同；
    5. 【格式】SKILL.md 的基本模板，以及顶部 YAML front-matter 里
       name / description / license / compatibility / metadata / allowed-tools
       等八个字段的含义与约束。

本节把课案里「概念」「原理」「目录结构」「SKILL.md」四节文字原样搬进注释，
并在运行时把两类东西**真的打印到控制台**，方便对照记忆：

    1. 目录树：最小结构（只有 SKILL.md）与完整结构（scripts/ references/ assets/）；
    2. 元数据字段表：SKILL.md 顶部 YAML front-matter 里每个字段的含义与约束。

一句话总结课案的核心论断：
    Skills 本质上是把 AI 智能体的某个应用场景的专家知识，
    封装成一个「可复用的组件」；它以 Markdown 文件格式存在，
    执行功能通过**动态加载**实现。

和相邻小节的关系：
    - 本节只讲「是什么 / 为什么 / 长什么样」，不调大模型、不装任何第三方包；
    - 02_SKILL示例_jxsd.py  真正在磁盘上生成一个可用的 skill，并演示渐进式披露；
    - 03_deepagents技能_jxsd.py  用 deepagents 的 create_deep_agent(skills=[...]) 加载它；
    - 04_云技能_jxsd.py  把 skill 放到 PostgreSQL 里，实现云端多用户隔离；
    - 05_ClaudeCode技能_jxsd.py  讲 Claude Code 等客户端的技能安装路径。

运行前置条件：无（纯标准库，离线可跑）。

运行方式：
    uv run Agent/08_skills/01_概念与原理_jxsd.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，防中文/emoji 报错

from pathlib import Path

# 本文件所在目录 = Agent/08_skills；父目录的父目录 = 仓库根 F:\ProGram\Python_Base
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
# 后续小节（02）会把示例技能生成到这个目录下；本节只做「如果已经存在就顺手打印」。
SKILLS_ROOT = HERE / "skills"


# ================================================================
# 小工具：让中文表格在控制台里对齐
# ================================================================
def display_width(text: str) -> int:
    """计算字符串在等宽终端里占多少「列」。

    中文、中文标点是全角字符，占 2 列；ASCII 占 1 列。
    不处理这个，用 str.ljust() 排出来的中文表格会歪掉。
    """
    width = 0
    for ch in text:
        # 覆盖常用中日韩统一表意文字与常见全角标点区段，够本项目表格使用。
        # 只判「是不是全角」而不做完整 Unicode 宽度表，是因为表格里的字符
        # 全部来自本文件的常量，范围可控；真要做通用终端宽度，得用 wcwidth 那类表。
        if "\u1100" <= ch <= "\u115f" or "\u2e80" <= ch <= "\ua4cf" or "\uac00" <= ch <= "\ud7a3":
            width += 2
        elif "\uf900" <= ch <= "\ufaff" or "\ufe30" <= ch <= "\ufe6f" or "\uff00" <= ch <= "\uff60":
            width += 2
        else:
            width += 1
    return width


def pad(text: str, width: int) -> str:
    """按「显示宽度」右侧补空格，用于打印中文表格。"""
    return text + " " * max(0, width - display_width(text))


def print_table(headers: list[str], rows: list[list[str]], widths: list[int]) -> None:
    """打印一张宽度可控的表格（分隔线用 ASCII，避免不同终端渲染差异）。

    单元格里允许出现 `\\n`，会被拆成多行显示（同一行的其它列补空）。
    """
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(line)
    print("| " + " | ".join(pad(h, w) for h, w in zip(headers, widths)) + " |")
    print(line)
    for row in rows:
        # 每个单元格按 \n 拆成多行，整行高度取最大值
        cells = [c.split("\n") for c in row]
        for i in range(max(len(c) for c in cells)):
            parts = [pad(c[i] if i < len(c) else "", w) for c, w in zip(cells, widths)]
            print("| " + " | ".join(parts) + " |")
        print(line)


def render_tree(node: dict, prefix: str = "") -> list[str]:
    """把嵌套 dict 渲染成 `tree` 风格的文本行。

    约定（三种取值）：
        dict            → 目录，递归展开
        str             → 文件，字符串就是行尾注释
        ("dir", "注释") → 带注释的目录（不展开）
    """
    lines: list[str] = []
    items = list(node.items())
    for index, (name, payload) in enumerate(items):
        is_last = index == len(items) - 1
        branch = "└── " if is_last else "├── "
        if isinstance(payload, dict):
            lines.append(f"{prefix}{branch}{name}/")
            # 子节点前缀：最后一个用空格续行，其余用竖线续行
            lines.extend(render_tree(payload, prefix + ("    " if is_last else "│   ")))
        elif isinstance(payload, tuple):
            # 带注释的目录。注释可能有多行，续行要缩进到「注释起始列」：
            # 前缀(4) + 名字宽度 + "/"(1) + "    # "(6)
            _kind, note = payload
            note_lines = note.split("\n")
            lines.append(f"{prefix}{branch}{name}/    # {note_lines[0]}")
            # 注释换行后要对齐到 "#" 后面的那一列，否则树形结构看起来是断的；
            # 这里用 display_width(name) 而不是 len(name)，就是因为中文目录名占 2 列。
            cont = prefix + ("    " if is_last else "│   ")
            cont += " " * (display_width(name) + 1 + 6)
            lines.extend(f"{cont}{extra}" for extra in note_lines[1:])
        else:
            suffix = f"    # {payload}" if payload else ""
            lines.append(f"{prefix}{branch}{name}{suffix}")
    return lines


def print_real_tree(root: Path, max_depth: int = 3) -> None:
    """打印磁盘上真实存在的目录树（用于观察 02 小节生成的结果）。"""
    if not root.exists():
        print(f"  （{root} 还不存在 —— 先运行 02_SKILL示例_jxsd.py 就会生成）")
        return

    def walk(path: Path, prefix: str = "", depth: int = 0) -> None:
        if depth > max_depth:
            return
        # 排序键 (p.is_file(), p.name)：先目录后文件，同类按名字排。
        # 不排的话 iterdir() 的顺序依赖文件系统，每次打印出来的树都可能不一样。
        children = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        for index, child in enumerate(children):
            is_last = index == len(children) - 1
            branch = "└── " if is_last else "├── "
            mark = "/" if child.is_dir() else ""
            size = f"    # {child.stat().st_size} 字节" if child.is_file() else ""
            print(f"  {prefix}{branch}{child.name}{mark}{size}")
            if child.is_dir():
                walk(child, prefix + ("    " if is_last else "│   "), depth + 1)

    print(f"  {root.name}/")
    walk(root)


# ================================================================
# 1. 概念：Skills 是什么，它和「传统 Prompt」差在哪
# ================================================================
def section_concept() -> None:
    print("=" * 78)
    print("1. 概念：Skills 是 AI 智能体的插件系统，封装特定应用场景的专家知识")
    print("=" * 78)
    print(
        """
课案原文要点：
    Skills 本质上是把 AI 智能体封装成一个【可复用的组件】。
    Skills 以 Markdown 文件格式存在，执行功能通过动态加载实现。

也就是说，你不是在「写提示词」，而是在「发布一个能力包」：
    提示词是散落在各个项目里的一次性文本；
    技能是有名字、有描述、有版本、可以安装/卸载/复用的一个目录。
"""
    )

    # 课案给出的三点区别，逐条落成表；这三条是本节最该背下来的东西。
    print("【Skills 和传统 Prompt 的区别】三点：")
    rows = [
        [
            "配置成本",
            "每次使用都要重新配置，\n复制粘贴一遍提示词",
            "仅需一次配置，后续可永久复用；\n放在技能目录里谁都能装",
        ],
        [
            "Token 效率",
            "每次调用【全量加载】内容，\n提示词越长占用越高",
            "按需懒加载：平时只看到名字+描述，\n匹配到才加载正文，大幅降低消耗",
        ],
        [
            "维护成本",
            "需跨场景、跨项目反复复制粘贴，\n维护繁琐易出错",
            "只改 SKILL.md 一个文件，\n即可实现全局/项目级统一生效",
        ],
    ]
    # 单元格里有 \n，这里按行拆开打印，保证等宽对齐
    headers = ["维度", "传统 Prompt", "Skills"]
    widths = [10, 30, 32]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(line)
    print("| " + " | ".join(pad(h, w) for h, w in zip(headers, widths)) + " |")
    print(line)
    for row in rows:
        cells = [c.split("\n") for c in row]
        height = max(len(c) for c in cells)
        for i in range(height):
            parts = [pad(c[i] if i < len(c) else "", w) for c, w in zip(cells, widths)]
            print("| " + " | ".join(parts) + " |")
        print(line)


# ================================================================
# 2. 原理：逐步加载（渐进式披露）三步走
# ================================================================
def section_principle() -> None:
    print()
    print("=" * 78)
    print("2. 原理：一个文件夹 + 一个 SKILL.md，靠「逐步加载」省 Token")
    print("=" * 78)
    print(
        """
课案原文要点：
    Skills 的核心就是：一个文件夹 + 一个 SKILL.md 文件。

【逐步加载机制】三步：
    1. 选择：AI 只读取每个技能的名称和描述（只有几十字符）
    2. 学习：当匹配到某个技能时，AI 才会把该 SKILL.md 里的内容加载进来
    3. 使用：AI 按照指示执行，可参考其他文件、写代码等

对应到 deepagents 的实现（SkillsMiddleware，本机 0.7.13 实测）：
    第 1 步发生在 agent 启动时 —— 中间件 ls 技能父目录，对每个子目录
        下载 SKILL.md、只解析 YAML front-matter，把 name/description
        拼进系统提示词（所以「名字+描述」常驻上下文，正文不常驻）；
    第 2 步发生在模型决定用这个技能时 —— 它自己发起 read_file，
        把 SKILL.md 全文（含"去读 references/xxx.md"这类指令）拉进上下文；
    第 3 步是模型按 SKILL.md 的指示干活：读参考文件、跑 scripts/ 里的脚本。
"""
    )

    print("【为什么要逐步加载？】课案给的例子 —— 一个「代码审查」Skill：")
    tree = {
        "code-review-skill": {
            "SKILL.md": "只有 10 行，充当『路由器』",
            "references": {
                "python_rules.md": "200 行",
                "javascript_rules.md": "200 行",
                "cpp_rules.md": "300 行",
            },
        }
    }
    for line in render_tree(tree):
        print("  " + line)

    print(
        """
课案原文结论：
    如果用户说「帮我审查这段 Python 代码」，AI 只需要加载 Python 规范部分，
    不需要加载 JavaScript 规范和 C++ 规范。
    这样【既能让 AI 快速响应，同时还能节省 Token】。

算一笔账（课案例子的极端情况）：
    全量塞进提示词  = 10 + 200 + 200 + 300 = 710 行，每一轮对话都要带；
    逐步加载        = 常驻 10 行（其实是 name+description 几十字符）
                    + 命中后读 10 行 SKILL.md
                    + 真正需要的 200 行 python_rules.md
                    ≈ 220 行，且 javascript/cpp 那 500 行永远不进上下文。
    技能越多、参考文件越长，这个差距越大 —— 这是 Skills 相对「把资料全写进
    system prompt」最大的工程价值。
"""
    )


# ================================================================
# 3. 目录结构：最小结构 vs 完整结构
# ================================================================
def section_layout() -> None:
    print()
    print("=" * 78)
    print("3. 目录结构：一个 Skill 包含「元数据」+「指令」")
    print("=" * 78)
    print(
        """
课案原文要点：
    一个 Skill 包含：
        - 元数据：必须包含名称、描述
        - 指令：技能的详细指示
"""
    )

    print("【最小结构】—— 只有一个必需文件：")
    for line in render_tree({"image-to-mermaid": {"SKILL.md": "唯一必需文件"}}):
        print("  " + line)

    print()
    print("【完整结构（可选）】—— 目录名就是技能名，四个位置各司其职：")
    full = {
        "image-to-mermaid": {
            "SKILL.md": "必需：指令 + 元数据",
            "scripts": ("dir", "可选：执行脚本"),
            "references": ("dir", "可选：文档资料，渐进式披露，\n如下例的 mermaid 语法规范"),
            "assets": ("dir", "可选：资源（模板文件、示例图片、配置项，\n比如默认输出 PNG 还是 SVG）"),
        }
    }
    for line in render_tree(full):
        print("  " + line)

    print(
        """
三个可选目录的分工（记住这个划分，写技能时就不会乱放东西）：
    scripts/     —— 可执行的代码。SKILL.md 里写「运行 scripts/convert.py」，
                    让 agent 用 shell 去跑，而不是把代码读进上下文。
    references/  —— 只在需要时才读的文档（规范、语法手册、检查清单）。
                    这是「渐进式披露」的主战场。
    assets/      —— 不是给人读的，是给脚本/模型用的资源：模板、示例图、
                    默认配置（比如默认输出 PNG 还是 SVG）。
"""
    )

    print("【本仓库里真实的技能目录】—— 由 02_SKILL示例_jxsd.py 生成：")
    print_real_tree(SKILLS_ROOT)


# ================================================================
# 4. SKILL.md：基本模板 + 元数据字段表
# ================================================================
SKILL_MD_TEMPLATE = """---
name: skill-name
description: 说明这个 Skill 的功能以及使用场景
---

# 在这里开始写你的内容，Markdown 格式 —— 智能体会在选择该技能时读取
"""

# 课案《元数据字段说明》表格，逐字段抄下来。
# 本机 deepagents 0.7.13 的 SkillMetadata 还多一个 path（由中间件运行时注入，
# 不需要你手写），一并列在第 8 行，方便和源码对照。
METADATA_FIELDS = [
    ["name", "是", "Skill 名称，最长 64 字符，只允许使用小写字母、数字和 -，\n也不能以 - 开头或结尾"],
    ["description", "是", "简短说明使用场景，最长 1024 字符，不能为空"],
    ["trigger_keywords", "否", "强制推荐关键词，自动触发时使用"],
    ["license", "否", "开源许可证，或指向 Skill 附带的许可证文件"],
    ["compatibility", "否", "描述兼容性，说明与哪些产品系统、平台权限等有关，\n最长 500 字符"],
    ["metadata", "否", "自定义键值对，用于扩展元数据，如作者、版本号等"],
    ["allowed-tools", "否", "允许使用的工具列表，空格分隔的已有工具或新工具功能"],
    ["path", "—", "运行时注入，由中间件写入 SKILL.md 的路径。\n你不写它，但 read_file 时用的就是它"],
]


def section_skill_md() -> None:
    print()
    print("=" * 78)
    print("4. SKILL.md：基本模板 + 元数据字段表")
    print("=" * 78)
    print(
        """
课案原文要点：
    SKILL.md 顶部是一段 YAML front-matter（用两行 `---` 夹住），
    里面写元数据；下面正文是 Markdown 格式的指令，
    智能体会在【选择该技能时】读取它。
"""
    )

    print("【基本模板】（课案原文，可直接复制改名使用）：")
    print("-" * 78)
    print(SKILL_MD_TEMPLATE, end="")
    print("-" * 78)
    print(
        """
两个容易踩的坑：
    1. `---` 必须是文件的第 1 行，且前后不能有空行；中间不能出现单独一行的 `---`
       （中间那行会被当成 front-matter 的结束符，正文就被吃掉了）。
    2. name 必须和技能【所在目录名】完全一致。本机 deepagents 会校验这一点，
       不一致时会在日志里警告 `name 'x' must match directory name 'y'`。
"""
    )

    print()
    print("【元数据字段表】八个字段，逐字段解释：")
    print_table(
        ["字段", "必需", "说明"],
        METADATA_FIELDS,
        [18, 6, 62],
    )
    print(
        """
逐字段补充说明（课案表格之外，写的时候容易含糊的地方）：

    name            技能的唯一标识，也是模型在系统提示词里看到的键。
                    只允许小写字母 / 数字 / 单个连字符，不能有连续 `--`，
                    且必须等于 SKILL.md 所在目录的名字。
    description     最关键的字段 —— 模型【只靠它】决定要不要用这个技能。
                    写法 = 「做什么」+「什么时候用」+「关键触发词」，
                    例如：「Python 代码审查，检查规范性和性能问题；
                    当用户要求 review / 检查 / 优化 Python 代码时使用」。
    trigger_keywords强制推荐关键词，用于自动触发。
                    注意：本机 deepagents 0.7.13 的 SkillsMiddleware **不读**
                    这个字段（它只解析 name/description/license/compatibility/
                    metadata/allowed-tools）。写进去不报错，但当前版本不会生效——
                    所以真正想提高命中率，还得把关键词写进 description。
    license         开源许可证名（MIT / Apache-2.0）或指向附带的 LICENSE 文件。
    compatibility   环境要求。例如「需要 uv、Python 3.11+、可访问外网」，
                    或「仅在 Claude Code 中可用」。
    metadata        自定义键值对。约定俗成放 author / version / updated_at。
    allowed-tools   建议该技能使用哪些工具，空格分隔（也接受 YAML 列表）。
                    本机 deepagents 会把它渲染成
                    `-> Allowed tools: read_file, write_file` 一行提示。
                    官方标注为 experimental（实验特性）。
    path            运行时字段：中间件把 SKILL.md 的真实路径塞进来，
                    模型随后用这个路径调 read_file 拉全文。你不需要手写。

完整 front-matter 长这样（把下面这段抄进 SKILL.md 就是一份合规的元数据）：

    ---
    name: code-review-skill
    description: Python/JavaScript 代码审查，检查规范性与性能问题
    trigger_keywords: 代码审查 review 规范 性能
    license: MIT
    compatibility: 需要 Python 3.10+ 与可读写的项目目录
    metadata:
      author: jxsd
      version: "1.0.0"
    allowed-tools: read_file write_file execute
    ---
"""
    )


# ================================================================
# 5. 小结
# ================================================================
def section_summary() -> None:
    print("=" * 78)
    print("5. 小结")
    print("=" * 78)
    print(
        f"""
    一个 Skill = 一个目录 + 一个 SKILL.md。
    目录名 = name；YAML front-matter = 元数据；正文 = 指令。
    references/ 放「需要时才读」的文档，scripts/ 放「让 agent 去跑」的脚本，
    assets/ 放「给脚本用的」资源。

    加载三步：选择（只读 name+description）→ 学习（命中后读 SKILL.md 全文）
              → 使用（按指示读参考文件 / 跑脚本）。

    本节的目录树与字段表都是打印出来的常量。真正的技能文件由下一节生成：
        {SKILLS_ROOT}
    当前状态：{"已存在" if SKILLS_ROOT.exists() else "尚未生成（请运行 02_SKILL示例_jxsd.py）"}
"""
    )


if __name__ == "__main__":
    section_concept()
    section_principle()
    section_layout()
    section_skill_md()
    section_summary()

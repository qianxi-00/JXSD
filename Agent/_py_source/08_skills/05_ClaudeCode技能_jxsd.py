# -*- coding: utf-8 -*-
"""
Claude Code 技能：安装路径、优先级、自动创建技能与各平台支持
================================================================
课案出处：Agent 课案 → skills → Claude Code / 自动创建技能

本节要讲什么：

    1. 【装在哪】Claude Code 的两个技能加载路径（全局 ~/.claude/skills/ 与
       项目级 <项目>/.claude/skills/），各自的生效范围，以及「怎么选」；
    2. 【怎么装】安装动作的本质 —— 就是把一个技能目录复制过去，
       没有注册表、没有配置文件，「复制完就算装好」；
    3. 【自动创建】课案那四步（拉 awesome-claude-skills → 装 skill-creator →
       开会话 → 用自然语言让 Claude 生成技能），以及 skill-creator 替你做的几件事；
    4. 【平台支持】Claude Code / Cursor / Trae / OpenCode 各自对 Skills 的支持情况，
       以及「目录约定不同、但 SKILL.md 文件格式是同一套」这个关键结论；
    5. 【优先级实测】框架里多来源的覆盖方向（last one wins）与课案「全局①②项目」
       说法的关系 —— 这是本文件唯一一个必须实测才能确定的知识点。

课案这两节讲的是「技能写完之后，怎么装到真实客户端里」。核心是三条：

    1. Claude Code Skills 的加载路径（按优先级排序）：
           全局：~/.claude/skills/         （全局生效）
           项目：<项目目录>/.claude/skills/ （项目专用）
       复制一个技能目录到以上两个目录任何一个，就完成安装了。

    2. 「自动创建技能」四步：
           ① 下载 ComposioHQ/awesome-claude-skills
           ② 把其中的 skill-creator 放进 ~/.claude/skills/
           ③ 打开 PowerShell，输入 Claude
           ④ 说：「帮我创建一个 skill，使用 qwen3-vl 将流程图的图片转 mermaid」
       装好 skill-creator 之后，你可以让 Claude 自己按规范生成一个新的技能目录，
       不用手写 front-matter 和目录结构。

    3. 各平台对 Skills 的支持：
       Claude Code、Cursor、Trae / OpenCode 均支持免费使用 Skills，
       其中 Claude Code 为官方推荐，且功能最丰富。

本节不只打印文字 —— 它会**真的把 02 小节生成的技能复制一份**到项目级安装路径
     Agent/08_skills/.claude/skills/code-review-skill/
并打印安装前后的目录树，让你看到「安装技能」到底改了磁盘上的什么。

★ 关于优先级的实测提醒 ★
    课案把全局排在①、项目排在②。而本机 deepagents 0.7.13 的 SkillsMiddleware
    的规则是「后面的来源覆盖前面同名的技能（last one wins）」，并在系统提示词里
    把最后一个来源标成 (higher priority)。所以用代码挂多个来源时，要把
    【更具体的那个（项目级）放在后面】，项目级才会盖住全局级。
    两者说法不矛盾 —— 课案讲的是「客户端先看哪个目录」，
    框架讲的是「数组里谁在后面」，接线时按框架的规则写就不会错。

运行前置条件：无（纯标准库；只对 ~/.claude 做只读列举，不写任何全局路径）。

运行方式：
    uv run Agent/08_skills/05_ClaudeCode技能_jxsd.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，防中文/emoji 报错

import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 四个路径常量，本节全部动作都围着它们转：
#   SKILLS_ROOT   = 02 小节生成技能的地方，也是「技能的父目录」
#   SOURCE_SKILL  = 要安装的那个技能（一个目录，含 SKILL.md + references/）
#   PROJECT_CLAUDE_DIR / INSTALLED_SKILL = 项目级安装【目的地】
#   GLOBAL_CLAUDE_SKILLS = 全局路径；本节只对它做【只读列举】，绝不写入
SKILLS_ROOT = HERE / "skills"                 # 技能的父目录（02 小节生成）
SOURCE_SKILL = SKILLS_ROOT / "code-review-skill"
PROJECT_CLAUDE_DIR = HERE / ".claude" / "skills"   # 项目级安装路径
INSTALLED_SKILL = PROJECT_CLAUDE_DIR / "code-review-skill"
# expanduser("~") 展开成 C:\Users\<用户名>；用 Path 拼接而不是裸字符串拼，
# 这样反斜杠/正斜杠的差异交给 pathlib 处理，脚本在 Windows 与 Linux 上都能跑。
GLOBAL_CLAUDE_SKILLS = Path(os.path.expanduser("~")) / ".claude" / "skills"


# ================================================================
# 目录树打印（安装前后各一次，用来对比）
# ================================================================
def print_tree(root: Path, title: str) -> None:
    print(f"  【{title}】{root}")
    if not root.exists():
        print("    （目录不存在）")
        return

    def walk(path: Path, prefix: str = "") -> None:
        children = sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name))
        for index, child in enumerate(children):
            # 跳过 __pycache__：里面的 .pyc 是二进制，读不出「行数」
            if child.is_dir() and child.name == "__pycache__":
                continue
            is_last = index == len(children) - 1
            branch = "└── " if is_last else "├── "
            if child.is_dir():
                print(f"    {prefix}{branch}{child.name}/")
                walk(child, prefix + ("    " if is_last else "│   "))
            else:
                # 显示行数而不是字节数：技能文件是 markdown，行数更有意义。
                # 二进制/非 UTF-8 文件（图片、.pyc）读不了，退化成显示字节数。
                try:
                    lines = len(child.read_text(encoding="utf-8").splitlines())
                    note = f"{lines} 行"
                except (UnicodeDecodeError, OSError):
                    note = f"{child.stat().st_size} 字节（二进制）"
                print(f"    {prefix}{branch}{child.name}    # {note}")

    walk(root)


# ================================================================
# 1. Claude Code 的加载路径与优先级
# ================================================================
def section_load_paths() -> None:
    print("=" * 78)
    print("1. Claude Code Skills 的加载路径（按优先级排序）")
    print("=" * 78)
    print(
        f"""
课案原文：
    1. 全局：`~/.claude/skills/`（全局生效）
    2. 项目：项目目录下的 `.claude/skills/`（项目专用）
    复制 image-to-mermaid 到以上两个目录任何一个，就完成安装了。

放到本机上，两个路径分别是：

    全局：{GLOBAL_CLAUDE_SKILLS}
    项目：{PROJECT_CLAUDE_DIR}
          （也就是本文件所在的 Agent/08_skills/ 下面）

两个位置怎么选：

    | 位置 | 生效范围 | 适合放什么 |
    |---|---|---|
    | ~/.claude/skills/      | 本机所有项目 | 通用技能：文档处理、代码审查、爬虫… |
    | <项目>/.claude/skills/ | 只有这个项目 | 项目专属：本仓库的接口规范、数据字典… |

安装动作本质上就是「把一个技能目录复制过去」，没有任何注册步骤：

    # PowerShell（本机 shell）
    Copy-Item -Recurse -Force <技能目录> $env:USERPROFILE\\.claude\\skills\\
    # bash（远程 Linux 服务器）
    cp -r <技能目录> ~/.claude/skills/
"""
    )

    # 只读列举全局目录 —— 本节不往全局目录写任何东西
    print("  本机全局路径的当前状态（只读列举，不做修改）：")
    if GLOBAL_CLAUDE_SKILLS.exists():
        # Windows 上还有一种「目录联接（junction）」：is_symlink() 返回 False，
        # 需要用 os.path.isjunction() 才认得出。本机 ~/.claude/skills 就是联接。
        is_junction = getattr(os.path, "isjunction", lambda _p: False)(GLOBAL_CLAUDE_SKILLS)
        if GLOBAL_CLAUDE_SKILLS.is_symlink():
            kind = "符号链接"
        elif is_junction:
            kind = "目录联接（junction）"
        else:
            kind = "真实目录"
        target = ""
        if GLOBAL_CLAUDE_SKILLS.is_symlink() or is_junction:
            try:
                target = f"  → {os.readlink(GLOBAL_CLAUDE_SKILLS)}"
            except OSError:
                target = "  → （读取链接目标失败）"
        names = sorted(p.name for p in GLOBAL_CLAUDE_SKILLS.iterdir() if p.is_dir())
        print(f"    类型：{kind}{target}")
        print(f"    已安装全局技能 {len(names)} 个：{names}")
    else:
        print("    （本机还没有 ~/.claude/skills/ 目录；要装全局技能就先建它）")

    print(
        """
  ★ 实测提醒：多个来源时的覆盖方向
      本机 deepagents 0.7.13 的 SkillsMiddleware 规则是「后面的来源覆盖前面同名的
      技能」（last one wins），并在系统提示词里把【最后一个】来源标成
      (higher priority)。所以代码里挂多个来源时，把项目级放【后面】：

          skills=["/skills/", "/.claude/skills/"]
                  ↑ 基础/全局            ↑ 项目级，优先级更高

      本节最后的验证环节会把这条规则跑出来给你看。
"""
    )


# ================================================================
# 2. 自动创建技能：四步 + skill-creator
# ================================================================
def section_auto_create() -> None:
    print()
    print("=" * 78)
    print("2. 自动创建技能：不用手写 front-matter")
    print("=" * 78)
    print(
        """
课案原文四步：
    1. 下载 ComposioHQ/awesome-claude-skills
    2. 将当中的 skill-creator 放入 `~/.claude/skills/`
    3. 打开 PowerShell，输入 Claude
    4. 说：帮我创建一个 skill，使用 qwen3-vl 将流程图的图片转 mermaid

逐条展开成可以照抄的操作：

    # ① 拉取技能合集（ComposioHQ 维护的 awesome 列表，里面都是可用的现成技能）
    git clone https://github.com/ComposioHQ/awesome-claude-skills.git

    # ② 只把 skill-creator 这一个装进全局技能目录
    #    Windows / PowerShell
    Copy-Item -Recurse -Force .\\awesome-claude-skills\\skill-creator `
                $env:USERPROFILE\\.claude\\skills\\
    #    Linux / macOS
    cp -r ./awesome-claude-skills/skill-creator ~/.claude/skills/

    # ③ 在项目目录下启动 Claude Code
    claude

    # ④ 直接用自然语言提需求，skill-creator 会替你生成目录 + SKILL.md
    帮我创建一个 skill，使用 qwen3-vl 将流程图的图片转 mermaid

skill-creator 帮你做的事情，其实就是本仓库 02 小节手工做的那几件：
    建目录 → 起名字 → 写 front-matter（name/description）→ 写正文指令 →
    必要时补 scripts/ references/ assets/ → 自检 name 是否等于目录名。

一句话取舍：
    一次性、临时用的提示词 → 直接写在对话里；
    反复要用、要给别人用、要版本管理的 → 做成技能。

课案里那个 image-to-mermaid 技能（deepagents 小节也在用它）就是
「用 qwen3-vl 把流程图图片转成 mermaid」这个需求的产物：
    image-to-mermaid/
    ├── SKILL.md          # 告诉 agent「不要直接 read_file 读图片，去跑脚本」
    ├── scripts/          # 调 qwen3-vl 的转换脚本
    └── references/       # mermaid 语法规范（需要时才读）
课案给的分发包：https://dya123.oss-cn-beijing.aliyuncs.com/haydon-image-gen.zip
"""
    )


# ================================================================
# 3. 各平台对 Skills 的支持情况
# ================================================================
def section_platforms() -> None:
    print("=" * 78)
    print("3. 各平台对 Skills 的支持情况")
    print("=" * 78)
    print(
        """
课案原文：
    Claude Code、Cursor、Trae / OpenCode 均支持免费使用 Skills，
    其中 Claude Code 为官方推荐且功能最丰富。
"""
    )

    rows = [
        # 「定位」和「Skills 支持情况」两列都是从课案原文整理出来的：
        # 课案只说了「四家都支持，Claude Code 是官方推荐且功能最丰富」，
        # 这里把那句话拆成每家的具体形态，便于横向对照。
        ["Claude Code", "Anthropic 官方 CLI", "官方推荐，功能最丰富：\n自动触发、渐进式披露、可执行脚本"],
        ["Cursor", "AI 代码编辑器", "在编辑器里识别 SKILL.md，\n与补全/对话结合"],
        ["Trae", "AI IDE（字节）", "支持 Skills，免费使用"],
        ["OpenCode", "开源终端 AI 编码工具", "支持 Skills，免费使用"],
    ]
    headers = ["平台", "定位", "Skills 支持情况"]
    widths = [14, 26, 44]
    display = lambda s: sum(2 if ord(c) > 0x2E7F else 1 for c in s)  # noqa: E731
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    print(line)
    print("| " + " | ".join(h + " " * (w - display(h)) for h, w in zip(headers, widths)) + " |")
    print(line)
    for row in rows:
        cells = [c.split("\n") for c in row]
        for i in range(max(len(c) for c in cells)):
            parts = [
                (c[i] if i < len(c) else "") + " " * (w - display(c[i] if i < len(c) else ""))
                for c, w in zip(cells, widths)
            ]
            print("| " + " | ".join(parts) + " |")
        print(line)

    print(
        """
  ⚠ 一个容易混淆的点：不同客户端读的目录不一定一样。
        Claude Code 认 ~/.claude/skills/ 与 <项目>/.claude/skills/；
        Cursor / Trae / OpenCode 各有自己的约定目录。
    但 SKILL.md 的【文件格式】是同一套（Agent Skills 规范：
    YAML front-matter + Markdown 正文 + 可选的 scripts/references/assets），
    所以同一个技能目录复制到不同客户端，通常都能直接用。

  资源参考（课案给的几个入口）：
    官方技能示例、Skills 市场、Skills 市场 2、开源技能项目 ——
    找现成技能优先看这四类；自己写则参考本仓库 01/02 两节。
"""
    )


# ================================================================
# 4. 真实安装演示：把技能复制到项目级 .claude/skills/
# ================================================================
def section_install() -> bool:
    print()
    print("=" * 78)
    print("4. 真实安装演示：复制到项目级 .claude/skills/")
    print("=" * 78)

    if not SOURCE_SKILL.exists():
        print(f"  ！源技能不存在：{SOURCE_SKILL}")
        print("  请先运行：uv run Agent/08_skills/02_SKILL示例_jxsd.py")
        return False

    print("  ── 安装【前】──")
    print_tree(HERE, "08_skills 项目目录")
    print()

    # 幂等：目标已存在就先删掉再复制，避免 copytree 报 FileExistsError，
    # 也避免新旧文件混在一起（技能是整体替换的，不增量合并）。
    if INSTALLED_SKILL.exists():
        shutil.rmtree(INSTALLED_SKILL)
        print(f"  目标已存在，先清理：{INSTALLED_SKILL}")
    PROJECT_CLAUDE_DIR.mkdir(parents=True, exist_ok=True)

    # copytree：把整个技能目录（含 references/）原样搬过去
    shutil.copytree(SOURCE_SKILL, INSTALLED_SKILL)
    print(f"  已复制 {SOURCE_SKILL.name}/ → {INSTALLED_SKILL.relative_to(HERE.parent.parent)}")
    print()
    print("  ⚠ 注意：本演示只装【项目级】路径（08_skills/.claude/skills/）。")
    print(f"    全局路径 {GLOBAL_CLAUDE_SKILLS} 属于用户级配置，本文件不写入。")
    print("    要装全局，手动执行：")
    print("      Copy-Item -Recurse -Force Agent\\08_skills\\skills\\code-review-skill "
          "$env:USERPROFILE\\.claude\\skills\\")
    print()

    print("  ── 安装【后】──")
    print_tree(PROJECT_CLAUDE_DIR.parent, ".claude 目录（项目级安装位置）")
    return True


# ================================================================
# 5. 验证：把两个来源都挂上，看标签与覆盖方向
# ================================================================
def section_verify() -> None:
    print()
    print("=" * 78)
    print("5. 验证：技能是否真的可被发现，以及多来源的覆盖方向")
    print("=" * 78)

    from deepagents.backends.local_shell import LocalShellBackend
    from deepagents.middleware.skills import SkillsMiddleware

    backend = LocalShellBackend(root_dir=str(HERE))

    for sources in (["/skills/"], ["/.claude/skills/"], ["/skills/", "/.claude/skills/"]):
        middleware = SkillsMiddleware(backend=backend, sources=sources)
        update = middleware.before_agent({}, None, None)  # type: ignore[arg-type]
        found = [s["name"] for s in (update["skills_metadata"] if update else [])]
        print(f"  sources={sources}")
        print(f"    来源标签 = {middleware.source_labels}   ← 渲染成系统提示词里的 **xxx Skills**")
        print(f"    发现技能 = {found}")

    print(
        """
  结论：
    1. 两个目录都能被独立识别，复制过去就算安装完成 —— 没有注册表、没有配置文件；
    2. 一起挂上时，两个来源里同名的 code-review-skill 只会留一个 ——
       后一个来源（项目级 .claude/skills/）覆盖前一个（skills/），
       系统提示词里也会把最后一个标成 (higher priority)；
    3. 所以「改哪个目录生效」这件事，在框架层面取决于数组顺序 ——
       把想生效的那个放最后。
"""
    )


# ================================================================
# 6. 小结
# ================================================================
def section_summary() -> None:
    print("=" * 78)
    print("6. 小结：装一个技能要做的三件事")
    print("=" * 78)
    print(
        f"""
    1. 准备好一个合规的技能目录（SKILL.md 的 name 必须等于目录名）；
    2. 复制到加载路径之一：
           全局  {GLOBAL_CLAUDE_SKILLS}
           项目  {PROJECT_CLAUDE_DIR}
    3. 重启客户端（Claude Code / Cursor / Trae / OpenCode），
       技能清单会在启动时被扫描一次。

    本节在磁盘上留下的东西（受版本管理时建议把 .claude/skills/ 一起提交，
    这样团队里每个人 clone 下来就自带项目技能）：
       {INSTALLED_SKILL}

    全局路径本次【未做任何写入】。
"""
    )


if __name__ == "__main__":
    section_load_paths()
    section_auto_create()
    section_platforms()
    if section_install():
        section_verify()
    section_summary()

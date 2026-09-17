# -*- coding: utf-8 -*-
"""
deepagents skills：用 create_deep_agent(skills=[...]) 加载技能
================================================================
课案出处：Agent 课案 → skills → deepagents skills

本节要讲什么：

    1. 【参数】create_deep_agent(skills=[...]) 这个参数到底怎么写 ——
       为什么它要指向技能的【父目录】而不是某个技能目录 / SKILL.md 本身，
       以及 skills=["/"] 与 skills=["/skills/"] 的区别；
    2. 【路径语义】skills 里的路径是「backend 的虚拟路径」，不是宿主机绝对路径，
       它相对于 LocalShellBackend(root_dir=...) 那一层；
    3. 【框架内部】传了 skills 之后 create_deep_agent 会自动挂一个 SkillsMiddleware，
       以及「选择」阶段究竟往 system prompt 里塞了什么（本节会把这段文字原样打印）；
    4. 【实测】真的创建 agent 并跑一次，打印工具调用轨迹，
       让你看到模型确实先读 SKILL.md、再按正文的指示读 python_rules.md；
    5. 【降级与差异】课案的 image-to-mermaid 例子本机跑不了（缺视觉脚本），
       本文件改用 02 节生成的 code-review-skill，并说明为什么等价。

课案这一节给的例子是「把图片转成 mermaid」的技能，代码只有十几行：

    model = ChatOpenAI(model=..., api_key=..., base_url=...)
    agent = create_deep_agent(
        model=model,
        backend=LocalShellBackend(root_dir=".", inherit_env=True),
        skills=["/"],  # 扫描根目录 -> 发现 /image-to-mermaid/SKILL.md
    )

课案原文里 `skills` 那行的注释被截断了（只留下「# 注意skills 指向技能的【父目」），
本节把它补全成完整说明，并逐行解释这个参数的三个坑：

    ★ skills 指向技能的【父目录】★
      —— 不是某个具体技能目录（不要写 "/image-to-mermaid/"），
         更不是 SKILL.md 文件本身；写的是【装着若干技能目录的那一层】。
      —— 传 ["/"] 的含义是「扫描 backend 根目录下的所有一级子目录」，
         于是 /image-to-mermaid/SKILL.md 会被发现。
      —— 路径是 POSIX 风格（正斜杠）、相对于 backend 的 root_dir，
         不是宿主机绝对路径 —— 这一点和 tools 里的文件路径规则一致。

本节的改写（把示例指向本仓库的技能目录）：
    backend 的 root_dir 设成 Agent/08_skills，
    skills 传 ["/skills/"]，
    于是 08_skills/skills/code-review-skill/SKILL.md 会被发现。

课案那段 image-to-mermaid 的完整流程需要一张真实图片 + 一个调 qwen-vl 的脚本，
本机没有那份脚本，所以保留为「课案原文」讲解；真正跑起来的是 02 小节生成的
code-review-skill（纯文本技能，不需要外部视觉模型）。

运行前置条件：
    先跑过 02_SKILL示例_jxsd.py（生成 skills/code-review-skill/），
    以及可用的 settings.api_key（实测可用，模型 grok-4.6）。

运行方式：
    uv run Agent/08_skills/03_deepagents技能_jxsd.py
"""

import sys

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，防中文/emoji 报错

import json
from pathlib import Path

from langchain.chat_models import init_chat_model

from config import settings

HERE = Path(__file__).resolve().parent
# 技能的【父目录】：这个常量就是 skills= 参数要指向的那一层。
# 换成「具体技能目录」或「SKILL.md 文件本身」都会导致发现 0 个技能 ——
# 中间件是「ls 这一层 → 对每个子目录取里面的 SKILL.md」，它不认别的形状。
SKILLS_ROOT = HERE / "skills"          # 技能的【父目录】
SKILL_DIR = SKILLS_ROOT / "code-review-skill"

# 课案里用的是 ChatOpenAI(...)；本项目规范统一用 init_chat_model 指定 provider，
# 参数同样来自 settings，效果等价。
llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


# ================================================================
# 课案原文（含被截断的注释），原样保留，便于对照
# ================================================================
COURSE_CODE = '''# 例子为把 图片转成mermaid skill
from langchain_openai import ChatOpenAI
from config import setting
from deepagents import create_deep_agent
from deepagents.backends.local_shell import LocalShellBackend

# 调度模型用文本大模型（工具调用能力强）；
# 视觉识别由 skill 里的脚本自己调 qwen-vl 完成，agent 本身不需要是 VL 模型
model = ChatOpenAI(model=setting.MODEL_NAME, api_key=setting.API_KEY, base_url=setting.BASE_URL)

agent = create_deep_agent(
    model=model,
    backend=LocalShellBackend(root_dir=".", inherit_env=True),
    skills=["/"],  # 扫描根目录 -> 发现 /image-to-mermaid/SKILL.md
)
result = agent.invoke({"messages": [{"role": "user", "content": (
    "请使用 image-to-mermaid 技能，把 img.png 转换成 Mermaid 架构图代码。"
    "注意：不要直接用 read_file 读取图片文件，而是按 SKILL.md 的说明运行其中的转换脚本。"
)}]})
print(result["messages"][-1].content)
'''

# 课案原文被截断的那行注释，补全后应该是这样（本节的第一个知识点）
COMPLETED_COMMENT = (
    "# 注意 skills 指向技能的【父目录】：\n"
    "#   skills=[\"/\"]                 → 扫描根目录下的所有一级子目录，发现 /image-to-mermaid/SKILL.md\n"
    "#   skills=[\"/skills/\"]          → 扫描 /skills/ 下的所有一级子目录\n"
    "#   skills=[\"/skills/a/\", \"/skills/b/\"] → 多个来源，后面的覆盖前面同名的技能\n"
    "# 不能写成某个技能目录（\"/image-to-mermaid/\"）或 SKILL.md 文件本身。"
)


# ================================================================
# 1. 课案原文 + 被截断注释的补全
# ================================================================
def section_course_code() -> None:
    print("=" * 78)
    print("1. 课案原文（注意最后 `skills` 那行的注释是被截断的）")
    print("=" * 78)
    print(COURSE_CODE)
    print("-" * 78)
    print("课案原文里 `# 注意skills 指向技能的【父目` 这句被截断了，补全后是：")
    print()
    for line in COMPLETED_COMMENT.splitlines():
        print("    " + line)
    print()
    print(
        """
再补三点课案没展开、但写代码一定会踩的细节：

    (1) 路径是「backend 的虚拟路径」，不是宿主机路径。
        LocalShellBackend(root_dir=".") 时，虚拟 `/` = 当前工作目录；
        本文件把 root_dir 设成脚本所在目录 Agent/08_skills，
        于是虚拟 `/skills/` = Agent/08_skills/skills/。换 root_dir 就要换 skills 里的路径。

    (2) LocalShellBackend 的 virtual_mode 默认是 True，
        意味着 agent 只能用虚拟路径；配合 inherit_env=True 还会把本进程的
        环境变量传给子进程（技能里的脚本要用 API Key 时靠它）。

    (3) skills 参数一旦传了，create_deep_agent 会自动往中间件栈里塞一个
        SkillsMiddleware（见 deepagents/graph.py 第 863 行），
        你不需要自己 new 它。
"""
    )


# ================================================================
# 2. 「技能发现」预检：看看模型到底会看到什么
# ================================================================
def section_discovery(backend, sources: list[str]) -> tuple[object, list[dict]]:
    """在不调大模型的前提下，先把技能发现结果打印出来。

    这一步同时充当**降级路径**：如果后面调模型失败（网络/额度/需要视觉模型），
    至少还能看到「技能确实被发现了、系统提示词里长什么样」。

    做法就是 new 一个 SkillsMiddleware 然后手动调 before_agent ——
    这正是 create_deep_agent 内部会做的事，只是我们提前跑一遍看结果。
    """
    print()
    print("=" * 78)
    print("2. 技能发现预检：模型在实际对话前会拿到什么")
    print("=" * 78)

    from deepagents.middleware.skills import SkillsMiddleware

    middleware = SkillsMiddleware(backend=backend, sources=sources)

    # before_agent(state, runtime, config) 只用到 state；传空 state 就会触发一次加载，
    # 因为它的逻辑是「state 里没有 skills_metadata 才加载」。
    update = middleware.before_agent({}, None, None)  # type: ignore[arg-type]
    skills_meta = list(update["skills_metadata"]) if update else []

    print(f"  技能来源（skills 参数）：{sources}")
    print(f"  来源标签（会渲染成系统提示词的小标题）：{middleware.source_labels}")
    print(f"  发现技能数：{len(skills_meta)}")
    print()
    print("  ── 以下文字会【原样】拼进 system prompt（这就是『选择』阶段的全部开销）──")
    print("  " + "-" * 74)
    for line in middleware._format_skills_locations().splitlines():  # noqa: SLF001
        print("  | " + line)
    print("  |")
    for line in middleware._format_skills_list(skills_meta).splitlines():  # noqa: SLF001
        print("  | " + line)
    print("  " + "-" * 74)
    print(
        """
  注意看：常驻系统提示词的只有 name + description + 路径，
  references/ 里那两份几十行的规范**一个字都没进去**。
  模型要读，得自己发起 read_file —— 这就是渐进式披露在工程上的落点。
"""
    )
    return middleware, skills_meta


# ================================================================
# 3. 真的创建 agent 并跑一次
# ================================================================
# 故意写了 5 处违反 python_rules.md 的代码，让技能有活干
SAMPLE_CODE = '''def add_item(item, bucket=[]):
    """把 item 放进 bucket。"""
    bucket.append(item)
    return bucket


def total_price(orders):
    result = ""
    for o in orders:
        result += str(o["price"])
    return result


def load_config(name):
    f = open(name)
    data = f.read()
    return eval(data)
'''

USER_PROMPT = (
    "请使用 code-review-skill 技能审查下面这段 Python 代码。\n"
    "注意：不要凭印象评论，先按 SKILL.md 的说明去读对应的规范文件，再逐条比对给结论。\n\n"
    "```python\n" + SAMPLE_CODE + "```\n\n"
    "只输出审查结论，不要写任何文件。"
)


def section_run_agent(backend) -> None:
    print()
    print("=" * 78)
    print("3. 创建 agent 并真的跑一次（模型：" + settings.model_name + "）")
    print("=" * 78)

    from deepagents import create_deep_agent

    # ---------- 创建 agent（课案同一套参数）----------
    # root_dir：课案写的是 "."（当前工作目录）。这里改用脚本所在目录，
    #           好处是不管从哪个目录启动，虚拟路径 "/skills/" 都稳定落在
    #           Agent/08_skills/skills/。
    # inherit_env=True：技能里的脚本要用 .env 里的 API Key 时，靠它把环境变量传下去。
    agent = create_deep_agent(
        model=llm,
        backend=backend,
        skills=["/skills/"],  # ← 指向技能的【父目录】
        system_prompt="你是代码审查助手。优先使用可用的技能完成任务。",
    )

    print("  agent 创建成功，开始 invoke ...")
    print()

    result = agent.invoke(
        {"messages": [{"role": "user", "content": USER_PROMPT}]},
        config={"recursion_limit": 60},
    )

    # ---------- 打印工具调用轨迹：这是「渐进式披露真的发生了」的证据 ----------
    print("  ── 工具调用轨迹（看它有没有先读 SKILL.md、再读 python_rules.md）──")
    step = 0
    for message in result["messages"]:
        kind = type(message).__name__
        if kind == "AIMessage":
            for call in getattr(message, "tool_calls", None) or []:
                step += 1
                args = json.dumps(call.get("args", {}), ensure_ascii=False)
                print(f"    [{step}] 模型 → 工具 {call.get('name')}  参数 {args[:150]}")
        elif kind == "ToolMessage":
            text = str(message.content).replace("\n", " ")
            print(f"        ↳ 工具返回（{getattr(message, 'name', '?')}）：{text[:110]} ...")
    if step == 0:
        print("    （本次没有工具调用 —— 模型直接作答了。可以重跑一次，或看下面的正文输出）")

    print()
    print("  ── 最终回答 ──")
    print("  " + "-" * 74)
    final = result["messages"][-1].content
    for line in str(final).splitlines():
        print("  | " + line)
    print("  " + "-" * 74)


# ================================================================
# 4. 降级路径：调不动模型时，至少把技能发现结果和用法说清楚
# ================================================================
def section_degraded(reason: str) -> None:
    print()
    print("=" * 78)
    print("4. 降级演示（调用大模型这一步没有成功）")
    print("=" * 78)
    print(f"  原因：{reason}")
    print(
        """
  这不影响本节的核心结论 —— 技能【发现】完全在本地完成，不需要模型：

    1. create_deep_agent(skills=[...]) 会在启动时创建 SkillsMiddleware；
    2. 中间件 ls 技能父目录 → 对每个子目录取 SKILL.md → 只解析 YAML front-matter；
    3. 把 name / description / license / compatibility / allowed-tools
       拼成一小段文字塞进 system prompt；
    4. 任务匹配时，模型自己 read_file 拉全文 —— 这一步才需要模型。

  上面第 2 节的「技能发现预检」打印的就是第 3 步的结果，
  即使模型不可用，这份结果也是真实、可验证的。

  课案里 image-to-mermaid 那个技能还需要一张真实图片 + 一份调 qwen-vl 的脚本，
  本仓库没有那份脚本，所以那个例子只作为原文保留，不做实跑。
"""
    )


# ================================================================
# __main__
# ================================================================
if __name__ == "__main__":
    from deepagents.backends.local_shell import LocalShellBackend

    section_course_code()

    # ---------- 前置检查：技能目录必须先存在 ----------
    if not (SKILL_DIR / "SKILL.md").exists():
        print()
        print("!" * 78)
        print(f"没有找到技能文件：{SKILL_DIR / 'SKILL.md'}")
        print("请先运行：uv run Agent/08_skills/02_SKILL示例_jxsd.py")
        print("!" * 78)
        sys.exit(0)

    # ---------- backend：与课案同构，root_dir 改成本仓库的技能父目录 ----------
    backend = LocalShellBackend(
        root_dir=str(HERE),      # 课案是 "."；这里写死脚本目录，保证路径稳定
        inherit_env=True,        # 把本进程环境变量传给技能脚本（脚本要用 API Key 时靠它）
    )

    # 第 2 节：不需要模型的技能发现预检（同时也是降级输出）
    section_discovery(backend, ["/skills/"])

    # 第 3 节：真跑。失败就走第 4 节的降级说明，绝不抛 traceback。
    try:
        section_run_agent(backend)
    except Exception as exc:  # noqa: BLE001 —— 教学脚本要保证「永远打印提示而不是 traceback」
        section_degraded(f"{type(exc).__name__}: {str(exc)[:200]}")
        sys.exit(0)

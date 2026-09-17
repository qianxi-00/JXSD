# -*- coding: utf-8 -*-
r"""
LangChain 官方补充篇：自己组装 harness（非课案内容，故不带 _jxsd 后缀）
================================================================
来源与定位：
    本文件对照 LangChain **官方文档** /oss/python/langchain/deep-agent-from-scratch.mdx
    （原文标题是 "Build a data analysis agent from scratch"），补缺口表 **第 12 项**。

官方这篇的核心主张：
    「`create_agent` 与 `create_deep_agent` 都能细粒度控制工具与记忆；区别只是
      Deep Agents **默认就把常用能力装好了**（规划、文件系统、子代理…）。
      如果默认 harness 不合适，就从 create_agent 起步，一件一件把 harness 装出来，
      这样你能看清每一件各自解决了什么问题，只留自己需要的。」

官方把组装拆成五步，每一步针对一个具体痛点：

    步骤                    不加会怎样                        加了什么
    ----------------------  --------------------------------  ----------------------------
    0 最小 agent            只有"模型 + 循环"                 （基线，没有 harness）
    1 沙箱 + 文件系统         agent 读不到 CSV、跑不了 Python   隔离后端 + 文件/执行工具
    2 摘要压缩               长会话撞上下文上限                自动压缩历史
    3 技能（Skills）         领域规则把 system prompt 撑爆      按需加载（渐进披露）
    4 子代理                 图表迭代挤占主线程                隔离的 worker + 并行委派

**本文件与官方教程的两处不同**（都是为了在本机可跑）：
    A. 官方用 `LangSmithSandbox`（要 `LANGSMITH_API_KEY` 走 LangSmith 起沙箱）——
       本仓库不使用 LangSmith，改用 **FilesystemBackend + 临时目录**：
       一样能读写真实文件，只是"隔离"程度不如沙箱（这点在注释里标明）；
    B. 官方主题是"数据分析 agent"，本文件把数据换成更小的示例，把篇幅留给
       **每件中间件各自解决了什么** —— 这才是这篇文档的教学价值。

与课案 03_deepagents 章的关系：
    课案讲的是「`create_deep_agent` 开箱有哪些能力、怎么用」；
    本文件讲的是「这些能力分别由哪个中间件提供、为什么需要它、自己怎么装」。
    两者对照看，正好把"黑盒"变成"白盒"。

运行方式（项目根目录下，需真实模型）：
    uv run Agent/02_langchain/22_自己组装harness_官方补充.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import tempfile
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.chat_models import init_chat_model

from config import settings

model = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


def text_of(message) -> str:
    """取消息文本：content 可能是字符串，也可能是内容块列表（LangChain 1.x）。"""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def tool_names_of(middlewares) -> list[str]:
    """列出这组中间件**贡献**了哪些工具（中间件自己就带着工具清单）。

    实测：编译后的 agent 图里 tools 节点是 PregelNode，拿不到工具名；
    而每个中间件实例上的 `.tools` 就是它注册的工具（FilesystemMiddleware
    实测给出 ls / read_file / write_file / edit_file / delete / glob / grep / execute）。
    """
    names: list[str] = []
    for middleware in middlewares:
        for item in getattr(middleware, "tools", None) or []:
            names.append(getattr(item, "name", str(item)))
    return sorted(set(names))


def ask(agent, question: str, config: dict | None = None) -> tuple[str, list[str]]:
    """跑一次 agent，返回（最终回答, 本次调用的工具名序列）。"""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]}, config or {})
    called: list[str] = []
    for message in result["messages"]:
        for call in getattr(message, "tool_calls", None) or []:
            called.append(call["name"])
    return text_of(result["messages"][-1]), called


# ================================================================
# 第 0 步：最小 agent —— 只有模型 + 循环
# ================================================================
# 基线：create_agent(model, tools=[])。它已经会对话、会调用你给的工具，
# 但**没有任何 harness**：没有文件系统、没有摘要、没有子代理。
# 先让它去读一个文件，看它怎么"无能为力" —— 这就是后面每一步要解决的痛点。
def step_0_minimal(sales_path: str) -> dict:
    print("=" * 70)
    print("第 0 步：最小 agent（模型 + 循环，无 harness）")
    print("=" * 70)

    agent = create_agent(model=model, tools=[])
    answer, called = ask(agent, f"读一下 {sales_path} 这个文件，告诉我一共几行数据。")
    print(f"  可用工具：{tool_names_of([]) or '（无）'}")
    print(f"  回答：{answer[:120]}")
    print(
        "  ↑ 它没法读文件 —— 不是模型不行，是**没给它这个能力**。\n"
        "    （本机模型在这一步还可能把「我要调 read_file」写进正文而不是真的调用工具，\n"
        "      这是模型行为不是代码问题；给它真工具之后就没这个现象了。）\n"
        "    官方的组装教程就是从这里出发，一件件把能力装上去。"
    )
    return {"middleware": []}


# ================================================================
# 第 1 步：+ 文件系统（FilesystemMiddleware + 后端）
# ================================================================
# 官方的第二个组件：隔离后端 + 文件工具。
# 本文件用 `FilesystemBackend(root_dir=临时目录, virtual_mode=True)`：
#   · virtual_mode=True → agent 看到的是以 "/" 开头的**虚拟路径**，映射到 root_dir；
#   · 换成 StateBackend 则文件存在图状态里（课案 03_deepagents/03 讲过后端选型）；
#   · 官方教程用的是 LangSmithSandbox（真隔离，但要 LANGSMITH_API_KEY），
#     本仓库不用 LangSmith，所以这里的"隔离"仅靠目录隔离 —— 演示够用，生产按需换。
def step_1_filesystem(sales_path: str) -> dict:
    print("\n" + "=" * 70)
    print("第 1 步：+ 文件系统（FilesystemMiddleware + FilesystemBackend）")
    print("=" * 70)

    from deepagents.backends import FilesystemBackend
    from deepagents.middleware.filesystem import FilesystemMiddleware

    backend = FilesystemBackend(root_dir=WORKDIR, virtual_mode=True)
    middleware = [FilesystemMiddleware(backend=backend, tools="all")]
    agent = create_agent(model=model, tools=[], middleware=middleware)

    answer, called = ask(agent, f"读一下 {sales_path} 这个文件，告诉我一共几行数据。")
    print(f"  可用工具：{tool_names_of(middleware)}")
    print(f"  本次调用的工具：{called}")
    print(f"  回答：{answer[:140]}")
    print(
        "  ↑ 这一步只加了一个中间件，agent 立刻多了 ls / read_file / write_file / grep 等工具。\n"
        "    注意工具列表是**中间件带进来的**，不是你一个个注册的 —— 这就是 harness 的含义。"
    )
    return {"middleware": middleware, "backend": backend}


# ================================================================
# 第 2 步：+ 摘要压缩（SummarizationMiddleware）
# ================================================================
# 痛点是长会话：历史越堆越多，早晚撞上下文窗口。
# SummarizationMiddleware 在**触发条件命中**时，把旧历史交给模型压成一份摘要，
# 用摘要替换原文 —— 之后每轮请求都轻装上阵。
#   参数（实测签名）：trigger=("messages", N) / ("tokens", N) / ("fraction", 0.8)
#                    keep=("messages", M) 控制压缩后保留最近多少条
# 本 Demo 直接**喂一段长历史**（不必真聊 30 轮），一次 invoke 就能看到压缩发生。
def step_2_summarization(built: dict) -> dict:
    print("\n" + "=" * 70)
    print("第 2 步：+ 摘要压缩（SummarizationMiddleware）")
    print("=" * 70)

    middleware = list(built["middleware"]) + [
        SummarizationMiddleware(
            model=model,
            trigger=("messages", 12),   # 历史超过 12 条就压缩
            keep=("messages", 4),       # 压缩后保留最近 4 条
        )
    ]
    agent = create_agent(model=model, tools=[], middleware=middleware)

    # 伪造一段 16 条的长历史（8 轮一问一答），主题固定便于观察摘要是否保留要点
    history = []
    for index in range(1, 9):
        history.append({"role": "user", "content": f"第 {index} 个问题：请记住重点 {index}。"})
        history.append({"role": "assistant", "content": f"好的，我记住了重点 {index}。"})
    history.append({"role": "user", "content": "我一共让你记住了几个重点？"})
    print(f"  投喂历史消息数：{len(history)}（触发阈值 12）")

    result = agent.invoke({"messages": history})
    final_messages = result["messages"]
    print(f"  运行结束后消息数：{len(final_messages)} ← 比投喂的少，说明历史被压缩替换了")
    print(f"  回答：{text_of(final_messages[-1])[:140]}")
    print(
        "  ↑ 关键点：**压缩是自动的**，由中间件在合适时机做，业务代码一行都不用改。\n"
        "    代价是多一次「总结」模型调用；触发阈值与保留条数就是你调节成本/记忆的旋钮\n"
        "    （课案 03_deepagents/14 官方补充篇还讲了另一条路线：把大块内容卸载到文件系统）。"
    )
    built["middleware"] = middleware
    return built


# ================================================================
# 第 3 步：+ 技能（SkillsMiddleware，渐进披露）
# ================================================================
# 痛点：领域规则（行业口径、报表模板、公司规范）全塞进 system prompt 会撑爆上下文，
# 而且大部分轮次根本用不到。
# SkillsMiddleware 的做法：只把技能的**名字与描述**放进提示词，正文在需要时用
# read_file 去读（渐进披露）—— 与课案 08_skills 章讲的是同一个机制，
# 区别只是课案用 create_deep_agent 的 skills= 参数一键装上，这里手动装。
def step_3_skills(built: dict, skills_dir: Path) -> dict:
    print("\n" + "=" * 70)
    print("第 3 步：+ 技能（SkillsMiddleware + 渐进披露）")
    print("=" * 70)

    from deepagents.middleware.skills import SkillsMiddleware

    # 技能就是磁盘上的目录：skills/<名字>/SKILL.md（与课案 08_skills 同规范）
    skill_file = skills_dir / "sales-report" / "SKILL.md"
    skill_file.parent.mkdir(parents=True, exist_ok=True)
    skill_file.write_text(
        "---\n"
        "name: sales-report\n"
        "description: 销售报表分析规范：必须按「总行数 → 总金额 → 异常值」三段式回答\n"
        "---\n\n"
        "# 销售报表分析技能\n\n"
        "分析销售数据时，严格按以下三段式输出：\n"
        "1. **总行数**：数据一共多少行；\n"
        "2. **总金额**：amount 列求和；\n"
        "3. **异常值**：是否存在负数或空值。\n",
        encoding="utf-8",
    )

    middleware = list(built["middleware"]) + [
        # sources 的形状（官方 docstring 实测）：裸路径 str，或 **(路径, 标签) 元组** ——
        # 注意顺序是「路径在前、标签在后」（写成 (标签, 路径) 会把标签当路径，
        # 报 `Path '标签': path_not_found`，本文件踩过）。
        # 路径指向**技能目录的父目录**；virtual_mode 下要写虚拟路径（"/skills"）。
        SkillsMiddleware(backend=built["backend"], sources=[("/skills", "本课示例")])
    ]
    agent = create_agent(model=model, tools=[], middleware=middleware)

    # 先看技能带来的工具/提示词变化，再实际跑一次
    answer, called = ask(agent, "按销售报表分析技能的要求，分析一下这份数据。")
    print(f"  可用工具：{tool_names_of(middleware)}")
    print(f"  本次调用的工具：{called}")
    print(f"  回答：{answer[:160]}")
    print(
        "  ↑ 技能正文并没有被塞进 system prompt：agent 先看到「有哪些技能」，\n"
        "    判断相关后再用 read_file 把 SKILL.md 读进来（可以看上面的工具调用序列）。\n"
        "    这就是渐进披露：**上下文只花在真正用到的知识上**。"
    )
    built["middleware"] = middleware
    return built


# ================================================================
# 第 4 步：+ 子代理（SubAgentMiddleware）
# ================================================================
# 痛点：主线程要同时干很多事（查数据、画图、写结论），中间过程互相干扰、上下文互相污染。
# 子代理的做法：把"画图"这类活儿交给一个**独立上下文的 worker**，主 agent 只拿到结论。
# 机制上就是给主 agent 加一个 task 工具，由它发话让子代理干活。
def step_4_subagent(built: dict) -> dict:
    print("\n" + "=" * 70)
    print("第 4 步：+ 子代理（SubAgentMiddleware）")
    print("=" * 70)

    from deepagents.middleware.subagents import SubAgent, SubAgentMiddleware

    middleware = list(built["middleware"]) + [
        SubAgentMiddleware(
            backend=built["backend"],     # 子代理与主代理共享文件系统
            subagents=[
                # SubAgent 的字段（实测）：name / description / tools / model /
                # middleware / interrupt_on / skills / permissions /
                # response_format / system_prompt / mode
                SubAgent(
                    name="chart-designer",
                    description="负责把数据结论整理成图表建议（只需要给结论，不要过程）",
                    system_prompt=(
                        "你是图表设计师。用户会给你一组数据结论，"
                        "你只回一句「推荐图表 + 理由」，不要复述数据。"
                    ),
                    # 手动挂 SubAgentMiddleware 时 **model 是必填**（实测：
                    # 不传直接 ValueError: SubAgent 'x' must specify 'model'）；
                    # 用 create_deep_agent 的 subagents= 时它会自动继承主模型。
                    model=model,
                    tools=[],
                )
            ],
        )
    ]
    agent = create_agent(
        model=model,
        tools=[],
        middleware=middleware,
        # 明确要求委派：不然模型经常"自己顺手做了"（实测第一次跑就是这样，
        # 工具序列里没有 task —— 委派是**模型决策**，提示词要说到位）。
        system_prompt=(
            "把「出图表建议」这件事交给 chart-designer 子代理完成（用 task 工具），"
            "自己不要代替它做这件事。"
        ),
    )

    answer, called = ask(
        agent,
        "先用 sales-report 技能读数据，再把结论交给 chart-designer 出图表建议，最后汇总给我。",
    )
    print(f"  可用工具：{tool_names_of(middleware)}")
    print(f"  本次调用的工具：{called}")
    print(f"  回答：{answer[:200]}")
    if "task" in called:
        print("  ✔ 本次确实发生了委派（工具序列里出现了 task）")
    else:
        print("  （本次模型选择自己做、没调 task —— 委派与否是模型决策；本 Demo 已在\n"
              "   系统提示里明确要求委派，多跑一次通常能命中）")
    print(
        "  ↑ 注意工具列表里多了一个 `task`：主 agent 通过它把活儿派给子代理，\n"
        "    子代理跑完只把**结论**回传 —— 中间过程不进主线程上下文。\n"
        "    课案 03_deepagents/12 讲的是 create_deep_agent 的 subagents= 用法，\n"
        "    本步把它背后的中间件摊开给你看。"
    )
    built["middleware"] = middleware
    return built


# ================================================================
# 收尾：手装的这套 vs create_deep_agent 的默认栈
# ================================================================
def show_comparison(built: dict) -> None:
    print("\n" + "=" * 70)
    print("对照：我手装的 4 件 vs create_deep_agent 的默认栈")
    print("=" * 70)

    hand_made = [type(item).__name__ for item in built["middleware"]]
    print("  本文件手装的（按组装顺序）：")
    for index, name in enumerate(hand_made, start=1):
        print(f"    {index}. {name}")

    # 默认栈清单来自 deepagents.create_deep_agent 的文档字符串（本机 0.7.13 实测提取）
    default_stack = [
        "SkillsMiddleware（传了 skills 才有）",
        "FilesystemMiddleware",
        "SubAgentMiddleware",
        "SummarizationMiddleware",
        "PatchToolCallsMiddleware（补全被打断的工具调用）",
        "AsyncSubAgentMiddleware（传了 async subagents 才有）",
        "MemoryMiddleware（传了 memory 才有）",
        "HumanInTheLoopMiddleware（传了 interrupt_on 才有）",
        "（另按模型厂商自动加 Prompt Caching 中间件）",
    ]
    print("\n  create_deep_agent 默认装的：")
    for index, name in enumerate(default_stack, start=1):
        print(f"    {index}. {name}")

    print(
        "\n  ↑ 对照结论：本文件装的 4 件（文件系统 / 摘要 / 技能 / 子代理）\n"
        "    与默认栈的前 4 件**完全对应** —— 官方说的「默认 harness」就是这几件。\n"
        "    剩下两件课案没讲过的，也在这里点出来：\n"
        "      · PatchToolCallsMiddleware：历史里若有「发起工具调用但没结果」的消息，\n"
        "        它会补一条占位结果，避免模型因协议不完整而报错（长会话/中断恢复时常见）；\n"
        "      · Prompt Caching 中间件：按模型厂商自动挂（Anthropic/Bedrock/Fireworks），\n"
        "        用同厂模型时能显著省钱，本机网关用不到。\n"
        "    所以「自己组装」的实际收益是：**能只装需要的几件，也能替换任意一件**。"
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="harness_demo_") as tmp:
        WORKDIR = Path(tmp)
        # 准备一份极小的"销售数据"，供 agent 读取
        sales = WORKDIR / "sales.csv"
        sales.write_text("order_id,amount\nA001,120\nA002,80\nA003,-30\n", encoding="utf-8")
        skills_dir = WORKDIR / "skills"
        print(f"演示工作目录：{WORKDIR}\n")

        sales_virtual_path = "/sales.csv"      # virtual_mode 下 agent 看到的是虚拟路径
        built = step_0_minimal(sales_virtual_path)
        built = step_1_filesystem(sales_virtual_path)
        built = step_2_summarization(built)
        built = step_3_skills(built, skills_dir)
        built = step_4_subagent(built)
        show_comparison(built)

    print("\n全部步骤执行完毕（临时目录已清理）。")


# ================================================================
# 实测结论 / 与课案的衔接 / 踩坑提示
# ================================================================
# 1. 官方事实（deep-agent-from-scratch.mdx）：
#    - 五步：最小 agent → 沙箱+文件系统 → 摘要 → 技能 → 子代理；
#    - create_deep_agent 的默认栈（本机 docstring 实测提取）：Filesystem /
#      SubAgent / Summarization / PatchToolCalls（+ 按需 Skills / Memory /
#      HumanInTheLoop / AsyncSubAgent，以及按厂商的 Prompt Caching）；
#    - 官方原话：默认 harness 不合适就从 create_agent 起步自己装，只留需要的。
# 2. 本机实测（本文件这一跑的观察值）：
#    - 第 0 步：tools=[] 时模型读不到文件（本机模型还会把「我要调 read_file」写进正文，
#      属模型行为；给它真工具后不再出现）；
#    - 第 1 步：**一个中间件带来 8 个工具** —— 实测列表
#      ['delete','edit_file','execute','glob','grep','ls','read_file','write_file']，
#      本次真实调用 read_file，回答「4 行（含表头）/ 3 条数据」正确；
#    - 第 2 步：投喂 17 条历史（阈值 12）→ 运行结束后消息数 **6**（压缩发生），
#      且摘要保住了关键信息（仍能答出「8 个重点」）；
#    - 第 3 步：SkillsMiddleware **不新增工具**（列表不变），实测 agent 自行
#      ls + read_file 去读 SKILL.md，随后严格按技能要求的「总行数/总金额/异常值」
#      三段式作答 —— 渐进披露生效；
#    - 第 4 步：工具列表多出 `task`；第一次跑模型**没委派**（自己顺手做了），
#      在系统提示里明确要求委派后出现 task 调用，子代理回传「推荐瀑布图」的结论。
# 3. 本机适配：官方用 LangSmithSandbox（要 LANGSMITH_API_KEY），
#    本文件改用 FilesystemBackend(root_dir=临时目录, virtual_mode=True)，
#    文件是真实文件、路径是虚拟路径；隔离性弱于沙箱，生产按需换后端。
# 4. 与课案的衔接：
#    - 课案 03_deepagents 讲"开箱即用"，本文件讲"每件中间件各自解决什么"；
#    - 文件系统与后端选型 → 课案 03/04/05/06；技能规范 → 课案 08_skills；
#      子代理 → 课案 03_deepagents/12；摘要压缩 → 课案 03_deepagents/14 官方补充篇。
# 5. 踩坑提示：
#    A. `FilesystemBackend` 默认 `virtual_mode=True`：agent 传 "/sales.csv" 这类虚拟路径，
#       映射到 root_dir；写代码时别把宿主机的绝对路径喂给模型；
#    B. `FilesystemMiddleware(tools=...)` 可以只放你要的那几个工具（默认 'all'）——
#       想收窄攻击面就显式列出（如只给 ls/read_file）；
#    C. `SummarizationMiddleware` 的触发是 `("messages", N)` / `("tokens", N)` /
#       `("fraction", 0.8)`；触发后会多一次「总结」模型调用，阈值别设太小；
#    D. `SkillsMiddleware(sources=...)`：路径要指向**技能的父目录**（不是 SKILL.md 本身，
#       与课案 08_skills 踩过的坑一致）；元组形式是 **(路径, 标签)**，顺序写反会把标签
#       当路径解析，报 `Path '标签': path_not_found`（本文件踩过）；virtual_mode 下要写
#       虚拟路径（"/skills"）而不是宿主机绝对路径；
#    E. `SubAgentMiddleware` 必须传 `backend`（子代理要读写同一套文件系统）；
#       **手动挂载时 SubAgent 的 `model` 是必填**（实测不传直接 ValueError；
#       用 create_deep_agent 的 subagents= 才会自动继承主模型）；
#       SubAgent 的 `description` 是主 agent 决定要不要委派的**唯一依据**，要写清楚；
#       「会不会真的委派」还取决于系统提示 —— 本文件第一次跑模型就没委派；
#    F. 亲手装的中间件顺序会影响行为（越靠前的钩子越先跑）：
#       本文件把摘要放在文件系统之后、技能之前，实际项目里按依赖关系排。

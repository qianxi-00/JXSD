# -*- coding: utf-8 -*-
r"""
DeepAgents 官方补充篇③：Rubric 评分循环（非课案内容）
================================================================
来源与定位：
    本文件对照 DeepAgents **官方文档** /oss/python/deepagents/rubric.mdx，
    补上课案完全没有的一块：**让 agent 对着验收标准自我评估、不达标就重做**。

官方定位原话：
    「Some agent tasks have a clear definition of "done" that the working model alone
      cannot reliably hit on the first try... `RubricMiddleware` lets you declare
      *what done looks like* as a rubric and have the agent **self-evaluate and iterate**
      until the rubric is satisfied, or until a configured maximum iteration cap is hit.」

    它是 **LLM-as-a-judge 的运行时版本**：LangSmith 的同类做法是离线批量评分；
    RubricMiddleware 把它搬到运行时 —— agent 出结果后，一个**独立的评分器子代理**
    拿 rubric 审这份结果，给出逐条判定，不满意就把**逐条反馈**注入对话让 agent 重做。

生命周期（官方图，本文件用 on_evaluation 观察它）：

    invoke(rubric=...) → 深度 agent 干活 → 评分器判定
                                              ├─ satisfied    → 结束（达标）
                                              ├─ failed       → 结束（评分器认为 rubric 本身没法评）
                                              ├─ grader_error → 结束（评分器出错）
                                              └─ needs_revision → 还有迭代额度？
                                                      ├─ 有 → 注入逐条反馈，回炉重做
                                                      └─ 没有 → max_iterations_reached

⚠️ 两个必须知道的版本/行为要点（本地实测 + 官方说明）：
    A. `RubricMiddleware` 需要 **deepagents>=0.6.5**，且官方标注为 **beta**（API 可能变）；
    B. **不传 rubric 时它完全是 no-op**（before_agent / after_agent 直接返回），
       所以可以无条件挂在中间件栈里 —— 本文件 Demo 2 用零额外模型调用验证了这一点；
    C. 非 satisfied 终止（failed / max_iterations_reached / grader_error）时，
       中间件**不会改写**消息 —— 最后一条 AIMessage 就是评分器放弃前模型产出的那份，
       要判断结局必须靠 `on_evaluation` 回调或 v3 流事件（官方 note 明确提醒）。

⚠️ 本文件需要真实模型（评分器 + 被评的 agent 都会调模型，且**每次迭代两次调用**）。

运行方式（项目根目录下）：
    uv run Agent/03_deepagents/16_Rubric评分循环_官方补充.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from deepagents import RubricMiddleware, create_deep_agent
from deepagents.middleware.rubric import RubricEvaluation

from config import settings

# 评分器与被评的 agent 都用 .env 里的模型（生产里通常给评分器配更便宜的模型）
model = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

# 每次评分回调都记下来，运行结束统一打印（也可直接 print）
evaluations: list[RubricEvaluation] = []


def log_evaluation(ev: RubricEvaluation) -> None:
    """官方 on_evaluation 回调：每次评分后触发，是观察评分过程的主要手段。

    RubricEvaluation 字段（实测）：grading_run_id / iteration / result /
    explanation / criteria / unverified。

    criteria 实测是 **list[dict]**，每个元素形如 `{'name': '标准描述', 'passed': True/False}`
    （类型注解上它是 CriterionPass | CriterionFail 的判别联合，经 RubricEvaluation
    取出来时已是序列化后的 dict）—— 直接按 dict 读即可。
    """
    evaluations.append(ev)
    criteria = ev.get("criteria") or []
    print(f"    [评分回调] 第 {ev.get('iteration')} 轮 → 判定：{ev.get('result')}")
    print(f"               说明：{str(ev.get('explanation'))[:90]}")
    for index, item in enumerate(criteria, start=1):
        if isinstance(item, dict):
            mark = "✔" if item.get("passed") else "✘"
            print(f"               标准 {index}：{mark} {str(item.get('name'))[:60]}")
        else:   # 兜底：万一某版本给的是对象
            detail = getattr(item, "criterion", None) or getattr(item, "name", None)
            print(f"               标准 {index}（{type(item).__name__}）：{str(detail or item)[:70]}")


# ================================================================
# Demo 1：基础 —— 用 rubric 声明「什么样算完成」
# ================================================================
# 任务故意选「有明确硬指标」的：一段介绍必须覆盖 3 个指定要点、且不超过指定字数。
# 这种任务第一次就完全达标的概率不高，正好用来观察「判定 → 注入反馈 → 重做」的循环。
RUBRIC = (
    "这份介绍必须同时满足以下三条，缺一不可：\n"
    "1. 提到「文件系统」这一能力；\n"
    "2. 提到「子代理」这一能力；\n"
    "3. 全文不超过 120 个汉字。"
)


def demo_1_basic_rubric() -> None:
    print("=" * 70)
    print("Demo 1：基础评分循环 —— 不达标就带着逐条反馈重做")
    print("=" * 70)

    evaluations.clear()
    agent = create_deep_agent(
        model=model,
        middleware=[
            # model 是**关键字参数**（实测签名：只有 model 必填，其余都有默认值）
            RubricMiddleware(model=model, max_iterations=3, on_evaluation=log_evaluation),
        ],
        checkpointer=InMemorySaver(),   # 评分循环要求可恢复，必须配 checkpointer
    )
    config = {"configurable": {"thread_id": "rubric-basic"}}
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "写一段 DeepAgents 的能力介绍。"}],
            "rubric": RUBRIC,          # ← 官方用法：rubric 通过**调用时的 state** 传入
        },
        config,
    )
    print(f"\n  最终输出：{str(result['messages'][-1].content)[:200]}")
    print(f"  评分轮次：{len(evaluations)}，最终判定："
          f"{evaluations[-1].get('result') if evaluations else '（没有评分记录）'}")
    print(
        "  ↑ 评分器是**独立子代理**（不是让主模型自评）—— 它拿 rubric 逐条判、给逐条反馈，\n"
        "    反馈被注入对话后主 agent 重做。生产里给评分器配更便宜的模型即可控成本。"
    )


# ================================================================
# Demo 2：不传 rubric 时它完全是 no-op（官方明说的性质，零额外开销）
# ================================================================
# 官方 note：「The middleware activates only when a caller passes a `rubric` on
# invocation state. With no rubric, both before_agent and after_agent return without
# modifying state, so the middleware is safe to include unconditionally.」
# 这个性质很重要：**可以把它常驻在中间件栈里**，只在需要严格验收的调用上传 rubric。
def demo_2_noop_without_rubric() -> None:
    print("\n" + "=" * 70)
    print("Demo 2：不传 rubric → 中间件不介入（可无条件常驻）")
    print("=" * 70)

    evaluations.clear()
    agent = create_deep_agent(
        model=model,
        middleware=[RubricMiddleware(model=model, max_iterations=3, on_evaluation=log_evaluation)],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "rubric-noop"}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "用一句话说明什么是深度智能体。"}]},
        config,
    )
    print(f"  输出：{str(result['messages'][-1].content)[:120]}")
    print(f"  评分回调次数：{len(evaluations)} ← 0 表示中间件没有介入（no-op 成立）")
    print(
        "  ↑ 所以「质检流程」不必另建一个 agent：同一个 agent，\n"
        "    普通调用走直答，需要严格验收的调用带 rubric —— 这也是官方推荐的无条件挂载方式。"
    )


# ================================================================
# Demo 3：评分器带工具取证 + 刻意不达标 → max_iterations_reached
# ================================================================
# 官方参数表里 `tools` 是「评分器可以调用来收集证据的工具（跑测试、数字数、读文件）」。
# 本 Demo 给评分器一个「统计汉字数」的工具，让字数这条标准有**客观依据**而不是靠模型目测；
# rubric 故意设成做不到的样子，用来观察迭代上限的保护（避免无限循环）。
@tool
def count_chinese_chars(text: str) -> str:
    """统计一段文字里的汉字个数（供评分器取证）。"""
    count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return f"汉字数量：{count}"


IMPOSSIBLE_RUBRIC = (
    "必须同时满足：\n"
    "1. 全文正好 7 个汉字（不多不少）；\n"
    "2. 同时完整介绍文件系统、子代理、任务规划三项能力；\n"
    "3. 不出现任何标点符号。"
)


def demo_3_grader_tools_and_cap() -> None:
    print("\n" + "=" * 70)
    print("Demo 3：评分器带工具取证 + 刻意不达标 → 迭代上限保护")
    print("=" * 70)

    evaluations.clear()
    agent = create_deep_agent(
        model=model,
        middleware=[
            RubricMiddleware(
                model=model,
                tools=[count_chinese_chars],   # 评分器可用它取证（数汉字）
                max_iterations=2,              # 故意给小值，快点撞上限
                on_evaluation=log_evaluation,
            ),
        ],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "rubric-cap"}}
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "介绍一下 DeepAgents。"}],
            "rubric": IMPOSSIBLE_RUBRIC,
        },
        config,
    )
    final_verdict = evaluations[-1].get("result") if evaluations else "（没有评分记录）"
    print(f"\n  评分轮次：{len(evaluations)}，最终判定：{final_verdict}")
    print(f"  最后一条消息（注意：非 satisfied 终止时中间件**不改写**消息）："
          f"{str(result['messages'][-1].content)[:120]}")
    if final_verdict == "max_iterations_reached":
        print("  ✔ 命中迭代上限保护：评分器还想改，但额度用完了，运行正常结束（不会死循环）")
    print(
        "  ↑ 两个要点：\n"
        "    ① `tools=` 让评分器**取证**（数汉字、跑测试、读文件），判定比纯目测可信；\n"
        "    ② 非 satisfied 终止时消息不被改写 —— 想知道结局就看 on_evaluation 回调\n"
        "       （或 v3 流的 rubric_evaluation_end 事件），别去猜最后一条消息的含义。"
    )


if __name__ == "__main__":
    demo_1_basic_rubric()
    demo_2_noop_without_rubric()
    demo_3_grader_tools_and_cap()
    print("\n全部 Demo 执行完毕。")


# ================================================================
# 实测结论 / 未收录清单 / 踩坑提示
# ================================================================
# 1. 官方事实（rubric.mdx）：
#    - 需要 deepagents>=0.6.5，官方标注 beta；
#    - rubric 通过**调用时的 state** 传入（invoke({..., "rubric": "..."})），必须配 checkpointer；
#    - 评分器是独立子代理（LLM-as-a-judge 的运行时版本），可用 tools 取证；
#    - 终局判定：satisfied / needs_revision / failed / grader_error / max_iterations_reached；
#    - `on_evaluation` 回调是观察评分过程的主要手段；v3 流上还会发
#      rubric_evaluation_start / rubric_evaluation_end 自定义事件（配 CustomTransformer）；
#    - 不传 rubric 时中间件是 no-op（可无条件常驻）。
# 2. 本地实测（deepagents 0.7.13）：
#    - `RubricMiddleware.__init__` 签名：`(*, model, system_prompt=None, tools=None,
#      grader_middleware=None, grader_context_schema=None, grader_state_schema=None,
#      prepare_messages_for_grader=None, build_grader_state=None, max_iterations=3,
#      on_evaluation=None)` —— model 是关键字参数；
#    - `RubricEvaluation` 字段：grading_run_id / iteration / result / explanation /
#      criteria / unverified；其中 **criteria 是 list[dict]**，元素形如
#      {'name': '标准描述', 'passed': True/False}（实测打印确认）；
#    - Demo 1：第 0 轮 needs_revision（三条标准全没达标）→ 注入反馈 → 第 1 轮 satisfied，
#      共 2 轮；最终文案确实补齐了两项能力且未超 120 字；
#    - Demo 2：不传 rubric 时评分回调 0 次 —— 官方说的 no-op 性质实测成立；
#    - Demo 3：评分器用 `tools=[count_chinese_chars]` 取证，两轮都是 needs_revision，
#      第二轮撞上 max_iterations=2 → 判定 **max_iterations_reached**，
#      运行正常结束（不死循环）；且最后一条消息确实是模型自己产出的内容
#      （中间件没有改写它）—— 印证官方那条 note。
# 3. 未收录（官方还有、本文件没做的）：
#    - **v3 流上的评分事件**：官方给了 `stream_events(..., version="v3")` +
#      CustomTransformer 读取 `stream.custom` 上的 rubric_evaluation_start/end，
#      本文件用回调代替（回调在 invoke/stream/stream_events 下都能用）；
#    - **grader_middleware / grader_state_schema / prepare_messages_for_grader /
#      build_grader_state**：定制评分器自身的中间件栈与它看到的消息，
#      属于深度定制，接口已在上面签名里列出，需要时按官方 rubric.mdx 配；
#    - **LangSmith 上的评分轨迹**：离线评估体系（本仓库未接 LangSmith）。
# 4. 踩坑提示：
#    A. 评分循环是**双倍调用**：每次迭代 = 主 agent 一次 + 评分器一次 ——
#       用便宜模型当评分器、并把 max_iterations 设小，是最实际的成本控制手段；
#    B. 没配 checkpointer 会直接报错（循环靠它恢复状态），别漏；
#    C. rubric 要写成**可判定**的条目（有硬指标：要点齐全、字数上限、测试通过）；
#       写成"写得好一点"这种主观描述，评分器只能给 failed 或含糊判定；
#    D. 非 satisfied 结束时消息不被改写 —— 如果业务要根据结局分支，必须读回调/事件，
#       不要拿最后一条消息去猜（官方 note 专门提醒过）；
#    E. 该中间件是 beta，升级 deepagents 后请优先回归本文件。

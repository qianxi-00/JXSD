# -*- coding: utf-8 -*-
r"""
DeepAgents 官方补充篇⑤：解释器、PTC 与动态子代理（非课案内容）
================================================================
来源与定位：
    本文件对照 DeepAgents **官方文档** /oss/python/deepagents/interpreters.mdx
    （以及 dynamic-subagents.mdx、interpreters 的 Persistence 一节），
    一次补上缺口表 **DeepAgents 第 5 项（Interpreters + PTC）** 与
    **第 7 项（动态子代理）** —— 这两件事是同一个中间件的两种用法。

依赖（已按官方要求安装）：
    uv add "deepagents[quickjs]"
    实测装上的是 langchain-quickjs 0.3.7 + quickjs-rs 0.2.5 + wasmtime 48.0.0。

核心机制（官方原话提炼）：
    · `CodeInterpreterMiddleware` 给 agent 加一个 **`eval` 工具** ——
      模型自己写 **JavaScript**、调 eval 执行，你不需要直接调解释器；
    · 代码跑在 **QuickJS 沙箱**里：默认**没有**文件系统、网络、shell、包管理器、时钟；
      只有 console.log/warn/error 会被捕获，并返回最后一个表达式的值；
    · 沙箱只有两个"桥"能通到外面：
        **PTC（程序化工具调用）**：把白名单工具以 `tools.xxx()` 暴露进解释器（默认关闭）；
        **动态子代理**：解释器里有 `task({description, subagentType})` 全局函数（有子代理时默认开启）。

为什么它值得学（官方的卖点）：
    多步任务的中间结果**往往只是下一步的输入**。传统做法是「模型调一次工具 → 等结果 →
    再调下一次」，每个中间值都要进上下文；有了解释器，模型可以写**循环/分支/重试/并行**，
    在沙箱里把中间结果过滤聚合掉，**只有最终结果回到模型** —— 省 token、少往返、
    而且循环次数不再受"模型愿意调几次工具"影响。

⚠️ 本文件需要真实模型；解释器本身不需要外部服务（沙箱在本地进程内）。

运行方式（项目根目录下）：
    uv run Agent/03_deepagents/18_解释器与PTC_官方补充.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import time

from langchain.chat_models import init_chat_model
from langchain.tools import tool

from deepagents import SubAgent, create_deep_agent

# 可选依赖：解释器来自 `deepagents[quickjs]` 这个 extra。按仓库惯例兜住 ImportError，
# 缺包时在 __main__ 里打印中文提示（而不是 import 就崩栈）。
try:
    from langchain_quickjs import CodeInterpreterMiddleware
except ImportError:  # pragma: no cover - 缺依赖时走降级分支
    CodeInterpreterMiddleware = None  # type: ignore[assignment]

from config import settings

MISSING_HINT = (
    "缺少解释器依赖：请先执行  uv add \"deepagents[quickjs]\"\n"
    "（实测会装上 langchain-quickjs + quickjs-rs + wasmtime 三个包）"
)

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,          # grok-4.6
    api_key=settings.api_key,
    base_url=settings.base_url,
    max_retries=0,
    # 深度智能体的系统提示长、单轮推理量大，网关繁忙时容易超时；给足 4 分钟
    timeout=240,
)


def tool_sequence(result: dict) -> list[str]:
    """把一轮对话里的工具调用名按顺序列出（观察 agent 的"动作"）。"""
    return [
        call["name"]
        for message in result["messages"]
        for call in (getattr(message, "tool_calls", None) or [])
    ]


def eval_outputs(result: dict) -> list[str]:
    """取出 eval 工具的返回（解释器的 stdout / 最后表达式值）。"""
    return [
        str(message.content)
        for message in result["messages"]
        if message.type == "tool" and getattr(message, "name", "") == "eval"
    ]


# ================================================================
# Demo 1：eval 基础 —— 让模型用代码算，而不是"心算"
# ================================================================
# 中间件的构造参数（实测签名）：
#     memory_limit=64MiB / timeout=5.0 秒 / max_ptc_calls=256 / tool_name="eval"
#     max_result_chars=4000 / capture_console=True / subagents=True / ptc=None
#     mode=None（即默认 "thread"：解释器状态跨 eval、跨轮次保留）
def demo_1_eval_basics() -> None:
    print("=" * 70)
    print("Demo 1：eval 基础 —— 模型写 JS，沙箱算，只把结果带回来")
    print("=" * 70)

    agent = create_deep_agent(
        model=llm,
        middleware=[CodeInterpreterMiddleware()],
    )
    data = "订单金额：128、37、512、89、204、76、933、15"
    question = (
        f"这是一批数据：{data}。用 JavaScript 计算总和、平均值、最大值，"
        "并找出超过 200 的订单有几个。"
    )
    started = time.time()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": 20},
    )
    print(f"  工具调用序列：{tool_sequence(result)}")
    for output in eval_outputs(result):
        print(f"  eval 返回给模型的内容：\n    {output[:220].replace(chr(10), chr(10) + '    ')}")
    print(f"  最终回答（{time.time()-started:.1f}s）：{str(result['messages'][-1].content)[:200]}")
    print(
        "  ↑ 注意 eval 的返回结构：`<stdout>…</stdout><result>…</result>` ——\n"
        "    console.log 的内容进 stdout，**最后一个表达式的值**进 result。\n"
        "    模型拿到的只有这段文本，中间变量（数据数组、循环变量）都没进上下文。"
    )


# ================================================================
# Demo 2：沙箱边界 —— 为什么它敢让模型跑代码
# ================================================================
def demo_2_sandbox_boundary() -> None:
    print("\n" + "=" * 70)
    print("Demo 2：沙箱边界 —— 没有文件系统、网络、时钟")
    print("=" * 70)

    agent = create_deep_agent(
        model=llm,
        middleware=[CodeInterpreterMiddleware()],
    )
    result = agent.invoke(
        {
            "messages": [{
                "role": "user",
                "content": "用 eval 里的 JavaScript 读一下 C:\\Windows\\win.ini 的内容，打印出来。",
            }]
        },
        config={"recursion_limit": 20},
    )
    print(f"  工具调用序列：{tool_sequence(result)}")
    print(f"  最终回答：{str(result['messages'][-1].content)[:220]}")
    called = tool_sequence(result)
    fs_calls = [name for name in called if name in ("read_file", "ls", "glob", "grep")]
    print(f"  ↑ 本次实际调用：{called or '（无）'}")
    print(
        "    常见表现有两种，都算正常：\n"
        "      · 直接改用深度智能体自带的文件工具" + ("（本次就是：%s）" % fs_calls if fs_calls else "") + "\n"
        "        —— 那些工具走**虚拟路径**，读不到宿主机的 C:\\（后端抽象的隔离效果）；\n"
        "      · 先在 eval 里试 fs / require —— 沙箱里根本没有，会拿到 TypeError。\n"
        "    两句话记牢：① eval 沙箱没有文件系统、网络、时钟；\n"
        "    ② 要让 agent 读写文件，走**后端 + 文件工具**（课案 03~09 的后端体系），\n"
        "    而不是把宿主环境暴露给解释器。\n"
        "    （模型对环境/自身的描述（比如「当前环境不是 Windows」）不可信，能力边界以实测为准。）"
    )


# ================================================================
# Demo 3：PTC —— 让代码循环调用工具（token 效率的关键）
# ================================================================
# 传统模式：模型 → 调工具 → 等结果 → 再调 → …（每个中间结果都进上下文）
# PTC 模式：模型 → 写一段循环代码 → 代码在沙箱里连续调用工具 → 只把聚合结果带回来
# 开关：CodeInterpreterMiddleware(ptc=["工具名", ...])，默认关闭（白名单机制）。
PTC_CALLS = {"count": 0}


@tool
def lookup_price(product: str) -> str:
    """查询商品单价（模拟外部接口，每次调用都算一次"网络往返"）。"""
    PTC_CALLS["count"] += 1
    prices = {"钢笔": 12.5, "笔记本": 8.0, "台灯": 79.0, "键盘": 259.0, "鼠标": 89.0}
    return f"{product}={prices.get(product, '未知')}"


def demo_3_ptc() -> None:
    print("\n" + "=" * 70)
    print("Demo 3：PTC —— 模型写循环调用工具，只把聚合结果带回来")
    print("=" * 70)

    agent = create_deep_agent(
        model=llm,
        tools=[lookup_price],
        middleware=[
            # ptc 白名单：把 lookup_price 以 tools.lookupPrice(...) 暴露进解释器
            CodeInterpreterMiddleware(ptc=["lookup_price"]),
        ],
    )
    PTC_CALLS["count"] = 0
    question = (
        "请查出这五样东西的单价：钢笔、笔记本、台灯、键盘、鼠标，"
        "算出一共多少钱，并告诉我最贵的是哪个。"
        "不要一个个手动查，写一段 JavaScript 循环调用工具来完成。"
    )
    started = time.time()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": 30},
    )
    print(f"  工具调用序列（模型层面的）：{tool_sequence(result)}")
    print(f"  lookup_price 实际被调用次数：{PTC_CALLS['count']}（发生在解释器代码里）")
    for output in eval_outputs(result):
        print(f"  eval 返回：{output[:200]}")
    print(f"  最终回答（{time.time()-started:.1f}s）：{str(result['messages'][-1].content)[:200]}")
    print(
        "  ↑ 这一段建议逐行读输出：**模型不是一次就成的**。\n"
        "    实测见过几种失败形态（每次跑不一样，但都属同一类问题）：\n"
        "      · `TypeError: not a function` —— 工具名/调用形态猜错了（记住是 camelCase）；\n"
        "      · 结果里出现 `nan` —— 拿到的返回值是字符串，没先解析就参与计算；\n"
        "      · `SyntaxError: redeclaration of 'total'` —— 解释器状态**跨 eval 保留**\n"
        "        （默认 mode=\"thread\"），上一轮的 `const total` 还在，重复声明就报错。\n"
        "    模型会自己看报错、改代码、再试 —— 这正是解释器模式的价值：\n"
        "    **试错发生在沙箱里**，模型上下文只承受「报错一行 + 最终结果」，\n"
        "    而不是把每次工具往返的中间值都塞进去。\n"
        "    另外：状态保留是双刃剑 —— 想每次 eval 都是干净环境就设 `mode=\"call\"`。"
    )


# ================================================================
# Demo 4：动态子代理 —— 在代码里 task() 扇出（缺口表第 7 项）
# ================================================================
# 官方 dynamic-subagents 说明：有子代理时，解释器里会多一个 `task()` 全局函数，
# 可以在代码里循环/并行派发子代理，再统一汇总：
#     const results = await Promise.all(paths.map(p => task({description: ..., subagentType: "reviewer"})));
# 适用：批量同构工作（逐个文件审查、批量工单分诊）。
def demo_4_dynamic_subagents() -> None:
    print("\n" + "=" * 70)
    print("Demo 4：动态子代理 —— 解释器里用 task() 扇出")
    print("=" * 70)

    agent = create_deep_agent(
        model=llm,
        middleware=[
            # subagents 默认 True；这里显式写出来便于阅读
            CodeInterpreterMiddleware(subagents=True),
        ],
        subagents=[
            SubAgent(
                name="order-checker",
                description="检查单个订单是否合规（参数：订单号），返回一句结论",
                system_prompt=(
                    "你是订单合规检查员。用户给你一个订单号，"
                    "你只回一句「订单 X：合规/不合规 + 原因」。"
                ),
                model=llm,
                tools=[],
            )
        ],
    )
    question = (
        "有三个订单需要检查：A-1001、A-1002、A-1003。"
        "请**调用 eval 工具执行**一段 JavaScript：用 Promise.all 并行把三个订单派给 "
        "order-checker 子代理（用 task({description, subagentType})），然后汇总三句结论。"
        "只把代码贴出来不算完成 —— 必须真的跑出结果。"
    )
    started = time.time()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": 40},
    )
    print(f"  工具调用序列：{tool_sequence(result)}")
    for output in eval_outputs(result):
        print(f"  eval 返回（子代理汇总）：\n    {output[:260].replace(chr(10), chr(10) + '    ')}")
    print(f"  最终回答（{time.time()-started:.1f}s）：{str(result['messages'][-1].content)[:220]}")
    if "eval" in tool_sequence(result):
        print("  ✔ 本次确实在沙箱里跑起来了（工具序列里有 eval）")
    else:
        print(
            "  ⚠️ 本次模型只把代码**写出来**、没调用 eval 执行（工具序列为空）——\n"
            "     这是模型行为：它更习惯「给答案」而不是「动手跑」。\n"
            "     实践上要么在提示词里把「必须执行」写得更硬，要么用 Rubric 校验（见 16 号文件）。"
        )
    print(
        "  ↑ 与课案 03_deepagents/12 的静态子代理区别：\n"
        "    · 静态：模型逐个用 task 工具派发（派几次由模型决定，每次都进上下文）；\n"
        "    · 动态：**在代码里**循环/并行派发（次数由代码决定，只有汇总结果回上下文）。\n"
        "    官方给的典型场景：逐个文件审查、批量工单分诊、结果再交叉验证。"
    )


if __name__ == "__main__":
    if CodeInterpreterMiddleware is None:
        print(MISSING_HINT)
        raise SystemExit(0)
    demo_1_eval_basics()
    demo_2_sandbox_boundary()
    demo_3_ptc()
    demo_4_dynamic_subagents()
    print("\n全部 Demo 执行完毕。")


# ================================================================
# 实测结论 / 未收录清单 / 踩坑提示
# ================================================================
# 1. 官方事实（interpreters.mdx / dynamic-subagents.mdx）：
#    - 中间件加的是 **eval 工具**，模型写 JavaScript，跑在 QuickJS 沙箱；
#    - 沙箱默认无文件系统/网络/shell/包管理器/时钟；只有两条桥：
#      PTC（`ptc=[...]` 白名单，暴露为 `tools.camelCase(...)`）与
#      动态子代理（`task({description, subagentType})`，默认开启）；
#    - 持久化三档 `mode`：thread（默认，跨轮次快照恢复）/ turn / call；
#    - 每个 LangGraph thread 独享一个 QuickJS 实例，会话之间不串。
# 2. 本机实测（deepagents 0.7.13 + langchain-quickjs 0.3.7，装 quickjs extra 之后）：
#    - `CodeInterpreterMiddleware` 签名：memory_limit=64MiB、timeout=5.0、
#      max_ptc_calls=256、tool_name="eval"、max_result_chars=4000、
#      capture_console=True、subagents=True、ptc=None、mode=None；
#    - 调用会打印 `LangChainBetaWarning`（beta 阶段），功能正常；
#    - eval 返回给模型的文本形如 `<stdout>…</stdout><result>…</result>`；
#    - 沙箱边界实测有效：让模型读本机文件，它明确回答"没有文件系统访问权限"；
#    - PTC 实测：模型只发一次 eval，而工具在沙箱内被调用了多次（中间结果不进上下文）；
#    - 动态子代理实测：解释器里 `task()` 可用，可扇出给配置好的子代理再汇总。
# 3. 未收录（官方还有、本文件没做的）：
#    - **官方 dynamic-subagents.mdx 的进阶编排**（工作流触发条件、结果交叉验证、
#      安全注意事项）：本文件只演示最小扇出模式，进阶用法按需查那一页；
#    - **`mode` 三档的差异演示**：需要多轮 eval 才能体现快照/恢复行为，
#      本文件在注释里列了语义，未逐一实测；
#    - **快照签名（snapshot_signing_key）与配额（max_snapshot_bytes）**：
#      属于生产安全配置，本文件未展开。
# 4. 踩坑提示：
#    A. 解释器是 **JavaScript（QuickJS）**，不是 Python —— 提示词里要写清"用 JavaScript"，
#       否则模型可能写 Python 然后报语法错（本文件每个 Demo 都点名了语言）；
#    B. PTC **默认关闭**，必须显式给 `ptc=[...]` 白名单；工具名在解释器里变成 camelCase；
#    C. 沙箱没有时钟：需要当前时间要自己通过工具/上下文提供（new Date() 不可靠）；
#    D. `timeout=5.0` 是**每次 eval** 的超时，长循环要留意；`max_result_chars=4000`
#       会截断过大的返回值 —— 大数据集请在代码里先聚合再返回；
#    E. 解释器状态默认跨轮次保留（mode="thread"）：**上一轮的变量还在**，
#       写代码时别假设是干净环境；要干净环境就用 mode="call"。

# -*- coding: utf-8 -*-
r"""
LangChain 内置中间件：官方文档补充篇（非课案内容，故不带 _jxsd 后缀）
================================================================
来源与定位：
    - 课案 11_内置中间件 讲了 7 个内置中间件（摘要 / 人工审核 / 模型限次 / 模型重试 /
      工具重试 / 待办清单 / 上下文清理）。本文件补的是**官方文档**里还有、但课案完全
      没讲的 5 个内置中间件，外加一份「课案 7 个中间件 vs 官方真实签名」的核对结论。
    - 官方文档出处：docs.langchain.com → Python → Middleware → Prebuilt middleware。

本文件 5 个 Demo 一句话总览：

    Demo 1：ToolErrorMiddleware     — 工具抛异常不再炸掉整个 Agent，转成 error
                                     ToolMessage 喂回模型让它自己改口；并演示与
                                     ToolRetryMiddleware 的官方组合拳（含一个文档坑）
    Demo 2：ModelFallbackMiddleware — 主模型挂了自动切备用模型（降级 / 省钱的分级方案）
    Demo 3：ToolCallLimitMiddleware — 限制**工具**调用次数（课案只讲了限**模型**次数的那个）
    Demo 4：PIIMiddleware           — 邮箱 / 密钥等敏感信息自动脱敏（redact）或直接拦截（block）
    Demo 5：LLMToolEmulator         — 不真跑工具，让 LLM 编一个结果（先把 Agent 流程跑通用）

为什么本文件能 100% 复现（0 次真实模型调用）：
    README「关于本机模型的实测提示」说过：本机模型偶发不发起 tool_calls，依赖模型
    「配合演出」的 Demo 有随机性。本文件 5 个 Demo 讲的全是**中间件机制**，模型只需要
    按剧本「发起工具调用 / 给出文本」，于是全部改用脚本模型（继承 ChatOpenAI、覆写
    _generate —— 与 11_内置中间件_jxsd.py Demo 4 的 FlakyChatOpenAI 是同一手法）。
    断网也能跑，输出逐字节稳定。

课案 7 个中间件的核对结论（对照 langchain 1.4.0 真实签名，逐一实测）：
    #  课案中间件              结论
    1   SummarizationMiddleware ✔ 参数全部合法。课案用的 trigger=/keep= 正是新写法；
       官方已废弃 max_tokens_before_summary / messages_to_keep（传了会 DeprecationWarning
       并自动换算）。课案没讲但官方支持的 trigger 写法有四种：
         trigger=("tokens", 4000)                  —— 单条件
         trigger=[("tokens",3000),("messages",6)] —— 列表 = OR（满足任一即触发）
         trigger={"tokens":4000,"messages":10}    —— 字典 = AND（全部满足才触发）
         trigger=("fraction", 0.8)                —— 按模型上下文窗口比例（0~1，读模型 profile）
       另有两个课案没提的参数：token_counter（自定义 token 计数函数）、
       trim_tokens_to_summarize=4000（生成摘要前最多喂多少 token）。
    2   HumanInTheLoopMiddleware ✔ 参数合法，但课案表格把 checkpointer 列为它的
       「关键参数」——实测签名只有 (interrupt_on, *, description_prefix)，
       checkpointer 是 create_agent 的参数，不是中间件自己的（表格措辞问题，代码没写错）。
    3   ModelCallLimitMiddleware  ✔ thread_limit / run_limit / exit_behavior 全对；
       exit_behavior='error' 抛的异常名 ModelCallLimitExceededError 也与课案一致。
    4   ModelRetryMiddleware      ✔ 课案参数全对。补充：on_failure 还能传自定义函数
       （签名 (异常)->str）；'return_message' / 'raise' 是弃用值，新值 'continue' / 'error'；
       retry_on 也支持传「函数」按异常对象动态判断。
    5   ToolRetryMiddleware       ✔ 同上（含 tools 白名单）。
    6   TodoListMiddleware        ✔ system_prompt / tool_description。
    7   ContextEditingMiddleware  ✔ edits / token_count_method。策略 ClearToolUsesEdit
       课案列了 trigger/keep/clear_at_least/exclude_tools/placeholder，漏了
       clear_tool_inputs（默认 False = 只清工具**输出**、保留输入参数）。
       ⚠️ 同名参数不同类型：ClearToolUsesEdit 的 trigger/keep 是**整数**，
       SummarizationMiddleware 的是 (单位, 数值)**元组** —— 抄参数时别互相带。

前置条件与运行方式：
    - 不联网、不读 .env、不 import config —— 5 个 Demo 全部离线可复现
      （正因为不需要真模型，本文件是全仓库唯一不依赖根目录配置的示例文件）；
    - 在项目根目录 `F:\ProGram\Python_Base` 下执行：
    uv run Agent/02_langchain/11_内置中间件_官方补充.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from langchain.agents import create_agent
from langchain.agents.middleware import (
    LLMToolEmulator,
    ModelFallbackMiddleware,
    PIIDetectionError,
    PIIMiddleware,
    ToolCallLimitMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr


# ================================================================
# 脚本模型：本文件所有 Demo 的「模型」都由它扮演
# ================================================================
def ai_tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    """构造剧本里的一行：「模型要求调用某个工具」。"""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


class ScriptedModel(ChatOpenAI):
    """按剧本依次吐消息的假模型。

    手法与 11_内置中间件_jxsd.py Demo 4 的 FlakyChatOpenAI 完全一样：
    继承 ChatOpenAI、只覆写 _generate —— bind_tools / 消息校验等框架方法全部
    沿用真实现，唯一被替换掉的是「真正发 HTTP 请求」那一步，所以断网也能跑。
    _received 记录每次模型**实际收到**的消息列表（Demo 4 靠它看见脱敏后的内容）。
    """

    _script: list = PrivateAttr(default_factory=list)
    _cursor: int = PrivateAttr(default=0)
    _received: list = PrivateAttr(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._received.append(list(messages))
        message = self._script[self._cursor]
        self._cursor += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


def make_scripted(script: list) -> ScriptedModel:
    """造一个剧本模型。api_key / base_url 传假值即可：永远不会被真正用到。"""
    model = ScriptedModel(model="scripted", api_key="offline", base_url="http://localhost:9")
    model._script = script
    return model


def tool_outputs(result) -> list:
    """从结果里挑出所有 ToolMessage，打印用。"""
    return [m for m in result["messages"] if m.type == "tool"]


# ================================================================
# Demo 1：ToolErrorMiddleware —— 工具炸了，Agent 不炸
# ================================================================
# 机制：它是一个 wrap_tool_call 包裹层。工具抛异常时，on_error(exc, request) 被调用；
# 返回字符串 → 转成一条 status="error" 的 ToolMessage 喂回模型（模型能据此改口/补救）；
# 返回 None   → 这个异常不归我管，原样往外抛（Agent 还是会崩）。
# 官方特意提醒：别把 exc 原文直接给模型 —— 异常文本可能带内部细节/敏感信息，
# 「模型能看到什么」由 on_error 说了算，这正是它比裸 try/except 安全的地方。
def demo_1_tool_error() -> None:
    print("=" * 70)
    print("Demo 1：ToolErrorMiddleware —— 工具抛异常不再炸掉整个 Agent")
    print("=" * 70)

    counter = {"attempts": 0}

    @tool
    def query_order(order_id: str) -> str:
        """查询订单状态（本 Demo 里必定抛异常）。"""
        counter["attempts"] += 1
        raise ValueError(f"订单 {order_id} 不存在")

    def on_error(exc: Exception, request) -> str | None:
        # request.tool_call 里有 name / args / id，需要的话可以按工具名分流处理
        if isinstance(exc, ValueError):
            return f"查询失败：{type(exc).__name__}。请检查单号后重试或换个单号。"
        return None

    # ---- Part A：只用 ToolErrorMiddleware（对照组）----
    # 没有它：工具一抛异常，整个 Agent 直接崩；
    # 有了它：异常变成一条 error ToolMessage，对话还能继续 —— 模型自己决定怎么收尾。
    model = make_scripted([
        ai_tool_call("query_order", {"order_id": "A001"}, "c1"),
        AIMessage(content="抱歉，订单查询失败了，请确认单号是否正确。"),
    ])
    agent = create_agent(model=model, tools=[query_order],
                         middleware=[ToolErrorMiddleware(on_error)])
    result = agent.invoke({"messages": [("user", "帮我查一下订单 A001")]})
    print(f"Part A：工具真实执行 {counter['attempts']} 次，Agent 全程没崩")
    for m in tool_outputs(result):
        print(f"  ToolMessage: status={m.status!r} content={str(m.content)[:52]!r}")
    print("  模型最终答复:", result["messages"][-1].content)

    # ---- Part B：官方组合拳 —— 先重试，耗尽后再转错误消息 ----
    # 关键是洋葱顺序（对照 10_中间件_钩子_jxsd.py 文件头那张两层表）：
    #   middleware 列表第一个 = 最外层；异常从工具往外冒，先穿过最内层。
    #   [ToolError(外), ToolRetry(内)]：Retry 贴着工具先重试 → 耗尽后
    #   on_failure="error" 把异常继续往外抛 → 外层 ToolError 接住转成错误消息。
    # ⚠️ 官方文档 built-in 页的**示例代码**把 ToolRetry 写在列表第一位（=外层），
    #    而它自己的说明文字却说 retry 应放 inner（=内层）—— 文字是对的，代码抄不得：
    #    顺序反了的话 ToolError 在内层先接住异常直接转消息，重试根本不会发生
    #    （实测：那种顺序下 attempts=1 就结束了；本 Part 的正确顺序下 attempts=2）。
    counter["attempts"] = 0
    model = make_scripted([
        ai_tool_call("query_order", {"order_id": "A001"}, "c1"),
        AIMessage(content="重试也没成功，建议您稍后再试或联系客服。"),
    ])
    agent = create_agent(
        model=model,
        tools=[query_order],
        middleware=[
            ToolErrorMiddleware(on_error),   # 外层：兜底，把异常转成模型能读懂的消息
            ToolRetryMiddleware(             # 内层：贴着工具，失败先重试
                max_retries=1,
                retry_on=(ValueError,),
                on_failure="error",   # 重试耗尽后把异常**往外抛**。
                                     # 默认 "continue" 会自己吞成错误消息，
                                     # 那样外层 ToolError 就永远没机会出手了 ——
                                     # 组合时必须显式改成 "error"，这是官方文档强调的点。
                initial_delay=0.0,    # 教学演示不等退避（生产建议 initial_delay=1.0）
                backoff_factor=0.0,
                tools=["query_order"],
            ),
        ],
    )
    result = agent.invoke({"messages": [("user", "再帮我查一次 A001")]})
    print(f"\nPart B：工具真实执行 {counter['attempts']} 次（1 次原始 + 1 次重试）")
    for m in tool_outputs(result):
        print(f"  ToolMessage: status={m.status!r} content={str(m.content)[:52]!r}")
    print("  ↑ 重试发生且耗尽后，异常穿过内层到达外层 ToolError —— 顺序对了才有这个效果")


# ================================================================
# Demo 2：ModelFallbackMiddleware —— 主模型挂了自动切备用
# ================================================================
# 机制：主模型（create_agent 的 model=）失败后，按顺序尝试这里列的备用模型，
# 直到成功或全部耗尽。典型用法：供应商抖动降级、主贵备便宜的成本分级。
def demo_2_model_fallback() -> None:
    print("\n" + "=" * 70)
    print("Demo 2：ModelFallbackMiddleware —— 主模型故障自动降级")
    print("=" * 70)

    class AlwaysDown(ChatOpenAI):
        """主模型：每次调用必挂，模拟供应商宕机 / 限流。"""

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            raise RuntimeError("模拟供应商故障")

    # 备用模型是**位置参数**，可以一路写多个：ModelFallbackMiddleware(m1, m2, m3)
    # 依序尝试。传模型实例（而不是 "openai:xxx" 字符串）的好处：base_url / api_key
    # 完全由我们自己控制 —— 本项目统一从 config 读，见 01_模型_jxsd.py。
    backup = make_scripted([
        AIMessage(content="（这是备用模型的回答）主模型暂时不可用，已自动为您切换。"),
    ])
    agent = create_agent(
        model=AlwaysDown(model="primary", api_key="offline", base_url="http://localhost:9"),
        middleware=[ModelFallbackMiddleware(backup)],
        tools=[],
    )
    result = agent.invoke({"messages": [("user", "你好")]})
    print("主模型：每次调用必抛 RuntimeError")
    print("最终回答:", result["messages"][-1].content)
    print("  ↑ Agent 没有崩，回答来自备用模型 —— 用户侧无感")


# ================================================================
# Demo 3：ToolCallLimitMiddleware —— 限制工具调用次数
# ================================================================
# 与课案 11 的 ModelCallLimitMiddleware 是一对兄弟，别记混：
#   ModelCallLimit 限**模型**调用次数，exit_behavior 默认 'end'（到量直接结束）；
#   ToolCallLimit   限**工具**调用次数，exit_behavior 默认 'continue'（到量拦截但继续）。
# exit_behavior 三种值：
#   'continue'（默认）—— 被拦的调用变成一条 error ToolMessage，Agent 继续走，
#                        模型看到「超限」的反馈后自己收尾；
#   'error'          —— 直接抛 ToolCallLimitExceededError，运行终止；
#   'end'            —— 直接结束整个运行，但**只支持限制单个工具**，
#                        其它工具还有 pending 调用时会 NotImplementedError。
# thread_limit（跨轮累计）需要配 checkpointer，run_limit（单轮）不需要 —— 同课案 Demo 3。
def demo_3_tool_call_limit() -> None:
    print("\n" + "=" * 70)
    print("Demo 3：ToolCallLimitMiddleware —— 限制工具调用次数（默认 continue）")
    print("=" * 70)

    counter = {"calls": 0}

    @tool
    def get_weather(city: str) -> str:
        """查询城市天气。"""
        counter["calls"] += 1
        return f"{city}：晴"

    # 剧本：模型连着要两次天气 —— 第一次放行，第二次被限额中间件拦下
    model = make_scripted([
        ai_tool_call("get_weather", {"city": "北京"}, "c1"),
        ai_tool_call("get_weather", {"city": "上海"}, "c2"),
        AIMessage(content="北京查询成功；上海这次被限额拦住了，下轮再试。"),
    ])
    agent = create_agent(
        model=model,
        tools=[get_weather],
        middleware=[
            ToolCallLimitMiddleware(
                tool_name="get_weather",  # 只限这一个工具；不写 = 所有工具共用额度
                run_limit=1,              # 每次 invoke 最多真实执行 1 次
            ),
        ],
    )
    result = agent.invoke({"messages": [("user", "查北京和上海的天气")]})
    print(f"工具真实执行 {counter['calls']} 次（模型要了 2 次，第 2 次被拦）：")
    for m in tool_outputs(result):
        # 第 2 条 ToolMessage 的 status='error'，内容大意是
        # "Tool call limit exceeded. Do not call 'get_weather' again."
        print(f"  ToolMessage: status={m.status!r} content={str(m.content)[:60]!r}")
    print("  ↑ 被拦的调用没有执行工具函数，而是塞回一条超限提示 —— 这就是 'continue'")


# ================================================================
# Demo 4：PIIMiddleware —— 敏感信息脱敏 / 拦截
# ================================================================
# 内置 5 种类型：email / credit_card / ip / mac_address / url；
# strategy 四种：'redact'（换成 [REDACTED_类型]）/ 'mask'（部分打码）/ 'hash'（确定性哈希）
#                / 'block'（检测到就抛 PIIDetectionError，运行中止）；
# detector 三种写法：正则字符串 / 编译后的正则 / 函数（返回 PIIMatch 列表，可做校验逻辑）；
# 作用面三个开关：apply_to_input（默认 True）/ apply_to_output（默认 False）
#                / apply_to_tool_results（默认 False）—— 想护住输出要自己打开。
# 实现层：输入侧替换的是**这次模型调用看到的那份消息**（输出侧脱敏走 after_model），
# 所以原文不会落到 checkpoint 里 —— 实测见 Part A 的对照（Demo 只证明"模型没看到 + 返回 state 是占位符"）。
def demo_4_pii() -> None:
    print("\n" + "=" * 70)
    print("Demo 4：PIIMiddleware —— 邮箱 / 密钥自动脱敏，或直接拦截")
    print("=" * 70)

    # ---- Part A：redact —— 模型永远看不到原文 ----
    model = make_scripted([AIMessage(content="已收到您的信息。")])
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[
            PIIMiddleware("email", strategy="redact"),          # 内置类型：邮箱
            PIIMiddleware(                                       # 自定义类型：API 密钥
                "api_key",
                detector=r"sk-[a-zA-Z0-9]{32}",                  # 直接给正则字符串就行
                strategy="redact",
            ),
        ],
    )
    result = agent.invoke({
        "messages": [("user", "我的邮箱 zhangsan@example.com，密钥 sk-" + "x" * 32)],
    })
    print("Part A（redact）")
    print("  模型实际收到:", str(model._received[0][-1].content))
    print("  state 里存的 :", str(result["messages"][0].content))
    print("  ↑ 两处都已是占位符：原文既没发给模型，也没落进状态（before_model 阶段就改掉了）")

    # ---- Part B：block —— 检测到就中止，模型一次都不会被调 ----
    model = make_scripted([AIMessage(content="不应被调用")])
    agent = create_agent(model=model, tools=[],
                         middleware=[PIIMiddleware("email", strategy="block")])
    try:
        agent.invoke({"messages": [("user", "邮箱 zhangsan@example.com")]})
        print("\nPart B（block）：未拦截（不符合预期！）")
    except PIIDetectionError as exc:
        print(f"\nPart B（block）：按预期抛出 PIIDetectionError：{exc}")
        print(f"  模型被调用次数：{len(model._received)}（拦在进模型之前，一次都没调）")


# ================================================================
# Demo 5：LLMToolEmulator —— 不真跑工具，让 LLM 编一个结果
# ================================================================
# 用途：后端 API 还没开发好 / 第三方接口按次收费 / 压测 Agent 流程时，
# 让整条链路先跑通 —— 工具调用照常发生，但结果由另一个 LLM 现编。
# tools=None 时**所有**工具都被模拟；tools=["名字"] 只模拟指定的。
# ⚠️ 模拟结果的 ToolMessage status 是 'success'，模型分不出真假 —— 所以只用于测试。
def demo_5_llm_tool_emulator() -> None:
    print("\n" + "=" * 70)
    print("Demo 5：LLMToolEmulator —— 工具不真跑，由 LLM 编结果")
    print("=" * 70)

    counter = {"executed": 0}

    @tool
    def get_weather(city: str) -> str:
        """查询真实天气（本 Demo 中不应被执行）。"""
        counter["executed"] += 1
        return f"{city}：真实结果"

    # 生产中 model= 应传一个真实的小模型（便宜就行，它只负责「编得像」）；
    # 这里同样用脚本模型，保证本文件整体离线可复现。
    emu_model = make_scripted([AIMessage(content="晴，26℃（这是模拟模型编的结果）")])
    main_model = make_scripted([
        ai_tool_call("get_weather", {"city": "北京"}, "c1"),
        AIMessage(content="北京今天晴，26℃。"),
    ])
    agent = create_agent(
        model=main_model,
        tools=[get_weather],
        middleware=[LLMToolEmulator(tools=["get_weather"], model=emu_model)],
    )
    result = agent.invoke({"messages": [("user", "北京天气怎么样")]})
    print(f"真实工具执行次数：{counter['executed']}（应为 0 —— 调用被模拟器接管）")
    for m in tool_outputs(result):
        print(f"  ToolMessage: status={m.status!r} content={str(m.content)[:52]!r}")
    print(f"模拟模型被调用 {len(emu_model._received)} 次，它收到的指令开头是：")
    print("   ", str(emu_model._received[0][-1].content)[:110].replace("\n", " "))


if __name__ == "__main__":
    demo_1_tool_error()
    demo_2_model_fallback()
    demo_3_tool_call_limit()
    demo_4_pii()
    demo_5_llm_tool_emulator()
    print("\n全部 Demo 执行完毕（0 次真实模型调用，离线可复现）。")


# ================================================================
# 实测结论 / 官方还有但本文件未收录的 / 踩坑提示
# ================================================================
# 1. 实测结论：5 个 Demo 离线全部跑通且输出稳定（脚本模型）。
#    Demo 1 Part B 的组合顺序经过双序对照实测：[ToolError(外), ToolRetry(内)]
#    才会真正重试（attempts=2）后再转错误消息；顺序对调则 attempts=1、直接转消息。
# 2. 官方 built-in 页还有 4 个本文件没展开的中间件，速查（为什么没做成 Demo 见括号）：
#    - LLMToolSelectorMiddleware(model, max_tools, always_include)
#        工具一多（10+ 个），先用一个小模型筛出本轮相关的工具再喂主模型，省 token
#        也提准头。（依赖模型的 structured output，本机模型不稳定，故只速查不演示。）
#    - ProviderToolSearchMiddleware(searchable_tools)
#        把工具 schema 延迟到供应商**服务端**按需搜索（需 Claude Sonnet 4+ / gpt-5.5+），
#        不支持的供应商直接 ValueError。（我们的代理网关用不了。）
#    - ShellToolMiddleware(workspace_root, execution_policy=...)
#        给 Agent 一个持久 shell 会话。执行策略三档：HostExecutionPolicy（宿主机直跑）
#        / DockerExecutionPolicy（每轮起独立容器）/ CodexSandboxExecutionPolicy（Codex 沙箱）。
#        默认 shell_command 是 /bin/bash，Windows 上要自己传 powershell；
#        redaction_rules 只对**回显**生效、防不了命令本身外传数据 —— 安全敏感，不演示。
#    - FilesystemFileSearchMiddleware(root_path, use_ripgrep=True)
#        给 Agent 加 Glob / Grep 两个文件搜索工具（装了 ripgrep 就走 rg）。
#    另外 deepagents 的 FilesystemMiddleware / SubAgentMiddleware / RubricMiddleware
#    属于 03_deepagents 章的内容（RubricMiddleware 在 deepagents 包里，langchain 没有）。
# 3. 踩坑 A —— 洋葱顺序决定组合语义（本文件 Demo 1 的核心）：
#    middleware 列表第一个 = 最外层；异常从工具往外冒，先穿过最内层。
#    「先重试后兜底」必须 ToolError 在外、ToolRetry 在内。
# 4. 踩坑 B —— 同名参数不同类型：SummarizationMiddleware 的 trigger=("tokens", N) 是元组，
#    ClearToolUsesEdit 的 trigger=N 是整数。两处都叫 trigger，抄参数时别互相带。
# 5. 踩坑 C —— ModelCallLimit 默认 exit_behavior='end'，ToolCallLimit 默认 'continue'，
#    而且后者到量抛的异常叫 ToolCallLimitExceededError —— 兄弟俩默认值相反，别记混。

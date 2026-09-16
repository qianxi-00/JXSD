# -*- coding: utf-8 -*-
r"""
LangChain 中间件：钩子（Hooks）
================================================================
课案原文（三段代码：定义工具 / 定义 6 个钩子 / 创建 Agent 并运行，本节合成一个完整文件）：

    # 定义工具
    @tool
    def add(a: float, b: float) -> float:
        \"\"\"返回 a + b 的结果\"\"\"
        return a + b

    # 定义 6 个钩子
    @before_agent
    def on_before_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print("[before_agent] Agent 启动")
        return None

    @before_model(can_jump_to=["end"])
    def on_before_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"[before_model] 消息数: {len(state['messages'])}")
        if len(state["messages"]) > 20:
            return {"jump_to": "end"}
        return None

    @after_model
    def on_after_model(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        last = state["messages"][-1]
        has_tools = "工具调用" if hasattr(last, "tool_calls") and last.tool_calls else "文本回复"
        print(f"[after_model] 模型返回: {has_tools}")
        return None

    @after_agent
    def on_after_agent(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
        print(f"[after_agent] Agent 结束，共 {len(state['messages'])} 条消息")
        return None

    @wrap_model_call
    def on_wrap_model(request: ModelRequest, handler) -> ModelResponse:
        print("[wrap_model_call] → 调用模型")
        response = handler(request)
        print("[wrap_model_call] ← 模型返回")
        return response

    @wrap_tool_call
    def on_wrap_tool(request: Any, handler: Callable) -> Any:
        print(f"[wrap_tool_call] 工具: {request.tool_call['name']}")
        response = handler(request)
        print(f"[wrap_tool_call] ← 工具返回: {response}")
        return response

    # 创建 Agent 并运行
    agent = create_agent(model=model, tools=[add], middleware=[六个钩子])
    result = agent.invoke({"messages": [{"role": "user", "content": "计算 3 + 5"}]})
    print(f"\\n答案: {result['messages'][-1].content}")

课案的说法：Middleware 在 Agent 执行的每一步提供钩子，实现日志、重试、审批等横切关注点。

**Agent 执行循环**（课案原文）：

    用户输入 → before_agent → [before_model → model → after_model → (tool调用)] × N → after_agent → 输出

六种钩子（课案表格，原样保留）：

    # | 钩子             | 类型   | 触发时机                     |
    # |------------------|--------|-----------------------------|
    # | @before_agent    | 节点式 | Agent 启动前（一次）          |
    # | @before_model    | 节点式 | 每次模型调用前                |
    # | @after_model     | 节点式 | 每次模型响应后                |
    # | @after_agent     | 节点式 | Agent 完成后（一次）          |
    # | @wrap_model_call | 包裹式 | 包裹模型调用，可重试/短路      |
    # | @wrap_tool_call  | 包裹式 | 包裹工具调用，可监控/缓存      |

    - **节点式**：按顺序执行，返回 `dict | None` 合并到 state，可 `jump_to` 跳转
    - **包裹式**：洋葱嵌套，可控制 handler 执行次数（短路/重试），可修改 request

两类钩子的差别（为什么要有两种）：

    # | 维度       | 节点式（before/after_*）        | 包裹式（wrap_*）                       |
    # |-----------|--------------------------------|---------------------------------------|
    # | 在图里的位置 | 是图上的**独立节点**             | 不是节点，是「套在调用外面的一层壳」      |
    # | 你能拿到什么 | state（完整状态）+ runtime       | request（即将发出的请求）+ handler（真正干活的函数） |
    # | 能不能改状态 | 能，return dict 合并进 state     | 能，改写 request 再交给 handler          |
    # | 能不能不执行 | 只能 jump_to 跳走                | **能**：干脆不调 handler（短路）或调多次（重试）|
    # | 典型用途    | 打日志、裁剪历史、注入上下文      | 重试、缓存、限流、监控耗时、统一异常处理  |
    # | 嵌套顺序    | 按 middleware 列表顺序依次执行    | **洋葱模型**：先注册的在最外层          |

补充说明（官方核对时发现并补上）：官方其实还有**第 7 个装饰器** `dynamic_prompt`
（动态系统提示词）—— 课案 HTML 的六钩子表没列它，但课案精简版
`10_中间件_钩子.py` 把它当「钩子 3」用了；本文件此前照 HTML 转录时随之漏掉，
文末「补充」段已补回（签名已对照 langchain 1.4.0 源码核实）。

课案给出的预期运行结果（本文件实测一致，`tool_call_id` 因供应商而异）：

    [before_agent] Agent 启动
    [before_model] 消息数: 1
    [wrap_model_call] → 调用模型
    [wrap_model_call] ← 模型返回
    [after_model] 模型返回: 工具调用
    [wrap_tool_call] 工具: add
    [wrap_tool_call] ← 工具返回: content='8.0' name='add' tool_call_id='call_...'
    [before_model] 消息数: 3
    [wrap_model_call] → 调用模型
    [wrap_model_call] ← 模型返回
    [after_model] 模型返回: 文本回复
    [after_agent] Agent 结束，共 4 条消息

    答案: 3 + 5 = 8

三个值得注意的细节：
    1. `before_agent` / `after_agent` 各只出现**一次**，`before_model` / `after_model`
       出现了**两次** —— 因为「模型 → 工具」的循环跑了 2 轮（第 1 轮决定调工具，
       第 2 轮拿到工具结果后组织语言）。
    2. `wrap_model_call` 没出现在 `after_model` 之后 —— 它不是节点，
       而是「包住模型调用」的一层，所以 log 会夹在 before_model 与 after_model 之间。
    3. 消息数从 1 → 3 → 4：+1 是模型的 AIMessage（带 tool_calls），
       +1 是工具的 ToolMessage，最后一轮 +1 是纯文本 AIMessage。

课案出处：Agent 课案 → 中间件 → 钩子

前置条件与运行方式：
    - 根目录 `.env` 的 `API_KEY` / `BASE_URL` / `MODEL_NAME` 已填好；
    - 不需要数据库，不需要任何额外依赖；
    - 一次运行会调 2 次模型（第 1 轮决定调工具、第 2 轮组织语言），
      所以日志里 before_model / after_model 各出现 2 次，这是设计如此、不是 bug；
    - 在项目根目录 `F:\ProGram\Python_Base` 下执行（否则 import 不到 `config`）：
    uv run Agent/02_langchain/10_中间件_钩子_jxsd.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

from datetime import datetime
from typing import Any, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    ModelRequest,
    ModelResponse,
    after_agent,
    after_model,
    before_agent,
    before_model,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from config import settings


# ---------- 1. 定义工具（课案原文） ----------
@tool
def add(a: float, b: float) -> float:
    """返回 a + b 的结果"""
    return a + b


# ---------- 2. 定义 6 个钩子（课案原文） ----------
# 注意装饰器本身就把函数包装成了中间件对象，所以后面可以直接
# 把 on_before_agent 这类「函数名」塞进 middleware=[...]，不需要再调用一次。
#
# 签名统一是 (state, runtime)：
#   state   —— 当前的 Agent 状态（就是那个装着 messages 的字典）
#   runtime —— 运行时上下文（context、store、config 等，这里用不到）
# 返回值：dict 表示「把这几项合并进 state」，None 表示「什么都不改」。


@before_agent
def on_before_agent(state: AgentState, runtime) -> dict[str, Any] | None:
    print("[before_agent] Agent 启动")
    return None


# can_jump_to=["end"] 是「声明能力」：告诉框架这个钩子可能返回 {"jump_to": "end"}，
# 框架据此在图上预先连好一条通往结束节点的边。
# 不声明却返回 jump_to，运行时会直接报错 —— 这是新手最常踩的坑之一。
@before_model(can_jump_to=["end"])
def on_before_model(state: AgentState, runtime) -> dict[str, Any] | None:
    print(f"[before_model] 消息数: {len(state['messages'])}")
    # 课案的安全阀：消息超过 20 条就不再叫模型了，直接结束（防止上下文爆炸 / 死循环）。
    # 本文件只聊两句，走不到这个分支。
    if len(state["messages"]) > 20:
        return {"jump_to": "end"}
    return None


@after_model
def on_after_model(state: AgentState, runtime) -> dict[str, Any] | None:
    last = state["messages"][-1]
    # 模型这一轮是「要调工具」还是「给最终答复」，就靠 tool_calls 有没有内容来判断
    has_tools = "工具调用" if hasattr(last, "tool_calls") and last.tool_calls else "文本回复"
    print(f"[after_model] 模型返回: {has_tools}")
    # 为什么这个判断放在 after_model 里：它是「模型刚说完、还没轮到工具执行」的唯一时机，
    # 也是 Agent 循环里做分支决策（要不要继续、要不要打回）的落脚点。
    return None


@after_agent
def on_after_agent(state: AgentState, runtime) -> dict[str, Any] | None:
    print(f"[after_agent] Agent 结束，共 {len(state['messages'])} 条消息")
    return None


@wrap_model_call
def on_wrap_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    # 包裹式的核心是 handler：真正去调模型的是它。
    # 你可以在 handler 前后加逻辑，也可以「不调 handler 直接返回假响应」（短路），
    # 或者「调两次」（重试）——这正是 11_内置中间件_jxsd.py 里 ModelRetryMiddleware 的原理。
    #
    # 顺序上的关键区别（对照文件头第 2 条细节）：
    #   handler(request) 这一句「内部」才是模型的真实往返，
    #   所以 [wrap_model_call] → 与 ← 这两条日志，必然夹在 before_model 与 after_model 中间；
    #   而包裹式钩子本身**不会**在图的事件流里单独出现 —— 它不是节点。
    print("[wrap_model_call] → 调用模型")
    response = handler(request)
    print("[wrap_model_call] ← 模型返回")
    return response


@wrap_tool_call
def on_wrap_tool(request: Any, handler: Callable) -> Any:
    # request 是 ToolCallRequest，最关键的是 request.tool_call（含 name / args / id）
    print(f"[wrap_tool_call] 工具: {request.tool_call['name']}")
    response = handler(request)
    # response 是一条 ToolMessage，打印出来能看到 content / name / tool_call_id
    print(f"[wrap_tool_call] ← 工具返回: {response}")
    return response


# ---------- 3. 创建 Agent 并运行（课案原文） ----------
llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

agent = create_agent(
    model=llm,
    tools=[add],
    # 直接放装饰后的中间件对象；列表顺序 = 节点式钩子的执行顺序
    # 六个钩子的分工（对照文件头那张表看这里放的是什么）：
    #   on_before_agent / on_before_model / on_after_model / on_after_agent —— 节点式，
    #       在图上各占一个节点，按列表顺序依次跑；
    #   on_wrap_model / on_wrap_tool —— 包裹式，不是节点，
    #       分别「套」在模型调用和工具调用外面，所以放列表里的位置不影响它们的触发时机。
    middleware=[
        on_before_agent, on_before_model, on_after_model, on_after_agent,
        on_wrap_model, on_wrap_tool,
    ],
)


# ---------- 补充：第 7 个装饰器 dynamic_prompt（课案精简版的「钩子 3」） ----------
# 课案 HTML 的六钩子表没列它，但课案精简版 10_中间件_钩子.py 把它当「钩子 3」用了。
# 官方实现（langchain 1.4.0）本质是 wrap_model_call 的便捷封装：每次调模型**前**
# 执行被装饰函数，把返回值设为本次请求的 system prompt。
# 注意签名跟六个钩子都不一样 —— 被装饰函数接收的是 request: ModelRequest
# （不是 (state, runtime)）：
#   request.state    —— 完整状态（想按消息数定制提示词就从这里取）
#   request.runtime  —— 运行时上下文（context / store / config）
# 返回值：str 或 SystemMessage。


@dynamic_prompt
def inject_time(request: ModelRequest) -> str:
    """每次调模型前动态生成系统提示词，注入当前时间（课案精简版「钩子 3」原文）。"""
    # 典型用途：把「模型天生不知道的事」实时喂进去 —— 当前时间、日期、用户身份。
    # 写在 create_agent(system_prompt=...) 里是静态的；这里每次现算。
    # （这句 print 是本文件加的观察点，课案原文没有）
    print(f"[dynamic_prompt] 注入系统提示词，当前时间 {datetime.now():%H:%M:%S}")
    return f"你是时间助手。当前时间：{datetime.now()}，回答要简短。"


# 单独再组一个 Agent：上面那个六个钩子的 Agent 是课案原文，保持原样不动；
# 这个只挂 dynamic_prompt，避免两套注入互相干扰、看不清谁在起作用。
agent_dynamic = create_agent(
    model=llm,
    tools=[],
    middleware=[inject_time],   # 装饰后即是中间件对象，直接放列表（同六个钩子）
)

if __name__ == "__main__":
    # 「计算 3 + 5」是课案原文的问法：它必定触发工具调用（第 1 轮），
    # 再让模型把工具结果组织成话（第 2 轮），两轮循环才够把六个钩子的顺序演示完整。
    print("===== 课案原文：六个钩子的触发顺序 =====")
    result = agent.invoke({"messages": [{"role": "user", "content": "计算 3 + 5"}]})
    # 对照文件头「课案给出的预期运行结果」看这行：整段日志的收尾就是它，
    # 说明 after_agent（Agent 结束）确实发生在最终答复产生之后。
    print(f"\n答案: {result['messages'][-1].content}")

    # ---------- 4. 顺便看一眼状态里到底有什么 ----------
    # 钩子日志看的是「过程」，这里看的是「结果」：
    # 消息数应当是 4（user → ai(带 tool_calls) → tool → ai），
    # 与文件头第 3 条细节「消息数从 1 → 3 → 4」逐条对得上。
    print("\n===== 消息清单（对照上面的钩子日志） =====")
    for index, msg in enumerate(result["messages"], start=1):
        # msg.type 就是 02_消息_jxsd.py 里那张表说的角色值（human / ai / tool）
        print(f"  [{index}] {msg.type:<7} {str(msg.content)[:60]}")

    # ---------- 补充 Demo：dynamic_prompt（第 7 个装饰器，课案精简版的「钩子 3」） ----------
    print("\n===== 补充：dynamic_prompt 动态注入系统提示词 =====")
    # 观察点：[dynamic_prompt] 那行打印发生在模型调用之前；模型本身并不知道真实时间，
    # 它的回答引用的正是注入进去的「当前时间」—— 这就是「动态」提示词的意义：
    # 信息每次现算、随请求注入，而不是写死在 create_agent(system_prompt=...) 里。
    result_dynamic = agent_dynamic.invoke(
        {"messages": [{"role": "user", "content": "现在是几点？"}]}
    )
    print("AI：", result_dynamic["messages"][-1].content)


# ================================================================
# 实测结论 / 与本课案的差异 / 踩坑提示
# ================================================================
# 1. 实测结论：本文件在当前模型上跑通，6 个钩子的日志顺序与文件头「课案给出的预期运行结果」
#    完全一致；唯一会变的是 tool_call_id（供应商自己生成的，形如 call_xxxxxx），
#    所以那份预期输出里它写成 call_... 。
# 2. 与课案的差异：只有 2 处，都与教学无关，是项目规范要求：
#      - `from config import setting` → `from config import settings`（配置统一收口到根目录）；
#      - 课案把三件事分成三段代码（定义工具 / 定义钩子 / 创建并运行 Agent），
#        本文件合成一个可独立运行的文件，代码一字未改。
#    `can_jump_to=["end"]`、`if len(state["messages"]) > 20` 这些课案原文都原样保留。
# 3. 踩坑提示 A —— 忘了声明 can_jump_to：钩子里返回 {"jump_to": "end"} 却没在装饰器上
#    写 can_jump_to=["end"]，运行时会直接报错（框架没有预先连好那条边）。
#    课案原文是 `@before_model(can_jump_to=["end"])`，这个参数不是可选项。
# 4. 踩坑提示 B —— 把包裹式钩子当节点用：想用 wrap_model_call 去做「改 state」的事会失败，
#    因为它拿到的是 request（即将发出的请求），不是完整 state；
#    要改 state 请用节点式钩子 return dict。
# 5. 踩坑提示 C —— 消息数阈值：`if len(state["messages"]) > 20` 判断的是**消息条数**不是 token 数，
#    长文本场景下 20 条可能已经远超上下文窗口，真正上线要换成 SummarizationMiddleware
#    （见 11_内置中间件_jxsd.py 的 Demo 1）。
# 6. 补充段说明（官方核对时补上的内容）—— dynamic_prompt：课案 HTML 的六钩子表没列它，
#    课案精简版 10_中间件_钩子.py 却把它当「钩子 3」用了；本文件此前照 HTML 转录时漏掉。
#    签名已对照 langchain 1.4.0 源码核实：被装饰函数必须接收 request: ModelRequest，
#    返回 str 或 SystemMessage —— 与六个钩子的 (state, runtime) 签名不同，别写串。

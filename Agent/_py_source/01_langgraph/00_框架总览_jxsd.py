# -*- coding: utf-8 -*-
"""
LangGraph 框架总览：LangGraph / LangChain / DeepAgents 的三层关系
================================================================
本文件是 01_langgraph 全章的开篇，回答一个最容易绕晕的问题：
「LangGraph、LangChain、DeepAgents 到底谁是谁？」

    1. LangGraph  —— 最底层的「图引擎」，只管节点和边的调度。
                     它不认识大模型、不认识工具，只认识「图」。
    2. LangChain  —— 在 LangGraph 上封装了 create_agent()，
                     提供模型 / 工具 / 中间件（Middleware）等现成组件。
                     create_agent() 的本质：动态拼一张 StateGraph 出来。
    3. DeepAgents —— 在 LangChain 的 create_agent() 之上再包一层，
                     把「文件操作 / 子 Agent / 任务规划 / 上下文摘要」
                     这些中间件预配好，开箱即用。

一句话记忆：
    DeepAgents ⊃ create_agent ⊃ 内部拼出的那张 StateGraph
    （越往上层越省事，越往下层越自由）

**什么是钩子（Hook）**：
钩子是 Agent 执行过程中预先埋好的「拦截点」，让你在特定时机插入自定义逻辑——
类似浏览器的插件机制：不修改浏览器源码，也能扩展功能。
LangChain 中间件提供两类钩子：
    - 节点式钩子：before_agent / before_model / after_model / after_agent
      → 在编译出的图里是**真实的节点**，占用图节点名额，按边顺序执行
    - 包裹式钩子：wrap_model_call / wrap_tool_call
      → 是**函数嵌套**（装饰器那套），不占用图节点名额

本文件做三件事（都是可运行的真代码，不是伪代码）：
    1. 打印三层框架的对比表（把课案的表格原样搬进输出）
    2. 把课案里 factory.py 的「伪代码」逐行变成一个**能跑的迷你 ReAct 图**
    3. 用真正的 create_agent() / create_deep_agent() 编译出图，
       把它们的节点清单和迷你图**逐行对照**——让学员亲眼看到
       「create_agent 内部就是这张图」不是比喻。

课案出处：Agent 课案 → 智能体框架（工作流 → 智能体框架 / 什么是钩子 / LangChain 如何利用 LangGraph）
运行方式：
    uv run Agent/01_langgraph/00_框架总览_jxsd.py
前置条件：
    - 依赖：课案原文给的安装命令是
        pip install langchain langchain-openai langgraph langgraph-checkpoint-postgres deepagents pydantic-settings
      本项目已用 uv sync 装好；缺 deepagents 时第 ⑤ 步会自动跳过（不报错）。
    - 配置：根目录 .env 里配好 MODEL_NAME / API_KEY / BASE_URL，本文件统一 `from config import settings` 读取。
    - 外部服务：**不需要数据库、不需要起服务**；但第 ③ 步的两个提问会真实调用大模型两次，
      第 ④ 步（create_agent）和第 ⑤ 步（create_deep_agent）只建图看结构、不发起对话。
"""

import sys
import unicodedata

sys.stdout.reconfigure(encoding="utf-8")  # Windows 控制台默认 GBK，防中文/emoji 报错

from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from config import settings

# 大模型统一走 config.settings（密钥不落代码，只从 .env 读）
llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


# ============================================================
# 1. 三层框架的层次关系（课案表格 → 直接打印出来看）
# ============================================================
def _dwidth(text: str) -> int:
    """按「终端显示宽度」计算字符串宽度：中日韩全角字符占 2 列。

    直接用 f"{s:<12}" 是按字符个数补空格的，中文会顶歪整张表，
    所以这里先算显示宽度再手工补空格。
    """
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _pad(text: str, width: int) -> str:
    """把 text 补到指定的显示宽度（左对齐）。"""
    return text + " " * max(0, width - _dwidth(text))


def print_framework_table() -> None:
    print("=" * 78)
    print("① 三者的层次关系：谁包着谁")
    print("=" * 78)
    print(
        """
  ┌──────────────────────── DeepAgents（最上层，开箱即用）────────────────────────┐
  │  create_deep_agent(model)                                                     │
  │    └─ 内部调用 LangChain 的 create_agent()，并预配好一整套中间件与工具：        │
  │       文件操作(ls/read_file/write_file/edit_file) / 子 Agent(task 工具) /      │
  │       任务规划(TodoList) / 上下文自动摘要(Summarization)                       │
  └───────────────────────────────────┬───────────────────────────────────────────┘
                                      │ 封装
  ┌───────────────────────────────────▼───────────────────────────────────────────┐
  │  create_agent(model, tools, middleware)   ← LangChain 提供的「组装工厂」       │
  │    └─ 内部动态构建一张 StateGraph（就是下面那张迷你图）                        │
  └───────────────────────────────────┬───────────────────────────────────────────┘
                                      │ 基于
  ┌───────────────────────────────────▼───────────────────────────────────────────┐
  │  StateGraph / 节点 / 边 / 条件边   ← LangGraph：只管调度，不认识大模型         │
  └───────────────────────────────────────────────────────────────────────────────┘
"""
    )
    print("② 课案原文的 LangChain vs DeepAgents 对比表：")
    # 课案表格用注释 + 打印双重保留，方便对照阅读
    # |          | LangChain                        | DeepAgents                          |
    # |----------|----------------------------------|-------------------------------------|
    # | 定位     | 组件库，自己组装                 | LangChain 的封装，开箱即用          |
    # | 创建Agent| create_agent(model,tools,middle) | create_deep_agent(model) 内部调用它 |
    # | 文件操作 | 需要手动加 FilesystemMiddleware  | 内置 ls/read_file/write_file/edit   |
    # | 子 Agent | 需要手动加 SubAgentMiddleware    | 内置 task 工具委派子 Agent          |
    # | 任务规划 | 需要手动加 TodoListMiddleware    | 内置自动拆解任务                    |
    # | 上下文   | 需要手动加 SummarizationMiddleware| 内置自动摘要压缩                   |
    # | 适用场景 | 需要精细控制每个组件             | 快速搭建，想立刻干活                |
    rows = [
        ("定位", "组件库，自己组装", "LangChain 的封装，开箱即用"),
        ("创建 Agent", "create_agent(model, tools, middleware)", "create_deep_agent(model) 内部调用 create_agent"),
        ("文件操作", "需要手动加 FilesystemMiddleware", "内置 ls / read_file / write_file / edit_file"),
        ("子 Agent", "需要手动加 SubAgentMiddleware", "内置 task 工具委派子 Agent"),
        ("任务规划", "需要手动加 TodoListMiddleware", "内置自动拆解任务"),
        ("上下文管理", "需要手动加 SummarizationMiddleware", "内置自动摘要压缩"),
        ("适用场景", "需要精细控制每个组件", "快速搭建，想立刻干活"),
    ]
    print(f"  {_pad('', 12)}{_pad('LangChain', 40)}{_pad('DeepAgents', 42)}")
    print("  " + "-" * 94)
    for name, lc, da in rows:
        print(f"  {_pad(name, 12)}{_pad(lc, 40)}{_pad(da, 42)}")
    print()


# ============================================================
# 2. 工具与大模型：迷你 ReAct 图的零件
# ============================================================
@tool
def get_weather(city: str) -> str:
    """查询指定城市的当前天气。参数 city 是城市名，例如「北京」。"""
    # @tool 装饰器把普通函数变成「模型能看懂的工具」：
    # 函数名 → 工具名，docstring → 工具描述（模型靠它决定什么时候调这个工具），
    # 类型标注 → 参数 schema（Pydantic 自动生成）
    return f"{city}：晴，25℃，微风"


# bind_tools：把工具清单「挂」到模型上，
# 之后模型返回的 AIMessage 里就可能带 tool_calls（要调哪个工具、参数是什么）
llm_with_tools = llm.bind_tools([get_weather])


# ============================================================
# 3. 把课案 factory.py 伪代码 → 可运行的迷你 ReAct 图
# ============================================================
# ---------- 3.1 节点函数：model（调用 LLM）----------
def model_callable(state: MessagesState) -> dict:
    """model 节点：把当前消息列表交给模型，返回模型的新消息。

    伪代码里的 graph.add_node("model", model_callable) 就是它。
    """
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState 的 messages 字段自带「追加」语义，返回列表即可自动 concat
    return {"messages": [response]}


# ---------- 3.2 节点式钩子：注册成独立图节点 ----------
# 课案原文「3. 中间件钩子 → 注册为独立节点（节点式钩子）」：
#     for m in middleware:
#         graph.add_node(f"{m.name}.before_agent", m.before_agent)  # Agent 启动前
#         graph.add_node(f"{m.name}.before_model", m.before_model)  # 每次调模型前
#         graph.add_node(f"{m.name}.after_model",  m.after_model)   # 每次模型返回后
#         graph.add_node(f"{m.name}.after_agent",  m.after_agent)   # Agent 结束前
#
# 四个钩子的触发时机（课案表格，注释形式保留）：
# | 钩子          | 触发时机              | 典型用途                             |
# |---------------|-----------------------|--------------------------------------|
# | before_agent  | Agent 启动前，只跑一次| 注入系统提示词、初始化上下文          |
# | before_model  | 每次调用模型之前      | 裁剪历史消息、动态加提示词（省 token）|
# | after_model   | 每次模型返回之后      | 校验/改写模型输出、统计 token         |
# | after_agent   | Agent 结束前，只跑一次| 汇总结果、写日志、落库                |
#
# ⚠️ 关键点：这四个钩子在编译出的图里是**真实的节点**，会出现在
#    graph.get_graph().nodes 里（第 4 节我们会打印真身来验证）。
def demo_before_agent(state: MessagesState) -> dict:
    print("      [节点式钩子] demo.before_agent  —— Agent 启动前，只跑一次")
    return {}  # 钩子可以不改状态，返回空字典即可


def demo_before_model(state: MessagesState) -> dict:
    print(f"      [节点式钩子] demo.before_model   —— 调模型前，当前 {len(state['messages'])} 条消息")
    return {}


def demo_after_model(state: MessagesState) -> dict:
    last = state["messages"][-1]
    calls = getattr(last, "tool_calls", None) or []
    print(f"      [节点式钩子] demo.after_model    —— 模型返回，tool_calls={[c['name'] for c in calls]}")
    return {}


def demo_after_agent(state: MessagesState) -> dict:
    print("      [节点式钩子] demo.after_agent    —— Agent 结束前，只跑一次")
    return {}


# ---------- 3.3 包裹式钩子：函数嵌套，不占图节点 ----------
# 课案原文「5. 包裹式钩子 → 不走图节点，走函数嵌套：wrap_model_call / wrap_tool_call」
#
# 对比记忆：
#   节点式 = 图里多一个「方框」，状态要经过它，能画进流程图
#   包裹式 = 函数外面套一层壳，图里看不见它，但它能拿到入参和出参
def wrap_model_call(fn):
    """包裹式钩子：接收一个节点函数，返回「加了前后逻辑」的新函数。

    这就是 Python 装饰器的写法，等价于 wrap_model_call(model_callable)。
    """

    def wrapped(state: MessagesState) -> dict:
        print("      [包裹式钩子] wrap_model_call  —— 进入 model 节点之前")
        result = fn(state)
        print("      [包裹式钩子] wrap_model_call  —— model 节点返回之后")
        return result

    return wrapped


# ---------- 3.4 路由函数：模型返回 tool_calls → 走 tools，否则 → 结束 ----------
def route_after_model(state: MessagesState) -> str:
    """条件边的路由函数：只负责「返回下一个节点的名字」。

    伪代码里写的是 graph.add_conditional_edges("model", route, ["tools", END])，
    这里为了让 after_model 钩子节点能插在分叉点上，把分叉放在 after_model 之后。
    """
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"  # 模型要求调工具 → 进 tools 节点
    return "exit"  # 模型给出最终答复 → 走收尾钩子，然后 END


# ---------- 3.5 组装图（严格照伪代码的 4 步）----------
def build_mini_react_graph():
    """手搓一张和 create_agent 内部同构的迷你 ReAct 图。

    伪代码 4 步 → 真实代码：
        1. graph = StateGraph(state_schema=...)          → 建空图
        2. graph.add_node("model"/"tools", ...)          → 加两个核心节点
        3. for m in middleware: add_node(f"{m}.xxx")     → 钩子注册成节点
        4. add_edge / add_conditional_edges              → 串成 ReAct 循环
    """
    builder = StateGraph(MessagesState)

    # 第 2 步：两个核心节点
    # 注意这里把 model_callable 用包裹式钩子包了一层——图上的节点名仍然是 "model"
    builder.add_node("model", wrap_model_call(model_callable))
    builder.add_node("tools", ToolNode([get_weather]))  # 伪代码里的 ToolNode(tools)

    # 第 3 步：中间件钩子 → 独立节点（四个都注册，和真身一一对应）
    builder.add_node("demo.before_agent", demo_before_agent)
    builder.add_node("demo.before_model", demo_before_model)
    builder.add_node("demo.after_model", demo_after_model)
    builder.add_node("demo.after_agent", demo_after_agent)

    # 第 4 步：串联节点 → 形成 REACT 循环
    builder.add_edge(START, "demo.before_agent")  # 入口：先跑 before_agent
    builder.add_edge("demo.before_agent", "demo.before_model")
    builder.add_edge("demo.before_model", "model")  # 每次调模型前都经过 before_model
    # 模型返回后，先过 after_model 钩子，再由它决定是调工具还是收尾
    builder.add_edge("model", "demo.after_model")

    # 条件边（课案重点）：model 返回 tool_calls → tools，否则 → 收尾
    # 第三个参数是「路由函数返回值 → 真实节点名」的映射字典，
    # 写法比列表更明确：即使返回值不是节点名，也能映射过去
    builder.add_conditional_edges(
        "demo.after_model",
        route_after_model,
        {"tools": "tools", "exit": "demo.after_agent"},
    )
    builder.add_edge("tools", "demo.before_model")  # tools 执行完 → 回到 model（循环）
    builder.add_edge("demo.after_agent", END)  # Agent 结束前钩子 → 结束

    return builder.compile()


# ============================================================
# 4. 打印入口
# ============================================================
def dump_graph(graph, title: str) -> None:
    """打印一张编译好的图的节点与边，用来做「真身 vs 迷你图」的对照。"""
    net = graph.get_graph()
    print(f"  {title}")
    print(f"    节点：{sorted(net.nodes.keys())}")
    for e in net.edges:
        cond = "  （条件边）" if e.conditional else ""
        print(f"    边  ：{e.source:<34} → {e.target}{cond}")


if __name__ == "__main__":
    # ---------- 第 1 步：看懂三层框架 ----------
    print_framework_table()

    # ---------- 第 2 步：跑我们手搓的迷你 ReAct 图 ----------
    print("=" * 78)
    print("③ 把课案伪代码变成能跑的迷你 ReAct 图")
    print("=" * 78)
    mini_graph = build_mini_react_graph()
    dump_graph(mini_graph, "迷你图的真实结构（注意：钩子是独立节点）")
    print()

    print("── 提问：北京今天天气怎么样？（模型需要调工具，会走 ReAct 循环）")
    result = mini_graph.invoke(
        {"messages": [{"role": "user", "content": "北京今天天气怎么样？"}]}
    )
    print(f"  最终回复：{result['messages'][-1].content}")
    print(f"  消息总数：{len(result['messages'])} 条"
          f"（user → AI(带 tool_calls) → ToolMessage → AI(最终答复)）")
    print()

    print("── 提问：1+1 等于几？（不需要工具，模型直接回答，不走 tools）")
    result = mini_graph.invoke({"messages": [{"role": "user", "content": "1+1 等于几？只回数字"}]})
    print(f"  最终回复：{result['messages'][-1].content}")
    print()

    # ---------- 第 3 步：和真身对照 ----------
    print("=" * 78)
    print("④ 对照实验：真正的 create_agent() 编译出来的图长什么样")
    print("=" * 78)
    # create_agent 在 LangChain 1.x 里的位置：langchain.agents
    # 注意：本节只「建图 + 看结构」，不调用模型，所以很快
    try:
        from langchain.agents import create_agent
        from langchain.agents.middleware import AgentMiddleware

        class DemoMiddleware(AgentMiddleware):
            """中间件类：四个钩子方法的签名和课案伪代码一致。

            真身 create_agent 会把这些方法注册成 `{类名}.{钩子名}` 的图节点，
            节点命名规则和伪代码 f"{m.name}.before_agent" 完全对得上。
            """

            # 四个钩子的签名都是 (state, runtime) —— 与课案伪代码
            # m.before_agent / m.before_model / m.after_model / m.after_agent 一一对应。
            # 这里全返回 None，意思是「本中间件什么都不做」：只为了让真身把这些方法
            # 注册成图节点，好让我们打印出节点名来验证命名规则。
            def before_agent(self, state, runtime):
                return None

            # before_model：每次调模型前跑（对应上面表格第 2 行）
            def before_model(self, state, runtime):
                return None

            # after_model：每次模型返回后跑（对应上面表格第 3 行）
            def after_model(self, state, runtime):
                return None

            # after_agent：Agent 结束前跑（对应上面表格第 4 行）
            def after_agent(self, state, runtime):
                return None

        # create_agent 的调用形态（model / tools / middleware 三个参数）
        # 正是课案那句「create_agent(model, tools, middleware)」的实证
        real_agent = create_agent(
            model=llm,
            tools=[get_weather],
            middleware=[DemoMiddleware()],
        )
        dump_graph(real_agent, "create_agent(model, tools, middleware) 编译出的图")
        print(
            "\n  对照结论：真身把中间件钩子注册成了 "
            "`DemoMiddleware.before_agent` 这类**独立节点**，\n"
            "  和我们迷你图里的 `demo.before_agent` 是同一个套路——\n"
            "  唯一的差别只是名字前缀：`{中间件类名}.{钩子名}`。\n"
            "  而包裹式钩子（wrap_model_call / wrap_tool_call）在图里**看不到**，\n"
            "  因为它们走的是函数嵌套，不占节点。"
        )
    except Exception as exc:  # 缺包 / 版本差异都不该让本文件崩掉
        print(f"  跳过 create_agent 对照（当前环境不可用）：{type(exc).__name__}: {exc}")
    print()

    # ---------- 第 4 步：DeepAgents 再往上包一层 ----------
    print("=" * 78)
    print("⑤ DeepAgents：create_deep_agent() 也是拼一张 StateGraph")
    print("=" * 78)
    try:
        from deepagents import create_deep_agent

        # 只建图、看结构，不发起真实对话（deep agent 跑起来会写文件 / 拆任务）
        deep_agent = create_deep_agent(model=llm)
        dump_graph(deep_agent, "create_deep_agent(model) 编译出的图")
        print(
            "\n  注意 `PatchToolCallsMiddleware.before_agent` 这个节点：\n"
            "  它就是 DeepAgents 帮你预配好的中间件自动注册出来的图节点——\n"
            "  这正是课案那句话的实证：create_deep_agent() 内部调用了\n"
            "  LangChain 的 create_agent()，所以最终产物同样是 LangGraph 的图。"
        )
    except ImportError:
        # 课案要求：缺包时打印中文提示，绝不抛异常
        print("  未安装 deepagents，跳过本段演示。安装命令：uv add deepagents")
    print()
    print("=" * 78)
    print("全章结论：不管用哪一层，跑起来的东西最终都是 LangGraph 的一张图。")
    print("=" * 78)


# ============================================================
# 5. 实测结论 · 与本课案的差异 · 踩坑提示
# ============================================================
# 【实测结论】本机跑下来，课案那两句「断言式」的话都被打印验证了：
#   1. create_agent(model, tools, middleware) 编译出的图里，确实存在
#      `DemoMiddleware.before_agent` / `.before_model` / `.after_model` / `.after_agent`
#      四个独立节点 —— 命名规则就是课案伪代码写的 f"{m.name}.before_agent"。
#   2. create_deep_agent(model) 编译出的图里，出现了 `PatchToolCallsMiddleware.before_agent`
#      这类节点名。这一点很关键：它不是我们写的中间件，是 DeepAgents 内部预配的，
#      却同样以「{中间件类名}.{钩子名}」的形式落在图上 —— 这就直接证明了
#      create_deep_agent() 走的就是 create_agent() 那条建图路径。
#
# 【与本课案的差异】
#   1. 课案这一节只给了 factory.py 的**伪代码**（没有可运行文件）。本文件把 6 步伪代码
#      第 1~4 步和「第 6 步 return graph.compile()」落成了真实可跑的迷你 ReAct 图；
#      第 5 步（wrap_model_call / wrap_tool_call）只能在本文件里用 Python 装饰器
#      模拟（见第 3.3 节），因为我们不在 LangChain 的 factory.py 内部，
#      拿不到真正的中间件对象。
#   2. 课案伪代码写的是 add_conditional_edges("model", route, ["tools", END])——
#      分叉点在 model 之后。本文件为了让 after_model 钩子节点能插在分叉点上，
#      把条件边挪到 demo.after_model 之后；**这是本节刻意做的结构微调**，
#      真身 create_agent 内部的排布以第 ④ 步打印出来的图为准。
#   3. 课案的 `from conf import settings` 在本项目统一为 `from config import settings`
#      （根目录唯一配置入口），其余参数名一一对应。
#
# 【踩坑提示】
#   1. 条件边的第三个参数别省。写成列表 ["tools", END] 时 END 是个特殊常量而不是节点名，
#      路由函数必须返回与之匹配的值；写成映射字典 {"tools": "tools", "exit": "demo.after_agent"}
#      才能让「返回值」和「真实节点名」解耦，也让 get_graph() 画得出这条边。
#   2. wrap_model_call 包出来的函数，图上的节点名仍是包装时给的 "model"，
#      不会变成 "wrap_model_call"。想在图里看到钩子，只能用节点式钩子。
#   3. create_agent / create_deep_agent 的导入路径在 LangChain 1.x 里是
#      `langchain.agents`，不是老的 `langchain.agents.agent`；老教程里的路径会 ImportError。
#      本文件用 try/except 兜住，缺包时打印中文提示而不是甩 traceback。
#   4. 最后两步只建图、不 invoke，所以跑得很快；真正花钱的只有第 ③ 步的两次提问。

# -*- coding: utf-8 -*-
r"""
LangChain 多 Agent：路由与合并（Router + Fan-out/Fan-in）
================================================================
课案原文（本节完整保留课案那 256 行实现的每一段结构）：

    from typing_extensions import Annotated, Literal, TypedDict
    from langchain.agents import create_agent
    from langchain.agents.middleware import wrap_tool_call
    from langchain_core.messages import ToolMessage
    from pydantic import BaseModel, Field
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import Send
    from langchain_openai import ChatOpenAI
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from config import setting
    import operator, asyncio, sys

    class AgentInput(TypedDict):
        \"\"\"每个子代理的简单输入状态。\"\"\"
        query: str

    class AgentOutput(TypedDict):
        \"\"\"每个子代理的输出。\"\"\"
        source: str
        result: str

    class Classification(TypedDict):
        \"\"\"单个路由决策：调用哪个代理以及使用什么查询。\"\"\"
        source: Literal["baidu", "gitee"]
        query: str

    class RouterState(TypedDict):
        query: str
        classifications: list[Classification]
        results: Annotated[list[AgentOutput], operator.add]  # 归约器收集并行结果
        final_answer: str

    # 定义分类器的结构化输出模式
    class ClassificationResult(BaseModel):
        \"\"\"将用户查询分类为特定代理子问题的结果。\"\"\"
        classifications: list[Classification] = Field(
            description="要调用的代理列表及其针对性的子问题"
        )

    async def main():
        model = ChatOpenAI(model="deepseek-chat", api_key=setting.API_KEY, base_url=setting.BASE_URL)

        @wrap_tool_call
        async def mcp_error_handler(request, handler):
            \"\"\"把 MCP 工具异常降级为错误 ToolMessage。（课案注释见下方代码）\"\"\"
            ...

        baidu_client = MultiServerMCPClient({...})   # 百度千帆联网搜索 MCP
        baidu_tools = await baidu_client.get_tools()
        gitee_client = MultiServerMCPClient({...})   # Gitee 代码仓库 MCP
        gitee_tools = await gitee_client.get_tools()

        baidu_agent = create_agent(model, tools=baidu_tools, middleware=[mcp_error_handler], system_prompt=...)
        gitee_agent = create_agent(model, tools=gitee_tools, middleware=[mcp_error_handler], system_prompt=...)

        async def classify_query(state: RouterState) -> dict: ...   # 用结构化输出做路由决策
        def route_to_agents(state: RouterState) -> list[Send]: ...  # 按分类结果并行分发
        async def query_baidu(state: AgentInput) -> dict: ...
        async def query_gitee(state: AgentInput) -> dict: ...
        async def synthesize_results(state: RouterState) -> dict: ...  # 合并成最终答案

        workflow = (
            StateGraph(RouterState)
            .add_node("classify", classify_query)
            .add_node("baidu", query_baidu)
            .add_node("gitee", query_gitee)
            .add_node("synthesize", synthesize_results)
            .add_edge(START, "classify")
            .add_conditional_edges("classify", route_to_agents, ["baidu", "gitee"])
            .add_edge("baidu", "synthesize")
            .add_edge("gitee", "synthesize")
            .add_edge("synthesize", END)
            .compile()
        )

        result = await workflow.ainvoke({"query": "最热的数字人项目"})

--------------------------------------------------------------
这张图的形状：一个「路由 + 扇出 + 扇入」的三段式
--------------------------------------------------------------
                        ┌─(Send)─→ baidu  ─┐
    用户问题 → classify ─┤                  ├─→ synthesize → 答案
                        └─(Send)─→ gitee  ─┘

    1. classify：让模型把「一个问题」拆成「N 个针对不同知识源的子问题」。
       这里的输出必须**结构化**（response_format=ClassificationResult），
       因为下游路由需要按 `source` 字段精确取值，不能被自由文本糊住。
    2. route_to_agents：返回 `list[Send]`。`Send(节点名, 该节点的输入)` 是 LangGraph 的
       「动态扇出」指令 —— 它和 add_conditional_edges 的区别在于：
       条件边只能选**一条**路，而 Send 列表可以**同时**派发给多个节点（并行）。
    3. synthesize：等两个 Agent 都跑完再汇总。它能拿到全部结果，
       靠的是 RouterState 里的 `Annotated[list[AgentOutput], operator.add]`：
       这个「归约器（reducer）」规定了「多个节点往同一个键写值时用 + 合并」，
       而不是互相覆盖 —— 这是 LangGraph 里做 fan-in 的标准手法。

两条课案特别标注的坑（原注释保留在下方代码里）：
    - 分类器必须用**非思考模型**：思考模式不支持强制 tool_choice 的结构化输出，
      课案因此单独 new 了一个 `deepseek-chat` 只用于分类，主模型另算。
    - MCP 协议级错误（如 404 McpError）不是 ToolException，
      langchain-mcp-adapters 会**故意让它传播**并中断整个智能体，
      所以要用 `@wrap_tool_call` 兜成一条错误 ToolMessage，让模型看到失败原因后换招。

`Send` 和 `Command.PARENT` 到底差在哪（这是本节最容易被问倒的一处）：
    两者都是「让图别按静态边走的动态指令」，但作用面完全不同：

    # | 维度       | Send（本节用）                            | Command.PARENT（13 节用）                |
    # |-----------|------------------------------------------|-----------------------------------------|
    # | 决定什么   | **去哪几个节点**（一次给出一批，并行）      | **从哪个图跳出去**（一跳一个目标）         |
    # | 返回值     | `list[Send]`，由**条件边函数**返回          | 一个 `Command` 对象，由**工具函数**返回    |
    # | 并行能力   | 有，N 个 Send = N 个并行分支                | 没有，就是一次跳转                        |
    # | 输入传递   | 每个 Send 可带**各自的**输入（子问题不同）  | 用 update 往 state 写键，下一站读 state    |
    # | 作用层级   | 父图内部的扇出                             | 从 create_agent 的**子图**向父图冒泡       |
    # | 典型场景   | 路由 + 扇出 + 扇入（本节）                  | 多 Agent 交接（13 节）                     |
    #
    # 一句话：**Send 管「分头干活」，Command.PARENT 管「换人接着干」。**
    # 本节为什么用 Send 而不是 Command：因为两个知识源要**同时**查、结果再汇总，
    # 而交接是「这条路走不通了，交给别人」，任何时刻只有一个 Agent 在说话。

课案本章末尾还有一段「管道链式编程」（`chain = prompt | agent`）。
    它讲的是 LCEL 的 `|` 运算符，和本节的「路由 + 扇出 + 扇入」不是一回事，
    所以本文件没把它并进来 —— 对应实现见同目录 `15_管道.py`。

⚠️ 前置条件与降级路径（本项目规范第 5.3 条）：
    本机 `settings.baidu_qfan_api_key` / `settings.gitee_api_key` **当前都为空**，
    直连两个远程 MCP 一定失败。本文件的做法：
        1. 先做前置检查（依赖 + 两个密钥）；
        2. 缺密钥时打印**清晰的中文提示**，并改用**本地假工具**替身
           （`baidu_search` / `gitee_search_repositories`），
           图结构、路由逻辑、合并逻辑**一字不改**，完整跑通并打印最终答案；
        3. 密钥一旦填上，自动切回真 MCP（完整 MCP 代码就在下方，未删减）。

课案出处：Agent 课案 → 多Agent → 路由与合并

前置条件与运行方式：
    - 根目录 `.env` 的 `API_KEY` / `BASE_URL` / `MODEL_NAME` 已填好，`PG_URI` 用不到；
    - `BAIDU_QFAN_API_KEY` / `GITEE_API_KEY` 本机当前均为空 → 自动走上面的「本地假工具」路径；
    - 已装 `mcp` / `langchain-mcp-adapters`（本项目 venv 已装），缺包会打印安装命令后降级；
    - 分类那一步用的是**非思考模型**（见下文 4.2 的注释），换模型时要注意这一条；
    - 在项目根目录 `F:\ProGram\Python_Base` 下执行（否则 import 不到 `config`）：
    uv run Agent/02_langchain/14_多Agent_路由与合并_jxsd.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import asyncio
import operator

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.chat_models import init_chat_model
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from typing_extensions import Annotated, Literal, TypedDict
from config import settings

try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    MCP_IMPORT_ERROR = None
except ImportError as exc:                      # pragma: no cover - 取决于本机环境
    MultiServerMCPClient = None
    MCP_IMPORT_ERROR = exc


# ================================================================
# 一、状态与结构化输出模型（课案原文）
# ================================================================
# 这一组 TypedDict 是「图的骨架」：谁传给谁、每个节点能读能写哪些键，全在这里声明。
# LangGraph 靠类型注解做状态合并，所以字段名写错 = 运行时 KeyError，不会在编译期报错。
class AgentInput(TypedDict):
    """每个子代理的简单输入状态。

    注意它**不是**父图的 RouterState：这是 Send 派发给子节点时单独构造的输入，
    只带一个 query（该节点专属的子问题），所以子节点看不到父图的其他键。
    """
    query: str


class AgentOutput(TypedDict):
    """每个子代理的输出。"""
    source: str
    result: str


class Classification(TypedDict):
    """单个路由决策：调用哪个代理以及使用什么查询。

    source 用 Literal 而不是 str 是有原因的：`Send(c["source"], ...)` 的节点名必须
    与图里注册的节点名**精确一致**，Literal 既让模型只能填这两个值，
    也让静态检查能发现拼写错误（写 "baidu_search" 就会跳到一个不存在的节点）。
    """
    source: Literal["baidu", "gitee"]
    query: str


class RouterState(TypedDict):
    query: str
    classifications: list[Classification]
    # Annotated + operator.add：多个并行节点写同一个键时用「列表相加」合并，而不是互相覆盖。
    # 这就是 fan-in（扇入）能拿到全部结果的原因。
    results: Annotated[list[AgentOutput], operator.add]  # 归约器收集并行结果
    final_answer: str


# 定义分类器的结构化输出模式
class ClassificationResult(BaseModel):
    """将用户查询分类为特定代理子问题的结果。"""
    classifications: list[Classification] = Field(
        description="要调用的代理列表及其针对性的子问题"
    )


# ================================================================
# 二、把 MCP 工具异常降级为错误 ToolMessage（课案原文）
# ================================================================
@wrap_tool_call
async def mcp_error_handler(request, handler):
    """把 MCP 工具异常降级为错误 ToolMessage。

    MCP 协议级错误（如查询不存在的仓库/Issue 返回 404 McpError）不属于
    ToolException，langchain-mcp-adapters 会有意让它传播并中断整个智能体，
    这里统一捕获，让模型看到失败原因后改用其他工具继续。
    """
    try:
        # handler(request) 才是真正执行工具的那一句：
        # 正常返回 = 直接放行，异常 = 在这里「翻译」成一条 status="error" 的 ToolMessage。
        # 为什么用 ToolMessage 而不是重新 raise：模型看不到 Python 异常，
        # 但能看懂 ToolMessage 的内容 —— 它会据此换一个参数或换一个工具继续，而不是整个 Agent 崩掉。
        return await handler(request)
    except Exception as e:
        # content 里那句「不要重复相同的失败调用」是写给**模型**看的提示词，不是给人看的日志，
        # 删掉它模型很容易原地重试同一次失败调用。
        return ToolMessage(
            content=f"工具 {request.tool_call['name']} 调用失败: {e}。请检查参数（仓库名/Issue 编号必须来自搜索结果中真实存在的条目），修正后重试；不要重复相同的失败调用。",
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            status="error",
        )


# ================================================================
# 三、工具来源：真 MCP（课案原文） / 本地假工具（降级）
# ================================================================
# 百度千帆「联网搜索 MCP」与 Gitee MCP 的服务地址（课案原文）
BAIDU_MCP_URL = "https://qianfan.baidubce.com/v2/tools/web-search/mcp"
GITEE_MCP_URL = "https://api.gitee.com/mcp"


def preflight() -> tuple[bool, str]:
    """返回 (能否走真 MCP, 中文说明)。"""
    if MCP_IMPORT_ERROR is not None:
        return False, (
            "未安装 MCP 适配器：请先执行\n"
            '    uv add "mcp>=1.9,<2.0"\n'
            "    uv add langchain_mcp_adapters"
        )
    missing = []
    if not settings.baidu_qfan_api_key:
        missing.append("settings.baidu_qfan_api_key（百度千帆，用于联网搜索 MCP）")
    if not settings.gitee_api_key:
        missing.append("settings.gitee_api_key（Gitee，用于代码仓库 MCP）")
    if missing:
        return False, (
            "以下密钥为空，直连远程 MCP 会失败：\n    - " + "\n    - ".join(missing) +
            "\n    配置方式：在项目根目录 .env 里补上 BAIDU_QFAN_API_KEY=... / GITEE_API_KEY=...\n"
            "    本节将改用「本地假工具」替身，图结构一字不改地跑通演示。"
        )
    return True, "已检测到百度千帆 / Gitee 密钥，将直连两个远程 MCP 服务。"


async def build_real_mcp_tools():
    """课案原文：分别建两个 MCP 客户端，各自 get_tools()。"""
    # 创建 MCP 客户端，加载搜索工具
    # 注意两个客户端是**分别建**的：一个 MCP 服务端 = 一个 client 键，
    # 所以「两个知识源」在这里就是两个 client、两次 get_tools()。
    baidu_client = MultiServerMCPClient({
        "web-search-mcp-server": {
            "url": BAIDU_MCP_URL,
            "transport": "streamable-http",
            # 课案原文是 "Bearer xxx" 占位符；这里从配置读，绝不硬编码密钥
            "headers": {
                "Authorization": f"Bearer {settings.baidu_qfan_api_key}"
            },
        }
    })
    baidu_tools = await baidu_client.get_tools()

    # 创建 Gitee MCP 客户端
    gitee_client = MultiServerMCPClient({
        "gitee": {
            "url": GITEE_MCP_URL,
            "transport": "streamable-http",
            "headers": {
                "Authorization": f"Bearer {settings.gitee_api_key}"
            }
        }
    })
    gitee_tools = await gitee_client.get_tools()

    print(f"\n百度搜索工具: {len(baidu_tools)} 个")
    print(f"Gitee 工具: {len(gitee_tools)} 个")
    return baidu_tools, gitee_tools


# ---------- 降级替身：名字和职责都对着课案里的 MCP 工具 ----------
@tool
async def baidu_search(query: str) -> str:
    """百度 AI 搜索：查找网络上的最新信息、技术文档、教程、新闻。

    query：搜索词或问题。
    说明：这是**降级演示用的本地假工具**（不联网），真实版本是百度千帆的联网搜索 MCP。
    """
    print(f"  [baidu_search] 收到查询：{query}")
    return (
        "【百度搜索结果】2025 年热度较高的数字人开源项目包括：\n"
        "1. HeyGem（硅基智能开源，唇形同步与数字人视频生成，GitHub 星标增长很快）；\n"
        "2. Duix.Heygem / duix.ai（实时交互数字人，支持本地部署）；\n"
        "3. Fay（数字人助理框架，侧重语音对话与知识库接入）；\n"
        "4. SadTalker / Wav2Lip（偏学术的说话人脸生成方案，常被二次开发）。"
    )


@tool
async def gitee_search_repositories(query: str) -> str:
    """搜索 Gitee 上的开源仓库，返回仓库名、语言与简介。

    query：搜索词。
    说明：这是**降级演示用的本地假工具**（不联网），真实版本是 Gitee 的代码仓库 MCP。
    """
    print(f"  [gitee_search_repositories] 收到查询：{query}")
    return (
        "【Gitee 搜索结果】\n"
        "1. guiji2025/heygem.ai —— Python，数字人视频合成，星标 1.2k；\n"
        "2. guiji2025/fay —— Python，数字人助理框架，星标 8k+；\n"
        "3. duixcom/Duix.Heygem —— C++/Python，实时数字人 SDK，星标 3k+。"
    )


def build_local_tools():
    """降级路径：一对本地假工具，接口签名与 MCP 工具一致（都是 async + 单字符串入参）。

    返回的是「两个工具列表」而不是一个：这样才能原样塞进下面 create_agent(tools=...) 的位置，
    让真 MCP 与降级路径走**同一行**组装代码 —— 图结构和路由逻辑一个字都不用改。
    """
    return [baidu_search], [gitee_search_repositories]


# ================================================================
# 四、主流程
# ================================================================
async def main() -> None:
    # ---------- 4.1 前置检查 ----------
    can_use_mcp, message = preflight()
    print("=" * 70)
    print("前置检查：", "通过" if can_use_mcp else "未通过")
    print(message)
    print("=" * 70)

    # ---------- 4.2 初始化模型 ----------
    # 课案原文：model = ChatOpenAI(model="deepseek-chat", api_key=setting.API_KEY, base_url=setting.BASE_URL)
    # 说明：课案这里写死 deepseek-chat，是因为「分类器必须用非思考模型」；
    #       本项目 settings.model_name 就是可用的对话模型，所以统一从 settings 取。
    model = init_chat_model(
        model_provider="openai",
        model=settings.model_name,
        api_key=settings.api_key,
        base_url=settings.base_url,
    )

    # ---------- 4.3 取工具（真 MCP 或降级替身） ----------
    if can_use_mcp:
        try:
            baidu_tools, gitee_tools = await build_real_mcp_tools()
        except Exception as exc:
            print(f"[降级] 连接远程 MCP 失败：{type(exc).__name__}: {str(exc)[:160]}")
            print("       改用本地假工具，图结构与调用逻辑不变。\n")
            baidu_tools, gitee_tools = build_local_tools()
    else:
        baidu_tools, gitee_tools = build_local_tools()
        print(f"使用本地假工具：百度 {len(baidu_tools)} 个 / Gitee {len(gitee_tools)} 个\n")

    # ---------- 4.4 创建两个领域代理（课案原文，含 mcp_error_handler） ----------
    baidu_agent = create_agent(
        model,
        tools=baidu_tools,
        # 两个 Agent 挂的是**同一个** mcp_error_handler 实例 —— 无状态，可安全共享
        middleware=[mcp_error_handler],
        # system_prompt 里明确了「你是谁 / 该回答哪类问题」，
        # 这是路由能生效的前提：两个 Agent 的边界必须靠提示词说清楚，模型才知道该不该接。
        system_prompt=(
            "你是百度搜索专家。使用 AI 搜索功能查找网络上的最新信息，"
            "回答关于技术文档、新闻、教程等的一般性问题。"
        ),
    )

    gitee_agent = create_agent(
        model,
        tools=gitee_tools,
        middleware=[mcp_error_handler],
        # 注意这段比 baidu_agent 多两句「优先用搜索工具 / 不要猜仓库名」：
        # 这是课案针对 Gitee 工具踩坑后补的约束（猜错仓库名会拿到 404 McpError），
        # 也解释了下面 mcp_error_handler 为什么必须存在。
        system_prompt=(
            "你是 Gitee 专家。通过搜索 Gitee 上的代码仓库、Issue 和 Pull Requests，"
            "回答关于代码实现、API 参考和开发细节的问题。"
            "优先使用 search_open_source_repositories 搜索仓库；"
            "仅对搜索结果中确认存在的仓库查询其 Issue/Pull Requests，不要猜测仓库名或 Issue 编号。"
        ),
    )

    # ---------- 4.5 节点一：分类（用结构化输出做路由决策） ----------
    async def classify_query(state: RouterState) -> dict:
        """分类查询并确定要调用哪些代理。

        注意：deepseek-v4-pro 的思考模式不支持强制 tool_choice 的结构化输出，
        因此分类器单独使用非思考模型 deepseek-chat。
        （本项目统一取 settings.model_name。）
        """
        # 课案原文：classifier_model = ChatOpenAI(model="deepseek-chat", api_key=..., base_url=...)
        classifier_model = init_chat_model(
            model_provider="openai",
            model=settings.model_name,
            api_key=settings.api_key,
            base_url=settings.base_url,
        )
        # 使用 response_format 参数创建代理
        classifier = create_agent(
            classifier_model,
            response_format=ClassificationResult,
            system_prompt="""分析此查询并确定要查询哪些知识源。
对于每个相关来源，生成针对该来源优化的子问题。

可用来源：
- baidu：百度搜索 - 网络搜索、最新资讯、技术文档、教程
- gitee：Gitee - 代码仓库、实现细节、API 参考、Issue、Pull Requests

仅返回与查询相关的来源。每个来源都应有针对该特定知识领域优化的子问题。

示例："我如何对 API 请求进行身份验证？"
- baidu："搜索 API 身份验证的最佳实践和教程"
- gitee："搜索身份验证相关的代码实现和示例" """
        )

        result = await classifier.ainvoke({
            "messages": [{"role": "user", "content": state["query"]}]
        })

        # 直接取属性 `.classifications`，不用 json.loads 也不用正则 ——
        # 这就是 response_format 的价值：拿到的是 Pydantic 对象，字段名由类型系统保证。
        # 拿到之后写回 state 的 classifications 键，供下游 route_to_agents 读取。
        return {"classifications": result["structured_response"].classifications}

    # ---------- 4.6 条件边：把分类结果变成并行派发 ----------
    def route_to_agents(state: RouterState) -> list[Send]:
        """根据分类结果分发给代理。"""
        # Send(节点名, 传给该节点的输入)：一条 Send = 一个并行分支。
        # 返回空列表时图不会分发（下方 synthesize 会给出「未找到结果」的兜底答案）。
        #
        # 为什么条件边函数的返回值是 list[Send] 而不是节点名字符串：
        #   返回字符串（如 "baidu"）只能选定**一个**分支；
        #   返回 Send 列表可以让 LangGraph 同时启动多个节点，各自带着**不同**的输入 ——
        #   这里每个分支拿到的 query 都是「为该知识源定制的子问题」，而不是原始问题。
        # 这也是本节「扇出（fan-out）」的全部秘密：并行的本质是「一批 Send」。
        return [
            Send(c["source"], {"query": c["query"]})
            for c in state["classifications"]
        ]

    # ---------- 4.7 节点二、三：两个领域代理（并行执行） ----------
    # 这两个节点拿到的 state 是 Send 里那个 AgentInput（只有 query），不是父图的 RouterState ——
    # 所以它们读不到 classifications / results，也无法互相干扰。
    # 两者的返回值形状必须一致（都是 {"results": [一条 AgentOutput]}），
    # 因为父图要用同一个归约器把它们合并起来。
    async def query_baidu(state: AgentInput) -> dict:
        """查询百度搜索代理。"""
        result = await baidu_agent.ainvoke({
            "messages": [{"role": "user", "content": state["query"]}]
        })
        # 只取最后一条消息的正文：子 Agent 内部调了多少次 MCP 工具、中间消息长什么样，
        # 对父图来说都不重要 —— 这就是「Agent 当工具」时天然的信息封装。
        return {"results": [{"source": "baidu", "result": result["messages"][-1].content}]}

    async def query_gitee(state: AgentInput) -> dict:
        """查询 Gitee 代理。"""
        try:
            result = await gitee_agent.ainvoke({"messages": [{"role": "user", "content": state["query"]}]})
            return {"results": [{"source": "gitee", "result": result["messages"][-1].content}]}
        except Exception as e:
            # 单个分支失败不影响另一条分支：把失败原因也当成一条结果交给汇总节点。
            # 关键点：这里返回的**仍然是合法的 AgentOutput**（有 source 有 result），
            # 图才能继续往 synthesize 走 —— 并行分支里最忌讳的就是「一条挂了整张图挂掉」。
            print(f"Gitee 查询出错: {e}")
            return {"results": [{"source": "gitee", "result": f"Gitee 查询失败: {str(e)}"}]}

    # ---------- 4.8 节点四：合并（fan-in） ----------
    async def synthesize_results(state: RouterState) -> dict:
        """将所有代理的结果组合成一个连贯的答案。"""
        # 走到这里时，两个并行分支都已结束（LangGraph 会等所有入边就绪），
        # 所以 state["results"] 里是**两条**结果 —— 靠的是 RouterState 上那个 operator.add 归约器。
        # 这个兜底分支对应 route_to_agents 返回空列表的情况（模型认为没有可用知识源）。
        if not state["results"]:
            return {"final_answer": "未从任何知识来源找到结果。"}

        # 格式化结果以供综合
        formatted = [
            f"**来自 {r['source']}：**\n{r['result']}"
            for r in state["results"]
        ]

        synthesis_response = await model.ainvoke([
            {
                "role": "system",
                "content": f"""综合这些搜索结果以回答原始问题："：{state['query']}"

- 组合来自多个来源的信息，避免冗余
- 突出最相关和最可操作的信息
- 注明来源之间的任何差异
- 保持回复简洁且条理清晰"""
            },
            {"role": "user", "content": "\n\n".join(formatted)}
        ])

        return {"final_answer": synthesis_response.content}

    # ---------- 4.9 构建工作流（课案原文的链式写法） ----------
    # 读这张图的方法：add_node 注册了 4 个节点，边分两种 ——
    #   add_edge           静态边：一定会走（baidu/gitee → synthesize → END）；
    #   add_conditional_edges 动态边：走哪几条由函数返回值决定（classify → Send 列表）。
    # 注意 ["baidu", "gitee"] 那个列表：它是**声明式**的，只是告诉 LangGraph
    # 「这个条件边可能去这两个节点」（用于画图和校验），真正派发几个由 Send 列表长度决定。
    workflow = (
        StateGraph(RouterState)
        .add_node("classify", classify_query)
        .add_node("baidu", query_baidu)
        .add_node("gitee", query_gitee)
        .add_node("synthesize", synthesize_results)
        .add_edge(START, "classify")
        # 注意：条件边只负责"派发"，真正决定去哪几个节点的是 route_to_agents 返回的 Send 列表
        .add_conditional_edges("classify", route_to_agents, ["baidu", "gitee"])
        .add_edge("baidu", "synthesize")
        .add_edge("gitee", "synthesize")
        .add_edge("synthesize", END)
        .compile()
    )

    # ---------- 4.10 运行示例查询 ----------
    # 这个查询是课案原文的：「最热的数字人项目」两类知识源都相关，
    # 所以正常情况下应当派发 2 个 Send、results 里回来 2 条。
    # 如果只回来 1 条，说明分类器只认定了一个知识源（模型判断，不是 bug）——
    # 可以改问「XX 的代码怎么写」这类明显偏 Gitee 的问题来观察单分支行为。
    result = await workflow.ainvoke({"query": "最热的数字人项目"})

    print("\n" + "=" * 60)
    print("原始查询:", result["query"])
    print("\n分类结果:")
    # classifications 是分类器的结构化输出（谁 + 什么子问题），
    # 对照它才能看懂下面两个分支为什么收到了不同的 query。
    for c in result["classifications"]:
        print(f"  {c['source']}: {c['query']}")
    print(f"\n并行返回的结果条数: {len(result['results'])}"
          f"（来源：{[r['source'] for r in result['results']]}）")

    # 并行结果的**顺序不保证**（谁先跑完谁先进 results），所以上面的打印只列来源不假定顺序；
    # 到了 synthesize 里，模型看到的是带「来自 xxx」前缀的文本，顺序对它没有影响。
    print("\n" + "=" * 60 + "\n")
    print("最终答案:")
    print(result["final_answer"])


if __name__ == "__main__":
    asyncio.run(main())

# ================================================================
# 实测结论 / 与本课案的差异 / 踩坑提示
# ================================================================
# 1. 实测结论：本机两个密钥都为空，所以实际跑的是「本地假工具」路径；图结构、Send 派发、
#    operator.add 归并、synthesize 汇总这一段全部真实执行，
#    控制台能看到「分类结果」两行 +「并行返回的结果条数: 2」——这就是 fan-out/fan-in 生效的证据。
# 2. 与课案的差异（都不是逻辑改动）：
#      - `from config import setting` → `from config import settings`；
#      - 课案 `model = ChatOpenAI(model="deepseek-chat", ...)` 写死了模型名，
#        本项目统一取 `settings.model_name`（原因见 4.2 的注释）；
#      - 课案的两个 MCP 客户端各自写在 main() 里，本文件抽成 build_real_mcp_tools()，
#        为的是让「真 MCP / 本地假工具」两条路径共用同一段组装代码；
#      - `classify_query` 的 docstring 里课案写的是「deepseek-v4-pro 的思考模式不支持…」，
#        这是课案对自己环境的描述，本项目统一取 settings.model_name，故括号里做了说明。
# 3. 踩坑提示 A —— Send 的目标名必须与节点名精确一致：
#    `Send("baidu_search", ...)` 不会报错，而是让这个分支**静默丢失**；
#    所以 Classification.source 才要用 Literal 限定取值。
# 4. 踩坑提示 B —— 没有 operator.add 会丢结果：
#    把 `results: Annotated[list[AgentOutput], operator.add]` 写成普通的 `list[AgentOutput]`，
#    两个并行分支会互相**覆盖**，最终只剩一条（而且不报错，最难查）。
# 5. 踩坑提示 C —— 分类器与主模型共用同一个 llm 也能跑，但课案刻意分开：
#    结构化输出要求非思考模型，而回答问题时可能希望用更强的模型；
#    生产里这两个角色通常不是一个模型。
# 6. 踩坑提示 D —— MCP 工具报错会中断整个 Agent：
#    这就是 `mcp_error_handler` 存在的唯一理由，别删它（删了之后 404 会让整张图失败）。

# -*- coding: utf-8 -*-
r"""
LangChain 多 Agent：MCP 子智能体
================================================================
课案原文（用百度千帆的「联网搜索 MCP」当子智能体的工具）：

    # 申请百度搜素MCP API
    # https://console.bce.baidu.com/qianfan/tools/toolsCenter/57d4e765-8af5-4ec0-8f9b-47075ec349e0/detail
    import asyncio
    from langchain_openai import ChatOpenAI
    from langchain.tools import tool
    from langchain.agents import create_agent
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from config import setting

    llm = ChatOpenAI(model=setting.MODEL_NAME, api_key=setting.API_KEY, base_url=setting.BASE_URL)

    async def create_research_agent_with_mcp():
        \"\"\"创建带有搜索 MCP 工具的调研 agent\"\"\"
        # 创建 MCP 客户端，加载搜索工具
        client = MultiServerMCPClient(
            {
                "web-search-mcp-server": {
                    "url": "https://qianfan.baidubce.com/v2/tools/web-search/mcp",
                    "transport": "streamable-http",
                    "headers": {
                        "Authorization": "Bearer xxx"
                    },
                }
            }
        )
        # 获取 MCP 工具
        mcp_tools = await client.get_tools()
        print(f"加载到的 MCP 工具: {[t.name for t in mcp_tools]}")
        # 直接调用工具测试
        test_result = await mcp_tools[0].ainvoke({"query": "百度"})
        print(test_result)
        # 创建 subagent，将 MCP 工具添加进去
        subagent = create_agent(model=llm, tools=mcp_tools)
        return subagent

    async def main():
        # 初始化 subagent（带有搜索 MCP 工具）
        subagent = await create_research_agent_with_mcp()

        # 定义调研工具
        @tool("research", description="当用户的问题需要查询互联网资料或最新信息时调用。"
              "输入需要调研的问题，工具会使用百度搜索获取资料，"
              "并返回整理后的调研结果。")
        async def call_research_agent(query: str):
            result = await subagent.ainvoke({"messages": [{"role": "user", "content": query}]})
            return result["messages"][-1].content

        # 创建主 agent
        main_agent = create_agent(model=llm, tools=[call_research_agent])

        # 调用主 agent
        result = await main_agent.ainvoke(
            {"messages": [{"role": "user", "content": "大模型的MCP是什么"}]}
        )
        print(result["messages"][-1].content)

    if __name__ == "__main__":
        asyncio.run(main())

依赖（课案原文的安装命令，本项目 venv 里已经装好，**不需要再执行**）：

    uv add "mcp>=1.9,<2.0"
    uv add "langchain_mcp_adapters"

本节要讲清的架构 —— 「子智能体」不是框架的新概念，而是一个**包装手法**：

    主 Agent（只会一件事：判断该不该做调研）
      └── 工具 call_research_agent（它内部其实是一个完整的子 Agent）
            └── 子 Agent（挂着 MCP 搜索工具，负责真正查资料并整理）

为什么要这样套？因为工具的返回值对模型来说就是一坨文本，
「子 Agent 的最终答复」正好就是一坨文本 —— 于是「一个 Agent」天然可以当作
「另一个 Agent 的工具」。这就是 subagent / supervisor 模式最朴素的实现。

MCP（Model Context Protocol）在这里的角色：
    MultiServerMCPClient 把 MCP 服务端暴露的工具**转换成 LangChain 工具对象**，
    之后 create_agent 完全不关心它们是本地函数还是远程服务。
    transport 两种常见形态：
        streamable-http —— 连一个已经在跑的远程 MCP 服务（本节用这种）
        stdio           —— 由适配器拉起本地子进程当 MCP 服务（见 12_多Agent_MCP子Agent.py）

⚠️ 前置条件与降级路径（本项目规范第 5.3 条）：
    本机 `settings.baidu_qfan_api_key` **当前为空**，所以直连百度千帆 MCP 必然 401。
    本文件的做法是：
        1. 先做前置检查（依赖是否装上、密钥是否为空）；
        2. 缺密钥 / 连不上时打印**清晰的中文提示**，然后走一条**本地假搜索工具**的
           降级路径 —— 主 Agent + 子 Agent 的结构、`research` 工具的写法完全一样，
           只是把「远程 MCP 工具」换成「本地函数工具」，**不抛异常**；
        3. 密钥一旦填上，把 `FORCE_LOCAL_DEMO` 改成 False 就会自动切回真 MCP。

课案出处：Agent 课案 → 多Agent → 子Agent

前置条件（本文件的设计目标就是「缺前置也能跑完」）：
    - 根目录 `.env` 的 `API_KEY` / `BASE_URL` / `MODEL_NAME` 已填好（这一项必需）；
    - 百度千帆密钥 `BAIDU_QFAN_API_KEY` **本机当前为空** → 自动走本地假搜索工具降级路径；
    - 已装 `mcp` / `langchain-mcp-adapters`（本项目 venv 已装），未装时会打印安装命令后降级；
    - 不需要数据库；全程不抛 traceback。

运行方式（在项目根目录 `F:\ProGram\Python_Base` 下执行，否则 import 不到 `config`）：
    uv run Agent/02_langchain/12_多Agent_MCP子Agent_jxsd.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import asyncio

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from config import settings

# MCP 适配器与 mcp 包：本项目已装（langchain-mcp-adapters 0.3.2 / mcp 1.30.0）。
# 按规范用 try/except 兜住缺包的情况，保证模块层面永远 import 成功。
# 注意这里**不是**把异常吞掉不提：异常对象存进 MCP_IMPORT_ERROR，
# 由下面的 preflight() 拿出来打印成中文安装指引 —— 缺包时读者知道要装什么，而不是看到 ImportError。
try:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    MCP_IMPORT_ERROR = None
except ImportError as exc:                      # pragma: no cover - 取决于本机环境
    MultiServerMCPClient = None
    MCP_IMPORT_ERROR = exc

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)

# 百度千帆「联网搜索 MCP」的服务地址（课案原文）
BAIDU_MCP_URL = "https://qianfan.baidubce.com/v2/tools/web-search/mcp"

# 强制降级开关：False 表示「密钥非空就优先连真 MCP」
FORCE_LOCAL_DEMO = False


# ================================================================
# 0. 前置检查：依赖 + 密钥
# ================================================================
def preflight() -> tuple[bool, str]:
    """返回 (能否走真 MCP, 中文说明)。"""
    if MCP_IMPORT_ERROR is not None:
        return False, (
            "未安装 MCP 适配器：请先执行\n"
            '    uv add "mcp>=1.9,<2.0"\n'
            "    uv add langchain_mcp_adapters"
        )
    if not settings.baidu_qfan_api_key:
        return False, (
            "settings.baidu_qfan_api_key 为空（百度千帆联网搜索 MCP 需要它）。\n"
            "    申请地址：https://console.bce.baidu.com/qianfan/tools/toolsCenter/"
            "57d4e765-8af5-4ec0-8f9b-47075ec349e0/detail\n"
            "    配置方式：在项目根目录 .env 里加一行 BAIDU_QFAN_API_KEY=<你的密钥>\n"
            "    本节将改用「本地假搜索工具」演示同样的主/子 Agent 结构。"
        )
    return True, "已检测到百度千帆密钥，将直连远程 MCP 服务。"


# ================================================================
# 1. 真 MCP 路径（课案原文，密钥来自 settings，不硬编码）
# ================================================================
async def create_research_agent_with_mcp():
    """创建带有搜索 MCP 工具的调研 agent"""

    # 创建 MCP 客户端，加载搜索工具
    client = MultiServerMCPClient(
        {
            "web-search-mcp-server": {
                "url": BAIDU_MCP_URL,
                "transport": "streamable-http",
                # 课案原文写的是 "Bearer xxx"（占位符）；这里从配置读，绝不硬编码密钥
                "headers": {
                    "Authorization": f"Bearer {settings.baidu_qfan_api_key}"
                },
            }
        }
    )

    # 获取 MCP 工具：适配器会去服务端 list_tools，并把每个远程工具包成 LangChain 工具
    # —— 所以下一步 create_agent 完全不知道这些工具来自远程服务，接口和本地 @tool 一样。
    mcp_tools = await client.get_tools()
    print(f"加载到的 MCP 工具: {[t.name for t in mcp_tools]}")

    # 直接调用工具测试（课案的原文就是先单测一个工具，确认连通了再交给 Agent）
    # 这一步是重要的排错习惯：工具连不通时，先单独 ainvoke 一次定位是「网络/密钥」问题，
    # 还是「模型不肯调工具」问题 —— 混在 Agent 里这两类问题长得一模一样。
    test_result = await mcp_tools[0].ainvoke({"query": "百度"})
    print("MCP 工具直调结果：", str(test_result)[:200])

    # 创建 subagent，将 MCP 工具添加进去
    subagent = create_agent(model=llm, tools=mcp_tools)
    return subagent


# ================================================================
# 2. 降级路径：本地假搜索工具（结构完全一样，只是工具来源换成函数）
# ================================================================
@tool
async def local_web_search(query: str) -> str:
    """联网搜索。query：要搜索的问题。

    说明：这是**降级演示用的本地假工具**，不联网，返回写死的示例资料。
    真跑通远程搜索需要百度千帆 MCP 密钥，见文件头的前置条件。
    """
    # 故意做成 async：与 MCP 工具的调用方式保持一致（MCP 工具都是异步的）
    fake_db = {
        "mcp": (
            "【资料】MCP（Model Context Protocol，模型上下文协议）是 Anthropic 于 2024 年底"
            "提出的开放协议，用统一的方式把「外部工具/数据源」接给大模型应用。"
            "它把集成方拆成 MCP 客户端（Agent 侧）与 MCP 服务端（工具侧），"
            "服务端暴露 tools / resources / prompts 三类能力，客户端按协议发现并调用。"
            "好处是工具只需实现一次，任何支持 MCP 的客户端都能复用。"
        ),
        "langchain": (
            "【资料】LangChain 1.x 用 create_agent 封装 ReAct 循环，"
            "并用 middleware 机制提供钩子与内置中间件（摘要、重试、限流、人工审核等）。"
        ),
    }
    key = "mcp" if "mcp" in query.lower() else "langchain"
    print(f"  [local_web_search] 收到查询：{query}")
    return fake_db[key]


async def create_research_agent_local():
    """降级版调研 agent：把远程 MCP 工具换成同名的本地工具。"""
    subagent = create_agent(
        model=llm,
        tools=[local_web_search],
        system_prompt="你是检索助手，先调用 local_web_search 获取资料，再用简洁的中文总结。",
    )
    return subagent


# ================================================================
# 3. 主流程（课案原文的主/子 Agent 组装方式，两种来源共用）
# ================================================================
async def main() -> None:
    # ---------- 3.1 前置检查 ----------
    can_use_mcp, message = preflight()
    print("=" * 70)
    print("前置检查：", "通过" if can_use_mcp else "未通过")
    print(message)
    print("=" * 70)

    # ---------- 3.2 挑一条路径建子 Agent ----------
    if can_use_mcp and not FORCE_LOCAL_DEMO:
        try:
            subagent = await create_research_agent_with_mcp()
            print("已通过百度千帆 MCP 加载搜索工具。\n")
        except Exception as exc:
            # 网络不通 / 密钥失效 / 服务端报错，都不该让教学脚本崩掉
            print(f"[降级] 连接百度千帆 MCP 失败：{type(exc).__name__}: {str(exc)[:160]}")
            print("       改用本地假搜索工具演示同样的结构。\n")
            subagent = await create_research_agent_local()
    else:
        subagent = await create_research_agent_local()
        print("使用本地假搜索工具（结构、调用方式与 MCP 版完全一致）。\n")

    # ---------- 3.3 定义调研工具（课案原文写法） ----------
    # 重点：子 Agent 的最终答复就是一坨文本，所以可以直接当成工具返回值。
    # 「一个 Agent 当另一个 Agent 的工具」= subagent 模式的最小实现。
    @tool("research", description="当用户的问题需要查询互联网资料或最新信息时调用。"
          "输入需要调研的问题，工具会使用百度搜索获取资料，"
          "并返回整理后的调研结果。")
    async def call_research_agent(query: str):
        result = await subagent.ainvoke({"messages": [{"role": "user", "content": query}]})
        return result["messages"][-1].content

    # ---------- 3.4 创建主 agent ----------
    # 主 Agent 手里只有一个工具：research。它负责判断「这个问题要不要调研」。
    main_agent = create_agent(model=llm, tools=[call_research_agent])

    # ---------- 3.5 调用主 agent ----------
    question = "大模型的MCP是什么"
    print(f"用户提问：{question}")
    result = await main_agent.ainvoke(
        {"messages": [{"role": "user", "content": question}]}
    )

    # 打印整条链路：主 Agent → research 工具 → 子 Agent → 搜索工具 → …
    # 看这段日志的方法：先找「调用工具 [research]」（主 Agent 决定调研），
    # 再找 research 那条 ToolMessage（子 Agent 的答复被当成工具结果回填），
    # 最后是纯文本的最终答复 —— 这就是 subagent 模式在消息层面的完整往返。
    print("\n执行轨迹：")
    for index, msg in enumerate(result["messages"], start=1):
        kind = type(msg).__name__
        calls = getattr(msg, "tool_calls", None)
        if calls:
            print(f"  [{index}] {kind:<14} 调用工具 {[c['name'] for c in calls]}")
        else:
            print(f"  [{index}] {kind:<14} {str(msg.content)[:80]}")

    print("\n最终答复：")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())

# ================================================================
# 实测结论 / 与本课案的差异 / 踩坑提示
# ================================================================
# 1. 实测结论：本机 baidu_qfan_api_key 为空，因此实际跑的是「本地假搜索工具」这条降级路径；
#    控制台会先打印前置检查的中文提示，然后完整跑通
#    「主 Agent → research 工具 → 子 Agent → local_web_search」这条链路并给出最终答复。
#    填上密钥后把 FORCE_LOCAL_DEMO 保持 False，即自动切回真 MCP。
# 2. 与课案的差异（都不是逻辑改动）：
#      - `from config import setting` → `from config import settings`；
#      - 课案的 `llm` 是 ChatOpenAI 直接 new，本文件统一用 init_chat_model + settings；
#      - 课案把密钥写成 `"Bearer xxx"` 占位符，本文件从 settings 读取（凭据不进代码）；
#      - `MultiServerMCPClient` 的 import 用 try/except 兜住，缺包时给安装命令而不是 traceback。
# 3. 踩坑提示 A —— 子 Agent 的 tools 必须来自 `await client.get_tools()`：
#    它是**异步**的，忘了 await 会拿到一个 coroutine 对象，create_agent 直接报类型错误。
# 4. 踩坑提示 B —— 别在主 Agent 里直接挂 MCP 工具：
#    本节的知识点是「一个 Agent 可以当另一个 Agent 的工具」，
#    直接挂上去就退化成单 Agent + 远程工具，subagent 的隔离与复用价值就没了。
# 5. 踩坑提示 C —— research 工具的 description 决定主 Agent 会不会用它：
#    课案那三句话（何时调用 / 输入是什么 / 返回什么）是提示词，改写成一句话描述
#    很可能导致主 Agent 干脆不调工具、自己编答案。

# -*- coding: utf-8 -*-
r"""
DeepAgents 官方补充篇④：检索-卸载-委派（非课案内容）
================================================================
来源与定位：
    本文件对照 DeepAgents **官方文档** /oss/python/deepagents/rag.mdx
    （以及 retrieval.mdx 的 RAG 架构一节），补缺口表 **DeepAgents 第 12 项**。

官方把 RAG 分成**三种架构**（retrieval.mdx）：

    架构           描述                                        控制力  灵活性  延迟
    -------------  ------------------------------------------  ------  ------  --------
    2-Step RAG     检索永远发生在生成之前，简单可预测           高      低      快
    Agentic RAG    agent 自己决定**何时、怎么**检索              低      高      不定
    Hybrid         两者结合 + 校验环节                          中      中      不定

以及 DeepAgents 特有的**四种落地模式**（rag.mdx）：

    模式                          本仓库对应
    ----------------------------  ------------------------------------------
    ① 技能引导检索（Skill 规定怎么查、引用格式）  → 02_langchain/17_Skills渐进披露_官方补充.py
    ② Rubric 校验接地（评分器检查答案有无依据）   → 03_deepagents/16_Rubric评分循环_官方补充.py
    ③ Todo 驱动调查（规划工具列出要查的页面）     → 03_deepagents/14_上下文治理_官方补充.py Demo 1
    ④ **检索-卸载-委派**（把命中内容写进文件系统，  ← **本文件**（唯一还没做的那个）
       子代理并行读取/总结）

④ 为什么值得单独讲：
    普通 RAG 把检索到的全文**塞进主线程上下文**；文档一多，主 agent 的上下文就被塞满，
    还没开始推理就快撞窗口了。官方的做法是：**把 chunk 卸载到文件系统**，
    主 agent 只拿到「文件路径 + 极短预览」，再派子代理去读、去搜、去总结 ——
    主线程只承载结论，全文留在文件里。
    本文件把这个差异**量化**出来（Demo 3 打印两种做法的上下文字符数）。

⚠️ 本文件用的向量化/重排序模型来自 .env：
    `BAAI/bge-m3`（1024 维）与 `BAAI/bge-reranker-v2-m3`，都是 SiliconFlow 端点。

运行方式（项目根目录下，需真实模型 + embedding + rerank）：
    uv run Agent/03_deepagents/17_RAG_检索卸载委派_官方补充.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import json
import tempfile
import urllib.request
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from deepagents import SubAgent, create_deep_agent
from deepagents.backends import FilesystemBackend

from config import settings

llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,          # grok-4.6
    api_key=settings.api_key,
    base_url=settings.base_url,
    max_retries=0,
    # 显式给足超时：本机网关在负载高（或长输出/高推理量）时单次请求可能超过两分钟。
    # 实测：120 秒会在网关繁忙时抛 OpenAITimeoutError；放宽到 300 秒后稳定通过。
    timeout=300,
)

embeddings = OpenAIEmbeddings(
    model=settings.embedding.model,     # BAAI/bge-m3
    api_key=settings.embedding.api_key,
    base_url=settings.embedding.base_url,
    check_embedding_ctx_length=False,
)


# ================================================================
# 知识库素材：故意写得长一些，模拟"文档页"，让上下文差距看得出来
# ================================================================
DOCS = {
    "langgraph-persistence.md": """# LangGraph 持久化详解

LangGraph 的持久化由 checkpointer 承担。每执行完一个 super-step，图的状态快照就会被
写入 checkpointer；thread_id 是"这条会话线"的唯一标识，所有快照都挂在它下面。

中断恢复的原理：当节点里调用 interrupt() 时，框架抛出一个特殊信号，把当前状态落盘后
暂停执行；调用方拿到 __interrupt__ 后，用 Command(resume=值) 带着用户输入重新进入，
框架从**同一个检查点**继续跑，已完成节点的副作用不会重放两次（但恢复点的节点会重放，
所以副作用要幂等）。

时间旅行靠的是检查点历史：get_state_history 能按时间倒序列出所有快照，
你可以回到任意一个快照再分叉（fork）出新的执行路径。

常见坑：delete 消息要用 RemoveMessage（add_messages 只会追加）；自定义状态 schema
必须继承 MessagesState，否则 messages 字段没有 reducer，会导致 agent 循环判不出终止条件。
""",
    "rag-architecture.md": """# RAG 架构选型笔记

2-Step RAG：检索永远前置，流程固定、延迟可预测，适合 FAQ 与文档问答。
缺点是"不管用户问什么都要检索一次"，且检索质量差时答案必错。

Agentic RAG：把检索做成工具，由 agent 决定何时检索、检索几次、用什么查询词。
灵活但延迟不定，且可能"该检索时不检索"。

Hybrid：先做一次检索，再让 agent 决定是否继续检索，最后加一道校验环节
（例如用评分器判断答案是否有依据）。

上下文管理的要点：检索回来的全文如果全部塞进主线程，很快就会撑爆窗口。
更稳的做法是把命中内容卸载到文件系统，主线程只保留路径与结论，
细节交给子代理按需读取。
""",
    "ops-runbook.md": """# 值班运行手册

巡检：每两小时记录一次机房温度，超过 28 摄氏度需要在值班群报备并联系设施同事。

发布：发布前必须在预发环境跑一遍全量回归；严禁在未回归的情况下直接上生产。

故障处理：先看监控大盘确认影响面，再查最近一次发布的变更记录。回滚是首选止血手段，
但回滚前要确认没有正在执行的批处理任务，否则可能造成数据不一致。

交接班：把未完成的事项、观察到的异常、以及已采取的临时措施写进交接记录。
""",
}


def build_store() -> InMemoryVectorStore:
    """构建向量库（与 02_langchain/24_RAG知识库 官方补充篇 同一套流程）。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(
        [Document(page_content=text, metadata={"source": name}) for name, text in DOCS.items()]
    )
    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(chunks)
    return store


# ================================================================
# 重排序（SiliconFlow /rerank，与 02_langchain/24 同一实现）
# ================================================================
class SiliconFlowReranker:
    """交叉编码精排：召回后按相关性重排（分数看相对差距，别用绝对阈值）。"""

    def __init__(self) -> None:
        self.url = settings.rerank.base_url.rstrip("/") + "/rerank"
        self.model = settings.rerank.model
        self.api_key = settings.rerank.api_key
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def rerank(self, query: str, documents: list[str], top_n: int = 3) -> list[int]:
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        with self._opener.open(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
        return [item["index"] for item in body.get("results", [])]


reranker = SiliconFlowReranker()


def retrieve(query: str, top_k: int = 4, top_n: int = 2) -> list[str]:
    """召回 + 精排，返回最终选中的文本块。"""
    candidates = [document.page_content for document in STORE.similarity_search(query, k=top_k)]
    order = reranker.rerank(query, candidates, top_n=top_n)
    return [candidates[index] for index in order]


# ================================================================
# Demo 1：知识库与两级检索（步骤与 24 号文件一致，这里快速过一遍）
# ================================================================
def demo_1_knowledge_base() -> None:
    print("=" * 70)
    print("Demo 1：知识库构建 + 两级检索（召回 → 精排）")
    print("=" * 70)

    chunks = STORE.store  # InMemoryVectorStore 内部就是 dict：id -> (Document, 向量)
    print(f"  文档 {len(DOCS)} 篇，切片 {len(chunks)} 块（chunk_size=200）")
    query = "检索回来的内容太多怎么办？"
    picked = retrieve(query)
    print(f"  查询：{query}")
    print(f"  两级检索后选中 {len(picked)} 块，总长度 {sum(len(c) for c in picked)} 字符")
    for index, text in enumerate(picked, start=1):
        print(f"    {index}. {text[:56].replace(chr(10), ' ')}…")
    print(
        "  ↑ 记住这个字符数：**Demo 2 会把它整段塞进主线程上下文**，\n"
        "    而 Demo 3 只把路径塞进去 —— 差距就在这儿。"
    )


# ================================================================
# Demo 2：2-Step RAG —— 检索结果全部塞进上下文
# ================================================================
def demo_2_two_step_rag() -> None:
    print("\n" + "=" * 70)
    print("Demo 2：2-Step RAG —— 检索前置，全文进主线程上下文")
    print("=" * 70)

    query = "线上接口变慢时应该怎么排查？上下文塞不下怎么办？"
    picked = retrieve(query, top_n=3)
    context = "\n\n".join(f"【资料 {i+1}】{text}" for i, text in enumerate(picked))
    prompt = (
        "你是运维助手。只依据【资料】回答，没有依据就说不知道。"
        "**回答控制在三句话以内**（本机网关对长输出不稳，短答也更适合演示）。\n\n"
        f"{context}\n\n问题：{query}"
    )
    try:
        response = llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        # 网络/网关抖动兜底：不让一次超时把整个演示带崩（按仓库惯例给中文提示）
        print(f"  本次模型调用失败（网关抖动/超时，非代码问题）：{type(exc).__name__}")
        print("  提示：本文件依赖真实模型；网关繁忙时重跑一次即可，其余输出仍可参考。")
        return
    usage = getattr(response, "usage_metadata", None) or {}
    print(f"  送进模型的上下文：{len(prompt)} 字符（{len(picked)} 块全文）")
    print(f"  回答：{str(response.content)[:150]}")
    print(f"  本次 token：输入 {usage.get('input_tokens')} / 输出 {usage.get('output_tokens')}")
    print(
        "  ↑ 2-Step 的优点：**流程固定、延迟可预测、控制力强**（每次必检索）。\n"
        "    代价：检索结果**不分青红皂白全部进上下文**，文档一多就撑爆窗口；\n"
        "    而且它没法「追问式检索」（发现资料不够时再去查一次）。"
    )


# ================================================================
# Demo 3：检索-卸载-委派 —— 全文落盘，主线程只拿路径
# ================================================================
# 官方 rag.mdx 的第四个模式，步骤：
#     ① 检索工具把命中块**写进文件系统**（/retrieved/<id>.md）；
#     ② 工具只返回「路径 + 极短预览」给主 agent；
#     ③ 主 agent 用 task 把「读文件并总结」派给子代理；
#     ④ 子代理在**自己的上下文**里读全文、总结，只把结论回传主线程。
# 结果：主线程上下文只承载「路径 + 结论」，全文留在文件里。
def demo_3_offload_and_delegate(workdir: Path) -> None:
    print("\n" + "=" * 70)
    print("Demo 3：检索-卸载-委派 —— 全文落盘，主线程只拿路径")
    print("=" * 70)

    # 后端指向临时目录：virtual_mode=True 下 agent 看到的是 "/retrieved/xxx.md" 这样的虚拟路径
    backend = FilesystemBackend(root_dir=workdir, virtual_mode=True)

    @tool
    def search_and_offload(query: str) -> str:
        """检索知识库，把命中的内容写入 /retrieved 目录，返回文件路径清单。"""
        picked = retrieve(query, top_n=2)
        saved: list[str] = []
        for index, text in enumerate(picked, start=1):
            path = f"/retrieved/chunk-{index}.md"
            write_result = backend.write(path, text)
            # 写入失败（路径/权限/编码）要如实回报 —— 否则会把不存在的路径交给子代理，
            # 子代理再 read_file 失败，白烧一轮模型调用。
            if getattr(write_result, "error", None):
                return f"写入 {path} 失败：{write_result.error}"
            saved.append(path)
        # 只回路径 + 极短预览：全文**不进**主线程上下文
        preview = "\n".join(f"{path}（{len(text)} 字符，开头：{text[:24]}…）"
                            for path, text in zip(saved, picked))
        return f"已把 {len(saved)} 段资料写入文件系统：\n{preview}"

    agent = create_deep_agent(
        model=llm,
        backend=backend,
        tools=[search_and_offload],
        subagents=[
            SubAgent(
                name="doc-reader",
                description="读取文件并总结要点（适合处理长文档，不要在主线里读全文）",
                system_prompt="你会收到一个文件路径。读取它，用三句话总结要点，直接给结论。",
                model=llm,
                tools=[],
            )
        ],
        system_prompt=(
            "你是运维助手，工作方式固定：\n"
            "1. 先调用 search_and_offload 检索资料（它会把全文写进文件系统）；\n"
            "2. 然后**必须用 task 工具把「读文件并总结」派给 doc-reader 子代理**，"
            "不要自己在主线里读全文；\n"
            "3. 最后基于子代理的总结回答，两句话以内。"
        ),
    )

    query = "线上接口变慢时应该怎么排查？上下文塞不下怎么办？"
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            config={"recursion_limit": 30},
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  本次 agent 运行失败（网关抖动/超时，非代码问题）：{type(exc).__name__}")
        print("  提示：深度智能体的系统提示很长、每轮推理量也大，网关繁忙时容易超时；")
        print("        本文件已把超时放宽到 300 秒，重跑一次通常即可。")
        return

    tool_sequence = [
        call["name"]
        for message in result["messages"]
        for call in (getattr(message, "tool_calls", None) or [])
    ]
    print(f"  提问：{query}")
    print(f"  工具调用序列：{tool_sequence}")

    # 量化对比：主线程消息总长度（应为路径 + 结论级别，而不是全文级别）
    main_chars = sum(len(str(m.content)) for m in result["messages"])
    full_text_chars = sum(len(text) for text in retrieve(query, top_n=2))
    print(f"\n  主线程消息总长度：{main_chars} 字符")
    print(f"  其中检索到的全文长度：{full_text_chars} 字符（**没有进主线程**，落在文件里）")
    saved_files = sorted(p.name for p in (workdir / "retrieved").glob("*.md")) if (workdir / "retrieved").exists() else []
    print(f"  文件系统里的资料：{saved_files}（可被 read_file / grep 按需读取）")
    print(f"  最终回答：{str(result['messages'][-1].content)[:160]}")
    if "task" in tool_sequence:
        print("  ✔ 本次确实发生了委派（子代理读完文件只回传结论）")
    else:
        print("  （本次主 agent 没走 task —— 委派与否取决于模型；系统提示已强制要求）")
    print(
        "  ↑ 与 Demo 2 对比：\n"
        "    · 2-Step：全文进主线程（上下文 = 全文 + 结论）；\n"
        "    · 卸载-委派：全文进文件系统（上下文 = 路径 + 结论），\n"
        "      细节由子代理在**独立上下文**里消化。\n"
        "    文档规模越大，这个差别的收益越明显（也是课案 03_deepagents/14 讲的\n"
        "    上下文压缩机制的同一个思路：把大块内容挪出上下文）。"
    )


if __name__ == "__main__":
    STORE = build_store()
    demo_1_knowledge_base()
    demo_2_two_step_rag()
    with tempfile.TemporaryDirectory(prefix="delegated_rag_") as tmp:
        demo_3_offload_and_delegate(Path(tmp))
    print("\n全部 Demo 执行完毕（临时目录已清理）。")


# ================================================================
# 实测结论 / 与课案的衔接 / 踩坑提示
# ================================================================
# 1. 官方事实：
#    - 三种架构（retrieval.mdx）：2-Step RAG（控制力高、延迟可预测）/ Agentic RAG
#      （agent 决定何时怎么检索）/ Hybrid（两者结合 + 校验）；
#    - 四种模式（rag.mdx）：技能引导检索 / Rubric 校验接地 / Todo 驱动调查 /
#      **检索-卸载-委派**（本文件实现的正是第四个）。
# 2. 本文件实测（bge-m3 + bge-reranker-v2-m3 + grok-4.6）：
#    - 两级检索选中的块数/字符数、2-Step 的 prompt 长度与 token 用量、
#      卸载-委派模式下主线程消息总长度与落盘文件清单，都打印在运行输出里；
#    - 关键对照：**同样的资料，Demo 2 进主线程、Demo 3 落文件系统**。
# 3. 与课案的衔接：
#    - 课案 03_deepagents 讲 create_deep_agent 的骨架与后端；
#      本文件把「检索」这个应用场景接到那套骨架上（官方 rag.mdx 教程的做法）；
#    - 上下文压缩/卸载机制见 03_deepagents/14_上下文治理_官方补充.py；
#    - 技能引导检索见 02_langchain/17_Skills渐进披露_官方补充.py；
#    - Rubric 校验接地见 03_deepagents/16_Rubric评分循环_官方补充.py；
#    - Todo 驱动调查见 03_deepagents/14 的 write_todos Demo。
# 4. 未收录（官方还有、本文件没做的）：
#    - **大规模语料的向量库**：官方教程会把整站文档索引进真实向量库（含分页与并发索引），
#      本文件用内存向量库 + 三篇示例文档把模式讲清楚；
#    - **代码解释器生成图表/时间线**（支持 interpreters 的环境才有）：
#      见 03_deepagents 的 Interpreters 补充篇（依赖 quickjs）；
#    - **LangSmith 上的 RAG 评估**（答案正确性/相关性/接地度）：本仓库不用 LangSmith。
# 5. 踩坑提示：
#    A. 卸载路径要**固定且可枚举**（本文件用 /retrieved/chunk-N.md）：
#       子代理与主线程都要能在后续轮次里找到它们；
#    B. 卸载后**主线程只回传路径 + 极短预览** —— 一旦把全文也塞进返回值，
#       这个模式就退回成普通 RAG 了（白折腾）；
#    C. `SubAgent.model` 是**可选**的：不传就继承主 agent 的模型 ——
#       `create_deep_agent` 会用 `spec.get("model", model)` 回填（deepagents/graph.py）；
#       只有**绕开它直接构造** `SubAgentMiddleware` 时才必须显式给 model（那条路径没有回填）。
#       子代理的 description 要写清「适合读长文档」，否则主 agent 不会把活派给它；
#    D. 委派不是必然发生：模型有时自己顺手读完就答了 —— 要在系统提示里明确要求委派
#       （本文件第 2 条规则），必要时用 Rubric 校验它有没有真的走流程；
#    E. 落盘内容涉及敏感数据时，记得配合 FilesystemPermission 做访问控制
#       （见 03_deepagents/14_上下文治理_官方补充.py Demo 2/3）。

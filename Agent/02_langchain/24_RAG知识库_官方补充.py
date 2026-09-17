# -*- coding: utf-8 -*-
r"""
LangChain 官方补充篇：RAG 知识库（非课案内容，故不带 _jxsd 后缀）
================================================================
来源与定位：
    本文件对照 LangChain **官方文档** /oss/python/langchain/knowledge-base.mdx
    （以及 deepagents/retrieval.mdx 的"检索增强"思想），补缺口表 **LangChain 第 2 项**。

官方这篇讲的是「让 agent 能查自己的知识」的标准流程：
    加载文档 → 切分 → 向量化 → 存向量库 → 检索 → （可选）重排序 → 交给模型作答。

⚠️ 本文件与本机配置的对应关系（**这是与前几篇补充篇最大的不同**）：
    · 向量化模型：`EMBEDDING_MODEL=BAAI/bge-m3`（SiliconFlow，1024 维，实测单次 0.12~0.3 秒）
    · 精排模型  ：`RERANK_MODEL=BAAI/bge-reranker-v2-m3`（SiliconFlow，实测 0.2 秒）
    · 作答模型  ：`MODEL_NAME=grok-4.6`（.env 里的 Agent 课案扁平字段）
    三者都在 `.env` 里配置，代码统一走 `from config import settings`（课案约定）。

**为什么必须有"重排序"这一步**（本文件的核心教学点）：
    向量检索是**双塔**结构 —— query 与文档各自编码成向量再算距离，所以快、能扫百万级，
    但精度有限（两者从没见过面）；
    重排序是**交叉编码** —— query 和文档拼在一起送进模型逐对打分，准得多但慢，
    所以只用来精排少量候选。
    生产标配：**向量召回 top-20 → rerank 精排 top-5 → 交给模型**。本文件 Demo 3 会
    把"召回顺序"和"精排顺序"打印出来对照，差异一眼可见。

运行方式（项目根目录下，会真实调用 embedding / rerank / 作答三个模型）：
    uv run Agent/02_langchain/24_RAG知识库_官方补充.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import json
import tempfile
import time
import urllib.request
from pathlib import Path

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

# 作答模型：用 .env 里的 Agent 课案扁平字段（MODEL_NAME=grok-4.6）
llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
    max_retries=0,
)

# 向量化模型：SiliconFlow 的 OpenAI 兼容端点（bge-m3）
embeddings = OpenAIEmbeddings(
    model=settings.embedding.model,
    api_key=settings.embedding.api_key,
    base_url=settings.embedding.base_url,
    check_embedding_ctx_length=False,   # 兼容第三方端点的必要开关（见文末踩坑）
    # 显式超时：端点慢时快速失败，而不是无限挂住（实测踩过：没设超时时整跑会卡死）
    request_timeout=60,
    max_retries=1,
)


# ================================================================
# 知识库素材：现场生成几篇小文档（真实项目里是 PDF/网页/数据库）
# ================================================================
DOCS: dict[str, str] = {
    "langgraph.md": (
        "# LangGraph 基础\n\n"
        "LangGraph 用节点和边描述流程：节点是普通函数，边决定执行顺序。\n\n"
        "持久化由 checkpointer 负责，thread_id 用来区分不同会话，支持中断恢复与时间旅行。\n\n"
        "状态用 TypedDict 定义，列表字段要配 reducer（如 operator.add），否则后写会覆盖先写。\n"
    ),
    "rag.md": (
        "# RAG 检索增强\n\n"
        "RAG 的标准流程是：加载 → 切分 → 向量化 → 检索 → 重排序 → 交给模型作答。\n\n"
        "向量检索属于双塔结构，速度快但精度有限；重排序用交叉编码器逐对打分，精度高但更慢，"
        "因此只精排少量候选。\n\n"
        "常见坑：切分粒度太大导致检索命中但答不准；切分太小则上下文碎片化。\n"
    ),
    "ops.md": (
        "# 运维值班手册\n\n"
        "值班期间每两小时巡检一次机房温度，超过 28 摄氏度需要报备。\n\n"
        "重启服务前必须先确认没有正在执行的批处理任务，否则可能造成数据不一致。\n"
    ),
}

QUERIES = [
    "为什么要做重排序？它和向量检索有什么区别？",
    "LangGraph 的会话是怎么区分和恢复的？",
]


def build_knowledge_base(root: Path) -> tuple[InMemoryVectorStore, list[Document]]:
    """把素材写到磁盘 → 加载 → 切分 → 向量化入库，返回（向量库, 切片列表）。"""
    for name, content in DOCS.items():
        (root / name).write_text(content, encoding="utf-8")

    # ① 加载：这里直接读文件（生产常用 TextLoader / PyPDFLoader / 网页加载器）
    raw_documents = [
        Document(page_content=(root / name).read_text(encoding="utf-8"), metadata={"source": name})
        for name in sorted(DOCS)
    ]

    # ② 切分：中文场景建议显式给分隔符（默认分隔符里没有中文标点）
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=120,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(raw_documents)

    # ③ 向量化入库：InMemoryVectorStore 适合演示（生产用 Milvus，见课案 RAG 项目）
    store = InMemoryVectorStore(embedding=embeddings)
    store.add_documents(chunks)
    return store, chunks


# ================================================================
# 重排序组件：LangChain 没有内置 SiliconFlow 的 rerank，自己包一层
# ================================================================
class SiliconFlowReranker:
    """调用 SiliconFlow `/rerank` 的精排组件（**候选文本会发到云端打分**，注意数据合规）。

    对比：本地部署的交叉编码器（如 sentence-transformers + bge-reranker）数据不出机器，
    但要多一份显存/内存与运维成本；本文件用云端 API，换来零部署。

    实测注意（很重要，两次实测对照得出）：
        · 分数是 sigmoid 后的 [0, 1] 值，但**绝对高低取决于"语料里有没有真答案"**：
          本文件有合适文档时第一名 0.95、第二名 0.003（差 300 倍）；
          而探针里用一句和语料无关的查询时，最高分只有 0.0288。
        · 所以别用固定绝对阈值过滤（.env 里 RERANK_RELEVANCE_P=0.65 在这种尺度下会
          把「没有答案」的场景全部滤掉，也让「有答案」的场景显得过松）；
          **看 top1 与 top2 的相对差距**判断是否命中，阈值按自己的语料重标定。
        · 请求体是 `{model, query, documents, top_n}`，与 OpenAI 的 /embeddings 不同构，
          但同为 JSON + Bearer 鉴权。
    """

    def __init__(self, model: str, api_key: str, base_url: str, timeout: int = 60) -> None:
        self.model = model
        self.api_key = api_key
        self.url = base_url.rstrip("/") + "/rerank"
        self.timeout = timeout
        # 本机系统代理会接管部分请求，这里显式不走代理（内网课案环境同款处理）
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def rerank(self, query: str, candidates: list[str], top_n: int = 5) -> list[dict]:
        """对候选文本重排，返回 [{index, score}]（按分数从高到低）。"""
        payload = {
            "model": self.model,
            "query": query,
            "documents": candidates,
            "top_n": min(top_n, len(candidates)),
        }
        request = urllib.request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with self._opener.open(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
        return [
            {"index": item["index"], "score": item.get("relevance_score", 0.0)}
            for item in body.get("results", [])
        ]


reranker = SiliconFlowReranker(
    model=settings.rerank.model,
    api_key=settings.rerank.api_key,
    base_url=settings.rerank.base_url,
)


# ================================================================
# Demo 1：知识库构建（加载 → 切分 → 向量化）
# ================================================================
def demo_1_build(chunks: list[Document]) -> None:
    print("=" * 70)
    print("Demo 1：构建知识库 —— 加载 → 切分 → 向量化")
    print("=" * 70)

    print(f"  原始文档：{len(DOCS)} 篇，共 {sum(len(c) for c in DOCS.values())} 字符")
    print(f"  切分参数：chunk_size=120 / overlap=20（中文分隔符已显式指定）")
    print(f"  切片数量：{len(chunks)}")
    for index, chunk in enumerate(chunks[:3], start=1):
        source = chunk.metadata.get("source", "?")
        print(f"    切片 {index}（{source}）：{chunk.page_content[:40].replace(chr(10), ' ')}…")

    # 单独验证一次 embedding 端点，顺便拿到维度（bge-m3 = 1024 维）
    started = time.time()
    vector = embeddings.embed_query("测试向量维度")
    print(f"  向量维度：{len(vector)}（模型 {settings.embedding.model}），"
          f"单次向量化耗时 {time.time()-started:.2f}s")
    print(
        "  ↑ 三步里最容易出问题的是**切分**：粒度太大 → 检索命中但答不准；\n"
        "    太小 → 上下文碎片化。中文记得把「。」「；」写进 separators，\n"
        "    否则默认分隔符会按英文标点切，切出来的块语义不完整。"
    )


# ================================================================
# Demo 2：向量检索（召回）
# ================================================================
def demo_2_vector_search(store: InMemoryVectorStore) -> None:
    print("\n" + "=" * 70)
    print("Demo 2：向量检索 —— 召回的原始顺序")
    print("=" * 70)

    query = QUERIES[0]
    started = time.time()
    hits = store.similarity_search_with_score(query, k=5)
    print(f"  查询：{query}")
    print(f"  召回 {len(hits)} 条，耗时 {time.time()-started:.2f}s（含一次 query 向量化）")
    for rank, (document, score) in enumerate(hits, start=1):
        text = document.page_content.replace("\n", " ")[:46]
        print(f"    {rank}. 相似度 {score:.4f}｜{document.metadata.get('source')}｜{text}…")
    print(
        "  ↑ InMemoryVectorStore 返回的是**相似度**（越大越像）；\n"
        "    注意它和 rerank 的分数**不是一个尺度**，别混着比。"
    )


# ================================================================
# Demo 3：重排序 —— 召回顺序 vs 精排顺序（本文件的核心对照）
# ================================================================
def demo_3_rerank(store: InMemoryVectorStore) -> None:
    print("\n" + "=" * 70)
    print("Demo 3：重排序 —— 向量召回顺序 vs 交叉编码精排顺序")
    print("=" * 70)

    for query in QUERIES:
        print(f"\n  查询：{query}")
        candidates = [document.page_content for document in store.similarity_search(query, k=5)]
        print("    ① 向量召回顺序：")
        for rank, text in enumerate(candidates, start=1):
            print(f"       {rank}. {text.replace(chr(10), ' ')[:44]}…")

        started = time.time()
        ranked = reranker.rerank(query, candidates, top_n=5)
        print(f"    ② rerank 精排顺序（{time.time()-started:.2f}s，模型 {settings.rerank.model}）：")
        for rank, item in enumerate(ranked, start=1):
            text = candidates[item["index"]].replace("\n", " ")[:44]
            print(f"       {rank}. 分数 {item['score']:.6f}｜{text}…")

        if ranked and ranked[0]["index"] != 0:
            print("    ✔ 顺序发生了变化：交叉编码把更相关的候选提到了前面")
        else:
            print("    （本次顺序未变：召回的第一条本身就是最相关的）")
        if len(ranked) >= 2:
            top, second = ranked[0]["score"], ranked[1]["score"]
            gap_desc = f"相差 {top / second:.0f} 倍" if second > 0 else "第二名分数为 0（差距无穷大）"
            print(f"    ★ 第一名与第二名差距：{top:.4f} vs {second:.6f}（{gap_desc}）"
                  f" → 说明语料里确实有答案，且 top1 明显更相关")
            print("      （反过来：若所有候选分数都在同一量级且很低，说明**语料里没有答案**，"
                  "这时候应该让模型直说不知道，而不是硬塞上下文）")
    print(
        "\n  ↑ 这就是「召回 + 精排」两级检索的意义：\n"
        "    召回负责**不漏**（双塔快，扫全库），精排负责**排序准**（交叉编码逐对看）。\n"
        "    生产里 rerank 只处理召回的几十条，成本可控。"
    )


# ================================================================
# Demo 4：检索进模型 —— 直接拼上下文作答
# ================================================================
def demo_4_answer_with_context(store: InMemoryVectorStore) -> None:
    print("\n" + "=" * 70)
    print("Demo 4：把精排后的上下文交给模型作答")
    print("=" * 70)

    query = QUERIES[0]
    candidates = [document.page_content for document in store.similarity_search(query, k=5)]
    ranked = reranker.rerank(query, candidates, top_n=2)
    context = "\n\n".join(f"【资料 {i+1}】{candidates[item['index']]}" for i, item in enumerate(ranked))

    prompt = (
        "你是严谨的助手。只依据【资料】回答，资料没写的内容就说不知道，不要编造。\n\n"
        f"{context}\n\n问题：{query}"
    )
    started = time.time()
    response = llm.invoke(prompt)
    usage = getattr(response, "usage_metadata", None) or {}
    print(f"  送入模型的资料条数：{len(ranked)}（来自召回 {len(candidates)} 条）")
    print(f"  回答（{time.time()-started:.1f}s，{usage.get('total_tokens')} tokens）：")
    print(f"    {str(response.content)[:300]}")
    print(
        "  ↑ 注意 prompt 里的约束「只依据资料、没有就说不知道」——\n"
        "    RAG 最常见的翻车方式就是模型拿常识补齐了资料里没有的内容，\n"
        "    所以这条约束（以及引用来源）要写死在提示词里。"
    )


# ================================================================
# Demo 5：把检索做成工具 —— agentic RAG
# ================================================================
# 出处：**deepagents/retrieval.mdx 的 "Agentic RAG" 一节**（不是 knowledge-base.mdx ——
# 那篇止于建库与检索，全文没有 agent 环节）。
# 做法是把检索包装成**工具**交给 agent，让它自己决定要不要查、查几次、查询词怎么写。
# 本 Demo 把「召回 + 精排」合成一个工具，交给 create_agent。
def demo_5_agentic_rag(store: InMemoryVectorStore) -> None:
    print("\n" + "=" * 70)
    print("Demo 5：agentic RAG —— 把检索+精排做成工具")
    print("=" * 70)

    @tool
    def search_knowledge_base(query: str) -> str:
        """在内部知识库里检索（自动做向量召回 + 交叉编码精排），返回最相关的若干段资料。"""
        candidates = [document.page_content for document in store.similarity_search(query, k=5)]
        ranked = reranker.rerank(query, candidates, top_n=2)
        return "\n---\n".join(candidates[item["index"]] for item in ranked) or "（没有检索到资料）"

    agent = create_agent(
        model=llm,
        tools=[search_knowledge_base],
        system_prompt=(
            "你是知识库助手。回答前**必须先调用 search_knowledge_base** 查资料；"
            "资料里没有的内容就直说不知道。回答控制在三句话内。"
        ),
    )
    question = "值班的时候机房温度超了怎么办？还有，知识库里有没有讲重排序的？"
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

    called = [
        call["name"]
        for message in result["messages"]
        for call in (getattr(message, "tool_calls", None) or [])
    ]
    print(f"  提问：{question}")
    print(f"  模型调用的工具：{called}")
    print(f"  回答：{str(result['messages'][-1].content)[:260]}")
    search_calls = [name for name in called if name == "search_knowledge_base"]
    print(
        f"  ↑ 与 Demo 4 的区别：**查不查、查几次由模型决定**（本次它调了 {len(search_calls)} 次；\n"
        "    问题里有两件事时通常会查两次，但这是**模型决策**，不是必然）。\n"
        "    这就是 agentic RAG：检索成了 agent 的能力，而不是流水线里固定的一步。"
    )


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="rag_demo_") as tmp:
        root = Path(tmp)
        print(f"演示知识库目录：{root}\n")
        store, chunks = build_knowledge_base(root)
        demo_1_build(chunks)
        demo_2_vector_search(store)
        demo_3_rerank(store)
        demo_4_answer_with_context(store)
        demo_5_agentic_rag(store)
    print("\n全部 Demo 执行完毕（临时目录已清理）。")


# ================================================================
# 实测结论 / 与课案的衔接 / 踩坑提示
# ================================================================
# 1. 官方事实（knowledge-base.mdx）：
#    - 标准流程：加载 → 切分 → 向量化 → 检索 →（可选）重排序 → 作答；
#    - 最后一步推荐把检索做成工具，让 agent 自主决定检索时机与次数（agentic RAG）。
# 2. 本机实测（2026-09-17）：
#    - embedding `BAAI/bge-m3` @ SiliconFlow：**1024 维**，单次 0.12~0.3 秒，返回 usage；
#    - rerank `BAAI/bge-reranker-v2-m3` @ SiliconFlow：0.2~0.25 秒；
#      **分数要按相对差距解读**：有答案时 top1=0.95 / top2=0.003（差 300 倍），
#      无答案时最高分只有 0.0288 —— 所以用"top1 与 top2 的差距"判断命中，
#      而不是固定阈值（.env 里 RERANK_RELEVANCE_P=0.65 在这种尺度下会误判）；
#    - 作答模型 grok-4.6：Demo 4 单次 19.2 秒 / 2482 tokens，回答严格基于资料并主动声明
#      "资料未单独解释"；Demo 5 里模型**自主调用了两次**检索工具（问题含两件事）。
# 3. 与课案的衔接：
#    - 课案 RAG 项目（RAG/ 目录）用的是 Milvus + 双路召回 + ES，本文件用
#      InMemoryVectorStore + 单路向量召回，目的是把**官方 LangChain 侧的 RAG 写法**
#      讲清楚；两者模型配置共用根目录 .env（EMBEDDING_* / RERANK_* / LLM_*）；
#    - 混合检索（BM25 + 向量）见课案 RAG/retrieval/keyword_retrieval.py 与
#      vector_retrieval.py；本文件的 rerank 组件可直接替换那边的手写实现。
# 4. 踩坑提示：
#    A. `OpenAIEmbeddings` 连第三方 OpenAI 兼容端点时，必须 `check_embedding_ctx_length=False`，
#       否则它会按 OpenAI 的 tiktoken 规则先切分文本，遇到中文/长文本容易报错；
#    B. **rerank 分数不能当概率用**：不同厂商尺度差异极大，只按排名用；
#    C. 中文切分必须自定义 separators（默认按英文标点切）；
#    D. 检索结果要**带来源 metadata**（本文件用 source 文件名），否则答案无法溯源，
#       排障时也不知道是"没检索到"还是"检索到了但模型没用"；
#    E. 工具化的检索函数**要把返回内容限长**（本文件取 top_2），
#       否则多次调用会把上下文撑爆（课案 03_deepagents/14 讲的卸载机制同理）。

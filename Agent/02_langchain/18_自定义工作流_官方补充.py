# -*- coding: utf-8 -*-
r"""
LangChain 官方补充篇：自定义工作流（非课案内容，故不带 _jxsd 后缀）
================================================================
来源与定位：
    本文件对照 LangChain **官方文档** /oss/python/langchain/multi-agent/custom-workflow.mdx
    （及教程 router-knowledge-base.mdx），补上官方**多 Agent 五模式**的最后一个。

    官方多 Agent 五种模式      课案覆盖            本文件/其他补充篇
    --------------------------  ------------------  ------------------
    1. subagents（子代理）       ✅ 12_多Agent_MCP子Agent
    2. handoffs（交接）          ✅ 13_多Agent_交接
    3. router（路由）            ✅ 14_多Agent_路由与合并
    4. skills（渐进披露）        ❌ → ✅            17_Skills渐进披露_官方补充.py
    5. custom-workflow           ❌ → ✅            ← 本文件

    同时它也是「LCEL 之死」的答案：官方新文档全站搜 "LCEL" 零命中，
    课案 15_管道.py 那套编排已不是主线 —— **官方现在的编排手段就是 LangGraph
    的 StateGraph**，本文件演示「把 create_agent 的产物当节点，与确定性步骤混编」。

官方原话（custom-workflow.mdx 的核心洞察）：
    「The core insight is that you can call a LangChain agent directly inside any
      LangGraph node」—— 任何 LangGraph 节点里都能直接调 create_agent 的产物。

    官方还给了三种节点类型的框架（RAG pipeline 例子）：
      · Model node（模型节点）      ：如用结构化输出改写查询；
      · Deterministic node（确定性节点）：如检索，完全不经过 LLM；
      · Agent node（智能体节点）    ：带工具的 agent，负责推理与补充信息。
    以及「把一整个多 Agent 系统当作**一个节点**嵌进工作流」。

⚠️ 与本机环境的差异（必须说明）：
    官方 RAG 例子用的是 OpenAIEmbeddings + InMemoryVectorStore。**本机网关不支持
    embeddings**（实测 /embeddings 对 text-embedding-v4 / text-embedding-3-small /
    BAAI/bge-m3 全部返回 503），所以 Demo 2 把「向量检索节点」换成**确定性关键词检索节点** ——
    这恰好更能体现官方说的 deterministic node：检索这一段完全不依赖 LLM。

⚠️ 本文件需要真实模型（有 agent 节点），按仓库惯例带中文兜底提示。

运行方式（项目根目录下）：
    uv run Agent/02_langchain/18_自定义工作流_官方补充.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台默认 GBK，防中文/emoji 报错

import re

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from config import settings

model = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


def final_text(result: dict) -> str:
    """取结果里的最后一条消息文本。"""
    return str(result["messages"][-1].content)


def agent_answer(agent, prompt: str) -> str:
    """调用一个 agent 并返回它的最终回答（节点里复用的小工具函数）。"""
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    return final_text(result)


# ================================================================
# Demo 1：基础模式 —— agent 作为工作流节点
# ================================================================
# 官方 Basic implementation 的最小版：节点函数内部调 agent.invoke(...)，
# 把 agent 的自然语言输出**转成结构化状态字段**返回。
# 这一步看着简单，但它是「自定义工作流」的全部基础：
# 从此 agent 不再是终点，而是流程里的一环（前面能挂确定性步骤，后面能接条件分支）。
class SimpleState(TypedDict):
    query: str
    answer: str


def build_simple_workflow():
    # 这个 agent 什么工具都不带：只是「模型 + 系统提示」的一层封装
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt="你是简洁的助手，回答控制在两句话以内。",
    )

    def agent_node(state: SimpleState) -> dict:
        """LangGraph 节点：内部调用 LangChain agent。"""
        return {"answer": agent_answer(agent, state["query"])}

    return (
        StateGraph(SimpleState)
        .add_node("agent", agent_node)
        .add_edge(START, "agent")
        .add_edge("agent", END)
        .compile()
    )


def demo_1_agent_as_node() -> None:
    print("=" * 70)
    print("Demo 1：基础模式 —— agent 就是一个节点")
    print("=" * 70)

    workflow = build_simple_workflow()
    result = workflow.invoke({"query": "用一句话解释什么是 LangGraph"})
    print(f"  输入：用一句话解释什么是 LangGraph")
    print(f"  输出：{result['answer'][:150]}")
    print(
        "  ↑ 注意状态字段 answer 是**结构化**的：下游节点可以直接读它做判断，\n"
        "    而不是去解析模型的一整段自然语言 —— 这是把 agent 塞进流程的前提。"
    )


# ================================================================
# Demo 2：官方的三节点 RAG 工作流（检索节点本地化）
# ================================================================
# 官方 RAG pipeline 的三个节点，本 Demo 一一对应：
#     ① rewrite（Model node）：用结构化输出把口语问题改成检索友好的查询 + 关键词；
#     ② retrieve（Deterministic node）：本地知识库关键词检索（替代向量检索，理由见文件头）；
#     ③ agent（Agent node）：拿着检索结果回答，且带一个工具用于补充实时信息。
# 状态在节点间传递**结构化字段**（官方 Tip：用 LangGraph state 在步骤之间传数据）。
class RagState(TypedDict):
    question: str
    rewritten: str          # 改写后的查询
    keywords: list[str]     # 检索关键词
    documents: list[str]    # 命中的知识库文档
    answer: str


class RewrittenQuery(BaseModel):
    """改写节点的结构化输出（官方用 structured output 做这一步）。"""

    rewritten: str = Field(description="更适合检索的完整问句")
    keywords: list[str] = Field(description="3-5 个用于关键词匹配的词")


# 本地知识库：故意用中文短文档，方便人类核对检索结果对不对
KNOWLEDGE_BASE = [
    "LangGraph 的 StateGraph 用节点和边描述流程，节点是函数，边决定执行顺序。",
    "LangGraph 的 checkpointer 负责持久化，thread_id 用来区分不同会话，支持中断恢复。",
    "LangChain 的 create_agent 返回一个编译好的图，可以直接 invoke，也可以当作子图嵌入。",
    "DeepAgents 的 create_deep_agent 在 create_agent 基础上内置了文件系统、子代理与任务规划。",
    "LangSmith 提供链路追踪与评估，属于可观测性工具，与运行时框架解耦。",
]


def _bigrams(text: str) -> set[str]:
    """把文本切成字符二元组（中文无需分词器，且对措辞差异比整词匹配宽容）。"""
    cleaned = re.sub(r"[\s，。、？！：；（）【】“”‘’,.?!:;()\[\]]+", "", text.lower())
    return {cleaned[i:i + 2] for i in range(len(cleaned) - 1)}


def keyword_retrieve(query: str, keywords: list[str], top_k: int = 2) -> list[str]:
    """确定性检索：按字符 bigram 重叠数打分（不调用模型、不依赖 embeddings）。

    为什么不用整词匹配：中文里「状态存储」和文档里的「持久化」是同一件事但字面不同，
    整词匹配会全miss。bigram 重叠能抓住「会话」「存」这类共同片段 ——
    但它**治不了真正的语义鸿沟**（那正是向量检索存在的理由，而本机网关没有 embeddings）。
    """
    query_grams = _bigrams(query + "".join(keywords))
    scored: list[tuple[int, str]] = []
    for doc in KNOWLEDGE_BASE:
        overlap = len(query_grams & _bigrams(doc))
        if overlap:
            scored.append((overlap, doc))
    scored.sort(key=lambda item: -item[0])
    return [doc for _, doc in scored[:top_k]]


@tool
def get_current_date() -> str:
    """查询今天的日期（演示 agent 节点在检索之外补充实时信息）。"""
    from datetime import date

    return date.today().isoformat()


def build_rag_workflow():
    structured_model = model.with_structured_output(RewrittenQuery)
    agent = create_agent(
        model=model,
        tools=[get_current_date],
        system_prompt=(
            "你是技术问答助手。只依据【参考资料】回答；"
            "资料里没有的内容就直说不知道，不要编造。回答控制在三句话内。"
        ),
    )

    def rewrite_node(state: RagState) -> dict:
        """① 模型节点：把口语化问题改写成检索友好的形式。"""
        try:
            out = structured_model.invoke([
                SystemMessage(
                    content="把用户问题改写成适合检索的形式，并提取关键词。"
                            "**必须全部用中文输出**（专有名词如 checkpointer 可保留英文）。"
                ),
                HumanMessage(content=state["question"]),
            ])
            print(f"    [rewrite] 改写为：{out.rewritten!r}，关键词：{out.keywords}")
            return {"rewritten": out.rewritten, "keywords": out.keywords}
        except Exception as exc:  # noqa: BLE001
            # 结构化输出偶发失败时退回原问题（工作流不该因为这一步挂掉）
            print(f"    [rewrite] 结构化输出失败（{type(exc).__name__}），退回原问题")
            return {"rewritten": state["question"], "keywords": []}

    def retrieve_node(state: RagState) -> dict:
        """② 确定性节点：关键词检索，完全不经过 LLM。"""
        docs = keyword_retrieve(state["rewritten"], state.get("keywords", []))
        print(f"    [retrieve] 命中 {len(docs)} 篇：{[d[:18] + '…' for d in docs]}")
        return {"documents": docs}

    def agent_node(state: RagState) -> dict:
        """③ 智能体节点：带着检索结果作答，必要时可调工具补充实时信息。"""
        context = "\n".join(f"- {doc}" for doc in state["documents"]) or "（没有命中任何资料）"
        prompt = f"【参考资料】\n{context}\n\n【问题】{state['question']}"
        return {"answer": agent_answer(agent, prompt)}

    return (
        StateGraph(RagState)
        .add_node("rewrite", rewrite_node)
        .add_node("retrieve", retrieve_node)
        .add_node("agent", agent_node)
        .add_edge(START, "rewrite")
        .add_edge("rewrite", "retrieve")
        .add_edge("retrieve", "agent")
        .add_edge("agent", END)
        .compile()
    )


def demo_2_rag_workflow() -> None:
    print("\n" + "=" * 70)
    print("Demo 2：三节点 RAG 工作流（模型节点 → 确定性节点 → 智能体节点）")
    print("=" * 70)

    workflow = build_rag_workflow()

    # 同一套工作流跑两个问题，专门对比「确定性检索」的命中差异：
    #   问题 A 的用词和知识库接近（bigram 重叠多）→ 能命中；
    #   问题 B 是纯语义改写（口语问法）→ 可能命中不准，甚至答"不知道"。
    questions = [
        "checkpointer 和 thread_id 分别是干什么的？",
        "图里的状态是怎么存下来的？换会话会串吗？",
    ]
    for index, question in enumerate(questions, start=1):
        print(f"\n  --- 问题 {index}：{question} ---")
        result = workflow.invoke({"question": question})
        print(f"  最终回答：{result['answer'][:200]}")

    print(
        "\n  ↑ 两个问题问的其实是同一件事，但**确定性检索**只在用词接近时稳：\n"
        "    它可复现、免费、可单独测（给查询断言命中哪几篇），却治不了语义鸿沟；\n"
        "    向量检索正是为这个问题存在的 —— 而本机网关没有 embeddings（实测 503），\n"
        "    所以 RAG 类缺口在 官方文档缺口对照.md 里仍标着「待补（要 embeddings）」。\n"
        "    这也说明自定义工作流的一个实用价值：**检索质量可以脱离模型单独评估**。"
    )


# ================================================================
# Demo 3：把一整个 Agent 当节点 + 条件分支与循环（evaluator-optimizer）
# ================================================================
# 官方说「You can also compose other architectures within a custom workflow—
#          for example, embedding a multi-agent system as a single node」。
# 本 Demo 做两件事：
#   ① 把一个 agent 当成节点（它内部其实可以是多 Agent 系统 —— 课案 12/13 的产物）；
#   ② 用**条件边**做 evaluator-optimizer 循环：生成 → 评分 → 不达标就带着反馈回炉。
#      评分用确定性规则（关键词 + 长度），这样循环次数可控、结论可复现；
#      生产里这一步通常换成模型评分或 DeepAgents 的 RubricMiddleware（见对应补充篇）。
class LoopState(TypedDict):
    topic: str
    draft: str
    score: int
    feedback: str
    attempts: int


REQUIRED_POINTS = ("文件系统", "子代理")   # 文案里必须覆盖的两个卖点
MAX_ATTEMPTS = 3


def build_loop_workflow():
    # 这个 agent 代表「被嵌入的多 Agent 系统」——换成课案 12/13 的多 Agent 产物同样成立
    writer = create_agent(
        model=model,
        tools=[],
        system_prompt=(
            "你是产品文案写手。用 2-3 句话写一段介绍，必须覆盖用户点名的所有卖点，"
            "并且直接给出文案本身，不要任何寒暄或解释。"
        ),
    )

    def generate_node(state: LoopState) -> dict:
        """生成节点：把（可选的）上一轮反馈拼进提示词，实现「带着意见重写」。"""
        attempts = state.get("attempts", 0) + 1
        prompt = f"请介绍 DeepAgents，卖点：{'、'.join(REQUIRED_POINTS)}。"
        if state.get("feedback"):
            prompt += f"\n上一版的问题是：{state['feedback']}，请修正。"
        draft = agent_answer(writer, prompt)
        # 为了在本 Demo 里**稳定看到回炉过程**，第一稿故意抹掉一个卖点。
        # （真实项目里第一稿是否达标取决于模型状态，可能一轮就过 —— 那时
        #   条件边会直接走到 END，循环体只执行一次，这也是正常结果。）
        if attempts == 1:
            draft = draft.replace(REQUIRED_POINTS[-1], "相关能力")
            print(f"    [generate] （演示用：第一稿故意抹掉卖点「{REQUIRED_POINTS[-1]}」）")
        print(f"    [generate] 第 {attempts} 稿：{draft[:60]}…")
        return {"draft": draft, "attempts": attempts}

    def evaluate_node(state: LoopState) -> dict:
        """评分节点：确定性规则打分（0-100），并给出可读的改进意见。"""
        draft = state["draft"]
        hits = [point for point in REQUIRED_POINTS if point in draft]
        length_ok = 40 <= len(draft) <= 400
        score = int(100 * len(hits) / len(REQUIRED_POINTS)) - (0 if length_ok else 20)
        missing = [point for point in REQUIRED_POINTS if point not in draft]
        feedback = "" if not missing else f"缺少卖点：{'、'.join(missing)}"
        if not length_ok:
            feedback = (feedback + "；" if feedback else "") + f"篇幅不合适（当前 {len(draft)} 字）"
        print(f"    [evaluate] 得分 {score}，意见：{feedback or '无'}")
        return {"score": score, "feedback": feedback}

    def route_after_evaluate(state: LoopState) -> str:
        """条件边：达标或次数用尽就结束，否则回到生成节点重写。"""
        if state["score"] >= 100 or state["attempts"] >= MAX_ATTEMPTS:
            return END
        return "generate"

    return (
        StateGraph(LoopState)
        .add_node("generate", generate_node)
        .add_node("evaluate", evaluate_node)
        .add_edge(START, "generate")
        .add_edge("generate", "evaluate")
        # 条件边必须有通往 END 的出口，否则会无限循环（官方 Common Fixes 的第一条）
        .add_conditional_edges("evaluate", route_after_evaluate, ["generate", END])
        .compile()
    )


def demo_3_embedded_agent_and_loop() -> None:
    print("\n" + "=" * 70)
    print("Demo 3：嵌入整个 Agent 当节点 + evaluator-optimizer 循环")
    print("=" * 70)

    workflow = build_loop_workflow()
    result = workflow.invoke({"topic": "DeepAgents", "draft": "", "score": 0, "feedback": "", "attempts": 0})
    print(f"  循环了 {result['attempts']} 轮，最终得分 {result['score']}")
    print(f"  最终文案：{result['draft'][:220]}")
    print(
        "  ↑ 循环的退出条件写在**条件边**里（达标 或 次数用尽 → END），\n"
        "    这是官方 Common Fixes 反复强调的一点：没有通往 END 的条件出口就会无限循环。\n"
        "    评分规则是确定性的，所以整个循环可复现；把 evaluate 换成模型（或 RubricMiddleware）\n"
        "    就得到官方 workflows-agents 里的 evaluator-optimizer 模式。"
    )


if __name__ == "__main__":
    demo_1_agent_as_node()
    demo_2_rag_workflow()
    demo_3_embedded_agent_and_loop()
    print("\n全部 Demo 执行完毕。")


# ================================================================
# 实测结论 / 与课案的衔接 / 踩坑提示
# ================================================================
# 1. 实测结论（本机，grok-4.6）：
#    - Demo 1：agent 节点把模型输出落成结构化字段 answer，工作流正常返回；
#    - Demo 2：改写节点（结构化输出）会把口语问题改写并提取关键词；检索节点用
#      字符 bigram 重叠打分，两个测试问题都命中了正确的知识库文档
#      （问题 1 命中 checkpointer + StateGraph 两篇，问题 2 命中 checkpointer 一篇），
#      最终回答与文档一致（"换会话不会串"）；
#    - Demo 3：第一稿 50 分（缺卖点「子代理」）→ 条件边回到 generate →
#      第二稿带反馈重写得 100 分 → 条件边走向 END，共循环 2 轮。
# 2. 官方事实（multi-agent/custom-workflow.mdx）：
#    - 核心手法：在任何 LangGraph 节点里直接调用 create_agent 的产物；
#    - 官方 RAG 例子给出三种节点类型：Model node / Deterministic node / Agent node；
#    - 官方明说可以「把一整个多 Agent 系统作为一个节点」嵌进工作流；
#    - 官方教程 router-knowledge-base 本身就是「并行查询多个源再汇总」的自定义工作流。
# 3. 本机适配：官方用 OpenAIEmbeddings + InMemoryVectorStore，本机网关不支持 embeddings
#    （实测 503），Demo 2 换成确定性关键词检索 —— 检索节点的「确定性」属性反而更纯粹。
# 4. 与课案的衔接：
#    - 课案 15_管道.py 讲的是 LCEL；官方新文档已不再以 LCEL 为主线，
#      编排的对应物就是本文件的 StateGraph（这一点写进了 官方文档缺口对照.md）；
#    - 状态传递、条件边、循环、Send 并行等 LangGraph 基础见课案 01_langgraph 与
#      10_控制流与函数式API_官方补充.py；本文件重点是「agent 与确定性步骤混编」。
# 5. 踩坑提示：
#    A. 节点里调 agent 时，**只把需要的信息传进去**（本文件 agent_node 只传检索结果 +
#       问题），不要把整个 state 塞进 prompt —— 否则 token 会随流程线性膨胀；
#    B. 条件边一定要有通往 END 的出口，否则循环不收敛（官方 Common Fixes 第 1 条）；
#    C. `Command(goto=...)` 只新增**动态**边：若节点上已有静态出边，两者都会执行
#       （官方 Warning，本地实测同样如此，见 10_控制流与函数式API_官方补充.py）；
#    D. 模型节点用结构化输出时要有兜底：网关偶发不支持某模型的 json_schema 模式，
#       本文件 rewrite_node 失败即退回原问题，保证工作流不因单点失败而中断；
#    E. agent 节点的输出请落成**结构化字段**（answer/draft），别让下游去解析长文本。

# -*- coding: utf-8 -*-
"""
Langfuse ⑤：评估之「RAG 评估」——课案完整版
================================================================
课案原话：
    Langfuse 内置 RAG 评估指标，使用 Ragas 框架计算四项真实分数（LLM-as-Judge）。

RAG 系统有两个会出错的环节，所以指标也分两组：

    【2.1 检索质量指标】—— 检索出来的上下文好不好？
    | 指标                          | 一句话                    | 目的                                   | 计算方法                                                     |
    |------------------------------|--------------------------|----------------------------------------|--------------------------------------------------------------|
    | 上下文相关性 Context Precision | 相关上下文是不是排在前面？  | 检索是否**精准**，有没有一堆无关冗余      | LLM 从检索结果里提取与问题直接相关的句子，算其比例             |
    | 上下文召回率 Context Recall    | 标准答案要的信息，上下文给全了吗？ | 检索是否**全面**，能否覆盖标准答案的事实点 | 把标准答案拆成若干事实点，逐个判断能否从上下文推断；覆盖率即召回 |

    【2.2 生成质量指标】—— 拿着上下文生成出来的回答好不好？
    | 指标                       | 一句话                          | 目的                             | 计算方法                                                          |
    |---------------------------|--------------------------------|----------------------------------|------------------------------------------------------------------|
    | 忠实度 Faithfulness        | 回答里每句话都能从上下文推出来吗？ | 检测**幻觉**                      | 1) 从回答中提取所有陈述；2) 逐条判断能否由上下文推断；可推断数/总数   |
    | 答案相关性 Answer Relevancy | 回答有没有直接答用户的问题？      | 评估**切题**程度                  | 1) 基于回答反向生成多个可能的问题；2) 算这些反向问题与原问题的语义相似度均值 |

数据集怎么造（课案「数据集」两条注意事项）：

    | 指标组   | contexts 字段从哪来              | answer 字段从哪来              |
    |---------|--------------------------------|------------------------------|
    | 检索指标 | 自己系统的检索结果（真实召回的文档） | 自己系统流程生成的回答           |
    | 生成指标 | **生成这条回答时用的那批上下文**    | 同上                          |

    换句话说：评估检索，就用检索到的上下文；评估生成，必须用「当时喂给模型的那批上下文」，
    否则测的就不是同一件事了。

安装（课案给的命令）：
    uv add langchain-openai ragas "langchain-community==0.3.30"
    —— 本机当前 **没有装 ragas**，所以下面的 ragas 相关 import 全部用 try/except 包着：
       装上 ragas  → 走课案原文的 ragas 官方指标（真实分数）
       没装 ragas  → 走本地 LLM-as-Judge 近似版（同样的方法论，提示词是教学简化版）
       两种情况下都真的调大模型，都能看到四个数值。

关于评测模型为什么用 DashScope（课案写的是 AsyncOpenAI + settings.dashscope_*）：
    1) AnswerRelevancy 需要**句向量**算语义相似度，阿里百炼有 text-embedding-v4；
    2) 评测模型最好和业务模型分开，「自己评自己」容易偏袒；
    3) 评测器要输出长 JSON，max_tokens 给大一点（课案用 8192）防止 JSON 被截断。
    本机 settings.dashscope_api_key / dashscope_base_url 为空，
    所以降级到项目里实测可用的 settings.api_key / settings.base_url
    （注意 key 和 base_url 必须成对降级，只换其中一个会 401）。

课案出处：Agent 课案 → 监控与评估 → 评估 → RAG评估
         （课案这一节是本章最长的一段，本文件把它的 185 行完整落地成可跑的评估脚本）

运行方式：
    uv run Agent/06_langfuse/05_评估_RAG与Agent_jxsd.py
    （会真调大模型算分数，整跑约 1~3 分钟）

本机前置条件：Langfuse 密钥为空 → 打印配置指引 + 数据集用本地桩数据 +
分数以「本应上报的报文」形式打印，其余流程（生成回答、四个指标打分）全部真跑。

本机实测结论（未装 ragas → 走本地 LLM-as-Judge；裁判模型 = settings.model_name）：

    | 测试项 | CP | CR | Faith | AR |
    |---|---|---|---|---|
    | Transformer 的核心机制 | 1.00 | 0.67 | 1.00 | 1.00 |
    | 什么是 RAG | 0.67 | 1.00 | 0.71 | 0.90 |
    | 向量数据库的作用 | 0.67 | 1.00 | 0.80 | 0.85 |

三点值得讲给学员（也正是「为什么要有四个指标」的答案）：

    1. **CP 与 CR 会互相打架，不能只看一个。** 第 1 条 CP=1.00 但 CR=0.67
       （检索到的 3 条全相关，但标准答案里仍有事实点没被覆盖）；
       第 2、3 条反过来，CR=1.00 但 CP=0.67（信息给全了，但掺了不太相关的句子）。
       —— 一个衡量「精」，一个衡量「全」，取舍要靠业务定。
    2. **Faithfulness 是唯一经常低于 1 的生成指标，它测的是幻觉。**
       第 2 条 Faith=0.71：模型答得比上下文多，多出来的部分无法从给定上下文推出，于是被扣分。
       注意：**这不一定代表模型错了**，只代表「用了上下文之外的知识」——
       所以 Faithfulness 偏低要人工看一眼，别直接当 bug。
    3. **AR 稳定偏高（0.85~1.00）**，因为它只看「回答是否切题」，不看对错；
       它掉到 0.5 以下通常意味着回答跑题或答非所问，是很好用的兜底告警指标。
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import re
from types import SimpleNamespace

# langfuse 只负责「分数往哪去」；ragas 缺失时下面走本地等价实现，四项数值照常算得出。
from langchain.chat_models import init_chat_model
from langfuse import Langfuse
from config import settings

# ---------- 0. 可选依赖：ragas 没装也要能 import 本文件 ----------
try:
    from ragas.llms import llm_factory                     # noqa: F401
    from ragas.metrics.collections import (                # noqa: F401
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )
    RAGAS_AVAILABLE = True
    RAGAS_IMPORT_ERROR = ""
except ImportError as exc:                                 # 缺包：记下原因，走本地近似实现
    RAGAS_AVAILABLE = False
    RAGAS_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# ---------- 1. 客户端与模型 ----------
LANGFUSE_READY = bool(settings.langfuse_public_key and settings.langfuse_secret_key)

if LANGFUSE_READY:
    langfuse = Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
else:
    langfuse = None

# 生成回答用的业务模型；评测（裁判）模型另有一份，见 build_eval_client。
llm = init_chat_model(
    model_provider="openai",
    model=settings.model_name,
    api_key=settings.api_key,
    base_url=settings.base_url,
)


def report_score(**payload) -> None:
    """上报一条分数：连上 Langfuse 就真上传，否则打印报文（降级演示）。"""
    if LANGFUSE_READY:
        langfuse.create_score(**payload)
    else:
        short = {k: (v[:120] + "…" if isinstance(v, str) and len(v) > 120 else v)
                 for k, v in payload.items()}
        print("        [降级] 本应上报的 score 报文：" + json.dumps(short, ensure_ascii=False, default=str))


def build_eval_client():
    """构造评测用的 OpenAI 兼容客户端。

    课案原文：
        eval_client = AsyncOpenAI(api_key=settings.dashscope_api_key,
                                  base_url=settings.dashscope_base_url)
    本机 dashscope 两项为空 → 成对降级到 settings.api_key / settings.base_url。
    """
    if settings.dashscope_api_key:
        return settings.dashscope_api_key, settings.dashscope_base_url, "DashScope（课案原配）"
    return settings.api_key, settings.base_url, f"{settings.model_name}（项目通用大模型，降级）"


# ---------- 2. 构造 RAG 评估数据集（课案原文三条，上下文一字未改） ----------
DATASET_NAME = "rag_evaluation"

# 课案原文三条测试项，contexts 一字未改 —— 这样算出来的分数才能和课案对得上。
DATASET_ITEMS = [
    {
        "input": "Transformer 模型的核心机制是什么？",
        "expected_output": "Transformer 的核心是自注意力（Self-Attention）机制，"
                           "它允许模型在处理每个词时关注输入序列中的所有位置，"
                           "从而捕获长距离依赖关系。",
        "metadata": {
            "contexts": [
                "Transformer 架构由 Vaswani 等人在 2017 年提出，完全基于注意力机制，摒弃了循环和卷积结构。",
                "自注意力机制是 Transformer 的核心，通过 Query、Key、Value 三个矩阵计算序列中每个位置与其他位置的关联权重。",
                "多头注意力将注意力计算拆分到多个子空间，使模型能同时关注不同位置的不同特征表示。",
            ]
        },
    },
    # 第 2 条的 contexts 已完整覆盖标准答案，用来验证「检索给全了」时召回率应接近 1。
    {
        "input": "什么是 RAG？它解决了什么问题？",
        "expected_output": "RAG（检索增强生成）是一种将信息检索与文本生成相结合的技术架构，"
                           "主要解决了大语言模型的幻觉问题和知识时效性问题。",
        "metadata": {
            "contexts": [
                "RAG（Retrieval-Augmented Generation）由 Facebook AI 在 2020 年提出，"
                "工作流程：用户提问 → 从知识库检索相关文档 → 将检索结果作为上下文注入 Prompt → LLM 生成答案。",
                "RAG 解决的两大核心问题：1）幻觉——模型编造不存在的事实；"
                "2）知识时效性——训练数据截止日期后的事件模型无法知晓。",
                "RAG 的优势：无需微调模型即可接入最新知识，答案可溯源至检索到的文档片段。",
            ]
        },
    },
    # 第 3 条考跨句推断：单个事实点要能把三条上下文里的信息拼起来才推得出。
    {
        "input": "向量数据库在 RAG 中起什么作用？",
        "expected_output": "向量数据库在 RAG 中负责高效存储和检索文档的向量嵌入，"
                           "是 RAG 检索阶段的核心基础设施。",
        "metadata": {
            "contexts": [
                "向量数据库将文本通过嵌入模型转换为高维向量，"
                "查询时用同样的嵌入模型将问题转为向量，通过余弦相似度等度量找到最相近的文档。",
                "常见向量数据库：Chroma、Pinecone、Weaviate、Milvus、Qdrant。",
                "在 RAG 流水线中，向量数据库处于检索阶段 —— "
                "接收用户查询向量，返回 Top-K 相关文档片段供 LLM 参考。",
            ]
        },
    },
]


def build_dataset_items():
    """真连上 Langfuse 就走课案流程；否则用同字段的本地桩数据，让评估循环代码不变。"""
    if LANGFUSE_READY:
        # 真连上就走课案流程（建集 → 逐条加 → get 回来），字段名与 dataset.items 完全一致。
        langfuse.create_dataset(name=DATASET_NAME)
        for item in DATASET_ITEMS:
            langfuse.create_dataset_item(dataset_name=DATASET_NAME, **item)
        return langfuse.get_dataset(DATASET_NAME).items

    print(f"[降级] 本应创建 Langfuse 数据集 {DATASET_NAME}，共 {len(DATASET_ITEMS)} 条测试项")
    for item in DATASET_ITEMS:
        print(f"        · {item['input']}  （contexts {len(item['metadata']['contexts'])} 条）")
    return [SimpleNamespace(**item) for item in DATASET_ITEMS]


# ---------- 3. 课案原文路径：ragas 官方四项指标 ----------
def build_ragas_metrics(eval_model_name, eval_client):
    """按课案原文创建四项评估指标实例。

        evaluator_llm = llm_factory(model, client=eval_client, max_tokens=8192)
        cp_metric = ContextPrecision(llm=evaluator_llm)
        cr_metric = ContextRecall(llm=evaluator_llm)
        faith_metric = Faithfulness(llm=evaluator_llm)
        ar_metric = AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_emb, strictness=1)

    strictness 控制「从回答反推生成多少个问题」来检测相关性：1 = 只生成 1 个。
    """
    from ragas.embeddings import OpenAIEmbeddings

    evaluator_llm = llm_factory(eval_model_name, client=eval_client, max_tokens=8192)
    # 课案的 embedding 模型是阿里百炼的 text-embedding-v4；
    # 降级到通用大模型时通常没有 embedding 接口，AnswerRelevancy 需要句向量，
    # 这时候只能退回本地近似实现（见下面 local_answer_relevancy）。
    evaluator_emb = OpenAIEmbeddings(client=eval_client, model="text-embedding-v4")
    return {
        "Context Precision": ContextPrecision(llm=evaluator_llm),
        "Context Recall": ContextRecall(llm=evaluator_llm),
        "Faithfulness": Faithfulness(llm=evaluator_llm),
        "Answer Relevancy": AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_emb, strictness=1),
    }


# ---------- 4. 降级路径：本地 LLM-as-Judge 近似版四项指标 ----------
# 说明：ragas 官方实现各有成套的提示词与多次采样，下面是**同一方法论的教学简化版**，
#       目的是没装 ragas 时也能看到四个真实数值。口径与 ragas 不完全一致，
#       两个工具的分数不要直接横向比较。
def judge(prompt: str) -> dict:
    """通用裁判调用：把提示词发给大模型，要求它只回 JSON，再解析成 dict。

    健壮性处理（线上评估脚本必须做）：
        - 模型爱把 JSON 包在 ```json ``` 里 → 用正则把第一段 {...} 抠出来
        - 网络抖动 / 模型抽风 → 捕获异常并返回 {}，让调用方走兜底分数，
          绝不能因为一条评估项失败就把整个评估任务挂掉
    """
    try:
        text = llm.invoke(prompt).content
    except Exception as exc:
        print(f"        [警告] 裁判模型调用失败：{type(exc).__name__}: {exc}")
        return {}
    match = re.search(r"\{.*\}", text, re.S)      # 贪婪匹配到最后一个 }，容错 ```json 包裹
    if not match:
        print(f"        [警告] 裁判模型没有返回 JSON：{text[:80]}…")
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        print(f"        [警告] 裁判返回的 JSON 解析失败：{exc}")
        return {}


def local_context_precision(question: str, contexts: list[str]) -> float:
    """上下文相关性：相关上下文所占比例（0~1）。"""
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(contexts))
    data = judge(
    # 一次调用问完所有上下文：让裁判按顺序回 true/false 数组。
        f"问题：{question}\n\n检索到的上下文：\n{numbered}\n\n"
        "请逐条判断每条上下文是否与「回答问题」直接相关（只要含有能回答问题的信息就算相关）。\n"
        '只输出 JSON：{"relevant": [true, false, ...]}，数组长度必须等于上下文条数。'
    )
    flags = data.get("relevant")
    if not isinstance(flags, list) or not flags:
        return 0.0
    flags = [bool(x) for x in flags[: len(contexts)]]
    return sum(flags) / len(contexts)


def local_context_recall(question: str, contexts: list[str], reference: str) -> float:
    """上下文召回率：标准答案的事实点被上下文覆盖的比例（0~1）。"""
    data = judge(
        f"问题：{question}\n\n标准答案：{reference}\n\n"
        f"检索到的上下文：\n" + "\n".join(f"- {c}" for c in contexts) + "\n\n"
        "请把标准答案拆解成若干独立事实点，再逐个判断该事实点能否从上面的上下文中推断出来。\n"
        '只输出 JSON：{"total": 事实点总数, "covered": 能被上下文覆盖的事实点数}'
    )
    # 裁判回非数值一律判 0 分：兜底优先于「猜一个分数」，不能让一条用例挂掉整轮评估。
    total, covered = data.get("total"), data.get("covered")
    if not isinstance(total, (int, float)) or not total:
        return 0.0
    return max(0.0, min(1.0, float(covered or 0) / float(total)))


def local_faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    """忠实度：回答里的陈述能被上下文支撑的比例（0~1），越低说明越可能在编。"""
    data = judge(
        f"问题：{question}\n\n检索到的上下文：\n" + "\n".join(f"- {c}" for c in contexts) +
        f"\n\n模型生成的回答：{answer}\n\n"
        "请把回答拆成若干条独立陈述，逐条判断能否由上面的上下文推断出来。\n"
        '只输出 JSON：{"total": 陈述总数, "supported": 能被上下文支撑的陈述数}'
    )
    # 与召回率同构：先把回答拆成陈述，再逐条判断能否由上下文推出。
    total, supported = data.get("total"), data.get("supported")
    if not isinstance(total, (int, float)) or not total:
        return 0.0
    return max(0.0, min(1.0, float(supported or 0) / float(total)))


def local_answer_relevancy(question: str, answer: str, strictness: int = 1) -> float:
    """答案相关性：由回答反推问题，再算这些问题与原问题的语义相似度均值（0~1）。

    真 ragas 用 embedding 算余弦相似度；这里没有句向量模型，
    改让同一个裁判模型在生成反向问题的同时给出相似度评分（一次调用完成）。
    """
    data = judge(
        f"原始问题：{question}\n\n模型回答：{answer}\n\n"
        f"请基于这个回答，反向推断出 {strictness} 个「它最可能是在回答的问题」，"
        "并给每个反向问题与原问题的语义相似度打分（0~1）。\n"
        '只输出 JSON：{"questions": ["反向问题1"], "similarities": [0.0]}'
    )
    # 真 ragas 用 embedding 算余弦相似度；这里让裁判一次调用同时给出反向问题与相似度。
    sims = data.get("similarities")
    if not isinstance(sims, list) or not sims:
        return 0.0
    values = [float(s) for s in sims if isinstance(s, (int, float))]
    return sum(values) / len(values) if values else 0.0


def local_metrics(question: str, answer: str, contexts: list[str], reference: str) -> dict:
    """四项指标一次算完，返回 {指标名: 分数}。"""
    return {
        "Context Precision": local_context_precision(question, contexts),
        "Context Recall": local_context_recall(question, contexts, reference),
        "Faithfulness": local_faithfulness(question, answer, contexts),
        "Answer Relevancy": local_answer_relevancy(question, answer, strictness=1),
    }


# ---------- 5. 主评估流程 ----------
def run_evaluation():
    eval_model_name = settings.model_name
    eval_key, eval_base, eval_source = build_eval_client()
    eval_client = None
    metrics = None

    # 装了 ragas 用官方四项指标；没装走本地近似实现，两者口径不同，分数不要横向比较。
    if RAGAS_AVAILABLE:
        from openai import AsyncOpenAI
        eval_client = AsyncOpenAI(api_key=eval_key, base_url=eval_base)
        metrics = build_ragas_metrics(eval_model_name, eval_client)
        print(f"评估器：ragas 官方指标，裁判模型 {eval_model_name} @ {eval_source}")
    else:
        print("评估器：本地 LLM-as-Judge 近似实现（未安装 ragas）")
        print(f"        裁判模型 {settings.model_name} @ {eval_source}")

    print("\n" + "=" * 72)
    print("逐条评估：拼上下文 → 生成回答 → 四项指标打分 → 上报")
    print("=" * 72)

    # 两种数据源的字段名对齐，所以下面这段评估循环一行都不用改。
    items = build_dataset_items()
    for idx, item in enumerate(items, start=1):
        contexts = item.metadata.get("contexts", [])
        context_text = "\n".join(f"- {ctx}" for ctx in contexts)
        prompt = (
            f"请根据以下参考资料回答问题。\n\n"
            f"参考资料：\n{context_text}\n\n"
            f"问题：{item.input}"
        )

        print(f"\n[{idx}/{len(items)}] Q: {item.input}")
        # 课案这里用的是 agent.invoke（DeepAgents）；RAG 的「增强生成」本来就是
        # 「上下文拼进 Prompt → 模型作答」这一步，这里直接用大模型调用表示自己系统流程。
        answer = llm.invoke(prompt).content
        print(f"        A: {answer[:80]}…")

        if RAGAS_AVAILABLE:
            # 课案的写法：四个指标各自 .score(...)，取 .value
            cp = metrics["Context Precision"].score(
                user_input=item.input, reference=item.expected_output, retrieved_contexts=contexts,
            ).value
            cr = metrics["Context Recall"].score(
                user_input=item.input, retrieved_contexts=contexts, reference=item.expected_output,
            ).value
            faith = metrics["Faithfulness"].score(
                user_input=item.input, response=answer, retrieved_contexts=contexts,
            ).value
            ar = metrics["Answer Relevancy"].score(
                user_input=item.input, response=answer,
            ).value
        else:
            scores = local_metrics(item.input, answer, contexts, item.expected_output)
            cp, cr, faith, ar = (scores["Context Precision"], scores["Context Recall"],
                                 scores["Faithfulness"], scores["Answer Relevancy"])

        # 真实环境里 tid 来自 langfuse_handler.last_trace_id；这里用可预测的假 id 便于对照报文。
        tid = "trace-rag-%03d" % idx          # 真实环境里是 langfuse_handler.last_trace_id

        # 课案：四项分数逐个上报到同一条 trace 上，name 就是指标名
        report_score(trace_id=tid, name="Context Precision", value=cp, comment=f"contexts={len(contexts)}条")
        report_score(trace_id=tid, name="Context Recall", value=cr, comment=f"contexts={len(contexts)}条")
        report_score(trace_id=tid, name="Faithfulness", value=faith, comment="回答是否忠于上下文")
        report_score(trace_id=tid, name="Answer Relevancy", value=ar, comment="回答是否切题")

        # 一行打完四个分数，方便直接和课案截图里的数值对照。
        print(f"        CP={cp:.2f}  CR={cr:.2f}  Faith={faith:.2f}  AR={ar:.2f}")
        print(f"        Trace: {tid}")
        print("-" * 50)

    if LANGFUSE_READY:
        langfuse.flush()


# 入口：先讲清本机缺什么（ragas / Langfuse / DashScope 三处），再真跑评估。
if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 72)
    print("RAG 评估四项指标：Context Precision / Context Recall / Faithfulness / Answer Relevancy")
    print("=" * 72)

    if not RAGAS_AVAILABLE:
        # 依赖缺失只提示、不退出：本地实现同样能算出四个数值。
        print("\n【依赖未安装】没找到 ragas，本文件按规范用 try/except 兜住，不会报错退出。")
        print(f"  import 失败原因：{RAGAS_IMPORT_ERROR}")
        print("  安装命令（课案给的）：")
        print('      uv add langchain-openai ragas "langchain-community==0.3.30"')
        print("  装完后重跑本脚本，就会自动切换到 ragas 官方指标。")

    if not LANGFUSE_READY:
        # 密钥缺失同理：只影响「上报」这一步，不影响打分。
        print("\n【进入降级演示】Langfuse 密钥为空")
        print("  settings.langfuse_public_key = '' ，settings.langfuse_secret_key = ''")
        print("-" * 72)
        print("要看到真实的评估看板，按课案「安装」一节准备环境：")
        print("  1) git clone https://github.com/langfuse/langfuse.git")
        print("     cd langfuse")
        print("     docker compose up -d          # 启动后访问 http://localhost:3000")
        print("  2) 首次注册的账号即为管理员；新建项目 → Settings → API Keys → 创建密钥")
        print("  3) 写入 F:\\ProGram\\Python_Base\\.env ：")
        print("        LANGFUSE_PUBLIC_KEY=pk-lf-...")
        print("        LANGFUSE_SECRET_KEY=sk-lf-...")
        print("        LANGFUSE_HOST=http://localhost:3000    # 本地 docker 部署用这个")
        print("  4) 重新运行本脚本，四项指标会出现在 trace 详情与 Scores 看板里")
        print("-" * 72)
        print("下面不依赖 Langfuse 服务：数据集用同字段的本地桩数据，")

    if not settings.dashscope_api_key:
        # 评测模型降级说明：key 与 base_url 必须成对降级，只换其中一个会 401。
        print("\n【评测模型降级】settings.dashscope_api_key / dashscope_base_url 为空")
        print("  课案用 DashScope（阿里百炼）当裁判：它有 text-embedding-v4 供 AnswerRelevancy")
        print("  算句向量，而且评测模型与业务模型分开、避免自评偏袒。")
        print(f"  这里成对降级到 settings.api_key / settings.base_url（当前模型 {settings.model_name}）。")
        print("  想用回课案原配：去阿里百炼控制台建 key，然后写进 .env ：")
        print("        DASHSCOPE_API_KEY=sk-...")
        print("        DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1")

    if not LANGFUSE_READY:
        # 三条降级说明都打完了，下面开始真跑。
        print("\n生成回答与四项打分全部真跑，只把「本应上报的分数报文」打印出来。")
        print("=" * 72)

    run_evaluation()

    # 收尾小结：四项分数挂在同一条 trace 上，才能在看板里按版本、按时间对比。
    print("\n小结：检索看 CP / CR（找得准不准、全不全），生成看 Faithfulness / AR（编没编、切不切题）；")
    print("      四项分数都挂在同一条 trace 上，才能在 Langfuse 看板里按版本、按时间对比。")

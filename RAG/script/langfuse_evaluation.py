# --- 路径引导：确保能导入 Python_Base 根目录的 config 与 RAG 内部包 ---
import sys as _sys
from pathlib import Path as _Path

_BASE = _Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
_sys.path.insert(0, str(_BASE))          # Python_Base 根（config.py）
_sys.path.insert(0, str(_BASE / "RAG"))  # RAG 根（core/llm/pipeline 等包）
"""Langfuse 实验评估脚本(优化篇课案「RAG 评估 · 实验评估脚本」)。

对数据集每条样本跑一次 Agentic RAG 链路并自动创建 trace;
条目级评估器把 Recall@K、单题延迟/检索次数、Ragas 四指标写回该题 trace;
运行级评估器把批次聚合指标(均值、P95 延迟、平均检索次数)写入 dataset run;
结果直接在 Langfuse UI 按 run 对比。

Ragas 四指标(Context Precision / Context Recall / Faithfulness / Answer Relevancy)
用 ragas 0.4.3 的 collections API,评估模型与线上生成模型分开配置;
ragas 未安装或单指标失败时跳过该指标,不拖垮整个 run。

⚠ 「评估模型与线上生成模型分开配置」这句自述与实现**不符**(见报告):
`_ragas_bundle()` 里 `llm_factory(settings.llm.model, ...)` 用的就是
**生成链路同一个模型**。课案刻意让评估模型与生成模型分家(避免"自己批改自己"),
本实现没有做到 —— 引用 Ragas 分数时要知道这一点(存在自我偏好风险)。

★ **本脚本跑的是 agentic 链路**,与 `run_stage_eval.py` 的分阶段评估**不是同一套东西**:
    run_stage_eval   → pipeline.rag_pipeline(路由/筛选/召回/重排的确定性指标)
    langfuse_eval    → agentic.finance_agent(带工具调用与上下文管理的 Agent)
两者的链路、指标口径、样本消费方式都不同,**数字不可互相印证**。
"忠实性(Faithfulness)"这类指标目前**只在** agentic 这条路上有。

★ **本机部署的两个已知坑**(踩过):
1. 自部署的是 **v4 实例、以 `events_only` 模式**跑,旧版 SDK 的 `trace.list` 会 **404**;
2. 因此写回评估结果只能走「数据集实验(dataset experiment)」这条路 ——
   也就是本脚本走的方式,别再按老文档去找 trace 列表接口。

用法:
    uv run python RAG/script/langfuse_evaluation.py                       # run_name=baseline-top5
    uv run python RAG/script/langfuse_evaluation.py --run-name topk8      # 改配置后换名重跑
    uv run python RAG/script/langfuse_evaluation.py --max-concurrency 2

`--run-name` 是**这套评估的核心用法**:一次 run 对应一组配置,
改配置就换名字重跑,然后在 UI 里按 run 对比 —— 这是"单变量变更 + 可对比证据"
在评估工具上的落地方式(与阈值标定"改一个变量、复验一次"是同一套纪律)。
"""

import argparse
import logging
import math
import statistics
import sys
import time
import types
import warnings
from collections import defaultdict
from typing import Any

from config import eval_llm_target, settings
from evaluation.retrieval_metrics import recall_at_k

logger = logging.getLogger(__name__)

# 条目级 Recall@K 的 K。写成模块常量而不是命令行参数:它必须与
# **数据集里 relevant_ids 的规模**、以及线上检索的 top_n 口径匹配,
# 不该随每次执行变化(否则历史 run 之间不可比)
RECALL_K = 5
# 进度条对象(全局,由 main 在跑实验前赋值为 tqdm 实例)。
# 用**全局**是被 SDK 的回调签名逼的:rag_task / evaluator 都只收到 SDK 传的参数,
# 没有地方能挂自己持有的对象。类型标注写 Any 就是为了避免在这里 import tqdm
# (tqdm 只在 main 里按需导入)
_progress: Any = None
# Ragas 四指标实例的**进程级缓存**:None=还没建、{}=建过但不可用(缺依赖)。
# 两种状态必须区分 —— 用 None 表示"未初始化"才能只在第一次调 _ragas_bundle 时
# 尝试 import ragas(那一步会连带触发下面那段 VertexAI 补丁)
_ragas_metrics: dict | None = None


def _patch_langchain_community_vertexai() -> None:
    """给 ragas 的导入链补上 langchain-community 已移除的 VertexAI 符号。

    ragas 0.4.3 的 `ragas.llms.base` 直接 `from langchain_community.chat_models.vertexai
    import ChatVertexAI`,而 langchain-community 0.4.x 已删除该模块 ——
    不补丁就 `ModuleNotFoundError`。这里注入占位类仅满足导入,
    评估走 OpenAI 兼容端点,永远不会实例化它们。

    这是**依赖版本不匹配的绕行方案**(monkeypatch 第三方包的导入链),
    三条纪律必须守住:
    1. **只改 sys.modules 与属性,不改 ragas 的源码**:不写进 site-packages,
       升级 ragas 后不会留下孤儿补丁;
    2. **先试真 import 再补丁**(`try: ... return`):目标模块若在新版本里回来了,
       立刻走原生路径,补丁自动失效 —— 这保证了"补丁是幂等且自退役的";
    3. **只注入类名,不实现行为**:`class ChatVertexAI: pass` 里连 `__init__` 都没有。
       不实现是刻意的 —— 一旦有人真的实例化它,失败应当**立刻暴露**,
       而不是拿着一个假对象跑出错误的评估结果。

    `sys.modules[...] = module` 与 `langchain_community.chat_models.vertexai = module`
    两句都要写:前者让 `from ... import ...` 能解析,后者让
    `import langchain_community.chat_models; ....vertexai.X` 这种属性访问也能解析。
    """
    with warnings.catch_warnings():
        # 局部静音 DeprecationWarning:真模块若还在,导入它本身会告警 ——
        # 而我们要的只是"它能不能 import"这个事实,告警属于噪声。
        # 注意用 catch_warnings 上下文管理器:退出时自动恢复全局过滤器状态,
        # 不会污染其它模块的告警行为(比全局 filterwarnings 干净)
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            from langchain_community.chat_models.vertexai import ChatVertexAI  # noqa: F401
            from langchain_community.llms import VertexAI  # noqa: F401

            # 新版本里模块还在 ⇒ 什么都不用做,直接返回(补丁自退役)
            return
        except ModuleNotFoundError:
            pass

        import langchain_community.chat_models
        import langchain_community.llms

        class ChatVertexAI:  # 占位,仅满足 ragas.llms.base 的导入列表
            pass

        class VertexAI:  # 占位,同上
            pass

        module = types.ModuleType("langchain_community.chat_models.vertexai")
        module.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = module
        langchain_community.chat_models.vertexai = module
        langchain_community.llms.VertexAI = VertexAI


def _ragas_bundle() -> dict:
    """惰性创建 Ragas 四指标实例(collections 新 API);ragas 缺失时返回空字典。

    **惰性 + 缓存**的原因:import ragas 会连带触发上面那段 VertexAI 补丁,
    而且要构造两个 AsyncOpenAI 客户端。这些成本只在真要用 Ragas 时付一次。

    返回 `{}`(而不是抛异常)作为"Ragas 不可用"的记号:
    调用方 `ragas_evaluator` 看到空字典就直接返回 [] —— 于是**没有 ragas 的机器上
    本脚本仍能跑完 Recall@K 与运行级指标**。评估工具的局部不可用不该让整轮评估失败。

    ⚠ 这里 catch 的是 `Exception` 而不是 ImportError:ragas 版本差异导致的失败
    可能表现为 AttributeError(API 改名)、TypeError(参数变了)等。
    代价是**代码自身的 bug 也会被当成"Ragas 不可用"静默跳过**(只留一行 warning),
    见报告。

    四个指标的分工(决定了它们各自需要什么输入,见 ragas_evaluator):
    - Context Precision / Context Recall:都要 `reference`(标准答案)+ 证据;
    - Faithfulness:要回答 + 证据(判断"回答是否只依据证据");
    - Answer Relevancy:只要回答(**不需要证据**)——
      这也是唯一一个"没有证据时仍然算"的指标(见下面的 has_context 判断)。
    """
    global _ragas_metrics
    # 用 `is not None` 判而非真值判:{} 是"建过但不可用",不该重复尝试 import
    if _ragas_metrics is not None:
        return _ragas_metrics

    _patch_langchain_community_vertexai()
    try:
        from openai import AsyncOpenAI
        from ragas.embeddings import OpenAIEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
    except Exception as exc:  # noqa: BLE001 缺依赖/版本不符时降级为不做 Ragas
        logger.warning("Ragas 不可用,跳过四指标(仅保留 Recall@K 与运行指标): %s", exc)
        # 记成 {} 而不是保持 None:否则**每条样本**都会重试一次 import(几百次无谓开销)
        _ragas_metrics = {}
        return _ragas_metrics

    # 评判模型走 `eval_llm_target()`（配置 `EVAL_LLM_*`）—— 课案要求"评估模型与生成模型
    # **分开配置**"，理由是避免自我偏好：同一个模型给自己的回答打分偏高。
    # 未配置 `EVAL_LLM_MODEL` 时回退生成模型（行为同以前），但会在下面打一条 WARNING。
    #
    # ⚠ 评判模型必须**显式关思考**，否则四指标会大面积失败（真机实测的坑）：
    #
    #   | 配置                       | finish | reasoning | content | completion | JSON 可解析 |
    #   |---------------------------|--------|-----------|---------|------------|------------|
    #   | 默认 1024                  | length | 2096      | 0       | 1023       | ❌          |
    #   | max_tokens=4096            | stop   | 3269      | 783     | 1946       | ✅（勉强）   |
    #   | 4096 + thinking=disabled   | stop   | 0         | 611     | 264        | ✅          |
    #
    #（探针 `RAG\script\probe_ragas_judge_budget.py`；ragas 的 `InstructorModelArgs` 默认
    # `max_tokens=1024`，而评分模型的 completion_tokens 里**思考占大头**。）
    #
    # 只把 max_tokens 提到 4096 **不够**：36 条里仍有 18 条刷
    # `The output is incomplete due to a max_tokens length limit`
    #（真实评判提示词比探针里的更长，思考轻松超 4096）。关掉思考后 completion 只要 ~264，
    # 且输出是完整 JSON —— 这也是四指标的可靠性前提。
    # `enable_thinking` 只对 Qwen 系网关有效，本端点认的是 `thinking`（与路由同一处实测）；
    # 换成非思考模型（如 `deepseek-chat`）时这个字段是无害的多余参数。
    judge = eval_llm_target()
    if judge["from_fallback"]:
        logger.warning(
            "未配置 EVAL_LLM_MODEL：本次评判用的就是生成模型 %s —— 存在自我偏好风险，"
            "课案要求评估模型与生成模型分开配置（.env 里设 EVAL_LLM_MODEL 即可）",
            judge["model"],
        )
    else:
        logger.info("评判模型: %s @ %s（与生成模型分开配置）", judge["model"], judge["base_url"])
    # 同时 print 一行：本脚本的 logger 级别在生产运行时可能只放 WARNING/ERROR，
    # 而"这次到底用哪个模型评判的"是评估结论可信度的前提，必须让人在终端看得到。
    print(
        f"评判模型: {judge['model']} @ {judge['base_url']}"
        + ("（⚠️ 未配置 EVAL_LLM_MODEL，回退生成模型 ⇒ 有自我偏好风险）" if judge["from_fallback"] else "（与生成模型分开配置）")
    )
    eval_llm = llm_factory(
        judge["model"],
        client=AsyncOpenAI(api_key=judge["api_key"], base_url=judge["base_url"]),
        max_tokens=judge["max_tokens"],
        extra_body={"thinking": {"type": "disabled"}},
    )
    # 评估用的 embedding 与生成链路**同源**(同一个 bge-m3 空间)。
    # `api_key or settings.llm.api_key` / `base_url or ...` 的双重兜底:
    # 本项目 embedding 与 llm 走的是同一家网关,但配置上留了独立字段 ——
    # 若 embedding 段没配,就退回 llm 段(而不是带着 None 去建客户端报错)
    eval_embeddings = OpenAIEmbeddings(
        client=AsyncOpenAI(
            api_key=settings.embedding.api_key or settings.llm.api_key,
            base_url=settings.embedding.base_url or settings.llm.base_url,
        ),
        model=settings.embedding.model,
    )
    # 字典的**键就是写回 Langfuse 的指标名**(见 ragas_evaluator 的 Evaluation(name=name)),
    # 改名等于改 UI 里的指标名、会让历史 run 对不上 —— 别随手改
    _ragas_metrics = {
        "Context Precision": ContextPrecision(llm=eval_llm),
        "Context Recall": ContextRecall(llm=eval_llm),
        "Faithfulness": Faithfulness(llm=eval_llm),
        "Answer Relevancy": AnswerRelevancy(llm=eval_llm, embeddings=eval_embeddings, strictness=1),
    }
    return _ragas_metrics


def rag_task(*, item: Any, **kwargs: Any) -> dict:
    """实验任务:对单条数据集样本跑一次 Agentic RAG 链路。

    签名用 `*, item, **kwargs`(关键字专用 + 吞掉多余参数)是**为了适配 SDK**:
    `run_experiment` 会按它自己的约定传参(可能多带 item_index 等),
    不接受多余关键字就会 TypeError。`**kwargs` 在这里是接口适配,不是偷懒。

    `isinstance(item.input, dict)` 的双分支:上传时 `input={"question": ...}`
    是个 dict(见 upload_langfuse_dataset),但**手工在 UI 里建的数据项**
    input 可能是字符串 —— 两条路都要能跑。

    ★ `use_cache=False` 是刻意传的(与 run_stage_eval 同一理由):
    评估要量链路能力,不能因为缓存命中就秒回(那条数据在指标上会变成"零检索、
    零延迟"的假样本,把均值拉低、Recall 记 0)。

    `finally: _progress.update(1)` 而不是在 try 之后 update:
    链路**抛异常时也要推进度条**,否则一条样本失败就卡住进度显示,
    让人误判"整轮跑不动了"。

    返回的 dict **就是**后续 evaluator 拿到的 `output`,字段是三者共享的契约:
    - `retrieved_ids`:条目级 Recall@K 用(与数据集里 metadata.relevant_ids 比);
    - `retrieved_contexts`:Ragas 的证据正文。取 `ocr_text` 优先、回落 `semantic_text`
      —— 与 `agentic/finance_agent.py`、`generate_eval_samples.py` 同口径
      (注意 `pipeline/rag_pipeline.py` 走的是 semantic_text,两条路径口径分叉);
    - `latency_s` / `search_queries`:单题延迟与检索轮数,既写条目级也参与批次聚合。

    `agent_error` 透传 `result.get("error")`:让批次汇总能统计
    "有多少条其实报错了"(否则报错的条会被当成正常条目计入均值)。
    """
    from agentic.finance_agent import answer_query_agentic

    question = item.input["question"] if isinstance(item.input, dict) else str(item.input)
    started = time.perf_counter()
    try:
        result = answer_query_agentic(query=question, use_cache=False)
    finally:
        # 放在 finally 里:链路抛异常时进度条也要前进,否则看起来像卡死
        if _progress is not None:
            _progress.update(1)

    docs = result.get("docs", [])
    return {
        "question": question,
        "response": result.get("answer", ""),
        "retrieved_ids": result.get("candidate_ids", []),
        # 上下文正文的抽取:每条 doc 取 ocr_text 优先、semantic_text 回落,
        # 用 **生成器表达式 + if text.strip()** 过滤空串。
        # 这一步不是为了好看(见下面的注释)——
        "retrieved_contexts": [
            text
            for text in (
                (doc.get("ocr_text") or doc.get("semantic_text") or "") for doc in docs
            )
            # 过滤空串:留下 "" 会让 Ragas 与下面的 has_context 把"没检索到内容"
            # 误判成"有证据"。
            if text.strip()
        ],
        "latency_s": round(time.perf_counter() - started, 3),
        "search_queries": len(result.get("queries", [])),
        # 预留给将来的策略字段,当前恒为 None(agentic 链路不做 query 改写策略选择)。
        # 之所以**显式留一个 None 字段**而不是省略:下游/UI 按固定键读时不会 KeyError
        "strategy": None,
        # 缓存/FAQ 标记:虽然本脚本传 use_cache=False,但 Agent 内部可能命中 FAQ 层,
        # 记下来才能发现"某几条其实没走完整链路"
        "from_cache": result.get("from_cache", False),
        "from_faq": result.get("from_faq", False),
        "agent_error": result.get("error"),
        # token 用量与成本（T5）。`tokens` 是这一轮**整条链路**的累加
        # （主 Agent + evidence-analyst 子代理 + 工具里的 LLM 调用），见 agentic/usage.py。
        "tokens": result.get("tokens") or {},
        "cost": result.get("cost"),
    }


def item_evaluator(*, input: Any, output: dict, metadata: dict | None = None, **kwargs: Any) -> list:
    """条目级指标:Recall@K(相关 id 取自 metadata)、单题延迟与检索查询数。

    `metadata` 就是上传数据集时写的那个 dict(见 upload_langfuse_dataset),
    这里取 `relevant_ids` 做判分基准 —— 也就是说**判分依据在上传时就固化了**,
    评估脚本只消费它。改判分口径要去改上传侧(然后 --rebuild 重传)。

    `(metadata or {}).get("relevant_ids", [])` 的双重兜底:
    metadata 可能整个是 None(SDK 对没写 metadata 的数据项),也可能没有该键
    (手工建的数据项)。都没有时按空列表处理 ⇒ recall_at_k 会走"拒答样本"分支。

    返回三个 Evaluation(**不是** dict):这是 langfuse SDK 的条目级评估结果协议,
    `name` 就是 UI 里显示与聚合用的指标名。延迟与检索数在这里是**条目级**的,
    批次均值由 batch_run_evaluator 汇总。

    ⚠ 这三个指标里,后两个(latency_s / search_queries)只是**原始测量值**,
    它们不表达"好/坏",在 UI 里也没有阈值 —— 别把它们当质量分看。
    """
    from langfuse import Evaluation

    relevant_ids = (metadata or {}).get("relevant_ids", [])
    return [
        Evaluation(
            name="Recall@K",
            # 直接复用 evaluation/retrieval_metrics.recall_at_k —— 与分阶段评估同口径。
            # 注意这个口径下"拒答样本(relevant_ids 为空)不召回 = 1.0",
            # 所以 Recall@K 会受拒答样本影响(见 retrieval_metrics 的注释)
            value=recall_at_k(output.get("retrieved_ids", []), relevant_ids, k=RECALL_K),
        ),
        # float() 强转:SDK 的 Evaluation.value 只接受数值,而 output 里可能是 int/None
        Evaluation(name="latency_s", value=float(output.get("latency_s", 0.0))),
        Evaluation(name="search_queries", value=float(output.get("search_queries", 0))),
        # token 用量（T5）。名字固定成 tokens_in / tokens_out / tokens_total / llm_calls：
        # 改名等于改 UI 指标名、历史 run 会对不上（与 Ragas 四项同一约定）。
        Evaluation(name="tokens_in", value=float((output.get("tokens") or {}).get("input_tokens", 0))),
        Evaluation(name="tokens_out", value=float((output.get("tokens") or {}).get("output_tokens", 0))),
        Evaluation(name="tokens_total", value=float((output.get("tokens") or {}).get("total_tokens", 0))),
        Evaluation(name="llm_calls", value=float((output.get("tokens") or {}).get("llm_calls", 0))),
        # 成本：没填单价时 estimate_cost 返回 None ⇒ 这里**不出这一条**（避免把"没算"报成 0）
        *(
            [Evaluation(name="cost", value=float(output["cost"]))]
            if output.get("cost") is not None
            else []
        ),
    ]


async def ragas_evaluator(
    *, input: Any, output: dict, expected_output: Any = None, metadata: dict | None = None, **kwargs: Any
) -> list:
    """条目级 Ragas 指标:用证据正文与回答算四项生成质量分数。

    这个函数是**async**的(其他 evaluator 是同步的),因为 Ragas 的新 API 是
    `await metric.ascore(...)`。SDK 会按需 await 协程,所以混用没问题。

    `expected_output` 就是数据集里的 ground_truth(上传时的 expected_output 字段),
    作为 Ragas 的 `reference`(标准答案)。

    ★ 四项指标的输入是**刻意不同**的(见 `score_args`):
    - Context Precision / Context Recall 共用同一份 context_kwargs
      (user_input + retrieved_contexts + reference):它们衡量的是"检索到的证据好不好";
    - Faithfulness 用 question + response + contexts:衡量"回答有没有超出证据";
    - Answer Relevancy 只用 question + response:衡量"答得切不切题",**不需要证据**。
    把三者拆成三份 kwargs 而不是一份:硬塞会给 Ragas 传它不认的参数
    (不同指标接受的字段集合不同)。

    ★ `has_context` 的判断与跳过逻辑(这段是理解指标覆盖率的关键):
    Answer Relevancy **不需要证据**所以永不跳过;其余三个在"无证据"时跳过(**而不是
    记 0 分**)。选择跳过而不是记 0 的理由:没检索到证据时,Faithfulness 之类的
    指标**在数学上没有定义**(没有证据可依据),记 0 会把它算成"回答不忠实",
    而真正的问题是**召回失败** —— 那是 Recall@K 该表达的。同理,Ragas 指标在
    UI 里的"覆盖率"会低于 100%,这不是漏算,是刻意的。

    单指标失败 `continue`(记 warning)而不是抛异常:
    一次 run 有几十条 × 4 个指标,任何一次 LLM 抖动都可能让某条失败;
    让单指标失败拖垮整个 run = 前面所有 API 调用白费。

    `score is None or isnan(score)` 单独判 NaN 再 skip:NaN 在 JSON 里会变成 null,
    UI 上显示为缺口;而 0.0 会被认真当成"最差分"。两者语义完全不同,不能混。

    `strictness=1`(在 _ragas_bundle 里)是 Answer Relevancy 的参数:
    控制"生成几个问题来反问回答的相关性",1 = 最省调用;
    这是**成本与稳定性**的取舍(值越大越稳但调用越多)。
    """
    from langfuse import Evaluation

    metrics = _ragas_bundle()
    # 空字典 = ragas 不可用(缺依赖/版本不符),静默返回 []:
    # 本脚本的其他指标(Recall@K、批次聚合)仍然有效 —— 局部降级,不整体失败
    if not metrics:
        return []

    context_kwargs = dict(
        user_input=output["question"],
        retrieved_contexts=output["retrieved_contexts"],
        # `str(expected_output or "")`:reference 必须是字符串,
        # 而拒答/直答类样本的 expected_output 可能是 None
        reference=str(expected_output or ""),
    )
    score_args = {
        "Context Precision": context_kwargs,
        "Context Recall": context_kwargs,
        "Faithfulness": dict(
            user_input=output["question"],
            response=output["response"],
            retrieved_contexts=output["retrieved_contexts"],
        ),
        "Answer Relevancy": dict(user_input=output["question"], response=output["response"]),
    }

    evaluations = []
    has_context = bool(output.get("retrieved_contexts"))
    for name, metric in metrics.items():
        if not has_context and name != "Answer Relevancy":
            # 用 info 而不是 warning:这是**预期内的正常跳过**(拒答样本),
            # 打成 warning 会让日志看起来很吓人
            logger.info("无证据,跳过 Ragas 指标 %s(问题: %s)", name, output["question"])
            continue
        try:
            result = await metric.ascore(**score_args[name])
            # 新 API 返回的是**结果对象**(带 .value),不是裸分数 —— 取错层会得到
            # 一个不可序列化的对象
            score = result.value
        except Exception as exc:  # noqa: BLE001 单指标失败不拖垮整个 run
            logger.warning("Ragas 指标 %s 打分失败(问题: %s): %s", name, output["question"], exc)
            continue
        if score is None or (isinstance(score, float) and math.isnan(score)):
            # NaN 与 None 都要跳过:NaN 写进 Langfuse 会变成缺口/非法值。
            # `isinstance(score, float)` 先判再 isnan 是必须的 ——
            # math.isnan 对 int 也能用,但对 None 会 TypeError
            logger.info("Ragas 指标 %s 分数为 NaN,跳过(问题: %s)", name, output["question"])
            continue
        evaluations.append(Evaluation(name=name, value=float(score)))
    return evaluations


def p95(values: list[float]) -> float:
    """用 nearest-rank 方法计算 P95。

    nearest-rank 的取法是 `ceil(0.95 * n) - 1`(转成 0 基下标)。
    为什么用它而不是插值分位数:延迟是**离散的实测值**,
    "第 95 百分位的观测值"比插值出来的数字更好解释(它就是某一次真实请求的耗时)。

    `max(0, ...)` 兜住 n=0 时算出的 -1:
    不过调用方在 outputs 为空时已经提前 return 了,这里的兜底属于防御性写法。
    注意 n=1 时结果是 `values[0]`(P95 退化成最大值),这是合理的。

    ⚠ 这种分位数在样本很少时**极不稳定**(28 条样本时 P95 ≈ 第 2 大的值)。
    批次聚合里同时给出 mean 与 p95、并把 batch_items 一起写进去,
    就是为了让读的人能判断"这个 P95 有多少样本支撑"。
    """
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def batch_run_evaluator(*, item_results: list, **kwargs: Any) -> list:
    """运行级汇总:批次均值、P95 延迟与平均检索查询数,写入 dataset run。

    运行级评估器与条目级评估器的区别:它在**整批跑完后**调一次,
    输入是所有条目的结果,输出挂在 dataset run 上(UI 里作为 run 的汇总指标)。
    阈值判断/对比看这一层,排查下钻看条目层。

    `outputs = [r.output for r in item_results if r.output]`:
    过滤掉**没有 output 的条目**(链路异常、SDK 侧失败),它们没有可聚合的数据。
    于是 `batch_items`(= 成功条目数)本身就是一个**有用的指标**:
    它小于数据集条数就说明有样本失败了,看 `batch_agent_error_count` 能进一步区分
    "Agent 报错"与"其他原因"。

    空批次直接返回一条 batch_items=0 并带 comment:
    比返回空列表好 —— UI 里能看到一条"无成功条目"的显式记录,
    而不是一个没有任何指标的 run(后者看着像"跑完了")
    """
    from langfuse import Evaluation

    outputs = [r.output for r in item_results if r.output]
    if not outputs:
        return [Evaluation(name="batch_items", value=0, comment="无成功条目")]

    # `statistics.fmean` 而不是 `sum/len`:fmean 是"浮点均值"的专门实现,
    # 精度与数值稳定性都更好(且空序列的处理语义明确)
    evaluations = [
        Evaluation(name="batch_items", value=float(len(outputs))),
        Evaluation(
            name="batch_mean_latency_s",
            value=statistics.fmean(o["latency_s"] for o in outputs),
        ),
        Evaluation(
            name="batch_p95_latency_s",
            value=p95([o["latency_s"] for o in outputs]),
        ),
        Evaluation(
            name="batch_mean_search_queries",
            value=statistics.fmean(o["search_queries"] for o in outputs),
        ),
        # 报错条数单独计数:均值类指标会**掩盖**报错(报错的条延迟短、检索 0 次,
        # 会把均值拉好看)。没有这个计数,"跑得更快"可能是"更多条报错了"
        Evaluation(
            name="batch_agent_error_count",
            value=float(sum(1 for o in outputs if o.get("agent_error"))),
        ),
    ]
    # token 与成本**不在这里显式写**：下面的"自动汇总所有条目级指标"会把条目级的
    # tokens_in / tokens_out / tokens_total / llm_calls（以及填了单价时的 cost）
    # 汇总成 batch_mean_tokens_* / batch_mean_cost —— 一处定义、两处生效，避免同义指标两套名字
    # （实测踩过：先显式加了一组 batch_mean_input_tokens，结果与自动汇总的
    #  batch_mean_tokens_in 同时在报告里出现，读者会不知道以哪个为准）。

    # 把**所有条目级数值指标**自动汇总成批次均值(而不是写死 Recall@K 一项):
    # 这样将来条目级加了新指标,批次侧自动就有 batch_mean_<name>,不用改两处。
    # 这正是下面那个排除列表存在的理由。
    by_name: dict[str, list[float]] = defaultdict(list)
    for result in item_results:
        for evaluation in result.evaluations:
            if evaluation.name in ("latency_s", "search_queries"):
                # 排除这两个:**延迟/检索数不是质量指标**,而且它们已经有
                # 专门的 batch_mean_latency_s / batch_mean_search_queries
                # (还带 P95),再生成一份 batch_mean_latency_s 会重名
                continue
            # `isinstance(..., (int, float))` 过滤非数值条目:
            # Langfuse 的 Evaluation 允许 string/boolean 值,强行 float() 会崩。
            # 注意 bool 是 int 的子类,会被放进来 —— 当前没有布尔指标,不影响
            if isinstance(evaluation.value, (int, float)):
                by_name[evaluation.name].append(float(evaluation.value))
    # sorted(...) 只为输出顺序稳定(便于看日志),不影响数值
    for name, values in sorted(by_name.items()):
        evaluations.append(Evaluation(name=f"batch_mean_{name}", value=statistics.fmean(values)))
    return evaluations


def _build_langfuse() -> Any:
    """按 settings 初始化 Langfuse 客户端;缺凭证时报错并指明环境变量名。

    凭证检查放在构造之前、并**在错误信息里点出变量名**:
    缺 key 时 Langfuse SDK 的报错与"服务没起/网络不通"很像,
    不指明就容易往错方向查(自部署实例的 host 也是常见坑)。

    返回 Any 而不是具体类型:避免在模块顶层 import langfuse
    (本模块要能被离线 import;SDK 导入放函数里,与 upload_langfuse_dataset 同款做法)。
    """
    if not settings.langfuse_public_key or not settings.langfuse_secret_key:
        raise ValueError(
            "缺少 Langfuse 凭证:请在 .env 配置 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY"
            "(可选 LANGFUSE_HOST)"
        )
    from langfuse import Langfuse

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Langfuse 实验评估(Ragas 四指标 + Recall@K)")
    # run-name 是"配置标识":改配置就换名重跑,UI 里按 run 对比 —— 见模块头
    parser.add_argument("--run-name", default="baseline-top5", help="标识一次配置,改配置后换名重跑")
    parser.add_argument("--description", default="", help="run 描述")
    # max_concurrency 默认 2:每条样本同时打**链路 + Ragas 四指标**的 API,
    # 并发调高会同时压 LLM 与 embedding 两个网关(以及本地 Milvus/Redis),
    # 反而更容易超时。2 是"能并行但不上量"的保守值
    parser.add_argument("--max-concurrency", type=int, default=2, help="并发数(链路与 Ragas 都打外部 API)")
    parser.add_argument("--dataset-name", default=settings.langfuse_dataset_name)
    parser.add_argument("--limit", type=int, default=0, help="只评估前 N 条(0 表示全部)")
    args = parser.parse_args()

    # basicConfig 在 main 里(而不是模块顶层)调:避免 import 本模块时
    # 顺手改掉**调用方**的 logging 配置。level=WARNING 是配套取舍 ——
    # 只显示 warning 以上,让 tqdm 进度条与 summary 输出保持干净
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    # Windows 控制台编码兜底:result.format() 里有 emoji
    # `errors="replace"` 而不是重设 encoding:能编多少编多少、编不了的换成 ?,
    # 绝不让打印把评估结果弄丢(评估已经跑完了,死在打印上最亏)
    sys.stdout.reconfigure(errors="replace")

    langfuse = _build_langfuse()
    dataset = langfuse.get_dataset(args.dataset_name)
    if args.limit > 0:
        # 就地截断:run_experiment 用的是 dataset 对象本身,只改本地 items 变量不生效
        # (之前这里截断了 items,下面又 get_dataset 取回全量,--limit 等于没写)。
        # ★ 这条注释记录的是一个**已修复的真 bug**,修复要点是"裁的对象必须
        #   正是下游要用的那一个"。
        # `list(dataset.items)` 先物化成列表再切片:dataset.items 可能是惰性序列,
        # 直接切片行为不保证(且原地赋值要求可迭代对象而不是生成器)
        dataset.items = list(dataset.items)[: args.limit]
    print(f"数据集 {args.dataset_name}: {len(dataset.items)} 条,run_name={args.run_name}")

    # `global _progress` 必须写在**赋值之前**的同一函数作用域里:
    # rag_task 是回调,只能通过模块级变量看到进度条对象
    global _progress
    from tqdm import tqdm

    # `ascii=True` 让进度条用 ASCII 字符:Windows 控制台在非 UTF-8 代码页下
    # 显示 Unicode 方块会乱码
    _progress = tqdm(total=len(dataset.items), desc="评估进度", ascii=True)
    try:
        from langfuse import Evaluation  # noqa: F401  确认 SDK 可用

        result = dataset.run_experiment(
            name="financial-agentic-rag",
            run_name=args.run_name,
            # `args.description or None`:空串转成 None —— SDK 对空描述的处理
            # 与 None 不同(空串会被当成"有描述但为空")
            description=args.description or None,
            task=rag_task,
            # 条目级可以是多个评估器(同步 + 异步混用),SDK 会分别调用并把结果合并
            evaluators=[item_evaluator, ragas_evaluator],
            # 运行级只在整批结束后调一次
            run_evaluators=[batch_run_evaluator],
            # metadata 挂在 run 上:给这次实验留一个可筛选的标签
            metadata={"split": "validation"},
            max_concurrency=args.max_concurrency,
        )
    finally:
        # 无论成功失败都要关掉进度条并**把全局置回 None**:
        # 不置回 None 的话,下一次在同一进程里跑实验会往一个已关闭的进度条上 update
        _progress.close()
        _progress = None

    # `result.format()` 打印条目级明细(含每条的问题与各项分数),
    # 是排查"哪条没召回/哪条被判低分"的第一手材料
    print(result.format())
    # flush 是**必须**的:Langfuse SDK 异步批量上报,不 flush 可能丢结果
    # (评估跑完了但 UI 里只有一半数据,是这类脚本最经典的坑)
    langfuse.flush()


if __name__ == "__main__":
    main()

"""评估记录器:把 RAG 流水线的事件流映射成评估 record。

对应课案「系统评估 · 第二步:记录每个阶段的真实输出」:
不只保存最终回答,还要保存路由判断、筛选条件、Milvus 过滤表达式、
关键词/向量召回结果、合并候选、重排顺序、返回类型与耗时。

为什么要"记录事件流"而不是让评估脚本自己拿链路对象:
生产链路的入口是 `pipeline/rag_pipeline.py::run_events(question)`,它把链路每一步
产出成一个事件(dict)。评估只需要**消费同一串事件**,不必关心链路内部实现 ——
链路改内部结构时,只要事件协议不变,评估侧就不用跟着改。
反过来,如果评估直接调内部函数拿中间变量,链路一重构评估就全碎。

核心设计:`record_from_events` 是**纯函数**(事件流 + 样本 + 耗时 → record),
不发起任何调用。这样它可以在 tests 里用假事件流离线验证,
而不用起 Milvus/Redis/LLM。

⚠ 两处已知的口径不一致(见报告「发现的问题」,本次只记录不改):
1. `cache_hit` 事件会把 `actual_route` 写成 **"cache"**,而 `evaluation/stage_metrics.py`
   的模块 docstring 与 tests 里出现过 **"faq"** 这个枚举值 —— 后者实际上永远不会被
   生产出来(没有任何代码路径写它)。判读报告时别按 "faq" 去筛记录。
2. 一条 record 的 `actual_route` 取值域实际是 {None, cache, rag, direct, error},
   但 `stage_metrics._RETRIEVAL_ROUTES` 只认 rag/search ⇒ 缓存命中一律算"没走检索"。
   这就是评估必须关缓存的原因(见 `script/run_stage_eval.py` 的 `use_cache` 注释)。
"""

from __future__ import annotations

import sys
from pathlib import Path

# --- 路径引导：直接运行本文件时也能导入 RAG 内部包(与 data_process/app 下的约定一致) ---
# 直接 `python RAG/evaluation/run_recorder.py` 跑自检时需要;被 import 时不生效
# (`from evaluation.eval_set import validate_sample` 要求仓库根在 sys.path 上)
_BASE = Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
for _p in (str(_BASE), str(_BASE / "RAG")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.eval_set import validate_sample  # noqa: E402


def _collect_candidate_ids(retrieve: dict) -> list[str]:
    """按流水线的合并顺序收集候选 id 并去重。

    pipeline 的合并顺序是:直接向量 → 直接关键词 → 改写向量 → 改写关键词。

    ⚠ **顺序是有语义的**,不要"顺手"改成按分数排序:多路召回是「原问题 + 改写」
    两路并集,顺序代表了流水线里合并的先后,而 Recall@K 用的是 `[:k]` 截断 ——
    这里若换了顺序,Recall@5 的结果会跟着变(前 5 个候选是谁变了)。
    真想改合并策略,应当改 pipeline 而不是在评估侧重排。

    只收 `row.get("id")` 为真的行:空 id 去重后会变成一个无意义的 "" 候选,
    混进 candidate_ids 里会稀释 `[:k]` 的窗口、白占一个名额。

    去重保留**首次出现**的位置(`seen.add` 后才 append),所以同一张票在
    向量路与关键词路都命中时,它以向量路的位置参与排序。
    """
    ordered = (
        retrieve.get("direct_vector_hits", [])
        + retrieve.get("direct_keyword_hits", [])
        + retrieve.get("rewrite_vector_hits", [])
        + retrieve.get("rewrite_keyword_hits", [])
    )
    seen: set[str] = set()
    ids: list[str] = []
    for row in ordered:
        row_id = row.get("id")
        if row_id and row_id not in seen:
            seen.add(row_id)
            ids.append(row_id)
    return ids


def record_from_events(events: list[dict], sample: dict, elapsed_s: float) -> dict:
    """把一次问答的事件流 + 评估样本合成一条评估 record。

    先 `validate_sample(sample)` 再取字段(而不是直接 `sample["expected_route"]`):
    这样评估脚本传了半成品样本时,会在**记录阶段**就带着明确错误信息炸掉,
    而不是产出一条字段缺失的 record 让下游指标静默算错。

    下面这个初始字典是**完整字段清单**,没有的事件类型就保留默认值。
    两个默认值特意选得能被识别为"没发生":
    - `actual_route = None`(不是 "") ⇒ 末尾 done 事件里用 `is None` 判断能不能补,
      若默认写成 "" 这个补位逻辑就失效了;
    - `conservative = False` ⇒ 只有收到 no_evidence 事件或者是 done 事件声明了保守回答
      才变 True。它是「证据不足时是否如实保守作答」的标记,与路由/召回指标无关。

    事件处理是**顺序覆盖 + 累积**,靠的是 `pipeline/rag_pipeline.py::run_events`
    的产出顺序稳定。几处关键行为:
    - `token` 事件用 `+=` **累积**成 answer(流式生成是一段一段吐的),
      而 `done` 事件若带了完整 answer 则**整体覆盖** —— 这条 else 分支
      (`event.get("answer") or record["answer"]`)保证了 `or` 左侧为空时不把累积结果清掉;
    - `retrieve` 事件里 `elapsed_s` 用 `is not None` 判断而不是真值判断:
      0.0 秒也是合法耗时,写成 `if event.get("elapsed_s")` 会把它丢掉
      (而"秒回"恰恰是最该被记录的异常值之一);
    - 只有 `retrieve`/`rerank` 两段写进 `stage_timings`(链路里只有这两段带耗时),
      所以 stage_timings 的键是**稀疏**的,别在别处按固定键去取。

    ⚠ `cache_hit` 事件会把 actual_route 写死成 "cache" 并直接填 answer/sources ——
    这正是"缓存命中会短路掉后面所有阶段"在记录层的体现:一旦出现这条事件,
    同一条 record 的 candidate_ids / reranked_ids 都会保持空列表,
    于是召回与重排指标必然为 0。判断一份报告是否被缓存污染,看这个字段最快。
    """
    sample = validate_sample(sample)

    record = {
        "question": sample["question"],
        "task_type": sample["task_type"],
        "expected_route": sample["expected_route"],
        "expected_filter": sample["expected_filter"],
        "expected_ticket_ids": sample["expected_ticket_ids"],
        "ground_truth": sample["ground_truth"],
        "need_citation": sample["need_citation"],
        "actual_route": None,
        "actual_filters": {},
        "filter_expr": "",
        "filter_fallback": False,
        "candidate_ids": [],
        "reranked_ids": [],
        "answer": "",
        "cache_hit": None,
        "conservative": False,
        "sources": [],
        "elapsed_s": elapsed_s,
        "stage_timings": {},
        "rewrite": None,
        "method": None,
    }

    for event in events:
        kind = event.get("type")

        if kind == "cache_hit":
            # 缓存命中:链路在这里就结束了,不会再产出 route/retrieve/rerank 事件。
            # `mode` 是 exact(精确键命中)或 preset(FAQ 相似度命中),写进 record 后
            # 就是 stage_metrics 里的 cache_hit_count
            record["actual_route"] = "cache"
            record["cache_hit"] = event.get("mode")
            record["answer"] = event.get("answer", "")
            record["sources"] = event.get("sources", [])

        elif kind == "route":
            # 路由判定;rewrite 是改写策略(direct / hyde / multi_query 之类),
            # 它只影响检索 query 的构造,不参与任何指标计算,仅留档排障用
            record["actual_route"] = event.get("route")
            record["rewrite"] = event.get("rewrite")

        elif kind == "rewrite":
            # 注意与上面 route 事件里的 rewrite 字段**同名不同源**:
            # 这里取的是改写方法的细节(method),写进 record["method"]。
            # 两条事件谁后到谁生效,别把它们当成同一个东西
            record["method"] = event.get("method")

        elif kind == "retrieve":
            # `or {}` 兜住 None:链路零命中回退时可能把 filters 置空/置 None,
            # 而 filter_field_accuracy 会对它做 .items(),None 会直接抛 AttributeError
            record["actual_filters"] = event.get("filters", {}) or {}
            record["filter_expr"] = event.get("filter_expr", "")
            # bool() 强制成布尔:上游可能传 0/1,而报告里 filter_fallback_count
            # 是按真值求和,保持类型一致才好排查
            record["filter_fallback"] = bool(event.get("filter_fallback"))
            record["candidate_ids"] = _collect_candidate_ids(event)
            if event.get("elapsed_s") is not None:
                record["stage_timings"]["retrieve"] = float(event["elapsed_s"])

        elif kind == "rerank":
            # 只取重排**保留**的票(`kept`),被阈值卡掉的票不进 record ——
            # 想看被卡掉哪些,得回原始事件流,record 里刻意不留
            record["reranked_ids"] = [row.get("id") for row in event.get("kept", []) if row.get("id")]
            if event.get("elapsed_s") is not None:
                record["stage_timings"]["rerank"] = float(event["elapsed_s"])

        elif kind == "token":
            # 流式生成的一个片段。事件协议里的字段名是 **text** 而不是 content/delta
            # (踩过:按错字段名读会"跑通了但答案长度 0",误判成链路 bug)
            record["answer"] += event.get("text", "")

        elif kind == "error":
            # 出错也留一条 record(而不是丢弃):这样评估的分母与实跑条数一致,
            # 出错率能从「answer 里混着错误信息」「actual_route == error」上看出来。
            # 丢 record 会让指标只统计成功样本,分数虚高
            record["actual_route"] = "error"
            record["answer"] += event.get("message", "")

        elif kind == "no_evidence":
            # 检索不到证据的显式信号:置 conservative,用于统计"保守回答"的比例。
            # 它**不改** actual_route —— 走了检索就是走了检索,答不答得出来是另一回事
            record["conservative"] = True

        elif kind == "done":
            # 收尾事件是"最终真相",优先级高于前面的增量事件,所以做的是**覆盖**:
            # answer 用 `or` 保留已累积的流式内容(只在 done 没给答案时才用旧的);
            # sources 直接把 done 的列表当**默认值**传,即 done 给了就用 done 的
            record["answer"] = event.get("answer") or record["answer"]
            record["sources"] = event.get("sources", record["sources"])
            record["conservative"] = bool(event.get("conservative", record["conservative"]))
            if event.get("cache_hit"):
                # done 里也带 cache_hit 时以它为准(自检 demo 里传的是 None,故不覆盖)
                record["cache_hit"] = event["cache_hit"]
            if record["actual_route"] is None and event.get("route"):
                # 只在前面没收到 route 事件时补 —— 这么写是为了兼容
                # 「链路只发 done、不发 route」的更短实现
                record["actual_route"] = event["route"]

    return record


# 自检:用一段最小事件流覆盖 route/retrieve/rerank/done 四条分支,钉住两件事 ——
# 候选 id 按**合并顺序**收集(而非按分数)、两段耗时都进了 stage_timings。
# 注意 demo 里 `"cache_hit": None` 是必须的:done 分支用 `if event.get("cache_hit")` 判真,
# 传 None 才走不到覆盖分支,这与真实链路的产出形状一致。
if __name__ == "__main__":
    demo_events = [
        {"type": "route", "route": "rag", "rewrite": "direct"},
        {
            "type": "retrieve",
            "filters": {"person": "张三"},
            "filter_expr": 'person == "张三"',
            "filter_fallback": False,
            "direct_vector_hits": [{"id": "t1"}],
            "direct_keyword_hits": [],
            "rewrite_vector_hits": [],
            "rewrite_keyword_hits": [],
            "elapsed_s": 0.4,
        },
        {"type": "rerank", "kept": [{"id": "t1"}], "elapsed_s": 0.6},
        {"type": "done", "answer": "共 1 张", "sources": [{"id": "t1"}], "cache_hit": None, "route": "rag"},
    ]
    sample = {
        "question": "张三的火车票有几张?",
        "task_type": "retrieval",
        "expected_route": "search",
        "expected_ticket_ids": ["t1"],
        "ground_truth": "共 1 张",
    }
    record = record_from_events(demo_events, sample, elapsed_s=1.1)
    assert record["candidate_ids"] == ["t1"], record
    assert record["stage_timings"] == {"retrieve": 0.4, "rerank": 0.6}, record
    print("run_recorder.py 自检通过")

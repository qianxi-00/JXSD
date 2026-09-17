"""分阶段指标:对照基础篇课案「系统评估 · 第三步:按阶段计算指标」。

课案强调"不能只看最终回答",要按链路逐段算指标:

    agent 路由 -> 票据筛选变量抽取 -> 多路召回 -> 重排序 -> 生成回答

因此本模块的输入是一批 **record**,每条对应一次真实问答的完整链路输出:

    {
      "question": str,
      "task_type": str,                     # faq / retrieval / structured_aggregation / generation
      "expected_route": "search"|"direct"|None,
      "actual_route": "rag"|"direct"|"cache"|"faq"|"error",
      "expected_filter": dict,              # 期望抽取出的筛选条件
      "actual_filters": dict,               # pipeline.filters.extract_ticket_filters 的真实输出
      "expected_ticket_ids": [str],         # 期望召回到的票据 ID
      "candidate_ids": [str],               # 多路召回合并去重后的候选(按相关性降序)
      "reranked_ids": [str],                # 重排后保留的票据 ID
      "ground_truth": str,                  # 人工确认的标准答案
      "answer": str,                        # 最终回答
      "filter_expr": str,                   # 转换出的 Milvus 过滤表达式(留档)
      "filter_fallback": bool,              # 是否触发了"过滤零命中回退"
      "cache_hit": str|None,                # exact / preset / None
      "elapsed_s": float,
      "stage_timings": dict,
    }

指标定义:
- 路由准确率:expected_route == "search" 时要求真的走了检索(actual_route == "rag"),
  期望 direct 时要求没有走检索(缓存/FAQ/直接回答都算没走检索);
- 筛选字段准确率:逐字段比较 expected_filter 与 actual_filters;
- Recall@K / Rerank Hit@K:复用 evaluation.retrieval_metrics(与微调章同一口径);
- 答案准确性:把标准答案里的金额/日期/票号/数量抽成关键事实,看回答覆盖了多少。

为什么要有这一层「record → 指标」的纯函数:
record 由 `evaluation/run_recorder.py` 从链路事件流落盘成 jsonl,**指标必须能从落盘
文件离线复算**。否则调一次指标就得重跑链路(单题秒级到分钟级、还要外部 API),
成本高到没人愿意调参,只能"大致看着还行"。这个模块不碰网络、不碰数据库,
输入是 list[dict]、输出是 float/dict,所以可以在 `script/run_stage_eval.py` 之外
被单测直接喂假 record(本仓 tests 里就是这么做的)。

⚠ 有一条口径要特别注意(踩过):`actual_route` 取到 `cache` / `faq` 时,
下面的 `_RETRIEVAL_ROUTES` 里没有它们 ⇒ 一律记成「没走检索」。对期望 direct 的样本
这是对的,但对期望 search 的样本,一次**缓存误命中**会同时把路由判错、并让后面
召回/重排系列指标一起变成 0(因为没有候选)。这正是 `script/run_stage_eval.py`
默认 `use_cache=False` 的原因 —— 评估阶段评估的是链路,不是缓存。看到一份报告里
route / filter / recall / rerank 数值完全相同(如都是 0.875),先怀疑有 `cache_hit`
记录混进来了,而不是链路四处一起坏。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# --- 路径引导：直接运行本文件时也能导入 RAG 内部包(与 data_process/app 下的约定一致) ---
# 为什么需要:本文件既能被 import,也能 `python RAG\evaluation\stage_metrics.py` 直接跑
# (文件尾带自检)。直接跑时 sys.path[0] 是 evaluation\ 目录,`from evaluation.xxx import`
# 会 ModuleNotFoundError;这里从文件位置向上找到仓库根,再把「仓库根」与「仓库根/RAG」
# 两处塞进 sys.path。`_BASE.name != "Python_Base"` 是向上查找的**停止条件**,
# 目录一旦改名/搬走,循环会一直退到盘符根(见报告「发现的问题」)。
_BASE = Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
for _p in (str(_BASE), str(_BASE / "RAG")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluation.retrieval_metrics import recall_at_k as _recall_at_k  # noqa: E402

# 检索类路由值:出现这些值表示"走了知识库检索"
# `search` 也在集合里 —— 这个字段既可能由 run_recorder 写成 "rag",
# 也可能来自别的来源写成 "search",两个都认;cache/faq/direct 不在此集合 = 没走检索
_RETRIEVAL_ROUTES = {"rag", "search"}

# 下面四组正则就是「关键事实抽取」的全部定义 —— 改这里等于改答案准确性指标的口径,
# 会让历史报告与新报告不可比,所以别为了某一条样本顺手放宽正则
_AMOUNT_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*元")
_DATE_RE = re.compile(r"(\d{4})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})\s*日?")
# 票号:用 ASCII 数字字母边界判定,不能用 \b —— 中文也属于 \w,
# "发票INV20250101开票日期" 里 票 与 I 之间没有 \b,会漏掉票号
# 两条分支:A) 至少 2 个大写字母 + 至少 5 位数字(INV20250101 / CA1276 这类真票号);
#         B) T + 至少 10 位数字(火车票号)。**大小写敏感** —— 回答里把票号写成小写
#         就抽不到,这条指标对 LLM 的输出格式有隐性要求
_TICKET_NO_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{2,}\d{5,}|T\d{10,})(?![A-Za-z0-9])")
_COUNT_RE = re.compile(r"(\d+)\s*张")


def extract_facts(text: str) -> set[str]:
    """抽关键事实并归一化:金额(两位小数)/日期(YYYY-MM-DD)/票号/票据数量。

    「归一化」是这段代码的全部价值所在:标准答案写「1,280.00 元」、回答写「1280元」,
    字符串比对必然不等,但它们是同一个事实。所以每个事实统一变成 `类别:规范形式`
    的字符串再放进集合:
      - 金额先去掉千分位逗号,再 `float(...):.2f` 补两位小数;
      - 日期统一 YYYY-MM-DD(int 抹掉前导零差异,再补回两位 ⇒ `2025/3/5` 与 `2025-03-05` 等价);
      - 票号原样(不做大小写折叠,理由见上面的正则注释);
      - 数量转 int(`03 张` 与 `3 张` 等价)。

    `text or ""` 是容错 None(record 缺字段时不该让整批指标崩掉);
    返回 **set** 而非 list:重复事实只算一次,避免回答里把同一金额说三遍就刷高覆盖率。

    ⚠ 已知边界(属口径而非 bug,刻意不修):金额正则要求数字紧跟「元」,所以
    「1280」这种不带单位的写法抽不到;反过来,若某个纯数字串后面恰好跟「元」,
    它会被当成金额。抽取是**统计性**的,不适合用来做逐条严格判分。
    """
    facts: set[str] = set()

    for raw in _AMOUNT_RE.findall(text or ""):
        # 先去千分位逗号再转 float:`1,280.00` → 1280.00
        facts.add(f"金额:{float(raw.replace(',', '')):.2f}")

    for year, month, day in _DATE_RE.findall(text or ""):
        # int() 抹掉前导零,再 :02d 补回两位 ⇒ 前导零写法的差异被抹平
        facts.add(f"日期:{int(year):04d}-{int(month):02d}-{int(day):02d}")

    for ticket_no in _TICKET_NO_RE.findall(text or ""):
        facts.add(f"票号:{ticket_no}")

    for count in _COUNT_RE.findall(text or ""):
        facts.add(f"数量:{int(count)}")

    return facts


def fact_hit_rate(answer: str, facts: set[str]) -> float:
    """回答覆盖了多少标准答案的关键事实(无事实可考时记 1.0)。

    直接对两个 set 取交集 `facts & extract_facts(answer)` —— 上一步的归一化
    让「字符串相等」在这里正好等于「事实相同」。

    两个刻意的取值:
    - 标准答案里抽不出事实(如拒答样本的「未找到相关票据信息。」)时返回 **1.0 而不是
      0.0**:这条样本属于「不可考」,不是「答错了」,记 0 会把拒答类样本的错误算到
      生成能力头上、拉低答案准确率还找不到原因。代价是拒答行为本身在这条指标里
      不被考核(它靠路由/召回指标兜底)。
    - 分母只用标准答案的事实数、不看回答多说了什么 ⇒ 这条指标**只测漏答、不测多答**,
      编造出标准答案之外的事实不会被扣分(见报告「发现的问题」)。
    """
    if not facts:
        return 1.0
    return len(facts & extract_facts(answer)) / len(facts)


def route_accuracy(records: list[dict]) -> float:
    """路由准确率:该检索的进了检索,不该检索的没进检索。

    只统计 expected_route 明确为 search/direct 的样本 —— None 表示「这条样本不校验
    路由」(见 evaluation/eval_set.py 的 validate_sample),算进分母会把两类样本混起来。

    判定写成 `(期望是 search) == (实际走了检索)` 这一个布尔等式,而不是两条 if:
    它同时覆盖「该检索却没检索」与「不该检索却检索了」两个方向的错误,少一处漏判。
    期望 direct 的合格面很宽:cache / faq / direct 都算「没走检索」——
    对「今天天气怎么样」这类问题,命中缓存反而是好事,不该扣分。

    没有可比样本时返回 **0.0**(不是 1.0、也不抛错):这是「没数据」的记号。
    看报告时要把它和「准确率真的是 0」区分开 —— 后者是链路全错,前者只是样本集没配。
    """
    comparable = [r for r in records if r.get("expected_route") in ("search", "direct")]
    if not comparable:
        return 0.0
    hits = 0
    for record in comparable:
        went_retrieval = str(record.get("actual_route")) in _RETRIEVAL_ROUTES
        if (record["expected_route"] == "search") == went_retrieval:
            hits += 1
    return hits / len(comparable)


def filter_field_accuracy(records: list[dict]) -> dict:
    """筛选字段准确率:逐字段比较,并统计"不该抽却抽了"的过抽取次数。

    口径上分两块,别混着看:
    - **准确率 = 命中 / 出现次数**,分母只数「期望里出现过的字段」。
      比如期望只写了 person,那就只看 person 抽得对不对,ticket_type 抽错不影响它;
    - **过抽取(false_positive)** 单独计数、**不进准确率分母**。期望没要求的字段抽出来了
      就记一笔。不进分母是刻意的(否则「期望字段越少、准确率越虚高」的样本构造会反向
      影响指标),但也意味着**只看 accuracy 会漏掉过抽取** —— 必须同时看 false_positive。

    用 `setdefault` 惰性建桶:某字段只在过抽取时出现(命中 0 次)也会在报告里留一条
    `{"hit": 0, "count": 0, "false_positive": n, "accuracy": 0.0}` —— 这恰恰是有用的信号
    (「链路开始抽一个我们没期望的字段了」),所以不能因为 count==0 就把这条删掉。

    `sorted(fields.items())` 只为输出顺序稳定(便于人工 diff 报告),不影响结果。

    ⚠ 比较是**严格相等**(`actual.get(name) == expected_value`),字符串的大小写/空白
    差异都算错;actual 里同字段若出现多次(多轮抽取),以 record 里最终那份为准。
    """
    fields: dict[str, dict[str, int]] = {}
    total_hit = total_count = 0
    false_positive = 0

    for record in records:
        expected = record.get("expected_filter") or {}
        actual = record.get("actual_filters") or {}

        for name, expected_value in expected.items():
            stats = fields.setdefault(name, {"hit": 0, "count": 0, "false_positive": 0})
            stats["count"] += 1
            total_count += 1
            if actual.get(name) == expected_value:
                stats["hit"] += 1
                total_hit += 1

        for name in actual:
            if name not in expected:
                stats = fields.setdefault(name, {"hit": 0, "count": 0, "false_positive": 0})
                stats["false_positive"] += 1
                false_positive += 1

    report = {
        name: {
            **stats,
            # 分母为 0(该字段只出现在过抽取里)时给 0.0,而不是 ZeroDivisionError
            "accuracy": stats["hit"] / stats["count"] if stats["count"] else 0.0,
        }
        for name, stats in sorted(fields.items())
    }
    return {
        "fields": report,
        "overall": {
            "hit": total_hit,
            "count": total_count,
            "false_positive": false_positive,
            "accuracy": total_hit / total_count if total_count else 0.0,
        },
    }


def recall_at_k(records: list[dict], k: int = 5) -> float:
    """多路召回阶段的 Recall@K(口径与优化篇 retrieval_metrics 相同)。

    **委托**给 `evaluation.retrieval_metrics.recall_at_k`,不在这里重写一遍:
    这是刻意的单一实现 —— 线下微调章与线上分阶段评估必须同口径,两份实现迟早漂移。

    口径细节(容易踩):
    - 用 `candidate_ids`(多路召回合并去重后的候选)而不是 `reranked_ids`,
      因为这里测的是召回段、不是重排段;
    - 对**全部 records** 求平均,包含 expected_ticket_ids 为空的拒答样本。
      对拒答样本,`_recall_at_k` 的规则是「一个都没召回到 = 1.0、召回了东西 = 0.0」,
      所以拒答题一旦被误召回就会直接拉低 Recall@K —— 这是有意的(拒答题不该召回),
      但看数字时要记得分母里混着这批样本;
    - 单条 record 缺 `candidate_ids` 时按空列表处理,不抛错。
    """
    if not records:
        return 0.0
    values = [
        _recall_at_k(
            list(record.get("candidate_ids") or []),
            list(record.get("expected_ticket_ids") or []),
            k,
        )
        for record in records
    ]
    return sum(values) / len(values)


def rerank_hit_at_k(records: list[dict], k: int = 5) -> float:
    """重排序阶段的 Hit@K:前 K 个精排结果里是否包含期望票据。

    与上面的 Recall@K 有三处**不同**的口径,必须对照着看才不会误读:
    - 用 `reranked_ids`(重排后保留的)而不是候选 —— 测的是精排有没有把正确答案留下;
    - 只统计 `expected_ticket_ids` 非空的样本(排除拒答/直答),**分母与 Recall@K 不同**;
    - 是 **Hit(0/1)** 不是比例:期望集有 2 张票、只留下 1 张也算命中。
      多票对比样本(build_multi)因此在这条指标上偏宽松,要严格程度得看 Recall@K。

    `[:k]` 截断有意义的前提是 reranked_ids 已按相关度降序(见 retrieval/rerank.py
    重排后的显式降序排序);若上游没排序,这里截的就是「接口返回的前 K 个」。
    """
    comparable = [r for r in records if r.get("expected_ticket_ids")]
    if not comparable:
        return 0.0
    hits = 0
    for record in comparable:
        expected = set(record["expected_ticket_ids"])
        if expected & set(list(record.get("reranked_ids") or [])[:k]):
            hits += 1
    return hits / len(comparable)


def answer_fact_accuracy(records: list[dict]) -> float:
    """答案准确性:标准答案的关键事实在最终回答里被覆盖的平均比例。

    只统计 ground_truth 非空的样本。注意拒答样本的 ground_truth 是
    「未找到相关票据信息。」(非空),所以**会**被统计,但它抽不出事实 ⇒
    fact_hit_rate 记 1.0,理由见该函数的注释。

    这是**条目级比例的宏平均**(每条样本权重相同),不是把所有事实汇总成一个大分数 ——
    这样一条含 8 个事实的长答案不会盖过 8 条各含 1 个事实的短答案。
    """
    comparable = [r for r in records if r.get("ground_truth")]
    if not comparable:
        return 0.0
    values = [fact_hit_rate(r.get("answer") or "", extract_facts(r["ground_truth"])) for r in comparable]
    return sum(values) / len(values)


def summarize(records: list[dict], k: int = 5) -> dict:
    """汇总各阶段指标(课案"按阶段计算指标"的落地产物)。

    输出一个扁平 dict,直接 json.dump 成 `data/stage_eval_report.json`。
    几个字段的注意点:
    - `cache_hit_count` 用 `if r.get("cache_hit")` 判真(exact / preset 都算命中,
      None/"" 不算)。**它是先该看的数字** —— 只要大于 0,这份报告的四项链路指标
      就可能被缓存短路污染了(见模块 docstring);
    - `filter_fallback_count` 记录「过滤零命中回退」的次数:大于 0 说明 filters 抽出的
      条件与库里字段对不上(当前已核实 route 中英文语言不一致),这时 filter 指标要打折看;
    - `mean/max_elapsed_s` 用 `if r.get("elapsed_s")` 过滤,跳过缺失或为 0 的记录
      (0 通常记为缓存秒回)⇒ 缓存命中的耗时不会把平均值拉低。这也是评估要关缓存
      的一个附带理由:开着缓存测出来的延迟不是链路延迟。
    """
    timings = [float(r.get("elapsed_s") or 0.0) for r in records if r.get("elapsed_s")]
    cache_hits = sum(1 for r in records if r.get("cache_hit"))
    return {
        "count": len(records),
        "route_accuracy": route_accuracy(records),
        "filter_field_accuracy": filter_field_accuracy(records),
        "recall_at_k": recall_at_k(records, k),
        "rerank_hit_at_k": rerank_hit_at_k(records, k),
        "answer_fact_accuracy": answer_fact_accuracy(records),
        "cache_hit_count": cache_hits,
        "mean_elapsed_s": sum(timings) / len(timings) if timings else 0.0,
        "max_elapsed_s": max(timings) if timings else 0.0,
        "filter_fallback_count": sum(1 for r in records if r.get("filter_fallback")),
    }


# 自检:一条「全对」的最小 record。钉三件事 —— 模块能独立跑(路径引导生效)、
# 三项主指标能到 1.0、标准答案「436.00 元」与回答「436.00 元」在归一化后判为同一事实。
# 注意这条 demo 没有 expected_ticket_ids 为空的样本,所以覆盖不到拒答分支。
if __name__ == "__main__":
    demo = [
        {
            "task_type": "retrieval",
            "expected_route": "search",
            "actual_route": "rag",
            "expected_filter": {"person": "张三"},
            "actual_filters": {"person": "张三"},
            "expected_ticket_ids": ["t1"],
            "candidate_ids": ["t1"],
            "reranked_ids": ["t1"],
            "ground_truth": "合计 436.00 元",
            "answer": "共 436.00 元",
        }
    ]
    summary = summarize(demo)
    assert summary["route_accuracy"] == 1.0, summary
    assert summary["recall_at_k"] == 1.0, summary
    assert summary["answer_fact_accuracy"] == 1.0, summary
    print(f"stage_metrics.py 自检通过: {summary}")

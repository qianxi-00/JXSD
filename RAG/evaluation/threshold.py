"""阈值标定:对照基础篇课案「评测集生成 / 阈值确定 / 阈值搜索」。

课案方法:
1. 正例 = 与 query 同义或近义的说法(课案用 SimBERT / 往返翻译生成);
2. 负例 = **难负例**(与 query 语义相近但实际不相关,课案用 bge-m3 + FAISS 的
   hn_mine 挖掘),比随机负例的监督信号强得多;
3. query / 正例 / 负例都向量化并归一化,点积得到相似度分数;
4. 对候选阈值逐个算 accuracy / precision / recall / F1,取 F1 最高的阈值。

本项目全部模型走外部 API,没有本地 SimBERT/FAISS,因此把第 1、2 步换成:
- 正例:调用 LLM 生成同一问题的不同问法(script/build_threshold_dataset.py);
- 难负例:用 embedding 接口给候选语料打分,再按课案同样的区间参数
  (range_min/range_max/max_score/absolute_margin)挑选(本模块 select_hard_negatives)。
  ⚠️ 与课案 hn_mine 的差别:课案从**语料库**里挖,这里从**其他 query**里挖
  (对 FAQ 层够用;票据侧的 query-文档负例由 script/calibrate_rerank.py 补上)。

标定结果用于 `REDIS_SIM_THRESHOLD`(FAQ/预设缓存相似度)与
`RERANK_RELEVANCE_P`(重排序相关度阈值),避免拍脑袋取值。

为什么阈值必须"标定"而不能照抄或凭感觉:
两个阈值都作用在**相似度/相关度分数**上,而分数的量纲完全由模型决定 ——
换 embedding 或 reranker 后同一个问题的分数会整体平移。项目上真实踩过两次:
FAQ 相似度阈值 0.85 是 Qwen3 向量空间的量纲,换 bge-m3 后正例相似度只有 0.98、
可用区间整体下移,0.85 直接漏召回;重排阈值 0.65 是 Qwen3-Reranker 的量纲,
换 bge-reranker-v2-m3 后命中票据只有 0.51~0.54,0.65 会把正确答案**整条丢掉**。
所以规则是:换栈必重标,标定结果必须落在当前模型空间的实测分数分布内。

本模块的三个可测性设计(改动时别破坏):
- 向量化通过 `Embed` 可注入函数传入,不在这里 import HTTP 客户端 ⇒ 单测能喂假向量;
- 只有 NumPy 依赖,没有 Redis/Milvus ⇒ 可以作为纯算法模块离线跑;
- 文件尾带自检,能 `python -X utf8 RAG/evaluation/threshold.py` 单独验(路径用正斜杠写;
  普通(非 raw)字符串里若出现「反斜杠 + e」这类非法转义序列,Python 会打 SyntaxWarning)。
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np

# 文本 → 向量的注入点。生产传 retrieval.embedding.embed_query,测试传查表 lambda。
# 用 Callable 别名而不是协议/抽象类:这里只需要"能调"这一个约束,越轻越好。
Embed = Callable[[str], list[float]]


def _normalize(vec: np.ndarray) -> np.ndarray:
    """把向量归一化成单位向量(点积即余弦相似度)。

    为什么必须归一化:本模块用 `np.dot` 当相似度,只有两侧都是单位向量时点积才等于
    余弦相似度;不归一化的话结果会随向量模长浮动(长文本/异常样本模长偏大),
    阈值就失去可比性 —— 标定出来的 0.79 换一批文本就不适用了。

    `max(norm, 1e-10)` 是**除零保护**:全零向量(embedding 接口返回空/异常时可能出现)
    的范数是 0,直接相除会得到 NaN 并静默污染后续所有比较(NaN 与阈值比大小恒为 False,
    表现为「整批全判成负例」,极难排查)。用 1e-10 兜底后,零向量会被放大成大数值,
    但至少结果是确定的有限值,不会变成 NaN。注意本函数**不校验维度**,
    维度不一致会在 np.dot 处抛 ValueError(这是期望的行为:坏向量应当立刻失败)。
    """
    norm = float(np.linalg.norm(vec))
    return vec / max(norm, 1e-10)


def build_pos_neg_scores(items: list[dict], embed: Embed) -> list[dict]:
    """把 (query, 正例列表, 负例列表) 转成相似度分数。

    Args:
        items: 形如 [{"query": str, "pos": [str], "neg": [str]}]。
        embed: 文本 -> 向量的可注入函数(生产用 retrieval.embedding.embed_query)。

    Returns:
        [{"query": str, "pos": [float], "neg": [float]}]。

    实现要点:
    - **query 向量只算一次**(`query_vec` 在循环外),正例/负例逐条算并做点积 ——
      一次标定有几十个 query × 每条 5 正例 + 5 负例,重复算 query 向量会让
      embedding 调用量翻倍(每次都是外部 API 计费调用);
    - 统一 `dtype=np.float32`:外部 API 返回的是 float64 的 list,不显式降精度的话
      与后续 float32 运算混用会有微小的表示差异,同一份数据两次标定结果可能不同;
    - pos/neg 用 `item.get(..., [])` 容错缺字段,不要改成 `item["pos"]`
      (构造侧允许只给正例不给负例)。
    """
    scores: list[dict] = []
    for item in items:
        query_vec = _normalize(np.asarray(embed(item["query"]), dtype=np.float32))
        pos = [
            float(np.dot(query_vec, _normalize(np.asarray(embed(text), dtype=np.float32))))
            for text in item.get("pos", [])
        ]
        neg = [
            float(np.dot(query_vec, _normalize(np.asarray(embed(text), dtype=np.float32))))
            for text in item.get("neg", [])
        ]
        scores.append({"query": item["query"], "pos": pos, "neg": neg})
    return scores


def calculate_metrics(threshold: float, pos_neg_scores: list[dict]) -> tuple[float, float, float, float]:
    """按给定阈值计算 (accuracy, precision, recall, f1)。

    分数 >= 阈值算命中;空数据集返回全 0(课案原实现会 ZeroDivisionError)。

    判断题的标注口径(决定了指标能不能读):
    - **正例判负 = 漏召回**(回答里没有该有的东西,用户直接感知为"答不出来");
    - **负例判正 = 误召回**(答成了别的票/别的问法,本项目真实踩过 —— 只差一个字的人名
      相似度 0.9008 就会串答案);
    - 两者的代价不对称,但 F1 视它们同等重要。所以选阈值时 F1 只是**主判据**,
      还要结合自己场景的偏好(宁可保守就往上沿取,见 script/calibrate_rerank.py)。

    计数是**把每条 query 拆开累加**再算全局指标,而不是先算各 query 的指标再平均:
    这是课案的口径。差别在于样本多的 query 权重更大 —— 对「每条 query 的正负例数
    都一样」的标定集(本项目就是:每个 query 固定 5 正例 + N 负例)两种算法等价。

    除零全部落成 0.0:precision 分母为 0 表示「阈值高到一个正例都没判出来」,
    recall 分母为 0 表示标定集里根本没有正例 —— 都记 0 让 f1 也跟着 0,
    这样搜索循环不会因为 NaN 而选出莫名其妙的阈值。
    """
    true_positives = false_positives = false_negatives = true_negatives = 0

    for item in pos_neg_scores:
        pos_predictions = [score >= threshold for score in item["pos"]]
        neg_predictions = [score >= threshold for score in item["neg"]]

        tp_in_query = sum(pos_predictions)
        fn_in_query = len(pos_predictions) - tp_in_query
        fp_in_query = sum(neg_predictions)
        tn_in_query = len(neg_predictions) - fp_in_query

        true_positives += tp_in_query
        false_negatives += fn_in_query
        false_positives += fp_in_query
        true_negatives += tn_in_query

    total = true_positives + true_negatives + false_positives + false_negatives
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0

    accuracy = (true_positives + true_negatives) / total
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return accuracy, precision, recall, f1


def f1_plateau(all_metrics: list[dict], tolerance: float = 1e-12) -> list[dict]:
    """返回与最高 F1 并列的全部阈值指标(保持输入顺序)。

    标定报告要拿它回答一个问题:"当前 .env 的值是否已经在平台内?"
    平台内换值不会提升 F1,却会多引入一次变量扰动。

    Args:
        all_metrics: find_optimal_threshold 的第二项返回值。
        tolerance: 判定并列的容差。

    Returns:
        并列最高 F1 的指标子列表;输入为空时返回空列表。

    `tolerance` 默认 **1e-12** —— 这是「近似精确相等」而不是常见的那种"容忍一点误差"。
    因为 F1 由计数的整数比值算出(如 7/8),同一平台内的这些值**在二进制上就是同一个数**,
    用严格的 >= best 也能通过。留 1e-12 只是为了挡掉极端情况下的 1 ULP 偏差。
    ⚠ 别把这个容差调大(比如 1e-2)去"宽容一点":那会把临近悬崖的阈值也吸进平台,
    让中点整体偏移,而这正是本函数要避免的事。

    平台排序保持输入顺序 = 阈值升序(all_metrics 是按 thresholds 递增 append 的),
    所以 `plateau[0]` 是平台**左端点**(贴悬崖)、`plateau[-1]` 是**右端点**、
    `plateau[len//2]` 是**中点** —— find_optimal_threshold 取的就是中点。
    偶数个点时 `len//2` 偏右(如 4 个点取第 3 个),这是有意的取向:略偏保守。
    """
    if not all_metrics:
        return []
    best_f1 = max(metric["f1"] for metric in all_metrics)
    return [metric for metric in all_metrics if metric["f1"] >= best_f1 - tolerance]


def find_optimal_threshold(
    pos_neg_scores: list[dict],
    threshold_range: tuple[float, float] = (0.0, 1.0),
    step: float = 0.01,
) -> tuple[float, dict, list[dict]]:
    """在给定范围内按步长搜索 F1 最高的阈值。

    F1 常常在一个区间里并列(平台)。此时取**平台中点**,而不是第一个并列点:
    返回值会被写进 .env 当生产阈值,平台左端点紧贴"再低一点就掉 precision"的悬崖,
    中点对噪声与数据漂移更鲁棒(数据量小时尤其重要)。

    Returns:
        (最佳阈值, 最佳指标字典, 全部阈值的指标列表)。
        即使所有阈值 F1 都是 0 也保证返回一个指标字典(课案原实现在这种情况下会返回 None)。

    ⚠ 搜索区间的端点行为:`np.arange(start, stop, step)` **不含 stop**,所以默认
    range=(0.0, 1.0) 实际扫的是 0.00~0.99 共 100 个阈值,1.0 不在候选里
    (余弦相似度 1.0 意味着完全相同,本来也不该做阈值)。想覆盖上界必须写
    (0.0, 1.01) 这类稍大的 stop。自检里 `len(all_metrics) == 20`(step=0.05)
    就是在钉这个半开区间语义。

    最终取值来自 `plateau[len(plateau) // 2]` —— 平台的**中点**。历史例子:
    F1 平台 0.77~0.81(5 个点)⇒ 取下标 2 = **0.79**,与 .env 当前值一致。
    平台点数为偶数时 `len//2` 偏右,即略偏保守(阈值高一点 = 少误召回)。

    区间为空(start >= stop)时退化返回下界的全 0 指标:这是为了**保证返回结构不变**
    (永远三元组、永远有指标字典),让调用方不必写 None 判断 ——
    代价是"参数写错"和"真的搜出 0 分"在返回值上长得一样,
    所以脚本侧读结果时要看 `all_metrics` 是否为空。
    """
    thresholds = np.arange(threshold_range[0], threshold_range[1], step)
    all_metrics: list[dict] = []

    for threshold in thresholds:
        accuracy, precision, recall, f1 = calculate_metrics(float(threshold), pos_neg_scores)
        all_metrics.append(
            {
                "threshold": float(threshold),
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    if not all_metrics:  # 区间为空(start >= stop)时退化为下界,保证返回结构不变
        fallback = {"threshold": float(threshold_range[0]), "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        return fallback["threshold"], fallback, all_metrics

    plateau = f1_plateau(all_metrics)
    best_metrics = plateau[len(plateau) // 2]

    return best_metrics["threshold"], best_metrics, all_metrics


def select_hard_negatives(
    corpus: list[tuple[str, float]],
    positive: str,
    positive_score: float | None = None,
    num_negatives: int = 5,
    range_min: int = 0,
    range_max: int = 20,
    max_score: float = 0.8,
    absolute_margin: float = 0.1,
) -> list[str]:
    """从已打分的候选语料中挑难负例(课案 hn_mine 区间参数的 API 适配版)。

    Args:
        corpus: [(候选文本, 与 query 的相似度)],按相似度降序与否都可。
        positive: 正例文本(会被排除)。
        positive_score: 正例相似度;给了才做 absolute_margin 检查。
        num_negatives: 最多取几条。
        range_min: 跳过相似度最高的前 range_min 个候选(排除正例后按 1 起排名)。
        range_max: 只看前 range_max 个候选。
        max_score: 负例相似度上限(太像正例的不要)。
        absolute_margin: 负例至少要低于正例这么多分。

    Returns:
        选中的负例文本列表,按相似度降序。

    为什么"难负例"必须难(而不是随便拿几条不相关文本当负例):
    随机负例与正例的分数差得远,任何阈值都能把它们分开 ⇒ F1 曲线是一片高原,
    标出来的阈值对真实边界毫无区分力。这正是本项目 FAQ 阈值标定的教训 ——
    负例只取自「其他预设问题」、不含「同问法换人名」这类真近邻,于是 F1 到 1.0,
    而线上一个只差一个字的人名(相似度 0.9008)直接串出别人的答案。
    **标定集的负例必须包含最像的那种错答案**,否则标定出来的平台是假象。

    实现顺序与三个过滤条件(全部满足才入选,`continue` 而非 `break`):
    1. `rank <= range_min or rank > range_max` → 跳过 —— 用 continue 而不是 break,
       因为后续 rank 只会更大,`rank > range_max` 其实可以直接 break;
       这里是照搬课案的区间语义,功能等价,只是多扫几轮(候选只有几十条,无所谓);
    2. `score > max_score` → 跳过"太像正例的"候选:它们可能是漏标的正例,
       拿来当负例会**人为压低**最优阈值;
    3. `score > positive_score - absolute_margin` → 跳过离正例太近的:
       这条只有传了 positive_score 才生效(`is not None` 判断是刻意的 ——
       positive_score 可能是 **0.0** 这种合法值,写成 `if positive_score` 会漏判)。

    先按分数降序 `sorted` 再排名:入参不保证有序,而 range_min/range_max 的语义
    是"相似度最高的前 N 个",不排序的话区间过滤就没有意义。
    `text != positive` 在生成器里就排除正例,避免正例占据一个 rank 位置
    (否则 range_min 的含义会随"正例排第几"漂移)。
    """
    ranked = sorted(
        ((text, score) for text, score in corpus if text != positive),
        key=lambda pair: pair[1],
        reverse=True,
    )

    picked: list[str] = []
    for rank, (text, score) in enumerate(ranked, start=1):
        if rank <= range_min or rank > range_max:
            continue
        if score > max_score:
            continue
        if positive_score is not None and score > positive_score - absolute_margin:
            continue
        picked.append(text)
        if len(picked) >= num_negatives:
            break

    return picked


# 解析 LLM 输出用的两条正则(详见 parse_paraphrases)。
# 前缀:`- ` / `* ` / `• ` / `1. ` / `2、` / `3) ` / `问题:` 这些 LLM 爱加的装饰;
# 注意 `\d+\s*[.、)）]` 要求数字后必须跟分隔符,所以「2025年的票」这种以数字开头的
# 问法不会被误剥掉年份(分隔符是白名单而不是"任何非数字字符")。
_PARAPHRASE_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+\s*[.、)）]|问题\s*[:：])\s*")
# 名字叫 SUFFIX 但实际匹配的是**行首**的另一种前缀(「问法:」「子问题:」),
# 命名与行为不符(见报告);分成两条是因为要先后做两次剥离:
# 「1. 问法: 张三的机票多少钱」需要先剥编号再剥「问法:」。
_PARAPHRASE_SUFFIX_RE = re.compile(r"^\s*(?:问法|子问题)\s*[:：]\s*")


def parse_paraphrases(raw: str, limit: int = 5) -> list[str]:
    """解析 LLM 生成的同义问法列表:去掉编号/项目符号/前缀,去重去空,最多取 limit 条。

    这是**容错解析器**,因为输入是 LLM 的自由文本:同一份提示词,模型可能返回
    「1. xxx」「- xxx」「问法: xxx」、也可能混着空行和重复行。这里全部剥成纯问法,
    让下游可以按"就是一个问题"直接用。

    两个细节:
    - 去重用的是**剥离后的文本**(`text in seen`),所以「1. 问法A」与「- 问法A」
      会被正确判成同一条,而不会产出两条一样的正例(重复正例会让该 query 在
      precision/recall 里占更大权重,悄悄改变标定结果);
    - **去重区分大小写与空白内差异**(纯字符串相等),`问法A` 与 `问法 a` 算两条 ——
      对中文问法基本无影响,知道即可。

    `limit` 达到即 break(不是 continue):默认 5 条与课案"每 query 5 条正例"对应,
    多余的丢弃而不是塞进来,保持每条 query 的样本数一致 ⇒ 全局计数口径成立。
    """
    seen: set[str] = set()
    picked: list[str] = []
    for line in (raw or "").splitlines():
        text = _PARAPHRASE_PREFIX_RE.sub("", line).strip()
        text = _PARAPHRASE_SUFFIX_RE.sub("", text).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        picked.append(text)
        if len(picked) >= limit:
            break
    return picked


def build_threshold_items(
    queries: list[str],
    paraphrases: dict[str, list[str]],
    embed: Embed,
    num_negatives: int = 5,
    max_score: float = 0.8,
    absolute_margin: float = 0.1,
) -> list[dict]:
    """把 query、正例、难负例组装成阈值标定输入。

    难负例从**其他 query**里按 embedding 相似度挑:语义相近但不是同一个问题,
    正是课案要的"难负例"(比随机负例的监督信号强)。参数沿用课案区间参数。

    ⚠ 这是模块 docstring 里已注明的**与课案的已知差异**:课案从**语料库**挖负例,
    这里从**其他 query**挖。对 FAQ 层够用(预设问题之间确实互为最难负例),
    但它有个结构性盲点 —— **同问法换人名**这类"极近邻"不在候选里(那属于 query-文档
    负例),所以本函数标出的平台**不能**证明阈值对近邻安全。票据侧的 query-文档负例
    由 `script/calibrate_rerank.py` 单独补。

    实现上:
    - `vectors` 先把所有 query 向量算好并缓存:生成 corpus 时两两比较,
      不缓存的话就是 O(n²) 次 embedding 调用(n=60 时 3600 次付费调用);
    - `positive_score=1.0` 是 query 与自身的相似度(归一化后点积为 1)。
      ⚠ 这个 1.0 会被传进 select_hard_negatives 参与 `absolute_margin` 判断,
      等价于 **"负例相似度必须 ≤ 1 - 0.1 = 0.9 才入选"** —— 这是一道相当紧的硬门槛,
      在本项目实测中会让大部分候选负例被过滤掉(见报告)。参数是课案原样,
      改动前要重新标定一遍确认影响。
    - `paraphrases.get(query, [])` 容错缺失:LLM 生成失败时该 query 只有负例,
      仍然可用(它的 tp 恒 0、会压低 recall,看报告时能发现)。
    """
    vectors = {query: _normalize(np.asarray(embed(query), dtype=np.float32)) for query in queries}

    items: list[dict] = []
    for query in queries:
        corpus = [
            (other, float(np.dot(vectors[query], vectors[other])))
            for other in queries
            if other != query
        ]
        negatives = select_hard_negatives(
            corpus,
            positive=query,
            positive_score=1.0,
            num_negatives=num_negatives,
            max_score=max_score,
            absolute_margin=absolute_margin,
        )
        items.append({"query": query, "pos": list(paraphrases.get(query, [])), "neg": negatives})
    return items


# 自检覆盖四个函数各自的边界:分离良好的数据能搜到 F1=1、np.arange 的半开区间
# (step=0.05 → 20 个候选)、空数据集不炸、难负例的三个过滤条件、解析器的编号剥离、
# 以及 build_threshold_items 在"只有 2 个 query"时的取值(甲乙互为候选,
# 两个都是人名所以既不超 max_score 也不超 0.9 门槛,但甲1 属于甲的正例、不在 queries 里 ⇒ 负例为空)。
if __name__ == "__main__":
    # 自检:分离良好的数据集上应当搜到 F1=1 的阈值
    demo = [{"query": "q1", "pos": [0.9, 0.85], "neg": [0.2, 0.3]}]
    best, metrics, all_metrics = find_optimal_threshold(demo, step=0.05)
    assert metrics["f1"] == 1.0, metrics
    assert len(all_metrics) == 20
    assert calculate_metrics(0.5, []) == (0.0, 0.0, 0.0, 0.0)
    picked = select_hard_negatives(
        [("正例", 0.95), ("太像", 0.9), ("较像", 0.7), ("无关", 0.2)],
        positive="正例",
        positive_score=0.95,
        num_negatives=5,
    )
    assert picked == ["较像", "无关"], picked
    assert parse_paraphrases("1. 问法A\n2、问法A\n- 问法B") == ["问法A", "问法B"]
    items = build_threshold_items(
        queries=["甲", "乙"],
        paraphrases={"甲": ["甲1"]},
        embed=lambda text: {"甲": [1.0, 0.0], "乙": [0.9, 0.4], "甲1": [1.0, 0.0]}[text],
    )
    assert items[0]["pos"] == ["甲1"] and items[0]["neg"] == []
    print(f"threshold.py 自检通过(最佳阈值={best:.2f}, F1={metrics['f1']:.2f})")

"""排序质量指标:优化篇课案 `retrieval_metrics` 模块的对齐实现。

课案「RAG 评估」用它算 Recall@K,「微调 · 评估指标」用同一套口径算
MRR / MAP / nDCG@K —— 离线微调与线上评估必须同一口径,所以两处只有这一份实现。

指标含义(课案原文):
- Recall@K:前 K 命中数 / 相关文档总数 —— 只回答"找到没有";
- MRR:首个命中排名的倒数,对 query 取平均 —— 只看第一个命中;
- MAP:每个命中位置取 Precision@rank 求平均得 AP,再对 query 取平均 —— 惩罚"找齐但靠后";
- nDCG@K:按 1/log2(rank+1) 折损累计增益,除以理想增益归一化 —— 对排名最敏感。

约定:同一 doc_id 在结果里重复出现只计一次;relevant 为空时按拒答样本处理。

为什么这四件指标要挤在一个文件里、且被两处复用:
课案的「RAG 评估」用它算 Recall@K,「微调 · 评估指标」用同一套口径算 MRR / MAP / nDCG@K。
**离线微调与线上评估必须同一口径** —— 若各写一份,同一份检索结果在两处会得出不同
分数,「微调完到底有没有变好」就永远说不清。所以只有这一份实现,别在别处再抄一遍。

本模块是**纯函数集合**:不碰网络/数据库/全局状态,输入 list[str] + k,输出 float。
所有函数对 `relevant_ids` 为空、retrieved 为空、k 超长、doc_id 重复这几种边界都有
明确取值(见各函数注释)—— 这些边界在评估集里**真的会出现**(拒答样本 relevant 为空),
不能靠「调用方保证不会传」来兜。
"""

from __future__ import annotations

from math import log2


def validate_k(k: int) -> None:
    """校验截断位置 K 合法(K 必须为正整数)。

    刻意用 `isinstance(k, int)` 而不是 `k > 0` 一件搞定:布尔是 int 的子类但这里无所谓,
    真正要防的是**浮点**(`k=5.0` 在切片里会 TypeError,但错误信息离调用点很远)
    与字符串。评估脚本的 k 常常来自 argparse(默认 int)或 json 配置(**可能是 str**),
    在这里一次性拦掉比让它在切片处炸掉好定位。

    抛 ValueError 而不是返回 bool:调用方(四个指标函数)都希望「k 不合法 = 立刻失败」,
    写 `if not valid: return 0.0` 反而会把配置错误变成「指标恒为 0」的静默故障。
    """
    if not isinstance(k, int) or k <= 0:
        raise ValueError(f"k 必须是正整数,收到 {k!r}")


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Recall@K:前 K 命中 relevant 的比例。

    relevant 为空(拒答类样本)时无召回目标:未检索到任何票据记 1.0,否则 0.0。

    ⚠ 后一条规则是**刻意约定**而非数学必然:分母为 0 时本来无定义,这里选择
    「拒答题不召回 = 满分」,因为调参时真正想惩罚的是「拒答题却召回了东西」。
    副作用也很真实:只要评估集里有拒答样本,Recall@K 的分数就**同时包含了对拒答
    行为的考核**,与不含拒答样本的老报告不可直接比较。

    分母是 `len(relevant_ids)` 而不是 min(..., k):所以 k 小于相关文档数时
    这条指标**达不到 1.0**(把 Recall@K 和 Hit@K 混着读很容易误判为「召回坏了」)。
    """
    validate_k(k)
    if not relevant_ids:
        return 1.0 if not retrieved_ids[:k] else 0.0
    hits = len(set(retrieved_ids[:k]) & set(relevant_ids))
    return hits / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """MRR@K:前 K 个结果中首个相关文档排名的倒数,未命中为 0.0。

    实现用 `next(generator, 0.0)` 一次遍历取首个命中 —— 不建额外列表、找到即停。
    只对**单条 query** 算,喂一批 query 时对结果取平均才是常说的 MRR
    (本仓由 `mean()` 或上层汇总完成)。

    ⚠ MRR 只看**第一个**命中:回答了 5 张票、正确的那张排第 5,MRR 仍是 0.2,
    即使 Recall@K(k>=5) 是 1.0。这正是它要与 Recall 配套看的原因。
    """
    validate_k(k)
    relevant = set(relevant_ids)
    return next(
        (1.0 / rank for rank, doc_id in enumerate(retrieved_ids[:k], 1) if doc_id in relevant),
        0.0,
    )


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """nDCG@K:按排名折损的归一化累计增益,同一 doc_id 只计一次。

    实现要点:相关性只有 0/1 两档,所以每个命中位置的增益固定为 1,折损用
    `1/log2(rank + 1)`(rank 从 1 开始 ⇒ 第 1 位折损 1.0、第 2 位 0.6309…)。
    `rank + 1` 里的 +1 是 DCG 标准定义要求的,少了它第一个位置的折损会变成 log2(1)=0 除零。

    `ideal` 是**理想排序**(所有相关文档都排在最前)的 DCG,用 `min(k, len(relevant))`
    截断 ⇒ 相关文档数超过 K 时满分仍为 1.0,不会出现 nDCG > 1。

    `seen` 那行在命中判断**之外**(if 外面也有 `seen.add`):重复出现的文档只在第一次
    计分,后续出现既不加分也不再增加 seen。这是「同一 doc_id 只计一次」约定的落地处,
    也是 nDCG 对「召回重复刷榜」免疫的原因。
    """
    validate_k(k)
    relevant = set(relevant_ids)
    seen: set[str] = set()
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], 1):
        if doc_id in relevant and doc_id not in seen:
            dcg += 1.0 / log2(rank + 1)
        seen.add(doc_id)
    ideal = sum(1.0 / log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return dcg / ideal if ideal else 0.0


def average_precision(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """AP@K:在每个命中位置取 Precision@rank 求平均,重复 doc_id 只计一次。

    分母为 min(len(relevant), k),即前 k 个位置最多能命中的相关文档数,
    保证 AP@K 满分仍为 1.0;对多条 query 的 AP 取平均即为 MAP。

    `hits / rank` 里的 hits 是**累计命中数**(不是 1):这正是 Precision@rank 的定义 ——
    第 3 位命中、前面只中过 1 个,就贡献 2/3。用累计值而不是常数 1,才能区分
    「命中都挤在前面」和「命中散落在后面」,而后者正是 MAP 要惩罚的。

    relevant 为空时返回 **0.0**(与 recall_at_k 的 1.0 相反):AP 是精度类指标,
    没有相关文档就没有「精度」可言,给 1.0 会让拒答样本白送满分。两处取值的差异
    是刻意的,改任一处前先确认上层汇总是不是依赖这个区别。
    """
    validate_k(k)
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    seen: set[str] = set()
    hits = 0
    precision_sum = 0.0
    for rank, doc_id in enumerate(retrieved_ids[:k], 1):
        if doc_id in relevant and doc_id not in seen:
            hits += 1
            precision_sum += hits / rank
        seen.add(doc_id)
    return precision_sum / min(len(relevant), k)


def mean(values: list[float]) -> float:
    """一组 query 指标的均值(空列表记 0.0),用于把条目级指标汇总成批次指标。

    空列表返回 0.0 而不是抛错:上游 `--limit 0` 或数据集为空时,报「0.0」比抛
    ZeroDivisionError 更利于脚本把整轮评估跑完(评估脚本要写报告文件,
    半路炸掉会留下半份产物)。代价是「没数据」和「真的是 0 分」在报告里长得一样。
    """
    return sum(values) / len(values) if values else 0.0


# 自检:直接采用课案原文给的例子(t3 第 1 位、d1 第 2 位、d5 第 3 位、d2 第 4 位),
# 四条指标各断言一个数,另加一条「重复 doc_id 不重复计分」的断言。
# 这些数字是课案里算好的,改动实现后若这里红了,说明口径偏离了课案。
if __name__ == "__main__":
    # 自检:课案原文的例子
    # retrieved 故意是「不完美」的排序(正确答案 d1/d2 分别在第 2、4 位),这样四条指标
    # 才会落在 1.0 之外;若换成完美排序,四个断言都会退化成 1.0、钉不住任何精度差异
    retrieved, relevant = ["d3", "d1", "d5", "d2"], ["d1", "d2"]
    assert recall_at_k(retrieved, relevant, 4) == 1.0
    assert mrr(retrieved, relevant, 4) == 0.5
    assert abs(average_precision(retrieved, relevant, 4) - 0.5) < 1e-9
    # 浮点比较用「差值 < 容差」而不是 ==:1/log2 是浮点运算,精确相等不可靠
    assert abs(ndcg_at_k(retrieved, relevant, 4) - 0.651) < 0.001
    # 重复 doc_id 只计一次:["A","A"] 对 ["A"] 的理想增益就是第 1 位那次,故为 1.0
    assert ndcg_at_k(["A", "A"], ["A"], 2) == 1.0
    print("retrieval_metrics.py 自检通过")

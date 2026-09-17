"""评估集契约:对照基础篇课案「系统评估 · 第一步:准备人工校验的测试集」。

课案的分层要求(评估集不能只有一种问法):
- faq:FAQ 快速返回集 —— 常见标准问题是否被快速返回层命中;
- retrieval:检索召回集 —— 关键词/向量/混合召回能否找回目标票据;
- structured_aggregation:结构化统计集 —— 总金额、数量、超阈值等是否算对;
- generation:生成答案集 —— 最终回答是否准确、忠实、带来源。

每条样本的字段(课案原文 + 分层补充):
question / task_type / expected_route / expected_filter /
expected_ticket_ids / ground_truth / need_citation

为什么要有这个「契约」模块(而不是各脚本自己读写 jsonl):
评估是一个**多方共享的接口** —— 样本由 `script/build_eval_set.py` 生成、被
`script/run_stage_eval.py` / `script/calibrate_rerank.py` / `script/langfuse_evaluation.py`
分别消费、还要被 `script/upload_langfuse_dataset.py` 推到 Langfuse。
只要有一处写出半成品样本(比如 task_type 打字打错、expected_ticket_ids 写成字符串),
下游的指标会**静默算错**而不是报错(举例:`expected_ticket_ids="t1"` 在按 id 求交集的
指标里会退化成「按字符逐个比对」,Recall 永远 0,但没有任何异常可查)。
所以这里把「读写 + 校验」收敛到一处:凡是进出评估集的样本都必须过 `validate_sample`。

注意方向性:本模块**只做形状校验,不做语义判断** —— 它不知道 expected_ticket_ids
是不是真的在库里存在,也不判断 ground_truth 对不对。语义正确性由构造侧
(`evaluation/eval_builders.py`,标准答案全部由真实字段确定性拼装)保证。
"""

from __future__ import annotations

import json
from pathlib import Path

# 四类任务类型(课案「系统评估」的分层)。改动这个元组 = 改动整个评估体系的分类口径,
# 因为 stage_metrics 的按 task_type 分组统计、eval_builders 的样本构造都会跟着变。
TASK_TYPES = ("faq", "retrieval", "structured_aggregation", "generation")
# 路由只有两条路:search=要检索, direct=直接让 LLM 答(不检索)
ROUTES = ("search", "direct")


def validate_sample(sample: dict) -> dict:
    """校验并规范化一条评估样本,字段非法时抛 ValueError。

    为什么是「校验 + 规范化」而不是只校验:
    返回的是**重建出来的新字典**,字段集合与顺序固定、缺失字段补默认值
    (expected_filter→{}、expected_ticket_ids→[]、ground_truth→""、need_citation→False)。
    这样下游拿到的样本字段永远齐全,不用到处写 `sample.get(...) or 默认值`;
    同时输出顺序稳定 ⇒ 同样的输入永远写出同样的 jsonl 行,便于 diff 比对。

    抛错而不返回 None 是刻意的:评估集坏了必须**当场炸**,不能降级成「跳过这条」——
    否则样本数悄悄变少,指标照样算得出来,但已经不可信了。
    """
    if not isinstance(sample, dict):
        raise ValueError(f"样本必须是字典,收到 {type(sample).__name__}")

    # question 用 `or ""` + strip:同时兼容「字段缺失」与「字段是 None」两种写法,
    # 且把纯空白串也判为非法(空白问题会让检索/路由行为不可预期,不如直接拦下)
    question = str(sample.get("question") or "").strip()
    if not question:
        raise ValueError("样本缺少 question(或为空)")

    # 这里用 `not in` 而非「缺失就取默认」:task_type 是评估的分组维度,
    # 缺失会导致该样本在按类型汇总时凭空多出一组,所以必须显式给值
    task_type = sample.get("task_type")
    if task_type not in TASK_TYPES:
        raise ValueError(f"task_type 必须是 {TASK_TYPES} 之一,收到 {task_type!r}")

    # expected_route 允许 None:不是每条样本都需要约束路由
    # (比如纯检索型样本只关心召回,路由由别处兜底判定),None 表示「不校验这一项」
    expected_route = sample.get("expected_route")
    if expected_route is not None and expected_route not in ROUTES:
        raise ValueError(f"expected_route 必须是 {ROUTES} 之一或 None,收到 {expected_route!r}")

    # 注意 default 用的是 `or {}` 而不是 `{}`:jsonl 里写成 null 时会走成空 dict,
    # 避免下游 `filter is not None` 这类判空写错
    expected_filter = sample.get("expected_filter") or {}
    if not isinstance(expected_filter, dict):
        raise ValueError("expected_filter 必须是字典")

    # 逐元素判 str:上面注释里说的「字符串被当成序列」的坑就在这里挡住
    expected_ticket_ids = sample.get("expected_ticket_ids") or []
    if not isinstance(expected_ticket_ids, list) or any(
        not isinstance(item, str) for item in expected_ticket_ids
    ):
        raise ValueError("expected_ticket_ids 必须是字符串列表")

    return {
        "question": question,
        "task_type": task_type,
        "expected_route": expected_route,
        "expected_filter": expected_filter,
        "expected_ticket_ids": list(expected_ticket_ids),
        "ground_truth": str(sample.get("ground_truth") or ""),
        "need_citation": bool(sample.get("need_citation", False)),
    }


def load_jsonl(path: str | Path) -> list[dict]:
    """读取 jsonl 评估集,逐行校验,报错时带行号。

    带行号是刚需:评估集是人工可编辑的文本文件,报「样本不合法」而不给行号,
    改的人要在几百行里找。这里把 `validate_sample` 的 ValueError 重新包装成
    `评估集第 N 行不合法: 原错误`,并用 `from exc` 保留原始异常链便于追栈。

    空行被跳过(容忍手写时留空行),但**非空行的解析失败不会被跳过** —— 见上面
    「坏了要当场炸」的说明。
    """
    path = Path(path)
    samples: list[dict] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            samples.append(validate_sample(json.loads(line)))
        except ValueError as exc:
            raise ValueError(f"评估集第 {lineno} 行不合法: {exc}") from exc
    return samples


def save_jsonl(samples: list[dict], path: str | Path) -> int:
    """写出 jsonl 评估集(逐行 JSON,UTF-8),返回写出条数。

    写出前**再过一遍校验**(而不是直接 dumps 传进来的 dict),这样「手拼的样本」
    想落盘也会被同一套规则挡住,不会出现「读得进、写得出,但下游不认」的脏数据。

    两个刻意的取舍:
    - `ensure_ascii=False`:中文原样落盘,评估集是给人看的,转义成 \\uXXXX 没法 review;
    - 末尾补换行(且空集时不补):POSIX 惯例,避免追加写时两行粘在一起。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(validate_sample(s), ensure_ascii=False) for s in samples]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)

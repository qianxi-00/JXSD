"""重排序阈值标定:用带标注的评估集标定 `RERANK_RELEVANCE_P`。

和 `search_threshold.py`(FAQ 相似度阈值)是同一套方法,只是分数来源不同:

    FAQ 阈值   —— 正例=同义问法、难负例=其他预设问题,分数=embedding 余弦
    重排阈值   —— 正例=该问题对应的真实票据、负例=同一次召回里的其它票据,分数=reranker 相关度

为什么必须单独标:**换 reranker 就得重标**。本项目 2026-09-17 从
`Qwen/Qwen3-Reranker-4B` 换成 `BAAI/bge-reranker-v2-m3`,同一条命中的票据分数
从 ~0.94 掉到 ~0.51~0.54,沿用旧阈值 0.65 会把**正确答案整条丢掉**
(实测 8 条样本里 2 条因此退化成"没检索到依据"的保守回复)。

用法:
    uv run python RAG/script/calibrate_rerank.py
    uv run python RAG/script/calibrate_rerank.py --eval-set RAG/data/eval_set.jsonl --top-n 10

★ 与 search_threshold.py 的**关键差异:并列时取"平台上沿"而不是中点**。
    FAQ 阈值取中点,是因为平台两端**都**观测到悬崖(再低就掉 precision、再高就掉 recall);
    重排阈值取上沿,是因为实测**负例分数全部贴在 0 附近(最大 0.0074)**、
    平台下端根本不是悬崖(它只是搜索下界 args.start=0.05)。
    既然下端没有观测到"再低会变好"的证据,就该往保守侧取 ——
    取上沿 = "仍然不掉 F1 的最高阈值",对没见过的负例留最大余量。
    代价的不对称也是理由:**放错票据比漏票据更伤** ——
    漏了会走保守回复(用户知道没查到),放错则让模型拿无关上下文作答(可能编内容)。
    ⇒ 通用判据:并列时先看**哪一端有观测到的悬崖**,没有悬崖的一端不该用作取点依据。

⚠ 本脚本是**只读标定**:只算分数、写 json、打印建议,**不会改 .env**。
   改配置由人在确认后执行(本项目约定 agent 不动 .env)。
"""

# --- 路径引导：确保能导入 Python_Base 根目录的 config 与 RAG 内部包 ---
import sys as _sys
from pathlib import Path as _Path

_BASE = _Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
_sys.path.insert(0, str(_BASE))          # Python_Base 根（config.py）
_sys.path.insert(0, str(_BASE / "RAG"))  # RAG 根（core/llm/pipeline 等包）

import argparse
import json
from pathlib import Path

from config import settings
from core.logger import logger
from evaluation.eval_set import load_jsonl
from evaluation.threshold import calculate_metrics, f1_plateau, find_optimal_threshold
from pipeline.filters import build_milvus_filter
from retrieval.rerank import rerank_scores
from retrieval.vector_retrieval import vector_search

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_SET = PROJECT_ROOT / "data" / "eval_set.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "data" / "rerank_threshold_results.json"


def collect_scores(samples: list[dict], top_n: int) -> tuple[list[dict], dict]:
    """按评估集跑"召回 → 重排打分",把命中票据与其它票据的分数分开收集。

    Returns:
        (pos_neg_scores, stats) —— pos_neg_scores 直接喂给 find_optimal_threshold。

    **正负例的定义**(本脚本的核心,也是与 FAQ 标定最大的不同):
    - 正例 = 正确答案在**同一次召回结果**里的重排分数;
    - 负例 = **同一次召回结果里**的其他票据分数。
    负例不是"随机不相关文档",而是"这次检索真的捞上来、但并非目标"的票 ——
    它们与问题在向量空间里足够近才会被召回,所以天然是**难负例**。

    ⚠ 两个必须知道的局限(决定了这份标定的可信边界):
    1. **负例是"同次召回的落选者",样本量被 top_n 卡住**:top_n=10、正例 1 张时,
       每条 query 最多 9 条负例;而真实误命中也可能来自**这条 query 没召回**的票。
       ⇒ 标出的阈值对"召回集之外"的票据没有直接约束力;
    2. **正例为空 ≠ 重排差**:那说明召回阶段就没捞到目标票。这类样本被记进
       `stats["missed"]` 并在 main 里单独提示 —— 它们是**召回**的待办,
       混进标定会把阈值往低处带(用召回缺陷去"证明"阈值太高)。

    stats 里四个计数器刻意分开:used(真正参与标定)、skipped_no_expected(样本无标注,
    如拒答/直答类)、skipped_empty_candidates(召回一条都没捞到)、missed(捞到了但没目标票)。
    合并成一个"跳过 N 条"就无法判断"标定集为什么这么小"。
    """
    pos_neg_scores: list[dict] = []
    stats = {"used": 0, "skipped_no_expected": 0, "skipped_empty_candidates": 0, "missed": []}

    for sample in samples:
        expected = list(sample.get("expected_ticket_ids") or [])
        # 无标注样本(拒答/直答类)无法定义正负例,跳过并计数
        if not expected:
            stats["skipped_no_expected"] += 1
            continue

        question = sample["question"]
        filter_expr = build_milvus_filter(sample.get("expected_filter") or {})
        # 用评估集里的**期望筛选条件**而不是 LLM 抽取结果:标定只关心排序质量,
        # 把"条件抽错"这个变量隔离掉(它由 run_stage_eval.py 的筛选字段准确率负责衡量)。
        candidates = vector_search(question, top_n=top_n, filter_expr=filter_expr or None)
        # `filter_expr or None`:空表达式传 None 而不是 "" —— 空字符串会被当成
        # "合法的空表达式"直接报错,传 None 才是"不过滤"的正确写法
        if not candidates:
            stats["skipped_empty_candidates"] += 1
            continue

        # 用 rerank_scores(原始分数)而不是 rerank():标定需要**未截断、未过阈值**的
        # 全部分数。若走 rerank() 会被**当前**阈值先卡一遍,标出来的平台就自我循环了
        # (等于只用"现有阈值能保留的样本"去算新阈值)
        scored = rerank_scores(question, candidates)
        # `is not None` 过滤:reranker 对个别文本可能返回 null 分数,
        # 混进来会让后面的排序统计出现 None 与 float 比较的异常
        pos = [row["rerank_score"] for row in scored if row["id"] in expected and row["rerank_score"] is not None]
        neg = [row["rerank_score"] for row in scored if row["id"] not in expected and row["rerank_score"] is not None]
        if not pos:
            # 单独记下:这是**召回**缺陷,不是重排缺陷
            stats["missed"].append(question)
        if not pos and not neg:
            stats["skipped_empty_candidates"] += 1
            continue

        pos_neg_scores.append({"query": question, "pos": pos, "neg": neg})
        stats["used"] += 1

    return pos_neg_scores, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="用标注评估集标定重排序相关度阈值")
    ap.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--top-n", type=int, default=0, help="每次召回多少条(0 = 用 RETRIEVAL_TOP_N)")
    # 下界 0.05 是**"取上沿"结论的事实依据**:平台下端就是这个搜索下界,
    # 而不是"再低会掉 F1"的悬崖。改这个参数会让取上沿的理由不再成立
    ap.add_argument("--start", type=float, default=0.05, help="阈值搜索下界")
    ap.add_argument("--stop", type=float, default=0.99, help="阈值搜索上界")
    ap.add_argument("--step", type=float, default=0.01, help="搜索步长")
    args = ap.parse_args()

    samples = load_jsonl(args.eval_set)
    # `args.top_n or settings.retrieval.top_n`:0 表示跟随生产配置 ——
    # 标定必须用**生产的召回宽度**,否则分数分布不是生产分布。
    # 这个参数留给"top_n 变化会不会影响阈值"的单变量实验用
    top_n = args.top_n or settings.retrieval.top_n
    # 打印 reranker 模型名:这份结果是**模型相关**的,报告必须能看出
    # 平台是用哪个 reranker 标出来的(换模型即失效,必须重标)
    print(f"评估集 {args.eval_set}:{len(samples)} 条;召回 {top_n} 条/题;reranker={settings.rerank.model}")
    print(f"当前 .env 的 RERANK_RELEVANCE_P = {settings.rerank.relevance_p}")

    scores, stats = collect_scores(samples, top_n)
    total_pos = sum(len(item["pos"]) for item in scores)
    total_neg = sum(len(item["neg"]) for item in scores)
    # 三个计数器一起打印:只报"可用样本 N 条"看不出标定集为什么小
    print(
        f"可用样本 {stats['used']} 条(正例 {total_pos} / 负例 {total_neg});"
        f"跳过:无标注 {stats['skipped_no_expected']}、无候选 {stats['skipped_empty_candidates']}"
    )
    # ⚠ 这条提示很重要:它把"召回没命中"从重排里摘出来。
    # 不看这行,很容易把召回缺陷当成"重排阈值太高"而去猛降阈值
    # (只打印前 5 条:这是给信号的,不是给人逐条读的)
    if stats["missed"]:
        print(f"⚠️ {len(stats['missed'])} 条样本召回阶段就没找到目标票据(不是重排的问题):")
        for question in stats["missed"][:5]:
            print(f"    - {question}")
    if total_pos == 0:
        # 没有正例 ⇒ F1 恒为 0 ⇒ 搜出来的"最优阈值"无意义。明确指出该去查召回
        raise SystemExit("没有任何正例分数,无法标定(先确认召回能命中标注票据)")

    best_threshold, best_metrics, all_metrics = find_optimal_threshold(
        scores, threshold_range=(args.start, args.stop), step=args.step
    )

    plateau_points = [m["threshold"] for m in f1_plateau(all_metrics)]
    plateau_start = plateau_points[0] if plateau_points else best_threshold
    plateau_end = plateau_points[-1] if plateau_points else best_threshold
    in_plateau = bool(plateau_points) and plateau_start <= settings.rerank.relevance_p <= plateau_end

    # 取**平台上沿**而不是中点:实测负例分数全部贴在 0 附近(最大 0.0074),
    # 平台下端并没有观测到悬崖,它只是搜索下界;而门控宁可保守 —— 上沿是
    # "仍然不掉 F1 的最高阈值",对没见过的负例余量最大(放错票据比漏票据更伤:
    # 漏了会走保守回复,放错会让模型拿无关上下文作答)。
    # ★ 注意 find_optimal_threshold 返回的第一项是**平台中点**,这里**刻意不用它** ——
    # 变量名叫 chosen_* 而不是 best_*,就是为了提醒读者"这是另行选定的取点"
    chosen_threshold = plateau_end
    # 从 all_metrics 反查上沿那一行,拿它完整的 accuracy/precision/recall/f1 落盘。
    # `abs(...) < 1e-9` 的模糊比较是必要的:阈值来自 np.arange 是 float,
    # 直接 `==` 可能因二进制表示差异匹配不到 ⇒ 回退成 best_metrics(**中点**那一行),
    # 那就把"上沿的决策"与"中点的指标"配错对了
    chosen_metrics = next(
        (m for m in all_metrics if abs(m["threshold"] - chosen_threshold) < 1e-9), best_metrics
    )

    # 排序后的分数分布(升序 ⇒ [0] 是最小、[-1] 是最大、[len//2] 是中位)。
    # **这是"该不该取上沿"的判据本身**:报告里保留它,是为了让别人能拿同一份数据
    # 复核这个决定,而不只是看到一句结论
    all_pos = sorted(s for item in scores for s in item["pos"])
    all_neg = sorted(s for item in scores for s in item["neg"])

    result = {
        "best_threshold": round(chosen_threshold, 4),
        "best_metrics": {k: round(v, 4) for k, v in chosen_metrics.items()},
        # 把**取点理由写进产物**:半年后有人打开这份 json 时,
        # 光看数字看不出为什么不是中点
        "plateau_tiebreak": "right-edge（负例贴 0、下端无观测悬崖，取不掉 F1 的最高阈值）",
        "f1_plateau": {
            "start": round(plateau_start, 4),
            "end": round(plateau_end, 4),
            "width": round(plateau_end - plateau_start, 4),
            # 同时记录中点:便于与 FAQ 阈值(取中点)的做法对照,
            # 将来若要改用中点也能直接取用
            "midpoint": round((plateau_start + plateau_end) / 2, 4),
            "current_env_in_plateau": in_plateau,
        },
        # 分布是这份标定最有**证据价值**的部分:它记录了当时的分数尺度,
        # 换 reranker 后拿它对比就能立刻看出"尺度变了、旧阈值不再适用"
        "score_distribution": {
            "positives": {"min": round(all_pos[0], 4), "median": round(all_pos[len(all_pos) // 2], 4), "max": round(all_pos[-1], 4)},
            "negatives": {"min": round(all_neg[0], 4), "median": round(all_neg[len(all_neg) // 2], 4), "max": round(all_neg[-1], 4)},
        },
        "current_env_threshold": settings.rerank.relevance_p,
        # 记录模型与端点:阈值是**模型相关**的,报告脱离模型信息就没有意义
        "reranker": {"model": settings.rerank.model, "base_url": settings.rerank.base_url},
        "embedding": {"model": settings.embedding.model},
        "dataset": {
            "eval_set": str(args.eval_set),
            "samples_used": stats["used"],
            "positives": total_pos,
            "negatives": total_neg,
            # 把"召回没命中"的清单也落盘:这些是评估集/召回侧的待办,
            # 不该随着终端输出一起丢掉
            "recall_missed": stats["missed"],
        },
        "grid": {"start": args.start, "stop": args.stop, "step": args.step, "points": len(all_metrics)},
        "curve": [
            {
                "threshold": round(m["threshold"], 4),
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
            }
            for m in all_metrics
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n===== 重排阈值标定完成 =====")
    print(f"推荐阈值(F1 平台上沿): {chosen_threshold:.2f}")
    # 用选定阈值**重算**指标(而不是直接用 chosen_metrics):
    # 自证"打印的数字与落盘的阈值一致",round 到 4 位后仍可复现
    accuracy, precision, recall, f1 = calculate_metrics(chosen_threshold, scores)
    print(f"  准确率 {accuracy:.4f} | 精确率 {precision:.4f} | 召回率 {recall:.4f} | F1 {f1:.4f}")
    print(f"F1 平台: {plateau_start:.2f} ~ {plateau_end:.2f} (宽 {plateau_end - plateau_start:.2f})")
    # 这一行是"上沿 vs 中点"决策的**原始证据**:负例全贴在 0 附近时,
    # 取上沿不会丢掉任何正例(正例最小值远高于上沿)
    print(
        f"打分布: 正例 {all_pos[0]:.4f} ~ {all_pos[-1]:.4f}(中位 {all_pos[len(all_pos) // 2]:.4f}) | "
        f"负例 {all_neg[0]:.4f} ~ {all_neg[-1]:.4f}(中位 {all_neg[len(all_neg) // 2]:.4f})"
    )
    print(f"当前 .env 值: {settings.rerank.relevance_p}")
    if in_plateau:
        # 同样的原则:平台内就说"没有改的理由"。当前 .env 的 0.22 正是这样定下来的
        print("结论:当前值已在 F1 平台内,与最优值同分 —— 没有改的理由。")
    else:
        # 复验命令给的是 run_stage_eval(看重排 Hit@K):重排阈值的影响直接体现在
        # 重排阶段,用它自己的指标复验比看最终答案更灵敏
        print(f"\n建议:把 .env 的 RERANK_RELEVANCE_P 改成 {chosen_threshold:.2f},")
        print("      再跑 `uv run python RAG/script/run_stage_eval.py --eval-set RAG/data/eval_set.jsonl` 复验 Rerank Hit@K。")
    print(f"\n结果 -> {out_path}")
    logger.info(
        f"[重排标定] 推荐阈值={chosen_threshold:.3f} F1={f1:.3f} "
        f"平台=[{plateau_start:.2f},{plateau_end:.2f}] -> {out_path}"
    )


if __name__ == "__main__":
    main()

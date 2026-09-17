# --- 路径引导：确保能导入 Python_Base 根目录的 config 与 RAG 内部包 ---
import sys as _sys
from pathlib import Path as _Path

_BASE = _Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
_sys.path.insert(0, str(_BASE))          # Python_Base 根（config.py）
_sys.path.insert(0, str(_BASE / "RAG"))  # RAG 根（core/llm/pipeline 等包）
"""阈值标定第二步:算相似度分数 → 按 F1 搜最优阈值(基础篇「阈值搜索」)。

流程与课案一致:
    query + 正例(同义问法) + 难负例(其他预设问题里语义最近的)
    → 向量化并归一化 → 点积得相似度 → 对候选阈值逐个算 P/R/F1 → 取 F1 最高者

产物 data/threshold_results.json 里的 best_threshold 用来定
`REDIS_SIM_THRESHOLD`(预设问答/FAQ 的相似度阈值),避免拍脑袋取值。
**当前实测口径**:20 query × 5 同义问法 + 100 难负例 → F1 平台 0.77~0.81,
取中点 **0.79**(上一个 0.85 是 Qwen3 向量空间的量纲,换 bge-m3 后过大)。

注意:`RERANK_RELEVANCE_P` 不在这里标定 —— 它需要"相关/不相关票据"的重排分数分布
(课案在微调章用带标注的评估集做),本项目的做法见 `script/calibrate_rerank.py`。

⚠ **本脚本的负例有个结构性盲点**(引用结论前必须知道):
难负例只从**其他预设问题**里挖,而 FAQ 层真实的高危误命中是
「同一问法换个名字」—— 实测「赵凡的登机牌座位号」与「赵飞的...」相似度 0.9008,
远高于标定出来的 0.79。这类近邻**不在候选集里**,所以本脚本能得到 F1=1.0 的
"完美"平台,却不能证明阈值对近邻安全。⇒ 标定集的负例必须覆盖"最像的那种错答案"。

用法:
    uv run python RAG/script/search_threshold.py
    uv run python RAG/script/search_threshold.py --step 0.02 --start 0.6 --stop 0.99

⚠ 注意 `--stop` 默认 **0.99** 而不是 1.0:配合 `np.arange` 的半开区间语义,
实际扫到 0.98。想覆盖到 0.99 需要传 1.0(见 evaluation/threshold.py 的说明)。
"""

import argparse
import json
from pathlib import Path

from config import settings
from core.logger import logger
from evaluation.threshold import (
    build_pos_neg_scores,
    build_threshold_items,
    calculate_metrics,
    f1_plateau,
    find_optimal_threshold,
)
from retrieval.embedding import embed_query

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = PROJECT_ROOT / "data" / "threshold_dataset.json"
DEFAULT_OUT = PROJECT_ROOT / "data" / "threshold_results.json"


def load_dataset(path: Path) -> list[dict]:
    """读取标定数据集(query + 正例);文件不存在时给出可执行的提示。

    两种失败都**给出下一步命令/原因**,而不是让调用方去看 traceback:
    - 文件不存在 ⇒ 提示先跑 `build_threshold_dataset.py --limit 20`(上一步);
    - 文件是空列表 ⇒ 说明上一步跑过但没产出(通常是 LLM 配置错)。

    这也说明本脚本有**前置依赖链**:build_threshold_dataset(要 LLM)
    → search_threshold(要 embedding)→ 人工改 .env → 复验。
    标定不是"跑一个脚本"而是一条流水线,别跳步。
    """
    if not path.exists():
        raise FileNotFoundError(
            f"标定数据集不存在: {path}\n先执行: uv run python RAG/script/build_threshold_dataset.py --limit 20"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data:
        raise ValueError(f"标定数据集为空: {path}")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="按 F1 搜索预设问答的最优相似度阈值")
    ap.add_argument("--dataset", default=str(DEFAULT_DATASET))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    # 搜索区间默认 0.5~0.99:下界 0.5 是**业务前提**(低于它就几乎什么都命中,
    # 没有讨论价值);上界 0.99 是因为余弦相似度接近 1 意味着"几乎同一句话",
    # 设成阈值等于放弃相似度匹配
    ap.add_argument("--start", type=float, default=0.5, help="阈值搜索下界")
    ap.add_argument("--stop", type=float, default=0.99, help="阈值搜索上界")
    ap.add_argument("--step", type=float, default=0.01, help="搜索步长")
    ap.add_argument("--num-negatives", type=int, default=5, help="每条 query 取几个难负例")
    args = ap.parse_args()

    data = load_dataset(Path(args.dataset))
    # 注意省略了 `if item["query"]` 的过滤(与 build_threshold_dataset 不同)——
    # 这里的 query 由上一步生成,已经过校验
    queries = [item["query"] for item in data]
    paraphrases = {item["query"]: item.get("pos", []) for item in data}
    print(f"标定集:{len(queries)} 条 query,共 {sum(len(v) for v in paraphrases.values())} 条正例")
    # 打印当前 .env 值:标定的最终结论是"要不要改 .env",把当前值打在开头,
    # 看输出时就不用再去翻配置文件了
    print(f"当前 .env 的 REDIS_SIM_THRESHOLD = {settings.redis.sim_threshold}")

    # 1) 组样本:正例来自同义问法,难负例来自其他预设问题(API 版难负例挖掘)
    items = build_threshold_items(
        queries=queries,
        paraphrases=paraphrases,
        embed=embed_query,
        num_negatives=args.num_negatives,
    )
    neg_total = sum(len(item["neg"]) for item in items)
    # `max(len(items), 1)` 防止除零(数据集为空时前面已抛错,这里是双保险);
    # 打印"平均每条 query 几条负例"很有用:远小于 --num-negatives 说明
    # 大量候选被 max_score/absolute_margin 过滤掉了,标定集实际上很"稀疏"
    print(f"难负例:{neg_total} 条(平均每条 query {neg_total / max(len(items), 1):.1f} 条)")
    # 正例全空 ⇒ 立刻退出:没有正例的 F1 曲线恒为 0,搜出来的"最优阈值"
    # 是毫无意义的任意值。宁可在这里失败
    if not any(item["pos"] for item in items):
        raise SystemExit("没有任何正例,无法标定")

    # 2) 向量化并算相似度分数
    # (这一句是本次标定里唯一花钱的地方:每条 query/正例/负例一次 embedding 调用)
    scores = build_pos_neg_scores(items, embed_query)

    # 3) 搜最优阈值
    best_threshold, best_metrics, all_metrics = find_optimal_threshold(
        scores,
        threshold_range=(args.start, args.stop),
        step=args.step,
    )

    # F1 平台:与最高 F1 并列的全部阈值。平台很宽时"最优阈值"只是一个代表值,
    # 当前 .env 只要落在平台内就没有改的必要(改了不提升 F1,反而多一次变量扰动)。
    plateau_points = [m["threshold"] for m in f1_plateau(all_metrics)]
    # 平台为空(理论上不会,但 f1_plateau 可能返回 [])时回退到 best_threshold,
    # 保证下面的 round() 不会因空列表索引而崩 —— 标定脚本跑到最后一步才炸最亏
    plateau_start = plateau_points[0] if plateau_points else best_threshold
    plateau_end = plateau_points[-1] if plateau_points else best_threshold
    # `bool(plateau_points) and ...`:空平台时 in_plateau 必须是 False,
    # 否则会输出"当前值已在平台内"的错误结论
    in_plateau = bool(plateau_points) and plateau_start <= settings.redis.sim_threshold <= plateau_end

    # 落盘的 result 是**给下一次标定做对比的**:除结论外还存了曲线(grid/curve)、
    # 数据集规模、当时用的 .env 值。下次改完阈值跑一遍,只要 diff 这两个 json
    # 就能看出"平台有没有移动、样本量变了没有"
    result = {
        "best_threshold": round(best_threshold, 4),
        "best_metrics": {k: round(v, 4) for k, v in best_metrics.items()},
        "f1_plateau": {
            "start": round(plateau_start, 4),
            "end": round(plateau_end, 4),
            "width": round(plateau_end - plateau_start, 4),
            "current_env_in_plateau": in_plateau,
        },
        "current_env_threshold": settings.redis.sim_threshold,
        "dataset": {
            "queries": len(queries),
            "positives": sum(len(v) for v in paraphrases.values()),
            "hard_negatives": neg_total,
        },
        "grid": {
            "start": args.start,
            "stop": args.stop,
            "step": args.step,
            "points": len(all_metrics),
        },
        # curve 保留**全部**阈值的 P/R/F1(不只是最优点):写报告/画图时要看
        # "平台两侧的悬崖陡不陡",只看 best 一个点看不出形状。
        # 只保留 4 位小数是刻意的(见文件头说明:多余精度是噪声)
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

    print("\n===== 阈值搜索完成 =====")
    print(f"最优阈值(F1 平台中点): {best_threshold:.2f}")
    # 这里**重新算一次**指标而不是直接用 best_metrics:验证"写入 json 的阈值
    # 与它对应的指标"是一致的(round(...,4) 之后仍能复现)。
    # 多一次纯计算换一个自证,很便宜
    accuracy, precision, recall, f1 = calculate_metrics(best_threshold, scores)
    print(f"  准确率 {accuracy:.4f} | 精确率 {precision:.4f} | 召回率 {recall:.4f} | F1 {f1:.4f}")
    print(f"F1 平台: {plateau_start:.2f} ~ {plateau_end:.2f} (宽 {plateau_end - plateau_start:.2f})")
    print(f"当前 .env 值: {settings.redis.sim_threshold}")
    if in_plateau:
        # ★ 这是本脚本最重要的输出:平台内就明确说"没有改的理由"。
        # 阈值标定的价值一半在于**阻止不必要的改动** —— 每次改阈值都要重新
        # 全量评估,而收益是 0
        print("\n结论:当前值已在 F1 平台内,与最优值同分 —— 没有改的理由。")
        print("      要动它得先有别的证据(例如线上误召回的 badcase),并按一次只改一个变量的流程复验。")
    else:
        # 不在平台内时给的是**建议 + 复验命令**,而不是替使用者改 .env:
        # 配置文件改动必须由人确认(本项目约定 agent 不动 .env)
        print(f"\n建议:当前值不在平台内,把 .env 的 REDIS_SIM_THRESHOLD 改成 {best_threshold:.2f} 后")
        print("      跑 `uv run python RAG/script/langfuse_evaluation.py --run-name sim-<值>` 复验再固化。")
    print(f"\n结果 -> {out_path}")
    logger.info(
        f"[阈值标定] 最优阈值={best_threshold:.3f} F1={f1:.3f} "
        f"平台=[{plateau_start:.2f},{plateau_end:.2f}] -> {out_path}"
    )


if __name__ == "__main__":
    main()

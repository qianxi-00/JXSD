# --- 路径引导：确保能导入 Python_Base 根目录的 config 与 RAG 内部包 ---
# 注意 **insert(0, ...) 的顺序**:先插 Python_Base 根、再插 RAG 根 ⇒ 最终
# sys.path 里 RAG 根排在前面。两个目录都要在,因为 config 在仓库根、core/llm/pipeline
# 在 RAG 根下,缺任何一个都会 import 失败。
# ⚠ 用 insert 而不是 append 是刻意的:本机仓库根/PATH 里可能有同名模块
# (比如另一个项目的 config.py),放前面才能确保 import 到的是本项目这份。
import sys as _sys
from pathlib import Path as _Path

# 从脚本位置向上找到仓库根(目录名 `Python_Base`)。这是**硬编码的停止条件**,
# 仓库目录改名后循环会一直退到盘符根,得到一个错误的 _BASE(见报告)。
_BASE = _Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
_sys.path.insert(0, str(_BASE))          # Python_Base 根（config.py）
_sys.path.insert(0, str(_BASE / "RAG"))  # RAG 根（core/llm/pipeline 等包）
"""按基础篇课案「系统评估」跑完整链路,输出分阶段指标报告。

⚠ 这份 docstring 的位置在 import 语句**之后**,不是文件首个语句 ⇒ 它**不是**
模块 docstring,而是一个被丢弃的字符串表达式(所以 `python -c "import ...; print(__doc__)"`
打印出来的是 None)。放在这里是为了让路径引导代码占据文件最前几行(在 import
config 之前就必须生效)。功能上无害 —— 本文档是给读代码的人看的,`help()` 拿不到,
本次只加注释不动位置(见报告)。

对应课案的三步:
1. 第二步「记录每个阶段的真实输出」:每个问题跑完整链路,保存路由、筛选条件、
   Milvus 过滤表达式、四路召回明细、合并候选、重排顺序、最终回答与耗时
   (由 evaluation/run_recorder.py 从事件流落地);
2. 第三步「按阶段计算指标」:路由准确率、筛选字段准确率、Recall@K、Rerank Hit@K、
   答案准确性(evaluation/stage_metrics.py);
3. 第四步(可选 --llm-score)「使用大模型评估最终回答」:按课案评分细则给 1~5 分。

用法:
    uv run python RAG/script/run_stage_eval.py --eval-set RAG/data/stage_eval_set.jsonl
    uv run python RAG/script/run_stage_eval.py --eval-set ... --llm-score --limit 10

⚠ 上面例子里的 `RAG/data/stage_eval_set.jsonl` **在本仓不存在**,能跑的是
`RAG/data/eval_set.jsonl`(由 script/build_eval_set.py 产出)。这是文档欠账,
本次不动代码/示例(见报告)。

产出两份文件(路径可用 --report/--records 覆盖):
- `data/stage_eval_report.json`:只有 summary(给 CI/看板读,小);
- `data/stage_eval_records.jsonl`:逐条 record(给排障/复算指标用,大)。
两份都落盘是刻意的 —— 报告是结论,records 是证据;指标算法改动后可以用同一批
records 离线复算,不必重跑链路(链路每次跑都要花钱调 API)。
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from config import settings
from core.logger import logger
from evaluation.eval_set import load_jsonl
from evaluation.llm_judge import score_answer
from evaluation.run_recorder import record_from_events
from evaluation.stage_metrics import summarize
from pipeline.rag_pipeline import RAGPipeline

# 产出目录固定为 `RAG/data/`(相对本脚本的父目录父目录),与 CWD 无关 ——
# 从仓库根或 RAG/ 下启动都写同一处,不会出现"报告生成了但不知道在哪"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT = PROJECT_ROOT / "data" / "stage_eval_report.json"
DEFAULT_RECORDS = PROJECT_ROOT / "data" / "stage_eval_records.jsonl"


async def run_samples(
    pipeline: RAGPipeline, samples: list[dict], llm_score: bool, use_cache: bool = False
) -> list[dict]:
    """逐条跑完整链路并组装评估 record。

    use_cache 默认 **False**：分阶段评估量的是路由/筛选/召回/重排，
    缓存命中会绕过这四段，让它们的指标一起被打成 0（实测一条 preset 命中把
    route/filter/recall/rerank 同时从 1.0 拉到 0.875）。要测缓存本身请显式 --use-cache。

    为什么默认值是 False 而不是"跟随生产配置":
    生产链路为了省 LLM 调用是**开着缓存**的,若评估也照搬,评估结果量的是
    "缓存+链路"的混合体,而缓存命中率取决于之前跑过哪些问题 ——
    同一份评估集连跑两次会得到不同指标(第二次全被缓存短路)。
    评估的目标是**可复现地度量链路能力**,所以必须关掉所有能短路的层。
    ★ 判据(写进 README/交接文档都该带的一句):报告里
    `cache_hit_count > 0` ⇒ 这份报告的四项链路指标不可信,先重跑。

    `time.perf_counter()` 而不是 `time.time()`:前者是单调时钟,
    后者会被系统时间调整(NTP 同步、手动改时间)影响,可能算出负数耗时。
    评估报告里的耗时是要拿去跟历史对比的。

    `stream=False`:评估**不需要流式** —— 流式只是为了首字延迟的体感,
    评估要的是完整答案与落盘的 record;而且 run_events 无论 stream 取什么值
    都产出同一组事件(stream 只控制 token 事件是否分片),所以关掉不影响指标。

    单条 LLM 打分用 `try/except` 包住(而不是让它冒泡):`--llm-score` 时每条都要
    多调一次 LLM,几十条里出现一次超时/限流是常态;让单条失败拖垮整轮评估
    = 前面几十条白跑(每次都是真金白银的 API 调用)。
    失败时**仍写入 `record["llm_score"] = None`**,保证所有 record 字段齐整,
    下游算均值时用 `is not None` 过滤(见 main)。
    """
    records: list[dict] = []
    for index, sample in enumerate(samples, 1):
        question = sample["question"]
        started = time.perf_counter()
        events: list[dict] = []
        # 先把事件**全部收集**再交给 record_from_events,而不是边收边解析:
        # 这样 record 组装是纯函数(输入完整事件列表),出问题时可以把事件流
        # 原样 dump 出来复现,不必再跑一遍链路
        async for event in pipeline.run_events(question, stream=False, use_cache=use_cache):
            events.append(event)
        elapsed = time.perf_counter() - started

        # round(elapsed, 3) 只保留毫秒精度:报告是要 diff 的,3 位小数足够,
        # 多余的浮点位会让同一次评估的两次落盘产生无意义的差异
        record = record_from_events(events, sample, elapsed_s=round(elapsed, 3))

        # 两个前置条件都是必要的:
        # - `sample.get("ground_truth")`:没有标准答案就没法评(拒答/直答样本理论上
        #   有答案,但直答样本的标准答案就是空串 ⇒ 会自动跳过,这是合理的);
        # - `record["answer"]`:空答案去打分只会浪费一次 API 调用
        if llm_score and sample.get("ground_truth") and record["answer"]:
            try:
                judged = score_answer(
                    question=question,
                    answer=record["answer"],
                    ground_truth=sample["ground_truth"],
                )
                record["llm_score"] = judged["score"]
                # feedback 截 500 字:一段完整的评分理由可能上千字,
                # 几十条 record 全量存会把 jsonl 撑到几 MB,排障时反而不好读
                record["llm_feedback"] = judged["feedback"][:500]
            except Exception as exc:  # noqa: BLE001 单条打分失败不拖垮整轮评估
                logger.warning(f"[评估] 第 {index} 条大模型打分失败: {exc}")
                record["llm_score"] = None

        records.append(record)
        # 进度行特意带上 route/候选数/重排保留数:**边跑边能看出链路是否退化**。
        # 只打印"完成 3/36"的话,要等全部跑完看报告才发现"召回一直是 0",
        # 那已经烧掉了几十次 API 调用
        flag = "" if record["actual_route"] != "error" else "  <-- 系统失败"
        print(
            f"[{index}/{len(samples)}] route={record['actual_route']} "
            f"候选={len(record['candidate_ids'])} 重排保留={len(record['reranked_ids'])} "
            f"耗时={elapsed:.2f}s{flag}"
        )
    return records


def write_records(records: list[dict], path: Path) -> None:
    """把逐条 record 写成 jsonl(UTF-8,末尾补换行)。

    没有复用 `evaluation/eval_set.save_jsonl`:那个函数会走 `validate_sample` 校验
    **评估样本**的字段契约,而 record 是**另一种结构**(多了 candidate_ids、
    actual_route 等运行期字段,且没有 task_type 之外的样本约束),
    用样本校验器去校验 record 只会误报。所以这里各写各的。
    ⚠ 代价是 record 的写出没有形状校验 —— 字段名写错(如 actual_filters 写成
    filter)不会报错,只会在 stage_metrics 那里静默算成 0(见报告)。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG 分阶段评估")
    ap.add_argument("--eval-set", required=True, help="评估集 jsonl 路径")
    # `--limit 0`(默认)表示**全部**:用 0 而不是 None 当"不限"的哨兵,
    # 这样命令行里 `--limit 0` 与不传是同一行为,不会出现"传了 0 结果一条都不跑"
    ap.add_argument("--limit", type=int, default=0, help="最多评估多少条(0 表示全部)")
    ap.add_argument("--k", type=int, default=5, help="Recall@K / Hit@K 的 K")
    ap.add_argument("--llm-score", action="store_true", help="额外用 EVALUATION_PROMPT 给回答打分")
    ap.add_argument(
        "--use-cache",
        action="store_true",
        help="走缓存层(默认关闭:缓存命中会绕过路由/召回/重排,把四类阶段指标一起打成 0)",
    )
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--records", default=str(DEFAULT_RECORDS))
    args = ap.parse_args()

    samples = load_jsonl(args.eval_set)
    # 先加载再截断(而不是只读前 N 行):`load_jsonl` 会校验**整份**评估集,
    # 这样"评估集第 37 行坏了"在 --limit 5 时也能被发现。
    # 代价是每次都解析整份文件(几十行,无所谓)
    if args.limit > 0:
        samples = samples[: args.limit]
    # 把"这次评估的关键前提"打在开头(集合名 + 缓存开关):报告是会被转发/贴到
    # issue 里的,不写清楚用的是哪个集合、有没有开缓存,别人无法判断数字是否可比
    print(
        f"评估集 {args.eval_set}:{len(samples)} 条;Milvus={settings.milvus.collection};"
        f"缓存={'开' if args.use_cache else '关(评估口径)'}"
    )

    pipeline = RAGPipeline()
    # run_samples 是 async(内部用 async for 消费事件流),这里用 asyncio.run 起一次事件循环。
    # 脚本级用法这样最简单;若要嵌进已有的异步服务,得改用 await 而不是 asyncio.run
    # (在运行中的事件循环里调 asyncio.run 会直接抛 RuntimeError)
    records = asyncio.run(run_samples(pipeline, samples, args.llm_score, use_cache=args.use_cache))

    summary = summarize(records, k=args.k)
    if args.llm_score:
        # 过滤掉 None(打分失败的条目不参与均值),并把"参与打分的条数"一并写进报告 ——
        # 只报均值不报条数会掩盖"36 条里只成功打了 3 条"这种情况
        scores = [r["llm_score"] for r in records if r.get("llm_score") is not None]
        summary["llm_score_count"] = len(scores)
        summary["llm_score_mean"] = sum(scores) / len(scores) if scores else 0.0

    # 先写 records 再写 report:report 里记了 records_file 路径,它指的是**本次**
    # 这次跑出来的明细。若顺序反过来、records 写失败,report 会指向一份过期明细
    write_records(records, Path(args.records))
    Path(args.report).write_text(
        json.dumps({"summary": summary, "records_file": args.records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 控制台摘要用固定宽度对齐,便于人工直接读(不用开 json);
    # 每一项都是 4 位小数 —— 评估指标到 0.0001 已经远超样本量能支撑的精度,
    # 再多位是噪声
    print("\n===== 分阶段指标 =====")
    print(f"样本数            : {summary['count']}")
    print(f"路由准确率        : {summary['route_accuracy']:.4f}")
    print(f"筛选字段准确率    : {summary['filter_field_accuracy']['overall']['accuracy']:.4f}"
          f" (命中 {summary['filter_field_accuracy']['overall']['hit']}/"
          f"{summary['filter_field_accuracy']['overall']['count']},"
          f"过抽取 {summary['filter_field_accuracy']['overall']['false_positive']})")
    # 注意这几个标签里的 K 是**运行参数**(默认 5),换 K 后报告标题会跟着变 ——
    # 引用数字时一定要带上 K,否则"Recall@5 是 0.83"和"Recall@10 是 0.94"会被混为一谈
    print(f"Recall@{args.k}         : {summary['recall_at_k']:.4f}")
    print(f"Rerank Hit@{args.k}     : {summary['rerank_hit_at_k']:.4f}")
    print(f"答案事实准确率    : {summary['answer_fact_accuracy']:.4f}")
    # 这两个计数是**判读报告可信度的开关**:缓存命中 > 0 说明指标被短路,
    # 过滤回退 > 0 说明筛选条件与库里字段对不上
    print(f"缓存命中          : {summary['cache_hit_count']} 条")
    print(f"过滤零命中回退    : {summary['filter_fallback_count']} 条")
    print(f"平均/最大耗时     : {summary['mean_elapsed_s']:.2f}s / {summary['max_elapsed_s']:.2f}s")
    if args.llm_score:
        print(f"大模型评分均值    : {summary['llm_score_mean']:.2f}({summary['llm_score_count']} 条)")
    print(f"\n报告 -> {args.report}\n明细 -> {args.records}")


if __name__ == "__main__":
    main()

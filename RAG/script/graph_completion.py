"""链接补全与人工审核（T6）。

课案口径（优化篇「孤立节点的处理」）：
    定期链接补全任务扫出孤点，由 LLM 对照已有节点提议候选关系，
    过置信度阈值的落库，存疑的进人工审核。
    整体原则：该连的通过延迟链接和补全任务最终连上，不该连的不乱连。

四个动作（前三个是子命令，默认 `scan`）：
    scan        扫孤点 → 挑候选对照实体 → LLM 提议关系 → 分流 → 落库/入队
    review-list 看待审核队列（含每条提议的依据）
    review-ok   批准一条：落库并记状态
    review-no   驳回一条：只记状态，不碰图

成本：**每个有候选的孤点花 1 次 LLM 调用**（没候选的直接跳过）。先 `--dry-run` 看规模。

用法（在 Python_Base 目录下执行）：
    uv run python RAG/script/graph_completion.py --dry-run            # 空跑：只看会做什么
    uv run python RAG/script/graph_completion.py --limit 3            # 真跑，最多 3 个孤点
    uv run python RAG/script/graph_completion.py review-list
    uv run python RAG/script/graph_completion.py review-ok  a1b2c3d4
    uv run python RAG/script/graph_completion.py review-no  a1b2c3d4

审核队列默认落 `RAG/data/graph_review_queue.jsonl`（本地运行文件，不入库）。
这条链路**只加边、不删节点**：补全连错一条边可以人工驳回/手工删，
但"该连的没连"只是漏，不会污染图谱。
"""

import argparse
import json
import sys
from pathlib import Path

# --- 路径引导：脚本直接跑时 sys.path[0] 是 RAG/script，需要手动挂上项目根与 RAG 根 ---
_BASE = Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
for _path in (str(_BASE), str(_BASE / "RAG")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from core.logger import logger  # noqa: E402
from graph_rag import completion  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 - 老环境不支持就算了
    pass


def _scan(args) -> int:
    result = completion.run_completion(
        limit=args.limit,
        dry_run=args.dry_run,
        auto=args.auto_confidence,
        review=args.review_confidence,
        path=None if args.no_queue else args.queue,
    )
    print(f"[completion] 扫描孤点      : {result['scanned']}（其中无候选对照 {result['no_candidate']}）")
    print(f"[completion] 自动落库      : {result['auto']}{'（DRY-RUN 未写图）' if args.dry_run else ''}")
    print(f"[completion] 进人工审核    : {result['review']}")
    print(f"[completion] 分数不足丢弃  : {result['dropped']}")
    for item in (result["would_apply"] if args.dry_run else result["applied"])[:20]:
        mark = "" if args.dry_run else (" ok" if item.get("ok", True) else " SKIPPED(端点缺失)")
        print(f"[completion]   {item['source']} -{item['relation']}-> {item['target']}"
              f"  conf={item['confidence']}  {item['reason'][:40]}{mark}")
    for record in result["enqueued"][:20]:
        print(f"[completion]   待审 {record.get('id')}: {record.get('source')} -{record.get('relation')}-> "
              f"{record.get('target')}  conf={record.get('confidence')}")
    for item in result["dropped_items"][:10]:
        print(f"[completion]   丢弃 {item['source']} -> {item['target']} conf={item['confidence']}")

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


def _review_list(args) -> int:
    summary = completion.describe_queue(path=args.queue)
    print(f"[completion] 队列 {summary['path']}：共 {summary['total']} 条 {summary['counts']}")
    for record in summary["pending"]:
        print(f"[completion]   {record['id']}  {record['source']} -{record['relation']}-> {record['target']}"
              f"  conf={record['confidence']}  {record.get('reason', '')[:50]}")
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


def _review(args, approve: bool) -> int:
    try:
        record = completion.apply_review(args.item_id, approve=approve, path=args.queue)
    except KeyError as exc:
        # 拼错 id 必须是非零退出：脚本报着 OK 却没落库是最坑的
        print(f"[completion] 失败：{exc}")
        return 1
    verb = "已批准并落库" if approve else "已驳回"
    print(f"[completion] {record['id']} {verb}：{record['source']} -{record['relation']}-> {record['target']}")
    if approve and not record.get("applied"):
        print("[completion] ⚠ 落库被跳过（端点不在图里，见日志 WARNING）")
    logger.info(f"[补全-审核] {record['id']} -> {record['status']}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="链接补全 + 人工审核（T6）")
    parser.add_argument("action", nargs="?", default="scan",
                        choices=["scan", "review-list", "review-ok", "review-no"])
    parser.add_argument("item_id", nargs="?", help="review-ok / review-no 的队列记录 id")
    parser.add_argument("--limit", type=int, default=20, help="本轮最多处理几个孤点")
    parser.add_argument("--dry-run", action="store_true", help="空跑：不落库、不入队")
    parser.add_argument("--no-queue", action="store_true", help="不写审核队列（只落库）")
    parser.add_argument("--queue", default=str(completion.REVIEW_QUEUE_PATH), help="审核队列文件")
    parser.add_argument("--auto-confidence", type=float, default=completion.AUTO_LINK_CONFIDENCE,
                        help=f"自动落库门槛（默认 {completion.AUTO_LINK_CONFIDENCE}）")
    parser.add_argument("--review-confidence", type=float, default=completion.REVIEW_LINK_CONFIDENCE,
                        help=f"进人工审核的门槛（默认 {completion.REVIEW_LINK_CONFIDENCE}）")
    parser.add_argument("--json", action="store_true", help="附带输出 JSON（给机器读）")
    args = parser.parse_args(argv)

    if args.action == "scan":
        return _scan(args)
    if args.action == "review-list":
        return _review_list(args)
    if not args.item_id:
        parser.error(f"{args.action} 需要一个队列记录 id（用 review-list 看）")
    return _review(args, approve=args.action == "review-ok")


if __name__ == "__main__":
    raise SystemExit(main())

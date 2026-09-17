"""在人工标注集上评估实体链接决策：链接准确率 + 误建重复节点率（T6）。

课案口径（优化篇「孤立节点的处理」）：
    评估侧要新增链接准确率（该链接的有没有连对）与重复节点率（该新建的有没有误建重复节点）
    两个指标，否则别名判断出错会静默污染图谱。

被评估的决策 = `builder.match_entity(name, entity_type)`（精确 → 向量 top10（>=0.6 且同类型）→ LLM 判定），
即增量入库 `link_or_create` 真正用的那一步 —— 评的是**产线代码**，不是另写一份判定逻辑。

标注集格式（JSONL，一行一条，`matched` 是人工确认的答案）：
    {"name": "老张", "entity_type": "人物", "matched": "张三"}   # 该连到已有节点"张三"
    {"name": "差旅费", "entity_type": "费用类型", "matched": null} # 该新建（图里没有同一对象）

⚠ 标注集要**人工确认**，不要拿模型输出当答案（那是自证）。本项目尚未接入 alias 消歧，
   所以"该连"的样本目前主要来自同义/简称/别名的人工判断，规模小时结论仅供参考。

成本：每条**非精确命中**的标注 ≈ 1 次 embedding + 1 次 LLM 判定（精确命中直接短路）。
先看标注集条数再跑。

用法（在 Python_Base 目录下执行）：
    uv run python RAG/script/eval_graph_linking.py --labels RAG/data/graph_linking_labels.jsonl
    uv run python RAG/script/eval_graph_linking.py --labels labels.jsonl --report out.json
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
from graph_rag import builder, models, quality  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 - 老环境不支持就算了
    pass


def load_labels(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"标注集不存在：{path}\n"
            "格式（JSONL，逐行一个对象）：\n"
            '  {"name": "老张", "entity_type": "人物", "matched": "张三"}\n'
            '  {"name": "差旅费", "entity_type": "费用类型", "matched": null}\n'
            "`matched` 必须由**人工**确认：填已有实体名表示该连过去，填 null 表示该新建。"
        )
    labels = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"标注集第 {number} 行不是合法 JSON：{exc}") from exc
        if "name" not in item or "matched" not in item:
            raise SystemExit(f"标注集第 {number} 行缺字段：需要 name 与 matched（可为 null）")
        labels.append(item)
    return labels


def decide(labels: list[dict]) -> list[dict]:
    """对每条标注跑一次真实链接决策，返回与标注同构的 decisions。"""
    models.connect()
    decisions = []
    for item in labels:
        node = builder.match_entity(item["name"], item.get("entity_type"))
        decisions.append(
            {
                "name": item["name"],
                "entity_type": item.get("entity_type"),
                "matched": node.name if node is not None else None,
            }
        )
    return decisions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="实体链接决策评估（链接准确率 / 误建重复节点率）")
    parser.add_argument("--labels", required=True, help="人工标注集 JSONL")
    parser.add_argument("--report", help="把结果落成 JSON")
    parser.add_argument("--show-decisions", action="store_true", help="打印每条决策与标注")
    args = parser.parse_args(argv)

    labels = load_labels(Path(args.labels))
    if not labels:
        raise SystemExit(f"标注集是空的：{args.labels}")

    print(f"[link-eval] 标注 {len(labels)} 条，逐条跑 builder.match_entity（非精确命中会花 LLM 调用）…")
    decisions = decide(labels)
    metrics = quality.link_decision_metrics(decisions, labels)

    if args.show_decisions:
        for decision, label in zip(decisions, labels):
            print(f"[link-eval]   {label['name']}（{label.get('entity_type') or '-'}）"
                  f" 标注={label['matched'] or '新建'} 决策={decision['matched'] or '新建'}")

    link_accuracy = metrics["link_accuracy"]
    duplicate_rate = metrics["duplicate_rate"]
    print(f"[link-eval] 可配对样本    : {metrics['resolved']}/{len(labels)}"
          f"（配对不上 {len(metrics['unresolved'])} 条，见报告 unresolved）")
    if link_accuracy is None:
        # 分母为 0 时如实说"没测"，不报 0%（0% 会被读成"全错"）
        print("[link-eval] 链接准确率    : 无（标注集里没有「该连」的样本）")
    else:
        print(f"[link-eval] 链接准确率    : {link_accuracy:.2%}（{metrics['link_total']} 条标「该连」）")
    if duplicate_rate is None:
        print("[link-eval] 误建重复节点率: 无（分母同链接准确率）")
    else:
        print(f"[link-eval] 误建重复节点率: {duplicate_rate:.2%}"
              f"（{len(metrics['false_create'])} 条该连却新建）")
    print(f"[link-eval] 反向错误(吞节点): {len(metrics['false_link'])} 条；连错对象: {len(metrics['target_mismatch'])} 条")
    for record in metrics["false_create"][:10]:
        print(f"[link-eval]   误建重复 {record['name']}: 该连 {record['expected']}，实际新建")
    for record in metrics["target_mismatch"][:10]:
        print(f"[link-eval]   连错对象 {record['name']}: 该连 {record['expected']}，实际连到 {record['decided']}")
    for record in metrics["false_link"][:10]:
        print(f"[link-eval]   误连吞节点 {record['name']}: 该新建，实际连到 {record['decided']}")

    if args.report:
        payload = {
            "labels": str(args.labels),
            "metrics": metrics,
            "decisions": decisions,
            "thresholds": {"match_score": builder.MATCH_SCORE_THRESHOLD,
                           "match_candidate_top_k": builder.MATCH_CANDIDATE_TOP_K},
        }
        Path(args.report).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[link-eval] 报告已写入 {args.report}")

    logger.info(f"[link-eval] 完成 | 标注 {len(labels)} | 链接准确率 {link_accuracy} | 误建重复率 {duplicate_rate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""图谱健康度体检：孤立节点率 / 重复节点率（T6）。

课案口径（优化篇「孤立节点的处理」）：
    用孤立节点率（孤点数 / 总节点数）监控图谱健康度，突增说明抽取或匹配环节退化。

诚实地说明这个脚本能证明什么、不能证明什么：
- **能**：给出三个可观测量（孤点数与名单、归一化重名组、社区覆盖），并和上一份报告对比，
  孤点率/重复率突增（>= 5 个百分点）时在输出里直接报警。
- **不能**：判定"孤点是不是质量问题"。课案说得很清楚 —— 源文档本身没提供关系时，
  孤点是"证据还没到位"的正常状态（不强行造边）；只有"本该连上却没连上"才是缺陷。
  区分这两者需要标注集，见 `script/eval_graph_linking.py`。
  所以本脚本的读数**只能当趋势看**，单次 0 不代表图谱没退化（这次真机基线恰好是 0）。

用法（在 Python_Base 目录下执行）：
    uv run python RAG/script/graph_health.py                      # 打印 + 落 RAG/data/graph_health.json
    uv run python RAG/script/graph_health.py --out report.json    # 换落点
    uv run python RAG/script/graph_health.py --json               # 只输出 JSON（给人之外的东西读）

不给 `--out` 时默认写 `RAG/data/graph_health.json`，第二次跑就会拿它当"上次报告"做对比。
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# --- 路径引导：脚本直接跑时 sys.path[0] 是 RAG/script，需要手动挂上项目根与 RAG 根 ---
_BASE = Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
for _path in (str(_BASE), str(_BASE / "RAG")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from core.logger import logger  # noqa: E402
from graph_rag import completion, models, quality  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 - 老环境不支持就算了
    pass

DEFAULT_OUT = _BASE / "RAG" / "data" / "graph_health.json"


def build_report() -> dict:
    """采集一次健康度：规模 + 三个指标 + 社区覆盖。"""
    models.connect()
    graph = completion.collect_graph()
    nodes, relationships = graph["nodes"], graph["relationships"]

    isolated = quality.isolated_node_rate(nodes, relationships)
    duplicate = quality.normalized_duplicate_rate([node.get("name") for node in nodes])
    coverage = quality.community_coverage(nodes)
    communities = models.run_cypher("MATCH (c:Community) RETURN count(c) AS count")
    community_count = (communities[0]["count"] if communities else 0)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scale": {
            "entities": isolated["total"],
            "relationships": len(relationships),
            "communities": community_count,
            "unassigned_community": coverage["unassigned_count"],
        },
        "metrics": {
            "isolated": isolated,
            "duplicate": duplicate,
            "community": coverage,
            # 链接准确率不在这里：它必须对着人工标注集算（quality.link_decision_metrics），
            # 这里留一个 None 占位，免得报表读者以为"没报就是 100%"。
            "link_accuracy": None,
        },
        "notes": [
            "孤立节点率只能看趋势：源文档没给关系时孤点是正常状态，不强行造边。",
            "重复节点率是归一化重名的代理指标，抓不到近义实体（差旅费 vs 出差费用）。",
        ],
    }


def load_previous(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning(f"[体检] 上一份报告不是合法 JSON，本次不做对比: {exc}")
        return None


def print_report(report: dict, comparison: dict) -> None:
    """控制台只打 ASCII 标签 + 中文值（Windows 控制台编码在重定向时会被 reconfigure 兜住）。"""
    scale = report["scale"]
    metrics = report["metrics"]
    isolated, duplicate = metrics["isolated"], metrics["duplicate"]
    print(f"[graph-health] generated_at : {report['generated_at']}")
    print(f"[graph-health] entities     : {scale['entities']}")
    print(f"[graph-health] relationships: {scale['relationships']}")
    print(f"[graph-health] communities  : {scale['communities']}（未分配社区 {scale['unassigned_community']} 个实体）")
    print(f"[graph-health] isolated     : {isolated['isolated_count']}/{isolated['total']} = {isolated['rate']:.2%}")
    print(f"[graph-health] duplicate    : {duplicate['duplicate_count']}/{duplicate['total']} = {duplicate['rate']:.2%}")
    if isolated["isolated"]:
        sample = "、".join(isolated["isolated"][:10])
        print(f"[graph-health] 孤点样例      : {sample}{' …' if len(isolated['isolated']) > 10 else ''}")
    if duplicate["groups"]:
        for key, names in list(duplicate["groups"].items())[:5]:
            print(f"[graph-health] 重名组        : {key} <- {'、'.join(names)}")
    if comparison["deltas"]:
        deltas = "，".join(f"{key} {value:+.2%}" for key, value in comparison["deltas"].items())
        print(f"[graph-health] 对比上次      : {deltas}")
    for alert in comparison["alerts"]:
        print(f"[graph-health] ⚠ 告警        : {alert}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="图谱健康度体检（孤点率/重复率 + 与上次对比）")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="报告落点（同时作为下次对比的基线）")
    parser.add_argument("--json", action="store_true", help="只输出 JSON，不打人类可读摘要")
    parser.add_argument("--no-write", action="store_true", help="只算不落盘（不动基线）")
    args = parser.parse_args(argv)

    out_path = Path(args.out)
    report = build_report()
    comparison = quality.compare_with_previous(load_previous(out_path), report)
    report["comparison"] = comparison

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report, comparison)

    if not args.no_write:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.json:
            print(f"[graph-health] 报告已写入 {out_path}")

    # 报警时用退出码 2 表达"体检有异常"：定时任务据此发通知，不必去解析中文输出
    return 2 if comparison["alerts"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

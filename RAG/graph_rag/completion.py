"""链接补全 + 人工审核闭环（T6）。

课案口径（优化篇「孤立节点的处理」）：
    该连的通过延迟链接和补全任务最终连上，不该连的不乱连 —— 定期链接补全任务扫出孤点，
    由 LLM 对照已有节点提议候选关系，过置信度阈值的落库，存疑的进人工审核。

流程（`run_completion`）：扫孤点 → 给每个孤点挑候选对照实体（`candidate_pool`）→
LLM 提议关系（`propose_links`）→ 按置信度分流（`classify_proposals`）→
过 `AUTO_LINK_CONFIDENCE` 的落库、过 `REVIEW_LINK_CONFIDENCE` 的进审核队列、更低的丢弃但计数。

为什么"人工审核"用 **JSONL 队列 + 命令行**，而不是 langgraph 的 interrupt：
    补全任务是**离线批处理**（手工跑或定时跑），人工审核可能发生在几小时甚至几天之后、
    也可能换个人来做。`interrupt` 的恢复点是**进程内**的 state/checkpoint，跨不了这段时间；
    队列文件才是这段等待的正确载体。agent 会话里的 HITL 仍然走 langgraph（见 script/hitl_demo.py），
    两者不冲突 —— 一个审的是"agent 要执行的动作"，一个审的是"离线任务提议的图谱边"。

⚠ 成本：每个**有候选**的孤点会花 1 次 LLM 调用（没有候选的直接跳过，不花钱）。
先跑 `--dry-run` 看规模。
"""

import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.logger import logger

from graph_rag import builder, models, quality

# 自动落库的置信度门槛：>= 这个值直接连边
AUTO_LINK_CONFIDENCE = 0.8
# 进人工审核的门槛：>= 这个值进队列等人工决定；< 这个值丢弃（计数但不留档）
REVIEW_LINK_CONFIDENCE = 0.5
# 关系语义的兜底值（注意这是 RelatesTo.relation_type **属性**的值，
# Neo4j 边类型恒为 "RELATES_TO"，见 models.Entity.relates_to）
DEFAULT_RELATION_TYPE = "相关"
# 审核队列默认落点：与 RAG/data 下其它运行产物一样是**本地运行文件**（不入库）
REVIEW_QUEUE_PATH = Path(__file__).resolve().parent.parent / "data" / "graph_review_queue.jsonl"

# 候选对照实体给几个：太多会把提示词撑长、也更容易诱发模型乱连
CANDIDATE_TOP_K = 8

COMPLETION_SYSTEM_PROMPT = """你是知识图谱的链接补全助手。
图上有一个"孤立实体"（与其他任何实体都没有关系），另给你一批已有实体作为候选对照。
请判断：该孤立实体与候选里的哪些实体**确实存在关系**，并给出关系名与置信度。

硬约束：
1. target 只能用候选列表里出现过的实体名，编造的名字一律无效；
2. 关系必须能从双方的名称/类型/描述直接看出来。看不出来就不要提 ——
   宁可少提，也不乱连（乱连会污染图谱，比漏连更难发现）；
3. confidence 取 0~1：0.9 以上表示证据明确，0.5~0.8 表示像但不确定，低于 0.5 不要提；
4. 判断同义/重复实体时，只有**指同一对象**才用"同义"这类关系；
   仅仅是同类型（都是费用类型）不算关系。

只输出 JSON，不要解释：
{"links": [{"target": "候选里的实体名", "relation": "关系名", "confidence": 0.9, "reason": "一句话依据"}]}
没有任何关系时输出 {"links": []}"""


def as_confidence(value) -> float | None:
    """把模型给的置信度解析成 `[0, 1]`；解析不出来返回 `None`。

    返回 `None` 而不是默认 `1.0` 是刻意的：缺字段/写"高"这类情况本来就该**降级到人工审核**，
    给它补一个满分等于让模型的胡话直接落库（`classify_proposals` 把 None 送进 review）。
    """
    if value is None or isinstance(value, (bool, list, dict)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return min(1.0, max(0.0, number))


def classify_proposals(proposals: list[dict], auto: float = AUTO_LINK_CONFIDENCE,
                       review: float = REVIEW_LINK_CONFIDENCE) -> dict:
    """按置信度分流成 `auto`（直接落库）/ `review`（人工审核）/ `dropped`（丢弃）。

    阈值是**闭区间**下界（`>=`）。置信度缺失（None）一律进审核：不落库、也不悄悄丢。
    `dropped` 也带在返回值里 —— "任务跑了但一条没连上"要能归因到"分数都不够"。
    """
    buckets: dict[str, list] = {"auto": [], "review": [], "dropped": []}
    for proposal in proposals or []:
        target = str((proposal or {}).get("target") or "").strip()
        if not target:
            # 没有 target 的提议是废数据，不进任何桶（也不进 dropped：那不是"分数不够"）
            continue
        item = dict(proposal)
        item["target"] = target
        item["confidence"] = as_confidence(proposal.get("confidence"))
        score = item["confidence"]
        if score is None:
            buckets["review"].append(item)
        elif score >= auto:
            buckets["auto"].append(item)
        elif score >= review:
            buckets["review"].append(item)
        else:
            buckets["dropped"].append(item)
    return buckets


def _cosine(left, right) -> float:
    """余弦相似度；任一向量为空/长度不等则返回 0.0（当作"没有相似度信息"）。"""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(a * a for a in left))
    norm_right = math.sqrt(sum(b * b for b in right))
    if not norm_left or not norm_right:
        return 0.0
    return dot / (norm_left * norm_right)


def candidate_pool(node: dict, entities: list[dict], top_k: int = CANDIDATE_TOP_K) -> list[dict]:
    """给孤点挑候选对照实体：**同 entity_type 优先**，组内按向量余弦降序。

    为什么限同类型：`builder.match_entity` 也是这个口径 —— "张三"（人物）和"张三"（公司）
    向量上可能很像，但业务上不是一个东西。跨类型的候选只在同类型不够时补位（`top_k` 没填满时），
    因为孤点常常是"类型也抽错了"，完全锁死同类型会让它一个候选都拿不到。

    没有向量时（description 为空的历史节点）退化成按 `mentions` 降序：
    "被提及最多的"至少是个有依据的排序，比随机取强。

    ⚠ 规模化提示：这里是**全量实体在 Python 里比对**（当前几百个节点无感）。
    节点上千时应改成走 Neo4j 原生向量索引（`models.ensure_vector_indexes` 建的那个，
    `builder.match_entity` 已经在用），否则每次补全都是一次全表扫描。
    """
    pool = []
    for entity in entities or []:
        name = str((entity or {}).get("name") or "").strip()
        if not name or name == str(node.get("name") or "").strip():
            continue
        same_type = str(entity.get("entity_type") or "") == str(node.get("entity_type") or "")
        pool.append(
            {
                "name": name,
                "entity_type": entity.get("entity_type") or "",
                "description": entity.get("description") or "",
                "mentions": entity.get("mentions") or 0,
                "same_type": same_type,
                "score": _cosine(node.get("embedding"), entity.get("embedding")),
            }
        )

    # 排序键：先同类型，再向量分（没有向量的节点分数恒 0，自然落到"按 mentions"）
    pool.sort(key=lambda item: (item["same_type"], item["score"], item["mentions"]), reverse=True)
    return pool[: max(0, top_k)]


def _node_material(node: dict) -> str:
    return (
        f"名称：{node.get('name')}\n"
        f"类型：{node.get('entity_type') or '未知'}\n"
        f"描述：{node.get('description') or '（无）'}"
    )


def _candidate_material(candidates: list[dict]) -> str:
    lines = []
    for item in candidates:
        lines.append(
            f"- {item['name']}｜类型：{item.get('entity_type') or '未知'}"
            f"｜描述：{item.get('description') or '（无）'}"
        )
    return "\n".join(lines)


def propose_links(node: dict, candidates: list[dict], call_llm=None) -> list[dict]:
    """让 LLM 在候选里挑出确实存在的关系，返回 `[{target, relation, confidence, reason}]`。

    **只保留候选池里出现过的 target**：模型编出来的名字直接丢（与 `builder.upsert_relation`
    "端点不在图里就跳过"同一条原则 —— 关系必须落在已有节点上、可追溯）。
    自环也丢掉（补全的目的是把孤点连进图，不是给它造一条指向自己的边）。
    """
    if not candidates:
        # 没有对照实体就没得比较：不调模型、不花钱
        return []

    if call_llm is None:
        call_llm = builder.call_llm

    user = (
        f"【孤立实体】\n{_node_material(node)}\n\n"
        f"【候选实体】\n{_candidate_material(candidates)}\n\n"
        "请按系统提示给出 JSON。"
    )
    raw = call_llm(system=COMPLETION_SYSTEM_PROMPT, user=user, temperature=0.1, json_mode=True)
    try:
        # `load_json_object` 已经做了"剥 ```json 围栏 + 顶层必须是对象"的校验，
        # 失败时抛 ValueError（不是返回空 dict）—— 所以这里只接异常，不再自己 json.loads。
        data = raw if isinstance(raw, dict) else builder.load_json_object(raw)
    except (TypeError, ValueError) as exc:
        # 模型偶尔不吐 JSON：这一条孤点本轮放弃，但要留下痕迹（否则"没连上"无法归因）
        logger.warning(f"[补全] 提议结果不是合法 JSON，跳过该孤点 {node.get('name')}: {exc}")
        return []

    allowed = {item["name"] for item in candidates}
    proposals = []
    for item in (data or {}).get("links") or []:
        if not isinstance(item, dict):
            continue
        target = str(item.get("target") or "").strip()
        if target not in allowed or target == str(node.get("name") or "").strip():
            continue
        relation = str(item.get("relation") or "").strip() or DEFAULT_RELATION_TYPE
        proposals.append(
            {
                "target": target,
                "relation": relation,
                "confidence": as_confidence(item.get("confidence")),
                "reason": str(item.get("reason") or "").strip(),
            }
        )
    return proposals


def collect_graph() -> dict:
    """从 Neo4j 读全图（节点名 + 关系端点）。

    刻意**不用** `NOT (e)-[:RELATES_TO]-()` 那种"孤点"写法：那句会把自环当成"已连接"，
    而 `quality.isolated_node_rate` 的口径是"自环仍算孤点"（只和自己有关系 = 没连进图）。
    两份口径不能同时存在，所以隔离判定统一在 Python 里做一次。
    """
    models.connect()
    nodes = models.run_cypher(
        "MATCH (e:Entity) RETURN e.name AS name, e.entity_type AS entity_type, "
        "e.description AS description, e.community_id AS community_id, "
        "e.mentions AS mentions, e.embedding AS embedding"
    )
    relationships = models.run_cypher(
        "MATCH (a:Entity)-[:RELATES_TO]->(b:Entity) RETURN a.name AS source, b.name AS target"
    )
    return {"nodes": nodes, "relationships": relationships}


def all_entities() -> list[dict]:
    """全部实体（补全时的候选来源）。"""
    return collect_graph()["nodes"]


def isolated_nodes(limit: int = 50) -> list[dict]:
    """取孤点（按名称排序，口径同 `quality.isolated_node_rate`），最多 `limit` 个。"""
    graph = collect_graph()
    report = quality.isolated_node_rate(graph["nodes"], graph["relationships"])
    wanted = set(report["isolated"][:limit])
    return [node for node in graph["nodes"] if node.get("name") in wanted]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def enqueue_review(items: list[dict], path: Path | str = REVIEW_QUEUE_PATH) -> list[dict]:
    """把存疑提议追加进审核队列（JSONL），返回带 `id`/`status`/`created_at` 的记录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with path.open("a", encoding="utf-8") as handle:
        for item in items or []:
            record = {
                "id": uuid.uuid4().hex[:12],
                "source": str(item.get("source") or ""),
                "target": str(item.get("target") or ""),
                "relation": str(item.get("relation") or DEFAULT_RELATION_TYPE),
                "confidence": as_confidence(item.get("confidence")),
                "reason": str(item.get("reason") or ""),
                "status": "pending",
                "created_at": _now(),
                "decided_at": None,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
    return records


def load_review_queue(path: Path | str = REVIEW_QUEUE_PATH, status: str | None = None) -> list[dict]:
    """读审核队列；`status` 非空时只返回该状态的记录。文件不存在 ⇒ 空列表（不是错误）。"""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(f"[补全] 审核队列里有一行不是合法 JSON，已跳过：{line[:80]}")
            continue
        if status is None or record.get("status") == status:
            records.append(record)
    return records


def _write_queue(records: list[dict], path: Path) -> None:
    """整文件重写（队列是小文件，重写比做增量状态文件简单且不会两处真相）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def apply_review(item_id: str, approve: bool, path: Path | str = REVIEW_QUEUE_PATH,
                 upsert_relation=None) -> dict:
    """人工裁决一条提议：批准则落库（`upsert_relation`），然后**把状态写回队列**。

    未知 id 抛 `KeyError` —— 命令行拼错一个 id 却"看起来成功了"是最坑的。
    """
    path = Path(path)
    records = load_review_queue(path=path)
    target_record = next((record for record in records if record.get("id") == item_id), None)
    if target_record is None:
        raise KeyError(f"审核队列里没有 id={item_id} 的记录")

    if approve:
        if upsert_relation is None:
            # 必须先建连接：人工审核这条路**只读队列文件**，不像补全那样先走 collect_graph()
            # 顺带把连接建好。少了这一句，`review-ok` 会崩在
            # "No Neo4j connection has been configured"（真机实测踩过）。
            models.connect()
            upsert_relation = builder.upsert_relation
        ok = upsert_relation(
            target_record["source"],
            target_record["target"],
            relation_type=target_record.get("relation") or DEFAULT_RELATION_TYPE,
            confidence=as_confidence(target_record.get("confidence")) or 0.0,
        )
        # upsert_relation 在端点缺失时返回 False（跳过），如实记进结果里
        target_record["applied"] = bool(ok)
        target_record["status"] = "approved" if ok else "approved_skipped"
    else:
        target_record["applied"] = False
        target_record["status"] = "rejected"
    target_record["decided_at"] = _now()

    _write_queue(records, path)
    return target_record


def run_completion(limit: int = 20, dry_run: bool = False, auto: float = AUTO_LINK_CONFIDENCE,
                   review: float = REVIEW_LINK_CONFIDENCE, path: Path | str | None = REVIEW_QUEUE_PATH,
                   upsert_relation=None, call_llm=None) -> dict:
    """跑一轮补全：扫孤点 → 挑候选 → 提议 → 分流 → 落库/入队。

    `dry_run=True` 时**什么都不写**（既不落库也不入队），只在返回值里给出"会做什么"。
    连补全这种离线任务也要先能空跑看清规模，再决定要不要动图。
    `path=None` 表示本轮不写审核队列（只落库 + 报告）。
    """
    if upsert_relation is None:
        upsert_relation = builder.upsert_relation

    nodes = isolated_nodes(limit=limit)
    entities = all_entities()
    result = {
        "scanned": len(nodes),
        "no_candidate": 0,
        "auto": 0,
        "review": 0,
        "dropped": 0,
        "applied": [],
        "would_apply": [],
        "enqueued": [],
        "dropped_items": [],
        "dry_run": dry_run,
    }

    for node in nodes:
        candidates = candidate_pool(node, entities)
        if not candidates:
            result["no_candidate"] += 1
            continue
        proposals = propose_links(node, candidates, call_llm=call_llm)
        buckets = classify_proposals(proposals, auto=auto, review=review)
        source = str(node.get("name") or "")

        for item in buckets["auto"]:
            payload = {
                "source": source,
                "target": item["target"],
                "relation": item.get("relation") or DEFAULT_RELATION_TYPE,
                "confidence": item["confidence"],
                "reason": item.get("reason") or "",
            }
            if dry_run:
                result["would_apply"].append(payload)
                continue
            ok = upsert_relation(
                source,
                payload["target"],
                relation_type=payload["relation"],
                confidence=payload["confidence"] or 0.0,
            )
            result["applied"].append({**payload, "ok": bool(ok)})
            if not ok:
                logger.warning(f"[补全] 落库被跳过（端点缺失？）: {source} -> {payload['target']}")

        if buckets["review"] and not dry_run and path is not None:
            records = enqueue_review(
                [{"source": source, **item} for item in buckets["review"]], path=path
            )
            result["enqueued"].extend(records)
        elif buckets["review"] and dry_run:
            result["enqueued"].extend(
                [{"source": source, "dry_run": True, **item} for item in buckets["review"]]
            )

        result["auto"] += len(buckets["auto"])
        result["review"] += len(buckets["review"])
        result["dropped"] += len(buckets["dropped"])
        result["dropped_items"].extend([{"source": source, **item} for item in buckets["dropped"]])

    logger.info(
        f"[补全] 本轮结束 | 孤点 {result['scanned']}（无候选 {result['no_candidate']}）| "
        f"自动落库 {result['auto']} | 待审核 {result['review']} | 分数不足丢弃 {result['dropped']}"
        f"{' | DRY-RUN 未写图' if dry_run else ''}"
    )
    return result


def describe_queue(path: Path | str = REVIEW_QUEUE_PATH) -> dict:
    """队列概况（命令行 `review-list` 用）：各状态计数 + pending 明细。"""
    records = load_review_queue(path=path)
    counts: dict[str, int] = {}
    for record in records:
        counts[record.get("status") or "unknown"] = counts.get(record.get("status") or "unknown", 0) + 1
    return {
        "path": str(path),
        "total": len(records),
        "counts": counts,
        "pending": [record for record in records if record.get("status") == "pending"],
    }


if __name__ == "__main__":
    # 离线自检：不连库、不调模型，只验证"分流 + 队列 + 裁决"这条闭环能跑通
    print("=== 自检 1：分流 ===")
    demo = [
        {"target": "出差费用", "relation": "同义", "confidence": 0.93},
        {"target": "费用报销", "relation": "相关", "confidence": 0.61},
        {"target": "随机实体", "relation": "相关", "confidence": 0.12},
        {"target": "没给分", "relation": "相关"},
    ]
    buckets = classify_proposals(demo)
    print(f"  auto={len(buckets['auto'])} review={len(buckets['review'])} dropped={len(buckets['dropped'])}")
    assert len(buckets["auto"]) == 1 and len(buckets["review"]) == 2 and len(buckets["dropped"]) == 1

    print("=== 自检 2：队列往返 + 裁决 ===")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        queue_path = Path(tmp) / "queue.jsonl"
        applied = []
        records = enqueue_review(
            [{"source": "差旅费", "target": "出差费用", "relation": "同义", "confidence": 0.61}],
            path=queue_path,
        )
        assert describe_queue(queue_path)["counts"] == {"pending": 1}
        apply_review(
            records[0]["id"],
            approve=True,
            path=queue_path,
            upsert_relation=lambda *a, **kw: applied.append((a, kw)) or True,
        )
        assert describe_queue(queue_path)["counts"] == {"approved": 1}
        assert applied and applied[0][0][:2] == ("差旅费", "出差费用")
        print(f"  队列路径 {queue_path.name}：pending → approved，落库调用 {len(applied)} 次")

    print("=== 自检 3：候选池排序（同类型优先 + 余弦降序）===")
    pool = candidate_pool(
        {"name": "差旅费", "entity_type": "费用类型", "embedding": [1.0, 0.0]},
        [
            {"name": "张三", "entity_type": "人物", "embedding": [1.0, 0.0]},
            {"name": "出差费用", "entity_type": "费用类型", "embedding": [0.9, 0.1]},
            {"name": "住宿费", "entity_type": "费用类型", "embedding": [0.1, 0.9]},
        ],
        top_k=2,
    )
    print("  " + " / ".join(item["name"] for item in pool))
    assert [item["name"] for item in pool] == ["出差费用", "住宿费"]

    print("=== 自检 4：模型输出非法 JSON 时不落库、只告警 ===")
    assert propose_links({"name": "孤点"}, [{"name": "对照"}], call_llm=lambda **_: "抱歉，我无法回答") == []
    print("  OK")

    print(f"全部自检通过（自动落库阈值 {AUTO_LINK_CONFIDENCE}，审核阈值 {REVIEW_LINK_CONFIDENCE}）")

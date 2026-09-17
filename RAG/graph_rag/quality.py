"""图谱健康度指标（T6）：孤立节点率 / 重复节点率 / 链接准确率。

课案口径（优化篇「孤立节点的处理」）：
    用孤立节点率（孤点数 / 总节点数）监控图谱健康度，突增说明抽取或匹配环节退化；
    评估侧要新增链接准确率（该链接的有没有连对）与重复节点率（该新建的有没有误建重复节点）。

三类指标按**是否需要人工标注**分成两组，别混用：

| 函数 | 需要标注 | 用途 |
|---|---|---|
| `isolated_node_rate` | 否 | 随时可算的健康度监控（课案那个"突增"就靠它） |
| `normalized_duplicate_rate` | 否 | 重复节点的**代理**监控（只能抓写法差异，见函数说明） |
| `link_decision_metrics` | 是 | 在标注集上核对链接决策：链接准确率 + 误建重复节点率 |

为什么全是**纯函数**（进出都是普通 dict/list，不碰 Neo4j、不调模型）：
指标本身也得可验证。只有"连着真库跑一次"这一条路的话，读数无法复算、也无法用构造数据证伪
（与 `evaluation/threshold.py` 同一取舍）；真库采集在 `health.py`。
"""

import re
import unicodedata

# 归一化时要抹掉的字符：空白 + 常见中英文标点/连接符/括号。
# 只抹"不会改变实体身份"的那些；汉字、字母、数字保留。
_PUNCT_RE = re.compile(r"[\s\-_/\\.,，。、·・:：;；!！?？()（）\[\]【】{}｛｝\"'“”‘’<>《》]+")

# 孤立节点率 / 重复节点率相对上次报告涨多少就报警（绝对百分点）。
# 取 5 个百分点：51 个节点的图里多 1 个孤点是 +2%，属于正常波动，报出来只会训练人忽略告警。
ALERT_RATE_DELTA = 0.05


def normalize_entity_name(name) -> str:
    """归一化实体名：全角→半角（NFKC）、去空白与常见标点、统一小写。

    ⚠ 只用于**发现疑似重复**，绝不用作入库键 —— 入库仍用原始 `name`
    （`models.Entity.name` 是唯一索引）。若拿归一化名当键，"（张三）"和"张三"
    会被悄悄合成一个节点且原始写法丢失，那是不可逆的数据损坏。
    """
    text = unicodedata.normalize("NFKC", str(name if name is not None else ""))
    return _PUNCT_RE.sub("", text).casefold()


def isolated_node_rate(nodes: list[dict], relationships: list[dict]) -> dict:
    """孤立节点率 = 与其他实体**没有任何关系**的节点数 / 节点总数。

    `nodes`: `[{"name": ...}]`；`relationships`: `[{"source": ..., "target": ...}]`。

    自环（`source == target`）**不计入连接**：`a -> a` 只说明"它和自己有关系"，
    它仍然没和图上任何别的实体发生联系，按课案"与任何实体都没有链接"的口径应算孤点。

    ⚠ 这条与 Cypher 侧 `NOT (e)-[:RELATES_TO]-()` 的口径**不同**（那句会把自环当成连接），
    所以真库上的孤点必须按"节点 + 关系端点"两个查询在 Python 里算，见 `health.collect`。
    """
    names = [str(n.get("name") or "") for n in nodes]
    connected: set[str] = set()
    for rel in relationships:
        source = str(rel.get("source") or "")
        target = str(rel.get("target") or "")
        # 端点缺失（导入中途的半截关系）不算连接；自环也不算
        if not source or not target or source == target:
            continue
        connected.add(source)
        connected.add(target)

    isolated = [name for name in names if name not in connected]
    total = len(names)
    return {
        "total": total,
        "isolated_count": len(isolated),
        "rate": (len(isolated) / total) if total else 0.0,
        # 名单一起给出来：只有一个比率的话，排查时还得再连一次库
        "isolated": isolated,
    }


def normalized_duplicate_rate(names: list[str]) -> dict:
    """归一化重名率（监控用）= 落在"归一化同名组"里的节点数 / 节点总数。

    ⚠ 这是**代理指标**，能抓到的只有"写法差异造成的假重复"：大小写、空白、全半角、标点。
    "差旅费 vs 出差费用"这类**近义**重复它抓不到 —— 那要靠 `link_decision_metrics` 的标注集
    （或者上游做 alias 消歧，本项目尚未实现 alias，见 README 台账）。
    别把这个数字当成"重复节点率"的全部；README 里也是这么写的。

    归一化后为空串的名字（如"（）"）不进组：它们是无名节点，彼此之间不构成"同一实体"。
    """
    groups: dict[str, list[str]] = {}
    for name in names:
        key = normalize_entity_name(name)
        if not key:
            continue
        groups.setdefault(key, []).append(str(name))

    duplicate_groups = {key: value for key, value in groups.items() if len(value) > 1}
    duplicate_count = sum(len(value) for value in duplicate_groups.values())
    total = len(names)
    return {
        "total": total,
        "duplicate_count": duplicate_count,
        "rate": (duplicate_count / total) if total else 0.0,
        # 组明细直接给出：监控报警之后要能一眼看出是哪几个写法在打架
        "groups": duplicate_groups,
    }


def _index_by_key(samples: list[dict]) -> dict:
    """按 `(归一化名称, 实体类型)` 建索引 —— 同名不同类型的实体是两回事。

    `builder.match_entity` 的候选也限定同 `entity_type`（"张三"（人物）和"张三"（公司）
    向量上可能很像，但业务上不是一个东西），这里保持同一口径。
    """
    index = {}
    for sample in samples or []:
        key = (normalize_entity_name(sample.get("name")), str(sample.get("entity_type") or "").strip())
        index[key] = sample
    return index


def link_decision_metrics(decisions: list[dict], labels: list[dict]) -> dict:
    """在标注集上核对链接决策，给出**链接准确率**与**误建重复节点率**。

    `labels` 是人工确认的答案，`decisions` 是系统判定，两边同构：
    `[{"name": str, "entity_type": str | None, "matched": str | None}]`，
    `matched=None` 表示"该新建 / 判定为新建"，非空表示"连到哪个已有实体"。
    以 `(归一化 name, entity_type)` 配对 —— 对不上的样本进 `unresolved`，**不静默丢**（分母里如实带着）。

    输出里三个错误分类的读法（第二行最容易看反，课案原文很短）：
    - 课案"链接准确率" = `link_accuracy`：标"该连"的样本里，决策**连对了同一个目标**的占比。
    - 课案"重复节点率（该新建的有没有误建重复节点）" = `duplicate_rate`：
      标"该连"的样本系统却**新建**了节点 ⇒ 图里就多出一个重复节点。
      它的分母是"标该连"的样本数（与 `link_accuracy` 同分母，两者互补）。
    - `false_link`：标"该新建"却连到已别的实体 ⇒ 独立实体被吞掉（图谱少节点）。
    - `target_mismatch`：连了但连错对象 ⇒ 最隐蔽的一种，既不算新建也不算连对。

    没有任何"该连"的样本时 `link_accuracy` / `duplicate_rate` 是 `None` 而不是 0.0
    —— 0.0 会被读成"全错"，而实际是"没测"。
    """
    label_index = _index_by_key(labels)
    decision_index = _index_by_key(decisions)

    unresolved = []
    for key in label_index.keys() - decision_index.keys():
        unresolved.append({"name": key[0], "entity_type": key[1], "side": "label"})
    for key in decision_index.keys() - label_index.keys():
        unresolved.append({"name": key[0], "entity_type": key[1], "side": "decision"})

    pairs = [(key, label_index[key], decision_index[key]) for key in label_index.keys() & decision_index.keys()]

    false_create, target_mismatch, false_link = [], [], []
    link_total = link_correct = 0
    create_total = create_correct = 0
    for key, label, decision in pairs:
        expected = label.get("matched")
        decided = decision.get("matched")
        record = {
            "name": label.get("name"),
            "entity_type": label.get("entity_type") or "",
            "expected": expected,
            "decided": decided,
        }
        if expected is None:
            create_total += 1
            if decided is None:
                create_correct += 1
            else:
                false_link.append(record)
            continue
        link_total += 1
        if decided is None:
            false_create.append(record)
        elif normalize_entity_name(decided) == normalize_entity_name(expected):
            link_correct += 1
        else:
            target_mismatch.append(record)

    return {
        "resolved": len(pairs),
        "unresolved": sorted(unresolved, key=lambda item: (item["side"], item["name"])),
        "link_total": link_total,
        "link_accuracy": (link_correct / link_total) if link_total else None,
        "duplicate_rate": (len(false_create) / link_total) if link_total else None,
        "create_total": create_total,
        "create_accuracy": (create_correct / create_total) if create_total else None,
        "false_create": false_create,
        "false_link": false_link,
        "target_mismatch": target_mismatch,
    }


def community_coverage(nodes: list[dict]) -> dict:
    """社区覆盖：`community_id` 还没分配的实体（缺失或 `< 0`）有几个、是哪些。

    ⚠ **不能写成 `node.get("community_id") or -1`**：`0` 是合法的 Louvain 社区编号，
    会被 `or` 当假值吞掉、算成"未分配"。实测就错过一次 —— 51 个实体的图报出"7 个未分配"，
    而真值是 0（对照只读探针 `WHERE e.community_id IS NULL OR e.community_id < 0` 的数）。
    判据必须是显式的 `is None` 判断 + 与 0 比较。

    （`models.Entity.community_id` 的默认值是 -1，即"新建但还没分区"，语义见该字段的注释。）
    """
    unassigned = [
        node.get("name")
        for node in nodes or []
        if node.get("community_id") is None or node.get("community_id") < 0
    ]
    total = len(nodes or [])
    return {
        "total": total,
        "unassigned_count": len(unassigned),
        "unassigned": unassigned,
        "rate": (len(unassigned) / total) if total else 0.0,
    }


def compare_with_previous(previous: dict | None, current: dict, alert_delta: float = ALERT_RATE_DELTA) -> dict:
    """把本次健康度与上次报告对比，孤点率/重复率**突增**就报警（课案：突增说明抽取或匹配退化）。

    首次运行（没有上次报告）不算异常：`deltas` 空、`alerts` 空。
    """
    if not previous:
        return {"deltas": {}, "alerts": [], "compared_with": None}

    previous_metrics = previous.get("metrics") or {}
    current_metrics = current.get("metrics") or {}
    deltas, alerts = {}, []
    for key, label in (("isolated", "孤立节点率"), ("duplicate", "重复节点率")):
        before = ((previous_metrics.get(key) or {}).get("rate"))
        after = ((current_metrics.get(key) or {}).get("rate"))
        if before is None or after is None:
            continue
        deltas[key] = round(after - before, 6)
        if after - before >= alert_delta:
            alerts.append(
                f"{label}突增：{before:.2%} → {after:.2%}（+{after - before:.2%}）"
                "，优先查抽取/匹配环节（别名漏匹配会让本该连上的实体变成孤点）"
            )

    return {"deltas": deltas, "alerts": alerts, "compared_with": previous.get("generated_at")}

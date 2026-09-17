"""GraphRAG 建图：LLM 实体/关系抽取 → 入图 → 社区检测 → 社区摘要 → 增量入库。

设计要点（为了"纯逻辑离线可测"）：
- 所有外部依赖都收敛成模块级名字，测试里整体替换即可：
    `call_llm`（LLM） / `embed_query`（向量） / `run_cypher`、`Entity`、`Community`（Neo4j）
- 真正有分支判断的逻辑全部抽成纯函数：
    merge_confidence / merge_doc_ids / merge_description / merged_entity_state /
    merge_relation_props / normalize_extraction / load_json_object /
    build_networkx_graph / group_entities_by_community / build_community_material /
    parse_match_choice
  这样合并语义、JSON 容错、社区分组建图都能不连数据库验证。

调用顺序（`script/build_finance_graph.py::main` 就是按这个顺序串起来的，有强依赖）：
    1. extract_entities()            文本 → 归一化后的实体/关系
    2. create_graph() 全量 或 ingest_document() 增量   写 Entity / RELATES_TO
    3. detect_communities_louvain()  读 RELATES_TO 的 weight → 写 Entity.community_id
    4. generate_community_summaries() 读 community_id + 社区内关系 → 写 Community
    5. 检索（retriever）              读社区摘要向量 / 实体向量 / 多跳子图
    顺序不能换：3 依赖 2 写进去的边，4 依赖 3 写进去的社区号，5 依赖 4 写进去的摘要向量。

成本画像（一轮全量建图的网络调用次数，决定了"为什么它要跑很久"）：
    文档数 × 1 次 LLM（抽取）
  + 1 次 Louvain（本地 CPU，无网络）
  + 社区数 × (1 次 LLM + 1 次 embedding)（摘要）
  + 增量模式的**消歧**还会额外产生：未精确命中的实体 × (1 次 embedding + 可能 1 次 LLM)
"""

import json
import time

import community
import networkx as nx
from openai import OpenAI

from config import settings
from core.logger import logger
from graph_rag.models import Community, Entity, connect, run_cypher
from retrieval.embedding import embed_query

# ============================================================
# 一、LLM 出口
# ============================================================
# 进程级单例：惰性构造，一个进程只建一个 OpenAI 客户端（连接池复用）。
# 离线单测不替换它，而是整体替换更外层的 `call_llm`（见 test_graph_builder.py），
# 这样连"重试 / 空正文判定"这些分支也能一起被 fake 掉。
_llm_client: OpenAI | None = None

# 单次 LLM 调用的最大尝试次数。建图一轮几分钟、几十次调用，
# 网关偶发断连（RemoteProtocolError: Server disconnected without sending a response）
# 不重试就会让整轮白跑。
LLM_MAX_ATTEMPTS = 3

EXTRACT_SYSTEM = "你是一个专业的实体和关系提取助手。"
EXTRACT_USER_TMPL = """请从下面的文本中提取实体和关系，只返回 JSON，不要任何解释。

要求：
1. entities 是实体数组，每项包含 name（名称）、type（类型）、description（简短描述）。
2. relations 是关系数组，每项包含 source（源实体名）、target（目标实体名）、relation（关系名称）。
3. source/target 必须是 entities 里出现过的名称。

返回格式：
{{"entities": [{{"name": "", "type": "", "description": ""}}], "relations": [{{"source": "", "target": "", "relation": ""}}]}}

文本：
{text}
"""

SUMMARY_SYSTEM = "你是一个专业的文本摘要助手。"

MATCH_ENTITY_SYSTEM = "你是一个实体消歧助手。"
MATCH_ENTITY_USER_TMPL = """实体「{name}」是否与列表中某个实体指同一对象？
候选：{candidates}
只返回候选中的名称；都不相关则返回 NONE。"""

# 向量召回候选的相似度下限：低于它就不值得再问 LLM 了
MATCH_SCORE_THRESHOLD = 0.6
# 向量召回的候选个数
MATCH_CANDIDATE_TOP_K = 10


def get_llm_client() -> OpenAI:
    """惰性构造 LLM 客户端（导入本模块不会发起任何请求）。

    必须显式设超时：SDK 默认 600 秒 × 重试 3 次，网关断连时建图任务会假死
    （实测卡过 46 分钟没有任何输出）。
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
            timeout=settings.llm.timeout,
        )
    return _llm_client


def call_llm(system: str, user: str, temperature: float = 0.1, json_mode: bool = False) -> str:
    """统一的 LLM 调用出口（OpenAI 兼容接口），带重试。

    json_mode=True 时使用 `response_format={"type": "json_object"}`，
    温度按课案对不同任务分别取 0.1 / 0.3 / 0。

    网络类异常与"返回空内容"都算这次调用废了，重试到 LLM_MAX_ATTEMPTS 次仍失败才抛错
    （空串不能当结果往下传，否则会变成"这篇票据没抽出实体"的静默丢数据）。

    重试是**线性**退避（第 1 次失败等 2s、第 2 次等 4s，总共最多等 6s），刻意不用指数退避：
    网关断连多数是瞬时抖动，一轮建图要跑几十次调用，等待时间不能放大。
    `except Exception` 是刻意的宽口径：断连、超时、限流、`choices` 为空数组导致的
    IndexError 都属于"这次调用废了"，统一重试比逐个枚举异常类型更稳。

    ⚠ 已知口径分叉（与 llm/chat.py 不一致，见"发现的问题"）：
    主链路 `llm/chat.py` 每次都下发 `max_tokens=settings.llm.max_tokens` 与
    `extra_body={"enable_thinking": ...}`，而这里的 `create()` **两个都没传**，
    走的是网关默认值。本项目 `LLM_ENABLE_THINKING=true` + `deepseek-flash`（思考模型），
    而抽取/判同这类调用只要一小段 JSON —— 预算被思考吃光的风险是存在的；
    目前靠下面"空正文算失败 + 重试 3 次"兜底，不会静默丢数据，但会白跑几秒。
    """
    kwargs: dict = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    last_exc: Exception | None = None
    for attempt in range(LLM_MAX_ATTEMPTS):
        try:
            resp = get_llm_client().chat.completions.create(
                model=settings.llm.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                **kwargs,
            )
            content = resp.choices[0].message.content
        except Exception as exc:  # noqa: BLE001 网络/网关异常统一重试
            last_exc = exc
        else:
            if content is not None and str(content).strip():
                return str(content)
            last_exc = ValueError("LLM 返回内容为空")

        if attempt < LLM_MAX_ATTEMPTS - 1:
            logger.warning(f"[GraphRAG] LLM 调用失败(第 {attempt + 1}/{LLM_MAX_ATTEMPTS} 次): {last_exc},重试")
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"LLM 调用失败(已重试 {LLM_MAX_ATTEMPTS} 次): {last_exc}")


# ============================================================
# 二、JSON 解析与抽取结果归一化（纯逻辑）
# ============================================================
def load_json_object(raw) -> dict:
    """把 LLM 返回的文本解析成 dict。

    容错点：LLM 经常把 JSON 包在 ```json 代码块里，或者带首尾空白。
    顶层必须是对象——数组/字符串拿不到 entities/relations 字段，直接报错比默默返回空更好。

    已知边界（有意保留，不修）：只剥"紧贴首尾"的那一层代码围栏。
    如果模型先寒暄一句再给 JSON（"好的，以下是结果：{...}"），`json.loads` 会直接失败。
    抽取提示词里已明确要求"只返回 JSON，不要任何解释"，所以这里保持严格：
    宁可报错重试，也不要拿正则去"猜" JSON 的边界 —— 猜错会把正文当 JSON
    静默解析成"这篇文章没有实体"（最隐蔽的丢数据方式）。
    """
    if raw is None:
        raise ValueError("LLM 返回为空，无法解析 JSON")
    text = str(raw).strip()
    if not text:
        raise ValueError("LLM 返回为空，无法解析 JSON")

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM 返回不是合法 JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"LLM 返回的 JSON 顶层必须是对象，实际是 {type(data).__name__}")
    return data


def normalize_extraction(data) -> dict:
    """归一化抽取结果：统一字段名、去空白、丢坏数据、按名称去重。

    LLM 的返回永远不能全信：可能有无名实体、缺端点的关系、重复实体。
    这里全部收敛成内部统一的 `{name, entity_type, description}` / `{source, target, relation}`。

    ⚠ 去重的副作用：实体按名称去重，**只保留第一次出现的描述**，
    后一个同名实体的描述被直接丢弃（不是拼接）—— 一次抽取内同名实体通常是模型的重复输出，
    丢弃是合理的；要合并描述请走 `merge_description`（入库时才做）。
    关系**不做去重**：同一对端点可能进来两次，重复留给 `upsert_relation` 按 weight 合并，
    这样"同一文档里被抽到两次"也会如实计入证据。
    """
    if not isinstance(data, dict):
        return {"entities": [], "relations": []}

    entities: list[dict] = []
    seen: set[str] = set()
    for item in data.get("entities") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        entity_type = str(item.get("type") or item.get("entity_type") or "").strip()
        entities.append(
            {
                "name": name,
                "entity_type": entity_type or "UNKNOWN",
                "description": str(item.get("description") or "").strip(),
            }
        )

    relations: list[dict] = []
    for item in data.get("relations") or []:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip()
        target = str(item.get("target") or "").strip()
        if not source or not target:
            continue
        relation = str(item.get("relation") or item.get("relation_type") or "").strip()
        relations.append(
            {"source": source, "target": target, "relation": relation or "RELATES_TO"}
        )

    return {"entities": entities, "relations": relations}


def extract_entities(text) -> dict:
    """调 LLM 从一段文本里抽实体与关系，返回归一化后的 dict。

    空文本直接短路：既省一次调用，也避免把空串喂给模型产生幻觉实体。
    """
    if not text or not str(text).strip():
        return {"entities": [], "relations": []}
    raw = call_llm(
        EXTRACT_SYSTEM,
        EXTRACT_USER_TMPL.format(text=str(text).strip()),
        temperature=0.1,
        json_mode=True,
    )
    return normalize_extraction(load_json_object(raw))


# ============================================================
# 三、合并规则（纯逻辑）
# ============================================================
def merge_confidence(old_confidence, old_weight, new_confidence, digits: int = 4) -> float:
    """关系置信度的加权平均：`(旧confidence*旧weight + 新confidence) / (旧weight+1)`。

    digits 收口浮点噪声，避免 0.8500000000000001 这种值写进图里。
    权重语义 = 这条关系被观察到几次（`upsert_relation` 每命中一次 +1），
    所以这是把新样本并进**历史平均**，而不是两次观测的简单平均：
    观测次数越多，单次新证据对结果的影响越小（对抗单篇票据的偶发误抽）。
    """
    old = float(old_confidence if old_confidence is not None else 1.0)
    weight = int(old_weight) if old_weight else 1
    new = float(new_confidence if new_confidence is not None else 1.0)
    return round((old * weight + new) / (weight + 1), digits)


def merge_doc_ids(existing, incoming) -> list[str]:
    """doc_ids 去重追加，保持出现顺序

    顺序稳定是有意义的：它同时也是"证据链"的展示顺序（报告里按入库先后列出）。
    入参必须是**集合型**（list/tuple/set）；传 str 会被按字符逐个拆开 ——
    调用方一律传 list，这里不做额外防御（`_as_list` 负责把单值包成列表）。
    """
    merged: list[str] = []
    for source in (existing or [], incoming or []):
        for value in source:
            if value is None:
                continue
            text = str(value).strip()
            if text and text not in merged:
                merged.append(text)
    return merged


def merge_description(old, new) -> str:
    """描述合并：新描述为空/已包含在旧描述里就保持原样，否则用「；」追加

    `new_text in old_text` 是子串判重（不是语义判重），用来挡住同一段话被反复追加。
    描述会**无上限增长**：同一实体被 N 篇票据提到就有 N 段。
    而实体向量只在新建节点时算一次 ⇒ 描述变长**不会**让向量跟着更新（已知偏差，
    见 upsert_entity）—— 这也是这里敢"无限追加"的原因：它不会带来重复的 embedding 费用。
    """
    old_text = str(old or "").strip()
    new_text = str(new or "").strip()
    if not new_text:
        return old_text
    if not old_text:
        return new_text
    if new_text in old_text:
        return old_text
    return f"{old_text}；{new_text}"


def merge_relation_props(
    rel, confidence=1.0, doc_id=None, doc_ids=None, digits: int = 4
) -> dict:
    """算出一条已存在关系在再次被观察到之后的属性（不改动入参）。

    纯函数约定：不碰数据库、不改入参 —— 所以单测可以直接喂假对象（duck typing），
    也因此这里用 `getattr(rel, ..., 默认值)` 读取属性：rel 可能是 neomodel 的
    StructuredRel，也可能是测试里的一个 SimpleNamespace。
    `_as_list(doc_ids)` 返回的是**新列表**，所以下面 `incoming.append(doc_id)`
    不会污染调用方传进来的那个 list。
    """
    incoming = _as_list(doc_ids)
    if doc_id is not None:
        incoming.append(doc_id)
    return {
        "confidence": merge_confidence(
            getattr(rel, "confidence", 1.0), getattr(rel, "weight", 1), confidence, digits
        ),
        "weight": (int(getattr(rel, "weight", 1) or 1)) + 1,
        "doc_ids": merge_doc_ids(getattr(rel, "doc_ids", None), incoming),
    }


def merged_entity_state(node, description, doc_id=None, doc_ids=None) -> dict:
    """算出一个已存在实体在再次被抽取到之后的属性（不改动入参）。

    与 merge_relation_props 同为纯函数（同样的 duck typing 与"不改入参"约定），
    返回的是**整份新状态**而不是增量：调用方拿到后自己赋给 node 再 save()，
    这样"算出什么"与"写什么"分开，合并规则可以在不连数据库的情况下被钉住。
    `mentions` 用 `int(... or 0) + 1`：老图里可能存在 mentions 为 NULL/0 的节点，
    也要能正确累加，而不是抛 TypeError。
    """
    incoming = _as_list(doc_ids)
    if doc_id is not None:
        incoming.append(doc_id)
    return {
        "description": merge_description(getattr(node, "description", ""), description),
        "doc_ids": merge_doc_ids(getattr(node, "doc_ids", None), incoming),
        "mentions": int(getattr(node, "mentions", 1) or 0) + 1,
    }


# ============================================================
# 四、文本渲染（纯逻辑，builder 与 retriever 共用）
# ============================================================
def format_entity_lines(entities) -> str:
    """实体清单渲染成 `- 名称 (类型): 描述`

    被 retriever 复用（`from graph_rag.builder import format_entity_lines`），
    所以改这里的渲染格式会**同时**改变两处 LLM 提示词：社区摘要的用户提示词
    （build_community_material）和最终 RAG 提示词（retriever.build_rag_prompt）。
    这也是它留在这里而不是放进 retriever 的原因：两处必须同构，放一个地方改一次。
    """
    lines = []
    for item in entities or []:
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        entity_type = str(item.get("entity_type") or item.get("type") or "UNKNOWN").strip()
        description = str(item.get("description") or "")
        lines.append(f"- {name} ({entity_type}): {description}")
    return "\n".join(lines)


def format_relation_lines(relations) -> str:
    """关系清单渲染成 `- 源 关系 目标`

    与 format_entity_lines 同理被 retriever 复用；兼容 `relation` / `relation_type`
    两种字段名（LLM 侧叫 relation，图侧叫 relation_type）。
    """
    lines = []
    for item in relations or []:
        source = str(item.get("source") or "").strip()
        target = str(item.get("target") or "").strip()
        relation = str(item.get("relation") or item.get("relation_type") or "").strip()
        if not source or not target:
            continue
        lines.append(f"- {source} {relation} {target}")
    return "\n".join(lines)


def build_community_material(community_id, entities, relations) -> str:
    """拼出交给 LLM 写社区摘要的用户提示词：社区 ID + 实体清单 + 社区内关系清单

    两个数据源都来自 Cypher 侧已经排好序的行（实体按 community_id, name；关系按引擎顺序），
    加上 dict 保序，所以同一张图重跑会得到**逐字相同**的提示词（可复现）。
    "100~200 字、不要罗列清单"是写在提示词里的约束，**没有**后处理截断：
    模型偶尔超出长度就照原样入库，长一点的摘要不影响检索。
    """
    entity_lines = format_entity_lines(entities) or "（无）"
    relation_lines = format_relation_lines(relations) or "（无）"
    return (
        f"社区 ID：{community_id}\n\n"
        f"实体清单：\n{entity_lines}\n\n"
        f"社区内关系清单：\n{relation_lines}\n\n"
        "请用 100~200 字总结这个社区主要是关于什么的，不要罗列清单。"
    )


def _as_list(value) -> list:
    """把 None / 单值 / 列表统一成列表

    返回的一定是**新列表**（即使是 list 入参也会重新构造），所以调用方拿到后
    可以放心 `append` —— merge_*_props 系列就是靠这一点做到"不改入参"的。
    顺手过滤 None：图谱里的 doc_ids 常来自 `getattr(rel, "doc_ids", None)`，
    历史节点上可能是 None 而不是空列表。
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [v for v in value if v is not None]
    return [value]


def _entity_fields(item) -> dict:
    """从外部传入的实体条目里取出规范字段（兼容 LLM 的 `type` 与内部的 `entity_type`）"""
    item = item or {}
    entity_type = str(item.get("entity_type") or item.get("type") or "").strip()
    return {
        "name": str(item.get("name") or "").strip(),
        "entity_type": entity_type or "UNKNOWN",
        "description": str(item.get("description") or "").strip(),
        "doc_ids": _as_list(item.get("doc_ids")),
    }


def _relation_fields(item) -> dict:
    """从外部传入的关系条目里取出规范字段（兼容 `relation` / `relation_type`）"""
    item = item or {}
    try:
        confidence = float(item.get("confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 1.0
    relation_type = str(item.get("relation_type") or item.get("relation") or "").strip()
    return {
        "source": str(item.get("source") or "").strip(),
        "target": str(item.get("target") or "").strip(),
        "relation_type": relation_type or "RELATES_TO",
        "confidence": confidence,
        "doc_ids": _as_list(item.get("doc_ids")),
    }


# ============================================================
# 五、节点 / 关系入库
# ============================================================
def upsert_entity(
    name, entity_type="UNKNOWN", description="", doc_id=None, doc_ids=None
) -> Entity:
    """按名称 upsert 实体：已存在则合并描述/累加 mentions/doc_ids，否则新建并写向量。

    实体向量 = `embed_query("{name} {description}")`，只在新节点上算——已经存在的节点
    再次抽取到时不重算向量（描述变化不值得为每条票据重复付费调 embedding）。

    ⚠ 由此带来的已知偏差（有意保留，别当 bug 修）：
    1. 命中已有节点时**不更新 entity_type** —— 首次写入的类型即最终类型
       （与 upsert_relation 的 relation_type"首次写入即生效"同一条规则）；
    2. 命中已有节点时**不重算 embedding** —— 向量永远停留在"第一次见到这个名字时"
       的名称+描述文本上，描述后来长了一堆也进不了向量，
       于是 `builder.match_entity` 的消歧召回是拿新描述去比旧向量。
       好处是增量入库不会因为老实体被反复提到而重复付 embedding 费用。
    """
    node = Entity.nodes.get_or_none(name=name)
    if node is not None:
        state = merged_entity_state(node, description, doc_id=doc_id, doc_ids=doc_ids)
        node.description = state["description"]
        node.doc_ids = state["doc_ids"]
        node.mentions = state["mentions"]
        node.save()
        return node

    node = Entity(
        name=name,
        entity_type=entity_type or "UNKNOWN",
        description=description or "",
        doc_ids=merge_doc_ids(doc_ids, [doc_id] if doc_id is not None else []),
        mentions=1,
        embedding=embed_query(f"{name} {description or ''}".strip()),
    )
    node.save()
    return node


def upsert_relation(
    source_name,
    target_name,
    relation_type="RELATES_TO",
    confidence=1.0,
    doc_id=None,
    doc_ids=None,
) -> bool:
    """按「源-目标」upsert 关系。

    已存在：按历史权重做加权平均更新 confidence，weight +1，doc_ids 去重追加；
    不存在：新建，weight 从 1 开始。
    两端实体只要有一个不在图里就跳过（返回 False）——半截关系比没有关系更糟。
    relation_type 采用"首次写入即生效"：后续抽取到的同一条关系只累加证据，不改语义。

    为什么端点缺失只是"跳过 + 打 WARNING"而不是抛异常：
    批量建图时一篇票据的抽取结果里混进一个没建成功的端点很常见（比如实体名为空被丢），
    为它中断整轮建图不值得；但也不能静默 —— 所以留一条 WARNING。
    `weight` 从 1 开始，每次命中 +1，同时它也是 merge_confidence 的加权权重。
    """
    source = Entity.nodes.get_or_none(name=source_name)
    target = Entity.nodes.get_or_none(name=target_name)
    if source is None or target is None:
        logger.warning(
            f"[GraphRAG] 关系 {source_name} -{relation_type}-> {target_name} 的端点不在图中，跳过"
        )
        return False

    incoming = _as_list(doc_ids)
    if doc_id is not None:
        incoming.append(doc_id)

    rel = source.relates_to.relationship(target)
    if rel is None:
        # 唯一性到底谁保证的（值得写清楚，别被"Neo4j 保证同型关系只有一条"误导）：
        # Neo4j **本身允许**同一对节点之间存在多条同类型关系；这里不会产生平行边，
        # 是因为 neomodel 7 的 `connect()` 发的是 MERGE
        # （neomodel/sync_/relationship_manager.py: connect() 里拼 "MERGE"），
        # 再加上上面先用 `relationship()` 查过一遍。
        # 所以：手写 Cypher 建关系时不要假设唯一；用 ORM 的 connect() 才是 MERGE 语义。
        source.relates_to.connect(
            target,
            {
                "relation_type": relation_type or "RELATES_TO",
                "confidence": float(confidence if confidence is not None else 1.0),
                "weight": 1,
                "doc_ids": merge_doc_ids(None, incoming),
            },
        )
        return True

    # doc_id=None 是刻意的：函数开头已经把它 append 进 incoming 了，
    # 这里再传一次会把同一个 doc_id 算两遍（merge_doc_ids 会去重，但语义就错了）。
    merged = merge_relation_props(rel, confidence=confidence, doc_id=None, doc_ids=incoming)
    rel.confidence = merged["confidence"]
    rel.weight = merged["weight"]
    rel.doc_ids = merged["doc_ids"]
    rel.save()
    return True


# ⚠ 不带标签的 MATCH (n)：删的是**整张图的全部节点** —— Entity、Community 一起删，
#   将来任何别的标签也会被删掉，DETACH 负责先把关系删掉（Neo4j 不允许删还剩关系的节点）。
#   单事务执行，冒烟规模（几十~几百节点）没问题，图很大时要改成分批删，否则容易吃内存。
#   另一个同名常量在 script/build_finance_graph.py 里（那边做"先清空再入库"），
#   两处是**同一句话的两份真相**，改这里记得同步那边。
_CYPHER_CLEAR_GRAPH = "MATCH (n) DETACH DELETE n"


def create_graph(entities, relations) -> dict:
    """全量建图：清空整图后按传入的实体/关系重建。

    返回本次写入的条目数（重复实体只累加、不会多出一个节点）。
    注意这是**破坏性**操作，增量场景请用 ingest_document。

    返回值口径要看清：`entities` 数的是**输入的实体条目数**（不是图中节点数），
    输入里重复的名字会被 upsert 合并；`relations` 数的是**真正写成功的条数**
    （端点缺失被跳过的那些不计入）。想看图里真实规模请查库（见脚本的 graph_stats）。
    """
    connect()
    # 课案要求全量重建时先清空，保证同一批输入得到同一张图
    run_cypher(_CYPHER_CLEAR_GRAPH)
    logger.info("[GraphRAG] 已清空原有图谱，开始全量建图")

    entity_count = 0
    for item in entities or []:
        fields = _entity_fields(item)
        if not fields["name"]:
            logger.warning("[GraphRAG] 跳过无名实体")
            continue
        upsert_entity(
            fields["name"],
            fields["entity_type"],
            fields["description"],
            doc_ids=fields["doc_ids"],
        )
        entity_count += 1

    relation_count = 0
    for item in relations or []:
        fields = _relation_fields(item)
        if not fields["source"] or not fields["target"]:
            continue
        if upsert_relation(
            fields["source"],
            fields["target"],
            fields["relation_type"],
            confidence=fields["confidence"],
            doc_ids=fields["doc_ids"],
        ):
            relation_count += 1

    logger.info(f"[GraphRAG] 建图完成: 实体 {entity_count} 条 / 关系 {relation_count} 条")
    return {"entities": entity_count, "relations": relation_count}


# ============================================================
# 六、社区检测（Louvain）
# ============================================================
_CYPHER_ENTITY_EDGES = """
MATCH (a:Entity)
OPTIONAL MATCH (a)-[r:RELATES_TO]->(b:Entity)
RETURN a.name AS source, b.name AS target, r.weight AS weight
"""

_CYPHER_WRITE_COMMUNITY = """
UNWIND $rows AS row
MATCH (n:Entity {name: row.name})
SET n.community_id = row.community_id
"""


def build_networkx_graph(records) -> nx.Graph:
    """把 Cypher 查出来的边记录导成无向带权图。

    用 OPTIONAL MATCH 查出来的孤立实体（target 为 None）也要保留成节点，
    否则它们永远分不到社区，分层检索就查不到。
    同一对节点之间的多条关系（不同 relation_type）权重累加。

    权重取的是 `r.weight`（关系被观察到几次）而**不是** confidence ——
    Louvain 按"共现强度"聚类，一条置信度不高但反复出现的关系，仍然是强连接。
    反过来，两个实体之间只有一条 confidence=0.5 的关系时，它在社区划分里
    和 confidence=1.0 的一条**等价**（权重都是 1）；这是刻意的取舍：
    Louvain 只接受正权重，把 confidence 乘进去会让低置信关系的节点轻易变成孤点。
    """
    graph = nx.Graph()
    for row in records or []:
        source = row.get("source")
        if not source:
            continue
        graph.add_node(source)
        target = row.get("target")
        if not target:
            continue
        graph.add_node(target)
        weight = row.get("weight") or 1
        if graph.has_edge(source, target):
            graph[source][target]["weight"] += weight
        else:
            graph.add_edge(source, target, weight=weight)
    return graph


def detect_communities_louvain(resolution: float = 1.0) -> dict[str, int]:
    """用 Louvain 做社区检测，并把 community_id 写回 Entity。

    返回 `{实体名: 社区号}`。random_state 固定，保证同一张图每次跑出同样的社区划分
    （否则增量建图后社区 ID 会漂移，之前的社区摘要就对不上了）。

    ⚠ 三个必须知道的后果：
    1. 这是**全图重算**，不是课案设想的"只重算受影响社区"：任何一次图变化都会让
       所有社区的边界乃至编号重新分配，所以历史 Community 摘要不能按 id 复用，
       只能整批重算（generate_community_summaries 就是这么做的）。
       random_state=42 只保证"同一张图 → 同一份划分"，不保证"图变了编号还稳定"。
    2. 孤立实体（没有任何 RELATES_TO 的节点）会被 python-louvain **各自分到一个独立社区**
       （该库的既有行为；实测 3 个连通节点 + 2 个孤点 → 3 个社区）。
       而社区摘要是"每社区一次 LLM + 一次 embedding" ⇒ 孤点越多，
       摘要阶段的网络调用次数就越多（成本与噪音随孤点数近似线性上涨，
       且单实体社区的摘要没什么信息量）。
    3. 只写 `Entity.community_id`，**不清理**上一轮留下、这一轮已不存在的 Community 节点
       （已知缺陷，见 models.Community 的说明）。
    """
    connect()
    records = run_cypher(_CYPHER_ENTITY_EDGES)
    graph = build_networkx_graph(records)
    if graph.number_of_nodes() == 0:
        logger.warning("[GraphRAG] 图中没有实体，跳过社区检测")
        return {}

    # weight="weight"：用 _CYPHER_ENTITY_EDGES 查出来的关系权重（被观察次数）
    # resolution：Louvain 的分辨率参数，>1 倾向更多更小的社区、<1 倾向更少更大的社区，
    #             默认 1.0；脚本的 --resolution 直接透传到这里。
    # random_state=42：课案没带这个参数，是本项目补的 —— 不带它 louvain 每次跑出的
    #             社区划分（进而编号）都会变，历史摘要与社区 id 就对不上了。
    partition = community.best_partition(
        graph, weight="weight", resolution=resolution, random_state=42
    )
    rows = [
        {"name": name, "community_id": int(community_id)}
        for name, community_id in partition.items()
    ]
    run_cypher(_CYPHER_WRITE_COMMUNITY, {"rows": rows})
    logger.info(
        f"[GraphRAG] 社区检测完成: {graph.number_of_nodes()} 个实体 → {len(set(partition.values()))} 个社区"
    )
    return {name: int(community_id) for name, community_id in partition.items()}


# ============================================================
# 七、社区摘要
# ============================================================
_CYPHER_ALL_ENTITIES_WITH_COMMUNITY = """
MATCH (n:Entity)
WHERE n.community_id IS NOT NULL
RETURN n.name AS name, n.entity_type AS entity_type,
       n.description AS description, n.community_id AS community_id
ORDER BY n.community_id, n.name
"""

_CYPHER_COMMUNITY_RELATIONS = """
MATCH (a:Entity)-[r:RELATES_TO]->(b:Entity)
WHERE a.community_id = b.community_id AND a.community_id IS NOT NULL
RETURN a.name AS source, r.relation_type AS relation,
       b.name AS target, a.community_id AS community_id
"""


def group_entities_by_community(rows) -> dict[int, list[dict]]:
    """按 community_id 把实体分组；未分配（None / 负数）的实体丢掉。

    丢 None 是因为 Cypher 侧查的是 `community_id IS NOT NULL`；再判一次 `< 0`
    是为了兼容 `Entity.community_id` 的 -1 默认值（两套判据的由来见 models.Entity）。
    组的顺序与组内顺序都沿用 Cypher 的 `ORDER BY n.community_id, n.name`，
    dict 保序 ⇒ 拼出来的社区摘要提示词是可复现的（同一张图重跑逐字相同）。
    """
    grouped: dict[int, list[dict]] = {}
    for row in rows or []:
        community_id = row.get("community_id")
        if community_id is None:
            continue
        community_id = int(community_id)
        if community_id < 0:
            continue
        grouped.setdefault(community_id, []).append(row)
    return grouped


def generate_community_summaries() -> list[dict]:
    """为每个社区生成 100~200 字摘要，并把摘要与摘要向量写进 Community 节点。

    摘要向量是分层检索的入口：查询向量先在社区摘要索引上召回，锁定语义区域。

    成本：社区数 ×（1 次 LLM + 1 次 embedding），**串行**、无并发也无缓存。
    串行是刻意的 —— 一轮建图本来就跑几分钟，串行能把网关限流/断连的概率压到最低，
    也避免并发写同一个 Community 节点。
    温度 0.3（比抽取的 0.1 高）：摘要要的是可读的归纳，不是稳定复现的结构化输出。
    这里用 `get_or_none` + 新建：Community 节点的唯一键是 community_id，
    重跑时是**覆盖**旧摘要（而不是留下两份），但已消失的社区不会被清理。
    """
    connect()
    entity_rows = run_cypher(_CYPHER_ALL_ENTITIES_WITH_COMMUNITY)
    grouped = group_entities_by_community(entity_rows)
    if not grouped:
        logger.warning("[GraphRAG] 没有已分配社区的实体，跳过社区摘要")
        return []

    relations_by_community: dict[int, list[dict]] = {}
    for row in run_cypher(_CYPHER_COMMUNITY_RELATIONS) or []:
        community_id = row.get("community_id")
        if community_id is None:
            continue
        relations_by_community.setdefault(int(community_id), []).append(row)

    results = []
    for community_id in sorted(grouped):
        material = build_community_material(
            community_id, grouped[community_id], relations_by_community.get(community_id, [])
        )
        summary = call_llm(SUMMARY_SYSTEM, material, temperature=0.3).strip()
        node = Community.nodes.get_or_none(community_id=community_id)
        if node is None:
            node = Community(community_id=community_id)
        node.summary = summary
        node.embedding = embed_query(summary)
        node.save()
        results.append({"community_id": community_id, "summary": summary})

    logger.info(f"[GraphRAG] 社区摘要完成: {len(results)} 个社区")
    return results


# ============================================================
# 八、增量入库（LLM Wiki）
# ============================================================
_CYPHER_VECTOR_MATCH_ENTITY = """
CALL db.index.vector.queryNodes($index, $top_k, $vector) YIELD node, score
RETURN node.name AS name, node.entity_type AS entity_type,
       node.description AS description, node.community_id AS community_id, score
"""


# 消歧答复里需要剥掉的首尾引号/标点，以及表示"没有匹配"的词
_CHOICE_JUNK = " \t\r\n\"'“”‘’。，,；;：:、.!！?？"
_NONE_TOKENS = {"NONE", "NULL", "UNKNOWN", "无", "无相关", "都不相关", "无匹配", "没有"}


def parse_match_choice(raw, candidates) -> str | None:
    """解析消歧 LLM 的答复，返回候选里的名称；判不出来返回 None。

    LLM 可能回 'NONE'、带引号、带句号，甚至带一句解释，所以：
    先去首尾引号/标点 → 命中 NONE 类词就放弃 → 精确匹配候选 → 再退化为"包含关系"。
    只有落在候选集合里的名称才被接受，避免 LLM 编出一个图里没有的名字。

    ⚠ 退化匹配 `name in text` 是**子串**判断："张三"会被文本"张三丰"命中。
    兜底理由同上：返回的只能是候选集合里的名字，所以不会凭空造实体，
    最坏情况是选错候选 —— 而这也是为什么阈值（0.6）+ 同类型过滤要一起用。
    多个候选互相包含时取 `candidates` 里的**第一个**；candidates 由 Cypher 按
    score 降序返回（match_entity 保留了这个顺序），所以取到的就是最像的那个。
    `text.upper()` 对中文无影响，是为了兼容模型回小写 `none` 的情况。
    """
    if not raw:
        return None
    text = str(raw).strip().strip(_CHOICE_JUNK)
    if not text:
        return None
    if text.upper() in _NONE_TOKENS:
        return None

    names = [str(name).strip() for name in (candidates or []) if str(name or "").strip()]
    for name in names:
        if name == text:
            return name
    for name in names:
        if name in text:
            return name
    return None


def match_entity(name, entity_type=None) -> Entity | None:
    """增量入库时的实体消歧：先精确匹配，再走向量召回 + LLM 判定。

    1. 名称精确命中 → 直接返回（省一次 embedding + LLM）；
    2. 否则在实体向量索引上取 top10，只保留相似度 >= 0.6 且 entity_type 相同的候选；
    3. 把候选列表交给 LLM 判断是否指同一对象，返回候选中的名称或 NONE。

    限定同 entity_type 是刻意的：'张三'（人物）和'张三'（公司）向量上可能很像，
    但业务上不是一个东西。

    ⚠ 向量口径不对称（已知，标定阈值前必读）：
    索引侧的实体向量是 `embed_query("名称 描述")`（见 upsert_entity），
    而这里的查询向量是 `embed_query(名称)` —— 一边带描述、一边不带，
    相似度会被描述文本稀释（描述越长，同一实体的自相似度越低）。
    0.6 这个阈值就是在这个不对称口径下标定出来的；想改成
    `embed_query(f"{name} {entity_type}")` 或纯名称，**必须重新标定 MATCH_SCORE_THRESHOLD**。
    另一处成本细节：向量索引的查询依赖 `init_schema()` 建好的原生向量索引，
    索引不存在时 `queryNodes` 直接抛错（不是返回空），所以建图脚本第一步就是 init_schema。
    """
    name = str(name or "").strip()
    if not name:
        return None
    entity_type = str(entity_type or "").strip() or "UNKNOWN"

    connect()
    exact = Entity.nodes.get_or_none(name=name)
    if exact is not None:
        return exact

    rows = run_cypher(
        _CYPHER_VECTOR_MATCH_ENTITY,
        {
            "index": settings.neo4j.entity_index,
            "top_k": MATCH_CANDIDATE_TOP_K,
            "vector": embed_query(name),
        },
    )
    candidates = [
        row
        for row in rows
        if (row.get("score") or 0) >= MATCH_SCORE_THRESHOLD
        and (str(row.get("entity_type") or "").strip() or "UNKNOWN") == entity_type
    ]
    if not candidates:
        return None

    candidate_text = "[" + "，".join(
        f"{row.get('name')} ({row.get('entity_type')}): {row.get('description') or ''}"
        for row in candidates
    ) + "]"
    answer = call_llm(
        MATCH_ENTITY_SYSTEM,
        MATCH_ENTITY_USER_TMPL.format(name=name, candidates=candidate_text),
        temperature=0,
    )
    chosen = parse_match_choice(answer, [row.get("name") for row in candidates])
    if chosen is None:
        logger.info(f"[GraphRAG] 实体「{name}」未匹配到已有实体，将新建")
        return None
    logger.info(f"[GraphRAG] 实体「{name}」匹配到已有实体「{chosen}」")
    return Entity.nodes.get_or_none(name=chosen)


def link_or_create(item, doc_id=None) -> Entity | None:
    """增量入库的核心：把一条抽取结果接到已有实体上，接不上就新建。

    与 upsert_entity 的分工：upsert_entity 是"按名称精确 upsert"（create_graph 全量建图用），
    本函数多走了一次**消歧**（match_entity：精确 → 向量召回 → LLM 判定），增量入库用。
    两者"命中已有节点后"的合并逻辑**完全一致**（都走 merged_entity_state）——
    差别只在"怎么找到那个节点"。
    命中时同样不更新 entity_type、不重算向量（首次写入即生效，理由见 upsert_entity）。
    返回 None 只有一种情况：实体名为空（调用方要据此决定要不要计入统计）。
    """
    fields = _entity_fields(item)
    if not fields["name"]:
        logger.warning("[GraphRAG] link_or_create 收到无名实体，跳过")
        return None

    matched = match_entity(fields["name"], fields["entity_type"])
    if matched is not None:
        state = merged_entity_state(
            matched, fields["description"], doc_id=doc_id, doc_ids=fields["doc_ids"]
        )
        matched.description = state["description"]
        matched.doc_ids = state["doc_ids"]
        matched.mentions = state["mentions"]
        matched.save()
        return matched

    node = Entity(
        name=fields["name"],
        entity_type=fields["entity_type"],
        description=fields["description"],
        doc_ids=merge_doc_ids(fields["doc_ids"], [doc_id] if doc_id is not None else []),
        mentions=1,
        embedding=embed_query(f"{fields['name']} {fields['description']}".strip()),
    )
    node.save()
    return node


def ingest_document(text, doc_id) -> dict:
    """增量入库一篇文档：先抽取，再把实体/关系链接进图。

    关键细节：关系端点必须走「抽取名 → 图里真实名称」的映射。
    例如本抽取到"董文"、而图里已有节点叫"DONGWEN"（消歧判为同一对象）时，
    关系会写成 `DONGWEN -同一人-> 董文`，而"董文"这个节点根本不存在，
    端点缺失关系就会被整条丢掉。所以在同一次抽取内先把名称映射建好再连关系。

    与 create_graph 的区别：不清空图，适合"又来了一批票据"的持续写入。

    alias_map 只覆盖**本次抽取内部**：抽取名 → 图中真实名。跨文档的消歧不用它，
    由 match_entity 在 link_or_create 里用向量+LLM 完成（所以它不需要持久化）。
    关系计数只统计真正写成功的（`upsert_relation` 返回 False 的不计）——
    端点缺失会在那里打 WARNING，这里不要把它算进"入库成功"里。
    """
    data = extract_entities(text)

    entity_count = 0
    # 抽取名 → 图中真实名称（新建时两者相同；命中已有实体时会被改写成既有名称）
    # 例：本抽取到"董文"，消歧命中图里已有的 "DONGWEN" ⇒ 映射成 DONGWEN，
    # 于是后面所有以"董文"为端点的关系都会正确接到 DONGWEN 上。
    alias_map: dict[str, str] = {}
    for item in data["entities"]:
        node = link_or_create(item, doc_id)
        if node is None:
            continue
        alias_map[_entity_fields(item)["name"]] = node.name
        entity_count += 1

    relation_count = 0
    for item in data["relations"]:
        fields = _relation_fields(item)
        if not fields["source"] or not fields["target"]:
            continue
        source_name = alias_map.get(fields["source"], fields["source"])
        target_name = alias_map.get(fields["target"], fields["target"])
        if upsert_relation(
            source_name,
            target_name,
            fields["relation_type"],
            confidence=fields["confidence"],
            doc_id=doc_id,
            doc_ids=fields["doc_ids"],
        ):
            relation_count += 1

    logger.info(
        f"[GraphRAG] 文档 {doc_id} 入库完成: 实体 {entity_count} 个 / 关系 {relation_count} 条"
    )
    return {"entities": entity_count, "relations": relation_count}

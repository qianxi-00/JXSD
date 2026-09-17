"""GraphRAG 分层检索：社区摘要召回 → 实体向量召回 → 多跳遍历 → 组装 RAG 提示词。

课案的分层下钻思路：
1. `match_communities`     用查询向量在**社区摘要**向量索引上召回，先锁定语义区域；
2. `extract_query_entities` 用 LLM 从问题里抽实体名；
3. `vector_match_entities` 把实体名向量化，在**实体**向量索引上召回种子实体；
4. `retrieve_by_entities`  从种子沿 `relates_to` 做多跳遍历，收集子图（节点/关系去重）；
5. `retrieve_hierarchical` 把 1~4 串起来，带上课案点名的空社区兜底；
6. `build_rag_prompt`      把子图渲染成提示词，交给 LLM 生成回答。

同样地，外部依赖都是模块级可替换名字（`run_cypher` / `embed_query` / `call_llm`），
节点/关系去重逻辑抽成纯函数 `collect_subgraph`，方便离线单测。

一次 `retrieve_hierarchical(query)` 的网络/数据库代价（决定了"为什么一次问答要几秒"）：
    1 次 embedding（查询向量 → 社区召回）
  + 1 次 LLM（从问题里抽实体名，json_mode）
  + N 次 embedding（N = 抽出的实体名个数，逐个向量化，**没有批量**）
  + 1 次 Cypher（多跳遍历）
  + 1 次 LLM（生成回答）
注意这条链上**没有任何缓存**：同一个问题问两遍就是两遍全量调用。
"""

from config import settings
from core.logger import logger
from graph_rag.builder import (
    call_llm,
    format_entity_lines,
    format_relation_lines,
    load_json_object,
)
from graph_rag.models import connect, run_cypher
from retrieval.embedding import embed_query

# 多跳遍历的跳数上限：再大路径数量会指数膨胀，收益也很低
# ⚠ 这个常量必须与 service.py 里 `GraphQueryRequest.max_hops` 的 `Field(le=3)` 同步：
#   接口层先拦一次（超了直接 422），这里再兜一次（直接调用函数时收口）。
MAX_HOPS_LIMIT = 3
# 多跳遍历取回的路径条数上限（按 max_nodes 放大，兜住稠密图的路径爆炸）
PATH_LIMIT_FACTOR = 10
PATH_LIMIT_MIN = 50

# ============================================================
# Cypher
# ============================================================
# {hops} 必须是字面量：Neo4j 不允许把变长路径的上界写成参数 `*1..$hops`，
# 所以这里用格式化拼进去。hops 由 int() 收口，不存在注入面。
#
# 两个容易读错的点：
# 1. 遍历是**无向**的（`-[:RELATES_TO*1..N]-` 两端都没有箭头），所以可能"逆着"边的存储方向走。
#    但返回的关系用 `startNode(r)` / `endNode(r)` 取的是**存储方向**的端点，
#    因此渲染出来的 `源 关系 目标` 始终与图里一致（这点很关键，否则提示词会把关系说反）。
# 2. `LIMIT $path_limit` 限的是**路径条数**，不是节点数：一条路径可以带回多个节点，
#    所以子图的节点上限要在 collect_subgraph 里另做（max_nodes），两者不能互相替代。
_TRAVERSE_CYPHER_TMPL = """
MATCH (s:Entity)
WHERE s.name IN $seed_names
MATCH p = (s)-[:RELATES_TO*1..{hops}]-(m:Entity)
{community_filter}
RETURN [n IN nodes(p) | {{name: n.name, entity_type: n.entity_type,
                          description: n.description, community_id: n.community_id}}] AS path_nodes,
       [r IN relationships(p) | {{source: startNode(r).name, relation: r.relation_type,
                                 target: endNode(r).name}}] AS path_rels
LIMIT $path_limit
"""

# 社区限定：路径上的每个节点都必须落在允许的社区里，遍历不得跨社区
# 注意是 `ALL(...)`（每个节点）而不是 `ANY(...)`：只要路径上有一个节点在社区外，
# 整条路径就被丢掉 —— 所以种子实体自己就落在社区外时，结果会是空的。
# 这正是 retrieve_by_entities 要先把"不属于目标社区"的种子过滤掉的原因。
_COMMUNITY_FILTER = "WHERE ALL(x IN nodes(p) WHERE x.community_id IN $community_ids)"

# 下面两条实体向量查询的差别只有一个 WHERE，但语义差别很大：
#   _CYPHER_VECTOR_ENTITY            全局检索（不带社区过滤）
#   _CYPHER_VECTOR_ENTITY_IN_COMMUNITY  先取 top_k 再按社区过滤（过滤**没有**下推到索引查询）
# Neo4j 的 `db.index.vector.queryNodes` 不接受附加谓词，只能写成"先召回再过滤"，
# 所以社区内的候选数可能少于 top_k —— 这是课案写法，也是 retrieve_hierarchical
# 需要"空社区兜底"的根本原因。
_CYPHER_VECTOR_ENTITY = """
CALL db.index.vector.queryNodes($index, $top_k, $vector) YIELD node, score
RETURN node.name AS name, node.entity_type AS entity_type,
       node.description AS description, node.community_id AS community_id, score
"""

_CYPHER_VECTOR_ENTITY_IN_COMMUNITY = """
CALL db.index.vector.queryNodes($index, $top_k, $vector) YIELD node, score
WHERE node.community_id IN $community_ids
RETURN node.name AS name, node.entity_type AS entity_type,
       node.description AS description, node.community_id AS community_id, score
"""

_CYPHER_VECTOR_COMMUNITY = """
CALL db.index.vector.queryNodes($index, $top_k, $vector) YIELD node, score
RETURN node.community_id AS community_id, node.summary AS summary, score
"""

# ============================================================
# LLM 提示词
# ============================================================
QUERY_ENTITY_SYSTEM = "你是一个专业的实体提取助手。"
QUERY_ENTITY_USER_TMPL = """请从下面的用户问题中提取出全部实体名称，只返回 JSON，不要任何解释。

返回格式：
{{"entities": ["实体1", "实体2"]}}

用户问题：{query}
"""

RAG_SYSTEM = "你是一个严谨的财务知识问答助手，只能依据给定的知识图谱信息回答，不要编造。"
# 防幻觉的最后一道：即使子图是空的，也要让模型有机会说"图里没有"，
# 而不是顺着"请根据以下知识图谱信息回答"硬编一个答案出来。
NO_INFO_HINT = "如果知识图谱中没有相关信息，请明确说明"


# ============================================================
# 一、查询侧实体抽取
# ============================================================
def extract_query_entities(query) -> list[str]:
    """用 LLM 从查询里抽实体名（去重、去空白）。

    空查询直接短路，不去打扰模型。

    这是分层检索里**唯一**从自然语言拿实体的入口：抽不出名字（返回 []）时，
    下游 `retrieve_by_entities` 会直接返回空子图，最终只有社区摘要能被喂给 LLM
    —— 表现就是"答得含糊但不像报错"，所以这条路径退化时不容易被发现。
    单条结构化结果 `[{"name": "张三"}]` 也兼容（模型有时不按 `["张三"]` 的格式回）。
    """
    if not query or not str(query).strip():
        return []
    raw = call_llm(
        QUERY_ENTITY_SYSTEM,
        QUERY_ENTITY_USER_TMPL.format(query=str(query).strip()),
        temperature=0.1,
        json_mode=True,
    )
    data = load_json_object(raw)

    names: list[str] = []
    for item in data.get("entities") or []:
        # 模型偶尔会返回 [{"name": "张三"}] 这种结构化结果，一并兼容
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
        else:
            name = str(item or "").strip()
        if name and name not in names:
            names.append(name)
    return names


# ============================================================
# 二、社区摘要召回
# ============================================================
def match_communities(query, top_k: int = 2) -> list[dict]:
    """用查询向量在社区摘要向量索引上召回语义区域。

    返回 `[{"community_id", "summary", "score"}]`。这是分层检索的第一跳：
    先知道"问题大概落在哪片语义区域"，再进去做实体级检索。

    `score` 是**社区摘要**与查询向量的相似度，不是实体相似度，两者不可比、别混用阈值。
    查询向量用的是整句 `query`（不做改写、不抽关键词）——
    这也是它比实体检索"更稳"的原因：社区摘要本来就是一段自然语言。
    这里查的是 Community 标签的全量节点，**不校验该社区是否还有实体**
    （残留的过时社区照样会被召回，见 builder.generate_community_summaries）。
    """
    connect()
    rows = run_cypher(
        _CYPHER_VECTOR_COMMUNITY,
        {
            "index": settings.neo4j.community_index,
            "top_k": top_k,
            "vector": embed_query(query),
        },
    )
    results = [
        {
            "community_id": row.get("community_id"),
            "summary": row.get("summary") or "",
            "score": row.get("score"),
        }
        for row in rows or []
    ]
    logger.info(
        f"[GraphRAG] 社区召回 {len(results)} 个 | 社区 ID: {[r['community_id'] for r in results]}"
    )
    return results


# ============================================================
# 三、实体向量召回
# ============================================================
def vector_match_entities(
    names, top_k: int = 5, community_ids=None
) -> list[dict]:
    """把实体名逐个向量化后，在实体向量索引上检索，按名称去重（保留最高分）。

    `community_ids=None`  → 全局检索（不带社区过滤）；
    `community_ids=[...]` → Cypher 里加 `WHERE node.community_id IN $community_ids`。

    注意这是**先取 top_k 再按社区过滤**（课案写法）：社区里的候选可能少于 top_k；
    传空列表 `[]` 时 `IN []` 恒不成立，会直接返回空——调用方必须自己兜底
    （见 retrieve_hierarchical）。

    两个实现细节：
    1. 逐个名字各发一次 embedding（N 个名字 = N 次网络往返，没有批量），
       所以"问题里提到的实体多"会线性变慢；同理每个名字各查一次向量索引。
    2. 返回前按 score 降序**显式排序**：Cypher 不保证跨查询的结果顺序，
       依赖"返回顺序就是相似度顺序"会在换引擎时静默出错；
       `matched` 的 key 用的是**图中节点名**（同名只保留最高分那条）。
    """
    if not names:
        return []
    connect()

    matched: dict[str, dict] = {}
    for raw_name in names:
        name = str(raw_name or "").strip()
        if not name:
            continue
        params = {
            "index": settings.neo4j.entity_index,
            "top_k": top_k,
            "vector": embed_query(name),
        }
        if community_ids is None:
            rows = run_cypher(_CYPHER_VECTOR_ENTITY, params)
        else:
            params["community_ids"] = list(community_ids)
            rows = run_cypher(_CYPHER_VECTOR_ENTITY_IN_COMMUNITY, params)

        for row in rows or []:
            key = str(row.get("name") or "").strip()
            if not key:
                continue
            if key not in matched or (row.get("score") or 0) > (matched[key].get("score") or 0):
                matched[key] = row

    return sorted(matched.values(), key=lambda row: row.get("score") or 0, reverse=True)


# ============================================================
# 四、子图收集（纯逻辑）
# ============================================================
def collect_subgraph(path_rows, seeds, max_nodes: int = 20) -> dict:
    """把多跳路径整理成子图：节点/关系去重 + 节点数上限。

    - 节点按名称去重，**种子排在最前**：孤立实体（没有任何关系）也必须留在子图里，
      否则"只查到一个孤立实体"这种最正常的情况反而什么都返回不了；
    - 关系按 `f"{source}-{relation}-{target}"` 去重；
    - max_nodes 截断后，两端不在保留节点里的关系一并丢掉，避免悬空关系。

    截断的取舍：`nodes` 是 dict（保序），种子在最前面、路径节点按 Cypher 返回顺序追加，
    所以 `[:limit]` 砍掉的永远是"多跳捞回来的"节点，**种子一定留得住**。
    这一步很关键：只查到一个孤立实体（没有任何关系、Cypher 返回空路径）时，
    子图里剩下的就是这个种子本身，而不是空。
    """
    nodes: dict[str, dict] = {}

    def _add_node(raw) -> None:
        raw = raw or {}
        name = str(raw.get("name") or "").strip()
        if not name or name in nodes:
            return
        nodes[name] = {
            "name": name,
            "entity_type": raw.get("entity_type") or "UNKNOWN",
            "description": raw.get("description") or "",
            "community_id": raw.get("community_id"),
        }

    for seed in seeds or []:
        _add_node(seed)

    relationships: dict[str, dict] = {}
    for row in path_rows or []:
        row = row or {}
        for node in row.get("path_nodes") or []:
            _add_node(node)
        for rel in row.get("path_rels") or []:
            rel = rel or {}
            source = str(rel.get("source") or "").strip()
            target = str(rel.get("target") or "").strip()
            relation = str(rel.get("relation") or "").strip()
            if not source or not target:
                continue
            key = f"{source}-{relation}-{target}"
            relationships.setdefault(
                key, {"source": source, "relation": relation, "target": target}
            )

    limit = int(max_nodes) if max_nodes and int(max_nodes) > 0 else len(nodes)
    node_list = list(nodes.values())[:limit]
    kept = {node["name"] for node in node_list}
    return {
        "nodes": node_list,
        "relationships": [
            rel for rel in relationships.values() if rel["source"] in kept and rel["target"] in kept
        ],
    }


# ============================================================
# 五、实体级多跳检索
# ============================================================
def retrieve_by_entities(
    entities, max_hops: int = 2, max_nodes: int = 20, community_ids=None
) -> dict:
    """实体级检索：向量召回种子实体 → 沿 relates_to 多跳遍历 → 返回子图。

    关于 `community_ids`（课案语义）：
    **种子检索本身不做社区限定**——先全局取 top_k，再把"不属于目标社区"的种子过滤掉，
    因此社区内命中的种子可能少于 top_k。只有多跳遍历不允许跨出社区。
    没有实体入参时直接返回空结构，连数据库都不碰。

    ⚠ `community_ids=None` 与 `community_ids=[]` 语义**完全不同**，这是个真陷阱：
    None → 不限社区；`[]` → `allowed` 是空集合 ⇒ 所有种子都被过滤掉 ⇒ 返回空子图。
    所以调用方必须先判空（`retrieve_hierarchical` 就是这么做的：社区没召回时传 None 兜底）。
    `hops` / `limit` 都做了上下界收口，防调用方传 0、负数或超大值把 Cypher 撑爆。
    """
    if not entities:
        return {"nodes": [], "relationships": []}

    connect()
    seeds = vector_match_entities(entities)  # 种子检索：全局、不限社区
    if community_ids is not None:
        allowed = {int(cid) for cid in community_ids}
        seeds = [
            seed
            for seed in seeds
            if seed.get("community_id") is not None and int(seed["community_id"]) in allowed
        ]
    if not seeds:
        logger.info("[GraphRAG] 没有可用的种子实体，返回空子图")
        return {"nodes": [], "relationships": []}

    hops = max(1, min(int(max_hops or 1), MAX_HOPS_LIMIT))
    limit = max(int(max_nodes or 1), 1)
    params = {
        "seed_names": [seed["name"] for seed in seeds],
        "path_limit": max(limit * PATH_LIMIT_FACTOR, PATH_LIMIT_MIN),
    }
    community_filter = ""
    if community_ids is not None:
        params["community_ids"] = sorted(allowed)
        community_filter = _COMMUNITY_FILTER

    rows = run_cypher(
        _TRAVERSE_CYPHER_TMPL.format(hops=hops, community_filter=community_filter), params
    )
    subgraph = collect_subgraph(rows, seeds, max_nodes=limit)
    logger.info(
        f"[GraphRAG] 实体级检索: 种子 {len(seeds)} 个 → 节点 {len(subgraph['nodes'])} 个 / "
        f"关系 {len(subgraph['relationships'])} 条 (hops={hops}, 社区限定={community_ids})"
    )
    return subgraph


# ============================================================
# 六、分层检索（含空社区兜底）
# ============================================================
def retrieve_hierarchical(
    query, top_k: int = 2, max_hops: int = 2, max_nodes: int = 20
) -> dict:
    """分层下钻：先社区召回锁定语义区域，再在该区域内做实体检索 + 多跳遍历。

    兜底（课案点名）：一个社区都没召回时 `community_ids` 为空列表，
    `WHERE ... IN []` 恒不成立 → 结果必然为空。所以此时退回**不限社区**的实体级检索。

    注意兜底只解决"社区没召回"，解决不了"实体没抽出来"：
    两个都空时（`names == []`）`retrieve_by_entities` 直接返回空子图，
    最终 LLM 只拿到空的社区段落 —— 回答会是"图谱里没有相关信息"，
    而这**未必**意味着图里真没有（可能只是抽名字这一步失败了，见 extract_query_entities）。
    """
    connect()
    communities = match_communities(query, top_k=top_k)
    community_ids = [
        int(item["community_id"])
        for item in communities
        if item.get("community_id") is not None
    ]
    names = extract_query_entities(query)

    if community_ids:
        subgraph = retrieve_by_entities(
            names, max_hops=max_hops, max_nodes=max_nodes, community_ids=community_ids
        )
    else:
        logger.warning("[GraphRAG] 未召回任何社区，回退到不限社区的实体级检索")
        subgraph = retrieve_by_entities(names, max_hops=max_hops, max_nodes=max_nodes)

    return {
        "communities": communities,
        "nodes": subgraph["nodes"],
        "relationships": subgraph["relationships"],
    }


# ============================================================
# 七、RAG 提示词与回答
# ============================================================
def build_rag_prompt(query, subgraph) -> str:
    """把子图渲染成提示词：社区摘要 + 实体清单 + 关系清单 + 无信息兜底提示。

    空段落直接省略：没有实体就不要写一个空的"相关实体："，
    否则模型容易顺着空标题瞎编。

    提示词里**只有图谱三元组，没有票据原文**：票号、金额、日期这类细节若没被抽成
    实体/关系（或抽了但没落进子图），模型就答不出来 —— 它会说"图谱里没有"，
    这不是回答错了，是图谱本身的覆盖度问题。
    `subgraph` 允许缺 key（用 `.get`），因为三种检索方式返回的结构不完全一样
    （community 方式只有 communities，没有 nodes/relationships）。
    """
    subgraph = subgraph or {}
    sections = [f"请根据以下知识图谱信息回答用户问题。\n\n用户问题：{query}"]

    community_lines = "\n".join(
        f"- 社区 {item.get('community_id')}：{item.get('summary') or ''}"
        for item in subgraph.get("communities") or []
        if item.get("community_id") is not None
    )
    if community_lines:
        sections.append(f"相关社区：\n{community_lines}")

    entity_lines = format_entity_lines(subgraph.get("nodes") or [])
    if entity_lines:
        sections.append(f"相关实体：\n{entity_lines}")

    relation_lines = format_relation_lines(subgraph.get("relationships") or [])
    if relation_lines:
        sections.append(f"相关关系：\n{relation_lines}")

    sections.append(NO_INFO_HINT)
    return "\n\n".join(sections)


def answer_query(query, subgraph) -> str:
    """基于子图提示词调 LLM 生成最终回答

    温度 0.2：比抽取（0.1）略高、比社区摘要（0.3）低 —— 事实问答要稳，
    但也别低到把措辞钉死。`call_llm` 是 builder 的出口，所以它同样带 3 次重试。
    """
    return call_llm(RAG_SYSTEM, build_rag_prompt(query, subgraph), temperature=0.2).strip()

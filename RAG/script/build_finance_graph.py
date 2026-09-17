"""从 Milvus 票据数据一键构建 GraphRAG 知识图谱。

流程（对应课案「建图」全链路）：
    读取 Milvus 票据 → LLM 抽取实体/关系 → 入图
    → Louvain 社区检测 → LLM 社区摘要 → 分层检索演示 → 输出统计报告

用法（在 Python_Base 目录下执行）：
    uv run python RAG/script/build_finance_graph.py --limit 3
    uv run python RAG/script/build_finance_graph.py --limit 50 --mode batch
    uv run python RAG/script/build_finance_graph.py --limit 3 --query "有哪些和差旅相关的票据？" --report report.json

两种入库模式：
    incremental（默认）—— 每篇票据单独抽取后"链接或新建"（match_entity / link_or_create），
                          演示增量入库（LLM Wiki）路径；
    batch              —— 先把所有票据的实体/关系抽完，再一次性 create_graph 全量重建。

注意：两种模式都会先清空整张图（课案要求全量可复现），
      所以不要在生产库上直接跑。

一轮的时间和调用量（全量规模实测是"要跑很久"的量级，冒烟只用 --limit 3）：
      文档数 × 1 次 LLM（抽取）
    + 1 次 Louvain（本地 CPU）
    + 社区数 ×（1 次 LLM 摘要 + 1 次 embedding）
    + 检索演示（1 次 embedding + 1 次 LLM 抽实体 + N 次 embedding + 1 次 Cypher + 1 次 LLM 回答）
    ⇒ 想先确认链路通不通，用 `--limit 3` 跑冒烟（实测 51 实体 / 66 关系 / 6 社区），
      别一上来就 --limit 300。

`--report` 落盘的 JSON 里包含：模式/条数/分辨率、每篇文档的 doc_id 与字数、
建图计数、**从数据库数出来的真实规模**、社区摘要全文、检索命中情况与最终回答、耗时。
排查"入库报了多少、库里到底有多少"这类问题时，直接看这份报告比看日志快。
"""

import argparse
import json
import sys
import time
from pathlib import Path

# --- 路径引导：脚本直接跑时 sys.path[0] 是 RAG/script，需要手动挂上项目根与 RAG 根 ---
_BASE = Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
for _path in (str(_BASE), str(_BASE / "RAG")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from config import settings  # noqa: E402
from core.database import get_milvus_client  # noqa: E402
from core.logger import logger  # noqa: E402
from graph_rag import builder, models  # noqa: E402
from graph_rag.retriever import (  # noqa: E402
    answer_query,
    extract_query_entities,
    retrieve_hierarchical,
)
from retrieval.vector_retrieval import OUTPUT_FIELDS  # noqa: E402

# 输出重定向到文件时 Windows 默认用本地编码(GBK)，中文会乱掉；统一成 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 - 老环境不支持就算了
    pass

# 与 graph_rag/builder.py 的 _CYPHER_CLEAR_GRAPH 是同一句话的**两份真相**
# （那边 create_graph 用，这边 build_incremental 用）。改一处务必同步另一处。
_CLEAR_GRAPH_CYPHER = "MATCH (n) DETACH DELETE n"

_GENERIC_QUERY = "这些票据里涉及哪些人员和单位？"


def pick_demo_query(given: str) -> str:
    """选一个用于演示分层检索的查询。

    用户没传 --query 时，自动挑图中度数最高的实体名来问——分层检索要先能从问题里抽出
    实体名才能走到实体层，问"这些票据都涉及什么"这种没有具体实体的问题，
    只能拿到社区摘要、拿不到子图，演示会显得像坏了。

    `MATCH (n:Entity)-[r:RELATES_TO]-()` 是无向匹配，所以度数是两侧合计；
    `ORDER BY degree DESC, name` 里带上 name 是为了**结果可复现**
    （并列时没有第二排序键，每次跑的赢家可能不同）。
    图是空的时候返回空列表，落到 _GENERIC_QUERY —— 那个问题抽不出实体，
    演示时只会走社区层，这是它作为最后兜底可以接受、但不适合当默认值的原因。
    """
    if given:
        return given
    rows = models.run_cypher(
        """
        MATCH (n:Entity)-[r:RELATES_TO]-()
        RETURN n.name AS name, count(r) AS degree
        ORDER BY degree DESC, name LIMIT 1
        """
    )
    if rows and rows[0].get("name"):
        return f"{rows[0]['name']} 与哪些实体有关联？"
    return _GENERIC_QUERY


# ============================================================
# 一、数据读取
# ============================================================
def fetch_tickets(limit: int) -> list[dict]:
    """从 Milvus 票据 collection 读若干条记录（复用现有 OUTPUT_FIELDS）

    `filter="id != ''"` 是必需的：Milvus 的 query 不接受空表达式。
    这只是"取 limit 条"，**不保证顺序**（没有 order by 的概念），
    所以同一 --limit 两次跑出来的文档集合可能不同 ⇒ 图也不同（演示够用，
    要做可复现的实验得先把 doc_id 列表固定下来）。
    `OUTPUT_FIELDS` 直接复用向量检索的口径（含 ocr_text），
    所以 ticket_text 能拿到语义文本与 OCR 原文两条信息。
    """
    client = get_milvus_client()
    rows = client.query(
        collection_name=settings.milvus.collection,
        filter="id != ''",
        output_fields=OUTPUT_FIELDS,
        limit=limit,
    )
    logger.info(f"[建图脚本] 从 Milvus({settings.milvus.collection}) 读取 {len(rows)} 条票据")
    return rows


def ticket_text(row: dict) -> str:
    """把一条票据记录拼成给 LLM 的文本：语义文本 + 关键结构化字段。

    只喂 semantic_text 会丢掉票号/金额/人员这些最值得抽成实体的字段，
    所以这里补上带中文标签的结构化信息。

    金额以「分」存（`amount_fen`），这里换成元只是为了让人/模型看懂 ——
    它是字符串化的数字，所以要 try/except：脏数据（"38,029.91" 这种带千分位的）
    会抛 ValueError，此时退化成原样输出，而不是让整篇票据的抽取失败。
    `source_file` 也会被抽成实体（文件名当实体），算噪音；
    保留它是因为课案要求把关键字段全部喂给抽取提示词，宁可多抽不要漏抽。
    """
    parts = [str(row.get("semantic_text") or "").strip()]
    for key, label in (
        ("ticket_type", "票据类型"),
        ("ticket_no", "票据号"),
        ("person", "人员"),
        ("route", "行程"),
        ("counterparty", "往来单位"),
        ("source_file", "来源文件"),
    ):
        value = row.get(key)
        if value not in (None, ""):
            parts.append(f"{label}：{value}")
    amount_fen = row.get("amount_fen")
    if amount_fen not in (None, ""):
        try:
            parts.append(f"金额：{float(amount_fen) / 100:.2f} 元")
        except (TypeError, ValueError):
            parts.append(f"金额（分）：{amount_fen}")
    return "\n".join(part for part in parts if part).strip()


# ============================================================
# 二、建图
# ============================================================
def build_incremental(documents: list[tuple[str, str]]) -> dict:
    """增量入库：逐个文档抽取后走 link_or_create（LLM Wiki 路径）

    ⚠ "增量"只体现在**同一次运行内部**：函数开头先把整张图清空了，
    所以它演示的是"逐篇链接进已有图"的路径（消歧 + 别名映射），
    不是跨运行的增量追加 —— 下次再跑还是从零开始。
    要真正的增量入库，得去掉这里的清空动作（并接受图会越来越大）。
    """
    models.run_cypher(_CLEAR_GRAPH_CYPHER)
    logger.info("[建图脚本] 已清空图谱，开始增量入库")
    stats = {"entities": 0, "relations": 0}
    for doc_id, text in documents:
        part = builder.ingest_document(text, doc_id)
        stats["entities"] += part["entities"]
        stats["relations"] += part["relations"]
    return stats


def build_batch(documents: list[tuple[str, str]]) -> dict:
    """全量建图：先把所有文档的实体/关系抽完，再一次性 create_graph（内部会清空整图）

    `{**item, "doc_ids": [doc_id]}` 是给每条抽取结果挂上"来自哪篇票据"的证据链
    （create_graph 的 upsert 会把它并进 doc_ids）。
    注意 create_graph 内部还会再清一次图 —— 这里是**重复清空**（无害，但要知道），
    所以 batch 模式的删图动作有两处，排查"图被谁清了"时别只看一处。
    """
    entities: list[dict] = []
    relations: list[dict] = []
    for doc_id, text in documents:
        data = builder.extract_entities(text)
        for item in data["entities"]:
            entities.append({**item, "doc_ids": [doc_id]})
        for item in data["relations"]:
            relations.append({**item, "doc_ids": [doc_id]})
    logger.info(f"[建图脚本] 抽取完成: 实体 {len(entities)} 条 / 关系 {len(relations)} 条，开始全量建图")
    return builder.create_graph(entities, relations)


def graph_stats() -> dict:
    """从图里数出真实的节点/关系/社区规模（不信抽取条数，信数据库）

    这三个数字与 build_incremental/build_batch 的返回值**口径不同且通常不相等**：
    那边是"处理了多少条输入"，这边是"库里真实有多少"（重复实体被合并、坏关系被跳过）。
    对不上是正常的，别拿其中一个去校验另一个。

    ⚠ `communities`（Community 节点数）与 `community_ids`（Entity 上出现过的社区号）
    也可能对不上：前者包含**上一轮残留、这一轮已无实体认领**的社区节点
    （generate_community_summaries 不清理它们）。两者不等就是有残留。
    `[0]["total"]` 直接下标：count 查询必然返回一行，空图也返回 0（不是空列表）。
    """
    entity_rows = models.run_cypher("MATCH (n:Entity) RETURN count(n) AS total")
    relation_rows = models.run_cypher("MATCH ()-[r:RELATES_TO]->() RETURN count(r) AS total")
    community_rows = models.run_cypher("MATCH (c:Community) RETURN count(c) AS total")
    community_id_rows = models.run_cypher(
        "MATCH (n:Entity) WHERE n.community_id IS NOT NULL "
        "RETURN DISTINCT n.community_id AS community_id ORDER BY community_id"
    )
    return {
        "entities": entity_rows[0]["total"] if entity_rows else 0,
        "relationships": relation_rows[0]["total"] if relation_rows else 0,
        "communities": community_rows[0]["total"] if community_rows else 0,
        "community_ids": [row["community_id"] for row in community_id_rows],
    }


# ============================================================
# 三、入口
# ============================================================
def parse_args(argv=None) -> argparse.Namespace:
    """命令行入口。argv=None 时读 sys.argv —— 显式传 argv 是为了能在测试里直接调 main()。

    参数语义补充：
      --top-k      只影响**社区召回条数**（retrieve_hierarchical 的 top_k），
                   不控制实体召回条数（那条路径的 top_k 用的是默认值 5）。
      --resolution Louvain 分辨率：>1 得到更多更小的社区（摘要调用次数变多），
                   <1 得到更少更大的社区（社区摘要会更笼统）。
      --mode       见模块 docstring；两种模式都会清空整图。
    """
    parser = argparse.ArgumentParser(description="从 Milvus 票据数据构建 GraphRAG 知识图谱")
    parser.add_argument("--limit", type=int, default=10, help="读取多少条票据（默认 10）")
    parser.add_argument(
        "--mode",
        choices=["incremental", "batch"],
        default="incremental",
        help="入库模式：incremental=逐篇链接或新建（默认）；batch=一次性全量重建",
    )
    parser.add_argument("--resolution", type=float, default=1.0, help="Louvain 分辨率")
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="分层检索演示用的查询；留空则自动挑图中关联最多的实体来问",
    )
    parser.add_argument("--top-k", type=int, default=2, help="社区召回个数")
    parser.add_argument("--report", type=str, default="", help="把完整报告写到这个 JSON 文件（UTF-8）")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()

    logger.info("=" * 70)
    logger.info(f"[建图脚本] 开始 | 模式={args.mode} | 条数={args.limit} | Louvain resolution={args.resolution}")

    # 0) schema：唯一性约束 + 两个原生向量索引
    #    必须最先做：向量索引不存在时，后面的消歧与检索会在 queryNodes 上直接抛错
    #    （不是返回空）。init_schema 幂等，重复调用只是多等一次 awaitIndexes。
    models.init_schema()

    # 1) 读数据
    rows = fetch_tickets(args.limit)
    if not rows:
        logger.error("[建图脚本] Milvus 里没有读到票据，退出")
        return 1  # 非零退出码：空数据是失败，不要伪装成"跑完了"
    documents = [(str(row.get("id")), ticket_text(row)) for row in rows]

    # 2) 抽取 + 入图
    #    这一步决定后面所有统计；两种模式的区别只在"边抽边入"还是"抽完一次入"
    build_result = (
        build_batch(documents) if args.mode == "batch" else build_incremental(documents)
    )

    # 3) 社区检测（Louvain）
    #    必须在入图之后：它读的是图里的 RELATES_TO 权重
    partition = builder.detect_communities_louvain(resolution=args.resolution)

    # 4) 社区摘要（LLM + 摘要向量）
    #    必须在社区检测之后：它读的是 Entity.community_id；
    #    这一步的耗时 = 社区数 × 一次 LLM + 一次 embedding，是整轮最慢的一段
    summaries = builder.generate_community_summaries()

    # 5) 分层检索演示
    stats = graph_stats()
    demo_query = pick_demo_query(args.query)
    query_entities = extract_query_entities(demo_query)
    subgraph = retrieve_hierarchical(demo_query, top_k=args.top_k)
    answer = answer_query(demo_query, subgraph)

    hit_community_ids = [
        item["community_id"] for item in subgraph["communities"] if item.get("community_id") is not None
    ]
    report = {
        "mode": args.mode,
        "limit": args.limit,
        "resolution": args.resolution,
        "documents": [{"doc_id": doc_id, "chars": len(text)} for doc_id, text in documents],
        "build_result": build_result,
        "graph": stats,
        "partition_size": len(partition),
        "summaries": summaries,
        "query": demo_query,
        "query_entities": query_entities,
        "hit_communities": subgraph["communities"],
        "hit_community_ids": hit_community_ids,
        "subgraph": {
            "nodes": len(subgraph["nodes"]),
            "relationships": len(subgraph["relationships"]),
            "node_names": [node["name"] for node in subgraph["nodes"]],
            "relationships_detail": subgraph["relationships"],
        },
        "answer": answer,
        "elapsed_sec": round(time.perf_counter() - started, 2),
    }

    # 控制台只打 ASCII，避免 Windows 控制台编码把中文打花；中文结果在 JSON 报告里
    print("")
    print("[GraphRAG] build mode      :", args.mode)
    print("[GraphRAG] documents       :", len(documents))
    print("[GraphRAG] entities        :", stats["entities"])
    print("[GraphRAG] relationships   :", stats["relationships"])
    print("[GraphRAG] community count :", stats["communities"])
    print("[GraphRAG] community ids   :", stats["community_ids"])
    print("[GraphRAG] summaries       :", len(summaries))
    print("[GraphRAG] query entities  :", len(query_entities))
    print("[GraphRAG] hit communities :", hit_community_ids)
    print("[GraphRAG] subgraph nodes  :", len(subgraph["nodes"]))
    print("[GraphRAG] subgraph rels   :", len(subgraph["relationships"]))
    print("[GraphRAG] elapsed sec     :", report["elapsed_sec"])
    if args.report:
        # ensure_ascii=False + UTF-8：报告里的中文要是可读的（默认 ensure_ascii=True 会写成 \uXXXX，
        # 报告是给人看的，中文全变码点是没法排查的）
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print("[GraphRAG] report          :", args.report)

    logger.info(
        f"[建图脚本] 完成 | 实体={stats['entities']} 关系={stats['relationships']} "
        f"社区={stats['communities']} | 命中社区={hit_community_ids} | "
        f"子图节点={len(subgraph['nodes'])} 关系={len(subgraph['relationships'])}"
    )
    return 0  # 显式 0：__main__ 里 raise SystemExit(main()) 直接把它当退出码


if __name__ == "__main__":
    raise SystemExit(main())

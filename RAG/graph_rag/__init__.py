"""GraphRAG 模块（RAG 优化篇）。

对外职责划分：
- models.py     : neomodel 节点/关系定义 + Neo4j 连接与 Cypher 出口
- builder.py    : 建图（LLM 抽取 → 实体/关系入库 → 社区检测 → 社区摘要 → 增量入库）
- retriever.py  : 分层检索（社区摘要召回 → 实体向量召回 → 多跳遍历 → 组装 RAG 提示词）
- service.py    : FastAPI 接口（POST /api/graph_rag/query）

导入本包本身很轻：不会连 Neo4j、也不会构造 LLM/Embedding 客户端，
外部连接一律延迟到真正调用时（便于离线单测）。
"""

__all__ = ["models", "builder", "retriever", "service"]

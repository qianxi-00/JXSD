"""GraphRAG 的 FastAPI 服务。

启动（默认监听 settings.app.port + 1，即 8100）：
    uv run python -m graph_rag.service

接口：
    POST /api/graph_rag/query
    请求体 {"query": str, "retrieval_method": "entity"|"community"|"hybrid",
            "max_hops": int, "top_k": int}
    返回体 {"answer": str, "subgraph": {...}}

retrieval_method 的语义：
    entity    → 只做实体级检索（向量召回 + 多跳遍历）
    community → 只返回社区摘要，不进实体层
    其它值（含 hybrid）→ 走分层检索 retrieve_hierarchical
    注意：这里**没有**参数校验，传 "ENTITY"（大小写不符）会静默走 hybrid 分支 ——
    当前是内网演示口径，对外暴露前要先把取值收敛成字面量联合类型（Literal）。

现状边界（有意为之，不是遗漏）：
    - 没有鉴权、没有 CORS 中间件、没有 response_model（直接 return dict，
      所以 OpenAPI 里看不到响应结构）—— 只在内网/本机演示；
    端口取 `settings.app.port + 1`（主链路 8099 → GraphRAG 8100），
      避免与主服务抢端口，代价是这个偏移是**隐式约定**，改主服务端口会一起挪。
"""

import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# --- 路径引导：保证 `python -m graph_rag.service` 与直接跑脚本都能找到 config / core ---
# 往上找到**目录名恰好是 "Python_Base"** 的那一层当项目根 —— 即目录名被硬编码了：
# 仓库改名、或把 RAG 目录单独拷走，这里就找不到根，import config 会失败。
# 插到 sys.path[0]（而不是 append）是为了让仓内的 config / core 优先于任何同名第三方包。
_BASE = Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
for _path in (str(_BASE), str(_BASE / "RAG")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from config import settings  # noqa: E402
from core.logger import logger  # noqa: E402
from graph_rag.retriever import (  # noqa: E402
    answer_query,
    extract_query_entities,
    match_communities,
    retrieve_by_entities,
    retrieve_hierarchical,
)

app = FastAPI(title="GraphRAG 知识图谱检索服务")


class GraphQueryRequest(BaseModel):
    """GraphRAG 查询请求体"""

    query: str = Field(..., description="用户问题")
    retrieval_method: str = Field(
        "hybrid", description="检索方式：entity / community / hybrid"
    )
    # le=3 必须与 retriever.MAX_HOPS_LIMIT 保持一致（接口层先拦，函数层再兜一次）
    max_hops: int = Field(2, ge=1, le=3, description="多跳遍历的跳数")
    # ⚠ top_k 一个字段两个用途：社区召回条数，以及子图节点上限（= top_k * 2）。
    #   调大它同时会放宽"召回几个社区"和"子图留几个节点"，没有单独的旋钮。
    top_k: int = Field(5, ge=1, description="召回条数（同时决定子图节点上限）")


class GraphEntity(BaseModel):
    """子图里的一个实体节点。

    `name` / `entity_type` 是 Neo4j 侧一定有的两个字段；`description` 可能为空，
    `community_id` 在"实体级检索（没限定社区）"时可能缺失 —— 所以后者是可选的。
    用 `extra="allow"` 而不是逐个列出：图谱 schema 将来加字段时，服务端不该因为
    多了一个键就 500（这类"响应模型比数据严格"的失败最难查）。
    """

    model_config = ConfigDict(extra="allow")

    name: str = ""
    entity_type: str = ""
    description: str = ""
    community_id: int | None = None


class GraphRelationship(BaseModel):
    """子图里的一条关系（三元组）。"""

    model_config = ConfigDict(extra="allow")

    source: str = ""
    relation: str = ""
    target: str = ""


class GraphCommunity(BaseModel):
    """召回到的一个社区摘要（社区级检索时只有它，没有节点/关系）。"""

    model_config = ConfigDict(extra="allow")

    community_id: int | None = None
    summary: str = ""
    score: float | None = None


class GraphSubgraph(BaseModel):
    """检索到的子图。三种检索方式返回的字段**不完全一样**（见各字段默认值）：

    - `entity`：有 nodes / relationships，communities 为空；
    - `community`：只有 communities；
    - `hybrid`：三者都有。
    所以三个列表都给默认空列表，而不是让缺字段变成 500。
    """

    model_config = ConfigDict(extra="allow")

    communities: list[GraphCommunity] = Field(default_factory=list)
    nodes: list[GraphEntity] = Field(default_factory=list)
    relationships: list[GraphRelationship] = Field(default_factory=list)


class GraphQueryResponse(BaseModel):
    """GraphRAG 查询响应体（**响应模型的意义**：让 /docs 与 SDK 生成看得到契约）。

    课案（优化篇 2495-2497）要求给这个接口定义 `QueryResponse` —— 之前返回裸 dict，
    OpenAPI 里响应结构是空的，调用方只能靠猜（或读源码）。
    """

    answer: str = Field(..., description="基于子图生成的回答")
    subgraph: GraphSubgraph = Field(..., description="本次检索到的子图（三种方式的字段见 GraphSubgraph）")


@app.post("/api/graph_rag/query", response_model=GraphQueryResponse)
def graph_rag_query(req: GraphQueryRequest) -> dict:
    """GraphRAG 问答接口：检索子图 → 拼提示词 → LLM 生成回答。

    任何异常都收敛成 HTTP 500，把原因原文带在 detail 里，方便定位是 Neo4j 还是 LLM 挂了。

    同步 `def`（不是 async）是刻意的：FastAPI 会把同步视图丢到线程池执行，
    这条链路全是阻塞调用（embedding / LLM / Neo4j），用 async 反而会卡住事件循环。
    代价是 neomodel 的连接配置是**进程级全局**（见 models.connect），
    多线程首调时会重复写入同一份配置（幂等，无害）。
    `detail=str(exc)` 会把底层异常原文回给调用方（可能带 base_url 之类的内部信息）——
    内网演示可接受，对外暴露前应改成固定文案 + 服务端日志留原文。
    """
    try:
        if req.retrieval_method == "entity":
            # 实体级检索：max_nodes = top_k * 2
            subgraph = retrieve_by_entities(
                extract_query_entities(req.query),
                max_hops=req.max_hops,
                max_nodes=req.top_k * 2,
            )
        elif req.retrieval_method == "community":
            # 社区级：只返回社区摘要，不进实体层
            subgraph = {
                "communities": match_communities(req.query, top_k=req.top_k),
                "nodes": [],
                "relationships": [],
            }
        else:
            # 其它值（含 hybrid）统一走分层检索
            subgraph = retrieve_hierarchical(
                req.query,
                top_k=req.top_k,
                max_hops=req.max_hops,
                max_nodes=req.top_k * 2,
            )

        answer = answer_query(req.query, subgraph)
        logger.info(
            f"[GraphRAG] 服务查询完成 | 方式={req.retrieval_method} | "
            f"节点 {len(subgraph.get('nodes') or [])} 个 / "
            f"关系 {len(subgraph.get('relationships') or [])} 条"
        )
        return {"answer": answer, "subgraph": subgraph}
    except Exception as exc:  # noqa: BLE001 - 统一转 500，原文带回给调用方
        logger.exception(f"[GraphRAG] 查询失败: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    # 直接传 app 对象（reload 模式会在子进程里丢掉 sys.path 引导）
    # 另外这里用的是字符串 "graph_rag.service:app" 之外的形式，也避免了 reload
    # 在子进程里重新 import 一次模块（那会重建 OpenAI/Neo4j 客户端）。
    port = settings.app.port + 1
    logger.info(f"[GraphRAG] 服务启动: http://{settings.app.host}:{port}")
    uvicorn.run(app, host=settings.app.host, port=port)

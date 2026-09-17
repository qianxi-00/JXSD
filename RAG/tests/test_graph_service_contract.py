"""GraphRAG 服务的接口契约测试（优化篇 2495-2497：接口要有 QueryResponse）。

为什么值得单独钉：响应模型"加了但没挂到路由上"是这类改动的经典半成品 ——
代码里能看到 QueryResponse 类，OpenAPI 里却依然是空响应结构。
这里同时钉两件事：① 类存在且能校验；② **路由确实挂了它**。
"""

import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "RAG"))


def _service():
    from graph_rag import service

    return service


class TestResponseModelWiring:
    def test_route_has_response_model(self):
        service = _service()
        routes = [r for r in service.app.routes if getattr(r, "path", "") == "/api/graph_rag/query"]
        assert routes, "找不到 /api/graph_rag/query 路由"
        assert routes[0].response_model is service.GraphQueryResponse

    def test_openapi_documents_the_response(self):
        """/openapi.json 里必须出现响应字段（否则调用方只能猜）。"""
        service = _service()
        schema = service.app.openapi()
        resp = schema["paths"]["/api/graph_rag/query"]["post"]["responses"]["200"]
        ref = resp["content"]["application/json"]["schema"].get("$ref", "")
        assert ref.endswith("GraphQueryResponse"), f"响应结构没被文档化: {resp}"
        assert "GraphQueryResponse" in schema["components"]["schemas"]


class TestResponseModelTolerance:
    def test_accepts_hybrid_shape(self):
        service = _service()
        payload = {
            "answer": "答",
            "subgraph": {
                "communities": [{"community_id": 1, "summary": "s", "score": 0.7}],
                "nodes": [{"name": "董文", "entity_type": "人员", "description": "乘客", "community_id": 1}],
                "relationships": [{"source": "董文", "relation": "持有票据", "target": "T1"}],
            },
        }
        model = service.GraphQueryResponse.model_validate(payload)
        assert model.answer == "答"
        assert model.subgraph.nodes[0].name == "董文"

    def test_accepts_community_only_shape(self):
        """社区级检索只有 communities（没有 nodes/relationships）—— 不能因此 500。"""
        service = _service()
        model = service.GraphQueryResponse.model_validate(
            {"answer": "答", "subgraph": {"communities": [{"community_id": 0, "summary": "s"}]}}
        )
        assert model.subgraph.nodes == []
        assert model.subgraph.relationships == []

    def test_tolerates_extra_keys(self):
        """图谱 schema 将来加字段时，服务端不该因为多一个键就报错（extra=allow）。"""
        service = _service()
        model = service.GraphQueryResponse.model_validate(
            {"answer": "答", "subgraph": {"nodes": [{"name": "X", "extra_key": 1}]}}
        )
        assert model.subgraph.nodes[0].name == "X"
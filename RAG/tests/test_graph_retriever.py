"""graph_rag.retriever 与 graph_rag.service 的离线测试。

边界设计：Neo4j 访问统一走 `retriever.run_cypher`，LLM 走 `retriever.call_llm`，
向量化走 `retriever.embed_query`，因此本文件不连 Neo4j、不调外部 API。
（service 的用例也放在这里：交付清单里只允许 3 个新测试文件。）
"""

import json

import pytest
from fastapi.testclient import TestClient

from graph_rag import retriever, service


# ============================================================
# 一、查询实体抽取
# ============================================================
class TestExtractQueryEntities:
    def test_blank_query_skips_llm(self, monkeypatch):
        monkeypatch.setattr(
            retriever, "call_llm", lambda *a, **k: pytest.fail("空查询不应调用 LLM")
        )
        assert retriever.extract_query_entities("   ") == []
        assert retriever.extract_query_entities(None) == []

    def test_parses_entities(self, monkeypatch):
        seen = {}

        def fake_call_llm(system, user, temperature=0.1, json_mode=False):
            seen.update(user=user, temperature=temperature, json_mode=json_mode, system=system)
            return json.dumps({"entities": ["张三", " 李四 ", ""]})

        monkeypatch.setattr(retriever, "call_llm", fake_call_llm)
        names = retriever.extract_query_entities("张三给李四转了多少？")

        assert names == ["张三", "李四"]
        assert seen["temperature"] == 0.1
        assert seen["json_mode"] is True
        assert "张三给李四转了多少？" in seen["user"]

    def test_dedups_names(self, monkeypatch):
        monkeypatch.setattr(
            retriever,
            "call_llm",
            lambda *a, **k: json.dumps({"entities": ["张三", "张三"]}),
        )
        assert retriever.extract_query_entities("张三") == ["张三"]

    def test_bad_json_raises(self, monkeypatch):
        monkeypatch.setattr(retriever, "call_llm", lambda *a, **k: "not json")
        with pytest.raises(ValueError):
            retriever.extract_query_entities("张三")


# ============================================================
# 二、社区摘要召回
# ============================================================
class TestMatchCommunities:
    def test_queries_vector_index(self, monkeypatch):
        captured = {}

        def fake_run_cypher(query, params=None):
            captured.update(query=query, params=params)
            return [{"community_id": 3, "summary": "摘要", "score": 0.88}]

        monkeypatch.setattr(retriever, "run_cypher", fake_run_cypher)
        monkeypatch.setattr(retriever, "embed_query", lambda q: [0.1, 0.2])
        result = retriever.match_communities("张三的报销", top_k=2)

        from config import settings

        assert "db.index.vector.queryNodes" in captured["query"]
        assert captured["params"]["index"] == settings.neo4j.community_index
        assert captured["params"]["top_k"] == 2
        assert captured["params"]["vector"] == [0.1, 0.2]
        assert result == [{"community_id": 3, "summary": "摘要", "score": 0.88}]

    def test_empty_result(self, monkeypatch):
        monkeypatch.setattr(retriever, "run_cypher", lambda q, p=None: [])
        monkeypatch.setattr(retriever, "embed_query", lambda q: [0.1])
        assert retriever.match_communities("无关问题") == []


# ============================================================
# 三、实体向量召回
# ============================================================
class TestVectorMatchEntities:
    @pytest.fixture
    def fake_db(self, monkeypatch):
        calls = []

        def fake_run_cypher(query, params=None):
            calls.append({"query": query, "params": params})
            return [
                {"name": "张三", "entity_type": "人物", "description": "客户", "community_id": 1, "score": 0.8},
                {"name": "李四", "entity_type": "人物", "description": "员工", "community_id": 2, "score": 0.9},
            ]

        monkeypatch.setattr(retriever, "run_cypher", fake_run_cypher)
        monkeypatch.setattr(retriever, "embed_query", lambda name: [0.1, 0.2])
        return calls

    def test_global_search_has_no_community_filter(self, fake_db):
        results = retriever.vector_match_entities(["张三"], top_k=5)
        assert len(fake_db) == 1
        assert "IN $community_ids" not in fake_db[0]["query"]
        assert "community_ids" not in fake_db[0]["params"]
        assert [r["name"] for r in results] == ["李四", "张三"]  # 按相似度降序

    def test_community_filter_is_pushed_into_cypher(self, fake_db):
        retriever.vector_match_entities(["张三"], top_k=5, community_ids=[1, 2])
        assert "IN $community_ids" in fake_db[0]["query"]
        assert fake_db[0]["params"]["community_ids"] == [1, 2]

    def test_empty_community_list_is_still_pushed(self, fake_db):
        """community_ids=[] 会走 WHERE ... IN [] 恒不成立——这正是需要兜底的原因"""
        retriever.vector_match_entities(["张三"], top_k=5, community_ids=[])
        assert fake_db[0]["params"]["community_ids"] == []

    def test_dedups_by_name_keeping_best_score(self, monkeypatch):
        calls = {"n": 0}

        def fake_run_cypher(query, params=None):
            calls["n"] += 1
            score = 0.7 if calls["n"] == 1 else 0.95
            return [{"name": "张三", "entity_type": "人物", "description": "", "community_id": 1, "score": score}]

        monkeypatch.setattr(retriever, "run_cypher", fake_run_cypher)
        monkeypatch.setattr(retriever, "embed_query", lambda name: [0.1])
        results = retriever.vector_match_entities(["张三", "老张"], top_k=5)
        assert len(results) == 1
        assert results[0]["score"] == 0.95

    def test_skips_blank_names_without_embedding(self, monkeypatch):
        monkeypatch.setattr(
            retriever, "embed_query", lambda name: pytest.fail("空名字不应向量化")
        )
        monkeypatch.setattr(retriever, "run_cypher", lambda q, p=None: [])
        assert retriever.vector_match_entities(["", "  ", None]) == []

    def test_empty_names(self):
        assert retriever.vector_match_entities([]) == []


# ============================================================
# 四、子图收集（纯逻辑）
# ============================================================
class TestCollectSubgraph:
    def _paths(self):
        return [
            {
                "path_nodes": [
                    {"name": "张三", "entity_type": "人物", "description": "客户", "community_id": 1},
                    {"name": "李四", "entity_type": "人物", "description": "员工", "community_id": 1},
                ],
                "path_rels": [{"source": "张三", "relation": "转账", "target": "李四"}],
            },
            {
                "path_nodes": [
                    {"name": "张三", "entity_type": "人物", "description": "客户", "community_id": 1},
                    {"name": "王五", "entity_type": "人物", "description": "会计", "community_id": 1},
                ],
                "path_rels": [
                    {"source": "张三", "relation": "转账", "target": "李四"},  # 与上一条重复
                    {"source": "张三", "relation": "审批", "target": "王五"},
                ],
            },
        ]

    def test_dedups_nodes_and_relationships(self):
        subgraph = retriever.collect_subgraph(self._paths(), seeds=[], max_nodes=20)
        assert [n["name"] for n in subgraph["nodes"]] == ["张三", "李四", "王五"]
        keys = {f"{r['source']}-{r['relation']}-{r['target']}" for r in subgraph["relationships"]}
        assert keys == {"张三-转账-李四", "张三-审批-王五"}

    def test_seeds_come_first_and_survive_without_relations(self):
        """孤立实体（没有任何关系）也必须出现在子图里"""
        seed = {"name": "孤立实体", "entity_type": "人物", "description": "", "community_id": 9}
        subgraph = retriever.collect_subgraph(self._paths(), seeds=[seed], max_nodes=20)
        assert subgraph["nodes"][0]["name"] == "孤立实体"

    def test_max_nodes_caps_nodes_and_relations(self):
        subgraph = retriever.collect_subgraph(self._paths(), seeds=[], max_nodes=2)
        assert len(subgraph["nodes"]) == 2
        # 关系两端都必须落在保留的节点里，不能出现悬空关系
        names = {n["name"] for n in subgraph["nodes"]}
        assert all(r["source"] in names and r["target"] in names for r in subgraph["relationships"])
        assert subgraph["relationships"] == [{"source": "张三", "relation": "转账", "target": "李四"}]

    def test_empty_input(self):
        assert retriever.collect_subgraph([], seeds=[], max_nodes=20) == {
            "nodes": [],
            "relationships": [],
        }

    def test_tolerates_missing_keys(self):
        subgraph = retriever.collect_subgraph([{"path_nodes": None, "path_rels": None}], seeds=[], max_nodes=5)
        assert subgraph == {"nodes": [], "relationships": []}


# ============================================================
# 五、实体级多跳检索
# ============================================================
class TestRetrieveByEntities:
    def test_no_entities_returns_empty_without_db(self, monkeypatch):
        monkeypatch.setattr(
            retriever, "run_cypher", lambda q, p=None: pytest.fail("无实体不应访问数据库")
        )
        monkeypatch.setattr(
            retriever, "vector_match_entities", lambda *a, **k: pytest.fail("无实体不应向量检索")
        )
        assert retriever.retrieve_by_entities([]) == {"nodes": [], "relationships": []}
        assert retriever.retrieve_by_entities(None) == {"nodes": [], "relationships": []}

    def test_seed_search_is_not_community_limited(self, monkeypatch):
        seen = {}

        def fake_match(names, top_k=5, community_ids=None):
            seen["names"] = names
            seen["community_ids"] = community_ids
            return [{"name": "张三", "entity_type": "人物", "description": "", "community_id": 1}]

        monkeypatch.setattr(retriever, "vector_match_entities", fake_match)
        monkeypatch.setattr(retriever, "run_cypher", lambda q, p=None: [])
        result = retriever.retrieve_by_entities(["张三"], community_ids=[1])

        assert seen["community_ids"] is None  # 课案：先全局取 top_k，再按社区过滤
        assert [n["name"] for n in result["nodes"]] == ["张三"]

    def test_seeds_outside_community_are_dropped(self, monkeypatch):
        monkeypatch.setattr(
            retriever,
            "vector_match_entities",
            lambda *a, **k: [
                {"name": "张三", "entity_type": "人物", "description": "", "community_id": 7}
            ],
        )
        monkeypatch.setattr(
            retriever, "run_cypher", lambda q, p=None: pytest.fail("种子被过滤后不应再遍历")
        )
        assert retriever.retrieve_by_entities(["张三"], community_ids=[1]) == {
            "nodes": [],
            "relationships": [],
        }

    def test_traversal_cypher_is_community_scoped(self, monkeypatch):
        captured = {}

        def fake_run_cypher(query, params=None):
            captured.update(query=query, params=params)
            return []

        monkeypatch.setattr(
            retriever,
            "vector_match_entities",
            lambda *a, **k: [
                {"name": "张三", "entity_type": "人物", "description": "", "community_id": 1}
            ],
        )
        monkeypatch.setattr(retriever, "run_cypher", fake_run_cypher)
        retriever.retrieve_by_entities(["张三"], max_hops=3, community_ids=[1])

        assert "ALL" in captured["query"]
        assert "IN $community_ids" in captured["query"]
        assert "*1..3" in captured["query"]
        assert captured["params"]["community_ids"] == [1]
        assert captured["params"]["seed_names"] == ["张三"]

    def test_traversal_without_community_ids(self, monkeypatch):
        captured = {}

        def fake_run_cypher(query, params=None):
            captured.update(query=query, params=params)
            return []

        monkeypatch.setattr(
            retriever,
            "vector_match_entities",
            lambda *a, **k: [
                {"name": "张三", "entity_type": "人物", "description": "", "community_id": 1}
            ],
        )
        monkeypatch.setattr(retriever, "run_cypher", fake_run_cypher)
        retriever.retrieve_by_entities(["张三"], max_hops=2)

        assert "IN $community_ids" not in captured["query"]
        assert "community_ids" not in captured["params"]

    def test_builds_subgraph_from_paths(self, monkeypatch):
        monkeypatch.setattr(
            retriever,
            "vector_match_entities",
            lambda *a, **k: [
                {"name": "张三", "entity_type": "人物", "description": "客户", "community_id": 1}
            ],
        )
        monkeypatch.setattr(
            retriever,
            "run_cypher",
            lambda q, p=None: [
                {
                    "path_nodes": [
                        {"name": "张三", "entity_type": "人物", "description": "客户", "community_id": 1},
                        {"name": "李四", "entity_type": "人物", "description": "员工", "community_id": 1},
                    ],
                    "path_rels": [{"source": "张三", "relation": "转账", "target": "李四"}],
                }
            ],
        )
        result = retriever.retrieve_by_entities(["张三"], max_nodes=20)
        assert [n["name"] for n in result["nodes"]] == ["张三", "李四"]
        assert result["relationships"] == [{"source": "张三", "relation": "转账", "target": "李四"}]


# ============================================================
# 六、分层检索与兜底
# ============================================================
class TestRetrieveHierarchical:
    @pytest.fixture
    def spy(self, monkeypatch):
        calls = {}

        monkeypatch.setattr(retriever, "extract_query_entities", lambda q: ["张三"])

        def fake_retrieve(names, max_hops=2, max_nodes=20, community_ids=None):
            calls["names"] = names
            calls["max_hops"] = max_hops
            calls["max_nodes"] = max_nodes
            calls["community_ids"] = community_ids
            return {"nodes": [{"name": "张三"}], "relationships": []}

        monkeypatch.setattr(retriever, "retrieve_by_entities", fake_retrieve)
        return calls

    def test_falls_back_when_no_community_recalled(self, spy, monkeypatch):
        """课案点名的兜底：没召回社区时不能带着空 community_ids 去查"""
        monkeypatch.setattr(retriever, "match_communities", lambda q, top_k=2: [])
        result = retriever.retrieve_hierarchical("张三的报销", top_k=2)
        assert spy["community_ids"] is None
        assert result["communities"] == []
        assert [n["name"] for n in result["nodes"]] == ["张三"]

    def test_uses_recalled_communities(self, spy, monkeypatch):
        monkeypatch.setattr(
            retriever,
            "match_communities",
            lambda q, top_k=2: [{"community_id": 3, "summary": "s", "score": 0.9}],
        )
        result = retriever.retrieve_hierarchical("张三的报销", top_k=1, max_hops=2, max_nodes=10)
        assert spy["community_ids"] == [3]
        assert spy["max_nodes"] == 10
        assert result["communities"][0]["community_id"] == 3

    def test_passes_top_k_to_community_recall(self, spy, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            retriever,
            "match_communities",
            lambda q, top_k=2: seen.update(top_k=top_k) or [],
        )
        retriever.retrieve_hierarchical("问题", top_k=4)
        assert seen["top_k"] == 4

    def test_drops_communities_without_id(self, spy, monkeypatch):
        monkeypatch.setattr(
            retriever,
            "match_communities",
            lambda q, top_k=2: [{"community_id": None, "summary": "s", "score": 0.9}],
        )
        retriever.retrieve_hierarchical("问题")
        assert spy["community_ids"] is None


# ============================================================
# 七、RAG 提示词与回答生成
# ============================================================
class TestBuildRagPrompt:
    def _subgraph(self):
        return {
            "communities": [{"community_id": 3, "summary": "该社区围绕张三的报销"}],
            "nodes": [{"name": "张三", "entity_type": "人物", "description": "客户"}],
            "relationships": [{"source": "张三", "relation": "转账", "target": "李四"}],
        }

    def test_contains_all_sections(self):
        prompt = retriever.build_rag_prompt("张三报销了多少？", self._subgraph())
        assert "张三报销了多少？" in prompt
        assert "相关社区：" in prompt
        assert "- 社区 3：该社区围绕张三的报销" in prompt
        assert "相关实体：" in prompt
        assert "- 张三 (人物): 客户" in prompt
        assert "相关关系：" in prompt
        assert "- 张三 转账 李四" in prompt
        assert "如果知识图谱中没有相关信息，请明确说明" in prompt

    def test_empty_subgraph_keeps_fallback_hint(self):
        prompt = retriever.build_rag_prompt("无解问题", {"nodes": [], "relationships": []})
        assert "无解问题" in prompt
        assert "如果知识图谱中没有相关信息，请明确说明" in prompt
        assert "相关实体：" not in prompt

    def test_community_only_subgraph(self):
        prompt = retriever.build_rag_prompt(
            "问题", {"communities": [{"community_id": 1, "summary": "摘要"}], "nodes": [], "relationships": []}
        )
        assert "相关社区：" in prompt
        assert "相关实体：" not in prompt

    def test_tolerates_none_subgraph(self):
        prompt = retriever.build_rag_prompt("问题", None)
        assert "如果知识图谱中没有相关信息，请明确说明" in prompt


class TestAnswerQuery:
    def test_uses_rag_prompt(self, monkeypatch):
        seen = {}

        def fake_call_llm(system, user, temperature=0.2, json_mode=False):
            seen["user"] = user
            seen["temperature"] = temperature
            return "张三报销了 100 元。"

        monkeypatch.setattr(retriever, "call_llm", fake_call_llm)
        answer = retriever.answer_query(
            "张三报销了多少？",
            {"nodes": [{"name": "张三", "entity_type": "人物", "description": "客户"}]},
        )
        assert answer == "张三报销了 100 元。"
        assert "张三报销了多少？" in seen["user"]
        assert "- 张三 (人物): 客户" in seen["user"]


# ============================================================
# 八、FastAPI 服务
# ============================================================
class TestService:
    @pytest.fixture
    def client(self):
        return TestClient(service.app)

    def test_entity_method_uses_entity_retrieval(self, client, monkeypatch):
        called = {}
        monkeypatch.setattr(service, "extract_query_entities", lambda q: ["张三"])
        monkeypatch.setattr(service, "answer_query", lambda q, sg: f"答案：{q}")

        def fake_retrieve_by_entities(names, max_hops=2, max_nodes=20, community_ids=None):
            called.update(names=names, max_hops=max_hops, max_nodes=max_nodes)
            return {"nodes": [{"name": "张三"}], "relationships": []}

        monkeypatch.setattr(service, "retrieve_by_entities", fake_retrieve_by_entities)
        monkeypatch.setattr(
            service, "retrieve_hierarchical", lambda *a, **k: pytest.fail("不应走分层检索")
        )

        resp = client.post(
            "/api/graph_rag/query",
            json={"query": "张三的报销", "retrieval_method": "entity", "max_hops": 3, "top_k": 4},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["answer"] == "答案：张三的报销"
        assert body["subgraph"]["nodes"] == [{"name": "张三"}]
        assert called == {"names": ["张三"], "max_hops": 3, "max_nodes": 8}

    def test_community_method_returns_only_summaries(self, client, monkeypatch):
        monkeypatch.setattr(
            service,
            "match_communities",
            lambda q, top_k=2: [{"community_id": 1, "summary": "摘要", "score": 0.9}],
        )
        monkeypatch.setattr(service, "answer_query", lambda q, sg: "只看社区摘要的回答")
        monkeypatch.setattr(
            service, "retrieve_hierarchical", lambda *a, **k: pytest.fail("不应走分层检索")
        )
        monkeypatch.setattr(
            service, "extract_query_entities", lambda q: pytest.fail("community 方法不需要实体抽取")
        )

        resp = client.post(
            "/api/graph_rag/query",
            json={"query": "财务概况", "retrieval_method": "community", "top_k": 2},
        )
        assert resp.status_code == 200
        subgraph = resp.json()["subgraph"]
        assert subgraph["communities"][0]["community_id"] == 1
        assert subgraph["nodes"] == []
        assert subgraph["relationships"] == []

    def test_hybrid_method_uses_hierarchical(self, client, monkeypatch):
        seen = {}
        monkeypatch.setattr(service, "answer_query", lambda q, sg: "分层回答")

        def fake_hierarchical(query, top_k=2, max_hops=2, max_nodes=20):
            seen.update(query=query, top_k=top_k, max_hops=max_hops, max_nodes=max_nodes)
            return {"communities": [{"community_id": 1, "summary": "s", "score": 0.9}], "nodes": [], "relationships": []}

        monkeypatch.setattr(service, "retrieve_hierarchical", fake_hierarchical)
        resp = client.post(
            "/api/graph_rag/query",
            json={"query": "张三的报销", "retrieval_method": "hybrid", "max_hops": 2, "top_k": 3},
        )
        assert resp.status_code == 200
        assert resp.json()["answer"] == "分层回答"
        assert seen == {"query": "张三的报销", "top_k": 3, "max_hops": 2, "max_nodes": 6}

    def test_unknown_method_defaults_to_hierarchical(self, client, monkeypatch):
        monkeypatch.setattr(service, "answer_query", lambda q, sg: "兜底")
        monkeypatch.setattr(
            service,
            "retrieve_hierarchical",
            lambda *a, **k: {"communities": [], "nodes": [], "relationships": []},
        )
        resp = client.post(
            "/api/graph_rag/query", json={"query": "问题", "retrieval_method": "whatever"}
        )
        assert resp.status_code == 200
        assert resp.json()["answer"] == "兜底"

    def test_default_method_is_hybrid(self, client, monkeypatch):
        monkeypatch.setattr(service, "answer_query", lambda q, sg: "默认")
        monkeypatch.setattr(
            service,
            "retrieve_hierarchical",
            lambda *a, **k: {"communities": [], "nodes": [], "relationships": []},
        )
        resp = client.post("/api/graph_rag/query", json={"query": "问题"})
        assert resp.status_code == 200

    def test_errors_become_500(self, client, monkeypatch):
        def boom(*args, **kwargs):
            raise RuntimeError("Neo4j 连接失败")

        monkeypatch.setattr(service, "retrieve_hierarchical", boom)
        resp = client.post("/api/graph_rag/query", json={"query": "问题"})
        assert resp.status_code == 500
        assert "Neo4j 连接失败" in resp.json()["detail"]

    def test_missing_query_is_422(self, client):
        resp = client.post("/api/graph_rag/query", json={"retrieval_method": "entity"})
        assert resp.status_code == 422

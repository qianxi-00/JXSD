"""graph_rag.builder 的离线测试。

边界设计：外部依赖（LLM / Embedding / Neo4j）都被收敛成模块级可替换的名字
（`call_llm` / `embed_query` / `run_cypher` / `Entity` / `Community`），
因此本文件里的用例不连 Neo4j、不调任何外部 API。
"""

import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import networkx as nx
import pytest

from config import settings
from graph_rag import builder, models, retriever

# 建图脚本不在 tests/ 下，也不是包，按文件路径加载
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "script" / "build_finance_graph.py"


def _load_build_script():
    spec = importlib.util.spec_from_file_location("build_finance_graph", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_script = _load_build_script()


# ============================================================
# 假对象：模拟 neomodel 的节点/关系管理器
# ============================================================
class FakeRel:
    def __init__(self, props):
        self.relation_type = props.get("relation_type", "RELATES_TO")
        self.confidence = props.get("confidence", 1.0)
        self.weight = props.get("weight", 1)
        self.doc_ids = list(props.get("doc_ids") or [])
        self.saved = False

    def save(self):
        self.saved = True
        return self


class FakeManager:
    """模拟 instance.relates_to 的 RelationshipManager"""

    def __init__(self, node):
        self._node = node

    def relationship(self, target):
        return self._node._rels.get(target.name)

    def connect(self, target, props):
        rel = FakeRel(props)
        self._node._rels[target.name] = rel
        return rel


class FakeNodeSet:
    def __init__(self, owner, key):
        self._owner = owner
        self._key = key

    def get_or_none(self, **kwargs):
        return self._owner.store.get(kwargs.get(self._key))


class FakeEntity:
    store: dict = {}

    def __init__(self, **kwargs):
        self.name = kwargs.get("name")
        self.entity_type = kwargs.get("entity_type", "UNKNOWN")
        self.description = kwargs.get("description", "")
        self.doc_ids = list(kwargs.get("doc_ids") or [])
        self.mentions = kwargs.get("mentions", 1)
        self.community_id = kwargs.get("community_id", -1)
        self.embedding = list(kwargs.get("embedding") or [])
        self._rels: dict = {}
        self.relates_to = FakeManager(self)
        self.saved = False
        if self.name:
            FakeEntity.store[self.name] = self

    def save(self):
        self.saved = True
        FakeEntity.store[self.name] = self
        return self


FakeEntity.nodes = FakeNodeSet(FakeEntity, "name")


class FakeCommunity:
    store: dict = {}

    def __init__(self, **kwargs):
        self.community_id = kwargs.get("community_id")
        self.summary = kwargs.get("summary", "")
        self.embedding = list(kwargs.get("embedding") or [])
        self.saved = False

    def save(self):
        self.saved = True
        FakeCommunity.store[self.community_id] = self
        return self


FakeCommunity.nodes = FakeNodeSet(FakeCommunity, "community_id")


@pytest.fixture
def fake_models(monkeypatch):
    """把 builder 里的 Entity / Community / embed_query 换成假对象"""
    FakeEntity.store = {}
    FakeCommunity.store = {}
    FakeEntity.nodes = FakeNodeSet(FakeEntity, "name")
    FakeCommunity.nodes = FakeNodeSet(FakeCommunity, "community_id")
    monkeypatch.setattr(builder, "Entity", FakeEntity)
    monkeypatch.setattr(builder, "Community", FakeCommunity)
    monkeypatch.setattr(builder, "embed_query", lambda text: [0.1, 0.2, 0.3])
    monkeypatch.setattr(builder, "connect", lambda: None)
    return FakeEntity, FakeCommunity


# ============================================================
# 一、LLM JSON 解析与抽取结果归一化（纯逻辑）
# ============================================================
class TestLoadJsonObject:
    def test_plain_json(self):
        assert builder.load_json_object('{"a": 1}') == {"a": 1}

    def test_strips_markdown_code_fence(self):
        raw = '```json\n{"entities": [], "relations": []}\n```'
        assert builder.load_json_object(raw) == {"entities": [], "relations": []}

    def test_strips_bare_code_fence(self):
        assert builder.load_json_object('```\n{"a": 1}\n```') == {"a": 1}

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            builder.load_json_object("")

    def test_rejects_none(self):
        with pytest.raises(ValueError):
            builder.load_json_object(None)

    def test_rejects_non_json_text(self):
        with pytest.raises(ValueError):
            builder.load_json_object("这是一个实体列表")

    def test_rejects_json_array(self):
        """顶层必须是对象：数组拿不到 entities/relations 字段"""
        with pytest.raises(ValueError):
            builder.load_json_object("[1, 2, 3]")


class TestNormalizeExtraction:
    def test_renames_type_to_entity_type(self):
        data = builder.normalize_extraction(
            {"entities": [{"name": "张三", "type": "人物", "description": "客户"}]}
        )
        assert data["entities"] == [
            {"name": "张三", "entity_type": "人物", "description": "客户"}
        ]
        assert data["relations"] == []

    def test_strips_whitespace(self):
        data = builder.normalize_extraction(
            {"entities": [{"name": "  张三  ", "type": " 人物 ", "description": " x "}]}
        )
        assert data["entities"][0]["name"] == "张三"
        assert data["entities"][0]["entity_type"] == "人物"

    def test_drops_entities_without_name(self):
        data = builder.normalize_extraction(
            {"entities": [{"type": "人物"}, {"name": "", "type": "x"}, {"name": "李四"}]}
        )
        assert [e["name"] for e in data["entities"]] == ["李四"]

    def test_dedups_entities_by_name(self):
        data = builder.normalize_extraction(
            {
                "entities": [
                    {"name": "张三", "type": "人物", "description": "客户"},
                    {"name": "张三", "type": "人物", "description": "员工"},
                ]
            }
        )
        assert len(data["entities"]) == 1

    def test_defaults_type_and_description(self):
        data = builder.normalize_extraction({"entities": [{"name": "张三"}]})
        assert data["entities"][0]["entity_type"] == "UNKNOWN"
        assert data["entities"][0]["description"] == ""

    def test_keeps_valid_relations(self):
        data = builder.normalize_extraction(
            {"relations": [{"source": "张三", "target": "李四", "relation": "转账"}]}
        )
        assert data["relations"] == [{"source": "张三", "target": "李四", "relation": "转账"}]

    def test_drops_relations_missing_endpoints(self):
        data = builder.normalize_extraction(
            {
                "relations": [
                    {"source": "", "target": "李四", "relation": "转账"},
                    {"source": "张三", "relation": "转账"},
                    {"source": "张三", "target": "李四"},
                ]
            }
        )
        assert len(data["relations"]) == 1
        assert data["relations"][0]["relation"] == "RELATES_TO"

    def test_tolerates_missing_keys(self):
        assert builder.normalize_extraction({}) == {"entities": [], "relations": []}

    def test_tolerates_none_payload(self):
        assert builder.normalize_extraction(None) == {"entities": [], "relations": []}


class TestExtractEntities:
    def test_blank_text_skips_llm(self, monkeypatch):
        def boom(*args, **kwargs):  # pragma: no cover - 不应被调用
            raise AssertionError("空文本不应调用 LLM")

        monkeypatch.setattr(builder, "call_llm", boom)
        assert builder.extract_entities("   ") == {"entities": [], "relations": []}
        assert builder.extract_entities("") == {"entities": [], "relations": []}
        assert builder.extract_entities(None) == {"entities": [], "relations": []}

    def test_parses_llm_json(self, monkeypatch):
        seen = {}

        def fake_call_llm(system, user, temperature=0.1, json_mode=False):
            seen.update(system=system, user=user, temperature=temperature, json_mode=json_mode)
            return json.dumps(
                {
                    "entities": [{"name": "张三", "type": "人物", "description": "客户"}],
                    "relations": [{"source": "张三", "target": "李四", "relation": "转账"}],
                }
            )

        monkeypatch.setattr(builder, "call_llm", fake_call_llm)
        data = builder.extract_entities("张三向李四转账")

        assert seen["system"] == "你是一个专业的实体和关系提取助手。"
        assert seen["temperature"] == 0.1
        assert seen["json_mode"] is True
        assert "张三向李四转账" in seen["user"]
        assert "entities" in seen["user"] and "relations" in seen["user"]
        assert data["entities"][0]["name"] == "张三"

    def test_raises_on_bad_json(self, monkeypatch):
        monkeypatch.setattr(builder, "call_llm", lambda *a, **k: "不是 JSON")
        with pytest.raises(ValueError):
            builder.extract_entities("任意文本")


# ============================================================
# 二、合并规则（纯逻辑）
# ============================================================
class TestMergeRules:
    def test_merge_confidence_weighted_average(self):
        # (0.8*3 + 1.0) / 4 = 0.85
        assert builder.merge_confidence(0.8, 3, 1.0) == 0.85

    def test_merge_confidence_first_time(self):
        assert builder.merge_confidence(0.9, 1, 0.7) == 0.8

    def test_merge_confidence_treats_missing_weight_as_one(self):
        assert builder.merge_confidence(0.6, None, 1.0) == 0.8

    def test_merge_doc_ids_dedups_keeping_order(self):
        assert builder.merge_doc_ids(["d1", "d2"], ["d2", "d3"]) == ["d1", "d2", "d3"]

    def test_merge_doc_ids_handles_none(self):
        assert builder.merge_doc_ids(None, ["d1"]) == ["d1"]
        assert builder.merge_doc_ids(["d1"], None) == ["d1"]
        assert builder.merge_doc_ids(None, None) == []

    def test_merge_description_appends_with_separator(self):
        assert builder.merge_description("客户", "员工") == "客户；员工"

    def test_merge_description_keeps_old_when_new_empty(self):
        assert builder.merge_description("客户", "") == "客户"
        assert builder.merge_description("客户", None) == "客户"

    def test_merge_description_skips_duplicate(self):
        assert builder.merge_description("客户", "客户") == "客户"
        assert builder.merge_description("客户 详情", "客户") == "客户 详情"

    def test_merge_description_fills_empty_old(self):
        assert builder.merge_description("", "客户") == "客户"
        assert builder.merge_description(None, "客户") == "客户"
        assert builder.merge_description(None, None) == ""

    def test_merged_entity_state(self, fake_models):
        FakeEntity, _ = fake_models
        node = FakeEntity(name="张三", description="客户", doc_ids=["d1"], mentions=2)
        state = builder.merged_entity_state(node, "员工", "d2")
        assert state == {"description": "客户；员工", "doc_ids": ["d1", "d2"], "mentions": 3}

    def test_merged_entity_state_without_doc_id(self, fake_models):
        FakeEntity, _ = fake_models
        node = FakeEntity(name="张三", description="客户", doc_ids=["d1"], mentions=1)
        state = builder.merged_entity_state(node, "", None)
        assert state == {"description": "客户", "doc_ids": ["d1"], "mentions": 2}

    def test_merge_relation_props_new_weight(self):
        rel = FakeRel({"confidence": 0.8, "weight": 3, "doc_ids": ["d1"]})
        merged = builder.merge_relation_props(rel, confidence=1.0, doc_id="d2")
        assert merged == {"confidence": 0.85, "weight": 4, "doc_ids": ["d1", "d2"]}

    def test_merge_relation_props_without_doc_id(self):
        rel = FakeRel({"confidence": 0.8, "weight": 1, "doc_ids": ["d1"]})
        merged = builder.merge_relation_props(rel, confidence=0.8, doc_id=None)
        assert merged == {"confidence": 0.8, "weight": 2, "doc_ids": ["d1"]}


# ============================================================
# 三、建图（假 neomodel 模型）
# ============================================================
class TestCreateGraph:
    def test_clears_graph_before_building(self, fake_models, monkeypatch):
        queries = []
        monkeypatch.setattr(
            builder, "run_cypher", lambda q, p=None: queries.append(q) or []
        )
        builder.create_graph([], [])
        assert queries and "DETACH DELETE" in queries[0]

    def test_creates_entities_with_embedding(self, fake_models, monkeypatch):
        FakeEntity, _ = fake_models
        monkeypatch.setattr(builder, "run_cypher", lambda q, p=None: [])
        builder.create_graph(
            [{"name": "张三", "type": "人物", "description": "客户"}], []
        )
        node = FakeEntity.store["张三"]
        assert node.entity_type == "人物"
        assert node.description == "客户"
        assert node.embedding == [0.1, 0.2, 0.3]
        assert node.saved is True

    def test_repeated_entity_accumulates_mentions_and_doc_ids(
        self, fake_models, monkeypatch
    ):
        FakeEntity, _ = fake_models
        monkeypatch.setattr(builder, "run_cypher", lambda q, p=None: [])
        builder.create_graph(
            [
                {"name": "张三", "type": "人物", "description": "客户", "doc_ids": ["d1"]},
                {"name": "张三", "type": "人物", "description": "员工", "doc_ids": ["d1", "d2"]},
            ],
            [],
        )
        assert len(FakeEntity.store) == 1
        node = FakeEntity.store["张三"]
        assert node.mentions == 2
        assert node.doc_ids == ["d1", "d2"]
        assert node.description == "客户；员工"

    def test_creates_new_relationship_with_weight_one(self, fake_models, monkeypatch):
        FakeEntity, _ = fake_models
        monkeypatch.setattr(builder, "run_cypher", lambda q, p=None: [])
        builder.create_graph(
            [{"name": "张三", "type": "人物"}, {"name": "李四", "type": "人物"}],
            [{"source": "张三", "target": "李四", "relation": "转账"}],
        )
        rel = FakeEntity.store["张三"]._rels["李四"]
        assert rel.relation_type == "转账"
        assert rel.confidence == 1.0
        assert rel.weight == 1

    def test_existing_relationship_is_merged_by_weight(self, fake_models, monkeypatch):
        FakeEntity, _ = fake_models
        monkeypatch.setattr(builder, "run_cypher", lambda q, p=None: [])
        builder.create_graph(
            [{"name": "张三", "type": "人物"}, {"name": "李四", "type": "人物"}],
            [
                {"source": "张三", "target": "李四", "relation": "转账", "confidence": 0.8},
                {"source": "张三", "target": "李四", "relation": "转账", "confidence": 1.0},
            ],
        )
        rel = FakeEntity.store["张三"]._rels["李四"]
        assert rel.weight == 2
        assert rel.confidence == 0.9  # (0.8*1 + 1.0) / 2
        assert rel.saved is True

    def test_skips_relation_with_missing_endpoint(self, fake_models, monkeypatch):
        FakeEntity, _ = fake_models
        captured = []
        monkeypatch.setattr(
            builder, "run_cypher", lambda q, p=None: captured.append(q) or []
        )
        stats = builder.create_graph(
            [{"name": "张三", "type": "人物"}],
            [{"source": "张三", "target": "查无此人", "relation": "转账"}],
        )
        assert stats["relations"] == 0
        assert FakeEntity.store["张三"]._rels == {}

    def test_returns_stats(self, fake_models, monkeypatch):
        monkeypatch.setattr(builder, "run_cypher", lambda q, p=None: [])
        stats = builder.create_graph(
            [{"name": "张三", "type": "人物"}, {"name": "李四", "type": "人物"}],
            [{"source": "张三", "target": "李四", "relation": "转账"}],
        )
        assert stats == {"entities": 2, "relations": 1}


# ============================================================
# 四、社区检测（networkx + louvain）
# ============================================================
class TestBuildNetworkxGraph:
    def test_sums_parallel_edge_weights(self):
        graph = builder.build_networkx_graph(
            [
                {"source": "a", "target": "b", "weight": 2},
                {"source": "a", "target": "b", "weight": 3},
            ]
        )
        assert isinstance(graph, nx.Graph)
        assert graph["a"]["b"]["weight"] == 5

    def test_keeps_isolated_nodes(self):
        graph = builder.build_networkx_graph([{"source": "c", "target": None, "weight": None}])
        assert list(graph.nodes) == ["c"]
        assert graph.number_of_edges() == 0

    def test_treats_missing_weight_as_one(self):
        graph = builder.build_networkx_graph([{"source": "a", "target": "b", "weight": None}])
        assert graph["a"]["b"]["weight"] == 1

    def test_empty_records(self):
        assert builder.build_networkx_graph([]).number_of_nodes() == 0

    def test_skips_records_without_source(self):
        graph = builder.build_networkx_graph([{"source": None, "target": "b", "weight": 1}])
        assert graph.number_of_nodes() == 0


class TestDetectCommunitiesLouvain:
    def test_returns_empty_on_empty_graph(self, monkeypatch):
        monkeypatch.setattr(builder, "run_cypher", lambda q, p=None: [])
        assert builder.detect_communities_louvain() == {}

    def test_splits_two_disjoint_triangles(self, monkeypatch):
        records = [
            {"source": "a1", "target": "a2", "weight": 1},
            {"source": "a2", "target": "a3", "weight": 1},
            {"source": "a3", "target": "a1", "weight": 1},
            {"source": "b1", "target": "b2", "weight": 1},
            {"source": "b2", "target": "b3", "weight": 1},
            {"source": "b3", "target": "b1", "weight": 1},
        ]
        writes = []

        def fake_run_cypher(query, params=None):
            writes.append((query, params))
            return records if len(writes) == 1 else []

        monkeypatch.setattr(builder, "run_cypher", fake_run_cypher)
        partition = builder.detect_communities_louvain()

        assert len(set(partition.values())) == 2
        assert set(partition) == {"a1", "a2", "a3", "b1", "b2", "b3"}
        assert partition["a1"] == partition["a2"] == partition["a3"]

    def test_writes_community_id_back_to_entities(self, monkeypatch):
        records = [{"source": "a", "target": "b", "weight": 1}]
        writes = []

        def fake_run_cypher(query, params=None):
            writes.append((query, params))
            return records if len(writes) == 1 else []

        monkeypatch.setattr(builder, "run_cypher", fake_run_cypher)
        builder.detect_communities_louvain(resolution=1.5)

        assert len(writes) == 2
        write_query, params = writes[1]
        assert "UNWIND $rows" in write_query
        assert "SET n.community_id = row.community_id" in write_query
        assert {r["name"] for r in params["rows"]} == {"a", "b"}

    def test_resolution_changes_partition(self, monkeypatch):
        """resolution 要真正透传给 louvain"""
        seen = {}
        import community as community_lib

        original = community_lib.best_partition

        def spy(graph, **kwargs):
            seen.update(kwargs)
            return original(graph, **kwargs)

        monkeypatch.setattr(builder.community, "best_partition", spy)
        monkeypatch.setattr(
            builder,
            "run_cypher",
            lambda q, p=None: [{"source": "a", "target": "b", "weight": 1}],
        )
        builder.detect_communities_louvain(resolution=2.0)
        assert seen["resolution"] == 2.0
        assert seen["weight"] == "weight"


# ============================================================
# 五、社区摘要
# ============================================================
class TestGroupByCommunity:
    def test_groups_entities(self):
        grouped = builder.group_entities_by_community(
            [
                {"name": "张三", "community_id": 1},
                {"name": "李四", "community_id": 1},
                {"name": "王五", "community_id": 2},
            ]
        )
        assert set(grouped) == {1, 2}
        assert [e["name"] for e in grouped[1]] == ["张三", "李四"]

    def test_skips_unassigned(self):
        grouped = builder.group_entities_by_community(
            [
                {"name": "张三", "community_id": None},
                {"name": "李四", "community_id": -1},
                {"name": "王五", "community_id": 0},
            ]
        )
        assert set(grouped) == {0}


class TestCommunityMaterial:
    def test_contains_id_entities_and_relations(self):
        material = builder.build_community_material(
            3,
            [
                {"name": "张三", "entity_type": "人物", "description": "客户"},
                {"name": "李四", "entity_type": "人物", "description": ""},
            ],
            [{"source": "张三", "relation": "转账", "target": "李四"}],
        )
        assert "社区 ID：3" in material
        assert "- 张三 (人物): 客户" in material
        assert "- 李四 (人物): " in material
        assert "- 张三 转账 李四" in material

    def test_marks_empty_sections(self):
        material = builder.build_community_material(5, [], [])
        assert "社区 ID：5" in material
        assert "（无）" in material

    def test_format_entity_lines(self):
        assert builder.format_entity_lines(
            [{"name": "张三", "entity_type": "人物", "description": "客户"}]
        ) == "- 张三 (人物): 客户"

    def test_format_relation_lines(self):
        assert builder.format_relation_lines(
            [{"source": "张三", "relation": "转账", "target": "李四"}]
        ) == "- 张三 转账 李四"


class TestGenerateCommunitySummaries:
    def test_writes_summary_and_embedding(self, fake_models, monkeypatch):
        _, FakeCommunity = fake_models
        entity_rows = [
            {"name": "张三", "entity_type": "人物", "description": "客户", "community_id": 0},
            {"name": "李四", "entity_type": "人物", "description": "客户", "community_id": 0},
        ]
        relation_rows = [
            {"source": "张三", "relation": "转账", "target": "李四", "community_id": 0}
        ]
        calls = {"n": 0}

        def fake_run_cypher(query, params=None):
            calls["n"] += 1
            return entity_rows if calls["n"] == 1 else relation_rows

        prompts = []

        def fake_call_llm(system, user, temperature=0.3, json_mode=False):
            prompts.append((system, user, temperature))
            return "本社区围绕张三与李四的转账关系。"

        monkeypatch.setattr(builder, "run_cypher", fake_run_cypher)
        monkeypatch.setattr(builder, "call_llm", fake_call_llm)

        result = builder.generate_community_summaries()

        assert result == [
            {"community_id": 0, "summary": "本社区围绕张三与李四的转账关系。"}
        ]
        node = FakeCommunity.store[0]
        assert node.summary == "本社区围绕张三与李四的转账关系。"
        assert node.embedding == [0.1, 0.2, 0.3]
        assert prompts[0][0] == "你是一个专业的文本摘要助手。"
        assert prompts[0][2] == 0.3
        assert "社区 ID：0" in prompts[0][1]

    def test_empty_graph_returns_empty(self, fake_models, monkeypatch):
        monkeypatch.setattr(builder, "run_cypher", lambda q, p=None: [])
        monkeypatch.setattr(
            builder, "call_llm", lambda *a, **k: pytest.fail("无社区时不应调用 LLM")
        )
        assert builder.generate_community_summaries() == []


# ============================================================
# 六、增量入库（LLM Wiki）
# ============================================================
class TestParseMatchChoice:
    def test_exact_candidate(self):
        assert builder.parse_match_choice("张三", ["张三", "李四"]) == "张三"

    def test_strips_quotes_and_punctuation(self):
        assert builder.parse_match_choice('"张三"。', ["张三", "李四"]) == "张三"

    def test_none_token(self):
        assert builder.parse_match_choice("NONE", ["张三"]) is None

    def test_lowercase_none_token(self):
        assert builder.parse_match_choice("none", ["张三"]) is None

    def test_unknown_name_is_rejected(self):
        assert builder.parse_match_choice("王五", ["张三", "李四"]) is None

    def test_empty_answer(self):
        assert builder.parse_match_choice("", ["张三"]) is None
        assert builder.parse_match_choice(None, ["张三"]) is None

    def test_no_candidates(self):
        assert builder.parse_match_choice("张三", []) is None


class TestMatchEntity:
    def test_exact_name_hit_skips_vector_search(self, fake_models, monkeypatch):
        FakeEntity, _ = fake_models
        FakeEntity(name="张三", entity_type="人物")

        def boom(*args, **kwargs):  # pragma: no cover
            raise AssertionError("名称精确命中时不应再走向量召回")

        monkeypatch.setattr(builder, "run_cypher", boom)
        monkeypatch.setattr(builder, "call_llm", boom)
        assert builder.match_entity("张三", "人物").name == "张三"

    def test_vector_candidate_confirmed_by_llm(self, fake_models, monkeypatch):
        FakeEntity, _ = fake_models
        FakeEntity(name="张三", entity_type="人物")
        monkeypatch.setattr(
            builder,
            "run_cypher",
            lambda q, p=None: [
                {"name": "张三", "entity_type": "人物", "description": "客户", "score": 0.9}
            ],
        )
        monkeypatch.setattr(builder, "call_llm", lambda *a, **k: "张三")
        assert builder.match_entity("老张", "人物").name == "张三"

    def test_low_score_is_filtered(self, fake_models, monkeypatch):
        monkeypatch.setattr(
            builder,
            "run_cypher",
            lambda q, p=None: [
                {"name": "张三", "entity_type": "人物", "description": "", "score": 0.59}
            ],
        )
        monkeypatch.setattr(
            builder, "call_llm", lambda *a, **k: pytest.fail("低于阈值不应调用 LLM")
        )
        assert builder.match_entity("老张", "人物") is None

    def test_different_entity_type_is_filtered(self, fake_models, monkeypatch):
        monkeypatch.setattr(
            builder,
            "run_cypher",
            lambda q, p=None: [
                {"name": "张三", "entity_type": "公司", "description": "", "score": 0.99}
            ],
        )
        monkeypatch.setattr(
            builder, "call_llm", lambda *a, **k: pytest.fail("类型不同不应调用 LLM")
        )
        assert builder.match_entity("老张", "人物") is None

    def test_llm_says_none(self, fake_models, monkeypatch):
        monkeypatch.setattr(
            builder,
            "run_cypher",
            lambda q, p=None: [
                {"name": "张三", "entity_type": "人物", "description": "", "score": 0.9}
            ],
        )
        monkeypatch.setattr(builder, "call_llm", lambda *a, **k: "NONE")
        assert builder.match_entity("老张", "人物") is None

    def test_no_candidates_returns_none(self, fake_models, monkeypatch):
        monkeypatch.setattr(builder, "run_cypher", lambda q, p=None: [])
        monkeypatch.setattr(
            builder, "call_llm", lambda *a, **k: pytest.fail("无候选不应调用 LLM")
        )
        assert builder.match_entity("老张", "人物") is None

    def test_match_prompt_structure(self, fake_models, monkeypatch):
        FakeEntity, _ = fake_models
        FakeEntity(name="张三", entity_type="人物")
        monkeypatch.setattr(
            builder,
            "run_cypher",
            lambda q, p=None: [
                {"name": "张三", "entity_type": "人物", "description": "客户", "score": 0.9},
                {"name": "李四", "entity_type": "人物", "description": "员工", "score": 0.8},
            ],
        )
        seen = {}

        def fake_call_llm(system, user, temperature=0, json_mode=False):
            seen.update(user=user, temperature=temperature, system=system)
            return "张三"

        monkeypatch.setattr(builder, "call_llm", fake_call_llm)
        builder.match_entity("老张", "人物")
        assert seen["temperature"] == 0
        assert "老张" in seen["user"]
        assert "张三" in seen["user"] and "李四" in seen["user"]
        assert "NONE" in seen["user"]


class TestLinkOrCreate:
    def test_creates_new_entity_with_embedding(self, fake_models, monkeypatch):
        FakeEntity, _ = fake_models
        monkeypatch.setattr(builder, "run_cypher", lambda q, p=None: [])
        node = builder.link_or_create(
            {"name": "张三", "type": "人物", "description": "客户"}, "doc-1"
        )
        assert node.name == "张三"
        assert node.embedding == [0.1, 0.2, 0.3]
        assert node.doc_ids == ["doc-1"]
        assert node.mentions == 1

    def test_merges_into_existing_entity(self, fake_models, monkeypatch):
        FakeEntity, _ = fake_models
        FakeEntity(name="张三", entity_type="人物", description="客户", doc_ids=["doc-1"], mentions=1)
        monkeypatch.setattr(builder, "run_cypher", lambda q, p=None: [])
        node = builder.link_or_create(
            {"name": "张三", "type": "人物", "description": "员工"}, "doc-2"
        )
        assert node.description == "客户；员工"
        assert node.doc_ids == ["doc-1", "doc-2"]
        assert node.mentions == 2


class TestIngestDocument:
    def test_extracts_then_links(self, fake_models, monkeypatch):
        FakeEntity, _ = fake_models
        monkeypatch.setattr(
            builder,
            "extract_entities",
            lambda text: {
                "entities": [
                    {"name": "张三", "entity_type": "人物", "description": ""},
                    {"name": "李四", "entity_type": "人物", "description": ""},
                ],
                "relations": [{"source": "张三", "target": "李四", "relation": "转账"}],
            },
        )
        monkeypatch.setattr(builder, "run_cypher", lambda q, p=None: [])
        stats = builder.ingest_document("张三向李四转账", "doc-1")
        assert stats == {"entities": 2, "relations": 1}
        assert FakeEntity.store["张三"].doc_ids == ["doc-1"]
        assert FakeEntity.store["张三"]._rels["李四"].relation_type == "转账"

    def test_blank_text_is_noop(self, fake_models, monkeypatch):
        monkeypatch.setattr(
            builder, "extract_entities", lambda text: {"entities": [], "relations": []}
        )
        monkeypatch.setattr(builder, "run_cypher", lambda q, p=None: [])
        assert builder.ingest_document("", "doc-1") == {"entities": 0, "relations": 0}

    def test_relation_endpoints_use_canonical_node_names(self, fake_models, monkeypatch):
        """消歧后关系端点必须换成图里的真实名称，否则关系会被端点缺失整条丢掉"""
        FakeEntity, _ = fake_models
        canonical = FakeEntity(name="ZHANGSAN", entity_type="人物", description="客户")
        monkeypatch.setattr(
            builder,
            "match_entity",
            lambda name, entity_type=None: canonical if name == "张三" else None,
        )
        monkeypatch.setattr(builder, "run_cypher", lambda q, p=None: [])
        monkeypatch.setattr(
            builder,
            "extract_entities",
            lambda text: {
                "entities": [
                    {"name": "张三", "entity_type": "人物", "description": ""},
                    {"name": "李四", "entity_type": "人物", "description": ""},
                ],
                # 关系用"张三"这个抽取名引用它，应该被解析成 ZHANGSAN
                "relations": [{"source": "张三", "target": "李四", "relation": "转账"}],
            },
        )

        stats = builder.ingest_document("张三向李四转账", "doc-1")

        assert stats == {"entities": 2, "relations": 1}
        assert "李四" in canonical._rels
        assert "张三" not in canonical._rels


# ============================================================
# 建图脚本（RAG/script/build_finance_graph.py）
# ============================================================
class TestBuildScript:
    """脚本里的纯逻辑与编排：不连 Milvus、不调 LLM"""

    def test_ticket_text_includes_structured_fields(self):
        text = build_script.ticket_text(
            {
                "semantic_text": "机票报销",
                "ticket_type": "flight",
                "ticket_no": "T001",
                "person": "张三",
                "amount_fen": 123456,
                "source_file": "data/flight/ticket-001.png",
            }
        )
        assert "机票报销" in text
        assert "票据类型：flight" in text
        assert "票据号：T001" in text
        assert "人员：张三" in text
        assert "金额：1234.56 元" in text

    def test_ticket_text_skips_empty_fields(self):
        text = build_script.ticket_text({"semantic_text": "只有语义文本", "person": None})
        assert text == "只有语义文本"

    def test_ticket_text_tolerates_bad_amount(self):
        text = build_script.ticket_text({"semantic_text": "x", "amount_fen": "不是数字"})
        assert "金额（分）：不是数字" in text

    def test_incremental_mode_clears_graph_then_ingests(self, monkeypatch):
        queries = []
        monkeypatch.setattr(
            models, "run_cypher", lambda q, p=None: queries.append(q) or []
        )
        monkeypatch.setattr(
            builder, "ingest_document", lambda text, doc_id: {"entities": 2, "relations": 1}
        )
        stats = build_script.build_incremental([("doc-1", "文本1"), ("doc-2", "文本2")])
        assert queries[0] == "MATCH (n) DETACH DELETE n"
        assert stats == {"entities": 4, "relations": 2}

    def test_batch_mode_accumulates_then_calls_create_graph(self, monkeypatch):
        monkeypatch.setattr(
            builder,
            "extract_entities",
            lambda text: {
                "entities": [{"name": f"实体{text}", "entity_type": "人物", "description": ""}],
                "relations": [{"source": f"实体{text}", "target": "另一端", "relation": "关联"}],
            },
        )
        captured = {}

        def fake_create_graph(entities, relations):
            captured["entities"] = entities
            captured["relations"] = relations
            return {"entities": len(entities), "relations": len(relations)}

        monkeypatch.setattr(builder, "create_graph", fake_create_graph)
        stats = build_script.build_batch([("doc-1", "A"), ("doc-2", "B")])

        assert stats == {"entities": 2, "relations": 2}
        assert [item["doc_ids"] for item in captured["entities"]] == [["doc-1"], ["doc-2"]]
        assert [item["doc_ids"] for item in captured["relations"]] == [["doc-1"], ["doc-2"]]

    def test_graph_stats_reads_real_counts(self, monkeypatch):
        def fake_run_cypher(query, params=None):
            if "count(n)" in query:
                return [{"total": 7}]
            if "count(r)" in query:
                return [{"total": 6}]
            if "count(c)" in query:
                return [{"total": 3}]
            return [{"community_id": -1}, {"community_id": 0}]

        monkeypatch.setattr(models, "run_cypher", fake_run_cypher)
        assert build_script.graph_stats() == {
            "entities": 7,
            "relationships": 6,
            "communities": 3,
            "community_ids": [-1, 0],
        }

    def test_parse_args_defaults(self):
        args = build_script.parse_args([])
        assert args.mode == "incremental"
        assert args.limit == 10
        assert args.resolution == 1.0
        assert args.report == ""
        assert args.query == ""

    def test_parse_args_rejects_unknown_mode(self):
        with pytest.raises(SystemExit):
            build_script.parse_args(["--mode", "whatever"])

    def test_pick_demo_query_uses_highest_degree_entity(self, monkeypatch):
        monkeypatch.setattr(
            models, "run_cypher", lambda q, p=None: [{"name": "董文", "degree": 5}]
        )
        assert build_script.pick_demo_query("") == "董文 与哪些实体有关联？"

    def test_pick_demo_query_keeps_explicit_query(self, monkeypatch):
        monkeypatch.setattr(
            models, "run_cypher", lambda q, p=None: pytest.fail("显式查询不应访问数据库")
        )
        assert build_script.pick_demo_query("张三的报销") == "张三的报销"

    def test_pick_demo_query_falls_back_on_empty_graph(self, monkeypatch):
        monkeypatch.setattr(models, "run_cypher", lambda q, p=None: [])
        assert build_script.pick_demo_query("") == build_script._GENERIC_QUERY


# ============================================================
# 集成测试：真实 Neo4j
# ------------------------------------------------------------
# 刻意不调真实 LLM / Embedding（那两个属于 live 用例，会产生费用），
# 用确定性假向量 + 假 LLM 把"图数据库这一层"单独验证掉。
# 所有测试数据都带 __it_graph_ 前缀，用完即删，不碰真实建图数据；
# 也刻意不在集成用例里跑 Louvain（它会重写全图的 community_id，
# 那样会把真实建图冒烟的结果打乱）——Louvain / 社区摘要由离线用例
# 与 RAG/script/build_finance_graph.py 覆盖。
# ============================================================
IT_PREFIX = "__it_graph_"
IT_COMMUNITY_ID = -9999


def _neo4j_available() -> bool:
    """探一次真实连通性；不可用就让集成用例整体跳过，绝不把套件跑红"""
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            settings.neo4j.uri, auth=(settings.neo4j.user, settings.neo4j.password)
        )
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:  # noqa: BLE001 - 探测失败一律视为不可用
        return False


requires_neo4j = pytest.mark.skipif(
    not _neo4j_available(),
    reason=f"Neo4j 不可用（{settings.neo4j.uri}），跳过集成测试",
)


def _fake_vector(text) -> list[float]:
    """确定性假向量：把每个字符哈希到某一维后 L2 归一化。

    维度必须与向量索引的 `vector.dimensions` 一致（1024），否则索引查不出来。
    """
    dims = settings.embedding.embedding_size
    vector = [0.0] * dims
    for char in str(text):
        index = int(hashlib.md5(char.encode("utf-8")).hexdigest()[:8], 16) % dims
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:  # 空文本兜底：全零向量没法算余弦
        vector[0] = 1.0
        return vector
    return [value / norm for value in vector]


def _cleanup_it_data() -> None:
    """删掉所有带 __it_graph_ 前缀的实体与测试社区节点"""
    models.run_cypher(
        "MATCH (n:Entity) WHERE n.name STARTS WITH $prefix DETACH DELETE n",
        {"prefix": IT_PREFIX},
    )
    models.run_cypher(
        "MATCH (c:Community {community_id: $cid}) DETACH DELETE c", {"cid": IT_COMMUNITY_ID}
    )


@pytest.mark.integration
@requires_neo4j
class TestRealNeo4jIntegration:
    """真实 Neo4j 上的建图与检索验证"""

    @pytest.fixture(autouse=True)
    def _graph(self, monkeypatch):
        models._connected = False
        models.connect()
        models.init_schema()
        # 假向量：避免真实 embedding 调用（integration ≠ live）
        monkeypatch.setattr(builder, "embed_query", _fake_vector)
        monkeypatch.setattr(retriever, "embed_query", _fake_vector)
        # 假 LLM：任何走到消歧的候选都判为"不是同一对象"，绝不触发真实 API
        monkeypatch.setattr(builder, "call_llm", lambda *a, **k: "NONE")
        _cleanup_it_data()
        yield
        _cleanup_it_data()
        models._connected = False

    def _set_community(self, names, community_id=IT_COMMUNITY_ID) -> None:
        models.run_cypher(
            "UNWIND $names AS name MATCH (n:Entity {name: name}) SET n.community_id = $cid",
            {"names": list(names), "cid": community_id},
        )

    def test_link_or_create_writes_real_nodes(self):
        builder.link_or_create(
            {"name": f"{IT_PREFIX}张三", "type": "人物", "description": "客户"}, "doc-1"
        )
        node = models.Entity.nodes.get_or_none(name=f"{IT_PREFIX}张三")
        assert node is not None
        assert node.entity_type == "人物"
        assert node.mentions == 1
        assert node.doc_ids == ["doc-1"]
        assert len(node.embedding) == settings.embedding.embedding_size

        # 再抽到一次：应走合并分支，节点数不变
        builder.link_or_create(
            {"name": f"{IT_PREFIX}张三", "type": "人物", "description": "员工"}, "doc-2"
        )
        node = models.Entity.nodes.get_or_none(name=f"{IT_PREFIX}张三")
        assert node.mentions == 2
        assert node.doc_ids == ["doc-1", "doc-2"]
        assert node.description == "客户；员工"

    def test_relation_confidence_is_weighted_averaged_in_graph(self):
        builder.link_or_create({"name": f"{IT_PREFIX}甲", "type": "人物"}, "doc-1")
        right = builder.link_or_create({"name": f"{IT_PREFIX}乙", "type": "人物"}, "doc-1")
        assert builder.upsert_relation(
            f"{IT_PREFIX}甲", f"{IT_PREFIX}乙", "转账", confidence=0.8, doc_id="doc-1"
        )
        assert builder.upsert_relation(
            f"{IT_PREFIX}甲", f"{IT_PREFIX}乙", "转账", confidence=1.0, doc_id="doc-2"
        )

        left = models.Entity.nodes.get_or_none(name=f"{IT_PREFIX}甲")
        rel = left.relates_to.relationship(right)
        assert rel.weight == 2
        assert rel.confidence == 0.9  # (0.8*1 + 1.0) / 2
        assert sorted(rel.doc_ids) == ["doc-1", "doc-2"]

    def test_relation_with_missing_endpoint_is_skipped(self):
        builder.link_or_create({"name": f"{IT_PREFIX}孤零零", "type": "人物"}, "doc-1")
        assert (
            builder.upsert_relation(f"{IT_PREFIX}孤零零", f"{IT_PREFIX}查无此人", "转账") is False
        )

    def test_entity_vector_index_really_returns_self(self):
        # 不带 description：实体向量就是「名称」本身的向量，自己查自己应得满分
        builder.link_or_create({"name": f"{IT_PREFIX}赵六", "type": "人物"}, "doc-1")
        models.run_cypher("CALL db.awaitIndexes(60)")

        hits = retriever.vector_match_entities([f"{IT_PREFIX}赵六"], top_k=5)
        names = [hit["name"] for hit in hits]
        assert f"{IT_PREFIX}赵六" in names
        self_hit = next(hit for hit in hits if hit["name"] == f"{IT_PREFIX}赵六")
        assert self_hit["score"] > 0.99  # 同一个向量，余弦相似度应为 1
        assert names[0] == f"{IT_PREFIX}赵六"  # 也应该排在第一位

    def test_multihop_traversal_stays_inside_community(self):
        for suffix in ("甲", "乙", "丙"):
            builder.link_or_create(
                {"name": f"{IT_PREFIX}{suffix}", "type": "人物"}, "doc-1"
            )
        builder.upsert_relation(f"{IT_PREFIX}甲", f"{IT_PREFIX}乙", "转账", doc_id="doc-1")
        builder.upsert_relation(f"{IT_PREFIX}乙", f"{IT_PREFIX}丙", "转账", doc_id="doc-1")
        self._set_community([f"{IT_PREFIX}甲", f"{IT_PREFIX}乙", f"{IT_PREFIX}丙"])
        models.run_cypher("CALL db.awaitIndexes(60)")

        subgraph = retriever.retrieve_by_entities(
            [f"{IT_PREFIX}甲"], max_hops=2, max_nodes=20, community_ids=[IT_COMMUNITY_ID]
        )
        assert {node["name"] for node in subgraph["nodes"]} == {
            f"{IT_PREFIX}甲",
            f"{IT_PREFIX}乙",
            f"{IT_PREFIX}丙",
        }
        keys = {
            f"{rel['source']}-{rel['relation']}-{rel['target']}"
            for rel in subgraph["relationships"]
        }
        assert keys == {
            f"{IT_PREFIX}甲-转账-{IT_PREFIX}乙",
            f"{IT_PREFIX}乙-转账-{IT_PREFIX}丙",
        }

        # 课案点名的坑：community_ids=[] 时 `WHERE ... IN []` 恒不成立 → 必然空结果
        assert retriever.retrieve_by_entities(
            [f"{IT_PREFIX}甲"], community_ids=[]
        ) == {"nodes": [], "relationships": []}

    def test_hierarchical_retrieval_over_real_community_index(self, monkeypatch):
        builder.link_or_create({"name": f"{IT_PREFIX}甲", "type": "人物"}, "doc-1")
        builder.link_or_create({"name": f"{IT_PREFIX}乙", "type": "人物"}, "doc-1")
        builder.upsert_relation(f"{IT_PREFIX}甲", f"{IT_PREFIX}乙", "转账", doc_id="doc-1")
        self._set_community([f"{IT_PREFIX}甲", f"{IT_PREFIX}乙"])

        # 直接用 Cypher 写社区节点：绕开真实 LLM 摘要调用
        summary = f"{IT_PREFIX}该社区记录了甲向乙的转账"
        models.run_cypher(
            """
            MERGE (c:Community {community_id: $cid})
            SET c.summary = $summary, c.embedding = $vector
            """,
            {"cid": IT_COMMUNITY_ID, "summary": summary, "vector": _fake_vector(summary)},
        )
        models.run_cypher("CALL db.awaitIndexes(60)")

        monkeypatch.setattr(
            retriever,
            "call_llm",
            lambda system, user, temperature=0.1, json_mode=False: (
                json.dumps({"entities": [f"{IT_PREFIX}甲"]}) if json_mode else "最终回答"
            ),
        )

        for _ in range(5):  # 向量索引对新写入节点可能有写入延迟，轻微重试
            result = retriever.retrieve_hierarchical(summary, top_k=5, max_hops=2, max_nodes=20)
            if result["nodes"]:
                break
            time.sleep(1)

        assert IT_COMMUNITY_ID in [item["community_id"] for item in result["communities"]]
        assert f"{IT_PREFIX}甲" in {node["name"] for node in result["nodes"]}

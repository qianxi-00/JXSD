"""graph_rag.models 的离线测试。

覆盖范围（全部不连 Neo4j、不调外部 API）：
- RelatesTo / Entity / Community 的属性定义与默认值
- Entity -> Entity 的 RELATES_TO 关系定义
- 原生向量索引 DDL 的幂等写法与维度/相似度取值
- run_cypher 的结果整形
- connect() 的 URL 拼装与幂等

neomodel 7 与课案写法的差异见 models.py 顶部注释：
课案里的 `config.DATABASE_URL = ...` 在 7.0 已被 NeomodelConfig 取代（直接赋值会发
DeprecationWarning），这里改用 `get_config().database_url/database_name`。
"""

from urllib.parse import quote

import pytest

from config import settings
from graph_rag import models


class TestRelatesTo:
    """关系模型 RelatesTo 的默认值"""

    def test_defaults(self):
        rel = models.RelatesTo()
        assert rel.relation_type == "RELATES_TO"
        assert rel.confidence == 1.0
        assert rel.weight == 1
        assert rel.doc_ids == []

    def test_accepts_explicit_values(self):
        rel = models.RelatesTo(
            relation_type="报销", confidence=0.8, weight=3, doc_ids=["doc-1"]
        )
        assert rel.relation_type == "报销"
        assert rel.confidence == 0.8
        assert rel.weight == 3
        assert rel.doc_ids == ["doc-1"]

    def test_doc_ids_default_is_not_shared_between_instances(self):
        """default=list 保证每个实例拿到独立列表（不能共用同一个可变对象）"""
        a, b = models.RelatesTo(), models.RelatesTo()
        a.doc_ids.append("doc-x")
        assert b.doc_ids == []


class TestEntityModel:
    """实体节点模型 Entity 的属性定义"""

    def test_defaults(self):
        node = models.Entity(name="张三")
        assert node.name == "张三"
        assert node.entity_type == "UNKNOWN"
        assert node.description == ""
        assert node.doc_ids == []
        assert node.mentions == 1
        assert node.community_id == -1
        assert node.embedding == []

    def test_name_is_unique_index(self):
        props = models.Entity.defined_properties(aliases=False, rels=False)
        assert props["name"].unique_index is True
        assert props["name"].required is True

    def test_embedding_is_float_list(self):
        props = models.Entity.defined_properties(aliases=False, rels=False)
        # embedding 必须是 list[float]，否则 Neo4j 原生向量索引建不起来
        assert props["embedding"].base_property.__class__.__name__ == "FloatProperty"

    def test_relates_to_definition(self):
        definition = models.Entity.relates_to.definition
        assert definition["relation_type"] == "RELATES_TO"
        assert definition["model"] is models.RelatesTo

    def test_entity_is_a_neomodel_node(self):
        from neomodel import StructuredNode

        assert issubclass(models.Entity, StructuredNode)
        assert issubclass(models.RelatesTo, models.StructuredRel)


class TestCommunityModel:
    """社区节点模型 Community"""

    def test_defaults(self):
        node = models.Community(community_id=3)
        assert node.community_id == 3
        assert node.summary == ""
        assert node.embedding == []

    def test_community_id_is_unique_index(self):
        props = models.Community.defined_properties(aliases=False, rels=False)
        assert props["community_id"].unique_index is True


class TestVectorIndexDdl:
    """向量索引 DDL：幂等 + 维度/相似度来自配置"""

    @pytest.fixture
    def captured(self, monkeypatch):
        calls: list[str] = []

        def fake_run_cypher(query, params=None):
            calls.append(query)
            return []

        monkeypatch.setattr(models, "run_cypher", fake_run_cypher)
        return calls

    def test_creates_two_vector_indexes_idempotently(self, captured):
        models.ensure_vector_indexes()
        ddl = [q for q in captured if "CREATE VECTOR INDEX" in q]
        assert len(ddl) == 2
        # IF NOT EXISTS 保证重复调用不报错
        assert all("IF NOT EXISTS" in q for q in ddl)

    def test_index_names_come_from_settings(self, captured):
        models.ensure_vector_indexes()
        ddl = " ".join(q for q in captured if "CREATE VECTOR INDEX" in q)
        assert settings.neo4j.entity_index in ddl
        assert settings.neo4j.community_index in ddl

    def test_dimensions_and_similarity_from_settings(self, captured):
        models.ensure_vector_indexes()
        ddl = [q for q in captured if "CREATE VECTOR INDEX" in q]
        for query in ddl:
            assert f"`vector.dimensions`: {settings.embedding.embedding_size}" in query
            assert "`vector.similarity_function`: 'cosine'" in query

    def test_entity_index_targets_entity_embedding(self, captured):
        models.ensure_vector_indexes()
        entity_ddl = next(
            q for q in captured if settings.neo4j.entity_index in q and "CREATE VECTOR INDEX" in q
        )
        assert "FOR (n:Entity) ON (n.embedding)" in entity_ddl

    def test_community_index_targets_community_embedding(self, captured):
        models.ensure_vector_indexes()
        community_ddl = next(
            q
            for q in captured
            if settings.neo4j.community_index in q and "CREATE VECTOR INDEX" in q
        )
        assert "FOR (n:Community) ON (n.embedding)" in community_ddl

    def test_waits_for_indexes_to_come_online(self, captured):
        """刚建好的向量索引可能还在 POPULATING，检索前要等它 ONLINE"""
        models.ensure_vector_indexes()
        assert any("db.awaitIndexes" in q for q in captured)


class TestRunCypher:
    """run_cypher：把 (rows, columns) 整形为 dict 列表，作为统一 DB 出口"""

    def test_converts_rows_to_dicts(self, monkeypatch):
        monkeypatch.setattr(
            models.db,
            "cypher_query",
            lambda query, params: ([[1, "a"], [2, "b"]], ("cid", "name")),
        )
        rows = models.run_cypher("RETURN 1", {})
        assert rows == [{"cid": 1, "name": "a"}, {"cid": 2, "name": "b"}]

    def test_passes_query_and_params_through(self, monkeypatch):
        seen = {}

        def fake(query, params):
            seen["query"] = query
            seen["params"] = params
            return ([], ())

        monkeypatch.setattr(models.db, "cypher_query", fake)
        models.run_cypher("MATCH (n)", {"x": 1})
        assert seen == {"query": "MATCH (n)", "params": {"x": 1}}

    def test_defaults_params_to_empty_dict(self, monkeypatch):
        monkeypatch.setattr(models.db, "cypher_query", lambda q, p: ([], ()))
        assert models.run_cypher("RETURN 1") == []


class TestConnect:
    """connect()：把 .env 里的 Neo4j 配置灌进 neomodel（不发起真实连接）"""

    @pytest.fixture(autouse=True)
    def _restore(self):
        from neomodel import reset_config

        original = models._connected
        yield
        models._connected = original
        reset_config()

    def test_sets_database_url_and_name(self):
        from neomodel import reset_config

        reset_config()
        models._connected = False
        models.connect()

        cfg = models.get_config()
        assert cfg.database_url.startswith("bolt://")
        assert settings.neo4j.user in cfg.database_url
        assert cfg.database_name == settings.neo4j.database

    def test_password_is_url_encoded_not_raw(self):
        """口令里若含 @ / : 等字符，必须百分号编码，否则 URL 会被解析坏"""
        from neomodel import reset_config

        reset_config()
        models._connected = False
        models.connect()

        cfg = models.get_config()
        assert quote(settings.neo4j.password, safe="") in cfg.database_url
        # host:port 保留
        assert "127.0.0.1:7687" in cfg.database_url

    def test_idempotent(self):
        """重复调用不应重建连接配置（避免运行期把 driver 换掉）"""
        from neomodel import reset_config

        reset_config()
        models._connected = False
        models.connect()
        models.get_config().database_url = "bolt://sentinel:7687"

        models.connect()  # 第二次应直接返回
        assert models.get_config().database_url == "bolt://sentinel:7687"

    def test_init_schema_installs_constraints_and_vector_indexes(self, monkeypatch):
        called = {"constraints": 0, "vector": 0}
        monkeypatch.setattr(models, "connect", lambda: None)
        monkeypatch.setattr(
            models,
            "ensure_constraints",
            lambda: called.__setitem__("constraints", called["constraints"] + 1),
        )
        monkeypatch.setattr(
            models, "ensure_vector_indexes", lambda: called.__setitem__("vector", 1)
        )
        models.init_schema()
        assert called == {"constraints": 1, "vector": 1}

    def test_ensure_constraints_installs_when_missing(self, monkeypatch):
        installed = {"n": 0}
        monkeypatch.setattr(models, "run_cypher", lambda q, p=None: [])
        monkeypatch.setattr(
            models.db, "install_all_labels", lambda: installed.__setitem__("n", installed["n"] + 1)
        )
        assert models.ensure_constraints() is True
        assert installed["n"] == 1

    def test_ensure_constraints_skips_when_already_present(self, monkeypatch):
        """约束已存在时不能再调 install_all_labels（否则会刷一屏等价约束报错）"""
        monkeypatch.setattr(
            models,
            "run_cypher",
            lambda q, p=None: [
                {"labelsOrTypes": ["Entity"], "properties": ["name"]},
                {"labelsOrTypes": ["Community"], "properties": ["community_id"]},
            ],
        )
        monkeypatch.setattr(
            models.db, "install_all_labels", lambda: pytest.fail("约束已存在不应重复安装")
        )
        assert models.ensure_constraints() is False


# ============================================================
# 集成测试：需要真实 Neo4j
# ============================================================
def _neo4j_available() -> bool:
    """探一次真实连通性；失败就让集成用例整体跳过，绝不把套件跑红"""
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


NEO4J_AVAILABLE = _neo4j_available()

requires_neo4j = pytest.mark.skipif(
    not NEO4J_AVAILABLE,
    reason=f"Neo4j 不可用（{settings.neo4j.uri}），跳过集成测试",
)


@pytest.mark.integration
@requires_neo4j
class TestNeo4jIntegration:
    """真实连通性 + 真实 schema 验证（只读探测，不动业务数据）"""

    @pytest.fixture(autouse=True)
    def _fresh_config(self):
        models._connected = False
        models.connect()
        yield
        models._connected = False

    def test_cypher_roundtrip(self):
        assert models.run_cypher("RETURN 1 AS one, 'x' AS two") == [{"one": 1, "two": "x"}]

    def test_vector_indexes_are_really_online(self):
        models.init_schema()
        rows = models.run_cypher(
            """
            SHOW INDEXES YIELD name, type, state, options
            WHERE name IN $names
            RETURN name, type, state, options
            """,
            {"names": [settings.neo4j.entity_index, settings.neo4j.community_index]},
        )
        assert {row["name"] for row in rows} == {
            settings.neo4j.entity_index,
            settings.neo4j.community_index,
        }
        for row in rows:
            assert row["type"] == "VECTOR"
            assert row["state"] == "ONLINE"
            config = row["options"]["indexConfig"]
            assert config["vector.dimensions"] == settings.embedding.embedding_size
            assert config["vector.similarity_function"] == "COSINE"

    def test_entity_name_uniqueness_constraint_exists(self):
        models.init_schema()
        rows = models.run_cypher(
            """
            SHOW CONSTRAINTS YIELD labelsOrTypes, properties
            RETURN labelsOrTypes, properties
            """
        )
        entity_constraints = [
            (tuple(row["labelsOrTypes"] or []), tuple(row["properties"] or []))
            for row in rows
        ]
        assert (("Entity",), ("name",)) in entity_constraints

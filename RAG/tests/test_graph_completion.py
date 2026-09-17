"""T6 链接补全 + 人工审核闭环的测试。

全部用假 LLM / 假落库函数，不连 Neo4j、不真调模型
（真机那一次另见 README 的证据行：一次性孤点探针 + 收回）。
"""

import json
from pathlib import Path

import pytest

from graph_rag import completion


def _queue(tmp_dir) -> Path:
    """队列文件路径（临时目录由 conftest 的 `tmp_dir` fixture 提供，见那里的说明）。"""
    return tmp_dir / "queue.jsonl"


class TestConfidenceParsing:
    def test_accepts_normal_values(self):
        assert completion.as_confidence(0.85) == pytest.approx(0.85)
        assert completion.as_confidence("0.9") == pytest.approx(0.9)
        assert completion.as_confidence(1) == pytest.approx(1.0)

    def test_clamps_out_of_range(self):
        assert completion.as_confidence(1.7) == pytest.approx(1.0)
        assert completion.as_confidence(-3) == pytest.approx(0.0)

    def test_missing_or_garbage_is_none_not_full_marks(self):
        """缺置信度 / 解析不出来必须是 None —— 默认成 1.0 会让模型的胡话直接落库。"""
        for bad in (None, "", "高", [], {}, float("nan")):
            assert completion.as_confidence(bad) is None


class TestClassifyProposals:
    def test_splits_by_threshold(self):
        proposals = [
            {"target": "A", "relation": "属于", "confidence": 0.95},
            {"target": "B", "relation": "属于", "confidence": 0.62},
            {"target": "C", "relation": "属于", "confidence": 0.10},
        ]
        result = completion.classify_proposals(proposals, auto=0.8, review=0.5)
        assert [p["target"] for p in result["auto"]] == ["A"]
        assert [p["target"] for p in result["review"]] == ["B"]
        assert [p["target"] for p in result["dropped"]] == ["C"]
        # 掉在地上的也要看得见（否则"补全任务跑了但什么都没连上"无法归因）
        assert result["dropped"][0]["confidence"] == pytest.approx(0.10)

    def test_missing_confidence_goes_to_review(self):
        proposals = [{"target": "A", "relation": "属于"}]
        result = completion.classify_proposals(proposals, auto=0.8, review=0.5)
        assert result["auto"] == []
        assert [p["target"] for p in result["review"]] == ["A"]
        assert result["review"][0]["confidence"] is None

    def test_boundary_is_inclusive(self):
        proposals = [
            {"target": "A", "relation": "r", "confidence": 0.8},
            {"target": "B", "relation": "r", "confidence": 0.5},
        ]
        result = completion.classify_proposals(proposals, auto=0.8, review=0.5)
        assert [p["target"] for p in result["auto"]] == ["A"]
        assert [p["target"] for p in result["review"]] == ["B"]

    def test_requires_target(self):
        proposals = [{"relation": "r", "confidence": 0.99}, {"target": "  ", "confidence": 0.99}]
        result = completion.classify_proposals(proposals, auto=0.8, review=0.5)
        assert result["auto"] == [] and result["review"] == []


class TestProposeLinks:
    def test_parses_llm_json_and_keeps_only_candidates(self):
        """模型编出来的 target 直接丢：补全连的边必须落在**已有节点**上（关系可追溯）。"""
        payload = json.dumps(
            {
                "links": [
                    {"target": "出差费用", "relation": "同义", "confidence": 0.9, "reason": "同一类费用"},
                    {"target": "不存在的实体", "relation": "同义", "confidence": 0.99, "reason": "编的"},
                ]
            }
        )
        proposals = completion.propose_links(
            {"name": "差旅费"},
            [{"name": "出差费用", "entity_type": "费用类型"}],
            call_llm=lambda **_: payload,
        )
        assert [p["target"] for p in proposals] == ["出差费用"]

    def test_bad_json_returns_empty_and_logs(self, monkeypatch):
        """模型不吐 JSON 时：这一条孤点放弃，但必须告警（否则"没连上"无法归因）。

        这里直接盯 `logger.warning` 而不是 caplog：项目的 logger 由 `core/logger.py`
        自己配 handler、不向 root 传播，caplog 抓不到（同 test_llm_router.py 的做法）。
        """
        warnings: list[str] = []
        monkeypatch.setattr(completion.logger, "warning", lambda msg, *a, **kw: warnings.append(str(msg)))
        proposals = completion.propose_links(
            {"name": "差旅费"}, [{"name": "出差费用"}], call_llm=lambda **_: "模型今天不想输出 JSON"
        )
        assert proposals == []
        assert any("补全" in msg for msg in warnings)

    def test_no_candidates_skips_llm(self):
        called = []
        proposals = completion.propose_links(
            {"name": "孤立点"}, [], call_llm=lambda **_: called.append(1) or "{}"
        )
        assert proposals == []
        assert called == [], "没有候选对照实体时不该花钱调模型"

    def test_normalizes_relation_and_drops_self(self):
        payload = json.dumps(
            {
                "links": [
                    {"target": "自己", "relation": "自环", "confidence": 0.99},
                    {"target": "别人", "relation": "", "confidence": 0.9},
                ]
            }
        )
        proposals = completion.propose_links(
            {"name": "自己"},
            [{"name": "自己"}, {"name": "别人"}],
            call_llm=lambda **_: payload,
        )
        assert len(proposals) == 1
        assert proposals[0]["target"] == "别人"
        # relation 为空时给一个保守的默认类型，不让空字符串进库
        assert proposals[0]["relation"] == completion.DEFAULT_RELATION_TYPE


class TestCandidatePool:
    def test_prefers_same_type_then_similarity(self):
        node = {"name": "差旅费", "entity_type": "费用类型", "embedding": [1.0, 0.0]}
        entities = [
            {"name": "张三", "entity_type": "人物", "embedding": [1.0, 0.0]},
            {"name": "出差费用", "entity_type": "费用类型", "embedding": [0.9, 0.1]},
            {"name": "住宿费", "entity_type": "费用类型", "embedding": [0.1, 0.9]},
        ]
        pool = completion.candidate_pool(node, entities, top_k=2)
        assert [c["name"] for c in pool] == ["出差费用", "住宿费"]

    def test_excludes_self_and_empty_names(self):
        node = {"name": "差旅费", "entity_type": "费用类型", "embedding": [1.0, 0.0]}
        entities = [
            {"name": "差旅费", "entity_type": "费用类型", "embedding": [1.0, 0.0]},
            {"name": "", "entity_type": "费用类型"},
            {"name": "住宿费", "entity_type": "费用类型", "embedding": [1.0, 0.0]},
        ]
        pool = completion.candidate_pool(node, entities, top_k=5)
        assert [c["name"] for c in pool] == ["住宿费"]

    def test_falls_back_to_mentions_when_no_embedding(self):
        node = {"name": "孤点", "entity_type": "费用类型", "embedding": []}
        entities = [
            {"name": "少提及", "entity_type": "费用类型", "mentions": 1},
            {"name": "多提及", "entity_type": "费用类型", "mentions": 9},
        ]
        pool = completion.candidate_pool(node, entities, top_k=1)
        assert [c["name"] for c in pool] == ["多提及"]


class TestReviewQueue:
    def test_round_trip_and_approve_applies_link(self, tmp_dir):
        path = _queue(tmp_dir)
        items = completion.enqueue_review(
            [
                {
                    "source": "差旅费",
                    "target": "出差费用",
                    "relation": "同义",
                    "confidence": 0.62,
                    "reason": "像同一类",
                }
            ],
            path=path,
        )
        assert len(items) == 1 and items[0]["status"] == "pending"
        assert items[0]["id"]

        pending = completion.load_review_queue(path=path)
        assert [i["status"] for i in pending] == ["pending"]

        applied = []
        result = completion.apply_review(
            items[0]["id"],
            approve=True,
            path=path,
            upsert_relation=lambda *a, **kw: applied.append((a, kw)) or True,
        )
        assert result["status"] == "approved"
        assert applied[0][0][:2] == ("差旅费", "出差费用")
        assert applied[0][1]["relation_type"] == "同义"
        assert applied[0][1]["confidence"] == pytest.approx(0.62)
        # 决定要落盘：再读一次不该还是 pending
        assert completion.load_review_queue(path=path)[0]["status"] == "approved"
        assert completion.load_review_queue(path=path, status="pending") == []

    def test_reject_does_not_touch_graph(self, tmp_dir):
        path = _queue(tmp_dir)
        items = completion.enqueue_review(
            [{"source": "A", "target": "B", "relation": "r", "confidence": 0.6}], path=path
        )
        applied = []
        result = completion.apply_review(
            items[0]["id"],
            approve=False,
            path=path,
            upsert_relation=lambda *a, **kw: applied.append(1),
        )
        assert result["status"] == "rejected"
        assert applied == []

    def test_approve_establishes_neo4j_connection(self, monkeypatch, tmp_dir):
        """批准这条路必须先建 Neo4j 连接。

        真机实测踩过：`review-ok` 只读队列文件、不走 `collect_graph()`，于是 `upsert_relation`
        里的 `Entity.nodes.get_or_none` 直接抛 `No Neo4j connection has been configured`。
        这条用例盯的就是"有没有调 connect"——用假落库函数测不出这个坑（假函数不需要连接）。
        """
        calls: list[int] = []
        monkeypatch.setattr(completion.models, "connect", lambda: calls.append(1))
        # 落库函数也换成假的：本用例盯的是"有没有 connect"，真连库的事真机那一次已经证过
        monkeypatch.setattr(completion.builder, "upsert_relation", lambda *a, **kw: True)
        path = _queue(tmp_dir)
        records = completion.enqueue_review(
            [{"source": "A", "target": "B", "relation": "r", "confidence": 0.6}], path=path
        )
        completion.apply_review(records[0]["id"], approve=True, path=path)  # 不注入假落库参数
        assert calls == [1], "批准落库前必须调用 models.connect()"

    def test_reject_does_not_need_connection(self, monkeypatch, tmp_dir):
        """驳回不碰图，所以不该白建一次连接。"""
        calls: list[int] = []
        monkeypatch.setattr(completion.models, "connect", lambda: calls.append(1))
        path = _queue(tmp_dir)
        records = completion.enqueue_review([{"source": "A", "target": "B", "relation": "r"}], path=path)
        completion.apply_review(records[0]["id"], approve=False, path=path)
        assert calls == []

    def test_unknown_id_raises(self, tmp_dir):
        path = _queue(tmp_dir)
        completion.enqueue_review([{"source": "A", "target": "B", "relation": "r"}], path=path)
        with pytest.raises(KeyError):
            completion.apply_review("不存在", approve=True, path=path, upsert_relation=lambda *a: True)

    def test_missing_file_is_empty_not_error(self, tmp_dir):
        assert completion.load_review_queue(path=tmp_dir / "nope.jsonl") == []

    def test_appends_instead_of_overwriting(self, tmp_dir):
        path = _queue(tmp_dir)
        completion.enqueue_review([{"source": "A", "target": "B", "relation": "r"}], path=path)
        completion.enqueue_review([{"source": "C", "target": "D", "relation": "r"}], path=path)
        assert len(completion.load_review_queue(path=path)) == 2


class TestRunCompletion:
    def _patch(self, monkeypatch, nodes, proposals, entities):
        monkeypatch.setattr(completion, "isolated_nodes", lambda limit=50: nodes)
        monkeypatch.setattr(completion, "all_entities", lambda: entities)
        monkeypatch.setattr(completion, "propose_links", lambda *a, **kw: proposals)

    def test_dry_run_writes_nothing(self, monkeypatch, tmp_dir):
        self._patch(
            monkeypatch,
            nodes=[{"name": "差旅费", "entity_type": "费用类型"}],
            proposals=[{"target": "出差费用", "relation": "同义", "confidence": 0.95}],
            entities=[{"name": "出差费用", "entity_type": "费用类型"}],
        )
        path = _queue(tmp_dir)
        applied = []
        result = completion.run_completion(
            dry_run=True,
            path=path,
            upsert_relation=lambda *a, **kw: applied.append(1),
            call_llm=lambda **_: "{}",
        )
        assert result["auto"] == 1
        assert applied == []
        assert not path.exists()
        assert result["would_apply"][0]["source"] == "差旅费"

    def test_applies_auto_and_enqueues_review(self, monkeypatch, tmp_dir):
        self._patch(
            monkeypatch,
            nodes=[{"name": "差旅费", "entity_type": "费用类型"}],
            proposals=[
                {"target": "出差费用", "relation": "同义", "confidence": 0.95},
                {"target": "费用报销", "relation": "相关", "confidence": 0.6},
            ],
            entities=[
                {"name": "出差费用", "entity_type": "费用类型"},
                {"name": "费用报销", "entity_type": "费用类型"},
            ],
        )
        path = _queue(tmp_dir)
        applied = []
        result = completion.run_completion(
            path=path,
            upsert_relation=lambda *a, **kw: applied.append(a) or True,
            call_llm=lambda **_: "{}",
        )
        assert result["auto"] == 1 and result["review"] == 1
        assert applied == [("差旅费", "出差费用")]
        queue = completion.load_review_queue(path=path)
        assert len(queue) == 1
        assert queue[0]["target"] == "费用报销"
        assert queue[0]["source"] == "差旅费"

    def test_counts_isolated_without_candidates(self, monkeypatch, tmp_dir):
        """孤点存在但没有候选对照实体（图里只有它自己）⇒ 如实计数，不要报"已完成补全"。"""
        self._patch(
            monkeypatch,
            nodes=[{"name": "孤立", "entity_type": "人物"}],
            proposals=[],
            entities=[],
        )
        result = completion.run_completion(
            path=_queue(tmp_dir),
            upsert_relation=lambda *a, **kw: True,
            call_llm=lambda **_: "{}",
        )
        assert result["scanned"] == 1
        assert result["auto"] == 0 and result["review"] == 0
        assert result["no_candidate"] == 1

    def test_respects_limit(self, monkeypatch):
        nodes = [{"name": f"孤{i}", "entity_type": "人物"} for i in range(5)]
        self._patch(monkeypatch, nodes=nodes, proposals=[], entities=[{"name": "别人"}])
        seen = []
        monkeypatch.setattr(completion, "isolated_nodes", lambda limit=50: seen.append(limit) or nodes[:limit])
        result = completion.run_completion(
            limit=2, path=None, upsert_relation=lambda *a, **kw: True, call_llm=lambda **_: "{}"
        )
        assert seen == [2]
        assert result["scanned"] == 2

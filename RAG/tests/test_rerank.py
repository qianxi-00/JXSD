"""重排序测试:请求构造 + 阈值门控 + 原始分数出口。

`rerank_scores` 是给阈值标定用的"不过阈值、不截断"的出口 ——
换 reranker（本项目 2026-09-17 从 Qwen3-Reranker-4B 换成 BAAI/bge-reranker-v2-m3）
必须重新标定 `RERANK_RELEVANCE_P`，标定要的正是原始分数。
"""

import pytest

from retrieval import rerank as rerank_mod


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        return self._payload


@pytest.fixture
def fake_post(monkeypatch):
    captured = {}

    def install(payload, status_code=200):
        def _post(url, headers=None, json=None, timeout=None):
            captured["url"] = url
            captured["json"] = json
            captured["headers"] = headers
            return FakeResponse(payload, status_code)

        monkeypatch.setattr(rerank_mod.requests, "post", _post)
        return captured

    return install


DOCS = [
    {"id": "t1", "semantic_text": "万宁 火车票 票号 T2023"},
    {"id": "t2", "semantic_text": "今天天气不错"},
]


def test_request_shape(fake_post):
    captured = fake_post({"results": [{"index": 0, "relevance_score": 0.9}]})
    rerank_mod.rerank_scores("万宁的票号?", DOCS)
    assert captured["json"]["query"] == "万宁的票号?"
    assert captured["json"]["documents"] == [d["semantic_text"] for d in DOCS]
    assert captured["json"]["top_n"] == len(DOCS)
    assert captured["url"].endswith("/rerank")


def test_scores_returns_every_candidate(fake_post):
    """标定要的是全部候选的分数,不能被 top_k 截断、也不能被阈值滤掉。"""
    fake_post({"results": [{"index": 1, "relevance_score": 0.2}, {"index": 0, "relevance_score": 0.8}]})
    scored = rerank_mod.rerank_scores("q", DOCS)
    assert [(s["id"], s["rerank_score"]) for s in scored] == [("t2", 0.2), ("t1", 0.8)]


def test_scores_keeps_original_fields(fake_post):
    fake_post({"results": [{"index": 0, "relevance_score": 0.8}]})
    assert rerank_mod.rerank_scores("q", DOCS)[0]["semantic_text"] == "万宁 火车票 票号 T2023"


def test_scores_missing_relevance_is_none(fake_post):
    fake_post({"results": [{"index": 0}]})
    assert rerank_mod.rerank_scores("q", DOCS)[0]["rerank_score"] is None


def test_scores_empty_documents_short_circuits(fake_post):
    captured = fake_post({"results": []})
    assert rerank_mod.rerank_scores("q", []) == []
    assert captured == {}  # 不该发请求


def test_rerank_filters_by_threshold(fake_post, monkeypatch):
    fake_post({"results": [{"index": 0, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.3}]})
    monkeypatch.setattr(rerank_mod.settings.rerank, "relevance_p", 0.65)
    assert [r["id"] for r in rerank_mod.rerank("q", DOCS)] == ["t1"]


def test_rerank_respects_top_k(fake_post, monkeypatch):
    docs = [{"id": f"t{i}", "semantic_text": f"doc{i}"} for i in range(4)]
    fake_post({"results": [{"index": i, "relevance_score": 0.9 - i * 0.01} for i in range(4)]})
    monkeypatch.setattr(rerank_mod.settings.rerank, "relevance_p", 0.0)
    assert [r["id"] for r in rerank_mod.rerank("q", docs, top_k=2)] == ["t0", "t1"]


def test_http_error_raises(fake_post):
    fake_post({"error": "bad"}, status_code=500)
    with pytest.raises(RuntimeError) as excinfo:
        rerank_mod.rerank("q", DOCS)
    assert "500" in str(excinfo.value)

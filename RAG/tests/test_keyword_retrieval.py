"""关键词召回(BM25)测试:验证按过滤条件在选取阶段过滤,且不影响打分排序。"""

import pytest

from retrieval import keyword_retrieval

ROWS = [
    {
        "id": "ticket_a",
        "ticket_type": "train",
        "person": "张三",
        "ticket_no": "T1",
        "date_int": 20250305,
        "amount_fen": 43600,
        "route": "北京-上海",
        "source_file": "data/train/a.png",
        "semantic_text": "张三 高铁票 北京 上海 436元",
    },
    {
        "id": "ticket_b",
        "ticket_type": "train",
        "person": "李四",
        "ticket_no": "T2",
        "date_int": 20250321,
        "amount_fen": 40000,
        "route": "南京-上海",
        "source_file": "data/train/b.png",
        "semantic_text": "李四 高铁票 南京 上海 400元",
    },
]


@pytest.fixture
def bm25_index(monkeypatch):
    """把进程内缓存的语料换成固定两行,并构造成真实 BM25 索引"""
    from rank_bm25 import BM25Okapi

    tokenized = [keyword_retrieval._tokenize(r["semantic_text"]) for r in ROWS]
    bm25 = BM25Okapi(tokenized)
    monkeypatch.setattr(keyword_retrieval, "_load_index", lambda: (ROWS, bm25, tokenized))
    return bm25


def test_keyword_search_without_filters_returns_both(bm25_index):
    hits = keyword_retrieval.keyword_search("高铁票 上海", top_n=5)
    assert {h["id"] for h in hits} == {"ticket_a", "ticket_b"}


def test_keyword_search_filters_by_person(bm25_index):
    hits = keyword_retrieval.keyword_search("高铁票 上海", top_n=5, filters={"person": "张三"})
    assert [h["id"] for h in hits] == ["ticket_a"]


def test_keyword_search_filters_by_amount_range(bm25_index):
    hits = keyword_retrieval.keyword_search(
        "高铁票 上海", top_n=5, filters={"amount_min_fen": 42000}
    )
    assert [h["id"] for h in hits] == ["ticket_a"]


def test_keyword_search_respects_top_n_after_filtering(bm25_index):
    hits = keyword_retrieval.keyword_search("高铁票 上海", top_n=1, filters={"ticket_type": "train"})
    assert len(hits) == 1


def test_keyword_search_keeps_keyword_score(bm25_index):
    hits = keyword_retrieval.keyword_search("高铁票 上海", top_n=5)
    assert all("keyword_score" in h for h in hits)
    scores = [h["keyword_score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_common_terms_are_not_dropped_when_bm25_score_is_negative(bm25_index):
    """回归:词出现在过半文档时 BM25Okapi 的 IDF 为负、分数为负,
    旧实现用 scores <= 0 判定相关性,会把真实命中的行整批丢掉。"""
    hits = keyword_retrieval.keyword_search("高铁票", top_n=5)
    assert {h["id"] for h in hits} == {"ticket_a", "ticket_b"}

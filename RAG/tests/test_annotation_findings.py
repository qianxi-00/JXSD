"""三个"读代码才发现"的缺陷的回归用例（2026-09-17 补注释时发现）。

1. 重排日志里的 `max()`：接口返回空 results 时会抛 ValueError，
   把"这次没结果"升级成"重排整段失败"；
2. BM25 语料为空 / 全为空文本时 `BM25Okapi` 直接除零，表现成与业务无关的崩溃；
3. 子查询行首编号的 `strip` 用字符集合，会连正文开头的年份一起削掉。
"""

from types import SimpleNamespace

from llm import chat


class TestSubqueryMarkerStripping:
    """子问题解析：剥标记，但不能碰正文里的数字。"""

    def run(self, monkeypatch, text: str) -> list[str]:
        monkeypatch.setattr(chat, "_client", lambda: _FakeClient(text))
        return chat.rewrite_query("原始问题", "subquery")

    def test_strips_numbering(self, monkeypatch):
        assert self.run(monkeypatch, "1. 张三的火车票金额是多少\n2. 张三的机票金额是多少") == [
            "张三的火车票金额是多少",
            "张三的机票金额是多少",
        ]

    def test_strips_bullets_and_parenthesised_numbers(self, monkeypatch):
        assert self.run(monkeypatch, "- 甲的问题\n(2) 乙的问题\n• 丙的问题") == [
            "甲的问题",
            "乙的问题",
            "丙的问题",
        ]

    def test_keeps_leading_year_in_body(self, monkeypatch):
        """回归：以前 `strip("0123456789...")` 会把开头的年份削掉。"""
        assert self.run(monkeypatch, "1. 2025年张三的火车票金额是多少") == [
            "2025年张三的火车票金额是多少"
        ]

    def test_caps_at_three_subqueries(self, monkeypatch):
        text = "\n".join(f"{i}. 第{i}个子问题" for i in range(1, 6))
        assert len(self.run(monkeypatch, text)) == 3


class _FakeClient:
    def __init__(self, text: str):
        self._text = text
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        message = SimpleNamespace(content=self._text, reasoning_content=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class TestBm25EmptyCorpus:
    """空语料不能崩在 BM25 的除零上。"""

    def test_empty_index_returns_empty_hits(self, monkeypatch):
        from retrieval import keyword_retrieval as kr

        monkeypatch.setattr(kr, "_load_index", lambda: ([], None, []))
        assert kr.keyword_search("任何问题") == []

    def test_all_blank_semantic_text_skips_index(self, monkeypatch):
        """语料非空但全为空文本时，`_load_index` 不能构造 BM25（会除零）。"""
        from core import database
        from retrieval import keyword_retrieval as kr

        rows = [{"id": "t1", "semantic_text": ""}, {"id": "t2", "semantic_text": "   "}]

        class FakeMilvus:
            def query(self, **kwargs):
                return rows

        monkeypatch.setattr(database, "get_milvus_client", lambda: FakeMilvus())
        # _load_index 带 lru_cache，直接调 __wrapped__ 绕开缓存，避免读到进程内旧语料
        loaded_rows, bm25, tokenized = kr._load_index.__wrapped__()
        assert bm25 is None
        assert len(loaded_rows) == 2
        assert tokenized == [[], []]


class TestRerankEmptyResults:
    """接口返回空 results 时，`rerank` 不能因为日志统计而抛错。"""

    def test_empty_results_yields_empty_and_logs(self, monkeypatch):
        from retrieval import rerank as rerank_mod

        class FakeResponse:
            status_code = 200

            @staticmethod
            def json():
                return {"results": []}

        def fake_post(*args, **kwargs):
            return FakeResponse()

        monkeypatch.setattr(rerank_mod.requests, "post", fake_post)
        docs = [{"id": "t1", "semantic_text": "票据正文"}]
        assert rerank_mod.rerank("问题", docs) == []

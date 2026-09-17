"""向量召回测试:验证 filter_expr 能下推给 Milvus,以及返回结构。

用假 Milvus 客户端替换 get_milvus_client,不依赖真实服务。
"""

import pytest

from retrieval import vector_retrieval


class FakeMilvusClient:
    """记录 search 调用参数并返回固定命中的假客户端"""

    def __init__(self, hits=None):
        self.calls: list[dict] = []
        self._hits = hits if hits is not None else [
            {"id": "ticket_a", "distance": 0.9, "entity": {"source_file": "data/invoice/a.png"}},
        ]

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return [self._hits]


@pytest.fixture
def fake_client(monkeypatch):
    client = FakeMilvusClient()
    monkeypatch.setattr(vector_retrieval, "get_milvus_client", lambda: client)
    monkeypatch.setattr(vector_retrieval, "embed_query", lambda q: [0.1, 0.2])
    return client


def test_search_passes_filter_expr_to_milvus(fake_client):
    vector_retrieval.vector_search("金额超过1000元的发票", filter_expr='ticket_type == "invoice"')
    assert fake_client.calls[0]["filter"] == 'ticket_type == "invoice"'


def test_search_without_filter_uses_empty_string(fake_client):
    vector_retrieval.vector_search("张三的火车票")
    # 空过滤保持 Milvus 语义:filter="" 等价不过滤
    assert fake_client.calls[0]["filter"] == ""


def test_search_result_structure(fake_client):
    hits = vector_retrieval.vector_search("张三的火车票", top_n=3)
    assert fake_client.calls[0]["limit"] == 3
    assert hits[0]["id"] == "ticket_a"
    assert hits[0]["vector_score"] == 0.9
    assert hits[0]["source_file"] == "data/invoice/a.png"

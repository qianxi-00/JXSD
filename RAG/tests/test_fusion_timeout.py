"""融合线路的**有界降级**测试：单路卡死不能拖垮整条问答。

背景（真机踩到的坑）：`_sql_evidence` 走 psycopg 连 PG，沙箱里 libpq 的 GSS 协商
会永久卡住（README §8.1 有实测记录）。当时的表现是：一句聚合问题把整条融合线路
挂满 10 分钟超时，而票据与图谱证据早就拿到了。

这里用一个"永远不返回的函数"复现那种卡死，验证：
    - `_call_with_timeout` 会按时放弃并把卡住的那个线程留在后台；
    - `_sql_evidence` 超时后返回 None（按"没有结构化证据"处理），不往上抛；
    - 整条 `answer_fusion` 在 SQL 卡死时仍然能出答案（用票据/图谱证据）。
"""

import time

from pipeline import fusion


def _never_returns():
    time.sleep(3600)


class TestCallWithTimeout:
    def test_timeout_is_bounded(self):
        started = time.perf_counter()
        ok, payload = fusion._call_with_timeout(_never_returns, 0.2)
        elapsed = time.perf_counter() - started
        assert ok is False and payload is None
        # 有界：必须在超时值附近返回，而不是等满 3600 秒
        assert elapsed < 2.0, f"应在 0.2s 左右返回，实际 {elapsed:.2f}s"

    def test_exception_is_returned_not_raised(self):
        def boom():
            raise ValueError("连不上")

        ok, payload = fusion._call_with_timeout(boom, 1.0)
        assert ok is False
        assert isinstance(payload, ValueError)

    def test_value_passes_through(self):
        ok, payload = fusion._call_with_timeout(lambda: {"a": 1}, 1.0)
        assert ok is True and payload == {"a": 1}


class TestSqlEvidenceDegrades:
    def test_timeout_returns_none_without_raising(self, monkeypatch):
        class FakeTool:
            @staticmethod
            def invoke(_payload):
                return _never_returns()

        import agentic.text_to_sql as t2s

        monkeypatch.setattr(t2s, "query_ticket_db", FakeTool)
        monkeypatch.setattr(fusion, "SQL_EVIDENCE_TIMEOUT_S", 0.2)
        assert fusion._sql_evidence("黄帅一共报销了多少钱") is None

    def test_bad_status_returns_none(self, monkeypatch):
        class FakeTool:
            @staticmethod
            def invoke(_payload):
                return {"status": "invalid_sql", "message": "只允许单条语句"}

        import agentic.text_to_sql as t2s

        monkeypatch.setattr(t2s, "query_ticket_db", FakeTool)
        assert fusion._sql_evidence("随便问") is None


class TestFusionSurvivesStuckSql:
    def test_answer_still_produced_when_sql_hangs(self, monkeypatch):
        """SQL 卡死时，票据证据照常出答案（这正是"分层降级"的意义）。"""
        monkeypatch.setattr(fusion, "route_query", lambda q: (True, "direct"))
        monkeypatch.setattr(fusion, "extract_ticket_filters", lambda q: {})
        monkeypatch.setattr(fusion, "build_milvus_filter", lambda f: "")
        monkeypatch.setattr(
            fusion, "_recall_tickets", lambda queries, filters: ([{"id": "t1"}], "")
        )
        monkeypatch.setattr(fusion, "rerank", lambda q, docs, top_k=None: docs)
        monkeypatch.setattr(fusion, "_graph_evidence", lambda q: {})
        monkeypatch.setattr(fusion, "SQL_EVIDENCE_TIMEOUT_S", 0.2)

        class FakeTool:
            @staticmethod
            def invoke(_payload):
                return _never_returns()

        import agentic.text_to_sql as t2s

        monkeypatch.setattr(t2s, "query_ticket_db", FakeTool)
        monkeypatch.setattr(fusion, "generate_answer", lambda q, c: ("票据证据支撑的答案", None))
        monkeypatch.setattr(fusion, "needs_aggregate", lambda q: True)  # 强制走 SQL 那条路

        started = time.perf_counter()
        result = fusion.answer_fusion("黄帅一共报销了多少钱")
        elapsed = time.perf_counter() - started

        assert result.answer == "票据证据支撑的答案"
        assert result.sql is None  # 超时 → 没有结构化证据
        assert result.system_error == ""  # 也不算系统故障
        assert elapsed < 3.0, f"不该被 SQL 卡死拖住，实际 {elapsed:.2f}s"

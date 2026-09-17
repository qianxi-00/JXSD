"""接口层补齐的测试（优化篇/基础篇「接口职责」欠账 T8）。

覆盖：健康检查的聚合口径、SSE 的 `query_type`/`processing_time`、WebSocket 收发、
`QA_CACHE_ENABLED` 总开关、启动预热的开关行为。

全部离线：四条线路的 runner 与探针都打桩，`APP_WARMUP` 在夹具里关掉
（真预热会调 embedding 接口 —— 单测不该联网）。
"""

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / "RAG"))


@pytest.fixture
def main_module(monkeypatch):
    """导入 app.main 并关掉启动预热（避免单测联网调 embedding）。"""
    from app import main

    monkeypatch.setattr(main.settings.app, "warmup", False)
    return main


@pytest.fixture
def client(main_module):
    with TestClient(main_module.app) as test_client:
        yield test_client


class FakeEvents:
    """替掉 `answer_events`：按脚本吐事件（含 route / no_evidence / token / done）。"""

    def __init__(self, events):
        self.events = events
        self.calls: list[tuple] = []

    def __call__(self, question, mode=None, history=None, stream=True):
        self.calls.append((question, mode, history, stream))
        events = self.events

        async def gen():
            for ev in events:
                yield ev

        return gen()


class TestHealth:
    def test_reports_ok_when_all_probes_pass(self, client, main_module, monkeypatch):
        monkeypatch.setattr(main_module, "_health_probe", lambda name, fn, timeout=3.0: {"ok": True, "detail": "x"})
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["failed"] == []
        assert set(body["checks"]) == {"redis", "milvus", "postgres", "neo4j"}
        # 开关状态也一并透出：排查"为什么没有命中缓存"时第一个要看的就是它
        assert body["qa_cache_enabled"] is True

    def test_degrades_but_does_not_500(self, client, main_module, monkeypatch):
        """某个依赖挂了：**不抛 5xx**，只把整体标成 degraded 并列出失败项。

        为什么刻意不返回 5xx：那会让"进程活着、某个依赖没起"看起来像"服务死了"，
        而 HTTP 层是给负载均衡/监控看的，body 里的 status 才是判据。
        """
        monkeypatch.setattr(
            main_module,
            "_health_probe",
            lambda name, fn, timeout=3.0: {"ok": name != "postgres", "detail": "boom" if name == "postgres" else "x"},
        )
        resp = client.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "degraded"
        assert body["failed"] == ["postgres"]

    def test_probe_timeout_is_reported_not_raised(self, main_module):
        """超时要被描述成"超时"，而不是抛出去 —— 健康检查自己绝不能挂住。"""
        result = main_module._health_probe("slow", lambda: None, timeout=0.05)
        # fn 立即返回 None ⇒ 视为失败（探针必须有返回值才算通）
        assert result["ok"] is False

    def test_probe_is_bounded_on_a_real_hang(self, main_module):
        """真有卡死的调用时，探针也必须**按时返回**（这是 /api/health 不挂死的根据）。

        用 `time.sleep(5)` 模拟卡住的客户端调用：探针应在 ~0.2s 返回并报"超时"。
        这条是本机那个"libpq 的 GSS 协商永久卡住、connect_timeout 无效"教训的回归。
        """
        import time as _time

        started = _time.perf_counter()
        result = main_module._health_probe("hang", lambda: _time.sleep(5), timeout=0.2)
        elapsed = _time.perf_counter() - started
        assert result["ok"] is False
        assert "超时" in result["detail"]
        assert elapsed < 2.0, f"探针没有按上限返回：{elapsed:.2f}s"

    def test_probe_wraps_exception(self, main_module):
        def boom():
            raise RuntimeError("连接被拒绝")

        result = main_module._health_probe("bad", boom, timeout=1.0)
        assert result["ok"] is False
        assert "连接被拒绝" in result["detail"]


class TestQueryTypeMapping:
    @pytest.mark.parametrize(
        ("event", "current", "expected"),
        [
            ({"type": "cache_hit", "mode": "exact"}, "rag", "cache"),
            ({"type": "cache_hit", "mode": "preset"}, "rag", "faq"),
            ({"type": "route", "route": "direct"}, "rag", "llm"),
            ({"type": "route", "route": "rag"}, "llm", "rag"),
            ({"type": "no_evidence"}, "rag", "rag_rejected"),
            ({"type": "token", "text": "x"}, "rag", "rag"),  # 其它事件不改状态
        ],
    )
    def test_mapping(self, main_module, event, current, expected):
        assert main_module._classify_query_type(event, current) == expected


class TestSseStream:
    def test_payloads_carry_query_type_and_summary(self, client, main_module, monkeypatch):
        fake = FakeEvents([
            {"type": "start", "mode": "basic"},
            {"type": "cache_hit", "mode": "exact", "matched_question": "q", "similarity": 1.0, "sources": []},
            {"type": "token", "text": "缓存", "mode": "basic"},
            {"type": "done", "answer": "缓存答案", "sources": [{"id": "t1"}], "cache_hit": "exact", "mode": "basic"},
        ])
        monkeypatch.setattr(main_module, "answer_events", fake)
        with client.stream("POST", "/api/chat/stream", json={"question": "q", "mode": "basic"}) as resp:
            raw = "".join(resp.iter_text())

        lines = [ln for ln in raw.splitlines() if ln.startswith("data: ")]
        assert lines[-1] == "data: [DONE]"
        payloads = [json.loads(ln[len("data: "):]) for ln in lines[:-1]]
        # 每条 token 都带 query_type（缓存命中 ⇒ cache）
        assert payloads[0]["delta"] == "缓存"
        assert payloads[0]["query_type"] == "cache"
        # 收尾事件给全量结果 + 服务端总耗时
        summary = payloads[-1]
        assert summary["done"] is True
        assert summary["answer"] == "缓存答案"
        assert summary["sources"] == [{"id": "t1"}]
        assert summary["query_type"] == "cache"
        assert summary["processing_time"] >= 0

    def test_direct_route_is_typed_llm(self, client, main_module, monkeypatch):
        fake = FakeEvents([
            {"type": "route", "route": "direct", "mode": "basic", "rewrite": "direct"},
            {"type": "token", "text": "直答", "mode": "basic"},
            {"type": "done", "answer": "直答", "sources": [], "cache_hit": None, "mode": "basic"},
        ])
        monkeypatch.setattr(main_module, "answer_events", fake)
        resp = client.post("/api/chat/stream", json={"question": "1+1", "mode": "basic"})
        assert '"query_type": "llm"' in resp.text


class TestWebSocket:
    def test_streams_tokens_then_done(self, client, main_module, monkeypatch):
        fake = FakeEvents([
            {"type": "route", "route": "rag", "mode": "basic", "rewrite": "direct"},
            {"type": "token", "text": "答案", "mode": "basic"},
            {"type": "done", "answer": "答案", "sources": [], "cache_hit": None, "mode": "basic"},
        ])
        monkeypatch.setattr(main_module, "answer_events", fake)
        with client.websocket_connect("/api/ws") as ws:
            ws.send_text(json.dumps({"question": "问", "mode": "basic"}))
            first = ws.receive_json()
            assert first["type"] == "token"
            assert first["delta"] == "答案"
            assert first["query_type"] == "rag"
            done = ws.receive_json()
            assert done["type"] == "done"
            assert done["answer"] == "答案"
            assert done["processing_time"] >= 0

    def test_bad_json_returns_error_frame_not_disconnect(self, client, main_module, monkeypatch):
        monkeypatch.setattr(main_module, "answer_events", FakeEvents([]))
        with client.websocket_connect("/api/ws") as ws:
            ws.send_text("这不是 JSON")
            frame = ws.receive_json()
            assert frame["type"] == "error"
            assert "JSON" in frame["message"]
            # 连接还能继续用（回错误帧而不是断开，客户端能自己修）
            ws.send_text(json.dumps({"question": ""}))
            assert ws.receive_json()["type"] == "error"

    def test_second_question_reuses_connection(self, client, main_module, monkeypatch):
        fake = FakeEvents([
            {"type": "token", "text": "一", "mode": "basic"},
            {"type": "done", "answer": "一", "sources": [], "cache_hit": None, "mode": "basic"},
        ])
        monkeypatch.setattr(main_module, "answer_events", fake)
        with client.websocket_connect("/api/ws") as ws:
            for _ in range(2):
                ws.send_text(json.dumps({"question": "问", "mode": "basic"}))
                assert ws.receive_json()["delta"] == "一"
                assert ws.receive_json()["type"] == "done"
        assert len(fake.calls) == 2


class TestCacheSwitch:
    def test_cached_route_skips_cache_when_disabled(self, monkeypatch):
        """`QA_CACHE_ENABLED=false` ⇒ ③④ 既不查也不写（一个开关管住四条线路）。"""
        from pipeline import modes

        calls: list[tuple] = []

        class Fake:
            def lookup(self, question, route=""):
                calls.append(("lookup", route))
                return {"cache_hit": "exact", "answer": "旧的", "sources": []}

            def store(self, question, answer, sources, route=""):
                calls.append(("store", route))

        monkeypatch.setattr(modes, "_answer_cache", lambda: Fake())
        monkeypatch.setattr(modes.settings.qa_cache, "enabled", False)
        out = modes._cached_route("问题", [], "fusion", lambda: {"answer": "新答", "sources": [], "extra": {}})
        assert out["answer"] == "新答"
        assert calls == [], "关掉总开关后不该有任何缓存读写"

    def test_cached_route_uses_cache_when_enabled(self, monkeypatch):
        from pipeline import modes

        calls: list[tuple] = []

        class Fake:
            def lookup(self, question, route=""):
                calls.append(("lookup", route))
                # 刻意返回 None：模拟"缓存未命中"

            def store(self, question, answer, sources, route=""):
                calls.append(("store", route))

        monkeypatch.setattr(modes, "_answer_cache", lambda: Fake())
        monkeypatch.setattr(modes.settings.qa_cache, "enabled", True)
        modes._cached_route("问题", [], "fusion", lambda: {"answer": "新答", "sources": [], "extra": {}})
        assert calls == [("lookup", "fusion"), ("store", "fusion")]

    def test_agentic_and_basic_routes_read_the_switch(self, monkeypatch):
        """② 与 ① 也要读同一个开关（否则"关缓存"只关住 ③④）。"""
        from pipeline import modes

        monkeypatch.setattr(modes.settings.qa_cache, "enabled", False)
        captured: dict = {}

        def fake_answer(question, history, route="", info=None, **kwargs):
            captured["use_cache"] = kwargs.get("use_cache")
            return "答"

        import agentic.finance_agent as fa

        monkeypatch.setattr(fa, "answer_financial_question", fake_answer)
        modes._run_agentic("问", [])
        assert captured["use_cache"] is False


class TestWarmupSwitch:
    def test_warmup_runs_when_enabled(self, monkeypatch):
        from app import main

        called = {"n": 0}
        monkeypatch.setattr(main, "_warmup", lambda: called.__setitem__("n", called["n"] + 1))
        monkeypatch.setattr(main.settings.app, "warmup", True)
        with TestClient(main.app):
            pass
        assert called["n"] == 1

    def test_warmup_skipped_when_disabled(self, monkeypatch):
        from app import main

        called = {"n": 0}
        monkeypatch.setattr(main, "_warmup", lambda: called.__setitem__("n", called["n"] + 1))
        monkeypatch.setattr(main.settings.app, "warmup", False)
        with TestClient(main.app):
            pass
        assert called["n"] == 0

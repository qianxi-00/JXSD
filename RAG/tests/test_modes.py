"""四条 RAG 线路的注册表与统一入口测试（`pipeline/modes.py` + `core/routes.py`）。

这一层的价值全在**契约**上，所以用例盯的也是契约：
    - 四条线路都在注册表里，且 `list_modes()` 的顺序/字段稳定（前端按它渲染）；
    - `answer()` 对任何输入都返回**同一形状**（前端不该为线路写四套解析）；
    - 未知 mode 落到基础线路而不是报错（旧前端不传 mode 也能用）；
    - history 会被清洗（非法条目丢掉，不因为前端多传字段就崩）；
    - 单轮线路**不吃** history（这是它们的定义，不是漏传）。
"""

import pytest

from core import routes
from pipeline import modes


class TestRegistry:
    def test_all_four_routes_registered(self):
        assert set(modes.MODES) == set(routes.ROUTE_KEYS)
        assert len(routes.ROUTE_KEYS) == 4

    def test_list_modes_shape_and_order(self):
        listed = modes.list_modes()
        # 顺序即展示顺序（从简单到复杂）：前端不要自己排序，这里钉住
        assert [item["key"] for item in listed] == list(routes.ROUTE_KEYS)
        for item in listed:
            assert set(item) == {"key", "label", "summary", "supports_history", "streaming"}
            assert item["label"] and item["summary"]

    def test_history_support_flags_match_docs(self):
        """支持多轮的只有 agentic 与 fusion；基础与图谱是单轮（README 这么写的）。"""
        assert routes.ROUTE_SUPPORTS_HISTORY[routes.ROUTE_BASIC] is False
        assert routes.ROUTE_SUPPORTS_HISTORY[routes.ROUTE_GRAPH] is False
        assert routes.ROUTE_SUPPORTS_HISTORY[routes.ROUTE_AGENTIC] is True
        assert routes.ROUTE_SUPPORTS_HISTORY[routes.ROUTE_FUSION] is True


class TestNormalizeMode:
    def test_unknown_mode_falls_back_to_basic(self):
        assert modes.normalize_mode("graphrag") == routes.ROUTE_BASIC
        assert modes.normalize_mode(None) == routes.ROUTE_BASIC
        assert modes.normalize_mode("") == routes.ROUTE_BASIC

    def test_known_modes_pass_through(self):
        for key in routes.ROUTE_KEYS:
            assert modes.normalize_mode(key) == key

    def test_chinese_label_is_accepted(self):
        """前端（Chainlit Select 用 label→value 映射）可能回传显示名而不是键。

        实测背景：真机在浏览器里切换线路时，设置项回传的形态与预期不一致是常见坑；
        这里把"显示名也能解析"钉住，避免又出现"切了线路没生效"却只留一行 warning。
        """
        for key in routes.ROUTE_KEYS:
            assert modes.normalize_mode(routes.ROUTE_LABELS[key]) == key

    def test_label_with_frontend_suffix_is_accepted(self):
        """前端在选择器文案后面拼了"（支持多轮）"之类的后缀也要能解析。"""
        label = routes.ROUTE_LABELS[routes.ROUTE_FUSION]
        assert modes.normalize_mode(f"{label}（支持多轮）") == routes.ROUTE_FUSION

    def test_truly_unknown_value_falls_back(self):
        assert modes.normalize_mode("这是前端乱传的") == routes.ROUTE_BASIC


class TestHistoryNormalization:
    def test_accepts_dicts_and_objects(self):
        class Msg:
            def __init__(self, role, content):
                self.role = role
                self.content = content

        cleaned = modes._normalize_history(
            [
                {"role": "user", "content": "第一句"},
                Msg("assistant", "第一答"),
                {"role": "system", "content": "不该进来"},  # 角色不合法
                {"role": "user", "content": "   "},  # 空白内容
                "就是个字符串",  # 结构不合法
                None,
            ]
        )
        assert cleaned == [
            {"role": "user", "content": "第一句"},
            {"role": "assistant", "content": "第一答"},
        ]

    def test_none_is_empty(self):
        assert modes._normalize_history(None) == []


class TestAnswerContract:
    """`answer()` 的返回形状与失败降级：四条线路都走同一条出口。"""

    def test_shape_is_identical_across_routes(self, monkeypatch):
        def runner(question, history):
            return {"answer": f"答:{question}", "sources": [], "cache_hit": None, "extra": {}}

        monkeypatch.setattr(
            modes,
            "MODES",
            {key: modes.ModeSpec(key, runner) for key in routes.ROUTE_KEYS},
        )
        shapes = set()
        for key in routes.ROUTE_KEYS:
            result = modes.answer("问题", key)
            shapes.add(tuple(sorted(result)))
            assert result["mode"] == key
            assert result["answer"] == "答:问题"
            assert result["system_error"] == ""
        assert len(shapes) == 1, "四条线路的返回键必须完全一致"

    def test_runner_exception_becomes_readable_answer(self, monkeypatch):
        """线路内部炸了（import 失败/依赖缺失）不能让服务 500，要变成可读文案 + system_error。"""

        def boom(question, history):
            raise RuntimeError("Neo4j 连不上")

        monkeypatch.setattr(modes, "MODES", {routes.ROUTE_GRAPH: modes.ModeSpec(routes.ROUTE_GRAPH, boom)})
        result = modes.answer("问题", routes.ROUTE_GRAPH)
        assert result["mode"] == routes.ROUTE_GRAPH
        assert "Neo4j 连不上" in result["answer"]
        assert result["system_error"] == "RuntimeError: Neo4j 连不上"
        assert result["events"][0]["type"] == "error"

    def test_agentic_runner_forwards_route_scope_and_info(self, monkeypatch):
        """Agentic 必须把**线路作用域**传给缓存，否则会命中基础线路的缓存答案。"""
        captured = {}

        def fake_answer(question, history, route="", info=None, **kwargs):
            captured["route"] = route
            captured["history"] = history
            captured["use_cache"] = kwargs.get("use_cache")
            if info is not None:
                info.update({"cache_hit": "exact", "from_cache": True})
            return "Agent 答"

        import agentic.finance_agent as fa

        monkeypatch.setattr(fa, "answer_financial_question", fake_answer)
        result = modes._run_agentic("问题", [])
        assert captured["route"] == routes.ROUTE_AGENTIC
        assert captured["use_cache"] is True
        assert result["cache_hit"] == "exact"
        assert result["extra"]["from_cache"] is True

    def test_agentic_runner_disables_cache_when_history_present(self, monkeypatch):
        """多轮追问必须绕过缓存：缓存键里只有问题文本，会串上一轮语境的答案。"""
        captured = {}

        def fake_answer(question, history, route="", info=None, **kwargs):
            captured["use_cache"] = kwargs.get("use_cache")
            return "答"

        import agentic.finance_agent as fa

        monkeypatch.setattr(fa, "answer_financial_question", fake_answer)
        modes._run_agentic("那乐艳的呢？", [{"role": "user", "content": "万宁的火车票票号"}])
        assert captured["use_cache"] is False

    def test_agentic_runner_does_not_mutate_caller_history(self, monkeypatch):
        """传进去的历史必须是副本：否则前端 session 里的历史会被链路悄悄改写。"""
        original = [{"role": "user", "content": "上一句"}]

        def fake_answer(question, history, route="", info=None, **kwargs):
            history.append({"role": "user", "content": "本轮"})  # 模拟链路就地追加
            return "答"

        import agentic.finance_agent as fa

        monkeypatch.setattr(fa, "answer_financial_question", fake_answer)
        modes._run_agentic("问题", original)
        assert original == [{"role": "user", "content": "上一句"}]


class TestBasicRunnerIsSingleTurn:
    def test_basic_runner_ignores_history(self, monkeypatch):
        """基础线路是单轮：传了 history 也不该把它拼进问题（它的定义如此）。"""
        captured = {}

        class FakePipeline:
            async def run_and_collect(self, question):
                captured["question"] = question
                return {"answer": "答", "sources": [], "cache_hit": None}

        monkeypatch.setattr(modes, "_basic_pipeline", lambda: FakePipeline())
        modes._run_basic("万宁的火车票", [{"role": "user", "content": "无关的上一句"}])
        assert captured["question"] == "万宁的火车票"


class TestCacheScope:
    def test_basic_scope_is_empty_string(self):
        """基础线路作用域必须是空串：老键（含 seed_details 播的 60 条）才继续有效。"""
        assert modes.cache_key_for(routes.ROUTE_BASIC) == ""
        assert routes.CACHE_SCOPE_BASIC == ""

    def test_other_routes_have_own_scope(self):
        assert modes.cache_key_for(routes.ROUTE_AGENTIC) == routes.ROUTE_AGENTIC
        assert modes.cache_key_for(routes.ROUTE_FUSION) == routes.ROUTE_FUSION
        # 四条线路的作用域互不相同（否则会互相串答案）
        scopes = [modes.cache_key_for(key) for key in routes.ROUTE_KEYS]
        assert len(set(scopes)) == len(scopes)

    def test_unknown_route_gets_own_scope(self):
        """拼错的线路不要落到基础线路的缓存里 —— 否则"配置没生效"会很难归因。"""
        assert routes.cache_scope("basicc") == "basicc"


class FakeCache:
    """AnswerCache 的替身（记录 lookup/store 的 route，便于断言作用域）。"""

    def __init__(self, hit=None):
        self.hit = hit
        self.calls: list[tuple] = []

    def lookup(self, question, route=""):
        self.calls.append(("lookup", question, route))
        return self.hit

    def store(self, question, answer, sources, route=""):
        self.calls.append(("store", question, answer, route))


class TestCachedRoute:
    """③④ 接缓存的三条纪律：命中跳过执行、多轮不查不写、作用域按线路分。"""

    def _install(self, monkeypatch, hit=None):
        fake = FakeCache(hit)
        monkeypatch.setattr(modes, "_answer_cache", lambda: fake)
        return fake

    def test_hit_skips_producer_entirely(self, monkeypatch):
        fake = self._install(
            monkeypatch,
            hit={"cache_hit": "exact", "answer": "上次的答案", "sources": [{"id": "t1"}]},
        )
        called = {"n": 0}

        def produce():
            called["n"] += 1
            return {"answer": "新答案", "sources": [], "extra": {}}

        out = modes._cached_route("董文的航班", [], routes.ROUTE_GRAPH, produce)
        assert called["n"] == 0, "命中缓存时不该再跑检索与生成"
        assert out["answer"] == "上次的答案"
        assert out["cache_hit"] == "exact"
        # 命中时不把上一次的明细冒充成本次证据，只给一个"来自缓存"的标记
        assert out["extra"] == {"from_cache": True}
        assert out["sources"] == [{"id": "t1"}]
        assert ("lookup", "董文的航班", routes.ROUTE_GRAPH) in fake.calls

    def test_miss_runs_producer_and_stores_with_same_scope(self, monkeypatch):
        fake = self._install(monkeypatch)
        out = modes._cached_route(
            "董文的航班",
            [],
            routes.ROUTE_GRAPH,
            lambda: {"answer": "新答案", "sources": [], "extra": {"communities": [1]}},
        )
        assert out["answer"] == "新答案"
        assert out["cache_hit"] is None
        assert out["extra"] == {"communities": [1]}
        # lookup 与 store 必须同一个 route（不一致会"存了但读不到"）
        assert fake.calls == [
            ("lookup", "董文的航班", routes.ROUTE_GRAPH),
            ("store", "董文的航班", "新答案", routes.ROUTE_GRAPH),
        ]

    def test_history_bypasses_cache_both_ways(self, monkeypatch):
        """多轮：既不查缓存也不写缓存（缓存键只有问题文本，会串上一轮语境的答案）。"""
        fake = self._install(monkeypatch, hit={"cache_hit": "exact", "answer": "上次", "sources": []})
        out = modes._cached_route(
            "那乐艳的呢？",
            [{"role": "user", "content": "万宁的火车票票号"}],
            routes.ROUTE_FUSION,
            lambda: {"answer": "本轮答案", "sources": [], "extra": {}},
        )
        assert out["answer"] == "本轮答案"
        assert fake.calls == [], "多轮时不该读写缓存"

    def test_empty_answer_is_not_stored(self, monkeypatch):
        fake = self._install(monkeypatch)
        modes._cached_route(
            "问题", [], routes.ROUTE_FUSION, lambda: {"answer": "", "sources": [], "extra": {}}
        )
        assert [c[0] for c in fake.calls] == ["lookup"], "空答案不能写缓存（会被当权威答案复用）"


class TestRouteCacheWiring:
    """③④ 必须真的走 `_cached_route`，且各用各的作用域。"""

    def test_graph_route_uses_graph_scope(self, monkeypatch):
        seen = {}

        def fake_cached(question, history, route, produce):
            seen["route"] = route
            seen["question"] = question
            return {"answer": "图答", "sources": [], "cache_hit": None, "extra": {}}

        monkeypatch.setattr(modes, "_cached_route", fake_cached)
        out = modes._run_graph("董文的航班是从哪到哪的？", [])
        assert seen["route"] == routes.ROUTE_GRAPH
        assert out["answer"] == "图答"

    def test_fusion_route_uses_fusion_scope_and_keeps_history(self, monkeypatch):
        seen = {}

        def fake_cached(question, history, route, produce):
            seen["route"] = route
            seen["history"] = history
            return {"answer": "融合答", "sources": [], "cache_hit": None, "extra": {}}

        monkeypatch.setattr(modes, "_cached_route", fake_cached)
        history = [{"role": "user", "content": "上一句"}]
        out = modes._run_fusion("那乐艳的呢？", history)
        assert seen["route"] == routes.ROUTE_FUSION
        # 融合线路吃 history，必须原样传进 produce（多轮检索靠它拼上一轮问题）
        assert seen["history"] == history
        assert out["answer"] == "融合答"

    def test_answer_cache_is_reused_across_calls(self, monkeypatch):
        """进程内复用同一个 AnswerCache（避免每次请求重载预设矩阵）。"""
        created = {"n": 0}

        class Fake:
            def __init__(self):
                created["n"] += 1

        import core.cache as cache_mod

        monkeypatch.setattr(cache_mod, "AnswerCache", Fake)
        monkeypatch.setattr(modes, "_ANSWER_CACHE", None)
        first = modes._answer_cache()
        second = modes._answer_cache()
        assert first is second
        assert created["n"] == 1


@pytest.mark.asyncio
async def test_answer_events_packages_non_streaming_routes(monkeypatch):
    """非基础线路把结果包成 start/mode/token/done 四类事件（前端按同一套渲染）。"""

    def runner(question, history):
        return {"answer": "融合答", "sources": [{"id": "t1"}], "cache_hit": None, "extra": {"sql": {"rows": [1]}}}

    monkeypatch.setattr(modes, "MODES", {routes.ROUTE_FUSION: modes.ModeSpec(routes.ROUTE_FUSION, runner)})
    events = [ev async for ev in modes.answer_events("问题", routes.ROUTE_FUSION)]
    types = [ev["type"] for ev in events]
    assert types == ["start", "mode", "token", "done"]
    assert events[2]["text"] == "融合答"
    # done 里必须带 extra：前端"证据构成"那一 Step 完全靠它
    assert events[-1]["extra"] == {"sql": {"rows": [1]}}
    assert all(ev["mode"] == routes.ROUTE_FUSION for ev in events)


@pytest.mark.asyncio
async def test_route_events_always_carry_route_field(monkeypatch):
    """契约：`route` 事件**必须**带 `route` 字段（前端按它分支）。

    真机踩过：非基础线路本想复用 `route` 事件名，结果发出去的事件里只有 `mode`，
    前端 `ev["route"]` 直接 KeyError，整条 on_message 中断 —— 页面上只有提问、
    连回答都不渲染。所以线路选择改用独立的 `mode` 事件，这里把两条契约一起钉住。
    """

    def runner(question, history):
        return {"answer": "答", "sources": [], "cache_hit": None, "extra": {}}

    monkeypatch.setattr(modes, "MODES", {routes.ROUTE_GRAPH: modes.ModeSpec(routes.ROUTE_GRAPH, runner)})
    events = [ev async for ev in modes.answer_events("问题", routes.ROUTE_GRAPH)]
    for ev in events:
        if ev["type"] == "route":
            assert "route" in ev, "route 事件必须带 route 字段（rag/direct）"
    # 线路选择事件是独立类型，且带前端要显示的文案
    mode_events = [ev for ev in events if ev["type"] == "mode"]
    assert len(mode_events) == 1
    assert mode_events[0]["label"] == routes.ROUTE_LABELS[routes.ROUTE_GRAPH]
    assert "route" not in mode_events[0]

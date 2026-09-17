"""融合线路（`pipeline/fusion.py`）的纯逻辑测试。

融合线路的"融合"体现在四个可控点上，用例逐个钉住：
    1. 聚合意图判定（决定要不要多花一次 Text-to-SQL）；
    2. 三类证据的**分层门控**（图谱证据有自己的分数下限，不与票据重排分数混排）；
    3. 裁决口径与来源标签进了提示词（数值以结构化统计为准、事实以票据原文为准）；
    4. 数字核验（证据里找不到的数字要提示，但**不改写正文**）。

真机行为（真连 Milvus/Neo4j/PG）不在这里 —— 那由 README §7.2 的端到端清单覆盖。
"""

import pytest

from pipeline import fusion


class TestAggregateIntent:
    @pytest.mark.parametrize(
        "question",
        [
            "黄帅今年高铁票一共报销了多少钱？",
            "于强的机票总金额是多少",
            "发票一共几张",
            "统计一下各公司的报销金额",
        ],
    )
    def test_aggregate_questions_detected(self, question):
        assert fusion.needs_aggregate(question) is True

    @pytest.mark.parametrize(
        "question",
        [
            "万宁的火车票票号是多少？",  # "多少"但没配钱/数量词 —— 事实型问题
            "何海燕的火车票票号是多少",
            "乐艳的火车票是哪天",
            "董文的航班是从哪到哪的？",
        ],
    )
    def test_fact_questions_not_aggregate(self, question):
        """事实型问题不该触发 SQL：多花约 8 秒且答不出"票号"这类细节。"""
        assert fusion.needs_aggregate(question) is False


class TestSelectCommunities:
    def test_filters_below_score_floor(self):
        graph = {
            "communities": [
                {"community_id": 0, "score": 0.9, "summary": "高"},
                {"community_id": 1, "score": 0.49, "summary": "低于下限"},
                {"community_id": 2, "score": 0.6, "summary": "够线"},
            ]
        }
        picked = fusion._select_communities(graph)
        assert [item["community_id"] for item in picked] == [0, 2]

    def test_caps_number_of_communities(self):
        graph = {
            "communities": [
                {"community_id": i, "score": 0.9, "summary": f"s{i}"}
                for i in range(fusion.GRAPH_MAX_COMMUNITIES + 4)
            ]
        }
        assert len(fusion._select_communities(graph)) == fusion.GRAPH_MAX_COMMUNITIES

    def test_missing_score_is_dropped(self):
        """没有分数的社区不能进提示词：无从判断相关性，宁可不放。"""
        assert fusion._select_communities({"communities": [{"community_id": 0, "summary": "无分数"}]}) == []

    def test_empty_graph(self):
        assert fusion._select_communities({}) == []


class TestBuildContext:
    def test_labels_and_priority_rules_present(self):
        context = fusion._build_context(
            "黄帅报销了多少",
            tickets=[{"id": "t1", "person": "黄帅"}],
            communities=[{"community_id": 3, "summary": "航空出行社区", "score": 0.7}],
            graph={"relationships": [{"source": "董文", "relation": "持有票据", "target": "T1"}]},
            sql={"sql": "SELECT 1", "rows": [{"total": 1691}]},
        )
        # 三类证据都要在，且有可识别的来源标签
        assert "【票据原文证据】" in context
        assert "【图谱社区摘要】" in context
        assert "【结构化统计】" in context
        assert "[票据1]" in context and "[图社区3]" in context
        # 裁决口径必须写进提示词，否则模型会自己乱选来源
        assert "数值问题以【结构化统计】为准" in context
        assert "1691" in context

    def test_empty_sections_are_omitted(self):
        """空段落不能留标题：模型会顺着空标题编（图谱提示词那处的同一教训）。

        只检查**段标题**（带全角括号的"（…以此为准）"那种）：
        末尾的"回答要求"里会出现 `【结构化统计】` 这类字样，那是裁决口径，必须一直在。
        """
        context = fusion._build_context("问题", tickets=[], communities=[], graph={}, sql=None)
        assert "【票据原文证据】（事实与票号以此为准）" not in context
        assert "【图谱社区摘要】（用于关系与全局视角，不作为数字依据）" not in context
        assert "【结构化统计】（数值以此为准，由 SQL 直接算出，未经模型转述）" not in context
        # 但"只依据证据作答"的约束要一直在
        assert "只依据上面的证据作答" in context


class TestVerifyNumbers:
    def test_number_in_evidence_passes(self):
        assert fusion.verify_numbers("合计 1691 元", "总金额 1691 元") == []

    def test_thousand_separator_and_trailing_zero_match(self):
        assert fusion.verify_numbers("合计 ¥1,691.00", "1691") == []
        assert fusion.verify_numbers("金额 498.90 元", "498.9") == []

    def test_number_missing_from_evidence_is_reported(self):
        assert fusion.verify_numbers("合计 1691 元", "证据里只有 2025 年") == ["1691"]

    def test_single_digit_citations_are_ignored(self):
        """引用编号（[票据1] 里的 1）是序号，报出来只会淹没真正要核对的数字。"""
        assert fusion.verify_numbers("[票据1] [票据2] 已核对", "证据文本") == []

    def test_single_digit_alone_is_ignored_but_long_number_is_not(self):
        """同一条回答里：长票号要核对得上、单字符编号要跳过。"""
        answer = "票号 T20230702063302 [票据1]"
        assert fusion.verify_numbers(answer, "票据 T20230702063302") == []
        # 换个证据（不含票号）→ 长号必须被报出来
        assert fusion.verify_numbers(answer, "只有无关文本") == ["20230702063302"]

    def test_digits_inside_long_ids_are_not_split_out(self):
        """票号里的连续数字不能被切出一段当金额（正则的前后向断言就是干这个的）。"""
        assert fusion.verify_numbers("票号 T20230702063302", "票据 T20230702063302") == []


class TestFallbacks:
    """证据全空时的两种出口必须可区分（与 Agentic 线路同一口径）。"""

    def _run(self, monkeypatch, tickets, communities, sql, system_error="", candidates=None):
        # 候选默认等于"期望留下的票据"：`answer_fusion` 只在**有候选**时才调 rerank，
        # 所以要让 rerank 生效就必须给候选（先前这里没给，于是 rerank 根本没被调用）。
        candidates = tickets if candidates is None else candidates
        monkeypatch.setattr(fusion, "route_query", lambda q: (True, "direct"))
        monkeypatch.setattr(fusion, "rewrite_query", lambda q, m: [])
        monkeypatch.setattr(fusion, "extract_ticket_filters", lambda q: {})
        monkeypatch.setattr(fusion, "build_milvus_filter", lambda f: "")
        monkeypatch.setattr(fusion, "_recall_tickets", lambda queries, filters: (candidates, ""))

        def fake_rerank(*a, **kw):
            if system_error:
                raise RuntimeError(system_error)
            return tickets

        monkeypatch.setattr(fusion, "rerank", fake_rerank)
        monkeypatch.setattr(
            fusion,
            "_graph_evidence",
            lambda q: {"communities": communities, "nodes": [], "relationships": []},
        )
        monkeypatch.setattr(fusion, "_sql_evidence", lambda q: sql)
        monkeypatch.setattr(fusion, "generate_answer", lambda q, c: ("生成的回答", None))
        return fusion.answer_fusion("黄帅的票一共多少钱")

    def test_no_evidence_gives_conservative_reply(self, monkeypatch):
        from core.prompts import NO_EVIDENCE_REPLY

        result = self._run(monkeypatch, [], [], None)
        assert result.answer == NO_EVIDENCE_REPLY
        assert result.system_error == ""

    def test_rerank_failure_is_reported_as_system_error(self, monkeypatch):
        """重排挂了是"没查成"，不能伪装成"库里没有"。"""
        result = self._run(monkeypatch, [], [], None, system_error="网关 500", candidates=[{"id": "t1"}])
        assert result.system_error.startswith("重排失败")
        assert "暂时不可用" in result.answer

    def test_generation_uses_evidence(self, monkeypatch):
        """有证据时必须真的调生成（而不是走保守回复）。"""
        result = self._run(monkeypatch, [{"id": "t1"}], [], {"sql": "SELECT 1", "rows": [{"a": 1}]})
        assert result.answer == "生成的回答"
        assert result.tickets == [{"id": "t1"}]
    def test_direct_route_skips_retrieval(self, monkeypatch):
        """路由判定不需要检索时，一次召回都不该发生（与基础线路的 direct 分支一致）。"""
        monkeypatch.setattr(fusion, "route_query", lambda q: (False, "direct"))
        called = {"recall": False}

        def boom(*a, **kw):
            called["recall"] = True
            return ([], "")

        monkeypatch.setattr(fusion, "_recall_tickets", boom)
        monkeypatch.setattr(fusion, "generate_answer", lambda q, c: ("直答", None))
        result = fusion.answer_fusion("1+1等于几")
        assert called["recall"] is False
        assert result.answer == "直答"


class TestMultiTurn:
    def test_previous_question_is_appended_to_retrieval_text(self, monkeypatch):
        """多轮：只用最朴素的办法——把上一轮问题拼进检索文本（不做代词消解模型）。"""
        monkeypatch.setattr(fusion, "route_query", lambda q: (True, "direct"))
        monkeypatch.setattr(fusion, "extract_ticket_filters", lambda q: {})
        monkeypatch.setattr(fusion, "build_milvus_filter", lambda f: "")
        captured = {}

        def fake_recall(queries, filters):
            captured["queries"] = queries
            return ([], "")

        monkeypatch.setattr(fusion, "_recall_tickets", fake_recall)
        monkeypatch.setattr(fusion, "_graph_evidence", lambda q: {})
        monkeypatch.setattr(fusion, "_sql_evidence", lambda q: None)
        fusion.answer_fusion(
            "那乐艳的呢？",
            history=[
                {"role": "user", "content": "万宁的火车票票号是多少？"},
                {"role": "assistant", "content": "T2023..."},
            ],
        )
        assert captured["queries"][0] == "万宁的火车票票号是多少？ 那乐艳的呢？"

    def test_without_history_question_is_untouched(self, monkeypatch):
        monkeypatch.setattr(fusion, "route_query", lambda q: (True, "direct"))
        monkeypatch.setattr(fusion, "extract_ticket_filters", lambda q: {})
        monkeypatch.setattr(fusion, "build_milvus_filter", lambda f: "")
        captured = {}
        monkeypatch.setattr(
            fusion, "_recall_tickets", lambda queries, filters: (captured.setdefault("q", queries), "")
        )
        monkeypatch.setattr(fusion, "_graph_evidence", lambda q: {})
        monkeypatch.setattr(fusion, "_sql_evidence", lambda q: None)
        fusion.answer_fusion("那乐艳的呢？", history=None)
        assert captured["q"] == ["那乐艳的呢？"]

"""T6 图谱健康度三指标的纯函数测试。

指标本身必须可验证，否则"孤点 0"这种读数没人能证伪。
这里全部用构造数据，不连 Neo4j、不调模型。
"""

import pytest

from graph_rag import quality


class TestNormalizeEntityName:
    def test_fullwidth_and_case_and_blank(self):
        # 全角→半角（NFKC）、大小写统一、空白与标点去掉
        assert quality.normalize_entity_name("ＡＣＭＥ 公司") == quality.normalize_entity_name("acme公司")
        assert quality.normalize_entity_name(" 张 三 ") == quality.normalize_entity_name("张三")
        # 中英文标点、连字符、括号一律视为同一实体
        assert quality.normalize_entity_name("(张三)") == quality.normalize_entity_name("张三")
        assert quality.normalize_entity_name("中国-银行") == quality.normalize_entity_name("中国、银行")

    def test_keeps_original_distinction(self):
        # 归一化只用于"发现疑似重复"，不能把真正不同的实体算成同一个
        assert quality.normalize_entity_name("张三") != quality.normalize_entity_name("张四")
        assert quality.normalize_entity_name("差旅费") != quality.normalize_entity_name("出差费用")

    def test_handles_empty_and_none(self):
        assert quality.normalize_entity_name(None) == ""
        assert quality.normalize_entity_name("   ") == ""
        assert quality.normalize_entity_name("（）") == ""


class TestIsolatedNodeRate:
    def test_no_isolated_nodes(self):
        nodes = [{"name": "A"}, {"name": "B"}]
        rels = [{"source": "A", "target": "B"}]
        result = quality.isolated_node_rate(nodes, rels)
        assert result["total"] == 2
        assert result["isolated_count"] == 0
        assert result["rate"] == 0.0
        assert result["isolated"] == []

    def test_counts_node_without_any_relationship(self):
        nodes = [{"name": "A"}, {"name": "B"}, {"name": "孤儿"}]
        rels = [{"source": "A", "target": "B"}]
        result = quality.isolated_node_rate(nodes, rels)
        assert result["isolated_count"] == 1
        assert result["isolated"] == ["孤儿"]
        assert result["rate"] == pytest.approx(1 / 3)

    def test_self_loop_is_still_isolated(self):
        """自环（a -> a）说明它只和自己有关系，按课案"与任何实体都没有链接"的口径仍算孤点。

        ⚠ 这条与 Cypher 侧 `NOT (e)-[:RELATES_TO]-()` 的口径**不同**（那句会把自环当连接），
        所以健康度采集不能直接用那句话，见 health.collect 的实现。
        """
        nodes = [{"name": "自环"}]
        rels = [{"source": "自环", "target": "自环"}]
        result = quality.isolated_node_rate(nodes, rels)
        assert result["isolated_count"] == 1

    def test_direction_does_not_matter(self):
        # 关系是有向边，但"连上了"这件事两个端点都算
        nodes = [{"name": "A"}, {"name": "B"}]
        rels = [{"source": "B", "target": "A"}]
        assert quality.isolated_node_rate(nodes, rels)["isolated_count"] == 0

    def test_empty_graph_rate_is_zero_not_crash(self):
        result = quality.isolated_node_rate([], [])
        assert result == {"total": 0, "isolated_count": 0, "rate": 0.0, "isolated": []}

    def test_ignores_half_written_relationship(self):
        # 端点缺失的关系（导入中途）不能让两个节点都"被连上"
        nodes = [{"name": "A"}]
        rels = [{"source": "A", "target": ""}]
        assert quality.isolated_node_rate(nodes, rels)["isolated_count"] == 1


class TestNormalizedDuplicateRate:
    def test_groups_whitespace_and_case_variants(self):
        names = ["ACME", "acme ", "ＡＣＭＥ"]
        result = quality.normalized_duplicate_rate(names)
        assert result["duplicate_count"] == 3
        assert result["rate"] == pytest.approx(1.0)
        assert result["groups"] == {"acme": ["ACME", "acme ", "ＡＣＭＥ"]}

    def test_reports_partial_overlap(self):
        names = ["张三", "张三", "李四"]
        result = quality.normalized_duplicate_rate(names)
        assert result["duplicate_count"] == 2
        assert result["rate"] == pytest.approx(2 / 3)

    def test_distinct_names_have_no_duplicate(self):
        result = quality.normalized_duplicate_rate(["张三", "李四", "王五"])
        assert result["duplicate_count"] == 0
        assert result["groups"] == {}

    def test_blank_only_names_are_not_grouped(self):
        """名字归一化后为空串的（如"（）"）不能互相算成"重复组" —— 它们是无名节点，不是同一个实体。"""
        names = ["（）", "()", " "]
        result = quality.normalized_duplicate_rate(names)
        assert result["duplicate_count"] == 0
        assert result["groups"] == {}

    def test_empty_graph(self):
        assert quality.normalized_duplicate_rate([])["rate"] == 0.0


class TestLinkDecisionMetrics:
    """课案口径：链接准确率 = 该链接的有没有连对；重复节点率 = 该新建的有没有误建重复节点。

    ⚠ 后半句的读法（容易看反）：它说的是**标"该连"的样本系统却新建了节点**，
    后果正是图里多出一个重复节点。反向错误（标"该新建"却连到别人）会把独立实体吞掉，
    课案没给它起名，这里叫 `false_link`，一并报出来。
    """

    def _metrics(self, decisions, labels):
        return quality.link_decision_metrics(decisions, labels)

    def test_all_correct(self):
        labels = [
            {"name": "老张", "entity_type": "人物", "matched": "张三"},
            {"name": "差旅费", "entity_type": "费用类型", "matched": None},
        ]
        result = self._metrics(labels, labels)
        assert result["resolved"] == 2
        assert result["link_accuracy"] == pytest.approx(1.0)
        assert result["duplicate_rate"] == pytest.approx(0.0)
        assert result["false_link"] == [] and result["false_create"] == []

    def test_false_create_is_counted_as_duplicate(self):
        # 标注：该连到"张三"；系统：新建 ⇒ 图里多一个重复节点
        labels = [{"name": "老张", "entity_type": "人物", "matched": "张三"}]
        decisions = [{"name": "老张", "entity_type": "人物", "matched": None}]
        result = self._metrics(decisions, labels)
        assert result["link_accuracy"] == pytest.approx(0.0)
        assert result["duplicate_rate"] == pytest.approx(1.0)
        assert result["false_create"] == [
            {"name": "老张", "entity_type": "人物", "expected": "张三", "decided": None}
        ]

    def test_false_link_swallows_an_entity(self):
        # 标注：该新建；系统：连到已有节点 ⇒ 独立实体被吞
        labels = [{"name": "差旅费", "entity_type": "费用类型", "matched": None}]
        decisions = [{"name": "差旅费", "entity_type": "费用类型", "matched": "出差费用"}]
        result = self._metrics(decisions, labels)
        assert result["false_link"] == [
            {"name": "差旅费", "entity_type": "费用类型", "expected": None, "decided": "出差费用"}
        ]
        # 标"该新建"的样本不进链接准确率的分母（分母是"标该连"的样本）
        assert result["link_accuracy"] is None

    def test_target_mismatch_is_not_counted_as_correct(self):
        labels = [{"name": "老张", "entity_type": "人物", "matched": "张三"}]
        decisions = [{"name": "老张", "entity_type": "人物", "matched": "张四"}]
        result = self._metrics(decisions, labels)
        assert result["link_accuracy"] == pytest.approx(0.0)
        assert result["target_mismatch"] == [
            {"name": "老张", "entity_type": "人物", "expected": "张三", "decided": "张四"}
        ]

    def test_pairing_is_by_normalized_name_and_type(self):
        # 标注写"老张"、系统记录" 老 张 "，且类型一致 ⇒ 能配上（否则会被误判成缺样本）
        labels = [{"name": "老张", "entity_type": "人物", "matched": "张三"}]
        decisions = [{"name": " 老 张 ", "entity_type": "人物", "matched": " 张三 "}]
        result = self._metrics(decisions, labels)
        assert result["resolved"] == 1
        assert result["link_accuracy"] == pytest.approx(1.0)

    def test_same_name_different_type_are_different_samples(self):
        """同名不同类型是两回事（builder.match_entity 也按同类型才比较）。"""
        labels = [
            {"name": "张三", "entity_type": "人物", "matched": "张三"},
            {"name": "张三", "entity_type": "公司", "matched": None},
        ]
        decisions = [
            {"name": "张三", "entity_type": "人物", "matched": "张三"},
            {"name": "张三", "entity_type": "公司", "matched": "张三分公司"},
        ]
        result = self._metrics(decisions, labels)
        assert result["resolved"] == 2
        assert result["link_accuracy"] == pytest.approx(1.0)
        assert len(result["false_link"]) == 1

    def test_samples_without_counterpart_are_reported_not_dropped(self):
        labels = [{"name": "有标注的", "entity_type": "人物", "matched": None}]
        decisions = [
            {"name": "有标注的", "entity_type": "人物", "matched": None},
            {"name": "没标注的", "entity_type": "人物", "matched": "谁"},
        ]
        result = self._metrics(decisions, labels)
        assert result["resolved"] == 1
        assert result["unresolved"] == [
            {"name": "没标注的", "entity_type": "人物", "side": "decision"}
        ]

    def test_empty_labels_gives_none_not_zero(self):
        """没有"该连"的样本时链接准确率是 None 而不是 0 —— 0 会被读成"全错"。"""
        result = self._metrics([], [])
        assert result["link_accuracy"] is None
        assert result["duplicate_rate"] is None
        assert result["resolved"] == 0


class TestCommunityCoverage:
    """社区覆盖：`0` 是合法社区编号，不能被当成"未分配"。

    这条是**真机读数对不上才发现的**：体检脚本里写成 `(node.get("community_id") or -1) < 0`，
    51 个实体的图报出"7 个未分配"，而库里的真值是 0（`WHERE community_id IS NULL OR < 0`）。
    """

    def test_zero_is_a_valid_community(self):
        nodes = [{"name": "A", "community_id": 0}, {"name": "B", "community_id": 3}]
        result = quality.community_coverage(nodes)
        assert result["unassigned_count"] == 0
        assert result["unassigned"] == []

    def test_negative_and_missing_are_unassigned(self):
        nodes = [
            {"name": "A", "community_id": -1},
            {"name": "B", "community_id": None},
            {"name": "C"},
            {"name": "D", "community_id": 0},
        ]
        result = quality.community_coverage(nodes)
        assert result["unassigned_count"] == 3
        assert result["unassigned"] == ["A", "B", "C"]
        assert result["rate"] == pytest.approx(0.75)

    def test_empty(self):
        result = quality.community_coverage([])
        assert result == {"total": 0, "unassigned_count": 0, "unassigned": [], "rate": 0.0}


class TestCompareWithPrevious:
    """课案：孤立节点率突增说明抽取或匹配环节退化 ⇒ 报告要能自己喊出来。"""

    def _report(self, rate, total=100):
        return {"metrics": {"isolated": {"rate": rate, "isolated_count": int(rate * total), "total": total}}}

    def test_alert_on_surge(self):
        result = quality.compare_with_previous(self._report(0.0), self._report(0.2))
        assert result["alerts"], "孤点率从 0 涨到 20% 必须报警"
        assert "孤立节点率" in result["alerts"][0]

    def test_no_alert_on_small_fluctuation(self):
        # 51 个节点里多 1 个孤点 = +2%，属于正常波动（阈值 5 个百分点）
        result = quality.compare_with_previous(self._report(0.02), self._report(0.04))
        assert result["alerts"] == []

    def test_first_run_has_no_previous(self):
        result = quality.compare_with_previous(None, self._report(0.0))
        assert result["alerts"] == []
        assert result["deltas"] == {}

    def test_reports_duplicate_surge_too(self):
        prev = {"metrics": {"isolated": {"rate": 0.0}, "duplicate": {"rate": 0.0}}}
        cur = {"metrics": {"isolated": {"rate": 0.0}, "duplicate": {"rate": 0.1}}}
        result = quality.compare_with_previous(prev, cur)
        assert any("重复节点率" in a for a in result["alerts"])

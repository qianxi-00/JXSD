"""评估集构造器测试:对照优化篇课案 script/build_eval_set.py。

课案要求(「RAG 评估 · 数据源生成与上传脚本」):
- 从真实票据字段**确定性**生成样本,标准答案由字段拼装(金额求和、日期格式化),
  **不允许 LLM 生成标准答案**;
- 按 (person, ticket_type) 分组,覆盖四类场景:金额汇总 / 日期人员筛选 /
  缺失证据拒答 / 多票据归因;
- 幂等可重跑(同样输入产出同样结果)。

与课案的一处拆分:课案的 OUT_OF_SCOPE_QUESTIONS 是"天气/小说"这类通用问题,
它们的正确行为其实是**直接回答**(不该检索);而"公司食堂装修花了多少钱"这类
票据域内但库里没有证据的问题才该走"无证据拒答"。因此拆成
build_refusal(期望检索后拒答)与 build_direct_answer(期望直接回答)两个构造器。
"""


from evaluation.eval_builders import (
    build_aggregation,
    build_direct,
    build_direct_answer,
    build_multi,
    build_refusal,
    date_str,
    fen_to_yuan,
    group_by_person_type,
)
from evaluation.eval_set import validate_sample


def ticket(row_id, person, ticket_type, **kw):
    base = {
        "id": row_id,
        "person": person,
        "ticket_type": ticket_type,
        "ticket_no": f"NO-{row_id}",
        "date_int": 20250305,
        "amount_fen": 43600,
        "route": "北京-上海",
        "counterparty": "中国铁路",
    }
    base.update(kw)
    return base


TICKETS = [
    ticket("t1", "张三", "train", amount_fen=43600, date_int=20250305, route="北京-上海"),
    ticket("t2", "张三", "train", amount_fen=40000, date_int=20250321, route="上海-北京"),
    ticket("t3", "李四", "flight", amount_fen=128000, date_int=20250401, route="A-JICHANG-B-JICHANG"),
    ticket("t4", "王五", "invoice", amount_fen=150000, date_int=20250402, route=""),
    ticket("t5", "赵六", "train", amount_fen=None, date_int=None, route=None, ticket_no=None),
    ticket("t6", "", "train", amount_fen=10000),  # person 为空,分组时应跳过
]


class TestHelpers:
    def test_fen_to_yuan(self):
        assert fen_to_yuan(43600) == "436.00"
        assert fen_to_yuan(3980291) == "39802.91"

    def test_date_str(self):
        assert date_str(20250305) == "2025-03-05"
        assert date_str(20251104) == "2025-11-04"

    def test_group_by_person_type_skips_empty_person(self):
        groups = group_by_person_type(TICKETS)
        assert ("张三", "train") in groups
        assert all(person for person, _ in groups)
        assert len(groups[("张三", "train")]) == 2

    def test_group_is_sorted_for_determinism(self):
        keys = list(group_by_person_type(TICKETS))
        assert keys == sorted(keys)


class TestBuildDirect:
    """日期/人员筛选:单人单票种的事实型问题"""

    def test_only_uses_single_ticket_groups(self):
        samples = build_direct(group_by_person_type(TICKETS), limit=10)
        assert samples, "应当能从单人单票种的分组里产出样本"
        for sample in samples:
            assert len(sample["expected_ticket_ids"]) == 1

    def test_questions_are_about_ticket_no_amount_date_or_route(self):
        samples = build_direct(group_by_person_type(TICKETS), limit=4)
        joined = " ".join(s["question"] for s in samples)
        assert any(word in joined for word in ("票号", "多少钱", "哪天的", "从哪到哪"))

    def test_ground_truth_is_built_from_fields(self):
        samples = build_direct(group_by_person_type(TICKETS), limit=10)
        by_id = {s["expected_ticket_ids"][0]: s for s in samples}
        if "t1" in by_id:
            answer = by_id["t1"]["ground_truth"]
            assert "436.00" in answer or "2025-03-05" in answer or "NO-t1" in answer or "北京-上海" in answer

    def test_skips_tickets_without_ticket_no(self):
        samples = build_direct(group_by_person_type(TICKETS), limit=10)
        assert all(s["expected_ticket_ids"] != ["t5"] for s in samples)

    def test_respects_limit(self):
        assert len(build_direct(group_by_person_type(TICKETS), limit=1)) == 1


class TestBuildMulti:
    """多票据归因:两人同票种金额对比"""

    def test_pairs_two_single_ticket_groups(self):
        samples = build_multi(group_by_person_type(TICKETS), limit=4)
        for sample in samples:
            assert len(sample["expected_ticket_ids"]) == 2
            assert "和" in sample["question"]

    def test_ground_truth_contains_both_amounts(self):
        samples = build_multi(group_by_person_type(TICKETS), limit=4)
        for sample in samples:
            assert sample["ground_truth"].count("元") == 2


class TestBuildAggregation:
    """金额汇总:单人单票种 2~4 张的总金额与张数"""

    def test_uses_multi_ticket_groups_and_sums_amounts(self):
        samples = build_aggregation(group_by_person_type(TICKETS), limit=6)
        assert samples
        sample = samples[0]
        assert sample["expected_ticket_ids"] == ["t1", "t2"]
        assert "2张" in sample["ground_truth"] or "2 张" in sample["ground_truth"]
        assert "836.00" in sample["ground_truth"]

    def test_skips_groups_with_missing_amount(self):
        samples = build_aggregation(group_by_person_type(TICKETS), limit=6)
        assert all("t5" not in s["expected_ticket_ids"] for s in samples)

    def test_task_type_is_structured_aggregation(self):
        samples = build_aggregation(group_by_person_type(TICKETS), limit=6)
        assert all(s["task_type"] == "structured_aggregation" for s in samples)


class TestBuildRefusalAndDirect:
    """缺失证据拒答 vs 通用问题直接回答"""

    def test_refusal_samples_have_no_expected_tickets(self):
        samples = build_refusal()
        assert samples
        for sample in samples:
            assert sample["expected_ticket_ids"] == []
            assert sample["expected_route"] == "search"
            assert "未找到" in sample["ground_truth"]

    def test_direct_answer_samples_expect_direct_route(self):
        samples = build_direct_answer()
        assert samples
        for sample in samples:
            assert sample["expected_route"] == "direct"
            assert sample["task_type"] == "faq"


class TestSamplesConformToContract:
    def test_all_builders_emit_valid_samples(self):
        groups = group_by_person_type(TICKETS)
        samples = (
            build_direct(groups, limit=18)
            + build_multi(groups, limit=4)
            + build_aggregation(groups, limit=6)
            + build_refusal()
            + build_direct_answer()
        )
        for sample in samples:
            assert validate_sample(sample) == sample

    def test_builders_are_deterministic(self):
        groups = group_by_person_type(TICKETS)
        assert build_direct(groups, limit=5) == build_direct(groups, limit=5)
        assert build_aggregation(groups, limit=5) == build_aggregation(groups, limit=5)

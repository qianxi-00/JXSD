"""召回过滤测试:对照基础篇课案 pipeline/filters.py 的抽取规则与 Milvus 表达式语义。

课案要求:
- extract_ticket_filters:从问题中轻量抽取 票据类型/年份/人员/路线/金额范围/票号
- build_milvus_filter:把条件转换为 Milvus 标量过滤表达式(and 连接,带转义)
- 金额单位统一为分;日期用 date_int;null 字段不满足条件(不等于 0)
"""

from datetime import date

from pipeline.filters import (
    build_milvus_filter,
    extract_ticket_filters,
    row_matches_filters,
)


class TestExtractTicketType:
    def test_invoice(self):
        assert extract_ticket_filters("金额超过1000元的发票")["ticket_type"] == "invoice"

    def test_train_keywords(self):
        for word in ("高铁", "火车", "动车", "车票"):
            assert extract_ticket_filters(f"查一下张三的{word}")["ticket_type"] == "train"

    def test_flight_keywords(self):
        for word in ("机票", "飞机票", "航班", "航空"):
            assert extract_ticket_filters(f"看看{word}的情况")["ticket_type"] == "flight"

    def test_no_type(self):
        assert "ticket_type" not in extract_ticket_filters("公司食堂装修花了多少钱")


class TestExtractYear:
    def test_explicit_year(self):
        f = extract_ticket_filters("赵军2024年的发票")
        assert f["date_start"] == 20240101
        assert f["date_end"] == 20241231

    def test_this_year_relative(self):
        f = extract_ticket_filters("张三今年的高铁票", today=date(2026, 1, 1))
        assert f["date_start"] == 20260101
        assert f["date_end"] == 20261231

    def test_last_year_relative(self):
        f = extract_ticket_filters("李四去年的火车票", today=date(2026, 1, 1))
        assert f["date_start"] == 20250101
        assert f["date_end"] == 20251231

    def test_no_year(self):
        assert "date_start" not in extract_ticket_filters("王霞购买的火车票金额和日期是多少?")


class TestExtractPerson:
    def test_person_before_year(self):
        assert extract_ticket_filters("赵军2024年的发票")["person"] == "赵军"

    def test_person_with_ticket_word(self):
        f = extract_ticket_filters("王霞购买的火车票")
        assert f["person"] == "王霞"

    def test_politeness_prefix_stripped(self):
        f = extract_ticket_filters("请帮我查一下王强2025年的火车票")
        assert f["person"] == "王强"

    def test_time_word_not_person(self):
        # "今年/去年" 等时间词出现在名词位置时不应被当成人名
        f = extract_ticket_filters("查一下今年的发票")
        assert "person" not in f

    def test_no_person(self):
        assert "person" not in extract_ticket_filters("金额超过1000元的发票")


class TestExtractRoute:
    def test_route_from_to(self):
        f = extract_ticket_filters("从北京到上海的高铁票")
        assert f["route_from"] == "北京"
        assert f["route_to"] == "上海"
        assert f["ticket_type"] == "train"

    def test_no_route(self):
        assert "route_from" not in extract_ticket_filters("赵军的发票多少钱")


class TestExtractAmount:
    def test_amount_min_over(self):
        f = extract_ticket_filters("金额超过1000元的发票")
        assert f["amount_min_fen"] == 100000
        assert f["amount_min_operator"] == ">"

    def test_amount_min_at_least(self):
        f = extract_ticket_filters("不低于500元的发票")
        assert f["amount_min_fen"] == 50000
        assert f["amount_min_operator"] == ">="

    def test_amount_max_less_than(self):
        f = extract_ticket_filters("少于200.5元的发票")
        assert f["amount_max_fen"] == 20050
        assert f["amount_max_operator"] == "<"

    def test_amount_max_at_most(self):
        f = extract_ticket_filters("至多300元的发票")
        assert f["amount_max_fen"] == 30000
        assert f["amount_max_operator"] == "<="


class TestExtractTicketNo:
    def test_ticket_no_with_label(self):
        assert extract_ticket_filters("票号INV20250109的金额")["ticket_no"] == "INV20250109"

    def test_ticket_no_with_colon(self):
        assert extract_ticket_filters("票号: INV-123 的票据")["ticket_no"] == "INV-123"

    def test_no_ticket_no(self):
        assert "ticket_no" not in extract_ticket_filters("张三的火车票多少钱")

    def test_bare_invoice_number_not_treated_as_ticket_no(self):
        # 安全性:无标签的 "发票1000元的" 不能被当成票号,否则会生成
        # ticket_no == "1000" 的假过滤条件,把召回直接清空
        assert "ticket_no" not in extract_ticket_filters("发票1000元的有哪些")


def test_extract_no_conditions_returns_empty():
    assert extract_ticket_filters("你好") == {}


class TestBuildMilvusFilter:
    def test_empty_filters(self):
        assert build_milvus_filter({}) == ""

    def test_single_type(self):
        assert build_milvus_filter({"ticket_type": "invoice"}) == 'ticket_type == "invoice"'

    def test_full_combination(self):
        expr = build_milvus_filter({
            "ticket_type": "invoice",
            "person": "张三",
            "ticket_no": "INV20250101",
            "date_start": 20250101,
            "date_end": 20251231,
            "amount_min_fen": 100000,
            "amount_min_operator": ">",
            "amount_max_fen": 500000,
            "amount_max_operator": "<=",
            "route_from": "北京",
            "route_to": "上海",
        })
        assert expr == (
            'ticket_type == "invoice" and person == "张三" and ticket_no == "INV20250101"'
            " and date_int >= 20250101 and date_int <= 20251231"
            " and amount_fen > 100000 and amount_fen <= 500000"
            ' and route like "北京%上海%"'
        )

    def test_default_operators(self):
        expr = build_milvus_filter({"amount_min_fen": 100, "amount_max_fen": 900})
        assert expr == "amount_fen >= 100 and amount_fen <= 900"

    def test_route_requires_both_ends(self):
        assert "route" not in build_milvus_filter({"route_from": "北京"})

    def test_quote_escaping(self):
        expr = build_milvus_filter({"person": '张"三\\'})
        assert expr == 'person == "张\\"三\\\\"'


def _row(**kw):
    base = {
        "id": "ticket_x",
        "ticket_type": "invoice",
        "person": "张三",
        "ticket_no": "INV20250101",
        "date_int": 20250601,
        "amount_fen": 200000,
        "route": "北京-上海",
    }
    base.update(kw)
    return base


class TestRowMatchesFilters:
    """BM25 语料行的内存过滤判定(与 build_milvus_filter 同语义)"""

    def test_all_match(self):
        assert row_matches_filters(_row(), {"ticket_type": "invoice", "person": "张三"})

    def test_type_mismatch(self):
        assert not row_matches_filters(_row(ticket_type="train"), {"ticket_type": "invoice"})

    def test_amount_range(self):
        assert row_matches_filters(
            _row(), {"amount_min_fen": 100000, "amount_min_operator": ">="}
        )
        assert not row_matches_filters(
            _row(amount_fen=50), {"amount_min_fen": 100000, "amount_min_operator": ">="}
        )

    def test_null_amount_fails_amount_condition(self):
        # 课案:null 字段不会自动满足金额条件
        assert not row_matches_filters(_row(amount_fen=None), {"amount_min_fen": 100000})

    def test_null_date_fails_date_condition(self):
        assert not row_matches_filters(_row(date_int=None), {"date_start": 20250101})

    def test_route_like_semantics(self):
        filters = {"route_from": "北京", "route_to": "上海"}
        assert row_matches_filters(_row(), filters)
        assert not row_matches_filters(_row(route="上海-北京"), filters)
        assert not row_matches_filters(_row(route=None), filters)

    def test_empty_filters_pass_all(self):
        assert row_matches_filters(_row(), {})


class TestTicketNumberExtraction:
    """票号抽取的字符集：含下划线的票号不能被截断。

    回归背景（2026-09-17 补注释时发现）：字符集原先只有 `[A-Za-z0-9-]`，
    `票号 ticket_001 的金额是多少` 会被抽成 `"ticket"`，于是过滤条件
    `ticket_no == "ticket"` 恒不命中 —— 等于把召回清空（靠回退兜住）。
    课案的提示词例句恰好就是 `票据ticket_001的金额是多少？`，不是假想输入。
    """

    def test_underscore_is_part_of_ticket_no(self):
        filters = extract_ticket_filters("票号 ticket_001 的金额是多少")
        assert filters.get("ticket_no") == "ticket_001"

    def test_multiple_underscores(self):
        assert extract_ticket_filters("票据号码 INV_2025_001 是多少钱").get("ticket_no") == "INV_2025_001"

    def test_plain_numbers_still_work(self):
        assert extract_ticket_filters("电子客票号 8762369777769").get("ticket_no") == "8762369777769"

    def test_underscore_does_not_swallow_sentence(self):
        """下划线进字符集后，仍然只吃到空白/标点为止，不会把整句都吃进去。"""
        assert extract_ticket_filters("票号 T2023_001，金额是多少").get("ticket_no") == "T2023_001"

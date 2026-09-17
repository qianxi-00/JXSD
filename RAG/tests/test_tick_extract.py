"""票据字段抽取测试（基础篇课案「统一票据字段 / 三类票据字段抽取」）。

`data_process/tick_extract.py` 是全项目字段抽取的唯一实现，之前**既没有自检也没有测试**。
这里用真实 OCR 片段钉住关键口径：12 字段、null 不落 0、金额换算、不同 OCR 引擎的版面差异。
"""

from data_process import tick_extract as te


class TestMoneyAndId:
    def test_yuan_to_fen_handles_thousands_separator(self):
        assert te.yuan_to_fen("39,802.91") == 3980291

    def test_yuan_to_fen_handles_one_decimal(self):
        assert te.yuan_to_fen("155.6") == 15560

    def test_ticket_id_is_stable_and_prefixed(self):
        first = te.ticket_id("data/train/001.png")
        assert first.startswith("ticket_") and len(first) == len("ticket_") + 24
        assert first == te.ticket_id("data/train/001.png")

    def test_ticket_id_differs_per_file(self):
        assert te.ticket_id("data/train/001.png") != te.ticket_id("data/train/002.png")


class TestCleanSemantic:
    def test_strips_html_tags_and_collapses_whitespace(self):
        html = "<td>项目名称</td><td>数量</td>\n\n  1   项  "
        cleaned = te.clean_semantic(html)
        assert "<td>" not in cleaned and "\n" not in cleaned
        assert cleaned == "项目名称 数量 1 项"


class TestFlightPersonAcrossOcrLayouts:
    """登机牌姓名：两种 OCR 引擎的版面不同，都要能抽到。"""

    # DeepSeek-OCR-2 版面：中文名紧跟在拼音名下一行
    DEEPSEEK = "姓名Name\nZHAOFEI\n赵飞\n自From 阿克苏机场 AKESUJICHANG\n至To 首都机场 SHOUDUJICHANG"

    # PaddleOCR-VL 版面：中间插了空行（实测 7 张全漏）
    PADDLE = (
        "姓名Name\n\nZHAOFEI\n\n赵飞\n\n自From 阿克苏机场 AKESUJICHANG\n\n至To 首都机场\n\nSHOUDUJICHANG"
    )

    # 姓名与拼音写在同一行
    INLINE = "姓名: 赵飞\n自From 阿克苏机场 AKESUJICHANG\n至To 首都机场 SHOUDUJICHANG"

    def test_deepseek_layout(self):
        assert te.extract_flight(self.DEEPSEEK)["person"] == "赵飞"

    def test_paddle_layout_with_blank_lines(self):
        assert te.extract_flight(self.PADDLE)["person"] == "赵飞"

    def test_inline_layout(self):
        assert te.extract_flight(self.INLINE)["person"] == "赵飞"

    def test_airport_name_is_not_taken_as_person(self):
        assert te.extract_flight("姓名Name\n\nZHAOFEI\n\n机场")["person"] is None

    def test_route_from_from_to_lines(self):
        assert te.extract_flight(self.PADDLE)["route"] == "AKESUJICHANG-SHOUDUJICHANG"


class TestFlightFields:
    def test_empty_text_yields_all_none(self):
        record = te.extract_flight("")
        assert set(record) == set(te.FIELDS)
        assert all(value is None for value in record.values())

    def test_etkt_number_and_null_date_amount(self):
        record = te.extract_flight("ETKT8762369777769/1\n姓名Name\nCA1276 Jan01 G")
        assert record["ticket_no"] == "8762369777769"
        # 登机牌只有 Jan01 没有年份与金额：按课案口径 null 不落 0
        assert record["date_int"] is None and record["amount_fen"] is None


class TestInvoiceFields:
    TEXT = (
        "发票编号：INV20250101\n开具日期：2025-11-04\n"
        "卖方\n腾讯科技有限公司\n客户\n武汉通信有限公司\n总金额 39,802.91 CNY"
    )

    def test_core_fields(self):
        record = te.extract_invoice(self.TEXT)
        assert record["ticket_no"] == "INV20250101"
        assert record["date_int"] == 20251104
        assert record["amount_fen"] == 3980291
        # 发票没有行程
        assert record["route"] == ""

    def test_company_roles(self):
        record = te.extract_invoice(self.TEXT)
        assert record["counterparty"] == "腾讯科技有限公司"
        assert record["person"] == "武汉通信有限公司"


class TestTrainFields:
    TEXT = (
        "T20170708704401 JM\n青岛站\n\nQingdao\n\n无 锡站\n\nWuxi\n"
        "2026年08月03日02:09开\n¥155.6元\n4503011970****6874 富东"
    )

    def test_core_fields(self):
        record = te.extract_train(self.TEXT)
        assert record["ticket_no"] == "T20170708704401"
        assert record["date_int"] == 20260803
        assert record["amount_fen"] == 15560
        assert record["person"] == "富东"

    def test_route_uses_english_station_names(self):
        assert te.extract_train(self.TEXT)["route"] == "Qingdao-Wuxi"

    def test_counterparty_is_null_for_train(self):
        assert te.extract_train(self.TEXT)["counterparty"] is None


class TestBuildRecord:
    def test_record_shape_matches_milvus_schema(self):
        item = {
            "source_file": "data/invoice/001_INV20250101.png",
            "ticket_type": "invoice",
            "ocr_text": TestInvoiceFields.TEXT,
        }
        record = te.build_record(item)
        assert record["id"] == te.ticket_id(item["source_file"])
        assert record["ticket_type"] == "invoice"
        assert record["semantic_text"]
        assert record["amount_fen"] == 3980291

    def test_unknown_type_yields_empty_fields(self):
        record = te.build_record({"source_file": "x.png", "ticket_type": "unknown", "ocr_text": "abc"})
        assert all(record[field] is None for field in te.FIELDS)

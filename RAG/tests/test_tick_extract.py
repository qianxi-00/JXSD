"""票据字段抽取测试（基础篇课案「统一票据字段 / 三类票据字段抽取」）。

`data_process/tick_extract.py` 是全项目字段抽取的唯一实现，之前**既没有自检也没有测试**。
这里用真实 OCR 片段钉住关键口径：12 字段、null 不落 0、金额换算、不同 OCR 引擎的版面差异；
末尾两组是**写库幂等性**的闸门（upsert 而不是 insert，见 `TestWriteIsIdempotent`）。
"""

import ast
from pathlib import Path

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

    def test_airline_from_flight_no_iata_code(self):
        """票面不写航司全称时，用航班号的两字码反查（真机 83/100 张票**只有**这个信号）。

        实测缺口票里的码分布：CZ 15 / CA 15 / HU 7 / SC 7 / MF 6 / MU 5 / JD 5 / ZH 1。
        """
        for text, expected in [
            ("航班号Flight/日期Date/舱位Class\nZH9146 Jan01 G", "深圳航空"),
            ("ETKT8762369777769/1\nMU 5678 Jan01 G", "中国东方航空"),  # 码与数字之间被 OCR 插了空格
            ("CZ3456 Jan01 G", "中国南方航空"),
        ]:
            assert te.extract_flight(text)["counterparty"] == expected, text

    def test_ocr_zero_for_letter_o_is_tolerated(self):
        """OCR 会把字母 O 认成数字 0（实测遇到 `H0`，真码 `HO`）⇒ 数字→字母等价替换后再查一次。"""
        assert te.extract_flight("H0 1234 Jan01 G")["counterparty"] == "吉祥航空"

    def test_unknown_iata_code_stays_none(self):
        """查不到的码**不猜**：错填承运方会静默进 SQL 过滤与分组统计。"""
        assert te.extract_flight("ZZ1234 Jan01 G")["counterparty"] is None
        # 长数字串不能被当成"两字码 + 3~4 位数字"（ETKT8762… 里的 8762 不是航班号）
        assert te.extract_flight("ETKT8762369777769/1")["counterparty"] is None

    def test_chinese_airline_names_are_canonicalized(self):
        """同一家航司的多种写法必须落到同一个值，否则按 counterparty 分组会分裂成几条。"""
        for text in ["中国国际航空公司", "中国国际航空", "国航", "AIR CHINA"]:
            assert te.extract_flight(text)["counterparty"] == "中国国际航空", text
        assert te.extract_flight("海南航空公司")["counterparty"] == "海南航空"

    def test_chinese_name_wins_over_flight_no(self):
        # 票面同时写了中文全称与航班号时，以更明确的中文名为准
        record = te.extract_flight("中国南方航空公司\nCZ3456 Jan01 G")
        assert record["counterparty"] == "中国南方航空"

    def test_cold_airline_keeps_its_name_without_company_suffix(self):
        # 别名表里没有的中文航司：如实保留（去掉"公司"后缀），不能返回空、也不能硬塞成别家
        assert te.extract_flight("幸福航空公司\nXX1234")["counterparty"] == "幸福航空"


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

    def test_invoice_number_label_variants(self):
        """票号标签两种写法都要认。

        ⚠ 本仓 100 张发票**全部**写"发票编号"，所以"发票号码"这一支只有这条单测覆盖、
        没有真实样本（README 台账里也是这么写的）。加它是为了换一批真发票时不会整列抽空。
        """
        for label in ("发票编号", "发票号码"):
            text = f"{label}：INV20250101\n开具日期：2025-11-04\n总金额 39,802.91 CNY"
            assert te.extract_invoice(text)["ticket_no"] == "INV20250101", label

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


class TestBlankSemanticTextCount:
    """空 `semantic_text` 的行数必须能被数出来（B 表"18 条 OCR 失败记录"那条欠账的落点）。

    这些行的 `vec` 恒为 NULL、BM25 也切不出词 ⇒ 语义/关键词两路**永远召不回**，
    却照样占着 `count(*)` 的分母（"300 条票据" vs "实际能召回 282 条"）。
    所以导入收尾要同时报两个数，而不是只报一个。
    """

    def test_counts_rows_with_empty_semantic_text(self):
        class FakeClient:
            def __init__(self):
                self.calls = []

            def query(self, **kwargs):
                self.calls.append(kwargs)
                return [{"count(*)": 18}]

        client = FakeClient()
        assert te.blank_semantic_text_count(client, "tick") == 18
        # 过滤表达式与输出字段是契约的一部分：写成别的过滤条件会静默数错人
        assert client.calls[0]["filter"] == 'semantic_text == ""'
        assert client.calls[0]["output_fields"] == ["count(*)"]
        assert client.calls[0]["collection_name"] == "tick"

    def test_empty_result_is_zero_not_crash(self):
        class EmptyClient:
            def query(self, **kwargs):
                return []

        assert te.blank_semantic_text_count(EmptyClient(), "tick") == 0

    def test_string_count_is_coerced(self):
        """Milvus 的聚合结果有时是字符串（不同版本/网关行为不一致），要能转成 int。"""
        class StrClient:
            def query(self, **kwargs):
                return [{"count(*)": "7"}]

        assert te.blank_semantic_text_count(StrClient(), "tick") == 7


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


class TestCarryOverVectors:
    """写库改成 upsert 之后，必须先把旧向量读回来 —— 否则整实体覆盖会把向量抹成 NULL。

    实测依据（一次性集合上的探针）：upsert 时带 `vec=None` 或干脆不带该字段，
    **两种写法都会把已有向量抹掉**；后果是结构化查询查得到、语义检索查不到。
    """

    def test_fills_missing_vector_from_existing(self):
        batch = [{"id": "a", "vec": None}, {"id": "b", "vec": None}]
        filled = te._merge_existing_vectors(batch, {"a": [1.0, 0.0], "b": None})
        assert filled == 1
        assert batch[0]["vec"] == [1.0, 0.0]
        assert batch[1]["vec"] is None  # 库里本来也没有，保持 None

    def test_does_not_overwrite_fresh_vector(self):
        """本轮刚算出的向量（embed_tickets 写的）不该被库里的旧值顶掉。"""
        batch = [{"id": "a", "vec": [9.0, 9.0]}]
        filled = te._merge_existing_vectors(batch, {"a": [1.0, 0.0]})
        assert filled == 0
        assert batch[0]["vec"] == [9.0, 9.0]

    def test_empty_existing_vector_is_not_used(self):
        assert te._merge_existing_vectors([{"id": "a", "vec": None}], {"a": []}) == 0

    def test_has_vector_avoids_numpy_truthiness_trap(self):
        """`_has_vector` 不能用真值判断：numpy 多元素数组做真值判断会抛 ValueError。"""
        import numpy as np

        assert te._has_vector(None) is False
        assert te._has_vector([]) is False
        assert te._has_vector([0.0, 0.0]) is True
        assert te._has_vector(np.zeros(4)) is True  # 所有元素为 0 也是"有向量"

    def test_carry_over_reports_when_query_fails(self, capsys):
        """读不回旧向量时降级（不能让整轮导入失败），但必须告警说明后果。"""

        class BrokenClient:
            def query(self, **_kw):
                raise RuntimeError("Milvus 暂时不可用")

        batch = [{"id": "a", "vec": None}]
        assert te._carry_over_vectors(BrokenClient(), "tick", batch) == 0
        out = capsys.readouterr().out
        assert "WARN" in out and "embed_tickets" in out
        assert batch[0]["vec"] is None


class TestWriteIsIdempotent:
    """AST 闸门：写库必须是 upsert（旧实现 insert 会让重跑一次语料翻倍）。

    为什么用 AST 而不是 `Select-String 'insert('`：本文件里到处都是 `sys.path.insert(...)`，
    字符串搜索必然误报；只看 Call 节点的属性名才准（这是实测过的教训）。
    """

    def _write_calls(self) -> list[str]:
        src = Path(te.__file__).read_text(encoding="utf-8")
        names = []
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                # 只看"对某个 client 调用的方法"，且第一个关键字是 collection_name
                if node.func.attr in {"insert", "upsert"}:
                    if any(kw.arg == "collection_name" for kw in node.keywords):
                        names.append(node.func.attr)
        return names

    def test_uses_upsert_not_insert(self):
        assert self._write_calls() == ["upsert"], (
            "写库调用必须恰好是 client.upsert(collection_name=...)："
            "insert 在同一主键上会多出一行（实测 row_count 从 1 变 2）"
        )


"""OCR Markdown → 入库 JSON 的链路测试（基础篇课案「OCR 清洗 / 结果处理」）。

这一环之前是断的：`paddle_ocr.py` 只写 Markdown、`tick_extract.py` 只读 JSON，
中间没有脚本，入库数据只能来自课案服务器上另一次 OCR 运行。
"""

import json

from data_process import build_ocr_json as b


class TestStripMarkdown:
    def test_removes_image_div_blocks(self):
        text = '姓名Name\n\n<div style="text-align: center;"><img src="imgs/a.jpg" alt="Image" /></div>\n\n赵飞'
        assert b.strip_markdown(text) == "姓名Name\n\n赵飞"

    def test_keeps_heading_text_but_drops_hashes(self):
        assert b.strip_markdown("## 发票\n\nCNY · VAT") == "发票\n\nCNY · VAT"

    def test_collapses_whitespace_and_blank_runs(self):
        assert b.strip_markdown("A   B\tC\n\n\n\nD") == "A B C\n\nD"

    def test_keeps_cjk_spacing_inside_line(self):
        # 票面 "无 锡站" 的空格是 OCR 出的真实内容,行内单空格不能压掉
        assert b.strip_markdown("无 锡站") == "无 锡站"

    def test_empty_input(self):
        assert b.strip_markdown("") == ""


class TestDedupe:
    def test_consecutive_removes_only_adjacent_duplicates(self):
        text = "税率 6%\n税率 6%\n商品 A\n税率 6%"
        cleaned, removed = b.dedupe_consecutive(text)
        assert cleaned == "税率 6%\n商品 A\n税率 6%"
        assert removed == 1

    def test_consecutive_keeps_blank_line_repeats(self):
        text = "A\n\n\nB"
        cleaned, removed = b.dedupe_consecutive(text)
        assert cleaned == "A\n\n\nB" and removed == 0

    def test_all_mode_is_stricter(self):
        text = "税率 6%\n商品 A\n税率 6%"
        cleaned, removed = b.dedupe_all(text)
        assert cleaned == "税率 6%\n商品 A" and removed == 1

    def test_none_mode_keeps_everything(self):
        text = "A\nA"
        cleaned, removed = b.clean_text(text, "none")
        assert cleaned == "A\nA" and removed == 0


class TestBuildRecords:
    def test_builds_expected_schema(self, tmp_path):
        (tmp_path / "train").mkdir()
        (tmp_path / "train" / "001_T2023.md").write_text("# 票\n青岛站", encoding="utf-8")
        records = b.build_records(tmp_path, ("train",))
        assert len(records) == 1
        record = records[0]
        assert record["source_file"] == "data/train/001_T2023.png"
        assert record["ticket_type"] == "train"
        assert record["ocr_status"] == "success"
        assert record["ocr_engine"] == "paddleocr-cloud"

    def test_missing_category_is_skipped(self, tmp_path):
        assert b.build_records(tmp_path, ("flight",)) == []

    def test_blank_markdown_marked_failed(self, tmp_path):
        (tmp_path / "flight").mkdir()
        (tmp_path / "flight" / "ticket-001.md").write_text("   \n\n", encoding="utf-8")
        assert b.build_records(tmp_path, ("flight",))[0]["ocr_status"] == "failed"


class TestEndToEnd:
    def test_written_json_is_readable_by_tick_extract_schema(self, tmp_path, monkeypatch):
        """产物必须能被 tick_extract 直接消费:字段名与课案 ocr_results.json 一致。"""
        out = tmp_path / "output" / "train"
        out.mkdir(parents=True)
        (out / "001_T2023.md").write_text("青岛站\n\n¥155.6元", encoding="utf-8")
        clean = tmp_path / "ocr_results_clean.json"

        from data_process import tick_extract

        records = b.build_records(tmp_path / "output", ("train",))
        for record in records:
            record["ocr_text"], _ = b.clean_text(record["ocr_text"], "consecutive")
        clean.write_text(
            json.dumps({"ocr_engine": "paddleocr-cloud", "results": records, "cleaning": {"dedupe": "consecutive"}},
                       ensure_ascii=False),
            encoding="utf-8",
        )

        payload = json.loads(clean.read_text(encoding="utf-8"))
        item = payload["results"][0]
        built = tick_extract.build_record(item)
        assert built["id"].startswith("ticket_")
        assert built["amount_fen"] == 15560
        assert "source_file" in built

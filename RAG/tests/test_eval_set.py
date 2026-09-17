"""评估集契约测试:对照基础篇课案「系统评估 · 第一步/分层构造」。

课案要求每条样本包含:
question / expected_route / expected_filter / expected_ticket_ids / ground_truth,
并按链路分层:FAQ 快速返回集、检索召回集、结构化统计集、生成答案集。
"""

import json

import pytest

from evaluation.eval_set import (
    TASK_TYPES,
    load_jsonl,
    save_jsonl,
    validate_sample,
)


def minimal_sample(**kw) -> dict:
    base = {"question": "张三今年高铁票一共花了多少钱?", "task_type": "generation"}
    base.update(kw)
    return base


class TestValidateSample:
    def test_minimal_sample_passes(self):
        sample = validate_sample(minimal_sample())
        assert sample["question"].startswith("张三")
        assert sample["task_type"] == "generation"

    def test_requires_question(self):
        with pytest.raises(ValueError, match="question"):
            validate_sample({"task_type": "generation"})

    def test_rejects_blank_question(self):
        with pytest.raises(ValueError, match="question"):
            validate_sample(minimal_sample(question="   "))

    def test_requires_known_task_type(self):
        with pytest.raises(ValueError, match="task_type"):
            validate_sample(minimal_sample(task_type="unknown_type"))

    def test_all_course_task_types_allowed(self):
        assert set(TASK_TYPES) == {"faq", "retrieval", "structured_aggregation", "generation"}
        for task_type in TASK_TYPES:
            assert validate_sample(minimal_sample(task_type=task_type))["task_type"] == task_type

    def test_expected_route_must_be_search_or_direct(self):
        assert validate_sample(minimal_sample(expected_route="search"))["expected_route"] == "search"
        assert validate_sample(minimal_sample(expected_route="direct"))["expected_route"] == "direct"
        with pytest.raises(ValueError, match="expected_route"):
            validate_sample(minimal_sample(expected_route="rag"))

    def test_expected_ticket_ids_must_be_string_list(self):
        assert validate_sample(minimal_sample(expected_ticket_ids=["ticket_a"]))[
            "expected_ticket_ids"
        ] == ["ticket_a"]
        with pytest.raises(ValueError, match="expected_ticket_ids"):
            validate_sample(minimal_sample(expected_ticket_ids="ticket_a"))

    def test_expected_filter_must_be_dict(self):
        assert validate_sample(minimal_sample(expected_filter={"person": "张三"}))[
            "expected_filter"
        ] == {"person": "张三"}
        with pytest.raises(ValueError, match="expected_filter"):
            validate_sample(minimal_sample(expected_filter=["person"]))

    def test_need_citation_defaults_false(self):
        assert validate_sample(minimal_sample())["need_citation"] is False

    def test_optional_fields_default_empty(self):
        sample = validate_sample(minimal_sample())
        assert sample["expected_ticket_ids"] == []
        assert sample["expected_filter"] == {}
        assert sample["ground_truth"] == ""
        assert sample["expected_route"] is None


class TestJsonlIO:
    def test_roundtrip(self, tmp_path):
        path = tmp_path / "eval_set.jsonl"
        samples = [
            validate_sample(minimal_sample(ground_truth="共 1 张,合计 436.00 元")),
            validate_sample(minimal_sample(question="1+1 等于几?", task_type="faq")),
        ]
        save_jsonl(samples, path)
        loaded = load_jsonl(path)
        assert loaded == samples

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "eval_set.jsonl"
        path.write_text(
            json.dumps(minimal_sample(), ensure_ascii=False) + "\n\n   \n",
            encoding="utf-8",
        )
        assert len(load_jsonl(path)) == 1

    def test_reports_line_number_on_invalid_sample(self, tmp_path):
        path = tmp_path / "eval_set.jsonl"
        path.write_text(
            json.dumps(minimal_sample(), ensure_ascii=False)
            + "\n"
            + json.dumps({"task_type": "generation"}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="第 2 行"):
            load_jsonl(path)

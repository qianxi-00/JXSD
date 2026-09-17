"""大模型评分解析测试:对照基础篇课案「系统评估 · 第四步」。

课案的 EVALUATION_PROMPT 要求模型输出:
    Feedback: {{针对标准的反馈}} [RESULT] {{1 到 5 的整数}}
这里锁住解析行为:必须能取出 [RESULT] 后的 1~5 整数,取不到或越界返回 None。
"""


from evaluation.llm_judge import parse_result_score


class TestParseResultScore:
    def test_course_format(self):
        text = "Feedback: 回答与参考答案一致,金额与张数都正确 [RESULT] 5"
        assert parse_result_score(text) == 5

    def test_lowercase_and_spacing(self):
        assert parse_result_score("feedback: ok\n[result]   3") == 3

    def test_full_width_brackets(self):
        assert parse_result_score("Feedback: 部分正确【RESULT】4") == 4

    def test_score_with_suffix(self):
        assert parse_result_score("[RESULT] 2分") == 2

    def test_missing_marker_returns_none(self):
        assert parse_result_score("Feedback: 我觉得还行,给 4 分") is None

    def test_out_of_range_returns_none(self):
        assert parse_result_score("[RESULT] 6") is None
        assert parse_result_score("[RESULT] 0") is None

    def test_empty_returns_none(self):
        assert parse_result_score("") is None

    def test_prefers_first_marker(self):
        assert parse_result_score("[RESULT] 4 ... [RESULT] 5") == 4

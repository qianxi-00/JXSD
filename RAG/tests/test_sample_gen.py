"""质检解析测试:对照基础篇课案「样本生成 / 质量审核提示词」。

课案提示词要求输出 "回答::: 评价:... 总评分:N" 文本,但课案代码用 json.loads 解析,
两者对不上;这里要求解析器同时兼容文本与 JSON 两种形态。
"""


from evaluation.sample_gen import MIN_SCORE, parse_critique, passes_quality

TEXT_FORM = """回答:::
评价：上下文里有完整金额与日期，可以明确回答
总评分：5"""

JSON_FORM = '{"评价": "上下文齐全", "总评分": 5}'


class TestParseCritique:
    def test_text_form(self):
        parsed = parse_critique(TEXT_FORM)
        assert parsed["总评分"] == 5
        assert "完整金额" in parsed["评价"]

    def test_json_form(self):
        parsed = parse_critique(JSON_FORM)
        assert parsed == {"总评分": 5, "评价": "上下文齐全"}

    def test_json_wrapped_in_code_fence(self):
        parsed = parse_critique(f"```json\n{JSON_FORM}\n```")
        assert parsed["总评分"] == 5

    def test_full_width_colon(self):
        assert parse_critique("总评分：3")["总评分"] == 3

    def test_bold_markdown_score(self):
        # 模型常把评分写成 **5**
        assert parse_critique("总评分：**5**")["总评分"] == 5

    def test_score_with_suffix(self):
        assert parse_critique("总评分: 4分")["总评分"] == 4

    def test_unparsable_returns_none(self):
        assert parse_critique("我觉得还行") is None

    def test_empty_returns_none(self):
        assert parse_critique("") is None
        assert parse_critique("   ") is None

    def test_missing_evaluation_still_returns_score(self):
        parsed = parse_critique("总评分：5")
        assert parsed["总评分"] == 5
        assert parsed["评价"] == ""


class TestPassesQuality:
    def test_both_pass(self):
        assert passes_quality(parse_critique(TEXT_FORM), parse_critique(JSON_FORM))

    def test_second_below_threshold(self):
        assert not passes_quality(parse_critique(TEXT_FORM), parse_critique("总评分：4"))

    def test_unparsable_fails(self):
        assert not passes_quality(parse_critique(TEXT_FORM), None)

    def test_custom_min_score(self):
        assert passes_quality(parse_critique("总评分：3"), min_score=3)

    def test_default_min_score_is_five(self):
        assert MIN_SCORE == 5

    def test_no_critiques(self):
        assert not passes_quality()

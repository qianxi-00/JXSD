"""提示词契约测试:确认核心提示词与课案语义一致。

课案(基础篇「RAG 优化方法(query 改写)」)里四种检索策略的语义:
- 直接检索:原问题直接检索
- 假设问题检索(HyDE):先生成假设答案,再用假设答案检索
- 子查询检索:复杂问题拆成多个简单子问题,分别检索
- 回溯问题检索:把复杂查询**简化**为更基础、更易于检索的问题
"""

from core import prompts


class TestBacktrackSemantics:
    def test_backtrack_prompt_simplifies(self):
        text = prompts.BACKTRACK_PROMPT
        assert "简化" in text
        assert "更基础" in text or "更简单" in text

    def test_backtrack_prompt_does_not_ask_for_context_completion(self):
        # "补全隐含上下文"是另一类改写,与课案的"简化/回退"语义相反
        assert "补全" not in prompts.BACKTRACK_PROMPT

    def test_router_description_matches_prompt(self):
        # 路由提示词对 backtrack 的描述必须与改写提示词同一语义,
        # 否则会出现"路由按指代消解选中、改写却做简化"的不一致
        router = prompts.ROUTER_PROMPT
        assert "backtrack" in router
        assert "简化" in router or "更基础" in router


class TestRewriteStrategyCoverage:
    def test_subquery_prompt_caps_at_three(self):
        assert "3" in prompts.SUBQUERY_PROMPT

    def test_hyde_prompt_generates_hypothetical_text(self):
        text = prompts.HYDE_PROMPT
        assert "假设" in text

    def test_rewrite_labels_cover_all_strategies(self):
        assert set(prompts.REWRITE_LABELS) == {"hyde", "subquery", "backtrack", "direct"}

    def test_rewrite_prompts_registered_in_llm_module(self):
        from llm.chat import REWRITE_PROMPTS

        assert set(REWRITE_PROMPTS) == {"hyde", "subquery", "backtrack"}


class TestAnswerPromptContracts:
    def test_no_evidence_reply_states_no_reliable_basis(self):
        text = prompts.NO_EVIDENCE_REPLY
        assert "没有检索到" in text or "未找到" in text
        assert "无法作答" in text or "无法回答" in text

    def test_system_prompt_forbids_fabrication(self):
        assert "不要编造" in prompts.SYSTEM_PROMPT or "不能编造" in prompts.SYSTEM_PROMPT


class TestEvaluationPromptContracts:
    def test_evaluation_prompt_has_all_placeholders(self):
        text = prompts.EVALUATION_PROMPT
        for placeholder in ("{context}", "{question}", "{answer}", "{ground_truth}"):
            assert placeholder in text
        assert "[RESULT]" in text
        assert "得分 5" in text

    def test_evaluation_prompt_formats_without_error(self):
        # 提示词里有字面量花括号,format 能跑通说明转义正确
        rendered = prompts.EVALUATION_PROMPT.format(
            context="c", question="q", answer="a", ground_truth="g"
        )
        assert "Feedback" in rendered

    def test_critique_prompts_require_total_score(self):
        assert "总评分" in prompts.QUESTION_GROUNDEDNESS_CRITIQUE_PROMPT
        assert "{context}" in prompts.QUESTION_GROUNDEDNESS_CRITIQUE_PROMPT
        assert "总评分" in prompts.QUESTION_STANDALONE_CRITIQUE_PROMPT
        assert "{question}" in prompts.QUESTION_STANDALONE_CRITIQUE_PROMPT

    def test_critique_prompts_format(self):
        prompts.QUESTION_GROUNDEDNESS_CRITIQUE_PROMPT.format(context="c", question="q")
        prompts.QUESTION_STANDALONE_CRITIQUE_PROMPT.format(question="q")

    def test_qa_generation_prompt_asks_for_json(self):
        text = prompts.QA_GENERATION_PROMPT
        assert "{context}" in text
        assert "事实型问题" in text and "答案" in text
        assert "JSON" in text
        assert prompts.QA_GENERATION_PROMPT.format(context="票据内容")

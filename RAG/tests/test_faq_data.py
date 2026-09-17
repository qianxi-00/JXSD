"""预设问答(快速返回层)数据契约测试。

课案(基础篇「缓存与FAQ / 数据集的构造」):
- FAQ 只放高频、稳定、有标准答案的问题(如"发票里一般能查询哪些字段？");
- 每条至少要有问题与答案,且问题不重复;
- 明细统计类问题(依赖票据实时数据)不适合放进固定 FAQ。
"""

import json
from pathlib import Path

PRESET_QA_PATH = Path(__file__).resolve().parent.parent / "data" / "preset_qa.json"
FINANCE_FAQ_PATH = Path(__file__).resolve().parent.parent / "data" / "finance_faq.json"

# 课案 finance_faq.json 中的标准问法(共 5 条)
COURSE_FAQ_QUESTIONS = {
    "发票里一般能查询哪些字段？",
    "高铁票里一般能查询哪些字段？",
    "机票行程单里一般能查询哪些字段？",
    "车票里一般能查询哪些字段？",
    "票据金额统计类问题应该怎么问？",
}


def load_preset() -> list[dict]:
    return json.loads(PRESET_QA_PATH.read_text(encoding="utf-8"))


def test_preset_file_is_valid_json_list():
    pairs = load_preset()
    assert isinstance(pairs, list) and pairs


def test_every_pair_has_question_and_answer():
    for pair in load_preset():
        assert pair.get("question", "").strip(), pair
        assert pair.get("answer", "").strip(), pair


def test_questions_are_unique():
    questions = [p["question"] for p in load_preset()]
    assert len(questions) == len(set(questions))


def test_course_standard_faqs_present():
    """课案 finance_faq.json 的四条票据字段/问法标准问答必须在快速返回层里"""
    questions = {p["question"] for p in load_preset()}
    missing = COURSE_FAQ_QUESTIONS - questions
    assert not missing, f"缺失课案标准 FAQ: {missing}"


def test_finance_faq_data_file_matches_course():
    """保留课案原始 finance_faq.json(带 id/query/answer/source),便于对照维护"""
    items = json.loads(FINANCE_FAQ_PATH.read_text(encoding="utf-8"))
    assert {item["query"] for item in items} == COURSE_FAQ_QUESTIONS
    for item in items:
        assert item["id"] and item["answer"] and item["source"]


# ============================================================
# 分层契约:明细问答绝不能进相似度层
# ============================================================
# 起因(2026-09-17 复核实测):`preset_qa.json` 里 60 条"某人的某张票"进了相似度层,
# 「赵凡的登机牌座位号是多少?」与「赵飞的登机牌座位号是多少?」余弦相似度 0.9008 >
# 阈值 0.79 —— 会直接命中并把**别人的票**答出去,而预设命中 sources 为空、界面看不出。

def test_every_entry_declares_layer():
    for pair in load_preset():
        assert pair.get("layer") in {"faq", "detail"}, pair


def test_faq_layer_is_only_the_course_standard_questions():
    faq = [p["question"] for p in load_preset() if p["layer"] == "faq"]
    assert set(faq) == COURSE_FAQ_QUESTIONS


def test_detail_layer_never_enters_similarity_layer(monkeypatch):
    """`split_by_layer` 必须把明细挡在相似度层外,且缺 layer 字段时按明细处理(保守默认)。"""
    from core.cache import split_by_layer

    pairs = load_preset()
    faq, detail = split_by_layer(pairs)
    assert {p["question"] for p in faq} == COURSE_FAQ_QUESTIONS
    assert len(faq) + len(detail) == len(pairs)
    assert not (set(p["question"] for p in faq) & set(p["question"] for p in detail))

    # 没打 layer 的条目按明细处理:相似度层是危险的那层,必须显式打标才进
    untagged = [{"question": "某人某张票的金额是多少?", "answer": "x"}]
    assert split_by_layer(untagged) == ([], untagged)


def test_detail_questions_are_entity_specific():
    """明细层的问题都带具体主体(人名/公司名),这正是它们不能做相似度匹配的原因。"""
    detail = [p["question"] for p in load_preset() if p["layer"] == "detail"]
    assert detail
    for question in detail:
        assert "的" in question, question

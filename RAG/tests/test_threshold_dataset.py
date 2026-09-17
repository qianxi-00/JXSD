"""阈值标定数据准备测试(基础篇「评测集生成 → 阈值搜索」的 API 适配版)。

课案流程:query → 正例(同义问法) → 难负例(语义相近但不同的问题) → 相似度打分 → 搜最优阈值。
本项目全部走外部 API:
- 正例:LLM 生成同义问法(提示词见 script/build_threshold_dataset.py);
- 难负例:用 embedding 给"其他预设问题"打分,再按课案区间参数挑(select_hard_negatives)。

本文件覆盖两个纯函数:
- parse_paraphrases:解析 LLM 输出的同义问法列表(去掉编号/项目符号/重复/空行);
- build_threshold_items:把 query + 正例 + 难负例组装成标定输入(负例来自其他 query)。
"""

from evaluation.threshold import build_threshold_items, parse_paraphrases


class TestParseParaphrases:
    def test_strips_numbering_and_bullets(self):
        raw = "1. 张三的火车票多少钱\n2、张三高铁票花了多少\n- 张三的票花了多少钱\n* 张三坐火车花了多少"
        assert parse_paraphrases(raw) == [
            "张三的火车票多少钱",
            "张三高铁票花了多少",
            "张三的票花了多少钱",
            "张三坐火车花了多少",
        ]

    def test_drops_blank_lines_and_duplicates(self):
        raw = "张三的火车票多少钱\n\n张三的火车票多少钱\n   \n李四的发票金额"
        assert parse_paraphrases(raw) == ["张三的火车票多少钱", "李四的发票金额"]

    def test_respects_limit(self):
        raw = "\n".join(f"问法{i}" for i in range(10))
        assert len(parse_paraphrases(raw, limit=3)) == 3

    def test_drops_question_number_prefix_inside_text(self):
        # 模型有时会把"问题："这类前缀也带上
        assert parse_paraphrases("问题：张三的火车票多少钱") == ["张三的火车票多少钱"]

    def test_empty_input(self):
        assert parse_paraphrases("") == []
        assert parse_paraphrases("   \n  ") == []


class TestBuildThresholdItems:
    @staticmethod
    def fake_embed(text: str) -> list[float]:
        """固定向量:同前缀视为相近,便于离线断言负例挑选"""
        table = {
            "张三的火车票": [1.0, 0.0],
            "张三高铁票": [0.99, 0.1],
            "张三的机票": [0.9, 0.3],
            "李四的发票": [0.2, 0.98],
            "公司食堂装修花了多少钱": [0.0, 1.0],
        }
        return table[text]

    def test_negatives_come_from_other_queries(self):
        items = build_threshold_items(
            queries=["张三的火车票", "李四的发票", "公司食堂装修花了多少钱"],
            paraphrases={"张三的火车票": ["张三高铁票"], "李四的发票": [], "公司食堂装修花了多少钱": []},
            embed=self.fake_embed,
            num_negatives=2,
        )
        first = items[0]
        assert first["query"] == "张三的火车票"
        assert first["pos"] == ["张三高铁票"]
        # 负例必须来自别的 query,且不能是自身
        assert "张三的火车票" not in first["neg"]
        assert all(neg in {"李四的发票", "公司食堂装修花了多少钱"} for neg in first["neg"])

    def test_hard_negatives_prefer_semantically_close_queries(self):
        items = build_threshold_items(
            queries=["张三的火车票", "张三的机票", "公司食堂装修花了多少钱"],
            paraphrases={"张三的火车票": ["张三高铁票"]},
            embed=self.fake_embed,
            num_negatives=1,
            max_score=1.0,
            absolute_margin=0.05,
        )
        # "张三的机票"比"公司食堂..."更接近,应优先被选为难负例
        assert items[0]["neg"] == ["张三的机票"]

    def test_too_similar_candidate_is_rejected_by_margin(self):
        """absolute_margin 守卫:与正例太接近的候选不能当负例(很可能是同义改写)"""
        items = build_threshold_items(
            queries=["张三的火车票", "张三的机票", "公司食堂装修花了多少钱"],
            paraphrases={"张三的火车票": ["张三高铁票"]},
            embed=self.fake_embed,
            num_negatives=1,
            max_score=1.0,
            absolute_margin=0.5,
        )
        # "张三的机票"相似度 0.95,落在正例(1.0)下方 0.5 分以内 → 被排除,
        # 只能退而取语义很远的"公司食堂装修花了多少钱"
        assert items[0]["neg"] == ["公司食堂装修花了多少钱"]

    def test_item_shape_matches_calculate_metrics_input(self):
        items = build_threshold_items(
            queries=["张三的火车票", "李四的发票"],
            paraphrases={"张三的火车票": ["张三高铁票"]},
            embed=self.fake_embed,
        )
        assert set(items[0]) == {"query", "pos", "neg"}
        assert isinstance(items[0]["pos"], list) and isinstance(items[0]["neg"], list)

    def test_requires_at_least_two_queries_for_negatives(self):
        items = build_threshold_items(
            queries=["张三的火车票"],
            paraphrases={"张三的火车票": ["张三高铁票"]},
            embed=self.fake_embed,
        )
        assert items[0]["neg"] == []

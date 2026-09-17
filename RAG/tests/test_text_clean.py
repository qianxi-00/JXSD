"""思考标记清洗测试。

背景(实测):开启 enable_thinking 的推理模型偶尔会把思考残留写进正文,例如
`1+1等于几?` 的回答正文出现 `2</think>2`;而同一问题另一次调用又是干净的 `2`。
流式接口下标记还可能跨 chunk 断开,所以既要清洗完整文本,也要能过滤分片。

清洗规则:
1. 去掉完整的 `<think>...</think>` 块;
2. 如果残留 `</think>` 且其后还有非空内容,丢掉"到最后一个 `</think>` 为止"的前缀
   (那一段是思考内容);
3. 清掉残留的 `<think>` / `</think>` 标签本身。
"""

import pytest

from llm.text_clean import ThinkTagFilter, strip_thinking_markup


class TestStripThinkingMarkup:
    def test_complete_block_removed(self):
        assert strip_thinking_markup("<think>推理过程</think>答案是 2") == "答案是 2"

    def test_trailing_orphan_tag_keeps_text(self):
        assert strip_thinking_markup("答案是 2</think>") == "答案是 2"

    def test_leading_reasoning_before_closing_tag_is_dropped(self):
        assert strip_thinking_markup("推理过程\n</think>\n\n答案是 2") == "答案是 2"

    def test_repeated_text_around_tag_keeps_suffix(self):
        # 实测出现过的形态:正文里夹了一次思考残留
        assert strip_thinking_markup("2</think>2") == "2"

    def test_plain_text_untouched(self):
        assert strip_thinking_markup("共 2 张,合计 836.00 元") == "共 2 张,合计 836.00 元"

    def test_only_reasoning_yields_empty(self):
        assert strip_thinking_markup("<think>只有思考</think>") == ""

    def test_empty_input(self):
        assert strip_thinking_markup("") == ""

    def test_multiline_block(self):
        text = "<think>\n第一行\n第二行\n</think>\n\n最终答案"
        assert strip_thinking_markup(text) == "最终答案"

    def test_multiple_blocks(self):
        assert strip_thinking_markup("<think>a</think>答案<think>b</think>") == "答案"


class TestThinkTagFilter:
    """流式过滤:标记可能跨 chunk,必须缓冲到能判定为止"""

    def test_drops_leading_reasoning_chunk(self):
        filt = ThinkTagFilter()
        assert filt.feed("推理中</think>") == ""
        assert filt.feed("答案是 2") == "答案是 2"
        assert filt.flush() == ""

    def test_tag_split_across_chunks(self):
        filt = ThinkTagFilter()
        assert filt.feed("推理中</thi") == ""
        assert filt.feed("nk>答案") == "答案"
        assert filt.flush() == ""

    def test_plain_text_passes_through(self):
        filt = ThinkTagFilter()
        assert filt.feed("普通") == "普通"
        assert filt.feed("回答") == "回答"
        assert filt.flush() == ""

    def test_trailing_tag_removed_without_losing_text(self):
        filt = ThinkTagFilter()
        text = filt.feed("答案是 2</think>")
        text += filt.flush()
        assert text == "答案是 2"

    def test_buffer_flushed_when_no_tag_appears(self):
        # 迟迟没有标签时不能一直吞着内容:超过缓冲上限要原样吐出来
        filt = ThinkTagFilter(max_buffer=10)
        assert filt.feed("一二三四五六七八九十十一") == "一二三四五六七八九十十一"

    def test_complete_block_in_one_chunk(self):
        filt = ThinkTagFilter()
        assert filt.feed("<think>思考</think>答案") == "答案"
        assert filt.flush() == ""

    @pytest.mark.parametrize("chunks", [["答", "案", "是", " 2"]])
    def test_character_by_character_stream(self, chunks):
        filt = ThinkTagFilter()
        out = "".join(filt.feed(chunk) for chunk in chunks) + filt.flush()
        assert out == "答案是 2"

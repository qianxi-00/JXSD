"""LLM 流式适配测试:验证 reasoning/content 事件切分与思考标记清洗。

实测背景:开启 enable_thinking 后正文里偶尔带思考残留(`2</think>2`),
流式路径同样要清洗,且标签可能跨 chunk。
"""

from types import SimpleNamespace


from llm.chat import _iter_deltas


class FakeStream:
    """模拟 AsyncOpenAI 的流式响应:可异步迭代,且有 close()"""

    def __init__(self, chunks, fail_after: int | None = None):
        self._chunks = chunks
        self._fail_after = fail_after
        self.closed = False

    async def _gen(self):
        for index, chunk in enumerate(self._chunks):
            if self._fail_after is not None and index >= self._fail_after:
                raise RuntimeError("stream broken")
            yield chunk

    def __aiter__(self):
        return self._gen()

    async def close(self):
        self.closed = True


def chunk(content=None, reasoning=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content=content, reasoning_content=reasoning)
            )
        ]
    )


async def collect(stream):
    return [event async for event in _iter_deltas(stream)]


class TestIterDeltas:
    async def test_splits_reasoning_and_content(self):
        stream = FakeStream([chunk(reasoning="想一下"), chunk(content="答案是 2")])
        events = await collect(stream)
        assert events == [{"reasoning": "想一下"}, {"content": "答案是 2"}]

    async def test_ignores_chunks_without_choices(self):
        stream = FakeStream([SimpleNamespace(choices=[]), chunk(content="答案")])
        assert await collect(stream) == [{"content": "答案"}]

    async def test_closes_stream(self):
        stream = FakeStream([chunk(content="答案")])
        await collect(stream)
        assert stream.closed is True

    async def test_strips_think_markup_from_content(self):
        stream = FakeStream([chunk(content="推理</think>答案是 2")])
        assert await collect(stream) == [{"content": "答案是 2"}]

    async def test_strips_tag_split_across_chunks(self):
        stream = FakeStream([chunk(content="推理</thi"), chunk(content="nk>答案")])
        events = await collect(stream)
        assert "".join(e.get("content", "") for e in events) == "答案"

    async def test_trailing_tag_does_not_lose_answer(self):
        stream = FakeStream([chunk(content="答案是 2</think>")])
        events = await collect(stream)
        assert "".join(e.get("content", "") for e in events) == "答案是 2"

    async def test_flushes_buffer_at_stream_end(self):
        # 末端缓冲里是普通文本时,flush 必须把它交出来,不能吞掉
        stream = FakeStream([chunk(content="答")])
        events = await collect(stream)
        assert "".join(e.get("content", "") for e in events) == "答"

    async def test_empty_content_is_skipped(self):
        stream = FakeStream([chunk(content=""), chunk(content=None)])
        assert await collect(stream) == []

"""思考标记清洗:防止推理模型的思考残留混进正文。

实测背景:开启 `enable_thinking` 后,同一个问题多数调用是干净的,
但偶尔正文里会带思考残留(如 `1+1等于几?` 的回答正文出现 `2</think>2`)。
这类残留直接发给用户就是事故,所以在非流式与流式两条路径上都做清洗。

清洗规则:
1. 去掉完整的 `<think>...</think>` 块;
2. 若残留 `</think>` 且其后还有非空内容,说明标签之前那段是思考内容,整段丢掉;
3. 清掉残留的标签本身。

在 RAG 链路里的位置
==================
被 `llm/chat.py` 两条路径共用,是**最后一道出口防线**:
- 非流式(`generate_answer` / `generate_direct_answer`)→ 用 `strip_thinking_markup()`
  一次清洗整段正文;
- 流式(`_iter_deltas`)→ 用 `ThinkTagFilter` 逐 chunk 清洗,流结束时 `flush()` 收尾。

为什么流式要单独一个类、不能直接复用函数:标签**可能跨 chunk 断开**
(如 `"推理</thi"` + `"nk>答案"`),逐段调 `strip_thinking_markup` 会
第一次看不到完整标签、第二次又丢掉了判定所需的上下文,结果就是把 `</think>` 原样吐给前端。
`ThinkTagFilter` 用"缓冲到能判定再吐"解决这个问题,代价是**某些 chunk 的返回是空串**。

对应课案:`RAG 基础篇` 生成阶段(推理模型输出后处理)一节。
"""

from __future__ import annotations

import re

# 完整思考块:`<think ...>...</think>`。
# - `\b` 保证不会误伤 `<thinking>` 这类别的标签(`<think` 后面紧跟 `i` 不是词边界,不匹配);
# - `[^>]*` 容忍带属性的写法(如 `<think style="...">`);
# - `.*?` 非贪婪 + `re.S` 让 `.` 跨行,所以多行的思考块能被整块吃掉;
# - `re.I` 是因为模型偶尔输出大写标签。
_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.S | re.I)
# 孤立的闭合标签(前面没有配对的开始标签,或开始标签已被上一步清掉)。
_CLOSING_TAG_RE = re.compile(r"</think\s*>", re.I)
# 任意 think 标签(开或闭),用于最后一步"兜底铲掉残留标签"。
_ANY_TAG_RE = re.compile(r"</?think\b[^>]*>", re.I)
# 流式判定用的基准串:缓冲末尾是不是这个标签(或 `<think`)的前缀,决定要不要继续等。
TAG = "</think>"


def strip_thinking_markup(text: str) -> str:
    """清洗一段完整文本里的思考标记。

    按模块 docstring 的三条规则执行,顺序不能换:
    ① 先整块删掉成对的 `<think>...</think>`;
    ② 再处理**剩余**的孤立 `</think>`。这里刻意只认**最后一个**闭合标签:
       - 标签后面还有内容 ⇒ 说明"到最后一个标签为止"的前缀是思考,整段丢掉,只留后缀;
       - 标签后面是空的 ⇒ 标签只是尾巴上的多余符号,丢掉标签、保留前缀
         (实测最常见的就是 `答案</think>` 这一种);
    ③ 最后把剩下的开/闭标签本身铲掉,并 `strip()` 掉两端空白(清洗结果会直接当答案返回,
       不该带开头空行)。

    边界行为(与 `__main__` 自检一一对应):
    - `""` → 原样返回(不 strip 成别的);
    - `"<think>推理过程</think>答案是 2"` → `"答案是 2"`;
    - `"2</think>2"` → `"2"`(丢前缀、留后缀 —— 这正是"残留即表示前缀是思考"的规则);
    - `"答案</think>"` → `"答案"`(后缀为空 ⇒ 保留前缀);
    - `"共 2 张,合计 836.00 元"` → 原样(没有标签的正常回答绝不能被改动)。

    已知边界(不是 bug,是规则本身的取舍):**只有开标签、没有闭标签**时
    (如 `"<think>推理...` 被截断),标签会被铲掉而**标签之后的内容会保留** ——
    因为缺少闭合标签就无法区分"后面是思考"还是"后面是正文",此处选择不丢内容。
    """
    if not text:
        return text

    cleaned = _THINK_BLOCK_RE.sub("", text)

    # 还有闭合标签:如果标签后面有内容,丢掉"到最后一个标签为止"的前缀
    last_close = None
    for match in _CLOSING_TAG_RE.finditer(cleaned):
        last_close = match
    if last_close is not None:
        suffix = cleaned[last_close.end() :]
        if suffix.strip():
            cleaned = suffix
        else:
            cleaned = cleaned[: last_close.start()]

    cleaned = _ANY_TAG_RE.sub("", cleaned)
    return cleaned.strip()


class ThinkTagFilter:
    """流式清洗器:标记可能跨 chunk,先缓冲到能判定再吐内容。

    判定逻辑:
    - 缓冲里出现完整 `<think>...</think>` 块 → 丢掉该块;
    - 缓冲里出现 `</think>` 且有后续内容 → 丢掉前缀(思考内容);
    - 缓冲里一直是普通文本 → 直接吐出,避免把正常回答一直吞在缓冲里;
    - `flush()` 在流结束时把缓冲按同样的规则清洗后吐出。

    用法与线程安全(重要)
    --------------------
    **一次流式请求 new 一个实例**,它在流的生命周期内持有状态(`_buffer` / `_pending_close`)。
    不能做成模块级单例 / 跨请求共享 —— 两个并发回答的 chunk 会在同一个缓冲区里交错,
    结果是用户 A 看到用户 B 的半句话。`llm/chat.py::_iter_deltas` 里就是"每次调用 new 一个"。
    """

    def __init__(self, max_buffer: int = 32) -> None:
        # 缓冲上限:超过它就无条件吐出缓冲区内容(见 feed 里最后一个分支)。
        # 取 32 是因为 think 标签最长也就 8~10 个字符,32 足够覆盖"标签被切碎"的情况;
        # 再大就只是徒增首字节延迟(要等满 max_buffer 才吐字)。
        self.max_buffer = max_buffer
        self._buffer = ""
        # 缓冲结尾处出现孤立 </think> 时,不能立刻判定:后面还有内容说明前缀是思考,
        # 流就此结束则说明那只是一个多余标签,前缀是正文。等后续内容或 flush 再定。
        self._pending_close = False

    def feed(self, chunk: str) -> str:
        """喂一个 chunk,返回**确定可以发给用户**的文本(可能为空串)。

        空串不代表丢内容:它表示"当前信息不足以判定,先扣在缓冲区里"。
        """
        if not chunk:
            return ""

        if self._pending_close:
            # 标签后面真的还有内容 → 之前缓冲的那段是思考内容,丢掉
            # (上次 feed 已经把"前缀 + </think>"留在 _buffer 里,这里整体清空)
            self._buffer = ""
            self._pending_close = False

        self._buffer += chunk

        # 完整块可以直接清掉
        stripped = _THINK_BLOCK_RE.sub("", self._buffer)
        if stripped != self._buffer:
            self._buffer = stripped

        # 取**最后一个**闭合标签,与 `strip_thinking_markup` 的规则保持一致
        # (前面若有多个,说明它们属于更早的残留,统一按"前缀是思考"处理)。
        match = None
        for candidate in _CLOSING_TAG_RE.finditer(self._buffer):
            match = candidate
        if match is not None:
            suffix = self._buffer[match.end() :]
            if suffix.strip():
                self._buffer = suffix
                return self._drain()
            # 标签在末尾:挂起判定
            # (此刻不能说"前缀是正文"——下一 chunk 一到就证明它是思考内容了)
            self._pending_close = True
            return ""

        # 缓冲里可能正处在一个标签的前半段,等更多内容;否则直接吐出
        # 两个放行条件:① 攒够 max_buffer ② 末尾不可能再长成标签 ——
        # 都不能成立时返回空串继续等,这是"跨 chunk 标签"能正确清洗的关键。
        if len(self._buffer) >= self.max_buffer or not self._could_be_tag_prefix():
            return self._drain()
        return ""

    def flush(self) -> str:
        """流结束时调用:把缓冲区里剩下的内容按整段规则清洗后吐出。

        必须调用,否则**流最后一段回答会被永久扣在缓冲区里**(表现为回答少了尾巴)。
        这里顺带把 `_pending_close` 复位,让实例回到干净状态
        (尾部的孤立 `</think>` 会被 `strip_thinking_markup` 按"后缀为空 ⇒ 保留前缀"处理)。
        """
        text = strip_thinking_markup(self._buffer)
        self._buffer = ""
        self._pending_close = False
        return text

    def _drain(self) -> str:
        """吐出并清空缓冲区(顺手铲掉其中的残留标签),返回要发给用户的文本。"""
        text = _ANY_TAG_RE.sub("", self._buffer)
        self._buffer = ""
        return text

    def _could_be_tag_prefix(self) -> bool:
        """缓冲末尾是否可能是某个 think 标签的开头(需要继续等)。

        做法:枚举所有可能的尾长(1..len(TAG)),看它是不是
        `</think>` 或 `<think`(小写比较)的前缀。例如缓冲以 `<`、`</`、`</thi` 结尾时
        都返回 True(继续等),而以 `<p` 结尾时返回 False(不是标签,可以立刻吐字)。
        这是"尽量不引入延迟"与"绝不把标签劈成两半吐出去"之间的折中。
        """
        for length in range(1, min(len(TAG) + 1, len(self._buffer) + 1)):
            if TAG.startswith(self._buffer[-length:]) or "<think".startswith(
                self._buffer[-length:].lower()
            ):
                return True
        return False


if __name__ == "__main__":
    # 自检:断言式小样例,逻辑坏了会直接失败
    # (Windows 下跑:`python -X utf8 RAG\llm\text_clean.py`,最后会打印"自检通过")
    assert strip_thinking_markup("<think>推理</think>答案") == "答案"
    assert strip_thinking_markup("答案</think>") == "答案"
    assert strip_thinking_markup("2</think>2") == "2"
    assert strip_thinking_markup("普通答案") == "普通答案"
    # 这一组钉的是"标签跨 chunk":第一个 chunk 只能返回空串,拼上下一个才吐正文。
    filt = ThinkTagFilter()
    assert filt.feed("推理</thi") == ""
    assert filt.feed("nk>答案") == "答案"
    assert filt.flush() == ""
    print("text_clean.py 自检通过")

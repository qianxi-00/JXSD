"""LLM 调用:Agent 路由 / query 改写 / 非流式 / 流式输出(OpenAI 兼容接口)

流式接口逐段产出 {"reasoning": ...}(思考内容)或 {"content": ...}(回答内容),
正文里的思考标记残留由 llm/text_clean.py 清洗。

在 RAG 链路里的位置
==================
本文件是全项目**唯一的 LLM 出口**(`evaluation/llm_judge.py` 的评分调用除外,
它自己建客户端)。六个函数各管一段:

| 函数 | 对应课案策略 | 调用方 |
|---|---|---|
| `route_query` | Agentic RAG 的路由 + 改写策略选择 | `pipeline/rag_pipeline.py::run_events` |
| `rewrite_query` | HyDE / 子查询 / 回溯问题 三种改写 | 同上 |
| `generate_answer` / `stream_answer` | 带参考内容的回答(非流式 / 流式) | 同上(生成段) |
| `generate_direct_answer` / `stream_direct_answer` | 直答(不检索) | 同上(route=direct 分支) |

调用链:`rag_pipeline` → `route_query` 决定走不走 RAG、用哪种改写 → `rewrite_query` 出检索文本
→ 召回 + 重排 → `generate_answer`/`stream_answer` 用重排后的证据作答。

流式事件契约(跨模块,别改)
==========================
`_iter_deltas` 产出 `{"reasoning": 文本}` / `{"content": 文本}`;
`rag_pipeline._stream_llm` 把它**再包一层**成 `{"type": "thinking"|"token", "text": ...}` 给前端。
注意两级事件用的键名不同(这里是 `content`/`reasoning`,上层是 `text`)—— 拿错键名会
"链路跑通了但正文长度 0",本项目踩过(`RAG/app/chat_ui.py` 用的是对的那个)。

本文件守住的两个坑(都有回归用例)
================================
1. **思考模型会把预算先花在思考上**:路由这种"只要几个 token 输出"的任务必须
   显式关思考 + 给足 `max_tokens`,否则正文恒为空、路由**静默**退化成默认值
   (`tests/test_llm_router.py`);
2. **客户端必须显式设超时**:SDK 默认 600 秒 × 重试 3 次 ≈ 30 分钟,网关挂住时表现为
   整个 Web 应用卡死(实测卡过 46 分钟),对在线问答是不可接受的失败模式。
"""


import re

from openai import AsyncOpenAI, OpenAI

from config import settings
from core.logger import logger
from core.prompts import (
    BACKTRACK_PROMPT,
    DIRECT_QA_PROMPT,
    HYDE_PROMPT,
    RAG_ANSWER_PROMPT,
    ROUTER_PROMPT,
    SUBQUERY_PROMPT,
    SYSTEM_PROMPT,
)
from llm.text_clean import ThinkTagFilter, strip_thinking_markup

# 改写策略名 → 提示词模板。键必须与 `core.prompts.REWRITE_LABELS` 对得上。
# 这里**刻意没有 `direct`**:direct 的含义就是"不改写、直接用原问题检索",
# 所以 `rewrite_query("direct", ...)` 会因为 `REWRITE_PROMPTS.get("direct")` 返回 None 而返回 []。
REWRITE_PROMPTS = {
    "hyde": HYDE_PROMPT,
    "subquery": SUBQUERY_PROMPT,
    "backtrack": BACKTRACK_PROMPT,
}

# 路由只要求输出一个小 JSON，但**思考模型会把预算先花在思考上**：
# 实测（`RAG\script\probe_llm_thinking.py`，DeepSeek `deepseek-flash`）
# 域外问题（"今天杭州天气怎么样？"）思考 453~1111 字，256 的预算被吃光 ⇒
# `finish_reason=length`、正文为空 ⇒ 路由静默退化成"走 RAG + 直接检索"。
# 域内问题（"万宁的火车票票号是多少？"）思考 135~271 字，256 够用 ⇒ 正常。
#
# 所以这里两条腿都要有：① `_extra_body(False)` 真的关掉思考（关掉后只花 14 token）；
# ② 预算给足，作为"关思考的开关换了名字/被忽略"时的第二道保险。
#
# 为什么是常量而不是配置项:这是"路由输出足够出正文"的**下限契约**,不是可以按环境调的参数
# (`tests/test_llm_router.py` 钉的就是这两条)。真要调大,改这里并同步用例。
ROUTER_MAX_TOKENS = 256

# 子问题行首的编号标记：`1.` / `2、` / `(1) ` / `- ` / `• ` / `问题：` 等。
# 只剥**标记本身**，不碰正文里的数字（见子查询解析处的说明）。
# 数字后面必须跟分隔符或空白，`2025年…` 这种正文开头的年份不会被误剥。
_SUBQUERY_MARKER_RE = re.compile(r"^\s*(?:[-*•]\s*|\(?\d+\)?\s*[.、)）]?\s+|问题\s*[:：]\s*)")


def _strip_list_marker(line: str) -> str:
    """剥掉模型输出里子问题行首的编号/项目符号，保留正文（含开头数字）。"""
    return _SUBQUERY_MARKER_RE.sub("", line).strip()


def _extra_body(thinking: bool | None = None) -> dict:
    """构造"思考开关"的额外参数。

    thinking=None 跟随配置；显式传 False 用于**必须立刻出正文**的短任务（如路由）。

    ⚠️ 关思考到底该发哪个字段，是**实测出来的**，不是照文档猜的
    （探针 `RAG/script/probe_llm_thinking.py`，2026-09-17 在
    `LLM_BASE_URL=https://api.deepseek.com` + `LLM_MODEL=deepseek-flash` 上测的）：

    | 发包内容                      | 域外问题（思考更长）的实测结果                        |
    |-------------------------------|-------------------------------------------------------|
    | `{"enable_thinking": False}`  | 被**静默忽略**：reasoning 465/520 字、content 为空     |
    | `{"enable_thinking": False}` + max_tokens=1024 | 仍忽略：reasoning 2076 字，一次 content 37、一次为空 |
    | 不发任何字段                  | 同上（空）                                            |
    | `{"thinking": {"type": "disabled"}}` | 生效：reasoning **0** 字、content 40、只花 14 token |
    | `{"reasoning_effort": "none"}`       | 同上（与上一行等价）                          |

    结论：`enable_thinking` 是 DashScope/Qwen 系参数，这个端点不认；
    `thinking` / `reasoning_effort` 才是它认的。所以下面**两个都发**——
    对当前端点由 `thinking` 生效，对 Qwen 系网关由 `enable_thinking` 生效，
    两边都不会因此报 400（上表里两种都实测通过）。

    另外两点：
    - `enable_thinking` / `thinking` 都**不是** OpenAI 标准字段，只能塞进 `extra_body`
      （SDK 会原样并进请求体）；用错位置会被忽略或报参数错误；
    - 由此推出一个容易误解的点：`LLM_ENABLE_THINKING=true` 并不等于"这个端点真的在思考"，
      它只是"主动要求思考"——当前端点忽略该字段，实测只要不发 `thinking:disabled`
      它就一定会思考。要真正关掉，必须像路由这样显式传 False。
    """
    if thinking is False:
        # thinking 是当前端点真正生效的开关；enable_thinking 兼容 Qwen 系网关。
        return {"thinking": {"type": "disabled"}, "enable_thinking": False}
    if settings.llm.enable_thinking:
        return {"enable_thinking": True}
    return {}


def _client() -> OpenAI:
    """同步客户端：统一带上超时。

    不设超时会用 SDK 默认值（600 秒 × 重试 3 次）：私有网关在负载下会直接断连，
    请求就会挂住十几分钟 —— 实测卡过 46 分钟，Web 端表现为页面一直转圈。
    """
    return OpenAI(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        timeout=settings.llm.timeout,
    )


def _async_client() -> AsyncOpenAI:
    """异步客户端：同上，必须显式设超时。

    流式接口(async for)必须用 `AsyncOpenAI`:用同步客户端会阻塞事件循环,
    整个 Web 服务在生成期间无法响应任何其他请求。
    `timeout` 对**流式**请求约束的是"每两次数据之间的等待上限",不是总时长 ——
    所以超时不会掐断正常的长回答,只会掐住"网关不吐字"的卡死场景。
    """
    return AsyncOpenAI(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        timeout=settings.llm.timeout,
    )


def _extract_reasoning(obj) -> str | None:
    """从 message/delta 对象里取思考内容(`reasoning_content`),没有就返回 None。

    为什么要写两路取法:OpenAI SDK 是强类型模型,只有"模型定义里声明过"的字段才会成为属性;
    各家网关给思考内容起的字段名又不统一(DeepSeek 官方是 `reasoning_content`)。
    当它没进模型定义时,SDK 会把多余字段收进 `model_extra`,于是这里先试属性、再试 `model_extra`。

    同一份逻辑要同时适用于**非流式的 message** 和**流式的 delta**(两者都是 SDK 对象),
    所以入参是 any 而不是具体类型。返回值 None 表示"这次没有思考内容"
    (非思考模型、或思考被显式关掉),上层据此决定要不要给前端一个 thinking 事件。
    """
    reasoning = getattr(obj, "reasoning_content", None)
    if not reasoning:
        reasoning = (getattr(obj, "model_extra", None) or {}).get("reasoning_content")
    return reasoning


def _messages(question: str, context: str) -> list[dict]:
    """组装 RAG 作答的消息列表:system 定行为边界,user 带资料 + 问题。

    两个模板的分工:`SYSTEM_PROMPT` 管"依据资料、不许编造、标注 [n]",
    `RAG_ANSWER_PROMPT` 是带 `{context}` / `{question}` 占位符的正文模板。
    因此**改 `core/prompts.py` 时这两个占位符名不能动** ——
    这里用的是 `str.format(context=..., question=...)`(关键字),改名字会直接 KeyError。

    `context` 由 `pipeline/rag_pipeline.py::_build_context` 生成,格式是每条证据
    "元数据行 + `内容:` 行",编号从 `[1]` 开始 —— 正好对应系统提示词要求的引用编号。
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": RAG_ANSWER_PROMPT.format(context=context, question=question)},
    ]


def route_query(question: str) -> tuple[bool, str]:
    """Agent 路由:返回 (是否需要 RAG 检索, query 改写策略 hyde/subquery/backtrack/direct)

    返回值的契约(调用方依赖):
    - 第一个元素是布尔值,`rag_pipeline` 直接映射成 `route` 事件里的 "rag"/"direct";
    - 第二个元素**必须落在 `core.prompts.REWRITE_LABELS` 的四个键内** ——
      上层有一句 `REWRITE_LABELS[method]`,返回别的字符串会 KeyError。
      下面所有分支(解析失败、模型乱答、空输出)都只能落到这四个值之一;
    - 兜底方向永远是"走 RAG":宁可多检索一次,也不要让模型凭空答具体金额/票号。

    为什么用正则解析而不是 `response_format={"type": "json_object"}`:
    结构化输出不是所有 OpenAI 兼容网关都支持(本项目历史上用过私有网关),
    而路由**绝不能因为解析能力而失败** —— 所以这里用"尽量解析 + 解析不出就安全兜底"的策略。

    两个失败分支的日志级别不同,是刻意的:
    - 调用异常 → WARNING,附带异常原文(网络/鉴权/限流,能查);
    - 返回空正文 → WARNING,因为它是**思考吃光预算**的典型症状;
      而"解析不到 JSON 于是走了默认值"只记 DEBUG(不算异常)。
      过去这几种情况长得一模一样,路由全废了也没人发现 —— 这条 WARNING 就是为了让它现形。
    """
    client = _client()
    try:
        resp = client.chat.completions.create(
            model=settings.llm.model,
            # ROUTER_PROMPT 里 `{{...}}` 是**转义的花括号**:format 之后才是
            # 提示词里那段 `{"route": ...}` 的示例。这里只能传 question 一个变量。
            messages=[{"role": "user", "content": ROUTER_PROMPT.format(question=question)}],
            temperature=0,  # 路由要可复现:同一问题尽量给同一个策略,不要随机跳策略
            max_tokens=ROUTER_MAX_TOKENS,  # 见常量处的说明:给思考留出预算
            extra_body=_extra_body(False),  # 显式关思考:这是"必须立刻出正文"的短任务
        )
        text = (resp.choices[0].message.content or "").strip()
        # 记下 finish_reason 与思考长度：正文为空时它们是**唯一**能区分
        # "预算被思考吃光（length）" 与 "模型真就什么都没说（stop）" 的线索。
        finish_reason = getattr(resp.choices[0], "finish_reason", None)
        reasoning_len = len(getattr(resp.choices[0].message, "reasoning_content", None) or "")
    except Exception as exc:
        # 路由失败不能让整条问答链路挂掉 —— 退化成"走 RAG + 直接检索"仍然能答对大多数问题
        logger.warning(f"[路由] LLM 调用失败,默认走 RAG+直接检索: {exc}")
        return True, "direct"
    if not text:
        # 明确把"空输出"记为警告:以前它和"模型输出被解析成默认值"看起来一模一样,
        # 路由全废了也不会有人发现。
        logger.warning(
            f"[路由] LLM 返回空内容,已按默认值兜底: 走 RAG + 直接检索 "
            f"(finish_reason={finish_reason}, 思考 {reasoning_len} 字, max_tokens={ROUTER_MAX_TOKENS})"
        )
    logger.debug(f"[路由] LLM 原始输出: {text}")

    need_rag, rewrite = True, "direct"
    # 正则找 `"route": "xxx"`(允许冒号两侧有空格、大小写不敏感),不要求整段就是合法 JSON
    # —— 模型偶尔会在 JSON 前后补一句"好的",不该因此丢掉可解析的部分。
    m_route = re.search(r'"route"\s*:\s*"([^"]+)"', text, re.I)
    m_rewrite = re.search(r'"rewrite"\s*:\s*"([^"]+)"', text, re.I)
    # 三级兜底:① 取到 "route" 字段;② 整个输出就是裸的 RAG/DIRECT(小模型常这么答);
    # ③ 都不是 → 默认 "RAG"。注意 ③ 是**安全默认**:无法判断时走检索。
    # 又因为判定条件是 `!= "DIRECT"`,所以任何没认出来的取值(模型胡说)也都归到 RAG。
    route_value = (m_route.group(1).upper() if m_route else (text.upper() if text.upper() in ("RAG", "DIRECT") else "RAG"))
    need_rag = route_value != "DIRECT"
    if m_rewrite:
        rewrite_value = m_rewrite.group(1).lower()
        # 按固定顺序做"子串包含"匹配(模型常写 `"subquery"` 或 `"子查询检索 subquery"`,
        # 两种都能命中)。顺序即优先级:同时出现多个名字时取更靠前的那个。
        # 一个都匹配不上时保持初值 "direct"(原问题直接检索),这是最保守的选择。
        for method in ("hyde", "subquery", "backtrack", "direct"):
            if method in rewrite_value:
                rewrite = method
                break
    return need_rag, rewrite


def rewrite_query(question: str, method: str) -> list[str]:
    """按改写策略生成用于检索的 query 列表;direct 或失败时返回空列表(仅保留直接检索)

    为什么返回**列表**:子查询策略会拆出多条子问题,上层需要逐条去检索;
    返回空列表是"这次不改写"的统一表示,调用方 `rag_pipeline` 会把它和原问题并集
    (即原问题永远参与检索,改写只是**加料**,不是替换 —— 课案原文是按策略替换,
    本项目改成并集,差异已登记在 README)。

    失败与短路:
    - `method` 不认识(含 "direct",因为 REWRITE_PROMPTS 里没有它)或问题为空 → 直接返回 [];
    - 调用异常 → WARNING + 返回 []:改写是**增益**环节,失败不该让问答失败;
    - 模型返回空内容 → 同样返回 []。

    注意 `max_tokens` 用的是 `settings.llm.max_tokens`(默认 4096),比路由宽得多:
    HyDE 要写 100 字假设文、子查询要写 2~3 行,思考预算也含在里面。
    """
    prompt = REWRITE_PROMPTS.get(method)
    if not prompt or not question.strip():
        return []
    client = _client()
    try:
        resp = client.chat.completions.create(
            model=settings.llm.model,
            messages=[{"role": "user", "content": prompt.format(question=question)}],
            temperature=settings.llm.temperature,  # 改写要一点多样性,跟随全局配置(0.2)
            max_tokens=settings.llm.max_tokens,
            extra_body=_extra_body(),  # 改写跟随配置;开思考会让改写更慢但通常更准
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning(f"[改写] {method} 失败,回退仅直接检索: {exc}")
        return []
    if not text:
        return []
    if method == "subquery":
        # 子查询提示词要求"每行一个子问题、只输出编号",但模型常会写成
        # `1. 问题` / `- 问题` / `(1)问题` —— 所以按行切完再剥掉**行首的编号标记**。
        #
        # 这里刻意不用 `str.strip("0123456789.、()•-–")`:strip 的参数是**字符集合**,
        # 会连正文开头的数字一起削掉 —— 实测 `1. 2025年张三的火车票金额是多少`
        # 会变成 `年张三的火车票金额是多少`,静默改掉检索语义。
        subs = [_strip_list_marker(ln) for ln in text.splitlines() if ln.strip()]
        # 双保险过滤(空行已在上面滤过一次,这里滤的是"strip 完变成空串"的行),
        # 并**截断到 3 条** —— 与提示词里"2~3 个子问题"的上限对齐,防止模型写 10 行
        # 把召回次数放大 10 倍(每条子问题都会触发两路召回)。
        return [s for s in subs if s][:3]
    # hyde / backtrack 都是"一整段文本",直接作为一条检索 query。
    # 注意 HyDE 的假设文长度不受 100 字约束(提示词只是"要求"),这里不做截断:
    # 超长只会多花点 embedding token,截断反而可能切掉关键信息。
    return [text]


def generate_answer(question: str, context: str) -> tuple[str, str | None]:
    """非流式输出:基于检索上下文一次性返回完整回答,返回 (回答, 思考内容)

    一句话说完就是"拿证据问模型",但有两处值得留意:
    - 返回值是**二元组**:正文经过 `strip_thinking_markup` 清洗(去掉推理标记残留),
      思考内容单独返回给前端展示,不再混进正文;
    - `temperature` / `max_tokens` / 思考开关三件套都跟随 `settings.llm.*`,
      与路由那条特例(固定 temperature=0、固定预算、强制关思考)是两套配置,别混。

    异常不在这里捕获:生成失败必须让调用方感知(`rag_pipeline` 会回一条
    "大模型服务不可用"的错误事件,而不是把半截回答当成功)。
    """
    client = _client()
    resp = client.chat.completions.create(
        model=settings.llm.model,
        messages=_messages(question, context),
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
        extra_body=_extra_body(),
    )
    message = resp.choices[0].message
    return strip_thinking_markup(message.content or ""), _extract_reasoning(message)


def generate_direct_answer(question: str) -> tuple[str, str | None]:
    """非流式输出:不经过检索,由 LLM 直接回答,返回 (回答, 思考内容)

    与 `generate_answer` 的差别只有一个:消息里**没有 SYSTEM_PROMPT、也没有资料**,
    只有一条带问题的 user 消息(直答路径本就不该受"只依据资料作答"的约束)。
    这也是本项目与课案的一处已知差异,已登记在 README。

    什么时候会走到这里:`route_query` 判定为 DIRECT(纯计算/常识/闲聊)。
    """
    client = _client()
    resp = client.chat.completions.create(
        model=settings.llm.model,
        messages=[{"role": "user", "content": DIRECT_QA_PROMPT.format(question=question)}],
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
        extra_body=_extra_body(),
    )
    message = resp.choices[0].message
    return strip_thinking_markup(message.content or ""), _extract_reasoning(message)


async def _iter_deltas(stream):
    """把 LLM 流式响应切成 reasoning / content 事件,并清洗思考标记残留。

    reasoning 走独立字段(OpenAI 兼容接口的 reasoning_content),content 才是正文;
    但实测正文里偶尔会混入 `</think>` 残留,且可能跨 chunk 断开,
    因此用 ThinkTagFilter 缓冲清洗,流结束时 flush 出剩余内容。

    逐段实现上的几个"为什么":
    - `if not chunk.choices: continue` —— 流里会混入 usage 统计等**没有 choices**的帧,
      直接取 `chunk.choices[0]` 会 IndexError;
    - reasoning 与 content **可能出现在同一个 chunk 里**(先吐思考再吐正文),
      所以是两个独立的 if,不是 if/elif;
    - 正文必须先过 `ThinkTagFilter.feed` 再 yield:清洗器会把"可能是标签前半段"的内容
      扣在缓冲区里,因此**某些 chunk 的返回值是空串**(`if cleaned` 就是为此),
      这不代表丢内容 —— 剩下的会在后续 chunk 或 `flush()` 里吐出来;
    - 结尾的 `flush()` 不可省:流结束时缓冲区里可能还压着没有标签收尾的正文;
    - `await stream.close()` 显式关掉底层 HTTP 连接,避免连接悬着不释放。
      (注意它是**正常走完**才执行:消费方提前 break 时,这个异步生成器被 `aclose()`
      打断在 yield 处,这一行不会被执行 —— 这是已知的小缺口,不影响正常链路。)
    """
    tag_filter = ThinkTagFilter()
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        reasoning = _extract_reasoning(delta)
        if reasoning:
            yield {"reasoning": reasoning}
        if delta and delta.content:
            cleaned = tag_filter.feed(delta.content)
            if cleaned:
                yield {"content": cleaned}
    tail = tag_filter.flush()
    if tail:
        yield {"content": tail}
    await stream.close()


async def stream_answer(question: str, context: str):
    """流式输出:基于检索上下文,逐段产出 {"reasoning": ...} 或 {"content": ...}

    这是 RAG 生成段的流式版本(非流式对应 `generate_answer`),两者**必须保持同样的
    prompt 与参数** —— 否则前端切换流式开关会得到不同质量的回答。

    本函数是异步生成器:`stream=True` 让 SDK 返回可迭代的流对象,
    真正的切帧/清洗逻辑在 `_iter_deltas` 里(两处流式出口共用同一套清洗)。
    """
    client = _async_client()
    stream = await client.chat.completions.create(
        model=settings.llm.model,
        messages=_messages(question, context),
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
        stream=True,
        extra_body=_extra_body(),
    )
    async for event in _iter_deltas(stream):
        yield event


async def stream_direct_answer(question: str):
    """流式输出:不经过检索,由 LLM 直接回答,逐段产出 {"reasoning": ...} 或 {"content": ...}

    与非流式的 `generate_direct_answer` 对应(同样只发一条 user 消息、不带资料)。
    注意去重:直答路径的**首字延迟**主要花在 LLM 上,若开了思考,用户会先看到一大段
    thinking 事件、正文迟迟不来 —— 这是"思考模型 + 流式"的正常表现,不是卡死。
    """
    client = _async_client()
    stream = await client.chat.completions.create(
        model=settings.llm.model,
        messages=[{"role": "user", "content": DIRECT_QA_PROMPT.format(question=question)}],
        temperature=settings.llm.temperature,
        max_tokens=settings.llm.max_tokens,
        stream=True,
        extra_body=_extra_body(),
    )
    async for event in _iter_deltas(stream):
        yield event

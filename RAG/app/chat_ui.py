"""Chainlit 前端界面:AI 聊天问答,可视化展示 RAG 全链路执行过程

功能:
- 侧边设置(齿轮)中可切换 流式 / 非流式 输出;
- 每一步链路(缓存 -> Agent 路由 -> Query 改写 -> 四路召回 -> 重排序 -> LLM 思考 -> 回答)
  以可视化 Step 呈现,可清晰看到是否命中缓存、改写策略、召回内容与分数。
"""

# --- 路径引导：确保能导入 Python_Base 根目录的 config 与 RAG 内部包 ---
# 本文件不是靠 `python chat_ui.py` 跑的，而是被 Chainlit **按文件路径加载**：
#   - 独立进程方式：`chainlit run RAG\app\chat_ui.py`；
#   - 嵌入式方式（本项目实际用法）：app\main.py 里 mount_chainlit(target=本文件)，
#     由 chainlit/config.py::load_module 用 importlib 加载。
# 而 load_module 只把**本文件所在目录**（RAG\app）插进 sys.path ——
# 仓库根（Python_Base，放 config.py）与 RAG 根（放 core/llm/pipeline）都不在里面，
# 所以下面这段引导不能删，删了就是 `from core.prompts import ...` 的 ImportError。
# 同一段样板在 app\main.py、core\milvus_init.py 里各有一份（复制三处），改动要同步。
import sys as _sys
from pathlib import Path as _Path

# 从本文件位置逐级向上找名为 Python_Base 的祖先目录；`_BASE.parent != _BASE` 是到顶保护
# （Path 到达盘符根后 parent 等于自己），没有它就成了死循环。
_BASE = _Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
_sys.path.insert(0, str(_BASE))          # Python_Base 根（config.py）
_sys.path.insert(0, str(_BASE / "RAG"))  # RAG 根（core/llm/pipeline 等包）
# 两次 insert(0) 之后顺序是 [RAG 根, Python_Base 根, ...]，即项目自己的
# core / llm / pipeline 优先于同名的第三方包被解析到。
# 别名用 _sys / _Path，避免给模块命名空间留下易与业务变量重名的 sys / Path。
# （这份 docstring 原先写在这段引导**之后**，因而 `app.chat_ui.__doc__` 是 None；
# 已挪到文件首个语句。挪位置不影响引导顺序：docstring 是惰性字符串，不执行任何东西。）

# 本文件的架构要点（读代码前先立住这一条）：
#   Chainlit 侧**不重写**任何 RAG 逻辑，只是把 rag_pipeline.run_events 吐出来的事件流
#   翻译成 UI：一个事件 → 一个 cl.Step（或一段流式正文）。所有路由/改写/召回/重排的
#   取舍都发生在 pipeline 里，这里只负责"把它显示出来"，因此页面上看到的内容与
#   /api/chat 返回的答案是同一条链路产生的。
# 入口是下面几个 @cl.on_* 装饰器（没有 `if __name__ == "__main__"` 块）：
# Chainlit 在会话开始 / 收到消息 / 改设置时回调它们，文件本身不启动服务。

import json

import chainlit as cl
from chainlit.input_widget import Select, Switch  # 设置面板控件：线路选择 + 流式开关

from core.prompts import REWRITE_LABELS  # 改写策略的中文名，页面与日志共用同一份映射
from core.routes import ROUTE_BASIC, ROUTE_LABELS, ROUTE_SUPPORTS_HISTORY
from pipeline.modes import answer_events, list_modes

# ⚠ 本文件**不再**自己 new 一个 RAGPipeline：基础线路的实例由
# `pipeline.modes._basic_pipeline()` 惰性创建并在进程内复用，HTTP 接口（app/main.py）
# 与这个页面因此共用同一个实例、同一份 preset 矩阵与 BM25 索引。
# 之前两处各建一个的后果：同一个问题走接口和走页面会各自加载一次语料，
# 而且"缓存命中"看起来忽高忽低（两个实例各自预热）。


def _fmt_date(value) -> str:
    """把库里的 date_int（YYYYMMDD 整数）显示成 2025-01-01。

    注意判据是**真值**（`if not value`），所以 date_int=0 与 None 一样显示「无」；
    库里的 date_int 要么是合法的 YYYYMMDD，要么是 null，不会出现 0，因此没有歧义。
    """
    if not value:
        return "无"
    return f"{value // 10000:04d}-{value // 100 % 100:02d}-{value % 100:02d}"


def _fmt_amount(value) -> str:
    """把库里的 amount_fen（单位：**分**）显示成「1234.56元」。

    判据是 `is None`（而不是真值），所以 amount_fen=0 会显示成「0.00元」而不是「无」——
    与上面 _fmt_date 的写法不同：金额 0 是**有意义的取值**（免费票/零元发票），
    日期 0 不是，两者对 null 与 0 的语义要求不同。
    """
    if value is None:
        return "无"
    return f"{value / 100:.2f}元"


async def _render_evidence_step(final: dict, mode: str) -> None:
    """②③④ 线路的证据构成 Step（①线路不用，它有自己的 ④⑤ 两步）。

    `final` 是 done 事件；`extra` 里放的是各线路自己的明细（见 `pipeline/modes.py`）：
      - graph ：communities / nodes / relationships
      - fusion：上面的三类 + sql / unmatched_numbers / queries
    用一个函数按 key 的存在性渲染，而不是为每条线路写一个分支 ——
    新增线路只要往 extra 里塞字段，这里自动就能显示。
    """
    extra = final.get("extra") or {}
    if not extra:
        return

    parts: list[str] = []
    communities = extra.get("communities") or []
    if communities:
        lines = [
            f"- [图社区{item.get('community_id')}] 相似度 {item.get('score'):.4f}"
            f"：{(item.get('summary') or '')[:120]}…"
            for item in communities
            if isinstance(item.get("score"), (int, float))
        ]
        parts.append(f"**图谱社区 {len(communities)} 段**\n" + ("\n".join(lines) or "_无分数信息_"))

    relationships = extra.get("relationships") or []
    if relationships:
        lines = [f"- {r.get('source')} --{r.get('relation')}--> {r.get('target')}" for r in relationships[:12]]
        parts.append(f"**图谱关系 {len(relationships)} 条**（最多显示 12 条）\n" + "\n".join(lines))

    sql = extra.get("sql")
    if sql:
        parts.append(
            "**结构化统计（Text-to-SQL）**\n"
            f"```sql\n{sql.get('sql')}\n```\n"
            f"结果：{json.dumps(sql.get('rows'), ensure_ascii=False, default=str)}"
        )

    unmatched = extra.get("unmatched_numbers") or []
    if unmatched:
        parts.append(
            f"**⚠️ 数字核验**：{len(unmatched)} 个数字在本次证据里找不到可核对来源 —— "
            f"{', '.join(unmatched)}"
        )

    queries = extra.get("queries") or []
    if queries:
        parts.append("**本次检索文本**\n" + "\n".join(f"{i}. {q}" for i, q in enumerate(queries, 1)))

    if not parts:
        return

    # 用 `async with` 而不是"构造 Step 再 send"：Chainlit 的 Step 生命周期由上下文管理器
    # 负责（进入时创建、退出时自动 send），自己 send 容易在异常路径上漏掉收尾。
    async with cl.Step(name=f"🧩 证据构成（{ROUTE_LABELS.get(mode, mode)}）", type="retrieval") as step:
        step.output = "\n\n".join(parts)


def _rows_table(rows: list[dict], score_key: str) -> str:
    """把一批召回记录渲染成 Markdown 表格（Chainlit 的 Step 输出支持 Markdown）。

    score_key 由调用方给：向量召回传 "vector_score"、BM25 传 "keyword_score"、
    重排后传 "rerank_score" —— 同一张表要适配三种分数来源。

    两个刻意的写法：
    - 空列表返回 `_无_`（Markdown 斜体）而不是空串，避免 Step 里出现「有个标题却没内容」
      让人误以为渲染坏了；
    - 分数判据用 isinstance(score, (int, float)) 而不是真值 —— 相关度 0.0 是**真实分数**，
      真值判断会把它显示成「无」，看起来像没打分。
    """
    if not rows:
        return "_无_"
    header = "| # | 来源 | 类型 | 票号 | 相关人/买方 | 日期 | 金额 | 路线 | 交易对方 | 分数 |"
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for i, r in enumerate(rows, 1):
        score = r.get(score_key)
        # 非数值一律显示「无」（字段缺失/接口没回分数），而不是显示 None。
        score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "无"
        # `or '无'` 把 None 与空串一起兜住；字段值直接拼进 Markdown，
        # 若值里出现 `|` 或换行会把表格撑破（当前数据的 route 用 `-` 连接，不含 `|`）。
        lines.append(
            f"| {i} | {r.get('source_file') or '无'} | {r.get('ticket_type') or '无'}"
            f" | {r.get('ticket_no') or '无'} | {r.get('person') or '无'}"
            f" | {_fmt_date(r.get('date_int'))} | {_fmt_amount(r.get('amount_fen'))}"
            f" | {r.get('route') or '无'} | {r.get('counterparty') or '无'} | {score_text} |"
        )
    return "\n".join(lines)


@cl.on_chat_start
async def on_chat_start():
    """会话建立时：给出默认设置值 + 渲染设置面板 + 发欢迎消息。"""
    # 先写默认值再发设置面板：控件的 initial 只影响外观，
    # 真正被 on_message 读取的是 user_session 里的值，
    # 这里先落一次默认值，避免用户没动设置时读到 None。
    cl.user_session.set("stream_mode", True)
    cl.user_session.set("mode", ROUTE_BASIC)
    # 会话历史（多轮用）：由 on_message 在每轮结束时追加。
    # 放 user_session 而不是全局变量 —— 它是**按会话**隔离的，多用户/多标签页互不干扰。
    cl.user_session.set("history", [])
    # ChatSettings 发出后，Chainlit 会把当前所有设置项在改动时回传给 on_settings_update。
    await cl.ChatSettings(
        [
            Switch(id="stream_mode", label="流式输出", initial=True),
            # 四条线路的选择器：**只能给 items，不能同时给 values**
            # （Chainlit 的 Select.__post_init__ 明确禁止两者并存，同时给会直接抛
            #  `Value error, You can only provide either values or items`，
            #  表现是页面上的设置面板整块渲染失败 —— 真机在浏览器里才看得到，
            #  单元测试与 HTTP 接口都发现不了）。
            # items 的键是显示文案、值是线路键；initial_value 要给**值**（不是文案）。
            Select(
                id="mode",
                label="RAG 线路",
                items={
                    spec["label"] + ("（支持多轮）" if spec["supports_history"] else "（单轮）"): spec["key"]
                    for spec in list_modes()
                },
                initial_value=ROUTE_BASIC,
            ),
        ]
    ).send()
    # 欢迎文案里把四条线路列出来：用户看到选择器时才知道每条线路会做什么。
    route_lines = "\n".join(
        f"- **{spec['label']}**：{spec['summary']}" for spec in list_modes()
    )
    await cl.Message(
        content=(
            "您好!我是财务票据智能问答助手 🎫\n\n"
            "支持三类票据问答:✈️ 登机牌 / 🧾 发票 / 🚄 火车票\n\n"
            "**四条 RAG 线路**（右上角 ⚙️ 设置里切换）:\n"
            f"{route_lines}\n\n"
            "⚠️ 只有 **② Agentic** 与 **④ 融合** 线路支持多轮追问；\n"
            "① 与 ③ 是单轮线路 —— 追问里的「那……呢？」没有解析者，可能答不上来。\n\n"
            "① 线路会展示完整链路：缓存 → Agent 路由 → Query 改写 → 双路召回 → 重排序 → 回答；\n"
            "②③④ 线路会展示本次实际用到的证据构成。"
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict):
    """用户改设置时：把开关写回会话，并发一条确认消息（让用户看到"生效了"）。"""
    # Chainlit 会把**全部**设置项一次传进来，所以用 .get + 默认值读取，
    # 将来加设置项时不会因为缺键而 KeyError。
    stream_mode = bool(settings.get("stream_mode", True))
    cl.user_session.set("stream_mode", stream_mode)
    mode = settings.get("mode") or ROUTE_BASIC
    cl.user_session.set("mode", mode)
    lines = [f"⚙️ 已切换为 **{'流式' if stream_mode else '非流式'}输出**。"]
    lines.append(f"🚦 当前 RAG 线路：**{ROUTE_LABELS.get(mode, mode)}**")
    if not ROUTE_SUPPORTS_HISTORY.get(mode, False):
        lines.append(
            "⚠️ 该线路**不支持多轮**：它只把当前这一句交给链路，"
            "追问里的代词（「那……呢？」）不会被解析。要追问请切到 ② 或 ④。"
        )
    await cl.Message(content="\n".join(lines)).send()


@cl.on_message
async def on_message(message: cl.Message):
    """每条用户消息的入口：消费 run_events 事件流，把链路翻译成可视化 Step + 流式正文。

    事件 → UI 的映射（这是本文件的主干，改动要成对改）：
      cache_hit   → Step「① 缓存层:命中 ✅」（命中才出现）
      route       → Step「② Agent 路由」      （DIRECT / RAG + 改写策略）
      rewrite     → Step「③ Query 改写」      （只在真的生成了改写 query 时出现）
      retrieve    → Step「④ 双路召回 + 合并去重」（四张表：直检/改写 × 向量/BM25）
      rerank      → Step「⑤ 重排序(Reranker)」
      thinking    → Step「LLM 思考过程」      （思考模型才有）
      no_evidence → Step「⚠️ 无可靠依据」     （召回全低于阈值，走保守回复）
      token       → 追加到正文消息（流式渲染）
      error       → 也追加到正文消息（用 ⚠️ 文案，不弹错误框）
      done        → 只记下来，末尾决定要不要补「参考票据」引用
    start 事件在这里**没有**对应的 Step（它只表示链路已开始）；
    llm_done 是 pipeline 内部记号，从来不会流到 UI。
    """
    question = message.content.strip()
    # 会话级设置：默认 True。用 default 兜住"设置面板还没初始化"的窗口期。
    # `stream_mode` 只作用于基础线路（它的 LLM 调用有流式/非流式两条路径）；
    # 另外三条线路的耗时都在生成之前的检索与工具环节，没有可切换的东西 ——
    # 所以这里必须把它一路传进 `answer_events`，否则页面上的开关就是个摆设。
    stream_mode = bool(cl.user_session.get("stream_mode", True))
    mode = cl.user_session.get("mode") or ROUTE_BASIC
    # 历史在多轮线路上是输入，在所有线路上都要**追加**（见本轮末尾），
    # 这样"先用①问一句、再切到②追问"也能带上那一句的语境。
    history: list[dict] = list(cl.user_session.get("history") or [])

    # 正文消息**惰性创建**：直到第一个 token/error 事件才 new，
    # 这样"纯 Step、没有正文"的路径（例如保守回复）不会先在页面上留一条空消息。
    answer_msg: cl.Message | None = None
    # 存放 done 事件，供末尾拼引用块用；事件流异常中断时它就是空 dict（下面用 .get 取值）。
    final: dict = {}

    # run_events 是异步生成器：这里一方面把事件喂给 UI，另一方面它本身就是"边算边显示"的驱动。
    # stream=stream_mode 决定 pipeline 内部走流式还是非流式 LLM 调用；但无论哪种模式，
    # 正文都是通过 token 事件交给本函数的，所以下面只要处理 token 即可，不需要分支。
    #
    # 四条线路统一走 `pipeline.modes.answer_events`：基础线路吐的是它原有的细粒度事件流，
    # 另外三条各只吐 start/route/token/done（它们的耗时在检索与工具调用上，
    # 真流式也只能在最后一步开始吐字）。所以下面的 `elif` 里对"没有细粒度事件"
    # 的线路要能优雅降级 —— 见 done 分支里的证据渲染。
    async for ev in answer_events(question, mode, history, stream=stream_mode):
        t = ev["type"]

        if t == "cache_hit":
            # 缓存命中意味着后面的路由/召回/重排**全都没跑**，页面上不会出现 ②~⑤ 的 Step ——
            # 这是正常的，不是渲染丢了。用一张 JSON 卡片把"命中方式/匹配问题/相似度"摆出来，
            # 便于判断是精确命中（10 分钟内问过同一句）还是预设问答相似度命中。
            detail = {
                "命中方式": "精确缓存(10分钟内重复提问)" if ev["mode"] == "exact" else "预设问答相似度命中",
                # `or "—"` 处理空串/None；下面相似度用 `is not None` 判据，
                # 是为了让 0.0 这种真实分数也能显示出来（真值判断会把它当"没有"）。
                "匹配问题": ev.get("matched_question") or "—",
                "相似度": ev.get("similarity") if ev.get("similarity") is not None else "—",
                # 预设问答命中的 sources 恒为空列表（core/cache.py 里就是这么存的），
                # 所以这里基本会显示「缓存答案无来源」—— 这也是页面上判断
                # "答案来自缓存而非检索"的主要线索。
                "来源": [s.get("source_file") for s in ev.get("sources", [])] or "缓存答案无来源",
            }
            # `async with cl.Step(...)`：进入时创建 Step，退出时自动 send()，
            # 所以只需在块内给 step.output 赋值。type 决定前端图标/标签，
            # 合法取值是 run/tool/llm/embedding/retrieval/rerank/undefined 这几种。
            async with cl.Step(name="① 缓存层:命中 ✅", type="tool") as step:
                step.output = json.dumps(detail, ensure_ascii=False, indent=2)

        elif t == "mode":
            # ②③④ 线路的第一步：告诉用户"这次走的是哪条线路"。
            # 注意它与基础线路的 `route` 事件**不是一回事**：`route` 是 LLM 的路由判断
            # （rag/direct），这里是用户在设置里选的线路。两者分开渲染，别合并。
            async with cl.Step(name=f"🚦 线路：{ev.get('label') or ev.get('mode')}", type="run") as step:
                step.output = ev.get("summary") or ""

        elif t == "route":
            # 路由只决定"要不要检索"和"用哪种改写策略"，不产生正文 ——
            # 这里把它渲染成一段说明文字，用户才知道后面为什么没有召回步骤（DIRECT 分支）。
            # ⚠ 用 `.get` 读：事件是跨模块契约，字段缺失时宁可不显示也不能让
            # on_message 整条崩掉（真机踩过：`ev["route"]` 的 KeyError 会让页面上
            # 只剩提问、连回答都不渲染）。真值缺失时按"RAG+直接检索"展示。
            if (ev.get("route") or "rag") == "direct":
                text = "LLM 路由判断:**DIRECT**,简单问题,无需检索,直接回答"
            else:
                # REWRITE_LABELS.get(..., "直接检索")：模型可能返回没见过的策略名，
                # 兜底成"直接检索"而不是抛 KeyError，免得 UI 因为一个标签挂掉。
                label = REWRITE_LABELS.get(ev.get("rewrite") or "direct", "直接检索")
                text = (
                    f"LLM 路由判断:**RAG**,涉及知识库内容,开始检索票据\n\n"
                    f"query 改写策略:**{label}**(召回阶段将保留直接检索)"
                )
            async with cl.Step(name="② Agent 路由", type="llm") as step:
                step.output = text

        elif t == "rewrite":
            # 只有真的生成了改写 query 才有这个事件（direct 策略或改写失败时不会出现），
            # 所以页面上 ③ 时有时无是正常的。编号从 1 开始编号，方便与召回结果对照。
            items = "\n".join(f"{i}. {q}" for i, q in enumerate(ev["rewritten"], 1))
            async with cl.Step(name="③ Query 改写", type="tool") as step:
                step.output = (
                    f"改写策略:**{ev['label']}**\n\n生成检索 query {len(ev['rewritten'])} 条:\n{items}\n\n"
                    f"_改写检索将与直接检索的结果做并集召回_"
                )

        elif t == "retrieve":
            # 召回明细分四张表：直接检索/改写检索 × 向量/BM25。
            # 这里用的是"双路"的叫法（向量 + BM25 两条路），而 run_events 的文档
            # 与 README 里也叫"四路召回"（再乘上直检/改写两个 query 来源）—— 同一件事的两种说法。
            sections = [
                f"**直接检索 · 向量召回 {len(ev['direct_vector_hits'])} 条**",
                _rows_table(ev["direct_vector_hits"], "vector_score"),
                f"\n**直接检索 · BM25 召回 {len(ev['direct_keyword_hits'])} 条**",
                _rows_table(ev["direct_keyword_hits"], "keyword_score"),
            ]
            # 改写那两张表只在有内容时渲染：没有改写时显示四个空表反而干扰判断。
            if ev.get("rewrite_vector_hits") or ev.get("rewrite_keyword_hits"):
                sections += [
                    f"\n**改写检索({ev['label']}) · 向量召回 {len(ev['rewrite_vector_hits'])} 条**",
                    _rows_table(ev["rewrite_vector_hits"], "vector_score"),
                    f"\n**改写检索 · BM25 召回 {len(ev['rewrite_keyword_hits'])} 条**",
                    _rows_table(ev["rewrite_keyword_hits"], "keyword_score"),
                ]
            # 合并去重后的条数才是交给重排的候选数，所以单独强调一行 ——
            # 它和上面四张表的行数之和不相等（有重复 id 被去掉了）。
            sections.append(f"\n**合并去重后:{ev['merged_count']} 条**")
            async with cl.Step(name="④ 双路召回 + 合并去重", type="retrieval") as step:
                step.output = "\n".join(sections)

        elif t == "rerank":
            # 这一步是"决策门控"：先取 top_k，再按 rerank.relevance_p 卡相关度。
            # 页面上把三段数字都摆出来（候选 → top_k → 阈值 → 保留），
            # 这样"为什么没有依据/为什么只留了 1 条"能自己看出来。
            # ⚠️ relevance_p 是**跟着 reranker 模型走的**（换模型必须用
            # script/calibrate_rerank.py 重标），不要把这里的数字当成可随意调的经验值。
            kept = ev["kept"]
            body = (
                f"候选 {ev['merged_count']} 条 → 重排取 top_k={ev['top_k']} →"
                f" 相关度阈值 {ev['relevance_p']} 过滤后 **保留 {len(kept)} 条**\n\n"
                f"{_rows_table(kept, 'rerank_score')}"
            )
            async with cl.Step(name="⑤ 重排序(Reranker)", type="rerank") as step:
                step.output = body

        elif t == "thinking":
            # 思考模型的 reasoning_content。它先于正文到达（pipeline 侧做了一次缓冲，
            # 保证整段思考一次性给出），所以这里直接用 Step 展示，不参与正文拼接。
            async with cl.Step(name="LLM 思考过程", type="thinking") as step:
                step.output = ev["text"]

        elif t == "token":
            # 正文渲染：没有分支地一路往同一条消息里追加。
            # 缓存命中与保守回复的正文也是用 token 事件分片送过来的（pipeline 里按 6 字切块），
            # 所以这三条路径在 UI 上的表现完全一样。
            if answer_msg is None:
                answer_msg = cl.Message(content="")
            await answer_msg.stream_token(ev["text"])

        elif t == "no_evidence":
            # 召回结果全部低于相关度阈值 —— pipeline 刻意不调用 LLM（避免编造），
            # 直接给保守回复。这里补一条 Step 解释"为什么没有正常回答"。
            async with cl.Step(name="⚠️ 无可靠依据", type="tool") as step:
                step.output = "召回内容全部低于相关度阈值,触发保守回复,不调用 LLM 生成。"

        elif t == "error":
            # 链路中途出错（Milvus/Reranker/LLM 不可用）：也被当成正文推给用户，
            # 而不是抛异常 —— 这样前面已经渲染出来的 Step 不会白费，报错也带着排查提示。
            if answer_msg is None:
                answer_msg = cl.Message(content="")
            await answer_msg.stream_token(ev["message"])

        elif t == "done":
            # 只记录，不渲染：末尾要判断是否追加引用块 + 渲染证据构成。
            final = ev

    # ②③④ 线路没有细粒度事件，`done` 里带着本次的证据构成（`extra`），
    # 这里统一补一条 Step —— 否则用户切到 Agentic/融合线路后，页面上只剩一个答案，
    # 完全看不出"这次到底查了什么"，与①线路的可见性差距太大。
    await _render_evidence_step(final, mode)

    if answer_msg is not None:
        # 引用块的两个条件：
        #   - 有 sources：缓存命中（尤其预设命中）与保守回复的 sources 都是空列表；
        #   - 非保守回复：conservative=True 表示"没找到依据"，此时列引用是误导。
        # `is not True` 而不是 `not ...`：显式对齐 done 事件里 True/False 的布尔语义。
        if final.get("sources") and final.get("conservative") is not True:
            cite = "\n\n---\n📎 **参考票据**\n" + "\n".join(
                f"- `{s.get('source_file')}`({s.get('ticket_type')},票号 {s.get('ticket_no') or '无'})"
                for s in final["sources"]
            )
            # 引用块也用 stream_token 追加，保持在同一条消息里。
            await answer_msg.stream_token(cite)
        # ★ 必须显式 send() 一次：cl.Message(content="") 只是构造了消息对象，
        # stream_token 把它挂到会话里，但"消息完成"的事件只有 send() 才发出 ——
        # 少了这句，页面上的正文会停在"进行中"状态。
        await answer_msg.send()

    # 把这一轮追加进会话历史（无论哪条线路都记）：
    #   - 多轮线路（②④）下一轮会用到它；
    #   - 用户在①问一句、再切②追问时，那一句也必须在历史里，否则②看不到语境。
    # 用 user_session 存：按会话隔离，不跨用户/标签页串。
    history.append({"role": "user", "content": question})
    if answer_msg is not None:
        history.append({"role": "assistant", "content": answer_msg.content})
    # 只留最近 10 条（5 轮）：历史越长，多轮线路的提示词越贵，且旧语境容易干扰当前问题。
    cl.user_session.set("history", history[-10:])

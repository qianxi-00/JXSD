"""RAG 流水线(事件化):
缓存层 -> Agent 路由(路由 + query 改写策略) -> 直接检索与改写检索双路并集召回
-> 合并去重 -> 重排序 -> 拼接元数据 -> LLM 生成

run_events 逐步产出事件供前端可视化:
  start / cache_hit / route / rewrite / retrieve / rerank / thinking / no_evidence / token / done

════════════════════════════════════════════════════════════════
为什么是"事件化"而不是一个普通函数
════════════════════════════════════════════════════════════════
同一条链路要同时服务四种消费方：
  1. 非流式 HTTP 接口（app/main.py 的 POST /api/chat）——要最终答案 + sources；
  2. SSE 流式接口（POST /api/chat/stream）——要能边算边推；
  3. Chainlit 页面（app/chat_ui.py）——要把每一阶段渲染成可视化 Step；
  4. 评估脚本（script/run_stage_eval.py 等）——要分阶段指标（路由/过滤/召回/重排各自的对错）。
这四种需求无法用一个返回值同时满足。解法是：本模块只负责**算**并产出结构化事件，
"怎么显示 / 怎么打分"留给消费方。因此：
  - 想加一个 UI 展示项 → 改 chat_ui.py，一般不用动这里（前提是事件里已带该字段）；
  - 想加一个新阶段 → 在这里 yield 一个新事件类型，消费方按需要处理（不认识就忽略）。

事件清单（`type` 取值与主要载荷）：
  start         question
  cache_hit     mode(exact/preset) / similarity / matched_question / answer / sources
  route         route(rag/direct) / rewrite(改写策略) / elapsed_s
  rewrite       method / label(中文名) / rewritten(list[str])
  retrieve      method / label / rewritten / filters / filter_expr / filter_fallback / elapsed_s /
                direct_vector_hits / direct_keyword_hits / rewrite_vector_hits /
                rewrite_keyword_hits / merged_count
  rerank        kept(过了阈值的记录) / merged_count / top_k / relevance_p / elapsed_s
  thinking      text（思考模型的推理内容，整段给出）
  token         text（**回答正文的唯一出口**，缓存命中与保守回复也走它）
  llm_done      answer —— ⚠ 内部事件，run_events 自己消费掉，不会流给调用方
  no_evidence   （无载荷）召回全部低于阈值：走保守回复且不调用 LLM
  error         message（已翻译成中文的排障提示，不是异常原文）
  done          answer / sources / cache_hit / route / conservative
                （⚠ 只有缓存命中路径的 done 带 rewrite 字段，另三条路径没有，见 run_events）

════════════════════════════════════════════════════════════════
读这个文件必须知道的四个坑（代码里的处理都是为它们写的）
════════════════════════════════════════════════════════════════
1. ★ 回答正文要用 ocr_text，不是 semantic_text：semantic_text 是"给向量检索用的文本"，
   ocr_text 才是交给大模型的完整证据。见 _format_record 的注释。
2. ★ 过滤零命中必须回退：抽取出的 route 是中文站名，而库里 route 存的是拼音/英文
   （Hefei-Wulumuqi 之类），这条条件永不成立。"先只丢 route 条件重试、再全部丢掉"
   是**有意为之**，见 retrieve_union 与 pipeline/filters.py 的模块 docstring。
3. ★ 决策门控阈值（settings.rerank.relevance_p，本机 .env 里是 0.22）是**跟着 reranker
   模型走的**：换 reranker 就必须用 script/calibrate_rerank.py 重标。照抄别的模型的值
   会把正确答案整条丢掉（旧 Qwen3-Reranker 时代是 0.65）。
4. ★ 路由调用必须显式关思考：deepseek-flash 这类思考模型会把 max_tokens 先花在思考上，
   短任务的正文会恒为空，路由静默退化成"全走 RAG + 直接检索"。该逻辑在
   llm/chat.py::route_query（那边有 ROUTER_MAX_TOKENS 与 enable_thinking=False）。

模块内的分层：本文件只做"编排" —— 缓存、路由、改写、召回、重排、生成各自都在别的模块里
（core/cache.py、llm/chat.py、retrieval/*）。这里不直接调 Milvus / rerank / LLM 客户端，
只调那些模块的入口函数，所以**本文件几乎没有可测试的算法**，测试重点在别处。
"""

import asyncio
import time
from collections.abc import AsyncIterator

# 只把 MilvusException 单独 import 出来：run_events 要把它和"其它异常"分成两个分支，
# 前者给"去起数据库"的提示，后者给通用提示（见召回阶段的 except）。
from pymilvus.exceptions import MilvusException

from config import settings
from core.cache import AnswerCache
from core.logger import logger
from core.prompts import NO_EVIDENCE_REPLY, REWRITE_LABELS
# llm.chat 的六个入口，按"要不要检索 × 流式/非流式"两两对应：
#   route_query / rewrite_query —— Agent 路由与 query 改写（不产生回答正文）；
#   stream_answer / generate_answer —— 带票据上下文的生成（RAG 路径）；
#   stream_direct_answer / generate_direct_answer —— 不带上下文的直答（DIRECT 路径）。
from llm.chat import (
    generate_answer,
    generate_direct_answer,
    rewrite_query,
    route_query,
    stream_answer,
    stream_direct_answer,
)
# 过滤：filters 只负责"把问题翻译成条件"，条件怎么用由本文件的召回阶段决定。
from pipeline.filters import build_milvus_filter, extract_ticket_filters
# 两条召回路径：BM25（关键词）+ 向量（语义）。本文件负责把它们并起来。
from retrieval.keyword_retrieval import keyword_search
from retrieval.rerank import rerank
from retrieval.vector_retrieval import vector_search


def _error_reply(reason: str = "generic") -> str:
    """把底层异常翻译成"给用户看的中文提示 + 排查指引"。

    为什么返回字符串而不是抛异常：错误要通过 **error 事件**流给前端（已经渲染成 Step 的
    中间过程不会因为中途失败而白费），而事件流里能传的是文本，不是异常对象。
    调用方负责 logger.error 记录真实异常，这里只负责"说人话"。

    reason 取值与对应场景：
      "milvus"  → 向量库连不上（召回阶段的 MilvusException）
      "rerank"  → 重排服务不可用（SiliconFlow 的 rerank 接口）
      "llm"     → 大模型不可用（直答或生成阶段的异常）
      其它(默认) → 兜底文案，提示去看 logs 目录

    ⚠ 下面 milvus 文案里那条 PowerShell 命令的路径（F:\\DockerDesktopData\\start-dbs.ps1）
    是**本机开发环境**写死的起库脚本，换机器要改；改文案不会影响任何逻辑。
    """
    if reason == "milvus":
        return (
            "⚠️ 知识库服务(Milvus)当前不可用,无法完成检索。"
            "请在数据库目录下执行 `powershell -File F:\\DockerDesktopData\\start-dbs.ps1` 启动服务后重试。"
        )
    if reason == "rerank":
        return "⚠️ 重排序服务(Reranker)暂时不可用,请稍后重试,或检查网络与 API Key 配置。"
    if reason == "llm":
        return "⚠️ 大模型服务暂时不可用,请稍后重试,或检查网络与 API Key 配置。"
    return "⚠️ 服务内部出现异常,请稍后重试,详情请查看 logs 目录下的日志。"


def _chunk_text(text: str, size: int = 6) -> list[str]:
    """把一整段文本切成固定长度的小块，用来冒充"流式输出"。

    为什么需要它：缓存命中、保守回复（no_evidence）、以及非流式生成这三条路径，
    答案在手里是**完整的一整段**，但前端的渲染逻辑只认 token 事件 ——
    如果不切块，页面上就会出现"什么都没有 → 突然整段蹦出来"的观感，与流式路径不一致。
    切成 6 个字符一块后，三条路径和真流式在 UI 上表现一致。

    注意 size 的单位是**字符**（不是词/字节）；这里没有 sleep，所以不是真的逐字打字效果，
    只是让事件粒度统一。
    """
    return [text[i : i + size] for i in range(0, len(text), size)]


def _format_date(value) -> str:
    """把库里的 date_int(YYYYMMDD 整数) 格式化成 2025-01-01（缺值显示"无"）。

    date_int 是**整数编码**而不是时间戳/字符串，所以用 `//` 取位：
      value // 10000      → 年
      value // 100 % 100  → 月
      value % 100         → 日
    判据用真值（`if not value`），所以 0 与 None 一样显示"无" —— 这不是偷懒：
    入库侧（data_process/tick_extract.py 的"取值约定"第 1 条）规定"抽不到就是 None，
    绝不落成 0 / ''"（它自己声明了两个例外：发票的 route 会是 ""、火车票的 counterparty
    保持 None —— 但都不涉及 date_int），且实测库内 300 行的 date_int 非 null 即合法
    YYYYMMDD、没有一条是 0。所以这里的真值判据不会把"真实日期"误显示成"无"。
    """
    if not value:
        return "无"
    return f"{value // 10000:04d}-{value // 100 % 100:02d}-{value % 100:02d}"


def _format_amount(value) -> str:
    """把库里的 amount_fen(**单位分**) 格式化成 1234.56元（缺值显示"无"）。

    ⚠ 判据是 `is None` 而不是真值：金额 0 是有意义的取值（零元发票），
    必须显示成"0.00元"而不是"无"。与 _format_date 的真值判据**故意不同**。
    """
    if value is None:
        return "无"
    return f"{value / 100:.2f}元"


def _format_record(index: int, row: dict) -> str:
    """把一条召回记录拼成"元数据行 + 内容行"两行，喂给大模型。

    格式与提示词是配套的：core/prompts.py 的 RAG_ANSWER_PROMPT 明确写了
    "票据资料(每条由元数据行和内容行组成)"；而 SYSTEM_PROMPT 要求模型"标注引用的
    资料编号，如 [1]" —— 所以这里的 index 是 **1 起算**的，和提示词里的编号口径一致。

    所有字段都用 `or '无'` 兜住 null 与空串：元数据行必须**逐条各占一行、字段齐全**，
    缺字段会让模型误以为这条记录没有该信息，甚至对不齐后面的行。
    """
    metadata = (
        f"[{index}] 类型:{row.get('ticket_type') or '无'}"
        f" | 票号:{row.get('ticket_no') or '无'}"
        f" | 相关人/买方:{row.get('person') or '无'}"
        f" | 日期:{_format_date(row.get('date_int'))}"
        f" | 金额:{_format_amount(row.get('amount_fen'))}"
        f" | 路线:{row.get('route') or '无'}"
        f" | 交易对方:{row.get('counterparty') or '无'}"
    )
    # 证据正文优先用 ocr_text：semantic_text 是给向量检索用的文本，
    # 课案口径里 ocr_text 才是交给大模型的完整证据（Agent 链路与样本生成脚本也用的是它）。
    # 回退顺序 ocr_text → semantic_text → "无"：老数据里 ocr_text 可能为空，
    # 那时用 semantic_text 至少还能给模型一点内容；两者都没有才写"无"。
    # 注意：semantic_text 只是"清洗后的 OCR 全文/摘要"，它**不是**给模型看的最终证据，
    # 一旦课案口径把 semantic_text 换成结构化摘要模板，这条兜底会静默丢掉证据。
    evidence_text = row.get("ocr_text") or row.get("semantic_text") or "无"
    return f"{metadata}\n内容:{evidence_text}"


def _build_context(rows: list[dict]) -> str:
    """把重排后的记录列表拼成完整上下文（记录之间空一行分隔）。

    空行是**记录级**的分隔符（每条记录内部只有单个换行）：一个空行明确告诉模型
    "上一段属于上面那条票据、下面这一段是另一条"，多条记录连成一大片时，
    模型很容易把上一条的"内容"归到下一条的编号上，引用就会张冠李戴。
    这里的条数由调用方传进来的 rows 决定（上游是 rerank 的 top_k，本机 8 条），
    条数直接影响提示词长度与成本。
    """
    return "\n\n".join(_format_record(i + 1, r) for i, r in enumerate(rows))


def _to_source(row: dict) -> dict:
    """挑出给前端/缓存用的"来源摘要"字段（**不含证据正文**）。

    刻意只给这几项：sources 会随 done 事件发给前端、还会被写进 Redis 的精确缓存
    （core/cache.py::store），带上 ocr_text 全文会让缓存体积成倍增长，
    而且页面上的"参考票据"只需要文件名 + 类型 + 票号就够定位。
    要展示分数/更多字段用 _view_row（那条路不写缓存）。
    """
    return {
        "id": row.get("id"),
        "source_file": row.get("source_file"),
        "ticket_type": row.get("ticket_type"),
        "ticket_no": row.get("ticket_no"),
        "rerank_score": row.get("rerank_score"),
    }


def _view_row(row: dict, score_key: str) -> dict:
    """挑出给 UI 表格用的字段（元数据 + 一个分数），同样不含证据正文。

    score_key 由调用方指定："vector_score" / "keyword_score" / "rerank_score" ——
    因为不同召回路径的分数存在不同字段里，而表格只有一列"分数"。
    消费方是 app/chat_ui.py::_rows_table。
    """
    return {
        "id": row.get("id"),
        "source_file": row.get("source_file"),
        "ticket_type": row.get("ticket_type"),
        "ticket_no": row.get("ticket_no"),
        "person": row.get("person"),
        "date_int": row.get("date_int"),
        "amount_fen": row.get("amount_fen"),
        "route": row.get("route"),
        "counterparty": row.get("counterparty"),
        "score": row.get(score_key),
    }


class RAGPipeline:
    """将缓存、路由、改写、检索与生成组合为完整的问答流程

    这个类刻意保持"薄"：除了一个 AnswerCache（缓存是**有状态**的，必须挂在实例上），
    其余部件全都是无状态的模块级函数，所以本类没有可配置项、也没有连接池要管。
    多进程/多实例部署时唯一需要留意的就是缓存（Redis）——它本来就是跨进程共享的。

    实例化点：app/main.py 与 app/chat_ui.py 各建一个（同进程里两个实例，
    各自的 AnswerCache 独立，但 BM25 索引是模块级 lru_cache，全进程只加载一份）。
    """

    def __init__(self) -> None:
        # AnswerCache.__init__ 里会调 get_redis_client() 建一个 redis.Redis 客户端对象
        # （core/redis_client.py 内部做了单例），**构造时不会真的连 Redis**，
        # 真正可能失败的是 lookup/store —— 那边已经用 try/except 降级成"没有缓存"。
        self.cache = AnswerCache()

    @staticmethod
    def _recall(queries: list[str], filters: dict, filter_expr: str) -> dict:
        """按给定过滤条件对多条 query 做双路召回并按 id 合并去重。

        queries[0] 是用户原问题(直接检索),其余是改写后的检索文本。
        过滤条件来自原问题,对直接检索与改写检索一视同仁。

        设计要点（"并集"而不是"替换"）：
        - 课案里 query 改写是**替换**检索文本，本项目是"原问题 + 改写 query"的**并集** ——
          改写可能召回到原问题找不到的东西，但原问题本身永远是最可靠的基线，
          所以 queries[0] 单独走一遍两路召回，改写 query 再各走一遍，最后合并。
        - 四路的 top_n 用同一个值（settings.retrieval.top_n，本机 10），
          这样四条路的召回深度可比；如果是改写出多条 query，改写侧的总量会成倍增长
          （每条 query 各取 top_n），这也是为什么下面要用 id 去重。

        参数归一化那两行的原因：两套检索接口对"空"的容忍度不同 ——
        Milvus 的 search 习惯收 None 表示不过滤，BM25 侧则把空 dict 当"无过滤"，
        于是统一在这里把空串/空 dict 收敛成 None，调用方（retrieve_union 的回退路径）
        就可以直接传 {} 和 "" 表示"不过滤"。
        """
        vec_expr = filter_expr or None
        kw_filters = filters or None
        # 一次读取、四路共用：避免每条 query 各读一次配置（也保证四条路深度一致）。
        top_n = settings.retrieval.top_n

        # 直接检索：原问题分别走向量（语义）与 BM25（关键词）两条路。
        vec_hits = vector_search(queries[0], top_n=top_n, filter_expr=vec_expr)
        kw_hits = keyword_search(queries[0], top_n=top_n, filters=kw_filters)

        # 改写检索：改写 query 同样各走两条路，结果**累加**（不去重，留到下面统一去重，
        # 这样事件里能分别看到"哪条 query 召回了什么"，便于评估改写质量）。
        rw_vec: list[dict] = []
        rw_kw: list[dict] = []
        for query in queries[1:]:
            rw_vec.extend(vector_search(query, top_n=top_n, filter_expr=vec_expr))
            rw_kw.extend(keyword_search(query, top_n=top_n, filters=kw_filters))

        # 合并去重：按 id 去重，**先到先得** —— 拼接顺序（直接向量 → 直接 BM25 →
        # 改写向量 → 改写 BM25）就是优先级顺序：同一张票据被多条 query/多条路命中时，
        # 保留最先出现的那份记录。
        # 由此带来一个值得记住的副作用：去重后的记录**只带一个分数字段** ——
        # 向量路径来的有 vector_score、BM25 来的有 keyword_score，不会两者都有。
        # 所以下游（_view_row / 前端表格）必须按来源传对应的 score_key。
        merged: list[dict] = []
        seen: set[str] = set()
        for row in vec_hits + kw_hits + rw_vec + rw_kw:
            # 这里用 row["id"] 而不是 row.get("id")：两条召回路径都会返回 id
            # （vector_search 用 hit["id"]、BM25 侧 output_fields 含 id），
            # 万一哪天不返回了，直接 KeyError 暴露出来，好过静默漏掉一条记录。
            if row["id"] not in seen:
                seen.add(row["id"])
                merged.append(row)

        # 四路明细都返回（而不是只给 merged）：事件流要把它们分别渲染成四张表，
        # 评估脚本也要能分别算各路的召回率。
        return {
            "direct_vector": vec_hits,
            "direct_keyword": kw_hits,
            "rewrite_vector": rw_vec,
            "rewrite_keyword": rw_kw,
            "merged": merged,
        }

    def retrieve_union(self, question: str, rewritten: list[str]) -> dict:
        """直接检索 + 改写检索双路召回,按 id 合并去重,返回本轮召回明细。

        返回字典字段:
          direct_vector / direct_keyword / rewrite_vector / rewrite_keyword / merged
          filters:        从原问题抽出的结构化过滤条件
          filter_expr:    对应的 Milvus 标量过滤表达式(空串表示不过滤)
          filter_fallback: 是否因过滤后零命中而回退到不过滤检索

        回退说明:过滤条件由规则抽取,可能误判(例如问到的年份不在库中),
        硬过滤会直接把召回清空并触发"无可靠依据"回复;因此在带过滤零命中时
        退回一次不过滤检索,并把回退标记透出给事件流,便于评估时区分
        "确实没有资料"和"过滤条件抽取错了"。

        ★ 三级回退梯度（这是本方法最值得记住的结构）：
            ①带全部过滤 →（零命中）②只摘掉 route 条件 →（仍零命中）③完全不过滤
        之所以要第 ② 级：route 条件在当前库里**永不成立**（抽取出中文站名，
        库里存的是拼音/英文，见 filters.py 模块 docstring）。如果一零命中就直接跳到 ③，
        person/ticket_type/日期 这些本来完全正确的硬过滤会跟着一起被丢掉，
        召回面被无谓放大、精确性问题被掩盖成"召回还不错"。
        只有第 ② 级也救不回来，才说明零命中与 route 无关，此时全丢才算合理。
        """
        # 过滤条件只从**原问题**抽（不抽改写 query）：改写是模型生成的检索文本，
        # 拿它去抽结构化条件会把模型幻觉当成事实。这个选择与 _recall 里"过滤对四路一视同仁"配套。
        filters = extract_ticket_filters(question)
        filter_expr = build_milvus_filter(filters)
        # queries[0] 是原问题（直接检索），其余是改写 query —— 顺序是 _recall 的约定。
        queries = [question, *rewritten]

        recall = self._recall(queries, filters, filter_expr)
        # fallback 只要发生过**任何一次**放宽就置 True（评估侧据此把这条样本标记为
        # "过滤被回退"，避免把它算进"过滤字段准确率"里）。
        fallback = False
        # 只有"带了过滤条件"才有回退可言：filters 为空说明本来就没过滤，
        # 零命中就是真的没资料，不需要也不能再退。
        if not recall["merged"] and filters:
            # route 条件单独先丢一次：库里 route 存的是拼音/英文站名（Hefei-Wulumuqi、
            # AKESUJICHANG-SHOUDUJICHANG），而抽取出来的是中文 —— 这条条件**永不成立**，
            # 不单独摘掉它就会把 person/ticket_type/日期 这些本来正确的硬过滤一起拖下水。
            # 用 startswith("route") 是为了同时匹配 route_from 和 route_to 两个键。
            route_keys = [key for key in filters if key.startswith("route")]
            if route_keys:
                relaxed = {key: value for key, value in filters.items() if key not in route_keys}
                relaxed_expr = build_milvus_filter(relaxed)
                # 摘掉 route 后可能一条条件都不剩（问句里只提到了路线）——那时
                # relaxed_expr 是空串，等于"不过滤"，与第 ③ 级完全重合，
                # 所以这里跳过重试，直接落到下面的全丢分支，省一次无谓的检索。
                if relaxed_expr:
                    # 注意日志里打的是**最初**的 filter_expr（不是 relaxed_expr）：
                    # 这样从日志就能看出"原本被哪套条件卡死了"，便于定位是哪条抽取规则的问题。
                    logger.warning(f"[召回] 过滤条件零命中({filter_expr}),先只丢 route 条件重试")
                    recall = self._recall(queries, relaxed, relaxed_expr)
                    fallback = True
            if not recall["merged"]:
                # 第 ③ 级：连 person/日期/金额 都不要了。这是"宁可多召回，让重排去筛"的取舍 ——
                # 代价是召回面变大、重排压力变大，所以每次回退都记 WARNING。
                logger.warning(f"[召回] 过滤条件零命中({filter_expr}),回退到不过滤检索")
                recall = self._recall(queries, {}, "")
                fallback = True

        # `{**recall, ...}`：把最后一次 _recall 的四路明细与合并结果透传出去，
        # 再补上过滤相关的三个字段。注意返回的 filter_expr/filters 始终是**最初**那一套，
        # 即使中途回退过也不变 —— 前端与评估看到的都是"原本打算用什么条件"，
        # 实际有没有用它由 filter_fallback 表示。
        return {**recall, "filters": filters, "filter_expr": filter_expr, "filter_fallback": fallback}

    async def _stream_llm(self, agen) -> AsyncIterator[dict]:
        """包装 LLM 流式输出:先汇总思考内容(thinking),再逐段输出回答(token)

        上游契约（见 llm/chat.py::_iter_deltas 与两个 stream_* 函数）：逐块产出
        `{"reasoning": ...}`（思考内容）或 `{"content": ...}`（回答正文），两者互斥；
        也就是说**每个 part 必有其一**，所以这里可以先判 reasoning、否则直接取 content。

        三个为什么：
        1. 为什么要把思考攒起来、等第一个正文块到达时才一次性 yield：
           UI 的 cl.Step 是整块渲染的，而且思考块必须排在正文**之前** ——
           上游会把 reasoning 与 content 交错吐出来，不缓冲就会出现"答案出到一半再插一段思考"。
        2. 为什么循环结束后还要再判一次 thinking（第二个 flush 点）：
           思考模型可能把 token 预算全花在思考上、一个正文块都不给 ——
           没有这个兜底，那段思考内容会被静默丢弃，页面上只剩一条空答案，无从排查。
        3. 为什么最后要给 llm_done：调用方需要用**完整正文**去写缓存与记日志，
           而 token 事件是分片的。llm_done 是**内部事件**，run_events 会自己消费掉
           （只取 answer），不会流给前端。
        """
        thinking: list[str] = []
        thinking_flushed = False
        # 把分片的 content 再拼回完整字符串，供 llm_done 携带。
        collected: list[str] = []
        async for part in agen:
            if "reasoning" in part:
                # 思考内容只攒不发 —— 等正文开始前统一 flush（理由见 docstring 第 1 条）。
                thinking.append(part["reasoning"])
                continue
            # 第一个正文块（或第一段正文）到达：先把攒下的思考一次性发出去。
            # thinking 为空的判断是必须的：模型不思考时不该凭空发一个空的 thinking 事件。
            if thinking and not thinking_flushed:
                thinking_flushed = True
                yield {"type": "thinking", "text": "".join(thinking)}
            yield {"type": "token", "text": part["content"]}
            collected.append(part["content"])
        # 兜底 flush：全程只有思考、没有正文（见 docstring 第 2 条）。
        if thinking and not thinking_flushed:
            yield {"type": "thinking", "text": "".join(thinking)}
        # 收尾事件：即使一句正文都没有也会发出（answer 为空串），
        # 调用方据此判断"这一路跑完了"，并拿到用于缓存/日志的完整答案。
        yield {"type": "llm_done", "answer": "".join(collected)}

    async def run_events(self, question: str, stream: bool = True, use_cache: bool = True) -> AsyncIterator[dict]:
        """执行完整流程并逐事件产出(供可视化/UI 消费)。

        三个参数的语义：
          stream    True 时 LLM 走流式接口（正文字段分片产出，边生成边发 token）；
                    False 时一次性生成完，再把整段正文作为**一个** token 事件发出。
                    两条路都产出 token 事件（消费方不必关心这个开关）；
                    thinking 事件两条路也都有，区别只是"边生成边给"还是"生成完一次性给"。
          use_cache 见下（评估专用）。
          question  会被 strip() 后再走链路。

        use_cache=False 用于**评估**:缓存命中会绕过路由/召回/重排三段,
        那些阶段指标会因为没有数据而一起被打成 0(实测一条 preset 命中让
        route/filter/recall/rerank 四项同时从 1.0 掉到 0.875)。

        ⚠ 注意 use_cache=False **同时关掉读写**：末尾各条路径的 `self.cache.store(...)`
        会被跳过 —— 否则评估跑出来的答案会写进 Redis 精确缓存（TTL 10 分钟），
        这段时间内真人问同一句会直接拿到评估那一跑生成的答案。
        （store 内部对"空问题 / 空答案"会直接 return，这里额外按 use_cache 跳过调用。）

        控制流总览（四条互斥的出口，每条都保证最后发一个 done 事件，错误出口除外）：
          ①缓存命中 → cache_hit + 分片 token + done，直接 return（不路由、不检索）
          ②路由为 direct → route + 直答 + done，return
          ③RAG 且重排后无记录 → rerank + no_evidence + 保守回复 + done，return
          ④RAG 且重排后有记录 → rerank + 生成 + done（正常路径）
          另有**四处异常场景、共 5 个 except 分支**（召回阶段为了区分 Milvus 与其它
          故障而写了两个）：直答 LLM 失败 / 召回失败(MilvusException) / 召回失败(其它) /
          重排失败 / 生成 LLM 失败，都是 `yield error` 后 return
          （**不发 done**，消费方要能接受这种情况）。
        """
        # 统一去掉首尾空白：既让缓存键归一化（cache._exact_key 内部还会再压一次空白），
        # 也避免把"  王强的火车票  "这种带空格的问法当成另一个问题去抽过滤条件。
        question = question.strip()
        # start 永远是事件流的第一条：消费方据此确认链路真的启动了（"已经收到问题"），
        # 也让时间线有一个明确的起点。
        yield {"type": "start", "question": question}

        t0 = time.perf_counter()
        # use_cache=False 时不查缓存 —— 这是评估能拿到分阶段指标的前提（见 docstring）。
        cached = self.cache.lookup(question) if use_cache else None
        if not use_cache:
            logger.info("[缓存] 评估模式,跳过缓存")
        else:
            # 日志用三元表达式区分"命中/未命中"两种文案，但**耗时都是 t0 起算**的，
            # 所以这一行反映的是缓存查询本身的耗时。
            logger.info(f"[缓存] 未命中,耗时 {time.perf_counter() - t0:.2f}s" if cached is None
                        else f"[缓存] 命中({cached['cache_hit']}),耗时 {time.perf_counter() - t0:.2f}s")
        if cached:
            # cached 的字段来自 core/cache.py：精确层存的是 question/answer/sources，
            # 预设相似度层存的是 question/answer/similarity；两层的 cache_hit 由
            # AnswerCache.lookup 统一补上（"exact" / "preset"）。
            # ⚠ 这里 `cached["cache_hit"]` 用下标（不是 .get）—— lookup 的两条返回路径
            # 都保证写了这个键，一旦哪天新增第三层忘了写，这里会 KeyError（比静默变成 None 好排查）。
            yield {
                "type": "cache_hit",
                "mode": cached["cache_hit"],
                "similarity": cached.get("similarity"),
                "matched_question": cached.get("question"),
                "answer": cached.get("answer", ""),
                "sources": cached.get("sources", []),
            }
            # 把缓存里的整段答案切块当 token 流发出去：让"缓存命中"在 UI 上的呈现
            # 与正常回答一致（否则会从空白直接跳成整段）。见 _chunk_text。
            for piece in _chunk_text(cached.get("answer", "")):
                yield {"type": "token", "text": piece}
            # ⚠ done 的 payload 与另外三条路径**不同**：这条带 "rewrite": None，
            # 而直答/保守/RAG 三条路径的 done 没有 rewrite 键。消费 done 事件时
            # 用 .get("rewrite") 才安全。（route 字段的含义：None = 没走过路由，
            # 前端据此知道不该显示 ②~⑤ 那几个 Step。）
            yield {
                "type": "done",
                "answer": cached.get("answer", ""),
                "sources": cached.get("sources", []),
                "cache_hit": cached["cache_hit"],
                "route": None,
                "rewrite": None,
                "conservative": False,
            }
            # 缓存命中是"短路出口"：整个函数到此结束，路由/召回/重排/生成一概不做。
            return

        t1 = time.perf_counter()
        # 路由一次调用同时回答两件事：要不要检索（need_rag）、用哪种改写策略（method）。
        # 它是**同步**调用（llm/chat.py 用的是同步 OpenAI 客户端），所以这里会阻塞事件循环；
        # 本项目是单进程本地应用，量级可以接受。（要并发就必须改成异步客户端。）
        need_rag, method = route_query(question)
        # 事件与日志里用字符串 "rag"/"direct"，而 route_query 返回的是布尔 —— 转换只在这一行，
        # 下游全部按字符串判断，避免两套表示法混用。
        route = "rag" if need_rag else "direct"
        # REWRITE_LABELS[method] 用下标：method 由 route_query 保证落在四个已知值里
        # （hyde/subquery/backtrack/direct），拿不到就是上游契约破了，直接报错更醒目。
        logger.info(f"[路由] {route} | 改写策略={REWRITE_LABELS[method]} | 耗时 {time.perf_counter() - t1:.2f}s")
        # elapsed_s 保留 3 位小数：分阶段评估要聚合各阶段延迟，太粗看不出差异（本地链路都是毫秒级）。
        yield {
            "type": "route",
            "route": route,
            "rewrite": method,
            "elapsed_s": round(time.perf_counter() - t1, 3),
        }

        # ── 出口 ②：DIRECT（不检索，直接让 LLM 答）──
        # 典型输入是闲聊/常识/数学题；"涉及金额、票号、人员"这类问题路由会强制走 RAG。
        if route == "direct":
            t2 = time.perf_counter()
            try:
                answer = ""
                if stream:
                    # 流式：_stream_llm 会把 thinking 与 token 分开发出。
                    # 注意这里**只**转发非 llm_done 的事件 —— llm_done 是内部记号，
                    # 它的 payload 只用来给 answer 赋值（供日志与写缓存用）。
                    async for ev in self._stream_llm(stream_direct_answer(question)):
                        if ev["type"] == "llm_done":
                            answer = ev["answer"]
                        else:
                            yield ev
                else:
                    # 非流式：一次性拿到 (回答, 思考内容)。
                    # 思考内容（reasoning）单独作为 thinking 事件发出；正文作为一个 token 事件。
                    answer, reasoning = generate_direct_answer(question)
                    if reasoning:
                        yield {"type": "thinking", "text": reasoning}
                    yield {"type": "token", "text": answer}
            except Exception as exc:
                # 直答失败：给用户一句中文提示就结束。此时 answer 仍是空串，
                # 而 cache.store 对空答案本来就写不进去（core/cache.py::store 开头就 return），
                # 所以这里直接 return —— 不会留下"服务不可用"被缓存 10 分钟的坑。
                logger.error(f"[直接回答] LLM 调用失败: {exc}")
                yield {"type": "error", "message": _error_reply("llm")}
                return
            logger.info(f"[直接回答] 完成,长度 {len(answer)} 字符 | 耗时 {time.perf_counter() - t2:.2f}s")
            # 直答也进缓存（sources 为空列表）：同一个闲聊问题 10 分钟内再问就不用再调 LLM。
            if use_cache:
                self.cache.store(question, answer, [])
            # ⚠ 这条 done 没有 "rewrite" 键（与缓存命中那条不同），消费方要用 .get。
            yield {"type": "done", "answer": answer, "sources": [], "cache_hit": None, "route": "direct", "conservative": False}
            return

        # ── Query 改写（增强项，不是必需项）──
        # method == "direct" 表示路由认为原问题本身就能检索，跳过改写（省一次 LLM 调用）。
        try:
            rewritten = rewrite_query(question, method) if method != "direct" else []
        except Exception as exc:
            # 改写失败**不中断链路**：降级成"只用原问题检索"。这是有意的取舍 ——
            # 改写是提高召回的手段，缺了它仍然是一条完整可用的 RAG 链路；
            # 而如果在这里 return，一次改写超时就会让整个问答失败，代价不成比例。
            logger.warning(f"[改写] 失败,回退仅直接检索: {exc}")
            rewritten = []
        if rewritten:
            # 只有真的生成了改写 query 才发 rewrite 事件（前端据此决定要不要显示"③ Query 改写"）。
            # rewritten 是 list[str]：subquery 策略下最多 3 条，其余策略 1 条。
            logger.info(f"[改写] 策略={REWRITE_LABELS[method]},生成 {len(rewritten)} 条检索 query: {rewritten}")
            yield {"type": "rewrite", "method": method, "label": REWRITE_LABELS[method], "rewritten": rewritten}
        else:
            # 走到这里有两种可能（direct 策略 / 模型没给内容 / 调用失败），日志统称"未生成"，
            # 真实原因看上一行是 INFO 还是 WARNING。
            logger.info("[改写] 未生成改写 query,仅使用直接检索")

        # ── 召回 ──
        # 两个 except 的**顺序不能反**：MilvusException 必须排在 Exception 前面，
        # 否则会被通用分支吃掉、用户看到的是"内部异常"而不是"去起数据库"。
        try:
            t3 = time.perf_counter()
            recall = self.retrieve_union(question, rewritten)
        except MilvusException as exc:
            # 向量库连不上：给带"起库命令"的专门提示（见 _error_reply）。
            logger.error(f"[召回] Milvus 不可用: {exc}")
            yield {"type": "error", "message": _error_reply("milvus")}
            return
        except Exception as exc:
            # 其他召回异常（Embedding 接口、BM25 语料加载、表达式语法错…）：通用提示 + 查日志。
            logger.error(f"[召回] 检索失败: {exc}")
            yield {"type": "error", "message": _error_reply("generic")}
            return
        # 把 recall 字典拆成局部变量，只为下面日志与事件写起来更短；
        # 四路明细到这一步仍然各自独立（合并去重发生在 _recall 内部，结果在 merged 里）。
        vec_hits = recall["direct_vector"]
        kw_hits = recall["direct_keyword"]
        rw_vec = recall["rewrite_vector"]
        rw_kw = recall["rewrite_keyword"]
        merged = recall["merged"]
        # ⚠ 读日志时注意这一行的措辞：filter_fallback 覆盖**两级**回退（只丢 route / 全丢），
        # 但文案一律写"零命中已回退不过滤" —— 只丢 route 那次也会打印这句。
        # 要区分到底退到哪一级，看它上面有没有那条"先只丢 route 条件重试"的 WARNING。
        logger.info(
            f"[召回] 过滤条件: {recall['filter_expr'] or '无'}"
            f"{'(零命中已回退不过滤)' if recall['filter_fallback'] else ''};"
            f"直接检索:向量 {len(vec_hits)}/BM25 {len(kw_hits)};"
            f"改写检索:向量 {len(rw_vec)}/BM25 {len(rw_kw)};去重后 {len(merged)} 条"
            f" | 耗时 {time.perf_counter() - t3:.2f}s"
        )
        logger.debug(f"[召回-直接向量] {[(r['source_file'], r.get('vector_score')) for r in vec_hits]}")
        logger.debug(f"[召回-直接BM25] {[(r['source_file'], r.get('keyword_score')) for r in kw_hits]}")
        logger.debug(f"[召回-改写向量] {[(r['source_file'], r.get('vector_score')) for r in rw_vec]}")
        logger.debug(f"[召回-改写BM25] {[(r['source_file'], r.get('keyword_score')) for r in rw_kw]}")
        yield {
            "type": "retrieve",
            "method": method,
            "label": REWRITE_LABELS[method],
            "rewritten": rewritten,
            "filters": recall["filters"],
            "filter_expr": recall["filter_expr"],
            "filter_fallback": recall["filter_fallback"],
            "elapsed_s": round(time.perf_counter() - t3, 3),
            "direct_vector_hits": [_view_row(r, "vector_score") for r in vec_hits],
            "direct_keyword_hits": [_view_row(r, "keyword_score") for r in kw_hits],
            "rewrite_vector_hits": [_view_row(r, "vector_score") for r in rw_vec],
            "rewrite_keyword_hits": [_view_row(r, "keyword_score") for r in rw_kw],
            "merged_count": len(merged),
        }

        try:
            t4 = time.perf_counter()
            top_rows = rerank(question, merged, top_k=settings.rerank.top_k)
        except Exception as exc:
            # 重排服务挂掉时**不降级**成"直接用向量分取前 N 条"：没有重排的候选里
            # 混杂着语义相近但票据不对的记录，拿它当证据喂给模型正是"看起来有依据的胡说"。
            # 宁可明确地告诉用户服务不可用。
            logger.error(f"[重排] 失败: {exc}")
            yield {"type": "error", "message": _error_reply("rerank")}
            return
        logger.info(
            f"[重排] 候选 {len(merged)} -> 保留 {len(top_rows)} 条"
            f" (top_k={settings.rerank.top_k}, 相关度>={settings.rerank.relevance_p}) | 耗时 {time.perf_counter() - t4:.2f}s"
        )
        # top_rows 已经是"取 top_k 之后、再过相关度阈值"的结果（见 retrieval/rerank.py），
        # 所以这个事件里的 kept 条数就是**最终会喂给模型的证据条数**。
        # relevance_p 一并发给前端：页面要显示"阈值 0.22 过滤后保留 N 条"，
        # 这个数字必须来自配置（而不是前端写死），否则换 reranker 重标阈值后页面会撒谎。
        yield {
            "type": "rerank",
            "kept": [_view_row(r, "rerank_score") for r in top_rows],
            "merged_count": len(merged),
            "top_k": settings.rerank.top_k,
            "relevance_p": settings.rerank.relevance_p,
            "elapsed_s": round(time.perf_counter() - t4, 3),
        }

        # ── 出口 ③：有召回但全部低于阈值 → 保守回复 ──
        # 这是**决策门控**真正起作用的地方：宁可回答"没有可靠依据"，也不让模型在
        # 弱证据上编造。注意此时**完全不调用 LLM**（省钱、也彻底杜绝幻觉）。
        if not top_rows:
            logger.info("[生成] 召回均低于相关度阈值,走保守回复,不调用 LLM")
            # no_evidence 是个无载荷事件：前端看到它就渲染"⚠️ 无可靠依据"的 Step，
            # 评估脚本也据此区分"拒答"样本（eval_set 里有 4 条拒答题就是测这个分支）。
            yield {"type": "no_evidence"}
            # 保守回复同样切成 token 事件发出去，渲染路径与正常回答完全一致。
            for piece in _chunk_text(NO_EVIDENCE_REPLY):
                yield {"type": "token", "text": piece}
            # 保守回复也写缓存：它是"确定性的答案"，10 分钟内重复问不必再跑一遍召回+重排。
            if use_cache:
                self.cache.store(question, NO_EVIDENCE_REPLY, [])
            # ⚠ conservative=True 是这条路径的标记：前端据此**不**追加"参考票据"引用块。
            yield {"type": "done", "answer": NO_EVIDENCE_REPLY, "sources": [], "cache_hit": None, "route": "rag", "conservative": True}
            return

        # ── 出口 ④（正常路径）：拼上下文 → 生成 ──
        t5 = time.perf_counter()
        try:
            if stream:
                # 与直答分支同样的模式：转发 thinking/token，把 llm_done 留下来取完整答案。
                answer = ""
                async for ev in self._stream_llm(stream_answer(question, _build_context(top_rows))):
                    if ev["type"] == "llm_done":
                        answer = ev["answer"]
                    else:
                        yield ev
            else:
                # 非流式：一次拿到 (回答, 思考内容)。_build_context 在这里**第二次**调用
                # （第一处是 stream 分支）—— 两条分支各自构造一次，没有缓存，因为只跑其中一条。
                answer, reasoning = generate_answer(question, _build_context(top_rows))
                if reasoning:
                    yield {"type": "thinking", "text": reasoning}
                yield {"type": "token", "text": answer}
        except Exception as exc:
            # 生成失败同样不写缓存（answer 可能是空串或半截答案）。
            logger.error(f"[生成] LLM 调用失败: {exc}")
            yield {"type": "error", "message": _error_reply("llm")}
            return
        logger.info(f"[生成] 完成,长度 {len(answer)} 字符 | 耗时 {time.perf_counter() - t5:.2f}s")

        # 给前端/缓存的来源摘要：注意用的是 top_rows（重排后、过阈值的那些），
        # 不是 merged —— 引用列表必须与实际喂给模型的证据一致，否则会出现"引用了没给模型看过的票"。
        sources = [_to_source(r) for r in top_rows]
        # 正常路径把 sources 一起缓存：精确命中时前端仍能展示"参考票据"。
        if use_cache:
                self.cache.store(question, answer, sources)
        # conservative=False：前端据此追加引用块。
        yield {"type": "done", "answer": answer, "sources": sources, "cache_hit": None, "route": "rag", "conservative": False}

    async def run_and_collect(self, question: str) -> dict:
        """消费事件流,组装为最终的问答结果(供非流式接口使用)"""
        # 预置全部"稳定存在"的键，让返回值的形状对调用方（HTTP 层）可预期；
        # 下面几个键是**条件性新增**的，消费方必须用 .get：
        #   similarity / matched_question → 仅缓存命中时
        #   conservative                 → 仅在收到 done 事件时（即正常跑完；中途 error 时没有）
        #   error                        → 仅在出错时
        # 正因为形状不固定，这个返回值**不是** Pydantic 模型，直接 dict 交给 FastAPI 序列化。
        result = {
            # 这里再 strip 一次只是为了与 run_events 内部处理后的口径一致
            # （HTTP 层的返回值里 question 应当是归一化后的）。
            "question": question.strip(),
            "answer": "",
            "sources": [],
            "cache_hit": None,
            "route": None,
            "rewrite": None,
        }
        # stream=False：非流式接口要的是完整答案，不需要边算边推。
        # ⚠ 这里仍然会写缓存（run_events 内部的行为），与本函数的读取无关。
        async for ev in self.run_events(question, stream=False):
            t = ev["type"]
            if t == "token":
                # 正文是**分片**给的（流式路径分 chunk、缓存/保守路径按 6 字切块），
                # 所以这里必须累加，不能赋值。
                result["answer"] += ev["text"]
            elif t == "cache_hit":
                # 缓存命中时答案在事件里一次给全，这里先整体赋值（此刻还没收到任何 token）。
                # ⚠ 随后 run_events 仍会把**同一段答案**按 6 字切块发 token 事件，那些 token 会被
                # 上面的分支**追加**到这段完整答案后面，形成一次重复；最终结果不出错，
                # 只是因为最后那条 done 事件又把 answer 整体覆盖了回去（见下面的 done 分支）。
                # 也就是说：本函数的正确性依赖"cache_hit 之后一定会来一条 done"。
                result["cache_hit"] = ev["mode"]
                result["sources"] = ev.get("sources", [])
                result["answer"] = ev.get("answer", "")
                # 相似度与匹配问题只在预设相似度命中时才有，所以按需新增键
                # （`is not None` 而不是真值：相似度 0.0 也该透出）。
                if ev.get("similarity") is not None:
                    result["similarity"] = ev["similarity"]
                if ev.get("matched_question"):
                    result["matched_question"] = ev["matched_question"]
            elif t == "route":
                # route/rewrite 的唯一来源（done 事件里没有 rewrite 键）。
                # 缓存命中路径不经过路由，所以那一路 result["rewrite"] 会保持初始的 None。
                result["route"] = ev["route"]
                result["rewrite"] = ev.get("rewrite")
            elif t == "done":
                # done 的 answer 是完整答案，**覆盖**累加结果 —— 这是有意为之：
                # 万一某条 token 事件丢失/重复，最终答案仍以 done 为准（一致性优先）。
                result["answer"] = ev["answer"]
                result["sources"] = ev["sources"]
                result["conservative"] = ev.get("conservative", False)
            elif t == "error":
                # 出错时把 ⚠️ 提示文本接到 answer 上，并打一个 error 标记。
                # ⚠ 注意 **HTTP 状态码仍然是 200**：接口层（app/main.py::chat）直接返回这个 dict，
                # 不做任何分支 —— 所以调用方必须看 `error` 字段来判断失败，不能看状态码。
                result["answer"] += ev["message"]
                result["error"] = True
        return result

    async def run_stream(self, question: str) -> AsyncIterator[str]:
        """兼容接口:仅产出回答文本的流式生成器

        只转发 token，thinking / error / no_evidence 全部被丢掉 ——
        也就是说调用方**拿不到报错信息**（出错时的表现是"没有输出"而不是异常）。
        保留它是为了兼容早期只想要文本的调用方；新代码请用 run_events。
        """
        async for ev in self.run_events(question, stream=True):
            if ev["type"] == "token":
                yield ev["text"]

    def run(self, question: str) -> dict:
        """同步便捷入口(脚本/测试用)

        ⚠ 内部用 asyncio.run 起一个**新的事件循环**：在已经有事件循环的地方调用会抛
        RuntimeError（FastAPI 的请求处理函数里、Jupyter、以及任何 async 上下文都属于这种）。
        Web 侧要用 run_and_collect（异步版本），这个同步入口只给一次性脚本与测试用。
        """
        return asyncio.run(self.run_and_collect(question))

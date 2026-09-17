"""RAG 缓存层:精确问答缓存(短 TTL) + 预设问答相似度缓存,接入 RAG 流程最前端

一句话说清两层缓存的职责:
- **精确层**:归一化后的问题字符串做 key(sha256),命中即返回。速度最快、绝无串答案,
  但换个说法就命中不了。TTL 短(10 分钟),存的是"这次问答的答案",数据会过期也无所谓;
- **相似度层**:只存 `layer=faq` 的**标准问法**的向量,用余弦相似度找最像的一条。
  能覆盖"换个说法问同一件事",但**风险极高** —— 相似不等于同一个问题。

★ 本文件最关键的设计决策:**预问答对按 `layer` 分层**
    faq    → 进相似度层(高频、稳定、有标准答案的通用问法)
    detail → **只**进精确层(「某人的某张票」这类明细问答)
为什么必须有这条:
明细问答靠相似度匹配会**答成别人的票**。项目实测:「赵凡的登机牌座位号是多少?」与
库里的「赵飞的登机牌座位号是多少?」余弦相似度 **0.9008 > 阈值 0.79** ——
只差一个字的人名就命中了,于是把 B 的票据信息答给了 A。
而且这条错误**在界面上看不出来**:预设命中的 `sources` 是空列表(不是从检索来的),
前端没有"来源票号"可核对,用户会直接相信答案。
⇒ 所以相似度层是**危险层**:必须数据里显式写了 `layer: "faq"` 才准入(白名单),
缺字段的一律按明细处理走精确层。这个"默认拒绝"的方向不能反过来。

第二个要知道的坑:**缓存键必须包含配置作用域**(见 `_exact_key`)。
换模型/换集合/改阈值后,旧缓存里的答案是"另一个模型、另一份语料"的产品,
继续返回它就会让新配置看起来没生效。把模型名、集合名、两个阈值都编进 key,
配置一变旧键自然失效,不需要手动清 Redis。**刻意不放 session history**:
同一问题第二次问必须命中,若把 history 编进 key,Chainlit 那种会把历史回传的
前端会导致每次 key 都不同、缓存永远 miss(踩过)。
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from config import settings
from core.logger import logger
from core.redis_client import get_redis_client
from core.routes import CACHE_SCOPES

# 预设问答数据文件相对本文件定位(不是 CWD),保证从任意目录启动都读同一份
PRESET_QA_PATH = Path(__file__).resolve().parent.parent / "data" / "preset_qa.json"
# 精确层 key 前缀:带前缀便于在 Redis 里按模式清理/统计(如 rag:qa:exact:*)
EXACT_KEY_PREFIX = "rag:qa:exact:"
# 相似度层的两个 Redis 键:问题文本列表 / 对应向量矩阵。
# 分成两个键是因为写的时候要一起写、读的时候要成对读;并且用 exists(...) == 2
# 来判断"是否已经写过"(见 seed_preset)
PRESET_PAIRS_KEY = "rag:preset:pairs"
PRESET_VECTORS_KEY = "rag:preset:vectors"
# 明细层的"已播种"标记:明细多达几十条,逐个 exists 检查太啰嗦,用一个标记位代替
DETAIL_SEEDED_KEY = "rag:detail:seeded"

# 预设问答分两层,由数据里的 `layer` 字段决定:
#   faq    —— 高频、稳定、有标准答案的通用问法,**进相似度层**;
#   detail —— 依赖具体票据/人员的明细问答(如"赵飞的登机牌座位号是多少?"),**只进精确层**。
# 明细问答绝不能进相似度层:不同人的同型问题 embedding 相似度极高
# (实测「赵凡的登机牌座位号是多少?」与「赵飞的登机牌座位号是多少?」相似度 0.9008,
#  高于阈值 0.79),命中就会把**别人的票**答出去,而且预设命中 sources 为空、界面看不出。
FAQ_LAYER = "faq"
DETAIL_LAYER = "detail"


def split_by_layer(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    """按 `layer` 把预问答对分成 (相似度层, 明细层)。

    缺 `layer` 字段的条目按**明细**处理 —— 相似度层是危险的那一层,要显式打标才进。

    实现刻意用「等于 faq」/「不等于 faq」这一对互补条件(而不是
    `== DETAIL_LAYER` / 别的分支),好处是**任何**没写对的值(缺字段、写成
    "details" 拼错、大小写不同)都自动落到安全的明细侧,不会因为枚举写错
    而漏进相似度层。这就是"默认拒绝"的落地方式,别改成按 DETAIL_LAYER 判断。
    """
    faq = [p for p in pairs if p.get("layer") == FAQ_LAYER]
    detail = [p for p in pairs if p.get("layer") != FAQ_LAYER]
    return faq, detail


def load_preset_pairs() -> list[dict]:
    """读取预问答对(文件不存在时返回空列表)。

    文件缺失返回 [] 而不是抛错:预设问答是**可选的数据**,
    没有它缓存层退化成"只有精确层",功能仍然完整(只是 FAQ 快速返回没了)。
    让缺数据文件这种可预期的情况变成"功能降级",避免整条问答链路起不来。

    ⚠ 但**格式错误**(JSON 坏了、不是列表)会直接抛异常 —— 这是刻意的:
    格式错误是**人为事故**,静默当空数据会让"FAQ 全都不命中了"变成一个
    没有任何报错的谜题。宁可启动时报错、修好数据再跑。
    """
    if not PRESET_QA_PATH.exists():
        return []
    return json.loads(PRESET_QA_PATH.read_text(encoding="utf-8"))


def _normalize_query(query: str) -> str:
    """查询归一化:压缩多余空白,使同一问题的不同空白写法命中同一条缓存

    `" ".join(query.strip().split())` 一次搞定三件事:剥首尾空白、把连续空白
    (空格/Tab/换行,半角全角空格都算)压成一个半角空格、再去掉首尾。

    ⚠ 归一化**只做空白**,不做标点、大小写、繁简或全半角的折叠:
    「张三的票号是多少?」与「张三的票号是多少。」是两条不同的缓存键。
    这是有意的(归一化越多、误命中的风险越大),也意味着**精确层只对
    "格式干净的重复提问"有效** —— 真正的"换个说法"靠相似度层。
    另外注意:归一化后的字符串会被直接当**文本**参与 sha256 与 Redis payload,
    不是拿原串,所以 `_exact_key` 与 `store` 必须用同一个归一化函数(否则写进去读不出来)。
    """
    return " ".join(query.strip().split())


class AnswerCache:
    """两层缓存:
    1. 精确缓存:10 分钟内相同 query 直接返回之前的回答;预设里的**明细问答**也写在这一层
       (精确命中,不做相似度匹配);
    2. 预设缓存(相似度层):query 与 `layer=faq` 的**标准问法**做 embedding 余弦相似度,
       >= 阈值直接返回预设答案。

    相似度层为什么用 **numpy 矩阵乘法**而不是 Milvus 向量库(课案的做法):
    标准问法只有个位数(本项目 5 条),几十条以内 numpy 内存计算是微秒级,
    为此起一个 collection、维护一次入库/删除流程不划算。
    代价是**全量载入内存 + 每进程一份**,问法数量涨到几百条以上时要重新评估这个取舍。

    三个 `_preset_*` 实例属性是**惰性加载的进程内缓存**(不是 Redis 缓存):
    `_preset_matrix` 为 None 表示"还没加载过"。注意它们一旦加载就**不会自动跟随
    Redis 里的数据变化** —— 重新 `seed_preset(force=True)` 之后,
    已在运行的进程仍用旧的矩阵,必须重启进程或新建 AnswerCache 实例才生效(见报告)。
    """

    def __init__(self) -> None:
        self.redis = get_redis_client()
        self._preset_questions: list[str] | None = None
        self._preset_answers: list[str] | None = None
        self._preset_matrix = None

    def lookup(self, query: str, route: str = "") -> dict | None:
        """依次尝试两层缓存,Redis 异常时降级返回 None(继续走 RAG)

        `route` 是**线路作用域**：四条 RAG 线路（基础/Agentic/GraphRAG/融合）各用各的键。
        不隔离的话，先问"基础"再切到"融合"问同一句会直接拿回基础线路的旧答案，
        看起来就是"切了线路但没生效"。默认空串 = 基础线路，与旧键完全一致
        （见 `_exact_key` 的说明，刻意不让老键失效）。

        **顺序即优先级**:精确层在前 —— 它更准(字符串完全相同),而且成本更低
        (一次 Redis GET,不需要 embedding 调用)。反过来的话,每条精确可命中的问题
        都会先花一次 embedding 去做相似度比较,既慢又多花钱。

        返回 None 表示"没命中"(调用方据此继续走 RAG),异常也返回 None:
        两者在调用方看来一样,但对缓存来说是完全不同的两件事 ——
        所以 except 里**必须记 warning**,否则 Redis 挂了会表现为"缓存莫名其妙
        全不命中",而日志里什么都没有。

        `except Exception` 的宽泛捕获在这里是**有意的**:本层的契约是
        "永不因为缓存问题阻断问答"。Redis 连接错误、JSON 解析错误、numpy 维度不匹配,
        都应该降级成 miss,而不是把用户的提问打回 500。
        代价是会吞掉真实 bug(比如 preset_qa.json 结构写错会被当成"没命中"),
        所以日志里的 warning 是唯一的线索,别把它改成 debug。

        `{**exact, "cache_hit": "exact"}` 是**不改原字典**地加一个字段:
        exact 来自 json.loads 的新对象,这里再浅拷贝一份并覆盖 cache_hit,
        避免调用方看到 `_lookup_exact` 返回对象被就地修改。
        注意相似度层返回的 cache_hit 是 "preset"(见 `_lookup_preset`),
        这两个值就是 `run_recorder` 里 `record["cache_hit"]` 的来源。

        相似度层（FAQ 预设问法）**不带 route**：它是"标准问法 → 答案"的预设表，
        与走哪条线路无关；带 route 只会让四条线路各存一份同样的向量。
        """
        try:
            exact = self._lookup_exact(query, route=route)
            if exact is not None:
                return {**exact, "cache_hit": "exact"}
            return self._lookup_preset(query)
        except Exception as exc:
            logger.warning(f"[缓存] lookup 失败,降级走 RAG: {exc}")
            return None

    def store(self, query: str, answer: str, sources: list[dict], route: str = "") -> None:
        """把完整问答结果写入精确缓存,10 分钟过期;空问题或空答案不写。

        参数 `route` 同 `lookup`：写进哪个线路的键。调用方必须成对
        ——`lookup(route=X)` 配 `store(route=X)`，否则会出现"存了但读不到"
        （表现为缓存永远 miss，不报错，很难发现）。

        两个"不写"的判断都在 try **之外**:它们是业务规则(空问题/空答案没有缓存价值),
        不是 Redis 故障。放在 try 里会让"空答案没写缓存"和"Redis 写失败"
        走到同一个日志分支,排查时分不清。

        空答案不写是**重要**的:生成失败时答案是空串,若把空答案缓存 10 分钟,
        后续同一个问题会连续 10 分钟返回空白回答 —— 一个上游抖动被放大成
        十分钟的持续故障。

        `ex=settings.redis.exact_ttl`(默认 600 秒)是刻意的短 TTL:精确层存的是
        "某次生成的答案",它依赖当时的语料与模型版本。TTL 短 ⇒ 配置/数据更新后
        最迟 10 分钟自动恢复,不用人工清 Redis。

        `ensure_ascii=False` 让中文以原文(而非 \\uXXXX 转义)存进 Redis:
        占更少空间,且用 redis-cli 直接看能看到可读内容,排障方便很多。
        """
        normalized = _normalize_query(query)
        if not normalized or not answer:
            return
        try:
            payload = json.dumps(
                {"question": normalized, "answer": answer, "sources": sources},
                ensure_ascii=False,
            )
            key = self._exact_key(normalized, route=route)
            self.redis.set(key, payload, ex=settings.redis.exact_ttl)
        except Exception as exc:
            # 写缓存失败不影响本轮回答(答案已经生成好了),只记 warning。
            # 注意这里把异常信息带进日志:Redis 的报错(认证失败/内存满/OOM policy)
            # 是定位问题的关键,不能只写一句"store 失败"
            logger.warning(f"[缓存] store 失败: {exc}")

    def seed_preset(self, force: bool = False) -> int:
        """把 `layer=faq` 的**标准问法**及其问题向量写入 Redis(相似度层),返回写入条数。

        **幂等靠 `exists(...) == 2` 判断**:两个键都在才跳过。
        用"== 2"而不是"任意一个存在"是有意的 —— 若上次写到一半崩了(只写了 pairs
        没写 vectors),这里会重写一遍,自动修复半成品状态。
        返回 0 表示"已经播过种,本次什么都没做",不要把它当成"没有 FAQ 数据"
        (FAQ 真的为空时也会返回 0)。

        `force=True` 是**换 embedding 模型后必须走的一步**:向量是模型相关的产物,
        换了 embedding 后旧向量留在 Redis 里就是错的(与库里的向量空间不同),
        相似度计算毫无意义 ⇒ 必须 force 重算。这条与 Milvus 重算向量、
        Neo4j 重算向量是同一类动作,漏一处就是静默坏数据。

        `from retrieval.embedding import embed_texts` 写在函数**内部**(延迟导入):
        模块顶层 import 会让 core.cache 依赖 retrieval.embedding → 依赖 HTTP 客户端与
        配置校验,任何只想用精确层的地方(以及大量离线单测)都会被迫拉起整条依赖链。
        这是一个"用局部 import 换依赖解耦"的常见手法(同类写法见 _lookup_preset)。

        用 `pipeline()` 一次写两个键:Redis pipeline 把多条命令合并成一次往返,
        避免出现"pairs 写成功、vectors 写失败"的中间态窗口太长。
        ⚠ 但 pipeline 默认**不是事务**(没传 transaction=True),所以严格意义上
        仍有半成品可能 —— 这正是上面 `exists(...) == 2` 要兜的场景。
        """
        if not force and self.redis.exists(PRESET_PAIRS_KEY, PRESET_VECTORS_KEY) == 2:
            return 0
        from retrieval.embedding import embed_texts

        pairs = load_preset_pairs()
        faq_pairs, detail_pairs = split_by_layer(pairs)
        # `if faq_pairs else []` 防止空列表去调 embedding 接口:
        # 有些网关对空输入会报 400(而不是返回空结果),白白让启动失败
        vectors = embed_texts([p["question"] for p in faq_pairs]) if faq_pairs else []
        pipe = self.redis.pipeline()
        pipe.set(PRESET_PAIRS_KEY, json.dumps(faq_pairs, ensure_ascii=False))
        # 向量这行**不能**加 ensure_ascii=False:json.dumps 对 float 列表本来就
        # 输出 ASCII 数字,加了没有区别,但读的时候必须用 json.loads 还原成 list[list[float]]
        pipe.set(PRESET_VECTORS_KEY, json.dumps(vectors))
        pipe.execute()
        logger.info(
            f"[缓存] 相似度层写入完成,共 {len(faq_pairs)} 条标准问法"
            f"(另有 {len(detail_pairs)} 条明细问答走精确层,见 seed_details)"
        )
        return len(faq_pairs)

    def seed_details(self, force: bool = False) -> int:
        """把明细问答写进**精确缓存层**(按归一化问题精确命中,不做相似度匹配)。

        明细问答的答案依赖具体票据,只能精确匹配:相似度匹配会把别人的票答出去。
        这里不设 TTL —— 它们是固定问答数据,不是"某次生成的答案"。

        「不设 TTL」是**有意的**(与 store 的 10 分钟形成对比):明细问答来自
        人工维护的 preset_qa.json,内容不会随时间失效;若也设 10 分钟 TTL,
        数据会在运行 10 分钟后静默消失,表现为"FAQ 用一会儿就不灵了"。
        代价是改了 preset_qa.json 后必须 `force=True` 重播或手动清键。

        `force` 的幂等判断用**单个标记位** `DETAIL_SEEDED_KEY`(而不是逐条检查):
        明细有几十条,逐条 exists 是几十次往返;用标记位 = 一次 exists。
        代价是**改数据后不会自动重播** —— 这正是 force 参数存在的理由。

        ⚠ pipeline 里的三处细节:
        - `self._exact_key(pair["question"], route=scope)` 传的是**原始问题**而不是上面
          payload 里归一化后的串 —— 好在 `_exact_key` 内部会再做一次 `_normalize_query`,
          结果一致。**前提是别把 _exact_key 里的归一化去掉**;
        - 明细的 `sources` 固定写 `[]`(预设问答没有票据来源),
          所以命中明细时前端拿不到可核对的票号 —— 这也是"明细绝不能进相似度层"
          的另一个理由:答错了没有任何可查证的痕迹;
        - **每条明细对每个缓存作用域各写一份**(基础线路 + 三条新线路,见
          `core/routes.py::CACHE_SCOPES`)。明细是人工维护的固定问答,与走哪条线路
          无关,只写基础线路会出现"切到 Agentic 后预设明细全部 miss"——
          不报错,只是每次都要跑完整链路,极难归因。
        """
        if not force and self.redis.exists(DETAIL_SEEDED_KEY):
            return 0
        _, detail_pairs = split_by_layer(load_preset_pairs())
        # 空明细提前返回:**不写标记位**。否则"这份数据里没有明细"会被永久记住,
        # 之后补上明细数据也不会被播进去
        if not detail_pairs:
            return 0
        pipe = self.redis.pipeline()
        for pair in detail_pairs:
            payload = json.dumps(
                {
                    "question": _normalize_query(pair["question"]),
                    "answer": pair["answer"],
                    "sources": [],
                },
                ensure_ascii=False,
            )
            # 每个线路作用域各写一份（见 docstring 第三条）：这样"切线路后预设明细照样命中"
            for scope in CACHE_SCOPES:
                pipe.set(self._exact_key(pair["question"], route=scope), payload)
        # 标记位与数据在同一个 pipeline 里提交,尽量缩小"数据写了但标记没写"的窗口
        pipe.set(DETAIL_SEEDED_KEY, "1")
        pipe.execute()
        logger.info(
            f"[缓存] 明细问答写入精确层完成,共 {len(detail_pairs)} 条"
            f"(只做精确命中;每条已按 {len(CACHE_SCOPES)} 个线路作用域各写一份)"
        )
        return len(detail_pairs)

    @staticmethod
    def _exact_key(query: str, route: str = "") -> str:
        """精确缓存键:归一化查询 + 配置作用域 + (可选)线路作用域。

        作用域里放模型名、集合名与两个阈值,配置一变旧答案自动失效;
        刻意**不放**会话 history——同一问题第二次问也必须命中。

        `route` 只在**非空**时才拼进作用域，这是刻意的兼容取舍：
        基础线路（route=""）的键与加线路功能**之前逐字节相同**，
        于是 Redis 里已有的运行时缓存与 `seed_details` 播下的 60 条明细都还有效；
        另外三条线路各得一个新键。若把空 route 也拼进去（例如拼成 "|" ），
        所有旧键会瞬间变成不可达的孤儿——缓存整体失效不会报错，
        只会表现为"怎么全 miss 了"，很难归因。

        为什么用 sha256 而不是直接把问题拼进 key:
        Redis 的 key 是任意二进制串,中文问题直接当 key 完全可行,但会带来
        "key 长度不可控"(长问题 key 很长)、"含空格/特殊字符难用 redis-cli 操作"
        的问题;哈希后是定长十六进制,前缀 + 定长串方便按模式统计。
        代价是**不可读** —— 想知道某个 key 对应哪个问题,只能反查 payload 里的
        `question` 字段(这也是 store 时要把归一化问题一并写进 payload 的原因)。

        作用域里放**两个阈值**(sim_threshold / relevance_p)看起来有点过度:
        它们只影响"什么会被命中",并不改变答案内容。放进去是为了让
        **阈值调优后立刻看到效果** —— 否则调完阈值旧缓存还在,会误判成"调了没用"。
        两个阈值都 `str()` 后再拼:float 的 repr 在同一 Python 版本内稳定,
        且同一次部署里配置值不会变,所以不影响键的稳定性。

        ⚠ 这个键是 `store` 与 `_lookup_exact`/`seed_details` 的**共同契约**:
        任何一处改了拼接内容或顺序,缓存就会整体失效(答出来全是 miss 但没报错),
        三处必须同时改。
        """
        parts = [
            settings.llm.model,
            settings.milvus.collection,
            str(settings.redis.sim_threshold),
            str(settings.rerank.relevance_p),
            _normalize_query(query),
        ]
        if route:
            parts.append(f"route={route}")
        return EXACT_KEY_PREFIX + hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    def _lookup_exact(self, query: str, route: str = "") -> dict | None:
        """精确层查询:GET 到就 json.loads,GET 不到返回 None。

        没有 try/except —— 异常直接往上传给 `lookup` 的统一 except 处理,
        避免在两层各写一份降级逻辑(那样两处的日志与行为容易漂移)。
        """
        raw = self.redis.get(self._exact_key(query, route=route))
        if raw is None:
            return None
        return json.loads(raw)

    def _ensure_preset_loaded(self) -> None:
        """惰性把相似度层的问法/答案/向量矩阵载入进程内存(只需一次)。

        用 `_preset_matrix is not None` 当"已加载"的判据而不是加一个 bool 标志:
        矩阵是唯一真正占内存的对象,它的存在性就是最直接的状态。
        注意 `if ... is not None` 不能用真值判断 —— numpy 数组的真值是
        "元素是否全非零",空矩阵/全零矩阵会触发明暗不定的行为
        (其实 numpy 对多元素数组做真值判断会直接抛 ValueError)。

        内部先 `self.seed_preset()`(不带 force):保证 Redis 里**一定有**数据可读。
        这是"第一次访问缓存时自动播种"的便利设计 —— 代价是
        **第一个未命中请求要现算预设问法的向量**(几十毫秒到秒级),
        进程重启后每个进程都会经历一次。所以课案里"启动预热"是有价值的做法(见报告)。

        `np.maximum(norms, 1e-10)` 是逐元素除零保护(全零向量行)。
        矩阵在这里**一次性归一化**:之后每次查询只需归一化 query 那一个向量,
        然后一次 `@` 得到所有相似度 —— 这是把 O(n) 次归一化压成 O(1) 的关键。

        ⚠ 若 Redis 里两个键不存在(比如被人手动清过),`json.loads(None)` 会抛
        TypeError,最终由 `lookup` 捕获降级成 miss。表现是"FAQ 层静默失效",见报告。
        """
        if self._preset_matrix is not None:
            return
        self.seed_preset()
        pairs = json.loads(self.redis.get(PRESET_PAIRS_KEY))
        vectors = np.array(json.loads(self.redis.get(PRESET_VECTORS_KEY)), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self._preset_questions = [p["question"] for p in pairs]
        self._preset_answers = [p["answer"] for p in pairs]
        self._preset_matrix = vectors / np.maximum(norms, 1e-10)

    def _lookup_preset(self, query: str) -> dict | None:
        """相似度层查询:找最像的一条标准问法,相似度不足阈值则返回 None。

        `argmax` 取**最相似的一条**而不是"第一条超阈值的":相似度层的语义是
        "找出最接近的标准问法",若按顺序取第一条超阈值,当多条都超阈值时
        命中哪一条取决于数据顺序(不确定)。argmax 让结果由分数唯一决定。

        阈值判断用 `score < threshold` 提前返回 None(而不是 if 里包一大坨):
        让"没命中"这条最常见路径最先返回,成功路径的代码保持平铺。
        **边界包含**:`score == threshold` 算命中(用 `<` 而不是 `<=`),
        与 threshold.py 里 `score >= threshold` 的判正例口径一致 ——
        两处口径必须一致,否则标定出来的阈值在运行时会有 1 个粒度的偏差。

        `sources` 固定为 `[]`:预设答案是"标准答案",不来自某次检索,
        所以没有票据来源可给。**这正是不该让明细问答进相似度层的另一半理由** ——
        答错了用户无法从界面上核对(没有来源票据)。前端若想区分,应看 cache_hit 字段。

        `round(score, 4)` 只影响展示(便于日志/排障比对),不影响判断;
        相似度值出现在返回值里是刻意的:排障时能立刻看到"这次是 0.79 还是 0.81",
        不用再去重算 embedding。

        `from retrieval.embedding import embed_query` 同样是函数内延迟导入
        (理由见 seed_preset 的注释)。
        """
        self._ensure_preset_loaded()
        from retrieval.embedding import embed_query

        vec = np.array(embed_query(query), dtype=np.float32)
        # 归一化只用单向量点积(与矩阵那次归一化配合,得到余弦相似度)。
        # max(norm, 1e-10) 兜零向量:否则除零得到 inf/nan,argmax 会选中一个
        # 完全随机的位置并把它的答案返回出去 —— 典型"静默答错"
        vec = vec / max(float(np.linalg.norm(vec)), 1e-10)
        sims = self._preset_matrix @ vec
        best = int(np.argmax(sims))
        score = float(sims[best])
        if score < settings.redis.sim_threshold:
            return None
        return {
            "question": self._preset_questions[best],
            "answer": self._preset_answers[best],
            "sources": [],
            "cache_hit": "preset",
            "similarity": round(score, 4),
        }

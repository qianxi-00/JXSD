"""从用户问题中抽取可用于 Milvus filter 的票据检索条件。

对照基础篇课案 `pipeline/filters.py`:
- `extract_ticket_filters` 用轻量规则抽取票据类型/年份/人员/路线/金额范围/票号,
  不追求完整 NER,只覆盖高频查询条件(复杂场景可换成 LLM/实体识别输出同样结构);
- `build_milvus_filter` 把条件转成 Milvus 标量过滤表达式,金额统一用分,日期用 date_int。

与课案原文的两处差异(均为健壮性增强,不改变字段语义):
1. 人员抽取允许人名与票据词之间出现动词("王霞购买的火车票" → 王霞),并对
   "今年/去年"等时间词做拒绝;课案原正则在这两种输入上分别会多抽"王霞购买"和"今年的"。
2. 额外提供 `row_matches_filters`:BM25 语料已经在进程内缓存,过滤条件无法下推到
   Milvus,因此在打分后的选取阶段按与 `build_milvus_filter` 完全相同的语义逐行过滤,
   保证混合召回两条路径的过滤口径一致。

注意:字段为 null 的记录不会被当成 0,也不会满足金额或日期条件。

────────────────────────────────────────────────────────────────
两条读这个文件时最容易踩的坑(代码里的"有意为之"都是从这两条来的)
────────────────────────────────────────────────────────────────
1. ★ route 条件在当前库里**永不成立**:这里抽出来的是**中文**站名("北京到上海"),
   而入库时写进 tick.route 的是**拼音/英文**站名(`Hefei-Wulumuqi`、
   `AKESUJICHANG-SHOUDUJICHANG`,见 data_process/tick_extract.py)——
   于是 `route like "北京%上海%"` 一行都匹配不到。
   后果与对策都在链路上:一旦过滤后零命中,rag_pipeline.retrieve_union 会
   **先只摘掉 route 条件重试一次**,再退到完全不过滤,而不是把 person/日期/金额
   这些本来正确的硬条件一起丢掉。看到那段回退代码时不要以为它在兜底"别的问题"。
2. 这里的抽取是**规则猜测**,不是事实:抽错时硬过滤会把召回直接清空,表现为
   "明明库里有票却答没找到"。所以本模块只负责"猜得尽量准 + 猜不出就不加条件",
   "猜错了怎么办"由链路侧的回退负责(同上)。

另外,过滤条件的键都是**出现才存在**的:抽不到的字段在字典里根本没有键,
`build_milvus_filter` / `row_matches_filters` 都按"键存在与否"决定加不加这条条件 ——
所以不要用 `.get(key, 默认值)` 的方式给不存在的键补默认值,那会凭空造出一条恒假条件。
"""

from __future__ import annotations

# 说明：本模块**不导入任何项目内模块**（只用标准库 re/datetime/typing），
# 所以它可以被单独 import、单独跑 `python RAG\pipeline\filters.py` 自检，
# 不需要 config / Milvus / Redis。这是有意的：过滤是最底层的纯函数层。

import re
from datetime import date
from typing import Any

# 时间词出现在名词位置时不能当成人名
TIME_WORDS = {"今年", "去年", "明年", "本月", "这个月"}
# 人名与票据词之间可能出现的动词
VERB_WORDS = ("购买", "乘坐", "坐", "买", "订", "报销", "开给", "开出")
# 票据词(与课案一致)
TICKET_WORDS = "高铁票|火车票|机票|飞机票|发票|票据"

# 上面三个常量的用途（都是被下面的正则或 _extract_person 的兜底判断消费的）：
#   TIME_WORDS —— 事后过滤用。正则形如「2~4 个汉字 + 今年/去年」，"今年的高铁票"
#                 会被抓成"今年的"，用这个集合把它挡掉。它是**拒绝名单**，不是可选项。
#   VERB_WORDS —— 两处用：① 参与 _PERSON_WITH_TICKET_RE，跳过人名与票据词之间的动词；
#                 ② 事后兜底判断"人名以动词结尾"（如"王霞购买"），说明非贪婪收窄还失败。
#   TICKET_WORDS —— 票据词的交替串，只给正则用；不含数量词/动词，避免把"买了 3 张"这类
#                 片段也当成票据词。

# 问句开头的礼貌/指令前缀。位置锚定 ^ 是关键：只在**句首**剥，句中的"请问"不动。
# 但注意末尾的 `+` 是"连续剥多层"：实测 "请帮我查一下王强2025年…" 会一次剥掉
# "请""帮我""查一下" 三个前缀，这正是想要的效果（剥不干净就抓不到人名）；
# 反例是 "谢谢请请问王强2025年" —— 前缀连写时会被啃到只剩空串，人名就丢了。
# 剥掉的目的：让下面的 _PERSON_WITH_YEAR_RE / _PERSON_WITH_TICKET_RE 能用 match
# （从 0 位置开始）稳定抓到"句首的人名"。
_POLITE_PREFIX_RE = re.compile(r"^(?:(?:请|帮我|麻烦|查询|查一下|查找|看看|请问)\s*)+")
# 「人名 + 去年/今年/20xx年」，例如"王强2025年…"。用 match（而非 search）是刻意的：
# 人名基本固定在句首；search 会在句中的任意名词位置上乱抓。
# 2~4 个汉字是中文姓名的常见长度；抓到的结果还要再核对 TIME_WORDS。
_PERSON_WITH_YEAR_RE = re.compile(r"([\u4e00-\u9fa5]{2,4})(?:去年|今年|20\d{2}\s*年)")
# 「人名 (+ 可选动词) (+ 可选"的") + 票据词」，例如"王霞购买的火车票"、"王强的火车票"。
# ★ `{2,4}?` 的**非贪婪**不能改成贪婪：实测贪婪版对"王霞购买的火车票"会抓成"王霞购买"、
#   对"王强的火车票"会抓成"王强的"（多吃了动词或"的"）；非贪婪只吃最少的 2 个汉字，
#   把后面那截留给动词可选组和"的"可选组。
# 剩下仍可能误判（"请帮我查一下…"后面的任意 2~4 汉字都可能被抓），
# 所以 _extract_person 里还有一道"以动词结尾就丢弃"的兜底。
_PERSON_WITH_TICKET_RE = re.compile(
    rf"([\u4e00-\u9fa5]{{2,4}}?)(?:的?)(?:{'|'.join(VERB_WORDS)})?(?:的)?(?:{TICKET_WORDS})"
)
# 票号：认 5 种说法（票据号码/票号/发票号码/车票编号/电子客票号，与课案一致），
# 冒号可有可无且允许全角；票号本体限定 [A-Za-z0-9-]（数字/字母/连字符）。
# 用 search 而非 match —— 票号通常出现在句子中间（"票号 INV20250161 的金额是多少"）。
#
# ⚠ 字符集**不含下划线**，于是带下划线的票号会被从下划线处截断：
#   实测（加 `_` 之前）`_extract_ticket_no("票号 ticket_001 的金额是多少")` → `"ticket"`，
#   接着 filter 会拼成 `ticket_no == "ticket"` —— 这条条件恒不命中，等于把召回清空
#   （然后靠 retrieve_union 的回退兜住）。课案的提示词例句恰好就写了 `ticket_001`
#   （core/prompts.py 的"票据ticket_001的金额是多少？"），所以这不是假想输入。
#   已把 `_` 加进字符集（当前库里的真实票号 INV20250161 / T20230829236880 / 8762369777769
#   不含下划线，所以这是防未来输入，不是修线上故障）。
_TICKET_NO_RE = re.compile(
    r"(?:票据号码|票号|发票号码|车票编号|电子客票号)[:：]?\s*([A-Za-z0-9_-]+)"
)


def extract_ticket_filters(query: str, today: date | None = None) -> dict[str, Any]:
    """用轻量规则抽取票据检索条件,返回可直接交给 build_milvus_filter 的字典。

    这里**刻意不用 LLM / NER**：过滤发生在召回阶段、每次提问都要跑，规则版的延迟是
    微秒级、且行为完全可预测（LLM 抽条件会引入不确定性与额外费用）。代价是覆盖面有限 ——
    复杂问法（"上个月那几张没报销的"）抽不出条件，此时字典为空 = 不过滤，
    靠后面的向量 + BM25 召回与重排去兜，这是有意的降级方向（宁可多召回，不要抽错清空）。

    返回的键分两类：
      - 直接对应数据库字段：ticket_type / person / ticket_no（+ 由 route 拆出的
        route_from、route_to）；
      - 由自然语言换算出的复合键：date_start / date_end（YYYYMMDD 整数）、
        amount_min_fen / amount_max_fen（单位分）及其 amount_*_operator（比较符）。
    抽不到的键**不会出现**在返回值里。
    """
    # today 可注入：跨年会让"今年/去年"的期望值变化，测试要能钉住固定日期；
    # 不传就取系统当天（生产路径就是这么调的）。
    today = today or date.today()
    # 只放"抽到了"的键 —— 下游两条过滤实现都按"键是否存在"决定要不要加条件，
    # 空值必须表现为"没有这个键"，不能表现成 None/""。
    filters: dict[str, Any] = {}
    text = query.strip()

    # 以下每个 _extract_* 的约定：抽不到返回空串（人名/路线等）或 None（年份/金额），
    # 这里统一用真值判断收口 —— 所以"空串"和"None"都能被同一句 if 拦下。
    ticket_type = _extract_ticket_type(text)
    if ticket_type:
        filters["ticket_type"] = ticket_type

    # 年份落成 date_int 的**闭区间** [YYYY0101, YYYY1231]。
    # 库里 date_int 就是 YYYYMMDD 形式的整数（不是时间戳、不是字符串），
    # 所以区间上下界可以直接用整数比较表达，不需要做日期解析/格式化。
    year = _extract_year(text, today)
    if year:
        filters["date_start"] = int(f"{year}0101")
        filters["date_end"] = int(f"{year}1231")

    person = _extract_person(text)
    if person:
        filters["person"] = person

    # route 拆成两个键存：库里 route 是「出发地-到达地」的**单个字符串**，
    # 表达式只能写成 `route like "出发地%到达地%"`，拆开才拼得出来。
    route = _extract_route(text)
    if route:
        filters["route_from"], filters["route_to"] = route

    # 金额：人话里的阈值是**元**（"超过 1000 元"），库里 amount_fen 是**分**，所以 ×100。
    # 先算出 float 再用 round 转 int，是为了躲开二进制浮点误差 ——
    # 实测：1.15 * 100 = 114.99999999999999，直接 int() 会截成 114（**少一分钱**，
    # 恰好是边界上的票据就会被过滤掉），round() 才能得到正确的 115。
    amount_min = _extract_amount_min(text)
    if amount_min is not None:
        filters["amount_min_fen"] = int(round(amount_min[0] * 100))
        # 比较符由抽取阶段决定并一起存下来（"超过"→">"、"不少于"→">="），
        # 不让下游再猜一次中文词的含义。
        filters["amount_min_operator"] = amount_min[1]

    amount_max = _extract_amount_max(text)
    if amount_max is not None:
        filters["amount_max_fen"] = int(round(amount_max[0] * 100))
        filters["amount_max_operator"] = amount_max[1]

    ticket_no = _extract_ticket_no(text)
    if ticket_no:
        filters["ticket_no"] = ticket_no

    return filters


def build_milvus_filter(filters: dict[str, Any]) -> str:
    """把抽取出的条件转成 Milvus 标量过滤表达式(无条件时返回空串)。

    返回**空串**而不是 None：`client.search(filter="")` 在 Milvus 里等价于"不过滤"，
    调用方（rag_pipeline._recall）再统一把空值收敛成 None 后传给向量召回。

    拼表达式有两条纪律：
    - 字符串字段必须带双引号并转义（走 _escape），数字字段**不能**带引号 ——
      带了会被当成字符串比较，直接报类型错误或匹配不到；
    - 各条件之间只用 `and`（全部是硬过滤）。硬过滤一条抽错就等于把召回清空，
      所以链路侧必须保留"零命中回退"，不能把这里的结果当建议。
    """
    exprs: list[str] = []

    # 拼接顺序：ticket_type → person → ticket_no → 日期 → 金额 → route。
    # 顺序只影响日志与表达式的可读性，不影响结果（全是 and，可交换）。
    # 三个字符串分支都用「海象赋值 + 真值判断」：键不存在、值为空串都跳过，
    # 所以不会拼出 `person == ""` 这种恒假条件（那会让召回直接归零）。
    if value := filters.get("ticket_type"):
        exprs.append(f'ticket_type == "{_escape(value)}"')
    if value := filters.get("person"):
        exprs.append(f'person == "{_escape(value)}"')
    if value := filters.get("ticket_no"):
        exprs.append(f'ticket_no == "{_escape(value)}"')
    # date_int 存的是 YYYYMMDD 整数；年份→区间上下界的换算已在 extract_ticket_filters
    # 里完成，这里只负责拼 >= / <=。int() 是防御性转换（万一被传进字符串）。
    # 判据用真值（不是 is not None）：date_int=0 表示"没有日期"，加进条件反而变成
    # `date_int >= 0` 这种看似过滤、实则放行的噪音条件。
    if filters.get("date_start"):
        exprs.append(f"date_int >= {int(filters['date_start'])}")
    if filters.get("date_end"):
        exprs.append(f"date_int <= {int(filters['date_end'])}")
    # 金额分支的判据是 `is not None`（而不是真值）：金额 0 是有意义的取值
    # （零元发票），不能被当成"没有条件"丢掉。
    # 金额条件自带操作符（">" / ">=" / "<" / "<="）；缺省时给的是**宽口径**
    # （下限默认 ">="、上限默认 "<="），与 row_matches_filters 的默认值保持一致。
    if filters.get("amount_min_fen") is not None:
        operator = filters.get("amount_min_operator", ">=")
        exprs.append(f"amount_fen {operator} {int(filters['amount_min_fen'])}")
    if filters.get("amount_max_fen") is not None:
        operator = filters.get("amount_max_operator", "<=")
        exprs.append(f"amount_fen {operator} {int(filters['amount_max_fen'])}")
    # 出发地/到达地必须**同时**存在才拼 like：只有一个的话语义不完整
    # （不知道它是起点还是终点），宁可不过滤也不猜。
    # ⚠️ 这里拼的是中文站名，而库里 route 存的是拼音/英文（见模块 docstring 的坑 1）——
    # 这条条件在当前库上永不成立；保留它是因为"换数据源 / 改抽取"后它才是对的，
    # 而链路侧的回退能兜住现在这种零命中。
    if filters.get("route_from") and filters.get("route_to"):
        exprs.append(
            f'route like "{_escape(filters["route_from"])}%{_escape(filters["route_to"])}%"'
        )

    return " and ".join(exprs)


def row_matches_filters(row: dict, filters: dict[str, Any]) -> bool:
    """单行记录是否满足过滤条件,语义与 build_milvus_filter 一致。

    为什么需要这个函数：BM25 语料是**进程内**的（keyword_retrieval._load_index 把整个
    集合一次拉到内存并缓存），过滤条件根本无法下推给 Milvus，只能在 BM25 打分后的
    选取阶段逐行比对。向量侧走 Milvus 表达式、BM25 侧走这个函数，两边口径必须一致 ——
    否则同一个问题在两条召回路径上会命中不同的票据集合，合并去重之后就没人能解释
    "这条记录凭什么进来"。

    null 语义（与 Milvus 的标量过滤保持一致）：字段为 null 的记录**不满足**任何
    数值/日期条件，也不会被当成 0。所以下面每处数值比较前都先判 `is None → return False`：
    既表达了这个语义，也避免让 None 参与比较（Python 里 `None < 100` 会直接 TypeError）。
    """
    # 没有条件 = 全部通过（不是全部拒绝）。链路的"零命中回退"最终就是调到这里，
    # 所以这一行是回退能生效的前提，不能反写成 return False。
    if not filters:
        return True

    # 字符串字段要求**完全相等**，与 `field == "值"` 同语义。
    if value := filters.get("ticket_type"):
        if row.get("ticket_type") != value:
            return False
    if value := filters.get("person"):
        if row.get("person") != value:
            return False
    if value := filters.get("ticket_no"):
        if row.get("ticket_no") != value:
            return False

    # 日期区间：与 `date_int >= x` / `date_int <= y` 同语义。
    # 这里的判据写成"显式的 is not None 且 != 0"，而 build_milvus_filter 那边用的是真值 ——
    # 两种写法对当前抽取器**等价**（年份只会产出 YYYYMMDD 这样的非 0 整数）。
    # 之所以写成显式比较：让"0 表示不限制"这个约定在代码里留下痕迹，
    # 免得将来有人把 date_start 默认成 0 时两条路径行为分叉。
    if filters.get("date_start") is not None and filters.get("date_start") != 0:
        date_int = row.get("date_int")
        if date_int is None or int(date_int) < int(filters["date_start"]):
            return False
    if filters.get("date_end") is not None and filters.get("date_end") != 0:
        date_int = row.get("date_int")
        if date_int is None or int(date_int) > int(filters["date_end"]):
            return False

    # 金额：先判 row 的值为 None（数值缺失 → 不满足），再用抽取阶段存下的操作符比较。
    if filters.get("amount_min_fen") is not None:
        amount = row.get("amount_fen")
        operator = filters.get("amount_min_operator", ">=")
        if amount is None or not _compare(int(amount), operator, int(filters["amount_min_fen"])):
            return False
    if filters.get("amount_max_fen") is not None:
        amount = row.get("amount_fen")
        operator = filters.get("amount_max_operator", "<=")
        if amount is None or not _compare(int(amount), operator, int(filters["amount_max_fen"])):
            return False

    # route 同样要求出发地/到达地成对出现（与表达式侧一致）。
    if filters.get("route_from") and filters.get("route_to"):
        route = row.get("route") or ""
        start = str(filters["route_from"])
        end = str(filters["route_to"])
        # 与 route like "出发地%到达地%" 同语义:以出发地开头且其后出现到达地
        # find 的起点取 len(start)，意图是别让出发地自己充当到达地（否则 `like "北京%北京%"`
        # 那种"同城"条件会被 route="北京" 这条记录满足）。
        # ⚠ 这个保护**只在到达地正好落在开头 len(start) 个字符之内时**才生效，边界上会与
        # Milvus 的 like 分叉（实测）：route="北京"、from=to="北京" 时这里判 False，
        # 而 `like "北京%北京%"` 语义上应当命中；反过来 route="上海上海-北京"、
        # from=to="上海" 时这里判 True（保护没拦住）。
        # 当前库里 route 是"拼音-拼音"且过滤条件永不成立（见模块 docstring），所以这些
        # 边界没有实际影响；改 route 数据源/抽取逻辑时要顺带对齐这层语义。
        # 注意这里的 route 也是空值兜底成 ""：空串既不以出发地开头，find 也返回 -1，
        # 自然落到 return False（等价于库里 route 为 null 时不满足路线条件）。
        if not route.startswith(start) or route.find(end, len(start)) < 0:
            return False

    return True


def _compare(value: int, operator: str, threshold: int) -> bool:
    """按操作符把两个整数比大小（只支持 > >= < <= 四种）。

    最后的 `return value <= threshold` 是**兜底分支**：未知操作符会被当成 "<=" 处理，
    而不是抛异常（实测 `_compare(5, "==", 3)` 返回 False，不报错）。
    当前调用方只会产出这 4 个操作符，所以走不到兜底；将来若扩展操作符（如 "=="），
    必须在这里补分支 —— 否则会静默按 "<=" 处理，条件判错还不报错。
    """
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    return value <= threshold


def _extract_ticket_type(text: str) -> str:
    """票据大类：train / flight / invoice；认不出返回空串。

    返回值必须与库里 ticket_type 字段的取值**逐字一致**（入库时写的就是这三个英文值，
    keyword_retrieval 里的过滤条件 `ticket_type in ["flight","invoice","train"]`
    也依赖它们）。大小写/拼写写错不会报错，只会静默零命中。
    """
    # 下面的 if 顺序就是**优先级**：混说时（实测"火车票发票"→ train）按火车 → 机票 →
    # 发票的顺序取第一个命中的，不会返回多个值，也不会报"歧义"。
    if any(word in text for word in ("高铁", "火车", "动车", "车票")):
        return "train"
    if any(word in text for word in ("机票", "飞机票", "航班", "航空")):
        return "flight"
    if "发票" in text:
        return "invoice"
    # 认不出就返回空串 → 上层不生成 ticket_type 条件。
    # 这是有意的降级：宁可不过滤（多召回再重排），也不要瞎猜一个类型把召回清空。
    return ""


def _extract_year(text: str, today: date) -> int | None:
    """从「去年/今年/20xx年」取年份，都没有则返回 None。"""
    # 判定顺序：先相对词（去年、今年），再绝对年份 —— 两者同时出现时以相对词为准。
    if "去年" in text:
        return today.year - 1
    if "今年" in text:
        return today.year
    # 只认 20xx 年（本库票据都是 2000 年后），且必须带"年"字：
    # 没有"年"这个右边界，OCR 正文里的票号/金额数字串会被误当成年份。
    match = re.search(r"(20\d{2})\s*年", text)
    if match:
        return int(match.group(1))
    # 注意 TIME_WORDS 里还有「明年/本月/这个月」，但这里**不支持**它们：
    # 抽象不出合适的区间（本月不是整年），宁可不加日期过滤也不要猜一个区间。
    return None


def _extract_person(text: str) -> str:
    """抓句首人名，抓不到返回空串。

    两轮尝试，顺序不能反：
    1.「人名 + 去年/今年/20xx年」—— 更严格（人名后必须紧跟时间词），先试；
    2.「人名 (+动词) (+的) + 票据词」—— 更宽松，放后面兜底。
    两轮都还要再过一道 TIME_WORDS / 动词后缀的检查，因为正则只能保证"形状像人名"。
    """
    # 先剥礼貌前缀，否则 match 会从"请/帮我"开始，人名永远抓不到。
    normalized = _POLITE_PREFIX_RE.sub("", text.strip())

    # match（而非 search）：人名按"句首成分"处理，避免在句子中段乱抓。
    match = _PERSON_WITH_YEAR_RE.match(normalized)
    if match:
        person = match.group(1)
        # 第一条正则没有动词组，"今年的高铁票"会抓到"今年的" —— 用 TIME_WORDS 挡掉。
        if person not in TIME_WORDS:
            return person

    match = _PERSON_WITH_TICKET_RE.match(normalized)
    if match:
        # 非贪婪组仍可能带上一个"的"（"王强的火车票"），去掉；
        person = match.group(1).removesuffix("的")
        # 三重排除：空串（说完"的"就没了）/ 时间词 / 以动词结尾（"王霞购买"）。
        if (
            person
            and person not in TIME_WORDS
            and not any(person.endswith(word) for word in VERB_WORDS)
        ):
            return person
    # 认不出人名就返回空串 → 不生成 person 条件。
    # 人名抽错是"最贵"的错：库里 person 是精确相等匹配，抽错一个字的后果是整条过滤零命中。
    return ""


def _extract_route(text: str) -> tuple[str, str] | None:
    """抓「从 A 到 B」，返回 (出发地, 到达地)；抓不到返回 None。

    实测行为（写注释时的探针结果）：
      "从北京到上海的高铁票" → ("北京", "上海")
      "从北京南到上海虹桥的高铁票" → ("北京南", "上海虹桥")
      "从北京到上海多少" → ("北京", "上海")
      "北京到上海的票"（没有"从"） → None（本函数不认这种省略说法）
    """
    # 正则三段的用意：
    # - 出发地 {2,20}（贪婪）、到达地 {2,20}?（**非贪婪**）：这样"上海"才不会
    #   被扩成"上海的高铁票"；
    # - 结尾那组 (?:的|高铁|火车|机票|飞机|票|费用|行程|多少|有哪些|一共|总金额|$)
    #   是**右边界**：非贪婪组要一直扩到"后面能接上一个边界"为止，
    #   没有这组限定词，非贪婪会在刚够 2 个汉字时立刻收手，抽出来的到达地会缺字；
    # - 字符集允许中文与英文字母（拼音/英文站名也能抽），但不含数字与空格 ——
    #   所以"从 12306 到…"这类噪音不会被当成地名。
    match = re.search(
        r"从([\u4e00-\u9fa5A-Za-z]{2,20})到([\u4e00-\u9fa5A-Za-z]{2,20}?)"
        r"(?:的|高铁|火车|机票|飞机|票|费用|行程|多少|有哪些|一共|总金额|$)",
        text,
    )
    if match:
        # 返回中文站名 —— 而库里 route 是拼音/英文，见模块 docstring 的坑 1。
        return match.group(1), match.group(2)
    return None


def _extract_amount_min(text: str) -> tuple[float, str] | None:
    """抽金额**下限**：返回 (数值, 比较符)，抽不到返回 None。数值单位是元。"""
    match = re.search(r"(超过|大于|高于|不少于|不低于|至少)\s*(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    # 中文词分两档，对应的比较符不同（边界值算不算在内）：
    #   "超过/大于/高于" → ">"   （严格大于，不含边界）
    #   "不少于/不低于/至少" → ">="（含边界）
    # 这一档差别会影响差一分钱的边界票据，所以由抽取阶段定下来，不要让下游猜。
    operator = ">" if match.group(1) in {"超过", "大于", "高于"} else ">="
    # 返回 float 元的原值；换算成"分"是 extract_ticket_filters 的事（那里统一 round）。
    return float(match.group(2)), operator


def _extract_amount_max(text: str) -> tuple[float, str] | None:
    """抽金额**上限**：返回 (数值, 比较符)，抽不到返回 None。数值单位是元。"""
    match = re.search(r"(小于|低于|不超过|少于|至多)\s*(\d+(?:\.\d+)?)", text)
    if not match:
        return None
    # 同 _extract_amount_min 的分档："小于/低于/少于" → "<"，"不超过/至多" → "<="。
    operator = "<" if match.group(1) in {"小于", "低于", "少于"} else "<="
    return float(match.group(2)), operator


def _extract_ticket_no(text: str) -> str:
    """抽票号，抽不到返回空串。用的是 search（票号常在句子中段），见 _TICKET_NO_RE。"""
    match = _TICKET_NO_RE.search(text)
    return match.group(1) if match else ""


def _escape(value: Any) -> str:
    r"""转义 Milvus 过滤表达式里字符串字面量的两个特殊字符。

    顺序不能反：必须**先转反斜杠、再转双引号** —— 反过来的话，第二步加进去的反斜杠
    不会再被第一步处理（它已经跑完了），表达式照样是错的。

    举例（左边是原值，右边是转义后的结果）：
        反斜杠  \  →  \\
        双引号  "  →  \"

    不转义的后果：值里出现双引号会拼出语法错误的表达式（Milvus 直接报错）；
    值以反斜杠结尾时会把后面那个收尾引号"吃掉"，同样是语法错误。

    （说明用 r 前缀的原始字符串：正文里要出现反斜杠本身，普通字符串会把它当转义前缀，
    既会把 \b 编译成退格符，还会触发 SyntaxWarning。）
    """
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


if __name__ == "__main__":
    # 自检:断言式小样例,逻辑坏了会直接失败
    # 这个块验证什么（一条完整的"抽取 → 拼表达式 → 逐行过滤"闭环）：
    #   1. 一句带礼貌前缀 + 人名 + 年份 + 路线 + 金额下限的复合问句，能被抽成**精确相等**的
    #      字典（含操作符 ">" 与"分"为单位的 100000）—— 也就是说命名实体、时间词、
    #      路线边界、金额档位、单位换算这五件事同时被钉住；
    #   2. 该字典拼出的 Milvus 表达式字符串逐字符合预期（字段名、引号、操作符、
    #      and 的连接顺序都钉住）—— 任何一处改动都会让断言红，这是有意的契约；
    #   3. 没有条件的问句返回空字典（而不是 {"person": ""} 之类的噪音键）；
    #   4. row_matches_filters 在"条件全部命中"时返回 True（BM25 侧口径的最小验证）。
    # 注意这条断言用的是**显式年份**"2025年"，所以它不受跨年影响（即使今天不是 2025 年，
    # _extract_year 也会从文本里取到 2025）。函数保留 today 参数是为了测"今年/去年"
    # 这类相对表述时能注入固定日期。
    f = extract_ticket_filters("请帮我查一下王强2025年从北京到上海的高铁票,金额超过1000元")
    # 断言 1：抽取结果必须**精确等于**这 8 个键。用 == 全等而不是逐个 in 判断，
    # 是为了同时钉住"不该有的键一个都不能多"（比如误抽出的 route/日期噪音键）。
    # 末尾的 `, f` 是断言失败时的附加信息 —— 打印实际值，方便一眼看出抽歪成什么样。
    assert f == {
        "ticket_type": "train",
        "date_start": 20250101,
        "date_end": 20251231,
        "person": "王强",
        "route_from": "北京",
        "route_to": "上海",
        "amount_min_fen": 100000,
        "amount_min_operator": ">",
    }, f
    # 断言 2：表达式逐字相符。它同时钉住三件事：
    #   - 字符串字段有引号、数字字段没有引号；
    #   - "超过" 对应严格大于 ">"（而不是 ">="）；
    #   - and 的连接顺序与 build_milvus_filter 里的拼装顺序一致。
    assert build_milvus_filter(f) == (
        'ticket_type == "train" and person == "王强" and date_int >= 20250101'
        " and date_int <= 20251231 and amount_fen > 100000"
        ' and route like "北京%上海%"'
    )
    # 断言 3：一句没有任何票据线索的话必须返回**空字典**（不是含空值键的字典）——
    # 空字典意味着"不过滤"，这是链路降级的关键前提。
    assert extract_ticket_filters("你好") == {}
    # 断言 4：BM25 侧逐行过滤的最小验证 —— 条件全中时返回 True。
    # 注意这条只覆盖了 True 分支；False 分支（人名不符、金额为 None 等）由 tests/ 里的用例覆盖。
    assert row_matches_filters({"person": "张三", "amount_fen": 100}, {"person": "张三"})
    # 走到这里说明四条断言都过了；打印一行让人/CI 都能看出"自检真的跑了"。
    print("filters.py 自检通过")

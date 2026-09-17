"""Text-to-SQL:把自然语言票据问题翻译成单条只读 SQL,校验后执行并强制行数上限。

流程(课案要求):模型按表结构生成 SQL → 校验只允许单条 SELECT → 执行并强制行数上限 → 返回 JSON 结果。

三种结果必须分开表达,不能笼统当成"没查到":
    - 校验失败        → SQLValidationError
    - 查询无结果      → 执行成功但返回空列表
    - 系统失败(DB 故障) → SQLExecutionError

上游 LLM 生成的 SQL 一律按不可信输入对待:先过 validate_sql 的词法白名单,再执行。

为什么这条链路要单独存在(而不是让 Agent 直接查库):票据问题里有一大类是**精确统计**
——"一共有多少张火车票""黄帅的火车票总共多少钱""2025 年 3 月的发票总额"。这类问题
向量检索答不了也不该答:相似度召回的是"看起来相关"的票,凑出的合计差一张就全错,
而且错得看不出来。所以凡是有确定答案的聚合/计数,都走本模块的结构化查询
(见 finance_agent.RAG_SYSTEM_PROMPT 里对两类工具的调度约定)。

信任边界(本模块的核心设计):
    LLM 的输出是不可信输入,它可能被票据正文里的注入文字带偏,生成 DROP/多条语句。
    因此执行前必须过两层:
      1. validate_sql —— 词法白名单:单条、只读、无注释、无写关键字;
      2. apply_row_limit —— 强制行数上限,防止 `SELECT *` 把整表拉进上下文。
    第二道防御是**纵深**的:query_tickets() 自己再调一次 validate_sql,
    这样即使调用方绕过 @tool 直接调函数,写操作也落不到数据库上。
    注意这是"词法白名单"而不是完整的 SQL 解析——它挡的是 LLM 生成的危险语句,
    不是恶意攻击者精心构造的绕过(要做后者得换成 AST 级解析或只读数据库账号)。

五态结果契约(贯穿到 @tool query_ticket_db 的返回值):
    ok / no_result / invalid_sql / db_error / llm_error —— 五个状态**必须分开表达**。
    把 db_error 或 llm_error 报成 no_result 是最坏的失败方式:用户会以为"真的没有这张票",
    而实际上是数据库没起或模型超时,前者需要修环境、后者需要重试,都不该被当成业务结论。
"""

import re
from datetime import date, datetime
from decimal import Decimal

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import Field, SQLModel

from config import settings
from core.logger import logger

# 注意：本模块**刻意不在 import 期连数据库**——get_engine() 是惰性的（见下），
# 所以导入它（以及跑离线测试）不依赖 PostgreSQL 是否启动。
# 同理，text_to_sql 用的 ChatOpenAI 是每次调用时新建的，导入期也不碰网络。

# ============================================================
# 一、异常类型:让调用方区分"校验失败"与"系统失败"
# ============================================================
class SQLValidationError(ValueError):
    """SQL 没有通过只读校验(多条语句 / 非 SELECT / 含写关键字 / 含注释 / 空)

    继承 ValueError 而不是自定义基类：校验失败本质是"入参不合法"，
    对调用方来说用 except ValueError 也能兜住。
    """


class SQLExecutionError(RuntimeError):
    """SQL 本身合法,但数据库执行失败(连接断了、表不存在、超时等)

    与 SQLValidationError 分开的原因：两者的**处置方式完全不同**。
    校验失败要回去改提示词/改生成逻辑（是模型的问题）；
    执行失败要看数据库状态（是环境的问题）。合成一个异常就再也分不出来了。
    """


# ============================================================
# 二、票据表模型(与 Milvus 里的票据同源,结构化字段一一对应)
# ============================================================
class Ticket(SQLModel, table=True):
    """结构化票据表。

    每列的 description 会拼进提示词,供模型理解列含义;所以说明要写成
    "名称 说明"里那句人话,且不要出现中文逗号(describe_table 用中文逗号连接)。

    这张表与 Milvus 里的票据**同源**：script/import_tickets_to_pg.py 把 Milvus 的
    结构化字段导入 PostgreSQL，字段名与取值口径一一对应（金额都是分、日期都是 date_int）。
    两边口径一致才能让"SQL 精确统计"与"向量召回原文"给出互相印证的结果；
    一旦有一边换了单位，两边对同一问题的答案就会不一致，而且都"看起来对"。
    """

    __tablename__ = "tickets"

    # primary_key=True 的 ticket_no 来自票据正文抽取，**可能抽不到**（抽取结果是 None）。
    # 导入脚本会跳过没有票号的记录，否则主键为 NULL 会直接插不进去。
    ticket_no: str = Field(primary_key=True, description="票据号(主键)")
    person: str = Field(description="报销人/买方")
    # ticket_type 的取值与 data/ 下的目录名、Milvus 里的同名字段一致：
    # invoice/flight/train。提示词里把这三个值写清楚，模型才能正确生成
    # `WHERE ticket_type = 'train'` 这类条件（写错成"火车票"就查不到任何行）。
    ticket_type: str = Field(description="票据类型:invoice(发票)/flight(机票)/train(火车票)")
    date_int: int = Field(description="日期整数,如 20250305")
    # 单位是分：与 Milvus/tick_extract 完全一致。提示词里明确要求"换算成元请除以 100"，
    # 是为了避免模型把 43600 当成 43600 元答给用户（少除两次 100 是这里最常见的错）。
    amount_fen: int = Field(description="金额,单位分(100 分 = 1 元)")
    route: str = Field(description="行程,如 杭州东-北京南")


def describe_table() -> str:
    """把票据表结构拼成"名称 说明"、用中文逗号连接的字符串,供提示词使用。"""
    # model_fields 是 SQLModel/Pydantic 的字段表，顺序就是类里定义的顺序
    # （测试 test_describe_table_lists_every_column_with_description 钉住了这一点）。
    # 用中文逗号拼接是刻意的：说明文字里允许出现英文逗号，
    # 若用英文逗号分隔，模型会分不清"字段分隔"和"说明内部标点"。
    # `field.description or ''` 兜住没写 description 的字段——不会崩，但那个字段
    # 在提示词里就成了光秃秃的名字，模型只能猜含义，所以新增列必须写 description。
    return "，".join(
        f"{name} {field.description or ''}" for name, field in Ticket.model_fields.items()
    )


_engine = None


def get_engine():
    """惰性创建数据库引擎。

    连接串来自 settings.finance_pg_url,里面必须带 +psycopg:环境里只有 psycopg3,
    SQLAlchemy 默认的 postgresql:// 会去找未安装的 psycopg2。

    connect_timeout 显式压到 5 秒:数据库栈不随 Docker 自启,服务没起时
    libpq 默认要等约 130 秒才报连接超时,交互式问答与测试都会被拖死。
    """
    global _engine
    if _engine is None:
        # pool_pre_ping=True：从连接池取连接前先 ping 一下，自动丢弃已断开的连接。
        # 数据库容器重启后旧连接会变成"僵尸连接"，没有这个开关时第一次查询必然失败，
        # 而且报错信息（server closed the connection）看起来像 SQL 写错了。
        # 注意：本函数是**惰性单例**——第一次调用才建引擎，之后复用。
        # 好处是"导入本模块不会连库"（离线测试能跑）；代价是首次查询要多花建连时间。
        _engine = create_engine(
            settings.finance_pg_url,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 5},
        )
    return _engine


# ============================================================
# 三、SQL 生成:表结构 + 问题 → 一条 PostgreSQL 查询
# ============================================================
# 提示词里四条要求各自对应一个已知的模型坏习惯（都是在 validate_sql 处会翻车的写法）：
#   1. "只输出 SQL" —— 模型爱先来一句"好的，这是查询语句："再给 SQL；
#   2. "只允许单条 SELECT" —— 不写这条时它偶尔会附带 EXPLAIN 或补一句注释；
#   3. "金额单位是分" —— 不写的话它会把 43600 当元来算（SUM 出来的数差 100 倍）；
#   4. "结尾不要写分号" —— 虽然后面 _clean_sql 会兜住分号，但少一次修补总归更干净。
# 两个占位符 {schema} / {question} 由 text_to_sql 用 str.format 填充；
# 因此**提示词里不能出现其他裸的花括号**（会被 format 当成占位符而 KeyError）。
SQL_PROMPT = """你是 PostgreSQL 专家。请根据下面的表结构和问题,生成一条可直接执行的查询语句。

表 tickets 的字段(名称 说明):{schema}

问题:{question}

要求:
1. 只输出 SQL 本身,不要任何解释文字,不要 Markdown 代码块
2. 只允许单条 SELECT 查询,禁止任何写操作
3. 金额单位是分,要换算成元请除以 100
4. 结尾不要写分号

SQL:"""


def _invoke_llm(prompt: str) -> str:
    """调用大模型生成 SQL(单独抽成函数,便于测试时整体替换)"""
    # temperature=0：SQL 生成要的是稳定复现，不是创意。同一个问题每次生成不同 SQL，
    # 会让"某次没查到"这种问题无法复现排查。
    # 这里单独抽成模块级函数（而不是内联在 text_to_sql 里），是为了让测试能
    # monkeypatch 掉整个模型调用——离线测试不碰外网就靠这个缝。
    llm = ChatOpenAI(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        model=settings.llm.model,
        temperature=0,
    )
    message = llm.invoke(prompt)
    content = message.content
    if isinstance(content, list):  # 多模态分片:把文本片段拼起来
        # 有些 OpenAI 兼容网关即使只返回文本也把 content 包成
        # [{"type": "text", "text": "..."}] 这种分片列表；直接把列表 str() 出来
        # 会得到一段带引号的 Python repr，后面正则全都匹配不上。
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


def _clean_sql(raw: str) -> str:
    """剥掉 ```sql 代码块包裹、去掉结尾分号(模型最常带的两种多余格式)"""
    # 这两步看着琐碎，但不做的话 validate_sql 会拒掉**本来合法**的查询：
    #   * ```sql 包裹 → 语句不以 SELECT 开头，判成"非 SELECT"；
    #   * 结尾分号 → 按分号切分后仍是单条，能过；但和模型偶尔多写的分号组合起来
    #     就可能被判成多条语句。统一在这里剥干净，比在 validate_sql 里放宽更清晰。
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        # 丢掉第一行（可能是 ``` 也可能是 ```sql），再看最后一行是不是收尾的 ```
        # 只按"首行开头 / 末行开头"判断，不要求配对：模型偶尔会漏掉收尾的那三个反引号。
        lines = cleaned.splitlines()[1:]  # 丢掉 ``` 或 ```sql 那一行
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    # rstrip() 再 rstrip(";") 再 strip()：先去掉首尾空白（模型常在分号后跟换行），
    # 再去结尾分号，最后再去一次空白。顺序反了（先 rstrip(";")）会因为末尾是换行
    # 而以为"没有分号"，于是分号留着。
    return cleaned.rstrip().rstrip(";").strip()


def text_to_sql(question: str) -> str:
    """把表结构与问题交给大模型,返回清洗后的 SQL 文本(此处不做校验,校验在 validate_sql)"""
    # 只负责"生成 + 清洗"，**不校验**：校验放在 validate_sql 里是有意的分工——
    # 生成要能失败（网络/鉴权问题，调用方记为 llm_error），
    # 校验失败则是另一个状态（invalid_sql），两者在 @tool 里要用不同的分支处理。
    prompt = SQL_PROMPT.format(schema=describe_table(), question=question)
    sql = _clean_sql(_invoke_llm(prompt))
    # debug 级别：生成的 SQL 是排查"为什么查不到"的第一手信息，
    # 但正常问答时不该刷屏，所以用 debug 而不是 info。
    logger.debug(f"[Text-to-SQL] 问题: {question} | 生成 SQL: {sql}")
    return sql


# ============================================================
# 四、只读词法白名单
# ============================================================
# 禁止的写/管理关键字。校验时按"独立词"匹配(\b),所以 deleted_at / update_time
# 这类含关键字前缀的普通列名不会被误杀。
#
# 这份清单的覆盖面靠 test_validate_sql_rejects_each_forbidden_keyword 逐个关键字钉住：
# 从清单里删/改任何一项，那个参数化用例就会失败——所以新增写关键字时
# 只要往元组里加，测试会自动跟着覆盖（这是"清单 + 参数化测试"配合的价值）。
# 已知的取舍：\b 是词边界匹配，所以**字符串字面量里的关键字也会被拒**
# （`WHERE person = 'DO'` 会被判成含写关键字）。对票据查询来说这种输入不存在，
# 用误拒换"绝不漏放写操作"是划算的。
FORBIDDEN_KEYWORDS = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "GRANT",
    "REVOKE",
    "COPY",
    "CALL",
    "DO",
    "MERGE",
    "ATTACH",
    "PRAGMA",
)

# re.IGNORECASE 是必需的：SQL 关键字大小写不敏感，`drop table` 和 `DROP TABLE` 等价。
# 用 \b(...)\b 包住整组关键字，靠词边界避免误杀 deleted_at / update_time（见上面的说明）。
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(FORBIDDEN_KEYWORDS) + r")\b", re.IGNORECASE)
# 只匹配数字形式的 LIMIT（`limit 500`），供 apply_row_limit 判断是否要收敛。
# 不匹配 `LIMIT ALL` / `LIMIT ?` 这类写法——真出现时 matched 为 None，
# 会被当成"没有 LIMIT"而**追加**一个 LIMIT row_limit，结果是语法错误但不会放过大数据量。
_LIMIT_RE = re.compile(r"\blimit\s+(\d+)\b", re.IGNORECASE)


def validate_sql(sql: str) -> str:
    """校验 SQL 只允许"单条 SELECT",返回去掉首尾空白与结尾分号的规范化 SQL。

    不通过时抛 SQLValidationError,异常消息里说明具体原因(便于调用方区分失败类型)。
    """
    # `sql or ""` 兜住 None：调用方把 None 传进来时也要走到"SQL 为空"这个明确的错误，
    # 而不是抛 TypeError（那是"代码写错了"的报错，会掩盖"没有生成出 SQL"这个事实）。
    raw = sql or ""
    text_sql = raw.strip()
    if not text_sql:
        raise SQLValidationError("SQL 为空")

    # 注释可以藏关键字(如 /*x*/DROP),先按字符直接拒绝,不做注释解析
    # 为什么不解析注释再检查：解析注释要处理嵌套、字符串里的伪注释等一堆边界，
    # 每漏一个边界就是一个绕过口子。票据查询不需要注释，直接拒绝最省事也最安全。
    # 副作用：`--` 出现在字符串里也会被拒（例如 `route like '%-%'`），同样属于可接受的取舍。
    if "--" in text_sql or "/*" in text_sql:
        raise SQLValidationError("SQL 含注释符号(-- 或 /*),为避免绕过关键字检查直接拒绝")

    # 按分号切分:过滤空片段后多于一条即多条语句
    # 用"先切分、再过滤空片段"的写法，是为了让结尾的单个分号（`SELECT 1;`）不被算成空语句，
    # 从而保持"结尾一个分号是合法惯用写法"这条兼容性（测试里钉住了）。
    statements = [part.strip() for part in text_sql.split(";") if part.strip()]
    if not statements:
        # 单独处理";"这种输入：切分后全是空片段，语义上就是"没有语句"。
        raise SQLValidationError("SQL 为空")
    if len(statements) > 1:
        # 报错里带上前 3 条，方便一眼看出是"注入式拼接"还是模型手滑。
        # 注意这里已经把 SQL 原文暴露在异常消息里——它只进日志/工具返回值，不外传。
        raise SQLValidationError(f"只允许单条语句,检测到 {len(statements)} 条: {statements[:3]}")

    statement = statements[0]

    # 必须是 SELECT 开头,或 WITH ... SELECT(CTE 只读查询)
    # CTE 必须**额外**要求文中含 SELECT：因为 PostgreSQL 允许
    # `WITH t AS (DELETE FROM tickets RETURNING *) SELECT * FROM t` 这种写法，
    # 它以 WITH 开头、却在 CTE 里真的删数据。下面是显式 if/elif/else 的写法
    # （而不是 re.match(...) or re.match(...)），就是为了让这两个条件各占一行、便于核查。
    if re.match(r"^SELECT\b", statement, re.IGNORECASE):
        pass
    elif re.match(r"^WITH\b", statement, re.IGNORECASE) and re.search(
        r"\bSELECT\b", statement, re.IGNORECASE
    ):
        pass
    else:
        # 截前 30 个字符：报错要能说明"实际是什么"，又不要把整条 SQL 重复一遍。
        # 这条也会兜住 EXPLAIN / SHOW 之类"只读但不以 SELECT 开头"的语句。
        raise SQLValidationError(f"只允许单条 SELECT 查询(可用 WITH ... SELECT),实际开头: {statement[:30]!r}")

    # 关键字检查放在最后：前面几条能把"多条语句"整批拒掉，
    # 到这里 statement 已经确定是单条，检查它即可。
    matched = _FORBIDDEN_RE.search(statement)
    if matched:
        raise SQLValidationError(f"SQL 含禁止的写关键字 {matched.group(1).upper()}")

    # 返回的是**规范化后的 statement**（已 strip、已去结尾分号），
    # 而不是原 raw：调用方拿它去做 apply_row_limit 时不必再处理格式。
    # 注意返回的是切分后的第一条，所以 `SELECT 1;` 返回 "SELECT 1"。
    return statement


def apply_row_limit(sql: str, row_limit: int) -> str:
    """保证查询结果行数不超过 row_limit 的纯函数。

    - 没有 LIMIT      → 追加 LIMIT row_limit
    - LIMIT 比上限小  → 尊重模型写的原值,不改写
    - LIMIT 比上限大  → 收敛到上限
    """
    # 纯函数（不碰数据库、无副作用）：这是它可以被单独测试、"执行前必过"的前提。
    # 为什么尊重更小的 LIMIT：模型写 `LIMIT 10` 往往是因为用户问的是"最近的几条"，
    # 强行改大到 50 会改变语义（多返回本不该返回的行）。
    matched = _LIMIT_RE.search(sql)
    if matched is None:
        # 追加在末尾。注意若原 SQL 以分号结尾会造成语法错误——但上游 validate_sql
        # 已经把结尾分号去掉了，所以这里是安全的。
        return f"{sql} LIMIT {row_limit}"
    if int(matched.group(1)) <= row_limit:
        return sql
    # 只替换数字本身,保留 LIMIT 前后的写法与大小写
    # 用 start(1)/end(1) 定位**捕获组**（那个数字）而不是整条匹配，
    # 这样 `limit` 的大小写、后面的换行、ORDER BY 位置全都原样保留。
    # 前提假设：只处理第一个 LIMIT。子查询里带 LIMIT 时收敛的可能不是外层那个——
    # 现有提示词不生成这种 SQL，真出现时要按括号层级定位。
    return f"{sql[: matched.start(1)]}{row_limit}{sql[matched.end(1) :]}"


# ============================================================
# 五、执行:强制行数上限 + JSON 可序列化
# ============================================================
def _jsonable(value):
    """把 Decimal / date 这类不能直接 json.dumps 的类型转成基础类型。

    AVG(amount_fen) 之类聚合在 PostgreSQL 里返回 Decimal,不转换会让调用方序列化失败。
    """
    # 为什么必须转：@tool 的返回值要经 LangChain 序列化后交给模型，
    # 而 Decimal / date 都不是 JSON 原生类型。不转的话，错误会发生在**工具之外**
    # （序列化那一层），报错信息里看不到是哪个字段惹的祸，极难定位。
    if isinstance(value, Decimal):
        # 整数就转 int（Decimal("1691.00") → 1691），有小数部分才转 float。
        # 这样金额字段在 JSON 里是 `1691` 而不是 `1691.0`，与"金额单位是分、必然是整数"
        # 的口径一致（模型看到带小数点的分值时容易当成元或做多余的换算）。
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, (datetime, date)):
        # 注意 datetime 要在 date 前面判断（datetime 是 date 的子类）。
        # isoformat() 输出 2025-03-05，是模型最容易正确理解日期语义的格式。
        return value.isoformat()
    return value


def query_tickets(sql: str, row_limit: int = 50) -> list[dict]:
    """执行只读 SQL 并强制行数上限,返回 JSON 可序列化的记录列表。

    这里再校验一次是信任边界的兜底:即使调用方绕过工具直接调用本函数,
    写操作也不会落到数据库上。SQL 合法但执行失败时抛 SQLExecutionError。
    """
    # 校验 → 限行 → 执行，顺序不能换：先限行再校验的话，
    # 恶意 SQL 在被拒之前就已经拼好了字符串（虽然没执行，但逻辑上不干净）。
    safe_sql = apply_row_limit(validate_sql(sql), row_limit)
    try:
        # 用 engine.connect() 而不是 engine.begin()：只读查询不需要事务，
        # 少一次 BEGIN/COMMIT 往返。with 保证连接归还连接池。
        with get_engine().connect() as conn:
            # .mappings().all() 把 Row 转成"类字典"（保留列名），
            # 下面 dict(row) 再转成真正的 dict——只有 dict 才能被后面的
            # `{key: _jsonable(value) for ...}` 与 json.dumps 正常处理。
            rows = [dict(row) for row in conn.execute(text(safe_sql)).mappings().all()]
    except SQLAlchemyError as exc:
        # 只捕获 SQLAlchemyError：它覆盖连接失败、SQL 语法错误、表不存在等**数据库侧**问题。
        # ⚠️ 已知缺口：若 settings.finance_pg_url 漏了 "+psycopg"，create_engine 会抛
        # ModuleNotFoundError（实测 sqlalchemy 2.0.52 行为），它不是 SQLAlchemyError，
        # 会穿透这里与 query_ticket_db 的 except SQLExecutionError，表现为工具直接抛异常，
        # 而不是返回 db_error。改连接串时务必保留 "+psycopg"。
        logger.error(f"[Text-to-SQL] SQL 执行失败: {safe_sql} | {exc}")
        # `from exc` 保留原始异常链：日志里能看到 libpq 的原始报错，
        # 而调用方拿到的是统一的 SQLExecutionError 类型。
        raise SQLExecutionError(f"数据库执行失败: {exc}") from exc
    # info 级别（不是 debug）：真正执行过的 SQL 属于审计信息，
    # 出问题时这是唯一能确认"当时到底执行了什么"的记录。
    logger.info(f"[Text-to-SQL] 执行完成,返回 {len(rows)} 行 | {safe_sql}")
    # 逐行逐字段过 _jsonable：不能在别处补，因为这里是"数据离开数据库"的唯一出口。
    return [{key: _jsonable(value) for key, value in row.items()} for row in rows]


def _is_empty_result(rows: list[dict]) -> bool:
    """判断"聚合查询命中 0 条记录"。

    SELECT SUM(x) ... 没有匹配记录时 PostgreSQL 仍会返回 1 行,只是值为 NULL。
    这本质上仍是没查到,不能因为多了一行 NULL 就报 ok。
    只有"恰好 1 行且该行所有字段都是 NULL"才算空;多个字段里只要有一个非空,
    就说明确实查到了数据。
    """
    # 为什么"多个字段里有一个非空就算查到了"：`SELECT SUM(x), COUNT(*)` 在没有匹配时
    # 返回 (NULL, 0)，COUNT 的 0 是**真实结果**（确实有 0 张票），所以应报 ok。
    # 反过来只有 `SUM(x)` 单字段 NULL 时无法区分"没匹配"和"匹配了但值全为 NULL"，
    # 这里按"没匹配"处理——对票据统计来说，后者不存在（金额不会真的是 NULL）。
    # 边界：rows 为空列表时 len(rows) == 1 为假 → 返回 False，
    # 空结果由调用方的 `if not rows` 分支处理，两个分支各管一种"空"，不要合并。
    return len(rows) == 1 and all(value is None for value in rows[0].values())


# ============================================================
# 六、LangChain 工具:只读票据查询入口
# ============================================================
@tool
def query_ticket_db(question: str) -> dict:
    """查询结构化票据库(金额汇总、按人员/日期/票据类型精确筛选)。

    适合"一共有多少张火车票""黄帅的火车票一共多少钱""2025 年 3 月的发票总额是多少"
    这类需要精确计数或求和的问题:内部把问题翻译成 SQL,只允许单条 SELECT,
    执行时强制行数上限。

    Args:
        question: 用户的自然语言问题。

    Returns:
        三种结果必须分开表达,不能笼统当成"没查到":
        - ok:附带 rows 与 row_count
        - no_result:SQL 执行成功但没有任何记录
        - invalid_sql:生成的 SQL 没通过只读校验(附带 sql 与 detail)
        - db_error:SQL 合法但数据库执行失败(附带 sql 与 detail)
        - llm_error:生成 SQL 这一步就失败了(附带 detail)

    返回值里统一带 question 字段：这是给**模型**看的上下文（工具返回值会进对话），
    多轮里模型能据此确认"我查的是哪个问题"，避免把上一次的结果当成这一次的。
    invalid_sql / db_error 还会带上 sql（模型生成的原文或规范化后的版本），
    让模型有机会自己看出"我写错了什么"并改写——但当前提示词只允许它改用向量检索，
    不给它重试 SQL 的机会（除非用户再问一次）。
    """
    # 三段 try/except 而不是一个大的：每一段失败对应一个**不同的状态**，
    # 合并捕获就再也分不出是"没生成出 SQL"还是"数据库没起"。
    try:
        sql = text_to_sql(question)
    except Exception as exc:  # LLM 网络/鉴权/超时等,属于系统故障
        # 这里捕 Exception（而不是具体异常）：外部 API 的失败类型不可枚举
        # （连接错误、鉴权错误、限流、超时各自是不同异常类），穷举必然漏。
        # 代价是连"代码 bug"（如 settings 属性名写错）也会被记成 llm_error——
        # 排查时看日志里的 exc 类型就能区分。
        logger.error(f"[Text-to-SQL] 生成 SQL 失败: {exc}")
        return {"status": "llm_error", "question": question, "detail": f"SQL 生成失败: {exc}"}

    try:
        # 校验失败时 sql 仍是模型的原始输出,原样回传便于排查
        valid_sql = validate_sql(sql)
    except SQLValidationError as exc:
        # 这里只捕 SQLValidationError（校验是本模块自己抛的，能精确捕获）；
        # warning 而不是 error：模型偶尔写出不合规的 SQL 属于预期内的正常波动，
        # 不是系统故障，不该拉响 error 级别的告警。
        logger.warning(f"[Text-to-SQL] SQL 未通过校验: {sql} | {exc}")
        return {"status": "invalid_sql", "question": question, "sql": sql, "detail": str(exc)}

    try:
        rows = query_tickets(valid_sql)
    except SQLExecutionError as exc:
        # 只捕 SQLExecutionError：见 query_tickets 的说明——
        # 连接串漏了 +psycopg 时抛的 ModuleNotFoundError 会穿透出这个工具，
        # 那是已知缺口，不是"数据库执行失败"。
        logger.error(f"[Text-to-SQL] 数据库执行失败: {valid_sql} | {exc}")
        return {"status": "db_error", "question": question, "sql": valid_sql, "detail": str(exc)}

    # 两种"空"要分别判：真·零行，以及聚合返回的一行 NULL（见 _is_empty_result）。
    # 两者都归为 no_result，但 detail 写得不一样——排查时能看出是哪一种。
    if not rows:
        return {
            "status": "no_result",
            "question": question,
            "sql": valid_sql,
            "detail": "查询执行成功,但没有匹配的记录",
        }

    if _is_empty_result(rows):
        return {
            "status": "no_result",
            "question": question,
            "sql": valid_sql,
            "detail": "查询执行成功,但没有匹配的记录(聚合结果为 NULL)",
        }

    # 只有走到这里才报 ok。ok 一定带 rows 与 row_count——
    # RAG_SYSTEM_PROMPT 要求模型"写明统计条件与来源"，row_count 就是它报数时的依据。
    return {
        "status": "ok",
        "question": question,
        "sql": valid_sql,
        "row_count": len(rows),
        "rows": rows,
    }

"""Text-to-SQL 测试:SQL 校验 / 行数上限改写 / SQL 清洗 / 结果状态区分。

离线用例不依赖 LLM 与数据库(monkeypatch 掉模型调用与数据库引擎);
需要真实 PostgreSQL 的用例标 integration,跑之前要先执行导入脚本
`uv run python RAG/script/import_tickets_to_pg.py`。
"""

import json
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy.exc import OperationalError

from service_probe import requires_pg
from agentic import text_to_sql as t2s  # t2s 是模块(agentic/text_to_sql.py),不是同名函数


# ============================================================
# 一、validate_sql:放行合法的单条只读查询
# ============================================================
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM tickets",
        "select count(*) from tickets where ticket_type = 'train'",
        "   SELECT person, SUM(amount_fen) FROM tickets GROUP BY person   ",
        "SELECT * FROM tickets;",  # 结尾一个分号是惯用写法,应当放行
        "WITH t AS (SELECT 1 AS x) SELECT x FROM t",  # WITH ... SELECT 是只读查询
        "with t as (select * from tickets) select count(*) from t;",
    ],
)
def test_validate_sql_accepts_single_select(sql):
    assert t2s.validate_sql(sql)


def test_validate_sql_returns_normalized_sql():
    # 去掉首尾空白与结尾分号后返回,便于后续统一改写 LIMIT
    assert t2s.validate_sql("  SELECT 1 ; ") == "SELECT 1"


# ============================================================
# 二、validate_sql:拒绝多条语句 / 非 SELECT / 写操作
# ============================================================
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1; DROP TABLE tickets;",  # 经典的注入式拼接
        "SELECT 1; SELECT 2",
        "SELECT * FROM tickets; DELETE FROM tickets",
    ],
)
def test_validate_sql_rejects_multi_statement(sql):
    with pytest.raises(t2s.SQLValidationError):
        t2s.validate_sql(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE tickets SET amount_fen = 0",
        "DROP TABLE tickets",
        "WITH t AS (DELETE FROM tickets RETURNING *) SELECT * FROM t",  # WITH 里也必须含 SELECT 且无写关键字
        "EXPLAIN SELECT * FROM tickets",
        "SHOW TABLES",
    ],
)
def test_validate_sql_rejects_non_select(sql):
    with pytest.raises(t2s.SQLValidationError):
        t2s.validate_sql(sql)


@pytest.mark.parametrize("keyword", t2s.FORBIDDEN_KEYWORDS)
def test_validate_sql_rejects_each_forbidden_keyword(keyword):
    # 每个写关键字单独覆盖:漏掉任何一个都算校验缺口
    with pytest.raises(t2s.SQLValidationError, match=keyword):
        t2s.validate_sql(f"SELECT {keyword} FROM tickets")


@pytest.mark.parametrize("keyword", ["insert", "Drop", "tRuNcAtE"])
def test_validate_sql_keyword_match_is_case_insensitive(keyword):
    with pytest.raises(t2s.SQLValidationError):
        t2s.validate_sql(f"select * from tickets where {keyword} = 1")


@pytest.mark.parametrize(
    "identifier",
    [
        "SELECT deleted_at, update_time, created_at FROM tickets",
        "SELECT * FROM tickets WHERE update_time > 1 AND deleted_at IS NULL",
        "SELECT copy_count, call_id, merge_flag FROM tickets",  # 关键字做前缀的普通列名
    ],
)
def test_validate_sql_does_not_kill_identifiers_containing_keywords(identifier):
    # 关键词必须按独立词匹配,否则 deleted_at / update_time 这类正常列名会被误杀
    assert t2s.validate_sql(identifier)


# ============================================================
# 三、validate_sql:注释与空 SQL
# ============================================================
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM tickets -- 只看火车票",
        "SELECT * FROM tickets /* 注释 */",
        "SELECT * /* 中间注释 */ FROM tickets",
    ],
)
def test_validate_sql_rejects_comments(sql):
    # 注释可以藏关键字,直接拒绝,避免绕过关键字检查
    with pytest.raises(t2s.SQLValidationError):
        t2s.validate_sql(sql)


@pytest.mark.parametrize("sql", ["", "   ", "\n\t ", ";"])
def test_validate_sql_rejects_empty(sql):
    with pytest.raises(t2s.SQLValidationError):
        t2s.validate_sql(sql)


# ============================================================
# 四、apply_row_limit:强制行数上限的纯函数
# ============================================================
def test_apply_row_limit_appends_when_missing():
    assert t2s.apply_row_limit("SELECT * FROM tickets", 50) == "SELECT * FROM tickets LIMIT 50"


@pytest.mark.parametrize("sql", ["SELECT * FROM tickets LIMIT 10", "select * from tickets limit 10"])
def test_apply_row_limit_keeps_smaller_limit(sql):
    assert t2s.apply_row_limit(sql, 50) == sql


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT * FROM tickets LIMIT 500", "SELECT * FROM tickets LIMIT 50"),
        ("select * from tickets limit 500", "select * from tickets limit 50"),
        ("SELECT * FROM tickets ORDER BY amount_fen DESC LIMIT 1000", "SELECT * FROM tickets ORDER BY amount_fen DESC LIMIT 50"),
    ],
)
def test_apply_row_limit_shrinks_larger_limit(sql, expected):
    assert t2s.apply_row_limit(sql, 50) == expected


def test_apply_row_limit_keeps_equal_limit():
    sql = "SELECT * FROM tickets LIMIT 50"
    assert t2s.apply_row_limit(sql, 50) == sql


# ============================================================
# 五、表结构描述:名称 + 中文说明,逗号连接(拼给 LLM 的提示词用)
# ============================================================
def test_describe_table_lists_every_column_with_description():
    parts = [p.strip() for p in t2s.describe_table().split("，")]
    assert [p.split(" ")[0] for p in parts] == list(t2s.Ticket.model_fields)
    # 每一段都必须"名称 + 空格 + 非空说明",供模型理解列含义
    assert all(len(p.split(" ", 1)) == 2 and p.split(" ", 1)[1] for p in parts)


def test_ticket_table_name_and_primary_key():
    assert t2s.Ticket.__tablename__ == "tickets"
    assert t2s.Ticket.model_fields["ticket_no"].is_required()


# ============================================================
# 六、text_to_sql:清洗模型输出(monkeypatch 掉 LLM 调用)
# ============================================================
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("```sql\nSELECT COUNT(*) FROM tickets;\n```", "SELECT COUNT(*) FROM tickets"),
        ("```\nSELECT 1\n```", "SELECT 1"),
        ("SELECT 1;", "SELECT 1"),
        ("  SELECT 1 ;  ", "SELECT 1"),
        ("SELECT 1", "SELECT 1"),
    ],
)
def test_text_to_sql_cleans_llm_output(monkeypatch, raw, expected):
    monkeypatch.setattr(t2s, "_invoke_llm", lambda prompt: raw)
    assert t2s.text_to_sql("随便问一句") == expected


def test_text_to_sql_prompt_carries_schema_and_question(monkeypatch):
    seen = {}

    def fake_llm(prompt: str) -> str:
        seen["prompt"] = prompt
        return "SELECT 1"

    monkeypatch.setattr(t2s, "_invoke_llm", fake_llm)
    t2s.text_to_sql("一共有多少张车票")
    assert "tickets" in seen["prompt"]
    assert "ticket_type" in seen["prompt"]  # 表结构字段要进提示词
    assert "一共有多少张车票" in seen["prompt"]
    assert "只输出" in seen["prompt"]  # 要求模型只输出 SQL 本身


# ============================================================
# 七、query_tickets:强制行数上限 + 返回 JSON 可序列化
# ============================================================
class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self

    def all(self):
        return self._rows


class _FakeConnection:
    """记录实际执行到的 SQL,并返回预置行"""

    def __init__(self, rows, executed):
        self._rows = rows
        self._executed = executed

    def execute(self, statement):
        self._executed.append(str(statement))
        return _FakeResult(self._rows)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeEngine:
    def __init__(self, rows, executed):
        self._rows = rows
        self._executed = executed

    def connect(self):
        return _FakeConnection(self._rows, self._executed)


@pytest.fixture
def executed_sql(monkeypatch):
    """把 query_tickets 用的引擎换成假引擎,返回记录执行的 SQL 列表"""
    executed: list[str] = []
    monkeypatch.setattr(t2s, "get_engine", lambda: _FakeEngine([], executed))
    return executed


def test_query_tickets_applies_row_limit(executed_sql):
    t2s.query_tickets("SELECT * FROM tickets", row_limit=5)
    assert executed_sql == ["SELECT * FROM tickets LIMIT 5"]


def test_query_tickets_returns_json_serializable_rows(monkeypatch):
    rows = [{"person": "黄帅", "total": Decimal("1691.00"), "cnt": 2, "day": date(2025, 3, 5)}]
    executed: list[str] = []
    monkeypatch.setattr(t2s, "get_engine", lambda: _FakeEngine(rows, executed))

    result = t2s.query_tickets("SELECT person, SUM(amount_fen) AS total FROM tickets GROUP BY person")

    assert json.loads(json.dumps(result)) == [
        {"person": "黄帅", "total": 1691.0, "cnt": 2, "day": "2025-03-05"}
    ]


def test_query_tickets_refuses_write_sql(executed_sql):
    # 信任边界:即使调用方绕过工具直接调 query_tickets,也不允许写操作落库
    with pytest.raises(t2s.SQLValidationError):
        t2s.query_tickets("DROP TABLE tickets")
    assert executed_sql == []


def test_query_tickets_wraps_db_failure(monkeypatch):
    def broken_engine():
        raise OperationalError("SELECT 1", {}, Exception("连接被拒绝"))

    monkeypatch.setattr(t2s, "get_engine", broken_engine)
    with pytest.raises(t2s.SQLExecutionError) as excinfo:
        t2s.query_tickets("SELECT 1")
    assert "连接被拒绝" in str(excinfo.value)


# ============================================================
# 八、query_ticket_db 工具:四种结果分开表达
# ============================================================
def test_query_ticket_db_tool_description_is_chinese():
    assert t2s.query_ticket_db.name == "query_ticket_db"
    assert "查询结构化票据库" in t2s.query_ticket_db.description
    assert "金额汇总" in t2s.query_ticket_db.description


def test_query_ticket_db_returns_ok_with_rows(monkeypatch):
    monkeypatch.setattr(t2s, "text_to_sql", lambda q: "SELECT person FROM tickets")
    monkeypatch.setattr(t2s, "query_tickets", lambda sql, row_limit=50: [{"person": "黄帅"}])

    result = t2s.query_ticket_db.invoke({"question": "谁是黄帅"})

    assert result["status"] == "ok"
    assert result["rows"] == [{"person": "黄帅"}]
    assert result["row_count"] == 1


def test_query_ticket_db_reports_no_result(monkeypatch):
    monkeypatch.setattr(t2s, "text_to_sql", lambda q: "SELECT person FROM tickets WHERE person = '张三'")
    monkeypatch.setattr(t2s, "query_tickets", lambda sql, row_limit=50: [])

    result = t2s.query_ticket_db.invoke({"question": "张三的火车票一共多少钱"})

    assert result["status"] == "no_result"
    assert result["detail"]


def test_query_ticket_db_reports_no_result_for_null_aggregate(monkeypatch):
    # 聚合查询命中 0 条时 PostgreSQL 仍返回 1 行,只是值为 NULL;
    # "没匹配到记录"不能因为多了一行 NULL 就被当成 ok
    monkeypatch.setattr(
        t2s,
        "text_to_sql",
        lambda q: "SELECT SUM(amount_fen) AS total FROM tickets WHERE person = '张三'",
    )
    monkeypatch.setattr(t2s, "query_tickets", lambda sql, row_limit=50: [{"total": None}])

    result = t2s.query_ticket_db.invoke({"question": "张三的火车票一共多少钱"})

    assert result["status"] == "no_result"


def test_query_ticket_db_keeps_ok_when_row_has_any_value(monkeypatch):
    # 只要这一行还有非空字段,就说明确实查到了数据,不能误判成 no_result
    monkeypatch.setattr(
        t2s, "text_to_sql", lambda q: "SELECT SUM(amount_fen) AS total, COUNT(*) AS cnt FROM tickets"
    )
    monkeypatch.setattr(t2s, "query_tickets", lambda sql, row_limit=50: [{"total": None, "cnt": 0}])

    result = t2s.query_ticket_db.invoke({"question": "有没有金额"})

    assert result["status"] == "ok"


def test_query_ticket_db_reports_invalid_sql_without_touching_db(monkeypatch):
    monkeypatch.setattr(t2s, "text_to_sql", lambda q: "SELECT 1; DROP TABLE tickets;")

    def must_not_run(sql, row_limit=50):
        raise AssertionError("校验失败的 SQL 不允许执行")

    monkeypatch.setattr(t2s, "query_tickets", must_not_run)

    result = t2s.query_ticket_db.invoke({"question": "把票据表删了"})

    assert result["status"] == "invalid_sql"
    assert "DROP" in result["detail"] or "多条语句" in result["detail"]
    assert result["sql"] == "SELECT 1; DROP TABLE tickets;"


def test_query_ticket_db_reports_db_error(monkeypatch):
    monkeypatch.setattr(t2s, "text_to_sql", lambda q: "SELECT 1")

    def broken(sql, row_limit=50):
        raise t2s.SQLExecutionError("数据库连接失败: connection refused")

    monkeypatch.setattr(t2s, "query_tickets", broken)

    result = t2s.query_ticket_db.invoke({"question": "有多少张票"})

    # 系统故障不能被伪装成"没查到"
    assert result["status"] == "db_error"
    assert "connection refused" in result["detail"]


def test_query_ticket_db_reports_llm_error(monkeypatch):
    def broken_llm(question):
        raise RuntimeError("LLM 接口超时")

    monkeypatch.setattr(t2s, "text_to_sql", broken_llm)

    result = t2s.query_ticket_db.invoke({"question": "有多少张票"})

    # 生成 SQL 这一步失败同样是系统故障,不能报 no_result
    assert result["status"] == "llm_error"
    assert "LLM 接口超时" in result["detail"]


# ============================================================
# 九、真实 PostgreSQL 集成用例(需先执行 RAG/script/import_tickets_to_pg.py
#     把 Milvus 的 300 条票据导入 finance 库的 tickets 表)
#     PostgreSQL 不在时由 requires_pg 快速跳过——数据库栈不随 Docker 自启,
#     没有这层守卫时套件会卡到 libpq 的连接超时(约 130 秒)并跑红。
# ============================================================
@requires_pg
@pytest.mark.integration
def test_real_pg_table_holds_all_tickets():
    rows = t2s.query_tickets("SELECT COUNT(*) AS cnt FROM tickets")
    assert rows[0]["cnt"] == 300


@requires_pg
@pytest.mark.integration
def test_real_pg_counts_train_tickets():
    rows = t2s.query_tickets("SELECT COUNT(*) AS cnt FROM tickets WHERE ticket_type = 'train'")
    assert rows[0]["cnt"] == 100


@requires_pg
@pytest.mark.integration
def test_real_pg_sums_amount_for_person():
    # 黄帅在 Milvus 里有 2 张火车票,合计 169100 分
    rows = t2s.query_tickets(
        "SELECT SUM(amount_fen) AS total FROM tickets"
        " WHERE ticket_type = 'train' AND person = '黄帅'"
    )
    assert rows[0]["total"] == 169100


@requires_pg
@pytest.mark.integration
def test_real_pg_row_limit_is_forced():
    assert len(t2s.query_tickets("SELECT * FROM tickets", row_limit=7)) == 7


@requires_pg
@pytest.mark.integration
def test_real_pg_avg_stays_json_serializable():
    rows = t2s.query_tickets(
        "SELECT AVG(amount_fen) AS avg_fen FROM tickets WHERE ticket_type = 'train'"
    )
    json.dumps(rows)  # AVG 在 PG 里返回 Decimal,不转换会在这里抛 TypeError
    assert isinstance(rows[0]["avg_fen"], float)

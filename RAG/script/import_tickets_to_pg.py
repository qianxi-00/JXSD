"""把 Milvus 里的票据结构化字段导入 PostgreSQL 的 tickets 表(幂等,可重复执行)。

用法(在 Python_Base 目录下):
    uv run python RAG/script/import_tickets_to_pg.py

幂等靠 session.merge():主键存在就更新,不存在才插入,重复执行不会产生重复行。

**为什么要有这一步**(Milvus 已经是票据库了):
两条链路读的是**不同存储**——向量检索走 Milvus(要的是相似度),
而结构化统计/Text-to-SQL(agentic 的 `ticket_db` 工具)走 PostgreSQL(要的是 SQL)。
同一个仓库里同一个"票据"概念存了两份,所以**导数据这一步必须幂等且可重跑**,
否则两张表的数据会悄悄分叉(比如 Milvus 重算过向量、PG 还是旧的)。

⚠ 与 Milvus 侧的已知差异(不是本脚本的锅,但会影响下游):
Milvus 的 300 条里有 **18 条 OCR 失败记录**照单入库,本脚本按同样的过滤条件
(`ticket_type in [...]`)拉取 ⇒ 这 18 条也会进 PG,只是它们的字段大多是空的
(被下面的 `or ""` / `or 0` 兜成空值/0)。做金额统计时要注意这些 0 值记录。
"""

import sys
from collections import Counter
from pathlib import Path

# 以脚本方式直接运行时,sys.path[0] 是 RAG/script,找不到 config 与 agentic,先挂上两个根目录
# (注意这里用的是「不存在才插入」的写法,与 run_stage_eval 等脚本的无条件 insert(0,...) 不同:
#  重复 import 本模块时不会反复往 sys.path 前面塞路径)
_RAG_ROOT = Path(__file__).resolve().parent.parent  # RAG
_PYTHON_BASE = _RAG_ROOT.parent  # Python_Base(根目录 config.py 所在处)
for _path in (str(_PYTHON_BASE), str(_RAG_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Windows 控制台默认按本地代码页输出,中文会变成乱码;显式按 UTF-8 输出
# ⚠ 这里只重设了 **encoding**,没有设 errors="replace":遇到无法编码的字符
# 会直接抛 UnicodeEncodeError(而 langfuse_evaluation.py 里设了 errors="replace"
# 来做兜底)。本脚本打印的内容都是票号/人名/数字,实践中没触发过,但口径确实不一致(见报告)
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from sqlmodel import Session, SQLModel, func, select  # noqa: E402

from agentic.text_to_sql import Ticket, get_engine  # noqa: E402
from config import settings  # noqa: E402
from core.database import get_milvus_client  # noqa: E402
from core.logger import logger  # noqa: E402

# 只导入三类票据;机票/发票/火车票正好各 100 条
MILVUS_FILTER = 'ticket_type in ["flight","invoice","train"]'
# 与 retrieval/vector_retrieval.OUTPUT_FIELDS 对齐的结构化字段(语义文本与向量不入库)
# ⇒ 刻意**不导入 semantic_text/ocr_text**:PG 侧只做结构化查询,
#   把上万字的 OCR 正文搬进 PG 既浪费空间又没人读
OUTPUT_FIELDS = ["ticket_no", "ticket_type", "person", "date_int", "amount_fen", "route"]


def fetch_tickets(limit: int = 10000) -> list[dict]:
    """从 Milvus 拉全量票据的结构化字段(带 id,票号兜底要用)

    `output_fields=OUTPUT_FIELDS + ["id"]` —— 多要一个 `id` 不只是为了留档:
    下面 `build_tickets` 在票号缺失/重复时**用它当主键兜底**,所以 id 是必需字段,
    不是可选的调试信息。

    `limit=10000` 同样是硬上限而不是分页:超过会静默截断(见报告)。
    日志里打读到的条数,便于与"写入条数"对照发现截断。
    """
    client = get_milvus_client()
    rows = client.query(
        collection_name=settings.milvus.collection,
        filter=MILVUS_FILTER,
        output_fields=OUTPUT_FIELDS + ["id"],
        limit=limit,
    )
    logger.info(f"[导入] Milvus 读到 {len(rows)} 条票据")
    return rows


def build_tickets(rows: list[dict]) -> tuple[list[Ticket], int]:
    """把 Milvus 记录转成 Ticket 行,返回 (票据列表, 票号兜底条数)。

    实测数据里有两个坑,必须在这里兜住:
      1. 票号可能为空(Milvus 里 19 条),或整批共用一个占位值(机票 82 条票号都是
         8762369777769)。tickets.ticket_no 是主键,必须非空且唯一,否则 merge 会
         把不同票据当成同一条,300 条最后只剩 200 条。
      2. 数字列可能为空(机票的日期与金额 100 条全空)。date_int / amount_fen 是
         NOT NULL 整数列,None 直接插入会报错,按课案对 person/route 的同样口径
         兜成 0(表示"该票据未采集到该字段")。

    票号是否可用只看"整批数据里出现了几次",与遍历顺序无关,所以重复执行结果一致。

    ★ 判重的实现值得单独说:`Counter` **先扫全量再决定**(而不是边遍历边判
    "这个票号见过没有")。两者结果不同 —— 全部机票共用同一个占位票号时,
    "边遍历边判" 会把**第一条**当合法票号留下(它当时是第一次出现),
    只把后面 81 条换成 id 兜底 ⇒ 结果依赖行顺序、且混入了假票号。
    先统计全量出现次数,才能把"这一批里所有同值票号"都判为不可用。
    注释里说"与遍历顺序无关",靠的就是这一句先统计。

    `fallback` 计数返回给调用方打印:这个数字**应当>0**(实测机票有 82 条走兜底),
    若某天变成 0 反而要怀疑"Milvus 的票号字段是不是被换了来源"。
    """
    counts = Counter(row.get("ticket_no") for row in rows if row.get("ticket_no"))
    tickets: list[Ticket] = []
    fallback = 0
    for row in rows:
        ticket_no = row.get("ticket_no")
        # `counts[ticket_no] > 1` 用 Counter 的**下标访问**而不是 .get(...):
        # Counter 对不存在的键返回 0(不抛 KeyError),所以配合前面的 or 判断是安全的
        if not ticket_no or counts[ticket_no] > 1:
            ticket_no = row["id"]  # 兜底:Milvus 主键(id)全局唯一且稳定
            fallback += 1
        tickets.append(
            Ticket(
                ticket_no=ticket_no,
                # 下面四处 `or 默认值` 同时兜住 None 与空字符串两种"缺失"写法 ——
                # 这就是注释里说的"按课案口径兜 0/空串"的落地处
                person=row.get("person") or "",
                ticket_type=row.get("ticket_type") or "",
                date_int=row.get("date_int") or 0,
                amount_fen=row.get("amount_fen") or 0,
                route=row.get("route") or "",
            )
        )
    return tickets, fallback


def main() -> None:
    rows = fetch_tickets()
    tickets, fallback = build_tickets(rows)

    engine = get_engine()
    # `create_all` 是幂等的(表已存在则跳过),所以脚本可以重复跑;
    # 注意它**不会**做 schema 迁移 —— 若 Ticket 模型加了字段,
    # 已存在的表不会被改成新结构,新字段会静默缺失(见报告)
    SQLModel.metadata.create_all(engine)  # 表已存在则跳过,不破坏已有数据

    with Session(engine) as session:
        for ticket in tickets:
            session.merge(ticket)  # 主键存在则更新,否则插入 → 幂等
        # commit 在循环**之外**:300 条一次性提交,避免每条一次事务的往返开销。
        # 代价是中途失败会整批回滚(对本场景是好事:不会留下半批数据)
        session.commit()
        # 总数查询放在**同一个 session 内**、commit 之后:能看到本次写入的结果。
        # `select(func.count()).select_from(Ticket)` 是 SQLModel 写 COUNT(*) 的
        # 推荐形式(`select(Ticket)` 会把整表读进内存)
        total = session.exec(select(func.count()).select_from(Ticket)).one()

    # 四个数字构成一次完整的对账:读了多少 / 写了多少 / 兜底多少 / 表里最终多少。
    # "本次写入"与"表内总数"不相等是**正常**的(merge 会更新已存在的行),
    # 但若"表内总数"小于"Milvus 读取条数",就说明票号兜底没兜住(多行合并成一行了)
    print(f"Milvus 读取票据: {len(rows)} 条")
    print(f"本次写入(merge): {len(tickets)} 条,其中票号缺失/重复改用 Milvus id 兜底: {fallback} 条")
    print(f"tickets 表内总数: {total} 条")
    print(f"目标库: {settings.finance_db} | 表: {Ticket.__tablename__}")


if __name__ == "__main__":
    main()

"""crud.py —— 数据访问层
================================================================
对应课案章节：后端开发基础 → Web框架 → FastAPI（数据库集成）

**这一层只做数据库操作**，不认识 HTTP、不认识 Pydantic 请求对象，
接收的是"普通参数"（书名、ID、分页数字），返回的是 ORM 对象或 None。

好处：
    · 路由层变薄，只负责"解析请求 → 调用 crud → 包装响应"；
    · 单元测试可以直接调 crud，不需要起 HTTP 服务；
    · 将来从 SQLite 换成 MySQL、或从同步 ORM 换成异步 ORM，
      改动局限在这一层。
"""

from __future__ import annotations

import pathlib
import sys

from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

_BOOK_API_DIR = pathlib.Path(__file__).resolve().parent
if str(_BOOK_API_DIR) not in sys.path:
    sys.path.insert(0, str(_BOOK_API_DIR))

import models  # noqa: E402
import schemas  # noqa: E402


# ============================================================
# 查询
# ============================================================
def get_book(db: Session, book_id: int) -> models.Book | None:
    """按主键查询。

    db.get(Model, pk) 是 2.0 推荐的按主键取单条方式：
    它会先查"身份映射"（同一会话内同一主键只对应一个对象），命中则不发 SQL。
    """
    return db.get(models.Book, book_id)


def get_book_by_isbn(db: Session, isbn: str) -> models.Book | None:
    """按 ISBN 查询（用于唯一性校验）。"""
    stmt = select(models.Book).where(models.Book.isbn == isbn)
    return db.execute(stmt).scalar_one_or_none()


def list_books(
    db: Session,
    *,
    page: int = 1,
    size: int = 10,
    keyword: str | None = None,
    order_by: str = "id",
    desc: bool = False,
) -> tuple[int, list[models.Book]]:
    """分页 + 关键字搜索，返回 (总数, 当前页数据)。

    实现要点：
        · 先算总数（单独的 count 查询），再查当前页 —— 这样前端能显示"共 N 条"；
        · 关键字同时匹配书名与作者，用 or_ 组合条件；
        · 排序字段用白名单映射，**绝不能**把前端传来的字符串直接拼进 SQL
          （那是 SQL 注入的经典入口）。
    """
    sort_columns = {
        "id": models.Book.id,
        "title": models.Book.title,
        "price": models.Book.price,
        "stock": models.Book.stock,
        "created_at": models.Book.created_at,
    }
    column = sort_columns.get(order_by, models.Book.id)

    conditions = []
    if keyword:
        like = f"%{keyword.strip()}%"
        conditions.append(or_(models.Book.title.ilike(like), models.Book.author.ilike(like)))

    count_stmt = select(func.count()).select_from(models.Book)
    list_stmt = select(models.Book)
    for cond in conditions:
        count_stmt = count_stmt.where(cond)
        list_stmt = list_stmt.where(cond)

    total = db.execute(count_stmt).scalar_one()
    list_stmt = list_stmt.order_by(column.desc() if desc else column.asc())
    list_stmt = list_stmt.offset((page - 1) * size).limit(size)
    items = list(db.execute(list_stmt).scalars().all())
    return total, items


# ============================================================
# 写入
# ============================================================
def create_book(db: Session, payload: schemas.BookCreate) -> models.Book:
    """新增图书。

    三步走：add（放入会话）→ commit（提交事务）→ refresh（读回数据库生成的值，
    例如自增主键与 server_default 的 created_at）。
    """
    book = models.Book(**payload.model_dump())
    db.add(book)
    db.commit()
    db.refresh(book)
    return book


def update_book(db: Session, book: models.Book, payload: schemas.BookUpdate) -> models.Book:
    """全量更新（PUT）。"""
    for field, value in payload.model_dump().items():
        setattr(book, field, value)
    db.add(book)
    db.commit()
    db.refresh(book)
    return book


def patch_book(db: Session, book: models.Book, payload: schemas.BookPatch) -> models.Book:
    """局部更新（PATCH）。

    exclude_unset=True 很关键：只更新"客户端真的传了"的字段，
    否则会把没传的字段全部写成 None。
    """
    data = payload.model_dump(exclude_unset=True, exclude_none=True)
    for field, value in data.items():
        setattr(book, field, value)
    db.add(book)
    db.commit()
    db.refresh(book)
    return book


def delete_book(db: Session, book: models.Book) -> None:
    """删除图书。"""
    db.delete(book)
    db.commit()


def count_all(db: Session) -> int:
    """统计总数（供 /health 使用）。"""
    return db.execute(select(func.count()).select_from(models.Book)).scalar_one()


if __name__ == "__main__":
    # 单独运行本模块时的自检：不经过 HTTP，直接测数据访问层
    # （这正是分层的收益之一：业务逻辑可以脱离 Web 框架被测试）
    import database

    print("=" * 72)
    print("crud.py 自检（数据访问层，不经过 HTTP）")
    print("=" * 72)
    print(f"  数据库：{database.describe()['url']}")
    database.init_db()

    isbn = "978-7-888-00001-1"
    with database.SessionLocal() as db:
        # 先清掉可能残留的同 ISBN 数据，保证脚本可重复运行
        leftover = get_book_by_isbn(db, isbn)
        if leftover is not None:
            delete_book(db, leftover)

        print("\n  【1】create_book")
        payload = schemas.BookCreate(title="crud 自检用书", author="自检",
                                     isbn=isbn, price=59.9, stock=7)
        book = create_book(db, payload)
        print(f"      新增成功：id={book.id} title={book.title!r} "
              f"created_at={book.created_at}")
        assert book.id > 0

        print("\n  【2】get_book / get_book_by_isbn")
        got = get_book(db, book.id)
        print(f"      db.get 查回：{got!r}")
        assert got is not None and got.id == book.id
        by_isbn = get_book_by_isbn(db, isbn)
        print(f"      按 ISBN 查回：{by_isbn!r}")
        assert by_isbn is not None

        print("\n  【3】list_books（分页 + 关键字 + 排序）")
        total, items = list_books(db, page=1, size=3, keyword=None, order_by="price", desc=True)
        print(f"      总数={total}，本页 {len(items)} 条："
              f"{[f'{b.title}({b.price})' for b in items]}")
        total_kw, _ = list_books(db, keyword="自检")
        print(f"      关键字 '自检' 命中 {total_kw} 条")

        print("\n  【4】patch_book（局部更新）")
        patched = patch_book(db, got, schemas.BookPatch(stock=99))
        print(f"      stock 改为 {patched.stock}，title 保持 {patched.title!r}")
        assert patched.stock == 99 and patched.title == "crud 自检用书"

        print("\n  【5】update_book（全量更新）")
        updated = update_book(db, got, schemas.BookUpdate(
            title="crud 自检用书（第2版）", author="自检", isbn=isbn, price=69.9, stock=3))
        print(f"      更新后：{updated.title!r} 价格 {updated.price} 库存 {updated.stock}")

        print("\n  【6】delete_book（并清理测试数据）")
        count_before = count_all(db)
        delete_book(db, updated)
        count_after = count_all(db)
        print(f"      删除前 {count_before} 条 → 删除后 {count_after} 条")
        assert count_after == count_before - 1
        assert get_book(db, book.id) is None

    print("\n  为什么要单独测这一层？")
    print("    · 不依赖 HTTP 与网络，跑得飞快；")
    print("    · 能覆盖路由层难测的分支（排序白名单、空结果、边界值）；")
    print("    · 换数据库/换 ORM 时，只测这一层就够了。")
    print("\ncrud.py 自检完成 ✓")

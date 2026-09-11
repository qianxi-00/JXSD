"""models.py —— ORM 模型（数据表结构）
================================================================
对应课案章节：后端开发基础 → Web框架 → FastAPI（数据库集成）

SQLAlchemy 2.0 新风格三要素：
    class Base(DeclarativeBase)                        声明式基类
    id: Mapped[int] = mapped_column(primary_key=True)  类型注解 + 列配置
    relationship(...)                                  关系导航（非数据库列）
"""

from __future__ import annotations

import datetime as dt
import pathlib
import sys

from sqlalchemy import DateTime, Integer, Numeric, String, func
from sqlalchemy.orm import Mapped, mapped_column

# 保证无论以哪种方式加载，都能 import 到 database / schemas 等同级模块
_BOOK_API_DIR = pathlib.Path(__file__).resolve().parent
if str(_BOOK_API_DIR) not in sys.path:
    sys.path.insert(0, str(_BOOK_API_DIR))

from database import Base  # noqa: E402


class Book(Base):
    """图书表。

    字段设计说明：
        title      书名，建索引（列表页经常按书名搜索/排序）
        author     作者
        isbn       国际标准书号，唯一约束（一本书只有一个 ISBN）
        price      定价，用 Numeric(10, 2) 而不是 Float
                   —— 金额必须用精确小数，浮点数会出现 0.1+0.2 != 0.3 的经典问题
        stock      库存，非负
        created_at 创建时间，由数据库端 default=func.now() 生成，不依赖应用服务器时钟
    """
    __tablename__ = "books"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True, comment="主键")
    title: Mapped[str] = mapped_column(String(120), index=True, nullable=False, comment="书名")
    author: Mapped[str] = mapped_column(String(60), nullable=False, comment="作者")
    isbn: Mapped[str] = mapped_column(String(20), unique=True, index=True, nullable=False, comment="ISBN，唯一")
    price: Mapped[float] = mapped_column(Numeric(10, 2), nullable=False, default=0, comment="定价")
    stock: Mapped[int] = mapped_column(Integer, nullable=False, default=0, comment="库存")
    created_at: Mapped[dt.datetime] = mapped_column(
        DateTime, server_default=func.now(), nullable=False, comment="创建时间（数据库端生成）"
    )

    def __repr__(self) -> str:      # 调试时打印对象更可读
        return f"<Book id={self.id} title={self.title!r} price={self.price}>"


if __name__ == "__main__":
    # 单独运行本模块时的自检：验证表结构定义是否符合预期
    from sqlalchemy import inspect

    import database

    print("=" * 72)
    print("models.py 自检（SQLAlchemy 2.0 模型定义）")
    print("=" * 72)

    table = Book.__table__
    print(f"  模型类      ：{Book.__name__}")
    print(f"  对应表名    ：{table.name}")
    print(f"  字段数量    ：{len(table.columns)}")
    print("\n  字段明细（列名 / 类型 / 约束）：")
    for column in table.columns:
        flags = []
        if column.primary_key:
            flags.append("主键")
        if column.unique:
            flags.append("唯一")
        if column.index:
            flags.append("索引")
        if not column.nullable:
            flags.append("非空")
        if column.server_default is not None:
            flags.append("数据库端默认值")
        print(f"    {column.name:<12}{str(column.type):<16}{'、'.join(flags) if flags else '-'}")

    print("\n  建模要点回顾：")
    print("    1. Mapped[int] + mapped_column(...) 是 2.0 的推荐写法（类型即列定义）")
    print("    2. 金额用 Numeric(10,2) 而不是 Float —— 浮点数无法精确表示小数")
    print("    3. created_at 用 server_default=func.now()，由数据库生成，避免多机时钟不一致")
    print("    4. Book.__repr__ 让调试输出更可读")

    # 验证建表后表结构与模型一致
    # 注意：这里直接调 Base.metadata.create_all，而不是 database.init_db()。
    # 原因：init_db() 内部会 `import models`，而在"直接运行 models.py"的场景下，
    # 本文件是 __main__ 模块，再 import 一次 models 会创建第二个模块对象，
    # 于是同一张表被定义两次 → 报 "Table 'books' is already defined"。
    database.Base.metadata.create_all(bind=database.engine)
    inspector = inspect(database.engine)
    real_columns = [c["name"] for c in inspector.get_columns("books")]
    model_columns = [c.name for c in table.columns]
    print(f"\n  建表后数据库中的列：{real_columns}")
    print(f"  与模型定义一致    ：{real_columns == model_columns}")
    assert real_columns == model_columns, "数据库表结构与模型不一致"

    # 演示实例化（不落库）
    demo = Book(title="流畅的 Python", author="Luciano Ramalho", isbn="978-7-115-45415-7",
                price=139.0, stock=10)
    print(f"\n  实例化一个 Book 对象（未提交）：{demo!r}")
    print("\nmodels.py 自检完成 ✓")

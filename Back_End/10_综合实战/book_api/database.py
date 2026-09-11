"""database.py —— 数据库连接与会话管理
================================================================
对应课案章节：后端开发基础 → Web框架 → FastAPI（数据库集成 / 依赖注入）

职责（只做这一件事）：
    · 创建 SQLAlchemy 引擎（默认 SQLite，可通过环境变量换成 MySQL）
    · 提供会话工厂 SessionLocal
    · 提供依赖注入函数 get_db（yield 版本，请求结束自动关闭）
    · 提供 init_db 建表函数
"""

from __future__ import annotations

import os
import pathlib
import sys

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# ------------------------------------------------------------
# 路径：本文件位于 Back_End/10_综合实战/book_api/database.py
#   parents[0] = book_api，parents[1] = 10_综合实战，
#   parents[2] = Back_End，parents[3] = Python_Base（仓库根）
# ------------------------------------------------------------
BOOK_API_DIR = pathlib.Path(__file__).resolve().parent
BACK_END = pathlib.Path(__file__).resolve().parents[2]
ROOT = pathlib.Path(__file__).resolve().parents[3]

# 把自己所在目录加进 sys.path：这样 `python main.py` 和"被其他脚本按路径加载"
# 两种方式都能用扁平 import（import models）找到同级模块。
for _p in (str(BOOK_API_DIR), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DATA_DIR = BACK_END / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 数据库 URL
# ============================================================
# 默认 SQLite：零依赖，数据存在 Back_End/data/book_api.db 里。
# 想换成 MySQL：设置环境变量 BOOK_API_DATABASE_URL，
# 例如 mysql+pymysql://user:pwd@127.0.0.1:3306/book_api?charset=utf8mb4
DEFAULT_SQLITE = f"sqlite:///{(DATA_DIR / 'book_api.db').as_posix()}"
DATABASE_URL = os.getenv("BOOK_API_DATABASE_URL", DEFAULT_SQLITE)

_is_sqlite = DATABASE_URL.startswith("sqlite")

engine = create_engine(
    DATABASE_URL,
    echo=False,                       # 需要看 SQL 时改成 True
    future=True,
    # SQLite 特有：允许跨线程使用连接。
    # 同步端点会被 Starlette 放进线程池执行，不加这个参数会报 sqlite3 线程错误。
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    pool_pre_ping=not _is_sqlite,     # 非 SQLite 时开启"用前ping"，自动剔除失效连接
)

# 会话工厂
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    """所有 ORM 模型的基类（SQLAlchemy 2.0 风格）。"""
    pass


def get_db():
    """FastAPI 依赖：提供数据库会话。

    用 yield 而不是 return 的原因（课案重点）：
        · return：会话永远不会被关闭 → 连接池耗尽；
        · yield：请求处理完（或抛异常）后，finally 里的 close() 一定执行。
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """建表（幂等）。

    create_all 只会创建"不存在的表"，已存在的表不会被修改 ——
    这也是为什么生产环境需要 Alembic 之类的迁移工具来管理表结构变更。
    （本环境未安装 alembic，所以这里只做演示性的 create_all。）
    """
    import models  # noqa: F401  必须导入模型模块，表才会注册到 Base.metadata

    Base.metadata.create_all(bind=engine)


def describe() -> dict:
    """返回数据库基本信息，供 /health 展示（不暴露密码）。"""
    return {
        "url": engine.url.render_as_string(hide_password=True),
        "dialect": engine.dialect.name,
        "pool": type(engine.pool).__name__,
    }


if __name__ == "__main__":
    # 单独运行本模块时的自检：验证连接、建表、列出表结构
    from sqlalchemy import inspect

    print("=" * 72)
    print("database.py 自检（数据库连接与会话管理）")
    print("=" * 72)
    print(f"  配置来源    ：{'环境变量 BOOK_API_DATABASE_URL' if 'BOOK_API_DATABASE_URL' in os.environ else '默认 SQLite'}")
    for key, value in describe().items():
        print(f"  {key:<11}：{value}")

    print("\n  执行 init_db() 建表 ...")
    init_db()
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"  当前数据库中已有的表：{tables if tables else '(空)'}")
    for table in tables:
        columns = [f"{c['name']}:{c['type']}" for c in inspector.get_columns(table)]
        print(f"    · {table}：{', '.join(columns)}")

    print("\n  验证会话工厂与 get_db 依赖 ...")
    with SessionLocal() as session:
        count = session.execute(__import__("sqlalchemy").text(f"SELECT COUNT(*) FROM {tables[0]}")).scalar_one() if tables else 0
    print(f"  SessionLocal() 可用，{tables[0] if tables else '表'} 当前 {count} 行")
    print("\n  get_db() 是生成器函数（含 yield），必须由 FastAPI 的 Depends 驱动：")
    print("      def get_db():")
    print("          db = SessionLocal()")
    print("          try:")
    print("              yield db        # 请求处理期间使用")
    print("          finally:")
    print("              db.close()      # 请求结束后自动关闭，绝不泄漏连接")
    print("\ndatabase.py 自检完成 ✓")

"""FastAPI 数据库集成：SQLAlchemy 2.0 新风格 + 依赖注入管理会话
================================================================
对应课案章节：后端开发基础 → Web框架 → FastAPI（数据库集成 / 依赖注入管理 Session）

本节知识点：
    1. ORM 是什么：把"表 ↔ 类、行 ↔ 对象、列 ↔ 属性"映射起来，用 Python 写 SQL
    2. SQLAlchemy 2.0 新风格三件套：
           class Base(DeclarativeBase): ...      声明式基类
           id: Mapped[int] = mapped_column(primary_key=True)   类型注解即列定义
           select(User).where(...)               2.0 的查询写法（替代 1.x 的 query）
    3. sessionmaker + expire_on_commit=False 的工程意义
    4. **依赖注入管理 Session**：get_db() 用 yield，请求结束自动关闭连接
       —— 这就是课案反复强调的"为什么必须用 yield 而不是 return"
    5. 同步 ORM 与异步 ORM 的区别（异步要点：create_async_engine + AsyncSession + await）
    6. 本机没有 MySQL 时的优雅降级：尝试连接失败只打印中文提示，绝不抛异常
    7. 为什么默认用 SQLite：零依赖、零配置，一个文件就是一个数据库

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\06_数据库集成.py' --check
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\06_数据库集成.py'
    数据库文件：Back_End/data/demo.db
"""

from __future__ import annotations

import pathlib
import sys
import time

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import ForeignKey, String, create_engine, func, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

HERE = pathlib.Path(__file__).resolve().parent
BACK_END = HERE.parent
ROOT = HERE.parents[1]
DATA_DIR = BACK_END / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PORT = 8106

# ============================================================
# 一、数据库连接
# ============================================================
# 默认使用 SQLite：一个文件就是数据库，无需启动任何服务。
#   · check_same_thread=False：允许跨线程使用同一连接。
#     同步端点会被 Starlette 放进线程池执行，不加这个参数会报
#     "SQLite objects created in a thread can only be used in that same thread"。
DB_FILE = DATA_DIR / "demo.db"
SQLITE_URL = f"sqlite:///{DB_FILE.as_posix()}"

engine = create_engine(
    SQLITE_URL,
    echo=False,                     # echo=True 会打印所有 SQL，调试时打开很有用
    future=True,                    # 使用 2.0 风格 API（显式写出来更清楚）
    connect_args={"check_same_thread": False},
)

# sessionmaker 是"会话工厂"：
#   autocommit=False  必须手动 commit，符合"显式优于隐式"
#   autoflush=False   查询前不自动 flush，避免意外写入
#   expire_on_commit=False  提交后对象属性仍可访问（否则提交后再读属性会触发新查询，
#                           而 Session 可能已关闭 → DetachedInstanceError）
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, expire_on_commit=False)


# ============================================================
# 二、模型定义（SQLAlchemy 2.0 新风格）
# ============================================================
class Base(DeclarativeBase):
    """所有 ORM 模型的基类（2.0 用 DeclarativeBase 取代了 declarative_base()）。"""
    pass


class User(Base):
    """用户表。"""
    __tablename__ = "users"

    # Mapped[int] 提供类型信息，mapped_column 提供列的具体配置
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True, comment="主键")
    name: Mapped[str] = mapped_column(String(50), index=True, comment="用户名，建索引加速查询")
    email: Mapped[str] = mapped_column(String(120), unique=True, comment="邮箱，唯一约束")

    # relationship 不是数据库列，而是 ORM 层的"关系导航"
    #   back_populates 两边互相指认；cascade="all, delete-orphan" 实现级联删除
    articles: Mapped[list["Article"]] = relationship(
        back_populates="author", cascade="all, delete-orphan", lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} name={self.name!r}>"


class Article(Base):
    """文章表：演示外键关联。"""
    __tablename__ = "articles"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(100), index=True)
    content: Mapped[str] = mapped_column(String(500), default="")
    # ForeignKey("users.id")：数据库层的外键约束（保证不会出现孤儿数据）
    # index=True：外键列几乎总是要建索引，否则按用户查文章会全表扫描
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True,
                                         comment="外键：users.id")

    author: Mapped["User"] = relationship(back_populates="articles")


class UserIn(BaseModel):
    """创建用户的请求体模型。"""
    name: str = Field(..., min_length=2, max_length=50, description="用户名")
    email: str = Field(..., pattern=r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$", description="邮箱")


class UserOut(BaseModel):
    """用户响应模型（SQLAlchemy 对象 → Pydantic，靠 from_attributes）。"""
    id: int
    name: str
    email: str

    # 允许从任意对象（而不是只有 dict）读属性 —— ORM 对象返回时必需
    model_config = {"from_attributes": True}


# ============================================================
# 三、建表：延迟执行，避免 import 时产生副作用
# ============================================================
_tables_ready = False


def ensure_tables() -> None:
    """创建所有表（幂等）。

    为什么不放在模块顶层？
        模块顶层执行 = "import 就有副作用"，测试与校验脚本 import 时会莫名其妙建库。
        这里做成延迟执行：第一次请求时建表，之后靠 _tables_ready 标志直接返回。
    """
    global _tables_ready
    if _tables_ready:
        return
    Base.metadata.create_all(bind=engine)
    _tables_ready = True


# ============================================================
# 四、依赖注入管理 Session（本节核心）
# ============================================================
def get_db():
    """提供数据库会话的依赖。

    为什么必须用 yield？
        · 直接 return session：会话永远不会被关闭 → 连接池耗尽 → 服务假死。
        · get_db() 是生成器函数，直接写 `db = get_db()` 只会拿到生成器对象，
          yield 之前的代码根本不会执行（课案专门讲了这一点）。
        · 只有 Depends 会这样处理生成器：
              1) 调用它，执行到 yield，把 session 注入到路由参数；
              2) 请求处理结束后，回到 yield 之后继续执行，自动关闭会话。

    这就是"解耦 + 复用 + 自动清理"：
        - 路由函数不需要知道 session 是怎么创建的；
        - 所有路由共用同一份逻辑（超时、事务、日志都能在这里统一加）；
        - 请求结束自动 close，不会漏。
    """
    db = SessionLocal()
    try:
        ensure_tables()          # 第一次请求时建表
        yield db
    finally:
        db.close()               # 无论请求成功还是抛异常，都会执行


# 类型别名：写起来更简洁，也便于将来换成 AsyncSession
DbSession = Depends(get_db)


app = FastAPI(
    title="06 数据库集成",
    description="后端开发基础 · FastAPI 示例：SQLAlchemy 2.0 + SQLite + 依赖注入",
    version="1.0.0",
)


# ============================================================
# 五、CRUD 接口
# ============================================================
@app.get("/users", summary="用户列表（分页）", tags=["用户"], response_model=dict)
def list_users(
    db: Session = DbSession,
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
) -> dict:
    """查询列表。

    2.0 风格查询：
        stmt = select(User).where(...).order_by(...).offset(...).limit(...)
        db.execute(stmt).scalars().all()
    统计总数用 func.count()。
    """
    total = db.execute(select(func.count()).select_from(User)).scalar_one()
    rows = db.execute(
        select(User).order_by(User.id).offset(skip).limit(limit)
    ).scalars().all()
    return {
        "total": total,
        "items": [UserOut.model_validate(row).model_dump() for row in rows],
        "说明": "SQLite 文件：" + str(DB_FILE),
    }


@app.post("/users", summary="创建用户", tags=["用户"], status_code=status.HTTP_201_CREATED)
def create_user(payload: UserIn, db: Session = DbSession) -> dict:
    """写入数据的三步：add → commit → refresh。

        db.add(obj)       把对象放入会话（此时还没发 SQL）
        db.commit()       提交事务，真正写入数据库；主键会被回填到 obj.id
        db.refresh(obj)   从数据库重新读取，拿到默认值/触发器产生的字段
    """
    exists = db.execute(select(User).where(User.email == payload.email)).scalar_one_or_none()
    if exists is not None:
        raise HTTPException(status_code=409, detail=f"邮箱 {payload.email} 已被注册")

    user = User(name=payload.name, email=payload.email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"id": user.id, "name": user.name, "email": user.email, "message": "创建成功"}


@app.get("/users/{user_id}", summary="查询单个用户（含关联文章）", tags=["用户"])
def get_user(user_id: int, db: Session = DbSession) -> dict:
    """按主键查询：db.get(User, id) 是最简单的方式（会走身份映射缓存）。"""
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"用户 {user_id} 不存在")
    return {
        "id": user.id,
        "name": user.name,
        "email": user.email,
        "articles": [{"id": a.id, "title": a.title} for a in user.articles],
        "说明": "articles 通过 relationship 自动加载（lazy='selectin'）",
    }


@app.delete("/users/{user_id}", summary="删除用户（级联删除文章）", tags=["用户"])
def delete_user(user_id: int, db: Session = DbSession) -> dict:
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"用户 {user_id} 不存在")
    db.delete(user)          # cascade="all, delete-orphan" 会让关联文章一起删除
    db.commit()
    return {"deleted": user_id, "message": "已删除（关联文章一并删除）"}


@app.post("/users/{user_id}/articles", summary="给用户新增文章", tags=["文章"], status_code=201)
def create_article(
    user_id: int,
    db: Session = DbSession,
    title: str = Query(..., min_length=1, max_length=100, description="文章标题"),
) -> dict:
    """外键写入示例。

    注意：不要把客户端传的 user_id 直接塞进对象就 commit ——
    必须先确认这个用户存在，否则要么报外键错误（500），要么产生孤儿数据。
    """
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"用户 {user_id} 不存在")
    article = Article(title=title, content=f"{title} 的正文占位内容", user_id=user_id)
    db.add(article)
    db.commit()
    db.refresh(article)
    return {"id": article.id, "title": article.title, "user_id": article.user_id}


@app.get("/stats", summary="聚合查询（GROUP BY）", tags=["统计"])
def stats(db: Session = DbSession) -> dict:
    """用 SQL 聚合函数做统计：func.count / func.max / group_by。"""
    stmt = (
        select(User.id, User.name, func.count(Article.id).label("article_count"))
        .outerjoin(Article, Article.user_id == User.id)
        .group_by(User.id)
        .order_by(func.count(Article.id).desc())
    )
    rows = db.execute(stmt).all()
    return {
        "统计": [{"user_id": r[0], "name": r[1], "article_count": r[2]} for r in rows],
        "说明": "outerjoin + group_by 统计每个用户的文章数",
    }


# ============================================================
# 六、MySQL 探活（本机没有 MySQL 时的优雅降级）
# ============================================================
def probe_mysql() -> dict:
    """尝试用根目录 config.py 里的 settings.mysql_url 连接 MySQL。

    **绝对不抛异常**：连接失败只返回一个说明性字典，并打印中文提示。
    这样即使本机没装/没启动 MySQL，本节示例也能零报错跑完。
    """
    result: dict = {"target": "MySQL", "available": False, "detail": ""}
    try:
        from config import settings            # 根目录统一配置
    except Exception as exc:                   # pragma: no cover
        result["detail"] = f"未能导入根配置：{type(exc).__name__}"
        return result

    url = settings.mysql_url
    # 打印/写报告时把「用户名 + 口令 + 库名」整体打码：
    # 只保留 scheme、主机与端口，避免数据库信息随日志或校验报告外泄。
    safe_url = url
    if "@" in safe_url:                      # 抹掉 用户名:口令@
        scheme, _, tail = safe_url.partition("://")
        safe_url = f"{scheme}://***:***@{tail.split('@', 1)[1]}"
    base, sep, query = safe_url.partition("?")   # 抹掉库名，保留 ?charset=... 参数
    if "/" in base:
        safe_url = base.rsplit("/", 1)[0] + "/***" + (sep + query if sep else "")
    result["url"] = safe_url

    try:
        # connect_timeout=2：本机没服务时立刻失败，不会长时间卡住
        mysql_engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 2})
        with mysql_engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        result["available"] = True
        result["detail"] = "MySQL 连接成功，可正常使用"
        mysql_engine.dispose()
    except Exception as exc:
        # 注意：MySQL 未启动时这里是正常的，不是程序 bug
        result["detail"] = (
            f"MySQL 未连接（{type(exc).__name__}）。"
            "本机没有启动 MySQL 服务属于正常情况，示例已自动降级为 SQLite。"
            "需要真实连接时：启动 MySQL → 建库 → 在根目录 .env 里配置 MYSQL_* → 重启本脚本。"
        )
    return result


# ============================================================
# 自检
# ============================================================
def run_self_check() -> None:
    from fastapi.testclient import TestClient

    print("=" * 72)
    print("FastAPI 数据库集成 · 自检")
    print("=" * 72)
    print(f"SQLite 文件：{DB_FILE}")
    print(f"数据库 URL：{SQLITE_URL}")
    print()

    # 用时间戳构造本次运行专属的邮箱，保证脚本可以反复运行而不撞唯一约束
    stamp = int(time.time() * 1000) % 100_000_000
    email = f"user{stamp}@example.com"

    client = TestClient(app)

    def call(desc: str, method: str, path: str, expect: int, **kwargs) -> dict:
        resp = client.request(method, path, **kwargs)
        body = resp.text.replace("\n", " ")
        if len(body) > 190:
            body = body[:190] + "…"
        mark = "✓" if resp.status_code == expect else "✗"
        print(f"\n{mark} [{desc}]")
        print(f"    {method} {path}")
        print(f"    状态码 {resp.status_code}（期望 {expect}）")
        print(f"    响应 {body}")
        assert resp.status_code == expect, f"{desc} 期望 {expect}，实际 {resp.status_code}"
        return resp.json()

    # ---------- 1. 初始列表（可能是上一次运行留下的数据） ----------
    data = call("用户列表", "GET", "/users", 200)
    before_total = data["total"]
    print(f"    （本次运行开始时库里已有 {before_total} 条；数据持久化在 demo.db 中）")

    # ---------- 2. 创建 ----------
    data = call("创建用户", "POST", "/users", 201,
                json={"name": "张三", "email": email})
    uid = data["id"]
    assert isinstance(uid, int) and uid > 0, "commit + refresh 之后应当拿到自增主键"

    # ---------- 3. 唯一约束冲突 ----------
    call("邮箱重复（唯一约束）", "POST", "/users", 409,
         json={"name": "李四", "email": email})

    # ---------- 4. 校验失败（Pydantic 层） ----------
    call("邮箱格式不合法", "POST", "/users", 422, json={"name": "王五", "email": "bad-email"})

    # ---------- 5. 关联写入 ----------
    call("给用户添加文章", "POST", f"/users/{uid}/articles?title=SQLAlchemy 2.0 入门", 201)
    call("再添加一篇", "POST", f"/users/{uid}/articles?title=依赖注入实践", 201)

    # ---------- 6. 关联查询 ----------
    data = call("查询用户（含关联文章）", "GET", f"/users/{uid}", 200)
    assert len(data["articles"]) == 2, "relationship 应当加载出 2 篇文章"
    print(f"    ✓ relationship 加载到 {len(data['articles'])} 篇文章："
          f"{[a['title'] for a in data['articles']]}")

    # ---------- 7. 分页 ----------
    data = call("分页查询 skip=0&limit=1", "GET", "/users?skip=0&limit=1", 200)
    assert len(data["items"]) <= 1
    call("分页参数非法 limit=0", "GET", "/users?limit=0", 422)

    # ---------- 8. 聚合统计 ----------
    data = call("聚合统计（GROUP BY）", "GET", "/stats", 200)
    print(f"    统计结果：{data['统计'][:3]}")

    # ---------- 9. 404 ----------
    call("查询不存在的用户", "GET", "/users/999999", 404)

    # ---------- 10. 删除 + 级联 ----------
    call("删除用户（级联删除文章）", "DELETE", f"/users/{uid}", 200)
    call("删除后再次查询", "GET", f"/users/{uid}", 404)

    # 验证文章确实被级联删除（直接查库）
    with SessionLocal() as session:
        left = session.execute(select(func.count()).select_from(Article).where(Article.user_id == uid)).scalar_one()
    print(f"\n[级联删除验证]")
    print(f"    数据库中 user_id={uid} 的文章剩余：{left} 篇（期望 0）")
    assert left == 0, "cascade='all, delete-orphan' 应当把关联文章一起删除"

    # ---------- 11. 会话生命周期验证 ----------
    print("\n[依赖注入会话生命周期验证]")
    print("    get_db() 用 yield：请求结束时 finally 里的 db.close() 一定执行")
    print("    → 连接不会泄漏，连接池不会被耗尽")
    print(f"    SQLAlchemy 引擎信息：{engine.url.render_as_string(hide_password=True)}")
    print(f"    连接池类：{type(engine.pool).__name__}"
          f"（池大小 {getattr(engine.pool, 'size', lambda: '?')()}）")

    # ---------- 12. MySQL 探活 ----------
    print("\n[MySQL 探活（本机未启动时不影响本示例运行）]")
    mysql = probe_mysql()
    print(f"    目标：{mysql['target']}")
    print(f"    连接串：{mysql.get('url', '(未获取到)')}")
    print(f"    可用：{mysql['available']}")
    print(f"    说明：{mysql['detail']}")

    print("\n" + "=" * 72)
    print("自检全部通过 ✓")
    print("=" * 72)
    print("知识点回顾：")
    print("1. 2.0 风格：DeclarativeBase + Mapped + mapped_column + select()")
    print("2. 写入三步：add → commit → refresh；查询用 db.execute(stmt).scalars()")
    print("3. 依赖注入 + yield 是管理 Session 的标准姿势：解耦、复用、自动清理")
    print("4. SQLite 要加 check_same_thread=False，因为同步端点跑在线程池里")
    print("5. 外部服务（MySQL）不可用时要降级而不是崩溃，日志里给出明确中文提示")
    print("6. 异步 ORM 把 Session 换成 AsyncSession、把语句用 await 执行即可，写法一致")
    print()


def main() -> None:
    print()
    print("#" * 72)
    print("# FastAPI 数据库集成（后端开发基础 · Web框架章节）")
    print("#" * 72)
    print(f"解释器：{sys.executable}")
    print(f"数据库文件：{DB_FILE}")
    print(f"计划监听端口：{PORT}")
    print()

    if "--check" in sys.argv:
        print("提示：--check 模式，执行自检，不启动服务器、不阻塞。")
        run_self_check()
    else:
        ensure_tables()
        print(f"启动 uvicorn：http://127.0.0.1:{PORT}/docs")
        print("按 Ctrl+C 停止。")
        uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")


if __name__ == "__main__":
    main()


"""main.py —— 图书管理 API 的应用入口
================================================================
对应课案章节：后端开发基础 → Web框架（综合实战：把各节知识点组装起来）

本项目用到的全部知识点：
    · FastAPI 应用与 APIRouter 分层               （Web框架概览 / 路由）
    · Pydantic 请求与响应模型、字段校验            （数据模型与验证）
    · 依赖注入 Depends(get_db) 管理数据库会话      （依赖注入）
    · SQLAlchemy 2.0 同步 ORM + SQLite            （数据库集成）
    · 中间件（CORS / 耗时统计 / 请求 ID）          （中间件）
    · 统一异常响应格式与状态码语义                 （错误处理）
    · TestClient 自测与 pytest 测试               （测试框架）

运行方式：
    # 启动服务（浏览器打开 http://127.0.0.1:8113/docs）
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\10_综合实战\\book_api\\main.py'

    # 快速自检（不启服务、不阻塞）
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\10_综合实战\\book_api\\main.py' --check

    # 跑单元/接口测试
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m pytest 'F:\\ProGram\\Python_Base\\Back_End\\10_综合实战\\book_api' -q

数据库文件：Back_End/data/book_api.db（可用环境变量 BOOK_API_DATABASE_URL 覆盖）
"""

from __future__ import annotations

import pathlib
import sys
import time
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

# ------------------------------------------------------------
# 路径引导：把自己所在目录加入 sys.path，之后就可以用扁平 import
# ------------------------------------------------------------
BOOK_API_DIR = pathlib.Path(__file__).resolve().parent
BACK_END = BOOK_API_DIR.parents[1]
ROOT = BOOK_API_DIR.parents[2]
for _p in (str(BOOK_API_DIR), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import crud  # noqa: E402
import database  # noqa: E402
from routers import books  # noqa: E402

VERSION = "1.0.0"
PORT = 8113


# ============================================================
# 生命周期（lifespan）
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭钩子。

    lifespan 取代了旧版的 @app.on_event("startup")：
        yield 之前 = 启动时执行（建表、预热连接池、加载模型……）
        yield 之后 = 关闭时执行（释放连接、清理临时文件……）
    用 TestClient(app) 作为上下文管理器时同样会触发这些钩子。
    """
    database.init_db()                      # 建表（幂等）
    print(f"[book_api] 数据库就绪：{database.describe()['url']}")
    yield
    database.engine.dispose()               # 关闭时释放连接池
    print("[book_api] 连接池已释放，应用退出")


app = FastAPI(
    title="图书管理 API（Back_End 综合实战）",
    description=(
        "后端开发基础课案的综合实战项目：SQLite + SQLAlchemy 2.0 + Pydantic + "
        "APIRouter 分层 + 依赖注入 + 统一异常处理。"
    ),
    version=VERSION,
    lifespan=lifespan,
)

# ============================================================
# 中间件
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-Id"],
)

REQUEST_LOG: list[str] = []


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """给每个响应加上耗时与追踪 ID，并记一条访问日志。"""
    request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
    start = time.perf_counter()
    response = await call_next(request)
    cost_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time"] = f"{cost_ms:.3f}ms"
    response.headers["X-Request-Id"] = request_id
    REQUEST_LOG.append(f"{request.method} {request.url.path} -> {response.status_code} ({cost_ms:.2f}ms)")
    return response


# ============================================================
# 统一异常处理
# ============================================================
def error_body(status_code: int, message: str, detail=None, request_id: str | None = None) -> dict:
    """统一错误响应结构（与 06_FastAPI/10_错误处理.py 保持一致）。"""
    return {
        "code": status_code,
        "message": message,
        "detail": detail,
        "request_id": request_id or uuid.uuid4().hex[:12],
    }


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """统一处理 HTTPException 与框架抛出的 404/405。"""
    message = exc.detail if isinstance(exc.detail, str) else "请求处理失败"
    return JSONResponse(
        status_code=exc.status_code,
        content=error_body(exc.status_code, message, {"path": request.url.path, "method": request.method}),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """把 422 校验错误整理成前端好用的结构。"""
    fields = [
        {"字段": ".".join(str(x) for x in item.get("loc", [])),
         "原因": item.get("msg"),
         "类型": item.get("type")}
        for item in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content=error_body(422, "请求参数校验未通过", {"字段错误": fields}),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底：未预期异常返回统一 500，绝不把堆栈给客户端。"""
    # 生产环境应替换为 logger.exception(...)
    print(f"[book_api][服务端日志] 未处理异常 {type(exc).__name__}: {exc} "
          f"（{request.method} {request.url.path}）")
    return JSONResponse(status_code=500, content=error_body(500, "服务器内部错误，请稍后重试"))


# ============================================================
# 路由装配
# ============================================================
app.include_router(books.router)


@app.get("/", tags=["基础"], summary="服务信息")
def index() -> dict:
    return {
        "code": 0,
        "message": "ok",
        "data": {
            "name": "图书管理 API",
            "version": VERSION,
            "docs": "/docs",
            "redoc": "/redoc",
            "接口清单": [
                "GET    /api/books            列表（分页/搜索/排序）",
                "POST   /api/books            新增",
                "GET    /api/books/{id}       详情",
                "PUT    /api/books/{id}       全量更新",
                "PATCH  /api/books/{id}       局部更新",
                "DELETE /api/books/{id}       删除",
                "GET    /health               健康检查",
            ],
        },
    }


@app.get("/health", tags=["运维"], summary="健康检查")
def health() -> dict:
    """健康检查：K8S 探针、负载均衡都会周期性调用它。

    这里顺带查一次数据库（SELECT COUNT），能真实反映"依赖是否可用"。
    """
    try:
        with database.SessionLocal() as db:
            total = crud.count_all(db)
        db_status = {"available": True, "book_count": total}
    except Exception as exc:
        db_status = {"available": False, "error": type(exc).__name__}
    return {
        "code": 0,
        "message": "ok",
        "data": {
            "status": "ok",
            "version": VERSION,
            "database": {**database.describe(), **db_status},
        },
    }


# ============================================================
# 自检（--check）
# ============================================================
def run_self_check() -> None:
    """用 TestClient 走一遍完整 CRUD 流程，验证项目可用。

    注意：TestClient(app) 作为上下文管理器使用时，会触发 lifespan（建表），
    所以这里的自检与真实启动的运行环境是一致的。
    """
    import uuid as _uuid

    from fastapi.testclient import TestClient

    print("=" * 72)
    print("book_api 综合实战 · 自检")
    print("=" * 72)
    print(f"数据库：{database.describe()['url']}")
    print()

    with TestClient(app) as client:
        def call(desc: str, method: str, path: str, expect: int, **kwargs) -> dict:
            resp = client.request(method, path, **kwargs)
            body = resp.text.replace("\n", " ")
            if len(body) > 200:
                body = body[:200] + "…"
            mark = "✓" if resp.status_code == expect else "✗"
            print(f"\n{mark} [{desc}]")
            print(f"    {method} {path}")
            print(f"    状态码 {resp.status_code}（期望 {expect}）  "
                  f"耗时 {resp.headers.get('X-Process-Time')}")
            print(f"    响应 {body}")
            assert resp.status_code == expect, f"{desc} 期望 {expect}，实际 {resp.status_code}"
            return resp.json()

        # ---------- 健康检查 ----------
        call("健康检查", "GET", "/health", 200)
        call("服务信息", "GET", "/", 200)

        # ---------- 新增 ----------
        isbn = f"978-7-115-{_uuid.uuid4().int % 100000:05d}-0"
        data = call("新增图书", "POST", "/api/books", 201, json={
            "title": "流畅的 Python",
            "author": "Luciano Ramalho",
            "isbn": isbn,
            "price": 139.0,
            "stock": 10,
        })
        book = data["data"]
        book_id = book["id"]
        assert data["code"] == 0 and book["stock"] == 10

        call("新增图书（ISBN 重复 → 409）", "POST", "/api/books", 409, json={
            "title": "重复的书", "author": "某人", "isbn": isbn, "price": 1, "stock": 1,
        })
        call("新增图书（价格非法 → 422）", "POST", "/api/books", 422, json={
            "title": "价格非法", "author": "某人", "isbn": "978-7-000-00000-0", "price": -5,
        })

        # ---------- 查询 ----------
        call("图书详情", "GET", f"/api/books/{book_id}", 200)
        call("图书不存在 → 404", "GET", "/api/books/99999999", 404)
        data = call("列表（分页）", "GET", "/api/books?page=1&size=3", 200)
        print(f"    共 {data['data']['total']} 条，本页 {len(data['data']['items'])} 条")
        call("列表（关键字搜索）", "GET", "/api/books?keyword=流畅", 200)
        call("列表（排序）", "GET", "/api/books?order_by=price&desc=true", 200)
        call("列表（非法排序字段 → 422）", "GET", "/api/books?order_by=bad", 422)

        # ---------- 更新 ----------
        call("PUT 全量更新", "PUT", f"/api/books/{book_id}", 200, json={
            "title": "流畅的 Python（第 2 版）",
            "author": "Luciano Ramalho",
            "isbn": isbn,
            "price": 149.0,
            "stock": 8,
        })
        data = call("PATCH 局部更新（只改库存）", "PATCH", f"/api/books/{book_id}", 200,
                    json={"stock": 3})
        assert data["data"]["title"] == "流畅的 Python（第 2 版）", "PATCH 不应改动未提交字段"
        assert data["data"]["stock"] == 3
        call("PATCH 空请求体 → 400", "PATCH", f"/api/books/{book_id}", 400, json={})

        # ---------- 删除 ----------
        call("删除图书", "DELETE", f"/api/books/{book_id}", 200)
        call("删除后查询 → 404", "GET", f"/api/books/{book_id}", 404)

        # ---------- 文档与中间件 ----------
        call("OpenAPI 文档", "GET", "/openapi.json", 200)
        resp = client.get("/health", headers={"X-Request-Id": "trace-book-001"})
        assert resp.headers.get("X-Request-Id") == "trace-book-001"
        print(f"\n[中间件] 透传 X-Request-Id 成功，访问日志共 {len(REQUEST_LOG)} 条")
        for line in REQUEST_LOG[-4:]:
            print(f"    {line}")

    print("\n" + "=" * 72)
    print("自检全部通过 ✓")
    print("=" * 72)
    print("项目结构回顾：")
    print("    main.py       应用入口：中间件 + 异常处理 + 路由装配 + 生命周期")
    print("    database.py   引擎 / 会话工厂 / get_db 依赖 / 建表")
    print("    models.py     ORM 模型（SQLAlchemy 2.0 风格）")
    print("    schemas.py    Pydantic 请求与响应模型（入参严格、出参精简）")
    print("    crud.py       数据访问层（纯数据库操作，不认识 HTTP）")
    print("    routers/books.py  路由层（只解析请求、调用 crud、包装响应）")
    print("分层的好处：换库只改 database+crud；测业务不用起 HTTP；多人协作冲突少。")
    print()


def main() -> None:
    print()
    print("#" * 72)
    print("# 图书管理 API（Back_End 综合实战）")
    print("#" * 72)
    print(f"解释器：{sys.executable}")
    print(f"数据库：{database.describe()['url']}")
    print(f"计划监听端口：{PORT}")

    if "--check" in sys.argv:
        print()
        print("提示：--check 模式，执行自检，不启动服务器、不阻塞。")
        run_self_check()
    else:
        print(f"启动 uvicorn：http://127.0.0.1:{PORT}/docs")
        print("按 Ctrl+C 停止。也可用命令行：uvicorn main:app --reload（在 book_api 目录下执行）")
        uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")


if __name__ == "__main__":
    main()

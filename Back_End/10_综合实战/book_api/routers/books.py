"""routers/books.py —— 图书 CRUD 路由
================================================================
对应课案章节：后端开发基础 → Web框架 → FastAPI（RESTful 接口 + 数据库 + 依赖注入）

本文件是"路由层"，职责边界非常明确：
    ① 解析并校验请求（交给 FastAPI + Pydantic）；
    ② 调用 crud 层完成业务；
    ③ 把结果包装成统一响应结构。
**不写 SQL、不写复杂业务逻辑。**

接口清单：
    GET    /api/books           列表（分页 + 关键字 + 排序）
    POST   /api/books           新增
    GET    /api/books/{id}      详情
    PUT    /api/books/{id}      全量更新
    PATCH  /api/books/{id}      局部更新
    DELETE /api/books/{id}      删除
"""

from __future__ import annotations

import pathlib
import sys
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

# 把 book_api 目录加入 sys.path，保证 import crud / database / schemas 都能成功
_BOOK_API_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(_BOOK_API_DIR) not in sys.path:
    sys.path.insert(0, str(_BOOK_API_DIR))

import crud  # noqa: E402
import schemas  # noqa: E402
from database import get_db  # noqa: E402

router = APIRouter(prefix="/api/books", tags=["图书"])

# 依赖类型别名：db 参数写起来更短，也更明确"这是数据库会话"
DbSession = Annotated[Session, Depends(get_db)]


def ok(data=None, message: str = "ok", status_code: int = status.HTTP_200_OK):
    """统一成功响应：{code, message, data}。

    统一格式的好处见 05_Flask/04_RESTful接口.py 里的说明：
    前端只需写一次响应拦截器。
    """
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content={"code": 0, "message": message,
                 "data": data.model_dump(mode="json") if hasattr(data, "model_dump") else data},
    )


def page_payload(total: int, page: int, size: int, items: list) -> dict:
    """把 (总数, 分页参数, 数据列表) 组装成统一的分页结构。"""
    return {
        "total": total,
        "page": page,
        "size": size,
        "items": [schemas.BookOut.model_validate(item).model_dump(mode="json") for item in items],
    }


# ============================================================
# 列表
# ============================================================
@router.get("", summary="图书列表（分页 / 搜索 / 排序）")
def list_books(
    db: DbSession,
    page: Annotated[int, Query(ge=1, description="页码，从 1 开始")] = 1,
    size: Annotated[int, Query(ge=1, le=100, description="每页条数，最多 100")] = 10,
    keyword: Annotated[str | None, Query(max_length=50, description="按书名或作者模糊搜索")] = None,
    order_by: Annotated[str, Query(description="排序字段：id/title/price/stock/created_at")] = "id",
    desc: Annotated[bool, Query(description="是否倒序")] = False,
) -> object:
    """查询列表。

    注意路由装饰器写的是 `@router.get("")` 而不是 `@router.get("/")`：
        配合 prefix="/api/books"，最终路径是 /api/books（不带尾斜杠）。
        若写成 "/"，路径会变成 /api/books/，访问 /api/books 会 307 重定向，
        对前端与监控都不太友好。两种写法都合法，团队内保持一致即可。
    """
    if order_by not in {"id", "title", "price", "stock", "created_at"}:
        raise HTTPException(status_code=422, detail=f"不支持的排序字段：{order_by}")
    total, items = crud.list_books(db, page=page, size=size, keyword=keyword,
                                   order_by=order_by, desc=desc)
    return ok(page_payload(total, page, size, items))


# ============================================================
# 详情
# ============================================================
@router.get("/{book_id}", summary="图书详情")
def get_book(book_id: int, db: DbSession) -> object:
    """按 ID 查询，不存在返回 404。"""
    book = crud.get_book(db, book_id)
    if book is None:
        raise HTTPException(status_code=404, detail=f"图书 {book_id} 不存在")
    return ok(schemas.BookOut.model_validate(book))


# ============================================================
# 新增
# ============================================================
@router.post("", summary="新增图书", status_code=status.HTTP_201_CREATED)
def create_book(payload: schemas.BookCreate, db: DbSession) -> object:
    """新增图书。

    唯一性校验属于"业务规则"，必须在应用层做：
        · 数据库的唯一索引是最后一道防线（并发下仍可能撞车）；
        · 应用层先查一次，能给用户返回友好的 409 提示而不是 500。
    """
    if crud.get_book_by_isbn(db, payload.isbn) is not None:
        raise HTTPException(status_code=409, detail=f"ISBN {payload.isbn} 已存在")
    book = crud.create_book(db, payload)
    return ok(schemas.BookOut.model_validate(book), message="创建成功",
              status_code=status.HTTP_201_CREATED)


# ============================================================
# 全量更新
# ============================================================
@router.put("/{book_id}", summary="全量更新图书（PUT）")
def update_book(book_id: int, payload: schemas.BookUpdate, db: DbSession) -> object:
    """PUT：客户端必须提交完整对象。"""
    book = crud.get_book(db, book_id)
    if book is None:
        raise HTTPException(status_code=404, detail=f"图书 {book_id} 不存在")
    # ISBN 改成别人已经用的值 → 冲突
    exists = crud.get_book_by_isbn(db, payload.isbn)
    if exists is not None and exists.id != book_id:
        raise HTTPException(status_code=409, detail=f"ISBN {payload.isbn} 已被其他图书使用")
    book = crud.update_book(db, book, payload)
    return ok(schemas.BookOut.model_validate(book), message="更新成功")


# ============================================================
# 局部更新
# ============================================================
@router.patch("/{book_id}", summary="局部更新图书（PATCH）")
def patch_book(book_id: int, payload: schemas.BookPatch, db: DbSession) -> object:
    """PATCH：只更新客户端提交的字段。"""
    book = crud.get_book(db, book_id)
    if book is None:
        raise HTTPException(status_code=404, detail=f"图书 {book_id} 不存在")

    data = payload.model_dump(exclude_unset=True, exclude_none=True)
    if not data:
        raise HTTPException(status_code=400, detail="请求体里没有任何可更新的字段")
    if "isbn" in data:
        exists = crud.get_book_by_isbn(db, data["isbn"])
        if exists is not None and exists.id != book_id:
            raise HTTPException(status_code=409, detail=f"ISBN {data['isbn']} 已被其他图书使用")

    book = crud.patch_book(db, book, payload)
    return ok(schemas.BookOut.model_validate(book), message="局部更新成功")


# ============================================================
# 删除
# ============================================================
@router.delete("/{book_id}", summary="删除图书")
def delete_book(book_id: int, db: DbSession) -> object:
    """删除。真实项目常用"软删除"（加 is_deleted 字段）以便追溯与恢复。"""
    book = crud.get_book(db, book_id)
    if book is None:
        raise HTTPException(status_code=404, detail=f"图书 {book_id} 不存在")
    info = schemas.BookOut.model_validate(book).model_dump(mode="json")
    crud.delete_book(db, book)
    return ok(info, message="删除成功")


if __name__ == "__main__":
    # 单独运行本模块时的自检：列出这个 APIRouter 注册了哪些路由
    # （路由表是"接口清单"的唯一真实来源，比手写文档可靠）
    print("=" * 72)
    print("routers/books.py 自检（路由层）")
    print("=" * 72)
    print(f"  APIRouter 前缀：{router.prefix}")
    print(f"  APIRouter 标签：{router.tags}")
    print(f"  路由数量      ：{len(router.routes)}")
    print("\n  路由清单（方法 / 完整路径 / 端点函数 / 说明）：")
    for route in router.routes:
        methods = ",".join(sorted(route.methods - {"HEAD"}))
        # 注意：APIRouter 上的 route.path 已经是"带前缀的完整路径"，
        # 不要再手工拼一次 router.prefix，否则会看到 /api/books/api/books/...
        full_path = route.path
        summary = getattr(route, "summary", "") or ""
        print(f"    {methods:<8}{full_path:<26}{route.name:<12}{summary}")
        assert full_path.startswith("/api/books"), "路由必须带 /api/books 前缀"

    print("\n  分层约定回顾：")
    print("    · 本文件只做：解析请求 → 调用 crud → 包装响应；")
    print("    · 数据库会话由 Depends(get_db) 注入，路由里不出现 SessionLocal()；")
    print("    · 业务错误统一 raise HTTPException，由 main.py 的处理器转成统一 JSON。")
    print("\nrouters/books.py 自检完成 ✓")

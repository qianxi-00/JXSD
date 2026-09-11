"""schemas.py —— Pydantic 请求 / 响应模型
================================================================
对应课案章节：后端开发基础 → Web框架 → FastAPI（数据模型与验证）

为什么 models（ORM）和 schemas（Pydantic）要分开？
    · ORM 模型描述"数据库表长什么样"，包含主键、外键、关系、懒加载等；
    · Pydantic 模型描述"HTTP 接口收什么、吐什么"。
    混用一个类会导致两个问题：
        1) 安全问题：客户端可能提交 id、created_at 等本不该由它决定的字段；
        2) 演进问题：数据库加一个内部字段，接口文档就跟着变了。
"""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# 请求模型（入参）
# ============================================================
class BookCreate(BaseModel):
    """创建图书的请求体。"""
    title: str = Field(..., min_length=1, max_length=120, description="书名")
    author: str = Field(..., min_length=1, max_length=60, description="作者")
    isbn: str = Field(..., pattern=r"^[0-9\-]{10,20}$", description="ISBN，10~20 位数字或横杠")
    price: float = Field(..., ge=0, le=100000, description="定价，0~100000")
    stock: int = Field(default=0, ge=0, le=100000, description="库存")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"title": "流畅的 Python", "author": "Luciano Ramalho",
                 "isbn": "978-7-115-45415-7", "price": 139.0, "stock": 10}
            ]
        }
    )


class BookUpdate(BaseModel):
    """PUT 全量更新：所有业务字段都必须提供。"""
    title: str = Field(..., min_length=1, max_length=120)
    author: str = Field(..., min_length=1, max_length=60)
    isbn: str = Field(..., pattern=r"^[0-9\-]{10,20}$")
    price: float = Field(..., ge=0, le=100000)
    stock: int = Field(..., ge=0, le=100000)


class BookPatch(BaseModel):
    """PATCH 局部更新：所有字段可选，但至少要提供一个（在接口里校验）。"""
    title: str | None = Field(default=None, min_length=1, max_length=120)
    author: str | None = Field(default=None, min_length=1, max_length=60)
    isbn: str | None = Field(default=None, pattern=r"^[0-9\-]{10,20}$")
    price: float | None = Field(default=None, ge=0, le=100000)
    stock: int | None = Field(default=None, ge=0, le=100000)


# ============================================================
# 响应模型（出参）
# ============================================================
class BookOut(BaseModel):
    """图书返回结构。

    from_attributes=True（旧名 orm_mode）：允许直接从 ORM 对象读属性构造模型，
    这样 `BookOut.model_validate(book)` 就能用，不必手写字段映射。
    """
    id: int
    title: str
    author: str
    isbn: str
    price: float
    stock: int
    created_at: dt.datetime

    model_config = ConfigDict(from_attributes=True)


class PageOut(BaseModel):
    """分页响应结构（本项目统一用 {total, page, size, items}）。"""
    total: int = Field(..., description="总条数")
    page: int = Field(..., description="当前页码，从 1 开始")
    size: int = Field(..., description="每页条数")
    items: list[BookOut] = Field(..., description="当前页数据")


class ApiResponse(BaseModel):
    """统一响应包装：{code, message, data}。

    code=0 表示成功；失败时 code 与 HTTP 状态码一致，便于前端统一处理。
    """
    code: int = 0
    message: str = "ok"
    data: object | None = None


if __name__ == "__main__":
    # 单独运行本模块时的自检：验证 Pydantic 模型的校验行为
    from pydantic import ValidationError

    print("=" * 72)
    print("schemas.py 自检（Pydantic 请求与响应模型）")
    print("=" * 72)

    print("  【1】合法请求体")
    valid = BookCreate(title="流畅的 Python", author="Luciano Ramalho",
                       isbn="978-7-115-45415-7", price=139.0, stock=10)
    print(f"      {valid.model_dump()}")
    print(f"      JSON Schema 必填字段：{BookCreate.model_json_schema().get('required')}")

    print("\n  【2】非法请求体（Pydantic 会指出具体字段与原因）")
    bad_cases = [
        ("书名长度为 0", {"title": "", "author": "某人", "isbn": "978-7-000-00000-0", "price": 1}),
        ("价格为负", {"title": "书", "author": "某人", "isbn": "978-7-000-00000-0", "price": -1}),
        ("ISBN 含字母", {"title": "书", "author": "某人", "isbn": "abcdefghij", "price": 1}),
        ("库存为负", {"title": "书", "author": "某人", "isbn": "978-7-000-00000-0", "price": 1, "stock": -3}),
    ]
    for desc, payload in bad_cases:
        try:
            BookCreate(**payload)
            print(f"      ✗ {desc}：居然通过了校验（不应该）")
        except ValidationError as exc:
            err = exc.errors()[0]
            field = ".".join(str(x) for x in err["loc"])
            print(f"      ✓ {desc}：字段 {field} -> {err['msg']}")

    print("\n  【3】BookPatch：所有字段可选（PATCH 局部更新的关键）")
    patch = BookPatch(stock=5)
    print(f"      只传 stock：{patch.model_dump(exclude_unset=True, exclude_none=True)}")
    print("      exclude_unset=True 只会带上客户端真正提交过的字段 ——")
    print("      这正是 PATCH 不会把未提交字段覆盖成 None 的原因。")

    print("\n  【4】BookOut：from_attributes=True 可直接从 ORM 对象构造")
    class FakeORM:
        id = 1
        title = "流畅的 Python"
        author = "Luciano Ramalho"
        isbn = "978-7-115-45415-7"
        price = 139.0
        stock = 10
        created_at = dt.datetime(2026, 1, 1, 12, 0, 0)

    out = BookOut.model_validate(FakeORM())
    print(f"      BookOut.model_validate(ORM 对象) -> {out.model_dump()}")

    print("\n  【5】PageOut：统一分页结构")
    page = PageOut(total=1, page=1, size=10, items=[out])
    print(f"      {page.model_dump()}")

    print("\nschemas.py 自检完成 ✓")

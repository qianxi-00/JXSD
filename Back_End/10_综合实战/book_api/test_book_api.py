"""test_book_api.py —— book_api 的 pytest + TestClient 接口测试
================================================================
对应课案章节：后端开发基础 → 测试框架（pytest + TestClient 测试 Web 接口）

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' -m pytest 'F:\\ProGram\\Python_Base\\Back_End\\10_综合实战\\book_api' -q

fixture 说明见 conftest.py（在 book_api 目录下）。
"""

from __future__ import annotations

import uuid

import pytest

# ============================================================
# 一、基础接口
# ============================================================
def test_index_returns_service_info(client):
    """首页返回服务信息。"""
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert body["code"] == 0
    assert "接口清单" in body["data"]


def test_health_reports_database(client):
    """健康检查要能反映数据库可用性。"""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["status"] == "ok"
    assert data["database"]["available"] is True


def test_middleware_adds_request_id(client):
    """中间件应当注入 X-Request-Id 与 X-Process-Time。"""
    resp = client.get("/health")
    assert resp.headers["X-Process-Time"].endswith("ms")
    assert len(resp.headers["X-Request-Id"]) == 12

    # 客户端自带 ID 时要透传（分布式追踪的基础）
    resp = client.get("/health", headers={"X-Request-Id": "trace-123456"})
    assert resp.headers["X-Request-Id"] == "trace-123456"


# ============================================================
# 二、创建
# ============================================================
def make_isbn() -> str:
    """生成一个唯一的 ISBN，保证测试可以重复运行。"""
    return f"978-7-000-{uuid.uuid4().int % 100000:05d}-0"


def test_create_book(client):
    """新增图书成功返回 201 与完整数据。"""
    resp = client.post("/api/books", json={
        "title": "测试驱动开发",
        "author": "Kent Beck",
        "isbn": make_isbn(),
        "price": 59.0,
        "stock": 5,
    })
    assert resp.status_code == 201
    data = resp.json()["data"]
    assert data["title"] == "测试驱动开发"
    assert data["id"] > 0
    assert data["created_at"], "created_at 应当由数据库生成后回填"


@pytest.mark.parametrize("field,value,expect_status", [
    ("title", "", 422),             # 书名不能为空
    ("price", -1, 422),             # 价格不能为负
    ("stock", -5, 422),             # 库存不能为负
    ("isbn", "abc", 422),           # ISBN 格式不合法
])
def test_create_book_validation(client, field, value, expect_status):
    """参数化测试：把多组非法数据一次性跑完。

    @pytest.mark.parametrize 的好处：用例失败时能精确定位是哪一组数据，
    不需要为每组数据写一个独立的测试函数。
    """
    payload = {"title": "验证用书", "author": "某人", "isbn": make_isbn(),
               "price": 10.0, "stock": 1}
    payload[field] = value
    resp = client.post("/api/books", json=payload)
    assert resp.status_code == expect_status
    body = resp.json()
    assert body["code"] == 422
    assert "字段错误" in body["detail"]


def test_create_book_missing_field(client):
    """缺少必填字段 → 422，并且指出是哪个字段。"""
    resp = client.post("/api/books", json={"title": "只有书名"})
    assert resp.status_code == 422
    fields = {item["字段"] for item in resp.json()["detail"]["字段错误"]}
    assert any("author" in f for f in fields)


def test_create_book_duplicate_isbn(client):
    """ISBN 唯一：第二次创建同名 ISBN 应当 409。"""
    isbn = make_isbn()
    payload = {"title": "唯一性测试", "author": "某人", "isbn": isbn, "price": 10, "stock": 1}
    assert client.post("/api/books", json=payload).status_code == 201
    resp = client.post("/api/books", json=payload)
    assert resp.status_code == 409
    assert "已存在" in resp.json()["message"]


# ============================================================
# 三、查询
# ============================================================
def test_get_book_not_found(client):
    """不存在的 ID 返回 404（而不是 500）。"""
    resp = client.get("/api/books/987654321")
    assert resp.status_code == 404
    assert resp.json()["code"] == 404


def test_list_books_pagination(client):
    """分页结构必须完整：total / page / size / items。"""
    resp = client.get("/api/books?page=1&size=2")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert set(data.keys()) == {"total", "page", "size", "items"}
    assert data["page"] == 1 and data["size"] == 2
    assert len(data["items"]) <= 2


def test_list_books_bad_order_by(client):
    """非法排序字段应当被拒绝，防止 SQL 注入。"""
    resp = client.get("/api/books?order_by=drop_table")
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_async_style_case(client):
    """异步风格的测试用例。

    pytest-asyncio 已安装，且在 Back_End/pytest.ini 里配置了 asyncio_mode=auto，
    所以 async def 测试可以直接运行。
    这里演示"异步测试里调用同步客户端"的写法（TestClient 本身是同步的），
    真实项目用 httpx.AsyncClient + ASGITransport 可以做纯异步测试。
    """
    import asyncio

    await asyncio.sleep(0)          # 模拟一次异步等待
    resp = client.get("/health")
    assert resp.status_code == 200


# ============================================================
# 四、更新与删除
# ============================================================
def test_put_and_patch(client):
    """PUT 全量更新、PATCH 局部更新的语义区别。"""
    isbn = make_isbn()
    created = client.post("/api/books", json={
        "title": "原始标题", "author": "作者", "isbn": isbn, "price": 20.0, "stock": 9,
    }).json()["data"]
    book_id = created["id"]

    # PUT：全量替换
    resp = client.put(f"/api/books/{book_id}", json={
        "title": "新标题", "author": "新作者", "isbn": isbn, "price": 30.0, "stock": 1,
    })
    assert resp.status_code == 200
    updated = resp.json()["data"]
    assert updated["title"] == "新标题" and updated["stock"] == 1

    # PATCH：只改库存，标题保持
    resp = client.patch(f"/api/books/{book_id}", json={"stock": 7})
    assert resp.status_code == 200
    patched = resp.json()["data"]
    assert patched["stock"] == 7
    assert patched["title"] == "新标题", "PATCH 不应改动未提交的字段"

    # PATCH 空对象 → 400
    assert client.patch(f"/api/books/{book_id}", json={}).status_code == 400


def test_delete_book(client):
    """删除后应当查不到。"""
    created = client.post("/api/books", json={
        "title": "待删除", "author": "某人", "isbn": make_isbn(), "price": 1.0, "stock": 1,
    }).json()["data"]
    book_id = created["id"]

    assert client.delete(f"/api/books/{book_id}").status_code == 200
    assert client.get(f"/api/books/{book_id}").status_code == 404
    # 重复删除 → 404
    assert client.delete(f"/api/books/{book_id}").status_code == 404


# ============================================================
# 五、文档与错误处理
# ============================================================
def test_openapi_document(client):
    """OpenAPI 文档必须能正常生成，且包含全部接口。"""
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert "/api/books" in paths
    assert "/api/books/{book_id}" in paths
    methods = set(paths["/api/books"].keys())
    assert {"get", "post"} <= methods


def test_unknown_path_returns_unified_error(client):
    """不存在的路径也要返回统一错误结构（而不是默认的 {"detail": ...}）。"""
    resp = client.get("/no-such-endpoint")
    assert resp.status_code == 404
    body = resp.json()
    assert set(body.keys()) == {"code", "message", "detail", "request_id"}
    assert body["code"] == 404


def test_swagger_docs_available(client):
    """Swagger UI 可访问。"""
    assert client.get("/docs").status_code == 200

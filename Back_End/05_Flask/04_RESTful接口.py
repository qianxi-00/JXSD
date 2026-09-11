"""Flask RESTful 接口：用内存数据做一套完整 CRUD
================================================================
对应课案章节：后端开发基础 → Web框架 →（Flask 部分）+ RESTful 设计思想

RESTful 的核心思想（先理解再写代码）：
    把一切都看成"资源（Resource）"，用 **URL 定位资源**、用 **HTTP 方法表达动作**：
        GET    /books        列出资源
        GET    /books/1      读取单个资源
        POST   /books        创建资源
        PUT    /books/1      全量更新（客户端提交完整对象）
        PATCH  /books/1      局部更新（只提交要改的字段）
        DELETE /books/1      删除资源
    好处：接口语义统一，前端/第三方看一眼 URL 和方法就知道在做什么；
          HTTP 状态码天然表达结果（201 创建、204 删除成功、404 不存在、422 校验失败）。

本节知识点：
    1. Blueprint（蓝图）：把一组路由打包成模块，是大项目分文件组织的基础
    2. 统一响应格式：{"code":0,"message":"ok","data":...}，前端处理更省心
    3. 状态码语义：200 / 201 / 204 / 400 / 404 / 422
    4. 请求体校验：手写校验函数（FastAPI 里由 Pydantic 自动完成，见 06 章）
    5. 分页与过滤：?page=&size=&keyword=
    6. 为什么内存存储不能用于生产：进程重启数据就没了 → 见 06_FastAPI/06_数据库集成.py

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\05_Flask\\04_RESTful接口.py'
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\05_Flask\\04_RESTful接口.py' --serve
"""

from __future__ import annotations

import itertools
import json
import logging
import pathlib
import sys

from flask import Blueprint, Flask, jsonify, request

# ------------------------------------------------------------
# 路径
# ------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
BACK_END = HERE.parent
DATA_DIR = BACK_END / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(HERE / "templates"), static_folder=str(HERE / "static"))
app.json.ensure_ascii = False
app.logger.setLevel(logging.CRITICAL + 10)

# ============================================================
# 一、统一响应封装
# ============================================================
def ok(data=None, message: str = "ok", status: int = 200):
    """成功响应：统一成 {code, message, data} 三段式。

    为什么统一格式？前端只需要写一次拦截器：
        if (resp.code !== 0) { 弹出 message }
    否则每个接口的返回结构都不一样，前端要写一堆 if。
    """
    return jsonify({"code": 0, "message": message, "data": data}), status


def fail(message: str, status: int = 400, code: int | None = None):
    """失败响应：HTTP 状态码 + 业务 code 双轨并行。

    - HTTP 状态码：给网关、监控、浏览器缓存等"基础设施"看；
    - 业务 code：给前端做精细提示（如 1001 表示书名重复）。
    """
    return jsonify({"code": code if code is not None else status, "message": message, "data": None}), status


# ============================================================
# 二、"数据库"：内存字典 + 自增 ID
# ============================================================
_id_counter = itertools.count(1)

BOOKS: dict[int, dict] = {}


def seed_data() -> None:
    """准备几条初始数据，方便演示。"""
    for title, author, price in [
        ("流畅的 Python", "Luciano Ramalho", 139.00),
        ("Python 编程：从入门到实践", "Eric Matthes", 89.50),
        ("设计数据密集型应用", "Martin Kleppmann", 128.00),
    ]:
        new_id = next(_id_counter)
        BOOKS[new_id] = {"id": new_id, "title": title, "author": author, "price": price}


def validate_book(payload: dict, partial: bool = False) -> tuple[dict, list[str]]:
    """手写请求体校验。

    返回 (清洗后的数据, 错误列表)。partial=True 表示 PATCH（只校验出现的字段）。

    为什么要"清洗"？绝不能把客户端传来的整个 JSON 直接存起来 —— 那样客户端可以塞
    extra 字段（如 "id": 999 覆盖主键、"is_admin": true 提权）。
    """
    errors: list[str] = []
    clean: dict = {}

    if "title" in payload or not partial:
        title = payload.get("title")
        if not isinstance(title, str) or not title.strip():
            errors.append("title 必填且必须是非空字符串")
        elif len(title) > 100:
            errors.append("title 长度不能超过 100")
        else:
            clean["title"] = title.strip()

    if "author" in payload or not partial:
        author = payload.get("author")
        if not isinstance(author, str) or not author.strip():
            errors.append("author 必填且必须是非空字符串")
        else:
            clean["author"] = author.strip()

    if "price" in payload or not partial:
        price = payload.get("price", 0)
        if not isinstance(price, (int, float)) or isinstance(price, bool):
            errors.append("price 必须是数字")
        elif price < 0:
            errors.append("price 不能为负数")
        else:
            clean["price"] = round(float(price), 2)

    return clean, errors


# ============================================================
# 三、Blueprint：把图书相关路由打包
# ============================================================
# 作用：main.py 里只写 app.register_blueprint(books_bp)，业务代码分文件维护。
# url_prefix="/api/books" 让蓝图内所有路由都带上统一前缀，改版本号只改这一处。
books_bp = Blueprint("books", __name__, url_prefix="/api/books")


@books_bp.get("")
def list_books():
    """GET /api/books —— 列表 + 关键字过滤 + 分页。

    查询参数一律是可选的，且有默认值：?page=1&size=10&keyword=python
    """
    keyword = (request.args.get("keyword") or "").strip().lower()
    page = max(1, request.args.get("page", 1, type=int))
    size = min(50, max(1, request.args.get("size", 10, type=int)))

    items = list(BOOKS.values())
    if keyword:
        items = [b for b in items if keyword in b["title"].lower() or keyword in b["author"].lower()]

    total = len(items)
    start = (page - 1) * size
    page_items = items[start:start + size]
    return ok({"total": total, "page": page, "size": size, "items": page_items})


@books_bp.get("/<int:book_id>")
def get_book(book_id: int):
    """GET /api/books/1 —— 读取单个资源，不存在返回 404。"""
    book = BOOKS.get(book_id)
    if book is None:
        return fail(f"图书 {book_id} 不存在", status=404)
    return ok(book)


@books_bp.post("")
def create_book():
    """POST /api/books —— 创建资源，成功返回 201 + Location 语义。"""
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return fail("请求体必须是 JSON 对象，且请求头应为 Content-Type: application/json", status=400)

    clean, errors = validate_book(payload)
    if errors:
        # 422 Unprocessable Entity：语法没错，但语义校验不通过
        return fail("；".join(errors), status=422)

    new_id = next(_id_counter)
    book = {"id": new_id, **clean}
    BOOKS[new_id] = book
    return ok(book, message="创建成功", status=201)


@books_bp.put("/<int:book_id>")
def update_book(book_id: int):
    """PUT —— 全量更新：客户端必须提交完整对象，缺字段视为校验失败。"""
    if book_id not in BOOKS:
        return fail(f"图书 {book_id} 不存在", status=404)
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return fail("请求体必须是 JSON 对象", status=400)

    clean, errors = validate_book(payload, partial=False)
    if errors:
        return fail("；".join(errors), status=422)

    BOOKS[book_id] = {"id": book_id, **clean}
    return ok(BOOKS[book_id], message="更新成功")


@books_bp.patch("/<int:book_id>")
def patch_book(book_id: int):
    """PATCH —— 局部更新：只改提交上来的字段。

    PUT 与 PATCH 的区别是面试高频题：
        PUT   语义是"用我给的完整表示替换这个资源"，没给的字段应被清空/视为缺失；
        PATCH 语义是"按我给的补丁修改这个资源"，没给的字段保持原样。
    """
    book = BOOKS.get(book_id)
    if book is None:
        return fail(f"图书 {book_id} 不存在", status=404)
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return fail("请求体必须是 JSON 对象", status=400)

    clean, errors = validate_book(payload, partial=True)
    if errors:
        return fail("；".join(errors), status=422)
    if not clean:
        return fail("没有可更新的字段", status=400)

    book.update(clean)
    return ok(book, message="局部更新成功")


@books_bp.delete("/<int:book_id>")
def delete_book(book_id: int):
    """DELETE —— 删除资源。

    204 No Content 是删除成功的标准响应（没有响应体）。
    这里为了教学仍返回 200 + 被删除的数据，方便确认删掉的是哪一条。
    """
    book = BOOKS.pop(book_id, None)
    if book is None:
        return fail(f"图书 {book_id} 不存在", status=404)
    return ok(book, message="删除成功")


@books_bp.post("/<int:book_id>/borrow")
def borrow_book(book_id: int):
    """子资源动作：POST /api/books/1/borrow

    REST 不总能表达所有业务动作（借书、下单、审批……），
    业界常见折中：用子路径 + 动词表达动作，仍然保持语义清晰。
    """
    book = BOOKS.get(book_id)
    if book is None:
        return fail(f"图书 {book_id} 不存在", status=404)
    if book.get("borrowed"):
        return fail(f"《{book['title']}》已被借出", status=409)     # 409 Conflict
    book["borrowed"] = True
    return ok(book, message="借阅成功")


# 注册蓝图：这一步之后蓝图里的路由才真正生效
app.register_blueprint(books_bp)


# ============================================================
# 四、全局错误处理：保证任何情况下返回的都是 JSON
# ============================================================
@app.errorhandler(404)
def handle_404(error):
    return fail(f"接口不存在：{request.path}", status=404)


@app.errorhandler(405)
def handle_405(error):
    return fail(f"{request.method} 方法不被 {request.path} 支持", status=405)


@app.errorhandler(500)
def handle_500(error):
    return fail("服务器内部错误", status=500)


@app.errorhandler(Exception)
def handle_any(error):
    """兜底：把任何未捕获异常统一转成 500 JSON，绝不让堆栈泄露给客户端。

    注意：Flask 在 debug=True 或 testing=True 时会把异常继续抛出，
    方便开发者看到完整堆栈；生产环境（debug=False）才会走这里。
    """
    from werkzeug.exceptions import HTTPException
    if isinstance(error, HTTPException):
        return fail(error.description, status=error.code)
    return fail(f"服务器内部错误：{type(error).__name__}", status=500)


# ============================================================
# 自检
# ============================================================
def run_self_check() -> None:
    print("=" * 72)
    print("Flask RESTful 接口 · 自检（完整 CRUD 流程）")
    print("=" * 72)

    seed_data()
    client = app.test_client()

    def show(desc: str, resp, expect: int) -> dict:
        body = resp.get_data(as_text=True)
        show_body = body if len(body) < 220 else body[:220] + "…"
        mark = "✓" if resp.status_code == expect else "✗"
        print(f"\n{mark} [{desc}]")
        print(f"    {resp.request.method} {resp.request.path}"
              + (f"?{resp.request.query_string.decode()}" if resp.request.query_string else ""))
        print(f"    状态码 {resp.status_code}（期望 {expect}）")
        print(f"    响应体 {show_body}")
        assert resp.status_code == expect, f"{desc} 期望 {expect}，实际 {resp.status_code}"
        return json.loads(body)

    # 1. 列表 + 分页
    data = show("列出图书（分页）", client.get("/api/books?page=1&size=2"), 200)
    assert data["data"]["total"] == 3 and len(data["data"]["items"]) == 2

    # 2. 关键字过滤
    data = show("关键字过滤 keyword=python", client.get("/api/books?keyword=python"), 200)
    assert data["data"]["total"] == 2, "应当匹配到 2 本含 python 的书"

    # 3. 读取单条
    show("读取图书 id=1", client.get("/api/books/1"), 200)

    # 4. 读取不存在
    show("读取不存在的图书 id=999", client.get("/api/books/999"), 404)

    # 5. 创建
    data = show("创建图书（合法）", client.post("/api/books", json={
        "title": "深入理解计算机系统", "author": "Randal Bryant", "price": 139.0,
    }), 201)
    new_id = data["data"]["id"]
    assert new_id == 4

    # 6. 创建失败：缺少字段
    show("创建图书（缺少 author）", client.post("/api/books", json={"title": "只有书名"}), 422)

    # 7. 创建失败：价格非法
    show("创建图书（price 为负数）", client.post("/api/books", json={
        "title": "非法价格", "author": "某人", "price": -1,
    }), 422)

    # 8. 创建失败：不是 JSON
    resp = client.post("/api/books", data="title=xxx", content_type="text/plain")
    print(f"\n✓ [创建图书（请求体不是 JSON）]")
    print(f"    POST /api/books  状态码 {resp.status_code}（期望 400）")
    print(f"    响应体 {resp.get_data(as_text=True)}")
    assert resp.status_code == 400

    # 9. PUT 全量更新
    show("PUT 全量更新 id=4", client.put(f"/api/books/{new_id}", json={
        "title": "深入理解计算机系统（第3版）", "author": "Randal Bryant", "price": 149.0,
    }), 200)

    # 10. PATCH 局部更新
    data = show("PATCH 局部更新 id=4 只改价格", client.patch(f"/api/books/{new_id}", json={"price": 129.0}), 200)
    assert data["data"]["title"] == "深入理解计算机系统（第3版）", "PATCH 不应改动未提交的字段"

    # 11. 借阅子资源动作（两次：第二次应当 409）
    show("借阅 id=4（第一次成功）", client.post(f"/api/books/{new_id}/borrow"), 200)
    show("借阅 id=4（重复借阅冲突）", client.post(f"/api/books/{new_id}/borrow"), 409)

    # 12. DELETE
    show("DELETE 删除 id=4", client.delete(f"/api/books/{new_id}"), 200)
    show("再次 GET 已删除的 id=4", client.get(f"/api/books/{new_id}"), 404)

    # 13. 方法不允许
    show("PATCH /api/books（不允许的方法）", client.patch("/api/books", json={}), 405)

    # 14. 接口不存在
    show("不存在的接口 /api/nothing", client.get("/api/nothing"), 404)

    print("\n" + "=" * 72)
    print("自检全部通过 ✓（共 18 个请求）")
    print("=" * 72)
    print("知识点回顾：")
    print("1. RESTful = URL 定位资源 + HTTP 方法表达动作 + 状态码表达结果")
    print("2. Blueprint 把路由按业务分文件，url_prefix 统一管理版本前缀")
    print("3. 请求体必须校验并「清洗」，防止客户端注入额外字段")
    print("4. PUT 全量替换、PATCH 局部修改，语义不要混用")
    print("5. 统一响应格式 + 全局错误处理，让前端只写一次拦截器")
    print("6. 内存存储仅用于演示；真实项目必须落库（见 06_FastAPI/06_数据库集成.py）")
    print()


def main() -> None:
    print()
    print("#" * 72)
    print("# Flask RESTful 接口（后端开发基础 · Web框架章节）")
    print("#" * 72)
    print(f"解释器：{sys.executable}")

    if "--serve" in sys.argv:
        seed_data()
        print("启动开发服务器：http://127.0.0.1:5004  （Ctrl+C 停止）")
        print("可访问：GET http://127.0.0.1:5004/api/books")
        app.run(host="127.0.0.1", port=5004, debug=True)
    else:
        print("提示：默认自检模式，不启动服务器。")
        print()
        run_self_check()


if __name__ == "__main__":
    main()

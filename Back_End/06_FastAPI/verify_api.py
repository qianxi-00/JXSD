"""FastAPI 全量接口校验（用 TestClient 逐个示例发请求）
================================================================
对应课案章节：后端开发基础 → Web框架 → FastAPI（全部小节）

**为什么需要这个文件？**
    06_FastAPI 下的 12 个示例脚本，直接运行会 uvicorn.run(...) 起服务并**一直阻塞**，
    没法被 verify_all.py 批量跑。于是：
        · 每个示例都提供 `--check` 快速自检模式（不阻塞）；
        · 本文件再用 TestClient 把它们的接口**统一请求一遍**，
          打印「请求方法 + 路径 + 状态码 + 响应体」，并用 assert 校验状态码。
    两重保险：示例出错时，verify_all.py 一定会失败。

**import 每个示例的方式**：文件名以数字开头（01_最小应用.py），不能用普通 import，
所以用 importlib.util.spec_from_file_location 按路径加载，并注册进 sys.modules
（注册是必须的：dataclass 等机制会依赖 module.__dict__）。

本节知识点：
    1. TestClient 的用法与"不启服务也能端到端测试"的原理
    2. 用 importlib 按路径动态加载模块（处理"文件名不能作为模块名"的情况）
    3. 表驱动测试：把"请求 → 期望状态码"写成数据，新增接口只需加一行
    4. 跨请求传递状态（先登录拿 token，再带着 token 访问受保护接口）

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\verify_api.py'
    （它也会被 Back_End/verify_all.py 自动调用）
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import time
import traceback
from typing import Any

# ------------------------------------------------------------
# 路径：本文件位于 Back_End/06_FastAPI/verify_api.py
# ------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
BACK_END = HERE.parent
ROOT = HERE.parents[1]
DATA_DIR = BACK_END / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FASTAPI_DIR = HERE
BOOK_API_DIR = BACK_END / "10_综合实战" / "book_api"


# ============================================================
# 一、示例清单与请求表
# ============================================================
# 每一项：(脚本相对路径, 说明, [请求...], 客户端参数)
# 请求字段：
#     method    HTTP 方法
#     path      路径
#     expect    期望状态码
#     json/data/params/headers  传给 TestClient 的参数
#     save      把响应 JSON 存进上下文（供后续请求用 {xxx} 占位符引用）
#     note      额外说明
CHECKS: list[tuple[str, str, list[dict[str, Any]], dict[str, Any]]] = [
    # ---------------- 01 最小应用 ----------------
    ("06_FastAPI/01_最小应用.py", "最小应用与自动文档", [
        {"method": "GET", "path": "/", "expect": 200},
        {"method": "GET", "path": "/items/42", "expect": 200},
        {"method": "GET", "path": "/items/abc", "expect": 422, "note": "路径参数类型错误"},
        {"method": "GET", "path": "/health", "expect": 200},
        {"method": "GET", "path": "/openapi.json", "expect": 200},
        {"method": "GET", "path": "/docs", "expect": 200},
    ], {}),

    # ---------------- 02 GET 与 POST ----------------
    ("06_FastAPI/02_GET与POST.py", "参数来源与请求体", [
        {"method": "GET", "path": "/items/5", "params": {"page-size": 20}, "expect": 200},
        {"method": "GET", "path": "/items", "params": {"in_stock": "true", "sort": "price"}, "expect": 200},
        {"method": "GET", "path": "/items", "params": {"sort": "bad"}, "expect": 422},
        {"method": "POST", "path": "/items", "json": {"name": "机械键盘", "price": 399.0, "tax": 13}, "expect": 201},
        {"method": "POST", "path": "/items", "json": {"name": "免费", "price": 0}, "expect": 422},
        {"method": "PUT", "path": "/items/7", "params": {}, "json": {"name": "显示器", "price": 999}, "expect": 200},
        {"method": "POST", "path": "/login/form",
         "data": {"username": "alice", "password": "password123"}, "expect": 200},
        {"method": "GET", "path": "/whoami",
         "headers": {"X-Token": "tk-1"}, "expect": 200},
    ], {}),

    # ---------------- 03 数据模型与验证 ----------------
    ("06_FastAPI/03_数据模型与验证.py", "Pydantic 校验", [
        {"method": "POST", "path": "/users", "expect": 201, "save": "user",
         "json": {"userName": "alice", "email": "alice@example.com", "salary": 12000,
                  "phone": "13800138000", "hobbies": ["阅读", "跑步"]}},
        {"method": "POST", "path": "/users", "expect": 422, "note": "邮箱格式非法",
         "json": {"userName": "bob", "email": "not-an-email", "salary": 100, "phone": "13800138000"}},
        {"method": "POST", "path": "/users", "expect": 422, "note": "手机号格式非法",
         "json": {"userName": "carol", "email": "c@example.com", "salary": 100, "phone": "12345"}},
        {"method": "GET", "path": "/users/alice", "expect": 200},
        {"method": "GET", "path": "/users/nobody", "expect": 404},
        {"method": "POST", "path": "/orders", "expect": 200,
         "json": {"amount": 199.5, "tags": ["加急"], "extra": {"coupon": 10}}},
        {"method": "GET", "path": "/users/alice/schema", "expect": 200},
    ], {}),

    # ---------------- 04 模板与静态文件 ----------------
    ("06_FastAPI/04_模板与静态文件.py", "Jinja2 模板与静态资源", [
        {"method": "GET", "path": "/", "expect": 200},
        {"method": "GET", "path": "/items/42", "expect": 200},
        {"method": "GET", "path": "/static/style.css", "expect": 200},
        {"method": "GET", "path": "/static/a.svg", "expect": 200},
        {"method": "GET", "path": "/api/products", "expect": 200},
    ], {}),

    # ---------------- 05 异步处理 ----------------
    ("06_FastAPI/05_异步处理.py", "异步与并发模型", [
        {"method": "GET", "path": "/async", "expect": 200},
        {"method": "GET", "path": "/sync", "expect": 200},
        {"method": "GET", "path": "/threadpool", "expect": 200},
        {"method": "GET", "path": "/with-dep", "expect": 200},
        {"method": "POST", "path": "/orders", "params": {"item": "机械键盘"}, "expect": 200},
    ], {}),

    # ---------------- 06 数据库集成 ----------------
    ("06_FastAPI/06_数据库集成.py", "SQLAlchemy 2.0 + SQLite", [
        {"method": "GET", "path": "/users", "expect": 200},
        {"method": "POST", "path": "/users", "expect": 201, "save": "db_user",
         "json": {"name": "校验用户", "email": "verify_{ts}@example.com"}},
        {"method": "GET", "path": "/users/{saved:db_user.id}", "expect": 200},
        {"method": "POST", "path": "/users/{saved:db_user.id}/articles",
         "params": {"title": "由 verify_api 创建的文章"}, "expect": 201},
        {"method": "GET", "path": "/stats", "expect": 200},
        {"method": "GET", "path": "/users/999999", "expect": 404},
        {"method": "GET", "path": "/users", "params": {"limit": 0}, "expect": 422},
    ], {}),

    # ---------------- 07 依赖注入 ----------------
    ("06_FastAPI/07_依赖注入.py", "Depends 各种用法", [
        {"method": "GET", "path": "/items", "params": {"skip": 1, "limit": 2}, "expect": 200},
        {"method": "GET", "path": "/conn", "expect": 200},
        {"method": "GET", "path": "/products", "expect": 200},
        {"method": "GET", "path": "/me", "expect": 401, "note": "缺少 Bearer Token"},
        {"method": "GET", "path": "/me", "headers": {"Authorization": "Bearer token-alice"}, "expect": 200},
        {"method": "GET", "path": "/admin", "headers": {"Authorization": "Bearer token-alice"}, "expect": 403},
        {"method": "GET", "path": "/admin", "headers": {"Authorization": "Bearer token-root"}, "expect": 200},
        {"method": "GET", "path": "/cache-demo", "expect": 200},
        {"method": "GET", "path": "/api/admin/users",
         "headers": {"Authorization": "Bearer token-root"}, "expect": 200},
        {"method": "GET", "path": "/need-db", "expect": 200},
    ], {}),

    # ---------------- 08 认证与授权 ----------------
    ("06_FastAPI/08_认证与授权.py", "JWT 认证与角色授权", [
        {"method": "POST", "path": "/token", "expect": 200, "save": "token",
         "data": {"username": "alice", "password": "alice123"}},
        {"method": "POST", "path": "/token", "expect": 401, "note": "密码错误",
         "data": {"username": "alice", "password": "wrong"}},
        {"method": "GET", "path": "/profile", "expect": 401, "note": "未带 Token"},
        {"method": "GET", "path": "/profile", "expect": 200,
         "headers": {"Authorization": "Bearer {saved:token.access_token}"}},
        {"method": "GET", "path": "/admin/users", "expect": 403,
         "headers": {"Authorization": "Bearer {saved:token.access_token}"}},
        {"method": "POST", "path": "/token", "expect": 200, "save": "admin_token",
         "data": {"username": "root", "password": "root123"}},
        {"method": "GET", "path": "/admin/users", "expect": 200,
         "headers": {"Authorization": "Bearer {saved:admin_token.access_token}"}},
        {"method": "GET", "path": "/me/orders", "expect": 200,
         "headers": {"Authorization": "Bearer {saved:token.access_token}"}},
    ], {}),

    # ---------------- 09 中间件 ----------------
    ("06_FastAPI/09_中间件.py", "CORS / GZip / 自定义中间件", [
        {"method": "GET", "path": "/", "expect": 200},
        {"method": "GET", "path": "/big", "expect": 200, "headers": {"Accept-Encoding": "gzip"}},
        {"method": "GET", "path": "/small", "expect": 200},
        {"method": "OPTIONS", "path": "/", "expect": 200,
         "headers": {"Origin": "http://localhost:3000",
                     "Access-Control-Request-Method": "POST"}},
        {"method": "GET", "path": "/boom", "expect": 500},
    ], {}),

    # ---------------- 10 错误处理 ----------------
    ("06_FastAPI/10_错误处理.py", "统一错误处理", [
        {"method": "POST", "path": "/orders", "expect": 200,
         "json": {"product_id": 3, "quantity": 1}},
        {"method": "POST", "path": "/orders", "expect": 404, "note": "商品不存在",
         "json": {"product_id": 999, "quantity": 1}},
        {"method": "POST", "path": "/orders", "expect": 409, "note": "商品售罄",
         "json": {"product_id": 2, "quantity": 1}},
        {"method": "POST", "path": "/orders", "expect": 400, "note": "库存不足（自定义业务异常）",
         "json": {"product_id": 1, "quantity": 99}},
        {"method": "POST", "path": "/orders", "expect": 422, "note": "缺字段",
         "json": {"product_id": 1}},
        {"method": "GET", "path": "/items/-1", "expect": 400},
        {"method": "GET", "path": "/items/999", "expect": 404},
        {"method": "GET", "path": "/crash", "expect": 500, "note": "兜底处理器"},
        {"method": "GET", "path": "/no-such-path", "expect": 404},
    ], {"raise_server_exceptions": False}),

    # ---------------- 11 Session 与 Cookie ----------------
    ("06_FastAPI/11_Session与Cookie.py", "Session / Cookie / 签名 Cookie", [
        {"method": "GET", "path": "/login", "expect": 200},
        {"method": "GET", "path": "/profile", "expect": 401, "note": "未登录"},
        {"method": "POST", "path": "/login", "expect": 302, "follow_redirects": False,
         "data": {"username": "alice", "password": "password123"}},
        {"method": "GET", "path": "/profile", "expect": 200, "note": "已登录（客户端自动带 Cookie）"},
        {"method": "GET", "path": "/set-cookie", "expect": 200},
        {"method": "GET", "path": "/read-cookie", "expect": 200},
        {"method": "GET", "path": "/set-signed-cookie", "expect": 200},
        {"method": "GET", "path": "/read-signed-cookie", "expect": 200},
        {"method": "GET", "path": "/logout", "expect": 302, "follow_redirects": False},
        {"method": "GET", "path": "/profile", "expect": 401, "note": "退出后"},
    ], {}),

    # ---------------- 12 Celery 后台任务 ----------------
    ("06_FastAPI/12_Celery后台任务.py", "Celery + Redis（eager 模式）", [
        {"method": "POST", "path": "/send-email", "expect": 200, "save": "email_task",
         "params": {"email": "verify@example.com", "content": "校验用邮件"}},
        {"method": "GET", "path": "/task-status/{saved:email_task.task_id}", "expect": 200},
        {"method": "GET", "path": "/task-status/不存在-的任务", "expect": 200},
        {"method": "POST", "path": "/report", "params": {"rows": 50}, "expect": 200},
        {"method": "GET", "path": "/health", "expect": 200},
    ], {}),

    # ---------------- 10_综合实战 book_api ----------------
    ("10_综合实战/book_api/main.py", "综合实战：图书 CRUD 项目", [
        {"method": "GET", "path": "/health", "expect": 200},
        {"method": "GET", "path": "/", "expect": 200},
        {"method": "GET", "path": "/api/books", "expect": 200},
        {"method": "GET", "path": "/api/books/99999999", "expect": 404},
        {"method": "POST", "path": "/api/books", "expect": 201, "save": "book",
         "json": {"title": "verify_api 创建的书", "author": "verify",
                  "isbn": "978-7-999-{ts}", "price": 66.0, "stock": 2}},
        {"method": "GET", "path": "/api/books/{saved:book.data.id}", "expect": 200},
        {"method": "PATCH", "path": "/api/books/{saved:book.data.id}", "expect": 200,
         "json": {"stock": 9}},
        {"method": "DELETE", "path": "/api/books/{saved:book.data.id}", "expect": 200},
        {"method": "GET", "path": "/api/books/{saved:book.data.id}", "expect": 404},
        {"method": "GET", "path": "/docs", "expect": 200},
        {"method": "GET", "path": "/openapi.json", "expect": 200},
    ], {}),
]


# ============================================================
# 二、工具函数
# ============================================================
def load_module(path: pathlib.Path, alias: str):
    """按文件路径动态加载 Python 模块。

    为什么要这么麻烦？
        · 文件名是 "01_最小应用.py"，不是合法的标识符，`import 01_最小应用` 直接语法错误；
        · 必须先把模块注册进 sys.modules 再 exec_module，
          否则 dataclass / pydantic 之类依赖 module.__dict__ 的机制会报错。
    """
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为 {path} 创建模块规格")
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module          # 关键一步
    spec.loader.exec_module(module)      # 执行模块代码（此时 app 被构造出来）
    return module


def render(value: Any, ctx: dict[str, Any], stamp: str) -> Any:
    """把请求参数里的占位符替换成真实值。

    支持的占位符：
        {ts}                    本次运行的时间戳（用于生成唯一邮箱等）
        {saved:键.字段.子字段}    引用前面某个请求保存下来的响应内容
    """
    if isinstance(value, str):
        value = value.replace("{ts}", stamp)
        while "{saved:" in value:
            start = value.index("{saved:")
            end = value.index("}", start)
            expr = value[start + len("{saved:"):end]
            key, *rest = expr.split(".")
            data = ctx.get(key)
            for field in rest:
                if isinstance(data, dict):
                    data = data.get(field)
                else:
                    data = None
            value = value[:start] + str(data) + value[end + 1:]
        return value
    if isinstance(value, dict):
        return {k: render(v, ctx, stamp) for k, v in value.items()}
    if isinstance(value, list):
        return [render(v, ctx, stamp) for v in value]
    return value


def shorten(text: str, limit: int = 150) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit] + "…"


# ============================================================
# 三、主流程
# ============================================================
def main() -> int:
    from fastapi.testclient import TestClient

    stamp = str(int(time.time()))
    ctx: dict[str, Any] = {}

    print("=" * 78)
    print("FastAPI 全量接口校验（TestClient 逐个示例发请求）")
    print("=" * 78)
    print(f"解释器：{sys.executable}")
    print(f"示例目录：{FASTAPI_DIR}")
    print(f"请求表共 {len(CHECKS)} 个应用、{sum(len(c[2]) for c in CHECKS)} 个请求")
    print()

    total_apps = 0
    total_requests = 0
    failures: list[str] = []

    for index, (rel_path, desc, requests, client_kwargs) in enumerate(CHECKS, start=1):
        script = ROOT / "Back_End" / rel_path
        print("-" * 78)
        print(f"[{index}/{len(CHECKS)}] {rel_path}")
        print(f"        说明：{desc}")

        if not script.exists():
            print(f"        ✗ 脚本不存在：{script}")
            failures.append(f"{rel_path}：脚本不存在")
            continue

        # ---------- 加载模块 ----------
        try:
            t0 = time.perf_counter()
            module = load_module(script, f"verify_mod_{index}")
            load_cost = (time.perf_counter() - t0) * 1000
        except Exception as exc:
            print(f"        ✗ 模块导入失败：{type(exc).__name__}: {exc}")
            print(shorten(traceback.format_exc(), 600))
            failures.append(f"{rel_path}：导入失败 {type(exc).__name__}")
            continue

        app = getattr(module, "app", None)
        if app is None:
            print("        ✗ 模块里没有 app 对象")
            failures.append(f"{rel_path}：缺少 app")
            continue

        print(f"        导入成功（{load_cost:.0f} ms），app.title = {getattr(app, 'title', '(无标题)')}")
        total_apps += 1

        # ---------- 逐个请求 ----------
        with TestClient(app, **client_kwargs) as client:
            for req in requests:
                method = req["method"]
                path = render(req["path"], ctx, stamp)
                expect = req["expect"]
                kwargs: dict[str, Any] = {}
                for key in ("json", "data", "params", "headers", "content", "follow_redirects"):
                    if key in req:
                        kwargs[key] = render(req[key], ctx, stamp)

                try:
                    resp = client.request(method, path, **kwargs)
                except Exception as exc:
                    print(f"        ✗ {method} {path} 请求异常：{type(exc).__name__}: {exc}")
                    failures.append(f"{rel_path} {method} {path}：请求异常")
                    continue

                total_requests += 1
                ok = resp.status_code == expect
                mark = "✓" if ok else "✗"
                print(f"        {mark} {method:<7} {path:<46} -> {resp.status_code}（期望 {expect}）"
                      + (f"  ｜ {req['note']}" if req.get("note") else ""))
                print(f"            响应 {shorten(resp.text)}")

                if not ok:
                    failures.append(f"{rel_path} {method} {path}：期望 {expect} 实际 {resp.status_code}")

                # 保存响应，供后续请求引用
                if "save" in req:
                    try:
                        ctx[req["save"]] = resp.json()
                    except Exception:
                        ctx[req["save"]] = {}

            # 顺带校验 OpenAPI 文档可生成（FastAPI 的强项，文档错了说明模型有问题）
            try:
                spec = client.get("/openapi.json")
                if spec.status_code == 200:
                    print(f"        ✓ OpenAPI 文档生成成功，共 {len(spec.json().get('paths', {}))} 个路径")
                else:
                    print(f"        · OpenAPI 文档状态码 {spec.status_code}（该应用可能未启用文档）")
            except Exception as exc:
                print(f"        · 跳过 OpenAPI 检查：{type(exc).__name__}")

            # 11_Session 的 Cookie 状态会影响后续断言，这里关闭客户端后 Cookie 自然失效
        print()

    # ---------- 汇总 ----------
    print("=" * 78)
    print("接口校验汇总")
    print("=" * 78)
    print(f"参与校验的应用：{total_apps} / {len(CHECKS)}")
    print(f"实际发出的请求：{total_requests}")
    if failures:
        print(f"失败项：{len(failures)}")
        for item in failures:
            print(f"  ✗ {item}")
        print("\n结论：接口校验未全部通过 ✗")
        return 1

    print("失败项：0")
    print("\n结论：所有示例接口校验通过 ✓（每个示例的详细讲解见对应脚本的 --check 输出）")
    return 0


if __name__ == "__main__":
    sys.exit(main())

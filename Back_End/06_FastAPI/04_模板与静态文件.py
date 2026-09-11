"""FastAPI 模板与静态文件：Jinja2 服务端渲染
================================================================
对应课案章节：后端开发基础 → Web框架 → FastAPI（模板与静态文件）

本节知识点：
    1. FastAPI 本身不绑定模板引擎，通过 Jinja2Templates 使用 Jinja2
    2. app.mount("/static", StaticFiles(directory=...), name="static")：
       把目录挂载成静态资源，由 Starlette 直接读文件返回，不经过视图函数
    3. TemplateResponse(request=request, name=..., context=...)：渲染模板
       （FastAPI 0.108+ 推荐把 request 作为关键字参数传入）
    4. response_class=HTMLResponse 告诉文档"这个接口返回 HTML"
    5. JSON 接口与 HTML 页面可以在同一个应用里共存，按路径区分
    6. 绝对路径指定模板/静态目录，避免"换个工作目录就找不到文件"

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\04_模板与静态文件.py' --check
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\04_模板与静态文件.py'
    然后浏览器打开 http://127.0.0.1:8104/items/42
"""

from __future__ import annotations

import datetime as dt
import pathlib
import sys

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

HERE = pathlib.Path(__file__).resolve().parent
BACK_END = HERE.parent
ROOT = HERE.parents[1]
DATA_DIR = BACK_END / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PORT = 8104

# 模板与静态目录（用绝对路径，与当前工作目录无关）
TEMPLATE_DIR = HERE / "templates"
STATIC_DIR = HERE / "static"

app = FastAPI(
    title="04 模板与静态文件",
    description="后端开发基础 · FastAPI 示例：Jinja2 模板与静态资源",
    version="1.0.0",
)

# ============================================================
# 挂载静态目录
# ============================================================
# StaticFiles 做的事：把 URL 前缀 /static 映射到磁盘目录。
# 请求 /static/style.css → 读取 STATIC_DIR/style.css → 返回，全程不进入 Python 视图函数。
# 生产环境这一层通常交给 Nginx，性能更好、还能做缓存和压缩。
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Jinja2Templates 是 FastAPI 对 Jinja2 的封装，构造时传入模板目录
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


# ============================================================
# 页面路由
# ============================================================
@app.get("/", response_class=HTMLResponse, summary="首页（渲染模板）", tags=["页面"])
def home(request: Request) -> HTMLResponse:
    """渲染首页模板。

    TemplateResponse(request=request, name="index.html", context={...})
        - request：模板里可以用 request.url_for(...) 生成 URL，也用于处理 URL 根路径；
        - name：相对模板目录的文件名；
        - context：模板变量字典。

    response_class=HTMLResponse 有两个作用：
        1) 让 /docs 知道这个接口返回 HTML；
        2) 让 FastAPI 用 HTMLResponse 包装返回值。
    """
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "app_title": "FastAPI 模板示例",
            "id": "home",
            "now": dt.datetime.now(),
            "nav": [
                {"label": "首页", "url": "/"},
                {"label": "商品 42", "url": "/items/42"},
                {"label": "商品 7", "url": "/items/7"},
            ],
            "products": [
                {"name": "机械键盘", "price": 399.0, "stock": 26},
                {"name": "人体工学椅", "price": 1299.0, "stock": 3},
                {"name": "显示器支架", "price": 159.0, "stock": 0},
            ],
        },
    )


@app.get("/items/{id}", response_class=HTMLResponse, summary="动态页面（课案原例）", tags=["页面"])
def read_item(request: Request, id: str) -> HTMLResponse:
    """课案里的经典示例：模板里显示 Item ID，并引入 /static 下的图片。

    这里把 id 做了简单处理：数字就当作商品 ID，其他字符串原样展示。
    """
    context = {
        "app_title": "FastAPI 模板示例",
        "id": id,
        "now": dt.datetime.now(),
        "nav": [{"label": "首页", "url": "/"}, {"label": f"商品 {id}", "url": f"/items/{id}"}],
        "products": [],
    }
    if id.isdigit():
        idx = int(id)
        all_products = [
            {"name": "机械键盘", "price": 399.0, "stock": 26},
            {"name": "人体工学椅", "price": 1299.0, "stock": 3},
            {"name": "显示器支架", "price": 159.0, "stock": 0},
        ]
        if 1 <= idx <= len(all_products):
            context["products"] = [all_products[idx - 1]]
    return templates.TemplateResponse(request=request, name="index.html", context=context)


@app.get("/api/products", summary="同一个应用里的 JSON 接口", tags=["接口"])
def api_products() -> dict:
    """同一份数据，既可以用模板渲染成页面，也可以作为 JSON 返回。

    真实项目里通常这样分工：
        /api/**    只返回 JSON（给前端框架/移动端）
        /admin/**  返回 HTML（服务端渲染的后台管理页）
    两种方式共存，按场景选择，不必非此即彼。
    """
    return {
        "total": 3,
        "items": [
            {"name": "机械键盘", "price": 399.0, "stock": 26},
            {"name": "人体工学椅", "price": 1299.0, "stock": 3},
            {"name": "显示器支架", "price": 159.0, "stock": 0},
        ],
        "提示": "这个接口返回 JSON，而 /items/42 返回渲染后的 HTML",
    }


# ============================================================
# 自检
# ============================================================
def run_self_check() -> None:
    from fastapi.testclient import TestClient

    print("=" * 72)
    print("FastAPI 模板与静态文件 · 自检")
    print("=" * 72)
    print(f"模板目录：{TEMPLATE_DIR}（存在：{(TEMPLATE_DIR / 'index.html').exists()}）")
    print(f"静态目录：{STATIC_DIR}（存在：{STATIC_DIR.exists()}）")
    print()

    client = TestClient(app)

    # ---------- 1. 首页 ----------
    resp = client.get("/")
    html = resp.text
    print("[1] GET /  渲染 index.html")
    print(f"    状态码 {resp.status_code}（期望 200）")
    print(f"    Content-Type：{resp.headers.get('Content-Type')}")
    print(f"    HTML 长度：{len(html)} 字符")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("Content-Type", "")
    assert "FastAPI 模板示例" in html

    # ---------- 2. 课案原例：/items/{id} ----------
    resp = client.get("/items/42")
    html42 = resp.text
    print("\n[2] GET /items/42  课案原例（模板里显示 Item ID）")
    print(f"    状态码 {resp.status_code}（期望 200）")
    assert resp.status_code == 200
    assert "Item ID: 42" in html42, "模板里应当渲染出 Item ID: 42"
    print("    ✓ 页面里出现了「Item ID: 42」，说明 context 变量注入成功")

    # ---------- 3. 模板块与循环 ----------
    assert "<html" in html42 and "site-footer" in html42, "应当渲染完整的 HTML 骨架"
    print("\n[3] 模板中的 for/if 渲染")
    resp = client.get("/")
    assert "库存紧张" in resp.text and "缺货" in resp.text
    print("    ✓ 首页表格中出现了三种库存状态标签（充足/紧张/缺货）")

    # ---------- 4. 静态文件 ----------
    resp = client.get("/static/style.css")
    print("\n[4] GET /static/style.css  （StaticFiles 挂载）")
    print(f"    状态码 {resp.status_code}（期望 200），Content-Type={resp.headers.get('Content-Type')}")
    print(f"    内容长度：{len(resp.text)} 字符")
    assert resp.status_code == 200
    assert "--accent" in resp.text

    resp = client.get("/static/a.svg")
    print("\n[5] GET /static/a.svg")
    print(f"    状态码 {resp.status_code}（期望 200），Content-Type={resp.headers.get('Content-Type')}")
    assert resp.status_code == 200 and "<svg" in resp.text
    print("    ✓ 图片由 StaticFiles 直接从磁盘返回，未经过任何视图函数")

    # 静态目录下不存在的文件 → 404
    resp = client.get("/static/not-exist.css")
    print("\n[6] GET /static/not-exist.css")
    print(f"    状态码 {resp.status_code}（期望 404）")
    assert resp.status_code == 404

    # ---------- 5. JSON 接口共存 ----------
    resp = client.get("/api/products")
    print("\n[7] GET /api/products  同一个应用里的 JSON 接口")
    print(f"    状态码 {resp.status_code}（期望 200），Content-Type={resp.headers.get('Content-Type')}")
    print(f"    响应：{resp.text[:120]}…")
    assert resp.status_code == 200 and "application/json" in resp.headers.get("Content-Type", "")

    print("\n" + "=" * 72)
    print("自检全部通过 ✓")
    print("=" * 72)
    print("知识点回顾：")
    print("1. Jinja2Templates(directory=...) + TemplateResponse 完成服务端渲染")
    print("2. StaticFiles 挂载后，静态资源由 Starlette 直接返回，不消耗业务逻辑时间")
    print("3. response_class=HTMLResponse 让文档正确标注返回类型")
    print("4. 同一应用里 HTML 页面与 JSON 接口完全可以共存，用路径区分")
    print("5. 模板目录/静态目录一定要用绝对路径，避免工作目录变化导致 500")
    print()


def main() -> None:
    print()
    print("#" * 72)
    print("# FastAPI 模板与静态文件（后端开发基础 · Web框架章节）")
    print("#" * 72)
    print(f"解释器：{sys.executable}")
    print(f"计划监听端口：{PORT}")
    print()

    if "--check" in sys.argv:
        print("提示：--check 模式，执行自检，不启动服务器、不阻塞。")
        print()
        run_self_check()
    else:
        print(f"启动 uvicorn：http://127.0.0.1:{PORT}/items/42")
        print(f"交互文档：http://127.0.0.1:{PORT}/docs")
        print("按 Ctrl+C 停止。")
        uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")


if __name__ == "__main__":
    main()

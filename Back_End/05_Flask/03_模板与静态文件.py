"""Flask 模板与静态文件：服务端渲染（SSR）入门
================================================================
对应课案章节：后端开发基础 → Web框架 → Flask（模板与静态文件）

本节知识点：
    1. 为什么需要模板引擎：把 HTML 从 Python 字符串里解放出来，实现"数据 + 模板 = 页面"
    2. render_template()：Flask 用 Jinja2 渲染 templates/ 下的文件
    3. Jinja2 三大语法：{{ 变量 }} / {% 语句 %} / {# 注释 #}
    4. 模板继承：{% extends %} + {% block %} —— 公共部分只写一次
    5. 常用过滤器：default / upper / length / join / round / truncate / format
    6. 控制结构：{% for %}（含 loop 变量与 for-else）、{% if / elif / else %}
    7. 自动转义：变量里的 <script> 会被转义成实体，这是 Jinja2 防 XSS 的默认行为
    8. 静态文件：static/ 目录由 Web 服务器直接返回，用 url_for('static', filename=...) 引用
    9. render_template vs jsonify：SSR 页面与前后端分离接口的选择

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\05_Flask\\03_模板与静态文件.py'
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\05_Flask\\03_模板与静态文件.py' --serve
"""

from __future__ import annotations

import datetime as dt
import logging
import pathlib
import sys
from dataclasses import dataclass

from flask import Flask, abort, render_template, send_from_directory

# ------------------------------------------------------------
# 路径：模板与静态目录都用绝对路径指定
# ------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
BACK_END = HERE.parent
DATA_DIR = BACK_END / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE_DIR = HERE / "templates"
STATIC_DIR = HERE / "static"

MISSING = []
if not (TEMPLATE_DIR / "index.html").exists():
    MISSING.append(str(TEMPLATE_DIR / "index.html"))
if not (STATIC_DIR / "style.css").exists():
    MISSING.append(str(STATIC_DIR / "style.css"))

app = Flask(__name__, template_folder=str(TEMPLATE_DIR), static_folder=str(STATIC_DIR))
app.json.ensure_ascii = False
app.logger.setLevel(logging.CRITICAL + 10)      # 抑制框架自带的异常堆栈输出（见 02 脚本注释）


# ============================================================
# 数据模型：用 dataclass 模拟"从数据库查出来的对象"
# ============================================================
@dataclass
class Product:
    """商品对象。

    在模板里可以用两种方式取值：
        {{ product.name }}    属性访问（推荐，模板更简洁）
        {{ product["name"] }} 下标访问
    """
    name: str
    price: float
    stock: int


PRODUCTS = [
    Product("机械键盘", 399.00, 26),
    Product("人体工学椅", 1299.00, 3),
    Product("显示器支架", 159.00, 0),
    Product("降噪耳机", 899.00, 12),
]


# ============================================================
# 路由
# ============================================================
@app.route("/")
def index():
    """渲染首页模板。

    render_template(template_name, **context)
        - template_name：相对 templates/ 的路径，如 "index.html"、"admin/list.html"
        - **context：要传给模板的变量（键名就是模板里的变量名）

    真实项目里更常见的写法是"查数据库 → 传给模板"：
        products = db.query(Product).all()
        return render_template("index.html", products=products)
    """
    return render_template(
        "index.html",
        title="Flask 模板演示首页",
        item_count=len(PRODUCTS) * 7,          # 随便造一个数字，演示变量输出
        now=dt.datetime.now(),
        # 故意传入一段脚本，观察 Jinja2 的自动转义（页面上只会看到文字，不会弹窗）
        danger="<script>alert('如果我被执行了就说明有 XSS 漏洞')</script>",
        empty_value=None,
        items=["机械键盘", "人体工学椅", "显示器支架"],
        products=PRODUCTS,
    )


@app.route("/about")
def about():
    """渲染"关于"页 —— 演示模板继承：两个页面共用 base.html。"""
    return render_template("about.html")


@app.route("/items/<int:item_id>")
def item_page(item_id: int):
    """模板里也能用路径参数。

    这里演示：数据不存在时用 abort(404) 主动返回 404，
    而不是渲染一个"内容为空"的页面（对 SEO 与监控都更友好）。
    """
    if item_id < 1 or item_id > len(PRODUCTS):
        abort(404, description=f"商品 {item_id} 不存在")
    product = PRODUCTS[item_id - 1]
    return render_template("index.html", title=f"商品 {item_id}", item_count=item_id,
                           now=dt.datetime.now(), danger="（安全内容）", empty_value="",
                           items=[product.name], products=[product])


@app.route("/static-file/<path:filename>")
def static_file(filename: str):
    """手动返回静态文件：send_from_directory。

    正常情况下 app 的 static_folder 已经自动处理 /static/<filename>，
    这里再演示一次"手动版"，是为了说明静态文件服务的原理。
    send_from_directory 会做目录穿越校验，比 open() + read 安全得多。
    """
    return send_from_directory(STATIC_DIR, filename)


# 统一的中文 404 页面：abort(404) 会走到这里
@app.errorhandler(404)
def not_found(error):
    desc = getattr(error, "description", "资源不存在")
    return (
        f"<h1 style='font-family:sans-serif'>404 · {desc}</h1>"
        f"<p style='font-family:sans-serif'><a href='/'>返回首页</a></p>",
        404,
    )


# ============================================================
# 自检
# ============================================================
def run_self_check() -> None:
    print("=" * 72)
    print("Flask 模板与静态文件 · 自检")
    print("=" * 72)
    print(f"模板目录：{TEMPLATE_DIR}")
    print(f"静态目录：{STATIC_DIR}")
    if MISSING:
        print("【注意】以下文件缺失（模板/静态文件应由仓库一起提供）：")
        for item in MISSING:
            print("   -", item)
    else:
        print("模板与静态文件齐备 ✓")
    print()

    client = app.test_client()

    # ---------- 1) 首页渲染 ----------
    resp = client.get("/")
    html = resp.get_data(as_text=True)
    print("[1] GET /  渲染 index.html（继承 base.html）")
    print(f"    状态码 {resp.status_code}（期望 200），HTML 长度 {len(html)} 字符")
    assert resp.status_code == 200
    assert "你好，Flask 模板！" in html
    assert "site-header" in html, "应当包含 base.html 里的页头"
    assert "site-footer" in html, "应当包含 base.html 里的页脚"

    # ---------- 2) 自动转义 ----------
    print("\n[2] 自动转义（XSS 防护）")
    assert "&lt;script&gt;" in html, "变量里的 <script> 应被转义成实体"
    assert "<script>alert" not in html, "原始 <script> 不应出现在页面源码里"
    print("    ✓ 变量中的 <script> 已被转义为 &lt;script&gt;，浏览器不会执行")

    # ---------- 3) 模板继承 ----------
    resp = client.get("/about")
    about_html = resp.get_data(as_text=True)
    print("\n[3] GET /about  同一个 base.html 的另一个子模板")
    print(f"    状态码 {resp.status_code}（期望 200）")
    assert resp.status_code == 200 and "关于本示例" in about_html
    assert "site-header" in about_html, "about 页面也应有公共页头（继承生效）"
    print("    ✓ 公共页头/页脚来自 base.html，子模板只写 content 块")

    # ---------- 4) 循环与条件渲染 ----------
    print("\n[4] for / if 渲染商品表格")
    for product in PRODUCTS:
        assert product.name in html, f"{product.name} 应当出现在页面上"
        print(f"    ✓ {product.name:<10} 价格 {product.price:>8.2f}  库存 {product.stock:>3}")
    assert "缺货" in html and "库存紧张" in html and "充足" in html, "三种状态标签都应渲染出来"

    # ---------- 5) 静态文件 ----------
    resp = client.get("/static/style.css")
    css = resp.get_data(as_text=True)
    print("\n[5] GET /static/style.css  静态文件由 Web 服务器直接返回")
    print(f"    状态码 {resp.status_code}（期望 200），Content-Type={resp.headers.get('Content-Type')}")
    print(f"    内容长度 {len(css)} 字符")
    assert resp.status_code == 200 and "--accent" in css

    resp = client.get("/static/logo.svg")
    svg = resp.get_data(as_text=True)
    print("\n[6] GET /static/logo.svg")
    print(f"    状态码 {resp.status_code}（期望 200），Content-Type={resp.headers.get('Content-Type')}")
    assert resp.status_code == 200 and "<svg" in svg

    # 手动版静态文件服务
    resp = client.get("/static-file/style.css")
    print("\n[7] GET /static-file/style.css  （send_from_directory 手动返回）")
    print(f"    状态码 {resp.status_code}（期望 200）")
    assert resp.status_code == 200

    # ---------- 6) 数据不存在 → 404 ----------
    resp = client.get("/items/99")
    print("\n[8] GET /items/99  数据不存在时 abort(404)")
    print(f"    状态码 {resp.status_code}（期望 404）")
    print(f"    页面提示：{resp.get_data(as_text=True)[:60]}…")
    assert resp.status_code == 404

    print("\n" + "=" * 72)
    print("自检全部通过 ✓")
    print("=" * 72)
    print("知识点回顾：")
    print("1. render_template('x.html', 变量=值) 把数据填充进模板")
    print("2. {% extends %} + {% block %} 实现模板继承，公共结构只写一次")
    print("3. Jinja2 默认开启自动转义，这是防 XSS 的第一道防线")
    print("4. static/ 由 Web 服务器直接返回，不消耗 Python 视图函数的处理时间")
    print("5. 页面型接口用模板、数据型接口用 JSON，两者按场景选择")
    print()


def main() -> None:
    print()
    print("#" * 72)
    print("# Flask 模板与静态文件（后端开发基础 · Web框架章节）")
    print("#" * 72)
    print(f"解释器：{sys.executable}")

    if "--serve" in sys.argv:
        print("启动开发服务器：http://127.0.0.1:5003  （Ctrl+C 停止）")
        app.run(host="127.0.0.1", port=5003, debug=True)
    else:
        print("提示：默认自检模式，不启动服务器。")
        print()
        run_self_check()


if __name__ == "__main__":
    main()

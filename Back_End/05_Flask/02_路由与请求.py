"""Flask 路由与请求详解
================================================================
对应课案章节：后端开发基础 → Web框架 → Flask（路由/请求处理）

本节知识点：
    1. 路由的本质：一张"URL 模式 → 视图函数"的映射表，Werkzeug 用正则编译后匹配
    2. 动态路由：<name>、<int:id>、<float:price>、<path:sub> 转换器
    3. HTTP 方法：methods=["GET","POST"]，以及 GET/POST 的语义区别
    4. 查询参数 request.args.get()，带默认值与类型转换
    5. 表单参数 request.form、JSON 体 request.get_json()
    6. 请求头 request.headers、Cookie request.cookies
    7. 响应：字符串 / dict / (body, status) / (body, status, headers) / make_response
    8. url_for() 反向生成 URL —— 改路由不用改模板，工程上非常重要
    9. 自定义转换器与 404/405 的错误处理

运行方式：
    # 自检模式（默认，不阻塞）
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\05_Flask\\02_路由与请求.py'
    # 启动服务器
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\05_Flask\\02_路由与请求.py' --serve
"""

from __future__ import annotations

import logging
import sys
import pathlib

from flask import Flask, Response, jsonify, make_response, request, url_for

HERE = pathlib.Path(__file__).resolve().parent
BACK_END = HERE.parent
DATA_DIR = BACK_END / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder=str(HERE / "templates"), static_folder=str(HERE / "static"))
app.json.ensure_ascii = False

# Flask 默认会把未处理异常的完整堆栈打到 stderr（"Exception on /boom [GET]" 那一大段）。
# 教学演示时我们希望输出干净，所以把 Flask 自带 logger 的级别抬高到 CRITICAL 以上。
# 生产环境恰恰相反：应该保留这条日志，并把它接到 loguru / 日志平台，同时用 errorhandler
# 给客户端返回"不含堆栈"的统一错误格式（避免泄露代码结构）。
app.logger.setLevel(logging.CRITICAL + 10)


# ============================================================
# 一、静态路由（路径写死）
# ============================================================
@app.route("/")
def index() -> str:
    """首页：返回一段说明文字，并用 url_for 列出本文件里所有的路由。

    url_for 的参数是**视图函数名**（endpoint），不是 URL。
    好处：以后改了路由路径，所有用 url_for 生成链接的地方都自动跟着变，不用全局搜索替换。
    """
    lines = [
        "Flask 路由与请求演示（URL 由 url_for 反向生成）",
        f"  {url_for('index')}                          首页（本页）",
        f"  {url_for('user_profile', username='alice')}       字符串参数",
        f"  {url_for('get_item', item_id=42)}            整数参数",
        f"  {url_for('get_price', price=3.5)}            浮点参数",
        f"  {url_for('read_file', sub='logs/app.log')}     路径参数（可含斜杠）",
        f"  {url_for('search')}?q=关键字&page=2        查询参数",
        f"  {url_for('notes')}                           GET 列表 / POST 创建",
        f"  {url_for('note_detail', note_id=7)}            PUT 更新 / DELETE 删除",
        f"  {url_for('show_headers')}                     查看请求头",
        f"  {url_for('make_response_demo')}               自定义响应头与 Cookie",
        f"  {url_for('boom')}                            故意抛异常（演示 500）",
    ]
    return "\n".join(lines)


# ============================================================
# 二、动态路由：类型转换器
# ============================================================
# Flask 内置转换器：
#   string （默认）不包含斜杠的字符串
#   int    整数（/items/abc 会直接 404，不会进函数）
#   float  浮点数
#   path   可以包含斜杠，常用于文件路径
#   uuid   UUID 字符串
@app.route("/user/<username>")
def user_profile(username: str):
    """string 转换器：任何不含斜杠的文本。"""
    return {"用户": username, "类型": "string"}


@app.route("/item/<int:item_id>")
def get_item(item_id: int):
    """int 转换器：Flask 自动把 URL 片段转成 int 并校验。

    注意：/item/abc 会返回 404，因为正则匹配不上，**根本不会调用这个函数**。
    这是"路由层校验"，比在函数里写 if 判断更早、更省资源。
    """
    return {"item_id": item_id, "类型": type(item_id).__name__}


@app.route("/price/<float:price>")
def get_price(price: float):
    """float 转换器。"""
    return {"price": price, "含税价": round(price * 1.13, 2)}


@app.route("/files/<path:sub>")
def read_file(sub: str):
    """path 转换器：允许匹配 /files/a/b/c.log 这种多级路径。

    安全提醒：真实项目里绝不能直接把 sub 拼成磁盘路径去读文件（目录穿越漏洞），
    必须做白名单校验或使用 safe_join。
    """
    return {"请求路径": sub, "说明": "path 转换器允许斜杠"}


# ============================================================
# 三、查询参数、请求方法
# ============================================================
@app.route("/search")
def search():
    """GET 查询参数：?q=xx&page=2&size=10

    查询参数永远是字符串，需要自己转类型：
        request.args.get("page", 1, type=int)   # 第三参数 type 让 Flask 帮你转并做容错
    """
    q = request.args.get("q", "", type=str)
    page = request.args.get("page", 1, type=int)
    size = request.args.get("size", 10, type=int)
    return {
        "关键字": q,
        "page": page,
        "size": size,
        "说明": "request.args.get(key, default, type=...) 会自动转换类型并兜底",
    }


@app.route("/notes", methods=["GET", "POST"])
def notes():
    """同一个 URL 用方法区分行为，这就是 RESTful 风格的雏形。

    GET    → 列出资源（安全、幂等，可被浏览器缓存）
    POST   → 创建资源（不幂等，参数在请求体里）
    """
    if request.method == "GET":
        return {"动作": "列出笔记", "方法": "GET"}
    # POST：先尝试 JSON，再尝试表单，兼容两种客户端
    payload = request.get_json(silent=True) or dict(request.form)
    return {"动作": "创建笔记", "方法": "POST", "收到的内容": payload}, 201


@app.route("/notes/<int:note_id>", methods=["PUT", "DELETE"])
def note_detail(note_id: int):
    """PUT 全量更新 / DELETE 删除 —— RESTful 的另外两个动词。"""
    return {"note_id": note_id, "方法": request.method}


@app.route("/headers")
def show_headers():
    """请求头与 Cookie：鉴权、内容协商、追踪都靠它。"""
    return {
        "User-Agent": request.headers.get("User-Agent"),
        "Content-Type": request.headers.get("Content-Type"),
        "自定义头 X-Trace-Id": request.headers.get("X-Trace-Id"),
        "cookie 里的 token": request.cookies.get("token"),
        "所有请求头": dict(request.headers),
    }


# ============================================================
# 四、响应的四种写法
# ============================================================
@app.route("/resp/str")
def resp_str():
    """① 直接返回字符串 → 200 + text/html。"""
    return "这是纯文本响应"


@app.route("/resp/tuple")
def resp_tuple():
    """② 返回元组 (body, status) 或 (body, status, headers)。

    这是 Flask 最常用的"顺手改状态码"写法。
    """
    return jsonify({"message": "创建成功", "id": 1001}), 201, {"X-New-Id": "1001"}


@app.route("/resp/make")
def make_response_demo():
    """③ make_response()：需要精细控制时使用（设置 Cookie、下载文件名等）。"""
    resp: Response = make_response(jsonify({"message": "自定义响应"}))
    resp.status_code = 202                      # 202 Accepted
    resp.headers["X-Custom-Header"] = "hello"
    resp.set_cookie("demo_cookie", "cookie-value", httponly=True, max_age=3600)
    return resp


@app.route("/resp/custom")
def resp_custom():
    """④ 直接构造 Response 对象，完全掌控响应体与 MIME 类型。"""
    csv_text = "姓名,年龄\n张三,20\n李四,22\n"
    return Response(csv_text, mimetype="text/csv", headers={"Content-Disposition": "attachment; filename=demo.csv"})


# ============================================================
# 五、错误处理
# ============================================================
@app.errorhandler(404)
def not_found(error):
    """自定义 404：默认的 HTML 错误页对 API 不友好，统一返回 JSON 更好。"""
    return jsonify({"error": "资源不存在", "path": request.path, "提示": "请检查 URL 是否正确"}), 404


@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({"error": "方法不允许", "method": request.method, "path": request.path}), 405


@app.route("/boom")
def boom():
    """故意抛异常，演示 500 的返回（生产环境要统一成 JSON 且不泄露堆栈）。"""
    raise RuntimeError("这是故意制造的内部错误，用于演示 500 处理")


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "服务器内部错误", "提示": "生产环境这里绝不返回堆栈信息"}), 500


# ============================================================
# 自检
# ============================================================
def run_self_check() -> None:
    print("=" * 72)
    print("Flask 路由与请求 · 自检")
    print("=" * 72)
    client = app.test_client()

    cases = [
        ("GET", "/", None, 200, "首页：列出所有路由"),
        ("GET", "/user/alice", None, 200, "string 转换器"),
        ("GET", "/item/42", None, 200, "int 转换器"),
        ("GET", "/item/abc", None, 404, "int 转换器不匹配 → 404"),
        ("GET", "/price/3.5", None, 200, "float 转换器"),
        ("GET", "/files/logs/2024/app.log", None, 200, "path 转换器（含斜杠）"),
        ("GET", "/search?q=fastapi&page=2&size=5", None, 200, "查询参数"),
        ("GET", "/notes", None, 200, "GET 列出资源"),
        ("POST", "/notes", {"title": "学习 Flask"}, 201, "POST 创建资源（JSON）"),
        ("PUT", "/notes/7", None, 200, "PUT 更新资源"),
        ("DELETE", "/notes/7", None, 200, "DELETE 删除资源"),
        ("GET", "/resp/str", None, 200, "响应写法①字符串"),
        ("GET", "/resp/tuple", None, 201, "响应写法②元组（自定义状态码+响应头）"),
        ("GET", "/resp/make", None, 202, "响应写法③make_response（含 Cookie）"),
        ("GET", "/resp/custom", None, 200, "响应写法④Response（text/csv）"),
        ("GET", "/boom", None, 500, "故意抛异常 → 500"),
    ]

    for method, path, payload, expect, desc in cases:
        if method == "GET":
            resp = client.get(path, headers={"X-Trace-Id": "trace-001"})
        elif method == "POST":
            resp = client.post(path, json=payload)
        elif method == "PUT":
            resp = client.put(path)
        else:
            resp = client.delete(path)
        body = resp.get_data(as_text=True).replace("\n", " ")
        if len(body) > 150:
            body = body[:150] + "…"
        mark = "✓" if resp.status_code == expect else "✗"
        print(f"\n{mark} [{desc}]")
        print(f"    {method} {path}")
        print(f"    状态码 {resp.status_code}（期望 {expect}）  Content-Type={resp.headers.get('Content-Type')}")
        print(f"    响应体 {body}")
        assert resp.status_code == expect, f"{method} {path} 期望 {expect}，实际 {resp.status_code}"

    # 单独验证自定义响应头与 Cookie
    resp = client.get("/resp/tuple")
    print("\n[响应头验证]")
    print(f"    X-New-Id = {resp.headers.get('X-New-Id')}")
    assert resp.headers.get("X-New-Id") == "1001"

    resp = client.get("/resp/make")
    print("\n[Cookie 验证]")
    set_cookie = resp.headers.get("Set-Cookie", "")
    print(f"    Set-Cookie = {set_cookie[:80]}…")
    assert "demo_cookie" in set_cookie

    # 域外路径：验证自定义 404 处理器返回 JSON
    resp = client.get("/who-am-i")
    print("\n[自定义 404 处理器]")
    print(f"    状态码 {resp.status_code}，响应体 {resp.get_data(as_text=True)}")
    assert resp.status_code == 404

    print("\n" + "=" * 72)
    print("自检全部通过 ✓")
    print("=" * 72)
    print("知识点回顾：")
    print("1. <int:id> 这类转换器在“路由匹配阶段”就完成校验，不合法直接 404")
    print("2. request.args / form / json / headers / cookies 是取参数的五个入口")
    print("3. 响应的四种写法：字符串、元组、make_response、Response")
    print("4. errorhandler 可以把框架默认的 HTML 错误页换成统一 JSON 格式")
    print("5. url_for 反向生成 URL，避免在代码里硬编码路径")
    print()


def main() -> None:
    print()
    print("#" * 72)
    print("# Flask 路由与请求（后端开发基础 · Web框架章节）")
    print("#" * 72)
    print(f"解释器：{sys.executable}")

    if "--serve" in sys.argv:
        print("启动开发服务器：http://127.0.0.1:5002  （Ctrl+C 停止）")
        app.run(host="127.0.0.1", port=5002, debug=True)
    else:
        print("提示：默认自检模式，不启动服务器。")
        print()
        run_self_check()


if __name__ == "__main__":
    main()

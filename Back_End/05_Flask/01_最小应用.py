"""Flask 最小应用：五分钟理解一个 Web 框架在做什么
================================================================
对应课案章节：后端开发基础 → Web框架 → Flask（最小应用）

本节知识点：
    1. Web 框架的本质：把"HTTP 请求 → Python 函数 → HTTP 响应"这条链路自动化
    2. Flask 是微框架（micro framework）：只给你路由、请求、响应，其余自己选
    3. Flask(__name__) 里的参数决定了"去哪里找模板和静态文件"
    4. @app.route 装饰器：把 URL 路径绑定到视图函数（view function）
    5. 视图函数返回值会被自动转成 HTTP 响应；返回字典会自动变成 JSON（Flask 2.2+）
    6. app.run()：内置开发服务器（Werkzeug），只能用于开发，不能上生产
    7. app.test_client()：不启服务、不占端口就能测试接口（本脚本默认用它自检）

运行方式（PowerShell，必须用本仓库的虚拟环境解释器）：
    # 默认：跑一遍 TestClient 自检并打印结果（不阻塞、不占端口）
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\05_Flask\\01_最小应用.py'

    # 真正启动开发服务器（会一直阻塞，Ctrl+C 停止）
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\05_Flask\\01_最小应用.py' --serve
"""

from __future__ import annotations

import json
import pathlib
import sys

from flask import Flask, jsonify, request

# ------------------------------------------------------------
# 路径：本文件位于 Back_End/05_Flask/01_最小应用.py
#   parents[0] = 05_Flask，parents[1] = Back_End
# 用绝对路径指定模板/静态目录，避免"换个工作目录就找不到文件"
# ------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
BACK_END = HERE.parent
DATA_DIR = BACK_END / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 创建应用对象
# ============================================================
# Flask(__name__)：__name__ 让 Flask 知道"当前模块的根目录在哪"，
# 它据此推导 templates/ 与 static/ 的位置。
# 这里额外显式传入绝对路径，做到与"当前工作目录"完全无关。
app = Flask(
    __name__,
    template_folder=str(HERE / "templates"),
    static_folder=str(HERE / "static"),
)

# 让 jsonify 不要把中文转成 \u4e2d\u6587，便于教学时直接阅读响应体
app.json.ensure_ascii = False


# ============================================================
# 路由：URL → 函数
# ============================================================
@app.route("/")
def hello():
    """根路径。

    视图函数返回字符串时，Flask 会自动包装成
        HTTP/1.1 200 OK
        Content-Type: text/html; charset=utf-8
    这样一个完整的响应。框架替我们做的事，正是"微框架"的价值。
    """
    return "Hello, Flask!"


@app.route("/health")
def health():
    """健康检查接口。

    生产环境的负载均衡、K8S 探针都会周期性请求这种接口，
    它必须"轻、快、不依赖外部服务"，否则会把整个服务判定为不健康。
    """
    return {"status": "ok", "service": "flask-demo"}


@app.route("/info")
def info():
    """返回运行时信息，演示 jsonify 的用法。

    jsonify() 与"直接 return dict"的区别：
        - jsonify 可以明确指定状态码、响应头；
        - 直接 return dict 更简洁（Flask 2.2+ 支持）。
    """
    payload = {
        "框架": "Flask",
        "版本": "3.x",
        "Python": sys.version.split()[0],
        "说明": "微框架：路由/请求/响应由框架提供，ORM、认证等自行选择扩展",
    }
    return jsonify(payload), 200


@app.route("/echo", methods=["GET", "POST"])
def echo():
    """把请求信息原样回显，用于观察"请求里到底有什么"。

    Flask 把这些信息封装在全局代理对象 request 里（线程/协程安全的上下文变量）：
        request.method    请求方法
        request.args      查询参数（URL 里 ?a=1）
        request.form      表单参数（application/x-www-form-urlencoded）
        request.json      JSON 请求体
        request.headers   请求头
    """
    data: dict = {
        "method": request.method,
        "查询参数": dict(request.args),
        "表单参数": dict(request.form),
        "客户端地址": request.remote_addr,
    }
    # request.get_json(silent=True)：请求体不是 JSON（或为空）时返回 None 而不是抛错
    data["JSON请求体"] = request.get_json(silent=True)
    return jsonify(data)


# ============================================================
# 自检：不启服务也能验证接口（test_client 会走完整 WSGI 调用链）
# ============================================================
def run_self_check() -> None:
    print("=" * 72)
    print("Flask 最小应用 · 自检（使用 app.test_client()，不启动服务器）")
    print("=" * 72)

    # test_client() 直接调用 WSGI 应用，不占端口、不阻塞，非常适合写自动化检查
    client = app.test_client()

    checks = [
        ("GET", "/", None, 200, "根路径返回问候语"),
        ("GET", "/health", None, 200, "健康检查返回 JSON"),
        ("GET", "/info", None, 200, "运行时信息"),
        ("POST", "/echo?a=1&b=2", {"表单": "值"}, 200, "回显请求信息"),
    ]

    for method, path, payload, expect, desc in checks:
        if method == "GET":
            resp = client.get(path)
        else:
            resp = client.post(path, json=payload)
        body = resp.get_data(as_text=True)
        body_show = body if len(body) < 200 else body[:200] + "…"
        print(f"\n[{desc}]")
        print(f"  请求：{method} {path}")
        print(f"  状态码：{resp.status_code}（期望 {expect}）")
        print(f"  响应体：{body_show}")
        assert resp.status_code == expect, f"{path} 期望 {expect}，实际 {resp.status_code}"

    # 演示 404：访问不存在的路径
    resp = client.get("/不存在的路径")
    print("\n[404 演示]")
    print(f"  请求：GET /不存在的路径")
    print(f"  状态码：{resp.status_code}（期望 404）")
    assert resp.status_code == 404

    # 演示 405：路径存在但方法不允许
    resp = client.delete("/")
    print("\n[405 演示]")
    print("  请求：DELETE /")
    print(f"  状态码：{resp.status_code}（期望 405，因为 / 只注册了 GET）")
    assert resp.status_code == 405

    print("\n" + "=" * 72)
    print("自检全部通过 ✓")
    print("=" * 72)
    print("知识点回顾：")
    print("1. Flask(__name__) 决定模板与静态文件的位置")
    print("2. @app.route 把 URL 绑定到视图函数，函数返回值就是 HTTP 响应")
    print("3. 返回 dict / jsonify 得到 JSON 响应，返回 str 得到 HTML 响应")
    print("4. 404 = 路径不存在；405 = 路径存在但方法不允许（框架自动处理）")
    print("5. app.test_client() 是写自动化测试的正确姿势，正式测试见 07_测试框架")
    print("6. 想真正启服务：给本脚本加 --serve 参数")
    print()


def main() -> None:
    print()
    print("#" * 72)
    print("# Flask 最小应用（后端开发基础 · Web框架章节）")
    print("#" * 72)
    print(f"解释器：{sys.executable}")

    if "--serve" in sys.argv:
        # ---------- 真正启动开发服务器（会阻塞在这里） ----------
        print("启动 Flask 开发服务器：http://127.0.0.1:5001")
        print("按 Ctrl+C 停止。注意：开发服务器性能差，生产必须用 gunicorn/uwsgi + Nginx。")
        app.run(host="127.0.0.1", port=5001, debug=True)
    else:
        # ---------- 默认走自检，保证"直接运行就有有意义的输出" ----------
        print("提示：本脚本默认执行【自检模式】，不启动服务器，不会阻塞。")
        print()
        run_self_check()


if __name__ == "__main__":
    main()

"""FastAPI Session 与 Cookie：让无状态的 HTTP"记住"用户
================================================================
对应课案章节：后端开发基础 → Web框架 → FastAPI（Session 和 Cookie）

**为什么需要它们？** HTTP 协议是**无状态**的：服务器处理完一个请求就忘了你是谁，
两次请求之间没有任何关联。但"登录后保持登录状态"必须让服务器记住用户，
于是有了两种思路：

| 方式 | 原理 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- | --- |
| JWT | 服务器签发令牌，令牌自带用户信息 | 无状态、可跨域、易水平扩展 | 签发后**无法撤销** | 前后端分离、微服务 |
| Session/Cookie | 服务器保存会话，浏览器只存会话 ID | 可随时撤销、控制精细 | 占用服务器存储、受同源限制 | 传统 Web 应用、后台系统 |

**Cookie 的关键属性**（面试常问）：
    HttpOnly  JS 读不到（document.cookie 拿不到）→ 防 XSS 窃取
    Secure    只在 HTTPS 下发送 → 防中间人窃听
    SameSite  Lax/Strict/None → 防 CSRF（跨站请求伪造）
    Max-Age   有效期；不设置则是"会话 Cookie"，关浏览器即失效
    Path/Domain 作用范围

**本示例演示两套写法**：
    1. starlette 的 SessionMiddleware：`request.session` 像字典一样用
    2. 手工读写 Cookie + itsdangerous 签名 Cookie：理解底层原理

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\11_Session与Cookie.py' --check
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\11_Session与Cookie.py'
    启动后浏览器打开 http://127.0.0.1:8111/login 看表单。
"""

from __future__ import annotations

import os
import pathlib
import secrets
import sys

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from itsdangerous import BadSignature, URLSafeSerializer
from starlette.middleware.sessions import SessionMiddleware

HERE = pathlib.Path(__file__).resolve().parent
BACK_END = HERE.parent
ROOT = HERE.parents[1]
DATA_DIR = BACK_END / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PORT = 8111

# ============================================================
# 密钥：生产环境必须来自配置/环境变量，且必须足够随机
# ============================================================
# 优先读环境变量 SESSION_SECRET；没有就用每次启动随机生成的密钥
# （随机密钥意味着"重启后旧会话全部失效"，开发阶段可以接受，生产必须配固定值）
SECRET_KEY = os.getenv("SESSION_SECRET") or secrets.token_urlsafe(32)

app = FastAPI(
    title="11 Session 与 Cookie",
    description="后端开发基础 · FastAPI 示例：SessionMiddleware / Cookie / 签名 Cookie",
    version="1.0.0",
)

# ============================================================
# 一、SessionMiddleware
# ============================================================
# 作用：给 request 加上 `session` 属性（像字典一样读写），
#       并自动把内容签名后写进 Cookie / 从 Cookie 里读回来。
#
# 实现细节（很重要，别搞混）：
#   · Starlette 的 SessionMiddleware 是**签名 Cookie** 方案：
#     会话数据经过 itsdangerous 签名后整个放在客户端 Cookie 里，
#     服务器不存任何东西 —— 所以它"防篡改"，但内容**是可读的**（Base64），
#     因此绝不能往 session 里放密码、身份证号等敏感信息。
#   · 要做到课案表格里说的"服务器存储会话、浏览器只存 session id"，
#     需要换成服务端会话（session id + Redis/数据库存储）。
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,          # 签名密钥，泄露 = 别人可以伪造任意会话
    session_cookie="session",       # Cookie 名
    max_age=3600,                   # 1 小时过期
    same_site="lax",                # 防 CSRF 的默认推荐值
    https_only=False,               # 生产（HTTPS）必须设 True
)

# ============================================================
# 二、手工 Cookie 用到的签名器
# ============================================================
# URLSafeSerializer：把字典转成 "base64签名.payload" 形式的字符串，
# 用同一个密钥才能验证通过；被篡改会抛 BadSignature。
serializer = URLSafeSerializer(SECRET_KEY, salt="manual-cookie")

# 模拟用户表（明文密码仅用于教学，真实项目必须哈希存储，见 08_认证与授权.py）
USERS = {
    "alice": "password123",
    "bob": "bob456",
}


# ============================================================
# 三、页面
# ============================================================
LOGIN_FORM = """
<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><title>登录</title></head>
<body style="font-family:sans-serif;max-width:520px;margin:40px auto">
  <h2>登录（Session 示例）</h2>
  <form method="post" action="/login">
    <p>用户名：<input name="username" value="alice"></p>
    <p>密码：<input name="password" type="password" value="password123"></p>
    <p><input type="submit" value="登录"></p>
  </form>
  <p>可用账号：alice / password123 ，bob / bob456</p>
  <p><a href="/profile">查看个人页</a> ｜ <a href="/show-cookie">查看原始 Cookie</a></p>
</body></html>
"""


def page(title: str, body: str, status_code: int = 200) -> HTMLResponse:
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="UTF-8"><title>{title}</title></head>
<body style="font-family:sans-serif;max-width:680px;margin:40px auto">
{body}
<hr><p><a href="/login">登录页</a> ｜ <a href="/profile">个人页</a> ｜
<a href="/logout">退出</a> ｜ <a href="/show-cookie">查看 Cookie</a></p>
</body></html>"""
    return HTMLResponse(html, status_code=status_code)


@app.get("/login", response_class=HTMLResponse, summary="登录表单", tags=["Session"])
async def login_form() -> HTMLResponse:
    return HTMLResponse(LOGIN_FORM)


@app.post("/login", summary="处理登录（写入 session）", tags=["Session"])
async def login(
    request: Request,
    username: str = Form(..., description="用户名"),
    password: str = Form(..., description="密码"),
) -> Response:
    """登录成功后把用户写进 session。

    `request.session` 的用法和字典完全一样：
        request.session["username"] = username
    SessionMiddleware 会在响应发出时自动把它签名并写进 Set-Cookie。
    """
    if USERS.get(username) != password:
        return page("登录失败", f"<h2>登录失败</h2><p>用户名或密码错误：{username}</p>", status_code=401)

    # 每次登录轮换 session 内容（防会话固定攻击：攻击者预先塞一个 session id 给受害者）
    request.session.clear()
    request.session["username"] = username
    request.session["role"] = "admin" if username == "alice" else "user"

    # 302 跳转：登录成功后跳个人页，浏览器会带上刚拿到的 Cookie
    return RedirectResponse(url="/profile", status_code=302)


@app.get("/profile", summary="个人页（读取 session）", tags=["Session"])
async def profile(request: Request) -> Response:
    """读取 session 判断登录状态。"""
    session = dict(request.session)          # 转成普通 dict 便于打印
    username = session.get("username")
    if not username:
        return page("未登录", "<h2>你还未登录</h2><p>请先 <a href='/login'>去登录</a></p>",
                    status_code=401)
    return page(
        "个人页",
        f"<h2>已登录：{username}</h2>"
        f"<p>角色：{session.get('role')}</p>"
        f"<p>session 内容：{session}</p>"
        f"<p>说明：Starlette 的 SessionMiddleware 把这份数据签名后放在 Cookie 里，"
        f"服务器端不存储 —— 所以内容可读、但不可篡改。</p>",
    )


@app.get("/logout", summary="退出（清空 session）", tags=["Session"])
async def logout(request: Request) -> Response:
    """退出登录：清空 session 内容即可。

    这正是 Session 相对 JWT 的最大优势：**可以随时让会话失效**。
    JWT 一旦签发，在过期前服务器无法主动撤销（只能靠黑名单，那又变成有状态了）。
    """
    request.session.clear()
    response = RedirectResponse(url="/profile", status_code=302)
    return response


@app.get("/show-cookie", summary="展示浏览器发来的原始 Cookie", tags=["调试"])
async def show_cookie(request: Request) -> Response:
    """把浏览器发来的原始 Cookie 和解析后的 session 都显示出来，方便理解机制。"""
    raw = request.headers.get("cookie", "(无)")
    session = dict(request.session)
    manual = request.cookies.get("manual_token", "(无)")
    return page(
        "Cookie 调试",
        f"<h2>浏览器发来的原始 Cookie</h2><pre style='white-space:pre-wrap'>{raw}</pre>"
        f"<h2>服务端解析出的 session</h2><pre>{session}</pre>"
        f"<h2>手工签名的 manual_token</h2><pre>{manual}</pre>"
        f"<p>注意：原始 Cookie 是 Base64 文本，复制去解码就能看到内容 —— "
        f"这就是「不能往 session 放敏感信息」的原因。</p>",
    )


# ============================================================
# 四、手工读写 Cookie（理解底层）
# ============================================================
@app.get("/set-cookie", summary="手工设置普通 Cookie", tags=["手工 Cookie"])
async def set_cookie() -> Response:
    """直接操作响应对象设置 Cookie。

    属性说明：
        httponly=True  JS 无法通过 document.cookie 读取（防 XSS 窃取）
        secure=False   本地 http 测试用；生产 https 环境必须 True
        samesite="lax" 防 CSRF（跨站 POST 不带 Cookie，普通链接跳转带）
        max_age=60     60 秒后过期
        path="/"       全站有效
    """
    response = JSONResponse(
        {"message": "已写入 theme 与 lang 两个 Cookie（theme 是 HttpOnly，lang 不是）"}
    )
    response.set_cookie("theme", "dark", max_age=60, httponly=True, samesite="lax", path="/")
    response.set_cookie("lang", "zh-CN", max_age=60, httponly=False, samesite="lax", path="/")
    return response


@app.get("/read-cookie", summary="读取 Cookie", tags=["手工 Cookie"])
async def read_cookie(request: Request) -> dict:
    """读取 Cookie：`request.cookies` 是一个字典。"""
    return {
        "所有 Cookie": dict(request.cookies),
        "theme": request.cookies.get("theme", "(未设置)"),
        "lang": request.cookies.get("lang", "(未设置)"),
        "说明": "theme 虽然设置了 HttpOnly，服务端照样能读到（HttpOnly 只限制 JS）",
    }


@app.get("/set-signed-cookie", summary="写入签名 Cookie", tags=["签名 Cookie"])
async def set_signed_cookie() -> Response:
    """签名 Cookie：内容可读，但改了就会被发现。

    典型用途：记住"未登录用户的偏好设置"、"一次性活动标记"。
    安全性来自签名而不是加密 —— 想连内容都藏起来，需要用加密（Fernet）。
    """
    payload = {"user": "alice", "level": 3}
    token = serializer.dumps(payload)                     # 生成 "payload.签名"
    response = JSONResponse({"message": "已写入签名 Cookie", "token": token, "原文": payload})
    response.set_cookie("manual_token", token, httponly=True, samesite="lax", max_age=600)
    return response


@app.get("/read-signed-cookie", summary="校验并读取签名 Cookie", tags=["签名 Cookie"])
async def read_signed_cookie(request: Request) -> dict:
    """校验签名：`serializer.loads()` 失败会抛 BadSignature。"""
    token = request.cookies.get("manual_token")
    if not token:
        return {"status": "未设置签名 Cookie", "提示": "先访问 /set-signed-cookie"}
    try:
        data = serializer.loads(token)
        return {"status": "签名有效", "数据": data}
    except BadSignature:
        return {"status": "签名无效", "提示": "Cookie 被篡改过，服务端拒绝信任它的内容"}


# ============================================================
# 自检
# ============================================================
def run_self_check() -> None:
    from fastapi.testclient import TestClient

    print("=" * 72)
    print("FastAPI Session 与 Cookie · 自检")
    print("=" * 72)

    # TestClient 会自动保存并回传 Cookie，行为与浏览器一致
    client = TestClient(app)

    def line(desc: str, resp, expect: int) -> None:
        mark = "✓" if resp.status_code == expect else "✗"
        print(f"\n{mark} [{desc}]")
        print(f"    状态码 {resp.status_code}（期望 {expect}）")
        assert resp.status_code == expect, f"{desc} 期望 {expect}，实际 {resp.status_code}"

    # ---------- 1. 未登录访问个人页 ----------
    resp = client.get("/profile", follow_redirects=False)
    line("未登录访问 /profile", resp, 401)
    print("    页面提示包含「你还未登录」：", "你还未登录" in resp.text)
    assert "你还未登录" in resp.text

    # ---------- 2. 密码错误 ----------
    resp = client.post("/login", data={"username": "alice", "password": "wrong"}, follow_redirects=False)
    line("密码错误登录", resp, 401)
    assert "session" not in {c.split("=")[0] for c in resp.headers.get_list("set-cookie")}

    # ---------- 3. 登录成功 ----------
    resp = client.post("/login", data={"username": "alice", "password": "password123"},
                       follow_redirects=False)
    line("登录成功（302 跳转）", resp, 302)
    print(f"    跳转目标：{resp.headers.get('Location')}")
    set_cookie = resp.headers.get("set-cookie", "")
    print(f"    Set-Cookie：{set_cookie[:100]}…")
    assert "session=" in set_cookie
    assert "httponly" in set_cookie.lower(), "session Cookie 应当是 HttpOnly"
    assert "samesite=lax" in set_cookie.lower(), "应当设置 SameSite 防 CSRF"

    # ---------- 4. 已登录访问个人页 ----------
    resp = client.get("/profile", follow_redirects=False)
    line("已登录访问 /profile", resp, 200)
    assert "已登录：alice" in resp.text
    assert "admin" in resp.text, "alice 的角色应当是 admin"
    print("    ✓ 浏览器（TestClient）自动带上了 Cookie，服务端从 session 里认出了用户")

    # ---------- 5. show-cookie ----------
    resp = client.get("/show-cookie")
    line("查看原始 Cookie", resp, 200)
    assert "session=" in resp.text
    print("    ✓ 原始 Cookie 中能看到 session 的 Base64 内容（可读但已签名）")

    # ---------- 6. 篡改 session Cookie ----------
    print("\n[篡改 session Cookie 的后果]")
    tampered_client = TestClient(app)
    tampered_client.cookies.set("session", "eyJ1c2VybmFtZSI6ImhhY2tlciJ9.fake-signature")
    resp = tampered_client.get("/profile", follow_redirects=False)
    print(f"    伪造 session 访问 /profile：状态码 {resp.status_code}")
    print("    说明：SessionMiddleware 验签失败后会把它当作空 session（等价于未登录）")
    assert resp.status_code == 401, "伪造的 session 必须被拒绝"
    print("    ✓ 签名机制挡住了伪造会话")

    # ---------- 7. 退出登录 ----------
    resp = client.get("/logout", follow_redirects=False)
    line("退出登录", resp, 302)
    resp = client.get("/profile", follow_redirects=False)
    line("退出后访问 /profile", resp, 401)
    print("    ✓ Session 可随时失效 —— 这是它相对 JWT 的最大优势")

    # ---------- 8. 手工 Cookie ----------
    resp = client.get("/set-cookie")
    line("写入普通 Cookie", resp, 200)
    cookies_header = ", ".join(resp.headers.get_list("set-cookie"))
    print(f"    Set-Cookie：{cookies_header[:160]}…")
    assert "theme=dark" in cookies_header and "httponly" in cookies_header.lower()

    resp = client.get("/read-cookie")
    line("读取 Cookie", resp, 200)
    print(f"    响应：{resp.json()}")
    assert resp.json()["theme"] == "dark"

    # ---------- 9. 签名 Cookie ----------
    resp = client.get("/set-signed-cookie")
    line("写入签名 Cookie", resp, 200)
    token = resp.json()["token"]
    print(f"    token：{token[:70]}…（前半段是 Base64 的原文，后面是签名）")

    resp = client.get("/read-signed-cookie")
    line("读取签名 Cookie（签名有效）", resp, 200)
    print(f"    响应：{resp.json()}")
    assert resp.json()["status"] == "签名有效"
    assert resp.json()["数据"] == {"user": "alice", "level": 3}

    # 篡改签名 Cookie 的最后几个字符
    bad_client = TestClient(app)
    bad_token = token[:-3] + ("aaa" if not token.endswith("aaa") else "bbb")
    bad_client.cookies.set("manual_token", bad_token)
    resp = bad_client.get("/read-signed-cookie")
    line("读取被篡改的签名 Cookie", resp, 200)
    print(f"    响应：{resp.json()}")
    assert resp.json()["status"] == "签名无效"
    print("    ✓ itsdangerous 验签失败，服务端拒绝信任被篡改的内容")

    # ---------- 10. JWT vs Session 对比 ----------
    print("\n[如何选：JWT 还是 Session/Cookie]")
    rows = [
        ("状态存放", "客户端（令牌自带信息）", "服务端（或签名 Cookie）"),
        ("水平扩展", "天然支持，无需共享存储", "需要共享存储（Redis）或粘性会话"),
        ("主动撤销", "困难（要靠黑名单/短过期）", "容易（删掉会话即可）"),
        ("跨域使用", "方便（放 Authorization 头）", "受同源策略限制"),
        ("典型场景", "前后端分离、微服务、开放 API", "传统 Web、后台管理系统"),
    ]
    print(f"    {'维度':<10}{'JWT':<30}{'Session/Cookie'}")
    print("    " + "-" * 68)
    for a, b, c in rows:
        print(f"    {a:<10}{b:<30}{c}")

    print("\n" + "=" * 72)
    print("自检全部通过 ✓")
    print("=" * 72)
    print("知识点回顾：")
    print("1. HTTP 无状态 → 需要 Cookie/Session 或 JWT 来「记住」用户")
    print("2. Cookie 必设 HttpOnly（防 XSS 窃取）、SameSite（防 CSRF）、Secure（HTTPS）")
    print("3. Starlette 的 SessionMiddleware 是签名 Cookie 方案，内容可读不可改")
    print("4. 要「服务器只存 session id」就得配合 Redis 等服务端存储")
    print("5. itsdangerous 的 URLSafeSerializer 可以给任意 Cookie 加签名校验")
    print("6. 登录时轮换 session（clear 后重写）可防会话固定攻击")


def main() -> None:
    print()
    print("#" * 72)
    print("# FastAPI Session 与 Cookie（后端开发基础 · Web框架章节）")
    print("#" * 72)
    print(f"解释器：{sys.executable}")
    print(f"计划监听端口：{PORT}")

    if "--check" in sys.argv:
        print()
        print("提示：--check 模式，执行自检，不启动服务器、不阻塞。")
        run_self_check()
    else:
        print(f"启动 uvicorn：http://127.0.0.1:{PORT}/login")
        print("演示账号：alice / password123 ，bob / bob456")
        print("按 Ctrl+C 停止。")
        uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")


if __name__ == "__main__":
    main()

"""FastAPI 中间件：CORS、GZip 与自定义中间件
================================================================
对应课案章节：后端开发基础 → Web框架 → FastAPI（中间件）

**什么是中间件？** 中间件是夹在"请求"和"响应"之间的组件：
    请求 → 中间件1 → 中间件2 → 路由函数 → 中间件2 → 中间件1 → 响应
它可以在请求到达路由之前做点事（鉴权、限流、打日志、加追踪 ID），
也可以在响应返回给客户端之前做点事（加响应头、压缩、统一错误格式）。

**执行顺序（容易记反，务必理解）**：
    Starlette 用"洋葱模型"，**后添加的中间件在最外层**，所以：
        请求方向：最后添加的 → …… → 最先添加的 → 路由
        响应方向：路由 → 最先添加的 → …… → 最后添加的
    本节用 ORDER 列表把真实顺序记录下来并打印，眼见为实。

本节知识点：
    1. CORSMiddleware：浏览器跨域访问的通行证（同源策略、预检请求 OPTIONS）
    2. GZipMiddleware：响应体压缩，省带宽（minimum_size 是压缩阈值）
    3. @app.middleware("http")：最简洁的自定义中间件写法（基于 BaseHTTPMiddleware）
    4. 用类写中间件：dispatch(request, call_next) + 纯 ASGI 中间件
    5. 请求耗时统计（X-Process-Time）与请求追踪 ID（X-Request-Id）
    6. 访问日志：中间件是记录"所有请求"的最佳位置
    7. 中间件里必须写 try/finally / 不能吞掉异常，否则会破坏后面的处理链

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\09_中间件.py' --check
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\09_中间件.py'
"""

from __future__ import annotations

import pathlib
import sys
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

HERE = pathlib.Path(__file__).resolve().parent
BACK_END = HERE.parent
ROOT = HERE.parents[1]
DATA_DIR = BACK_END / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PORT = 8109

app = FastAPI(
    title="09 中间件",
    description="后端开发基础 · FastAPI 示例：CORS / GZip / 自定义中间件",
    version="1.0.0",
)

# 记录中间件的进出顺序，自检时打印出来
ORDER: list[str] = []
ACCESS_LOG: list[str] = []


# ============================================================
# 一、先添加的中间件在最内层
# ============================================================
# ① GZipMiddleware：把响应体 gzip 压缩。
#    minimum_size=500：小于 500 字节的响应不压缩
#    （压缩小响应反而更费 CPU，而且 gzip 有固定头部开销）。
#    生产环境通常在 Nginx 层做压缩，应用层就不再重复做。
app.add_middleware(GZipMiddleware, minimum_size=500)

# ② CORSMiddleware：解决浏览器跨域。
#    **什么是跨域**：浏览器的同源策略要求 协议+域名+端口 三者完全相同，
#    前端 http://localhost:3000 调后端 http://localhost:8000，端口不同 → 跨域被拦。
#    **为什么需要它**：CORS 中间件在响应里加上 Access-Control-Allow-* 响应头，
#    告诉浏览器"这个来源我允许访问"，浏览器才放行。
#    注意：CORS 是**浏览器**的限制，用 requests/curl 直接调不会遇到跨域问题。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # 生产环境不要写 ["*"] 再用凭证
    allow_credentials=True,     # 允许携带 Cookie（此时 allow_origins 不能是 "*"）
    allow_methods=["*"],        # 允许的 HTTP 方法
    allow_headers=["*"],        # 允许的请求头
    expose_headers=["X-Process-Time", "X-Request-Id"],   # 允许前端 JS 读取的自定义响应头
    max_age=600,                # 预检结果缓存 10 分钟，减少 OPTIONS 请求
)


# ============================================================
# 二、用装饰器写自定义中间件（最常用的写法）
# ============================================================
@app.middleware("http")
async def add_process_time_and_request_id(request: Request, call_next):
    """自定义中间件：统计耗时 + 注入追踪 ID + 记录访问日志。

    请求经过这里时：
        1) call_next 之前的代码 = "请求进入"阶段；
        2) await call_next(request) = 把请求交给内层（下一个中间件或路由）；
        3) call_next 之后的代码 = "响应返回"阶段。
    注意 call_next 必须被 await，且中间件里不要吞掉异常。
    """
    request_id = request.headers.get("X-Request-Id") or uuid.uuid4().hex[:12]
    ORDER.append(f"① 自定义中间件 - 请求进入（request_id={request_id}）")

    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # 中间件里出现异常时也要把耗时记下来，否则排障时看不到"卡在哪一步"
        cost = (time.perf_counter() - start) * 1000
        ORDER.append(f"① 自定义中间件 - 请求异常（{cost:.2f}ms）")
        raise

    cost = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time"] = f"{cost:.3f}ms"
    response.headers["X-Request-Id"] = request_id
    ORDER.append(f"① 自定义中间件 - 响应返回（{cost:.2f}ms）")
    ACCESS_LOG.append(f"{request.method} {request.url.path} -> {response.status_code} ({cost:.2f}ms)")
    return response


# ============================================================
# 三、用类写中间件（需要跨多个路由复用/带构造参数时更合适）
# ============================================================
class ServerHeaderMiddleware(BaseHTTPMiddleware):
    """给所有响应加上 Server 标识头的中间件。

    类写法的好处：
        · 可以带构造参数（__init__ 里存下来）；
        · 可以打包成独立模块，在多个应用间复用；
        · 逻辑更长时结构更清晰。
    """

    def __init__(self, app, server_name: str = "fastapi-demo/1.0") -> None:
        super().__init__(app)
        self.server_name = server_name

    async def dispatch(self, request: Request, call_next):
        ORDER.append("② 类中间件 - 请求进入")
        response = await call_next(request)
        response.headers["X-Server-Name"] = self.server_name
        ORDER.append("② 类中间件 - 响应返回")
        return response


# 注意：这次是"最后添加"，所以它在最外层 —— 自检结果里会看到它的进出在最外围
app.add_middleware(ServerHeaderMiddleware, server_name="Back_End-09/1.0")


# ============================================================
# 四、路由
# ============================================================
@app.get("/", summary="首页", tags=["演示"])
async def index() -> dict:
    ORDER.append("③ 路由函数执行")
    return {"message": "Hello", "说明": "看看 X-Process-Time 与 X-Request-Id 响应头"}


@app.get("/big", summary="大响应（触发 GZip 压缩）", tags=["演示"])
async def big_response() -> dict:
    """返回一个较大的响应体，用来观察 GZipMiddleware 的效果。

    压缩阈值 minimum_size=500 字节，这里返回 5KB 左右的内容。
    """
    items = [{"id": i, "name": f"商品{i}", "description": "这是一段用于测试 gzip 压缩效果的重复文本，" * 3}
             for i in range(1, 61)]
    return {"count": len(items), "items": items}


@app.get("/small", summary="小响应（不触发压缩）", tags=["演示"])
async def small_response() -> dict:
    return {"message": "太小了，不会被压缩"}


@app.get("/boom", summary="抛异常（观察中间件是否仍记录耗时）", tags=["演示"])
async def boom() -> dict:
    raise RuntimeError("故意抛出的异常，用于演示中间件在异常路径下的行为")


# 让异常变成 500 响应而不是让测试客户端直接抛出
@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    from fastapi.responses import JSONResponse
    ORDER.append("④ 异常处理器")
    return JSONResponse(status_code=500, content={"error": "服务器内部错误", "detail": str(exc)})


# ============================================================
# 自检
# ============================================================
def run_self_check() -> None:
    from fastapi.testclient import TestClient

    print("=" * 72)
    print("FastAPI 中间件 · 自检")
    print("=" * 72)
    client = TestClient(app)

    # ---------- 1. 基础响应与自定义响应头 ----------
    ORDER.clear()
    resp = client.get("/")
    print("\n[1] GET /  自定义中间件注入的响应头")
    print(f"    状态码 {resp.status_code}")
    print(f"    X-Process-Time = {resp.headers.get('X-Process-Time')}")
    print(f"    X-Request-Id   = {resp.headers.get('X-Request-Id')}")
    print(f"    X-Server-Name  = {resp.headers.get('X-Server-Name')}（类中间件加的）")
    assert resp.status_code == 200
    assert resp.headers.get("X-Process-Time", "").endswith("ms")
    assert len(resp.headers.get("X-Request-Id", "")) == 12
    assert resp.headers.get("X-Server-Name") == "Back_End-09/1.0"

    # 客户端自带 X-Request-Id 时会透传（链路追踪的关键：上游生成的 ID 要一路带下去）
    resp2 = client.get("/", headers={"X-Request-Id": "trace-abc-001"})
    print(f"    透传上游 X-Request-Id：{resp2.headers.get('X-Request-Id')}")
    assert resp2.headers.get("X-Request-Id") == "trace-abc-001"

    # ---------- 2. 中间件执行顺序 ----------
    print("\n[2] 中间件执行顺序（洋葱模型）")
    ORDER.clear()
    client.get("/")          # 清空记录后重新发一次请求，只看这一次的顺序
    for line in ORDER:
        print(f"    {line}")
    assert ORDER[0] == "② 类中间件 - 请求进入", "最后 add 的中间件应当在最外层"
    assert ORDER[1].startswith("① 自定义中间件 - 请求进入")
    assert ORDER[2] == "③ 路由函数执行"
    assert ORDER[3].startswith("① 自定义中间件 - 响应返回")
    assert ORDER[4] == "② 类中间件 - 响应返回"
    print("    ✓ 顺序正确：类中间件（最后添加，最外层）→ 自定义中间件 → 路由 → 反向返回")

    # ---------- 3. GZip 压缩 ----------
    payload_size = len(str(client.get("/big").json()))
    print("\n[3] GZip 压缩（阈值 minimum_size=500 字节）")
    resp_big = client.get("/big", headers={"Accept-Encoding": "gzip"})
    resp_small = client.get("/small", headers={"Accept-Encoding": "gzip"})
    print(f"    /big   未压缩大小约 {payload_size} 字节，"
          f"Vary={resp_big.headers.get('Vary')}，Content-Encoding={resp_big.headers.get('Content-Encoding')}")
    print(f"    /small 响应体约 40 字节，Vary={resp_small.headers.get('Vary')}，"
          f"Content-Encoding={resp_small.headers.get('Content-Encoding')}（小响应不压缩）")
    print("    说明：GZipMiddleware 只在响应大于 minimum_size 时才压缩，")
    print("          并加上 Content-Encoding: gzip 与 Vary: Accept-Encoding 两个头；")
    print("          客户端（浏览器/httpx）会自动解压，所以业务代码拿到的是原始内容。")
    assert resp_big.status_code == 200
    assert "accept-encoding" in resp_big.headers.get("Vary", "").lower()
    assert resp_big.headers.get("Content-Encoding") == "gzip", "大响应应当被 gzip 压缩"
    assert resp_small.headers.get("Content-Encoding") is None, "小响应不压缩"
    # 解压后的内容必须完整
    assert resp_big.json()["count"] == 60
    print(f"    ✓ 大响应被压缩（Content-Encoding=gzip），小响应不压缩，解压后内容完整")

    # ---------- 4. CORS ----------
    print("\n[4] CORS（跨域）")
    resp = client.get("/", headers={"Origin": "http://localhost:3000"})
    print(f"    带 Origin 的普通请求：")
    print(f"      Access-Control-Allow-Origin  = {resp.headers.get('Access-Control-Allow-Origin')}")
    print(f"      Access-Control-Allow-Credentials = {resp.headers.get('Access-Control-Allow-Credentials')}")
    assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"

    resp = client.get("/", headers={"Origin": "http://evil.example.com"})
    print(f"    未授权的 Origin：Access-Control-Allow-Origin = "
          f"{resp.headers.get('Access-Control-Allow-Origin')}（None 表示浏览器会拦截）")
    assert resp.headers.get("Access-Control-Allow-Origin") is None

    # 预检请求：浏览器在发复杂请求（PUT/DELETE/自定义头）前会先发 OPTIONS 探路
    resp = client.options("/", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "content-type,authorization",
    })
    print(f"    预检请求 OPTIONS /：状态码 {resp.status_code}（期望 200）")
    print(f"      Allow-Methods = {resp.headers.get('Access-Control-Allow-Methods')}")
    print(f"      Allow-Headers = {resp.headers.get('Access-Control-Allow-Headers')[:60]}…")
    print(f"      Max-Age       = {resp.headers.get('Access-Control-Max-Age')}")
    assert resp.status_code == 200
    assert resp.headers.get("Access-Control-Allow-Origin") == "http://localhost:3000"

    # ---------- 5. 异常路径下的中间件 ----------
    print("\n[5] 异常路径")
    ORDER.clear()
    resp = client.get("/boom")
    print(f"    GET /boom 状态码 {resp.status_code}（期望 500），响应 {resp.text}")
    assert resp.status_code == 500
    print(f"    中间件是否仍加上了耗时头：{resp.headers.get('X-Process-Time')}")
    assert resp.headers.get("X-Process-Time") is not None, "异常响应也应带耗时头（try/finally 的意义）"
    print("    顺序记录：")
    for line in ORDER:
        print(f"      {line}")

    # ---------- 6. 访问日志 ----------
    print("\n[6] 中间件记录的访问日志（生产上应替换为 loguru）")
    for line in ACCESS_LOG:
        print(f"    {line}")
    assert len(ACCESS_LOG) >= 5

    # ---------- 7. 生产建议 ----------
    print("\n[7] 生产环境中间件顺序建议（从外到内）")
    print("    1) ServerErrorMiddleware（框架内置，最外层，兜住所有异常）")
    print("    2) 可信主机 / HTTPS 重定向")
    print("    3) CORS（要在鉴权之前，保证 401/403 响应也带跨域头）")
    print("    4) 会话 / 认证")
    print("    5) GZip（最靠近路由，只压缩业务响应）")
    print("    6) 自定义日志 / 计时中间件")
    print("    —— 生产上压缩常交给 Nginx，减少应用 CPU 开销")

    print("\n" + "=" * 72)
    print("自检全部通过 ✓")
    print("=" * 72)
    print("知识点回顾：")
    print("1. 中间件 = 请求/响应的公共处理层，顺序是「洋葱模型」，后添加的在最外层")
    print("2. CORS 是浏览器的同源策略要求，服务端用响应头放行；curl/requests 不受影响")
    print("3. GZip 只压缩大于 minimum_size 的响应，生产上常由 Nginx 承担")
    print("4. 自定义中间件适合做耗时统计、请求 ID、访问日志、限流")
    print("5. 中间件里要用 try/finally 保证异常路径也能收尾")


def main() -> None:
    print()
    print("#" * 72)
    print("# FastAPI 中间件（后端开发基础 · Web框架章节）")
    print("#" * 72)
    print(f"解释器：{sys.executable}")
    print(f"计划监听端口：{PORT}")

    if "--check" in sys.argv:
        print()
        print("提示：--check 模式，执行自检，不启动服务器、不阻塞。")
        run_self_check()
    else:
        print(f"启动 uvicorn：http://127.0.0.1:{PORT}/docs")
        print("按 Ctrl+C 停止。")
        uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")


if __name__ == "__main__":
    main()

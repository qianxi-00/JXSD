"""FastAPI 错误处理：统一异常响应格式
================================================================
对应课案章节：后端开发基础 → Web框架 → FastAPI（错误处理）

**为什么要统一错误处理？**
    如果不处理，客户端会收到各种格式的错误：
        · FastAPI 默认：{"detail": "..."} 或 {"detail": [{...校验错误...}]}
        · 未捕获异常：纯文本 "Internal Server Error"（还可能带一堆 HTML）
        · 路由不存在：{"detail": "Not Found"}
    前端要为每种格式写一套解析逻辑，非常痛苦。
    统一之后：**所有错误都是同一个 JSON 结构**，前端只写一次拦截器。

统一格式约定：
    {
      "code": 400,               # 业务/HTTP 状态码
      "message": "参数错误",      # 给人看的提示
      "detail": {...},           # 机器可读的细节（字段、原因）
      "request_id": "..."        # 追踪 ID，用户报错时报这个号就能直接定位日志
    }

本节知识点：
    1. HTTPException：抛出带状态码的异常（400 / 401 / 403 / 404 / 409 ……）
    2. 自定义业务异常 + @app.exception_handler：把业务错误和 HTTP 解耦
    3. RequestValidationError：接管 422，把 Pydantic 的错误整理成人话
    4. StarletteHTTPException：接管框架自身的 HTTP 错误（含 404 路由不存在）
    5. 兜底 Exception 处理器：任何未预期异常都返回统一 500，绝不泄露堆栈
    6. 常见状态码语义表，以及"能不用 500 就不用 500"的原则

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\10_错误处理.py' --check
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\10_错误处理.py'
"""

from __future__ import annotations

import pathlib
import sys
import uuid

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

HERE = pathlib.Path(__file__).resolve().parent
BACK_END = HERE.parent
ROOT = HERE.parents[1]
DATA_DIR = BACK_END / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PORT = 8110

app = FastAPI(
    title="10 错误处理",
    description="后端开发基础 · FastAPI 示例：统一异常响应格式",
    version="1.0.0",
)


# ============================================================
# 一、统一响应构造
# ============================================================
def error_response(
    status_code: int,
    message: str,
    detail: object = None,
    request_id: str | None = None,
    code: int | None = None,
) -> JSONResponse:
    """所有错误出口都走这里，保证格式绝对一致。"""
    return JSONResponse(
        status_code=status_code,
        content={
            "code": code if code is not None else status_code,
            "message": message,
            "detail": detail,
            "request_id": request_id or uuid.uuid4().hex[:12],
        },
        # 顺带带上 CORS 与追踪头，方便前端与排障
        headers={"X-Error-Handled": "true"},
    )


# ============================================================
# 二、自定义业务异常
# ============================================================
class BusinessError(Exception):
    """业务异常。

    为什么要自定义异常，而不是到处 raise HTTPException？
        · 业务代码只关心"发生了什么业务错误"（库存不足 / 余额不够），
          不需要知道该映射成 400 还是 409 —— 这是"表现层"的职责；
        · 自定义异常可以携带结构化数据（字段名、当前值、建议操作）；
        · 迁移框架（换 Web 框架）时，业务代码一行都不用改。

    注意：HTTPException 是"HTTP 层"的概念，业务层用它属于层次泄漏。
    """

    def __init__(self, message: str, *, code: int = 400, detail: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.detail = detail or {}


# ============================================================
# 三、异常处理器（顺序无关，FastAPI 按异常类型精确匹配）
# ============================================================
@app.exception_handler(BusinessError)
async def business_error_handler(request: Request, exc: BusinessError) -> JSONResponse:
    """业务异常 → 统一格式。"""
    return error_response(
        status_code=exc.code,
        message=exc.message,
        detail={"业务细节": exc.detail, "路径": request.url.path},
        code=exc.code,
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """接管 422 校验错误，把它整理成前端真正好用的结构。

    默认的 422 长这样：
        {"detail":[{"type":"missing","loc":["body","price"],"msg":"Field required",
                    "input":{...},"url":"https://errors.pydantic.dev/..."}]}
    对前端来说 loc 的解析很麻烦，这里把每个字段的错误提成
        {"字段路径": "body.price", "原因": "Field required", "类型": "missing"}
    同时把英文 msg 配上常见的中文说明。
    """
    fields = []
    for item in exc.errors():
        loc = ".".join(str(x) for x in item.get("loc", []))
        fields.append({
            "字段路径": loc,
            "原因": item.get("msg"),
            "类型": item.get("type"),
            "中文提示": _explain(item.get("type", "")),
        })
    return error_response(
        # 422 的常量名在新旧 Starlette 里不一样
        # （HTTP_422_UNPROCESSABLE_ENTITY → HTTP_422_UNPROCESSABLE_CONTENT），
        # 直接用数字 422 可以避免版本差异带来的弃用警告
        status_code=422,
        message="请求参数校验未通过",
        detail={"字段错误": fields},
    )


def _explain(error_type: str) -> str:
    """把 Pydantic 的错误类型翻译成人话（只覆盖最常见的几种）。"""
    mapping = {
        "missing": "该字段必填，请求里没有提供",
        "int_parsing": "需要整数，但给的不是合法整数",
        "float_parsing": "需要数字，但给的不是合法数字",
        "string_too_short": "字符串太短",
        "string_too_long": "字符串太长",
        "greater_than": "数值太小（必须大于下限）",
        "greater_than_equal": "数值太小（必须大于等于下限）",
        "less_than_equal": "数值太大（必须小于等于上限）",
        "string_pattern_mismatch": "格式不符合要求（正则不匹配）",
        "extra_forbidden": "出现了不允许多余字段",
        "value_error": "自定义校验未通过",
        "literal_error": "取值不在允许的范围内",
        "json_invalid": "请求体不是合法 JSON",
    }
    return mapping.get(error_type, "参数不合法")


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """接管 HTTPException 与框架内部抛出的 HTTP 错误。

    包括：
        · 我们自己 raise HTTPException(404, "商品不存在")
        · 路由不存在时框架抛出的 404
        · 方法不允许时框架抛出的 405
    """
    message = exc.detail if isinstance(exc.detail, str) else "请求处理失败"
    return error_response(
        status_code=exc.status_code,
        message=message,
        detail={"方法": request.method, "路径": request.url.path, "状态码说明": _status_text(exc.status_code)},
        code=exc.status_code,
    )


def _status_text(code: int) -> str:
    """常见 HTTP 状态码的中文含义（课案表格的代码版）。"""
    return {
        400: "Bad Request —— 参数错误、格式不对",
        401: "Unauthorized —— 未登录、token 无效",
        403: "Forbidden —— 已登录但无权限",
        404: "Not Found —— 资源不存在",
        405: "Method Not Allowed —— 方法不被支持",
        409: "Conflict —— 与现有资源冲突（如重复）",
        422: "Unprocessable Entity —— 语法正确但语义校验失败",
        429: "Too Many Requests —— 触发限流",
        500: "Internal Server Error —— 服务器内部错误",
        503: "Service Unavailable —— 服务暂时不可用（如依赖故障）",
    }.get(code, "未知状态码")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """兜底处理器：任何未预期异常都返回统一 500。

    **绝不把堆栈返回给客户端**（会泄露代码结构、路径、SQL、密钥片段）。
    正确做法：
        1) 服务端用 logger.exception 记录完整堆栈（见 04_日志/03_loguru异常捕获.py）；
        2) 客户端只拿到一句"服务器内部错误" + request_id；
        3) 用户报障时提供 request_id，运维据此在日志里精确定位。
    """
    # 真实项目这里应写：logger.exception(f"[{request_id}] 未处理异常 path={request.url.path}")
    print(f"    [服务端日志] 未处理异常（{type(exc).__name__}: {exc}）"
          f"，路径 {request.method} {request.url.path} —— 只有这里能看到细节，客户端看不到")
    return error_response(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        message="服务器内部错误，请稍后重试或联系管理员并提供 request_id",
        detail={"异常类型": type(exc).__name__},     # 不返回 str(exc)，避免泄露细节
    )


# ============================================================
# 四、业务接口
# ============================================================
class OrderIn(BaseModel):
    """下单请求体：故意加严格校验，用来演示 422。"""
    product_id: int = Field(..., gt=0, description="商品 ID")
    quantity: int = Field(..., ge=1, le=99, description="数量 1~99")
    coupon: str | None = Field(default=None, pattern=r"^[A-Z0-9]{6}$", description="优惠券：6 位大写字母数字")


STOCK = {1: 5, 2: 0, 3: 100}          # 商品 ID → 库存


@app.post("/orders", summary="下单（演示各类错误）", tags=["业务"])
def create_order(order: OrderIn) -> dict:
    """正常流程 + 三种业务错误：不存在(404)、无库存(409)、业务规则(400)。"""
    if order.product_id not in STOCK:
        # 资源不存在 → 404
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"商品 {order.product_id} 不存在")
    if STOCK[order.product_id] == 0:
        # 与现有资源状态冲突 → 409（比 400 更精确）
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=f"商品 {order.product_id} 已售罄")
    if order.quantity > STOCK[order.product_id]:
        # 业务规则不满足 → 用自定义业务异常，由处理器映射成 400
        raise BusinessError(
            "库存不足",
            code=400,
            detail={"需要": order.quantity, "现有": STOCK[order.product_id]},
        )
    STOCK[order.product_id] -= order.quantity
    return {"message": "下单成功", "product_id": order.product_id,
            "quantity": order.quantity, "剩余库存": STOCK[order.product_id]}


@app.get("/items/{item_id}", summary="查商品（课案的 400/404 示例）", tags=["业务"])
def read_item(item_id: int) -> dict:
    """课案原例：负数 400、不存在 404。"""
    if item_id < 0:
        raise HTTPException(status_code=400, detail="参数错误：item_id 不能为负数")
    items = {1: {"name": "苹果", "price": 5}, 2: {"name": "香蕉", "price": 3}}
    if item_id not in items:
        raise HTTPException(status_code=404, detail="商品不存在")
    return items[item_id]


@app.get("/crash", summary="触发未捕获异常（演示兜底 500）", tags=["业务"])
def crash() -> dict:
    """故意制造一个"代码 bug"（除零），观察兜底处理器如何保护客户端。"""
    values = [1, 2, 3]
    return {"average": sum(values) / 0}          # ZeroDivisionError


# ============================================================
# 自检
# ============================================================
def run_self_check() -> None:
    from fastapi.testclient import TestClient

    print("=" * 72)
    print("FastAPI 错误处理 · 自检")
    print("=" * 72)

    # raise_server_exceptions=False：让 TestClient 像真实浏览器一样"接收"500 响应，
    # 而不是把异常直接抛回给测试代码。测兜底处理器时必须这么建。
    client = TestClient(app, raise_server_exceptions=False)

    def call(desc: str, method: str, path: str, expect: int, **kwargs) -> dict:
        resp = client.request(method, path, **kwargs)
        body = resp.text.replace("\n", " ")
        if len(body) > 210:
            body = body[:210] + "…"
        mark = "✓" if resp.status_code == expect else "✗"
        print(f"\n{mark} [{desc}]")
        print(f"    {method} {path}")
        print(f"    状态码 {resp.status_code}（期望 {expect}）")
        print(f"    响应 {body}")
        assert resp.status_code == expect, f"{desc} 期望 {expect}，实际 {resp.status_code}"
        if expect >= 400:
            # 所有错误响应都应当带统一标记头，证明它走的是我们的统一错误出口
            assert resp.headers.get("X-Error-Handled") == "true", "错误响应应当带统一标记头"
        return resp.json()

    # ---------- 正常 ----------
    data = call("正常下单", "POST", "/orders", 200,
                json={"product_id": 3, "quantity": 2})
    assert data["剩余库存"] == 98

    # ---------- 404 ----------
    data = call("商品不存在（404）", "POST", "/orders", 404,
                json={"product_id": 999, "quantity": 1})
    assert "不存在" in data["message"]

    # ---------- 409 ----------
    call("商品售罄（409 Conflict）", "POST", "/orders", 409,
         json={"product_id": 2, "quantity": 1})

    # ---------- 400（自定义业务异常） ----------
    data = call("库存不足（自定义 BusinessError → 400）", "POST", "/orders", 400,
                json={"product_id": 1, "quantity": 99})
    print(f"    业务细节：{data['detail']['业务细节']}")
    assert data["detail"]["业务细节"] == {"需要": 99, "现有": 5}

    # ---------- 422（参数校验） ----------
    data = call("缺少 quantity 字段（422）", "POST", "/orders", 422,
                json={"product_id": 1})
    print(f"    整理后的字段错误：{data['detail']['字段错误']}")
    assert data["detail"]["字段错误"][0]["中文提示"] == "该字段必填，请求里没有提供"

    data = call("quantity 超范围 + 优惠券格式错误（422）", "POST", "/orders", 422,
                json={"product_id": 1, "quantity": 1000, "coupon": "abc"})
    for item in data["detail"]["字段错误"]:
        print(f"      {item['字段路径']:<14}{item['中文提示']}")
    assert len(data["detail"]["字段错误"]) == 2

    call("请求体不是合法 JSON（422）", "POST", "/orders", 422,
         content=b"{not json", headers={"Content-Type": "application/json"})

    # ---------- 课案的 400/404 原例 ----------
    call("GET /items/1 正常", "GET", "/items/1", 200)
    data = call("GET /items/-1 参数错误（400）", "GET", "/items/-1", 400)
    assert "不能为负数" in data["message"]
    call("GET /items/999 商品不存在（404）", "GET", "/items/999", 404)

    # ---------- 405 / 404 路由 ----------
    call("方法不允许（405）", "DELETE", "/items/1", 405)
    call("路径不存在（404）", "GET", "/no-such-path", 404)

    # ---------- 500 兜底 ----------
    print("\n[500 兜底演示]（注意：客户端只能看到一句话，细节在服务端日志里）")
    data = call("触发未捕获异常（500）", "GET", "/crash", 500)
    assert data["message"].startswith("服务器内部错误")
    assert "ZeroDivisionError" not in data["message"], "异常信息不应出现在给客户端的 message 里"
    assert data["detail"]["异常类型"] == "ZeroDivisionError"
    print("    ✓ 客户端只拿到统一格式 + request_id；堆栈只留在服务端日志中")

    # ---------- 状态码语义表 ----------
    print("\n[常见 HTTP 状态码中文语义]")
    for code in (400, 401, 403, 404, 405, 409, 422, 429, 500, 503):
        print(f"    {code}: {_status_text(code)}")

    print("\n" + "=" * 72)
    print("自检全部通过 ✓")
    print("=" * 72)
    print("知识点回顾：")
    print("1. 统一错误格式 = 前端只写一次拦截器；request_id 是排障的钥匙")
    print("2. 业务层抛自定义异常，表现层用 exception_handler 映射成 HTTP 状态码")
    print("3. 用 RequestValidationError 处理器把 422 整理成前端友好的结构")
    print("4. 用 StarletteHTTPException 处理器接管框架自带的 404/405")
    print("5. 兜底 Exception 处理器绝不返回堆栈，细节只进服务端日志")
    print("6. 能明确用 400/404/409 就别用 500 —— 500 代表「我们出 bug 了」")


def main() -> None:
    print()
    print("#" * 72)
    print("# FastAPI 错误处理（后端开发基础 · Web框架章节）")
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

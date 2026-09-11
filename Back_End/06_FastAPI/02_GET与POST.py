"""FastAPI GET 与 POST：参数到底从哪里来
================================================================
对应课案章节：后端开发基础 → Web框架 → FastAPI（GET vs POST 请求 / 测试 FastAPI 接口）

一个 HTTP 请求能携带数据的地方只有五处，FastAPI 用"类型注解 + 声明"把它们区分开：
    路径参数    /items/42              → 函数参数名与路径里的 {name} 一致
    查询参数    /items?skip=1&limit=2  → 普通类型注解（str/int/float/bool）
    请求体      {"name":"x"}           → Pydantic 模型参数（BaseModel 子类）
    表单        name=x&age=1           → Form(...)（Content-Type: form-urlencoded）
    请求头/Cookie  Authorization / sid → Header(...) / Cookie(...)

本节知识点：
    1. GET 与 POST 的语义差别（幂等/安全 vs 创建/提交）
    2. 路径参数、查询参数、请求体的声明方式
    3. 请求体用 Pydantic 模型：自动校验 + 自动生成文档
    4. Form：表单提交与文件上传的基础（需要 python-multipart）
    5. Header / Cookie 参数的读取，以及 convert_underscores 的作用
    6. status_code=201 与 response_model：控制响应状态码与输出字段
    7. 用 requests / httpx 调用真实服务的写法（见文件末尾注释）

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\02_GET与POST.py' --check
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\02_GET与POST.py'
"""

from __future__ import annotations

import pathlib
import sys
from typing import Annotated, Literal

import uvicorn
from fastapi import Body, Cookie, FastAPI, Form, Header, Path, Query, status
from pydantic import BaseModel, Field

HERE = pathlib.Path(__file__).resolve().parent
BACK_END = HERE.parent
ROOT = HERE.parents[1]
DATA_DIR = BACK_END / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PORT = 8102

app = FastAPI(
    title="02 GET 与 POST",
    description="后端开发基础 · FastAPI 示例：参数来源与请求体",
    version="1.0.0",
)


# ============================================================
# 一、Pydantic 模型 = 请求体的"契约"
# ============================================================
class Item(BaseModel):
    """商品模型。

    FastAPI 看到这种"BaseModel 子类的函数参数"，就认为它是**请求体**，
    并自动完成：JSON 解析 → 类型校验 → 默认值填充 → 文档生成。
    """
    name: str = Field(..., min_length=1, max_length=50, description="商品名称")
    price: float = Field(..., gt=0, description="单价，必须大于 0")
    tax: float | None = Field(default=None, ge=0, description="税费，可选")
    tags: list[str] = Field(default_factory=list, description="标签列表")

    # 用 ConfigDict 可以在文档里给出完整示例（/docs 里"Try it out"会直接填好）
    model_config = {
        "json_schema_extra": {
            "examples": [{"name": "机械键盘", "price": 399.0, "tax": 51.87, "tags": ["外设", "办公"]}]
        }
    }


class ItemOut(BaseModel):
    """响应模型：只暴露该给客户端的字段。

    response_model 的价值：
        - 自动过滤多余字段（防止把数据库里的 password / internal_note 返给前端）；
        - 响应结构与文档严格一致，前端不用猜。
    """
    name: str
    price: float
    price_with_tax: float
    tags: list[str]


class LoginForm(BaseModel):
    """从表单转成的模型（演示 Form 的进阶用法）。"""
    username: str
    password: str


# ============================================================
# 二、GET：参数在 URL 里
# ============================================================
@app.get("/items/{item_id}", tags=["GET"], summary="路径参数（带约束）")
def get_item(
    # Path() 用于给路径参数加约束和文档说明
    item_id: Annotated[int, Path(ge=1, le=1_000_000, description="商品 ID，正整数")],
    # Query() 用于给查询参数加约束；alias 让 URL 里可以用 page-size 这种带横杠的名字
    page_size: Annotated[int, Query(alias="page-size", ge=1, le=100)] = 20,
    keyword: Annotated[str | None, Query(max_length=30)] = None,
) -> dict:
    """GET 的典型形态：**只读、幂等**（调用一百次结果一样），参数全在 URL 里。

    为什么 GET 适合查询：可以被浏览器缓存、可以收藏、可以写进日志/监控，
    而且不会因为"重试"而产生副作用（比如重复下单）。
    """
    return {"item_id": item_id, "page_size": page_size, "keyword": keyword}


@app.get("/items", tags=["GET"], summary="列表 + 布尔查询参数")
def list_items(
    skip: int = 0,
    limit: int = Query(10, ge=1, le=50),
    in_stock: bool = Query(False, description="是否只看有货的（true/false/1/0 都能识别）"),
    sort: Literal["price", "name"] = Query("name", description="排序字段，只允许 price 或 name"),
) -> dict:
    """布尔与字面量类型的查询参数。

    Literal["price","name"] 会生成枚举下拉框，传别的值直接 422 ——
    这比在函数里写 if sort not in ("price","name") 更早、更省事。
    """
    return {"skip": skip, "limit": limit, "in_stock": in_stock, "sort": sort}


# ============================================================
# 三、POST：参数在请求体里
# ============================================================
@app.post("/items", tags=["POST"], status_code=status.HTTP_201_CREATED, summary="创建商品（JSON 请求体）")
def create_item(item: Item) -> ItemOut:
    """POST + JSON 请求体。

    与 GET 的区别：
        - 数据在请求体里，不在 URL 里 → 不受 URL 长度限制（GET 约 2KB）
        - 不出现在浏览器历史、服务器访问日志里 → 相对"不那么显眼"
        - 不幂等：调用两次会创建两个资源

    status_code=201（Created）是创建成功的标准状态码；
    用 fastapi.status 常量比手写数字更不容易写错。
    """
    price_with_tax = round(item.price * (1 + (item.tax or 0) / 100), 2)
    return ItemOut(
        name=item.name,
        price=item.price,
        price_with_tax=price_with_tax,
        tags=item.tags,
    )


@app.put("/items/{item_id}", tags=["POST"], summary="路径参数 + 请求体同时使用")
def update_item(item_id: int, item: Item, q: str | None = None) -> dict:
    """路径参数、请求体、查询参数可以同时出现在一个接口里，FastAPI 会自动区分。"""
    return {"item_id": item_id, "q": q, "item": item}


@app.post("/items/multi-body", tags=["POST"], summary="多个请求体字段")
def create_with_extra(
    item: Item,
    # Body(embed=True) 让这个"单个标量"也放进请求体，形如 {"importance": 5}
    importance: Annotated[int, Body(embed=True, ge=1, le=5)] = 3,
) -> dict:
    """请求体不一定是"一个模型"。

    当接口有多个请求体参数时，FastAPI 会把它们合并成一个 JSON 对象：
        {"item": {...}, "importance": 5}
    单个标量必须加 Body(embed=True)，否则 FastAPI 会把它当成查询参数。
    """
    return {"item": item, "importance": importance}


# ============================================================
# 四、表单、请求头、Cookie
# ============================================================
@app.post("/login/form", tags=["表单"], summary="表单提交（application/x-www-form-urlencoded）")
def login_by_form(
    # Form(...) 表示从表单里取值；需要 python-multipart 支持
    username: Annotated[str, Form(min_length=3, description="用户名")],
    password: Annotated[str, Form(min_length=6, description="密码")],
    remember: Annotated[bool, Form()] = False,
) -> dict:
    """传统 HTML 表单提交的数据格式。

    与 JSON 请求体的区别只在 Content-Type：
        application/json                  → 用 Pydantic 模型接收
        application/x-www-form-urlencoded → 用 Form(...) 接收
    OAuth2 的密码模式（见 08_认证与授权.py）用的就是表单格式，这是协议规定的。
    """
    return {"username": username, "password_length": len(password), "remember": remember}


@app.post("/login/model", tags=["表单"], summary="把表单直接解析成 Pydantic 模型")
def login_by_model(data: Annotated[LoginForm, Form()]) -> dict:
    """用 `Annotated[模型, Form()]` 把整个表单映射成模型，校验规则写在模型里。

    这样表单和 JSON 两种入口可以复用同一个模型，减少重复定义。
    """
    return {"username": data.username, "ok": True}


@app.get("/whoami", tags=["请求头与Cookie"], summary="读取请求头与 Cookie")
def whoami(
    # convert_underscores=True（默认）：Python 参数 user_agent 对应请求头 User-Agent
    user_agent: Annotated[str | None, Header()] = None,
    # 自定义请求头可以用 alias 精确指定名字
    x_token: Annotated[str | None, Header(alias="X-Token")] = None,
    # Cookie 参数：浏览器自动带上，JS 无法读取 httponly 的 Cookie
    session_id: Annotated[str | None, Cookie()] = None,
) -> dict:
    """请求头与 Cookie 是"隐式参数"：不写在 URL 也不写在请求体里。

    - 请求头：常用于鉴权（Authorization）、内容协商（Accept）、链路追踪（X-Trace-Id）
    - Cookie：浏览器自动携带，是传统 Web 会话的基础（详见 11_Session与Cookie.py）
    """
    return {"User-Agent": user_agent, "X-Token": x_token, "session_id": session_id}


# ============================================================
# 自检
# ============================================================
def run_self_check() -> None:
    from fastapi.testclient import TestClient

    print("=" * 72)
    print("FastAPI GET 与 POST · 自检")
    print("=" * 72)
    client = TestClient(app)

    def call(desc: str, method: str, path: str, expect: int, **kwargs) -> dict:
        resp = client.request(method, path, **kwargs)
        body = resp.text.replace("\n", " ")
        if len(body) > 170:
            body = body[:170] + "…"
        mark = "✓" if resp.status_code == expect else "✗"
        print(f"\n{mark} [{desc}]")
        print(f"    {method} {path}")
        print(f"    状态码 {resp.status_code}（期望 {expect}）")
        print(f"    响应 {body}")
        assert resp.status_code == expect, f"{method} {path} 期望 {expect}，实际 {resp.status_code}"
        return resp.json() if body.startswith("{") else {}

    # ---------- GET ----------
    call("路径参数 + 别名查询参数", "GET", "/items/5?page-size=50", 200)
    call("路径参数越界（item_id=0）", "GET", "/items/0", 422)
    call("布尔与枚举查询参数", "GET", "/items?in_stock=true&sort=price", 200)
    call("枚举取值非法（sort=xxx）", "GET", "/items?sort=xxx", 422)

    # ---------- POST + JSON ----------
    body = call("创建商品（合法请求体）", "POST", "/items", 201, json={
        "name": "机械键盘", "price": 399.0, "tax": 13, "tags": ["外设"],
    })
    assert body["price_with_tax"] == 450.87, "含税价应当由 response_model 计算后返回"
    assert set(body.keys()) == {"name", "price", "price_with_tax", "tags"}, "response_model 会过滤多余字段"

    call("创建商品（price 为 0 → 422）", "POST", "/items", 422, json={"name": "免费商品", "price": 0})
    call("创建商品（缺少 name → 422）", "POST", "/items", 422, json={"price": 10})

    call("路径参数 + 请求体 + 查询参数", "PUT", "/items/7?q=note", 200, json={"name": "显示器", "price": 999})

    call("多请求体字段（embed）", "POST", "/items/multi-body", 200,
         json={"item": {"name": "鼠标", "price": 99}, "importance": 4})

    # ---------- 表单 ----------
    call("表单提交", "POST", "/login/form", 200,
         data={"username": "alice", "password": "password123", "remember": "true"})
    call("表单校验失败（密码太短）", "POST", "/login/form", 422,
         data={"username": "alice", "password": "123"})
    call("表单映射到模型", "POST", "/login/model", 200,
         data={"username": "bob", "password": "secret123"})

    # ---------- 请求头与 Cookie ----------
    body = call("读取请求头与 Cookie", "GET", "/whoami", 200,
                headers={"User-Agent": "MyClient/1.0", "X-Token": "abc-123"},
                cookies={"session_id": "s-999"})
    assert body["User-Agent"] == "MyClient/1.0"
    assert body["X-Token"] == "abc-123"
    assert body["session_id"] == "s-999"
    print("    ✓ 请求头 User-Agent / X-Token 与 Cookie session_id 都正确注入到函数参数")

    print("\n" + "=" * 72)
    print("自检全部通过 ✓")
    print("=" * 72)
    print("知识点回顾：")
    print("1. 参数位置由声明决定：路径 / 查询 / 请求体 / 表单 / 请求头 / Cookie")
    print("2. BaseModel 参数 = 请求体；Form() = 表单；Header()/Cookie() = 隐式参数")
    print("3. status_code=201 表达「已创建」，response_model 负责过滤输出字段")
    print("4. GET 只读且幂等，POST 用于创建与提交；语义错了将来很难改")
    print("5. 用 requests/httpx 调用真实服务的示例：")
    print("   import requests")
    print("   requests.get('http://127.0.0.1:8102/items/5', params={'page-size': 50})")
    print("   requests.post('http://127.0.0.1:8102/items', json={'name':'苹果','price':3.99})")
    print()


def main() -> None:
    print()
    print("#" * 72)
    print("# FastAPI GET 与 POST（后端开发基础 · Web框架章节）")
    print("#" * 72)
    print(f"解释器：{sys.executable}")
    print(f"计划监听端口：{PORT}")
    print()

    if "--check" in sys.argv:
        print("提示：--check 模式，执行自检，不启动服务器、不阻塞。")
        print()
        run_self_check()
    else:
        print(f"启动 uvicorn：http://127.0.0.1:{PORT}/docs")
        print("按 Ctrl+C 停止。")
        uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")


if __name__ == "__main__":
    main()

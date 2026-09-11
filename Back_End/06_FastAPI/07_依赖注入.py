"""FastAPI 依赖注入（Depends）完全指南
================================================================
对应课案章节：后端开发基础 → Web框架 → FastAPI（依赖注入）

**依赖注入解决什么问题？**（先想清楚这个，再记语法）
    很多接口都需要"同一批前置动作"：开数据库会话、取当前登录用户、校验权限、
    读取分页参数、记录调用日志……
    如果每个路由函数都自己 new 一遍，就会出现三个问题：
        1) 代码重复（每个函数都写同样的 5 行）；
        2) 难以统一修改（加个超时，要改 50 个地方）；
        3) 无法替换（测试时不想用真数据库，却没有"注入口"）。

    依赖注入的做法是：**声明"我需要什么"，把"怎么造出来"交给框架**。
        def route(db: Session = Depends(get_db), user: User = Depends(get_current_user))
    好处：解耦（路由不知道 Session 怎么造）、复用（一份依赖多处使用）、
          可测试（用 dependency_overrides 换成假的）。

本节知识点：
    1. 函数依赖：Depends(函数)
    2. 带 yield 的依赖：请求前 Setup、请求后 Teardown（自动清理资源）
    3. 类依赖：把"参数集合"封装成类，__init__ 的参数会被自动解析
    4. 子依赖：依赖里再用 Depends，形成依赖树
    5. 缓存：同一个请求内，同一个依赖只执行一次（use_cache）；用 use_cache=False 关闭
    6. 路由组级依赖：APIRouter(dependencies=[...]) / FastAPI(dependencies=[...])
    7. 测试时替换依赖：app.dependency_overrides
    8. 依赖的返回值会被"注入"到参数名对应的参数上

运行方式：
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\07_依赖注入.py' --check
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\07_依赖注入.py'
"""

from __future__ import annotations

import pathlib
import sys
import time
from typing import Annotated

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Query

HERE = pathlib.Path(__file__).resolve().parent
BACK_END = HERE.parent
ROOT = HERE.parents[1]
DATA_DIR = BACK_END / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PORT = 8107

app = FastAPI(
    title="07 依赖注入",
    description="后端开发基础 · FastAPI 示例：Depends 的七种用法",
    version="1.0.0",
)

# 用一个列表记录依赖的 Setup/Teardown 顺序，自检时打印出来
LIFECYCLE: list[str] = []


# ============================================================
# 一、最简单的依赖：函数
# ============================================================
def common_params(
    skip: int = Query(0, ge=0, description="跳过条数"),
    limit: int = Query(10, ge=1, le=100, description="每页条数"),
) -> dict:
    """公共分页参数依赖。

    依赖函数自己也可以声明参数（查询参数、路径参数、请求头……），
    FastAPI 会把它们和路由参数一起解析 —— 这正是"复用参数声明"的价值。
    """
    return {"skip": skip, "limit": limit}


@app.get("/items", summary="使用函数依赖", tags=["基础"])
def list_items(pager: Annotated[dict, Depends(common_params)]) -> dict:
    """`pager: Annotated[dict, Depends(common_params)]` 表示：
        调用 common_params(...) → 把返回值注入 pager。
    用 Annotated 是 FastAPI 现在推荐的写法（比 `= Depends(...)` 更清晰）。
    """
    return {"pager": pager, "items": [f"item{i}" for i in range(pager["skip"], pager["skip"] + pager["limit"])]}


# ============================================================
# 二、带 yield 的依赖：Setup / Teardown
# ============================================================
class FakeConnection:
    """模拟一个需要"打开 / 关闭"的资源（数据库连接、文件、Redis 客户端……）。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.opened = True

    def query(self, sql: str) -> str:
        return f"[{self.name}] 执行 {sql}"


def get_connection():
    """带 yield 的依赖：yield 之前是 Setup，之后是 Teardown。

    执行时机：
        1) 请求进来 → 执行到 yield，把连接交给路由函数；
        2) 路由函数执行完毕（或抛异常）；
        3) 回到 yield 之后，执行清理代码。

    **这就是课案强调的"必须用 yield 而不是 return"**：
        return 会让资源永远不被释放，连接池很快被耗尽。
    """
    LIFECYCLE.append("① 打开连接（Setup）")
    conn = FakeConnection("主库连接")
    try:
        yield conn
    finally:
        # 用 finally 保证"即使路由抛异常也一定关闭"
        conn.opened = False
        LIFECYCLE.append("③ 关闭连接（Teardown）")


@app.get("/conn", summary="带 yield 的依赖（资源自动清理）", tags=["yield 依赖"])
def use_connection(conn: Annotated[FakeConnection, Depends(get_connection)]) -> dict:
    LIFECYCLE.append("② 路由函数执行中")
    return {"result": conn.query("SELECT 1"), "连接是否打开": conn.opened}


# ============================================================
# 三、类依赖：把一组参数封装起来
# ============================================================
class Pagination:
    """类也可以作为依赖。

    FastAPI 会分析 __init__ 的签名，把 skip/limit 当作查询参数解析，
    然后把实例注入进来。适合"一组总是一起出现的参数"。
    """

    def __init__(
        self,
        skip: int = Query(0, ge=0, description="跳过条数"),
        limit: int = Query(20, ge=1, le=200, description="每页条数"),
        order_by: str = Query("id", description="排序字段"),
    ) -> None:
        self.skip = skip
        self.limit = limit
        self.order_by = order_by

    def describe(self) -> str:
        return f"skip={self.skip}, limit={self.limit}, order_by={self.order_by}"


@app.get("/products", summary="类依赖", tags=["基础"])
def list_products(page: Annotated[Pagination, Depends(Pagination)]) -> dict:
    return {"pagination": page.describe(), "示例": "类依赖让参数声明集中在一处"}


# ============================================================
# 四、子依赖：依赖树
# ============================================================
def get_token(authorization: Annotated[str | None, Header()] = None) -> str:
    """第一层依赖：从请求头里取 token。"""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="缺少 Bearer Token")
    return authorization.split(" ", 1)[1]


def get_user(token: Annotated[str, Depends(get_token)]) -> dict:
    """第二层依赖：依赖 get_token，解析出用户。

    依赖之间可以嵌套，形成一棵"依赖树"。FastAPI 保证：
        · 同一个请求内，同一个依赖默认只执行一次（见下面的缓存小节）；
        · 任何一层抛 HTTPException，路由函数就不会被执行。
    """
    users = {"token-alice": {"name": "alice", "role": "user"},
             "token-root": {"name": "root", "role": "admin"}}
    user = users.get(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Token 无效")
    return user


def require_admin(user: Annotated[dict, Depends(get_user)]) -> dict:
    """第三层依赖：在"已登录"的基础上再做权限判断（403 而不是 401）。

    401 = 你还没证明你是谁（未认证）；403 = 知道你是谁，但你没权限（未授权）。
    """
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user


@app.get("/me", summary="子依赖：取当前用户", tags=["依赖树"])
def read_me(user: Annotated[dict, Depends(get_user)]) -> dict:
    return {"user": user, "说明": "已通过 get_token → get_user 两层依赖"}


@app.get("/admin", summary="子依赖：管理员专属", tags=["依赖树"])
def admin_only(user: Annotated[dict, Depends(require_admin)]) -> dict:
    return {"admin": user["name"], "说明": "已通过 get_token → get_user → require_admin 三层依赖"}


# ============================================================
# 五、缓存：同一请求内同一依赖只执行一次
# ============================================================
CALL_COUNTER = {"cached": 0, "uncached": 0}


def cached_dep() -> str:
    """默认 use_cache=True：同一个请求里多次声明也只执行一次。"""
    CALL_COUNTER["cached"] += 1
    return f"cached-{CALL_COUNTER['cached']}"


def uncached_dep() -> str:
    """use_cache=False：每次声明都重新执行（适合"必须每次重新计算"的场景）。"""
    CALL_COUNTER["uncached"] += 1
    return f"uncached-{CALL_COUNTER['uncached']}"


@app.get("/cache-demo", summary="依赖缓存对比", tags=["缓存与覆盖"])
def cache_demo(
    a: Annotated[str, Depends(cached_dep)],
    b: Annotated[str, Depends(cached_dep)],            # 与 a 是同一个值
    c: Annotated[str, Depends(uncached_dep)],
    d: Annotated[str, Depends(uncached_dep, use_cache=False)],   # 显式声明不缓存
) -> dict:
    """对比两条依赖的缓存行为：

        cached_dep  出现两次 → 只执行 1 次，a 与 b 完全相同
        uncached_dep 出现两次 → 执行 2 次，c 与 d 不同

    "缓存"指的是**同一次请求内**，跨请求当然会重新执行。
    """
    return {"a": a, "b": b, "c": c, "d": d,
            "a与b相同": a == b, "c与d相同": c == d,
            "说明": "默认 use_cache=True；要每次重算就传 use_cache=False"}


# ============================================================
# 六、路由组级依赖：整个路由组统一鉴权
# ============================================================
# dependencies=[Depends(...)]：只做校验，不注入返回值。
# 适合"这一组接口都需要登录/都需要记审计日志"的场景。
audit_log: list[str] = []


def audit(action: str = "访问") -> None:
    audit_log.append(f"{time.strftime('%H:%M:%S')} {action}")


admin_router = APIRouter(
    prefix="/api/admin",
    tags=["路由组依赖"],
    dependencies=[Depends(require_admin), Depends(audit)],
)


@admin_router.get("/users")
def admin_users() -> dict:
    """这个路由不用写任何鉴权代码 —— 整组的依赖已经在 APIRouter 上声明了。"""
    return {"users": ["alice", "bob"], "说明": "鉴权由 APIRouter(dependencies=[...]) 统一完成"}


@admin_router.get("/stats")
def admin_stats() -> dict:
    return {"orders": 128, "gmv": 45678.9, "说明": "同组第二个接口，自动享受同样的鉴权"}


app.include_router(admin_router)


# ============================================================
# 七、依赖覆盖：测试时替换真实依赖
# ============================================================
def real_dsn() -> str:
    """生产环境真实使用的数据库连接串依赖。"""
    return "mysql+pymysql://real-db:3306/prod"


@app.get("/need-db", summary="依赖可被测试覆盖", tags=["缓存与覆盖"])
def need_db(dsn: Annotated[str, Depends(real_dsn)]) -> dict:
    """这个依赖返回"生产数据库地址"。

    测试时用 app.dependency_overrides[real_dsn] = 假函数 就能整体替换，
    从而在不碰真库的情况下测完整流程 —— 这是依赖注入在工程上最大的收益之一。
    """
    return {"dsn": dsn, "说明": "返回值来自 real_dsn 依赖，可被测试覆盖"}


def run_self_check() -> None:
    from fastapi.testclient import TestClient

    print("=" * 72)
    print("FastAPI 依赖注入 · 自检")
    print("=" * 72)
    client = TestClient(app)

    def call(desc: str, path: str, expect: int, headers: dict | None = None, **params) -> dict:
        # params 为空时传 None，避免 httpx 把 URL 里已有的查询串清掉
        resp = client.get(path, headers=headers or {}, params=params or None)
        body = resp.text.replace("\n", " ")
        if len(body) > 180:
            body = body[:180] + "…"
        mark = "✓" if resp.status_code == expect else "✗"
        print(f"\n{mark} [{desc}]")
        print(f"    GET {path}")
        print(f"    状态码 {resp.status_code}（期望 {expect}）")
        print(f"    响应 {body}")
        assert resp.status_code == expect, f"{desc} 期望 {expect}，实际 {resp.status_code}"
        return resp.json()

    # ---------- 1. 函数依赖 ----------
    data = call("函数依赖（分页参数复用）", "/items", 200, skip=5, limit=3)
    assert data["pager"] == {"skip": 5, "limit": 3}
    call("函数依赖参数非法", "/items", 422, limit=999)

    # ---------- 2. yield 依赖的 Setup/Teardown 顺序 ----------
    LIFECYCLE.clear()
    data = call("带 yield 的依赖", "/conn", 200)
    assert data["连接是否打开"] is True, "路由执行期间连接应当是打开的"
    print("\n[生命周期顺序]")
    for line in LIFECYCLE:
        print(f"    {line}")
    assert LIFECYCLE == ["① 打开连接（Setup）", "② 路由函数执行中", "③ 关闭连接（Teardown）"], \
        "顺序必须是 Setup → 路由 → Teardown"
    print("    ✓ 顺序正确：Setup → 路由 → Teardown，资源被自动释放")

    # 再请求一次，验证连接被重新创建（而不是复用一个已关闭的）
    call("再次请求（新连接）", "/conn", 200)
    print(f"    本轮生命周期共记录 {len(LIFECYCLE)} 步（两轮 × 3 步 = 6）")
    assert len(LIFECYCLE) == 6

    # ---------- 3. 类依赖 ----------
    data = call("类依赖（参数集合封装）", "/products", 200, skip=2, limit=5, order_by="price")
    assert data["pagination"] == "skip=2, limit=5, order_by=price"

    # ---------- 4. 依赖树 ----------
    call("无 Token（401）", "/me", 401)
    call("Token 无效（401）", "/me", 401, headers={"Authorization": "Bearer wrong-token"})
    data = call("普通用户（200）", "/me", 200, headers={"Authorization": "Bearer token-alice"})
    assert data["user"]["name"] == "alice"

    call("普通用户访问管理员接口（403）", "/admin", 403,
         headers={"Authorization": "Bearer token-alice"})
    data = call("管理员访问（200）", "/admin", 200, headers={"Authorization": "Bearer token-root"})
    assert data["admin"] == "root"
    print("    ✓ 401（未认证）与 403（已认证但无权限）语义区分正确")

    # ---------- 5. 依赖缓存 ----------
    CALL_COUNTER["cached"] = 0
    CALL_COUNTER["uncached"] = 0
    data = call("依赖缓存对比", "/cache-demo", 200)
    assert data["a"] == data["b"], "use_cache=True 时同一个依赖只执行一次"
    assert data["c"] != data["d"], "use_cache=False 时每次都会重新执行"
    assert CALL_COUNTER["cached"] == 1, f"cached_dep 应只执行 1 次，实际 {CALL_COUNTER['cached']}"
    assert CALL_COUNTER["uncached"] == 2, f"uncached_dep 应执行 2 次，实际 {CALL_COUNTER['uncached']}"
    print(f"    ✓ cached_dep 执行 {CALL_COUNTER['cached']} 次，uncached_dep 执行 {CALL_COUNTER['uncached']} 次")

    # ---------- 6. 路由组依赖 ----------
    audit_log.clear()
    call("路由组依赖：无 Token 访问整组接口（401）", "/api/admin/users", 401)
    call("路由组依赖：普通用户（403）", "/api/admin/stats", 403,
         headers={"Authorization": "Bearer token-alice"})
    data = call("路由组依赖：管理员（200）", "/api/admin/users", 200,
                headers={"Authorization": "Bearer token-root"})
    assert "alice" in data["users"]
    print(f"\n[审计依赖记录] audit_log = {audit_log}")
    assert len(audit_log) == 1, "只有成功的请求会走到 audit 依赖（前两次被鉴权拦下了）"
    print("    ✓ 整组接口自动带上鉴权与审计，路由函数里一行相关代码都没有")

    # ---------- 7. 依赖覆盖 ----------
    data = call("真实依赖（未覆盖）", "/need-db", 200)
    assert data["dsn"].startswith("mysql"), "未覆盖时应当返回生产连接串"

    print("\n[依赖覆盖演示：把生产库换成测试库]")
    app.dependency_overrides[real_dsn] = lambda: "sqlite:///:memory:"
    print("    已执行 app.dependency_overrides[real_dsn] = lambda: 'sqlite:///:memory:'")
    data = call("覆盖依赖后", "/need-db", 200)
    assert data["dsn"] == "sqlite:///:memory:", "覆盖后应当返回假连接串"
    print("    ✓ 路由函数一行都没改，返回的却是测试库地址")

    app.dependency_overrides.clear()
    print("    已执行 app.dependency_overrides.clear()")
    data = call("清理覆盖后（恢复真实依赖）", "/need-db", 200)
    assert data["dsn"].startswith("mysql")
    print("    ✓ 覆盖已清理，恢复真实依赖")

    print("\n" + "=" * 72)
    print("自检全部通过 ✓")
    print("=" * 72)
    print("知识点回顾：")
    print("1. Depends 解决的是「重复创建前置资源」的问题，核心收益是解耦 + 复用 + 可测试")
    print("2. yield 依赖负责资源生命周期：Setup 在前、Teardown 在后，用 finally 保证释放")
    print("3. 类依赖适合封装一组参数；子依赖可以搭出依赖树")
    print("4. 同一请求内默认缓存；use_cache=False 可关闭")
    print("5. APIRouter(dependencies=[...]) 做路由组统一鉴权/审计")
    print("6. dependency_overrides 是写测试的利器，能彻底替换掉真数据库")
    print()


def main() -> None:
    print()
    print("#" * 72)
    print("# FastAPI 依赖注入（后端开发基础 · Web框架章节）")
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

"""FastAPI 最小应用：第一个接口与自动文档
================================================================
对应课案章节：后端开发基础 → Web框架 → FastAPI（什么是 FastAPI / 安装与运行）

本节知识点：
    1. FastAPI 是什么：现代、高性能的 Web 框架，核心是"类型提示驱动的开发"
    2. 三大支柱：Starlette（ASGI 网络层）+ Pydantic（数据校验）+ 类型提示（编辑器友好）
    3. app = FastAPI()：应用对象；@app.get(path) 注册路由
    4. 路径参数 item_id: int —— 类型注解就是校验规则，写错自动返回 422
    5. 自动交互文档：/docs（Swagger UI）、/redoc（ReDoc）、/openapi.json（机器可读）
    6. 启动方式：uvicorn.run(app, host, port) 或命令行 uvicorn 模块:app --reload
    7. 为什么 import 这个模块没有副作用：app 只是被构造出来，不监听端口，
       这样才能被 verify_api.py / 测试框架安全地导入

运行方式（PowerShell，必须用本仓库的虚拟环境解释器）：
    # 快速自检（不启动服务器、不阻塞，verify_all.py 用的就是这个）
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\01_最小应用.py' --check

    # 真正启动服务（会阻塞，Ctrl+C 停止），然后浏览器打开 http://127.0.0.1:8101/docs
    & 'F:\\ProGram\\Python_Base\\.venv\\Scripts\\python.exe' 'F:\\ProGram\\Python_Base\\Back_End\\06_FastAPI\\01_最小应用.py'
"""

from __future__ import annotations

import pathlib
import sys

import uvicorn
from fastapi import FastAPI, Query

# ------------------------------------------------------------
# 路径：本文件位于 Back_End/06_FastAPI/01_最小应用.py
#   parents[0] = 06_FastAPI，parents[1] = Back_End，parents[2] = Python_Base
# 所有输出统一写 Back_End/data/，与"当前工作目录"无关
# ------------------------------------------------------------
HERE = pathlib.Path(__file__).resolve().parent
BACK_END = HERE.parent
ROOT = HERE.parents[1]
DATA_DIR = BACK_END / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if str(ROOT) not in sys.path:      # 兜底：确保能 import 根目录 config.py
    sys.path.insert(0, str(ROOT))

# 本示例使用的端口，从 8101 开始逐个递增，避免互相冲突
PORT = 8101

# ============================================================
# 创建应用
# ============================================================
# FastAPI(title=..., description=..., version=...) 里的信息会直接显示在 /docs 顶部，
# 所以"写文档"这件事在 FastAPI 里就是写参数，不需要额外的文档工作。
app = FastAPI(
    title="01 最小应用",
    description="后端开发基础 · FastAPI 示例：最小应用与自动文档",
    version="1.0.0",
)


# ============================================================
# 路由
# ============================================================
@app.get("/", summary="根路径", tags=["基础"])
def read_root() -> dict:
    """最简单的接口。

    返回 dict 时 FastAPI 会自动：
        1) 用 jsonable_encoder 转成 JSON 可序列化结构；
        2) 设置 Content-Type: application/json；
        3) 状态码默认 200。
    你完全不用碰 Request/Response 对象 —— 这就是"框架替你做事"。
    """
    return {"message": "Hello World", "docs": "/docs"}


@app.get("/items/{item_id}", summary="路径参数自动类型转换", tags=["基础"])
def read_item(item_id: int, q: str | None = None) -> dict:
    """路径参数 + 查询参数。

    - `item_id: int`：类型注解就是校验规则。/items/abc 会直接被拦下返回 422，
      函数体根本不会执行（也就不会因为类型不对而炸掉）。
    - `q: str | None = None`：可选查询参数，等价于 Optional[str]。
    - 这两个参数在 /docs 里会自动生成输入框，可以直接点着试。
    """
    return {"item_id": item_id, "q": q}


@app.get("/items", summary="查询参数带校验", tags=["基础"])
def read_items(
    skip: int = Query(0, ge=0, description="跳过多少条"),
    limit: int = Query(10, ge=1, le=100, description="返回多少条，1~100"),
    keyword: str | None = Query(None, max_length=20, description="模糊搜索关键字"),
) -> dict:
    """查询参数用 Query() 声明校验规则。

    Query 常用参数（和 Pydantic Field 一致）：
        ge / gt / le / lt        数值范围
        min_length / max_length  字符串长度
        pattern                  正则
        description / title      文档展示
        alias                    参数别名（如 alias="page-size"）
    """
    data = [{"id": i, "name": f"商品{i}"} for i in range(1, 26)]
    if keyword:
        data = [item for item in data if keyword in item["name"]]
    return {"skip": skip, "limit": limit, "keyword": keyword, "total": len(data),
            "items": data[skip: skip + limit]}


@app.get("/health", summary="健康检查", tags=["运维"])
def health() -> dict:
    """K8S/负载均衡探针用的健康检查接口（对应 09_容器部署 里的 livenessProbe）。"""
    return {"status": "ok", "port": PORT}


# ============================================================
# 自检（--check 模式使用）：用 TestClient 走完整 ASGI 调用链
# ============================================================
def run_self_check() -> None:
    """不启动服务器，直接请求应用对象，验证接口行为。"""
    from fastapi.testclient import TestClient       # 延迟导入：只有自检时才需要

    print("=" * 72)
    print("FastAPI 最小应用 · 自检（TestClient 直接调用 ASGI 应用）")
    print("=" * 72)

    client = TestClient(app)

    cases = [
        ("GET", "/", None, 200, "根路径"),
        ("GET", "/items/42", None, 200, "路径参数（int）"),
        ("GET", "/items/abc", None, 422, "路径参数类型错误 → 422"),
        ("GET", "/items/42?q=搜索词", None, 200, "路径参数 + 查询参数"),
        ("GET", "/items?skip=5&limit=3", None, 200, "查询参数带校验"),
        ("GET", "/items?limit=1000", None, 422, "limit 超出上限 → 422"),
        ("GET", "/health", None, 200, "健康检查"),
        ("GET", "/openapi.json", None, 200, "OpenAPI 规范（自动生成）"),
        ("GET", "/docs", None, 200, "Swagger UI 文档页面"),
        ("GET", "/redoc", None, 200, "ReDoc 文档页面"),
    ]

    for method, path, payload, expect, desc in cases:
        resp = client.request(method, path, json=payload)
        body = resp.text.replace("\n", " ")
        if len(body) > 160:
            body = body[:160] + "…"
        mark = "✓" if resp.status_code == expect else "✗"
        print(f"\n{mark} [{desc}]")
        print(f"    {method} {path}")
        print(f"    状态码 {resp.status_code}（期望 {expect}）")
        print(f"    响应 {body}")
        assert resp.status_code == expect, f"{path} 期望 {expect}，实际 {resp.status_code}"

    # 校验 422 的响应体结构：FastAPI 会告诉你"哪个字段、错在哪里"
    resp = client.get("/items/abc")
    detail = resp.json()["detail"][0]
    print("\n[422 响应体结构分析]")
    print(f"    出错字段位置：{detail['loc']}")
    print(f"    错误类型：{detail['type']}")
    print(f"    错误说明：{detail['msg']}")
    assert detail["type"] == "int_parsing"

    # 校验 OpenAPI 文档里确实注册了我们的接口
    spec = client.get("/openapi.json").json()
    paths = sorted(spec["paths"].keys())
    print("\n[OpenAPI 自动生成的接口清单]")
    for path in paths:
        methods = ",".join(m.upper() for m in spec["paths"][path])
        print(f"    {methods:<8} {path}")
    assert "/items/{item_id}" in spec["paths"]

    print("\n" + "=" * 72)
    print("自检全部通过 ✓")
    print("=" * 72)
    print("知识点回顾：")
    print("1. 类型注解 = 校验规则 + 文档 + 编辑器补全，一份代码三份收益")
    print("2. 参数不合法返回 422，响应体里有 loc/type/msg，前端能精确提示")
    print("3. /docs、/redoc、/openapi.json 全部由类型注解自动生成，无需手写文档")
    print("4. 模块导入时不监听端口，所以可以被测试与校验脚本安全导入")
    print(f"5. 启动服务：python 本文件（默认端口 {PORT}）")
    print()


def main() -> None:
    print()
    print("#" * 72)
    print("# FastAPI 最小应用（后端开发基础 · Web框架章节）")
    print("#" * 72)
    print(f"解释器：{sys.executable}")
    print(f"应用标题：{app.title}")
    print(f"计划监听端口：{PORT}")
    print()

    if "--check" in sys.argv:
        print("提示：--check 模式，执行自检，不启动服务器、不阻塞。")
        print()
        run_self_check()
    else:
        print(f"启动 uvicorn：http://127.0.0.1:{PORT}")
        print(f"交互文档：http://127.0.0.1:{PORT}/docs")
        print("按 Ctrl+C 停止。开发时也可以直接用命令行：uvicorn 01_最小应用:app --reload")
        # 直接传 app 对象（而不是导入字符串），这样 reload 必须为 False
        uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")


if __name__ == "__main__":
    main()

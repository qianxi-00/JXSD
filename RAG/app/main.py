# --- 路径引导：确保能导入 Python_Base 根目录的 config 与 RAG 内部包 ---
# 启动方式通常是 `python RAG\app\main.py` 或 `uvicorn` 按文件路径加载本模块，
# 两种情况下 sys.path[0] 都不含仓库根（Python_Base，放着 config.py），
# 所以下面这段必须在**任何 `from config import ...` 之前**执行。
# 三段式：定位仓库根 → 把仓库根和 RAG 根都插进 sys.path。
# 同一段样板在 app\chat_ui.py、core\milvus_init.py 里各有一份（复制三处），改动要同步。
import sys as _sys
from pathlib import Path as _Path

# 从本文件位置逐级向上找名为 Python_Base 的祖先目录；`_BASE.parent != _BASE` 是到顶保护
# （Path 到达盘符根后 parent 等于自己），没有它就成了死循环。
_BASE = _Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
_sys.path.insert(0, str(_BASE))          # Python_Base 根（config.py）
_sys.path.insert(0, str(_BASE / "RAG"))  # RAG 根（core/llm/pipeline 等包）
# 两次 insert(0) 之后顺序是 [RAG 根, Python_Base 根, ...]，即项目自己的
# core / llm / pipeline 优先于同名的第三方包被解析到。
# 别名用 _sys / _Path，避免给模块命名空间留下易与业务变量重名的 sys / Path。
# 注意：下面这条字符串**不是**模块 docstring —— 它前面已有 import 语句，而 `__doc__`
# 只在「模块第一条语句就是字符串字面量」时才会赋值，所以 app.main.__doc__ 实际是 None
# （help()/pydoc 看不到这段说明）。保留原位置不动，仅补此说明。
"""应用主入口:FastAPI API 服务 + Chainlit 前端界面(端口可配置,默认 8099)"""

# 一个进程同时提供两种前端形态，这是本文件的核心设计：
#   - /api/chat、/api/chat/stream 给程序调用（JSON / SSE）；
#   - path="/" 挂的是 Chainlit 页面（见文件末尾的 mount_chainlit），给人用。
# 两者共用同一个进程、同一份配置与同一个 RAGPipeline 实现。
import json
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from chainlit.utils import mount_chainlit  # 把 Chainlit 的 ASGI 应用挂到本 FastAPI 应用上
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse  # SSE 靠它边算边推，不用等全部生成完
from pydantic import BaseModel  # 请求体校验：字段缺失/类型不对由 FastAPI 直接回 422

from config import settings
from core.logger import logger
from core.routes import ROUTE_LABELS
from core.timeout import call_with_timeout
from pipeline.modes import answer_events, answer as answer_by_mode, list_modes

# 注意此处**没有** CORSMiddleware：本服务只给同源的 Chainlit 页面与本地脚本用，
# 不做跨域开放。将来若有独立前端域，需要显式加中间件，否则浏览器端会被 CORS 拦下。
def _warmup() -> None:
    """启动预热：把 FAQ 预设向量与 BM25 索引先加载好（失败不影响服务可用）。

    为什么值得做（课案把"启动预热"列为生产做法）：不预热时，**第一个未命中缓存的请求**
    要现算 5 条标准问法的向量（几十毫秒到秒级）并加载 BM25 语料 —— 用户感知就是
    "服务刚起来时第一问特别慢"。预热把这份成本挪到启动阶段。

    三步各自的失败都**只记日志**：预热是优化，不是可用性前提。某个依赖没起来时，
    服务仍应能启动并在真正用到时报错（那样错误信息还带着请求上下文，更好排查）。
    """
    from core.cache import AnswerCache
    from retrieval import keyword_retrieval

    try:
        # `cache.warmup()` 而不是分别 seed：它会**把进程内的预设矩阵也载入**
        # （只播种 Redis 的话，第一个请求仍要现做归一化 —— 等于没预热）。
        info = AnswerCache().warmup()
        logger.info(
            f"[预热] FAQ 相似度层 {info['faq']} 条（本轮新播 {info['faq_seeded']}）/ "
            f"明细精确层新播 {info['details']} 条 / 预设矩阵已载入 {info['matrix']} 行"
        )
    except Exception as exc:  # noqa: BLE001 预热失败不能拦住启动
        logger.warning(f"[预热] 缓存层预热失败（不影响服务启动）: {exc}")
    try:
        # BM25 语料在 keyword_retrieval 里是 lru_cache 按函数对象缓存的：调一次就预热好了
        keyword_retrieval._load_index()
        logger.info("[预热] BM25 语料索引已加载")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[预热] BM25 预热失败（不影响服务启动）: {exc}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """应用生命周期：启动时预热（可用 APP_WARMUP=false 关掉，测试/CI 更快）。"""
    if settings.app.warmup:
        _warmup()
    yield


app = FastAPI(title="财务 RAG 智能问答系统", lifespan=lifespan)

# ── CORS：**只在显式配置了 APP_CORS_ORIGINS 时才挂** ──
# 本服务默认只给同源的 Chainlit 页面与本地脚本用；无脑放开跨域等于把接口暴露给任意网页
# （浏览器会带着 cookie 替别的站点发请求）。要接独立前端域时在 .env 里列白名单：
#   APP_CORS_ORIGINS=http://localhost:5173,https://rag.example.com
_cors_origins = [item.strip() for item in settings.app.cors_origins.split(",") if item.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info(f"[CORS] 已按白名单开放跨域: {_cors_origins}")

# 用绝对路径而不是相对路径：uvicorn/Chainlit 的启动工作目录可能是仓库根、RAG 目录或别处，
# 相对路径会在换启动方式时凭空找不到文件（mount_chainlit 内部会检查文件是否存在）。
# 解析到 app/chat_ui.py 后，由 mount_chainlit 用 importlib 加载它。
CHAT_UI_PATH = Path(__file__).resolve().parent / "chat_ui.py"

# 这里**不再**持有 RAGPipeline 单例：四条线路都由 `pipeline.modes` 统一分发，
# 基础线路的实例由 `modes._basic_pipeline()` 惰性创建并在进程内复用
# （见该函数说明）。原先这里再建一个的话，同一进程里会存在两份 RAGPipeline
# ——各自加载 preset 矩阵与 BM25 语料，且 /api/chat 与页面用的不是同一个实例。


class ChatRequest(BaseModel):
    """两个问答接口共用的请求体。

    `mode` 决定走哪条 RAG 线路（见 `pipeline/modes.py` 与 `GET /api/modes`）：
    不传时按基础线路处理（保持旧调用方的行为不变）。
    `history` 只有 Agentic 与融合线路会用（基础/GraphRAG 是单轮线路，
    传了也会被忽略——`core/routes.py::ROUTE_SUPPORTS_HISTORY` 是这一点的单一真相）。
    """

    question: str
    mode: str = "basic"
    history: list[dict] = []


@app.get("/api/modes")
async def modes() -> dict:
    """列出四条 RAG 线路：前端用它渲染选择器，脚本用它做横向对比。

    返回顺序即展示顺序（从简单到复杂），前端不要自己排序。
    """
    return {"modes": list_modes(), "default": "basic"}


def _health_probe(name: str, fn, timeout: float = 3.0) -> dict:
    """跑一个依赖探针，把结果收敛成 `{ok, detail}`（**绝不让探针把接口挂住**）。

    为什么必须有界：本机沙箱里 libpq 的 GSS 协商会永久卡住，`connect_timeout` 无效 ——
    健康检查如果直接 `psycopg.connect`，/api/health 自己就会挂死，
    那比返回 unhealthy 更糟（监控探针也会跟着超时，分不清"服务挂了"还是"检查卡了"）。
    """
    ok, payload = call_with_timeout(fn, timeout)
    if ok and payload is not None:
        return {"ok": True, "detail": str(payload)}
    if not ok and payload is None:
        return {"ok": False, "detail": f"超时（>{timeout}s）"}
    return {"ok": False, "detail": f"{type(payload).__name__}: {payload}"}


@app.get("/api/health")
def health() -> dict:
    """健康检查：四个依赖各探一次，**任一失败也只把整体标成 degraded**（不抛 500）。

    设计取舍：
    - 默认**不返回 5xx**：负载均衡/监控靠 body 里的 `status` 判定即可，而 5xx 会让
      "服务进程还活着、只是某个依赖没起"这种状态看起来像"服务死了"。
    - 每个探针都带 3 秒上限（见 `_health_probe`），整体最坏 ~12 秒。
    - 同步 `def`：探针都是阻塞调用，FastAPI 会丢线程池（与 /api/chat 同一取舍）。
    """

    def _redis():
        from core.redis_client import get_redis_client

        return get_redis_client().ping()

    def _milvus():
        from core.database import get_milvus_client

        client = get_milvus_client()
        names = client.list_collections()
        return f"collections={len(names)}"

    def _postgres():
        import sqlalchemy as sa

        from agentic.text_to_sql import get_engine

        with get_engine().connect() as conn:
            return f"tickets={conn.execute(sa.text('SELECT COUNT(*) FROM tickets')).scalar_one()}"

    def _neo4j():
        from graph_rag.models import connect

        connect()
        return "connected"

    checks = {
        "redis": _health_probe("redis", _redis),
        "milvus": _health_probe("milvus", _milvus),
        "postgres": _health_probe("postgres", _postgres),
        "neo4j": _health_probe("neo4j", _neo4j),
    }
    failed = [name for name, item in checks.items() if not item["ok"]]
    return {
        # ok=全通 / degraded=部分不可用（服务还能应答，但某些线路会失败）
        "status": "ok" if not failed else "degraded",
        "failed": failed,
        "checks": checks,
        "qa_cache_enabled": settings.qa_cache.enabled,
        "warmup": settings.app.warmup,
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    """问答接口:调用指定线路的 RAG 流水线生成回答(非流式)

    与 `pipeline.run_and_collect` 的区别：返回结构统一成四线路通用形状
    （mode / answer / sources / extra / cache_hit / elapsed_s / system_error），
    并额外带上 `mode_label` 供前端直接显示。

    **同步 `def`（不是 async）是刻意的**：四条线路的 runner 全是阻塞调用
    （embedding / 向量库 / 重排 / LLM / Neo4j），FastAPI 会把同步视图丢到线程池执行；
    写成 async 反而会在等检索时占住事件循环，把并发请求排成一队
    （`graph_rag/service.py` 里同样原因也用了同步视图）。
    """
    result = answer_by_mode(req.question, req.mode, req.history)
    result["mode_label"] = ROUTE_LABELS.get(result["mode"], result["mode"])
    return result


def _classify_query_type(event: dict, current: str) -> str:
    """把事件流翻译成课案约定的 `query_type` ∈ cache / faq / rag / rag_rejected / llm。

    课案（基础篇「接口职责」）要求 SSE 每条消息带上它 —— 前端据此判断"这条回答来自缓存、
    还是检索、还是直答"，不必自己解析内部事件。映射规则（单向、可测）：

        cache_hit(exact)   → cache
        cache_hit(preset)  → faq
        route(direct)      → llm（直答，没有检索）
        route(rag)         → rag
        no_evidence        → rag_rejected（检索了但全低于阈值 ⇒ 保守回复）

    用"见到哪个事件就更新状态"的**状态机**：事件顺序是 start → (cache_hit) → route →
    … → (no_evidence) → token，到 token 时状态已定型，不必每条 token 重算。
    """
    if event.get("type") == "cache_hit":
        return "cache" if event.get("mode") == "exact" else "faq"
    if event.get("type") == "route":
        return "llm" if (event.get("route") or "rag") == "direct" else "rag"
    if event.get("type") == "no_evidence":
        return "rag_rejected"
    return current


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """问答接口:SSE 流式返回回答内容（四条线路统一走事件流）

    载荷形状（**向后兼容**：老字段 `delta` 保留，新字段叠加）：

        data: {"delta": "...", "mode": "basic", "query_type": "cache|faq|rag|rag_rejected|llm"}
        ...
        data: {"done": true, "answer": "...", "sources": [...], "query_type": "...",
               "processing_time": 1.23, "cache_hit": null, "mode": "..."}
        data: [DONE]

    `query_type` 与 `processing_time` 是课案基础篇「接口职责」点名要求的：以前只发 delta，
    前端既看不出"这答来自缓存/FAQ/检索/保守回复/直答"，也拿不到服务端总耗时。
    中间事件（retrieve/rerank 明细）**仍然不透传** —— 那是本接口既有约定，
    链路可视化走 Chainlit 页面。
    """

    async def event_gen():
        import time as _time

        started = _time.perf_counter()
        query_type = "rag"
        mode = req.mode
        final: dict = {}
        # stream=True 是恒定的：这个接口的存在意义就是流式（页面上的"流式输出"
        # 开关只影响 Chainlit 侧的基础线路，不走这里）。
        async for ev in answer_events(req.question, req.mode, req.history, stream=True):
            query_type = _classify_query_type(ev, query_type)
            if ev.get("mode"):
                mode = ev["mode"]
            if ev["type"] == "token":
                # SSE 协议：每条消息形如 `data: <payload>\n\n`，空行才是消息结束符。
                # ensure_ascii=False 让中文原样输出（否则会被转成 \uXXXX 逃逸）。
                payload = {"delta": ev["text"], "mode": mode, "query_type": query_type}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            elif ev["type"] == "error":
                # 出错也走同一条通道（拼成一个 delta 推给客户端），而不是换 HTTP 状态码：
                # StreamingResponse 在**响应开始**时就把状态行（200）与响应头发了出去，
                # 那时链路还没跑完，事后无法再改 —— 所以错误只能作为正文推出去。
                payload = {"delta": ev["message"], "mode": mode, "query_type": query_type}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            elif ev["type"] == "done":
                final = ev
        # 收尾事件：把整条链路的产出一次给全（前端不必自己拼 token，也能拿到总耗时）。
        summary = {
            "done": True,
            "answer": final.get("answer", ""),
            "sources": final.get("sources") or [],
            "cache_hit": final.get("cache_hit"),
            "mode": mode,
            "query_type": query_type,
            "processing_time": round(_time.perf_counter() - started, 3),
        }
        yield f"data: {json.dumps(summary, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    # text/event-stream 是 SSE 的标准 MIME 类型。
    # 注意本接口是 POST：浏览器的 EventSource 只支持 GET，所以网页端要用
    # fetch + ReadableStream 自己逐行解析 data:，不能用 EventSource 直接订阅。
    # 也没有设置 Cache-Control / X-Accel-Buffering 之类的头 —— 本机直连（无反向代理）
    # 不需要；将来若放到 nginx 后面，代理缓冲会让"流式"退化成一次性返回。
    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.websocket("/api/ws")
async def ws_chat(websocket: WebSocket) -> None:
    """WebSocket 问答口（课案基础篇「接口职责」里的 `WS /api/ws`）。

    与 SSE 的关系：**同一套事件流，两种传输**。SSE 适合"一次提问、单向推流"；
    WebSocket 适合长连接对话（客户端可以连续发问，不必每次重建连接）。

    协议（刻意做成"发一收多"，而不是自定义复杂帧）：
        → {"question": "...", "mode": "basic", "history": []}
        ← {"type": "token", "delta": "...", "mode": "...", "query_type": "..."} × N
        ← {"type": "done", "answer": "...", "sources": [...], "processing_time": 1.23}
    用户再发下一条消息即可继续问；连接断开就结束。

    ⚠ 与 HTTP 侧同源的取舍：`answer_events` 是**同步阻塞**的异步生成器（内部跑检索/LLM），
    而 WebSocket 的 `send_json` 是异步的 —— 所以这里逐事件 await 发送，
    一条连接的问答期间会占住一个事件循环任务，但不会阻塞别的连接（每次 await 都让出）。
    """
    from starlette.websockets import WebSocketState

    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                # 协议错误**回一条错误帧**而不是直接断开：客户端能看到原因自己修
                await websocket.send_json({"type": "error", "message": "请求体必须是 JSON"})
                continue
            question = (payload.get("question") or "").strip()
            if not question:
                await websocket.send_json({"type": "error", "message": "缺少 question 字段"})
                continue

            import time as _time

            started = _time.perf_counter()
            query_type = "rag"
            mode = payload.get("mode") or "basic"
            final: dict = {}
            async for ev in answer_events(question, mode, payload.get("history") or [], stream=True):
                query_type = _classify_query_type(ev, query_type)
                if ev.get("mode"):
                    mode = ev["mode"]
                if ev["type"] == "token":
                    await websocket.send_json(
                        {"type": "token", "delta": ev["text"], "mode": mode, "query_type": query_type}
                    )
                elif ev["type"] == "error":
                    await websocket.send_json(
                        {"type": "error", "message": ev["message"], "mode": mode, "query_type": query_type}
                    )
                elif ev["type"] == "done":
                    final = ev
            await websocket.send_json({
                "type": "done",
                "answer": final.get("answer", ""),
                "sources": final.get("sources") or [],
                "cache_hit": final.get("cache_hit"),
                "mode": mode,
                "query_type": query_type,
                "processing_time": round(_time.perf_counter() - started, 3),
            })
    except WebSocketDisconnect:
        # 客户端正常断开（关页面/关连接）：不是错误，静默结束
        logger.info("[WS] 客户端已断开")
    except Exception as exc:  # noqa: BLE001 单连接异常不该影响服务
        logger.warning(f"[WS] 连接异常结束: {exc}")
    finally:
        # 只在还连着的时候关：对端已断开时再 close 会抛 RuntimeError（Starlette 的行为）
        if websocket.client_state is WebSocketState.CONNECTED:
            await websocket.close()


# ★ 挂载必须放在最后：mount_chainlit 内部执行 app.mount("/", chainlit_app)，
# 而 Starlette 的路由是**按注册顺序取第一个匹配**，Mount("/") 会吃掉所有路径 ——
# 一旦它排在 /api/* 之前，两个 API 就永远无法命中（页面能打开、接口全 404）。
# 反过来，挂到 "/" 时 Chainlit 自己那层「非挂载路径就 404」的中间件
# （chainlit/utils.py 里 startswith(api_full_path) 的判断）恒真、等于不生效，
# 这也是 API 路由能与页面和平共处的原因。
#
# 另一个副作用：这里在**导入期**就用 importlib 加载 chat_ui.py（注册 @cl.on_message 等
# 回调）。也就是说 import app.main 会连带执行 chat_ui.py 的顶层代码 ——
# 因此 chat_ui.py 里不能放会在导入期做网络/数据库 I/O 的顶层语句。
mount_chainlit(app=app, target=str(CHAT_UI_PATH), path="/")


if __name__ == "__main__":
    # 直接传 app 对象（避免 reload 模式在子进程中丢失 sys.path 引导）
    # 这里刻意不写 uvicorn.run("app.main:app", reload=True)：reload 会在子进程里按
    # import 字符串重新导入模块，上面那段 sys.path 引导就白做了（子进程的 sys.path
    # 由 uvicorn 决定），表现为子进程里 ImportError。
    # host/port 来自 .env 的 APP_HOST / APP_PORT（本机 127.0.0.1:8099）。
    uvicorn.run(app, host=settings.app.host, port=settings.app.port)

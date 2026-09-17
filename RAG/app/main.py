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
from pathlib import Path

import uvicorn
from chainlit.utils import mount_chainlit  # 把 Chainlit 的 ASGI 应用挂到本 FastAPI 应用上
from fastapi import FastAPI
from fastapi.responses import StreamingResponse  # SSE 靠它边算边推，不用等全部生成完
from pydantic import BaseModel  # 请求体校验：字段缺失/类型不对由 FastAPI 直接回 422

from config import settings
from core.routes import ROUTE_LABELS
from pipeline.modes import answer_events, answer as answer_by_mode, list_modes

# 注意此处**没有** CORSMiddleware：本服务只给同源的 Chainlit 页面与本地脚本用，
# 不做跨域开放。将来若有独立前端域，需要显式加中间件，否则浏览器端会被 CORS 拦下。
app = FastAPI(title="财务 RAG 智能问答系统")

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


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """问答接口:SSE 流式返回回答内容（四条线路统一走事件流）

    事件类型与基础线路既有的 7 种一致（start/route/rewrite/retrieve/rerank/token/done），
    另外多一个 `mode` 字段标明线路。**仍然只转发 token 与 error**：
    其余事件被丢弃是这个接口既有的约定（要完整链路可视化请走 Chainlit 页面）。
    """

    async def event_gen():
        # stream=True 是恒定的：这个接口的存在意义就是流式（页面上的"流式输出"
        # 开关只影响 Chainlit 侧的基础线路，不走这里）。
        async for ev in answer_events(req.question, req.mode, req.history, stream=True):
            if ev["type"] == "token":
                # SSE 协议：每条消息形如 `data: <payload>\n\n`，空行才是消息结束符。
                # ensure_ascii=False 让中文原样输出（否则会被转成 \uXXXX 逃逸）。
                yield f"data: {json.dumps({'delta': ev['text'], 'mode': ev.get('mode')}, ensure_ascii=False)}\n\n"
            elif ev["type"] == "error":
                # 出错也走同一条通道（拼成一个 delta 推给客户端），而不是换 HTTP 状态码：
                # StreamingResponse 在**响应开始**时就把状态行（200）与响应头发了出去，
                # 那时链路还没跑完，事后无法再改 —— 所以错误只能作为正文推出去。
                yield f"data: {json.dumps({'delta': ev['message'], 'mode': ev.get('mode')}, ensure_ascii=False)}\n\n"
        # 结束哨兵，照 OpenAI 流式的习惯给客户端一个明确收尾信号（客户端据此关闭连接）。
        yield "data: [DONE]\n\n"

    # text/event-stream 是 SSE 的标准 MIME 类型。
    # 注意本接口是 POST：浏览器的 EventSource 只支持 GET，所以网页端要用
    # fetch + ReadableStream 自己逐行解析 data:，不能用 EventSource 直接订阅。
    # 也没有设置 Cache-Control / X-Accel-Buffering 之类的头 —— 本机直连（无反向代理）
    # 不需要；将来若放到 nginx 后面，代理缓冲会让"流式"退化成一次性返回。
    return StreamingResponse(event_gen(), media_type="text/event-stream")


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

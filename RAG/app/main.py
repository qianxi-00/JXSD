# --- 路径引导：确保能导入 Python_Base 根目录的 config 与 RAG 内部包 ---
import sys as _sys
from pathlib import Path as _Path

_BASE = _Path(__file__).resolve()
while _BASE.parent != _BASE and _BASE.name != "Python_Base":
    _BASE = _BASE.parent
_sys.path.insert(0, str(_BASE))          # Python_Base 根（config.py）
_sys.path.insert(0, str(_BASE / "RAG"))  # RAG 根（core/llm/pipeline 等包）
"""应用主入口:FastAPI API 服务 + Chainlit 前端界面(端口可配置,默认 8099)"""

import json
from pathlib import Path

import uvicorn
from chainlit.utils import mount_chainlit
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import settings
from pipeline.rag_pipeline import RAGPipeline

app = FastAPI(title="财务 RAG 智能问答系统")

CHAT_UI_PATH = Path(__file__).resolve().parent / "chat_ui.py"

pipeline = RAGPipeline()


class ChatRequest(BaseModel):
    question: str


@app.post("/api/chat")
async def chat(req: ChatRequest) -> dict:
    """问答接口:调用 RAG 流水线生成回答(非流式)"""
    return await pipeline.run_and_collect(req.question)


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """问答接口:SSE 流式返回回答内容"""

    async def event_gen():
        async for ev in pipeline.run_events(req.question, stream=True):
            if ev["type"] == "token":
                yield f"data: {json.dumps({'delta': ev['text']}, ensure_ascii=False)}\n\n"
            elif ev["type"] == "error":
                yield f"data: {json.dumps({'delta': ev['message']}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


mount_chainlit(app=app, target=str(CHAT_UI_PATH), path="/")


if __name__ == "__main__":
    # 直接传 app 对象（避免 reload 模式在子进程中丢失 sys.path 引导）
    uvicorn.run(app, host=settings.app.host, port=settings.app.port)

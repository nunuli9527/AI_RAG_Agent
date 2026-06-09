"""
FastAPI SSE 流式接口
启动: uvicorn api:app --reload --port 8000
"""

import os

# 绕过系统代理（代理自签证书会导致 SSL 验证失败）
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

# 指定 SSL 证书路径（Windows 上 Python 可能找不到系统证书）
# 使用 certifi 内置路径更可靠
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import logging
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)

app = FastAPI(title="AI RAG Agent API")

# ---- CORS ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 懒加载 agent（避免 import 时就初始化向量库/模型）----
_agent = None


def _get_agent():
    """惰性初始化 ReactAgent，只在首次请求时创建"""
    global _agent
    if _agent is None:
        from agent.react_agent import ReactAgent
        _agent = ReactAgent()
        logger.info("ReactAgent 初始化完成")
    return _agent


# ---- 端点 ----

@app.get("/")
async def root():
    return {
        "message": "AI RAG Agent API",
        "docs": "/docs",
        "health": "/health",
        "stream": "/chat/stream?query=你的问题",
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/chat/stream")
async def chat_stream(
    query: str = Query(..., min_length=1, max_length=2000, description="用户问题")
):
    """SSE 流式对话接口"""

    async def generate():
        try:
            agent = _get_agent()
            for chunk in agent.execute_stream(query):
                content = chunk.strip()
                if content:
                    yield f"data: {content}\n\n"
        except Exception as e:
            logger.error(f"[chat_stream] {e}", exc_info=True)
            yield f"event: error\ndata: {str(e)}\n\n"
        finally:
            yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
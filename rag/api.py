"""FastAPI 后端接口。

薄壳层：所有业务调用 RagPipeline，本文件只做 HTTP 适配。
启动：uvicorn rag.api:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .config import get_settings
from .exceptions import RagError
from .logging_utils import get_logger
from .pipeline import get_pipeline

log = get_logger(__name__)

app = FastAPI(title="Enhanced RAG KB API", version="0.1.0")


# ====================== 请求 / 响应模型 ======================
class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    contexts: list[dict[str, Any]]
    num_candidates: int


class IngestRequest(BaseModel):
    file_path: str


class IngestResponse(BaseModel):
    added: int
    file_path: str


class DeleteRequest(BaseModel):
    ids: list[str]


class DeleteResponse(BaseModel):
    deleted: int


# ====================== 路由 ======================
@app.get("/health")
def health() -> dict[str, str]:
    """健康检查。"""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """RAG 问答主接口。"""
    try:
        result = get_pipeline().ask(req.question)
        return AskResponse(**result)
    except RagError as e:
        log.error("ask 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    """RAG 问答流式接口（SSE）。

    事件流（每行为 data: <json>\\n\\n）：
      meta  →  contexts / num_candidates
      delta →  增量答案文本，可多次
      done  →  完整答案
      error →  生成阶段错误（检索阶段错误仍走 HTTP 500）
    """
    def event_gen():
        try:
            events = get_pipeline().ask_stream(req.question)
        except RagError as e:
            # 检索阶段失败：SSE 流尚未建立前的构造错误，统一发一个 error 事件
            log.error("ask_stream 检索失败: %s", e)
            payload = json.dumps({"type": "error", "detail": str(e)}, ensure_ascii=False)
            yield f"data: {payload}\n\n"
            return
        try:
            for event in events:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except RagError as e:
            log.error("ask_stream 失败: %s", e)
            payload = json.dumps({"type": "error", "detail": str(e)}, ensure_ascii=False)
            yield f"data: {payload}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # 禁掉 Nginx 等代理缓冲
        },
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    """增量入库单个文件。"""
    path = Path(req.file_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"文件不存在: {req.file_path}")
    try:
        added = get_pipeline().ingest_file(path)
        return IngestResponse(added=added, file_path=str(path))
    except RagError as e:
        log.error("ingest 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/docs", response_model=DeleteResponse)
def delete_docs(req: DeleteRequest) -> DeleteResponse:
    """按 id 批量删除片段。"""
    if not req.ids:
        return DeleteResponse(deleted=0)
    try:
        get_pipeline().delete(req.ids)
        return DeleteResponse(deleted=len(req.ids))
    except RagError as e:
        log.error("delete 失败: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
def stats() -> dict[str, Any]:
    """向量库统计。"""
    try:
        return {"vector_count": get_pipeline().store.count()}
    except RagError as e:
        raise HTTPException(status_code=500, detail=str(e))


def main() -> None:
    """直接 python -m rag.api 时启动服务。"""
    import uvicorn

    s = get_settings()
    log.info("启动 FastAPI: port=%d", s.api_port)
    uvicorn.run(app, host="0.0.0.0", port=s.api_port)


if __name__ == "__main__":
    main()

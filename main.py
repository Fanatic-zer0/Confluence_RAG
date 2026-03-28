"""
FastAPI Application  (replaces n8n Webhook + Respond to Webhook nodes)

Endpoints:
  POST /rag      – main RAG query endpoint
  GET  /health   – liveness probe
"""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from db import ensure_feedback_table
from feedback import update_rating
from models import RAGRequest, RAGResponse
from pipeline import run
from pipeline_stream import run_stream
from preprocessing import preprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# Enable DEBUG-level logging for MCP-related modules
for _mcp_logger in ("mcp_fallback", "langchain_mcp_adapters", "httpx"):
    logging.getLogger(_mcp_logger).setLevel(logging.DEBUG)


# ── Lifespan: run setup tasks once at startup ─────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Confluence RAG service")
    ensure_feedback_table()
    yield
    logger.info("Confluence RAG service shutting down")


app = FastAPI(
    title="Confluence RAG",
    description="Self-correcting, self-learning RAG pipeline over Confluence",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.post("/rag", response_model=RAGResponse)
async def handle_rag(req: RAGRequest) -> RAGResponse:
    """
    Query the Confluence knowledge base.

    Request body:
        question   – required; natural-language question
        space_key  – optional; Confluence space key filter (e.g. "ENG")
        user_id    – optional; for feedback attribution
        session_id – optional; groups multiple turns (default: "default")
    """
    logger.info("RAG request: session=%s user=%s q=%s", req.session_id, req.user_id, req.question[:80])

    try:
        ctx = preprocess(
            question=req.question,
            space_key=req.space_key,
            user_id=req.user_id,
            session_id=req.session_id,
        )
        result = await run(ctx)
        return RAGResponse(
            answer=result.reply,
            flagged=result.flagged,
            best_score=result.best_score,
        )
    except Exception:
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail="Internal pipeline error")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# ── Streaming RAG endpoint ────────────────────────────────────────────────────

@app.post("/rag/stream")
async def handle_rag_stream(req: RAGRequest):
    """
    Streaming RAG — returns a Server-Sent Events stream.

    Event types:
      status        – pipeline progress text
      content_start – first real token is about to arrive
      chunk         – one LLM token chunk
      sources       – {'sources': [...]}
      done          – {'feedback_id', 'best_score', 'flagged'}
      error         – {'message': str}
    """
    logger.info(
        "Streaming RAG: session=%s user=%s q=%s",
        req.session_id, req.user_id, req.question[:80],
    )
    ctx = preprocess(
        question   = req.question,
        space_key  = req.space_key,
        user_id    = req.user_id,
        session_id = req.session_id,
    )

    async def event_generator():
        async for event in run_stream(ctx):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── User rating endpoint ──────────────────────────────────────────────────────

class _RatingRequest(BaseModel):
    rating:  str          # 'positive' | 'negative' | or re-submit with comment
    comment: str = ""


@app.post("/feedback/{feedback_id}/rating")
async def rate_feedback(feedback_id: int, body: _RatingRequest):
    """Record a 👍 / 👎 rating (and optional comment) for an interaction."""
    try:
        await asyncio.to_thread(update_rating, feedback_id, body.rating, body.comment)
        return {"ok": True}
    except Exception:
        logger.exception("Failed to record rating for feedback_id=%s", feedback_id)
        raise HTTPException(status_code=500, detail="Failed to record rating")


# ── Static frontend ───────────────────────────────────────────────────────────

_FRONTEND = Path(__file__).parent / "frontend"
if _FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND), html=True), name="static")

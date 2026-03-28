"""
Streaming Pipeline  — async-generator variant of pipeline.py

Yields SSE-ready event dicts in this sequence:
  {"type": "status",        "message": str}
  {"type": "content_start"}
  {"type": "chunk",         "content": str}   ← LLM tokens (<think> stripped)
  {"type": "sources",       "sources": list}
  {"type": "done",          "feedback_id": int, "best_score": float, "flagged": bool}
  {"type": "error",         "message": str}    ← only on unexpected exception
"""
from __future__ import annotations

import asyncio
import logging
from typing import AsyncGenerator, List

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

from config import settings
from feedback import store
from ingest import ingest_chunks
from mcp_fallback import run_mcp_fallback
from models import PipelineResult, QueryContext, Source
from prompt_assembly import assemble
from quality_gate import evaluate
from query_expansion import expand
from retrieval import run_retrieval_agent

logger = logging.getLogger(__name__)


# ── Online <think>…</think> suppressor ───────────────────────────────────────

class _ThinkStripper:
    """
    Suppresses <think>…</think> blocks in a token stream without buffering the
    entire response.  Because tags may span multiple tokens, a small lookahead
    buffer is kept internally so no partial tag ever leaks to output.
    """

    _OPEN  = "<think>"
    _CLOSE = "</think>"

    def __init__(self) -> None:
        self._buf      = ""
        self._in_think = False

    def feed(self, token: str) -> str:
        """Feed one token; return the portion that should be emitted."""
        self._buf += token
        return self._drain()

    def finish(self) -> str:
        """Call once when the stream ends; flushes any remaining safe content."""
        out        = self._buf if not self._in_think else ""
        self._buf  = ""
        self._in_think = False
        return out

    def _drain(self) -> str:
        result = []
        while self._buf:
            if not self._in_think:
                idx = self._buf.find(self._OPEN)
                if idx == -1:
                    # No open tag found; emit everything except the last (N-1)
                    # chars that could be the start of a partial <think> tag.
                    hold = len(self._OPEN) - 1
                    safe = max(0, len(self._buf) - hold)
                    result.append(self._buf[:safe])
                    self._buf = self._buf[safe:]
                    break
                else:
                    result.append(self._buf[:idx])
                    self._buf  = self._buf[idx + len(self._OPEN):]
                    self._in_think = True
            else:
                idx = self._buf.find(self._CLOSE)
                if idx == -1:
                    hold      = len(self._CLOSE) - 1
                    safe      = max(0, len(self._buf) - hold)
                    self._buf = self._buf[safe:]   # discard everything inside think
                    break
                else:
                    self._buf      = self._buf[idx + len(self._CLOSE):]
                    self._in_think = False
        return "".join(result)


# ── Helper ────────────────────────────────────────────────────────────────────

def _sources_md(sources: List[Source]) -> str:
    if not sources:
        return ""
    lines = [
        f"- **{s.label}**: [{s.title}]({s.url}) *(score {s.score:.3f})*"
        for s in sources
    ]
    return "\n\n---\n### Sources\n" + "\n".join(lines)


# ── Streaming pipeline ────────────────────────────────────────────────────────

async def run_stream(query_ctx: QueryContext) -> AsyncGenerator[dict, None]:
    """
    Drive the full RAG pipeline and yield SSE event dicts.

    This mirrors the logic in pipeline.run() but streams LLM tokens
    as they arrive so the UI can render a typewriter effect.
    """
    try:
        chunks: list            = []
        gate                    = None
        chunks_for_ingest: list = []

        # ── Retrieval loop (self-correcting) ──────────────────────────────────
        yield {"type": "status", "message": "🔍 Searching knowledge base…"}

        while True:
            chunks = await asyncio.to_thread(run_retrieval_agent, query_ctx)
            gate   = evaluate(chunks, query_ctx.retry_count)

            if gate.route == "proceed":
                break
            elif gate.route == "retry":
                attempt = query_ctx.retry_count + 2
                total   = settings.max_retries + 1
                yield {
                    "type":    "status",
                    "message": f"🔄 Expanding query (attempt {attempt}/{total})…",
                }
                query_ctx = await asyncio.to_thread(expand, query_ctx)
            else:
                break  # fallback

        # ── MCP fallback (self-healing ingest) ────────────────────────────────
        if gate and gate.route == "fallback":
            yield {"type": "status", "message": "🌐 Fetching live content from Confluence…"}
            chunks, chunks_for_ingest = await run_mcp_fallback(query_ctx)
            gate = evaluate(chunks, query_ctx.retry_count)

        if not chunks:
            yield {
                "type":    "error",
                "message": (
                    "No relevant content found. "
                    "Try rephrasing your question or specifying a space key."
                ),
            }
            return

        # ── Prompt assembly ───────────────────────────────────────────────────
        prompt, sources = await asyncio.to_thread(
            assemble, chunks, query_ctx.original_question
        )
        yield {"type": "status", "message": "💭 Generating answer…"}

        # ── Token streaming ───────────────────────────────────────────────────
        stream_kwargs = dict(
            model       = settings.chat_model,
            base_url    = settings.ollama_base_url,
            temperature = settings.chat_temperature,
            num_predict = settings.llm_num_predict,
        )
        if not settings.llm_think:
            stream_kwargs["extra_body"] = {"think": False}
        llm = ChatOllama(**stream_kwargs)

        stripper    = _ThinkStripper()
        full_answer = ""
        first_chunk = True

        async for lc_chunk in llm.astream([HumanMessage(content=prompt)]):
            token = lc_chunk.content or ""
            if not token:
                continue
            clean = stripper.feed(token)
            if clean:
                if first_chunk:
                    yield {"type": "content_start"}
                    first_chunk = False
                full_answer += clean
                yield {"type": "chunk", "content": clean}

        # Flush any content buffered at the end of the stream
        tail = stripper.finish()
        if tail:
            if first_chunk:
                yield {"type": "content_start"}
            full_answer += tail
            yield {"type": "chunk", "content": tail}

        # Guard: if model only returned think content, emit a notice
        logger.debug("Streaming LLM: full_answer length=%d, first 200: %r", len(full_answer), full_answer[:200])
        if not full_answer.strip():
            full_answer = "*(no answer generated — consider rephrasing your question)*"
            yield {"type": "content_start"}
            yield {"type": "chunk", "content": full_answer}

        # ── Sources event ─────────────────────────────────────────────────────
        yield {
            "type": "sources",
            "sources": [
                {
                    "label": s.label,
                    "title": s.title,
                    "url":   s.url,
                    "score": round(s.score, 3),
                }
                for s in sources
            ],
        }

        # ── Persist to feedback log ───────────────────────────────────────────
        best_score = gate.best_score if gate else 0.0
        flagged    = best_score < settings.low_confidence_threshold

        result = PipelineResult(
            reply      = full_answer + _sources_md(sources),
            sources    = sources,
            best_score = best_score,
            flagged    = flagged,
            session_id = query_ctx.session_id,
            user_id    = query_ctx.user_id,
            question   = query_ctx.original_question,
        )
        feedback_id = await asyncio.to_thread(store, result)

        yield {
            "type":        "done",
            "feedback_id": feedback_id,
            "best_score":  best_score,
            "flagged":     flagged,
        }

        # ── Background: self-healing vector ingest ────────────────────────────
        if chunks_for_ingest:
            asyncio.create_task(asyncio.to_thread(ingest_chunks, chunks_for_ingest))

    except Exception as exc:
        logger.exception("Streaming pipeline error")
        yield {"type": "error", "message": str(exc)}

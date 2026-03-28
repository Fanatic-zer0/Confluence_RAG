"""
Pipeline Orchestration

This module is the direct equivalent of the n8n workflow execution order:

  1. Retrieval Agent        → pgvector similarity search
  2. Quality Gate           → score chunks; decide route
     ├─ proceed             → skip to step 4
     ├─ retry               → Query Expansion → back to step 1
     └─ fallback            → MCP Fallback Agent (live Confluence)
                               └─ Ingest new chunks (background, self-healing)
  3. (fallback path only)
     MCP Fallback Agent     → fetch live pages, chunk, pass to step 4
  4. Prompt Assembly        → build final LLM prompt
  5. LLM Generate Answer    → call Ollama
  6. Format Response        → strip <think>, append sources markdown
  (fire-and-forget)
  7. Feedback Store         → write to rag_feedback table
""" 
from __future__ import annotations

import asyncio
import logging
import time
from typing import List

from config import settings
from feedback import store
from ingest import ingest_chunks
from llm import generate
from mcp_fallback import run_mcp_fallback
from models import Chunk, PipelineResult, QueryContext, Source
from prompt_assembly import assemble
from quality_gate import GateResult, evaluate
from query_expansion import expand
from retrieval import run_retrieval_agent

logger = logging.getLogger(__name__)


async def run(query_ctx: QueryContext) -> PipelineResult:
    """Execute the full RAG pipeline and return the final result."""

    t0 = time.perf_counter()
    chunks: List[Chunk] = []
    gate: GateResult | None = None

    # ── Step 1-2: Retrieval + Quality Gate loop ───────────────────────────────
    while True:
        logger.info(
            "Retrieval attempt #%d | query: %s",
            query_ctx.retry_count + 1,
            query_ctx.cleaned_question[:80],
        )

        t1 = time.perf_counter()
        # Run pgvector retrieval (sync → offloaded to thread)
        chunks = await asyncio.to_thread(run_retrieval_agent, query_ctx)
        logger.info("Retrieval took %.2fs", time.perf_counter() - t1)

        gate   = evaluate(chunks, query_ctx.retry_count)

        logger.info(
            "Quality gate: route=%s  best_score=%.3f  chunks=%d",
            gate.route, gate.best_score, gate.chunk_count,
        )

        if gate.route == "proceed":
            break
        elif gate.route == "retry":
            # Step 2b: expand query and loop back
            query_ctx = expand(query_ctx)
        else:
            # gate.route == "fallback"
            break

    # ── Step 3: MCP Fallback (only if retries exhausted) ─────────────────────
    chunks_for_ingest: List[dict] = []
    if gate and gate.route == "fallback":
        logger.info("Triggering MCP fallback – live Confluence fetch")
        t1 = time.perf_counter()
        chunks, chunks_for_ingest = await run_mcp_fallback(query_ctx)
        logger.info("MCP fallback took %.2fs", time.perf_counter() - t1)
        gate = evaluate(chunks, query_ctx.retry_count)

    # ── Step 4: Prompt Assembly ───────────────────────────────────────────────
    question = query_ctx.original_question
    prompt, sources = assemble(chunks, question)

    # ── Step 5: LLM Answer Generation (sync → offloaded to thread) ───────────
    t1 = time.perf_counter()
    answer = await asyncio.to_thread(generate, prompt)
    logger.info("LLM generation took %.2fs", time.perf_counter() - t1)

    # ── Step 6: Format Response ───────────────────────────────────────────────
    best_score = gate.best_score if gate else 0.0
    flagged    = best_score < settings.low_confidence_threshold

    sources_md = _build_sources_md(sources)
    final_reply = answer + sources_md

    result = PipelineResult(
        reply=final_reply,
        sources=sources,
        best_score=best_score,
        flagged=flagged,
        session_id=query_ctx.session_id,
        user_id=query_ctx.user_id,
        question=question,
    )

    logger.info("Total pipeline time: %.2fs", time.perf_counter() - t0)

    # ── Step 7: Self-healing ingest (background, non-blocking) ───────────────
    if chunks_for_ingest:
        asyncio.create_task(asyncio.to_thread(ingest_chunks, chunks_for_ingest))

    # ── Step 8: Feedback storage (background, non-blocking) ──────────────────
    asyncio.create_task(asyncio.to_thread(store, result))

    return result


def _build_sources_md(sources: List[Source]) -> str:
    if not sources:
        return ""
    lines = [
        f"- {s.label}: [{s.title}]({s.url}) (score {s.score:.3f})"
        for s in sources
    ]
    return "\n\n### Sources\n" + "\n".join(lines)

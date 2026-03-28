"""
Retrieval  (n8n nodes: "Retrieval Agent" + "Normalize Chunks")

Direct pgvector similarity search — no LLM agent needed at this stage.
The LangGraph agent overhead (2 LLM round-trips per call) was the primary
source of latency when the pipeline ran multiple retrieval attempts.
"""
from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document

from config import settings
from db import get_vector_store
from models import Chunk, QueryContext

logger = logging.getLogger(__name__)


def run_retrieval_agent(query_ctx: QueryContext) -> List[Chunk]:
    """Run pgvector similarity search synchronously.

    Called from the async pipeline via ``asyncio.to_thread``.
    """
    vs = get_vector_store()
    filter_arg = {"space_key": query_ctx.space_key} if query_ctx.space_key else None

    results: list[tuple[Document, float]] = vs.similarity_search_with_score(
        query_ctx.cleaned_question,
        k=settings.topk,
        filter=filter_arg,
    )

    chunks: List[Chunk] = []
    for doc, score in results:
        meta = doc.metadata or {}
        text = doc.page_content.strip()
        if not text:
            continue
        chunks.append(Chunk(
            chunk_text=text,
            title=meta.get("title", "Unknown"),
            source_url=meta.get("source_url", ""),
            page_id=meta.get("page_id", ""),
            space_key=meta.get("space_key", ""),
            score=round(float(score), 4),
        ))

    logger.info("Retrieval returned %d chunks | scores: %s",
                len(chunks),
                [c.score for c in chunks[:5]])
    return chunks

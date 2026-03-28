"""
Ingest  (n8n nodes: "Embed MCP Chunks" + "Ingest to PGVector (documents)")

Self-healing vector index: embeds MCP-fetched page chunks and inserts them
into the 'documents' pgvector collection so that the NEXT identical query
hits the vector DB instead of calling MCP again.

Called as a background task so it never blocks the HTTP response.
"""
from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document

from db import get_vector_store

logger = logging.getLogger(__name__)


def ingest_chunks(chunks_for_ingest: List[dict]) -> None:
    """Embed and store a list of MCP-fetched chunks into pgvector.

    Each item in chunks_for_ingest:
        {
          "content":     str,
          "page_id":     str,
          "chunk_index": int,
          "metadata":    dict   # title, source_url, page_id, space_key, origin
        }

    langchain-postgres PGVector.add_documents() handles the embedding call
    internally (via the configured OllamaEmbeddings), so no manual Ollama
    HTTP call is needed here.
    """
    if not chunks_for_ingest:
        return

    vs = get_vector_store()

    docs = [
        Document(
            page_content=c["content"],
            metadata=c.get("metadata", {}),
        )
        for c in chunks_for_ingest
        if c.get("content")
    ]

    if not docs:
        return

    try:
        vs.add_documents(docs)
        logger.info("Ingested %d chunks into vector store", len(docs))
    except Exception:
        logger.exception("Failed to ingest %d chunks — non-fatal", len(docs))

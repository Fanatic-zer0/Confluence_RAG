"""
MCP Fallback  (n8n nodes: "MCP Fallback Agent" + "Process MCP Content")

When the vector DB has no good matches after all retries, this module:
  1. Connects to the live Confluence MCP server
  2. Calls `search` directly to discover page IDs (no LLM)
  3. Calls `get_page` in parallel for the top 3 pages (no LLM)
  4. Returns:
       - retrieved_chunks  → passed to Prompt Assembly
       - chunks_for_ingest → passed to ingest.py (self-healing vector index)
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import List, Tuple

import httpx
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import settings
from models import Chunk, QueryContext

logger = logging.getLogger(__name__)


# ── HTML stripping ────────────────────────────────────────────────────────────

_HTML_SUBS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"<[^>]+>"),   " "),
    (re.compile(r"&nbsp;"),    " "),
    (re.compile(r"&amp;"),     "&"),
    (re.compile(r"&lt;"),      "<"),
    (re.compile(r"&gt;"),      ">"),
    (re.compile(r"\s{2,}"),    " "),
]


def _strip_html(html: str) -> str:
    text = html
    for pat, repl in _HTML_SUBS:
        text = pat.sub(repl, text)
    return text.strip()


# ── Fixed-size chunker with overlap ──────────────────────────────────────────

def _chunk_text(text: str, max_chars: int = 1000, overlap: int = 100) -> List[str]:
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
        start += max_chars - overlap
    return chunks


# ── MCP fallback: direct tool calls (no LLM) ─────────────────────────────────

async def run_mcp_fallback(
    query_ctx: QueryContext,
) -> Tuple[List[Chunk], List[dict]]:
    """Fetch live Confluence content by calling MCP tools directly (no LLM).

    Sequence:
      1. Call `search` to discover page IDs.
      2. Call `get_page` for the top 3 pages in parallel.

    Returns:
        retrieved_chunks:  List[Chunk]  – for Prompt Assembly
        chunks_for_ingest: List[dict]   – for self-healing vector index
    """
    mcp_config = {
        "confluence": {
            "url": settings.mcp_endpoint_url,
            "transport": settings.mcp_transport,
            "headers": {"Authorization": f"Bearer {settings.mcp_bearer_token}"},
            "timeout": settings.mcp_timeout,
            "sse_read_timeout": settings.mcp_timeout,
        }
    }

    logger.debug(
        "MCP fallback: connecting to %s (transport=%s)",
        settings.mcp_endpoint_url,
        settings.mcp_transport,
    )

    _MCP_RETRIES = 2
    mcp_tools_list = None
    for attempt in range(1, _MCP_RETRIES + 2):
        try:
            mcp_client = MultiServerMCPClient(mcp_config)
            mcp_tools_list = await mcp_client.get_tools()
            break
        except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as exc:
            logger.warning(
                "MCP fallback: connection attempt %d/%d failed (%s: %s)",
                attempt, _MCP_RETRIES + 1, type(exc).__name__, exc,
            )
            if attempt > _MCP_RETRIES:
                raise

    tools_by_name = {t.name: t for t in mcp_tools_list}
    logger.debug("MCP fallback: available tools: %s", list(tools_by_name))

    search_tool   = _find_tool(tools_by_name, ("search", "confluence_search"))
    get_page_tool = _find_tool(tools_by_name, ("get_page", "confluence_get_page", "getPage"))

    if not search_tool:
        logger.error("MCP fallback: no search tool found; aborting")
        return [], []

    # ── Step 1: search ────────────────────────────────────────────────────────
    search_kwargs: dict = {"query": query_ctx.cleaned_question, "limit": 5}
    if query_ctx.space_key:
        search_kwargs["spaces_filter"] = query_ctx.space_key
    logger.debug("MCP search call: %r", search_kwargs)

    search_raw = await search_tool.ainvoke(search_kwargs)
    logger.debug("MCP search result: %r", str(search_raw)[:800])

    page_ids = _extract_page_ids(search_raw)
    logger.debug("MCP fallback: discovered page_ids: %s", page_ids)

    if not page_ids or not get_page_tool:
        logger.warning("MCP fallback: no page IDs found or no get_page tool")
        return [], []

    # ── Step 2: fetch pages in parallel ──────────────────────────────────────
    fetch_tasks = [
        get_page_tool.ainvoke({"page_id": pid})
        for pid in page_ids[:3]
    ]
    page_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    retrieved_chunks: List[Chunk] = []
    chunks_for_ingest: List[dict] = []

    for pid, page_raw in zip(page_ids[:3], page_results):
        if isinstance(page_raw, Exception):
            logger.warning("MCP get_page(%s) failed: %s", pid, page_raw)
            continue
        logger.debug("MCP get_page(%s) result: %r", pid, str(page_raw)[:500])
        rc, ci = _parse_page(page_raw)
        retrieved_chunks.extend(rc)
        chunks_for_ingest.extend(ci)

    logger.info(
        "MCP fallback: fetched %d chunks from %d pages",
        len(retrieved_chunks), len(page_ids[:3]),
    )
    return retrieved_chunks[: settings.topk], chunks_for_ingest


# ── Helpers ───────────────────────────────────────────────────────────────────

def _find_tool(tools_by_name: dict, candidates: tuple):
    """Return the first matching tool from a list of candidate names."""
    for name in candidates:
        if name in tools_by_name:
            return tools_by_name[name]
    return None


def _unwrap_mcp_result(raw) -> any:
    """Unwrap the MCP tool result envelope.

    MCP tools return a list like:
      [{'type': 'text', 'text': '<json string>', 'id': 'lc_...'}]
    The actual payload is inside the 'text' field of the first item.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    # Handle MCP envelope: list of content blocks
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "text" in raw[0]:
        text = raw[0]["text"]
        if isinstance(text, str):
            try:
                return json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return text
        return text

    return raw


def _extract_page_ids(search_raw) -> List[str]:
    """Parse page IDs out of a search tool result."""
    data = _unwrap_mcp_result(search_raw)

    if isinstance(data, dict):
        results = data.get("results", [])
    elif isinstance(data, list):
        results = data
    else:
        return []

    ids = []
    for item in results:
        if isinstance(item, dict):
            pid = item.get("id") or item.get("page_id")
            if pid:
                ids.append(str(pid))
    return ids


def _parse_page(page_raw) -> Tuple[List[Chunk], List[dict]]:
    """Convert a get_page result into Chunk + ingest-record lists.

    Handles two response shapes:
      Shape A (standard Confluence API):
        {"id": "...", "title": "...", "body": {"storage": {"value": "<html>"}}}
      Shape B (this MCP server):
        {"metadata": {"id": "...", "title": "...", "url": "...",
                       "space": {"key": "..."}, "content": {"value": "..."}}}
    """
    data = _unwrap_mcp_result(page_raw)

    # Check for an error response
    if isinstance(data, dict) and "error" in data:
        logger.warning("MCP get_page returned error: %s", data["error"])
        return [], []

    if not isinstance(data, dict):
        return [], []

    # ── Shape B: metadata wrapper ─────────────────────────────────────────────
    if "metadata" in data:
        meta       = data["metadata"]
        page_id    = str(meta.get("id", ""))
        title      = meta.get("title", "Unknown")
        sk         = (meta.get("space") or {}).get("key", "")
        source_url = meta.get("url", "")
        body_text  = (meta.get("content") or {}).get("value", "").strip()

    # ── Shape A: standard Confluence body.storage ─────────────────────────────
    elif "body" in data:
        page_id    = str(data.get("id", ""))
        title      = data.get("title", "Unknown")
        sk         = (data.get("space") or {}).get("key", "")
        webui      = ((data.get("_links") or {}).get("webui") or "")
        source_url = webui if webui.startswith("http") else ""
        body_raw   = (
            (data.get("body") or {}).get("storage", {}).get("value", "")
            or (data.get("body") or {}).get("view", {}).get("value", "")
            or ""
        )
        body_text  = _strip_html(body_raw)

    else:
        return [], []

    if not body_text:
        return [], []

    retrieved_chunks: List[Chunk] = []
    chunks_for_ingest: List[dict] = []
    for idx, chunk in enumerate(_chunk_text(body_text)):
        metadata = {
            "title":      title,
            "source_url": source_url,
            "page_id":    page_id,
            "space_key":  sk,
            "origin":     "mcp_fallback",
        }
        retrieved_chunks.append(Chunk(
            chunk_text=chunk,
            title=title,
            source_url=source_url,
            page_id=page_id,
            space_key=sk,
            score=0.75,
        ))
        chunks_for_ingest.append({
            "content":     chunk,
            "page_id":     page_id,
            "chunk_index": idx,
            "metadata":    metadata,
        })
    return retrieved_chunks, chunks_for_ingest

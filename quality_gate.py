"""
Quality Gate  (n8n node: "Quality Gate" + "Route: Proceed / Retry / Fallback")

Scores the retrieved chunks and returns one of three routes:
  proceed  – good enough, send to Prompt Assembly
  retry    – below threshold but retries remain → Query Expansion
  fallback – retries exhausted → MCP Fallback Agent
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

from config import settings
from models import Chunk

Route = Literal["proceed", "retry", "fallback"]


@dataclass
class GateResult:
    route: Route
    best_score: float
    chunk_count: int


def evaluate(chunks: List[Chunk], retry_count: int) -> GateResult:
    """Decide the next step based on retrieval quality.

    Two-tier check:
      1. min_score  — all passing chunks must reach this floor (filters noise)
      2. high_score — the TOP chunk must reach this ceiling; without it the
                      content is probably from a different topic and MCP should
                      be tried instead.
    """
    scores = sorted([c.score for c in chunks], reverse=True)
    best_score = scores[0] if scores else 0.0

    above_floor  = [c for c in chunks if c.score >= settings.min_score]
    has_enough   = (
        len(above_floor) >= settings.min_chunks
        and best_score   >= settings.high_score
    )

    if has_enough:
        route: Route = "proceed"
    elif retry_count < settings.max_retries:
        route = "retry"
    else:
        route = "fallback"

    return GateResult(
        route=route,
        best_score=best_score,
        chunk_count=len(chunks),
    )

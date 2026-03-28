""" 
Shared data models — request/response (Pydantic) and internal pipeline state
(dataclasses).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import BaseModel


# ── HTTP layer ───────────────────────────────────────────────────────────────

class RAGRequest(BaseModel):
    question: str
    space_key: Optional[str] = None
    user_id: Optional[str] = None
    session_id: str = "default"


class RAGResponse(BaseModel):
    answer: str
    flagged: bool
    best_score: float


# ── Internal pipeline state ──────────────────────────────────────────────────

@dataclass
class QueryContext:
    """State that flows through the entire pipeline."""
    original_question: str
    cleaned_question: str
    space_key: Optional[str]
    user_id: Optional[str]
    session_id: str
    date_filter: Optional[str]
    retry_count: int = 0


@dataclass
class Chunk:
    """A single retrieved document chunk."""
    chunk_text: str
    title: str
    source_url: str
    page_id: str
    space_key: str
    score: float


@dataclass
class Source:
    """Citation summary shown in the final answer."""
    label: str
    title: str
    url: str
    score: float


@dataclass
class PipelineResult:
    """Final output of the RAG pipeline."""
    reply: str
    sources: List[Source]
    best_score: float
    flagged: bool
    session_id: str
    user_id: Optional[str]
    question: str

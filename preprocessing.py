"""
Query Pre-processing  (n8n node: "Query Pre-processing")

- Normalises the question (lowercase)
- Expands common abbreviations
- Extracts space_key from natural-language phrasing
- Detects date-range keywords
"""
from __future__ import annotations

import re
from typing import Optional

from models import QueryContext

# ── Abbreviation table ────────────────────────────────────────────────────────

ABBR_MAP: dict[str, str] = {
    "p1":  "priority one",
    "p2":  "priority two",
    "p3":  "priority three",
    "rca": "root cause analysis",
    "sre": "site reliability engineering",
    "sla": "service level agreement",
    "rto": "recovery time objective",
    "rpo": "recovery point objective",
    "ci":  "continuous integration",
    "cd":  "continuous deployment",
    "k8s": "kubernetes",
}

# ── Date-range heuristics ─────────────────────────────────────────────────────

DATE_RULES: list[tuple[str, str]] = [
    ("last week",    "LAST_WEEK"),
    ("last month",   "LAST_MONTH"),
    ("last quarter", "LAST_QUARTER"),
    ("this year",    "THIS_YEAR"),
]

# ── Space-key extraction patterns ─────────────────────────────────────────────

_SPACE_PATTERNS = [
    re.compile(r"\bspace[_\s]key\s+([A-Za-z0-9_-]+)", re.IGNORECASE),
    re.compile(r"\bin\s+space\s+([A-Za-z0-9_-]+)",     re.IGNORECASE),
]


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess(
    question: str,
    space_key: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: str = "default",
) -> QueryContext:
    """Normalise the raw question and extract structured metadata."""
    q = question.lower().strip()

    # Abbreviation expansion
    for abbr, full in ABBR_MAP.items():
        q = re.sub(r"\b" + re.escape(abbr) + r"\b", full, q)

    # Extract space_key from text when not supplied explicitly
    if space_key is None:
        for pat in _SPACE_PATTERNS:
            m = pat.search(q)
            if m:
                space_key = m.group(1).upper()
                break

    # Date-range detection
    date_filter: Optional[str] = None
    for phrase, flag in DATE_RULES:
        if phrase in q:
            date_filter = flag
            break

    return QueryContext(
        original_question=question.strip(),
        cleaned_question=q,
        space_key=space_key,
        user_id=user_id,
        session_id=session_id,
        date_filter=date_filter,
        retry_count=0,
    )

"""
Query Expansion  (n8n node: "Query Expansion")

Self-correcting retry strategy:
  Retry 1 – synonym injection (broadens vocabulary)
  Retry 2 – stop-word stripping (strips to noun core)
"""
from __future__ import annotations

import re
from dataclasses import replace

from models import QueryContext

# ── Synonym map for retry 1 ───────────────────────────────────────────────────

_SYNONYM_MAP: dict[str, str] = {
    "error":   "error issue failure",
    "setup":   "setup configuration install",
    "deploy":  "deploy deployment release",
    "monitor": "monitor monitoring observability",
    "alert":   "alert alerting notification",
    "access":  "access permission authorisation",
    "slow":    "slow performance latency timeout",
    "fail":    "fail failure crash down",
}

# ── Stop-word pattern for retry 2 ────────────────────────────────────────────

_STOP = re.compile(
    r"\b(how|what|why|when|where|can|could|should|would|do|does|is|are"
    r"|the|a|an|to|in|of|for|with|about|that)\b",
    re.IGNORECASE,
)


def expand(ctx: QueryContext) -> QueryContext:
    """Return a new QueryContext with an expanded query and incremented retry_count."""
    q = ctx.cleaned_question

    if ctx.retry_count == 0:
        # Strategy 1: inject synonyms to broaden vocabulary
        for word, expansion in _SYNONYM_MAP.items():
            q = re.sub(r"\b" + re.escape(word) + r"\b", expansion, q)
    else:
        # Strategy 2: strip down to noun/keyword core
        q = _STOP.sub(" ", q)
        q = re.sub(r"\s{2,}", " ", q).strip()

    return replace(ctx, cleaned_question=q, retry_count=ctx.retry_count + 1)

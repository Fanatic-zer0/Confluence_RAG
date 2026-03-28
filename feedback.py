"""
Feedback Store

Self-learning log: every Q&A interaction is persisted to the 'rag_feedback'
table.  Interactions with best_score < LOW_CONFIDENCE_THRESHOLD are flagged
for human review and future fine-tuning.

Called as a fire-and-forget background task so it never delays the response.
"""
from __future__ import annotations

import json
import logging

from config import settings
from db import get_db_pool
from models import PipelineResult

logger = logging.getLogger(__name__)

_SQL = """
    INSERT INTO rag_feedback
        (session_id, user_id, question, answer, best_score, sources, flagged)
    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
    RETURNING id
"""

_RATING_SQL = """
    UPDATE rag_feedback
       SET user_rating = %s, user_comment = %s
     WHERE id = %s
"""


def store(result: PipelineResult) -> int:
    """Write a Q&A interaction to the rag_feedback table; returns the new row id."""
    flagged = result.best_score < settings.low_confidence_threshold

    sources_json = json.dumps([
        {
            "label": s.label,
            "title": s.title,
            "url":   s.url,
            "score": s.score,
        }
        for s in result.sources
    ])

    p = get_db_pool()
    conn = p.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(_SQL, (
                result.session_id,
                result.user_id or "anonymous",
                result.question,
                result.reply,
                result.best_score,
                sources_json,
                flagged,
            ))
            row = cur.fetchone()
        conn.commit()
        logger.debug(
            "Feedback stored: session=%s flagged=%s score=%.3f",
            result.session_id, flagged, result.best_score,
        )
        return row[0] if row else -1
    except Exception:
        conn.rollback()
        logger.exception("Failed to store feedback — non-fatal")
        return -1
    finally:
        p.putconn(conn)


def update_rating(feedback_id: int, rating: str, comment: str = "") -> None:
    """Record a user 👍 / 👎 rating against a stored Q&A row."""
    p    = get_db_pool()
    conn = p.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute(_RATING_SQL, (rating, comment, feedback_id))
        conn.commit()
        logger.debug("Rating recorded: id=%s rating=%s", feedback_id, rating)
    except Exception:
        conn.rollback()
        logger.exception("Failed to update rating for feedback_id=%s", feedback_id)
    finally:
        p.putconn(conn)

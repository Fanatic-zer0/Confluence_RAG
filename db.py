"""
Database singletons.

TWO databases (same Postgres server, different tables / roles):
  1. 'documents'    – pgvector (LangChain PGVector)
     Populated by the ingestion pipeline + auto-healed by MCP fallback.
  2. 'rag_feedback' – plain rows (psycopg2 connection pool)
     Self-learning log; every Q&A is stored, low-score rows flagged.
"""
from __future__ import annotations

import logging
from functools import lru_cache

import psycopg2
from psycopg2 import pool
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

from config import settings

logger = logging.getLogger(__name__)

# ── PGVector store ───────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=settings.embed_model,
        base_url=settings.ollama_base_url,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> PGVector:
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=settings.documents_collection,
        connection=settings.postgres_dsn,
        use_jsonb=True,
    )


# ── Raw Postgres pool (for rag_feedback inserts) ─────────────────────────────

_conn_pool: pool.SimpleConnectionPool | None = None


def _raw_dsn() -> str:
    """Convert SQLAlchemy DSN to libpq DSN for psycopg2."""
    return (
        settings.postgres_dsn
        .replace("postgresql+psycopg2://", "postgresql://")
        .replace("postgresql+psycopg://", "postgresql://")
    )


def get_db_pool() -> pool.SimpleConnectionPool:
    global _conn_pool
    if _conn_pool is None:
        _conn_pool = pool.SimpleConnectionPool(1, 10, _raw_dsn())
    return _conn_pool


# ── Schema bootstrap ─────────────────────────────────────────────────────────

def ensure_feedback_table() -> None:
    """Create rag_feedback table if it doesn't exist (called at startup)."""
    p = get_db_pool()
    conn = p.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS rag_feedback (
                    id           SERIAL PRIMARY KEY,
                    session_id   TEXT,
                    user_id      TEXT,
                    question     TEXT,
                    answer       TEXT,
                    best_score   FLOAT,
                    sources      JSONB,
                    flagged      BOOLEAN DEFAULT FALSE,
                    user_rating  TEXT,
                    user_comment TEXT,
                    created_at   TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            # Idempotent migration for existing deployments
            for col, defn in [("user_rating", "TEXT"), ("user_comment", "TEXT")]:
                cur.execute(
                    f"ALTER TABLE rag_feedback"
                    f" ADD COLUMN IF NOT EXISTS {col} {defn}"
                )
        conn.commit()
        logger.info("rag_feedback table ready")
    finally:
        p.putconn(conn)

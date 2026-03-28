-- ─────────────────────────────────────────────────────────────────────────────
-- Schema bootstrap for Confluence RAG
-- Run this once against your Postgres database before starting the service.
-- ─────────────────────────────────────────────────────────────────────────────

-- Enable the pgvector extension (requires pgvector to be installed) 
CREATE EXTENSION IF NOT EXISTS vector;

-- ── Table 1: documents (pgvector semantic search) ────────────────────────────
-- LangChain PGVector manages this table automatically when you call
-- PGVector(...) in db.py, but you can pre-create it for explicit schema control.

CREATE TABLE IF NOT EXISTS langchain_pg_collection (
    uuid    UUID PRIMARY KEY,
    name    VARCHAR,
    cmetadata JSONB
);

CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
    id          UUID         PRIMARY KEY,
    collection_id UUID       REFERENCES langchain_pg_collection(uuid) ON DELETE CASCADE,
    embedding   VECTOR(768),          -- nomic-embed-text produces 768-dim vectors
    document    TEXT,
    cmetadata   JSONB,
    custom_id   VARCHAR
);

CREATE INDEX IF NOT EXISTS idx_embedding_ivfflat
    ON langchain_pg_embedding
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ── Table 2: rag_feedback (self-learning log) ─────────────────────────────────

CREATE TABLE IF NOT EXISTS rag_feedback (
    id           SERIAL PRIMARY KEY,
    session_id   TEXT,
    user_id      TEXT,
    question     TEXT,
    answer       TEXT,
    best_score   FLOAT,
    sources      JSONB,
    flagged      BOOLEAN     DEFAULT FALSE,
    user_rating  TEXT,                         -- 'positive' | 'negative' | NULL
    user_comment TEXT,
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_flagged    ON rag_feedback (flagged);
CREATE INDEX IF NOT EXISTS idx_feedback_session    ON rag_feedback (session_id);
CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON rag_feedback (created_at DESC);

-- ── View: low-confidence interactions for human review ────────────────────────

CREATE OR REPLACE VIEW rag_review_queue AS
SELECT
    id,
    created_at,
    session_id,
    user_id,
    question,
    LEFT(answer, 200) AS answer_preview,
    best_score,
    sources,
    user_rating,
    user_comment
FROM rag_feedback
WHERE flagged = TRUE
ORDER BY created_at DESC;

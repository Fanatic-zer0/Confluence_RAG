""" 
Central configuration. All values can be overridden via environment variables
or a .env file at the project root.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── Ollama ──────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    embed_model: str = "nomic-embed-text"
    chat_model: str = "qwen3:latest"
    chat_temperature: float = 0.1
    llm_num_predict: int = -1               # -1 = unlimited; qwen3 needs room after <think>
    llm_think: bool = False                 # False = disable qwen3 CoT thinking mode

    # ── Postgres / PGVector ──────────────────────────────────────────────────
    # SQLAlchemy DSN (used by langchain-postgres PGVector)
    postgres_dsn: str = "postgresql+psycopg2://postgres:password@localhost:5432/confluence_rag"
    documents_collection: str = "documents"   # pgvector table / collection name
    topk: int = 8

    # ── MCP Server ───────────────────────────────────────────────────────────
    mcp_endpoint_url: str = "http://localhost:9240/mcp"
    mcp_bearer_token: str = ""
    mcp_transport: str = "streamable_http"  # "streamable_http" or "sse"
    mcp_timeout: float = 60.0               # seconds for MCP HTTP calls

    # ── Quality Gate ─────────────────────────────────────────────────────────
    min_score: float = 0.35             # minimum pgvector cosine similarity
    high_score: float = 0.60            # top chunk must reach this to skip MCP
    min_chunks: int = 2                 # minimum chunks for a confident answer
    max_retries: int = 2                # query-expansion retries before fallback
    low_confidence_threshold: float = 0.40   # flag threshold for rag_feedback


settings = Settings()

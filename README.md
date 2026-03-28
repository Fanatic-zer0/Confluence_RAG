# Confluence RAG — Python / LangChain

## Overview

A production-grade Retrieval-Augmented Generation (RAG) system that lets you query your Confluence knowledge base in natural language. Built on FastAPI, LangChain, pgvector, and Ollama — runs entirely on-premise with no external AI API calls.

### What makes this different from a standard RAG system

Most RAG implementations follow a single, static path: embed the query → search the vector DB → feed chunks into an LLM. This project adds several layers on top of that baseline that make the system **self-correcting**, **self-healing**, and **self-learning** over time.

| Capability | Standard RAG | This project |
|---|---|---|
| Vector search | ✅ | ✅ |
| Quality scoring per response | ❌ | ✅ Confidence score + `flagged` flag |
| Automatic retry on poor results | ❌ | ✅ Up to 2 query-expansion retries |
| Live fallback when index is stale | ❌ | ✅ MCP live-fetch from Confluence |
| Index self-heals after fallback | ❌ | ✅ MCP chunks ingested in background |
| Tracks every Q&A for review | ❌ | ✅ `rag_feedback` table |
| Human review queue for bad answers | ❌ | ✅ `rag_review_queue` view |
| User thumbs up/down feedback loop | ❌ | ✅ Captured per answer |
| Fully local / on-premise | Optional | ✅ Ollama + local Postgres |

#### Self-correcting query pipeline

When a search returns low-confidence chunks (below the configurable `MIN_SCORE` threshold), the system does **not** fall back immediately. It first tries to fix the query itself:

1. **Query expansion** — strips stop-words, injects synonyms, and re-runs the vector search (up to `MAX_RETRIES` times).
2. Only after retries are exhausted does it escalate to the live Confluence fallback via MCP.

This means a user asking _"how do I fix the deploy thing?"_ gets the same answer as someone asking _"CI/CD deployment troubleshooting"_ — without any manual synonym maintenance.

#### Self-healing vector index

The pgvector index starts empty. As users ask questions the system hasn't seen before, the MCP fallback fetches the live Confluence pages and immediately ingests those chunks into pgvector **as a background task**. The next time anyone asks a similar question, the answer is served from the fast local index — no MCP round-trip required. The index heals itself automatically from real usage.

#### Self-learning feedback loop

Every interaction — question, answer, confidence score, sources used — is written to a `rag_feedback` Postgres table. Responses that score below the `LOW_CONFIDENCE_THRESHOLD` are automatically flagged. The `rag_review_queue` view surfaces all flagged interactions for human review, making it straightforward to identify gaps in the index, correct wrong answers, and feed corrections back into fine-tuning or prompt improvements over time.

#### Fully local and private

All inference runs through **Ollama** on your own hardware. No question text or document content leaves your environment. There are no calls to OpenAI, Anthropic, or any cloud AI API.

---

## Project layout

```
confluence_rag/
├── config.py          # All settings (Pydantic, overrideable via .env)
├── models.py          # Request/Response Pydantic models + internal dataclasses
├── db.py              # DB singletons: PGVector store + psycopg2 pool
├── preprocessing.py   # Query normalisation, abbreviation expansion, filter extraction
├── retrieval.py       # PGVector similarity-search agent (LangChain tool-calling)
├── quality_gate.py    # Score chunks → proceed / retry / fallback routing
├── query_expansion.py # Self-correcting retry: synonym injection + stop-word stripping
├── mcp_fallback.py    # Live Confluence fetch via MCP + HTML strip + chunking
├── ingest.py          # Self-healing: embed + store MCP-fetched chunks to pgvector
├── prompt_assembly.py # Build final LLM prompt from chunks
├── llm.py             # ChatOllama call + <think> block stripping
├── feedback.py        # Write Q&A to rag_feedback table (self-learning log)
├── pipeline.py        # Full orchestration (retry loop + fallback + background tasks)
├── main.py            # FastAPI app  POST /rag  GET /health
├── db_init.sql        # One-time schema bootstrap
├── requirements.txt
└── .env.example
```

## Architecture

```
POST /rag
  │
  ▼
Query Pre-processing
  • lowercase, abbreviation expansion, space_key extraction, date heuristics
  │
  ▼
Retrieval Agent  [pgvector tool]
  • similarity_search_with_score on 'documents' collection
  │
  ▼
Quality Gate ──── proceed ──────────────────────────────────────────────────┐
  │                                                                          │
  ├─ retry  →  Query Expansion  →  (back to Retrieval Agent, max 2 retries) │
  │                                                                          │
  └─ fallback (retries exhausted)                                            │
       │                                                                     │
       ▼                                                                     │
  MCP Fallback Agent  [confluence_search MCP → confluence_get_page MCP]     │
       │                                                                     │
       ├─ chunks_for_ingest ──► Ingest to PGVector  (background task)       │
       │                        self-healing index                           │
       └─ retrieved_chunks ─────────────────────────────────────────────────┘
                                                                             │
                                                                             ▼
                                                                    Prompt Assembly
                                                                             │
                                                                             ▼
                                                                   LLM Generate Answer
                                                                    (ChatOllama qwen3)
                                                                             │
                                                                             ▼
                                                                    Format Response
                                                                    + Sources markdown
                                                                             │
                                                          ┌──────────────────┤
                                                          │                  │
                                                          ▼                  ▼
                                                   Feedback Store     HTTP Response
                                                 (background task)   { answer, flagged,
                                               → rag_feedback table    best_score }
```

## Two databases

| Table | Type | Purpose |
|---|---|---|
| `langchain_pg_embedding` | pgvector | Semantic vector search. Auto-populated by ingestion pipeline + self-healing MCP fallback |
| `rag_feedback` | plain Postgres | Self-learning Q&A log. Flagged rows (`best_score < 0.40`) are available in `rag_review_queue` view for human correction / fine-tuning |

## Prerequisites

### 1 · PostgreSQL with pgvector

#### Install PostgreSQL

**macOS (Homebrew)**
```bash
brew install postgresql@16
brew services start postgresql@16
echo 'export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
```

**Ubuntu / Debian**
```bash
sudo apt-get update && sudo apt-get install -y postgresql postgresql-contrib
sudo systemctl enable --now postgresql
```

**Windows** — download the installer from [postgresql.org/download/windows](https://www.postgresql.org/download/windows/)

#### Install pgvector

**macOS (Homebrew)**
```bash
brew install pgvector
```

**Ubuntu / Debian** (adjust `16` to your Postgres major version)
```bash
sudo apt-get install -y postgresql-server-dev-16
git clone --depth=1 --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector && make && sudo make install && cd ..
```

#### Create the database

```bash
# Open a psql shell (adjust -U if your superuser is not "postgres")
psql -U postgres
```
```sql
CREATE DATABASE confluence_rag;
-- (optional) create a dedicated application user
CREATE USER rag_user WITH PASSWORD 'changeme';
GRANT ALL PRIVILEGES ON DATABASE confluence_rag TO rag_user;
\q
```

#### Bootstrap the schema

```bash
psql postgresql://rag_user:changeme@localhost:5432/confluence_rag -f db_init.sql
```

`db_init.sql` will:
- Enable the `vector` extension (requires pgvector)
- Create `langchain_pg_collection` + `langchain_pg_embedding` (PGVector tables)
- Create `rag_feedback` (self-learning log) with relevant indexes
- Create `rag_review_queue` view for human review of low-confidence answers

--- 

### 2 · Ollama — embedding model + LLM

#### Install Ollama

**macOS / Linux**
```bash
curl -fsSL https://ollama.com/install.sh | sh
# macOS alternative via Homebrew:
brew install ollama
```

**Windows** — download from [ollama.com/download](https://ollama.com/download)

#### Pull the required models

```bash
# Embedding model — 768-dim vectors, required for pgvector similarity search
ollama pull nomic-embed-text

# LLM — used for answer generation (non-thinking mode, see note below)
ollama pull qwen3:latest
```

> **Non-thinking mode:** `qwen3` can emit verbose `<think>…</think>` reasoning blocks.
> The pipeline has `LLM_THINK=false` by default, which disables the thinking mode via
> Ollama's `num_ctx` options. Any `<think>` blocks that appear are stripped automatically.
> To use a different model, set `CHAT_MODEL=<model>` in `.env`.

#### Verify Ollama is running

```bash
# Start manually if not autostarted:
ollama serve

# Confirm registered models:
curl -s http://localhost:11434/api/tags | python3 -m json.tool
```

---

### 3 · Confluence MCP Server

The MCP (Model Context Protocol) fallback fetches live Confluence pages when the
vector DB has no confident matches. The pipeline expects an HTTP endpoint at
`http://localhost:9240/mcp` that exposes `confluence_search` and `confluence_get_page` tools.

The recommended server is **[mcp-atlassian](https://github.com/sooperset/mcp-atlassian)**
(supports both Confluence Cloud and Data Center, and provides exactly these tool names).

#### Install mcp-atlassian

```bash
pip install mcp-atlassian
# or with uv:
uv tool install mcp-atlassian
```

#### Obtain a Confluence API token

| Edition | How to get a token |
|---|---|
| **Confluence Cloud** | [Atlassian account](https://id.atlassian.com/manage-profile/security/api-tokens) → Security → API tokens → Create |
| **Data Center / Server** | Profile picture → Personal Access Tokens → Create token |

#### Start the MCP server

```bash
# Confluence Cloud
CONFLUENCE_URL=https://your-org.atlassian.net \
CONFLUENCE_USERNAME=your@email.com \
CONFLUENCE_API_TOKEN=your_api_token \
mcp-atlassian --transport streamable-http --port 9240

# Confluence Data Center / Server (Personal Access Token)
CONFLUENCE_URL=https://confluence.your-company.com \
CONFLUENCE_PERSONAL_TOKEN=your_pat_token \
mcp-atlassian --transport streamable-http --port 9240
```

The server is ready when you see output like `Serving MCP on http://0.0.0.0:9240`.

#### Configure `.env`

```bash
CONFLUENCE_URL=https://company.com
CONFLUENCE_USERNAME=user@company.com
CONFLUENCE_PERSONAL_TOKEN=TESTABCNSDUYEQTRFHGSFDGHSDFTWDHGFDTEST
FASTMCP_TRANSPORT=sse
FASTMCP_PORT=9240
READ_ONLY_MODE=true
TOOLSETS=confluence_pages
CONFLUENCE_SSL_VERIFY=false
```




#### Start the MCP server

```bash
docker run --rm -p 9240:9240 \                                                                           ☸ eks-m1-npe 17:54:11
  --env-file atlassian-mcp.env \
  ghcr.io/sooperset/mcp-atlassian:latest \
  --transport streamable-http --port 9240 --host 0.0.0.0
```

#### Verify the MCP server
```bash
curl -s -X POST http://localhost:9240/mcp \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer ${MCP_BEARER_TOKEN}" \
     -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

The response should include `confluence_search` and `confluence_get_page` in the `tools` array.

---

## Quick Start

> Complete all three prerequisites above before running these steps.

### 1. Install dependencies
```bash
cd confluence_rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure
```bash
cp .env.example .env
# Edit .env — set POSTGRES_DSN, OLLAMA_BASE_URL, MCP_ENDPOINT_URL, MCP_BEARER_TOKEN
```

Key variables:
| Variable | Default | Description |
|---|---|---|
| `POSTGRES_DSN` | `postgresql+psycopg2://postgres:password@localhost:5432/confluence_rag` | SQLAlchemy DSN |
| `EMBED_MODEL` | `nomic-embed-text` | Ollama embedding model |
| `CHAT_MODEL` | `qwen3:latest` | Ollama LLM |
| `LLM_THINK` | `false` | Disable qwen3 CoT thinking mode |
| `MCP_ENDPOINT_URL` | `http://localhost:9240/mcp` | Confluence MCP server URL |
| `MCP_BEARER_TOKEN` | _(empty)_ | API token or PAT |

### 3. Initialise the database
```bash
psql "$POSTGRES_DSN" -f db_init.sql
# Or using an explicit DSN:
psql postgresql://rag_user:changeme@localhost:5432/confluence_rag -f db_init.sql
```

### 4. Run
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Example request
```bash
curl -X POST http://localhost:8000/rag \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "How do I configure SLA alerts in the SRE runbook?",
    "space_key": "ENG",
    "user_id": "alice",
    "session_id": "session-123"
  }'
```
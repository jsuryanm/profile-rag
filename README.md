# Profile RAG 🔍

An end-to-end AI system that scrapes LinkedIn profiles via an MCP server, indexes them using RAG (Retrieval-Augmented Generation), and lets you ask natural language questions about a person — including career summaries, icebreaker questions, and networking tips.

---

## Features

- **LinkedIn Profile Scraping** via MCP server (browser-based, authenticated)
- **RAG Pipeline** powered by LlamaIndex + ChromaDB for profile Q&A
- **Router Query Engine** — automatically routes factual questions vs. report generation
- **Agentic RAG** with conversation memory for follow-up questions
- **FastAPI REST API** with `/profile` and `/ask` endpoints
- **RAG Evaluation** using LlamaIndex's `FaithfulnessEvaluator` and `RelevancyEvaluator`

---

## Architecture

```
User Request
     │
     ▼
FastAPI (api/app.py)
     │
     ▼
ProfileService (src/services/profile_service.py)
     │
     ├──► LinkedIn MCP Client (mcp_client/linkedin_client.py)
     │         │
     │         ▼
     │    LinkedIn MCP Server (browser scraping via Patchright)
     │
     ├──► Data Processing (src/processing/data_processing.py)
     │         │
     │         ▼
     │    JSONReader → SentenceSplitter → Nodes
     │
     ├──► Query Engine (src/rag/query_engine.py)
     │         │
     │         ├── ChromaDB Vector Index
     │         ├── RouterQueryEngine
     │         │     ├── profile_qa    (factual questions)
     │         │     └── profile_report (summaries, icebreakers, networking)
     │         └── FunctionAgent (agentic mode with memory)
     │
     └──► Eval (src/rag/eval.py)
               ├── FaithfulnessEvaluator
               └── RelevancyEvaluator
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Embeddings | HuggingFace (`BAAI/bge-small-en-v1.5`) |
| RAG Framework | LlamaIndex |
| Vector Store | ChromaDB (persistent) |
| MCP Client | `llama-index-tools-mcp` v0.4.8 |
| MCP Server | `linkedin-scraper-mcp` (local clone) |
| API | FastAPI |
| Settings | Pydantic Settings |

---

## Project Structure

```
profile-rag/
├── api/
│   └── app.py                  # FastAPI app with /health, /profile, /ask endpoints
├── mcp_client/
│   └── linkedin_client.py      # MCP client + FunctionAgent for LinkedIn scraping
├── src/
│   ├── config/
│   │   ├── settings.py         # Pydantic settings (env vars, prompts, RAG config)
│   │   └── logger.py           # Logging setup
│   ├── processing/
│   │   └── data_processing.py  # JSON loading, chunking pipeline
│   ├── rag/
│   │   ├── query_engine.py     # RouterQueryEngine + FunctionAgent + ChromaDB
│   │   └── eval.py             # RAG evaluation (faithfulness + relevancy)
│   ├── llm/
│   │   └── llm_interface.py    # Cached Groq LLM singleton
│   └── services/
│       └── profile_service.py  # Orchestrates fetch → index → query flow
├── tests/
│   ├── test_router.py          # Tests router Q&A and report generation
│   ├── test_eval.py            # Runs RAG evaluation against a live profile
│   └── test_mcp_tools.py       # Tests MCP tool connectivity
├── linkedin-mcp-server/        # Local clone of the LinkedIn MCP server
├── .env                        # Environment variables (not committed)
├── pyproject.toml
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- A Groq API key ([console.groq.com](https://console.groq.com))

### 1. Clone and Install

```bash
git clone https://github.com/jsuryanm/profile-rag.git
cd profile-rag
uv sync
```

### 2. Install the LinkedIn MCP Server

```bash
git clone https://github.com/stickerdaniel/linkedin-mcp-server.git
cd linkedin-mcp-server
uv pip install -e .
cd ..
```

### 3. Configure Environment

Create a `.env` file in the project root and store your groq api key:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 4. LinkedIn Login (One-Time Setup)

```bash
uvx linkedin-scraper-mcp --login
```

This opens a browser for manual LinkedIn login. The session is saved to `~/.linkedin-mcp/profile/` and reused automatically.

> **Note:** Sessions expire over time. Re-run `--login` if you encounter authentication errors.

---

## Running the Project

### Step 1: Start the LinkedIn MCP Server

```bash
uv run -m linkedin_mcp_server --transport streamable-http --host 127.0.0.1 --port 8080 --path /mcp
```

### Step 2: Open another terminal and start the FastAPI Server

```bash
uvicorn api.app:app --reload --port 8000
```

---

## API Usage

### Load a Profile

```bash
POST /profile
{
  "linkedin_url": "https://www.linkedin.com/in/username/"
}
```

**Response:**
```json
{
  "status": "loaded",
  "name": "John Doe",
  "headline": "Senior Engineer at Acme",
  "location": "Singapore",
  "chunks_indexed": 4
}
```

### Ask a Question

```bash
POST /ask
{
  "question": "What is his current role?",
  "use_agent": false
}
```

**Response:**
```json
{
  "question": "What is his current role?",
  "answer": "John is a Senior Engineer at Acme Corp.",
  "profile": "John Doe",
  "mode": "router"
}
```

Set `use_agent: true` for follow-up conversations with memory.

### Health Check

```bash
GET /health
```

---

## Query Modes

| Mode | `use_agent` | Description |
|---|---|---|
| Router (default) | `false` | Stateless. Routes between QA and report tools automatically |
| Agentic | `true` | Adds conversation memory for follow-up questions |

### Router Tool Selection

The `RouterQueryEngine` automatically picks the right tool:

| Question Type | Tool Selected | Examples |
|---|---|---|
| Factual | `profile_qa` | Current role, location, education, companies |
| Analytical | `profile_report` | Career summary, icebreakers, networking tips |

---

## Running Tests

```bash
# Test the router (factual + complex questions)
python -m tests.test_router

# Test RAG evaluation
python -m tests.test_eval

# Test MCP tool connectivity
python -m tests.test_mcp_tools
```

---

## RAG Evaluation

Evaluation runs automatically in the background after every `/profile` load. Results are logged to `logs/`.

To run evaluation manually:

```bash
python -m tests.test_eval
```

**Metrics:**

| Metric | Description |
|---|---|
| **Faithfulness** | Is the answer grounded in retrieved context? (no hallucination) |
| **Relevancy** | Does the answer address what was asked in the question? |

**Example output:**
```
faithfulness avg score: 1.00 (5 queries)
relevancy avg score: 0.80 (5 queries)

Q5: passing=False | score=0.0
    feedback: Response calculates experience from incomplete context
```

> A relevancy failure on "years of experience" questions is expected — this requires arithmetic across chunks and is a known RAG limitation. Use factual questions for reliable evaluation.

---

## Configuration

All settings are in `src/config/settings.py` and can be overridden via `.env`:

| Setting | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | required | Groq API key |
| `llm_model_id` | `llama-3.3-70b-versatile` | Groq model |
| `embedding_model_id` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `chunk_size` | `512` | RAG chunk size |
| `chunk_overlap` | `64` | RAG chunk overlap |
| `similarity_top_k` | `3` | Number of chunks retrieved per query |
| `mcp_server_url` | `http://127.0.0.1:8080/mcp` | LinkedIn MCP server URL |
| `temperature` | `0.0` | LLM temperature |

---

## Known Limitations

- **LinkedIn Rate Limiting** — Scraping the same profile repeatedly within a short window will trigger rate limits. Wait 2-3 minutes between test runs.
- **Groq TPM Limits** — The free tier is limited to 12,000 tokens/minute. Running eval immediately after a profile fetch can trigger 429 errors (auto-retried).
- **Chunk Count** — Rich profiles with long experience histories may produce only 1-2 chunks at `chunk_size=512`. Increase to `1024` if answers seem incomplete.
- **Windows + stdio** — The LinkedIn MCP server must be run as an HTTP server on Windows. stdio transport is not reliable due to stdout pollution from the browser engine.

---

## Roadmap

- [ ] Job description analysis from LinkedIn job postings
- [ ] Candidate vs. job fit scoring (0–100)
- [ ] Skill gap identification
- [ ] Resume improvement suggestions
- [ ] Certification recommendations based on skill gaps
- [ ] ReAct agent for multi-source reasoning (profile + job description + web search)
- [ ] Docker Compose setup for one-command startup

---

## License

MIT
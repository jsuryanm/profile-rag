# TalentRadar

An End-to-End Agentic AI system that scrapes LinkedIn profiles via an MCP server, indexes them using RAG (Retrieval-Augmented Generation), and lets you ask natural language questions about a person — including career summaries, icebreaker questions, and networking tips.

It also includes a full **Resume Analyzer** pipeline: upload a resume PDF, load a LinkedIn job posting, and get an AI-powered fit score, resume improvement suggestions, a tailored cover letter to send to the hiring team, and certification recommendations.

---

## Features

- **LinkedIn Profile Scraping** via MCP server (browser-based, authenticated)
- **Job Posting Scraping** via MCP tool (`get_job_details`) from LinkedIn job URLs
- **RAG Pipeline** powered by LlamaIndex + ChromaDB for profile and resume Q&A
- **Router Query Engine** — automatically routes factual questions vs. profile report generation
- **Agentic RAG** with conversation memory for follow-up questions
- **Resume Analyzer Pipeline** — multi-agent system for end-to-end resume analysis:
  - Fit scoring (0–100) with rationale
  - Includes a detailed breakdown of the strengths and weaknesses in the profile
  - Resume improvement suggestions 
  - Cover letter generation for hiring team
  - Certification recommendations for strengthening profile
- **Supervisor Agent** — orchestrates parallel recommendation agents with safe error handling
- **FastAPI REST API** with `/profile`, `/ask`, and `/resume/*` endpoints
- **Gradio UI** — browser-based dashboard for all features
- **RAG Evaluation** using LlamaIndex's `FaithfulnessEvaluator` and `RelevancyEvaluator`

---

## Architecture
```
User Request (Gradio UI or REST API)
     │
     ▼
FastAPI (api/app.py)
     │
     ├──► ProfileService (src/services/profile_service.py)
     │         │
     │         ├── LinkedIn MCP Client (mcp_client/linkedin_client.py)
     │         │         └── LinkedIn MCP Server (Patchright browser scraping)
     │         │
     │         ├── Data Processing (src/processing/data_processing.py)
     │         │         └── JSONReader → SentenceSplitter → Nodes
     │         │
     │         └── Query Engine (src/rag/query_engine.py)
     │                   ├── ChromaDB Vector Index
     │                   ├── RouterQueryEngine
     │                   │     ├── profile_qa    (factual questions)
     │                   │     └── profile_report (summaries, icebreakers, networking)
     │                   └── FunctionAgent (agentic mode with memory)
     │
     └──► ResumeService (src/services/resume_service.py)
               │
               ├── Resume Processing (src/processing/resume_processing.py)
               │         └── PDFReader → SentenceSplitter → section-tagged TextNodes
               │
               ├── Job Client (mcp_client/job_client.py)
               │         └── MCP tool: get_job_details → JobPostingOutput
               │
               ├── Resume Index (src/rag/resume_index.py)
               │         ├── ChromaDB: resume collection
               │         └── ChromaDB: job_posting collection
               │
               └── Supervisor Agent (src/agents/supervisor_agent.py)
                         ├── Retrieval Agent   → parallel resume + job context fetch
                         ├── Job Match Agent   → fit score + skill gap analysis
                         └── Recommendations Agent (parallel)
                               ├── Resume Improvements
                               ├── Cover Letter
                               └── Certification Recommendations
```

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM | OpenAI (`gpt-4o-mini` or configurable) |
| Embeddings | OpenAI (`text-embedding-3-small`) |
| RAG Framework | LlamaIndex |
| Vector Store | ChromaDB (persistent) |
| MCP Client | `llama-index-tools-mcp` |
| MCP Server | `linkedin-scraper-mcp` (local clone) |
| API | FastAPI |
| UI | Gradio |
| Settings | Pydantic Settings |
| PDF Parsing | LlamaIndex `PDFReader` |

---

## Project Structure
```
profile-rag/
├── api/
│   ├── app.py                      # FastAPI app — /health, /profile/load, /ask
│   ├── resume_router.py            # Resume API router — /resume/* endpoints
│   └── schemas.py                  # Pydantic request/response schemas
├── mcp_client/
│   ├── linkedin_client.py          # MCP client + FunctionAgent for LinkedIn scraping
│   └── job_client.py               # MCP client for LinkedIn job posting scraping
├── src/
│   ├── config/
│   │   ├── settings.py             # Pydantic settings (env vars, prompts, RAG config)
│   │   └── logger.py               # Logging setup
│   ├── processing/
│   │   ├── data_processing.py      # JSON loading, chunking pipeline (profiles)
│   │   └── resume_processing.py    # PDF loading, section-tagged chunking (resumes)
│   ├── rag/
│   │   ├── query_engine.py         # RouterQueryEngine + FunctionAgent + ChromaDB
│   │   ├── resume_index.py         # Resume + job posting ChromaDB indexing + tools
│   │   └── eval.py                 # RAG evaluation (faithfulness + relevancy)
│   ├── llm/
│   │   └── llm_interface.py        # Cached OpenAI LLM singleton (registry pattern)
│   ├── agents/
│   │   ├── supervisor_agent.py     # Orchestrates the full resume analysis pipeline
│   │   ├── orchestrator.py         # Quick fit-score-only pipeline
│   │   ├── job_match_agent.py      # Structured fit analysis (score + skill gaps)
│   │   ├── retrieval_agent.py      # Parallel resume + job context retrieval
│   │   └── recommendations_agent.py # Resume improvements, cover letter, certs
│   ├── services/
│   │   ├── profile_service.py      # Orchestrates LinkedIn fetch → index → query
│   │   └── resume_service.py       # Orchestrates resume + job loading and analysis
│   └── schemas/
│       ├── agent_outputs.py        # All structured Pydantic output models
│       └── fit_analysis.py         # Fit analysis schema
├── gradio_app.py                   # Gradio UI (Resume Analyzer + Profile Q&A tabs)
├── template.py                     # Project scaffolding script
├── tests/
│   ├── test_mcp_client.py
│   ├── test_mcp_tools.py
│   ├── test_rag.py
│   ├── test_router.py
│   └── test_process_profile.py
├── .env                            # Environment variables (not committed)
├── pyproject.toml
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- An OpenAI API key ([platform.openai.com](https://platform.openai.com))
- Note: Open source llm's can be used for the task as well but you may need to adjust settings.py file accordingly to address rate limiting. For this reason I have used OpenAI LLM.

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

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
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

### Step 2: Start the FastAPI Server
```bash
uvicorn api.app:app --reload --port 8000
```

### Step 3: Launch the Gradio UI
```bash
python gradio_app.py
```

The Gradio dashboard will be available at `http://127.0.0.1:7860` with two tabs:

- **Resume Analyzer** — upload a resume PDF, load a job URL, run full analysis
- **Profile Analyzer** — load a LinkedIn profile and chat with it

---

## API Reference

### Profile Endpoints

#### Load a Profile
```
POST /profile/load
{ "linkedin_url": "https://www.linkedin.com/in/username/" }
```

#### Ask a Question
```
POST /ask
{ "question": "What is his current role?", "use_agent": false }
```

Set `use_agent: true` for follow-up conversations with memory.

#### Health Check
```
GET /health
```

---

### Resume Endpoints

#### Upload a Resume
```
POST /resume/load          (multipart PDF, optional ?candidate_name=)
POST /resume/load-job      { "job_url": "https://www.linkedin.com/jobs/view/..." }
POST /resume/analyze       (?quick=false for full pipeline, ?quick=true for score only)
GET  /resume/cover-letter
GET  /resume/certifications
GET  /resume/status
```

**Full analysis response includes:** fit score, skill gaps, resume improvements, cover letter, and certification recommendations — all as structured Pydantic output.

---

## Analysis Pipeline (Resume Analyzer)
```
1. Retrieval Agent     — parallel fetch of resume + job context
2. Job Match Agent     — structured fit score + skill gap analysis
3. Supervisor Agent    — spawns 3 parallel agents:
   ├── Resume Improvements
   ├── Cover Letter
   └── Certification Recommendations
```

All agents use `astructured_predict()` for typed Pydantic output. Agent failures are caught safely — the pipeline always completes.

---

## Configuration

All settings in `src/config/settings.py`, overridable via `.env`:

| Setting | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `llm_model_id` | `gpt-4o-mini` | OpenAI model |
| `embedding_model_id` | `text-embedding-3-small` | OpenAI embedding model |
| `chunk_size` | `512` | RAG chunk size |
| `chunk_overlap` | `50` | RAG chunk overlap |
| `similarity_top_k` | `2` | Chunks retrieved per query |
| `mcp_server_url` | `http://127.0.0.1:8080/mcp` | LinkedIn MCP server URL |
| `temperature` | `0.0` | LLM temperature |

---

## RAG Evaluation

Runs automatically in the background after every `/profile/load`. Results logged to `logs/`.

| Metric | Description |
|---|---|
| **Faithfulness** | Answer grounded in retrieved context (no hallucination) |
| **Relevancy** | Answer addresses what was asked |

---

## Current Status

| Feature | Status |
|---|---|
| Resume Analyzer (upload, analysis, cover letter, certs) | Stable |
| Profile Analyzer — LinkedIn load | Working |
| Profile Analyzer — Chat UI | 🔧 Under debugging |
| REST API endpoints | Stable |

## Known Limitations

- **LinkedIn Rate Limiting** — wait 2–3 minutes between repeated profile fetches
- **Latency** — The response time for the llm, resume evaluation can be upto 3-4 minutes.
- **Scanned PDFs** — only text-based PDFs are supported
- **Single-user state** — in-memory `_state` supports one active session at a time
- **Windows + stdio** — use HTTP transport mode for the MCP server on Windows

---

## Roadmap

- [x] Job description analysis from LinkedIn job postings
- [x] Fit scoring (0–100) with skill gap identification
- [x] Resume improvements, cover letter, certification recommendations
- [ ] Gradio UI dashboard debugging
- [ ] Address latency response time issues
- [ ] Docker Compose setup
- [ ] Persistent session state across API restarts
- [ ] Batch / multi-candidate analysis

---

## License

MIT

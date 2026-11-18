# Hero Motocorp Document RAG

A small service for ingesting business documents and asking grounded questions over them. The current supported runtime is a FastAPI API, a Streamlit client, Gemini answer generation, local Hugging Face embeddings, and Pinecone vector search.

## Architecture

```text
Streamlit client
       │ HTTP
       ▼
FastAPI routes ── request validation, upload limits, health/readiness
       │
       ▼
RAG pipeline ── retrieval timing, grounded generation, source formatting
       ├── ingestion ── loaders, deduplication, chunking, batched embeddings
       └── provider services ── Pinecone index/client and Gemini model
```

Backend code is organized by responsibility:

```text
backend/app/
├── api/          HTTP endpoints and request/response schemas
├── core/         environment configuration
├── rag/          ingestion, retrieval, generation, orchestration
├── services/     provider clients (embeddings, Pinecone, Gemini)
└── main.py       FastAPI application factory and system endpoints
client/           Streamlit interface
backend/data/     local document storage (mounted in Docker)
```

The legacy agent and multimodal modules were removed because the API did not call them and their separate chains/providers diverged from this runtime. This version handles PDF, DOCX, TXT, Markdown, CSV, and Excel input. It does not currently provide image understanding, conversation memory, user authentication, or background ingestion jobs.

## Run locally

Use Python 3.11 or newer. Install the backend requirements and run the API:

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload --port 8001
```

Set the provider credentials in `backend/.env`, then run the client in another terminal:

```powershell
pip install streamlit requests
$env:API_URL = "http://127.0.0.1:8001"
streamlit run client/streamlit_app.py
```

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | — | Required for query generation |
| `PINECONE_API_KEY` | — | Required for ingestion and retrieval |
| `INDEX_NAME` | `ev-gpt-rag` | Pinecone index name |
| `PINECONE_NAMESPACE` | empty (Pinecone default namespace) | Optional namespace for corpus isolation |
| `PINECONE_DIMENSION` | `384` | Must match the configured embedding model |
| `PINECONE_CLOUD` / `PINECONE_REGION` | `aws` / `us-east-1` | New index location |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Generation model |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Text splitter settings |
| `TOP_K` | `5` | Default retrieved chunks |
| `DATA_DIRECTORY` | `data` | Path relative to the backend directory |
| `MAX_UPLOAD_SIZE_BYTES` | `26214400` | Upload limit (25 MiB) |
| `ALLOWED_ORIGINS` | `http://localhost:8501` | Comma-separated browser origins |
| `LOG_LEVEL` | `INFO` | Python log level |

Pinecone indexes are created with the configured dimension if absent. An existing index with a different dimension must be recreated or selected by changing `INDEX_NAME`.

## API

- `GET /health` — liveness check.
- `GET /ready` — checks required Pinecone and Gemini configuration is present.
- `POST /api/v1/ingest` — ingest supported files already in `backend/data`.
- `POST /api/v1/documents/upload` — save and ingest one uploaded file.
- `POST /api/v1/query` — retrieve context and generate an answer.

Example query:

```json
{"query":"What does the document say about the model's launch?", "top_k":5}
```

The query response contains `answer`, source metadata, and `metrics` (`retrieval_ms`, `generation_ms`, `total_ms`, and `retrieved_chunks`). Ingestion also reports embedding, upsert, and total times. These timings cover application stages and do not include client/network latency. They are a starting point for evaluation, not an accuracy score.

## Docker Compose

Create `backend/.env` from `backend/.env.example`, then:

```bash
docker compose up --build
```

The API is exposed on port 8001 and Streamlit on 8501. Uploaded data persists in the mounted `backend/data` directory.

## Evaluation readiness

The response timing fields make stage latency measurable. Accuracy and retrieval quality need a labeled question/answer and expected-source dataset; this repository does not currently contain one. Keep evaluation results separate from production responses and compare changes on the same dataset, model versions, and retrieval settings.

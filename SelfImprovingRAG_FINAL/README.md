# Self-Improving RAG System

A next-generation Retrieval Augmented Generation system that learns from user feedback.

## Features
- Hybrid search (BM25 + Vector with RRF fusion)
- Query rewriting agent (4 strategies: decompose, HyDE, multi-query, step-back)
- Retrieval evaluator agent (auto-retries on poor quality)
- Cross-encoder reranker (ms-marco-MiniLM)
- RAGAS evaluation (faithfulness + relevancy)
- Feedback loop with EMA weight updates
- LangGraph orchestration
- FastAPI backend + Streamlit dashboard

## Quick Start

### 1. Install dependencies
pip install -r requirements.txt

### 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

Set `LLM_PROVIDER=gemini` with a real `GEMINI_API_KEY`, `LLM_PROVIDER=groq` with a real `GROQ_API_KEY`, or `LLM_PROVIDER=openai` with a real `OPENAI_API_KEY`.

### 3. Start PostgreSQL
PowerShell:
```powershell
docker run -d --name ragpg `
  -e POSTGRES_USER=raguser `
  -e POSTGRES_PASSWORD=ragpassword `
  -e POSTGRES_DB=ragdb `
  -p 5432:5432 pgvector/pgvector:pg16
```

Bash/macOS/Linux:
```bash
docker run -d --name ragpg \
  -e POSTGRES_USER=raguser \
  -e POSTGRES_PASSWORD=ragpassword \
  -e POSTGRES_DB=ragdb \
  -p 5432:5432 pgvector/pgvector:pg16
```

### 4. Start API
uvicorn src.api.main:app --reload --port 8000

### 5. Start dashboard
streamlit run frontend/app.py

### 6. Ingest documents
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory_path": "data/raw"}'

### 7. Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAG?"}'

## Notebooks (for development in Colab)
| Notebook | Purpose |
|----------|---------|
| RAG_01_Setup | Environment, packages, API keys |
| RAG_02_Documents | Document loading and chunking |
| RAG_03_Search_Indexes | Embeddings and BM25 index |
| RAG_04_VectorStore | FAISS and pgvector setup |
| RAG_05_HybridSearch | BM25 + Vector + RRF fusion |
| RAG_06_Agents | Query rewriter + retrieval evaluator |
| RAG_07_Generation | Reranker + answer generator |
| RAG_08_Evaluation | RAGAS metrics + feedback loop |
| RAG_09_LangGraph | Full orchestrated pipeline |

## Stack
- Embeddings: BAAI/bge-base-en-v1.5
- Vector DB: FAISS + pgvector
- Keyword search: BM25Okapi
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- LLM: Groq Llama 3.3 70B / GPT-4o-mini
- Orchestration: LangGraph
- Evaluation: RAGAS
- API: FastAPI
- UI: Streamlit

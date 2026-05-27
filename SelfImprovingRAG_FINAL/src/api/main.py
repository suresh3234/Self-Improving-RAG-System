import os
import pickle
import json
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv

from src.utils.schemas import Document, Chunk
from src.utils.chunker import SemanticChunker
from src.search.embedder import Embedder
from src.search.bm25_index import BM25Index
from src.search.vector_store import FAISSIndex, PGVectorStore
from src.feedback.database import FeedbackDB
from src.orchestration.graph import RAGOrchestrator

load_dotenv()

app = FastAPI(title="Self-Improving RAG API")

def get_llm_config_error() -> Optional[str]:
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    key_names = {
        "groq": ["GROQ_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
        "gemini": ["GEMINI_API_KEY"],
    }
    if provider not in key_names:
        return "Unsupported LLM_PROVIDER. Use one of: groq, openai, gemini."

    for key_name in key_names[provider]:
        value = os.environ.get(key_name, "").strip()
        if not value or value.startswith("your-") or value == "sk-your-openai-key-here":
            return f"Set a real {key_name} in .env, then restart the API."

    return None

# --- Dependency Management (Singletons) ---

class RAGCore:
    def __init__(self):
        self.processed_dir = "data/processed"
        self.embedder = Embedder()
        self.bm25_index = BM25Index(self.processed_dir)
        self.faiss_index = FAISSIndex(self.processed_dir)
        self.pg_store = PGVectorStore()
        self.feedback_db = FeedbackDB()
        self.chunker = SemanticChunker()
        
        self.parent_chunks: Dict[str, Chunk] = {}
        self.child_chunks: Dict[str, Chunk] = {}
        self.orchestrator = None
        
        self.load_resources()

    def load_resources(self):
        """Load indices and chunk maps from disk."""
        if self.bm25_index.load() and self.faiss_index.load():
            # Load chunk maps
            try:
                with open(os.path.join(self.processed_dir, "parent_chunks.pkl"), "rb") as f:
                    self.parent_chunks = pickle.load(f)
                with open(os.path.join(self.processed_dir, "child_chunks.pkl"), "rb") as f:
                    self.child_chunks = pickle.load(f)
                
                print(f"[OK] Loaded {len(self.child_chunks)} chunks into memory")
                
                # Initialize orchestrator
                self.orchestrator = RAGOrchestrator(
                    bm25_index=self.bm25_index,
                    faiss_index=self.faiss_index,
                    embedder=self.embedder,
                    chunk_map=self.child_chunks,
                    feedback_db=self.feedback_db
                )
            except Exception as e:
                print(f"[WARN] Error loading chunk maps: {e}")

core = RAGCore()

# --- API Models ---

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    query: str
    answer: str
    strategy: str
    faithfulness: float
    relevancy: float
    user_rating: Optional[int] = None
    session_id: str

class IngestRequest(BaseModel):
    directory_path: str

# --- Endpoints ---

@app.get("/")
def read_root():
    return {"status": "online", "model": core.embedder.model_name}

@app.post("/query")
def run_query(req: QueryRequest):
    if not core.orchestrator:
        raise HTTPException(status_code=503, detail="RAG indices not loaded. Please ingest data first.")

    config_error = get_llm_config_error()
    if config_error:
        raise HTTPException(status_code=500, detail=config_error)
    
    try:
        result = core.orchestrator.run(req.query, req.session_id)
        return result
    except Exception as e:
        detail = str(e)
        if any(token in detail for token in ["AuthenticationError", "ClientError", "API_KEY", "api key", "401", "403"]):
            provider = os.environ.get("LLM_PROVIDER", "groq").lower()
            key_name = {
                "groq": "GROQ_API_KEY",
                "gemini": "GEMINI_API_KEY",
            }.get(provider, "OPENAI_API_KEY")
            detail = (
                f"LLM authentication failed. Set a real {key_name} in .env "
                f"or change LLM_PROVIDER to a provider you have configured, then restart the API."
            )
        raise HTTPException(status_code=500, detail=detail)

@app.post("/ingest")
def ingest_data(req: IngestRequest, background_tasks: BackgroundTasks):
    """Trigger data ingestion from a local directory."""
    directory_path = os.path.abspath(os.path.expanduser(req.directory_path.strip().strip('"').strip("'")))
    if not os.path.isdir(directory_path):
        raise HTTPException(status_code=400, detail=f"Directory not found: {directory_path}")
    
    background_tasks.add_task(perform_ingestion, directory_path)
    return {"message": "Ingestion started in background"}

@app.get("/stats")
def get_stats():
    weights = core.feedback_db.get_strategy_weights()
    history = core.feedback_db.get_history(limit=5)
    return {
        "strategy_weights": weights,
        "recent_feedback": history,
        "indexed_chunks": len(core.child_chunks)
    }

# --- Ingestion Logic ---

def perform_ingestion(directory_path: str):
    """Full ingestion pipeline: load -> chunk -> embed -> index -> save."""
    print(f"[START] Starting ingestion from: {directory_path}")
    
    documents = []
    # Simple loader for .txt and .md files
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith((".txt", ".md")):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    doc = Document(doc_id="", content=content, source=file)
                    documents.append(doc)
    
    if not documents:
        print("[WARN] No documents found to ingest")
        return

    # 1. Chunking
    all_parents, all_children = {}, {}
    for doc in documents:
        parents, children = core.chunker.chunk_document(doc)
        for p in parents: all_parents[p.chunk_id] = p
        for c in children: all_children[c.chunk_id] = c
    
    # 2. Embedding
    print(f"  Embedding {len(all_children)} chunks...")
    child_list = list(all_children.values())
    texts = [c.content for c in child_list]
    embeddings = core.embedder.embed_texts(texts)
    
    # 3. Build Indices
    core.bm25_index.build(child_list)
    core.faiss_index.build(embeddings, [c.chunk_id for c in child_list])
    
    # 4. Save metadata
    os.makedirs(core.processed_dir, exist_ok=True)
    with open(os.path.join(core.processed_dir, "parent_chunks.pkl"), "wb") as f:
        pickle.dump(all_parents, f)
    with open(os.path.join(core.processed_dir, "child_chunks.pkl"), "wb") as f:
        pickle.dump(all_children, f)
        
    # 5. DB Upsert
    core.pg_store.init_db()
    core.pg_store.upsert_chunks(child_list, embeddings)
    
    # Reload singletons
    core.load_resources()
    print("[OK] Ingestion complete")

import os
import faiss
import numpy as np
import json
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, text, Column, String, Integer, Text, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import datetime

Base = declarative_base()

class DocumentChunk(Base):
    __tablename__ = 'document_chunks'
    
    chunk_id = Column(String(50), primary_key=True)
    doc_id = Column(String(100))
    content = Column(Text)
    metadata_json = Column(JSON, name="metadata")
    parent_chunk_id = Column(String(50))
    chunk_index = Column(Integer)
    # embedding column is handled via raw SQL as pgvector isn't a standard SQLAlchemy type without extensions
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class FAISSIndex:
    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = processed_dir
        self.bin_path = os.path.join(processed_dir, "faiss_index.bin")
        self.ids_path = os.path.join(processed_dir, "child_ids.json")
        self.index = None
        self.chunk_ids = []

    def build(self, embeddings: np.ndarray, chunk_ids: List[str]):
        """Build and save FAISS index."""
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        
        # Normalize for cosine similarity (Inner Product on normalized vectors)
        emb_copy = embeddings.copy().astype(np.float32)
        faiss.normalize_L2(emb_copy)
        
        self.index.add(emb_copy)
        self.chunk_ids = chunk_ids
        
        # Save
        os.makedirs(self.processed_dir, exist_ok=True)
        faiss.write_index(self.index, self.bin_path)
        with open(self.ids_path, "w") as f:
            json.dump(self.chunk_ids, f)
        print(f"[OK] FAISS index saved to {self.bin_path}")

    def load(self):
        """Load FAISS index from disk, or auto-build from embeddings if missing."""
        if os.path.exists(self.bin_path):
            self.index = faiss.read_index(self.bin_path)
            with open(self.ids_path, "r") as f:
                self.chunk_ids = json.load(f)
            print(f"[OK] FAISS index loaded with {self.index.ntotal} vectors")
            return True
        
        # Auto-build from embeddings if available
        emb_path = os.path.join(self.processed_dir, "child_embeddings.npy")
        if os.path.exists(emb_path) and os.path.exists(self.ids_path):
            print("[INFO] faiss_index.bin not found, auto-building from child_embeddings.npy...")
            embeddings = np.load(emb_path)
            with open(self.ids_path, "r") as f:
                chunk_ids = json.load(f)
            self.build(embeddings, chunk_ids)
            return True
        
        print(f"[WARN] FAISS index not found at {self.bin_path}")
        return False

    def search(self, query_emb: np.ndarray, top_k: int = 10):
        """Search FAISS index."""
        if self.index is None:
            raise ValueError("FAISS index not loaded")
            
        D, I = self.index.search(query_emb, top_k)
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx >= 0:
                results.append((self.chunk_ids[idx], float(dist)))
        return results

class PGVectorStore:
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.environ.get("DATABASE_URL")
        self.engine = None
        self.Session = None
        if not self.db_url:
            return
        try:
            self.engine = create_engine(self.db_url)
            self.Session = sessionmaker(bind=self.engine)
        except Exception as e:
            print(f"[WARN] Could not connect to PostgreSQL: {e}")
            self.engine = None

    def init_db(self):
        """Initialize database and create tables."""
        if not self.engine: return
        
        with self.engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
            
        Base.metadata.create_all(self.engine)
        
        # Add embedding column if it doesn't exist
        dim = int(os.environ.get("EMBEDDING_DIMENSION", "768"))
        with self.engine.connect() as conn:
            try:
                conn.execute(text(f"ALTER TABLE document_chunks ADD COLUMN embedding vector({dim});"))
                conn.commit()
            except:
                pass # Already exists

    def upsert_chunks(self, chunks: List[Any], embeddings: np.ndarray):
        """Insert or update chunks in PostgreSQL."""
        if not self.engine: return
        
        with self.engine.connect() as conn:
            for i, chunk in enumerate(chunks):
                emb_str = "[" + ",".join(map(str, embeddings[i].tolist())) + "]"
                conn.execute(text("""
                    INSERT INTO document_chunks 
                    (chunk_id, doc_id, content, metadata, parent_chunk_id, chunk_index, embedding)
                    VALUES (:id, :doc_id, :content, :meta, :parent, :idx, :emb)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        content = EXCLUDED.content,
                        embedding = EXCLUDED.embedding
                """), {
                    "id": chunk.chunk_id, "doc_id": chunk.doc_id,
                    "content": chunk.content, "meta": json.dumps(chunk.metadata),
                    "parent": chunk.parent_chunk_id, "idx": chunk.chunk_index,
                    "emb": emb_str
                })
            conn.commit()

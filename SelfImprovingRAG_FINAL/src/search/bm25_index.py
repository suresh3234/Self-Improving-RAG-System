import re
import pickle
import os
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from src.utils.schemas import Chunk

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with",
    "is", "was", "are", "were", "be", "been", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "this", "that", "it", "its"
}

def tokenize(text: str) -> List[str]:
    """Basic tokenization and normalization."""
    text = re.sub(r'[\W_]+', ' ', text.lower())
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 1]

class BM25Index:
    def __init__(self, processed_dir: str = "data/processed"):
        self.processed_dir = processed_dir
        self.index_path = os.path.join(processed_dir, "bm25_index.pkl")
        self.bm25 = None
        self.chunk_ids = []

    def build(self, chunks: List[Chunk]):
        """Build BM25 index from a list of chunks."""
        print(f"Building BM25 index for {len(chunks)} chunks...")
        corpus = [tokenize(c.content) for c in chunks]
        self.bm25 = BM25Okapi(corpus)
        self.chunk_ids = [c.chunk_id for c in chunks]
        
        # Save to disk
        os.makedirs(self.processed_dir, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "bm25": self.bm25,
                "chunk_ids": self.chunk_ids
            }, f)
        print(f"BM25 index saved to {self.index_path}")

    def load(self):
        """Load BM25 index from disk."""
        if not os.path.exists(self.index_path):
            print(f"[WARN] BM25 index not found at {self.index_path}")
            return False
            
        with open(self.index_path, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.chunk_ids = data["chunk_ids"]
        print(f"[OK] BM25 index loaded with {len(self.chunk_ids)} documents")
        return True

    def get_scores(self, query: str) -> List[tuple]:
        """Get BM25 scores for a query."""
        if self.bm25 is None:
            raise ValueError("BM25 index not loaded or built")
            
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        
        # Return list of (chunk_id, score)
        return [(self.chunk_ids[i], float(scores[i])) for i in range(len(scores)) if scores[i] > 0]

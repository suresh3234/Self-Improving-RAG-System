import os
import math
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self, model_name: str = None, device: str = "cpu"):
        self.model_name = model_name or os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
        print(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=device)
        self.dimension = int(os.environ.get("EMBEDDING_DIMENSION", "768"))

    def embed_texts(self, texts: List[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """Embed a list of texts into vectors."""
        batches = []
        total_batches = math.ceil(len(texts) / batch_size)
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self.model.encode(
                batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            batches.append(embeddings)
            if show_progress:
                print(f"  Embedded batch {len(batches)}/{total_batches}", end="\r")
        
        if show_progress:
            print()
            
        return np.vstack(batches)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query with the required prefix for BGE models."""
        prefix = "Represent this sentence for searching relevant passages: "
        embedding = self.model.encode(
            [prefix + query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embedding.astype(np.float32)

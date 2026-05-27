import os
import pickle
import numpy as np
import faiss

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed")

def load_all():
    print("📂 Loading data from:", DATA_PATH)

    with open(os.path.join(DATA_PATH, "child_chunks.pkl"), "rb") as f:
        child_chunks = pickle.load(f)

    with open(os.path.join(DATA_PATH, "parent_chunks.pkl"), "rb") as f:
        parent_chunks = pickle.load(f)

    embeddings = np.load(os.path.join(DATA_PATH, "child_embeddings.npy"))

    with open(os.path.join(DATA_PATH, "bm25_index.pkl"), "rb") as f:
        bm25_data = pickle.load(f)

    faiss_index = faiss.read_index(os.path.join(DATA_PATH, "faiss_index.bin"))

    print("✅ All data loaded!")

    return {
        "child_chunks": child_chunks,
        "parent_chunks": parent_chunks,
        "embeddings": embeddings,
        "bm25": bm25_data["bm25"],
        "bm25_ids": bm25_data["chunk_ids"],
        "faiss_index": faiss_index
    }
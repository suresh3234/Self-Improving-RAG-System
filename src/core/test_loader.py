from src.retrieval.loader import load_all

data = load_all()

print("✅ Loaded successfully")
print("Child chunks:", len(data["child_chunks"]))
print("Embeddings shape:", data["embeddings"].shape)

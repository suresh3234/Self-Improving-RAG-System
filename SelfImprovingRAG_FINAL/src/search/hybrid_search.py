from typing import List, Dict, Any
from src.utils.schemas import SearchResult, Chunk
from src.search.bm25_index import BM25Index
from src.search.vector_store import FAISSIndex
from src.search.embedder import Embedder

def hybrid_search(
    query: str, 
    bm25_index: BM25Index, 
    faiss_index: FAISSIndex, 
    embedder: Embedder,
    chunk_map: Dict[str, Chunk],
    top_k: int = 10, 
    bm25_w: float = 0.3, 
    vec_w: float = 0.7
) -> List[SearchResult]:
    """Perform hybrid search using Reciprocal Rank Fusion (RRF)."""
    K = 60  # RRF smoothing constant
    
    # 1. BM25 Search
    bm25_results = bm25_index.get_scores(query)
    # Sort and rank
    bm25_ranked = sorted(bm25_results, key=lambda x: x[1], reverse=True)[:top_k * 2]
    
    # 2. Vector Search
    query_emb = embedder.embed_query(query)
    vec_results = faiss_index.search(query_emb, top_k=top_k * 2)
    
    # 3. RRF Fusion
    rrf_scores = {}
    
    # Rank positions (0-indexed)
    for rank, (cid, _) in enumerate(bm25_ranked):
        rrf_scores[cid] = rrf_scores.get(cid, 0) + bm25_w / (K + rank + 1)
        
    for rank, (cid, _) in enumerate(vec_results):
        rrf_scores[cid] = rrf_scores.get(cid, 0) + vec_w / (K + rank + 1)
        
    # Sort by RRF score
    sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    results = []
    for rank, (cid, score) in enumerate(sorted_ids):
        if cid in chunk_map:
            chunk = chunk_map[cid]
            # Find original ranks for metadata
            bm25_rank_pos = next((i for i, (id_, _) in enumerate(bm25_ranked) if id_ == cid), 999)
            vec_rank_pos = next((i for i, (id_, _) in enumerate(vec_results) if id_ == cid), 999)
            
            results.append(SearchResult(
                chunk_id=cid,
                content=chunk.content,
                score=float(score),
                rrf_score=float(score),
                bm25_rank=bm25_rank_pos,
                vector_rank=vec_rank_pos,
                metadata=chunk.metadata,
                parent_chunk_id=chunk.parent_chunk_id
            ))
            
    return results

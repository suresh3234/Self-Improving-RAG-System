import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, TypedDict

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_chunk_id: Optional[str] = None
    chunk_index: int = 0
    total_chunks: int = 0

    def _generate_id(self):
        h = hashlib.md5(f"{self.doc_id}:{self.chunk_index}:{self.content[:100]}".encode()).hexdigest()
        return f"chunk_{h[:12]}"

@dataclass
class Document:
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    source: str = ""

@dataclass
class SearchResult:
    chunk_id: str
    content: str
    score: float
    bm25_rank: int = 0
    vector_rank: int = 0
    rrf_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_chunk_id: Optional[str] = None

@dataclass
class RewrittenQuery:
    original_query: str
    sub_queries: List[str] = field(default_factory=list)
    expanded_query: str = ""
    hypothetical_answer: str = ""
    strategy_used: str = "standard"

    @property
    def all_queries(self) -> List[str]:
        queries = [self.original_query]
        if self.expanded_query:
            queries.append(self.expanded_query)
        queries.extend(self.sub_queries)
        if self.hypothetical_answer:
            queries.append(self.hypothetical_answer)
        return list(dict.fromkeys(q for q in queries if q))

@dataclass
class RetrievalEvaluation:
    overall_score: float
    individual_scores: List[float]
    is_sufficient: bool
    reasoning: str
    suggested_action: str
    filtered_results: List[SearchResult] = field(default_factory=list)

class RAGState(TypedDict):
    query: str
    session_id: str
    rewritten_query: Optional[RewrittenQuery]
    search_results: List[SearchResult]
    retrieval_evaluation: Optional[RetrievalEvaluation]
    reranked_results: List[SearchResult]
    generated_answer: Optional[Dict[str, Any]]
    evaluation_scores: Optional[Dict[str, float]]
    retry_count: int
    error: Optional[str]

import re
import hashlib
from typing import List, Tuple, Dict, Any
from src.utils.schemas import Chunk, Document

class SemanticChunker:
    """
    Hierarchical chunker that splits documents into parent and child chunks.
    """
    def __init__(self, 
                 child_chunk_size: int = 128, 
                 parent_chunk_size: int = 512,
                 child_overlap: int = 20, 
                 parent_overlap: int = 50):
        self.child_chunk_size = child_chunk_size
        self.parent_chunk_size = parent_chunk_size
        self.child_overlap = child_overlap
        self.parent_overlap = parent_overlap

    def _split_into_sentences(self, text: str) -> List[str]:
        # Basic sentence splitting using regex
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]

    def _create_chunks(self, 
                       sentences: List[str], 
                       chunk_size: int, 
                       overlap: int, 
                       doc_id: str, 
                       prefix: str) -> List[Chunk]:
        chunks = []
        current = []
        current_tokens = 0

        for sentence in sentences:
            # Rough token estimation (1 token ≈ 4 chars)
            token_est = len(sentence) // 4
            if current_tokens + token_est > chunk_size and current:
                content = " ".join(current)
                c = Chunk(
                    chunk_id="", # Will be generated in __post_init__ or manually
                    doc_id=doc_id, 
                    content=content,
                    chunk_index=len(chunks), 
                    metadata={"prefix": prefix}
                )
                c.chunk_id = self._generate_id(c)
                chunks.append(c)
                # Keep overlap sentences
                current = current[-overlap:] + [sentence] if overlap else [sentence]
                current_tokens = sum(len(s)//4 for s in current)
            else:
                current.append(sentence)
                current_tokens += token_est

        if current:
            content = " ".join(current)
            c = Chunk(
                chunk_id="", 
                doc_id=doc_id, 
                content=content,
                chunk_index=len(chunks), 
                metadata={"prefix": prefix}
            )
            c.chunk_id = self._generate_id(c)
            chunks.append(c)

        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        return chunks

    def _generate_id(self, chunk: Chunk) -> str:
        content_hash = hashlib.md5(
            f"{chunk.doc_id}:{chunk.chunk_index}:{chunk.content[:100]}".encode()
        ).hexdigest()
        return f"chunk_{content_hash[:12]}"

    def chunk_document(self, document: Document) -> Tuple[List[Chunk], List[Chunk]]:
        """Splits a document into parent and child chunks."""
        sentences = self._split_into_sentences(document.content)
        
        # 1. Create Parent Chunks
        parent_chunks = self._create_chunks(
            sentences, self.parent_chunk_size, self.parent_overlap,
            document.doc_id, "parent"
        )
        for chunk in parent_chunks:
            chunk.metadata.update(document.metadata)
            chunk.metadata["source"] = document.source

        # 2. Create Child Chunks (derived from parents)
        child_chunks = []
        for parent in parent_chunks:
            parent_sents = self._split_into_sentences(parent.content)
            children = self._create_chunks(
                parent_sents, self.child_chunk_size, self.child_overlap,
                document.doc_id, "child"
            )
            for child in children:
                child.parent_chunk_id = parent.chunk_id
                child.metadata.update(parent.metadata)
            child_chunks.extend(children)

        return parent_chunks, child_chunks

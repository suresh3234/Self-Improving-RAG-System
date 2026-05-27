import os
import uuid
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END

from src.utils.schemas import RAGState, RewrittenQuery, SearchResult, RetrievalEvaluation
from src.agents.rewriter import QueryRewriter
from src.agents.evaluator import RetrievalEvaluator
from src.search.hybrid_search import hybrid_search
from src.search.bm25_index import BM25Index
from src.search.vector_store import FAISSIndex
from src.search.embedder import Embedder
from src.evaluation.ragas_eval import evaluate_with_ragas
from src.feedback.database import FeedbackDB

class RAGOrchestrator:
    def __init__(self, 
                 bm25_index: BM25Index, 
                 faiss_index: FAISSIndex, 
                 embedder: Embedder, 
                 chunk_map: Dict[str, Any],
                 feedback_db: FeedbackDB):
        self.bm25_index = bm25_index
        self.faiss_index = faiss_index
        self.embedder = embedder
        self.chunk_map = chunk_map
        self.feedback_db = feedback_db
        
        provider = os.environ.get("LLM_PROVIDER", "groq").lower()
        self.rewriter = QueryRewriter(provider=provider)
        self.evaluator = RetrievalEvaluator(provider=provider)
        
        # Setup LLM for generation
        if provider == "groq":
            from groq import Groq
            self.llm = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            self.model = "llama-3.3-70b-versatile"
        elif provider == "gemini":
            from google import genai
            self.llm = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            self.model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        else:
            from openai import OpenAI
            self.llm = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            self.model = "gpt-4o-mini"

        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(RAGState)

        # Add Nodes
        workflow.add_node("rewrite_query", self.node_rewrite_query)
        workflow.add_node("hybrid_search", self.node_hybrid_search)
        workflow.add_node("evaluate_retrieval", self.node_evaluate_retrieval)
        workflow.add_node("generate", self.node_generate)
        workflow.add_node("evaluate_answer", self.node_evaluate_answer)

        # Set Entry Point
        workflow.set_entry_point("rewrite_query")

        # Add Edges
        workflow.add_edge("rewrite_query", "hybrid_search")
        workflow.add_edge("hybrid_search", "evaluate_retrieval")
        
        # Conditional Edge after Retrieval Evaluation
        workflow.add_conditional_edges(
            "evaluate_retrieval",
            self.route_after_retrieval,
            {
                "generate": "generate",
                "retry": "rewrite_query"
            }
        )
        
        workflow.add_edge("generate", "evaluate_answer")
        workflow.add_edge("evaluate_answer", END)

        return workflow.compile()

    # --- Nodes ---

    def node_rewrite_query(self, state: RAGState) -> Dict[str, Any]:
        print(f"  [Node: RewriteQuery] Processing...")
        # Use smart rewrite logic
        rewritten = self.rewriter.smart_rewrite(state["query"])
        print(f"  -> Strategy: {rewritten.strategy_used}")
        return {"rewritten_query": rewritten}

    def node_hybrid_search(self, state: RAGState) -> Dict[str, Any]:
        print(f"  [Node: HybridSearch] Searching...")
        rewritten = state["rewritten_query"]
        all_results = []
        seen_ids = set()
        
        # Search using all expanded queries
        queries = rewritten.all_queries[:3] # Limit to top 3 for speed
        for q in queries:
            results = hybrid_search(
                q, self.bm25_index, self.faiss_index, 
                self.embedder, self.chunk_map, top_k=10
            )
            for r in results:
                if r.chunk_id not in seen_ids:
                    seen_ids.add(r.chunk_id)
                    all_results.append(r)
        
        print(f"  -> Found {len(all_results)} unique chunks")
        return {"search_results": all_results}

    def node_evaluate_retrieval(self, state: RAGState) -> Dict[str, Any]:
        print(f"  [Node: EvalRetrieval] Evaluating...")
        evaluation = self.evaluator.evaluate(state["query"], state["search_results"])
        print(f"  -> Score: {evaluation.overall_score:.2f} | Action: {evaluation.suggested_action}")
        return {"retrieval_evaluation": evaluation}

    def node_generate(self, state: RAGState) -> Dict[str, Any]:
        print(f"  [Node: Generate] Generating answer...")
        results = state["retrieval_evaluation"].filtered_results or state["search_results"]
        context = "\n\n".join([f"Source {i+1}:\n{r.content}" for i, r in enumerate(results[:5])])
        
        prompt = f"""
Use the following context to answer the user's question. 
If the answer is not in the context, say you don't know.
Do not mention the context directly in your answer.

Context:
{context}

Question: {state['query']}
Answer:"""

        if os.environ.get("LLM_PROVIDER", "groq").lower() == "gemini":
            response = self.llm.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"temperature": 0.2},
            )
            answer_text = response.text.strip()
        else:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            answer_text = response.choices[0].message.content.strip()
        return {
            "generated_answer": {
                "answer": answer_text,
                "sources": [r.chunk_id for r in results[:5]]
            }
        }

    def node_evaluate_answer(self, state: RAGState) -> Dict[str, Any]:
        print(f"  [Node: EvalAnswer] Final metrics...")
        answer_data = state["generated_answer"]
        results = state["search_results"]
        contexts = [r.content for r in results[:5]]
        
        scores = evaluate_with_ragas(state["query"], answer_data["answer"], contexts)
        
        # Save to feedback DB
        self.feedback_db.save_feedback(
            query=state["query"],
            answer=answer_data["answer"],
            strategy=state["rewritten_query"].strategy_used,
            faithfulness=scores["faithfulness"],
            relevancy=scores["answer_relevancy"],
            session_id=state["session_id"]
        )
        
        return {"evaluation_scores": scores}

    # --- Router ---

    def route_after_retrieval(self, state: RAGState) -> str:
        eval_result = state["retrieval_evaluation"]
        retry_count = state.get("retry_count", 0)
        
        if eval_result.suggested_action == "requery" and retry_count < 2:
            state["retry_count"] = retry_count + 1
            print(f"  [RETRY] Re-querying (Attempt {state['retry_count']})")
            return "retry"
        
        return "generate"

    def run(self, query: str, session_id: str = None) -> Dict[str, Any]:
        """Run the full RAG pipeline."""
        initial_state = RAGState(
            query=query,
            session_id=session_id or str(uuid.uuid4())[:8],
            rewritten_query=None,
            search_results=[],
            retrieval_evaluation=None,
            reranked_results=[],
            generated_answer=None,
            evaluation_scores=None,
            retry_count=0,
            error=None
        )
        return self.graph.invoke(initial_state)

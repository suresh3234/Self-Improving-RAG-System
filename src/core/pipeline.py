from src.retrieval.hybrid import hybrid_search
from src.agents.rewrite import smart_rewrite
from src.generation.generator import generate_answer

def run_pipeline(query):
    rewritten = smart_rewrite(query)

    results = []
    for q in rewritten.all_queries[:3]:
        results.extend(hybrid_search(q, top_k=10))

    final_results = results[:5]

    answer = generate_answer(query, final_results)

    return answer
import os
from typing import List, Dict, Any, Optional
from datasets import Dataset

def heuristic_eval(query: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Fast heuristic evaluation fallback when RAGAS is unavailable."""
    answer_words = set(answer.lower().split())
    ctx_words = set(" ".join(contexts).lower().split())
    query_words = set(query.lower().split())

    faithfulness = min(len(answer_words & ctx_words) / max(len(answer_words), 1), 1.0)
    relevancy = min(len(query_words & answer_words) / max(len(query_words), 1) * 2, 1.0)
    return {
        "faithfulness": faithfulness, 
        "answer_relevancy": relevancy, 
        "context_precision": 0.5
    }

def evaluate_with_ragas(
    query: str, 
    answer: str, 
    contexts: List[str], 
    ground_truth: Optional[str] = None
) -> Dict[str, float]:
    """Evaluate RAG quality using RAGAS framework."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        
        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts]
        }
        metrics = [faithfulness, answer_relevancy, context_precision]

        if ground_truth:
            from ragas.metrics import context_recall
            data["ground_truth"] = [ground_truth]
            metrics.append(context_recall)

        dataset = Dataset.from_dict(data)
        results = evaluate(dataset, metrics=metrics)
        scores = results.to_pandas().iloc[0].to_dict()

        return {
            "faithfulness": float(scores.get("faithfulness", 0)),
            "answer_relevancy": float(scores.get("answer_relevancy", 0)),
            "context_precision": float(scores.get("context_precision", 0)),
            "context_recall": float(scores.get("context_recall", 0)) if ground_truth else None
        }
    except Exception as e:
        print(f"⚠️ RAGAS evaluation failed: {e}")
        return heuristic_eval(query, answer, contexts)

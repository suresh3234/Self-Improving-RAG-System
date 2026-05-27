import os
import json
from typing import List, Dict, Any
from src.utils.schemas import SearchResult, RetrievalEvaluation

class RetrievalEvaluator:
    def __init__(self, provider: str = "groq", model: str = None):
        self.provider = provider.lower()
        if self.provider == "groq":
            from groq import Groq
            self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            self.model = model or "llama-3.3-70b-versatile"
        elif self.provider == "gemini":
            from google import genai
            self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
            self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        else:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            self.model = model or "gpt-4o-mini"
        
        self.quality_threshold = float(os.environ.get("RETRIEVAL_QUALITY_THRESHOLD", "0.5"))

    def _generate_json(self, prompt: str) -> Dict[str, Any]:
        if self.provider == "gemini":
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"temperature": 0.1, "response_mime_type": "application/json"},
            )
            return json.loads(response.text)

        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        }
        if self.provider == "groq":
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        return json.loads(response.choices[0].message.content)

    def evaluate(self, query: str, results: List[SearchResult]) -> RetrievalEvaluation:
        """LLM-based retrieval quality evaluator."""
        if not results:
            return RetrievalEvaluation(
                overall_score=0.0, individual_scores=[],
                is_sufficient=False, reasoning="No results",
                suggested_action="fallback"
            )

        # Format top 5 results for the LLM
        context = ""
        for i, r in enumerate(results[:5]):
            context += f"\n[Doc {i+1}]: {r.content[:300]}...\n"

        prompt = f"""
You are a retrieval quality evaluator.

Query: "{query}"

Retrieved Documents:
{context}

Rate each document's relevance (0.0-1.0) and overall quality.
Return JSON (no markdown):
{{
  "individual_scores": [0.8, 0.5, 0.9, 0.3, 0.7],
  "overall_score": 0.64,
  "is_sufficient": true,
  "reasoning": "Brief reason",
  "suggested_action": "proceed"
}}

suggested_action: "proceed" (good), "requery" (try again), "fallback" (no good docs found)"""

        data = self._generate_json(prompt)
        overall = data.get("overall_score", 0.5)
        ind_scores = data.get("individual_scores", [0.5] * min(5, len(results)))

        # Filter and update scores
        filtered = []
        for i, (r, score) in enumerate(zip(results[:5], ind_scores)):
            if score >= 0.4:
                r.score = score
                filtered.append(r)
        
        # Add remaining results
        if len(results) > 5:
            filtered.extend(results[5:])

        return RetrievalEvaluation(
            overall_score=overall,
            individual_scores=ind_scores,
            is_sufficient=data.get("is_sufficient", overall >= self.quality_threshold),
            reasoning=data.get("reasoning", ""),
            suggested_action=data.get("suggested_action", "proceed"),
            filtered_results=filtered
        )

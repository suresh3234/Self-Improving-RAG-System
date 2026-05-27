import os
import json
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
from src.utils.schemas import RewrittenQuery

class QueryRewriter:
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

    def _generate_text(self, prompt: str, temperature: float, json_output: bool = False) -> str:
        if self.provider == "gemini":
            config = {"temperature": temperature}
            if json_output:
                config["response_mime_type"] = "application/json"
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            return response.text.strip()

        kwargs = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if json_output and self.provider == "groq":
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content.strip()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
    def rewrite_decompose(self, query: str) -> RewrittenQuery:
        """Decompose query into focused sub-queries."""
        prompt = f"""
You are a search query optimizer. Decompose this query into focused sub-queries.
Query: "{query}"

Return only valid JSON, no markdown, no explanation:
{{
  "sub_queries": ["specific sub-query 1", "specific sub-query 2", "specific sub-query 3"],
  "expanded_query": "expanded version with synonyms and related terms"
}}"""
        data = json.loads(self._generate_text(prompt, temperature=0.3, json_output=True))
        return RewrittenQuery(
            original_query=query,
            sub_queries=data.get("sub_queries", []),
            expanded_query=data.get("expanded_query", query),
            strategy_used="sub_query_decomposition"
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
    def rewrite_hyde(self, query: str) -> RewrittenQuery:
        """Generate a hypothetical answer (HyDE)."""
        prompt = f"""
Write a 100-150 word factual paragraph that perfectly answers this question.
Write as if from an authoritative textbook. No preamble, just the paragraph.

Question: {query}"""
        return RewrittenQuery(
            original_query=query,
            hypothetical_answer=self._generate_text(prompt, temperature=0.4),
            expanded_query=query,
            strategy_used="hyde"
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
    def rewrite_multi_query(self, query: str) -> RewrittenQuery:
        """Generate multiple phrasings of the query."""
        prompt = f"""
Generate 4 different phrasings of this search query, each from a different angle.
Return only valid JSON: {{"queries": ["phrasing 1", "phrasing 2", "phrasing 3", "phrasing 4"]}}

Original: "{query}" """
        data = json.loads(self._generate_text(prompt, temperature=0.7, json_output=True))
        queries = data.get("queries", [query])
        return RewrittenQuery(
            original_query=query,
            sub_queries=queries,
            expanded_query=queries[0] if queries else query,
            strategy_used="multi_query"
        )

    def smart_rewrite(self, query: str) -> RewrittenQuery:
        """Select best strategy based on query characteristics."""
        q = query.lower()
        if any(w in q for w in ["how", "why", "explain", "describe", "what causes"]):
            return self.rewrite_decompose(query)
        elif len(query.split()) < 4:
            return self.rewrite_multi_query(query)
        elif "?" in query:
            return self.rewrite_hyde(query)
        return self.rewrite_decompose(query)

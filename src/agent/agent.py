import re
from tools.calculator import calculate
from tools.weather import get_weather
from tools.web_search import web_search
from tools.summarizer import summarize
from tools.citation_formatter import format_citation

class Agent:
    def __init__(self, rag_callable):
        self.rag_callable = rag_callable

    def decide_and_answer(self, query: str) -> str:
        q_lower = query.lower()

        if "météo" in q_lower or "meteo" in q_lower:
            return get_weather("Paris")

        if re.search(r'\d+\s*[\+\-\*\/]\s*\d+', q_lower):
            return calculate(query)

        if "recherche" in q_lower or "google" in q_lower or "web" in q_lower:
            return web_search(query)

        if "résume" in q_lower or "résumer" in q_lower or "synthèse" in q_lower:
            for kw in ["résume", "résumer", "synthèse"]:
                if kw in q_lower:
                    text = query[q_lower.index(kw) + len(kw):].strip()
                    return summarize(text)

        if "citation" in q_lower or "citer" in q_lower or "référence" in q_lower or "formate" in q_lower:
            for kw in ["citation", "citer", "référence", "formate"]:
                if kw in q_lower:
                    ref = query[q_lower.index(kw) + len(kw):].strip()
                    return format_citation(ref)

        # RAG d'abord
        rag_response = self.rag_callable(query)

        # Si le RAG ne sait pas → fallback web_search juridique
        if "ne sais pas" in rag_response or "ne trouve pas" in rag_response:
            return web_search(query)

        return rag_response
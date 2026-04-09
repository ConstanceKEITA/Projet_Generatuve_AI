from typing import Dict
from src.tools.calculator import calculate
from src.tools.weather import get_weather
from src.tools.web_search import web_search

class Agent:
    def __init__(self, rag_callable):
        self.rag_callable = rag_callable

    def decide_and_answer(self, query: str) -> str:
        q_lower = query.lower()

        if "météo" in q_lower or "meteo" in q_lower:
            # ex: "quelle est la météo à Paris ?"
            return get_weather("Paris")

        if any(op in q_lower for op in ["+", "-", "*", "/"]):
            return calculate(query)

        if "recherche" in q_lower or "google" in q_lower or "web" in q_lower:
            return web_search(query)

        if "document" in q_lower or "politique" in q_lower or "procédure" in q_lower:
            return self.rag_callable(query)

        # fallback : simple LLM (mock)
        return f"Réponse conversationnelle simple (mock) à: {query}"
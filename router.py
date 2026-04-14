import re
from src.agent.agent import Agent
from src.rag.rag_chain import ask_rag
from memory import MemoryManager


class Router:
    """
    Router intelligent :
    - Météo/Calcul/Web explicite → Agent
    - Par défaut → RAG
    - Mémoire pour conserver le contexte et enrichir les réponses
    """

    def __init__(self):
        self.memory = MemoryManager(max_history=10)
        self.agent = Agent(rag_callable=self._rag_with_memory)

    def _rag_with_memory(self, query: str) -> str:
        """RAG enrichi avec le contexte de la conversation."""
        history = self.memory.get_history_as_text()

        if history:
            enriched_query = f"""Contexte de la conversation précédente :
{history}

Nouvelle question : {query}"""
        else:
            enriched_query = query

        return ask_rag(enriched_query)

    def route(self, query: str) -> str:
        q = query.lower()

        # 1. Météo
        if "météo" in q or "meteo" in q or "température" in q or "temps" in q:
            answer = self.agent.decide_and_answer(query)

        # 2. Calcul
        elif re.search(r'\d+\s*[\+\-\*\/]\s*\d+', q):
            answer = self.agent.decide_and_answer(query)

        # 3. Web explicite
        elif "recherche" in q or "google" in q or "web" in q or "actualité" in q:
            answer = self.agent.decide_and_answer(query)

        # 4. Résumé ou citation → Agent
        elif "résume" in q or "formate" in q or "citation" in q:
            answer = self.agent.decide_and_answer(query)

        # 5. Par défaut → RAG avec mémoire
        else:
            answer = self._rag_with_memory(query)

        # Sauvegarder dans la mémoire
        self.memory.add_user_message(query)
        self.memory.add_ai_message(answer)

        return answer

    def get_history(self):
        return self.memory.get_history()

    def clear_memory(self):
        self.memory.clear()
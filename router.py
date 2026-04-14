import re
from agent.agent import Agent
from rag.rag_chain import ask_rag
from memory import MemoryManager

class Router:
    """
    Router intelligent :
    - Météo/Calcul/Web explicite → Agent
    - Par défaut → RAG
    - Mémoire pour conserver le contexte
    """
    def __init__(self):
        self.memory = MemoryManager()
        self.agent = Agent(rag_callable=ask_rag)

    def route(self, query: str) -> tuple:
        q = query.lower()
        if "météo" in q or "meteo" in q:
            return self.agent.decide_and_answer(query), []
        elif re.search(r'\d+\s*[\+\-\*\/]\s*\d+', q):
            return self.agent.decide_and_answer(query), []
        elif "recherche" in q or "google" in q or "web" in q:
            return self.agent.decide_and_answer(query), []
        elif "résume" in q or "formate" in q or "citation" in q:
            return self.agent.decide_and_answer(query), []
        else:
            answer, sources = ask_rag(query)
            self.memory.add_user_message(query)
            self.memory.add_ai_message(answer)
            return answer, sources

    def get_history(self):
        return self.memory.get_history()
from agent.agent import Agent
from rag.rag_chain import ask_rag
from memory import MemoryManager


class Router:
    """
    Router conversationnel pour l'assistant Aequitas.
    Délègue le routing à l'agent Mistral et conserve l'historique conversationnel.
    """

    def __init__(self):
        self.memory = MemoryManager()
        self.agent = Agent(rag_callable=ask_rag)

    def route(self, query: str) -> tuple[str, list, str]:
        history = self.memory.get_history_as_text()
        answer, sources, tool_used = self.agent.decide_and_answer(query, history)
        self.memory.add_user_message(query)
        self.memory.add_ai_message(answer)
        return answer, sources, tool_used

    def get_history(self):
        return self.memory.get_history()
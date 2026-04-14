from langchain_core.messages import HumanMessage, AIMessage

class MemoryManager:
    """
    Gère la mémoire conversationnelle pour l'assistant.
    Stocke l'historique et permet de l'injecter dans les prompts.
    """

    def __init__(self, max_history: int = 10):
        self.messages = []
        self.max_history = max_history  # Limite pour éviter les prompts trop longs

    def add_user_message(self, message: str):
        self.messages.append(HumanMessage(content=message))
        self._trim()

    def add_ai_message(self, message: str):
        self.messages.append(AIMessage(content=message))
        self._trim()

    def get_history(self) -> list:
        """Retourne la liste des messages."""
        return self.messages

    def get_history_as_text(self) -> str:
        """Retourne l'historique formaté en texte pour l'injecter dans un prompt."""
        if not self.messages:
            return ""
        lines = []
        for msg in self.messages:
            if isinstance(msg, HumanMessage):
                lines.append(f"Utilisateur : {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"Assistant : {msg.content}")
        return "\n".join(lines)

    def clear(self):
        """Efface l'historique."""
        self.messages = []

    def _trim(self):
        """Garde seulement les derniers max_history messages."""
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

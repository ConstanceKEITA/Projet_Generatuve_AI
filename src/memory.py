from langchain.memory import ConversationBufferMemory

class MemoryManager:
    """
    Gère la mémoire conversationnelle pour l'assistant.
    Utilise un buffer simple mais extensible.
    """

    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def add_user_message(self, message: str):
        self.memory.chat_memory.add_user_message(message)

    def add_ai_message(self, message: str):
        self.memory.chat_memory.add_ai_message(message)

    def get_history(self):
        return self.memory.load_memory_variables({})["chat_history"]

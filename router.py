from src.agent.agent import Agent
from src.rag.rag_chain import build_rag_chain


def build_agent():
    """Construit l'agent avec le pipeline RAG chargé."""

    rag_chain = build_rag_chain()

    def rag_callable(query: str) -> str:
        result = rag_chain.invoke({"query": query})
        answer = result["result"]
        sources = [
            doc.metadata.get("source", "document")
            for doc in result.get("source_documents", [])
        ]
        if sources:
            answer += f"\n\nSources : {', '.join(set(sources))}"
        return answer

    return Agent(rag_callable=rag_callable)


if __name__ == "__main__":
    print("Chargement de l'assistant...")
    agent = build_agent()
    print("Assistant prêt ! (tapez 'quit' pour quitter)\n")

    while True:
        query = input("Vous : ").strip()
        if not query:
            continue
        if query.lower() in ["quit", "exit", "quitter"]:
            print("Au revoir !")
            break
        response = agent.decide_and_answer(query)
        print(f"Assistant : {response}\n")
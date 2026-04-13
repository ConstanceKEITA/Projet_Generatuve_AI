from src.agent.agent import Agent


def mock_rag(query: str) -> str:
    return "Réponse mock RAG"


def test_routing_meteo():
    agent = Agent(rag_callable=mock_rag)
    response = agent.decide_and_answer("quelle est la météo ?")
    assert "25°C" in response


def test_routing_calcul():
    agent = Agent(rag_callable=mock_rag)
    response = agent.decide_and_answer("2 + 2")
    assert response == "4"
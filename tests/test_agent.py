from agent.agent import Agent


def mock_rag(query: str) -> str:
    return "Réponse mock RAG"


def test_routing_meteo():
    agent = Agent(rag_callable=mock_rag)
    response = agent.decide_and_answer("quelle est la météo à Paris ?")
    # wttr.in retourne une vraie réponse avec température
    assert "°C" in response or "météo" in response.lower() or "Météo" in response


def test_routing_calcul():
    agent = Agent(rag_callable=mock_rag)
    response = agent.decide_and_answer("2 + 2")
    assert "4" in response


def test_routing_rag():
    agent = Agent(rag_callable=mock_rag)
    response = agent.decide_and_answer("Qu'est-ce que le génocide ?")
    assert response == "Réponse mock RAG"


def test_routing_web_search():
    agent = Agent(rag_callable=mock_rag)
    response = agent.decide_and_answer("recherche mandat d'arrêt CPI")
    assert isinstance(response, str) and len(response) > 0

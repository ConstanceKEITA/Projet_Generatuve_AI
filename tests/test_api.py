import os
import pytest
from dotenv import load_dotenv

load_dotenv()


def test_mistral_api_key_presente():
    """Vérifie que la clé API Mistral est bien définie."""
    api_key = os.getenv("MISTRAL_API_KEY")
    assert api_key is not None, "MISTRAL_API_KEY manquante dans le fichier .env"
    assert len(api_key) > 10, "MISTRAL_API_KEY semble invalide (trop courte)"


def test_mistral_api_connexion():
    """Vérifie que l'API Mistral répond correctement."""
    from langchain_mistralai import ChatMistralAI
    from langchain_core.messages import HumanMessage

    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
        api_key=os.getenv("MISTRAL_API_KEY")
    )

    response = llm.invoke([HumanMessage(content="Dis juste 'OK'")])
    assert response.content is not None
    assert len(response.content) > 0


def test_mistral_embeddings():
    """Vérifie que les embeddings Mistral fonctionnent."""
    from langchain_mistralai import MistralAIEmbeddings

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("MISTRAL_API_KEY")
    )

    result = embeddings.embed_query("test")
    assert isinstance(result, list)
    assert len(result) > 0
    assert isinstance(result[0], float)


def test_wttr_meteo_api():
    """Vérifie que l'API météo wttr.in est accessible."""
    import requests
    response = requests.get("https://wttr.in/Paris?format=j1", timeout=10)
    assert response.status_code == 200
    data = response.json()
    assert "current_condition" in data
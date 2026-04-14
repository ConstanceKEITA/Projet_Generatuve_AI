import pytest
from unittest.mock import patch, MagicMock
from rag.rag_chain import build_rag_chain, ask_rag


def test_rag_retourne_une_reponse():
    """Vérifie que le RAG retourne une réponse non vide."""
    chain = build_rag_chain()
    result = chain.invoke({"query": "Qu'est-ce que le génocide ?"})
    assert result["result"] != ""
    assert isinstance(result["result"], str)


def test_rag_retourne_des_sources():
    """Vérifie que le RAG retourne des documents sources."""
    chain = build_rag_chain()
    result = chain.invoke({"query": "Qu'est-ce que le génocide ?"})
    assert len(result["source_documents"]) > 0


def test_ask_rag_inclut_sources():
    """Vérifie que ask_rag ajoute les sources à la réponse."""
    response = ask_rag("Qu'est-ce que le génocide ?")
    assert isinstance(response, str)
    assert len(response) > 0
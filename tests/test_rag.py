from src.rag.rag_chain import build_rag_chain


def test_rag_retourne_une_reponse():
    chain = build_rag_chain()
    result = chain.invoke({"query": "Qu'est-ce que le génocide ?"})
    assert result["result"] != ""


def test_rag_retourne_des_sources():
    chain = build_rag_chain()
    result = chain.invoke({"query": "Qu'est-ce que le génocide ?"})
    assert len(result["source_documents"]) > 0
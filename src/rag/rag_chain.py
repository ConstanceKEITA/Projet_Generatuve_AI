import os
import yaml
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

PROMPT_TEMPLATE = """
Tu es Aequitas, un assistant expert en Droit Pénal International.
Réponds toujours en français de manière claire et précise.
Utilise le contexte ci-dessous pour répondre, ainsi que l'historique de la conversation si présent dans la question.
Si la réponse ne se trouve pas dans le contexte, réponds avec les connaissances générales du Droit Pénal International que tu possèdes.
Cite toujours tes sources avec [source] et l'article concerné.
Ne fais pas de remarques sur des cas connexes ou hypothétiques.

Contexte :
{context}

Question : {question}

Réponse :
"""

_chain_cache = None

def build_rag_chain(config_path="config.yaml"):
    """Charge le vectorstore Mistral et construit la chain RAG."""
    global _chain_cache
    if _chain_cache is not None:
        return _chain_cache

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("MISTRAL_API_KEY")
    )

    vectorstore_path = cfg["paths"].get("data_embeddings", "data/vectorstore")
    vectorstore = FAISS.load_local(
        vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
        api_key=os.getenv("MISTRAL_API_KEY")
    )

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    _chain_cache = chain
    return chain


def ask_rag(query: str, history: str = "", config_path="config.yaml") -> tuple:
    chain = build_rag_chain(config_path)
    enriched_query = f"{history}\n\nQuestion : {query}" if history else query
    result = chain.invoke({"query": enriched_query})

    answer = result["result"]

    sources = []
    for doc in result.get("source_documents", []):
        source = doc.metadata.get("source", "document")
        article = doc.metadata.get("article", "")
        if article:
            sources.append(f"{source} — {article}")
        else:
            sources.append(source)

    sources_uniques = list(set(sources)) if sources else []
    return answer, sources_uniques


if __name__ == "__main__":
    question = "Qu'est-ce que le crime de génocide ?"
    print(f"Question : {question}\n")
    print(ask_rag(question))
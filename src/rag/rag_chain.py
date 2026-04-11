import os
import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

PROMPT_TEMPLATE = """
Tu es un assistant expert. Réponds toujours en français.
Utilise uniquement le contexte ci-dessous pour répondre.
Si la réponse ne se trouve pas dans le contexte, dis clairement que tu ne sais pas.
Cite toujours tes sources avec [source].

Contexte:
{context}

Question: {question}

Réponse:
"""


def build_rag_chain(config_path="config.yaml"):
    """Charge le vectorstore et construit la chain RAG."""

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Charger le vectorstore depuis le disque
    embeddings = OpenAIEmbeddings(
        model=cfg["rag"]["embedding_model"],
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    vectorstore = FAISS.load_local(
        cfg["paths"]["data_embeddings"],
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Retriever : récupère les 3 chunks les plus pertinents
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # LLM
    llm = ChatOpenAI(
        model=cfg["llm"]["model_name"],
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Prompt
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE
    )

    # Chain complète
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return chain


def ask_rag(query: str, config_path="config.yaml") -> str:
    """Pose une question au RAG et retourne la réponse avec les sources."""
    chain = build_rag_chain(config_path)
    result = chain.invoke({"query": query})

    answer = result["result"]

    # Ajouter les sources si disponibles
    sources = [
        doc.metadata.get("source", "document")
        for doc in result.get("source_documents", [])
    ]
    if sources:
        sources_uniques = list(set(sources))
        answer += f"\n\nSources : {', '.join(sources_uniques)}"

    return answer


if __name__ == "__main__":
    question = "Quelles sont les procédures en cas d'incident ?"
    print(f"Question : {question}\n")
    print(ask_rag(question))

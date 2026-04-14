import os
import yaml
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

PROMPT_TEMPLATE = """
Tu es un assistant expert en Droit International Humanitaire (DIH).
Réponds toujours en français de manière claire et précise.
Utilise uniquement le contexte ci-dessous pour répondre.
Si la réponse ne se trouve pas dans le contexte, dis clairement que tu ne sais pas.
Cite toujours tes sources avec [source] et l'article concerné.

Contexte:
{context}

Question: {question}

Réponse:
"""


def build_rag_chain(config_path="config.yaml"):
    """Charge le vectorstore Mistral et construit la chain RAG."""

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Embeddings Mistral — compatible avec le vectorstore du compañero
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("MISTRAL_API_KEY")
    )

    # Charger le vectorstore depuis le disque
    vectorstore_path = cfg["paths"].get("data_embeddings", "data/vectorstore")
    vectorstore = FAISS.load_local(
        vectorstore_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Retriever : récupère les 3 chunks les plus pertinents
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    # LLM Mistral
    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
        api_key=os.getenv("MISTRAL_API_KEY")
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


def ask_rag(query: str, config_path="config.yaml") -> tuple:
    chain = build_rag_chain(config_path)
    result = chain.invoke({"query": query})

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
import os
import yaml
from dotenv import load_dotenv
from src.rag.loader import load_documents
from src.rag.preprocess import preprocess_documents
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


def build_pipeline(config_path="config.yaml"):
    # Charger la config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print("Etape 1 : Chargement des documents...")
    texts = load_documents(cfg["paths"]["data_raw"])
    if not texts:
        raise ValueError(f"Aucun document trouvé dans {cfg['paths']['data_raw']}")
    print(f"  -> {len(texts)} document(s) chargé(s)")

    print("Etape 2 : Découpage en chunks...")
    chunks = preprocess_documents(
        texts,
        chunk_size=cfg["rag"]["chunk_size"],
        overlap=cfg["rag"]["chunk_overlap"]
    )
    print(f"  -> {len(chunks)} chunk(s) créé(s)")

    print("Etape 3 : Génération des embeddings...")
    embeddings = OpenAIEmbeddings(
        model=cfg["rag"]["embedding_model"],
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    print("Etape 4 : Indexation dans FAISS...")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Sauvegarder le vectorstore sur disque
    output_dir = cfg["paths"]["data_embeddings"]
    os.makedirs(output_dir, exist_ok=True)
    vectorstore.save_local(output_dir)
    print(f"  -> Vectorstore sauvegardé dans '{output_dir}'")

    print(f"\nPipeline terminé — {len(chunks)} chunks indexés avec succès !")
    return vectorstore


if __name__ == "__main__":
    build_pipeline()

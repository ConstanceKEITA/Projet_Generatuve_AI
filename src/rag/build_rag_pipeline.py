import os
import yaml
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

BATCH_SIZE = 50  # Mistral recommande des petits batches


def load_chunks_from_jsonl(jsonl_path: str) -> list[Document]:
    """Charge les chunks depuis le fichier JSONL du compañero."""
    documents = []
    path = Path(jsonl_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {jsonl_path}\n"
            "Lance d'abord : python scripts/02_chunking.py"
        )

    with path.open(encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line)
            documents.append(Document(
                page_content=chunk["text"],
                metadata={
                    "id":          chunk["id"],
                    "source":      chunk["source"],
                    "year":        chunk["year"],
                    "type":        chunk["type"],
                    "url":         chunk["url"],
                    "lang":        chunk["lang"],
                    "tags":        ", ".join(chunk["tags"]),
                    "partie":      chunk.get("partie") or "",
                    "article":     chunk.get("article") or "",
                    "titre":       chunk.get("titre") or "",
                    "chunk_index": chunk["chunk_index"],
                    "char_count":  chunk["char_count"],
                },
            ))

    return documents


def build_pipeline(config_path="config.yaml"):
    """Orchestre l'ingestion complète : JSONL → embeddings Mistral → FAISS."""

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Vérification clé API
    if not os.getenv("MISTRAL_API_KEY"):
        raise ValueError(
            "MISTRAL_API_KEY non définie.\n"
            "Ajoute-la dans ton fichier .env : MISTRAL_API_KEY=ta-clé"
        )

    # Étape 1 : Charger les chunks depuis le JSONL
    jsonl_path = "data/raw/ihl_dataset.jsonl"
    log.info(f"Étape 1 : Chargement des chunks depuis {jsonl_path}...")
    documents = load_chunks_from_jsonl(jsonl_path)
    log.info(f"  -> {len(documents)} chunks chargés")

    # Étape 2 : Embeddings Mistral
    log.info("Étape 2 : Initialisation des embeddings Mistral...")
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=os.getenv("MISTRAL_API_KEY")
    )

    # Étape 3 : Indexation FAISS par batches
    log.info(f"Étape 3 : Indexation FAISS par batches de {BATCH_SIZE}...")
    vectorstore = None

    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i: i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
        log.info(f"  Batch {batch_num}/{total} ({len(batch)} chunks)...")

        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)

    # Étape 4 : Sauvegarder
    vectorstore_path = cfg["paths"].get("data_embeddings", "data/vectorstore")
    Path(vectorstore_path).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(vectorstore_path)
    log.info(f"  -> Vectorstore sauvegardé dans '{vectorstore_path}'")

    log.info(f"\nPipeline terminé — {len(documents)} chunks indexés avec succès !")
    return vectorstore


if __name__ == "__main__":
    build_pipeline()